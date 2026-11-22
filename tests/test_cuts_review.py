"""The simplicity cuts' review (notes/research/reviews/v032-cuts-review.md): S1, a profile's
start says only what can be done there (the page: the profile's key; `clarity chat
--profile`: the command line's way out); S2, a `framing` record (a development build of
v0.3.2, before cut 3) is refused on resume and export, not ignored. Made-up documents,
scratch roots."""

from __future__ import annotations

import json
import shutil

from clarity_chat import cli
from clarity_chat.commands import Commands
from clarity_chat.export import export_log
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_search_b4 import quiet
from tests.test_web import command
from tests.test_web_search_c2 import make, root  # noqa: F401

KEY = "Take it out of search under [documents] in the profile's file."


def searches(root, name: str) -> None:  # noqa: F811
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + f'\nsearch = ["{name}"]\n')


def test_the_page_s_start_says_only_the_profile_s_key(root, make):  # noqa: F811
    searches(root, "glossary")
    (root / "local/documents/glossary/manifest.json").write_text("{garbage", encoding="utf-8")
    client, _ = make()
    unreadable = command(client, "profile", "chat").json()["error"]
    assert unreadable.startswith("Profile chat searches glossary, but its manifest.json")
    assert unreadable.endswith(f"). {KEY}"), unreadable
    shutil.rmtree(root / "local/documents/glossary")
    missing = command(client, "profile", "chat").json()["error"]
    assert missing == f"Profile chat searches glossary, which isn't in your documents. {KEY}"


def test_clarity_chat_s_start_names_the_command_line_s_way(  # noqa: F811
    root,  # noqa: F811
    chat_tokenizer,
    monkeypatch,
    capsys,
):
    searches(root, "glossary")
    (root / "local/documents/glossary/manifest.json").write_text("{garbage", encoding="utf-8")
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    monkeypatch.setattr(
        cli, "Session", lambda r, **kw: Session(r, backend=backend,
                                                tokenizer_loader=lambda r, v: chat_tokenizer)
    )  # fmt: skip
    # No console (Windows CI has none for prompt_toolkit): a reader that's never reached,
    # since the start is refused before the first prompt.
    monkeypatch.setattr(cli, "prompt_reader", lambda: lambda default: "/quit")
    assert cli.main(["--root", str(root), "chat", "--profile", "chat"]) == 1
    said = capsys.readouterr().err
    assert "`clarity document add FILE --name glossary` adds the file again." in said, said
    assert "/documents" not in said and "take it out of search under [documents]" in said


def framing_log(root, chat_tokenizer, tmp_path):  # noqa: F811
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    c = Commands(session)
    c.run("profile", "chat", quiet)
    c.run("say", "Tell me about the wren", quiet)
    assert session.conversation is not None
    records = [json.loads(line) for line in session.conversation.log.path.read_text().splitlines()]
    turn = next(i for i, r in enumerate(records) if r["type"] == "turn")
    records.insert(turn, {"schema_version": 1, "time": records[turn]["time"], "type": "framing",
                          "conversation_id": records[turn]["conversation_id"],
                          "framing": "A made-up line."})  # fmt: skip
    log = tmp_path / "dev.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return session, log


def test_a_framing_record_is_refused_on_resume_and_export(root, chat_tokenizer, tmp_path):  # noqa: F811
    session, log = framing_log(root, chat_tokenizer, tmp_path)
    for act in (lambda: session.resume(log), lambda: export_log(session, log)):
        try:
            act()
            sentence = "no refusal"
        except ConfigError as e:
            sentence = str(e)
        assert sentence.startswith(f"Can't read the records of {log} ("), sentence
        assert "a framing record, from a development build of v0.3.2" in sentence, sentence
