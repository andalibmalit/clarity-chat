"""B4 (v0.3.2): try a search with no model call (`/try`, `clarity document try`), and blind
runs that search (decisions 14 and 15). Made-up documents, a scratch root."""

from __future__ import annotations

import json

import pytest

from clarity_chat import cli
from clarity_chat.batch import RunError, run_batch
from clarity_chat.commands import Commands
from clarity_chat.documents import library_add, reading
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE
from tests.fakes import FakeBackend

GLOSSARY = (
    b"## Wren\nThe wren is a small brown bird with a loud song.\n\n"
    b"## Dipper\nThe dipper walks under water in winter.\n\n"
    b"## Heron\nThe heron stands still in shallow water.\n"
)
OPENINGS = "Tell me about the wren.\n---\nWhat lives near water?\n"


def quiet(event: dict) -> None:
    pass


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    profiles = fetched_root / "profiles"
    searching = CHAT_PROFILE + '\n[documents]\nsearch = ["glossary"]\n'
    (profiles / "chat.toml").write_text(searching, encoding="utf-8")
    small = searching.replace("max_tokens = 16", "max_tokens = 16").replace(
        'search = ["glossary"]', 'search = ["glossary"]\nbudget = 60'
    )
    (profiles / "small.toml").write_text(small, encoding="utf-8")
    (fetched_root / "openings.txt").write_text(OPENINGS, encoding="utf-8")
    return fetched_root


def chat(root, tokenizer):
    backend = FakeBackend([f" R{i}." for i in range(30)], tokenizer=tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    c = Commands(session)
    c.run("profile", "chat", quiet)
    return session, c


# --- /try and `clarity document try`


def test_try_gives_what_the_message_s_turn_then_holds_and_logs_nothing(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    assert session.conversation is not None
    log = session.conversation.log.path
    before = log.read_bytes()
    tried = c.run("try", "Tell me about the wren", quiet)["found"]
    assert log.read_bytes() == before
    c.run("say", "Tell me about the wren", quiet)
    turn = next(r for r in read_records(log) if r["type"] == "turn" and r["role"] == "user")
    assert tried == turn["found"]
    assert [p["id"] for p in tried["parts"]] == ["glossary:1"]


def test_try_in_a_chat_that_searches_nothing_says_so(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    c.run("search", "none", quiet)
    with pytest.raises(ConfigError, match="^This chat searches no documents yet.$"):
        c.run("try", "Tell me about the wren", quiet)


def test_try_on_a_changed_document_is_refused_with_the_way_out(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="^Clarity can't search glossary: a file in its"):
        c.run("try", "Tell me about the wren", quiet)


def test_document_try_equals_a_new_chat_s_first_message(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    c.run("say", "What lives near water?", quiet)
    assert session.conversation is not None
    first = next(
        r["found"]
        for r in read_records(session.conversation.log.path)
        if r["type"] == "turn" and r["role"] == "user"
    )
    other = Session(root, tokenizer_loader=lambda r, v: chat_tokenizer)
    assert other.try_documents(["glossary"], "chat", "What lives near water?") == first


def test_the_terminal_prints_try_without_the_parts_text(root, chat_tokenizer):
    from clarity_chat.cli import format_found

    session, c = chat(root, chat_tokenizer)
    found = c.run("try", "Why does the wren sing?", quiet)["found"]
    text = format_found(found)
    assert text.splitlines()[0] == "Searched for: wren, sing (ignored: why, does, the)."
    assert '  1. glossary:1 "Wren": title matches wren; shares words: wren (' in text  # song
    assert "small brown bird" not in text
    nothing = format_found(c.run("try", "what is it?", quiet)["found"])
    assert nothing == (
        "Nothing to search for: your message holds only common words and single letters or "
        "digits (what, is, it)."
    )


def test_document_try_on_the_command_line(root, chat_tokenizer, monkeypatch, capsys):
    import clarity_chat.session as session_module

    real = session_module.Session.__init__

    def with_tokenizer(self, *args, **kwargs):
        kwargs.setdefault("tokenizer_loader", lambda r, v: chat_tokenizer)
        real(self, *args, **kwargs)

    monkeypatch.setattr(session_module.Session, "__init__", with_tokenizer)
    argv = ["--root", str(root), "document", "try", "glossary", "--profile", "chat"]
    assert cli.main([*argv, "Tell me about the wren"]) == 0
    assert capsys.readouterr().out.startswith("Searched for: wren (ignored: tell, me, about, the).")
    assert cli.main([*argv, "Tell me about the wren", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["parts"][0]["id"] == "glossary:1"


# --- blind runs


def run(root, tokenizer, profiles=("chat",), seeds=2, setup=None):
    backend = FakeBackend([" A reply."] * 20, tokenizer=tokenizer)
    run_dir = root / "runs" / "run-test"
    session = Session(
        root, backend=backend, logs_dir=run_dir, tokenizer_loader=lambda r, v: tokenizer
    )
    if setup:
        setup(session)
    output: list[str] = []
    run_batch(
        session, "run-test", run_dir, root / "openings.txt", list(profiles), seeds, output.append
    )
    return run_dir, backend, "".join(output)


def test_a_run_searches_once_per_opening_and_profile_and_reuses_it_for_every_seed(
    root, chat_tokenizer, monkeypatch
):
    from clarity_chat.documents import search_documents as real

    searches = []
    monkeypatch.setattr(
        "clarity_chat.session.search_documents", lambda *a: searches.append(a[1]) or real(*a)
    )
    run_dir, _, _ = run(root, chat_tokenizer, seeds=3)
    turns = [r for r in read_records(run_dir / "calls.jsonl") if r.get("role") == "user"]
    by_opening: dict[str, list] = {}
    for t in turns:
        by_opening.setdefault(t["text"], []).append(t["found"])
    assert len(by_opening) == 2
    assert all(
        len(found) == 3 and found[0] == found[1] == found[2] for found in by_opening.values()
    )
    # Each opening: once in the pre-check, once for the run's three seeds.
    assert sorted(searches) == sorted([*by_opening, *by_opening])
    record = next(r for r in read_records(run_dir / "calls.jsonl") if r["type"] == "run")
    assert set(record["documents"]) == {"glossary"}


def test_the_pre_check_refuses_a_stale_document_before_any_model_loads(root, chat_tokenizer):
    (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
    backend = FakeBackend([" A reply."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(ConfigError, match="^Clarity can't search glossary: a file in its"):
        run_batch(
            session, "run-x", root / "runs" / "run-x", root / "openings.txt", ["chat"], 1, print
        )
    assert backend.calls == []


def test_a_build_that_changes_mid_run_stops_it(root, chat_tokenizer):
    def change_after_first(session):
        real = session.block
        calls = []

        def block(text):
            calls.append(text)
            if len(calls) == 2:  # the second opening: the document was rebuilt meanwhile
                from clarity_chat.commands import document_rebuild

                path = root / "local/documents/glossary/text.md"
                path.write_text(path.read_text(encoding="utf-8") + "\n## Owl\nNew.\n")
                document_rebuild(root, "glossary")
            return real(text)

        session.block = block

    with pytest.raises(RunError, match="glossary changed during the run"):
        run(root, chat_tokenizer, seeds=1, setup=change_after_first)


def test_two_profiles_that_got_different_parts_are_said(root, chat_tokenizer):
    _, _, output = run(root, chat_tokenizer, profiles=("chat", "small"), seeds=1)
    assert "Opening 1: the profiles got different parts (chat: glossary:1; small: none)." in output
