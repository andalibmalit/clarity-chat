"""The v0.3.2 close fix batch (the status file's "finders ruled"): the three finders'
findings and the QA-driver review's could 3, at the command and terminal level. Made-up
documents, scratch roots."""

from __future__ import annotations

import json

import pytest

from clarity_chat import cli
from clarity_chat.cli import ChatLoop
from clarity_chat.commands import Commands
from clarity_chat.documents import library_add, library_documents
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE
from tests.fakes import FakeBackend
from tests.test_search_b4 import chat, quiet, root, run  # noqa: F401

CHAT_WAY = "`/documents add --replace PATH glossary` adds the file again."
CLI_WAY = "`clarity document add FILE --name glossary --replace` adds the file again."


def exit_code(argv: list[str]) -> int:
    """`clarity`'s exit code, an argparse refusal (exit 2) included."""
    try:
        return cli.main(argv)
    except SystemExit as e:
        return int(e.code or 0)


def refusal(act) -> str:
    try:
        act()
    except ConfigError as e:
        return str(e)
    return "no refusal"


def garble_manifest(project) -> None:
    (project / "local/documents/glossary/manifest.json").write_text("{garbage", encoding="utf-8")


def change_original(project) -> None:
    folder = project / "local/documents/glossary"
    original = next(folder.glob("original.*"))
    original.write_bytes(original.read_bytes() + b"Made-up change.\n")
    (folder / "text.md").write_text("## Wren\nEdited.\n", encoding="utf-8")


# --- surface S1: `clarity document try` with several names


def test_document_try_takes_several_names_then_the_text(root, capsys):  # noqa: F811
    library_add(root, b"## Owl\nA made-up owl near water.\n", "notes.md")
    argv = ["--root", str(root), "document", "try", "glossary", "notes", "--profile", "chat"]
    assert exit_code([*argv, "water"]) == 0
    out = capsys.readouterr().out
    assert "glossary:" in out and "notes:" in out


def test_document_try_with_one_word_asks_for_the_text(root, capsys):  # noqa: F811
    argv = ["--root", str(root), "document", "try", "glossary", "--profile", "chat"]
    assert exit_code(argv) == 1
    assert "Name at least one document, then the message." in capsys.readouterr().err


# --- siblings S1: a blind run says one profile searched and another didn't


def test_a_run_with_a_searching_and_a_plain_profile_says_they_got_different_parts(
    root,  # noqa: F811
    chat_tokenizer,
):
    (root / "profiles" / "plain.toml").write_text(CHAT_PROFILE, encoding="utf-8")
    _, _, output = run(root, chat_tokenizer, profiles=("chat", "plain"), seeds=1)
    assert "the profiles got different parts" in output


# --- QA-driver could 3: a profile's file outside the project is refused before it's touched


def test_a_missing_file_outside_the_project_says_outside_not_missing(root):  # noqa: F811
    from tests.test_profiles import BASE, write_profile

    missing = (root.parent / "made-up-missing.txt").as_posix()
    write_profile(root, "p", BASE + f'header = {{ file = "{missing}" }}\n')
    said = refusal(lambda: load_profile(root, "p", load_defaults(root)))
    assert "outside the project folder" in said and "not found" not in said, said


# --- surface S2 + siblings C2: each front end's way out


def test_chat_rebuild_of_a_changed_original_names_the_chat_s_command(root, chat_tokenizer):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    change_original(root)
    said = refusal(lambda: c.run("documents", "rebuild glossary", quiet))
    assert said.endswith(CHAT_WAY), said
    assert "Delete the folder" not in said


def test_chat_send_with_an_unreadable_document_names_the_chat_s_command(root, chat_tokenizer):  # noqa: F811
    from clarity_chat.session import SearchRefused

    _, c = chat(root, chat_tokenizer)
    garble_manifest(root)
    with pytest.raises(SearchRefused) as e:
        c.run("say", "Tell me about the wren", quiet)
    assert e.value.terminal.endswith(CHAT_WAY), e.value.terminal
    assert "Delete the folder" not in e.value.terminal


def test_document_check_of_an_unreadable_document_names_the_cli_s_command(root, capsys):  # noqa: F811
    garble_manifest(root)
    cli.main(["--root", str(root), "document", "check", "glossary"])
    out = capsys.readouterr().out
    assert out.strip().endswith(CLI_WAY), out
    assert "Delete the folder" not in out


def test_the_library_s_reason_holds_no_way_out(root):  # noqa: F811
    garble_manifest(root)
    (row,) = [d for d in library_documents(root) if d["name"] == "glossary"]
    assert row["why"].startswith("glossary's manifest.json can't be read (")
    assert row["why"].endswith(").")


# --- surface coulds


def test_documents_add_replace_with_no_path_says_its_usage(root, chat_tokenizer):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    assert refusal(lambda: c.run("documents", "add --replace", quiet)) == (
        "Usage: /documents [add [--replace] PATH [NAME] | rebuild NAME]"
    )


def test_a_missing_spaced_path_is_named_whole(root, chat_tokenizer, tmp_path):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    said = refusal(lambda: c.run("documents", f"add {tmp_path}/nope file.md", quiet))
    assert "nope file.md" in said, said


def test_try_in_a_text_mode_chat_says_n71(root, chat_tokenizer):  # noqa: F811
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    c = Commands(session)
    c.run("profile", "text", quiet)
    assert refusal(lambda: c.run("try", "the wren", quiet)) == (
        "Searching needs a chat-mode profile; text is in text mode."
    )


def test_a_default_name_folds_accents():
    from clarity_chat.documents.library import document_name

    assert document_name("été.md", None, set()) == ("ete", None)


def test_framed_is_false_when_nothing_was_added(root, chat_tokenizer):  # noqa: F811
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + "budget = 1\n", encoding="utf-8")
    _, c = chat(root, chat_tokenizer)
    c.run("say", "Tell me about the wren", quiet)
    turns = c.run("history", "", quiet)["turns"]
    found = [t["found"] for t in turns if t["role"] == "user"][-1]
    assert found["parts"] == [] and found.get("framed") is False
    assert c.run("try", "Tell me about the wren", quiet)["framed"] is False


def test_a_name_not_in_the_library_is_said_one_way(root, chat_tokenizer, capsys):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    assert refusal(lambda: c.run("documents", "rebuild nope", quiet)) == (
        "nope isn't in your documents."
    )
    cli.main(["--root", str(root), "document", "check", "zzz"])
    assert "zzz isn't in your documents." in capsys.readouterr().out


# --- siblings S2: the terminal says what search added, after the status line


def test_the_terminal_says_what_search_added_after_say_and_edit(root, chat_tokenizer):  # noqa: F811
    session, _ = chat(root, chat_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", read=lambda d: "", write=output.append)
    loop.handle("Tell me about the wren")
    said = "".join(output)
    assert '  searched for wren: added glossary:1 "Wren"' in said, said
    output.clear()
    loop.handle("/edit Tell me about the heron")
    assert '  searched for heron: added glossary:3 "Heron"' in "".join(output)


# --- siblings C1: /new and /profile name the search list they drop


def test_new_names_the_chat_s_search_list_it_drops(root, chat_tokenizer):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    library_add(root, b"## Owl\nA made-up owl.\n", "notes.md")
    c.run("search", "glossary notes", quiet)
    result = c.run("new", "", quiet)
    assert result.get("search_not_kept") == (
        "Not kept: this chat's search list (glossary and notes). The new chat searches its "
        "profile's: glossary."
    )


def test_new_says_nothing_when_the_list_is_the_profile_s(root, chat_tokenizer):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    assert "search_not_kept" not in c.run("new", "", quiet)


# --- siblings C3: parts.jsonl Clarity didn't write is `changed` in the search list too


def test_searching_reports_a_parts_refusal_as_changed(root, chat_tokenizer, monkeypatch):  # noqa: F811
    from clarity_chat.documents import PartsRefused

    session, c = chat(root, chat_tokenizer)

    def refuse(self, name):
        raise PartsRefused("glossary's parts.jsonl isn't one Clarity writes.")

    monkeypatch.setattr(Session, "_scanned", refuse)
    (entry,) = c.run("search", "", quiet)["searching"]
    assert entry["state"] == "changed", entry


def test_json_documents_rebuild_names_the_chat_s_command(root, chat_tokenizer):  # noqa: F811
    import io

    from clarity_chat.json_chat import JsonChat

    session, _ = chat(root, chat_tokenizer)
    change_original(root)
    out = io.StringIO()
    JsonChat(session, io.BytesIO(), out).execute(
        {"command": "documents", "args": "rebuild glossary"}
    )
    last = json.loads(out.getvalue().splitlines()[-1])
    assert last.get("message", last.get("error", "")).endswith(CHAT_WAY), last
