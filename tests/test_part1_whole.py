"""Part 1's whole-branch review (notes/research/reviews/v032-part1-whole.md; the status
file's "part 1 whole-branch rulings"). PM1: a search refusal carries its problems as data,
and each front end words its own way out: the page the board's sentences, the terminal
and --json their commands. Made-up documents, scratch roots."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

from clarity_chat.batch import run_batch
from clarity_chat.cli import ChatLoop
from clarity_chat.documents import library_add, reading
from clarity_chat.json_chat import JsonChat
from clarity_chat.profiles import ConfigError
from clarity_chat.session import SearchRefused, Session
from tests.fakes import FakeBackend
from tests.test_documents_board import chat, quiet, refusal

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n"
PAGE_ONLY = ("Documents…", "Rebuild cuts", "with Rebuild", "Rebuild it.")


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    library_add(fetched_root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    return fetched_root


def folder(root: Path, name: str = "glossary") -> Path:
    return root / "local/documents" / name


def no_fts5(monkeypatch) -> None:
    import sqlite3

    from clarity_chat.documents import search as search_module

    class NoFts5:
        def execute(self, *args):
            raise sqlite3.OperationalError("no such module: fts5")

    search_module._indexes.clear()
    monkeypatch.setattr(search_module.sqlite3, "connect", lambda *a, **k: NoFts5())


def make(root: Path, how: str, monkeypatch) -> None:
    """Make the searched document `how`, after the chat started."""
    if how == "changed":
        (folder(root) / "text.md").write_text("Edited.\n", encoding="utf-8")
    elif how == "missing":
        shutil.rmtree(folder(root))
    elif how == "unreadable":
        (folder(root) / "manifest.json").write_text('{"schema": 1}', encoding="utf-8")
    elif how == "no_fts5":
        no_fts5(monkeypatch)


NAME = {"changed": "glossary", "missing": "glossary", "unreadable": "glossary",
        "control": "odd", "no_fts5": "glossary"}  # fmt: skip
TERMINAL = {
    "changed": "Clarity can't search glossary: a file in its folder has changed since Clarity "
    "cut it into parts. `clarity document rebuild glossary` cuts them again from its text.md.",
    "missing": "Clarity can't search glossary: its folder in local/documents/ is gone. Stop "
    "searching it with /search, or add the file again: `clarity document add FILE --name "
    "glossary`.",
    "unreadable": "Clarity can't search glossary: its manifest.json can't be read (it isn't a "
    "manifest Clarity writes). Delete the folder local/documents/glossary/, then add the file "
    "again.",
    "control": 'Part 1 of odd ("Wren") holds </s>, which this model reads as a control token, '
    "not text. Take it out of local/documents/odd/text.md; `clarity document rebuild odd` then "
    "cuts its parts again. Or stop searching odd with /search.",
    "no_fts5": "Word search needs SQLite's FTS5, which the Python running Clarity doesn't "
    "have. Use a Python whose SQLite includes it, such as the one uv installs. To send without "
    "searching, stop searching every document in this chat.",
}
KINDS = list(TERMINAL)


def started(root: Path, tokenizer, how: str) -> Session:
    path = root / "profiles" / "chat.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text + f'\n[documents]\nsearch = ["{NAME[how]}"]\n', encoding="utf-8")
    backend = FakeBackend([" R."], tokenizer=tokenizer)
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)


@pytest.mark.parametrize("how", KINDS)
def test_the_terminal_names_its_own_way_out(root, chat_tokenizer, monkeypatch, how):
    session = started(root, chat_tokenizer, how)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    make(root, how, monkeypatch)
    output.clear()
    loop.handle("Tell me about the wren")
    said = "".join(output)
    assert f"! Nothing was sent.\n{TERMINAL[how]}" in said
    assert not any(words in said for words in PAGE_ONLY), said
    assert loop.draft == "Tell me about the wren"


@pytest.mark.parametrize("how", KINDS)
def test_json_gets_the_terminal_s_words_and_the_problems(root, chat_tokenizer, monkeypatch, how):
    session = started(root, chat_tokenizer, how)
    stdout = io.StringIO()
    json_chat = JsonChat(session, io.BytesIO(), stdout)
    json_chat.execute({"command": "profile", "args": "chat"})
    make(root, how, monkeypatch)
    json_chat.execute({"command": "say", "args": "Tell me about the wren"})
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    (error,) = [e for e in events if e["event"] == "error"]
    assert error["type"] == "SearchRefused"
    assert error["message"] == f"Nothing was sent.\n{TERMINAL[how]}"
    (problem,) = error.get("problems") or [{}]
    state = "control_tokens" if how == "control" else how
    assert problem.get("state") == state
    assert problem.get("document") == (None if how == "no_fts5" else NAME[how])


def test_a_control_token_problem_holds_the_part(root, chat_tokenizer, monkeypatch):
    session = started(root, chat_tokenizer, "control")
    Commands = __import__("clarity_chat.commands", fromlist=["Commands"]).Commands
    c = Commands(session)
    c.run("profile", "chat", quiet)
    with pytest.raises(SearchRefused) as refused:
        c.run("say", "Tell me about the wren", quiet)
    assert getattr(refused.value, "problems", None) == [
        {"document": "odd", "state": "control_tokens", "ordinal": 1, "title": "Wren",
         "page": None, "tokens": ["</s>"]}
    ]  # fmt: skip


def test_the_page_and_the_log_keep_the_board_s_words(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    make(root, "missing", None)
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said == (
        "Nothing was sent.\nClarity can't search glossary: its folder in local/documents/ is "
        "gone. Stop searching it, or add the file again in Documents…."
    )


def test_a_run_s_pre_check_names_the_terminal_command(root, chat_tokenizer):
    (folder(root) / "text.md").write_text("Edited.\n", encoding="utf-8")
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    (root / "openings.txt").write_text("Tell me about the wren.\n", encoding="utf-8")
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(SearchRefused) as refused:
        run_batch(session, "run-x", root / "runs" / "run-x", root / "openings.txt", ["chat"], 1,
                  print)  # fmt: skip
    assert refused.value.terminal == (
        "Clarity can't search glossary: a file in its folder has changed since Clarity cut it "
        "into parts. `clarity document rebuild glossary` cuts them again from its text.md."
    )
    assert backend.calls == []


def test_a_profile_s_missing_document_names_the_terminal_command(root, chat_tokenizer):
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["birds"]\n')
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(ConfigError) as refused:
        session.new_conversation("chat")
    assert str(refused.value) == (
        "Profile chat searches birds, which isn't in your documents. Add it in Documents…, or "
        "take it out of search under [documents] in the profile's file."
    )
    assert getattr(refused.value, "terminal", None) == (
        "Profile chat searches birds, which isn't in your documents. Add it with `clarity "
        "document add FILE --name birds`, or take it out of search under [documents] in the "
        "profile's file."
    )
