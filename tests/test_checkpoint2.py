"""v0.3.2 Checkpoint 2's shoulds with code (notes/plans/v0.3.2-status.md, "Checkpoint 2
rulings"): S5 (a text-mode log can't make a chat send documents), S6 (the terminal's /search
says how many parts hold a control token or </part>), S7 (the terminal's /history shows what
search added) and S8 (every error type a command can raise is in reference.md's table).
Made-up documents and logs, scratch roots."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

import clarity_chat
from clarity_chat.backends import BackendError
from clarity_chat.cli import ChatLoop, format_history
from clarity_chat.commands import Commands
from clarity_chat.documents import library_add
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_documents_board import chat, quiet, refusal

REFERENCE = Path(__file__).parent.parent / "docs" / "reference.md"


# --- S5: a text-mode log can't make the chat send documents


def text_chat(root, tokenizer) -> tuple[Session, Path]:
    backend = FakeBackend([" R."], tokenizer=tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    c = Commands(session)
    c.run("profile", "text", quiet)
    c.run("say", "Made-up.", quiet)
    assert session.conversation is not None
    return session, session.conversation.log.path


def resumed(root, tokenizer, path: Path) -> str:
    backend = FakeBackend([" R."], tokenizer=tokenizer)
    fresh = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    return refusal(lambda: fresh.resume(path))


def test_a_search_record_in_a_text_mode_log_is_refused(root, tiny_tokenizer):
    session, path = text_chat(root, tiny_tokenizer)
    assert session.conversation is not None
    record = {
        "type": "search",
        "conversation_id": session.conversation.conversation_id,
        "documents": ["glossary"],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    said = resumed(root, tiny_tokenizer, path)
    assert said.startswith(f"Can't read the records of {path}"), said
    assert "a text-mode chat's search record isn't one Clarity writes" in said


@pytest.mark.parametrize("what", ["found", "attachments"])
def test_a_text_mode_turn_holding_documents_is_refused(
    root, tiny_tokenizer, chat_tokenizer, tmp_path, what
):
    if what == "found":
        searching, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
        c.run("say", "Tell me about the wren", quiet)
    else:
        searching, c, _ = chat(root, chat_tokenizer)
        notes = tmp_path / "notes.md"
        notes.write_text("Made-up notes.\n", encoding="utf-8")
        c.run("attach", str(notes), quiet)
        c.run("say", "Read this.", quiet)
    assert searching.conversation is not None
    records = [json.loads(line) for line in searching.conversation.log.path.open()]
    (sent,) = [r for r in records if r["type"] == "turn" and r["role"] == "user"]
    fields = {k: sent[k] for k in ("attachments", "framing", "framing_origin", "placement")}
    if what == "found":
        fields["found"] = sent["found"]
    session, path = text_chat(root, tiny_tokenizer)
    lines = [json.loads(line) for line in path.open()]
    for record in lines:
        if record["type"] == "turn" and record["role"] == "user":
            record.update(fields)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    said = resumed(root, tiny_tokenizer, path)
    assert "a text-mode chat's turn with documents isn't one Clarity writes" in said, said


@pytest.fixture
def root(fetched_root):
    library_add(fetched_root, b"## Wren\nThe wren is a small brown bird.\n", "glossary.md")
    return fetched_root


# --- S6: /search's counts in the terminal


def test_the_terminal_s_search_line_counts_control_tokens_and_closing_tags(root, chat_tokenizer):
    library_add(root, b"## Wren\nIt sings </s>.\n\n## Lark\nIt sings </s> too.\n", "odd.md")
    library_add(root, b"## Heron\nIt stands </part> still.\n", "tags.md")
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    output.clear()
    loop.handle("/search odd tags glossary")
    assert "".join(output).splitlines() == [
        "Searching (set in this chat): odd, tags, glossary.",
        "  odd: 2 parts hold a control token for this model. A message that finds one isn't sent.",
        "  tags: 1 part holds </part> or </document>; it's sent as it is.",
    ]


# --- S7: /history shows what search added


def history_of(found: dict) -> list[str]:
    turn = {"role": "user", "text": "Tell me", "marks": [], "notes": [], "found": found}
    return format_history({"turns": [turn]}).splitlines()[1:]


def test_history_shows_the_words_and_the_parts_added():
    found = {
        "words": ["wren", "sing"],
        "ignored": ["why"],
        "documents": ["glossary"],
        "matched": 3,
        "tokens": 120,
        "parts": [
            {"id": "glossary:1", "title": "Wren", "rank": 1},
            {"id": "glossary:3", "title": None, "rank": 2},
        ],
        "left_out": 1,
    }
    assert history_of(found) == [
        '  searched for wren, sing: added glossary:1 "Wren", glossary:3 (120 tokens)'
    ]


@pytest.mark.parametrize(
    ("words", "matched", "line"),
    [
        ([], 0, "  searched: nothing to search for, only common words and single letters or "
         "digits"),
        (["wren"], 0, "  searched for wren: nothing matched"),
        (["wren"], 2, "  searched for wren: 2 matched, none added (none fit the budget)"),
    ],
)  # fmt: skip
def test_history_says_when_search_added_nothing(words, matched, line):
    found = {
        "words": words,
        "ignored": ["it"],
        "documents": ["glossary"],
        "matched": matched,
        "tokens": 0,
        "parts": [],
        "left_out": matched,
    }
    assert history_of(found) == [line]


def test_history_prints_a_title_through_shown():
    found = {
        "words": ["wren"],
        "ignored": [],
        "documents": ["odd"],
        "matched": 1,
        "tokens": 9,
        "parts": [{"id": "odd:1", "title": "\x1b[31mred", "rank": 1}],
        "left_out": 0,
    }
    assert history_of(found) == ['  searched for wren: added odd:1 "\\x1b[31mred" (9 tokens)']


# --- S8: the error types


def raised_types() -> set[str]:
    """Every exception class Clarity defines that a command can raise to a front end: the
    kinds of ConfigError and BackendError."""
    import pkgutil

    for module in pkgutil.walk_packages(clarity_chat.__path__, "clarity_chat."):
        if "transformers" not in module.name:
            __import__(module.name)
    found: set[str] = set()
    todo: list[type[Exception]] = [ConfigError, BackendError]
    while todo:
        kind = todo.pop()
        module = inspect.getmodule(kind)
        if module is not None and module.__name__.startswith("clarity_chat"):
            found.add(kind.__name__)
        todo += kind.__subclasses__()
    return found


def test_every_error_type_is_in_the_reference_s_table():
    row = next(
        line for line in REFERENCE.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `error` |")
    )  # fmt: skip
    listed = set(re.findall(r"`(\w+)`", row))
    assert raised_types() - listed == set()
    assert {"SearchRefused", "DocumentRefused"} <= listed
