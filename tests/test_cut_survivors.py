"""The real bugs that survive the simplicity cuts (notes/research/rag-simplicity-advice.md,
"Real bugs"): the walkthrough's M2 (T1 only), the fix review's S2, the "title counts twice"
fact and T20. M3 and S10 are browser checks, in test_browser_search.py and
test_browser_attach.py. Made-up documents, scratch roots."""

from __future__ import annotations

import sqlite3

import pytest

from clarity_chat import cli
from clarity_chat.commands import Commands
from clarity_chat.documents import SENTENCES
from clarity_chat.session import SearchRefused, Session
from tests.fakes import FakeBackend
from tests.test_search_b4 import quiet
from tests.test_web_search_c2 import get, make, root, text_of  # noqa: F401

T1 = "A change applies from your next message. Regenerate keeps the parts a message already has."


def test_the_chooser_says_a_change_applies_from_the_next_message(root, make):  # noqa: F811
    client, _ = make('search = ["glossary"]')  # M2, T1: a static line under the heading
    assert T1 in text_of(get(client, "picker", library="1"))


def started_with_an_unreadable_document(root, chat_tokenizer) -> SearchRefused:  # noqa: F811
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    (root / "local/documents/glossary/manifest.json").write_text("{garbage", encoding="utf-8")
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    try:
        Commands(session).run("profile", "chat", quiet)
    except SearchRefused as e:
        return e
    pytest.fail("the profile started")


def test_a_profile_start_names_the_profile_and_its_key(root, chat_tokenizer):  # noqa: F811
    refused = started_with_an_unreadable_document(root, chat_tokenizer)  # fix review S2
    key = "Or take it out of search under [documents] in the profile's file."
    chat, shell = refused.terminal, cli.outside_chat(refused).terminal
    assert chat.startswith("Profile chat searches glossary, but its manifest.json"), chat
    assert "`/documents add PATH glossary` adds the file again." in chat and key in chat, chat
    assert shell.startswith("Profile chat searches glossary, but its manifest.json"), shell
    assert "`clarity document add FILE --name glossary` adds the file again." in shell, shell
    assert key in shell and "/documents" not in shell, shell


def test_a_title_word_counts_more_not_twice():
    """bm25 with the title weighted 2 and the text 1, and the heading in the text too: a
    title word that's also in the text scores as three uses in the text (the same score),
    so "counts twice" is wrong and "counts more" is right."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE p USING fts5(title, text, tokenize='porter unicode61')")
    rows = [("wren", "wren x y"), ("y", "wren wren wren"), ("y", "wren x z"), ("q", "r s t")]
    db.executemany("INSERT INTO p VALUES (?, ?)", rows)
    score = dict(db.execute("SELECT rowid, bm25(p, 2, 1) FROM p WHERE p MATCH 'wren'"))
    assert score[1] == pytest.approx(score[2]) and score[1] < score[3]
    assert "twice" not in SENTENCES["details_caption"]
    assert "counts more" in SENTENCES["details_caption"]
    assert "count double" not in SENTENCES["details_search"]


def test_details_key_names_the_parts():  # T20
    assert SENTENCES["highlight_added"] == (
        "Highlighted: the framing line, documents and parts Clarity put in your messages."
    )
