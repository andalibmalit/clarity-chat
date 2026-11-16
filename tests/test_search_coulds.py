"""The coulds from Checkpoint 2 and the security review that change behaviour
(notes/plans/v0.3.2-status.md, both rulings): a name twice in a search list, refused on
every path (CP2 C2 / security C3); resume's bad name said with its record (C7); the stop's
fallback counts found parts (CP2 C3); the page's context figure skips a refused send (CP2
C4); a logged part's sha256 checked (CP2 C5); the word rule NFC-normalized (security C2);
a Windows junction refused (C5); a stale temporary folder removed (C6); `engine`'s values
checked (C8); a page upload dropped once added (C9). Made-up documents, scratch roots."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.commands import Commands
from clarity_chat.documents import (
    DocumentRefused,
    checked_attachments,
    library_add,
    library_state,
    reading,
    words_of,
)
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session, attached_tokens
from tests.fakes import FakeBackend
from tests.test_documents_board import chat, quiet, refusal
from tests.test_web import GET, ORIGIN, PORT, SECRET
from tests.test_web_shell import command

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n\n## Heron\nThe heron stands still.\n"


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def logged_search(session, documents) -> Path:
    """The chat's log with its `search` record's documents replaced."""
    assert session.conversation is not None
    log = session.conversation.log.path
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["type"] == "search":
            row["documents"] = documents
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return log


# --- a name twice; resume's bad name


def test_a_search_record_naming_a_document_twice_is_refused_by_resume(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    c.run("search", "glossary", quiet)
    log = logged_search(session, ["glossary", "glossary"])
    c.run("new", "", quiet)
    assert refusal(lambda: session.resume(log)) == (
        f"{log}: its search record names glossary twice. Name each document once."
    )


def test_document_try_naming_a_document_twice_is_refused(root, chat_tokenizer):
    session = Session(root, backend=FakeBackend([" R."], tokenizer=chat_tokenizer),
                      tokenizer_loader=lambda r, v: chat_tokenizer)  # fmt: skip
    assert refusal(lambda: session.try_documents(["glossary", "glossary"], "chat", "wren")) == (
        "Name each document once."
    )


def test_resume_s_bad_name_names_the_record(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    c.run("search", "glossary", quiet)
    log = logged_search(session, ["../\x1b[31m"])
    c.run("new", "", quiet)
    assert refusal(lambda: session.resume(log)) == (
        f"{log}: its search record names ../\\x1b[31m, which can't name a document. A "
        "document's name is 1 to 64 characters: a to z, 0 to 9, - and _, starting with a "
        "letter or a digit."
    )


# --- the stop's fallback; the context figure


def test_the_stop_s_fallback_counts_found_parts(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    conversation = session.conversation
    assert conversation is not None and conversation.head is not None
    user = conversation.turns[conversation.turns[conversation.head].parent_id or ""]
    assert user.found is not None
    tokens = user.found["tokens"]
    said = attached_tokens({"unavailable": "made-up"}, conversation, user)
    assert said == f", {tokens:,} of them for parts found in your documents, counted when found"


def test_the_context_figure_skips_a_refused_send(root, chat_tokenizer):
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    backend = FakeBackend([" R0.", " R1."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        command(client, "say", "Hello")
        sent = command(client, "prompt").json()["result"]["call"]
        (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
        assert command(client, "say", "Tell me about the wren").json()["type"] == "SearchRefused"
        picker = client.get("/view/picker", headers=GET).text
    n, ctx = sent["n_prompt_client"], sent["n_ctx"]
    assert f"Context {n / ctx:.0%} · {n:,} of {ctx:,} tokens" in picker


# --- a logged part's sha256; engine's values


def found_turn(root, chat_tokenizer) -> dict:
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    assert session.conversation is not None
    rows = [json.loads(line) for line in session.conversation.log.path.open()]
    (user,) = [r for r in rows if r["type"] == "turn" and r["role"] == "user"]
    return user


def test_a_logged_part_whose_text_isn_t_its_sha256_is_refused(root, chat_tokenizer):
    user = found_turn(root, chat_tokenizer)
    checked_attachments(user)
    user["found"]["parts"][0]["text"] += " Changed."
    with pytest.raises(TypeError, match="a found part's sha256 isn't its text's"):
        checked_attachments(user)


@pytest.mark.parametrize("key", ["name", "sqlite", "tokenizer", "weights", "word_rule",
                                 "common_words", "query"])  # fmt: skip
def test_engine_s_values_are_type_checked(root, chat_tokenizer, key):
    user = found_turn(root, chat_tokenizer)
    value = user["found"]["engine"][key]
    user["found"]["engine"][key] = 7 if isinstance(value, str) else "x"
    with pytest.raises(TypeError):
        checked_attachments(user)


# --- the word rule, NFC


def test_decomposed_accents_give_the_same_words():
    composed = "Is the naïve café open?"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert words_of(decomposed) == words_of(composed) == (["naïve", "café", "open"], ["is", "the"])


# --- the library: a junction, a stale temporary folder


def test_a_junction_in_place_of_a_document_folder_is_refused(root, monkeypatch):
    target = root / "local/documents/glossary"
    real = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda p: p == target or real(p))
    assert library_state(root, "glossary") == "unreadable"
    with pytest.raises(DocumentRefused, match="^glossary's folder in local/documents/ is a link"):
        library_add(root, b"Made-up.\n", "glossary.md", replace=True)


def test_add_removes_a_stale_temporary_folder_of_its_own_name(root):
    library = root / "local/documents"
    stale = library / ".herons-ab12cd_9"
    stale.mkdir()
    (stale / "original.txt").write_text("Made-up copy.\n", encoding="utf-8")
    other = library / ".glossary-zz99zz_0"  # another name's: kept
    other.mkdir()
    aside = library / ".herons-old-ab12cd_9"  # a replaced document's leftovers: kept
    aside.mkdir()
    library_add(root, b"## Heron\nTall.\n", "herons.md")
    assert not stale.exists() and other.is_dir() and aside.is_dir()


# --- the page's upload, dropped once added


def test_a_page_upload_is_dropped_once_added(root, chat_tokenizer):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    uploads = {"u-1": (b"## Heron\nTall.\n", "herons.md")}
    page = Commands(session, uploads=uploads)
    page.run("documents", "add u-1", quiet)
    assert uploads == {}
    assert refusal(lambda: page.run("documents", "add u-1", quiet)) == (
        "That file isn't uploaded any more. Choose it again."
    )


def test_a_refused_page_add_keeps_the_upload(root, chat_tokenizer):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    uploads = {"u-1": (b"## Wren\nAgain.\n", "glossary.md")}
    page = Commands(session, uploads=uploads)
    assert refusal(lambda: page.run("documents", "add u-1", quiet)).startswith(
        "There's already a document named glossary."
    )
    assert "u-1" in uploads  # Give this one a different name: the same upload
    assert isinstance(ConfigError("x"), Exception)
