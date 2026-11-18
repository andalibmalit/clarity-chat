"""Tests for the mutations Checkpoints 1 and 2 saw survive (notes/research/reviews/
v032-checkpoint-1-mutation-rerun.md, v032-checkpoint-2.md): each holds a rule the code
keeps today but no test pinned. Made-up documents and logs, scratch roots."""

from __future__ import annotations

import json

import pytest

from clarity_chat.documents import (
    library_add,
    library_documents,
    library_state,
    reading,
    words_of,
)
from clarity_chat.documents.insert import FOUND_PART_TYPES, found_types
from clarity_chat.export import export_log
from clarity_chat.profiles import ConfigError
from tests.pdfs import make_pdf
from tests.test_documents_board import chat, quiet

GLOSSARY = (
    b"## Wren\nThe wren is a small brown bird.\n\n"
    b"## Dipper\nThe dipper walks under water.\n\n"
    b"## Heron\nThe heron stands still in water.\n"
)


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def users(session) -> list[dict]:
    assert session.conversation is not None
    lines = session.conversation.log.path.read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["type"] == "turn" and r["role"] == "user"]


# --- Checkpoint 1's survivors


def test_replacing_a_pdf_document_with_a_text_file_leaves_no_original_pdf(root):  # 9
    library_add(root, make_pdf(["The wren.", "The dipper."]), "guide.pdf")
    (root / "local/documents/guide/mine.txt").write_text("Mine.\n", encoding="utf-8")
    library_add(root, b"# Text\n\nNow text.\n", "guide.md")
    assert sorted(p.name for p in (root / "local/documents/guide").iterdir()) == [
        "manifest.json",
        "mine.txt",
        "original.txt",
        "parts.jsonl",
        "text.md",
    ]


def test_a_repeated_common_word_is_listed_once_in_ignored():  # 10
    assert words_of("The cat and the dog and the wren") == (
        ["cat", "dog", "wren"],
        ["the", "and"],
    )


def test_a_full_manifest_of_another_schema_is_unreadable(root):  # 11
    path = root / "local/documents/glossary/manifest.json"
    manifest = json.loads(path.read_bytes())
    path.write_text(json.dumps({**manifest, "schema": 2}), encoding="utf-8")
    assert library_state(root, "glossary") == "unreadable"
    (listed,) = library_documents(root)
    assert listed["why"].startswith(
        "glossary's manifest.json can't be read (it isn't a manifest Clarity writes)."
    )


# --- Checkpoint 2's survivors


def test_an_edit_that_keeps_the_subject_gets_its_part_again(root, chat_tokenizer):  # S2, 1
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    c.run("edit", "Tell me more about the wren, please", quiet)
    edited = users(session)[-1]
    assert edited["origin"] == "edit"
    assert [p["id"] for p in edited["found"]["parts"]] == ["glossary:1"]
    assert edited["found"]["left_out"] == []


def wrong(value):
    """A value of another JSON type than `value`."""
    if isinstance(value, str):
        return 7
    return [] if value is None else "x"


FOUND_FIELDS = [  # a words found's (v0.4.1: the types are by method)
    ("found", "method"),
    *(("found", key) for key in found_types({"method": "words"})),
    *(("found", "parts", 0, key) for key in [*FOUND_PART_TYPES, "title_words", "words"]),
    *(("found", "documents", 0, key) for key in ("name", "build_id", "original_sha256")),
    *(("found", "left_out", 0, key) for key in ("id", "rank", "score", "why")),
]


@pytest.mark.parametrize("path", FOUND_FIELDS, ids=lambda p: ".".join(map(str, p)))
def test_a_found_field_of_the_wrong_type_is_refused_by_resume_and_export(
    root, chat_tokenizer, path
):  # S12, 6 and 7
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    c.run("say", "The wren again, and the heron", quiet)  # a part left out: already in the chat
    assert session.conversation is not None
    log = session.conversation.log.path
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    changed = 0
    for row in rows:
        node = row
        try:
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = wrong(node[path[-1]])
        except (KeyError, IndexError, TypeError):
            continue
        changed += 1
        break
    assert changed, path
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    c.run("new", "", quiet)
    for read in (lambda: session.resume(log), lambda: export_log(session, log)):
        with pytest.raises(ConfigError) as refused:
            read()
        assert str(refused.value).startswith("Can't read the records of"), refused.value


def test_the_last_search_record_wins_on_resume(root, chat_tokenizer):  # S13, 10
    session, c, _ = chat(root, chat_tokenizer)
    c.run("search", "glossary", quiet)
    c.run("search", "none", quiet)
    assert session.conversation is not None
    log = session.conversation.log.path
    c.run("new", "", quiet)
    session.resume(log)
    assert c.run("search", "", quiet) == {"searching": [], "origin": "chat", "method": "words"}


def test_a_search_record_whose_documents_isn_t_a_list_is_refused(root, chat_tokenizer):  # S13, 17
    session, c, _ = chat(root, chat_tokenizer)
    c.run("search", "glossary", quiet)
    assert session.conversation is not None
    log = session.conversation.log.path
    log.write_text(
        log.read_text(encoding="utf-8").replace('["glossary"]', '"glossary"'), encoding="utf-8"
    )
    c.run("new", "", quiet)
    with pytest.raises(ConfigError) as refused:
        session.resume(log)
    assert str(refused.value).startswith("Can't read the records of"), refused.value


# --- The runner's report on part 1 (#262, #265)


def test_replace_leaves_the_new_manifest(root):  # 262
    import hashlib

    data = b"## Heron\nTall.\n\n## Egret\nWhite.\n\n## Crane\nGrey.\n"
    library_add(root, data, "glossary.md")
    manifest = json.loads((root / "local/documents/glossary/manifest.json").read_bytes())
    assert manifest["original"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["parts"] == 3
    assert library_state(root, "glossary") == "ready"
