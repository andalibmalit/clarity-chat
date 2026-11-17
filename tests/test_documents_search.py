"""A3 (v0.3.2): word search (decision 8). The word rule, the common words, FTS5's ranking and
its reasons, and the order across systems. Made-up documents only."""

from __future__ import annotations

import time

import pytest

from clarity_chat.documents import (
    DocumentRefused,
    Stale,
    Unreadable,
    cut_parts,
    library_add,
    load_document,
    reading,
    search_documents,
    words_of,
)


def loaded(name: str, text: str, filename: str = "made-up.md"):
    """A document as search reads it, without the library: its parts, and the build ID the
    library would give them (the index is kept by build ID, so a made-up one shared by two
    tests' texts would hand one the other's)."""
    from clarity_chat.documents import Loaded, build_id, parts_jsonl

    parts = cut_parts(text, filename)
    return Loaded(name, build_id(parts_jsonl(parts)), f"sha-{name}", parts)


def ids(found: dict) -> list[str]:
    return [m["id"] for m in found["matches"]]


# --- the word rule


@pytest.mark.parametrize(
    ("message", "words", "ignored"),
    [
        ("Why does the winter wren sing so long?", ["winter", "wren", "sing", "long"],
         ["why", "does", "the", "so"]),
        ("wren Wren WREN", ["wren"], []),
        ("what's up", [], ["what", "s", "up"]),
        ("foo_bar baz", ["foo", "bar", "baz"], []),
        ("🐦 wren", ["wren"], []),
        ("what is it", [], ["what", "is", "it"]),
        ("A 42 x", ["42"], ["a", "x"]),
        ("", [], []),
    ],
)  # fmt: skip
def test_the_word_rule(message, words, ignored):
    assert words_of(message) == (words, ignored)


# --- the two empty states, and a query that never raises


DOCS = [
    loaded(
        "birds",
        "## Winter wren\nA small brown bird with a long song.\n\n"
        "## Dipper\nIt walks under water and sings in winter.\n\n"
        "Notes on singing birds.\n",
    )
]


def test_nothing_to_search_for_and_nothing_matched_are_two_states():
    nothing = search_documents(DOCS, "what is it?")
    assert (nothing["words"], nothing["matches"]) == ([], [])
    missed = search_documents(DOCS, "volcano basalt")
    assert (missed["words"], missed["matches"]) == (["volcano", "basalt"], [])


@pytest.mark.parametrize(
    "message",
    ['he said "wren*" NEAR (dipper', '"', "*", "NEAR", "(", ")", "-wren", "wren:", "^wren", "''"],
)
def test_punctuation_and_operators_never_raise(message):
    search_documents(DOCS, message)


def test_the_engine_and_query_are_recorded():
    found = search_documents(DOCS, 'Wren "x" dipper')
    assert found["engine"]["query"] == '"wren" OR "dipper"'
    assert found["engine"]["tokenizer"] == "porter unicode61"
    assert found["engine"]["weights"] == {"title": 2, "text": 1}
    assert (found["engine"]["word_rule"], found["engine"]["common_words"]) == (1, 1)
    assert found["documents"] == [
        {"name": "birds", "build_id": DOCS[0].build_id, "original_sha256": "sha-birds"}
    ]
    assert (found["method"], found["scan_depth"], found["searched_for"]) == (
        "words",
        1,
        'Wren "x" dipper',
    )


# --- the order and the reasons


def test_a_title_match_outranks_a_text_match_of_the_same_word():
    doc = loaded("d", "## Wren\nSmall.\n\n## Other\nA wren was here.\n\n" + "## Filler\nx y\n" * 5)
    assert ids(search_documents([doc], "wren")) == ["d:1", "d:2"]


def test_equal_scores_order_by_document_then_ordinal():
    text = "## Wren\nSame text.\n\n## Wren\nSame text.\n\n## Lake\nOther.\n"
    found = search_documents([loaded("b", text), loaded("a", text)], "wren")
    assert ids(found) == ["a:1", "a:2", "b:1", "b:2"]
    assert len({m["score"] for m in found["matches"]}) == 1


def test_the_reasons_are_what_fts5_matched_stemmed_forms_included():
    found = search_documents(DOCS, "winter singing")
    by_id = {m["id"]: m for m in found["matches"]}
    assert by_id["birds:1"]["title_words"] == ["winter"]
    assert by_id["birds:1"]["words"] == ["winter"]  # the heading line is in the text too
    assert by_id["birds:2"]["words"] == ["winter", "singing"]  # "sings" matches "singing"
    assert by_id["birds:2"]["title_words"] == []
    assert [m["rank"] for m in found["matches"]] == list(range(1, len(found["matches"]) + 1))


def test_the_same_message_and_build_give_the_same_result():
    first = search_documents(DOCS, "winter wren song")
    assert search_documents(DOCS, "winter wren song") == first
    from clarity_chat.documents import search as search_module

    search_module._indexes.clear()  # built again from nothing
    assert search_documents(DOCS, "winter wren song") == first


def test_fts5_missing_is_a_refusal(monkeypatch):
    import sqlite3

    from clarity_chat.documents import search as search_module

    class NoFts5:
        def execute(self, *args):
            raise sqlite3.OperationalError("no such module: fts5")

    search_module._indexes.clear()
    monkeypatch.setattr(search_module.sqlite3, "connect", lambda *a, **k: NoFts5())
    with pytest.raises(DocumentRefused, match="^Word search needs SQLite's FTS5"):
        search_documents(DOCS, "wren")


# --- loading from the library: stale is a refusal


def test_parts_not_built_or_a_missing_document_are_refused_before_a_search(tmp_path):
    import shutil

    reading._last = None
    root = tmp_path / "project"
    root.mkdir()
    library_add(root, b"## Wren\nA small bird.\n", "birds.md")
    assert load_document(root, "birds").parts[0]["title"] == "Wren"
    (root / "local/documents/birds/parts.jsonl").write_text("Edited.\n", encoding="utf-8")
    with pytest.raises(Unreadable) as unbuilt:
        load_document(root, "birds")
    assert unbuilt.value.said == "birds's parts.jsonl isn't the one Clarity built."
    shutil.rmtree(root / "local/documents/birds")
    with pytest.raises(Stale) as refused:
        load_document(root, "birds")
    assert refused.value.state == "missing"


# --- the golden order (S8): the same on macOS, Linux and Windows


GOLDEN = "\n\n".join(
    [
        "## Wren\nThe winter wren sings a long song in the wet woods.",
        "## Dipper\nThe dipper walks under water in winter streams.",
        "## Heron\nA tall grey bird that stands still in shallow water.",
        "## Kingfisher\nA bright bird that dives for fish from a branch over water.",
        "## Song\nMany small birds sing at dawn; the wren sings loudest for its size.",
        "## Winter\nIn winter some birds leave and others stay in the woods.",
        "Notes: the wren and the dipper share the same streams in winter.",
        *(f"## Stone {i}\nA made-up note on clay, rock and sand, number {i}." for i in range(20)),
    ]
)
GOLDEN_ORDER = [
    ("golden:1", -7.7213),
    ("golden:5", -5.0382),
    ("golden:6", -4.5728),
    ("golden:2", -4.4062),
    ("golden:3", -2.072),
    ("golden:4", -1.9557),
]


def test_the_golden_order_is_the_same_on_every_system():
    found = search_documents(
        [loaded("golden", GOLDEN)], "Why does the winter wren sing near water?"
    )
    assert [(m["id"], round(m["score"], 4)) for m in found["matches"]] == GOLDEN_ORDER


# --- size


def test_twenty_thousand_parts_are_searched_within_a_loose_bound():
    parts = "\n\n".join(f"## Headword {i}\nA made-up meaning, number {i}." for i in range(20_000))
    doc = loaded("big", parts)
    assert len(doc.parts) == 20_000
    started = time.monotonic()
    found = search_documents([doc], "headword 19999 meaning")
    assert time.monotonic() - started < 30  # 0.5 s on the M2 Max; SYS measured 0.46 s
    assert found["matches"][0]["id"] == "big:20000"


def test_reference_lists_the_common_words_exactly():
    from pathlib import Path

    from clarity_chat.documents import COMMON_WORDS

    text = (Path(__file__).parent.parent / "docs" / "reference.md").read_text(encoding="utf-8")
    line = next(t for t in text.splitlines() if t.startswith("**Common words, version 1**"))
    listed = line.split(": ", 1)[1].rstrip(".").split(", ")
    assert listed == sorted(COMMON_WORDS)
