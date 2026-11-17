"""v0.3.2 Checkpoint 1's shoulds and coulds (notes/research/reviews/v032-checkpoint-1.md, as
the main session ruled them in the status file). Scratch roots, made-up documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clarity_chat.commands import document_add
from clarity_chat.documents import (
    FRAMING,
    Attached,
    DocumentRefused,
    content,
    cut_parts,
    fill,
    library_documents,
    library_state,
    load_document,
    reading,
    search_documents,
    words_of,
)
from tests.test_documents_found import counting_room, matches
from tests.test_documents_search import loaded


@pytest.fixture
def root(tmp_path):
    reading._last = None
    project = tmp_path / "project"
    project.mkdir()
    return project


def file(tmp_path: Path, name: str, data: bytes | str) -> str:
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return str(path)


def answer(act) -> str:
    """A refusal's sentence, or what else happened: a test's own answer to assert on."""
    try:
        act()
        return "no refusal"
    except DocumentRefused as e:
        return str(e)
    except Exception as e:  # noqa: BLE001 - the bug's own exception, as the answer
        return f"{type(e).__name__}"


# --- S1: contraction fragments are common words


def test_contraction_fragments_aren_t_searched_for():
    words, ignored = words_of(
        "I don't think you're right, I'll check what I've done; it isn't, won't, doesn't"
    )
    assert words == ["think", "right", "check"]
    assert {"don", "re", "ll", "ve", "isn", "won", "doesn"} <= set(ignored)


# --- S4: the manifest's types, and no path in a sentence


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(original="x"),
        lambda m: m["original"].update(kind="exe"),
        lambda m: m["text"].update(page_starts="x"),
        lambda m: m.update(parts="many"),
    ],
)
def test_a_manifest_with_a_wrong_type_is_unreadable_not_a_crash(root, tmp_path, change):
    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    document_add(root, file(tmp_path, "other.md", "Made-up other.\n"))
    path = root / "local/documents/notes/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert answer(lambda: library_state(root, "notes")) == "no refusal"
    assert library_state(root, "notes") == "unreadable"
    listed = answer(lambda: library_documents(root))
    assert listed == "no refusal", listed
    assert [d["state"] for d in library_documents(root)] == ["unreadable", "ready"]
    said = answer(lambda: load_document(root, "notes"))
    assert said.startswith("notes's manifest.json can't be read"), said


def test_a_manifest_that_can_t_be_opened_says_why_without_a_path(root, tmp_path):
    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    path = root / "local/documents/notes/manifest.json"
    path.unlink()
    path.mkdir()
    said = answer(lambda: load_document(root, "notes"))
    assert said.startswith("notes's manifest.json can't be read"), said
    assert str(root) not in said and "/" not in said.split("(")[1].split(")")[0]


# --- S5: a write that fails is a refusal


def test_a_write_that_fails_is_a_refusal_without_a_path(root, tmp_path, monkeypatch):
    """The system refusing the write (a folder you can't write, a full disk), injected so it
    runs on every system: its path stays out of the sentence."""
    from clarity_chat.documents import library as library_module

    folder = str(root / "local/documents")

    def refused(*args, **kwargs):
        raise PermissionError(13, "Permission denied", folder)

    monkeypatch.setattr(library_module.tempfile, "mkdtemp", refused)
    said = answer(lambda: document_add(root, file(tmp_path, "notes.md", "Made-up.\n")))
    assert said == "Can't write notes in local/documents/: Permission denied.", said
    assert str(root) not in said


# --- S6: one message never adds the same text twice


def test_a_paragraph_two_documents_share_is_added_once():
    shared = "## Wren\nThe same made-up paragraph in both.\n"
    docs = [
        loaded("x", shared + "\n## Lake\nOther.\n"),
        loaded("y", shared),
    ]
    room, _ = counting_room()
    filled = fill(room, "wren", [], FRAMING, search_documents(docs, "wren"), 10**6, "default")
    assert [p["id"] for p in filled["parts"]] == ["x:1"]
    assert filled["left_out"] == [
        {
            "id": "y:1",
            "rank": 2,
            "score": filled["left_out"][0]["score"],
            "why": "already in the chat",
        }
    ]


# --- S7: Details' Attachments row and a part's `added` entry


def test_a_part_s_added_entry_adds_no_attachments_row():
    from clarity_chat.web import attachment_rows

    call = {
        "message_spans": [
            {
                "turn_id": "t1",
                "added": [
                    {"what": "framing", "chars": [0, 5], "tokens": [0, 5]},
                    {"what": "part", "index": 0, "chars": [7, 20], "tokens": [5, 9]},
                ],
            }
        ]
    }
    rows = attachment_rows(call, [{"turn_id": "t1", "text": "Why?", "attachments": []}])
    assert [r["what"] for r in rows] == ["Framing line"]


# --- S8: tests that can fail


def sized(sizes: list[int]) -> dict:
    found = matches(len(sizes))
    for m, size in zip(found["matches"], sizes, strict=True):
        m["text"] = "x" * size
    return found


def test_the_fill_stops_at_the_first_that_doesn_t_fit_rather_than_skipping_it():
    room, _ = counting_room()
    # Part 2 is too big for what's left; 3 to 8 are small and would fit after it.
    filled = fill(room, "Q", [], FRAMING, sized([50, 2000, 10, 10, 10, 10, 10, 10]), 400, "x")
    assert [p["id"] for p in filled["parts"]] == ["d:1"]
    assert [(o["id"], o["why"]) for o in filled["left_out"]] == [
        (f"d:{i}", "budget") for i in range(2, 9)
    ]


def test_a_part_that_costs_exactly_what_s_left_is_added():
    room, _ = counting_room()
    one = fill(room, "Q", [], FRAMING, sized([50]), 10**6, "x")["tokens"]
    assert [p["id"] for p in fill(room, "Q", [], FRAMING, sized([50]), one, "x")["parts"]] == [
        "d:1"
    ]
    assert fill(room, "Q", [], FRAMING, sized([50]), one - 1, "x")["parts"] == []


def test_left_out_lists_fifty_and_matched_counts_all():
    room, _ = counting_room()
    filled = fill(room, "Q", [], FRAMING, matches(70, 10), 10**6, "x")
    assert (len(filled["parts"]), len(filled["left_out"]), filled["matched"]) == (8, 50, 70)


def test_sixty_word_sentences_pack_to_two_hundred_and_forty():
    from tests.test_documents_parts import words

    text = " ".join(words(60, ".") for _ in range(10))
    assert [p["words"] for p in cut_parts(text, "notes.md")] == [240, 240, 120]


# --- C1: load reads the manifest once, so its build ID is the one it read


def test_load_s_build_id_is_the_parts_it_read(root, tmp_path, monkeypatch):
    from clarity_chat.documents import search as search_module

    document_add(root, file(tmp_path, "notes.md", "## A\nMade-up.\n"))
    real = search_module.read_manifest
    reads = []

    def counted(*args, **kwargs):
        reads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(search_module, "read_manifest", counted)
    loaded = load_document(root, "notes")
    assert len(reads) == 1
    data = (root / "local/documents/notes/parts.jsonl").read_bytes()
    import hashlib

    assert loaded.build_id == hashlib.sha256(data).hexdigest()


# --- C2: reasons only for the considered matches


def test_reasons_are_found_for_the_considered_matches_only():
    text = "\n\n".join(f"## Wren {i}\nA made-up wren, number {i}." for i in range(20))
    found = search_documents([loaded("d", text)], "wren")
    assert all(m["title_words"] == ["wren"] for m in found["matches"][:8])
    assert all(m["title_words"] == [] and m["words"] == [] for m in found["matches"][8:])


# --- C4: an unknown placement is refused, not sent as start


def test_an_unknown_placement_raises():
    with pytest.raises(ValueError, match="placement"):
        content("Q", [Attached("n.md", "x\n", None)], FRAMING, [], "middle")


# --- "For B1 and B2": a turn at placement "end" reads back


def test_a_turn_at_placement_end_is_read_back():
    from clarity_chat.documents import checked_attachments

    record = {
        "attachments": [
            {
                "name": "n.md", "kind": "text", "bytes": 2, "sha256": "0" * 64,
                "extractor": {}, "unit": "line", "total": 1, "how": "whole", "range": [1, 1],
                "marker": None, "text": "x\n", "extracted_sha256": "1" * 64, "empty_units": 0,
                "control_tokens": [], "has_closing_tag": False, "tokens": 9,
            }
        ],
        "framing": FRAMING,
        "framing_origin": "default",
        "placement": "end",
    }  # fmt: skip
    assert checked_attachments(record)["placement"] == "end"
