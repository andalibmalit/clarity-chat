"""A1 (v0.3.2): the library, `local/documents/NAME/` (decisions 5, 6 and 20), and the
`clarity document add|list` subcommands. Each test works in its own scratch
root, with made-up documents only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from clarity_chat import cli
from clarity_chat.commands import document_add
from clarity_chat.documents import (
    DocumentRefused,
    checked_library_name,
    default_name,
    library_documents,
    library_state,
    load_document,
    reading,
)
from tests.pdfs import aes_pdf, make_pdf

GLOSSARY = (
    "# Made-up glossary\n\n## Wren\nA small brown bird.\n\n## Dipper\nIt walks under water.\n"
)


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


def library(root: Path) -> list[str]:
    """Everything under local/documents/, as relative paths: what a refusal must not change."""
    base = root / "local" / "documents"
    return sorted(p.relative_to(root).as_posix() for p in base.rglob("*")) if base.exists() else []


# --- add: the four files and their shapes


def test_add_writes_the_four_files_and_says_so(root, tmp_path):
    result = document_add(root, file(tmp_path, "Glossary.md", GLOSSARY))
    folder = "local/documents/glossary"
    assert result["written"] == [
        f"{folder}/original.txt",
        f"{folder}/text.md",
        f"{folder}/parts.jsonl",
        f"{folder}/manifest.json",
    ]
    assert library(root) == sorted([folder, *result["written"]])
    path = root / folder
    assert (path / "original.txt").read_bytes() == GLOSSARY.encode()
    assert (path / "text.md").read_text(encoding="utf-8") == GLOSSARY
    parts = [json.loads(line) for line in (path / "parts.jsonl").read_text().splitlines()]
    assert [p["title"] for p in parts] == ["Made-up glossary", "Wren", "Dipper"]
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest == result["added"]
    assert list(manifest) == [
        "schema", "name", "added", "original", "extractor", "text", "chunker", "build_id",
        "parts", "titled",
    ]  # fmt: skip
    assert manifest["original"] == {
        "name": "Glossary.md",
        "kind": "text",
        "bytes": len(GLOSSARY.encode()),
        "sha256": hashlib.sha256(GLOSSARY.encode()).hexdigest(),
    }
    assert manifest["text"] == {
        "sha256": hashlib.sha256(GLOSSARY.encode()).hexdigest(),
        "bytes": len(GLOSSARY.encode()),
        "page_starts": None,
    }
    assert manifest["build_id"] == hashlib.sha256((path / "parts.jsonl").read_bytes()).hexdigest()
    assert (manifest["parts"], manifest["titled"]) == (3, 3)
    assert manifest["chunker"] == {
        "name": "clarity-parts",
        "version": 1,
        "max_words": 250,
        "title_chars": 120,
    }


@pytest.mark.parametrize(
    ("name", "data", "original"),
    [
        ("guide.pdf", make_pdf(["Made-up page one.", "Made-up page two."]), "original.pdf"),
        ("page.html", b"<html><body><p>Made-up page.</p></body></html>", "original.html"),
        ("notes.txt", b"Made-up notes.\n", "original.txt"),
        ("notes.markdown", b"Made-up notes.\n", "original.txt"),  # never from the name
    ],
)
def test_the_original_s_name_comes_from_its_kind(root, tmp_path, name, data, original):
    result = document_add(root, file(tmp_path, name, data))
    assert result["written"][0].endswith("/" + original)


def test_a_pdf_keeps_its_page_starts(root, tmp_path):
    pages = ["Made-up page one.", "Made-up page two.", "Made-up page three."]
    result = document_add(root, file(tmp_path, "guide.pdf", make_pdf(pages)))
    text = (root / "local/documents/guide/text.md").read_text(encoding="utf-8")
    starts = result["added"]["text"]["page_starts"]
    assert [text[s:].startswith(page) for s, page in zip(starts, pages, strict=True)] == [True] * 3


# --- add twice, a crash mid-add


def test_adding_a_taken_name_replaces_the_four_files_and_keeps_anything_else(root, tmp_path):
    document_add(root, file(tmp_path, "notes.md", "Made-up first.\n"))
    mine = root / "local/documents/notes/my-own.txt"
    mine.write_text("Made-up, mine.\n", encoding="utf-8")
    result = document_add(root, file(tmp_path, "notes.md", "Made-up second.\n"))
    path = root / "local/documents/notes"
    assert (path / "text.md").read_text(encoding="utf-8") == "Made-up second.\n"
    assert mine.read_text(encoding="utf-8") == "Made-up, mine.\n"
    assert sorted(p.name for p in path.iterdir()) == sorted(
        ["my-own.txt", "original.txt", "text.md", "parts.jsonl", "manifest.json"]
    )
    assert result["added"]["text"]["bytes"] == len(b"Made-up second.\n")
    assert [p.name for p in path.parent.iterdir()] == ["notes"]  # nothing left aside


def test_parts_over_what_a_search_reads_are_refused_at_add(root, tmp_path, monkeypatch):
    from clarity_chat.documents import library as library_module

    monkeypatch.setattr(library_module, "PARTS_BYTES", 2**20)  # 1 MB, for a small file
    data = "".join(f"## Made-up {n}\nA made-up line.\n\n" for n in range(20_000))
    with pytest.raises(DocumentRefused) as refused:
        document_add(root, file(tmp_path, "notes.md", data))
    assert str(refused.value) == (
        "Clarity can't add notes: its parts would be over 1 MB. Use a smaller file, or split it "
        "and use the part you need."
    )
    assert library(root) == []


def test_a_crash_mid_add_leaves_no_folder(root, tmp_path, monkeypatch):
    from clarity_chat.documents import library as library_module

    real = library_module.write

    def crash(path, data):
        if path.name == "manifest.json":
            raise OSError("made-up crash")
        real(path, data)

    monkeypatch.setattr(library_module, "write", crash)
    with pytest.raises(DocumentRefused, match="^Can't write notes in local/documents/: OSError."):
        document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))  # a refusal since S5
    assert library(root) == []


# --- names (decision 20)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Notes.MD", "notes"),
        ("Birds of the Carbon River.pdf", "birds-of-the-carbon-river"),
        ("-lead.md", "lead"),
        ("__x.md", "x"),
        ("con.txt", None),
        ("none.md", None),
        ("日本語.md", None),
        ("a" * 80 + ".md", "a" * 64),
    ],
)
def test_the_default_name(filename, expected):
    assert default_name(filename) == expected


@pytest.mark.parametrize(
    "name", ["Notes", "none", "con", "lpt9", "-x", "../x", "a/b", "x\x01", "", "a" * 65, "é"]
)
def test_a_name_outside_the_rule_is_refused(name):
    with pytest.raises(DocumentRefused):
        checked_library_name(name)


def test_a_name_with_no_letter_or_digit_asks_for_one(root, tmp_path):
    with pytest.raises(DocumentRefused, match="^Clarity can't make a document name from"):
        document_add(root, file(tmp_path, "日本語.md", "Made-up.\n"))
    result = document_add(root, file(tmp_path, "日本語.md", "Made-up.\n"), "nihongo")
    assert result["added"]["name"] == "nihongo"
    with pytest.raises(DocumentRefused):
        document_add(root, file(tmp_path, "x.md", "Made-up.\n"), "../escape")
    assert not (root / "local" / "escape").exists()


# --- refusals reach add unchanged, and write nothing


@pytest.mark.parametrize(
    ("name", "data", "starts"),
    [
        ("scan.pdf", make_pdf(["", ""]), "scan.pdf has no text Clarity can read."),
        ("locked.pdf", aes_pdf(["Made-up."]), "locked.pdf is protected"),
        ("latin.md", "caf\xe9\n".encode("latin-1"), "latin.md isn't UTF-8 text."),
        ("empty.md", b"", "empty.md is empty."),
    ],
)
def test_v0_3_1_s_refusals_reach_add_and_nothing_is_written(root, tmp_path, name, data, starts):
    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    before = library(root)
    with pytest.raises(DocumentRefused) as refused:
        document_add(root, file(tmp_path, name, data))
    assert str(refused.value).startswith(starts), refused.value
    assert library(root) == before


def test_the_time_limit_reaches_add(root, tmp_path, monkeypatch):
    from tests.test_documents_extract import stub

    stub(monkeypatch, "import time; time.sleep(60)")
    monkeypatch.setattr(reading, "TIME_LIMIT", 1.0)
    with pytest.raises(DocumentRefused, match="^slow.md took longer than 1 seconds to read."):
        document_add(root, file(tmp_path, "slow.md", "Made-up.\n"))
    assert library(root) == []


def test_too_big_is_refused_before_it_s_read(root, tmp_path, monkeypatch):
    monkeypatch.setattr(reading, "MAX_BYTES", 10)
    with pytest.raises(DocumentRefused, match="^big.md is too big"):
        document_add(root, file(tmp_path, "big.md", "Made-up, too long.\n"))
    assert library(root) == []


# --- state


@pytest.mark.parametrize("how", ["parts deleted", "parts grown"])
def test_parts_that_aren_t_the_build_are_refused_with_add_again(root, tmp_path, how):
    from clarity_chat.documents import Unreadable

    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    path = root / "local/documents/notes"
    if how == "parts deleted":
        (path / "parts.jsonl").unlink()
    else:
        with open(path / "parts.jsonl", "ab") as f:
            f.write(b"\n")
    with pytest.raises(Unreadable) as refused:
        load_document(root, "notes")
    assert refused.value.said.startswith("notes"), refused.value.said
    assert str(refused.value).endswith(
        "Add the file again with the paperclip's Search it, under the same name."
    )


def test_an_edited_original_or_text_md_changes_nothing(root, tmp_path):
    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    path = root / "local/documents/notes"
    before = load_document(root, "notes")
    (path / "original.txt").write_bytes(b"Made-up, other.\n")
    (path / "text.md").write_bytes(b"Made-up, edited.\n")
    assert library_state(root, "notes") == "ready"
    assert load_document(root, "notes") == before


def test_a_folder_that_s_gone_is_missing(root, tmp_path):
    import shutil

    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    shutil.rmtree(root / "local/documents/notes")
    assert library_state(root, "notes") == "missing"


def test_list_gives_each_document_and_skips_other_folders(root, tmp_path):
    document_add(root, file(tmp_path, "glossary.md", GLOSSARY))
    (root / "local/documents/.glossary-made-up").mkdir()
    (root / "local/documents/Not A Name").mkdir()
    (listed,) = library_documents(root)
    assert listed.pop("added").endswith("Z")  # the manifest's time, the page's row date (C1)
    assert [listed] == [
        {
            "name": "glossary",
            "kind": "text",
            "original": "glossary.md",
            "bytes": len(GLOSSARY.encode()),
            "parts": 3,
            "titled": 3,
            "state": "ready",
        }
    ]


def test_a_manifest_another_program_wrote_is_refused_naming_it(root, tmp_path):
    document_add(root, file(tmp_path, "notes.md", "Made-up.\n"))
    (root / "local/documents/notes/manifest.json").write_text('{"schema": 1}', encoding="utf-8")
    with pytest.raises(DocumentRefused, match="^notes's manifest.json can't be read"):
        load_document(root, "notes")
    assert library_documents(root)[0]["state"] == "unreadable"


# --- the subcommands


def test_the_subcommands_say_what_they_write(root, tmp_path, capsys):
    path = file(tmp_path, "Glossary.md", GLOSSARY)
    assert cli.main(["--root", str(root), "document", "add", path]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "Reading Glossary.md…",
        "Writing local/documents/glossary/: a copy of the file, its text and its parts.",
        "Clarity wrote local/documents/glossary/: original.txt, text.md, parts.jsonl and "
        "manifest.json (3 parts, 3 with a title).",
    ]
    # Replaced, said before and after (the same bytes: the text read last is used again)
    assert cli.main(["--root", str(root), "document", "add", path]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Replacing glossary in local/documents/glossary/: a copy of the file, its text and its "
        "parts, in place of the ones there.",
        "Clarity replaced local/documents/glossary/: original.txt, text.md, parts.jsonl and "
        "manifest.json (3 parts, 3 with a title).",
    ]
    assert cli.main(["--root", str(root), "document", "list"]) == 0
    assert capsys.readouterr().out.strip().endswith("3 parts, 3 titled  ready")
    assert cli.main(["--root", str(root), "document", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["documents"][0]["name"] == "glossary"


# --- Checkpoint 1's M1: a document folder that's a link


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_linked_document_folder_is_refused_and_nothing_outside_changes(root, tmp_path):
    from clarity_chat.commands import document_list

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "text.md").write_text("Made-up, not Clarity's.\n", encoding="utf-8")
    (elsewhere / "keep.docx").write_bytes(b"made-up")
    (root / "local" / "documents").mkdir(parents=True)
    (root / "local" / "documents" / "notes").symlink_to(elsewhere, target_is_directory=True)
    before = sorted((p.name, p.read_bytes()) for p in elsewhere.iterdir())
    says = "notes's folder in local/documents/ is a link, which Clarity doesn't follow."
    answers = []
    for act in (
        lambda: load_document(root, "notes"),
        lambda: document_add(root, file(tmp_path, "notes.md", "Made-up.\n")),
    ):
        try:
            act()
            answers.append("no refusal")
        except DocumentRefused as e:
            answers.append(str(e))
        except Exception as e:  # noqa: BLE001 - what the bug raised, as the test's answer
            answers.append(type(e).__name__)
    assert sorted((p.name, p.read_bytes()) for p in elsewhere.iterdir()) == before
    assert [a.startswith(says) for a in answers] == [True] * 2, answers
    assert library_state(root, "notes") == "unreadable"
    assert document_list(root)["documents"][0]["state"] == "unreadable"
    assert sorted((p.name, p.read_bytes()) for p in elsewhere.iterdir()) == before
    assert sorted(p.name for p in (root / "local" / "documents").iterdir()) == ["notes"]
