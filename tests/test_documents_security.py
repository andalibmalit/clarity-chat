"""v0.3.2's security review, in tests (notes/plans/v0.3.2-status.md, "security review
rulings"): SM1 (a part holding U+2028, U+2029 or U+0085 reads whole), SS1 (no file in a
document folder is read through a link, or as a FIFO or a folder), SS2 (parts.jsonl read as
untrusted), SS3 (a deeply nested manifest is one unreadable document), SS4 (a failed
--replace never hides the old document) and SS5 (the terminal prints titles through
shown()). Made-up documents, scratch roots; a reader that would hang runs in a thread with
a timeout."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from clarity_chat import cli
from clarity_chat.cli import format_found
from clarity_chat.documents import (
    DocumentRefused,
    library_add,
    library_check,
    library_documents,
    library_parts,
    library_rebuild,
    library_state,
    load_document,
    reading,
)
from clarity_chat.documents import library as library_module
from clarity_chat.logs import read_records
from tests.test_documents_board import chat, quiet, refusal

NOTES = b"# Wren\n\nThe wren is a small brown bird.\n\n# Dipper\n\nIt walks under water.\n"


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, NOTES, "notes.md")
    return fetched_root


def folder(root: Path, name: str = "notes") -> Path:
    return root / "local/documents" / name


def finishes(act, seconds: float = 10.0) -> tuple[bool, object]:
    """Run `act` in a thread: whether it finished in time, and what it returned or raised."""
    out: list[object] = []

    def run():
        try:
            out.append(act())
        except Exception as e:  # noqa: BLE001 - the test looks at what it raised
            out.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    return not thread.is_alive(), out[0] if out else None


def outcome(act) -> object:
    """What `act` returns, or the error it raised as "Type: message", so a red test fails
    on its assertion rather than on the error the bug raises."""
    try:
        return act()
    except Exception as e:  # noqa: BLE001 - the test compares it
        return f"{type(e).__name__}: {e}"


def retamper(root: Path, name: str, change) -> None:
    """parts.jsonl's lines changed by `change`, and the manifest's build ID and file stats
    made to match: the stale check passes, so only the parts' own check can refuse."""
    path = folder(root, name)
    lines = [json.loads(line) for line in (path / "parts.jsonl").read_bytes().split(b"\n")[:-1]]
    for part in lines:
        change(part)
    data = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in lines).encode("utf-8")
    (path / "parts.jsonl").write_bytes(data)
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["build_id"] = hashlib.sha256(data).hexdigest()
    for file in manifest["files"]:
        info = (path / file).stat()
        manifest["files"][file] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def logged(session) -> list[dict]:
    assert session.conversation is not None
    return list(read_records(session.conversation.log.path))


# --- SM1: a line separator inside a part


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
def test_a_part_holding_a_line_separator_reads_whole(root, chat_tokenizer, separator):
    paragraph = f"# Wren\n\nThe winter wren sings{separator}a long song."  # the chunker's part
    library_add(root, f"{paragraph}\n".encode(), "songs.md")
    listed = {d["name"]: d for d in library_documents(root)}
    assert listed["songs"]["state"] == "ready"
    assert outcome(lambda: library_check(root, "songs")["state"]) == "ready"
    assert outcome(lambda: [p["text"] for p in library_parts(root, "songs")[0]]) == [paragraph]
    session, c, _ = chat(root, chat_tokenizer, 'search = ["songs"]')
    assert c.run("try", "Tell me about the wren", quiet)["found"]["parts"][0]["text"] == paragraph
    c.run("say", "Tell me about the wren", quiet)
    assert c.run("state", "", quiet)["searching"][0]["state"] == "ready"


# --- SS1: links, FIFOs and folders inside a document folder


LINKABLE = ["text.md", "original.txt", "parts.jsonl", "manifest.json"]


def link(root: Path, tmp_path: Path, file: str) -> Path:
    if os.name == "nt":
        pytest.skip("symlinks need privileges on Windows")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    private = elsewhere / "private.txt"
    private.write_bytes((folder(root) / file).read_bytes().replace(b"wren", b"secret"))
    (folder(root) / file).unlink()
    (folder(root) / file).symlink_to(private)
    return private


@pytest.mark.parametrize("file", LINKABLE)
def test_a_linked_file_in_a_document_folder_is_never_read(root, tmp_path, file):
    private = link(root, tmp_path, file)
    before = private.read_bytes()
    sentence = (  # the close batch (surface S2): the page's way out
        f"notes's {file} can't be read (it's a link). Add the file again in Documents… and "
        "choose Replace."
    )
    assert library_state(root, "notes") in ("changed", "unreadable")
    assert refusal(lambda: load_document(root, "notes")) in ("notes is changed.", sentence)
    checked = refusal(lambda: library_check(root, "notes"))
    assert checked == sentence or library_check(root, "notes")["state"] == "unreadable"
    if file != "parts.jsonl":  # Rebuild replaces a linked parts.jsonl: the link, not its target
        assert refusal(lambda: library_rebuild(root, "notes")) == sentence
    assert private.read_bytes() == before
    if library_state(root, "notes") == "ready":
        assert all("secret" not in p["text"] for p in library_parts(root, "notes")[0])


def special(path: Path) -> None:
    """A file that isn't a regular one: a FIFO, or on Windows (no FIFOs) a folder."""
    path.unlink()
    if hasattr(os, "mkfifo"):
        os.mkfifo(path)
    else:
        path.mkdir()


@pytest.mark.parametrize("file", LINKABLE)
def test_a_fifo_in_a_document_folder_never_hangs_a_reader(root, file):
    special(folder(root) / file)
    state = library_documents(root)[0]["state"]  # before Rebuild, which replaces parts.jsonl
    sentence = (  # the close batch (surface S2): the page's way out
        f"notes's {file} can't be read (it isn't a regular file). Add the file again in "
        "Documents… and choose Replace."
    )
    for act in (
        lambda: library_documents(root),
        lambda: library_check(root, "notes"),
        lambda: library_rebuild(root, "notes"),
        lambda: load_document(root, "notes"),
    ):
        done, result = finishes(act)
        assert done, f"{file}: a reader hung"
        if isinstance(result, DocumentRefused):
            assert str(result) in (sentence, "notes is changed.") or "parts" in file, result
    assert state in ("changed", "unreadable")
    if file == "manifest.json":  # the row's reason holds no way out (the close batch, S2)
        assert library_documents(root)[0]["why"] == sentence.split(" Add the file")[0]


# --- SS2: parts.jsonl, read as untrusted


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(page='1" injected="yes'),
        lambda p: p.update(text=7),
        lambda p: p.update(title=["a", "list"]),
        lambda p: p.update(sha256=hashlib.sha256(b"other").hexdigest()),
        lambda p: p.update(ordinal=p["ordinal"] + 5),
        lambda p: p.update(extra="key"),
    ],
    ids=["page", "text", "title", "sha256", "ordinal", "extra"],
)
def test_a_tampered_parts_file_is_refused_everywhere(root, chat_tokenizer, change):
    retamper(root, "notes", change)
    said = "notes's parts.jsonl isn't one Clarity writes. Rebuild it."
    assert refusal(lambda: library_parts(root, "notes")) == said
    session, c, _ = chat(root, chat_tokenizer)
    assert refusal(lambda: c.run("search", "notes", quiet)) == (  # as Send says it (fix 2's C1)
        "Clarity can't search notes: its parts.jsonl isn't one Clarity writes. Rebuild it."
    )
    session, c, _ = chat(root, chat_tokenizer, 'search = ["notes"]')
    # Rebuild fixes it, so it's changed, as Send says it (the close batch: siblings C3).
    assert c.run("state", "", quiet)["searching"][0]["state"] == "changed"
    assert refusal(lambda: c.run("try", "Tell me about the wren", quiet)) == (
        "Clarity can't search notes: its parts.jsonl isn't one Clarity writes. Rebuild it."
    )
    assert refusal(lambda: c.run("say", "Tell me about the wren", quiet)) == (
        "Nothing was sent.\nClarity can't search notes: its parts.jsonl isn't one Clarity "
        "writes. Rebuild it."
    )
    assert all("found" not in r for r in logged(session))  # CM1's path: the turn, no parts
    library_rebuild(root, "notes")
    assert library_state(root, "notes") == "ready"


# --- SS3: a manifest nested too deep


def test_a_deeply_nested_manifest_is_one_unreadable_document(root, chat_tokenizer):
    library_add(root, b"# Heron\n\nTall.\n", "herons.md")
    (folder(root) / "manifest.json").write_text("[" * 200_000, encoding="utf-8")
    listed = outcome(lambda: {d["name"]: d for d in library_documents(root)})
    assert isinstance(listed, dict), listed
    assert listed["notes"]["state"] == "unreadable" and listed["herons"]["state"] == "ready"
    assert listed["notes"]["why"].startswith("notes's manifest.json can't be read (")
    assert library_state(root, "notes") == "unreadable"
    session, c, _ = chat(root, chat_tokenizer)
    assert refusal(lambda: c.run("search", "notes", quiet)).startswith(  # N32 (close batch)
        "Clarity can't search notes: its manifest.json can't be read ("
    )


def test_a_deeply_nested_parts_line_is_refused(root):
    retamper(root, "notes", lambda p: p.update(title=None))
    path = folder(root) / "parts.jsonl"
    data = path.read_bytes() + b"[" * 200_000 + b"\n"
    path.write_bytes(data)
    manifest = json.loads((folder(root) / "manifest.json").read_bytes())
    manifest["build_id"] = hashlib.sha256(data).hexdigest()
    info = path.stat()
    manifest["files"]["parts.jsonl"] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    (folder(root) / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert outcome(lambda: library_parts(root, "notes")) == (
        "PartsRefused: notes's parts.jsonl isn't one Clarity writes. Rebuild it."
    )


# --- SS4: --replace never hides the old document


def test_a_failed_rename_puts_the_old_document_back(root, monkeypatch):
    (folder(root) / "my-notes.txt").write_text("Mine.\n", encoding="utf-8")
    real = os.replace
    calls = []

    def second_fails(a, b):
        calls.append((a, b))
        if len(calls) == 2:
            raise PermissionError(13, "Permission denied")
        return real(a, b)

    monkeypatch.setattr(library_module.os, "replace", second_fails)
    said = refusal(lambda: library_add(root, b"# New\n\nNew text.\n", "notes.md", replace=True))
    monkeypatch.setattr(library_module.os, "replace", real)
    assert said == "Can't write notes in local/documents/: Permission denied."
    assert library_state(root, "notes") == "ready"
    assert (folder(root) / "my-notes.txt").read_text(encoding="utf-8") == "Mine.\n"
    assert "wren" in (folder(root) / "text.md").read_text(encoding="utf-8")
    assert sorted(p.name for p in (root / "local/documents").iterdir()) == ["notes"]


def test_replace_refuses_a_target_that_isn_t_a_folder(root):
    path = root / "local/documents/plain"
    path.write_text("Not a document.\n", encoding="utf-8")
    said = refusal(lambda: library_add(root, b"Made-up.\n", "plain.md", replace=True))
    assert said == (
        "local/documents/plain isn't a folder, so Clarity can't replace it. Move it away, then "
        "add the file again."
    )
    assert path.read_text(encoding="utf-8") == "Not a document.\n"
    assert sorted(p.name for p in (root / "local/documents").iterdir()) == ["notes", "plain"]


def test_an_old_folder_that_can_t_be_cleared_is_named(root):
    (folder(root) / "text.md").unlink()
    (folder(root) / "text.md").mkdir()
    (folder(root) / "text.md" / "kept.txt").write_text("Mine.\n", encoding="utf-8")
    said = refusal(lambda: library_add(root, b"# New\n\nNew text.\n", "notes.md", replace=True))
    assert said.startswith("Clarity replaced notes, but couldn't clear its old folder")
    hidden = said.rsplit("local/documents/", 1)[1].removesuffix("/.")
    assert (root / "local/documents" / hidden / "text.md" / "kept.txt").is_file()
    assert library_state(root, "notes") == "ready"


# --- SS5: titles in the terminal


def test_format_found_prints_a_title_through_shown(root, chat_tokenizer):
    heading = "\x1b]0;OWNED\x07Wren \x1b[31mred"
    library_add(root, f"# {heading}\n\nThe wren sings.\n".encode(), "odd.md")
    session, c, _ = chat(root, chat_tokenizer, 'search = ["odd"]')
    text = format_found(c.run("try", "Tell me about the wren", quiet)["found"])
    assert "\x1b" not in text and "\x07" not in text
    assert '"\\x1b]0;OWNED\\x07Wren \\x1b[31mred"' in text
    assert cli is not None
