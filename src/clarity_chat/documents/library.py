"""The library (v0.3.2 decisions 5, 6 and 20): documents Clarity searches, kept on disk under
`local/documents/NAME/` as four files:

    original.EXT   the bytes you added (EXT by kind: pdf, html or txt)
    text.md        the extracted text: the truth for everything after it, yours to fix
    parts.jsonl    its parts (parts.py)
    manifest.json  what was built from what, and each file's size and time

A document is built in a temporary folder beside its target and renamed into place, so a
crash leaves no half document. Stale is an error, never a silent rebuild: a document whose
files don't match its manifest is `changed`, a folder that's gone is `missing`, and each
front end says the way out."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clarity_chat.documents.parts import CHUNKER, build_id, cut_parts, parts_jsonl
from clarity_chat.documents.reading import Document, DocumentRefused, extract, read_path
from clarity_chat.profiles import NAME_RULE, library_name_problem

LIBRARY = Path("local") / "documents"
SCHEMA = 1
EXTENSIONS = {"pdf": "pdf", "html": "html", "text": "txt"}


def checked_library_name(name: str) -> str:
    """A library name, refused unless it's decision 20's (profiles' rule, which a profile's
    `search` list uses too): it only ever becomes one folder under local/documents/."""
    problem = library_name_problem(name)
    if problem is not None:
        raise DocumentRefused(problem)
    return name


def default_name(filename: str) -> str | None:
    """A file's default name: its stem lowercased, other characters as `-`, from its first
    letter or digit, at most 64 characters. None when no letter or digit is left (a name in
    another script, say), or the result is reserved: then you name it."""
    stem = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9_-]", "-", stem)
    name = re.sub(r"^[^a-z0-9]+", "", name)[:64]
    return name if library_name_problem(name) is None else None


def folder(root: Path, name: str) -> Path:
    """A document's folder: one folder under local/documents/, never a link (Checkpoint 1's
    M1: replacing through a link would delete and move files outside the project)."""
    library = root / LIBRARY
    path = library / checked_library_name(name)
    if path.is_symlink() or path.resolve().parent != library.resolve():
        raise DocumentRefused(
            f"{name}'s folder in local/documents/ is a link, which Clarity doesn't follow. "
            "Put the folder itself there."
        )
    return path


def original_name(kind: str) -> str:
    return f"original.{EXTENSIONS[kind]}"


def page_starts(doc: Document) -> list[int] | None:
    """Where each page starts in a PDF's text (pages joined by a blank line, as v0.3.1's
    whole is); None for text and HTML."""
    if doc.unit != "page":
        return None
    starts, at = [], 0
    for page in doc.units:
        starts.append(at)
        at += len(page) + 2
    return starts


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stats(path: Path, names: list[str]) -> dict[str, dict[str, int]]:
    """Each file's size and modification time, which the quick check compares."""
    found = {}
    for name in names:
        info = (path / name).stat()
        found[name] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    return found


def write(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def built(
    text: str,
    name: str,
    original: dict[str, Any],
    extractor: dict[str, Any],
    starts: list[int] | None,
    edited: bool,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """text.md's bytes, parts.jsonl's, and the manifest without its file stats."""
    parts = cut_parts(text, original["name"], starts)
    jsonl = parts_jsonl(parts)
    encoded = text.encode("utf-8")
    manifest = {
        "schema": SCHEMA,
        "name": name,
        "added": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "original": original,
        "extractor": extractor,
        "text": {
            "sha256": sha256(encoded),
            "bytes": len(encoded),
            "edited": edited,
            "page_starts": starts,
        },
        "chunker": CHUNKER,
        "build_id": build_id(jsonl),
        "parts": len(parts),
        "titled": sum(1 for p in parts if p["title"] is not None),
    }
    return encoded, jsonl, manifest


def add(
    root: Path,
    data: bytes,
    filename: str,
    name: str | None = None,
    replace: bool = False,
    cancel: threading.Event | None = None,
    on_read: Callable[[], None] | None = None,
    on_write: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Add a file to the library: read it as v0.3.1 does (its child, its limits), then write
    the four files. `on_write` gets the folder (relative to the project) just before
    anything is written. Nothing is written on a refusal."""
    if name is None:
        name = default_name(filename)
        if name is None:
            raise DocumentRefused(
                f"Clarity can't make a document name from {filename}. Give it one: {NAME_RULE}"
            )
    target = folder(root, name)
    if target.exists() and not replace:
        raise DocumentRefused(f"There's already a document named {name}.")
    doc = extract(data, filename, cancel, on_read)
    original = {"name": doc.name, "kind": doc.kind, "bytes": doc.bytes, "sha256": doc.sha256}
    text_md, jsonl, manifest = built(
        doc.text, name, original, doc.extractor, page_starts(doc), edited=False
    )
    relative = target.relative_to(root).as_posix()
    if on_write is not None:
        on_write(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [original_name(doc.kind), "text.md", "parts.jsonl", "manifest.json"]
    building = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
    try:
        write(building / files[0], data)
        write(building / "text.md", text_md)
        write(building / "parts.jsonl", jsonl)
        manifest["files"] = stats(building, files[:3])
        write(building / "manifest.json", manifest_bytes(manifest))
        swap_in(building, target)
    finally:
        if building.exists():
            shutil.rmtree(building)
    return {"added": manifest, "written": [f"{relative}/{f}" for f in files]}


KNOWN = ("original.pdf", "original.html", "original.txt", "text.md", "parts.jsonl")


def swap_in(building: Path, target: Path) -> None:
    """The new folder renamed into place. A document already there is renamed aside first,
    its four known files deleted, and anything else you put in its folder moved across."""
    if not target.exists():
        os.replace(building, target)
        return
    aside = Path(tempfile.mkdtemp(prefix=f".{target.name}-old-", dir=target.parent))
    os.rmdir(aside)
    os.replace(target, aside)
    os.replace(building, target)
    for item in aside.iterdir():
        if item.name in (*KNOWN, "manifest.json"):
            item.unlink()
        else:
            os.replace(item, target / item.name)
    os.rmdir(aside)


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def read_manifest(root: Path, name: str) -> dict[str, Any]:
    """A document's manifest, read as untrusted: a folder that's gone is `missing`; a
    manifest that isn't one Clarity writes is refused, naming it."""
    path = folder(root, name) / "manifest.json"
    if not path.parent.is_dir():
        raise Stale(name, "missing")
    try:
        manifest = json.loads(path.read_bytes())
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ValueError("not a manifest Clarity writes")
        for key in ("original", "text", "build_id", "files", "parts", "titled"):
            manifest[key]  # noqa: B018 - each must be there
    except (OSError, ValueError, KeyError) as e:
        raise DocumentRefused(
            f"{name}'s manifest.json can't be read ({type(e).__name__}: {e}). Add the file "
            "again to build it anew."
        ) from None
    return manifest


class Stale(DocumentRefused):
    """A document that doesn't match its manifest (`changed`) or is gone (`missing`). Each
    front end words it with its own way out."""

    def __init__(self, name: str, state: str) -> None:
        self.name, self.state = name, state
        super().__init__(f"{name} is {state}.")


def state(root: Path, name: str) -> str:
    """`ready`, `changed` (a file's size or time differs from the manifest), `missing`, or
    `unreadable` (a manifest Clarity can't read, or a folder that's a link): the quick check
    before each search."""
    try:
        manifest = read_manifest(root, name)
    except Stale as e:
        return e.state
    except DocumentRefused:
        return "unreadable"
    path = folder(root, name)
    try:
        now = stats(path, list(manifest["files"]))
    except OSError:
        return "changed"
    return "ready" if now == manifest["files"] else "changed"


def documents(root: Path) -> list[dict[str, Any]]:
    """The library, by name: each document's manifest facts and state. A folder whose name
    isn't a document's (a temporary one, say) isn't listed."""
    base = root / LIBRARY
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.iterdir()):
        if not path.is_dir() or library_name_problem(path.name) is not None:
            continue  # a link to a folder is listed, as unreadable, below
        try:
            manifest = read_manifest(root, path.name)
        except DocumentRefused as e:
            found.append({"name": path.name, "state": "unreadable", "why": str(e)})
            continue
        found.append(
            {
                "name": path.name,
                "kind": manifest["original"]["kind"],
                "original": manifest["original"]["name"],
                "bytes": manifest["original"]["bytes"],
                "parts": manifest["parts"],
                "titled": manifest["titled"],
                "edited": manifest["text"]["edited"],
                "state": state(root, path.name),
            }
        )
    return found


def check(root: Path, name: str) -> dict[str, Any]:
    """The full check: every file hashed against the manifest. `problems` names each file
    that differs; the state is `ready` only with none."""
    manifest = read_manifest(root, name)
    path = folder(root, name)
    expected = {
        original_name(manifest["original"]["kind"]): manifest["original"]["sha256"],
        "text.md": manifest["text"]["sha256"],
        "parts.jsonl": manifest["build_id"],
    }
    problems = []
    for file, digest in expected.items():
        try:
            if sha256((path / file).read_bytes()) != digest:
                problems.append(f"{file} has changed")
        except OSError:
            problems.append(f"{file} is missing")
    return {"name": name, "state": "changed" if problems else "ready", "problems": problems}


def parts_of(root: Path, name: str) -> list[dict[str, Any]]:
    """A document's parts, read from parts.jsonl and checked against its build ID."""
    manifest = read_manifest(root, name)
    data = (folder(root, name) / "parts.jsonl").read_bytes()
    if build_id(data) != manifest["build_id"]:
        raise Stale(name, "changed")
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


def control_token_parts(
    parts: list[dict[str, Any]], special_tokens_in: Callable[[str], list[str]]
) -> list[dict[str, Any]]:
    """The parts holding text a model's tokenizer reads as control tokens."""
    found = []
    for part in parts:
        tokens = special_tokens_in(part["text"])
        if tokens:
            found.append({"ordinal": part["ordinal"], "tokens": tokens})
    return found


def rebuild(root: Path, name: str) -> dict[str, Any]:
    """Build a document's parts again from its text.md, which is read as your input (v0.3.1's
    checks: UTF-8, no NUL, at most 64 MB). Pages are dropped: an edited text has none."""
    manifest = read_manifest(root, name)
    path = folder(root, name)
    data, _ = read_path(str(path / "text.md"))
    doc = extract(data, "text.md")
    original_file = original_name(manifest["original"]["kind"])
    text_md, jsonl, rebuilt = built(
        doc.text, name, manifest["original"], manifest["extractor"], None, edited=True
    )
    rebuilt["added"] = manifest["added"]
    rebuilt["text"]["sha256"] = sha256(data)  # the file as it is, BOM and all
    rebuilt["text"]["bytes"] = len(data)
    del text_md  # the file stays as you wrote it
    write(path / "parts.jsonl.new", jsonl)
    os.replace(path / "parts.jsonl.new", path / "parts.jsonl")
    rebuilt["files"] = stats(path, [original_file, "text.md", "parts.jsonl"])
    write(path / "manifest.json.new", manifest_bytes(rebuilt))
    os.replace(path / "manifest.json.new", path / "manifest.json")
    relative = path.relative_to(root).as_posix()
    return {"rebuilt": rebuilt, "written": [f"{relative}/parts.jsonl", f"{relative}/manifest.json"]}
