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

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clarity_chat.documents import reading
from clarity_chat.documents.parts import (
    CHUNKER,
    TITLE_CHARS,
    build_id,
    closing_tag_parts,
    cut_parts,
    parts_jsonl,
)
from clarity_chat.documents.reading import (
    SIZE_LIMIT,
    Document,
    DocumentRefused,
    extract,
)
from clarity_chat.profiles import library_name_problem

LIBRARY = Path("local") / "documents"
TAKEN = "There's already a document named"  # each front end adds its way out (S16)
SCHEMA = 1
EXTENSIONS = {"pdf": "pdf", "html": "html", "text": "txt"}
MANIFEST_BYTES = 16 * 1024 * 1024  # far past any manifest Clarity writes; a bigger one is cut


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
    library = root / LIBRARY  # itself may be a link: a library on another disk is followed
    path = library / checked_library_name(name)
    if path.is_symlink() or path.is_junction() or path.resolve().parent != library.resolve():
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
    """Each file's size and modification time, which the quick check compares: of the file
    itself, so a link put in its place reads as changed."""
    found = {}
    for name in names:
        info = (path / name).lstat()
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
    on_write: Callable[[str, bool], None] | None = None,
) -> dict[str, Any]:
    """Add a file to the library: read it as v0.3.1 does (its child, its limits), then write
    the four files. `on_write` gets the folder (relative to the project), and whether it
    replaces a document there, just before anything is written. Nothing is written on a
    refusal."""
    if name is None:
        name = default_name(filename)
        if name is None:
            raise DocumentRefused(
                f"Clarity can't make a document name from {filename}. Give it one."
            )
    target = folder(root, name)
    if target.exists() and not replace:
        raise DocumentRefused(f"{TAKEN} {name}.")
    if target.exists() and not target.is_dir():  # never moved aside (the security review's S4)
        raise DocumentRefused(
            f"local/documents/{name} isn't a folder, so Clarity can't replace it. Move it away, "
            "then add the file again."
        )
    doc = extract(data, filename, cancel, on_read)
    original = {"name": doc.name, "kind": doc.kind, "bytes": doc.bytes, "sha256": doc.sha256}
    text_md, jsonl, manifest = built(
        doc.text, name, original, doc.extractor, page_starts(doc), edited=False
    )
    relative = target.relative_to(root).as_posix()
    replacing = target.exists()
    if on_write is not None:
        on_write(relative, replacing)
    files = [original_name(doc.kind), "text.md", "parts.jsonl", "manifest.json"]
    with writing(name):
        target.parent.mkdir(parents=True, exist_ok=True)
        # mkdtemp's build folder; an aside is `.NAME.old-…`, which this never matches
        stale = re.compile(rf"\.{re.escape(name)}-[a-z0-9_]{{8}}")
        for left in target.parent.iterdir():  # a build a hard crash left: its copy goes
            if stale.fullmatch(left.name) and left.is_dir() and not left.is_symlink():
                shutil.rmtree(left)
        building = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
        try:
            write(building / files[0], data)
            write(building / "text.md", text_md)
            write(building / "parts.jsonl", jsonl)
            manifest["files"] = stats(building, files[:3])
            write(building / "manifest.json", manifest_bytes(manifest))
            swap_in(building, target, replace)
        finally:
            if building.exists():
                shutil.rmtree(building)
    written = [f"{relative}/{f}" for f in files]
    return {"added": manifest, "written": written, "replaced": replacing}


@contextmanager
def writing(name: str) -> Generator[None]:
    """A write the system refuses (a folder you can't write, a full disk, a file another
    program holds open) is a refusal that says why, never a path."""
    try:
        yield
    except OSError as e:
        raise DocumentRefused(
            f"Can't write {name} in local/documents/: {e.strerror or type(e).__name__}."
        ) from None


KNOWN = ("original.pdf", "original.html", "original.txt", "text.md", "parts.jsonl")


def swap_in(building: Path, target: Path, replace: bool) -> None:
    """The new folder renamed into place. Without `replace`, a folder that appeared there
    meanwhile (another add) stays, and this one is refused (Checkpoint 1's S5). With it, a
    document already there is renamed aside first, its four known files deleted, and
    anything else you put in its folder moved across."""
    if not replace or not target.exists():
        try:
            os.rename(building, target)  # never over a folder: one there refuses it
        except OSError:
            if target.exists():
                raise DocumentRefused(f"{TAKEN} {target.name}.") from None
            raise
        return
    # A prefix no name makes (a name holds no "."), so another document's stale-build
    # cleanup can never match it (part 1's whole-branch C5).
    aside = Path(tempfile.mkdtemp(prefix=f".{target.name}.old-", dir=target.parent))
    os.rmdir(aside)
    os.replace(target, aside)
    try:
        os.replace(building, target)
    except OSError:  # Windows refusing a folder another program holds: the old one back
        try:
            os.replace(aside, target)
        except OSError as e:  # that failed too: say where it is (the fix review's C9)
            raise DocumentRefused(
                f"Clarity couldn't replace {target.name} or put it back "
                f"({e.strerror or type(e).__name__}). It's in local/documents/{aside.name}/."
            ) from None
        raise
    try:
        for item in aside.iterdir():
            if item.name in (*KNOWN, "manifest.json"):
                item.unlink()
            else:
                os.replace(item, target / item.name)
        os.rmdir(aside)
    except OSError as e:  # replaced, but your files are in the hidden folder: named
        raise DocumentRefused(
            f"Clarity replaced {target.name}, but couldn't clear its old folder "
            f"({e.strerror or type(e).__name__}). What's left of it is in "
            f"local/documents/{aside.name}/."
        ) from None


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def read_manifest(root: Path, name: str) -> dict[str, Any]:
    """A document's manifest, read as untrusted: a folder that's gone is `missing`; a
    manifest that isn't one Clarity writes is refused, naming it."""
    path = folder(root, name) / "manifest.json"
    if not path.parent.is_dir():
        raise Stale(name, "missing")
    try:
        manifest = json.loads(read_file(name, path, MANIFEST_BYTES))
        if not typed(manifest):
            raise ValueError("it isn't a manifest Clarity writes")
    except OSError as e:  # the reason, never the path (str(e) holds it)
        why = e.strerror or type(e).__name__
        raise DocumentRefused(unreadable(name, "manifest.json", why)) from None
    except RecursionError:  # nested too deep: this document only (the security review's S3)
        raise DocumentRefused(unreadable(name, "manifest.json", "it's nested too deep")) from None
    except ValueError as e:
        raise DocumentRefused(unreadable(name, "manifest.json", str(e))) from None
    return manifest


def unreadable(name: str, file: str, why: str) -> str:
    """The way out of a file in a document's folder that Clarity can't read (the board
    walk's W4): the folder goes, then the file is added again."""
    return (
        f"{name}'s {file} can't be read ({why}). Delete the folder local/documents/{name}/, "
        "then add the file again."
    )


def read_file(name: str, path: Path, limit: int) -> bytes:
    """A file in a document's folder, read as untrusted (the security review's S1): never
    through a link, never a FIFO or a device (opened without blocking and checked on its
    descriptor, as v0.3.1's reader is), and at most `limit` + 1 bytes, so a file swapped in
    can't fill memory: its hash then differs from the manifest's."""
    link = unreadable(name, path.name, "it's a link")
    if path.is_symlink() or path.is_junction():
        raise DocumentRefused(link)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    special = unreadable(name, path.name, "it isn't a regular file")
    try:
        fd = os.open(path, flags)
    except OSError as e:
        if e.errno == errno.ELOOP:  # a link put there since the check above
            raise DocumentRefused(link) from None
        if path.is_dir():  # Windows won't open a folder; POSIX opens it, and fstat says
            raise DocumentRefused(special) from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DocumentRefused(special)
        with os.fdopen(fd, "rb", closefd=False) as f:
            return f.read(limit + 1)
    finally:
        os.close(fd)


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def typed(manifest: Any) -> bool:
    """Whether a manifest holds what the readers rely on, each of the type Clarity writes:
    read from disk, it's untrusted (Checkpoint 1's S4)."""
    try:
        original, text, files = manifest["original"], manifest["text"], manifest["files"]
        starts = text["page_starts"]
        return (
            manifest["schema"] == SCHEMA
            and isinstance(original["name"], str)
            and original["kind"] in EXTENSIONS
            and _int(original["bytes"])
            and isinstance(original["sha256"], str)
            and isinstance(text["sha256"], str)
            and _int(text["bytes"])
            and isinstance(text["edited"], bool)
            and (starts is None or (isinstance(starts, list) and all(map(_int, starts))))
            and isinstance(manifest["build_id"], str)
            and _int(manifest["parts"])
            and _int(manifest["titled"])
            and isinstance(manifest["extractor"], dict)
            and isinstance(manifest["added"], str)
            and isinstance(files, dict)
            and set(files) == {original_name(original["kind"]), "text.md", "parts.jsonl"}
            and all(
                isinstance(f, dict)
                and set(f) == {"bytes", "mtime_ns"}
                and all(map(_int, f.values()))
                for f in files.values()
            )
        )
    except (KeyError, TypeError, AttributeError):
        return False


class PartsRefused(DocumentRefused):
    """A parts.jsonl that isn't one Clarity writes (the security review's S2): Rebuild cuts
    it again, so a search says that, not "unreadable"."""


class Stale(DocumentRefused):
    """A document that doesn't match its manifest (`changed`) or is gone (`missing`). Each
    front end words it with its own way out."""

    def __init__(self, name: str, state: str) -> None:
        self.name, self.state = name, state
        super().__init__(f"{name} is {state}.")


def state(root: Path, name: str, manifest: dict[str, Any] | None = None) -> str:
    """`ready`, `changed` (a file's size or time differs from the manifest), `missing`, or
    `unreadable` (a manifest Clarity can't read, or a folder that's a link): the quick check
    before each search. `manifest`: one already read."""
    if manifest is None:
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
                "state": state(root, path.name, manifest),  # read once (part 1's C4)
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
            if sha256(read_file(name, path / file, manifest["files"][file]["bytes"])) != digest:
                problems.append(f"{file} has changed")
        except OSError:
            problems.append(f"{file} is missing")
    found: dict[str, Any] = {"name": name, "state": "changed" if problems else "ready"}
    found["problems"] = problems
    if not problems:  # decision 7: listed, as v0.3.1 lists </document>
        found["closing_tag_parts"] = closing_tag_parts(parts_of(root, name, manifest)[0])
    return found


def parts_of(
    root: Path, name: str, manifest: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], str]:
    """A document's parts, read from parts.jsonl, and the build ID they are: the one in
    `manifest` (read now, if not given), or the document is `changed`. Each line is read as
    untrusted (the security review's S2), split on "\n" only: a part's text may hold U+2028,
    U+2029 or U+0085, which JSON writes raw (its M1)."""
    if manifest is None:
        manifest = read_manifest(root, name)
    path = folder(root, name) / "parts.jsonl"
    try:
        data = read_file(name, path, manifest["files"]["parts.jsonl"]["bytes"])
    except OSError:
        raise Stale(name, "changed") from None
    if build_id(data) != manifest["build_id"]:
        raise Stale(name, "changed")
    try:
        *lines, end = data.decode("utf-8").split("\n")
        parts = [json.loads(line) for line in lines]
        ok = end == "" and all(written(part, n) for n, part in enumerate(parts, 1))
    except (ValueError, RecursionError):
        ok = False
    if not ok:
        raise PartsRefused(f"{name}'s parts.jsonl isn't one Clarity writes. Rebuild it.")
    return parts, manifest["build_id"]


PART_KEYS = {"ordinal", "title", "text", "chars", "page", "words", "sha256"}


def written(part: Any, ordinal: int) -> bool:
    """Whether a line of parts.jsonl is a part as the chunker writes it: its page goes into
    the block, its title into the block and the terminal, its sha256 into `already`."""
    try:
        title, chars = part["title"], part["chars"]
        return (
            set(part) == PART_KEYS
            and _int(part["ordinal"])
            and part["ordinal"] == ordinal
            and (
                title is None
                or (isinstance(title, str) and "\n" not in title and len(title) <= TITLE_CHARS)
            )
            and isinstance(part["text"], str)
            and part["sha256"] == sha256(part["text"].encode("utf-8"))
            and (part["page"] is None or _int(part["page"]))
            and isinstance(chars, list)
            and len(chars) == 2
            and all(map(_int, chars))
            and _int(part["words"])
        )
    except (KeyError, TypeError, AttributeError, ValueError):  # ValueError: a lone surrogate
        return False


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


class OriginalChanged(DocumentRefused):
    """A rebuild refused because the original isn't the file the text came from: the way
    out is to add the file again (Checkpoint 1's S3; N72). `said` is the reason, which the
    terminal ends with its own command."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.said = (
            f"Clarity can't rebuild {name}: its original file has changed or is gone, so its "
            "text.md may no longer match it."
        )
        super().__init__(
            f"{self.said} Delete the folder local/documents/{name}/, then add the file again."
        )


def rebuild(root: Path, name: str) -> dict[str, Any]:
    """Build a document's parts again from its text.md, which is read as your input (v0.3.1's
    checks: UTF-8, no NUL, at most 64 MB). An edited text drops its pages; an unchanged one
    (only its time moved: a copy, a sync, a save) keeps them and stays unedited. The
    original must be the file the text came from, or the way out is adding it again."""
    manifest = read_manifest(root, name)
    path = folder(root, name)
    original_file = original_name(manifest["original"]["kind"])
    try:
        original = read_file(name, path / original_file, manifest["original"]["bytes"])
    except OSError:
        raise OriginalChanged(name) from None
    if sha256(original) != manifest["original"]["sha256"]:
        raise OriginalChanged(name)
    data = edited_text(name, path / "text.md")
    doc = extract(data, "text.md")
    unchanged = sha256(data) == manifest["text"]["sha256"]
    starts = manifest["text"]["page_starts"] if unchanged else None
    edited = manifest["text"]["edited"] if unchanged else True
    _, jsonl, rebuilt = built(
        doc.text, name, manifest["original"], manifest["extractor"], starts, edited
    )
    rebuilt["added"] = manifest["added"]
    rebuilt["text"]["sha256"] = sha256(data)  # the file as it is, BOM and all
    rebuilt["text"]["bytes"] = len(data)
    with writing(name):
        replace_file(path / "parts.jsonl", jsonl)
        rebuilt["files"] = stats(path, [original_file, "text.md", "parts.jsonl"])
        replace_file(path / "manifest.json", manifest_bytes(rebuilt))
    relative = path.relative_to(root).as_posix()
    return {
        "rebuilt": rebuilt,
        "written": [f"{relative}/parts.jsonl", f"{relative}/manifest.json"],
        # The board walk's W5: an edit of a PDF's text.md loses its page numbers, said once.
        "pages_dropped": manifest["text"]["page_starts"] is not None and starts is None,
    }


def edited_text(name: str, path: Path) -> bytes:
    """text.md's bytes, read as your input, with Rebuild's own sentences (the board walk's
    S1): each names the document and text.md, and says to save it, then rebuild."""
    try:
        data = read_file(name, path, reading.MAX_BYTES)  # never through a link (the review's S1)
    except OSError as e:
        raise DocumentRefused(unreadable(name, "text.md", e.strerror or type(e).__name__)) from None
    if len(data) > reading.MAX_BYTES:
        raise DocumentRefused(
            f"{name}: text.md is too big: Clarity reads files up to {SIZE_LIMIT}. Shorten it, "
            "then rebuild."
        )
    if b"\x00" in data:
        raise DocumentRefused(
            f"{name}: text.md isn't text Clarity can read. Save it as UTF-8 text, then rebuild."
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise DocumentRefused(
            f"{name}: text.md isn't UTF-8 text. Save text.md as UTF-8 text, then rebuild."
        ) from None
    return data


def replace_file(path: Path, data: bytes) -> None:
    """`path`'s contents replaced whole: written beside it under a name of its own, then
    renamed over it, so two rebuilds never share a half-written file."""
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(handle)
    try:
        write(Path(temporary), data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
