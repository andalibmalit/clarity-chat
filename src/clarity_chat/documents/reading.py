"""Reading a file's text for an attachment: one function from (bytes, name) to a document,
used by every front end, the extraction run in a child process with limits (v0.3.1 A1)."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from clarity_chat.profiles import ConfigError

MAX_BYTES = 64 * 1024 * 1024  # a larger file is refused before it's read
MAX_NAME = 255
TIME_LIMIT = 120.0  # seconds; the child's own deadline is a little later
MEMORY_CAP = 1024 * 1024 * 1024  # the child's peak memory, in bytes
OUTPUT_CAP = 256 * 1024 * 1024  # the child's answer, in bytes
CHILD = [sys.executable, "-I", "-m", "clarity_chat.documents.child"]  # tests run stubs
MEMORY_EXIT, DEADLINE_EXIT = 70, 71  # as in child.py, which the parent doesn't import


class DocumentRefused(ConfigError):
    """A file Clarity won't attach, with the sentence that says why and what to do. Every
    front end shows it as a refusal; the session goes on."""


@dataclass(frozen=True)
class Document:
    """A file's extracted text, in units: a PDF's pages, or lines (split on "\\n" only, each
    keeping its "\\n") for everything else."""

    name: str
    kind: str  # text | html | pdf
    bytes: int
    sha256: str
    extractor: dict[str, Any]
    unit: str  # page | line
    units: tuple[str, ...]
    empty_units: int

    @property
    def total(self) -> int:
        return len(self.units)

    def text_of(self, first: int, last: int) -> str:
        """Units first to last (1-based, inclusive): pages joined by a blank line, as the
        whole is; lines as they are, since each keeps its newline."""
        joiner = "\n\n" if self.unit == "page" else ""
        return joiner.join(self.units[first - 1 : last])

    @property
    def text(self) -> str:
        return self.text_of(1, self.total)

    @property
    def extracted_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


BIDI = frozenset("\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def checked_name(name: str) -> str:
    """A file's name as the record keeps it: never a path, never a control character."""
    # Control characters; lone surrogates (a Linux name that isn't UTF-8, or --json), which
    # can't be written as UTF-8; line and paragraph separators, which act as newlines; and
    # text-direction controls, which show a name reversed (notes\u202edm.txt). Not all of
    # Cf: U+200C is common in Persian names, and U+200D joins emoji.
    if any(unicodedata.category(c) in ("Cc", "Cs", "Zl", "Zp") or c in BIDI for c in name):
        raise DocumentRefused(
            "A file name can't hold a newline, a control character, a text-direction control "
            "or a broken character."
        )
    if not name or len(name) > MAX_NAME:
        raise DocumentRefused(f"A file name must be 1 to {MAX_NAME} characters.")
    return name


def kind_of(data: bytes, name: str) -> str:
    """A PDF by its first bytes, whatever its name; HTML by its name's ending; text else."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    return "html" if name.lower().endswith((".html", ".htm")) else "text"


def read_path(path: str) -> tuple[bytes, str]:
    """A file's bytes and name. Opened without blocking and checked on its descriptor, so a
    FIFO or a device swapped in can't hang the worker; only a regular file of at most
    MAX_BYTES is read."""
    name = checked_name(Path(path).name)
    try:
        # O_BINARY on Windows, where text mode would turn CRLF into LF and stop at 0x1A.
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
    except OSError as e:  # the sentence says why, never the path (str(e) holds it)
        raise DocumentRefused(f"Can't open {name}: {e.strerror or type(e).__name__}.") from e
    except ValueError as e:  # a NUL or a lone surrogate in a folder's name (UnicodeEncodeError)
        raise DocumentRefused(
            f"Can't open {name}: its path holds a character the system can't take."
        ) from e
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise DocumentRefused(f"{name} isn't a file Clarity can attach.")
        if info.st_size > MAX_BYTES:
            raise DocumentRefused(too_big(name))
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(MAX_BYTES + 1)
    except OSError as e:  # a failing disk or mount after the open (EIO), a Windows lock
        raise DocumentRefused(f"Can't read {name}: {e.strerror or type(e).__name__}.") from e
    finally:
        os.close(fd)
    if len(data) > MAX_BYTES:
        raise DocumentRefused(too_big(name))
    return data, name


def too_big(name: str) -> str:
    return f"{name} is too big: Clarity reads files of at most {MAX_BYTES // 1024 // 1024} MB."


_last: Document | None = None  # the last document read, kept so a range change re-reads nothing


def extract(data: bytes, name: str, cancel: threading.Event | None = None) -> Document:
    """The document in `data`, or a DocumentRefused. The same bytes read the same way as the
    last call are read once: the answer is kept, by the bytes' sha256 and the kind (the
    name decides HTML or text, so the same bytes can be read two ways)."""
    global _last
    name = checked_name(name)
    if len(data) > MAX_BYTES:
        raise DocumentRefused(too_big(name))
    sha256 = hashlib.sha256(data).hexdigest()
    kind = kind_of(data, name)
    if _last is not None and (_last.sha256, _last.kind) == (sha256, kind):
        return replace(_last, name=name)
    answer = run_child({"name": name, "kind": kind}, data, cancel)
    if "refused" in answer:
        raise DocumentRefused(answer["refused"])
    if "failed" in answer:
        raise DocumentRefused(f"Clarity couldn't read {name} ({answer['failed']}).")
    _last = Document(
        name=name,
        kind=kind,
        bytes=len(data),
        sha256=sha256,
        extractor=answer["extractor"],
        unit=answer["unit"],
        units=tuple(answer["units"]),
        empty_units=answer["empty_units"],
    )
    return _last


def run_child(request: dict[str, Any], data: bytes, cancel: threading.Event | None) -> dict:
    """The child's answer to `request` and `data`. Killed at the time limit, on `cancel`,
    past the output cap, and if the wait itself is interrupted (Ctrl-C): never left
    running. Its stderr is discarded and it's always reaped."""
    name = request["name"]
    started = time.monotonic()
    request = {**request, "memory_cap": MEMORY_CAP, "seconds": TIME_LIMIT + 5}
    try:
        proc = subprocess.Popen(
            CHILD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"LC_ALL": "C.UTF-8"},  # nothing of Clarity's environment: no tokens, no paths
            # Its own group on POSIX: a terminal's Ctrl-C is Clarity's to turn into a cancel,
            # and an orphan still ends at its own deadline.
            process_group=0 if os.name == "posix" else None,
        )
    except OSError as e:  # too many open files, the interpreter gone mid-session
        raise DocumentRefused(f"Clarity couldn't start its reader: {e.strerror or e}.") from e
    cap = OUTPUT_CAP // 1024 // 1024
    too_long = f"{name} is too long to read: what its reader sent back passed {cap} MB."
    try:
        stdin, stdout = proc.stdin, proc.stdout
        assert stdin is not None and stdout is not None
        out: list[bytes] = []
        size = [0]

        def write() -> None:
            try:
                stdin.write(json.dumps(request).encode("ascii") + b"\n")
                stdin.write(data)
                stdin.close()
            except OSError:  # the child ended early (a limit, a stub): its exit code says why
                pass

        def read() -> None:
            while chunk := stdout.read(1 << 16):
                size[0] += len(chunk)
                if size[0] > OUTPUT_CAP:
                    return
                out.append(chunk)

        threads = [threading.Thread(target=f, daemon=True) for f in (write, read)]
        for thread in threads:
            thread.start()
        why = None
        while proc.poll() is None:
            if time.monotonic() - started > TIME_LIMIT:
                why = f"{name} took longer than {TIME_LIMIT:.0f} seconds to read."
            elif cancel is not None and cancel.is_set():
                why = f"Stopped reading {name}."
            elif size[0] > OUTPUT_CAP:
                why = too_long
            if why:
                break
            time.sleep(0.01)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    for thread in threads:
        thread.join(5)
    if why is None and size[0] > OUTPUT_CAP:
        why = too_long
    if why is None and proc.returncode == MEMORY_EXIT:
        why = f"{name} took more than {MEMORY_CAP // 1024 // 1024} MB of memory to read."
    if why is None and proc.returncode == DEADLINE_EXIT:
        why = f"{name} took longer than {TIME_LIMIT:.0f} seconds to read."
    if why:
        raise DocumentRefused(why)
    try:
        answer = json.loads(b"".join(out))
    except ValueError:
        raise DocumentRefused(
            f"Clarity couldn't read {name}: its reader stopped (exit {proc.returncode})."
        ) from None
    if not isinstance(answer, dict):
        raise DocumentRefused(f"Clarity couldn't read {name}: its reader gave no answer.")
    return answer
