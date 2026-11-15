"""Reading a file's text for an attachment: one function from (bytes, name) to a document,
used by every front end, the extraction run in a child process with limits (v0.3.1 A1)."""

from __future__ import annotations

import hashlib
import sys
import threading
from dataclasses import dataclass
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


def checked_name(name: str) -> str:
    return name  # A1's stub: no checks yet


def kind_of(data: bytes, name: str) -> str:
    return "text"


def read_path(path: str) -> tuple[bytes, str]:
    raise DocumentRefused("Not built yet.")


_last: Document | None = None


def extract(data: bytes, name: str, cancel: threading.Event | None = None) -> Document:
    """A1's stub: no child, no refusals."""
    return Document(
        name=name,
        kind=kind_of(data, name),
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        extractor={},
        unit="line",
        units=tuple(data.decode("utf-8", "replace").split("\n")),
        empty_units=0,
    )


def run_child(request: dict[str, Any], data: bytes, cancel: threading.Event | None) -> dict:
    return {}
