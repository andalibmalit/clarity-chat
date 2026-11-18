"""The vectors a meaning search compares: a cache beside a document's parts (decision 9).

`local/documents/NAME/vectors-KEY.f32` holds one little-endian float32 vector per part, in
part order, with no header. KEY is the sha256 of the document's build ID and the embedder's
setup, so a new build or setup is a new file. A file that isn't there, or doesn't read, is
built again before the search; writing one deletes the folder's other vectors files.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from clarity_chat.documents.embedder import DIMENSIONS, TooLong, input_of
from clarity_chat.documents.library import Unreadable, folder, read_file, write, writing
from clarity_chat.documents.search import Loaded


class PartTooLong(Exception):
    """A part whose input passes the embedder's 8,192 tokens (decision 8)."""

    def __init__(self, document: str, part: dict[str, Any], tokens: int) -> None:
        super().__init__(document)
        self.document, self.part, self.tokens = document, part, tokens


def key_of(build_id: str, setup: str) -> str:
    return hashlib.sha256(f"{build_id}\n{setup}".encode()).hexdigest()


def split(data: bytes) -> Sequence[Sequence[float]] | None:
    """The file's vectors, read as untrusted: each finite and of length 1 within 1e-3."""
    values = memoryview(data).cast("f")
    vectors = [values[i : i + DIMENSIONS] for i in range(0, len(values), DIMENSIONS)]
    for v in vectors:
        if not all(map(math.isfinite, v)) or abs(math.sqrt(math.fsum(x * x for x in v)) - 1) > 1e-3:
            return None
    return vectors


def vectors_for(
    root: Path,
    doc: Loaded,
    embedder: Any,
    setup: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[Sequence[Sequence[float]], str]:
    """`doc`'s vectors for this setup, and the sha256 of the bytes they were read from:
    read whole through the library's reader (no links, regular files only), or built
    first, with `progress` told before the first part and after each."""
    here = folder(root, doc.name)
    path = here / f"vectors-{key_of(doc.build_id, setup)}.f32"
    size = len(doc.parts) * DIMENSIONS * 4
    try:
        data = read_file(doc.name, path, size)
        vectors = split(data) if len(data) == size else None
        if vectors is not None:
            return vectors, hashlib.sha256(data).hexdigest()
    except (Unreadable, OSError):
        pass  # not there, or not a file Clarity reads: built again
    tell = progress or (lambda name, done, total: None)
    tell(doc.name, 0, len(doc.parts))
    built = []
    for done, part in enumerate(doc.parts, 1):
        try:
            ids = embedder.ids(input_of(part["title"], part["text"]))
        except TooLong as e:
            raise PartTooLong(doc.name, part, e.tokens) from None
        built.append(embedder.vector(ids))
        tell(doc.name, done, len(doc.parts))
    data = b"".join(built)
    with writing(doc.name):
        partial = path.with_name(path.name + ".part")
        partial.unlink(missing_ok=True)  # a link left there is removed, never written through
        write(partial, data)
        os.replace(partial, path)
        for old in here.glob("vectors-*"):
            if old != path:
                old.unlink()
    vectors = split(data)
    assert vectors is not None  # each was checked as it came (Embedder.vector)
    return vectors, hashlib.sha256(data).hexdigest()
