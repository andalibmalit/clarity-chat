"""File helpers: verbatim reads, atomic writes, hashes."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path


def read_text(path: Path) -> str:
    """Read UTF-8 text verbatim: no newline translation, so a trailing newline or CRLF stays."""
    with path.open(encoding="utf-8", newline="") as f:
        return f.read()


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text verbatim via a temp file, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: a reader can briefly lock the target
        time.sleep(0.2)
        os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
