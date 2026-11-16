"""Word search (v0.3.2 A3): a stub, so A3's tests fail on their assertions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarity_chat.documents.library import parts_of, read_manifest

CONSIDERED = 8
COMMON_WORDS: frozenset[str] = frozenset()
_indexes: dict = {}


def words_of(message: str) -> tuple[list[str], list[str]]:
    return [], []


@dataclass(frozen=True)
class Loaded:
    name: str
    build_id: str
    original_sha256: str
    parts: list[dict[str, Any]]


def load(root: Path, name: str) -> Loaded:
    manifest = read_manifest(root, name)
    return Loaded(name, "", "", parts_of(root, name))


def search(docs: list[Loaded], message: str) -> dict[str, Any]:
    return {"words": [], "ignored": [], "matches": [], "engine": {}, "documents": []}
