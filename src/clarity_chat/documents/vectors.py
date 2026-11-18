"""The vectors a meaning search compares (v0.4.1 A3): a bare stub, so its tests fail."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from clarity_chat.documents.search import Loaded


class PartTooLong(Exception):
    def __init__(self, document: str, part: dict[str, Any], tokens: int) -> None:
        super().__init__(document)
        self.document, self.part, self.tokens = document, part, tokens


def key_of(build_id: str, setup: str) -> str:
    return ""


def vectors_for(
    root: Path,
    doc: Loaded,
    embedder: Any,
    setup: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Sequence[float]], str]:
    return [], ""
