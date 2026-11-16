"""Parts, version 1 (v0.3.2 decision 7): a stub, so A2's tests fail on their assertions."""

from __future__ import annotations

from typing import Any

CHUNKER = {"name": "clarity-parts", "version": 1, "max_words": 250, "title_chars": 120}


def cut_parts(text: str, name: str, page_starts: list[int] | None = None) -> list[dict]:
    return []


def parts_jsonl(parts: list[dict[str, Any]]) -> bytes:
    return b""


def build_id(jsonl: bytes) -> str:
    return ""


def closing_tag_parts(parts: list[dict[str, Any]]) -> list[int]:
    return []
