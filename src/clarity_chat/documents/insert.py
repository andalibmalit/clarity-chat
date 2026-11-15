"""The insertion path (A2's stub)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clarity_chat.documents.extract import Document

FRAMING = ""
SENTENCES: dict[str, str] = {"stub": "Not built yet."}


@dataclass(frozen=True)
class Attached:
    name: str
    text: str
    marker: str | None


@dataclass(frozen=True)
class Room:
    tokens: Callable[[str], int]
    room_left: Callable[[int], int]
    max_tokens: int


@dataclass(frozen=True)
class Fit:
    tokens: int
    free: int
    left: int
    fits: bool


def framing_of(profile: Any) -> tuple[str, str]:
    return "", "default"


def piece(
    doc: Document, profile: Any, unit: str | None = None, first: int = 0, last: int = 0
) -> Attached:
    return Attached(doc.name, doc.text, None)


def content(message: str, attached: list[Attached], framing: str) -> str:
    return message


def scan(text: str, special_tokens: Callable[[str], list[str]]) -> dict[str, Any]:
    return {}


def free(room: Room, pending: list[Attached], framing: str) -> int:
    return 0


def fit(room: Room, pending: list[Attached], new: Attached, framing: str) -> Fit:
    return Fit(0, 0, 0, True)


def start_that_fits(
    room: Room, pending: list[Attached], framing: str, doc: Document, profile: Any, keep: int = 0
) -> tuple[int, Fit] | None:
    return None
