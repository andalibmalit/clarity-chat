"""The library (v0.3.2 A1): a stub, so A1's tests fail on their assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clarity_chat.documents.reading import DocumentRefused

LIBRARY = Path("local") / "documents"


class Stale(DocumentRefused):
    def __init__(self, name: str, state: str) -> None:
        self.name, self.state = name, state
        super().__init__(f"{name} is {state}.")


def checked_library_name(name: str) -> str:
    return name


def default_name(filename: str) -> str | None:
    return None


def add(
    root: Path, data: bytes, filename: str, name: str | None = None, *args: Any, **kw: Any
) -> dict[str, Any]:
    return {"added": {"name": name, "parts": 0, "titled": 0}, "written": ["x/y"]}


def state(root: Path, name: str) -> str:
    return "ready"


def documents(root: Path) -> list[dict[str, Any]]:
    return []


def check(root: Path, name: str) -> dict[str, Any]:
    return {"name": name, "state": "ready", "problems": []}


def parts_of(root: Path, name: str) -> list[dict[str, Any]]:
    return []


def control_token_parts(
    parts: list[dict[str, Any]], special_tokens_in: Any
) -> list[dict[str, Any]]:
    return []


def rebuild(root: Path, name: str) -> dict[str, Any]:
    return {"rebuilt": {"name": name, "parts": 0, "titled": 0}, "written": ["x/y"]}


def write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
