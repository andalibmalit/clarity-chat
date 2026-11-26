"""Making a steering vector from a file of contrasting prompt pairs (`clarity vector make`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pair:
    """One pair: the two sides differ only in their system messages here. `entries` names
    where the system messages, the message and the opening came from, as the file says."""

    positive_system: str
    negative_system: str
    message: str
    opening: str
    entries: tuple[str, str, str]


@dataclass(frozen=True)
class Pairs:
    path: Path
    stem: str
    sha256: str
    text: str
    description: str
    author: str
    form: str
    counts: dict[str, int]
    pairs: list[Pair]


def read_pairs(path: Path) -> Pairs:
    return Pairs(path, "", "", "", "", "", "", {}, [])
