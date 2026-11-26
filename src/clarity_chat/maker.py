"""Making a steering vector from a file of contrasting prompt pairs (`clarity vector make`)."""

from __future__ import annotations

import hashlib
import itertools
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from clarity_chat.profiles import ConfigError, shown

if TYPE_CHECKING:
    from clarity_chat.session import Prepared, Session

MAX_BYTES = 4 * 1024 * 1024
MIN_PAIRS, MAX_PAIRS = 2, 1024  # one pair has nothing to agree with; a crossed file multiplies
STEM = re.compile(r"[A-Za-z0-9_-]+")  # the vector's file name starts with it
SIDES = ("positive", "negative")
NEEDED = {
    "description": "one line saying what the two sides contrast",
    "author": 'who or what wrote the pairs, as you say it ("me" is fine)',
    "messages": "the user messages, each read with every system pair",
    "systems": "one [[systems]] table per system pair, with positive and negative",
}


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
    path: Path  # as given
    stem: str
    sha256: str  # of the bytes read
    text: str
    description: str
    author: str
    form: str
    counts: dict[str, int]
    pairs: list[Pair]


def read_pairs(path: Path) -> Pairs:
    """Read a pairs file and expand it, or refuse it with a sentence naming the entry."""
    name = shown(str(path))
    if not STEM.fullmatch(path.stem):
        raise ConfigError(
            "Name the pairs file with letters, digits, - and _ only: its name becomes the vector's."
        )
    try:
        with path.open("rb") as f:
            data = f.read(MAX_BYTES + 1)
    except FileNotFoundError:
        raise ConfigError(f"The pairs file {name} doesn't exist.") from None
    except (OSError, ValueError) as e:  # ValueError: a NUL in the path
        raise ConfigError(f"Can't read the pairs file {name}: {e}") from None
    if len(data) > MAX_BYTES:
        raise ConfigError(f"{name} is over 4 MB; a pairs file is smaller.")
    try:
        text = data.decode("utf-8")
        table = tomllib.loads(text)
    except ValueError as e:  # not UTF-8, not TOML, or an integer of over 4,300 digits
        raise ConfigError(f"{name} isn't a TOML file: {e}.") from None

    def unknown(table: dict, allowed: tuple[str, ...], where: str) -> None:
        if extra := sorted(set(table) - set(allowed)):
            raise ConfigError(
                f"{name}: unknown keys {extra} in {where}. Allowed there: {', '.join(allowed)}."
            )

    def needed(key: str, where: str = "") -> NoReturn:
        what = NEEDED.get(key, "that side's system message")
        raise ConfigError(f"{name} needs {key}{where}: {what}.")

    def strings(key: str, default: list[str] | None = None) -> list[str]:
        value = table.get(key, default)
        if value is None:
            needed(key)
        if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
            raise ConfigError(f"{name}: {key} must be a list of strings.")
        return value

    unknown(table, ("author", "description", "messages", "openings", "systems"), "the top level")
    systems = table.get("systems")
    if systems is None:
        needed("systems")
    if not isinstance(systems, list):
        raise ConfigError(f"{name}: systems must be a list of [[systems]] tables.")
    for i, entry in enumerate(systems, 1):  # first, so a key that landed in one is named there
        where = f"system pair {i}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{name}: {where} must be a table with positive and negative.")
        unknown(entry, SIDES, where)
        for side in SIDES:
            if side not in entry:
                needed(side, f" in {where}")
            if not isinstance(entry[side], str):
                raise ConfigError(f"{name}: {side} in {where} must be a string.")
        if entry["positive"] == entry["negative"]:
            raise ConfigError(f"{name}: {where} has the same text on both sides.")

    def header(key: str) -> str:
        value = table.get(key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{name}: {key} must be a string.")
        if value is None or not value.strip():
            needed(key)
        return value

    description, author = header("description"), header("author")
    messages, openings = strings("messages"), strings("openings", [""])
    for i, message in enumerate(messages, 1):
        if not message.strip():
            raise ConfigError(f"{name}: message {i} is empty; a message needs text.")
    # Counted before expanding: a short crossed file can ask for millions.
    n = len(systems) * len(messages) * len(openings)
    if n < MIN_PAIRS:
        raise ConfigError(f"{name} makes {n} pair{'' if n == 1 else 's'}; make needs at least 2.")
    if n > MAX_PAIRS:
        raise ConfigError(f"{name} makes {n:,} pairs; make takes at most 1,024.")
    sides = [(entry["positive"], entry["negative"]) for entry in systems]
    pairs = [
        Pair(*sides[s - 1], message, opening, (f"system pair {s}", f"message {m}", f"opening {o}"))
        for s, (m, message), (o, opening) in itertools.product(
            range(1, len(sides) + 1), enumerate(messages, 1), enumerate(openings, 1)
        )
    ]
    return Pairs(
        path,
        path.stem,
        hashlib.sha256(data).hexdigest(),
        text,
        description,
        author,
        "crossed",
        {"systems": len(systems), "messages": len(messages), "openings": len(openings)},
        pairs,
    )


@dataclass(frozen=True)
class Rendered:
    pair: int  # from 1, in the order read_pairs expands them
    side: str
    text: str
    ids: list[int]


def prepare(session: Session, name: str) -> Prepared:
    return session.prepare(name)


def render_pairs(prepared: Prepared, pairs: Pairs, ctx_size: int) -> list[Rendered]:
    return [Rendered(0, "", "", [])]
