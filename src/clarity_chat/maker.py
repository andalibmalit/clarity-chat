"""Making a steering vector from a file of contrasting prompt pairs (`clarity vector make`)."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import re
import tomllib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from clarity_chat.backends import BackendError
from clarity_chat.profiles import ConfigError, Profile, load_profile, shown
from clarity_chat.render import render_chat, tokenize

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


@dataclasses.dataclass(frozen=True)
class Pair:
    """One pair: the two sides differ only in their system messages here. `entries` names
    where the system messages, the message and the opening came from, as the file says."""

    positive_system: str
    negative_system: str
    message: str
    opening: str
    entries: tuple[str, str, str]


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class Rendered:
    pair: int  # from 1, in the order read_pairs expands them
    side: str
    text: str
    ids: list[int]


def prepare(session: Session, name: str) -> Prepared:
    """The profile, prepared, once its backend and mode are checked: before any weights are
    looked for, so a llama.cpp profile hears why it can't be used and not a fetch hint."""

    def usable(profile: Profile) -> None:
        if profile.backend != "transformers":
            raise ConfigError(
                f"Profile {profile.name} runs on llama.cpp; make reads the model on the "
                'Transformers backend. Use a profile with backend = "transformers".'
            )
        if profile.mode != "chat":
            raise ConfigError(
                f"Profile {profile.name} is in text mode; make needs a chat-mode profile, "
                "whose system message each pair replaces."
            )

    usable(load_profile(session.root, name, session.defaults))
    prepared = session.prepare(name)
    usable(prepared.profile)  # read again by prepare: the file may have changed since
    return prepared


def render_pairs(prepared: Prepared, pairs: Pairs, ctx_size: int) -> list[Rendered]:
    """Each side of each pair as this profile would send its message, with the pair's
    system message in place of the profile's and the opening after the generation prompt."""
    name, tokenizer = shown(str(pairs.path)), prepared.tokenizer
    rendered = []
    for i, pair in enumerate(pairs.pairs, 1):
        system, message, opening = pair.entries
        sides = {"positive": pair.positive_system, "negative": pair.negative_system}
        fields = {f"{system}'s {side} system message": text for side, text in sides.items()}
        for where, text in {**fields, message: pair.message, opening: pair.opening}.items():
            if found := tokenizer.special_tokens_in(text):
                raise ConfigError(
                    f"{name}: {where} contains {found[0]}, which the tokenizer reads as a "
                    "control token, not text."
                )
        for side, system_text in sides.items():
            sided = dataclasses.replace(prepared.profile, system=system_text)
            text = render_chat(prepared.template_tokenizer, sided, [("user", pair.message)])
            text += pair.opening
            ids, _ = tokenize(tokenizer, text, "chat")
            if len(ids) > ctx_size:
                raise ConfigError(
                    f"Pair {i}'s {side} prompt is {len(ids)} tokens, more than this "
                    f"profile's context size, {ctx_size}."
                )
            rendered.append(Rendered(i, side, text, ids))
    return rendered


@dataclasses.dataclass(frozen=True)
class Block:
    """What make read at one block: the vector's direction there and the readout's numbers."""

    direction: Any  # float32: the mean difference, positive minus negative
    size: float  # the mean length of the block's output at the last token, over every prompt
    gap: float  # the mean difference's length
    agreement: float  # each pair's difference's cosine with it, averaged over the pairs


@contextmanager
def reading(model: Any) -> Generator[dict[int, Any]]:
    """Read-only hooks on blocks 1 to n-1, the blocks steering can add to: after each forward
    pass the dict holds each block's output at the last token, in float32 on the CPU (the
    tensor `steering.applied` adds to). The hooks come off however the block ends."""
    import torch

    rows: dict[int, Any] = {}

    def hook(b: int) -> Callable[..., None]:
        def read(module: Any, args: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise BackendError(f"Block {b} returned {type(output).__name__}, not a tensor.")
            rows[b] = output[0, -1].detach().to(device="cpu", dtype=torch.float32).numpy()

        return read

    blocks = model.model.layers
    handles = []
    try:
        for b in range(1, len(blocks)):
            handles.append(blocks[b].register_forward_hook(hook(b)))
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def read_blocks(
    model: Any, prompts: list[Rendered], on_progress: Callable[[int, int], None]
) -> dict[int, Block]:
    """Run each prompt alone and keep running sums per block (no per-pair data): the
    differences, the differences at length 1, and the activations' lengths. `prompts` are
    render_pairs', each pair's positive side then its negative."""
    import numpy as np
    import torch

    total, n = len(prompts), len(prompts) // 2
    diffs: dict[int, Any] = {}
    units: dict[int, Any] = {}
    lengths: dict[int, float] = {}
    positive: dict[int, Any] = {}
    with reading(model) as rows, torch.no_grad():
        for k, prompt in enumerate(prompts, 1):
            model.model(input_ids=torch.tensor([prompt.ids], device=model.device), use_cache=False)
            for b, row in rows.items():
                if not np.isfinite(row).all():
                    raise ConfigError(
                        f"Block {b}'s output for pair {prompt.pair}'s {prompt.side} prompt "
                        "isn't finite."
                    )
                row = row.astype(np.float64)
                lengths[b] = lengths.get(b, 0.0) + float(np.linalg.norm(row))
                if prompt.side == "positive":
                    positive[b] = row
                    continue
                d = positive[b] - row
                length = np.linalg.norm(d)
                if length == 0:
                    raise ConfigError(
                        f"Pair {prompt.pair}'s two prompts give the same output at block {b}, "
                        "so it has no direction to add."
                    )
                diffs[b] = diffs.get(b, 0) + d
                units[b] = units.get(b, 0) + d / length
            if 10 * k // total > 10 * (k - 1) // total:
                on_progress(k, total)
    blocks = {}
    for b in sorted(diffs):
        mean = diffs[b] / n
        gap = float(np.linalg.norm(mean))
        if gap == 0:
            raise ConfigError(
                f"At block {b} the pairs' differences add up to zero, so there's no direction "
                "to add; check for a pair written twice with its sides swapped."
            )
        agreement = float(units[b] @ mean) / (n * gap)
        blocks[b] = Block(mean.astype(np.float32), lengths[b] / total, gap, agreement)
    return blocks
