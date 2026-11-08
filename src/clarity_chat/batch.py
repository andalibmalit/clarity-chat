"""Batch runs for blind tests: each profile's first reply to each opening.

`clarity run` writes everything to a new `runs/<run-id>/` folder, never to `logs/`, so a
later transcript export can't pick up held-out openings:
- `calls.jsonl`: a `run` record, then every conversation's records (one conversation per
  opening, profile and seed), and each server's start record and output log.
- `key.json`: for each opening, which label is which profile, seed and call.
- `blind.md`: each opening, then its replies shuffled and labelled A, B, …, each followed
  by empty `Rating:` and `Note:` lines. Nothing else.

Every profile and opening is checked before a model loads. Any error, a failed after-call
check or Ctrl-C stops the run before `key.json` and `blind.md` are written.
"""

from __future__ import annotations

import json
import random
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError
from clarity_chat.files import read_text, sha256_text, write_text_atomic
from clarity_chat.logs import LogFile, client_git, machine
from clarity_chat.models import portable_path
from clarity_chat.profiles import SEED_MAX, ConfigError, Setting
from clarity_chat.render import build_prompt
from clarity_chat.session import ContextError, Prepared, Session, fits_context

FIRST_SEED = 333  # the user's choice: a run's seeds are 333, 334, …
SPLIT_RULE = (
    'Openings are separated by lines that are exactly "---". CRLF line endings become LF '
    "first, so the editor's line endings never change the token IDs."
)
TRIM_RULE = (
    "Each opening is str.strip()ped before it is sent. Replies are str.strip()ped in "
    "blind.md only; calls.jsonl keeps them as generated."
)
SHUFFLE_RULE = (
    "random.Random(shuffle_seed).shuffle, applied to each opening's replies in turn; before "
    "shuffling they are in profile order as given, then seed order."
)


class RunError(Exception):
    """The run stopped; `key.json` and `blind.md` were not written."""


def split_openings(text: str) -> list[str]:
    """Openings from a file's text, per SPLIT_RULE and TRIM_RULE. An empty one is an error."""
    sections: list[list[str]] = [[]]
    for line in text.replace("\r\n", "\n").split("\n"):
        if line == "---":
            sections.append([])
        else:
            sections[-1].append(line)
    openings = ["\n".join(lines).strip() for lines in sections]
    empty = [i for i, opening in enumerate(openings, 1) if not opening]
    if empty:
        raise ConfigError(
            f"Opening(s) {empty} of {len(openings)} are empty. Each '---' line separates two "
            "openings, so the file can't start or end with one, or have two in a row."
        )
    return openings


def label(index: int) -> str:
    """A, B, …, Z, AA, AB, … for index 0, 1, …"""
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


@dataclass
class Reply:
    profile: str  # as given on the command line
    seed: int
    call_id: str
    conversation_id: str
    text: str  # untrimmed, as generated


def check_run(
    session: Session, profile_names: list[str], openings: list[str], n_seeds: int
) -> list[Prepared]:
    """Everything that can be checked before a model loads: the profiles load, they share
    max_tokens, and every opening fits every profile's context."""
    if n_seeds < 1 or FIRST_SEED + n_seeds - 1 > SEED_MAX:
        raise ConfigError(f"--seeds must be from 1 to {SEED_MAX - FIRST_SEED + 1}.")
    if len(set(profile_names)) != len(profile_names):
        raise ConfigError(f"A profile is listed twice: {profile_names}.")
    prepared = [session.prepare(name) for name in profile_names]
    max_tokens = {p.profile.name: p.profile.settings["max_tokens"].value for p in prepared}
    if len(set(max_tokens.values())) > 1:
        raise ConfigError(
            f"The profiles' max_tokens differ ({max_tokens}); reply length would give the "
            "setup away. Give them the same max_tokens."
        )
    n_ctx = session.backend.n_ctx
    for p in prepared:
        limit = p.profile.settings["max_tokens"].value
        for i, opening in enumerate(openings, 1):
            prompt = build_prompt(p.tokenizer, p.profile, [("user", opening)], p.template_tokenizer)
            if not fits_context(len(prompt.ids), limit, n_ctx):
                raise ConfigError(
                    f"Opening {i} with profile {p.profile.name} is {len(prompt.ids)} tokens; "
                    f"with max_tokens {limit} it doesn't fit the context of {n_ctx} tokens."
                )
    return prepared


def run_batch(
    session: Session,
    run_id: str,
    run_dir: Path,
    openings_path: Path,
    profile_names: list[str],
    n_seeds: int = 1,
    write: Callable[[str], None] = print,
) -> Path:
    """Run every (opening, profile, seed) call, then write key.json and blind.md.
    Returns the path of blind.md. The terminal shows progress and warnings, never replies."""
    text = read_text(openings_path)
    openings = split_openings(text)
    prepared = check_run(session, profile_names, openings, n_seeds)
    seeds = [FIRST_SEED + k for k in range(n_seeds)]
    log = LogFile(run_dir / "calls.jsonl")
    log.append(
        {
            "type": "run",
            "run_id": run_id,
            "openings_file": portable_path(str(openings_path.resolve()), session.root),
            "openings_sha256": sha256_text(text),
            "n_openings": len(openings),
            "profiles": profile_names,
            "seeds": seeds,
            "split_rule": SPLIT_RULE,
            "trim_rule": TRIM_RULE,
            "client": client_git(),
            "machine": machine(),
        }
    )
    # Group the calls by model, so each model loads once.
    model_of = {name: p.model.name for name, p in zip(profile_names, prepared, strict=True)}
    models = list(dict.fromkeys(model_of.values()))
    order = sorted(profile_names, key=lambda name: models.index(model_of[name]))  # stable
    total = len(openings) * len(profile_names) * n_seeds
    replies: dict[tuple[int, str, int], Reply] = {}
    shown: set[str] = set()
    server_id = None

    def warn(message: str) -> None:
        if message not in shown:  # each warning once; they never contain reply text
            shown.add(message)
            write(f"\n! {message}\n")

    write(f"Run {run_id}: {total} calls.\n")
    for name in order:
        for i, opening in enumerate(openings, 1):
            for seed in seeds:
                where = f"opening {i}, profile {name}, seed {seed}"
                try:
                    conversation = session.new_conversation(
                        name, log=log, run={"run_id": run_id, "opening": i}
                    )
                    record = session.backend.server_record or {}
                    if record.get("server_id") != server_id:
                        server_id = record.get("server_id")
                        for warning in record.get("warnings", []):
                            warn(warning)
                    outcome = session.say(opening, seed=Setting(seed, "run"))
                except (ContextError, BackendError, ConfigError) as e:
                    raise RunError(f"{where.capitalize()}: {e}") from e
                call = outcome.record
                failed = [f for f in call["flags"] if f.startswith("check_failed:")]
                if call["stop_type"] == "cancelled":
                    raise RunError(f"Cancelled during {where}.")
                if failed:
                    raise RunError(f"Call {call['call_id']} ({where}) failed {failed}.")
                for warning in outcome.warnings:
                    warn(warning)
                replies[i, name, seed] = Reply(
                    name, seed, call["call_id"], conversation.conversation_id, call["output_text"]
                )
                write(f"\r{len(replies)}/{total}")
    write("\n")

    shuffle_seed = secrets.randbelow(2**32)
    rng = random.Random(shuffle_seed)
    blind: list[str] = []
    key_openings: list[dict[str, Any]] = []
    for i, opening in enumerate(openings, 1):
        group = [replies[i, name, seed] for name in profile_names for seed in seeds]
        rng.shuffle(group)
        labels = {label(k): reply for k, reply in enumerate(group)}
        key_openings.append(
            {
                "opening": i,
                "labels": {
                    tag: {
                        "profile": r.profile,
                        "seed": r.seed,
                        "call_id": r.call_id,
                        "conversation_id": r.conversation_id,
                    }
                    for tag, r in labels.items()
                },
            }
        )
        blind.append(f"# Opening {i}\n\n{opening}\n")
        for tag, r in labels.items():
            blind.append(f"## {tag}\n\n{r.text.strip()}\n\nRating:\nNote:\n")
    key = {
        "run_id": run_id,
        "openings_file": portable_path(str(openings_path.resolve()), session.root),
        "openings_sha256": sha256_text(text),
        "profiles": profile_names,
        "seeds": seeds,
        "split_rule": SPLIT_RULE,
        "trim_rule": TRIM_RULE,
        "shuffle_seed": shuffle_seed,
        "shuffle_rule": SHUFFLE_RULE,
        "calls": "calls.jsonl",
        "openings": key_openings,
    }
    write_text_atomic(run_dir / "key.json", json.dumps(key, indent=2, ensure_ascii=False) + "\n")
    blind_path = run_dir / "blind.md"
    write_text_atomic(blind_path, "\n".join(blind))
    return blind_path
