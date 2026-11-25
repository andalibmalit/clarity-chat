"""Export for Inspect (v0.4.3): a finished batch run as one Inspect log per profile.

Written through inspect_ai's own writer, at the one version the inspect extra pins, and
imported only here, so starting Clarity never imports it. Inspect's own fields hold only
what is exactly true of Clarity's calls; everything else is in metadata, where each sample's
is exactly the plain export's row for its call (docs/design.md, "Export for Inspect")."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clarity_chat.export import (
    RECOMPUTE_NOTE,
    STEERED_NOTE,
    ConversationExport,
    run_conversations,
    steered_by_vector,
)
from clarity_chat.logs import LOG_NAME, unreadable
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session

if TYPE_CHECKING:  # imported only when an export runs
    from inspect_ai.log import EvalLog, EvalSample
    from inspect_ai.model import StopReason

Writer = Callable[..., object]  # inspect_ai's write_eval_log

INSPECT_AI = "0.3.291"  # the inspect extra's pin in pyproject.toml; a test checks they match
INSPECT_FORMAT_VERSION = 1  # the shape of what Clarity puts in Inspect's fields
INSPECT_NOTE = (
    "Clarity made these calls, not Inspect. Clarity built each prompt's token IDs itself "
    "(chat template, BOS, special tokens) and sent them with sampler settings that Inspect "
    "has no fields for, so this log's generate config is empty. Running this log again in "
    "Inspect won't reproduce these calls: its model, clarity/PROFILE, names no Inspect "
    "provider. Each sample's metadata holds what the model got (`prompt_ids`, "
    "`prompt_text`), every setting with its origin, and the facts replay compares. Each "
    "message's text is that message's part of `prompt_text`. Text the chat template adds is "
    "only in `prompt_text`: its role markers, and any text of its own, like the system "
    "message OLMo 3's template adds when the profile has none, or the dates Llama 3.1's puts "
    "in its system header. The viewer formats text as Markdown, so leading spaces and "
    "characters like `*` can change on screen; the Raw button on a sample's Messages tab "
    "shows the exact characters, and the metadata (`prompt_text`, `output_text`) holds them. "
    "The `mark` score's value is the reply's labels (\"unmarked\" when it has none); it is "
    "a label, not a measurement."
)
TAGS = ["clarity", "not-run-by-inspect"]
WHOLE_TEXT = "text mode: the template has no message roles"
# Inspect's stop_reason for each stop type; any other is "unknown". A "limit" is keyed on its
# flag, since llama-server's limit also ends a call whose context filled (`truncated`); its
# other limits, n_indent and t_max_predict_ms, Clarity turns off. Both backends have flagged
# every "limit" from the first commit; one with neither flag is "unknown".
STOP_REASONS: dict[str, StopReason] = {"eos": "stop", "word": "stop"}
LIMIT_REASONS: dict[str, StopReason] = {"length": "max_tokens", "context_full": "model_length"}

Group = tuple[str, list[ConversationExport]]  # a profile's name and its conversations


def export_run(session: Session, run_dir: Path) -> dict[str, Any]:
    """Write a finished run as RUN_DIR/inspect/N-PROFILE.eval, one log per profile in the
    run's order, replacing any there. Returns each file's path, whether it replaced one and
    its calls, and the plain export's steering and documents facts."""
    write = inspect_writer()
    log = run_dir / "calls.jsonl"
    try:
        run, conversations = run_conversations(session, run_dir)
        groups = by_profile(run, conversations, run_dir)
        logs = [eval_log(run, n, name, group) for n, (name, group) in enumerate(groups, 1)]
    except (LookupError, StopIteration, TypeError, AttributeError, ValueError) as e:
        # a log or key.json not Clarity's, or a bug
        raise ConfigError(unreadable(e, f"the records of {log}")) from e
    folder = run_dir / "inspect"
    folder.mkdir(exist_ok=True)
    files = []
    for (n, (name, _)), eval_log_ in zip(enumerate(groups, 1), logs, strict=True):
        path = folder / f"{n}-{name}.eval"
        replaced = path.exists()
        write_privately(write, eval_log_, path)
        files.append(
            {"path": str(path), "replaced": replaced, "calls": len(eval_log_.samples or [])}
        )
    rows = [row for c in conversations for row in c.rows]
    return {
        "files": files,
        "steered": steered_by_vector(rows),
        "attachments": any(row["attachments"] for row in rows),
        "found": any(row["found"] for row in rows),
    }


def inspect_writer() -> Writer:
    """inspect_ai's writer, at the pinned version (states 4 and 5)."""
    try:
        from inspect_ai.log import write_eval_log
    except ImportError as e:
        raise ConfigError(
            "Export for Inspect needs the inspect extra: run uv sync --extra inspect (add "
            "--extra hf if you use the Transformers backend), then try again."
        ) from e
    installed = version("inspect-ai")
    if installed != INSPECT_AI:
        raise ConfigError(
            f"Export for Inspect is pinned to inspect_ai {INSPECT_AI}, and {installed} is "
            "installed. Run uv sync --extra inspect to install the pinned version."
        )
    return write_eval_log


def by_profile(
    run: dict[str, Any], conversations: list[ConversationExport], run_dir: Path
) -> list[Group]:
    """The conversations by profile, in the run record's order. key.json names each
    conversation's profile as --profiles gave it, so two profile files with one name stay
    apart, and the log's order (by model) doesn't matter. The file's name is the calls'
    logged `profile.name`, so it must be one Clarity writes (state 11)."""
    key = json.loads((run_dir / "key.json").read_text(encoding="utf-8"))
    given = {
        label["conversation_id"]: label["profile"]
        for opening in key["openings"]
        for label in opening["labels"].values()
    }
    groups: dict[str, list[ConversationExport]] = {profile: [] for profile in run["profiles"]}
    for conversation in conversations:
        conversation_id = conversation.records[0]["conversation_id"]
        if not any(r["type"] == "call" for r in conversation.records):
            raise ValueError(f"conversation {conversation_id} has no call")
        if given.get(conversation_id) not in groups:
            raise ValueError(f"key.json gives conversation {conversation_id} no listed profile")
        groups[given.pop(conversation_id)].append(conversation)
    if given:
        raise ValueError(f"key.json names conversations it doesn't hold: {sorted(given)}")
    named = []
    for profile, group in groups.items():
        calls = [r for c in group for r in c.records if r["type"] == "call"]
        names = {call["profile"]["name"] for call in calls}
        if len(names) != 1 or len(calls) != sum(len(c.rows) for c in group):
            raise ValueError(f"its calls for profile {profile} don't match key.json's")
        [name] = names
        if not isinstance(name, str) or not LOG_NAME.fullmatch(name):
            raise ConfigError(
                f"{name}, a profile name in {run_dir / 'calls.jsonl'}, isn't one Clarity "
                "writes, so it can't name the export's file."
            )
        named.append((name, group))
    return named


def eval_log(run: dict[str, Any], n: int, name: str, group: list[ConversationExport]) -> EvalLog:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalResults, EvalSpec, EvalStats
    from inspect_ai.model import GenerateConfig

    model = f"clarity/{name}"
    records = [r for c in group for r in c.records]
    calls = {r["call_id"]: r for r in records if r["type"] == "call"}
    turns = {r["turn_id"]: r for r in records if r["type"] == "turn"}
    rows = [row for c in group for row in c.rows]
    chat = tomllib.loads(calls[rows[0]["call_id"]]["profile"]["toml"])["mode"] == "chat"
    samples = [sample(row, turns, chat, model, run["seeds"]) for row in rows]
    openings = list(range(1, run["n_openings"] + 1))
    return EvalLog(
        status="success",
        eval=EvalSpec(
            eval_id=stable_id(run["run_id"], n, "eval"),
            run_id=run["run_id"],
            created=run["time"],
            task="clarity-run",  # one task for every profile's log, so the grid lines up
            task_id=stable_id(run["run_id"], n, "task"),
            tags=TAGS,
            dataset=EvalDataset(name="clarity-run", samples=len(openings), sample_ids=openings),
            model=model,
            model_generate_config=GenerateConfig(),  # empty: see INSPECT_NOTE
            config=EvalConfig(epochs=len(run["seeds"])),
            # The note at the top: the viewer folds the many-keyed header, but shows a string.
            metadata={
                "inspect_note": INSPECT_NOTE,
                "clarity": run_header(run, [c.header for c in group], rows),
            },
        ),
        samples=samples,
        results=EvalResults(total_samples=len(rows), completed_samples=len(rows), scores=[]),
        # The calls' own times: an empty EvalStats shows as 1969 and a NaN duration (G0, V7).
        stats=EvalStats(started_at=run["time"], completed_at=calls[rows[-1]["call_id"]]["time"]),
    )


def sample(
    row: dict[str, Any], turns: dict[str, dict[str, Any]], chat: bool, model: str, seeds: list
) -> EvalSample:
    """A call as a sample: its prompt's messages as slices of `prompt_text`, its reply, its
    `mark` score, and its plain-export row, flat, as the metadata."""
    from inspect_ai.log import EvalSample
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageUser,
        ModelOutput,
    )
    from inspect_ai.scorer import Score

    roles = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }
    call_id, text, spans = row["call_id"], row["prompt_text"], row["message_spans"]
    messages: list[Any] = []
    if chat and isinstance(spans, list):
        for span in (s for s in spans if s["kind"] == "message"):
            start, end = span["chars"]
            kind = roles[span["role"]]
            messages.append(
                kind(
                    id=f"{call_id}-{len(messages)}",
                    content=text[start:end],
                    metadata={"clarity": span},
                )
            )
    else:
        why = (
            WHOLE_TEXT if not chat else f"the message spans are unavailable: {spans['unavailable']}"
        )
        messages.append(
            ChatMessageUser(
                id=f"{call_id}-0", content=text, metadata={"clarity": {"whole_prompt": why}}
            )
        )
    # The input is the reply's own message, the opening as the model got it, in both modes
    # (the turn's text when the spans are unavailable): one opening's samples show the same
    # input unless search added parts to it.
    asked = (
        [s["chars"] for s in spans if s.get("turn_id") == row["parent_id"]]
        if isinstance(spans, list)
        else []
    )
    reply = ChatMessageAssistant(
        id=f"{call_id}-{len(messages)}", content=row["output_text"], model=model
    )
    reason = STOP_REASONS.get(row["stop_type"], "unknown")
    if row["stop_type"] == "limit":
        reason = next((LIMIT_REASONS[f] for f in row["flags"] if f in LIMIT_REASONS), "unknown")
    labels = sorted(m["label"] for m in row["marks"])
    return EvalSample(
        id=row["run"]["opening"],
        epoch=seeds.index(row["settings"]["seed"]["value"]) + 1,
        input=text[asked[0][0] : asked[0][1]] if asked else turns[row["parent_id"]]["text"],
        target="",
        messages=[*messages, reply],
        output=ModelOutput(
            model=model, choices=[ChatCompletionChoice(message=reply, stop_reason=reason)]
        ),
        scores={
            "mark": Score(
                value=", ".join(labels) or "unmarked",
                answer=row["output_text"],
                explanation="\n\n".join(note["text"] for note in row["notes"]) or None,
                metadata={"marks": row["marks"], "notes": row["notes"]},
            )
        },
        metadata=row,
        uuid=call_id,
    )


def run_header(
    run: dict[str, Any], headers: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """The plain export's header for a profile's conversations together, with the run
    record in place of a conversation ID."""
    groups: dict[str, list[str]] = {}
    for header in headers:
        for group in header["conditions"]:
            groups.setdefault(json.dumps(group["conditions"], sort_keys=True), []).extend(
                group["call_ids"]
            )
    counts = ("calls", "failed_calls_left_out", "calls_without_a_turn_left_out")
    return {
        **{k: v for k, v in headers[0].items() if k != "conversation_id"},
        "run": run,
        **{count: sum(h[count] for h in headers) for count in counts},
        "conditions": [{"conditions": json.loads(k), "call_ids": ids} for k, ids in groups.items()],
        "note": RECOMPUTE_NOTE
        + (STEERED_NOTE if any(r["steering"] != "none" for r in rows) else ""),
        "inspect_format_version": INSPECT_FORMAT_VERSION,
    }


def stable_id(run_id: str, n: int, kind: str) -> str:
    """An ID that two exports of one run share: 22 hex digits of SHA-256 of the run's ID,
    the profile's place and what it names."""
    return hashlib.sha256(f"{run_id} {n} {kind}".encode()).hexdigest()[:22]


def write_privately(write: Writer, log: EvalLog, path: Path) -> None:
    """Inspect's writer makes its own temporary file beside its target, readable by others
    (0666 less the umask), so it writes inside a 0700 folder; the file is made 0600 and moved
    into place, and the folder is removed, also when the write fails."""
    folder = Path(tempfile.mkdtemp(dir=path.parent))
    try:
        inside = folder / path.name
        write(log, str(inside))
        os.chmod(inside, 0o600)
        os.replace(inside, path)
    except OSError as e:
        raise ConfigError(f"Can't write the export to {path}: {e.strerror or e}.") from e
    finally:
        shutil.rmtree(folder, ignore_errors=True)
