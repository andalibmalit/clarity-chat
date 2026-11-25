"""Export for Inspect (v0.4.3): a finished batch run as one Inspect log per profile, and a
chat as one Inspect log.

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
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from clarity_chat.export import (
    RECOMPUTE_NOTE,
    STEERED_NOTE,
    ConversationExport,
    conversation_rows,
    export_result,
    require_log,
    run_conversations,
)
from clarity_chat.logs import LOG_NAME, chat_summary, read_records, unreadable
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session

if TYPE_CHECKING:  # imported only when an export runs
    from inspect_ai.log import EvalLog, EvalSample
    from inspect_ai.model import StopReason

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


def export_run(session: Session, run_dir: Path) -> dict[str, Any]:
    """Write a finished run as RUN_DIR/inspect/N-PROFILE.eval, one log per profile in the
    run's order, replacing any there. Returns each file's path, whether it replaced one and
    its calls, and whether any call had documents attached or found."""
    write = inspect_writer()
    log = run_dir / "calls.jsonl"
    try:
        run, conversations = run_conversations(session, run_dir)
        groups = by_profile(run, conversations, run_dir)
        logs = [run_log(run, n, name, group) for n, (name, group) in enumerate(groups, 1)]
    except (LookupError, StopIteration, TypeError, AttributeError, ValueError) as e:
        # a log or key.json not Clarity's, or a bug
        raise ConfigError(unreadable(e, f"the records of {log} and its key.json")) from e
    folder = run_dir / "inspect"
    if folder.is_symlink():  # a run folder from someone else could carry one
        raise ConfigError(
            f"{folder} is a link, and the export writes only into the run's own folder. "
            "Remove the link, then export again."
        )
    paths = [folder / f"{n}-{name}.eval" for n, (name, _) in enumerate(groups, 1)]
    files = [
        {"path": str(path), "replaced": path.exists(), "calls": len(log_.samples or [])}
        for path, log_ in zip(paths, logs, strict=True)
    ]
    write_privately(write, list(zip(logs, paths, strict=True)), folder)
    rows = [row for c in conversations for row in c.rows]
    return {
        "files": files,
        "attachments": any(row["attachments"] for row in rows),
        "found": any(row["found"] for row in rows),
    }


def export_chat(session: Session, log_path: Path) -> dict[str, Any]:
    """Write a conversation as logs/exports/ID.eval, ID the log file's name, replacing any
    there. Each reply is its own sample, `id` TURN-N (its message's turn and its place among
    that message's replies) and `epoch` 1: the viewer reads uneven epochs as missing replies
    (G0, V5). Returns what the plain export returns."""
    write = inspect_writer()
    require_log(log_path)
    # Named by the log's file, as the plain export is: it must be a name Clarity writes.
    if not LOG_NAME.fullmatch(log_path.stem):
        raise ConfigError(
            f"{log_path.name} isn't a log name Clarity writes, so it can't name the export."
        )
    try:
        records = list(read_records(log_path))
        header, rows = conversation_rows(session, records, str(log_path))
        log = chat_log(records, header, rows, chat_summary(log_path)["title"])
    except (LookupError, TypeError, AttributeError, ValueError) as e:  # not Clarity's, or a bug
        raise ConfigError(unreadable(e, f"the records of {log_path}")) from e
    folder = session.logs_dir / "exports"
    path = folder / f"{log_path.stem}.eval"
    replaced = path.exists()
    write_privately(write, [(log, path)], folder)
    return export_result(path, replaced, header, rows, "inspect")


def inspect_writer() -> Callable[..., object]:  # write_eval_log
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
) -> list[tuple[str, list[ConversationExport]]]:
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
                f"{name!r}, a profile name in {run_dir / 'calls.jsonl'}, isn't one Clarity "
                "writes, so it can't name the export's file."
            )
        # A profile's name is its file's stem, and `profile` names the file as --profiles
        # gave it: a path to a .toml, or a name that profiles/ adds .toml to. So a key.json
        # whose profiles are swapped wholesale, which the names alone don't catch, is.
        if name not in {PureWindowsPath(p).stem for p in (profile, f"{profile}.toml")}:
            raise ValueError(f"key.json gives profile {profile} the calls of profile {name}")
        named.append((name, group))
    return named


def run_log(run: dict[str, Any], n: int, name: str, group: list[ConversationExport]) -> EvalLog:
    """A profile's calls in a run: each opening a sample, each seed an epoch."""
    model = f"clarity/{name}"
    records = [r for c in group for r in c.records]
    calls = {r["call_id"]: r for r in records if r["type"] == "call"}
    turns = {r["turn_id"]: r for r in records if r["type"] == "turn"}
    rows = [row for c in group for row in c.rows]
    seeds = run["seeds"]
    samples = [
        sample(
            row,
            turns,
            calls[row["call_id"]],
            model,
            row["run"]["opening"],
            seeds.index(row["settings"]["seed"]["value"]) + 1,
        )
        for row in rows
    ]
    taken: set[tuple[Any, int]] = set()
    for s in samples:  # Inspect keeps one sample per (id, epoch): another would be lost
        if (s.id, s.epoch) in taken:
            raise ValueError(
                f"two of {name}'s calls have opening {s.id} and seed {seeds[s.epoch - 1]}"
            )
        taken.add((s.id, s.epoch))
    return eval_log(
        ids=(run["run_id"], n),
        task="clarity-run",  # one task for every profile's log, so the grid lines up
        created=run["time"],
        dataset=("clarity-run", list(range(1, run["n_openings"] + 1))),
        model=model,
        epochs=len(seeds),
        header=run_header(run, [c.header for c in group], rows),
        samples=samples,
        times=(run["time"], calls[rows[-1]["call_id"]]["time"]),
    )


def chat_log(
    records: list[dict[str, Any]],
    header: dict[str, Any],
    rows: list[dict[str, Any]],
    title: str | None,
) -> EvalLog:
    """A conversation's calls: each reply its own sample, TURN-N, epoch 1."""
    conversation = records[0]  # rebuild refuses a log that starts with anything else
    model = f"clarity/{conversation['profile']['name']}"
    calls = {r["call_id"]: r for r in records if r["type"] == "call"}
    turns = {r["turn_id"]: r for r in records if r["type"] == "turn"}
    replies: dict[str, int] = {}  # each message's replies so far
    samples = []
    for row in rows:
        place = replies[row["parent_id"]] = replies.get(row["parent_id"], 0) + 1
        sample_id = f"{row['parent_id']}-{place}"
        samples.append(sample(row, turns, calls[row["call_id"]], model, sample_id, 1))
    # The calls' own times; a chat with no reply has only its start.
    times = [calls[row["call_id"]]["time"] for row in rows] or [conversation["time"]]
    return eval_log(
        ids=(header["conversation_id"], 1),
        task="clarity-chat",
        created=conversation["time"],
        dataset=(title or header["conversation_id"], [s.id for s in samples]),
        model=model,
        epochs=1,
        header={**header, "inspect_format_version": INSPECT_FORMAT_VERSION},
        samples=samples,
        times=(times[0], times[-1]),
    )


def eval_log(
    *,
    ids: tuple[str, int],
    task: str,
    created: str,
    dataset: tuple[str, list[Any]],
    model: str,
    epochs: int,
    header: dict[str, Any],
    samples: list[EvalSample],
    times: tuple[str, str],
) -> EvalLog:
    """An Inspect log whose identity comes from logged IDs (`ids`: the run's or
    conversation's, and the log's place), so two exports of one log read back equal."""
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalResults, EvalSpec, EvalStats
    from inspect_ai.model import GenerateConfig

    name, sample_ids = dataset
    return EvalLog(
        status="success",
        eval=EvalSpec(
            eval_id=stable_id(*ids, "eval"),
            run_id=ids[0],
            created=created,
            task=task,
            task_id=stable_id(*ids, "task"),
            tags=TAGS,
            dataset=EvalDataset(name=name, samples=len(sample_ids), sample_ids=sample_ids),
            model=model,
            model_generate_config=GenerateConfig(),  # empty: see INSPECT_NOTE
            config=EvalConfig(epochs=epochs),
            # The note at the top: the viewer folds the many-keyed header, but shows a string.
            metadata={"inspect_note": INSPECT_NOTE, "clarity": header},
        ),
        samples=samples,
        results=EvalResults(total_samples=len(samples), completed_samples=len(samples), scores=[]),
        # The calls' own times: an empty EvalStats shows as 1969 and a NaN duration (G0, V7).
        stats=EvalStats(started_at=times[0], completed_at=times[1]),
    )


def sample(
    row: dict[str, Any],
    turns: dict[str, dict[str, Any]],
    call: dict[str, Any],
    model: str,
    sample_id: int | str,
    epoch: int,
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
    chat = tomllib.loads(call["profile"]["toml"])["mode"] == "chat"
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
    # (the turn's text when the spans are unavailable): a run's samples of one opening show
    # the same input unless search added parts to it.
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
        id=sample_id,
        epoch=epoch,
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


def write_privately(
    write: Callable[..., object], logs: list[tuple[EvalLog, Path]], folder: Path
) -> None:
    """Inspect's writer makes its own temporary file beside its target, readable by others
    (0666 less the umask), so every log is written inside one 0700 folder and made 0600
    first, and only then moved into place: a failed write replaces no file. No order of
    moves makes several files one step, so a failed move names the logs already moved. The
    folder is removed, also when a write fails."""
    path, temporary, moved = folder, None, []
    try:
        folder.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=folder))
        for log, path in logs:
            write(log, str(temporary / path.name))
            os.chmod(temporary / path.name, 0o600)
        for _, path in logs:
            os.replace(temporary / path.name, path)
            moved.append(path.name)
    except OSError as e:
        done = (
            f" {' and '.join(moved)} {'was' if len(moved) == 1 else 'were'} already written, "
            "so export again to bring the logs back in step."
            if moved
            else ""
        )
        raise ConfigError(f"Can't write the export to {path}: {e.strerror or e}.{done}") from e
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
