"""Plain export: a conversation as one JSON Lines file a notebook can load.

Line 1 says what the file holds and the conditions its calls were computed under; then one
line per completed call. The fields are in `docs/reference.md`, "Export"."""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

from clarity_chat.files import write_text_atomic
from clarity_chat.logs import client_git, read_records
from clarity_chat.profiles import ConfigError, load_profile
from clarity_chat.render import Prompt, message_spans
from clarity_chat.replay import facts_from_call
from clarity_chat.resume import rebuild
from clarity_chat.session import Session, marks_and_notes

FORMAT_VERSION = 1
RECOMPUTE_NOTE = (
    "Recomputing these calls under other conditions gives different numbers. Run "
    "`clarity_chat.recompute.check` with your loaded model first; it lists what differs."
)
# Copied from the call record as logged.
CALL_FIELDS = (
    "call_id",
    "turn_id",
    "parent_id",
    "regen_of",
    "prompt_ids",
    "output_ids",
    "prompt_text",
    "output_text",
    "settings",
    "n_ctx",
    "stop_type",
    "flags",
)
MODEL_PINS = ("name", "repo", "revision", "route", "gguf_sha256", "gguf_path")


def export_log(session: Session, log_path: Path, out: Path | None = None) -> dict[str, Any]:
    """Write `log_path`'s conversation to `out` (default `logs/exports/ID.jsonl`), replacing
    any file there. Returns the path, whether a file was replaced, and the counts."""
    if not log_path.is_file():
        raise ConfigError(
            f"No conversation or log file {log_path}. `clarity chats` or /chats lists them."
        )
    try:
        return _export(session, log_path, out)
    except (KeyError, TypeError, AttributeError) as e:  # a log Clarity didn't write
        raise ConfigError(f"{log_path} can't be exported: a record lacks or mangles {e}.") from e


def _export(session: Session, log_path: Path, out: Path | None) -> dict[str, Any]:
    records = list(read_records(log_path))
    state = rebuild(records, str(log_path))  # refuses a batch run's calls.jsonl
    turns = {t["turn_id"]: t for t in state.turns}
    current = marks_and_notes(records)
    calls = [r for r in records if r.get("type") == "call"]
    done = [c for c in calls if c.get("turn_id")]  # a reply turn: it produced output
    rows = []
    groups: dict[str, list[str]] = {}  # the conditions, as JSON, to the calls they cover
    for call in done:
        server = state.servers.get((call.get("backend") or {}).get("server_id") or "")
        key = json.dumps(conditions(call, server), sort_keys=True)
        groups.setdefault(key, []).append(call["call_id"])
        rows.append(export_call(session, call, server, turns, current.get(call["turn_id"], {})))
    conversation_id = state.start["conversation_id"]
    # A log is untrusted, and its ID names the default file: it must stay in logs/exports/.
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", conversation_id):
        raise ConfigError(
            f"{log_path}: its conversation ID {conversation_id!r} isn't one Clarity writes."
        )
    header = {
        "type": "export",
        "format_version": FORMAT_VERSION,
        "clarity": {"version": version("clarity-chat"), **client_git()},
        "conversation_id": conversation_id,
        "calls": len(rows),
        "failed_calls_left_out": len(calls) - len(done),
        "conditions": [
            {"conditions": json.loads(key), "call_ids": ids} for key, ids in groups.items()
        ],
        "note": RECOMPUTE_NOTE,
    }
    path = out or session.logs_dir / "exports" / f"{conversation_id}.jsonl"
    replaced = path.exists()
    lines = [json.dumps(row, ensure_ascii=False) for row in [header, *rows]]
    try:  # private, like the log it comes from
        write_text_atomic(path, "\n".join(lines) + "\n", private=True)
    except OSError as e:  # a folder, no permission, a file a notebook holds open on Windows
        raise ConfigError(f"Can't write the export to {path}: {e.strerror or e}.") from e
    return {
        "path": str(path),
        "replaced": replaced,
        "calls": len(rows),
        "failed_calls_left_out": header["failed_calls_left_out"],
    }


def conditions(call: dict[str, Any], server: dict[str, Any] | None) -> dict[str, Any]:
    """What a call was computed under: the backend, the weights, and every fact replay
    compares except Clarity's own commit and the tokenizer libraries, which can't change
    the numbers. So two server starts that differ in anything that can share no group."""
    backend = (call.get("backend") or {}).get("backend") or (server or {}).get("backend")
    model = call.get("model") or {}
    pins = ("repo", "revision") + (() if backend == "transformers" else ("gguf_sha256",))
    skip = {"client", "libraries", "weights", "gguf_sha256"}
    facts = facts_from_call(call, server)
    return {
        "backend": backend,
        "weights": {k: model.get(k) for k in pins},
        **{k: v for k, v in facts.items() if k not in skip},
    }


def export_call(
    session: Session,
    call: dict[str, Any],
    server: dict[str, Any] | None,
    turns: dict[str, dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    path = path_to(turns, call["turn_id"])
    spans = call.get("message_spans")
    if spans is None:  # logged before v0.2.1
        spans = derive_spans(session, call, [turns[t] for t in path[:-1]])
    return {
        "type": "call",
        **{k: call.get(k) for k in CALL_FIELDS},
        "edit_of": turns.get(call["parent_id"], {}).get("edit_of"),  # on the message
        "path": path,
        "cancelled": call.get("stop_type") == "cancelled",
        "message_spans": spans,
        "marks": [
            {k: m.get(k) for k in ("label", "origin", "time")} for m in annotations.get("marks", [])
        ],
        "notes": [
            {k: n.get(k) for k in ("text", "origin", "time")} for n in annotations.get("notes", [])
        ],
        "model": {k: (call.get("model") or {}).get(k) for k in MODEL_PINS},
        **({"logprobs": call["logprobs"]} if "logprobs" in call else {}),
        "facts": facts_from_call(call, server),
    }


def path_to(turns: dict[str, dict[str, Any]], turn_id: str) -> list[str]:
    """The turn IDs from the start of the conversation to `turn_id`."""
    path: list[str] = []
    here: str | None = turn_id
    while here is not None:
        path.append(here)
        here = turns[here].get("parent_id")
    return path[::-1]


def derive_spans(
    session: Session, call: dict[str, Any], path: list[dict[str, Any]]
) -> list[dict[str, Any]] | dict[str, str]:
    """A call's message spans, for a call logged without them: the profile and tokenizers as
    logged, and B3's function, which proves its result against the logged prompt or says why
    it can't. A template that inserts today's date can't be proved on a later day."""
    try:
        profile = load_profile(
            session.root, call["profile"]["name"], session.defaults, logged=call["profile"]
        )
        model = call["model"]
        tokenizer = session.load_tokenizer(model["tokenizer_repo"], model["tokenizer_revision"])
        template = tokenizer
        if profile.mode == "chat" and profile.template_repo and profile.template_revision:
            template = session.load_tokenizer(profile.template_repo, profile.template_revision)
        turns = [t for t in path if t.get("origin") != "priming"]  # the profile renders those
        prompt = Prompt(
            text=call["prompt_text"],
            ids=call["prompt_ids"],
            bos=call["bos"],
            template=None,
            special_in_content=[],
        )
        return message_spans(
            tokenizer,
            profile,
            [(t["role"], t["text"]) for t in turns],
            [t["turn_id"] for t in turns],
            prompt,
            template,
        )
    except Exception as e:  # a tokenizer not in the cache, a record Clarity didn't write
        return {"unavailable": f"{type(e).__name__}: {e}"}
