"""Plain export: a conversation as one JSON Lines file a notebook can load.

Line 1 says what the file holds and the conditions its calls were computed under; then one
line per completed call. The fields are in `docs/reference.md`, "Export"."""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

from clarity_chat.documents import added_ranges, checked_attachments, message_content
from clarity_chat.files import write_text_atomic
from clarity_chat.logs import LOG_NAME, client_git, read_records, unreadable
from clarity_chat.profiles import ConfigError, Profile, load_profile
from clarity_chat.render import Prompt, Tokenizer, message_spans
from clarity_chat.replay import facts_from_call
from clarity_chat.resume import rebuild
from clarity_chat.session import Session, marks_and_notes

FORMAT_VERSION = 1
RECOMPUTE_NOTE = (
    "Recomputing these calls under other conditions can give different numbers, sometimes "
    "by a tiny amount and sometimes enough to change the likeliest token. Run "
    "`clarity_chat.recompute.check` with your loaded model first; it lists what differs."
)
STEERED_NOTE = (
    " Some calls were steered; apply the same vector with `clarity_chat.steering.applied` "
    "before recomputing them."
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
    """Write `log_path`'s conversation to `out` (default `logs/exports/ID.jsonl`, ID the log
    file's name), replacing any file there. Returns the path, whether a file was replaced,
    and the counts."""
    require_log(log_path)
    try:
        return _export(session, log_path, out)
    except (KeyError, TypeError, AttributeError) as e:  # a log Clarity didn't write, or a bug
        raise ConfigError(unreadable(e, f"the records of {log_path}")) from e


def require_log(log_path: Path) -> None:
    if not log_path.is_file():
        raise ConfigError(
            f"No conversation or log file {log_path}. `clarity chats` lists "
            "conversations; name a run by its folder, runs/RUN_ID."
        )


def _export(session: Session, log_path: Path, out: Path | None) -> dict[str, Any]:
    header, rows = conversation_rows(session, list(read_records(log_path)), str(log_path))
    if out is None:
        # Named by the log's file, the ID `clarity chats` lists (a copied log keeps the
        # conversation ID, so its export mustn't overwrite the original's). It names a file
        # in logs/exports/, so it must be a name Clarity would write.
        if not LOG_NAME.fullmatch(log_path.stem):
            raise ConfigError(
                f"{log_path.name} isn't a log name Clarity writes, so it can't name the "
                "export. Pass --out PATH."
            )
        out = session.logs_dir / "exports" / f"{log_path.stem}.jsonl"
    path = out
    replaced = path.exists()
    lines = [json.dumps(row, ensure_ascii=False) for row in [header, *rows]]
    try:  # private, like the log it comes from
        write_text_atomic(path, "\n".join(lines) + "\n", private=True)
    except OSError as e:  # a folder, no permission, a file a notebook holds open on Windows
        raise ConfigError(f"Can't write the export to {path}: {e.strerror or e}.") from e
    return export_result(path, replaced, header, rows, "jsonl")


def export_result(
    path: Path, replaced: bool, header: dict[str, Any], rows: list[dict[str, Any]], form: str
) -> dict[str, Any]:
    """A conversation's export, in either format (`jsonl` or `inspect`): where it went and
    what it holds."""
    return {
        "path": str(path),
        "format": form,
        "replaced": replaced,
        "calls": len(rows),
        "failed_calls_left_out": header["failed_calls_left_out"],
        "calls_without_a_turn_left_out": header["calls_without_a_turn_left_out"],
        "steered": steered_by_vector(rows),
        "attachments": any(row["attachments"] for row in rows),  # it holds documents' text
        "found": any(row["found"] for row in rows),  # and the text of parts search added
    }


def conversation_rows(
    session: Session, records: list[dict[str, Any]], where: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One conversation's export: its header and a row per call with a reply. `where` names
    the records' file in refusals. Both formats, plain and Inspect, are built from these."""
    state = rebuild(records, where)  # refuses a batch run's calls.jsonl
    for turn in state.turns:  # an attachment missing a field: KeyError, refused
        checked_attachments(turn)
    turns = {t["turn_id"]: t for t in state.turns}
    current = marks_and_notes(records)
    calls = [r for r in records if r.get("type") == "call"]
    done = [c for c in calls if c.get("turn_id")]  # a reply turn: it produced output
    # The call is written before its turn: a crash between them leaves a call with no turn,
    # which resume carries on without. Leave it out too, and say so.
    torn = [c for c in done if c["turn_id"] not in turns]
    done = [c for c in done if c["turn_id"] in turns]
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
            f"{where}: its conversation ID {conversation_id!r} isn't one Clarity writes."
        )
    header = {
        "type": "export",
        "format_version": FORMAT_VERSION,
        "clarity": {"version": version("clarity-chat"), **client_git()},
        "conversation_id": conversation_id,
        "calls": len(rows),
        "failed_calls_left_out": len(calls) - len(done) - len(torn),
        "calls_without_a_turn_left_out": len(torn),
        "conditions": [
            {"conditions": json.loads(key), "call_ids": ids} for key, ids in groups.items()
        ],
        "note": RECOMPUTE_NOTE
        + (STEERED_NOTE if any(row["steering"] != "none" for row in rows) else ""),
    }
    return header, rows


def holds(result: dict[str, Any]) -> str | None:
    """The export's statement of whose text it holds: the terminal's and the page's."""
    whose = [
        *(["the documents attached in this chat"] if result["attachments"] else []),
        *(["the parts search added from your documents"] if result["found"] else []),
    ]
    return f"It holds the text of {' and of '.join(whose)}." if whose else None


def steered_by_vector(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each vector the exported calls were steered with (file and sha256), and those calls."""
    vectors: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row["steering"] != "none":
            key = (row["steering"]["file"], row["steering"]["sha256"])
            vectors.setdefault(key, []).append(row["call_id"])
    return [
        {"file": file, "sha256": sha256, "call_ids": ids} for (file, sha256), ids in vectors.items()
    ]


def conditions(call: dict[str, Any], server: dict[str, Any] | None) -> dict[str, Any]:
    """What a call was computed under: the backend, the weights, and every fact replay
    compares except Clarity's own commit and the tokenizer libraries, which can't change
    the numbers. So two server starts that differ in anything that can share no group."""
    backend = (call.get("backend") or {}).get("backend") or (server or {}).get("backend")
    model = call.get("model") or {}
    pins = ("repo", "revision") + (() if backend == "transformers" else ("gguf_sha256",))
    skip = {"client", "libraries", "weights", "gguf_sha256"}
    facts = facts_from_call(call, server)
    steering = logged_steering(call)
    return {
        "backend": backend,
        "weights": {k: model.get(k) for k in pins},
        **{k: v for k, v in facts.items() if k not in skip},
        # What `steering.applied` sets on a model, so the recompute check can compare them.
        "steering": {k: steering.get(k) for k in ("sha256", "scale", "layers")}
        if isinstance(steering, dict)
        else "none",
    }


def logged_steering(call: dict[str, Any]) -> dict[str, Any] | str:
    """A call's steering as logged, or "none": a value, not null, which readers take as "not
    recorded". A call logged before v0.2.2, or on llama.cpp, wasn't steered."""
    request = call.get("request")
    steering = request.get("steering") if isinstance(request, dict) else None
    if steering is None:
        return "none"
    if not isinstance(steering, dict):  # a log is untrusted: never read it as unsteered
        raise ConfigError(
            f"Call {call.get('call_id')}'s logged steering isn't one Clarity writes: {steering!r}."
        )
    return steering


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
        "attachments": attachments_on(path, turns, spans),
        "found": found_on(path, turns, spans),
        "marks": [
            {k: m.get(k) for k in ("label", "origin", "time")} for m in annotations.get("marks", [])
        ],
        "notes": [
            {k: n.get(k) for k in ("text", "origin", "time")} for n in annotations.get("notes", [])
        ],
        "model": {k: (call.get("model") or {}).get(k) for k in MODEL_PINS},
        **({"logprobs": call["logprobs"]} if "logprobs" in call else {}),
        "steering": logged_steering(call),
        "facts": facts_from_call(call, server),
    }


def attachments_on(
    path: list[str], turns: dict[str, dict[str, Any]], spans: list[dict] | dict[str, str]
) -> list[dict[str, Any]]:
    """The documents attached on a call's path, each with its message's turn and its tokens
    in the prompt (from the spans' `added`; null when the spans are unavailable)."""
    added = {
        (e.get("turn_id"), a["index"]): a["tokens"]
        for e in (spans if isinstance(spans, list) else [])
        for a in e.get("added", [])
        if a["what"] == "attachment"
    }
    keys = ("name", "how", "range", "sha256", "extracted_sha256")
    return [
        {"turn_id": t, **{k: a[k] for k in keys}, "tokens": added.get((t, i))}
        for t in path
        for i, a in enumerate(turns[t].get("attachments") or [])
    ]


def found_on(
    path: list[str], turns: dict[str, dict[str, Any]], spans: list[dict] | dict[str, str]
) -> list[dict[str, Any]]:
    """Each searched message on a call's path (v0.3.2 decision 16): its words, the
    documents' builds, and each part added with its tokens in the prompt (from the spans'
    `added`; null when the spans are unavailable)."""
    added = {
        (e.get("turn_id"), a["index"]): a["tokens"]
        for e in (spans if isinstance(spans, list) else [])
        for a in e.get("added", [])
        if a["what"] == "part"
    }
    return [
        {
            "turn_id": t,
            "method": found["method"],
            **({"words": found["words"]} if found["method"] == "words" else {}),
            "documents": [
                {"name": d["name"], "build_id": d["build_id"]} for d in found["documents"]
            ],
            "parts": [
                {
                    "id": p["id"],
                    "sha256": p["sha256"],
                    "score": p["score"],
                    "tokens": added.get((t, i)),
                }  # fmt: skip
                for i, p in enumerate(found["parts"])
            ],
        }
        for t in path
        if (found := turns[t].get("found")) is not None
    ]


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
        profile, tokenizer, template = logged_template(session, call)
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
            [(t["role"], message_content(t["text"], t)) for t in turns],
            [t["turn_id"] for t in turns],
            prompt,
            template,
            [added_ranges(t["text"], t) for t in turns],
        )
    except Exception as e:  # a tokenizer not in the cache, a record Clarity didn't write
        return {"unavailable": f"{type(e).__name__}: {e}"}


def logged_template(session: Session, call: dict[str, Any]) -> tuple[Profile, Tokenizer, Tokenizer]:
    """A call's profile as logged, its model's tokenizer and the tokenizer whose chat template
    rendered it."""
    profile = load_profile(
        session.root, call["profile"]["name"], session.defaults, logged=call["profile"]
    )
    model = call["model"]
    tokenizer = session.load_tokenizer(model["tokenizer_repo"], model["tokenizer_revision"])
    template = tokenizer
    if profile.mode == "chat" and profile.template_repo and profile.template_revision:
        template = session.load_tokenizer(profile.template_repo, profile.template_revision)
    return profile, tokenizer, template


class ConversationExport(NamedTuple):
    """One conversation of a batch run, as the plain export builds it."""

    header: dict[str, Any]
    rows: list[dict[str, Any]]  # each with the call's `run`
    records: list[dict[str, Any]]  # its own records as logged, in log order


def run_conversations(
    session: Session, run_dir: Path
) -> tuple[dict[str, Any], list[ConversationExport]]:
    """A finished batch run's `run` record and its conversations, in the log's order. Each
    is rebuilt as a chat log of its own would be: its records, with the run's server starts
    (logged once, under the first conversation to use each server) after its first."""
    log = run_dir / "calls.jsonl"
    records = list(read_records(log))
    if not records or records[0]["type"] != "run":
        found = f"starts with a {records[0]['type']} record" if records else "has no records"
        error = ValueError(f"it {found}, not a run record")
        raise ConfigError(unreadable(error, f"the records of {log}"))
    run = records[0]
    if not (run_dir / "key.json").is_file():
        raise ConfigError(
            f"Run {run['run_id']!r} stopped before it finished (it has no key.json), so it "
            "isn't exported."
        )
    servers = [r for r in records if r["type"] == "server_start"]
    own: dict[str, list[dict[str, Any]]] = {}
    for record in records[1:]:  # a torn_line, read by no reader, belongs to no conversation
        if record["type"] not in ("server_start", "torn_line"):
            own.setdefault(record["conversation_id"], []).append(record)
    conversations = []
    for kept in own.values():
        header, rows = conversation_rows(session, kept[:1] + servers + kept[1:], str(log))
        calls = {r["call_id"]: r for r in kept if r["type"] == "call"}
        rows = [{**row, "run": calls[row["call_id"]]["run"]} for row in rows]
        conversations.append(ConversationExport(header, rows, kept))
    return run, conversations
