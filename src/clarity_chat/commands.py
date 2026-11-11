"""Every chat command in one table, for the terminal, `clarity chat --json` and the web UI.

`Commands(session).run(name, args, emit)` runs one command and returns a JSON-able result.
`emit` receives events while it runs: `text` (reply text as it streams), `reply_start`
(before each reply of `/regen N`, with `k`, `n` and its `seed`), `status` (a model starting,
ready) and `record` (every record appended to the log). Errors are raised
(ConfigError, ContextError, BackendError); each front end shows them its own way.

Commands run one at a time, except `cancel`, which runs during a call.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clarity_chat.backends import BackendError
from clarity_chat.logs import list_chats
from clarity_chat.profiles import ConfigError
from clarity_chat.session import CallOutcome, CancelledBeforeReply, Session

Emit = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Command:
    usage: str  # "/mark LABEL"
    help: str
    run: Callable[[Commands, str, Emit], dict[str, Any]]
    during_call: bool = False  # runs while a call is in progress (only cancel)
    needs_conversation: bool = True
    cancellable: bool = False  # /cancel can stop it (it sends a request to the model)


class Commands:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.busy = threading.Lock()  # held while a command runs: one at a time
        self.cancel_event: threading.Event | None = None  # the running command's
        self.running: str | None = None  # the running command's name

    def run(self, name: str, args: str, emit: Emit) -> dict[str, Any]:
        command = COMMANDS.get(name)
        if command is None:
            raise ConfigError(f"Unknown command /{name}. /help lists them.")
        if command.needs_conversation and self.session.conversation is None:
            raise ConfigError("No conversation is running. Start one with /profile NAME.")
        if command.during_call:
            return command.run(self, args, emit)
        if not self.busy.acquire(blocking=False):
            raise ConfigError("Another command is still running; only /cancel runs now.")
        # Made before the command starts, so a cancel that arrives early isn't lost.
        self.cancel_event = threading.Event()
        self.running = name
        self.session.on_record = lambda record: emit({"event": "record", "record": record})
        try:
            return command.run(self, args, emit)
        finally:
            self.session.on_record = None
            self.cancel_event = None
            self.running = None
            self.busy.release()

    def call(self, emit: Emit, start: Callable[..., CallOutcome]) -> dict[str, Any]:
        outcome = start(
            on_text=lambda text: emit({"event": "text", "text": text}), cancel=self.cancel_event
        )
        return {"call": outcome.record, "warnings": outcome.warnings}


def needs(args: str, usage: str) -> str:
    if not args:
        raise ConfigError(f"Usage: {usage}")
    return args


def _say(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    text = needs(args, "/say TEXT")
    return c.call(emit, lambda **kw: c.session.say(text, **kw))


def _regen(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """N replies to the current message, one after another, each a new branch. A cancel stops
    the reply being written and the rest. A failure after the first reply stops the rest and
    is returned as `stopped`, with the replies already made; the head stays at the last."""
    if not args:
        n = 1
    elif args.isdigit() and int(args) > 0:
        n = int(args)
    else:
        raise ConfigError("Usage: /regen [N], where N is a whole number of replies, 1 or more.")
    replies: list[dict[str, Any]] = []
    for k in range(1, n + 1):
        if replies and c.cancel_event is not None and c.cancel_event.is_set():
            break
        try:
            emit({"event": "reply_start", "k": k, "n": n, "seed": c.session.regen_seed()})
            replies.append(c.call(emit, c.session.regen))
        except (BackendError, CancelledBeforeReply, ConfigError) as e:
            if not replies:
                raise
            return {"replies": replies, "stopped": f"Reply {k} of {n}: {e}"}
        if replies[-1]["call"]["stop_type"] == "cancelled":
            break
    return {"replies": replies}


def _edit(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    text = needs(args, "/edit TEXT")
    return c.call(emit, lambda **kw: c.session.edit(text, **kw))


def _cancel(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """Stop the running command if it can be stopped; say what was stopped, or why not."""
    event, running = c.cancel_event, c.running
    if event is None or running is None:
        return {"cancelled": False, "reason": "Nothing is running."}
    if not COMMANDS[running].cancellable:
        return {"cancelled": False, "reason": f"/{running} can't be cancelled."}
    event.set()
    return {"cancelled": True, "command": running}


def _undo(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    head = c.session.undo()
    return {"head": head.turn_id if head else None, "head_text": head.text if head else None}


def _set(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    name, _, value = needs(args, "/set NAME [VALUE]").partition(" ")
    record = c.session.set_setting(name, value.strip())
    conversation = c.session.conversation
    assert conversation is not None
    return {"record": record, "setting": conversation.settings[name].to_json()}


def _title(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return {"record": c.session.set_title(needs(args, "/title TEXT"))}


def _chats(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return {"chats": list_chats(c.session.logs_dir)}


def _history(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    conversation = c.session.conversation
    assert conversation is not None
    return {"head": conversation.head, "turns": c.session.history()}


def _branches(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    conversation = c.session.conversation
    assert conversation is not None
    return {"head": conversation.head, "forks": conversation.forks()}


def _switch(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    head, render = c.session.switch(needs(args, "/switch TURN_ID"))
    return {"head": head.turn_id, "head_text": head.text, "render": render}


def _resume(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    log = needs(args, "/resume ID")
    emit({"event": "status", "status": "starting", "log": log})
    record = c.session.resume(c.session.log_for(log))
    conversation = c.session.conversation
    assert conversation is not None
    emit({"event": "status", "status": "ready", "conversation_id": conversation.conversation_id})
    head = conversation.turns.get(record["head"]) if record["head"] else None
    return {
        "resume": record,
        "conversation_id": conversation.conversation_id,
        "profile": conversation.profile.name,
        "model": conversation.profile.model,
        "mode": conversation.profile.mode,
        "head_text": head.text if head else None,
        "warnings": [
            *(conversation.started_server or {}).get("warnings", []),
            *conversation.warnings,
        ],
    }


def _annotate(kind: str) -> Callable[[Commands, str, Emit], dict[str, Any]]:
    """mark and unmark take an optional label; note needs its text."""
    return lambda c, args, emit: {
        "records": c.session.annotate(kind, needs(args, "/note TEXT") if kind == "note" else args)
    }


def _prompt(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    assert c.session.conversation is not None
    return {"call": c.session.conversation.last_call}


LEAST_LIKELY = 10  # how many of a reply's least likely chosen tokens /probs lists


def _probs(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """The current reply's token probabilities: the full record, and its least likely
    chosen tokens with their text and the alternatives the model ranked above them."""
    conversation = c.session.conversation
    assert conversation is not None
    call = conversation.last_call
    if call is None or call.get("turn_id") is None:
        return {
            "call_id": None,
            "logprobs": None,
            "least_likely": [],
            "unavailable": "no reply yet",
        }
    logprobs = call.get("logprobs")
    if logprobs is None or "unavailable" in logprobs:
        reason = (
            logprobs["unavailable"]
            if logprobs
            else "not recorded; on a Transformers profile, /set logprobs 5 records them"
        )
        return {
            "call_id": call["call_id"],
            "logprobs": logprobs,
            "least_likely": [],
            "unavailable": reason,
        }
    decode = c.session.tokenizer.decode if c.session.tokenizer else str
    tokens = logprobs["tokens"]
    ranked = sorted(range(len(tokens)), key=lambda i: tokens[i]["logprob"])[:LEAST_LIKELY]
    least = [
        {
            "index": i,
            "id": tokens[i]["id"],
            "text": decode([tokens[i]["id"]]),
            "logprob": tokens[i]["logprob"],
            "top": [{"id": t, "text": decode([t]), "logprob": lp} for t, lp in tokens[i]["top"]],
        }
        for i in sorted(ranked)
    ]
    return {"call_id": call["call_id"], "logprobs": logprobs, "least_likely": least}


def _settings(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return settings_view(c.session)


def _replay(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return {"replay": c.session.replay_current(c.cancel_event)}


def _new(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    conversation = c.session.conversation
    assert conversation is not None
    profile = conversation.profile  # its file, so a path works too; else (resumed) its name
    return _start(c, str(profile.path) if profile.path.is_file() else profile.name, emit)


def _profile(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return _start(c, needs(args, "/profile NAME"), emit)


def _start(c: Commands, profile: str, emit: Emit) -> dict[str, Any]:
    emit({"event": "status", "status": "starting", "profile": profile})
    new = c.session.new_conversation(profile)
    emit({"event": "status", "status": "ready", "conversation_id": new.conversation_id})
    return {
        "conversation_id": new.conversation_id,
        "profile": new.profile.name,
        "model": new.profile.model,
        "mode": new.profile.mode,
        "warnings": [*(new.started_server or {}).get("warnings", []), *new.warnings],
    }


def _help(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return {"commands": [{"usage": cmd.usage, "help": cmd.help} for cmd in COMMANDS.values()]}


def _quit(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return {"quit": True}


def settings_view(session: Session) -> dict[str, Any]:
    """Everything /settings shows, as data: each value with its origin."""
    conversation = session.conversation
    assert conversation is not None
    profile = conversation.profile
    last = conversation.last_call or {}
    backend_settings = last.get("backend_settings")
    if backend_settings is None:
        if session.backend.name == "transformers":
            from clarity_chat.backends.transformers_hf import NEUTRAL_CONFIG as neutral
        else:
            from clarity_chat.backends.llama_cpp import NEUTRAL_REQUEST as neutral

        backend_settings = {k: {"value": v, "origin": "backend"} for k, v in neutral.items()}
        started = session.backend.server_record or {}
        if "eos_token_id" in started:  # Transformers: the end tokens, and where they came from
            backend_settings["eos_token_id"] = {
                "value": started["eos_token_id"],
                "origin": started.get("eos_token_id_from", "model"),
            }
    recommended = session.model_lock.get("recommended")
    model_file = (session.backend.server_record or {}).get("model_file") or {}
    return {
        "profile": {"name": profile.name, "mode": profile.mode, "model": profile.model},
        "settings": {name: s.to_json() for name, s in conversation.settings.items()},
        "backend": session.backend.name,
        "backend_settings": backend_settings,
        "eog_tokens": model_file.get("eog_tokens"),
        "recommended": recommended and {**recommended, "used": profile.use_recommended},
        "not_applied": last.get("not_applied") or [],
        "ctx_size": conversation.ctx_size.to_json(),
        "stop": profile.stop,
    }


COMMANDS: dict[str, Command] = {
    "say": Command(
        "/say TEXT",
        "send a message (plain text without / does the same)",
        _say,
        cancellable=True,
    ),
    "prompt": Command(
        "/prompt",
        "the exact prompt text of the last call, its token IDs and their count",
        _prompt,
    ),
    "probs": Command(
        "/probs",
        "the current reply's least likely tokens and the alternatives (needs logprobs on)",
        _probs,
    ),
    "settings": Command("/settings", "every setting, its value and where it came from", _settings),
    "set": Command(
        "/set NAME [VALUE]",
        "change a setting for this chat (no value: back to the profile's)",
        _set,
    ),
    "regen": Command(
        "/regen [N]",
        "N more replies to your last message (default 1; seed + 1, + 2, …), each a new branch",
        _regen,
        cancellable=True,
    ),
    "edit": Command(
        "/edit [TEXT]",
        "replace your last message (prefilled to edit) and get a new reply",
        _edit,
        cancellable=True,
    ),
    "undo": Command(
        "/undo", "go back past the last exchange; your next message starts a branch", _undo
    ),
    "history": Command(
        "/history",
        "the conversation so far, on the current branch, with marks and notes",
        _history,
    ),
    "branches": Command(
        "/branches", "where this conversation branches, with the turn IDs to /switch to", _branches
    ),
    "switch": Command(
        "/switch TURN_ID", "go to another branch: its newest reply (nothing is sent)", _switch
    ),
    "mark": Command(
        "/mark [LABEL]",
        "label the current reply (for example landed, dropped; default interesting)",
        _annotate("mark"),
    ),
    "unmark": Command(
        "/unmark [LABEL]",
        "remove the current reply's mark (name the label if it has several)",
        _annotate("unmark"),
    ),
    "note": Command(
        "/note TEXT", "add a note to the current reply (replaces its note)", _annotate("note")
    ),
    "replay": Command(
        "/replay",
        "re-send the current reply's call and compare the output",
        _replay,
        cancellable=True,
    ),
    "title": Command("/title TEXT", "name this conversation in the chat list", _title),
    "chats": Command(
        "/chats",
        "your conversations in logs/, newest first, with their titles",
        _chats,
        needs_conversation=False,
    ),
    "resume": Command(
        "/resume ID",
        "continue a logged conversation (its ID or log file) as it was logged; says what differs",
        _resume,
        needs_conversation=False,
    ),
    "new": Command("/new", "start a new conversation with the same profile", _new),
    "profile": Command(
        "/profile NAME",
        "start a new conversation with another profile",
        _profile,
        needs_conversation=False,
    ),
    "cancel": Command(
        "/cancel",
        "stop the reply being written (Ctrl-C does this in the terminal)",
        _cancel,
        during_call=True,
        needs_conversation=False,
    ),
    "help": Command("/help", "this list", _help, needs_conversation=False),
    "quit": Command(
        "/quit", "stop the server and exit (also Ctrl-D)", _quit, needs_conversation=False
    ),
}
