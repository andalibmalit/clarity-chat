"""Every chat command in one table, for the terminal, `clarity chat --json` and the web UI.

`Commands(session).run(name, args, emit)` runs one command and returns a JSON-able result.
`emit` receives events while it runs: `text` (reply text as it streams), `reply_start`
(before each reply of `/regen N`, with `k`, `n` and its `seed`), `status` (a model starting,
ready) and `record` (every record appended to the log). Errors are raised
(ConfigError, ContextError, BackendError); each front end shows them its own way.

Commands run one at a time, except `cancel` and `state`, which run during a call.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError
from clarity_chat.logs import list_chats
from clarity_chat.profiles import ConfigError, profile_path
from clarity_chat.session import CallOutcome, CancelledBeforeReply, Session

Emit = Callable[[dict[str, Any]], None]
MESSAGE_COMMANDS = {"say", "edit"}  # their args are message text, sent exactly


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
        self.streamed: list[str] = []  # the reply being written, so far (for /state)
        self.model_state = "none"  # none, starting or ready
        # Paths only from the command line (--profile, --resume), which you type. Commands
        # will also come from the browser (v0.3.0), so they take names and IDs.
        self.paths = False

    def run(self, name: str, args: str, emit: Emit, *, paths: bool = False) -> dict[str, Any]:
        command = COMMANDS.get(name)
        if command is None:
            raise ConfigError(f"Unknown command /{name}. /help lists them.")
        if name not in MESSAGE_COMMANDS:  # a message goes exactly; names and labels don't
            args = args.strip()
        if command.needs_conversation and self.session.conversation is None:
            raise ConfigError("No conversation is running. Start one with /profile NAME.")
        if command.during_call:
            return command.run(self, args, emit)
        if not self.busy.acquire(blocking=False):
            raise ConfigError("Another command is still running; only /cancel runs now.")
        # Made before the command starts, so a cancel that arrives early isn't lost.
        self.cancel_event = threading.Event()
        self.running = name
        self.paths = paths
        self.streamed = []

        def tracked(event: dict[str, Any]) -> None:
            if event["event"] == "text":
                self.streamed.append(event["text"])
            elif event["event"] == "reply_start":
                self.streamed = []
            elif event["event"] == "status":
                self.model_state = event["status"]
            emit(event)

        self.session.on_record = lambda record: tracked({"event": "record", "record": record})
        try:
            return command.run(self, args, tracked)
        finally:
            if self.model_state == "starting":  # a start that failed: whatever runs now
                self.model_state = "ready" if self.session.conversation else "none"
            self.session.on_record = None
            self.cancel_event = None
            self.streamed = []
            self.running = None
            self.paths = False
            self.busy.release()

    def call(self, emit: Emit, start: Callable[..., CallOutcome]) -> dict[str, Any]:
        outcome = start(
            on_text=lambda text: emit({"event": "text", "text": text}), cancel=self.cancel_event
        )
        return {"call": outcome.record, "warnings": outcome.warnings}


CALL_ID = re.compile(r"call-[0-9a-f]{12}")  # as logs.new_id makes them


def target(c: Commands, args: str) -> tuple[dict[str, Any] | None, str]:
    """The call a command's first word names, and the rest of its arguments. Only a whole
    call ID names a call, and it must be one of this conversation's, failed calls included;
    any other first word is the command's own text. None: the current reply."""
    first, _, rest = args.partition(" ")
    if not CALL_ID.fullmatch(first):
        return None, args
    conversation = c.session.conversation
    assert conversation is not None
    call = conversation.calls.get(first)
    if call is None:
        raise ConfigError(f"{first} isn't a call in this conversation.")
    return call, rest.strip()


def needs(args: str, usage: str) -> str:
    if not args:
        raise ConfigError(f"Usage: {usage}")
    return args


def _say(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    text = needs(args, "/say TEXT")
    return c.call(emit, lambda **kw: c.session.say(text, **kw))


def _regen(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """N replies to the current message, one after another, each a new branch; or, with
    `same`, one for each seed used for it that has no reply under the current settings.
    A cancel stops the reply being written and the rest, whenever it comes. A failure after
    the first reply stops the rest and is returned as `stopped`, with the replies already
    made; the head stays at the last."""
    if args == "same":
        conversation = c.session.conversation
        assert conversation is not None
        message = conversation.current_reply("regenerate").parent_id
        assert message is not None
        seeds = c.session.seeds_to_make(message)
        if not seeds:
            raise ConfigError(
                "Every seed used for this message already has a reply under the current "
                "settings. Change a setting first, or make a new reply with /regen."
            )
        return {**regen_replies(c, emit, list(seeds)), "seeds": seeds}
    # ASCII digits only: str.isdigit() also accepts "²", which int() refuses.
    n = int(args) if re.fullmatch(r"[0-9]+", args) else 0 if args else 1
    if n < 1:
        raise ConfigError(
            "Usage: /regen [N], where N is a whole number of replies, 1 or more; or /regen same."
        )
    return regen_replies(c, emit, [None] * n)


def regen_replies(c: Commands, emit: Emit, seeds: list[int | None]) -> dict[str, Any]:
    """One reply per entry of `seeds`, each with that seed or, for None, the next one."""
    n = len(seeds)
    replies: list[dict[str, Any]] = []
    for k, given in enumerate(seeds, 1):
        if replies and c.cancel_event is not None and c.cancel_event.is_set():
            break
        try:
            seed = c.session.regen_seed() if given is None else given
            emit({"event": "reply_start", "k": k, "n": n, "seed": seed})
            replies.append(c.call(emit, lambda seed=seed, **kw: c.session.regen(seed=seed, **kw)))
        except CancelledBeforeReply:
            if not replies:
                raise
            break
        except (BackendError, ConfigError) as e:
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


def _profiles(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    from clarity_chat.models import PROFILES_NOT_CHECKED, list_profiles

    return {"profiles": list_profiles(c.session.root), "not_checked": PROFILES_NOT_CHECKED}


def _history(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    conversation = c.session.conversation
    assert conversation is not None
    return {"head": conversation.head, "turns": c.session.history()}


def _branches(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """The forks on the path to head; with `all`, every branch end and every marked reply."""
    conversation = c.session.conversation
    assert conversation is not None
    if args == "all":
        return conversation.branch_ends(c.session.marks_and_notes())
    if args:
        raise ConfigError("Usage: /branches [all]")
    return {"head": conversation.head, "forks": conversation.forks()}


def _compare(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """The replies to the current message, or to the message the call named answers."""
    call, rest = target(c, args)
    if rest:
        raise ConfigError("Usage: /compare [CALL_ID]")
    conversation = c.session.conversation
    assert conversation is not None
    if call is None:
        return c.session.compare(conversation.current_reply("compare").turn_id)
    if not call.get("turn_id"):
        raise ConfigError(f"Call {call['call_id']} failed, so it has no reply to compare.")
    return c.session.compare(call["turn_id"])


def _switch(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    head, render = c.session.switch(needs(args, "/switch TURN_ID"))
    return {"head": head.turn_id, "head_text": head.text, "render": render}


def _resume(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    log = needs(args, "/resume ID")
    if not c.paths and not re.fullmatch(r"[A-Za-z0-9_-]+", log):
        raise ConfigError("/resume takes a conversation ID from /chats, not a path.")
    # An ID means logs/ID.jsonl only, never a file of that name in the working folder.
    path = c.session.log_for(log) if c.paths else c.session.logs_dir / f"{log}.jsonl"
    emit({"event": "status", "status": "starting", "log": log})
    record = c.session.resume(path)
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
    """mark and unmark take an optional label; note needs its text. Each can name a call."""

    def run(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
        call, text = target(c, args)
        if kind == "note":
            needs(text, "/note [CALL_ID] TEXT")
        return {"records": c.session.annotate(kind, text, call)}

    return run


def _prompt(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """The latest call's record, or the call named. `repr` is the terminal's way to show it."""
    call, rest = target(c, args)
    if rest not in ("", "repr"):
        raise ConfigError("Usage: /prompt [CALL_ID]")
    assert c.session.conversation is not None
    return {"call": call or c.session.conversation.last_call}


LEAST_LIKELY = 10  # how many of a reply's least likely chosen tokens /probs lists


def _probs(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """A reply's token probabilities: the full record; every token with its text, its place
    in the reply and the alternatives; and its least likely chosen tokens. The current reply,
    or the call named. Without probabilities, `reason` is a code and `unavailable` says why."""
    named, rest = target(c, args)
    if rest:
        raise ConfigError("Usage: /probs [CALL_ID]")
    conversation = c.session.conversation
    assert conversation is not None
    if named is not None:
        call = named
    else:  # the reply at head, which is on screen; last_call can be a failed call (/prompt)
        head = conversation.turns.get(conversation.head or "")
        call = conversation.calls.get(head.call_id or "") if head else None

    def unavailable(reason: str, why: str) -> dict[str, Any]:
        return {
            "call_id": call["call_id"] if call else None,
            "logprobs": call.get("logprobs") if call else None,
            "tokens": [],
            "least_likely": [],
            "reason": reason,
            "unavailable": why,
        }

    if call is None:
        return unavailable("no_reply", "no reply yet")
    if call.get("turn_id") is None:
        return unavailable("failed", "the call failed, so there is no reply")
    logprobs = call.get("logprobs")
    if logprobs is None:
        on_llama_cpp = (call.get("backend") or {}).get("backend", "llama.cpp") == "llama.cpp"
        why = "not recorded; on a Transformers profile, /set logprobs 5 records them"
        return unavailable("llama_cpp" if on_llama_cpp else "off", why)
    if "unavailable" in logprobs:
        why = logprobs["unavailable"]
        return unavailable("cancelled" if why == "cancelled" else "other", why)
    tokenizer = c.session.tokenizer
    assert tokenizer is not None
    reply = call["output_text"]
    ids = [t["id"] for t in logprobs["tokens"]]
    tokens, at = [], 0
    for i, (t, text) in enumerate(
        zip(logprobs["tokens"], token_texts(tokenizer, ids), strict=True)
    ):
        start = min(at, len(reply))
        at += len(text)
        tokens.append(
            {
                "index": i,
                "id": t["id"],
                "text": text,
                "start": start,
                "end": min(at, len(reply)),  # a stop string or end token lies past the reply
                "logprob": t["logprob"],
                "top": [
                    {"id": o, "text": tokenizer.decode([o]), "logprob": lp} for o, lp in t["top"]
                ],
            }
        )
    ranked = sorted(range(len(tokens)), key=lambda i: tokens[i]["logprob"])[:LEAST_LIKELY]
    least = [tokens[i] for i in sorted(ranked)]
    return {
        "call_id": call["call_id"],
        "logprobs": logprobs,
        "tokens": tokens,
        "least_likely": least,
    }


def token_texts(tokenizer: Any, ids: list[int]) -> list[str]:
    """Each token's share of the decoded text. A character split across tokens (é in a
    byte-level vocabulary) belongs to the token that completes it, so the texts join up."""
    texts, shown = [], ""
    for i in range(len(ids)):
        text = tokenizer.decode(ids[: i + 1])
        if i < len(ids) - 1:
            text = text.rstrip("\ufffd")  # a character still incomplete
        if not text.startswith(shown):  # a decoder that rewrites earlier text: keep the joins
            texts.append("")
            continue
        texts.append(text[len(shown) :])
        shown = text
    return texts


def _export(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    from clarity_chat.export import export_log

    if args:  # commands will also come from the browser (v0.3.0): no path from a command
        raise ConfigError("/export takes no path: it writes to logs/exports/ID.jsonl.")
    conversation = c.session.conversation
    assert conversation is not None
    return export_log(c.session, conversation.log.path)


def _settings(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    return settings_view(c.session)


def _replay(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    call, rest = target(c, args)
    if rest:
        raise ConfigError("Usage: /replay [CALL_ID]")
    return {"replay": c.session.replay_current(c.cancel_event, call)}


def _new(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    conversation = c.session.conversation
    assert conversation is not None
    profile = conversation.profile
    if not conversation.resumed and profile.path.is_file():  # a file you named: a path works
        return _start(c, str(profile.path), emit)
    # A resumed conversation's profile path comes from its log, which is untrusted: reuse it
    # only inside the project, else start the logged name from profiles/ or local/profiles/.
    root = c.session.root.resolve()
    if profile.path.is_file() and profile.path.resolve().is_relative_to(root):
        return _start(c, str(profile.path), emit)
    return _start(c, named_profile(c, profile.name), emit)


def _profile(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    name = needs(args, "/profile NAME")
    return _start(c, name if c.paths else named_profile(c, name), emit)


def named_profile(c: Commands, name: str) -> str:
    """A profile name's file in profiles/ or local/profiles/: never a path, and never a file
    in the working folder."""
    if Path(name).name != name or name.startswith(".") or name.endswith(".toml") or "\\" in name:
        raise ConfigError(
            "/profile takes a profile name from profiles/ or local/profiles/, not a path."
        )
    path = profile_path(c.session.root, name)
    if not path.is_file():
        raise ConfigError(f"Profile {name!r} not found in profiles/ or local/profiles/.")
    return str(path)


def _start(c: Commands, profile: str, emit: Emit) -> dict[str, Any]:
    emit({"event": "status", "status": "starting", "profile": Path(profile).stem})
    new = c.session.new_conversation(profile)
    emit({"event": "status", "status": "ready", "conversation_id": new.conversation_id})
    return {
        "conversation_id": new.conversation_id,
        "profile": new.profile.name,
        "model": new.profile.model,
        "mode": new.profile.mode,
        "warnings": [*(new.started_server or {}).get("warnings", []), *new.warnings],
    }


def _state(c: Commands, args: str, emit: Emit) -> dict[str, Any]:
    """Everything a page needs to draw itself, at any moment, mid-reply included: the
    conversation, the command running and the reply so far, and what's loaded."""
    from importlib.metadata import version

    from clarity_chat.logs import chat_summary, client_git

    session, conversation = c.session, c.session.conversation
    out: dict[str, Any] = {
        "running": c.running,
        "reply_so_far": "".join(c.streamed),
        "model": c.model_state,
        "clarity": {"version": version("clarity-chat"), **client_git()},
        "llama_cpp_commit": session.server_config.llama_cpp_commit,
        "conversation_id": None,
        "title": None,
        "profile": None,
        "head": None,
        "turns": [],
        "unanswered": None,
        "warnings": [],
        "resume": None,
    }
    if conversation is None:
        return out
    return {
        **out,
        "conversation_id": conversation.conversation_id,
        "title": chat_summary(conversation.log.path)["title"],
        "profile": conversation.profile.name,
        "head": conversation.head,
        "turns": session.history(),
        "unanswered": conversation.unanswered,
        "warnings": [
            *(conversation.started_server or {}).get("warnings", []),
            *conversation.warnings,
        ],
        "resume": conversation.resume_record,
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
        "steering": {
            "file": profile.steering.file,
            "sha256": profile.vector.sha256,
            "layers": profile.layers,
        }
        if profile.steering is not None and profile.vector is not None
        else None,
    }


COMMANDS: dict[str, Command] = {
    "say": Command(
        "/say TEXT",
        "send a message (plain text without / does the same)",
        _say,
        cancellable=True,
    ),
    "prompt": Command(
        "/prompt [CALL_ID]",
        "the exact prompt text of the last call (or the call named), its token IDs and count",
        _prompt,
    ),
    "probs": Command(
        "/probs [CALL_ID]",
        "a reply's least likely tokens and the alternatives (needs logprobs on)",
        _probs,
    ),
    "settings": Command("/settings", "every setting, its value and where it came from", _settings),
    "set": Command(
        "/set NAME [VALUE]",
        "change a setting for this chat (no value: back to the profile's)",
        _set,
    ),
    "regen": Command(
        "/regen [N | same]",
        "N more replies to your last message (default 1; next seeds up), each a new branch; "
        "same: again with its seeds, under the current settings",
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
        "/branches [all]",
        "where this conversation branches, with the turn IDs to /switch to (all: every end)",
        _branches,
    ),
    "compare": Command(
        "/compare [CALL_ID]",
        "the replies to your last message (or the call named's): what differs, where they part",
        _compare,
    ),
    "switch": Command(
        "/switch TURN_ID", "go to another branch: its newest reply (nothing is sent)", _switch
    ),
    "mark": Command(
        "/mark [CALL_ID] [LABEL]",
        "label the current reply or the call named (for example landed; default interesting)",
        _annotate("mark"),
    ),
    "unmark": Command(
        "/unmark [CALL_ID] [LABEL]",
        "remove a reply's mark (name the label if it has several)",
        _annotate("unmark"),
    ),
    "note": Command(
        "/note [CALL_ID] TEXT",
        "add a note to the current reply or the call named (replaces its note)",
        _annotate("note"),
    ),
    "replay": Command(
        "/replay [CALL_ID]",
        "re-send the current reply's call (or the call named) and compare the output",
        _replay,
        cancellable=True,
    ),
    "export": Command(
        "/export",
        "write this conversation for a notebook, to logs/exports/ID.jsonl",
        _export,
    ),
    "title": Command("/title TEXT", "name this conversation in the chat list", _title),
    "chats": Command(
        "/chats",
        "your conversations in logs/, newest first, with their titles",
        _chats,
        needs_conversation=False,
    ),
    "profiles": Command(
        "/profiles",
        "every profile you can start, its model and backend, and whether the model is fetched",
        _profiles,
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
    "state": Command(
        "/state",
        "where things are: the conversation, the command running and its reply so far",
        _state,
        during_call=True,
        needs_conversation=False,
    ),
    "help": Command("/help", "this list", _help, needs_conversation=False),
    "quit": Command(
        "/quit", "stop the server and exit (also Ctrl-D)", _quit, needs_conversation=False
    ),
}
