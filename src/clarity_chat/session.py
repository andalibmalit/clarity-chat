"""Conversations, turns and calls.

A conversation is a tree of turns (each turn records its parent and origin); `head` is
the turn the next message continues from, so regenerating or editing adds a branch and
never overwrites. A call renders the path from the root to a user turn, tokenizes it,
checks it fits the context, sends the token IDs to the backend, and writes one `call`
record, whether it succeeded, failed or was cancelled.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarity_chat import steering
from clarity_chat.backends import Backend, BackendError, Result
from clarity_chat.logs import LogFile, client_git, machine, new_id, read_records
from clarity_chat.models import check_weights_quick, lock_entry, source_weights
from clarity_chat.profiles import (
    SEED_MAX,
    SETTING_NAMES,
    ConfigError,
    ModelEntry,
    Profile,
    ServerConfig,
    Setting,
    check_backend_settings,
    check_samplers,
    check_server_flags,
    check_setting,
    load_defaults,
    load_models,
    load_profile,
    load_server_config,
    parse_setting,
    profile_path,
    resolve_settings,
    steering_path,
)
from clarity_chat.render import (
    Prompt,
    Tokenizer,
    build_prompt,
    chat_template_source,
    library_versions,
    message_spans,
    unread_kwargs,
)
from clarity_chat.replay import (
    compare_output,
    deep_diff,
    environment_diff,
    facts_from_call,
    find_call,
    first_difference,
    not_recorded,
    unexplained,
)
from clarity_chat.resume import rebuild, settings_diff

CONTEXT_WARN_FRACTION = 0.8


class ContextError(Exception):
    """The prompt plus max_tokens doesn't fit the context; nothing was sent."""


class CancelledBeforeReply(BackendError):
    """The call was cancelled before its first token: logged like a failed call, no reply."""


def room_left(n_ctx: int, n_prompt: int, max_tokens: int) -> int:
    """How many more prompt tokens fit; a call is allowed only at 0 or more.

    With context shift off, llama-server returns at most n_ctx - n_prompt tokens and marks
    the reply truncated once prompt + generated reaches n_ctx, even when that last token was
    also the max_tokens-th. So a call needs n_prompt + max_tokens < n_ctx; then the server
    never marks it truncated. A live test checks both sides of the boundary."""
    return n_ctx - n_prompt - max_tokens - 1


@dataclass
class Turn:
    turn_id: str
    parent_id: str | None
    role: str  # user | assistant
    origin: str  # user | model | edit | priming
    text: str
    call_id: str | None = None
    seed: int | None = None
    regen_of: str | None = None  # a regenerated reply: the call ID of the first reply
    edit_of: str | None = None  # an edited message: the user turn it replaces


@dataclass
class Prepared:
    """A profile with everything it needs, checked, before the model starts."""

    profile: Profile
    model: ModelEntry
    lock: dict[str, Any]
    weights: Path
    tokenizer: Tokenizer
    template_tokenizer: Tokenizer
    template: dict[str, Any] | None
    unread_kwargs: list[str] = field(default_factory=list)  # template_kwargs it never reads


@dataclass
class CallOutcome:
    record: dict[str, Any]
    result: Result | None
    turn: Turn | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    conversation_id: str
    profile: Profile
    log: LogFile
    ctx_size: Setting  # the context size it runs at
    turns: dict[str, Turn] = field(default_factory=dict)
    head: str | None = None
    run: dict[str, Any] | None = None  # set for a batch run's conversations
    started_server: dict[str, Any] | None = None  # the server_start, if this one started it
    last_call: dict[str, Any] | None = None  # the call of the reply at head (for /prompt)
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)  # call records by ID
    warnings: list[str] = field(default_factory=list)  # shown when it starts, and logged
    chat_settings: dict[str, Any] = field(default_factory=dict)  # set with /set
    resumed: bool = False  # rebuilt from its log: /switch checks the prompt renders the same
    # The model's recommended values the settings came from, as logged; None for a new
    # conversation, whose calls log today's models.lock.
    logged_recommended: dict[str, Any] | None = None

    @property
    def settings(self) -> dict[str, Setting]:
        """The profile's settings, with this conversation's /set values over them."""
        chat = {name: Setting(value, "chat") for name, value in self.chat_settings.items()}
        return {**self.profile.settings, **chat}

    def path_to(self, turn_id: str | None) -> list[Turn]:
        """Turns from the root to `turn_id`, in order."""
        path = []
        while turn_id is not None:
            turn = self.turns[turn_id]
            path.append(turn)
            turn_id = turn.parent_id
        return path[::-1]

    def add_turn(
        self,
        role: str,
        origin: str,
        text: str,
        parent_id: str | None,
        turn_id: str | None = None,
        **extra: Any,
    ) -> Turn:
        turn = Turn(turn_id or new_id("turn"), parent_id, role, origin, text, **extra)
        self.turns[turn.turn_id] = turn
        self.log.append(
            {
                "type": "turn",
                "conversation_id": self.conversation_id,
                "turn_id": turn.turn_id,
                "parent_id": parent_id,
                "role": role,
                "origin": origin,
                "text": text,
                "call_id": turn.call_id,
                **{k: v for k, v in extra.items() if k in ("regen_of", "edit_of") and v},
            }
        )
        return turn

    def replies_to(self, user_turn_id: str) -> list[Turn]:
        """The model's replies to a user turn, oldest first: the first, then regenerations."""
        return [
            t for t in self.turns.values() if t.parent_id == user_turn_id and t.role == "assistant"
        ]

    def move_head(self, turn_id: str | None, reason: str, **extra: Any) -> Turn | None:
        """Move `head` where a new reply doesn't: `undo`, `switch` or a `failed` call. Logged
        as a `head` record (with `extra` fields), so a resumed conversation continues from
        the same place."""
        self.head = turn_id
        head = self.turns.get(turn_id) if turn_id else None
        self.last_call = self.calls.get(head.call_id or "") if head else None
        self.log.append(
            {
                "type": "head",
                "conversation_id": self.conversation_id,
                "turn_id": turn_id,
                "reason": reason,
                **extra,
            }
        )
        return head

    def newest_replies(self) -> dict[str, Turn]:
        """For each turn, the most recent reply on the branch that starts there (itself
        included). One pass over the replies, oldest first, so newer ones win."""
        newest: dict[str, Turn] = {}
        for turn in self.turns.values():
            if turn.role == "assistant":
                for on_path in self.path_to(turn.turn_id):
                    newest[on_path.turn_id] = turn
        return newest

    def forks(self) -> list[dict[str, Any]]:
        """Each place on the path to `head` where the conversation branches (or, at `head`
        itself, continues: a branch you went back from with /undo). Options without a
        reply (a message whose call failed) are left out."""
        on_path = [t.turn_id for t in self.path_to(self.head)]
        newest = self.newest_replies()
        out = []
        for at in [None, *on_path]:
            options = [
                {
                    "turn_id": t.turn_id,
                    "role": t.role,
                    "text": t.text,
                    "current": t.turn_id in on_path,
                    "goes_to": newest[t.turn_id].turn_id,
                }
                for t in self.turns.values()
                if t.parent_id == at and t.turn_id in newest
            ]
            if len(options) > 1 or (options and at == self.head):
                turn = self.turns[at] if at else None
                out.append(
                    {
                        "at": at,
                        "at_role": turn.role if turn else None,
                        "at_text": turn.text if turn else None,
                        "options": options,
                    }
                )
        return out

    def current_reply(self, action: str) -> Turn:
        """The reply at `head`; an error naming `action` if there isn't one."""
        turn = self.turns.get(self.head) if self.head else None
        if turn is None or turn.role != "assistant":
            raise ConfigError(f"No reply to {action} yet.")
        return turn


class Session:
    """The client's state: config, the backend, and the current conversation."""

    def __init__(
        self,
        root: Path,
        backend: Backend | None = None,
        logs_dir: Path | None = None,
        tokenizer_loader: Callable[[str, str], Tokenizer] = Tokenizer.from_hub,
        on_server: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        """`on_server` hears each llama-server's port, PID and key when it starts, and None
        when it stops (the instance lock keeps them, to clean up after a crash)."""
        self.root = root
        self.on_server = on_server
        self.on_record: Callable[[dict[str, Any]], None] | None = None  # every record logged
        self.defaults = load_defaults(root)
        self.models = load_models(root)
        self.server_config: ServerConfig = load_server_config(root)
        self.logs_dir = logs_dir or root / "logs"
        self.tokenizer_loader = tokenizer_loader
        # An injected backend (tests) serves every profile; otherwise one per backend name
        # and device, made when a profile first needs it. Only one is ever loaded.
        self.fixed_backend = backend
        self.backends: dict[tuple[str, str | None], Backend] = {}
        self.backend = backend or self.backend_for("llama.cpp", None)
        self.conversation: Conversation | None = None
        self.model: ModelEntry | None = None
        self.model_lock: dict[str, Any] = {}
        self.tokenizer: Tokenizer | None = None
        self.template_tokenizer: Tokenizer | None = None
        self.template: dict[str, Any] | None = None
        self.weights: Path | None = None  # what the current model was started from
        self._tokenizers: dict[tuple[str, str], Tokenizer] = {}

    # --- setup ----------------------------------------------------------------------------

    def load_profile(self, name: str) -> Profile:
        return load_profile(self.root, name, self.defaults)

    def backend_for(self, name: str, device: str | None) -> Backend:
        if self.fixed_backend is not None:
            return self.fixed_backend
        if (name, device) not in self.backends:
            if name == "transformers":
                from clarity_chat.backends.transformers_hf import TransformersBackend

                assert device is not None
                backend: Backend = TransformersBackend(device, self.server_config.ctx_size)
            else:
                from clarity_chat.backends.llama_cpp import LlamaCppBackend

                backend = LlamaCppBackend(self.server_config, self.logs_dir, self.on_server)
            self.backends[name, device] = backend
        return self.backends[name, device]

    def use_server_flags(self, flags: dict[str, Any]) -> None:
        """Start llama-server with these flags instead of server.toml's (replay --logged-flags)."""
        import dataclasses

        from clarity_chat.backends.llama_cpp import LlamaCppBackend

        check_server_flags(flags, "the logged server_start's flags")
        self.close()
        self.server_config = dataclasses.replace(self.server_config, flags=flags)
        backend = LlamaCppBackend(self.server_config, self.logs_dir, self.on_server)
        self.backends["llama.cpp", None] = self.backend = backend

    def use_backend(self, name: str, device: str | None) -> Backend:
        """Make a backend the active one, closing the other first (one model in memory)."""
        backend = self.backend_for(name, device)
        if backend is not self.backend:
            self.backend.close()
            self.backend = backend
        return backend

    def load_tokenizer(self, repo: str, revision: str) -> Tokenizer:
        if (repo, revision) not in self._tokenizers:
            self._tokenizers[repo, revision] = self.tokenizer_loader(repo, revision)
        return self._tokenizers[repo, revision]

    def tokenizers_for(self, profile: Profile, model: ModelEntry) -> tuple[Tokenizer, Tokenizer]:
        """The model's tokenizer, which always tokenizes, and the tokenizer whose chat
        template renders chat mode (the profile's template_repo, else the model's)."""
        tokenizer = self.load_tokenizer(model.tokenizer_repo, model.tokenizer_revision)
        if profile.mode == "chat" and profile.template_repo and profile.template_revision:
            return tokenizer, self.load_tokenizer(profile.template_repo, profile.template_revision)
        return tokenizer, tokenizer

    def prepare(self, profile_name: str, logged: dict[str, Any] | None = None) -> Prepared:
        """Load a profile and everything it needs short of starting the model: its model
        entry and lock, the weights (size and time only), tokenizers and chat template.
        `logged`: a log's `profile` field, to use the profile as it was logged."""
        profile = load_profile(self.root, profile_name, self.defaults, logged)
        if profile.model not in self.models:
            raise ConfigError(
                f"Profile {profile.name}: model {profile.model!r} is not in models.toml."
            )
        model = self.models[profile.model]
        entry = lock_entry(self.root, model)
        if profile.use_recommended:
            recommended = entry.get("recommended")
            if recommended is None:
                raise ConfigError(
                    f"models.lock has no recommended settings for {model.name}. "
                    f"Run `clarity fetch {model.name}` again."
                )
            profile.settings = resolve_settings(
                profile.raw_settings, self.defaults, str(profile.path), recommended["values"]
            )
        weights = self.weights_for(profile.backend, entry)
        if profile.steering is not None:
            profile.vector, profile.layers = self.load_vector(
                profile.steering.file, profile.steering.sha256, profile.steering.layers, weights
            )
        tokenizer, template_tokenizer = self.tokenizers_for(profile, model)
        if profile.mode == "chat":
            template = chat_template_source(template_tokenizer, profile)
            text = template_tokenizer.tok.chat_template
            unread = unread_kwargs(text, profile.template_kwargs)
        else:
            template, unread = None, []
        return Prepared(
            profile, model, entry, weights, tokenizer, template_tokenizer, template, unread
        )

    def load_vector(
        self, file: str, sha256: str | None, layers: object, weights: Path
    ) -> tuple[steering.Vector, list[int]]:
        """Load a steering vector and check it against the model's config.json. Returns it
        and the blocks it steers. The caller has checked `file` is inside the project."""
        vector = steering.load(self.root / file, sha256)
        try:
            config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ConfigError(f"Can't read {weights / 'config.json'}: {e}") from e
        if not isinstance(config, dict):
            raise ConfigError(f"{weights / 'config.json'} isn't a JSON object.")
        return vector, steering.check(vector, config, layers)

    @staticmethod
    def steering_for(profile: Profile, values: dict[str, Any]) -> steering.Steering | None:
        """What steers a call on this profile with these settings: nothing at scale 0."""
        if values["steering_scale"] == 0:
            return None
        # Every path refuses a non-zero scale without a vector (check_backend_settings); if
        # one ever slips through, fail rather than send an unsteered call.
        assert profile.vector is not None, "a non-zero steering_scale with no vector"
        assert profile.steering is not None and profile.layers is not None
        return steering.Steering(
            profile.steering.file, profile.vector, values["steering_scale"], profile.layers
        )

    def logged_steering(
        self, call: dict[str, Any], backend: str, weights: Path
    ) -> steering.Steering | None:
        """The steering a logged call was made with, checked like a profile's: a log is
        untrusted input. A call logged before v0.2.2 has none. A vector that's gone or
        changed is refused, never replayed unsteered."""
        request = call.get("request")
        logged = request.get("steering") if isinstance(request, dict) else None
        if logged is None:
            return None
        where = f"Call {call['call_id']}'s logged steering"
        if (
            not isinstance(logged, dict)
            or set(logged) != {"file", "sha256", "scale", "layers"}
            or not isinstance(logged["file"], str)
            or not isinstance(logged["sha256"], str)
        ):
            raise ConfigError(f"{where} isn't one Clarity writes: {logged!r}.")
        if backend != "transformers":
            raise ConfigError(
                f"{where} is on a {backend} call; steering runs on the Transformers backend."
            )
        scale = check_setting("steering_scale", logged["scale"], where)
        if scale == 0:  # Clarity logs no steering at scale 0
            raise ConfigError(f"{where} isn't one Clarity writes: its scale is 0.")
        file = logged["file"]
        steering_path(self.root, file, where)
        try:
            vector, layers = self.load_vector(file, logged["sha256"], logged["layers"], weights)
        except ConfigError as e:
            raise ConfigError(
                f"Call {call['call_id']} was steered with {file} (sha256 {logged['sha256']}), "
                f"and it can't be used now: {e}"
            ) from e
        return steering.Steering(file, vector, scale, layers)

    def ctx_size_for(self, profile: Profile) -> Setting:
        """The profile's context size, else server.toml's ctx-size."""
        if profile.ctx_size is not None:
            return Setting(profile.ctx_size, "profile")
        return Setting(self.server_config.ctx_size, "server.toml")

    def weights_for(self, backend: str, entry: dict[str, Any]) -> Path:
        """llama.cpp loads the GGUF; Transformers loads the official weights it came from."""
        if backend == "transformers":
            return source_weights(self.root, entry)
        return check_weights_quick(self.root, entry)

    def new_conversation(
        self,
        profile: str | Prepared,
        log: LogFile | None = None,
        run: dict[str, Any] | None = None,
    ) -> Conversation:
        """Start a conversation with a profile (a name, a path, or one already prepared),
        starting or switching the model as needed.

        `log`: a log file to share (a batch run writes all its conversations to one file);
        by default each conversation gets its own file in `logs_dir`. `run` is copied into
        the conversation's records and calls."""
        # Everything that can fail on the profile alone is checked before the server or the
        # session changes, so a bad /new leaves the current conversation usable.
        prepared = profile if isinstance(profile, Prepared) else self.prepare(profile)
        model = prepared.model
        ctx_size = self.ctx_size_for(prepared.profile)
        started = self.start_model(prepared, ctx_size)
        conversation_id = new_id("conversation")
        log = log or LogFile(self.logs_dir / f"{conversation_id}.jsonl")
        log.on_append = self._appended
        conversation = Conversation(
            conversation_id,
            prepared.profile,
            log,
            run=run,
            started_server=started,
            ctx_size=ctx_size,
        )
        conversation.warnings += self._profile_warnings(prepared)
        log.append(
            {
                "type": "conversation",
                "conversation_id": conversation_id,
                **({"run": run} if run else {}),
                "profile": prepared.profile.to_json(),
                "settings": {k: s.to_json() for k, s in prepared.profile.settings.items()},
                "ctx_size": ctx_size.to_json(),
                "model": model.to_json(),
                "client": client_git(),
                "machine": machine(),
                "warnings": conversation.warnings,
            }
        )
        self.conversation = conversation
        self._log_server_start()
        return conversation

    @staticmethod
    def _profile_warnings(prepared: Prepared) -> list[str]:
        """Warnings shown when a conversation starts or resumes, and logged."""
        unread = prepared.unread_kwargs
        if not unread:
            return []
        return [
            f"The chat template never reads template_kwargs {unread}, so they change "
            "nothing. Check their names against the template (/prompt shows its output)."
        ]

    def start_model(self, prepared: Prepared, ctx_size: Setting) -> dict[str, Any] | None:
        """Start (or keep) the model a prepared profile needs, at `ctx_size`. Returns the
        server_start record if a server started. On failure the current conversation's
        model is restarted, so it stays usable."""
        previous = self.backend
        try:
            backend = self.use_backend(prepared.profile.backend, prepared.profile.device)
            started = backend.start(prepared.model, prepared.weights, ctx_size.value)
        except (ConfigError, BackendError) as e:
            self._restore(previous, e)
            raise
        self.model, self.model_lock, self.weights = prepared.model, prepared.lock, prepared.weights
        self.tokenizer = prepared.tokenizer
        self.template_tokenizer = prepared.template_tokenizer
        self.template = prepared.template
        return started

    def _restore(self, previous: Backend, error: Exception) -> None:
        """After a failed switch, restart the current conversation's model, so a bad /new or
        /profile leaves that conversation usable. If that fails too, no conversation runs."""
        if self.conversation is None or self.model is None or self.weights is None:
            return
        if self.backend is not previous:
            self.backend.close()
            self.backend = previous
        try:
            previous.start(self.model, self.weights, self.conversation.ctx_size.value)
        except (ConfigError, BackendError) as again:
            self.conversation = None
            raise BackendError(
                f"{error} Restarting {self.model.name} for the current conversation failed "
                f"too ({again}). Start a conversation with /profile NAME."
            ) from error

    def _log_server_start(self) -> None:
        """A log file has the record of each server its calls use, before the first call."""
        conversation = self.conversation
        record = self.backend.server_record
        if conversation is None or record is None:
            return
        if record["server_id"] not in conversation.log.servers_logged:
            conversation.log.append({**record, "conversation_id": conversation.conversation_id})
            conversation.log.servers_logged.add(record["server_id"])

    def close(self) -> None:
        self.backend.close()
        for backend in self.backends.values():
            backend.close()

    # --- calls ----------------------------------------------------------------------------

    def say(
        self,
        text: str,
        on_text: Callable[[str], None] | None = None,
        seed: Setting | None = None,
        cancel: threading.Event | None = None,
    ) -> CallOutcome:
        """Add a user turn after `head` and get the model's reply. `seed` replaces the
        profile's seed for this call (a batch run's seeds). `cancel`: an event the caller
        makes for this call and may set from any thread to stop it."""
        conversation = self._require_conversation()
        user = conversation.add_turn("user", "user", text, conversation.head)
        conversation.head = user.turn_id
        return self.call(user, seed=seed, on_text=on_text, cancel=cancel)

    def regen_seed(self) -> int:
        """The next regeneration's seed: the first reply's seed plus k, where k counts this
        and earlier regenerations of the current message."""
        return self._regen_start()[3]

    def _regen_start(self) -> tuple[Turn, Turn, Turn, int]:
        """The current reply, its message, that message's first reply, and the next seed."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("regenerate")
        assert reply.parent_id is not None
        replies = conversation.replies_to(reply.parent_id)
        first = next((t for t in replies if t.regen_of is None), replies[0])
        if first.seed is None or first.call_id is None:
            raise ConfigError("The first reply to this message has no seed to start from.")
        seed = first.seed + 1 + sum(t.regen_of is not None for t in replies)
        if seed > SEED_MAX:
            raise ConfigError(
                f"The next regeneration seed, {seed}, is above {SEED_MAX}. Edit the message or "
                "start a new conversation instead."
            )
        return reply, conversation.turns[reply.parent_id], first, seed

    def regen(
        self, on_text: Callable[[str], None] | None = None, cancel: threading.Event | None = None
    ) -> CallOutcome:
        """Another reply to the current message, as a new branch, with `regen_seed()`."""
        reply, user, first, seed = self._regen_start()
        return self.call(
            user,
            seed=Setting(seed, "regen"),
            regen_of=first.call_id,
            on_text=on_text,
            cancel=cancel,
            back_to=reply.turn_id,  # a failed regeneration leaves you where you were
        )

    def edit(
        self,
        text: str,
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> CallOutcome:
        """Replace the current message with `text`, as a new branch, and get a reply."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("edit the message of")
        assert reply.parent_id is not None
        original = conversation.turns[reply.parent_id]
        user = conversation.add_turn(
            "user", "edit", text, original.parent_id, edit_of=original.turn_id
        )
        conversation.head = user.turn_id
        # A failed edit leaves you where you were.
        return self.call(user, on_text=on_text, cancel=cancel, back_to=reply.turn_id)

    def undo(self) -> Turn | None:
        """Move back past the last exchange. Returns the new head (None at the start). The
        undone turns stay in the log; the next message branches from the new head."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("undo")
        assert reply.parent_id is not None
        return conversation.move_head(conversation.turns[reply.parent_id].parent_id, "undo")

    def set_setting(self, name: str, text: str) -> dict[str, Any]:
        """Change a setting for this conversation (`text` empty: back to the profile's value).
        Checked like a profile, and logged as a `settings` record with every /set value."""
        conversation = self._require_conversation()
        if name not in SETTING_NAMES:
            raise ConfigError(f"/set: unknown setting {name!r}. Known: {list(SETTING_NAMES)}.")
        chat = dict(conversation.chat_settings)
        if text:
            chat[name] = parse_setting(name, text, "/set")
        elif name in chat:
            del chat[name]
        else:
            raise ConfigError(f"{name} isn't set in this chat. Usage: /set NAME VALUE")
        trial = {**conversation.profile.settings}
        trial.update({k: Setting(v, "chat") for k, v in chat.items()})
        check_samplers(trial, "/set")
        steered = conversation.profile.steering is not None
        check_backend_settings(trial, conversation.profile.backend, "/set", steered)
        conversation.chat_settings = chat
        return conversation.log.append(
            {
                "type": "settings",
                "conversation_id": conversation.conversation_id,
                "name": name,
                "value": chat.get(name),
                "chat_settings": chat,
            }
        )

    def history(self) -> list[dict[str, Any]]:
        """The conversation from its start to `head`: each turn with its current marks and
        notes. What a front end needs to draw the chat, without reading the log."""
        conversation = self._require_conversation()
        notes = marks_and_notes(read_records(conversation.log.path, annotation_line))
        return [
            {
                "turn_id": t.turn_id,
                "role": t.role,
                "origin": t.origin,
                "text": t.text,
                "call_id": t.call_id,
                "regen_of": t.regen_of,
                "edit_of": t.edit_of,
                "marks": [
                    {"mark_id": m.get("mark_id"), "label": m["label"]}
                    for m in notes.get(t.turn_id, {}).get("marks", [])
                ],
                "notes": [
                    {"note_id": n.get("note_id"), "text": n["text"]}
                    for n in notes.get(t.turn_id, {}).get("notes", [])
                ],
            }
            for t in conversation.path_to(conversation.head)
        ]

    def set_title(self, title: str) -> dict[str, Any]:
        """Name this conversation in the chat list. Without one, the list shows the first
        words of the first message; no model is ever asked for a title."""
        conversation = self._require_conversation()
        return conversation.log.append(
            {"type": "title", "conversation_id": conversation.conversation_id, "title": title}
        )

    def switch(self, turn_id: str) -> tuple[Turn, dict[str, Any] | None]:
        """Go to another branch: the newest reply on the branch that starts at `turn_id`
        (a turn ID from /branches). Nothing is sent. In a resumed conversation, also
        returns (and logs in the `head` record) whether that reply's prompt renders to the
        same token IDs today."""
        conversation = self._require_conversation()
        if turn_id not in conversation.turns:
            raise ConfigError(f"This conversation has no turn {turn_id}. /branches lists them.")
        reply = conversation.newest_replies().get(turn_id)
        if reply is None:
            raise ConfigError(f"Turn {turn_id} has no reply to go to.")
        render = self.check_render(conversation, reply) if conversation.resumed else None
        extra = {"render": render} if render else {}
        head = conversation.move_head(reply.turn_id, "switch", **extra)
        assert head is not None
        return head, render

    def annotate(self, kind: str, text: str) -> list[dict[str, Any]]:
        """A `mark`, `unmark` or `note` on the current reply. See `annotate_call`."""
        conversation = self._require_conversation()
        reply = conversation.current_reply(kind)
        about = {
            "conversation_id": conversation.conversation_id,
            "call_id": reply.call_id,
            "turn_id": reply.turn_id,
        }
        return annotate_call(conversation.log, about, kind, text, "live")

    def build_prompt(
        self,
        user_turn: Turn,
        conversation: Conversation | None = None,
        prepared: Prepared | None = None,
    ) -> Prompt:
        """The prompt for `user_turn`, with the running model's tokenizers, or `prepared`'s
        (to check a conversation before its model starts)."""
        conversation = conversation or self._require_conversation()
        tokenizer = prepared.tokenizer if prepared else self.tokenizer
        template_tokenizer = prepared.template_tokenizer if prepared else self.template_tokenizer
        assert tokenizer is not None
        turns = [(t.role, t.text) for t in prompt_turns(conversation, user_turn)]
        return build_prompt(tokenizer, conversation.profile, turns, template_tokenizer)

    def message_spans(self, user_turn: Turn, prompt: Prompt) -> list[dict] | dict[str, str]:
        """Where each message sits in `prompt` (render.message_spans), with turn IDs."""
        conversation = self._require_conversation()
        assert self.tokenizer is not None
        path = prompt_turns(conversation, user_turn)
        return message_spans(
            self.tokenizer,
            conversation.profile,
            [(t.role, t.text) for t in path],
            [t.turn_id for t in path],
            prompt,
            self.template_tokenizer,
        )

    @staticmethod
    def resolve_settings(
        settings: dict[str, Setting], seed: Setting | None = None
    ) -> tuple[dict[str, Setting], bool]:
        """Settings for one call with a concrete seed, and whether it was drawn."""
        settings = dict(settings)
        if seed is not None:
            settings["seed"] = seed
            return settings, False
        if settings["seed"].value == "random":
            settings["seed"] = Setting(secrets.randbelow(SEED_MAX + 1), settings["seed"].origin)
            return settings, True
        return settings, False

    def call(
        self,
        user_turn: Turn,
        seed: Setting | None = None,
        regen_of: str | None = None,
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        back_to: str | None = None,
    ) -> CallOutcome:
        """Get the reply to `user_turn`. If the call fails, `head` goes to `back_to`
        (default: before the message, which becomes a dead branch)."""
        conversation = self._require_conversation()
        profile = conversation.profile
        assert self.model is not None and self.tokenizer is not None
        self._log_server_start()
        call_id = new_id("call")
        settings, seed_drawn = self.resolve_settings(conversation.settings, seed)
        values = {name: s.value for name, s in settings.items()}
        record: dict[str, Any] = {
            "type": "call",
            "conversation_id": conversation.conversation_id,
            "call_id": call_id,
            "parent_id": user_turn.turn_id,
            "regen_of": regen_of,
            **({"run": conversation.run} if conversation.run else {}),
            "profile": profile.to_json(),
            "settings": {k: s.to_json() for k, s in settings.items()},
            "seed_drawn": seed_drawn,
            "backend": self.backend.describe(),
            "model": {
                **self.model.to_json(),
                "gguf_sha256": self.model_lock["gguf"]["sha256"],
                "gguf_path": self.model_lock["gguf"]["path"],
                "converted_with": {
                    "llama_cpp": self.model_lock.get("llama_cpp"),
                    "converter": {
                        k: v
                        for k, v in self.model_lock.get("converter", {}).items()
                        if k != "packages"
                    },
                },
            },
            "recommended": (
                (conversation.logged_recommended or self.model_lock.get("recommended"))
                if profile.use_recommended
                else None
            ),
            "template": self.template,
            "libraries": library_versions(),
            "client": client_git(),
            "n_ctx": self.backend.n_ctx,
            "stop": profile.stop,
        }
        warnings: list[str] = []
        try:
            prompt = self.build_prompt(user_turn)
            record.update(
                bos=prompt.bos,
                prompt_text=prompt.text,
                prompt_ids=prompt.ids,
                n_prompt_client=len(prompt.ids),
                special_tokens_in_content=prompt.special_in_content,
                message_spans=self.message_spans(user_turn, prompt),
            )
            new_message = self.tokenizer.special_tokens_in(user_turn.text)
            if new_message:
                warnings.append(
                    f"Your message contains {new_message}: the tokenizer reads these as control "
                    "tokens, not text. /prompt shows the IDs."
                )
            n_ctx, max_tokens = self.backend.n_ctx, values["max_tokens"]
            if room_left(n_ctx, len(prompt.ids), max_tokens) < 0:
                raise ContextError(
                    f"The prompt is {len(prompt.ids)} tokens; with max_tokens {max_tokens} it "
                    f"needs more than the context of {n_ctx} tokens. Nothing was sent. Start a "
                    "new conversation (/new) or lower max_tokens."
                )
            if len(prompt.ids) > CONTEXT_WARN_FRACTION * n_ctx:
                warnings.append(
                    f"The prompt uses {len(prompt.ids)} of {n_ctx} context tokens "
                    f"({len(prompt.ids) / n_ctx:.0%})."
                )
            if cancel is not None and cancel.is_set():  # cancelled while the prompt was built
                result = Result(
                    text="",
                    token_ids=[],
                    stop_type="cancelled",
                    cancelled_during="prompt",
                    flags=["cancelled"],
                )
            else:
                result = self.backend.complete(
                    prompt.ids,
                    values,
                    profile.stop,
                    on_text=on_text,
                    cancel=cancel,
                    steering=self.steering_for(profile, values),
                )
            if result.stop_type == "cancelled" and not result.token_ids:
                raise CancelledBeforeReply("Cancelled before the reply started.", result)
        except (ContextError, BackendError, ConfigError) as e:
            partial = e.partial if isinstance(e, BackendError) else None
            if partial is not None:
                record.update(result_fields(partial)[0])
            record["error"] = f"{type(e).__name__}: {e}"
            record["warnings"] = warnings
            conversation.log.append(record)
            # A failed message names its call, so a resume shows it in /prompt as the chat did.
            failed = {"call_id": call_id} if back_to is None else {}
            conversation.move_head(back_to or user_turn.parent_id, "failed", **failed)
            if back_to is None:
                conversation.last_call = record  # /prompt shows the failed message's call
            raise
        fields, server_prompt = result_fields(result)
        record.update(fields)
        ignored = [e["name"] for e in result.not_applied if e.get("affects_output")]
        if ignored:
            record["flags"].append("not_applied")
            warnings.append(
                f"{self.backend.name} can't apply {ignored} as set, so they had no effect or "
                "ran differently. /settings has the reasons."
            )
        if prompt.special_in_content:
            record["flags"].append("special_tokens_in_content")
        self._check_output(prompt.ids, result, server_prompt, record)
        record["warnings"] = warnings
        # The call record names the reply's turn, and is written before the turn record.
        reply_id = new_id("turn")
        record["turn_id"] = reply_id
        conversation.log.append(record)
        reply = conversation.add_turn(
            "assistant",
            "model",
            result.text,
            user_turn.turn_id,
            turn_id=reply_id,
            call_id=call_id,
            seed=values["seed"],
            regen_of=regen_of,
        )
        conversation.head = reply.turn_id
        conversation.last_call = conversation.calls[call_id] = record
        return CallOutcome(record, result, reply, warnings)

    def _check_output(
        self,
        prompt_ids: list[int],
        result: Result,
        server_prompt: str | None,
        record: dict[str, Any],
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Checks that need the tokenizer: the server's decoding of the prompt, and whether
        the text matches the output IDs."""
        tokenizer = tokenizer or self.tokenizer
        assert tokenizer is not None
        flags, checks = record["flags"], record["checks"]
        if server_prompt is not None:
            expected = tokenizer.decode(prompt_ids)
            ok = server_prompt == expected
            checks["prompt_echo"] = {"ok": ok}
            if not ok:
                checks["prompt_echo"].update(server=server_prompt, client=expected)
                flags.append("check_failed:prompt_echo")
        if not result.text.strip():
            flags.append("empty_reply")
        if result.stop_type in ("eos", "limit"):
            decoded = tokenizer.tok.decode(
                result.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if decoded != result.text:
                # e.g. text llama-server held back as a possible stop string when the limit hit
                flags.append("text_differs_from_ids")
                record["decoded_output_text"] = decoded

    # --- replay ---------------------------------------------------------------------------

    def replay_current(self, cancel: threading.Event | None = None) -> dict[str, Any]:
        """Replay the current reply's call on the running server."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("replay")
        assert reply.call_id is not None
        return self.replay(conversation.log.path, reply.call_id, conversation.log, cancel=cancel)

    def replay(
        self,
        log_path: Path,
        call_id: str,
        log: LogFile | None = None,
        flags_from: str = "",
        found: tuple[dict[str, Any], dict[str, Any] | None] | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Re-send a logged call's token IDs and request, compare the output, list what
        differs from the logged facts, and append a `replay` record to the same log.
        `found`: the call and server records, if the caller already read them. A cancelled
        replay compares and logs nothing."""
        call, server = found or find_call(log_path, call_id)
        name = call["model"]["name"]
        if name not in self.models:
            raise ConfigError(f"Call {call_id} used model {name!r}, which isn't in models.toml.")
        model = self.models[name]
        entry = lock_entry(self.root, model)
        if self.conversation is not None and self.model is not None and self.model.name != name:
            raise ConfigError(
                f"Call {call_id} used {name}, but this conversation uses {self.model.name}. "
                f"Replay it with `clarity replay {log_path} {call_id}`."
            )
        if self.conversation is not None and call["n_ctx"] != self.backend.n_ctx:
            raise ConfigError(
                f"Call {call_id} ran with a context of {call['n_ctx']} tokens, but this "
                f"conversation uses {self.backend.n_ctx}. Replay it with "
                f"`clarity replay {log_path} {call_id}`."
            )
        described = call.get("backend") or {}
        name = described.get("backend", "llama.cpp")
        device = described.get("device") if name == "transformers" else None
        weights = self.weights_for(name, entry)
        steer = self.logged_steering(call, name, weights)
        previous = self.backend
        try:
            backend = self.use_backend(name, device)
            backend.start(model, weights, call["n_ctx"])  # the call's own size, as logged
        except (ConfigError, BackendError) as e:
            self._restore(previous, e)  # as start_model does: the conversation stays usable
            raise
        self.model, self.model_lock, self.weights = model, entry, weights
        tokenizer = self.load_tokenizer(
            call["model"]["tokenizer_repo"], call["model"]["tokenizer_revision"]
        )
        values = {k: v["value"] for k, v in call["settings"].items()}
        result = self.backend.complete(
            call["prompt_ids"],
            values,
            call["stop"],
            request=call["request"],
            cancel=cancel,
            steering=steer,
        )
        if result.stop_type == "cancelled":
            raise ConfigError("Replay cancelled: nothing was compared or logged.")
        fields, server_prompt = result_fields(result)
        self._check_output(call["prompt_ids"], result, server_prompt, fields, tokenizer)
        now_server = self.backend.server_record
        logged_facts = facts_from_call(call, server)
        now_facts = self._facts_now(model, entry)
        diffs = environment_diff(logged_facts, now_facts)
        missing = not_recorded(logged_facts, now_facts)
        comparison = compare_output(call, fields)
        record: dict[str, Any] = {
            "type": "replay",
            "conversation_id": call["conversation_id"],
            "replay_id": new_id("replay"),
            "call_id": call_id,
            "server_id": (now_server or {}).get("server_id"),
            "flags_from": flags_from or "server.toml",
            "n_ctx_from": "the call's log",
            "logged_server_found": server is not None,
            **comparison,
            "unexplained_difference": unexplained(comparison["identical"], diffs),
            "differences": diffs,
            **({"not_recorded": missing} if missing else {}),
            "backend": self.backend.describe(),
            **{
                k: fields[k]
                for k in (
                    "output_text",
                    "output_ids",
                    "stop_type",
                    "stopping_word",
                    "flags",
                    "checks",
                    "request",
                    "backend_settings",
                    "not_applied",
                    "tokens_evaluated",
                    "tokens_predicted",
                    "timings",
                )
            },
            **{k: fields[k] for k in ("decoded_output_text", "logprobs") if k in fields},
        }
        if not comparison["identical"]:
            record["logged_output_text"] = call["output_text"]
        log = log or LogFile(log_path)
        log.on_append = self._appended
        if now_server and now_server["server_id"] not in log.servers_logged:
            log.append({**now_server, "conversation_id": call["conversation_id"]})
            log.servers_logged.add(now_server["server_id"])
        return log.append(record)

    def _facts_now(self, model: ModelEntry, entry: dict[str, Any]) -> dict[str, Any]:
        """The facts a replay or resume compares, as they are now."""
        return facts_from_call(
            {
                "model": {
                    "gguf_sha256": entry["gguf"]["sha256"],
                    "repo": model.repo,
                    "revision": model.revision,
                },
                "client": client_git(),
                "libraries": library_versions(),
            },
            self.backend.server_record,
        )

    # --- resume ---------------------------------------------------------------------------

    def log_for(self, conversation: str) -> Path:
        """A conversation's log: a path to a file, or its ID (from /chats) in logs/."""
        path = Path(conversation)
        by_id = self.logs_dir / f"{conversation}.jsonl"
        return by_id if not path.is_file() and by_id.is_file() else path

    def resume(self, path: Path) -> dict[str, Any]:
        """Continue a logged conversation, appending to its log. It uses the profile, the
        resolved settings and the context size as logged, not today's files, and starts
        where the log says it was. Returns the `resume` record, which says what differs
        now: the profile file, settings today's files would give, the machine and
        libraries, and whether the current reply's prompt still renders to the same IDs."""
        if not path.is_file():
            raise ConfigError(f"No conversation or log file {path}. /chats lists them.")
        try:
            return self._resume(path)
        except (KeyError, TypeError, AttributeError) as e:  # a log Clarity didn't write
            raise ConfigError(f"{path} can't be resumed: a record lacks or mangles {e}.") from e

    def _resume(self, path: Path) -> dict[str, Any]:
        records = list(read_records(path))
        state = rebuild(records, str(path))
        start = state.start
        prepared = self.prepare(start["profile"]["name"], logged=start["profile"])
        profile_file, now = self._profile_now(start["profile"])
        today = {k: s.to_json() for k, s in (now or prepared).profile.settings.items()}
        # A setting the log doesn't have (one added since) comes from the logged profile
        # resolved today, and is listed.
        filled = {k: s.to_json() for k, s in prepared.profile.settings.items()}
        filled = {k: v for k, v in filled.items() if k not in start["settings"]}
        logged = {**filled, **start["settings"]}
        prepared.profile.settings = {k: Setting(v["value"], v["origin"]) for k, v in logged.items()}
        last = state.last_call
        n_ctx = last["n_ctx"] if last else (start.get("ctx_size") or {}).get("value")
        ctx_size = Setting(n_ctx, "log") if n_ctx else self.ctx_size_for(prepared.profile)
        chat = {k: check_setting(k, v, f"{path}: /set") for k, v in state.chat_settings.items()}
        merged = {**prepared.profile.settings, **{k: Setting(v, "chat") for k, v in chat.items()}}
        check_samplers(merged, f"{path}: /set")
        steered = prepared.profile.steering is not None
        check_backend_settings(merged, prepared.profile.backend, f"{path}: /set", steered)
        # Everything from the log is built before the model starts, so a log that can't be
        # read leaves the current conversation and its model as they were.
        log = LogFile(path)
        conversation = Conversation(
            start["conversation_id"],
            prepared.profile,
            log,
            calls=state.calls,
            chat_settings=chat,
            ctx_size=ctx_size,
            resumed=True,
            # The settings are the logged ones, so their source is too.
            # Failed calls too: a chat whose only call failed still logged them.
            logged_recommended=next(
                (
                    r["recommended"]
                    for r in reversed(records)
                    if r.get("type") == "call" and r.get("recommended")
                ),
                None,
            ),
        )
        for t in state.turns:
            call = state.calls.get(t.get("call_id") or "")
            seed = call["settings"]["seed"]["value"] if call else None
            conversation.turns[t["turn_id"]] = Turn(
                t["turn_id"], t["parent_id"], t["role"], t["origin"], t["text"],
                t.get("call_id"), seed, t.get("regen_of"), t.get("edit_of"),
            )  # fmt: skip
        conversation.head = state.head
        conversation.warnings += self._profile_warnings(prepared)
        head = conversation.turns.get(state.head) if state.head else None
        conversation.last_call = state.failed_call or (
            state.calls.get(head.call_id or "") if head else None
        )
        # Checks that read the log or the disk run before the model starts: a failure
        # after it would leave the running conversation on another model or size.
        render = self.check_render(conversation, head, prepared)
        if not os.access(path, os.W_OK):
            raise ConfigError(f"Clarity can't write to {path}, so it can't continue it there.")
        conversation.started_server = self.start_model(prepared, ctx_size)
        environment = missing = None
        if last:
            server = state.servers.get((last.get("backend") or {}).get("server_id", ""))
            then = facts_from_call(last, server)
            now = self._facts_now(prepared.model, prepared.lock)
            environment, missing = environment_diff(then, now), not_recorded(then, now)
        log.on_append = self._appended
        log.servers_logged = set(state.servers)
        self.conversation = conversation
        record = log.append(
            {
                "type": "resume",
                "conversation_id": conversation.conversation_id,
                "head": state.head,
                "head_guessed": state.head_guessed,
                "warnings": conversation.warnings,
                "unanswered": state.unanswered,
                "ctx_size": ctx_size.to_json(),
                "profile_file": profile_file,
                "model_entry": deep_diff(start["model"], prepared.model.to_json()),
                "settings_differ_today": [
                    *settings_diff(start["settings"], today),
                    *({"name": k, "logged": None, "today": v} for k, v in filled.items()),
                ],
                "chat_settings": chat,
                "compared_with_call": last["call_id"] if last else None,
                "differences": environment,
                "not_recorded": missing,
                "render": render,
            }
        )
        self._log_server_start()
        return record

    def _profile_now(self, logged: dict[str, Any]) -> tuple[str, Prepared | None]:
        """The profile's file today (at its logged path, else by name in this project),
        prepared, and whether it matches the logged text: same, changed or missing. A log is
        untrusted, so a path or name that leads outside the project folder isn't read."""
        path = Path(logged["path"])
        name = str(path) if path.is_file() else logged["name"]
        found = path if path.is_file() else profile_path(self.root, name)
        if not found.resolve().is_relative_to(self.root.resolve()):
            return "outside the project folder", None
        try:
            now = self.prepare(name)
        except ConfigError:  # gone, or no longer loads
            return ("changed" if profile_path(self.root, name).is_file() else "missing"), None
        same = (now.profile.toml_text, now.profile.files) == (logged["toml"], logged["files"])
        return "same" if same else "changed", now

    def check_render(
        self, conversation: Conversation, reply: Turn | None, prepared: Prepared | None = None
    ) -> dict[str, Any] | None:
        """Render the prompt of `reply`'s call again, with today's template and libraries,
        and compare the token IDs with the logged ones. None if there's no reply."""
        call = conversation.calls.get(reply.call_id or "") if reply else None
        if reply is None or call is None:
            return None
        assert reply.parent_id is not None
        try:
            user_turn = conversation.turns[reply.parent_id]
            ids = self.build_prompt(user_turn, conversation, prepared).ids
        except ConfigError as e:
            return {"call_id": call["call_id"], "identical": False, "error": str(e)}
        logged = call["prompt_ids"]
        result: dict[str, Any] = {"call_id": call["call_id"], "identical": ids == logged}
        if ids != logged:
            result.update(
                n_ids={"logged": len(logged), "now": len(ids)},
                first_different_token=first_difference(logged, ids),
            )
        return result

    def _appended(self, record: dict[str, Any]) -> None:
        if self.on_record:
            self.on_record(record)

    def _require_conversation(self) -> Conversation:
        if self.conversation is None:
            raise ConfigError("No conversation is running. Start one with /profile NAME.")
        return self.conversation


MARK_DEFAULT = "interesting"


ANNOTATION_TYPES = ("mark", "unmark", "note", "torn_line")


def annotation_line(raw: bytes) -> bool:
    """A `read_records` filter for the records `marks_and_notes` reads. Call records, the
    bulk of a long log, are passed over unread. A damaged line that isn't one of these
    is passed over too, as the chat list does; resume and replay still read every line."""
    return any(f'"type": "{kind}"'.encode() in raw for kind in ANNOTATION_TYPES)


def marks_and_notes(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Each reply's current marks and notes, by turn ID: {"marks": [...], "notes": [...]}.
    An `unmark` removes the mark it names (or, for a mark from before mark IDs, the ID-less
    marks with its label on that reply). A note hides only the note it `replaces`, so notes
    from before note IDs all show."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        kind = record.get("type")
        if kind not in ("mark", "unmark", "note"):
            continue
        current = out.setdefault(record["turn_id"], {"marks": [], "notes": []})
        if kind == "mark":
            current["marks"].append(record)
        elif kind == "unmark":
            current["marks"] = [m for m in current["marks"] if not _unmarks(record, m)]
        else:
            replaced = record.get("replaces")
            notes = [n for n in current["notes"] if not replaced or n.get("note_id") != replaced]
            current["notes"] = [*notes, record]
    return out


def _unmarks(unmark: dict[str, Any], mark: dict[str, Any]) -> bool:
    if unmark.get("mark_id"):
        return mark.get("mark_id") == unmark["mark_id"]
    return not mark.get("mark_id") and mark.get("label") == unmark.get("label")


def annotate_call(
    log: LogFile, about: dict[str, Any], kind: str, text: str, origin: str
) -> list[dict[str, Any]]:
    """Append a `mark` (label, default "interesting"), `unmark` (one per mark removed: those
    with the label, or the only label there is) or `note` (replacing the reply's current
    note) about one reply. `about` has its conversation_id, call_id and turn_id. `origin`
    is "live" (in a chat) or "review" (`clarity mark`, `clarity note` afterwards)."""
    current = marks_and_notes(read_records(log.path, annotation_line)).get(about["turn_id"], {})
    base = {"type": kind, **about, "origin": origin}
    if kind == "mark":
        return [log.append({**base, "mark_id": new_id("mark"), "label": text or MARK_DEFAULT})]
    if kind == "note":
        if not text:
            raise ConfigError("A note needs a text.")
        notes = [n for n in current.get("notes", []) if n.get("note_id")]
        replaces = notes[-1]["note_id"] if notes else None
        note = {**base, "note_id": new_id("note"), "text": text, "replaces": replaces}
        return [log.append(note)]
    marks = current.get("marks", [])
    labels = sorted({m["label"] for m in marks})
    if text:
        marks = [m for m in marks if m["label"] == text]
    elif len(labels) > 1:
        raise ConfigError(f"This reply has marks {', '.join(labels)}. Name the one to remove.")
    if not marks:
        raise ConfigError(f"This reply has no mark {text}." if text else "This reply has no mark.")
    # One unmark per mark; for marks from before mark IDs, one per label covers them all.
    removals = {m.get("mark_id") or f"label:{m['label']}": m for m in marks}.values()
    return [
        log.append({**base, "mark_id": m.get("mark_id"), "label": m["label"]}) for m in removals
    ]


def annotate_log(
    path: Path,
    kind: str,
    call_id: str,
    text: str,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """A review-time mark, unmark or note on a logged call. Starts no server, reads no
    model, and only appends to the log. `on_record` sees each record appended."""
    if not path.is_file():
        raise ConfigError(f"Log file {path} not found.")
    call = None
    wanted = json.dumps({"call_id": call_id}, ensure_ascii=False)[1:-1].encode()  # as logged

    def parse(raw: bytes) -> bool:
        return wanted in raw and b'"type": "call"' in raw or b'"type": "torn_line"' in raw

    for record in read_records(path, parse):
        if record.get("type") == "call" and record.get("call_id") == call_id:
            call = record
    if call is None:
        raise ConfigError(f"{path} has no call {call_id}.")
    if not call.get("turn_id"):
        raise ConfigError(f"Call {call_id} failed, so it has no reply to {kind}.")
    about = {k: call[k] for k in ("conversation_id", "call_id", "turn_id")}
    log = LogFile(path)
    log.on_append = on_record
    return annotate_call(log, about, kind, text, "review")


def prompt_turns(conversation: Conversation, user_turn: Turn) -> list[Turn]:
    """The turns a prompt for `user_turn` holds: its path, without the profile's priming
    (which the profile renders itself)."""
    return [t for t in conversation.path_to(user_turn.turn_id) if t.origin != "priming"]


def result_fields(result: Result) -> tuple[dict[str, Any], str | None]:
    """A Result as call-record fields, plus the server's decoding of the prompt (if any)."""
    response = dict(result.response)
    server_prompt = response.pop("prompt", None)
    fields = {
        "backend_settings": result.backend_settings,
        "not_applied": result.not_applied,
        "request": result.request,
        "output_text": result.text,
        "output_ids": result.token_ids,
        "stop_type": result.stop_type,
        **({"cancelled_during": result.cancelled_during} if result.cancelled_during else {}),
        "stopping_word": result.stopping_word,
        "tokens_evaluated": result.n_prompt,
        "tokens_predicted": result.n_predicted,
        "tokens_cached": response.get("tokens_cached"),
        "truncated": response.get("truncated"),
        "generation_settings": response.get("generation_settings"),
        "timings": result.timings,
        "response": {k: v for k, v in response.items() if k != "generation_settings"},
        "flags": list(result.flags),
        "checks": dict(result.checks),
        **({"logprobs": result.logprobs} if result.logprobs is not None else {}),
    }
    return fields, server_prompt
