"""Conversations, turns and calls.

A conversation is a tree of turns (each turn records its parent and origin); `head` is
the turn the next message continues from, so regenerating or editing adds a branch and
never overwrites. A call renders the path from the root to a user turn, tokenizes it,
checks it fits the context, sends the token IDs to the backend, and writes one `call`
record, whether it succeeded, failed or was cancelled.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarity_chat.backends import Backend, BackendError, Result
from clarity_chat.logs import LogFile, client_git, machine, new_id, read_records
from clarity_chat.models import check_weights_quick, lock_entry, source_weights
from clarity_chat.profiles import (
    SEED_MAX,
    ConfigError,
    ModelEntry,
    Profile,
    ServerConfig,
    Setting,
    load_defaults,
    load_models,
    load_profile,
    load_server_config,
    resolve_settings,
)
from clarity_chat.render import (
    Prompt,
    Tokenizer,
    build_prompt,
    chat_template_source,
    library_versions,
    unread_kwargs,
)
from clarity_chat.replay import (
    compare_output,
    environment_diff,
    facts_from_call,
    find_call,
    not_recorded,
    unexplained,
)

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
    turns: dict[str, Turn] = field(default_factory=dict)
    head: str | None = None
    run: dict[str, Any] | None = None  # set for a batch run's conversations
    started_server: dict[str, Any] | None = None  # the server_start, if this one started it
    last_call: dict[str, Any] | None = None  # the call of the reply at head (for /prompt)
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)  # call records by ID
    warnings: list[str] = field(default_factory=list)  # shown when it starts, and logged

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
    ) -> None:
        self.root = root
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

                backend = LlamaCppBackend(self.server_config, self.logs_dir)
            self.backends[name, device] = backend
        return self.backends[name, device]

    def use_server_flags(self, flags: dict[str, Any]) -> None:
        """Start llama-server with these flags instead of server.toml's (replay --logged-flags)."""
        import dataclasses

        from clarity_chat.backends.llama_cpp import LlamaCppBackend

        self.close()
        self.server_config = dataclasses.replace(self.server_config, flags=flags)
        backend = LlamaCppBackend(self.server_config, self.logs_dir)
        self.backends["llama.cpp", None] = self.backend = backend

    def use_backend(self, name: str, device: str | None) -> Backend:
        """Make a backend the active one, closing the other first (one model in memory)."""
        backend = self.backend_for(name, device)
        if backend is not self.backend:
            self.backend.close()
            self.backend = backend
        return backend

    def _load_tokenizer(self, repo: str, revision: str) -> Tokenizer:
        if (repo, revision) not in self._tokenizers:
            self._tokenizers[repo, revision] = self.tokenizer_loader(repo, revision)
        return self._tokenizers[repo, revision]

    def tokenizers_for(self, profile: Profile, model: ModelEntry) -> tuple[Tokenizer, Tokenizer]:
        """The model's tokenizer, which always tokenizes, and the tokenizer whose chat
        template renders chat mode (the profile's template_repo, else the model's)."""
        tokenizer = self._load_tokenizer(model.tokenizer_repo, model.tokenizer_revision)
        if profile.mode == "chat" and profile.template_repo and profile.template_revision:
            return tokenizer, self._load_tokenizer(profile.template_repo, profile.template_revision)
        return tokenizer, tokenizer

    def prepare(self, profile_name: str) -> Prepared:
        """Load a profile and everything it needs short of starting the model: its model
        entry and lock, the weights (size and time only), tokenizers and chat template."""
        profile = self.load_profile(profile_name)
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
        previous = self.backend
        ctx_size = self.ctx_size_for(prepared.profile)
        try:
            backend = self.use_backend(prepared.profile.backend, prepared.profile.device)
            started = backend.start(model, prepared.weights, ctx_size.value)
        except (ConfigError, BackendError) as e:
            self._restore(previous, e)
            raise
        self.model, self.model_lock, self.weights = model, prepared.lock, prepared.weights
        self.tokenizer = prepared.tokenizer
        self.template_tokenizer = prepared.template_tokenizer
        self.template = prepared.template
        conversation_id = new_id("conversation")
        log = log or LogFile(self.logs_dir / f"{conversation_id}.jsonl")
        conversation = Conversation(
            conversation_id, prepared.profile, log, run=run, started_server=started
        )
        unread = prepared.unread_kwargs
        if unread:
            conversation.warnings.append(
                f"The chat template never reads template_kwargs {unread}, so they change "
                "nothing. Check their names against the template (/prompt shows its output)."
            )
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

    def _restore(self, previous: Backend, error: Exception) -> None:
        """After a failed switch, restart the current conversation's model, so a bad /new or
        /profile leaves that conversation usable. If that fails too, no conversation runs."""
        if self.conversation is None or self.model is None or self.weights is None:
            return
        if self.backend is not previous:
            self.backend.close()
            self.backend = previous
        try:
            ctx_size = self.ctx_size_for(self.conversation.profile).value
            previous.start(self.model, self.weights, ctx_size)
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

    def regen(
        self, on_text: Callable[[str], None] | None = None, cancel: threading.Event | None = None
    ) -> CallOutcome:
        """Another reply to the current message, as a new branch. Its seed is the first
        reply's seed plus k, where k counts this and earlier regenerations of that message."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("regenerate")
        assert reply.parent_id is not None
        user = conversation.turns[reply.parent_id]
        replies = conversation.replies_to(user.turn_id)
        first = next((t for t in replies if t.regen_of is None), replies[0])
        if first.seed is None or first.call_id is None:
            raise ConfigError("The first reply to this message has no seed to start from.")
        seed = first.seed + 1 + sum(t.regen_of is not None for t in replies)
        if seed > SEED_MAX:
            raise ConfigError(
                f"The next regeneration seed, {seed}, is above {SEED_MAX}. Edit the message or "
                "start a new conversation instead."
            )
        try:
            return self.call(
                user,
                seed=Setting(seed, "regen"),
                regen_of=first.call_id,
                on_text=on_text,
                cancel=cancel,
            )
        except (ContextError, BackendError, ConfigError):
            conversation.head = reply.turn_id  # a failed regeneration leaves you where you were
            conversation.last_call = conversation.calls.get(reply.call_id or "")
            raise

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
        try:
            return self.call(user, on_text=on_text, cancel=cancel)
        except (ContextError, BackendError, ConfigError):
            conversation.head = reply.turn_id  # a failed edit leaves you where you were
            conversation.last_call = conversation.calls.get(reply.call_id or "")
            raise

    def undo(self) -> Turn | None:
        """Move back past the last exchange. Returns the new head (None at the start). The
        undone turns stay in the log; the next message branches from the new head."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("undo")
        assert reply.parent_id is not None
        conversation.head = conversation.turns[reply.parent_id].parent_id
        head = conversation.turns.get(conversation.head) if conversation.head else None
        conversation.last_call = conversation.calls.get(head.call_id or "") if head else None
        return head

    def annotate(self, kind: str, text: str) -> dict[str, Any]:
        """A `mark` (label) or `note` (text) record on the current reply's call."""
        conversation = self._require_conversation()
        reply = conversation.current_reply(kind)
        return conversation.log.append(
            annotation(
                kind,
                conversation.conversation_id,
                reply.call_id,
                reply.turn_id,
                text,
                "live",
            )
        )

    def build_prompt(self, user_turn: Turn) -> Prompt:
        conversation = self._require_conversation()
        assert self.tokenizer is not None
        path = conversation.path_to(user_turn.turn_id)
        turns = [(t.role, t.text) for t in path if t.origin != "priming"]
        return build_prompt(self.tokenizer, conversation.profile, turns, self.template_tokenizer)

    @staticmethod
    def resolve_settings(
        profile: Profile, seed: Setting | None = None
    ) -> tuple[dict[str, Setting], bool]:
        """The profile's settings for one call with a concrete seed, and whether it was drawn."""
        settings = dict(profile.settings)
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
    ) -> CallOutcome:
        conversation = self._require_conversation()
        profile = conversation.profile
        assert self.model is not None and self.tokenizer is not None
        self._log_server_start()
        call_id = new_id("call")
        settings, seed_drawn = self.resolve_settings(profile, seed)
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
            "recommended": self.model_lock.get("recommended") if profile.use_recommended else None,
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
                result = Result(text="", token_ids=[], stop_type="cancelled")
            else:
                result = self.backend.complete(
                    prompt.ids, values, profile.stop, on_text=on_text, cancel=cancel
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
            conversation.last_call = record
            conversation.head = user_turn.parent_id  # the failed message becomes a dead branch
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

    def replay_current(self) -> dict[str, Any]:
        """Replay the current reply's call on the running server."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("replay")
        assert reply.call_id is not None
        return self.replay(conversation.log.path, reply.call_id, conversation.log)

    def replay(
        self,
        log_path: Path,
        call_id: str,
        log: LogFile | None = None,
        flags_from: str = "",
        found: tuple[dict[str, Any], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        """Re-send a logged call's token IDs and request, compare the output, list what
        differs from the logged facts, and append a `replay` record to the same log.
        `found`: the call and server records, if the caller already read them."""
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
        backend = self.use_backend(name, device)
        backend.start(model, weights, call["n_ctx"])  # the call's own size, as logged
        self.model, self.model_lock, self.weights = model, entry, weights
        tokenizer = self._load_tokenizer(
            call["model"]["tokenizer_repo"], call["model"]["tokenizer_revision"]
        )
        values = {k: v["value"] for k, v in call["settings"].items()}
        result = self.backend.complete(
            call["prompt_ids"], values, call["stop"], request=call["request"]
        )
        if result.stop_type == "cancelled":
            raise ConfigError("Replay cancelled (Ctrl-C): nothing was compared or logged.")
        fields, server_prompt = result_fields(result)
        self._check_output(call["prompt_ids"], result, server_prompt, fields, tokenizer)
        now_server = self.backend.server_record
        logged_facts = facts_from_call(call, server)
        now_facts = facts_from_call(
            {
                "model": {
                    "gguf_sha256": entry["gguf"]["sha256"],
                    "repo": model.repo,
                    "revision": model.revision,
                },
                "client": client_git(),
                "libraries": library_versions(),
            },
            now_server,
        )
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
        }
        if not comparison["identical"]:
            record["logged_output_text"] = call["output_text"]
        log = log or LogFile(log_path)
        if now_server and now_server["server_id"] not in log.servers_logged:
            log.append({**now_server, "conversation_id": call["conversation_id"]})
            log.servers_logged.add(now_server["server_id"])
        return log.append(record)

    def _require_conversation(self) -> Conversation:
        if self.conversation is None:
            raise ConfigError("No conversation is running. Start one with /profile NAME.")
        return self.conversation


def annotation(
    kind: str, conversation_id: str, call_id: str | None, turn_id: str, text: str, origin: str
) -> dict[str, Any]:
    """A `mark` (label) or `note` (text) record. `origin` is "live" (/mark, /note in a
    chat) or "review" (`clarity mark`, `clarity note` on a log afterwards)."""
    return {
        "type": kind,
        "conversation_id": conversation_id,
        "call_id": call_id,
        "turn_id": turn_id,
        "label" if kind == "mark" else "text": text,
        "origin": origin,
    }


def annotate_log(path: Path, kind: str, call_id: str, text: str) -> dict[str, Any]:
    """Append a review-time mark or note on a logged call. Starts no server, reads no
    model, and only appends to the log."""
    if not path.is_file():
        raise ConfigError(f"Log file {path} not found.")
    if not text:
        raise ConfigError(f"A {kind} needs a {'label' if kind == 'mark' else 'text'}.")
    call = None
    for record in read_records(path):
        if record.get("type") == "call" and record.get("call_id") == call_id:
            call = record
    if call is None:
        raise ConfigError(f"{path} has no call {call_id}.")
    if not call.get("turn_id"):
        raise ConfigError(f"Call {call_id} failed, so it has no reply to {kind}.")
    record = annotation(kind, call["conversation_id"], call_id, call["turn_id"], text, "review")
    return LogFile(path).append(record)


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
    }
    return fields, server_prompt
