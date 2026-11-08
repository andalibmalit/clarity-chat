"""Conversations, turns and calls.

A conversation is a tree of turns (each turn records its parent and origin); `head` is
the turn the next message continues from, so regenerating or editing adds a branch and
never overwrites. A call renders the path from the root to a user turn, tokenizes it,
checks it fits the context, sends the token IDs to the backend, and writes one `call`
record, whether it succeeded, failed or was cancelled.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarity_chat.backends import Backend, BackendError, Result
from clarity_chat.logs import LogFile, client_git, machine, new_id
from clarity_chat.models import check_weights_quick, lock_entry
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
)

CONTEXT_WARN_FRACTION = 0.8


class ContextError(Exception):
    """The prompt plus max_tokens doesn't fit the context; nothing was sent."""


def fits_context(n_prompt: int, max_tokens: int, n_ctx: int) -> bool:
    """With context shift off, llama-server returns at most n_ctx - n_prompt tokens and marks
    the reply truncated once prompt + generated reaches n_ctx, even when that last token was
    also the max_tokens-th. So a call is allowed only if n_prompt + max_tokens < n_ctx; then
    the server never marks it truncated. A live test checks both sides of the boundary."""
    return n_prompt + max_tokens < n_ctx


@dataclass
class Turn:
    turn_id: str
    parent_id: str | None
    role: str  # user | assistant
    origin: str  # user | model | edit | priming
    text: str
    call_id: str | None = None
    seed: int | None = None


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
    servers_logged: set[str] = field(default_factory=set)
    last_call: dict[str, Any] | None = None

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
            }
        )
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
        if backend is None:
            from clarity_chat.backends.llama_cpp import LlamaCppBackend

            backend = LlamaCppBackend(self.server_config, self.logs_dir)
        self.backend = backend
        self.conversation: Conversation | None = None
        self.model: ModelEntry | None = None
        self.model_lock: dict[str, Any] = {}
        self.tokenizer: Tokenizer | None = None
        self.template_tokenizer: Tokenizer | None = None
        self.template: dict[str, Any] | None = None
        self._tokenizers: dict[tuple[str, str], Tokenizer] = {}

    # --- setup ----------------------------------------------------------------------------

    def load_profile(self, name: str) -> Profile:
        return load_profile(self.root, name, self.defaults)

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

    def new_conversation(self, profile_name: str, log_dir: Path | None = None) -> Conversation:
        """Start a conversation with a profile, starting or switching the model as needed."""
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
        weights = check_weights_quick(self.root, entry)
        # Everything that can fail on the profile alone is checked before the server or the
        # session changes, so a bad /new leaves the current conversation usable.
        tokenizer, template_tokenizer = self.tokenizers_for(profile, model)
        template = (
            chat_template_source(template_tokenizer, profile) if profile.mode == "chat" else None
        )
        self.backend.start(model, weights)
        self.model, self.model_lock = model, entry
        self.tokenizer, self.template_tokenizer, self.template = (
            tokenizer,
            template_tokenizer,
            template,
        )
        conversation_id = new_id("conversation")
        log = LogFile((log_dir or self.logs_dir) / f"{conversation_id}.jsonl")
        conversation = Conversation(conversation_id, profile, log)
        log.append(
            {
                "type": "conversation",
                "conversation_id": conversation_id,
                "profile": profile.to_json(),
                "settings": {k: s.to_json() for k, s in profile.settings.items()},
                "model": model.to_json(),
                "client": client_git(),
                "machine": machine(),
            }
        )
        self.conversation = conversation
        self._log_server_start()
        return conversation

    def _log_server_start(self) -> None:
        """Each conversation file starts with the record of the server it uses."""
        conversation = self.conversation
        record = self.backend.server_record
        if conversation is None or record is None:
            return
        if record["server_id"] not in conversation.servers_logged:
            conversation.log.append({**record, "conversation_id": conversation.conversation_id})
            conversation.servers_logged.add(record["server_id"])

    def close(self) -> None:
        self.backend.close()

    # --- calls ----------------------------------------------------------------------------

    def say(self, text: str, on_text: Callable[[str], None] | None = None) -> CallOutcome:
        """Add a user turn after `head` and get the model's reply."""
        conversation = self._require_conversation()
        user = conversation.add_turn("user", "user", text, conversation.head)
        conversation.head = user.turn_id
        return self.call(user, on_text=on_text)

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
            if not fits_context(len(prompt.ids), max_tokens, n_ctx):
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
            result = self.backend.complete(prompt.ids, values, profile.stop, on_text=on_text)
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
        if prompt.special_in_content:
            record["flags"].append("special_tokens_in_content")
        self._check_output(prompt, result, server_prompt, record)
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
        )
        conversation.head = reply.turn_id
        conversation.last_call = record
        return CallOutcome(record, result, reply, warnings)

    def _check_output(
        self, prompt: Prompt, result: Result, server_prompt: str | None, record: dict[str, Any]
    ) -> None:
        """Checks that need the tokenizer: the server's decoding of the prompt, and whether
        the text matches the output IDs."""
        assert self.tokenizer is not None
        flags, checks = record["flags"], record["checks"]
        if server_prompt is not None:
            expected = self.tokenizer.decode(prompt.ids)
            ok = server_prompt == expected
            checks["prompt_echo"] = {"ok": ok}
            if not ok:
                checks["prompt_echo"].update(server=server_prompt, client=expected)
                flags.append("check_failed:prompt_echo")
        if not result.text.strip():
            flags.append("empty_reply")
        if result.stop_type in ("eos", "limit"):
            decoded = self.tokenizer.tok.decode(
                result.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if decoded != result.text:
                # e.g. text llama-server held back as a possible stop string when the limit hit
                flags.append("text_differs_from_ids")
                record["decoded_output_text"] = decoded

    def _require_conversation(self) -> Conversation:
        if self.conversation is None:
            raise ConfigError("No conversation yet.")
        return self.conversation


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
