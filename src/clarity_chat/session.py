"""Conversations, turns and calls.

A conversation is a tree of turns (each turn records its parent and origin); `head` is
the turn the next message continues from, so regenerating or editing adds a branch and
never overwrites. A call renders the path from the root to a user turn, tokenizes it,
checks it fits the context, sends the token IDs to the backend, and writes one `call`
record, whether it succeeded, failed or was cancelled.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jinja2

from clarity_chat import steering
from clarity_chat.backends import Backend, BackendError, Result
from clarity_chat.documents import (
    EMBEDDER_MODEL,
    SENTENCES,
    TURN_FIELDS,
    Attached,
    Document,
    DocumentRefused,
    Embedder,
    EmbedderFailed,
    Fit,
    Loaded,
    PartTooLong,
    Room,
    SearchRefused,
    Stale,
    TooLong,
    Unreadable,
    added_ranges,
    attached_in,
    checked_attachments,
    checked_library_name,
    content,
    control_token_problem,
    counted,
    covers_whole,
    fill,
    fit,
    found_in,
    framing_of,
    library_manifest,
    library_parts,
    library_parts_stamp,
    library_state,
    load_document,
    meaning_search,
    message_content,
    more_than,
    piece,
    scan,
    search_documents,
    start_that_fits,
    summaries,
    vectors_for,
    with_text,
)
from clarity_chat.logs import LogFile, client_git, machine, new_id, read_records, unreadable
from clarity_chat.models import check_weights_quick, lock_entry, read_lock, source_weights
from clarity_chat.profiles import (
    BUDGET,
    MIN_SIMILARITY,
    SEARCH_METHOD,
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
    fetch_hint,
    library_name_problem,
    load_defaults,
    load_models,
    load_profile,
    load_server_config,
    model_files,
    named_twice,
    parse_setting,
    profile_path,
    resolve_settings,
    shown,
    steering_path,
)
from clarity_chat.render import (
    Prompt,
    Tokenizer,
    build_prompt,
    chat_template_source,
    library_versions,
    message_spans,
    prompt_text,
    unread_kwargs,
    writes_date,
)
from clarity_chat.replay import (
    REPLAY_NEEDS,
    compare_output,
    deep_diff,
    environment_diff,
    facts_from_call,
    find_call,
    first_difference,
    model_for_call,
    no_model_for_call,
    not_recorded,
    unexplained,
)
from clarity_chat.resume import rebuild, settings_diff, what_differs

CONTEXT_WARN_FRACTION = 0.8


class ContextError(Exception):
    """The prompt plus max_tokens doesn't fit the context; nothing was sent."""


class DoesntFit(DocumentRefused):
    """An attachment that doesn't fit the free context. `start`: for a whole file, the
    largest k whose units 1 to k fit, or None if not even the first does."""

    def __init__(self, sentence: str, start: int | None = None, roomy: int | None = None) -> None:
        super().__init__(sentence)
        self.start = start
        self.roomy = roomy  # the largest start that still leaves max_tokens: the sheet's


class HEAD:
    """Room's default: count after the conversation's head."""


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
    attachments: list[dict[str, Any]] = field(default_factory=list)  # a user turn's (v0.3.1)
    framing: str | None = None
    framing_origin: str | None = None
    placement: str | None = None
    found: dict[str, Any] | None = None  # a searched message's (v0.3.2)

    @property
    def content(self) -> str:
        """The message as the model gets it: with its block, if any."""
        return message_content(self.text, self.attached_fields())

    def attached_fields(self) -> dict[str, Any]:
        """The turn record's block fields: only for a turn with attachments or a search."""
        if not self.attachments and self.found is None:
            return {}
        fields = {name: getattr(self, name) for name in TURN_FIELDS}
        return {**fields, "found": self.found} if self.found is not None else fields


@dataclass(frozen=True)
class NewChat:
    """What /new would start on, for the check's new-chat numbers (the siblings finder, S3):
    today's profile file, its tokenizers, and the context the new chat gets."""

    profile: Profile
    tokenizer: Tokenizer
    template: Tokenizer
    n_ctx: int
    key: str  # what the counts depend on, for the check's cache


@dataclass(frozen=True)
class Counts:
    """A check's counts that don't depend on the range asked for, kept for the next check."""

    before: int  # the prompt's tokens with what's waiting
    whole: Fit
    new_chat: Fit | None  # None: the profile /new would start can't be read
    start: tuple[int, Fit] | None  # the start that fits
    roomy: tuple[int, Fit] | None  # the start that still leaves max_tokens


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
    writes_date: bool = False  # the chat template can write today's date


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
    # What differs today, from the resume record (what_differs), until the next call.
    resume_differ: dict[str, Any] | None = None
    unanswered: dict[str, Any] | None = None  # your last message, if its call failed
    # The model's recommended values the settings came from, as logged; None for a new
    # conversation, whose calls log today's models.lock.
    logged_recommended: dict[str, Any] | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)  # attachments waiting
    # The documents a message's search reads (v0.3.2): the profile's list until /search
    # changes it ("chat"), logged as `search` records and resumed.
    searching: list[str] = field(default_factory=list)
    searching_origin: str = "profile"
    search_method: str = SEARCH_METHOD  # the whole-branch review's could 2

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
                **turn.attached_fields(),
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

    def branch_ends(self, notes: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
        """Every branch end (a reply with no newer reply after it), newest first, and every
        marked reply, newest mark first. `notes`: marks_and_notes of the log. Each says
        where its branch left the rest: `from`, the deepest fork on its path."""
        newest = self.newest_replies()
        order = {turn_id: i for i, turn_id in enumerate(self.turns)}
        current_end = newest.get(self.head) if self.head else None
        on_head_path = {t.turn_id for t in self.path_to(self.head)}
        children: dict[str | None, list[Turn]] = {}  # built once: forks are looked up often
        for t in self.turns.values():
            if t.turn_id in newest:  # as forks() does: a message with no reply isn't a branch
                children.setdefault(t.parent_id, []).append(t)

        def came_from(turn: Turn) -> tuple[dict[str, Any] | None, Turn | None]:
            """The deepest fork above `turn` (or at it), and the reply that fork made."""
            path = self.path_to(turn.turn_id)
            for i in range(len(path) - 1, -1, -1):
                here = path[i]
                siblings = children.get(here.parent_id, [])
                if len(siblings) > 1:
                    reply = here if here.role == "assistant" else path[i + 1]
                    return {
                        "kind": "reply" if here.role == "assistant" else "edit",
                        "k": siblings.index(here) + 1,
                        "n": len(siblings),
                        "seed": reply.seed,
                    }, reply
            return None, None

        def marks(turn_id: str) -> list[dict[str, Any]]:
            found = notes.get(turn_id, {}).get("marks", [])
            return [{"mark_id": m.get("mark_id"), "label": m["label"]} for m in found]

        ends = []
        for turn in sorted(self.turns.values(), key=lambda t: -order[t.turn_id]):
            if turn.role != "assistant" or newest.get(turn.turn_id) is not turn:
                continue
            path = [t for t in self.path_to(turn.turn_id) if t.origin != "priming"]
            source, reply = came_from(turn)
            settings = (self.calls.get(turn.call_id or "") or {}).get("settings", {})
            ends.append(
                {
                    "turn_id": turn.turn_id,
                    "call_id": turn.call_id,
                    "text": turn.text,
                    "turns": len(path),
                    "marked": sum(bool(marks(t.turn_id)) for t in path),
                    "seed": turn.seed,
                    "steering_scale": settings.get("steering_scale", {}).get("value"),
                    "from": source,
                    "continued": source is not None and reply is not turn,
                    "current": turn is current_end,
                }
            )
        marked = []
        newest_mark = {  # the time of each reply's newest current mark, as logged
            t: max(m.get("time", "") for m in n["marks"]) for t, n in notes.items() if n["marks"]
        }
        for turn_id in sorted(newest_mark, key=newest_mark.__getitem__, reverse=True):
            if turn_id not in self.turns:
                continue
            turn = self.turns[turn_id]
            marked.append(
                {
                    "turn_id": turn_id,
                    "call_id": turn.call_id,
                    "text": turn.text,
                    "marks": marks(turn_id),
                    "notes": [  # as /history gives them: older logs can hold several
                        {"note_id": n.get("note_id"), "text": n["text"]}
                        for n in notes[turn_id].get("notes", [])
                    ],
                    "from": came_from(turn)[0],
                    "current": turn_id in on_head_path,
                }
            )
        return {"head": self.head, "ends": ends, "marked": marked}

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
        on_server: Callable[..., None] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """`on_server` hears each llama-server's port, PID and key when it starts, and None
        when it stops (the instance lock keeps them, to clean up after a crash); the
        embedder's come with a second argument, "embedder"."""
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
        self._checked: tuple[tuple, Counts] | None = None  # the last check's counts
        # /state's parts reads, by build ID and parts.jsonl's size and time (cut 1)
        self._parts_checked: dict[tuple, DocumentRefused | None] = {}
        self.model: ModelEntry | None = None
        self.model_lock: dict[str, Any] = {}
        self.tokenizer: Tokenizer | None = None
        self.template_tokenizer: Tokenizer | None = None
        self.template: dict[str, Any] | None = None
        self.weights: Path | None = None  # what the current model was started from
        self._tokenizers: dict[tuple[str, str], Tokenizer] = {}
        self._embedder = embedder
        self.on_building: Callable[[str, int, int], None] | None = None
        self.last_closest: float | None = None  # the last meaning search's best similarity

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

    def embedder(self) -> Embedder:
        """bge-m3's server, made on a meaning search's first need and stopped at close (v0.4.1
        decision 3); its first vector starts it."""
        if self._embedder is None:
            on_server = self.on_server  # D4: recorded as "embedder", beside the chat's server
            self._embedder = Embedder(
                self.root,
                self.server_config,
                self.logs_dir,
                self.load_tokenizer,
                (lambda info: on_server(info, "embedder")) if on_server else None,
            )
        return self._embedder

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

    def profile_for(
        self, profile_name: str, logged: dict[str, Any] | None = None
    ) -> tuple[Profile, ModelEntry, dict[str, Any]]:
        """A profile with its settings resolved, its model entry and lock entry: what a new
        chat's numbers need, short of the weights."""
        profile = load_profile(self.root, profile_name, self.defaults, logged)
        if profile.model not in self.models:
            raise ConfigError(
                f"Profile {profile.name}: model {profile.model!r} isn't in models.toml or "
                "local/models.toml."
            )
        if profile.model == EMBEDDER_MODEL:  # its pickle would load unhashed (fix review M1)
            raise ConfigError("bge-m3 is meaning search's model; a profile can't chat with it.")
        model = self.models[profile.model]
        entry = lock_entry(self.root, model)
        if profile.use_recommended:
            recommended = entry.get("recommended")
            if recommended is None:
                raise ConfigError(
                    f"{model_files(self.root, model.name)[1]} has no recommended settings for "
                    f"{model.name}. "
                    f"{fetch_hint(model.name)}"
                )
            profile.settings = resolve_settings(
                profile.raw_settings, self.defaults, str(profile.path), recommended["values"]
            )
        return profile, model, entry

    def prepare(self, profile_name: str, logged: dict[str, Any] | None = None) -> Prepared:
        """Load a profile and everything it needs short of starting the model: its model
        entry and lock, the weights (size and time only), tokenizers and chat template.
        `logged`: a log's `profile` field, to use the profile as it was logged."""
        profile, model, entry = self.profile_for(profile_name, logged)
        weights = self.weights_for(profile.backend, entry)
        if profile.steering is not None:
            profile.vector, profile.layers = self.load_vector(
                profile.steering.file, profile.steering.sha256, profile.steering.layers, weights
            )
        tokenizer, template_tokenizer = self.tokenizers_for(profile, model)
        if profile.mode == "chat":
            template = chat_template_source(template_tokenizer, profile)
            text = template_tokenizer.tok.chat_template
            try:  # each parses the template: refused here, as the first message's would be
                unread = unread_kwargs(text, profile.template_kwargs)
                dated = writes_date(text)
            except jinja2.TemplateError as e:
                raise ConfigError(
                    f"Profile {profile.name}: the chat template failed: {type(e).__name__}: {e}"
                ) from e
        else:
            template, unread, dated = None, [], False
        return Prepared(
            profile, model, entry, weights, tokenizer, template_tokenizer, template, unread, dated
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
        self._profile_documents(prepared.profile)
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
            searching=list(prepared.profile.search),
            search_method=prepared.profile.search_method or SEARCH_METHOD,
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
        warnings = []
        if prepared.unread_kwargs:
            warnings.append(
                f"The chat template never reads template_kwargs {prepared.unread_kwargs}, so "
                "they change nothing. Check their names against the template: the prompt "
                "shows its output."
            )
        if prepared.writes_date:
            warnings.append(
                "This model's chat template can write today's date, so the same messages can "
                "make a different prompt on another day; replay sends the logged token IDs, so "
                "it isn't affected."
            )
        return warnings

    def start_model(self, prepared: Prepared, ctx_size: Setting) -> dict[str, Any] | None:
        """Start (or keep) the model a prepared profile needs, at `ctx_size`. Returns the
        server_start record if a server started. On failure the current conversation's
        model is restarted, so it stays usable (see _restore)."""
        previous = self.backend
        try:
            backend = self.use_backend(prepared.profile.backend, prepared.profile.device)
            started = backend.start(prepared.model, prepared.weights, ctx_size.value)
        except (ConfigError, BackendError) as e:
            self._restore(previous, e, (prepared.model, prepared.weights, ctx_size.value))
            raise
        self.model, self.model_lock, self.weights = prepared.model, prepared.lock, prepared.weights
        self.tokenizer = prepared.tokenizer
        self.template_tokenizer = prepared.template_tokenizer
        self.template = prepared.template
        return started

    def _restore(self, previous: Backend, error: Exception, tried: tuple) -> None:
        """After a failed switch, restart the current conversation's model, so a bad /new or
        /profile leaves that conversation usable. If that fails too, no conversation runs.
        If the start that failed was the current conversation's own (`tried`: model,
        weights and context size, on the same backend), as "Start the model again", a
        Replay or /new on the same profile after a crash make, the same start would fail
        the same way: the conversation stays, its model stopped, to be tried again."""
        if self.conversation is None or self.model is None or self.weights is None:
            return
        own = (self.model, self.weights, self.conversation.ctx_size.value)
        if self.backend is previous and tried == own:
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
                f"too ({again}). Start a conversation with a profile to go on."
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
        if self._embedder is not None:
            self._embedder.close()

    # --- calls ----------------------------------------------------------------------------

    def say(
        self,
        text: str,
        on_text: Callable[[str], None] | None = None,
        seed: Setting | None = None,
        cancel: threading.Event | None = None,
        block: dict[str, Any] | None = None,
    ) -> CallOutcome:
        """Add a user turn after `head` and get the model's reply. `seed` replaces the
        profile's seed for this call (a batch run's seeds). `cancel`: an event the caller
        makes for this call and may set from any thread to stop it. `block`: the message's
        block fields, already found (a batch run searches once per opening and profile)."""
        conversation = self._require_conversation()
        if block is None:
            attached, refused = self._block(text, conversation.pending, conversation.head, True)
        else:
            attached, refused = block, None
        user = conversation.add_turn("user", "user", text, conversation.head, **attached)
        conversation.head, conversation.pending = user.turn_id, []
        return self.call(user, seed=seed, on_text=on_text, cancel=cancel, refused=refused)

    def _block(
        self,
        text: str,
        attachments: list[dict[str, Any]],
        parent: str | None,
        at_send: bool = False,
        where: str = "chat",
    ) -> tuple[dict[str, Any], SearchRefused | None]:
        """A new message's block fields: its attachments and, when the chat searches, what
        its text found in the chat's documents (decisions 2, 3, 9), counted after the turns
        to `parent`. A stale document is a refusal the call logs as failed (decision 6),
        returned rather than raised so the message is logged first; `at_send` heads it
        "Nothing was sent." (a message or an edit, not /try or a run)."""
        conversation = self._require_conversation()
        profile = conversation.profile
        if not attachments and not conversation.searching:
            return {}, None
        framing, origin = framing_of(profile)
        fields: dict[str, Any] = {
            "attachments": attachments,
            "framing": framing,
            "framing_origin": origin,
            "placement": profile.placement,
        }
        if not conversation.searching:
            return fields, None
        already = frozenset(
            p["sha256"] for t in conversation.path_to(parent) for p in found_in(t.attached_fields())
        )
        try:
            room = self.room(after=parent)
            found = self._found(
                text,
                conversation.searching,
                room,
                profile,
                fields,
                already,
                at_send,
                where,
                method=conversation.search_method,
            )
        except SearchRefused as e:
            return fields, e
        return {**fields, "found": found}, None

    def _found(
        self,
        text: str,
        names: list[str],
        room: Room,
        profile: Profile,
        fields: dict[str, Any],
        already: frozenset[str] = frozenset(),
        at_send: bool = False,
        where: str = "chat",
        *,
        method: str,
    ) -> dict[str, Any]:
        """What `text` finds in the documents `names`, filled to `profile`'s budget into a
        block holding `fields`' attachments: one function for a message, /try and a batch
        run (SYS: "one object, three uses"). Every document a search can't read is named in
        one refusal, each with its way out (the board walk's point 6); `at_send` heads it
        "Nothing was sent."; `where` (a chat, a run, `document try`) says how the
        terminal stops searching."""
        problems: list[dict[str, Any]] = []
        docs = []
        for name in names:
            problem = self._unsearchable(name)
            if problem is None:
                try:
                    docs.append(load_document(self.root, name))
                    continue
                except Stale as e:  # gone since the check just now
                    problem = {"document": name, "state": e.state}
                except DocumentRefused as e:  # parts.jsonl isn't the one built (cut 1)
                    problem = unreadable_problem(name, e)
            problems.append(problem)
        if problems:
            raise SearchRefused(problems, at_send, where)
        budget, origin = budget_of(profile)
        if method == "meaning":
            found = self._meaning(docs, text, profile, at_send, where)
        else:
            try:
                found = search_documents(docs, text)
            except DocumentRefused:  # no FTS5 in this Python: how to send without it
                raise SearchRefused([{"state": "no_fts5"}], at_send, where) from None
        return fill(
            room,
            text,
            attached_in(fields),
            fields["framing"],
            found,
            budget,
            origin,
            fields["placement"],
            already,
        )

    def _meaning(
        self, docs: list[Loaded], text: str, profile: Profile, at_send: bool, where: str
    ) -> dict[str, Any]:
        """What `text` finds in `docs` by meaning (decisions 9 to 12), unfilled: the
        embedder started, each document's vectors read or built, the message's vector,
        and the parts at or above the minimum. States 5 to 7 are refused as problems; a
        vectors file Clarity can't write is said as it is."""

        def refused(problem: dict[str, Any]) -> SearchRefused:
            return SearchRefused([problem], at_send, where)

        embedder = self.embedder()
        if not embedder.fetched():  # state 5, a file check: before any process
            raise refused({"state": "no_embedder"})
        try:
            engine = embedder.start()
            ids = embedder.ids(text)
        except TooLong as e:
            raise refused({"state": "too_long", "tokens": e.tokens}) from None
        except EmbedderFailed as e:
            raise refused({"state": "embedder", "why": str(e)}) from None
        except ConfigError as e:  # its file changed since the check, or no tokenizer
            raise refused({"state": "embedder", "why": f"The embedder didn't start: {e}"}) from None
        try:
            vectors = {
                d.name: vectors_for(self.root, d, embedder, engine["setup"], self.on_building)
                for d in docs
            }
            query = embedder.vector(ids)
        except PartTooLong as e:
            keys = ("ordinal", "title", "page")
            raise refused(
                {
                    "document": e.document,
                    "state": "too_long",
                    "tokens": e.tokens,
                    **{k: e.part[k] for k in keys},
                }  # fmt: skip
            ) from None
        except EmbedderFailed as e:
            raise refused({"state": "embedder", "why": str(e)}) from None
        minimum, origin = minimum_of(profile)
        # Every part scored once: the best is /try's closest when none reaches the minimum.
        query_vector = memoryview(query).cast("f")
        found_vectors = {name: v for name, (v, _) in vectors.items()}
        scored = meaning_search(docs, found_vectors, query_vector)
        self.last_closest = scored[0]["score"] if scored else None
        return {
            "method": "meaning",
            "searched_for": text,
            "documents": [
                {
                    "name": d.name,
                    "build_id": d.build_id,
                    "original_sha256": d.original_sha256,
                    "vectors_sha256": vectors[d.name][1],
                }  # fmt: skip
                for d in docs
            ],
            "engine": engine,
            "query": {"tokens": len(ids), "sha256": hashlib.sha256(query).hexdigest()},
            "min_similarity": minimum,
            "min_similarity_origin": origin,
            "matches": [m for m in scored if m["score"] >= minimum],
        }

    def _unsearchable(self, name: str) -> dict[str, Any] | None:
        """Why a search can't read `name` (N30, N31, N32), as a problem, or None."""
        state = library_state(self.root, name)
        if state == "ready":
            return None
        if state == "missing":
            return {"document": name, "state": "missing"}
        try:
            library_manifest(self.root, name)
        except DocumentRefused as e:  # unreadable: the reason; each front end its way out
            return unreadable_problem(name, e)
        return None  # ready since the check just now: the search reads it

    def document_builds(self, names: list[str]) -> dict[str, str]:
        """Each document's build ID, as a search now would read it (a stale one refused)."""
        try:
            return {name: load_document(self.root, name).build_id for name in names}
        except Stale as e:
            raise SearchRefused([{"document": e.name, "state": e.state}], where="run") from None
        except Unreadable as e:  # its parts aren't the ones built: add it again (cut 1)
            raise SearchRefused([unreadable_problem(e.name, e)], where="run") from None

    def block(self, text: str, where: str = "chat") -> dict[str, Any]:
        """A next message's block fields, searched now, a refusal raised (/try, and a batch
        run with `where` "run")."""
        conversation = self._require_conversation()
        fields, refused = self._block(text, conversation.pending, conversation.head, where=where)
        if refused is not None:
            raise refused
        return fields

    def try_search(self, text: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """/try: what a message with `text` would find now, with no model call and nothing
        logged: the same object a message's turn would hold (decision 14), and the problem
        Send would refuse it for, a found part's control token, or None (part 1's S4)."""
        conversation = self._require_conversation()
        if not conversation.searching:
            raise ConfigError("This chat searches no documents yet.")
        found = self.block(text)["found"]
        return found, self._control_token_problem(found, self.tokenizer)

    @staticmethod
    def _control_token_problem(found: dict[str, Any], tokenizer: Any) -> dict[str, Any] | None:
        """The first added part holding the tokenizer's control tokens, as Send's check
        finds it, or None."""
        problems = (control_token_problem(part, tokenizer) for part in found["parts"])
        return next((p for p in problems if p is not None), None)

    def try_documents(
        self, names: list[str], profile_name: str, text: str, where: str = "try"
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """`clarity document try`: what a first message with `text` would find in `names` in
        a new chat on `profile_name`, counted with its tokenizer, with no model, and the
        problem Send would refuse it for. A run's pre-check says `where` "run": its names are
        the profile's list."""
        for name in names:
            checked_library_name(name)
        if named_twice(names) is not None:
            raise ConfigError("Name each document once.")
        profile, model, _ = self.profile_for(profile_name)
        if profile.mode != "chat":  # N71, as /search says it
            raise ConfigError(SENTENCES["text_mode_search"].format(profile=profile.name))
        method = profile.search_method or SEARCH_METHOD
        if where == "try" and method == "meaning":  # decision 17: no embedder, no lock
            raise ConfigError(
                "clarity document try searches by words only; /try in a chat searches by meaning."
            )
        self._profile_documents(profile, where)
        tokenizer, template = self.tokenizers_for(profile, model)
        new = NewChat(profile, tokenizer, template, self.ctx_size_for(profile).value, "")
        framing, origin = framing_of(profile)
        fields = {"attachments": [], "framing": framing, "framing_origin": origin}
        found = self._found(
            text,
            names,
            self.room(new),
            profile,
            {**fields, "placement": profile.placement},
            where=where,
            method=method,  # a run's pre-check searches as its messages will
        )
        return found, self._control_token_problem(found, tokenizer)

    def room(
        self,
        new: NewChat | None = None,
        cancel: threading.Event | None = None,
        name: str = "",
        after: str | None | type[HEAD] = HEAD,
    ) -> Room:
        """How to count for the next message: its prompt's tokens for a given content, after
        the chat so far (to `after`, default the head: an edit counts from before the
        message it replaces; or, with `new`, in the chat /new would start), and the free
        context a prompt of that many tokens leaves. With `cancel`, each count first checks
        it, and a set one stops counting `name`."""
        if new is None:
            conversation = self._require_conversation()
            assert self.tokenizer is not None
            tokenizer, template = self.tokenizer, self.template_tokenizer
            parent = conversation.head if after is HEAD else cast("str | None", after)
            profile, turns = conversation.profile, conversation.path_to(parent)
            settings, n_ctx = conversation.settings, self.backend.n_ctx
        else:  # a new chat starts from its profile's settings, not this chat's /set values
            tokenizer, template, profile, turns = new.tokenizer, new.template, new.profile, []
            settings, n_ctx = new.profile.settings, new.n_ctx
        path = [(t.role, t.content) for t in turns if t.origin != "priming"]
        max_tokens = settings["max_tokens"].value

        def tokens(next_content: str) -> int:
            if cancel is not None and cancel.is_set():
                raise DocumentRefused(SENTENCES["stopped_counting"].format(name=name))
            return len(
                build_prompt(tokenizer, profile, [*path, ("user", next_content)], template).ids
            )

        frame = prompt_text(profile, [*path, ("user", "")], template or tokenizer)
        return Room(
            tokens,
            lambda n: room_left(n_ctx, n, max_tokens),
            max_tokens,
            frame_bytes=len(frame.encode("utf-8")),
            token_bytes=tokenizer.max_token_bytes,
        )

    def new_chat(self, profile_name: str) -> NewChat | None:
        """What /new on `profile_name` would start on, read from today's files; None when
        they can't start a chat (/new would refuse, and says why when pressed). "" names none."""
        conversation = self._require_conversation()
        if not profile_name:
            return None
        try:
            profile, model, _ = self.profile_for(profile_name)
            tokenizer, template = self.tokenizers_for(profile, model)
        except ConfigError:
            return None
        if profile.mode != "chat":  # a text-mode chat can't take the file
            return None
        ctx = self.ctx_size_for(profile).value
        same = model == self.model and ctx == conversation.ctx_size.value  # the server stays
        n_ctx = self.backend.n_ctx if same else ctx
        settings = {k: v.to_json() for k, v in profile.settings.items()}
        key = json.dumps([profile.to_json(), settings, n_ctx], sort_keys=True, default=str)
        return NewChat(profile, tokenizer, template, n_ctx, key)

    def attaching(self) -> tuple[Conversation, Profile, str]:
        """The conversation, its profile and framing line, refusing a text-mode profile."""
        conversation = self._require_conversation()
        profile = conversation.profile
        if profile.mode != "chat":
            raise ConfigError(
                f"Attaching needs a chat-mode profile; {profile.name} is in text mode."
            )
        return conversation, profile, framing_of(profile)[0]

    def attach(
        self,
        doc: Document,
        unit: str | None = None,
        first: int = 0,
        last: int = 0,
        cancel: threading.Event | None = None,
        aside: int = 0,
    ) -> tuple[dict[str, Any], Fit]:
        """Add `doc`, whole or units `first` to `last`, to the attachments waiting for your
        next message, if it fits. Returns its record, as the turn will log it, and its fit:
        the free context before it and what it leaves. A refusal's suggested start leaves
        max_tokens and `aside` (a searching chat's budget, v0.3.2's S1)."""
        conversation, profile, framing = self.attaching()
        assert self.tokenizer is not None
        new = with_text(doc, piece(doc, profile, unit, first, last), unit, first, last)
        pending = attached_in({"attachments": conversation.pending})
        room = self.room(cancel=cancel, name=doc.name)
        before = self._room_for_a_file(room, pending, framing)
        result = fit(room, pending, new, framing, before)
        if not result.fits:
            start = roomy = None
            if covers_whole(doc, unit, first, last):
                checked: dict[int, Fit] = {}
                found = start_that_fits(room, pending, framing, doc, profile, checked=checked)
                start = found[0] if found else None
                found = start_that_fits(
                    room, pending, framing, doc, profile, room.max_tokens + aside, checked
                )
                roomy = found[0] if found else None
            raise DoesntFit(
                f"{doc.name} doesn't fit: it needs {more_than(result.tokens, result.at_least)} "
                "tokens and "
                f"{result.free:,} are free in this chat.",
                start,
                roomy,
            )
        assert result.tokens is not None  # it fits: counted
        whole = covers_whole(doc, unit, first, last)
        record = {
            "name": doc.name,
            "kind": doc.kind,
            "bytes": doc.bytes,
            "sha256": doc.sha256,
            "extractor": doc.extractor,
            "unit": doc.unit,
            "total": doc.total,
            "how": "whole" if whole else "excerpt",
            "range": [1, doc.total] if whole else [first, last],
            "marker": new.marker,
            "text": new.text,
            "extracted_sha256": doc.extracted_sha256,
            "empty_units": doc.empty_units,
            **scan(new.text, self.tokenizer.special_tokens_in),
            "tokens": result.tokens,
        }
        conversation.pending.append(record)
        return record, result

    def check(
        self,
        doc: Document,
        unit: str | None = None,
        first: int = 0,
        last: int = 0,
        cancel: threading.Event | None = None,
        new_profile: str | None = None,
        aside: int = 0,
    ) -> dict[str, Any]:
        """What attaching `doc` would cost and leave, attaching nothing: the whole, the start
        that fits, the preselected start (leaving max_tokens and `aside` more), the range given, and
        the whole in the chat /new would start (`new_profile`, the profile it starts, "" for
        none; by default this chat's profile file)."""
        conversation, profile, framing = self.attaching()
        assert self.tokenizer is not None
        pending = attached_in({"attachments": conversation.pending})
        room = self.room(cancel=cancel, name=doc.name)
        new = self.new_chat(str(conversation.profile.path) if new_profile is None else new_profile)
        # The whole, the new chat and the two starts are kept for the last document and chat
        # state, so a range change in the sheet renders only the range (part 1's fix review,
        # should 5: at a 128K context one whole render takes seconds and gigabytes). A name
        # is inside its block, so it changes every count (the records finder, 1).
        key = (
            doc.sha256, doc.kind, doc.name, profile.name, conversation.conversation_id,
            conversation.head,
            tuple((a["sha256"], a["name"], tuple(a["range"])) for a in conversation.pending),
            room.max_tokens, self.backend.n_ctx, new.key if new else None, aside,
        )  # fmt: skip
        if self._checked is None or self._checked[0] != key:
            before = self._room_for_a_file(room, pending, framing)
            checked: dict[int, Fit] = {}
            self._checked = (
                key,
                Counts(
                    before,
                    fit(room, pending, piece(doc, profile), framing, before),
                    None
                    if new is None
                    else fit(
                        self.room(new, cancel, doc.name),
                        [],
                        piece(doc, new.profile),
                        framing_of(new.profile)[0],
                    ),
                    start_that_fits(room, pending, framing, doc, profile, checked=checked),
                    start_that_fits(
                        room, pending, framing, doc, profile, room.max_tokens + aside, checked
                    ),
                ),
            )
        counts = self._checked[1]
        whole, new_chat = counts.whole, counts.new_chat
        units = doc.unit + "s"

        def shown(k_fit: tuple[int, Fit] | None, first: int = 1) -> dict[str, Any] | None:
            if k_fit is None:
                return None
            k, found = k_fit
            return {
                "unit": units,
                "range": [first, k],
                "tokens": found.tokens,
                "tokens_at_least": found.at_least,
                "left": found.left,
                "marker": piece(doc, profile, units, first, k).marker,
            }

        chosen = None
        if unit is not None:
            excerpt = with_text(doc, piece(doc, profile, unit, first, last), unit, first, last)
            chosen = shown((last, fit(room, pending, excerpt, framing, counts.before)), first)
            assert chosen is not None
            chosen["text"] = excerpt.text  # for the sheet's extracted text (C1)
        return {
            "name": doc.name,
            "kind": doc.kind,
            "bytes": doc.bytes,
            "unit": doc.unit,
            "total": doc.total,
            "empty_units": doc.empty_units,
            "encrypted": bool(doc.extractor.get("encrypted")),
            "tokens_whole": whole.tokens,  # null when not counted: then at least this many
            "tokens_whole_at_least": whole.at_least,
            "free": whole.free,
            "fits": whole.fits,
            "left_whole": whole.left,
            "start_that_fits": shown(counts.start),
            "preselected": shown(counts.roomy),
            "max_tokens": room.max_tokens,  # what the numbers were counted with
            "aside": aside,  # a searching chat's budget, which the preselected start leaves too
            "aside_origin": budget_of(conversation.profile)[1] if aside else None,
            "chosen": chosen,
            # null and false when today's profile file can't start a chat: /new would refuse
            "free_new_chat": new_chat and new_chat.free,
            "fits_new_chat": bool(new_chat and new_chat.fits),
            "tokens_new_chat": new_chat and new_chat.tokens,
            "left_new_chat": new_chat and new_chat.left,
            "already_in_chat": self.already_in_chat(doc.sha256),
            **scan(doc.text, self.tokenizer.special_tokens_in),
            "framing": framing,
            "framing_origin": framing_of(profile)[1],
            # The sheet's excerpt sentence follows the profile's setting, not one range's
            # marker: a range that covers the whole file has none (Checkpoint 3, must 3).
            "excerpt_marker_origin": "default"
            if profile.excerpt_marker is None
            else "profile"
            if profile.excerpt_marker
            else "none",
            "text": doc.text,
        }

    @staticmethod
    def _room_for_a_file(room: Room, pending: list[Attached], framing: str) -> int:
        """The prompt's tokens with what's waiting, refusing first when the chat has no room
        left at all (part 1's fix review, should 3): no count then, and no negative number."""
        before = room.tokens(content("", pending, framing))
        if room.room_left(before) < 1:
            raise DocumentRefused(SENTENCES["no_room"])
        return before

    def already_in_chat(self, sha256: str) -> bool:
        """Whether a file with these bytes is waiting, or attached on this branch."""
        conversation = self._require_conversation()
        turns = conversation.path_to(conversation.head)
        attached = [*conversation.pending, *(a for t in turns for a in t.attachments)]
        return any(a["sha256"] == sha256 for a in attached)

    def regen_seed(self) -> int:
        """The next regeneration's seed: the highest seed used for the current message, plus
        1. For replies made only by /regen, that's the first reply's seed plus k."""
        return self._regen_start()[3]

    def _regen_start(self) -> tuple[Turn, Turn, Turn, int]:
        """The current reply, its message, that message's first reply, and the next seed."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("regenerate")
        assert reply.parent_id is not None
        replies = conversation.replies_to(reply.parent_id)
        first = next((t for t in replies if t.regen_of is None), replies[0])
        seeds = [t.seed for t in replies if t.seed is not None]
        if not seeds or first.call_id is None:
            raise ConfigError("No reply to this message has a seed to start from.")
        seed = max(seeds) + 1
        if seed > SEED_MAX:
            raise ConfigError(
                f"The next regeneration seed, {seed}, is above {SEED_MAX}. Edit the message or "
                "start a new conversation instead."
            )
        return reply, conversation.turns[reply.parent_id], first, seed

    def current_values(self) -> dict[str, Any]:
        """The values the next call would use, by name, without the seed."""
        conversation = self._require_conversation()
        return {k: s.value for k, s in conversation.settings.items() if k != "seed"}

    def seeds_to_make(self, message_id: str) -> list[int]:
        """The seeds used for a message's replies that have no reply under the current
        settings, in the order they were first used: what /regen same makes."""
        conversation = self._require_conversation()
        now = self.current_values()
        seeds: dict[int, bool] = {}  # seed -> it has a reply under the current settings
        for turn in conversation.replies_to(message_id):
            call = conversation.calls.get(turn.call_id or "")
            if call is None or turn.seed is None:
                continue
            same = setting_values(call) == now
            seeds[turn.seed] = seeds.get(turn.seed, False) or same
        return [seed for seed, made in seeds.items() if not made]

    def compare(self, reply_id: str) -> dict[str, Any]:
        """The replies to the message `reply_id` answers, on every branch, oldest first:
        grouped by their settings (all but the seed), what's the same in all, where two
        replies with the same seed in different groups part, and, for the current message,
        what /regen same would make. Reads only; the head stays."""
        conversation = self._require_conversation()
        message = conversation.turns[conversation.turns[reply_id].parent_id or ""]
        replies = [
            t for t in conversation.replies_to(message.turn_id) if t.call_id in conversation.calls
        ]
        calls = [conversation.calls[t.call_id or ""] for t in replies]
        values = [setting_values(call) for call in calls]
        names = sorted(set().union(*values))
        differ = [n for n in names if any(v.get(n) != values[0].get(n) for v in values)]
        groups: list[dict[str, Any]] = []
        group_of = []
        for v in values:
            key = {n: v.get(n) for n in differ}
            if key not in groups:
                groups.append(key)
            group_of.append(groups.index(key))
        notes = self.marks_and_notes()
        newest = conversation.newest_replies()
        on_head_path = {t.turn_id for t in conversation.path_to(conversation.head)}
        out_replies = []
        for k, (turn, call) in enumerate(zip(replies, calls, strict=True), 1):
            annotations = notes.get(turn.turn_id, {})
            out_replies.append(
                {
                    "k": k,
                    "call_id": turn.call_id,
                    "turn_id": turn.turn_id,
                    "text": turn.text,
                    "seed": turn.seed,
                    "steering_scale": call["settings"].get("steering_scale", {}).get("value"),
                    "stop_type": call.get("stop_type"),
                    "group": group_of[k - 1],
                    "marks": [
                        {"mark_id": m.get("mark_id"), "label": m["label"]}
                        for m in annotations.get("marks", [])
                    ],
                    "notes": [
                        {"note_id": n.get("note_id"), "text": n["text"]}
                        for n in annotations.get("notes", [])
                    ],
                    "current": turn.turn_id in on_head_path,
                    "later_turns": len(conversation.path_to(newest[turn.turn_id].turn_id))
                    - len(conversation.path_to(turn.turn_id)),
                }
            )
        pairs = []
        for i, a in enumerate(calls):
            for j in range(i + 1, len(calls)):
                b = calls[j]
                if group_of[i] == group_of[j] or replies[i].seed != replies[j].seed:
                    continue
                same = a["output_ids"] == b["output_ids"]
                token = None if same else first_difference(a["output_ids"], b["output_ids"])
                texts = (a["output_text"], b["output_text"])
                shorter = min(len(a["output_ids"]), len(b["output_ids"]))
                pairs.append(
                    {
                        "seed": replies[i].seed,
                        "a": a["call_id"],
                        "b": b["call_id"],
                        "first_different_token": token,
                        # One stops where the other goes on: its text begins the other's.
                        # (its end token, if any, is where the IDs part).
                        "prefix": token is not None
                        and token >= shorter - 1
                        and (texts[1].startswith(texts[0]) or texts[0].startswith(texts[1])),
                        "first_different_char": None
                        if token is None
                        else self._chars_before(b, token),
                    }
                )
        head = conversation.turns.get(conversation.head or "")
        again = None
        if head is not None and head.role == "assistant" and head.parent_id == message.turn_id:
            seeds = self.seeds_to_make(message.turn_id)
            now = self.current_values()
            changed = {n: now[n] for n in sorted(now) if any(v.get(n) != now[n] for v in values)}
            again = {"seeds": seeds, "now": changed} if seeds else None
        return {
            "message": {"turn_id": message.turn_id, "text": message.text},
            "replies": out_replies,
            "groups": [{"settings": key} for key in groups],
            "same": {n: values[0].get(n) for n in names if n not in differ},
            "same_prompt": all(c["prompt_ids"] == calls[0]["prompt_ids"] for c in calls),
            "pairs": pairs,
            "again": again,
        }

    def _chars_before(self, call: dict[str, Any], token: int) -> int:
        """How many characters of a reply's text its first `token` output tokens make."""
        assert self.tokenizer is not None
        text = self.tokenizer.decode(call["output_ids"][:token]).rstrip("\ufffd")
        return min(len(text), len(call["output_text"]))

    def regen(
        self,
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        seed: int | None = None,
    ) -> CallOutcome:
        """Another reply to the current message, as a new branch, with `seed` (default
        `regen_seed()`)."""
        reply, user, first, next_seed = self._regen_start()
        seed = next_seed if seed is None else seed
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
        drop: Sequence[int] = (),
    ) -> CallOutcome:
        """Replace the current message with `text`, as a new branch, and get a reply. `drop`:
        the attachments to leave out, by their place in the message, from 1 (Q2)."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("edit the message of")
        assert reply.parent_id is not None
        original = conversation.turns[reply.parent_id]
        n = len(original.attachments)
        for i in drop:
            if not n:
                raise ConfigError(f"--drop {i}: this message has no attachments.")
            if not 1 <= i <= n:
                raise ConfigError(
                    f"--drop {i}: this message has {counted(n, 'attachment')}. "
                    f"Use a number from 1 to {n}."
                )
        kept = [a for i, a in enumerate(original.attachments, 1) if i not in drop]
        # The attachments are kept, but those dropped; the new text is searched again
        # (decision 2, Q6).
        attached, refused = self._block(text, kept, original.parent_id, True)
        user = conversation.add_turn(
            "user", "edit", text, original.parent_id, edit_of=original.turn_id, **attached
        )
        conversation.head = user.turn_id
        # A failed edit leaves you where you were.
        return self.call(
            user, on_text=on_text, cancel=cancel, back_to=reply.turn_id, refused=refused
        )

    def undo(self) -> Turn | None:
        """Move back past the last exchange. Returns the new head (None at the start). The
        undone turns stay in the log; the next message branches from the new head."""
        conversation = self._require_conversation()
        reply = conversation.current_reply("undo")
        assert reply.parent_id is not None
        return conversation.move_head(conversation.turns[reply.parent_id].parent_id, "undo")

    def _profile_documents(self, profile: Profile, where: str = "chat") -> None:
        """A profile's `search` list names documents in the library: one that isn't there
        is refused when a chat starts, naming the key (v0.3.2 B1)."""
        for name in profile.search:
            state = library_state(self.root, name)
            if state == "missing":  # N37, each front end its way to add it
                problem = {"document": name, "state": "profile_missing", "profile": profile.name}
                raise SearchRefused([problem], where=where)
            if state == "unreadable":  # a link, or a manifest Clarity can't read: why
                try:
                    library_manifest(self.root, name)
                except Unreadable as e:  # each front end its way out, and the profile's key
                    problem = {**unreadable_problem(name, e), "profile": profile.name}
                    raise SearchRefused([problem], where=where) from None
                except DocumentRefused as e:  # a linked folder: its own way out
                    raise ConfigError(f"Profile {profile.name} searches {name}. {e}") from None

    def set_search(self, names: list[str]) -> dict[str, Any]:
        """The documents this chat searches, from now on; logged as a `search` record with
        the whole list. A document being added must be in the library and readable; one
        already in the list stays whatever its state, and Send refuses until it's stopped
        (the board walk's W3: "Search it" can never write a copy and be refused for another
        document)."""
        conversation = self._require_conversation()
        profile = conversation.profile
        if profile.mode != "chat":  # N71
            raise ConfigError(SENTENCES["text_mode_search"].format(profile=profile.name))
        for name in names:
            checked_library_name(name)
            if name in conversation.searching:
                continue
            state = library_state(self.root, name)
            if state == "missing":
                raise DocumentRefused(f"{name} isn't in your documents.")
            if state == "unreadable":  # refused, saying why, and each front end its way out
                try:
                    library_manifest(self.root, name)  # a linked folder: its own refusal
                except Unreadable as e:
                    raise SearchRefused([unreadable_problem(name, e)]) from None
            if state == "ready":
                try:
                    library_parts(self.root, name)  # read as untrusted (the review's S2)
                except Stale as e:  # gone under the quick check (the fix review's C6)
                    raise SearchRefused([{"document": name, "state": e.state}]) from None
                except Unreadable as e:  # not the parts it built: add it again (cut 1)
                    raise SearchRefused([unreadable_problem(name, e)]) from None
        if named_twice(names) is not None:
            raise ConfigError("Name each document once.")
        conversation.searching, conversation.searching_origin = list(names), "chat"
        return conversation.log.append(
            {
                "type": "search",
                "conversation_id": conversation.conversation_id,
                "documents": list(names),
            }
        )

    def budget(self, conversation: Conversation | None = None) -> tuple[int, str]:
        """Search's budget for a message, and where it comes from: the profile, or Clarity's
        default."""
        return budget_of((conversation or self._require_conversation()).profile)

    def aside(self) -> int:
        """What a suggested start leaves free for search: the budget, when this chat searches
        any document (v0.3.2's S1); else 0."""
        conversation = self.conversation
        return self.budget()[0] if conversation is not None and conversation.searching else 0

    def search_warnings(self, conversation: Conversation) -> list[dict[str, Any]]:
        """What /search says before a message is sent (O1): bge-m3 not fetched, for a chat
        that searches by meaning. A file check: nothing starts."""
        meaning = conversation.searching and conversation.search_method == "meaning"
        return [{"state": "no_embedder"}] if meaning and not self.embedder().fetched() else []

    def searching(self, conversation: Conversation | None = None) -> list[dict[str, Any]]:
        """The documents this chat searches, each with its state: ready, missing or
        unreadable (with why). A control token is said when a message finds its part (cut 5)."""
        conversation = conversation or self.conversation
        if conversation is None:
            return []
        found = []
        for name in conversation.searching:
            state = library_state(self.root, name)
            entry: dict[str, Any] = {"name": name, "state": state}
            try:
                if state == "ready":
                    self._parts_read(name)
                elif state == "unreadable":  # why, as /documents says it (the review's C7)
                    library_manifest(self.root, name)
            except Stale as e:  # gone under the quick check: its state (the review's C6)
                entry["state"] = e.state
            except DocumentRefused as e:  # a manifest or parts Clarity can't read: the reason
                problem = unreadable_problem(name, e)
                entry.update(state="unreadable", why=problem["why"], add_again=problem["add_again"])
            found.append(entry)
        return found

    def _parts_read(self, name: str) -> None:
        """A ready document's parts read and checked, kept by build ID and parts.jsonl's
        size and time: /state asks often, and parts that changed are read (and refused)."""
        manifest = library_manifest(self.root, name)
        stamp = library_parts_stamp(self.root, name)  # None: the read below says why
        key = (manifest["build_id"], stamp)
        if stamp is None or key not in self._parts_checked:
            try:
                library_parts(self.root, name, manifest)
                self._parts_checked[key] = None
            except DocumentRefused as e:
                self._parts_checked[key] = e
        if (refused := self._parts_checked[key]) is not None:
            raise refused

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

    def history(self, conversation: Conversation | None = None) -> list[dict[str, Any]]:
        """The conversation from its start to `head`: each turn with its current marks and
        notes. What a front end needs to draw the chat, without reading the log. /state
        passes the conversation it holds: another thread can be switching it."""
        conversation = conversation or self._require_conversation()
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
                **({"attachments": summaries(t.attachments)} if t.attachments else {}),
                **({"found": found_summary(t.found)} if t.found is not None else {}),
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

    def marks_and_notes(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Each reply's current marks and notes, read from this conversation's log."""
        conversation = self._require_conversation()
        return marks_and_notes(read_records(conversation.log.path, annotation_line))

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

    def annotate(
        self, kind: str, text: str, call: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """A `mark`, `unmark` or `note` on the current reply, or on `call`'s reply. See
        `annotate_call`."""
        conversation = self._require_conversation()
        if call is None:
            reply = conversation.current_reply(kind)
            call_id, turn_id = reply.call_id, reply.turn_id
        elif not call.get("turn_id"):
            raise ConfigError(f"Call {call['call_id']} failed, so it has no reply to {kind}.")
        else:
            call_id, turn_id = call["call_id"], call["turn_id"]
        about = {
            "conversation_id": conversation.conversation_id,
            "call_id": call_id,
            "turn_id": turn_id,
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
        path = prompt_turns(conversation, user_turn)
        turns = [(t.role, t.content) for t in path]
        prompt = build_prompt(tokenizer, conversation.profile, turns, template_tokenizer)
        for i, t in enumerate(path):  # control tokens in an attachment: name it (M4)
            found = [
                {"index": n, "name": a["name"], "tokens": tokenizer.special_tokens_in(a["text"])}
                for n, a in enumerate(t.attachments)
            ]
            found = [f for f in found if f["tokens"]]
            for entry in prompt.special_in_content:
                if found and entry["in"] == f"turn {i + 1} ({t.role})":
                    entry["attachments"] = found
        return prompt

    def message_spans(self, user_turn: Turn, prompt: Prompt) -> list[dict] | dict[str, str]:
        """Where each message sits in `prompt` (render.message_spans), with turn IDs."""
        conversation = self._require_conversation()
        assert self.tokenizer is not None
        path = prompt_turns(conversation, user_turn)
        return message_spans(
            self.tokenizer,
            conversation.profile,
            [(t.role, t.content) for t in path],
            [t.turn_id for t in path],
            prompt,
            self.template_tokenizer,
            [added_ranges(t.text, t.attached_fields()) for t in path],
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
        refused: SearchRefused | None = None,
    ) -> CallOutcome:
        """Get the reply to `user_turn`. If the call fails, `head` goes to `back_to`
        (default: before the message, which becomes a dead branch). `refused`: a search's
        refusal, logged as this call's failure."""
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
            # Refused with the prompt recorded, as a ContextError is: nothing is sent.
            if refused is not None:
                raise refused
            self._control_tokens_in_found(user_turn)
            new_message = self.tokenizer.special_tokens_in(user_turn.text)
            if new_message:
                warnings.append(
                    f"Your message contains {new_message}: the tokenizer reads these as control "
                    "tokens, not text. The prompt's token IDs, in the call's details, show them."
                )
            for a in user_turn.attachments:
                if a["control_tokens"]:
                    warnings.append(
                        f"{a['name']} contains {a['control_tokens']}: the tokenizer reads these "
                        "as control tokens, not text. It's sent as it is."
                    )
            n_ctx, max_tokens = self.backend.n_ctx, values["max_tokens"]
            over = -room_left(n_ctx, len(prompt.ids), max_tokens)
            if over > 0:
                of_them = attached_tokens(record["message_spans"], conversation, user_turn)
                # Only a failed send puts its attachments back as chips, so removing one is
                # advice you can follow only then: not after /regen or /edit (ruling 9,
                # the fact-check's row 53b, the whole-branch review's must 2).
                waiting = back_to is None and bool(user_turn.attachments)
                advice = SENTENCES["remove_waiting" if waiting else "stop_advice"]
                # max_tokens as typed, a setting; the counts with separators (ruling 2).
                raise ContextError(
                    f"The prompt is {len(prompt.ids):,} tokens{of_them}; with max_tokens "
                    f"{max_tokens} it doesn't fit in the context of {n_ctx:,} tokens. Nothing "
                    f"was sent. {SENTENCES['too_long'].format(over=f'{over:,}')} {advice}"
                )
            if len(prompt.ids) > CONTEXT_WARN_FRACTION * n_ctx:
                warnings.append(
                    f"The prompt uses {len(prompt.ids):,} of {n_ctx:,} context tokens "
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
                raise CancelledBeforeReply("Stopped before the reply started.", result)
        except (ContextError, BackendError, ConfigError) as e:
            partial = e.partial if isinstance(e, BackendError) else None
            if partial is not None:
                record.update(result_fields(partial)[0])
            record["error"] = f"{type(e).__name__}: {e}"
            record["warnings"] = warnings
            # As logged (with its time), so it reads the same live and after a resume.
            record = conversation.log.append(record)
            # A failed message names its call, so a resume shows it in /prompt as the chat did.
            conversation.calls[call_id] = record  # readable by its ID (/prompt CALL_ID)
            failed = {"call_id": call_id} if back_to is None else {}
            conversation.move_head(back_to or user_turn.parent_id, "failed", **failed)
            if back_to is None:  # a message, sent and failed
                conversation.last_call = record  # /prompt shows the failed message's call
                conversation.unanswered = {"turn_id": user_turn.turn_id, "text": user_turn.text}
                conversation.pending = list(user_turn.attachments)  # waiting again
                conversation.resume_differ = None  # it used the resumed one up
            elif user_turn.origin == "edit":  # as a resume reads it: the edit is the last turn
                conversation.unanswered = None
            raise
        fields, server_prompt = result_fields(result)
        record.update(fields)
        ignored = [e["name"] for e in result.not_applied if e.get("affects_output")]
        if ignored:
            record["flags"].append("not_applied")
            warnings.append(
                f"{self.backend.name} can't apply {ignored} as set, so they had no effect or "
                "ran differently. The settings list has the reasons."
            )
        if prompt.special_in_content:
            record["flags"].append("special_tokens_in_content")
        self._check_output(prompt.ids, result, server_prompt, record)
        record["warnings"] = warnings
        # The call record names the reply's turn, and is written before the turn record.
        reply_id = new_id("turn")
        record["turn_id"] = reply_id
        record = conversation.log.append(record)  # as logged, as after a resume
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
        conversation.unanswered = conversation.resume_differ = None
        conversation.last_call = conversation.calls[call_id] = record
        return CallOutcome(record, result, reply, warnings)

    def _control_tokens_in_found(self, user_turn: Turn) -> None:
        """A found part holding this model's control token stops the call (decision 12): you
        didn't choose that text for this message, and a stray end-of-turn token ends your
        turn early. The refusal names the part and the way out."""
        assert self.tokenizer is not None
        problem = self._control_token_problem({"parts": found_in(user_turn.attached_fields())},
                                              self.tokenizer)  # fmt: skip
        if problem is not None:  # the board's N33
            raise SearchRefused([problem], at_send=True)

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

    def replay_current(
        self, cancel: threading.Event | None = None, call: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Replay the current reply's call, or `call`, on the running server."""
        conversation = self._require_conversation()
        if call is None:
            call_id = conversation.current_reply("replay").call_id
            assert call_id is not None
        else:
            call_id = call["call_id"]
        return self.replay(conversation.log.path, call_id, conversation.log, cancel=cancel)

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
        # Checked before anything starts: a log another program wrote can leave one out.
        missing = [field for field in REPLAY_NEEDS if field not in call]
        if missing:
            raise ConfigError(
                f"Call {call_id} can't be replayed: its record has no {', '.join(missing)}."
            )
        name = call["model"]["name"]
        described = call.get("backend") or {}
        backend_name = described.get("backend", "llama.cpp")
        if self.conversation is None:  # clarity replay: the call's file, under any name
            lock = read_lock(self.root)["models"]
            lock = {n: lock.get(n, {}) for n in self.models}
            used = model_for_call(call["model"], backend_name, lock)
            if used is None:
                raise ConfigError(no_model_for_call(call_id, call["model"]))
            name = used
        if name not in self.models:
            raise ConfigError(
                f"Call {call_id} used model {name!r}, which isn't in models.toml or "
                "local/models.toml."
            )
        model = self.models[name]
        entry = lock_entry(self.root, model)
        if self.conversation is not None and self.model is not None and self.model.name != name:
            raise ConfigError(
                f"Call {call_id} used {name}, but this conversation uses {self.model.name}. "
                "Resume the conversation it's from to replay it there."
            )
        if self.conversation is not None and call["n_ctx"] != self.backend.n_ctx:
            raise ConfigError(
                f"Call {call_id} ran with a context of {call['n_ctx']} tokens, but this "
                f"conversation uses {self.backend.n_ctx}. Resume the conversation it's from "
                "to replay it there."
            )
        device = described.get("device") if backend_name == "transformers" else None
        weights = self.weights_for(backend_name, entry)
        steer = self.logged_steering(call, backend_name, weights)
        previous = self.backend
        try:
            backend = self.use_backend(backend_name, device)
            backend.start(model, weights, call["n_ctx"])  # the call's own size, as logged
        except (ConfigError, BackendError) as e:
            # as start_model does: the conversation stays usable
            self._restore(previous, e, (model, weights, call["n_ctx"]))
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
            raise ConfigError("Replay stopped: nothing was compared or logged.")
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
            "model_used": model.name,  # the models.toml entry it ran on
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
            raise ConfigError(f"No conversation or log file {path}. The chat list has them.")
        try:
            return self._resume(path)
        except (KeyError, TypeError, AttributeError) as e:  # a log Clarity didn't write, or a bug
            raise ConfigError(unreadable(e, f"the records of {path}")) from e

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
        chat = {
            k: check_setting(k, v, f"{path}: a setting changed in the chat")
            for k, v in state.chat_settings.items()
        }
        merged = {**prepared.profile.settings, **{k: Setting(v, "chat") for k, v in chat.items()}}
        check_samplers(merged, f"{path}: a setting changed in the chat")
        steered = prepared.profile.steering is not None
        check_backend_settings(
            merged, prepared.profile.backend, f"{path}: a setting changed in the chat", steered
        )
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
        text_mode = prepared.profile.mode != "chat"  # no documents (Checkpoint 2's S5)
        if text_mode and state.searching:
            raise TypeError("a text-mode chat's search record isn't one Clarity writes")
        for t in state.turns:
            call = state.calls.get(t.get("call_id") or "")
            seed = call["settings"]["seed"]["value"] if call else None
            attached = checked_attachments(t)  # a missing field: KeyError, refused
            if text_mode and attached:
                raise TypeError("a text-mode chat's turn with documents isn't one Clarity writes")
            conversation.turns[t["turn_id"]] = Turn(
                t["turn_id"], t["parent_id"], t["role"], t["origin"], t["text"],
                t.get("call_id"), seed, t.get("regen_of"), t.get("edit_of"), **attached,
            )  # fmt: skip
        conversation.head = state.head
        # D4: the logged profile's method, else the last searched message's, else the
        # default today, as any setting added since takes.
        searched = [t["found"]["method"] for t in state.turns if "found" in t]
        conversation.search_method = prepared.profile.search_method or (
            searched[-1] if searched else SEARCH_METHOD
        )
        if state.searching is not None:  # a `search` record: this chat's own list
            for name in state.searching:  # refused before it's ever a path (S14), said
                if (problem := library_name_problem(name)) is not None:  # with its record
                    raise DocumentRefused(
                        f"{path}: its search record names {shown(name)}, which can't name a "
                        f"document. {problem}"
                    )
            if (twice := named_twice(state.searching)) is not None:
                raise DocumentRefused(
                    f"{path}: its search record names {twice} twice. Name each document once."
                )
            conversation.searching = list(state.searching)
            conversation.searching_origin = "chat"
        else:
            conversation.searching = list(prepared.profile.search)
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
        differ = what_differs(record)
        conversation.resume_differ = {"count": sum(i["count"] for i in differ), "items": differ}
        conversation.unanswered = state.unanswered
        if state.unanswered:  # its attachments wait again, as they did live
            conversation.pending = list(conversation.turns[state.unanswered["turn_id"]].attachments)
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
            raise ConfigError("No conversation is running. Start one with a profile first.")
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


def attached_tokens(spans: Any, conversation: Conversation, user_turn: Turn) -> str:
    """For the stop's sentence: how many of the prompt's tokens attachments take. Exact from
    the call's spans (each document's block and the framing line); without them, each
    attachment's cost when it was attached, said as such."""
    if isinstance(spans, list):
        added = [a for e in spans for a in e.get("added", [])]
        n = sum(a["tokens"][1] - a["tokens"][0] for a in added)
        kinds = {a["what"] for a in added}
        what = ", ".join(
            [*(["attached documents"] if "attachment" in kinds else []),
             *(["parts found in your documents"] if "part" in kinds else [])]  # N34
        )  # fmt: skip
        return f", {n:,} of them in {what} and their framing lines" if n else ""
    path = prompt_turns(conversation, user_turn)  # Checkpoint 2's C3: found parts too
    attached = sum(a["tokens"] for t in path for a in t.attachments)
    found = sum(t.found["tokens"] for t in path if t.found is not None)
    n, what, when = attached + found, [], []
    if attached:
        what, when = ["attached documents"], ["attached"]
    if found:
        what, when = [*what, "parts found in your documents"], [*when, "found"]
    return (
        f", {n:,} of them for {' and '.join(what)}, counted when {' or '.join(when)}" if n else ""
    )


def budget_of(profile: Profile) -> tuple[int, str]:
    """Search's budget for a message on `profile`, and where it comes from."""
    return (BUDGET, "default") if profile.budget is None else (profile.budget, "profile")


def preparing_line(name: str, total: int) -> str:
    """Said once before a document's vectors are built (state 4), by the terminal and a run."""
    parts = f"{total:,} part" + ("" if total == 1 else "s")
    return f"Preparing {name} for meaning search: {parts}, saved in local/documents/{name}/."


def minimum_of(profile: Profile) -> tuple[float, str]:
    """Meaning search's minimum similarity on `profile`, and where it comes from."""
    if profile.min_similarity is None:
        return MIN_SIMILARITY, "default"
    return profile.min_similarity, "profile"


def left_out_counts(found: dict[str, Any]) -> dict[str, int]:
    """How many matches were left out, by why: every one, not only the 50 listed (fix 2's
    S2). The considered ones are all listed; the rest weren't considered."""
    counts = Counter(o["why"] for o in found["left_out"] if o["why"] != "not considered")
    unconsidered = found["matched"] - min(found["matched"], found["considered"])
    return {**counts, **({"not considered": unconsidered} if unconsidered else {})}


def unreadable_problem(name: str, refused: DocumentRefused) -> dict[str, Any]:
    """A document search can't read, as a problem: its reason, and whether adding the file
    again is the way out (`add_again`), which each front end words (surface S2). A linked
    folder's sentence holds its own."""
    return {
        "document": name,
        "state": "unreadable",
        "why": reason(refused),
        "add_again": isinstance(refused, Unreadable),
    }


def reason(refused: DocumentRefused) -> str:
    """What a library refusal says is wrong, without a way out: each front end adds its own
    (the close batch: surface S2)."""
    return refused.said if isinstance(refused, Unreadable) else str(refused)


def found_summary(found: dict[str, Any]) -> dict[str, Any]:
    """/history's view of a turn's `found`, which /state's turns repeat: counts, tokens,
    words, the budget, what was left out by why, and each part's ID, title, page, tokens and
    why it matched, with no text (the Shapes table's commands; part 1's whole-branch S1).
    By meaning (v0.4.1): the minimum, and each part's ID, title, rank and similarity."""
    if found["method"] == "meaning":
        keys = ("id", "title", "page", "rank", "score")  # page: could 5
        return {
            "method": "meaning",
            "documents": [d["name"] for d in found["documents"]],
            **{k: found[k] for k in ("matched", "tokens", "budget", "considered")},
            "min_similarity": found["min_similarity"],
            "parts": [{k: p[k] for k in keys} for p in found["parts"]],
        }
    keys = ("id", "title", "page", "rank", "title_words", "words", "tokens")
    return {
        "method": "words",
        "words": found["words"],
        "ignored": found["ignored"],
        "documents": [d["name"] for d in found["documents"]],
        "matched": found["matched"],
        "tokens": found["tokens"],
        "budget": found["budget"],
        "parts": [{k: p[k] for k in keys} for p in found["parts"]],
        "left_out": left_out_counts(found),
        "considered": found["considered"],
    }


def prompt_turns(conversation: Conversation, user_turn: Turn) -> list[Turn]:
    """The turns a prompt for `user_turn` holds: its path, without the profile's priming
    (which the profile renders itself)."""
    return [t for t in conversation.path_to(user_turn.turn_id) if t.origin != "priming"]


# Settings added after v0.1.0, with the value a call made before them ran with: neither
# feature existed, so it was off.
BEFORE_THEY_EXISTED = {"logprobs": 0, "steering_scale": 0.0}


def setting_values(call: dict[str, Any]) -> dict[str, Any]:
    """A call's setting values by name, without the seed: what /compare groups replies by,
    and /regen same matches. A setting from before its time reads as off."""
    values = {k: v["value"] for k, v in call["settings"].items() if k != "seed"}
    return {**BEFORE_THEY_EXISTED, **values}


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
