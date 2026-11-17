"""Backends turn token IDs plus settings into a reply.

A backend implements `start`, `complete`, `close` and `describe`. To add one, write a
module with a class like `LlamaCppBackend`: map the client's setting names to the
backend's own, report any setting it can't apply in `not_applied`, and return a `Result`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from clarity_chat.profiles import ModelEntry

if TYPE_CHECKING:
    from clarity_chat.steering import Steering


# What each check after a call checks, and what its failure means: the one place for this
# text. The web interface's Details panel shows it, and docs/reference.md's table of checks
# repeats it (tests/test_docs.py checks they match). The failures' words: notes/research/
# v030-teaching-check.md, item 16. server_props is in logs before v0.2.1 only.
CHECK_HELP: dict[str, tuple[str, str]] = {
    "prompt_echo": (
        "The server’s decoding of the token IDs equals the prompt text.",
        (
            "The server’s text for the token IDs differs from Clarity’s, so the model file and "
            "the tokenizer may disagree."
        ),
    ),
    "settings_echo": (
        "Every value sent comes back unchanged in the server’s generation_settings.",
        "A value Clarity sent didn’t come back unchanged from the server.",
    ),
    "unaccounted_settings": (
        "The server reports no setting Clarity didn’t send or expect.",
        "The server reported a setting Clarity didn’t send or expect.",
    ),
    "tokens_evaluated": (
        "The server read as many prompt tokens as were sent.",
        "The server read a different number of prompt tokens than were sent.",
    ),
    "no_cache_reuse": (
        "No prompt tokens came from a cache.",
        "Some prompt tokens came from a cache.",
    ),
    "server_running": (
        "The llama-server process Clarity started is still running.",
        (
            "The model’s server had stopped. The reply is complete, but the next message needs "
            "the server started again."
        ),
    ),
    "suppress_tokens": (
        (
            "llama.cpp only, on a model file that bans tokens (tokenizer.ggml.suppress_tokens):"
            " no banned token is in the reply."
        ),
        "A token the model file bans is in the reply, so llama-server didn’t apply the ban.",
    ),
    "server_props": (
        (
            "Logs before v0.2.1 only: after the call, /props showed the server still running, "
            "with the model and context size it started with."
        ),
        (
            "After the call, the server had stopped, /props named another model or context size,"
            " or /props couldn’t be read."
        ),
    ),
    "generation_config": (
        "Transformers only: the effective generation config equals what was sent.",
        "Transformers’ generation settings differed from what was sent.",
    ),
    "steering": (
        (
            "Transformers only, on a steered call: every forward pass ran each steered block "
            "exactly once, and there was a pass for every generated token."
        ),
        (
            "A forward pass didn’t run each steered block exactly once, or a generated token had"
            " no pass."
        ),
    ),
}


class BackendError(Exception):
    """The backend failed. `partial` holds whatever reply arrived before the failure.
    `server_stopped`: the model's server process has exited, so it needs starting again."""

    def __init__(
        self, message: str, partial: Result | None = None, server_stopped: bool = False
    ) -> None:
        super().__init__(message)
        self.partial = partial
        self.server_stopped = server_stopped


@dataclass
class Result:
    text: str  # the output exactly as the backend returned it
    token_ids: list[int]  # output token IDs (include stop-string and end-of-sequence tokens)
    stop_type: str  # eos | word | limit | cancelled | none
    cancelled_during: str | None = None  # "prompt" (before the first token) or "reply"
    stopping_word: str = ""
    n_prompt: int | None = None  # the backend's own prompt token count
    n_predicted: int | None = None  # the backend's own count of generated tokens
    timings: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)  # what was sent (prompt IDs left out)
    response: dict[str, Any] = field(default_factory=dict)  # the backend's final response
    backend_settings: dict[str, dict[str, Any]] = field(default_factory=dict)  # origin backend
    not_applied: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {ok, ...}
    logprobs: dict[str, Any] | None = None  # with the logprobs setting on (Transformers only)


class Backend(Protocol):
    name: str

    def start(self, model: ModelEntry, weights: Path, ctx_size: int) -> dict[str, Any] | None:
        """Load `model` with a context of `ctx_size` tokens (restarting if another model or
        size is loaded). Returns a server_start record, if any."""
        ...

    @property
    def n_ctx(self) -> int: ...

    @property
    def server_record(self) -> dict[str, Any] | None: ...

    @property
    def stopped(self) -> bool:
        """The model's server process has exited since it started (it needs starting again)."""
        ...

    def complete(
        self,
        token_ids: list[int],
        settings: dict[str, Any],
        stop: list[str],
        on_text: Callable[[str], None] | None = None,
        request: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
        steering: Steering | None = None,
    ) -> Result:
        """Generate a reply. `request`, for replay, is a logged request body: send it
        exactly as logged, or raise BackendError; never rebuild it from `settings`.
        `cancel`, set from any thread, stops the call and returns what arrived so far.
        `steering`: the vector to add, live or on replay; a backend that can't apply it
        raises BackendError."""
        ...

    def describe(self) -> dict[str, Any]:
        """Backend name and versions, for every call record."""
        ...

    def close(self) -> None: ...
