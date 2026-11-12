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


class BackendError(Exception):
    """The backend failed. `partial` holds whatever reply arrived before the failure."""

    def __init__(self, message: str, partial: Result | None = None) -> None:
        super().__init__(message)
        self.partial = partial


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
