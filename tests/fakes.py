"""A fake backend for unit tests: scripted replies, llama-server-style stop handling."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError, Result
from clarity_chat.profiles import ModelEntry


class FakeBackend:
    name = "fake"

    def __init__(
        self, replies: list[str] | None = None, n_ctx: int | None = None, tokenizer: Any = None
    ) -> None:
        self.replies = list(replies or [])
        self.tokenizer = tokenizer  # gives replies real output token IDs
        self.calls: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.ctx_sizes: list[int] = []  # the ctx_size of each start
        self._n_ctx = n_ctx  # if given, it overrides the started size (to test small contexts)
        self.record: dict[str, Any] | None = None
        self.fail_with: str | None = None
        self.cancel_after: int | None = None  # characters streamed before a simulated Ctrl-C
        self.on_call: Callable[[], None] | None = None  # runs mid-call, e.g. a cancel
        self.start_warnings: list[str] = []
        self.result_flags: list[str] = []  # e.g. "check_failed:echo", added to every reply
        self.not_applied: list[dict[str, Any]] = []  # reported with every reply
        self.logprobs: dict[str, Any] | None = None  # returned with every reply

    @property
    def n_ctx(self) -> int:
        return self._n_ctx or (self.ctx_sizes[-1] if self.ctx_sizes else 512)

    @property
    def server_record(self) -> dict[str, Any] | None:
        return self.record

    def start(self, model: ModelEntry, weights: Path, ctx_size: int) -> dict[str, Any] | None:
        if self.started and (self.started[-1], self.ctx_sizes[-1]) == (model.name, ctx_size):
            return None
        self.started.append(model.name)
        self.ctx_sizes.append(ctx_size)
        self.record = {
            "type": "server_start",
            "server_id": f"server-fake{len(self.started)}",
            "backend": self.name,
            "argv": ["fake", str(weights)],
            "warnings": list(self.start_warnings),
        }
        return self.record

    def close(self) -> None:
        self.record = None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}

    def complete(
        self,
        token_ids: list[int],
        settings: dict[str, Any],
        stop: list[str],
        on_text: Callable[[str], None] | None = None,
        request: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> Result:
        self.calls.append(
            {
                "token_ids": list(token_ids),
                "settings": dict(settings),
                "stop": stop,
                "request": request,
            }
        )
        if self.fail_with:
            raise BackendError(self.fail_with, Result(text="", token_ids=[], stop_type="none"))
        reply = self.replies.pop(0) if self.replies else " ok"
        if self.on_call:
            self.on_call()
        cancel_after = self.cancel_after
        if cancel is not None and cancel.is_set() and cancel_after is None:
            cancel_after = 0  # as a backend checks between tokens: nothing arrived yet
        if cancel_after is not None:
            text = reply[:cancel_after]
            if on_text:
                on_text(text)
            return Result(
                text=text,
                token_ids=list(range(len(text))),
                stop_type="cancelled",
                cancelled_during="reply" if text else "prompt",
                flags=["cancelled"],
            )
        stop_type, stopping_word = "eos", ""
        for word in stop:
            if word in reply:
                reply, stop_type, stopping_word = reply[: reply.index(word)], "word", word
        if on_text:
            on_text(reply)
        ids = self.tokenizer.encode(reply) if self.tokenizer else list(range(len(reply)))
        return Result(
            text=reply,
            token_ids=ids,
            stop_type=stop_type,
            stopping_word=stopping_word,
            n_prompt=len(token_ids),
            n_predicted=len(reply),
            request={"n_predict": settings["max_tokens"], "seed": settings["seed"], "stop": stop},
            flags=list(self.result_flags),
            not_applied=list(self.not_applied),
            logprobs=self.logprobs,
        )
