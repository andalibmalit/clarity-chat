"""A fake backend for unit tests: scripted replies, with call records shaped as llama.cpp's.

tests/test_contract.py holds it to the shapes the real backends write (tests/contract.py).
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError, Result
from clarity_chat.backends.llama_cpp import (
    NEUTRAL_REQUEST,
    PROTOCOL_REQUEST,
    build_request,
    check_generation_settings,
)
from clarity_chat.profiles import ModelEntry

# llama-server's timing names (its final chunk), which the fake reports as zero.
TIMINGS = (
    "predicted_ms",
    "predicted_n",
    "predicted_per_second",
    "predicted_per_token_ms",
    "prompt_ms",
    "prompt_n",
    "prompt_per_second",
    "prompt_per_token_ms",
)


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
        # The golden logs were written by the fake before v0.2.3, which ignored max_tokens and
        # left the end token out of the output IDs. True replays them as they were made.
        self.before_v0_2_3 = False
        self.stopped = False  # the server has exited: calls fail until the next start
        self.piece_delay = 0.0  # seconds between streamed pieces (a page can reload meanwhile)

    @property
    def n_ctx(self) -> int:
        return self._n_ctx or (self.ctx_sizes[-1] if self.ctx_sizes else 512)

    @property
    def server_record(self) -> dict[str, Any] | None:
        return self.record

    def start(self, model: ModelEntry, weights: Path, ctx_size: int) -> dict[str, Any] | None:
        same = self.started and (self.started[-1], self.ctx_sizes[-1]) == (model.name, ctx_size)
        if same and not self.stopped:
            return None
        self.stopped = False
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
        server_id = self.record["server_id"] if self.record else None
        return {
            "backend": self.name,
            "server_id": server_id,
            "build_info": "fake",
            "llama_cpp_commit": "f" * 40,
        }

    def complete(
        self,
        token_ids: list[int],
        settings: dict[str, Any],
        stop: list[str],
        on_text: Callable[[str], None] | None = None,
        request: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
        steering: Any = None,
    ) -> Result:
        self.calls.append(
            {
                "token_ids": list(token_ids),
                "settings": dict(settings),
                "stop": stop,
                "request": request,
                "steering": steering,
            }
        )
        if self.fail_with:
            raise BackendError(self.fail_with, Result(text="", token_ids=[], stop_type="none"))
        if self.stopped:  # as llama_cpp.py words it
            raise BackendError(
                "llama-server has stopped (exit code -9), so nothing was sent. "
                "Start the model again to go on.",
                server_stopped=True,
            )
        reply = self.replies.pop(0) if self.replies else " ok"
        if self.on_call:
            self.on_call()
        # The request llama-server would get, or on a replay the logged one, sent as it is.
        sent = (
            build_request(token_ids, settings, stop)
            if request is None
            else {**request, "prompt": list(token_ids)}
        )
        if steering is not None and request is None:
            sent["steering"] = steering.to_json()  # the fake stands in for Transformers here
        logged = {k: v for k, v in sent.items() if k != "prompt"}
        backend_settings = {
            k: {"value": sent[k], "origin": "backend"} for k in NEUTRAL_REQUEST if k in sent
        }
        streamed: list[str] = []

        def cancelled(text: str) -> Result:
            # As llama.cpp: the stream ends before the final chunk, so no counts or timings.
            return Result(
                text=text,
                token_ids=self._encode(text),
                stop_type="cancelled",
                cancelled_during="reply" if text else "prompt",
                request=logged,
                backend_settings=backend_settings,
                flags=["cancelled"],
            )

        cancel_after = self.cancel_after
        if cancel is not None and cancel.is_set() and cancel_after is None:
            cancel_after = 0  # as a backend checks between tokens: nothing arrived yet
        if cancel_after is not None:
            text = reply[:cancel_after]
            if on_text and text:
                on_text(text)
            return cancelled(text)
        stop_type, stopping_word = "eos", ""
        for word in stop:
            if word in reply:
                reply, stop_type, stopping_word = reply[: reply.index(word)], "word", word
        ids = self._encode(reply)
        if len(ids) > settings["max_tokens"] and not self.before_v0_2_3:
            ids = ids[: settings["max_tokens"]]
            reply = self.tokenizer.decode(ids) if self.tokenizer else reply[: len(ids)]
            stop_type, stopping_word = "limit", ""
        # In pieces, a word at a time, with a cancel checked between them as llama.cpp does.
        for piece in re.findall(r"\s*\S+\s*", reply) or [reply]:
            if cancel is not None and cancel.is_set():
                return cancelled("".join(streamed))
            streamed.append(piece)
            if on_text and piece:
                on_text(piece)
            if self.piece_delay:
                time.sleep(self.piece_delay)
        end = self.tokenizer.tok.eos_token_id if self.tokenizer else None
        if stop_type == "eos" and end is not None and not self.before_v0_2_3:
            ids.append(end)  # the end token is in the output IDs, as on both real backends
        timings = {name: 0.0 for name in TIMINGS} | {"cache_n": 0, "prompt_n": len(token_ids)}
        echo = {k: v for k, v in logged.items() if k not in PROTOCOL_REQUEST or k == "stream"}
        checks = check_generation_settings(sent, echo)
        checks["tokens_evaluated"] = {
            "ok": True,
            "client": len(token_ids),
            "server": len(token_ids),
        }
        checks["no_cache_reuse"] = {"ok": True, "cache_n": 0}
        checks["server_running"] = {"ok": True}
        response = {
            "content": reply,
            "has_new_line": "\n" in reply,
            "id_slot": 0,
            "index": 0,
            "model": "fake",
            "stop": True,
            "stop_type": stop_type,
            "stopping_word": stopping_word,
            "timings": timings,
            "tokens": [],
            "tokens_cached": len(token_ids) + len(ids),
            "tokens_evaluated": len(token_ids),
            "tokens_predicted": len(ids),
            "truncated": False,
            "generation_settings": echo,
            **({"prompt": self.tokenizer.decode(token_ids)} if self.tokenizer else {}),
        }
        return Result(
            text=reply,
            token_ids=ids,
            stop_type=stop_type,
            stopping_word=stopping_word,
            n_prompt=len(token_ids),
            n_predicted=len(ids),
            timings=timings,
            request=logged,
            response=response,
            backend_settings=backend_settings,
            flags=(["length"] if stop_type == "limit" else []) + list(self.result_flags),
            checks=checks,
            not_applied=list(self.not_applied),
            logprobs=self.logprobs,
        )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text) if self.tokenizer else list(range(len(text)))
