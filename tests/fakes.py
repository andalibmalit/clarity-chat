"""A fake backend for unit tests: scripted replies, with call records shaped as llama.cpp's.

tests/test_contract.py holds it to the shapes the real backends write (tests/contract.py).
"""

from __future__ import annotations

import os
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


class FakeEmbedder:
    """documents.embedder.Embedder offline (v0.4.1 A3): made-up but deterministic vectors,
    each word hashed to one of the 1,024 places and the sum normalized, so made-up texts
    sharing words score higher. What it imitates is gated live in tests/test_embed.py:
    <s> … </s> around the IDs and the refusal past 8,192 (the boundary and no-truncation
    tests); one vector per input (T2); a unit vector, checked (T1, and test_embedder's
    checks). `device` is a setup fact a test can change; `fail_at` makes the nth vector
    fail as the real one's checks would; `made` counts vectors; `texts` lists each text
    `ids()` was given, in order (its word split can't see whitespace, so a test of the input
    rule reads these)."""

    def __init__(self, device: str = "MTL0") -> None:
        self.device = device
        self.fail_at: int | None = None
        self.made = 0
        self.texts: list[str] = []
        self.is_fetched = True  # False: bge-m3 not fetched (state 5)
        # A line its next start says once, as the real one's A5 re-check does
        self.notice: str | None = None
        self.on_notice: Callable[[str], None] | None = None

    def fetched(self) -> bool:
        return self.is_fetched

    def ids(self, text: str) -> list[int]:
        import hashlib

        from clarity_chat.documents import TooLong

        self.texts.append(text)
        words = re.findall(r"\w+", text.lower())
        ids = [0, *(3 + int(hashlib.sha256(w.encode()).hexdigest()[:6], 16) for w in words), 2]
        if len(ids) > 8192:
            raise TooLong(len(ids))
        return ids

    def start(self) -> dict[str, Any]:
        from clarity_chat.documents.embedder import FLAGS, setup_of

        facts = {
            "name": "llama-server-embedding",
            "model": "bge-m3",
            "gguf_sha256": "0" * 64,
            "tokenizer_repo": "BAAI/bge-m3",
            "tokenizer_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "tokenizers": "0.0",
            "llama_cpp": "0" * 40,
            "flags": FLAGS,
            "input_rule": 1,
            "device": self.device,
            "machine": {"os": "made up", "build": "0", "chip": "made up"},
        }
        if self.notice is not None and self.on_notice is not None:
            self.on_notice(self.notice)
            self.notice = None
        return {**facts, "setup": setup_of(facts)}

    def vector(self, ids: list[int]) -> bytes:
        import math
        import struct

        from clarity_chat.documents import EmbedderFailed

        self.made += 1
        if self.made == self.fail_at:
            raise EmbedderFailed("The embedder sent a vector Clarity can't use: made up.")
        values = [0.0] * 1024
        for i in ids[1:-1]:
            values[i % 1024] += 1.0
        values[0] += 0.0 if any(values) else 1.0  # no words: still a unit vector
        norm = math.sqrt(sum(v * v for v in values))
        return struct.pack("<1024f", *(v / norm for v in values))

    def close(self) -> None:
        pass


class FakeRunner:
    """setup's `run(argv, env)` offline (v0.4.2 A2): records each argv and environment and
    plays back a scripted result, (exit code, output), in order, for git's and cmake's
    steps; anything unscripted exits 0 with no output. `--version` isn't scripted: cmake
    reports 4.4.5, and llama-server `llama_commit` (the pin's short form by default).
    `effects` maps a step to a callable that leaves the files the real step would: `init`
    leaves FOLDER/.git (setup's never-touch check reads it), `build` leaves BUILD/bin's
    llama-server. What it imitates is gated live in tests/test_setup_live.py (-m setup): the
    shallow fetch by full hash and the build."""

    def __init__(
        self, results: list[tuple[int, str]] | None = None, llama_commit: str | None = None
    ) -> None:
        from clarity_chat.profiles import load_server_config
        from tests.conftest import REPO

        self.results = list(results or [])
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.llama_commit = llama_commit or load_server_config(REPO).llama_cpp_commit[:7]
        self.effects: dict[str, Any] = {
            "init": lambda argv: (Path(argv[2]) / ".git").mkdir(),
            "build": self.built,
        }

    @staticmethod
    def step(argv: list[str]) -> str:
        if argv[:2] == ["git", "-C"] and len(argv) > 3:
            return argv[3]
        if "--version" in argv:
            return "version"
        return "build" if "--build" in argv else "configure" if "-S" in argv else argv[0]

    def built(self, argv: list[str]) -> None:
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        binary = Path(argv[argv.index("--build") + 1]) / "bin" / exe
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"made up")

    def __call__(self, argv: list[str], env: dict[str, str]) -> tuple[int, str]:
        self.calls.append((list(argv), dict(env)))
        step = self.step(argv)
        if step == "version":
            if Path(argv[0]).name.startswith("cmake"):
                return 0, "cmake version 4.4.5\n"
            commit = self.llama_commit
            return 0, f"version: 1 (build 1, commit {commit})\nbuilt with made-up for arm64\n"
        if step in self.effects:
            self.effects[step](argv)
        return self.results.pop(0) if self.results else (0, "")

    @property
    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]

    def of(self, step: str) -> list[list[str]]:
        """The argvs of one kind of step: "git", "cmake" (configure and build) or "version"."""
        kinds = {
            "git": lambda a: a[0] == "git",
            "cmake": lambda a: self.step(a) in ("configure", "build"),
            "version": lambda a: self.step(a) == "version",
        }
        return [argv for argv in self.argvs if kinds[step](argv)]
