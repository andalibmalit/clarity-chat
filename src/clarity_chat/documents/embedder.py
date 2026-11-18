"""Meaning search's embedder: bge-m3 on a second llama-server (v0.4.1 decisions 3 to 6).

Its flags are constants here, never server.toml's or a profile's, so a change to the chat
server's flags never changes a vector. Each text goes alone, as bge-m3's own token IDs, and
each vector is checked before it's used. A failure stops the embedder; the next need
starts it again.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import struct
from collections.abc import Callable
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx

from clarity_chat.backends import BackendError
from clarity_chat.backends.llama_cpp import LlamaServer
from clarity_chat.logs import machine
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, ServerConfig, load_models

MODEL = "bge-m3"
DIMENSIONS = 1024
MAX_TOKENS = 8192  # what bge-m3 reads; attention that isn't causal needs it in one ubatch
INPUT_RULE = 1  # a part's input: its title, a blank line, then its text (decision 7)
FLAGS: dict[str, Any] = {
    "embedding": True,
    "pooling": "cls",  # bge-m3's 1_Pooling/config.json
    "embd-normalize": 2,  # L2, as its sentence-transformers modules do
    "parallel": 1,  # one slot: each input is evaluated alone
    "no-kv-unified": True,
    "ctx-size": MAX_TOKENS,
    "batch-size": MAX_TOKENS,
    "ubatch-size": MAX_TOKENS,
    "n-gpu-layers": "all",
    "flash-attn": "on",  # no full attention matrix at 8,192 tokens (decision 5)
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "cache-ram": 0,
    "cache-reuse": 0,
    "ctx-checkpoints": 0,
    "no-cache-idle-slots": True,
    "no-context-shift": True,
    "fit": "off",  # layers never move to the CPU on a tight machine
    "threads": 4,
    "threads-batch": 4,
    "log-verbosity": 4,  # the placement check reads lines logged at level 4
    "no-jinja": True,
    "chat-template": "chatml",
}


def input_of(title: str | None, text: str) -> str:
    """A part's input, rule 1 (decision 7): its title, a blank line, then its text; the text
    alone when it has no title."""
    return f"{title}\n\n{text}" if title else text


# The facts a meaning found's `engine` logs, beside `setup`, their hash (decision 9).
ENGINE_FACTS = (
    "name", "model", "gguf_sha256", "tokenizer_repo", "tokenizer_revision", "tokenizers",
    "llama_cpp", "flags", "input_rule", "device", "machine",
)  # fmt: skip


class EmbedderFailed(Exception):
    """The embedder couldn't start, stopped, or sent a vector that failed a check."""


class TooLong(Exception):
    """A text past MAX_TOKENS: refused, never cut (decision 8)."""

    def __init__(self, tokens: int) -> None:
        super().__init__(tokens)
        self.tokens = tokens


def setup_of(facts: dict[str, Any]) -> str:
    """The setup: the sha256 of the embedder's facts as canonical JSON (decision 9)."""
    text = json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(text.encode("utf-8")).hexdigest()


def machine_facts() -> dict[str, Any]:
    """The OS version and build, and the chip: Metal compiles its kernels for them."""
    m = machine()
    if m["macos"]:
        mac = m["macos"]
        return {"os": f"macOS {mac['version']}", "build": mac["build"], "chip": mac["chip"]}
    return {"os": f"{m['os']} {m['os_release']}", "build": m["os_version"], "chip": m["machine"]}


def checked(body: Any, sent: int) -> bytes:
    """The one vector in a /v1/embeddings reply; a ValueError says what's wrong with it."""
    data = body["data"]
    if len(data) != 1 or data[0]["index"] != 0:
        raise ValueError(f"{len(data)} vectors for one input")
    vector = base64.b64decode(data[0]["embedding"], validate=True)
    if len(vector) != DIMENSIONS * 4:
        raise ValueError(f"{len(vector) // 4} values, not {DIMENSIONS}")
    values = struct.unpack(f"<{DIMENSIONS}f", vector)
    if not all(map(math.isfinite, values)):
        raise ValueError("a value that isn't a finite number")
    norm = math.sqrt(math.fsum(x * x for x in values))
    if abs(norm - 1) > 1e-3:  # llama-server sends zeros when a lookup fails
        raise ValueError(f"length {norm:.6g}, not 1")
    if body["usage"]["prompt_tokens"] != sent:
        raise ValueError(f"{body['usage']['prompt_tokens']} tokens counted, {sent} sent")
    return vector


class Embedder:
    """bge-m3's llama-server: started by the first vector a session needs, again after a
    failure, and stopped at close."""

    def __init__(
        self,
        root: Path,
        server_config: ServerConfig,
        logs_dir: Path,
        load_tokenizer: Callable[[str, str], Any],
    ) -> None:
        self.root = root
        self.config = dataclasses.replace(server_config, flags=dict(FLAGS))
        self.logs_dir = logs_dir
        self.load_tokenizer = load_tokenizer
        self.server: LlamaServer | None = None
        self.engine: dict[str, Any] = {}  # the setup's facts, while it runs

    def ids(self, text: str) -> list[int]:
        """`text`'s token IDs from bge-m3's own tokenizer, with <s> and </s>."""
        model = load_models(self.root)[MODEL]
        tokenizer = self.load_tokenizer(model.tokenizer_repo, model.tokenizer_revision)
        ids = list(tokenizer.tok.encode(text, add_special_tokens=True))
        if len(ids) > MAX_TOKENS:
            raise TooLong(len(ids))
        return ids

    def start(self) -> dict[str, Any]:
        """Start it unless it runs; its facts (found.engine). Not fetched is a ConfigError,
        before any process; a process that died since its start fails this need."""
        if self.server is not None:
            proc = self.server.proc
            if proc is None or proc.poll() is not None:
                self.close()
                raise EmbedderFailed("The embedder stopped. Clarity starts it again next time.")
            return self.engine
        model = load_models(self.root)[MODEL]
        entry = lock_entry(self.root, model)
        self.server = LlamaServer(self.config, check_weights_quick(self.root, entry), self.logs_dir)
        try:
            record = self.server.start()
            if record["props"].get("total_slots") != 1:
                raise BackendError(
                    f"llama-server reports {record['props'].get('total_slots')!r} slots, not 1"
                )
        except (ConfigError, BackendError) as e:
            self.close()
            raise EmbedderFailed(f"The embedder didn't start: {e}") from e
        facts = {
            "name": "llama-server-embedding",
            "model": MODEL,
            "gguf_sha256": entry["gguf"]["sha256"],
            "tokenizer_repo": model.tokenizer_repo,
            "tokenizer_revision": model.tokenizer_revision,
            "tokenizers": version("tokenizers"),
            "llama_cpp": self.config.llama_cpp_commit,
            "flags": FLAGS,
            "input_rule": INPUT_RULE,
            "device": ",".join(d["name"] for d in record["placement"]["devices"]) or "CPU",
            "machine": machine_facts(),
        }
        self.engine = {**facts, "setup": setup_of(facts)}
        return self.engine

    def vector(self, ids: list[int]) -> bytes:
        """`ids`' vector: 1,024 little-endian float32s, checked (decision 6)."""
        self.start()
        assert self.server is not None
        try:
            r = httpx.post(
                f"{self.server.url}/v1/embeddings",
                headers=self.server.headers,
                json={"input": [ids], "encoding_format": "base64", "embd_normalize": 2},
                timeout=120,
                trust_env=False,
                verify=False,
            )
            r.raise_for_status()
            return checked(r.json(), len(ids))
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as e:
            self.close()
            raise EmbedderFailed(f"The embedder sent a vector Clarity can't use: {e}") from e

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.server, self.engine = None, {}
