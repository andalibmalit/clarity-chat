"""Meaning search's embedder (v0.4.1 A1): a bare stub, so its tests fail on behaviour."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx  # noqa: F401 (the tests' fake server replaces httpx.post here)

from clarity_chat.profiles import ServerConfig

MODEL = "bge-m3"
DIMENSIONS = 1024
MAX_TOKENS = 8192
INPUT_RULE = 1
FLAGS: dict[str, Any] = {}


class EmbedderFailed(Exception):
    pass


class TooLong(Exception):
    def __init__(self, tokens: int) -> None:
        super().__init__(tokens)
        self.tokens = tokens


def setup_of(facts: dict[str, Any]) -> str:
    return ""


class Embedder:
    def __init__(
        self,
        root: Path,
        server_config: ServerConfig,
        logs_dir: Path,
        load_tokenizer: Callable[[str, str], Any],
    ) -> None:
        self.root = root
        self.server: Any = None

    def ids(self, text: str) -> list[int]:
        return []

    def start(self) -> dict[str, Any]:
        return {}

    def vector(self, ids: list[int]) -> bytes:
        return b""

    def close(self) -> None:
        pass
