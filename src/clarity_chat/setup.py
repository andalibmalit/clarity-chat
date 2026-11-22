"""`clarity setup` (v0.4.2): a Mac with Apple silicon from a clone to a first chat. Stub."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3
LLAMA_URL = "https://github.com/ggml-org/llama.cpp"
SIZES: dict[str, tuple[int, int]] = {"source": (0, 0), "build": (0, 0)}


@dataclass(frozen=True)
class Mac:
    system: str
    machine: str
    memory: int
    free: int


def this_mac(root: Path) -> Mac:
    raise NotImplementedError


class Setup:
    def __init__(
        self,
        root: Path,
        mac: Mac,
        run: Callable[..., tuple[int, str]],
        ask: Callable[[str], str],
        say: Callable[[str], None],
        state: Callable[[Path], dict[str, Any]],
        home: Path,
    ) -> None:
        pass

    def go(self) -> None:
        pass
