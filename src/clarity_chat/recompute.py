"""The recompute check, for a notebook: what differs between an export's conditions and a
loaded model, and how far the logged probabilities move when recomputed under it."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def installed(package: str) -> str | None:
    """A package's installed version, without importing it (torch is optional)."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def check(
    export: str | Path, model: Any, *, repo: str | None = None, revision: str | None = None
) -> Any:
    raise NotImplementedError
