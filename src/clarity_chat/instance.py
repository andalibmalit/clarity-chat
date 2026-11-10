"""One running Clarity per project: `chat`, `run` and `replay` hold a lock while they run.

Two at once would load two models (more memory than a laptop has) and could append to the
same log. The lock is `local/clarity.lock`, held with the operating system's file lock, so
the kernel releases it however the process ends. What a second process needs to know (who
holds it, since when) is in `local/clarity.json`, a separate file, because on Windows a
locked file can't be read by anyone else.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from clarity_chat.profiles import ConfigError

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def _try_lock(fd: int) -> bool:
    try:
        if sys.platform == "win32":
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def read_info(root: Path) -> dict[str, Any]:
    """What the last holder wrote; empty if nothing (or nothing readable) is there."""
    try:
        return json.loads((root / "local" / "clarity.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_info(root: Path, info: dict[str, Any]) -> None:
    """Replace clarity.json in one step, readable only by you (it will hold a server key)."""
    path = root / "local" / "clarity.json"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    os.replace(tmp, path)


@contextmanager
def hold(root: Path, command: str) -> Generator[None]:
    """Hold the project's lock while the block runs, or stop if another Clarity holds it."""
    local = root / "local"
    local.mkdir(exist_ok=True)
    fd = os.open(local / "clarity.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not _try_lock(fd):
            info = read_info(root)
            since = info.get("since", "")[11:16] or "an unknown time"
            raise ConfigError(
                f"Clarity is already running in this project (pid {info.get('pid', '?')}, "
                f"`clarity {info.get('command', '?')}`, since {since}). Close it first."
            )
        write_info(
            root,
            {"pid": os.getpid(), "command": command, "since": datetime.now().isoformat()},
        )
        try:
            yield
        finally:
            (local / "clarity.json").unlink(missing_ok=True)
    finally:
        os.close(fd)  # closing the descriptor releases the lock
