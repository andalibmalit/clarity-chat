"""Log records: one append-only JSONL file per conversation.

Every record has `schema_version`, `type` and `time`, and all but `run` have
`conversation_id`. Record types: `run` (the first record of a batch run's `calls.jsonl`),
`conversation`, `server_start`, `turn`, `call`, `mark`, `note`, `replay`. Readers ignore
keys they don't know, so new fields (token probabilities, reasoning splits, exports) need
no format change. A line is never rewritten; a correction is a new record pointing at the
old one.

A `mark` or `note` has `origin`: "live" (/mark, /note in a chat) or "review" (`clarity
mark`, `clarity note` on a log afterwards). One without `origin` predates the field and was
made live; the field needed no schema_version change.

Output token IDs in a `call` record are exactly what the backend generated: they include
the tokens of a stop string and an end-of-sequence token, while `output_text` (the
backend's text, as returned) leaves out the stop string and special tokens.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import subprocess
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_append_lock = threading.Lock()


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id(kind: str) -> str:
    """`call-3f9a…`; conversations and runs also start with the local time, so they sort."""
    if kind in ("conversation", "run"):
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    return f"{kind}-{secrets.token_hex(6)}"


class LogFile:
    """Append-only JSONL. Each record is one line, written in one append and synced.

    One os.write on an O_APPEND descriptor, so another process appending to the same file
    (`clarity mark` on a live chat's log) adds its line after this one, not inside it, on
    local file systems. Windows only emulates O_APPEND; the instance lock keeps a second
    chat away there, and mark and note write one short line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.servers_logged: set[str] = set()  # each server_start is written once per file

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        full = {"schema_version": SCHEMA_VERSION, "time": now_utc(), **record}
        data = memoryview((json.dumps(full, ensure_ascii=False) + "\n").encode("utf-8"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        with _append_lock:
            fd = os.open(self.path, flags, 0o644)
            try:
                while data:  # a short write (a full disk) is rare; finish the line
                    data = data[os.write(fd, data) :]
                os.fsync(fd)
            finally:
                os.close(fd)
        return full


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Records in file order. Splits only on "\\n" (never on U+2028 and friends)."""
    with path.open(encoding="utf-8", newline="\n") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --- environment facts -----------------------------------------------------------------


def _git(folder: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(folder), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def git_state(folder: Path) -> dict[str, Any]:
    """Commit and dirty flag of a git checkout. `dirty` is None if git couldn't tell."""
    head = _git(folder, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "error": head.stderr.strip()}
    status = _git(folder, "status", "--porcelain", "--untracked-files=no")
    state: dict[str, Any] = {"commit": head.stdout.strip()}
    if status.returncode == 0:
        state["dirty"] = bool(status.stdout.strip())
    else:
        state["dirty"] = None
        state["error"] = status.stderr.strip()
    return state


@cache
def client_git() -> dict[str, Any]:
    """The client's own commit and dirty flag, read once: running code can't change mid-process."""
    return git_state(Path(__file__).resolve().parent)


@cache
def machine() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "macos": _macos(),
        "hostname_hash": _hostname_hash(),
    }


def _macos() -> dict[str, str | None] | None:
    """macOS version, build and chip. Metal compiles llama.cpp's GPU kernels when the server
    starts, so a macOS update can change output; the chip picks kernel variants too."""
    if platform.system() != "Darwin":
        return None
    return {
        "version": platform.mac_ver()[0],
        "build": _command_output(["sw_vers", "-buildVersion"]),
        "chip": _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),  # None if blocked
    }


def _command_output(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _hostname_hash() -> str:
    """Tells machines apart in replay diffs without writing the hostname into logs."""
    import hashlib

    return hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:12]
