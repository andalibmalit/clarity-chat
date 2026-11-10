"""Log records: one append-only JSONL file per conversation.

Every record has `schema_version`, `type` and `time`, and all but `run` and `torn_line`
have `conversation_id`. Record types: `run` (the first record of a batch run's
`calls.jsonl`), `conversation`, `server_start`, `turn`, `call`, `mark`, `note`, `replay`,
`torn_line` (the line before it was cut off). Readers ignore
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
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from clarity_chat.profiles import ConfigError

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
    chat away there, and mark and note write one short line.

    A last line with no line end was cut off (a crash or a full disk while it was written).
    The next append ends it and adds a `torn_line` record naming it, in the same write.
    Nothing is rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.servers_logged: set[str] = set()  # each server_start is written once per file
        self.on_append: Callable[[dict[str, Any]], None] | None = None  # sees each record

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        full = {"schema_version": SCHEMA_VERSION, "time": now_utc(), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        with _append_lock:
            fd = os.open(self.path, flags, 0o644)
            try:
                torn = _torn_tail(fd)
                lines = (b"\n" + _line(torn) if torn else b"") + _line(full)
                data = memoryview(lines)
                while data:  # a short write (a full disk) is rare; finish the line
                    data = data[os.write(fd, data) :]
                os.fsync(fd)
            finally:
                os.close(fd)
        if self.on_append:
            if torn:
                self.on_append(torn)
            self.on_append(full)
        return full


def _line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _torn_tail(fd: int) -> dict[str, Any] | None:
    """A `torn_line` record if the file's last line has no line end, else None."""
    size = os.lseek(fd, 0, os.SEEK_END)
    if size == 0:
        return None
    os.lseek(fd, -1, os.SEEK_END)
    if os.read(fd, 1) == b"\n":
        return None
    os.lseek(fd, 0, os.SEEK_SET)
    newlines, last_newline, read = 0, -1, 0
    # Only up to `size`: another process may be appending meanwhile.
    while read < size and (chunk := os.read(fd, min(1 << 20, size - read))):
        newlines += chunk.count(b"\n")
        if (i := chunk.rfind(b"\n")) >= 0:
            last_newline = read + i
        read += len(chunk)
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "torn_line",
        "time": now_utc(),
        "line": newlines + 1,
        "bytes": size - last_newline - 1,
    }


def torn_message(record: dict[str, Any], where: str) -> str:
    return (
        f"Line {record['line']} of {where} has no line end, so it was cut off (a crash, a "
        "full disk, or another program). It stays as it was; a torn_line record after it "
        "says so."
    )


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Records in file order. Splits only on "\\n" (never on U+2028 and friends). A last
    line with no line end isn't a record yet. Lines that aren't records are skipped only if
    the next record is a `torn_line` naming the last of them."""
    bad: list[int] = []  # malformed lines in a row, until the next record explains them
    with path.open("rb") as f:  # bytes split only on b"\n"; a broken UTF-8 line is malformed
        for number, line in enumerate(f, 1):
            if not line.endswith(b"\n"):
                break  # not finished: still being written, or cut off (the next append says)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:  # includes UnicodeDecodeError
                bad.append(number)
                continue
            if bad:
                # A torn_line names the last of them: a mending write can be cut off too.
                if record.get("type") != "torn_line" or record.get("line") != bad[-1]:
                    break
                bad = []
            yield record
    if bad:
        raise ConfigError(
            f"Line {bad[0]} of {path.name} isn't a complete record, and no torn_line record "
            f"says it was cut off. Clarity won't guess what it held; the file is at {path}."
        )


TITLE_WORDS = 8  # a title that isn't set: the first words of the first message


def list_chats(logs_dir: Path) -> list[dict[str, Any]]:
    """The conversations in `logs_dir`, newest first (their IDs start with the time). A log
    that can't be read is listed with its `error`, so one bad file doesn't hide the rest."""
    chats = []
    for path in sorted(logs_dir.glob("*.jsonl"), reverse=True):
        try:
            chats.append(chat_summary(path))
        except (OSError, ValueError, ConfigError) as e:
            chats.append({"conversation_id": path.stem, "path": path.as_posix(), "error": str(e)})
    return chats


def chat_summary(path: Path) -> dict[str, Any]:
    """A conversation's ID, profile, model, start time and title. Reads the log as far as
    its first message; after that, only lines that may be `title` records (`/title`)."""
    summary: dict[str, Any] = {"conversation_id": path.stem, "path": path.as_posix()}
    summary.update(profile=None, model=None, started=None, title=None, title_set=False)
    first_message = False
    with path.open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break  # not finished (see read_records)
            if first_message and b'"type": "title"' not in line:
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "conversation":
                summary["profile"] = (record.get("profile") or {}).get("name")
                summary["model"] = (record.get("model") or {}).get("name")
                summary["started"] = record.get("time")
            elif kind == "turn" and record.get("role") == "user" and not first_message:
                first_message = True
                if not summary["title_set"]:
                    summary["title"] = " ".join(record["text"].split()[:TITLE_WORDS])
            elif kind == "title":
                summary["title"], summary["title_set"] = record["title"], True
    return summary


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
