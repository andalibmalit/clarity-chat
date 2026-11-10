"""One running Clarity per project: `chat`, `run` and `replay` hold a lock while they run.

Two at once would load two models (more memory than a laptop has) and could append to the
same log. The lock is `local/clarity.lock`, held with the operating system's file lock, so
the system releases it however the process ends (on Windows, possibly after a short
delay). What a second process needs to know (who holds it, since when) is in
`local/clarity.json`, a separate file, because on Windows a locked file can't be read by
anyone else.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
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
        info = json.loads((root / "local" / "clarity.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def write_info(root: Path, info: dict[str, Any]) -> None:
    """Replace clarity.json in one step, readable only by you (it will hold a server key)."""
    path = root / "local" / "clarity.json"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    os.replace(tmp, path)


class Held:
    """The lock, held. Records the running llama-server, so the next start can stop it if
    this process dies without stopping it (kill -9, a crash, a closed laptop lid)."""

    def __init__(self, root: Path, info: dict[str, Any], orphan: str | None) -> None:
        self.root, self.info = root, info
        self.orphan = orphan  # what was found of a server a crashed run left behind

    def record_server(self, server: dict[str, Any] | None) -> None:
        self.info["server"] = server
        write_info(self.root, self.info)


def stop_orphan(server: dict[str, Any], wait_s: float = 120) -> str:
    """A server a crashed run left: stopped only if its port answers /props with its key and
    its model, which proves it is that server. Otherwise nothing is killed. A server still
    loading its model answers 503 to everything, so it gets up to `wait_s` to finish."""
    import time

    import httpx

    try:
        port, pid, key, model = (server[k] for k in ("port", "pid", "api_key", "model_path"))
    except (KeyError, TypeError):
        return "An earlier Clarity left a server record that can't be read; nothing was stopped."
    url = f"http://127.0.0.1:{port}/props"
    headers = {"Authorization": f"Bearer {key}"}
    deadline = time.monotonic() + wait_s
    while True:
        try:
            # Plain HTTP on 127.0.0.1, as in the llama.cpp backend: no proxy, no CA bundle.
            r = httpx.get(url, headers=headers, timeout=2, trust_env=False, verify=False)
        except httpx.HTTPError:
            return (
                f"An earlier Clarity left a server on port {port}, but nothing answers there now."
            )
        if r.status_code != 503 or time.monotonic() > deadline:
            break
        time.sleep(1)
    try:
        props = r.json() if r.status_code == 200 else {}
    except ValueError:
        props = {}
    if Path(props.get("model_path", "")) != Path(model):
        return f"Port {port} answers, but not as the server an earlier Clarity left: left alone."
    os.kill(pid, signal.SIGTERM)
    return (
        f"Stopped llama-server (pid {pid}, port {port}), left running "
        "by an earlier Clarity that didn't exit normally."
    )


def _exit_on_signal(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)  # unwinds normally: the server stops, the lock is freed


@contextmanager
def hold(root: Path, command: str) -> Generator[Held]:
    """Hold the project's lock while the block runs, or stop if another Clarity holds it.
    In the main thread, SIGTERM and SIGHUP (a closed terminal) exit through the same
    cleanup as a normal exit."""
    local = root / "local"
    local.mkdir(exist_ok=True)
    fd = os.open(local / "clarity.lock", os.O_RDWR | os.O_CREAT, 0o600)
    previous: dict[int, Any] = {}
    try:
        if not _try_lock(fd):
            info = read_info(root)
            since = info.get("since", "")[11:16] or "an unknown time"
            raise ConfigError(
                f"Clarity is already running in this project (pid {info.get('pid', '?')}, "
                f"`clarity {info.get('command', '?')}`, since {since}). Close it first."
            )
        left = read_info(root).get("server")  # only a run that didn't exit normally leaves one
        held = Held(
            root,
            {"pid": os.getpid(), "command": command, "since": datetime.now().isoformat()},
            stop_orphan(left) if left else None,
        )
        write_info(root, held.info)
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGTERM", "SIGHUP"):  # Windows has no SIGHUP
                if hasattr(signal, name):
                    number = getattr(signal, name)
                    previous[number] = signal.signal(number, _exit_on_signal)
        try:
            yield held
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            if not held.info.get("server"):  # a server that didn't stop stays recorded
                (local / "clarity.json").unlink(missing_ok=True)
    finally:
        if sys.platform == "win32":  # Microsoft asks for an unlock before the close
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass  # not locked: another instance holds it
        os.close(fd)  # on Linux and macOS, closing the descriptor releases the lock
