"""Drive `clarity` in a real terminal for a QA pass, and save what it printed.

    uv run --project REPO python REPO/scripts/qa_session.py --root ROOT --out FILE \
        [--steps STEPS] -- CLARITY_ARGS...

Runs `clarity --root ROOT CLARITY_ARGS` in a pseudo-terminal (macOS and Linux), types
each line of STEPS once the output has gone quiet, then Ctrl-D, and writes everything
printed to FILE. A step line `<ctrl-c>` sends Ctrl-C; `<wait N>` waits N more seconds;
lines starting with `#` are skipped. `--init` first makes ROOT a scratch project that
uses this checkout's config, profiles and fetched models, so a QA chat never writes to
the checkout's own `logs/`. The owner allowlists it to run outside the sandbox, which
blocks pseudo-terminals and hides the GPU. So it stays narrow: scratch paths only,
profiles by name, logs only inside the scratch root, and no `--out`.
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import BinaryIO

REPO = Path(__file__).resolve().parents[1]
QA_DIR = REPO / "notes" / "qa"  # local-only; scratch roots, steps and transcripts
QA_COMMANDS = {"chat", "chats", "replay", "export"}
QUIET_S = 2.0  # output silent this long means the program is waiting for input
STEP_TIMEOUT_S = 600.0


def init_root(root: Path) -> None:
    """A scratch project: copies of the config files, links to profiles and models."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("defaults.toml", "models.toml", "models.lock"):
        shutil.copy(REPO / name, root / name)
    server = (REPO / "server.toml").read_text(encoding="utf-8")
    llama = (REPO / "vendor" / "llama.cpp").as_posix()
    old = 'llama_cpp_dir = "vendor/llama.cpp"'
    if old not in server:
        raise SystemExit(f"server.toml has no {old!r} to rewrite; update init_root.")
    server = server.replace(old, f'llama_cpp_dir = "{llama}"')
    (root / "server.toml").write_text(server, encoding="utf-8", newline="\n")
    for name in ("profiles", "prompts", "models"):
        link = root / name
        if not link.exists():
            link.symlink_to(REPO / name, target_is_directory=True)


def read_until_quiet(fd: int, log: BinaryIO, timeout: float) -> None:
    """Copy output to `log` until it has been silent for QUIET_S or the program ends."""
    deadline = time.monotonic() + timeout
    last = time.monotonic()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:  # the program ended
                return
            if not data:
                return
            log.write(data)
            log.flush()
            last = time.monotonic()
        elif time.monotonic() - last >= QUIET_S:
            return


# Chat commands whose argument names a file or folder: a bare name or ID only.
NAMED = ("/profile", "/resume", "/export")


def bare(value: str) -> bool:
    """A profile name or conversation ID, not a path."""
    return bool(value) and not any(c in value for c in "/\\:~") and not value.startswith(".")


def parse_steps(path: Path) -> list[str]:
    """The steps, checked before anything starts, so a typo can't stop a session midway and
    no step names a file outside the scratch root."""
    steps = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#"):
            continue
        command, _, arg = line.partition(" ")
        if command in NAMED and arg and not bare(arg.strip()):
            raise SystemExit(f"{path}:{n}: {command} takes a name or ID here, not {arg!r}")
        if line.startswith("<") and line != "<ctrl-c>":
            try:
                assert line.startswith("<wait ") and line.endswith(">")
                float(line[6:-1])
            except (AssertionError, ValueError):
                raise SystemExit(f"{path}:{n}: unknown step {line!r}") from None
        steps.append(line)
    return steps


def check_clarity_args(parser: argparse.ArgumentParser, root: Path, argv: list[str]) -> None:
    """Profiles by name; a log or conversation as an ID or a path inside the scratch root;
    no --out."""
    command, rest = argv[0], argv[1:]
    for i, arg in enumerate(rest):
        if arg == "--out" or arg.startswith("--out="):
            parser.error("--out isn't allowed: an export goes to the scratch root's logs/exports/")
        if arg.startswith("-"):
            continue
        before = rest[i - 1] if i else ""
        if before == "--profile" and not bare(arg):
            parser.error(f"--profile takes a profile name here, not {arg!r}")
        named = before in ("--resume",) or (command in ("replay", "export") and i == 0)
        if named and not bare(arg) and not Path(arg).resolve().is_relative_to(root.resolve()):
            parser.error(f"{arg} must be an ID or a path inside {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("clarity_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    clarity_args = [a for a in args.clarity_args if a != "--"]
    # Scratch paths only (never the checkout's own logs/), and only the commands a QA pass
    # uses (no fetch: no downloads).
    for path in (args.root, args.out, args.steps):
        if path is not None and not path.resolve().is_relative_to(QA_DIR):
            parser.error(f"{path} must be inside {QA_DIR}")
    if not clarity_args or clarity_args[0] not in QA_COMMANDS:
        parser.error(f"the Clarity command must be one of {sorted(QA_COMMANDS)}")
    check_clarity_args(parser, args.root, clarity_args)
    steps = parse_steps(args.steps) if args.steps else []
    if args.init:
        init_root(args.root)
    clarity = Path(sys.executable).parent / "clarity"
    argv = [str(clarity), "--root", str(args.root), *clarity_args]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # pty.fork: the child leads its own session with the pty as its terminal, so Ctrl-C
    # typed into the pty reaches it as SIGINT, as in a real terminal.
    pid, primary = pty.fork()
    if pid == 0:
        os.execv(argv[0], argv)
    with args.out.open("wb") as log:
        log.write(f"$ clarity --root {args.root} {' '.join(clarity_args)}\n".encode())
        read_until_quiet(primary, log, STEP_TIMEOUT_S)
        for n, step in enumerate(steps, 1):
            log.write(f"\n[qa step {n}: {step}]\n".encode())
            if step == "<ctrl-c>":
                os.write(primary, b"\x03")
            elif step.startswith("<wait "):
                time.sleep(float(step[6:-1]))
            else:
                os.write(primary, step.encode() + b"\r")
            read_until_quiet(primary, log, STEP_TIMEOUT_S)
        if steps:
            log.write(b"\n[qa end: ctrl-d]\n")
            os.write(primary, b"\x04")  # end of input: the chat quits as a user's would
        read_until_quiet(primary, log, STEP_TIMEOUT_S)
        status = wait(pid)
        log.write(f"\n[qa exit status {status}]\n".encode())
    os.close(primary)
    return 0


def wait(pid: int) -> int:
    """The child's exit status; stopped with SIGTERM if it doesn't end within 30 s."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.2)
    os.kill(pid, signal.SIGTERM)
    return os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])


if __name__ == "__main__":
    sys.exit(main())
