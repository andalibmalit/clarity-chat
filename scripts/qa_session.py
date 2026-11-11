"""Drive `clarity` in a real terminal for a QA pass, and save what it printed.

    uv run --project REPO python REPO/scripts/qa_session.py --root ROOT --out FILE \
        [--steps STEPS] -- CLARITY_ARGS...

Runs `clarity --root ROOT CLARITY_ARGS` in a pseudo-terminal (macOS and Linux), types
each line of STEPS once the output has gone quiet, and writes everything printed to FILE.
A step line `<ctrl-c>` sends Ctrl-C; `<wait N>` waits N more seconds; lines starting
with `#` are skipped. `--init` first makes ROOT a scratch project that uses this
checkout's config, profiles and fetched models, so a QA chat never writes to the
checkout's own `logs/`. It runs only Clarity, so the owner can allowlist it to reach
the GPU (the sandbox hides it).
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import shutil
import subprocess
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
    server = server.replace('llama_cpp_dir = "vendor/llama.cpp"', f'llama_cpp_dir = "{llama}"')
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("clarity_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    clarity_args = [a for a in args.clarity_args if a != "--"]
    # Allowlisted outside the sandbox, so it stays narrow: scratch paths only (never the
    # checkout's own logs/), and only the commands a QA pass uses (no fetch: no downloads).
    for path in (args.root, args.out, args.steps):
        if path is not None and not path.resolve().is_relative_to(QA_DIR):
            parser.error(f"{path} must be inside {QA_DIR}")
    if not clarity_args or clarity_args[0] not in QA_COMMANDS:
        parser.error(f"the Clarity command must be one of {sorted(QA_COMMANDS)}")
    if args.init:
        init_root(args.root)
    steps = []
    if args.steps:
        for line in args.steps.read_text(encoding="utf-8").splitlines():
            if not line.startswith("#"):
                steps.append(line)
    clarity = Path(sys.executable).parent / "clarity"
    argv = [str(clarity), "--root", str(args.root), *clarity_args]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    primary, secondary = pty.openpty()
    proc = subprocess.Popen(argv, stdin=secondary, stdout=secondary, stderr=secondary)
    os.close(secondary)
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
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
        read_until_quiet(primary, log, 1.0)
        log.write(f"\n[qa exit status {proc.returncode}]\n".encode())
    os.close(primary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
