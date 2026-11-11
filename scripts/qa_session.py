"""Drive `clarity` in a real terminal for a QA pass, and save what it printed.

    uv run --project REPO python REPO/scripts/qa_session.py --root ROOT --out FILE \
        [--steps STEPS] [--init] -- CLARITY_ARGS...

Runs `clarity --root ROOT CLARITY_ARGS` in a pseudo-terminal (macOS and Linux), types
each line of STEPS once the output has gone quiet, then Ctrl-D, and writes everything
printed to FILE. A step line `<ctrl-c>` sends Ctrl-C; `<ctrl-c after S>` sends it S
seconds after the step before it was typed, without waiting for quiet (to cancel a reply
while it streams); `<wait N>` waits N more seconds; lines starting with `#` are skipped.
Clarity runs in ROOT, so relative paths in CLARITY_ARGS are relative to ROOT.

The owner allowlists this script to run outside the sandbox, which blocks
pseudo-terminals and hides the GPU, so it keeps Clarity to the scratch root and to
SmolLM2 whatever the steps say. Every run rewrites the root's models.toml, models.lock
and profiles from this checkout, listing SmolLM2 only (Clarity loads models by name
through models.lock), and checks that server.toml differs from the checkout's in its
flags only, that the weights match models.lock's sha256, and that nothing in the root is
a link (Clarity and the driver follow links, so one could reach a private folder).
`--init` also resets server.toml. Arguments are parsed with Clarity's own parser and
steps with its own command splitter, so they are checked as Clarity will read them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import signal
import stat
import sys
import time
import tomllib
import unicodedata
from pathlib import Path
from typing import BinaryIO

REPO = Path(__file__).resolve().parents[1]
QA_DIR = REPO / "notes" / "qa"  # local-only; scratch roots, steps and transcripts
QA_COMMANDS = {"chat", "chats", "replay", "export"}
QA_MODEL = "smollm2-135m-instruct"
QA_PROFILES = ("smol-chat", "smol-text")
NAMED = {"profile", "resume", "export"}  # chat commands whose argument names a file
ID = re.compile(r"[A-Za-z0-9_-]+")  # a profile name or conversation ID: never a path
QUIET_S = 2.0  # output silent this long means the program is waiting for input
STEP_TIMEOUT_S = 600.0


class Refused(Exception):
    """A root, argument or step the driver won't run."""


# --- the scratch root ----------------------------------------------------------------


def prepare_root(root: Path, repo: Path = REPO, init: bool = False) -> None:
    """Make `root` a scratch project that can load SmolLM2 only (see the module docstring)."""
    from clarity_chat.files import sha256_file

    root.mkdir(parents=True, exist_ok=True)
    for name in ("profiles", "models", "prompts"):  # an older layout linked to the checkout
        if (root / name).is_symlink():
            (root / name).unlink()
    no_links(root)
    left_server(root)
    shutil.copy(repo / "defaults.toml", root / "defaults.toml")
    models = tomllib.loads((repo / "models.toml").read_text(encoding="utf-8"))[QA_MODEL]
    lines = [f"[{QA_MODEL}]", *(f"{k} = {json.dumps(v)}" for k, v in models.items())]
    (root / "models.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock = json.loads((repo / "models.lock").read_text(encoding="utf-8"))
    entry = lock["models"][QA_MODEL]
    lock = {**lock, "models": {QA_MODEL: entry}}
    (root / "models.lock").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    # The root's own copy of the weights, with the modified time models.lock recorded: QA
    # renames a file over it to test the restart, which mustn't touch the checkout's. Checked
    # by hash each run: Clarity checks only size and time, which can be set by hand.
    weights, sha256 = Path(entry["gguf"]["path"]), entry["gguf"]["sha256"]
    for child in (root / "models").glob("*"):
        if child.name != weights.parts[1]:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    if not (root / weights).is_file() or sha256_file(root / weights) != sha256:
        if sha256_file(repo / weights) != sha256:
            raise Refused(f"{repo / weights} doesn't match models.lock; run clarity verify.")
        (root / weights).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / weights, root / weights)
    profiles = root / "profiles"
    if profiles.exists():
        shutil.rmtree(profiles)
    profiles.mkdir()
    for name in QA_PROFILES:
        shutil.copy(repo / "profiles" / f"{name}.toml", profiles / f"{name}.toml")
    chat = (repo / "profiles" / "smol-chat.toml").read_text(encoding="utf-8")
    hf = chat.replace('backend = "llama.cpp"', 'backend = "transformers"\ndevice = "cpu"', 1)
    (profiles / "smol-chat-hf.toml").write_text(hf, encoding="utf-8", newline="\n")
    check_server_toml(root, repo, init)


def no_links(root: Path) -> None:
    """Plain files and folders only: no symbolic link, and no file with a second name (a hard
    link to a file outside the root)."""
    for folder, folders, files in os.walk(root, followlinks=False):
        for name in [*folders, *files]:
            info = (Path(folder) / name).lstat()
            if stat.S_ISLNK(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink > 1):
                raise Refused(f"{Path(folder) / name} is a link; the root holds plain files only.")


def left_server(root: Path) -> None:
    """Clarity stops a server a crashed run recorded in local/clarity.json, outside the
    sandbox. The agent can write that file, so the driver won't run on a record."""
    path = root / "local" / "clarity.json"
    if not path.exists():
        return
    try:
        server = json.loads(path.read_text(encoding="utf-8")).get("server")
    except (ValueError, AttributeError):
        server = "unreadable"
    if server:
        raise Refused(
            f"{path} records a server from an earlier run. Ask the owner to check that no "
            "llama-server is still running and to delete the file."
        )


def check_server_toml(root: Path, repo: Path, init: bool) -> None:
    """The checkout's server.toml with an absolute llama_cpp_dir; QA may change its flags."""
    text = (repo / "server.toml").read_text(encoding="utf-8")
    old = 'llama_cpp_dir = "vendor/llama.cpp"'
    if old not in text:
        raise Refused(f"server.toml has no {old!r} to rewrite; update the QA driver.")
    llama = (repo / "vendor" / "llama.cpp").as_posix()
    expected = text.replace(old, f"llama_cpp_dir = {json.dumps(llama)}")
    path = root / "server.toml"
    if init or not path.exists():
        path.write_text(expected, encoding="utf-8", newline="\n")
        return
    have = tomllib.loads(path.read_text(encoding="utf-8"))
    want = tomllib.loads(expected)
    changed = sorted(k for k in {*have, *want} - {"flags"} if have.get(k) != want.get(k))
    if changed:
        raise Refused(f"{path}: {changed} differ from the checkout's; only [flags] may change.")


# --- Clarity's arguments -------------------------------------------------------------


def check_clarity_args(root: Path, argv: list[str]) -> argparse.Namespace:
    """Parsed by Clarity's own parser: profiles by name, a log or conversation as an ID or a
    path inside the root, no --out, and only the commands a QA pass uses."""
    from clarity_chat.cli import build_parser

    if not argv or argv[0] not in QA_COMMANDS:
        raise Refused(f"the Clarity command must be one of {sorted(QA_COMMANDS)}")
    args = build_parser()[0].parse_args(["--root", str(root), *argv])
    if getattr(args, "out", None) is not None:
        raise Refused("--out isn't allowed: an export goes to the root's logs/exports/")
    if getattr(args, "profile", None) is not None and not ID.fullmatch(args.profile):
        raise Refused(f"--profile takes a profile name here, not {args.profile!r}")
    for name in ("resume", "conversation"):  # an ID, or a path
        value = getattr(args, name, None)
        if value is not None and not (ID.fullmatch(str(value)) or inside(root, str(value))):
            raise Refused(f"{value} must be an ID or a path inside {root}")
    log = getattr(args, "log", None)  # replay's LOG is always a path to Clarity
    if log is not None and not inside(root, str(log)):
        raise Refused(f"{log} must be a path inside {root}")
    return args


def inside(root: Path, value: str) -> bool:
    """A path inside the root, judged from the root, where Clarity runs."""
    path = Path(value) if Path(value).is_absolute() else root / value
    return path.resolve().is_relative_to(root.resolve())


# --- steps ---------------------------------------------------------------------------


def parse_steps(path: Path, json_mode: bool) -> list[str]:
    """The steps, checked before anything starts, so a typo can't stop a session midway and
    no step names a file outside the root. No control characters: the line editor acts on
    them (Ctrl-A moves to the start of the line), so a step could become another command."""
    from clarity_chat.cli import split_command

    steps = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line or line.startswith("#"):
            continue
        where = f"{path}:{n}"
        if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in line):
            raise Refused(f"{where}: a control character; use <ctrl-c> for Ctrl-C")
        if line.startswith("<") and line != "<ctrl-c>":
            if not re.fullmatch(r"<(wait|ctrl-c after) [0-9]+(\.[0-9]+)?>", line):
                raise Refused(f"{where}: unknown step {line!r}")
        elif json_mode:
            check_json_step(where, line)
        elif line.startswith("/"):
            name, arg = split_command(line)
            if name in NAMED and arg.strip() and not ID.fullmatch(arg.strip()):
                raise Refused(f"{where}: /{name} takes a name or ID here, not {arg!r}")
        steps.append(line)
    return steps


def check_json_step(where: str, line: str) -> None:
    try:
        step = json.loads(line)
    except ValueError:
        return  # Clarity answers with an InputError and acts on nothing
    if not isinstance(step, dict):
        return
    if not isinstance(step.get("command"), str):
        raise Refused(f"{where}: a command that isn't a string")
    if step["command"] not in NAMED:
        return
    arg = step.get("args", "")
    if not isinstance(arg, str) or (arg.strip() and not ID.fullmatch(arg.strip())):
        raise Refused(f"{where}: {step['command']} takes a name or ID here, not {arg!r}")


# --- running -------------------------------------------------------------------------


def read_until_quiet(fd: int, log: BinaryIO, timeout: float, quiet: float = QUIET_S) -> None:
    """Copy output to `log` until it has been silent for `quiet` seconds, `timeout` passes
    or the program ends."""
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
        elif time.monotonic() - last >= quiet:
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
    # Scratch paths only, never the checkout's own logs/.
    for path in (args.root, args.out, args.steps):
        if path is not None and not path.resolve().is_relative_to(QA_DIR):
            parser.error(f"{path} must be inside {QA_DIR}")
    try:
        clarity = check_clarity_args(args.root, clarity_args)
        json_mode = bool(getattr(clarity, "json", False))  # parsed: --js counts too
        steps = parse_steps(args.steps, json_mode) if args.steps else []
        prepare_root(args.root, init=args.init)
    except Refused as e:
        parser.error(str(e))
    program = Path(sys.executable).parent / "clarity"
    root = args.root.resolve()
    argv = [str(program), "--root", str(root), *clarity_args]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import pty  # POSIX only; the checks above also run in the tests on Windows

    # pty.fork: the child leads its own session with the pty as its terminal, so Ctrl-C
    # typed into the pty reaches it as SIGINT, as in a real terminal.
    pid, primary = pty.fork()
    if pid == 0:
        # In the root, so nothing resolves against the checkout. No downloads: everything QA
        # needs is fetched already.
        os.chdir(root)
        os.execve(argv[0], argv, {**os.environ, "HF_HUB_OFFLINE": "1"})
    with args.out.open("wb") as log:
        log.write(f"$ clarity --root {args.root} {' '.join(clarity_args)}\n".encode())
        read_until_quiet(primary, log, STEP_TIMEOUT_S)
        for n, step in enumerate(steps, 1):
            log.write(f"\n[qa step {n}: {step}]\n".encode())
            if step == "<ctrl-c>" or step.startswith("<ctrl-c after "):
                os.write(primary, b"\x03")
            elif step.startswith("<wait "):
                time.sleep(float(step[6:-1]))
            else:
                os.write(primary, step.encode() + b"\r")
            later = steps[n] if n < len(steps) else ""
            if timed := re.fullmatch(r"<ctrl-c after ([0-9.]+)>", later):
                read_until_quiet(primary, log, float(timed[1]), quiet=float("inf"))
            else:
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
