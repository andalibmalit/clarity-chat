"""Copy named models' entries from local/models.lock to the shipped models.lock.

    uv run python scripts/refresh_lock.py NAME [NAME …] [--root PATH]

Every fetch writes local/models.lock (v0.4.2 A10), so the shipped lock changes only here:
the maintainer fetches a shipped model at its new pin, then copies its entry. A name not in
local/models.lock, or not in the shipped models.toml (your own entries never go in the
shipped lock), is refused, and nothing is written. So is an entry models.toml no longer
pins (lock_entry's check), or one with a path outside Clarity's folder and the home folder,
which would carry this computer's layout into the shipped lock; a command's first word, the
program that ran it, may be anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from clarity_chat.models import LOCK_FILE, lock_entry, read_lock_file, write_lock_file
from clarity_chat.profiles import LOCAL_LOCK, ConfigError, read_models, read_toml


def paths(entry: dict[str, Any]) -> list[str]:
    """The paths an entry records: portable_path made each relative or ~/ when it could.
    A command's first word is the program that ran it (uv from Homebrew, say), not a path
    into anyone's folder, so it isn't one of them."""
    command = entry.get("converter", {}).get("command") or []
    words = [word for argv in command for word in argv[1:]]
    fields = [
        entry.get("source_dir"),
        entry.get("tokenizer_dir"),
        entry.get("gguf", {}).get("path"),
    ]
    files = [item.get("file") for item in entry.get("source_files", [])]  # v0.4.7 R
    return [text for text in [*fields, *files, *words] if isinstance(text, str)]


def absolute(text: str) -> bool:
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", metavar="NAME")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    local = read_lock_file(args.root / LOCAL_LOCK)["models"]
    shipped = set(read_toml(args.root / "models.toml"))
    for name in args.names:
        if name not in local:
            print(f"{name} isn't in local/models.lock: fetch it here first.", file=sys.stderr)
            return 1
        if name not in shipped:
            print(f"{name} isn't in models.toml, so it isn't a shipped model.", file=sys.stderr)
            return 1
        try:
            lock_entry(args.root, read_models(args.root / "models.toml")[name])
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 1
        for text in paths(local[name]):
            if absolute(text):
                print(
                    f"{name}'s entry in local/models.lock has the path {text}, outside "
                    "Clarity's folder and the home folder, so it isn't copied to the shipped "
                    "lock. Fetch it again with the Hugging Face cache under the home folder.",
                    file=sys.stderr,
                )
                return 1
    lock = read_lock_file(args.root / LOCK_FILE)
    lock["models"] |= {name: local[name] for name in args.names}
    write_lock_file(args.root / LOCK_FILE, lock)
    for name in args.names:
        print(f"models.lock: {name} copied from local/models.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
