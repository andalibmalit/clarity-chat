"""Copy named models' entries from local/models.lock to the shipped models.lock.

    uv run python scripts/refresh_lock.py NAME [NAME …] [--root PATH]

Every fetch writes local/models.lock (v0.4.2 A10), so the shipped lock changes only here:
the maintainer fetches a shipped model at its new pin, then copies its entry. A name not in
local/models.lock, or not in the shipped models.toml (your own entries never go in the
shipped lock), is refused, and nothing is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clarity_chat.models import LOCK_FILE, read_lock_file, write_lock_file
from clarity_chat.profiles import LOCAL_LOCK, read_toml


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
    lock = read_lock_file(args.root / LOCK_FILE)
    lock["models"] |= {name: local[name] for name in args.names}
    write_lock_file(args.root / LOCK_FILE, lock)
    for name in args.names:
        print(f"models.lock: {name} copied from local/models.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
