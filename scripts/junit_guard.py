"""Fail a CI job whose live tests skipped or didn't run (stub)."""

from __future__ import annotations

import sys
from pathlib import Path


def problem(path: Path) -> str | None:
    """Why the run in `path`, pytest's JUnit XML, doesn't count; None when it does."""
    return None


def main(argv: list[str]) -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
