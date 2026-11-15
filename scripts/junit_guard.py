"""Fail a CI job whose live tests skipped or didn't run.

    uv run python scripts/junit_guard.py live.xml

Reads pytest's JUnit XML (`--junitxml`) and exits 1 when a test was skipped or none ran.
The live tier skips itself when llama-server or the model is missing, and one test skips
without llama-quantize; in CI such a skip would pass silently. Exits 0, with the count,
when every test ran. pytest's own exit status covers failures.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def counts(path: Path) -> tuple[int, list[str]]:
    """How many tests the run in `path` (pytest's JUnit XML) holds, and each skip's reason."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skips = [skip for case in cases if (skip := case.find("skipped")) is not None]
    return len(cases), [skip.get("message", "") for skip in skips]


def problem(path: Path) -> str | None:
    """Why the run in `path` doesn't count; None when it does."""
    tests, skips = counts(path)
    if not tests:
        return f"{path}: no test ran."
    if skips:
        reasons = "; ".join(sorted(set(skips)))
        return f"{path}: {len(skips)} skipped ({reasons}); every live test must run."
    return None


def main(argv: list[str]) -> int:
    (name,) = argv
    path = Path(name)
    if (message := problem(path)) is not None:
        print(message, file=sys.stderr)
        return 1
    print(f"{path}: {counts(path)[0]} tests ran, none skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
