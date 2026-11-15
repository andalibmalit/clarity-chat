"""Fail a CI job whose live tests skipped or didn't run.

    uv run python scripts/junit_guard.py live.xml [--at-least N]

Reads pytest's JUnit XML (`--junitxml`) and exits 1 when a test was skipped, failed or
errored, or fewer than N ran (default 1).
The live tier skips itself when llama-server or the model is missing, and one test skips
without llama-quantize; in CI such a skip would pass silently. Exits 0, with the count,
when every test ran and passed.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def problem(path: Path, at_least: int = 1) -> str | None:
    """Why the run in `path`, pytest's JUnit XML, doesn't count; None when it does."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skips = [skip for case in cases if (skip := case.find("skipped")) is not None]
    failed = [
        case for case in cases if case.find("failure") is not None or case.find("error") is not None
    ]
    if not cases:
        return f"{path}: no test ran."
    if len(cases) < at_least:
        return f"{path}: {len(cases)} tests ran, fewer than {at_least}; was a test deselected?"
    if failed:
        return f"{path}: {len(failed)} failed or errored."
    if skips:
        reasons = "; ".join(sorted({skip.get("message", "") for skip in skips}))
        return f"{path}: {len(skips)} skipped ({reasons}); every live test must run."
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--at-least", type=int, default=1)
    args = parser.parse_args(argv)
    path = args.path
    if (message := problem(path, args.at_least)) is not None:
        print(message, file=sys.stderr)
        return 1
    tests = len(list(ET.parse(path).getroot().iter("testcase")))
    print(f"{path}: {tests} tests ran, none skipped or failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
