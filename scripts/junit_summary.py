"""Write a browser run's counts and failures as Markdown, for a CI job's summary.

    uv run python scripts/junit_summary.py browser.xml ENGINE >> "$GITHUB_STEP_SUMMARY"

Reads pytest's JUnit XML (`--junitxml`). The other browsers' job is allowed to fail until
its failures are read (v0.3.1, E2), so the summary shows them without opening the log.
Always exits 0: the pytest step's own result is the job's.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def summary(path: Path, engine: str) -> str:
    """The heading with the counts, then one line per failed or errored test."""
    title = f"### The browser tier in {engine}"
    if not path.is_file():
        return f"{title}: no report (the run stopped before it)\n"
    passed = failed = skipped = 0
    lines = []
    for case in ET.parse(path).getroot().iter("testcase"):
        problem = case.find("failure")
        problem = problem if problem is not None else case.find("error")
        if problem is not None:
            failed += 1
            name = f"{case.get('classname')}::{case.get('name')}"
            first = "".join((problem.get("message") or "").splitlines()[:1])
            lines.append(f"- {name}: {first[:200]}")
        elif case.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    head = f"{title}: {passed} passed, {failed} failed, {skipped} skipped"
    return "\n".join([head, *lines]) + "\n"


if __name__ == "__main__":
    print(summary(Path(sys.argv[1]), sys.argv[2]), end="")
