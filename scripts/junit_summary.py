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

DRIVER_GONE = "Playwright connection closed"  # every test after the driver died says it


def summary(path: Path, engine: str) -> str:
    """The heading with the counts, then one line per failed or errored test."""
    title = f"### The browser tier in {engine}"
    if not path.is_file():
        return f"{title}: no report (the run stopped before it)\n"
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except ET.ParseError:  # a run killed while it wrote the report
        return f"{title}: report unreadable (the run stopped writing it)\n"
    passed = failed = skipped = 0
    lines = []
    stopped_in, after = None, 0  # the test Playwright's driver died in, and the tests after
    for case in cases:
        problem = case.find("failure")
        problem = problem if problem is not None else case.find("error")
        if problem is not None:
            failed += 1
            name = f"{case.get('classname')}::{case.get('name')}"
            if stopped_in is not None:
                after += 1
                continue
            if any(DRIVER_GONE in (p.get("message") or "") for p in case if p.tag == "error"):
                stopped_in = name  # its own failure is listed; the rest follow from it
            first = "".join((problem.get("message") or "").splitlines()[:1])
            lines.append(f"- {name}: {first[:200]}")
        elif case.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    head = f"{title}: {passed} passed, {failed} failed, {skipped} skipped"
    if stopped_in is not None:
        lines.insert(
            0,
            f"Playwright's driver stopped during {stopped_in}; the {after} tests after it "
            "errored because of that, not because of the page.",
        )
    return "\n".join([head, *lines]) + "\n"


if __name__ == "__main__":
    print(summary(Path(sys.argv[1]), sys.argv[2]), end="")
