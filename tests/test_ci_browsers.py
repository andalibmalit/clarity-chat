"""E2 (v0.3.1): the other browsers' CI job, offline. Its pass and fail counts go to the job
summary, so a reader sees the failures without opening the log; and the getting-started
line says which browser the page has been checked in."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_browser" name="test_a[webkit]" time="1"/>
<testcase classname="tests.test_browser" name="test_b[webkit]" time="1">
<failure message="AssertionError: made up">made up</failure></testcase>
<testcase classname="tests.test_browser_attach" name="test_c[webkit]" time="1">
<error message="TimeoutError">made up</error></testcase>
<testcase classname="tests.test_browser" name="test_d[webkit]" time="0">
<skipped message="made up"/></testcase>
</testsuite></testsuites>"""


def summary_module():
    path = REPO / "scripts" / "junit_summary.py"
    assert path.is_file(), "scripts/junit_summary.py"
    spec = importlib.util.spec_from_file_location("junit_summary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_summary_counts_and_names_each_failure(tmp_path):
    xml = tmp_path / "browser.xml"
    xml.write_text(JUNIT, encoding="utf-8")
    text = summary_module().summary(xml, "WebKit")
    assert text.splitlines()[0] == "### The browser tier in WebKit: 1 passed, 2 failed, 1 skipped"
    assert "- tests.test_browser::test_b[webkit]: AssertionError: made up" in text
    assert "- tests.test_browser_attach::test_c[webkit]: TimeoutError" in text


def test_a_missing_report_is_said_in_the_summary(tmp_path):
    text = summary_module().summary(tmp_path / "browser.xml", "Firefox")
    assert text == "### The browser tier in Firefox: no report (the run stopped before it)\n"


def test_the_job_runs_webkit_and_firefox_allowed_to_fail_with_its_summary():
    workflow = (REPO / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "\n  browser-others:\n" in workflow
    job = workflow.split("\n  browser-others:\n", 1)[1].split("\n  live-cpu:", 1)[0]
    assert "continue-on-error: true" in job and "engine: [webkit, firefox]" in job
    assert "uv run pytest -m browser --browser ${{ matrix.engine }}" in job
    assert "scripts/junit_summary.py" in job and '>> "$GITHUB_STEP_SUMMARY"' in job


def test_getting_started_says_the_page_has_been_checked_in_chrome_only():
    text = (REPO / "docs/getting-started.md").read_text(encoding="utf-8")
    assert "The page has been checked in Chrome only." in text
