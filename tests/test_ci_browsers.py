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
    assert "continue-on-error: true" in job and "runs-on: ${{ matrix.os }}" in job
    # WebKit on macOS (it closed at the first test on ubuntu-latest), Firefox on Linux.
    assert "- engine: webkit\n            os: macos-latest\n" in job
    assert "- engine: firefox\n            os: ubuntu-latest\n" in job
    assert "uv run pytest -m browser --browser ${{ matrix.engine }}" in job
    assert "scripts/junit_summary.py" in job and '>> "$GITHUB_STEP_SUMMARY"' in job


def test_getting_started_says_which_browsers_test_the_page():
    """The final fact-check's getting-started row, in the main session's words: both other
    engines' runs pass, but don't block a change."""
    text = (REPO / "docs/getting-started.md").read_text(encoding="utf-8")
    assert "The page has been checked in Chrome only." not in text
    assert (
        "The page is tested in Chrome on every change. Automated tests also run it in Firefox "
        "and in WebKit, the engine Safari uses; those runs don't block a change yet, so a "
        "problem there can slip through." in text
    )


# --- the whole-branch review's should 8 and 9


def test_the_other_browsers_job_has_a_time_limit():
    workflow = (REPO / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  browser-others:\n", 1)[1].split("\n  live-cpu:", 1)[0]
    assert "timeout-minutes: 30" in job


def test_an_unreadable_report_is_said_in_the_summary(tmp_path):
    xml = tmp_path / "browser.xml"
    xml.write_text(JUNIT[:200], encoding="utf-8")  # a run killed while writing it
    text = summary_module().summary(xml, "WebKit")
    assert (
        text == "### The browser tier in WebKit: report unreadable (the run stopped writing it)\n"
    )


def test_chrome_s_channel_is_only_for_chromium_and_clipboard_grants_are_guarded():
    for name in ("test_browser.py", "test_browser_attach.py"):
        source = (REPO / "tests" / name).read_text(encoding="utf-8")
        assert 'browser_name == "chromium"' in source, name
    browser = (REPO / "tests/test_browser.py").read_text(encoding="utf-8")
    grants = browser.count("grant_permissions(")
    assert grants == browser.count(
        'if browser_name == "chromium":\n        context.grant_permissions('
    )


def test_the_guide_says_where_the_whole_extracted_text_is_and_the_readme_attaching():
    guides = (REPO / "docs/guides.md").read_text(encoding="utf-8")
    assert "**Show extracted text** shows the start and end of what will go in" in guides
    assert "the whole is in **Details** after you send" in guides
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Attach a document to a message" in readme


# --- WebKit's driver stopped at the first test on ubuntu-latest (CI 43754075636)

DRIVER_STOPPED = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_browser" name="test_first[webkit]" time="1">
<failure message="playwright._impl._errors.TargetClosedError: Page.wait_for_selector: Target
page, context or browser has been closed">made up</failure>
<error message="playwright._impl._errors.TargetClosedError: Playwright connection closed">x
</error></testcase>
<testcase classname="tests.test_browser" name="test_second[webkit]" time="0">
<error message="playwright._impl._errors.TargetClosedError: Playwright connection closed">x
</error></testcase>
<testcase classname="tests.test_browser_attach" name="test_third[webkit]" time="0">
<error message="playwright._impl._errors.TargetClosedError: Playwright connection closed">x
</error></testcase>
</testsuite></testsuites>"""


def test_a_driver_that_stopped_is_one_finding_not_one_per_test(tmp_path):
    """When Playwright's driver dies, every later test errors with "Playwright connection
    closed": the summary names the test it died in and counts the rest as its consequence."""
    xml = tmp_path / "browser.xml"
    xml.write_text(DRIVER_STOPPED, encoding="utf-8")
    text = summary_module().summary(xml, "WebKit")
    assert text.splitlines()[0] == "### The browser tier in WebKit: 0 passed, 3 failed, 0 skipped"
    assert (
        "Playwright's driver stopped during tests.test_browser::test_first[webkit]; the 2 "
        "tests after it errored because of that, not because of the page." in text
    )
    assert "test_second" not in text and "test_third" not in text


def test_the_webkit_summary_says_the_redirect_file_isn_t_tested_there(tmp_path):
    """Playwright's WebKit driver stops on the redirect file's file:// to http:// hop (CI
    43754075636 on Linux, 43756769126 on macOS), so in WebKit the tests open the page by its
    link: the summary says what that leaves out. Other engines' summaries don't."""
    xml = tmp_path / "browser.xml"
    xml.write_text(JUNIT, encoding="utf-8")
    line = "WebKit opens the page by its link; the redirect file isn't tested in WebKit."
    assert line in summary_module().summary(xml, "webkit")
    assert line not in summary_module().summary(xml, "firefox")
