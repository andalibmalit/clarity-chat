"""v0.3.1's close, the page: the file picker once a read has ended, after a redraw while it
ran (Firefox CI, run 43755817667). Made-up documents only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from tests.test_browser import until
from tests.test_browser_attach import (  # noqa: F401 - fixtures, used by name
    NOTES,
    browser_type_launch_args,
    chat,
    choose,
    clarity,
)
from tests.test_documents_extract import stub

pytestmark = pytest.mark.browser


def test_a_redraw_during_a_read_leaves_the_picker_free_once_it_ends(clarity, page, monkeypatch):  # noqa: F811
    """A redraw while this page's read ran (a status event, say) drew "attach" running. Once
    the read ends, Choose another file and the paperclip open the picker at once, not after
    the next poll: pressed meanwhile, neither did anything, and the sheet didn't say why."""
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(1.5)")  # ends with no text: a refusal
    chat(page, redirect)
    choose(page, "slow.md", NOTES)
    state = "clarity.command('state')"
    until(page, lambda: page.evaluate(state)["result"]["reading"], what="the read started")
    page.evaluate("clarity.refresh()")  # drawn while the read runs
    page.wait_for_selector("#attach-sheet [role=alert]")
    with page.expect_file_chooser(timeout=5000):
        page.click("#attach-sheet [data-action=attach-pick]")
    page.click("[data-action=attach-close]")
    with page.expect_file_chooser(timeout=5000):
        page.click("#clip")
