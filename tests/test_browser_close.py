"""v0.3.1's close, the page: the file picker once a read has ended, after a redraw while it
ran (Firefox CI, run 43755817667), and the fix batch's page findings. Made-up documents only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from tests.pdfs import make_pdf
from tests.test_browser import until
from tests.test_browser_attach import (  # noqa: F401 - fixtures, used by name
    NOTES,
    browser_type_launch_args,
    chat,
    choose,
    clarity,
    ready,
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
    until(page, lambda: state(page)["reading"], what="the read started")
    page.evaluate("clarity.refresh()")  # drawn while the read runs
    page.wait_for_selector("#attach-sheet [role=alert]")
    with page.expect_file_chooser(timeout=5000):
        page.click("#attach-sheet [data-action=attach-pick]")
    page.click("[data-action=attach-close]")
    with page.expect_file_chooser(timeout=5000):
        page.click("#clip")


def state(page) -> dict:
    return page.evaluate("clarity.command('state')")["result"]


def chips(page) -> list[str]:
    return page.eval_on_selector_all("#chips .chip .nm", "names => names.map(n => n.textContent)")


def add(page, name: str, data: bytes = NOTES) -> None:
    count = len(chips(page))
    choose(page, name, data)
    ready(page)
    page.click("[data-action=attach-add]")
    until(page, lambda: len(chips(page)) == count + 1, what="the chip")


NOT_KEPT = (
    "Not kept: notes.md, waiting for your next message. An attachment is logged only with "
    "the message it's sent with."
)


def test_new_chat_says_what_it_didn_t_keep(clarity, page):  # noqa: F811
    """The close's must A: the terminal says "Not kept" when /new drops attachments; the
    page now does too."""
    redirect, _ = clarity
    chat(page, redirect)
    add(page, "notes.md")
    page.click("#clarity-button")
    page.click("#new-chat")
    page.wait_for_selector(f"#notice [role=status]:has-text('{NOT_KEPT[:40]}')", timeout=5000)
    assert page.text_content("#notice").strip() == NOT_KEPT
    assert chips(page) == []


def test_quit_says_what_it_didn_t_keep(clarity, page):  # noqa: F811
    redirect, _ = clarity
    chat(page, redirect)
    add(page, "notes.md")
    page.click("#clarity-button")
    page.click("[data-action=quit]")
    page.wait_for_selector("body[data-page=stopped]")
    assert page.is_visible("#stopped-kept")
    assert page.text_content("#stopped-kept") == NOT_KEPT


def test_a_refusal_in_the_sheet_leaves_the_focus_on_its_title(clarity, page):  # noqa: F811
    """The final fix review's should 1: every refusal drawn in the sheet threw, so the
    focus was lost in a modal sheet."""
    redirect, _ = clarity
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    chat(page, redirect)
    choose(page, "scan.pdf", make_pdf(["", ""]))
    page.wait_for_selector("#attach-sheet [role=alert]")
    until(page, lambda: page.evaluate("document.activeElement.id") == "attach-title", what="focus")
    assert errors == []


def test_x_removes_its_own_chip_when_two_share_bytes(clarity, page):  # noqa: F811
    """Fix review 7, surface should 2: × on a copy removed the first chip with its bytes."""
    redirect, _ = clarity
    chat(page, redirect)
    add(page, "a.md")
    add(page, "copy-of-a.md")
    page.click("#chips [aria-label='Remove copy-of-a.md']")
    until(page, lambda: len(chips(page)) == 1, what="one chip removed")
    assert chips(page) == ["a.md"]


def test_a_cancel_that_lands_after_the_read_still_stops_it(clarity, page):  # noqa: F811
    """Fix review could 1, surface should 3: the read's answer was held until Cancel was
    pressed; the sheet says the read stopped, and draws no choices to attach."""
    redirect, _ = clarity
    chat(page, redirect)
    held = []
    page.route("**/view/attach?*", lambda route: held.append((route, route.fetch())))
    choose(page, "notes.md", NOTES)
    until(page, lambda: held, what="the read answered")
    page.click("#attach-sheet [data-action=attach-cancel]")
    until(
        page,
        lambda: (
            page.evaluate(
                "document.querySelector('[data-action=attach-cancel]')?.getAttribute('aria-disabled')"
            )
            == "true"
        ),
        what="Cancel pressed",
    )
    route, response = held[0]
    route.fulfill(response=response)
    page.unroute("**/view/attach?*")
    page.wait_for_selector("#attach-sheet :text('Stopped reading notes.md.')", timeout=5000)
    assert page.query_selector("#attach-sheet [role=radiogroup]") is None


def test_a_dollar_in_a_name_is_shown_as_typed(clarity, page):  # noqa: F811
    """Surface could 4: "Stopped reading NAME." replaced `$'` and `$&` as patterns."""
    redirect, _ = clarity
    chat(page, redirect)
    held = []
    page.route("**/upload?*", lambda route: held.append(route))
    choose(page, "a$'b.md", NOTES)
    until(page, lambda: held, what="the upload sent")
    page.click("#attach-sheet [data-action=attach-cancel]")
    page.unroute("**/upload?*")
    page.wait_for_selector("#attach-sheet [role=status]")
    assert page.text_content("#attach-sheet [role=status]").strip() == "Stopped reading a$'b.md."


def test_each_press_opens_the_picker_on_a_fresh_input(clarity, page):  # noqa: F811
    """Firefox CI (run 43766644268): a second picker on the same input sometimes never
    opened. Each press gets its own input, still `#attach-file` (the QA driver's upload
    step), and the file chosen in it is read."""
    redirect, _ = clarity
    chat(page, redirect)
    with page.expect_file_chooser() as first:
        page.click("#clip")
    with page.expect_file_chooser() as second:
        page.click("#clip")
    assert first.value.element.evaluate("input => input.isConnected") is False
    assert second.value.element.evaluate("input => input.id") == "attach-file"
    second.value.set_files({"name": "notes.md", "mimeType": "text/markdown", "buffer": NOTES})
    ready(page)


def test_the_paperclip_says_why_it_waits_during_a_reply(clarity, page):  # noqa: F811
    """QA 2 (task 17): greyed while a reply streams, with nothing saying why."""
    redirect, backend = clarity
    chat(page, redirect)
    backend.replies, backend.piece_delay = [" Slowly" * 200], 0.2
    page.fill("#message", "Hi")
    page.click("#send")
    page.wait_for_selector("#writing-box:not([hidden])")
    assert page.get_attribute("#clip", "aria-disabled") == "true"
    assert page.is_visible("#clip-waits")
    assert page.text_content("#clip-waits") == "Waits until the reply is done."
    page.click("#stop")
