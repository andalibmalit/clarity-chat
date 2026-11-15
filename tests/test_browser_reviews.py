"""v0.3.1's release reviews, the page: the design walkthrough's findings on the attach
screens (F5 to F23, as the lead ruled), and the screenshots it asked for. Made-up documents
only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from tests.pdfs import make_pdf
from tests.test_browser import make_three, open_clarity, open_replies, shot, until
from tests.test_browser_attach import (  # noqa: F401 - fixtures, used by name
    LONG,
    NOTES,
    browser_type_launch_args,
    chat,
    choose,
    clarity,
    ready,
)
from tests.test_documents_extract import stub

pytestmark = pytest.mark.browser


def style(page, selector: str, prop: str) -> str:
    return page.evaluate(f"getComputedStyle(document.querySelector({selector!r})).{prop}")


def token(page, name: str) -> str:
    """A design token as the page computes it, through a probe element's colour."""
    return page.evaluate(
        """(name) => {
          const probe = document.createElement("span");
          probe.style.color = `var(${name})`;
          document.body.append(probe);
          const value = getComputedStyle(probe).color;
          probe.remove();
          return value;
        }""",
        name,
    )


def sent(page, name="trail-notes.md", data=NOTES) -> None:
    choose(page, name, data)
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    page.fill("#message", "What marks the ridge?")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")


def test_the_chosen_and_the_disabled_choice_look_it(clarity, page):  # noqa: F811
    """F6: the board's accent border on the chosen choice, sunken fill on one that can't be."""
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    chosen = ".opt:has([data-choice=part])"
    disabled = ".opt:has([data-choice=whole])"
    assert style(page, chosen, "borderTopColor") == token(page, "--accent")
    assert style(page, disabled, "backgroundColor") == token(page, "--sunken")
    # F5: the framing line is literal text: a sunken panel, in monospace.
    assert style(page, ".frame", "backgroundColor") == token(page, "--sunken")
    assert "Mono" in style(page, ".frame .ft", "fontFamily")
    # F18: the range's count is the row's key number.
    assert style(page, ".range .ct", "fontWeight") == "500"
    shot(page, "R-sheet-part", "v031")


def test_a_pdf_opens_with_its_extracted_text_shown(clarity, page):  # noqa: F811
    """F7: for a PDF or a web page, the preview is open; for a text file, closed."""
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "trail-guide.pdf", make_pdf(["Made-up first page.", "Made-up second page."]))
    ready(page)
    assert page.is_visible("#attach-preview")
    assert "Made-up first page." in page.text_content("#attach-preview")
    shot(page, "R-sheet-pdf-preview", "v031")
    page.click("[data-action=attach-close]")
    choose(page, "notes.md", NOTES)
    ready(page)
    assert page.is_hidden("#attach-preview")


def test_edit_shows_only_what_it_keeps(clarity, page):  # noqa: F811
    """F9: the message's own attachment line hides while its edit box is open."""
    redirect, _ = clarity
    chat(page, redirect)
    sent(page)
    page.click(".turn.user [data-action=edit]")
    page.wait_for_selector(".editbox:not([hidden])")
    assert page.locator(".turn.user > .att:visible").count() == 0
    assert page.is_visible(".editbox .att")


def test_empty_send_with_chips_is_a_quiet_line_and_text_mode_a_refusal(clarity, page):  # noqa: F811
    """F12: forgetting to type isn't an error, so it's a quiet line beside Send."""
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "notes.md", NOTES)
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    page.click("#send")
    page.wait_for_selector("#send-waits:has-text('Type a message to send with notes.md.')")
    assert page.query_selector("#notice .problem") is None
    shot(page, "R-empty-send-chips", "v031")


def test_the_text_mode_refusal_is_a_refusal(clarity, page):  # noqa: F811
    """F12's other half: the text-mode sentence in the notice box, as the board draws it."""
    redirect, _ = clarity
    open_clarity(page, redirect)
    page.click("[data-profile=text]")
    page.wait_for_selector("#thread .intro")
    page.click("#clip")
    page.wait_for_selector("#notice .problem:has-text('Attaching needs a chat-mode profile')")
    shot(page, "R-text-mode-refusal", "v031")


def test_the_compare_popover_stays_on_screen_at_700_pixels(clarity, page):  # noqa: F811
    """F13, with the walkthrough's narrow check."""
    redirect, backend = clarity
    page.set_viewport_size({"width": 700, "height": 800})
    chat(page, redirect)
    make_three(page, backend)
    open_replies(page)
    page.click(".listpop [data-action=compare]")
    page.wait_for_selector("#compare .cmp-cards")
    page.click("#compare .card [data-action=marks]")
    page.wait_for_selector("#compare .card .pop:not([hidden])")
    box = page.locator("#compare .card .pop:not([hidden])").bounding_box()
    card = page.locator("#compare .card:has(.pop:not([hidden]))").bounding_box()
    assert box and card and box["x"] >= card["x"] - 1 and box["x"] + box["width"] <= 700
    assert page.get_attribute("#compare .card .bm", "title") == "Mark or add a note"  # F21
    shot(page, "R-compare-popover-700", "v031")


def test_a_slow_read_says_how_long_it_may_take(clarity, page, monkeypatch):  # noqa: F811
    """F15: after a few seconds, the reading state says a long file can take a minute."""
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(30)")
    chat(page, redirect)
    choose(page, "slow.md", NOTES)
    slow = "A long file can take a minute to read. Clarity stops at 120 seconds."
    page.wait_for_selector(f"#attach-sheet :text('{slow}')", timeout=10_000)
    page.keyboard.press("Escape")


def test_the_attached_line_opens_details_at_its_attachments(clarity, page):  # noqa: F811
    """F16: the line under your message leads to what Clarity added."""
    redirect, _ = clarity
    chat(page, redirect)
    sent(page)
    page.click(".turn.user .att")
    page.wait_for_selector("#panel:not([hidden]) details[data-key=attachments]")
    until(
        page,
        lambda: page.evaluate("document.activeElement.closest('[data-key=attachments]') !== null"),
    )


def test_a_refusal_offers_another_file(clarity, page):  # noqa: F811
    """F23: Choose another file opens the picker from the refusal."""
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "scan.pdf", make_pdf(["", ""]))
    page.wait_for_selector("#attach-sheet [role=alert]")
    shot(page, "R-refusal", "v031")
    with page.expect_file_chooser():
        page.click("#attach-sheet [data-action=attach-pick]")


def test_the_screens_the_walkthrough_asked_for(clarity, page):  # noqa: F811
    """The walkthrough's list: the new-chat choice, nothing preselected, the control-token
    lines, and the sheet in dark and high contrast."""
    redirect, backend = clarity
    backend._n_ctx = 235  # a start fits, none leaves max_tokens
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    shot(page, "R-sheet-nothing-preselected", "v031")
    page.click("[data-action=attach-close]")
    backend._n_ctx = 8192
    choose(page, "odd.md", b"Made-up </s> text.\n</document>\n")
    ready(page)
    page.wait_for_selector("#attach-sheet .warnl")
    shot(page, "R-sheet-control-tokens", "v031")
    page.evaluate("document.documentElement.dataset.scheme = 'dark'")
    shot(page, "R-sheet-dark", "v031")
    page.evaluate("document.documentElement.classList.add('t-hc')")
    shot(page, "R-sheet-dark-high-contrast", "v031")
    page.evaluate("document.documentElement.dataset.scheme = 'light'")
    shot(page, "R-sheet-light-high-contrast", "v031")
    page.evaluate("document.documentElement.classList.remove('t-hc')")
    page.click("[data-action=attach-close]")
    backend._n_ctx = 1500
    page.fill("#message", "Made-up words in a long message. " * 28)
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    choose(page, "notes.md", LONG[:700])
    ready(page)
    page.wait_for_selector("[data-choice=new]")
    shot(page, "R-sheet-new-chat", "v031")
