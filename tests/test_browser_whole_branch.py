"""v0.3.1's whole-branch review, the page: Cancel that names its upload (should 4), chips
and Start new chat pressed twice (should 5), the file picker in the press (should 6), a
read after a reload (could 13), a failed Compare redraw (could 14) and a choice made while a
range is counted (could 15). Made-up documents only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from tests.test_browser import make_three, open_replies, until
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


def state(page) -> dict:
    return page.evaluate("clarity.command('state')")["result"]


def test_cancel_before_the_read_starts_means_it_never_starts(clarity, page, monkeypatch):  # noqa: F811
    """Should 4: the read is held on its way to the server; Cancel, then let it go: the file
    is never read, and the sheet's sentence is the server's answer."""
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(3)")
    chat(page, redirect)
    held = []
    page.route("**/view/attach?*", lambda route: held.append(route))
    choose(page, "slow.md", NOTES)
    until(page, lambda: held, what="the read asked for")
    page.click("#attach-sheet [data-action=attach-cancel]")
    held[0].continue_()
    page.unroute("**/view/attach?*")
    page.wait_for_selector("#attach-sheet :text('Stopped reading slow.md.')")
    seen = []
    for _ in range(15):  # 1.5 s: a read that started would show for 3 s
        seen.append(state(page)["reading"])
        page.wait_for_timeout(100)
    assert seen == [None] * 15 and state(page)["running"] is None


def test_two_quick_presses_on_two_chips_never_remove_a_third(clarity, page):  # noqa: F811
    """Should 5: the second × waits for the first; it never removes the chip that moved
    into its place."""
    redirect, _ = clarity
    chat(page, redirect)
    for name in ("a.md", "b.md", "c.md"):
        choose(page, name, f"Made-up {name}.\n".encode())
        ready(page)
        page.click("[data-action=attach-add]")
        page.wait_for_selector(f"#chips :text('{name}')")
    page.evaluate(
        """() => {
          document.querySelector("[aria-label='Remove a.md']").click();
          document.querySelector("[aria-label='Remove b.md']").click();
        }"""
    )
    until(page, lambda: "a.md" not in page.text_content("#chips"))
    page.wait_for_timeout(500)  # nothing to wait for: a wrong removal shows now, if ever
    names = [a["name"] for a in state(page)["pending"]]
    assert "c.md" in names and "a.md" not in names
    assert "/attach" not in page.text_content("#notice")  # no terminal command on the page


def test_start_new_chat_pressed_twice_makes_one_chat(clarity, page):  # noqa: F811
    redirect, backend = clarity
    backend._n_ctx = 1500
    chat(page, redirect)
    page.fill("#message", "Made-up words in a long message. " * 28)
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    choose(page, "notes.md", LONG[:700])
    ready(page)
    page.click("[data-choice=new][role=radio]")
    chats = len(page.evaluate("clarity.command('chats')")["result"]["chats"])
    page.evaluate(
        """() => {
          const add = document.querySelector("[data-action=attach-add]");
          add.click();
          add.click();
        }"""
    )
    page.wait_for_selector("#chips .chip")
    page.wait_for_timeout(500)  # nothing to wait for: a second chat is made now, if ever
    assert len(page.evaluate("clarity.command('chats')")["result"]["chats"]) == chats + 1


def test_the_picker_opens_in_the_press_without_a_round_trip(clarity, page):  # noqa: F811
    """Should 6: in a chat-mode chat, the paperclip opens the picker at once, so a browser
    keeps the press's activation; nothing waits on the server first."""
    redirect, _ = clarity
    chat(page, redirect)
    held = []
    page.route(
        "**/command",
        lambda route: (
            held.append(route)
            if '"command":"attach"' in (route.request.post_data or "")
            else route.continue_()
        ),
    )
    with page.expect_file_chooser(timeout=5000):
        page.click("#clip")
    for route in held:
        route.continue_()


def test_a_reload_mid_read_shows_the_reading_line_with_cancel(clarity, page, monkeypatch):  # noqa: F811
    """Could 13: the page that reloads during a read says so, and its Cancel stops it."""
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(30)")
    chat(page, redirect)
    choose(page, "slow.md", NOTES)
    until(page, lambda: state(page)["reading"], what="the read started")
    page.reload()
    page.wait_for_selector("body[data-page=ready]")
    page.wait_for_selector("#reading-line:has-text('Reading slow.md…')")
    page.click("#reading-line [data-action=reading-cancel]")
    until(page, lambda: state(page)["running"] is None, what="the read stopped")


def test_a_failed_compare_redraw_after_a_mark_says_so(clarity, page):  # noqa: F811
    """Could 14."""
    redirect, backend = clarity
    chat(page, redirect)
    make_three(page, backend)
    open_replies(page)
    page.click(".listpop [data-action=compare]")
    page.wait_for_selector("#compare .cmp-cards")
    page.click("#compare .card [data-action=marks]")
    page.wait_for_selector("#compare .card .pop:not([hidden])")
    page.route("**/view/compare?*", lambda route: route.fulfill(status=500, body="Made-up."))
    page.click("#compare .card .pop [data-action=mark]")
    page.wait_for_selector("#compare .problem")  # in Compare: the notice line is hidden there


def test_a_choice_made_while_a_range_is_counted_stays(clarity, page):  # noqa: F811
    """Could 15: typing a range, then choosing Attach whole before it's counted."""
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "notes.md", NOTES)
    ready(page)
    page.fill("#attach-to", "2")
    page.click("[data-choice=whole][role=radio]")
    page.wait_for_timeout(1000)  # past the 400 ms read and its answer
    assert page.get_attribute("[data-choice=whole][role=radio]", "aria-checked") == "true"


def test_the_box_being_typed_in_is_the_same_box_after_its_count(clarity, page):  # noqa: F811
    """Page mutation 35 (the whole-branch review): an answer that lands while you type keeps
    the box you're in, so its caret and any key in flight stay."""
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    page.fill("#attach-to", "5")
    page.evaluate("document.getElementById('attach-to').kept = true")
    until(
        page, lambda: (page.get_attribute("[data-choice=part]", "data-args") or "").endswith(" 1-5")
    )
    assert page.evaluate("document.activeElement.id") == "attach-to"
    assert page.evaluate("document.getElementById('attach-to').kept") is True


def test_the_chips_row_hides_when_no_chip_waits(clarity, page):  # noqa: F811
    """Page mutation 37 (the whole-branch review): no empty row above the message box."""
    redirect, _ = clarity
    chat(page, redirect)
    hidden = "document.getElementById('chips').hidden"  # an empty row has no box to see
    assert page.evaluate(hidden) is True
    choose(page, "a.md", b"Made-up a.\n")
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    assert page.evaluate(hidden) is False
    page.click("[aria-label='Remove a.md']")
    until(page, lambda: page.evaluate(hidden) is True, what="the row hidden again")
