"""v0.4.4 B3b: the Settings panel's Context size row in a real browser: a preset starts the
model again (the start state, then the memory sentence), a refusal under the row with the
old preset still chosen, a size that isn't a preset, Details' own size and Replay at another
size, the context-full way out, and a restart whose old size fails too (state 31). The rows'
drawing is tested offline in tests/test_web_settings_context.py. Made-up text only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import time

import pytest

from clarity_chat.profiles import ConfigError
from tests.test_browser import chat_on, clarity, say, settled, shot  # noqa: F401
from tests.test_browser_attach import browser_type_launch_args  # noqa: F401
from tests.test_web_settings_context import MEMORY

pytestmark = pytest.mark.browser

ROW = "#panel [data-setting=ctx_size]"


def open_settings(page) -> None:
    page.click("#composer-row [data-action=settings]")
    page.wait_for_selector("#panel:not([hidden]) [data-setting=ctx_size]")


def pressed(page) -> list[str]:
    return page.locator(f"{ROW} .choice[aria-pressed=true]").all_text_contents()


def starts(backend, seconds: float = 1.5, refuse: int | None = None, memory: bool = True):
    """Each start takes `seconds`, measures MEMORY on Metal, and refuses the size `refuse`
    as llama.cpp's memory check would, the old size running on. FakeBackend has no trained
    maximum, so 16,384 starts here (on SmolLM2, trained on 8,192, it's refused)."""
    started = backend.start

    def start(model, weights, ctx_size):
        time.sleep(seconds)
        if ctx_size == refuse:
            raise ConfigError(f"At ctx_size {ctx_size}, the model needs more memory (made up).")
        record = started(model, weights, ctx_size)
        if record is not None and memory:
            record["memory"] = [MEMORY]
        return record

    backend.start = start


def test_a_preset_starts_the_model_again_then_says_its_memory(clarity, page):  # noqa: F811
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    settled(page)
    open_settings(page)
    assert pressed(page) == ["8,192"]
    shot(page, "B3b-context-row", "v044-b3b")
    starts(backend)
    page.click(f"{ROW} [data-set='ctx_size 16384']")
    # The start state (decision 6): the thread keeps the chat, Send and the panel wait.
    page.wait_for_selector("#thread .starting:has-text('Starting text…')")
    page.wait_for_selector("#settings-waits:has-text('Waits until the model has started.')")
    assert page.text_content("#send-waits") == "Waits until the model has started."
    assert page.get_attribute(f"{ROW} [data-set='ctx_size 4096']", "aria-disabled") == "true"
    assert page.is_visible("#title")  # the same chat: its title stays
    shot(page, "B3b-starting", "v044-b3b")
    page.wait_for_selector(f"{ROW} .choice[aria-pressed=true]:has-text('16,384')")
    settled(page)
    assert pressed(page) == ["16,384"]
    assert page.text_content(f"{ROW} .tag") == "chat"
    assert page.is_visible(f"{ROW} p.gl:has-text('At 16,384 tokens, the model uses 6,012 MiB')")
    assert page.text_content(f"{ROW} .btn:not(.choice)") == "Back to 8,192 (server.toml)"
    assert page.locator("#panel [data-action=reset-all]").count() == 0  # ruling 9
    shot(page, "B3b-memory", "v044-b3b")
    page.click(f"{ROW} .btn:not(.choice)")
    page.wait_for_selector(f"{ROW} .tag:has-text('server.toml')")
    assert pressed(page) == ["8,192"]
    assert backend.ctx_sizes == [8192, 16384, 8192]


def test_a_refused_size_leaves_the_old_preset_chosen(clarity, page):  # noqa: F811
    _, redirect, backend = clarity
    chat_on(page, redirect)
    open_settings(page)
    starts(backend, seconds=0.5, refuse=32768)
    page.click(f"{ROW} [data-set='ctx_size 32768']")
    page.wait_for_selector(f"{ROW} .refused:not([hidden])")
    settled(page)
    assert page.is_visible(f"{ROW} .refused")  # still there after the redraws
    assert page.text_content(f"{ROW} .refused") == (
        "At ctx_size 32768, the model needs more memory (made up)."
    )
    assert page.text_content(f"{ROW} .note") == (
        "Not applied: the next reply still uses 8,192 (server.toml)."
    )
    assert pressed(page) == ["8,192"]
    assert page.text_content(f"{ROW} .tag") == "server.toml"
    shot(page, "B3b-refused", "v044-b3b")


def test_a_size_that_isn_t_a_preset_is_listed_with_its_origin(clarity, page):  # noqa: F811
    _, redirect, _ = clarity
    chat_on(page, redirect)
    page.evaluate("clarity.command('set', 'ctx_size 12288')")
    open_settings(page)
    assert page.locator(f"{ROW} .choice").all_text_contents() == [
        "4,096",
        "8,192",
        "12,288",
        "16,384",
        "32,768",
    ]
    assert pressed(page) == ["12,288"]
    assert page.text_content(f"{ROW} .tag") == "chat"


def test_details_and_replay_keep_a_call_s_own_size(clarity, page):  # noqa: F811
    """State 28: Details shows the size the call ran at; state 27: Replay at another size
    says to set it back."""
    _, redirect, backend = clarity
    backend.replies = [" One.", " Two."]
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    settled(page)
    call = page.evaluate("clarity.command('prompt')")["result"]["call"]["call_id"]
    page.evaluate("clarity.command('set', 'ctx_size 16384')")
    say(page, "Again")
    page.wait_for_selector(".reply.last:has-text('Two.') .tb")
    settled(page)
    first = f"[data-call='{call}']"
    page.hover(first)  # an earlier reply's Details shows on hover
    page.locator(f"{first} [data-action=details]:visible").first.click()
    page.wait_for_selector("#panel:not([hidden]) .sec")
    page.click("#panel details[data-key=settings] summary")  # Settings as logged, collapsed
    size = "#panel details[data-key=settings] dt:text-is('context') + dd"
    assert page.is_visible(size)
    assert page.text_content(size) == "8,192"
    page.click("#panel [data-action=replay]")
    page.wait_for_selector("#panel .refused:not([hidden])")
    assert page.text_content("#panel .refused") == (
        f"Call {call} ran with a context of 8,192 tokens, but this chat uses 16,384. Set "
        "ctx_size to 8192 to replay it here."
    )
    shot(page, "B3b-replay-other-size", "v044-b3b")


def test_a_message_that_doesn_t_fit_says_to_raise_the_size(clarity, page):  # noqa: F811
    """State 30: the clause shows whenever no cap is known."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    page.evaluate("clarity.command('set', 'ctx_size 16')")
    say(page, "A made-up message that is far too long for this context.")
    page.wait_for_selector("#thread [data-failed]")
    assert page.text_content("#thread [data-failed] [role=alert]").endswith(
        "Shorten your message, start a new chat, lower max_tokens, or raise ctx_size."
    )


def test_a_restart_that_fails_twice_ends_the_chat_on_the_first_screen(clarity, page):  # noqa: F811
    """State 31 (ruling 10): the refusal above the box on the first screen, Settings closed,
    the profiles open."""
    _, redirect, backend = clarity
    chat_on(page, redirect)
    open_settings(page)

    def fails(model, weights, ctx_size):
        raise ConfigError(f"ctx_size {ctx_size} failed (made up).")

    backend.start = fails
    page.click(f"{ROW} [data-set='ctx_size 16384']")
    page.wait_for_selector("#notice [role=alert]:has-text('failed too')")
    assert page.text_content("#notice [role=alert]").startswith(
        "! Failed: ctx_size 16384 failed (made up)."
    )
    assert page.is_hidden("#panel")
    page.wait_for_selector("#picker-menu:not([hidden])")
    shot(page, "B3b-state-31", "v044-b3b")


def test_a_view_opened_in_the_panel_starts_at_its_top(clarity, page):  # noqa: F811
    """v0.4.4 QA (Q-b): Details opened scrolled, "What the model got" under the panel's
    header, as the panel kept Settings' scroll. Another view, or Details of another reply,
    starts at its top."""
    _, redirect, _ = clarity
    page.set_viewport_size({"width": 1280, "height": 480})  # both views scroll
    chat_on(page, redirect)
    for n, text in enumerate(("Hi", "Again"), 1):
        say(page, text)
        # .reply.last .tb alone is already there from the first reply: wait for this one's row
        page.wait_for_function(
            f"document.querySelectorAll('#thread .reply[data-call]').length === {n}"
        )
        page.wait_for_selector(".reply.last .tb")
        settled(page)
    first, last = page.locator("#thread .reply[data-call]").evaluate_all(
        "rows => rows.map((row) => row.dataset.call)"
    )
    open_settings(page)
    page.evaluate("document.getElementById('panel').scrollTop = 10000")
    assert page.evaluate("document.getElementById('panel').scrollTop") > 0
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector(f"#panel .body[data-call='{last}']")
    assert page.evaluate("document.getElementById('panel').scrollTop") == 0
    page.evaluate("document.getElementById('panel').scrollTop = 10000")
    assert page.evaluate("document.getElementById('panel').scrollTop") > 0
    page.click(f"#thread .reply[data-call='{first}'] [data-action=details]")
    page.wait_for_selector(f"#panel .body[data-call='{first}']")
    assert page.evaluate("document.getElementById('panel').scrollTop") == 0
    shot(page, "Q-b-details-at-top", "v044-close")
