"""v0.4.4 B2: the Settings panel's Documents group in a real browser: a budget changed,
refused and reset; the method's buttons, the minimum row going and coming back with it, and
O1's words under the row when meaning is chosen without bge-m3; Reset all leaving the size.
The rows' drawing (origins, help, bases after a resume, waiting) is tested offline in
tests/test_web_settings_documents.py. FakeEmbedder stands in for bge-m3. Made-up text only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from clarity_chat.documents import library_add, reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_browser import served, settled, until
from tests.test_browser_attach import browser_type_launch_args  # noqa: F401
from tests.test_browser_meaning import NO_EMBEDDER, start
from tests.test_meaning import GLOSSARY, MEANING, profile

pytestmark = pytest.mark.browser


@pytest.fixture
def clarity(fetched_root, chat_tokenizer):
    """The page with a profile, `meaning`, that searches `notes` by meaning: (redirect, the
    embedder)."""
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    backend = FakeBackend([" A made-up reply."] * 9, tokenizer=chat_tokenizer)
    fake = FakeEmbedder()
    session = Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    for _, redirect in served(fetched_root, session):
        yield redirect, fake


def open_documents(page, redirect) -> None:
    start(page, redirect, "meaning")
    page.click("#composer-row [data-action=settings]")
    page.wait_for_selector("#panel:not([hidden]) [data-setting=budget]")


def documents(page) -> dict:
    return page.evaluate("clarity.command('settings')")["result"]["documents"]


def test_a_budget_refused_changed_and_back(clarity, page):
    redirect, _ = clarity
    open_documents(page, redirect)
    # A bad budget: the sentence under its row, and the box keeps what you typed.
    page.fill("#set-budget", "0")
    page.press("#set-budget", "Enter")
    page.wait_for_selector("[data-setting=budget] .refused:not([hidden])")
    assert page.text_content("[data-setting=budget] .refused") == (
        "budget must be a positive whole number of tokens, got 0."
    )
    assert page.text_content("[data-setting=budget] .note") == (
        "Not applied: the next reply still uses 1,500 (default)."
    )
    assert page.input_value("#set-budget") == "0"
    assert page.get_attribute("#set-budget", "aria-invalid") == "true"  # mutation, review
    page.fill("#set-budget", "800")
    page.press("#set-budget", "Enter")
    page.wait_for_selector("[data-setting=budget] .tag:has-text('chat')")
    assert documents(page)["budget"]["value"] == 800
    assert page.text_content("[data-setting=budget] [data-set]") == "Back to 1,500 (default)"
    page.wait_for_selector("#composer-row .ch:has-text('1 changed')")
    page.click("[data-setting=budget] [data-set]")
    page.wait_for_selector("[data-setting=budget] .tag:has-text('default')")
    assert documents(page)["budget"] == {
        "value": 1500,
        "origin": "default",
        "base": {"value": 1500, "origin": "default"},
    }


def test_the_method_s_buttons_and_the_minimum_row(clarity, page):
    redirect, fake = clarity
    open_documents(page, redirect)
    assert page.get_attribute("[data-set='search_method meaning']", "aria-pressed") == "true"
    page.click("[data-set='search_method words']")
    page.wait_for_selector("[data-setting=min_similarity]", state="detached")
    assert page.get_attribute("[data-set='search_method words']", "aria-pressed") == "true"
    assert page.text_content("[data-setting=search_method] [data-set=search_method]") == (
        "Back to meaning (profile)"
    )
    # Meaning again, now without bge-m3: O1's words under the row (state 12).
    fake.is_fetched = False
    page.click("[data-set='search_method meaning']")
    page.wait_for_selector("[data-setting=min_similarity]")
    page.wait_for_selector("[data-setting=search_method] .note:not([hidden])")
    assert page.text_content("[data-setting=search_method] .note") == NO_EMBEDDER
    assert documents(page)["search_method"]["value"] == "meaning"


def test_reset_all_leaves_the_size(clarity, page):
    redirect, _ = clarity
    open_documents(page, redirect)
    assert "result" in page.evaluate("clarity.command('set', 'ctx_size 16384')")
    settled(page)
    page.fill("#set-budget", "800")
    page.press("#set-budget", "Enter")
    page.wait_for_selector("#panel [data-action=reset-all]")
    page.click("#panel [data-action=reset-all]")
    page.wait_for_selector("[data-setting=budget] .tag:has-text('default')")
    settings = page.evaluate("clarity.command('settings')")["result"]
    assert settings["documents"]["budget"]["origin"] == "default"
    assert settings["ctx_size"]["value"] == 16384


# --- The whole-branch review: should 1 and the page mutations that survived


def refuse_budget(page) -> None:
    page.fill("#set-budget", "0")
    page.press("#set-budget", "Enter")
    page.wait_for_selector("[data-setting=budget] .refused:not([hidden])")


def meaning_without_bge_m3(page, fake) -> None:
    """O1's words under the method row (state 12)."""
    page.click("[data-set='search_method words']")
    page.wait_for_selector("[data-setting=min_similarity]", state="detached")
    fake.is_fetched = False
    page.click("[data-set='search_method meaning']")
    page.wait_for_selector("[data-setting=search_method] .note:not([hidden])")


def test_closing_settings_forgets_a_refusal_and_the_missing_embedder(clarity, page):
    redirect, fake = clarity
    open_documents(page, redirect)
    meaning_without_bge_m3(page, fake)
    refuse_budget(page)
    page.click("#panel [data-action=panel-close]")
    page.click("#composer-row [data-action=settings]")
    page.wait_for_selector("#panel:not([hidden]) [data-setting=budget]")
    settled(page)
    assert page.is_hidden("[data-setting=budget] .refused")
    assert page.is_hidden("[data-setting=search_method] .note")


def test_a_new_chat_draws_settings_without_the_last_chat_s_refusal(clarity, page):
    """Should 1: a refusal and O1's words belong to the chat they came from; Settings stays
    open across a new chat and is drawn again for it."""
    redirect, fake = clarity
    open_documents(page, redirect)
    meaning_without_bge_m3(page, fake)
    refuse_budget(page)
    fake.is_fetched = True  # the new chat starts as it would; only the old words could show
    before = page.evaluate("clarity.state.conversation_id")
    page.click("#clarity-button")  # New chat is in the Clarity menu
    page.click("#new-chat")
    until(page, lambda: page.evaluate("clarity.state.conversation_id") not in (None, before),
          what="the new chat")  # fmt: skip
    settled(page)
    page.wait_for_selector("[data-setting=budget] .refused", state="hidden", timeout=5_000)
    page.wait_for_selector("[data-setting=search_method] .note", state="hidden", timeout=5_000)


def test_reset_all_forgets_the_missing_embedder(clarity, page):
    """Mutation: Reset all starts with no O1 words; only its own /set can bring them back."""
    redirect, fake = clarity
    open_documents(page, redirect)
    meaning_without_bge_m3(page, fake)
    fake.is_fetched = True
    page.click("#panel [data-action=reset-all]")
    page.wait_for_selector("[data-setting=search_method] .tag:has-text('profile')")
    settled(page)
    page.wait_for_selector("[data-setting=search_method] .note", state="hidden", timeout=5_000)
