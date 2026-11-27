"""v0.4.7 B5: the Settings panel's Strength row on llama.cpp, in a real browser. A new
strength starts the model again and the next reply's facts line says so, "Off (0)" runs it
unsteered, a start that fails puts its refusal under the row with the old strength running,
Reset all with Strength and a sampler changed starts the model once, and during a reply the
row waits, so a change applies only after the reply.

The real llama.cpp backend on the fake llama-server (tests/fake_llama.py), as
tests/test_steering_chat.py runs it offline: CI's browser job has no model, so SmolLM2's
restart is `-m live`'s (tests/test_live.py). Each test counts the starts by the scale each
one passed to llama-server. Made-up text only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import threading

import pytest

from clarity_chat import profiles
from clarity_chat.backends import llama_cpp
from clarity_chat.session import Session
from tests.fake_llama import fake_llama, startup
from tests.test_browser import chat_on, say, served, settled, until
from tests.test_browser_attach import browser_type_launch_args  # noqa: F401
from tests.test_steering import steer_root, steered
from tests.test_steering_chat import backend_on, scales
from tests.test_steering_llama_cpp import model_like

pytestmark = pytest.mark.browser

ROW = "#panel [data-setting=steering_scale]"
FACTS = ".reply.last .facts"


@pytest.fixture
def steered_chat(monkeypatch, root, tiny_tokenizer):
    """Clarity served on the real llama.cpp backend and the fake server, whose tiny model
    is a made-up llama file of width 8 and 4 blocks: (the redirect file, what the fake
    saw). The profile `steered` steers with calm.gguf at ×4.0."""
    steer_root(root)
    model_like(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    steered(root, backend="llama.cpp")
    session = Session(root, backend=backend_on(root), tokenizer_loader=lambda r, v: tiny_tokenizer)
    profiles.set_hint_folder(root)
    try:
        for _, redirect in served(root, session):
            yield redirect, seen
    finally:
        profiles.set_hint_folder(None)


def open_settings(page) -> None:
    page.click("#composer-row [data-action=settings]")
    page.wait_for_selector("#panel:not([hidden]) [data-setting=steering_scale]")


def steering_fact(page) -> str:
    """The last reply's facts line says its steering first, then any other fact (the fake's
    replies stop at the token limit, which the line says too)."""
    return (page.text_content(f"{FACTS} p") or "").split(" · ")[0]


def reply(page, text: str = "Hi") -> str:
    """Send `text` and return what the new reply's facts line says of its steering."""
    count = page.locator(".reply").count()
    say(page, text)
    until(page, lambda: page.locator(".reply").count() > count, what="the new reply")
    page.wait_for_selector(FACTS)
    settled(page)
    return steering_fact(page)


def strength(page, value: str) -> None:
    """Type a strength into the row's box and press Enter, as a person would."""
    page.fill(f"{ROW} #set-steering_scale", value)
    page.press(f"{ROW} #set-steering_scale", "Enter")


def test_a_new_strength_starts_the_model_again_and_the_reply_says_it(steered_chat, page):
    redirect, seen = steered_chat
    chat_on(page, redirect, "steered")
    assert reply(page) == "steered ×4.0"
    open_settings(page)
    strength(page, "2.0")
    page.wait_for_selector(f"{ROW} .tag:has-text('chat')")
    settled(page)
    assert scales(seen) == [4.0, 2.0]
    assert page.text_content(f"{ROW} [data-set=steering_scale]") == "Back to 4.0 (profile)"
    page.keyboard.press("Escape")
    assert reply(page, "Again") == "steered ×2.0"
    assert scales(seen) == [4.0, 2.0]  # the reply started nothing


def test_off_starts_the_model_with_no_vector_and_the_reply_says_not_steered(steered_chat, page):
    redirect, seen = steered_chat
    chat_on(page, redirect, "steered")
    open_settings(page)
    page.click(f"{ROW} [data-set='steering_scale 0']")
    page.wait_for_selector(f"{ROW} .tag:has-text('chat')")
    settled(page)
    assert scales(seen) == [4.0, 0.0]
    assert page.locator(f"{ROW} [data-set='steering_scale 0']").count() == 0  # it's off
    page.keyboard.press("Escape")
    assert reply(page) == "not steered"


def test_a_strength_that_cant_start_is_refused_under_the_row(steered_chat, page):
    redirect, seen = steered_chat
    chat_on(page, redirect, "steered")
    open_settings(page)
    seen["cvec"] = "fails"  # the new start's log shows no vector (state 11) ...
    seen["during_start"] = lambda: seen.update(cvec="loads", during_start=None)  # ... once
    strength(page, "2.0")
    page.wait_for_selector(f"{ROW} .refused:not([hidden])")
    settled(page)
    assert page.is_visible(f"{ROW} .refused")  # still there after the redraws
    assert (page.text_content(f"{ROW} .refused") or "").startswith(
        "llama-server started without the vector: its start log doesn't show"
    )
    assert page.text_content(f"{ROW} .note") == (
        "Not applied: the next reply still uses 4.0 (profile)."
    )
    assert page.text_content(f"{ROW} .tag") == "profile"
    assert scales(seen) == [4.0, 2.0, 4.0], "the old strength wasn't started again"
    page.keyboard.press("Escape")
    assert reply(page) == "steered ×4.0"


def test_reset_all_with_strength_and_a_sampler_changed_starts_the_model_once(steered_chat, page):
    """Decision 10: Reset all is one /set per changed setting, Strength kept, so on
    llama.cpp the model starts again once, for the strength."""
    redirect, seen = steered_chat
    chat_on(page, redirect, "steered")
    open_settings(page)
    page.fill("#set-temperature", "0.3")
    page.press("#set-temperature", "Enter")
    page.wait_for_selector("#panel [data-setting=temperature] .tag:has-text('chat')")
    strength(page, "2.0")
    page.wait_for_selector(f"{ROW} .tag:has-text('chat')")
    settled(page)
    assert scales(seen) == [4.0, 2.0]
    page.click("#panel [data-action=reset-all]")
    page.wait_for_selector(f"{ROW} .tag:has-text('profile')")
    page.wait_for_selector("#panel [data-setting=temperature] .tag:has-text('profile')")
    settled(page)
    assert page.input_value("#set-temperature") == "0.6"
    assert page.input_value("#set-steering_scale") == "4.0"
    assert scales(seen) == [4.0, 2.0, 4.0], "Reset all didn't start the model exactly once"
    page.keyboard.press("Escape")
    assert reply(page) == "steered ×4.0"


def test_during_a_reply_strength_waits_and_a_change_applies_after_it(
    steered_chat, page, monkeypatch
):
    """The row waits while a reply is written, as every row does, so the reply runs at the
    strength it started with; the change made after it starts the model again."""
    redirect, seen = steered_chat
    chat_on(page, redirect, "steered")
    open_settings(page)
    gate = threading.Event()
    stream = llama_cpp.httpx.stream

    def held(*args, **kwargs):  # the reply waits here until the test lets it go
        gate.wait(30)
        return stream(*args, **kwargs)

    monkeypatch.setattr(llama_cpp.httpx, "stream", held)
    try:
        page.evaluate("() => { clarity.command('say', 'Hi'); }")
        page.wait_for_selector("#panel .waits")
        assert page.text_content("#panel .waits") == "Changes wait until this reply is done."
        assert page.get_attribute("#set-steering_scale", "aria-disabled") == "true"
        assert not page.is_editable("#set-steering_scale")
        # Playwright won't click an aria-disabled button, so the click is forced, as a
        # person's would land; the page sends nothing for a control that waits.
        page.click(f"{ROW} [data-set='steering_scale 0']", force=True)
        assert scales(seen) == [4.0], "the model started again during the reply"
    finally:
        gate.set()
    page.wait_for_selector("#panel .waits", state="detached", timeout=20_000)
    page.wait_for_selector(FACTS)
    settled(page)
    assert steering_fact(page) == "steered ×4.0"
    assert scales(seen) == [4.0]
    strength(page, "2.0")
    page.wait_for_selector(f"{ROW} .tag:has-text('chat')")
    settled(page)
    assert scales(seen) == [4.0, 2.0]
    page.keyboard.press("Escape")
    assert reply(page, "Again") == "steered ×2.0"
