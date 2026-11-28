"""v0.4.4 B3b: the Settings panel's Context size row (states 20 to 28 on the page): the four
presets as buttons, the size in use pressed, a size that isn't a preset listed with its
origin, the memory sentence under the row, Back to the base, Reset all leaving the size, the
rows waiting, and the start state while the model starts again. Made-up text only. The
page's own clicks are in tests/test_browser_settings_context.py."""

from __future__ import annotations

import re
import threading
import time

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.profiles import SETTING_HELP
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import ORIGIN, PORT, SECRET, command, wait_until, writing
from tests.test_web_settings import panel, result, row
from tests.test_web_shell import get_view

# What llama.cpp measured at the start, as metal_memory reads it (state 20's figures).
MEMORY = {
    "device": "MTL0",
    "total_mib": 6012,
    "model_mib": 4610,
    "kv_mib": 1152,
    "compute_mib": 250,
    "limit_mib": 25559,
}


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    """The page on a text chat at server.toml's 8,192."""
    backend = FakeBackend([" One two three."] * 9, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client, session


def context(fragment: str) -> str:
    """The Context group: from its heading to More settings."""
    found = re.search(r'<h3 class="sh">Context</h3>(.*?)<details class="more"', fragment, re.S)
    assert found, "no Context group"
    return found.group(1)


def presets(fragment: str) -> list[tuple[str, str, str]]:
    """The size buttons: (what it sends, aria-pressed, its label)."""
    return re.findall(
        r'<button type="button" class="btn choice" data-set="(ctx_size \d+)" '
        r'aria-pressed="(true|false)"[^>]*>([\d,]+)</button>',
        row(context(fragment), "ctx_size"),
    )


def test_the_context_row_has_the_four_presets_and_the_size_in_use_pressed(chat):
    """Decision 5: the row after Common (text mode has no Documents group), its origin and
    help, which no longer says the model starts again (ruling 11)."""
    client, _ = chat
    fragment = panel(client)
    assert presets(fragment) == [
        ("ctx_size 4096", "false", "4,096"),
        ("ctx_size 8192", "true", "8,192"),
        ("ctx_size 16384", "false", "16,384"),
        ("ctx_size 32768", "false", "32,768"),
    ]
    assert fragment.index('<h3 class="sh">Common</h3>') < fragment.index(
        '<h3 class="sh">Context</h3>'
    )
    r = row(context(fragment), "ctx_size")
    assert '<span class="tag">server.toml</span>' in r
    assert SETTING_HELP["ctx_size"] == (
        "The context size in tokens: the prompt and the reply together must fit in it. A larger "
        "context usually takes more memory."
    )
    assert f'<p class="gl">{SETTING_HELP["ctx_size"]}</p>' in r
    assert 'data-set="ctx_size"' not in r, "a size that wasn't changed has no way back"


def test_a_size_set_here_is_pressed_with_its_way_back_and_its_memory(chat):
    """States 20 and 26: the chat's size, the memory measured at its start under the
    presets, and Back to the base with its origin."""
    client, session = chat
    result(client, "set", "ctx_size 16384")
    session.backend.record["memory"] = [MEMORY]
    fragment = panel(client)
    assert [p for _, p, label in presets(fragment) if p == "true"] == ["true"]
    assert ("ctx_size 16384", "true", "16,384") in presets(fragment)
    r = row(context(fragment), "ctx_size")
    assert '<span class="tag">chat</span>' in r
    assert (
        '<p class="gl">At 16,384 tokens, the model uses 6,012 MiB of GPU memory (4,610 for the '
        "model, 1,152 for the context cache, 250 for computing), out of Metal's recommended "
        "maximum for this GPU, 25,559 MiB.</p>"
    ) in r
    assert '<button type="button" class="btn" data-set="ctx_size"' in r
    assert ">Back to 8,192 (server.toml)</button>" in r


def test_a_size_that_isn_t_a_preset_is_listed_in_its_place_with_its_origin(chat):
    """Decision 5: drawn in size order, pressed, its origin on the row's tag."""
    client, _ = chat
    result(client, "set", "ctx_size 12288")
    fragment = panel(client)
    assert [label for _, _, label in presets(fragment)] == [
        "4,096",
        "8,192",
        "12,288",
        "16,384",
        "32,768",
    ]
    assert ("ctx_size 12288", "true", "12,288") in presets(fragment)
    assert '<span class="tag">chat</span>' in row(context(fragment), "ctx_size")


def test_without_metal_s_figure_there_is_no_memory_sentence(chat):
    """State 21: FakeBackend's start logs no memory, as the Transformers backend's."""
    client, _ = chat
    result(client, "set", "ctx_size 16384")
    assert "the model uses" not in row(context(panel(client)), "ctx_size")


def test_reset_all_is_hidden_when_the_size_is_the_only_change(chat):
    """Ruling 9 (S4, state 18): the size has its own way back, so it isn't counted."""
    client, _ = chat
    result(client, "set", "ctx_size 16384")
    assert 'data-action="reset-all"' not in panel(client)
    assert "changed" not in get_view(client, "picker")


def test_settings_say_the_size_s_base(chat):
    """The row's Back to needs the base: /settings' ctx_size has it, as `documents` does."""
    client, _ = chat
    result(client, "set", "ctx_size 16384")
    assert result(client, "settings")["ctx_size"] == {
        "value": 16384,
        "origin": "chat",
        "base": {"value": 8192, "origin": "server.toml"},
    }


def test_the_presets_wait_while_a_reply_runs(chat):
    """State 14: every preset waits and says why."""
    client, session = chat
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 1.0
    worker = threading.Thread(target=lambda: command(client, "say", "Hi"))
    worker.start()
    wait_until(lambda: writing(client))
    try:
        fragment = panel(client)
        during = worker.is_alive()
    finally:
        worker.join(10)
    assert during
    controls = re.findall(r"<button[^>]*>", row(context(fragment), "ctx_size"))
    assert len(controls) == 4
    assert all('aria-disabled="true"' in c and "settings-waits" in c for c in controls)


def test_the_thread_keeps_the_chat_and_says_its_model_is_starting(chat):
    """Decision 6: a new size starts the model again, drawn as Start the model again is:
    the chat's turns stay, with the start state under them, named by the chat's profile."""
    client, session = chat
    result(client, "say", "Hello")
    started = session.backend.start

    def slow(model, weights, ctx_size, on_start=None, **kwargs):
        if on_start:  # as llama.cpp's start: its checks pass, then the slow load
            on_start()
        time.sleep(1.0)
        return started(model, weights, ctx_size, **kwargs)

    session.backend.start = slow
    worker = threading.Thread(target=lambda: command(client, "set", "ctx_size 16384"))
    worker.start()
    try:
        wait_until(lambda: result(client, "state")["model"] == "starting")
        thread = get_view(client, "thread")
        picker = get_view(client, "picker")
    finally:
        worker.join(10)
    assert "Hello" in thread
    assert '<div class="starting"><p class="st" role="status">' in thread
    assert "Starting text…" in thread
    assert "Waits until the model has started." in picker


def test_each_choice_group_is_named_by_its_key(chat):
    """Mutation #24: the Context and method rows' buttons are a group labelled by the row's
    key, so a screen reader says which setting a button sets. The method row's: in
    tests/test_web_settings_documents.py."""
    client, _ = chat
    r = row(context(panel(client)), "ctx_size")
    assert '<span class="key" id="key-ctx_size">ctx_size</span>' in r
    assert '<div class="ctl" role="group" aria-labelledby="key-ctx_size">' in r
