"""/state: one read a page can make at any moment, mid-reply included, to draw itself."""

from __future__ import annotations

import threading

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.session import Session
from tests.fakes import FakeBackend


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hi there, you."] * 3, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    return Commands(session), backend


def state(c):
    return c.run("state", "", lambda e: None)


def test_state_answers_before_any_conversation(commands):
    c, _ = commands
    result = state(c)
    assert result.get("conversation_id") is None and result["model"] == "none"
    assert result["running"] is None and result["turns"] == []
    assert result["clarity"]["version"] and "commit" in result["clarity"]
    assert len(result["llama_cpp_commit"]) == 40


def test_state_answers_while_a_reply_streams_with_its_text_so_far(commands):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    seen = []

    def on_event(event):
        if event["event"] == "text" and len(seen) < 2:
            seen.append(state(c))  # during the call, from inside it

    c.run("say", "Hi", on_event)
    assert [s.get("running") for s in seen] == ["say", "say"]
    assert seen[0]["reply_so_far"] == " Hi "  # the fake streams a word at a time
    assert seen[1]["reply_so_far"] == " Hi there, "
    after = state(c)
    assert after["running"] is None and after["reply_so_far"] == ""
    assert [t["text"] for t in after["turns"]] == ["Hi", " Hi there, you."]
    assert after["model"] == "ready" and after["head"] == after["turns"][-1]["turn_id"]


def test_reading_turns_from_another_thread_during_calls_is_safe(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.replies = [" word" * 50] * 20
    errors: list[BaseException] = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            try:
                state(c)
            except BaseException as e:  # noqa: BLE001 - any error here is the finding
                errors.append(e)
                return

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(20):
            c.run("say", f"m{i}", lambda e: None)
    finally:
        done.set()
        thread.join()
    assert errors == []
    assert len(state(c)["turns"]) == 40


def test_state_after_a_cancel_and_after_a_failure(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.cancel_after = 4
    c.run("say", "Hi", lambda e: None)
    backend.cancel_after = None
    after_cancel = state(c)
    assert after_cancel.get("unanswered") is None and after_cancel["running"] is None
    assert after_cancel["turns"][-1]["text"] == " Hi "
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        c.run("say", "Again", lambda e: None)
    after_failure = state(c)
    assert after_failure.get("unanswered", {}).get("text") == "Again"
    assert after_failure["head"] == after_cancel["head"]  # back where it was
    backend.fail_with = None
    c.run("say", "Third", lambda e: None)
    assert state(c).get("unanswered") is None


def test_state_restores_the_title_warnings_and_resume_differences(commands, fetched_root):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    c.run("title", "A made-up title", lambda e: None)
    conversation_id = c.session.conversation.conversation_id
    assert state(c).get("title") == "A made-up title"
    c.run("resume", conversation_id, lambda e: None)
    result = state(c)
    assert result.get("resume", {}).get("type") == "resume"
    assert result["warnings"] == [] and result["profile"] == "text"


def test_the_starting_event_names_the_profile_not_its_file(commands):
    c, _ = commands
    events: list[dict] = []
    c.run("profile", "text", events.append)
    starting = next(e for e in events if e.get("status") == "starting")
    assert starting["profile"] == "text"


def test_the_terminal_prints_state_in_one_line(commands):
    from clarity_chat.cli import format_state

    c, _ = commands
    assert format_state(state(c)).startswith("No conversation; no model loaded.")
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    text = format_state(state(c))
    assert f"Conversation {c.session.conversation.conversation_id} (profile text)" in text
    assert "2 turns on this branch; the model is ready." in text
