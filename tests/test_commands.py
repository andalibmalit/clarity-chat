"""The command registry: one table that the terminal, --json and the web UI all use."""

from __future__ import annotations

import pytest

from clarity_chat.commands import Commands
from clarity_chat.profiles import ConfigError
from clarity_chat.session import CancelledBeforeReply, Session
from tests.fakes import FakeBackend


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!", " Again."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    return Commands(session), backend


def test_a_call_emits_its_text_and_every_record_it_logs(commands):
    c, _ = commands
    events: list[dict] = []
    started = c.run("profile", "text", events.append)
    assert started["profile"] == "text" and started["mode"] == "text"
    assert [e["status"] for e in events if e["event"] == "status"] == ["starting", "ready"]
    events.clear()
    result = c.run("say", "Hi", events.append)
    assert "".join(e["text"] for e in events if e["event"] == "text") == " Hello!"
    logged = [e["record"]["type"] for e in events if e["event"] == "record"]
    assert logged == ["turn", "call", "turn"]
    assert result["call"]["output_text"] == " Hello!"


def test_cancel_runs_during_a_call_and_stops_it(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.on_call = lambda: c.run("cancel", "", lambda e: None)  # from another thread, say
    with pytest.raises(CancelledBeforeReply):
        c.run("say", "Hi", lambda e: None)
    nothing = {"cancelled": False, "reason": "Nothing is running."}
    assert c.run("cancel", "", lambda e: None) == nothing


def test_cancel_says_when_the_running_command_cant_be_stopped(commands, fetched_root):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    answers = []
    backend.start_warnings = []
    # A model start can't be cancelled: /profile to another context size restarts it.
    (fetched_root / "profiles" / "big.toml").write_text(
        "ctx_size = 4096\n" + (fetched_root / "profiles" / "text.toml").read_text("utf-8"),
        encoding="utf-8",
    )
    original = backend.start

    def start(*args):
        answers.append(c.run("cancel", "", lambda e: None))
        return original(*args)

    backend.start = start
    c.run("profile", "big", lambda e: None)
    assert answers == [{"cancelled": False, "reason": "/profile can't be cancelled."}]


def test_commands_need_a_conversation_except_a_few(commands):
    c, _ = commands
    with pytest.raises(ConfigError, match="No conversation is running"):
        c.run("say", "Hi", lambda e: None)
    assert c.run("help", "", lambda e: None)["commands"][0]["usage"] == "/say TEXT"
    with pytest.raises(ConfigError, match="Unknown command /nope"):
        c.run("nope", "", lambda e: None)


def test_one_command_at_a_time_and_an_early_cancel_still_lands(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)

    def meanwhile():
        with pytest.raises(ConfigError, match="Another command is still running"):
            c.run("undo", "", lambda e: None)
        c.run("cancel", "", lambda e: None)

    backend.on_call = meanwhile
    with pytest.raises(CancelledBeforeReply):
        c.run("say", "Hi", lambda e: None)
    backend.on_call = None
    assert c.run("say", "Hi", lambda e: None)["call"]["output_text"] == " Again."


def test_regen_n_makes_n_branches_with_consecutive_seeds(commands):
    c, backend = commands
    backend.replies = [" first", " a", " b", " c"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    events: list[dict] = []
    result = c.run("regen", "3", events.append)
    starts = [e for e in events if e["event"] == "reply_start"]
    assert [(e["k"], e["n"], e["seed"]) for e in starts] == [(1, 3, 43), (2, 3, 44), (3, 3, 45)]
    calls = [r["call"] for r in result["replies"]]
    assert [call["settings"]["seed"]["value"] for call in calls] == [43, 44, 45]
    assert [call["output_text"] for call in calls] == [" a", " b", " c"]
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.turns[conversation.head].text == " c"  # the head is the last reply
    assert "stopped" not in result


def test_plain_regen_returns_a_list_of_one(commands):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    result = c.run("regen", "", lambda e: None)
    assert len(result["replies"]) == 1 and result["replies"][0]["call"]["regen_of"]


@pytest.mark.parametrize("args", ["0", "-1", "two", "1.5", "²", "٣"])
def test_regen_n_must_be_a_positive_whole_number(commands, args):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    with pytest.raises(ConfigError, match=r"/regen \[N\]"):
        c.run("regen", args, lambda e: None)


def test_a_failure_after_the_first_reply_stops_the_rest_and_keeps_the_replies(commands):
    c, backend = commands
    backend.replies = [" first", " a", " b"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    calls = []

    def fail_on_second():
        calls.append(1)
        if len(calls) == 2:
            backend.fail_with = "server gone"

    original = backend.complete

    def complete(*args, **kwargs):
        fail_on_second()
        return original(*args, **kwargs)

    backend.complete = complete
    result = c.run("regen", "3", lambda e: None)
    assert [r["call"]["output_text"] for r in result["replies"]] == [" a"]
    assert result["stopped"].startswith("Reply 2 of 3: server gone")
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.turns[conversation.head].text == " a"


def test_a_cancel_stops_the_reply_being_written_and_the_rest(commands):
    c, backend = commands
    backend.replies = [" first", " a", " longer reply", " c"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    calls = []

    def cancel_the_second():
        calls.append(1)
        if len(calls) == 2:
            backend.cancel_after = 3
            c.run("cancel", "", lambda e: None)

    backend.on_call = cancel_the_second
    result = c.run("regen", "3", lambda e: None)
    calls_made = [r["call"] for r in result["replies"]]
    assert [call["stop_type"] for call in calls_made] == ["eos", "cancelled"]
    assert len(backend.calls) == 3  # say, then two regenerations: the third never started
