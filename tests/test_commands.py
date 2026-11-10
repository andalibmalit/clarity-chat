"""The command registry: one table that the terminal, --json and the web UI all use."""

from __future__ import annotations

import pytest

from clarity_chat.commands import COMMANDS, Commands
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
    assert c.run("cancel", "", lambda e: None) == {"cancelled": False}  # nothing running now


def test_commands_need_a_conversation_except_a_few(commands):
    c, _ = commands
    with pytest.raises(ConfigError, match="No conversation is running"):
        c.run("say", "Hi", lambda e: None)
    assert c.run("help", "", lambda e: None)["commands"][0]["usage"] == "/say TEXT"
    with pytest.raises(ConfigError, match="Unknown command /nope"):
        c.run("nope", "", lambda e: None)


def test_only_cancel_runs_during_a_call():
    assert [name for name, cmd in COMMANDS.items() if cmd.during_call] == ["cancel"]
