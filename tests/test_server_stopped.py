"""A model server that stopped between calls: the next call says so, a status event marks
the model stopped, and resuming the conversation starts it again (v0.2.3 item 13)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.logs import read_records
from clarity_chat.profiles import load_defaults, load_server_config
from clarity_chat.session import Session
from tests.fakes import FakeBackend


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" one", " two", " three"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    return c, backend


def test_llama_cpp_says_its_server_stopped_and_sends_nothing(root):
    from clarity_chat.backends import llama_cpp

    backend = llama_cpp.LlamaCppBackend(load_server_config(root), root / "logs")
    exited = SimpleNamespace(poll=lambda: -9)
    # Port 9 (discard): nothing listens, so a request would fail as a connection error.
    server = SimpleNamespace(proc=exited, url="http://127.0.0.1:9", headers={})
    backend.server = server  # ty: ignore[invalid-assignment]
    settings = {**load_defaults(root), "seed": 42, "max_tokens": 4}
    with pytest.raises(BackendError, match="llama-server has stopped") as caught:
        backend.complete([1, 2], settings, [])
    assert getattr(caught.value, "server_stopped", False) is True
    assert "exit code -9" in str(caught.value) and "nothing was sent" in str(caught.value)


def test_the_next_call_says_the_model_stopped_and_a_status_event_marks_it(commands):
    c, backend = commands
    c.run("say", "a", lambda e: None)
    backend.stopped = True
    events: list[dict] = []
    with pytest.raises(BackendError, match="has stopped"):
        c.run("say", "b", events.append)
    assert {"event": "status", "status": "stopped"} in events
    assert c.run("state", "", lambda e: None)["model"] == "stopped"
    failed = c.session.conversation.last_call
    assert failed["error"].startswith("BackendError") and "has stopped" in failed["error"]


def test_resuming_the_open_chat_starts_the_model_again(commands):
    """Its side effects, said in the docs: a resume record, and the chat counts as resumed."""
    c, backend = commands
    c.run("say", "a", lambda e: None)
    backend.stopped = True
    with pytest.raises(BackendError):
        c.run("say", "b", lambda e: None)
    conversation_id = c.session.conversation.conversation_id
    starts = len(backend.started)
    c.run("resume", conversation_id, lambda e: None)
    assert len(backend.started) == starts + 1 and backend.stopped is False
    assert c.run("state", "", lambda e: None)["model"] == "ready"
    assert c.session.conversation.resumed is True
    reply = c.run("say", "b", lambda e: None)["call"]
    assert reply["output_text"] == " two"
    log = list(read_records(c.session.conversation.log.path))
    assert [r["type"] for r in log].count("resume") == 1
    assert log[-1]["type"] == "turn" and log[-1]["text"] == " two"


def test_the_terminal_says_how_to_start_it_again(fetched_root, tiny_tokenizer):
    from clarity_chat.cli import ChatLoop

    backend = FakeBackend([" one"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda d: "", write=output.append)
    loop.start()
    backend.stopped = True
    loop.handle("Hi")
    assert session.conversation is not None
    assert f"/resume {session.conversation.conversation_id} starts it again." in "".join(output)


def test_a_server_that_stops_during_regen_n_is_reported_stopped(commands):
    """Review: /regen N kept the failure as `stopped`, and no status event went out."""
    c, backend = commands
    c.run("say", "a", lambda e: None)
    backend.replies = [" two", " three"]
    events: list[dict] = []

    def on_event(event):
        events.append(event)
        if event["event"] == "reply_start" and event["k"] == 2:
            backend.stopped = True

    result = c.run("regen", "2", on_event)
    assert len(result["replies"]) == 1 and "has stopped" in result["stopped"]
    assert {"event": "status", "status": "stopped"} in events
    assert c.run("state", "", lambda e: None)["model"] == "stopped"


def test_a_failed_resume_after_a_stop_still_says_stopped(commands):
    """Review: a start that failed set the model "ready" whenever a conversation was open."""
    c, backend = commands
    c.run("say", "a", lambda e: None)
    backend.stopped = True
    with pytest.raises(BackendError):
        c.run("say", "b", lambda e: None)
    with pytest.raises(Exception):  # noqa: B017 - which error isn't the point
        c.run("resume", "no-such-conversation", lambda e: None)
    assert c.run("state", "", lambda e: None)["model"] == "stopped"


def test_a_replay_that_starts_the_server_again_leaves_it_ready(commands):
    """Review: replay restarts a stopped server without an event."""
    c, backend = commands
    first = c.run("say", "a", lambda e: None)["call"]
    backend.stopped = True
    with pytest.raises(BackendError):
        c.run("say", "b", lambda e: None)
    backend.replies = [" one"]
    c.run("replay", first["call_id"], lambda e: None)
    assert c.run("state", "", lambda e: None)["model"] == "ready"


def test_llama_cpp_says_its_server_stopped_when_the_stream_just_ends(monkeypatch, root):
    """Review: a stream that ends without its final chunk, from a process that exited."""
    from tests.test_llama_cpp import fake_backend

    backend = fake_backend(monkeypatch, root, [{"content": "Hi", "tokens": [5]}])
    settings = {**load_defaults(root), "seed": 42, "max_tokens": 4}
    with pytest.raises(BackendError, match="stopped during the call") as caught:
        backend.complete([1, 2], settings, [])
    assert caught.value.server_stopped is True


def test_llama_cpp_says_its_server_stopped_when_the_connection_drops(monkeypatch, root):
    """Final checkpoint, finding 6: the connection-error path, with the process reaped late."""
    import httpx

    from clarity_chat.backends import llama_cpp
    from tests.test_llama_cpp import ExitedServer

    backend = llama_cpp.LlamaCppBackend(load_server_config(root), root / "logs")
    monkeypatch.setattr(backend, "server", ExitedServer())

    def drop(*args, **kwargs):
        raise httpx.ReadError("connection reset")

    monkeypatch.setattr(llama_cpp.httpx, "stream", drop)
    settings = {**load_defaults(root), "seed": 42, "max_tokens": 4}
    with pytest.raises(BackendError, match="stopped during the call") as caught:
        backend.complete([1, 2], settings, [])
    assert caught.value.server_stopped is True


def test_the_terminal_suggests_resume_when_regen_n_finds_the_server_stopped(
    fetched_root, tiny_tokenizer
):
    """Fix review, finding 3."""
    from clarity_chat.cli import ChatLoop

    backend = FakeBackend([" one", " two", " three"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda d: "", write=output.append)
    loop.start()
    loop.handle("Hi")
    backend.on_call = lambda: setattr(backend, "stopped", True)  # after reply 1's own check
    loop.handle("/regen 2")
    assert session.conversation is not None
    assert f"/resume {session.conversation.conversation_id} starts it again." in "".join(output)
