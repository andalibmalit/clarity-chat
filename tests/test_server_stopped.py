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
