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


# --- The tripwire review after C5 (findings 1 and 3)


def test_a_llama_start_refused_before_launch_leaves_the_server_stopped(root):
    """Finding 1: "Start the model again" after llama.cpp moved or was rebuilt: start()
    replaces the exited server, then refuses before launching it. The model is stopped
    (not ready with no server), and a call says so instead of posting to nothing."""
    from clarity_chat.backends import llama_cpp
    from clarity_chat.profiles import ConfigError, load_models

    backend = llama_cpp.LlamaCppBackend(load_server_config(root), root / "logs")
    exited = SimpleNamespace(poll=lambda: 1)
    backend.server = SimpleNamespace(proc=exited, record={}, gguf=None, stop=lambda: None)  # ty: ignore[invalid-assignment]
    assert backend.stopped
    weights = root / "m.gguf"
    weights.write_bytes(b"GGUF")
    with pytest.raises(ConfigError):  # root has no llama.cpp checkout
        backend.start(load_models(root)["tiny"], weights, 512)
    assert backend.stopped is True
    settings = {**load_defaults(root), "seed": 42, "max_tokens": 4}
    with pytest.raises(BackendError) as caught:
        backend.complete([1, 2], settings, [])
    assert caught.value.server_stopped is True


def refuse(backend, error: Exception) -> None:
    def refused(*args):
        raise error

    backend.start = refused


@pytest.mark.parametrize("kind", ["config", "backend"])
def test_a_refused_restart_of_the_open_chat_says_stopped_and_keeps_it(commands, kind):
    """Findings 1 and 3: /resume of the open chat after its server stopped, refused with a
    ConfigError (llama.cpp moved) or a BackendError, keeps the chat with its model stopped,
    and sends status: stopped."""
    from clarity_chat.profiles import ConfigError

    c, backend = commands
    c.run("say", "a", lambda e: None)
    conversation = c.session.conversation
    backend.stopped = True
    error = ConfigError("llama.cpp moved.") if kind == "config" else BackendError("No GPU.")
    refuse(backend, error)
    events: list[dict] = []
    with pytest.raises(type(error)) as caught:
        c.run("resume", conversation.log.path.stem, events.append)
    assert str(caught.value) == str(error)
    assert c.session.conversation is conversation
    assert {"event": "status", "status": "stopped"} in events
    assert c.run("state", "", lambda e: None)["model"] == "stopped"


def test_a_replay_with_a_refused_restart_keeps_the_chat(commands):
    """Finding 3: Details' Replay with the server stopped tries the same start once."""
    c, backend = commands
    first = c.run("say", "a", lambda e: None)["call"]
    conversation = c.session.conversation
    backend.stopped = True
    refuse(backend, BackendError("llama-server failed to start: exited with code 1:\nline"))
    with pytest.raises(BackendError) as caught:
        c.run("replay", first["call_id"], lambda e: None)
    assert "failed too" not in str(caught.value)
    assert c.session.conversation is conversation


def test_new_on_the_same_profile_with_a_refused_start_keeps_the_chat(commands):
    """Finding 3: /new on the same profile after a crash: the same start, tried once."""
    c, backend = commands
    c.run("say", "a", lambda e: None)
    conversation = c.session.conversation
    backend.stopped = True
    refuse(backend, BackendError("llama-server failed to start: exited with code 1:\nline"))
    with pytest.raises(BackendError) as caught:
        c.run("new", "", lambda e: None)
    assert "failed too" not in str(caught.value)
    assert c.session.conversation is conversation
    assert c.run("state", "", lambda e: None)["model"] == "stopped"


def test_with_no_chat_open_a_stopped_server_reads_as_no_model(fetched_root, tiny_tokenizer):
    """A refused first start leaves a server with no process (finding 1): with no chat
    open there's nothing to start again, so /state says none, as before."""
    backend = FakeBackend([], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    backend.stopped = True
    assert Commands(session).run("state", "", lambda e: None)["model"] == "none"


def test_the_terminal_names_the_log_to_resume_a_copied_chat(fetched_root, tiny_tokenizer):
    """Finding 4: /resume takes the log file's name, which differs from the conversation
    ID for a copied or renamed log (CP4 finding 10's sibling, in the terminal)."""
    import shutil

    from clarity_chat.cli import ChatLoop

    backend = FakeBackend([" one", " two"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    first = session.new_conversation("text")
    session.say("a")
    copy = shutil.copy(first.log.path, session.logs_dir / "my-copy.jsonl")
    output: list[str] = []
    loop = ChatLoop(session, "", read=lambda d: "", write=output.append, resume=str(copy))
    loop.start()
    resumed = session.conversation
    assert resumed is not None and resumed.conversation_id != "my-copy"
    backend.stopped = True
    loop.handle("Hi")
    assert "/resume my-copy starts it again." in "".join(output)


# --- The whole-branch review (multi-agent, batch 1)


def test_a_model_file_gone_at_start_says_so_plainly(root):
    """Finding 9: a GGUF missing at start raised FileNotFoundError, outside the handlers
    that keep the chat and say what to do."""
    from clarity_chat.backends import llama_cpp
    from clarity_chat.profiles import ConfigError, load_models

    backend = llama_cpp.LlamaCppBackend(load_server_config(root), root / "logs")
    missing = root / "models" / "tiny" / "gone.gguf"
    caught: Exception | None = None
    try:
        backend.start(load_models(root)["tiny"], missing, 512)
    except Exception as e:  # noqa: BLE001 - which error is the point
        caught = e
    assert isinstance(caught, ConfigError), repr(caught)
    assert str(missing) in str(caught) and "uv run clarity verify tiny" in str(caught)


def test_the_reply_is_in_reply_so_far_or_the_turns_at_every_moment(commands):
    """Finding 7: the reply's turn record cleared reply_so_far before the head moved, so an
    at-once /state in between saw the reply in neither."""
    c, _ = commands
    seen: list[dict] = []

    def emit(event):
        record = event.get("record") or {}
        if record.get("type") == "turn" and record.get("role") == "assistant":
            seen.append(c.run("state", "", lambda e: None))  # at once, as the page's read

    c.run("say", "a", emit)
    (state,) = seen
    texts = [t["text"] for t in state["turns"]]
    assert state["reply_so_far"] == " one" or texts[-1:] == [" one"], state["reply_so_far"]
    assert not (state["reply_so_far"] and texts[-1:] == [" one"])  # and never both


def test_the_terminal_hint_works_for_a_dotted_log_name(fetched_root, tiny_tokenizer):
    """The Opus review, SHOULD 2: the hint names the log, and /resume takes that name."""
    import shutil

    from clarity_chat.cli import ChatLoop

    backend = FakeBackend([" one", " two"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    first = session.new_conversation("text")
    session.say("a")
    shutil.copy(first.log.path, session.logs_dir / "chat.v2.jsonl")
    output: list[str] = []
    loop = ChatLoop(session, "", read=lambda d: "", write=output.append, resume="chat.v2")
    loop.start()
    backend.stopped = True
    loop.handle("Hi")
    assert "/resume chat.v2 starts it again." in "".join(output)
    output.clear()
    loop.handle("/resume chat.v2")
    assert "Resumed" in "".join(output), "".join(output)
