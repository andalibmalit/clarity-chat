"""`clarity chat --json`: commands in as JSON lines, events out."""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import time

import pytest

from clarity_chat.cli import ChatLoop
from clarity_chat.commands import COMMANDS
from clarity_chat.json_chat import JsonChat
from clarity_chat.logs import read_records
from clarity_chat.session import Session
from tests.fakes import FakeBackend


@pytest.fixture
def make(fetched_root, tiny_tokenizer):
    def make(replies=None):
        backend = FakeBackend(replies, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        return session, backend

    return make


def run(session, lines, first=()):
    stdout = io.StringIO()
    stdin = io.BytesIO("".join(line + "\n" for line in lines).encode())
    JsonChat(session, stdin, stdout).run(list(first))
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def line(command, args="", **extra):
    return json.dumps({"command": command, "args": args, **extra})


def test_a_scripted_chat(make):
    session, _ = make([" Hello!"])
    events = run(
        session,
        [
            line("say", "Hi", id=1),
            line("set", "temperature 0.9", id="t"),
            "not json",
            '{"args": "no command"}',
            line("nope"),
            line("quit"),
            line("say", "never read"),
        ],
        first=[{"command": "profile", "args": "text"}],
    )
    statuses = [e["status"] for e in events if e["event"] == "status"]
    assert events[0]["event"] == "status" and statuses == ["starting", "ready"]
    started = next(e for e in events if e["event"] == "result")
    assert started["command"] == "profile" and started["result"]["profile"] == "text"
    said = next(e for e in events if e.get("id") == 1)
    assert said["event"] == "result" and said["result"]["call"]["output_text"] == " Hello!"
    assert {"event": "text", "text": " Hello!", "at": 0, "end": 7} in events
    logged = [e["record"]["type"] for e in events if e["event"] == "record"]
    assert logged == [
        r["type"] for r in read_records(session.conversation.log.path)
    ]  # record events are the log, in order
    assert next(e for e in events if e.get("id") == "t")["result"]["setting"] == {
        "value": 0.9,
        "origin": "chat",
    }
    errors = [e for e in events if e["event"] == "error"]
    assert [e["type"] for e in errors] == ["InputError", "InputError", "ConfigError"]
    assert "Unknown command /nope" in errors[2]["message"]
    assert events[-1] == {"event": "result", "command": "quit", "result": {"quit": True}}


def test_cancel_runs_while_a_reply_is_written(make):
    session, backend = make([" a long reply"])
    read_end, write_end = os.pipe()
    stdin = os.fdopen(read_end, "rb")
    writer = os.fdopen(write_end, "w", encoding="utf-8")
    stdout = io.StringIO()
    chat = JsonChat(session, stdin, stdout)

    def cancel_mid_call():
        writer.write(line("cancel", id="c") + "\n")
        writer.flush()
        deadline = time.monotonic() + 5
        event = chat.commands.cancel_event
        assert event is not None  # made when the command was accepted
        while not event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

    backend.on_call = cancel_mid_call
    writer.write(line("say", "Hi", id="s") + "\n")
    writer.flush()
    thread = threading.Thread(target=chat.run, args=([{"command": "profile", "args": "text"}],))
    thread.start()
    deadline = time.monotonic() + 5
    while '"id": "s"' not in stdout.getvalue() and time.monotonic() < deadline:
        time.sleep(0.01)
    writer.close()
    thread.join(5)
    events = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert next(e for e in events if e.get("id") == "c")["result"] == {
        "cancelled": True,
        "command": "say",
    }
    said = next(e for e in events if e.get("id") == "s")
    assert said["event"] == "error" and said["type"] == "CancelledBeforeReply"


def test_a_closed_output_stops_writing_and_the_log_stays_whole(make):
    class Closed(io.StringIO):
        def write(self, s):
            raise BrokenPipeError

    session, _ = make([" one", " two"])
    session.new_conversation("text")
    stdin = io.BytesIO((line("say", "a") + "\n" + line("say", "b") + "\n").encode())
    chat = JsonChat(session, stdin, Closed())
    chat.run([])  # returns instead of raising
    assert chat.closed
    turns = [r for r in read_records(session.conversation.log.path) if r["type"] == "turn"]
    assert [t["text"] for t in turns] in (["a", " one"], ["a", " one", "b", " two"])
    assert session.conversation.head == turns[-1]["turn_id"]


# Every command in the table, in an order that gives each one something to act on.
EVERY_COMMAND = [
    ("help", lambda s: ""),
    ("chats", lambda s: ""),
    ("profile", lambda s: "text"),
    ("say", lambda s: "Hi"),
    ("prompt", lambda s: ""),
    ("settings", lambda s: ""),
    ("probs", lambda s: ""),
    ("set", lambda s: "temperature 0.9"),
    ("regen", lambda s: ""),
    ("edit", lambda s: "Hi again"),
    ("history", lambda s: ""),
    ("compare", lambda s: ""),
    ("branches", lambda s: ""),
    ("switch", lambda s: s.conversation.head),
    ("mark", lambda s: ""),
    ("unmark", lambda s: ""),
    ("note", lambda s: "made-up note"),
    ("title", lambda s: "made-up title"),
    ("replay", lambda s: ""),
    ("export", lambda s: ""),
    ("undo", lambda s: ""),
    ("cancel", lambda s: ""),
    ("resume", lambda s: s.conversation.conversation_id),
    ("new", lambda s: ""),
    ("profiles", lambda s: ""),
    ("state", lambda s: ""),
    ("quit", lambda s: ""),
]


def test_every_command_works_as_a_slash_command_and_as_json(make):
    assert sorted(name for name, _ in EVERY_COMMAND) == sorted(COMMANDS)
    session, _ = make()
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda d: "", write=output.append)
    for name, args in EVERY_COMMAND:
        output.clear()
        keep_going = loop.handle(f"/{name} {args(session)}".rstrip())
        assert keep_going == (name != "quit"), name
        if name != "quit":
            text = "".join(output)
            assert text.strip() and not text.startswith("!"), (name, text)
    session, _ = make()
    chat = JsonChat(session, io.BytesIO(), io.StringIO())
    for name, args in EVERY_COMMAND:
        chat.stdout = io.StringIO()
        chat.execute({"command": name, "args": args(session)})
        last = json.loads(chat.stdout.getvalue().splitlines()[-1])
        assert last["event"] == "result", (name, last)


def test_clarity_chat_json_reads_utf8_and_survives_a_bad_byte(fetched_root, monkeypatch, capsys):
    from clarity_chat import cli

    raw = b'\xff\n{"command": "help", "args": "caf\xc3\xa9"}\n'
    # A console that decodes pipes as cp1252, as Windows does.
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252"))
    done = []
    thread = threading.Thread(
        target=lambda: done.append(cli.main(["--root", str(fetched_root), "chat", "--json"])),
        daemon=True,
    )
    thread.start()
    thread.join(10)
    assert done == [0], "clarity chat --json hung or failed"
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert events[0]["type"] == "InputError" and "UTF-8" in events[0]["message"]
    assert events[1]["command"] == "help" and events[1]["event"] == "result"


def test_cancel_says_what_it_stopped_or_why_not(make):
    session, backend = make()
    events: list[dict] = []
    chat = JsonChat(session, io.BytesIO(), io.StringIO())
    chat.emit = lambda event: events.append(event)  # type: ignore[method-assign]
    chat.execute({"command": "profile", "args": "text"})
    chat.execute({"command": "say", "args": "a"})
    chat.execute({"command": "cancel", "id": "c"})  # nothing runs now
    nothing = events[-1]["result"]
    assert nothing["cancelled"] is False and "Nothing" in nothing["reason"]
    backend.on_call = lambda: chat.execute({"command": "cancel", "id": "c2"})
    chat.execute({"command": "replay", "id": "r"})
    stopped = next(e for e in events if e.get("id") == "c2")["result"]
    assert stopped == {"cancelled": True, "command": "replay"}
    replayed = events[-1]
    assert replayed["event"] == "error" and "cancelled" in replayed["message"].lower()
    assert not [r for r in read_records(session.conversation.log.path) if r["type"] == "replay"]


def test_deeply_nested_input_is_an_input_error_and_reading_goes_on(make):
    session, _ = make([" Hi."])
    events = run(session, ["[" * 100_000, line("help", id=2)])
    errors = [e for e in events if e["event"] == "error"]
    assert [e["type"] for e in errors] == ["InputError"]
    assert any(e.get("id") == 2 and e["event"] == "result" for e in events)


def test_an_unexpected_error_is_reported_and_ends_the_session(make):
    session, backend = make([" Hi."])

    def fail():
        raise OSError("disk full")

    backend.on_call = fail
    stdout = io.StringIO()
    stdin = io.BytesIO((line("say", "Hi", id=1) + "\n" + line("help", id=2) + "\n").encode())
    chat = JsonChat(session, stdin, stdout)
    chat.run([{"command": "profile", "args": "text"}])
    events = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert chat.failed  # so `clarity chat --json` exits with status 1
    error = events[-1]
    assert error == {
        "event": "error",
        "command": "say",
        "id": 1,
        "type": "OSError",
        "message": "disk full",
    }
    assert not any(e.get("id") == 2 for e in events)  # the session ended there


def test_a_failed_start_from_the_command_line_ends_the_session_with_status_1(
    fetched_root, monkeypatch, capsys
):
    """As in the terminal: if --profile can't start, nothing after it runs, and the exit
    status says so."""
    from clarity_chat import cli

    raw = b'{"command": "help"}\n'
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    argv = ["--root", str(fetched_root), "chat", "--json", "--profile", "no-such-profile"]
    done = []
    thread = threading.Thread(target=lambda: done.append(cli.main(argv)), daemon=True)
    thread.start()
    thread.join(10)
    assert done == [1], "clarity chat --json hung, or exited 0 after a failed start"
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [e["event"] for e in events if e["event"] != "status"] == ["error"]
    assert events[-1]["command"] == "profile" and events[-1]["type"] == "ConfigError"


def test_the_start_takes_a_path_but_input_lines_dont(make, fetched_root):
    session, _ = make([" Hello!"])
    path = str(fetched_root / "profiles" / "text.toml")
    events = run(
        session, [line("profile", path, id=1)], first=[{"command": "profile", "args": path}]
    )
    started = next(e for e in events if e["event"] == "result")
    assert started["result"]["profile"] == "text"
    refused = next(e for e in events if e.get("id") == 1)
    assert refused["event"] == "error" and "not a path" in refused["message"]


@pytest.mark.parametrize("first", [["--profile", "no-such-profile"], []])
def test_quitting_with_input_still_open_exits_cleanly(root, first):
    """A script keeps Clarity's input open while it waits. The reader thread is then blocked
    reading it at exit, and Python's shutdown aborted (status -6, "Fatal Python error")."""
    import subprocess
    import sys

    code = "import sys; from clarity_chat.cli import main; sys.exit(main(sys.argv[1:]))"
    argv = [sys.executable, "-c", code, "--root", str(root), "chat", "--json", *first]
    with subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        assert process.stdin is not None
        if not first:  # no start: quit by command, input left open
            process.stdin.write(b'{"command": "quit"}\n')
            process.stdin.flush()
        process.wait(timeout=60)
        err = process.stderr.read().decode() if process.stderr else ""
    assert "Fatal Python error" not in err
    assert process.returncode == (1 if first else 0)


class Chunks(io.RawIOBase):
    """Raw input that hands out the given chunks one read at a time, like a pipe, and
    counts the reads."""

    def __init__(self, chunks):
        self.chunks, self.reads = [bytes(c) for c in chunks], 0

    def readable(self):
        return True

    def readinto(self, buffer):
        self.reads += 1
        if not self.chunks:
            return 0
        chunk = self.chunks.pop(0)
        if len(chunk) > len(buffer):
            chunk, self.chunks[:0] = chunk[: len(buffer)], [chunk[len(buffer) :]]
        buffer[: len(chunk)] = chunk
        return len(chunk)


def read_all(make, stdin):
    session, _ = make()
    pending = queue.Queue()
    JsonChat(session, stdin, io.StringIO()).read(pending)
    commands = []
    while (command := pending.get()) is not None:
        commands.append(command)
    return commands


def test_input_is_read_in_large_chunks(make):
    one = (line("settings", "x" * 1000) + "\n").encode()
    data = one * (1_000_000 // len(one) + 1)
    stdin = Chunks([data])
    commands = read_all(make, stdin)
    assert len(commands) == data.count(b"\n")
    assert stdin.reads < 100


def test_lines_and_characters_split_across_reads_arrive_whole(make):
    text = (
        json.dumps({"command": "say", "args": "é", "id": 1}, ensure_ascii=False) + "\n"
    ).encode()
    cut = text.index("é".encode()) + 1  # between the character's two bytes
    chunks = [text[:5], text[5:cut], text[cut:] + line("help", id=2).encode()]
    commands = read_all(make, Chunks(chunks))  # the last line has no newline
    assert [(c["command"], c.get("args"), c["id"]) for c in commands] == [
        ("say", "é", 1),
        ("help", "", 2),
    ]


def test_state_and_cancel_run_as_soon_as_they_are_read(make):
    """A page reloaded mid-reply asks /state; it mustn't wait behind the reply."""
    import queue

    session, _ = make()
    lines = [
        json.dumps({"command": "say", "args": "Hi"}),
        json.dumps({"command": "state", "args": ""}),
        json.dumps({"command": "cancel", "args": ""}),
    ]
    stdout = io.StringIO()
    chat = JsonChat(session, io.BytesIO("\n".join(lines).encode() + b"\n"), stdout)
    pending: queue.Queue = queue.Queue()
    chat.read(pending)  # reading only: nothing from the queue has run
    queued = [pending.get_nowait() for _ in range(pending.qsize())]
    assert [c and c["command"] for c in queued] == ["say", None]
    results = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [r["event"] for r in results] == ["result", "result"]
    assert results[0]["result"]["running"] is None  # state, before the say ran
