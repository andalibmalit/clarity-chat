"""`clarity chat --json`: commands in as JSON lines, events out."""

from __future__ import annotations

import io
import json
import os
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
    stdin = io.StringIO("".join(line + "\n" for line in lines))
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
    assert {"event": "text", "text": " Hello!"} in events
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
    stdin = os.fdopen(read_end, "r", encoding="utf-8")
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
    assert next(e for e in events if e.get("id") == "c")["result"] == {"cancelled": True}
    said = next(e for e in events if e.get("id") == "s")
    assert said["event"] == "error" and said["type"] == "CancelledBeforeReply"


def test_a_closed_output_stops_writing_and_the_log_stays_whole(make):
    class Closed(io.StringIO):
        def write(self, s):
            raise BrokenPipeError

    session, _ = make([" one", " two"])
    session.new_conversation("text")
    stdin = io.StringIO(line("say", "a") + "\n" + line("say", "b") + "\n")
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
    ("set", lambda s: "temperature 0.9"),
    ("regen", lambda s: ""),
    ("edit", lambda s: "Hi again"),
    ("branches", lambda s: ""),
    ("switch", lambda s: s.conversation.head),
    ("mark", lambda s: ""),
    ("unmark", lambda s: ""),
    ("note", lambda s: "made-up note"),
    ("title", lambda s: "made-up title"),
    ("replay", lambda s: ""),
    ("undo", lambda s: ""),
    ("cancel", lambda s: ""),
    ("resume", lambda s: s.conversation.conversation_id),
    ("new", lambda s: ""),
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
    chat = JsonChat(session, io.StringIO(), io.StringIO())
    for name, args in EVERY_COMMAND:
        chat.stdout = io.StringIO()
        chat.execute({"command": name, "args": args(session)})
        last = json.loads(chat.stdout.getvalue().splitlines()[-1])
        assert last["event"] == "result", (name, last)


@pytest.mark.xfail(strict=True, reason="checkpoint 3: stdin is decoded by the console")
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
