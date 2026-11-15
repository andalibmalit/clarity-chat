"""Part 1's whole-branch review, should 6 (v0.3.1): the same bytes through the terminal,
`--json` and the page reach the one function and make the same record (A1's sibling paths),
and an error while reading is a refusal on each, never a stopped session (M5's front-end
half). Made-up documents only."""

from __future__ import annotations

import io
import json

import pytest
from starlette.testclient import TestClient

import clarity_chat.documents as documents
from clarity_chat import web
from clarity_chat.cli import ChatLoop
from clarity_chat.documents import reading
from clarity_chat.json_chat import JsonChat
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import GET, ORIGIN, PORT, SECRET, command
from tests.test_web_upload import upload, view

NOTES = b"Line one.\nLine two.\nLine three.\n"
REFUSAL = "Clarity couldn't read notes.md (RuntimeError: made-up failure)."


def session_for(root, tokenizer) -> Session:
    reading._last = None
    backend = FakeBackend([" Hello!"] * 3, tokenizer=tokenizer)
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)


def terminal(root, tokenizer, path) -> tuple[Session, str]:
    session = session_for(root, tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    output.clear()
    assert loop.handle(f"/attach {path}")
    assert loop.handle("/state")  # the session goes on
    return session, "".join(output)


def json_chat(root, tokenizer, path) -> tuple[Session, list[dict]]:
    session = session_for(root, tokenizer)
    stdout = io.StringIO()
    chat = JsonChat(session, io.BytesIO(), stdout)
    assert chat.execute({"command": "profile", "args": "chat"}, at_once=False)
    assert chat.execute({"command": "attach", "args": str(path)})
    assert chat.execute({"command": "state", "args": ""})  # the session goes on
    return session, [json.loads(line) for line in stdout.getvalue().splitlines()]


def page(root, tokenizer) -> tuple[Session, dict]:
    session = session_for(root, tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as c:
        command(c, "profile", "chat")
        upload_id = upload(c, NOTES, "notes.md").json()["upload"]
        answer = command(c, "attach", upload_id).json()
        assert view(c, "pending").status_code == 200  # the session goes on
        assert c.get("/view/thread", headers=GET).status_code == 200
    return session, answer


@pytest.fixture
def notes(tmp_path):
    path = tmp_path / "notes.md"
    path.write_bytes(NOTES)
    return path


def test_the_same_bytes_make_the_same_record_on_every_front_end(
    fetched_root, chat_tokenizer, notes
):
    sessions = [
        terminal(fetched_root, chat_tokenizer, notes)[0],
        json_chat(fetched_root, chat_tokenizer, notes)[0],
        page(fetched_root, chat_tokenizer)[0],
    ]
    records = [s.conversation.pending for s in sessions if s.conversation is not None]
    assert len(records) == 3
    assert records[0] == records[1] == records[2] and len(records[0]) == 1


def test_an_error_while_reading_is_a_refusal_on_every_front_end(
    fetched_root, chat_tokenizer, notes, monkeypatch
):
    def broken(data, name, cancel=None, on_read=None):
        raise RuntimeError("made-up failure")

    monkeypatch.setattr(documents, "extract", broken)
    _, printed = terminal(fetched_root, chat_tokenizer, notes)
    assert f"! {REFUSAL}\n" in printed
    _, events = json_chat(fetched_root, chat_tokenizer, notes)
    refused = next(e for e in events if e.get("command") == "attach")
    assert (refused["event"], refused["message"]) == ("error", REFUSAL)
    assert events[-1]["event"] == "result"  # /state answered after it
    _, answer = page(fetched_root, chat_tokenizer)
    assert answer["error"] == REFUSAL and not answer.get("stopped")
