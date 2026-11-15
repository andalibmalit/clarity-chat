"""The security review's shoulds 2 to 4 (v0.3.1): numbers too long for int(), a parse that
mustn't backtrack, and --attach reading exactly the file named. Made-up documents only."""

from __future__ import annotations

import time

import pytest

from clarity_chat.commands import Commands, attach_args, read_files
from clarity_chat.documents import reading
from clarity_chat.profiles import ConfigError
from tests.test_cli_attach import chat_loop
from tests.test_session_attach import chat_session
from tests.test_web import GET, ORIGIN, PORT, SECRET, command
from tests.test_web_upload import GOOD, raw, upload, view

HUGE = "1" * 4301  # Python's int() refuses more than 4,300 digits


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


@pytest.fixture
def web_chat(fetched_root, chat_tokenizer):
    from starlette.testclient import TestClient

    from clarity_chat import web
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        yield client, session, app


# --- should 2: a number of any length is a refusal, never a ValueError


def test_regen_with_a_huge_number_is_refused(chat):
    session, _ = chat
    c = Commands(session)
    c.run("say", "Hi", lambda e: None)
    with pytest.raises(ConfigError) as e:
        c.run("regen", HUGE, lambda e: None)
    assert str(e.value).startswith("Usage: /regen [N]")


def test_detach_with_a_huge_number_is_refused(chat):
    session, _ = chat
    with pytest.raises(ConfigError) as e:
        Commands(session).run("detach", HUGE, lambda e: None)
    assert str(e.value) == "No attachment 1111111111…. /attach lists them."


def test_attach_with_a_huge_range_is_refused(chat, tmp_path):
    session, _ = chat
    path = tmp_path / "notes.md"
    path.write_text("Made-up.\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        Commands(session).run("attach", f"{path} lines 1-{HUGE}", lambda e: None)
    assert str(e.value) == "A range's numbers have at most 9 digits."


def test_the_attach_view_refuses_a_huge_range(web_chat):
    client, _, _ = web_chat
    upload_id = upload(client).json()["upload"]
    answer = view(client, "attach", upload=upload_id, unit="lines", first="1", last=HUGE)
    assert answer.status_code == 400
    assert client.get("/view/pending", headers=GET).status_code == 200  # the session goes on


def test_a_huge_content_length_is_too_big(web_chat):
    _, _, app = web_chat
    code, taken, _ = raw(app, {**GOOD, "Content-Length": HUGE}, [b"x"])
    assert (code, taken) == (413, 0)


# --- should 3: the parse is linear


@pytest.mark.parametrize(
    ("args", "parsed"),
    [
        ("notes.md", ("notes.md", None, 0, 0, False)),
        ("notes.md check", ("notes.md", None, 0, 0, True)),
        ("trail notes.md lines 2-30", ("trail notes.md", "lines", 2, 30, False)),
        ("a  b.pdf pages 1-2 check", ("a  b.pdf", "pages", 1, 2, True)),
        ("notes.md lines 1-2x", ("notes.md lines 1-2x", None, 0, 0, False)),
        ("lines 1-2", ("lines 1-2", None, 0, 0, False)),
        ("check", ("check", None, 0, 0, False)),
    ],
)
def test_attach_args(args, parsed):
    assert attach_args(args) == parsed


def test_a_megabyte_of_whitespace_parses_at_once():
    """The old pattern backtracked: 40,000 spaces froze every thread for 28 s."""
    started = time.monotonic()
    attach_args("x" + " " * (1024 * 1024) + "y check")
    attach_args("x" + " " * (1024 * 1024) + "lines 1-")
    assert time.monotonic() - started < 2


# --- should 4: --attach reads exactly the file named


@pytest.mark.parametrize("name", ["notes check", "notes pages 1-2", "notes lines 1-1 check"])
def test_dash_dash_attach_takes_the_named_file_whole(fetched_root, chat_tokenizer, tmp_path, name):
    named = tmp_path / name
    named.write_text("The file named.\n", encoding="utf-8")
    (tmp_path / "notes").write_text("A different made-up file.\n", encoding="utf-8")
    loop, _, _ = chat_loop(fetched_root, chat_tokenizer, attach=read_files([str(named)]))
    loop.start()
    (record,) = loop.session.conversation.pending
    assert (record["name"], record["how"], record["text"]) == (name, "whole", "The file named.\n")
