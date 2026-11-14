"""The Branches panel (v0.3.0 item C4d), offline: every branch end and every marked reply,
from /branches all (data walk, V3Branches)."""

from __future__ import annotations

import html
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    backend = FakeBackend([], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client, session


def result(client, name, args=""):
    answer = command(client, name, args).json()
    assert "result" in answer, answer
    return answer["result"]


def panel(client) -> str:
    response = client.get("/view/branches", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    return html.unescape(response.text)


def test_every_branch_end_newest_first_says_where_it_came_from(chat):
    client, session = chat
    session.backend.replies = [" One.", " Two.", " Three.", " Four."]
    result(client, "say", "Hi")
    result(client, "regen", "2")
    result(client, "edit", "Hello")
    result(client, "mark")
    ends = result(client, "branches", "all")["ends"]
    fragment = panel(client)
    assert f"{len(ends)} branch ends, newest first" in fragment
    positions = [fragment.index(f'data-turn="{end["turn_id"]}"') for end in ends]
    assert positions == sorted(positions)  # in /branches all's order, newest first
    labels = {
        ("edit", True): "From your edited message · seed {seed}",
        ("edit", False): "From your edited message · seed {seed}",
        ("reply", True): "From reply {k} of {n} · seed {seed}",
        ("reply", False): "Reply {k} of {n} · seed {seed}, not continued",
    }
    for end in ends:
        source = end["from"]
        label = labels[(source["kind"], end["continued"])].format(**source)
        assert label in fragment
        where = f"{end['turns']} messages" + (f" · {end['marked']} marked" if end["marked"] else "")
        assert where in fragment
        if end["current"]:
            assert "You are here" in fragment
        else:
            assert f'data-switch="{end["turn_id"]}"' in fragment
    assert "Nothing is deleted: trying again, or going back, adds a branch." in fragment


def test_marked_replies_with_their_labels_notes_and_branch(chat):
    client, session = chat
    session.backend.replies = [" One.", " Two."]
    first = result(client, "say", "Hi")["call"]["call_id"]
    result(client, "regen")
    result(client, "mark", f"{first} landed")
    result(client, "note", f"{first} Loose, but mine.")
    marked = result(client, "branches", "all")["marked"]
    assert len(marked) == 1
    fragment = panel(client)
    assert "1 marked reply, newest first" in fragment  # Checkpoint 4, finding 15: singular
    assert "Marked: landed" in fragment and "Note: Loose, but mine." in fragment
    source = marked[0]["from"]
    assert (
        f"on the branch from reply {source['k']} of {source['n']} · seed {source['seed']}"
        in fragment
    )
    assert f'data-reveal="{marked[0]["turn_id"]}"' in fragment


def test_with_nothing_marked_the_marked_list_says_how_to_mark(chat):
    client, session = chat
    session.backend.replies = [" One.", " Two."]
    result(client, "say", "Hi")
    result(client, "regen")
    fragment = panel(client)
    empty = "No marked replies in this chat yet. Mark one with the bookmark beside a reply"
    assert f"{empty}, then find it here." in fragment
    assert "Turn “Marked only” off to see all 2 branch ends." in fragment


def test_while_a_reply_is_written_the_panel_waits(chat):
    client, session = chat
    result(client, "say", "Hi")
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    worker = threading.Thread(target=lambda: command(client, "say", "Again"))
    worker.start()
    threading.Event().wait(0.4)
    try:
        fragment = panel(client)
        during = worker.is_alive()  # it didn't wait for the reply
    finally:
        worker.join(10)
    assert during and "Waits until the reply is done." in fragment


# --- Checkpoint 4's findings

HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}'


def test_untrusted_text_in_branches_is_escaped(chat):
    """B5's fixture, for Branches: a reply, a mark's label and a note."""
    client, session = chat
    session.backend.replies = [HOSTILE, " Two."]
    first = result(client, "say", "Hi")["call"]["call_id"]
    result(client, "regen")
    result(client, "mark", f"{first} <b>label</b>")
    result(client, "note", f"{first} {HOSTILE}")
    raw = client.get("/view/branches", headers={**AUTH, **SAME}).text
    assert "<script>" not in raw and "<img" not in raw and "<b>" not in raw
    fragment = html.unescape(raw)
    assert "<b>label</b>" in fragment and f"Note: {HOSTILE}" in fragment
    assert "&lt;script&gt;" in raw  # the reply (cut to 16 characters by max_tokens)


def test_while_a_model_starts_the_panel_says_so(chat):
    """Finding 14: not "Waits until the reply is done." during a start."""
    from tests.test_web_shell import start_held

    client, session = chat
    result(client, "say", "Hi")
    gate, worker = start_held(client, session, "", how="new")
    try:
        fragment = panel(client)
    finally:
        gate.set()
        worker.join(10)
    assert "Waits until the model has started." in fragment
