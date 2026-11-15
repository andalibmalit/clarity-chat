"""The Branches panel (v0.3.0 item C4d), offline: every branch end and every marked reply,
from /branches all (data walk, V3Branches)."""

from __future__ import annotations

import html
import re
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command, wait_until, writing


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
    assert f"{len(ends)} branches, newest first" in fragment
    positions = [fragment.index(f'data-turn="{end["turn_id"]}"') for end in ends]
    assert positions == sorted(positions)  # in /branches all's order, newest first
    labels = {
        ("edit", True): "From your edited message · seed {seed}",
        ("edit", False): "From your edited message · seed {seed}",
        ("reply", True): "From reply {k} of {n} · seed {seed}",
        ("reply", False): "Reply {k} of {n} · seed {seed}, not continued",
    }
    ends_list = fragment.split('<ul class="cards" data-list="ends">')[1].split("</ul>")[0]
    cards = {
        re.search(r'data-turn="([^"]+)"', card).group(1): card  # ty: ignore
        for card in ends_list.split("<li ")[1:]
    }
    for end in ends:  # each card's own words, not somewhere in the panel
        card = cards[end["turn_id"]]
        source = end["from"]
        label = labels[(source["kind"], end["continued"])].format(**source)
        assert f'<span class="lbl">{label}</span>' in card
        where = f"{end['turns']} turns" + (f" · {end['marked']} marked" if end["marked"] else "")
        assert f'<p class="d2">{where}</p>' in card
        if end["current"]:
            assert "You are here" in card and "data-switch" not in card
        else:
            assert f'data-switch="{end["turn_id"]}"' in card and "You are here" not in card
    assert (
        "Nothing is deleted: Regenerate, Edit, or a message sent after going back adds a branch."
        in fragment
    )


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


def card(fragment: str, turn_id: str, marked: bool = False) -> str:
    """One card's opening tag, from the ends list or the marked list."""
    lists = fragment.split('<div data-list="marked"')
    found = re.search(rf'<li class="[^"]*" data-turn="{turn_id}">', lists[1 if marked else 0])
    assert found, turn_id
    return found.group(0)


def test_a_continued_end_a_steered_end_and_the_current_cards(chat):
    """Checkpoint 4's weak-red check: a reply end that went on, a steered one, and the
    current end and current marked card outlined (`bcard cur`)."""
    client, session = chat
    session.backend.replies = [" One.", " Two.", " Three."]
    result(client, "say", "Hi")
    result(client, "regen")
    first = result(client, "branches", "all")["ends"][1]
    result(client, "say", "More")  # reply 2 of 2 goes on
    result(client, "mark")
    session.conversation.calls[first["call_id"]]["settings"]["steering_scale"] = {
        "value": 0.5,
        "origin": "chat",
    }
    found = result(client, "branches", "all")
    current = next(e for e in found["ends"] if e["current"])
    assert current["continued"] and current["from"]["k"] == 2
    fragment = panel(client)
    assert "From reply 2 of 2 · seed" in fragment
    assert f"Reply 1 of 2 · seed {first['seed']}, not continued · steered ×0.5" in fragment
    assert card(fragment, current["turn_id"]) == (
        f'<li class="bcard cur" data-turn="{current["turn_id"]}">'
    )
    assert card(fragment, first["turn_id"]) == f'<li class="bcard" data-turn="{first["turn_id"]}">'
    (mark,) = found["marked"]
    assert mark["current"]
    assert card(fragment, mark["turn_id"], marked=True) == (
        f'<li class="bcard cur" data-turn="{mark["turn_id"]}">'
    )


def test_an_end_with_no_fork_above_it_is_headed_by_its_seed(chat):
    client, session = chat
    session.backend.replies = [" One."]
    result(client, "say", "Hi")
    (end,) = result(client, "branches", "all")["ends"]
    assert end["from"] is None
    fragment = panel(client)
    assert f'<span class="lbl">seed {end["seed"]}</span>' in fragment


def test_with_nothing_marked_the_marked_list_says_how_to_mark(chat):
    client, session = chat
    session.backend.replies = [" One.", " Two."]
    result(client, "say", "Hi")
    result(client, "regen")
    fragment = panel(client)
    empty = "No marked replies in this chat yet. Mark one with the bookmark beside a reply"
    assert f"{empty}, then find it here." in fragment
    assert "Turn “Marked only” off to see all 2 branches." in fragment


def test_while_a_reply_is_written_the_panel_waits(chat):
    client, session = chat
    result(client, "say", "Hi")
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 1.0
    worker = threading.Thread(target=lambda: command(client, "say", "Again"))
    worker.start()
    wait_until(lambda: writing(client))  # the reply is being written
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
