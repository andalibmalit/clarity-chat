"""The thread's view (v0.3.0 item C2), offline: messages, your message's counter and Edit,
the toolbars, marks and notes, the facts line, and the failed, refused and stopped states.

Values on screen are checked against what the command table returns (data walk, V3Chat and
V3Writing).
"""

from __future__ import annotations

import html
import re
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command
from tests.test_web_shell import get_view, start_held

# The counter's name is visually hidden text, not an aria-label on a span (Checkpoint 3).
COUNTER = (
    '<span class="n"><span aria-hidden="true">{k} / {n}</span>'
    '<span class="vh">Version {k} of {n}</span></span>'
)


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    """A chat on the text profile, served in process: (client, session)."""
    backend = FakeBackend([" Hello!", " Again.", " More."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client, session


def result(client, name, args=""):
    answer = command(client, name, args).json()
    assert "result" in answer, answer
    return answer["result"]


def thread(client) -> str:
    return html.unescape(get_view(client, "thread"))


def test_messages_are_plain_text_in_order_with_their_calls(chat):
    client, _ = chat
    result(client, "say", "Hi *there*\n  two lines")
    turns = result(client, "state")["turns"]
    fragment = thread(client)
    texts = re.findall(r'<div class="msg">(.*?)</div>', fragment, re.S)
    assert texts == [t["text"] for t in turns] == ["Hi *there*\n  two lines", " Hello!"]
    assert f'data-call="{turns[1]["call_id"]}"' in fragment


def test_your_message_shows_which_version_it_is_and_switches(chat):
    client, _ = chat
    result(client, "say", "Hi")
    result(client, "edit", "Hello")
    forks = result(client, "branches")["forks"]
    options = next(f for f in forks if f["at"] is None)["options"]
    assert [o["text"] for o in options] == ["Hi", "Hello"]
    fragment = thread(client)
    assert COUNTER.format(k=2, n=2) in fragment
    assert f'data-switch="{options[0]["turn_id"]}"' in fragment
    assert 'aria-label="Next version of your message" disabled' in fragment  # the last
    result(client, "switch", options[0]["turn_id"])
    fragment = thread(client)
    assert COUNTER.format(k=1, n=2) in fragment
    assert 'aria-label="Previous version of your message" disabled' in fragment  # the first
    assert f'data-switch="{options[1]["turn_id"]}"' in fragment  # Next goes to version 2


def test_edit_is_on_your_last_message_only_and_holds_its_text(chat):
    client, _ = chat
    result(client, "say", "First")
    result(client, "say", "Second <b>")
    fragment = thread(client)
    assert fragment.count('data-action="edit"') == 1
    assert re.search(r"<textarea[^>]*>Second <b></textarea>", fragment)


def test_the_last_reply_and_an_earlier_one_get_their_toolbars(chat):
    client, _ = chat
    result(client, "say", "Hi")
    fragment = thread(client)
    assert 'aria-label="Actions for this reply"' in fragment
    assert 'data-action="regen"' in fragment and ">Replies<" in fragment
    result(client, "regen")
    fork = result(client, "branches")["forks"][-1]
    assert len(fork["options"]) == 2
    assert f">{len(fork['options'])} replies<" in thread(client)
    result(client, "say", "Again")
    fragment = thread(client)
    assert fragment.count('aria-label="Actions for an earlier reply"') == 1
    assert fragment.count('data-action="regen"') == 1  # only on the last reply


def test_marks_and_the_note_show_in_the_reply_s_popover(chat):
    client, _ = chat
    call_id = result(client, "say", "Hi")["call"]["call_id"]
    assert 'aria-label="Mark this reply"' in thread(client)
    result(client, "note", f"{call_id} Worth a <second> look")
    assert 'aria-label="Not marked, has a note. Open the note"' in thread(client)
    result(client, "mark", call_id)
    result(client, "mark", f"{call_id} odd")
    turn = result(client, "state")["turns"][-1]
    labels = [m["label"] for m in turn["marks"]]
    assert labels == ["interesting", "odd"]
    fragment = thread(client)
    assert f'aria-label="Marked: {", ".join(labels)}. Edit marks and note"' in fragment
    for label in labels:
        assert f'aria-label="Remove the mark {label}" data-unmark="{label}"' in fragment
    assert "Note: “Worth a <second> look”" in fragment and ">Edit note<" in fragment
    assert "Mark as interesting" not in fragment  # it's marked


@pytest.mark.parametrize(
    ("call", "plain", "warn"),
    [
        ({"flags": []}, [], []),
        (
            {"flags": ["length"], "settings": {"max_tokens": {"value": 1024}}},
            ["stopped at the token limit (max_tokens 1024)"],
            [],
        ),
        ({"flags": ["cancelled"]}, ["stopped by you"], []),
        ({"flags": ["empty_reply"]}, ["empty reply (nothing, or only blank space)"], []),
        ({"flags": [], "request": {"steering": {"scale": 0.5}}}, ["steered ×0.5"], []),
        ({"flags": ["special_tokens_in_content"]}, [], ["! text read as a control token"]),
        (
            {"flags": ["incomplete_output_ids"]},
            [],
            ["! some output token IDs are missing, so replay compares text"],
        ),
        ({"flags": ["text_differs_from_ids"]}, [], ["! the text differs from its token IDs"]),
        ({"flags": ["check_failed:no_cache_reuse"]}, [], ["! 1 check failed: no_cache_reuse"]),
        (
            {"flags": ["check_failed:settings_echo", "check_failed:tokens_evaluated"]},
            [],
            ["! 2 checks failed: settings_echo, tokens_evaluated"],
        ),
        (
            {
                "flags": ["not_applied"],
                "not_applied": [
                    {"name": "a", "affects_output": True},
                    {"name": "b", "affects_output": False},
                ],
            },
            [],
            ["! 1 setting not applied"],
        ),
        (
            {"flags": [], "warnings": ["The prompt uses 6970 of 8192 context tokens (85%)."]},
            [],
            ["! The prompt uses 6970 of 8192 context tokens (85%)."],
        ),
    ],
)
def test_the_facts_line_words(call, plain, warn):
    facts = web.facts_line(call, steerable=False)
    assert facts["plain"] == plain
    assert [w["text"] for w in facts["warn"]] == warn


def test_a_steerable_profile_unsteered_says_so_and_each_stop_names_its_next_step():
    assert web.facts_line({"flags": []}, steerable=True)["plain"] == ["not steered"]
    full = web.facts_line({"flags": ["context_full"]}, steerable=False)
    assert full["context_full"] is True and full["plain"] == []
    stopped = web.facts_line({"flags": ["check_failed:server_running"]}, steerable=False)
    assert stopped["restart"] is True
    cached = web.facts_line({"flags": ["check_failed:no_cache_reuse"]}, steerable=False)
    assert cached["restart"] is False  # only a stopped server needs starting again


def test_the_facts_line_is_the_last_reply_s_call(chat):
    client, session = chat
    session.backend.replies = [" " + "word " * 40]  # longer than max_tokens 16
    call = result(client, "say", "Hi")["call"]
    assert "length" in call["flags"]
    max_tokens = call["settings"]["max_tokens"]["value"]
    fragment = thread(client)
    assert f"stopped at the token limit (max_tokens {max_tokens})" in fragment
    session.backend.result_flags = ["check_failed:server_running"]
    result(client, "say", "Again")
    fragment = thread(client)
    assert "! 1 check failed: server_running" in fragment and 'data-action="restart"' in fragment
    assert fragment.count('class="facts"') == 1


def test_a_failed_message_leaves_the_thread_and_is_shown_by_its_call(chat):
    client, session = chat
    session.backend.fail_with = "A made-up failure."
    answer = command(client, "say", "Hi").json()
    assert answer["type"] == "BackendError"
    state = result(client, "state")
    call = result(client, "prompt")["call"]
    assert state["unanswered"]["text"] == "Hi" and call["error"].startswith("BackendError: ")
    fragment = thread(client)
    assert [t["role"] for t in state["turns"]] == []  # your message is back in the box
    assert f'data-failed="{call["call_id"]}"' in fragment
    # The alert holds only the sentence; Details sits beside it (V3Writing E5).
    assert "<p data-alert><strong>! Failed:</strong> A made-up failure.</p>" in fragment
    assert 'data-action="details"' in fragment
    assert "Try again" not in fragment and 'data-action="restart"' not in fragment


def test_when_the_server_stopped_the_failed_line_offers_to_start_it_again(chat):
    client, session = chat
    session.backend.stopped = True
    command(client, "say", "Hi")
    assert result(client, "state")["model"] == "stopped"
    assert 'data-action="restart"' in thread(client)


def test_a_message_cancelled_before_the_reply_started_says_so(chat):
    client, session = chat
    session.backend.cancel_after = 0
    assert command(client, "say", "Hi").json()["type"] == "CancelledBeforeReply"
    fragment = thread(client)
    assert re.search(
        r'role="status" data-failed="call-[0-9a-f]+">Cancelled before the reply started.<', fragment
    )
    assert "Failed" not in fragment


def test_a_message_that_doesnt_fit_shows_the_sentence_whole(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Yes."], n_ctx=160, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        refused = command(client, "say", "x" * 200).json()
        fragment = thread(client)
    assert refused["type"] == "ContextError"
    assert f"<strong>!</strong> {refused['error']}" in fragment and "Failed" not in fragment


def test_while_a_reply_is_written_the_thread_has_no_controls(chat):
    client, session = chat
    result(client, "say", "Hi")
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    worker = threading.Thread(target=lambda: command(client, "say", "Again"))
    worker.start()
    threading.Event().wait(0.4)
    try:
        assert result(client, "state")["running"]["command"] == "say"
        fragment = thread(client)
        answered_during = worker.is_alive()  # it didn't wait for the reply (no queued reads)
    finally:
        worker.join(10)
    assert answered_during
    assert ">Again<" in fragment  # your message is in
    for control in ('data-action="regen"', 'data-action="edit"', 'data-action="marks"'):
        assert control not in fragment


def test_starting_the_model_again_keeps_the_thread_and_says_so_at_its_end(chat):
    client, session = chat
    result(client, "say", "Hi")
    chat_id = result(client, "state")["conversation_id"]
    gate, worker = start_held(client, session, chat_id, how="resume")
    try:
        fragment = thread(client)
    finally:
        gate.set()
        worker.join(10)
    assert fragment.index(" Hello!") < fragment.index("Starting text…")


def test_a_reply_without_a_call_id_is_drawn_without_controls(chat):
    """Checkpoint 3: a log another program wrote can have a reply with no call ID. The view
    draws it, asks /prompt for nothing, and gives it no controls that would send /mark None."""
    client, session = chat
    result(client, "say", "Hi")
    conversation = session.conversation
    conversation.turns[conversation.head].call_id = None
    response = client.get("/view/thread", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    assert not command(client, "state").json().get("stopped")  # the session goes on
    fragment = html.unescape(response.text)
    assert " Hello!" in fragment and 'data-call="None"' not in fragment
    assert 'data-action="marks"' not in fragment and 'data-action="regen"' not in fragment


DRAWN_ACTIONS = re.compile(r'data-action="([\w-]+)"')


@pytest.mark.xfail(
    strict=True,
    reason="Chats, Appearance, Settings, the replies pop-out and Details are handled by C3-C5",
)
def test_every_drawn_action_has_a_handler():
    from pathlib import Path

    folder = Path(web.__file__).parent
    drawn = set()
    for path in [*(folder / "web_templates").glob("*.html"), folder / "web_static" / "index.html"]:
        drawn |= set(DRAWN_ACTIONS.findall(path.read_text(encoding="utf-8")))
    code = (folder / "web_static" / "app.js").read_text(encoding="utf-8")
    thread = code[code.index("const THREAD = {") : code.index("};", code.index("const THREAD = {"))]
    handled = (
        set(re.findall(r'action === "([\w-]+)"', code))
        | set(re.findall(r'^  "?([\w-]+)"?(?::|\()', thread, re.M))
        | set(re.findall(r"clarity\.on\.([\w]+)\s*=", code))
    )
    assert drawn - handled == set()


def test_a_failed_line_needs_your_message_back_in_the_box():
    """A failed call whose message isn't unanswered (a later undo, say) draws no line."""
    call = {"error": "BackendError: A made-up failure.", "call_id": "call-0123456789ab"}
    assert web.failed_line(call, {"unanswered": None}) is None
    line = web.failed_line(call, {"unanswered": {"turn_id": "turn-1", "text": "Hi"}})
    assert line == {
        "kind": "BackendError",
        "message": "A made-up failure.",
        "call_id": call["call_id"],
    }


def test_a_new_chat_starting_from_a_chat_doesnt_draw_the_old_one(chat):
    client, session = chat
    profile = (session.root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (session.root / "profiles" / "second.toml").write_text(profile, encoding="utf-8")
    result(client, "say", "Distinctive old words")
    gate, worker = start_held(client, session, "second")
    try:
        fragment = thread(client)
    finally:
        gate.set()
        worker.join(10)
    assert "Starting second…" in fragment and "Distinctive old words" not in fragment


def test_new_chat_names_the_profile_it_starts(chat):
    client, session = chat
    gate, worker = start_held(client, session, "", how="new")
    try:
        fragment = thread(client)
    finally:
        gate.set()
        worker.join(10)
    assert "Starting text…" in fragment
