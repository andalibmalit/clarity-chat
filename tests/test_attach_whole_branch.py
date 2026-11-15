"""v0.3.1's whole-branch review (notes/research/reviews/v031-whole-branch.md), the server
side: its three musts, ruling 2 both ways (should 10), the survivors' tests (should 11) and
an empty /say's sentence (could 17). Made-up documents only."""

from __future__ import annotations

import html
import re

import pytest

from clarity_chat import web
from clarity_chat.commands import Commands, check_lines
from clarity_chat.documents import SENTENCES, reading
from clarity_chat.profiles import ConfigError
from clarity_chat.session import ContextError
from tests.test_session_attach import chat_session, doc
from tests.test_web import GET, command
from tests.test_web_attach import NOTES, make, sheet
from tests.test_web_upload import upload

__all__ = ["make"]  # the fixture, used by name

LINES = "".join(f"Made-up line {n}.\n" for n in range(2000)).encode()


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def details(client, call_id: str):
    return client.get("/view/details", params={"call": call_id}, headers=GET)


def attach(client, data=NOTES, name="notes.md", args="") -> None:
    upload_id = upload(client, data, name).json()["upload"]
    assert "error" not in command(client, "attach", f"{upload_id} {args}".strip()).json()


# --- must 1: Details reads a call's attachments from its own prompt, not today's path


def test_a_failed_send_s_details_list_its_attachment(make):
    client, _, backend = make()
    attach(client)
    backend.fail_with = "made-up failure"
    command(client, "say", "What's in it?")
    call = command(client, "prompt").json()["result"]["call"]
    answer = details(client, call["call_id"])
    assert answer.status_code == 200, answer.text
    assert "notes.md · whole" in text_of(answer.text)


@pytest.mark.parametrize("move", ["edit", "undo"])
def test_details_of_a_call_off_the_path_still_lists_its_attachments(make, move):
    client, _, _ = make()
    attach(client)
    command(client, "say", "What's in it?")
    first = command(client, "prompt").json()["result"]["call"]["call_id"]
    command(client, *(("edit", "Hello there") if move == "edit" else ("undo", "")))
    answer = details(client, first)
    assert answer.status_code == 200, answer.text
    assert "notes.md · whole" in text_of(answer.text)


def test_prompt_lists_the_attachments_in_the_call_s_prompt(make):
    client, _, _ = make()
    attach(client)
    command(client, "say", "What's in it?")
    result = command(client, "prompt").json()["result"]
    assert len(result.get("prompt_attachments") or []) == 1
    (entry,) = result["prompt_attachments"]
    assert entry["text"] == "What's in it?"
    assert [a["name"] for a in entry["attachments"]] == ["notes.md"]


def test_without_spans_a_call_before_any_attachment_isn_t_said_to_have_one(make, monkeypatch):
    import clarity_chat.session as session_module

    client, _, _ = make()
    monkeypatch.setattr(session_module, "message_spans", lambda *a: {"unavailable": "made up"})
    command(client, "say", "Hi")
    first = command(client, "prompt").json()["result"]["call"]["call_id"]
    attach(client)
    command(client, "say", "And now?")
    assert "Not listed" not in text_of(details(client, first).text)


# --- must 2: the stop offers removing a waiting attachment only after a send


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


@pytest.mark.parametrize("how", ["regen", "edit"])
def test_after_regen_or_edit_the_stop_offers_no_waiting_attachment(chat, how):
    session, backend = chat
    backend._n_ctx = 4000
    session.attach(doc("Made-up words. " * 100))
    session.say("a")
    backend._n_ctx = 1200
    with pytest.raises(ContextError) as e:
        session.regen() if how == "regen" else session.edit("b")
    assert str(e.value).endswith(SENTENCES["stop_advice"])
    assert session.conversation.pending == []


# --- must 3: the terminal's check lines follow the profile's marker setting


def test_the_terminal_s_excerpt_line_for_a_range_that_covers_the_file(chat, tmp_path):
    session, _ = chat
    path = tmp_path / "notes.md"
    path.write_bytes(NOTES)
    check = Commands(session).run("attach", f"{path} lines 1-3 check", lambda e: None)["check"]
    lines = check_lines(check)
    assert SENTENCES["excerpt"].format(units="lines") in lines
    assert SENTENCES["no_marker"] not in lines


# --- should 10: a setting as typed; every count and range end with separators


def test_range_ends_get_separators_in_the_sheet_the_thread_and_the_chips(make):
    client, _, _ = make(n_ctx=30_000)  # the whole doesn't fit; 1,500 lines do
    upload_id = upload(client, LINES, "long.md").json()["upload"]
    check = command(client, "attach", f"{upload_id} check").json()["result"]["check"]
    start = check["start_that_fits"]["range"][1]
    assert start >= 1000
    page = text_of(sheet(client, upload=upload_id))
    assert f"At most lines 1 to {start:,} fit" in page
    attach(client, LINES, "long.md", "lines 1-1500")
    chips = text_of(client.get("/view/pending", headers=GET).text)
    assert "long.md · lines 1–1,500 ·" in chips
    command(client, "say", "And these?")
    assert "lines 1–1,500 of 2,000" in text_of(client.get("/view/thread", headers=GET).text)


def test_the_terminal_s_ranges_get_separators_and_its_command_stays_bare(chat, tmp_path):
    session, backend = chat
    backend._n_ctx = 30_000
    path = tmp_path / "long.md"
    path.write_bytes(LINES)
    commands = Commands(session)
    check = commands.run("attach", f"{path} lines 1-1500 check", lambda e: None)["check"]
    lines = "\n".join(check_lines(check))
    start = check["start_that_fits"]["range"][1]
    assert f"The start that fits: lines 1 to {start:,}" in lines
    assert "Lines 1 to 1,500:" in lines
    with pytest.raises(ConfigError) as e:
        commands.run("attach", str(path), lambda e: None)
    assert re.search(r"/attach \S+ lines 1-\d+", str(e.value))  # a command to copy: bare


# --- should 11: the survivors


def test_a_check_counts_again_after_another_file_is_attached(chat, tmp_path):
    session, _ = chat
    other = tmp_path / "other.md"
    other.write_bytes(NOTES)
    path = tmp_path / "long.md"
    path.write_bytes(LINES[:3000])
    commands = Commands(session)
    before = commands.run("attach", f"{path} check", lambda e: None)["check"]["free"]
    commands.run("attach", str(other), lambda e: None)
    cost = session.conversation.pending[0]["tokens"]
    after = commands.run("attach", f"{path} check", lambda e: None)["check"]["free"]
    assert after == before - cost


def test_compare_draws_no_bookmark_while_a_command_changes_the_chat(make, monkeypatch):
    client, session, backend = make()
    command(client, "say", "Hi")
    backend.replies = [" Two."]
    command(client, "regen")
    call = command(client, "state").json()["result"]["turns"][-1]["call_id"]
    monkeypatch.setattr(web, "changing", lambda state: {"command": "say", "args": "x"})
    page = client.get("/view/compare", params={"call": call}, headers=GET).text
    assert 'data-action="marks"' not in page


def test_a_noted_unmarked_card_is_labelled_as_the_thread_labels_it(make):
    client, _, backend = make()
    command(client, "say", "Hi")
    backend.replies = [" Two."]
    command(client, "regen")
    replies = command(client, "compare").json()["result"]["replies"]
    command(client, "note", f"{replies[0]['call_id']} made-up note")
    page = html.unescape(
        client.get("/view/compare", params={"call": replies[1]["call_id"]}, headers=GET).text
    )
    assert 'aria-label="Not marked, has a note. Open the note, reply 1 of 2"' in page


def test_the_thread_line_and_details_with_big_numbers_and_a_long_message(make):
    client, _, _ = make(n_ctx=8000)
    attach(client, LINES[:2400], "long.md")
    message = "A made-up question that is much longer than forty characters?"
    command(client, "say", message)
    thread = text_of(client.get("/view/thread", headers=GET).text)
    tokens = command(client, "history").json()["result"]["turns"][-2]["attachments"][0]["tokens"]
    assert tokens >= 1000 and f"· {tokens:,} tokens when attached" in thread
    call = command(client, "prompt").json()["result"]["call"]["call_id"]
    assert f"“{message[:40].rstrip()}…”" in text_of(details(client, call).text)


def test_an_eight_line_text_has_no_more_lines(make):
    client, _, _ = make()
    eight = "".join(f"Made-up {n}.\n" for n in range(8)).encode()
    upload_id = upload(client, eight, "eight.md").json()["upload"]
    assert "more lines" not in text_of(sheet(client, upload=upload_id))


# --- could 17: an empty /say has one sentence, the page's and the terminal's


def test_an_empty_say_says_type_a_message(make):
    client, _, _ = make()
    assert command(client, "say", "").json()["error"] == "Type a message to send."


# --- should 4: a Cancel names its upload, so a read not yet started never starts


def test_a_cancelled_upload_is_refused_before_it_is_read(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    answer = command(client, "cancel", upload_id).json()["result"]
    assert answer["cancelled"] is True and answer.get("upload") == upload_id
    refused = command(client, "attach", f"{upload_id} check").json()
    assert refused.get("error") == "Stopped reading notes.md."
    assert command(client, "state").json()["result"]["reading"] is None


# --- should 6: the page knows the mode without a round trip


def test_state_says_the_chat_s_mode(make):
    client, _, _ = make()
    assert command(client, "state").json()["result"].get("mode") == "chat"
    command(client, "profile", "text")
    assert command(client, "state").json()["result"].get("mode") == "text"
