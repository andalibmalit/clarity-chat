"""Branches: /regen, /edit, /undo, plus marks and notes, with the fake backend."""

from __future__ import annotations

import pytest

from clarity_chat.cli import ChatLoop
from clarity_chat.logs import read_records
from clarity_chat.profiles import SEED_MAX, ConfigError
from clarity_chat.session import Session
from tests.conftest import TEXT_PROFILE
from tests.fakes import FakeBackend


@pytest.fixture
def make_session(fetched_root, tiny_tokenizer):
    def make(replies=None, profile="text"):
        backend = FakeBackend(replies, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        session.new_conversation(profile)
        return session, backend

    return make


def records(session, kind):
    return [r for r in read_records(session.conversation.log.path) if r["type"] == kind]


def sent_text(backend, tiny_tokenizer, index=-1):
    return tiny_tokenizer.decode(backend.calls[index]["token_ids"])


# --- /regen -------------------------------------------------------------------------------


def test_regen_adds_sibling_replies_with_seed_plus_k(make_session, tiny_tokenizer):
    session, backend = make_session([" one", " two", " three"])
    first = session.say("Hi")
    second = session.regen()
    third = session.regen()
    assert [c["settings"]["seed"] for c in backend.calls] == [42, 43, 44]
    assert second.record["settings"]["seed"] == {"value": 43, "origin": "regen"}
    assert second.record["regen_of"] == third.record["regen_of"] == first.record["call_id"]
    assert first.turn.parent_id == second.turn.parent_id == third.turn.parent_id
    assert backend.calls[0]["token_ids"] == backend.calls[2]["token_ids"]  # same prompt
    assert session.conversation.head == third.turn.turn_id
    regen_turns = [t for t in records(session, "turn") if t.get("regen_of")]
    assert len(regen_turns) == 2


def test_the_next_message_continues_from_the_regenerated_reply(make_session, tiny_tokenizer):
    session, backend = make_session([" one", " two", " three"])
    session.say("Hi")
    session.regen()
    session.say("More")
    expected = "User: Hi\nAssistant: two\nUser: More\nAssistant:"
    assert sent_text(backend, tiny_tokenizer).endswith(expected)


def test_regen_needs_a_reply(make_session):
    session, _ = make_session()
    with pytest.raises(ConfigError, match="No reply to regenerate"):
        session.regen()


def test_a_failed_regen_stays_on_the_reply(make_session):
    session, backend = make_session([" one"])
    first = session.say("Hi")
    backend.fail_with = "server gone"
    with pytest.raises(Exception, match="server gone"):
        session.regen()
    assert session.conversation.head == first.turn.turn_id


def test_regen_refuses_a_seed_past_the_top(make_session, fetched_root):
    (fetched_root / "profiles" / "top.toml").write_text(
        TEXT_PROFILE.replace("seed = 42", f"seed = {SEED_MAX}"), encoding="utf-8"
    )
    session, _ = make_session(profile="top")
    session.say("Hi")
    with pytest.raises(ConfigError, match="above"):
        session.regen()


# --- /edit --------------------------------------------------------------------------------


def test_edit_branches_from_the_original_messages_parent(make_session, tiny_tokenizer):
    session, backend = make_session([" one", " two", " three"])
    session.say("First")
    original = session.say("Secnd")
    edited = session.edit("Second")
    assert sent_text(backend, tiny_tokenizer).endswith(
        "User: First\nAssistant: one\nUser: Second\nAssistant:"
    )
    conversation = session.conversation
    user = conversation.turns[edited.turn.parent_id]
    assert user.origin == "edit"
    assert user.parent_id == conversation.turns[original.turn.parent_id].parent_id
    edit_record = next(t for t in records(session, "turn") if t["origin"] == "edit")
    assert edit_record["edit_of"] == original.turn.parent_id
    assert backend.calls[-1]["settings"]["seed"] == 42  # a new message, the profile's seed


def test_regen_after_edit_starts_from_the_edited_messages_first_reply(make_session):
    session, backend = make_session([" one", " two", " three"])
    session.say("Hi")
    session.edit("Hello")
    session.regen()
    assert [c["settings"]["seed"] for c in backend.calls] == [42, 42, 43]


# --- /undo --------------------------------------------------------------------------------


def test_undo_moves_back_and_the_next_message_branches(make_session, tiny_tokenizer):
    session, backend = make_session([" one", " two", " three"])
    first = session.say("a")
    session.say("b")
    assert session.undo() == first.turn
    third = session.say("c")
    expected = "User: a\nAssistant: one\nUser: c\nAssistant:"
    assert sent_text(backend, tiny_tokenizer).endswith(expected)
    assert session.conversation.turns[third.turn.parent_id].parent_id == first.turn.turn_id


def test_undo_to_the_start_and_then_nothing_left(make_session):
    session, _ = make_session()
    session.say("a")
    assert session.undo() is None
    with pytest.raises(ConfigError, match="No reply to undo"):
        session.undo()


# --- marks and notes ----------------------------------------------------------------------


def test_marks_and_notes_point_at_the_current_reply(make_session):
    session, _ = make_session([" one", " two"])
    session.say("a")
    second = session.say("b")
    session.annotate("mark", "landed")
    session.annotate("note", "made-up note text")
    (mark,) = records(session, "mark")
    (note,) = records(session, "note")
    assert mark["call_id"] == note["call_id"] == second.record["call_id"]
    assert mark["label"] == "landed" and note["text"] == "made-up note text"


def test_a_mark_needs_a_reply(make_session):
    session, _ = make_session()
    with pytest.raises(ConfigError, match="No reply to mark"):
        session.annotate("mark", "x")


# --- the commands -------------------------------------------------------------------------


@pytest.fixture
def chat(make_session):
    session, backend = make_session([" one", " two", " three", " four"])
    output: list[str] = []
    inputs: list[str] = []
    defaults: list[str] = []

    def read(default):
        defaults.append(default)
        return inputs.pop(0) if inputs else ""

    loop = ChatLoop(session, "text", read=read, write=output.append)
    return loop, output, backend, inputs, defaults


def test_regen_command_shows_the_regen_seed(chat):
    loop, output, _, _, _ = chat
    loop.handle("Hi")
    output.clear()
    loop.handle("/regen")
    assert "seed 43 (regen)" in "".join(output)


def test_edit_command_prefills_the_last_message(chat, tiny_tokenizer):
    loop, output, backend, inputs, defaults = chat
    loop.handle("Hi thre")
    inputs.append("Hi there")
    loop.handle("/edit")
    assert defaults == ["Hi thre"]
    assert sent_text(backend, tiny_tokenizer).endswith("User: Hi there\nAssistant:")


def test_edit_command_with_text_and_cancel(chat, tiny_tokenizer):
    loop, output, backend, _, _ = chat
    loop.handle("Hi")
    loop.handle("/edit Hello")
    assert sent_text(backend, tiny_tokenizer).endswith("User: Hello\nAssistant:")
    output.clear()
    loop.handle("/edit")  # the reader returns nothing
    assert "Edit cancelled" in "".join(output)
    assert len(backend.calls) == 2


def test_undo_mark_and_note_commands(chat):
    loop, output, _, _, _ = chat
    loop.handle("/undo")
    assert "No reply to undo" in "".join(output)
    loop.handle("Hi")
    loop.handle("/mark landed")
    loop.handle("/note")
    loop.handle("/undo")
    text = "".join(output)
    assert "Marked call-" in text
    assert "Usage: /note TEXT" in text
    assert "continues at the start" in text


def test_profile_command_switches_profile(chat, fetched_root):
    loop, output, _, _, _ = chat
    (fetched_root / "profiles" / "other.toml").write_text(TEXT_PROFILE, encoding="utf-8")
    loop.handle("/profile other")
    assert loop.session.conversation.profile.name == "other"
    assert "with profile other" in "".join(output)
    loop.handle("/profile")
    assert "Usage: /profile NAME" in "".join(output)


# --- review fixes -------------------------------------------------------------------------


def test_a_failed_edit_stays_on_the_reply(make_session):
    session, backend = make_session([" one", " two"])
    session.say("a")
    second = session.say("b")
    backend.fail_with = "server gone"
    with pytest.raises(Exception, match="server gone"):
        session.edit("b2")
    assert session.conversation.head == second.turn.turn_id
    assert session.conversation.last_call["call_id"] == second.record["call_id"]


def test_prompt_follows_the_head_after_undo(make_session):
    session, _ = make_session([" one", " two"])
    first = session.say("a")
    session.say("b")
    session.undo()
    assert session.conversation.last_call["call_id"] == first.record["call_id"]
    session.undo()
    assert session.conversation.last_call is None


def test_note_command_says_noted(chat):
    loop, output, _, _, _ = chat
    loop.handle("Hi")
    loop.handle("/note made-up note")
    assert "Noted call-" in "".join(output)
