"""Branches: /regen, /edit, /undo, plus marks and notes, with the fake backend."""

from __future__ import annotations

import json
import threading

import pytest

from clarity_chat import cli
from clarity_chat.backends import BackendError
from clarity_chat.cli import ChatLoop
from clarity_chat.logs import SCHEMA_VERSION, read_records
from clarity_chat.profiles import SEED_MAX, ConfigError
from clarity_chat.session import Session, annotate_log, marks_and_notes
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


def records(session, kind=None):
    return [
        r for r in read_records(session.conversation.log.path) if kind is None or r["type"] == kind
    ]


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


def test_live_marks_and_notes_say_so(make_session):
    session, _ = make_session()
    session.say("a")
    assert session.annotate("mark", "landed")[0]["origin"] == "live"
    assert session.annotate("note", "made-up note")[0]["origin"] == "live"


def test_a_mark_without_a_label_is_interesting(make_session):
    session, _ = make_session()
    session.say("a")
    assert session.annotate("mark", "")[0]["label"] == "interesting"


def current(session):
    return marks_and_notes(records(session))[session.conversation.head]


def test_unmark_removes_the_named_mark_or_the_only_one(make_session):
    session, _ = make_session()
    session.say("a")
    session.annotate("mark", "landed")
    session.annotate("mark", "dropped")
    with pytest.raises(ConfigError, match="has marks dropped, landed"):
        session.annotate("unmark", "")
    with pytest.raises(ConfigError, match="no mark other"):
        session.annotate("unmark", "other")
    (unmark,) = session.annotate("unmark", "landed")
    assert unmark["mark_id"] == records(session, "mark")[0]["mark_id"]
    assert [m["label"] for m in current(session)["marks"]] == ["dropped"]
    session.annotate("unmark", "")
    assert current(session)["marks"] == []


def test_a_new_note_replaces_the_current_one(make_session):
    session, _ = make_session()
    session.say("a")
    (first,) = session.annotate("note", "made-up first note")
    (second,) = session.annotate("note", "made-up second note")
    assert first["replaces"] is None and second["replaces"] == first["note_id"]
    assert [n["text"] for n in current(session)["notes"]] == ["made-up second note"]


def test_marks_and_notes_from_before_ids_still_count():
    old = {"conversation_id": "c", "call_id": "call-1", "turn_id": "turn-1"}
    rows = [
        {"type": "mark", **old, "label": "landed"},
        {"type": "mark", **old, "label": "dropped"},
        {"type": "note", **old, "text": "made-up old note"},
        {"type": "note", **old, "text": "made-up older note"},
        {"type": "unmark", **old, "mark_id": None, "label": "landed"},
        {"type": "note", **old, "note_id": "note-1", "text": "made-up new", "replaces": None},
    ]
    view = marks_and_notes(rows)["turn-1"]
    assert [m["label"] for m in view["marks"]] == ["dropped"]
    assert len(view["notes"]) == 3  # ID-less notes all show


def test_the_chat_says_when_a_note_is_replaced(make_session):
    session, _ = make_session()
    output: list[str] = []
    chat = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    chat.handle("a")
    chat.handle("/note made-up note")
    chat.handle("/note made-up better note")
    chat.handle("/mark")
    chat.handle("/unmark")
    text = "".join(output)
    assert "Noted call-" in text and "Replaced the note on call-" in text
    assert text.rstrip().endswith("interesting.")


# --- review-time marks and notes (clarity mark / clarity note) ----------------------------


def test_review_marks_and_notes_point_at_an_earlier_call(make_session, monkeypatch, capsys):
    session, _ = make_session([" one", " two"])
    first = session.say("a").record
    session.say("b")
    log = session.conversation.log.path
    before = log.read_bytes()
    # No session, backend or server: building one would fail the test.
    monkeypatch.setattr("clarity_chat.session.Session.__init__", None)
    assert cli.main(["mark", str(log), first["call_id"], "dropped"]) == 0
    assert cli.main(["note", str(log), first["call_id"], "made-up review note"]) == 0
    assert f"Marked {first['call_id']} dropped ({first['turn_id']})" in capsys.readouterr().out
    after = log.read_bytes()
    assert after.startswith(before)  # appended, never rewritten
    mark, note = [json.loads(line) for line in after[len(before) :].splitlines()]
    assert mark == {
        "schema_version": SCHEMA_VERSION,
        "time": mark["time"],
        "type": "mark",
        "conversation_id": first["conversation_id"],
        "call_id": first["call_id"],
        "turn_id": first["turn_id"],
        "label": "dropped",
        "origin": "review",
        "mark_id": mark["mark_id"],
    }
    assert mark["mark_id"].startswith("mark-")
    assert note["text"] == "made-up review note" and note["origin"] == "review"


def test_review_marks_on_a_cancelled_call(make_session):
    session, backend = make_session()
    backend.cancel_after = 1
    call = session.say("a").record
    assert call["stop_type"] == "cancelled"
    (record,) = annotate_log(session.conversation.log.path, "mark", call["call_id"], "cut")
    assert record["turn_id"] == call["turn_id"]


@pytest.mark.parametrize(
    ("call_id", "label", "message"),
    [
        ("call-nope", "x", "has no call call-nope"),
        (None, "--remove", "This reply has no mark"),
        ("failed", "x", "failed, so it has no reply to mark"),
    ],
)
def test_review_mark_errors(make_session, capsys, call_id, label, message):
    session, backend = make_session([" one"])
    ok = session.say("a").record
    if call_id == "failed":
        backend.fail_with = "server gone"
        with pytest.raises(BackendError):
            session.say("b")
        call_id = records(session, "call")[-1]["call_id"]
    log = session.conversation.log.path
    before = log.read_bytes()
    assert cli.main(["mark", str(log), call_id or ok["call_id"], label]) == 1
    assert message in capsys.readouterr().err
    assert log.read_bytes() == before


def test_review_unmark(make_session, capsys):
    session, _ = make_session()
    call = session.say("a").record
    log = str(session.conversation.log.path)
    assert cli.main(["mark", log, call["call_id"]]) == 0
    assert cli.main(["mark", "--remove", log, call["call_id"]]) == 0
    assert f"Unmarked {call['call_id']} interesting" in capsys.readouterr().out
    assert current(session)["marks"] == []
    assert records(session, "unmark")[0]["origin"] == "review"


def test_review_mark_needs_the_log_file(tmp_path, capsys):
    assert cli.main(["mark", str(tmp_path / "nope.jsonl"), "call-x", "x"]) == 1
    assert "not found" in capsys.readouterr().err


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


def test_regen_n_shows_a_header_and_a_status_for_each_reply(chat):
    loop, output, _, _, _ = chat
    loop.handle("Hi")
    output.clear()
    loop.handle("/regen 2")
    text = "".join(output)
    assert "— reply 1 of 2, seed 43 —\n two" in text
    assert "— reply 2 of 2, seed 44 —\n three" in text
    assert "reply 1: [" in text and "seed 43 (regen)" in text
    assert "reply 2: [" in text and "seed 44 (regen)" in text
    assert text.endswith("/branches lists them.\n")


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
    assert "Usage: /note [CALL_ID] TEXT" in text
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


# --- final review fixes -------------------------------------------------------------------


def test_edit_text_is_sent_exactly_as_typed(chat, tiny_tokenizer):
    loop, _, backend, _, _ = chat
    loop.handle("Hi")
    loop.handle("/edit  two spaces, trailing space ")
    expected = "User:  two spaces, trailing space \nAssistant:"
    assert sent_text(backend, tiny_tokenizer).endswith(expected)
    loop.handle("/edit\nnew line first")
    assert sent_text(backend, tiny_tokenizer).endswith("User: new line first\nAssistant:")


def test_a_failed_switch_restarts_the_current_model(make_session, fetched_root):
    from tests.conftest import MODELS, fake_fetch

    (fetched_root / "models.toml").write_text(
        MODELS + "\n" + MODELS.replace("[tiny]", "[tiny2]"), encoding="utf-8"
    )
    fake_fetch(fetched_root, "tiny2")
    (fetched_root / "profiles" / "other.toml").write_text(
        TEXT_PROFILE.replace('model = "tiny"', 'model = "tiny2"'), encoding="utf-8"
    )
    session, backend = make_session([" one", " two"])
    first = session.conversation
    start = backend.start

    def fail_for_tiny2(model, weights, ctx_size):
        if model.name == "tiny2":
            backend.started.append("broken")  # as if the old server had been stopped
            raise ConfigError("tiny2 won't start")
        return start(model, weights, ctx_size)

    backend.start = fail_for_tiny2
    with pytest.raises(ConfigError, match="won't start"):
        session.new_conversation("other")
    assert session.conversation is first
    assert backend.started[-1] == "tiny"  # restarted for the current conversation
    assert session.say("still here").record["output_text"] == " one"


def test_when_the_restart_fails_too_no_conversation_runs(make_session):
    session, backend = make_session()

    def always_fail(model, weights, ctx_size):
        raise ConfigError("no GPU")

    backend.start = always_fail
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.handle("/new")
    assert session.conversation is None
    assert "failed too" in "".join(output)
    output.clear()
    loop.handle("/prompt")
    loop.handle("hello")
    assert "".join(output).count("Start one with /profile NAME") == 2


def test_settings_a_backend_cannot_apply_are_flagged_and_warned(make_session):
    session, backend = make_session([" one"])
    backend.not_applied = [
        {"name": "presence_penalty", "value": 0.5, "reason": "none", "affects_output": True},
        {"name": "repeat_last_n", "value": 64, "reason": "none", "affects_output": False},
    ]
    outcome = session.say("Hi")
    assert "not_applied" in outcome.record["flags"]
    assert any("['presence_penalty']" in w for w in outcome.warnings)


def test_a_review_mark_after_a_torn_line_says_so(make_session, monkeypatch, capsys):
    session, _ = make_session([" one"])
    call = session.say("a").record
    log = session.conversation.log.path
    with log.open("ab") as f:
        f.write(b'{"type": "no')
    monkeypatch.setattr("clarity_chat.session.Session.__init__", None)
    assert cli.main(["mark", str(log), call["call_id"], "landed"]) == 0
    assert "has no line end, so it was cut off" in capsys.readouterr().err
    kinds = [r["type"] for r in read_records(log)]
    assert kinds[-2:] == ["torn_line", "mark"]


# --- head records, /branches, /switch -----------------------------------------------------


def test_undo_and_failed_calls_log_where_head_went(make_session):
    session, backend = make_session([" one", " two"])
    first = session.say("a")
    session.say("b")
    session.undo()
    backend.fail_with = "server gone"
    with pytest.raises(BackendError):
        session.say("c")
    with pytest.raises(BackendError):
        session.regen()
    heads = [(h["turn_id"], h["reason"]) for h in records(session, "head")]
    assert heads == [(first.turn.turn_id, r) for r in ("undo", "failed", "failed")]


def test_a_new_reply_writes_no_head_record(make_session):
    session, _ = make_session([" one"])
    session.say("a")
    session.regen()
    assert records(session, "head") == []


def test_switch_goes_to_the_newest_reply_on_a_branch(make_session):
    session, _ = make_session([" one", " two", " three", " four"])
    first = session.say("a")
    session.say("b")
    regen = session.regen()  # " three"
    session.say("c")  # " four", after the regenerated reply
    conversation = session.conversation
    head, render = session.switch(regen.turn.turn_id)  # the branch continues past it
    assert render is None  # checked only in a resumed conversation
    assert head.text == " four"
    second = records(session, "turn")[3]  # the reply " two"
    assert session.switch(second["turn_id"])[0].text == " two"
    assert conversation.last_call["call_id"] == second["call_id"]
    assert records(session, "head")[-1] == {**records(session, "head")[-1], "reason": "switch"}
    assert session.switch(first.turn.parent_id)[0].text == " four"  # the newest under "a"
    with pytest.raises(ConfigError, match="no turn turn-nope"):
        session.switch("turn-nope")


def test_a_failed_message_is_not_a_branch_to_switch_to(make_session):
    session, backend = make_session([" one"])
    session.say("a")
    backend.fail_with = "server gone"
    with pytest.raises(BackendError):
        session.say("b")
    dead = records(session, "turn")[-1]
    assert dead["text"] == "b"
    with pytest.raises(ConfigError, match="has no reply"):
        session.switch(dead["turn_id"])
    assert session.conversation.forks() == []


def test_forks_list_regens_edits_and_what_undo_left(make_session):
    session, _ = make_session([" one", " two", " three"])
    first = session.say("a")
    session.regen()
    session.edit("a2")
    forks = session.conversation.forks()
    assert [len(f["options"]) for f in forks] == [2]  # "a" and "a2" at the start
    assert [o["current"] for o in forks[0]["options"]] == [False, True]
    session.switch(first.turn.turn_id)
    forks = session.conversation.forks()
    assert [len(f["options"]) for f in forks] == [2, 2]  # and the two replies to "a"
    session.undo()  # back to the start: both messages are branches to go back to
    (fork,) = session.conversation.forks()
    assert fork["at"] is None and not any(o["current"] for o in fork["options"])


def test_the_terminal_lists_branches_and_switches(make_session):
    session, _ = make_session([" one", " two"])
    output: list[str] = []
    chat = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    chat.handle("/branches")
    assert "no branches yet" in "".join(output)
    chat.handle("a")
    chat.handle("/regen")
    output.clear()
    chat.handle("/branches")
    text = "".join(output)
    assert "Your message 'a':" in text and "reply 2 of 2" in text
    first_reply = records(session, "turn")[1]["turn_id"]
    chat.handle(f"/switch {first_reply}")
    assert "Switched. The next message continues after: ' one'" in "".join(output)


def test_a_failed_edit_or_regen_logs_head_back_on_the_reply(make_session):
    session, backend = make_session([" one"])
    first = session.say("a")
    backend.fail_with = "server gone"
    for attempt in (lambda: session.edit("a2"), session.regen):
        with pytest.raises(BackendError):
            attempt()
        assert records(session, "head")[-1]["turn_id"] == first.turn.turn_id
        assert session.conversation.last_call["call_id"] == first.record["call_id"]


def test_an_early_cancel_logs_where_head_went(make_session):
    from clarity_chat.session import CancelledBeforeReply

    session, backend = make_session([" one", " two"])
    first = session.say("a")
    cancel = threading.Event()
    backend.on_call = cancel.set
    with pytest.raises(CancelledBeforeReply):
        session.say("b", cancel=cancel)
    assert records(session, "head")[-1] == {
        **records(session, "head")[-1],
        "turn_id": first.turn.turn_id,
        "reason": "failed",
    }


def test_undo_to_the_start_logs_a_null_head(make_session):
    session, _ = make_session([" one", " two"])
    session.say("a")
    session.undo()
    assert records(session, "head")[-1]["turn_id"] is None
    assert session.say("b").turn.parent_id == records(session, "turn")[-2]["turn_id"]


def test_unmark_of_old_duplicate_marks_writes_one_record(make_session):
    session, _ = make_session()
    reply = session.say("a").turn
    log = session.conversation.log
    for _ in range(2):  # pre-0.2 marks: no mark_id
        log.append(
            {
                "type": "mark",
                "conversation_id": "c",
                "call_id": reply.call_id,
                "turn_id": reply.turn_id,
                "label": "landed",
                "origin": "live",
            }
        )
    (unmark,) = session.annotate("unmark", "")
    assert unmark["mark_id"] is None and current(session)["marks"] == []


def test_forks_carry_the_turn_they_branch_at(make_session):
    session, _ = make_session([" one", " two"])
    session.say("a")
    session.regen()
    (fork,) = session.conversation.forks()
    assert fork["at_role"] == "user" and fork["at_text"] == "a"


def test_history_is_the_current_branch_with_marks_and_notes(make_session):
    session, _ = make_session([" one", " two", " three"])
    session.say("a")
    session.regen()  # " two" replaces " one" on this branch
    session.annotate("mark", "landed")
    session.annotate("note", "made-up note")
    session.annotate("note", "made-up better note")
    session.say("b")
    turns = session.history()
    assert [(t["role"], t["text"]) for t in turns] == [
        ("user", "a"),
        ("assistant", " two"),
        ("user", "b"),
        ("assistant", " three"),
    ]
    assert [m["label"] for m in turns[1]["marks"]] == ["landed"]
    assert [n["text"] for n in turns[1]["notes"]] == ["made-up better note"]
    assert turns[1]["regen_of"] == records(session, "call")[0]["call_id"]
    output: list[str] = []
    ChatLoop(session, "text", read=lambda d: "", write=output.append).handle("/history")
    assert "model: two\n  marked: landed\n  note: made-up better note\nyou: b" in "".join(output)


def test_history_and_marks_parse_only_the_records_they_need(make_session, monkeypatch):
    from clarity_chat import session as session_module

    session, _ = make_session([" one"])
    session.say("a")
    log = session.conversation.log.path
    parsed: list[str] = []

    def spy(path, parse=None):
        for record in read_records(path, parse):
            parsed.append(record["type"])
            yield record

    monkeypatch.setattr(session_module, "read_records", spy)
    session.annotate("mark", "landed")
    session.annotate("note", "made-up note")
    turns = session.history()
    annotate_log(log, "unmark", records(session, "call")[0]["call_id"], "")
    # Call records are the bulk of a long log; reading marks never parses them, except the
    # one call `clarity mark` looks up.
    assert parsed.count("call") == 1 and "turn" not in parsed
    assert [m["label"] for m in turns[1]["marks"]] == ["landed"]
    assert [n["text"] for n in turns[1]["notes"]] == ["made-up note"]
    assert marks_and_notes(read_records(log))[turns[1]["turn_id"]]["marks"] == []
