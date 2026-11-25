"""A3 (v0.3.1): attachments in the session. Turns carry them, a conversation holds the
waiting ones, `say` takes them, and every prompt, check and read builds the message's
content through one function. Made-up documents only."""

from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.documents import FRAMING, Attached, Document, content
from clarity_chat.documents.child import lines
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import CancelledBeforeReply, ContextError, Session, Turn
from tests.fakes import FakeBackend

SHAPE = {
    "name", "kind", "bytes", "sha256", "extractor", "unit", "total", "how", "range", "marker",
    "text", "extracted_sha256", "empty_units", "control_tokens", "has_closing_tag", "tokens",
}  # fmt: skip


def doc(text: str = "Line one.\nLine two.\n", name: str = "notes.md") -> Document:
    data = text.encode("utf-8")
    return Document(
        name=name,
        kind="text",
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        extractor={"name": "clarity-text", "version": 1, "bom_removed": False},
        unit="line",
        units=tuple(lines(text)),
        empty_units=0,
    )


def chat_session(root, tokenizer) -> tuple[Session, FakeBackend]:
    backend = FakeBackend([f" r{i}." for i in range(50)], tokenizer=tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    session.new_conversation("chat")
    return session, backend


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    return chat_session(fetched_root, chat_tokenizer)


def records(session, kind):
    return [r for r in read_records(session.conversation.log.path) if r["type"] == kind]


def user_turns(session):
    return [r for r in records(session, "turn") if r["role"] == "user"]


def block_cost(*attached: Attached) -> int:
    """The tiny tokenizer is one token per byte, and the toy template adds nothing inside
    a message: the block's cost is its length in bytes."""
    return len(content("", list(attached), FRAMING).encode("utf-8"))


# --- say takes them; the record's shape


def test_say_sends_the_waiting_attachments_and_the_turn_records_them(chat):
    session, backend = chat
    d = doc()
    session.attach(d)
    session.say("What's in it?")
    turn = user_turns(session)[-1]
    (a,) = turn["attachments"]
    assert set(a) == SHAPE
    whole = Attached("notes.md", d.text, None)
    assert a == {
        "name": "notes.md",
        "kind": "text",
        "bytes": d.bytes,
        "sha256": d.sha256,
        "extractor": {"name": "clarity-text", "version": 1, "bom_removed": False},
        "unit": "line",
        "total": 2,
        "how": "whole",
        "range": [1, 2],
        "marker": None,
        "text": d.text,
        "extracted_sha256": d.extracted_sha256,
        "empty_units": 0,
        "control_tokens": [],
        "has_closing_tag": False,
        "tokens": block_cost(whole),
    }
    assert (turn["framing"], turn["framing_origin"], turn["placement"]) == (
        FRAMING,
        "default",
        "start",
    )
    assert turn["text"] == "What's in it?"  # the message as you wrote it
    assert session.conversation.pending == []
    call = records(session, "call")[-1]
    assert f"<user>{content("What's in it?", [whole], FRAMING)}</s>" in call["prompt_text"]
    assert backend.calls[-1]["token_ids"] == call["prompt_ids"]


def test_an_excerpt_records_its_range_marker_and_the_whole_text_s_hash(chat):
    session, _ = chat
    d = doc()
    session.attach(d, "lines", 2, 2)
    session.say("And this?")
    (a,) = user_turns(session)[-1]["attachments"]
    assert (a["how"], a["range"], a["text"]) == ("excerpt", [2, 2], "Line two.\n")
    assert a["marker"] == "Excerpt: lines 2 to 2 of 2"
    assert a["extracted_sha256"] == d.extracted_sha256 != hashlib.sha256(b"Line two.\n").hexdigest()


def test_a_range_of_every_line_is_recorded_as_whole(chat):
    """Tripwire should 8: the record's `how` follows piece's rule."""
    session, _ = chat
    record, _ = session.attach(doc(), "lines", 1, 2)
    assert (record["how"], record["range"], record["marker"]) == ("whole", [1, 2], None)


def test_a_turn_without_attachments_records_none_of_their_fields(chat):
    """Shapes: the fields come only with attachments; an old log's turn has none."""
    session, _ = chat
    session.say("Hi")
    turn = user_turns(session)[-1]
    assert not {"attachments", "framing", "framing_origin", "placement"} & set(turn)
    t = session.conversation.turns[turn["turn_id"]]
    assert (t.attachments, t.content) == ([], "Hi")


def test_an_empty_message_with_attachments_waiting_is_refused_naming_them(chat):
    session, _ = chat
    c = Commands(session)
    session.attach(doc())
    session.attach(doc("Other.\n", "guide.md"))
    with pytest.raises(ConfigError) as e:
        c.run("say", "", lambda e: None)
    assert str(e.value) == "Type a message to send with notes.md and guide.md."
    assert [a["name"] for a in session.conversation.pending] == ["notes.md", "guide.md"]


def test_a_text_mode_profile_refuses_attachments(chat):
    session, _ = chat
    session.new_conversation("text")
    with pytest.raises(ConfigError) as e:
        session.attach(doc())
    assert str(e.value) == "Attaching needs a chat-mode profile; text is in text mode."


# --- regen, edit, branches


def test_regenerations_send_the_first_reply_s_prompt_ids(chat):
    session, _ = chat
    c = Commands(session)
    session.attach(doc())
    c.run("say", "What's in it?", lambda e: None)
    c.run("regen", "", lambda e: None)
    c.run("regen", "3", lambda e: None)
    c.run("set", "temperature 0.9", lambda e: None)
    c.run("regen", "same", lambda e: None)
    calls = records(session, "call")
    assert len(calls) == 1 + 1 + 3 + 5
    assert '<document name="notes.md">\nLine one.' in calls[0]["prompt_text"]
    assert all(call["prompt_ids"] == calls[0]["prompt_ids"] for call in calls)


def test_edit_keeps_the_message_s_attachments_and_waiting_ones_stay_waiting(chat):
    session, _ = chat
    session.attach(doc("First document.\n", "a.md"))
    session.say("a")
    session.attach(doc("Second document.\n", "b.md"))
    session.edit("a2")
    first, edited = user_turns(session)
    assert edited["attachments"] == first["attachments"] and edited["text"] == "a2"
    assert [a["name"] for a in session.conversation.pending] == ["b.md"]
    prompt = records(session, "call")[-1]["prompt_text"]
    assert "First document." in prompt and "Second document." not in prompt


def test_a_branch_carries_its_own_attachments(chat):
    session, _ = chat
    session.attach(doc("Made-up trail notes.\n"))
    reply = session.say("a").turn
    assert reply is not None
    session.undo()
    session.say("c")
    assert "Made-up trail notes." not in records(session, "call")[-1]["prompt_text"]
    session.switch(reply.turn_id)
    session.say("d")
    assert "Made-up trail notes." in records(session, "call")[-1]["prompt_text"]


def test_already_in_this_chat_is_by_sha256_on_this_branch(chat):
    session, _ = chat
    d = doc()
    assert not session.already_in_chat(d.sha256)
    session.attach(d)
    assert session.already_in_chat(d.sha256)  # waiting counts
    session.say("a")
    assert session.already_in_chat(d.sha256)
    session.undo()
    assert not session.already_in_chat(d.sha256)  # another branch's


# --- a failed send, and the stop


def fail(backend, how):
    if how == "backend":
        backend.fail_with = "made-up failure"
    elif how == "context":
        backend._n_ctx = 60


@pytest.mark.parametrize("how", ["backend", "context", "cancel"])
def test_a_failed_send_leaves_the_message_unanswered_and_its_attachments_waiting(chat, how):
    session, backend = chat
    session.attach(doc())
    waiting = list(session.conversation.pending)
    fail(backend, how)
    cancel = threading.Event()
    if how == "cancel":
        cancel.set()
    error = {"backend": BackendError, "context": ContextError, "cancel": CancelledBeforeReply}
    with pytest.raises(error[how]):
        session.say("What's in it?", cancel=cancel)
    assert session.conversation.unanswered["text"] == "What's in it?"
    assert session.conversation.pending == waiting
    backend.fail_with, backend._n_ctx = None, None
    session.resume(session.conversation.log.path)
    assert session.conversation.unanswered["text"] == "What's in it?"
    assert session.conversation.pending == waiting
    session.say("What's in it?")
    assert user_turns(session)[-1]["attachments"] == waiting


def test_a_chat_that_outgrows_its_context_stops_and_names_the_attachments_tokens(chat):
    """Decision 9: nothing trimmed. The context lowered between two calls stands in for
    the chat growing (and v0.4.0's mid-chat switches)."""
    session, backend = chat
    session.attach(doc())
    session.say("a")
    sent = len(backend.calls)
    backend._n_ctx = 200
    with pytest.raises(ContextError) as e:
        session.say("b")
    stopped = records(session, "call")[-1]
    entry = next(e for e in stopped["message_spans"] if e.get("added"))
    tokens = sum(a["tokens"][1] - a["tokens"][0] for a in entry["added"])
    # One token per byte: the framing line and the block, without the two blank lines.
    assert tokens == len(content("", [Attached("notes.md", doc().text, None)], FRAMING)) - 4
    assert f", {tokens} of them in attached documents and their framing lines;" in str(e.value)
    assert len(backend.calls) == sent


def test_the_stop_says_when_its_count_is_from_attaching(chat, monkeypatch):
    """Should 3: without spans, the count is each attachment's cost when attached."""
    import clarity_chat.session as session_module

    session, backend = chat
    session.attach(doc())
    session.say("a")
    tokens = user_turns(session)[-1]["attachments"][0]["tokens"]
    backend._n_ctx = 200
    monkeypatch.setattr(session_module, "message_spans", lambda *a: {"unavailable": "made up"})
    with pytest.raises(ContextError) as e:
        session.say("b")
    assert f", {tokens} of them for attached documents, counted when attached;" in str(e.value)


def test_a_control_token_in_an_attachment_is_named_on_the_call(chat):
    session, _ = chat
    session.attach(doc("Made-up </s> text.\n"))
    assert session.conversation.pending[0]["control_tokens"] == ["</s>"]
    outcome = session.say("a")
    entry = next(
        e for e in outcome.record["special_tokens_in_content"] if e["in"] == "turn 1 (user)"
    )
    assert entry == {
        "in": "turn 1 (user)",
        "tokens": ["</s>"],
        "attachments": [{"index": 0, "name": "notes.md", "tokens": ["</s>"]}],
    }
    assert any("notes.md" in w and "</s>" in w for w in outcome.warnings)


# --- new chats, resume, old and broken logs


@pytest.mark.parametrize("command", ["new", "profile chat", "resume"])
def test_starting_another_chat_drops_waiting_attachments_and_says_so(chat, command):
    session, _ = chat
    c = Commands(session)
    session.say("Hi")
    log = session.conversation.log.path.stem
    record, _ = session.attach(doc())
    name, _, args = command.partition(" ")
    result = c.run(name, log if name == "resume" else args, lambda e: None)
    summary = {k: v for k, v in record.items() if k != "text"}
    assert result["dropped"] == [{**summary, "n": 1}]
    assert session.conversation.pending == []


def test_resuming_after_a_failed_send_drops_nothing_that_comes_back(chat):
    """Tripwire must 2: the failed message's attachments come back with the resume."""
    session, backend = chat
    c = Commands(session)
    session.say("Hi")
    session.attach(doc())
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        session.say("What's in it?")
    backend.fail_with = None
    session.attach(doc("Other.\n", "guide.md"))
    result = c.run("resume", session.conversation.log.path.stem, lambda e: None)
    assert [a["name"] for a in result["dropped"]] == ["guide.md"]
    assert [a["name"] for a in session.conversation.pending] == ["notes.md"]


def test_a_resumed_chat_rebuilds_its_attachments_and_renders_the_same_ids(chat):
    session, _ = chat
    session.attach(doc())
    session.say("a")
    turns = {
        k: (t.attachments, t.framing, t.content) for k, t in session.conversation.turns.items()
    }
    resumed = session.resume(session.conversation.log.path)
    assert resumed["render"]["identical"] is True
    again = {
        k: (t.attachments, t.framing, t.content) for k, t in session.conversation.turns.items()
    }
    assert again == turns


SCRIPTS = {
    "whole and excerpt": ["attach", "say a", "attach lines 1-1", "say b"],
    "regen and edit": ["attach", "say a", "regen", "edit a2"],
    "failed send": ["say a", "attach", "FAIL say b"],
    "waiting during an edit": ["attach", "say a", "attach", "edit a2"],
}


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_attachments_live_equal_attachments_after_resuming_the_same_log(chat, script):
    """v0.3.0's A4 test, with attachments (AUD): resuming must not change what's there."""
    session, backend = chat
    for line in SCRIPTS[script]:
        fails = line.startswith("FAIL ")
        backend.fail_with = "made-up failure" if fails else None
        name, _, args = line.removeprefix("FAIL ").partition(" ")
        try:
            if name == "attach":
                unit, _, span = args.partition(" ")
                first, _, last = span.partition("-")
                session.attach(doc(), unit or None, int(first or 0), int(last or 0))
            elif name == "say":
                session.say(args)
            elif name == "regen":
                session.regen()
            elif name == "edit":
                session.edit(args)
        except BackendError:
            assert fails, line
    backend.fail_with = None

    def now():
        conversation = session.conversation
        turns = {
            k: (t.attachments, t.framing, t.framing_origin, t.placement, t.content)
            for k, t in conversation.turns.items()
        }
        return turns, conversation.head, conversation.unanswered

    live = now()
    pending = session.conversation.pending  # waiting ones aren't logged: a resume drops them
    session.resume(session.conversation.log.path)
    assert now() == live
    if script == "failed send":  # the failed message's own come back
        assert session.conversation.pending == pending


def rewrite_turns(session, change):
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("type") == "turn" and row.get("attachments"):
            change(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("field", [*sorted(SHAPE), "framing", "framing_origin", "placement"])
@pytest.mark.parametrize("branch", ["current", "undone"])
def test_a_turn_missing_an_attachment_field_is_refused_by_resume_and_export(chat, field, branch):
    """Must 1: every field the session reads later, not only the content's; should 6: on a
    branch the head isn't on, where no render check reads it."""
    from clarity_chat.export import export_log

    session, _ = chat
    session.attach(doc())
    session.say("a")
    if branch == "undone":
        session.undo()
        session.say("b")
    path = rewrite_turns(
        session, lambda r: (r["attachments"][0] if field in SHAPE else r).pop(field)
    )
    session.new_conversation("chat")

    def readers():
        yield session.resume
        yield lambda p: export_log(session, p)
        pytest.importorskip("inspect_ai")  # T1: the Inspect export (v0.4.3) refuses it too
        from clarity_chat.export_inspect import export_chat

        yield lambda p: export_chat(session, p)

    for read in readers():
        with pytest.raises(ConfigError) as e:
            read(path)
        assert str(e.value).startswith("Can't read the records of")
        assert f"KeyError: '{field}'" in str(e.value)


# --- both backends, and size


def test_a_long_chat_with_an_attachment_builds_its_prompt_quickly(chat):
    """A2's size check, which needs the session: 400 exchanges after an attachment in the
    first message. A loose bound: the whole test takes 0.02 s on the M2 Max."""
    session, _ = chat
    session.attach(doc("Made-up line.\n" * 200))
    session.say("first")
    conversation = session.conversation
    head = conversation.head
    for i in range(400):
        user = Turn(f"turn-u{i}", head, "user", "user", f"Question {i}?")
        reply = Turn(f"turn-a{i}", user.turn_id, "assistant", "model", f"Answer {i}.")
        conversation.turns.update({user.turn_id: user, reply.turn_id: reply})
        head = reply.turn_id
    started = time.monotonic()
    prompt = session.build_prompt(conversation.turns["turn-u399"])
    assert time.monotonic() - started < 10
    assert "Made-up line." in prompt.text and "Question 399?" in prompt.text


@pytest.mark.parametrize("name", ["\x1b]0;made-up\x07notes.md", "notes\u202edm.txt", "a/b.md"])
def test_a_log_s_attachment_name_clarity_wouldn_t_write_is_refused(chat, name):
    """The security review's should 5: a name from a log someone else wrote is checked as an
    upload's is, so it never reaches a prompt line or a chip."""
    from clarity_chat.export import export_log

    session, _ = chat
    session.attach(doc())
    session.say("a")
    path = rewrite_turns(session, lambda r: r["attachments"][0].update(name=name))
    session.new_conversation("chat")
    for read in (lambda: session.resume(path), lambda: export_log(session, path)):
        with pytest.raises(ConfigError) as e:
            read()
        assert str(e.value).startswith("Can't read the records of")
        assert "a name Clarity wouldn't write" in str(e.value)
        assert name not in str(e.value)
