"""/state: one read a page can make at any moment, mid-reply included, to draw itself."""

from __future__ import annotations

import threading

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.session import Session
from tests.fakes import FakeBackend


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hi there, you."] * 3, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    return Commands(session), backend


def state(c):
    return c.run("state", "", lambda e: None)


def test_state_answers_before_any_conversation(commands):
    c, _ = commands
    result = state(c)
    assert result.get("conversation_id") is None and result["model"] == "none"
    assert result["running"] is None and result["turns"] == []
    assert result["clarity"]["version"] and "commit" in result["clarity"]
    assert len(result["llama_cpp_commit"]) == 40


def test_state_answers_while_a_reply_streams_with_its_text_so_far(commands):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    seen = []

    def on_event(event):
        if event["event"] == "text" and len(seen) < 2:
            seen.append(state(c))  # during the call, from inside it

    c.run("say", "Hi", on_event)
    assert [s["running"] for s in seen] == [
        {"command": "say", "args": "Hi", "k": None, "n": None}
    ] * 2
    assert seen[0]["reply_so_far"] == " Hi "  # the fake streams a word at a time
    assert seen[1]["reply_so_far"] == " Hi there, "
    after = state(c)
    assert after["running"] is None and after["reply_so_far"] == ""
    assert [t["text"] for t in after["turns"]] == ["Hi", " Hi there, you."]
    assert after["model"] == "ready" and after["head"] == after["turns"][-1]["turn_id"]


def test_reading_turns_from_another_thread_during_calls_is_safe(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.replies = [" word" * 50] * 20
    errors: list[BaseException] = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            try:
                state(c)
            except BaseException as e:  # noqa: BLE001 - any error here is the finding
                errors.append(e)
                return

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(20):
            c.run("say", f"m{i}", lambda e: None)
    finally:
        done.set()
        thread.join()
    assert errors == []
    assert len(state(c)["turns"]) == 40


def test_state_after_a_cancel_and_after_a_failure(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.cancel_after = 4
    c.run("say", "Hi", lambda e: None)
    backend.cancel_after = None
    after_cancel = state(c)
    assert after_cancel.get("unanswered") is None and after_cancel["running"] is None
    assert after_cancel["turns"][-1]["text"] == " Hi "
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        c.run("say", "Again", lambda e: None)
    after_failure = state(c)
    assert after_failure.get("unanswered", {}).get("text") == "Again"
    assert after_failure["head"] == after_cancel["head"]  # back where it was
    backend.fail_with = None
    c.run("say", "Third", lambda e: None)
    assert state(c).get("unanswered") is None


def test_state_restores_the_title_warnings_and_resume_differences(commands, fetched_root):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    c.run("title", "A made-up title", lambda e: None)
    conversation_id = c.session.conversation.conversation_id
    assert state(c).get("title") == "A made-up title"
    c.run("resume", conversation_id, lambda e: None)
    result = state(c)
    assert result["resume_differ"] == {"count": 0, "items": []}
    assert result["warnings"] == [] and result["profile"] == "text"


def test_the_starting_event_names_the_profile_not_its_file(commands):
    c, _ = commands
    events: list[dict] = []
    c.run("profile", "text", events.append)
    starting = next(e for e in events if e.get("status") == "starting")
    assert starting["profile"] == "text"


def test_the_terminal_prints_state_in_one_line(commands):
    from clarity_chat.cli import format_state

    c, _ = commands
    assert format_state(state(c)).startswith("No conversation; no model loaded.")
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    text = format_state(state(c))
    assert f"Conversation {c.session.conversation.conversation_id} (profile text)" in text
    assert "2 turns on this branch; the model is ready." in text


def test_the_reply_so_far_starts_again_with_each_reply_of_regen_n(commands):
    """Mutation check: during /regen 2, the second reply's text, not both."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    backend.replies = [" First one.", " Second one."]
    seen: list[str] = []

    def on_event(event):
        if event["event"] == "text":
            seen.append(state(c)["reply_so_far"])

    c.run("regen", "2", on_event)
    assert seen[-1] == " Second one."


def test_a_failed_edit_leaves_nothing_unanswered_live_or_after_a_resume(commands):
    """Checkpoint 1, finding 1: a reloaded page mustn't put the edit back as a new message."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        c.run("edit", "Hello instead", lambda e: None)
    backend.fail_with = None
    assert state(c)["unanswered"] is None
    c.run("resume", c.session.conversation.conversation_id, lambda e: None)
    assert state(c)["unanswered"] is None


def test_state_says_which_reply_of_regen_n_is_being_written(commands):
    """Checkpoint 1, finding 3 (DW section 4): a page reloaded mid-/regen 3 draws "2 of 3"."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    backend.replies = [" One.", " Two.", " Three."]
    seen: list[dict] = []

    def on_event(event):
        if event["event"] == "text":
            seen.append(state(c)["running"])

    c.run("regen", "3", on_event)
    assert seen[-1] == {"command": "regen", "args": "3", "k": 3, "n": 3}
    assert state(c)["running"] is None


def test_state_gives_what_differs_after_a_resume_until_the_next_call(commands, fetched_root):
    """Checkpoint 1, finding 3: the same count and sentences /resume gave, not the raw record."""
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    profile = fetched_root / "profiles" / "text.toml"
    profile.write_text(profile.read_text("utf-8").replace("0.6", "0.7"), "utf-8")
    resumed = c.run("resume", c.session.conversation.conversation_id, lambda e: None)
    assert state(c).get("resume_differ") == resumed["differ"] and resumed["differ"]["count"]
    c.run("say", "Again", lambda e: None)
    assert state(c).get("resume_differ") is None


def test_a_failed_start_leaves_no_model_starting(commands):
    """Checkpoint 1, finding 4; the final checkpoint showed the first version couldn't fail
    (an unknown name is refused before "starting"): here the start itself fails."""
    from clarity_chat.profiles import ConfigError

    c, backend = commands

    def fails(model, weights, ctx_size):
        raise ConfigError("no GPU")

    backend.start = fails
    with pytest.raises(ConfigError):
        c.run("profile", "text", lambda e: None)
    assert c.session.conversation is None and state(c)["model"] == "none"


def test_running_has_no_k_or_n_after_a_regen(commands):
    """Final checkpoint, finding 5: a /say after /regen 3 isn't "reply 3 of 3"."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    backend.replies = [" a", " b", " c", " d"]
    c.run("regen", "3", lambda e: None)
    seen: list[dict] = []
    c.run(
        "say", "Next", lambda e: seen.append(state(c)["running"]) if e["event"] == "text" else None
    )
    assert seen[-1] == {"command": "say", "args": "Next", "k": None, "n": None}


def test_unanswered_comes_back_after_a_resume(commands):
    """Checkpoint 1, finding 5."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        c.run("say", "Lost one", lambda e: None)
    backend.fail_with = None
    c.run("resume", c.session.conversation.conversation_id, lambda e: None)
    assert state(c)["unanswered"]["text"] == "Lost one"


def test_state_gives_the_servers_start_warnings(commands):
    """Checkpoint 1, finding 10."""
    c, backend = commands
    backend.start_warnings = ["A made-up start warning."]
    c.run("profile", "text", lambda e: None)
    assert state(c)["warnings"] == ["A made-up start warning."]


def test_state_on_400_exchanges_is_quick(fetched_root, tiny_tokenizer):
    """The plan's size test for item 10."""
    import time

    backend = FakeBackend([" ok"] * 400, n_ctx=100_000, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("text")
    for _ in range(400):
        session.say("m")
    c = Commands(session)
    started = time.perf_counter()
    assert len(state(c)["turns"]) == 800
    assert time.perf_counter() - started < 5


def test_a_failed_edit_after_a_failed_message_leaves_nothing_unanswered(commands):
    """Review: live and resumed answers agree on the same log."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        c.run("say", "A", lambda e: None)
    with pytest.raises(BackendError):
        c.run("edit", "Hello instead", lambda e: None)
    backend.fail_with = None
    live = state(c)["unanswered"]
    c.run("resume", c.session.conversation.conversation_id, lambda e: None)
    assert live == state(c)["unanswered"] is None


# --- A1 (v0.3.0): text events carry their position, so a page joins /state and the stream


def test_text_events_carry_at_and_end_in_the_reply(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.replies = [" Hi 🙂 there."]
    events: list[dict] = []
    reply = c.run("say", "Hi", events.append)["call"]["output_text"]
    texts = [e for e in events if e["event"] == "text"]
    assert [(e.get("at"), e.get("end")) for e in texts][0][0] == 0
    assert all(a.get("end") == b.get("at") for a, b in zip(texts, texts[1:], strict=False))
    assert texts[-1].get("end") == len(reply)  # code points: the emoji counts as one


@pytest.mark.parametrize("command", ["say Hi", "regen 2"])
def test_a_page_that_reads_state_mid_reply_joins_the_stream_exactly(commands, command):
    """Wherever the page reads /state, it draws the last reply exactly: from the snapshot plus
    the events at or past reply_end if the snapshot was of that reply, else from that reply's
    own events (it began after the snapshot)."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    if command.startswith("regen"):
        c.run("say", "Hi", lambda e: None)
    for moment in range(1, 8):
        backend.replies = [" One 🙂 two", " Three four five"]
        events: list[dict] = []
        seen: dict = {}

        def on_event(event, moment=moment, events=events, seen=seen):
            events.append(event)
            texts = sum(e["event"] == "text" for e in events)
            if event["event"] == "text" and texts == moment and not seen:
                seen.update(state=state(c), index=len(events))

        name, _, args = command.partition(" ")
        result = c.run(name, args, on_event)
        last = (result.get("replies") or [result])[-1]["call"]["output_text"]
        if not seen:
            continue  # the replies had fewer pieces than this moment
        snapshot, after = seen["state"], events[seen["index"] :]
        starts = [i for i, e in enumerate(after) if e["event"] == "reply_start"]
        if starts:  # the last reply began after the snapshot
            later = after[starts[-1] :]
            joined = "".join(e["text"] for e in later if e["event"] == "text")
        else:
            end = snapshot.get("reply_end", 10**9)
            tail = [e for e in after if e["event"] == "text" and e.get("at", -1) >= end]
            joined = snapshot["reply_so_far"] + "".join(e["text"] for e in tail)
        assert joined == last, (command, moment)


def test_no_running_command_means_no_reply_so_far(commands):
    """A1b: the snapshot is consistent."""
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    after = state(c)
    assert after["running"] is None and after["reply_so_far"] == "" and after.get("reply_end") == 0


# --- A2 (v0.3.0): reads that run during a call


def test_chats_profiles_and_settings_answer_while_a_reply_is_written(commands):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    answers: dict[str, dict] = {}

    def on_event(event):
        if event["event"] == "text" and not answers:
            for name in ("chats", "profiles", "settings"):
                answers[name] = c.run(name, "", lambda e: None)

    c.run("say", "Hi", on_event)
    assert set(answers) == {"chats", "profiles", "settings"}
    assert answers["settings"]["settings"]["temperature"]["value"] == 0.6


def test_settings_waits_while_a_model_starts(commands):
    """Decision 5: during a start, only state, cancel, chats and profiles run."""
    from clarity_chat.profiles import ConfigError

    c, _ = commands
    c.run("profile", "text", lambda e: None)
    refused: list[str] = []

    def on_event(event):
        if event.get("status") == "starting":
            try:
                c.run("settings", "", lambda e: None)
            except ConfigError as e:
                refused.append(str(e))
            c.run("chats", "", lambda e: None)  # this one runs

    c.run("new", "", on_event)
    assert refused and "still running" in refused[0]


# --- A3 (v0.3.0): reads from another thread across every command that changes the chat


def test_state_and_settings_from_another_thread_across_the_commands(commands):
    """Any exception in the reader, other than the refusal while a command can't share,
    fails the test."""
    from clarity_chat.profiles import ConfigError

    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.replies = [" word" * 3] * 200
    errors: list[BaseException] = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            for name in ("state", "settings", "chats"):
                try:
                    c.run(name, "", lambda e: None)
                except ConfigError as e:
                    if "still running" not in str(e):
                        errors.append(e)
                except BaseException as e:  # noqa: BLE001 - any error here is the finding
                    errors.append(e)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(8):
            c.run("say", f"m{i}", lambda e: None)
            c.run("regen", "2", lambda e: None)
            c.run("edit", f"edited {i}", lambda e: None)
            c.run("undo", "", lambda e: None)
            ends = c.run("branches", "all", lambda e: None)["ends"]
            c.run("switch", ends[-1]["turn_id"], lambda e: None)
            c.run("resume", c.session.conversation.conversation_id, lambda e: None)
            if i % 4 == 3:
                c.run("new", "", lambda e: None)
    finally:
        done.set()
        thread.join()
    assert errors == []


# --- A4 (v0.3.0): live and resumed agree; every setting has a value in old logs

SCRIPTS = {
    "regen and same": ["say a", "regen 2", "set temperature 0.9", "regen same"],
    "failed message": ["say a", "FAIL say b"],
    "failed edit": ["say a", "FAIL say b", "FAIL edit c"],
    "cancel": ["say a", "CANCEL say b"],
    "marks and notes": ["say a", "say b", "mark landed", "note made up", "unmark"],
    "undo and switch": ["say a", "regen", "undo", "say c"],
    "edit": ["say a", "edit a2", "say b"],
}


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_state_live_equals_state_after_resuming_the_same_log(fetched_root, tiny_tokenizer, script):
    """The bug class v0.2.3's reviews found twice: what a page draws must not change when
    the same log is resumed (apart from what the resume itself adds)."""
    from clarity_chat.profiles import ConfigError

    backend = FakeBackend([f" r{i}." for i in range(20)], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    for line in SCRIPTS[script]:
        mode = ""
        if line.startswith(("FAIL ", "CANCEL ")):
            mode, line = line.split(" ", 1)
        backend.fail_with = "made-up failure" if mode == "FAIL" else None
        backend.cancel_after = 2 if mode == "CANCEL" else None
        name, _, args = line.partition(" ")
        try:
            c.run(name, args, lambda e: None)
        except (BackendError, ConfigError):
            assert mode == "FAIL", line
        backend.fail_with, backend.cancel_after = None, None
    keys = ("conversation_id", "title", "profile", "head", "turns", "unanswered")
    live = {k: state(c)[k] for k in keys}
    c.run("resume", live["conversation_id"], lambda e: None)
    resumed = {k: state(c)[k] for k in keys}
    assert resumed == live


def test_every_setting_is_in_the_oldest_log_or_has_its_value_from_before():
    """A setting added later (v0.4.0, v0.5.0) must say what calls before it ran with."""
    import json
    from pathlib import Path

    from clarity_chat.profiles import SETTING_NAMES
    from clarity_chat.session import BEFORE_THEY_EXISTED

    golden = Path(__file__).parent / "fixtures" / "golden" / "v0.1.0.jsonl"
    calls = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines()]
    oldest = next(r for r in calls if r["type"] == "call")["settings"]
    missing = set(SETTING_NAMES) - set(oldest) - set(BEFORE_THEY_EXISTED) - {"seed"}
    assert missing == set()
