"""Resume: a conversation rebuilt from its log continues as it was logged."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clarity_chat.cli import ChatLoop
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend

GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "v0.1.0.jsonl"


@pytest.fixture
def new_session(fetched_root, tiny_tokenizer):
    def make(replies=None):
        backend = FakeBackend(replies, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        return session, backend

    return make


def kinds(path, kind):
    return [r for r in read_records(path) if r["type"] == kind]


def test_a_resumed_chat_continues_where_it_was(new_session):
    session, _ = new_session([" one", " two", " three"])
    session.new_conversation("text")
    first = session.say("a")
    session.say("b")
    session.regen()
    session.set_setting("temperature", "0.9")
    session.undo()  # back to the reply to "a"
    path = session.conversation.log.path
    before = path.read_bytes()

    resumed, backend = new_session([" four"])
    record = resumed.resume(path)
    conversation = resumed.conversation
    assert path.read_bytes().startswith(before)  # appended to the same log
    assert conversation.conversation_id == session.conversation.conversation_id
    assert conversation.head == record["head"] == first.turn.turn_id
    assert conversation.turns.keys() == session.conversation.turns.keys()
    assert conversation.chat_settings == record["chat_settings"] == {"temperature": 0.9}
    assert conversation.last_call["call_id"] == first.record["call_id"]
    assert record["head_guessed"] is False and record["profile_file"] == "same"
    assert record["render"] == {"call_id": first.record["call_id"], "identical": True}
    assert record["ctx_size"] == {"value": 8192, "origin": "log"}
    nxt = resumed.say("c")
    assert nxt.turn.parent_id is not None
    assert conversation.turns[nxt.turn.parent_id].parent_id == first.turn.turn_id
    assert nxt.record["settings"]["temperature"] == {"value": 0.9, "origin": "chat"}
    assert backend.ctx_sizes == [8192]


def test_resume_uses_the_logged_profile_and_settings_not_todays(new_session, fetched_root):
    session, _ = new_session([" one"])
    session.new_conversation("text")
    session.say("a")
    path = session.conversation.log.path
    profile = fetched_root / "profiles" / "text.toml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace("temperature = 0.6", "temperature = 0.7"),
        encoding="utf-8",
    )
    resumed, _ = new_session([" two"])
    record = resumed.resume(path)
    assert record["profile_file"] == "changed"
    assert record["settings_differ_today"] == [
        {
            "name": "temperature",
            "logged": {"value": 0.6, "origin": "profile"},
            "today": {"value": 0.7, "origin": "profile"},
        }
    ]
    assert resumed.say("b").record["settings"]["temperature"]["value"] == 0.6


def test_a_message_whose_call_never_finished_resumes_before_it(new_session):
    session, _ = new_session([" one"])
    session.new_conversation("text")
    first = session.say("a")
    session.conversation.add_turn("user", "user", "lost", first.turn.turn_id)  # then a crash
    resumed, _ = new_session()
    assert resumed.resume(session.conversation.log.path)["head"] == first.turn.turn_id


def test_a_changed_rendering_is_shown_and_checked_again_on_switch(new_session, tmp_path):
    session, _ = new_session([" one", " two"])
    session.new_conversation("text")
    first = session.say("a")
    second = session.say("b")
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:  # as if the template had rendered "b" differently back then
        if row.get("call_id") == second.record["call_id"] and row["type"] == "call":
            row["prompt_ids"] = row["prompt_ids"][:-1]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    resumed, _ = new_session()
    render = resumed.resume(path)["render"]
    assert render["identical"] is False and render["n_ids"]["now"] == render["n_ids"]["logged"] + 1
    _, render = resumed.switch(first.turn.turn_id)  # the newest reply on that branch: " two"
    assert render["identical"] is False
    assert kinds(path, "head")[-1]["render"] == render  # logged, too
    resumed.undo()
    assert resumed.switch(first.turn.parent_id)[1]["call_id"] == second.record["call_id"]


def test_a_batch_runs_calls_file_is_refused(new_session, tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"schema_version": 1, "type": "run", "time": "t"}\n', encoding="utf-8")
    session, _ = new_session()
    with pytest.raises(ConfigError, match="A batch run's calls.jsonl can't be resumed"):
        session.resume(path)


def test_the_v0_1_0_golden_log_resumes(new_session, tmp_path):
    path = Path(shutil.copy(GOLDEN, tmp_path / GOLDEN.name))
    session, _ = new_session([" Sure."])
    record = session.resume(path)
    assert record["head_guessed"] is True
    # The log ends with a failed call on "Are you there?": it continues before that message.
    assert record["head"] == "turn-04fa9a616f6a"
    # Logged at /project/profiles/text.toml; this project's "text" profile has the same text.
    assert record["profile_file"] == "same"
    assert record["render"] == {"call_id": "call-78a5a5886108", "identical": True}
    assert session.say("Still there?").turn.text == " Sure."
    assert kinds(path, "resume")[0]["compared_with_call"] == "call-78a5a5886108"


def test_the_terminal_resumes_and_says_what_differs(new_session, fetched_root, tmp_path):
    path = Path(shutil.copy(GOLDEN, tmp_path / GOLDEN.name))
    session, _ = new_session()
    output: list[str] = []
    ChatLoop(session, "", read=lambda d: "", write=output.append, resume=str(path)).start()
    text = "".join(output)
    assert "Resumed 20261110-154328-2b2bb5 with profile text" in text
    assert "This log is from v0.1.0" in text
    assert "Changed since its last call (call-78a5a5886108)" in text
    assert "renders to the same token IDs today" in text


def test_resume_takes_a_conversation_id_from_the_chat_list(new_session):
    session, _ = new_session([" one"])
    session.new_conversation("text")
    session.say("a")
    conversation_id = session.conversation.conversation_id
    resumed, _ = new_session()
    assert resumed.log_for(conversation_id) == session.conversation.log.path
    assert resumed.resume(resumed.log_for(conversation_id))["conversation_id"] == conversation_id


def test_resume_checks_restored_set_values(new_session):
    session, _ = new_session([" one"])
    session.new_conversation("text")
    session.say("a")
    session.conversation.log.append(
        {"type": "settings", "conversation_id": "c", "name": "top_k", "value": -1,
         "chat_settings": {"top_k": -1}}
    )  # fmt: skip
    resumed, backend = new_session()
    with pytest.raises(ConfigError, match="top_k must be"):
        resumed.resume(session.conversation.log.path)
    assert backend.ctx_sizes == []  # refused before any model started


def test_a_log_that_cant_be_rebuilt_leaves_the_running_chat_alone(new_session, tmp_path):
    session, backend = new_session([" one", " two"])
    session.new_conversation("text")
    session.say("a")
    running = session.conversation
    other, _ = new_session([" x"])
    other.new_conversation("text")
    other.say("b")
    path = other.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["type"] == "call":
            del row["settings"]["seed"]  # a log Clarity didn't write
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ConfigError, match="can't be resumed: a record lacks or mangles 'seed'"):
        session.resume(path)
    assert session.conversation is running and backend.ctx_sizes == [8192]  # not restarted


def test_resume_lists_what_it_takes_from_today(new_session, fetched_root):
    session, _ = new_session([" one"])
    session.new_conversation("text")
    session.say("a")
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["settings"]["repeat_last_n"]  # as if the setting were added after this log
    rows[0]["model"]["tokenizer_revision"] = "f" * 40  # as if models.toml changed since
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    resumed, _ = new_session()
    record = resumed.resume(path)
    assert {
        "name": "repeat_last_n",
        "logged": None,
        "today": {"value": 64, "origin": "default"},
    } in record["settings_differ_today"]
    assert [d["path"] for d in record["model_entry"]] == ["tokenizer_revision"]
    assert record["not_recorded"] is not None
