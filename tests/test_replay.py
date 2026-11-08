"""Replay: comparisons, environment differences, and replays through the fake backend."""

from __future__ import annotations

import pytest

from clarity_chat import cli
from clarity_chat.cli import ChatLoop, format_replay
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.replay import compare_output, deep_diff, environment_diff, unexplained
from clarity_chat.session import Session
from tests.fakes import FakeBackend

# --- pure functions -----------------------------------------------------------------------


def test_deep_diff_lists_changed_added_and_removed_paths():
    old = {"a": 1, "b": {"c": [1, 2]}, "gone": True}
    new = {"a": 1, "b": {"c": [1, 3]}, "added": 0}
    assert deep_diff(old, new) == [
        {"path": "added", "logged": "(missing)", "now": 0},
        {"path": "b.c[1]", "logged": 2, "now": 3},
        {"path": "gone", "logged": True, "now": "(missing)"},
    ]
    assert deep_diff({"x": [1]}, {"x": [1, 2]}) == [{"path": "x", "logged": [1], "now": [1, 2]}]
    assert deep_diff({"a": 1}, {"a": 1}) == []


def test_environment_diff_names_each_fact_and_caps_long_lists():
    logged = {"build_info": "b1-aaa", "props": {str(i): i for i in range(60)}}
    now = {"build_info": "b2-bbb", "props": {str(i): -i - 1 for i in range(60)}}
    diffs = environment_diff(logged, now)
    assert diffs["build_info"] == [{"path": "(value)", "logged": "b1-aaa", "now": "b2-bbb"}]
    assert len(diffs["props"]) == 51 and diffs["props"][-1]["path"] == "…"


def call(ids, stop="eos", text="x", flags=()):
    return {"output_ids": ids, "stop_type": stop, "output_text": text, "flags": list(flags)}


def test_identical_means_same_ids_and_stop_type():
    assert compare_output(call([1, 2]), call([1, 2]))["identical"]
    different = compare_output(call([1, 2, 3]), call([1, 5]))
    assert not different["identical"]
    assert different["first_different_token"] == 1
    assert different["n_output_ids"] == {"logged": 3, "now": 2}
    assert not compare_output(call([1]), call([1], stop="limit"))["identical"]


def test_incomplete_ids_compare_by_text():
    logged = call([1], text="same", flags=["incomplete_output_ids"])
    result = compare_output(logged, call([1, 2], text="same"))
    assert result["identical"] and "text" in result["compared"]


def test_unexplained_only_when_nothing_that_matters_differs():
    assert unexplained(False, {})
    assert unexplained(False, {"client": [{}], "libraries": [{}]})
    assert not unexplained(False, {"build_info": [{}]})
    assert not unexplained(True, {})


# --- replays ------------------------------------------------------------------------------


@pytest.fixture
def make_session(fetched_root, tiny_tokenizer):
    def make(replies=None, logs_dir=None):
        backend = FakeBackend(replies, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root,
            backend=backend,
            logs_dir=logs_dir,
            tokenizer_loader=lambda r, v: tiny_tokenizer,
        )
        return session, backend

    return make


def test_replay_resends_the_logged_ids_and_request(make_session):
    session, backend = make_session([" Hello!", " Hello!"])
    session.new_conversation("text")
    outcome = session.say("Hi")
    record = session.replay_current()
    assert backend.calls[1]["token_ids"] == backend.calls[0]["token_ids"]
    assert backend.calls[1]["request"] == outcome.record["request"]
    assert record["identical"] and record["differences"] == {}
    assert not record["unexplained_difference"]
    logged = [r for r in read_records(session.conversation.log.path) if r["type"] == "replay"]
    assert logged[0]["call_id"] == outcome.record["call_id"]


def test_a_different_output_with_nothing_changed_is_flagged(make_session):
    session, _ = make_session([" Hello!", " Bye"])
    session.new_conversation("text")
    session.say("Hi")
    record = session.replay_current()
    assert not record["identical"]
    assert record["unexplained_difference"]
    assert record["logged_output_text"] == " Hello!" and record["output_text"] == " Bye"
    text = format_replay(record)
    assert "Different" in text and "nondeterminism" in text


def test_replay_from_a_log_file_in_a_new_session(make_session, tmp_path):
    first, _ = make_session([" Hello!"], logs_dir=tmp_path)
    conversation = first.new_conversation("text")
    call_id = first.say("Hi").record["call_id"]
    second, backend = make_session([" Hello!"])
    record = second.replay(conversation.log.path, call_id)
    assert record["identical"]
    assert backend.started == ["tiny"]
    kinds = [r["type"] for r in read_records(conversation.log.path)]
    assert kinds[-2:] == ["server_start", "replay"]  # the replaying server is logged first


def test_replay_errors(make_session):
    session, backend = make_session()
    session.new_conversation("text")
    with pytest.raises(ConfigError, match="No reply to replay"):
        session.replay_current()
    backend.fail_with = "server gone"
    with pytest.raises(Exception, match="server gone"):
        session.say("Hi")
    failed = next(r for r in read_records(session.conversation.log.path) if r["type"] == "call")
    path = session.conversation.log.path
    with pytest.raises(ConfigError, match="no reply to compare"):
        session.replay(path, failed["call_id"])
    with pytest.raises(ConfigError, match="has no call"):
        session.replay(path, "call-nope")


def test_replay_command_in_the_chat(make_session):
    session, _ = make_session([" Hello!", " Hello!"])
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.start()
    loop.handle("Hi")
    output.clear()
    loop.handle("/replay")
    assert "Identical: same output token IDs and stop type" in "".join(output)
    assert "Nothing recorded differs" in "".join(output)


def test_replay_command_exit_status(make_session, fetched_root, monkeypatch, capsys):
    first, _ = make_session([" Hello!"], logs_dir=fetched_root / "logs")
    conversation = first.new_conversation("text")
    call_id = first.say("Hi").record["call_id"]
    replies = iter([[" Hello!"], [" Bye"]])

    def session(root, logs_dir):
        return make_session(next(replies), logs_dir=logs_dir)[0]

    monkeypatch.setattr(cli, "Session", session)
    argv = ["--root", str(fetched_root), "replay", str(conversation.log.path), call_id]
    assert cli.main(argv) == 0
    assert "Identical" in capsys.readouterr().out
    assert cli.main(argv) == 1
    assert "Different" in capsys.readouterr().out
