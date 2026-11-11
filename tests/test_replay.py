"""Replay: comparisons, environment differences, and replays through the fake backend."""

from __future__ import annotations

import pytest

from clarity_chat import cli
from clarity_chat.cli import ChatLoop, format_replay
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.replay import (
    compare_output,
    deep_diff,
    environment_diff,
    not_recorded,
    unexplained,
)
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
    assert not compare_output(logged, call([1, 2], text="same", stop="limit"))["identical"]


def test_unexplained_only_when_nothing_that_matters_differs():
    assert unexplained(False, {})
    assert unexplained(False, {"client": [{}], "libraries": [{}]})
    assert not unexplained(False, {"build_info": [{}]})
    # The machine and PyTorch's environment variables can change a replay's output.
    assert not unexplained(False, {"machine": [{}]}) and not unexplained(False, {"env": [{}]})
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

    def session(root, logs_dir, on_server=None):
        return make_session(next(replies), logs_dir=logs_dir)[0]

    monkeypatch.setattr(cli, "Session", session)
    argv = ["--root", str(fetched_root), "replay", str(conversation.log.path), call_id]
    assert cli.main(argv) == 0
    assert "Identical" in capsys.readouterr().out
    assert cli.main(argv) == 1
    assert "Different" in capsys.readouterr().out


# --- review fixes -------------------------------------------------------------------------


def test_a_cancelled_replay_logs_nothing(make_session):
    session, backend = make_session([" Hello!", " Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    backend.cancel_after = 2
    with pytest.raises(ConfigError, match="cancelled"):
        session.replay_current()
    kinds = [r["type"] for r in read_records(session.conversation.log.path)]
    assert "replay" not in kinds


def test_the_replay_record_has_the_request_it_sent(make_session):
    session, _ = make_session([" Hello!", " Hello!"])
    session.new_conversation("text")
    outcome = session.say("Hi")
    record = session.replay_current()
    assert record["request"] == outcome.record["request"]
    assert "tokens_evaluated" in record and "backend_settings" in record


def test_text_comparison_reports_a_character_position():
    logged = call([1], text="abcd", flags=["incomplete_output_ids"])
    result = compare_output(logged, call([1], text="abXd"))
    assert result["first_different_char"] == 2 and "first_different_token" not in result


def test_a_missing_log_file_is_a_clear_error(tmp_path):
    from clarity_chat.replay import find_call

    with pytest.raises(ConfigError, match="not found"):
        find_call(tmp_path / "nope.jsonl", "call-x")


def test_transformers_calls_compare_torch_and_transformers_versions():
    from clarity_chat.replay import facts_from_call

    call = {"model": {"repo": "r", "revision": "v", "gguf_sha256": "a"}, "client": {}}
    server = {"backend": "transformers", "torch": "2.15.0", "transformers": "5.22.2"}
    logged = facts_from_call(call, server)
    now = facts_from_call(
        {**call, "model": {**call["model"], "gguf_sha256": "b"}}, {**server, "torch": "2.16.0"}
    )
    diffs = environment_diff(logged, now)
    assert set(diffs) == {"torch"}  # the GGUF, which Transformers never reads, isn't compared
    assert not unexplained(False, diffs)


def test_transformers_calls_compare_the_attention_implementation():
    from clarity_chat.replay import facts_from_call

    call = {"model": {"repo": "r", "revision": "v"}, "client": {}}
    server = {"backend": "transformers", "attn_implementation": "sdpa"}
    logged = facts_from_call(call, server)
    now = facts_from_call(call, {**server, "attn_implementation": "eager"})
    diffs = environment_diff(logged, now)
    assert set(diffs) == {"attn_implementation"}
    assert not unexplained(False, diffs)  # another attention kernel can explain another reply


def test_logged_flags_only_for_llama_server_calls(fetched_root, capsys):
    import json

    log = fetched_root / "logs" / "x.jsonl"
    log.parent.mkdir()
    rows = [
        {"type": "server_start", "server_id": "backend-1", "backend": "transformers"},
        {
            "type": "call",
            "call_id": "call-1",
            "conversation_id": "c",
            "backend": {"backend": "transformers", "server_id": "backend-1"},
            "output_ids": [1],
            "stop_type": "eos",
        },
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    argv = ["--root", str(fetched_root), "replay", str(log), "call-1", "--logged-flags"]
    assert cli.main(argv) == 1
    assert "only to calls made with llama-server" in capsys.readouterr().err


def test_llama_cpp_calls_compare_graph_splits():
    from clarity_chat.replay import facts_from_call

    call = {"model": {"gguf_sha256": "a"}, "client": {}}
    server = {"backend": "llama.cpp", "graph_splits": {"prompt": 1, "reply": 1}}
    now = {**server, "graph_splits": {"prompt": 2, "reply": 2}}
    diffs = environment_diff(facts_from_call(call, server), facts_from_call(call, now))
    assert set(diffs) == {"graph_splits"}
    assert not unexplained(False, diffs)  # work moved to another device can change a reply


def test_replay_uses_the_calls_logged_context_size(make_session, fetched_root, tmp_path):
    text = (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (fetched_root / "profiles" / "big.toml").write_text(f"ctx_size = 16384\n{text}", "utf-8")
    first, _ = make_session([" Hello!"], logs_dir=tmp_path)
    conversation = first.new_conversation("big")
    call_id = first.say("Hi").record["call_id"]

    in_chat, _ = make_session()  # a chat at server.toml's 8192 can't replay a 16384 call
    in_chat.new_conversation("text")
    with pytest.raises(ConfigError, match="context of 16384 tokens, but this conversation"):
        in_chat.replay(conversation.log.path, call_id)

    alone, backend = make_session([" Hello!"])  # `clarity replay`: no conversation running
    record = alone.replay(conversation.log.path, call_id)
    assert backend.ctx_sizes == [16384]
    assert record["identical"] and record["n_ctx_from"] == "the call's log"


def test_an_older_logs_server_compares_without_false_differences():
    from clarity_chat.replay import facts_from_call, not_recorded

    call = {"model": {"gguf_sha256": "a"}, "client": {}}
    v010 = {"backend": "llama.cpp", "props": {"endpoint_slots": True, "ui": True, "n": 1}}
    now = {
        "backend": "llama.cpp",
        "props": {"endpoint_slots": False, "ui": False, "n": 1},  # --no-slots, --no-webui
        "graph_splits": {"prompt": 1, "reply": 1},  # not logged before v0.2.0
    }
    logged, current = facts_from_call(call, v010), facts_from_call(call, now)
    assert environment_diff(logged, current) == {}
    assert not_recorded(logged, current) == ["graph_splits"]
    assert unexplained(False, environment_diff(logged, current))


def test_end_tokens_are_a_fact_on_each_backend():
    from clarity_chat.replay import facts_from_call

    call = {"model": {}, "client": {}}
    llama = {"backend": "llama.cpp", "model_file": {"eog_tokens": [{"id": 2}, {"id": 0}]}}
    hf = {"backend": "transformers", "eos_token_id": [2]}
    assert facts_from_call(call, llama)["end_tokens"] == [0, 2]
    assert facts_from_call(call, hf)["end_tokens"] == [2]
    fewer = {**llama, "model_file": {"eog_tokens": [{"id": 2}]}}
    diffs = environment_diff(facts_from_call(call, llama), facts_from_call(call, fewer))
    assert set(diffs) == {"end_tokens"}
    assert unexplained(False, diffs) is False  # another end token can change a reply


def test_end_tokens_an_older_log_lacks_are_listed_as_not_recorded():
    from clarity_chat.replay import facts_from_call

    call = {"model": {}, "client": {}}
    old = facts_from_call(call, {"backend": "llama.cpp", "model_file": {}})
    now = facts_from_call(call, {"backend": "llama.cpp", "model_file": {"eog_tokens": [{"id": 2}]}})
    assert "end_tokens" in not_recorded(old, now)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"no-context-shift": None}, "must set ['no-context-shift']"),
        ({"cache-reuse": 256}, "cache-reuse must be 0"),
        ({"parallel": 4}, "parallel must be 1"),
        ({"log-verbosity": 0}, "log-verbosity must be"),
        ({"flash-attn": "maybe"}, 'flash-attn must be "on" or "off"'),
        ({"kv-unified": True, "no-kv-unified": True}, "exactly one of kv-unified"),
    ],
)
def test_logged_flags_get_every_check_server_toml_gets(fetched_root, capsys, change, error):
    """Every Clarity release required these flags and values, so only a log Clarity didn't
    write fails; it would otherwise start a server with context shift or cache reuse on."""
    import json

    from clarity_chat.profiles import load_server_config

    log = fetched_root / "logs" / "x.jsonl"
    log.parent.mkdir()
    flags = {**load_server_config(fetched_root).flags, **change}
    flags = {k: v for k, v in flags.items() if v is not None}
    rows = [
        {"type": "server_start", "server_id": "server-1", "backend": "llama.cpp", "flags": flags},
        {
            "type": "call",
            "call_id": "call-1",
            "conversation_id": "c",
            "backend": {"backend": "llama.cpp", "server_id": "server-1"},
            "output_ids": [1],
            "stop_type": "eos",
        },
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    argv = ["--root", str(fetched_root), "replay", str(log), "call-1", "--logged-flags"]
    assert cli.main(argv) == 1
    assert error in capsys.readouterr().err


def test_logged_flags_outside_the_allowed_list_are_refused(fetched_root, capsys):
    """A shared or hand-edited log must not reach llama-server's argv: later flags win there,
    so `host`, `slots`, `webui`, `api-key` or `tools` would undo Clarity's own."""
    import json

    from clarity_chat.profiles import load_server_config

    log = fetched_root / "logs" / "x.jsonl"
    log.parent.mkdir()
    flags = {**load_server_config(fetched_root).flags, "host": "0.0.0.0", "tools": "all"}
    rows = [
        {"type": "server_start", "server_id": "server-1", "backend": "llama.cpp", "flags": flags},
        {
            "type": "call",
            "call_id": "call-1",
            "conversation_id": "c",
            "backend": {"backend": "llama.cpp", "server_id": "server-1"},
            "output_ids": [1],
            "stop_type": "eos",
        },
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    argv = ["--root", str(fetched_root), "replay", str(log), "call-1", "--logged-flags"]
    assert cli.main(argv) == 1
    assert "'host' isn't in Clarity's allowed list" in capsys.readouterr().err


def test_a_failed_server_restart_in_replay_leaves_the_chat_usable(
    fetched_root, tiny_tokenizer, monkeypatch
):
    """The server died, /replay's restart fails: the next message must not crash on the
    failed server's empty record (code review, verifier E)."""
    from clarity_chat.backends import BackendError, Result, llama_cpp
    from clarity_chat.profiles import load_server_config

    fail = {"on": False}

    def start(server):
        if fail["on"]:
            raise ConfigError("stale build (simulated)")
        server.proc = type("Running", (), {"poll": lambda self: None})()
        server.record = {"type": "server_start", "server_id": server.server_id}
        return server.record

    def complete(self, token_ids, settings, stop, on_text=None, request=None, cancel=None):
        ids = tiny_tokenizer.encode(" ok")
        return Result(text=" ok", token_ids=ids, stop_type="eos", n_prompt=len(token_ids))

    monkeypatch.setattr(llama_cpp.LlamaServer, "start", start)
    monkeypatch.setattr(llama_cpp.LlamaServer, "stop", lambda server: None)
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "complete", complete)
    backend = llama_cpp.LlamaCppBackend(load_server_config(fetched_root), fetched_root / "logs")
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("text")
    session.say("Hi")
    assert backend.server is not None
    exited = type("Exited", (), {"poll": lambda self: 1})()
    monkeypatch.setattr(backend.server, "proc", exited)  # the server died
    fail["on"] = True
    with pytest.raises((ConfigError, BackendError)):
        session.replay_current()
    fail["on"] = False
    try:
        session.say("Again")
    except (ConfigError, BackendError):
        pass  # said plainly; a KeyError would be the crash


@pytest.mark.parametrize("backend", ["llama.cpp", "transformers"])
def test_a_call_whose_server_record_is_missing_lists_facts_as_not_recorded(backend):
    """A trimmed or hand-edited log: the call's backend picks the facts, and what the log
    lacks is not recorded, not a difference that hides an unexplained one."""
    from clarity_chat.replay import facts_from_call

    call = {"model": {"repo": "r", "revision": "v"}, "client": {}, "backend": {"backend": backend}}
    now_server = (
        {"backend": "llama.cpp", "props": {"n_ctx": 8192}, "llama_cpp": {"commit": "c"}}
        if backend == "llama.cpp"
        else {"backend": "transformers", "torch": "2", "transformers": "5"}
    )
    logged = facts_from_call(call, None)
    now = facts_from_call(call, now_server)
    diffs = environment_diff(logged, now)
    assert diffs == {}, diffs
    assert unexplained(False, diffs)
    assert "weights" not in not_recorded(logged, now)


def test_a_replay_whose_text_differs_from_its_ids_logs_the_decoded_text(make_session):
    """`text_differs_from_ids` promises `decoded_output_text` beside it, in a replay too."""
    session, backend = make_session([" Hello!", " Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    backend.tokenizer = None  # the fake then returns IDs that don't decode to its text
    record = session.replay_current()
    assert "text_differs_from_ids" in record["flags"]
    assert "decoded_output_text" in record


def test_a_machine_value_unknown_on_one_side_is_not_recorded_not_a_difference():
    """A sandbox hides the chip's name (None), and a newer Clarity may record a new machine
    key. Neither is a difference, so a replay that differs is still flagged unexplained."""
    from clarity_chat.replay import environment_diff, not_recorded

    machine = {"os": "Darwin", "macos": {"chip": "Apple M2 Max", "build": "25G332"}}
    for logged, now in [
        (machine, {**machine, "macos": {**machine["macos"], "chip": None}}),
        ({**machine, "macos": {**machine["macos"], "chip": None}}, machine),
        ({"os": "Darwin", "macos": machine["macos"]}, {**machine, "gpu_cores": 30}),
    ]:
        diffs = environment_diff({"machine": logged}, {"machine": now})
        assert diffs == {} and unexplained(False, diffs)
        missing = not_recorded({"machine": logged}, {"machine": now})
        assert any(name.startswith("machine.") for name in missing), missing
    other = {**machine, "macos": {**machine["macos"], "chip": "Apple M3"}}
    assert "machine" in environment_diff({"machine": machine}, {"machine": other})
