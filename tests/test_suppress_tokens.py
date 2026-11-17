"""v0.4.0 A1: the model file's banned tokens (`tokenizer.ggml.suppress_tokens`), read from
llama-server's startup log, logged, shown by /settings and checked after each reply.

The startup logs are written in llama.cpp's format (tests/fake_llama.py). Made-up text only.
"""

from __future__ import annotations

import json

import pytest

from clarity_chat import cli, commands
from clarity_chat.backends import llama_cpp
from clarity_chat.backends.llama_cpp import LlamaCppBackend
from clarity_chat.export import export_log
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError, load_server_config
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE, MODELS, fake_fetch
from tests.fake_llama import (
    PREFIX,
    SUPPRESS_KEY,
    array_value,
    banning,
    dump,
    fake_llama,
    kv_line,
    startup,
)

BANNED = [258883, 258882]  # gemma-4-12B-it's list (the plan's Shapes)
REFUSAL = "llama-server's startup log doesn't give this model file's whole list"


def parse(text: str):
    return llama_cpp.suppress_tokens(text)


def refused(text: str) -> None:
    with pytest.raises(ConfigError, match=REFUSAL):
        parse(text)


# --- the parser (decision 2) ----------------------------------------------------------------


def test_a_dump_without_the_key_bans_nothing():
    assert parse(dump()) == []


def test_the_listed_ids_are_read():
    assert parse(dump(banning(BANNED))) == BANNED


def test_a_negative_id_is_kept_as_listed():
    # llama.cpp drops IDs outside the vocabulary; they can't be in a reply either.
    assert parse(dump(banning([-1, 5]))) == [-1, 5]


def test_an_empty_list_bans_nothing():
    assert parse(dump((SUPPRESS_KEY, "arr[i32,0]", "[]"))) == []


def test_the_forty_character_rule():
    """Brackets and separators count: five 6-digit IDs make 40 characters and fit; six
    5-digit IDs make 42 and are cut."""
    five = [100000, 100001, 100002, 100003, 100004]
    assert len(array_value(five)) == 40 and parse(dump(banning(five))) == five
    six = [10000, 10001, 10002, 10003, 10004, 10005]
    assert array_value(six).endswith("...")
    refused(dump(banning(six)))


def test_a_cut_list_is_refused():
    seven = [12345 + i for i in range(7)]
    line = banning(seven)
    assert line[1] == "arr[i32,7]" and line[2].endswith("...")
    refused(dump(line))


def test_a_value_over_forty_characters_is_refused():
    value = "[" + ", ".join(["1234567"] * 5) + "]"  # 45 characters, whole
    refused(dump((SUPPRESS_KEY, "arr[i32,5]", value)))


def test_fewer_ids_than_the_count_is_refused():
    refused(dump((SUPPRESS_KEY, "arr[i32,3]", "[258883, 258882]")))


def test_another_type_is_refused():
    refused(dump((SUPPRESS_KEY, "arr[i64,2]", "[258883, 258882]")))


def test_two_lines_naming_the_key_are_refused():
    refused(dump(banning(BANNED), banning([7])))


def test_a_fake_type_line_before_the_key_does_not_hide_it():
    """Decision 8's invariant: the whole output is searched, so text from the file that
    prints a fake end of the dump can't hide the real key line."""
    fake_name = f"x\n{PREFIX}llama_model_loader: - type  f32:   61 tensors\n{PREFIX}y"
    assert parse(dump((fake_name, "str", "z"), banning(BANNED))) == BANNED


def test_no_key_line_at_all_is_refused_even_with_the_dump_header():
    text = startup()
    without = "".join(line for line in text.splitlines(True) if " - kv " not in line)
    assert "Dumping metadata keys/values" in without
    refused(without)
    refused("")


def test_the_key_name_is_right_aligned_in_42():
    line = kv_line(31, SUPPRESS_KEY, "arr[i32,2]", "[258883, 258882]")
    assert len(SUPPRESS_KEY) == 30 and f": {' ' * 12}{SUPPRESS_KEY} " in line
    assert parse(dump(banning(BANNED))) == BANNED


def test_the_line_is_read_with_or_without_the_logs_prefix():
    """llama-server's own lines carry a timestamp and level (seen at the pin); a log with
    the prefix turned off has none."""
    line = kv_line(32, SUPPRESS_KEY, "arr[i32,2]", "[28, 1234]")
    assert line.startswith("0.00.062.265 I llama_model_loader: - kv  32: ")
    assert parse(dump(banning(BANNED)).replace(PREFIX, "")) == BANNED
    assert parse(dump(banning(BANNED))) == BANNED


def test_windows_line_ends():
    assert parse(dump(banning(BANNED)).replace("\n", "\r\n")) == BANNED


def test_the_key_inside_another_keys_value_is_not_a_key_line():
    inside = kv_line(9, SUPPRESS_KEY, "arr[i32,1]", "[5]").strip()
    assert parse(dump(("general.name", "str", inside))) == []


# --- the server start and the call (the backend) -----------------------------------------


def started(monkeypatch, root, text):
    from tests.test_llama_cpp import started_server

    return started_server(monkeypatch, root, text)


def test_every_server_start_records_the_list(monkeypatch, root):
    from tests.test_llama_cpp import STARTUP_GPU

    _, _, record = started(monkeypatch, root, STARTUP_GPU + dump(banning(BANNED)))
    assert json.loads(record)["model_file"].get("suppress_tokens") == BANNED
    _, _, record = started(monkeypatch, root, STARTUP_GPU + dump())
    assert json.loads(record)["model_file"].get("suppress_tokens") == []


def test_a_list_the_log_cuts_refuses_the_start_and_stops_the_server(monkeypatch, root, tmp_path):
    from tests.test_llama_cpp import STARTUP_GPU

    config = load_server_config(root)
    server = llama_cpp.LlamaServer(config, root / "model.gguf", root / "logs")
    stopped = []
    monkeypatch.setattr(server, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(llama_cpp, "llama_config_files", lambda: [])
    monkeypatch.setattr(llama_cpp, "git_state", lambda d: {"commit": config.llama_cpp_commit})
    monkeypatch.setattr(llama_cpp, "server_binary", lambda d: tmp_path / "llama-server")
    seven = [12345 + i for i in range(7)]

    def launch(argv, env):
        server.output = [STARTUP_GPU + dump(banning(seven))]

    monkeypatch.setattr(server, "_launch", launch)
    props = {"build_info": "b1-" + config.llama_cpp_commit[:7]}
    monkeypatch.setattr(server, "get_props", lambda: props)
    monkeypatch.setattr(server, "check_props", lambda p: [])
    with pytest.raises(ConfigError, match=REFUSAL):
        server.start()
    assert stopped == [True]


def call_on(monkeypatch, root, banned, reply_ids):
    from tests.test_llama_cpp import FINAL, SETTINGS, fake_backend

    backend = fake_backend(monkeypatch, root, [{"content": "Hi", "tokens": reply_ids}, FINAL])
    backend.server.record = {"model_file": {"suppress_tokens": banned}}
    return backend.complete([1, 2], SETTINGS, [])


def test_a_call_on_a_banning_file_logs_the_ban_and_checks_it(monkeypatch, root):
    result = call_on(monkeypatch, root, BANNED, [5])
    assert result.backend_settings.get("suppress_tokens") == {
        "value": BANNED,
        "origin": "model file",
    }
    assert result.checks.get("suppress_tokens") == {"ok": True}
    assert "check_failed:suppress_tokens" not in result.flags


def test_a_banned_token_in_the_reply_fails_the_check(monkeypatch, root):
    result = call_on(monkeypatch, root, BANNED, [5, 258882])
    assert result.checks.get("suppress_tokens") == {"ok": False}
    assert "check_failed:suppress_tokens" in result.flags


def test_a_call_on_a_file_that_bans_nothing_has_no_entry_and_no_check(monkeypatch, root):
    result = call_on(monkeypatch, root, [], [5])
    assert "suppress_tokens" not in result.backend_settings
    assert "suppress_tokens" not in result.checks


# --- a session on the real backend, with llama-server faked -------------------------------

MODELS_MORE = MODELS + "".join(
    MODELS.split("\n", 1)[1].join([f"\n[{name}]\n", ""]) for name in ("banned", "banned7")
)


def z_id(tokenizer) -> int:
    (token,) = tokenizer.encode("Z")
    return token


@pytest.fixture
def banned_root(fetched_root, chat_tokenizer):
    """Three models: tiny bans nothing, banned bans Z (and an ID past the vocabulary), and
    banned7's ban is too long for the log. Each lock lists suppress_tokens among the
    recommended values Clarity doesn't send, as `fetch` writes it for such a file."""
    (fetched_root / "models.toml").write_text(MODELS_MORE, encoding="utf-8")
    for name in ("banned", "banned7"):
        fake_fetch(fetched_root, name)
        (fetched_root / "profiles" / f"{name}.toml").write_text(
            CHAT_PROFILE.replace('model = "tiny"', f'model = "{name}"'),
            encoding="utf-8",
            newline="\n",
        )
    lock_path = fetched_root / "models.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for entry in lock["models"].values():
        entry["recommended"]["not_applied"]["suppress_tokens"] = [z_id(chat_tokenizer)]
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return fetched_root


@pytest.fixture
def llama(monkeypatch, banned_root, chat_tokenizer):
    """A session on the real llama.cpp backend: (session, what the fake server saw)."""
    ban = [z_id(chat_tokenizer), 258882]
    startups = {
        "tiny": startup(),
        "banned": startup(banning(ban)),
        "banned7": startup(banning([12345 + i for i in range(7)])),
    }
    seen = fake_llama(monkeypatch, banned_root, startups, chat_tokenizer, replies=[])
    seen["ban"] = ban
    backend = LlamaCppBackend(load_server_config(banned_root), banned_root / "logs")
    session = Session(banned_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    yield session, seen
    session.close()


def entry(view) -> dict | None:
    return view["backend_settings"].get("suppress_tokens")


def test_settings_show_the_ban_before_and_after_the_first_call(llama):
    session, seen = llama
    session.new_conversation("banned")
    view = commands.settings_view(session)
    assert entry(view) == {"value": seen["ban"], "origin": "model file"}
    assert "suppress_tokens" not in view["recommended"]["not_applied"]
    outcome = session.say("Hi")
    assert outcome.record["backend_settings"].get("suppress_tokens", {}).get("origin") == (
        "model file"
    )
    assert outcome.record["checks"].get("suppress_tokens") == {"ok": True}
    view = commands.settings_view(session)
    assert entry(view) == {"value": seen["ban"], "origin": "model file"}
    assert "suppress_tokens" not in view["recommended"]["not_applied"]
    shown = cli.format_settings(view)
    assert f"  suppress_tokens = {seen['ban']!r}  model file" in shown


def test_settings_on_a_file_that_bans_nothing_are_unchanged(llama):
    session, _ = llama
    session.new_conversation("chat")
    for _ in range(2):  # before and after a call
        view = commands.settings_view(session)
        assert entry(view) is None
        assert "suppress_tokens" in view["recommended"]["not_applied"]
        outcome = session.say("Hi")
        assert "suppress_tokens" not in outcome.record["backend_settings"]
        assert "suppress_tokens" not in outcome.record["checks"]


def test_settings_without_the_list_keep_the_not_applied_line(banned_root, chat_tokenizer):
    """A backend whose start records no list (Transformers; the fake stands in) leaves
    the model's recommended suppress_tokens listed as not applied."""
    from tests.fakes import FakeBackend

    session = Session(
        banned_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: chat_tokenizer
    )
    session.new_conversation("banned")
    view = commands.settings_view(session)
    assert entry(view) is None and "suppress_tokens" in view["recommended"]["not_applied"]


def test_settings_after_resuming_a_log_whose_calls_have_no_entry(llama):
    """A conversation begun before v0.4.0 on a banning file: /settings reads the running
    server, whatever the last call holds (decision 3)."""
    session, seen = llama
    session.new_conversation("banned")
    session.say("Hi")
    assert session.conversation is not None
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["type"] == "call":
            row["backend_settings"].pop("suppress_tokens", None)
            row["checks"].pop("suppress_tokens", None)
        if row["type"] == "server_start":
            row["model_file"].pop("suppress_tokens", None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    session.new_conversation("chat")
    session.resume(path)
    view = commands.settings_view(session)
    assert entry(view) == {"value": seen["ban"], "origin": "model file"}
    assert "suppress_tokens" not in view["recommended"]["not_applied"]


def test_a_banned_token_in_a_reply_is_flagged(llama):
    session, seen = llama
    session.new_conversation("banned")
    seen["replies"].append("Zap")
    record = session.say("Hi").record
    assert record["checks"].get("suppress_tokens") == {"ok": False}
    assert "check_failed:suppress_tokens" in record["flags"]


def test_every_server_start_in_the_log_records_the_list(llama):
    session, seen = llama
    session.new_conversation("chat")
    session.say("Hi")
    session.new_conversation("banned")  # /new and /profile start a server the same way
    session.say("Hi")
    assert session.conversation is not None
    starts = [r for r in read_records(session.conversation.log.path) if r["type"] == "server_start"]
    assert [s["model_file"].get("suppress_tokens") for s in starts] == [seen["ban"]]
    logs = sorted((session.root / "logs").glob("*.jsonl"))
    starts = [r for p in logs for r in read_records(p) if r["type"] == "server_start"]
    assert sorted(len(s["model_file"].get("suppress_tokens", "?")) for s in starts) == [0, 2]


def test_a_refused_start_stops_its_server_and_the_chat_goes_on(llama):
    session, seen = llama
    with pytest.raises(ConfigError, match=REFUSAL):
        session.new_conversation("banned7")
    assert seen["procs"][-1].terminated
    first = session.new_conversation("chat")
    with pytest.raises(ConfigError, match=REFUSAL):
        session.new_conversation("banned7")  # as /profile banned7 in a running chat
    # The refused server stopped, and the chat's own model started again (_restore).
    refused_proc, restarted = seen["procs"][-2:]
    assert refused_proc.terminated and not restarted.terminated
    assert session.conversation is first
    assert session.say("still here").record["output_text"] == " ok"


def test_a_replay_on_a_banning_file_carries_the_ban_and_the_check(llama):
    session, _ = llama
    session.new_conversation("banned")
    call = session.say("Hi").record
    assert session.conversation is not None
    path = session.conversation.log.path
    session.close()
    replayed = Session(
        session.root,
        backend=LlamaCppBackend(load_server_config(session.root), path.parent),
        tokenizer_loader=session.tokenizer_loader,
    )
    record = replayed.replay(path, call["call_id"])
    replayed.close()
    assert record["identical"]
    assert record["backend_settings"].get("suppress_tokens", {}).get("origin") == "model file"
    assert record["checks"].get("suppress_tokens") == {"ok": True}


def test_the_json_settings_result_has_the_ban(llama):
    from tests.test_json_chat import line, run

    session, seen = llama
    events = run(session, [line("profile", "banned"), line("settings")])
    (result,) = [e["result"] for e in events if e.get("command") == "settings"]
    assert result["backend_settings"].get("suppress_tokens") == {
        "value": seen["ban"],
        "origin": "model file",
    }


def test_a_batch_run_stops_on_a_banned_token(llama, banned_root):
    from clarity_chat.batch import RunError, run_batch

    session, seen = llama
    seen["replies"].append("Zap")
    (banned_root / "openings.txt").write_text("Hello there.\n", encoding="utf-8")
    run_dir = banned_root / "runs" / "run-test"
    session.logs_dir = run_dir
    with pytest.raises(RunError):
        run_batch(session, "run-test", run_dir, banned_root / "openings.txt", ["banned"], 1, print)
    calls = [r for r in read_records(run_dir / "calls.jsonl") if r["type"] == "call"]
    assert "check_failed:suppress_tokens" in calls[-1]["flags"]


def test_a_log_without_the_new_fields_resumes_exports_and_replays(llama):
    """T1 for v0.4.0's optional fields: a log before v0.4.0 has none of them."""
    session, _ = llama
    session.new_conversation("banned")
    call = session.say("Hi").record
    assert session.conversation is not None
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["type"] == "call":
            row["backend_settings"].pop("suppress_tokens", None)
            row["checks"].pop("suppress_tokens", None)
        if row["type"] == "server_start":
            row["model_file"].pop("suppress_tokens", None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    session.new_conversation("chat")
    session.resume(path)
    export_log(session, path)
    assert session.replay(path, call["call_id"])["identical"]
