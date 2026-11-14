"""The command registry: one table that the terminal, --json and the web UI all use."""

from __future__ import annotations

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.profiles import ConfigError
from clarity_chat.session import CancelledBeforeReply, Session
from tests.fakes import FakeBackend


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!", " Again."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    return Commands(session), backend


def test_a_call_emits_its_text_and_every_record_it_logs(commands):
    c, _ = commands
    events: list[dict] = []
    started = c.run("profile", "text", events.append)
    assert started["profile"] == "text" and started["mode"] == "text"
    assert [e["status"] for e in events if e["event"] == "status"] == ["starting", "ready"]
    events.clear()
    result = c.run("say", "Hi", events.append)
    assert "".join(e["text"] for e in events if e["event"] == "text") == " Hello!"
    logged = [e["record"]["type"] for e in events if e["event"] == "record"]
    assert logged == ["turn", "call", "turn"]
    assert result["call"]["output_text"] == " Hello!"


def test_cancel_runs_during_a_call_and_stops_it(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    backend.on_call = lambda: c.run("cancel", "", lambda e: None)  # from another thread, say
    with pytest.raises(CancelledBeforeReply):
        c.run("say", "Hi", lambda e: None)
    nothing = {"cancelled": False, "reason": "Nothing is running."}
    assert c.run("cancel", "", lambda e: None) == nothing


def test_cancel_says_when_the_running_command_cant_be_stopped(commands, fetched_root):
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    answers = []
    backend.start_warnings = []
    # A model start can't be cancelled: /profile to another context size restarts it.
    (fetched_root / "profiles" / "big.toml").write_text(
        "ctx_size = 4096\n" + (fetched_root / "profiles" / "text.toml").read_text("utf-8"),
        encoding="utf-8",
    )
    original = backend.start

    def start(*args):
        answers.append(c.run("cancel", "", lambda e: None))
        return original(*args)

    backend.start = start
    c.run("profile", "big", lambda e: None)
    assert answers == [{"cancelled": False, "reason": "/profile can't be cancelled."}]


def test_commands_need_a_conversation_except_a_few(commands):
    c, _ = commands
    with pytest.raises(ConfigError, match="No conversation is running"):
        c.run("say", "Hi", lambda e: None)
    assert c.run("help", "", lambda e: None)["commands"][0]["usage"] == "/say TEXT"
    with pytest.raises(ConfigError, match="Unknown command /nope"):
        c.run("nope", "", lambda e: None)


def test_one_command_at_a_time_and_an_early_cancel_still_lands(commands):
    c, backend = commands
    c.run("profile", "text", lambda e: None)

    def meanwhile():
        with pytest.raises(ConfigError, match="Another command is still running"):
            c.run("undo", "", lambda e: None)
        c.run("cancel", "", lambda e: None)

    backend.on_call = meanwhile
    with pytest.raises(CancelledBeforeReply):
        c.run("say", "Hi", lambda e: None)
    backend.on_call = None
    assert c.run("say", "Hi", lambda e: None)["call"]["output_text"] == " Again."


def test_regen_n_makes_n_branches_with_consecutive_seeds(commands):
    c, backend = commands
    backend.replies = [" first", " a", " b", " c"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    events: list[dict] = []
    result = c.run("regen", "3", events.append)
    starts = [e for e in events if e["event"] == "reply_start"]
    assert [(e["k"], e["n"], e["seed"]) for e in starts] == [(1, 3, 43), (2, 3, 44), (3, 3, 45)]
    calls = [r["call"] for r in result["replies"]]
    assert [call["settings"]["seed"]["value"] for call in calls] == [43, 44, 45]
    assert [call["output_text"] for call in calls] == [" a", " b", " c"]
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.turns[conversation.head].text == " c"  # the head is the last reply
    assert "stopped" not in result


def test_plain_regen_returns_a_list_of_one(commands):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    result = c.run("regen", "", lambda e: None)
    assert len(result["replies"]) == 1 and result["replies"][0]["call"]["regen_of"]


@pytest.mark.parametrize("args", ["0", "-1", "two", "1.5", "²", "٣"])
def test_regen_n_must_be_a_positive_whole_number(commands, args):
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    with pytest.raises(ConfigError, match=r"/regen \[N\]"):
        c.run("regen", args, lambda e: None)


def test_a_failure_after_the_first_reply_stops_the_rest_and_keeps_the_replies(commands):
    c, backend = commands
    backend.replies = [" first", " a", " b"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    calls = []

    def fail_on_second():
        calls.append(1)
        if len(calls) == 2:
            backend.fail_with = "server gone"

    original = backend.complete

    def complete(*args, **kwargs):
        fail_on_second()
        return original(*args, **kwargs)

    backend.complete = complete
    result = c.run("regen", "3", lambda e: None)
    assert [r["call"]["output_text"] for r in result["replies"]] == [" a"]
    assert result["stopped"].startswith("Reply 2 of 3: server gone")
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.turns[conversation.head].text == " a"


def test_a_cancel_stops_the_reply_being_written_and_the_rest(commands):
    c, backend = commands
    backend.replies = [" first", " a", " longer reply", " c"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    calls = []

    def cancel_the_second():
        calls.append(1)
        if len(calls) == 2:
            backend.cancel_after = 3
            c.run("cancel", "", lambda e: None)

    backend.on_call = cancel_the_second
    result = c.run("regen", "3", lambda e: None)
    calls_made = [r["call"] for r in result["replies"]]
    assert [call["stop_type"] for call in calls_made] == ["eos", "cancelled"]
    assert len(backend.calls) == 3  # say, then two regenerations: the third never started


@pytest.mark.parametrize("how", ["failure", "cancel"])
def test_a_first_reply_that_fails_or_is_cancelled_before_it_starts_raises(commands, how):
    """With no reply made, there's nothing to return: the error reaches the caller, so
    --json sends an error event rather than an empty list."""
    c, backend = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    if how == "failure":
        backend.fail_with = "server gone"
    else:
        backend.on_call = lambda: c.run("cancel", "", lambda e: None)
    with pytest.raises(BackendError) as raised:
        c.run("regen", "3", lambda e: None)
    assert isinstance(raised.value, CancelledBeforeReply) == (how == "cancel")


def test_a_cancel_before_a_later_reply_starts_ends_like_any_other_cancel(commands):
    """Cancelled before reply 2's first token: the same result as a cancel mid-reply or
    between replies, with no `stopped`, and the head at the last completed reply."""
    c, backend = commands
    backend.replies = [" first", " a", " b", " c"]
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    calls = []

    def cancel_before_the_second():
        calls.append(1)
        if len(calls) == 2:
            c.run("cancel", "", lambda e: None)

    backend.on_call = cancel_before_the_second
    result = c.run("regen", "3", lambda e: None)
    assert "stopped" not in result
    assert [r["call"]["output_text"] for r in result["replies"]] == [" a"]
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.turns[conversation.head].text == " a"


def test_commands_that_parse_names_ignore_edge_spaces_from_scripts(commands):
    """--json passes args as sent; a leading space mustn't make an empty setting name or a
    label /unmark can't find."""
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    c.run("set", " temperature 0.5 ", lambda e: None)
    conversation = c.session.conversation
    assert conversation is not None and conversation.chat_settings["temperature"] == 0.5
    marked = c.run("mark", " landed ", lambda e: None)["records"]
    assert marked[0]["label"] == "landed"
    assert c.run("unmark", "landed", lambda e: None)["records"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("profile", "PROFILES/text.toml"),
        ("profile", "../profiles/text"),
        ("resume", "LOG"),
        ("resume", "../logs/LOGNAME"),
    ],
)
def test_a_command_takes_names_and_ids_not_paths(commands, fetched_root, name, args):
    """From v0.3.0 the browser sends commands too, and the trust statement counts them as
    untrusted: a path could make Clarity read any .toml or append to any log. The command
    line's --profile and --resume, typed by you, keep their paths."""
    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    log = c.session.conversation.log.path
    args = args.replace("PROFILES", str(fetched_root / "profiles")).replace("LOGNAME", log.stem)
    with pytest.raises(ConfigError, match="not a path"):
        c.run(name, args.replace("LOG", str(log)), lambda e: None)


def test_the_command_lines_paths_still_work(commands, fetched_root):
    c, _ = commands
    profile = str(fetched_root / "profiles" / "text.toml")
    assert c.run("profile", profile, lambda e: None, paths=True)["profile"] == "text"
    c.run("say", "Hi", lambda e: None)
    log = c.session.conversation.log.path
    c.run("resume", str(log), lambda e: None, paths=True)
    assert c.run("resume", log.stem, lambda e: None)["conversation_id"] == log.stem


def test_a_name_is_never_looked_up_in_the_working_folder(
    commands, fetched_root, tmp_path, monkeypatch
):
    """A typed name or ID means profiles/ or local/profiles/, and logs/; not a file that
    happens to sit in the folder Clarity runs from."""
    import shutil

    c, _ = commands
    c.run("profile", "text", lambda e: None)
    c.run("say", "Hi", lambda e: None)
    log = c.session.conversation.log.path
    here = tmp_path / "cwd"
    here.mkdir()
    shutil.copy(fetched_root / "profiles" / "text.toml", here / "evil.toml")
    shutil.copy(log, here / "planted")
    monkeypatch.chdir(here)
    with pytest.raises(ConfigError, match="not a path"):
        c.run("profile", "evil.toml", lambda e: None)
    with pytest.raises(ConfigError, match="No conversation"):
        c.run("resume", "planted", lambda e: None)


def test_new_after_a_resume_doesnt_read_a_logged_profile_outside_the_project(
    fetched_root, tmp_path_factory, tiny_tokenizer
):
    """A log is untrusted: its profile path can name any .toml file. /new starts the logged
    profile by name from the project instead."""
    import shutil

    outside = tmp_path_factory.mktemp("outside")  # not under fetched_root
    shutil.copy(fetched_root / "profiles" / "text.toml", outside / "text.toml")

    def commands():
        backend = FakeBackend([" Hello!", " Again."], tokenizer=tiny_tokenizer)
        loader = lambda r, v: tiny_tokenizer  # noqa: E731
        return Commands(Session(fetched_root, backend=backend, tokenizer_loader=loader))

    first = commands()
    first.run("profile", str(outside / "text.toml"), lambda e: None, paths=True)
    first.run("say", "Hi", lambda e: None)
    conversation_id = first.session.conversation.conversation_id
    text = (outside / "text.toml").read_text(encoding="utf-8")
    (outside / "text.toml").write_text("private_key_name = 1\n" + text, encoding="utf-8")
    second = commands()
    second.run("resume", conversation_id, lambda e: None)
    started = second.run("new", "", lambda e: None)  # reads profiles/text.toml
    assert started["profile"] == "text"
    assert second.session.conversation.profile.path.parent == fetched_root / "profiles"


def test_profiles_lists_every_profile_before_any_conversation(commands, fetched_root):
    """Shipped and local, steered and not, fetched and not, and a broken file with its error."""
    c, _ = commands
    profiles, local = fetched_root / "profiles", fetched_root / "local" / "profiles"
    local.mkdir(parents=True)
    text = (profiles / "text.toml").read_text(encoding="utf-8")
    (local / "mine.toml").write_text('description = "Made up."\n' + text, encoding="utf-8")
    steered = text.replace('backend = "llama.cpp"', 'backend = "transformers"').replace(
        "[settings]", "[settings]\nsteering_scale = 4.0"
    )
    steering = f'\n[steering]\nfile = "local/vectors/calm.gguf"\nsha256 = "{"0" * 64}"\n'
    (profiles / "steered.toml").write_text(steered + steering, encoding="utf-8")
    (fetched_root / "models.toml").write_text(
        (fetched_root / "models.toml").read_text(encoding="utf-8")
        + '\n[other]\nroute = "convert"\nrepo = "made-up/Other-1B-base"\n'
        + f'revision = "{"1" * 40}"\ntokenizer_repo = "made-up/Other-1B"\n'
        + f'tokenizer_revision = "{"1" * 40}"\n',
        encoding="utf-8",
    )
    (profiles / "unfetched.toml").write_text(
        text.replace('model = "tiny"', 'model = "other"'), encoding="utf-8"
    )
    (profiles / "broken.toml").write_text('mode = "text"\nmodel = 3\n', encoding="utf-8")
    result = c.run("profiles", "", lambda e: None)
    assert c.session.conversation is None
    by_name = {p["name"]: p for p in result["profiles"]}
    assert by_name["text"] == {
        "name": "text",
        "model": "tiny",
        "model_name": "tiny",
        "backend": "llama.cpp",
        "description": None,
        "steered": False,
        "local": False,
        "fetched": True,
        "error": None,
    }
    assert by_name["mine"]["local"] and by_name["mine"]["description"] == "Made up."
    assert by_name["steered"]["steered"] and by_name["steered"]["backend"] == "transformers"
    # The model's name comes from its tokenizer's repo: a GGUF repo is often a repackager's.
    assert by_name["unfetched"]["model_name"] == "Other-1B"
    assert by_name["unfetched"]["fetched"] is False and by_name["unfetched"]["error"] is None
    assert "backend is required" in by_name["broken"]["error"]
    assert by_name["broken"]["model"] is None
    assert "tokenizer or the server" in result["not_checked"]
    names = [p["name"] for p in result["profiles"]]
    assert names == sorted(names[:-1]) + ["mine"]  # shipped by name, then your own


def test_profiles_lists_a_name_in_both_folders_as_an_error(commands, fetched_root):
    c, _ = commands
    (fetched_root / "local" / "profiles").mkdir(parents=True)
    text = (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (fetched_root / "local" / "profiles" / "text.toml").write_text(text, encoding="utf-8")
    errors = [p["error"] for p in c.run("profiles", "", lambda e: None)["profiles"]]
    assert sum(bool(e) and "both" in e for e in errors) == 2


@pytest.fixture
def three_replies(fetched_root, tiny_tokenizer):
    """A chat with three replies on one path: Hello, Again, Third."""
    backend = FakeBackend([" Hello!", " Again.", " Third."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    ids = [c.run("say", text, lambda e: None)["call"]["call_id"] for text in ("a", "b", "c")]
    return c, backend, ids


def run(c, line):
    name, _, args = line.partition(" ")
    return c.run(name, args, lambda e: None)


def test_annotations_name_an_earlier_reply_by_its_call_id(three_replies):
    c, _, (first, second, third) = three_replies
    (mark,) = run(c, f"mark {first} landed")["records"]
    assert (mark["call_id"], mark["label"]) == (first, "landed")
    (note,) = run(c, f"note {second} was better")["records"]
    assert (note["call_id"], note["text"]) == (second, "was better")
    (unmark,) = run(c, f"unmark {first}")["records"]
    assert unmark["call_id"] == first
    head = c.session.conversation.head
    assert c.session.conversation.turns[head].call_id == third  # nothing moved


def test_a_label_that_only_looks_like_a_call_id_is_a_label(three_replies):
    c, _, (_, _, third) = three_replies
    for label in ("call-me", "call-3f9a1b2c4d5", "call-3F9A1B2C4D5E", "call-3f9a1b2c4d5e7"):
        (mark,) = run(c, f"mark {label}")["records"]
        assert (mark["call_id"], mark["label"]) == (third, label)


@pytest.mark.parametrize("command", ["mark", "unmark", "note x", "prompt", "replay"])
def test_an_unknown_call_id_is_refused_never_read_as_text(three_replies, command):
    c, _, _ = three_replies
    name, _, rest = command.partition(" ")
    with pytest.raises(ConfigError, match="call-000000000000 isn't a call in this conversation"):
        c.run(name, f"call-000000000000 {rest}".strip(), lambda e: None)


@pytest.mark.parametrize("args", ["junk", "call-me"])
def test_replay_refuses_anything_but_a_call_id(three_replies, args):
    c, _, _ = three_replies
    with pytest.raises(ConfigError, match=r"Usage: /replay \[CALL_ID\]"):
        c.run("replay", args, lambda e: None)


def test_prompt_and_replay_read_an_earlier_call(three_replies):
    c, backend, (first, _, _) = three_replies
    assert run(c, f"prompt {first}")["call"]["call_id"] == first
    backend.replies = [" Hello!"]
    replay = run(c, f"replay {first}")["replay"]
    assert replay["call_id"] == first and replay["identical"]


def test_a_failed_call_can_be_read_by_its_id_but_not_marked_or_replayed(three_replies):
    c, backend, _ = three_replies
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        run(c, "say d")
    failed = c.session.conversation.last_call
    assert failed["error"] and not failed.get("turn_id")
    backend.fail_with = None
    run(c, "say e")  # the failed call is no longer the latest
    assert run(c, f"prompt {failed['call_id']}")["call"] == failed
    with pytest.raises(ConfigError, match="failed, so it has no reply to mark"):
        run(c, f"mark {failed['call_id']}")
    with pytest.raises(ConfigError, match="has no reply to compare with"):
        run(c, f"replay {failed['call_id']}")
    # After a resume, the same call reads the same.
    log = c.session.conversation.conversation_id
    run(c, f"resume {log}")
    assert run(c, f"prompt {failed['call_id']}")["call"] == failed


def test_a_reply_cancelled_mid_way_can_be_marked_by_its_id(three_replies):
    c, backend, _ = three_replies
    backend.cancel_after = 3
    cancelled = run(c, "say d")["call"]
    backend.cancel_after = None
    run(c, "say e")
    (mark,) = run(c, f"mark {cancelled['call_id']}")["records"]
    assert mark["call_id"] == cancelled["call_id"]


def test_profiles_lists_a_profile_that_isnt_utf8_with_its_error(commands, fetched_root):
    """Checkpoint review, finding 2: one bad file mustn't break the list."""
    c, _ = commands
    local = fetched_root / "local" / "profiles"
    local.mkdir(parents=True)
    (local / "latin.toml").write_bytes('description = "café"\n'.encode("latin-1"))
    by_name = {p["name"]: p for p in c.run("profiles", "", lambda e: None)["profiles"]}
    assert by_name["latin"]["error"] and by_name["text"]["error"] is None


@pytest.mark.parametrize("gap", ["\n", "\t", "  "])
def test_a_call_id_is_followed_by_any_whitespace(three_replies, gap):
    """Checkpoint review, finding 3: Alt+Enter or a tab after the ID still names the call."""
    c, _, (first, _, _) = three_replies
    (note,) = c.run("note", f"{first}{gap}was better", lambda e: None)["records"]
    assert (note["call_id"], note["text"]) == (first, "was better")


def test_replaying_a_cancelled_reply_by_its_id_is_refused(three_replies):
    """Review finding 8: the refusal goes through Session.replay; now tested by ID."""
    c, backend, _ = three_replies
    backend.cancel_after = 3
    cancelled = run(c, "say d")["call"]
    backend.cancel_after = None
    run(c, "say e")
    with pytest.raises(ConfigError, match="was cancelled, so its reply is incomplete"):
        run(c, f"replay {cancelled['call_id']}")
