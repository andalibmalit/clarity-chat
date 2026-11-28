"""The chat keeps its steering in force on llama.cpp (v0.4.7 B4; decisions 5 and 10; states
3 to 3c, 10a and 10b).

llama.cpp adds a vector when the model starts, so a llama.cpp call runs only on a server
started with exactly its steering: a new strength starts the model again, a resume and a
replay start the server their call needs, and a replay in a chat that runs other steering
is refused. The real llama.cpp backend on the fake llama-server (tests/fake_llama.py); the
live tier runs the same path on SmolLM2 (tests/test_live.py).
"""

from __future__ import annotations

from typing import Any

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.backends.llama_cpp import LlamaCppBackend
from clarity_chat.commands import Commands
from clarity_chat.logs import LogFile, read_records
from clarity_chat.profiles import ConfigError, load_server_config
from tests.fake_llama import fake_llama, startup
from tests.test_meaning import raised
from tests.test_steering import directions, sha, steer_root, steered, write
from tests.test_steering_llama_cpp import model_like

CALM = "local/vectors/calm.gguf"  # steer_root's vector: blocks 1 and 2, width 8
HOW = "llama.cpp applies a vector when the model starts."


def backend_on(root) -> LlamaCppBackend:
    return LlamaCppBackend(load_server_config(root), root / "logs")


def session_on(root, tiny_tokenizer, backend=None):
    from clarity_chat.session import Session

    backend = backend or backend_on(root)
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)


@pytest.fixture
def chat(monkeypatch, root, tiny_tokenizer):
    """The command table over a session on the real llama.cpp backend and the fake server,
    whose tiny model is a made-up llama file of width 8 and 4 blocks: (commands, what the
    fake saw). The profile `steered` steers with calm.gguf at ×4."""
    steer_root(root)
    model_like(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    session = session_on(root, tiny_tokenizer)
    steered(root, backend="llama.cpp")
    yield Commands(session), seen
    session.close()


def quiet(event: dict[str, Any]) -> None:
    pass


def scales(seen) -> list[float]:
    """The scale each start passed to llama-server, 0 for a start with no vector."""
    out = []
    for argv in seen["argv"]:
        if "--control-vector-scaled" in argv:
            out.append(float(argv[argv.index("--control-vector-scaled") + 1].rpartition(":")[2]))
        else:
            out.append(0.0)
    return out


def statuses(events) -> list[str]:
    return [e["status"] for e in events if e["event"] == "status"]


def each_call_ran_on_its_server(path) -> int:
    """Every call's request.steering equals its server's steering; returns the call count."""
    rows = list(read_records(path))
    servers = {r["server_id"]: r for r in rows if r["type"] == "server_start"}
    calls = [r for r in rows if r["type"] == "call"]
    for call in calls:
        server = servers[call["backend"]["server_id"]]
        assert call["request"]["steering"] == server["steering"], call["call_id"]
    return len(calls)


def say(c: Commands, text: str = "Hi") -> dict[str, Any]:
    """The call's record; a refused call fails here, on this assertion."""
    out: dict[str, Any] = {}
    refused = raised(lambda: out.update(c.run("say", text, quiet)))
    assert refused is None, f"the call was refused: {refused!r}"
    return out["call"]


# --- /set steering_scale (decision 5; states 3 to 3c) ----------------------------------


def test_a_new_strength_starts_the_model_again_and_the_chat_keeps_its_turns(chat):
    c, seen = chat
    c.run("profile", "steered", quiet)
    say(c)
    conversation = c.session.conversation
    assert conversation is not None
    turns = dict(conversation.turns)
    events: list[dict] = []
    result = c.run("set", "steering_scale 2.0", events.append)
    assert statuses(events) == ["starting", "ready"]
    assert scales(seen) == [4.0, 2.0]
    assert result["setting"] == {"value": 2.0, "origin": "chat"}
    assert c.session.conversation is conversation and conversation.turns == turns
    assert say(c, "Again")["request"]["steering"]["scale"] == 2.0
    assert each_call_ran_on_its_server(conversation.log.path) == 2


def test_on_transformers_a_new_strength_starts_nothing_and_says_nothing(root, tiny_tokenizer):
    from tests.fakes import FakeBackend

    steer_root(root)
    backend = FakeBackend([" Hi."] * 4, tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    c = Commands(session_on(root, tiny_tokenizer, backend))
    c.run("profile", steered(root), quiet)
    events: list[dict] = []
    c.run("set", "steering_scale 2.0", events.append)
    assert statuses(events) == []
    assert len(backend.started) == 1
    say(c)
    assert backend.calls[-1]["steering"].scale == 2.0


@pytest.mark.parametrize("command", ["new", "replay"])
def test_on_transformers_a_start_that_fails_after_a_new_strength_keeps_the_chat(
    root, tiny_tokenizer, command
):
    """Review finding 7: on Transformers the strength isn't part of the start, so a failed
    /new on the same profile, or a replay of a call made at the profile's strength, is the
    chat's own start: the chat stays, its model stopped, as without the /set."""
    from tests.fakes import FakeBackend

    steer_root(root)
    backend = FakeBackend([" Hi."] * 4, tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    c = Commands(session_on(root, tiny_tokenizer, backend))
    c.run("profile", steered(root), quiet)
    first = say(c)["call_id"]  # at the profile's ×4
    c.run("set", "steering_scale 2.0", quiet)
    conversation = c.session.conversation
    backend.stopped = True

    def refused(*args, **kwargs):
        raise BackendError("The model couldn't load.")

    backend.start = refused
    args = first if command == "replay" else ""
    caught = raised(lambda: c.run(command, args, quiet))
    assert isinstance(caught, BackendError), f"not refused: {caught!r}"
    assert "failed too" not in str(caught), str(caught)
    assert c.session.conversation is conversation


def test_another_setting_or_the_strength_in_force_starts_nothing(chat):
    c, seen = chat
    c.run("profile", "steered", quiet)
    events: list[dict] = []
    for args in ("temperature 0.5", "steering_scale 4.0", "steering_scale 4"):
        c.run("set", args, events.append)
    assert statuses(events) == []
    assert scales(seen) == [4.0]


def test_a_new_strength_sent_during_a_reply_waits_for_it(chat):
    c, seen = chat
    c.run("profile", "steered", quiet)
    during: list[Any] = []

    def meanwhile(event: dict[str, Any]) -> None:
        if event["event"] == "text" and not during:
            during.append(c.at_once_now("set"))
            during.append(raised(lambda: c.run("set", "steering_scale 2.0", quiet)))
            during.append(len(seen["argv"]))

    c.run("say", "Hi", meanwhile)
    waited, refused, starts = during
    assert waited is False and isinstance(refused, ConfigError) and starts == 1
    assert "Another command is still running" in str(refused)
    c.run("set", "steering_scale 2.0", quiet)
    assert scales(seen) == [4.0, 2.0]


def test_a_new_strength_that_cant_start_leaves_the_old_one_running(chat):
    c, seen = chat
    c.run("profile", "steered", quiet)
    say(c)
    seen["cvec"] = "fails"  # the new start's log shows no vector (state 11) ...
    seen["during_start"] = lambda: seen.update(cvec="loads", during_start=None)  # ... once
    events: list[dict] = []
    refused = raised(lambda: c.run("set", "steering_scale 2.0", events.append))
    assert isinstance(refused, BackendError), f"not refused: {refused!r}"
    assert scales(seen) == [4.0, 2.0, 4.0], "the old strength wasn't started again"
    assert statuses(events) == ["starting", "ready"]
    conversation = c.session.conversation
    assert conversation is not None, "the chat was stopped"
    assert conversation.settings["steering_scale"].value == 4.0
    assert "steering_scale" not in conversation.chat_settings
    assert say(c, "Again")["request"]["steering"]["scale"] == 4.0
    assert each_call_ran_on_its_server(conversation.log.path) == 2


def test_reset_all_with_a_changed_strength_starts_the_model_once(chat):
    """Decision 10: the page's Reset all is one /set per changed setting, Strength kept."""
    c, seen = chat
    c.run("profile", "steered", quiet)
    for args in ("temperature 0.5", "steering_scale 2.0", "top_k 5"):
        c.run("set", args, quiet)
    assert scales(seen) == [4.0, 2.0]
    for name in ("temperature", "steering_scale", "top_k"):
        c.run("set", name, quiet)
    assert scales(seen) == [4.0, 2.0, 4.0]
    conversation = c.session.conversation
    assert conversation is not None and conversation.chat_settings == {}
    assert conversation.settings == conversation.profile.settings


# --- resume ----------------------------------------------------------------------------


def test_a_resume_starts_at_the_chat_s_last_strength(chat):
    c, seen = chat
    root = c.session.root
    c.run("profile", "steered", quiet)
    say(c)
    c.run("set", "steering_scale 2.0", quiet)
    live = say(c, "Again")
    conversation = c.session.conversation
    assert conversation is not None
    path, settings = conversation.log.path, conversation.settings
    c.run("profile", steered(root, "plain", steering="", scale=None, backend="llama.cpp"), quiet)
    refused = raised(lambda: c.run("resume", conversation.conversation_id, quiet))
    assert refused is None, f"the resume was refused: {refused!r}"
    assert scales(seen)[-1] == 2.0
    resumed = c.session.conversation
    assert resumed is not None and resumed.settings == settings
    assert say(c, "Once more")["request"]["steering"] == live["request"]["steering"]
    assert each_call_ran_on_its_server(path) == 3


# --- replay in a chat (decision 5; states 10a and 10b) ---------------------------------


def test_a_replay_at_another_strength_says_which_to_set_and_sends_nothing(chat):
    c, seen = chat
    c.run("profile", "steered", quiet)
    call = say(c)
    c.run("set", "steering_scale 2.0", quiet)
    sent = len(seen["requests"])
    refused = raised(lambda: c.session.replay_current(call=call))
    assert str(refused) == (
        f"Call {call['call_id']} was steered ×4.0, but this chat's model runs at ×2.0: {HOW} "
        "Set steering_scale to 4.0 to replay it here."
    )
    assert len(seen["requests"]) == sent
    c.run("set", "steering_scale 4.0", quiet)
    seen["replies"].append(call["output_text"])
    assert c.session.replay_current(call=call)["identical"]


def test_the_strength_sentence_says_unsteered_at_0(chat):
    c, _ = chat
    c.run("profile", "steered", quiet)
    steered_call = say(c)
    c.run("set", "steering_scale 0", quiet)
    plain_call = say(c, "Again")
    refused = raised(lambda: c.session.replay_current(call=steered_call))
    assert str(refused) == (
        f"Call {steered_call['call_id']} was steered ×4.0, but this chat's model runs "
        f"unsteered: {HOW} Set steering_scale to 4.0 to replay it here."
    )
    c.run("set", "steering_scale 4.0", quiet)
    refused = raised(lambda: c.session.replay_current(call=plain_call))
    assert str(refused) == (
        f"Call {plain_call['call_id']} wasn't steered, but this chat's model runs at ×4.0: "
        f"{HOW} Set steering_scale to 0 to replay it here."
    )


def test_a_replay_with_other_steering_than_the_chat_can_run_says_to_resume_its_conversation(chat):
    c, seen = chat
    root = c.session.root
    c.run("profile", "steered", quiet)
    call = say(c)
    conversation = c.session.conversation
    assert conversation is not None
    path = conversation.log.path
    other = write(root / "local" / "vectors" / "other.gguf", directions(1, 2, width=8, seed=1))
    tables = {
        "other-vector": f'file = "local/vectors/other.gguf"\nsha256 = "{sha(other.read_bytes())}"',
        "other-blocks": f'file = "{CALM}"\nsha256 = "{sha((root / CALM).read_bytes())}"\n'
        "layers = [2]",
        "no-vector": "",
    }
    expected = (
        f"Call {call['call_id']} was steered with {CALM}, blocks [1, 2], which this chat's "
        "model can't run. Resume its conversation to replay it there."
    )  # the page shows it too, so it names no terminal command (tests/test_class.py's T2)
    for name, table in tables.items():
        scale = None if table == "" else "4.0"
        c.run(
            "profile", steered(root, name, steering=table, scale=scale, backend="llama.cpp"), quiet
        )
        sent = len(seen["requests"])
        refused = raised(lambda: c.session.replay(path, call["call_id"]))
        assert str(refused) == expected, name
        assert len(seen["requests"]) == sent, name


# --- replay outside a chat, and a batch run --------------------------------------------


def test_clarity_replay_starts_the_server_each_call_needs(monkeypatch, root, tiny_tokenizer):
    steer_root(root)
    model_like(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    session = session_on(root, tiny_tokenizer)
    log = LogFile(root / "logs" / "three.jsonl")
    calls = []
    try:
        for scale in ("4.0", "2.0", "0"):
            name = steered(root, f"at-{scale}", scale=scale, backend="llama.cpp")
            session.new_conversation(name, log=log)
            calls.append(session.say("Hi").record)
    finally:
        session.close()
    assert scales(seen) == [4.0, 2.0, 0.0]
    replaying = session_on(root, tiny_tokenizer)  # clarity replay: no conversation
    try:
        for call in calls:
            seen["replies"].append(call["output_text"])
            try:
                outcome = replaying.replay(log.path, call["call_id"])
            except ConfigError as e:  # a refusal fails here, on this assertion
                raise AssertionError(f"{call['call_id']} wasn't replayed: {e}") from e
            assert outcome["identical"], call["call_id"]
    finally:
        replaying.close()
    assert scales(seen) == [4.0, 2.0, 0.0, 4.0, 2.0, 0.0]


def test_a_batch_of_two_strengths_starts_two_servers(monkeypatch, root, tiny_tokenizer):
    from clarity_chat.batch import run_batch

    steer_root(root)
    model_like(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    (root / "openings.txt").write_text("The kettle sang.\n---\nAsk me.\n", encoding="utf-8")
    names = [steered(root, f"at-{s}", scale=s, backend="llama.cpp") for s in ("4.0", "2.0")]
    out = root / "batch-out"
    session = session_on(root, tiny_tokenizer)
    try:
        ran = raised(
            lambda: run_batch(
                session, "run-test", out, root / "openings.txt", names, 1, lambda text: None
            )
        )
    finally:
        session.close()
    assert ran is None, f"the run stopped: {ran!r}"
    assert scales(seen) == [4.0, 2.0]
    (log,) = out.glob("*.jsonl")
    assert each_call_ran_on_its_server(log) == 4


def test_a_new_strength_for_a_changed_vector_starts_nothing_and_the_old_one_runs_on(chat):
    """State 9a's `/set steering_scale` form (moved from B3): the re-hash before the start
    refuses with steering.load's sentence; the server at ×4 answers the next reply."""
    c, seen = chat
    c.run("profile", "steered", quiet)
    say(c)
    root = c.session.root
    pinned = sha((root / CALM).read_bytes())
    write(root / CALM, directions(1, 2, width=8, seed=7))
    now = sha((root / CALM).read_bytes())
    refused = raised(lambda: c.run("set", "steering_scale 2.0", quiet))
    assert str(refused) == f"{root / CALM} has sha256 {now}, not the pinned {pinned}."
    assert scales(seen) == [4.0]
    assert not seen["procs"][-1].terminated, "the old server was stopped"
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.settings["steering_scale"].value == 4.0
    assert say(c, "Again")["request"]["steering"]["scale"] == 4.0
    assert each_call_ran_on_its_server(conversation.log.path) == 2


def test_a_chat_at_two_strengths_resumes_exports_and_opens_each_call(chat):
    """The class tests' T3 (tests/test_class.py), its steered llama.cpp row (moved from B3):
    accepted and logged means resumable, exportable and readable in Details."""
    from clarity_chat.export import export_log
    from clarity_chat.export_inspect import export_chat

    c, seen = chat
    c.run("profile", "steered", quiet)
    say(c)
    c.run("set", "steering_scale 2.0", quiet)
    say(c, "Again")
    conversation = c.session.conversation
    assert conversation is not None
    path = conversation.log.path
    c.run("profile", "steered", quiet)
    record = c.session.resume(path)
    assert record["render"] is None or record["render"]["identical"], record["render"]
    assert export_log(c.session, path)["path"]
    assert export_chat(c.session, path)["path"]
    for call in [r for r in read_records(path) if r["type"] == "call"]:
        assert c.run("prompt", call["call_id"], quiet)


# --- the strength in float32 (Checkpoint 1, finding 1) ---------------------------------
# llama-server reads the scale as a float32. 1e-46 is 0.0 there and 1e39 is inf, so the
# record would say a strength the server didn't run. 3e38 is a float32, but times a
# direction it overflows; only the start, which every path reaches, sees the product.
ROUNDS_AWAY = ["1e-46", "1e39"]
OVERFLOWS = "3e38"


def edit_records(path, change) -> None:
    """Rewrite a log in place, `change` called on each record: a hand-edited log."""
    import json

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        change(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("scale", [*ROUNDS_AWAY, OVERFLOWS])
def test_a_strength_float32_cant_hold_is_refused_and_the_old_one_runs_on(chat, scale):
    c, seen = chat
    c.run("profile", "steered", quiet)
    say(c)
    refused = raised(lambda: c.run("set", f"steering_scale {scale}", quiet))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert scales(seen) == [4.0], "a server was started"
    assert not seen["procs"][-1].terminated, "the old server was stopped"
    assert say(c, "Again")["request"]["steering"]["scale"] == 4.0


@pytest.mark.parametrize("scale", [*ROUNDS_AWAY, OVERFLOWS])
def test_clarity_replay_refuses_a_logged_strength_float32_cant_hold(
    monkeypatch, root, tiny_tokenizer, scale
):
    steer_root(root)
    model_like(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    session = session_on(root, tiny_tokenizer)
    try:
        session.new_conversation(steered(root, backend="llama.cpp"))
        call = session.say("Hi").record
        assert session.conversation is not None
        path = session.conversation.log.path
    finally:
        session.close()

    def change(row):
        if row["type"] == "call":
            row["request"]["steering"]["scale"] = float(scale)
            row["settings"]["steering_scale"]["value"] = float(scale)

    edit_records(path, change)
    replaying = session_on(root, tiny_tokenizer)
    try:
        refused = raised(lambda: replaying.replay(path, call["call_id"]))
    finally:
        replaying.close()
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert scales(seen) == [4.0], "a server was started"


@pytest.mark.parametrize("scale", [*ROUNDS_AWAY, OVERFLOWS, "abc"])
def test_a_resume_of_a_logged_strength_float32_cant_hold_is_refused(chat, scale):
    """The conversation record's own settings are checked like a profile's before anything
    starts or stops: the running chat stays."""
    c, seen = chat
    c.run("profile", "steered", quiet)
    say(c)
    conversation = c.session.conversation
    assert conversation is not None
    path = conversation.log.path
    value: Any = scale if scale == "abc" else float(scale)

    def change(row):
        if row["type"] == "conversation":
            row["settings"]["steering_scale"]["value"] = value

    edit_records(path, change)
    c.run("profile", "steered", quiet)
    running = c.session.conversation
    started = len(seen["argv"])
    refused = raised(lambda: c.run("resume", conversation.conversation_id, quiet))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert len(seen["argv"]) == started, "a server was started"
    assert not seen["procs"][-1].terminated, "the running server was stopped"
    assert c.session.conversation is running, "the running chat was stopped"
    assert say(c, "Again")["request"]["steering"]["scale"] == 4.0


# --- a refused new strength starts nothing, and says so (QA finding 1) -----------------
# The vector's checks run inside the start, before the old server stops. A /set they
# refuse used to send "starting" then "ready" though no server started.


def changed(root) -> str:
    pinned = sha((root / CALM).read_bytes())
    write(root / CALM, directions(1, 2, width=8, seed=7))
    return f"{root / CALM} has sha256 {sha((root / CALM).read_bytes())}, not the pinned {pinned}."


def missing(root) -> str:
    import os

    os.remove(root / CALM)
    return f"The vector file {root / CALM} doesn't exist."


@pytest.mark.parametrize(
    ("args", "refusal"),
    [
        ("steering_scale 2.0", changed),
        ("steering_scale 2.0", missing),
        (
            f"steering_scale {OVERFLOWS}",
            lambda root: f"{float(OVERFLOWS)} x direction.1 isn't finite in float32.",
        ),  # fmt: skip
        ("steering_scale", changed),  # back to the profile's ×4, from ×2
    ],
    ids=["changed vector", "missing vector", "overflow", "no value, changed vector"],
)
def test_a_refused_new_strength_says_no_start(chat, args, refusal):
    c, seen = chat
    c.run("profile", "steered", quiet)
    if args == "steering_scale":
        c.run("set", "steering_scale 2.0", quiet)
    say(c)
    running = scales(seen)
    sentence = refusal(c.session.root)
    events: list[dict] = []
    refused = raised(lambda: c.run("set", args, events.append))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused) == sentence
    assert statuses(events) == [], "a start was said, though none ran"
    assert scales(seen) == running, "a server was started"
    assert not seen["procs"][-1].terminated, "the old server was stopped"
    assert say(c, "Again")["request"]["steering"]["scale"] == running[-1]
