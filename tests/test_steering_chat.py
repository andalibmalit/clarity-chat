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
    refused = raised(lambda: c.run("resume", str(path), quiet))
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


def test_a_replay_with_other_steering_than_the_chat_can_run_is_sent_to_the_terminal(chat):
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
        "model can't run. Replay it with `clarity replay` from the terminal, or resume its "
        "conversation."
    )
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
