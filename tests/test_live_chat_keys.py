"""Live checks of `/set ctx_size` on the managed llama-server (v0.4.4's Live tiers, the
sprint audit's M1). Each goes through the command a chat sends, never start_model.

Run with `uv run pytest -m live`. Skipped as test_live.py is: without llama-server built or
SmolLM2 fetched. CI's live-cpu job runs all but the `gpu`-marked check.
"""

from __future__ import annotations

from typing import Any

import pytest

from clarity_chat.commands import Commands
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.test_live import REPO, _available, all_checks_ok

pytestmark = pytest.mark.live

if (reason := _available()) is not None:
    pytestmark = [pytest.mark.live, pytest.mark.skip(reason=reason)]


def quiet(_event: dict) -> None:
    pass


def ctx_flag(argv: list[str]) -> str:
    return argv[argv.index("--ctx-size") + 1]


@pytest.fixture(scope="module")
def sized(tmp_path_factory) -> dict[str, Any]:
    """One smol-chat through every size step: a call at 8,192, `/set ctx_size 4096` and a
    call, `/set ctx_size 16384` (refused) and a call, `/set ctx_size` (the reset) and a
    call, then `/replay` of the 4,096 call in the chat at 8,192."""
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    c = Commands(session)
    try:
        c.run("profile", "smol-chat", quiet)
        calls = {"8192": c.run("say", "Name one colour.", quiet)["call"]}
        set_4096 = c.run("set", "ctx_size 4096", quiet)
        calls["4096"] = c.run("say", "Name another.", quiet)["call"]
        with pytest.raises(ConfigError) as refused:
            c.run("set", "ctx_size 16384", quiet)
        conversation = session.conversation
        assert conversation is not None
        size_after_refusal = conversation.ctx_size.to_json()
        calls["after the refusal"] = c.run("say", "And one more.", quiet)["call"]
        c.run("set", "ctx_size", quiet)
        calls["reset"] = c.run("say", "Name a fruit.", quiet)["call"]
        with pytest.raises(ConfigError) as other_size:
            c.run("replay", calls["4096"]["call_id"], quiet)
        log = conversation.log.path
    finally:
        session.close()
    return {
        "calls": calls,
        "set_4096": set_4096,
        "refused": refused.value,
        "size_after_refusal": size_after_refusal,
        "other_size": other_size.value,
        "log": log,
        "records": list(read_records(log)),
    }


def server_of(sized: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    server_id = call["backend"]["server_id"]
    starts = [r for r in sized["records"] if r["type"] == "server_start"]
    return next(r for r in starts if r["server_id"] == server_id)


def test_set_ctx_size_4096_starts_a_new_server_at_that_size(sized):
    before, after = sized["calls"]["8192"], sized["calls"]["4096"]
    all_checks_ok(after)
    assert sized["set_4096"]["setting"] == {"value": 4096, "origin": "chat"}
    old, new = server_of(sized, before), server_of(sized, after)
    assert new["server_id"] != old["server_id"]
    assert ctx_flag(old["argv"]) == "8192" and before["n_ctx"] == 8192
    assert ctx_flag(new["argv"]) == "4096"
    assert new["props"]["default_generation_settings"]["n_ctx"] == 4096
    assert after["n_ctx"] == 4096
    settings = [r for r in sized["records"] if r["type"] == "settings"]
    assert settings[0]["name"] == "ctx_size" and settings[0]["value"] == 4096


@pytest.mark.gpu  # CI's CPU build measures no device memory
def test_at_4096_the_context_cache_is_smaller_than_at_8192(sized):
    old = server_of(sized, sized["calls"]["8192"])["memory"]
    new = server_of(sized, sized["calls"]["4096"])["memory"]
    assert old and new, "a start on the GPU logged no memory"
    assert new[0]["kv_mib"] < old[0]["kv_mib"], (new, old)


def test_16384_is_refused_and_the_next_call_runs_at_the_old_size(sized):
    """Above SmolLM2's trained 8,192: the existing refusal, and the chat goes on at 4,096."""
    assert "trained on" in str(sized["refused"]), sized["refused"]
    assert sized["size_after_refusal"] == {"value": 4096, "origin": "chat"}
    call = sized["calls"]["after the refusal"]
    all_checks_ok(call)
    assert call["n_ctx"] == 4096
    assert server_of(sized, call)["props"]["default_generation_settings"]["n_ctx"] == 4096
    sizes = [r["value"] for r in sized["records"] if r["type"] == "settings"]
    assert sizes == [4096, None], "the refused size was logged as set"


def test_a_reset_goes_back_to_8192_and_the_4096_call_replays_at_its_size(sized, tmp_path):
    reset = sized["calls"]["reset"]
    all_checks_ok(reset)
    assert reset["n_ctx"] == 8192
    assert ctx_flag(server_of(sized, reset)["argv"]) == "8192"
    # In the chat, now at 8,192, the 4,096 call's replay is refused (decision 12) ...
    assert "Set ctx_size to 4096" in str(sized["other_size"]), sized["other_size"]
    # ... and from its log it runs at the size it was made at.
    call = sized["calls"]["4096"]
    other = Session(REPO, logs_dir=tmp_path)
    try:
        record = other.replay(sized["log"], call["call_id"])
        n_ctx = other.backend.n_ctx
    finally:
        other.close()
    assert n_ctx == 4096
    assert record["n_ctx_from"] == "the call's log"
    assert record["identical"], record
    assert set(record["differences"]) <= {"client", "libraries"}, record["differences"]
