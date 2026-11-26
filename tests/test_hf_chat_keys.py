"""Live checks of `/set ctx_size` on the Transformers backend (v0.4.4's Live tiers, the sprint
audit's M1): the size is a limit Clarity keeps, so a change reloads nothing, and a size
above the model's positions is refused with the old size kept (S12). Each goes through the
command a chat sends, never start_model.

Run with `uv run pytest -m hf -v`. Skipped as test_hf.py is.
"""

from __future__ import annotations

from typing import Any

import pytest

from clarity_chat.backends.transformers_hf import TransformersBackend
from clarity_chat.commands import Commands
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.test_hf import REPO, _available, checks_ok, profile

pytestmark = pytest.mark.hf

if (reason := _available()) is not None:
    pytestmark = [pytest.mark.hf, pytest.mark.skip(reason=reason)]


def quiet(_event: dict) -> None:
    pass


@pytest.fixture(scope="module")
def sized(tmp_path_factory) -> dict[str, Any]:
    """smol-chat on Transformers: a call at 8,192, `/set ctx_size 4096` and a call, then
    `/set ctx_size 16384` (above SmolLM2's positions) and a call."""
    folder = tmp_path_factory.mktemp("profiles")
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    c = Commands(session)
    try:
        c.run("profile", profile(folder, "smol-chat"), quiet)
        calls = {"8192": c.run("say", "Name one colour.", quiet)["call"]}
        backend = session.backend
        assert isinstance(backend, TransformersBackend) and backend.record is not None
        loaded, positions = backend.model, backend.record["max_position_embeddings"]
        set_4096 = c.run("set", "ctx_size 4096", quiet)
        same_model = session.backend is backend and backend.model is loaded
        calls["4096"] = c.run("say", "Name another.", quiet)["call"]
        with pytest.raises(ConfigError) as refused:
            c.run("set", "ctx_size 16384", quiet)
        conversation = session.conversation
        assert conversation is not None
        size_after_refusal = conversation.ctx_size.to_json()
        calls["after the refusal"] = c.run("say", "And one more.", quiet)["call"]
        log = conversation.log.path
    finally:
        session.close()
    return {
        "calls": calls,
        "positions": positions,
        "set_4096": set_4096,
        "same_model": same_model,
        "refused": refused.value,
        "size_after_refusal": size_after_refusal,
        "records": list(read_records(log)),
    }


def test_set_ctx_size_4096_reloads_nothing_and_the_call_runs_at_4096(sized):
    before, after = sized["calls"]["8192"], sized["calls"]["4096"]
    checks_ok(after)
    assert before["n_ctx"] == 8192
    assert sized["set_4096"]["setting"] == {"value": 4096, "origin": "chat"}
    assert sized["same_model"], "the size change loaded the model again"
    starts = [r for r in sized["records"] if r["type"] == "server_start"]
    assert len(starts) == 1, "a size change logged a new start"
    assert after["backend"]["server_id"] == before["backend"]["server_id"]
    assert after["n_ctx"] == 4096


def test_a_size_above_the_model_s_positions_is_refused_and_the_old_size_runs_on(sized):
    """S12: the refused size isn't left behind for the next call."""
    assert sized["positions"] < 16384, "Test bug: 16,384 isn't above SmolLM2's positions."
    assert "positions" in str(sized["refused"]), sized["refused"]
    assert sized["size_after_refusal"] == {"value": 4096, "origin": "chat"}
    call = sized["calls"]["after the refusal"]
    checks_ok(call)
    assert call["n_ctx"] == 4096
    sizes = [r["value"] for r in sized["records"] if r["type"] == "settings"]
    assert sizes == [4096], "the refused size was logged as set"
