"""Golden logs: one small made-up log written by each released version's code, never edited.

Current code must still read, replay and annotate every one of them: replay is Clarity's
promise, and it covers logs written by older versions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clarity_chat.logs import SCHEMA_VERSION, read_records
from clarity_chat.session import Session, annotate_log
from tests.fakes import FakeBackend

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
LOGS = sorted(GOLDEN.glob("*.jsonl"))


def test_the_golden_logs_are_found():
    assert GOLDEN / "v0.1.0.jsonl" in LOGS  # an empty glob would skip every test below


VECTOR = Path(__file__).parent / "fixtures" / "steering" / "smollm2-repeng.gguf"


@pytest.fixture
def steerable(fetched_root) -> Path:
    """The project v0.2.2's golden log was written in: the tiny model has Transformers weights
    (a config.json the repeng test vector fits) and the vector sits where the log says."""
    from clarity_chat.models import read_lock, write_lock_entry

    weights = fetched_root / "models" / "tiny-hf"
    weights.mkdir(parents=True)
    config = {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30}
    (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
    entry = read_lock(fetched_root)["models"]["tiny"]
    write_lock_entry(fetched_root, "tiny", {**entry, "source_dir": "models/tiny-hf"})
    (fetched_root / "local" / "vectors").mkdir(parents=True)
    shutil.copy(VECTOR, fetched_root / "local" / "vectors" / VECTOR.name)
    return fetched_root


@pytest.fixture(params=LOGS, ids=lambda p: p.stem)
def golden(request, tmp_path) -> Path:
    """A copy, because replay and marks append to the log."""
    return Path(shutil.copy(request.param, tmp_path / request.param.name))


def test_every_record_reads(golden):
    rows = list(read_records(golden))
    assert all(r["schema_version"] <= SCHEMA_VERSION for r in rows)
    assert rows[0]["type"] == "conversation"
    assert {"turn", "call", "mark", "note", "replay"} <= {r["type"] for r in rows}


def test_every_reply_replays_identically(golden, steerable, tiny_tokenizer):
    calls = [r for r in read_records(golden) if r["type"] == "call" and r.get("turn_id")]
    complete = [c for c in calls if c["stop_type"] != "cancelled"]
    backend = FakeBackend([c["output_text"] for c in complete], tokenizer=tiny_tokenizer)
    backend.before_v0_2_3 = True  # the fake that wrote these logs
    session = Session(steerable, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    for call in complete:
        record = session.replay(golden, call["call_id"])
        assert record["identical"], call["call_id"]
        assert backend.calls[-1]["token_ids"] == call["prompt_ids"]
        sent = backend.calls[-1]["steering"]  # None for every call before v0.2.2
        assert (sent and sent.to_json()) == call["request"].get("steering")


def test_a_reply_can_be_marked_after_the_fact(golden):
    call = next(r for r in read_records(golden) if r["type"] == "call")
    annotate_log(golden, "mark", call["call_id"], "interesting")
    last = json.loads(golden.read_text(encoding="utf-8").splitlines()[-1])
    assert last["type"] == "mark" and last["call_id"] == call["call_id"]


def test_every_golden_log_resumes(golden, steerable, tiny_tokenizer):
    backend = FakeBackend([" More."], tokenizer=tiny_tokenizer)
    session = Session(steerable, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    record = session.resume(golden)
    assert record["render"] is None or record["render"]["identical"], record["render"]
    turn = session.say("One more made-up message.").turn
    assert turn is not None and turn.text == " More."
    # v0.2.2's log ends after /set steering_scale 0: resume keeps it, so no steering.
    assert backend.calls[-1]["steering"] is None


def test_the_v0_2_0_golden_log_holds_every_new_record_type():
    kinds = {r["type"] for r in read_records(GOLDEN / "v0.2.0.jsonl")}
    assert {"head", "unmark", "settings", "title", "resume", "torn_line"} <= kinds


def test_the_v0_2_1_golden_log_holds_the_new_call_fields():
    calls = [r for r in read_records(GOLDEN / "v0.2.1.jsonl") if r["type"] == "call"]
    assert all(isinstance(c["message_spans"], list) for c in calls)
    assert any("tokens" in (c.get("logprobs") or {}) for c in calls)
    assert all("logprobs" in c["settings"] for c in calls)
    regens = [c for c in calls if c.get("regen_of")]
    assert len(regens) == 4 and regens[-1]["stop_type"] == "cancelled"


def test_the_v0_2_2_golden_log_holds_a_steered_and_an_unsteered_call():
    rows = list(read_records(GOLDEN / "v0.2.2.jsonl"))
    calls = [r for r in rows if r["type"] == "call"]
    assert calls[0]["request"]["steering"]["scale"] == 4.0
    assert calls[0]["profile"]["toml"].count("[steering]") == 1
    assert calls[-1]["settings"]["steering_scale"] == {"value": 0.0, "origin": "chat"}
    assert not calls[-1]["request"].get("steering")
    assert any(r["type"] == "settings" and r["name"] == "steering_scale" for r in rows)
