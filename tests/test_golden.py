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


@pytest.fixture(params=LOGS, ids=lambda p: p.stem)
def golden(request, tmp_path) -> Path:
    """A copy, because replay and marks append to the log."""
    return Path(shutil.copy(request.param, tmp_path / request.param.name))


def test_every_record_reads(golden):
    rows = list(read_records(golden))
    assert all(r["schema_version"] <= SCHEMA_VERSION for r in rows)
    assert rows[0]["type"] == "conversation"
    assert {"turn", "call", "mark", "note", "replay"} <= {r["type"] for r in rows}


def test_every_reply_replays_identically(golden, fetched_root, tiny_tokenizer):
    calls = [r for r in read_records(golden) if r["type"] == "call" and r.get("turn_id")]
    complete = [c for c in calls if c["stop_type"] != "cancelled"]
    backend = FakeBackend([c["output_text"] for c in complete], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    for call in complete:
        record = session.replay(golden, call["call_id"])
        assert record["identical"], call["call_id"]
        assert backend.calls[-1]["token_ids"] == call["prompt_ids"]


def test_a_reply_can_be_marked_after_the_fact(golden):
    call = next(r for r in read_records(golden) if r["type"] == "call")
    annotate_log(golden, "mark", call["call_id"], "interesting")
    last = json.loads(golden.read_text(encoding="utf-8").splitlines()[-1])
    assert last["type"] == "mark" and last["call_id"] == call["call_id"]
