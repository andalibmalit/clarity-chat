"""FakeBackend keeps to the contract the real backends' call records set (tests/contract.py).

The shapes come from SmolLM2 runs on each backend (`-m live` and `-m hf`), committed in
fixtures/backend_shapes.json. These tests read that file; they need no model.
"""

from __future__ import annotations

import json

import pytest

from clarity_chat.session import Session
from tests import contract
from tests.fakes import FakeBackend

LONG = " The sea is wide and the sky is grey, and the gulls cry out at the end of day."


def saved_shapes() -> dict:
    return json.loads(contract.SHAPES.read_text(encoding="utf-8"))


@pytest.fixture
def fake_shapes(fetched_root, chat_tokenizer):
    replies = [" Blue.", " Paris.", " Rome.\nUser: And Spain?", LONG, LONG]
    backend = FakeBackend(replies, tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    return contract.run_scenarios(session, chat="chat", text="text")


@pytest.mark.parametrize("scenario", contract.SCENARIOS)
def test_the_fake_backend_writes_llama_cpps_record_shapes(fake_shapes, scenario):
    found = contract.differences(saved_shapes()["llama.cpp"][scenario], fake_shapes[scenario])
    unexpected = {k: v for k, v in found.items() if k not in contract.FAKE_DIFFERS}
    assert not unexpected


@pytest.mark.parametrize("scenario", contract.SCENARIOS)
def test_the_real_backends_differ_only_where_written_down(scenario):
    shapes = saved_shapes()
    found = contract.differences(shapes["llama.cpp"][scenario], shapes["transformers"][scenario])
    unexpected = {k: v for k, v in found.items() if k not in contract.BACKENDS_DIFFER}
    assert not unexpected


def test_every_written_down_difference_is_still_real():
    """An allowance nothing needs any more is removed, so the contract doesn't loosen."""
    shapes = saved_shapes()
    real = set()
    for scenario in contract.SCENARIOS:
        real |= set(
            contract.differences(shapes["llama.cpp"][scenario], shapes["transformers"][scenario])
        )
    assert set(contract.BACKENDS_DIFFER) <= real
