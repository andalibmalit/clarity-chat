"""The backend contract: the call record every backend writes, for each way a reply ends.

The live tiers run SCENARIOS on SmolLM2 through each real backend and keep each record's
shape in `fixtures/backend_shapes.json` (committed). Offline, `test_contract.py` runs the
same scenarios on FakeBackend and holds it to the llama.cpp shapes, so a fake that drifts
from the real backends fails in CI. Differences that are expected are written down below.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

SHAPES = Path(__file__).parent / "fixtures" / "backend_shapes.json"

# Record fields that hold dicts whose keys are part of the contract.
NESTED = ("request", "backend", "backend_settings", "timings", "response", "generation_settings")

# The four ways a reply ends (Result.stop_type). "word" runs in text mode, the rest in chat.
SCENARIOS = ("eos", "word", "limit", "cancelled")

Allowances = dict[tuple[str, ...], tuple[str, frozenset[str]]]  # difference -> why, where
ALL = frozenset(SCENARIOS)
REPLIED = frozenset({"eos", "word", "limit"})  # llama.cpp's final chunk came
CANCELLED = frozenset({"cancelled"})

# Where FakeBackend may differ from llama.cpp's record, why, and in which scenarios. Each
# entry must still be a real difference in each scenario it names, so none goes stale.
FAKE_DIFFERS: Allowances = {
    ("keys", "generation_settings"): (
        "llama-server's own echo; the fake echoes what was sent",
        REPLIED,
    ),
}

# Where Transformers' record differs from llama.cpp's, why, and in which scenarios.
BACKENDS_DIFFER: Allowances = {
    ("keys", "request"): ("each backend logs its own request body", ALL),
    ("keys", "backend"): ("describe() names each backend's own versions", ALL),
    ("keys", "backend_settings"): ("each backend's neutral values have their own names", ALL),
    ("keys", "response"): ("Transformers logs the effective GenerationConfig instead", ALL),
    ("keys", "timings"): ("Transformers measures four of llama-server's timing names", ALL),
    ("checks",): ("each backend runs its own checks", ALL),
    ("keys", "generation_settings"): ("llama-server's echo; Transformers has none", REPLIED),
    ("fields", "generation_settings"): ("llama-server's echo; Transformers has none", REPLIED),
    ("fields", "tokens_cached"): ("llama-server reports it; Transformers has no cache", REPLIED),
    ("fields", "truncated"): ("llama-server reports it; Transformers has none", REPLIED),
    # On a cancel, llama.cpp's stream ends before its final chunk, which holds the counts.
    ("fields", "tokens_evaluated"): ("llama.cpp has no count after a cancel", CANCELLED),
    ("fields", "tokens_predicted"): ("llama.cpp has no count after a cancel", CANCELLED),
}


def allowed(table: Allowances, scenario: str) -> set[tuple[str, ...]]:
    """The differences `table` allows in `scenario`."""
    return {key for key, (_, scenarios) in table.items() if scenario in scenarios}


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def shape(record: dict[str, Any], pieces: list[str], end_ids: list[int]) -> dict[str, Any]:
    """What the contract compares: field names and types, nested keys, each check's keys,
    the flags, whether the streamed pieces add up to the reply, and whether the output IDs
    end with an end token."""
    return {
        "fields": {k: type_name(v) for k, v in sorted(record.items())},
        "keys": {k: sorted(record[k]) for k in NESTED if isinstance(record.get(k), dict)},
        "checks": {k: sorted(v) for k, v in sorted(record.get("checks", {}).items())},
        # not_applied depends on the profile's settings and the backend, not the stop: the
        # smol profiles' sampler order isn't Transformers' fixed order.
        "flags": sorted(f for f in record["flags"] if f != "not_applied"),
        "pieces_join_to_output_text": "".join(pieces) == record["output_text"],
        "output_ids_end_with_end_token": bool(record["output_ids"])
        and record["output_ids"][-1] in end_ids,
    }


def run_scenarios(session: Any, chat: str, text: str) -> dict[str, dict[str, Any]]:
    """Each scenario on `session`, with the chat-mode and text-mode profiles given."""
    return {scenario: run_one(session, scenario, chat, text) for scenario in SCENARIOS}


def run_one(session: Any, scenario: str, chat: str, text: str) -> dict[str, Any]:
    pieces: list[str] = []
    cancel = threading.Event()

    def on_text(piece: str) -> None:
        pieces.append(piece)
        if scenario == "cancelled" and len(pieces) == 1:
            # From another thread, between two pieces, as a UI's Stop button does.
            setter = threading.Thread(target=cancel.set)
            setter.start()
            setter.join()

    if scenario == "word":
        session.new_conversation(text)
        session.say("What is the capital of France?")
        record = session.say("And of Italy?", on_text=on_text).record
    else:
        session.new_conversation(chat)
        if scenario == "limit":
            session.set_setting("max_tokens", "4")
        message = "Name one colour." if scenario == "eos" else "Write a long poem."
        record = session.say(message, on_text=on_text, cancel=cancel).record
    assert record["stop_type"] == scenario, (scenario, record["output_text"])
    return shape(record, pieces, [session.tokenizer.tok.eos_token_id])


def differences(a: dict[str, Any], b: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Where two scenario shapes differ, keyed like FAKE_DIFFERS."""
    found: dict[tuple[str, ...], Any] = {}
    for part in ("fields", "keys"):
        for name in sorted(set(a[part]) | set(b[part])):
            if a[part].get(name) != b[part].get(name):
                found[(part, name)] = (a[part].get(name), b[part].get(name))
    for part in ("checks", "flags", "pieces_join_to_output_text", "output_ids_end_with_end_token"):
        if a[part] != b[part]:
            found[(part,)] = (a[part], b[part])
    return found


def check_or_write(backend: str, shapes: dict[str, dict[str, Any]]) -> None:
    """Live tiers: compare with the committed shapes; on a difference, write the new ones
    and fail, so the change is read in the diff and committed on purpose."""
    saved = json.loads(SHAPES.read_text(encoding="utf-8")) if SHAPES.exists() else {}
    if saved.get(backend) == shapes:
        return
    saved[backend] = shapes
    SHAPES.write_text(json.dumps(saved, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    raise AssertionError(
        f"{backend}'s call records changed shape: {SHAPES.name} is rewritten. Read the diff, "
        "then commit it (and fix FakeBackend if test_contract.py now fails)."
    )
