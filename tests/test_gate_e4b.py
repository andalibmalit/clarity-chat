"""v0.4.2 A0, the first model's gate (rule 15: the design rests on it), before any setup code;
and since A4, the shipped first model's live check.

A gate: on a scratch root that `notes/tools/gate_root.py` made and `clarity fetch` filled,
with `HF_HOME` the same folder the fetch used (empty before it, so a gated tokenizer repo
shows):

    CLARITY_GATE_ROOT=ROOT GATE_MODEL=NAME HF_HOME=HF uv run pytest -m gate -v -s

The shipped first model (A4): with none of those set, on this checkout's gemma-4-e4b, once
`uv run clarity fetch gemma-4-e4b` has fetched it (logs in a temporary folder):

    uv run pytest -m gate -v -s

On the real llama-server at the pin: the model loads; the profile's template turns thinking
off (the prompt ends at the model's turn) and writes no date; the start records the file's
suppress_tokens; a call replays Identical; bge-m3's GGUF (a copy in the root) has the
lock's sha256; and both servers' logged memory, the model's and
bge-m3's at 8,192 tokens, is printed with this Mac's limit: the 16 GB verdict is read from
those figures, not asserted here. Made-up text.
Evaluation only: nothing here ships the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clarity_chat.files import sha256_file
from clarity_chat.models import read_lock
from clarity_chat.session import Session

pytestmark = pytest.mark.gate

DATE_WARNING = "This model's chat template can write today's date"


@pytest.fixture
def gate(tmp_path):
    root, name = os.environ.get("CLARITY_GATE_ROOT"), os.environ.get("GATE_MODEL")
    if root is None and name is None:  # A4: the shipped first model, in this checkout
        from tests.conftest import REPO

        root, name, logs = str(REPO), "gemma-4-e4b", tmp_path / "logs"  # never REPO/logs
    else:
        assert root and name, "Set both CLARITY_GATE_ROOT and GATE_MODEL (gate_root.py)."
        assert os.environ.get("HF_HOME"), "Set HF_HOME to the folder the fetch used."
        logs = Path(root) / "logs"
    session = Session(Path(root), logs_dir=logs)
    try:
        yield session, name
    finally:
        session.close()


def totals(record: dict) -> dict[str, tuple[float, int]]:
    return {m["device"]: (m["total_mib"], m["limit_mib"]) for m in record["memory"]}


def test_the_first_model_passes_the_gate(gate):
    session, name = gate
    conversation = session.new_conversation(name)
    start = conversation.started_server
    assert start is not None, "the model didn't start a server"
    banned = start["model_file"]["suppress_tokens"]
    print(f"\n{name}: suppress_tokens {banned!r}")
    assert not any(w.startswith(DATE_WARNING) for w in conversation.warnings), conversation.warnings
    record = session.say("Name one made-up bird and its song.").record
    ending = record["prompt_text"][-40:]
    print(f"prompt ends {ending!r}")
    assert record["prompt_text"].endswith("<|turn>model\n"), "thinking isn't off"
    # Gemma 4's template turns thinking on with a <|think|> token in the system turn; the
    # ending alone doesn't show it's absent (Checkpoint 1's M2).
    assert "<|think|>" not in record["prompt_text"], "the prompt asks the model to think"
    print(f"reply {record['output_text'][:120]!r}")
    for check, result in record["checks"].items():
        assert result.get("ok", True), (check, result)
    replayed = session.replay_current()
    assert replayed["identical"], replayed
    # bge-m3's GGUF is a copy (gate_root.py): its bytes must be the lock's.
    lock = read_lock(session.root)["models"]["bge-m3"]["gguf"]
    assert sha256_file(session.root / lock["path"]) == lock["sha256"], "bge-m3's copy differs"
    embedder = session.embedder()
    embedder.start()
    assert embedder.server is not None
    chat, beside = totals(start), totals(embedder.server.record)
    for device, (mib, limit) in chat.items():
        other = beside.get(device, (0.0, 0))[0]
        print(
            f"{device}: {name} {mib:,.0f} MiB + bge-m3 {other:,.0f} MiB = {mib + other:,.0f} MiB; "
            f"this Mac's limit {limit:,} MiB"
        )
