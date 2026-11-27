"""Live tests of Gemma 4 E4B's official weights on the Transformers backend (v0.4.7 A3).

Run with `uv run pytest -m hf -v -s -k gemma` (needs `uv sync --extra hf`,
`uv run clarity fetch gemma-4-e4b-official`, and setup's `gemma-4-e4b` for the prompt-ID
test). Each test loads the bf16 weights on MPS, about 16 GB, one model at a time: check
that no other model is loaded and that about 16 GB is free before running them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clarity_chat.models import lock_entry, source_weights
from clarity_chat.profiles import ConfigError, load_models
from clarity_chat.session import Session
from tests.test_hf import checks_ok, make_and_steer, profile, project

pytestmark = pytest.mark.hf

REPO = Path(__file__).resolve().parents[1]
OFFICIAL, Q4_0 = "gemma-4-e4b-official", "gemma-4-e4b"
SYSTEM = '"Answer in one short paragraph."'  # made-up, like every message here
ON_OFFICIAL = {"model": f'"{OFFICIAL}"', "device": '"mps"', "system": SYSTEM}


def _available() -> str | None:
    try:
        import torch
    except ImportError:
        return "torch is not installed (uv sync --extra hf)"
    if not torch.backends.mps.is_available():
        return "MPS isn't available"
    try:
        source_weights(REPO, lock_entry(REPO, load_models(REPO)[OFFICIAL]))
    except ConfigError as e:
        return str(e)
    return None


if (reason := _available()) is not None:
    pytestmark = [pytest.mark.hf, pytest.mark.skip(reason=reason)]


def official(folder: Path, **extra: str) -> str:
    """A copy of profiles/gemma-4-e4b.toml (thinking off, the model's recommended settings,
    seed 42) on the official weights, on Transformers on MPS, with a system message."""
    return profile(folder, Q4_0, **ON_OFFICIAL, **extra)


def test_gemma_loads_chats_repeats_and_replays(tmp_path, capsys):
    """The load: Clarity's attention setting and bf16, and the start record's positions read
    from text_config. A reply ends on one of the three end tokens with every check passing;
    the same seed gives the same reply; a replay after a reload is Identical."""
    import torch

    folder = tmp_path / "profiles"
    folder.mkdir()
    name = official(folder)
    message = "Reply with one word: yes."
    session = Session(REPO, logs_dir=tmp_path / "chat")
    try:
        records = []
        for _ in range(2):
            session.new_conversation(name)
            session.set_setting("max_tokens", "256")
            records.append(session.say(message).record)
        server = session.backend.server_record
        assert server is not None
        model = session.backend.model  # ty: ignore[unresolved-attribute]
        with capsys.disabled():
            gb = 1e9
            print(
                f"\nMPS after the load: {torch.mps.current_allocated_memory() / gb:.2f} GB "
                f"allocated, {torch.mps.driver_allocated_memory() / gb:.2f} GB by the driver"
            )
            print(f"reply: {records[0]['output_text']!r}")
        assert session.conversation is not None
        log = session.conversation.log.path
    finally:
        session.close()
    assert server["attn_implementation"] == "sdpa"
    assert model.config.get_text_config()._attn_implementation == "sdpa"
    assert model.dtype == torch.bfloat16
    assert server["device"] == "mps"
    assert server["max_position_embeddings"] == 131072
    eos = server["eos_token_id"]
    assert len(eos) == 3, eos
    first, second = records
    checks_ok(first)
    assert first["stop_type"] == "eos", first["output_text"]
    assert first["output_ids"][-1] in eos
    assert first["settings"]["seed"]["value"] == 42
    assert second["output_ids"] == first["output_ids"]
    other = Session(REPO, logs_dir=tmp_path / "replay")
    try:
        replayed = other.replay(log, first["call_id"])
    finally:
        other.close()
    assert replayed["identical"], replayed
    assert set(replayed["differences"]) <= {"client", "libraries"}, replayed["differences"]


def test_making_a_vector_on_gemma_and_steering_with_its_snippet(tmp_path, capsys):
    """`clarity vector make` on the made-up 8-pair file writes blocks 1 to 41 of Gemma's 42
    and a snippet at block 21; the snippet's profile steers, with the steering check
    counting each steered block once per pass (Gemma shares its key-value cache across its
    last blocks), and replays Identical. With -s it prints the vector's path, for G14."""
    root = project(tmp_path)
    record = make_and_steer(root, tmp_path, capsys, Q4_0, "Tell me about rivers.", **ON_OFFICIAL)
    assert record["vector"]["model_hint"] == "gemma4"
    assert record["vector"]["blocks"] == list(range(1, 42))


def test_gemma_s_two_entries_give_the_same_prompt_ids(tmp_path):
    """The Q4_0 profile's llama.cpp call and the same profile on the official weights on
    Transformers log the same prompt IDs for the same message: both render with
    google/gemma-4-E4B-it at the same revision. The switch stops llama-server first."""
    folder = tmp_path / "profiles"
    folder.mkdir()
    message = "What do rivers carry to the sea?"
    session = Session(REPO, logs_dir=tmp_path)
    records = {}
    try:
        for backend, name in (
            ("llama.cpp", profile(folder, Q4_0, "llama.cpp", system=SYSTEM)),
            ("transformers", official(folder)),
        ):
            session.new_conversation(name)
            session.set_setting("max_tokens", "16")
            records[backend] = session.say(message).record
            checks_ok(records[backend])
    finally:
        session.close()
    llama, hf = records["llama.cpp"], records["transformers"]
    assert llama["model"]["name"] == Q4_0 and hf["model"]["name"] == OFFICIAL
    assert llama["prompt_ids"] == hf["prompt_ids"]
