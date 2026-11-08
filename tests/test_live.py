"""Live tests: the real llama-server (server.toml, GPU) and the fetched SmolLM2-135M-Instruct.

Run with `uv run pytest -m live`. Skipped if llama-server isn't built or SmolLM2 isn't fetched.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from clarity_chat.backends.llama_cpp import LlamaCppBackend, server_binary
from clarity_chat.logs import read_records
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.render import Tokenizer
from clarity_chat.session import ContextError, Session

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]
MODEL = "smollm2-135m-instruct"


def _available() -> str | None:
    try:
        server_binary(load_server_config(REPO).llama_cpp_dir)
        check_weights_quick(REPO, lock_entry(REPO, load_models(REPO)[MODEL]))
    except ConfigError as e:
        return str(e)
    return None


if (reason := _available()) is not None:
    pytestmark = [pytest.mark.live, pytest.mark.skip(reason=reason)]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    session.new_conversation("smol-text")
    yield session
    session.close()


def all_checks_ok(record: dict) -> None:
    failed = {name: check for name, check in record["checks"].items() if not check["ok"]}
    assert not failed, failed


def test_server_start_is_logged_with_props_and_gpu_placement(session):
    record = session.backend.server_record
    assert record["props"]["model_path"].endswith(f"{MODEL}-bf16.gguf")
    assert record["props"]["default_generation_settings"]["n_ctx"] == 8192
    assert record["placement"]["devices"], "no GPU device in use"
    assert record["warnings"] == []
    assert "--no-context-shift" in record["argv"]


def test_text_mode_reply_stops_before_the_next_user_line(session):
    session.new_conversation("smol-text")
    session.say("What is the capital of France?")
    outcome = session.say("And of Italy?")
    record = outcome.record
    all_checks_ok(record)
    assert record["stop_type"] == "word", record["output_text"]
    assert record["stopping_word"] == "\nUser:"
    assert "\nUser:" not in record["output_text"]
    assert record["tokens_evaluated"] == record["n_prompt_client"]
    assert record["timings"]["cache_n"] == 0
    assert "incomplete_output_ids" not in record["flags"]


def test_same_seed_same_server_gives_identical_output_ids(session):
    """If this fails, no replay result means anything (the brief: stop and report)."""
    outputs = []
    for _ in range(2):
        session.new_conversation("smol-text")
        outputs.append(session.say("Tell me a short story about a robot.").record)
    assert outputs[0]["prompt_ids"] == outputs[1]["prompt_ids"]
    assert outputs[0]["output_ids"] == outputs[1]["output_ids"]
    assert outputs[0]["stop_type"] == outputs[1]["stop_type"]


def test_over_long_prompt_is_refused_before_sending(session):
    session.new_conversation("smol-text")
    with pytest.raises(ContextError, match="Nothing was sent"):
        session.say("word " * 9000)
    calls = [r for r in read_records(session.conversation.log.path) if r["type"] == "call"]
    assert calls[-1]["error"].startswith("ContextError")
    assert "output_ids" not in calls[-1]


def test_ctrl_c_cancels_the_reply_and_the_server_keeps_working(session):
    session.new_conversation("smol-text")

    def interrupt(_text: str) -> None:
        raise KeyboardInterrupt

    outcome = session.say("Write a long poem about the sea.", on_text=interrupt)
    assert outcome.record["stop_type"] == "cancelled"
    assert "cancelled" in outcome.record["flags"]
    all_checks_ok(session.say("Thanks.").record)


@pytest.fixture(scope="module")
def small_context_backend(tmp_path_factory):
    """A second server with a 256-token context, to find llama-server's exact limit."""
    config = load_server_config(REPO)
    config = dataclasses.replace(config, flags={**config.flags, "ctx-size": 256})
    backend = LlamaCppBackend(config, tmp_path_factory.mktemp("logs-small"))
    model = load_models(REPO)[MODEL]
    backend.start(model, check_weights_quick(REPO, lock_entry(REPO, model)))
    yield backend
    backend.close()


def counting_ids(n: int) -> list[int]:
    """`n` token IDs of "1 2 3 …": the model keeps counting instead of stopping early."""
    model = load_models(REPO)[MODEL]
    tokenizer = Tokenizer.from_hub(model.tokenizer_repo, model.tokenizer_revision)
    ids = tokenizer.encode(" ".join(str(i) for i in range(1, 400)))
    assert len(ids) >= n
    return ids[:n]


MAX_TOKENS = 32
SETTINGS = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 1,
    "max_tokens": MAX_TOKENS,
    "samplers": ["temperature"],
}


def test_context_guard_matches_llama_servers_limit(small_context_backend):
    from clarity_chat.session import fits_context

    n_ctx, max_tokens = 256, MAX_TOKENS
    fitting = n_ctx - max_tokens - 1  # the largest prompt the client allows
    assert fits_context(fitting, max_tokens, n_ctx)
    assert not fits_context(fitting + 1, max_tokens, n_ctx)

    full = small_context_backend.complete(counting_ids(fitting), SETTINGS, [])
    assert full.stop_type == "limit"
    assert full.response["truncated"] is False
    assert full.n_predicted == max_tokens

    one_more = small_context_backend.complete(counting_ids(fitting + 1), SETTINGS, [])
    assert one_more.response["truncated"] is True  # the client is right to refuse this
    assert one_more.n_predicted == max_tokens - 1


def test_llama_arg_variables_are_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "1024")
    backend = LlamaCppBackend(load_server_config(REPO), tmp_path)
    model = load_models(REPO)[MODEL]
    try:
        record = backend.start(model, check_weights_quick(REPO, lock_entry(REPO, model)))
        assert record is not None
        assert "LLAMA_ARG_CTX_SIZE" in record["env_removed"]
        assert record["props"]["default_generation_settings"]["n_ctx"] == 8192
    finally:
        backend.close()
    assert backend.server is None
