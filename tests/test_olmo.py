"""OLMo 3 Instruct and Base through the managed llama-server (opt-in: `pytest -m olmo`).

Needs both models fetched (`clarity fetch olmo-3-7b-instruct`, `clarity fetch olmo-3-7b-base`).
Only one 7B model is loaded at a time: switching profiles stops the old server first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models
from clarity_chat.session import Session

REPO = Path(__file__).resolve().parents[1]
INSTRUCT, BASE = "olmo-3-7b-instruct", "olmo-3-7b-base"
# generation_config.json's eos_token_id. The brief expected <|im_end|> (100265), but OLMo 3's
# template ends the last assistant message with eos_token, <|endoftext|> (100257), and the
# model does the same: both are its declared end tokens.
END_TOKENS = {100265, 100257}

pytestmark = pytest.mark.olmo


def _missing() -> str | None:
    try:
        models = load_models(REPO)
        for name in (INSTRUCT, BASE):
            check_weights_quick(REPO, lock_entry(REPO, models[name]))
    except ConfigError as e:
        return str(e)
    return None


if (reason := _missing()) is not None:
    pytestmark = [pytest.mark.olmo, pytest.mark.skip(reason=reason)]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    yield session
    session.close()


def checks_ok(record: dict) -> None:
    failed = {name: check for name, check in record["checks"].items() if not check["ok"]}
    assert not failed, failed


@pytest.mark.parametrize(("profile", "n_prompt"), [("1a-stock", 37), ("1b-blank", 15)])
def test_instruct_chat_reply_ends_with_an_end_token(session, profile, n_prompt):
    session.new_conversation(profile)
    record = session.say("hi").record
    checks_ok(record)
    assert record["n_prompt_client"] == n_prompt
    assert record["stop_type"] == "eos", record["output_text"]
    assert record["output_ids"][-1] in END_TOKENS


def test_same_seed_same_server_gives_identical_output_ids(session):
    session.new_conversation("1b-blank")
    first = session.say("Name three rivers.").record
    session.new_conversation("1b-blank")
    second = session.say("Name three rivers.").record
    assert first["prompt_ids"] == second["prompt_ids"]
    assert first["output_ids"] == second["output_ids"]
    assert first["stop_type"] == second["stop_type"]


@pytest.mark.parametrize("profile", ["4-raw-instruct", "5-raw-base"])
def test_raw_text_mode_on_both_models(session, profile):
    session.new_conversation(profile)
    record = session.say("Hello, who are you?").record
    checks_ok(record)
    if record["stop_type"] == "word":
        assert "\nUser:" not in record["output_text"]


def test_switching_models_restarts_the_server(session):
    session.new_conversation("4-raw-instruct")
    instruct_path = session.backend.server_record["props"]["model_path"]
    session.new_conversation("5-raw-base")
    base_path = session.backend.server_record["props"]["model_path"]
    assert instruct_path.endswith(f"{INSTRUCT}-bf16.gguf")
    assert base_path.endswith(f"{BASE}-bf16.gguf")
