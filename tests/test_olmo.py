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


def test_replay_after_a_long_call_and_after_a_restart(session):
    """The long call wraps OLMo's sliding-window KV ring (4608 cells) during prefill."""
    session.new_conversation("1b-blank")
    first = session.say("Name three rivers.").record
    ids, stop = first["prompt_ids"], first["stop"]
    settings = {name: s["value"] for name, s in first["settings"].items()}
    assert session.tokenizer is not None
    long_ids = session.tokenizer.encode(" word" * 6000)[:5000]
    assert len(long_ids) == 5000
    backend = session.backend
    backend.complete(long_ids, {**settings, "max_tokens": 1}, [])
    assert backend.complete(ids, settings, stop).token_ids == first["output_ids"]
    assert session.model is not None
    backend.close()
    backend.start(session.model, check_weights_quick(REPO, session.model_lock), backend.n_ctx)
    assert backend.complete(ids, settings, stop).token_ids == first["output_ids"]


@pytest.mark.parametrize("profile", ["4-raw-instruct", "5-raw-base"])
def test_raw_text_mode_on_both_models(session, profile):
    session.new_conversation(profile)
    record = session.say("Hello, who are you?").record
    checks_ok(record)
    if record["stop_type"] == "word":
        assert "\nUser:" not in record["output_text"]


def on_transformers(folder: Path, base: str) -> str:
    """A copy of profiles/<base>.toml on the Transformers backend (needs `uv sync --extra hf`)."""
    pytest.importorskip("torch")
    text = (REPO / "profiles" / f"{base}.toml").read_text(encoding="utf-8")
    path = folder / f"{base}-transformers.toml"
    path.write_text(text.replace('backend = "llama.cpp"', 'backend = "transformers"'), "utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("profile", "llama_cpp", "transformers"),
    [
        ("4-raw-instruct", [100257, 100265], [100257, 100265]),
        # The base model's generation_config.json lists no end token, so Transformers takes the
        # tokenizer's, <|endoftext|>; llama.cpp also finds <|im_end|> by name.
        ("5-raw-base", [100257, 100265], [100257]),
    ],
)
def test_each_backend_stops_on_its_own_end_tokens(
    session, tmp_path, profile, llama_cpp, transformers
):
    from clarity_chat.replay import facts_from_call

    found = {}
    for name in (profile, on_transformers(tmp_path, profile)):
        session.new_conversation(name)
        server = session.backend.server_record
        found[server["backend"]] = facts_from_call({"model": {}}, server)["end_tokens"]
    assert found == {"llama.cpp": llama_cpp, "transformers": transformers}


def test_switching_models_restarts_the_server(session):
    session.new_conversation("4-raw-instruct")
    instruct_path = session.backend.server_record["props"]["model_path"]
    session.new_conversation("5-raw-base")
    base_path = session.backend.server_record["props"]["model_path"]
    assert instruct_path.endswith(f"{INSTRUCT}-bf16.gguf")
    assert base_path.endswith(f"{BASE}-bf16.gguf")
