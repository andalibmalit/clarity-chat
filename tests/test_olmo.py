"""OLMo 3 Instruct and Base through the managed llama-server (opt-in: `pytest -m olmo`).

Needs both models fetched (`clarity fetch olmo-3-7b-instruct`, `clarity fetch olmo-3-7b-base`).
Only one 7B model is loaded at a time: switching profiles stops the old server first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clarity_chat.models import check_weights_quick, lock_entry, source_weights
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


def on_transformers(folder: Path, base: str, max_tokens: int | None = None) -> str:
    """A copy of profiles/<base>.toml on the Transformers backend. Skips without the `hf`
    extra or the model's official weights. Used through the module's session, so switching
    to it stops llama-server first: only one 7B model is in memory."""
    pytest.importorskip("torch")
    text = (REPO / "profiles" / f"{base}.toml").read_text(encoding="utf-8")
    model = INSTRUCT if INSTRUCT in text else BASE
    try:
        source_weights(REPO, lock_entry(REPO, load_models(REPO)[model]))
    except ConfigError as e:
        pytest.skip(str(e))
    text = text.replace('backend = "llama.cpp"', 'backend = "transformers"')
    # The copy lives elsewhere, so the shipped profile's relative file paths must be absolute.
    text = text.replace('"../prompts/', f'"{(REPO / "prompts").as_posix()}/')
    if max_tokens is not None:
        text = text.replace("max_tokens = 1024", f"max_tokens = {max_tokens}")
    path = folder / f"{base}-transformers.toml"
    path.write_text(text, encoding="utf-8")
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


def test_the_base_model_in_text_mode_on_transformers_replays(session, tmp_path):
    """Needs A3: the base model's generation_config.json lists no end token. The replay runs
    after the model is unloaded, so it loads again."""
    session.new_conversation(on_transformers(tmp_path, "5-raw-base", max_tokens=64))
    record = session.say("Hello, who are you?").record
    checks_ok(record)
    assert session.backend.server_record["eos_token_id_from"].startswith("tokenizer")
    assert session.conversation is not None
    log = session.conversation.log.path
    session.backend.close()
    replayed = session.replay(log, record["call_id"])
    assert replayed["identical"], replayed


def test_logprobs_change_no_output_on_transformers(session, tmp_path):
    """On MPS, asking generate for its logits changes how the cache is cropped each step,
    and OLMo 3 has sliding-window layers; SmolLM2 and the CPU take neither path."""
    records = {}
    for n in (0, 5):
        name = on_transformers(tmp_path, "1b-blank", max_tokens=64)
        with open(name, "a", encoding="utf-8") as f:
            f.write(f"logprobs = {n}\n")  # [settings] is the file's last table
        session.new_conversation(name)
        records[n] = session.say("Name three rivers.").record
        checks_ok(records[n])
    assert records[0]["output_ids"] == records[5]["output_ids"]
    assert len(records[5]["logprobs"]["tokens"]) == len(records[5]["output_ids"])
    # The recompute check's own loop on the same loaded model: gap 0 on the model and
    # device the experiments use, not only on SmolLM2.
    from clarity_chat.recompute import measure

    measured = measure(records[5], session.backend.model, "transformers")["measured"]
    assert measured is not None
    assert (measured["largest_gap"], measured["top_changed"]) == (0.0, 0), measured


def test_switching_models_restarts_the_server(session):
    session.new_conversation("4-raw-instruct")
    instruct_path = session.backend.server_record["props"]["model_path"]
    session.new_conversation("5-raw-base")
    base_path = session.backend.server_record["props"]["model_path"]
    assert instruct_path.endswith(f"{INSTRUCT}-bf16.gguf")
    assert base_path.endswith(f"{BASE}-bf16.gguf")


def test_steering_olmo_3_instruct_on_mps(tmp_path):
    """The four steering checks of tests/test_hf.py on the model the experiments use, with a
    random direction the test writes across blocks 10 to 20. Loads about 15 GB at a time:
    each session closes before the next starts."""
    import numpy as np

    from tests.test_hf import check_steering, project, steered, write_vector

    try:
        source_weights(REPO, lock_entry(REPO, load_models(REPO)[INSTRUCT]))
    except ConfigError as e:
        pytest.skip(str(e))
    root = project(tmp_path)
    rng = np.random.default_rng(0)
    blocks = {}
    for n in range(10, 21):
        v = rng.standard_normal(4096).astype(np.float32)
        blocks[n] = v / np.linalg.norm(v)
    vectors = root / "local" / "vectors"
    zero_sha = write_vector(vectors / "zero.gguf", {n: v * 0 for n, v in blocks.items()}, "olmo3")
    real_sha = write_vector(vectors / "random.gguf", blocks, "olmo3")
    folder = tmp_path / "profiles"
    folder.mkdir()
    short = {"max_tokens": "48"}
    zero = steered(folder, "1a-stock", "local/vectors/zero.gguf", zero_sha, "8.0", **short)
    real = steered(folder, "1a-stock", "local/vectors/random.gguf", real_sha, "8.0", **short)
    check_steering(root, tmp_path, zero, real, "Describe a quiet harbour in two sentences.")
