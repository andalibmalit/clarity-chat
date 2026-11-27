"""Llama 3.1 8B Instruct and base on both backends (opt-in: `pytest -m llama`).

Needs the models fetched (`clarity fetch llama-3.1-8b-instruct`, `clarity fetch
llama-3.1-8b-base`; both are gated, see the getting-started guide), and the `hf` extra for
the Transformers tests. Each model is about 16 GB, and only one is loaded at a time:
switching profiles stops the old model first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clarity_chat.models import check_weights_quick, lock_entry, source_weights
from clarity_chat.profiles import ConfigError, load_models
from clarity_chat.session import Session

REPO = Path(__file__).resolve().parents[1]
INSTRUCT, BASE = "llama-3.1-8b-instruct", "llama-3.1-8b-base"
MODEL = {"llama-3.1-8b-chat": INSTRUCT, "llama-3.1-8b-base": BASE}
BOS = 128000  # <|begin_of_text|>
# Written by the Instruct template even with no system message; the date is its fixed default.
HEADER = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
)

pytestmark = pytest.mark.llama


def need(model: str, backend: str = "llama.cpp") -> None:
    """Skip unless `model` is fetched (and, for Transformers, torch and the official weights)."""
    try:
        entry = lock_entry(REPO, load_models(REPO)[model])
        if backend == "transformers":
            pytest.importorskip("torch")
            source_weights(REPO, entry)
        else:
            check_weights_quick(REPO, entry)
    except ConfigError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    yield session
    session.close()


def checks_ok(record: dict) -> None:
    failed = {name: check for name, check in record["checks"].items() if not check["ok"]}
    assert not failed, failed


def copy(folder: Path, base: str, backend: str, **extra: str) -> str:
    """profiles/<base>.toml on `backend`, outside the repo, so its prompt file path is made
    absolute."""
    from tests.test_hf import profile

    path = Path(profile(folder, base, backend, **extra))
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"../prompts/', f'"{(REPO / "prompts").as_posix()}/'), "utf-8")
    return str(path)


@pytest.mark.parametrize("profile", ["llama-3.1-8b-chat", "llama-3.1-8b-base"])
def test_one_bos_the_dated_header_and_the_same_prompt_ids_on_both_backends(
    session, tmp_path, profile
):
    need(MODEL[profile])
    need(MODEL[profile], "transformers")
    records = {}
    for backend in ("llama.cpp", "transformers"):
        session.new_conversation(copy(tmp_path, profile, backend, max_tokens="16"))
        records[backend] = session.say("Hello, who are you?").record
        checks_ok(records[backend])
    llama, hf = records["llama.cpp"], records["transformers"]
    assert llama["prompt_ids"] == hf["prompt_ids"]
    ids, text = llama["prompt_ids"], llama["prompt_text"]
    assert ids[0] == BOS and ids.count(BOS) == 1
    if profile == "llama-3.1-8b-chat":
        # The template writes the BOS as text; Clarity's encoding adds none of its own.
        assert not llama["bos"]["prepended"] and text.startswith("<|begin_of_text|>" + HEADER)
        start = text.index("Today Date")
        spans = [s for s in llama["message_spans"] if s["chars"][0] <= start < s["chars"][1]]
        assert [s["kind"] for s in spans] == ["template"]  # shown as the template's text
    else:
        # No template: the tokenizer's BOS is prepended, and no header is written.
        assert llama["bos"]["prepended"] and "Today Date" not in text


def test_a_chat_at_16384_runs_llama_3_1_at_that_size(session, tmp_path):
    """v0.4.6 L1 (v0.4.4's carried S2): a real model at 16,384, under its trained maximum
    of 131,072. `/set ctx_size 16384` is accepted with the chat as its origin, the server
    starts at it, and the call runs at it with every check passing. Before the steering
    test, which closes the module's session."""
    from clarity_chat.logs import read_records
    from clarity_chat.profiles import Setting

    need(INSTRUCT)
    session.new_conversation(copy(tmp_path, "llama-3.1-8b-chat", "llama.cpp", max_tokens="8"))
    session.set_setting("ctx_size", "16384")
    conversation = session.conversation
    assert conversation is not None
    assert conversation.ctx_size == Setting(16384, "chat")
    record = session.say("Reply with one word: ready.").record
    checks_ok(record)
    assert record["n_ctx"] == 16384
    starts = {
        r["server_id"]: r
        for r in read_records(conversation.log.path)
        if r["type"] == "server_start"
    }
    argv = starts[record["backend"]["server_id"]]["argv"]
    assert argv[argv.index("--ctx-size") + 1] == "16384", argv


def test_steering_llama_3_1_instruct_on_mps(session, tmp_path):
    """The four steering checks of tests/test_hf.py on Llama 3.1 8B Instruct (model type
    `llama`), with a random direction the test writes across blocks 10 to 20, as the OLMo 3
    test does. The module's session closes first, so only one 16 GB model is loaded."""
    import numpy as np

    session.close()
    need(INSTRUCT, "transformers")

    from tests.test_hf import check_steering, project, steered, write_vector

    root = project(tmp_path)
    rng = np.random.default_rng(0)
    blocks = {}
    for n in range(10, 21):
        v = rng.standard_normal(4096).astype(np.float32)
        blocks[n] = v / np.linalg.norm(v)
    vectors = root / "local" / "vectors"
    zero_sha = write_vector(vectors / "zero.gguf", {n: v * 0 for n, v in blocks.items()}, "llama")
    real_sha = write_vector(vectors / "random.gguf", blocks, "llama")
    folder = tmp_path / "profiles"
    folder.mkdir()
    short = {"max_tokens": "48"}
    base = "llama-3.1-8b-chat"
    zero = steered(folder, base, "local/vectors/zero.gguf", zero_sha, "8.0", **short)
    real = steered(folder, base, "local/vectors/random.gguf", real_sha, "8.0", **short)
    check_steering(root, tmp_path, zero, real, "Describe a quiet harbour in two sentences.")


def test_steering_llama_3_1_instruct_on_llama_cpp(session, tmp_path):
    """X3 (v0.4.7): B3's live checks on Llama 3.1 8B Instruct on llama.cpp, with the random
    direction across blocks 10 to 20 that the Transformers test above writes. The module's
    session closes first, so only one model is loaded."""
    import numpy as np

    from tests.test_hf import steered, write_vector
    from tests.test_live import check_steering_on_llama_cpp, steering_project

    session.close()  # a later test that needs it starts its model again
    need(INSTRUCT)
    root = steering_project(tmp_path)
    rng = np.random.default_rng(0)
    blocks = {}
    for n in range(10, 21):
        v = rng.standard_normal(4096).astype(np.float32)
        blocks[n] = v / np.linalg.norm(v)
    vectors = root / "local" / "vectors"
    zero_sha = write_vector(vectors / "zero.gguf", {n: v * 0 for n, v in blocks.items()}, "llama")
    real_sha = write_vector(vectors / "random.gguf", blocks, "llama")
    folder = tmp_path / "profiles"
    folder.mkdir()
    short = {"backend": "llama.cpp", "max_tokens": "48"}
    base = "llama-3.1-8b-chat"
    zero = steered(folder, base, "local/vectors/zero.gguf", zero_sha, "8.0", **short)
    real = steered(folder, base, "local/vectors/random.gguf", real_sha, "8.0", **short)
    message = "Describe a quiet harbour in two sentences."
    check_steering_on_llama_cpp(root, tmp_path / "logs", zero, real, message)
