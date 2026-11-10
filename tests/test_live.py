"""Live tests: the real llama-server (server.toml, GPU) and the fetched SmolLM2-135M-Instruct.

Run with `uv run pytest -m live`. Skipped if llama-server isn't built or SmolLM2 isn't fetched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from clarity_chat.backends.llama_cpp import LlamaCppBackend, server_binary
from clarity_chat.logs import read_records
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.render import Tokenizer
from clarity_chat.replay import find_call
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
    placement = record["placement"]
    assert placement["devices"], "no GPU device in use"
    if sys.platform == "darwin":
        assert placement["devices"][0]["name"].startswith("MTL"), placement
        # llama.cpp keeps the token-embedding table on the CPU, so only some buffers are Metal.
        assert any(b["buffer"].startswith("MTL") for b in placement["model_buffers"]), placement
    assert record["warnings"] == []
    assert "--no-context-shift" in record["argv"]


def test_the_server_refuses_requests_without_its_key(session):
    import httpx

    url = session.backend.server.url
    assert httpx.get(f"{url}/health", trust_env=False).status_code == 200  # public
    assert httpx.get(f"{url}/props", trust_env=False).status_code == 401
    body = {"prompt": [1, 2], "n_predict": 1}
    assert httpx.post(f"{url}/completion", json=body, trust_env=False).status_code == 401
    assert httpx.get(f"{url}/", trust_env=False).status_code != 200  # --no-webui
    assert "api_keys: (redacted by Clarity)" in session.backend.server.output_text()


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


def test_chat_mode_reply_ends_with_the_end_of_turn_token(session):
    session.new_conversation("smol-chat")  # same model: the server keeps running
    outcome = session.say("Name one colour.")
    record = outcome.record
    all_checks_ok(record)
    assert record["prompt_text"].endswith("<|im_start|>assistant\n")
    assert record["bos"]["prepended"] is False
    assert record["template"]["sha256"]
    assert record["stop_type"] == "eos", record["output_text"]
    im_end = session.tokenizer.tok.convert_tokens_to_ids("<|im_end|>")
    assert record["output_ids"][-1] == im_end
    follow_up = session.say("And another?").record
    all_checks_ok(follow_up)
    assert f"{record['output_text']}<|im_end|>" in follow_up["prompt_text"]


def test_same_seed_same_server_gives_identical_output_ids(session):
    """If this fails, no replay result means anything (the brief: stop and report)."""
    outputs = []
    for _ in range(2):
        session.new_conversation("smol-text")
        outputs.append(session.say("Tell me a short story about a robot.").record)
    assert outputs[0]["prompt_ids"] == outputs[1]["prompt_ids"]
    assert outputs[0]["output_ids"] == outputs[1]["output_ids"]
    assert outputs[0]["stop_type"] == outputs[1]["stop_type"]


def test_replay_after_a_longer_call_and_after_a_restart(session):
    """Replaying a call from an older log always follows other calls and a server restart."""
    session.new_conversation("smol-text")
    first = session.say("Tell me about the sea.").record
    ids, stop = first["prompt_ids"], first["stop"]
    settings = {name: s["value"] for name, s in first["settings"].items()}
    backend = session.backend
    backend.complete(counting_ids(300), {**settings, "max_tokens": 8}, [])  # longer, different
    assert backend.complete(ids, settings, stop).token_ids == first["output_ids"]
    assert session.model is not None
    backend.close()
    backend.start(session.model, check_weights_quick(REPO, session.model_lock), backend.n_ctx)
    assert backend.complete(ids, settings, stop).token_ids == first["output_ids"]


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
    backend = LlamaCppBackend(config, tmp_path_factory.mktemp("logs-small"))
    model = load_models(REPO)[MODEL]
    backend.start(model, check_weights_quick(REPO, lock_entry(REPO, model)), 256)
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
    from clarity_chat.session import room_left

    n_ctx, max_tokens = 256, MAX_TOKENS
    fitting = n_ctx - max_tokens - 1  # the largest prompt the client allows
    assert room_left(n_ctx, fitting, max_tokens) == 0
    assert room_left(n_ctx, fitting + 1, max_tokens) == -1

    full = small_context_backend.complete(counting_ids(fitting), SETTINGS, [])
    assert full.stop_type == "limit"
    assert full.response["truncated"] is False
    assert full.n_predicted == max_tokens

    # At prompt + max_tokens == n_ctx the server still returns every token but marks the
    # reply truncated; the client refuses this case so an allowed call is never marked.
    one_more = small_context_backend.complete(counting_ids(fitting + 1), SETTINGS, [])
    assert one_more.response["truncated"] is True
    assert one_more.n_predicted == max_tokens
    assert "context_full" in one_more.flags

    # One token further, the context really cuts the reply: n_predicted = n_ctx - n_prompt.
    two_more = small_context_backend.complete(counting_ids(fitting + 2), SETTINGS, [])
    assert two_more.response["truncated"] is True
    assert two_more.n_predicted == n_ctx - (fitting + 2)


def test_llama_arg_variables_are_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "1024")
    backend = LlamaCppBackend(load_server_config(REPO), tmp_path)
    model = load_models(REPO)[MODEL]
    try:
        record = backend.start(model, check_weights_quick(REPO, lock_entry(REPO, model)), 8192)
        assert record is not None
        assert "LLAMA_ARG_CTX_SIZE" in record["env_removed"]
        assert record["props"]["default_generation_settings"]["n_ctx"] == 8192
    finally:
        backend.close()
    assert backend.server is None


def test_a_batch_run_with_both_modes(tmp_path):
    from clarity_chat.batch import run_batch

    openings = tmp_path / "openings.txt"
    openings.write_text("Name a colour.\n---\nCount to three.\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    session = Session(REPO, logs_dir=run_dir)
    try:
        blind = run_batch(
            session, "run-live", run_dir, openings, ["smol-chat", "smol-text"], 2, lambda s: None
        )
    finally:
        session.close()
    rows = list(read_records(run_dir / "calls.jsonl"))
    calls = [r for r in rows if r["type"] == "call"]
    assert len(calls) == 2 * 2 * 2
    for call in calls:
        all_checks_ok(call)
        assert call["settings"]["seed"]["origin"] == "run"
    assert sum(r["type"] == "server_start" for r in rows) == 1  # one model, one server
    assert blind.read_text(encoding="utf-8").count("Rating:") == 8


def test_replay_on_the_running_server_is_identical_and_nothing_differs(session):
    session.say("Name a fruit.")
    record = session.replay_current()
    assert record["identical"], record
    assert record["differences"] == {}, record["differences"]
    all_checks_ok(record)


def test_replay_from_the_log_on_a_new_server_with_logged_flags(session, tmp_path):
    outcome = session.say("Name a river.")
    path = session.conversation.log.path
    other = Session(REPO, logs_dir=tmp_path)
    _, server = find_call(path, outcome.record["call_id"])
    assert server is not None
    other.use_server_flags(server["flags"])
    try:
        record = other.replay(path, outcome.record["call_id"], flags_from="logged")
    finally:
        other.close()
    assert record["identical"], record
    assert record["server_id"] != server["server_id"]
    # A new server process: only facts that can't change output may differ.
    assert set(record["differences"]) <= {"client", "libraries"}, record["differences"]


GGUF_ENTRY = """\
[smol-q8]
# A third-party quantization, used only to test the GGUF-download route.
route = "gguf"
repo = "bartowski/SmolLM2-135M-Instruct-GGUF"
revision = "09816acd5d99df7be770d85ea30822623dab342c"
file = "SmolLM2-135M-Instruct-Q8_0.gguf"
sha256 = "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"
tokenizer_repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer_revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
"""


def test_the_gguf_download_route_end_to_end(tmp_path, monkeypatch):
    """Downloads 145 MB once (then the Hugging Face cache has it)."""
    import shutil

    import huggingface_hub.constants

    from clarity_chat.models import fetch

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)  # conftest sets it

    root = tmp_path / "project"
    (root / "profiles").mkdir(parents=True)
    shutil.copy(REPO / "defaults.toml", root)
    server = (REPO / "server.toml").read_text(encoding="utf-8")
    llama_dir = (REPO / "vendor" / "llama.cpp").as_posix()
    server = server.replace('llama_cpp_dir = "vendor/llama.cpp"', f'llama_cpp_dir = "{llama_dir}"')
    (root / "server.toml").write_text(server, encoding="utf-8")
    (root / "models.toml").write_text(GGUF_ENTRY, encoding="utf-8")
    profile = (REPO / "profiles" / "smol-text.toml").read_text(encoding="utf-8")
    profile = profile.replace('model = "smollm2-135m-instruct"', 'model = "smol-q8"')
    (root / "profiles" / "q8.toml").write_text(profile, encoding="utf-8")

    entry = fetch(root, load_models(root)["smol-q8"], load_server_config(root))
    assert entry["gguf"]["path"].startswith("~/"), entry["gguf"]["path"]  # the HF cache
    assert entry["gguf"]["path"].endswith("/SmolLM2-135M-Instruct-Q8_0.gguf")  # not the blob
    session = Session(root, logs_dir=tmp_path / "logs")
    try:
        session.new_conversation("q8")
        outcome = session.say("Name a colour.")
    finally:
        session.close()
    all_checks_ok(outcome.record)  # includes the echo check: tokenizer and GGUF agree
    assert outcome.record["model"]["gguf_sha256"] == entry["gguf"]["sha256"]
