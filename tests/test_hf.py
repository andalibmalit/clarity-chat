"""Live tests of the Transformers backend with SmolLM2-135M-Instruct (its official weights).

Run with `uv run pytest -m hf -v` (needs `uv sync --extra hf` and SmolLM2 fetched).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clarity_chat.models import lock_entry, source_weights
from clarity_chat.profiles import ConfigError, load_models
from clarity_chat.session import Session

pytestmark = pytest.mark.hf

REPO = Path(__file__).resolve().parents[1]
MODEL = "smollm2-135m-instruct"


def _available() -> str | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "torch is not installed (uv sync --extra hf)"
    try:
        source_weights(REPO, lock_entry(REPO, load_models(REPO)[MODEL]))
    except ConfigError as e:
        return str(e)
    return None


if (reason := _available()) is not None:
    pytestmark = [pytest.mark.hf, pytest.mark.skip(reason=reason)]


def profile(folder: Path, base: str, backend: str = "transformers", **extra: str) -> str:
    """A copy of profiles/<base>.toml with another backend; `extra` adds top-level keys or,
    for settings, replaces `name = value` lines."""
    text = (REPO / "profiles" / f"{base}.toml").read_text(encoding="utf-8")
    text = text.replace('backend = "llama.cpp"', f'backend = "{backend}"')
    lines = []
    for line in text.splitlines():
        name = line.split("=")[0].strip()
        lines.append(f"{name} = {extra.pop(name)}" if name in extra else line)
    head = "\n".join(f"{k} = {v}" for k, v in extra.items())
    path = folder / f"{base}-{backend}-{len(list(folder.iterdir()))}.toml"
    path.write_text(head + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def checks_ok(record: dict) -> None:
    failed = {name: check for name, check in record["checks"].items() if not check["ok"]}
    assert not failed, failed


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    yield session
    session.close()


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("profiles")


def test_text_mode_reply_stops_before_the_next_user_line(session, folder):
    session.new_conversation(profile(folder, "smol-text"))
    outcome = session.say("Hi! Who are you?")
    record = outcome.record
    checks_ok(record)
    assert record["backend"]["device"] == "mps"
    assert record["backend"]["attn_implementation"] == "sdpa"
    assert record["timings"]["predicted_ms"] > 0 and record["timings"]["predicted_n"] > 0
    assert record["stop_type"] in ("word", "eos", "limit")
    assert "\nUser:" not in record["output_text"]
    if record["stop_type"] == "word":
        assert record["stopping_word"] == "\nUser:"
    assert {e["name"] for e in record["not_applied"]} >= {"presence_penalty", "samplers"}


def test_chat_mode_reply_ends_with_an_end_token(session, folder):
    session.new_conversation(profile(folder, "smol-chat"))
    outcome = session.say("Say hello in one word.")
    checks_ok(outcome.record)
    assert outcome.record["stop_type"] == "eos"
    eos = session.backend.server_record["eos_token_id"]
    assert outcome.record["output_ids"][-1] in eos


def test_same_seed_twice_on_mps_gives_identical_output_ids(session, folder):
    name = profile(folder, "smol-text")
    outputs = []
    for _ in range(2):
        session.new_conversation(name)
        outputs.append(session.say("Tell me about rivers.").record["output_ids"])
    assert outputs[0] == outputs[1]


def test_replay_after_a_reload_on_mps(session, folder, tmp_path):
    session.new_conversation(profile(folder, "smol-text"))
    outcome = session.say("Name three birds.")
    other = Session(REPO, logs_dir=tmp_path)
    try:
        record = other.replay(session.conversation.log.path, outcome.record["call_id"])
    finally:
        other.close()
    assert record["identical"], record
    assert set(record["differences"]) <= {"client", "libraries"}, record["differences"]


def test_same_seed_twice_on_cpu_gives_identical_output_ids(tmp_path):
    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    name = profile(folder, "smol-text", device='"cpu"')
    try:
        outputs = []
        for _ in range(2):
            session.new_conversation(name)
            record = session.say("Tell me about rivers.").record
            checks_ok(record)
            assert record["backend"]["device"] == "cpu"
            outputs.append(record["output_ids"])
    finally:
        session.close()
    assert outputs[0] == outputs[1]


def test_both_backends_get_the_same_ids_and_greedy_outputs_are_compared(tmp_path, capsys):
    """The same profile and message give the same prompt IDs on both backends. Their greedy
    outputs are compared for information only: the runtimes compute differently."""
    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    records = {}
    try:
        for backend in ("llama.cpp", "transformers"):
            session.new_conversation(profile(folder, "smol-chat", backend, temperature="0.0"))
            records[backend] = session.say("What is the capital of France?").record
            if backend == "transformers":
                # The switch stopped llama-server first: only one model in memory.
                assert session.backends["llama.cpp", None].server_record is None
    finally:
        session.close()
    llama, hf = records["llama.cpp"], records["transformers"]
    assert llama["prompt_ids"] == hf["prompt_ids"]
    same = llama["output_ids"] == hf["output_ids"]
    with capsys.disabled():
        print(f"\ngreedy outputs identical across backends: {same}")
        print(f"  llama.cpp:    {llama['output_text']!r}")
        print(f"  transformers: {hf['output_text']!r}")


def test_each_backend_stops_on_its_own_end_tokens(tmp_path):
    """llama.cpp also stops on tokens it recognizes by name: SmolLM2's <|endoftext|> (0) and
    <reponame> (4, a fill-in-the-middle token) as well as <|im_end|> (2), where Transformers
    stops on generation_config.json's 2 alone. Each is shown and compared as a fact; this test
    notices if either backend changes its set."""
    from clarity_chat.replay import facts_from_call

    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    found = {}
    try:
        for backend in ("llama.cpp", "transformers"):
            session.new_conversation(profile(folder, "smol-chat", backend))
            server = session.backend.server_record
            found[backend] = facts_from_call({"model": {}}, server)["end_tokens"]
    finally:
        session.close()
    assert found == {"llama.cpp": [0, 2, 4], "transformers": [2]}


def recompute_logprobs(backend, prompt_ids: list[int], output_ids: list[int]) -> list[float]:
    """Each reply token's log-probability, recomputed as generation ran: the prompt in one
    pass, then one token per step with the cache."""
    import torch

    from clarity_chat.backends.transformers_hf import log_softmax

    model, device = backend.model, backend.device
    found = []
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=device)
        out = model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
        mask_length = len(prompt_ids)
        for token in output_ids:
            logprobs = log_softmax(out.logits[:, -1, :])
            found.append(logprobs[0, token].item())
            mask_length += 1
            out = model(
                torch.tensor([[token]], device=device),
                attention_mask=torch.ones(1, mask_length, dtype=torch.long, device=device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    return found


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("temperature", ["0.0", "0.8"])
def test_logprobs_change_no_output_and_match_a_step_by_step_recompute(
    tmp_path, device, temperature
):
    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    records = {}
    try:
        for n in (0, 5):
            name = profile(folder, "smol-chat", device=f'"{device}"', temperature=temperature)
            with open(name, "a", encoding="utf-8") as f:
                f.write(f"logprobs = {n}\n")  # [settings] is the file's last table
            session.new_conversation(name)
            records[n] = session.say("Name three rivers.").record
            checks_ok(records[n])
        logged = records[5]["logprobs"]
        recomputed = recompute_logprobs(
            session.backend, records[5]["prompt_ids"], records[5]["output_ids"]
        )
    finally:
        session.close()
    assert records[0]["output_ids"] == records[5]["output_ids"]
    assert "logprobs" not in records[0]
    assert logged["view"] == "raw logits, before any sampler" and logged["n"] == 5
    assert [t["id"] for t in logged["tokens"]] == records[5]["output_ids"]
    assert [t["logprob"] for t in logged["tokens"]] == recomputed


def test_olmo_3_instruct_on_mps_repeats_and_replays(tmp_path):
    """The model the experiments use: same seed twice, then a replay after a reload.
    Loads about 15 GB once per session; the first session is closed before the second
    loads, and no llama-server runs, so only one copy is in memory."""
    try:
        source_weights(REPO, lock_entry(REPO, load_models(REPO)["olmo-3-7b-instruct"]))
    except ConfigError as e:
        pytest.skip(str(e))
    folder = tmp_path / "profiles"
    folder.mkdir()
    name = profile(folder, "1a-stock", max_tokens="96")
    session = Session(REPO, logs_dir=tmp_path / "a")
    try:
        records = []
        for _ in range(2):
            conversation = session.new_conversation(name)
            records.append(session.say("Describe a quiet harbour in two sentences.").record)
        for record in records:
            checks_ok(record)
        log = conversation.log.path
    finally:
        session.close()
    assert records[0]["output_ids"] == records[1]["output_ids"]
    other = Session(REPO, logs_dir=tmp_path / "b")
    try:
        replayed = other.replay(log, records[1]["call_id"])
    finally:
        other.close()
    assert replayed["identical"], replayed
