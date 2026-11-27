"""Live tests of the Transformers backend with SmolLM2-135M-Instruct (its official weights).

Run with `uv run pytest -m hf -v` (needs `uv sync --extra hf` and SmolLM2 fetched).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clarity_chat.models import lock_entry, source_weights
from clarity_chat.profiles import ConfigError, load_models
from clarity_chat.session import Session
from tests import contract

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


def test_mps_with_the_cpu_fallback_switched_on_is_refused_on_the_real_device(monkeypatch, tmp_path):
    """The offline test fakes MPS being available; this one starts the real device."""
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    try:
        with pytest.raises(ConfigError, match="PYTORCH_ENABLE_MPS_FALLBACK is set"):
            session.new_conversation(profile(folder, "smol-text"))
    finally:
        session.close()


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


def test_the_recompute_check_on_an_export_of_a_chat_with_probabilities(tmp_path):
    """C2, as a notebook would run it: the model loaded fresh in bf16 on the CPU gives a gap
    of 0 and no changed top choices; a float32 load is named as a different precision and
    moves the numbers."""
    import torch
    from transformers import AutoModelForCausalLM

    from clarity_chat.export import export_log
    from clarity_chat.recompute import check

    model = load_models(REPO)[MODEL]
    pins = {"repo": model.repo, "revision": model.revision}
    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    try:
        name = profile(folder, "smol-chat", device='"cpu"', temperature="0.8")
        with open(name, "a", encoding="utf-8") as f:
            f.write("logprobs = 5\n")  # [settings] is the file's last table
        session.new_conversation(name)
        session.say("Name three rivers.")
        assert session.conversation is not None
        export = Path(export_log(session, session.conversation.log.path)["path"])
    finally:
        session.close()
    weights = source_weights(REPO, lock_entry(REPO, model))
    reports = {}
    for dtype in (torch.bfloat16, torch.float32):
        loaded = AutoModelForCausalLM.from_pretrained(
            weights, dtype=dtype, attn_implementation="sdpa"
        )
        reports[dtype] = check(export, loaded, **pins)
    same, wider = reports[torch.bfloat16], reports[torch.float32]
    assert [f["fact"] for f in same.groups[0]["facts"] if not f["same"]] == [], str(same)
    measured = same.calls[0]["measured"]
    assert measured["largest_gap"] == 0.0 and measured["top_changed"] == 0
    assert [f["fact"] for f in wider.groups[0]["facts"] if not f["same"]] == ["precision"]
    assert wider.calls[0]["measured"]["largest_gap"] > 0


# --- steering -------------------------------------------------------------------------------

REPENG = REPO / "tests" / "fixtures" / "steering" / "smollm2-repeng.gguf"
REPENG_SHA256 = "ed3236f9f64dc4714b402b9a32f7e209471d3108033eff28dc4ad9230d1724f2"


def project(tmp_path: Path) -> Path:
    """A scratch project folder with this checkout's config and models, so a test's vectors
    sit inside a project folder without touching your local/."""
    import shutil

    root = tmp_path / "project"
    (root / "local" / "vectors").mkdir(parents=True)
    for name in ("defaults.toml", "server.toml", "models.toml", "models.lock"):
        shutil.copy(REPO / name, root / name)
    return root


def write_vector(path: Path, blocks: dict, model_type: str) -> str:
    """Write a vector with gguf's reference writer; returns its sha256."""
    import hashlib

    import gguf

    writer = gguf.GGUFWriter(path, "controlvector")
    writer.add_string("controlvector.model_hint", model_type)
    writer.add_uint32("controlvector.layer_count", len(blocks))
    for n, values in blocks.items():
        writer.add_tensor(f"direction.{n}", values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steered(folder: Path, base: str, file: str, sha256: str, scale: str, **extra: str) -> str:
    """profile(folder, base, ...) with a [steering] table and steering_scale."""
    path = Path(profile(folder, base, **extra))
    text = path.read_text(encoding="utf-8").replace(
        "[settings]", f"[settings]\nsteering_scale = {scale}"
    )
    text += f'\n[steering]\nfile = "{file}"\nsha256 = "{sha256}"\n'
    path.write_text(text, encoding="utf-8")
    return str(path)


def check_steering(root: Path, tmp_path: Path, zero: str, real: str, message: str) -> None:
    """The four live checks: a zero vector at a non-zero scale changes nothing, greedy and
    seeded (so the hook path itself adds nothing); the real vector changes the reply; a
    reply cancelled mid-way leaves the model unsteered; a replay after a reload is steered
    and identical."""
    import threading

    session = Session(root, logs_dir=tmp_path / "a")
    try:

        def run(name: str, temperature: str, scale: str | None = None) -> dict:
            session.new_conversation(name)
            session.set_setting("temperature", temperature)
            if scale is not None:
                session.set_setting("steering_scale", scale)
            return session.say(message).record

        for temperature in ("0", "0.8"):  # greedy, then sampled with the profile's seed
            plain = run(zero, temperature, scale="0")
            zeroed = run(zero, temperature)
            assert plain["request"]["steering"] is None
            assert zeroed["request"]["steering"]["scale"] != 0
            checks_ok(zeroed)
            assert zeroed["checks"]["steering"]["ok"]
            assert zeroed["output_ids"] == plain["output_ids"]
        # A reply that ends on its end token: on MPS, Transformers runs one more pass then,
        # and undoes it. The zero vector keeps the reply ending where an unsteered one does.
        session.new_conversation(zero)
        session.set_setting("temperature", "0")
        session.set_setting("max_tokens", "256")
        ended = session.say("Reply with one word: yes.").record
        assert ended["stop_type"] == "eos", ended["output_text"]
        checks_ok(ended)
        plain = run(real, "0", scale="0")
        steered_record = run(real, "0")
        checks_ok(steered_record)
        assert steered_record["output_ids"] != plain["output_ids"]
        assert session.conversation is not None
        log = session.conversation.log.path
        cancel = threading.Event()
        session.new_conversation(real)
        session.set_setting("temperature", "0")
        cut = session.say(message, on_text=lambda text: cancel.set(), cancel=cancel).record
        assert cut["stop_type"] == "cancelled" and cut["output_ids"]
        assert "steering" not in cut["checks"]
        assert run(real, "0", scale="0")["output_ids"] == plain["output_ids"]
    finally:
        session.close()
    other = Session(root, logs_dir=tmp_path / "b")
    try:
        replayed = other.replay(log, steered_record["call_id"])
    finally:
        other.close()
    assert replayed["identical"], replayed
    assert replayed["request"]["steering"] == steered_record["request"]["steering"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_steering_smollm2_with_repengs_vector(tmp_path, device):
    import shutil

    import numpy as np

    from clarity_chat.steering import load

    root = project(tmp_path)
    shutil.copy(REPENG, root / "local" / "vectors" / "repeng.gguf")
    blocks = load(REPENG, REPENG_SHA256).directions
    zeros = {n: np.zeros_like(v) for n, v in blocks.items()}
    zero_sha = write_vector(root / "local" / "vectors" / "zero.gguf", zeros, "llama")
    folder = tmp_path / "profiles"
    folder.mkdir()
    on = {"device": f'"{device}"'}
    zero = steered(folder, "smol-chat", "local/vectors/zero.gguf", zero_sha, "4.0", **on)
    real = steered(folder, "smol-chat", "local/vectors/repeng.gguf", REPENG_SHA256, "4.0", **on)
    check_steering(root, tmp_path, zero, real, "Tell me about rivers.")


# --- making a vector -------------------------------------------------------------------------


def make_and_steer(
    root: Path, tmp_path: Path, capsys, base: str, message: str, then=None, **extra: str
):
    """`clarity vector make` on the made-up 8-pair file, twice, then the printed snippet
    pasted into a copy of profiles/<base>.toml with steering_scale = 1.0. Checks both files;
    that the read hook gives exactly Transformers' hidden_states at every block below the
    last; that the snippet's profile steers (the steering check counts each pass) and
    differs from unsteered at seed 42; and that the steered call replays identically. Prints
    each make's time, the vector's path and the largest remake difference per block (run
    with -s); nothing about that difference is asserted until it is known (decision 11).
    `then(session, model, vector, profile)`, if given, runs on the make's loaded model.
    Returns the first make's record."""
    import json
    import time

    import numpy as np
    import torch

    from clarity_chat import steering
    from clarity_chat.maker import make, reading
    from clarity_chat.steering import load
    from tests.test_maker_render import PAIRS

    folder = tmp_path / "profiles"
    folder.mkdir()
    plain = profile(folder, base, **extra)
    vectors, made, said = root / "local" / "vectors", {}, []
    session = Session(root, logs_dir=tmp_path / "make")
    try:
        for stem in ("brief", "again"):
            (root / f"{stem}.toml").write_text(PAIRS, encoding="utf-8", newline="\n")
            started = time.monotonic()
            assert make(session, root / f"{stem}.toml", plain) == 0
            said.append(f"{base} {extra}: made {stem} in {time.monotonic() - started:.1f} s")
            out = capsys.readouterr().out
            name = f"{stem}-{Path(plain).stem}"
            record = json.loads((vectors / f"{name}.json").read_text(encoding="utf-8"))
            vector = load(vectors / f"{name}.gguf", record["vector"]["sha256"])
            made[stem] = (out[out.index("[steering]") :], record, vector)
        model: Any = session.backend.model  # ty: ignore[unresolved-attribute]
        ids = made["brief"][1]["prompts"][0]["ids"]
        with reading(model) as rows, torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids], device=model.device),
                output_hidden_states=True,
                use_cache=False,
            )
        n = len(steering.model_blocks(model))
        assert sorted(rows) == list(range(1, n))
        for b in range(1, n - 1):  # the last entry of hidden_states comes after the final norm
            expected = out.hidden_states[b + 1][0, -1].to(device="cpu", dtype=torch.float32)
            assert np.array_equal(rows[b], expected.numpy()), b
        if then is not None:
            then(session, model, made["brief"][2], plain)
    finally:
        session.close()
    snippet, record, vector = made["brief"]
    again = made["again"][2]
    largest = {
        b: float(np.abs(vector.directions[b] - again.directions[b]).max())
        for b in record["vector"]["blocks"]
    }
    said.append(f"{base} {extra}: the vector: {root / record['vector']['file']}")
    said.append(f"{base} {extra}: the largest remake difference per block: {largest}")
    steered_name = Path(profile(folder, base, **extra))
    text = steered_name.read_text(encoding="utf-8")
    text = text.replace("[settings]", "[settings]\nsteering_scale = 1.0", 1) + "\n" + snippet
    steered_name.write_text(text, encoding="utf-8")
    session = Session(root, logs_dir=tmp_path / "chat")
    try:
        session.new_conversation(str(steered_name))
        steered_record = session.say(message).record
        checks_ok(steered_record)
        assert steered_record["checks"]["steering"]["ok"]
        assert steered_record["request"]["steering"] == {
            "file": record["vector"]["file"],
            "sha256": record["vector"]["sha256"],
            "scale": 1.0,
            "layers": [n // 2],
        }
        assert session.conversation is not None
        log = session.conversation.log.path
        session.new_conversation(str(steered_name))
        session.set_setting("steering_scale", "0")
        unsteered = session.say(message).record
        seeds = {r["settings"]["seed"]["value"] for r in (steered_record, unsteered)}
        assert seeds == {42}
        assert steered_record["output_ids"] != unsteered["output_ids"]
    finally:
        session.close()
    other = Session(root, logs_dir=tmp_path / "replay")
    try:
        replayed = other.replay(log, steered_record["call_id"])
    finally:
        other.close()
    assert replayed["identical"], replayed
    with capsys.disabled():
        print("\n" + "\n".join(said))
    return record


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_making_a_vector_on_smollm2(tmp_path, capsys, device):
    root = project(tmp_path)
    make_and_steer(
        root, tmp_path, capsys, "smol-chat", "Tell me about rivers.", device=f'"{device}"'
    )


NUMBERING_PAIRS = {1: (2,), 15: (14, 16), 29: (28,)}  # block 0 can't be steered; 30 isn't


def gap(top: dict[int, float], reference: Any) -> float:
    """The largest difference in log-probability over `top`'s tokens, the measure of
    "nearer" in X1. `reference` gives a log-probability for every token in the vocabulary."""
    return max(abs(logprob - float(reference[token])) for token, logprob in top.items())


def test_block_numbering_matches_on_both_backends(tmp_path, capsys):
    """X1: llama.cpp's direction.N steers the block that Transformers' direction.N steers.
    repeng's block 15 direction is written alone at block b and added at scale 64. llama.cpp's
    first-token top 10 must be nearer to Transformers' at b than at each neighbour, and nearer
    than to llama.cpp's own unsteered numbers. Transformers runs in float32 on the CPU, where
    the two backends differ least unsteered (probe 2: 0.032; in bf16, 0.114)."""
    import httpx
    import torch
    from transformers import AutoModelForCausalLM

    from clarity_chat.backends.llama_cpp import LlamaCppBackend
    from clarity_chat.steering import Vector, applied, load

    root = project(tmp_path)
    for name in ("models", "vendor"):
        (root / name).symlink_to(REPO / name, target_is_directory=True)
    direction = load(REPENG, REPENG_SHA256).directions[15]
    folder = tmp_path / "profiles"
    folder.mkdir()
    settings = {"backend": "llama.cpp", "temperature": "0.0", "max_tokens": "1"}
    message = "Tell me about rivers."
    weights = source_weights(REPO, lock_entry(REPO, load_models(REPO)[MODEL]))
    model = AutoModelForCausalLM.from_pretrained(
        weights, dtype=torch.float32, attn_implementation="sdpa"
    )
    vocabulary = model.config.vocab_size
    session = Session(root, logs_dir=tmp_path / "logs")

    def llama_cpp(profile_path: str, n_probs: int) -> tuple[list[int], dict[int, float]]:
        """Start llama-server through Clarity, then ask it directly for the first token's
        top `n_probs` log-probabilities, before sampling."""
        session.new_conversation(profile_path)
        record = session.say(message).record
        checks_ok(record)
        backend = session.backends["llama.cpp", None]
        assert isinstance(backend, LlamaCppBackend) and backend.server is not None
        server = backend.server
        body = {
            "prompt": record["prompt_ids"],
            "n_predict": 1,
            "n_probs": n_probs,
            "post_sampling_probs": False,
            "cache_prompt": False,
            "stream": False,
        }
        response = httpx.post(
            f"{server.url}/completion",
            headers=server.headers,
            json=body,
            timeout=120,
            trust_env=False,
        )
        response.raise_for_status()
        top = response.json()["completion_probabilities"][0]["top_logprobs"]
        return record["prompt_ids"], {item["id"]: item["logprob"] for item in top}

    def transformers(ids: list[int], block: int | None) -> torch.Tensor:
        """Transformers' log-probabilities for the first token, steered at `block`."""
        inputs = torch.tensor([ids])
        if block is None:
            with torch.no_grad():
                return torch.log_softmax(model(inputs).logits[0, -1].float(), dim=-1)
        vector = Vector(REPENG, REPENG_SHA256, {block: direction}, "llama")
        with applied(model, vector, 64.0, [block]) as runs, torch.no_grad():
            logprobs = torch.log_softmax(model(inputs).logits[0, -1].float(), dim=-1)
        assert runs == {block: 1}
        return logprobs

    said = []
    try:
        plain_path = profile(folder, "smol-chat", **settings)
        ids, plain_top = llama_cpp(plain_path, 10)
        _, plain_all = llama_cpp(plain_path, vocabulary)  # every token, for gap() to read
        hf_plain = transformers(ids, None)
        unsteered = gap(plain_top, hf_plain)
        said.append(f"unsteered, llama.cpp against Transformers (float32, CPU): {unsteered:.4f}")
        for b, neighbours in NUMBERING_PAIRS.items():
            at = b  # where llama.cpp's file puts the direction; b + 1 is the seen-failing run
            file = f"local/vectors/at-{at}.gguf"
            sha256 = write_vector(root / file, {at: direction}, "llama")
            path = Path(steered(folder, "smol-chat", file, sha256, "64.0", **settings))
            path.write_text(path.read_text(encoding="utf-8") + f"layers = [{at}]\n", "utf-8")
            steered_ids, top = llama_cpp(str(path), 10)
            assert steered_ids == ids
            hf = {n: transformers(ids, n) for n in (b, *neighbours)}
            at_b = gap(top, hf[b])
            at_neighbours = {n: gap(top, hf[n]) for n in neighbours}
            own = gap(top, plain_all)
            moved = max(abs(float(hf[b][t]) - float(hf_plain[t])) for t in top)
            said.append(
                f"block {b}: llama.cpp against Transformers at {b}: {at_b:.4f}; "
                + "; ".join(f"at {n}: {g:.4f}" for n, g in at_neighbours.items())
                + f"; against its own unsteered: {own:.4f}; "
                f"Transformers steered against unsteered: {moved:.4f} "
                f"({moved / unsteered if unsteered else float('inf'):.1f} times the unsteered gap)"
            )
            for n, g in at_neighbours.items():
                assert at_b < g, f"block {b}: nearer at {n} ({g:.4f}) than at {b} ({at_b:.4f})"
            assert at_b < own, f"block {b}: nearer its own unsteered numbers ({own:.4f})"
            assert moved >= 10 * unsteered, f"block {b}: steering moved Transformers too little"
    finally:
        session.close()
        with capsys.disabled():
            print("\n" + "\n".join(said))


def test_the_recompute_check_on_a_steered_export(tmp_path):
    """C1, as a notebook would run it: under `applied` with the call's vector, scale and
    blocks, a gap of 0 and no changed top choices; without it, steering is listed and the
    numbers move."""
    import shutil

    import torch
    from transformers import AutoModelForCausalLM

    from clarity_chat.export import export_log
    from clarity_chat.recompute import check
    from clarity_chat.steering import applied, load

    root = project(tmp_path)
    shutil.copy(REPENG, root / "local" / "vectors" / "repeng.gguf")
    folder = tmp_path / "profiles"
    folder.mkdir()
    name = steered(
        folder, "smol-chat", "local/vectors/repeng.gguf", REPENG_SHA256, "4.0", device='"cpu"'
    )
    text = (
        Path(name).read_text(encoding="utf-8").replace("[settings]", "[settings]\nlogprobs = 5", 1)
    )
    Path(name).write_text(text, encoding="utf-8")
    session = Session(root, logs_dir=tmp_path / "logs")
    try:
        session.new_conversation(name)
        session.say("Name three rivers.")
        assert session.conversation is not None
        export = Path(export_log(session, session.conversation.log.path)["path"])
    finally:
        session.close()
    model = load_models(REPO)[MODEL]
    pins = {"repo": model.repo, "revision": model.revision}
    weights = source_weights(REPO, lock_entry(REPO, model))
    loaded = AutoModelForCausalLM.from_pretrained(
        weights, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    with applied(loaded, load(REPENG, REPENG_SHA256), 4.0):
        same = check(export, loaded, **pins)
    plain = check(export, loaded, **pins)
    assert [f["fact"] for f in same.groups[0]["facts"] if not f["same"]] == [], str(same)
    measured = same.calls[0]["measured"]
    assert measured["largest_gap"] == 0.0 and measured["top_changed"] == 0
    assert [f["fact"] for f in plain.groups[0]["facts"] if not f["same"]] == ["steering"]
    assert plain.calls[0]["measured"]["largest_gap"] > 0


def test_the_backend_contract_on_transformers(tmp_path):
    """The call record's shape for each stop type, kept in fixtures/backend_shapes.json."""
    session = Session(REPO, logs_dir=tmp_path / "logs")
    (tmp_path / "profiles").mkdir()
    chat, text = (profile(tmp_path / "profiles", base) for base in ("smol-chat", "smol-text"))
    try:
        shapes = contract.run_scenarios(session, chat=chat, text=text)
    finally:
        session.close()
    contract.check_or_write("transformers", shapes)


def test_an_attachment_gives_the_same_prompt_ids_on_both_backends(tmp_path):
    """v0.3.1: the attachment's block is in the content, which both backends get as IDs."""
    from clarity_chat.documents import extract

    session = Session(REPO, logs_dir=tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    notes = b"Made-up trail notes.\nThree cairns mark the ridge.\nNo water after the pass.\n"
    records = {}
    try:
        for backend in ("llama.cpp", "transformers"):
            session.new_conversation(profile(folder, "smol-chat", backend, temperature="0.0"))
            session.attach(extract(notes, "trail-notes.md"), "lines", 1, 2)
            records[backend] = session.say("What marks the ridge?").record
            checks_ok(records[backend])
    finally:
        session.close()
    llama, hf = records["llama.cpp"], records["transformers"]
    assert "Excerpt: lines 1 to 2 of 3" in hf["prompt_text"]
    assert llama["prompt_ids"] == hf["prompt_ids"]


@pytest.mark.parametrize("method", ["words", "meaning"])
def test_a_searched_message_gives_the_same_prompt_ids_on_both_backends(
    tmp_path, monkeypatch, method
):
    """v0.3.2 (Checkpoint 2's S4): the found parts are in the content, which both backends
    get as IDs. The library is a scratch one: the repo's local/documents/ is never read.
    v0.4.1 (Checkpoint 1's S4): by meaning too, with FakeEmbedder (the parts block is the
    same insert path whichever method chose its parts)."""
    from clarity_chat.documents import library as library_module
    from clarity_chat.documents import library_add
    from tests.fakes import FakeEmbedder

    scratch = tmp_path / "scratch"
    trail = b"## Ridge\nThree cairns mark the ridge.\n\n## Pass\nNo water after the pass.\n"
    library_add(scratch, trail, "trail.md")
    monkeypatch.setattr(library_module, "LIBRARY", scratch / "local" / "documents")
    session = Session(REPO, logs_dir=tmp_path / "logs", embedder=FakeEmbedder())  # ty: ignore[invalid-argument-type]
    folder = tmp_path / "profiles"
    folder.mkdir()
    minimum = "min_similarity = 0.0\n" if method == "meaning" else ""
    documents = f'\n[documents]\nsearch_method = "{method}"\n{minimum}'  # stated, both
    records = {}
    try:
        for backend in ("llama.cpp", "transformers"):
            path = Path(profile(folder, "smol-chat", backend, temperature="0.0"))
            path.write_text(path.read_text(encoding="utf-8") + documents, encoding="utf-8")
            session.new_conversation(str(path))
            session.set_search(["trail"])
            records[backend] = session.say("What marks the ridge?").record
            checks_ok(records[backend])
    finally:
        session.close()
    llama, hf = records["llama.cpp"], records["transformers"]
    assert '<part document="trail" title="Ridge">' in hf["prompt_text"]
    assert llama["prompt_ids"] == hf["prompt_ids"]
