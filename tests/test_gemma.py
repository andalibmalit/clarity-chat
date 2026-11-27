"""Live tests of Gemma 4 E4B's official weights on the Transformers backend (v0.4.7 A3).

Run with `uv run pytest -m hf -v -s -k gemma` (needs `uv sync --extra hf`,
`uv run clarity fetch gemma-4-e4b-official`, and setup's `gemma-4-e4b` for the prompt-ID
test). Each test loads the bf16 weights on MPS, about 16 GB, one model at a time: check
that no other model is loaded and that about 16 GB is free before running them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from clarity_chat.models import lock_entry, source_weights
from clarity_chat.profiles import ConfigError, load_models
from clarity_chat.session import Session
from tests.test_hf import checks_ok, make_and_steer, profile, project

if TYPE_CHECKING:
    from clarity_chat.steering import Vector

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
    last blocks), and replays Identical. With -s it prints the vector's path, for G14.
    On the make's load, it also runs first_tokens, A3's fourth test, for G14 and X2."""
    root = project(tmp_path)

    def then(session: Session, model: Any, vector: Vector, plain: str) -> None:
        first_tokens(session, model, vector, plain, tmp_path, capsys)

    message = "Tell me about rivers."
    record = make_and_steer(root, tmp_path, capsys, Q4_0, message, then, **ON_OFFICIAL)
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


FIRST_TOKEN_MESSAGES = (  # made-up
    "Tell me about rivers.",
    "Suggest a name for a small boat.",
    "What should I pack for a day in the hills?",
)


def first_tokens(
    session: Session, model: Any, vector: Vector, plain: str, folder: Path, capsys
) -> None:
    """A3's fourth test, G14's reference (run A): for each made-up message, the prompt IDs
    the profile's chat sends and the first token's top 50 log-probabilities, unsteered and
    steered with the vector at block 21 and scale 1.0. Clarity's logprobs stop at 20, so
    the scores come from the loaded model, steered through steering.applied. Printed, and
    written as JSON to the test's folder; asserted only that steering moves them."""
    import json

    import torch

    from clarity_chat import steering
    from clarity_chat.render import build_prompt

    prepared = session.prepare(plain, steered=False)
    block = len(steering.model_blocks(model)) // 2

    def top(ids: list[int]) -> list[list[float]]:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
        scores = torch.log_softmax(out.logits[0, -1].float(), dim=-1).topk(50)
        return [[int(i), float(v)] for v, i in zip(scores.values, scores.indices, strict=True)]

    runs = []
    for message in FIRST_TOKEN_MESSAGES:
        turns = [("user", message)]
        ids = build_prompt(
            prepared.tokenizer, prepared.profile, turns, prepared.template_tokenizer
        ).ids
        unsteered = top(ids)
        with steering.applied(model, vector, 1.0, [block]) as count:
            steered = top(ids)
        assert count == {block: 1}, count
        runs.append(
            {"message": message, "prompt_ids": ids, "unsteered": unsteered, "steered": steered}
        )
    path = folder / "first-tokens.json"
    found = {"vector": str(vector.path), "sha256": vector.sha256, "block": block, "scale": 1.0}
    path.write_text(json.dumps({**found, "runs": runs}, indent=1), encoding="utf-8")
    with capsys.disabled():
        print(
            "\nThe first token's top 50, read through steering.applied (Clarity's logprobs "
            f"stop at 20): {path}"
        )
        for run in runs:
            print(
                f"{run['message']!r}: unsteered {run['unsteered'][:5]}, "
                f"steered {run['steered'][:5]}"
            )
    for run in runs:
        assert run["steered"] != run["unsteered"], run["message"]


READINGS = ("the same way", "the other way", "can't tell")


def reading(a_shift: float | None, c_shift: float | None, unsteered: float | None) -> str:
    """G14's reading of one token (S6): C's shift against A's, beside A to C's unsteered
    difference. A token missing from any of the four top 50s reads "can't tell"."""
    if a_shift is None or c_shift is None or unsteered is None or abs(c_shift) <= unsteered:
        return "can't tell"
    return "the same way" if (c_shift > 0) == (a_shift > 0) else "the other way"


def test_gemma_official_weights_on_transformers_then_the_q4_0_file_on_llama_cpp(tmp_path, capsys):
    """X2 (v0.4.7). One vector, made on the official weights, at block 21 and scale 1.0.
    Run A is the official weights on Transformers (bf16, MPS); run C is the shipped Q4_0
    file on llama.cpp (Metal), with the same [steering] snippet. They differ in the
    backend, the weights and the quantisation at once, so nothing here says why C does or
    doesn't follow A. For three made-up messages, both runs get the same prompt IDs and give
    the first token's top 50, unsteered and steered; A's come from the loaded model through
    steering.applied (Clarity's logprobs stop at 20), C's from llama-server's /completion.
    Asserted: steering moves each run's numbers; MPS gives its memory back before C starts;
    every check passes; each run's steered call replays Identical. Printed (with -s) and
    written to x2.json: A to C's unsteered difference, how steering moved A's five
    likeliest steered tokens in each run, G14's reading of each with the counts, each run's
    likeliest token, each start's memory and the seconds a `/set steering_scale` took on C.
    Loads one model at a time, about 16 GB."""
    import gc
    import json
    import time

    import httpx
    import torch

    from clarity_chat import steering
    from clarity_chat.backends.llama_cpp import LlamaCppBackend, memory_breakdown, metal_memory
    from clarity_chat.maker import make
    from clarity_chat.render import build_prompt
    from tests.test_live import steering_project
    from tests.test_maker_render import PAIRS

    root = steering_project(tmp_path)
    folder = tmp_path / "profiles"
    folder.mkdir()
    plain = official(folder)
    said: list[str] = []
    gb = 1e9

    def with_snippet(path: str, snippet: str) -> str:
        text = Path(path).read_text(encoding="utf-8")
        text = text.replace("[settings]", "[settings]\nsteering_scale = 1.0", 1)
        Path(path).write_text(text + "\n" + snippet, encoding="utf-8")
        return path

    def run_a(session: Session, vector: Vector, block: int) -> tuple[list[dict], list, Any]:
        """A's top 50s from the loaded model, the prompt IDs both runs get, and the
        tokenizer. Its locals hold the model, so they go when it returns."""
        model: Any = session.backend.model  # ty: ignore[unresolved-attribute]
        prepared = session.prepare(plain, steered=False)

        def top(ids: list[int]) -> dict[int, float]:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            scores = torch.log_softmax(out.logits[0, -1].float(), dim=-1).topk(50)
            return {int(i): float(v) for v, i in zip(scores.values, scores.indices, strict=True)}

        runs, prompts = [], []
        for message in FIRST_TOKEN_MESSAGES:
            ids = build_prompt(
                prepared.tokenizer,
                prepared.profile,
                [("user", message)],
                prepared.template_tokenizer,
            ).ids
            unsteered = top(ids)
            with steering.applied(model, vector, 1.0, [block]) as count:
                steered = top(ids)
            assert count == {block: 1}, count
            runs.append({"unsteered": unsteered, "steered": steered})
            prompts.append(ids)
        return runs, prompts, prepared.tokenizer

    # Run A: make the vector on the official weights, then steer with its snippet.
    before = torch.mps.current_allocated_memory()
    session = Session(root, logs_dir=tmp_path / "a")
    try:
        (root / "brief.toml").write_text(PAIRS, encoding="utf-8", newline="\n")
        assert make(session, root / "brief.toml", plain) == 0
        out = capsys.readouterr().out
        snippet = out[out.index("[steering]") :]
        name = f"brief-{Path(plain).stem}"
        made = json.loads((root / "local" / "vectors" / f"{name}.json").read_text("utf-8"))
        vector = steering.load(
            root / "local" / "vectors" / f"{name}.gguf", made["vector"]["sha256"]
        )
        steered_a = with_snippet(official(folder), snippet)
        session.new_conversation(steered_a)
        session.set_setting("max_tokens", "16")
        a_call = session.say(FIRST_TOKEN_MESSAGES[0]).record
        checks_ok(a_call)
        assert a_call["checks"]["steering"]["ok"]
        [block] = a_call["request"]["steering"]["layers"]
        assert block == 21
        a_server = session.backend.server_record
        assert a_server is not None
        loaded = torch.mps.current_allocated_memory()
        said.append(
            f"A's start: {loaded / gb:.2f} GB allocated on MPS, "
            f"{torch.mps.driver_allocated_memory() / gb:.2f} GB by the driver"
        )
        a_runs, prompts, tokenizer = run_a(session, vector, block)
        assert session.conversation is not None
        a_log = session.conversation.log.path
    finally:
        session.close()
    gc.collect()
    torch.mps.empty_cache()
    after = torch.mps.current_allocated_memory()
    said.append(
        f"MPS before A: {before / gb:.2f} GB; with A loaded: {loaded / gb:.2f} GB; "
        f"after A closed: {after / gb:.2f} GB"
    )
    assert after - before < (loaded - before) / 10, "MPS kept A's memory"

    # Run C: the Q4_0 file on llama.cpp, with the same snippet.
    steered_c = with_snippet(profile(folder, Q4_0, "llama.cpp", system=SYSTEM), snippet)
    session = Session(root, logs_dir=tmp_path / "c")

    def top_c() -> tuple[list[dict[int, float]], dict[int, str], str]:
        """C's top 50 for each prompt, before sampling, from the chat's own server."""
        backend = session.backends["llama.cpp", None]
        assert isinstance(backend, LlamaCppBackend) and backend.server is not None
        server = backend.server
        tops, texts = [], {}
        for ids in prompts:
            body = {
                "prompt": ids,
                "n_predict": 1,
                "n_probs": 50,
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
            tops.append({item["id"]: item["logprob"] for item in top})
            texts |= {item["id"]: item["token"] for item in top}
        metal = metal_memory(server.record["memory"])
        return tops, texts, memory_breakdown(metal) if metal else "no Metal reading"

    try:
        session.new_conversation(steered_c)
        session.set_setting("max_tokens", "16")
        c_call = session.say(FIRST_TOKEN_MESSAGES[0]).record
        checks_ok(c_call)
        assert c_call["checks"]["steering_start"]["ok"]
        assert c_call["request"]["steering"] == a_call["request"]["steering"]
        assert c_call["prompt_ids"] == prompts[0]
        c_steered, texts, memory = top_c()
        said.append(f"C's steered start: {memory}")
        started = time.monotonic()
        session.set_setting("steering_scale", "0")
        said.append(f"C's /set steering_scale 0 took {time.monotonic() - started:.1f} s")
        c_unsteered, more, memory = top_c()
        texts |= more
        said.append(f"C's unsteered start: {memory}")
        assert session.conversation is not None
        c_log = session.conversation.log.path
    finally:
        session.close()

    a_side = f"A: Transformers, {a_server['dtype']}, {a_server['device']}"
    c_side = f"C: llama.cpp, {Q4_0} (the shipped Q4_0 file), Metal"
    counts = dict.fromkeys(READINGS, 0)
    figures = []

    def shown(value: float | None) -> str:
        return "missing" if value is None else f"{value:+.3f}"

    for message, a, c_u, c_s in zip(
        FIRST_TOKEN_MESSAGES, a_runs, c_unsteered, c_steered, strict=True
    ):
        a_u, a_s = a["unsteered"], a["steered"]
        assert a_s != a_u, f"A: steering moved nothing for {message!r}"
        assert c_s != c_u, f"C: steering moved nothing for {message!r}"
        both = set(a_u) & set(c_u)
        gaps = [abs(a_u[t] - c_u[t]) for t in both] or [float("nan")]
        said.append(
            f"\n{message!r}. Unsteered, {a_side} against {c_side}: {len(both)} of 50 tokens in "
            f"both, largest difference {max(gaps):.3f}, mean {sum(gaps) / len(gaps):.3f}"
        )
        for run, u, s in (("A", a_u, a_s), ("C", c_u, c_s)):
            likeliest = [tokenizer.decode([max(d, key=d.__getitem__)]) for d in (u, s)]
            said.append(
                f"  {run}'s likeliest token: unsteered {likeliest[0]!r}, steered {likeliest[1]!r}"
            )
        for t in sorted(a_s, key=a_s.__getitem__, reverse=True)[:5]:
            a_shift = a_s[t] - a_u[t] if t in a_u else None
            c_shift = c_s[t] - c_u[t] if t in c_s and t in c_u else None
            gap = abs(a_u[t] - c_u[t]) if t in a_u and t in c_u else None
            read = reading(a_shift, c_shift, gap)
            counts[read] += 1
            figures.append(
                {
                    "message": message,
                    "token": t,
                    "a_shift": a_shift,
                    "c_shift": c_shift,
                    "unsteered_difference": gap,
                    "reading": read,
                }
            )
            said.append(
                f"  {texts.get(t, tokenizer.decode([t]))!r} ({t}): A's shift {shown(a_shift)}, "
                f"C's shift {shown(c_shift)}, unsteered difference {shown(gap)}: {read}"
            )
    said.append(
        "\nG14's readings over 15 tokens: " + ", ".join(f"{r} {n}" for r, n in counts.items())
    )
    path = tmp_path / "x2.json"
    record = {"a": a_runs, "c_unsteered": c_unsteered, "c_steered": c_steered, "tokens": figures}
    path.write_text(json.dumps({"prompt_ids": prompts, **record}, indent=1), encoding="utf-8")
    with capsys.disabled():
        print("\n" + "\n".join(said) + f"\nThe figures: {path}")

    for log, call in ((a_log, a_call), (c_log, c_call)):
        other = Session(root, logs_dir=tmp_path / "replay")
        try:
            replayed = other.replay(log, call["call_id"])
        finally:
            other.close()
        assert replayed["identical"], (call["backend"], replayed)
