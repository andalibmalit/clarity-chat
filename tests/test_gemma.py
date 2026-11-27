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
        replayed = other.replay(log, second["call_id"])  # the call in the last log
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


X2_MESSAGES = (  # made-up, of varied shape: some first words are near certain, some open
    *FIRST_TOKEN_MESSAGES,
    "Is a tomato a fruit?",
    "Write one sentence about the moon.",
    "List three colours.",
    "Why do leaves change colour in autumn?",
    "Finish this line: The old lighthouse",
    "How many legs does a spider have?",
    "Give me a word that rhymes with cat.",
    "Describe a quiet morning in a small town.",
    "What is two plus two?",
)
SCALES = (1.0, -1.0, 0.5, 2.0)  # the vector's strengths; 0.5 and 2.0 are printed only
RANDOMS = [f"random {k}" for k in range(19)]  # above all 19 by chance: 1 in 20 at most
SEED = 47  # the random directions', fixed before the run
READINGS = ("the same way", "the other way", "can't tell")


def spearman(x: Any, y: Any) -> float:
    """The rank correlation of two arrays: the correlation of their ranks, a tie at its
    average rank. numpy only: the test environment has no scipy."""
    import numpy as np

    def ranked(values: Any) -> Any:
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        return (ends - (counts - 1) / 2)[inverse]

    return float(np.corrcoef(ranked(x), ranked(y))[0, 1])


def likeliest(values: Any, n: int) -> set[int]:
    import numpy as np

    return set(np.argpartition(values, -n)[-n:].tolist())


def score(a_u: Any, a_s: Any, u: Any, s: Any) -> float:
    """X2's score: the rank correlation of A's shift (a_s - a_u, steered minus unsteered)
    with another shift (s - u), over A's 50 likeliest unsteered and 50 likeliest steered
    tokens. The set comes from A's two runs alone."""
    over = sorted(likeliest(a_u, 50) | likeliest(a_s, 50))
    return spearman((a_s - a_u)[over], (s - u)[over])


def reading(value: float, controls: list[float], replaced: int) -> str:
    """One prompt's reading (Q-if-2's closer look, fixed before the run): "the same way" at
    a score of 0.5 or more and above all 19 random directions' scores; "the other way" at
    -0.5 or less; otherwise "can't tell", as is a prompt where the vector replaces fewer
    than 5 of A's top 50."""
    if replaced >= 5 and value >= 0.5 and value > max(controls):
        return READINGS[0]
    return READINGS[1] if replaced >= 5 and value <= -0.5 else READINGS[2]


def negated_check(
    forward: float, crossed: float, back: float, back_crossed: float, alike: float
) -> str:
    """Each a score averaged over the prompts: C's forward run against A's forward shift and
    against A's negated one; C's negated run against A's negated shift and against A's
    forward one; A's negated shift against A's forward one. Void when A's two shifts
    correlate at 0.5 or more; passes when C's forward run scores at least 0.2 higher against
    A's forward shift than against A's negated one, and C's negated run the reverse."""
    if alike >= 0.5:
        return "void"
    return "passes" if forward - crossed >= 0.2 and back - back_crossed >= 0.2 else "fails"


def verdict(average: float, random_averages: list[float], readings: list[str], negated: str) -> str:
    """X2's verdict over the prompts (the closer look): "follows", "doesn't follow" or
    "still can't tell"."""
    same, other = readings.count(READINGS[0]), readings.count(READINGS[1])
    above = average > max(random_averages)
    if above and same >= 10 and not other and negated != "fails":
        return "follows"
    if not above or other or same <= 3 or negated == "fails":
        return "doesn't follow"
    return "still can't tell"


def test_gemma_official_weights_on_transformers_then_the_q4_0_file_on_llama_cpp(tmp_path, capsys):
    """X2 (v0.4.7), with Q-if-2's closer look. One vector, made on the official weights, at
    block 21. Run A is the official weights on Transformers (bf16, MPS); run C is the
    shipped Q4_0 file on llama.cpp (Metal), with the same [steering] snippet. They differ in
    the backend, the weights and the quantisation at once, so nothing here says why C does
    or doesn't follow A. For 12 made-up messages, both runs get the same prompt IDs and give
    the first token's log-probabilities over the whole vocabulary: A's from the loaded model
    through steering.applied (Clarity's logprobs stop at 20), C's from llama-server's
    /completion. The conditions: unsteered twice; the vector at 1.0, -1.0, 0.5 and 2.0; 19
    random directions at 1.0 with the norm of the vector's block-21 direction.
    Asserted: steering moves each run's numbers; MPS gives its memory back before C starts;
    every check passes; each run's steered call replays Identical. Printed (with -s) and
    written to x2.json, asserted not at all: each prompt's score against the random
    directions' scores, its reading, the top-10 count and the negated scores; the verdict;
    the negated check; how far each random direction moves A; the run-to-run floor; A to
    C's unsteered difference; each run's likeliest token; each start's memory, and the
    seconds a `/set steering_scale` and C's replies took. One model at a time, about 16 GB."""
    import gc
    import json
    import time

    import httpx
    import numpy as np
    import torch

    from clarity_chat import steering
    from clarity_chat.backends.llama_cpp import LlamaCppBackend, memory_breakdown, metal_memory
    from clarity_chat.maker import make
    from clarity_chat.render import build_prompt
    from tests.test_hf import write_vector
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

    def run_a(session: Session, conditions: dict, block: int) -> tuple[list[dict], list, Any]:
        """A's first-token log-probabilities over the whole vocabulary under each condition,
        the prompt IDs both runs get, and the tokenizer. Its locals hold the model, so they
        go when it returns."""
        model: Any = session.backend.model  # ty: ignore[unresolved-attribute]
        prepared = session.prepare(plain, steered=False)
        runs, prompts = [], []
        for message in X2_MESSAGES:
            ids = build_prompt(
                prepared.tokenizer,
                prepared.profile,
                [("user", message)],
                prepared.template_tokenizer,
            ).ids
            run = {}
            for name, (vector, scale) in conditions.items():
                with steering.applied(model, vector, scale, [block]) as count, torch.no_grad():
                    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
                assert count == {block: int(scale != 0)}, (name, count)
                run[name] = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
            runs.append(run)
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
        rng = np.random.default_rng(SEED)
        norm = np.linalg.norm(vector.directions[block])
        randoms = {}
        for random in RANDOMS:
            direction = rng.standard_normal(vector.width).astype(np.float32)
            direction *= np.float32(norm / np.linalg.norm(direction))
            file = f"local/vectors/{random.replace(' ', '-')}.gguf"
            sha256 = write_vector(root / file, {block: direction}, str(vector.model_hint))
            lines = f'[steering]\nfile = "{file}"\nsha256 = "{sha256}"\nlayers = [{block}]\n'
            randoms[random] = (steering.load(root / file, sha256), lines)
        conditions = {
            "unsteered": (vector, 0.0),
            "unsteered again": (vector, 0.0),
            **{f"vector {s}": (vector, s) for s in SCALES},
            **{random: (v, 1.0) for random, (v, _) in randoms.items()},
        }
        a_runs, prompts, tokenizer = run_a(session, conditions, block)
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

    # Run C: the Q4_0 file on llama.cpp, with the same snippet, then each random direction's.
    on_c = {"backend": "llama.cpp", "system": SYSTEM}
    plain_c = profile(folder, Q4_0, **on_c)
    steered_c = with_snippet(profile(folder, Q4_0, **on_c), snippet)
    random_c = {r: with_snippet(profile(folder, Q4_0, **on_c), s) for r, (_, s) in randoms.items()}
    vocabulary = len(a_runs[0]["unsteered"])
    c_runs: list[dict] = [{} for _ in prompts]
    seconds: list[float] = []
    session = Session(root, logs_dir=tmp_path / "c")

    def keep_c(condition: str) -> str:
        """C's first-token log-probabilities for each prompt, before sampling, from the
        chat's own server, over the whole vocabulary. (Asked for fewer, say 20,000 if a
        reply is too slow, a token missing from the reply sits at its bottom rank.)"""
        backend = session.backends["llama.cpp", None]
        assert isinstance(backend, LlamaCppBackend) and backend.server is not None
        server = backend.server
        for ids, run in zip(prompts, c_runs, strict=True):
            body = {
                "prompt": ids,
                "n_predict": 1,
                "n_probs": vocabulary,
                "post_sampling_probs": False,
                "cache_prompt": False,
                "stream": False,
            }
            started = time.monotonic()
            response = httpx.post(
                f"{server.url}/completion",
                headers=server.headers,
                json=body,
                timeout=300,
                trust_env=False,
            )
            response.raise_for_status()
            top = response.json()["completion_probabilities"][0]["top_logprobs"]
            seconds.append(time.monotonic() - started)
            got = np.array([item["logprob"] for item in top], dtype=np.float32)
            run[condition] = np.full(vocabulary, got.min(), dtype=np.float32)
            run[condition][[item["id"] for item in top]] = got
        metal = metal_memory(server.record["memory"])
        return memory_breakdown(metal) if metal else "no Metal reading"

    try:
        session.new_conversation(steered_c)
        session.set_setting("max_tokens", "16")
        c_call = session.say(FIRST_TOKEN_MESSAGES[0]).record
        checks_ok(c_call)
        assert c_call["checks"]["steering_start"]["ok"]
        assert c_call["request"]["steering"] == a_call["request"]["steering"]
        assert c_call["prompt_ids"] == prompts[0]
        assert session.conversation is not None
        c_log = session.conversation.log.path
        said.append(f"C's steered start: {keep_c('vector 1.0')}")
        for scale in SCALES[1:]:
            session.set_setting("steering_scale", str(scale))
            keep_c(f"vector {scale}")
        started = time.monotonic()
        session.set_setting("steering_scale", "0")
        said.append(f"C's /set steering_scale 0 took {time.monotonic() - started:.1f} s")
        said.append(f"C's unsteered start: {keep_c('unsteered')}")
        for random, path in random_c.items():
            session.new_conversation(path)
            keep_c(random)
        session.new_conversation(plain_c)
        keep_c("unsteered again")
    finally:
        session.close()
    said.append(
        f"C's {len(seconds)} replies, each the whole vocabulary ({vocabulary:,} tokens): "
        f"{sum(seconds) / len(seconds):.2f} s on average, {max(seconds):.2f} s at most"
    )

    a_side = f"A: Transformers, {a_server['dtype']}, {a_server['device']}"
    c_side = f"C: llama.cpp, {Q4_0} (the shipped Q4_0 file), Metal"
    forward, backward = "vector 1.0", "vector -1.0"
    rows, saved = [], []

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def against(a: dict, reference: str, runs: dict, condition: str) -> float:
        """Another run's shift under `condition` scored against A's under `reference`."""
        return score(a["unsteered"], a[reference], runs["unsteered"], runs[condition])

    for message, ids, a, c in zip(X2_MESSAGES, prompts, a_runs, c_runs, strict=True):
        a_u, c_u = a["unsteered"], c["unsteered"]
        assert (a[forward] != a_u).any(), f"A: steering moved nothing for {message!r}"
        assert (c[forward] != c_u).any(), f"C: steering moved nothing for {message!r}"

        top_50 = sorted(likeliest(a_u, 50))
        row = {
            "message": message,
            "replaced": 50 - len(likeliest(a_u, 50) & likeliest(a[forward], 50)),
            "score": against(a, forward, c, forward),
            "controls": [against(a, forward, c, r) for r in RANDOMS],
            "same_direction": [against(a, r, c, r) for r in RANDOMS],
            "a_moved": {
                r: [
                    50 - len(likeliest(a_u, 50) & likeliest(a[r], 50)),
                    float(np.abs(a[r] - a_u)[top_50].mean()),
                ]
                for r in [*RANDOMS, forward]
            },  # fmt: skip
            "scales": {str(s): against(a, f"vector {s}", c, f"vector {s}") for s in SCALES},
            "negated": [
                against(a, backward, c, forward),
                against(a, backward, c, backward),
                against(a, forward, c, backward),
                against(a, forward, a, backward),
            ],
            "top_10": len(likeliest(a[forward], 10) & likeliest(c[forward], 10)),
            "floor": [float(np.abs(r["unsteered again"] - r["unsteered"]).max()) for r in (a, c)],
        }
        row["reading"] = reading(row["score"], row["controls"], row["replaced"])
        rows.append(row)
        crossed, back, back_crossed, alike = row["negated"]
        both = likeliest(a_u, 50) & likeliest(c_u, 50)
        gaps = [abs(float(a_u[t] - c_u[t])) for t in both] or [float("nan")]
        said += [
            f"\n{message!r}: score {row['score']:+.2f}, the 19 random directions "
            f"{min(row['controls']):+.2f} to {max(row['controls']):+.2f}; the vector replaces "
            f"{row['replaced']} of A's top 50; {row['top_10']} of A's 10 likeliest steered "
            f"tokens are among C's 10: {row['reading']}",
            f"  Negated: C's forward run against A's negated shift {crossed:+.2f}; C's negated "
            f"run against A's negated {back:+.2f}, against A's forward {back_crossed:+.2f}; A's "
            f"negated against A's forward {alike:+.2f}. At 0.5: {row['scales']['0.5']:+.2f}; "
            f"at 2.0: {row['scales']['2.0']:+.2f}",
            f"  Unsteered, {a_side} against {c_side}: {len(both)} of 50 tokens in both, "
            f"largest difference {max(gaps):.3f}, mean {mean(gaps):.3f}",
        ]
        for run, u, s in (("A", a_u, a[forward]), ("C", c_u, c[forward])):
            likely = [tokenizer.decode([int(np.argmax(v))]) for v in (u, s)]
            said.append(
                f"  {run}'s likeliest token: unsteered {likely[0]!r}, steered {likely[1]!r}"
            )
        for t in np.argsort(a[forward])[::-1][:5].tolist():
            said.append(
                f"  {tokenizer.decode([t])!r} ({t}): A's shift {a[forward][t] - a_u[t]:+.3f}, "
                f"C's shift {c[forward][t] - c_u[t]:+.3f}"
            )
        tokens = sorted(set().union(*(likeliest(r[k], 50) for r in (a, c) for k in r)))
        each = {"a": {k: v[tokens].tolist() for k, v in a.items()}}
        each["c"] = {k: v[tokens].tolist() for k, v in c.items()}
        saved.append({**row, "prompt_ids": ids, "tokens": tokens, **each})

    readings = [row["reading"] for row in rows]
    average = mean([row["score"] for row in rows])
    random_averages = [mean([row["controls"][k] for row in rows]) for k in range(len(RANDOMS))]
    crossed, back, back_crossed, alike = (mean([r["negated"][k] for r in rows]) for k in range(4))
    negated = negated_check(average, crossed, back, back_crossed, alike)
    result = verdict(average, random_averages, readings, negated)
    said += [
        f"\nOver {len(rows)} prompts: " + ", ".join(f"{r} {readings.count(r)}" for r in READINGS),
        f"The vector's average score {average:+.3f}; the 19 random directions' averages "
        f"{min(random_averages):+.3f} to {max(random_averages):+.3f}, "
        f"{sum(r >= average for r in random_averages)} at or above it",
        f"The negated check: {negated}. Averaged: C's forward run against A's forward shift "
        f"{average:+.3f}, against A's negated {crossed:+.3f}; C's negated run against A's "
        f"negated {back:+.3f}, against A's forward {back_crossed:+.3f}; A's negated against "
        f"A's forward {alike:+.3f}",
        "The run-to-run floor, the largest difference between the two unsteered runs over "
        f"the whole vocabulary: A {max(r['floor'][0] for r in rows):.6f}, "
        f"C {max(r['floor'][1] for r in rows):.6f}",
        "How far each direction moves A, averaged over the prompts (of A's unsteered top 50: "
        "tokens replaced, mean absolute shift), and the average score of C under the same "
        "direction against A under it:",
    ]
    for r in [*RANDOMS, forward]:
        replaced = mean([row["a_moved"][r][0] for row in rows])
        moved = mean([row["a_moved"][r][1] for row in rows])
        same = (
            average
            if r == forward
            else mean([row["same_direction"][RANDOMS.index(r)] for row in rows])
        )
        said.append(
            f"  {'the vector' if r == forward else r}: {replaced:.1f}, {moved:.3f}; {same:+.3f}"
        )
    said.append(f"X2's verdict: {result}")
    path = tmp_path / "x2.json"
    record = {
        "vector": {"path": str(vector.path), "sha256": vector.sha256, "block": block},
        "seed": SEED,
        "n_probs": vocabulary,
        "verdict": result,
        "negated_check": negated,
        "average": average,
        "random_averages": random_averages,
        "prompts": saved,
    }
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    with capsys.disabled():
        print("\n" + "\n".join(said) + f"\nThe figures: {path}")

    for log, call in ((a_log, a_call), (c_log, c_call)):
        other = Session(root, logs_dir=tmp_path / "replay")
        try:
            replayed = other.replay(log, call["call_id"])
        finally:
            other.close()
        assert replayed["identical"], (call["backend"], replayed)
