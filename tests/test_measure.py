"""Measurements behind design choices (opt-in: `uv run pytest -m measure -v -s`).

M1: does closing the stream stop llama-server's prompt processing, or does the next request
wait for it? A3 (cancel) needs the answer. M2: how much memory OLMo 3 takes at two context
sizes, with flash attention on and off. A4 (the memory check) needs it. M3: how fast each
backend chats with OLMo 3, to decide which one a steered model should run on. M4: how much
faster llama.cpp gets on a quantized copy of the same model. M5 (v0.4.1 A2): the embedder's
memory, speed and vectors beside OLMo 3 (needs bge-m3 fetched too; `-k embedder`).

Needs the built llama-server and OLMo 3 Instruct fetched (M3's Transformers half also needs
`uv sync --extra hf`). Measures and prints; asserts only that the servers started. Each test
prints the power source and Low Power Mode first: on battery or in Low Power Mode, speeds drop.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from clarity_chat.backends.llama_cpp import LlamaServer
from clarity_chat.models import check_weights_quick, lock_entry, source_weights
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.session import Session

REPO = Path(__file__).resolve().parents[1]
MODEL = "olmo-3-7b-instruct"
LONG_PROMPT = [1000 + i % 5000 for i in range(6000)]  # ordinary tokens, no special ones
SHORT_PROMPT = LONG_PROMPT[:16]

pytestmark = pytest.mark.measure


def _weights() -> Path | None:
    try:
        return check_weights_quick(REPO, lock_entry(REPO, load_models(REPO)[MODEL]))
    except ConfigError:
        return None


if (WEIGHTS := _weights()) is None:
    pytestmark = [pytest.mark.measure, pytest.mark.skip(reason=f"{MODEL} isn't fetched")]


def power_state() -> str:
    """The power source and Low Power Mode, from `pmset`; "unknown" off macOS or if hidden."""
    try:
        battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True)
        settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True)
    except OSError:
        return "power source unknown, Low Power Mode unknown"
    source = re.search(r"drawing from '([^']+)'", battery.stdout)
    low = re.search(r"^\s*lowpowermode\s+([01])\b", settings.stdout, re.MULTILINE)
    low_power = {"0": "off", "1": "on"}[low.group(1)] if low else "unknown"
    return f"power source {source.group(1) if source else 'unknown'}, Low Power Mode {low_power}"


@pytest.fixture(autouse=True)
def print_power_state():
    print(f"\n{power_state()}")


def server(tmp_path: Path, **flags) -> LlamaServer:
    config = load_server_config(REPO)
    config = dataclasses.replace(config, flags={**config.flags, **flags})
    assert WEIGHTS is not None
    started = LlamaServer(config, WEIGHTS, tmp_path)
    started.start()
    return started


def request(prompt: list[int], stream: bool) -> dict:
    # The ping interval Clarity's own requests use (llama_cpp.PROTOCOL_REQUEST).
    return {
        "prompt": prompt,
        "n_predict": 1,
        "stream": stream,
        "cache_prompt": False,
        "sse_ping_interval": 1,
    }


def short_call_s(s: LlamaServer) -> float:
    start = time.monotonic()
    r = httpx.post(
        f"{s.url}/completion",
        headers=s.headers,
        json=request(SHORT_PROMPT, False),
        timeout=600,
        trust_env=False,
    )
    r.raise_for_status()
    return round(time.monotonic() - start, 3)


def test_m1_cancel_during_prompt_processing(tmp_path):
    s = server(tmp_path)
    try:
        idle = short_call_s(s)

        # A full long call: when each stream line arrives (pings while the prompt is read?).
        lines: list[tuple[float, str]] = []
        start = time.monotonic()
        with httpx.stream(
            "POST",
            f"{s.url}/completion",
            headers=s.headers,
            json=request(LONG_PROMPT, True),
            timeout=600,
            trust_env=False,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.strip():
                    lines.append((round(time.monotonic() - start, 3), line[:60]))
        full = round(time.monotonic() - start, 3)

        # The same call, closed 0.5 s in, then a short call at once.
        start = time.monotonic()
        with httpx.stream(
            "POST",
            f"{s.url}/completion",
            headers=s.headers,
            json=request(LONG_PROMPT, True),
            timeout=600,
            trust_env=False,
        ) as r:
            r.raise_for_status()
            time.sleep(0.5)
        closed_after = round(time.monotonic() - start, 3)
        after_cancel = short_call_s(s)
    finally:
        s.stop()
    print(
        "\nM1 "
        + json.dumps(
            {
                "prompt_tokens": len(LONG_PROMPT),
                "short_call_idle_s": idle,
                "long_call_full_s": full,
                "stream_lines_seconds_and_start": lines[:12],
                "n_stream_lines": len(lines),
                "closed_after_s": closed_after,
                "short_call_after_cancel_s": after_cancel,
            },
            indent=1,
        )
    )


MEMORY_LINE = re.compile(
    r"(model buffer size|KV buffer size|compute buffer size|splits = "
    r"|recommendedMaxWorkingSetSize|MiB free|n_ctx|flash_attn|n_swa)",
)


@pytest.mark.parametrize("flash_attn", ["on", "off"])
@pytest.mark.parametrize("ctx_size", [8192, 32768])
def test_m2_memory(tmp_path, ctx_size, flash_attn):
    s = server(tmp_path, **{"ctx-size": ctx_size, "flash-attn": flash_attn})
    try:
        startup = s.output_text()
    finally:
        s.stop()
    lines = [line.rstrip() for line in startup.splitlines() if MEMORY_LINE.search(line)]
    print(f"\nM2 ctx-size={ctx_size} flash-attn={flash_attn}")
    print("\n".join(lines))


# About 1,500 tokens of made-up text: the prompt of a chat a few turns in.
LONG_MESSAGE = "Describe this harbour in one paragraph. " + (
    "The harbour was quiet, and the boats sat still on grey water. " * 110
)


def speed_profile(folder: Path, backend: str) -> str:
    """profiles/1a-stock.toml on `backend`, with a reply short enough to time."""
    text = (REPO / "profiles" / "1a-stock.toml").read_text(encoding="utf-8")
    text = text.replace('backend = "llama.cpp"', f'backend = "{backend}"')
    text = text.replace("max_tokens = 1024", "max_tokens = 128")
    path = folder / f"speed-{backend}.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def per_second(n: int, ms: float) -> str:
    return f"{n / (ms / 1000):,.1f}" if ms else "?"


@pytest.mark.parametrize("backend", ["llama.cpp", "transformers"])
def test_m3_backend_speed(tmp_path, backend):
    """Each turn re-reads the whole prompt on both backends (Clarity keeps no prompt cache),
    so prompt speed counts on every turn. Turn 1 includes any warm-up; turn 2 is the steadier
    figure. On Transformers, the prompt time runs to the first generated token."""
    memory = ""
    if backend == "transformers":
        torch = pytest.importorskip("torch")
        try:
            source_weights(REPO, lock_entry(REPO, load_models(REPO)[MODEL]))
        except ConfigError as e:
            pytest.skip(str(e))
    session = Session(REPO, logs_dir=tmp_path / "logs")
    try:
        start = time.monotonic()
        session.new_conversation(speed_profile(tmp_path, backend))
        ready = time.monotonic() - start
        records = [
            session.say(LONG_MESSAGE).record,
            session.say("Now say that in one sentence.").record,
        ]
        if backend == "transformers" and torch.backends.mps.is_available():
            mib = torch.mps.driver_allocated_memory() / 2**20
            memory = f"; PyTorch holds {mib:,.0f} MiB on the GPU"
    finally:
        session.close()
    print(f"\nM3 {backend}: model ready after {ready:.1f} s{memory}")
    for turn, record in enumerate(records, 1):
        t = record["timings"]
        print(
            f"  turn {turn}: prompt {t['prompt_n']} tokens at "
            f"{per_second(t['prompt_n'], t['prompt_ms'])} tokens/s; reply {t['predicted_n']} "
            f"tokens at {per_second(t['predicted_n'], t['predicted_ms'])} tokens/s"
        )


@pytest.mark.parametrize("quant", ["bf16", "Q8_0", "Q4_K_M"])
def test_m4_quantized_speed(tmp_path, quant):
    """M3 compared the backends on the same unquantized weights. This is llama.cpp alone,
    on the fetched bf16 file and on copies llama-quantize makes of it. Every reply is 128
    tokens (the end token is ignored), and the copies are deleted afterwards."""
    config = load_server_config(REPO)
    assert WEIGHTS is not None
    weights = WEIGHTS
    if quant != "bf16":
        tool = config.llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if not tool.exists():
            pytest.skip(
                "llama-quantize isn't built: cmake --build vendor/llama.cpp/build "
                "--config Release -j 8 --target llama-quantize"
            )
        weights = tmp_path / f"{MODEL}-{quant}.gguf"
        subprocess.run(
            [str(tool), str(WEIGHTS), str(weights), quant], check=True, capture_output=True
        )
    s = LlamaServer(config, weights, tmp_path)
    try:
        size = weights.stat().st_size / 2**30
        start = time.monotonic()
        s.start()
        ready = time.monotonic() - start
        timings = []
        for _ in range(2):
            body = {**request(LONG_PROMPT[:1600], False), "n_predict": 128, "ignore_eos": True}
            r = httpx.post(
                f"{s.url}/completion", headers=s.headers, json=body, timeout=600, trust_env=False
            )
            r.raise_for_status()
            timings.append(r.json()["timings"])
    finally:
        s.stop()
        if quant != "bf16":
            weights.unlink(missing_ok=True)
    print(f"\nM4 {quant}: file {size:.1f} GiB, model ready after {ready:.1f} s")
    for turn, t in enumerate(timings, 1):
        print(
            f"  call {turn}: prompt {t['prompt_n']} tokens at {t['prompt_per_second']:,.1f} "
            f"tokens/s; reply {t['predicted_n']} tokens at "
            f"{t['predicted_per_second']:,.1f} tokens/s"
        )


# --- M5 (v0.4.1 A2): the embedder beside OLMo 3 7B ------------------------------------------


def embedder_fetched() -> bool:
    try:
        check_weights_quick(REPO, lock_entry(REPO, load_models(REPO)["bge-m3"]))
        return True
    except (ConfigError, KeyError):
        return False


def gate_ids(embedder) -> list[list[int]]:
    """The -m embed gate's texts (short, long to ~7,800 tokens, queries), as IDs."""
    from tests.test_embed import LONG, QUERIES, SHORT, counter, grown, hub_tokenizer

    model = load_models(REPO)["bge-m3"]
    tokenizer = hub_tokenizer(model.tokenizer_repo, model.tokenizer_revision)
    texts = SHORT + [grown(u(), t, counter(tokenizer), s) for u, t, s in LONG] + QUERIES
    return [embedder.ids(text) for text in texts]


def hashes(embedder, ids: list[list[int]]) -> list[str]:
    import hashlib

    return [hashlib.sha256(embedder.vector(x)).hexdigest() for x in ids]


@pytest.mark.skipif(not embedder_fetched(), reason="bge-m3 isn't fetched")
def test_m5_embedder_beside_the_chat_model(tmp_path):
    """A2: with OLMo 3 7B's chat server at 8,192, the embedder's "MiB free" (does Metal's
    figure count the chat server? decision 14), its model and compute buffers and graph
    splits (decision 5: over 2.5 GB, or more than one split, stops for the owner), a 600-part
    build's time (under ~15 s cuts D3, 5.2), the embedder's log size after it (C8), and the
    gate's vectors beside the chat model against the same with none loaded (DS5: any
    difference is a Metal determinism failure, decision 3). Prints; asserts DS5 only."""
    from clarity_chat.documents import Embedder, library_add, load_document, vectors_for
    from tests.test_embed import hub_tokenizer

    def made() -> Embedder:
        return Embedder(REPO, load_server_config(REPO), tmp_path, hub_tokenizer)

    alone = made()
    try:
        ids = gate_ids(alone)
        alone.start()
        before = hashes(alone, ids)
    finally:
        alone.close()

    chat = server(tmp_path, **{"ctx-size": 8192})
    embedder = made()
    try:
        embedder.start()
        assert embedder.server is not None
        mine, theirs = embedder.server.record, chat.record
        for name, record in (("chat (OLMo 3 7B, 8,192)", theirs), ("embedder (bge-m3)", mine)):
            print(f"\n{name}: devices {record['placement']['devices']}")
            print(f"  memory {record['memory']}")
            print(f"  graph splits {record['graph_splits']}")
        chat_free = theirs["placement"]["devices"][0]["free_mib"]
        embedder_free = mine["placement"]["devices"][0]["free_mib"]
        chat_total = theirs["memory"][0]["total_mib"]
        counts = embedder_free < chat_free - chat_total / 2
        print(
            f"\nDecision 14: the chat server saw {chat_free:,} MiB free and uses "
            f"{chat_total:,.0f}; the embedder then saw {embedder_free:,} MiB free: the figure "
            + (
                "counts the running chat server (A4 not needed)."
                if counts
                else "is per process (A4 is needed)."
            )  # fmt: skip
        )
        own = mine["memory"][0]
        splits = (mine["graph_splits"] or {}).get("prompt")
        stop = own["model_mib"] + own["compute_mib"] > 2.5 * 1024 or splits != 1
        print(
            f"Decision 5: the embedder's weights {own['model_mib']:,.0f} MiB + compute "
            f"{own['compute_mib']:,.0f} MiB, total {own['total_mib']:,.0f} MiB, graph splits "
            f"{splits}: "
            + (
                "STOP for the owner (over 2.5 GB, or flash attention off the GPU)."
                if stop
                else "within the plan."
            )  # fmt: skip
        )

        beside = hashes(embedder, ids)
        changed = [i for i, (a, b) in enumerate(zip(before, beside, strict=True)) if a != b]

        root = tmp_path / "root"
        text = "".join(
            f"## Headword {i}\nA made-up entry, number {i}, about tides, bread and bicycles.\n\n"
            for i in range(600)
        )
        library_add(root, text.encode(), "big.md")
        doc = load_document(root, "big")
        started = time.monotonic()
        vectors_for(root, doc, embedder, embedder.engine["setup"])
        took = time.monotonic() - started
        size = embedder.server.log_path.stat().st_size
        print(
            f"\nA 600-part build ({len(doc.parts)} parts) took {took:.1f} s, "
            f"{len(doc.parts) / took:.0f} parts a second: "
            + ("under 15 s, so D3 is cut (5.2)." if took < 15 else "15 s or more: D3 stays.")
        )
        print(f"The embedder's log after it: {size:,} bytes ({embedder.server.log_path.name})")
    finally:
        embedder.close()
        chat.stop()
    assert not changed, (
        f"DS5: {len(changed)} of {len(ids)} gate texts gave other vectors beside the chat "
        f"model than with none loaded (texts {changed}): a Metal determinism failure."
    )
