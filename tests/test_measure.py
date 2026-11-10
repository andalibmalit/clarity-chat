"""Two measurements for v0.2.0's design (opt-in: `uv run pytest -m measure -v -s`).

M1: does closing the stream stop llama-server's prompt processing, or does the next request
wait for it? A3 (cancel) needs the answer. M2: how much memory OLMo 3 takes at two context
sizes, with flash attention on and off. A4 (the memory check) needs it.

Needs the built llama-server and OLMo 3 Instruct fetched. Measures and prints; asserts only
that the servers started.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path

import httpx
import pytest

from clarity_chat.backends.llama_cpp import LlamaServer
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config

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


def server(tmp_path: Path, **flags) -> LlamaServer:
    config = load_server_config(REPO)
    config = dataclasses.replace(config, flags={**config.flags, **flags})
    assert WEIGHTS is not None
    started = LlamaServer(config, WEIGHTS, tmp_path)
    started.start()
    return started


def request(prompt: list[int], stream: bool) -> dict:
    return {"prompt": prompt, "n_predict": 1, "stream": stream, "cache_prompt": False}


def short_call_s(url: str) -> float:
    start = time.monotonic()
    r = httpx.post(
        f"{url}/completion", json=request(SHORT_PROMPT, False), timeout=600, trust_env=False
    )
    r.raise_for_status()
    return round(time.monotonic() - start, 3)


def test_m1_cancel_during_prompt_processing(tmp_path):
    s = server(tmp_path)
    try:
        idle = short_call_s(s.url)

        # A full long call: when each stream line arrives (pings while the prompt is read?).
        lines: list[tuple[float, str]] = []
        start = time.monotonic()
        with httpx.stream(
            "POST",
            f"{s.url}/completion",
            json=request(LONG_PROMPT, True),
            timeout=600,
            trust_env=False,
        ) as r:
            for line in r.iter_lines():
                if line.strip():
                    lines.append((round(time.monotonic() - start, 3), line[:60]))
        full = round(time.monotonic() - start, 3)

        # The same call, closed 0.5 s in, then a short call at once.
        start = time.monotonic()
        with httpx.stream(
            "POST",
            f"{s.url}/completion",
            json=request(LONG_PROMPT, True),
            timeout=600,
            trust_env=False,
        ):
            time.sleep(0.5)
        closed_after = round(time.monotonic() - start, 3)
        after_cancel = short_call_s(s.url)
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
