"""The recompute check: what differs between an export's conditions and a loaded model, and
how far the logged probabilities move when recomputed (plan item C2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clarity_chat.recompute import check, installed

TORCH, TRANSFORMERS = installed("torch"), installed("transformers")

VOCAB = 8
DIFFERENT = "different weights: this is a different model"
UNKNOWN = "unknown: pass repo= and revision="


def write_export(path: Path, calls: list[dict], conditions: dict | None = None) -> Path:
    found = conditions or {
        "backend": "transformers",
        "weights": {"repo": "org/tiny", "revision": "a" * 40},
        "device": "cpu",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "torch": TORCH,
        "transformers": TRANSFORMERS,
        "end_tokens": [7],
        "machine": None,
    }
    header = {
        "type": "export",
        "format_version": 1,
        "calls": len(calls),
        "conditions": [{"conditions": found, "call_ids": [c["call_id"] for c in calls]}],
    }
    lines = [json.dumps(row) for row in [header, *calls]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stand_in(
    *,
    name: str = "org/tiny",
    revision: str | None = "a" * 40,
    dtype: str = "bfloat16",
    attn: str = "sdpa",
    eos: list[int] | None = None,
):
    """A model with only what the facts read: config, dtype, device, generation_config."""
    return SimpleNamespace(
        config=SimpleNamespace(name_or_path=name, _commit_hash=revision, _attn_implementation=attn),
        dtype=f"torch.{dtype}",
        device=SimpleNamespace(type="cpu"),
        generation_config=SimpleNamespace(eos_token_id=[7] if eos is None else eos),
    )


def test_the_module_imports_without_torch():
    code = "import sys; sys.modules['torch'] = None; import clarity_chat.recompute"
    subprocess.run([sys.executable, "-c", code], check=True)


def facts(report) -> dict[str, dict]:
    return {f["fact"]: f for f in report.groups[0]["facts"]}


def test_the_same_conditions_list_nothing_as_different(tmp_path):
    report = check(write_export(tmp_path / "e.jsonl", []), stand_in())
    assert [f["fact"] for f in report.groups[0]["facts"] if not f["same"]] == []


@pytest.mark.parametrize(
    ("model", "kwargs", "fact", "says"),
    [
        (stand_in(name="org/other"), {}, "weights", DIFFERENT),
        (stand_in(revision="b" * 40), {}, "weights", DIFFERENT),
        (stand_in(name="/home/me/finetune", revision=None), {}, "weights", UNKNOWN),
        (stand_in(name="org/other"), {"repo": "org/tiny", "revision": "a" * 40}, "weights", None),
        (stand_in(dtype="float32"), {}, "precision", "bfloat16"),
        (stand_in(attn="eager"), {}, "attention", "sdpa"),
        (stand_in(eos=[2]), {}, "end tokens", "[7]"),
    ],
)  # fmt: skip
def test_each_fact_that_differs_is_named(tmp_path, model, kwargs, fact, says):
    report = check(write_export(tmp_path / "e.jsonl", []), model, **kwargs)
    found = facts(report)[fact]
    if says is None:
        assert found["same"]
    else:
        assert not found["same"] and says in str(report)


def test_versions_and_backend_are_compared(tmp_path):
    llama = {
        "backend": "llama.cpp",
        "weights": {"repo": "org/tiny", "revision": "a" * 40, "gguf_sha256": "0" * 64},
        "llama_cpp": {"commit": "f" * 40},
        "build_info": "b1",
        "end_tokens": [7],
        "machine": None,
    }
    report = check(write_export(tmp_path / "e.jsonl", [], llama), stand_in())
    assert not facts(report)["backend"]["same"]
    other = write_export(tmp_path / "f.jsonl", [])
    header = json.loads(other.read_text(encoding="utf-8"))
    header["conditions"][0]["conditions"]["torch"] = "0.0.1"
    other.write_text(json.dumps(header) + "\n", encoding="utf-8")
    report = check(other, stand_in())
    assert not facts(report)["torch"]["same"] and facts(report)["transformers"]["same"]


def call(call_id: str, logprobs: dict | None) -> dict:
    row = {"call_id": call_id, "prompt_ids": [1, 2], "output_ids": [3, 4, 5]}
    return {**row, **({"logprobs": logprobs} if logprobs is not None else {})}


@pytest.mark.parametrize("logprobs", [None, {"unavailable": "cancelled"}])
def test_a_call_without_probabilities_gets_the_facts_and_a_reason(tmp_path, logprobs):
    report = check(
        write_export(tmp_path / "e.jsonl", [call("c1", logprobs)]),
        stand_in(),
    )
    assert report.calls == [{"call_id": "c1", "measured": None, "why": report.calls[0]["why"]}]
    assert "logprobs" in report.calls[0]["why"]


def test_the_report_gives_numbers_not_verdicts(tmp_path):
    text = str(
        check(
            write_export(tmp_path / "e.jsonl", [call("c1", None)]),
            stand_in(dtype="float32"),
        )
    ).lower()
    for word in ("safe", "ok", "pass", "fail", "acceptable", "negligible", "threshold"):
        assert f" {word}" not in text, word


# --- the measured part: needs torch ------------------------------------------------------


def fixed_model(table):
    """A stand-in causal LM whose next-token scores depend only on the last token, with a
    cache it checks is passed back: so a recompute that skips the cache fails."""
    torch = pytest.importorskip("torch")

    class Fixed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = stand_in().config
            self.generation_config = stand_in().generation_config
            self.dtype = torch.bfloat16
            self.device = torch.device("cpu")
            self.seen: list[int] = []

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
            ids = input_ids[0].tolist()
            if past_key_values is not None:
                assert past_key_values == self.seen, "the cache wasn't passed back"
            self.seen = [*(past_key_values or []), *ids]
            logits = torch.stack([table[i] for i in ids])[None]
            return SimpleNamespace(logits=logits, past_key_values=list(self.seen))

    return Fixed(), torch


def logged_from(table, prompt, output, torch):
    from clarity_chat.backends.transformers_hf import log_softmax

    tokens = []
    for previous, token in zip([prompt[-1], *output[:-1]], output, strict=True):
        row = log_softmax(table[previous][None])[0]
        top = int(torch.argmax(row))
        tokens.append({"id": token, "logprob": row[token].item(), "top": [[top, row[top].item()]]})
    return {"view": "raw logits, before any sampler", "n": 1, "tokens": tokens}


def test_recomputing_under_the_same_scores_gives_a_gap_of_0_and_no_changes(tmp_path):
    torch = pytest.importorskip("torch")
    table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))
    model, _ = fixed_model(table)
    row = call("c1", logged_from(table, [1, 2], [3, 4, 5], torch))
    report = check(write_export(tmp_path / "e.jsonl", [row]), model)
    measured = report.calls[0]["measured"]
    assert measured == {"steps": 3, "largest_gap": 0.0, "top_changed": 0}


def test_one_changed_score_gives_the_exact_gap_and_change_count(tmp_path):
    torch = pytest.importorskip("torch")
    from clarity_chat.backends.transformers_hf import log_softmax

    table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))
    logged = logged_from(table, [1, 2], [3, 4, 5], torch)
    changed = table.clone()
    changed[3, 0] = 50.0  # after token 3 (step 2), token 0 now dominates
    model, _ = fixed_model(changed)
    report = check(
        write_export(tmp_path / "e.jsonl", [call("c1", logged)]),
        model,
    )
    expected = abs(log_softmax(changed[3][None])[0, 4].item() - logged["tokens"][1]["logprob"])
    assert report.calls[0]["measured"] == {"steps": 3, "largest_gap": expected, "top_changed": 1}
