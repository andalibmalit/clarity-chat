"""The recompute check: what differs between an export's conditions and a loaded model, and
how far the logged probabilities move when recomputed (plan item C2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clarity_chat.backends.transformers_hf import torch_env
from clarity_chat.logs import machine
from clarity_chat.recompute import check, installed

TORCH, TRANSFORMERS = installed("torch"), installed("transformers")

VOCAB = 8
DIFFERENT = "different weights: this is a different model"
UNKNOWN = "unknown: pass revision="
GGUF = "the export ran on a GGUF file, not on these weights"


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
        "env": torch_env(),
        "deterministic_algorithms": False,
        "machine": machine(),
    }
    header = {
        "type": "export",
        "format_version": 1,
        "calls": len(calls),
        "conditions": [{"conditions": found, "call_ids": [c["call_id"] for c in calls]}],
    }
    lines = [json.dumps(row, ensure_ascii=False) for row in [header, *calls]]  # as export does
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stand_in(
    *,
    name: str = "org/tiny",
    dtype: str = "bfloat16",
    attn: str = "sdpa",
    eos: list[int] | None = None,
    device: str = "cpu",
):
    """A model with only what the facts read: config, dtype, device, generation_config. Like
    a real Transformers 5 model, its config names the repo but not the commit."""
    return SimpleNamespace(
        config=SimpleNamespace(name_or_path=name, _attn_implementation=attn),
        dtype=f"torch.{dtype}",
        device=SimpleNamespace(type=device),
        generation_config=SimpleNamespace(eos_token_id=[7] if eos is None else eos),
    )


PINS = {"revision": "a" * 40}


def test_the_module_imports_without_torch():
    code = "import sys; sys.modules['torch'] = None; import clarity_chat.recompute"
    subprocess.run([sys.executable, "-c", code], check=True)


def facts(report) -> dict[str, dict]:
    return {f["fact"]: f for f in report.groups[0]["facts"]}


def test_the_same_conditions_list_nothing_as_different(tmp_path):
    report = check(write_export(tmp_path / "e.jsonl", []), stand_in(), **PINS)
    assert [f["fact"] for f in report.groups[0]["facts"] if not f["same"]] == []


@pytest.mark.parametrize(
    ("model", "kwargs", "fact", "says"),
    [
        (stand_in(name="org/other"), {}, "weights", DIFFERENT),
        (stand_in(), {"revision": "b" * 40}, "weights", DIFFERENT),
        (stand_in(), {}, "weights", UNKNOWN),
        (stand_in(name="/home/me/finetune"), PINS, "weights", UNKNOWN),
        (stand_in(name="org/other"), {"repo": "org/tiny", **PINS}, "weights", None),
        (stand_in(), PINS, "weights", None),
        (stand_in(dtype="float32"), PINS, "precision", "bfloat16"),
        (stand_in(attn="eager"), PINS, "attention", "sdpa"),
        (stand_in(device="mps"), PINS, "device", "cpu"),
        (stand_in(eos=[2]), PINS, "end tokens", "[7]"),
    ],
)  # fmt: skip
def test_each_fact_that_differs_is_named(tmp_path, model, kwargs, fact, says):
    report = check(write_export(tmp_path / "e.jsonl", []), model, **kwargs)
    found = facts(report)[fact]
    if says is None:
        assert found["same"]
    else:
        assert not found["same"] and says in str(report)
        assert f"{fact}: the same" not in str(report)


def test_versions_and_backend_are_compared(tmp_path):
    llama = {
        "backend": "llama.cpp",
        "weights": {"repo": "org/tiny", "revision": "a" * 40, "gguf_sha256": "0" * 64},
        "llama_cpp": {"commit": "f" * 40},
        "build_info": "b1",
        "end_tokens": [7],
        "machine": None,
    }
    report = check(write_export(tmp_path / "e.jsonl", [], llama), stand_in(), **PINS)
    assert not facts(report)["backend"]["same"]
    other = write_export(tmp_path / "f.jsonl", [])
    header = json.loads(other.read_text(encoding="utf-8"))
    header["conditions"][0]["conditions"]["torch"] = "0.0.1"
    other.write_text(json.dumps(header) + "\n", encoding="utf-8")
    report = check(other, stand_in(), **PINS)
    assert not facts(report)["torch"]["same"] and facts(report)["transformers"]["same"]


def test_a_gguf_isnt_the_same_weights_and_torch_facts_dont_apply(tmp_path):
    """A llama.cpp call ran on a converted (maybe quantized) file, never on these weights."""
    llama = {
        "backend": "llama.cpp",
        "weights": {"repo": "org/tiny", "revision": "a" * 40, "gguf_sha256": "0" * 64},
        "end_tokens": [7],
        "machine": None,
    }
    report = check(write_export(tmp_path / "e.jsonl", [], llama), stand_in(), **PINS)
    text = str(report)
    assert facts(report)["weights"]["same"] is False and GGUF in text
    assert "weights: the same" not in text and "not recorded" not in text
    assert {"precision", "device", "attention", "torch", "transformers"}.isdisjoint(facts(report))


def test_the_torch_version_is_the_loaded_ones(tmp_path, monkeypatch):
    """Linux and Windows wheels carry a local tag (+cu128, +cpu) in torch.__version__, which
    the logged value has and the package metadata doesn't."""
    export = write_export(tmp_path / "e.jsonl", [])
    header = json.loads(export.read_text(encoding="utf-8"))
    header["conditions"][0]["conditions"]["torch"] = "9.9.9+cu128"
    export.write_text(json.dumps(header) + "\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="9.9.9+cu128"))
    assert facts(check(export, stand_in(), **PINS))["torch"]["same"]


def test_a_log_instead_of_an_export_is_a_clear_error(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"type": "conversation", "schema_version": 1}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="isn't a Clarity export"):
        check(log, stand_in())
    later = write_export(tmp_path / "e.jsonl", [])
    header = json.loads(later.read_text(encoding="utf-8"))
    later.write_text(json.dumps({**header, "format_version": 2}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="format 2"):
        check(later, stand_in())


def test_a_line_separator_inside_a_message_is_read(tmp_path):
    """The export writes U+2028 raw (ensure_ascii=False); str.splitlines would split there."""
    row = {**call("c1", None), "output_text": "one\u2028two\u0085three"}
    report = check(write_export(tmp_path / "e.jsonl", [row]), stand_in(), **PINS)
    assert report.calls[0]["call_id"] == "c1"


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
    why = report.calls[0]["why"]
    if logprobs:  # probabilities were on; the call was cancelled
        assert "cancelled" in why and "/set" not in why
    else:
        assert "/set logprobs" in why


def test_a_llama_cpp_call_isnt_told_to_turn_probabilities_on(tmp_path):
    llama = {"backend": "llama.cpp", "weights": {"repo": "org/tiny"}, "end_tokens": [7]}
    report = check(write_export(tmp_path / "e.jsonl", [call("c1", None)], llama), stand_in())
    assert "/set" not in report.calls[0]["why"] and "llama.cpp" in report.calls[0]["why"]


def test_the_report_gives_numbers_not_verdicts(tmp_path):
    text = str(
        check(
            write_export(tmp_path / "e.jsonl", [call("c1", None)]),
            stand_in(dtype="float32"),
            **PINS,
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
            self.kept: list[int | None] = []

        def get_input_embeddings(self):
            return SimpleNamespace(num_embeddings=len(table))

        def forward(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=True,
            logits_to_keep=None,
        ):
            self.kept.append(logits_to_keep)
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
    assert model.kept[0] == 1, "the prompt pass keeps only the last position, as generate does"


def test_bf16_scores_go_through_the_shared_reduction(tmp_path):
    """Logged values came from transformers_hf.log_softmax (CPU, float32); bf16 scores reduced
    any other way differ from them."""
    torch = pytest.importorskip("torch")
    table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(1)) * 8
    table = table.to(torch.bfloat16)
    model, _ = fixed_model(table)
    row = call("c1", logged_from(table, [1, 2], [3, 4, 5], torch))
    report = check(write_export(tmp_path / "e.jsonl", [row]), model)
    assert report.calls[0]["measured"]["largest_gap"] == 0.0


def test_a_tie_for_the_top_choice_isnt_a_change(tmp_path):
    """The logged top came from torch.topk, which may list either of two tied tokens first."""
    torch = pytest.importorskip("torch")
    table = torch.zeros(VOCAB, VOCAB)
    table[:, 5] = table[:, 6] = 4.0  # 5 and 6 tie after every token
    logged = logged_from(table, [1, 2], [6, 6], torch)
    for entry in logged["tokens"]:
        entry["top"] = [[6, entry["top"][0][1]]]  # topk put the later one first
    model, _ = fixed_model(table)
    report = check(
        write_export(tmp_path / "e.jsonl", [call("c1", logged) | {"output_ids": [6, 6]}]), model
    )
    assert report.calls[0]["measured"]["top_changed"] == 0


def test_ids_outside_the_models_vocabulary_get_the_facts_and_a_reason(tmp_path):
    torch = pytest.importorskip("torch")
    table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))
    logged = logged_from(table, [1, 2], [3, 4, 5], torch)
    row = {**call("c1", logged), "prompt_ids": [1, VOCAB + 100]}
    model, _ = fixed_model(table)
    report = check(write_export(tmp_path / "e.jsonl", [row]), model)
    assert report.calls[0]["measured"] is None and "vocabulary" in report.calls[0]["why"]
    assert report.groups[0]["facts"]


def test_each_conditions_group_gets_its_own_facts_and_calls(tmp_path):
    """A resumed conversation can span two server starts with different conditions."""
    torch = pytest.importorskip("torch")
    table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))
    model, _ = fixed_model(table)
    rows = [call(c, logged_from(table, [1, 2], [3, 4, 5], torch)) for c in ("c1", "c2")]
    path = write_export(tmp_path / "e.jsonl", rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    first = header["conditions"][0]["conditions"]
    header["conditions"] = [
        {"conditions": first, "call_ids": ["c1"]},
        {"conditions": {**first, "dtype": "float32"}, "call_ids": ["c2"]},
    ]
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")
    report = check(path, model)
    assert [g["call_ids"] for g in report.groups] == [["c1"], ["c2"]]
    precision = [{f["fact"]: f for f in g["facts"]}["precision"]["same"] for g in report.groups]
    assert precision == [True, False]
    assert [c["call_id"] for c in report.calls] == ["c1", "c2"]


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


def edited(path: Path, **conditions) -> Path:
    header = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
    header["conditions"][0]["conditions"].update(conditions)
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    return path


def test_the_environment_deterministic_setting_and_machine_are_compared(tmp_path, monkeypatch):
    """Conditions the export records that can change the numbers: PyTorch's environment
    variables (PYTORCH_ENABLE_MPS_FALLBACK), deterministic algorithms, and the machine."""
    torch = SimpleNamespace(__version__=TORCH, are_deterministic_algorithms_enabled=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    other = {**machine(), "hostname_hash": "another machine", "cpu_count": 999}
    export = edited(
        write_export(tmp_path / "e.jsonl", []),
        env={**torch_env(), "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        deterministic_algorithms=True,
        machine=other,
    )
    report = check(export, stand_in(), **PINS)
    found = facts(report)
    assert [found[f]["same"] for f in ("environment", "deterministic algorithms", "machine")] == [
        False,
        False,
        False,
    ]
    machine_line = next(line for line in str(report).splitlines() if "machine" in line)
    assert "hostname_hash" in machine_line and "cpu_count" in machine_line
    assert "os_release" not in machine_line  # only the keys that differ


def test_an_older_export_without_these_facts_says_not_recorded(tmp_path):
    export = edited(
        write_export(tmp_path / "e.jsonl", []),
        env=None,
        deterministic_algorithms=None,
        machine=None,
    )
    found = facts(check(export, stand_in(), **PINS))
    assert [found[f]["same"] for f in ("environment", "deterministic algorithms", "machine")] == [
        None,
        None,
        None,
    ]
