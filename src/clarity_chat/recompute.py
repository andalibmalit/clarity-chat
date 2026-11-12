"""The recompute check, for a notebook: what differs between an export's conditions and a
loaded model, and how far the logged probabilities move when recomputed under it.

    from clarity_chat.recompute import check
    print(check("logs/exports/ID.jsonl", model))

It gives facts and numbers, never a verdict: whether a difference matters depends on the
experiment."""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DIFFERENT = "different weights: this is a different model"
UNKNOWN = "unknown: pass revision="
GGUF = "the export ran on a GGUF file, not on these weights"
TORCH_FACTS = (
    "precision",
    "device",
    "attention",
    "torch",
    "transformers",
    "environment",
    "deterministic algorithms",
)
HOW = (
    "Recomputed step by step with the cache, as generation ran: the prompt all at once, then "
    "one reply token at a time. A changed top choice means the token the model now ranks "
    "first at that step differs from the one it ranked first when the call was logged. The "
    "reply's own token was sampled, so it isn't always the top choice."
)


def installed(package: str) -> str | None:
    """A package's installed version, without importing it (torch is optional)."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


@dataclass
class Report:
    """`groups`: per conditions group, its call IDs and each fact: `fact`, `export`, `here`,
    `same` (True, False, or None: unknown, because the export didn't record it or one side
    couldn't read it), for weights `says`, and for the machine's unknown values `summary`.
    `calls`: per call, `measured` (`steps`, `largest_gap`, `top_changed`) or None with
    `why`."""

    groups: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []
        for group in self.groups:
            lines.append(f"Calls {', '.join(group['call_ids']) or '(none)'}:")
            for f in group["facts"]:
                if f.get("summary"):
                    lines.append(f"  {f['fact']}: {f['summary']}")
                elif f["same"]:
                    lines.append(f"  {f['fact']}: the same ({shown(f['here'])})")
                elif f["same"] is None and not f.get("says"):
                    lines.append(f"  {f['fact']}: not recorded in the export")
                else:
                    says = f"{f['says']}; " if f.get("says") else ""
                    lines.append(
                        f"  {f['fact']}: {says}{shown(f['export'])} in the export, "
                        f"{shown(f['here'])} here"
                    )
        for c in self.calls:
            m = c["measured"]
            if m is None:
                lines.append(f"{c['call_id']}: facts only; {c['why']}")
            else:
                lines.append(
                    f"{c['call_id']}: largest log-probability gap {m['largest_gap']:.6g} over "
                    f"{m['steps']} steps; top choice changed at {m['top_changed']} of "
                    f"{m['steps']} steps"
                )
        if any(c["measured"] for c in self.calls):
            lines.append(HOW)
        return "\n".join(lines)


def loaded_torch() -> Any:
    """torch, imported if installed (a notebook passing a model has it already), else None."""
    if sys.modules.get("torch") is None and installed("torch"):
        import torch  # noqa: F401
    return sys.modules.get("torch")


def torch_version() -> str | None:
    """The loaded module's version: Linux and Windows wheels add a local tag (+cpu, +cu128)
    that the logged torch.__version__ has and the package metadata doesn't."""
    return getattr(loaded_torch(), "__version__", None) or installed("torch")


def shown(value: Any) -> str:
    return "not recorded" if value is None else str(value)


def check(
    export: str | Path, model: Any, *, repo: str | None = None, revision: str | None = None
) -> Report:
    """Compare `export` (a `clarity export` file) with `model` (a loaded Transformers model).
    Transformers 5 doesn't keep the commit a model was loaded from, so pass `revision` (and
    `repo` when the model was loaded from a folder); without it the weights read "unknown"."""
    # Split on "\n" only: the export writes U+2028 and its kin raw, and splitlines splits there.
    lines = Path(export).read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line]
    header = rows[0] if rows else {}
    if header.get("type") != "export" or header.get("format_version") != 1:
        found = f"format {header.get('format_version')}" if header.get("type") == "export" else ""
        raise ValueError(
            f"{export} isn't a Clarity export (format 1){f': it is {found}' if found else ''}. "
            "Make one with /export or `clarity export ID`."
        )
    calls = {r["call_id"]: r for r in rows[1:]}
    report = Report()
    for group in header["conditions"]:
        logged = group["conditions"]
        found = facts(logged, model, repo, revision)
        report.groups.append({"call_ids": group["call_ids"], "facts": found})
        for call_id in group["call_ids"]:
            report.calls.append(measure(calls[call_id], model, logged.get("backend")))
    return report


def facts(
    logged: dict[str, Any], model: Any, repo: str | None, revision: str | None
) -> list[dict[str, Any]]:
    config = model.config
    repo = repo or config.name_or_path
    weights = logged.get("weights") or {}
    here = f"{repo}@{revision}" if revision else repo
    local = not repo or repo.startswith(("/", ".", "~")) or Path(repo).is_absolute()
    gguf = logged.get("backend") != "transformers" or "gguf_sha256" in weights
    if not local and repo != weights.get("repo"):
        says, same = DIFFERENT, False
    elif revision and not local and revision != weights.get("revision"):
        says, same = DIFFERENT, False
    elif gguf:
        says, same = GGUF, False
    elif local or not revision:
        says, same = UNKNOWN, None
    else:
        says, same = None, True
    eos = model.generation_config.eos_token_id
    eos = sorted([eos] if isinstance(eos, int) else eos or [])
    from clarity_chat.backends.transformers_hf import torch_env
    from clarity_chat.logs import machine

    torch = loaded_torch()
    deterministic = torch.are_deterministic_algorithms_enabled() if torch else None
    pairs = [
        ("backend", logged.get("backend"), "transformers"),
        ("precision", logged.get("dtype"), str(model.dtype).removeprefix("torch.")),
        ("device", logged.get("device"), model.device.type),
        ("attention", logged.get("attn_implementation"), config._attn_implementation),
        ("torch", logged.get("torch"), torch_version()),
        ("transformers", logged.get("transformers"), installed("transformers")),
        ("end tokens", logged.get("end_tokens"), eos),
        ("environment", logged.get("env"), torch_env()),
        ("deterministic algorithms", logged.get("deterministic_algorithms"), deterministic),
        # What `clarity_chat.steering.applied` has added to the model, if anything. An export
        # made before v0.2.2 has no steering: nothing could be steered then.
        (
            "steering",
            logged.get("steering", "none"),
            getattr(model, "clarity_steering", None) or "none",
        ),
    ]
    if logged.get("backend") != "transformers":  # llama.cpp has none of these
        pairs = [p for p in pairs if p[0] not in TORCH_FACTS]
    exported = f"{weights.get('repo')}@{weights.get('revision')}"
    found: list[dict[str, Any]] = [
        {"fact": "weights", "export": exported, "here": here, "same": same}
    ]
    if says:
        found[0]["says"] = says
    # None in the export means not recorded (an older log, a missing server record): unknown,
    # which isn't the same as different.
    found += [
        {"fact": f, "export": e, "here": h, "same": None if e is None else e == h}
        for f, e, h in pairs
    ]
    found.append(machine_fact(logged.get("machine"), machine()))
    return found


def machine_fact(logged: dict[str, Any] | None, here: dict[str, Any]) -> dict[str, Any]:
    """The machine, value by value (`macos.chip`), shown as only the values that differ: a
    whole record would bury them. A value one side couldn't read (None, as when a sandbox
    hides the chip's name) is unknown, not a difference."""
    if logged is None:
        return {"fact": "machine", "export": None, "here": here, "same": None}
    old, new = flat(logged), flat(here)
    keys = sorted({*old, *new})
    differ = [k for k in keys if None not in (old.get(k), new.get(k)) and old[k] != new[k]]
    unknown = [k for k in keys if (old.get(k) is None) != (new.get(k) is None)]
    shown = differ or unknown
    fact: dict[str, Any] = {
        "fact": "machine",
        "export": {k: old.get(k) for k in shown} if shown else logged,
        "here": {k: new.get(k) for k in shown} if shown else "this machine",
        "same": False if differ else None if unknown else True,
    }
    if unknown and not differ:  # one plain line: which value, and what the other side says
        fact["summary"] = "; ".join(
            f"{k} not known here ({old[k]} in the export)"
            if new.get(k) is None
            else f"{k} not recorded in the export ({new[k]} here)"
            for k in unknown
        )
    return fact


def flat(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"macos": {"chip": X}} -> {"macos.chip": X}."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def measure(call: dict[str, Any], model: Any, backend: str | None) -> dict[str, Any]:
    """Each logged reply token's log-probability recomputed step by step with the cache, as
    generation ran. In bf16, one pass over the whole sequence differs even under the same
    conditions (one 60-token SmolLM2 run on a CPU: a gap of 0.139, about as large as loading
    in float32), because its sums run in another order and round differently."""
    logprobs = call.get("logprobs")
    output = call["output_ids"]
    if backend != "transformers":
        why = "llama.cpp calls have no logged probabilities yet (they come in v0.4.0)"
    elif not logprobs:
        why = "the call has no logprobs; the measured part needs them on (/set logprobs 5)"
    elif "tokens" not in logprobs:
        why = f"its logprobs are unavailable: {logprobs.get('unavailable')}"
    elif len(logprobs["tokens"]) != len(output):
        why = f"{len(logprobs['tokens'])} logged probabilities for {len(output)} reply tokens"
    else:
        why = outside_vocabulary(call, model)
    if why or not logprobs:
        return {"call_id": call["call_id"], "measured": None, "why": why}
    logged = logprobs["tokens"]
    import torch

    from clarity_chat.backends.transformers_hf import log_softmax

    # generate runs the output layer on the last position only; so does the prompt pass here.
    keep = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )
    device, gaps, changed = model.device, [], 0
    with torch.inference_mode():
        ids = torch.tensor([call["prompt_ids"]], device=device)
        out = model(ids, attention_mask=torch.ones_like(ids), use_cache=True, **keep)
        length = len(call["prompt_ids"])
        for step, (token, entry) in enumerate(zip(output, logged, strict=True)):
            row = log_softmax(out.logits[:, -1, :])[0]
            gaps.append(abs(row[token].item() - entry["logprob"]))
            # Changed only if the logged top choice is no longer the highest: with a tie, topk
            # (when logged) and argmax can list different tokens first.
            changed += bool(row[entry["top"][0][0]] < row.max())
            if step == len(output) - 1:
                break
            length += 1
            out = model(
                torch.tensor([[token]], device=device),
                attention_mask=torch.ones(1, length, dtype=torch.long, device=device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    measured = {"steps": len(gaps), "largest_gap": max(gaps, default=0.0), "top_changed": changed}
    return {"call_id": call["call_id"], "measured": measured}


def outside_vocabulary(call: dict[str, Any], model: Any) -> str | None:
    """A reason if the call holds a token ID the model has no embedding for: another model."""
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    size = getattr(embeddings, "num_embeddings", None)
    largest = max([*call["prompt_ids"], *call["output_ids"]], default=-1)
    if size is not None and largest >= size:
        return f"token ID {largest} is outside this model's vocabulary of {size}"
    return None
