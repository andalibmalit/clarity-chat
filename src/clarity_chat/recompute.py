"""The recompute check, for a notebook: what differs between an export's conditions and a
loaded model, and how far the logged probabilities move when recomputed under it.

    from clarity_chat.recompute import check
    print(check("logs/exports/ID.jsonl", model))

It gives facts and numbers, never a verdict: whether a difference matters depends on the
experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DIFFERENT = "different weights: this is a different model"
UNKNOWN = "unknown: pass repo= and revision="
HOW = (
    "Recomputed step by step with the cache, as generation ran: the prompt all at once, then "
    "one reply token at a time. A changed top choice means the token the model now ranks "
    "first at that step isn't the one it ranked first when the call was logged."
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
    `same` (True, False, or None if the export didn't record it) and, for weights, `says`.
    `calls`: per call, `measured` (`steps`, `largest_gap`, `top_changed`) or None with
    `why`."""

    groups: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []
        for group in self.groups:
            lines.append(f"Calls {', '.join(group['call_ids']) or '(none)'}:")
            for f in group["facts"]:
                if f["same"]:
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


def shown(value: Any) -> str:
    return "not recorded" if value is None else str(value)


def check(
    export: str | Path, model: Any, *, repo: str | None = None, revision: str | None = None
) -> Report:
    """Compare `export` (a `clarity export` file) with `model` (a loaded Transformers model).
    `repo` and `revision` name its weights when the model's config can't: a fine-tune saved
    locally, say."""
    rows = [json.loads(line) for line in Path(export).read_text(encoding="utf-8").splitlines()]
    header, calls = rows[0], {r["call_id"]: r for r in rows[1:]}
    report = Report()
    for group in header["conditions"]:
        found = facts(group["conditions"], model, repo, revision)
        report.groups.append({"call_ids": group["call_ids"], "facts": found})
        for call_id in group["call_ids"]:
            report.calls.append(measure(calls[call_id], model))
    return report


def facts(
    logged: dict[str, Any], model: Any, repo: str | None, revision: str | None
) -> list[dict[str, Any]]:
    config = model.config
    if not repo:
        repo, revision = config.name_or_path, getattr(config, "_commit_hash", None)
    weights = logged.get("weights") or {}
    here = f"{repo}@{revision}" if repo else None
    local = not repo or repo.startswith(("/", ".", "~")) or Path(repo).is_absolute()
    if local or not revision:
        says = UNKNOWN
    elif (repo, revision) != (weights.get("repo"), weights.get("revision")):
        says = DIFFERENT
    else:
        says = None
    eos = model.generation_config.eos_token_id
    eos = sorted([eos] if isinstance(eos, int) else eos or [])
    pairs = [
        ("backend", logged.get("backend"), "transformers"),
        ("precision", logged.get("dtype"), str(model.dtype).removeprefix("torch.")),
        ("device", logged.get("device"), model.device.type),
        ("attention", logged.get("attn_implementation"), config._attn_implementation),
        ("torch", logged.get("torch"), installed("torch")),
        ("transformers", logged.get("transformers"), installed("transformers")),
        ("end tokens", logged.get("end_tokens"), eos),
    ]
    exported = f"{weights.get('repo')}@{weights.get('revision')}"
    found: list[dict[str, Any]] = [
        {"fact": "weights", "export": exported, "here": here, "same": says is None}
    ]
    if says:
        found[0]["says"] = says
    # None in the export means not recorded (an older log, a missing server record): unknown,
    # which isn't the same as different.
    found += [
        {"fact": f, "export": e, "here": h, "same": None if e is None else e == h}
        for f, e, h in pairs
    ]
    return found


def measure(call: dict[str, Any], model: Any) -> dict[str, Any]:
    """Each logged reply token's log-probability recomputed step by step with the cache, as
    generation ran. One pass over the whole sequence would differ even under the same
    conditions (measured on SmolLM2: a gap of 0.139, as large as a precision change)."""
    logprobs = call.get("logprobs")
    if not logprobs or "tokens" not in logprobs:
        why = (logprobs or {}).get("unavailable") or "the call has no logprobs"
        return {
            "call_id": call["call_id"],
            "measured": None,
            "why": f"{why}; the measured part needs logprobs on (/set logprobs 5)",
        }
    output, logged = call["output_ids"], logprobs["tokens"]
    if len(logged) != len(output):
        why = f"{len(logged)} logged probabilities for {len(output)} reply tokens"
        return {"call_id": call["call_id"], "measured": None, "why": why}
    import torch

    from clarity_chat.backends.transformers_hf import log_softmax

    device, gaps, changed = model.device, [], 0
    with torch.inference_mode():
        ids = torch.tensor([call["prompt_ids"]], device=device)
        out = model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
        length = len(call["prompt_ids"])
        for step, (token, entry) in enumerate(zip(output, logged, strict=True)):
            row = log_softmax(out.logits[:, -1, :])[0]
            gaps.append(abs(row[token].item() - entry["logprob"]))
            changed += int(torch.argmax(row)) != entry["top"][0][0]
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
