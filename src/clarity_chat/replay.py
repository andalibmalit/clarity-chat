"""Replay: re-send a logged call's token IDs and request, and compare.

"Identical" means the same output token IDs and stop type; if the logged IDs were
incomplete (a token ended mid-character), the text is compared instead, and the record
says so. A difference is a finding, not an error. The replay record lists every recorded
fact that differs from now, and flags `unexplained_difference` when the output differs
but nothing that could change it did (only the client's commit or rendering libraries,
which can't affect a replay: the token IDs are sent as logged).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError

# Differences that can't change a replay's output: it sends the logged IDs and request.
CANNOT_CHANGE_OUTPUT = {"client", "libraries"}
MAX_DIFFS = 50  # per fact; a /props diff can be long
# /props fields that can't affect output, so they aren't compared: media_marker is a random
# string per start that marks images in a text prompt (checked live; Clarity sends IDs);
# endpoint_slots and ui report --no-slots and --no-webui, which Clarity sets since v0.2.0.
PROPS_NOT_COMPARED = {"media_marker", "endpoint_slots", "ui"}


def find_call(path: Path, call_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The call record and the start record of the server that ran it (None if not logged)."""
    call = None
    servers: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise ConfigError(f"Log file {path} not found.")
    for record in read_records(path):
        if record.get("type") == "server_start":
            servers.setdefault(record["server_id"], record)
        elif record.get("type") == "call" and record.get("call_id") == call_id:
            call = record
    if call is None:
        raise ConfigError(f"{path} has no call {call_id}.")
    if call.get("error") or "output_ids" not in call:
        raise ConfigError(f"Call {call_id} has no reply to compare with ({call.get('error')}).")
    if call.get("stop_type") == "cancelled":
        raise ConfigError(f"Call {call_id} was cancelled, so its reply is incomplete.")
    server_id = (call.get("backend") or {}).get("server_id")
    return call, servers.get(server_id) if server_id else None


def deep_diff(old: Any, new: Any, path: str = "") -> list[dict[str, Any]]:
    """Paths where two JSON values differ, as [{path, logged, now}]."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in old:
                out.append({"path": sub, "logged": "(missing)", "now": new[key]})
            elif key not in new:
                out.append({"path": sub, "logged": old[key], "now": "(missing)"})
            else:
                out += deep_diff(old[key], new[key], sub)
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        out = []
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            out += deep_diff(a, b, f"{path}[{i}]")
        return out
    return [] if old == new else [{"path": path or "(value)", "logged": old, "now": new}]


def not_recorded(logged: dict[str, Any], now: dict[str, Any]) -> list[str]:
    """Facts the logged call didn't record (an older version of Clarity), so they can't be
    compared, and machine values known on one side only (`machine.macos.chip`: a sandbox
    hides the chip's name; a newer Clarity records a new key). Listed in the replay record,
    so the gap is visible."""
    names = [name for name in now if logged.get(name) is None and now[name] is not None]
    if logged.get("machine") is not None and now.get("machine") is not None:
        found = deep_diff(logged["machine"], now["machine"])
        names += [f"machine.{d['path']}" for d in found if one_sided(d)]
    return sorted(names)


def one_sided(diff: dict[str, Any]) -> bool:
    """A value known on one side only: unknown, not different."""
    return (diff["logged"] in (None, "(missing)")) != (diff["now"] in (None, "(missing)"))


def environment_diff(logged: dict[str, Any], now: dict[str, Any]) -> dict[str, list[dict]]:
    """Each fact that differs, with its differing paths. `logged` and `now` map fact names
    (build_info, flags, props, machine, env, gguf_sha256, client, libraries) to values.
    Facts the log didn't record are left out (see not_recorded)."""
    diffs = {}
    skipped = set(not_recorded(logged, now))
    for name in sorted((set(logged) | set(now)) - skipped):
        found = deep_diff(logged.get(name), now.get(name))
        if name == "machine":  # values known on one side only are not recorded, above
            found = [d for d in found if not one_sided(d)]
        if found:
            diffs[name] = found[:MAX_DIFFS] + (
                [{"path": "…", "logged": f"{len(found) - MAX_DIFFS} more", "now": ""}]
                if len(found) > MAX_DIFFS
                else []
            )
    return diffs


def facts_from_call(call: dict[str, Any], server: dict[str, Any] | None) -> dict[str, Any]:
    """The facts a replay compares, as the log recorded them. Each backend has its own:
    llama-server's build, flags and /props, or PyTorch's and Transformers' versions.

    `end_tokens` are the IDs a reply stops on. Each backend finds its own, and they can differ:
    llama.cpp also stops on tokens it recognizes by name, such as SmolLM2's <|endoftext|>
    and <reponame>."""
    # The call names its backend; a log missing the server record still gets the right facts,
    # and what it lacks is None (not recorded) rather than an empty value that differs.
    server = server or {}
    backend = (call.get("backend") or {}).get("backend") or server.get("backend")
    if backend == "transformers":
        eos = server.get("eos_token_id")
        return {
            "end_tokens": sorted(eos) if eos is not None else None,
            "torch": server.get("torch"),
            "transformers": server.get("transformers"),
            "device": server.get("device"),
            "dtype": server.get("dtype"),
            "attn_implementation": server.get("attn_implementation"),
            "env": server.get("env_torch_kept"),
            "deterministic_algorithms": server.get("deterministic_algorithms"),
            "weights": {k: call["model"].get(k) for k in ("repo", "revision")},
            "machine": server.get("machine"),
            "client": call.get("client"),
            "libraries": call.get("libraries"),
        }
    eog = (server.get("model_file") or {}).get("eog_tokens")
    return {
        "end_tokens": sorted(t["id"] for t in eog) if eog is not None else None,
        "build_info": server.get("build_info"),
        "llama_cpp": (
            {k: v for k, v in server["llama_cpp"].items() if k != "dir"}
            if server.get("llama_cpp") is not None
            else None
        ),
        "flags": server.get("flags"),
        "graph_splits": server.get("graph_splits"),
        "props": (
            {k: v for k, v in server["props"].items() if k not in PROPS_NOT_COMPARED}
            if server.get("props") is not None
            else None
        ),
        "env": server.get("env_llama_ggml_kept"),
        "machine": server.get("machine"),
        "gguf_sha256": call["model"].get("gguf_sha256"),
        "client": call.get("client"),
        "libraries": call.get("libraries"),
    }


def compare_output(call: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compare a replay's fields (output_ids, output_text, stop_type, flags) with the call's."""
    flags = [*call.get("flags", []), *new.get("flags", [])]
    by_text = "incomplete_output_ids" in flags
    same_stop = call["stop_type"] == new["stop_type"]
    if by_text:
        identical = same_stop and call["output_text"] == new["output_text"]
        compared = "text and stop type (the output IDs were incomplete)"
    else:
        identical = same_stop and call["output_ids"] == new["output_ids"]
        compared = "output token IDs and stop type"
    result: dict[str, Any] = {"identical": identical, "compared": compared}
    if not identical:
        old_ids, new_ids = call["output_ids"], new["output_ids"]
        result.update(
            stop_types={"logged": call["stop_type"], "now": new["stop_type"]},
            n_output_ids={"logged": len(old_ids), "now": len(new_ids)},
        )
        if by_text:
            texts = call["output_text"], new["output_text"]
            result["first_different_char"] = first_difference(*texts)
        else:
            result["first_different_token"] = first_difference(old_ids, new_ids)
    return result


def first_difference(old: Any, new: Any) -> int:
    """The first index where two sequences differ (the shorter length if one is a prefix)."""
    return next(
        (i for i, (a, b) in enumerate(zip(old, new, strict=False)) if a != b),
        min(len(old), len(new)),
    )


def unexplained(identical: bool, diffs: dict[str, Any]) -> bool:
    """The output differs, yet nothing that could change it differs."""
    return not identical and set(diffs) <= CANNOT_CHANGE_OUTPUT
