"""The llama.cpp backend: the client starts its own llama-server and talks to /completion.

Facts below were checked against the llama.cpp source at the commit in server.toml
(tools/server/server-schema.cpp, server-task.cpp, server-context.cpp; common/arg.cpp).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from clarity_chat import steering as vectors
from clarity_chat.backends import BackendError, Result
from clarity_chat.logs import git_state, machine, new_id
from clarity_chat.profiles import (
    WAY_BY_WORDS,
    WORDS_WAY_OUT,
    ConfigError,
    ModelEntry,
    ServerConfig,
    fetch_hint,
)

NAME = "llama.cpp"

# Every sampling field llama-server reports in `generation_settings` (or reads per request)
# that profiles don't set, held at a fixed neutral value so no server default applies.
# Logged with origin "backend".
NEUTRAL_REQUEST: dict[str, Any] = {
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    "typical_p": 1.0,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 64,
    "dry_sequence_breakers": ["\n", ":", '"', "*"],  # must be non-empty; unused while DRY is off
    "mirostat": 0,  # above 0, mirostat replaces the whole samplers chain
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "adaptive_target": -1.0,
    "adaptive_decay": 0.9,
    "n_keep": 0,
    "n_discard": 0,
    "ignore_eos": False,
    "logit_bias": [],  # acts even though it is not in samplers
    "n_probs": 0,
    "min_keep": 0,
    "grammar": "",  # acts even though it is not in samplers
    "grammar_lazy": False,
    "post_sampling_probs": False,
    "backend_sampling": False,
    "lora": [],
    "n_indent": 0,
    "t_max_predict_ms": -1,
    "n_cache_reuse": 0,
    "n_cmpl": 1,
    "reasoning_budget_tokens": -1,
    "reasoning_control": False,
}

# Request fields that control the protocol, not the output.
PROTOCOL_REQUEST: dict[str, Any] = {
    "stream": True,
    "return_tokens": True,
    "cache_prompt": False,  # llama.cpp warns that prompt caching can change results
    "timings_per_token": False,
    "return_progress": False,
    "sse_ping_interval": 1,  # keeps reads short, so Ctrl-C is handled promptly everywhere
}

# `generation_settings` keys the client doesn't send, with the value they must have
# (None = log only).
EXPECTED_UNSENT: dict[str, Any] = {
    "max_tokens": "n_predict",  # an alias: must equal n_predict
    "stream": True,
    "generation_prompt": "",
    "grammar_triggers": [],
    "preserved_tokens": [],
    "speculative.types": "none",
    "reasoning_in_content": False,
    "chat_format": None,
    "reasoning_format": None,
}

# Settings that are floats on the server (compared at float32 precision).
FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    *(k for k, v in NEUTRAL_REQUEST.items() if isinstance(v, float)),
}


def to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def steering_argv(steering: vectors.Steering | None) -> list[str]:
    """llama-server's two control-vector flags (decision 4). The scale is written as the
    float32 Transformers multiplies by, in text std::stof reads back as that float32; the
    range runs from the first steered block to the last, both included."""
    if steering is None:
        return []
    return [
        "--control-vector-scaled",
        f"{steering.vector.path.absolute()}:{to_float32(steering.scale)!r}",
        "--control-vector-layer-range",
        str(min(steering.layers)),
        str(max(steering.layers)),
    ]


steering_key = vectors.start_key


def same_value(name: str, sent: Any, got: Any) -> bool:
    if name in FLOAT_FIELDS and isinstance(got, (int, float)) and isinstance(sent, (int, float)):
        return to_float32(float(sent)) == to_float32(float(got))
    return sent == got


def map_settings(settings: dict[str, Any], stop: list[str]) -> dict[str, Any]:
    """The client's setting names to llama-server's request fields."""
    return {
        "temperature": settings["temperature"],
        "top_k": settings["top_k"],
        "top_p": settings["top_p"],
        "min_p": settings["min_p"],
        "repeat_penalty": settings["repeat_penalty"],
        "repeat_last_n": settings["repeat_last_n"],
        "presence_penalty": settings["presence_penalty"],
        "frequency_penalty": settings["frequency_penalty"],
        "seed": settings["seed"],
        "n_predict": settings["max_tokens"],
        "samplers": list(settings["samplers"]),
        "stop": list(stop),
    }


def exited_with(proc: Any, wait: float = 1.0) -> int | None:
    """The server process's exit code, waiting up to `wait` seconds: a dropped connection
    can be seen before the process is reaped. None if it's still running."""
    if proc is None:
        return None
    deadline = time.monotonic() + wait
    while (code := proc.poll()) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return code


def stopped_during(code: int, partial: Result) -> BackendError:
    return BackendError(
        f"llama-server stopped during the call (exit code {code}). Start the model again to go on.",
        partial,
        server_stopped=True,
    )


def build_request(token_ids: list[int], settings: dict[str, Any], stop: list[str]) -> dict:
    if not isinstance(settings["seed"], int):
        raise BackendError(f"Seed must be a concrete integer, got {settings['seed']!r}.")
    return {
        "prompt": list(token_ids),
        **map_settings(settings, stop),
        **NEUTRAL_REQUEST,
        **PROTOCOL_REQUEST,
    }


def check_generation_settings(request: dict, gs: dict) -> dict[str, dict[str, Any]]:
    """Every value sent must come back unchanged; every reported key must be accounted for."""
    mismatched = {}
    for name, sent in request.items():
        if name == "prompt" or name not in gs:
            continue
        if not same_value(name, sent, gs[name]):
            mismatched[name] = {"sent": sent, "got": gs[name]}
    for name, expected in EXPECTED_UNSENT.items():
        if expected is None or name not in gs:
            continue
        if expected == "n_predict":
            expected = request["n_predict"]
        if gs[name] != expected:
            mismatched[name] = {"expected": expected, "got": gs[name]}
    unaccounted = sorted(set(gs) - set(request) - set(EXPECTED_UNSENT))
    return {
        "settings_echo": {"ok": not mismatched, "mismatched": mismatched},
        "unaccounted_settings": {"ok": not unaccounted, "keys": unaccounted},
    }


def sse_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """The event stream's lines, split on "\\n" only, a "\\r" before it dropped. llama-server
    writes U+2028, U+2029 and U+0085 raw inside its JSON, where httpx's iter_lines would
    split as str.splitlines does. Each line is decoded whole, so a chunk can end anywhere."""
    pending = b""
    for chunk in chunks:
        *lines, pending = (pending + chunk).split(b"\n")
        for line in lines:
            yield line.removesuffix(b"\r").decode("utf-8", errors="replace")
    if pending:
        yield pending.removesuffix(b"\r").decode("utf-8", errors="replace")


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """One line of llama-server's event stream: a JSON chunk, or None for anything else."""
    if not line.startswith("data: "):
        return None  # blank separator, or a ": ping" comment
    payload = line[len("data: ") :]
    if payload == "[DONE]":
        return None
    return json.loads(payload)


# --- the managed server ----------------------------------------------------------------


DEVICE_RE = re.compile(r"using device (\S+) \((.*?)\) \(.*?\) - (\d+) MiB free")
BUFFER_RE = re.compile(r"(\S+) model buffer size = +([\d.]+) MiB")


EOG_RE = re.compile(r"EOG token += (\d+) '(.*)'\r?$", re.MULTILINE)


def eog_tokens(startup_output: str) -> list[dict[str, Any]] | None:
    """The tokens that end a reply, as llama.cpp read them from the GGUF (its startup log).
    None if the log has no such line (a llama.cpp update may change it): unknown, not none."""
    found = [{"id": int(m[1]), "text": m[2]} for m in EOG_RE.finditer(startup_output)]
    return found or None


# llama.cpp's dump of the model file's keys: one line per key, after the log's timestamp and
# level when it has them (src/llama-model-loader.cpp; common/log.cpp). A value is cut to 40
# characters, an array's brackets and separators included (gguf_kv_to_str).
KEY_LINE_RE = re.compile(
    r"^(?:[\d.]+ I )?llama_model_loader: - kv +\d+: +(\S+) (\S+) += (.*?)\r?$", re.MULTILINE
)
SUPPRESS_KEY = "tokenizer.ggml.suppress_tokens"
# Every line naming that key, in any shape and at any level, so a line the pattern above
# can't read is refused rather than read as no ban.
NAMED_RE = re.compile(
    rf"^(?:[\d.]+ \S )?llama_model_loader: - kv +\d+: +{re.escape(SUPPRESS_KEY)}\s", re.MULTILINE
)

# The dump's announced size, and every key line in any shape: a dump cut short is refused,
# not read as no ban (v0.4.1 X2). A key's name can add lines; a value can't (llama.cpp
# writes its newlines as \n), so a file can't make a short dump look whole.
ANNOUNCED_RE = re.compile(
    r"^(?:[\d.]+ \S )?llama_model_loader: loaded meta data with (\d+) key-value pairs", re.M
)
DUMPED_RE = re.compile(r"^(?:[\d.]+ \S )?llama_model_loader: - kv +\d+:", re.MULTILINE)


def suppress_tokens(startup_output: str) -> list[int]:
    """The token IDs the model file bans on every reply (llama-server adds minus infinity to
    each one's logit, so no request can lift the ban), from llama-server's dump of the
    file's keys. [] when the dump has no such key. Anything else is refused: no key line
    read at all, two lines naming the key or one in another shape, another type, a list the
    log cut, or a count that doesn't match. Every line of the whole output is searched and
    counted, so text from the file can't hide the real key line (docs/design.md, "The model
    file's banned tokens are recorded")."""
    keys = KEY_LINE_RE.findall(startup_output)
    found = [(kind, value) for name, kind, value in keys if name == SUPPRESS_KEY]
    whole = ANNOUNCED_RE.findall(startup_output) == [str(len(DUMPED_RE.findall(startup_output)))]
    if whole and len(NAMED_RE.findall(startup_output)) == len(found):
        if keys and not found:
            return []
        if len(found) == 1:
            kind, value = found[0]
            count = re.fullmatch(r"arr\[i32,([0-9]+)\]", kind)
            # The length first, so the pattern only ever reads 40 characters.
            listed = len(value) <= 40 and re.fullmatch(r"\[(-?[0-9]+(?:, -?[0-9]+)*)?\]", value)
            if count and listed:
                ids = [int(i) for i in value[1:-1].split(", ")] if value != "[]" else []
                if count[1] == str(len(ids)):  # as text: a count of any length can't raise
                    return ids
    raise ConfigError(
        "llama-server's startup log doesn't give this model file's whole list of banned "
        f"tokens ({SUPPRESS_KEY}), so Clarity can't record what llama-server keeps out of "
        "every reply, and this version of Clarity can't run the file."
    )


# "graph: nodes = 1131, splits = 2, …", or "graph (pp bs=512, tg bs=1): … splits = 2 / 3, …"
# when prompt processing and generation differ (src/llama-context.cpp).
SPLITS_RE = re.compile(r"graph(?: \(.*?\))?: nodes = [^,]+, splits = (\d+)(?: / (\d+))?")


# llama-server caps the context at the model's training context, with only this warning
# (tools/server/server-context.cpp); its /props then reports the capped size.
TRAINING_CONTEXT_RE = re.compile(r"exceeds the training context of the model \((\d+)\)")


def graph_splits(startup_output: str) -> dict[str, int] | None:
    """How many pieces llama.cpp cut the compute graph into, for prompt processing and for
    generation. On Metal, 2 is normal with the whole model on the GPU (measured: M2); what
    matters is a change between a call and its replay."""
    found = list(SPLITS_RE.finditer(startup_output))
    if not found:
        return None
    m = found[-1]  # a steered start reserves the graph again with the vector added
    return {"prompt": int(m[1]), "reply": int(m[2] or m[1])}


# "graph: nodes = 939, …"; a loaded vector adds one node per block in its range.
NODES_RE = re.compile(r"graph(?: \(.*?\))?: nodes = (\d+)")
# llama.cpp's loader says "control vector …" or "direction tensor …" when it fails
# (common/common.cpp, src/llama-adapter.cpp); quoted without the time and level.
VECTOR_ERROR = ("control vector", "direction tensor")
LOG_PREFIX_RE = re.compile(r"^[\d.]+ \S ")


def check_vector_start(startup_output: str, steering: vectors.Steering) -> None:
    """Decision 6's checks once a steered server is up. The file is the one hashed just
    before the start (its inode, size and times are unchanged), and the log shows the vector added:
    llama.cpp reserves the graph again with one more node per block in the range. A failed
    load leaves a healthy server that answers unsteered, so this is the only sign."""
    vector = steering.vector
    try:
        st = vector.path.stat()
        unchanged = vectors.file_identity(st) == vector.identity
    except OSError:
        unchanged = False
    if not unchanged:
        raise ConfigError(
            f"{vector.path} changed while the model was starting, so Clarity stopped the model "
            "rather than answer with a vector it hadn't hashed."
        )
    nodes = [int(m[1]) for m in NODES_RE.finditer(startup_output)]
    first, last = min(steering.layers), max(steering.layers)
    if len(nodes) < 2 or nodes[-1] - nodes[0] != last - first + 1:
        said = [
            LOG_PREFIX_RE.sub("", line)
            for line in startup_output.splitlines()
            if any(words in line for words in VECTOR_ERROR)
        ]
        raise BackendError(
            "llama-server started without the vector: its start log doesn't show "
            f"{steering.file} added at blocks {first} to {last}. Clarity stopped it. "
            f"llama.cpp said: {'; '.join(said) or 'nothing'}."
        )


def last_reservation(startup_output: str) -> str:
    """The log from the last graph reservation's compute buffers on. A loaded control
    vector makes llama.cpp reserve the graph a second time (`sched_reserve`, which prints
    each compute buffer and then the graph line); the second one's buffers replace the
    first's (G13: the same sizes, so summing both would double them)."""
    graphs = list(SPLITS_RE.finditer(startup_output))
    return startup_output[graphs[-2].end() :] if len(graphs) >= 2 else startup_output


def device_placement(startup_output: str) -> dict[str, Any]:
    """Which devices llama-server used and where the model's weights went (its startup log)."""
    devices = [
        {"name": m[1], "description": m[2], "free_mib": int(m[3])}
        for m in DEVICE_RE.finditer(startup_output)
    ]
    buffers = [{"buffer": m[1], "mib": float(m[2])} for m in BUFFER_RE.finditer(startup_output)]
    return {"devices": devices, "model_buffers": buffers}


KV_RE = re.compile(r"(\S+) KV buffer size = +([\d.]+) MiB")
COMPUTE_RE = re.compile(r"(\S+) compute buffer size = +([\d.]+) MiB")


def memory_use(startup_output: str, placement: dict[str, Any]) -> list[dict[str, Any]]:
    """Per GPU: its limit, and what the model, the context (KV) cache and the compute buffers
    took on it, from llama.cpp's startup log. On Metal, the "MiB free" llama.cpp prints is
    recommendedMaxWorkingSetSize minus what this process had allocated
    (ggml-metal-device.m), so at start it is about the limit itself: other apps' memory
    isn't counted (measured: 25,558 of 25,559 MiB "free" with other apps open)."""
    model = [(b["buffer"], b["mib"]) for b in placement["model_buffers"]]
    kv = [(m[1], float(m[2])) for m in KV_RE.finditer(startup_output)]
    compute = [(m[1], float(m[2])) for m in COMPUTE_RE.finditer(last_reservation(startup_output))]

    def on(device: str, buffers: list[tuple[str, float]]) -> float:
        return round(sum(mib for name, mib in buffers if name.startswith(device)), 2)

    found = []
    for device in placement["devices"]:
        name = device["name"]
        use = {
            "model_mib": on(name, model),
            "kv_mib": on(name, kv),
            "compute_mib": on(name, compute),
        }
        found.append(
            {
                "device": name,
                "limit_mib": device["free_mib"],
                **use,
                "total_mib": round(sum(use.values()), 2),
            }
        )
    return found


# The servers this process runs that have started (v0.4.1 A4): the chat server, and meaning
# search's embedder. Each is removed once it has exited.
RUNNING: list[LlamaServer] = []


def metal_memory(memory: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The Metal device's measured memory, if the start logged one (decision 8)."""
    return next((m for m in memory if m["device"].startswith("MTL")), None)


def memory_breakdown(m: dict[str, Any]) -> str:
    return (
        f"{m['total_mib']:,.0f} MiB of GPU memory ({m['model_mib']:,.0f} for the model, "
        f"{m['kv_mib']:,.0f} for the context cache, {m['compute_mib']:,.0f} for computing)"
    )


def metal_limit(m: dict[str, Any]) -> str:
    return f"Metal's recommended maximum for this GPU, {m['limit_mib']:,} MiB."


def memory_sentence(ctx_size: int, m: dict[str, Any]) -> str:
    """What a size uses on Metal (`m`, metal_memory's entry), said after `/set ctx_size` and
    on /settings (state 20)."""
    return f"At {ctx_size:,} tokens, the model uses {memory_breakdown(m)}, out of {metal_limit(m)}"


def memory_problem(
    ctx_size: int,
    memory: list[dict[str, Any]],
    embedder: bool = False,
    beside: tuple[int, list[dict[str, Any]]] | None = None,
) -> str | None:
    """Metal only for now. Apple documents recommendedMaxWorkingSetSize as "an approximation
    of how much memory ... this GPU device can allocate without affecting its runtime
    performance". Other GPUs log the figures only (until the Linux release).

    Metal's limit counts only this process's memory, so a start beside the other server
    (`beside`: its ctx_size and memory) adds that one's total on the same GPU (v0.4.1 A4,
    decision 14). The embedder's 8,192 is fixed: its ways out are the chat model's."""
    for m in memory:
        if not m["device"].startswith("MTL"):
            continue
        other = next((o for o in beside[1] if o["device"] == m["device"]), None) if beside else None
        total = m["total_mib"] + (other["total_mib"] if other else 0)
        if total <= m["limit_mib"]:
            continue
        if not embedder and m["total_mib"] > m["limit_mib"]:  # the chat model alone
            return (
                f"At ctx_size {ctx_size}, the model needs {memory_breakdown(m)}, more than "
                f"{metal_limit(m)} "
                + (
                    "The model alone is over the limit, so no ctx_size fits: use a smaller "
                    "model or a smaller quantization."
                    if m["model_mib"] > m["limit_mib"]
                    else "Lower ctx_size: the context cache grows with it."
                )
            )
        limit = f"more than {metal_limit(m)}"
        if other is None or (embedder and m["total_mib"] > m["limit_mib"]):
            # the embedder alone, with or without a chat server beside it: its 8,192 is
            # fixed, and nothing the chat server could give up would make room (#1)
            return (
                f"Meaning search's embedder needs {m['total_mib']:,.0f} MiB of GPU memory, "
                f"{limit} {WAY_BY_WORDS[0].upper()}{WAY_BY_WORDS[1:]}"
            )
        mine, theirs = (m, other) if embedder else (other, m)
        chat_ctx = beside[0] if embedder and beside else ctx_size
        return (
            f"Meaning search's embedder needs {mine['total_mib']:,.0f} MiB of GPU memory and "
            f"the chat model {theirs['total_mib']:,.0f} MiB at ctx_size {chat_ctx}: together "
            f"{total:,.0f} MiB, {limit} Use a smaller chat model or a lower ctx_size. "
            f"{WORDS_WAY_OUT}"
        )
    return None


def placement_problem(flags: dict[str, Any], placement: dict[str, Any]) -> str | None:
    """With GPU layers requested, the weights must be on a named GPU, not quietly on the CPU."""
    if flags.get("n-gpu-layers") in (0, "0") or flags.get("device") == "none":
        return None
    gpu_devices = [d for d in placement["devices"] if d["name"] and d["description"]]
    # CUDA_Host and Vulkan_Host are pinned system memory, not the GPU's.
    gpu_buffers = [
        b
        for b in placement["model_buffers"]
        if not b["buffer"].startswith("CPU") and not b["buffer"].endswith("_Host")
    ]
    if not gpu_devices or not gpu_buffers:
        return (
            f"n-gpu-layers = {flags.get('n-gpu-layers')!r}, but llama-server put the model on "
            f"the CPU (devices {placement['devices']}, buffers {placement['model_buffers']}). "
            "Is the GPU visible to this process (a sandbox can hide it)?"
        )
    return None


def llama_config_files() -> list[Path]:
    """Config files llama.cpp would read before any flag (common/arg.cpp)."""
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("PROGRAMDATA", "")) / "llama.cpp" / "config.ini",
            Path(os.environ.get("APPDATA", "")) / "llama.cpp" / "config.ini",
        ]
    else:
        config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        candidates = [
            Path("/etc/llama.cpp/config.ini"),
            Path(config_home) / "llama.cpp" / "config.ini",
        ]
    return [p for p in candidates if p.is_file()]


# Variables llama-server reads as flags (common/arg.cpp `set_env`): all removed.
ENV_PREFIX_REMOVED = "LLAMA_ARG_"
ENV_REMOVED = {"LLAMA_API_KEY", "HF_TOKEN", "MTMD_BACKEND_DEVICE"}


# Names that look like they hold a secret (for example LLAMA_CLOUD_API_KEY, from other
# tools): their values are never logged.
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASS|AUTH|CREDENTIAL", re.IGNORECASE)
REDACTED = "(value not logged: the name looks like a secret)"

# At log-verbosity 4 llama-server prints the API key's last four characters
# (tools/server/server-http.cpp); Clarity replaces the line in its output and log.
API_KEY_LINE = re.compile(r"api_keys: \*\*\*\*\S*")
API_KEY_REDACTED = "api_keys: (redacted by Clarity)"


def server_env() -> tuple[dict[str, str], list[str], dict[str, str]]:
    """The environment for llama-server, the names removed, and the LLAMA_*/GGML_* variables
    kept (with values: some, such as GGML_METAL_BF16_DISABLE, change how it computes; values
    of names that look like secrets are replaced by REDACTED)."""
    env, removed = {}, []
    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIX_REMOVED) or key in ENV_REMOVED:
            removed.append(key)
        else:
            env[key] = value
    kept = {
        k: REDACTED if SECRET_NAME.search(k) else env[k]
        for k in sorted(env)
        if k.startswith(("LLAMA_", "GGML_"))
    }
    return env, sorted(removed), kept


def stale_build(build_info: str, pinned_commit: str) -> str | None:
    """The binary is what runs, so one built from another commit is an error, like a checkout
    at another commit. build_info ends in the short commit it was built from."""
    if pinned_commit.startswith(build_info.rsplit("-", 1)[-1] or "?"):
        return None
    return (
        f"llama-server reports build {build_info!r}, but server.toml pins "
        f"{pinned_commit[:12]}: rebuild llama-server at that commit (docs/getting-started.md)."
    )


def start_warnings(checkout: dict[str, Any], kept: dict[str, str]) -> list[str]:
    """Facts about this server that you can act on and that could change its output."""
    warnings = []
    if checkout.get("dirty"):
        warnings.append(
            "The llama.cpp checkout has uncommitted changes, so this server can't be rebuilt "
            "exactly from its commit."
        )
    if kept:
        warnings.append(
            f"llama-server inherits {list(kept)} from your environment. Some GGML_*/LLAMA_* "
            "variables change how it computes; the values are logged. Unset them unless you "
            "mean to keep them."
        )
    return warnings


def server_binary(llama_cpp_dir: Path) -> Path:
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    for candidate in (
        llama_cpp_dir / "build" / "bin" / exe,
        llama_cpp_dir / "build" / "bin" / "Release" / exe,
    ):
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"llama-server not found under {llama_cpp_dir / 'build'}. Build it first (see README)."
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    """One llama-server process, started and stopped by the client."""

    def __init__(
        self,
        config: ServerConfig,
        gguf: Path,
        logs_dir: Path,
        on_change: Callable[[dict[str, Any] | None], None] | None = None,
        steering: vectors.Steering | None = None,
    ) -> None:
        self.config = config
        self.steering = steering  # the vector the server is started with (llama.cpp's flags)
        self.on_change = on_change  # told the port, PID and key on start, None on stop
        # Not resolved: a Hugging Face cache symlink keeps the GGUF's file name for the server.
        self.gguf = gguf.absolute()
        self.server_id = new_id("server")
        self.log_path = logs_dir / f"{self.server_id}.log"  # the server's own output
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.output: list[str] = []
        self._output_lock = threading.Lock()
        self.record: dict[str, Any] = {}
        # A new key per start, so a web page in a browser can't use the server (only /health
        # is public). It goes in the child's environment, not argv, which is logged.
        self.api_key = secrets.token_urlsafe(32)
        self.embedding = bool(config.flags.get("embedding"))  # meaning search's embedder

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def output_text(self) -> str:
        with self._output_lock:
            return "".join(self.output)

    def _drain(self, stream: Any) -> None:
        # Private, like the chat's log: the server's output can hold prompt text.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        fd = os.open(self.log_path, flags, 0o600)
        with open(fd, "a", encoding="utf-8", newline="") as log:
            for raw in iter(stream.readline, b""):
                line = API_KEY_LINE.sub(API_KEY_REDACTED, raw.decode("utf-8", errors="replace"))
                with self._output_lock:
                    self.output.append(line)
                log.write(line)
                log.flush()

    def start(self) -> dict[str, Any]:
        found = llama_config_files()
        if found:
            raise ConfigError(
                f"llama.cpp config file(s) {[str(p) for p in found]} would apply settings "
                "before any flag. Remove or rename them."
            )
        checkout = git_state(self.config.llama_cpp_dir)
        if checkout.get("commit") is None:
            raise ConfigError(
                f"{self.config.llama_cpp_dir} has no commit git can read "
                f"({checkout.get('error')}), so it can't be checked against the "
                f"{self.config.llama_cpp_commit} server.toml pins. Check out that commit there "
                "and rebuild."
            )
        if checkout.get("commit") != self.config.llama_cpp_commit:
            raise ConfigError(
                f"{self.config.llama_cpp_dir} is at {checkout.get('commit')}, but server.toml "
                f"pins {self.config.llama_cpp_commit}. Check out that commit and rebuild."
            )
        binary = server_binary(self.config.llama_cpp_dir)
        env, removed, kept = server_env()
        env = {**env, "LLAMA_API_KEY": self.api_key}  # after server_env: not "inherited"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        attempts = []
        for _ in range(3):  # a free port can be taken between choosing it and binding it
            self.port = free_port()
            # Clarity's own flags come last: llama-server takes the last value of a flag.
            argv = [
                str(binary),
                "--model",
                str(self.gguf),
                *self.config.flag_argv(),
                *steering_argv(self.steering),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--no-slots",  # /slots would show prompts to anyone holding the key
                "--no-webui",  # the built-in web UI is public even with a key
            ]
            error = self._launch(argv, env)
            if error is None:
                break
            attempts.append({"port": self.port, "error": error})
            if "bind" not in error.lower() and "address" not in error.lower():
                raise BackendError(f"llama-server failed to start: {error}")
        else:
            raise BackendError(f"llama-server failed to start after retries: {attempts}")

        startup = self.output_text()
        if "using config file:" in startup:
            self.stop()
            raise ConfigError("llama-server loaded a config file (see its startup output).")
        if self.steering is not None:
            try:
                check_vector_start(startup, self.steering)
            except (ConfigError, BackendError):
                self.stop()
                raise
        placement = device_placement(startup)
        problem = placement_problem(self.config.flags, placement)
        memory = memory_use(startup, placement)
        # The other server (this one joins RUNNING once it has started): a chat server and an
        # embedder never have a second of their kind, since a chat model's start closes the
        # old one first. Counted only while its process runs.
        others = [s for s in RUNNING if s.proc is not None and s.proc.poll() is None]
        beside = (others[0].config.ctx_size, others[0].record["memory"]) if others else None
        problem = problem or memory_problem(self.config.ctx_size, memory, self.embedding, beside)
        if problem:
            self.stop()
            raise ConfigError(problem)
        try:
            banned = suppress_tokens(startup)
        except ConfigError:
            self.stop()
            raise
        try:
            props = self.get_props()
        except (httpx.HTTPError, ValueError) as e:
            self.stop()
            raise BackendError(f"llama-server's /props failed after startup: {e}") from e
        problems = self.check_props(props)
        if problems:
            self.stop()
            capped = TRAINING_CONTEXT_RE.search(startup)
            if capped:
                raise ConfigError(
                    f"ctx_size {self.config.ctx_size} is more than this model was trained on "
                    f"({capped[1]} tokens), and llama-server would quietly use {capped[1]}. "
                    f"Set ctx_size to {capped[1]} or less."
                )
            raise BackendError(
                "llama-server is not the server the client started: " + "; ".join(problems)
            )
        build_info = str(props.get("build_info", ""))
        stale = stale_build(build_info, self.config.llama_cpp_commit)
        if stale:
            self.stop()
            raise ConfigError(stale)
        warnings = start_warnings(checkout, kept)
        self.record = {
            "type": "server_start",
            "server_id": self.server_id,
            "backend": NAME,
            "argv": argv,
            "flags": self.config.flags,
            "steering": self.steering.to_json() if self.steering is not None else None,
            "port_attempts": attempts,
            "api_key": "random per start; value not logged",
            "env_removed": removed,
            "env_llama_ggml_kept": kept,
            "llama_cpp": {"dir": self.config.llama_cpp_dir.as_posix(), **checkout},
            "build_info": build_info,
            "placement": placement,
            "memory": memory,
            "model_file": {"eog_tokens": eog_tokens(startup), "suppress_tokens": banned},
            "graph_splits": graph_splits(startup),
            "props": props,
            "startup_output": startup,
            "warnings": warnings,
            "machine": machine(),
        }
        RUNNING.append(self)
        return self.record

    def _launch(self, argv: list[str], env: dict[str, str]) -> str | None:
        """Start the process and wait for /health. Returns an error message, or None."""
        with self._output_lock:
            self.output = []
        # A new process group, so the terminal's Ctrl-C reaches the client but not the server.
        if os.name == "nt":
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.proc = proc
        if self.on_change:  # recorded before the model loads: a crash then still finds it
            self.on_change(
                {
                    "port": self.port,
                    "pid": proc.pid,
                    "api_key": self.api_key,
                    "model_path": str(self.gguf),
                }
            )
        threading.Thread(target=self._drain, args=(proc.stdout,), daemon=True).start()
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._stopped()
                time.sleep(0.2)  # let the drain thread catch the last lines
                return f"exited with code {proc.returncode}:\n{self.output_text()[-3000:]}"
            try:
                r = httpx.get(f"{self.url}/health", timeout=2, trust_env=False, verify=False)
                if r.status_code == 200:
                    return None
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        self.stop()
        return f"no healthy response within {self.config.startup_timeout_s} s"

    def get_props(self) -> dict[str, Any]:
        r = httpx.get(
            f"{self.url}/props", headers=self.headers, timeout=10, trust_env=False, verify=False
        )
        r.raise_for_status()
        return r.json()

    def check_props(self, props: dict[str, Any]) -> list[str]:
        """Is this our server, with our model and context size, and still running?"""
        problems = []
        if self.proc is None or self.proc.poll() is not None:
            problems.append("the llama-server process the client started is not running")
        model_path = props.get("model_path")
        if model_path is None or Path(model_path).resolve() != self.gguf.resolve():
            problems.append(f"/props model_path is {model_path!r}, expected {str(self.gguf)!r}")
        n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
        if n_ctx != self.config.ctx_size:
            problems.append(f"/props n_ctx is {n_ctx!r}, expected {self.config.ctx_size}")
        return problems

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        self._stopped()  # only once it has exited: if stopping fails, the record stays

    def _stopped(self) -> None:
        if self in RUNNING:
            RUNNING.remove(self)
        if self.on_change:
            self.on_change(None)


class LlamaCppBackend:
    name = NAME

    def __init__(
        self,
        config: ServerConfig,
        logs_dir: Path,
        on_server: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.config = config
        self.logs_dir = logs_dir
        self.on_server = on_server
        self.server: LlamaServer | None = None
        self.model: ModelEntry | None = None
        self.weights_identity: tuple[int, int, int] | None = None  # of the running server's file

    @property
    def n_ctx(self) -> int:
        return (self.server.config if self.server else self.config).ctx_size

    @property
    def server_record(self) -> dict[str, Any] | None:
        # A server that failed to start has an empty record: report it as none.
        return (self.server.record or None) if self.server else None

    @property
    def stopped(self) -> bool:
        """A server that has no live process: it exited, or a start replaced it and then
        refused before launching (llama.cpp moved or rebuilt, say)."""
        if self.server is None:
            return False
        proc = self.server.proc
        return proc is None or proc.poll() is not None

    def start(
        self,
        model: ModelEntry,
        weights: Path,
        ctx_size: int,
        steering: vectors.Steering | None = None,
    ) -> dict[str, Any] | None:
        """Start llama-server with `weights`, a context of `ctx_size` and `steering`'s vector.
        The old server stops first, so two chat models never share memory (meaning search's
        embedder, a second LlamaServer, can run beside it; the memory check counts both).
        Returns the new server_start record, or None if the same GGUF already runs at that
        size with the same steering."""
        # `fetch` replaces a file by rename: same path, new bytes, while the server still has
        # the old ones open. Its inode, size and modification time tell the two apart without
        # hashing (fetch and verify hash). An overwrite in place isn't covered.
        try:
            st = weights.stat()
        except FileNotFoundError:
            raise ConfigError(
                f"The model file {weights} is missing. {fetch_hint(model.name)}"
            ) from None
        identity = (st.st_ino, st.st_size, st.st_mtime_ns)
        server = self.server
        if (
            server
            and server.gguf == weights.absolute()
            and server.config.ctx_size == ctx_size
            and self.weights_identity == identity
            and steering_key(server.steering) == steering_key(steering)
            and server.proc
            and server.proc.poll() is None
        ):
            return None
        if steering is not None:
            # Hashed again before every start: the server reads the path, maybe hours after
            # the chat checked it. On a mismatch nothing stops, and the old server runs on.
            vector = steering.vector
            steering = dataclasses.replace(
                steering, vector=vectors.load(vector.path, vector.sha256)
            )
        self.close()
        self.model = model
        # The size goes into the flags, so argv, the /props check and the log all agree.
        config = dataclasses.replace(self.config, flags={**self.config.flags, "ctx-size": ctx_size})
        self.server = LlamaServer(config, weights, self.logs_dir, self.on_server, steering)
        record = self.server.start()
        self.weights_identity = identity
        return record

    def describe(self) -> dict[str, Any]:
        record = self.server_record or {}
        return {
            "backend": NAME,
            "server_id": record.get("server_id"),
            "build_info": record.get("build_info"),
            "llama_cpp_commit": self.config.llama_cpp_commit,
        }

    def complete(
        self,
        token_ids: list[int],
        settings: dict[str, Any],
        stop: list[str],
        on_text: Callable[[str], None] | None = None,
        request: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
        steering: vectors.Steering | None = None,
    ) -> Result:
        """`request`: a logged request body (prompt left out) to send exactly as it was,
        for replay; otherwise it is built from `settings` and `stop`. `cancel` is checked
        between stream lines; the request asks llama-server for a ping each second while it
        reads the prompt (sse_ping_interval), so this returns within about a second even
        then. The server stops once it finishes its current batch of prompt tokens
        (measured: M1)."""
        if self.server is None:
            raise BackendError("No model loaded.")
        if steering_key(self.server.steering) != steering_key(steering):
            raise BackendError("This server wasn't started with the call's steering.")
        logged_steering = steering.to_json() if steering is not None else None
        # A replay sends the logged request exactly: its steering must be the one applied.
        # A request logged before v0.4.7 has no steering key: unsteered.
        if request is not None and request.get("steering") != logged_steering:
            raise BackendError(
                f"The logged request's steering {request.get('steering')!r} isn't the "
                f"steering given ({logged_steering!r})."
            )
        proc = self.server.proc
        if self.stopped:
            what = (
                f"llama-server has stopped (exit code {proc.poll()})"
                if proc is not None
                else "llama-server didn't start"
            )
            raise BackendError(
                f"{what}, so nothing was sent. Start the model again to go on.",
                server_stopped=True,
            )
        if request is None:
            request = build_request(token_ids, settings, stop)
            logged_request = {k: v for k, v in request.items() if k != "prompt"}
            logged_request["steering"] = logged_steering  # logged, never sent (the flags)
        else:
            logged_request = dict(request)  # as logged: an old one has no steering key
            request = {k: v for k, v in request.items() if k != "steering"}
            request["prompt"] = list(token_ids)
        # What was actually sent: on replay, the logged values, not today's table.
        backend_settings = {
            k: {"value": request[k], "origin": "backend"} for k in NEUTRAL_REQUEST if k in request
        }
        banned = self.server.record["model_file"]["suppress_tokens"]  # applied to every reply
        if banned:
            backend_settings["suppress_tokens"] = {"value": banned, "origin": "model file"}
        texts: list[str] = []
        ids: list[int] = []

        def partial(stop_type: str, flags: list[str]) -> Result:
            return Result(
                text="".join(texts),
                token_ids=list(ids),
                stop_type=stop_type,
                cancelled_during=(
                    ("reply" if ids else "prompt") if stop_type == "cancelled" else None
                ),
                request=logged_request,
                backend_settings=backend_settings,
                flags=flags,
            )

        final: dict[str, Any] | None = None
        timeout = httpx.Timeout(connect=10, read=600, write=60, pool=10)
        try:
            # The server is plain HTTP on 127.0.0.1. trust_env=False ignores proxy variables;
            # verify=False stops httpx loading a CA bundle that a plain-HTTP call never uses.
            with httpx.stream(
                "POST",
                f"{self.server.url}/completion",
                headers=self.server.headers,
                json=request,
                timeout=timeout,
                trust_env=False,
                verify=False,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise BackendError(
                        f"llama-server returned HTTP {response.status_code}: {response.text}",
                        partial("none", []),
                    )
                for line in sse_lines(response.iter_bytes()):
                    if cancel is not None and cancel.is_set():
                        # Leaving the stream closes it, which tells llama-server to stop.
                        return partial("cancelled", ["cancelled"])
                    try:
                        chunk = parse_sse_line(line)
                    except ValueError as e:
                        raise BackendError(
                            f"llama-server sent a malformed stream chunk: {line[:200]!r}",
                            partial("none", []),
                        ) from e
                    if chunk is None:
                        continue
                    if "error" in chunk:
                        raise BackendError(
                            f"llama-server error: {json.dumps(chunk['error'])}",
                            partial("none", []),
                        )
                    text = chunk.get("content", "")
                    texts.append(text)
                    ids.extend(chunk.get("tokens", []))
                    if text and on_text:
                        on_text(text)
                    if chunk.get("stop"):
                        final = chunk
                        break
        except KeyboardInterrupt:
            # Closing the stream tells llama-server to stop generating.
            return partial("cancelled", ["cancelled"])
        except httpx.HTTPError as e:
            if (code := exited_with(proc)) is not None:
                raise stopped_during(code, partial("none", [])) from e
            raise BackendError(f"llama-server connection failed: {e}", partial("none", [])) from e
        if final is None:
            if (code := exited_with(proc)) is not None:
                raise stopped_during(code, partial("none", []))
            raise BackendError(
                "llama-server's stream ended without a final response.", partial("none", [])
            )

        gs = final.get("generation_settings", {})
        timings = final.get("timings", {})
        checks = check_generation_settings(request, gs)
        checks["tokens_evaluated"] = {
            "ok": final.get("tokens_evaluated") == len(token_ids),
            "client": len(token_ids),
            "server": final.get("tokens_evaluated"),
        }
        checks["no_cache_reuse"] = {
            "ok": timings.get("cache_n") == 0,
            "cache_n": timings.get("cache_n"),
        }
        # /props was checked at start. Nothing can change a running server's model or context
        # (server.toml allows no router flags and no --props), and its key keeps other
        # clients out, so what's left to check is that the process is still the one started.
        # The reply is complete: a failure here is a failed check, never a lost reply.
        proc = self.server.proc
        exit_code = proc.poll() if proc is not None else "never started"
        checks["server_running"] = {"ok": exit_code is None}
        if exit_code is not None:
            checks["server_running"]["exit_code"] = exit_code
        if banned:  # the reply and the list are in the same record, so only ok is logged
            checks["suppress_tokens"] = {"ok": not set(ids) & set(banned)}
        if steering is not None:  # the server's start passed check_vector_start
            checks["steering_start"] = {"ok": True, "server_id": self.server.server_id}

        flags = []
        if final.get("truncated"):
            flags.append("context_full")
        elif final.get("stop_type") == "limit":
            flags.append("length")
        if len(ids) < (final.get("tokens_predicted") or 0):
            flags.append("incomplete_output_ids")
        flags += [f"check_failed:{name}" for name, check in checks.items() if not check["ok"]]
        return Result(
            text="".join(texts),
            token_ids=ids,
            stop_type=final.get("stop_type", "none"),
            stopping_word=final.get("stopping_word", ""),
            n_prompt=final.get("tokens_evaluated"),
            n_predicted=final.get("tokens_predicted"),
            timings=timings,
            request=logged_request,
            response=final,
            backend_settings=backend_settings,
            not_applied=[],
            flags=flags,
            checks=checks,
        )

    def close(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None
