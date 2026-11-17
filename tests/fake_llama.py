"""The real llama.cpp backend with llama-server faked: its startup log, its process and its
replies. Each start prints the startup log given for its model, in llama.cpp's own format.

The key dump is written as llama.cpp writes it, read in the source at the local checkout
(f477f79) and seen in the output of the build at the pin (b1b770d): `src/llama-model-loader.cpp`
prints `"%s: - kv %3d: %42s %-16s = %s\\n"` per key, and an array's value is built while it
is at most 40 characters, then cut to its first 37 and "..." (`gguf_kv_to_str`,
`src/llama-impl.cpp`). llama-server's log puts a timestamp and the level before each line
(`common_init` turns both on; `common/log.cpp`): PREFIX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clarity_chat.backends import llama_cpp

SUPPRESS_KEY = "tokenizer.ggml.suppress_tokens"
PREFIX = "0.00.062.265 I "  # minutes.seconds.milliseconds.microseconds, then I for INFO

# The device and buffer lines a start on a Mac's GPU prints (the placement check needs them).
DEVICE_LINES = """\
llama_model_load_from_file_impl: using device MTL0 (Apple M2 Max) (unknown id) - 25558 MiB free
"""
BUFFER_LINES = """\
print_info: EOG token             = 2 '</s>'
load_tensors: offloaded 31/31 layers to GPU
load_tensors:   CPU_Mapped model buffer size =    56.25 MiB
load_tensors:  MTL0_Mapped model buffer size =   256.50 MiB
"""


def array_value(ids: list[int]) -> str:
    """An i32 array's value as llama.cpp prints it: built while at most 40 characters, a
    whole list closed with "]", and anything over 40 cut to 37 and "..."."""
    value, j = "[", 0
    while j < len(ids) and len(value) <= 40:
        value += (", " if j else "") + str(ids[j])
        j += 1
    if j == len(ids):
        value += "]"
    return value if len(value) <= 40 else value[:37] + "..."


def kv_line(i: int, name: str, kind: str, value: str) -> str:
    return f"{PREFIX}llama_model_loader: - kv {i:3d}: {name:>42} {kind:<16} = {value}\n"


def dump(*keys: tuple[str, str, str]) -> str:
    """The key dump llama-server prints at log-verbosity 4: a few of SmolLM2's keys, then
    `keys` as (name, type, value)."""
    rows = [
        ("general.architecture", "str", "llama"),
        ("general.name", "str", "Tiny"),
        ("llama.context_length", "u32", "8192"),
        *keys,
        ("tokenizer.ggml.model", "str", "gpt2"),
    ]
    return (
        f"{PREFIX}llama_model_loader: loaded meta data with {len(rows)} key-value pairs and "
        "272 tensors from model.gguf (version GGUF V3 (latest))\n"
        f"{PREFIX}llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not "
        "apply in this output.\n"
        + "".join(kv_line(i, *row) for i, row in enumerate(rows))
        + f"{PREFIX}llama_model_loader: - type  f32:   61 tensors\n"
        f"{PREFIX}llama_model_loader: - type bf16:  211 tensors\n"
    )


def banning(ids: list[int]) -> tuple[str, str, str]:
    """The suppress key's row for a file that bans `ids`."""
    return (SUPPRESS_KEY, f"arr[i32,{len(ids)}]", array_value(ids))


def startup(*keys: tuple[str, str, str]) -> str:
    """A whole startup log: the device, the key dump, the end token and the buffers."""
    return DEVICE_LINES + dump(*keys) + BUFFER_LINES


class Running:
    """llama-server's process: running until stopped."""

    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


class Stream:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self.lines = lines

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def iter_bytes(self):  # as llama-server's body arrives: lines, each ending "\n"
        for line in self.lines:
            yield line.encode("utf-8") + b"\n"


def fake_llama(
    monkeypatch, root: Path, startups: dict[str, str], tokenizer: Any, replies=None
) -> dict:
    """Fake llama-server for every LlamaServer: `startups` maps a model's name (its folder
    under models/) to the startup log its start prints. Replies are `replies` in turn
    (" ok" after), as the tokenizer's IDs plus its end token. Returns what happened: the
    processes started, each request, and the queue of replies (`replies`), to add to."""
    import json

    from clarity_chat.profiles import load_server_config
    from tests.test_llama_cpp import echoed

    pinned = load_server_config(root).llama_cpp_commit

    queue = list(replies or [])
    seen: dict[str, Any] = {"procs": [], "requests": [], "replies": queue}

    def launch(self, argv: list[str], env: dict[str, str]) -> None:
        self.output = [startups[self.gguf.parent.name]]
        self.proc = Running()
        seen["procs"].append(self.proc)
        return None

    def props(self) -> dict[str, Any]:
        return {
            "model_path": str(self.gguf),
            "default_generation_settings": {"n_ctx": self.config.ctx_size},
            "build_info": "b1-" + self.config.llama_cpp_commit[:7],
        }

    def stream(method, url, headers, json, **kwargs):  # noqa: A002 (httpx's own name)
        seen["requests"].append(json)
        text = queue.pop(0) if queue else " ok"
        ids = [*tokenizer.encode(text), tokenizer.tok.eos_token_id]
        final = {
            "content": "",
            "tokens": [],
            "stop": True,
            "stop_type": "eos",
            "stopping_word": "",
            "tokens_evaluated": len(json["prompt"]),
            "tokens_predicted": len(ids),
            "timings": {"cache_n": 0},
            "truncated": False,
            "prompt": tokenizer.decode(json["prompt"]),
            "generation_settings": echoed(json),
        }
        chunks = [{"content": text, "tokens": ids}, final]
        return Stream(["data: " + dumps(chunk) for chunk in chunks])

    dumps = json.dumps
    monkeypatch.setattr(llama_cpp, "llama_config_files", lambda: [])
    monkeypatch.setattr(llama_cpp, "git_state", lambda d: {"commit": pinned, "dirty": False})
    monkeypatch.setattr(llama_cpp, "server_binary", lambda d: Path("llama-server"))
    monkeypatch.setattr(llama_cpp.LlamaServer, "_launch", launch)
    monkeypatch.setattr(llama_cpp.LlamaServer, "get_props", props)
    monkeypatch.setattr(llama_cpp.LlamaServer, "check_props", lambda self, p: [])
    monkeypatch.setattr(llama_cpp.httpx, "stream", stream)
    return seen
