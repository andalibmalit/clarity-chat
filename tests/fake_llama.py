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
ERROR = "0.00.062.265 E "  # E for ERROR

# The device and buffer lines a start on a Mac's GPU prints (the placement check needs them,
# each on its own line between the file's size and the vocabulary, and after the tensors'
# load line: src/llama.cpp:306, src/llama-model.cpp:2133 and :2139).
DEVICE_LINES = (
    f"{PREFIX}llama_prepare_model_devices: using device MTL0 (Apple M2 Max) (unknown id) - "
    "25558 MiB free\n"
)
BUFFER_LINES = (
    f"{PREFIX}load_tensors: offloaded 31/31 layers to GPU\n"
    f"{PREFIX}load_tensors:   CPU_Mapped model buffer size =    56.25 MiB\n"
    f"{PREFIX}load_tensors:  MTL0_Mapped model buffer size =   256.50 MiB\n"
)
FILE_LINES = (  # src/llama-model-loader.cpp:1979-1984
    f"{PREFIX}print_info: file format = GGUF V3 (latest)\n"
    f"{PREFIX}print_info: file type   = BF16\n"
    f"{PREFIX}print_info: file size   = 256.63 MiB (16.00 BPW) \n"
)
VOCABULARY = (  # src/llama-vocab.cpp, then the model's print_info
    f"{PREFIX}load: printing all EOG tokens:\n"
    f"{PREFIX}load:   - 2 ('</s>')\n"
    f"{PREFIX}print_info: arch                  = llama\n"
    f"{PREFIX}print_info: EOG token             = 2 '</s>'\n"
)
TENSOR_LOAD_LINE = (  # src/llama-model.cpp:1728
    f"{PREFIX}load_tensors: loading model tensors, this can take a while... (load_mode = mmap)\n"
)
EOG_BIAS = f"{PREFIX}cmn  common_init_: added </s> logit bias = -inf\n"  # common/common.cpp:1360


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


def loading(
    *keys: tuple[str, str, str], devices: str = DEVICE_LINES, buffers: str = BUFFER_LINES
) -> str:
    """The startup log up to the context, in llama.cpp's order: the key dump, the file's
    size, the devices, the vocabulary, the tensors' load and buffers, the end token's bias."""
    return dump(*keys) + FILE_LINES + devices + VOCABULARY + TENSOR_LOAD_LINE + buffers + EOG_BIAS


def reservation(nodes: int, splits: int = 2, mtl0: float = 22.17) -> str:
    """One graph reservation as `sched_reserve` prints it (`src/llama-context.cpp:926` and
    `:940`): each device's compute buffer, then the graph. The buffers and the node count
    are G13's for SmolLM2 on Metal; the input counts are made up, since Clarity reads none."""
    buffers = "".join(
        f"{PREFIX}sched_reserve: {device:>10} compute buffer size = {mib:8.2f} MiB\n"
        for device, mib in (("MTL0", mtl0), ("CPU", 12.52))
    )
    return buffers + (
        f"{PREFIX}sched_reserve: graph: nodes = {nodes}, splits = {splits}, input objects = 4, "
        "input tensors = 4\n"
    )


NODES = 939  # SmolLM2's graph (G13)

# The lines either side of every graph reservation at the start: the context's
# construction (`src/llama-context.cpp:180`), which reserves it once, and the server's
# slots (`tools/server/server-context.cpp:1485`), printed after `common_init_from_params`
# has applied the vector and run the warmup that reserves it again.
CONTEXT_LINE = f"{PREFIX}llama_context: constructing llama_context\n"
SLOTS_LINE = (
    f"{PREFIX}srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192, kv_unified = "
    "'false'\n"
)
KV_LINE = f"{PREFIX}llama_kv_cache:       MTL0 KV buffer size =   180.00 MiB\n"  # G13's


def loaded(*reservations: str, kv: str = KV_LINE) -> str:
    """The model's load in the context, as `model_load` reads it: the context's line, its
    cache's buffer (`src/llama-kv-cache.cpp:280`), the graph reservations, the slots."""
    return CONTEXT_LINE + kv + "".join(reservations) + SLOTS_LINE


LOADED = loaded(reservation(NODES))


def startup(*keys: tuple[str, str, str]) -> str:
    """A whole startup log, to the slots line: `loading`, then one reservation."""
    return loading(*keys) + LOADED


# --- start logs a model file has written into (Checkpoint 1, finding 7) ------------------


def graph_line(nodes: int, splits: int = 2) -> str:
    """A whole graph line, prefix and all, as a model file's text could print it."""
    return reservation(nodes, splits).splitlines()[-1]


def named(name: str) -> str:
    """A start log whose model is called `name`: the key dump prints it as the file has
    it, cut to 40 characters, with each newline as \\n (`src/llama-model-loader.cpp:902`)."""
    return startup().replace("= Tiny\n", f"= {name}\n")


def token_text(text: str) -> str:
    """A start log whose end token's text is `text`, printed as it is, newlines and all
    (`print_info` in `src/llama-vocab.cpp:4335-4346`), before the context exists."""
    eog = "EOG token             = 2 '</s>'"
    return startup().replace(eog, f"EOS token             = 2 '{text}'")


def template_example(text: str) -> str:
    """The chat template's example as llama-server prints it at log-verbosity 4, rendered
    with its newlines (`tools/server/server-context.cpp:1703`), after the slots line."""
    return f"{PREFIX}srv          init: init: chat template, example_format: '{text}'\n"


def vector_lines(argv: list[str], mode: str) -> str:
    """What a start with a control vector adds to the log, by `mode`: "loads" (llama.cpp
    reserves the graph again, one node more per block in the range), "fails" (its loader's
    two error lines and no second reservation: the server runs on unsteered), "silent" (no
    second reservation and no line), or "wrong" (a second reservation one node off). The
    error lines are `common/common.cpp:2008` and `:2114`, through `COM_ERR`'s format."""
    path = argv[argv.index("--control-vector-scaled") + 1].rpartition(":")[0]
    i = argv.index("--control-vector-layer-range")
    blocks = int(argv[i + 2]) - int(argv[i + 1]) + 1
    first = CONTEXT_LINE + KV_LINE + reservation(NODES)
    if mode == "loads":
        return first + reservation(NODES + blocks) + SLOTS_LINE
    if mode == "wrong":
        return first + reservation(NODES + blocks + 1) + SLOTS_LINE
    if mode == "fails":
        return (
            first
            + f"{ERROR}cmn  common_contr: failed to load control vector file from {path}\n"
            + f"{ERROR}cmn  common_contr: no valid control vector files passed\n"
            + SLOTS_LINE
        )
    assert mode == "silent", mode
    return first + SLOTS_LINE


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
    processes started, each request, and the queue of replies (`replies`), to add to.

    A start whose argv has a control vector has `vector_lines` in `seen["cvec"]`'s mode
    ("loads" unless set) in place of `LOADED`, then `seen["after"]` if a test set it, and calls
    `seen["during_start"]` if a test set it. Each start's argv is in `seen["argv"]`."""
    import json

    from clarity_chat.profiles import load_server_config
    from tests.test_llama_cpp import echoed

    pinned = load_server_config(root).llama_cpp_commit

    queue = list(replies or [])
    seen: dict[str, Any] = {"procs": [], "requests": [], "replies": queue, "argv": []}

    def launch(self, argv: list[str], env: dict[str, str]) -> None:
        printed = startups[self.gguf.parent.name]  # or a function of the size (v0.4.4 B3a)
        text = str(printed(self.config.ctx_size) if callable(printed) else printed)
        seen["argv"].append(argv)
        if "--control-vector-scaled" in argv:  # its load takes the place of the plain one
            text = text.replace(LOADED, "")
            text += vector_lines(argv, seen.get("cvec", "loads")) + seen.get("after", "")
            if seen.get("during_start"):
                seen["during_start"]()
        self.output = [text]
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
