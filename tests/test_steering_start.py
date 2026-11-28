"""A steered llama.cpp start (v0.4.7 B3; decisions 4 to 7; states 1, 2, 9a, 9b, 11, 12, 15).

The real llama.cpp backend on the fake llama-server (tests/fake_llama.py), whose start log
gains a steered start's lines: llama.cpp's second graph reservation when the vector loads,
its loader's error lines when it doesn't, or neither. G13's live test (tests/test_live.py)
is where those lines were seen; the live tier runs the same path on SmolLM2.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from clarity_chat import steering
from clarity_chat.backends import BackendError, llama_cpp
from clarity_chat.backends.llama_cpp import (
    LlamaCppBackend,
    device_placement,
    graph_splits,
    memory_use,
    placement_problem,
)
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from tests.fake_llama import (
    CONTEXT_LINE,
    LOADED,
    NODES,
    PREFIX,
    SLOTS_LINE,
    dump,
    fake_llama,
    graph_line,
    loaded,
    loading,
    reservation,
    startup,
    template_example,
    token_text,
)
from tests.test_ctx_size import flag
from tests.test_llama_cpp import SETTINGS
from tests.test_meaning import raised
from tests.test_steering import FIXTURE, directions, sha, steer_root, steered, write
from tests.test_steering_llama_cpp import model_like

CALM = "local/vectors/calm.gguf"  # steer_root's vector: blocks 1 and 2, width 8


@pytest.fixture
def server(monkeypatch, root, tiny_tokenizer):
    """The llama.cpp backend on the fake server, before any start: (backend, what the fake
    saw, the tiny model's entry, its GGUF)."""
    steer_root(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    backend = LlamaCppBackend(load_server_config(root), root / "logs")
    model = load_models(root)["tiny"]
    yield backend, seen, model, root / "models" / "tiny" / "tiny-bf16.gguf"
    backend.close()


def steer(root: Path, scale: float = 4.0, layers=(1, 2), file: str = CALM) -> steering.Steering:
    path = root / file
    vector = steering.load(path, sha(path.read_bytes()))
    return steering.Steering(file, vector, scale, list(layers))


def started(backend, model, weights, steer_with) -> BaseException | None:
    return raised(lambda: backend.start(model, weights, 8192, steering=steer_with))


# --- what the start passes (decision 4) --------------------------------------------------


def test_a_steered_start_passes_both_flags_after_server_toml_s(server):
    backend, seen, model, weights = server
    root = weights.parents[2]
    write(root / "local" / "vectors" / "three.gguf", directions(1, 2, 3, width=8))
    refused = started(
        backend, model, weights, steer(root, 4.0, (1, 2, 3), "local/vectors/three.gguf")
    )
    assert refused is None, f"the steered start was refused: {refused!r}"
    argv = seen["argv"][-1]
    vector = (root / "local" / "vectors" / "three.gguf").absolute()
    assert flag(argv, "--control-vector-scaled") == f"{vector}:4.0"
    at = argv.index("--control-vector-layer-range")
    assert argv[at + 1 : at + 3] == ["1", "3"]
    assert argv.index("--ctx-size") < argv.index("--control-vector-scaled") < argv.index("--host")


def test_an_unsteered_start_passes_neither_flag(server):
    backend, seen, model, weights = server
    refused = started(backend, model, weights, None)
    assert refused is None, f"the unsteered start was refused: {refused!r}"
    assert not [a for a in seen["argv"][-1] if a.startswith("--control-vector")]


REPENG = steering.load(FIXTURE, sha(FIXTURE.read_bytes()))  # blocks 10, 15 and 20
TINY = float(np.finfo(np.float32).tiny)


def normal_float32(x: float) -> bool:
    with np.errstate(over="ignore"):
        return bool(np.isfinite(np.float32(x))) and abs(float(np.float32(x))) >= TINY


@given(st.floats(allow_nan=False, allow_infinity=False).filter(normal_float32))
@example(1 + 2**-24)  # as a double's text, stof reads 1 + 2**-23 (review finding 1)
def test_the_scale_s_text_reads_back_as_its_float32(scale):
    """llama.cpp reads SCALE with std::stof, in one rounding: the text must name exactly the
    float32 that Transformers multiplies by, so any reader that rounds correctly gets it.
    A subnormal scale is left out: stof refuses it."""
    argv_for = getattr(llama_cpp, "steering_argv", None)
    assert argv_for is not None, "llama_cpp has no steering_argv"
    argv = argv_for(steering.Steering("v.gguf", REPENG, scale, [10, 15, 20]))
    text = argv[1].rpartition(":")[2]
    assert float(text) == float(np.float32(scale)), (scale, text)
    assert argv[3:] == ["10", "20"]


# --- when the server is kept (decision 5) -----------------------------------------------


def test_the_same_steering_keeps_the_server_and_any_change_starts_a_new_one(server):
    backend, seen, model, weights = server
    root = weights.parents[2]
    write(root / "local" / "vectors" / "other.gguf", directions(1, 2, width=8, seed=1))
    runs = [
        ("first", steer(root), 1),
        ("the same", steer(root), 1),
        ("a new scale", steer(root, 2.0), 2),
        ("a new file", steer(root, 2.0, file="local/vectors/other.gguf"), 3),
        ("new blocks", steer(root, 2.0, (2,), "local/vectors/other.gguf"), 4),
        ("none", None, 5),
        ("none again", None, 5),
    ]
    for what, steer_with, starts in runs:
        refused = started(backend, model, weights, steer_with)
        assert refused is None, f"{what}: refused: {refused!r}"
        assert len(seen["procs"]) == starts, f"{what}: {len(seen['procs'])} starts"


# --- the start check (decision 6; states 9a, 9b, 11) ------------------------------------

NOT_STEERED = (
    "llama-server started without the vector: its start log doesn't show "
    "local/vectors/calm.gguf added at blocks 1 to 2. Clarity stopped it. llama.cpp said: {}."
)


@pytest.mark.parametrize(
    "mode, said",
    [
        (
            "fails",
            "cmn  common_contr: failed to load control vector file from {path}; "
            "cmn  common_contr: no valid control vector files passed",
        ),
        ("silent", "nothing"),
        ("wrong", "nothing"),
    ],
)
def test_a_start_without_the_vector_is_stopped_with_llama_cpp_s_lines(server, mode, said):
    backend, seen, model, weights = server
    root = weights.parents[2]
    seen["cvec"] = mode
    refused = started(backend, model, weights, steer(root))
    path = (root / CALM).absolute()
    assert isinstance(refused, BackendError), f"not refused as state 11: {refused!r}"
    assert str(refused) == NOT_STEERED.format(said.format(path=path))
    assert seen["procs"][-1].terminated, "the unsteered server was left running"


def test_a_node_count_in_the_model_file_s_metadata_doesn_t_pass_a_failed_load(
    monkeypatch, server, tiny_tokenizer
):
    """Checkpoint 1, finding 7: the key dump prints `general.name` as the file has it, so a
    name that reads like a graph line mustn't count as llama.cpp's reservation. Two nodes
    under the real one is what a load at calm's blocks 1 and 2 would add."""
    backend, _, model, weights = server
    root = weights.parents[2]
    spoofed = startup().replace("= Tiny\n", f"= graph: nodes = {NODES - 2}\n")
    assert spoofed != startup()
    seen = fake_llama(monkeypatch, root, {"tiny": spoofed}, tiny_tokenizer)
    seen["cvec"] = "fails"
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, BackendError), f"not refused as state 11: {refused!r}"
    assert str(refused).startswith(NOT_STEERED.partition("{}")[0])
    assert seen["procs"][-1].terminated, "the unsteered server was left running"


def test_a_graph_line_in_a_token_s_text_doesn_t_pass_a_failed_load(
    monkeypatch, server, tiny_tokenizer
):
    """Checkpoint 1, finding 7, route 1: `print_info` prints a special token's text with its
    newlines, before the context and its reservations, so the text can hold a whole graph
    line. Only the lines from the context's construction on are read."""
    backend, _, model, weights = server
    root = weights.parents[2]
    spoofed = token_text(f"x\n{graph_line(NODES - 2)}\n")
    assert spoofed != startup()
    seen = fake_llama(monkeypatch, root, {"tiny": spoofed}, tiny_tokenizer)
    seen["cvec"] = "fails"
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, BackendError), f"not refused as state 11: {refused!r}"
    assert seen["procs"][-1].terminated, "the unsteered server was left running"


def test_a_refusal_quotes_no_model_file_text(monkeypatch, server, tiny_tokenizer):
    """Checkpoint 1, finding 12: a token's text can hold the words llama.cpp's loader
    uses, but only the loader's own lines are quoted."""
    backend, _, model, weights = server
    root = weights.parents[2]
    spoofed = token_text("x\nthe control vector loaded fine\n")
    seen = fake_llama(monkeypatch, root, {"tiny": spoofed}, tiny_tokenizer)
    seen["cvec"] = "silent"
    refused = started(backend, model, weights, steer(root))
    assert str(refused) == NOT_STEERED.format("nothing")


def test_a_graph_line_in_the_chat_template_s_example_doesn_t_pass_a_failed_load(server):
    """Route 2: llama-server prints the template's example, newlines and all, after the
    model has loaded. Only the lines up to the server's slots line are read."""
    backend, seen, model, weights = server
    root = weights.parents[2]
    seen["cvec"] = "fails"
    seen["after"] = template_example(f"x\n{graph_line(NODES + 2)}\n")
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, BackendError), f"not refused as state 11: {refused!r}"
    assert seen["procs"][-1].terminated, "the unsteered server was left running"


def test_the_splits_logged_are_llama_cpp_s_not_the_template_example_s(server):
    backend, seen, model, weights = server
    root = weights.parents[2]
    seen["after"] = template_example(f"x\n{graph_line(NODES + 2, splits=9)}\n")
    record = backend.start(model, weights, 8192, steering=steer(root))
    assert record["graph_splits"] == {"prompt": 2, "reply": 2}


# A real start prints the context's line and the slots line once each: `fit` is off and
# there is no draft, MTP or mmproj model (fix review, finding 1).
TWO_CONTEXTS = (
    "llama-server's start log has 2 lines that look like llama.cpp's "
    '"llama_context: constructing llama_context", where a real start prints one. Text from '
    "the model file can print such lines, so Clarity can't read the start's figures, and "
    "stopped the model."
)


def test_a_context_line_after_the_model_loads_refuses_a_good_start(server):
    backend, seen, model, weights = server
    root = weights.parents[2]
    seen["after"] = template_example(f"x\n{CONTEXT_LINE}")
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused) == TWO_CONTEXTS
    assert seen["procs"][-1].terminated, "the server was left running"


def test_a_context_line_in_the_template_example_refuses_a_start_rather_than_read_no_memory(
    monkeypatch, server, tiny_tokenizer
):
    """The fix review's first probe: with the window empty, the cache and compute read as 0
    and a start too big for the GPU would go ahead."""
    backend, _, model, weights = server
    seen = fake_llama(
        monkeypatch,
        weights.parents[2],
        {"tiny": startup() + template_example(f"x\n{CONTEXT_LINE}x")},
        tiny_tokenizer,
    )
    refused = started(backend, model, weights, None)
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused) == TWO_CONTEXTS
    assert seen["procs"][-1].terminated, "the server was left running"


def test_a_forged_model_load_in_the_template_example_doesn_t_pass_a_failed_load(server):
    """The fix review's second probe: a context line, two reservations a vector's blocks
    apart and a slots line, after the real slots line of a start whose vector failed."""
    backend, seen, model, weights = server
    root = weights.parents[2]
    seen["cvec"] = "fails"
    forged = f"{CONTEXT_LINE}{graph_line(NODES)}\n{graph_line(NODES + 2)}\n{SLOTS_LINE}"
    seen["after"] = template_example(f"x\n{forged}x")
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused) == TWO_CONTEXTS
    assert seen["procs"][-1].terminated, "the unsteered server was left running"


def test_a_slots_line_in_a_token_s_text_refuses_a_start(monkeypatch, server, tiny_tokenizer):
    backend, _, model, weights = server
    seen = fake_llama(
        monkeypatch, weights.parents[2], {"tiny": token_text(f"x\n{SLOTS_LINE}x")}, tiny_tokenizer
    )
    refused = started(backend, model, weights, None)
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert '2 lines that look like llama.cpp\'s "load_model: initializing, n_slots"' in str(refused)
    assert seen["procs"][-1].terminated, "the server was left running"


def test_a_vector_changed_before_a_new_start_starts_nothing_and_the_old_server_runs_on(server):
    """State 9a: the re-hash before the start. `/set steering_scale`'s form waits for B4."""
    backend, seen, model, weights = server
    root = weights.parents[2]
    first = steer(root)
    assert started(backend, model, weights, first) is None
    pinned = first.vector.sha256
    write(root / CALM, directions(1, 2, width=8, seed=7))
    now = sha((root / CALM).read_bytes())
    refused = started(backend, model, weights, dataclasses.replace(first, scale=2.0))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused) == f"{root / CALM} has sha256 {now}, not the pinned {pinned}."
    assert len(seen["procs"]) == 1 and not seen["procs"][0].terminated
    reply = raised(lambda: backend.complete([1, 2], SETTINGS, [], steering=first))
    assert not isinstance(reply, Exception), f"the old server didn't answer: {reply!r}"


def test_a_vector_that_changes_during_the_start_stops_the_server(server):
    """State 9b: the size and time after the start equal those of the read before it."""
    backend, seen, model, weights = server
    root = weights.parents[2]
    path = root / CALM

    def touch() -> None:
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    seen["during_start"] = touch
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, ConfigError), f"not refused as state 9b: {refused!r}"
    assert str(refused) == (
        f"{path} changed while the model was starting, so Clarity stopped the model rather "
        "than answer with a vector it hadn't hashed."
    )
    assert seen["procs"][-1].terminated, "the server was left running"


def test_a_vector_swapped_in_with_its_old_time_during_the_start_stops_the_server(server):
    """State 9b, review finding F1: another file of the same size renamed into place with
    the old file's modified time (as `rsync -a` or `cp -p` leave it) is caught too."""
    backend, seen, model, weights = server
    root = weights.parents[2]
    path = root / CALM

    def swap() -> None:
        old = path.stat()
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF  # other bytes, the same size
        other = path.with_name("other.gguf")
        other.write_bytes(bytes(data))
        os.utime(other, ns=(old.st_atime_ns, old.st_mtime_ns))
        os.replace(other, path)
        new = path.stat()
        assert (new.st_size, new.st_mtime_ns) == (old.st_size, old.st_mtime_ns)

    seen["during_start"] = swap
    refused = started(backend, model, weights, steer(root))
    assert isinstance(refused, ConfigError), f"not refused as state 9b: {refused!r}"
    assert str(refused).startswith(f"{path} changed while the model was starting")
    assert seen["procs"][-1].terminated, "the server was left running"


# --- the readers take the last reservation (decision 7, must 2) ---------------------------

KV = "0.00.062.265 I llama_kv_cache:       MTL0 KV buffer size =   180.00 MiB\n"


def test_a_steered_start_s_memory_and_splits_are_the_last_reservation_s():
    plain = loading() + loaded(reservation(NODES))
    steered_log = loading() + loaded(reservation(NODES), reservation(NODES + 11, splits=3))
    figures = memory_use(steered_log, device_placement(steered_log))
    assert figures == memory_use(plain, device_placement(plain))
    assert figures[0]["compute_mib"] == 22.17
    assert graph_splits(steered_log) == {"prompt": 3, "reply": 3}


def test_a_graph_line_in_the_model_file_s_metadata_isn_t_read_as_the_splits():
    """Checkpoint 1, finding 7: only `sched_reserve`'s own line gives the splits."""
    spoofed = loading().replace("= Tiny\n", "= graph: nodes = 1, splits = 9\n")
    assert graph_splits(spoofed) is None
    assert graph_splits(spoofed + LOADED) == {"prompt": 2, "reply": 2}


# --- the model file's text in the placement and memory lines (whole-branch review, finding
# 2). Model files are untrusted (design.md): their text reaches the start log in the key
# dump (a key's name as it is, its value escaped and cut to 40 characters), in the
# vocabulary's and print_info's lines and the end tokens' logit biases (as it is, newlines
# and all) and in the chat template's example. -------------------------------------------

GPU = (
    f"{PREFIX}llama_prepare_model_devices: using device MTL0 (Apple M2 Max) (unknown id) - "
    "25558 MiB free\n"
)
ON_GPU = (
    f"{PREFIX}load_tensors: offloaded 31/31 layers to GPU\n"
    f"{PREFIX}load_tensors:   CPU_Mapped model buffer size =    56.25 MiB\n"
    f"{PREFIX}load_tensors:  MTL0_Mapped model buffer size =   256.50 MiB\n"
)
ON_CPU = (  # the GPU hidden, as a sandbox can: no device line
    f"{PREFIX}load_tensors: offloaded 0/31 layers to GPU\n"
    f"{PREFIX}load_tensors:   CPU_Mapped model buffer size =   312.75 MiB\n"
)
FORGED_GPU = (
    f"{PREFIX}llama_prepare_model_devices: using device MTL0 (A) (B) - 1 MiB free\n"
    f"{PREFIX}load_tensors:  MTL0_Mapped model buffer size =     1.00 MiB\n"
)
FORGED_LOAD = (  # a buffer line right after a load_tensors line (fix review, finding 4)
    f"{PREFIX}load_tensors: offloaded 31/31 layers to GPU\n"
    f"{PREFIX}load_tensors:  MTL0_Mapped model buffer size = 1.00 MiB\n"
)


def start_log(
    devices: str = GPU, buffers: str = ON_GPU, eog: str = "</s>", unused: str = ""
) -> str:
    """A whole start log in the order llama-server prints it at the pin, read in its source
    (src/llama-model-loader.cpp:1979, src/llama.cpp:306, src/llama-vocab.cpp,
    src/llama-model.cpp:1728, :2133 and :2139, common/common.cpp:1360,
    src/llama-context.cpp:180) and seen in a real start's log: the key dump, the file's
    size, the devices, the vocabulary, the tensors' load and their buffers, the end token's
    logit bias, then the context. `eog` is the end token's text and `unused` the name of a
    tensor the file has and llama.cpp doesn't use."""
    tensor = (
        f"{PREFIX[:-2]}W model has unused tensor {unused} (size = 64 bytes) -- ignoring\n"
        if unused
        else ""
    )
    return (
        dump()
        + f"{PREFIX}print_info: file format = GGUF V3 (latest)\n"
        + f"{PREFIX}print_info: file type   = BF16\n"
        + f"{PREFIX}print_info: file size   = 256.63 MiB (16.00 BPW) \n"
        + devices
        + f"{PREFIX}load: printing all EOG tokens:\n"
        + f"{PREFIX}load:   - 2 ('{eog}')\n"
        + f"{PREFIX}print_info: arch                  = llama\n"
        + f"{PREFIX}print_info: EOG token             = 2 '{eog}'\n"
        + f"{PREFIX}load_tensors: loading model tensors, this can take a while... "
        "(load_mode = mmap)\n"
        + tensor
        + buffers
        + f"{PREFIX}cmn  common_init_: added {eog} logit bias = -inf\n"
        + CONTEXT_LINE
        + KV
        + reservation(NODES)
        + SLOTS_LINE
    )


def test_a_start_log_in_llama_cpp_s_order_reads_as_before():
    log = start_log()
    assert device_placement(log) == {
        "devices": [{"name": "MTL0", "description": "Apple M2 Max", "free_mib": 25558}],
        "model_buffers": [
            {"buffer": "CPU_Mapped", "mib": 56.25},
            {"buffer": "MTL0_Mapped", "mib": 256.5},
        ],
    }
    assert memory_use(log, device_placement(log))[0] == {
        "device": "MTL0",
        "limit_mib": 25558,
        "model_mib": 256.5,
        "kv_mib": 180.0,
        "compute_mib": 22.17,
        "total_mib": 458.67,
    }


@pytest.mark.parametrize(
    "spoofed",
    [
        # the review's probe: a 35-character name in the key dump
        start_log().replace("= Tiny\n", "= MTL0 compute buffer size = 9999 MiB\n"),
        # an end token's text, printed as it is before the context
        start_log(eog=f"x\n{PREFIX}llama_kv_cache:       MTL0 KV buffer size =  9999.00 MiB\nx"),
        start_log(
            eog=f"x\n{PREFIX}sched_reserve:       MTL0 compute buffer size =  9999.00 MiB\nx"
        ),
        # the chat template's example, after the slots line
        start_log()
        + template_example(
            f"x\n{PREFIX}sched_reserve:       MTL0 compute buffer size =  9999.00 MiB\nx"
        ),
    ],
    ids=["key dump", "token text KV", "token text compute", "template example"],
)
def test_the_model_file_s_text_adds_no_memory(spoofed):
    assert memory_use(spoofed, device_placement(spoofed)) == memory_use(
        start_log(), device_placement(start_log())
    )


@pytest.mark.parametrize(
    "spoofed",
    [
        # the review's probe: two metadata strings in the key dump (30 and 38 characters)
        start_log(devices="", buffers=ON_CPU)
        .replace("= Tiny\n", "= MTL0 model buffer size = 1 MiB\n")
        .replace("= gpt2\n", "= using device MTL0 (A) (B) - 1 MiB free\n"),
        # an end token's text: in the vocabulary's lines, print_info's and the logit bias's
        start_log(devices="", buffers=ON_CPU, eog=f"x\n{FORGED_GPU}{FORGED_LOAD}x"),
    ],
    ids=["key dump", "token text"],
)
def test_the_model_file_s_text_doesn_t_put_a_cpu_start_on_the_gpu(spoofed):
    """The case the placement check exists for: a GPU hidden from llama-server."""
    plain = start_log(devices="", buffers=ON_CPU)
    assert device_placement(spoofed) == device_placement(plain)
    assert placement_problem({"n-gpu-layers": "all"}, device_placement(spoofed)) is not None


def test_an_unused_tensor_on_a_cpu_only_build_keeps_the_model_s_buffer():
    """Fix review, deletion 1: a build without GPU offload prints no "offloaded" line
    (src/llama-model.cpp:2121), so an unused tensor's warning (src/llama-model-loader.cpp:1275)
    comes right before the buffer line. Its name is from llama.cpp's own table
    (src/llama-arch.cpp:1293-1305), not the file's text."""
    cpu = f"{PREFIX}load_tensors:   CPU_Mapped model buffer size =   312.75 MiB\n"
    log = start_log(devices="", buffers=cpu, unused="rope_freqs.weight")
    assert device_placement(log)["model_buffers"] == [{"buffer": "CPU_Mapped", "mib": 312.75}]


def test_a_key_name_in_the_model_file_doesn_t_forge_the_gpu_s_limit():
    """Fix review, finding 2: the key dump prints a key's name as it is (`%42s`,
    src/llama-model-loader.cpp:904), so a name can print a file size, a device and a
    vocabulary line before the real ones. With two file-size lines no device is read, so a
    start asked to use the GPU is refused rather than checked against a made-up limit."""
    name = (
        f"x\n{PREFIX}print_info: file size   = 1 MiB\n"
        f"{PREFIX}llama_prepare_model_devices: using device MTL0 (Apple M2 Max) (unknown id) - "
        f"999999 MiB free\n{PREFIX}load: x"
    )
    spoofed = start_log().replace(dump(), dump((name, "str", "x")))
    assert spoofed != start_log()
    placement = device_placement(spoofed)
    assert placement["devices"] == []
    assert placement_problem({"n-gpu-layers": "all"}, placement) is not None


def test_a_second_tensor_load_line_in_the_model_file_s_text_refuses_a_gpu_start():
    """A token's text can print a whole tensor load, first line and all; the real one can't
    be told from it, so no buffer is read and the start is refused, as a start on the CPU.
    The GPU is there and the weights went to the CPU, so only a forged buffer could pass."""
    load = f"{PREFIX}load_tensors: loading model tensors, this can take a while...\n"
    spoofed = start_log(buffers=ON_CPU, eog=f"x\n{load}{FORGED_LOAD}x")
    assert placement_problem({"n-gpu-layers": "all"}, device_placement(spoofed)) is not None


# --- the call record (decision 7; states 2, 12, 15) -------------------------------------


@pytest.fixture
def chat(monkeypatch, root, tiny_tokenizer):
    """A session on the real llama.cpp backend and the fake server, whose tiny model is a
    made-up llama file of width 8 and 4 blocks: (session, what the fake saw)."""
    from clarity_chat.session import Session

    steer_root(root)
    model_like(root)
    seen = fake_llama(monkeypatch, root, {"tiny": startup()}, tiny_tokenizer)
    backend = LlamaCppBackend(load_server_config(root), root / "logs")
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    yield session, seen
    session.close()


def test_a_steered_llama_cpp_call_logs_its_steering_and_sends_none(chat):
    from clarity_chat.cli import format_status
    from clarity_chat.logs import read_records

    session, seen = chat
    root = session.root
    logged = {
        "file": CALM,
        "sha256": sha((root / CALM).read_bytes()),
        "scale": 4.0,
        "layers": [1, 2],
    }
    session.new_conversation(steered(root, backend="llama.cpp"))
    said = raised(lambda: session.say("Hi"))
    assert said is None, f"the steered call was refused: {said!r}"
    assert session.conversation is not None
    rows = list(read_records(session.conversation.log.path))
    (start,) = [r for r in rows if r["type"] == "server_start"]
    (call,) = [r for r in rows if r["type"] == "call"]
    assert start["steering"] == logged
    assert call["request"]["steering"] == logged
    assert call["checks"]["steering_start"] == {"ok": True, "server_id": start["server_id"]}
    assert "steering" not in call["checks"], "llama.cpp ran Transformers' check"
    assert "steering" not in seen["requests"][-1], "the steering was sent to llama-server"
    assert "steered ×4.0" in format_status(call, steerable=True)


def test_an_unsteered_llama_cpp_call_logs_null_and_its_replay_sends_none(chat):
    session, seen = chat
    session.new_conversation(steered(session.root, backend="llama.cpp", scale="0"))
    record = session.say("Hi").record
    assert record["request"].get("steering", "absent") is None
    assert "steering_start" not in record["checks"]
    assert session.conversation is not None
    seen["replies"].append(record["output_text"])
    replay = session.replay(session.conversation.log.path, record["call_id"])
    assert replay["identical"], replay
    assert "steering" not in seen["requests"][-1], "the replay sent the logged steering"


def test_a_call_whose_steering_isnt_the_server_s_is_refused_and_nothing_is_sent(server):
    """State 12: `complete` refuses a call the running server wasn't started for."""
    backend, seen, model, weights = server
    root = weights.parents[2]
    assert started(backend, model, weights, None) is None
    for steer_with in (steer(root), steer(root, 2.0)):
        refused = raised(lambda s=steer_with: backend.complete([1, 2], SETTINGS, [], steering=s))
        assert isinstance(refused, BackendError), f"not refused: {refused!r}"
        assert str(refused) == "This server wasn't started with the call's steering."
    assert seen["requests"] == []
    assert started(backend, model, weights, steer(root)) is None
    refused = raised(lambda: backend.complete([1, 2], SETTINGS, [], steering=None))
    assert str(refused) == "This server wasn't started with the call's steering."
    assert seen["requests"] == []


def test_a_replay_whose_request_and_steering_disagree_is_refused(server):
    backend, seen, model, weights = server
    root = weights.parents[2]
    steer_with = steer(root)
    assert started(backend, model, weights, steer_with) is None
    result = backend.complete([1, 2], SETTINGS, [], steering=steer_with)
    request = {**result.request, "steering": {**result.request["steering"], "scale": 2.0}}
    sent = len(seen["requests"])
    refused = raised(
        lambda: backend.complete([1, 2], SETTINGS, [], request=request, steering=steer_with)
    )
    assert isinstance(refused, BackendError), f"not refused: {refused!r}"
    assert len(seen["requests"]) == sent


# --- old logs: a llama.cpp call with no request.steering --------------------------------


def test_an_old_llama_cpp_call_without_steering_reads_as_unsteered(chat):
    from clarity_chat.cli import format_status
    from clarity_chat.export import export_log
    from tests.test_steering import doctor

    session, seen = chat
    session.new_conversation(steered(session.root, backend="llama.cpp", scale="0"))
    record = session.say("Hi").record
    assert session.conversation is not None
    path = session.conversation.log.path
    doctor(path, record["call_id"], lambda row: row["request"].pop("steering", None))
    old = next(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("call_id") == record["call_id"]
        and json.loads(line)["type"] == "call"
    )
    assert "steering" not in old["request"]
    assert "not steered" in format_status(old, steerable=True)
    rows = [
        json.loads(line)
        for line in Path(export_log(session, path)["path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["steering"] for row in rows[1:]] == ["none"]
    seen["replies"].append(record["output_text"])
    replay = session.replay(path, record["call_id"])
    assert replay["identical"], replay
    assert "steering" not in seen["requests"][-1]
