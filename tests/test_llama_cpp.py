"""The llama.cpp backend without a server: request building, checks, stream parsing, env."""

from __future__ import annotations

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.backends.llama_cpp import (
    NEUTRAL_REQUEST,
    build_request,
    check_generation_settings,
    device_placement,
    eog_tokens,
    graph_splits,
    llama_config_files,
    parse_sse_line,
    placement_problem,
    server_env,
    start_warnings,
)

SETTINGS = {
    "temperature": 0.6,
    "top_k": 0,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42,
    "max_tokens": 1024,
    "samplers": ["top_k", "top_p", "min_p", "temperature"],
}


def echoed(request):
    """What llama-server would report back: floats at float32, plus its own keys."""
    import struct

    gs = {}
    for key, value in request.items():
        if key in (
            "prompt",
            "return_tokens",
            "cache_prompt",
            "return_progress",
            "sse_ping_interval",
            "n_cache_reuse",
            "n_cmpl",
            "n_indent",
            "t_max_predict_ms",
            "reasoning_budget_tokens",
            "reasoning_control",
        ):
            continue
        if isinstance(value, float):
            value = struct.unpack("f", struct.pack("f", value))[0]
        gs[key] = value
    gs.update(
        max_tokens=request["n_predict"],
        generation_prompt="",
        grammar_triggers=[],
        preserved_tokens=[],
        chat_format="Content-only",
        reasoning_format="none",
        reasoning_in_content=False,
        **{"speculative.types": "none"},
    )
    return gs


def test_request_sends_ids_every_setting_and_neutral_extras():
    request = build_request([1, 2, 3], SETTINGS, ["\nUser:"])
    assert request["prompt"] == [1, 2, 3]
    assert request["n_predict"] == 1024
    assert request["samplers"] == ["top_k", "top_p", "min_p", "temperature"]
    assert request["stop"] == ["\nUser:"]
    assert request["cache_prompt"] is False
    assert request["return_tokens"] is True
    assert request["stream"] is True
    for key, value in NEUTRAL_REQUEST.items():
        assert request[key] == value
    assert request["mirostat"] == 0
    assert request["logit_bias"] == []


def test_request_needs_a_concrete_seed():
    with pytest.raises(BackendError):
        build_request([1], {**SETTINGS, "seed": "random"}, [])


def test_echo_check_passes_at_float32_precision():
    request = build_request([1], SETTINGS, [])
    checks = check_generation_settings(request, echoed(request))
    assert checks["settings_echo"]["ok"], checks
    assert checks["unaccounted_settings"]["ok"], checks


def test_echo_check_flags_a_changed_value():
    request = build_request([1], SETTINGS, [])
    gs = echoed(request)
    gs["top_p"] = 1.0  # e.g. clamped by the server
    gs["samplers"] = ["top_k", "temperature"]  # e.g. an unknown name skipped
    checks = check_generation_settings(request, gs)
    assert not checks["settings_echo"]["ok"]
    assert set(checks["settings_echo"]["mismatched"]) == {"top_p", "samplers"}


def test_echo_check_flags_a_new_unknown_sampling_key():
    request = build_request([1], SETTINGS, [])
    gs = {**echoed(request), "brand_new_sampler": 0.3}
    checks = check_generation_settings(request, gs)
    assert checks["unaccounted_settings"] == {"ok": False, "keys": ["brand_new_sampler"]}


def test_echo_check_flags_speculative_decoding_or_logit_bias():
    request = build_request([1], SETTINGS, [])
    gs = {**echoed(request), "speculative.types": "ngram", "logit_bias": [{"token": 5, "bias": -1}]}
    mismatched = check_generation_settings(request, gs)["settings_echo"]["mismatched"]
    assert set(mismatched) == {"speculative.types", "logit_bias"}


def test_sse_lines():
    assert parse_sse_line('data: {"content": "hi", "tokens": [5]}') == {
        "content": "hi",
        "tokens": [5],
    }
    assert parse_sse_line(": ping") is None
    assert parse_sse_line("") is None
    assert parse_sse_line("data: [DONE]") is None


def test_server_env_removes_flag_variables(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "0")
    monkeypatch.setenv("LLAMA_API_KEY", "x")
    monkeypatch.setenv("GGML_METAL_SOMETHING", "1")
    env, removed, kept = server_env()
    assert "LLAMA_ARG_CTX_SIZE" not in env
    assert {"LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_N_GPU_LAYERS", "LLAMA_API_KEY"} <= set(removed)
    assert kept["GGML_METAL_SOMETHING"] == "1"  # values are logged, not just names


def test_values_of_secret_looking_variables_are_never_logged(monkeypatch):
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "made-up-key-123")
    monkeypatch.setenv("GGML_SOME_TOKEN", "made-up-token")
    env, _, kept = server_env()
    assert env["LLAMA_CLOUD_API_KEY"] == "made-up-key-123"  # the server still gets it
    assert "made-up" not in str(kept)
    assert set(kept) >= {"LLAMA_CLOUD_API_KEY", "GGML_SOME_TOKEN"}  # the names are logged


COMMIT = "f477f795cb071baf5b8af6488378366a72034ac4"


def test_a_clean_matching_server_has_no_start_warnings():
    assert start_warnings("b1-f477f79", COMMIT, {"dirty": False}, {}) == []


def test_start_warnings_name_what_you_can_act_on():
    warnings = start_warnings("b1-0000000", COMMIT, {"dirty": True}, {"GGML_X": "1"})
    text = " ".join(warnings)
    assert "rebuild llama-server" in text
    assert "uncommitted changes" in text
    assert "GGML_X" in text


def test_config_files_are_found(monkeypatch, tmp_path):
    if __import__("os").name == "nt":
        monkeypatch.setenv("APPDATA", str(tmp_path))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not [p for p in llama_config_files() if tmp_path in p.parents]
    (tmp_path / "llama.cpp").mkdir()
    (tmp_path / "llama.cpp" / "config.ini").write_text("[*]\n", encoding="utf-8")
    assert tmp_path / "llama.cpp" / "config.ini" in llama_config_files()


STARTUP_GPU = """\
llama_model_load_from_file_impl: using device MTL0 (Apple M2 Max) (unknown id) - 25558 MiB free
print_info: EOG token             = 100257 '<|endoftext|>'
print_info: EOG token             = 100265 '<|im_end|>'
load_tensors: offloaded 31/31 layers to GPU
load_tensors:   CPU_Mapped model buffer size =    56.25 MiB
load_tensors:  MTL0_Mapped model buffer size =   256.50 MiB
"""
STARTUP_HIDDEN_GPU = """\
load_tensors: offloaded 31/31 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   312.75 MiB
"""


def test_gpu_placement_is_read_from_startup_output():
    placement = device_placement(STARTUP_GPU)
    assert placement["devices"] == [
        {"name": "MTL0", "description": "Apple M2 Max", "free_mib": 25558}
    ]
    assert {"buffer": "MTL0_Mapped", "mib": 256.5} in placement["model_buffers"]
    assert placement_problem({"n-gpu-layers": "all"}, placement) is None


@pytest.mark.parametrize("host_buffer", ["CUDA_Host", "Vulkan_Host"])
def test_pinned_host_buffers_count_as_cpu(host_buffer):
    # CUDA and Vulkan name pinned system-memory buffers "<backend>_Host": not on the GPU.
    startup = (
        "llama_model_load_from_file_impl: using device CUDA0 (NVIDIA RTX 4090) (0000:01:00.0)"
        " - 23000 MiB free\n"
        "load_tensors:   CPU_Mapped model buffer size =    56.25 MiB\n"
        f"load_tensors: {host_buffer:>12} model buffer size =   256.50 MiB\n"
    )
    problem = placement_problem({"n-gpu-layers": "all"}, device_placement(startup))
    assert problem is not None and "put the model on the CPU" in problem


def test_model_quietly_on_cpu_is_an_error_when_gpu_layers_are_asked_for():
    placement = device_placement(STARTUP_HIDDEN_GPU)
    problem = placement_problem({"n-gpu-layers": "all"}, placement)
    assert problem is not None and "put the model on the CPU" in problem
    assert placement_problem({"n-gpu-layers": 0}, placement) is None


class FakeStream:
    def __init__(self, lines):
        self.status_code = 200
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self.lines


class PropsFail:
    url = "http://127.0.0.1:1"
    headers = {"Authorization": "Bearer test-key"}

    def get_props(self):
        import httpx

        raise httpx.ConnectError("server gone")

    def check_props(self, props):
        return []


def fake_backend(monkeypatch, root, lines):
    import json

    from clarity_chat.backends import llama_cpp
    from clarity_chat.profiles import load_server_config

    backend = llama_cpp.LlamaCppBackend(load_server_config(root), root / "logs")
    monkeypatch.setattr(backend, "server", PropsFail())  # only what complete() uses
    encoded = [line if isinstance(line, str) else "data: " + json.dumps(line) for line in lines]
    monkeypatch.setattr(llama_cpp.httpx, "stream", lambda *a, **k: FakeStream(encoded))
    return backend


FINAL = {
    "content": "",
    "tokens": [],
    "stop": True,
    "stop_type": "eos",
    "tokens_evaluated": 2,
    "tokens_predicted": 1,
    "timings": {"cache_n": 0},
    "generation_settings": {},
}


def test_a_props_failure_after_the_reply_is_a_failed_check_not_a_lost_reply(monkeypatch, root):
    backend = fake_backend(monkeypatch, root, [{"content": "Hi", "tokens": [5]}, FINAL])
    result = backend.complete([1, 2], SETTINGS, [])
    assert result.text == "Hi"
    assert result.checks["server_props"]["ok"] is False
    assert "check_failed:server_props" in result.flags


def test_a_malformed_stream_chunk_is_a_backend_error_with_the_partial_reply(monkeypatch, root):
    backend = fake_backend(monkeypatch, root, [{"content": "Hi", "tokens": [5]}, "data: {oops"])
    with pytest.raises(BackendError, match="malformed") as caught:
        backend.complete([1, 2], SETTINGS, [])
    assert caught.value.partial is not None and caught.value.partial.text == "Hi"


def started_server(monkeypatch, root, startup, props_ok=True, on_change=None):
    """A LlamaServer.start() with every outside step faked: what it launched, and the record."""
    import json
    from pathlib import Path

    from clarity_chat.backends import llama_cpp
    from clarity_chat.profiles import load_server_config

    config = load_server_config(root)
    server = llama_cpp.LlamaServer(config, root / "model.gguf", root / "logs", on_change)
    launched = {}

    def launch(argv, env):
        launched.update(argv=argv, env=env)
        server.output = [startup]
        exited = type("Exited", (), {"pid": 4242, "poll": lambda self: 0})()
        monkeypatch.setattr(server, "proc", exited)
        return None

    monkeypatch.setattr(llama_cpp, "llama_config_files", lambda: [])
    monkeypatch.setattr(llama_cpp, "git_state", lambda d: {"commit": config.llama_cpp_commit})
    monkeypatch.setattr(llama_cpp, "server_binary", lambda d: Path("llama-server"))
    monkeypatch.setattr(server, "_launch", launch)
    props = {
        "model_path": str(root / "model.gguf"),
        "default_generation_settings": {"n_ctx": config.ctx_size},
        "build_info": "b1-" + config.llama_cpp_commit[:7],
    }
    monkeypatch.setattr(server, "get_props", lambda: props)
    monkeypatch.setattr(server, "check_props", lambda p: [] if props_ok else ["n_ctx differs"])
    record = server.start()
    return server, launched, json.dumps(record)


def test_graph_splits_are_read_from_startup_output():
    same = "llama_context: graph: nodes = 1131, splits = 2, input objects = 3, input tensors = 4"
    assert graph_splits(same) == {"prompt": 2, "reply": 2}
    differ = (
        "llama_context: graph (pp bs=512, tg bs=1): nodes = 1131 / 1100, splits = 2 / 3, "
        "input objects = 3, input tensors = 4"
    )
    assert graph_splits(differ) == {"prompt": 2, "reply": 3}
    assert graph_splits("no such line") is None


def test_end_of_reply_tokens_are_logged(monkeypatch, root):
    _, _, record = started_server(monkeypatch, root, STARTUP_GPU)
    assert '"eog_tokens": [{"id": 100257, "text": "<|endoftext|>"}, {"id": 100265' in record
    windows = STARTUP_GPU.replace("\n", "\r\n")
    assert [t["id"] for t in eog_tokens(windows)] == [100257, 100265]


def test_the_server_key_goes_in_the_environment_and_is_never_logged(monkeypatch, root):
    server, launched, record = started_server(monkeypatch, root, STARTUP_GPU)
    assert launched["env"]["LLAMA_API_KEY"] == server.api_key
    assert server.api_key not in " ".join(launched["argv"])
    assert server.api_key not in record
    assert '"api_key": "random per start; value not logged"' in record
    assert {"--no-slots", "--no-webui"} <= set(launched["argv"])
    assert server.headers == {"Authorization": f"Bearer {server.api_key}"}


def test_each_start_draws_a_new_key(monkeypatch, root):
    first, _, _ = started_server(monkeypatch, root, STARTUP_GPU)
    second, _, _ = started_server(monkeypatch, root, STARTUP_GPU)
    assert first.api_key != second.api_key


def test_the_key_suffix_line_is_redacted_in_the_output_and_the_server_log(root):
    import io

    from clarity_chat.backends.llama_cpp import LlamaServer
    from clarity_chat.profiles import load_server_config

    server = LlamaServer(load_server_config(root), root / "model.gguf", root / "logs")
    server.log_path.parent.mkdir(parents=True)
    lines = b"main: loading model\nsrv          init: api_keys: ****Wx9z\nmain: ready\n"
    server._drain(io.BytesIO(lines))
    for text in (server.output_text(), server.log_path.read_text(encoding="utf-8")):
        assert "Wx9z" not in text
        assert "api_keys: (redacted by Clarity)" in text
        assert "main: ready" in text


def test_the_context_size_goes_into_the_servers_flags_and_a_new_size_restarts(monkeypatch, root):
    from clarity_chat.backends import llama_cpp
    from clarity_chat.profiles import load_models, load_server_config

    sizes = []

    def start(server):
        sizes.append(server.config.flags["ctx-size"])
        server.proc = type("Running", (), {"poll": lambda self: None})()
        return {}

    monkeypatch.setattr(llama_cpp.LlamaServer, "start", start)
    monkeypatch.setattr(llama_cpp.LlamaServer, "stop", lambda server: None)
    backend = llama_cpp.LlamaCppBackend(load_server_config(root), root / "logs")
    model = load_models(root)["tiny"]
    backend.start(model, root / "m.gguf", 16384)
    assert backend.n_ctx == 16384 and backend.config.ctx_size == 8192  # server.toml untouched
    assert backend.start(model, root / "m.gguf", 16384) is None
    backend.start(model, root / "m.gguf", 4096)
    assert sizes == [16384, 4096]


def test_a_size_above_the_training_context_names_the_limit(monkeypatch, root):
    from clarity_chat.profiles import ConfigError

    warning = (
        "srv    load_model: the slot context (65536) exceeds the training context of the "
        "model (32768) - capping\n"
    )
    with pytest.raises(ConfigError, match=r"trained on \(32768 tokens\).*32768 or less"):
        started_server(monkeypatch, root, STARTUP_GPU + warning, props_ok=False)


def test_a_cancel_is_checked_between_stream_lines(monkeypatch, root):
    import threading

    cancel = threading.Event()
    cancel.set()  # set while llama-server is still reading the prompt: only pings so far
    backend = fake_backend(monkeypatch, root, [": ping", {"content": "Hi", "tokens": [5]}, FINAL])
    result = backend.complete([1, 2], SETTINGS, [], cancel=cancel)
    assert (result.stop_type, result.cancelled_during, result.text) == ("cancelled", "prompt", "")

    cancel = threading.Event()
    lines = [{"content": "Hi", "tokens": [5]}, {"content": "!", "tokens": [6]}, FINAL]
    backend = fake_backend(monkeypatch, root, lines)
    result = backend.complete(
        [1, 2], SETTINGS, [], on_text=lambda text: cancel.set(), cancel=cancel
    )
    assert (result.stop_type, result.cancelled_during, result.text) == ("cancelled", "reply", "Hi")
    assert result.flags == ["cancelled"]


# OLMo 3 7B at ctx-size 32768, flash attention off, on an M2 Max (measured: M2).
STARTUP_OLMO_32K = """\
llama_prepare_model_devices: using device MTL0 (Apple M2 Max) (unknown id) - 25558 MiB free
load_tensors:   CPU_Mapped model buffer size =   783.42 MiB
load_tensors:  MTL0_Mapped model buffer size = 13137.44 MiB
llama_kv_cache:       MTL0 KV buffer size =  4096.00 MiB
llama_kv_cache:       MTL0 KV buffer size =  1728.00 MiB
sched_reserve:       MTL0 compute buffer size =  2193.01 MiB
sched_reserve:        CPU compute buffer size =   137.02 MiB
"""


def test_memory_use_sums_each_gpus_buffers():
    from clarity_chat.backends.llama_cpp import memory_use

    memory = memory_use(STARTUP_OLMO_32K, device_placement(STARTUP_OLMO_32K))
    assert memory == [
        {
            "device": "MTL0",
            "limit_mib": 25558,
            "model_mib": 13137.44,
            "kv_mib": 5824.0,  # the full-attention layers' cache grows with ctx; the SWA one not
            "compute_mib": 2193.01,
            "total_mib": 21154.45,
        }
    ]


def test_more_than_the_gpu_had_free_is_refused_on_metal_only():
    from clarity_chat.backends.llama_cpp import memory_problem

    over = {"device": "MTL0", "limit_mib": 20000, "model_mib": 13137.44}
    over |= {"kv_mib": 5824.0, "compute_mib": 2193.01, "total_mib": 21154.45}
    problem = memory_problem(32768, [over])
    assert problem is not None
    assert "At ctx_size 32768, the model needs 21,154 MiB" in problem
    assert "recommended limit for this GPU, 20,000 MiB. Lower ctx_size" in problem
    assert memory_problem(32768, [{**over, "device": "CUDA0"}]) is None  # logged only
    assert memory_problem(32768, [{**over, "limit_mib": 25558}]) is None
    too_big = {**over, "limit_mib": 12000}  # the model alone is over the limit
    assert "no ctx_size fits" in (memory_problem(8192, [too_big]) or "")


def test_the_memory_figures_are_logged_and_an_overflow_stops_the_server(monkeypatch, root):
    from clarity_chat.profiles import ConfigError

    _, _, record = started_server(monkeypatch, root, STARTUP_OLMO_32K)
    assert '"memory": [{"device": "MTL0", "limit_mib": 25558' in record
    tight = STARTUP_OLMO_32K.replace("25558 MiB free", "20000 MiB free")
    with pytest.raises(ConfigError, match="recommended limit for this GPU, 20,000 MiB"):
        started_server(monkeypatch, root, tight)


def test_the_server_is_recorded_when_launched_and_cleared_once_it_exits(root):
    import dataclasses
    import sys

    from clarity_chat.backends.llama_cpp import LlamaServer
    from clarity_chat.profiles import load_server_config

    config = dataclasses.replace(load_server_config(root), startup_timeout_s=0.5)
    heard = []
    server = LlamaServer(config, root / "model.gguf", root / "logs", heard.append)
    server.log_path.parent.mkdir(parents=True)
    server.port = 1  # nothing answers /health there: the launch times out and stops it
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    assert server._launch(sleeper, dict(__import__("os").environ)) is not None
    assert heard[0] == {
        "port": 1,
        "pid": heard[0]["pid"],
        "api_key": server.api_key,
        "model_path": str(root / "model.gguf"),
    }
    assert heard[-1] is None and server.proc is not None and server.proc.poll() is not None
