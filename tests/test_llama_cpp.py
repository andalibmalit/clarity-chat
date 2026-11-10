"""The llama.cpp backend without a server: request building, checks, stream parsing, env."""

from __future__ import annotations

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.backends.llama_cpp import (
    NEUTRAL_REQUEST,
    build_request,
    check_generation_settings,
    device_placement,
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


def started_server(monkeypatch, root, startup):
    """A LlamaServer.start() with every outside step faked: what it launched, and the record."""
    import json
    from pathlib import Path

    from clarity_chat.backends import llama_cpp
    from clarity_chat.profiles import load_server_config

    config = load_server_config(root)
    server = llama_cpp.LlamaServer(config, root / "model.gguf", root / "logs")
    launched = {}

    def launch(argv, env):
        launched.update(argv=argv, env=env)
        server.output = [startup]
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
    monkeypatch.setattr(server, "check_props", lambda p: [])
    record = server.start()
    return server, launched, json.dumps(record)


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
