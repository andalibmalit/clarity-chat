"""The Transformers backend's settings mapping and checks (no torch needed)."""

from __future__ import annotations

import pytest

from clarity_chat.backends.transformers_hf import (
    NEUTRAL_CONFIG,
    check_config,
    cut_at_stop,
    generation_values,
    held_back,
    not_applied,
    torch_env,
)
from clarity_chat.models import source_weights
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from tests.conftest import TEXT_PROFILE

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
    "max_tokens": 16,
    "samplers": ["top_k", "top_p", "min_p", "temperature"],
    "logprobs": 0,
}


def test_every_value_is_explicit_and_sampling_maps_through():
    values = generation_values(SETTINGS, ["\nUser:"], [2, 3])
    assert values["do_sample"] is True
    assert values["temperature"] == 0.6 and values["top_p"] == 0.95 and values["top_k"] == 0
    assert values["min_p"] is None  # 0 means off: no processor
    assert values["max_new_tokens"] == 16
    assert values["eos_token_id"] == [2, 3] and values["pad_token_id"] == 2
    assert values["stop_strings"] == ["\nUser:"]
    assert all(values[k] == v for k, v in NEUTRAL_CONFIG.items())


def test_temperature_zero_is_greedy_with_the_warpers_off():
    values = generation_values({**SETTINGS, "temperature": 0.0, "top_k": 40}, [], [2])
    assert values["do_sample"] is False
    assert (values["temperature"], values["top_k"], values["top_p"]) == (1.0, 0, 1.0)
    assert values["stop_strings"] is None


def test_a_different_sampler_order_and_missing_penalties_are_not_applied():
    names = {entry["name"]: entry for entry in not_applied(SETTINGS)}
    assert "['temperature', 'top_k', 'top_p', 'min_p']" in names["samplers"]["reason"]
    assert set(names) == {"samplers", "presence_penalty", "frequency_penalty", "repeat_last_n"}
    in_order = {**SETTINGS, "samplers": ["temperature", "top_k", "top_p", "min_p"]}
    assert "samplers" not in {e["name"] for e in not_applied(in_order)}


def test_the_config_check_catches_changed_and_unaccounted_fields():
    sent = {"top_k": 0, "temperature": 0.6}
    assert check_config({"top_k": 0, "temperature": 0.6, "max_length": 20}, sent)["ok"]
    changed = check_config({"top_k": 50, "temperature": 0.6}, sent)
    assert not changed["ok"] and changed["mismatched"]["top_k"] == {"sent": 0, "effective": 50}
    extra = check_config({"top_k": 0, "temperature": 0.6, "some_new_sampler": 0.5}, sent)
    assert not extra["ok"] and extra["unaccounted"] == {"some_new_sampler": 0.5}


def test_text_is_cut_at_the_earliest_stop_string():
    assert cut_at_stop(" Hi.\nUser: more", ["\nUser:"]) == (" Hi.", "\nUser:")
    assert cut_at_stop(" Hi.", ["\nUser:"]) == (" Hi.", "")
    assert cut_at_stop("a|b#c", ["#", "|"]) == ("a", "|")


def test_streaming_holds_back_a_possible_stop_string_start():
    assert held_back(" Hi.\nUs", ["\nUser:"]) == 3
    assert held_back(" Hi.", ["\nUser:"]) == 0
    assert held_back("x", []) == 0


def test_torch_variables_are_logged_with_secrets_redacted(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    monkeypatch.setenv("TORCH_SOME_TOKEN", "made-up")
    env = torch_env()
    assert env["PYTORCH_MPS_FAST_MATH"] == "1"
    assert "made-up" not in str(env)


# --- profiles and weights -----------------------------------------------------------------


def hf_profile(root, extra):
    text = TEXT_PROFILE.replace('backend = "llama.cpp"', 'backend = "transformers"')
    (root / "profiles" / "hf.toml").write_text(
        text.replace("[settings]", extra + "\n[settings]"), encoding="utf-8"
    )
    return load_profile(root, "hf", load_defaults(root))


def test_transformers_profiles_default_to_mps(root):
    assert hf_profile(root, "").device == "mps"
    assert hf_profile(root, 'device = "cpu"').device == "cpu"
    with pytest.raises(ConfigError, match="device must be one of"):
        hf_profile(root, 'device = "tpu"')


def test_device_is_only_for_transformers(root):
    (root / "profiles" / "x.toml").write_text(
        TEXT_PROFILE.replace("[settings]", 'device = "cpu"\n[settings]'), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="device applies only"):
        load_profile(root, "x", load_defaults(root))


def test_source_weights_need_the_convert_route(tmp_path):
    with pytest.raises(ConfigError, match="ready-made GGUF"):
        source_weights(tmp_path, {"name": "m", "route": "gguf"})
    (tmp_path / "src").mkdir()
    with pytest.raises(ConfigError, match="no config.json"):
        source_weights(tmp_path, {"name": "m", "route": "convert", "source_dir": "src"})
    (tmp_path / "src" / "config.json").write_text("{}", encoding="utf-8")
    assert source_weights(tmp_path, {"name": "m", "route": "convert", "source_dir": "src"}) == (
        tmp_path / "src"
    )


def test_only_differences_that_can_change_the_reply_count():
    neutral = {e["name"]: e["affects_output"] for e in not_applied(SETTINGS)}
    # top_k 0 and min_p 0 are off, so only top_p and temperature act: the order matters.
    assert neutral == {
        "samplers": True,
        "presence_penalty": False,
        "frequency_penalty": False,
        "repeat_last_n": False,
    }
    greedy = {e["name"]: e["affects_output"] for e in not_applied({**SETTINGS, "temperature": 0})}
    assert greedy["samplers"] is False
    penalised = {**SETTINGS, "presence_penalty": 0.5, "repeat_penalty": 1.1}
    flags = {e["name"]: e["affects_output"] for e in not_applied(penalised)}
    assert flags["presence_penalty"] and flags["repeat_last_n"]


def test_every_call_record_names_the_attention_implementation():
    # describe() goes into every call record. The value is fixed by Clarity, not read back
    # from the library, so it's there before a model loads (and with no torch installed).
    from clarity_chat.backends.transformers_hf import ATTN_IMPLEMENTATION, TransformersBackend

    described = TransformersBackend("cpu", 512).describe()
    assert described["attn_implementation"] == ATTN_IMPLEMENTATION == "sdpa"


@pytest.mark.parametrize("generation_config", ['{"bos_token_id": 0}', None])
def test_end_tokens_come_from_the_tokenizer_when_generation_config_has_none(
    tmp_path, generation_config
):
    """OLMo 3 base's generation_config.json has no eos_token_id; Qwen3.5-9B ships none."""
    from clarity_chat.backends.transformers_hf import end_tokens
    from tests.conftest import build_tokenizer

    build_tokenizer(tmp_path, None, False)
    if generation_config is not None:
        (tmp_path / "generation_config.json").write_text(generation_config, encoding="utf-8")
    ids, origin = end_tokens(tmp_path)
    assert ids == [257]  # </s>
    assert origin == "tokenizer (generation_config.json has none)"


def test_end_tokens_come_from_generation_config_when_it_lists_them(tmp_path):
    from clarity_chat.backends.transformers_hf import end_tokens
    from tests.conftest import build_tokenizer

    build_tokenizer(tmp_path, None, False)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [5, 6]}', encoding="utf-8")
    assert end_tokens(tmp_path) == ([5, 6], "generation_config.json")


def test_no_end_token_anywhere_is_an_error(tmp_path):
    import json

    from clarity_chat.backends.transformers_hf import end_tokens
    from tests.conftest import build_tokenizer

    build_tokenizer(tmp_path, None, False)
    path = tmp_path / "tokenizer_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.pop("eos_token", None)
    path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "special_tokens_map.json").unlink(missing_ok=True)
    with pytest.raises(ConfigError, match="no end token"):
        end_tokens(tmp_path)


def test_mps_with_the_cpu_fallback_switched_on_is_refused(monkeypatch, tmp_path):
    """PYTORCH_ENABLE_MPS_FALLBACK would run what MPS lacks on the CPU without notice."""
    torch = pytest.importorskip("torch")
    from clarity_chat.backends.transformers_hf import TransformersBackend
    from clarity_chat.profiles import load_models
    from tests.conftest import REPO

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)  # on any machine
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model = next(iter(load_models(REPO).values()))
    with pytest.raises(ConfigError, match="PYTORCH_ENABLE_MPS_FALLBACK"):
        TransformersBackend("mps", 512).start(model, tmp_path, 512)


def custom_code_folder(folder):
    """A model folder whose config and tokenizer name their own Python code, which records
    that it ran."""
    import json

    folder.mkdir()
    auto_map = {
        "AutoConfig": "custom.Cfg",
        "AutoModelForCausalLM": "custom.Model",
        "AutoTokenizer": ["custom.Tok", None],
    }
    (folder / "config.json").write_text(
        json.dumps({"model_type": "made_up_xyz", "auto_map": auto_map}), encoding="utf-8"
    )
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": ["custom.Tok", None]}}), encoding="utf-8"
    )
    (folder / "custom.py").write_text(
        "open(__file__ + '.ran', 'w').close()\n"
        "from transformers import PretrainedConfig, PreTrainedTokenizer\n"
        "class Cfg(PretrainedConfig):\n    model_type = 'made_up_xyz'\n"
        "class Tok(PreTrainedTokenizer):\n    pass\n"
        "class Model:\n    pass\n",
        encoding="utf-8",
    )
    return folder


def test_a_models_own_code_is_refused_never_asked_about(tmp_path, monkeypatch):
    """Transformers asks on stdin whether to run a folder's code; in the terminal a typed "y"
    would run it. Clarity runs only its own code and locked libraries (design.md)."""
    from clarity_chat.backends.transformers_hf import end_tokens
    from clarity_chat.render import Tokenizer

    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")
    folder = custom_code_folder(tmp_path / "m")
    for load in (lambda: Tokenizer.from_dir(folder), lambda: end_tokens(folder)):
        with pytest.raises(ConfigError, match="its own Python code"):
            load()
    assert asked == [] and not (folder / "custom.py.ran").exists()


def test_the_model_loader_refuses_a_models_own_code(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    from clarity_chat.render import from_pretrained

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    folder = custom_code_folder(tmp_path / "m")
    with pytest.raises(ConfigError, match="its own Python code"):
        from_pretrained(AutoModelForCausalLM, folder)
    assert not (folder / "custom.py.ran").exists()


def test_each_sampling_setting_reaches_its_own_generation_field():
    """As for llama.cpp's request: no swapped field and no fixed value, for any values."""
    from hypothesis import given
    from hypothesis import settings as hypothesis_settings
    from hypothesis import strategies as st

    from tests.test_llama_cpp import any_settings

    @hypothesis_settings(max_examples=200, deadline=None)
    @given(any_settings(), st.floats(0.01, 2, allow_nan=False), st.floats(0.01, 1))
    def check(settings, temperature, min_p):
        settings = {**settings, "temperature": temperature, "min_p": min_p, "logprobs": 0}
        values = generation_values(settings, [], [2])
        assert values["do_sample"] is True
        for name, field in [
            ("temperature", "temperature"),
            ("top_k", "top_k"),
            ("top_p", "top_p"),
            ("min_p", "min_p"),
            ("repeat_penalty", "repetition_penalty"),
            ("max_tokens", "max_new_tokens"),
        ]:
            assert values[field] == settings[name], (name, field)

    check()


# --- what a load leaves out, on tiny models saved in bf16 -----------------------------------


def saved(folder, model, tokenizer, drop=(), add=()):
    """`model` saved in `folder` with `tokenizer`, its weights file without the tensors named
    in `drop` and with a made-up tensor for each name in `add`."""
    import torch
    from safetensors.torch import load_file, save_file

    model.to(torch.bfloat16).save_pretrained(folder)
    tokenizer.tok.save_pretrained(folder)
    file = folder / "model.safetensors"
    tensors = {k: v for k, v in load_file(file).items() if k not in drop}
    assert len(tensors) == len(load_file(file)) - len(drop)  # each name in drop was there
    tensors |= {name: torch.zeros(2, dtype=torch.bfloat16) for name in add}
    save_file(tensors, file, metadata={"format": "pt"})
    return folder


def tiny_llama():
    torch = pytest.importorskip("torch")
    import transformers

    config = transformers.LlamaConfig(
        vocab_size=258,
        eos_token_id=257,
        bos_token_id=256,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config)


def started(folder):
    from clarity_chat.backends.transformers_hf import TransformersBackend
    from clarity_chat.profiles import ModelEntry

    model = ModelEntry("tiny", "safetensors", "test/tiny", "0" * 40, "t/t", "0" * 40)
    backend = TransformersBackend("cpu", 64)
    try:
        return backend, backend.start(model, folder, 64)
    except BaseException:
        backend.close()
        raise


def test_a_checkpoint_without_a_weight_the_model_needs_is_refused(tmp_path, chat_tokenizer):
    """Transformers fills a weight the checkpoint lacks at random and loads on: a reply from
    it isn't the model's, and its replay matches only by the seed."""
    gone = ["model.layers.1.mlp.down_proj.weight", "model.norm.weight"]
    folder = saved(tmp_path / "m", tiny_llama(), chat_tokenizer, drop=gone)
    with pytest.raises(ConfigError) as refused:
        started(folder)
    message = str(refused.value)
    assert message.startswith("tiny's weights lack 2 tensors the model needs"), message
    assert all(name in message for name in gone), message


def test_a_full_checkpoint_loads_with_nothing_left_unloaded(tmp_path, chat_tokenizer):
    backend, record = started(saved(tmp_path / "m", tiny_llama(), chat_tokenizer))
    backend.close()
    assert record is not None and record.get("weights_not_loaded") == []


def test_a_tensor_the_model_doesnt_use_is_logged_not_refused(tmp_path, chat_tokenizer):
    """Older checkpoints can carry harmless extras, which Transformers leaves unloaded."""
    folder = saved(tmp_path / "m", tiny_llama(), chat_tokenizer, add=["made_up.weight"])
    backend, record = started(folder)
    backend.close()
    assert record is not None and record.get("weights_not_loaded") == ["made_up.weight"]


def tiny_gemma4_with_towers():
    """Gemma 4 as Google ships it: the text model of tests/test_steering.py's gemma_config,
    with an audio and a vision tower. Gemma 4's audio tower has no `sdpa` attention."""
    torch = pytest.importorskip("torch")
    import transformers

    from tests.test_steering import gemma_config

    text = gemma_config(vocab_size=258).text_config
    text.eos_token_id, text.bos_token_id, text.pad_token_id = 257, 256, 257
    audio = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=32)
    vision = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )
    config = transformers.Gemma4Config(
        text_config=text.to_dict(), vision_config=vision, audio_config=audio
    )
    torch.manual_seed(0)
    return transformers.Gemma4ForConditionalGeneration(config)


TOWERS = ["model.audio_tower", "model.embed_audio", "model.embed_vision", "model.vision_tower"]


def test_gemma4_loads_its_text_model_only_and_logs_the_towers_left_out(tmp_path, chat_tokenizer):
    """Clarity sends text only, and the audio tower can't load with sdpa attention."""
    backend, record = started(saved(tmp_path / "m", tiny_gemma4_with_towers(), chat_tokenizer))
    try:
        assert record is not None and record.get("weights_not_loaded") == TOWERS
        assert backend.model.model.audio_tower is None
        assert backend.model.model.vision_tower is None
        assert record["attn_implementation"] == "sdpa"
    finally:
        backend.close()


def test_gemma4_refuses_a_tensor_outside_its_towers_it_doesnt_use(tmp_path, chat_tokenizer):
    """The towers are the only tensors left out on purpose: another means a checkpoint that
    doesn't match the model Transformers built."""
    extra = "model.language_model.made_up.weight"
    folder = saved(tmp_path / "m", tiny_gemma4_with_towers(), chat_tokenizer, add=[extra])
    with pytest.raises(ConfigError) as refused:
        started(folder)
    message = str(refused.value)
    assert message.startswith("tiny's weights have 1 tensor the model doesn't use"), message
    assert extra in message, message


GEMMA_LINE = "Gemma 4's audio and vision parts aren't loaded: Clarity runs its text model only."


def test_gemma4s_start_says_its_towers_arent_loaded(tmp_path, chat_tokenizer):
    """On screen and in the log, not only in weights_not_loaded."""
    backend, record = started(saved(tmp_path / "m", tiny_gemma4_with_towers(), chat_tokenizer))
    backend.close()
    assert record is not None and GEMMA_LINE in record["warnings"], record["warnings"]
