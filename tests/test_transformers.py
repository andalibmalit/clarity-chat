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
