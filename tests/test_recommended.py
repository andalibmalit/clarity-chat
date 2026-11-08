"""The model's own recommended settings: opt-in, explicit, logged with origin "model"."""

from __future__ import annotations

from clarity_chat.models import map_generation_config
from clarity_chat.session import Session
from tests.fakes import FakeBackend

SOURCE = {"file": "generation_config.json", "repo": "a/b", "revision": "c" * 40}


def test_olmo_style_generation_config():
    config = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 32768,
        "eos_token_id": [100265, 100257],
        "transformers_version": "4.57.1",
    }
    mapped = map_generation_config(config, SOURCE)
    assert mapped["values"] == {"temperature": 0.6, "top_p": 0.95}
    # nothing dropped silently, but max_new_tokens never becomes max_tokens
    assert mapped["not_applied"] == {"max_new_tokens": 32768, "eos_token_id": [100265, 100257]}
    assert "top_k" not in mapped["values"]  # absent stays neutral, not Transformers' 50


def test_repetition_penalty_maps_to_repeat_penalty():
    assert map_generation_config({"repetition_penalty": 1.1}, SOURCE)["values"] == {
        "repeat_penalty": 1.1
    }


def test_do_sample_false_means_greedy():
    mapped = map_generation_config({"do_sample": False, "temperature": 0.7}, SOURCE)
    assert mapped["values"]["temperature"] == 0.0
    assert any("greedy" in note for note in mapped["notes"])


def test_out_of_range_value_is_listed_not_applied():
    mapped = map_generation_config({"top_p": 1.5}, SOURCE)
    assert mapped["values"] == {}
    assert mapped["not_applied"] == {"top_p": 1.5}


def write_profile(root, use_recommended, settings="max_tokens = 16\nseed = 1\n"):
    (root / "profiles" / "rec.toml").write_text(
        'backend = "llama.cpp"\nmodel = "tiny"\nmode = "text"\nuser_name = "U"\n'
        f'assistant_name = "A"\nuse_recommended = {str(use_recommended).lower()}\n'
        f"[settings]\n{settings}",
        encoding="utf-8",
    )


def start(root, tokenizer):
    session = Session(root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tokenizer)
    session.new_conversation("rec")
    return session


def test_use_recommended_fills_unset_settings_with_origin_model(fetched_root, tiny_tokenizer):
    write_profile(fetched_root, True, "top_p = 0.5\nmax_tokens = 16\n")
    session = start(fetched_root, tiny_tokenizer)
    settings = session.conversation.profile.settings
    assert settings["temperature"].value == 0.7
    assert settings["temperature"].origin == "model"
    assert settings["top_p"].value == 0.5  # the profile wins
    assert settings["top_p"].origin == "profile"
    assert settings["top_k"].origin == "default"
    record = session.say("hi").record
    assert record["settings"]["temperature"] == {"value": 0.7, "origin": "model"}
    assert record["recommended"]["source"]["file"] == "generation_config.json"


def test_without_use_recommended_the_model_values_are_not_used(fetched_root, tiny_tokenizer):
    write_profile(fetched_root, False)
    session = start(fetched_root, tiny_tokenizer)
    assert session.conversation.profile.settings["temperature"].value == 1.0
    assert session.say("hi").record["recommended"] is None
