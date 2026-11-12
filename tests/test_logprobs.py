"""Token probabilities (the `logprobs` setting): Transformers only, refused on llama.cpp."""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest

from clarity_chat.commands import Commands
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from clarity_chat.session import Session
from tests.conftest import TEXT_PROFILE
from tests.fakes import FakeBackend

VOCAB = 258  # the tiny tokenizer: 256 bytes, <s> and </s>


def test_logprobs_on_a_llama_cpp_profile_is_refused_at_load(root):
    (root / "profiles" / "p.toml").write_text(
        TEXT_PROFILE.replace("[settings]", "[settings]\nlogprobs = 5"), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="Transformers backend for now"):
        load_profile(root, "p", load_defaults(root))
    hf = TEXT_PROFILE.replace('backend = "llama.cpp"', 'backend = "transformers"')
    (root / "profiles" / "hf.toml").write_text(
        hf.replace("[settings]", "[settings]\nlogprobs = 5"), encoding="utf-8"
    )
    assert load_profile(root, "hf", load_defaults(root)).settings["logprobs"].value == 5


@pytest.mark.parametrize("value", ["-1", "21", "2.5"])
def test_logprobs_is_an_integer_from_0_to_20(root, value):
    (root / "profiles" / "p.toml").write_text(
        TEXT_PROFILE.replace("[settings]", f"[settings]\nlogprobs = {value}"), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="from 0 \\(off\\) to 20"):
        load_profile(root, "p", load_defaults(root))


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    return c


def test_set_logprobs_on_a_llama_cpp_profile_is_refused(commands):
    with pytest.raises(ConfigError, match="Transformers backend for now"):
        commands.run("set", "logprobs 5", lambda e: None)
    commands.run("set", "logprobs 0", lambda e: None)  # off is fine everywhere


def test_probs_lists_the_least_likely_tokens_in_reply_order(commands):
    commands.run("say", "Hi", lambda e: None)
    conversation = commands.session.conversation
    assert conversation is not None and conversation.last_call is not None
    assert "/set logprobs 5" in commands.run("probs", "", lambda e: None)["unavailable"]
    tokenizer = commands.session.tokenizer
    assert tokenizer is not None
    chosen = tokenizer.encode("abcdefghijkl")
    (z,) = tokenizer.encode("z")
    tokens = [
        {"id": t, "logprob": -0.1 * (i % 7), "top": [[t, -0.1 * (i % 7)], [z, -3.0]]}
        for i, t in enumerate(chosen)
    ]
    view = "raw logits, before any sampler"
    conversation.last_call["logprobs"] = {"view": view, "n": 2, "tokens": tokens}
    result = commands.run("probs", "", lambda e: None)
    least = result["least_likely"]
    assert len(least) == 10
    assert [t["index"] for t in least] == sorted(t["index"] for t in least)
    assert 0 not in [t["index"] for t in least] and 7 not in [t["index"] for t in least]
    assert least[0]["top"][1] == {"id": z, "text": "z", "logprob": -3.0}
    assert result["logprobs"]["tokens"] == tokens  # the full record, for scripts

    from clarity_chat.cli import format_probs

    text = format_probs(result)
    assert text.startswith(f"12 tokens in {result['call_id']} ({view}).")
    assert f"#6 'g' {math.exp(-0.6):.1%}   model's likeliest others: 'z' 5.0%" in text


def test_unavailable_probabilities_say_why(commands):
    commands.run("say", "Hi", lambda e: None)
    conversation = commands.session.conversation
    assert conversation is not None and conversation.last_call is not None
    conversation.last_call["logprobs"] = {"unavailable": "cancelled"}
    result = commands.run("probs", "", lambda e: None)
    assert result["unavailable"] == "cancelled"

    from clarity_chat.cli import format_probs

    assert format_probs(result) == "No token probabilities for this reply: cancelled."


# --- the Transformers backend, with a stand-in model ----------------------------------------


def test_reduce_logits_gives_each_tokens_log_probability_and_the_top_n():
    torch = pytest.importorskip("torch")
    from clarity_chat.backends.transformers_hf import reduce_logits

    gen = torch.Generator().manual_seed(0)
    logits = tuple(torch.randn(1, VOCAB, generator=gen) for _ in range(3))
    ids = [5, 9, 257]
    reduced = reduce_logits(logits, ids, 3)
    assert reduced["view"] == "raw logits, before any sampler" and reduced["n"] == 3
    for step, token, row in zip(logits, ids, reduced["tokens"], strict=True):
        expected = torch.log_softmax(step[0], dim=-1)
        assert row["id"] == token
        assert row["logprob"] == pytest.approx(expected[token].item(), abs=1e-6)
        top = torch.topk(expected, 3)
        assert [i for i, _ in row["top"]] == top.indices.tolist()
    assert "unavailable" in reduce_logits(logits, ids[:2], 3)


class StandInModel:
    """Streams fixed tokens through generate's streamer and returns one logits row per step,
    as Transformers does with output_logits and return_dict_in_generate."""

    def __init__(self, torch, steps):
        self.torch, self.steps = torch, steps

    def _prepare_generation_config(self, config):
        return config, {}

    def generate(self, inputs, attention_mask, generation_config, tokenizer, streamer):
        streamer.put(inputs)
        rows = []
        for token in self.steps:
            row = self.torch.zeros(1, VOCAB)
            row[0, token] = 4.0
            rows.append(row)
            streamer.put(self.torch.tensor([token]))
        return SimpleNamespace(logits=tuple(rows) if generation_config.output_logits else None)


def stand_in_backend(torch, tiny_tokenizer):
    from clarity_chat.backends.transformers_hf import TransformersBackend

    backend = TransformersBackend("cpu", 512)
    backend.model = StandInModel(torch, [5, 9, 257])
    backend.tokenizer = tiny_tokenizer.tok
    backend.eos_ids, backend.eos_from = [257], "generation_config.json"
    return backend


SETTINGS = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 1,
    "max_tokens": 8,
    "samplers": ["temperature"],
}


def test_a_call_with_logprobs_asks_for_logits_and_records_them(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    backend = stand_in_backend(torch, tiny_tokenizer)
    result = backend.complete([1, 2], {**SETTINGS, "logprobs": 2}, [])
    assert result.token_ids == [5, 9, 257] and result.stop_type == "eos"
    config = result.request["generation_config"]
    assert config["output_logits"] and config["return_dict_in_generate"]
    assert result.request["logprobs"] == 2
    assert result.logprobs is not None
    assert [t["id"] for t in result.logprobs["tokens"]] == [5, 9, 257]
    assert [t["top"][0][0] for t in result.logprobs["tokens"]] == [5, 9, 257]


def test_a_call_without_logprobs_asks_for_no_logits(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    backend = stand_in_backend(torch, tiny_tokenizer)
    result = backend.complete([1, 2], {**SETTINGS, "logprobs": 0}, [])
    assert not result.request["generation_config"]["output_logits"]
    assert not result.request["generation_config"]["return_dict_in_generate"]
    assert result.logprobs is None


def test_a_cancelled_call_records_its_probabilities_as_unavailable(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    backend = stand_in_backend(torch, tiny_tokenizer)
    cancel = threading.Event()
    cancel.set()
    result = backend.complete([1, 2], {**SETTINGS, "logprobs": 2}, [], cancel=cancel)
    assert result.stop_type == "cancelled"
    assert result.logprobs == {"unavailable": "cancelled"}


def test_a_replay_takes_logprobs_from_the_logged_request(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    backend = stand_in_backend(torch, tiny_tokenizer)
    first = backend.complete([1, 2], {**SETTINGS, "logprobs": 2}, [])
    again = backend.complete([1, 2], {**SETTINGS, "logprobs": 0}, [], request=first.request)
    assert again.logprobs == first.logprobs
    old = {**first.request}
    del old["logprobs"]  # a call logged before v0.2.1
    old["generation_config"] = {**old["generation_config"], "output_logits": False}
    assert backend.complete([1, 2], SETTINGS, [], request=old).logprobs is None


def test_a_replay_records_the_probabilities_it_gets(commands):
    """B2: a replay of a call with probabilities stores them in the replay record."""
    backend = commands.session.backend
    backend.logprobs = {"view": "raw logits, before any sampler", "n": 1, "tokens": []}
    call = commands.run("say", "Hi", lambda e: None)["call"]
    assert call["logprobs"] == backend.logprobs
    replayed = commands.session.replay_current()
    assert replayed["logprobs"] == backend.logprobs


def test_probs_shows_the_reply_on_screen_after_a_failed_message(commands):
    """A failed /say leaves the earlier reply at head; /probs must describe that one, not
    the failed call (code review, verifier A)."""
    from clarity_chat.backends import BackendError

    backend = commands.session.backend
    backend.logprobs = {"view": "raw logits, before any sampler", "n": 1, "tokens": []}
    first = commands.run("say", "Hi", lambda e: None)["call"]
    backend.fail_with = "server gone"
    with pytest.raises(BackendError):
        commands.run("say", "Again", lambda e: None)
    result = commands.run("probs", "", lambda e: None)
    assert result["call_id"] == first["call_id"]
    assert "unavailable" not in result


def test_a_transformers_replay_uses_the_logged_end_tokens(tiny_tokenizer):
    """generate stops on the logged request's eos_token_id; the replay's stop type and
    backend_settings must say the same, even if today's model files list others."""
    torch = pytest.importorskip("torch")
    backend = stand_in_backend(torch, tiny_tokenizer)
    first = backend.complete([1, 2], {**SETTINGS, "logprobs": 0}, [])
    assert first.stop_type == "eos"
    backend.eos_ids, backend.eos_from = [99], "generation_config.json"  # a later revision
    again = backend.complete([1, 2], SETTINGS, [], request=first.request)
    assert again.stop_type == "eos"
    assert again.backend_settings["eos_token_id"]["value"] == [257]
