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
    assert "Turn on logprobs" in commands.run("probs", "", lambda e: None)["unavailable"]
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
    assert result["unavailable"] == "the reply was stopped, so none were recorded"

    from clarity_chat.cli import format_probs

    assert format_probs(result) == (
        "No token probabilities for this reply: the reply was stopped, so none were recorded."
    )


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


def fake_logprobs(tokenizer, ids):
    (z,) = tokenizer.encode("z")
    return {
        "view": "raw logits, before any sampler",
        "n": 2,
        "tokens": [{"id": t, "logprob": -0.5, "top": [[t, -0.5], [z, -3.0]]} for t in ids],
    }


def test_probs_gives_every_tokens_text_and_place_in_the_reply(commands):
    """The Probabilities view shades each token, so each has its text and offsets."""
    backend, tokenizer = commands.session.backend, commands.session.tokenizer
    backend.replies = [" Café, olé!"]  # é is two bytes: two tokens for one character
    ids = tokenizer.encode(" Café, olé!") + [tokenizer.tok.eos_token_id]
    backend.logprobs = fake_logprobs(tokenizer, ids)
    call = commands.run("say", "Hi", lambda e: None)["call"]
    tokens = commands.run("probs", "", lambda e: None).get("tokens", [])
    assert [t["id"] for t in tokens] == call["output_ids"]
    reply = call["output_text"]
    assert "".join(reply[t["start"] : t["end"]] for t in tokens) == reply
    assert "".join(t["text"] for t in tokens) == reply + "</s>"  # the end token, as text
    assert [t["start"] for t in tokens] == sorted(t["start"] for t in tokens)
    assert tokens[-1]["start"] == tokens[-1]["end"] == len(reply)  # past the reply's end
    assert tokens[0]["top"][1] == {"id": tokenizer.encode("z")[0], "text": "z", "logprob": -3.0}


def test_probs_reads_an_earlier_call_by_its_id(commands):
    backend, tokenizer = commands.session.backend, commands.session.tokenizer
    backend.replies = [" One.", " Two."]
    backend.logprobs = fake_logprobs(tokenizer, tokenizer.encode(" One.") + [1])
    first = commands.run("say", "a", lambda e: None)["call"]
    backend.logprobs = None
    commands.run("say", "b", lambda e: None)
    result = commands.run("probs", first["call_id"], lambda e: None)
    assert result["call_id"] == first["call_id"]
    assert "".join(t["text"] for t in result["tokens"]).startswith(" One.")


@pytest.mark.parametrize(
    ("backend_name", "logprobs", "reason"),
    [
        ("llama.cpp", None, "llama_cpp"),
        ("transformers", None, "off"),
        ("transformers", {"unavailable": "cancelled"}, "cancelled"),
    ],
)
def test_probs_without_probabilities_gives_a_reason_code(commands, backend_name, logprobs, reason):
    backend = commands.session.backend
    backend.name, backend.logprobs = backend_name, logprobs
    call = commands.run("say", "Hi", lambda e: None)["call"]
    result = commands.run("probs", call["call_id"], lambda e: None)
    assert result.get("reason") == reason and result["unavailable"]
    assert result.get("tokens") == [] and result["least_likely"] == []


def test_probs_of_a_failed_call_says_it_has_no_reply(commands):
    from clarity_chat.backends import BackendError

    commands.session.backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        commands.run("say", "Hi", lambda e: None)
    failed = commands.session.conversation.last_call
    result = commands.run("probs", failed["call_id"], lambda e: None)
    assert result.get("reason") == "failed"


def test_probs_offsets_hold_with_a_special_token_inside_the_reply(commands):
    """Transformers writes output_text without special tokens; a base model can write one
    mid-reply (checkpoint review, finding 1)."""
    backend, tokenizer = commands.session.backend, commands.session.tokenizer
    backend.replies = [" ab"]
    eos = tokenizer.tok.eos_token_id
    ids = tokenizer.encode(" a") + [eos] + tokenizer.encode("b")
    backend.logprobs = fake_logprobs(tokenizer, ids)
    reply = commands.run("say", "Hi", lambda e: None)["call"]["output_text"]
    tokens = commands.run("probs", "", lambda e: None)["tokens"]
    spans = [reply[t["start"] : t["end"]] for t in tokens]
    assert spans == [" ", "a", "", "b"]
    assert [t["text"] for t in tokens] == [" ", "a", "</s>", "b"]


def test_probs_of_a_call_cancelled_before_its_reply_says_cancelled(commands):
    """Review finding 7: logged without a turn, but it was cancelled, not failed."""
    from clarity_chat.session import CancelledBeforeReply

    backend = commands.session.backend
    backend.on_call = lambda: commands.run("cancel", "", lambda e: None)
    with pytest.raises(CancelledBeforeReply):
        commands.run("say", "Hi", lambda e: None)
    cancelled = commands.session.conversation.last_call
    assert commands.run("probs", cancelled["call_id"], lambda e: None)["reason"] == "cancelled"
