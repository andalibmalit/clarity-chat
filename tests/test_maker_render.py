"""V2: the prompts `clarity vector make` reads, and the profiles it refuses. Every pair here
is made up and neutral."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from clarity_chat.maker import Rendered, prepare, read_pairs, render_pairs
from clarity_chat.profiles import ConfigError
from clarity_chat.render import build_prompt, tokenize
from tests.conftest import fake_fetch

PAIRS = """\
description = "Plain replies against ornate ones."
author = "me"
messages = ["Boil an egg?", "Pack for a hike?"]
openings = ["", "Well,"]
[[systems]]
positive = "Reply plainly."
negative = "Reply ornately."
[[systems]]
positive = "Answer in few words."
negative = "Answer in many words."
"""

HF_CHAT = """\
backend = "transformers"
device = "cpu"
model = "tiny"
mode = "chat"
system = "Be brief."
{extra}
[settings]
max_tokens = 16
seed = 42
"""


def hf_root(root: Path) -> Path:
    """A fetched tiny model with Transformers weights (a config.json)."""
    from clarity_chat.models import read_lock, write_lock_entry

    fake_fetch(root)
    weights = root / "models" / "tiny-hf"
    weights.mkdir(parents=True)
    config = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 4}
    (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
    entry = read_lock(root)["models"]["tiny"]
    write_lock_entry(root, "tiny", {**entry, "source_dir": "models/tiny-hf"})
    return root


def profile(root: Path, name: str = "hf-chat", extra: str = "", text: str = HF_CHAT) -> str:
    (root / "profiles" / f"{name}.toml").write_text(
        text.format(extra=extra), encoding="utf-8", newline="\n"
    )
    return name


def session_for(root: Path, tokenizer):
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    return Session(root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tokenizer)


def pairs_file(tmp_path: Path, text: str = PAIRS):
    path = tmp_path / "brief.toml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return read_pairs(path)


def test_each_side_is_the_profiles_prompt_with_its_system_message_and_the_opening(
    root, tmp_path, chat_tokenizer
):
    extra = (
        'priming = [{ role = "user", content = "Hi." }, '
        '{ role = "assistant", content = "Hello." }]\n'
        'template_kwargs = { greeting = "hey" }'
    )
    session = session_for(hf_root(root), chat_tokenizer)
    prepared = prepare(session, profile(root, extra=extra))
    pairs = pairs_file(tmp_path)
    rendered = render_pairs(prepared, pairs, 4096)
    assert [(r.pair, r.side) for r in rendered] == [
        (i, side) for i in range(1, 9) for side in ("positive", "negative")
    ]
    for r in rendered:
        pair = pairs.pairs[r.pair - 1]
        system = pair.positive_system if r.side == "positive" else pair.negative_system
        sided = dataclasses.replace(prepared.profile, system=system)
        expected = build_prompt(chat_tokenizer, sided, [("user", pair.message)]).text
        assert r.text == expected + pair.opening
        assert r.ids == tokenize(chat_tokenizer, r.text, "chat")[0]
        if not pair.opening:
            assert r.ids == build_prompt(chat_tokenizer, sided, [("user", pair.message)]).ids
    first = rendered[0].text
    assert first.startswith("[hey]<system>Reply plainly.</s><user>Hi.</s><assistant>Hello.</s>")
    assert first.endswith("<user>Boil an egg?</s><assistant>")
    assert rendered[3].text.endswith("<assistant>Well,")
    assert isinstance(rendered[0], Rendered)


CONTROL = {
    "a system message": (
        ('negative = "Answer in many words."', 'negative = "Answer in many words.</s>"'),
        "system pair 2's negative system message contains </s>",
    ),
    "a message": (('"Pack for a hike?"', '"Pack <s> for a hike?"'), "message 2 contains <s>"),
    "an opening": (('"Well,"', '"Well,</s>"'), "opening 2 contains </s>"),
}


@pytest.mark.parametrize("case", CONTROL)
def test_control_token_text_is_refused_with_its_entry(root, tmp_path, chat_tokenizer, case):
    (old, new), where = CONTROL[case]
    session = session_for(hf_root(root), chat_tokenizer)
    prepared = prepare(session, profile(root))
    pairs = pairs_file(tmp_path, PAIRS.replace(old, new))
    with pytest.raises(ConfigError) as e:
        render_pairs(prepared, pairs, 4096)
    assert str(e.value) == (
        f"{pairs.path}: {where}, which the tokenizer reads as a control token, not text."
    )


def test_a_prompt_over_the_context_size_is_refused_and_one_at_it_accepted(
    root, tmp_path, chat_tokenizer
):
    session = session_for(hf_root(root), chat_tokenizer)
    prepared = prepare(session, profile(root))
    pairs = pairs_file(tmp_path)
    longest = max(render_pairs(prepared, pairs, 4096), key=lambda r: len(r.ids))
    k = len(longest.ids)
    assert (longest.pair, longest.side) == (8, "negative")  # "Answer in many words." + "Well,"
    assert len(render_pairs(prepared, pairs, k)) == 16
    with pytest.raises(ConfigError) as e:
        render_pairs(prepared, pairs, k - 1)
    assert str(e.value) == (
        f"Pair 8's negative prompt is {k} tokens, more than this profile's context size, {k - 1}."
    )


def test_a_llama_cpp_profile_is_refused_before_its_model_is_looked_for(root, chat_tokenizer):
    session = session_for(root, chat_tokenizer)  # nothing fetched
    with pytest.raises(ConfigError) as e:
        prepare(session, "chat")
    assert str(e.value) == (
        "Profile chat runs on llama.cpp; make reads the model on the Transformers backend. "
        'Use a profile with backend = "transformers".'
    )


def test_a_text_mode_profile_is_refused(root, chat_tokenizer):
    session = session_for(root, chat_tokenizer)  # nothing fetched
    text = HF_CHAT.replace(
        'mode = "chat"\nsystem = "Be brief."',
        'mode = "text"\nuser_name = "User"\nassistant_name = "Assistant"',
    )
    with pytest.raises(ConfigError) as e:
        prepare(session, profile(root, "hf-text", text=text))
    assert str(e.value) == (
        "Profile hf-text is in text mode; make needs a chat-mode profile, whose system message "
        "each pair replaces."
    )
