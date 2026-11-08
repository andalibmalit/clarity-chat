from __future__ import annotations

import pytest

from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from clarity_chat.render import bos_decision, build_prompt, render_text, tokenize
from tests.conftest import build_tokenizer


@pytest.fixture
def text_profile(root):
    return load_profile(root, "text", load_defaults(root))


def test_text_document_format(text_profile):
    turns = [("user", "Hello there"), ("assistant", " Hi! How can I help?"), ("user", "Two\nlines")]
    assert render_text(text_profile, turns) == (
        "A conversation.\n\n"
        "User: Hello there\n"
        "Assistant: Hi! How can I help?\n"
        "User: Two\nlines\n"
        "Assistant:"
    )


def test_text_document_ends_without_trailing_space(text_profile):
    assert render_text(text_profile, [("user", "hi")]).endswith("\nAssistant:")


def test_model_output_is_used_exactly_as_returned(text_profile):
    doc = render_text(
        text_profile, [("user", "a"), ("assistant", "no leading space  "), ("user", "b")]
    )
    assert "Assistant:no leading space  \nUser: b\n" in doc


def test_priming_text_goes_after_the_header(root):
    (root / "profiles" / "p.toml").write_text(
        'backend = "llama.cpp"\nmodel = "tiny"\nmode = "text"\nuser_name = "U"\n'
        'assistant_name = "A"\nheader = "H\\n"\npriming = "U: x\\nA: y\\n"\n',
        encoding="utf-8",
    )
    profile = load_profile(root, "p", load_defaults(root))
    assert render_text(profile, [("user", "z")]) == "H\nU: x\nA: y\nU: z\nA:"


def test_tokens_are_ids_of_the_exact_text(text_profile, tiny_tokenizer):
    prompt = build_prompt(tiny_tokenizer, text_profile, [("user", "héllo ✓")])
    assert tiny_tokenizer.decode(prompt.ids) == prompt.text
    assert prompt.bos["prepended"] is False


def test_bos_prepended_when_the_tokenizer_config_says_so(text_profile, bos_tokenizer):
    prompt = build_prompt(bos_tokenizer, text_profile, [("user", "hi")])
    assert prompt.bos["add"] is True
    assert prompt.bos["prepended"] is True
    assert prompt.bos["reason"] == "tokenizer_config.json add_bos_token"
    assert prompt.ids[0] == bos_tokenizer.bos[1]
    assert prompt.ids[1:] == bos_tokenizer.encode(prompt.text)
    assert prompt.expected_server_text == "<s>" + prompt.text


def test_bos_not_doubled_when_the_text_starts_with_it(bos_tokenizer):
    ids, decision = tokenize(bos_tokenizer, "<s>hello", "text")
    assert decision["text_starts_with_bos"] is True
    assert decision["prepended"] is False
    assert ids.count(bos_tokenizer.bos[1]) == 1


def test_no_bos_key_falls_back_to_what_the_tokenizer_does(tmp_path):
    tok = build_tokenizer(tmp_path, None, True)
    decision = bos_decision(tok, "text")
    assert decision["add"] is True
    assert "tokenizer's behavior" in decision["reason"]


@pytest.mark.parametrize(("config_add", "adds"), [(True, False), (False, True)])
def test_config_and_tokenizer_disagreeing_is_an_error(tmp_path, config_add, adds):
    tok = build_tokenizer(tmp_path, config_add, adds)
    with pytest.raises(ConfigError, match="Can't decide BOS"):
        bos_decision(tok, "text")


def test_chat_mode_never_prepends_bos(bos_tokenizer):
    ids, decision = tokenize(bos_tokenizer, "hi", "chat")
    assert decision["prepended"] is False
    assert ids == bos_tokenizer.encode("hi")
