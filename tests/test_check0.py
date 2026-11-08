"""The profiles shipped in profiles/: they load, and the Check 0 setups differ only as intended."""

from __future__ import annotations

import pytest

from clarity_chat.profiles import load_defaults, load_profile
from clarity_chat.render import build_prompt
from tests.conftest import REPO
from tests.test_chat import OLMO_BLANK_SYSTEM, OLMO_NO_SYSTEM

CHECK0 = ["1a-stock", "1b-blank", "2-system", "3-primed", "4-raw-instruct", "5-raw-base"]


def repo_profile(name):
    return load_profile(REPO, name, load_defaults(REPO))


@pytest.mark.parametrize("path", sorted((REPO / "profiles").glob("*.toml")), ids=lambda p: p.stem)
def test_every_shipped_profile_loads(path):
    repo_profile(path.stem)


def test_check0_profiles_share_every_setting():
    settings = [repo_profile(name).settings for name in CHECK0]
    assert all(s == settings[0] for s in settings)
    assert settings[0]["max_tokens"].value == 1024
    assert settings[0]["temperature"].value == 0.6


def test_check0_models_and_modes():
    got = {name: (p.model, p.mode) for name in CHECK0 for p in [repo_profile(name)]}
    assert got["1a-stock"] == ("olmo-3-7b-instruct", "chat")
    assert got["4-raw-instruct"] == ("olmo-3-7b-instruct", "text")
    assert got["5-raw-base"] == ("olmo-3-7b-base", "text")


@pytest.mark.parametrize(
    ("name", "expected"), [("1a-stock", OLMO_NO_SYSTEM), ("1b-blank", OLMO_BLANK_SYSTEM)]
)
def test_stock_and_blank_render_the_briefs_prompts(olmo_tokenizer, name, expected):
    prompt = build_prompt(olmo_tokenizer, repo_profile(name), [("user", "hi")])
    assert prompt.text == expected


def test_system_and_priming_files_reach_the_prompt(olmo_tokenizer):
    system = build_prompt(olmo_tokenizer, repo_profile("2-system"), [("user", "hi")])
    assert system.text.startswith("<|im_start|>system\nPLACEHOLDER SYSTEM PROMPT")
    primed = build_prompt(olmo_tokenizer, repo_profile("3-primed"), [("user", "hi")])
    assert "<|im_start|>user\nPLACEHOLDER: an earlier user message.<|im_end|>" in primed.text
    assert primed.text.index("earlier assistant reply") < primed.text.index("user\nhi")


def test_raw_instruct_document(olmo_tokenizer):
    prompt = build_prompt(olmo_tokenizer, repo_profile("4-raw-instruct"), [("user", "hi")])
    assert prompt.text == "A conversation between User and Assistant.\n\nUser: hi\nAssistant:"
    assert prompt.bos["prepended"] is False  # OLMo 3's tokenizer config sets no BOS
