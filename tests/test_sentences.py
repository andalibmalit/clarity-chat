"""Sentences a web user can meet name no terminal command (v0.2.3 item 11), and the
fact-check's fixes. Usage errors from typed commands keep their command names."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.profiles import ConfigError
from clarity_chat.session import ContextError, Session
from tests.fakes import FakeBackend

COMMAND = re.compile(r"(^|[\s(`'\"])/[a-z]+|clarity [a-z]+")


def names_no_command(text: str) -> bool:
    return COMMAND.search(text) is None


@pytest.fixture
def make(fetched_root, tiny_tokenizer):
    def make(replies=None, n_ctx=None):
        backend = FakeBackend(replies or [" ok"] * 5, n_ctx=n_ctx, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        session.new_conversation("text")
        return session, backend

    return make


def test_the_context_refusal(make):
    session, _ = make(n_ctx=64)
    with pytest.raises(ContextError) as caught:
        session.say("word " * 40)
    assert "Nothing was sent" in str(caught.value) and names_no_command(str(caught.value))


def test_the_control_token_warning(make):
    session, _ = make()
    (warning,) = [w for w in session.say("a</s>b").warnings if "control token" in w]
    assert names_no_command(warning)


def test_the_unread_template_kwargs_warning():
    prepared = SimpleNamespace(unread_kwargs=["greting"])
    (warning,) = Session._profile_warnings(prepared)  # ty: ignore[invalid-argument-type]
    assert "greting" in warning and names_no_command(warning)


def test_the_not_applied_warning_names_only_settings_that_could_change_the_reply(make):
    session, backend = make()
    backend.not_applied = [
        {"name": "presence_penalty", "value": 0.5, "reason": "r", "affects_output": True},
        {"name": "repeat_last_n", "value": 64, "reason": "r", "affects_output": False},
    ]
    outcome = session.say("Hi")
    (warning,) = [w for w in outcome.warnings if "can't apply" in w]
    assert "presence_penalty" in warning and "repeat_last_n" not in warning
    assert names_no_command(warning)


def test_the_failed_switch_sentence(make):
    session, backend = make()

    def always_fail(model, weights, ctx_size):
        raise ConfigError("no GPU")

    backend.start = always_fail
    with pytest.raises(BackendError) as caught:
        session.new_conversation("text")
    assert "failed too" in str(caught.value) and names_no_command(str(caught.value))


def test_the_replay_refusals(make, fetched_root):
    session, _ = make()
    call = session.say("Hi").record
    path = session.conversation.log.path
    text = (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (fetched_root / "profiles" / "big.toml").write_text("ctx_size = 4096\n" + text, "utf-8")
    session.new_conversation("big")
    with pytest.raises(ConfigError) as caught:
        session.replay(path, call["call_id"])
    assert "context of 8192" in str(caught.value) and names_no_command(str(caught.value))


def test_the_busy_refusal(make):
    session, _ = make()
    c = Commands(session)
    c.busy.acquire()
    with pytest.raises(ConfigError) as caught:
        c.run("say", "Hi", lambda e: None)
    assert "still running" in str(caught.value) and names_no_command(str(caught.value))


def test_the_two_reasons_for_no_probabilities_are_worded_apart(make):
    session, backend = make()
    c = Commands(session)
    backend.name = "llama.cpp"
    llama = c.run("say", "a", lambda e: None)["call"]
    backend.name = "transformers"
    off = c.run("say", "b", lambda e: None)["call"]
    sentences = [c.run("probs", x["call_id"], lambda e: None)["unavailable"] for x in (llama, off)]
    assert sentences[0] != sentences[1]
    assert all(names_no_command(s) for s in sentences)
    assert "llama.cpp" in sentences[0]


def test_a_prompt_that_starts_with_bos_is_not_labelled_without_one():
    """Llama 3.1's template writes its own BOS: Clarity prepends none, yet the prompt has one."""
    from clarity_chat.cli import format_prompt

    record = {
        "prompt_text": "<s>Hi",
        "prompt_ids": [1, 72, 105],
        "n_prompt_client": 3,
        "bos": {"bos_token_id": 1, "prepended": False, "reason": "chat mode"},
    }
    header = format_prompt(record, as_repr=False).splitlines()[0]
    assert "starts with the start token (BOS)" in header and "no BOS" not in header
    record["prompt_ids"], record["prompt_text"] = [72, 105], "Hi"
    assert "no start token (BOS)" in format_prompt(record, as_repr=False).splitlines()[0]
