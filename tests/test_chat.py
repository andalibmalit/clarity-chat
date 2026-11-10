"""Chat mode: template rendering, token IDs, BOS, and the call record."""

from __future__ import annotations

import pytest

from clarity_chat.files import read_text, sha256_text
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from clarity_chat.render import build_prompt, chat_template_source
from clarity_chat.session import ContextError, Session
from tests.conftest import FIXTURES
from tests.fakes import FakeBackend

OLMO_NO_SYSTEM = (
    "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently "
    "have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nhi"
    "<|im_end|>\n<|im_start|>assistant\n"
)
OLMO_BLANK_SYSTEM = (
    "<|im_start|>system\n <|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
)


def chat_profile(root, extra=""):
    path = root / "profiles" / "c.toml"
    path.write_text(
        f'backend = "llama.cpp"\nmodel = "tiny"\nmode = "chat"\n{extra}\n', encoding="utf-8"
    )
    return load_profile(root, "c", load_defaults(root))


@pytest.mark.parametrize(
    ("system_line", "expected_text", "expected_count"),
    [("", OLMO_NO_SYSTEM, 37), ('system = " "', OLMO_BLANK_SYSTEM, 15)],
)
def test_olmo3_instruct_prompt_for_hi(
    root, olmo_tokenizer, system_line, expected_text, expected_count
):
    prompt = build_prompt(olmo_tokenizer, chat_profile(root, system_line), [("user", "hi")])
    assert prompt.text == expected_text
    assert len(prompt.ids) == expected_count
    assert prompt.bos["prepended"] is False
    assert olmo_tokenizer.decode(prompt.ids) == expected_text


def test_olmo3_assistant_turn_is_closed_by_the_template(root, olmo_tokenizer):
    turns = [("user", "hi"), ("assistant", "Hello."), ("user", "bye")]
    prompt = build_prompt(olmo_tokenizer, chat_profile(root, 'system = " "'), turns)
    assert "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nbye<|im_end|>\n" in (
        prompt.text
    )
    assert prompt.text.endswith("<|im_start|>assistant\n")


def test_logged_template_hash_is_the_template_files_hash(root, olmo_tokenizer):
    source = chat_template_source(olmo_tokenizer, chat_profile(root))
    file_text = read_text(FIXTURES / "olmo3-instruct" / "chat_template.jinja")
    assert source["sha256"] == sha256_text(file_text)


def test_messages_go_system_then_priming_then_conversation(root, chat_tokenizer):
    profile = chat_profile(
        root,
        'system = "S"\npriming = [{role = "user", content = "p1"}, '
        '{role = "assistant", content = "p2"}]',
    )
    prompt = build_prompt(chat_tokenizer, profile, [("user", "u1"), ("assistant", "a1")])
    assert prompt.text == (
        "<system>S</s><user>p1</s><assistant>p2</s><user>u1</s><assistant>a1</s><assistant>"
    )


def test_no_system_key_means_no_system_message(root, chat_tokenizer):
    prompt = build_prompt(chat_tokenizer, chat_profile(root), [("user", "u")])
    assert prompt.text == "<user>u</s><assistant>"


def test_template_kwargs_reach_the_template(root, chat_tokenizer):
    profile = chat_profile(root, '[template_kwargs]\ngreeting = "hey"')
    prompt = build_prompt(chat_tokenizer, profile, [("user", "u")])
    assert prompt.text.startswith("[hey]")
    assert prompt.template is not None and prompt.template["kwargs"] == {"greeting": "hey"}
    assert prompt.template["unread_kwargs"] == []


def test_a_template_kwarg_the_template_never_reads_is_a_logged_warning(
    fetched_root, chat_tokenizer
):
    profiles = fetched_root / "profiles"
    text = (profiles / "chat.toml").read_text(encoding="utf-8")
    (profiles / "typo.toml").write_text(
        text + '\n[template_kwargs]\ngreting = "hey"\n', encoding="utf-8"
    )
    backend = FakeBackend(["ok"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    conversation = session.new_conversation("typo")
    assert "never reads template_kwargs ['greting']" in conversation.warnings[0]
    first = next(read_records(conversation.log.path))
    assert first["warnings"] == conversation.warnings
    assert session.say("hi").record["template"]["unread_kwargs"] == ["greting"]


def test_chat_mode_never_prepends_bos(root, chat_tokenizer):
    """The tokenizer config says add_bos_token = true; chat mode still adds nothing."""
    prompt = build_prompt(chat_tokenizer, chat_profile(root), [("user", "u")])
    assert prompt.bos["config_add_bos_token"] is True
    assert prompt.bos["prepended"] is False
    assert prompt.ids == chat_tokenizer.encode(prompt.text)


def test_another_repos_template_is_tokenized_by_the_models_tokenizer(
    root, chat_tokenizer, tiny_tokenizer
):
    prompt = build_prompt(tiny_tokenizer, chat_profile(root), [("user", "u")], chat_tokenizer)
    assert prompt.text == "<user>u</s><assistant>"
    assert prompt.ids == tiny_tokenizer.encode(prompt.text)
    assert prompt.template is not None
    assert prompt.template["path"] == chat_tokenizer.source["path"]


def test_a_tokenizer_without_a_chat_template_is_an_error(root, tiny_tokenizer):
    with pytest.raises(ConfigError, match="no single chat template"):
        build_prompt(tiny_tokenizer, chat_profile(root), [("user", "u")])


@pytest.fixture
def chat_session(fetched_root, chat_tokenizer):
    def make(replies=None, n_ctx=512):
        backend = FakeBackend(replies, n_ctx=n_ctx, tokenizer=chat_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer
        )
        session.new_conversation("chat")
        return session, backend

    return make


def test_chat_call_sends_the_rendered_ids_and_logs_the_template(chat_session, chat_tokenizer):
    session, backend = chat_session(["Hi there"])
    session.say("Hello")
    outcome = session.say("Again")
    sent = backend.calls[-1]
    expected = (
        "<system>Be brief.</s><user>Hello</s><assistant>Hi there</s><user>Again</s><assistant>"
    )
    assert chat_tokenizer.decode(sent["token_ids"]) == expected
    assert sent["stop"] == []
    record = outcome.record
    assert record["prompt_text"] == expected
    assert record["template"]["sha256"] == sha256_text(chat_tokenizer.tok.chat_template)
    assert record["template"]["kwargs"] == {}
    assert record["bos"]["prepended"] is False
    calls = [r for r in read_records(session.conversation.log.path) if r["type"] == "call"]
    assert len(calls) == 2


def test_chat_context_guard_refuses_before_sending(chat_session):
    session, backend = chat_session(n_ctx=40)  # the prompt is about 30 tokens; max_tokens 16
    with pytest.raises(ContextError, match="Nothing was sent"):
        session.say("Hello")
    assert backend.calls == []


def test_a_failing_template_is_a_config_error(root, chat_tokenizer, tmp_path):
    from tests.conftest import build_tokenizer

    strict = build_tokenizer(
        tmp_path / "strict", None, False, "{{ raise_exception('roles must alternate') }}"
    )
    with pytest.raises(ConfigError, match="roles must alternate"):
        build_prompt(strict, chat_profile(root), [("user", "u")])


def test_control_token_text_in_content_is_reported(root, chat_tokenizer):
    prompt = build_prompt(chat_tokenizer, chat_profile(root), [("user", "a</s>b")])
    assert prompt.special_in_content == [{"in": "turn 1 (user)", "tokens": ["</s>"]}]


def test_control_token_text_in_a_message_warns_and_flags(chat_session):
    session, _ = chat_session(["ok"])
    outcome = session.say("a</s>b")
    assert "special_tokens_in_content" in outcome.record["flags"]
    assert any("control tokens" in w for w in outcome.warnings)
    plain = session.say("fine")
    assert not any("control tokens" in w for w in plain.warnings)  # only the new message warns
    assert "special_tokens_in_content" in plain.record["flags"]  # still in the prompt


def test_unread_kwargs_parses_a_real_template(olmo_tokenizer):
    from clarity_chat.render import unread_kwargs

    template = olmo_tokenizer.tok.chat_template
    assert unread_kwargs(template, {"tools": [], "no_such_key": 1}) == ["no_such_key"]
