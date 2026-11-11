"""Where each message sits in a call's prompt (call.message_spans)."""

from __future__ import annotations

import pytest

from clarity_chat.profiles import load_defaults, load_profile
from clarity_chat.render import Tokenizer, build_prompt, message_spans
from tests.conftest import CHAT_PROFILE, TEXT_PROFILE, build_tokenizer

# SmolLM2-135M-Instruct's chat template, from its tokenizer_config.json at commit
# 12fd25f77366fa6b3b4b768ec3050bf629380bac (Apache 2.0). It adds a default system message.
SMOLLM2_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging "
    "Face<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
TURNS = [("user", "Hi there"), ("assistant", " Hello!"), ("user", "Name a river.")]


def profile_for(root, text: str):
    (root / "profiles" / "p.toml").write_text(text, encoding="utf-8")
    return load_profile(root, "p", load_defaults(root))


def spans_for(tokenizer, profile, turns, template=None):
    prompt = build_prompt(tokenizer, profile, turns, template)
    ids = [f"turn-{i}" for i in range(len(turns))]
    return prompt, message_spans(tokenizer, profile, turns, ids, prompt, template)


def assert_covers(prompt, spans) -> None:
    """Every token and character, in order, and each message's text where it says."""
    assert isinstance(spans, list), spans
    tokens = [s["tokens"] for s in spans]
    assert tokens[0][0] == 0 and tokens[-1][1] == len(prompt.ids)
    assert all(a[1] == b[0] for a, b in zip(tokens, tokens[1:], strict=False))
    chars = [s["chars"] for s in spans if s["kind"] != "bos"]
    assert chars[0][0] == 0 and chars[-1][1] == len(prompt.text)
    assert all(a[1] == b[0] for a, b in zip(chars, chars[1:], strict=False))


def messages(prompt, spans) -> list[tuple]:
    return [
        (s.get("turn_id") or s.get("source"), prompt.text[s["chars"][0] : s["chars"][1]])
        for s in spans
        if s["kind"] == "message"
    ]


def test_text_mode_spans(root, tiny_tokenizer):
    profile = profile_for(root, TEXT_PROFILE)
    prompt, spans = spans_for(tiny_tokenizer, profile, TURNS)
    assert_covers(prompt, spans)
    assert messages(prompt, spans) == [
        ("turn-0", "Hi there"),
        ("turn-1", " Hello!"),
        ("turn-2", "Name a river."),
    ]
    assert spans[0] == {"kind": "profile", "chars": [0, 23], "tokens": [0, 23]}  # header + name
    assert spans[-1]["kind"] == "profile" and prompt.text.endswith("Assistant:")


def test_a_prepended_bos_is_its_own_span(root, bos_tokenizer):
    profile = profile_for(root, TEXT_PROFILE)
    prompt, spans = spans_for(bos_tokenizer, profile, TURNS[:1])
    assert prompt.bos["prepended"]
    assert spans[0] == {"kind": "bos", "chars": [0, 0], "tokens": [0, 1]}
    assert_covers(prompt, spans)


def test_smollm2_template_and_its_default_system_message(root, tmp_path):
    tokenizer = build_tokenizer(tmp_path / "smol", None, False, SMOLLM2_TEMPLATE)
    profile = profile_for(root, CHAT_PROFILE.replace('system = "Be brief."\n', ""))
    prompt, spans = spans_for(tokenizer, profile, TURNS)
    assert_covers(prompt, spans)
    assert messages(prompt, spans) == [
        ("turn-0", "Hi there"),
        ("turn-1", " Hello!"),
        ("turn-2", "Name a river."),
    ]
    first = spans[0]
    assert first["kind"] == "template"
    assert "named SmolLM" in prompt.text[first["chars"][0] : first["chars"][1]]


def test_olmo_3_template(root, olmo_tokenizer):
    profile = profile_for(root, CHAT_PROFILE)
    prompt, spans = spans_for(olmo_tokenizer, profile, TURNS)
    assert_covers(prompt, spans)
    assert messages(prompt, spans) == [
        ("system", "Be brief."),
        ("turn-0", "Hi there"),
        ("turn-1", " Hello!"),
        ("turn-2", "Name a river."),
    ]


def toy(tmp_path, template: str) -> Tokenizer:
    return build_tokenizer(tmp_path / "toy", None, False, template)


LOOP = "{% for m in messages %}<{{ m.role }}>{{ BODY }}</s>{% endfor %}<assistant>"


def test_a_template_that_trims_marks_the_trimmed_messages(root, tmp_path):
    tokenizer = toy(tmp_path, LOOP.replace("BODY", "m.content | trim"))
    profile = profile_for(root, CHAT_PROFILE)
    prompt, spans = spans_for(tokenizer, profile, [("user", "  padded \n"), ("assistant", "x")])
    assert_covers(prompt, spans)
    found = [s for s in spans if s["kind"] == "message"]
    assert [s.get("trimmed", False) for s in found] == [False, True, False]
    assert prompt.text[found[1]["chars"][0] : found[1]["chars"][1]] == "padded"


def test_a_template_that_rewrites_a_message_gives_no_spans(root, tmp_path):
    tokenizer = toy(tmp_path, LOOP.replace("BODY", "m.content | upper"))
    profile = profile_for(root, CHAT_PROFILE)
    _, spans = spans_for(tokenizer, profile, TURNS)
    assert spans == {
        "unavailable": "the chat template changes, leaves out, repeats or reorders message 1"
    }


def test_a_template_that_inserts_the_date_has_it_as_template_text(root, tmp_path):
    tokenizer = toy(
        tmp_path, "Today is {{ strftime_now('%Y-%m-%d') }}.\n" + LOOP.replace("BODY", "m.content")
    )
    profile = profile_for(root, CHAT_PROFILE)
    prompt, spans = spans_for(tokenizer, profile, TURNS)
    assert_covers(prompt, spans)
    assert spans[0]["kind"] == "template" and prompt.text.startswith("Today is 20")


def test_a_template_that_fails_on_placeholders_gives_a_reason_not_an_error(root, tmp_path):
    body = "{% if 'CLARITY' in m.content %}{{ raise_exception('no') }}{% endif %}{{ m.content }}"
    tokenizer = toy(tmp_path, LOOP.replace("{{ BODY }}", body))
    profile = profile_for(root, CHAT_PROFILE)
    _, spans = spans_for(tokenizer, profile, TURNS)
    assert set(spans) == {"unavailable"} and "no" in spans["unavailable"]


def test_a_template_that_reorders_messages_gives_no_spans(root, tmp_path):
    reverse = "{% for m in messages | reverse %}<{{ m.role }}>{{ m.content }}</s>{% endfor %}"
    tokenizer = toy(tmp_path, reverse + "<assistant>")
    profile = profile_for(root, CHAT_PROFILE)
    _, spans = spans_for(tokenizer, profile, TURNS)
    assert "unavailable" in spans and "reorders message 2" in spans["unavailable"]


def test_a_long_chat_gets_spans_quickly(root, tmp_path):
    import time

    tokenizer = toy(tmp_path, LOOP.replace("BODY", "m.content"))
    profile = profile_for(root, CHAT_PROFILE)
    turns = [("user" if i % 2 == 0 else "assistant", f"message {i} " * 20) for i in range(400)]
    start = time.monotonic()
    prompt, spans = spans_for(tokenizer, profile, turns)
    assert_covers(prompt, spans)
    assert time.monotonic() - start < 2  # finding the placeholders once each, not per pair


def test_an_empty_message_gets_a_zero_length_span(root, tmp_path):
    tokenizer = toy(tmp_path, LOOP.replace("BODY", "m.content"))
    profile = profile_for(root, CHAT_PROFILE)
    prompt, spans = spans_for(tokenizer, profile, [("user", "")])
    assert_covers(prompt, spans)
    empty = [s for s in spans if s.get("turn_id") == "turn-0"][0]
    assert empty["chars"][0] == empty["chars"][1] and empty["tokens"][0] == empty["tokens"][1]


def test_a_token_crossing_an_edge_belongs_to_the_span_it_starts_in(root, olmo_tokenizer):
    """Text mode writes "User: " then the message; OLMo 3's tokenizer makes " hi" one token,
    which starts in the name part."""
    profile = profile_for(root, TEXT_PROFILE)
    prompt, spans = spans_for(olmo_tokenizer, profile, [("user", "hi there")])
    assert_covers(prompt, spans)
    name, message = spans[0], spans[1]
    assert name["kind"] == "profile" and name["edge_token"] == "end"
    assert message["turn_id"] == "turn-0" and "edge_token" not in message
    tokens = prompt.ids[message["tokens"][0] : message["tokens"][1]]
    assert olmo_tokenizer.decode(tokens) == " there"


def test_each_call_records_its_spans(make_session):
    session, _ = make_session([" one", " two"], profile="chat")
    session.say("Hi")
    record = session.say("Again").record
    spans = record["message_spans"]
    conversation = session.conversation
    assert conversation is not None
    path = [t.turn_id for t in conversation.path_to(conversation.head)][:-1]
    assert [s["turn_id"] for s in spans if s.get("turn_id")] == path


@pytest.fixture
def make_session(fetched_root, chat_tokenizer):
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    def make(replies, profile="text"):
        backend = FakeBackend(replies, tokenizer=chat_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer
        )
        session.new_conversation(profile)
        return session, backend

    return make


@pytest.mark.parametrize("case", ["upper", "lower"])
def test_a_case_changing_template_is_caught_whatever_the_random_part(
    root, tmp_path, monkeypatch, case
):
    """The placeholder must change under any case mapping, even if its random part is all
    digits (about 1 run in 280), so the reason is the same every run."""
    monkeypatch.setattr("clarity_chat.render.secrets.token_hex", lambda n: "0" * (2 * n))
    tokenizer = toy(tmp_path, LOOP.replace("BODY", f"m.content | {case}"))
    profile = profile_for(root, CHAT_PROFILE)
    _, spans = spans_for(tokenizer, profile, TURNS)
    assert spans == {
        "unavailable": "the chat template changes, leaves out, repeats or reorders message 1"
    }
