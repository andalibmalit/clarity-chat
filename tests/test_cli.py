from __future__ import annotations

import pytest

from clarity_chat.cli import ChatLoop, visible_lines
from clarity_chat.session import Session
from tests.fakes import FakeBackend


@pytest.fixture
def loop(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!", " Again."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    chat = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    chat.start()
    output.clear()
    return chat, output, backend


def test_message_streams_the_reply_and_a_status_line(loop):
    chat, output, _ = loop
    assert chat.handle("Hi")
    text = "".join(output)
    assert " Hello!" in text
    assert "stop eos" in text
    assert "seed 42" in text
    assert "context 43 of 8,192" in text  # 36 prompt tokens (one per byte) + 7 reply tokens


def test_prompt_shows_text_ids_and_count(loop):
    chat, output, backend = loop
    chat.handle("Hi")
    output.clear()
    chat.handle("/prompt")
    text = "".join(output)
    assert "A conversation.\n\nUser: Hi\nAssistant:" in text
    assert str(backend.calls[0]["token_ids"]) in text
    assert f"{len(backend.calls[0]['token_ids'])} tokens" in text


def test_prompt_repr_makes_whitespace_visible(loop):
    chat, output, _ = loop
    chat.handle("Hi")
    output.clear()
    chat.handle("/prompt repr")
    assert "'A conversation.\\n'" in "".join(output)


def test_settings_lists_values_and_origins(loop):
    chat, output, _ = loop
    chat.handle("/settings")
    text = "".join(output)
    assert "temperature" in text and "profile" in text
    assert "repeat_penalty" in text and "default" in text
    assert "mirostat = 0" in text


def test_settings_lists_the_end_of_reply_tokens(loop):
    chat, output, backend = loop
    backend.record = {**backend.record, "model_file": {"eog_tokens": [{"id": 2, "text": "</s>"}]}}
    chat.handle("/settings")
    assert "end-of-reply tokens (from the model file): 2 '</s>'" in "".join(output)


def test_settings_gives_the_reason_a_setting_was_not_applied(loop):
    chat, output, backend = loop
    backend.not_applied = [
        {
            "name": "presence_penalty",
            "value": 0.5,
            "reason": "no equivalent",
            "affects_output": True,
        }
    ]
    chat.handle("Hi")
    output.clear()
    chat.handle("/settings")
    assert "  presence_penalty: no equivalent" in "".join(output)


def test_slash_slash_sends_a_literal_slash(loop):
    chat, _, backend = loop
    chat.handle("//not a command")
    assert backend.calls


def test_new_and_quit(loop):
    chat, output, _ = loop
    first = chat.session.conversation.conversation_id
    assert chat.handle("/new")
    assert chat.session.conversation.conversation_id != first
    assert chat.handle("/quit") is False


def test_unknown_command(loop):
    chat, output, backend = loop
    assert chat.handle("/bogus")
    assert "Unknown command" in "".join(output)
    assert not backend.calls


def test_visible_lines():
    assert visible_lines("a \n\tb\n") == ["'a \\n'", "'\\tb\\n'"]
    assert visible_lines("x") == ["'x'"]


def test_new_reloads_a_profile_given_as_a_path(fetched_root, tiny_tokenizer, tmp_path):
    outside = tmp_path / "elsewhere" / "exp.toml"
    outside.parent.mkdir()
    outside.write_text(
        (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    backend = FakeBackend([" Hi"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    chat = ChatLoop(session, str(outside), read=lambda default: "", write=output.append)
    chat.start()
    assert chat.handle("/new")
    assert session.conversation is not None
    assert session.conversation.profile.path == outside
    assert "New conversation" in "".join(output)


def test_a_failing_new_keeps_the_chat_running(loop):
    chat, output, _ = loop
    first = chat.session.conversation
    (first.profile.path).write_text("not = [valid", encoding="utf-8")
    assert chat.handle("/new")
    assert "! " in "".join(output)
    assert chat.session.conversation is first


def test_server_warnings_show_once_per_server(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hi"], tokenizer=tiny_tokenizer)
    backend.start_warnings = ["llama-server inherits ['GGML_X']"]
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    chat = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    chat.start()
    assert "! llama-server inherits" in "".join(output)
    output.clear()
    chat.handle("/new")  # same model, same server: no repeat
    assert "inherits" not in "".join(output)


def test_an_early_cancel_offers_the_message_back(loop):
    chat, output, backend = loop
    backend.on_call = chat.session.cancel
    chat.handle("Hi there")
    text = "".join(output)
    assert "Cancelled before the reply started. No reply was added." in text
    assert "Your message is back in the input" in text
    assert chat.draft == "Hi there"
