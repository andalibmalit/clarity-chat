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
    chat = ChatLoop(session, "text", read=lambda: "", write=output.append)
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
