from __future__ import annotations

from pathlib import Path

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
    assert "context 44 of 8,192" in text  # 36 prompt tokens (one per byte) + 7 reply + end


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
    output.clear()
    backend.name = "llama.cpp"
    backend.record = {**backend.record, "model_file": {"eog_tokens": None}}
    chat.handle("/settings")
    assert "end-of-reply tokens: not found in llama-server's startup log" in "".join(output)


def test_settings_shows_where_the_transformers_end_tokens_came_from(loop):
    chat, output, backend = loop
    backend.name = "transformers"
    origin = "tokenizer (generation_config.json has none)"
    backend.record = {**backend.record, "eos_token_id": [257], "eos_token_id_from": origin}
    chat.handle("/settings")
    assert f"eos_token_id = [257]  {origin}" in "".join(output)


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
    backend.cancel_after = 0  # Ctrl-C before the first token
    chat.handle("Hi there")
    text = "".join(output)
    assert "Cancelled before the reply started. No reply was added." in text
    assert "Your message is back in the input" in text
    assert chat.draft == "Hi there"


@pytest.mark.parametrize("line", ["/edit  ", "/edit \n\n"])
def test_an_edit_of_only_whitespace_is_cancelled(loop, line):
    chat, output, backend = loop
    chat.handle("Hi")
    output.clear()
    chat.handle(line)
    assert "Edit cancelled." in "".join(output)
    assert len(backend.calls) == 1


def test_cancel_between_calls_says_there_is_nothing_to_cancel(loop):
    chat, output, _ = loop
    chat.handle("/cancel")
    assert "Nothing to cancel." in "".join(output)


def test_the_chat_says_when_it_ends_a_torn_line(loop):
    chat, output, _ = loop
    chat.handle("a")
    with chat.session.conversation.log.path.open("ab") as f:
        f.write(b'{"type": "no')
    chat.handle("b")
    text = "".join(output)
    assert "Line 6 of this conversation's log has no line end, so it was cut off" in text


def test_set_in_the_terminal_and_in_settings(loop):
    chat, output, _ = loop
    chat.handle("/set temperature 0.9")
    chat.handle("/settings")
    chat.handle("/set temperature")
    text = "".join(output)
    assert "temperature = 0.9 for this chat." in text
    assert "temperature is back to 0.6 (profile)." in text
    assert any("temperature" in line and "chat" in line for line in text.splitlines())


def test_title_and_chats_in_the_terminal(loop, fetched_root, capsys):
    from clarity_chat import cli

    chat, output, _ = loop
    chat.handle("Hi")
    chat.handle("/title Made-up greeting")
    chat.handle("/chats")
    text = "".join(output)
    assert "Titled 'Made-up greeting'." in text
    assert f"{chat.session.conversation.conversation_id}  text  Made-up greeting" in text
    assert cli.main(["--root", str(fetched_root), "chats"]) == 0
    assert "Made-up greeting" in capsys.readouterr().out


def test_redirected_output_is_utf8_whatever_the_locale(loop, fetched_root, monkeypatch):
    """Windows writes pipes and files in its ANSI code page, which can't hold every title."""
    import io
    import sys

    from clarity_chat import cli

    chat, _, _ = loop
    chat.handle("Hi")
    chat.handle("/title 日本語 → made-up 🙂")
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="cp1252"))
    assert cli.main(["--root", str(fetched_root), "chats"]) == 0
    sys.stdout.flush()
    assert "日本語 → made-up 🙂".encode() in out.getvalue()


def test_ctrl_c_during_a_call_cancels_it_and_the_chat_goes_on(loop):
    """Ctrl-C anywhere in a call, not only while the stream is read, cancels the call
    (docs/reference.md): the chat keeps running and the log keeps the call."""
    import signal

    from clarity_chat.logs import read_records

    chat, output, backend = loop
    backend.on_call = lambda: signal.raise_signal(signal.SIGINT)
    try:
        keep_going = chat.handle("Hi")
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped the call and would quit the chat")
    assert keep_going is True
    assert "No reply was added" in "".join(output)
    conversation = chat.session.conversation
    assert conversation is not None
    calls = [r for r in read_records(conversation.log.path) if r["type"] == "call"]
    assert calls and calls[-1]["stop_type"] == "cancelled"
    backend.on_call = None
    output.clear()
    assert chat.handle("Again") is True and "! " not in "".join(output)


def test_ctrl_c_during_a_command_that_cant_be_cancelled_still_stops_clarity(loop, fetched_root):
    """A model start can't be cancelled, so Ctrl-C keeps its old effect there."""
    import signal

    chat, _, backend = loop
    text = (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (fetched_root / "profiles" / "big.toml").write_text("ctx_size = 4096\n" + text, "utf-8")
    original = backend.start

    def start(*args):
        signal.raise_signal(signal.SIGINT)
        return original(*args)

    backend.start = start
    with pytest.raises(KeyboardInterrupt):
        chat.handle("/profile big")


def test_say_sends_its_text_exactly_like_a_plain_line(loop, tiny_tokenizer):
    """Owner's decision (2026-11-11): only the one space after /say is dropped."""
    chat, _, backend = loop
    chat.handle("/say  hi  ")
    sent = tiny_tokenizer.decode(backend.calls[-1]["token_ids"])
    assert sent.endswith("User:  hi  \nAssistant:")


def test_the_terminal_starts_from_a_profile_path(fetched_root, tiny_tokenizer):
    """--profile PATH and --resume PATH come from the command line, which you type."""
    backend = FakeBackend([" Hello!"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    path = str(fetched_root / "profiles" / "text.toml")
    output: list[str] = []
    chat = ChatLoop(session, path, read=lambda default: "", write=output.append)
    chat.start()
    assert "Ready:" in "".join(output)
    chat.handle("Hi")
    assert session.conversation is not None
    log = str(session.conversation.log.path)
    again = ChatLoop(session, "text", read=lambda default: "", write=output.append, resume=log)
    again.start()
    assert session.conversation is not None
    assert session.conversation.conversation_id == Path(log).stem
