"""v0.4.0 A6: the terminal's /prompt shows each control character as ⟨U+001B⟩, the page's
notation, so text in a prompt can't rewrite the terminal. The text sent is unchanged.

Made-up text only.
"""

from __future__ import annotations

import pytest

from clarity_chat.cli import ChatLoop, visible_lines
from clarity_chat.commands import COMMANDS
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_json_chat import line, run

# ESC (with an ANSI colour), BEL, CR, DEL, NEL (U+0085, a C1 control) and RLO (U+202E).
CONTROLS = {"\x1b": "⟨U+001B⟩", "\x07": "⟨U+0007⟩", "\r": "⟨U+000D⟩", "\x7f": "⟨U+007F⟩"}
CONTROLS |= {"\x85": "⟨U+0085⟩", "‮": "⟨U+202E⟩"}
MESSAGE = "A \x1b[31mred\x1b[0m bell\x07, a\rreturn, del\x7f, nel\x85 and ‮flip."
KEPT = "tab\there\nnew line, emoji 🦜 and a joiner 👩‍🔬."


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Fine."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.start()
    loop.handle(MESSAGE + " " + KEPT)
    output.clear()
    return loop, output, session, backend


def test_prompt_shows_each_control_character_in_the_pages_notation(chat):
    loop, output, session, _ = chat
    loop.handle("/prompt")
    shown = "".join(output)
    for char, notation in CONTROLS.items():
        assert char not in shown, repr(char)
        assert notation in shown, notation
    assert "\x1b[31mred" not in shown and "⟨U+001B⟩[31mred⟨U+001B⟩[0m" in shown


def test_newline_tab_printable_text_emoji_and_a_joiner_are_unchanged(chat):
    loop, output, _, _ = chat
    loop.handle("/prompt")
    assert KEPT in "".join(output)


def test_prompt_repr_and_what_was_sent_are_unchanged(chat):
    loop, output, session, backend = chat
    record = session.conversation.last_call
    assert MESSAGE in record["prompt_text"]  # logged as sent
    assert backend.calls[-1]["token_ids"] == record["prompt_ids"]
    assert session.tokenizer.decode(record["prompt_ids"]) == record["prompt_text"]
    loop.handle("/prompt repr")
    assert "\n".join(visible_lines(record["prompt_text"])) in "".join(output)


def test_the_json_prompt_result_holds_the_raw_text(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Fine."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    events = run(session, [line("profile", "text"), line("say", MESSAGE), line("prompt")])
    (result,) = [e["result"] for e in events if e.get("command") == "prompt"]
    assert MESSAGE in result["call"]["prompt_text"]


def test_help_says_how_control_characters_show():
    assert "⟨U+001B⟩" in COMMANDS["prompt"].help


def test_the_text_direction_controls_are_the_pages():
    """cli can't import the documents package (its boundary), so it spells the list out."""
    from clarity_chat.cli import SHOWN_CONTROLS
    from clarity_chat.documents.reading import SHOWN_BIDI

    assert SHOWN_BIDI.items() <= SHOWN_CONTROLS.items()
    assert {chr(i) for i in SHOWN_CONTROLS} - set(map(chr, SHOWN_BIDI)) == {
        chr(i) for i in [*range(0x20), *range(0x7F, 0xA0), 0x2028, 0x2029]
    } - {"\n", "\t"}


# --- v0.4.1 X1: the streamed reply and /history show them the same way ---------------------

# FakeBackend streams a word at a time, so ESC ends a chunk on its own and "[2J" starts the
# next: the escape is split across chunks, as a model's reply can split it.
REPLY = " \x1b [2J cleared, ‮flip and a\u2028line\u2029break."
RAW = ("\x1b", "‮", "\u2028", "\u2029")
NOTATIONS = ("⟨U+001B⟩", "⟨U+202E⟩", "⟨U+2028⟩", "⟨U+2029⟩")


@pytest.fixture
def replying(fetched_root, tiny_tokenizer):
    backend = FakeBackend([REPLY], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.start()
    loop.handle("/set max_tokens 200")  # the tiny tokenizer spends bytes on each character
    output.clear()
    return loop, output, session


def test_the_streamed_reply_shows_each_control_character(replying):
    loop, output, session = replying
    loop.handle("Hello")
    assert output[0] == " ⟨U+001B⟩ ", output  # ESC ended its chunk: "[2J" came in the next
    shown = "".join(output)
    for raw, notation in zip(RAW, NOTATIONS, strict=True):
        assert raw not in shown and notation in shown, notation
    assert session.conversation.last_call["output_text"].startswith(REPLY)  # logged raw


def test_history_shows_each_control_character(replying):
    loop, output, _ = replying
    loop.handle("Please \x1b[2J clear the screen.")
    output.clear()
    loop.handle("/history")
    shown = "".join(output)
    assert "Please ⟨U+001B⟩[2J clear" in shown
    for raw, notation in zip(RAW, NOTATIONS, strict=True):
        assert raw not in shown and notation in shown, notation


def test_prompt_shows_the_line_and_paragraph_separators(replying):
    loop, output, _ = replying
    loop.handle("one\u2028two\u2029three")
    output.clear()
    loop.handle("/prompt")
    assert "one⟨U+2028⟩two⟨U+2029⟩three" in "".join(output)


def test_the_json_text_events_stay_raw(fetched_root, tiny_tokenizer):
    backend = FakeBackend([REPLY], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    events = run(
        session, [line("profile", "text"), line("set", "max_tokens 200"), line("say", "Hello")]
    )
    assert "".join(e["text"] for e in events if e.get("event") == "text") == REPLY
