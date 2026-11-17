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
        chr(i) for i in [*range(0x20), *range(0x7F, 0xA0)]
    } - {"\n", "\t"}
