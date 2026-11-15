"""B2 (v0.3.1): what the terminal prints for /attach, /detach and --attach. Of what the loop
prints, not a pty. Made-up documents only."""

from __future__ import annotations

import io
import json

import pytest

from clarity_chat import cli
from clarity_chat.cli import ChatLoop
from clarity_chat.commands import read_files
from clarity_chat.documents import FRAMING, reading
from clarity_chat.json_chat import JsonChat
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.pdfs import encrypted_pdf, make_pdf

NOTES = "Line one.\nLine two.\nLine three.\n"


@pytest.fixture
def notes(tmp_path):
    reading._last = None
    path = tmp_path / "notes.md"
    path.write_text(NOTES, encoding="utf-8")
    return path


def chat_loop(root, tokenizer, reads=(), attach=None, resume=None):
    backend = FakeBackend([" Hello!"] * 5, tokenizer=tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    output: list[str] = []
    inputs = iter(reads)

    def read(default: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    loop = ChatLoop(session, "chat", read, output.append, resume=resume, attach=attach)
    return loop, output, backend


@pytest.fixture
def loop(fetched_root, chat_tokenizer):
    chat, output, backend = chat_loop(fetched_root, chat_tokenizer)
    chat.start()
    output.clear()
    return chat, output, backend


def test_attach_says_it_s_reading_then_what_it_took_and_what_s_left(loop, notes):
    chat, output, _ = loop
    chat.handle(f"/attach {notes}")
    (a,) = chat.session.conversation.pending
    text = "".join(output)
    assert text.startswith("Reading notes.md…\n")
    assert f"Attached notes.md, whole, {a['tokens']:,} tokens. That leaves " in text
    assert " tokens for your message and the rest of the chat.\n" in text


def test_check_prints_the_sheet_s_numbers_and_sentences(loop, notes):
    chat, output, _ = loop
    chat.handle(f"/attach {notes} check")
    check = chat.commands.run("attach", f"{notes} check", lambda e: None)["check"]
    text = "".join(output)
    assert "notes.md: text, 3 lines." in text
    assert (
        f"It takes up {check['tokens_whole']:,} tokens of the context for the rest of the chat. "
        f"That leaves {check['left_whole']:,} tokens for your message and the rest of the chat."
    ) in text
    assert f"{check['free']:,} tokens free in this chat, after setting aside max_tokens" in text
    assert "The start that fits" not in text  # the whole fits (part 1's review, could 14)
    assert f"Framing line (Clarity's default): {FRAMING}" in text
    assert chat.session.conversation.pending == []


def test_a_refusal_prints_the_numbers_and_the_command_that_fits(loop, tmp_path):
    chat, output, backend = loop
    path = tmp_path / "long.md"
    path.write_text("".join(f"Made-up line {n}.\n" for n in range(200)), encoding="utf-8")
    backend._n_ctx = 600
    chat.handle(f"/attach {path}")
    text = "".join(output)
    assert "! long.md doesn't fit: it needs " in text
    assert f"for your message and the rest of the chat: /attach {path} lines 1-" in text


def test_a_line_before_the_prompt_lists_what_s_waiting(fetched_root, chat_tokenizer, notes):
    chat, output, _ = chat_loop(fetched_root, chat_tokenizer, reads=[f"/attach {notes}", "/help"])
    chat.run()
    text = "".join(output)
    tokens = chat.session.conversation.pending[0]["tokens"]
    waiting = f"Waiting for your next message: 1. notes.md, whole, {tokens:,} tokens.\n"
    assert text.count(waiting) == 2  # before the prompt after /attach, and after /help


@pytest.mark.parametrize("how", ["/quit", "Ctrl-D"])
def test_quitting_says_waiting_attachments_aren_t_kept(fetched_root, chat_tokenizer, notes, how):
    reads = [f"/attach {notes}", *(["/quit"] if how == "/quit" else [])]
    chat, output, _ = chat_loop(fetched_root, chat_tokenizer, reads=reads)
    chat.run()
    assert "".join(output).endswith(
        "Not kept: notes.md, waiting for your next message. An attachment is logged only "
        "with the message it's sent with.\n"
    )


def test_detach_prints_what_it_removed(loop, notes):
    chat, output, _ = loop
    chat.handle(f"/attach {notes}")
    output.clear()
    chat.handle("/detach 1")
    assert "".join(output) == "Detached notes.md. Nothing is waiting for your next message.\n"


# --- --attach


def test_dash_dash_attach_attaches_after_the_start(fetched_root, chat_tokenizer, notes):
    files = read_files([str(notes)])
    chat, output, _ = chat_loop(fetched_root, chat_tokenizer, attach=files)
    chat.start()
    text = "".join(output)
    assert text.index("Ready:") < text.index("Attached notes.md, whole,")
    assert [a["name"] for a in chat.session.conversation.pending] == ["notes.md"]


def test_dash_dash_attach_with_resume(fetched_root, chat_tokenizer, notes):
    first, _, _ = chat_loop(fetched_root, chat_tokenizer, reads=["Hi"])
    first.run()
    log = first.session.conversation.log.path.stem
    files = read_files([str(notes)])
    chat, output, _ = chat_loop(fetched_root, chat_tokenizer, attach=files, resume=log)
    chat.start()
    assert [a["name"] for a in chat.session.conversation.pending] == ["notes.md"]


@pytest.mark.parametrize(
    ("name", "data", "sentence"),
    [
        (
            "notes.md",
            b"\xff\xfe made-up",
            "notes.md isn't UTF-8 text. Save it as PDF or as UTF-8 text and attach that.",
        ),
        (
            "scan.pdf",
            make_pdf(["", ""]),
            "scan.pdf has no text Clarity can read. Its pages may be scanned images, which "
            "Clarity doesn't read. Use a PDF app's text recognition (OCR) to add the text, then "
            "attach that copy.",
        ),
        (
            "locked.pdf",
            encrypted_pdf(["Made-up page."], user_password="made-up"),
            "locked.pdf needs a password. Save a copy without one and attach that.",
        ),
    ],
)
def test_an_unattachable_dash_dash_attach_file_stops_before_the_model_starts(
    fetched_root, tmp_path, monkeypatch, capsys, name, data, sentence
):
    """S12, B2's Checks: unreadable, scanned or locked, refused before a 16 GB load."""
    bad = tmp_path / name
    bad.write_bytes(data)
    started = []
    monkeypatch.setattr(Session, "new_conversation", lambda *a, **k: started.append(a))
    code = cli.main(
        ["--root", str(fetched_root), "chat", "--profile", "chat", "--attach", str(bad)]
    )
    assert (code, started) == (1, [])
    assert capsys.readouterr().err == f"clarity: {sentence}\n"


def test_json_attaches_after_its_start_and_goes_on_after_a_refusal(
    fetched_root, chat_tokenizer, notes, tmp_path
):
    other = tmp_path / "long.md"
    other.write_text("".join(f"Made-up line {n}.\n" for n in range(2000)), encoding="utf-8")
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    stdout = io.StringIO()
    chat = JsonChat(session, io.BytesIO(b'{"command": "attach"}\n'), stdout)
    chat.commands.preread = read_files([str(other), str(notes)])
    chat.run(
        [{"command": "profile", "args": "chat"}],
        then=[{"command": "attach", "args": str(p)} for p in (other, notes)],
    )
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    done = [(e["event"], e.get("command")) for e in events if e["event"] in ("result", "error")]
    assert done == [
        ("result", "profile"),
        ("error", "attach"),
        ("result", "attach"),
        ("result", "attach"),
    ]
    assert [a["name"] for a in events[-1]["result"]["pending"]] == ["notes.md"]


def test_check_says_more_than_when_only_the_bound_was_taken(loop, tmp_path):
    """The security review's must 1: a file that surely doesn't fit isn't counted whole."""
    chat, output, backend = loop
    path = tmp_path / "long.md"
    path.write_text("".join(f"Made-up line {n}.\n" for n in range(20_000)), encoding="utf-8")
    chat.handle(f"/attach {path} lines 1-19000 check")
    text = "".join(output)
    assert "Doesn't fit: needs more than " in text
    assert "Lines 1 to 19000: Doesn't fit: needs more than " in text


def test_the_terminal_shows_a_name_s_control_characters_visibly():
    """The security review's should 5: a name printed in the terminal can't send it escape
    sequences, whatever reached it."""
    from clarity_chat.cli import format_pending

    a = {"n": 1, "name": "\x1b]0;made-up\x07notes\u202e.md", "how": "whole", "tokens": 5}
    line = format_pending([a])
    assert "\x1b" not in line and "\x07" not in line and "\u202e" not in line
    assert "\\x1b]0;made-up\\x07notes\\u202e.md" in line


def test_check_s_lines_come_from_the_one_table(loop, notes):
    """Should 5: format_check writes no sentence of its own."""
    from clarity_chat.documents import SENTENCES

    chat, output, _ = loop
    chat.handle(f"/attach {notes} lines 2-3 check")
    text = "".join(output)
    assert "notes.md: text, 3 lines." in text
    assert SENTENCES["framing"].format(origin="Clarity's default", framing="") in text
    assert "Lines 2 to 3: " in text and SENTENCES["excerpt"] in text
    assert "The start that fits" not in text  # the whole fits: no start lines (could 14)
