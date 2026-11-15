"""Part 1's whole-branch review (v0.3.1): its musts, the behaviour behind shoulds 3, 4, 7
and 9, and coulds 10 and 11. Made-up documents only."""

from __future__ import annotations

import pytest

from clarity_chat import cli
from clarity_chat.commands import Commands, read_files
from clarity_chat.documents import DocumentRefused, reading
from clarity_chat.profiles import ConfigError
from tests.pdfs import encrypted_pdf
from tests.test_cli_attach import chat_loop
from tests.test_session_attach import chat_session

NOTES = "Line one.\nLine two.\nLine three.\n"
LONG = "".join(f"Made-up line {n}.\n" for n in range(200))


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


@pytest.fixture
def notes(tmp_path):
    reading._last = None
    path = tmp_path / "notes.md"
    path.write_text(NOTES, encoding="utf-8")
    return path


def readings(c: Commands, args: str) -> list[dict]:
    """The `reading` events one command emits, and /state's `reading` during each."""
    seen: list[dict] = []

    def emit(event):
        if event["event"] == "reading":
            seen.append(event)

    c.run("attach", args, emit)
    return seen


# --- must 1: `reading` only when a file is read


def test_a_second_attach_of_the_same_file_reads_nothing(chat, notes):
    session, _ = chat
    c = Commands(session)
    assert [e["name"] for e in readings(c, f"{notes} check")] == ["notes.md"]
    assert readings(c, str(notes)) == []  # the cache answers: no child, no "Reading"
    assert c.run("state", "", lambda e: None)["reading"] is None


def test_a_check_then_an_attach_of_one_upload_reads_once(chat):
    session, _ = chat
    uploads = {"upload-0123456789ab": (NOTES.encode(), "notes.md")}
    c = Commands(session, uploads=uploads)
    assert len(readings(c, "upload-0123456789ab lines 1-2 check")) == 1
    assert readings(c, "upload-0123456789ab lines 2-3 check") == []
    assert readings(c, "upload-0123456789ab lines 1-2") == []


# --- must 2 and #44: the rooms follow max_tokens as each chat has it


def test_a_new_chat_s_room_uses_the_profile_s_max_tokens(chat, notes):
    session, _ = chat
    c = Commands(session)
    c.run("say", "Hi", lambda e: None)
    before = c.run("attach", f"{notes} check", lambda e: None)["check"]
    c.run("set", "max_tokens 200", lambda e: None)
    after = c.run("attach", f"{notes} check", lambda e: None)["check"]
    assert after["free"] == before["free"] - (200 - 16)  # this chat: /set counts
    c.run("new", "", lambda e: None)
    fresh = c.run("attach", f"{notes} check", lambda e: None)["check"]
    assert after["free_new_chat"] == fresh["free"] == before["free_new_chat"]


# --- should 3: the terminal says what a start dropped


@pytest.mark.parametrize("command", ["/new", "/profile chat", "/resume LOG"])
def test_the_terminal_says_what_a_start_dropped(fetched_root, chat_tokenizer, notes, command):
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    loop.handle("Hi")
    loop.handle(f"/attach {notes}")
    output.clear()
    log = loop.session.conversation.log.path.stem
    loop.handle(command.replace("LOG", log))
    assert "Not kept: notes.md, waiting for your next message." in "".join(output)


# --- should 4: --attach says what it's reading


def test_dash_dash_attach_says_it_s_reading_before_the_start(
    fetched_root, notes, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "cmd_chat", lambda *a, **k: print("Starting profile chat …") or 0)
    assert (
        cli.main(["--root", str(fetched_root), "chat", "--profile", "chat", "--attach", str(notes)])
        == 0
    )
    assert capsys.readouterr().out == "Reading notes.md…\nStarting profile chat …\n"


def test_dash_dash_attach_uses_what_it_read_before_the_start(fetched_root, chat_tokenizer, notes):
    """#45: the file is read once; gone by the start, it's still attached, unread again."""
    files = read_files([str(notes)])
    notes.unlink()
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer, attach=files)
    loop.start()
    assert "Reading" not in "".join(output)
    assert [a["name"] for a in loop.session.conversation.pending] == ["notes.md"]


# --- could 10: a repeated --attach is refused, never collapsed


def test_a_repeated_dash_dash_attach_is_refused(notes):
    with pytest.raises(ConfigError) as e:
        read_files([str(notes), str(notes)])
    assert str(e.value) == f"--attach names {notes} twice. Name each file once."


# --- should 9 and could 11: the refusal names the start that leaves room for a message


def test_the_terminal_refusal_names_the_start_that_leaves_room(chat, tmp_path):
    session, backend = chat
    path = tmp_path / "long.md"
    path.write_text(LONG, encoding="utf-8")
    backend._n_ctx = 2000
    c = Commands(session)
    check = c.run("attach", f"{path} check", lambda e: None)["check"]
    start, roomy = check["start_that_fits"]["range"][1], check["preselected"]["range"][1]
    assert roomy < start
    with pytest.raises(DocumentRefused) as e:
        c.run("attach", str(path), lambda e: None)
    assert str(e.value).endswith(
        " The largest start found that still leaves at least 16 tokens (your max_tokens) for "
        f"your message and the rest of the chat: /attach {path} lines 1-{roomy}. Lines 1 to "
        f"{start} fit."
    )


def test_the_page_s_refusal_names_no_terminal_command(chat):
    session, backend = chat
    backend._n_ctx = 2000
    c = Commands(session, uploads={"upload-0123456789ab": (LONG.encode(), "long.md")})
    check = c.run("attach", "upload-0123456789ab check", lambda e: None)["check"]
    start, roomy = check["start_that_fits"]["range"][1], check["preselected"]["range"][1]
    with pytest.raises(DocumentRefused) as e:
        c.run("attach", "upload-0123456789ab", lambda e: None)
    assert "/attach" not in str(e.value)
    assert str(e.value).endswith(
        " The largest start found that still leaves at least 16 tokens (your max_tokens) for "
        f"your message and the rest of the chat: lines 1 to {roomy}. Lines 1 to {start} fit."
    )


# --- should 7: behaviour no test needed


def test_detach_changes_what_s_waiting(chat, notes, tmp_path):
    session, _ = chat
    c = Commands(session)
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / name).write_text(f"{name} made-up.\n", encoding="utf-8")
        c.run("attach", str(tmp_path / name), lambda e: None)
    c.run("detach", "2", lambda e: None)
    assert [a["name"] for a in session.conversation.pending] == ["a.md", "c.md"]
    c.run("detach", "", lambda e: None)
    assert session.conversation.pending == []


def test_an_excerpt_that_doesn_t_fit_gets_no_suggestion(chat, tmp_path):
    """#41: a start from line 1 answers a different question from lines 5-190."""
    session, backend = chat
    path = tmp_path / "long.md"
    path.write_text(LONG, encoding="utf-8")
    backend._n_ctx = 2000
    with pytest.raises(DocumentRefused) as e:
        Commands(session).run("attach", f"{path} lines 5-190", lambda e: None)
    assert str(e.value).startswith("long.md doesn't fit: it needs ")
    assert str(e.value).endswith(" are free in this chat.")


def test_check_lists_what_the_sheet_warns_about(chat, tmp_path):
    """#46 to #48: encrypted, already in this chat, control tokens and </document>."""
    session, _ = chat
    c = Commands(session)
    locked = tmp_path / "locked.pdf"
    locked.write_bytes(encrypted_pdf(["Made-up page."]))
    assert c.run("attach", f"{locked} check", lambda e: None)["check"]["encrypted"] is True
    marked = tmp_path / "marked.md"
    marked.write_text("Made-up </s> and </document> text.\n", encoding="utf-8")
    c.run("attach", str(marked), lambda e: None)
    check = c.run("attach", f"{marked} check", lambda e: None)["check"]
    assert check["already_in_chat"] is True
    assert (check["control_tokens"], check["has_closing_tag"]) == (["</s>"], True)
