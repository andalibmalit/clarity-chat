"""B1 (v0.3.1): /attach and /detach, the same table for the terminal, --json and the page.
Made-up documents only."""

from __future__ import annotations

import threading
import time

import pytest

import clarity_chat.documents as documents
from clarity_chat.commands import Commands
from clarity_chat.documents import FRAMING, DocumentRefused, reading
from clarity_chat.profiles import ConfigError
from tests.test_documents_extract import stub
from tests.test_session_attach import chat_session

RANGE = {"unit", "range", "tokens", "tokens_at_least", "left", "marker"}
CHECK = {
    "name", "kind", "bytes", "unit", "total", "empty_units", "encrypted", "tokens_whole",
    "tokens_whole_at_least", "free", "fits", "left_whole", "start_that_fits", "preselected",
    "chosen", "free_new_chat", "fits_new_chat", "already_in_chat", "control_tokens",
    "has_closing_tag", "framing", "framing_origin", "text",
}  # fmt: skip
NOTES = "Line one.\nLine two.\nLine three.\n"


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None  # the extraction cache: each test reads its own files
    return chat_session(fetched_root, chat_tokenizer)


@pytest.fixture
def notes(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(NOTES, encoding="utf-8")
    return path


def run(c: Commands, name: str, args: str = "", events: list | None = None) -> dict:
    return c.run(name, args, (events.append if events is not None else lambda e: None))


def summary(record: dict, n: int) -> dict:
    return {**{k: v for k, v in record.items() if k != "text"}, "n": n}


# --- attach, check, the list and detach


def test_attach_adds_the_whole_file_and_gives_what_s_free_and_left(chat, notes):
    session, _ = chat
    c = Commands(session)
    result = run(c, "attach", str(notes))
    (record,) = session.conversation.pending
    assert record["how"] == "whole" and record["text"] == NOTES and record["name"] == "notes.md"
    assert result == {
        "attached": summary(record, 1),
        "pending": [summary(record, 1)],
        "free": result["free"],
        "left": result["free"] - record["tokens"],
    }
    assert result["free"] > 0


def test_attach_a_range_adds_an_excerpt(chat, notes):
    session, _ = chat
    result = run(Commands(session), "attach", f"{notes} lines 2-3")
    assert result["attached"]["how"] == "excerpt" and result["attached"]["range"] == [2, 3]
    assert session.conversation.pending[0]["text"] == "Line two.\nLine three.\n"


def test_a_path_with_spaces_is_everything_before_the_range(chat, tmp_path):
    session, _ = chat
    path = tmp_path / "trail notes check.md"
    path.write_text(NOTES, encoding="utf-8")
    result = run(Commands(session), "attach", f"{path} lines 1-1")
    assert result["attached"]["name"] == "trail notes check.md"


def test_check_reads_and_counts_and_attaches_nothing(chat, notes):
    session, _ = chat
    session.say("Hi")
    result = run(Commands(session), "attach", f"{notes} check")
    check = result["check"]
    assert set(result) == {"check"} and set(check) == CHECK
    assert session.conversation.pending == []
    assert (check["name"], check["kind"], check["unit"], check["total"]) == (
        "notes.md",
        "text",
        "line",
        3,
    )
    assert check["text"] == NOTES and check["encrypted"] is False
    assert check["fits"] is True and check["left_whole"] == check["free"] - check["tokens_whole"]
    assert set(check["start_that_fits"]) == RANGE and check["start_that_fits"]["range"] == [1, 3]
    assert set(check["preselected"]) == RANGE and check["chosen"] is None
    assert check["free_new_chat"] > check["free"] and check["fits_new_chat"] is True
    assert (check["framing"], check["framing_origin"]) == (FRAMING, "default")
    assert check["already_in_chat"] is False and check["control_tokens"] == []


def test_check_with_a_range_gives_that_range_s_numbers(chat, notes):
    session, _ = chat
    chosen = run(Commands(session), "attach", f"{notes} lines 2-2 check")["check"]["chosen"]
    assert chosen["range"] == [2, 2] and chosen["unit"] == "lines"
    assert chosen["marker"] == "Excerpt: lines 2 to 2 of 3"


def test_a_whole_file_that_doesn_t_fit_is_refused_with_the_numbers_and_the_start_that_fits(
    chat, tmp_path
):
    session, backend = chat
    path = tmp_path / "long.md"
    path.write_text("".join(f"Made-up line {n}.\n" for n in range(200)), encoding="utf-8")
    backend._n_ctx = 2000  # counted exactly: the bound alone (a quarter) would fit
    c = Commands(session)
    check = run(c, "attach", f"{path} check")["check"]
    assert check["tokens_whole"] is not None
    k, roomy = check["start_that_fits"]["range"][1], check["preselected"]["range"][1]
    with pytest.raises(DocumentRefused) as e:
        run(c, "attach", str(path))
    assert str(e.value) == (
        f"long.md doesn't fit: it needs {check['tokens_whole']:,} tokens and "
        f"{check['free']:,} are free in this chat. The largest start found that still leaves "
        "at least 16 tokens (your max_tokens) for your message and the rest of the chat: "
        f"/attach {path} lines 1-{roomy}. Lines 1 to {k} fit."
    )
    assert session.conversation.pending == []


def test_when_not_even_the_first_unit_fits_the_refusal_says_so(chat, notes):
    session, backend = chat
    backend._n_ctx = 120
    with pytest.raises(DocumentRefused) as e:
        run(Commands(session), "attach", str(notes))
    assert str(e.value).endswith(" Not even its first line fits.")


def test_attach_alone_lists_and_detach_removes(chat, notes, tmp_path):
    session, _ = chat
    c = Commands(session)
    other = tmp_path / "other.md"
    other.write_text("Other.\n", encoding="utf-8")
    run(c, "attach", str(notes))
    run(c, "attach", str(other))
    first, second = session.conversation.pending
    assert run(c, "attach") == {"pending": [summary(first, 1), summary(second, 2)]}
    assert run(c, "detach", "1") == {
        "detached": [summary(first, 1)],
        "pending": [summary(second, 1)],
    }
    with pytest.raises(ConfigError) as e:
        run(c, "detach", "2")
    assert str(e.value) == "No attachment 2. /attach lists them."
    assert run(c, "detach") == {"detached": [summary(second, 1)], "pending": []}


@pytest.mark.parametrize("args", ["0", "x", "1 2", "-1"])
def test_detach_takes_one_number_from_the_list(chat, notes, args):
    session, _ = chat
    c = Commands(session)
    run(c, "attach", str(notes))
    with pytest.raises(ConfigError) as e:
        run(c, "detach", args)
    assert str(e.value) == f"No attachment {args}. /attach lists them."


# --- the page: upload IDs, never paths


def test_from_the_page_a_path_is_refused(chat, notes):
    session, _ = chat
    with pytest.raises(ConfigError) as e:
        run(Commands(session, uploads={}), "attach", str(notes))
    assert str(e.value) == "From the page, attach a file by choosing it."


def test_from_the_page_an_upload_id_attaches_once(chat):
    session, _ = chat
    uploads = {"upload-0123456789ab": (NOTES.encode(), "notes.md")}
    c = Commands(session, uploads=uploads)
    assert run(c, "attach", "upload-0123456789ab check")["check"]["name"] == "notes.md"
    assert "upload-0123456789ab" in uploads  # check keeps it: Add comes after
    assert run(c, "attach", "upload-0123456789ab lines 1-2")["attached"]["range"] == [1, 2]
    assert uploads == {}  # used: dropped
    with pytest.raises(ConfigError) as e:
        run(c, "attach", "upload-0123456789ab")
    assert str(e.value) == "That file isn't uploaded any more. Choose it again."


def test_the_terminal_reads_paths_whatever_its_name(chat, tmp_path):
    """A file named like an upload ID is a path in the terminal."""
    session, _ = chat
    path = tmp_path / "upload-0123456789ab"
    path.write_text(NOTES, encoding="utf-8")
    result = run(Commands(session), "attach", str(path))
    assert result["attached"]["name"] == "upload-0123456789ab"


# --- state, history, events


def test_state_lists_what_s_waiting_and_what_s_being_read(chat, notes, monkeypatch):
    session, _ = chat
    c = Commands(session)
    seen = []
    real = documents.extract

    def extract(data, name, cancel=None, on_read=None):
        def told():
            assert on_read is not None
            on_read()
            seen.append(run(c, "state")["reading"])

        return real(data, name, cancel, told)

    monkeypatch.setattr(documents, "extract", extract)
    run(c, "attach", str(notes))
    assert seen == [{"name": "notes.md"}]
    state = run(c, "state")
    assert state["reading"] is None
    assert state["pending"] == [summary(session.conversation.pending[0], 1)]


def test_history_lists_a_message_s_attachments_without_their_text(chat, notes):
    session, _ = chat
    c = Commands(session)
    run(c, "attach", str(notes))
    record = session.conversation.pending[0]
    run(c, "say", "What's in it?")
    user = next(t for t in run(c, "history")["turns"] if t["role"] == "user")
    assert user["attachments"] == [summary(record, 1)]
    assert "attachments" not in next(t for t in run(c, "history")["turns"] if t["role"] != "user")


def test_the_turn_s_record_event_carries_its_attachments(chat, notes):
    session, _ = chat
    c = Commands(session)
    run(c, "attach", str(notes))
    events: list = []
    run(c, "say", "What's in it?", events)
    turn = next(
        e["record"]
        for e in events
        if e["event"] == "record" and e["record"]["type"] == "turn"
        and e["record"]["role"] == "user"
    )  # fmt: skip
    assert turn["attachments"][0]["text"] == NOTES


def test_attach_queues_behind_a_command_and_cancel_stops_its_reading(chat, notes, monkeypatch):
    session, _ = chat
    c = Commands(session)
    assert not c.at_once_now("attach")
    stub(monkeypatch, "import time; time.sleep(30)")
    errors: list[BaseException] = []

    def attach():
        try:
            run(c, "attach", str(notes))
        except BaseException as e:  # noqa: BLE001 - the test reads it
            errors.append(e)

    thread = threading.Thread(target=attach)
    thread.start()
    deadline = time.monotonic() + 10
    while run(c, "state")["reading"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert run(c, "cancel") == {"cancelled": True, "command": "attach"}
    thread.join(10)
    assert [str(e) for e in errors] == ["Stopped reading notes.md."]


def test_any_error_while_reading_is_a_refusal_not_a_stopped_session(chat, notes, monkeypatch):
    """M5: every exception from the documents package reaches the user as a refusal."""
    session, _ = chat

    def broken(data, name, cancel=None, on_read=None):
        raise RuntimeError("made-up failure")

    monkeypatch.setattr(documents, "extract", broken)
    with pytest.raises(ConfigError) as e:
        run(Commands(session), "attach", str(notes))
    assert str(e.value) == "Clarity couldn't read notes.md (RuntimeError: made-up failure)."


def test_a_text_mode_chat_refuses_attach(chat, notes):
    session, _ = chat
    session.new_conversation("text")
    with pytest.raises(ConfigError) as e:
        run(Commands(session), "attach", str(notes))
    assert str(e.value) == "Attaching needs a chat-mode profile; text is in text mode."


def test_attach_reads_a_tilde_as_the_home_folder(chat, tmp_path, monkeypatch):
    """The security review's could 8: `/attach ~/notes.md` said "Can't open notes.md"."""
    session, _ = chat
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notes.md").write_text(NOTES, encoding="utf-8")
    assert run(Commands(session), "attach", "~/notes.md")["attached"]["name"] == "notes.md"
