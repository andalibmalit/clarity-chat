"""C1 (v0.3.1): the paperclip, the attach sheet and the chips, in a real browser (the plan's
C1 Checks; the frozen board V31Attach). Made-up documents only, built in memory.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import sys
import threading

import pytest

from clarity_chat import web
from clarity_chat.documents import reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.pdfs import encrypted_pdf, make_pdf
from tests.test_browser import free_port, open_clarity, shot, until
from tests.test_documents_extract import stub

pytestmark = pytest.mark.browser

NOTES = b"Made-up trail notes.\nThree cairns mark the ridge.\nNo water after the pass.\n"
LONG = "".join(f"Made-up line {n}.\n" for n in range(200)).encode()


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """The installed Chrome on the Mac, as test_browser.py drives it."""
    channel = {"channel": "chrome"} if sys.platform == "darwin" else {}
    return {**browser_type_launch_args, **channel}


@pytest.fixture
def clarity(fetched_root, chat_tokenizer):
    """Clarity on the chat profile, served by `web.serve`: (redirect, backend)."""
    reading._last = None
    backend = FakeBackend([" A made-up reply."] * 9, tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    port, opened, servers = free_port(), [], []
    thread = threading.Thread(
        target=web.serve,
        kwargs={
            "root": fetched_root,
            "port": port,
            "session_factory": lambda: session,
            "opener": opened.append,
            "on_ready": servers.append,
            "write": lambda text: None,
            "read_lines": False,
        },
        daemon=True,
    )
    thread.start()
    for _ in range(200):
        if servers and servers[0].started:
            break
        threading.Event().wait(0.05)
    yield opened[0], backend
    servers[0].should_exit = True
    thread.join(10)


def chat(page, redirect) -> None:
    open_clarity(page, redirect)
    page.click("[data-profile=chat]")
    page.wait_for_selector("#thread .intro")


def choose(page, name: str, data: bytes) -> None:
    """The paperclip, then the system's file picker (Playwright sets its file)."""
    with page.expect_file_chooser() as chooser:
        page.click("#clip")
    chooser.value.set_files({"name": name, "mimeType": "application/octet-stream", "buffer": data})


def ready(page) -> None:
    page.wait_for_selector("#attach-sheet [role=radiogroup]")


def test_a_file_that_fits_is_added_as_a_chip_and_sent_with_the_message(clarity, page):
    redirect, backend = clarity
    chat(page, redirect)
    choose(page, "trail-notes.md", NOTES)
    ready(page)
    assert page.text_content("#attach-title").strip() == "trail-notes.md"
    assert page.get_attribute("[data-choice=whole][role=radio]", "aria-checked") == "true"
    shot(page, "C1-sheet-fits", "v031")
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    assert page.is_hidden("#attach-sheet")
    assert "trail-notes.md · whole ·" in page.text_content("#chips")
    page.fill("#message", "What marks the ridge?")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    assert page.is_hidden("#chips")
    state = page.evaluate("clarity.command('state')")["result"]
    sent = state["turns"][-2]["attachments"]
    assert state["pending"] == [] and sent[0]["name"] == "trail-notes.md"


def test_the_preselected_part_then_one_line_then_send_gets_a_reply(clarity, page):
    """M1: the part the sheet preselects leaves max_tokens, so a first question fits."""
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    assert page.get_attribute("[data-choice=whole][role=radio]", "aria-disabled") == "true"
    assert page.get_attribute("[data-choice=part][role=radio]", "aria-checked") == "true"
    shot(page, "C1-sheet-part", "v031")
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    assert "long.md · lines 1-" in page.text_content("#chips")
    page.fill("#message", "Line one?")  # within max_tokens (16): what M1 leaves room for
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    assert page.text_content(".reply.last .msg").startswith(" A made-up reply")  # 16 tokens


def test_a_changed_range_shows_its_count_without_reading_again(clarity, page):
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    readings = page.evaluate("clarity.command('state')")["result"]["reading"]
    assert readings is None
    page.fill("#attach-from", "5")
    page.fill("#attach-to", "9")
    page.press("#attach-to", "Tab")
    page.wait_for_selector("#attach-body .range .ct:has-text('tokens')")
    until(page, lambda: "Excerpt: lines 5 to 9 of 200" in page.text_content("#attach-body"))
    assert page.get_attribute("[data-choice=part][role=radio]", "data-args").endswith(" lines 5-9")


def test_reading_shows_with_cancel_and_cancel_says_it_stopped(clarity, page, monkeypatch):
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(30)")
    chat(page, redirect)
    choose(page, "slow.md", NOTES)
    page.wait_for_selector("#attach-sheet :text('Reading slow.md…')")
    shot(page, "C1-sheet-reading", "v031")
    page.click("#attach-sheet [data-action=attach-cancel]")
    page.wait_for_selector("#attach-sheet :text('Stopped reading slow.md.')")  # ruling 8


def test_new_chat_with_this_document(clarity, page):
    redirect, backend = clarity
    backend._n_ctx = 1500
    chat(page, redirect)
    page.fill("#message", "Made-up words in a long message. " * 28)  # about 900 tokens
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    choose(page, "notes.md", LONG[:700])
    ready(page)
    page.click("[data-choice=new][role=radio]")
    assert page.text_content("[data-action=attach-add]").strip() == "Start new chat"
    first = page.evaluate("clarity.command('state')")["result"]["conversation_id"]
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    state = page.evaluate("clarity.command('state')")["result"]
    assert state["conversation_id"] != first and state["pending"][0]["name"] == "notes.md"


def test_remove_a_chip_and_send_with_chips_and_no_text(clarity, page):
    redirect, _ = clarity
    chat(page, redirect)
    for name in ("a.md", "b.md"):
        choose(page, name, f"Made-up {name}.\n".encode())
        ready(page)
        page.click("[data-action=attach-add]")
        page.wait_for_selector(f"#chips :text('{name}')")
    page.click("[aria-label='Remove a.md']")
    until(page, lambda: "a.md" not in page.text_content("#chips"))
    page.click("#send")
    page.wait_for_selector("#notice :text('Type a message to send with b.md.')")


@pytest.mark.parametrize(
    ("name", "data", "sentence"),
    [
        ("old-notes.txt", b"\xff\xfe made-up", "old-notes.txt isn't UTF-8 text. Save it as PDF"),
        ("scan.pdf", make_pdf(["", ""]), "scan.pdf has no text Clarity can read."),
        (
            "locked.pdf",
            encrypted_pdf(["Made-up."], user_password="made-up"),
            "locked.pdf needs a password.",
        ),
    ],
)
def test_a_refusal_shows_in_the_sheet(clarity, page, name, data, sentence):
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, name, data)
    page.wait_for_selector("#attach-sheet [role=alert]")
    assert sentence in page.text_content("#attach-sheet [role=alert]")


def test_a_file_too_big_is_refused_with_its_name(clarity, page, monkeypatch):
    redirect, _ = clarity
    monkeypatch.setattr(web, "MAX_BYTES", 10)
    chat(page, redirect)
    choose(page, "big.pdf", NOTES)
    page.wait_for_selector("#attach-sheet [role=alert]:has-text('big.pdf is too big')")


def test_the_paperclip_waits_while_a_reply_is_written(clarity, page):
    redirect, backend = clarity
    chat(page, redirect)
    backend.replies = [" One two three four five six seven eight nine ten."]
    backend.piece_delay = 0.4
    page.fill("#message", "Hi")
    page.click("#send")
    page.wait_for_selector("#writing:has-text('two')")
    assert page.get_attribute("#clip", "aria-disabled") == "true"
    page.click("#clip", force=True)  # Playwright waits on aria-disabled; a person needn't
    page.wait_for_selector("#clip-waits:has-text('Waits until the reply is done.')")
    assert page.is_hidden("#attach-sheet")
