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


TYPES = {".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain"}


def choose(page, name: str, data: bytes) -> None:
    """The paperclip, then the system's file picker (Playwright sets its file), with the type
    a browser gives a file of that name (Checkpoint 3, should 9)."""
    with page.expect_file_chooser() as chooser:
        page.click("#clip")
    kind = TYPES.get(name[name.rfind(".") :], "application/octet-stream")
    chooser.value.set_files({"name": name, "mimeType": kind, "buffer": data})


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
    assert "long.md · lines 1–" in page.text_content("#chips")
    page.fill("#message", "Line one?")  # within max_tokens (16): what M1 leaves room for
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    assert page.text_content(".reply.last .msg").startswith(" A made-up reply")  # 16 tokens


def test_a_changed_range_shows_its_count_without_reading_again(clarity, page, monkeypatch):
    redirect, backend = clarity
    backend._n_ctx = 2000
    started = []
    real = reading.run_child
    monkeypatch.setattr(reading, "run_child", lambda *a: started.append(1) or real(*a))
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    assert started == [1]  # the server runs in this process: the child started once
    page.fill("#attach-from", "5")
    page.fill("#attach-to", "9")
    page.press("#attach-to", "Tab")
    page.wait_for_selector("#attach-body .range .ct:has-text('tokens')")
    until(page, lambda: "Excerpt: lines 5 to 9 of 200" in page.text_content("#attach-body"))
    assert page.get_attribute("[data-choice=part][role=radio]", "data-args").endswith(" lines 5-9")
    assert started == [1]  # and not again for the range


def test_reading_shows_with_cancel_and_cancel_says_it_stopped(clarity, page, monkeypatch):
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(30)")
    chat(page, redirect)
    choose(page, "slow.md", NOTES)
    page.wait_for_selector("#attach-sheet :text('Reading slow.md…')")
    shot(page, "C1-sheet-reading", "v031")
    page.click("#attach-sheet [data-action=attach-cancel]")
    page.wait_for_selector("#attach-sheet :text('Stopped reading slow.md.')")  # ruling 8
    until(  # the read itself stopped, not only the sentence (the whole-branch review, 16)
        page, lambda: page.evaluate("clarity.command('state')")["result"]["running"] is None
    )


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
    assert page.get_attribute("#chips", "role") == "list"  # Checkpoint 3, could 13
    page.click("[aria-label='Remove a.md']")
    until(page, lambda: "a.md" not in page.text_content("#chips"))
    page.click("#send")
    page.wait_for_selector("#send-waits:has-text('Type a message to send with b.md.')")  # F12


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
    page.wait_for_selector(".reply.last .tb")  # the reply is done: the why goes (page mutation)
    until(page, lambda: page.text_content("#clip-waits") == "")
    assert page.get_attribute("#clip", "aria-disabled") is None


# --- Checkpoint 3 (C1): Add, Cancel, the paperclip, the part's count and the radios


def part_args(page) -> str:
    return page.get_attribute("[data-choice=part][role=radio]", "data-args") or ""


def test_add_attaches_the_range_in_the_boxes_when_clicked_at_once(clarity, page):
    """Must 1: Add right after typing attaches what the boxes say, or waits."""
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    page.fill("#attach-to", "5")
    page.click("[data-action=attach-add]", force=True)  # at once; it may wait
    until(page, lambda: page.is_hidden("#attach-sheet") or part_args(page).endswith(" 1-5"))
    pending = page.evaluate("clarity.command('state')")["result"]["pending"]
    assert [a["range"] for a in pending] in ([], [[1, 5]])
    if not pending:
        page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    assert "long.md · lines 1–5 ·" in page.text_content("#chips")


def test_a_cleared_box_makes_add_wait(clarity, page):
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    assert page.get_attribute("[data-action=attach-add]", "aria-disabled") != "true"
    page.fill("#attach-to", "")
    assert page.get_attribute("[data-action=attach-add]", "aria-disabled") == "true"
    assert page.is_visible("#attach-why")


def test_the_paperclip_waits_during_a_replay(clarity, page):
    """Must 2, scenario A: a Replay runs, so the paperclip waits and says why."""
    redirect, backend = clarity
    chat(page, redirect)
    page.fill("#message", "Hi")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel:not([hidden]) .sec")
    backend.replies, backend.piece_delay = [" Slowly," * 10], 0.3
    page.click("#panel [data-action=replay]")
    page.wait_for_selector("#panel [role=status]:has-text('Replaying…')")
    choosers = []
    page.on("filechooser", lambda chooser: choosers.append(chooser))
    page.click("#clip", force=True)
    page.wait_for_selector("#clip-waits:has-text('Waits until the command is done.')")
    assert choosers == [] and page.is_hidden("#attach-sheet")


def test_cancel_during_the_upload_stops_it_and_says_so(clarity, page):
    """Must 2, scenario B: Cancel before the bytes are up aborts the upload; the sheet says
    it stopped, and no reading follows."""
    redirect, _ = clarity
    chat(page, redirect)
    held, failed = [], []
    page.route("**/upload?*", lambda route: held.append(route))  # held until the test lets go
    page.on("requestfailed", lambda request: failed.append(request.url))
    choose(page, "slow.md", NOTES)
    until(page, lambda: held, what="the upload sent")
    page.click("#attach-sheet [data-action=attach-cancel]")
    page.wait_for_selector("#attach-sheet :text('Stopped reading slow.md.')")
    until(page, lambda: any("/upload?" in url for url in failed), what="the upload aborted")
    try:
        held[0].continue_()
    except Exception:  # the page aborted it: nothing to continue
        pass
    page.wait_for_timeout(500)  # nothing to wait for: a read that went on shows now, if ever
    assert page.query_selector("#attach-sheet [role=radiogroup]") is None
    state = page.evaluate("clarity.command('state')")["result"]
    assert state["running"] is None and state["reading"] is None


def test_escape_while_reading_stops_the_read(clarity, page, monkeypatch):
    """Must 2, scenario C: Escape on "Reading…" cancels the read and says so."""
    redirect, _ = clarity
    stub(monkeypatch, "import time; time.sleep(30)")
    chat(page, redirect)
    choose(page, "slow.md", NOTES)
    page.wait_for_selector("#attach-sheet :text('Reading slow.md…')")
    until(page, lambda: page.evaluate("clarity.command('state')")["result"]["reading"])
    page.keyboard.press("Escape")
    page.wait_for_selector("#attach-sheet :text('Stopped reading slow.md.')")
    until(page, lambda: page.evaluate("clarity.command('state')")["result"]["running"] is None)


def test_choosing_the_part_counts_its_range_at_once(clarity, page):
    """Should 4: the part, chosen with nothing counted, is counted; Add then says why."""
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "notes.md", NOTES)
    ready(page)
    add = "[data-action=attach-add]"
    assert page.get_attribute(add, "aria-describedby") == "attach-whole-note"
    page.click("[data-choice=part][role=radio]")
    page.wait_for_selector("#attach-body .range .ct:has-text('tokens')")
    assert page.get_attribute(add, "aria-disabled") != "true"
    assert page.get_attribute(add, "aria-describedby") == "attach-part-note"


def test_the_paperclip_refuses_in_text_mode_before_the_picker(clarity, page):
    """Should 5: the session's sentence shows under the composer, and nothing is uploaded."""
    redirect, _ = clarity
    open_clarity(page, redirect)
    page.click("[data-profile=text]")
    page.wait_for_selector("#thread .intro")
    choosers = []
    page.on("filechooser", lambda chooser: choosers.append(chooser))
    page.click("#clip")
    sentence = "Attaching needs a chat-mode profile; text is in text mode."
    page.wait_for_selector(f"#notice .problem:has-text('{sentence}')")  # a refusal box (F12)
    assert choosers == [] and page.is_hidden("#attach-sheet")


def test_a_file_far_over_the_limit_is_refused_in_the_sheet(clarity, page, monkeypatch):
    """Should 8: the 413 comes while the browser is still sending; could 16: its title is
    the read refusals' one."""
    redirect, _ = clarity
    monkeypatch.setattr(web, "MAX_BYTES", 1024 * 1024)
    chat(page, redirect)
    choose(page, "big.pdf", b"%PDF-" + b"0" * (5 * 1024 * 1024))
    page.wait_for_selector("#attach-sheet [role=alert]")
    assert "big.pdf is too big" in page.text_content("#attach-sheet [role=alert]")
    assert page.text_content("#attach-title").strip() == "Not attached"


def test_show_extracted_text_opens_and_closes_the_preview(clarity, page):
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "notes.md", NOTES)
    ready(page)
    button = "[data-action=attach-preview]"
    assert page.is_hidden("#attach-preview")
    page.click(button)
    assert page.is_visible("#attach-preview")
    assert page.get_attribute(button, "aria-expanded") == "true"
    assert "Three cairns mark the ridge." in page.text_content("#attach-preview")
    assert page.text_content(button) == "Hide extracted text"
    page.click(button)
    assert page.is_hidden("#attach-preview") and page.text_content(button) == "Show extracted text"


def test_the_choices_are_one_tab_stop_and_the_arrows_choose(clarity, page):
    """Could 13: the radio group's pattern."""
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "notes.md", NOTES)
    ready(page)
    whole, part = "[data-choice=whole][role=radio]", "[data-choice=part][role=radio]"
    assert page.get_attribute(whole, "tabindex") == "0"
    assert page.get_attribute(part, "tabindex") == "-1"
    page.focus(whole)
    page.keyboard.press("ArrowDown")
    until(page, lambda: page.get_attribute(part, "aria-checked") == "true")
    page.wait_for_selector("#attach-body .range .ct:has-text('tokens')")
    assert page.evaluate("document.activeElement.dataset.choice") == "part"


def test_send_with_chips_waits_while_a_reply_is_written(clarity, page):
    """Could 13: an empty box with chips, during a reply, waits like any send."""
    redirect, backend = clarity
    chat(page, redirect)
    page.fill("#message", "Hi")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    choose(page, "a.md", b"Made-up a.\n")
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    backend.replies = [" One two three four five six seven eight nine ten."]
    backend.piece_delay = 0.4
    page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#writing:has-text('two')")
    page.focus("#message")
    page.keyboard.press("Enter")
    page.wait_for_selector("#send-waits:has-text('Waits until the reply is done.')")
    assert "Type a message" not in page.text_content("#notice")


def test_an_answer_for_a_range_typed_over_is_dropped(clarity, page):
    """Must 1 (page mutation): a slow answer for an older range never replaces the newer one."""
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    held, done = [], []
    page.route(
        "**/view/attach?*",
        lambda route: held.append(route) if "last=5" in route.request.url else route.continue_(),
    )
    page.on("requestfinished", lambda request: done.append(request.url))
    page.fill("#attach-to", "5")
    page.press("#attach-to", "Tab")
    until(page, lambda: held, what="the read for lines 1-5 sent")
    page.fill("#attach-to", "7")
    page.press("#attach-to", "Tab")
    until(page, lambda: part_args(page).endswith(" 1-7"), what="lines 1-7 counted")
    for route in held:
        route.continue_()
    until(page, lambda: sum("last=5" in url for url in done) == len(held), what="1-5 answered")
    page.wait_for_timeout(500)  # nothing to wait for: a late answer is drawn now, if ever
    assert part_args(page).endswith(" 1-7") and page.input_value("#attach-to") == "7"


# --- C2: your message's line, Edit's line, chips back after a failed send, Branches


def test_the_sent_message_shows_its_attachment_line_live_and_after_a_reload(clarity, page):
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "trail-notes.md", NOTES)
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    page.fill("#message", "What marks the ridge?")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    turn = page.evaluate("clarity.command('history')")["result"]["turns"][-2]
    tokens = turn["attachments"][0]["tokens"]
    line = f"Attached: trail-notes.md · whole · {tokens:,} tokens when attached"
    assert line in page.text_content(".turn.user .att")
    shot(page, "C2-thread-line", "v031")
    page.reload()
    page.wait_for_selector(".reply.last .tb")
    assert line in page.text_content(".turn.user .att")
    page.click(".reply.last [data-action=details]")  # for the design walkthrough
    page.wait_for_selector("#panel mark.added")
    shot(page, "C2-details-added", "v031")


def test_edit_shows_what_it_keeps(clarity, page):
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "trail-notes.md", NOTES)
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    page.fill("#message", "What marks the ridge?")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    page.click(".turn.user [data-action=edit]")
    page.wait_for_selector(".editbox:not([hidden])")
    keeps = "Keeps: trail-notes.md. An edit can't add or remove attachments."
    assert page.text_content(".editbox .att").strip() == keeps
    shot(page, "C2-edit-keeps", "v031")


def test_a_failed_send_brings_the_chips_back_and_offers_removing_one(clarity, page):
    redirect, backend = clarity
    backend._n_ctx = 2000
    chat(page, redirect)
    choose(page, "long.md", LONG)
    ready(page)
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#chips .chip")
    backend._n_ctx = 400  # the chat shrinks under the waiting chip: the send stops
    page.fill("#message", "Line one?")
    page.click("#send")
    page.wait_for_selector("[role=alert]:has-text('Nothing was sent.')")
    assert "remove an attachment and attach a shorter part" in page.text_content("#thread")
    page.wait_for_selector("#chips .chip")
    assert "long.md" in page.text_content("#chips") and page.input_value("#message") == "Line one?"
    shot(page, "C2-stop-chips-back", "v031")


def test_the_branches_button_says_whether_its_panel_is_open(clarity, page):
    redirect, backend = clarity
    chat(page, redirect)
    page.fill("#message", "Hi")
    page.click("#send")
    page.wait_for_selector(".reply.last .tb")
    backend.replies = [" Second one."]
    page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#branches-button:not([hidden])")
    assert page.get_attribute("#branches-button", "aria-expanded") == "false"
    page.click("#branches-button")
    page.wait_for_selector("#panel .bcard")
    assert page.get_attribute("#branches-button", "aria-expanded") == "true"
    page.click("#panel [data-action=panel-close]")
    until(page, lambda: page.get_attribute("#branches-button", "aria-expanded") == "false")
