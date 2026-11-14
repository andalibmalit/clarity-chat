"""The page in a real browser: the secret, the reload, the live tab, a page on another port,
and a reply joined after a reload mid-reply (v0.3.0 item B6); the shell's flows (C1).

Run with `uv run pytest -m browser` (needs `uv sync --group browser`; on the Mac it drives
the installed Chrome, on CI Playwright's Chromium). The server is uvicorn on a free port with
the fake backend; nothing reaches a model.
"""

from __future__ import annotations

import http.server
import re
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from clarity_chat import web
from clarity_chat.session import Session
from tests.conftest import TEXT_PROFILE
from tests.fakes import FakeBackend
from tests.test_web_shell import OTHER_MODEL

pytestmark = pytest.mark.browser
SCREENS = Path(__file__).resolve().parent.parent / "notes" / "research" / "v030-screens"


def shot(page, name: str, item: str = "c1", full_page: bool = True) -> None:
    """A screenshot for the design review, kept in a local-only folder: notes/ is gitignored,
    so CI, which has no such folder, saves none."""
    if SCREENS.is_dir():
        (SCREENS / item).mkdir(exist_ok=True)
        page.screenshot(path=SCREENS / item / f"{name}.png", full_page=full_page)


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    channel = {"channel": "chrome"} if sys.platform == "darwin" else {}
    return {**browser_type_launch_args, **channel}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def clarity(fetched_root, tiny_tokenizer):
    """Clarity served by `web.serve` on a free port: (origin, the redirect file, backend)."""
    long = TEXT_PROFILE.replace("max_tokens = 16", "max_tokens = 200")
    (fetched_root / "profiles" / "long.toml").write_text(long, encoding="utf-8")
    with (fetched_root / "models.toml").open("a", encoding="utf-8") as f:  # one not fetched
        f.write(OTHER_MODEL)
    far = TEXT_PROFILE.replace('model = "tiny"', 'model = "other"')
    (fetched_root / "profiles" / "far.toml").write_text(far, encoding="utf-8")
    backend = FakeBackend([], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
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
    yield f"http://127.0.0.1:{port}", opened[0], backend
    servers[0].should_exit = True
    thread.join(10)


def open_clarity(page, redirect: Path) -> None:
    page.goto(redirect.as_uri())
    page.wait_for_selector("body[data-page=ready]")


def test_the_link_opens_the_page_and_leaves_the_address_bar(clarity, page):
    origin, redirect, _ = clarity
    open_clarity(page, redirect)
    assert page.url == f"{origin}/" and "key=" not in page.url
    assert page.is_visible("#app")
    assert not redirect.exists()  # gone at the first request with the secret


def test_a_reload_keeps_working(clarity, page):
    _, redirect, _ = clarity
    open_clarity(page, redirect)
    page.reload()
    page.wait_for_selector("body[data-page=ready]")


def test_a_tab_without_the_secret_is_told_where_the_link_is(clarity, context):
    origin, _, _ = clarity
    page = context.new_page()
    page.goto(f"{origin}/")
    page.wait_for_selector("body[data-page=no-key]")
    assert page.text_content("[data-for=no-key] p") == (
        "Open Clarity from the window where you started it: press Enter there."
    )
    shot(page, "A3-no-secret")


def test_a_secret_from_an_earlier_start_is_cleared(clarity, page):
    origin, _, _ = clarity
    page.goto(f"{origin}/#key=from-an-earlier-start")
    page.wait_for_selector("body[data-page=no-key]")
    assert page.evaluate("sessionStorage.getItem('clarity-key')") is None


def test_a_page_on_another_port_can_neither_read_nor_act(clarity, page, tmp_path):
    origin, redirect, _ = clarity
    chat_on(page, redirect)  # Clarity is running, with a chat and its secret in that tab
    turns = len(page.evaluate("clarity.command('state')")["result"]["turns"])
    attacker = tmp_path / "attack.html"
    attacker.write_text(
        f"""<!doctype html><script>
        const say = JSON.stringify({{command: "say", args: "hi", tab: "x"}});
        const tries = [
          fetch("{origin}/view/thread", {{headers: {{Authorization: "Bearer guess"}}}}),
          fetch("{origin}/command", {{method: "POST", headers: {{"Content-Type":
            "application/json"}}, body: say}}),
          fetch("{origin}/events?tab=x"),
        ];
        // A "simple" request needs no preflight: no-cors, text/plain. Its answer can't be
        // read (opaque), so what counts is that it did nothing (security review, 10).
        const simple = fetch("{origin}/command", {{method: "POST", mode: "no-cors",
          headers: {{"Content-Type": "text/plain"}}, body: say}}).catch(() => null);
        Promise.allSettled(tries.map(t => t.then(r => r.text()))).then(async results => {{
          await simple;
          document.title = results.map(r => r.status === "fulfilled" ? "READ:" + r.value
            : "blocked").join("|");
        }});
        </script>""",
        encoding="utf-8",
    )
    handler = lambda *a: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp_path))  # noqa: E731
    other = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), handler)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        evil = page.context.new_page()
        evil.goto(f"http://127.0.0.1:{other.server_address[1]}/attack.html")
        evil.wait_for_function("document.title.includes('|')")
        results = evil.title().split("|")
    finally:
        other.shutdown()
    # Nothing came back that a page could read: blocked by CORS, or refused without the secret.
    assert all(r == "blocked" or "Open Clarity from the window" in r for r in results), results
    # And nothing happened: the no-cors text/plain POST added no turn.
    assert len(page.evaluate("clarity.command('state')")["result"]["turns"]) == turns


def test_a_reply_reloaded_mid_way_joins_exactly_with_an_emoji(clarity, page):
    _, redirect, backend = clarity
    open_clarity(page, redirect)
    page.evaluate("clarity.command('profile', 'long')")
    reply = " One 🙂 two 🙂🙂 three four five six seven eight nine ten"
    backend.replies, backend.piece_delay = [reply], 0.25
    # A function returning nothing, so evaluate doesn't wait for the command's promise.
    page.evaluate("() => { clarity.command('say', 'Hi'); }")
    page.wait_for_timeout(900)

    def slow(route):  # the reloaded page's /state answers late, so pieces are buffered
        time.sleep(0.6)  # meanwhile, some at positions /state's reply_end already covers
        route.continue_()

    page.route("**/command", slow)
    page.reload()  # mid-reply: the new page reads /state, then joins the stream
    page.wait_for_selector("body[data-last-call]", timeout=20_000)
    assert page.text_content("#writing") == reply


def test_trusted_types_let_only_swap_put_html_in_the_page(clarity, page):
    """The CSP's Trusted Types: swap() works through its policy; any other sink throws."""
    _, redirect, _ = clarity
    open_clarity(page, redirect)
    swapped = page.evaluate(
        """() => clarity.view("thread").then(html => {
             clarity.swap(document.getElementById("writing"), html);
             return document.querySelector("#writing ol.thread") !== null;
           })"""
    )
    assert swapped is True
    refused = page.evaluate(
        """() => { try { document.body.innerHTML = "<b>x</b>"; return "allowed"; }
                   catch (e) { return e.name; } }"""
    )
    assert refused == "TypeError"


# --- the shell (C1)


def hold_start(backend, seconds: float) -> None:
    """Make the backend's model start take `seconds`, so the page shows it starting."""
    real = backend.start

    def slow(*args):
        time.sleep(seconds)
        return real(*args)

    backend.start = slow


def focused(page) -> str:
    return page.evaluate("document.activeElement.id || document.activeElement.dataset.profile")


def test_with_no_profile_chosen_a_pick_starts_the_model(clarity, page):
    _, redirect, backend = clarity
    open_clarity(page, redirect)
    # B1: the picker is open, focus on its first row, and the message box waits.
    assert page.is_visible("#picker-menu")
    assert page.text_content("#picker-menu .mhead") == "Choose a profile to start"
    assert focused(page) == "chat"
    assert page.is_disabled("#message") and page.is_disabled("#send")
    shot(page, "B1-no-profile")
    hold_start(backend, 1.5)
    page.click("[data-profile=text]")
    # B2: starting, from the status events; you can type, and Send and the picker wait.
    page.wait_for_selector("#thread [role=status]")
    assert page.text_content("#thread [role=status]") == "Starting text…"
    assert page.get_attribute("#picker-button", "aria-disabled") == "true"
    assert page.text_content("#picker-waits") == "Waits until the model has started."
    assert page.is_enabled("#message")
    # Send waits with its reason (aria-disabled), reachable, as the picker does (rule 5).
    assert page.get_attribute("#send", "aria-disabled") == "true"
    assert page.get_attribute("#send", "disabled") is None
    assert page.text_content("#send-waits") == "Waits until the model has started."
    shot(page, "B2-starting")
    # B5: ready.
    page.wait_for_selector("#thread .intro")
    assert page.text_content("#picker-button .nm") == "text · tiny"
    assert page.text_content("#title-text") == "New chat"
    assert page.is_enabled("#send") and focused(page) == "message"
    shot(page, "B5-new-chat")
    # C3: in a chat the picker ticks the profile; Escape closes it and returns focus.
    page.click("#picker-button")
    assert page.get_attribute("[data-profile=text]", "aria-checked") == "true"
    assert focused(page) == "text"
    shot(page, "C3-picker-in-a-chat")
    page.keyboard.press("Escape")
    assert page.is_hidden("#picker-menu") and focused(page) == "picker-button"


def test_a_refused_start_shows_the_backends_sentence(clarity, page):
    _, redirect, _ = clarity
    open_clarity(page, redirect)
    assert page.text_content("[data-profile=far] .d2") == "runs on llama.cpp · not fetched"
    shot(page, "C6-not-fetched")
    page.click("[data-profile=far]")
    page.wait_for_selector("#notice [role=alert]")
    assert page.text_content("#notice [role=alert]") == (
        "! far didn’t start. Model 'other' is not fetched. In a new terminal window, in "
        "Clarity's folder, run: uv run clarity fetch other. Then try again."
    )
    assert page.text_content("#picker-button .nm") == "Choose a profile"
    assert page.is_disabled("#message")
    shot(page, "B3-not-fetched")


def test_with_nothing_fetched_the_first_screen_says_so(clarity, page, fetched_root):
    _, redirect, _ = clarity
    (fetched_root / "models" / "tiny" / "tiny-bf16.gguf").unlink()
    open_clarity(page, redirect)
    assert page.text_content("#picker-none").startswith("No model is on this computer yet")
    shot(page, "B1-nothing-fetched")


def test_a_start_warning_shows_whole(clarity, page):
    _, redirect, backend = clarity
    warning = (
        "The chat template never reads template_kwargs ['enable_thinkng'], so they change "
        "nothing. Check their names against the template text in the prompt."
    )
    backend.start_warnings = [warning]
    open_clarity(page, redirect)
    page.click("[data-profile=text]")
    page.wait_for_selector("#thread .intro")
    assert page.text_content("#thread .problem") == f"! {warning}"
    shot(page, "B6-start-warning")


def test_the_picker_waits_while_a_reply_is_written(clarity, page):
    _, redirect, backend = clarity
    open_clarity(page, redirect)
    page.click("[data-profile=text]")
    page.wait_for_selector("#thread .intro")
    backend.replies, backend.piece_delay = [" A slow reply, one piece at a time."], 0.3
    page.evaluate("() => { clarity.command('say', 'Hi'); }")
    page.wait_for_selector("#picker-button[aria-disabled=true]")
    assert page.text_content("#picker-waits") == "Waits until the reply is done."
    page.click("#picker-button", force=True)  # Playwright won't click aria-disabled otherwise
    assert page.is_hidden("#picker-menu")  # it waits: it doesn't open
    shot(page, "C5-picker-while-writing")


def test_quit_shows_the_stopped_page(clarity, page):
    _, redirect, _ = clarity
    open_clarity(page, redirect)
    page.keyboard.press("Escape")  # the first screen's picker
    page.click("#clarity-button")
    assert focused(page) == "new-chat"
    shot(page, "A6-clarity-menu")
    page.click("[data-action=quit]")
    page.wait_for_selector("body[data-page=stopped]")
    assert page.text_content("#stopped-plain") == "Clarity has stopped."
    assert page.is_hidden("#stopped-error")
    shot(page, "A5-stopped")


def test_an_unexpected_error_shows_it_whole(clarity, page):
    _, redirect, backend = clarity
    open_clarity(page, redirect)
    page.evaluate("clarity.command('profile', 'text')")

    def boom():
        raise RuntimeError("made-up: out of memory")

    backend.on_call = boom
    page.evaluate("() => { clarity.command('say', 'Hi'); }")
    page.wait_for_selector("body[data-page=stopped]")
    assert page.text_content("#stopped-crash") == "Clarity stopped after an unexpected error."
    assert page.text_content("#stopped-error") == "RuntimeError: made-up: out of memory"
    shot(page, "A5-unexpected-error")


def test_about_shows_the_version_and_help_opens_the_docs_at_this_commit(clarity, page):
    _, redirect, _ = clarity
    open_clarity(page, redirect)
    state = page.evaluate("clarity.command('state')")["result"]
    page.keyboard.press("Escape")
    page.click("#clarity-button")
    commit = state["clarity"]["commit"]
    docs = f"https://github.com/andalibmalit/clarity-chat/tree/{commit}/docs"
    assert page.get_attribute("#help", "href") == docs
    page.click("[data-action=about]")
    page.wait_for_selector("#about[open] dd")
    assert page.text_content("#about dd") == state["clarity"]["version"]
    assert commit in page.text_content("#about")
    shot(page, "P2-about")
    page.keyboard.press("Escape")
    assert page.is_hidden("#about") and focused(page) == "clarity-button"


def test_a_second_tab_offers_use_it_here_and_takes_over(clarity, context):
    _, redirect, _ = clarity
    found = re.search(r'url=([^"]+)"', redirect.read_text(encoding="utf-8"))
    assert found
    link = found.group(1)
    first, second = context.new_page(), context.new_page()
    first.goto(link)
    first.wait_for_selector("body[data-page=ready]")
    second.goto(link)
    second.wait_for_selector("body[data-page=not-live]")
    assert second.text_content("[data-for=not-live] h1") == "Clarity is open in another tab."
    shot(second, "A4-another-tab")
    second.click("#use-here")
    second.wait_for_selector("body[data-page=ready]")
    first.wait_for_selector("body[data-page=not-live]")


# --- the thread (C2)


def chat_on(page, redirect, profile: str = "text") -> None:
    """Open Clarity and start a profile: `long` for replies longer than 16 tokens."""
    open_clarity(page, redirect)
    page.click(f"[data-profile={profile}]")
    page.wait_for_selector("#thread .intro")


def say(page, text: str) -> None:
    page.fill("#message", text)
    page.click("#send")


def test_send_writes_and_stop_keeps_the_part_that_arrived(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    backend.replies = [" One two three four five six seven eight nine ten eleven twelve."]
    backend.piece_delay = 0.3
    say(page, "Hi")
    # E1: writing, in a status region; Stop where Send was; the picker waits.
    page.wait_for_selector("#writing:has-text('two')")  # not wait_for_function: the CSP
    assert page.text_content("#writing-status") == "Writing…"
    assert page.is_visible("#stop") and page.is_hidden("#send")
    assert page.text_content("#stop-label") == "Stop"
    assert page.get_attribute("#picker-button", "aria-disabled") == "true"
    assert page.get_attribute("#writing", "aria-live") is None  # the reply isn't announced
    assert focused(page) == "message"  # sent from the box, focus stays there (V3Access O3)
    shot(page, "E1-writing", "c2")
    page.click("#stop")
    # E3: the part that arrived is kept, and the facts line says who stopped it.
    page.wait_for_selector(".reply.last .facts")
    assert page.text_content(".reply.last .facts") == "stopped by you"
    state = page.evaluate("clarity.command('state')")["result"]
    assert page.text_content(".reply.last .msg") == state["turns"][-1]["text"]
    assert page.is_visible("#send") and page.is_hidden("#writing-box")
    assert page.get_attribute("#picker-button", "aria-disabled") is None
    assert page.evaluate("document.activeElement.closest('.reply.last .tb') !== null")
    # The status region stays in the page, empty, so the next "Writing…" is announced.
    assert page.evaluate("document.getElementById('writing-status').closest('[hidden]')") is None
    shot(page, "E3-stopped-by-you", "c2")


def test_several_replies_say_which_is_being_written(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.replies = [" A first one, " * 5, " A second one, " * 5, " A third."]
    backend.piece_delay = 0.4
    page.evaluate("() => { clarity.command('regen', '3'); }")
    page.wait_for_selector("#writing-status:has-text('Writing reply 2 of 3')")
    assert page.text_content("#stop-label") == "Stop the rest"
    assert page.locator("#writing-dots span").count() == 3
    shot(page, "E2-several-replies", "c2")
    page.wait_for_selector("#stop", state="hidden", timeout=20_000)  # done: the page knew
    assert page.text_content(".reply.last [data-action=replies]") == "4 replies"


def test_a_failed_message_comes_back_to_the_box(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend.fail_with = "A made-up failure."
    say(page, "Hi there")
    page.wait_for_selector("#thread [data-failed]")
    assert page.text_content("#thread [data-failed] [role=alert]") == "! Failed: A made-up failure."
    assert page.is_visible("#thread [data-failed] [data-action=details]")
    assert page.input_value("#message") == "Hi there"
    assert page.locator("text=Try again").count() == 0
    shot(page, "E5-failed-message", "c2")


def test_a_failed_regenerate_offers_try_again(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last [data-action=regen]")
    backend.fail_with = "A made-up failure."
    page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#notice [role=alert]")
    assert page.text_content("#notice [role=alert]") == "! Failed: A made-up failure."
    failed = page.get_attribute("#notice [data-call]", "data-call")  # the failed call's own
    record = page.evaluate(f"clarity.command('prompt', '{failed}')")["result"]["call"]
    assert record["error"] == "BackendError: A made-up failure."
    assert page.is_visible("#notice [data-action=details]")
    shot(page, "E5-failed-regenerate", "c2")
    backend.fail_with = None
    page.click("#notice [data-action=regen]")
    page.wait_for_selector(".reply.last [data-action=replies]:has-text('2 replies')")


def test_when_the_server_stopped_start_the_model_again(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.stopped = True
    say(page, "Again")
    page.wait_for_selector("#thread [data-action=restart]")
    assert page.is_enabled("#send")  # no wait without a reason: Send fails with the sentence
    shot(page, "E6-server-stopped", "c2")
    hold_start(backend, 1.5)
    page.click("#thread [data-action=restart]")
    page.wait_for_selector("#thread .starting")
    assert page.text_content("#thread .starting [role=status]") == "Starting text…"
    assert page.input_value("#message") == "Again"  # your text stays in the box
    shot(page, "E6-starting-again", "c2")
    page.wait_for_selector("#thread .starting", state="detached")
    page.click("#send")
    page.wait_for_selector(".reply.last .msg:has-text('ok')")


def test_a_message_that_doesnt_fit_stays_in_the_box(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend._n_ctx = 160
    say(page, "x" * 200)
    page.wait_for_selector("#thread [data-failed]")
    sentence = page.text_content("#thread [data-failed] [role=alert]")
    assert sentence.startswith("! The prompt is ") and "Failed" not in sentence
    assert page.input_value("#message") == "x" * 200
    shot(page, "E7-doesnt-fit", "c2")


def test_stopped_before_any_text_says_so(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend.cancel_after = 0
    say(page, "Hi")
    page.wait_for_selector("#thread [data-failed][role=status]")
    assert page.text_content("#thread [data-failed]") == "Stopped before the reply started."
    assert page.input_value("#message") == "Hi"
    shot(page, "E4-stopped-before-text", "c2")


def test_edit_your_message_then_step_through_its_versions(clarity, page, context):
    _, redirect, _ = clarity
    context.grant_permissions(["clipboard-read", "clipboard-write"])
    chat_on(page, redirect)
    say(page, "Hi")
    page.click(".turn.user [data-action=edit]")
    assert page.input_value("#edit-text") == "Hi" and focused(page) == "edit-text"
    shot(page, "D2-editing", "c2")
    page.fill("#edit-text", "Hello")
    page.click("[data-edit] button[type=submit]")
    page.wait_for_selector(".msgbar .n")
    assert page.text_content(".msgbar .n [aria-hidden]") == "2 / 2"
    assert page.text_content(".msgbar .n .vh") == "Version 2 of 2"
    page.click(".reply.last [data-action=copy]")
    page.wait_for_selector(".reply.last .copied:has-text('Copied')")
    assert page.evaluate("navigator.clipboard.readText()") == page.text_content(".reply.last .msg")
    shot(page, "D1-D3-counter-and-toolbar", "c2")
    page.click("[aria-label='Previous version of your message']")
    page.wait_for_selector(".msgbar .n [aria-hidden]:has-text('1 / 2')")
    assert page.text_content(".turn.user .msg") == "Hi"


def test_marks_and_a_note_on_a_reply(clarity, page):
    _, redirect, _ = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.click(".reply.last .bm")
    assert page.is_visible(".reply.last .pop") and "Not marked" in page.text_content(".pop .ph")
    page.click(".pop [data-action=mark]")
    page.wait_for_selector(".pop .lab:has-text('interesting')")
    page.click(".pop [data-action=note-edit]")
    page.fill(".pop textarea", "Short and kind.")
    page.click(".pop [data-note] button[type=submit]")
    page.wait_for_selector(".pop .noted")
    assert page.text_content(".pop .noted") == "Note: “Short and kind.”"
    turn = page.evaluate("clarity.command('state')")["result"]["turns"][-1]
    assert [m["label"] for m in turn["marks"]] == ["interesting"]
    assert turn["notes"][-1]["text"] == "Short and kind."
    shot(page, "D6-marks-and-note", "c2")
    page.keyboard.press("Escape")
    assert page.is_hidden(".pop") and page.evaluate(
        "document.activeElement.classList.contains('bm')"
    )
    page.click(".reply.last .bm")
    page.click("[aria-label='Remove the mark interesting']")
    page.wait_for_selector(".pop .ph:has-text('Not marked')")


def test_the_facts_line_shows_a_failed_check(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend.result_flags = ["check_failed:no_cache_reuse"]
    say(page, "Hi")
    page.wait_for_selector(".reply.last .facts")
    assert (
        page.text_content(".reply.last .facts .warn")
        == "! 1 check failed: no_cache_reuse · Details"
    )
    shot(page, "D5-facts-line", "c2")


def test_after_a_reload_mid_reply_the_page_learns_the_reply_ended(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    reply = " Slowly," * 12 + " done."
    backend.replies, backend.piece_delay = [reply], 0.3
    page.evaluate("() => { clarity.command('say', 'Hi'); }")
    page.wait_for_timeout(600)
    page.reload()
    page.wait_for_selector("#stop", state="visible")
    page.wait_for_selector("#stop", state="hidden", timeout=20_000)
    assert page.get_attribute("#picker-button", "aria-disabled") is None
    assert page.text_content(".reply.last .msg") == reply


# --- Checkpoint 3's findings


def test_a_tab_showing_another_tab_stays_there_when_the_live_one_goes(clarity, context):
    """Finding 1: a tab showing A4 takes over only by "Use it here", never by polling (the
    live tab reloading opens the same window, less often; closing it holds it open)."""
    _, redirect, _ = clarity
    found = re.search(r'url=([^"]+)"', redirect.read_text(encoding="utf-8"))
    assert found
    first, second = context.new_page(), context.new_page()
    first.goto(found.group(1))
    first.wait_for_selector("body[data-page=ready]")
    second.goto(found.group(1))
    second.wait_for_selector("body[data-page=not-live]")
    first.close()
    second.wait_for_timeout(4000)  # longer than the page's reconnect backoff
    assert second.evaluate("document.body.dataset.page") == "not-live"


def test_a_failed_edit_opens_the_edit_again_with_its_text(clarity, page):
    """Finding 4: Send then retries the edit; the message box stays as it was."""
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.click(".turn.user [data-action=edit]")
    backend.fail_with = "A made-up failure."
    page.fill("#edit-text", "Hello")
    page.click("[data-edit] button[type=submit]")
    page.wait_for_selector("#notice [role=alert]")
    assert page.is_visible("#edit-text") and page.input_value("#edit-text") == "Hello"
    assert page.input_value("#message") == ""


def test_a_view_that_fails_and_a_refused_switch_say_so(clarity, page):
    """Finding 5: a redraw that fails, or a refused /switch, shows the sentence."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.click(".turn.user [data-action=edit]")
    page.fill("#edit-text", "Hello")
    page.click("[data-edit] button[type=submit]")
    page.wait_for_selector(".msgbar .n")
    page.evaluate("document.querySelector('[data-switch]').dataset.switch = 'turn-000000000000'")
    page.click("[data-switch]")
    page.wait_for_selector("#notice [role=alert]")
    assert "turn-000000000000" in page.text_content("#notice [role=alert]")
    page.route("**/view/thread", lambda route: route.fulfill(status=503, body="A made-up refusal."))
    page.evaluate("clarity.refresh()")
    page.wait_for_selector("#notice [role=alert]:has-text('A made-up refusal.')")


def test_after_a_reload_mid_reply_the_page_reads_only_state_until_it_ends(clarity, page):
    """Finding 6: the wait for a reply this page didn't send reads /state, not every view."""
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    reply = " Slowly," * 14 + " done."
    backend.replies, backend.piece_delay = [reply], 0.3
    page.evaluate("() => { clarity.command('say', 'Hi'); }")
    page.wait_for_timeout(600)
    views: list[str] = []
    page.on("request", lambda r: views.append(r.url) if "/view/thread" in r.url else None)
    page.reload()
    page.wait_for_selector("#stop", state="visible")
    page.wait_for_selector("#stop", state="hidden", timeout=20_000)
    page.wait_for_selector(".reply.last .tb")
    assert len(views) <= 2, views  # the page load's draw and the one at the reply's end


def test_a_start_warning_is_alerted_once_not_on_every_redraw(clarity, page):
    """Finding 7: a redraw doesn't announce the same sentence again."""
    _, redirect, backend = clarity
    backend.start_warnings = ["A made-up warning."]
    chat_on(page, redirect)
    assert page.get_attribute("#thread .problem p", "role") == "alert"
    page.evaluate("clarity.refresh()")
    page.wait_for_timeout(500)
    assert page.text_content("#thread .problem p") == "! A made-up warning."
    assert page.get_attribute("#thread .problem p", "role") is None


def test_a_second_message_is_written_from_its_own_start(clarity, page):
    """Finding 9, a guard for C2's fix to B6: the reply so far is reset for each message."""
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    backend.replies = [" First reply here.", " Second reply, written slowly, word by word."]
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.piece_delay = 0.3
    say(page, "Again")
    page.wait_for_selector("#writing:has-text('Second reply,')")
    assert page.text_content("#writing").startswith(" Second reply,")


# --- replies and Compare (C3)


def wait_for(page, selector: str, timeout: int = 30_000) -> None:
    """wait_for_selector, printing what the page shows if it times out (a flake's evidence)."""
    try:
        page.wait_for_selector(selector, timeout=timeout)
    except Exception:
        for label, script in (
            ("button", "document.querySelector('.reply.last [data-action=replies]')?.textContent"),
            ("notice", "document.getElementById('notice').textContent"),
            ("page state", "JSON.stringify(clarity.state && clarity.state.running)"),
            ("replies", "clarity.command('compare').then(a => a.result.replies.length)"),
            ("running now", "clarity.command('state').then(a => JSON.stringify(a.result.running))"),
            ("head now", "clarity.command('state').then(a => a.result.head)"),
            ("panel current", "document.querySelector('#panel .bcard.cur')?.dataset.turn"),
            ("last reply", "document.querySelector('.reply.last')?.dataset.turn"),
        ):
            print("DUMP", label, page.evaluate(script))
        raise


def open_replies(page) -> None:
    page.click(".reply.last [data-action=replies]")
    page.wait_for_selector(".listpop")


def test_more_replies_from_the_pop_out_of_one_reply(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    open_replies(page)
    # F5: one reply, so only More replies, prefilled 3; focus on the box.
    assert page.text_content(".listpop .lh span") == "1 reply to this message"
    assert page.input_value("#more-n") == "3" and focused(page) == "more-n"
    assert page.locator(".listpop .lrow").count() == 0
    # Under the reply's toolbar, not over the reply (design walkthrough, finding 17).
    toolbar = page.locator(".reply.last .tb").bounding_box()
    box = page.locator(".listpop").bounding_box()
    assert toolbar and box and box["y"] >= toolbar["y"] + toolbar["height"] - 1
    page.fill("#more-n", "2")
    assert page.text_content("#more-make") == "Make 2"  # "Make N", as Again's button
    page.fill("#more-n", "3")
    shot(page, "F5-one-reply", "c3")
    page.keyboard.press("Escape")
    assert (
        page.is_hidden(".listpop")
        and page.evaluate("document.activeElement.dataset.action") == "replies"
    )
    open_replies(page)
    backend.replies = [" Second, " * 4, " Third, " * 4, " Fourth."]
    backend.piece_delay = 0.3
    page.click(".listpop button[type=submit]")
    page.wait_for_selector("#writing-status:has-text('Writing reply 2 of 3')")
    assert page.text_content("#stop-label") == "Stop the rest"
    page.wait_for_selector("#stop", state="hidden", timeout=20_000)
    assert page.text_content(".reply.last [data-action=replies]") == "4 replies"


def make_three(page, backend) -> None:
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.replies = [" Second one.", " Third one."]
    page.evaluate("clarity.command('regen', '2')")
    page.evaluate("clarity.refresh()")
    page.wait_for_selector(".reply.last [data-action=replies]:has-text('3 replies')")


def test_the_rows_and_showing_another_reply(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    make_three(page, backend)
    open_replies(page)
    compared = page.evaluate("clarity.command('compare')")["result"]
    rows = page.locator(".listpop .lrow")
    assert rows.count() == 3
    for i, reply in enumerate(compared["replies"]):
        assert rows.nth(i).locator(".n").text_content() == f"reply {reply['k']} of 3"
        assert rows.nth(i).locator(".df").text_content() == f"seed {reply['seed']}"
    assert page.text_content(".listpop .lrow.cur .act") == "You are here"
    shot(page, "F2-three-replies", "c3")
    rows.nth(0).locator(".go").click()
    page.wait_for_selector(".reply.last .msg:has-text('ok')")  # reply 1's text
    assert page.is_hidden(".listpop")


def test_again_with_the_same_seeds_after_a_change_then_compare(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    make_three(page, backend)
    page.evaluate("clarity.command('set', 'temperature 0.3')")
    page.evaluate("clarity.refresh()")
    page.wait_for_selector(".reply.last .tb")
    open_replies(page)
    assert page.text_content(".listpop .again span") == (
        "Again with the same 3 seeds, now at temperature 0.3"
    )
    shot(page, "F3-again", "c3")
    backend.replies = [" ok, then more.", " Second one!", " Third one."]
    page.click(".listpop [data-action=again]")
    wait_for(page, ".reply.last [data-action=replies]:has-text('6 replies')")
    open_replies(page)
    assert "You set temperature to 0.3" in page.text_content(".listpop .divider")
    assert page.locator(".listpop [data-action=again]").count() == 0
    shot(page, "F1-six-replies", "c3")
    page.click(".listpop [data-action=compare]")
    page.wait_for_selector("#compare:not([hidden]) .cmp-grid")
    assert focused(page) == "compare-back" and page.is_hidden("#threadarea")
    assert page.locator("#compare .split").count() >= 1
    shot(page, "K2-matched-seeds", "c3")
    page.set_viewport_size({"width": 400, "height": 800})
    shot(page, "K4-narrow", "c3", full_page=False)
    page.set_viewport_size({"width": 1280, "height": 720})
    page.keyboard.press("Escape")
    assert page.is_hidden("#compare") and page.is_visible("#threadarea")
    assert page.evaluate("document.activeElement.dataset.action") == "replies"


def test_compare_cards_when_only_the_seed_differs_and_when_both_do(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    make_three(page, backend)
    open_replies(page)
    page.click(".listpop [data-action=compare]")
    page.wait_for_selector("#compare .cmp-cards")
    assert page.text_content("#compare .cmp-lead") == (
        "These replies differ only by seed, so where they part says nothing about a setting."
    )
    shot(page, "K1-seed-only", "c3")
    page.click("#compare-back")
    page.evaluate("clarity.command('set', 'temperature 0.3')")
    page.evaluate("clarity.command('regen')")
    page.evaluate("clarity.refresh()")
    wait_for(page, ".reply.last [data-action=replies]:has-text('4 replies')")
    open_replies(page)
    page.click(".listpop [data-action=compare]")
    page.wait_for_selector("#compare .cmp-cards")  # advice in the lead's style, not an error
    assert "differ in seed and in temperature" in page.text_content("#compare .cmp-lead")
    assert page.query_selector("#compare .problem") is None
    shot(page, "K3-seed-and-setting", "c3")


def test_an_earlier_reply_s_pop_out_lists_without_more_replies(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    make_three(page, backend)
    say(page, "And then?")
    page.wait_for_selector(".reply.early [data-action=replies]")
    page.click(".reply.early [data-action=replies]")
    page.wait_for_selector(".listpop")
    assert page.locator(".listpop .lrow").count() == 3
    assert page.locator("#more-n").count() == 0
    assert "continued, 2 more turns" in page.text_content(".listpop .lrow.cur")
    shot(page, "F1-earlier-fork", "c3")


# --- Settings (C4a)


def open_settings(page) -> None:
    page.click("#composer-row [data-action=settings]")
    page.wait_for_selector("#panel:not([hidden]) [data-setting=temperature]")


def test_settings_change_a_value_go_back_and_a_refusal(clarity, page):
    _, redirect, _ = clarity
    chat_on(page, redirect)
    open_settings(page)
    assert focused(page) == "panel-title"
    assert page.input_value("#set-temperature") == "0.6"
    shot(page, "I1-settings", "c4a")
    page.fill("#set-temperature", "0.3")
    page.press("#set-temperature", "Enter")
    page.wait_for_selector("[data-setting=temperature] .tag:has-text('chat')")
    settings = page.evaluate("clarity.command('settings')")["result"]["settings"]
    assert settings["temperature"] == {
        "value": 0.3,
        "origin": "chat",
        "profile": {"value": 0.6, "origin": "profile"},
    }
    page.wait_for_selector("#composer-row .ch:has-text('1 changed')")
    assert page.text_content("[data-setting=temperature] [data-set]") == "Back to 0.6 (profile)"
    shot(page, "I3-changed", "c4a")
    # A refused value: the backend's sentence under the row; the box keeps what you typed.
    page.fill("#set-top_p", "1.5")
    page.press("#set-top_p", "Enter")
    page.wait_for_selector("[data-setting=top_p] .refused:not([hidden])")
    assert page.text_content("[data-setting=top_p] .refused") == (
        "top_p must be a number from 0 to 1, got 1.5."
    )
    assert page.text_content("[data-setting=top_p] .note") == (
        "Not applied: the next reply still uses 0.95 (profile)."
    )
    assert page.input_value("#set-top_p") == "1.5"
    shot(page, "I7-refused", "c4a")
    page.click("[data-setting=temperature] [data-set]")
    page.wait_for_selector("[data-setting=temperature] .tag:has-text('profile')")
    assert page.locator("#panel [data-action=reset-all]").count() == 0
    # A slider sends /set when it's released (a change), not on every step.
    page.click("#panel details.more summary")
    page.locator("[data-setting=min_p] input[type=range]").fill("0.05")
    page.wait_for_selector("[data-setting=min_p] .tag:has-text('chat')")
    assert (
        page.evaluate("clarity.command('settings')")["result"]["settings"]["min_p"]["value"] == 0.05
    )
    assert page.is_visible("[data-setting=top_k]")  # More settings stayed open
    page.keyboard.press("Escape")
    assert page.is_hidden("#panel")
    assert page.evaluate("document.activeElement.dataset.action") == "settings"


def test_settings_wait_while_a_reply_is_written(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    open_settings(page)
    backend.replies, backend.piece_delay = [" Slowly," * 10], 0.3
    page.evaluate("() => { clarity.command('say', 'Hi'); }")
    page.wait_for_selector("#panel .waits")
    assert page.text_content("#panel .waits") == "Changes wait until this reply is done."
    assert page.get_attribute("#set-temperature", "aria-disabled") == "true"
    shot(page, "I8-waiting", "c4a")
    page.wait_for_selector("#panel .waits", state="detached", timeout=20_000)
    assert page.get_attribute("#set-temperature", "aria-disabled") is None


# --- Details (C4b)


def test_details_on_a_reply_and_its_replay(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel:not([hidden]) .sec")
    call = page.evaluate("clarity.command('prompt')")["result"]["call"]
    assert page.text_content("#panel-title") == "Details · reply 1 of 1"
    assert focused(page) == "panel-title"
    assert page.is_visible("#panel pre[data-pane=text]")
    shot(page, "G1-what-the-model-got", "c4b")
    page.click("#panel .seg [data-pane=ids]")
    assert page.is_hidden("#panel pre[data-pane=text]")
    assert page.text_content("#panel pre[data-pane=ids]") == " ".join(map(str, call["prompt_ids"]))
    backend.replies, backend.piece_delay = [" Slowly," * 10], 0.3
    page.click("#panel [data-action=replay]")
    page.wait_for_selector("#panel [role=status]:has-text('Replaying…')")
    assert focused(page) is None and page.evaluate("document.activeElement.textContent") == "Stop"
    shot(page, "G4-replaying", "c4b")
    page.wait_for_selector("#panel .result", timeout=20_000)
    replays = page.evaluate(f"clarity.command('prompt', '{call['call_id']}')")["result"]["replays"]
    assert len(replays) == 1 and not replays[0]["identical"]
    assert "replayed 1×, different" in page.text_content("#panel")
    shot(page, "G4-replayed", "c4b")
    # Stop: the replay is cancelled, and the backend's sentence says nothing was logged.
    backend.replies = [" Slowly," * 10]
    page.click("#panel [data-action=replay]")
    page.wait_for_selector("#panel [data-action=replay-stop]")
    page.click("#panel [data-action=replay-stop]")
    page.wait_for_selector("#panel .refused:not([hidden])", timeout=20_000)
    assert "Replay stopped" in page.text_content("#panel .refused")
    page.wait_for_timeout(500)  # every redraw after the replay has landed
    assert focused(page) == "replay"
    page.keyboard.press("Escape")
    assert page.is_hidden("#panel")


def test_details_on_a_failed_call(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend.fail_with = "A made-up failure."
    say(page, "Hi")
    page.wait_for_selector("#thread [data-failed] [data-action=details]")
    page.click("#thread [data-failed] [data-action=details]")
    page.wait_for_selector("#panel:not([hidden]) .sec")
    assert page.text_content("#panel-title") == "Details · failed call"
    assert "! Failed: A made-up failure." in page.text_content("#panel")
    assert page.locator("#panel [data-action=replay]").count() == 0
    shot(page, "G8-failed-call", "c4b")


# --- Probabilities (C4c)


def test_probabilities_shade_the_reply_and_a_token_shows_its_chance(clarity, page):
    _, redirect, backend = clarity
    tokenizer = backend.tokenizer
    chat_on(page, redirect, "long")
    backend.replies = [" Café, olé!"]
    ids = tokenizer.encode(" Café, olé!") + [tokenizer.tok.eos_token_id]
    (z,) = tokenizer.encode("z")
    backend.logprobs = {
        "view": "raw logits, before any sampler",
        "n": 2,
        "tokens": [
            {"id": t, "logprob": -0.05 * (i % 40), "top": [[t, -0.05 * (i % 40)], [z, -3.0]]}
            for i, t in enumerate(ids)
        ],
    }
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel [data-key=probs]")
    assert "2 likeliest per token" in page.text_content("#panel [data-key=probs] summary")
    page.click("#panel [data-key=probs] summary")
    page.wait_for_selector("#thread .msg.shaded .tok")
    call = page.evaluate("clarity.command('prompt')")["result"]["call"]
    shaded = page.text_content("#thread .msg.shaded")
    assert shaded.startswith(call["output_text"])  # the reply, then its end token
    page.click("#thread .msg.shaded .tok >> nth=2")
    info = page.text_content("#panel .tokinfo:not([hidden])")
    assert "chance" in info and "The likeliest tokens here" in info
    page.keyboard.press("ArrowRight")
    assert page.get_attribute("#thread .msg.shaded .tok >> nth=3", "aria-pressed") == "true"
    shot(page, "H1-shaded", "c4c")
    page.click("#panel [data-key=probs] summary")  # closing the section clears the shading
    page.wait_for_selector("#thread .msg.shaded", state="detached")


def test_tokens_that_cant_be_placed_are_listed_and_can_be_picked(clarity, page):
    """Checkpoint 4, finding 5: the panel's list answers clicks and arrow keys."""
    _, redirect, backend = clarity
    tokenizer = backend.tokenizer
    chat_on(page, redirect, "long")
    backend.replies = [" Hello"]
    (z,) = tokenizer.encode("z")
    backend.logprobs = {
        "view": "raw logits, before any sampler",
        "n": 2,
        "tokens": [{"id": z, "logprob": -0.5, "top": [[z, -0.5], [z, -3.0]]}] * 3,
    }
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel [data-key=probs]")
    page.click("#panel [data-key=probs] summary")
    page.click("#panel .toklist .tok >> nth=1")
    assert page.get_attribute("#panel .toklist .tok >> nth=1", "aria-pressed") == "true"
    assert "chance" in page.text_content("#panel .tokinfo:not([hidden])")
    page.keyboard.press("ArrowRight")
    assert page.get_attribute("#panel .toklist .tok >> nth=2", "aria-pressed") == "true"
    assert page.query_selector("#thread .msg.shaded") is None
    shot(page, "H1-unplaced", "c4c")


def test_shading_leaves_a_message_being_edited_hidden(clarity, page):
    """Checkpoint 4, finding 17: shade() unhid every hidden message, Edit's too."""
    _, redirect, backend = clarity
    tokenizer = backend.tokenizer
    chat_on(page, redirect, "long")
    backend.replies = [" Hello"]
    ids = tokenizer.encode(" Hello")
    backend.logprobs = {
        "view": "raw logits, before any sampler",
        "n": 1,
        "tokens": [{"id": t, "logprob": -0.5, "top": [[t, -0.5]]} for t in ids],
    }
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.click(".turn.user [data-action=edit]")
    assert page.is_hidden(".turn.user .msg")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel [data-key=probs]")
    page.click("#panel [data-key=probs] summary")
    page.wait_for_selector("#thread .msg.shaded .tok")
    page.click("#panel [data-key=probs] summary")  # closing it unhides the reply only
    page.wait_for_selector("#thread .msg.shaded", state="detached")
    assert page.is_hidden(".turn.user .msg") and page.is_visible(".reply.last .msg")


@pytest.mark.parametrize(
    ("backend_name", "heading", "name"),
    [
        ("llama.cpp", "not on llama.cpp, for now", "H3-llama-cpp"),
        ("transformers", "not recorded", "H2-not-recorded"),
    ],
)
def test_probabilities_say_why_there_are_none(clarity, page, backend_name, heading, name):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend.name = backend_name
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel [data-key=probs]")
    assert page.text_content("#panel [data-key=probs] summary .n") == heading
    page.click("#panel [data-key=probs] summary")
    if backend_name == "llama.cpp":
        href = page.get_attribute("#panel [data-key=probs] a[data-docs]", "href")
        assert href.endswith("/docs/guides.md#use-the-transformers-backend")
    shot(page, name, "c4c")


# --- Branches (C4d)


def shows_marked_only(page, on: bool) -> None:
    """Which list "Marked only" shows, by visibility: text_content() reads hidden text too."""
    ends, marked = "#panel ul[data-list=ends]", "#panel div[data-list=marked]"
    assert page.is_visible(marked if on else ends) and page.is_hidden(ends if on else marked)
    assert page.get_attribute("#marked-only", "aria-pressed") == str(on).lower()


def test_branches_appear_with_the_second_and_take_you_to_one(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    assert page.is_hidden("#branches-button")  # one branch: nothing to list
    backend.replies = [" Second one."]
    page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#branches-button:not([hidden])")
    assert page.text_content("#branches-count") == "2"
    page.click("#branches-button")
    page.wait_for_selector("#panel .bcard")
    ends = page.evaluate("clarity.command('branches', 'all')")["result"]["ends"]
    assert page.locator("#panel [data-list=ends] .bcard").count() == len(ends) == 2
    shot(page, "J2-branch-ends", "c4d")
    shows_marked_only(page, False)
    page.click("#marked-only")
    shows_marked_only(page, True)
    assert "No marked replies in this chat yet." in page.text_content(
        "#panel div[data-list=marked]"
    )
    shot(page, "J3-nothing-marked", "c4d")
    page.click("#marked-only")
    shows_marked_only(page, False)
    first = next(e for e in ends if not e["current"])
    page.click(f"#panel [data-switch='{first['turn_id']}']")
    wait_for(page, f"#panel .bcard.cur[data-turn='{first['turn_id']}']")
    assert page.text_content(".reply.last .msg") == first["text"]


def test_a_marked_reply_picked_in_branches_is_outlined(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    first = page.evaluate("clarity.command('prompt')")["result"]["call"]
    page.evaluate(f"clarity.command('mark', '{first['call_id']} landed')")
    backend.replies = [" Second one."]
    page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#branches-button:not([hidden])")
    page.click("#branches-button")
    page.wait_for_selector("#panel .bcard")
    shows_marked_only(page, False)
    page.click("#marked-only")
    shows_marked_only(page, True)
    page.click("#marked-only")
    shows_marked_only(page, False)
    page.click("#marked-only")
    page.wait_for_selector("#panel [data-list=marked] .bcard")
    shows_marked_only(page, True)
    assert page.text_content("#panel [data-list=marked] .lbl") == "Marked: landed"
    shot(page, "J2-marked-only", "c4d")
    page.click("#panel [data-list=marked] [data-reveal]")
    page.wait_for_selector(f"li.revealed[data-turn='{first['turn_id']}']")


# --- Checkpoint 4's findings


def test_reset_all_puts_every_changed_setting_back(clarity, page):
    """Finding 13 (board I9): one /set NAME per changed row."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    open_settings(page)
    for name, value in (("temperature", "0.3"), ("top_p", "0.5")):
        page.fill(f"#set-{name}", value)
        page.press(f"#set-{name}", "Enter")
        page.wait_for_selector(f"[data-setting={name}] .tag:has-text('chat')")
    page.click("#panel [data-action=reset-all]")
    page.wait_for_selector("#panel [data-action=reset-all]", state="detached")
    settings = page.evaluate("clarity.command('settings')")["result"]["settings"]
    assert all(s["origin"] != "chat" for s in settings.values())


def test_an_empty_box_is_refused_in_the_page_and_sends_nothing(clarity, page):
    """Finding 4: an empty box would send `/set NAME`, which puts the setting back."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    open_settings(page)
    page.fill("#set-temperature", "0.3")
    page.press("#set-temperature", "Enter")
    page.wait_for_selector("[data-setting=temperature] .tag:has-text('chat')")
    page.fill("#set-temperature", "")
    page.press("#set-temperature", "Enter")
    page.wait_for_selector("[data-setting=temperature] .refused:not([hidden])")
    assert page.text_content("[data-setting=temperature] .refused") == (
        "Type a value; an empty box changes nothing."
    )
    settings = page.evaluate("clarity.command('settings')")["result"]["settings"]
    assert settings["temperature"]["value"] == 0.3  # nothing was sent


# --- Chats and export (C5)

UNREADABLE = 'not a record\n{"type": "title"}\n'


def new_chat(page) -> str:
    """Start a new chat on the same profile; returns the earlier chat's log ID."""
    first = page.evaluate("clarity.command('state')")["result"]["log_id"]
    page.click("#clarity-button")
    page.click("#new-chat")
    page.wait_for_selector("#thread .intro")
    return first


def open_drawer(page) -> None:
    page.click("#chats-handle")
    page.wait_for_selector("#drawer:not([hidden]) .crow")


def test_a_chat_picked_in_the_drawer_reopens_and_says_what_differs(clarity, page):
    _, redirect, _ = clarity
    root = redirect.parent.parent
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    first = new_chat(page)
    profile = root / "profiles" / "text.toml"
    profile.write_text(profile.read_text("utf-8") + "\n# changed\n", "utf-8")
    open_drawer(page)
    assert page.get_attribute("#chats-handle", "aria-expanded") == "true"
    current = page.evaluate("clarity.command('state')")["result"]["log_id"]
    assert page.get_attribute('#drawer .crow[aria-current="true"]', "data-resume") == current
    shot(page, "L1-drawer", "c5")
    page.click(f"#drawer [data-resume='{first}']")
    page.wait_for_selector("#drawer", state="hidden")
    page.wait_for_selector("#thread .resume")
    assert page.text_content(".turn.user .msg") == "Hi"
    assert "1 thing differs from when this chat was written" in page.text_content("#thread .resume")
    assert page.is_hidden("#differ")
    shot(page, "L3-closed", "c5")
    page.click("#thread [data-action=differ]")
    assert page.is_visible("#differ") and "The profile's file has changed" in page.text_content(
        "#differ"
    )
    assert page.get_attribute("#thread [data-action=differ]", "aria-expanded") == "true"
    shot(page, "L3-open", "c5")
    say(page, "Again")  # the line goes with the next reply
    page.wait_for_selector("#thread .resume", state="detached")


def test_a_refused_chat_says_so_under_its_row_and_the_open_one_stays(clarity, page):
    _, redirect, _ = clarity
    root = redirect.parent.parent
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    (root / "logs" / "20261028-101544-7c2e9d.jsonl").write_text(UNREADABLE, "utf-8")
    open_drawer(page)
    row = "#drawer [data-resume='20261028-101544-7c2e9d']"
    assert page.text_content(f"{row} .bad") == "! can’t be read"
    page.click(row)
    item = "#drawer li:has([data-resume='20261028-101544-7c2e9d'])"
    page.wait_for_selector(f"{item} .problem:not([hidden])")
    problem = page.text_content(f"{item} .problem")
    assert problem.startswith("! Didn’t open. ") and "20261028-101544-7c2e9d" in problem
    assert page.is_visible("#drawer") and page.text_content(".turn.user .msg") == "Hi"
    shot(page, "L6-refused", "c5")
    page.keyboard.press("Escape")
    page.wait_for_selector("#drawer", state="hidden")
    say(page, "Still here")
    page.wait_for_selector(".reply.last .msg:has-text('ok')")


def test_with_no_chats_the_drawer_says_so(clarity, page):
    _, redirect, _ = clarity
    open_clarity(page, redirect)
    page.keyboard.press("Escape")  # the picker opens first
    page.click("#chats-handle")
    page.wait_for_selector("#drawer:not([hidden]) .empty")
    assert page.text_content("#drawer .empty") == (
        "No chats yet. Each chat you start is listed here."
    )
    assert page.get_attribute("#drawer .newchat", "aria-disabled") == "true"
    shot(page, "L2-no-chats", "c5")


def test_rename_in_place_waits_on_an_empty_box_and_cancels(clarity, page):
    _, redirect, _ = clarity
    chat_on(page, redirect)
    assert page.is_hidden("#title")  # nothing to rename before the first message
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.click("#title")
    page.click("#title-menu [data-action=rename]")
    assert page.is_visible("#rename") and page.is_hidden("#title")
    assert focused(page) == "rename-text" and page.input_value("#rename-text") == "Hi"
    page.fill("#rename-text", "")
    assert page.get_attribute("#rename-save", "aria-disabled") == "true"
    page.keyboard.press("Escape")
    assert page.is_hidden("#rename") and page.text_content("#title-text") == "Hi"
    page.click("#title")
    page.click("#title-menu [data-action=rename]")
    page.fill("#rename-text", "Field notes")
    shot(page, "L8-rename", "c5")
    page.click("#rename-save")
    page.wait_for_selector("#title-text:has-text('Field notes')")
    assert page.evaluate("clarity.command('state')")["result"]["title"] == "Field notes"
    open_drawer(page)
    assert page.text_content('#drawer .crow[aria-current="true"] .t') == "Field notes"


def test_export_shows_the_sheet_and_downloads_the_file(clarity, page):
    _, redirect, _ = clarity
    root = redirect.parent.parent
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    (root / "logs" / "exports").write_text("in the way", "utf-8")
    page.click("#title")
    page.click("#title-menu [data-action=export]")
    page.wait_for_selector("#export-sheet[open] #export-title")
    assert page.text_content("#export-title") == "Not exported"
    assert "Can't write the export to" in page.text_content("#export-sheet .problem")
    shot(page, "M2-not-exported", "c5")
    (root / "logs" / "exports").unlink()
    page.click("#export-sheet [data-action=export]")  # Try again
    page.wait_for_selector("#export-title:has-text('Exported for a Python notebook')")
    path = page.text_content("#export-sheet .path")
    assert Path(path).is_file()
    href = page.get_attribute("#export-sheet a[data-docs]", "href")
    assert href.endswith("/docs/guides.md#take-a-conversation-to-a-notebook")
    shot(page, "M1-exported", "c5")
    with page.expect_download() as info:
        page.click("#export-sheet [data-action=download]")
    download = info.value
    assert download.suggested_filename == Path(path).name
    assert Path(download.path()).read_bytes() == Path(path).read_bytes()
    page.click("#export-sheet [data-action=export-close]")
    assert not page.evaluate("document.getElementById('export-sheet').open")


def test_a_refused_restart_shows_the_server_s_output(clarity, page):
    from clarity_chat.backends import BackendError

    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.stopped = True
    say(page, "Again")
    page.wait_for_selector("#thread [data-action=restart]")

    def refused(*args):
        raise BackendError("llama-server failed to start: exited with code 1:\nline one\nline two")

    real = backend.start
    backend.start = refused
    page.click("#thread [data-action=restart]")
    page.wait_for_selector("#notice .outbox")
    assert page.text_content("#notice [role=alert]") == (
        "! The model didn’t start again. llama-server failed to start: exited with code 1:"
    )
    assert page.text_content("#notice .outbox") == "line one\nline two"
    # The chat stays, its server stopped, as C2 draws it: the button to try again, and
    # your message in the box.
    page.wait_for_selector("#thread [data-action=restart]")
    assert page.text_content(".turn.user .msg") == "Hi" and page.input_value("#message") == "Again"
    assert page.evaluate("clarity.command('state')")["result"]["model"] == "stopped"
    shot(page, "C23-restart-refused", "c5")
    backend.start = real  # a retry that works
    page.click("#thread [data-action=restart]")
    page.wait_for_selector("#thread [data-action=restart]", state="detached")
    page.click("#send")
    page.wait_for_selector(".reply.last .msg:has-text('ok')")


# --- Appearance and access (C6)


def open_appearance(page) -> None:
    page.click("#clarity-button")
    page.click("#clarity-menu [data-action=appearance]")
    page.wait_for_selector("#appearance[open]")


def style(page, selector: str, name: str) -> str:
    return page.evaluate(f"getComputedStyle(document.querySelector('{selector}')).{name}")


def tab_indexes(page, selector: str) -> list[int]:
    return page.evaluate(f"[...document.querySelectorAll('{selector}')].map(b => b.tabIndex)")


def classes(page) -> list[str]:
    return page.evaluate("[...document.documentElement.classList]")


def test_appearance_applies_at_once_is_kept_and_returns_focus(clarity, page):
    _, redirect, _ = clarity
    chat_on(page, redirect)
    open_appearance(page)
    assert page.evaluate("document.activeElement.value") == "notebook"
    assert page.is_hidden("#motion-row")  # nothing moves in Notebook
    assert page.text_content("#theme-desc") == "Notebook: serif text on warm paper. The default."
    page.click("#appearance .card:has(input[value=sky])")
    assert "t-sky" in classes(page) and "scenic" in classes(page)
    assert page.is_visible(".scene.sky") and page.is_visible("#motion-row")
    assert page.get_attribute("#motion", "aria-checked") == "true"
    shot(page, "N1-sky", "c6")
    page.click("#motion")  # off: the scenery pauses where it is
    assert page.get_attribute("#motion", "aria-checked") == "false"
    assert "motion-off" in classes(page)
    assert style(page, ".sky .flight", "animationPlayState") == "paused"
    page.check("#appearance input[value=dark]")
    assert page.get_attribute("html", "data-scheme") == "dark"
    page.keyboard.press("Escape")
    page.wait_for_selector("#appearance", state="hidden")
    assert focused(page) == "clarity-button"
    page.reload()  # kept in the browser's storage
    page.wait_for_selector("body[data-page=ready]")
    assert {"t-sky", "scenic", "motion-off"} <= set(classes(page))
    assert page.get_attribute("html", "data-scheme") == "dark"
    open_appearance(page)
    page.click("#appearance .card:has(input[value=hc])")
    width = page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--focus-width')"
    )  # noqa: E501
    assert width.strip() == "3px" and page.is_hidden("#motion-row")
    shot(page, "N5-high-contrast", "c6")


def test_motion_is_off_by_default_when_the_system_asks(clarity, page):
    _, redirect, _ = clarity
    page.emulate_media(reduced_motion="reduce")
    chat_on(page, redirect)
    open_appearance(page)
    page.click("#appearance .card:has(input[value=space])")
    assert page.get_attribute("#motion", "aria-checked") == "false"
    assert page.text_content("#motion-why") == "Off because your system asks for reduced motion."
    assert page.is_visible("#motion-why")
    assert style(page, ".space .shoot", "animationPlayState") == "paused"
    shot(page, "N4-reduced-motion", "c6")
    page.click("#motion")  # your choice stands over the default
    assert page.get_attribute("#motion", "aria-checked") == "true" and page.is_hidden("#motion-why")


def test_a_toolbar_is_one_tab_stop_with_arrow_keys_inside(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    toolbar = '.reply.last .tb[role="toolbar"]'
    stops = tab_indexes(page, f"{toolbar} button")
    assert stops.count(0) == 1 and stops[0] == 0
    page.focus(f"{toolbar} button >> nth=0")
    page.keyboard.press("ArrowRight")
    assert page.evaluate("document.activeElement.dataset.action") == "regen"
    page.keyboard.press("End")
    assert page.evaluate("document.activeElement.dataset.action") == "details"
    page.keyboard.press("Home")
    assert page.evaluate("document.activeElement.dataset.action") == "copy"
    # Regenerate: focus goes to Stop, then back to Regenerate on the new reply's toolbar.
    backend.replies, backend.piece_delay = [" A longer second reply."], 0.2
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Enter")
    page.wait_for_selector("#stop:not([hidden])")
    assert focused(page) == "stop"
    page.wait_for_selector(".reply.last [data-action=replies]:has-text('2 replies')")
    page.wait_for_function("document.activeElement.dataset.action === 'regen'")
    assert page.text_content("#finished") == "Reply finished"  # for screen readers only
    assert page.text_content("#writing-status") == ""


def test_enter_sends_and_shift_enter_starts_a_new_line(clarity, page):
    _, redirect, _ = clarity
    chat_on(page, redirect)
    page.fill("#message", "one")
    page.press("#message", "Shift+Enter")
    page.keyboard.type("two")
    assert page.input_value("#message") == "one\ntwo"
    page.press("#message", "Enter")
    page.wait_for_selector(".turn.user .msg")
    assert page.text_content(".turn.user .msg") == "one\ntwo"
    assert focused(page) == "message"  # sent from the box: focus stays there


def test_the_replies_rows_are_one_tab_stop_with_up_and_down(clarity, page):
    _, redirect, backend = clarity
    chat_on(page, redirect)
    backend.replies = [" One.", " Two.", " Three."]
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    page.evaluate("clarity.command('regen', '2')")
    page.evaluate("clarity.refresh()")
    page.wait_for_selector(".reply.last [data-action=replies]:has-text('3 replies')")
    page.click(".reply.last [data-action=replies]")
    page.wait_for_selector(".listpop .rows .go")
    stops = tab_indexes(page, ".listpop .rows .go")
    assert stops == [0, -1]
    page.focus(".listpop .rows .go >> nth=0")
    page.keyboard.press("ArrowDown")
    assert page.evaluate("document.activeElement.textContent").startswith("Show this reply")
    assert tab_indexes(page, ".listpop .rows .go") == [-1, 0]


def test_a_narrow_window_shows_branches_as_its_icon_and_count(clarity, page):
    _, redirect, backend = clarity
    page.set_viewport_size({"width": 720, "height": 800})
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.replies = [" Second one."]
    page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#branches-button:not([hidden])")
    assert page.get_attribute("#branches-button", "aria-label") == "Branches, 2"
    assert page.is_hidden("#branches-button .lbl") and page.is_visible("#branches-count")
    page.click(".reply.last [data-action=details]")
    page.wait_for_selector("#panel:not([hidden])")
    assert style(page, "#panel", "position") == "absolute"
    shot(page, "O5-720", "c6")


def test_the_shipped_fonts_load_from_clarity_itself(clarity, page):
    """Rasa and IBM Plex come from /static (CSP font-src 'self'), not the system or the web."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    for family in ("Rasa", "IBM Plex Sans", "IBM Plex Mono"):
        loaded = page.evaluate(f"document.fonts.load('16px \"{family}\"').then(f => f.length)")
        assert loaded >= 1, family
        assert page.evaluate(f"document.fonts.check('16px \"{family}\"')"), family


def test_the_title_menu_waits_while_a_reply_is_written(clarity, page):
    """The tripwire review after C5, finding 5b: Rename and Export wait during a reply."""
    _, redirect, backend = clarity
    chat_on(page, redirect, "long")
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    backend.replies = [" One two three four five six seven eight nine ten."]
    backend.piece_delay = 0.3
    say(page, "Again")
    page.wait_for_selector("#writing:has-text('two')")
    assert page.get_attribute("#title", "aria-disabled") == "true"
    # force: Playwright would otherwise wait for aria-disabled to go, i.e. the reply's end.
    page.click("#title", force=True)
    assert page.is_hidden("#title-menu") and page.is_visible("#stop")
    page.click("#stop")
    page.wait_for_selector(".reply.last .tb")
    assert page.get_attribute("#title", "aria-disabled") is None


def test_picking_the_open_chat_in_the_drawer_just_closes_it(clarity, page):
    """The tripwire review after C5, finding 10: no /resume of the chat that's open."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    open_drawer(page)
    page.click('#drawer .crow[aria-current="true"]')
    page.wait_for_selector("#drawer", state="hidden")
    state = page.evaluate("clarity.command('state')")["result"]
    assert state["resume_differ"] is None  # a resume would have set it
    assert page.query_selector("#thread .resume") is None


def test_a_title_has_a_length_limit_and_a_long_one_is_cut_with_an_ellipsis(clarity, page):
    """Finding 11."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    assert page.get_attribute("#rename-text", "maxlength") == "120"
    page.evaluate(f"clarity.command('title', '{'word ' * 23}')")
    page.evaluate("clarity.refresh()")
    page.wait_for_selector("#title-text:has-text('word word')")
    assert style(page, "#title-text", "textOverflow") == "ellipsis"


# --- The design walkthrough at bd59488


def test_a_dropped_connection_keeps_trying_and_reloads_when_clarity_answers(clarity, page):
    """Departures list: a lost connection said "Clarity has stopped" after 3.5 s and never
    tried again. It says Clarity isn't answering, keeps trying quietly, and loads the page
    again once Clarity answers."""
    _, redirect, _ = clarity
    chat_on(page, redirect)
    say(page, "Hi")
    page.wait_for_selector(".reply.last .tb")
    for path in ("**/events*", "**/command"):
        page.route(path, lambda route: route.abort())
    page.evaluate("clarity.dropStream()")  # as if the connection had dropped
    page.wait_for_selector("body[data-page=lost]")
    assert page.text_content("[data-for=lost] h1") == "Clarity isn't answering. Trying again…"
    page.wait_for_timeout(6000)  # longer than the old 3 tries
    assert page.get_attribute("body", "data-page") == "lost"
    shot(page, "A5-not-answering")
    page.unroute("**/events*")
    page.unroute("**/command")
    page.wait_for_selector("body[data-page=ready]", timeout=20000)
    page.wait_for_selector(".turn.user .msg")
    assert page.text_content(".turn.user .msg") == "Hi"
