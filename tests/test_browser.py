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


def shot(page, name: str) -> None:
    """A screenshot for the design review, kept in a local-only folder: notes/ is gitignored,
    so CI, which has no such folder, saves none."""
    if SCREENS.is_dir():
        (SCREENS / "c1").mkdir(exist_ok=True)
        page.screenshot(path=SCREENS / "c1" / f"{name}.png", full_page=True)


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
    assert page.text_content("[data-for=no-key] p") == "Open Clarity from the link in the terminal."
    shot(page, "A3-no-secret")


def test_a_secret_from_an_earlier_start_is_cleared(clarity, page):
    origin, _, _ = clarity
    page.goto(f"{origin}/#key=from-an-earlier-start")
    page.wait_for_selector("body[data-page=no-key]")
    assert page.evaluate("sessionStorage.getItem('clarity-key')") is None


def test_a_page_on_another_port_can_neither_read_nor_act(clarity, page, tmp_path):
    origin, redirect, _ = clarity
    open_clarity(page, redirect)  # Clarity is running, with its secret in that tab
    attacker = tmp_path / "attack.html"
    attacker.write_text(
        f"""<!doctype html><script>
        const tries = [
          fetch("{origin}/view/thread", {{headers: {{Authorization: "Bearer guess"}}}}),
          fetch("{origin}/command", {{method: "POST", headers: {{"Content-Type":
            "application/json"}}, body: JSON.stringify({{command: "say", args: "hi", tab: "x"}})}}),
          fetch("{origin}/events?tab=x"),
        ];
        Promise.allSettled(tries.map(t => t.then(r => r.text()))).then(results => {{
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
    assert all(r == "blocked" or "Open Clarity from the link" in r for r in results), results


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
    assert page.is_enabled("#message") and page.is_disabled("#send")
    shot(page, "B2-starting")
    # B5: ready.
    page.wait_for_selector("#thread .intro")
    assert page.text_content("#picker-button .nm") == "tiny · text"
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
        "! far didn’t start. Model 'other' is not fetched. Run `clarity fetch other`."
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
