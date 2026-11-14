"""The page in a real browser (v0.3.0 item B6): the secret, the reload, the live tab, a page
on another port, and a reply joined after a reload mid-reply.

Run with `uv run pytest -m browser` (needs `uv sync --group browser`; on the Mac it drives
the installed Chrome, on CI Playwright's Chromium). The server is uvicorn on a free port with
the fake backend; nothing reaches a model.
"""

from __future__ import annotations

import http.server
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

pytestmark = pytest.mark.browser


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
    assert page.text_content("#status") == "Ready."
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
    assert page.text_content("#status") == "Open Clarity from the link in the terminal."


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
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')")
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
