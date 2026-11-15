"""E1 (v0.3.1, the sprint audit's finding 5): sent or said why. In each waiting state, every
control on screen either sends a command, opens something, or puts a reason on screen;
none does nothing. Commands that would change the chat are recorded and answered with an
empty result, so one control's press doesn't change the state the next is pressed in.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import json
import time

import pytest

from tests.test_browser import open_clarity
from tests.test_browser_attach import (  # noqa: F401 - fixtures, used by name
    NOTES,
    browser_type_launch_args,
    chat,
    choose,
    clarity,
    ready,
)

pytestmark = pytest.mark.browser

# Commands that only read (commands.READS, with state, profiles and chats) reach the server;
# any other is recorded as sent and answered with an empty result.
READS = {"state", "history", "branches", "compare", "prompt", "probs", "settings", "profiles"}
READS |= {"chats"}

TAG = """() => {
  // The controls a person can press now: in the open dialog if there is one, else the app.
  const root = document.querySelector("dialog[open]") || document.getElementById("app");
  const out = [];
  for (const el of root.querySelectorAll("button")) {
    if (el.disabled || el.closest("[hidden], [inert]") || !el.getClientRects().length) continue;
    // A chosen option pressed again stays chosen, which the screen already shows.
    if (el.getAttribute("role") === "radio" && el.getAttribute("aria-checked") === "true") continue;
    el.dataset.e1 = String(out.length);
    out.push((el.getAttribute("aria-label") || el.textContent || el.id).trim().slice(0, 40));
  }
  return out;
}"""

SHOWN = """() => {
  // The text on screen: without the reply being written (it changes by itself), and
  // without text only a screen reader gets (.vh), which isn't a reason on screen.
  let text = document.body.innerText;
  const box = document.getElementById("writing-box");
  if (box && !box.hidden) text = text.replace(box.innerText, "");
  for (const hidden of document.querySelectorAll(".vh")) {
    if (hidden.innerText) text = text.replace(hidden.innerText, "");
  }
  return text;
}"""


def idle(page, redirect, backend, again: bool) -> None:
    if not again:
        chat(page, redirect)
        # Room for a reply (or a replay of this call) that lasts minutes: see writing().
        page.evaluate("clarity.command('set', 'max_tokens 4000')")
        page.fill("#message", "Hi")
        page.click("#send")
    page.wait_for_selector(".reply.last .tb")


def writing(page, redirect, backend, again: bool) -> None:
    if not again:  # a reply that takes minutes: every press happens while it's written
        idle(page, redirect, backend, False)
        backend.replies, backend.piece_delay = [" Slowly" * 600], 0.3
        page.click(".reply.last [data-action=regen]")
    page.wait_for_selector("#writing-box:not([hidden])")


def replaying(page, redirect, backend, again: bool) -> None:
    if not again:
        idle(page, redirect, backend, False)
        page.click(".reply.last [data-action=details]")
        page.wait_for_selector("#panel:not([hidden]) .sec")
        backend.replies, backend.piece_delay = [" Slowly" * 600], 0.3
        page.click("#panel [data-action=replay]")
        page.wait_for_selector("#panel [role=status]:has-text('Replaying…')")
        page.click("#panel [data-action=panel-close]")
    page.wait_for_selector(".reply.last")


def first_screen(page, redirect, backend, again: bool) -> None:
    if not again:
        open_clarity(page, redirect)
    page.wait_for_selector("#picker-menu:not([hidden])")


def sheet(page, redirect, backend, again: bool) -> None:
    if not again:
        idle(page, redirect, backend, False)
    page.wait_for_selector(".reply.last .tb")
    choose(page, "notes.md", NOTES)
    ready(page)


STATES = {
    "first screen": first_screen,
    "a chat, idle": idle,
    "a reply being written": writing,
    "a Replay running": replaying,
    "the attach sheet": sheet,
}


@pytest.mark.parametrize("state", list(STATES))
def test_every_control_sends_or_says_why(clarity, page, state):  # noqa: F811 - the fixture
    redirect, backend = clarity
    # So Copy answers. WebKit and Firefox (E2's CI job) have no such permissions to grant.
    if page.context.browser.browser_type.name == "chromium":
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    sent: list[str] = []
    asked: list[str] = []  # views and uploads, after a press

    def command(route) -> None:
        name = json.loads(route.request.post_data or "{}").get("command")
        if name != "state":
            sent.append(name)
        if name in READS:
            route.continue_()
        else:
            route.fulfill(status=200, content_type="application/json", body='{"result": {}}')

    page.on("request", lambda r: asked.append(r.url) if "/view/" in r.url else None)
    page.on("filechooser", lambda chooser: asked.append("file chooser"))
    page.on("popup", lambda popup: asked.append("popup"))
    setup = STATES[state]
    setup(page, redirect, backend, False)  # for real; then commands that change are held
    page.route("**/command", command)
    page.reload()  # the state as a page drawn now shows it: each press starts from this
    page.wait_for_selector("body[data-page=ready]")
    setup(page, redirect, backend, True)
    names = page.evaluate(TAG)
    assert names, f"no controls in {state}"
    silent = []
    for k, name in enumerate(names):
        page.evaluate(TAG)  # the same controls, tagged again after a reload
        before = page.evaluate(SHOWN)
        sent.clear()
        asked.clear()
        page.click(f'[data-e1="{k}"]', force=True)
        deadline = time.monotonic() + 2
        acted = False
        while time.monotonic() < deadline and not acted:
            page.wait_for_timeout(100)
            acted = bool(sent or asked) or page.evaluate(SHOWN) != before
        if not acted:
            silent.append(name)
        page.reload()  # back to the state, with the server's own record of it
        page.wait_for_selector("body[data-page=ready]")
        setup(page, redirect, backend, True)
    assert silent == [], f"{state}: pressed, nothing sent and no reason shown: {silent}"
