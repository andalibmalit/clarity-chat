"""E1 (v0.3.1, the sprint audit's finding 5): sent or said why. In each waiting state, every
control on screen, and one level into whatever it opens (a menu, a popover, a panel, a
dialog), either sends a command, opens something, or puts a reason on screen; none does
nothing. A control that waits with `aria-disabled` passes when the reason it's described
by is on screen. Natively disabled buttons are out of scope. Commands that would change the
chat are recorded and answered with an empty result, so one press doesn't change the state
the next is pressed in.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from tests.test_browser import hold_start, open_clarity, until
from tests.test_browser_attach import (  # noqa: F401 - fixtures, used by name
    LONG,
    NOTES,
    browser_type_launch_args,
    chat,
    choose,
    clarity,
    ready,
)
from tests.test_documents_extract import stub

pytestmark = pytest.mark.browser

# Commands that only read (commands.READS, with state, profiles and chats) reach the server;
# any other is recorded as sent and answered with an empty result.
READS = {"state", "history", "branches", "compare", "prompt", "probs", "settings", "profiles"}
READS |= {"chats"}

# The controls a person can press now: in the open dialog if there is one, else the app.
# `opened` keeps only those in something a press opens that weren't there before.
TAG = """(opened) => {
  const root = document.querySelector("dialog[open]") || document.getElementById("app");
  for (const el of document.querySelectorAll("[data-e1]")) delete el.dataset.e1;
  // What a press opens: a menu, a popover, the panel, the drawer, a dialog or Compare.
  const OPENS = "[role=menu], .pop, .listpop, #panel, #drawer, dialog[open], #compare";
  const out = [];
  for (const el of root.querySelectorAll("button")) {
    if (opened && !el.closest(OPENS)) continue;
    if (el.disabled || el.closest("[hidden], [inert]") || !el.getClientRects().length) continue;
    // Inside a closed <details> (not its summary) it has a box but isn't shown or pressable.
    const shut = el.closest("details:not([open])");
    if (shut && !el.closest("summary")) continue;
    // A chosen option or a pressed toggle pressed again stays so, as the screen shows.
    if (el.getAttribute("role") === "radio" && el.getAttribute("aria-checked") === "true") continue;
    if (el.getAttribute("aria-pressed") === "true" && el.closest(".seg")) continue;
    if (opened && el.dataset.before) continue;
    if (!opened) el.dataset.before = "1";
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

# A control that waits (aria-disabled) and is described by a reason already on screen.
SAYS_WHY = """(k) => {
  const el = document.querySelector(`[data-e1="${k}"]`);
  if (!el || el.getAttribute("aria-disabled") !== "true") return false;
  const ids = (el.getAttribute("aria-describedby") || "").split(" ").filter(Boolean);
  return ids.some((id) => {
    const why = document.getElementById(id);
    return why && !why.closest("[hidden]") && !why.classList.contains("vh") && why.innerText.trim();
  });
}"""


@dataclass
class Ctx:
    page: Any
    redirect: Any
    backend: Any
    monkeypatch: Any


def idle(c: Ctx, again: bool) -> None:
    if not again:
        chat(c.page, c.redirect)
        # Room for a reply (or a replay of this call) that lasts minutes: see writing().
        c.page.evaluate("clarity.command('set', 'max_tokens 4000')")
        c.page.fill("#message", "Hi")
        c.page.click("#send")
    c.page.wait_for_selector(".reply.last .tb")


def writing(c: Ctx, again: bool) -> None:
    if not again:  # a reply that takes minutes: every press happens while it's written
        idle(c, False)
        c.backend.replies, c.backend.piece_delay = [" Slowly" * 1500], 0.3
        c.page.click(".reply.last [data-action=regen]")
    c.page.wait_for_selector("#writing-box:not([hidden])")


def replaying(c: Ctx, again: bool) -> None:
    """A command that isn't a reply runs: a Replay."""
    if not again:
        idle(c, False)
        c.page.click(".reply.last [data-action=details]")
        c.page.wait_for_selector("#panel:not([hidden]) .sec")
        c.backend.replies, c.backend.piece_delay = [" Slowly" * 1500], 0.3
        c.page.click("#panel [data-action=replay]")
        c.page.wait_for_selector("#panel [role=status]:has-text('Replaying…')")
        c.page.click("#panel [data-action=panel-close]")
    c.page.wait_for_selector(".reply.last")


def first_screen(c: Ctx, again: bool) -> None:
    if not again:
        open_clarity(c.page, c.redirect)
    c.page.wait_for_selector("#picker-menu:not([hidden])")


def starting(c: Ctx, again: bool) -> None:
    """A model starting (held for minutes)."""
    if not again:
        open_clarity(c.page, c.redirect)
        hold_start(c.backend, 600)
        c.page.click("[data-profile=chat]")
    c.page.wait_for_selector("#thread [role=status]:has-text('Starting')")


def reading_line(c: Ctx, again: bool) -> None:
    """A file being read that this page didn't start (a reload mid-read): the reading line."""
    if not again:
        stub(c.monkeypatch, "import time; time.sleep(600)")
        chat(c.page, c.redirect)
        choose(c.page, "slow.md", NOTES)
        until(c.page, lambda: c.page.evaluate("clarity.command('state')")["result"]["reading"])
        c.page.reload()
    c.page.wait_for_selector("#reading-line:not([hidden])")


def sheet(c: Ctx, again: bool) -> None:
    if not again:
        idle(c, False)
    c.page.wait_for_selector(".reply.last .tb")
    choose(c.page, "notes.md", NOTES)
    ready(c.page)


def doesnt_fit(c: Ctx, again: bool) -> None:
    """The sheet for a file whose whole doesn't fit: Attach whole waits, saying why."""
    if not again:
        c.backend._n_ctx = 2000
        chat(c.page, c.redirect)
    choose(c.page, "long.md", LONG)
    ready(c.page)


STATES = {
    "first screen": first_screen,
    "a model starting": starting,
    "a chat, idle": idle,
    "a reply being written": writing,
    "a Replay running": replaying,
    "the reading line": reading_line,
    "the attach sheet": sheet,
    "a file that doesn't fit": doesnt_fit,
}


# A control's own state as it shows it: chosen, pressed or open.
OWN = """(k) => {
  const el = document.querySelector(`[data-e1="${k}"]`);
  return el ? ["aria-checked", "aria-pressed", "aria-expanded"].map((a) => el.getAttribute(a)) : [];
}"""


def press(page, k: int, sent: list, asked: list) -> bool:
    """Press control `k` (as tagged); whether it sent, opened or said something, or changed
    its own shown state (a radio chosen, a pane switched)."""
    before = page.evaluate(SHOWN)
    own = page.evaluate(OWN, k)
    if page.evaluate(SAYS_WHY, k):
        return True
    sent.clear()
    asked.clear()
    # In the middle of its scroller, as a person would bring it before pressing it.
    page.locator(f'[data-e1="{k}"]').evaluate("el => el.scrollIntoView({block: 'center'})")
    page.click(f'[data-e1="{k}"]', force=True)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        page.wait_for_timeout(100)
        if sent or asked or page.evaluate(SHOWN) != before:
            return True
        if page.evaluate(OWN, k) not in (own, []):
            return True
    return False


def opened(page) -> list[str]:
    """The controls a press opened, once what it opens is drawn: the list stops growing."""
    seen: list[str] = []
    for _ in range(15):  # up to 1.5 s, for a view's answer
        page.wait_for_timeout(100)
        now = page.evaluate(TAG, True)
        if now and now == seen:
            return now
        seen = now
    return seen


@pytest.mark.parametrize("state", list(STATES))
def test_every_control_sends_or_says_why(clarity, page, state, monkeypatch):  # noqa: F811
    redirect, backend = clarity
    # So Copy answers. WebKit and Firefox (E2's CI job) have no such permissions to grant.
    if page.context.browser.browser_type.name == "chromium":
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    sent: list[str] = []
    asked: list[str] = []  # views, uploads, choosers and popups, after a press

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
    c = Ctx(page, redirect, backend, monkeypatch)
    setup = STATES[state]
    setup(c, False)  # for real; then commands that change are held

    def back() -> None:  # the state as a page drawn now shows it: each press starts here
        page.reload()
        page.wait_for_selector("body[data-page=ready]")
        setup(c, True)

    page.route("**/command", command)
    back()
    names = page.evaluate(TAG, False)
    assert names, f"no controls in {state}"
    silent = []
    for k, name in enumerate(names):
        page.evaluate(TAG, False)  # the same controls, tagged again after a reload
        if not press(page, k, sent, asked):
            silent.append(name)
        inside = opened(page)  # what the press opened, one level in
        for j, child in enumerate(inside):
            back()
            page.evaluate(TAG, False)
            page.click(f'[data-e1="{k}"]', force=True)
            again = opened(page)
            assert again[j : j + 1] == [child], f"{state}: {name} opened {again}, not {inside}"
            if not press(page, j, sent, asked):
                silent.append(f"{name} › {child}")
        back()
    page.unroute("**/command")
    page.evaluate("clarity.command('cancel')")  # a read held for minutes stops here
    assert silent == [], f"{state}: pressed, nothing sent and no reason shown: {silent}"
