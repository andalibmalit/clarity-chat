"""Drive Clarity's web interface in a real browser for a QA pass, and save what it showed.

    uv run --project REPO --group browser python REPO/scripts/qa_browser.py \
        --root ROOT --out DIR --steps STEPS [--init]

Prepares ROOT as qa_session.py does (a scratch project that loads SmolLM2 only), starts
`clarity --root ROOT ui` with no browser of its own, opens its link in Chrome through
Playwright, and runs STEPS, one per line:

    click SELECTOR          click the element (a CSS selector, the rest of the line)
    fill SELECTOR :: TEXT   put TEXT in a text box
    type TEXT               type at the focus
    press KEY               press a key or a chord: Enter, Escape, Shift+Enter, ArrowRight
    wait SELECTOR           wait until it's visible (a model start or a reply: up to 10 min)
    gone SELECTOR           wait until it's hidden or removed
    sleep SECONDS           wait that long
    shot NAME               save DIR/NN-NAME.png and the page's text as DIR/NN-NAME.txt
    reload                  load the page again
    width PIXELS            resize the window to that width

Lines starting with `#` are skipped. Everything happens through the page, as a user's
clicks and keys: no script runs in it and no other address is opened, and the web
interface takes no file paths. DIR/transcript.txt gets each step, what failed, and what
`clarity ui` printed. The exit status is 0 when every step ran, 1 when one failed (the
rest are skipped), 2 when the root, the paths or the steps are refused.

The owner allowlists this script to run outside the sandbox (which hides the GPU and
can't launch Chrome), so the root, DIR and STEPS must be inside notes/qa/.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from playwright.sync_api import Page

REPO = Path(__file__).resolve().parents[1]
QA_DIR = REPO / "notes" / "qa"  # local-only; scratch roots, steps and what they showed
VERBS = {"click", "fill", "type", "press", "wait", "gone", "sleep", "shot", "reload", "width"}
NAME = re.compile(r"[A-Za-z0-9_-]+")
KEY = re.compile(r"[A-Za-z0-9+]+")
LONG_MS = 600_000  # a model start or a reply on a slow machine
START_S = 120.0  # `clarity ui` prints its link once it listens


class Refused(Exception):
    """A path or step the driver won't run."""


def qa_session() -> ModuleType:
    """scripts/qa_session.py, for its scratch root (prepare_root)."""
    spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_steps(path: Path) -> list[tuple[str, str]]:
    """The steps as (verb, argument), all checked before anything starts."""
    steps = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip() or line.startswith("#"):
            continue
        verb, _, arg = line.strip().partition(" ")
        where = f"{path}:{n}"
        if verb not in VERBS:
            raise Refused(f"{where}: unknown step {verb!r}; the steps are {sorted(VERBS)}")
        if verb in ("click", "wait", "gone", "type") and not arg:
            raise Refused(f"{where}: {verb} needs an argument")
        if verb == "fill" and " :: " not in arg:
            raise Refused(f"{where}: fill takes SELECTOR :: TEXT")
        if verb == "press" and not KEY.fullmatch(arg):
            raise Refused(f"{where}: press takes a key name, such as Enter or Shift+Enter")
        if verb == "shot" and not NAME.fullmatch(arg):
            raise Refused(f"{where}: shot takes a name of letters, digits, - and _")
        if verb in ("sleep", "width") and not re.fullmatch(r"[0-9]+(\.[0-9]+)?", arg):
            raise Refused(f"{where}: {verb} takes a number")
        if verb == "reload" and arg:
            raise Refused(f"{where}: reload takes nothing")
        steps.append((verb, arg))
    return steps


def run_step(page: Page, verb: str, arg: str, out: Path, n: int) -> None:
    if verb == "click":
        page.click(arg)
    elif verb == "fill":
        selector, text = arg.split(" :: ", 1)
        page.fill(selector, text)
    elif verb == "type":
        page.keyboard.type(arg)
    elif verb == "press":
        page.keyboard.press(arg)
    elif verb == "wait":
        page.wait_for_selector(arg, timeout=LONG_MS)
    elif verb == "gone":
        page.wait_for_selector(arg, state="hidden", timeout=LONG_MS)
    elif verb == "sleep":
        time.sleep(float(arg))
    elif verb == "shot":
        page.screenshot(path=out / f"{n:02d}-{arg}.png", full_page=True)
        text = page.locator("body").inner_text()
        (out / f"{n:02d}-{arg}.txt").write_text(text, encoding="utf-8")
    elif verb == "reload":
        page.reload()
        page.wait_for_selector("body[data-page]", state="attached", timeout=LONG_MS)
    elif verb == "width":
        page.set_viewport_size({"width": int(float(arg)), "height": 900})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    for path in (args.root, args.out, args.steps):  # scratch paths only
        if not path.resolve().is_relative_to(QA_DIR):
            parser.error(f"{path} must be inside {QA_DIR}")
    session = qa_session()
    try:
        steps = parse_steps(args.steps)
        session.prepare_root(args.root, init=args.init)
    except (Refused, session.Refused) as e:
        parser.error(str(e))
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    program = Path(sys.executable).parent / "clarity"
    # BROWSER=true: `clarity ui` opens its redirect file with a command that does nothing,
    # and the driver opens it in Playwright's browser instead. No downloads: SmolLM2 is
    # fetched already.
    env = {**os.environ, "BROWSER": "true", "HF_HUB_OFFLINE": "1"}
    clarity = subprocess.Popen(
        [str(program), "--root", str(root), "ui"],
        cwd=root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = (out / "transcript.txt").open("w", encoding="utf-8")
    log.write(f"$ clarity --root {args.root} ui\n")
    status = 0
    try:
        assert clarity.stdout is not None
        deadline = time.monotonic() + START_S
        line = ""
        while "Clarity is at" not in line and time.monotonic() < deadline:
            line = clarity.stdout.readline()
            if not line:
                break
            log.write(line.split("#key=")[0] + ("#key=…\n" if "#key=" in line else ""))
        if "Clarity is at" not in line:
            log.write("[qa: clarity ui didn't print its link]\n")
            return 1
        status = drive(root / "local" / "clarity-open.html", steps, out, log)
    finally:
        clarity.send_signal(signal.SIGINT)  # as Ctrl-C in its terminal
        try:
            rest, _ = clarity.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            clarity.kill()
            rest, _ = clarity.communicate()
        log.write(rest or "")
        log.write(f"\n[qa: clarity ui exited with status {clarity.returncode}]\n")
        log.close()
    return status


def drive(redirect: Path, steps: list[tuple[str, str]], out: Path, log: TextIO) -> int:
    from playwright.sync_api import Error, sync_playwright

    channel = {"channel": "chrome"} if sys.platform == "darwin" else {}
    with sync_playwright() as p:
        browser = p.chromium.launch(**channel)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(redirect.as_uri())
        page.wait_for_selector("body[data-page=ready]", timeout=LONG_MS)
        for n, (verb, arg) in enumerate(steps, 1):
            log.write(f"[qa step {n}: {verb} {arg}]\n".rstrip() + "\n")
            log.flush()
            try:
                run_step(page, verb, arg, out, n)
            except Error as e:
                log.write(f"[qa step {n} failed: {str(e).splitlines()[0]}]\n")
                page.screenshot(path=out / f"{n:02d}-failed.png", full_page=True)
                browser.close()
                return 1
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
