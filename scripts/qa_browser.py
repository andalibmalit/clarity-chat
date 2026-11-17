"""Drive Clarity's web interface in a real browser for a QA pass, and save what it showed.

    uv run --project REPO --group browser python REPO/scripts/qa_browser.py \
        --root ROOT --out DIR --steps STEPS [--init]

Prepares ROOT as qa_session.py does (a scratch project that loads SmolLM2 only, refused if
it holds local/models.toml or local/models.lock), starts
`clarity --root ROOT ui` with no browser of its own, opens its link in Chrome through
Playwright, and runs STEPS, one per line. SELECTOR is a Playwright selector: CSS, or
`text=Regenerate` and `role=button[name="Copy"]` (none of its engines runs script).

    click SELECTOR          click the element (the rest of the line)
    fill SELECTOR :: TEXT   put TEXT in a text box; `fill SELECTOR ::` empties it
    upload SELECTOR :: FILE give a file input FILE, a file in the scratch root (relative
                            to it, or absolute), as choosing it in the picker would
    type TEXT               type at the focus
    press KEY               press a key, or Shift with one: Enter, Escape, Tab, Shift+Tab,
                            Shift+Enter, Space, ArrowRight, Home (no Meta, Control or Alt,
                            so no paste from the clipboard)
    wait SELECTOR           wait until it's visible (a model start or a reply: up to 2 min)
    gone SELECTOR           wait until it's hidden or removed (up to 2 min)
    sleep SECONDS           wait that long, up to 60
    shot NAME               save DIR/NN-NAME.png, the page's text as DIR/NN-NAME.txt, and its
                            accessibility tree (roles, names, values, states) as
                            DIR/NN-NAME.aria.yml
    reload                  load the page again
    width PIXELS            resize the window to that width

Lines starting with `#` are skipped. Everything happens through the page, as a user's
clicks and keys: no script runs in it, and the web interface takes no file paths. The
browser may load only Clarity (http://127.0.0.1:8765/) and local files: any other
request is aborted and written to the transcript as `[qa: blocked URL]`, so a link's
address is evidence without being fetched. To test Ctrl-C mid-reply, end the steps right
after the send: the driver then stops Clarity as Ctrl-C does, while the reply streams.

DIR/transcript.txt gets each step, what Clarity printed as it printed it (with the link's
secret cut), the page's JavaScript and console errors, what was blocked, and a failure
whole, with DIR/NN-failed.png, .txt and .aria.yml. The exit status is 0 when every step
ran, 1 when one failed (the rest are skipped), 2 when the root, the paths or the steps
are refused. Every file is created new: DIR must not hold an earlier run, and nothing is
written through a link.

Not driven here; these stay in the owner's by-hand script, so report them as not
covered: a second tab and "Use it here", Enter in the terminal, a closed terminal,
opening by the printed link, Safari and Firefox, the file Download saves, and what Copy
put on the clipboard (a click on Copy may still write to the owner's real clipboard).

The owner allowlists this script to run outside the sandbox (which hides the GPU and
can't launch Chrome), so the root, DIR and STEPS must be in folders below notes/qa/.
"""

from __future__ import annotations

import argparse
import errno
import functools
import importlib.util
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page, Route

REPO = Path(__file__).resolve().parents[1]
QA_DIR = REPO / "notes" / "qa"  # local-only; scratch roots, steps and what they showed
CLARITY = "http://127.0.0.1:8765/"  # web.PORT
VERBS = {
    *("click", "fill", "upload", "type", "press", "wait", "gone"),
    *("sleep", "shot", "reload", "width"),
}
NAME = re.compile(r"[A-Za-z0-9_-]+")
KEY = re.compile(r"(Shift\+)?[A-Za-z0-9]+")
LINK_S = 90  # `clarity ui` prints its link once it listens, in seconds on SmolLM2
WAIT_MS = 120_000  # SmolLM2 starts and replies in seconds; a wrong selector fails in time
SLEEP_MAX_S = 60


class Refused(Exception):
    """A path or step the driver won't run."""


class Transcript:
    """DIR/transcript.txt, written from the steps, Clarity's output and the page's events."""

    def __init__(self, file: IO[str]) -> None:
        self.file = file
        self.lock = threading.Lock()

    def write(self, text: str) -> None:
        with self.lock:
            self.file.write(text)
            self.file.flush()


@functools.cache
def qa_session() -> ModuleType:
    """scripts/qa_session.py, for its scratch root (prepare_root) and its attachment rule."""
    spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create(path: Path, mode: str) -> IO:
    """Open a new file at `path`, so a link planted in DIR can't send a write elsewhere and
    nothing is replaced. Anything already there, a dangling link included, is refused first:
    on Windows, exclusive mode alone opens through a dangling link. Exclusive mode stays, for
    a file that appears in between."""
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, "already exists", str(path))
    return path.open(mode) if "b" in mode else path.open(mode, encoding="utf-8")


def new_file(path: Path, data: str | bytes) -> None:
    with create(path, "xb" if isinstance(data, bytes) else "x") as f:
        f.write(data)


def fill_parts(arg: str) -> tuple[str, str] | None:
    """`SELECTOR :: TEXT` as (selector, text); TEXT may be empty."""
    selector, sep, text = arg.partition(" ::")
    if not sep or not selector or (text and not text.startswith(" ")):
        return None
    return selector, text[1:]


def parse_steps(path: Path, root: Path | None = None) -> list[tuple[str, str]]:
    """The steps as (verb, argument), all checked before anything starts."""
    steps = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        verb, _, arg = line.strip().partition(" ")
        where = f"{path}:{n}"
        if verb not in VERBS:
            raise Refused(f"{where}: unknown step {verb!r}; the steps are {sorted(VERBS)}")
        if verb in ("click", "wait", "gone", "type") and not arg:
            raise Refused(f"{where}: {verb} needs an argument")
        if verb == "fill" and fill_parts(arg) is None:
            raise Refused(f"{where}: fill takes SELECTOR :: TEXT")
        if verb == "upload":
            upload_file(where, arg, root)
        if verb == "press" and not KEY.fullmatch(arg):
            raise Refused(f"{where}: press takes a key name, or Shift+ one, such as Shift+Enter")
        if verb == "shot" and not NAME.fullmatch(arg):
            raise Refused(f"{where}: shot takes a name of letters, digits, - and _")
        if verb == "sleep" and not (
            re.fullmatch(r"[0-9]+(\.[0-9]+)?", arg) and float(arg) <= SLEEP_MAX_S
        ):
            raise Refused(f"{where}: sleep takes a number of seconds, up to {SLEEP_MAX_S}")
        if verb == "width" and not re.fullmatch(r"[0-9]+", arg):
            raise Refused(f"{where}: width takes a whole number of pixels")
        if verb == "reload" and arg:
            raise Refused(f"{where}: reload takes nothing")
        steps.append((verb, arg))
    return steps


def upload_file(where: str, arg: str, root: Path | None) -> Path:
    """The file an `upload SELECTOR :: FILE` step gives the page, resolved. The driver reads it
    outside the sandbox, so it must be a plain file in the scratch root, judged by
    qa_session.py's rule for attachments (links followed, `~` expanded, no second name).
    Checked when the steps are read and again when the step runs."""
    parts = fill_parts(arg)
    if parts is None or not parts[1]:
        raise Refused(f"{where}: upload takes SELECTOR :: FILE")
    if root is None:
        raise Refused(f"{where}: upload needs the scratch root to judge its file")
    value = parts[1]
    if (problem := qa_session().attach_problem(root, value)) is None:
        # The path given to the page is judged itself, as given: inside the root, a plain
        # file, one name. attach_problem's readings are the extra check.
        try:
            path = (root / Path(value).expanduser()).resolve()  # an absolute path replaces root
            if not path.is_relative_to(root.resolve()):
                problem = f"{value!r} is outside it"
            elif not path.is_file():
                problem = f"{value!r} isn't a file"
            elif path.stat().st_nlink > 1:
                problem = f"{value!r} has a second name (a hard link)"
        except (RuntimeError, ValueError, OSError) as e:
            problem = f"{value!r} can't be judged ({type(e).__name__})"
    if problem is not None:
        raise Refused(f"{where}: upload takes a file in the scratch root: {problem}")
    return path


def allowed(url: str) -> bool:
    """Whether the browser may load `url`: Clarity's own address, or anything not on the
    network (the redirect file, data: and blob: URLs)."""
    return url.startswith(CLARITY) or not url.startswith(("http://", "https://"))


def shot(page: Page, out: Path, stem: str) -> None:
    new_file(out / f"{stem}.png", page.screenshot(full_page=True))
    new_file(out / f"{stem}.txt", page.locator("body").inner_text())
    new_file(out / f"{stem}.aria.yml", page.locator("body").aria_snapshot())


def run_step(page: Page, verb: str, arg: str, out: Path, n: int, root: Path | None = None) -> None:
    if verb == "click":
        page.click(arg)
    elif verb == "fill":
        parts = fill_parts(arg)
        assert parts is not None  # parse_steps checked it
        page.fill(*parts)
    elif verb == "upload":
        selector = arg.partition(" ::")[0]
        page.set_input_files(selector, upload_file(f"step {n}", arg, root))
    elif verb == "type":
        page.keyboard.type(arg)
    elif verb == "press":
        page.keyboard.press(arg)
    elif verb == "wait":
        page.wait_for_selector(arg, timeout=WAIT_MS)
    elif verb == "gone":
        page.wait_for_selector(arg, state="hidden", timeout=WAIT_MS)
    elif verb == "sleep":
        page.wait_for_timeout(float(arg) * 1000)  # time.sleep would hold every routed request
    elif verb == "shot":
        shot(page, out, f"{n:02d}-{arg}")
    elif verb == "reload":
        page.reload()
        page.wait_for_selector("body[data-page]", state="attached", timeout=WAIT_MS)
    elif verb == "width":
        page.set_viewport_size({"width": int(arg), "height": 900})


def cut(line: str) -> str:
    """A line of Clarity's output with the link's secret cut."""
    return line.split("#key=")[0] + "#key=…\n" if "#key=" in line else line


def copy_output(
    stream: IO[str], log: Transcript, linked: threading.Event, ready: threading.Event
) -> None:
    """Copy Clarity's output to the transcript as it comes, so the pipe never fills.
    `ready` is set once Clarity prints its link (`linked` too) or exits."""
    for line in stream:
        log.write(cut(line))
        if "Clarity is at" in line:
            linked.set()
            ready.set()
    ready.set()


def stop(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)  # so the cleanup in `finally` runs and frees port 8765


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    qa_dir = QA_DIR.resolve()
    root, out, steps_path = (path.resolve() for path in (args.root, args.out, args.steps))
    for given, path in ((args.root, root), (args.out, out), (args.steps, steps_path)):
        if path == qa_dir or not path.is_relative_to(qa_dir):  # scratch paths only
            parser.error(f"{given} must be in a folder below {qa_dir}")
    session = qa_session()
    try:
        steps = parse_steps(steps_path, root)
        session.prepare_root(root, init=args.init)
        out.mkdir(parents=True, exist_ok=True)
        file = create(out / "transcript.txt", "x")
    except (Refused, session.Refused) as e:
        parser.error(str(e))
    except FileExistsError:
        parser.error(f"{out} already holds a run's transcript.txt (or a link there): use a new DIR")
    log = Transcript(file)
    log.write(f"$ clarity --root {args.root} ui\n")
    for name in ("SIGTERM", "SIGHUP"):  # Windows has no SIGHUP
        if (signum := getattr(signal, name, None)) is not None:
            signal.signal(signum, stop)
    program = Path(sys.executable).parent / "clarity"
    # BROWSER=true: `clarity ui` opens its redirect file with a command that does nothing,
    # and the driver opens it in Playwright's browser instead. No downloads: SmolLM2 is
    # fetched already. PYTHONUNBUFFERED: whatever Clarity prints reaches the transcript at once.
    env = {**os.environ, "BROWSER": "true", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    clarity = subprocess.Popen(
        [str(program), "--root", str(root), "ui"],
        cwd=root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert clarity.stdout is not None
    linked, ready = threading.Event(), threading.Event()
    reader = threading.Thread(
        target=copy_output, args=(clarity.stdout, log, linked, ready), daemon=True
    )
    reader.start()
    try:
        # Clarity prints its link once it listens, or why it can't and exits.
        if not ready.wait(LINK_S):
            log.write(f"[qa: clarity ui printed no link in {LINK_S} s]\n")
            return 1
        if not linked.is_set():
            log.write("[qa: clarity ui didn't print its link]\n")
            return 1
        return drive(root / "local" / "clarity-open.html", steps, out, log, root)
    finally:
        clarity.send_signal(signal.SIGINT)  # as Ctrl-C in its terminal
        try:
            clarity.wait(timeout=30)
        except subprocess.TimeoutExpired:
            clarity.kill()
            clarity.wait()
        reader.join(timeout=5)
        log.write(f"\n[qa: clarity ui exited with status {clarity.returncode}]\n")
        file.close()


def drive(
    redirect: Path, steps: list[tuple[str, str]], out: Path, log: Transcript, root: Path
) -> int:
    from playwright.sync_api import Error, sync_playwright

    def route(route: Route) -> None:
        if allowed(route.request.url):
            route.continue_()
        else:
            log.write(f"[qa: blocked {route.request.url}]\n")
            route.abort()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.route("**/*", route)  # covers new tabs too
        page = context.new_page()
        page.on("pageerror", lambda error: log.write(f"[page error: {error}]\n"))
        page.on(
            "console",
            lambda message: (
                log.write(f"[console error: {message.text}]\n") if message.type == "error" else None
            ),
        )
        n = 0
        try:
            page.goto(redirect.as_uri())
            page.wait_for_selector("body[data-page=ready]", timeout=WAIT_MS)
            for n, (verb, arg) in enumerate(steps, 1):
                log.write(f"[qa step {n}: {verb} {arg}]\n" if arg else f"[qa step {n}: {verb}]\n")
                run_step(page, verb, arg, out, n, root)
        except (Error, OSError, Refused) as e:
            where = f"step {n}" if n else "the first load"
            log.write(f"[qa: {where} failed: {e}]\n")  # whole: it says why
            try:
                shot(page, out, f"{n:02d}-failed")
            except (Error, OSError) as e:
                log.write(f"[qa: no failure screenshot: {e}]\n")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
