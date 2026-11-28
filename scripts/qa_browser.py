"""Drive Clarity's web interface in a real browser for a QA pass, and save what it showed.

    uv run --project REPO --group browser python REPO/scripts/qa_browser.py \
        --root ROOT --out DIR --steps STEPS [--init]
    ... qa_browser.py --root ROOT --out DIR --session start [--init]
    ... qa_browser.py --out DIR --session step 'STEP'
    ... qa_browser.py --out DIR --session stop

Prepares ROOT as qa_session.py does (a scratch project that loads SmolLM2 and bge-m3 only,
refused if it holds local/models.toml or local/models.lock, or a link; checked again before
each step, since Clarity reads the root while it runs), starts
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

Session mode (v0.4.8) runs the same steps one per command, so each can be chosen after
reading the screen before it. `start` prepares ROOT and starts Clarity and Chrome as a
run does, and returns once the page has loaded, printing the paths of its shot
(DIR/00-start.png, .txt and .aria.yml); the session itself (`--session run`, which only
`start` should run) goes on in the background. Each `step` runs one step line and shoots
the screen after it as DIR/NN-step (NN-failed when it failed), then prints the shot's
paths, the page's errors since the last step and the URLs it blocked. It exits 0 when the
step ran, 1 when it failed (the session stays up), 2 when it was refused. `stop` stops
Clarity as Ctrl-C does and closes Chrome. A session also stops after 15 minutes with no
step, when Clarity exits, and when a step finds a link or local model entries in the root.
The commands and the session talk only through files in DIR/session/ (NN.step, NN.reply,
stop, ended, and run.log for the session's own output), never a port, and none of those
files is read through a link.

Not driven here; these stay in the owner's by-hand script, so report them as not
covered: a second tab and "Use it here", Enter in the terminal, a closed terminal,
opening by the printed link, Safari and Firefox, the file Download saves, and what Copy
put on the clipboard (a click on Copy may still write to the owner's real clipboard).

The owner allowlists this script to run outside the sandbox (which hides the GPU and
can't launch Chrome), so the root, DIR and STEPS must be in folders below notes/qa/.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import functools
import importlib.util
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import IO, TYPE_CHECKING, Any

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
IDLE_S = 15 * 60  # a session with no step this long stops
STEP_S = WAIT_MS / 1000 + 30  # a step's reply: a `wait` step's 2 min, then the shot
START_S = LINK_S + STEP_S  # Clarity's link, then the first load and its shot
STOP_S = STEP_S + 60  # a step still running, then Clarity's 30 s to stop
POLL_S = 0.1
LINE_MAX = 1000  # bytes in a step line, as qa_session.py allows
REPLY_MAX = 1_000_000
STEP_FILE = re.compile(r"([0-9]{2,})\.step")
SESSION, ENDED = "session", "ended"  # DIR/session/, and the file saying how it ended


class Refused(Exception):
    """A path or step the driver won't run."""


class RootChanged(Refused):
    """The root gained a link or local model entries while Clarity ran: the run ends."""


class Transcript:
    """DIR/transcript.txt, written from the steps, Clarity's output and the page's events."""

    def __init__(self, file: IO[str]) -> None:
        self.file = file
        self.lock = threading.Lock()
        self.errors: list[str] = []  # the page's, until a session's next reply takes them
        self.urls: list[str] = []

    def write(self, text: str) -> None:
        with self.lock:
            self.file.write(text)
            self.file.flush()

    def error(self, text: str) -> None:
        """A JavaScript error or a console error, as `page error: …` or `console error: …`."""
        self.write(f"[{text}]\n")
        self.errors.append(text)

    def blocked(self, url: str) -> None:
        self.write(f"[qa: blocked {url}]\n")
        self.urls.append(url)

    def take(self) -> dict[str, list[str]]:
        """The errors and blocked URLs since the last call, for a step's reply."""
        seen = {"errors": self.errors, "blocked": self.urls}
        self.errors, self.urls = [], []
        return seen


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
    lines = enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
    return [step for n, line in lines if (step := parse_line(line, f"{path}:{n}", root))]


def parse_line(line: str, where: str, root: Path | None) -> tuple[str, str] | None:
    """One step line as (verb, argument), or None for a blank line or a comment."""
    if not line.strip() or line.startswith("#"):
        return None
    verb, _, arg = line.strip().partition(" ")
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
    return verb, arg


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


def shot(page: Page, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}{ext}" for ext in (".png", ".txt", ".aria.yml")]
    new_file(paths[0], page.screenshot(full_page=True))
    new_file(paths[1], page.locator("body").inner_text())
    new_file(paths[2], page.locator("body").aria_snapshot())
    return paths


def run_step(page: Page, verb: str, arg: str, out: Path, n: int, root: Path | None = None) -> None:
    if root is not None:  # again before each step: Clarity reads the root while it runs
        session = qa_session()
        try:
            session.no_links(root)
            session.no_local_models(root)
        except session.Refused as e:
            raise RootChanged(f"step {n} not run: {e}") from e
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


@contextlib.contextmanager
def clarity_ui(root: Path, log: Transcript) -> Iterator[subprocess.Popen[str] | None]:
    """`clarity --root ROOT ui`, yielded once it has printed its link (None, with why in the
    transcript, when it didn't), then stopped as Ctrl-C does."""
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
            yield None
        elif not linked.is_set():
            log.write("[qa: clarity ui didn't print its link]\n")
            yield None
        else:
            yield clarity
    finally:
        clarity.send_signal(signal.SIGINT)  # as Ctrl-C in its terminal
        try:
            clarity.wait(timeout=30)
        except subprocess.TimeoutExpired:
            clarity.kill()
            clarity.wait()
        reader.join(timeout=5)
        log.write(f"\n[qa: clarity ui exited with status {clarity.returncode}]\n")


@contextlib.contextmanager
def chrome(log: Transcript) -> Iterator[Page]:
    """Chrome through Playwright, with a fresh profile, loading only Clarity and local files;
    the page's errors and each blocked request go to the transcript."""
    from playwright.sync_api import sync_playwright

    def route(route: Route) -> None:
        if allowed(route.request.url):
            route.continue_()
        else:
            log.blocked(route.request.url)
            route.abort()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.route("**/*", route)  # covers new tabs too
        page = context.new_page()
        page.on("pageerror", lambda error: log.error(f"page error: {error}"))
        page.on(
            "console",
            lambda message: (
                log.error(f"console error: {message.text}") if message.type == "error" else None
            ),
        )
        yield page


def page_error() -> type[Exception]:
    """Playwright's error, which a failing step raises; imported only to drive."""
    from playwright.sync_api import Error

    return Error


def load(page: Page, redirect: Path) -> None:
    page.goto(redirect.as_uri())
    page.wait_for_selector("body[data-page=ready]", timeout=WAIT_MS)


def failed(page: Page, out: Path, log: Transcript, n: int, error: Exception) -> list[Path]:
    """Write why step n (0: the first load) failed, whole, and shoot the screen it left."""
    log.write(f"[qa: {f'step {n}' if n else 'the first load'} failed: {error}]\n")
    try:
        return shot(page, out, f"{n:02d}-failed")
    except (page_error(), OSError) as e:
        log.write(f"[qa: no failure screenshot: {e}]\n")
        return []


def stop(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)  # so the cleanup in `finally` runs and frees port 8765


def stop_on_signals() -> None:
    for name in ("SIGTERM", "SIGHUP"):  # Windows has no SIGHUP
        if (signum := getattr(signal, name, None)) is not None:
            signal.signal(signum, stop)


def run_steps(root: Path, out: Path, steps_path: Path, init: bool, given_root: Path) -> int:
    """Step-list mode: every step checked, then Clarity and Chrome started and the steps run."""
    session = qa_session()
    try:
        steps = parse_steps(steps_path, root)
        session.prepare_root(root, init=init)
        out.mkdir(parents=True, exist_ok=True)
        file = create(out / "transcript.txt", "x")
    except session.Refused as e:
        raise Refused(str(e)) from e
    except FileExistsError as e:
        raise Refused(
            f"{out} already holds a run's transcript.txt (or a link there): use a new DIR"
        ) from e
    with file:
        log = Transcript(file)
        log.write(f"$ clarity --root {given_root} ui\n")
        stop_on_signals()
        with clarity_ui(root, log) as clarity:
            if clarity is None:
                return 1
            return drive(root / "local" / "clarity-open.html", steps, out, log, root)


def drive(
    redirect: Path, steps: list[tuple[str, str]], out: Path, log: Transcript, root: Path
) -> int:
    with chrome(log) as page:
        n = 0
        try:
            load(page, redirect)
            for n, (verb, arg) in enumerate(steps, 1):
                log.write(f"[qa step {n}: {verb} {arg}]\n" if arg else f"[qa step {n}: {verb}]\n")
                run_step(page, verb, arg, out, n, root)
        except (page_error(), OSError, Refused) as e:
            failed(page, out, log, n, e)
            return 1
    return 0


# --- session mode: the background process ----------------------------------------------


def read_own(path: Path, limit: int) -> str:
    """A file in DIR/session/, read only if it is a plain file with one name and at most
    `limit` bytes. The session and the commands run outside the sandbox, so a link planted
    there mustn't carry a file the sandbox denies into a step, a reply or the transcript.
    (No following, and no waiting on a pipe, on macOS and Linux.)"""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
                raise Refused(f"{path} isn't a plain file of at most {limit} bytes, one name")
            return f.read(limit).decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"{path} can't be read as a file of its own: {e}") from e


def publish(path: Path, text: str) -> None:
    """Write `path` whole, so its reader never sees half: a new file beside it, renamed into
    place. Refused, as create() refuses, when anything is at `path` already."""
    part = path.with_name(f"{path.name}.part")
    new_file(part, text)
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, "already exists", str(path))
    os.rename(part, path)


def run_session(root: Path, out: Path, init: bool) -> int:
    """The session, which `--session start` runs in the background: the root prepared,
    Clarity and Chrome started, then serve(). DIR/session/ended says how it ended."""
    folder = session_folder(out)
    session = qa_session()
    why, code = f"it stopped on an error or a signal; see {folder / 'run.log'}", 1
    try:
        try:
            session.prepare_root(root, init=init)
            file = create(out / "transcript.txt", "x")
        except (Refused, session.Refused, FileExistsError) as e:
            why, code = f"refused: {e}", 2
            return code
        with file:
            log = Transcript(file)
            log.write(f"$ clarity --root {root} ui\n")
            stop_on_signals()
            with clarity_ui(root, log) as clarity:
                if clarity is None:
                    why = "clarity ui didn't start; see the transcript"
                else:
                    redirect = root / "local" / "clarity-open.html"
                    with chrome(log) as page:
                        why = serve(page, redirect, out, log, root, lambda: clarity.poll() is None)
                    code = 0
            log.write(f"[qa: session ended: {why}]\n")
        return code
    finally:
        publish(folder / ENDED, json.dumps({"reason": why, "refused": code == 2}))


def serve(
    page: Page, redirect: Path, out: Path, log: Transcript, root: Path, running: Callable[[], bool]
) -> str:
    """The first screen as step 0, then each step DIR/session/NN.step holds, in order, each
    answered in NN.reply. Returns why the session ended."""
    folder = out / SESSION

    def answer(n: int, reply: dict[str, Any]) -> str | None:
        try:
            publish(folder / f"{n:02d}.reply", json.dumps({**reply, **log.take()}, default=str))
        except OSError as e:
            return f"step {n}'s reply can't be written: {e}"
        return reply.get("ended")

    try:
        load(page, redirect)
        first: dict[str, Any] = {"status": "ran", "shot": shot(page, out, "00-start")}
    except (page_error(), OSError) as e:
        first = {"status": "failed", "message": str(e), "shot": failed(page, out, log, 0, e)}
        first["ended"] = "the first load failed"
    n, last, ended = 1, time.monotonic(), answer(0, first)
    while ended is None:
        if os.path.lexists(folder / "stop"):
            return "stopped"
        if not running():
            return "clarity ui exited"
        if time.monotonic() - last > IDLE_S:
            return f"no step in {IDLE_S:g} s"
        if os.path.lexists(folder / f"{n:02d}.step"):
            ended = answer(n, session_step(page, folder, out, log, root, n))
            n, last = n + 1, time.monotonic()
        else:
            page.wait_for_timeout(POLL_S * 1000)  # not time.sleep: it holds the page's requests
    return ended


def session_step(
    page: Page, folder: Path, out: Path, log: Transcript, root: Path, n: int
) -> dict[str, Any]:
    """Step n: its line read and checked as a step list's lines are, run, and the screen
    shot after it. A refused step leaves the page untouched."""
    try:
        line = read_own(folder / f"{n:02d}.step", LINE_MAX).removesuffix("\n")
        if re.search(r"[\x00-\x1f\x7f]", line):
            raise Refused(f"step {n}: a step is one line, with no control character")
        if (step := parse_line(line, f"step {n}", root)) is None:
            raise Refused(f"step {n}: a blank line or a comment isn't a step")
    except Refused as e:
        log.write(f"[qa: step {n} refused: {e}]\n")
        return {"status": "refused", "message": str(e)}
    verb, arg = step
    log.write(f"[qa step {n}: {verb} {arg}]\n" if arg else f"[qa step {n}: {verb}]\n")
    try:
        run_step(page, verb, arg, out, n, root)
        return {"status": "ran", "shot": shot(page, out, f"{n:02d}-step")}
    except (page_error(), OSError, Refused) as e:
        reply = {"status": "failed", "message": str(e), "shot": failed(page, out, log, n, e)}
        if isinstance(e, RootChanged):  # as a step list's run, the session ends
            reply["ended"] = f"step {n} found the root changed"
        return reply


# --- session mode: the commands --------------------------------------------------------


def session_folder(out: Path) -> Path:
    folder = out / SESSION
    if folder.is_symlink() or not folder.is_dir():
        raise Refused(f"{out} holds no session: start one with --session start")
    return folder


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(read_own(path, REPLY_MAX))
    except json.JSONDecodeError as e:
        raise Refused(f"{path} isn't a session's reply: {e}") from e
    if not isinstance(data, dict):
        raise Refused(f"{path} isn't a session's reply")
    return data


def await_files(
    folder: Path, names: list[str], wait_s: float, alive: Callable[[], bool] = lambda: True
) -> tuple[str, dict[str, Any]] | None:
    """The first of `names` to appear in `folder`, read; None after `wait_s`, or once the
    process that would write it has exited."""
    deadline = time.monotonic() + wait_s
    while True:
        last_look = time.monotonic() > deadline or not alive()
        for name in names:
            if os.path.lexists(folder / name):
                return name, read_json(folder / name)
        if last_look:
            return None
        time.sleep(POLL_S)


def report(label: str, reply: dict[str, Any]) -> str:
    """A reply as the commands print it."""
    message = f": {reply['message']}" if reply.get("message") else ""
    lines = [f"{label} {reply['status']}{message}", *(f"shot: {p}" for p in reply.get("shot", []))]
    for key, title in (("errors", "console and page errors"), ("blocked", "blocked URLs")):
        lines.append(f"{title} since the last step: {len(reply.get(key, [])) or 'none'}")
        lines += [f"  {item}" for item in reply.get(key, [])]
    if reply.get("ended"):
        lines.append(f"the session has ended: {reply['ended']}")
    return "\n".join(lines)


def start_session(root: Path, out: Path, init: bool) -> int:
    """Start the session in the background, and return once it shows the first screen."""
    out.mkdir(parents=True, exist_ok=True)
    folder = out / SESSION
    if os.path.lexists(out / "transcript.txt") or os.path.lexists(folder):
        raise Refused(f"{out} already holds a run or a session: use a new DIR")
    folder.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--out"]
    command += [str(out), "--session", "run", *(["--init"] if init else [])]
    with create(folder / "run.log", "x") as output:  # the session's own output
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # not stopped with the shell that ran `start`
        )
    got = await_files(folder, ["00.reply", ENDED], START_S, alive=lambda: child.poll() is None)
    if got is None:
        print(f"the session showed no first screen; its output is in {folder / 'run.log'}")
        return 1
    name, reply = got
    if name == ENDED:
        print(f"the session didn't start: {reply['reason']}")
        return 2 if reply.get("refused") else 1
    print(report("step 0 (the first screen)", reply))
    if reply["status"] != "ran":
        return 1
    print(f"session {child.pid} is up; send each step with --out {out} --session step 'STEP'")
    return 0


def send_step(out: Path, line: str) -> int:
    """Send one step to the session in DIR and print its reply: 0 ran, 1 failed, 2 refused."""
    folder = session_folder(out)
    if os.path.lexists(folder / ENDED):
        print(f"the session has ended: {read_json(folder / ENDED)['reason']}")
        return 2
    sent = sorted(int(m[1]) for path in folder.iterdir() if (m := STEP_FILE.fullmatch(path.name)))
    if sent and not os.path.lexists(folder / f"{sent[-1]:02d}.reply"):
        raise Refused(f"step {sent[-1]} hasn't finished: wait for it, or stop the session")
    n = sent[-1] + 1 if sent else 1
    try:
        publish(folder / f"{n:02d}.step", line + "\n")
    except FileExistsError as e:
        raise Refused(f"another step is being sent ({e.filename})") from e
    got = await_files(folder, [f"{n:02d}.reply", ENDED], STEP_S)
    if got is None:
        print(f"step {n}: no reply in {STEP_S:g} s; it may still be running")
        return 1
    name, reply = got
    if name == ENDED:
        print(f"the session ended before step {n} ran: {reply['reason']}")
        return 1
    print(report(f"step {n}", reply))
    return {"ran": 0, "failed": 1}.get(reply["status"], 2)


def send_stop(out: Path) -> int:
    """Ask the session in DIR to stop, and wait until it has."""
    folder = session_folder(out)
    with contextlib.suppress(FileExistsError):  # asked already
        new_file(folder / "stop", "")
    got = await_files(folder, [ENDED], STOP_S)
    if got is None:
        print(f"the session didn't stop in {STOP_S:g} s")
        return 1
    print(f"the session has ended: {got[1]['reason']}\ntranscript: {out / 'transcript.txt'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--session", choices=("start", "step", "stop", "run"))
    parser.add_argument("step", nargs="?", help="with --session step: one step line")
    args = parser.parse_args(argv)
    mode = f"--session {args.session}" if args.session else "a run of --steps"
    if (args.step is not None) != (args.session == "step"):
        parser.error("--session step needs a step, and only it takes one")
    needs = {None: ["--root", "--steps"], "start": ["--root"], "run": ["--root"]}
    takes = needs.get(args.session, []) + (["--init"] if args.session in needs else [])
    given = {"--root": args.root, "--steps": args.steps, "--init": args.init}
    for name in (name for name, value in given.items() if value and name not in takes):
        parser.error(f"{mode} takes no {name}")
    for name in (name for name in needs.get(args.session, []) if not given[name]):
        parser.error(f"{mode} needs {name}")
    qa_dir = QA_DIR.resolve()
    for given in (path for path in (args.root, args.out, args.steps) if path is not None):
        if given.resolve() == qa_dir or not given.resolve().is_relative_to(qa_dir):
            parser.error(f"{given} must be in a folder below {qa_dir}")  # scratch paths only
    root, out = (args.root or Path()).resolve(), args.out.resolve()
    try:
        if args.session == "step":
            return send_step(out, args.step)
        if args.session == "stop":
            return send_stop(out)
        if args.session == "start":
            return start_session(root, out, args.init)
        if args.session == "run":
            return run_session(root, out, args.init)
        return run_steps(root, out, args.steps.resolve(), args.init, args.root)
    except Refused as e:
        parser.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
