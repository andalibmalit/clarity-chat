"""The browser QA driver's session mode (v0.4.8 Q0, notes/plans/qa-driver-step-mode.md): start
a session once, send one step per command and read that step's shot before choosing the next,
then stop. Offline: Clarity and Chrome are fakes, and the session runs in a thread where
`--session start` would put it in the background. The driver runs outside the sandbox, so the
refusals and limits of step-list mode hold here too, and nothing in DIR/session/ is read
through a link."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_browser", REPO / "scripts" / "qa_browser.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
real_session = qa.qa_session()


class FakeError(Exception):
    """Playwright's Error, for a step that fails."""


class FakePage:
    """What a session uses of a Playwright page. `#missing` matches nothing; `#noisy` logs a
    console error and a blocked request, as the real page's handlers would."""

    def __init__(self, log):
        self.log = log
        self.done = []
        self.keyboard = self

    def goto(self, url):
        self.done.append(("goto", url))

    def wait_for_selector(self, selector, **kwargs):
        pass

    def click(self, selector):
        if selector == "#missing":
            raise FakeError(f"no element matches {selector}")
        if selector == "#noisy":
            self.log.error("console error: a made-up failure")
            self.log.blocked("https://example.com/")
        self.done.append(("click", selector))

    def press(self, key):
        self.done.append(("press", key))

    def screenshot(self, full_page=False):
        return b"png"

    def locator(self, selector):
        return self

    def inner_text(self):
        return f"after {self.done[-1]}"

    def aria_snapshot(self):
        return '- button "Send"'

    def wait_for_timeout(self, ms):
        time.sleep(ms / 1000)


class FakeSession:
    """qa_session.py with a scratch root that needs no model; the root checks are real."""

    Refused = real_session.Refused
    no_links = staticmethod(real_session.no_links)
    no_local_models = staticmethod(real_session.no_local_models)

    @staticmethod
    def prepare_root(root, init=False):
        root.mkdir(parents=True, exist_ok=True)


class Child:
    """The background process `start` spawns: here a thread running the driver's `run`."""

    def __init__(self, argv):
        self.code = None
        self.pid = 4242
        self.thread = threading.Thread(target=self.run, args=(argv,), daemon=True)
        self.thread.start()

    def run(self, argv):
        self.code = run(argv)

    def poll(self):
        return None if self.thread.is_alive() else self.code


class World:
    """The fakes one test shares: the spawned children, the page, and Clarity's state."""

    def __init__(self, qa_dir):
        self.root, self.out = qa_dir / "root", qa_dir / "run1"
        self.children = []
        self.page = None
        self.clarity_exited = False

    def popen(self, argv, **kwargs):
        script, *rest = argv[1:]
        assert Path(script) == REPO / "scripts" / "qa_browser.py", argv  # only the session
        self.children.append((argv, kwargs, child := Child(rest)))
        return child

    @contextlib.contextmanager
    def clarity_ui(self, root, log):
        log.write("Clarity is at http://127.0.0.1:8765/#key=…\n")
        yield self
        log.write("\n[qa: clarity ui exited with status 0]\n")

    def poll(self):  # Clarity's process
        return 0 if self.clarity_exited else None

    @contextlib.contextmanager
    def chrome(self, log):
        self.page = FakePage(log)
        yield self.page

    def ended(self, within=5.0):
        for _, _, child in self.children:
            child.thread.join(within)
        return all(not child.thread.is_alive() for _, _, child in self.children)


def run(argv) -> int:
    try:
        return qa.main([str(arg) for arg in argv])
    except SystemExit as e:
        return int(e.code or 0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    world = World(qa_dir)
    monkeypatch.setattr(qa, "QA_DIR", qa_dir)
    monkeypatch.setattr(qa, "qa_session", lambda: FakeSession)
    monkeypatch.setattr(qa.signal, "signal", lambda *args: None)  # the session is a thread here
    monkeypatch.setattr(qa.subprocess, "Popen", world.popen)
    monkeypatch.setattr(qa, "clarity_ui", world.clarity_ui)
    monkeypatch.setattr(qa, "chrome", world.chrome)
    monkeypatch.setattr(qa, "page_error", lambda: FakeError)
    yield world
    if not world.ended(within=0):
        (world.out / "session" / "stop").touch()
        world.ended()


def start(world) -> int:
    return run(["--root", world.root, "--out", world.out, "--session", "start"])


def step(world, line) -> int:
    return run(["--out", world.out, "--session", "step", line])


def stop(world) -> int:
    return run(["--out", world.out, "--session", "stop"])


def test_start_returns_once_the_page_has_loaded_and_prints_the_first_screen(world, capsys):
    code = start(world)
    printed = capsys.readouterr().out
    assert code == 0, printed
    for ext in (".png", ".txt", ".aria.yml"):
        assert f"{world.out.resolve() / '00-start'}{ext}" in printed
        assert (world.out / f"00-start{ext}").is_file()
    argv, kwargs, _ = world.children[0]
    assert argv[-2:] == ["--session", "run"]  # the session itself, in the background
    assert kwargs["start_new_session"] is True  # not stopped with the agent's shell
    redirect = world.root.resolve() / "local" / "clarity-open.html"
    assert world.page.done == [("goto", redirect.as_uri())]
    assert stop(world) == 0
    assert world.ended()


def test_a_step_runs_a_failing_step_leaves_the_session_up_and_stop_ends_it(world, capsys):
    assert start(world) == 0
    capsys.readouterr()
    assert step(world, "click #ok") == 0
    printed = capsys.readouterr().out
    assert f"{world.out.resolve() / '01-step.png'}" in printed
    assert (world.out / "01-step.txt").read_text(encoding="utf-8") == "after ('click', '#ok')"
    assert step(world, "click #missing") == 1
    printed = capsys.readouterr().out
    assert "no element matches #missing" in printed
    assert (world.out / "02-failed.png").is_file()
    assert step(world, "press Escape") == 0  # still up
    assert stop(world) == 0
    assert world.ended()
    assert world.page.done[1:] == [("click", "#ok"), ("press", "Escape")]
    transcript = (world.out / "transcript.txt").read_text(encoding="utf-8")
    assert "[qa step 1: click #ok]" in transcript and "[qa step 3: press Escape]" in transcript
    assert "#key=…" in transcript and "[qa: session ended: stopped]" in transcript


@pytest.mark.parametrize(
    "line", ["goto https://example.com", "eval document.title", "press Meta+v", "sleep 3600", ""]
)
def test_a_step_outside_the_language_is_refused_and_the_session_stays_up(world, capsys, line):
    assert start(world) == 0
    assert step(world, line) == 2
    assert "refused" in capsys.readouterr().out
    assert world.page.done[1:] == []  # the page wasn't touched
    assert step(world, "click #ok") == 0


def test_each_step_prints_the_console_errors_and_blocked_urls_since_the_last(world, capsys):
    assert start(world) == 0
    capsys.readouterr()
    assert step(world, "click #noisy") == 0
    printed = capsys.readouterr().out
    assert "a made-up failure" in printed and "https://example.com/" in printed
    assert step(world, "click #ok") == 0
    printed = capsys.readouterr().out
    assert "a made-up failure" not in printed and "https://example.com/" not in printed


def test_the_session_stops_after_its_idle_time(world, monkeypatch, capsys):
    monkeypatch.setattr(qa, "IDLE_S", 0.3)
    assert start(world) == 0
    assert world.ended()
    assert step(world, "click #ok") == 2
    assert "no step in 0.3 s" in capsys.readouterr().out
    assert world.page.done[1:] == []


def test_the_session_stops_when_clarity_exits(world, capsys):
    assert start(world) == 0
    world.clarity_exited = True
    assert world.ended()
    assert step(world, "click #ok") == 2
    assert "clarity ui exited" in capsys.readouterr().out


def test_a_link_planted_in_the_root_fails_the_step_and_ends_the_session(world, tmp_path, capsys):
    """Step-list mode's root check before each step: a run ends on it, and so does a session."""
    assert start(world) == 0
    assert step(world, "click #ok") == 0
    (world.root / "local").mkdir()
    (world.root / "local" / "elsewhere").symlink_to(tmp_path)
    assert step(world, "click #two") == 1
    assert "is a link" in capsys.readouterr().out
    assert world.ended()
    assert world.page.done[1:] == [("click", "#ok")]


def test_a_step_file_planted_as_a_link_is_not_read_through(world, tmp_path, capsys):
    """The session runs outside the sandbox: a link at DIR/session/01.step must not carry a
    file the sandbox denies into the page or the transcript."""
    secret = tmp_path / "secret.txt"
    secret.write_text("click #a-private-line\n", encoding="utf-8")
    assert start(world) == 0
    (world.out / "session" / "01.step").symlink_to(secret)
    for _ in range(50):
        if (world.out / "session" / "01.reply").exists():
            break
        time.sleep(0.1)
    reply = json.loads((world.out / "session" / "01.reply").read_text(encoding="utf-8"))
    assert reply["status"] == "refused"
    assert world.page.done[1:] == []
    assert step(world, "click #ok") == 0  # step 2: the session stays up
    assert "private" not in (world.out / "transcript.txt").read_text(encoding="utf-8")


def test_a_reply_planted_as_a_link_is_not_printed(world, tmp_path, capsys):
    """The step command also runs outside the sandbox, and prints the reply it reads."""
    secret = tmp_path / "secret.txt"
    secret.write_text('{"status": "ran", "message": "a private line"}', encoding="utf-8")
    assert start(world) == 0
    (world.out / "session" / "01.reply").symlink_to(secret)
    code = step(world, "click #ok")
    captured = capsys.readouterr()
    assert code == 2, captured
    assert "private" not in captured.out + captured.err


def test_start_refuses_a_dir_that_holds_a_session_before_anything_starts(world, capsys):
    (world.out / "session").mkdir(parents=True)
    assert start(world) == 2
    assert "use a new DIR" in capsys.readouterr().err
    assert world.children == []


def test_a_step_with_no_session_in_dir_is_refused(world, capsys):
    world.out.mkdir()
    assert step(world, "click #ok") == 2
    assert "holds no session" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "says"),
    [
        (["--session", "step", "click #ok", "--root", "ROOT"], "takes no --root"),
        (["--session", "step"], "needs a step"),
        (["--session", "start", "--steps", "STEPS"], "takes no --steps"),
        (["--session", "stop", "--init"], "takes no --init"),
        (["--session", "start"], "needs --root"),
    ],
)
def test_each_mode_takes_only_its_own_options(world, capsys, args, says):
    paths = {"ROOT": world.root, "STEPS": world.root.parent / "steps.txt"}
    assert run(["--out", world.out, *(paths.get(arg, arg) for arg in args)]) == 2
    assert says in capsys.readouterr().err
    assert world.children == []


def test_step_list_mode_still_runs_its_steps(world):
    """The same Clarity and Chrome as a session's; one run, no background process."""
    steps = world.root.parent / "steps.txt"
    steps.write_text("click #ok\nshot after\n", encoding="utf-8")
    assert run(["--root", world.root, "--out", world.out, "--steps", steps]) == 0
    assert world.page.done[1:] == [("click", "#ok")]
    assert (world.out / "02-after.png").is_file()
    assert world.children == []
    assert not os.path.lexists(world.out / "session")
