"""The browser QA driver's checks: its steps are parsed and refused before anything runs,
and its paths must be scratch paths (it runs outside the sandbox). Playwright isn't
needed here; the driver imports it only to drive."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_browser", REPO / "scripts" / "qa_browser.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def steps(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "steps.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_ordinary_steps_parse_as_verb_and_argument(tmp_path):
    parsed = qa.parse_steps(
        steps(
            tmp_path,
            "# start a chat",
            "click [data-profile=smol-chat]",
            "wait #thread .intro",
            "fill #message :: Hello :: there",
            "press Shift+Enter",
            "type more",
            "gone #stop",
            "sleep 1.5",
            "shot after-reply",
            "reload",
            "width 720",
        )
    )
    assert parsed[0] == ("click", "[data-profile=smol-chat]")
    assert ("fill", "#message :: Hello :: there") in parsed and ("reload", "") in parsed
    assert len(parsed) == 10


@pytest.mark.parametrize(
    "line",
    [
        "goto https://example.com",  # no other address
        "eval document.title",  # no script in the page
        "click",
        "fill #message Hello",
        "press Enter; rm",  # a key name only, so nothing after it
        "shot ../../elsewhere",
        "sleep forever",
        "width wide",
        "reload now",
    ],
)
def test_steps_outside_the_list_are_refused(tmp_path, line):
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(tmp_path, line))


@pytest.mark.parametrize(
    "line",
    [
        "press Meta+v",  # would paste the owner's clipboard into the page
        "press Control+v",
        "press Alt+x",
        "press Meta+a",
        "sleep 3600",  # holds the port until someone kills the driver
    ],
)
def test_chords_that_reach_the_clipboard_and_long_sleeps_are_refused(tmp_path, line):
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(tmp_path, line))


class FakePage:
    """What run_step uses of a Playwright page. Like Playwright, a screenshot given a path
    writes it in this process, following a link."""

    def __init__(self):
        self.fills = []

    def fill(self, selector, text):
        self.fills.append((selector, text))

    def screenshot(self, path=None, full_page=False):
        if path is not None:
            Path(path).write_bytes(b"png")
        return b"png"

    def locator(self, selector):
        return self

    def inner_text(self):
        return "the page's text"

    def aria_snapshot(self):
        return '- button "Send"'


def test_fill_can_empty_a_box(tmp_path):
    try:
        parsed = qa.parse_steps(
            steps(tmp_path, "fill #set-temperature :: ", "fill #set-top_p ::", "fill #m :: a :: b")
        )
    except qa.Refused as e:  # what the driver does today
        parsed = [("refused", str(e))]
    page = FakePage()
    for verb, arg in parsed:
        if verb == "fill":
            qa.run_step(page, verb, arg, tmp_path, 1)
    assert page.fills == [("#set-temperature", ""), ("#set-top_p", ""), ("#m", "a :: b")], parsed


@pytest.mark.parametrize("name", ["01-a.png", "01-a.txt"])
def test_a_shot_never_writes_through_a_link_planted_in_dir(tmp_path, name):
    out = tmp_path / "run"
    out.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    (out / name).symlink_to(outside)
    with pytest.raises(FileExistsError):
        qa.run_step(FakePage(), "shot", "a", out, 1)
    assert outside.read_text(encoding="utf-8") == "keep"


def started(*args, **kwargs):
    raise AssertionError("clarity ui started")


class FakeSession:
    class Refused(Exception):
        pass

    @staticmethod
    def prepare_root(root, init=False):
        root.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("planted", ["link", "dangling link", "an earlier run"])
def test_dir_holding_a_link_or_a_run_is_refused_before_clarity_starts(
    tmp_path, monkeypatch, planted
):
    qa_dir = tmp_path / "qa"
    out = qa_dir / "run1"
    out.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    if planted == "link":
        outside.write_text("keep", encoding="utf-8")
    if planted == "an earlier run":
        (out / "transcript.txt").write_text("the earlier run", encoding="utf-8")
    else:
        (out / "transcript.txt").symlink_to(outside)
    path = steps(qa_dir, "reload")

    monkeypatch.setattr(qa, "QA_DIR", qa_dir)
    monkeypatch.setattr(qa, "qa_session", lambda: FakeSession)
    monkeypatch.setattr(qa.subprocess, "Popen", started)
    argv = ["qa_browser.py", "--root", str(qa_dir / "root"), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--steps", str(path)])
    with pytest.raises(SystemExit) as caught:
        qa.main()
    assert caught.value.code == 2
    if planted == "link":
        assert outside.read_text(encoding="utf-8") == "keep"
    if planted == "dangling link":
        assert not outside.exists()


@pytest.mark.parametrize("which", ["--root", "--out", "--steps"])
def test_each_path_must_be_below_notes_qa(tmp_path, monkeypatch, which):
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    paths = {"--root": qa_dir / "root", "--out": qa_dir / "out", "--steps": steps(qa_dir, "reload")}
    paths[which] = qa_dir if which == "--root" else tmp_path / "elsewhere"
    monkeypatch.setattr(qa, "QA_DIR", qa_dir)
    monkeypatch.setattr(qa, "qa_session", lambda: FakeSession)
    monkeypatch.setattr(qa.subprocess, "Popen", started)
    monkeypatch.setattr(
        sys, "argv", ["qa_browser.py", *(str(x) for kv in paths.items() for x in kv)]
    )
    with pytest.raises(SystemExit) as caught:
        qa.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:8765/", True),
        ("http://127.0.0.1:8765/events", True),
        ("file:///tmp/clarity-open.html", True),
        ("data:text/plain,x", True),
        ("https://github.com/andalibmalit/clarity-chat/tree/main/docs", False),
        ("http://127.0.0.1:8080/", False),
        ("http://127.0.0.1:8765.example.com/", False),
        ("http://localhost:8765/", False),
    ],
)
def test_the_browser_reaches_only_clarity(url, ok):
    assert qa.allowed(url) is ok


def test_a_sleep_lets_the_page_s_requests_through(tmp_path, monkeypatch):
    """QA of v0.3.0, items 1 and 2: the driver routes every request through a handler that
    runs in Playwright's own loop, and `sleep` was time.sleep, which stops that loop, so
    every request the page made during a sleep waited for the next step. Redraws landed
    after the next click (Details opened the reply that was last before the redraw). A
    sleep waits in Playwright, which keeps serving the requests."""
    waited: list[float] = []
    page = FakePage()
    page.wait_for_timeout = waited.append  # ty: ignore[unresolved-attribute]
    import time

    monkeypatch.setattr(time, "sleep", lambda seconds: waited.append(-seconds))
    qa.run_step(page, "sleep", "2.5", tmp_path, 1)
    assert waited == [2500]


@pytest.mark.parametrize("planted", ["local/models.toml", "local/models.lock", "a link"])
def test_a_file_planted_during_a_run_refuses_the_next_step(tmp_path, planted):
    """The QA-driver review's must 1 and its link should: Clarity reads the lock afresh at
    each model start and each time the picker opens, so before each step the root is
    checked again, as qa_session.py does before typing one. Step 1 runs; something is
    planted meanwhile; step 2 is refused before the page is touched."""
    root = tmp_path / "root"
    root.mkdir()
    page = FakePage()
    qa.run_step(page, "fill", "#m :: one", tmp_path, 1, root=root)
    (root / "local").mkdir()
    if planted == "a link":
        try:
            (root / "local" / "elsewhere").symlink_to(tmp_path)
        except OSError:
            pytest.skip("symlinks need privileges on Windows")
    else:
        (root / planted).write_text("", encoding="utf-8")
    with pytest.raises(qa.Refused):
        qa.run_step(page, "fill", "#m :: two", tmp_path, 2, root=root)
    assert page.fills == [("#m", "one")]
