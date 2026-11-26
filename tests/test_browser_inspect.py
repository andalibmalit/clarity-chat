"""The gate for v0.4.3's export for Inspect (G0, in the browser): what Inspect's viewer shows
for a log shaped like the plan's, before Clarity's writer exists. The logs are built with
inspect_ai's own types (`tests/inspect_gate_logs.py`); the offline half is
`test_inspect_gate.py`.

Run with `uv run pytest -m browser` (needs `uv sync --group browser --extra inspect`).
`inspect view` serves the logs on 127.0.0.1 with HOME set to a scratch folder; every request
the page makes is recorded, and any not to the viewer's own address is stopped and reported.
Each check saves a screenshot, and the page's text beside it, to a local-only folder.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = pytest.mark.browser
pytest.importorskip("inspect_ai")

from tests import inspect_gate_logs as gate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCREENS = REPO / "notes" / "research" / "v043-inspect-screens"
# What the page shows takes a moment after load: the viewer reads each log through its API.
SETTLE = 15_000


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args, browser_name):
    # Chrome on the Mac for Chromium only, as in test_browser.py.
    mac_chrome = sys.platform == "darwin" and browser_name == "chromium"
    channel = {"channel": "chrome"} if mac_chrome else {}
    return {**browser_type_launch_args, **channel}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Viewer:
    origin: str
    home: Path
    logs: dict[str, Path]
    requests: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)


@contextmanager
def serve(log_dir: Path, root: Path, logs: dict[str, Path]) -> Generator[Viewer]:
    """`inspect view` on LOG_DIR, on a free port, with HOME a scratch folder in ROOT."""
    home = root / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INSPECT_", "XDG_"))}
    port = free_port()
    command = [sys.executable, "-m", "inspect_ai", "view", "--log-dir", str(log_dir)]
    with (root / "view.out").open("w", encoding="utf-8") as out:
        server = subprocess.Popen(
            [*command, "--host", "127.0.0.1", "--port", str(port)],
            cwd=root,
            env={**env, "HOME": str(home)},
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    origin = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(origin, timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        yield Viewer(origin, home, logs)
    finally:
        server.terminate()
        server.wait(10)


@pytest.fixture(scope="module")
def viewer(tmp_path_factory) -> Iterator[Viewer]:
    """`inspect view` on the gate's logs, until the module's tests end."""
    root = tmp_path_factory.mktemp("viewer")
    with serve(root / "logs", root, gate.write_gate_logs(root / "logs")) as viewer:
        yield viewer


def watched(viewer: Viewer, context):
    """A page whose every request is recorded; one not to the viewer is stopped (V10)."""
    viewer.requests.clear()
    viewer.stopped.clear()

    def route(request_route):
        url = request_route.request.url
        viewer.requests.append(url)
        if url.startswith(f"{viewer.origin}/") or url.startswith(("data:", "blob:")):
            request_route.continue_()
        else:
            viewer.stopped.append(url)
            request_route.abort()

    context.route("**/*", route)
    page = context.new_page()
    page.set_viewport_size({"width": 1600, "height": 1000})
    return page


@pytest.fixture
def page(viewer, context):
    return watched(viewer, context)


def show(page, viewer: Viewer, route: str, waits_for: str) -> str:
    """The viewer at hash ROUTE, once WAITS_FOR shows: the page's text."""
    page.goto(f"{viewer.origin}/#{route}")
    page.get_by_text(waits_for, exact=False).first.wait_for(timeout=SETTLE)
    page.wait_for_load_state("networkidle")
    return page_text(page)


def page_text(page) -> str:
    """The page's text, each run of newlines made one: WebKit sometimes puts a blank line
    between two elements where Chromium doesn't (CI run 45601593181), and the asserts that
    read across elements shouldn't depend on it."""
    return re.sub(r"\n+", "\n", page.inner_text("body"))


def record(page, name: str) -> None:
    """A screenshot and the page's text, for the main session's rulings: notes/ is
    gitignored, so CI, which has no such folder, saves none."""
    if SCREENS.parent.is_dir():
        SCREENS.mkdir(exist_ok=True)
        page.screenshot(path=SCREENS / f"{name}.png", full_page=True)
        (SCREENS / f"{name}.txt").write_text(page.inner_text("body"), encoding="utf-8")


def finding(*parts: object) -> None:
    """A line for the main session's rulings, in the same local-only folder."""
    if SCREENS.parent.is_dir():
        SCREENS.mkdir(exist_ok=True)
        with (SCREENS / "findings.txt").open("a", encoding="utf-8") as f:
            f.write(" ".join(str(p) for p in parts) + "\n")


def sample_route(log: str, sample_id: object, epoch: int, tab: str) -> str:
    return f"/logs/{log}/samples/sample/{sample_id}/{epoch}/{tab}"


def only_local(viewer: Viewer) -> None:
    assert viewer.requests, "no request recorded"
    assert viewer.stopped == []


# --- The design-deciding checks first: V4 to V6


def test_v4_the_mark_score_shows_in_the_grid_the_logs_list_and_the_sample(viewer, page):
    """The samples grid names the score's column after it (MARK); the log's own list has
    Answer and Score columns; the sample's header shows both; a filter on `mark` works."""
    text = show(page, viewer, "/samples/run", "landed, too long")
    record(page, "V4-grid-every-reply-scored")
    assert "MARK" in text and "SCORE" not in text
    assert text.count("unmarked") == 6 and text.count("landed") == 2
    text = show(page, viewer, "/logs/run/1-smol-chat.eval", "Blue, green and grey.")
    record(page, "V4-log-list")
    assert "ANSWER" in text and "SCORE" in text and text.count("unmarked") == 3
    page.locator(".cm-content").first.click()
    page.keyboard.type('mark == "landed"')
    page.get_by_text("Blue, green and grey.").wait_for(state="detached", timeout=SETTLE)
    record(page, "V4-log-list-filtered")
    text = page_text(page)
    assert "Indented by three spaces." in text and "It whistles" not in text
    route = sample_route("run/1-smol-chat.eval", 1, 1, "scoring")
    text = show(page, viewer, route, "landed")
    record(page, "V4-sample-scoring")
    assert "ANSWER" in text and "SCORE" in text
    text = show(page, viewer, "/samples/marked-only", "landed")
    record(page, "V4-grid-marked-only")
    assert "unmarked" not in text and text.count("landed") == 1
    text = show(page, viewer, "/logs/marked-only/1-smol-chat.eval", "Name three colours")
    record(page, "V4-log-list-marked-only")
    unscored = [gate.REPLIES[("smol-chat", o, e)] for o, e in ((1, 2), (2, 1), (2, 2))]
    finding(
        "V4 marked-only log list, unscored replies' Answer shown:", [r in text for r in unscored]
    )
    assert "Indented by three spaces." in text
    only_local(viewer)


def test_v5_epochs_even_and_uneven(viewer, page):
    """Even epochs show as they are. Uneven ones list every reply, but the header and the
    count read as if two replies were missing: 2 samples x 3 epochs = 6, and 4 are there."""
    text = show(page, viewer, "/logs/run/1-smol-chat.eval", "Name three colours")
    record(page, "V5-even-epochs")
    assert "clarity-run — 2 x 2 samples" in text and "4 Samples" in text
    text = show(page, viewer, "/logs/chat/c0ffee.eval", "One colour of the sea?")
    record(page, "V5-uneven-epochs-list")
    assert "Two colours of the sea?" in text
    for reply in ("Grey.", "Blue.", "Green.", "Teal and navy."):
        assert reply in text, reply
    assert "clarity-chat — 1.3333333333333333 x 3 samples" in text
    assert "4 / 6 Samples" in text
    text = show(page, viewer, sample_route("chat/c0ffee.eval", "t1", 3, "messages"), "Green.")
    record(page, "V5-uneven-epoch-3")
    assert "Epoch 3" in text
    only_local(viewer)


def test_v6_the_samples_grid_lists_both_profiles_replies(viewer, page):
    """The cross-log grid lists all 8 replies with Task, Model, Log File, Id, Epoch, Input
    and the MARK score. It has no Answer column and no Tags column, and its Columns
    picker offers neither: the grid builds Answer only for a single log's list (spec
    departure, ruled after G0). Sorted by Id, each opening's replies sit together."""
    text = show(page, viewer, "/samples/run", "clarity/smol-text")
    record(page, "V6-grid")
    for value in ("clarity/smol-chat", "clarity/smol-text", "clarity-run", *gate.OPENINGS.values()):
        assert value in text, value
    for header in ("TASK", "MODEL", "LOG FILE", "ID", "EPOCH", "INPUT", "MARK"):
        assert f"\n{header}\n" in text, header
    assert "ANSWER" not in text and "TAGS" not in text
    assert "8 rows" in text and "Blue, green and grey." not in text
    page.get_by_text("Id", exact=True).first.click()  # the header; CSS shows it as ID
    page.wait_for_timeout(1500)
    record(page, "V6-grid-sorted-by-id")
    text = page_text(page)
    assert "\nID\n" in text, "the Id column is still shown"
    rows = [line for line in text.split("\n") if line in gate.OPENINGS.values()]
    assert len(rows) == 8 and len({tuple(rows[:4]), tuple(rows[4:])}) == 2, rows
    for half in (rows[:4], rows[4:]):  # each opening's 4 replies, both profiles', together
        assert len(set(half)) == 1, rows
    page.get_by_text("Columns", exact=True).first.click()
    page.wait_for_timeout(1000)
    record(page, "V6-grid-columns-picker")
    picker = page_text(page).split("8 rows", 1)[0]
    finding("V6 columns picker:", picker.split("Columns", 1)[-1].replace("\n", " "))
    assert "Answer" not in picker and "Tags" not in picker and "\nmark\n" in picker
    only_local(viewer)


# --- The rest decide the docs


def test_v1_the_tags_and_the_note(viewer, page):
    """The tags show beside the task's name at the top of a log; the log's metadata, with
    `inspect_note`, shows on its Info tab."""
    text = show(page, viewer, "/logs/run/1-smol-chat.eval", "not-run-by-inspect")
    record(page, "V1-tags-in-log-header")
    assert "clarity\nnot-run-by-inspect" in text
    show(page, viewer, "/logs/run/1-smol-chat.eval/info", "clarity-run")
    page.wait_for_timeout(2000)
    record(page, "V1-log-info")
    text = page_text(page)
    finding("V1 note on the Info tab:", "Clarity made these calls" in text)
    assert "Clarity made these calls" in text
    only_local(viewer)


def test_v2_a_samples_metadata_with_a_long_prompt_ids(viewer, page):
    text = show(page, viewer, sample_route("run/1-smol-chat.eval", 2, 2, "info"), "prompt_ids")
    record(page, "V2-sample-metadata-long")
    for key in ("call_id", "prompt_text", "message_spans", "output_text", "settings"):
        assert key in text, key
    text = show(page, viewer, sample_route("run/1-smol-chat.eval", 1, 2, "info"), "prompt_ids")
    record(page, "V2-sample-metadata-short")
    only_local(viewer)


def test_v3_a_messages_metadata(viewer, page):
    """A user message's metadata shows under it, folded: `clarity:` with Object(3) for a
    span (Object(1) for a whole text-mode prompt). The system message's doesn't show."""
    route = sample_route("run/1-smol-chat.eval", 1, 2, "messages")
    text = show(page, viewer, route, "Blue, green and grey.")
    record(page, "V3-messages-chat-mode")
    assert text.count("METADATA\nclarity:\nObject(3)") == 1
    system, user = text.index(gate.SYSTEM), text.index("USER\n")
    assert "METADATA" not in text[system:user] and "METADATA" in text[user:]
    page.get_by_text("Object(3)").first.click()
    page.wait_for_timeout(1000)
    record(page, "V3-messages-chat-mode-opened")
    text = page_text(page)
    finding("V3 opened span shows chars:", "chars" in text, "role" in text)
    route = sample_route("run/2-smol-text.eval", 1, 1, "messages")
    text = show(page, viewer, route, "Object(1)")
    record(page, "V3-messages-text-mode")
    assert "METADATA\nclarity:\nObject(1)" in text
    only_local(viewer)


def test_v7_an_empty_generate_config_and_no_times(viewer, page):
    """The empty generate config shows nothing: the Task tab's Config lists only `epochs`.
    The log has no start or end time, which shows as 1970's epoch (in local time) and a
    duration of NaN, at the top of every log and on the Task tab."""
    text = show(page, viewer, "/logs/run/1-smol-chat.eval/models", "clarity/smol-chat")
    record(page, "V7-models-tab")
    assert "DURATION\nNaN days NaN hr NaN min NaN sec" in text
    text = show(page, viewer, "/logs/run/1-smol-chat.eval/task", "TASK INFO")
    record(page, "V7-task-tab")
    assert "1969-12-31" in text or "1970-01-01" in text
    assert text.rstrip().endswith("CONFIG\nepochs\n2")
    only_local(viewer)


def test_v8_the_model_shows_with_no_error(viewer, page):
    text = show(page, viewer, "/logs/run/1-smol-chat.eval", "clarity/smol-chat")
    record(page, "V8-log")
    assert "error" not in text.lower()
    only_local(viewer)


def test_v9_html_in_a_reply_shows_as_text_and_nothing_runs(viewer, page):
    route = sample_route("run/2-smol-text.eval", 2, 2, "messages")
    text = show(page, viewer, route, "plain after")
    record(page, "V9-html-reply")
    assert page.evaluate("window.gateRan") is None
    images = page.locator(f'img[src="{gate.REMOTE_IMAGE}"]').count()
    finding("V9 script text shown:", "<script>" in text, "img elements:", images)
    assert not any(gate.REMOTE_IMAGE in url for url in viewer.requests)
    only_local(viewer)


def test_v9b_html_in_an_opening_and_a_note_shows_as_text_and_nothing_runs(viewer, page):
    """v0.4.3's review, C4: V9's check for the other text a person writes, an opening (the
    sample's input, in the log's list and in the sample) and a note (the mark score's
    explanation, on the sample's Scoring tab, beside the note "reads well")."""
    scoring = sample_route("run/1-smol-chat.eval", 2, 1, "scoring")
    for route, waits_for, name in (
        ("/logs/run/1-smol-chat.eval", "plain after", "V9b-html-opening-list"),
        (sample_route("run/1-smol-chat.eval", 2, 1, "messages"), "plain after", "V9b-html-opening"),
        (scoring, "reads well", "V9b-html-note"),
    ):
        text = show(page, viewer, route, waits_for)
        record(page, name)
        assert page.evaluate("window.gateRan") is None, name
        images = page.locator(f'img[src="{gate.REMOTE_IMAGE}"]').count()
        finding(f"{name} script text shown:", "<script>" in text, "img elements:", images)
    assert not any(gate.REMOTE_IMAGE in url for url in viewer.requests)
    only_local(viewer)


def test_v10_no_request_leaves_the_viewers_address(viewer, page):
    routes = {
        "/samples/run": "landed",
        "/logs/run/2-smol-text.eval/info": "clarity-run",
        sample_route("run/2-smol-text.eval", 2, 2, "messages"): "plain after",
        sample_route("run/1-smol-chat.eval", 2, 2, "info"): "prompt_ids",
        "/logs/chat/c0ffee.eval": "Teal and navy.",
    }
    for route, waits_for in routes.items():
        show(page, viewer, route, waits_for)
    only_local(viewer)
    finding("V10 requests:", sorted({u.split("?")[0] for u in viewer.requests}))


def test_v11_markdown_in_a_reply(viewer, page):
    """The reply is rendered as Markdown: its leading spaces are dropped and `* ` starts a
    bulleted line, so the star is gone; the `<think>` tag shows as text. The header's
    Answer shows the text on one line, its whitespace collapsed."""
    route = sample_route("run/1-smol-chat.eval", 1, 1, "messages")
    text = show(page, viewer, route, "a starred line")
    record(page, "V11-markdown-reply")
    assert "   Indented" not in text
    assert "Indented by three spaces. * a starred line <think>a thought</think> after it" in text
    # The blank line is the reply's own paragraph break, so this reads the page as it is
    assert "\n\na starred line\n<think>a thought</think> after it" in page.inner_text("body")
    assert text.count("* a starred line") == 1  # the header's, not the message's
    page.get_by_text("Raw", exact=True).first.click()
    page.wait_for_timeout(1500)
    record(page, "V11-raw")
    raw = page.inner_text("body")
    finding(
        "V11 Raw: three leading spaces kept:",
        "   Indented by three spaces." in raw,
        "'* a starred line' kept as text:",
        raw.count("* a starred line") > 1,
        "<think> kept:",
        raw.count("<think>a thought</think>") > 1,
    )
    only_local(viewer)


def test_v12_what_inspect_view_writes_under_home(viewer, page):
    """`inspect view` writes a trace file and its port's pid file in Inspect's data folder
    under HOME, and neither holds sample text."""
    show(page, viewer, sample_route("run/1-smol-chat.eval", 1, 2, "messages"), "Blue, green")
    files = sorted(p for p in viewer.home.rglob("*") if p.is_file())
    finding("V12 files:", [str(p.relative_to(viewer.home)) for p in files])
    names = sorted(f"{p.parent.name}/{p.suffix}" for p in files)
    assert names == ["ports/.pid", "traces/.log"], files
    for path in files:
        content = path.read_bytes()
        finding("V12", path.name, len(content), content[:400])
        for secret in (*gate.OPENINGS.values(), gate.SYSTEM, "Blue, green and grey."):
            assert secret.encode() not in content, (path, secret)
