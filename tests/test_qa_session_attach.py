"""The terminal QA driver's guard on attachments (v0.3.1, Lane 2). The driver runs Clarity
outside the sandbox, so an attachment path outside the scratch root would let a QA step read
a file the sandbox denies into a transcript the agent reads. Every reading of the path is
judged: as typed, with `~` expanded, and with a trailing `check`, `pages A-B` or `lines A-B`
removed, each resolved from the root with links followed."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A scratch root holding notes.md, plus a link inside it to a file outside."""
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "notes.md").write_text("made-up notes\n", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("not for the transcript\n", encoding="utf-8")
    (root / "docs" / "away.md").symlink_to(outside)
    return root


OUTSIDE = [
    "../secret.txt",  # up out of the root
    "docs/away.md",  # a link inside the root to a file outside
    "~/made-up-qa-file.md",  # Clarity, or a shell, may expand ~
    "/etc/hosts",
    "/etc/hosts check",
    "../secret.txt pages 1-2",
    "../secret.txt lines 3-9",
    "docs/away.md lines 1-2 check",  # Clarity removes both; the review's should 1
    "docs/away.md pages 1-2 check",
    "docs/notes.md/../notes.md",  # `..` after a file: the system and resolve() differ
]


@pytest.mark.parametrize("path", OUTSIDE)
def test_an_attach_option_outside_the_root_is_refused(root, path):
    with pytest.raises(qa.Refused):
        qa.check_clarity_args(root, ["chat", "--profile", "smol-chat", "--attach", path])


@pytest.mark.parametrize("path", ["docs/notes.md", "ROOT/docs/notes.md"])
def test_an_attach_option_inside_the_root_is_allowed(root, path):
    path = path.replace("ROOT", str(root))
    qa.check_clarity_args(root, ["chat", "--profile", "smol-chat", "--attach", path])


def steps(root: Path, *lines: str) -> Path:
    path = root.parent / "steps.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("path", OUTSIDE)
@pytest.mark.parametrize("command", ["/attach", "/att"])  # a prefix may name it too
def test_an_attach_step_outside_the_root_is_refused(root, path, command):
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, f"{command} {path}"), json_mode=False, root=root)


@pytest.mark.parametrize("path", OUTSIDE)
def test_an_attach_command_in_json_outside_the_root_is_refused(root, path):
    line = json.dumps({"command": "attach", "args": path})
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, line), json_mode=True, root=root)


def test_attach_steps_inside_the_root_pass_as_written(root):
    lines = [
        "/attach docs/notes.md",
        "/attach docs/notes.md lines 1-5",
        "/attach docs/notes.md check",
    ]
    assert qa.parse_steps(steps(root, *lines), json_mode=False, root=root) == lines
    line = json.dumps({"command": "attach", "args": f"{root}/docs/notes.md check"})
    assert qa.parse_steps(steps(root, line), json_mode=True, root=root) == [line]


@pytest.mark.skipif(not hasattr(os, "link"), reason="no hard links")
def test_a_file_with_a_second_name_is_refused(root, tmp_path):
    """A hard link inside the root to a file outside: it resolves inside, but it's the
    outside file."""
    os.link(tmp_path / "secret.txt", root / "docs" / "twin.md")
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, "/attach docs/twin.md"), json_mode=False, root=root)


# --- the Fable review of the guard (notes/research/reviews/v031-qa-driver-review.md) ------


def refusal(call) -> object:
    """What `call` raised (or returned): a Refused, or what the driver did instead."""
    try:
        return call()
    except Exception as e:  # a crash is what the driver did, not what it should
        return e


@pytest.mark.parametrize(
    "line",
    [
        "//attach ../secret.txt",  # sent as a message; after an early cancel it's the draft
        "///attach ../secret.txt",  # one slash fewer each time round
        "/say /attach docs/away.md",
        "/edit /attach ../secret.txt",
        "/say //attach ../secret.txt",
    ],
)
def test_a_message_that_would_come_back_as_an_attach_command_is_judged(root, line):
    """Must 1: a cancelled message goes back in the input with one slash fewer, and the next
    typed line sends it as a command."""
    lines = [line, "<ctrl-c after 0.1>", " ", "hello"]
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, *lines), json_mode=False, root=root)


def test_messages_that_start_with_a_slash_and_attach_inside_pass(root):
    lines = ["//hello there", "/say /not-a-command", "//attach docs/notes.md"]
    assert qa.parse_steps(steps(root, *lines), json_mode=False, root=root) == lines


def test_a_link_whose_name_ends_in_a_space_is_judged_as_clarity_opens_it(root, tmp_path):
    """Should 1: --attach and read_files open the value as given, unstripped."""
    (root / "docs" / "sp.md ").symlink_to(tmp_path / "secret.txt")
    argv = ["chat", "--profile", "smol-chat", "--attach", "docs/sp.md "]
    with pytest.raises(qa.Refused):
        qa.check_clarity_args(root, argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--attach=../secret.txt"],
        ["--att", "../secret.txt"],
        ["--attach", "docs/notes.md", "--attach", "../secret.txt"],
    ],
)
def test_every_spelling_of_the_option_is_judged(root, argv):
    with pytest.raises(qa.Refused):
        qa.check_clarity_args(root, ["chat", "--profile", "smol-chat", *argv])


def test_a_step_line_over_1000_bytes_is_refused(root):
    """Should 4: a terminal in line mode limits a line to 1,024 bytes on macOS, so Clarity
    might get less than the line the guard judged."""
    long = "made-up words " * 72  # any step: Clarity reads every line the same way
    assert len(long.encode()) > 1000
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, long), json_mode=False, root=root)


@pytest.mark.parametrize("args", ["x\u0000", "\ud800.md"])
def test_a_path_the_guard_cant_judge_is_refused_not_a_crash(root, args):
    """Could 1: a NUL or a lone surrogate."""
    line = json.dumps({"command": "attach", "args": args})  # escaped, so no raw control
    got = refusal(lambda: qa.parse_steps(steps(root, line), json_mode=True, root=root))
    assert isinstance(got, qa.Refused), got


def test_a_format_character_in_a_command_name_is_refused(root):
    """Could 2: a zero-width space the line editor may pass through."""
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, "/att\u200bach ../secret.txt"), json_mode=False, root=root)


# --- main's order, and the check again before each step is typed (should 3) -------------


@pytest.fixture
def driven(root, tmp_path, monkeypatch):
    """main() with the terminal faked: the order of its stages, and what it typed."""
    (root / "docs" / "away.md").unlink()  # the fixture's link would fail the start scan
    happened: list[str] = []
    typed: list[bytes] = []
    read_fd, write_fd = os.pipe()
    os.close(write_fd)

    def fork():
        happened.append("clarity started")
        return 4242, read_fd

    def prepare_root(path, init=False):
        happened.append("root scanned")
        qa.no_links(path)

    real_parse = qa.parse_steps

    def parse_steps(*args, **kwargs):
        happened.append("steps judged")
        return real_parse(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "pty", types.SimpleNamespace(fork=fork))
    monkeypatch.setattr(qa, "QA_DIR", tmp_path)
    monkeypatch.setattr(qa, "prepare_root", prepare_root)
    monkeypatch.setattr(qa, "parse_steps", parse_steps)
    monkeypatch.setattr(qa, "send", lambda fd, log, data: typed.append(data))
    monkeypatch.setattr(qa, "wait", lambda pid: 0)

    def run(*lines: str, between=None) -> tuple[int, str]:
        reads = iter(range(1000))

        def read_until_quiet(fd, log, timeout, quiet=0.0):
            if next(reads) == 0 and between:  # after the start, before the first step
                between()

        monkeypatch.setattr(qa, "read_until_quiet", read_until_quiet)
        out = tmp_path / "out.txt"
        argv = ["qa_session.py", "--root", str(root), "--out", str(out)]
        argv += ["--steps", str(steps(root, *lines)), "--", "chat", "--profile", "smol-chat"]
        monkeypatch.setattr(sys, "argv", argv)
        status = qa.main()
        return status, out.read_text(encoding="utf-8", errors="replace")

    return happened, typed, run


def test_main_judges_the_steps_and_scans_the_root_before_clarity_starts(driven):
    happened, typed, run = driven
    status, _ = run("/attach docs/notes.md")
    assert status == 0
    assert happened == ["steps judged", "root scanned", "clarity started"]
    assert typed == [b"/attach docs/notes.md\r", b"\x04"]


def test_main_refuses_bad_steps_before_clarity_starts(driven):
    happened, _, run = driven
    with pytest.raises(SystemExit):
        run("/attach ../secret.txt")
    assert "clarity started" not in happened


def test_a_link_planted_after_the_start_is_caught_before_its_step_is_typed(driven, root, tmp_path):
    """Should 3: the steps were judged before Clarity started; an agent acting beside its
    own run could plant a link during a long wait."""
    _, typed, run = driven

    def plant():
        (root / "docs" / "notes.md").unlink()
        (root / "docs" / "notes.md").symlink_to(tmp_path / "secret.txt")

    status, transcript = run("hello", "/attach docs/notes.md", between=plant)
    assert not any(b"attach" in data for data in typed), typed
    assert "[qa:" in transcript and "not typed" in transcript, transcript


# --- the reviewer's confirmation: two coulds -----------------------------------------------


@pytest.mark.parametrize(
    ("line", "refused"),
    [("/" * 999, False), ("/" * 900 + "attach ../secret.txt", True)],
)
def test_a_step_of_many_slashes_is_judged_without_crashing(root, line, refused):
    """A line of slashes went one recursion deep per slash, so 999 raised RecursionError."""
    got = refusal(lambda: qa.parse_steps(steps(root, line), json_mode=False, root=root))
    assert isinstance(got, qa.Refused) if refused else got == [line], got


def test_a_file_gone_between_the_listing_and_its_check_is_skipped(root, monkeypatch):
    """no_links runs again while Clarity runs, and Clarity removes files of its own."""
    (root / "docs" / "away.md").unlink()
    real_walk = os.walk

    def walk(top, followlinks=False):
        yield str(root), [], ["gone-by-now.txt"]  # listed, then removed before the lstat
        yield from real_walk(top, followlinks=followlinks)

    monkeypatch.setattr(qa.os, "walk", walk)
    assert refusal(lambda: qa.no_links(root)) is None


def test_attach_args_failing_for_another_reason_is_refused_not_skipped(root, monkeypatch):
    """The v0.3.2 review's could 2: if Clarity's attach_args ever raised a ValueError, the
    guard dropped its readings and went on judging the rest."""
    from clarity_chat import commands

    def raises(args):
        raise ValueError("made up")

    monkeypatch.setattr(commands, "attach_args", raises)
    assert qa.attach_problem(root, "docs/notes.md") is not None
