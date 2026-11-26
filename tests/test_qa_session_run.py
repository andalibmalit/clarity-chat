"""The terminal QA driver's guard on `clarity run`, `mark` and `note` (v0.4.3's QA pass makes
a batch run). The driver runs Clarity outside the sandbox, so `run --openings FILE` must name
a file in the scratch root, judged as `document add`'s FILE is; `--profiles` takes names
only, as `chat --profile` does; a mark's or note's log must be inside the root, as replay's
is; and a mark's label and a note's text may hold no control character, as a step may not."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A scratch root holding openings and a log, plus links inside it to files outside."""
    root = tmp_path / "root"
    (root / "logs").mkdir(parents=True)
    (root / "openings.txt").write_bytes(b"Made-up opening one\n---\nMade-up opening two\n")
    (root / "logs" / "a.jsonl").write_bytes(b"{}\n")
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"not for the transcript\n")
    (root / "away.txt").symlink_to(outside)
    (root / "logs" / "away.jsonl").symlink_to(outside)
    return root


def outcome(call) -> object:
    """What `call` returned, or what it raised: a crash or a refusal is the driver's answer."""
    try:
        return call()
    except (Exception, SystemExit) as e:
        return e


# --- clarity run --openings FILE --profiles NAME…


def test_a_run_inside_the_root_is_allowed(root):
    argv = ["run", "--openings", "openings.txt", "--profiles", "smol-chat", "smol-text"]
    got = outcome(lambda: qa.check_clarity_args(root, [*argv, "--seeds", "3"]))
    assert not isinstance(got, BaseException), got


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "away.txt",  # a link inside the root to a file outside
        "~/made-up-qa-openings.txt",
        "/etc/hosts",
        "openings.txt/../openings.txt",  # `..` after a file
    ],
)
def test_run_openings_outside_the_root_are_refused(root, path):
    with pytest.raises(qa.Refused, match="scratch root|'\\.\\.'"):
        qa.check_clarity_args(root, ["run", "--openings", path, "--profiles", "smol-chat"])


@pytest.mark.skipif(not hasattr(os, "link"), reason="no hard links")
def test_run_openings_with_a_second_name_are_refused(root, tmp_path):
    os.link(tmp_path / "secret.txt", root / "twin.txt")
    with pytest.raises(qa.Refused, match="second name"):
        qa.check_clarity_args(root, ["run", "--openings", "twin.txt", "--profiles", "smol-chat"])


@pytest.mark.parametrize(
    "profiles",
    [
        ["/x/evil.toml"],
        ["smol-chat", "../evil"],  # the second of several
        ["profiles/smol-chat.toml"],  # a path inside the root: names only here
    ],
)
def test_run_profiles_take_names_only(root, profiles):
    argv = ["run", "--openings", "openings.txt", "--profiles", *profiles]
    with pytest.raises(qa.Refused, match="profile names"):
        qa.check_clarity_args(root, argv)


# --- clarity mark LOG CALL [LABEL] and clarity note LOG CALL TEXT


@pytest.mark.parametrize(
    "argv",
    [
        ["mark", "logs/a.jsonl", "c1"],
        ["mark", "logs/a.jsonl", "c1", "made-up label"],
        ["mark", "logs/a.jsonl", "c1", "--remove"],
        ["note", "logs/a.jsonl", "c1", "A made-up note, with ünïcode and a / slash."],
    ],
)
def test_a_mark_or_note_on_a_log_in_the_root_is_allowed(root, argv):
    got = outcome(lambda: qa.check_clarity_args(root, argv))
    assert not isinstance(got, BaseException), got


@pytest.mark.parametrize("command", [["mark"], ["note", "made-up note"]])
@pytest.mark.parametrize(
    "log",
    [
        "../secret.txt",
        "logs/away.jsonl",  # a link inside the root to a file outside
        "/etc/hosts",
    ],
)
def test_a_mark_or_note_on_a_log_outside_the_root_is_refused(root, command, log):
    argv = [command[0], log, "c1", *command[1:]]
    with pytest.raises(qa.Refused, match="must be a path inside"):
        qa.check_clarity_args(root, argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["note", "logs/a.jsonl", "c1", "made-up\x01note"],  # Ctrl-A
        ["note", "logs/a.jsonl", "c1", "\x1b[2Jmade-up note"],  # an escape: clears the screen
        ["note", "logs/a.jsonl", "c1", "made-up\nnote"],
        ["note", "logs/a.jsonl", "c1", "made-up note"],  # a line separator
        ["mark", "logs/a.jsonl", "c1", "\x1b[31mmade-up"],  # a mark's label is written too
    ],
)
def test_a_control_character_in_a_marks_or_notes_text_is_refused(root, argv):
    with pytest.raises(qa.Refused, match="control character"):
        qa.check_clarity_args(root, argv)
