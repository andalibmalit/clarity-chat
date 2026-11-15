"""The terminal QA driver's guard on attachments (v0.3.1, Lane 2). The driver runs Clarity
outside the sandbox, so an attachment path outside the scratch root would let a QA step read
a file the sandbox denies into a transcript the agent reads. Every reading of the path is
judged: as typed, with `~` expanded, and with a trailing `check`, `pages A-B` or `lines A-B`
removed, each resolved from the root with links followed."""

from __future__ import annotations

import importlib.util
import json
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
    "~/.ssh/id_ed25519",  # Clarity, or a shell, may expand ~
    "/etc/hosts",
    "/etc/hosts check",
    "../secret.txt pages 1-2",
    "../secret.txt lines 3-9",
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
