"""The terminal QA driver's guard on v0.3.2's document library: `clarity document add FILE`
and `/documents add PATH [NAME]` read a file, and the driver runs Clarity outside the
sandbox, so the path must name a file in the scratch root, judged as /attach's is (v0.3.2,
the QA-driver lane). `/documents add` reads `PATH NAME` whole when that exists, and
otherwise splits off the last word as the name: both readings are judged."""

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
    (root / "docs" / "notes.md").write_bytes(b"made-up notes\n")
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"not for the transcript\n")
    (root / "docs" / "away.md").symlink_to(outside)
    return root


def steps(root: Path, *lines: str) -> Path:
    path = root.parent / "steps.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def outcome(call) -> object:
    """What `call` returned, or what it raised: a crash or a refusal is the driver's answer."""
    try:
        return call()
    except (Exception, SystemExit) as e:
        return e


OUTSIDE = [
    "../secret.txt",
    "docs/away.md",  # a link inside the root to a file outside
    "~/made-up-qa-file.md",
    "/etc/hosts",
    "docs/notes.md/../notes.md",  # `..` after a file
]


# --- clarity document add FILE


def test_document_add_inside_the_root_is_allowed(root):
    got = outcome(lambda: qa.check_clarity_args(root, ["document", "add", "docs/notes.md"]))
    assert not isinstance(got, BaseException), got


@pytest.mark.parametrize("path", OUTSIDE)
def test_document_add_outside_the_root_is_refused(root, path):
    with pytest.raises(qa.Refused, match="scratch root|'\\.\\.'|second name"):
        qa.check_clarity_args(root, ["document", "add", path, "--name", "made-up"])


def test_the_other_document_actions_are_allowed(root):
    for argv in (["document", "list"], ["document", "check"], ["document", "rebuild", "notes"]):
        got = outcome(lambda argv=argv: qa.check_clarity_args(root, argv))
        assert not isinstance(got, BaseException), (argv, got)


# --- /documents add PATH [NAME], typed or in --json


@pytest.mark.parametrize("path", OUTSIDE)
@pytest.mark.parametrize(
    "line",
    [
        "/documents add {path}",
        "/documents add {path} made-up-name",  # the last word may be the name
        "/doc add {path}",  # a prefix may name it too
        "//documents add {path}",  # a message a cancel would put back as the command
        "/say /documents add {path}",
    ],
)
def test_a_documents_add_step_outside_the_root_is_refused(root, path, line):
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, line.format(path=path)), json_mode=False, root=root)


@pytest.mark.parametrize("path", OUTSIDE)
def test_a_documents_add_command_in_json_outside_the_root_is_refused(root, path):
    line = json.dumps({"command": "documents", "args": f"add {path}"})
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, line), json_mode=True, root=root)


def test_documents_steps_inside_the_root_pass_as_written(root):
    lines = [
        "/documents",
        "/documents add docs/notes.md",
        "/documents add docs/notes.md made-up-name",
        "/documents rebuild made-up-name",
    ]
    assert qa.parse_steps(steps(root, *lines), json_mode=False, root=root) == lines
    line = json.dumps({"command": "documents", "args": "add docs/notes.md"})
    assert qa.parse_steps(steps(root, line), json_mode=True, root=root) == [line]


# --- the v0.3.2 driver review's could 1: each reading pinned on its own


def test_a_link_whose_name_holds_a_space_is_judged_whole(root, tmp_path):
    """Clarity reads the rest whole when it exists: only the whole reading sees this link."""
    (root / "docs" / "away two.md").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, "/documents add docs/away two.md"), json_mode=False, root=root)


def test_a_path_ending_in_a_space_is_judged_as_clarity_splits_it(root, tmp_path):
    """Two spaces before the name: Clarity's split leaves a path ending in a space, which
    only the unstripped rsplit reading holds."""
    (root / "docs" / "sp.md ").symlink_to(tmp_path / "secret.txt")
    line = "/documents add docs/sp.md  made-up-name"
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, line), json_mode=False, root=root)


@pytest.mark.skipif(not hasattr(os, "link"), reason="no hard links")
def test_a_file_with_a_second_name_is_refused_both_ways(root, tmp_path):
    os.link(tmp_path / "secret.txt", root / "docs" / "twin.md")
    with pytest.raises(qa.Refused, match="second name"):
        qa.parse_steps(steps(root, "/documents add docs/twin.md"), json_mode=False, root=root)
    with pytest.raises(qa.Refused, match="second name"):
        qa.check_clarity_args(root, ["document", "add", "docs/twin.md"])


# --- --replace before PATH (since 6839b3e): every reading Clarity can make is judged


@pytest.mark.parametrize("path", OUTSIDE)
@pytest.mark.parametrize(
    "line",
    [
        "/documents add --replace {path}",
        "/documents add --replace {path} made-up-name",
        "/documents add  --replace  {path} made-up-name",  # Clarity strips the spaces
    ],
)
def test_a_documents_add_replace_step_outside_the_root_is_refused(root, path, line):
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, line.format(path=path)), json_mode=False, root=root)


@pytest.mark.parametrize("path", OUTSIDE)
def test_a_documents_add_replace_in_json_outside_the_root_is_refused(root, path):
    line = json.dumps({"command": "documents", "args": f"add --replace {path} made-up-name"})
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(root, line), json_mode=True, root=root)


def test_a_documents_add_replace_step_inside_the_root_passes(root):
    lines = ["/documents add --replace docs/notes.md notes"]
    assert qa.parse_steps(steps(root, *lines), json_mode=False, root=root) == lines


@pytest.mark.parametrize("path", OUTSIDE)
@pytest.mark.parametrize("where", ["after", "before"])
def test_document_add_replace_outside_the_root_is_refused(root, path, where):
    argv = ["document", "add", path, "--replace"] if where == "after" else [
        "document", "add", "--replace", path]  # fmt: skip
    with pytest.raises(qa.Refused, match="scratch root|'\\.\\.'|second name"):
        qa.check_clarity_args(root, argv)
