"""The browser QA driver's `upload` step (v0.3.1, Lane 2): it gives the page only a file
under the scratch root. The driver reads the file outside the sandbox, so a path or link
outside the root would carry a file the sandbox denies into the page, the log and the
transcript. The path is judged when the steps are read and again when the step runs, with
links followed. End to end it needs the page's `/upload` route (v0.3.1 B3); this is the
guard alone."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_browser", REPO / "scripts" / "qa_browser.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "notes.md").write_text("made-up notes\n", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("not for the page\n", encoding="utf-8")
    (root / "docs" / "away.md").symlink_to(outside)
    return root


def steps(root: Path, *lines: str) -> Path:
    path = root.parent / "steps.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakePage:
    def __init__(self):
        self.files = []

    def set_input_files(self, selector, files):
        self.files.append((selector, files))


def test_an_upload_inside_the_root_parses_and_gives_the_page_that_file(root):
    try:
        parsed = qa.parse_steps(steps(root, "upload #attach-file :: docs/notes.md"), root=root)
    except qa.Refused as e:  # what the driver did before it had the step
        parsed = [("refused", str(e))]
    assert parsed == [("upload", "#attach-file :: docs/notes.md")], parsed
    # A run's root holds no link (prepare_root refuses one, and each step checks again).
    (root / "docs" / "away.md").unlink()
    page = FakePage()
    qa.run_step(page, "upload", "#attach-file :: docs/notes.md", root.parent, 1, root=root)
    assert page.files == [("#attach-file", root.resolve() / "docs" / "notes.md")]


@pytest.mark.parametrize(
    "path", ["../secret.txt", "docs/away.md", "~/made-up-qa-file.md", "/etc/hosts", "docs/none.md"]
)
def test_an_upload_outside_the_root_or_of_no_file_is_refused_when_read(root, path):
    with pytest.raises(qa.Refused, match="scratch root"):
        qa.parse_steps(steps(root, f"upload #attach-file :: {path}"), root=root)


def test_a_link_planted_after_the_steps_were_read_is_refused_when_the_step_runs(root, tmp_path):
    """Refused before the upload is judged: each step scans the root for links first."""
    (root / "docs" / "later.md").symlink_to(tmp_path / "secret.txt")
    page = FakePage()
    with pytest.raises(qa.Refused, match="is a link"):
        qa.run_step(page, "upload", "#attach-file :: docs/later.md", root.parent, 1, root=root)
    assert page.files == []


@pytest.mark.skipif(not hasattr(os, "link"), reason="no hard links")
def test_a_file_with_a_second_name_is_refused(root, tmp_path):
    os.link(tmp_path / "secret.txt", root / "docs" / "twin.md")
    with pytest.raises(qa.Refused, match="scratch root"):
        qa.parse_steps(steps(root, "upload #attach-file :: docs/twin.md"), root=root)


# --- the Fable review (notes/research/reviews/v031-qa-driver-review.md) ------------------


def test_a_link_whose_name_ends_in_a_space_is_refused_when_the_step_runs(root, tmp_path):
    """Should 2: the page must get the very path that was judged, unstripped."""
    (root / "docs" / "later.md ").symlink_to(tmp_path / "secret.txt")
    page = FakePage()
    try:
        qa.run_step(page, "upload", "#attach-file :: docs/later.md ", root.parent, 1, root=root)
    except qa.Refused:
        pass
    assert page.files == []


@pytest.mark.parametrize("path", ["~nosuchuser-for-qa/x.md", "docs/notes.md/../notes.md"])
def test_a_path_the_guard_cant_judge_or_with_dot_dot_is_refused(root, path):
    """Could 1 (an unknown ~user crashed the driver) and should 4 (`..`)."""
    try:
        got = qa.parse_steps(steps(root, f"upload #attach-file :: {path}"), root=root)
    except Exception as e:  # a crash is what the driver did, not what it should
        got = e
    assert isinstance(got, qa.Refused), got
