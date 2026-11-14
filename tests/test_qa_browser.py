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
        "press Enter; rm",
        "shot ../../elsewhere",
        "sleep forever",
        "width wide",
        "reload now",
    ],
)
def test_steps_outside_the_list_are_refused(tmp_path, line):
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps(tmp_path, line))


def test_paths_outside_notes_qa_are_refused(tmp_path, monkeypatch):
    path = steps(tmp_path, "reload")
    argv = ["qa_browser.py", "--root", str(tmp_path), "--out", str(tmp_path), "--steps", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        qa.main()
    assert caught.value.code == 2
