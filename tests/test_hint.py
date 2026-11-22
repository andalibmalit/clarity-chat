"""v0.4.2 A8: the fetch hint names Clarity's folder (decision 14; option (a), the main
session's ruling). run_command sets the folder once it knows the project root and clears
it when the command ends; unset, the hint keeps its words from before."""

from __future__ import annotations

from pathlib import Path

import pytest

from clarity_chat import profiles
from clarity_chat.profiles import fetch_hint

BEFORE = (
    "In a new terminal window, in Clarity's folder, run: uv run clarity fetch tiny. Then try again."
)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def named(folder: Path) -> str:
    profiles.set_hint_folder(folder)
    try:
        return fetch_hint("tiny")
    finally:
        profiles.set_hint_folder(None)


def test_unset_the_hint_keeps_its_words():
    assert fetch_hint("tiny") == BEFORE


def test_under_the_home_folder_the_hint_names_it_from_there(home):
    assert named(home / "Clarity") == (
        "In a new Terminal window, run: cd ~/Clarity && uv run clarity fetch tiny. Then try again."
    )


def test_outside_the_home_folder_the_hint_names_the_whole_path(home, tmp_path):
    assert named(tmp_path / "elsewhere" / "Clarity") == (
        f"In a new Terminal window, run: cd {tmp_path}/elsewhere/Clarity && uv run clarity "
        "fetch tiny. Then try again."
    )


def test_a_folder_with_a_space_and_a_quote_is_quoted_so_the_shell_reads_it_back(home):
    import shlex

    said = named(home / "It's my Clarity")
    assert said.startswith("In a new Terminal window, run: cd ~/'It'")
    command = said.removeprefix("In a new Terminal window, run: ").removesuffix(". Then try again.")
    words = shlex.split(command)
    assert words[:2] == ["cd", "~/It's my Clarity"]  # the shell expands ~/ before the quote
    assert words[2:] == ["&&", "uv", "run", "clarity", "fetch", "tiny"]


def test_a_command_run_sets_the_folder_and_clears_it_when_it_ends(root, monkeypatch, home):
    from clarity_chat import cli

    seen: list[str] = []

    def verify(at, names):
        seen.append(fetch_hint("tiny"))
        return 0

    monkeypatch.setattr(cli, "cmd_verify", verify)
    assert cli.main(["--root", str(root), "verify"]) == 0
    assert seen == [
        f"In a new Terminal window, run: cd {root} && uv run clarity fetch tiny. Then try again."
    ]
    assert fetch_hint("tiny") == BEFORE  # cleared: a later run or test can't read it
