"""The docs name everything a user can set or type, and their links lead somewhere."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from clarity_chat.cli import HELP_TAIL, main
from clarity_chat.commands import COMMANDS
from clarity_chat.profiles import (
    PROFILE_KEYS_CHAT,
    PROFILE_KEYS_COMMON,
    PROFILE_KEYS_TEXT,
    SETTING_NAMES,
)
from tests.conftest import REPO

REFERENCE = (REPO / "docs" / "reference.md").read_text(encoding="utf-8")
DOCS = [REPO / "README.md", REPO / "CONTRIBUTING.md", *sorted((REPO / "docs").glob("*.md"))]


def cli_commands(capsys: pytest.CaptureFixture[str]) -> list[str]:
    with pytest.raises(SystemExit):
        main(["--help"])
    usage = capsys.readouterr().out
    match = re.search(r"\{([a-z,]+)\}", usage)
    assert match, usage
    return match.group(1).split(",")


def test_reference_names_every_cli_command(capsys):
    commands = cli_commands(capsys)
    assert "chat" in commands
    missing = [c for c in commands if f"`clarity {c}" not in REFERENCE]
    assert not missing, f"docs/reference.md doesn't describe: {missing}"


def test_reference_names_every_chat_command():
    extra = re.findall(r"^  (/{1,2}\w+)", HELP_TAIL, flags=re.MULTILINE)
    commands = sorted({f"/{name}" for name in COMMANDS} | set(extra))
    assert "/prompt" in commands and "//text" in commands
    missing = [c for c in commands if f"`{c}" not in REFERENCE]
    assert not missing, f"docs/reference.md doesn't describe: {missing}"


def test_reference_names_every_profile_key_and_setting():
    names = PROFILE_KEYS_COMMON | PROFILE_KEYS_CHAT | PROFILE_KEYS_TEXT | set(SETTING_NAMES)
    missing = sorted(n for n in names if f"`{n}`" not in REFERENCE)
    assert not missing, f"docs/reference.md doesn't describe: {missing}"


def test_the_llama_cpp_commit_in_the_docs_is_the_pinned_one():
    pinned = tomllib.loads((REPO / "server.toml").read_text(encoding="utf-8"))["llama_cpp_commit"]
    for path in (REPO / "README.md", REPO / "docs" / "getting-started.md"):
        commits = re.findall(r"checkout ([0-9a-f]{40})", path.read_text(encoding="utf-8"))
        assert commits, f"{path.name} names no llama.cpp commit"
        assert set(commits) == {pinned}, f"{path.name}: {commits} != server.toml's {pinned}"


def outside_code(text: str) -> str:
    """The text with fenced code blocks removed, so `# comments` in them aren't headings."""
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def anchors(path: Path) -> set[str]:
    """GitHub's heading anchors: lower case, punctuation dropped, spaces as hyphens."""
    headings = re.findall(r"^#+ (.+)$", outside_code(path.read_text(encoding="utf-8")), re.M)
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in headings}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_lead_to_a_file_and_heading(doc):
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", outside_code(doc.read_text(encoding="utf-8"))):
        if re.match(r"[a-z]+:", target):
            continue  # a web link
        file_part, _, anchor = target.partition("#")
        path = (doc.parent / file_part).resolve() if file_part else doc
        if not path.exists():
            broken.append(target)
        elif anchor and path.suffix == ".md" and anchor not in anchors(path):
            broken.append(target)
    assert not broken, f"{doc.name} has broken links: {broken}"
