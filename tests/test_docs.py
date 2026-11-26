"""The docs name everything a user can set or type, and their links lead somewhere."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from clarity_chat.cli import HELP_TAIL, main
from clarity_chat.commands import COMMANDS
from clarity_chat.profiles import (
    CHAT_KEYS,
    PROFILE_KEYS_CHAT,
    PROFILE_KEYS_COMMON,
    PROFILE_KEYS_TEXT,
    SETTING_HELP,
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


def settings_table() -> dict[str, dict[str, str]]:
    """docs/reference.md's Settings table, by setting: each column's text."""
    section = REFERENCE.split("\n## Settings\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    table = {}
    for line in rows[1:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        table[cells[0].strip("`")] = dict(zip(header, cells, strict=True))
    return table


def test_every_setting_has_one_line_of_help_and_the_docs_show_the_same():
    """AGENTS.md: each explanation lives in one place, the settings table, which the UI and
    the docs reuse. profiles.SETTING_HELP is that place; the docs' column repeats it."""
    names = {*SETTING_NAMES, *CHAT_KEYS, "ctx_size"}  # v0.4.4: what /set takes (B1, B3b)
    assert names <= set(SETTING_HELP), "a setting has no help"
    table = settings_table()
    assert set(table) == names
    for name, help_text in SETTING_HELP.items():
        assert table[name].get("What it does") == help_text, name
    # S8: the profile's [documents] row points to the table, which holds their explanation.
    (documents,) = [
        line for line in REFERENCE.splitlines() if line.startswith("| `documents` | A table")
    ]
    assert "explained in [Settings](#settings)" in documents


def written_checks() -> set[str]:
    """Every check name the code writes into a call record's `checks`: `checks["NAME"]`,
    `checks = {"NAME": …}`, or a `"NAME": {"ok": …}` entry, in src/clarity_chat."""
    src = REPO / "src" / "clarity_chat"
    found: set[str] = set()
    for path in [*src.glob("*.py"), *(src / "backends").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        found |= set(re.findall(r'checks\["(\w+)"\]', text))
        found |= set(re.findall(r'checks = \{"(\w+)"', text))
        found |= set(re.findall(r'"(\w+)": \{"ok"', text))
    return found


# Every check the code writes, and server_props, which logs before v0.2.1 have.
CHECK_NAMES = written_checks() | {"server_props"}


def test_the_check_names_are_found_in_the_code():
    """The scan behind CHECK_NAMES finds each backend's checks (the tripwire review after
    C5: derive them rather than list them by hand)."""
    assert {"prompt_echo", "settings_echo", "server_running", "generation_config"} <= (
        written_checks()
    )


def checks_table() -> dict[str, dict[str, str]]:
    """docs/reference.md's table of checks after each call, by check: each column's text."""
    found = re.search(r"\nChecks after each call:\n\n((?:\|.*\n)+)", REFERENCE)
    if not found:
        return {}
    rows = [line for line in found.group(1).splitlines() if not line.startswith("|---")]
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    table = {}
    for line in rows[1:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        table[cells[0].strip("`")] = dict(zip(header, cells, strict=True))
    return table


def test_every_check_has_its_lines_and_the_docs_show_the_same():
    """Checkpoint 4, finding 9: what a check checks and what its failure means live in one
    place, backends.CHECK_HELP, which Details shows; the docs' table repeats it."""
    from clarity_chat import backends

    check_help = getattr(backends, "CHECK_HELP", {})
    assert set(check_help) == CHECK_NAMES
    table = checks_table()
    assert set(table) == CHECK_NAMES
    for name, (checks, fails) in check_help.items():
        assert table[name]["What it checks"] == checks, name
        assert table[name]["A failure means"] == fails, name


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
