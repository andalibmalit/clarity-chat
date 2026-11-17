"""The QA driver's path guard against Clarity's own parse (the v0.3.2 QA-driver review's
should 1): Clarity's real `/documents` and `/attach` run each step the guard allows, with the
file read replaced by a recorder, and every path Clarity would open must be inside the
scratch root. Run with `os.path.exists` true and false, since `/documents add` picks its
reading by what exists. A test of strings the author thought of once let a
flagged path through (84d0cda's driver, when /documents add took `--replace`)."""

from __future__ import annotations

import importlib.util
import itertools
import os
from pathlib import Path

import pytest

from clarity_chat import documents
from clarity_chat.cli import split_command
from clarity_chat.commands import Commands
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

HEADS = ["/documents add", "/attach"]
NAMES = ["", " made-up-name", "  made-up-name"]
SEPARATORS = [" ", "  "]


def paths(root: Path) -> list[str]:
    return [
        "docs/notes.md",  # inside
        str(root.parent / "secret.txt"),  # outside, absolute
        "~/made-up-qa-file.md",
        "../secret.txt",
        "docs/away.md",  # a link inside the root to a file outside
        "docs/notes.md/../notes.md",
    ]


@pytest.fixture
def setup(fetched_root, chat_tokenizer, monkeypatch):
    (fetched_root / "docs").mkdir()
    (fetched_root / "docs" / "notes.md").write_bytes(b"made-up notes\n")
    outside = fetched_root.parent / "secret.txt"
    outside.write_bytes(b"not for the transcript\n")
    (fetched_root / "docs" / "away.md").symlink_to(outside)
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    c = Commands(session)
    c.run("profile", "chat", lambda e: None)
    monkeypatch.chdir(fetched_root)  # the driver runs Clarity in the scratch root
    return fetched_root, c


def opened(c: Commands, line: str, exists: bool, monkeypatch) -> list[str]:
    """The paths Clarity's own command would read for `line`."""
    read: list[str] = []

    def record(path, *args, **kwargs):
        read.append(path)
        raise documents.DocumentRefused("recorded")

    name, args = split_command(line)
    with monkeypatch.context() as m:
        m.setattr(documents, "read_path", record)
        m.setattr(os.path, "exists", lambda p: exists)
        try:
            c.run(name, args, lambda e: None)
        except (ConfigError, documents.DocumentRefused):
            pass
    return read


def test_every_step_the_guard_allows_reads_only_inside_the_root(setup, monkeypatch):
    root, c = setup
    base = root.resolve()
    allowed = leaks = 0
    for head, path, name, sep in itertools.product(HEADS, paths(root), NAMES, SEPARATORS):
        if head == "/attach" and name:
            continue  # /attach takes no name
        line = f"{head}{sep}{path}{name}"
        try:
            qa.check_step("t", line, False, root)
        except qa.Refused:
            continue
        allowed += 1
        for exists in (True, False):
            for p in opened(c, line, exists, monkeypatch):
                resolved = Path(os.path.expanduser(p))
                resolved = (resolved if resolved.is_absolute() else root / resolved).resolve()
                if not resolved.is_relative_to(base):
                    leaks += 1
                    print("LEAK", repr(line), exists, p)
    assert allowed, "the table allowed nothing: it tests nothing"
    assert leaks == 0
