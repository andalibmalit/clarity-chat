"""Dependencies whose version can change what Clarity sends or records (v0.3.1 D0)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_pypdf_is_pinned_exactly():
    """Its version changes the text extracted from a PDF, so it's pinned with `==` (v0.3.1,
    decision 12): a bump is an exactness commit, never a range quietly moving."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pins = [d for d in project["dependencies"] if re.match(r"pypdf\b", d)]
    assert len(pins) == 1 and re.fullmatch(r"pypdf==\d+(\.\d+)+", pins[0]), pins


def test_dependabot_gives_pypdf_its_own_pull_request():
    """Like tokenizers, transformers and jinja2, pypdf is left out of the monthly group, so a
    bump comes alone, to be checked as an exactness change."""
    config = (REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    excluded = re.search(r"exclude-patterns:\n((?:\s+- .+\n)+)", config)
    assert excluded and '- "pypdf"' in excluded.group(1), config
