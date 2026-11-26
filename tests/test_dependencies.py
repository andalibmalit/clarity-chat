"""Dependencies whose version can change what Clarity sends or records (v0.3.1 D0)."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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


def test_inspect_ai_is_pinned_exactly():
    """Export for Inspect writes through inspect_ai's own writer, and Inspect promises its
    API, not its log format, so the extra pins one version with `==` (v0.4.3, decision 1): a
    bump is its own commit and reruns the gate's tests."""
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert len(extras.get("inspect", [])) == 1, extras
    assert re.fullmatch(r"inspect-ai==\d+(\.\d+)+", extras["inspect"][0]), extras["inspect"]


def test_dependabot_gives_inspect_ai_its_own_pull_request():
    """A bump of inspect-ai can change the log Clarity writes, so it comes alone."""
    config = (REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    excluded = re.search(r"exclude-patterns:\n((?:\s+- .+\n)+)", config)
    assert excluded and '- "inspect-ai"' in excluded.group(1), config


# The locked versions of the packages that can change what Clarity sends or extracts. Edit
# this table only in that dependency's own commit (CONTRIBUTING.md), so that adding an extra
# or re-resolving the lock can't move one of them silently: pyproject.toml pins tokenizers,
# transformers and jinja2 only with `>=` (v0.4.3 I0, the plan advisor's S5).
EXACTNESS = {
    "jinja2": "3.1.6",
    "pypdf": "7.1.0",
    "tokenizers": "0.23.3",
    "transformers": "5.22.2",
}


def test_the_exactness_packages_are_locked_at_their_table_versions():
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    locked = {}
    for package in lock["package"]:
        if package["name"] in EXACTNESS:
            locked.setdefault(package["name"], []).append(package["version"])
    assert locked == {name: [version] for name, version in EXACTNESS.items()}


def test_starting_clarity_does_not_import_inspect_ai():
    """inspect_ai is imported only inside the Inspect writer, so a missing extra costs nothing
    until an export for Inspect is asked for, and its import can't change a chat (v0.4.3, the
    plan's constraints; the advisor's C2). Run in a fresh interpreter, with the extra
    installed, so the check isn't true only because inspect_ai is missing."""
    pytest.importorskip("inspect_ai")
    code = (
        "import sys, clarity_chat.cli, clarity_chat.web\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'inspect_ai'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_nothing_clarity_runs_imports_a_dev_group_package():
    """The install and the launcher run uv with `--no-dev` (v0.4.6 S1), so a module of the dev
    group's that `src/` imports would fail only on an installed Mac, never in a checkout. The
    group's names are read from pyproject.toml; their modules from what's installed, keeping
    only the modules no other installed distribution also provides."""
    import ast
    from importlib.metadata import packages_distributions

    def norm(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    groups = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    dev = {norm(re.split(r"[^A-Za-z0-9_.-]", d)[0]) for d in groups["dependency-groups"]["dev"]}
    modules = {name.replace("-", "_") for name in dev}  # a module named as its package
    for module, dists in packages_distributions().items():
        if {norm(d) for d in dists} <= dev:
            modules.add(module)
    assert {"hypothesis", "pytest", "xdist", "gguf"} <= modules, modules  # read, not guessed
    found = []
    for path in sorted((REPO / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            found += [f"{path.relative_to(REPO)}: {n}" for n in names if n.split(".")[0] in modules]
    assert found == []
