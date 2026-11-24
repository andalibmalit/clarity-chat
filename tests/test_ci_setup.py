"""v0.4.2 E1: CI's macOS job for setup (.github/workflows/setup.yml), read as text, as
test_ci_browsers reads tests.yml. It runs only when what setup depends on changes, on an
Apple silicon runner: `clarity setup`'s fetch and build (-m setup), never a model or torch."""

from __future__ import annotations

from tests.conftest import REPO

WORKFLOW = REPO / ".github" / "workflows" / "setup.yml"
WATCHED = (
    "src/clarity_chat/setup.py",
    # What the setup tier runs of other modules: git_state (the checkout's state) and
    # server_binary (the build's llama-server), the whole-branch review's C12.
    "src/clarity_chat/logs.py",
    "src/clarity_chat/backends/llama_cpp.py",
    "install.sh",
    "server.toml",
    "pyproject.toml",
    "uv.lock",
    "tests/test_setup_live.py",
    "tests/test_install.py",
    ".github/workflows/setup.yml",
)


def text() -> str:
    assert WORKFLOW.is_file(), "no .github/workflows/setup.yml"
    return WORKFLOW.read_text(encoding="utf-8")


def test_it_runs_when_what_setup_depends_on_changes():
    workflow = text()
    on = workflow.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    for trigger in ("push", "pull_request"):
        block = on.split(f"  {trigger}:\n", 1)[1].split("\n  pull_request:", 1)[0]
        for path in WATCHED:
            assert f"      - {path}\n" in block + "\n", (trigger, path)


def test_it_fetches_and_builds_on_apple_silicon_with_the_setup_group():
    workflow = text()
    assert "runs-on: macos-latest" in workflow  # arm64 since macos-14
    assert "timeout-minutes: 30" in workflow
    assert "uv sync --locked --group setup" in workflow
    assert "uv run --group setup pytest -m setup -v -s -rs --junitxml=setup.xml" in workflow  # E1b
    # A skip or an empty run fails the job, as live-cpu's guard does.
    assert "uv run python scripts/junit_guard.py setup.xml --at-least 1" in workflow
    assert "--extra hf" not in workflow  # no torch: the tier fetches and builds only


def test_it_runs_the_install_command_s_tests_on_the_mac():
    """A6: install.sh's tests run on every offline job, and here too, on the platform it's
    for, whenever install.sh or what it runs changes."""
    assert "uv run pytest tests/test_install.py -v" in text()


def test_its_actions_are_pinned_as_tests_yml_pins_them():
    workflow = text()
    tests = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    for action in ("actions/checkout@", "astral-sh/setup-uv@"):
        line = next(row for row in workflow.splitlines() if action in row).strip()
        assert line.removeprefix("- ") in tests, line
    assert "permissions:\n  contents: read" in workflow
