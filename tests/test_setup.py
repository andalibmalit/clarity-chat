"""v0.4.2 A2: `clarity setup`'s checks, its one question and llama.cpp's source, offline.

The Mac's facts (platform, memory, free disk) and the home folder are parameters; every
command goes through FakeRunner, and git_state through a fake, so nothing here runs git,
touches the network or reads the checkout's vendor/llama.cpp. What the fakes imitate is
gated live by `-m setup` (tests/test_setup_live.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clarity_chat import setup as clarity_setup
from clarity_chat.profiles import ConfigError, load_server_config
from clarity_chat.setup import GIB, SIZES, Mac, Setup
from tests.fakes import FakeRunner

MAC = Mac(system="Darwin", machine="arm64", memory=32 * GIB, free=212 * 10**9)


class Says:
    """What setup said, and the answers it gets, in order."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.lines: list[str] = []
        self.asked: list[str] = []
        self.answers = list(answers or ["y"])

    def say(self, line: str) -> None:
        self.lines.append(line)

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.answers.pop(0)


def pin(root: Path) -> str:
    return load_server_config(root).llama_cpp_commit


def states(root: Path, **by_folder: dict[str, Any]):
    """A git_state fake: each folder's state by name (`partial`, `source`); a folder not
    named isn't a checkout. Records the folders asked about."""
    asked: list[Path] = []

    def state(folder: Path) -> dict[str, Any]:
        asked.append(folder)
        key = "partial" if folder.name.endswith(".partial") else "source"
        return by_folder.get(key, {"commit": None, "dirty": None, "error": "not a git repo"})

    state.asked = asked  # ty: ignore[unresolved-attribute]
    return state


def made(root: Path, mac: Mac = MAC, run=None, says=None, state=None, home=None) -> Setup:
    says = says or Says()
    return Setup(
        root,
        mac,
        run if run is not None else FakeRunner(),
        says.ask,
        says.say,
        state or states(root, partial={"commit": pin(root), "dirty": False}),
        home=home or root.parent,
    )


def binary(root: Path) -> Path:
    """Where the build puts llama-server, made so the build step finds it done."""
    path = load_server_config(root).llama_cpp_dir / "build" / "bin" / "llama-server"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"made up")
    return path


def source(root: Path) -> Path:
    folder = load_server_config(root).llama_cpp_dir
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# --- the Mac (states 1 and 2), before anything runs


@pytest.mark.parametrize(
    "system, machine", [("Linux", "x86_64"), ("Windows", "AMD64"), ("Darwin", "x86_64")]
)
def test_another_platform_is_refused_before_any_step(root, system, machine):
    run, says = FakeRunner(), Says()
    with pytest.raises(ConfigError) as refused:
        made(root, Mac(system, machine, 32 * GIB, MAC.free), run, says).go()
    assert str(refused.value) == (
        f"Setup needs a Mac with Apple silicon; this computer is {system} on {machine}. To "
        'install by hand, follow "Install by hand" in docs/getting-started.md.'
    )
    assert run.calls == [] and says.asked == []


def test_less_than_16_gb_of_memory_is_refused_and_16_passes(root):
    run, says = FakeRunner(), Says()
    with pytest.raises(ConfigError) as refused:
        made(root, Mac("Darwin", "arm64", 15 * GIB, MAC.free), run, says).go()
    assert str(refused.value) == "Clarity needs 16 GB of memory; this Mac has 15 GB."
    assert run.calls == [] and says.asked == []
    made(root, Mac("Darwin", "arm64", 16 * GIB, MAC.free)).go()


# --- the disk (state 7) and the one question (states 12 and 13)


def missing_bytes(*pieces: str) -> tuple[int, int]:
    return sum(SIZES[p][0] for p in pieces), sum(SIZES[p][1] for p in pieces)


def test_too_little_disk_is_refused_before_the_question(root):
    _, keep = missing_bytes("source", "build")
    free = keep + 5 * 10**9 - 1
    run, says = FakeRunner(), Says()
    with pytest.raises(ConfigError) as refused:
        made(root, Mac("Darwin", "arm64", 32 * GIB, free), run, says).go()
    need = f"{(keep + 5 * 10**9) / 1e9:.1f}"
    assert str(refused.value) == (
        f"Setup needs {need} GB free for what it still has to keep, with 5 GB to spare; this "
        f"disk has {free / 1e9:.1f} GB free."
    )
    assert run.calls == [] and says.asked == []


def question(download: int, keep: int, folder: str = "~/Clarity") -> str:
    return (
        f"Setup will download about {download / 1e9:.1f} GB and keep about {keep / 1e9:.1f} GB "
        f"in {folder} and your Hugging Face cache. Free: 212.0 GB. Continue? [Y/n] "
    )


def test_the_question_adds_up_what_is_missing(root):
    folder = f"~/{root.name}"  # the home folder is root's parent here
    says = Says()
    made(root, says=says).go()
    assert says.asked == [question(*missing_bytes("source", "build"), folder)]
    source(root)
    says = Says()
    made(root, says=says, state=states(root, source={"commit": pin(root), "dirty": False})).go()
    assert says.asked == [question(*missing_bytes("build"), folder)]


def test_nothing_missing_asks_nothing_and_says_each_step_is_done(root):
    source(root)
    binary(root)
    says, run = Says(answers=[]), FakeRunner()
    made(
        root, run=run, says=says, state=states(root, source={"commit": pin(root), "dirty": False})
    ).go()
    assert says.asked == [] and run.calls == []
    assert "llama.cpp's source: already done." in says.lines


def test_no_runs_nothing(root):
    run, says = FakeRunner(), Says(answers=["n"])
    made(root, run=run, says=says).go()
    assert run.calls == []
    assert says.lines[-1:] == ["Setup stopped. Nothing was downloaded."]
    assert not (load_server_config(root).llama_cpp_dir.parent / "llama.cpp.partial").exists()


# --- llama.cpp's source (states 8, 9 and 13)


def git_argvs(root: Path) -> list[list[str]]:
    partial = str(load_server_config(root).llama_cpp_dir.parent / "llama.cpp.partial")
    commit = pin(root)
    return [
        ["git", "-C", partial, "init", "-q"],
        ["git", "-C", partial, "fetch", "--depth", "1", "-q", clarity_setup.LLAMA_URL, commit],
        ["git", "-C", partial, "checkout", "--detach", "-q", commit],
    ]


def test_the_source_is_fetched_into_partial_and_renamed(root):
    run, says = FakeRunner(), Says()
    made(root, run=run, says=says).go()
    assert run.argvs == git_argvs(root)
    folder = load_server_config(root).llama_cpp_dir
    assert folder.is_dir() and not (folder.parent / "llama.cpp.partial").exists()
    assert f"Fetching llama.cpp's source at {pin(root)[:12]} …" in says.lines


def test_git_runs_with_a_cleared_environment(root, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/made/up/gitconfig")
    run = FakeRunner()
    made(root, run=run, home=Path("/Users/made-up")).go()
    assert run.calls, "git never ran"
    for _, env, _ in run.calls:
        assert set(env) == {
            "PATH", "HOME", "LANG", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL",
            "GIT_TERMINAL_PROMPT",
        }  # fmt: skip
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null" and env["HOME"] == "/Users/made-up"


def test_a_failed_fetch_leaves_partial_and_the_next_run_starts_again(root):
    run, says = FakeRunner([(0, ""), (128, "fatal: unable to access 'made up'")]), Says()
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, says=says).go()
    assert str(refused.value) == (
        "Fetching llama.cpp's source failed: fatal: unable to access 'made up'. Run setup "
        "again: it starts the fetch over."
    )
    partial = load_server_config(root).llama_cpp_dir.parent / "llama.cpp.partial"
    assert partial.is_dir() and not load_server_config(root).llama_cpp_dir.exists()
    run, says = FakeRunner(), Says()
    made(root, run=run, says=says).go()
    assert "Deleting vendor/llama.cpp.partial, left by a setup that didn't finish." in says.lines
    assert run.argvs == git_argvs(root)


def test_a_partial_that_isn_t_at_the_pin_and_clean_isn_t_renamed(root):
    state = states(root, partial={"commit": pin(root), "dirty": True})
    with pytest.raises(ConfigError) as refused:
        made(root, state=state).go()
    assert "didn't check out" in str(refused.value)
    assert not load_server_config(root).llama_cpp_dir.exists()


@pytest.mark.parametrize(
    "state, why",
    [
        ({"commit": "0" * 40, "dirty": False}, "is at commit 000000000000, not the pinned "),
        ({"commit": "PIN", "dirty": True}, "has uncommitted changes"),
        (None, "isn't a git checkout"),
    ],
    ids=["another-commit", "dirty", "not-a-checkout"],
)
def test_a_source_setup_didn_t_make_is_refused_untouched(root, state, why):
    folder = source(root)
    (folder / "mine.txt").write_text("made up", encoding="utf-8")
    if state is not None and state["commit"] == "PIN":
        state = {**state, "commit": pin(root)}
    run, says = FakeRunner(), Says()
    fake = states(root, source=state) if state is not None else states(root)
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, says=says, state=fake).go()
    assert str(refused.value).startswith(f"vendor/llama.cpp {why}")
    assert str(refused.value).endswith(
        "so setup leaves it alone: it may be your own work. Move it out of vendor/, then run "
        "setup again."
    )
    assert run.calls == [] and says.asked == []
    assert sorted(p.name for p in folder.iterdir()) == ["mine.txt"]


# --- the lock (state 22)


def test_setup_while_clarity_runs_is_refused(root, capsys, monkeypatch):
    from clarity_chat import cli, instance

    monkeypatch.setattr(clarity_setup, "this_mac", lambda r: MAC)
    with instance.hold(root, "ui"):
        assert cli.main(["--root", str(root), "setup"]) == 1
    assert "Clarity is already running in this project" in capsys.readouterr().err
