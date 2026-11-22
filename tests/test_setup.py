"""v0.4.2 A2: `clarity setup`'s checks, its one question and llama.cpp's source, offline.

The Mac's facts (platform, memory, free disk) and the home folder are parameters; every
command goes through FakeRunner, and git_state through a fake, so nothing here runs git,
touches the network or reads the checkout's vendor/llama.cpp. What the fakes imitate is
gated live by `-m setup` (tests/test_setup_live.py).
"""

from __future__ import annotations

import os
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


def made(
    root: Path,
    mac: Mac = MAC,
    run=None,
    says=None,
    state=None,
    home=None,
    models=(),
    fetch=None,
    desktop=None,
    uv="/Users/made-up/.local/bin/uv",
) -> Setup:
    """A Setup on fakes. `models`: the models it sets up (none by default: A2's tests);
    `fetch`: a fake fetch (a FakeFetch)."""
    says = says or Says()
    return Setup(
        root,
        mac,
        run if run is not None else FakeRunner(),
        says.ask,
        says.say,
        state or states(root, partial={"commit": pin(root), "dirty": False}),
        home=home or root.parent,
        models=models,
        fetch=fetch or FakeFetch(),
        desktop=desktop or desktop_of(root),
        uv=uv,
    )


def desktop_of(root: Path) -> Path:
    """A Desktop of the test's own (root's parent is shared by a session's tests)."""
    desktop = root / "Desktop"
    desktop.mkdir(exist_ok=True)
    return desktop


class FakeFetch:
    """models.fetch offline: records each name, and leaves what a real fetch would (a
    weights file and its lock entry, conftest's fake_fetch), or raises `fails[name]`."""

    def __init__(self, fails: dict[str, Exception] | None = None) -> None:
        self.names: list[str] = []
        self.fails = fails or {}

    def __call__(self, root: Path, model: Any, server: Any) -> dict[str, Any]:
        from tests.conftest import fake_fetch

        self.names.append(model.name)
        if model.name in self.fails:
            raise self.fails[model.name]
        fake_fetch(root, model.name)
        return {}


def binary(root: Path) -> Path:
    """Where the build puts llama-server, made so the build step finds it done."""
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"  # as server_binary finds it
    path = load_server_config(root).llama_cpp_dir / "build" / "bin" / exe
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
        # HOME as setup builds it, from a Path: backslashes on Windows
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull and env["HOME"] == str(Path("/Users/made-up"))


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


# --- A3: the models and the end (states 13, 14 and 16)

TINY2 = """
[tiny2]
route = "gguf"
repo = "test/tiny2-gguf"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "tiny2.gguf"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"  # fake_fetch's
tokenizer_repo = "test/tiny2"
tokenizer_revision = "0123456789abcdef0123456789abcdef01234567"
"""


@pytest.fixture
def two_models(root, monkeypatch):
    """The made-up `tiny` (convert route) and `tiny2` (gguf route), with sizes, and
    llama.cpp's source and build already there."""
    with (root / "models.toml").open("a", encoding="utf-8") as f:
        f.write(TINY2)
    monkeypatch.setitem(SIZES, "tiny", (3 * 10**9, 4 * 10**9))
    monkeypatch.setitem(SIZES, "tiny2", (2 * 10**9, 2 * 10**9))
    source(root)
    binary(root)
    return root, states(root, source={"commit": pin(root), "dirty": False})


def test_each_missing_model_is_fetched_in_order_after_one_question(two_models):
    root, state = two_models
    says, fetch = Says(), FakeFetch()
    made(root, says=says, state=state, models=("tiny", "tiny2"), fetch=fetch).go()
    download, keep = missing_bytes("tiny", "tiny2", "converter")
    assert says.asked == [question(download, keep, f"~/{root.name}")]
    assert fetch.names == ["tiny", "tiny2"]


def test_a_fetched_model_is_already_done_and_nothing_is_asked(two_models):
    from tests.conftest import fake_fetch

    root, state = two_models
    fake_fetch(root, "tiny")
    fake_fetch(root, "tiny2")
    says, fetch = Says(answers=[]), FakeFetch()
    made(root, says=says, state=state, models=("tiny", "tiny2"), fetch=fetch).go()
    assert fetch.names == [] and says.asked == []
    assert "tiny: already done." in says.lines and "tiny2: already done." in says.lines


def test_the_converter_counts_only_while_a_convert_model_is_missing(two_models):
    from tests.conftest import fake_fetch

    root, state = two_models
    fake_fetch(root, "tiny")  # the convert route's model is there
    says = Says()
    made(root, says=says, state=state, models=("tiny", "tiny2"), fetch=FakeFetch()).go()
    assert says.asked == [question(*missing_bytes("tiny2"), f"~/{root.name}")]


def test_a_failed_fetch_stops_setup_and_the_next_run_fetches_only_what_s_missing(two_models):
    root, state = two_models
    failing = FakeFetch({"tiny2": ConfigError("A made-up refusal.")})
    with pytest.raises(ConfigError, match="A made-up refusal."):
        made(root, state=state, models=("tiny", "tiny2"), fetch=failing).go()
    assert failing.names == ["tiny", "tiny2"]
    again = FakeFetch()
    made(root, state=state, models=("tiny", "tiny2"), fetch=again).go()
    assert again.names == ["tiny2"]


def test_the_end_says_how_to_start_next_time(two_models):
    root, state = two_models
    says = Says()
    made(root, says=says, state=state, models=("tiny",), fetch=FakeFetch()).go()
    assert says.lines[-1] == (
        "Setup is done. Next time, double-click Clarity.command on your Desktop to start "
        "Clarity, or run the same command again to check the setup."
    )


def test_the_sizes_table_covers_setup_s_models_as_models_toml_has_them():
    from clarity_chat.profiles import load_models
    from clarity_chat.setup import SETUP_MODELS
    from tests.conftest import REPO

    shipped = load_models(REPO)
    assert SETUP_MODELS and set(SETUP_MODELS) <= set(shipped)
    for name in SETUP_MODELS:
        download, keep = SIZES[name]
        assert 0 < download <= keep, name  # a model keeps at least what it downloads
    if any(shipped[name].route == "convert" for name in SETUP_MODELS):
        assert SIZES["converter"][0] > 0


# --- A7: the launcher (state 15; decision 13)


def done_root(root: Path) -> tuple[Path, Any]:
    """Everything before the launcher already done."""
    source(root)
    binary(root)
    return root, states(root, source={"commit": pin(root), "dirty": False})


def launcher_lines(folder: Path, uv: str = "/Users/made-up/.local/bin/uv") -> str:
    import shlex

    return f"cd {shlex.quote(str(folder))}\nexec {shlex.quote(uv)} run clarity ui\n"


def test_the_launcher_is_written_with_two_lines_and_mode_0755(tmp_path, root):
    import shlex

    folder = tmp_path / "Its folder, with a space and a 'quote'"
    folder.mkdir()
    for name in ("defaults.toml", "server.toml", "models.toml"):
        (folder / name).write_bytes((root / name).read_bytes())
    _, state = done_root(folder)
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    says = Says()
    made(folder, says=says, state=state, desktop=desktop, uv="/Users/a b/.local/bin/uv").go()
    launcher = desktop / "Clarity.command"
    assert launcher.is_file(), "no launcher on the Desktop"
    text = launcher.read_text(encoding="utf-8")
    assert text == launcher_lines(folder, "/Users/a b/.local/bin/uv")
    assert shlex.split(text.splitlines()[0]) == ["cd", str(folder)]
    assert shlex.split(text.splitlines()[1]) == ["exec", "/Users/a b/.local/bin/uv", "run",
                                                 "clarity", "ui"]  # fmt: skip
    # Windows keeps only a read-only flag (chmod there can't mark a file executable), so
    # its 0755 reads back as 0666. Setup runs on macOS; the file's content is checked above.
    assert launcher.stat().st_mode & 0o777 == (0o666 if os.name == "nt" else 0o755)
    assert "Made Clarity.command on your Desktop: double-click it to start Clarity." in says.lines


def test_the_same_launcher_is_left_and_already_done(root):
    root, state = done_root(root)
    launcher = desktop_of(root) / "Clarity.command"
    launcher.write_text(launcher_lines(root), encoding="utf-8")
    launcher.chmod(0o755)
    before = launcher.stat().st_mtime_ns
    says = Says()
    made(root, says=says, state=state).go()
    assert launcher.stat().st_mtime_ns == before
    assert "Clarity.command on your Desktop: already done." in says.lines


def test_another_file_of_that_name_is_left_alone_and_the_lines_printed(root):
    root, state = done_root(root)
    launcher = desktop_of(root) / "Clarity.command"
    launcher.write_text("echo mine\n", encoding="utf-8")
    says = Says()
    made(root, says=says, state=state).go()
    assert launcher.read_text(encoding="utf-8") == "echo mine\n"
    said = "\n".join(says.lines)
    assert (
        "Your Desktop already has a Clarity.command that setup didn't write, so it's left "
        "alone. To start Clarity from a file of your own, put these two lines in it:" in said
    )
    assert launcher_lines(root).rstrip("\n") in said


def test_a_desktop_macos_won_t_let_setup_write_is_said_and_setup_still_ends(root, monkeypatch):
    root, state = done_root(root)
    desktop = desktop_of(root)

    def refused(self, *args, **kwargs):
        raise PermissionError(1, "Operation not permitted", str(self))

    monkeypatch.setattr(Path, "write_text", refused)
    says = Says()
    made(root, says=says, state=state).go()  # no exception: setup ends with 0
    said = "\n".join(says.lines)
    assert (
        "Setup couldn't write Clarity.command to your Desktop (Operation not permitted). To "
        "make the launcher yourself, put these two lines in a file named Clarity.command "
        "there:" in said
    )
    assert launcher_lines(root).rstrip("\n") in said
    assert not (desktop / "Clarity.command").exists()
