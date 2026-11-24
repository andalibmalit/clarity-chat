"""v0.4.2 A2: `clarity setup`'s checks, its one question and llama.cpp's source, offline.

The Mac's facts (platform, memory, free disk) and the home folder are parameters; every
command goes through FakeRunner, and git_state through a fake, so nothing here runs git,
touches the network or reads the checkout's vendor/llama.cpp. What the fakes imitate is
gated live by `-m setup` (tests/test_setup_live.py).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from clarity_chat import setup as clarity_setup
from clarity_chat.profiles import ConfigError, load_server_config
from clarity_chat.setup import GIB, SIZES, Mac, Setup
from tests.fakes import FakeRunner

MAC = Mac(system="Darwin", machine="arm64", memory=32 * GIB, free=212 * 10**9)


@pytest.fixture(autouse=True)
def made_up_cmake(monkeypatch, tmp_path_factory):
    """Every test here stands in a made-up cmake for the setup group's, so `clarity setup`
    through cli.main gets past its cmake check whatever this environment synced: CI syncs
    without the setup group, and a local .venv may keep its cmake from an earlier sync. The
    test of the refusal sets its own missing path."""
    cmake = tmp_path_factory.mktemp("venv") / "cmake"
    cmake.write_text("made up", encoding="utf-8")
    monkeypatch.setattr(clarity_setup, "venv_cmake", lambda: cmake)


CMAKE = "/Users/made-up/Clarity/.venv/bin/cmake"  # the venv's own, by path


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
    envs: list[dict[str, str] | None] = []  # the environment each git_state ran with (C7)

    def state(folder: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
        asked.append(folder)
        envs.append(env)
        key = "partial" if folder.name.endswith(".partial") else "source"
        return by_folder.get(key, {"commit": None, "dirty": None, "error": "not a git repo"})

    state.asked = asked  # ty: ignore[unresolved-attribute]
    state.envs = envs  # ty: ignore[unresolved-attribute]
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
    cmake=CMAKE,
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
        cmake=cmake,
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


def source(root: Path, checkout: bool = True) -> Path:
    """vendor/llama.cpp, with a .git of its own (a checkout, whose state the fake gives)
    unless `checkout` is False (a plain folder)."""
    folder = load_server_config(root).llama_cpp_dir
    folder.mkdir(parents=True, exist_ok=True)
    if checkout:
        (folder / ".git").mkdir(exist_ok=True)
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
        f"in {folder} and your Hugging Face and uv caches. Free: 212.0 GB. Continue? [Y/n] "
    )


def test_the_question_adds_up_what_is_missing(root):
    folder = f"~/{root.name}"  # the home folder is root's parent here
    says = Says()
    made(root, says=says).go()
    assert says.asked == [question(*missing_bytes("source", "build"), folder)]
    shutil.rmtree(source(root) / "build")  # the source there, llama-server not built
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
    assert says.asked == [] and run.of("git") == [] and run.of("cmake") == []
    assert "llama.cpp's source: already done." in says.lines


@pytest.mark.parametrize("no", ["n", "no", "N"])
def test_no_runs_nothing(root, no):
    run, says = FakeRunner(), Says(answers=[no])
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
    assert run.of("git") == git_argvs(root)
    folder = load_server_config(root).llama_cpp_dir
    assert folder.is_dir() and not (folder.parent / "llama.cpp.partial").exists()
    assert f"Fetching llama.cpp's source at {pin(root)[:12]} …" in says.lines


def test_git_runs_with_a_cleared_environment(root, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/made/up/gitconfig")
    run = FakeRunner()
    made(root, run=run, home=Path("/Users/made-up")).go()
    assert run.calls, "git never ran"
    for argv, env in run.calls:
        if argv[0] != "git":
            continue
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
    assert run.of("git") == git_argvs(root)


@pytest.mark.parametrize("dirty", [True, None], ids=["dirty", "status-failed"])
def test_a_partial_that_isn_t_at_the_pin_and_clean_isn_t_renamed(root, dirty):
    """At the pin, with changes or with a git status that failed (dirty None, the
    whole-branch review's S7): neither is known clean, so neither is renamed."""
    state = states(root, partial={"commit": pin(root), "dirty": dirty})
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
    folder = source(root, checkout=state is not None)
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
    assert sorted(p.name for p in folder.iterdir()) == (
        [".git", "mine.txt"] if state is not None else ["mine.txt"]
    )


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

    # Checkpoint 1's C6: a shebang first, so the file runs under sh whatever the login shell
    return f"#!/bin/sh\ncd {shlex.quote(str(folder))}\nexec {shlex.quote(uv)} run clarity ui\n"


def test_the_launcher_is_written_with_three_lines_and_mode_0755(tmp_path, root):
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
    assert text.splitlines()[0] == "#!/bin/sh"
    assert shlex.split(text.splitlines()[1]) == ["cd", str(folder)]
    assert shlex.split(text.splitlines()[2]) == ["exec", "/Users/a b/.local/bin/uv", "run",
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
        "alone. To start Clarity from a file of your own, put these three lines in it:" in said
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
        "make the launcher yourself, put these three lines in a file named Clarity.command "
        "there:" in said
    )
    assert launcher_lines(root).rstrip("\n") in said
    assert not (desktop / "Clarity.command").exists()


# --- Checkpoint 1's fix batch


def test_home_desktop_and_uv_are_required():
    """M1: nothing defaults to the real home folder or Desktop."""
    import inspect

    params = inspect.signature(Setup).parameters
    defaulted = [n for n in ("home", "desktop", "uv") if params[n].default is not params[n].empty]
    assert defaulted == []


def cli_setup(root: Path, monkeypatch, stdin: str) -> int:
    """`clarity setup` through cli.main, on this Mac's stand-in, with `stdin` typed."""
    import io

    from clarity_chat import cli

    monkeypatch.setattr(clarity_setup, "this_mac", lambda r: MAC)
    monkeypatch.setattr(clarity_setup, "SETUP_MODELS", ())
    monkeypatch.setenv("UV", "/Users/made-up/.local/bin/uv")
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    try:
        return cli.main(["--root", str(root), "setup"])
    except EOFError:  # input() at the question met the end of stdin
        return -1


@pytest.mark.parametrize("stdin", ["", "no\n", "n\n"], ids=["eof", "no", "n"])
def test_eof_or_no_at_the_question_stops_with_nothing_downloaded(root, monkeypatch, capsys, stdin):
    """M3: under `curl | sh` without a terminal, input() meets EOF: that's "no", not a
    traceback."""
    assert cli_setup(root, monkeypatch, stdin) == 0
    out = capsys.readouterr().out
    assert "Setup stopped. Nothing was downloaded." in out
    assert not load_server_config(root).llama_cpp_dir.parent.exists()


def git(folder: Path, *args: str) -> None:
    import subprocess

    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(
        ["git", "-C", str(folder), "-c", "user.name=made-up", "-c", "user.email=m@example.com",
         *args], check=True, capture_output=True, env=env,
    )  # fmt: skip


def test_a_plain_folder_inside_a_git_clone_isn_t_a_checkout(root):
    """S1: git_state in vendor/llama.cpp finds the clone around it (Clarity's own), so a
    plain folder there read as a checkout at the clone's commit. Real git, in tmp_path."""
    from clarity_chat.logs import git_state

    git(root, "init", "-q")
    git(root, "add", "defaults.toml")
    git(root, "commit", "-q", "-m", "made up")
    folder = source(root, checkout=False)
    (folder / "mine.txt").write_text("made up", encoding="utf-8")
    run = FakeRunner()
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, state=git_state).go()
    assert str(refused.value).startswith("vendor/llama.cpp isn't a git checkout")
    assert run.calls == []


def test_git_state_reads_the_folder_with_the_env_it_s_given(root, tmp_path, monkeypatch):
    """The whole-branch review's C2: setup passes git_state its cleared env, and git runs in
    that env, not this process's. Here GIT_DIR points at another repository: git_state
    with setup's env still returns the folder's own commit, and without it, the other's.
    Real git, in tmp_path; guards, green now."""
    from clarity_chat.logs import git_state

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for folder, name in ((root, "defaults.toml"), (elsewhere, "made-up.txt")):
        (folder / name).touch()
        git(folder, "init", "-q")
        git(folder, "add", name)
        git(folder, "commit", "-q", "-m", "made up")
    own, other = (git_state(folder)["commit"] for folder in (root, elsewhere))
    assert own != other
    monkeypatch.setenv("GIT_DIR", str(elsewhere / ".git"))
    assert git_state(root)["commit"] == other  # the inherited env reads the other repository
    assert git_state(root, made(root).git_env())["commit"] == own


def test_nothing_missing_needs_no_free_disk(root):
    """S4: the disk check is for what's still to keep; with nothing missing it isn't made."""
    source(root)
    binary(root)
    state = states(root, source={"commit": pin(root), "dirty": False})
    try:
        made(root, Mac("Darwin", "arm64", 32 * GIB, 1), state=state, says=Says(answers=[])).go()
    except ConfigError as e:
        pytest.fail(f"refused with nothing to keep: {e}")


def test_the_build_lines_are_getting_started_s():
    from tests.conftest import REPO

    doc = (REPO / "docs" / "getting-started.md").read_text(encoding="utf-8")
    for line in clarity_setup.BUILD_LINES:
        assert line.replace("-j N", "-j 8") in doc, line


def test_setup_refuses_without_uv(root, monkeypatch, capsys):
    """S6: the launcher needs uv's path; without it, setup says so before anything."""
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(clarity_setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(clarity_setup, "this_mac", lambda r: MAC)
    monkeypatch.setattr(clarity_setup, "SETUP_MODELS", ())
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("n\n"))
    from clarity_chat import cli

    assert cli.main(["--root", str(root), "setup"]) == 1
    assert (
        "clarity: Setup can't find uv, which starts Clarity. Install it as Getting started "
        "says, then run setup again." in capsys.readouterr().err
    )


def test_uv_s_path_keeps_its_link(root, monkeypatch, tmp_path):
    """S6: absolute(), not resolve(): ~/.local/bin/uv is often a link to a versioned file,
    which an update replaces; the launcher keeps the link."""
    from clarity_chat import cli

    target = tmp_path / "uv-0.9.0"
    target.write_text("made up", encoding="utf-8")
    link = tmp_path / "bin" / "uv"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks need privileges on Windows")
    seen: dict[str, Any] = {}

    class Recorded:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

        def go(self) -> None:
            pass

    monkeypatch.setenv("UV", str(link))
    monkeypatch.setattr(clarity_setup, "Setup", Recorded)
    monkeypatch.setattr(clarity_setup, "this_mac", lambda r: MAC)
    assert cli.main(["--root", str(root), "setup"]) == 0
    assert seen["uv"] == str(link)


def test_a_link_in_vendor_s_place_is_refused_naming_where_it_points(root, tmp_path):
    """S7: a dangling link would be replaced by the rename; a live one is named with its
    target, the folder the user would look in."""
    vendor = load_server_config(root).llama_cpp_dir
    vendor.parent.mkdir(parents=True)
    nowhere = tmp_path / "nowhere"
    try:
        vendor.symlink_to(nowhere, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges on Windows")
    run = FakeRunner()
    with pytest.raises(ConfigError) as refused:
        made(root, run=run).go()
    assert str(refused.value).startswith(
        f"vendor/llama.cpp is a link to {os.readlink(vendor)}, which doesn't exist, so setup "
        "leaves it alone"
    )
    assert run.calls == [] and vendor.is_symlink()
    nowhere.mkdir()
    with pytest.raises(ConfigError) as refused:
        made(root, run=run).go()
    assert str(refused.value).startswith(
        f"vendor/llama.cpp (a link to {vendor.resolve()}) isn't a git checkout"
    )


# --- Checkpoint 1's S8 to S11, C2, C5 and C7


def test_a_dangling_link_named_clarity_command_is_left_alone(root, tmp_path):
    """S8 (#13): a link counts as a file setup didn't write, even when it points nowhere."""
    root, state = done_root(root)
    launcher = desktop_of(root) / "Clarity.command"
    try:
        launcher.symlink_to(tmp_path / "nowhere")
    except OSError:
        pytest.skip("symlinks need privileges on Windows")
    says = Says()
    made(root, says=says, state=state).go()
    assert launcher.is_symlink() and not (tmp_path / "nowhere").exists()
    assert any(line.startswith("Your Desktop already has a Clarity.command") for line in says.lines)


def test_a_missing_desktop_is_said_and_setup_ends(root):
    """S8 (#15): any OSError from the write is said in its own words, not only macOS's."""
    root, state = done_root(root)
    says = Says()
    made(root, says=says, state=state, desktop=root / "no-such-desktop").go()
    said = "\n".join(says.lines)
    assert (
        "Setup couldn't write Clarity.command to your Desktop (No such file or directory)" in said
    )


def test_a_partial_at_another_commit_isn_t_renamed(root):
    """S9 (#3): the rename needs the pin, not only a clean tree."""
    state = states(root, partial={"commit": "0" * 40, "dirty": False})
    with pytest.raises(ConfigError, match="didn't check out"):
        made(root, state=state).go()
    assert not load_server_config(root).llama_cpp_dir.exists()


def test_a_source_whose_git_status_failed_is_refused(root):
    """S9 (#1): git_state's dirty is None when `git status` failed; that isn't clean."""
    source(root)
    run = FakeRunner()
    state = states(root, source={"commit": pin(root), "dirty": None, "error": "made up"})
    with pytest.raises(ConfigError, match="so setup leaves it alone"):
        made(root, run=run, state=state).go()
    assert run.calls == []


def test_git_s_whole_environment(root):
    """S10 (#5, #7): every value but PATH, not only the names."""
    run = FakeRunner()
    made(root, run=run, home=Path("/Users/made-up")).go()
    for argv, env in run.calls:
        if argv[0] != "git":
            continue
        assert {k: v for k, v in env.items() if k != "PATH"} == {
            "HOME": str(Path("/Users/made-up")), "LANG": "C", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
        }  # fmt: skip


def test_git_state_runs_with_git_s_cleared_environment(root):
    """C7: the rename check reads git_state under the same environment as the fetch."""
    state = states(root, partial={"commit": pin(root), "dirty": False})
    setup = made(root, state=state, home=Path("/Users/made-up"))
    setup.go()
    assert state.envs and all(env == setup.git_env() for env in state.envs)


def test_a_model_whose_weights_are_gone_is_fetched_again(two_models):
    """S11 (#11): its lock entry alone isn't "already done"."""
    from tests.conftest import fake_fetch

    root, state = two_models
    fake_fetch(root, "tiny")
    (root / "models" / "tiny" / "tiny-bf16.gguf").unlink()  # where fake_fetch puts it
    fetch = FakeFetch()
    made(root, state=state, models=("tiny",), fetch=fetch).go()
    assert fetch.names == ["tiny"]


def test_bge_m3_keeps_at_least_its_gguf():
    """C2: the sizes table against the lock's own figure."""
    import json

    from tests.conftest import REPO

    lock = json.loads((REPO / "models.lock").read_text(encoding="utf-8"))
    assert SIZES["bge-m3"][1] >= lock["models"]["bge-m3"]["gguf"]["size"]


def test_an_os_error_without_its_own_words_is_still_said(root, monkeypatch):
    """C5: an OSError's strerror can be None; the sentence then names the error itself."""
    root, state = done_root(root)

    def refused(self, *args, **kwargs):
        raise OSError("made up")

    monkeypatch.setattr(Path, "write_text", refused)
    says = Says()
    made(root, says=says, state=state).go()
    said = "\n".join(says.lines)
    assert "Setup couldn't write Clarity.command to your Desktop (made up)." in said


# --- A2's build step (states 10 and 11; decisions 4 and 6)


def cmake_argvs(root: Path, cmake: str = CMAKE) -> list[list[str]]:
    """Getting started's two lines, as setup runs them: the venv's cmake by path, the
    folders under this root, -j the CPU count."""
    import shlex

    argvs = []
    for line in clarity_setup.BUILD_LINES:
        words = shlex.split(line.replace("-j N", f"-j {os.cpu_count() or 1}"))
        argvs.append([cmake, *(str(root / w) if w.startswith("vendor/") else w for w in words[1:])])
    return argvs


def test_llama_server_at_the_pin_is_already_done(root):
    root, state = done_root(root)
    run, says = FakeRunner(), Says(answers=[])
    made(root, run=run, says=says, state=state).go()
    assert run.of("cmake") == []
    assert "llama-server: already done." in says.lines


def test_a_missing_llama_server_is_built_with_getting_started_s_lines(root):
    source(root)
    run, says = FakeRunner(), Says()
    made(
        root, run=run, says=says, state=states(root, source={"commit": pin(root), "dirty": False})
    ).go()
    assert run.of("cmake") == cmake_argvs(root)
    log = root / "vendor" / "llama.cpp" / "build" / "clarity-build.log"
    assert log.read_text(encoding="utf-8").startswith("cmake version 4.4.5")
    assert any(line.startswith(f"Building llama-server at {pin(root)[:12]} with cmake 4.4.5")
               for line in says.lines)  # fmt: skip
    assert "llama-server is built." in says.lines
    assert says.lines[-1].startswith("Setup is done.")


def test_a_llama_server_from_another_commit_is_built_again(root):
    root, state = done_root(root)
    setup_built(root)  # setup's own build folder (the security review's S2)
    run = FakeRunner(llama_commit="0000000")
    built = run.effects["build"]

    def at_the_pin(argv):  # the new build reports the pin
        built(argv)
        run.llama_commit = pin(root)[:7]

    run.effects["build"] = at_the_pin
    says = Says()
    made(root, run=run, says=says, state=state).go()
    assert run.of("cmake") == cmake_argvs(root)
    assert says.asked == [question(*missing_bytes("build"), f"~/{root.name}")]


def test_a_llama_server_that_can_t_run_is_built_again(root):
    """A llama-server the system can't run (OSError: a truncated file, another
    architecture's) isn't built, so setup builds it again instead of stopping with a
    traceback (the whole-branch review's C5)."""
    root, state = done_root(root)
    setup_built(root)
    run, says = FakeRunner(), Says()
    fake = run.__call__
    runs = []

    def runner(argv, env):
        if Path(argv[0]).name.startswith("llama-server") and not runs:
            runs.append(argv)
            raise OSError(8, "Exec format error")
        return fake(argv, env)

    try:
        made(root, run=runner, says=says, state=state).go()
    except OSError as error:
        raise AssertionError(f"setup stopped with a traceback: {error!r}") from error
    assert runs and run.of("cmake") == cmake_argvs(root)
    assert "llama-server is built." in says.lines


def test_a_build_that_doesn_t_report_the_pin_is_refused(root):
    source(root)
    run = FakeRunner(llama_commit="0000000")
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, state=states(root, source={"commit": pin(root), "dirty": False})).go()
    assert str(refused.value) == (
        f"The llama-server setup built reports commit 0000000, not the pinned {pin(root)[:12]}. "
        "The build's output is in vendor/llama.cpp/build/clarity-build.log."
    )


def test_a_failed_build_is_refused_naming_its_log(root):
    source(root)
    run = FakeRunner([(1, "CMake Error: made up\n")])  # the first scripted step: configure
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, state=states(root, source={"commit": pin(root), "dirty": False})).go()
    assert str(refused.value) == (
        "Building llama-server failed: CMake Error: made up. Its whole output is in "
        "vendor/llama.cpp/build/clarity-build.log."
    )
    log = root / "vendor" / "llama.cpp" / "build" / "clarity-build.log"
    assert "CMake Error: made up" in log.read_text(encoding="utf-8")


BUILD_CHANGER_NAMES = [
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "OBJC", "OBJCXX", "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET", "CMAKE_PREFIX_PATH", "GGML_METAL", "LLAMA_CURL",
]  # fmt: skip


@pytest.mark.parametrize("name", BUILD_CHANGER_NAMES)
def test_each_name_that_can_change_a_build_is_said_and_dropped(root, monkeypatch, name):
    """Every name BUILD_CHANGERS matches (one of each prefix's) is said and kept out of
    cmake's environment (the whole-branch review's C1)."""
    for other in [k for k in os.environ if clarity_setup.BUILD_CHANGERS.fullmatch(k)]:
        monkeypatch.delenv(other)
    monkeypatch.setenv(name, "made-up")
    source(root)
    run, says = FakeRunner(), Says()
    made(root, run=run, says=says, state=states(root, source={"commit": pin(root), "dirty": False}),
         home=Path("/Users/made-up")).go()  # fmt: skip
    assert all(name not in env for argv, env in run.calls if argv[0] == CMAKE)
    assert (
        f"The build runs without {name} from your environment, so they can't change it."
        in says.lines
    ), says.lines


def test_the_build_runs_with_four_names_and_says_which_it_dropped(root, monkeypatch):
    monkeypatch.setenv("CC", "/made/up/cc")
    monkeypatch.setenv("SDKROOT", "/made/up/sdk")
    monkeypatch.setenv("GGML_METAL", "0")
    source(root)
    run, says = FakeRunner(), Says()
    made(root, run=run, says=says, state=states(root, source={"commit": pin(root), "dirty": False}),
         home=Path("/Users/made-up")).go()  # fmt: skip
    cmake_envs = [env for argv, env in run.calls if argv[0] == CMAKE]
    assert cmake_envs
    for env in cmake_envs:
        assert set(env) <= {"PATH", "HOME", "TMPDIR", "LANG"} and env["HOME"] == str(
            Path("/Users/made-up")
        )
    assert (
        "The build runs without CC, GGML_METAL and SDKROOT from your environment, so they "
        "can't change it." in says.lines
    )


def test_setup_needs_the_cmake_of_its_own_environment(root, monkeypatch, capsys, tmp_path):
    from clarity_chat import cli

    monkeypatch.setattr(clarity_setup, "venv_cmake", lambda: tmp_path / "no-cmake")
    monkeypatch.setattr(clarity_setup, "this_mac", lambda r: MAC)
    monkeypatch.setattr(clarity_setup, "SETUP_MODELS", ())
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("n\n"))
    monkeypatch.setenv("UV", "/Users/made-up/.local/bin/uv")
    assert cli.main(["--root", str(root), "setup"]) == 1
    assert (
        "clarity: Setup needs the cmake in its own environment. Run it as: uv run --group "
        "setup clarity setup" in capsys.readouterr().err
    )


def test_another_platform_is_said_before_uv_or_cmake(root, monkeypatch, capsys, tmp_path):
    """State 1's sentence comes first: on Linux, without uv or the setup group's cmake,
    setup says it needs a Mac, not how to run it with the setup group (the whole-branch
    review's C4)."""
    from clarity_chat import cli

    monkeypatch.setattr(clarity_setup, "venv_cmake", lambda: tmp_path / "no-cmake")
    monkeypatch.setattr(clarity_setup, "this_mac", lambda r: Mac("Linux", "x86_64", 32 * GIB, 0))
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no uv on it
    assert cli.main(["--root", str(root), "setup"]) == 1
    assert capsys.readouterr().err.startswith(
        "clarity: Setup needs a Mac with Apple silicon; this computer is Linux on x86_64."
    )


# --- the security review's S2, S3 and C2: setup builds into, and deletes, only what it made


def setup_built(root: Path) -> None:
    """Mark the build folder as setup's: it holds clarity-build.log, written before cmake."""
    build = load_server_config(root).llama_cpp_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    (build / "clarity-build.log").write_text("cmake version 4.4.5\n", encoding="utf-8")


def test_the_cmake_argvs_name_server_toml_s_folder(root):
    """S3: llama_cpp_dir's folder, not vendor/llama.cpp, whatever server.toml names."""
    toml = root / "server.toml"
    text = toml.read_text(encoding="utf-8")
    toml.write_text(text.replace('"vendor/llama.cpp"', '"third_party/llama.cpp"'), encoding="utf-8")
    folder = root / "third_party" / "llama.cpp"
    n = str(os.cpu_count() or 1)
    assert made(root).cmake_argvs() == [
        [CMAKE, "-S", str(folder), "-B", str(folder / "build"), "-DCMAKE_BUILD_TYPE=Release",
         "-DLLAMA_CURL=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF"],
        [CMAKE, "--build", str(folder / "build"), "--config", "Release", "-j", n, "--target",
         "llama-server"],
    ]  # fmt: skip


def refused_build(root: Path, thing: str) -> str:
    n = os.cpu_count() or 1
    return (
        f"llama-server needs building at {pin(root)[:12]}, but {thing}, so setup leaves it "
        "alone: it may be your own work. Setup builds only into a build folder it made, never "
        f"through a link. To build llama-server yourself, run Getting started's two lines in "
        f"~/{root.name}, then run setup again:\n"
        "cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release "
        "-DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF\n"
        f"cmake --build vendor/llama.cpp/build --config Release -j {n} --target llama-server"
    )


@pytest.mark.parametrize("kind", ["their-build", "build-link", "source-link"])
def test_a_build_folder_setup_didn_t_make_is_refused_and_no_cmake_runs(root, tmp_path, kind):
    """S2: a checkout at the pin whose llama-server is stale is built again only in a build
    folder setup made (it holds clarity-build.log), and never through a link. Anything else
    is refused before the question, with Getting started's two lines, and left as it was."""
    vendor = load_server_config(root).llama_cpp_dir
    elsewhere = tmp_path / "elsewhere"
    if kind == "their-build":
        source(root)
        thing = "vendor/llama.cpp/build holds a build setup didn't make"
        mine = vendor / "build"
    elif kind == "build-link":
        source(root)
        mine = elsewhere
        elsewhere.mkdir()
        try:
            (vendor / "build").symlink_to(elsewhere, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks need privileges on Windows")
        thing = f"vendor/llama.cpp/build is a link to {os.readlink(vendor / 'build')}"
    else:
        vendor.parent.mkdir(parents=True)
        mine = elsewhere
        (elsewhere / ".git").mkdir(parents=True)
        try:
            vendor.symlink_to(elsewhere, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks need privileges on Windows")
        thing = f"vendor/llama.cpp is a link to {os.readlink(vendor)}"
    binary(root)  # through the link, if there is one: llama-server from another commit
    if kind == "their-build":
        (vendor / "build" / "CMakeCache.txt").write_text("made up", encoding="utf-8")
    else:
        setup_built(root)  # even a build folder setup made isn't built through a link
    before = sorted(str(p.relative_to(mine)) for p in mine.rglob("*"))
    run, says = FakeRunner(llama_commit="0000000"), Says()
    state = states(root, source={"commit": pin(root), "dirty": False})
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, says=says, state=state).go()
    assert str(refused.value) == refused_build(root, thing)
    assert [argv for argv, _ in run.calls if argv[0] == CMAKE] == [] and says.asked == []
    assert sorted(str(p.relative_to(mine)) for p in mine.rglob("*")) == before


def partial_of(root: Path) -> Path:
    return load_server_config(root).llama_cpp_dir.parent / "llama.cpp.partial"


def test_a_partial_that_is_a_link_is_refused_untouched(root, tmp_path):
    """C2: a .partial setup would delete is one it made: a folder, not a link."""
    partial, mine = partial_of(root), tmp_path / "mine"
    mine.mkdir()
    (mine / ".git").mkdir()
    partial.parent.mkdir(parents=True)
    try:
        partial.symlink_to(mine, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges on Windows")
    run, says = FakeRunner(), Says()
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, says=says).go()
    assert str(refused.value) == (
        f"vendor/llama.cpp.partial is a link to {os.readlink(partial)}, so setup leaves it "
        "alone: it may be your own work. Move it out of vendor/, then run setup again."
    )
    assert run.calls == [] and says.asked == [] and partial.is_symlink()
    assert sorted(p.name for p in mine.iterdir()) == [".git"]


def test_a_partial_without_git_is_refused_untouched(root):
    """C2: one with no .git isn't a fetch setup started (git init makes .git first)."""
    partial = partial_of(root)
    partial.mkdir(parents=True)
    (partial / "mine.txt").write_text("made up", encoding="utf-8")
    run, says = FakeRunner(), Says()
    with pytest.raises(ConfigError) as refused:
        made(root, run=run, says=says).go()
    assert str(refused.value) == (
        "vendor/llama.cpp.partial isn't a fetch setup started (it has no .git), so setup "
        "leaves it alone: it may be your own work. Move it out of vendor/, then run setup again."
    )
    assert run.calls == [] and says.asked == []
    assert sorted(p.name for p in partial.iterdir()) == ["mine.txt"]
