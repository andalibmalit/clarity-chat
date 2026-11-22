"""`clarity setup` (v0.4.2): a Mac with Apple silicon, from a clone of Clarity to a first chat.

Each step checks its state first and says "already done" when it is, so setup can be run
again after an interruption (decision 10). In order: the platform, the memory, the source
setup never made (refused untouched), the disk, one question with sizes before anything is
downloaded, then llama.cpp's source. The build, the models and the launcher follow (A2's
build step, A3, A7). The Mac's facts, the home folder, the commands run (`run`), the
question's answer (`ask`), what's said (`say`) and git_state are parameters, so the tests
need no Mac, network or git.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarity_chat.backends.llama_cpp import server_binary
from clarity_chat.logs import git_state
from clarity_chat.profiles import ConfigError, load_server_config

GIB = 1024**3
MEMORY_FLOOR = 16 * GIB  # decision 2 (Q6b): the first model beside bge-m3
SPARE = 5 * 10**9  # free disk left after everything setup keeps (decision 9)
LLAMA_URL = "https://github.com/ggml-org/llama.cpp"
# What each missing piece downloads and keeps, in bytes (decision 9). The source: a shallow
# fetch at the pin (its .git) and the checkout; the build: llama-server's build folder.
# Measured on the owner's checkout at the pin (2026-11-22: .git 41 MB, the tree 193 MB, a
# build 148 MB), rounded up; A2's -m setup run confirms them. A3 adds the models.
SIZES: dict[str, tuple[int, int]] = {
    "source": (50 * 10**6, 250 * 10**6),
    "build": (0, 150 * 10**6),
}

Runner = Callable[[list[str], dict[str, str], "Path | None"], tuple[int, str]]


@dataclass(frozen=True)
class Mac:
    """What setup checks before it does anything: platform, memory and free disk (bytes)."""

    system: str
    machine: str
    memory: int
    free: int


def this_mac(root: Path) -> Mac:
    if sys.platform == "win32":  # refused at the platform check; Windows has no os.sysconf
        memory = 0
    else:
        memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return Mac(platform.system(), platform.machine(), memory, shutil.disk_usage(root).free)


def run_logged(argv: list[str], env: dict[str, str], log: Path | None) -> tuple[int, str]:
    """A step's command, its output kept (and added to `log`, if given)."""
    import subprocess

    done = subprocess.run(
        argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )  # fmt: skip
    if log is not None:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(argv)}\n{done.stdout}")
    return done.returncode, done.stdout


def gb(n: int) -> str:
    return f"{n / 1e9:.1f}"


class Setup:
    def __init__(
        self,
        root: Path,
        mac: Mac,
        run: Runner,
        ask: Callable[[str], str],
        say: Callable[[str], None],
        state: Callable[[Path], dict[str, Any]] = git_state,
        home: Path | None = None,
    ) -> None:
        self.root, self.mac, self.run, self.ask, self.say = root, mac, run, ask, say
        self.state = state
        self.home = home or Path.home()
        config = load_server_config(root)
        self.pin = config.llama_cpp_commit
        self.source = config.llama_cpp_dir
        self.partial = self.source.parent / f"{self.source.name}.partial"

    def go(self) -> None:
        """Every step, in order; a ConfigError says why setup stopped."""
        self.check_mac()
        source_done = self.check_source()
        missing = [] if source_done else ["source"]
        if not self.built():
            missing.append("build")
        download = sum(SIZES[piece][0] for piece in missing)
        keep = sum(SIZES[piece][1] for piece in missing)
        if keep + SPARE > self.mac.free:
            raise ConfigError(
                f"Setup needs {gb(keep + SPARE)} GB free for what it still has to keep, with "
                f"5 GB to spare; this disk has {gb(self.mac.free)} GB free."
            )
        if missing and not self.agreed(download, keep):
            self.say("Setup stopped. Nothing was downloaded.")
            return
        if source_done:
            self.say("llama.cpp's source: already done.")
        else:
            self.fetch_source()

    def built(self) -> bool:
        """Whether llama-server is there. Whether it's from the pin is the build step's check
        (A2's build step, held for Q2 and Q3a)."""
        try:
            server_binary(self.source)
        except ConfigError:
            return False
        return True

    def check_mac(self) -> None:
        """States 1 and 2, before anything runs or is written."""
        mac = self.mac
        if (mac.system, mac.machine) != ("Darwin", "arm64"):
            raise ConfigError(
                f"Setup needs a Mac with Apple silicon; this computer is {mac.system} on "
                f'{mac.machine}. To install by hand, follow "Install by hand" in '
                "docs/getting-started.md."
            )
        if mac.memory < MEMORY_FLOOR:
            raise ConfigError(
                f"Clarity needs 16 GB of memory; this Mac has {mac.memory // GIB} GB."
            )

    def check_source(self) -> bool:
        """Whether vendor/llama.cpp is there at the pin; one setup didn't make is refused,
        untouched (state 9: it may be the user's own work, decision 5)."""
        if not self.source.exists():
            return False
        state = self.state(self.source)
        if state.get("commit") is None:
            why = "isn't a git checkout"
        elif state["commit"] != self.pin:
            why = f"is at commit {state['commit'][:12]}, not the pinned {self.pin[:12]}"
        elif state.get("dirty") is not False:
            why = "has uncommitted changes"
        else:
            return True
        raise ConfigError(
            f"vendor/llama.cpp {why}, so setup leaves it alone: it may be your own work. Move "
            "it out of vendor/, then run setup again."
        )

    def agreed(self, download: int, keep: int) -> bool:
        """The one question (state 12), before anything is downloaded."""
        answer = self.ask(
            f"Setup will download about {gb(download)} GB and keep about {gb(keep)} GB in "
            f"{self.shown(self.root)} and your Hugging Face cache. Free: {gb(self.mac.free)} "
            "GB. Continue? [Y/n] "
        )
        return answer.strip().lower() in ("", "y", "yes")

    def shown(self, folder: Path) -> str:
        """A folder as the user knows it: ~/-relative under the home folder."""
        if folder.is_relative_to(self.home):
            return f"~/{folder.relative_to(self.home).as_posix()}"
        return folder.as_posix()

    def git_env(self) -> dict[str, str]:
        """git with nothing of the user's: no GIT_* variables, and neither the global nor
        the system config, so a url.insteadOf rule can't fetch from elsewhere."""
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }

    def fetch_source(self) -> None:
        """llama.cpp at the pin, fetched into .partial and renamed only once git_state says
        it's at the pin and clean (state 8; decision 5). A .partial left by a setup that
        didn't finish is deleted first."""
        if self.partial.exists():
            self.say("Deleting vendor/llama.cpp.partial, left by a setup that didn't finish.")
            shutil.rmtree(self.partial)
        self.say(f"Fetching llama.cpp's source at {self.pin[:12]} …")
        self.partial.mkdir(parents=True)
        where = ["git", "-C", str(self.partial)]
        for step in (
            ["init", "-q"],
            ["fetch", "--depth", "1", "-q", LLAMA_URL, self.pin],
            ["checkout", "--detach", "-q", self.pin],
        ):
            code, output = self.run([*where, *step], self.git_env(), None)
            if code != 0:
                message = output.strip().splitlines()[-1] if output.strip() else f"exit {code}"
                raise ConfigError(
                    f"Fetching llama.cpp's source failed: {message}. Run setup again: it "
                    "starts the fetch over."
                )
        state = self.state(self.partial)
        if state.get("commit") != self.pin or state.get("dirty") is not False:
            raise ConfigError(
                f"llama.cpp's source didn't check out at {self.pin[:12]} cleanly. Run setup "
                "again: it starts the fetch over."
            )
        self.partial.rename(self.source)
        self.say("llama.cpp's source is in vendor/llama.cpp.")


def cmd_setup(root: Path) -> int:
    """`clarity setup`, in the terminal: the real Mac, commands and question."""
    Setup(root, this_mac(root), run_logged, input, print).go()
    return 0
