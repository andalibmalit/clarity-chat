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
import re
import shlex
import shutil
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clarity_chat.backends.llama_cpp import server_binary
from clarity_chat.logs import git_state
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.models import fetch as fetch_model
from clarity_chat.profiles import (
    ConfigError,
    ModelEntry,
    ServerConfig,
    load_models,
    load_server_config,
    shell_folder,
)

GIB = 1024**3
MEMORY_FLOOR = 16 * GIB  # decision 2 (Q6b): the first model beside bge-m3
SPARE = 5 * 10**9  # free disk left after everything setup keeps (decision 9)
LLAMA_URL = "https://github.com/ggml-org/llama.cpp"
# What each missing piece downloads and keeps, in bytes (decision 9). The source: a shallow
# fetch at the pin (its .git) and the checkout; the build: llama-server's build folder.
# Measured on the owner's checkout at the pin (2026-11-22: .git 41 MB, the tree 193 MB, a
# build 148 MB), rounded up; A2's -m setup run confirms them. A model keeps what it
# downloads (the Hugging Face cache) and, for the convert route, its GGUF: bge-m3's pickle
# is 2,271,145,830 bytes (its file on Hugging Face) and its GGUF 1,157,671,136 (models.lock),
# plus its tokenizer. The
# converter's environment (torch, transformers and the rest, in uv's cache and in
# models/.converter) is an estimate
# until the owner's fresh-account run measures it.
SIZES: dict[str, tuple[int, int]] = {
    "source": (50 * 10**6, 250 * 10**6),
    "build": (0, 150 * 10**6),
    "converter": (500 * 10**6, 1_200 * 10**6),
    "bge-m3": (2_300 * 10**6, 3_460 * 10**6),
    # The Q4_0 file is 4,590,807,392 bytes and its tokenizer.json 32,156,428 (Hugging Face);
    # the gguf route keeps both in the Hugging Face cache, nothing more.
    "gemma-4-e4b": (4_630 * 10**6, 4_630 * 10**6),
}
# The models setup fetches, in order: the first chat model (decision 8, Q6a to Q6c), then
# bge-m3, meaning search's embedder.
SETUP_MODELS: tuple[str, ...] = ("gemma-4-e4b", "bge-m3")
# What can change a build from the environment; the build runs without all of it.
BUILD_CHANGERS = re.compile(
    r"(CC|CXX|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|OBJC|OBJCXX|SDKROOT|MACOSX_DEPLOYMENT_TARGET"
    r"|CMAKE_\w*|GGML_\w*|LLAMA_\w*)"
)
# Getting started's two build lines (decision 4), N the CPU count; a test checks the doc says
# the same. Setup can't run them yet (A2's build step waits for Q2 and Q3a), so it says them.
BUILD_LINES = (
    "cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release "
    "-DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF",
    "cmake --build vendor/llama.cpp/build --config Release -j N --target llama-server",
)

Runner = Callable[[list[str], dict[str, str]], tuple[int, str]]


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


def run_logged(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    """A step's command, with its output kept. (The held build step adds a log file.)"""
    import subprocess

    done = subprocess.run(
        argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )  # fmt: skip
    return done.returncode, done.stdout


def shown_link(path: str) -> str:
    """A link's target as the user writes it: without the \\\\?\\ prefix Windows' readlink
    and resolve() add to a long path (CI run 45085021195)."""
    return path.removeprefix("\\\\?\\")


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
        state: Callable[[Path, dict[str, str]], dict[str, Any]] = git_state,
        *,
        home: Path,  # required, as desktop and uv are: nothing defaults to the real ones
        desktop: Path,
        uv: str,
        cmake: str,
        models: tuple[str, ...] = SETUP_MODELS,
        fetch: Callable[[Path, ModelEntry, ServerConfig], dict[str, Any]] = fetch_model,
    ) -> None:
        self.root, self.mac, self.run, self.ask, self.say = root, mac, run, ask, say
        self.state = state
        self.home, self.desktop, self.uv, self.cmake = home, desktop, uv, cmake
        self.models, self.fetch = models, fetch
        config = self.config = load_server_config(root)
        self.pin = config.llama_cpp_commit
        # server.toml's folder as written, not resolved: load_server_config follows a link,
        # and setup must see one to leave it alone (Checkpoint 1's S7).
        written = tomllib.loads((root / "server.toml").read_text(encoding="utf-8"))
        self.source = Path(os.path.normpath(root / written["llama_cpp_dir"]))
        self.partial = self.source.parent / f"{self.source.name}.partial"

    def go(self) -> None:
        """Every step, in order; a ConfigError says why setup stopped."""
        self.check_mac()
        source_done = self.check_source()
        missing = [] if source_done else ["source"]
        built = self.built()
        if not built:
            missing.append("build")
        entries = load_models(self.root)
        models = [entries[name] for name in self.models]
        to_fetch = [m for m in models if not self.fetched(m)]
        missing += [m.name for m in to_fetch]
        if any(m.route == "convert" for m in to_fetch):  # the converter runs only for these
            missing.append("converter")
        download = sum(SIZES[piece][0] for piece in missing)
        keep = sum(SIZES[piece][1] for piece in missing)
        if missing and keep + SPARE > self.mac.free:  # nothing missing keeps nothing
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
        if built:
            self.say("llama-server: already done.")
        else:
            self.build()
        for model in models:  # in order; a failed fetch stops here, and a run again goes on
            if model in to_fetch:
                self.fetch(self.root, model, self.config)
            else:
                self.say(f"{model.name}: already done.")
        self.launcher()
        self.say(
            "Setup is done. Next time, double-click Clarity.command on your Desktop to start "
            "Clarity, or run the same command again to check the setup."
        )

    def launcher(self) -> None:
        """~/Desktop/Clarity.command (state 15; decision 13): three lines, #!/bin/sh (so it
        runs under sh whatever the login shell: Checkpoint 1's C6), `cd` to this folder and
        `exec` uv's absolute path, so it needs no shell profile. A file of that name
        setup didn't write is left alone; a Desktop macOS won't let setup write to is said.
        Either way the lines are printed, and setup still ends without an error."""
        lines = (
            f"#!/bin/sh\ncd {shlex.quote(str(self.root))}\n"
            f"exec {shlex.quote(self.uv)} run clarity ui\n"
        )
        shown = lines.rstrip("\n")
        path = self.desktop / "Clarity.command"
        if path.is_file() and path.read_text(encoding="utf-8", errors="replace") == lines:
            self.say("Clarity.command on your Desktop: already done.")
            return
        if path.exists() or path.is_symlink():
            self.say(
                "Your Desktop already has a Clarity.command that setup didn't write, so it's "
                f"left alone. To start Clarity from a file of your own, put these three lines in "
                f"it:\n{shown}"
            )
            return
        try:
            path.write_text(lines, encoding="utf-8")
            path.chmod(0o755)
        except OSError as e:  # "Don't Allow" at the Desktop prompt, or a managed Mac
            self.say(
                "Setup couldn't write Clarity.command to your Desktop "
                f"({e.strerror or e}). To make the launcher yourself, put these three lines in "
                f"a file named Clarity.command there:\n{shown}"
            )
            return
        self.say("Made Clarity.command on your Desktop: double-click it to start Clarity.")

    def fetched(self, model: ModelEntry) -> bool:
        """Its lock entry, and its weights' size and time against it (state 13)."""
        try:
            check_weights_quick(self.root, lock_entry(self.root, model))
        except ConfigError:
            return False
        return True

    def built(self) -> bool:
        """Whether llama-server is there and reports the pin (state 13); one from another
        commit is built again (state 10)."""
        try:
            binary = server_binary(self.source)
        except ConfigError:
            return False
        return self.pin.startswith(self.reported(binary) or "?")

    def reported(self, binary: Path) -> str | None:
        """The commit `llama-server --version` says it was built from."""
        _, output = self.run([str(binary), "--version"], self.build_env())
        found = re.search(r"commit (\w+)\)", output)
        return found[1] if found else None

    def build_env(self) -> dict[str, str]:
        """cmake's environment: PATH, HOME, TMPDIR and LANG only, so nothing else from the
        user's (CC, CFLAGS, SDKROOT, CMAKE_*, GGML_*, LLAMA_* …) changes the build unseen
        (decision 4)."""
        env = {k: os.environ[k] for k in ("PATH", "TMPDIR", "LANG") if k in os.environ}
        return {**env, "HOME": str(self.home)}

    def build(self) -> None:
        """llama-server at the pin, with Getting started's two lines and the venv's cmake
        (decisions 4 and 6). Its output goes to vendor/llama.cpp/build/clarity-build.log,
        which starts with cmake's version; a failure, or a binary that doesn't report the
        pin, is refused naming it (state 11)."""
        env, build = self.build_env(), self.source / "build"
        log, named = build / "clarity-build.log", "vendor/llama.cpp/build/clarity-build.log"
        build.mkdir(parents=True, exist_ok=True)
        _, version = self.run([self.cmake, "--version"], env)
        log.write_text(version, encoding="utf-8")
        dropped = sorted(k for k in os.environ if BUILD_CHANGERS.fullmatch(k))
        if dropped:
            names = ", ".join(dropped[:-1]) + " and " + dropped[-1] if dropped[1:] else dropped[0]
            self.say(
                f"The build runs without {names} from your environment, so they can't change it."
            )
        cmake = version.split()[2] if version.startswith("cmake version ") else "?"
        self.say(
            f"Building llama-server at {self.pin[:12]} with cmake {cmake} (a few minutes; its "
            f"output goes to {named}) …"
        )
        for argv in self.cmake_argvs():
            code, output = self.run(argv, env)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"$ {shlex.join(argv)}\n{output}")
            if code != 0:
                last = output.strip().splitlines()[-1] if output.strip() else f"exit {code}"
                raise ConfigError(
                    f"Building llama-server failed: {last}. Its whole output is in {named}."
                )
        commit = self.reported(server_binary(self.source))
        if not self.pin.startswith(commit or "?"):
            raise ConfigError(
                f"The llama-server setup built reports commit {commit}, not the pinned "
                f"{self.pin[:12]}. The build's output is in {named}."
            )
        self.say("llama-server is built.")

    def cmake_argvs(self) -> list[list[str]]:
        """BUILD_LINES as argvs: the venv's cmake by path, vendor/ folders under this root."""
        n = str(os.cpu_count() or 1)
        argvs = []
        for line in BUILD_LINES:
            words = shlex.split(line.replace("-j N", f"-j {n}"))
            folder = self.source.parent.parent  # vendor/'s parent: the root the lines are in
            argvs.append(
                [
                    self.cmake,
                    *(str(folder / w) if w.startswith("vendor/") else w for w in words[1:]),
                ]
            )
        return argvs

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
        named = "vendor/llama.cpp"
        if self.source.is_symlink():  # the rename would replace a link: never touch it (S7)
            if not self.source.exists():
                target = shown_link(os.readlink(self.source))
                raise ConfigError(
                    f"vendor/llama.cpp is a link to {target}, which doesn't exist, so setup "
                    "leaves it alone: it may be your own work. Move it out of vendor/, then run "
                    "setup again."
                )
            named = f"vendor/llama.cpp (a link to {shown_link(str(self.source.resolve()))})"
        if not self.source.exists():
            return False
        # Only its own .git: git_state in a plain folder finds the clone around it (S1).
        state = self.state(self.source, self.git_env()) if (self.source / ".git").exists() else {}
        if state.get("commit") is None:
            why = "isn't a git checkout"
        elif state["commit"] != self.pin:
            why = f"is at commit {state['commit'][:12]}, not the pinned {self.pin[:12]}"
        elif state.get("dirty") is not False:
            why = "has uncommitted changes"
        else:
            return True
        raise ConfigError(
            f"{named} {why}, so setup leaves it alone: it may be your own work. Move it out of "
            "vendor/, then run setup again."
        )

    def agreed(self, download: int, keep: int) -> bool:
        """The one question (state 12), before anything is downloaded."""
        answer = self.ask(
            f"Setup will download about {gb(download)} GB and keep about {gb(keep)} GB in "
            f"{shell_folder(self.root, self.home)} and your Hugging Face and uv caches. Free: "
            f"{gb(self.mac.free)} GB. Continue? [Y/n] "
        )
        return answer.strip().lower() in ("", "y", "yes")

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
            code, output = self.run([*where, *step], self.git_env())
            if code != 0:
                message = output.strip().splitlines()[-1] if output.strip() else f"exit {code}"
                raise ConfigError(
                    f"Fetching llama.cpp's source failed: {message}. Run setup again: it "
                    "starts the fetch over."
                )
        state = self.state(self.partial, self.git_env())  # as the fetch ran (C7)
        if state.get("commit") != self.pin or state.get("dirty") is not False:
            raise ConfigError(
                f"llama.cpp's source didn't check out at {self.pin[:12]} cleanly. Run setup "
                "again: it starts the fetch over."
            )
        self.partial.rename(self.source)
        self.say("llama.cpp's source is in vendor/llama.cpp.")


def answer(prompt: str) -> str:
    """The question's answer from the terminal; the end of stdin (no terminal under
    `curl | sh`) is "no" (Checkpoint 1's M3)."""
    try:
        return input(prompt)
    except EOFError:
        print()
        return "n"


def venv_cmake() -> Path:
    """The cmake of the environment running setup (`uv run --group setup`), by path."""
    return Path(sys.executable).parent / ("cmake.exe" if sys.platform == "win32" else "cmake")


def cmd_setup(root: Path) -> int:
    """`clarity setup`, in the terminal: the real Mac, commands and question, the real home
    folder and Desktop. uv's path, for the launcher: `uv run` gives it in UV, else the one
    on PATH; absolute, not resolved, so a link that an update repoints still works (S6)."""
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise ConfigError(
            "Setup can't find uv, which starts Clarity. Install it as Getting started says, "
            "then run setup again."
        )
    cmake = venv_cmake()
    if not cmake.is_file():
        raise ConfigError(
            "Setup needs the cmake in its own environment. Run it as: uv run --group setup "
            "clarity setup"
        )
    home = Path.home()
    Setup(
        root, this_mac(root), run_logged, answer, print,
        home=home, desktop=home / "Desktop", uv=str(Path(uv).absolute()), cmake=str(cmake),
        models=SETUP_MODELS,
    ).go()  # fmt: skip
    return 0
