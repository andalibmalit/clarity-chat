"""v0.4.2 A6: the install command, install.sh, run as `curl … | sh` runs it: piped into sh,
with stub commands on PATH that record their arguments, their environment and where their
standard input comes from. /dev/tty is swapped, in the copy each test pipes, for a file that
stands for the terminal (or a path that can't be opened, for no terminal), since a sandbox
has no pseudo-terminals to give. HOME is a temporary folder; nothing is downloaded and
nothing touches the real home folder. The real run on a fresh Mac is the owner's (the plan's
By hand).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO

if sys.platform == "win32":  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="no POSIX sh on Windows")
TERMINAL = "made-up terminal"  # what the file standing for /dev/tty holds

SCRIPT = REPO / "install.sh"
GIB = 1024**3
# The made-up uv installer the curl stub serves: it puts a uv stub in ~/.local/bin.
INSTALLER = """#!/bin/sh
mkdir -p "$HOME/.local/bin"
cp "$STUB_UV" "$HOME/.local/bin/uv"
"""


class Bench:
    """A made-up Mac for install.sh: its facts, its home folder, its commands' log."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.bin = tmp_path / "bin"
        self.log = tmp_path / "log"
        self.home.mkdir()
        self.bin.mkdir()
        self.log.write_text("", encoding="utf-8")
        self.facts = {"SYSTEM": "Darwin", "MACHINE": "arm64", "MEMORY": str(32 * GIB),
                      "FREE_KB": str(500 * 10**6), "TOOLS": "1"}  # fmt: skip
        self.uv_on_path = True
        self.installer = INSTALLER.encode()

    def stub(self, name: str, body: str) -> Path:
        path = self.bin / name
        path.write_text(f"#!/bin/sh\nLOG={self.log}\n{body}", encoding="utf-8")
        path.chmod(0o755)
        return path

    def write_stubs(self) -> None:
        f = self.facts
        self.stub(
            "uname", f'case "$1" in -s) echo {f["SYSTEM"]};; -m) echo {f["MACHINE"]};; esac\n'
        )
        self.stub("sysctl", f"echo {f['MEMORY']}\n")
        self.stub(
            "df",
            "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
            f"printf '/dev/disk3s1 900000000 1 {f['FREE_KB']} 1%% /\\n'\n",
        )
        self.stub(
            "xcode-select",
            f'echo "xcode-select $*" >> "$LOG"\n'
            f'[ "$1" = "-p" ] && {{ [ "{f["TOOLS"]}" = 1 ] && exit 0 || exit 2; }}\nexit 0\n',
        )
        installer = self.root / "installer.sh"
        installer.write_bytes(self.installer)
        self.stub(
            "curl",
            'echo "curl $*" >> "$LOG"\n'
            'out=""; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
            f'cp "{installer}" "$out"\n',
        )
        self.stub(
            "git",
            'echo "git $*" >> "$LOG"\n'
            "for last; do :; done\n"  # the clone's folder is its last argument
            'case "$*" in *clone*) mkdir -p "$last"; : > "$last/defaults.toml";'
            ' : > "$last/server.toml";'
            ' printf \'name = "clarity-chat"\\n\' > "$last/pyproject.toml";; esac\n',
        )
        uv = self.stub(
            "uv",
            'echo "uv $*" >> "$LOG"\n'
            'case "$*" in *clarity*) echo "stdin $* $(head -n 1)" >> "$LOG";; esac\n'
            "echo \"env $(env | cut -d= -f1 | sort | tr '\\n' ' ')\" >> \"$LOG\"\n",
        )
        if not self.uv_on_path:
            stub_uv = self.root / "uv-stub"
            uv.rename(stub_uv)
            self.stub_uv = stub_uv

    def script(self, text: str | None = None, tty: bool = True) -> str:
        """install.sh, with its uv installer's sha256 swapped for the made-up installer's,
        so the check runs against bytes the curl stub serves, and /dev/tty for the file
        that stands for the terminal (or, without `tty`, a path that can't be opened)."""
        if text is None:
            assert SCRIPT.is_file(), "no install.sh"  # in the test: a red fails, not errors
            text = SCRIPT.read_text(encoding="utf-8")
        digest = hashlib.sha256(self.installer).hexdigest()
        text = re.sub(r"UV_INSTALLER_SHA256=\w+", f"UV_INSTALLER_SHA256={digest}", text)
        terminal = self.root / "terminal"
        terminal.write_text(f"{TERMINAL}\n", encoding="utf-8")
        return text.replace("/dev/tty", str(terminal if tty else self.root / "no" / "tty"))

    def run(self, text: str | None = None, tty: bool = True) -> subprocess.CompletedProcess:
        """`… | sh`: the script on sh's stdin, as curl's pipe gives it."""
        self.write_stubs()
        env = {"HOME": str(self.home), "PATH": f"{self.bin}:/usr/bin:/bin"}
        if not self.uv_on_path:
            env["STUB_UV"] = str(self.stub_uv)
        return subprocess.run(
            ["sh"], input=self.script(text, tty), env=env, capture_output=True, text=True,
            timeout=60, check=False,
        )  # fmt: skip

    def lines(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def ran(self, prefix: str) -> list[str]:
        return [line for line in self.lines() if line.startswith(prefix)]


@pytest.fixture
def bench(tmp_path) -> Bench:
    return Bench(tmp_path)


def text() -> str:
    assert SCRIPT.is_file(), "no install.sh"
    return SCRIPT.read_text(encoding="utf-8")


def test_shellcheck_finds_nothing():
    shellcheck = Path(sys.executable).parent / "shellcheck"
    script = text()
    assert script  # read before running, so a missing file fails here
    done = subprocess.run(
        [str(shellcheck), "--shell=sh", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_its_version_is_pyproject_s_and_every_curl_is_https_only():
    import tomllib

    version = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    script = text()
    assert f"\nVERSION=v{version['version']}\n" in script
    assert re.search(r"\nUV_VERSION=\d+\.\d+\.\d+\n", script)
    assert re.search(r"\nUV_INSTALLER_SHA256=[0-9a-f]{64}\n", script)
    curls = [line for line in script.splitlines() if re.search(r"\bcurl\b", line.split("#")[0])]
    assert curls and all("--proto '=https' --tlsv1.2" in line for line in curls), curls


def test_its_body_is_one_function_called_on_the_last_line():
    lines = [line for line in text().splitlines() if line.strip()]
    assert lines[-1] == 'main "$@"'


def test_a_mac_installs_clarity_and_runs_setup_and_the_page_on_the_terminal(bench):
    done = bench.run()
    assert done.returncode == 0, done.stderr
    clarity = bench.home / "Clarity"
    clones = bench.ran("git ")
    assert len(clones) == 1 and "clone --depth 1 --branch v" in clones[0]
    assert clones[0].endswith(str(clarity))
    uv = bench.ran("uv ")
    assert uv == [
        "uv sync --locked --group setup",
        "uv run --group setup clarity setup",
        "uv run clarity ui",
    ]
    stdins = [line for line in bench.ran("stdin ") if " clarity " in line]
    assert len(stdins) == 2
    for line in stdins:  # setup's question and the page read the terminal, not the pipe
        assert line.endswith(TERMINAL), line
    envs = bench.ran("env ")
    assert envs
    for line in envs:  # env -i: only HOME and PATH, and what sh itself sets
        assert set(line.split()[1:]) <= {"HOME", "PATH", "PWD", "SHLVL", "_", "OLDPWD"}, line


def test_a_script_cut_off_at_any_line_runs_nothing(bench):
    lines = text().splitlines(keepends=True)
    for cut in range(len(lines)):
        bench.log.write_text("", encoding="utf-8")
        bench.run("".join(lines[:cut]))
        assert bench.lines() == [], (cut, bench.lines())
        assert list(bench.home.iterdir()) == [], cut


def test_without_a_terminal_it_stops_before_anything(bench):
    done = bench.run(tty=False)
    assert done.returncode != 0
    assert "The install needs a terminal to ask you one question." in done.stderr
    assert bench.lines() == [] and list(bench.home.iterdir()) == []


@pytest.mark.parametrize(
    "fact, value, said",
    [
        ("SYSTEM", "Linux", "is for a Mac with Apple silicon; this computer is Linux on arm64"),
        ("MACHINE", "x86_64", "is for a Mac with Apple silicon; this computer is Darwin on x86_64"),
        ("MEMORY", str(8 * GIB), "Clarity needs 16 GB of memory; this Mac has 8 GB."),
        ("FREE_KB", "1000", "free in your home folder"),
    ],
    ids=["linux", "intel", "8-gb", "disk"],
)
def test_another_mac_is_refused_with_nothing_written(bench, fact, value, said):
    bench.facts[fact] = value
    done = bench.run()
    assert done.returncode != 0 and said in done.stderr, done.stderr
    assert bench.lines() == [] and list(bench.home.iterdir()) == []


def test_without_the_command_line_tools_it_starts_apple_s_installer_and_stops(bench):
    bench.facts["TOOLS"] = "0"
    done = bench.run()
    assert done.returncode != 0
    assert "click Install, then paste the command again" in done.stderr
    assert bench.ran("xcode-select --install") and not bench.ran("git ") and not bench.ran("uv ")


def test_without_uv_its_installer_is_checked_then_run(bench):
    bench.uv_on_path = False
    done = bench.run()
    assert done.returncode == 0, done.stderr
    (fetch,) = bench.ran("curl ")
    assert "--proto =https --tlsv1.2" in fetch and re.search(
        r"https://astral\.sh/uv/[\d.]+/", fetch
    )
    assert (bench.home / ".local" / "bin" / "uv").is_file()
    assert bench.ran("uv sync --locked --group setup")


def test_an_installer_with_another_sha256_is_refused_and_not_run(bench):
    bench.uv_on_path = False
    bench.run()  # writes the stubs
    bench.log.write_text("", encoding="utf-8")
    script = bench.script()
    wrong = re.sub(r"UV_INSTALLER_SHA256=\w+", "UV_INSTALLER_SHA256=" + "0" * 64, script)
    done = bench.run(wrong)
    assert done.returncode != 0
    assert "uv's installer doesn't match its checked sha256" in done.stderr
    assert not (bench.home / ".local").exists() and not bench.ran("uv ") and not bench.ran("git ")


def test_a_clarity_folder_that_isn_t_clarity_is_refused(bench):
    (bench.home / "Clarity").mkdir()
    (bench.home / "Clarity" / "mine.txt").write_text("made up", encoding="utf-8")
    done = bench.run()
    assert done.returncode != 0
    assert "~/Clarity exists and isn't Clarity" in done.stderr
    assert not bench.ran("git ") and not bench.ran("uv ")
    assert sorted(p.name for p in (bench.home / "Clarity").iterdir()) == ["mine.txt"]


def test_clarity_already_there_goes_straight_to_its_setup(bench):
    clarity = bench.home / "Clarity"
    clarity.mkdir()
    for name in ("defaults.toml", "server.toml"):
        (clarity / name).write_text("", encoding="utf-8")
    (clarity / "pyproject.toml").write_text('name = "clarity-chat"\n', encoding="utf-8")
    done = bench.run()
    assert done.returncode == 0, done.stderr
    assert "Clarity is already in ~/Clarity; checking its setup." in done.stdout
    assert not bench.ran("git ") and bench.ran("uv run --group setup clarity setup")
