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
# The made-up uv installer the curl stub serves: it logs its environment's names and puts a
# uv stub in ~/.local/bin. The paths are written in, since the install runs the installer
# with HOME and PATH only.
INSTALLER = """#!/bin/sh
echo "installer-env $(env | cut -d= -f1 | sort | tr '\\n' ' ')" >> "{log}"
mkdir -p "$HOME/.local/bin"
cp "{stub_uv}" "$HOME/.local/bin/uv"
"""
SH_SETS = {"PWD", "SHLVL", "_", "OLDPWD"}  # what sh itself puts in a child's environment


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
                      "FREE_KB": str(500 * 10**6), "TOOLS": "1",
                      "TAG": version()}  # fmt: skip  # what `git describe` finds in ~/Clarity
        self.uv_on_path = True
        self.stub_uv = tmp_path / "uv-stub"
        self.installer = INSTALLER.format(stub_uv=self.stub_uv, log=self.log).encode()
        self.served: bytes | None = None  # what the curl stub serves, if not the installer
        self.curl_fails = False
        self.tmp = tmp_path / "tmp"  # mktemp's folder, as Terminal's TMPDIR gives it

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
        installer.write_bytes(self.installer if self.served is None else self.served)
        self.stub(
            "curl",
            'echo "curl $*" >> "$LOG"\n'
            'out=""; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
            'echo "curl-mode $(ls -l "$out" | cut -c1-10)" >> "$LOG"\n'
            f'[ "{int(self.curl_fails)}" = 1 ] && {{ echo "curl: (22) made up" >&2; exit 22; }}\n'
            f'cp "{installer}" "$out"\n',
        )
        self.stub(
            "git",
            'echo "git $*" >> "$LOG"\n'
            "echo \"git-env $(env | cut -d= -f1 | sort | tr '\\n' ' ')\" >> \"$LOG\"\n"
            "for last; do :; done\n"  # the clone's folder is its last argument
            'case "$*" in *clone*) mkdir -p "$last"; : > "$last/defaults.toml";'
            ' : > "$last/server.toml";'
            ' printf \'name = "clarity-chat"\\n\' > "$last/pyproject.toml";;'
            f' *describe*) [ -n "{f["TAG"]}" ] && echo "{f["TAG"]}" && exit 0; exit 128;; esac\n',
        )
        uv = self.stub(
            "uv",
            'echo "uv $*" >> "$LOG"\n'
            'case "$*" in *clarity*) echo "stdin $* $(head -n 1)" >> "$LOG";; esac\n'
            "echo \"env $(env | cut -d= -f1 | sort | tr '\\n' ' ')\" >> \"$LOG\"\n",
        )
        if not self.uv_on_path:
            uv.rename(self.stub_uv)

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

    def run(
        self, text: str | None = None, tty: bool = True, shell: str = "sh"
    ) -> subprocess.CompletedProcess:
        """`… | sh`: the script on sh's stdin, as curl's pipe gives it. `shell` is another
        POSIX sh to pipe it into instead (dash, Ubuntu's sh)."""
        self.write_stubs()
        self.tmp.mkdir(exist_ok=True)
        env = {"HOME": str(self.home), "PATH": f"{self.bin}:/usr/bin:/bin", "TMPDIR": str(self.tmp)}
        # Its working folder is the test's own, so a relative path lands there, not in the
        # checkout (nothing_outside checks).
        return subprocess.run(
            [shell], input=self.script(text, tty), env=env, capture_output=True, text=True,
            timeout=60, check=False, cwd=self.root,
        )  # fmt: skip

    def lines(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def ran(self, prefix: str) -> list[str]:
        return [line for line in self.lines() if line.startswith(prefix)]


@pytest.fixture
def bench(tmp_path) -> Bench:
    return Bench(tmp_path)


@pytest.fixture(autouse=True)
def nothing_outside():
    """No test here writes in the checkout: a made-up installer once wrote a file named
    `{log}` there, run unformatted from the checkout's folder."""
    before = set(REPO.iterdir())
    yield
    assert set(REPO.iterdir()) == before, set(REPO.iterdir()) ^ before


def text() -> str:
    assert SCRIPT.is_file(), "no install.sh"
    return SCRIPT.read_text(encoding="utf-8")


def version() -> str:
    """install.sh's VERSION, the release tag it clones."""
    found = re.search(r"\nVERSION=(\S+)\n", text())
    assert found
    return found[1]


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
    sha = re.search(r"\nUV_INSTALLER_SHA256=([0-9a-f]{64})\n", script)
    assert sha and len(set(sha[1])) > 1, "UV_INSTALLER_SHA256 isn't a real sha256"
    curls = [line for line in script.splitlines() if re.search(r"\bcurl\b", line.split("#")[0])]
    assert curls and all("--proto '=https' --tlsv1.2" in line for line in curls), curls


@pytest.mark.parametrize("doc", ["README.md", "docs/getting-started.md"])
def test_the_docs_command_fetches_this_script_at_its_own_tag(doc):
    """The release checklist's "install.sh's VERSION and the README's command name the new
    tag", checked: a command naming another tag would fetch another release's script."""
    version = re.search(r"\nVERSION=(\S+)\n", text())
    assert version
    url = "https://raw.githubusercontent.com/andalibmalit/clarity-chat/{}/install.sh"
    commands = re.findall(r"curl .*install\.sh.*", (REPO / doc).read_text(encoding="utf-8"))
    assert commands
    for command in commands:
        assert command.startswith("curl --proto '=https' --tlsv1.2 -fsSL "), command
        assert url.format(version[1]) in command, command


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
    (git_env,) = bench.ran("git-env ")  # nothing of the user's git, and never a prompt
    assert set(git_env.split()[1:]) - SH_SETS == {
        "HOME", "PATH", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT"
    }, git_env  # fmt: skip
    uv = bench.ran("uv ")
    assert uv == [
        "uv sync --locked --inexact --group setup",  # (ii): a hand install keeps its extras
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
        assert set(line.split()[1:]) <= {"HOME", "PATH"} | SH_SETS, line


def test_a_script_cut_off_at_any_line_runs_nothing(bench):
    lines = text().splitlines(keepends=True)
    for cut in range(len(lines)):
        bench.log.write_text("", encoding="utf-8")
        bench.run("".join(lines[:cut]))
        assert bench.lines() == [], (cut, bench.lines())
        assert list(bench.home.iterdir()) == [], cut


@pytest.mark.parametrize("shell", ["sh", "/bin/dash"])
def test_without_a_terminal_it_stops_before_anything(bench, shell):
    """In dash too (CI run 45105458693: Ubuntu's sh is dash, and macOS ships /bin/dash): a
    redirection that fails on `:`, a special builtin, ends a POSIX sh before it says why."""
    done = bench.run(tty=False, shell=shell)
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
    (env,) = bench.ran("installer-env ")  # run once, with HOME and PATH only
    assert set(env.split()[1:]) <= {"HOME", "PATH"} | SH_SETS, env
    assert bench.ran("uv sync --locked --inexact --group setup")


def test_an_installer_with_another_sha256_is_refused_and_not_run(bench):
    bench.uv_on_path = False
    bench.served = bench.installer.replace(b"mkdir", b"echo made up; mkdir")  # not the pinned
    done = bench.run()
    assert done.returncode != 0
    assert "uv's installer doesn't match its checked sha256" in done.stderr
    assert not (bench.home / ".local").exists() and not bench.ran("uv ") and not bench.ran("git ")
    assert list(bench.tmp.iterdir()) == []  # removed on the way out


def test_uv_s_installer_is_private_while_checked_and_removed_after(bench):
    bench.uv_on_path = False
    assert bench.run().returncode == 0
    assert bench.ran("curl-mode ") == ["curl-mode -rw-------"]  # 0600 from birth (mktemp)
    assert list(bench.tmp.iterdir()) == []


def test_a_download_of_uv_s_installer_that_fails_stops_and_leaves_nothing(bench):
    bench.uv_on_path = False
    bench.curl_fails = True
    done = bench.run()
    assert done.returncode != 0 and "curl: (22) made up" in done.stderr
    assert list(bench.tmp.iterdir()) == []
    assert not (bench.home / ".local").exists() and not bench.ran("uv ") and not bench.ran("git ")


def test_a_clarity_folder_that_isn_t_clarity_is_refused(bench):
    (bench.home / "Clarity").mkdir()
    (bench.home / "Clarity" / "mine.txt").write_text("made up", encoding="utf-8")
    done = bench.run()
    assert done.returncode != 0
    assert "~/Clarity exists and isn't Clarity" in done.stderr
    assert not bench.ran("git ") and not bench.ran("uv ")
    assert sorted(p.name for p in (bench.home / "Clarity").iterdir()) == ["mine.txt"]


def clarity_there(bench: Bench) -> None:
    clarity = bench.home / "Clarity"
    clarity.mkdir()
    for name in ("defaults.toml", "server.toml"):
        (clarity / name).write_text("", encoding="utf-8")
    (clarity / "pyproject.toml").write_text('name = "clarity-chat"\n', encoding="utf-8")


def test_clarity_already_there_goes_straight_to_its_setup(bench):
    clarity_there(bench)
    done = bench.run()
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("Clarity is already in ~/Clarity; checking its setup.\n")
    assert not bench.ran("git clone") and bench.ran("uv run --group setup clarity setup")


# --- the security review's S1, C1, C3, (i), (ii) and (v)


@pytest.mark.parametrize("tag", ["v0.0.1", ""], ids=["another-tag", "no-tag"])
def test_clarity_at_another_version_is_said_and_not_updated(bench, tag):
    """C3: an older ~/Clarity keeps its code and pins; the install says so in one sentence."""
    clarity_there(bench)
    bench.facts["TAG"] = tag
    done = bench.run()
    assert done.returncode == 0, done.stderr
    at = tag or "a commit with no release tag"
    assert done.stdout.startswith(
        f"~/Clarity holds Clarity at {at}, not {version()}, and the install doesn't update it.\n"
        "Clarity is already in ~/Clarity; checking its setup.\n"
    ), done.stdout
    (describe,) = bench.ran("git ")
    assert describe == f"git -C {bench.home / 'Clarity'} describe --tags --exact-match"
    (git_env,) = bench.ran("git-env ")
    assert set(git_env.split()[1:]) - SH_SETS == {
        "HOME", "PATH", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT"
    }, git_env  # fmt: skip
    assert bench.ran("uv run --group setup clarity setup")


def test_uv_from_an_earlier_run_is_found_without_the_shell_profile(bench):
    """(i): a second run in the same Terminal window, before the shell profile is read
    again, uses ~/.local/bin/uv rather than downloading its installer again."""
    bench.uv_on_path = False
    bench.write_stubs()
    (bench.home / ".local" / "bin").mkdir(parents=True)
    bench.stub_uv.rename(bench.home / ".local" / "bin" / "uv")
    done = bench.run()
    assert done.returncode == 0, done.stderr
    assert not bench.ran("curl ") and not bench.ran("installer-env ")
    assert bench.ran("uv sync --locked --inexact --group setup")


def test_a_free_disk_figure_that_isn_t_a_number_is_refused(bench):
    """(v): `[` errors on a word, which counts as false, so the check passed."""
    bench.facts["FREE_KB"] = "made-up"
    done = bench.run()
    assert done.returncode != 0
    assert (
        "The install couldn't tell how much disk your home folder has free (df said "
        '"made-up"), so it stopped.' in done.stderr
    ), done.stderr
    assert bench.lines() == [] and list(bench.home.iterdir()) == []


def test_uv_s_required_version_is_the_build_backend_s_range():
    """S1: a uv outside uv_build's range would fetch and run the build backend unhashed
    (uv.lock doesn't lock it); required-version makes such a uv refuse before it runs."""
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    (backend,) = pyproject["build-system"]["requires"]
    assert backend.startswith("uv_build")
    assert pyproject.get("tool", {}).get("uv", {}).get("required-version") == backend.removeprefix(
        "uv_build"
    )
