"""The gate for v0.4.3's export for Inspect (G0, offline): what inspect_ai does with a log
shaped like the plan's, before Clarity's writer exists. The logs are built with inspect_ai's
own types (`tests/inspect_gate_logs.py`). Each test pins what the pinned inspect_ai does, so
a bump that changes it fails here (CONTRIBUTING.md). The viewer's half is
`test_browser_inspect.py`.

- S1, scoring later: the command the docs give keeps `mark` and adds the new score.
- S2, the round trip, and what sample summaries keep.
- S3, side effects of importing and writing, and the private write (the advisor's M1).
- S4, what `inspect score` leaves on disk.
"""

from __future__ import annotations

import gzip
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from tests import inspect_gate_logs as gate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# The guide's example scorer, with the inner function annotated too: without that, `inspect
# score` refuses it ("requires a return type annotation"; the advisor's C1).
SCORER = """\
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy()])
def mentions_blue():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1 if "blue" in state.output.completion.lower() else 0)

    return score
"""


@pytest.fixture(scope="module")
def logs(tmp_path_factory) -> dict[str, Path]:
    return gate.write_gate_logs(tmp_path_factory.mktemp("gate"))


def inspect_env(home: Path) -> dict[str, str]:
    """This environment, with an empty HOME and none of Inspect's or XDG's settings, so
    whatever Inspect writes for the user lands where the test can see it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INSPECT_", "XDG_", "PYTHON"))}
    return {**env, "HOME": str(home), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def inspect_cli(args: list[str], cwd: Path, home: Path) -> subprocess.CompletedProcess:
    """`inspect ARGS` as a user would run it, non-interactively, from `cwd`."""
    return subprocess.run(
        [sys.executable, "-m", "inspect_ai", *args],
        cwd=cwd,
        env=inspect_env(home),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def scored(log: Path, tmp_path: Path, *flags: str) -> tuple[subprocess.CompletedProcess, Path]:
    """`inspect score` on a copy of LOG with the guide's scorer and FLAGS: the result, and
    the scored copy's path."""
    shutil.copy(log, tmp_path / log.name)
    (tmp_path / "scorer.py").write_text(SCORER, encoding="utf-8")
    (tmp_path / "home").mkdir(exist_ok=True)
    out = tmp_path / "scored.eval"
    args = ["score", log.name, "--scorer", "scorer.py@mentions_blue", *flags]
    return inspect_cli([*args, "--output-file", out.name], tmp_path, tmp_path / "home"), out


def samples_of(log) -> list:
    assert log.samples is not None
    return log.samples


def by_key(log) -> dict:
    """Samples by (id, epoch): nothing depends on their order in a file (the advisor's C3)."""
    return {(s.id, s.epoch): s for s in samples_of(log)}


# --- S1: scoring later


@pytest.mark.parametrize("which", ["chat-mode", "chat"])  # even epochs, then uneven
def test_s1_the_docs_scoring_command_keeps_mark_and_adds_the_new_score(logs, tmp_path, which):
    from inspect_ai.log import read_eval_log

    flags = ("--model", "none", "--action", "append")
    old = os.umask(0o022)
    try:
        result, out = scored(logs[which], tmp_path, *flags)
    finally:
        os.umask(old)
    assert result.returncode == 0, result.stderr
    if os.name != "nt":
        # The scored copy holds the whole conversation, and Inspect writes it with the
        # umask's default mode, readable by others: so the docs add `chmod 600` (M1).
        assert stat.S_IMODE(out.stat().st_mode) == 0o644
    before, after = by_key(read_eval_log(str(logs[which]))), by_key(read_eval_log(str(out)))
    assert after.keys() == before.keys()
    for key, sample in after.items():
        assert sorted(sample.scores) == ["mark", "mentions_blue"], key
        assert sample.scores["mark"] == before[key].scores["mark"], key
    assert read_eval_log(str(out)).eval.model == read_eval_log(str(logs[which])).eval.model


def test_s1_without_model_none_scoring_fails_naming_the_model(logs, tmp_path):
    """Inspect tries to resolve the log's model, and clarity/PROFILE names no provider: it
    fails loudly, and writes nothing (pre-check 3)."""
    result, out = scored(logs["chat-mode"], tmp_path, "--action", "append")
    assert result.returncode != 0
    assert "Model API clarity of model 'clarity/smol-chat' not recognized." in result.stderr
    assert not out.exists()


def test_s1_without_action_append_scoring_drops_mark(logs, tmp_path):
    """Run non-interactively without --action, `inspect score` overwrites the scores: the
    `mark` score is gone. So the docs give --action append."""
    from inspect_ai.log import read_eval_log

    result, out = scored(logs["chat-mode"], tmp_path, "--model", "none")
    assert result.returncode == 0, result.stderr
    assert {tuple(s.scores) for s in samples_of(read_eval_log(str(out)))} == {("mentions_blue",)}


# --- S2: the round trip


def test_s2_every_value_reads_back_as_written(logs):
    from inspect_ai.log import read_eval_log

    for which, (profile, mode) in {
        "chat-mode": ("smol-chat", "chat"),
        "text-mode": ("smol-text", "text"),
    }.items():
        log = read_eval_log(str(logs[which]))
        written = by_key(gate.run_log(profile, mode))
        assert by_key(log).keys() == written.keys()
        for key, sample in by_key(log).items():
            assert sample.metadata == gate.row(profile, key[0], key[1], mode)
            assert sample.metadata == written[key].metadata
            assert sample.scores == written[key].scores
            assert [m.metadata for m in sample.messages] == [
                m.metadata for m in written[key].messages
            ]
            assert [m.text for m in sample.messages] == [m.text for m in written[key].messages]
            assert sample.input == gate.OPENINGS[key[0]]
            assert sample.uuid == written[key].uuid
        assert log.eval.metadata == {
            "clarity": {
                "run": {
                    "kind": "run",
                    "run_id": gate.RUN_ID,
                    "profiles": ["smol-chat", "smol-text"],
                },
                "inspect_format_version": 1,
                "inspect_note": gate.NOTE,
            }
        }
        assert log.eval.tags == gate.TAGS
        assert log.eval.model_generate_config.model_dump(exclude_none=True) == {}


def test_s2_what_the_writer_leaves_as_it_was(logs):
    """The fields the plan leaves empty read back empty; `packages` reads back as {}."""
    from inspect_ai.log import read_eval_log

    spec = read_eval_log(str(logs["chat-mode"])).eval
    assert (spec.task_file, spec.solver, spec.revision, spec.scorers) == (None, None, None, None)
    assert spec.packages == {}
    assert all(not s.events for s in samples_of(read_eval_log(str(logs["chat-mode"]))))


def test_s2_summaries_keep_small_values_and_drop_or_shorten_large_ones(logs):
    """Sample summaries, which the viewer's lists and the samples grid read, keep a
    metadata value of up to 1 KB as it is (the advisor's S2). A larger value that isn't a
    string is replaced by a notice. A string is shortened to 1,024 characters with its
    whitespace collapsed, which is why the exact characters are only in the full sample."""
    from inspect_ai.log import read_eval_log_sample_summaries

    summaries = {(s.id, s.epoch): s for s in read_eval_log_sample_summaries(str(logs["chat-mode"]))}
    for (opening, epoch), summary in summaries.items():
        row = gate.row("smol-chat", int(opening), epoch, "chat")
        for field in ("call_id", "stop_type", "cancelled", "flags", "output_ids", "run"):
            assert summary.metadata[field] == row[field], field
        assert summary.metadata["prompt_text"] == " ".join(row["prompt_text"].split())
    long, short = summaries[gate.LONG_PROMPT[1:]], summaries[(1, 2)]
    assert long.metadata["prompt_ids"] == "Key removed from summary (> 1k)"
    assert short.metadata["prompt_ids"] == gate.row("smol-chat", 1, 2, "chat")["prompt_ids"]
    # The score's answer too: the Markdown reply loses its leading spaces and line breaks.
    scores = summaries[(1, 1)].scores
    assert scores is not None
    markdown = scores["mark"]
    assert markdown.answer == " ".join(gate.MARKDOWN_REPLY.split())
    assert markdown.value == "landed"


# --- S3: side effects


PROBE = """\
import logging, os, signal, sys, threading, warnings
sys.path.insert(0, sys.argv[2])


def snapshot():
    root = logging.getLogger()
    return {
        "environ": dict(os.environ),
        "sigint": repr(signal.getsignal(signal.SIGINT)),
        "sigterm": repr(signal.getsignal(signal.SIGTERM)),
        "root handlers": [repr(h) for h in root.handlers],
        "root level": root.level,
        "excepthook": repr(sys.excepthook),
        "thread excepthook": repr(threading.excepthook),
        "warning filters": len(warnings.filters),
        "cwd": os.getcwd(),
    }


before = snapshot()
import inspect_ai.log, inspect_ai.model, inspect_ai.scorer
imported = snapshot()
from tests.inspect_gate_logs import run_log
target = os.path.join(sys.argv[1], "1-smol-chat.eval")
inspect_ai.log.write_eval_log(run_log("smol-chat", "chat"), target)
inspect_ai.log.read_eval_log(target)
written = snapshot()
for name, after in (("import", imported), ("write", written)):
    changed = sorted(k for k in before if before[k] != after[k])
    print(name, changed)
"""


def test_s3_importing_and_writing_change_nothing_a_chat_depends_on(tmp_path):
    """In a fresh interpreter: importing inspect_ai's log, model and scorer modules, then
    writing and reading a log, leave the environment, the SIGINT and SIGTERM handlers, the
    logging setup, the exception hooks and the warning filters as they were, and write
    nothing under HOME or beside the target. So Part 2 can write in-process (decision 11)."""
    home, folder = tmp_path / "home", tmp_path / "out"
    home.mkdir()
    folder.mkdir()
    (tmp_path / "probe.py").write_text(PROBE, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "probe.py"), str(folder), str(REPO)],
        cwd=tmp_path,
        env=inspect_env(home),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["import []", "write []"]
    assert list(home.rglob("*")) == []
    assert [p.name for p in folder.iterdir()] == ["1-smol-chat.eval"]


def test_s3_the_writers_temporary_file_is_made_only_inside_a_private_folder(tmp_path, monkeypatch):
    """Inspect's writer makes its temporary file in the target's folder at 0666 less the
    umask (the advisor's M1). Given a target inside a 0700 folder made by mkdtemp (decision
    9), that's where the temporary file is made; the file is then set to 0600 and moved out,
    and the folder removed, leaving only the log."""
    from inspect_ai._util import atomic_write
    from inspect_ai.log import read_eval_log, write_eval_log

    made = []
    original = atomic_write._create_local_tempfile

    def recorded(target_dir: str, *args, **kwargs):
        fd, path = original(target_dir, *args, **kwargs)
        made.append(Path(path).parent)
        return fd, path

    monkeypatch.setattr(atomic_write, "_create_local_tempfile", recorded)
    folder = tmp_path / "inspect"
    folder.mkdir()
    private = Path(tempfile.mkdtemp(dir=folder))
    private_mode = private.stat().st_mode
    try:
        write_eval_log(gate.run_log("smol-chat", "chat"), str(private / "1-smol-chat.eval"))
        os.chmod(private / "1-smol-chat.eval", 0o600)
        os.replace(private / "1-smol-chat.eval", folder / "1-smol-chat.eval")
    finally:
        shutil.rmtree(private)
    assert made == [private]
    assert [p.name for p in folder.iterdir()] == ["1-smol-chat.eval"]
    if os.name != "nt":
        assert stat.S_IMODE(private_mode) == 0o700
        assert stat.S_IMODE((folder / "1-smol-chat.eval").stat().st_mode) == 0o600
    assert len(samples_of(read_eval_log(str(folder / "1-smol-chat.eval")))) == 4


# --- S4: what `inspect score` leaves on disk

TRACE_DIR = "from inspect_ai._util.trace import inspect_trace_dir; print(inspect_trace_dir())"


def test_s4_inspect_score_leaves_trace_files_without_sample_text(logs, tmp_path):
    """`inspect score` writes trace files in Inspect's data folder under HOME (on macOS,
    ~/Library/Application Support/inspect_ai/traces/), an empty `logs` folder in its working
    folder, and the scorer's bytecode beside it. The trace files name the log files it
    wrote, not what the samples hold."""
    result, out = scored(logs["chat-mode"], tmp_path, "--model", "none", "--action", "append")
    assert result.returncode == 0, result.stderr
    home = tmp_path / "home"
    if sys.platform != "win32":  # Windows's data folder is under APPDATA, not HOME
        where = subprocess.run(
            [sys.executable, "-c", TRACE_DIR], env=inspect_env(home), capture_output=True, text=True
        )
        assert where.returncode == 0, where.stderr
        traces = Path(where.stdout.strip())
        assert traces.is_relative_to(home)
        written = sorted(p for p in home.rglob("*") if p.is_file())
        assert written and all(p.parent == traces for p in written), written
        text = "\n".join(gzip.decompress(p.read_bytes()).decode("utf-8") for p in written)
        assert "Log Write: scored.eval" in text
        for secret in (*gate.OPENINGS.values(), gate.SYSTEM, "Blue, green and grey."):
            assert secret not in text
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["1-smol-chat.eval", "__pycache__", "home", "logs", "scored.eval", "scorer.py"]
    assert list((tmp_path / "logs").iterdir()) == []
    assert out.exists()
