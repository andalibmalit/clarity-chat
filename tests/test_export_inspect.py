"""Export for Inspect (v0.4.3 I3): a finished run as one Inspect log per profile, read back
with inspect_ai's own reader. Every text is made up."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from clarity_chat import cli, export_inspect
from clarity_chat.batch import run_batch
from clarity_chat.export import run_conversations
from clarity_chat.export_inspect import INSPECT_NOTE, export_run
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.conftest import REPO
from tests.fakes import FakeBackend
from tests.inspect_gate_logs import NOTE

pytest.importorskip("inspect_ai")

OPENINGS = "The kettle sang twice.\n---\n  Ask me about otters.  \n"
WHOLE_TEXT = "text mode: the template has no message roles"
Records = list[dict[str, Any]]


def batch(root: Path, tokenizer, profiles: list[str]) -> tuple[Session, Path]:
    """A finished run of `profiles`, two openings, two seeds."""
    (root / "openings.txt").write_text(OPENINGS, encoding="utf-8")
    run_dir = root / "runs" / "run-test"
    backend = FakeBackend(None, n_ctx=512, tokenizer=tokenizer)
    session = Session(
        root, backend=backend, logs_dir=run_dir, tokenizer_loader=lambda r, v: tokenizer
    )
    run_batch(session, "run-test", run_dir, root / "openings.txt", profiles, 2, lambda line: None)
    return session, run_dir


@pytest.fixture
def run(fetched_root, chat_tokenizer):
    """A finished run of the chat and text profiles, two openings, two seeds."""
    return batch(fetched_root, chat_tokenizer, ["chat", "text"])


def records_of(run_dir: Path) -> Records:
    return list(read_records(run_dir / "calls.jsonl"))


def calls_of(run_dir: Path, profile: str | None = None) -> Records:
    return [
        r
        for r in records_of(run_dir)
        if r["type"] == "call" and profile in (None, r["profile"]["name"])
    ]


def rewrite(run_dir: Path, records: Records) -> None:
    text = "".join(json.dumps(r) + "\n" for r in records)
    (run_dir / "calls.jsonl").write_text(text, encoding="utf-8")


def read(path: Path | str):
    from inspect_ai.log import read_eval_log

    assert Path(path).is_file(), f"the export wrote no {path}"
    return read_eval_log(str(path))


def rows_of(session: Session, run_dir: Path) -> dict[str, dict[str, Any]]:
    """The plain export's row for every call, with its `run`: what each sample's metadata is."""
    _, conversations = run_conversations(session, run_dir)
    return {row["call_id"]: row for c in conversations for row in c.rows}


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_a_run_gives_one_log_per_profile_in_the_runs_order(run):
    session, run_dir = run
    result = export_run(session, run_dir)
    folder = run_dir / "inspect"
    assert [f["path"] for f in result["files"]] == [
        str(folder / "1-chat.eval"),
        str(folder / "2-text.eval"),
    ]
    assert [(f["replaced"], f["calls"]) for f in result["files"]] == [(False, 4), (False, 4)]
    assert sorted(p.name for p in folder.iterdir()) == ["1-chat.eval", "2-text.eval"]
    if os.name != "nt":
        assert {(folder / n).stat().st_mode & 0o777 for n in ("1-chat.eval", "2-text.eval")} == {
            0o600
        }


def test_two_profiles_with_one_name_keep_their_own_calls(fetched_root, chat_tokenizer):
    """The probe of Checkpoint 1's M1: profiles/chat.toml and alt/chat.toml share a name,
    and the run calls them by model group (text, chat, alt), not in --profiles' order."""
    alt = fetched_root / "alt" / "chat.toml"
    alt.parent.mkdir()
    chat = (fetched_root / "profiles" / "chat.toml").read_text(encoding="utf-8")
    alt.write_text("ctx_size = 400\n" + chat, encoding="utf-8")
    session, run_dir = batch(fetched_root, chat_tokenizer, ["text", str(alt), "chat"])
    paths = {r["call_id"]: r["profile"]["path"] for r in calls_of(run_dir)}
    in_log_order = list(dict.fromkeys(paths.values()))
    assert in_log_order[2] == alt.as_posix()  # the log's order isn't --profiles'
    result = export_run(session, run_dir)
    names = [Path(f["path"]).name for f in result["files"]]
    assert names == ["1-text.eval", "2-chat.eval", "3-chat.eval"]
    logs = [read(f["path"]) for f in result["files"]]
    held = [{paths[s.metadata["call_id"]] for s in log.samples} for log in logs]
    assert held == [{in_log_order[0]}, {alt.as_posix()}, {in_log_order[1]}]


def test_each_sample_is_its_calls_row_and_its_messages_are_slices_of_the_prompt(run):
    session, run_dir = run
    export_run(session, run_dir)
    rows = rows_of(session, run_dir)
    chat, text = (
        read(run_dir / "inspect" / "1-chat.eval"),
        read(run_dir / "inspect" / "2-text.eval"),
    )
    assert len(chat.samples) == len(text.samples) == 4
    for log in (chat, text):
        for sample in log.samples:
            row = rows[sample.metadata["call_id"]]
            assert sample.metadata == row  # flat: the plain export's row plus `run` (S2)
            assert (sample.id, sample.epoch) == (
                row["run"]["opening"],
                row["settings"]["seed"]["value"] - 41,
            )
            reply = sample.messages[-1]
            assert (reply.role, reply.text) == ("assistant", row["output_text"])
            assert sample.output.completion == row["output_text"]
            assert sample.output.model == log.eval.model
            assert sample.events == [] and sample.target == ""
        assert {(s.id, s.epoch) for s in log.samples} == {(1, 1), (1, 2), (2, 1), (2, 2)}
        assert log.eval.config.epochs == 2
    for sample in chat.samples:
        row = rows[sample.metadata["call_id"]]
        spans = [s for s in row["message_spans"] if s["kind"] == "message"]
        assert [m.role for m in sample.messages[:-1]] == [s["role"] for s in spans]
        for message, span in zip(sample.messages[:-1], spans, strict=True):
            start, end = span["chars"]
            assert message.text == row["prompt_text"][start:end]
            assert message.metadata == {"clarity": span}
    for sample in text.samples:
        row = rows[sample.metadata["call_id"]]
        [message, _] = sample.messages
        assert (message.role, message.text) == ("user", row["prompt_text"])
        assert message.metadata == {"clarity": {"whole_prompt": WHOLE_TEXT}}
    # Input is the opening's span in both modes, so one opening's rows match (S6).
    inputs = {(s.id, s.input) for s in chat.samples}
    assert inputs == {(s.id, s.input) for s in text.samples}
    assert inputs == {(1, "The kettle sang twice."), (2, "Ask me about otters.")}


def test_the_log_says_clarity_made_the_calls_and_carries_the_header(run):
    from inspect_ai.model import GenerateConfig

    session, run_dir = run
    export_run(session, run_dir)
    run_record, conversations = run_conversations(session, run_dir)
    records = records_of(run_dir)
    for n, profile in enumerate(("chat", "text"), 1):
        log = read(run_dir / "inspect" / f"{n}-{profile}.eval")
        calls = calls_of(run_dir, profile)
        ids = [c["call_id"] for c in calls]
        headers = [c.header for c in conversations if c.rows[0]["call_id"] in ids]
        groups: dict[str, list[str]] = {}
        for header in headers:
            for group in header["conditions"]:
                key = json.dumps(group["conditions"], sort_keys=True)
                groups.setdefault(key, []).extend(group["call_ids"])
        first = {k: v for k, v in headers[0].items() if k != "conversation_id"}
        assert log.eval.metadata == {
            "inspect_note": INSPECT_NOTE,  # top level, so the viewer's Info tab shows it
            "clarity": {
                **first,
                "run": run_record,
                "calls": 4,
                "conditions": [
                    {"conditions": json.loads(k), "call_ids": v} for k, v in groups.items()
                ],
                "inspect_format_version": 1,
            },
        }
        assert sorted(i for v in groups.values() for i in v) == sorted(ids)
        assert log.status == "success"
        assert log.eval.model == f"clarity/{profile}"
        assert log.eval.model_generate_config == GenerateConfig()
        assert log.eval.tags == ["clarity", "not-run-by-inspect"]
        assert (log.eval.task, log.eval.run_id) == ("clarity-run", "run-test")
        assert utc(log.eval.created) == utc(records[0]["time"])
        assert (log.eval.revision, log.eval.solver, log.eval.task_file) == (None, None, None)
        assert log.eval.scorers is None
        assert (log.results.total_samples, log.results.completed_samples) == (4, 4)
        assert log.results.scores == []
        # V7: the times are the calls', so the viewer shows no 1969 and no NaN.
        assert utc(log.stats.started_at) == utc(records[0]["time"])
        assert utc(log.stats.completed_at) == utc(calls[-1]["time"])
    assert INSPECT_NOTE == NOTE  # the spec's sentence, as the gate showed it
    reference = (REPO / "docs" / "reference.md").read_text(encoding="utf-8")
    assert f'`inspect_note` reads: "{INSPECT_NOTE}"' in reference  # the docs quote it whole


def test_two_exports_of_one_run_read_back_equal_and_the_second_replaces(run, tmp_path):
    session, run_dir = run
    folder = run_dir / "inspect"
    export_run(session, run_dir)
    first = {n: read(folder / n) for n in ("1-chat.eval", "2-text.eval")}
    again = export_run(session, run_dir)
    assert [f["replaced"] for f in again["files"]] == [True, True]
    assert sorted(p.name for p in folder.iterdir()) == ["1-chat.eval", "2-text.eval"]
    for name, log in first.items():
        assert read(folder / name).model_dump() == log.model_dump()  # ids and uuids too
    ids = [s.uuid for log in first.values() for s in log.samples]
    ids += [m.id for log in first.values() for s in log.samples for m in s.messages]
    ids += [log.eval.eval_id for log in first.values()]
    assert len(set(ids)) == len(ids)


def test_marks_and_notes_become_the_mark_score(run):
    session, run_dir = run
    log_path = str(run_dir / "calls.jsonl")
    marked, noted, *_ = calls_of(run_dir, "chat")
    for label in ("too long", "landed"):
        assert cli.main(["mark", log_path, marked["call_id"], label]) == 0
    assert cli.main(["note", log_path, marked["call_id"], "made-up note two"]) == 0
    # A note replaces the reply's current one; notes from before note IDs all show.
    records = records_of(run_dir)
    note = next(r for r in records if r["type"] == "note")
    older = {k: v for k, v in note.items() if k != "note_id"}
    records.insert(records.index(note), {**older, "text": "made-up note one"})
    rewrite(run_dir, records)
    assert cli.main(["note", log_path, noted["call_id"], "made-up note three"]) == 0
    export_run(session, run_dir)
    rows = rows_of(session, run_dir)
    scores = {}
    for name in ("1-chat.eval", "2-text.eval"):
        for sample in read(run_dir / "inspect" / name).samples:
            row = rows[sample.metadata["call_id"]]
            [score] = sample.scores.values()
            assert list(sample.scores) == ["mark"]
            assert score.answer == row["output_text"]
            assert score.metadata == {"marks": row["marks"], "notes": row["notes"]}
            scores[row["call_id"]] = (score.value, score.explanation)
    assert scores.pop(marked["call_id"]) == (
        "landed, too long",
        "made-up note one\n\nmade-up note two",
    )
    assert scores.pop(noted["call_id"]) == ("unmarked", "made-up note three")
    assert set(scores.values()) == {("unmarked", None)}


def test_unavailable_spans_give_one_whole_prompt_message_and_the_turns_text(run):
    session, run_dir = run
    records = records_of(run_dir)
    call = next(r for r in records if r["type"] == "call" and r["profile"]["name"] == "chat")
    call["message_spans"] = {"unavailable": "made-up reason"}
    rewrite(run_dir, records)
    export_run(session, run_dir)
    log = read(run_dir / "inspect" / "1-chat.eval")
    sample = next(s for s in log.samples if s.metadata["call_id"] == call["call_id"])
    [message, _] = sample.messages
    assert (message.role, message.text) == ("user", call["prompt_text"])
    assert message.metadata == {
        "clarity": {"whole_prompt": "the message spans are unavailable: made-up reason"}
    }
    assert sample.input == "The kettle sang twice."  # the turn's logged text


def test_stop_types_become_stop_reasons_and_a_limit_is_keyed_on_its_flag(run):
    session, run_dir = run
    records = records_of(run_dir)
    calls = [r for r in records if r["type"] == "call"]  # four chat, then four text
    changes = [
        ("word", []),
        ("none", []),
        ("limit", ["length"]),  # max_tokens reached
        ("limit", ["context_full"]),  # llama-server's `truncated`: the context filled
        ("limit", []),  # neither flag: not one Clarity logs
        ("cancelled", []),
    ]
    for call, (stop_type, flags) in zip(calls, changes, strict=False):
        call["stop_type"], call["flags"] = stop_type, flags
    rewrite(run_dir, records)
    export_run(session, run_dir)
    reasons = {
        s.metadata["call_id"]: s.output.choices[0].stop_reason
        for name in ("1-chat.eval", "2-text.eval")
        for s in read(run_dir / "inspect" / name).samples
    }
    assert [reasons[c["call_id"]] for c in calls] == [
        "stop",
        "unknown",
        "max_tokens",
        "model_length",
        "unknown",
        "unknown",
        "stop",  # eos, as the run left them
        "stop",
    ]


def test_the_file_is_private_from_birth_and_a_failed_write_leaves_nothing(run, monkeypatch):
    import inspect_ai.log

    session, run_dir = run
    folder = run_dir / "inspect"
    real = inspect_ai.log.write_eval_log
    seen: list[tuple[Path, int, list[str]]] = []

    def watched(log, location, *args, **kwargs):
        target = Path(location)
        real(log, location, *args, **kwargs)
        seen.append((target.parent, target.parent.stat().st_mode & 0o777, os.listdir(folder)))

    monkeypatch.setattr(inspect_ai.log, "write_eval_log", watched)
    export_run(session, run_dir)
    assert len(seen) == 2
    for parent, mode, present in seen:
        assert parent.parent == folder and parent.name not in ("1-chat.eval", "2-text.eval")
        assert parent.name in present and not parent.exists()  # made, used, removed
        if os.name != "nt":
            assert mode == 0o700

    def failing(log, location, *args, **kwargs):
        Path(location).write_bytes(b"half a file")
        raise OSError(28, "No space left on device")

    shutil_free = sorted(p.name for p in folder.iterdir())
    monkeypatch.setattr(inspect_ai.log, "write_eval_log", failing)
    with pytest.raises(ConfigError) as e:
        export_run(session, run_dir)
    assert str(e.value) == (
        f"Can't write the export to {folder / '1-chat.eval'}: No space left on device."
    )
    assert sorted(p.name for p in folder.iterdir()) == shutil_free  # no temporary folder


def test_a_profile_name_that_cant_name_a_file_is_refused_before_any_file(run):
    session, run_dir = run
    records = records_of(run_dir)
    for record in records:
        if record["type"] == "call" and record["profile"]["name"] == "text":
            record["profile"]["name"] = "../text"
    rewrite(run_dir, records)
    with pytest.raises(ConfigError) as e:
        export_run(session, run_dir)
    assert str(e.value) == (
        f"../text, a profile name in {run_dir / 'calls.jsonl'}, isn't one Clarity writes, so "
        "it can't name the export's file."
    )
    assert not (run_dir / "inspect").exists()


def listed_but_never_called(run_dir: Path) -> None:
    records = records_of(run_dir)
    records[0]["profiles"].append("ghost")
    rewrite(run_dir, records)


def a_conversation_without_its_call(run_dir: Path) -> None:
    records = records_of(run_dir)
    first = next(r for r in records if r["type"] == "call")
    rewrite(run_dir, [r for r in records if r is not first])


def a_key_naming_another_profile(run_dir: Path) -> None:
    key = json.loads((run_dir / "key.json").read_text(encoding="utf-8"))
    next(iter(key["openings"][0]["labels"].values()))["profile"] = "stranger"
    (run_dir / "key.json").write_text(json.dumps(key), encoding="utf-8")


@pytest.mark.parametrize(
    "edit", [listed_but_never_called, a_conversation_without_its_call, a_key_naming_another_profile]
)
def test_a_run_whose_records_dont_agree_is_refused_as_unreadable(run, edit):
    """Checkpoint 1's S1: each of these was a traceback (StopIteration), or an export of the
    wrong calls, not a refusal."""
    session, run_dir = run
    edit(run_dir)
    with pytest.raises(ConfigError) as e:
        export_run(session, run_dir)
    assert str(e.value).startswith(f"Can't read the records of {run_dir / 'calls.jsonl'} (")
    assert not (run_dir / "inspect").exists()


def test_without_the_extra_or_with_another_version_the_export_is_refused(run, monkeypatch):
    session, run_dir = run
    for name in ("inspect_ai", "inspect_ai.log"):
        monkeypatch.setitem(sys.modules, name, None)  # the import fails
    with pytest.raises(ConfigError) as e:
        export_run(session, run_dir)
    assert str(e.value) == (
        "Export for Inspect needs the inspect extra: run uv sync --extra inspect (add --extra "
        "hf if you use the Transformers backend), then try again."
    )
    monkeypatch.undo()
    monkeypatch.setattr(export_inspect, "INSPECT_AI", "0.0.1")
    with pytest.raises(ConfigError) as e:
        export_run(session, run_dir)
    assert str(e.value) == (
        "Export for Inspect is pinned to inspect_ai 0.0.1, and 0.3.291 is installed. Run uv "
        "sync --extra inspect to install the pinned version."
    )
    assert not (run_dir / "inspect").exists()


def test_the_pin_constant_is_the_extras_pin():
    import tomllib

    from tests.conftest import REPO

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extra = pyproject["project"]["optional-dependencies"]["inspect"]
    assert extra == [f"inspect-ai=={export_inspect.INSPECT_AI}"]


# --- I4: `clarity export … --format inspect` for a run

NOTE_LINE = (
    "Inspect didn't make these calls: running them again in Inspect won't send the same "
    "prompts. Each sample's metadata holds what the model got."
)


def export(root: Path, *args: str) -> int:
    return cli.main(["--root", str(root), "export", *args])


def test_clarity_export_takes_a_runs_id_its_folder_or_its_calls_jsonl(run, capsys):
    _, run_dir = run
    root, folder = run_dir.parent.parent, run_dir / "inspect"
    given = ["run-test", str(run_dir), str(run_dir / "calls.jsonl")]
    for n, target in enumerate(given):
        assert export(root, target, "--format", "inspect") == 0, capsys.readouterr().err
        replaced = "; replaced the earlier export" if n else ""
        assert capsys.readouterr().out.splitlines() == [
            f"Wrote 2 Inspect logs to {folder}{os.sep}: 1-chat.eval (4 calls{replaced}), "
            f"2-text.eval (4 calls{replaced}).",
            NOTE_LINE,
        ]
    assert sorted(p.name for p in folder.iterdir()) == ["1-chat.eval", "2-text.eval"]


def test_the_terminal_says_whose_text_the_logs_hold(run, capsys, monkeypatch):
    """A run's openings can't attach documents, but its profiles can search them."""
    _, run_dir = run
    real = export_inspect.export_run

    def found(session, run_dir):
        return {**real(session, run_dir), "found": True}

    monkeypatch.setattr(export_inspect, "export_run", found)
    assert export(run_dir.parent.parent, "run-test", "--format", "inspect") == 0
    assert capsys.readouterr().out.splitlines()[2:] == [
        "It holds the text of the parts search added from your documents."
    ]


def refused(root: Path, capsys, *args: str) -> str:
    assert export(root, *args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err.removeprefix("clarity: ").rstrip("\n")


def test_an_id_naming_both_a_conversation_and_a_run_is_refused(run, capsys):
    _, run_dir = run
    root = run_dir.parent.parent
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "run-test.jsonl").write_text("{}\n", encoding="utf-8")
    for fmt in ("inspect", "jsonl"):
        assert refused(root, capsys, "run-test", "--format", fmt) == (
            "run-test names a conversation (logs/run-test.jsonl) and a run (runs/run-test/). "
            "Pass the path of the one you mean."
        )
    assert not (run_dir / "inspect").exists()
    assert not (root / "logs" / "exports").exists()


def test_an_id_naming_neither_says_where_runs_are(fetched_root, capsys):
    for fmt in ("inspect", "jsonl"):
        assert refused(fetched_root, capsys, "no-such-id", "--format", fmt) == (
            "No conversation, log file or run no-such-id. `clarity chats` lists "
            "conversations; each run is a folder in runs/."
        )


def test_a_folder_without_calls_jsonl_is_no_run(run, capsys):
    """Checkpoint 1's S1: a folder was a run whatever it held, so the plain export called any
    folder a batch run, and the Inspect export failed with FileNotFoundError."""
    _, run_dir = run
    root = run_dir.parent.parent
    empty = root / "runs" / "half"
    empty.mkdir()
    for given in (str(empty), "half"):
        for fmt in ("inspect", "jsonl"):
            assert refused(root, capsys, given, "--format", fmt) == (
                f"No conversation, log file or run {given}. `clarity chats` lists "
                "conversations; each run is a folder in runs/."
            )
    assert not (empty / "inspect").exists()


def test_the_terminal_names_the_folder_in_full(run, capsys, monkeypatch):
    """A relative path given is printed resolved, so the line says where the logs are."""
    _, run_dir = run
    root = run_dir.parent.parent
    monkeypatch.chdir(root)
    assert export(root, "runs/run-test", "--format", "inspect") == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith(f"Wrote 2 Inspect logs to {(run_dir / 'inspect').resolve()}{os.sep}: ")


def test_format_inspect_takes_no_out(run, capsys, tmp_path):
    _, run_dir = run
    root = run_dir.parent.parent
    out = str(tmp_path / "x.eval")
    assert refused(root, capsys, "run-test", "--format", "inspect", "--out", out) == (
        "--format inspect takes no --out: a run's logs go in runs/RUN_ID/inspect/, and a "
        "chat's in logs/exports/."
    )
    assert not (run_dir / "inspect").exists()


def test_the_plain_export_of_a_run_names_format_inspect(run, capsys):
    _, run_dir = run
    root = run_dir.parent.parent
    for given in ("run-test", str(run_dir), str(run_dir / "calls.jsonl")):
        assert refused(root, capsys, given) == (
            f"{given} is a batch run, and the plain export takes one conversation. Export a "
            "run for Inspect with --format inspect."
        )
    assert not (root / "logs" / "exports").exists()


def test_a_conversation_for_inspect_waits_for_part_2(fetched_root, chat_tokenizer, capsys):
    """Removed in C1, when a chat exports for Inspect."""
    session = Session(
        fetched_root,
        backend=FakeBackend(None, n_ctx=512, tokenizer=chat_tokenizer),
        tokenizer_loader=lambda r, v: chat_tokenizer,
    )
    session.new_conversation("text")
    session.say("Made-up message.")
    assert session.conversation is not None
    cid = session.conversation.conversation_id
    for given in (cid, str(session.conversation.log.path)):
        assert refused(fetched_root, capsys, given, "--format", "inspect") == (
            f"{given} is a conversation. Export for Inspect takes only a run until v0.4.3's "
            "Part 2 adds chats."
        )
    assert not (fetched_root / "logs" / "exports").exists()


def test_resuming_a_runs_calls_jsonl_says_it_isnt_a_chat(run):
    session, run_dir = run
    with pytest.raises(ConfigError) as e:
        session.resume(run_dir / "calls.jsonl")
    assert str(e.value) == (
        f"{run_dir / 'calls.jsonl'} starts with run, not a conversation record. A batch run's "
        "calls.jsonl can't be resumed, or exported as a chat."
    )
