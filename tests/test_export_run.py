"""Reading a batch run for export (v0.4.3 I2): each of its conversations gives exactly the
rows the plain export gives for that conversation alone, each with the call's `run`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from clarity_chat import cli
from clarity_chat.export import export_log, run_conversations
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.test_batch import batch  # noqa: F401 (a fixture)

Records = list[dict[str, Any]]


def session_for(run_dir: Path, backend, tokenizer) -> Session:
    root = run_dir.parent.parent
    return Session(root, backend=backend, logs_dir=run_dir, tokenizer_loader=lambda r, v: tokenizer)


def write_records(path: Path, records: Records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def alone(records: Records, conversation_id: str) -> Records:
    """One conversation of a run as a chat log of its own: its records, with the run's
    server starts after its `conversation` record."""
    own = [r for r in records if r.get("conversation_id") == conversation_id]
    servers = [r for r in records if r["type"] == "server_start" and r not in own]
    return own[:1] + servers + own[1:]


def plain(session: Session, records: Records, conversation_id: str, folder: Path) -> Records:
    """What the plain export writes for that conversation alone: header, then rows."""
    log = write_records(folder / f"{conversation_id}.jsonl", alone(records, conversation_id))
    out = folder / f"{conversation_id}-export.jsonl"
    export_log(session, log, out)
    return list(read_records(out))


def check_against_plain(session: Session, run_dir: Path, folder: Path) -> list[Any]:
    """Every conversation's header and rows equal the plain export's; the rows add `run`."""
    records = list(read_records(run_dir / "calls.jsonl"))
    run, conversations = run_conversations(session, run_dir)
    assert run == records[0]
    calls = {r["call_id"]: r for r in records if r["type"] == "call"}
    assert [c.header["conversation_id"] for c in conversations] == list(
        dict.fromkeys(r["conversation_id"] for r in records if r["type"] == "conversation")
    )
    for conversation in conversations:
        header, *rows = plain(session, records, conversation.header["conversation_id"], folder)
        assert conversation.header == header
        assert [{k: v for k, v in r.items() if k != "run"} for r in conversation.rows] == rows
        for row in conversation.rows:
            assert row["run"] == calls[row["call_id"]]["run"]
        own = [r for r in records if r.get("conversation_id") == header["conversation_id"]]
        assert conversation.records == [r for r in own if r["type"] != "server_start"]
    return conversations


def test_a_runs_conversations_give_the_plain_exports_rows_with_run(
    batch,  # noqa: F811
    tiny_tokenizer,
    tmp_path,
):
    _, run_dir, backend, _ = batch(seeds=2)
    log = run_dir / "calls.jsonl"
    calls = [r for r in read_records(log) if r["type"] == "call"]
    assert cli.main(["mark", str(log), calls[1]["call_id"], "landed"]) == 0
    assert cli.main(["note", str(log), calls[2]["call_id"], "made-up review note"]) == 0
    session = session_for(run_dir, backend, tiny_tokenizer)
    conversations = check_against_plain(session, run_dir, tmp_path)
    rows = [row for c in conversations for row in c.rows]
    assert len(rows) == 8  # two profiles, two openings, two seeds
    assert sorted((r["run"]["opening"], r["settings"]["seed"]["value"]) for r in rows) == [
        (1, 42), (1, 42), (1, 43), (1, 43), (2, 42), (2, 42), (2, 43), (2, 43),
    ]  # fmt: skip
    by_id = {r["call_id"]: r for r in rows}
    assert [m["label"] for m in by_id[calls[1]["call_id"]]["marks"]] == ["landed"]
    assert [n["text"] for n in by_id[calls[2]["call_id"]]["notes"]] == ["made-up review note"]


def drop_spans(records: Records) -> None:  # before v0.2.1
    for r in records:
        if r["type"] == "call":
            del r["message_spans"]


def drop_mark_ids(records: Records) -> None:  # before v0.2.0
    for r in records:
        if r["type"] in ("mark", "unmark"):
            r.pop("mark_id", None)


def drop_steering(records: Records) -> None:  # before v0.2.2
    for r in records:
        if r["type"] == "call":
            r["request"].pop("steering", None)


def drop_documents(records: Records) -> None:  # before v0.3.2
    records[0].pop("documents", None)


@pytest.mark.parametrize("edit", [drop_spans, drop_mark_ids, drop_steering, drop_documents])
def test_an_older_runs_records_export_as_the_plain_export_reads_them(
    batch,  # noqa: F811
    tiny_tokenizer,
    tmp_path,
    edit: Callable[[Records], None],
):
    _, run_dir, backend, _ = batch(seeds=2)
    log = run_dir / "calls.jsonl"
    call = [r for r in read_records(log) if r["type"] == "call"][0]
    assert cli.main(["mark", str(log), call["call_id"], "landed"]) == 0
    records = list(read_records(log))
    edit(records)
    write_records(log, records)
    session = session_for(run_dir, backend, tiny_tokenizer)
    conversations = check_against_plain(session, run_dir, tmp_path)
    rows = {row["call_id"]: row for c in conversations for row in c.rows}
    assert [m["label"] for m in rows[call["call_id"]]["marks"]] == ["landed"]
    if edit is drop_spans:  # derived from the logged profile and tokenizer, as resume does
        assert all(isinstance(row["message_spans"], list) for row in rows.values())
    if edit is drop_steering:
        assert {row["steering"] for row in rows.values()} == {"none"}


def test_a_run_without_key_json_is_refused(batch, tiny_tokenizer):  # noqa: F811
    _, run_dir, backend, _ = batch()
    (run_dir / "key.json").unlink()
    with pytest.raises(ConfigError) as e:
        run_conversations(session_for(run_dir, backend, tiny_tokenizer), run_dir)
    assert str(e.value) == (
        "Run run-test stopped before it finished (it has no key.json), so it isn't exported."
    )


def test_a_stopped_runs_partial_calls_are_refused(batch, fetched_root, tiny_tokenizer):  # noqa: F811
    def setup(backend):
        start = backend.start

        def interrupt_after_first_call(model, weights, ctx_size):
            if backend.calls:
                raise KeyboardInterrupt
            return start(model, weights, ctx_size)

        backend.start = interrupt_after_first_call

    with pytest.raises(Exception, match="Interrupted"):
        batch(setup=setup)
    run_dir = fetched_root / "runs" / "run-test"
    assert any(r["type"] == "call" for r in read_records(run_dir / "calls.jsonl"))
    with pytest.raises(ConfigError, match="stopped before it finished"):
        run_conversations(session_for(run_dir, None, tiny_tokenizer), run_dir)


def test_a_folder_whose_calls_jsonl_isnt_a_runs_is_refused(batch, tiny_tokenizer):  # noqa: F811
    _, run_dir, backend, _ = batch()
    log = run_dir / "calls.jsonl"
    records = list(read_records(log))
    first = next(r["conversation_id"] for r in records if r["type"] == "conversation")
    write_records(log, alone(records, first))  # a chat log where the run's was
    with pytest.raises(ConfigError) as e:
        run_conversations(session_for(run_dir, backend, tiny_tokenizer), run_dir)
    assert str(e.value) == (
        f"Can't read the records of {log} (ValueError: it starts with a conversation record, "
        "not a run record). A log another program wrote or edited can cause this; if not, "
        "it's a bug in Clarity."
    )


def test_a_torn_line_in_a_runs_log_is_left_out_as_the_chat_log_reader_leaves_it(
    batch,  # noqa: F811
    tiny_tokenizer,
):
    _, run_dir, backend, _ = batch()
    log = run_dir / "calls.jsonl"
    call = [r for r in read_records(log) if r["type"] == "call"][0]
    with log.open("ab") as f:
        f.write(b'{"type": "no')  # cut off: the mark below ends it with a torn_line record
    assert cli.main(["mark", str(log), call["call_id"], "landed"]) == 0
    assert any(r["type"] == "torn_line" for r in read_records(log))
    _, conversations = run_conversations(session_for(run_dir, backend, tiny_tokenizer), run_dir)
    rows = {row["call_id"]: row for c in conversations for row in c.rows}
    assert [m["label"] for m in rows[call["call_id"]]["marks"]] == ["landed"]
