"""Export for Inspect, Part 2 (v0.4.3 C1): a chat as one Inspect log, a sample per reply
(G0's V5), read back with inspect_ai's own reader. Every text is made up."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from clarity_chat import cli
from clarity_chat.export import conversation_rows
from clarity_chat.export_inspect import INSPECT_FORMAT_VERSION, INSPECT_NOTE, export_chat
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_export import Tagged

pytest.importorskip("inspect_ai")

GOLDEN = sorted((Path(__file__).parent / "fixtures" / "golden").glob("*.jsonl"))


@pytest.fixture
def make(fetched_root, chat_tokenizer):
    def make(backend=None) -> Session:
        return Session(
            fetched_root,
            backend=backend or FakeBackend(None, n_ctx=512, tokenizer=chat_tokenizer),
            tokenizer_loader=lambda r, v: chat_tokenizer,
        )

    return make


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def read(path: Path | str):
    from inspect_ai.log import read_eval_log

    assert Path(path).is_file(), f"the export wrote no {path}"
    return read_eval_log(str(path))


def records_of(log: Path) -> list[dict]:
    return list(read_records(log))


def two_messages_regens_and_an_edit(session: Session) -> Path:
    """Two messages, two /regens of the first, and an /edit of the second."""
    session.new_conversation("chat")
    session.say("The kettle sang twice.")
    session.regen()
    session.regen()
    session.say("Ask me about otters.")
    session.edit("Ask me about herons.")
    assert session.conversation is not None
    return session.conversation.log.path


def test_a_chat_is_one_log_with_a_sample_per_reply(make):
    session = make()
    log = two_messages_regens_and_an_edit(session)
    result = export_chat(session, log)
    path = session.logs_dir / "exports" / f"{log.stem}.eval"
    assert result["path"] == str(path) and not result["replaced"] and result["calls"] == 5
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # no temporary left
    users = [r["turn_id"] for r in records_of(log) if r["type"] == "turn" and r["role"] == "user"]
    first, second, edited = users
    samples = read(path).samples
    # Each reply its own sample, epoch 1: uneven epochs read as missing replies (V5).
    assert sorted((s.id, s.epoch) for s in samples) == sorted(
        [
            (f"{first}-1", 1),
            (f"{first}-2", 1),
            (f"{first}-3", 1),
            (f"{second}-1", 1),
            (f"{edited}-1", 1),
        ]
    )


def test_each_sample_is_its_calls_row_and_runs_from_the_start_to_its_reply(make):
    session = make()
    log = two_messages_regens_and_an_edit(session)
    export_chat(session, log)
    records = records_of(log)
    _, rows = conversation_rows(session, records, str(log))
    turns = {r["turn_id"]: r for r in records if r["type"] == "turn"}
    samples = {
        s.metadata["call_id"]: s for s in read(Path(export_chat(session, log)["path"])).samples
    }
    assert len(samples) == len(rows) == 5
    for row in rows:
        sample = samples[row["call_id"]]
        assert sample.metadata == row  # the plain export's row, with no `run`
        assert sample.input == turns[row["parent_id"]]["text"]
        *asked, reply = sample.messages
        spans = [s for s in row["message_spans"] if s["kind"] == "message"]
        assert [(m.role, m.text) for m in asked] == [
            (s["role"], row["prompt_text"][s["chars"][0] : s["chars"][1]]) for s in spans
        ]
        # From the start: every turn on the reply's path, in order.
        assert [s["turn_id"] for s in spans if s.get("turn_id")] == row["path"][:-1]
        assert (reply.role, reply.text) == ("assistant", row["output_text"])
        assert sample.output.choices[0].stop_reason == "stop"


def test_the_log_is_the_chats_with_its_header_title_and_times(make):
    session = make()
    log = two_messages_regens_and_an_edit(session)
    session.set_title("Kettles and herons")
    path = Path(export_chat(session, log)["path"])
    records = records_of(log)
    header, rows = conversation_rows(session, records, str(log))
    conversation = records[0]
    calls = {r["call_id"]: r for r in records if r["type"] == "call"}
    spec = read(path).eval
    assert spec.task == "clarity-chat"
    assert spec.run_id == conversation["conversation_id"]
    assert spec.model == "clarity/chat"
    assert spec.tags == ["clarity", "not-run-by-inspect"]
    assert spec.config.epochs == 1
    assert spec.dataset.name == "Kettles and herons"
    assert spec.dataset.samples == 5
    assert spec.metadata == {
        "inspect_note": INSPECT_NOTE,
        "clarity": {**header, "inspect_format_version": INSPECT_FORMAT_VERSION},
    }
    stored = read(path)  # Inspect writes its times with microseconds
    assert utc(stored.eval.created) == utc(conversation["time"])
    assert utc(stored.stats.started_at) == utc(calls[rows[0]["call_id"]]["time"])
    assert utc(stored.stats.completed_at) == utc(calls[rows[-1]["call_id"]]["time"])


def test_two_exports_read_back_equal_and_the_second_replaces(make, tmp_path):
    session = make()
    log = two_messages_regens_and_an_edit(session)
    first = Path(export_chat(session, log)["path"])
    before = read(first)
    result = export_chat(session, log)
    assert result["replaced"] and result["path"] == str(first)
    assert sorted(p.name for p in first.parent.iterdir()) == [first.name]
    assert read(first).model_dump() == before.model_dump()  # ids and uuids too
    ids = [s.uuid for s in before.samples] + [m.id for s in before.samples for m in s.messages]
    assert len(set(ids)) == len(ids)


def test_an_untitled_chat_takes_its_first_words_and_one_without_messages_its_id(make):
    session = make()
    session.new_conversation("chat")
    session.say("The kettle sang twice.")
    log = session.conversation.log.path
    assert read(export_chat(session, log)["path"]).eval.dataset.name == "The kettle sang twice."
    session.new_conversation("chat")
    empty = session.conversation.log.path
    exported = read(export_chat(session, empty)["path"])
    assert exported.eval.dataset.name == session.conversation.conversation_id
    assert exported.samples == []


def test_a_cancelled_reply_is_unknown_and_failed_and_torn_calls_are_counted(make, chat_tokenizer):
    backend = FakeBackend([" Hello!", " Cut short here", " ok", " Torn."], tokenizer=chat_tokenizer)
    session = make(backend)
    session.new_conversation("chat")
    session.say("Hi")
    backend.cancel_after = 3
    session.say("Again")
    backend.cancel_after = None
    backend.fail_with = "server gone"
    with pytest.raises(Exception, match="server gone"):
        session.say("Once more")
    backend.fail_with = None
    session.say("And once more")
    log = session.conversation.log.path
    raw = log.read_bytes().splitlines(keepends=True)
    assert json.loads(raw[-1])["type"] == "turn"
    log.write_bytes(b"".join(raw[:-1]) + raw[-1][:20])  # the last reply's turn line cut off
    result = export_chat(session, log)
    assert result["calls"] == 2
    assert result["failed_calls_left_out"] == 1
    assert result["calls_without_a_turn_left_out"] == 1
    exported = read(result["path"])
    header = exported.eval.metadata["clarity"]
    assert (header["failed_calls_left_out"], header["calls_without_a_turn_left_out"]) == (1, 1)
    reasons = {s.metadata["cancelled"]: s.output.choices[0].stop_reason for s in exported.samples}
    assert reasons == {False: "stop", True: "unknown"}


def test_a_chat_over_two_server_starts_keeps_both_condition_groups(make, chat_tokenizer):
    first = make(Tagged("one", replies=[" Hello!"], tokenizer=chat_tokenizer))
    first.new_conversation("chat")
    first.say("Hi")
    log = first.conversation.log.path
    second = make(Tagged("two", replies=[" Again."], tokenizer=chat_tokenizer))
    second.resume(log)
    second.say("And again")
    exported = read(export_chat(second, log)["path"])
    groups = exported.eval.metadata["clarity"]["conditions"]
    assert len(groups) == 2
    assert sorted(i for g in groups for i in g["call_ids"]) == sorted(
        s.metadata["call_id"] for s in exported.samples
    )
    assert sorted(s.metadata["facts"]["machine"]["chip"] for s in exported.samples) == [
        "one",
        "two",
    ]


def test_a_text_mode_chat_gives_one_whole_prompt_message(make):
    session = make()
    session.new_conversation("text")
    session.say("The kettle sang twice.")
    log = session.conversation.log.path
    [sample] = read(export_chat(session, log)["path"]).samples
    [message, reply] = sample.messages
    assert (message.role, message.text) == ("user", sample.metadata["prompt_text"])
    assert message.metadata == {
        "clarity": {"whole_prompt": "text mode: the template has no message roles"}
    }
    assert reply.text == sample.metadata["output_text"]


def test_a_log_name_clarity_wouldnt_write_is_refused(make, tmp_path):
    session = make()
    session.new_conversation("chat")
    session.say("Hi")
    copied = Path(shutil.copy(session.conversation.log.path, tmp_path / "a copy.jsonl"))
    with pytest.raises(ConfigError) as e:
        export_chat(session, copied)
    assert (
        str(e.value) == "a copy.jsonl isn't a log name Clarity writes, so it can't name the export."
    )
    assert not (session.logs_dir / "exports").exists()


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda p: p.stem)
def test_every_golden_log_exports_and_reads_back(
    fetched_root, tiny_tokenizer, chat_tokenizer, golden, tmp_path
):
    """With the tokenizer each was written with, as test_golden.py loads them."""
    first = json.loads(golden.read_text(encoding="utf-8").splitlines()[0])
    tokenizer = chat_tokenizer if 'mode = "chat"' in first["profile"]["toml"] else tiny_tokenizer
    session = Session(
        fetched_root, backend=FakeBackend(None), tokenizer_loader=lambda r, v: tokenizer
    )
    log = Path(shutil.copy(golden, tmp_path / golden.name))
    _, rows = conversation_rows(session, records_of(log), str(log))
    exported = read(export_chat(session, log)["path"])
    assert {s.metadata["call_id"]: s.metadata for s in exported.samples} == {
        row["call_id"]: row for row in rows
    }  # by call: nothing depends on the samples' order in the file (C3)
    assert len({(s.id, s.epoch) for s in exported.samples}) == len(rows)


def export(root: Path, *args: str) -> int:
    return cli.main(["--root", str(root), "export", *args])


def test_clarity_export_takes_a_conversations_id_or_its_log(make, capsys):
    session = make()
    session.new_conversation("chat")
    session.say("Hi")
    log = session.conversation.log.path
    path = session.logs_dir / "exports" / f"{log.stem}.eval"
    for given, replaced in ((log.stem, ""), (str(log), " (replaced the earlier export)")):
        assert export(session.root, given, "--format", "inspect") == 0
        assert capsys.readouterr().out.splitlines() == [
            f"Wrote {path}{replaced}: 1 call(s). Before recomputing it in a notebook, run "
            "clarity_chat.recompute.check (docs/guides.md).",
            "Inspect didn't make these calls: running them again in Inspect won't send the "
            "same prompts. Each sample's metadata holds what the model got.",
        ]
