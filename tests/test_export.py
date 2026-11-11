"""Plain export: a conversation as one JSON Lines file a notebook can load (plan item C1)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from clarity_chat import cli
from clarity_chat.export import RECOMPUTE_NOTE, export_log
from clarity_chat.logs import list_chats, read_records
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE, build_tokenizer
from tests.fakes import FakeBackend

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
# A made-up probability record, as the Transformers backend logs one.
LOGPROBS = {
    "view": "raw logits, before any sampler",
    "n": 1,
    "tokens": [{"id": 5, "logprob": -0.5, "top": [[5, -0.5]]}],
}


@pytest.fixture
def make(fetched_root, tiny_tokenizer):
    def make(replies=None, backend=None):
        backend = backend or FakeBackend(replies, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        return session, backend

    return make


def read_export(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows[0], rows[1:]


def logged_calls(log: Path) -> dict[str, dict]:
    return {r["call_id"]: r for r in read_records(log) if r["type"] == "call"}


def test_every_exported_field_equals_the_logs_value(make):
    session, backend = make([" Hello!", " Again.", " Edited."])
    backend.logprobs = LOGPROBS
    session.new_conversation("text")
    session.say("Hi")
    session.regen()
    session.edit("Hi there")
    log = session.conversation.log.path
    result = export_log(session, log)
    header, rows = read_export(Path(result["path"]))
    assert header["type"] == "export" and result["calls"] == len(rows) == 3
    logged = logged_calls(log)
    for row in rows:
        call = logged[row["call_id"]]
        for key in ("turn_id", "parent_id", "regen_of", "prompt_ids", "output_ids",
                    "prompt_text", "output_text", "settings", "n_ctx", "stop_type", "flags",
                    "logprobs", "message_spans"):  # fmt: skip
            assert row[key] == call.get(key), key
        model = call["model"]
        assert row["model"]["repo"] == model["repo"]
        assert row["model"]["gguf_sha256"] == model["gguf_sha256"]
        assert row["path"][-1] == row["turn_id"] and row["path"][-2] == row["parent_id"]
    edited = rows[-1]
    assert edited["edit_of"] is not None


def test_failed_calls_are_counted_cancelled_ones_flagged_and_replays_left_out(make):
    session, backend = make([" Hello!", " Cut short here", " ok"])
    session.new_conversation("text")
    session.say("Hi")
    session.replay_current()
    backend.cancel_after = 3
    session.say("Again")
    backend.cancel_after = None
    backend.fail_with = "server gone"
    with pytest.raises(Exception, match="server gone"):
        session.say("Once more")
    result = export_log(session, session.conversation.log.path)
    header, rows = read_export(Path(result["path"]))
    assert header["failed_calls_left_out"] == 1
    assert [row["cancelled"] for row in rows] == [False, True]
    assert all(row["type"] == "call" for row in rows)  # no replay records


class Tagged(FakeBackend):
    """A fake whose server start names its machine, so two starts' facts differ."""

    def __init__(self, tag: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tag = tag

    def start(self, model, weights, ctx_size):
        record = super().start(model, weights, ctx_size)
        if record is not None:
            record["server_id"] += f"-{self.tag}"
            record["machine"] = {"chip": self.tag}
        return record

    def describe(self):
        return {"backend": self.name, "server_id": self.record and self.record["server_id"]}


def test_each_calls_facts_come_from_its_own_server_start(make, tiny_tokenizer):
    first, _ = make(backend=Tagged("one", replies=[" Hello!"], tokenizer=tiny_tokenizer))
    first.new_conversation("text")
    first.say("Hi")
    log = first.conversation.log.path
    second, _ = make(backend=Tagged("two", replies=[" Again."], tokenizer=tiny_tokenizer))
    second.resume(log)
    second.say("And again")
    header, rows = read_export(Path(export_log(second, log)["path"]))
    assert [row["facts"]["machine"] for row in rows] == [{"chip": "one"}, {"chip": "two"}]
    groups = header["conditions"]
    assert len(groups) == 2
    covered = [call_id for group in groups for call_id in group["call_ids"]]
    assert sorted(covered) == sorted(row["call_id"] for row in rows)  # each in exactly one


def test_marks_and_notes_are_the_current_ones(make):
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    session.annotate("mark", "keep")
    session.annotate("mark", "drop")
    session.annotate("unmark", "drop")
    session.annotate("note", "first thought")
    session.annotate("note", "second thought")
    _, rows = read_export(Path(export_log(session, session.conversation.log.path)["path"]))
    assert [m["label"] for m in rows[0]["marks"]] == ["keep"]
    assert [n["text"] for n in rows[0]["notes"]] == ["second thought"]


def without_spans(log: Path) -> None:
    """Rewrite a log as a Clarity before v0.2.1 would have: calls with no message_spans."""
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("message_spans", None)
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("golden", sorted(GOLDEN.glob("*.jsonl")), ids=lambda p: p.stem)
def test_older_calls_get_derived_spans_or_a_reason(make, golden, tmp_path):
    session, _ = make()
    log = Path(shutil.copy(golden, tmp_path / golden.name))
    turns = {r["turn_id"]: r for r in read_records(log) if r["type"] == "turn"}
    _, rows = read_export(Path(export_log(session, log)["path"]))
    assert rows
    for row in rows:
        spans = row["message_spans"]
        if isinstance(spans, dict):
            assert isinstance(spans["unavailable"], str) and spans["unavailable"]
            continue
        for span in spans:
            if span["kind"] == "message" and span.get("turn_id"):
                start, end = span["chars"]
                text = row["prompt_text"][start:end]
                assert text == turns[span["turn_id"]]["text"] or span.get("trimmed")


def test_derived_spans_match_the_logged_ones(make):
    session, _ = make([" Hello!", " Again."])
    session.new_conversation("text")
    session.say("Hi")
    session.say("More")
    log = session.conversation.log.path
    logged = [c["message_spans"] for c in logged_calls(log).values()]
    without_spans(log)
    _, rows = read_export(Path(export_log(session, log)["path"]))
    assert [row["message_spans"] for row in rows] == logged


def test_a_template_with_a_date_exported_on_a_later_day_has_no_spans(fetched_root, tmp_path):
    template = (
        "Today is {{ strftime_now('%Y-%m-%d') }}.\n"
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}</s>{% endfor %}<assistant>"
    )
    tokenizer = build_tokenizer(tmp_path / "toy", None, False, template)
    (fetched_root / "profiles" / "dated.toml").write_text(CHAT_PROFILE, encoding="utf-8")
    session = Session(
        fetched_root,
        backend=FakeBackend([" Hello!"], tokenizer=tokenizer),
        tokenizer_loader=lambda r, v: tokenizer,
    )
    session.new_conversation("dated")
    session.say("Hi")
    assert session.conversation is not None
    log = session.conversation.log.path
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    for row in rows:  # as if logged on 2001-01-01 by a Clarity that logged no spans
        if row["type"] == "call":
            row.pop("message_spans")
            text = "Today is 2001-01-01." + row["prompt_text"].split(".", 1)[1]
            row["prompt_text"], row["prompt_ids"] = text, tokenizer.encode(text)
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _, out = read_export(Path(export_log(session, log)["path"]))
    assert set(out[0]["message_spans"]) == {"unavailable"}


def test_text_mode_priming_is_one_span_with_no_role(make, fetched_root):
    text = (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8")
    primed = text + 'priming = "User: An earlier line.\\nAssistant: An earlier reply.\\n"\n'
    (fetched_root / "profiles" / "primed.toml").write_text(primed, encoding="utf-8")
    session, _ = make([" Hello!"])
    session.new_conversation("primed")
    session.say("Hi")
    _, rows = read_export(Path(export_log(session, session.conversation.log.path)["path"]))
    priming = [s for s in rows[0]["message_spans"] if s.get("source") == "priming"]
    assert len(priming) == 1 and "role" not in priming[0]


def test_the_header_says_what_it_holds_and_how_to_check_it(make):
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    header, _ = read_export(Path(export_log(session, session.conversation.log.path)["path"]))
    assert header["format_version"] == 1
    assert header["conversation_id"] == session.conversation.conversation_id
    assert header["note"] == RECOMPUTE_NOTE
    assert "clarity_chat.recompute.check" in RECOMPUTE_NOTE
    assert {"version", "commit"} <= set(header["clarity"])


def test_a_second_export_replaces_the_file_and_says_so(make, fetched_root, capsys):
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    conversation_id = session.conversation.conversation_id
    first = export_log(session, session.conversation.log.path)
    path = fetched_root / "logs" / "exports" / f"{conversation_id}.jsonl"
    assert Path(first["path"]) == path and not first["replaced"]
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600  # as private as the log
    root = ["--root", str(fetched_root)]
    assert cli.main([*root, "export", conversation_id]) == 0
    out = capsys.readouterr().out
    assert str(path) in out and "replaced" in out
    assert [c["conversation_id"] for c in list_chats(fetched_root / "logs")] == [conversation_id]


def test_slash_export_and_clarity_export_write_the_same_file(make, fetched_root, tmp_path):
    from clarity_chat.commands import Commands

    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    by_command = tmp_path / "a.jsonl"
    result = Commands(session).run("export", str(by_command), lambda e: None)
    assert Path(result["path"]) == by_command
    by_cli = tmp_path / "b.jsonl"
    log = str(session.conversation.log.path)
    assert cli.main(["--root", str(fetched_root), "export", log, "--out", str(by_cli)]) == 0

    def body(path):  # the header holds no time, so the files match line for line
        return path.read_text(encoding="utf-8")

    assert body(by_command) == body(by_cli)


def test_exporting_400_exchanges_with_derived_spans_stays_quick(make):
    session, backend = make()
    backend.replies = [" ok"] * 400
    session.new_conversation("text")
    for i in range(400):
        session.say(f"Message {i}")
    log = session.conversation.log.path
    without_spans(log)
    started = time.monotonic()
    result = export_log(session, log)
    assert result["calls"] == 400
    assert time.monotonic() - started < 60  # loose: catches a blow-up, not a slowdown
