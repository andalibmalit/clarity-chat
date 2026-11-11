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
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE, TOY_TEMPLATE, build_tokenizer
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
    lines = path.read_text(encoding="utf-8").split("\n")  # not splitlines: U+2028 is raw
    rows = [json.loads(line) for line in lines if line]
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


def test_a_call_whose_turn_line_was_cut_off_is_left_out_and_counted(make):
    """The call record is written before its reply's turn. A crash between them leaves a call
    with no turn; resume carries on without it, and so does the export."""
    from clarity_chat.logs import LogFile

    session, _ = make([" Hello!", " Again!"])
    session.new_conversation("text")
    session.say("Hi")
    session.say("More")
    log = session.conversation.log.path
    raw = log.read_bytes().splitlines(keepends=True)
    assert json.loads(raw[-1])["type"] == "turn"
    log.write_bytes(b"".join(raw[:-1]) + raw[-1][:20])  # the last reply's turn line cut off
    LogFile(log).append({"type": "title", "conversation_id": "x", "title": "after the crash"})
    result = export_log(session, log)
    header, rows = read_export(Path(result["path"]))
    assert len(rows) == 1 and header["calls_without_a_turn_left_out"] == 1
    assert result["calls_without_a_turn_left_out"] == 1
    assert header["failed_calls_left_out"] == 0


class Tagged(FakeBackend):
    """A fake whose server start names its machine, so two starts' facts differ."""

    def __init__(self, tag: str, machine: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tag, self.chip = tag, machine or tag
        self.extra: dict = {}  # other server_start fields

    def start(self, model, weights, ctx_size):
        record = super().start(model, weights, ctx_size)
        if record is not None:
            record["server_id"] += f"-{self.tag}"
            record["machine"] = {"chip": self.chip}
            record.update(self.extra)
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
        assert isinstance(spans, list), spans  # every golden call derives today
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
    primed = 'priming = "User: An earlier line.\\nAssistant: An earlier reply.\\n"\n' + text
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


def test_slash_export_and_clarity_export_write_the_same_file(make, fetched_root):
    from clarity_chat.commands import Commands

    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    result = Commands(session).run("export", "", lambda e: None)
    path = Path(result["path"])
    assert path.parent == fetched_root / "logs" / "exports"
    by_command = path.read_text(encoding="utf-8")  # the header holds no time
    log = str(session.conversation.log.path)
    assert cli.main(["--root", str(fetched_root), "export", log]) == 0
    assert path.read_text(encoding="utf-8") == by_command


def test_slash_export_takes_no_path(make, tmp_path):
    """Commands will also come from the browser (v0.3.0), so /export writes only to
    logs/exports/; `clarity export --out` writes elsewhere."""
    from clarity_chat.commands import Commands

    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ConfigError, match="takes no path"):
        Commands(session).run("export", str(elsewhere / "a.jsonl"), lambda e: None)
    assert list(elsewhere.iterdir()) == []


def test_exporting_400_exchanges_with_derived_spans_stays_quick(make, tiny_tokenizer):
    backend = FakeBackend([" ok"] * 400, n_ctx=100_000, tokenizer=tiny_tokenizer)
    session, _ = make(backend=backend)
    session.new_conversation("text")
    for i in range(400):
        session.say(f"Message {i}")
    log = session.conversation.log.path
    without_spans(log)
    started = time.monotonic()
    result = export_log(session, log)
    assert result["calls"] == 400
    assert time.monotonic() - started < 60  # loose: catches a blow-up, not a slowdown


def test_a_log_whose_conversation_id_is_a_path_is_refused(make, tmp_path):
    """A log is untrusted: its ID names the default file, so it mustn't leave logs/exports/."""
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    assert session.conversation is not None
    log = session.conversation.log.path
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    rows[0]["conversation_id"] = "../../escaped"
    edited = tmp_path / "edited.jsonl"
    edited.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ConfigError, match="conversation ID"):
        export_log(session, edited)
    assert not list(tmp_path.parent.glob("**/escaped.jsonl"))


def test_two_starts_that_differ_only_in_a_server_flag_are_two_groups(make, tiny_tokenizer):
    """Every fact replay compares splits the conditions, not only machine and build."""
    first, one = make(backend=Tagged("one", "m", replies=[" Hello!"], tokenizer=tiny_tokenizer))
    one.extra = {"flags": {"flash-attn": "on"}}
    first.new_conversation("text")
    first.say("Hi")
    assert first.conversation is not None
    log = first.conversation.log.path
    second, two = make(backend=Tagged("two", "m", replies=[" Again."], tokenizer=tiny_tokenizer))
    two.extra = {"flags": {"flash-attn": "off"}}
    second.resume(log)
    second.say("And again")
    header, _ = read_export(Path(export_log(second, log)["path"]))
    assert len(header["conditions"]) == 2


def test_a_path_that_cant_be_written_is_a_clear_error_and_leaves_no_temp_file(make, tmp_path):
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    assert session.conversation is not None
    target = tmp_path / "a-folder"
    target.mkdir()
    with pytest.raises(ConfigError, match="a-folder"):
        export_log(session, session.conversation.log.path, target)
    assert not list(tmp_path.glob(".a-folder.*.tmp"))  # it held the whole conversation


def test_an_unknown_conversation_is_a_clear_error(fetched_root, capsys):
    assert cli.main(["--root", str(fetched_root), "export", "no-such-id"]) == 1
    assert capsys.readouterr().err.startswith("clarity: ")


def test_a_log_with_a_call_missing_a_field_is_refused(make, tmp_path):
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    assert session.conversation is not None
    rows = [
        json.loads(line)
        for line in session.conversation.log.path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["type"] == "call":
            del row["model"]
    edited = tmp_path / "edited.jsonl"
    edited.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ConfigError, match="can't be exported"):
        export_log(session, edited)


def test_a_turn_whose_parent_comes_later_is_refused_not_looped_on(make, tmp_path):
    """A parent_id cycle in an edited log would make following the path loop for ever."""
    session, _ = make([" Hello!"])
    session.new_conversation("text")
    session.say("Hi")
    assert session.conversation is not None
    rows = [
        json.loads(line)
        for line in session.conversation.log.path.read_text(encoding="utf-8").splitlines()
    ]
    turns = [r for r in rows if r["type"] == "turn"]
    turns[0]["parent_id"] = turns[-1]["turn_id"]
    edited = tmp_path / "edited.jsonl"
    edited.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ConfigError, match="parent"):
        export_log(session, edited)


def test_derived_spans_match_the_logged_ones_in_chat_mode(make, fetched_root, tmp_path):
    """A system message and priming messages: the labels `system` and `priming N`, and the
    chat template, all derived as they were logged."""
    chat = (fetched_root / "profiles" / "chat.toml").read_text(encoding="utf-8")
    priming = (
        'priming = [{ role = "user", content = "An earlier question." },'
        ' { role = "assistant", content = "An earlier answer." }]\n'
    )
    (fetched_root / "profiles" / "primed-chat.toml").write_text(priming + chat, encoding="utf-8")
    tokenizer = build_tokenizer(tmp_path / "toy", None, False, TOY_TEMPLATE)
    session = Session(
        fetched_root,
        backend=FakeBackend([" Hello!", " Again."], tokenizer=tokenizer),
        tokenizer_loader=lambda r, v: tokenizer,
    )
    session.new_conversation("primed-chat")
    session.say("Hi")
    session.say("More")
    assert session.conversation is not None
    log = session.conversation.log.path
    logged = [c["message_spans"] for c in logged_calls(log).values()]
    assert {s.get("source") for s in logged[0]} >= {"system", "priming 1", "priming 2"}
    without_spans(log)
    _, rows = read_export(Path(export_log(session, log)["path"]))
    assert [row["message_spans"] for row in rows] == logged


# Server records shaped like the real backends' (docs/reference.md, server_start).
LLAMA_SERVER = {
    "type": "server_start",
    "server_id": "server-1",
    "backend": "llama.cpp",
    "flags": {"ctx-size": 8192, "flash-attn": "on"},
    "llama_cpp": {"commit": "f" * 40, "dirty": False, "dir": "/home/me/clarity/vendor/llama.cpp"},
    "build_info": "b1-f477f79",
    "props": {"n_ctx": 8192, "media_marker": "<random>"},
    "model_file": {"eog_tokens": [{"id": 2, "text": "</s>"}, {"id": 0, "text": "<|e|>"}]},
    "graph_splits": {"prompt": 2, "reply": 2},
    "env_llama_ggml_kept": {},
    "machine": {"chip": "M2 Max"},
}
TRANSFORMERS_SERVER = {
    "type": "server_start",
    "server_id": "server-2",
    "backend": "transformers",
    "device": "mps",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "torch": "2.15.0",
    "transformers": "5.22.2",
    "eos_token_id": [2],
    "env_torch_kept": {},
    "deterministic_algorithms": False,
    "machine": {"chip": "M2 Max"},
}


@pytest.mark.parametrize("server", [LLAMA_SERVER, TRANSFORMERS_SERVER], ids=["llama", "hf"])
def test_conditions_from_real_shaped_server_records(server):
    from clarity_chat.export import conditions

    call = {
        "backend": {"backend": server["backend"], "server_id": server["server_id"]},
        "model": {"repo": "org/m", "revision": "a" * 40, "gguf_sha256": "0" * 64},
        "client": {"commit": "c"},
        "libraries": {"tokenizers": "0"},
    }
    found = conditions(call, server)
    assert found["backend"] == server["backend"]
    assert found["end_tokens"] == ([0, 2] if server["backend"] == "llama.cpp" else [2])
    assert "client" not in found and "libraries" not in found  # can't change the numbers
    assert None not in found.values(), found  # every fact read from its real field
    if server["backend"] == "llama.cpp":
        assert "dir" not in found["llama_cpp"] and "media_marker" not in found["props"]
        assert found["weights"]["gguf_sha256"] == "0" * 64
    else:
        assert found["attn_implementation"] == "sdpa" and found["dtype"] == "bfloat16"
