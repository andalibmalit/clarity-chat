"""Export for Inspect, Part 2 (v0.4.3 C1): a chat as one Inspect log, a sample per reply
(G0's V5), read back with inspect_ai's own reader. Every text is made up."""

from __future__ import annotations

import io
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
from tests.test_export_inspect import mkdir_on_a_file

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


def test_a_chats_log_asks_the_viewer_to_show_its_text_as_written(make):
    """Not as Markdown, as for a run's logs (I4, the fact-check's section 5)."""
    session = make()
    log = two_messages_regens_and_an_edit(session)
    viewer = read(Path(export_chat(session, log)["path"])).eval.viewer
    assert viewer is not None and viewer.trust_content is False, viewer


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


def test_an_exports_entry_that_is_a_file_is_refused(make):
    """mkdir's error was a traceback (the fix review's C-b)."""
    session = make()
    session.new_conversation("chat")
    session.say("Hi")
    exports = session.logs_dir / "exports"
    exports.write_text("not a folder", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        export_chat(session, session.conversation.log.path)
    said = mkdir_on_a_file(exports)
    assert str(e.value) == f"Can't write the export to {exports}: {said}."


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


NOTE_LINE = (
    "Inspect didn't make these calls: running them again in Inspect won't send the same "
    "prompts. Each sample's metadata holds what the model got."
)


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
        # No recompute check: it reads a plain export, not an .eval (the whole-branch M1).
        assert capsys.readouterr().out.splitlines() == [
            f"Wrote {path}{replaced}: 1 call(s).",
            NOTE_LINE,
        ]


# C2: /export inspect, through the command table: the terminal chat, --json and the page.

REFUSED_ARGS = (
    "/export takes no path. /export inspect writes the conversation for Inspect, to "
    "logs/exports/ID.eval."
)


def test_slash_export_inspect_writes_the_chats_log_and_says_its_format(make):
    from clarity_chat.commands import Commands

    session = make()
    session.new_conversation("chat")
    session.say("The kettle sang twice.")
    log = session.conversation.log.path
    result = Commands(session).run("export", "inspect", lambda e: None)
    assert result["format"] == "inspect"
    assert result["path"] == str(session.logs_dir / "exports" / f"{log.stem}.eval")
    assert len(read(result["path"]).samples) == 1
    # What `clarity export ID --format inspect` returns, `format` included (S3).
    assert result == {**export_chat(session, log), "replaced": False}
    plain = Commands(session).run("export", "", lambda e: None)
    assert plain["format"] == "jsonl" and plain["path"].endswith(f"{log.stem}.jsonl")


@pytest.mark.parametrize("args", ["notes.jsonl", "inspect now", "INSPECT", "jsonl"])
def test_slash_export_takes_inspect_and_nothing_else(make, tmp_path, args):
    from clarity_chat.commands import Commands

    session = make()
    session.new_conversation("chat")
    session.say("Hi")
    with pytest.raises(ConfigError) as e:
        Commands(session).run("export", args, lambda e: None)
    assert str(e.value) == REFUSED_ARGS
    assert not (session.logs_dir / "exports").exists()


def test_the_terminal_chat_says_where_the_log_went_and_whose_calls_they_are(make):
    from clarity_chat.cli import ChatLoop

    session = make()
    session.new_conversation("chat")
    output: list[str] = []
    loop = ChatLoop(session, "chat", read=lambda d: "", write=output.append)
    loop.handle("Hi")
    output.clear()
    loop.handle("/export inspect")
    path = session.logs_dir / "exports" / f"{session.conversation.log.path.stem}.eval"
    assert "".join(output).splitlines() == [f"Wrote {path}: 1 call(s).", NOTE_LINE]
    output.clear()
    loop.handle("/export")  # the plain export's line: the recompute check, and no note
    assert "".join(output).splitlines() == [
        f"Wrote {path.with_suffix('.jsonl')}: 1 call(s). Before recomputing it in a notebook, "
        "run clarity_chat.recompute.check (docs/guides.md)."
    ]
    output.clear()
    loop.handle("/export notes.jsonl")
    assert "".join(output).strip() == f"! {REFUSED_ARGS}"


def test_json_chat_exports_for_inspect(make):
    from clarity_chat.json_chat import JsonChat

    session = make()
    stdout = io.StringIO()
    stdin = io.BytesIO(
        "".join(
            json.dumps(c) + "\n"
            for c in (
                {"command": "say", "args": "Hi"},
                {"command": "export", "args": "inspect", "id": "e"},
                {"command": "export", "args": "elsewhere", "id": "x"},
            )
        ).encode()
    )
    JsonChat(session, stdin, stdout).run([{"command": "profile", "args": "chat"}])
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    exported = next(e for e in events if e.get("id") == "e")
    assert exported["event"] == "result" and exported["result"]["format"] == "inspect"
    assert read(exported["result"]["path"]).samples
    refused = next(e for e in events if e.get("id") == "x")
    assert (refused["event"], refused["message"]) == ("error", REFUSED_ARGS)


def test_the_page_serves_the_newest_export_whichever_format(fetched_root, chat_tokenizer):
    """Through POST /command, in `clarity ui`'s worker thread: inspect_ai's writer runs its
    own event loop there too. GET /export serves the file under its own name."""
    from starlette.testclient import TestClient

    from clarity_chat import web
    from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command

    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        command(client, "say", "Hi")
        answer = command(client, "export", "inspect").json()
        assert answer.get("result", {}).get("format") == "inspect", answer
        assert session.conversation is not None
        stem = session.conversation.log.path.stem
        served = client.get("/export", headers={**AUTH, **SAME})
        assert served.status_code == 200
        eval_path = session.logs_dir / "exports" / f"{stem}.eval"
        assert served.content == eval_path.read_bytes()
        assert f'filename="{stem}.eval"' in served.headers["content-disposition"]
        assert served.headers["content-type"] == "application/zip"
        command(client, "export")
        served = client.get("/export", headers={**AUTH, **SAME})
        jsonl = session.logs_dir / "exports" / f"{stem}.jsonl"
        assert served.content == jsonl.read_bytes()
        assert f'filename="{stem}.jsonl"' in served.headers["content-disposition"]
        assert served.headers["content-type"].startswith("application/x-ndjson")


# The whole-branch review's S1: the documents and steering lines (state 18), and C2: the IDs.


@pytest.mark.parametrize(
    ("attach", "search", "said"),
    [
        (False, False, ""),
        (True, False, " It holds the text of the documents attached in this chat."),
        (False, True, " It holds the text of the parts search added from your documents."),
        (True, True, " It holds the text of the documents attached in this chat and of the "
         "parts search added from your documents."),
    ],
)  # fmt: skip
def test_a_chats_inspect_log_says_whose_text_it_holds(
    fetched_root, chat_tokenizer, tmp_path, attach, search, said
):
    """The privacy line for a file someone may share, as the plain export's says it."""
    from clarity_chat.documents import library_add, reading
    from tests.test_documents_board import chat, quiet
    from tests.test_search_b3 import GLOSSARY

    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    session, c, _ = chat(fetched_root, chat_tokenizer, 'search = ["glossary"]' if search else "")
    if attach:
        notes = tmp_path / "notes.md"
        notes.write_text("Made-up notes.\n", encoding="utf-8")
        c.run("attach", str(notes), quiet)
    c.run("say", "Tell me about the wren", quiet)
    result = c.run("export", "inspect", quiet)
    assert (result["attachments"], result["found"]) == (attach, search)
    assert cli.format_export(result).splitlines() == [
        f"Wrote {result['path']}: 1 call(s).{said}",
        NOTE_LINE,
    ]


def test_a_chats_inspect_log_names_the_vector_that_steered_its_calls(root, tiny_tokenizer):
    """The export sheet (v0.4.4) names the vector from `steered`, as for the plain export."""
    from clarity_chat.export import export_log
    from tests.test_steering import steer_root, steered

    steer_root(root)
    backend = FakeBackend([" Hi."] * 4, tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation(steered(root))
    first = session.say("Hi").record
    session.set_setting("steering_scale", "0")
    session.say("Again")
    assert session.conversation is not None
    log = session.conversation.log.path
    result = export_chat(session, log)
    logged = first["request"]["steering"]
    assert result["steered"] == [
        {"file": logged["file"], "sha256": logged["sha256"], "call_ids": [first["call_id"]]}
    ]
    assert result["steered"] == export_log(session, log)["steered"]


def test_a_chats_ids_hash_its_conversation_id_not_its_file_name(make):
    """So a copied log, which keeps its conversation ID, reads back with the original's IDs.
    Place 1: a chat is one log, as a run's first profile is."""
    import hashlib

    session = make()
    session.new_conversation("chat")
    session.say("Hi")
    assert session.conversation is not None
    original = session.conversation.log.path
    copy = Path(shutil.copy(original, original.with_name("copy.jsonl")))
    conversation_id = session.conversation.conversation_id

    def hashed(kind: str) -> str:
        return hashlib.sha256(f"{conversation_id} 1 {kind}".encode()).hexdigest()[:22]

    for log in (original, copy):
        spec = read(export_chat(session, log)["path"]).eval
        assert (spec.eval_id, spec.task_id) == (hashed("eval"), hashed("task"))


# --- X2 (v0.4.4): the template's own text before the first message -------------------------


def bare_chat(make, root: Path, tokenizer) -> tuple[Session, Path]:
    """A chat on a profile with no system message, its tokenizer's template, one message."""
    from tests.conftest import CHAT_PROFILE

    text = CHAT_PROFILE.replace('system = "Be brief."\n', "")
    (root / "profiles" / "bare.toml").write_text(text, encoding="utf-8", newline="\n")
    session = Session(
        root,
        backend=FakeBackend(None, n_ctx=4096, tokenizer=tokenizer),
        tokenizer_loader=lambda r, v: tokenizer,
    )
    session.new_conversation("bare")
    session.say("The kettle sang twice.")
    assert session.conversation is not None
    return session, session.conversation.log.path


def test_a_template_s_own_system_message_is_a_system_message(make, fetched_root, olmo_tokenizer):
    """OLMo 3's template writes a whole system turn when the profile has none: the export
    shows exactly the text before the first message, markers included, as the template's."""
    session, log = bare_chat(make, fetched_root, olmo_tokenizer)
    export_chat(session, log)
    [sample] = read(session.logs_dir / "exports" / f"{log.stem}.eval").samples
    row = sample.metadata
    [first_message] = [s for s in row["message_spans"] if s["kind"] == "message"]
    before = row["prompt_text"][: first_message["chars"][0]]
    assert "<|im_start|>system\n" in before and "<|im_end|>" in before  # a whole turn
    assert [m.role for m in sample.messages] == ["system", "user", "assistant"]
    system, user, _ = sample.messages
    assert (system.role, system.text) == ("system", before)
    assert system.metadata == {"clarity": row["message_spans"][0]}
    assert row["message_spans"][0]["kind"] == "template"  # the template wrote it
    assert (user.role, user.text) == ("user", "The kettle sang twice.")


def test_a_template_with_no_default_shows_its_header_as_written(make, fetched_root, chat_tokenizer):
    """The toy template writes only the first message's header: the box holds that, exact.
    (X2 taken literally; the main session rules on this case.)"""
    session, log = bare_chat(make, fetched_root, chat_tokenizer)
    export_chat(session, log)
    [sample] = read(session.logs_dir / "exports" / f"{log.stem}.eval").samples
    assert [m.role for m in sample.messages] == ["system", "user", "assistant"]
    system, user, _ = sample.messages
    assert (system.role, system.text) == ("system", "<s><user>")
    assert system.metadata["clarity"]["kind"] == "template"
    assert user.text == "The kettle sang twice."


def test_a_call_with_its_own_system_message_gets_no_other(make):
    session = make()
    log = two_messages_regens_and_an_edit(session)
    export_chat(session, log)
    for sample in read(session.logs_dir / "exports" / f"{log.stem}.eval").samples:
        roles = [m.role for m in sample.messages]
        assert roles.count("system") == 1 and sample.messages[0].text == "Be brief."
