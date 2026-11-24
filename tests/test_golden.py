"""Golden logs: one small made-up log written by each released version's code, never edited.

Current code must still read, replay and annotate every one of them: replay is Clarity's
promise, and it covers logs written by older versions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clarity_chat.logs import SCHEMA_VERSION, read_records
from clarity_chat.session import Session, annotate_log
from tests.conftest import MODELS
from tests.fakes import FakeBackend

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
# Written by the fake before it took llama.cpp's shapes (end token, max_tokens).
BEFORE_V0_2_3 = {"v0.1.0", "v0.2.0", "v0.2.1", "v0.2.2"}
LOGS = sorted(GOLDEN.glob("*.jsonl"))


def test_the_golden_logs_are_found():
    assert GOLDEN / "v0.1.0.jsonl" in LOGS  # an empty glob would skip every test below


VECTOR = Path(__file__).parent / "fixtures" / "steering" / "smollm2-repeng.gguf"


@pytest.fixture
def steerable(fetched_root) -> Path:
    """The project v0.2.2's golden log was written in: the tiny model has Transformers weights
    (a config.json the repeng test vector fits) and the vector sits where the log says."""
    from clarity_chat.models import read_lock, write_lock_entry

    weights = fetched_root / "models" / "tiny-hf"
    weights.mkdir(parents=True)
    config = {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30}
    (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
    entry = read_lock(fetched_root)["models"]["tiny"]
    write_lock_entry(fetched_root, "tiny", {**entry, "source_dir": "models/tiny-hf"})
    # v0.4.0's log is on your own entry, `mine` (B4): tiny's pins, in local/models.toml.
    (fetched_root / "local").mkdir(exist_ok=True)  # the fetch made it (A10)
    (fetched_root / "local" / "models.toml").write_text(
        MODELS.replace("[tiny]", "[mine]"), encoding="utf-8"
    )
    write_lock_entry(fetched_root, "mine", {**entry, "name": "mine"})
    (fetched_root / "local" / "vectors").mkdir(parents=True)
    shutil.copy(VECTOR, fetched_root / "local" / "vectors" / VECTOR.name)
    return fetched_root


@pytest.fixture(params=LOGS, ids=lambda p: p.stem)
def golden(request, tmp_path) -> Path:
    """A copy, because replay and marks append to the log."""
    return Path(shutil.copy(request.param, tmp_path / request.param.name))


@pytest.fixture
def golden_tokenizer(golden, tiny_tokenizer, chat_tokenizer):
    """The tokenizer the log was written with: v0.3.1's chat-mode log (attachments need
    chat mode) used the tests' chat tokenizer, with the toy template; the others are in
    text mode, on the plain one."""
    first = json.loads(golden.read_text(encoding="utf-8").splitlines()[0])
    return chat_tokenizer if 'mode = "chat"' in first["profile"]["toml"] else tiny_tokenizer


def test_every_record_reads(golden):
    rows = list(read_records(golden))
    assert all(r["schema_version"] <= SCHEMA_VERSION for r in rows)
    assert rows[0]["type"] == "conversation"
    assert {"turn", "call", "mark", "note", "replay"} <= {r["type"] for r in rows}


def test_every_reply_replays_identically(golden, steerable, golden_tokenizer):
    calls = [r for r in read_records(golden) if r["type"] == "call" and r.get("turn_id")]
    complete = [c for c in calls if c["stop_type"] != "cancelled"]
    backend = FakeBackend([c["output_text"] for c in complete], tokenizer=golden_tokenizer)
    backend.before_v0_2_3 = golden.stem in BEFORE_V0_2_3  # the fake that wrote these logs
    session = Session(steerable, backend=backend, tokenizer_loader=lambda r, v: golden_tokenizer)
    for call in complete:
        record = session.replay(golden, call["call_id"])
        assert record["identical"], call["call_id"]
        assert backend.calls[-1]["token_ids"] == call["prompt_ids"]
        sent = backend.calls[-1]["steering"]  # None for every call before v0.2.2
        assert (sent and sent.to_json()) == call["request"].get("steering")


def test_a_reply_can_be_marked_after_the_fact(golden):
    call = next(r for r in read_records(golden) if r["type"] == "call")
    annotate_log(golden, "mark", call["call_id"], "interesting")
    last = json.loads(golden.read_text(encoding="utf-8").splitlines()[-1])
    assert last["type"] == "mark" and last["call_id"] == call["call_id"]


def test_every_golden_log_resumes(golden, steerable, golden_tokenizer):
    backend = FakeBackend([" More."], tokenizer=golden_tokenizer)
    session = Session(steerable, backend=backend, tokenizer_loader=lambda r, v: golden_tokenizer)
    record = session.resume(golden)
    assert record["render"] is None or record["render"]["identical"], record["render"]
    turn = session.say("One more made-up message.").turn
    assert turn is not None and turn.text == " More."
    # v0.2.2's log ends after /set steering_scale 0: resume keeps it, so no steering.
    assert backend.calls[-1]["steering"] is None


def test_the_v0_2_0_golden_log_holds_every_new_record_type():
    kinds = {r["type"] for r in read_records(GOLDEN / "v0.2.0.jsonl")}
    assert {"head", "unmark", "settings", "title", "resume", "torn_line"} <= kinds


def test_the_v0_2_1_golden_log_holds_the_new_call_fields():
    calls = [r for r in read_records(GOLDEN / "v0.2.1.jsonl") if r["type"] == "call"]
    assert all(isinstance(c["message_spans"], list) for c in calls)
    assert any("tokens" in (c.get("logprobs") or {}) for c in calls)
    assert all("logprobs" in c["settings"] for c in calls)
    regens = [c for c in calls if c.get("regen_of")]
    assert len(regens) == 4 and regens[-1]["stop_type"] == "cancelled"


def test_the_v0_2_2_golden_log_holds_a_steered_and_an_unsteered_call():
    rows = list(read_records(GOLDEN / "v0.2.2.jsonl"))
    calls = [r for r in rows if r["type"] == "call"]
    assert calls[0]["request"]["steering"]["scale"] == 4.0
    assert calls[0]["profile"]["toml"].count("[steering]") == 1
    assert calls[-1]["settings"]["steering_scale"] == {"value": 0.0, "origin": "chat"}
    assert not calls[-1]["request"].get("steering")
    assert any(r["type"] == "settings" and r["name"] == "steering_scale" for r in rows)


def test_the_v0_2_3_golden_log_holds_regen_same_a_failed_call_and_end_tokens():
    rows = list(read_records(GOLDEN / "v0.2.3.jsonl"))
    calls = [r for r in rows if r["type"] == "call"]
    seeds = [(c["settings"]["seed"]["value"], c["settings"]["temperature"]["value"]) for c in calls]
    assert seeds[:4] == [(42, 0.6), (43, 0.6), (42, 0.9), (43, 0.9)]  # /regen same's seeds
    assert all(c["output_ids"][-1] == 257 for c in calls[:4])  # the end token </s>
    assert calls[4]["error"] and not calls[4].get("turn_id")  # a failed call
    assert calls[5]["stop_type"] == "cancelled" and calls[5]["timings"] == {}
    assert "description" in calls[0]["profile"]["toml"]
    assert any(r["type"] == "mark" and r["call_id"] == calls[0]["call_id"] for r in rows)


def test_the_v0_3_0_golden_log_holds_a_rename_a_stop_and_a_resume_with_a_change():
    from clarity_chat.resume import what_differs

    rows = list(read_records(GOLDEN / "v0.3.0.jsonl"))
    calls = [r for r in rows if r["type"] == "call"]
    assert any(r["type"] == "title" and r["title"] == "Weather, made up" for r in rows)
    stopped = calls[2]  # the page's Stop: /cancel from another thread, mid-stream
    assert stopped["stop_type"] == "cancelled" and stopped["cancelled_during"] == "reply"
    resume = next(r for r in rows if r["type"] == "resume")
    assert resume["compared_with_call"] == stopped["call_id"]
    item = what_differs(resume)[-1]  # what the page lists as "client · commit OLD → NEW"
    assert item["call"] == stopped["call_id"]
    assert item["changes"] == {"client": [{"path": "commit", "logged": "0" * 40, "now": "1" * 40}]}
    assert calls[-1]["client"]["commit"] == "1" * 40  # a reply after the resume


def test_every_golden_log_answers_the_six_reads(golden, steerable, golden_tokenizer):
    """Fix review, batch 6, item 4 (from the reviewer's scratch test): a resumed golden log
    through every read the web interface asks, for each of its calls. None says it can't
    read the chat's records; another refusal (a failed call has no reply to compare) is a
    sentence, not a crash."""
    from clarity_chat.commands import Commands
    from clarity_chat.profiles import ConfigError

    backend = FakeBackend([" More."], tokenizer=golden_tokenizer)
    session = Session(steerable, backend=backend, tokenizer_loader=lambda r, v: golden_tokenizer)
    c = Commands(session)
    c.run("resume", str(golden), lambda e: None, paths=True)
    calls = [r["call_id"] for r in read_records(golden) if r["type"] == "call"]
    asked = [(name, "") for name in ("history", "branches", "settings")]
    asked += [(name, call) for call in calls for name in ("compare", "prompt", "probs")]
    unreadable = []
    for name, args in asked:
        try:
            c.run(name, args, lambda e: None)
        except ConfigError as e:
            if "Can't read" in str(e):
                unreadable.append((name, args, str(e)))
    assert unreadable == []


def test_every_golden_log_draws_in_every_view(golden, steerable, golden_tokenizer):
    """Fix review, batch 6, item 4: a golden log opened by the web interface, then each view
    of the chat and of each of its calls, answers 200 and says nothing about records it
    can't read."""
    from starlette.testclient import TestClient

    from clarity_chat import web
    from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command

    logs = steerable / "logs"
    logs.mkdir(exist_ok=True)
    shutil.copy(golden, logs / golden.name)
    backend = FakeBackend([" More."], tokenizer=golden_tokenizer)
    session = Session(steerable, backend=backend, tokenizer_loader=lambda r, v: golden_tokenizer)
    calls = [r for r in read_records(golden) if r["type"] == "call"]
    views = [f"/view/{name}" for name in ("thread", "branches", "settings", "chats", "picker")]
    views += [f"/view/details?call={c['call_id']}" for c in calls]
    # Compare is for a reply; a failed call's has its own sentence, and no control asks it.
    views += [f"/view/compare?call={c['call_id']}" for c in calls if c.get("turn_id")]
    bad = []
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        answer = command(client, "resume", golden.stem).json()
        assert "result" in answer, answer
        for path in views:
            response = client.get(path, headers={**AUTH, **SAME})
            if response.status_code != 200 or "Can't read" in response.text:
                bad.append((path, response.status_code, response.text[:200]))
    assert bad == []


def test_the_v0_3_1_golden_log_holds_a_whole_attachment_and_an_excerpt():
    rows = list(read_records(GOLDEN / "v0.3.1.jsonl"))
    sent = [r for r in rows if r["type"] == "turn" and r.get("attachments")]
    assert [(t["attachments"][0]["how"], t["attachments"][0]["range"]) for t in sent] == [
        ("whole", [1, 3]),
        ("excerpt", [2, 3]),
    ]
    assert sent[0]["framing_origin"] == "default" and sent[0]["placement"] == "start"
    assert sent[1]["attachments"][0]["marker"] == "Excerpt: lines 2 to 3 of 4"
    calls = [r for r in rows if r["type"] == "call"]
    assert all(any(s.get("added") for s in c["message_spans"]) for c in calls)
    assert {"mark", "note", "replay"} <= {r["type"] for r in rows}


def test_the_v0_3_2_golden_log_holds_a_search_found_parts_and_a_miss():
    rows = list(read_records(GOLDEN / "v0.3.2.jsonl"))
    assert [r["documents"] for r in rows if r["type"] == "search"] == [["glossary"], []]
    searched = [r for r in rows if r["type"] == "turn" and "found" in r]
    assert [t["text"] for t in searched] == ["Tell me about the wren", "Any herons?"]
    hit, miss = (t["found"] for t in searched)
    assert [p["id"] for p in hit["parts"]] == ["glossary:2"] and hit["parts"][0]["text"]
    assert [d["name"] for d in hit["documents"]] == ["glossary"]
    assert (miss["matched"], miss["parts"]) == (0, [])
    calls = [r for r in rows if r["type"] == "call"]
    assert any(a["what"] == "part" for s in calls[0]["message_spans"] for a in s.get("added", []))
    replay = next(r for r in rows if r["type"] == "replay")
    assert replay["identical"] and replay["call_id"] == calls[0]["call_id"]


def test_the_v0_4_0_golden_log_holds_a_ban_its_check_and_your_own_entry():
    """A1's three fields (the file's ban on server_start; each call's backend_settings entry
    with origin `model file`, and its check, failed once), on your own entry `mine` (B4),
    and a replay that names the entry it ran on (A3's model_used)."""
    rows = list(read_records(GOLDEN / "v0.4.0.jsonl"))
    (start,) = [r for r in rows if r["type"] == "server_start"]
    banned = start["model_file"]["suppress_tokens"]
    assert len(banned) == 2
    calls = [r for r in rows if r["type"] == "call"]
    assert {c["model"]["name"] for c in calls} == {"mine"}
    for call in calls:
        assert call["backend_settings"]["suppress_tokens"] == {
            "value": banned,
            "origin": "model file",
        }
    assert [c["checks"]["suppress_tokens"]["ok"] for c in calls] == [True, False]
    assert "check_failed:suppress_tokens" in calls[1]["flags"]
    replay = next(r for r in rows if r["type"] == "replay")
    assert replay["identical"] and replay["model_used"] == "mine"


def test_the_v0_4_1_golden_log_holds_a_meaning_search_found_parts_and_a_miss():
    """v0.4.1's meaning search: a found by meaning (the engine and its setup, the query, the
    minimum, each part's score), a message no part reached the minimum for, and a replay,
    identical, that never searched."""
    rows = list(read_records(GOLDEN / "v0.4.1.jsonl"))
    searched = [r for r in rows if r["type"] == "turn" and "found" in r]
    assert [t["text"] for t in searched] == ["Tell me about the wren", "Any herons?"]
    hit, miss = (t["found"] for t in searched)
    assert hit["method"] == miss["method"] == "meaning"
    assert hit["engine"]["model"] == "bge-m3" and len(hit["engine"]["setup"]) == 64
    assert hit["min_similarity"] == 0.1 and hit["min_similarity_origin"] == "profile"
    assert [p["id"] for p in hit["parts"]] == ["glossary:2"]
    assert hit["parts"][0]["score"] >= hit["min_similarity"] and hit["parts"][0]["text"]
    assert (miss["matched"], miss["parts"]) == (0, [])
    replay = next(r for r in rows if r["type"] == "replay")
    calls = [r for r in rows if r["type"] == "call"]
    assert replay["identical"] and replay["call_id"] == calls[0]["call_id"]


def test_the_v0_4_2_golden_log_holds_a_start_s_build_and_a_replay_that_compared_it():
    """v0.4.2's replay compares how llama-server was built (A1): the start logged its build
    line and system_info's features, /props' build_info is a shallow clone's "b1-COMMIT",
    and the replay, identical, found no difference and listed nothing as not recorded."""
    from clarity_chat.replay import build_info_commit, facts_from_call

    rows = list(read_records(GOLDEN / "v0.4.2.jsonl"))
    (start,) = [r for r in rows if r["type"] == "server_start"]
    (call,) = [r for r in rows if r["type"] == "call"]
    facts = facts_from_call(call, start)
    assert facts["build"]["line"].endswith("with AppleClang 21.0.0.21000334 for Darwin arm64")
    assert "MATMUL_INT8 = 1" in facts["build"]["features"]
    assert start["build_info"].startswith("b1-")
    assert facts["build_info"] == build_info_commit(start["build_info"]) == start["build_info"][3:]
    replay = next(r for r in rows if r["type"] == "replay")
    assert replay["identical"] and replay["call_id"] == call["call_id"]
    assert replay["differences"] == {} and "not_recorded" not in replay
