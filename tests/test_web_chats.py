"""The Chats drawer, reopening and renaming a chat, and the export sheet (v0.3.0 item C5),
offline. Values on screen are checked against /chats, /state, /resume and /export (data walk,
V3Chats and V3Export)."""

from __future__ import annotations

import html
import re
import shutil
from datetime import date
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command, wait_until, writing

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
UNREADABLE = 'not a record\n{"type": "title"}\n'  # a bad line before the end: not torn
HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}'


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" One.", " Two.", " Three."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client, session


def result(client, name, args=""):
    answer = command(client, name, args).json()
    assert "result" in answer, answer
    return answer["result"]


def view(client, name: str) -> str:
    response = client.get(f"/view/{name}", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    return html.unescape(response.text)


def rows(fragment: str) -> list[str]:
    return re.findall(r'<li><button type="button" class="crow[^"]*"[^>]*>.*?</button>', fragment)


# --- The drawer (L1, L2)


def test_the_drawer_lists_every_chat_newest_first_today_and_earlier(chat):
    client, session = chat
    result(client, "say", "Hi there, a first message with more than eight words in it")
    shutil.copy(GOLDEN / "v0.2.3.jsonl", session.logs_dir / "20261030-162210-e41b07.jsonl")
    (session.logs_dir / "20261028-101544-7c2e9d.jsonl").write_text(UNREADABLE, "utf-8")
    result(client, "new")  # a chat with no messages yet
    chats = result(client, "chats")["chats"]
    state = result(client, "state")
    fragment = view(client, "chats")
    listed = rows(fragment)
    assert [re.findall(r'data-resume="([^"]+)"', r)[0] for r in listed] == [
        c["log_id"] for c in chats
    ]
    today, earlier = fragment.split("<h3>Earlier</h3>")
    assert "<h3>Today</h3>" in today
    current = next(r for r in listed if 'aria-current="true"' in r)
    assert f'data-resume="{state["log_id"]}"' in current and "No messages" in current
    first = next(c for c in chats if c["title"] and c["title"].startswith("Hi there"))
    hhmm = f"{first['log_id'][9:11]}:{first['log_id'][11:13]}"  # QA 7: today's time
    assert (
        f'<span class="t">{first["title"]}</span><span class="p"><span class="dt">{hhmm}</span>'
        " · text</span>" in today
    )
    old = next(c for c in chats if c["log_id"] == "20261030-162210-e41b07")
    assert (
        f'<span class="t">{old["title"]}</span><span class="p"><span class="dt">30 Oct</span>'
        f" · {old['profile']}</span>" in earlier
    )
    bad = next(r for r in listed if "20261028-101544-7c2e9d" in r)
    assert '<span class="t idm">20261028-101544-7c2e9d</span>' in bad
    assert '<span class="bad">! can’t be read</span>' in bad
    assert fragment.count('aria-current="true"') == 1


def test_an_earlier_year_shows_in_the_date():
    assert web.chat_date("20251230-101010-abcdef", date(2026, 11, 14)) == ("earlier", "30 Dec 2025")
    assert web.chat_date("20261114-090000-abcdef", date(2026, 11, 14)) == ("today", "09:00")
    assert web.chat_date("20261113-235959-abcdef", date(2026, 11, 14)) == ("earlier", "13 Nov")
    assert web.chat_date("old-v010", date(2026, 11, 14)) == ("earlier", None)


def test_with_no_chats_the_drawer_says_so_and_new_chat_waits(fetched_root, tiny_tokenizer):
    session = Session(
        fetched_root, backend=FakeBackend([]), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        fragment = view(client, "chats")
    assert "No chats yet. Each chat you start is listed here." in fragment
    assert re.search(r'<button type="button" class="newchat"[^>]*aria-disabled="true"', fragment)
    assert not rows(fragment)


def test_untrusted_text_in_the_drawer_is_escaped(chat):
    """B5's fixture, for Chats: a title."""
    client, _ = chat
    result(client, "say", "Hi")
    result(client, "title", HOSTILE)
    raw = client.get("/view/chats", headers={**AUTH, **SAME}).text
    assert "<script>" not in raw and "<img" not in raw and HOSTILE in html.unescape(raw)


# --- Reopening a chat (L3, L4, L5, L6)


def two_chats(client, session) -> str:
    """A chat with a reply, then a new one; returns the first chat's log ID."""
    result(client, "say", "Hi")
    first = result(client, "state")["log_id"]
    result(client, "new")
    return first


def test_the_count_of_what_differs_is_the_terminal_s():
    """Teaching check, item 23: the count and the list come from one function."""
    items = [{"text": "A.", "count": 1}, {"text": "B.", "count": 2}]
    assert web.differ_line({"count": 3, "items": items}) == "3 things differ"
    assert web.differ_line({"count": 1, "items": items[:1]}) == "1 thing differs"


def test_reopened_the_line_counts_what_differs_and_lists_it(chat, fetched_root):
    client, session = chat
    first = two_chats(client, session)
    profile = fetched_root / "profiles" / "text.toml"
    profile.write_text(profile.read_text("utf-8") + "\n# changed\n", "utf-8")
    differ = result(client, "resume", first)["differ"]
    assert differ["count"] == 1
    fragment = view(client, "thread")
    assert "1 thing differs from when this chat was written" in fragment
    assert 'aria-expanded="false" aria-controls="differ"' in fragment
    listed = re.search(r'<ul id="differ"[^>]*hidden>(.*?)</ul>', fragment, re.S)
    texts = [item["text"] for item in differ["items"]]
    assert listed and re.findall(r"<li>(.*?)</li>", listed.group(1)) == texts


def test_reopened_with_nothing_different_says_so_and_goes_after_the_next_reply(chat):
    client, session = chat
    first = two_chats(client, session)
    assert result(client, "resume", first)["differ"]["count"] == 0
    nothing = "Nothing Clarity checks differs from when this chat was written."
    assert nothing in view(client, "thread")
    result(client, "say", "Again")
    fragment = view(client, "thread")
    assert "Nothing differs" not in fragment and "differ from when" not in fragment


def test_reopened_with_an_unanswered_message_says_it_s_back_in_the_box(chat):
    client, session = chat
    result(client, "say", "Hi")
    session.backend.fail_with = "A made-up failure."
    command(client, "say", "Lost one")
    first = result(client, "state")["log_id"]
    session.backend.fail_with = None
    result(client, "new")
    result(client, "resume", first)
    state = result(client, "state")
    assert state["unanswered"]["text"] == "Lost one"
    fragment = view(client, "thread")
    assert (
        "<strong>!</strong> Your last message got no reply (its call failed or was cut off). "
        "It’s back in the box: send it to get one." in fragment
    )
    assert "Nothing Clarity checks differs from when this chat was written." in fragment
    assert "Failed:" not in fragment  # the resumed chat says it once, as L5 draws it


def test_a_refused_resume_leaves_the_chat_you_were_in_usable(chat):
    """Plan item (V3Chats): a refusal while opening a chat leaves the current one as it was."""
    client, session = chat
    result(client, "say", "Hi")
    before = result(client, "state")
    (session.logs_dir / "20261028-101544-7c2e9d.jsonl").write_text(UNREADABLE, "utf-8")
    answer = command(client, "resume", "20261028-101544-7c2e9d").json()
    assert answer["type"] == "ConfigError"
    after = result(client, "state")
    assert (after["log_id"], after["turns"]) == (before["log_id"], before["turns"])
    assert after["model"] == "ready"
    assert result(client, "say", "Still here")["call"]["output_text"]


# --- The export sheet (M1, M2)


def exported(client) -> str:
    return view(client, "export")


def test_the_sheet_says_where_the_file_is_and_what_it_holds(chat):
    client, session = chat
    result(client, "say", "Hi")
    session.backend.fail_with = "A made-up failure."
    command(client, "say", "Again")
    session.backend.fail_with = None
    first = result(client, "export")
    assert first["failed_calls_left_out"] == 1 and not first["replaced"] and not first["steered"]
    fragment = exported(client)
    assert "Exported for a Python notebook" in fragment
    assert f'The file stays at <span class="path">{first["path"]}</span>' in fragment
    assert "It replaced the earlier export of this chat." not in fragment
    assert "Readable only by you (on Mac and Linux)." in fragment
    assert (
        "Left out: 1 call that produced no reply. Replays aren’t exported: they re-ran replies "
        "already here." in fragment
    )
    assert "Some replies were steered" not in fragment and "send the vector too" not in fragment
    assert 'data-docs="guides.md#take-a-conversation-to-a-notebook"' in fragment
    again = result(client, "export")
    assert again["replaced"]
    assert "It replaced the earlier export of this chat." in exported(client)


def test_a_steered_export_names_the_vector_to_send():
    fragment = html.unescape(
        web.templates.get_template("export.html").render(
            **web.export_sheet(
                {
                    "result": {
                        "path": "/x/logs/exports/c.jsonl",
                        "replaced": False,
                        "calls": 3,
                        "failed_calls_left_out": 0,
                        "calls_without_a_turn_left_out": 2,
                        "steered": [
                            {"file": "local/vectors/calm.gguf", "sha256": "a3", "call_ids": ["x"]}
                        ],
                        "attachments": False,
                        "found": False,
                    }
                }
            )
        )
    )
    assert "Some replies were steered: apply the same vector before recomputing them." in fragment
    assert "The file names the steering vector but doesn’t contain it, so send the vector too:" in (
        fragment
    )
    assert '<span class="path">local/vectors/calm.gguf</span>' in fragment
    assert "Left out: 2 calls that produced no reply." in fragment


def test_a_vector_path_with_control_characters_shows_them_escaped_with_no_copy():
    """Security follow-up (batch 3), 5: a vector's path comes from the log, which another
    program can write. One holding control or invisible characters (an escape sequence, a
    right-to-left override) shows them as escapes, with no Copy, since a copy wouldn't name
    the file. A plain path keeps its Copy."""
    odd = "local/vectors/\x1b[31mcalm\u202eftg.gguf"
    vectors = [
        {"file": odd, "sha256": "a3", "call_ids": ["x"]},
        {"file": "local/vectors/plain.gguf", "sha256": "b4", "call_ids": ["y"]},
    ]
    answer = {
        "result": {
            "path": "/x/logs/exports/c.jsonl",
            "replaced": False,
            "calls": 2,
            "failed_calls_left_out": 0,
            "calls_without_a_turn_left_out": 0,
            "steered": vectors,
            "attachments": False,
            "found": False,
        }
    }
    raw = web.templates.get_template("export.html").render(**web.export_sheet(answer))
    fragment = html.unescape(raw)
    assert "\x1b" not in fragment and "\u202e" not in fragment
    assert '<span class="path">local/vectors/\\x1b[31mcalm\\u202eftg.gguf</span>' in fragment
    assert fragment.count("Copy the vector’s path") == 1
    assert 'data-copy="local/vectors/plain.gguf"' in fragment


def test_a_failed_export_says_why_and_offers_to_try_again(chat):
    client, session = chat
    result(client, "say", "Hi")
    (session.logs_dir / "exports").write_text("in the way", "utf-8")
    answer = command(client, "export").json()
    assert "Can't write the export to" in answer["error"]
    fragment = exported(client)
    assert "Not exported" in fragment and f"<strong>!</strong> {answer['error']}" in fragment
    assert "Nothing was written, and any earlier export of this chat is unchanged." in fragment
    assert 'data-action="export"' in fragment and "Download" not in fragment


def test_the_sheet_is_for_this_chat_only(chat):
    client, _ = chat
    result(client, "say", "Hi")
    result(client, "export")
    result(client, "new")
    response = client.get("/view/export", headers={**AUTH, **SAME})
    assert response.status_code == 409
    assert response.text == "Export this chat first; the file is made when you export it."


# --- The tripwire review after C5


def test_a_resent_message_that_fails_again_shows_its_own_failure(chat):
    """Finding 2: after a resume, the line about what differs goes with the next message,
    sent or failed, so a new failure shows its sentence and Details, not "back in the box"."""
    client, session = chat
    result(client, "say", "Hi")
    session.backend.fail_with = "A made-up failure."
    command(client, "say", "Lost one")
    first = result(client, "state")["log_id"]
    session.backend.fail_with = None
    result(client, "new")
    result(client, "resume", first)
    session.backend.fail_with = "The second, different failure."
    command(client, "say", "Lost one")
    assert result(client, "state")["resume_differ"] is None
    fragment = view(client, "thread")
    assert "<strong>! Failed:</strong> The second, different failure." in fragment
    assert 'data-action="details"' in fragment.split("The second, different failure.")[1]
    assert "back in the box" not in fragment


def test_spans_without_a_reason_draw_the_prompt_unmarked(chat):
    """Finding 13: a `message_spans` dict without "unavailable" (a log another program
    wrote) mustn't answer 500."""
    client, session = chat
    call = result(client, "say", "Hi")["call"]
    session.conversation.calls[call["call_id"]]["message_spans"] = {}
    lenient = TestClient(client.app, base_url=ORIGIN, raise_server_exceptions=False)
    response = lenient.get(f"/view/details?call={call['call_id']}", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    assert "Not highlighted: the log gives no reason." in html.unescape(response.text)


def test_the_drawer_marks_a_copied_log_by_its_own_name(chat):
    """Finding 5a: the open chat is marked by /state's log_id, which a copied log's
    conversation ID doesn't match."""
    client, session = chat
    result(client, "say", "Hi")
    original = result(client, "state")["log_id"]
    shutil.copy(session.logs_dir / f"{original}.jsonl", session.logs_dir / "my-copy.jsonl")
    result(client, "resume", "my-copy")
    assert result(client, "state")["conversation_id"] == original  # the copy's ID
    listed = rows(view(client, "chats"))
    current = [r for r in listed if 'aria-current="true"' in r]
    assert len(current) == 1 and 'data-resume="my-copy"' in current[0]


@pytest.mark.parametrize("how", ["say", "start"])
def test_the_drawer_rows_wait_while_the_chat_changes(chat, how):
    """Finding 5b: while a reply is written or a model starts, the rows and New chat wait,
    and say why."""
    import threading

    from tests.test_web_shell import start_held

    client, session = chat
    result(client, "say", "Hi")
    if how == "say":
        session.backend.replies, session.backend.piece_delay = [" A slow reply"], 1.0
        worker = threading.Thread(target=lambda: command(client, "say", "Again"))
        worker.start()
        wait_until(lambda: writing(client))  # the reply is being written
        gate = None
    else:
        gate, worker = start_held(client, session, "", how="new")
    try:
        fragment = view(client, "chats")
    finally:
        if gate:
            gate.set()
        worker.join(10)
    why = "Waits until the reply is done." if how == "say" else "Waits until the model has started."
    assert f'<p class="waits" id="chats-waits">{why}</p>' in fragment
    controls = re.findall(r'<button type="button" class="(?:crow|newchat)"[^>]*>', fragment)
    assert controls and all('aria-disabled="true"' in c for c in controls)


def test_the_export_is_for_the_open_log_not_a_copy_with_its_id(chat):
    """Finding 6: the sheet and the Download are this start's export of the open log. A
    copy of the log has the same conversation ID, but its export wasn't made."""
    client, session = chat
    result(client, "say", "Hi")
    original = result(client, "state")["log_id"]
    result(client, "export")
    assert client.get("/export", headers={**AUTH, **SAME}).status_code == 200
    shutil.copy(session.logs_dir / f"{original}.jsonl", session.logs_dir / "my-copy.jsonl")
    result(client, "resume", "my-copy")
    for path in ("/export", "/view/export"):
        response = client.get(path, headers={**AUTH, **SAME})
        assert response.status_code == 409, path
        assert response.text == "Export this chat first; the file is made when you export it."


@pytest.mark.parametrize("view_name", ["chats", "branches"])
def test_a_command_other_than_a_reply_says_the_command(chat, view_name):
    """Finding 9: a running command that isn't a reply or a start (an export, a switch)
    isn't "the reply"."""
    from tests.test_web_thread import held_read

    client, _ = chat
    result(client, "say", "Hi")
    result(client, "regen")
    gate, holder, seen, restore = held_read(client, "export", "")
    try:
        fragment = view(client, view_name)
    finally:
        gate.set()
        holder.join(5)
        restore()
    assert "export" in seen
    assert "Waits until the command is done." in fragment
    assert "Waits until the reply is done." not in fragment


def test_a_failed_regenerate_after_a_resume_keeps_what_differs(chat):
    """Finding 8: only a message (back_to None) uses up the resumed chat's line; a failed
    Regenerate or Edit doesn't, since the resumed message is still where it was."""
    client, session = chat
    first = two_chats(client, session)
    result(client, "resume", first)
    session.backend.fail_with = "A made-up failure."
    command(client, "regen")
    assert result(client, "state")["resume_differ"] is not None


# --- The Opus whole-branch review (batch 4)


def test_a_dotted_log_name_opens_by_name_and_chats_rows_say_both_ids(chat):
    """SHOULD 2 and 6: one rule for a log's name (dots allowed, not first; no separators),
    used by /resume and export alike, so a log named chat.v2 opens from the drawer; /chats
    rows give log_id (the file's name, what /resume takes) and conversation_id (the logged
    ID); the drawer compares log_id with /state's log_id."""
    client, session = chat
    result(client, "say", "Hi")
    original = result(client, "state")
    logs = session.logs_dir
    shutil.copy(logs / f"{original['log_id']}.jsonl", logs / "chat.v2.jsonl")
    (logs / ".hidden.jsonl").write_text("", "utf-8")
    rows_by_log = {c.get("log_id"): c for c in result(client, "chats")["chats"]}
    assert "chat.v2" in rows_by_log and ".hidden" in rows_by_log, list(rows_by_log)
    assert rows_by_log["chat.v2"]["conversation_id"] == original["conversation_id"]
    assert rows_by_log["chat.v2"]["openable"] is True
    assert rows_by_log[".hidden"]["openable"] is False
    resumed = command(client, "resume", "chat.v2").json()
    assert "result" in resumed, resumed
    assert result(client, "state")["log_id"] == "chat.v2"
    fragment = view(client, "chats")
    current = [r for r in rows(fragment) if 'aria-current="true"' in r]
    assert len(current) == 1 and 'data-resume="chat.v2"' in current[0]


# --- QA of v0.3.0 (batch 5)


def test_today_s_chats_show_the_time_they_started():
    """QA 7: Today's rows had only the title and profile, so a day of "Name one animal."
    chats couldn't be told apart. They show the local time the chat started (HH:MM)."""
    assert web.chat_date("20261114-090500-abcdef", date(2026, 11, 14)) == ("today", "09:05")
    assert web.chat_date("20261114-235959-abcdef", date(2026, 11, 14)) == ("today", "23:59")


def test_what_changed_since_the_last_call_says_each_change_old_to_new(chat):
    """QA 6 (run23): "Changed since its last call (…): client." named a category. Each
    change reads as Details' "Changed since the call" does: "client · commit OLD → NEW"."""
    import json

    client, session = chat
    result(client, "say", "Hi")
    assert session.conversation is not None
    log = session.conversation.log.path
    first = log.stem
    records = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    for record in records:
        if record.get("type") == "call":
            now = record["client"]["commit"]
            record["client"]["commit"] = "a" * 40
    log.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
    result(client, "new")
    assert result(client, "resume", first)["differ"]["count"] >= 1
    listed = re.search(r'<ul id="differ"[^>]*>(.*?)</ul>', view(client, "thread"), re.S)
    assert listed and f"client · commit {'a' * 40} → {now}" in listed.group(1)


def test_a_log_whose_name_resume_wont_take_is_listed_and_cant_be_picked(chat):
    """Fix review, batch 6, item 11: a log named so /resume won't take it (".copy", say) is
    a row by its file name saying why, which can't be picked. The review's mutation (the
    row offered) survived: no view test drew it."""
    client, session = chat
    result(client, "say", "Hi")
    logs = session.logs_dir
    shutil.copy(logs / f"{result(client, 'state')['log_id']}.jsonl", logs / ".copy.jsonl")
    row = next(r for r in rows(view(client, "chats")) if 'data-resume=".copy"' in r)
    assert 'aria-disabled="true"' in row and "aria-describedby" not in row
    assert '<span class="t idm">.copy</span>' in row
    assert "! its file name can’t be opened from here" in row
