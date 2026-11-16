"""v0.3.1's close, the server side: the final fix review
(notes/research/reviews/v031-fix-review-final.md) and the three finders (surface, siblings,
records, in notes/research/reviews/v031/). Made-up documents only."""

from __future__ import annotations

import asyncio
import html
import io
import json
import re
from pathlib import Path
from typing import cast

import pytest

from clarity_chat import cli, web
from clarity_chat.commands import Commands, read_files
from clarity_chat.documents import SENTENCES, DocumentRefused, checked_attachments, reading
from clarity_chat.documents import child as child_module
from clarity_chat.json_chat import JsonChat
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.pdfs import make_pdf
from tests.test_attach_whole_branch import text_of
from tests.test_cli_attach import chat_loop
from tests.test_session_attach import chat_session, doc
from tests.test_web import GET, command
from tests.test_web_attach import NOTES, make, sheet
from tests.test_web_upload import upload

__all__ = ["make"]  # the fixture, used by name

ROOT = Path(__file__).parent.parent


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


def quiet(event):
    pass


# --- Must A: "Not kept" at exit in --json, and the page's sentence


def test_json_end_of_input_with_attachments_waiting_says_they_aren_t_kept(
    fetched_root, chat_tokenizer, tmp_path
):
    reading._last = None
    path = tmp_path / "notes.md"
    path.write_bytes(NOTES)
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    stdin = io.BytesIO(json.dumps({"command": "attach", "args": str(path)}).encode() + b"\n")
    stdout = io.StringIO()
    JsonChat(session, stdin, stdout).run([{"command": "profile", "args": "chat"}])
    last = json.loads(stdout.getvalue().splitlines()[-1])
    assert last["event"] == "result" and last["command"] == "quit", last
    assert [a["name"] for a in last["result"]["dropped"]] == ["notes.md"]


def test_json_end_of_input_with_nothing_waiting_adds_nothing(fetched_root, chat_tokenizer):
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    stdout = io.StringIO()
    JsonChat(session, io.BytesIO(), stdout).run([{"command": "profile", "args": "chat"}])
    last = json.loads(stdout.getvalue().splitlines()[-1])
    assert last["command"] == "profile", last


def test_the_page_carries_the_not_kept_sentence(make):
    client, _, _ = make()
    shell = client.get("/").text
    assert 'data-not-kept="Not kept: {names}, waiting for your next message.' in shell


# --- Must B: the check's cache knows the file's name


def test_the_check_counts_each_name_of_the_same_bytes(chat):
    session, _ = chat
    short = session.check(doc("Same bytes.\n", "a.md"))
    longer = "a-much-much-longer-file-name-than-before.md"
    counted = session.check(doc("Same bytes.\n", longer))
    record, _ = session.attach(doc("Same bytes.\n", longer))
    assert counted["tokens_whole"] == record["tokens"]
    assert short["tokens_whole"] < counted["tokens_whole"]


def test_the_check_s_cache_knows_the_waiting_attachments_names(chat):
    session, _ = chat
    session.attach(doc("Waiting.\n", "w.md"))
    session.check(doc("Checked.\n", "c.md"))
    session.conversation.pending[0]["name"] = "a-much-longer-name-for-the-same-bytes.md"
    after = session.check(doc("Checked.\n", "c.md"))
    _, fitted = session.attach(doc("Checked.\n", "c.md"))
    assert after["left_whole"] == fitted.left


# --- fix review 2: /state has `mode` before a chat


def test_state_has_the_same_keys_before_and_after_a_start(fetched_root, chat_tokenizer):
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    commands = Commands(session)
    before = commands.run("state", "", quiet)
    assert before.get("mode", "missing") is None
    commands.run("profile", "chat", quiet)
    assert set(before) == set(commands.run("state", "", quiet))


# --- fix review 3, siblings S5, surface could 6: one range wording, with separators


def test_the_terminal_s_waiting_line_has_separators_and_the_total(chat):
    session, _ = chat
    session.attach(doc("x\n" * 2000, "long.md"), "lines", 1, 1900)
    commands = Commands(session)
    line = cli.format_pending(commands.run("attach", "", quiet)["pending"])
    assert "long.md, lines 1–1,900 of 2,000, " in line


def test_the_chip_and_the_thread_say_a_range_the_same_way(make):
    client, _, _ = make()
    upload_id = upload(client, b"x\n" * 2000, "long.md").json()["upload"]
    command(client, "attach", f"{upload_id} lines 3-1900")
    chip = text_of(client.get("/view/pending", headers=GET).text)
    assert "lines 3–1,900 of 2,000" in chip
    command(client, "say", "What's in it?")
    thread = text_of(client.get("/view/thread", headers=GET).text)
    assert "lines 3–1,900 of 2,000" in thread


# --- fix review 4: the page-count line in the terminal's check, and in guides.md


def test_the_terminal_s_check_of_a_pdf_says_how_pages_are_counted(chat, tmp_path):
    session, _ = chat
    commands = Commands(session)
    path = tmp_path / "book.pdf"
    path.write_bytes(make_pdf(["Made-up first page.", "Made-up second page."]))
    check = commands.run("attach", f"{path} check", quiet)["check"]
    assert SENTENCES["pdf_pages"] in cli.format_check(check)
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    check = commands.run("attach", f"{notes} check", quiet)["check"]
    assert SENTENCES["pdf_pages"] not in cli.format_check(check)


def guides_paragraph(starting: str) -> str:
    text = (ROOT / "docs" / "guides.md").read_text(encoding="utf-8")
    (paragraph,) = [p for p in text.split("\n\n") if p.startswith(starting)]
    return paragraph


def test_guides_say_how_pages_are_counted_where_pages_a_b_is_introduced():
    assert SENTENCES["pdf_pages"] in guides_paragraph("`/attach notes.md check`")
    assert SENTENCES["pdf_pages"] not in guides_paragraph("In the web interface, the paperclip")


def test_guides_don_t_say_an_excerpt_goes_in_complete():
    assert "go in complete" not in guides_paragraph("`/attach notes.md check`")
    assert "an excerpt unless the profile sets no marker" in guides_paragraph(
        "`/attach notes.md check`"
    )  # the final fact-check's row 40


# --- fix review 5: Settings waits, and says why, during a file read


class HeldWorker:
    """A worker whose /state says `running`, for a view drawn mid-command."""

    def __init__(self, client, running):
        self.client, self.running = client, running

    async def ask(self, name, args=""):
        result = command(self.client, name, args).json()["result"]
        if name == "state":
            result["running"] = self.running
        return {"result": result}


def test_settings_wait_with_a_reason_during_a_file_read(make):
    client, _, _ = make()
    held = HeldWorker(client, {"command": "attach", "args": "u-1 check"})
    page = asyncio.run(web._settings(cast(web.Worker, held), {}))
    assert 'aria-disabled="true"' in page
    assert "Changes wait until the file is read." in text_of(page)


def test_settings_during_a_reply_still_say_the_reply(make):
    client, _, _ = make()
    held = HeldWorker(client, {"command": "say", "args": "Hi"})
    assert "Changes wait until this reply is done." in text_of(
        asyncio.run(web._settings(cast(web.Worker, held), {}))
    )


# --- fix review 6 and surface could 7: reference.md


def test_reference_lists_cancel_s_upload_and_reads_during_a_file_read():
    text = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    (row,) = [line for line in text.splitlines() if line.startswith("| `cancel` |")]
    assert "`upload`" in row
    assert "while a reply is being written or replayed, or a file is read" in text


# --- fix review 7, surface should 2: the chips carry no key


def test_the_chips_carry_their_number_and_no_key(make):
    client, _, _ = make()
    for name in ("a.md", "copy-of-a.md"):
        upload_id = upload(client, NOTES, name).json()["upload"]
        command(client, "attach", upload_id)
    chips = client.get("/view/pending", headers=GET).text
    assert "data-key" not in chips
    assert re.findall(r'data-n="(\d+)" aria-label="Remove ([^"]+)"', chips) == [
        ("1", "a.md"),
        ("2", "copy-of-a.md"),
    ]


# --- siblings S2: the terminal gives back a message that wasn't sent


def test_a_message_too_long_comes_back_in_the_input(fetched_root, chat_tokenizer):
    loop, output, backend = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    backend._n_ctx = 60
    output.clear()
    message = " ".join(["word"] * 40)
    loop.handle(message)
    assert "Your message is back in the input to edit or send again." in "".join(output)
    assert loop.draft == message


def test_a_failed_call_s_message_comes_back_escaped(fetched_root, chat_tokenizer):
    loop, output, backend = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    backend.fail_with = "made-up failure"
    output.clear()
    loop.handle("/say /quit is a word here")
    assert "Your message is back in the input to edit or send again." in "".join(output)
    assert loop.draft == "//quit is a word here"


# --- siblings S3: the new chat's numbers are the new chat's


def test_the_new_chat_s_numbers_use_the_profile_new_would_start(chat, fetched_root, tmp_path):
    session, _ = chat
    path = fetched_root / "profiles" / "chat.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("max_tokens = 16", "max_tokens = 300"), encoding="utf-8")
    commands = Commands(session)
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    offered = commands.run("attach", f"{notes} check", quiet)["check"]["free_new_chat"]
    commands.run("new", "", quiet)
    assert commands.run("attach", f"{notes} check", quiet)["check"]["free"] == offered


# --- siblings S4: the terminal names a message's attachments


def test_history_names_a_message_s_attachments(fetched_root, chat_tokenizer, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    loop.handle(f"/attach {notes}")
    tokens = loop.session.conversation.pending[0]["tokens"]
    loop.handle("Summarize")
    output.clear()
    loop.handle("/history")
    assert f"  Attached: notes.md · whole · {tokens:,} tokens when attached\n" in "".join(output)


def test_edit_says_what_it_keeps(fetched_root, chat_tokenizer, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    loop.handle(f"/attach {notes}")
    loop.handle("Summarize")
    output.clear()
    loop.handle("/edit Summarize better")
    assert SENTENCES["keeps"].format(names="notes.md") in "".join(output)


# --- records 2: an attachment's fields are checked for type


def turn(**changed):
    attachment = {
        "name": "notes.md", "kind": "text", "bytes": 10, "sha256": "0" * 64,
        "extractor": {"name": "clarity-text", "version": 1, "bom_removed": False},
        "unit": "line", "total": 3, "how": "whole", "range": [1, 3], "marker": None,
        "text": "Made-up.\n", "extracted_sha256": "1" * 64, "empty_units": 0,
        "control_tokens": [], "has_closing_tag": False, "tokens": 40,
    }  # fmt: skip
    fields = {"framing": "Made-up framing.", "framing_origin": "default", "placement": "start"}
    for key, value in changed.items():
        (fields if key in fields else attachment)[key] = value
    return {"attachments": [attachment], **fields}


def test_a_well_formed_attachment_passes():
    assert checked_attachments(turn())["attachments"][0]["tokens"] == 40


@pytest.mark.parametrize(
    "changed",
    [
        {"tokens": None},
        {"tokens": "40"},
        {"tokens": True},
        {"range": None},
        {"range": [1]},
        {"range": [1, "3"]},
        {"how": None},
        {"how": "part"},
        {"unit": "pages"},
        {"kind": "doc"},
        {"marker": 5},
        {"text": None},
        {"total": 3.0},
        {"empty_units": None},
        {"bytes": "10"},
        {"control_tokens": "<s>"},
        {"control_tokens": [5]},
        {"has_closing_tag": "no"},
        {"extractor": None},
        {"sha256": None},
        {"extracted_sha256": 5},
        {"framing": None},
        {"framing_origin": "elsewhere"},
        {"placement": "middle"},  # "end" is a placement since v0.3.2
    ],
)
def test_a_wrong_typed_field_is_refused_as_unreadable(changed):
    with pytest.raises(TypeError):
        checked_attachments(turn(**changed))


# --- records 3: an excerpt with no text is refused, never sent empty


def test_an_excerpt_of_pages_with_no_text_is_refused(chat, tmp_path):
    session, _ = chat
    commands = Commands(session)
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf(["Made-up one.", "", "Made-up three."]))
    sentence = "Pages 2 to 2 of scan.pdf have no text Clarity can read."
    with pytest.raises(DocumentRefused, match=re.escape(sentence)):
        commands.run("attach", f"{path} pages 2-2", quiet)
    with pytest.raises(DocumentRefused, match=re.escape(sentence)):
        commands.run("attach", f"{path} pages 2-2 check", quiet)
    assert session.conversation.pending == []
    commands.run("attach", f"{path} pages 1-2", quiet)  # a range with some text attaches


def test_the_sheet_shows_an_empty_range_s_sentence_in_the_part(make):
    client, _, _ = make()
    answer = upload(client, make_pdf(["Made-up one.", "", "Made-up three."]), "scan.pdf")
    text = sheet(client, upload=answer.json()["upload"], unit="pages", first="2", last="2")
    assert "Pages 2 to 2 of scan.pdf have no text Clarity can read." in text


# --- surface could 5: the detach refusal names no terminal command


def test_detach_s_refusal_names_no_command(chat):
    session, _ = chat
    with pytest.raises(ConfigError) as refused:
        Commands(session).run("detach", "3", quiet)
    assert str(refused.value) == "No attachment 3 is waiting."


# --- siblings S6 and fix review 14: two sentences into SENTENCES


def test_stopped_counting_and_the_empty_send_with_chips_are_in_sentences(chat):
    session, _ = chat
    assert SENTENCES.get("stopped_counting") == "Stopped counting {name}."
    assert SENTENCES.get("empty_with_attachments") == "Type a message to send with {names}."
    session.attach(doc("Waiting.\n", "w.md"))
    with pytest.raises(ConfigError, match="^Type a message to send with w.md.$"):
        Commands(session).run("say", "", quiet)
    reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    assert "Stopped counting {name}." in reference
    assert "Type a message to send with {names}." in reference


# --- siblings S7: --attach reads ~ as the home folder, as /attach does


def test_attach_flag_expands_the_home_folder(tmp_path, monkeypatch):
    reading._last = None
    (tmp_path / "notes.md").write_bytes(NOTES)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    try:
        found = read_files(["~/notes.md"])
    except DocumentRefused as e:
        pytest.fail(f"refused: {e}")
    assert found["~/notes.md"].name == "notes.md"


# --- fix review could 10: a MemoryError is the memory refusal only under the limit


def test_a_memory_error_without_the_limit_is_a_failure(monkeypatch, capsysbinary):
    """On macOS and Windows (no address-space limit) one impossible allocation isn't the
    process using its memory cap: it's the reader failing, as before CI's fix."""
    exits = []

    def exit_(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(child_module.os, "_exit", exit_)
    monkeypatch.setattr(child_module, "watch", lambda cap, seconds: False)
    request = json.dumps({"memory_cap": 1, "seconds": 1, "kind": "text", "name": "n.md"})
    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(request.encode() + b"\nx"), encoding="utf-8")
    )

    def no_memory(*args):
        raise MemoryError

    monkeypatch.setattr(child_module, "extract", no_memory)
    try:
        child_module.main()
    except SystemExit:
        pass
    assert exits == []
    assert json.loads(capsysbinary.readouterr().out) == {"failed": "MemoryError"}


# --- fix review could 11: the slow line names what stops at 120 seconds


def test_the_slow_line_says_it_s_the_read_that_stops():
    assert reading.slow_read() == (
        "A long file can take a minute to read. Clarity stops reading at 120 seconds."
    )


# --- records could 4 and 5: two sentences in reference.md


def test_reference_says_tokens_aren_t_a_sum_and_a_failed_message_s_come_back():
    text = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    assert "so they don't add up to the prompt's cost" in text
    assert "including any you removed before quitting" in text


# --- the mutation gap: F7's web page


def test_a_web_page_s_extracted_text_opens_shown(make):
    client, _, _ = make()
    page = b"<html><head><title>T</title></head><body><p>Made-up page.</p></body></html>"
    upload_id = upload(client, page, "page.html").json()["upload"]
    preview = re.findall(r'id="attach-preview"[^>]*>', sheet(client, upload=upload_id))
    assert len(preview) == 1
    assert " hidden" not in preview[0]


@pytest.mark.parametrize(
    "change",
    [
        (
            'mode = "chat"\nsystem = "Be brief."',
            'mode = "text"\nuser_name = "U"\nassistant_name = "A"',
        ),
        ('model = "tiny"', 'model = "gone"'),
    ],
)
def test_no_new_chat_is_offered_when_new_would_refuse_it(chat, fetched_root, tmp_path, change):
    """Today's profile file is in text mode, or names a model that isn't there: /new can't
    start a chat that takes the file, so the check offers none."""
    session, _ = chat
    path = fetched_root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8").replace(*change), encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    check = Commands(session).run("attach", f"{notes} check", quiet)["check"]
    assert (check["fits_new_chat"], check["free_new_chat"]) == (False, None)


# --- queue item 8: the fix reviewer's confirmation and QA's findings


def test_an_empty_excerpt_is_refused_without_a_marker_too(fetched_root, chat_tokenizer, tmp_path):
    """Confirmation should 1: with `excerpt_marker = ""` the excerpt has no marker; it's
    still an excerpt, and empty."""
    path = fetched_root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nexcerpt_marker = ""\n')
    reading._last = None
    session, _ = chat_session(fetched_root, chat_tokenizer)
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(make_pdf(["Made-up one.", "", "Made-up three."]))
    with pytest.raises(DocumentRefused, match="^Pages 2 to 2 of scan.pdf have no text"):
        Commands(session).run("attach", f"{scan} pages 2-2", quiet)
    assert session.conversation is not None and session.conversation.pending == []


def test_a_failed_edit_comes_back_as_the_edit(fetched_root, chat_tokenizer):
    """Confirmation should 2: Enter on what comes back edits again, with the edit's
    attachments; it never sends a new message."""
    loop, output, backend = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    loop.handle("Hi")
    backend.fail_with = "made-up failure"
    loop.handle("/edit Hello instead")
    assert loop.draft == "/edit Hello instead"


def text_pane(page: str) -> str:
    (pane,) = re.findall(r'<pre class="pre" data-pane="text">(.*?)</pre>', page, re.S)
    return html.unescape(pane)


def test_a_text_direction_control_shows_as_a_marker_not_reversed(make):
    """QA 1: Details' Text pane and the extracted-text preview showed "rtl: txet live"."""
    client, _, _ = make()
    odd = "rtl: ‮evil text\n".encode()
    upload_id = upload(client, odd, "odd.md").json()["upload"]
    preview = sheet(client, upload=upload_id)
    preview = preview[preview.index('id="attach-preview"') :]
    assert "‮" not in preview and "rtl: ⟨U+202E⟩evil text" in preview
    command(client, "attach", upload_id)
    command(client, "say", "What's in it?")
    call = command(client, "prompt").json()["result"]["call"]["call_id"]
    pane = text_pane(client.get("/view/details", params={"call": call}, headers=GET).text)
    assert "‮" not in pane and "rtl: ⟨U+202E⟩evil text" in pane


def test_the_terminal_says_a_file_is_already_in_the_chat(fetched_root, chat_tokenizer, tmp_path):
    """QA 3: the page's note, in the terminal; a note, not a refusal."""
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    output.clear()
    loop.handle(f"/attach {notes}")
    assert SENTENCES["already"] not in "".join(output)
    output.clear()
    loop.handle(f"/attach {notes}")
    assert SENTENCES["already"] in "".join(output)
    assert len(loop.session.conversation.pending) == 2


def test_attach_alone_prints_the_waiting_line_once(fetched_root, chat_tokenizer, tmp_path):
    """QA 4: its answer, and not again before the next prompt."""
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer, reads=[f"/attach {notes}", "/attach"])
    loop.run()
    assert "".join(output).count("Waiting for your next message: 1. notes.md") == 2


def test_undo_says_the_message_s_attachments_went_with_it(fetched_root, chat_tokenizer, tmp_path):
    """QA 9."""
    notes = tmp_path / "notes.md"
    notes.write_bytes(NOTES)
    loop, output, _ = chat_loop(fetched_root, chat_tokenizer)
    loop.start()
    loop.handle(f"/attach {notes}")
    loop.handle("Summarize")
    output.clear()
    loop.handle("/undo")
    assert "Its attachments went with it: notes.md." in "".join(output)
    output.clear()
    loop.handle("Hi")
    loop.handle("/undo")
    assert "attachments went" not in "".join(output)
