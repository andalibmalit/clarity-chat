"""C1 (v0.3.1): the attach sheet and the chips, as the views draw them, offline. The sheet
shows `/attach ID [range] check`'s numbers in the table's words (SENTENCES), as the frozen
board (V31Attach) lays them out. Made-up documents only."""

from __future__ import annotations

import html
import re

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import FRAMING, SENTENCES, reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import GET, ORIGIN, PORT, SECRET, Unsafe, command
from tests.test_web_upload import upload

NOTES = b"Line one.\nLine two.\nLine three.\n"
LONG = "".join(f"Made-up line {n}.\n" for n in range(200)).encode()


@pytest.fixture
def make(fetched_root, chat_tokenizer):
    """A page's app on the chat profile: (client, session, backend), with `n_ctx` and the
    profile's own TOML lines (`extra`) as asked."""
    clients = []

    def make(n_ctx=None, extra=""):
        reading._last = None
        if extra:
            path = fetched_root / "profiles" / "chat.toml"
            path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")
        backend = FakeBackend([" Hello!"] * 9, n_ctx=n_ctx, tokenizer=chat_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer
        )
        client = TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN)
        client.__enter__()
        clients.append(client)
        command(client, "profile", "chat")
        return client, session, backend

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def sheet(client, **params) -> str:
    answer = client.get("/view/attach", params=params, headers=GET)
    assert answer.status_code == 200, answer.text
    return html.unescape(answer.text)


def checked(client, upload_id, args="") -> dict:
    return command(client, "attach", f"{upload_id} {args} check".replace("  ", " ")).json()[
        "result"
    ]["check"]


def radio(text: str, choice: str) -> str:
    found = re.search(rf'<button[^>]*data-choice="{choice}"[^>]*>', text)
    assert found, choice
    return found.group(0)


def value(text: str, element_id: str, attribute: str = "value") -> str:
    found = re.search(rf'<input[^>]*id="{element_id}"[^>]*>', text)
    assert found, element_id
    got = re.search(rf'{attribute}="([^"]*)"', found.group(0))
    assert got, attribute
    return got.group(1)


def test_a_file_that_fits_is_offered_whole_with_its_numbers(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    check = checked(client, upload_id)
    text = sheet(client, upload=upload_id)
    assert '<h2 id="attach-title"' in text and "notes.md" in text
    assert "Text · 3 lines · 32 bytes" in text
    assert SENTENCES["free"].format(free=f"{check['free']:,}", max_tokens="16") in text
    whole = radio(text, "whole")
    assert 'aria-checked="true"' in whole and 'aria-disabled="true"' not in whole
    assert f'data-args="{upload_id}"' in whole
    assert SENTENCES["whole"].format(tokens=f"{check['tokens_whole']:,}") in text
    assert SENTENCES["left"].format(left=f"{check['left_whole']:,}") in text
    assert 'aria-checked="false"' in radio(text, "part")
    assert 'data-choice="new"' not in text  # it fits here: no new chat (ruling 1)


def test_a_file_that_doesn_t_fit_preselects_the_part_that_leaves_room(make):
    client, _, _ = make(n_ctx=2000)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    check = checked(client, upload_id)
    pre, start = check["preselected"]["range"][1], check["start_that_fits"]["range"][1]
    text = sheet(client, upload=upload_id)
    whole = radio(text, "whole")
    assert 'aria-disabled="true"' in whole and 'aria-checked="false"' in whole
    assert (
        SENTENCES["doesnt_fit"].format(
            tokens=f"{check['tokens_whole']:,}", free=f"{check['free']:,}"
        )
        in text
    )
    part = radio(text, "part")
    assert 'aria-checked="true"' in part and f'data-args="{upload_id} lines 1-{pre}"' in part
    assert value(text, "attach-from") == "1" and value(text, "attach-to") == str(pre)
    assert value(text, "attach-to", "max") == str(start)  # the box's limit: the start that fits
    assert SENTENCES["filled_in"].format(units="lines", last=pre, max_tokens="16") in text
    assert f"At most lines 1 to {start} fit" in text
    assert "Excerpt: lines 1 to " in text  # the marker it will carry


def test_a_range_shows_its_own_count_and_what_it_leaves(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    chosen = checked(client, upload_id, "lines 2-3")["chosen"]
    text = sheet(client, upload=upload_id, unit="lines", first="2", last="3")
    assert 'aria-checked="true"' in radio(text, "part")
    assert f"{chosen['tokens']:,} tokens" in text
    assert SENTENCES["left"].format(left=f"{chosen['left']:,}") in text
    excerpt = SENTENCES["excerpt"].format(units="lines")
    assert excerpt in text and "Excerpt: lines 2 to 3 of 3" in text
    assert "Clarity filled in" not in text  # not the suggested range


def test_a_range_outside_the_file_says_so_in_the_part(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    text = sheet(client, upload=upload_id, unit="lines", first="2", last="9")
    refusal = "notes.md has 3 lines: give a range from 1 to 3, the first no larger than the last."
    assert refusal in text
    assert 'aria-checked="true"' in radio(text, "part")


def test_new_chat_with_this_document_when_it_fits_only_there(make):
    client, _, _ = make(n_ctx=1500)
    command(client, "say", "Made-up words in a long message. " * 28)  # about 900 tokens
    upload_id = upload(client, LONG[:700], "notes.md").json()["upload"]
    check = checked(client, upload_id)
    assert not check["fits"] and check["fits_new_chat"]
    text = sheet(client, upload=upload_id)
    assert 'data-choice="new"' in text
    assert "Starts a new chat on this profile with the whole file attached" in text
    assert SENTENCES["whole"].format(tokens=f"{check['tokens_new_chat']:,}") in text


def test_when_nothing_fits_add_waits_and_says_why(make):
    client, _, _ = make(n_ctx=120)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    text = sheet(client, upload=upload_id)
    assert SENTENCES["none_fits"].format(unit="line") in text
    add = re.search(r'<button[^>]*data-action="attach-add"[^>]*>', text)
    assert add is not None and 'aria-disabled="true"' in add.group(0)
    assert SENTENCES["choose"] in text


def test_the_extracted_text_shows_the_range_s_start_and_end(make):
    client, _, _ = make()
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    text = sheet(client, upload=upload_id, unit="lines", first="11", last="30")
    assert 'aria-label="Extracted text"' in text
    assert "Made-up line 10.\n" in text and "Made-up line 15.\n" in text  # its first six
    assert "Made-up line 16." not in text and "Made-up line 27." not in text
    assert SENTENCES["more_lines"].format(count="12") in text
    assert "Made-up line 28.\nMade-up line 29.\n" in text  # its last two


def test_the_framing_line_shows_its_source_and_how_to_change_it(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    text = sheet(client, upload=upload_id)
    assert SENTENCES["framing_label"].format(origin=SENTENCES["origin_default"]) in text
    # v0.3.2 C4: an Edit in place of the guide's link (the board's ⑦).
    assert FRAMING in text and 'data-action="framing-edit"' in text


def test_a_profile_without_a_framing_line_says_so(make):
    client, _, _ = make(extra='\n[documents]\nframing = ""\n')
    upload_id = upload(client, NOTES).json()["upload"]
    text = sheet(client, upload=upload_id)
    assert SENTENCES["no_framing"].format(origin=SENTENCES["origin_profile"]) in text


def test_a_refusal_shows_in_the_sheet(make):
    client, _, _ = make()
    upload_id = upload(client, b"\xff\xfe made-up", "old-notes.txt").json()["upload"]
    text = sheet(client, upload=upload_id)
    assert (
        "! old-notes.txt isn't UTF-8 text. Save it as PDF or as UTF-8 text and use that copy."
        in re.sub(r"<[^>]+>", "", text)
    )
    assert 'data-choice="whole"' not in text


def test_the_chips_name_each_attachment_with_a_way_to_remove_it(make):
    client, session, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    command(client, "attach", upload_id)
    tokens = session.conversation.pending[0]["tokens"]
    chips = client.get("/view/pending", headers=GET).text
    shown = html.unescape(re.sub(r"<[^>]+>", "", chips))
    assert f"notes.md · whole · {tokens:,} tokens" in shown
    assert 'data-action="detach" data-n="1" aria-label="Remove notes.md"' in chips


def test_the_sheet_and_chips_hold_hostile_text_as_text(make):
    from tests.test_web import HOSTILE

    client, _, _ = make()
    name = HOSTILE.replace("/", "")
    upload_id = upload(client, HOSTILE.encode() + b"\n</document>\n", name).json()["upload"]
    pages = [client.get("/view/attach", params={"upload": upload_id}, headers=GET).text]
    command(client, "attach", upload_id)
    pages.append(client.get("/view/pending", headers=GET).text)
    for page in pages:
        parser = Unsafe()
        parser.feed(page)
        assert parser.found == []


def test_one_too_big_sentence_names_the_file(make):
    """The board's ruling 4: the upload's 413 names the file, as the terminal's does."""
    from tests.test_web_upload import GOOD, raw

    client, _, _ = make()
    headers = {**GOOD, "Content-Length": str(64 * 1024 * 1024 + 1)}
    code, _, text = raw(client.app, headers, [b"x"], "tab=tab-1&name=big.pdf")
    assert (code, text) == (
        413,
        "big.pdf is too big: Clarity reads files up to 64 MB (67,108,864 bytes).",
    )
