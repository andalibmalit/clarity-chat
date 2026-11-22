"""Checkpoint 3 (v0.3.1 C1): the attach sheet's server side. Its musts 3 and 7, the server
half of shoulds 4 to 6, should 10's untested lines, and coulds 11 and 12. Made-up
documents and made-up TOML only."""

from __future__ import annotations

import html
import re

import pytest

from clarity_chat.documents import SENTENCES, DocumentRefused, reading
from tests.pdfs import aes_pdf, encrypted_pdf
from tests.test_documents_extract import stub
from tests.test_web import GET, Unsafe, command
from tests.test_web_attach import LONG, NOTES, checked, make, radio, sheet, value
from tests.test_web_upload import upload

__all__ = ["make"]  # the fixture, used by name

MARKER = '\nexcerpt_marker = "Made-up part: {units} {first} to {last}."\n'  # chat's [documents]


def add_button(text: str) -> str:
    found = re.search(r'<button[^>]*data-action="attach-add"[^>]*>', text)
    assert found
    return found.group(0)


def part_notes(text: str) -> str:
    found = re.search(r'<div id="attach-part-note">(.*?)</div>', text, re.S)
    assert found, "the part's notes"
    return found.group(1)


# --- must 3: the excerpt sentence follows the profile's marker setting


def test_a_range_that_covers_the_file_still_says_excerpts_carry_a_marker(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    text = sheet(client, upload=upload_id, unit="lines", first="1", last="3")
    excerpt = SENTENCES["excerpt"].format(units="lines")
    assert excerpt in text and SENTENCES["no_marker"] not in text


def test_a_profile_with_no_marker_says_so_before_any_range(make):
    client, _, _ = make(extra='\nexcerpt_marker = ""\n')  # into the chat profile's [documents]
    upload_id = upload(client, NOTES).json()["upload"]
    assert checked(client, upload_id)["excerpt_marker_origin"] == "none"
    text = sheet(client, upload=upload_id)
    assert SENTENCES["no_marker"] in text and "after a line that says" not in text


def test_the_marker_label_names_the_profile_s_own_marker(make):
    client, _, _ = make(extra=MARKER)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    assert checked(client, upload_id)["excerpt_marker_origin"] == "profile"
    text = sheet(client, upload=upload_id, unit="lines", first="2", last="5")
    assert SENTENCES["marker_label"].format(origin=SENTENCES["origin_profile"]) in text
    assert "Made-up part: lines 2 to 5." in text


def test_the_default_marker_is_labelled_clarity_s(make):
    client, _, _ = make()
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    assert checked(client, upload_id)["excerpt_marker_origin"] == "default"
    text = sheet(client, upload=upload_id, unit="lines", first="2", last="5")
    assert SENTENCES["marker_label"].format(origin=SENTENCES["origin_default"]) in text


# --- must 7: a failed preselect read is drawn as its refusal, not as no preselection


def test_a_failed_preselect_read_shows_its_refusal(make, monkeypatch):
    client, session, _ = make(n_ctx=2000)
    real = session.check

    def check(doc, unit=None, first=0, last=0, cancel=None, new_profile=None, aside=0):  # noqa: ANN001, ANN202
        if unit is not None:  # the second read, for the preselected part
            raise DocumentRefused("Stopped reading long.md.")
        return real(doc, unit, first, last, cancel, new_profile, aside)

    monkeypatch.setattr(session, "check", check)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    text = sheet(client, upload=upload_id)
    assert "Not attached" in text and "! Stopped reading long.md." in re.sub(r"<[^>]+>", "", text)
    assert 'role="radiogroup"' not in text


# --- should 4: the boxes' opening values, and what Add's description points at


def test_with_nothing_preselected_the_boxes_open_on_one_to_one(make):
    client, _, _ = make(n_ctx=235)  # a start fits, but none leaves max_tokens
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    check = checked(client, upload_id)
    assert check["start_that_fits"] is not None and check["preselected"] is None
    text = sheet(client, upload=upload_id)
    assert (value(text, "attach-from"), value(text, "attach-to")) == ("1", "1")
    assert 'data-counted="false"' in radio(text, "part")


def test_a_file_that_fits_opens_the_boxes_on_its_preselected_end(make):
    client, _, _ = make(n_ctx=3680)  # it fits, with less than max_tokens left
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    check = checked(client, upload_id)
    pre = check["preselected"]["range"][1]
    assert check["fits"] and pre < check["total"]
    text = sheet(client, upload=upload_id)
    assert (value(text, "attach-from"), value(text, "attach-to")) == ("1", str(pre))


def test_add_is_described_by_the_chosen_option_s_note(make):
    client, _, _ = make()
    upload_id = upload(client, NOTES).json()["upload"]
    text = sheet(client, upload=upload_id)
    assert 'aria-describedby="attach-whole-note"' in add_button(text)
    assert 'data-note="attach-part-note"' in radio(text, "part")
    ranged = sheet(client, upload=upload_id, unit="lines", first="2", last="3")
    assert 'aria-describedby="attach-part-note"' in add_button(ranged)
    assert 'data-counted="true"' in radio(ranged, "part")
    assert SENTENCES["left"].split("{")[0] in part_notes(ranged)


def test_with_nothing_chosen_add_is_described_by_choose(make):
    client, _, _ = make(n_ctx=120)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    assert 'aria-describedby="attach-why"' in add_button(sheet(client, upload=upload_id))


# --- should 5: in text mode, /attach alone refuses with the session's sentence


def test_attach_alone_refuses_in_text_mode(make):
    client, _, _ = make()
    command(client, "profile", "text")
    answer = command(client, "attach", "").json()
    assert answer["error"] == "Attaching needs a chat-mode profile; text is in text mode."


# --- should 6: the preview is capped, each line and so the whole


def test_a_one_megabyte_line_is_cut_in_the_preview(make):
    client, _, _ = make()
    line = b"Made-up words go on. " * 50_000  # 1,050,000 bytes, one line
    upload_id = upload(client, line + b"\n", "one-line.md").json()["upload"]
    answer = client.get("/view/attach", params={"upload": upload_id}, headers=GET)
    assert answer.status_code == 200 and len(answer.text) < 40_000
    more = SENTENCES["more_characters"].format(count=f"{len(line) - 2000:,}")
    assert more in html.unescape(answer.text)


# --- should 10: the sheet's lines that no test asserted


def test_the_sheet_warns_of_control_tokens_and_the_closing_tag(make):
    client, _, _ = make()
    data = b"Made-up text with </s> in it.\n</document>\n"
    upload_id = upload(client, data, "odd.md").json()["upload"]
    text = sheet(client, upload=upload_id)
    assert "! " + SENTENCES["control_tokens"].format(tokens="</s>") in text
    assert "! " + SENTENCES["closing_tag"] in text


def test_the_sheet_says_a_file_is_already_in_this_chat(make):
    client, _, _ = make()
    first = upload(client, NOTES).json()["upload"]
    command(client, "attach", first)
    again = upload(client, NOTES).json()["upload"]
    assert SENTENCES["already"] in sheet(client, upload=again)


def test_new_chat_names_the_waiting_chips_it_drops_and_its_own_numbers(make):
    client, _, _ = make(n_ctx=1500)
    command(client, "say", "Made-up words in a long message. " * 28)  # about 900 tokens
    command(client, "attach", upload(client, b"Made-up.\n", "small.md").json()["upload"])
    upload_id = upload(client, LONG[:700], "notes.md").json()["upload"]
    check = checked(client, upload_id)
    assert not check["fits"] and check["fits_new_chat"]
    text = sheet(client, upload=upload_id)
    assert SENTENCES["not_kept"].format(names="small.md") in text
    assert SENTENCES["whole"].format(tokens=f"{check['tokens_new_chat']:,}") in text
    assert SENTENCES["left"].format(left=f"{check['left_new_chat']:,}") in text
    assert 'data-note="attach-new-note"' in radio(text, "new")


def test_a_chosen_range_that_doesn_t_fit_waits_with_its_reason(make):
    client, _, _ = make(n_ctx=2000)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    chosen = checked(client, upload_id, "lines 1-150")["chosen"]
    assert chosen["left"] is None or chosen["left"] < 0
    text = sheet(client, upload=upload_id, unit="lines", first="1", last="150")
    part = radio(text, "part")
    assert 'aria-checked="true"' in part and 'data-ok="false"' in part
    assert SENTENCES["doesnt_fit"].split("{")[0] in part_notes(text)
    assert 'class="ct"' not in text


def test_the_facts_line_names_empty_pages_and_encryption(make):
    client, _, _ = make()
    data = encrypted_pdf(["Made-up page.", "", "Made-up again."])
    upload_id = upload(client, data, "locked-lightly.pdf").json()["upload"]
    text = sheet(client, upload=upload_id)
    facts = "PDF · 3 pages, " + SENTENCES["empty_units"].format(count="1") + ", "
    assert facts + SENTENCES["encrypted"] + " · " in text


def test_the_preselected_sentence_uses_separators(make):
    client, _, _ = make(n_ctx=4000)
    command(client, "set", "max_tokens 1024")
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    assert checked(client, upload_id)["preselected"] is not None
    page = html.unescape(sheet(client, upload=upload_id))
    assert "the largest start it found that leaves at least 1,024." in page


@pytest.mark.parametrize(
    ("name", "data", "setup", "sentence"),
    [
        ("aes.pdf", aes_pdf(["Made-up."]), None, "aes.pdf is protected, even if"),
        ("slow.md", NOTES, "import time; time.sleep(60)", "slow.md took longer than 1 seconds"),
        ("huge.md", NOTES, "import sys; sys.exit(70)", "huge.md took more than 1 GB of memory"),
    ],
)
def test_the_reading_limits_show_in_the_sheet(make, monkeypatch, name, data, setup, sentence):
    client, _, _ = make()
    if setup:
        stub(monkeypatch, setup)
        monkeypatch.setattr(reading, "TIME_LIMIT", 1.0)
    upload_id = upload(client, data, name).json()["upload"]
    text = re.sub(r"<[^>]+>", "", sheet(client, upload=upload_id))
    assert "Not attached" in text and f"! {sentence}" in text


def test_hostile_text_with_a_direction_override_and_a_long_line_stays_text(make):
    from tests.test_web import HOSTILE

    client, _, _ = make()
    data = b"<b>" * 350_000 + "\n\u202eMade-up reversed.\n".encode() + HOSTILE.encode()
    upload_id = upload(client, data, "hostile.md").json()["upload"]
    page = client.get("/view/attach", params={"upload": upload_id}, headers=GET).text
    parser = Unsafe()
    parser.feed(page)
    assert parser.found == [] and len(page) < 60_000
    assert "⟨U+202E⟩Made-up reversed." in html.unescape(page)  # shown, never reordering (QA 1)


# --- could 12: the chips draw a range with an en dash


def test_a_chip_draws_its_range_with_an_en_dash(make):
    client, _, _ = make()
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    command(client, "attach", f"{upload_id} lines 2-5")
    chips = html.unescape(re.sub(r"<[^>]+>", "", client.get("/view/pending", headers=GET).text))
    assert "long.md · lines 2–5 of 200 ·" in chips


# --- could 13: the chips are a list


def test_the_chips_are_list_items(make):
    client, _, _ = make()
    command(client, "attach", upload(client, NOTES).json()["upload"])
    chips = client.get("/view/pending", headers=GET).text
    assert '<span class="chip" role="listitem">' in chips


# --- the lead's ruling on point 2: "Stopped reading NAME." has one source, SENTENCES


def test_stopped_reading_comes_from_the_table_on_the_page_and_in_the_reader(make):
    from clarity_chat import web

    client, _, _ = make()
    shell = client.get("/", headers=GET).text
    assert 'data-stopped="Stopped reading {name}."' in shell
    script = (web.STATIC / "app.js").read_text(encoding="utf-8")
    assert "Stopped reading" not in script  # the page reads it from the shell
    assert SENTENCES.get("stopped") == "Stopped reading {name}."
    index = (web.STATIC / "index.html").read_text(encoding="utf-8")
    assert "Stopped reading" not in index  # the server fills it in
