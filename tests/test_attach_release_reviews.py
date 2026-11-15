"""v0.3.1's release reviews, the server side: the final fact-check's FIX rows and its three
unsourced claims, and the design walkthrough's findings (F1 to F25) and ruling 2, as the
lead ruled. The sentences are asserted as literals: the table is held equal to them.
Made-up documents only."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from clarity_chat.documents import SENTENCES, reading
from clarity_chat.documents.reading import extract
from clarity_chat.session import ContextError
from tests.pdfs import aes_pdf, make_pdf, shared_stream_pdf
from tests.test_documents_extract import stub
from tests.test_session_attach import chat_session, doc
from tests.test_web import GET, command
from tests.test_web_attach import LONG, NOTES, make, sheet
from tests.test_web_upload import upload

__all__ = ["make"]  # the fixture, used by name

REPO = Path(__file__).resolve().parents[1]

NEW = {
    "free": "{free} tokens free in this chat, after setting aside {max_tokens} for the reply "
    "(your max_tokens).",
    "whole": "It takes up {tokens} tokens in every later prompt of this chat.",
    "left": "That leaves {left} tokens for your messages and the replies.",
    "reply_share": "Each later reply can use up to {max_tokens} of them (your max_tokens).",
    "filled_in": "Clarity filled in {units} 1 to {last}: the largest start it found that "
    "leaves at least {max_tokens}.",
    "at_most": "At most {units} 1 to {last} fit, leaving {left}.",
    "longer_chat": "For a longer chat, attach fewer {units}.",
    "pdf_pages": "Pages are counted from the file's first page, not the numbers printed on them.",
    "excerpt": "Only the {units} you choose are sent, after a line that says they're an excerpt.",
    "framing_label": "Clarity introduces the document to the model with this line ({origin}):",
    "marker_label": "The excerpt starts with this line, so the model knows it's partial "
    "({origin}):",
    "control_tokens": "Holds text the tokenizer reads as control tokens ({tokens}); it's sent "
    "as it is, and the call is flagged. The model may read them as its own markers, such as "
    "the end of a message.",
    "keeps": "Keeps: {names}. An edit can't add or remove attachments.",
    "stays_waiting": "{names} stays waiting for your next message.",
    "stay_waiting": "{names} stay waiting for your next message.",
    "chip_tokens": "{tokens} tokens when attached",
    "attachments_caption": "Your message's line counts each attachment as it cost when it was "
    "attached: its block, the blank lines Clarity puts around it (not highlighted here) and, "
    "for a message's first, the framing line.",
    "highlight_added": "Highlighted: the framing line and documents Clarity put in your messages.",
    "too_long": "It's {over} tokens too long.",
    "stop_advice": "Shorten your message, start a new chat, or lower max_tokens.",
    "remove_waiting": "Shorten your message, remove an attachment and attach a shorter part, "
    "start a new chat, or lower max_tokens.",
    "slow_read": "A long file can take a minute to read. Clarity stops at {seconds} seconds.",
    "too_slow": "{name} took longer than {seconds} seconds to read. Attach a smaller file, or "
    "split it and attach the part you need.",
    "too_much_memory": "{name} took more than {memory} of memory to read. Attach a smaller "
    "file, or split it and attach the part you need.",
    "protected": "{name} is protected, even if it opens without a password. Export or print "
    "it to a new PDF and attach that.",
    "no_text": "{name} has no text Clarity can read. Its pages may be scanned images, which "
    "Clarity doesn't read. Use a PDF app's text recognition (OCR) to add the text, then "
    "attach that copy.",
}


def test_the_table_holds_the_reviews_sentences():
    assert {k: SENTENCES.get(k) for k in NEW} == NEW


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", page))


# --- the stop: the fact-check's FIX 1, F8, ruling 2 (a setting as typed)


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


def test_the_stop_says_by_how_much_and_what_to_do(chat):
    session, backend = chat
    session.set_setting("max_tokens", "1024")
    backend._n_ctx = 6000
    session.attach(doc("Made-up words. " * 300))  # 4,500 tokens: one per character
    backend._n_ctx = 4000
    with pytest.raises(ContextError) as e:
        session.say("a")
    said = str(e.value)
    n = int(re.search(r"The prompt is ([0-9,]+) tokens", said).group(1).replace(",", ""))  # ty: ignore
    assert "; with max_tokens 1024 it doesn't fit in the context of 4,000 tokens." in said
    assert f"Nothing was sent. It's {n + 1024 + 1 - 4000:,} tokens too long." in said
    assert said.endswith(
        "Shorten your message, remove an attachment and attach a shorter part, start a new "
        "chat, or lower max_tokens."
    )


def test_the_stop_without_this_message_s_attachments_names_the_other_steps(chat):
    session, backend = chat
    backend._n_ctx = 6000
    session.attach(doc("Made-up words. " * 300))
    session.say("a")
    backend._n_ctx = 4000
    with pytest.raises(ContextError) as e:
        session.say("b")
    assert str(e.value).endswith("Shorten your message, start a new chat, or lower max_tokens.")


# --- the docs: the fact-check's FIX 2 to 6 and its three unsourced claims


def test_the_design_doc_s_bound_names_its_premise_and_its_sources():
    design = (REPO / "docs/design.md").read_text(encoding="utf-8")
    assert "For a byte-level tokenizer with no normalizer, no token covers more text" in design
    assert "the chat models Clarity ships use (SmolLM2, OLMo 3, Llama 3.1)" in design
    assert "as each of Clarity's models uses" not in design
    assert "later search results" not in design
    assert "is built so that later sources of text can use it" in design
    assert "shared_stream_pdf" in design  # the 7 KB PDF's source
    assert "11.9 s" in design and "3.9 GB" in design  # the 128K figure's source


def test_the_guide_and_the_changelog_say_only_what_holds():
    guides = (REPO / "docs/guides.md").read_text(encoding="utf-8")
    assert "the largest start it found that still leaves at least `max_tokens`" in guides
    assert NEW["pdf_pages"] in guides  # F2
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "every control either does something" not in changelog


def test_the_design_doc_s_small_pdf_is_the_fixture_it_names():
    data = shared_stream_pdf(40, 256 * 1024)
    assert len(data) < 8 * 1024
    assert len(extract(data, "shared.pdf").text) >= 10_000_000


# --- the sheet: F1, F2, F4, F5, F14, F17, F19, F22


def part_notes(text: str) -> str:
    found = re.search(r'<div id="attach-part-note">(.*?)</div>', text, re.S)
    assert found, "the part's notes"
    return text_of(found.group(1))


def test_the_suggested_range_says_what_it_buys(make):
    client, _, _ = make(n_ctx=2000)
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    check = command(client, "attach", f"{upload_id} check").json()["result"]["check"]
    pre, start = check["preselected"], check["start_that_fits"]
    chosen = command(client, "attach", f"{upload_id} lines 1-{pre['range'][1]} check").json()
    left = chosen["result"]["check"]["chosen"]["left"]
    notes = part_notes(sheet(client, upload=upload_id))
    assert (
        f"That leaves {left:,} tokens for your messages and the replies. Each later reply can "
        "use up to 16 of them (your max_tokens)."
    ) in notes
    assert (
        f"Clarity filled in lines 1 to {pre['range'][1]}: the largest start it found that "
        "leaves at least 16."
    ) in notes
    assert f"At most lines 1 to {start['range'][1]} fit, leaving {start['left']:,}." in notes
    assert "For a longer chat, attach fewer lines." in notes
    typed = part_notes(sheet(client, upload=upload_id, unit="lines", first="2", last="5"))
    assert "Clarity filled in" not in typed and "At most lines 1 to" in typed


def test_a_pdf_s_range_says_how_pages_are_counted(make):
    client, _, _ = make()
    upload_id = upload(client, make_pdf(["Made-up one.", "Made-up two."]), "book.pdf").json()
    text = text_of(sheet(client, upload=upload_id["upload"]))
    assert NEW["pdf_pages"] in text
    notes_id = upload(client, NOTES).json()["upload"]
    assert NEW["pdf_pages"] not in text_of(sheet(client, upload=notes_id))


def test_the_sheet_s_words_follow_the_walkthrough(make):
    client, _, _ = make()
    command(client, "set", "max_tokens 1024")
    upload_id = upload(client, b"Made-up </s> text.\n" * 3, "odd.md").json()["upload"]
    check = command(client, "attach", f"{upload_id} check").json()["result"]["check"]
    text = text_of(sheet(client, upload=upload_id))
    free = f"{check['free']:,}"
    assert f"{free} tokens free in this chat, after setting aside 1,024 for the reply" in text
    assert f"It takes up {check['tokens_whole']:,} tokens in every later prompt of this chat." in (
        text
    )
    assert "Only the lines you choose are sent, after a line that says they're an excerpt." in text
    assert "Clarity introduces the document to the model with this line (Clarity's default):" in (
        text
    )
    assert "The model may read them as its own markers, such as the end of a message." in text
    add = re.search(
        r'<button[^>]*data-action="attach-add"[^>]*>([^<]*)<', sheet(client, upload=upload_id)
    )
    assert add and add.group(1) == "Add to message"  # F17


def test_the_marker_s_label_says_what_it_does(make):
    client, _, _ = make()
    upload_id = upload(client, LONG, "long.md").json()["upload"]
    text = text_of(sheet(client, upload=upload_id, unit="lines", first="2", last="5"))
    assert (
        "The excerpt starts with this line, so the model knows it's partial (Clarity's default):"
        in text
    )


def test_the_reading_state_s_slow_line_comes_from_the_table(make):
    client, _, _ = make()
    shell = client.get("/", headers=GET).text
    assert (
        'data-slow="A long file can take a minute to read. Clarity stops at 120 seconds."' in shell
    )


# --- the refusals: F11, F25


@pytest.mark.parametrize(
    ("name", "data", "setup", "sentence"),
    [
        ("aes.pdf", aes_pdf(["Made-up."]), None, NEW["protected"].format(name="aes.pdf")),
        ("scan.pdf", make_pdf(["", ""]), None, NEW["no_text"].format(name="scan.pdf")),
        (
            "slow.md",
            NOTES,
            "import time; time.sleep(60)",
            "slow.md took longer than 1 seconds to read. Attach a smaller file, or split it and "
            "attach the part you need.",
        ),
        (
            "huge.md",
            NOTES,
            "import sys; sys.exit(70)",
            "huge.md took more than 1 GB of memory to read. Attach a smaller file, or split it "
            "and attach the part you need.",
        ),
    ],
)
def test_the_refusals_say_what_to_do(make, monkeypatch, name, data, setup, sentence):
    client, _, _ = make()
    if setup:
        stub(monkeypatch, setup)
        monkeypatch.setattr(reading, "TIME_LIMIT", 1.0)
    upload_id = upload(client, data, name).json()["upload"]
    assert f"! {sentence}" in text_of(sheet(client, upload=upload_id))


def test_a_refusal_offers_another_file(make):
    client, _, _ = make()
    upload_id = upload(client, b"\xff\xfe made-up", "old-notes.txt").json()["upload"]
    page = sheet(client, upload=upload_id)
    assert 'data-action="attach-pick">Choose another file</button>' in page  # F23


# --- the thread, Edit, the chips and Details: F3, F10, F24


def sent(client, data=NOTES, name="notes.md", message="What's in it?") -> None:
    upload_id = upload(client, data, name).json()["upload"]
    command(client, "attach", upload_id)
    command(client, "say", message)


def test_edit_says_it_can_t_change_attachments_and_which_stay_waiting(make):
    client, _, _ = make()
    sent(client)
    command(client, "attach", upload(client, b"Made-up guide.\n", "guide.md").json()["upload"])
    thread = client.get("/view/thread", headers=GET).text
    form = re.search(r'<form class="editbox".*?</form>', thread, re.S)
    assert form
    words = text_of(form.group(0))
    assert "Keeps: notes.md. An edit can't add or remove attachments." in words
    assert "guide.md stays waiting for your next message." in words


def test_a_chip_says_its_tokens_are_the_cost_when_attached(make):
    client, session, _ = make()
    command(client, "attach", upload(client, NOTES).json()["upload"])
    tokens = session.conversation.pending[0]["tokens"]
    chips = text_of(client.get("/view/pending", headers=GET).text)
    assert f"notes.md · whole · {tokens:,} tokens when attached" in chips


def test_details_counts_each_block_lists_the_framing_line_and_explains_the_thread_s_figure(make):
    client, _, _ = make()
    sent(client)
    call = command(client, "prompt").json()["result"]["call"]
    added = [a for e in call["message_spans"] for a in e.get("added", [])]
    framing, block = (a["tokens"] for a in added)
    page = text_of(client.get("/view/details", params={"call": call["call_id"]}, headers=GET).text)
    assert f"{block[0]:,} to {block[1] - 1:,} · {block[1] - block[0]:,} tokens" in page
    assert "Framing line" in page
    assert f"{framing[0]:,} to {framing[1] - 1:,} · {framing[1] - framing[0]:,} tokens" in page
    assert NEW["attachments_caption"] in page
    assert NEW["highlight_added"] in page
