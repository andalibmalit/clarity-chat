"""C1 (v0.3.2): the Documents… page, drawn by /view/documents from /documents, /state and
/try, against the frozen board (notes/research/v032-boards/V32Search.dc.html, sentences
N1-N72 in V32Search-notes.md) and its rulings. Made-up documents, scratch roots."""

from __future__ import annotations

import html
import re
from datetime import UTC

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import library_add, reading
from clarity_chat.session import Session, found_summary
from tests.fakes import FakeBackend
from tests.test_web import GET, ORIGIN, PORT, SECRET, command
from tests.test_web_upload import upload

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n\n## Dipper\nIt walks under water.\n"


@pytest.fixture
def make(fetched_root, chat_tokenizer):
    """A page's app: (client, session), with the chat profile's [documents] lines as asked
    and a chat open unless `chat` is False."""
    clients = []

    def make(documents: str = "", chat: bool = True):
        reading._last = None
        if documents:
            path = fetched_root / "profiles" / "chat.toml"
            text = path.read_text(encoding="utf-8")
            path.write_text(f"{text}\n[documents]\n{documents}\n", encoding="utf-8")
        backend = FakeBackend([" Hello!"] * 9, tokenizer=chat_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer
        )
        client = TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN)
        client.__enter__()
        clients.append(client)
        if chat:
            command(client, "profile", "chat")
        return client, session

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def page(client, **params) -> str:
    answer = client.get("/view/documents", params=params, headers=GET)
    assert answer.status_code == 200, answer.text
    return answer.text


def text_of(fragment: str) -> str:
    """The fragment's words as a reader sees them: tags gone, entities decoded, spaces one."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def row(fragment: str, name: str) -> str:
    found = re.search(rf'<tr data-name="{name}">.*?</tr>', fragment, re.S)
    assert found, name
    return text_of(found.group(0))


# --- the library


ADDED = "2026-11-16T02:00:00Z"  # a fixed time (the fix check's could B)


def test_the_rows_equal_documents(root, make):
    import json

    path = root / "local/documents/glossary/manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_bytes()), "added": ADDED}), "utf-8")
    client, _ = make()
    listed = command(client, "documents").json()["result"]
    (doc,) = listed["documents"]
    fragment = page(client)
    assert doc["added"] == ADDED  # the row's date (C1)
    assert row(fragment, "glossary") == (
        f"glossary glossary.md · added {web.date_words(ADDED)}; 2 parts, 2 with a title "
        f"Text {doc['bytes']} bytes 2 Ready"
    )
    words = text_of(fragment)
    assert (
        "Documents Clarity can search. A chat searches only the ones chosen in its Searching "
        "list. Clarity cuts each document into parts of at most 250 words, and a heading starts "
        "a new part." in words
    )
    assert (
        f"Each is kept in a folder of its own in {listed['folder']}. The copy of your file is "
        "named original, with .pdf, .html or .txt for its kind." in words
    )
    assert (
        "Search counts a word in a part's title twice. A part's title is the Markdown heading "
        "above it in the document's text.md: a line that starts with one to six # characters, a "
        "space and some text. A code file has no headings. A saved web page gets them from its "
        "own headings; a PDF, or a text file without such lines, has none." in words
    )


def test_an_unreadable_row_says_why(root, make):
    (root / "local/documents/glossary/manifest.json").write_text('{"schema": 1}', "utf-8")
    client, _ = make()
    assert row(page(client), "glossary") == (
        "glossary ! Can’t be read glossary's manifest.json can't be read (it isn't a manifest "
        "Clarity writes). Add the file again in Documents…, with the same name."
    )  # the close batch (surface S2): the page's way out


# --- Add


def test_add_says_where_the_copy_goes_with_the_name_box(root, make):
    client, _ = make()
    upload_id = upload(client, b"## Heron\nTall.\n", "Herons.md").json()["upload"]
    fragment = page(client, upload=upload_id)
    said = text_of(fragment)
    assert "Herons.md" in said
    assert (
        "Clarity keeps a copy in local/documents/herons/: the file, its text, its parts and a "
        "record of them." in said
    )
    assert re.search(r'<input[^>]*id="documents-name"[^>]*value="herons"', fragment)


def test_a_taken_name_says_it_replaces(root, make):  # cut 1
    client, _ = make()
    upload_id = upload(client, b"## Wren\nAgain.\n", "glossary.md").json()["upload"]
    said = text_of(page(client, upload=upload_id))
    assert "glossary is already in your documents. Adding this file replaces its copy" in said


def test_a_name_clarity_can_t_make_offers_no_replace(root, make):
    client, _ = make()
    upload_id = upload(client, b"Made-up.\n", "日本語.md").json()["upload"]
    assert "documents-replace" not in page(client, upload=upload_id)


def test_a_file_clarity_can_t_name_asks_for_one_with_the_rule(root, make):
    client, _ = make()
    upload_id = upload(client, b"Made-up.\n", "日本語.md").json()["upload"]
    said = text_of(page(client, upload=upload_id))
    assert "Clarity can't make a document name from 日本語.md. Give it one." in said
    assert (
        "A document's name is 1 to 64 characters: a to z, 0 to 9, - and _, starting with a "
        "letter or a digit." in said
    )


def test_added_s_notice_offers_search_it_in_this_chat(root, make):
    client, session = make()
    upload_id = upload(client, b"## Heron\nTall.\n\n## Egret\nWhite.\n", "herons.md").json()[
        "upload"
    ]
    command(client, "documents", f"add {upload_id} herons")
    fragment = page(client, notice="1")  # the page's own Add (part 2 review's S2)
    assert (
        "Added herons. Clarity wrote local/documents/herons/: original.txt, text.md, parts.jsonl "
        "and manifest.json (2 parts, 2 with a title)." in text_of(fragment)
    )
    assert 'data-action="documents-search-here" data-name="herons"' in fragment
    after = text_of(page(client, searched="herons"))
    assert "This chat searches herons from your next message." in after


def test_with_no_chat_open_added_s_notice_has_no_search_button(root, make):
    client, _ = make(chat=False)
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    command(client, "documents", f"add {upload_id} herons")
    fragment = page(client, notice="1")  # the page's own Add (part 2 review's S2)
    assert "Added herons." in text_of(fragment)
    assert "documents-search-here" not in fragment


def test_added_s_notice_has_no_search_button_when_the_chat_searches_it(root, make):  # could 5
    import shutil

    client, _ = make()
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    command(client, "documents", f"add {upload_id} herons")
    command(client, "search", "herons")
    shutil.rmtree(root / "local/documents/herons")  # gone, and still in the chat's list
    page(client)  # the first Added notice is drawn and spent
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    command(client, "documents", f"add {upload_id} herons")
    fragment = page(client, notice="1")  # the page's own Add (part 2 review's S2)
    assert "Added herons." in text_of(fragment)
    assert "documents-search-here" not in fragment


# --- the try box


def test_the_try_box_in_a_text_mode_chat(root, make):  # should 6
    client, _ = make()
    command(client, "profile", "text")
    assert "Searching needs a chat-mode profile; text is in text mode." in text_of(page(client))


def test_the_try_box_with_no_chat_open(root, make):
    client, _ = make(chat=False)
    fragment = page(client)
    assert (
        "Open a chat to try a search. A try searches that chat's documents and counts tokens for "
        "its profile." in text_of(fragment)
    )
    assert re.search(r'<input[^>]*id="documents-try"[^>]*disabled', fragment)


def test_the_try_box_in_a_chat_that_searches_nothing(root, make):
    client, _ = make()
    assert "This chat searches no documents. Choose one in Searching to try a search." in text_of(
        page(client)
    )


def test_the_try_box_with_an_empty_library(root, make):
    import shutil

    shutil.rmtree(root / "local/documents/glossary")
    client, _ = make()
    assert (
        'No documents yet. Add a file, or press the paperclip in a chat and choose "Search it".'
        in text_of(page(client))
    )


def test_the_try_box_s_result_equals_try(root, make):
    client, _ = make('search = ["glossary"]')
    said = text_of(page(client))
    assert (
        "Type a message. Clarity shows the parts it would add if you sent it next in this chat, "
        "which searches glossary. Token counts are for this chat's profile, chat. Nothing is "
        "sent to the model, and nothing is logged." in said
    )
    found = command(client, "try", "Tell me about the wren").json()["result"]["found"]
    (part,) = found["parts"]
    tried = text_of(page(client, **{"try": "Tell me about the wren"}))
    assert f"1 part · {found['tokens']:,} tokens" in tried
    assert f'1 glossary · "Wren" {part["tokens"]:,} in its title: wren; in its text: wren' in tried
    # The first part's count holds the framing line (the fill's increment): said once under
    # the list (the C1 tripwire review's should 3), with no separate framing note.
    assert found["tokens"] == part["tokens"]
    assert "The first part's count includes the line Clarity puts before the parts." in tried
    assert "The total also counts" not in tried
    assert (
        "Words searched: wren. Ignored (common words and single letters or digits): tell, me, "
        "about, the." in tried
    )


@pytest.mark.parametrize(
    ("message", "said"),
    [
        ("what is it?", "Every word in your message is a common word or a single letter or "
         "digit, so nothing was searched and nothing was added."),
        ("🙂", "Your message has no words to search for, so nothing was added."),
        ("owls", 'No part of glossary has any of the words searched, so nothing was added. Search '
         'matches words and close forms of them ("sing" finds "singing"), not meaning.'),
    ],
)  # fmt: skip
def test_the_try_box_s_empty_states(root, make, message, said):
    client, _ = make('search = ["glossary"]')
    assert said in text_of(page(client, **{"try": message}))


def test_the_try_box_s_refusal_has_its_buttons(root, make):
    client, _ = make('search = ["glossary"]')
    (root / "local/documents/glossary/parts.jsonl").write_text("Edited.\n", encoding="utf-8")
    fragment = page(client, **{"try": "Tell me about the wren"})
    assert (
        "Clarity can't search glossary: its parts.jsonl isn't the one Clarity built. Add the "
        "file again in Documents…, with the same name." in text_of(fragment)
    )
    assert 'data-action="documents-stop" data-name="glossary"' in fragment


def test_a_control_token_s_refusal_shows_under_the_parts(root, make):
    library_add(root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    client, _ = make('search = ["odd"]')
    said = text_of(page(client, **{"try": "Tell me about the wren"}))
    assert (
        'Part 1 of odd ("Wren") holds </s>, which this model reads as a control token, not '
        "text." in said
    )


# --- the C1 tripwire review's musts


@pytest.mark.parametrize("message", ["", "   "])
def test_an_empty_try_runs_nothing_and_says_no_command(root, make, message):  # must 1
    client, _ = make('search = ["glossary"]')
    fragment = page(client, **{"try": message})
    assert "Usage" not in fragment and "/try" not in fragment
    assert 'class="parts"' not in fragment and 'role="alert">' not in fragment.split("try")[-1]
    assert 'class="problem"' not in fragment  # the fix check's could D


@pytest.mark.parametrize("added", ["yesterday", "2026-13-01T10:00:00Z", "2026-00-05T10:00:00Z"])
def test_a_manifest_with_a_bad_date_is_one_unreadable_row(root, make, added):  # must 2
    import json

    from clarity_chat.documents import library_state

    library_add(root, b"## Heron\nTall.\n", "herons.md")
    path = root / "local/documents/herons/manifest.json"
    manifest = json.loads(path.read_bytes())
    path.write_text(json.dumps({**manifest, "added": added}), encoding="utf-8")
    assert library_state(root, "herons") == "unreadable"
    client, _ = make()
    answer = client.get("/view/documents", headers=GET)
    assert answer.status_code == 200, answer.text
    assert row(answer.text, "herons").startswith("herons ! Can’t be read herons's manifest.json")
    assert row(answer.text, "glossary").endswith("Ready")


# --- the C1 tripwire review's shoulds and coulds: found_view's branches, the refusal's buttons


def made_up_part(rank: int, **extra) -> dict:
    return {"id": f"g:{rank}", "document": "g", "ordinal": rank, "title": None, "page": None,
            "rank": rank, "title_words": [], "words": ["wren"], "tokens": 50, **extra}  # fmt: skip


def made_up_found(parts: list[dict], left: list[tuple[int, str]], matched: int) -> dict:
    """/history's summary (found_view's input since C3) of a made-up `found`."""
    return found_summary({"words": ["wren"], "ignored": [], "matched": matched,
            "considered": 8, "budget": 1500, "tokens": sum(p["tokens"] for p in parts),
            "parts": parts, "documents": [{"name": "g"}],
            "left_out": [{"id": f"g:{r}", "rank": r, "score": -1.0, "why": why}
                         for r, why in left]}, framed=True)  # fmt: skip


def test_found_view_s_notes_and_already_rows():  # should 2
    found = made_up_found([made_up_part(1)], [(2, "already in the chat"), (3, "budget"),
                          (4, "budget")], 12)  # fmt: skip
    view = web.found_view(found)
    assert view["summary"] == "1 part · 50 tokens · 1 more already in the chat"
    assert [(r["rank"], r["label"], r["why"]) for r in view["rows"]] == [
        (1, "g · part 1", "in its text: wren"),
        (2, "g · part 2", "already in the chat: not added again"),
    ]
    assert view["notes"] == [
        "2 more matched but were left out: Clarity adds parts in order and stops at the first "
        "that doesn't fit the budget of 1,500 tokens.",
        "4 more matched but weren't considered: Clarity considers only the best 8 matches.",
        "The first part's count includes the line Clarity puts before the parts.",
    ]


def test_found_view_s_nothing_new_and_nothing_fit():  # should 2
    new = web.found_view(made_up_found([], [(1, "already in the chat")], 1))
    assert new["summary"] == "nothing new · 1 already in the chat"
    fit = web.found_view(made_up_found([], [(1, "budget"), (2, "budget")], 2))
    assert fit["summary"] == "nothing added · 2 didn't fit the budget"
    assert fit["notes"][0].startswith("2 matched but were left out:")


def test_already_and_budget_together_say_nothing_fit():  # the part 2 runner's #309
    found = made_up_found([], [(1, "already in the chat"), (2, "budget")], 2)
    assert web.found_view(found)["summary"] == "nothing added · 1 didn't fit the budget"


def test_a_part_s_label_names_its_page():  # should 2
    assert web.part_label(made_up_part(5, page=3)) == "g · page 3 · part 5"
    assert web.part_label(made_up_part(5, page=3, title="Wren")) == 'g · "Wren", page 3'


def test_a_refusal_s_buttons_follow_its_problem():  # could 5
    lines = web.refusal_lines({"error": "x", "problems": [
        {"document": "a", "state": "missing"},
        {"document": "b", "state": "unreadable", "why": "b's parts.jsonl isn't one Clarity "
         "writes.", "add_again": True},
        {"state": "no_fts5"},
    ]})  # fmt: skip
    assert [line["stop"] for line in lines] == ["a", "b", None]


# --- the C1 fix check's should A: when the first part's count holds the framing line


FIRST_COUNT = "The first part's count includes the line Clarity puts before the parts."


def test_no_framing_line_no_first_count_sentence(root, make):
    client, _ = make('search = ["glossary"]\nframing = ""')
    tried = text_of(page(client, **{"try": "Tell me about the wren"}))
    assert 'glossary · "Wren"' in tried and FIRST_COUNT not in tried


def test_an_attached_message_s_framing_line_is_the_attachment_s(root, make):
    client, _ = make('search = ["glossary"]')
    upload_id = upload(client, b"Made-up notes.\n", "notes.md").json()["upload"]
    assert "attached" in command(client, "attach", upload_id).json()["result"]
    tried = text_of(page(client, **{"try": "Tell me about the wren"}))
    assert 'glossary · "Wren"' in tried and FIRST_COUNT not in tried


def test_the_row_s_date_is_the_local_one():  # could 7, with a zone (could B)
    from datetime import timedelta, timezone

    assert web.date_words(ADDED, UTC) == "16 Nov 2026"
    assert web.date_words(ADDED, timezone(timedelta(hours=-5))) == "15 Nov 2026"
