"""C1 (v0.3.2): the Documents… page, drawn by /view/documents from /documents, /state and
/try, against the frozen board (notes/research/v032-boards/V32Search.dc.html, sentences
N1-N72 in V32Search-notes.md) and its rulings. Made-up documents, scratch roots."""

from __future__ import annotations

import html
import re

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import library_add, reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.pdfs import make_pdf
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


def test_the_rows_equal_documents(root, make):
    client, _ = make()
    listed = command(client, "documents").json()["result"]
    (doc,) = listed["documents"]
    fragment = page(client)
    assert "added" in doc  # the row's date (C1)
    said = row(fragment, "glossary")
    added = re.sub(r"T.*", "", doc["added"])
    year, month, day = (int(n) for n in added.split("-"))
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    assert said == (
        f"glossary glossary.md · added {day} {months[month - 1]} {year}; 2 parts, 2 with a title "
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
        "own headings; a PDF, or a text file without such lines, has none until you add them to "
        "its text.md and rebuild, and a PDF whose text.md you edit and rebuild loses its page "
        "numbers." in words
    )


def test_a_changed_row_says_why_and_offers_rebuild(root, make):
    library_add(root, make_pdf(["The wren.", "The dipper."]), "guide.pdf")
    (root / "local/documents/guide/text.md").write_text("Edited.\n", encoding="utf-8")
    client, _ = make()
    fragment = page(client)
    assert row(fragment, "guide").endswith(
        "! Changed A file in its folder has changed since Clarity cut it into parts. Rebuild If "
        "you edited text.md, its parts will lose their page numbers."
    )
    assert re.search(r'data-action="documents-rebuild" data-name="guide"', fragment)
    assert "Rebuild" not in row(fragment, "glossary")


def test_an_unreadable_row_says_why(root, make):
    (root / "local/documents/glossary/manifest.json").write_text('{"schema": 1}', "utf-8")
    client, _ = make()
    assert row(page(client), "glossary") == (
        "glossary ! Can’t be read glossary's manifest.json can't be read (it isn't a manifest "
        "Clarity writes). Delete the folder local/documents/glossary/, then add the file again."
    )


def test_rebuild_s_notice_says_when_pages_were_dropped(root, make):
    library_add(root, make_pdf(["The wren.", "The dipper."]), "guide.pdf")
    client, _ = make()
    said = text_of(page(client, rebuilt="guide", dropped="1"))
    assert (
        "Rebuilt guide from its text.md: 1 part, 0 with a title. Wrote parts.jsonl and "
        "manifest.json in local/documents/guide/. Its parts no longer have page numbers." in said
    )
    assert "page numbers." not in text_of(page(client, rebuilt="guide")).split("Wrote")[1][:120]


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


def test_a_taken_name_asks_for_another(root, make):
    client, _ = make()
    upload_id = upload(client, b"## Wren\nAgain.\n", "glossary.md").json()["upload"]
    said = text_of(page(client, upload=upload_id))
    assert "There's already a document named glossary. Give this one a different name." in said


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
    fragment = page(client, added="herons")
    assert (
        "Added herons. Clarity wrote local/documents/herons/: original.txt, text.md, parts.jsonl "
        "and manifest.json (2 parts, 2 with a title)." in text_of(fragment)
    )
    assert 'data-action="documents-search-here" data-name="herons"' in fragment
    after = text_of(page(client, searched="herons"))
    assert "This chat searches herons from your next message." in after


def test_with_no_chat_open_added_s_notice_has_no_search_button(root, make):
    client, _ = make(chat=False)
    fragment = page(client, added="glossary")
    assert "Added glossary." in text_of(fragment)
    assert "documents-search-here" not in fragment


# --- the try box


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
    # The first part's tokens hold the framing line and its wrapper (the fill's increment),
    # so the total equals the rows and N52's note isn't drawn (it shows only when they differ).
    assert found["tokens"] == part["tokens"]
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
    (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
    fragment = page(client, **{"try": "Tell me about the wren"})
    assert (
        "Clarity can't search glossary: a file in its folder has changed since Clarity cut it "
        "into parts. Rebuild cuts them again from its text.md." in text_of(fragment)
    )
    assert 'data-action="documents-rebuild" data-name="glossary"' in fragment
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
