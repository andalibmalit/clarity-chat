"""C2 (v0.3.2): "Search it" on the attach sheet, the Searching chooser beside the picker,
Edit's searched line, and the owner's Q12 Replace, against the frozen board (① and its
sentences N1-N28, N60, N71) and its rulings. Made-up documents, scratch roots."""

from __future__ import annotations

import html
import re
import shutil

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import library_add, library_documents, reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import GET, ORIGIN, PORT, SECRET, command
from tests.test_web_upload import upload

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n\n## Dipper\nIt walks under water.\n"


@pytest.fixture
def make(fetched_root, chat_tokenizer):
    clients = []

    def make(documents: str = "", n_ctx=None, profile: str = "chat"):
        reading._last = None
        if documents:
            path = fetched_root / "profiles" / "chat.toml"
            text = path.read_text(encoding="utf-8")
            path.write_text(f"{text}\n[documents]\n{documents}\n", encoding="utf-8")
        backend = FakeBackend([" Hello!"] * 9, n_ctx=n_ctx, tokenizer=chat_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer
        )
        client = TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN)
        client.__enter__()
        clients.append(client)
        command(client, "profile", profile)
        return client, session

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def get(client, name: str, **params) -> str:
    answer = client.get(f"/view/{name}", params=params, headers=GET)
    assert answer.status_code == 200, answer.text
    return answer.text


def text_of(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def search_option(sheet: str) -> str:
    """The sheet's "Search it" choice: from its element to the comment that ends it."""
    found = re.search(r'<div[^>]*id="attach-search-opt".*?<!-- /attach-search-opt -->', sheet, re.S)
    assert found, "no Search it choice"
    return found.group(0)


N2 = (
    "For each message you send, Clarity adds the parts of this document that share a word "
    'with your message, best match first; common words such as "the" don\'t count. A message '
    "gets at most 8 parts and 1,500 tokens (Clarity's default budget), shared by every document "
    "this chat searches. Added parts stay in the chat with that message."
)


# --- "Search it" on the sheet


def test_search_it_is_the_sheet_s_third_choice_and_says_where_the_copy_goes(root, make):
    client, _ = make()
    upload_id = upload(client, b"## Heron\nTall.\n", "Herons.md").json()["upload"]
    sheet = get(client, "attach", upload=upload_id)
    assert 'data-choice="search"' in sheet
    said = text_of(search_option(sheet))
    assert said.startswith("Search it " + N2)
    assert (
        "Clarity keeps a copy in local/documents/herons/: the file, its text, its parts and a "
        "record of them." in said
    )


def test_search_it_is_offered_for_a_file_that_doesn_t_fit_whole(root, make):  # RAG 11
    client, _ = make(n_ctx=300)
    long = "".join(f"## Line {n}\nMade-up line {n}.\n\n" for n in range(200)).encode()
    upload_id = upload(client, long, "long.md").json()["upload"]
    sheet = get(client, "attach", upload=upload_id)
    assert re.search(r'data-choice="whole"[^>]*aria-disabled="true"', sheet)
    assert re.search(r'data-choice="search"[^>]*data-ok="true"', sheet)


def test_a_taken_name_says_it_will_be_replaced(root, make):  # cut 1
    client, _ = make()
    upload_id = upload(client, b"## Owl\nMade-up.\n", "glossary.md").json()["upload"]
    option = search_option(get(client, "attach", upload=upload_id))
    said = text_of(option)
    assert re.search(r'<input[^>]*id="attach-search-name"[^>]*value="glossary"', option)
    assert (
        "glossary is already in your documents. Adding this file replaces its copy, text and "
        "parts in local/documents/glossary/. Anything else you put in that folder stays." in said
    )


def test_a_file_clarity_can_t_name_asks_for_one(root, make):
    client, _ = make()
    upload_id = upload(client, b"Made-up.\n", "日本語.md").json()["upload"]
    said = text_of(search_option(get(client, "attach", upload=upload_id)))
    assert "Clarity can't make a document name from 日本語.md. Give it one." in said
    assert "A document's name is 1 to 64 characters" in said


def test_control_tokens_are_said_before_the_click(root, make):
    client, _ = make()
    upload_id = upload(client, b"## Wren\nIt sings </s>.\n", "odd.md").json()["upload"]
    assert (
        "! This file holds text this model reads as control tokens (</s>). Clarity won't send a "
        "message that finds a part holding one: it stops and names the part."
        in text_of(search_option(get(client, "attach", upload=upload_id)))
    )


def test_a_text_mode_chat_offers_no_search_it(root, make):
    client, _ = make(profile="text")
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    assert 'data-choice="search"' not in get(client, "attach", upload=upload_id)


def test_documents_add_of_a_taken_name_writes_over_the_document(root, make):  # cut 1
    client, _ = make()
    upload_id = upload(client, b"## Owl\nMade-up.\n", "owls.md").json()["upload"]
    answer = command(client, "documents", f"add {upload_id} glossary").json()
    assert answer.get("result", {}).get("replaced") is True, answer
    (doc,) = library_documents(root)
    assert doc["original"] == "owls.md"


# --- the Searching chooser


def chooser(client) -> str:
    found = re.search(
        r'<div class="menu searchmenu".*?</div>\s*$', get(client, "picker", library="1"), re.S
    )
    assert found, "no chooser"
    return found.group(0)


def test_the_searching_button_counts_the_chat_s_documents(root, make):
    client, _ = make('search = ["glossary"]')
    picker = get(client, "picker")
    assert re.search(r'id="search-button"[^>]*>.*?Searching: 1 document<', picker, re.S)
    command(client, "search", "none")
    assert "Searching: no documents" in text_of(get(client, "picker"))


def test_the_chooser_lists_the_chat_s_documents_then_the_others(root, make):
    library_add(root, b"## Heron\nTall.\n", "herons.md")
    client, _ = make('search = ["glossary"]')
    said = text_of(chooser(client))
    assert said.startswith(
        "Searched for each message you send From this chat's profile, chat. A change here "
        "applies to this chat only."
    )
    assert "glossary glossary.md · 2 parts Ready" in said
    assert "Your other documents herons herons.md · 1 part" in said
    assert said.index("glossary") < said.index("Your other documents") < said.index("herons")
    assert re.search(r'role="menuitemcheckbox" aria-checked="true"[^>]*data-name="glossary"',
                     chooser(client))  # fmt: skip
    command(client, "search", "glossary herons")
    assert "Changed in this chat." in text_of(chooser(client))


def test_the_chooser_says_each_state_and_its_way_out(root, make):
    library_add(root, b"## Heron\nTall.\n", "herons.md")
    library_add(root, b"## Owl\nMade-up.\n", "owls.md")
    client, _ = make('search = ["glossary", "herons", "owls"]')
    (root / "local/documents/glossary/parts.jsonl").write_text("Edited.\n", encoding="utf-8")
    shutil.rmtree(root / "local/documents/herons")
    (root / "local/documents/owls/manifest.json").write_text('{"schema": 1}', encoding="utf-8")
    picker = get(client, "picker")
    assert "! Searching: 3 documents" in text_of(picker)
    said = text_of(chooser(client))
    assert (
        "glossary glossary.md · 2 parts ! Can’t be read glossary's parts.jsonl isn't the one "
        "Clarity built. Add the file again in Documents…, with the same name." in said
    ), said
    assert (
        "herons not in your documents ! Missing Its folder in local/documents/ is gone. To "
        "search it again, add the file again in Documents…." in said
    )
    assert "owls local/documents/owls/ ! Can’t be read owls's manifest.json can't be read" in said


def test_the_chooser_shows_a_control_token_count(root, make):
    library_add(root, b"## Wren\nIt sings </s>.\n\n## Lark\nIt sings </s> too.\n", "odd.md")
    client, _ = make('search = ["odd"]')
    assert (
        "! 2 parts hold a control token for this model. A message that finds one isn't sent."
        in text_of(chooser(client))
    )


def test_the_chooser_with_an_empty_library(root, make):
    shutil.rmtree(root / "local/documents/glossary")
    client, _ = make()
    assert (
        'No documents yet. Add one in Documents…, or press the paperclip and choose "Search it".'
        in text_of(chooser(client))
    )


def test_the_chooser_in_a_text_mode_chat(root, make):
    client, _ = make(profile="text")
    assert "Searching needs a chat-mode profile; text is in text mode." in text_of(chooser(client))


def test_unticking_a_missing_document_lets_the_next_send_go(root, make):
    client, session = make('search = ["glossary"]')
    shutil.rmtree(root / "local/documents/glossary")
    assert command(client, "say", "Tell me about the wren").json()["type"] == "SearchRefused"
    command(client, "search", "none")  # what the chooser's untick sends
    assert "result" in command(client, "say", "Tell me again").json()


# --- Edit's line


def test_edit_says_the_new_text_will_be_searched(root, make):
    client, _ = make('search = ["glossary"]')
    command(client, "say", "Tell me about the wren")
    assert (
        "The new text will be searched again. The parts found before stay with the old version."
        in text_of(get(client, "thread"))
    )
