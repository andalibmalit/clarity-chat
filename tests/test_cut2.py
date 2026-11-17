"""Cut 2 of the simplicity cuts: the Documents… page goes. The paperclip's "Search it" adds a
file, its Name box always shown and filled in; the Searching chooser lists the library; no
sentence on the page sends you to Documents…. `/try` and `clarity document try` stay. Made-up
documents, scratch roots."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from clarity_chat.documents import SENTENCES, way_out
from clarity_chat.documents.refused import problem_words
from tests.test_web_search_c2 import get, make, root, search_option, text_of  # noqa: F401
from tests.test_web_upload import upload

INDEX = Path(__file__).resolve().parents[1] / "src/clarity_chat/web_static/index.html"


def test_there_is_no_documents_page(root, make):  # noqa: F811
    client, _ = make()
    from tests.test_web import GET

    assert client.get("/view/documents", headers=GET).status_code == 404


def test_no_menu_or_chooser_opens_it(root, make):  # noqa: F811
    index = INDEX.read_text(encoding="utf-8")
    assert 'data-action="documents"' not in index and "documents-sheet" not in index
    client, _ = make('search = ["glossary"]')
    assert 'data-action="documents"' not in get(client, "picker", library="1")


def test_search_it_always_shows_the_name_box_filled_in(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    option = search_option(get(client, "attach", upload=upload_id))
    box = re.search(r'<input[^>]*id="attach-search-name"[^>]*>', option)
    assert box, option
    assert 'type="text"' in box.group(0) and 'value="herons"' in box.group(0), box.group(0)


def test_no_page_sentence_sends_you_to_documents(root):  # noqa: F811
    problems = [
        {"document": "g", "state": "missing"},
        {"document": "g", "state": "unreadable", "why": "g's manifest.json can't be read (x).",
         "add_again": True},
        {"document": "g", "state": "profile_missing", "profile": "chat"},
        {"document": "g", "state": "control_tokens", "ordinal": 1, "title": None, "page": None,
         "tokens": ["</s>"]},
    ]  # fmt: skip
    said = [problem_words(p, terminal=False, where="chat", at_send=True) for p in problems]
    said += [way_out("g", "page"), *(s for s in SENTENCES.values() if isinstance(s, str))]
    assert [s for s in said if "Documents…" in s] == []


def test_the_empty_chooser_points_to_the_paperclip(root, make):  # noqa: F811
    shutil.rmtree(root / "local/documents/glossary")
    client, _ = make()
    assert 'No documents yet. Press the paperclip and choose "Search it".' in text_of(
        get(client, "picker", library="1")
    )
