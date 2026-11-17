"""The v0.3.2 close fix batch on the page: each refusal's way out stays inside Clarity
(Documents… → Add with Replace), and the chooser's other documents show their state.
Made-up documents, scratch roots."""

from __future__ import annotations

import re

from clarity_chat.documents import library_add
from tests.test_web import command
from tests.test_web_search_c2 import GLOSSARY, get, make, root, text_of  # noqa: F401
from tests.test_web_upload import upload

PAGE_WAY = "Add the file again in Documents… and choose Replace."


def garble_manifest(project) -> None:
    (project / "local/documents/glossary/manifest.json").write_text("{garbage", encoding="utf-8")


def test_documents_row_for_an_unreadable_document_says_the_page_s_way(root, make):  # noqa: F811
    client, _ = make()
    garble_manifest(root)
    said = text_of(get(client, "documents"))
    # The reason can hold parentheses of its own (a JSON error's "(char 1)").
    assert re.search(r"glossary's manifest\.json can't be read \(.*\)\. " + re.escape(PAGE_WAY),
                     said), said  # fmt: skip
    assert "Delete the folder" not in said


def test_send_with_an_unreadable_document_says_the_page_s_way(root, make):  # noqa: F811
    client, _ = make('search = ["glossary"]')
    garble_manifest(root)
    answer = command(client, "say", "Tell me about the wren").json()
    assert answer.get("error", "").endswith(PAGE_WAY), answer
    assert PAGE_WAY in text_of(get(client, "thread"))


def test_page_rebuild_of_a_changed_original_says_the_page_s_way(root, make):  # noqa: F811
    client, _ = make()
    folder = root / "local/documents/glossary"
    original = next(folder.glob("original.*"))
    original.write_bytes(original.read_bytes() + b"Made-up change.\n")
    (folder / "text.md").write_text("## Wren\nEdited.\n", encoding="utf-8")
    answer = command(client, "documents", "rebuild glossary").json()
    assert answer.get("error", "").endswith(PAGE_WAY), answer


def test_add_of_the_same_file_offers_replace_when_its_document_isn_t_ready(root, make):  # noqa: F811
    client, _ = make()
    (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
    upload_id = upload(client, GLOSSARY, "glossary.md").json()["upload"]
    fragment = get(client, "documents", upload=upload_id)
    said = text_of(fragment)
    assert "Already in your documents as glossary. Nothing new is added to them." in said
    assert 'data-action="documents-replace"' in fragment
    assert "Replace glossary with this file" in said


def test_add_of_the_same_file_offers_no_replace_when_its_document_is_ready(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, GLOSSARY, "glossary.md").json()["upload"]
    assert 'data-action="documents-replace"' not in get(client, "documents", upload=upload_id)


def test_the_chooser_shows_an_unreadable_other_document_s_state(root, make):  # noqa: F811
    library_add(root, b"## Owl\nA made-up owl.\n", "notes.md")
    client, _ = make('search = ["glossary"]')
    (root / "local/documents/notes/manifest.json").write_text("{garbage", encoding="utf-8")
    chooser = get(client, "picker", library="1")
    others = text_of(chooser.split("Your other documents", 1)[-1])
    assert "notes" in others and "Can’t be read" in others, others
