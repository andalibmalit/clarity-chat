"""The v0.3.2 close fix batch on the page: each refusal's way out stays inside Clarity
(the paperclip's Search it, under the same name), and the chooser's other documents show
their state. Made-up documents, scratch roots."""

from __future__ import annotations

from clarity_chat.documents import library_add
from tests.test_web import command
from tests.test_web_search_c2 import GLOSSARY, get, make, root, text_of  # noqa: F401

PAGE_WAY = "Add the file again with the paperclip's Search it, under the same name."


def garble_manifest(project) -> None:
    (project / "local/documents/glossary/manifest.json").write_text("{garbage", encoding="utf-8")


def test_send_with_an_unreadable_document_says_the_page_s_way(root, make):  # noqa: F811
    client, _ = make('search = ["glossary"]')
    garble_manifest(root)
    answer = command(client, "say", "Tell me about the wren").json()
    assert answer.get("error", "").endswith(PAGE_WAY), answer
    assert PAGE_WAY in text_of(get(client, "thread"))


def test_the_chooser_shows_an_unreadable_other_document_s_state(root, make):  # noqa: F811
    library_add(root, b"## Owl\nA made-up owl.\n", "notes.md")
    client, _ = make('search = ["glossary"]')
    (root / "local/documents/notes/manifest.json").write_text("{garbage", encoding="utf-8")
    chooser = get(client, "picker", library="1")
    others = text_of(chooser.split("Your other documents", 1)[-1])
    assert "notes" in others and "Can’t be read" in others, others
