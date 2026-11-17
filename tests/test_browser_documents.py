"""C1 (v0.3.2): the Documents… page in a real browser: Clarity ▾ → Documents…, Add (the
system's file picker, then /documents add), "Search it in this chat", and the try
box, each sending its command and drawing its result. Made-up documents only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from clarity_chat.documents import library_add, library_documents
from tests.test_browser import open_clarity, shot
from tests.test_browser_attach import browser_type_launch_args, clarity  # noqa: F401

pytestmark = pytest.mark.browser

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n\n## Dipper\nIt walks under water.\n"


def documents(page, redirect) -> None:
    open_clarity(page, redirect)
    page.click("[data-profile=chat]")
    page.wait_for_selector("#thread .intro")
    page.click("#clarity-button")
    page.click("[data-action=documents]")
    page.wait_for_selector("#documents-sheet[open] .try")


def test_add_writes_the_copy_lists_it_and_searches_it_here(clarity, page, fetched_root):  # noqa: F811
    redirect, _ = clarity
    documents(page, redirect)
    with page.expect_file_chooser() as chooser:
        page.click("[data-action=documents-add]")
    chooser.value.set_files({"name": "Herons.md", "mimeType": "text/markdown",
                             "buffer": b"## Heron\nTall.\n\n## Egret\nWhite.\n"})  # fmt: skip
    page.wait_for_selector("#documents-add-form")
    assert "Clarity keeps a copy in local/documents/herons/" in page.inner_text(
        "#documents-add-form"
    )
    shot(page, "documents-add", "v032-c1")
    page.click("#documents-add-form button[type=submit]")
    page.wait_for_selector("#documents-body .notice")
    assert "Added herons. Clarity wrote local/documents/herons/" in page.inner_text(
        "#documents-body .notice"
    )
    assert [d["name"] for d in library_documents(fetched_root)] == ["herons"]
    assert page.locator("tr[data-name=herons]").count() == 1  # the page lists it (could 6)
    page.click("[data-action=documents-search-here]")
    page.wait_for_selector("#documents-body .notice:has-text('from your next message')")
    assert "This chat searches herons from your next message." in page.inner_text(
        "#documents-body .notice"
    )
    shot(page, "documents-searched", "v032-c1")


def test_a_taken_name_is_replaced_said_before(clarity, page, fetched_root):  # noqa: F811
    library_add(fetched_root, GLOSSARY, "glossary.md")  # cut 1: a taken name is replaced
    redirect, _ = clarity
    documents(page, redirect)
    with page.expect_file_chooser() as chooser:
        page.click("[data-action=documents-add]")
    chooser.value.set_files({"name": "glossary.md", "mimeType": "text/markdown",
                             "buffer": b"## Owl\nMade-up.\n"})  # fmt: skip
    page.wait_for_selector("#documents-add-form:has-text('is already in your documents')")
    page.click("#documents-add-form button[type=submit]")
    page.wait_for_selector("#documents-body .notice")
    assert "Clarity replaced" in page.inner_text("#documents-body .notice")
    (doc,) = library_documents(fetched_root)
    assert (doc["name"], doc["parts"]) == ("glossary", 1)


def test_the_try_box_shows_what_try_finds(clarity, page, fetched_root):  # noqa: F811
    library_add(fetched_root, GLOSSARY, "glossary.md")
    path = fetched_root / "profiles" / "chat.toml"  # the chooser is C2's: the profile's list
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    redirect, backend = clarity
    documents(page, redirect)
    page.wait_for_selector("#documents-try:not([disabled])")
    page.fill("#documents-try", "Tell me about the wren")
    page.press("#documents-try", "Enter")
    page.wait_for_selector("#documents-body .parts .part")
    said = page.inner_text("#documents-body .try")
    assert 'glossary · "Wren"' in said and "in its title: wren" in said
    shot(page, "documents-try", "v032-c1")
    assert "Words searched: wren." in said
    assert backend.calls == []  # nothing is sent to the model


def test_a_file_add_refuses_shows_its_sentence_in_the_page(clarity, page, fetched_root):  # noqa: F811
    """V31's reader sentences reach the page (the C1 tripwire review's should 4)."""
    redirect, _ = clarity
    documents(page, redirect)
    with page.expect_file_chooser() as chooser:
        page.click("[data-action=documents-add]")
    chooser.value.set_files({"name": "latin.md", "mimeType": "text/markdown",
                             "buffer": "caf\xe9\n".encode("latin-1")})  # fmt: skip
    page.wait_for_selector("#documents-add-form")
    page.click("#documents-add-form button[type=submit]")
    page.wait_for_selector("#documents-body .refused:not([hidden])")
    assert "latin.md isn't UTF-8 text." in page.inner_text("#documents-body .refused")
    assert library_documents(fetched_root) == []
