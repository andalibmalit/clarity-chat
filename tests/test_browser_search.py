"""C2 (v0.3.2): "Search it", Replace and the Searching chooser in a real browser, each
sending its command and drawing the result from /state. Made-up documents only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import shutil

import pytest

from clarity_chat.documents import library_add, library_documents
from tests.test_browser import open_clarity, shot
from tests.test_browser_attach import browser_type_launch_args, choose, clarity  # noqa: F401

pytestmark = pytest.mark.browser

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n\n## Dipper\nIt walks under water.\n"


def chat(page, redirect) -> None:
    open_clarity(page, redirect)
    page.click("[data-profile=chat]")
    page.wait_for_selector("#thread .intro")


def test_search_it_adds_the_file_and_searches_it(clarity, page, fetched_root):  # noqa: F811
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "herons.md", b"## Heron\nTall.\n\n## Egret\nWhite.\n")
    page.wait_for_selector("#attach-search")
    page.click("#attach-search")
    assert page.inner_text("[data-action=attach-add]") == "Search it"
    shot(page, "search-it", "v032-c2")
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#notice:has-text('from your next message')")
    assert "This chat searches herons from your next message." in page.inner_text("#notice")
    assert [d["name"] for d in library_documents(fetched_root)] == ["herons"]
    assert "Searching: 1 document" in page.inner_text("#search-button")
    page.click("#search-button")
    page.wait_for_selector("#search-menu:not([hidden])")
    shot(page, "chooser", "v032-c2")
    assert page.get_attribute("#search-menu [data-name=herons]", "aria-checked") == "true"


def test_replace_writes_over_a_taken_name(clarity, page, fetched_root):  # noqa: F811
    library_add(fetched_root, GLOSSARY, "glossary.md")
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "glossary.md", b"## Owl\nMade-up.\n")
    page.wait_for_selector("#attach-search")
    page.click("#attach-search")
    page.click("[data-action=attach-replace]")
    assert page.is_visible("#attach-replace-note")
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#notice:has-text('from your next message')")
    (doc,) = library_documents(fetched_root)
    assert doc["original"] == "glossary.md" and doc["parts"] == 1


def test_unticking_a_missing_document_lets_send_go(clarity, page, fetched_root):  # noqa: F811
    library_add(fetched_root, GLOSSARY, "glossary.md")
    path = fetched_root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    redirect, backend = clarity
    chat(page, redirect)
    shutil.rmtree(fetched_root / "local/documents/glossary")
    page.fill("#message", "Tell me about the wren")
    page.click("#send")
    page.wait_for_selector("#thread .problem")
    page.click("#search-button")
    page.wait_for_selector("#search-menu:not([hidden])")
    assert "! Missing" in page.inner_text("#search-menu")
    page.click("#search-menu [data-name=glossary]")
    page.wait_for_selector("#search-button:has-text('Searching: no documents')")
    page.keyboard.press("Escape")
    page.fill("#message", "Tell me again")
    page.click("#send")
    page.wait_for_selector("#thread .turn.assistant")
    assert backend.calls
