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


def test_a_name_typed_after_replace_isn_t_replaced(clarity, page, fetched_root):  # noqa: F811
    library_add(fetched_root, GLOSSARY, "glossary.md")
    library_add(fetched_root, b"## Heron\nTall.\n", "herons.md")
    redirect, _ = clarity
    chat(page, redirect)
    choose(page, "glossary.md", b"## Owl\nMade-up.\n")
    page.wait_for_selector("#attach-search")
    page.click("#attach-search")
    page.click("[data-action=attach-replace]")
    page.fill("#attach-search-name", "herons")  # the note named glossary, not herons
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#attach-body .sheet-error:has-text('already a document named herons')")
    herons = next(d for d in library_documents(fetched_root) if d["name"] == "herons")
    assert herons["original"] == "herons.md"


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


def searching_chat(page, redirect, fetched_root) -> None:
    library_add(fetched_root, GLOSSARY, "glossary.md")
    path = fetched_root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    chat(page, redirect)


def test_from_documents_opens_its_list_and_details(clarity, page, fetched_root):  # noqa: F811
    redirect, backend = clarity  # C3
    searching_chat(page, redirect, fetched_root)
    backend.replies, backend.piece_delay = [" Slowly," * 8], 0.3
    page.fill("#message", "Tell me about the wren")
    page.click("#send")
    # The parts are found before the call: the line shows while the reply is written.
    page.wait_for_selector(".writing-found [data-action=found]")
    page.click(".writing-found [data-action=found]")  # opened while written, kept open after
    reply = "#thread .turn.assistant:not(.writing-found)"
    page.wait_for_selector(f"{reply} .found .parts:not([hidden])")
    line = page.locator(f"{reply} [data-action=found]")
    # A flex row: innerText puts its items on lines of their own.
    assert " ".join(line.inner_text().split()).startswith("From documents · 1 part · ")
    assert 'glossary · "Wren"' in page.inner_text("#thread .found .parts")
    shot(page, "from-documents", "v032-c3")
    page.click("#thread .found [data-section=documents]")
    page.wait_for_selector("#panel details[data-key=documents][open]")
    assert "Words searched" in page.inner_text("#panel details[data-key=documents]")
    shot(page, "details-documents", "v032-c3")


def test_a_refused_send_s_stop_searching_it_lets_send_go(clarity, page, fetched_root):  # noqa: F811
    redirect, backend = clarity  # C3, W6
    searching_chat(page, redirect, fetched_root)
    shutil.rmtree(fetched_root / "local/documents/glossary")
    page.fill("#message", "Tell me about the wren")
    page.click("#send")
    page.wait_for_selector("#thread .problem.refused")
    assert page.inner_text("#thread .problem.refused").startswith("! Nothing was sent.")
    shot(page, "refused", "v032-c3")
    page.click("#thread .problem.refused [data-action=refused-stop]")
    page.wait_for_selector("#thread .problem.refused", state="detached")
    page.fill("#message", "Tell me again")
    page.click("#send")
    page.wait_for_selector("#thread .turn.assistant")
    assert backend.calls


def test_new_chat_with_these_documents_starts_one_that_searches_them(clarity, page, fetched_root):  # noqa: F811
    library_add(fetched_root, GLOSSARY, "glossary.md")  # the part 2 whole-branch review's S4
    redirect, backend = clarity
    chat(page, redirect)  # the profile searches nothing: the chat's list is the chooser's
    page.click("#search-button")
    page.wait_for_selector("#search-menu:not([hidden])")
    page.click("#search-menu [data-name=glossary]")
    page.wait_for_selector("#search-button:has-text('Searching: 1 document')")
    page.keyboard.press("Escape")
    backend._n_ctx = 60
    page.fill("#message", "Tell me about the wren " * 4)
    page.click("#send")
    page.wait_for_selector("[data-action=new-with-documents]")
    backend._n_ctx = None  # the new chat starts with the usual context
    page.click("[data-action=new-with-documents]")
    page.wait_for_selector("#thread .intro")
    page.wait_for_selector("#search-button:has-text('Searching: 1 document')")
