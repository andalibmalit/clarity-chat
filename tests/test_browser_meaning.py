"""v0.4.1 C1: meaning search on the page in a real browser, live and after a reload: each
part's reason in "From documents", Details' line, "Search it"'s warning when bge-m3 isn't
fetched, and the refused Send. FakeEmbedder stands in for bge-m3. Made-up text only.

Run with `uv run pytest -m browser`.
"""

from __future__ import annotations

import pytest

from clarity_chat.documents import library_add, reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_browser import open_clarity, served, settled
from tests.test_browser_attach import browser_type_launch_args, choose  # noqa: F401
from tests.test_meaning import ASKED, GLOSSARY, MEANING, profile

pytestmark = pytest.mark.browser

NO_EMBEDDER = (
    "Meaning search needs bge-m3, which isn't fetched. Getting started shows how to fetch it. "
    'Or search by words: set search_method to "words".'
)


@pytest.fixture
def clarity(fetched_root, chat_tokenizer):
    """The page with a profile, `meaning`, that searches `notes` by meaning: (redirect, the
    embedder)."""
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    profile(fetched_root, "nothing", '\n[documents]\nsearch_method = "meaning"\n')
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    backend = FakeBackend([" A made-up reply."] * 9, tokenizer=chat_tokenizer)
    fake = FakeEmbedder()
    session = Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    for _, redirect in served(fetched_root, session):
        yield redirect, fake


def start(page, redirect, name: str) -> None:
    open_clarity(page, redirect)
    page.click(f"[data-profile={name}]")
    page.wait_for_selector("#thread .intro")


def test_from_documents_and_details_say_the_similarity(clarity, page):
    redirect, _ = clarity
    start(page, redirect, "meaning")
    page.fill("#message", ASKED)
    page.click("#send")
    reply = "#thread .turn.assistant"
    page.wait_for_selector(f"{reply} .found [data-action=found]")
    settled(page)
    for reloaded in (False, True):
        if reloaded:
            page.reload()
            page.wait_for_selector(f"{reply} .found [data-action=found]")
        page.click(f"{reply} .found [data-action=found]")
        page.wait_for_selector(f"{reply} .found .parts:not([hidden])")
        assert 'notes · "Photosynthesis"' in page.inner_text(f"{reply} .found .parts")
        assert "similar in meaning: " in page.inner_text(f"{reply} .found .parts")
    page.click("#thread .found [data-section=documents]")
    page.wait_for_selector("#panel details[data-key=documents][open]")
    said = page.inner_text("#panel details[data-key=documents]")
    assert "meaning, by bge-m3 (minimum similarity 0.2, this profile)" in said


def test_search_it_warns_when_bge_m3_isn_t_fetched(clarity, page):
    redirect, fake = clarity
    fake.is_fetched = False
    start(page, redirect, "nothing")
    choose(page, "herons.md", b"## Heron\nTall.\n")
    page.wait_for_selector("#attach-search")
    page.click("#attach-search")
    page.click("[data-action=attach-add]")
    page.wait_for_selector("#notice:has-text('from your next message')")
    assert NO_EMBEDDER in page.inner_text("#notice")


def test_send_refused_names_getting_started(clarity, page):
    redirect, fake = clarity
    fake.is_fetched = False
    start(page, redirect, "meaning")
    page.fill("#message", ASKED)
    page.click("#send")
    # settled() alone can return before the send starts (CI's Firefox run 45085021195): wait
    # for the refusal's box, as test_browser_search's refused send does.
    page.wait_for_selector("#thread .problem.refused")
    settled(page)  # the refused say has ended and its redraw landed
    assert f"! Nothing was sent. {NO_EMBEDDER}" in " ".join(page.inner_text("#thread").split())
