"""v0.4.4 B2: the Settings panel's Documents group in a chat-mode chat (states 16 to 19): the
method as two buttons, the minimum and the budget as number boxes, each with its origin and
help; the minimum hidden for words; Back to the base; Reset all covering the group but not
the size; the rows waiting while a reply runs; and none of it in a text-mode chat. Ruling 5:
whole numbers of 1,000 or more get separators where the page writes them, not in a box.
FakeEmbedder stands in for bge-m3. Made-up text only. The page's own clicks are in
tests/test_browser_settings_documents.py."""

from __future__ import annotations

import html
import re
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import SENTENCES, library_add, reading
from clarity_chat.profiles import SETTING_HELP
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_meaning import ASKED, GLOSSARY, MEANING, profile
from tests.test_web import ORIGIN, PORT, SECRET, command, wait_until, writing
from tests.test_web_meaning import resumed
from tests.test_web_settings import panel, result, row
from tests.test_web_shell import get_view

NO_EMBEDDER = (
    "Meaning search needs bge-m3, which isn't fetched. Getting started shows how to fetch it. "
    'Or search by words: set search_method to "words".'
)


@pytest.fixture
def documents(fetched_root, chat_tokenizer):
    """The page on a chat that searches `notes` by meaning (profile `meaning`, minimum 0.2);
    profile `plain` searches it with no method of its own."""
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    profile(fetched_root, "plain", '\n[documents]\nsearch = ["notes"]\n')
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    backend = FakeBackend([" Hello!"] * 20, tokenizer=chat_tokenizer)
    session = Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=FakeEmbedder(),  # ty: ignore[invalid-argument-type]
    )
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "meaning")
        yield client, session


def group(fragment: str) -> str:
    """The Documents group: from its heading to the next section."""
    found = re.search(r'<h3 class="sh">Documents</h3>(.*?)<details class="more"', fragment, re.S)
    assert found, "no Documents group"
    return found.group(1)


def method_buttons(fragment: str) -> dict[str, str]:
    """The method row's two buttons: {label: aria-pressed}."""
    buttons = re.findall(
        r'<button type="button" class="btn choice" data-set="search_method (\w+)" '
        r'aria-pressed="(true|false)"[^>]*>(\w+)</button>',
        row(fragment, "search_method"),
    )
    assert all(value == label for value, _, label in buttons), buttons
    return {label: pressed for _, pressed, label in buttons}


def test_a_chat_mode_panel_has_the_documents_group_after_common(documents):
    """State 16: "Documents" after Common, its applies line, and three rows with their
    origins and help (K2's placement row isn't built)."""
    client, _ = documents
    fragment = panel(client)
    assert fragment.index('<h3 class="sh">Common</h3>') < fragment.index(
        '<h3 class="sh">Documents</h3>'
    )
    rows = group(fragment)
    assert f'<p class="gl">{SENTENCES["searching_applies"]}</p>' in rows
    assert re.findall(r'data-setting="(\w+)"', rows) == [
        "search_method",
        "min_similarity",
        "budget",
    ]
    for name, origin in (
        ("search_method", "profile"),
        ("min_similarity", "profile"),
        ("budget", "default"),
    ):
        r = row(fragment, name)
        assert f'<span class="tag">{origin}</span>' in r and SETTING_HELP[name] in r
        assert "Back to" not in r
    assert method_buttons(fragment) == {"meaning": "true", "words": "false"}
    assert 'type="number"' not in row(fragment, "search_method")
    assert re.search(
        r'<input type="number" class="num" id="set-min_similarity"[^>]* value="0.2"', fragment
    )
    # Ruling 5: the box keeps the plain number; the page's own words have the separator.
    budget = row(fragment, "budget")
    assert 'value="1500" step="1" data-in-use="1,500 (default)"' in budget
    assert 'data-action="reset-all"' not in fragment


def test_words_hides_the_minimum_and_offers_the_way_back(documents):
    """States 17 and 10: words has no minimum, so its row goes; the method row's Back to
    names the base and its origin."""
    client, _ = documents
    result(client, "set", "search_method words")
    fragment = panel(client)
    assert re.findall(r'data-setting="(\w+)"', group(fragment)) == ["search_method", "budget"]
    assert method_buttons(fragment) == {"meaning": "false", "words": "true"}
    r = row(fragment, "search_method")
    assert '<span class="tag">chat</span>' in r
    assert 'data-set="search_method">Back to meaning (profile)<' in r
    result(client, "set", "search_method")
    assert 'data-setting="min_similarity"' in group(panel(client))


def test_a_changed_budget_goes_back_to_its_base_and_reset_all_leaves_the_size(documents):
    """States 7, 10 and 18: the row's Back to has the separator (ruling 5); Reset all names
    the Documents rows and not ctx_size, which has its own way back; the picker counts
    them."""
    client, _ = documents
    result(client, "set", "budget 800")
    result(client, "set", "min_similarity 0.5")
    result(client, "set", "ctx_size 16384")
    fragment = panel(client)
    r = row(fragment, "budget")
    assert '<span class="tag">chat</span>' in r and 'value="800"' in r
    assert 'data-set="budget">Back to 1,500 (default)<' in r
    assert 'data-set="min_similarity">Back to 0.2 (profile)<' in row(fragment, "min_similarity")
    names = re.search(r'data-action="reset-all" data-names="([^"]*)"', fragment)
    assert names and sorted(names.group(1).split()) == ["budget", "min_similarity"]
    assert "· 2 changed" in get_view(client, "picker")


def test_a_sampling_row_s_way_back_has_the_separator(documents):
    """Ruling 5 on a sampling row: "Back to 2,048 (profile)"; the box keeps 100."""
    client, session = documents
    long = (session.root / "profiles" / "meaning.toml").read_text("utf-8")
    (session.root / "profiles" / "long.toml").write_text(
        long.replace("max_tokens = 16", "max_tokens = 2048"), "utf-8"
    )
    command(client, "profile", "long")
    result(client, "set", "max_tokens 100")
    r = row(panel(client), "max_tokens")
    assert 'data-set="max_tokens">Back to 2,048 (profile)<' in r and 'value="100"' in r


def test_a_text_mode_chat_has_no_documents_group(documents):
    """States 9 and 16: search needs a chat-mode profile."""
    client, _ = documents
    command(client, "profile", "text")
    fragment = panel(client)
    assert "Documents" not in fragment and "search_method" not in fragment
    assert 'data-setting="temperature"' in fragment


def test_a_resumed_chat_shows_chat_origins_and_logged_bases(documents):
    """States 11 and 25: the chat's /set values come back with origin chat; a profile with
    no method gets the last searched message's, origin log, which is also its base
    (decision 3)."""
    client, _ = documents
    command(client, "profile", "plain")
    result(client, "set", "budget 800")
    result(client, "say", ASKED)
    resumed(client)
    fragment = panel(client)
    budget = row(fragment, "budget")
    assert '<span class="tag">chat</span>' in budget and 'value="800"' in budget
    assert 'data-set="budget">Back to 1,500 (default)<' in budget
    assert '<span class="tag">log</span>' in row(fragment, "search_method")
    assert method_buttons(fragment) == {"meaning": "true", "words": "false"}
    result(client, "set", "search_method words")
    method = row(panel(client), "search_method")
    assert '<span class="tag">chat</span>' in method
    assert 'data-set="search_method">Back to meaning (log)<' in method


def test_the_method_row_carries_the_words_for_bge_m3_not_fetched(documents):
    """State 12: the page shows O1's words under the row when /set answers with the warning
    (the browser test chooses it)."""
    client, _ = documents
    raw = get_view(client, "settings")  # escaped: the words have quotes of their own
    words = re.search(r'data-setting="search_method".*?data-no-embedder="([^"]*)"', raw, re.S)
    assert words and html.unescape(words.group(1)) == NO_EMBEDDER


def test_the_documents_rows_wait_while_a_reply_runs(documents):
    """State 14: every Documents control waits and says why."""
    client, session = documents
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 1.0
    worker = threading.Thread(target=lambda: command(client, "say", "Hi"))
    worker.start()
    wait_until(lambda: writing(client))
    try:
        fragment = panel(client)
        during = worker.is_alive()
    finally:
        worker.join(10)
    assert during
    controls = re.findall(r"<(?:input|button)[^>]*>", group(fragment))
    assert len(controls) == 4  # two method buttons, two boxes
    assert all('aria-disabled="true"' in c and "settings-waits" in c for c in controls)
