"""v0.4.1 C1: the page's meaning search, in the views it already has. Each part's reason in
"From documents"; Details' one line for the method, model and minimum, and what the minimum
did when nothing matched; the chooser's heading; "Search it"'s slot for how parts are
chosen, and its warning when bge-m3 isn't fetched; states 5 to 7 where the page shows
refusals now; and a words chat drawn exactly as v0.3.2 drew it. FakeEmbedder stands in for
bge-m3 (its live gate is tests/test_embed.py). Made-up text only."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import library_add, reading
from clarity_chat.logs import read_records
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_meaning import ASKED, GLOSSARY, MEANING, profile
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command
from tests.test_web_search_c2 import chooser, get, search_option, text_of
from tests.test_web_upload import upload

GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "v0.3.2.jsonl"
SECTION = r'<details class="sec" data-key="documents".*?</details>'


@pytest.fixture
def meaning(fetched_root, chat_tokenizer):
    """The page on a chat that searches `notes` by meaning (minimum 0.2, the profile's)."""
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    backend = FakeBackend([" Hello!"] * 20, tokenizer=chat_tokenizer)
    fake = FakeEmbedder()
    session = Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "meaning")
        yield client, session, fake


def found_lines(client) -> list[str]:
    return [
        text_of(line)
        for line in re.findall(r'<div class="found".*?<!-- /found -->', get(client, "thread"), re.S)
    ]


def documents_section(client, call_id: str) -> str:
    section = re.search(SECTION, get(client, "details", call=call_id), re.S)
    assert section, "no Documents section"
    return text_of(section.group(0))


def last_found(client) -> dict:
    """The last message's found as /history gives the page it."""
    turns = command(client, "history").json()["result"]["turns"]
    return [t["found"] for t in turns if t["role"] == "user"][-1]


def last_call(client) -> str:
    turns = command(client, "state").json()["result"]["turns"]
    return next(t["call_id"] for t in reversed(turns) if t["role"] == "assistant")


def resumed(client) -> None:
    log_id = command(client, "state").json()["result"]["log_id"]
    command(client, "new", "")
    assert "error" not in command(client, "resume", log_id).json()


# --- "From documents": each part's reason (state 2)


def test_from_documents_gives_each_part_s_similarity(meaning):
    client, _, _ = meaning
    command(client, "say", ASKED)
    found = last_found(client)
    assert found["method"] == "meaning" and found["parts"]
    reasons = " ".join(
        f'{i + 1} notes · "{p["title"]}" similar in meaning: {p["score"]:.2f}'
        for i, p in enumerate(found["parts"])
    )
    tokens = f"{found['tokens']:,}"
    n = len(found["parts"])
    expected = (
        f"From documents · {n} part{'s' if n > 1 else ''} · {tokens} tokens Rank Part {reasons} "
        "Found when you sent this message. Regenerating reuses these parts; it doesn't search "
        "again. Show in Details"
    )
    assert found_lines(client) == [expected]
    assert found_lines(client) == [expected]  # a reload draws it again from the record
    resumed(client)
    assert found_lines(client) == [expected]


# --- Details: one line for the method, model and minimum (state 2)


def test_details_names_the_method_model_and_minimum(meaning):
    client, _, _ = meaning
    command(client, "say", ASKED)
    found = last_found(client)
    best = found["parts"][0]
    for _ in range(2):  # before and after a resume
        said = documents_section(client, last_call(client))
        assert "Search meaning, by bge-m3 (minimum similarity 0.2, this profile)" in said
        assert f'1 notes · "{best["title"]}" similar in meaning: {best["score"]:.2f}' in said
        assert "Words searched" not in said and "Rank 1 is the best match" not in said  # C4
        resumed(client)


def test_the_minimum_s_origin_is_clarity_s_default_when_the_profile_doesn_t_set_it(meaning):
    client, session, _ = meaning
    profile(session.root, "plain", MEANING.replace("min_similarity = 0.2\n", ""))
    command(client, "profile", "plain")
    command(client, "say", ASKED)
    said = documents_section(client, last_call(client))
    assert "Search meaning, by bge-m3 (minimum similarity 0.5, Clarity's default)" in said


# --- Nothing reached the minimum (state 3)


def test_nothing_matched_says_what_the_minimum_did(meaning):
    client, session, _ = meaning
    profile(session.root, "strict", MEANING.replace("0.2", "0.99"))
    command(client, "profile", "strict")
    command(client, "say", ASKED)
    told = "No part of notes reached the minimum similarity of 0.99, so nothing was added."
    (line,) = found_lines(client)
    assert line.startswith(f"From documents · nothing matched {told} Searched when you sent")
    assert "share a word" not in line and "Words searched" not in line
    said = documents_section(client, last_call(client))
    assert said.startswith("Documents nothing matched")
    assert said.endswith(told)
    assert "Search meaning, by bge-m3 (minimum similarity 0.99, this profile)" in said


# --- The chooser's heading (state 14) and "Search it" (states 13 and 5)


def test_the_chooser_s_heading_says_meaning(meaning):
    client, _, _ = meaning
    assert text_of(chooser(client)).startswith("Searched by meaning for each message you send")


def search_it(client) -> str:
    upload_id = upload(client, b"## Heron\nTall.\n", "Herons.md").json()["upload"]
    return text_of(search_option(get(client, "attach", upload=upload_id)))


def test_search_it_says_parts_are_chosen_by_meaning_and_the_minimum(meaning):
    """The main session's ruling on C1 (UI matches backend): the sheet names the minimum, so
    nobody expects a part every time ("hi" adds none at 0.5)."""
    client, session, _ = meaning
    assert search_it(client).startswith(
        "Search it For each message you send, Clarity adds the parts of this document whose "
        "meaning is close enough to your message (similarity at least 0.2, this profile), best "
        "match first. A message gets at most 8 parts and 1,500 tokens (Clarity's default "
        "budget), shared by every document this chat searches. Added parts stay in the chat "
        "with that message."
    )
    profile(session.root, "plain", MEANING.replace("min_similarity = 0.2\n", ""))
    command(client, "profile", "plain")
    assert "(similarity at least 0.5, Clarity's default)" in search_it(client)
    state = command(client, "state").json()["result"]
    assert state.get("min_similarity") == {"value": 0.5, "origin": "default"}


def test_search_it_carries_the_warning_for_bge_m3_not_fetched(meaning):
    """O1's warning (state 5): the sheet carries its words; Search it says them after its
    /search answers with the warning (the browser test sends it)."""
    client, _, _ = meaning
    upload_id = upload(client, b"## Heron\nTall.\n", "Herons.md").json()["upload"]
    sheet = get(client, "attach", upload=upload_id)
    words = re.search(r'data-no-embedder="([^"]*)"', search_option(sheet))
    assert words, "no warning words on Search it"
    assert text_of(words.group(1)) == (
        "Meaning search needs bge-m3, which isn't fetched. Getting started shows how to fetch "
        'it. Or set search_method = "words" under [documents] in the profile and start a new '
        "chat."
    )


# --- Refused at Send (states 5 to 7): a line in the box the page shows now


def refusal(client) -> str:
    box = re.search(
        r'<div class="problem refused".*?<!-- /refused -->', get(client, "thread"), re.S
    )
    assert box, "no refusal box"
    return text_of(box.group(0))


def test_send_refused_for_bge_m3_not_fetched_names_getting_started(meaning):
    client, _, fake = meaning
    fake.is_fetched = False
    assert command(client, "say", ASKED).json()["type"] == "SearchRefused"
    assert refusal(client) == (
        "! Nothing was sent. Meaning search needs bge-m3, which isn't fetched. Getting started "
        'shows how to fetch it. Or set search_method = "words" under [documents] in the '
        "profile and start a new chat."
    )


def test_send_refused_by_the_embedder_gives_its_reason(meaning):
    client, _, fake = meaning
    fake.fail_at = 1
    assert command(client, "say", ASKED).json()["type"] == "SearchRefused"
    assert refusal(client) == (
        "! Nothing was sent. The embedder sent a vector Clarity can't use: made up. To send "
        "without searching, stop searching every document in this chat."
    )


def test_send_refused_for_a_message_past_8192_tokens(meaning):
    client, _, _ = meaning
    assert command(client, "say", "a. " * 8200).json()["type"] == "SearchRefused"
    assert refusal(client) == (
        "! Nothing was sent. Your message is too long for meaning search (8,202 tokens of "
        "8,192). Shorten it. To send without searching, stop searching every document in this "
        "chat."
    )


# --- Every view draws a meaning chat (T1's every-view test; Checkpoint 1's could 7)


def test_every_view_draws_a_meaning_chat(meaning):
    client, session, _ = meaning
    command(client, "say", ASKED)
    profile(session.root, "strict", MEANING.replace("0.2", "0.99"))
    command(client, "profile", "strict")
    command(client, "say", ASKED)
    calls = [
        r["call_id"] for r in read_records(session.conversation.log.path) if r["type"] == "call"
    ]
    views = ["thread", "branches", "settings", "chats", "picker"]
    paths = [f"/view/{v}" for v in views] + [
        f"/view/{v}?call={c}" for c in calls for v in ("details", "compare")
    ]
    bad = []
    for path in paths:
        response = client.get(path, headers={**AUTH, **SAME})
        if response.status_code != 200 or "Can't read" in response.text:
            bad.append((path, response.status_code, response.text[:200]))
    assert bad == []


# --- A words chat draws v0.3.2's line and section exactly (golden v0.3.2's searched chat)

# Drawn by the page at 5b1383a (v0.4.1 before C1), which draws them as v0.3.2 did.
V032_LINES = [
    'From documents · 1 part · 186 tokens Rank Part 1 glossary · "Wren" in its title: wren; '
    "in its text: wren Words searched: wren. Ignored (common words and single letters or "
    "digits): tell, me, about, the. Found when you sent this message. Regenerating reuses "
    "these parts; it doesn't search again. Show in Details",
    "From documents · nothing matched No part of glossary has any of the words searched, so "
    'nothing was added. Search matches words and close forms of them ("sing" finds '
    '"singing"), not meaning. Words searched: herons. Ignored (common words and single '
    "letters or digits): any. Searched when you sent this message. Regenerating doesn't "
    "search again. Show in Details",
]
V032_SECTIONS = {
    "call-2cad462f3995": "Documents 1 part · 186 of 1,500 tokens Searched for “Tell me about "
    "the wren” Words searched wren Ignored tell, me, about, the Where start of your message · "
    "tokens 24 to 207 Framing line tokens 24 to 113 · 90 tokens, counted in the budget Search "
    "words, titles weighted double (word rule 1 · common words 1 · SQLite 3.52.0) Considered 1 "
    "part matched; Clarity considers the best 8 at most. Rank Part Why Tokens in the prompt 1 "
    'glossary · "Wren" in its title: wren; in its text: wren 116 to 207 · 92 tokens Rank 1 is '
    "the best match; a word in a title counts more than one in the text. Searched: glossary "
    "ccfd6f42… (each document’s build). Each part’s text and score are in this message’s log "
    "record.",
    "call-8d8d5c68615a": "Documents nothing matched Searched for “Any herons?” Words searched "
    "herons Ignored any Search words, titles weighted double (word rule 1 · common words 1 · "
    "SQLite 3.52.0) Rank 1 is the best match; a word in a title counts more than one in the "
    "text. Searched: glossary ccfd6f42… (each document’s build). Each part’s text and score "
    "are in this message’s log record.",
}


def test_a_words_chat_draws_v0_3_2_s_line_and_section_exactly(fetched_root, chat_tokenizer):
    logs = fetched_root / "logs"
    logs.mkdir(exist_ok=True)
    shutil.copy(GOLDEN, logs / GOLDEN.name)
    backend = FakeBackend([" More."], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        assert "result" in command(client, "resume", GOLDEN.stem).json()
        assert found_lines(client) == V032_LINES
        assert {c: documents_section(client, c) for c in V032_SECTIONS} == V032_SECTIONS
