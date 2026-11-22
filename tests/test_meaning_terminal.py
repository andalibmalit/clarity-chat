"""v0.4.1 B1: meaning search in the terminal and --json (the states table, T and J).

The line after a searched message, /try's lines, /search's method and its warning when
bge-m3 isn't fetched (O1), the progress line and `building` events while vectors are built,
and states 5 to 7's refusals, each with a way out that works as written. FakeEmbedder
stands in for bge-m3 (tests/fakes.py names the live test behind each behaviour). Made-up
text only.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from clarity_chat.cli import ChatLoop
from clarity_chat.documents import SearchRefused, library_add, reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_json_chat import line, run
from tests.test_meaning import ASKED, GLOSSARY, MEANING, profile, raised

NOT_FETCHED = (  # state 5, the terminal's words
    "Meaning search needs bge-m3, which isn't fetched. In a new terminal window, in "
    "Clarity's folder, run: uv run clarity fetch bge-m3. Then try again. Or set "
    'search_method = "words" under [documents] in your own profile (copy a shipped one into '
    "local/profiles/ first) and start a new chat."
)


def floored(score: float) -> str:
    """A score as Clarity shows it under a two-place minimum: rounded down to 2 places."""
    return f"{math.floor(score * 100) / 100:.2f}"


def session_on(root, chat_tokenizer, embedder: Any) -> Session:
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    return Session(
        root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=embedder,
    )


@pytest.fixture
def terminal(fetched_root, chat_tokenizer):
    """A terminal chat on a profile that searches `notes` by meaning (min 0.2)."""
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    fake = FakeEmbedder()
    session = session_on(fetched_root, chat_tokenizer, fake)
    output: list[str] = []
    loop = ChatLoop(session, "meaning", read=lambda default: "", write=output.append)
    loop.start()
    output.clear()
    return loop, output, session, fake


def found_of(session: Session) -> dict[str, Any]:
    conversation = session.conversation
    assert conversation is not None and conversation.head is not None
    reply = conversation.turns[conversation.head]
    found = conversation.turns[reply.parent_id or ""].found
    assert found is not None
    return found


# --- states 2 and 3: the line after a message, and /try ------------------------------------


def test_the_line_after_a_message_says_what_meaning_search_added(terminal):
    loop, output, session, _ = terminal
    loop.handle(ASKED)
    found = found_of(session)
    added = ", ".join(
        f'{p["id"]} "{p["title"]}" (similar in meaning: {floored(p["score"])})'
        for p in found["parts"]
    )
    assert f"  searched by meaning: added {added} ({found['tokens']:,} tokens)\n" in output


def test_parts_that_matched_but_didn_t_fit_say_none_added(fetched_root, chat_tokenizer):
    """The whole-branch review's #14: a budget smaller than any part."""
    reading._last = None
    profile(fetched_root, "tight", MEANING + "budget = 1\n")
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    output: list[str] = []
    session = session_on(fetched_root, chat_tokenizer, FakeEmbedder())
    loop = ChatLoop(session, "tight", read=lambda default: "", write=output.append)
    loop.start()
    loop.handle(ASKED)
    found = found_of(session)
    assert found["matched"] and not found["parts"]
    assert f"  searched by meaning: {found['matched']:,} matched, none added\n" in output


def test_a_score_is_shown_to_the_minimum_s_places_and_never_below_it():
    """The whole-branch review's could 6 (the main session's ruling): rounded down to the
    larger of 2 places and the minimum's own, and a part at or above the minimum never
    reads below it."""
    from clarity_chat.cli import format_found, searched_line

    def found(score: float, minimum: float) -> dict:
        part = {"id": "notes:1", "title": "", "rank": 1, "score": score, "tokens": 9}
        return {
            "method": "meaning", "min_similarity": minimum, "matched": 1, "parts": [part],
            "tokens": 9, "budget": 1500, "considered": 8, "left_out": [],
        }  # fmt: skip

    for score, minimum, shown in [
        (0.5521, 0.552, "0.552"),
        (0.552, 0.552, "0.552"),
        (0.3, 0.3, "0.30"),  # the float nearest 0.3 is a little below it: still "0.30" (#20)
        (0.719, 0.5, "0.71"),  # rounded down, not to nearest
        (0.5, 0.5, "0.50"),
    ]:
        said = f"(similar in meaning: {shown})"
        assert said in searched_line(found(score, minimum)), (score, minimum)
        assert f"similar in meaning: {shown} (9 tokens)" in format_found(found(score, minimum))


def test_nothing_matched_by_meaning(fetched_root, chat_tokenizer):
    reading._last = None
    profile(fetched_root, "strict", MEANING.replace("0.2", "0.99"))
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    output: list[str] = []
    session = session_on(fetched_root, chat_tokenizer, FakeEmbedder())
    loop = ChatLoop(session, "strict", read=lambda default: "", write=output.append)
    loop.start()
    loop.handle(ASKED)
    assert "  searched by meaning: nothing matched\n" in output
    output.clear()
    loop.handle("/try " + ASKED)
    assert "Searched by meaning (minimum 0.99).\nNothing matched; the closest was" in "".join(
        output
    )


def test_try_lists_each_part_with_its_similarity(terminal):
    loop, output, session, _ = terminal
    found, _ = session.try_search(ASKED)
    output.clear()
    loop.handle("/try " + ASKED)
    shown = "".join(output)
    assert shown.startswith("Searched by meaning (minimum 0.2).\n")
    added = len(found["parts"])
    parts = "part" if added == 1 else "parts"
    assert (
        f"{found['matched']:,} matched; {added} {parts} added, {found['tokens']:,} tokens of the "
        f"budget of {found['budget']:,}."
    ) in shown
    for p in found["parts"]:
        assert (
            f'  {p["rank"]}. {p["id"]} "{p["title"]}": similar in meaning: {floored(p["score"])} '
            f"({p['tokens']:,} tokens)"
        ) in shown


# --- state 14 and O1: /search ---------------------------------------------------------------


def test_search_says_the_method(terminal):
    loop, output, session, _ = terminal
    loop.handle("/search")
    assert "Searching by meaning (the profile's list): notes.\n" in output
    loop.handle("/profile chat")
    loop.handle("/search notes")
    output.clear()
    loop.handle("/search")
    assert "Searching by words (set in this chat): notes.\n" in output


def test_search_warns_when_bge_m3_isn_t_fetched_and_not_when_it_is(terminal):
    loop, output, session, fake = terminal
    assert loop.commands.run("search", "", lambda e: None).get("warnings") == []
    fake.is_fetched = False
    result = loop.commands.run("search", "", lambda e: None)
    assert result.get("warnings") == [{"state": "no_embedder"}]
    output.clear()
    loop.handle("/search")
    assert f"! {NOT_FETCHED}\n" in "".join(output)  # under /search's line, one block
    assert fake.made == 0 and fake.texts == []  # a file check: nothing started


# --- state 4: the progress line and `building` events ----------------------------------------


def test_the_first_search_says_it_prepares_the_document_once(terminal):
    loop, output, _, _ = terminal
    loop.handle(ASKED)
    said = "Preparing notes for meaning search: 3 parts, saved in local/documents/notes/.\n"
    assert output.count(said) == 1
    output.clear()
    loop.handle("Why does rock wear away?")
    assert not [o for o in output if o.startswith("Preparing")]  # read, not built


def test_json_sends_building_events(fetched_root, chat_tokenizer):
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    session = session_on(fetched_root, chat_tokenizer, FakeEmbedder())
    events = run(session, [line("profile", "meaning"), line("say", ASKED)])
    building = [e for e in events if e.get("event") == "building"]
    assert building == [
        {"event": "building", "name": "notes", "done": done, "total": 3} for done in range(4)
    ]


# --- states 5 to 7: refusals and their ways out ----------------------------------------------


def test_bge_m3_not_fetched_refuses_send_with_the_fetch_and_words(terminal, chat_tokenizer):
    loop, output, session, fake = terminal
    fake.is_fetched = False
    refused = raised(lambda: loop.commands.run("say", ASKED, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert refused.problems == [{"state": "no_embedder"}]
    assert refused.terminal == f"Nothing was sent.\n{NOT_FETCHED}"
    page = str(refused)  # the page's words, which the log keeps: no terminal command
    assert page.startswith("Nothing was sent.\nMeaning search needs bge-m3, which isn't fetched.")
    assert "uv run" not in page and "Getting started" in page
    assert fake.made == 0 and fake.texts == []
    # The way out, as written: search_method = "words" in the profile, then a new chat.
    profile(session.root, "meaning", MEANING.replace('"meaning"', '"words"'))
    loop.handle("/new")
    loop.handle("leaves light sugar")
    assert found_of(session)["method"] == "words"


def test_an_embedder_failure_refuses_with_the_reason_and_the_way_to_turn_search_off(terminal):
    loop, output, _, fake = terminal
    fake.fail_at = 1  # the first part's vector
    refused = raised(lambda: loop.commands.run("say", ASKED, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert refused.terminal == (
        "Nothing was sent.\nThe embedder sent a vector Clarity can't use: made up. "
        "`/search none` turns search off for this chat."
    )
    assert str(refused) == (
        "Nothing was sent.\nThe embedder sent a vector Clarity can't use: made up. "
        "To send without searching, stop searching every document in this chat."
    )


def test_a_message_past_8192_tokens_sends_after_search_none(terminal):
    """State 7's way out, as written: `/search none`, then the same message sends."""
    loop, output, session, _ = terminal
    long = "word " * 8191
    refused = raised(lambda: loop.commands.run("say", long, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert "Shorten it, or turn search off with `/search none`." in refused.terminal
    loop.handle("/search none")
    loop.commands.run("say", "word " * 100, lambda e: None)
    conversation = session.conversation
    assert conversation is not None and conversation.head is not None


def test_the_terminal_never_prints_a_part_s_text(terminal):
    loop, output, _, _ = terminal
    loop.handle(ASKED)
    loop.handle("/try " + ASKED)
    loop.handle("/history")
    shown = "".join(output)
    for text in ("How leaves turn light into sugar.", "How water and wind wear rock away."):
        assert text not in shown


def test_a_score_shows_for_any_minimum_and_any_number_a_log_holds():
    """The fix review's F2: rounding past 28 digits raised instead of answering."""
    from clarity_chat.commands import shown_score

    def said(score: float, minimum: float) -> str | Exception:
        try:
            return shown_score(score, minimum)
        except Exception as e:  # a red fails on the assertion below, not on this
            return e

    tiny = said(0.3, 1e-30)
    assert isinstance(tiny, str) and tiny.startswith("0.29999999999999998889"), tiny
    huge = said(1e30, 0.5)
    assert huge == "1000000000000000019884624838656.00", huge
