"""C3 (v0.3.2): "From documents" above a reply and its list, Details' Documents section,
the refusal box at Send with its buttons (W6), and "New chat with these documents", against
the frozen board (② ③ ⑤, N29-N36, N44-N66) and its rulings. Made-up documents only."""

from __future__ import annotations

import re
import shutil

from clarity_chat import web
from clarity_chat.documents import library_add
from clarity_chat.session import found_summary
from tests.test_web import command
from tests.test_web_search_c2 import GLOSSARY, get, make, root, text_of  # noqa: F401

SEARCHES = 'search = ["glossary"]'


def thread(client) -> str:
    return get(client, "thread")


def found_line(client) -> str:
    """The last "From documents" line and its list, as text."""
    lines = re.findall(r'<div class="found".*?<!-- /found -->', thread(client), re.S)
    assert lines, "no From documents line"
    return text_of(lines[-1])


def last_call(client) -> str:
    turns = command(client, "state").json()["result"]["turns"]
    return next(t["call_id"] for t in reversed(turns) if t["role"] == "assistant")


# --- the record behind the line: /history's found widens (no text)


def test_history_s_found_gives_the_list_what_it_draws(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    command(client, "say", "And the wren again")
    turns = command(client, "history").json()["result"]["turns"]
    first, again = [t["found"] for t in turns if t["role"] == "user"]
    assert first.get("considered") == 8
    assert [p.get("page") for p in first["parts"]] == [None]
    assert again["left_out"].get("already in the chat") == 1


# --- "From documents" above a reply


def test_the_line_says_what_was_added_and_opens_on_its_list(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    page = thread(client)
    assert re.search(r'<button[^>]*data-action="found"[^>]*aria-expanded="false"', page)
    said = found_line(client)
    assert re.match(r"From documents · 1 part · [\d,]+ tokens Rank Part 1 glossary · "
                    r'"Wren" in its title: wren; in its text: wren', said)  # fmt: skip
    ignored = "Ignored (common words and single letters or digits): tell, me, about, the."
    assert f"Words searched: wren. {ignored}" in said
    assert said.endswith(
        "Found when you sent this message. Regenerating reuses these parts; it doesn't search "
        "again. Show in Details"
    )
    call = last_call(client)
    assert re.search(rf'data-action="attached" data-reply="{call}" data-section="documents"', page)


def test_nothing_matched_says_so_and_teaches(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about herons")
    said = found_line(client)
    assert said.startswith("From documents · nothing matched")
    assert (
        "No part of glossary has any of the words searched, so nothing was added. Search matches "
        'words and close forms of them ("sing" finds "singing"), not meaning.' in said
    )
    assert "Searched when you sent this message. Regenerating doesn't search again." in said


def test_nothing_to_search_for(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Is it the?")
    said = found_line(client)
    assert said.startswith("From documents · nothing to search for")
    assert (
        "Every word in your message is a common word or a single letter or digit, so nothing "
        "was searched and nothing was added." in said
    )


def test_a_part_already_in_the_chat_is_counted(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    command(client, "say", "And the wren again")
    said = found_line(client)
    assert said.startswith("From documents · nothing new · 1 already in the chat")
    assert "not added again" not in said  # a count, not a row (cut 4)


def test_a_chat_that_searches_nothing_shows_no_line(root, make):  # noqa: F811
    client, _ = make()
    command(client, "say", "Tell me about the wren")
    assert 'data-action="found"' not in thread(client)


# --- Details' Documents section


def test_prompt_gives_the_call_s_found(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    read = command(client, "prompt").json()["result"]
    assert (read.get("found") or {}).get("searched_for") == "Tell me about the wren"


def test_details_documents_section(root, make):  # noqa: F811
    library_add(root, b"## Heron\nTall.\n", "herons.md")
    client, _ = make('search = ["glossary", "herons"]')
    command(client, "say", "Tell me about the wren")
    call = command(client, "prompt").json()["result"]["call"]
    added = [a for s in call["message_spans"] for a in s.get("added", [])]
    framing = next(a["tokens"] for a in added if a["what"] == "framing")
    part = next(a["tokens"] for a in added if a["what"] == "part")
    details = get(client, "details", call=call["call_id"])
    section = re.search(r'<details class="sec" data-key="documents".*?</details>', details, re.S)
    assert section, "no Documents section"
    said = text_of(section.group(0))
    assert re.match(r"Documents 1 part · [\d,]+ of 1,500 tokens", said)
    assert "Searched for “Tell me about the wren”" in said
    assert "Words searched wren Ignored tell, me, about, the" in said
    assert f"Where start of your message · tokens {framing[0]:,} to {part[1] - 1:,}" in said
    assert (
        f"Framing line tokens {framing[0]:,} to {framing[1] - 1:,} · "
        f"{framing[1] - framing[0]:,} tokens, counted in the budget" in said
    )
    assert re.search(r"Search words, titles weighted double \(word rule 1 · common words 1 · "
                     r"SQLite [\d.]+\)", said)  # fmt: skip
    assert "Considered 1 part matched; Clarity considers the best 8 at most." in said
    assert "Left out" not in said
    assert (
        f'Rank Part Why Tokens in the prompt 1 glossary · "Wren" in its title: wren; in its '
        f"text: wren {part[0]:,} to {part[1] - 1:,} · {part[1] - part[0]:,} tokens" in said
    )
    assert re.search(
        r"Rank 1 is the best match; a word in a title counts more than one in the text\. "
        r"Searched: glossary "
        r"[0-9a-f]{8}… · herons [0-9a-f]{8}… \(each document’s build\)\. Each part’s text and "
        r"score are in this message’s log record\.",
        said,
    )


def test_details_when_nothing_matched(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about herons")
    details = get(client, "details", call=last_call(client))
    section = re.search(r'<details class="sec" data-key="documents".*?</details>', details, re.S)
    assert section, "no Documents section"
    said = text_of(section.group(0))
    assert said.startswith("Documents nothing matched")
    assert "Considered" not in said and "Where" not in said and "Rank Part" not in said


def test_details_names_what_was_left_out(root, make):  # noqa: F811
    client, _ = make(SEARCHES + "\nbudget = 1")
    command(client, "say", "Tell me about the wren")
    said = text_of(get(client, "details", call=last_call(client)))
    assert "Documents nothing added" in said
    assert "Left out rank 1, glossary part 1: the budget (1, this profile)" in said


# --- the refusal box at Send (W6) and the ContextError box


def test_a_refused_send_says_nothing_was_sent_with_a_line_and_buttons(root, make):  # noqa: F811
    library_add(root, b"## Heron\nTall.\n", "herons.md")
    client, _ = make('search = ["glossary", "herons"]')
    (root / "local/documents/glossary/parts.jsonl").write_text("Edited.\n", encoding="utf-8")
    shutil.rmtree(root / "local/documents/herons")
    assert command(client, "say", "Tell me about the wren").json()["type"] == "SearchRefused"
    box = re.search(r'<div class="problem refused".*?<!-- /refused -->', thread(client), re.S)
    assert box, "no refusal box"
    said = text_of(box.group(0))
    assert said.startswith("! Nothing was sent. Clarity can't search glossary: its parts.jsonl")
    assert 'data-action="refused-stop" data-name="glossary"' in box.group(0)
    assert 'data-action="refused-stop" data-name="herons"' in box.group(0)
    assert "! Failed" not in said
    command(client, "search", "glossary")  # what herons' Stop searching it sends
    page = thread(client)
    assert "herons" not in text_of(page) and "Clarity can't search glossary" in text_of(page)
    library_add(root, GLOSSARY, "glossary.md")  # cut 1: the file added again
    after = text_of(thread(client))  # no line left: the box goes, and Send works
    assert "Nothing was sent." not in after and "Failed" not in after


def test_a_full_context_offers_a_new_chat_with_these_documents(root, make):  # noqa: F811
    client, _ = make(SEARCHES, n_ctx=40)
    command(client, "say", "Tell me about the wren " * 3)
    page = thread(client)
    assert 'data-action="new-with-documents"' in page
    assert "New chat with these documents" in text_of(page)
    assert (
        "Starts a new chat on this profile that searches glossary. This chat stays in Chats."
        in text_of(page)
    )


# --- the part 2 whole-branch review's S4: C3's Checks with no test


HOSTILE = (
    b"## Wren <img src=x onerror=alert(1)>\nThe wren <b>sings</b> &amp; <script>x()</script>.\n"
)


def test_a_hostile_part_stays_text_in_the_line_details_and_chooser(root, make):  # noqa: F811
    library_add(root, HOSTILE, "odd.md")
    client, _ = make('search = ["glossary", "odd"]')
    command(client, "say", "Tell me about the wren")
    pages = [
        thread(client),
        get(client, "details", call=last_call(client)),
        get(client, "picker", library="1"),
    ]
    for page in pages:
        assert "<img src=x" not in page and "<script>x()" not in page and "<b>sings" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in pages[0] + pages[1]
    assert 'odd · "Wren <img src=x onerror=alert(1)>"' in found_line(client)


def test_the_line_reads_the_same_after_a_resume(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    before = found_line(client)
    log_id = command(client, "state").json()["result"]["log_id"]
    command(client, "new", "")
    assert "error" not in command(client, "resume", log_id).json()
    assert found_line(client) == before


def test_details_highlights_exactly_each_part_s_range(root, make):  # noqa: F811
    import html

    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    call = command(client, "prompt").json()["result"]["call"]
    text = call["prompt_text"]
    added = [
        text[a["chars"][0] : a["chars"][1]]
        for s in call["message_spans"]
        for a in s.get("added", [])
    ]
    page = get(client, "details", call=call["call_id"])
    pane = page.split('data-pane="text"', 1)[1].split("</pre>", 1)[0]
    marks = [html.unescape(m) for m in re.findall(r'<mark class="added">(.*?)</mark>', pane, re.S)]
    assert marks == added and any("<part " in m for m in marks)


# --- found_view on made-up summaries (from the Documents… page's tests, before cut 2)


def made_up_part(rank: int, **extra) -> dict:
    return {"id": f"g:{rank}", "document": "g", "ordinal": rank, "title": None, "page": None,
            "rank": rank, "title_words": [], "words": ["wren"], "tokens": 50, **extra}  # fmt: skip


def made_up_found(parts: list[dict], left: list[tuple[int, str]], matched: int) -> dict:
    """/history's summary (found_view's input since C3) of a made-up `found`."""
    return found_summary({"words": ["wren"], "ignored": [], "matched": matched,
            "considered": 8, "budget": 1500, "tokens": sum(p["tokens"] for p in parts),
            "parts": parts, "documents": [{"name": "g"}],
            "left_out": [{"id": f"g:{r}", "rank": r, "score": -1.0, "why": why}
                         for r, why in left]})  # fmt: skip


def test_found_view_s_rows_and_notes():  # should 2; one number per part since cut 4
    found = made_up_found([made_up_part(1)], [(2, "already in the chat"), (3, "budget"),
                          (4, "budget")], 12)  # fmt: skip
    view = web.found_view(found)
    assert view["summary"] == "1 part · 50 tokens · 1 more already in the chat"
    assert view["rows"] == [{"rank": 1, "label": "g · part 1", "why": "in its text: wren"}]
    assert view["notes"] == [
        "2 more matched but were left out: Clarity adds parts in order and stops at the first "
        "that doesn't fit the budget of 1,500 tokens.",
        "4 more matched but weren't considered: Clarity considers only the best 8 matches.",
    ]


def test_found_view_s_nothing_new_and_nothing_fit():  # should 2
    new = web.found_view(made_up_found([], [(1, "already in the chat")], 1))
    assert new["summary"] == "nothing new · 1 already in the chat"
    fit = web.found_view(made_up_found([], [(1, "budget"), (2, "budget")], 2))
    assert fit["summary"] == "nothing added · 2 didn't fit the budget"
    assert fit["notes"][0].startswith("2 matched but were left out:")


def test_already_and_budget_together_say_nothing_fit():  # the part 2 runner's #309
    found = made_up_found([], [(1, "already in the chat"), (2, "budget")], 2)
    assert web.found_view(found)["summary"] == "nothing added · 1 didn't fit the budget"


def test_a_part_s_label_names_its_page():  # should 2
    assert web.part_label(made_up_part(5, page=3)) == "g · page 3 · part 5"
    assert web.part_label(made_up_part(5, page=3, title="Wren")) == 'g · "Wren", page 3'
