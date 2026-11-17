"""What a search found, as "From documents" draws it (found_view, from /history's summary).
Its tests were the Documents… page's (C1) until cut 2 took the page away. Made-up parts."""

from __future__ import annotations

from clarity_chat import web
from clarity_chat.session import found_summary


def made_up_part(rank: int, **extra) -> dict:
    return {"id": f"g:{rank}", "document": "g", "ordinal": rank, "title": None, "page": None,
            "rank": rank, "title_words": [], "words": ["wren"], "tokens": 50, **extra}  # fmt: skip


def made_up_found(parts: list[dict], left: list[tuple[int, str]], matched: int) -> dict:
    """/history's summary (found_view's input since C3) of a made-up `found`."""
    return found_summary({"words": ["wren"], "ignored": [], "matched": matched,
            "considered": 8, "budget": 1500, "tokens": sum(p["tokens"] for p in parts),
            "parts": parts, "documents": [{"name": "g"}],
            "left_out": [{"id": f"g:{r}", "rank": r, "score": -1.0, "why": why}
                         for r, why in left]}, framed=True)  # fmt: skip


def test_found_view_s_notes_and_already_rows():  # should 2
    found = made_up_found([made_up_part(1)], [(2, "already in the chat"), (3, "budget"),
                          (4, "budget")], 12)  # fmt: skip
    view = web.found_view(found)
    assert view["summary"] == "1 part · 50 tokens · 1 more already in the chat"
    assert [(r["rank"], r["label"], r["why"]) for r in view["rows"]] == [
        (1, "g · part 1", "in its text: wren"),
        (2, "g · part 2", "already in the chat: not added again"),
    ]
    assert view["notes"] == [
        "2 more matched but were left out: Clarity adds parts in order and stops at the first "
        "that doesn't fit the budget of 1,500 tokens.",
        "4 more matched but weren't considered: Clarity considers only the best 8 matches.",
        "The first part's count includes the line Clarity puts before the parts.",
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
