"""A4 (v0.3.2): the insertion path, extended (decisions 1, 9 and 10). Found parts go in the
block after the attachments; the budget fill; no duplicates; placement at the end; the
attach sheet's suggested start setting a searching chat's budget aside. Made-up text only."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from clarity_chat.documents import (
    FRAMING,
    Attached,
    Room,
    added_ranges,
    content,
    fill,
    message_content,
    search_documents,
)
from tests.test_documents_search import loaded
from tests.test_session_attach import chat_session, doc

PART = {"document": "birds", "title": "Winter wren: song", "page": 41, "text": "It sings."}


def counting_room(budget_room: int = 10**6) -> tuple[Room, list[str]]:
    """A room counting one token a character, keeping each render."""
    renders: list[str] = []

    def tokens(text: str) -> int:
        renders.append(text)
        return len(text)

    room = Room(tokens, lambda n: budget_room - n, 16, frame_bytes=0, token_bytes=10**9)
    return room, renders


# --- the block, as the Shapes table gives it


def test_a_part_s_block_exactly():
    text = content("Why?", [], FRAMING, [PART])
    assert text == (
        f"{FRAMING}\n\n"
        '<part document="birds" title="Winter wren: song" page="41">\nIt sings.\n</part>'
        "\n\nWhy?"
    )


def test_a_part_without_a_title_or_page_leaves_them_out_and_escapes_the_rest():
    part = {"document": "a&b", "title": None, "page": None, "text": 'Says "hi" <b>\n'}
    assert content("Q", [], "", [part]) == (
        '<part document="a&amp;b">\nSays "hi" <b>\n</part>\n\nQ'
    )
    titled = {**part, "title": 'T "x" & y'}
    assert '<part document="a&amp;b" title="T &quot;x&quot; &amp; y">' in content(
        "Q", [], "", [titled]
    )


def test_attachments_then_parts_and_at_the_end_the_message_first():
    attached = [Attached("notes.md", "Line.\n", None)]
    start = content("Why?", attached, FRAMING, [PART])
    end = content("Why?", attached, FRAMING, [PART], "end")
    block = start.removesuffix("\n\nWhy?")
    assert start.index('<document name="notes.md">') < start.index("<part ")
    assert end == "Why?\n\n" + block


def test_placement_end_for_an_attachment_alone():
    attached = [Attached("notes.md", "Line.\n", None)]
    assert content("Why?", attached, FRAMING, [], "end") == (
        f'Why?\n\n{FRAMING}\n\n<document name="notes.md">\nLine.\n</document>'
    )


FIELDS = st.fixed_dictionaries(
    {
        "attachments": st.lists(
            st.fixed_dictionaries(
                {
                    "name": st.text(min_size=1, max_size=8),
                    "text": st.text(max_size=30),
                    "marker": st.one_of(st.none(), st.text(max_size=10)),
                }
            ),
            max_size=2,
        ),
        "framing": st.text(max_size=20),
        "placement": st.sampled_from(["start", "end"]),
        "found": st.fixed_dictionaries(
            {
                "parts": st.lists(
                    st.fixed_dictionaries(
                        {
                            "document": st.text(min_size=1, max_size=8),
                            "title": st.one_of(st.none(), st.text(max_size=10)),
                            "page": st.one_of(st.none(), st.integers(1, 999)),
                            "text": st.text(max_size=30),
                        }
                    ),
                    max_size=3,
                )
            }
        ),
    }
)


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=30), FIELDS)
def test_the_block_is_rebuilt_from_the_record_and_its_ranges_slice_it(text, fields):
    whole = message_content(text, fields)
    ranges = added_ranges(text, fields)
    if not fields["attachments"] and not fields["found"]["parts"]:
        assert (whole, ranges) == (text, [])
        return
    pieces = [whole[a:b] for a, b in (r["chars"] for r in ranges)]
    if fields["placement"] == "end":
        assert whole == "\n\n".join([text, *pieces])
    else:
        assert whole == "\n\n".join([*pieces, text])
    kinds = [r["what"] for r in ranges]
    parts = len(fields["found"]["parts"])
    assert kinds[len(kinds) - parts :] == ["part"] * parts


# --- the fill


def matches(n: int, size: int = 100) -> dict:
    """A search result with n matches, each part `size` characters of made-up text."""
    return {
        "words": ["x"],
        "matches": [
            {
                "id": f"d:{i}",
                "document": "d",
                "ordinal": i,
                "title": None,
                "page": None,
                "text": f"{i:>4} " + "x" * (size - 5),
                "sha256": f"sha-{i}",
                "rank": i,
                "score": -10.0 + i,
            }
            for i in range(1, n + 1)
        ],
    }


def test_the_fill_considers_eight_adds_in_order_and_stops_at_the_first_that_doesn_t_fit():
    room, renders = counting_room()
    # 128 characters a part in the block, the framing line counted with the first: three fit 520.
    filled = fill(room, "Q", [], FRAMING, matches(12), 520, "profile")
    assert [p["id"] for p in filled["parts"]] == ["d:1", "d:2", "d:3"]
    assert [(o["id"], o["why"]) for o in filled["left_out"]] == [
        *((f"d:{i}", "budget") for i in range(4, 9)),
        *((f"d:{i}", "not considered") for i in range(9, 13)),
    ]
    assert len(renders) <= 9
    assert (filled["considered"], filled["matched"], filled["budget"]) == (8, 12, 520)
    assert filled["budget_origin"] == "profile" and "matches" not in filled


def test_the_fill_renders_at_most_nine_times():
    room, renders = counting_room()
    filled = fill(room, "Q", [], FRAMING, matches(40, 10), 10**6, "default")
    assert len(filled["parts"]) == 8 and len(renders) == 9


def test_tokens_are_the_exact_cost_framing_and_wrappers_included():
    room, _ = counting_room()
    filled = fill(room, "Q", [], FRAMING, matches(3), 10**6, "default")
    added = filled["parts"]
    assert len(added) == 3  # else an empty fill would pass the sums below
    expected = len(content("Q", [], FRAMING, added)) - len("Q")
    assert filled["tokens"] == expected  # the framing line counts: search is why it's there
    assert sum(p["tokens"] for p in added) == expected


def test_with_attachments_the_framing_line_counts_with_them():
    room, _ = counting_room()
    attached = [Attached("notes.md", "Line.\n", None)]
    filled = fill(room, "Q", attached, FRAMING, matches(1), 10**6, "default")
    assert len(filled["parts"]) == 1
    without = len(content("Q", attached, FRAMING))
    assert filled["tokens"] == len(content("Q", attached, FRAMING, filled["parts"])) - without
    assert filled["tokens"] < len(FRAMING) + 100


def test_a_part_already_in_the_chat_isn_t_added_again_and_costs_nothing():
    room, renders = counting_room()
    filled = fill(
        room, "Q", [], FRAMING, matches(3), 10**6, "default", already=frozenset({"sha-1"})
    )
    assert [p["id"] for p in filled["parts"]] == ["d:2", "d:3"]
    assert filled["left_out"] == [
        {"id": "d:1", "rank": 1, "score": -9.0, "why": "already in the chat"}
    ]
    assert len(renders) == 3  # no render for the part already there


def test_nothing_found_adds_nothing():
    room, _ = counting_room()
    filled = fill(room, "Q", [], FRAMING, {"words": [], "matches": []}, 1500, "default")
    assert (filled["parts"], filled["tokens"], filled["matched"]) == ([], 0, 0)


def test_the_fill_with_a_real_search_puts_the_best_first():
    text = "## Wren\nA small bird.\n\n## Lake\nCold water.\n\n" + "## Stone\nClay.\n\n" * 6
    docs = [loaded("birds", text)]
    room, _ = counting_room()
    filled = fill(room, "wren", [], FRAMING, search_documents(docs, "wren"), 1500, "default")
    assert [p["id"] for p in filled["parts"]] == ["birds:1"]
    assert filled["parts"][0]["title_words"] == ["wren"]


# --- the attach sheet when the chat searches (S1)


def test_the_suggested_start_sets_the_budget_aside(fetched_root, chat_tokenizer):
    session, backend = chat_session(fetched_root, chat_tokenizer)
    backend._n_ctx = 2000
    long = doc("".join(f"Made-up line {n}.\n" for n in range(400)), "long.md")
    plain = session.check(long)["preselected"]
    searching = session.check(long, aside=500)
    assert searching["aside"] == 500
    assert searching["preselected"]["left"] >= 16 + 500
    assert searching["preselected"]["range"][1] < plain["range"][1]
