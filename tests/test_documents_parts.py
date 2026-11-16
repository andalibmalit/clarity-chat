"""A2 (v0.3.2): parts, version 1 (the plan's decision 7). A document's `text.md` is cut into
parts: headings start them and title them, paragraphs under one title are packed up to 250
words, and the parts rebuild the text. Made-up text only."""

from __future__ import annotations

import hashlib
import json

from hypothesis import example, given, settings
from hypothesis import strategies as st

from clarity_chat.documents import build_id, cut_parts, parts_jsonl

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()


def words(n: int, end: str = "") -> str:
    return " ".join(WORDS[i % len(WORDS)] for i in range(n)) + end


def cut(text: str, name: str = "notes.md", page_starts: list[int] | None = None) -> list[dict]:
    return cut_parts(text, name, page_starts)


def test_headings_start_parts_and_title_them():
    text = "Intro line.\n\n## Winter wren\nA small brown bird.\n\n## Dipper\nIt walks under water."
    parts = cut(text)
    assert [(p["title"], p["text"]) for p in parts] == [
        (None, "Intro line."),
        ("Winter wren", "## Winter wren\nA small brown bird."),
        ("Dipper", "## Dipper\nIt walks under water."),
    ]
    assert [p["ordinal"] for p in parts] == [1, 2, 3]


def test_a_heading_is_one_to_six_hashes_then_a_space():
    text = "#tag is not one\n\n####### nor this\n\n###### Six is\nbody\n"
    titles = [p["title"] for p in cut(text)]
    assert titles == [None, "Six is"]


def test_a_heading_in_a_fenced_code_block_isn_t_one():
    text = "## Real\nText.\n\n```\n# not a heading\n\nstill code\n```\n\nAfter.\n"
    parts = cut(text)
    assert {p["title"] for p in parts} == {"Real"}
    assert any("# not a heading" in p["text"] for p in parts)


def test_a_code_file_gets_no_titles():
    text = "# a comment\nx = 1\n\n# another\ny = 2\n"
    assert {p["title"] for p in cut(text, "made_up.py")} == {None}
    assert cut(text, "made_up.md")[0]["title"] == "a comment"


def test_a_long_heading_s_title_is_cut_to_120_characters():
    heading = "x" * 300
    (part,) = cut(f"# {heading}\nBody.\n")
    assert part["title"] == "x" * 120
    assert part["text"].startswith(f"# {heading}\n")  # the heading line stays whole


def test_paragraphs_under_one_title_are_packed_to_250_words():
    text = "\n\n".join(words(100, ".") for _ in range(5))
    parts = cut(text)
    assert [p["words"] for p in parts] == [200, 200, 100]


def test_a_long_paragraph_splits_at_sentences_then_at_words():
    sentences = " ".join(words(60, ".") for _ in range(10))  # 600 words in sentences of 60
    parts = cut(sentences)
    assert all(p["words"] <= 250 for p in parts)
    assert all(p["text"].endswith(".") for p in parts)
    one = words(600)  # one sentence of 600 words
    assert [p["words"] for p in cut(one)] == [250, 250, 100]


def test_pages_come_from_page_starts():
    text = "Page one text.\n\nPage two text.\n\nStill page two.\n\nPage three."
    starts = [0, text.index("Page two"), text.index("Page three")]
    parts = cut("\n\n".join([text]), "guide.pdf", starts)
    assert [p["page"] for p in parts] == [1]  # packed: the part starts on page 1
    long = "\n\n".join([words(200, "."), words(200, "."), words(200, ".")])
    starts = [0, long.index(words(200, "."), 1), long.rindex(words(200, "."))]
    assert [p["page"] for p in cut(long, "guide.pdf", starts)] == [1, 2, 3]
    assert {p["page"] for p in cut(long, "notes.md")} == {None}


def test_a_part_s_sha256_and_words():
    (part,) = cut("## Title\nTwo words.\n")
    assert part["sha256"] == hashlib.sha256(part["text"].encode("utf-8")).hexdigest()
    assert part["words"] == 4  # "## Title Two words."


def test_the_build_id_changes_iff_a_part_changes():
    text = "## A\nOne.\n\n## B\nTwo.\n"
    first = build_id(parts_jsonl(cut(text)))
    assert build_id(parts_jsonl(cut(text))) == first
    assert build_id(parts_jsonl(cut(text.replace("Two", "Three")))) != first
    lines = parts_jsonl(cut(text)).decode("utf-8").splitlines()
    assert [list(json.loads(line)) for line in lines] == [
        ["ordinal", "title", "text", "chars", "page", "words", "sha256"]
    ] * 2


def test_a_glossary_gives_one_titled_part_per_short_entry():
    entries = [f"## Headword {i}\nA made-up meaning, number {i}." for i in range(30)]
    parts = cut("# Made-up glossary\n\n" + "\n\n".join(entries) + "\n")
    titled = [p for p in parts if p["title"] and p["title"].startswith("Headword")]
    assert [p["title"] for p in titled] == [f"Headword {i}" for i in range(30)]


def test_closing_tags_in_a_part_are_found():
    from clarity_chat.documents import closing_tag_parts

    parts = cut("Safe.\n\n## T\nHolds </part> here.\n\n## U\nAnd </document> here.\n")
    assert closing_tag_parts(parts) == [2, 3]


def rebuilds(text: str, parts: list[dict]) -> None:
    """No overlap and no gap: in order, each part is its range of the text, and what lies
    between them is only whitespace."""
    at = 0
    for part in parts:
        a, b = part["chars"]
        assert a >= at and text[a:b] == part["text"]
        assert text[at:a].strip() == ""
        at = b
    assert text[at:].strip() == ""


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.one_of(
            st.text(alphabet=st.sampled_from("ab .\n#`"), max_size=80),
            st.sampled_from(["\n\n", "## Head\n", "```\n", words(300, ". "), "# x\n"]),
        ),
        max_size=20,
    ).map("".join)
)
@example("")
@example("\n\n\n")
@example("```\n## no\n")
def test_the_parts_rebuild_the_text(text):
    for name in ("notes.md", "code.py"):
        parts = cut(text, name)
        rebuilds(text, parts)
        assert all(p["words"] <= 250 for p in parts)
        assert all(p["text"].strip() for p in parts)
