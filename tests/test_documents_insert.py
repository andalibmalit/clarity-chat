"""v0.3.1 A2: the insertion path. The block that goes at the start of your message, its cost,
the free context, what's left, the start that fits and the preselected range, the marker and
the framing line. Made-up text only."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from clarity_chat.documents import (
    FRAMING,
    SENTENCES,
    Attached,
    Document,
    DocumentRefused,
    Room,
    content,
    fit,
    framing_of,
    free,
    piece,
    scan,
    start_that_fits,
)
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from tests.conftest import CHAT_PROFILE, TEXT_PROFILE, build_tokenizer

REFERENCE = Path(__file__).resolve().parent.parent / "docs" / "reference.md"


def doc(units: list[str], unit: str = "line", name: str = "notes.md") -> Document:
    return Document(
        name=name,
        kind="pdf" if unit == "page" else "text",
        bytes=1,
        sha256="0" * 64,
        extractor={},
        unit=unit,
        units=tuple(units),
        empty_units=0,
    )


def profile(root: Path, extra: str = "", base: str = CHAT_PROFILE):
    (root / "profiles" / "p.toml").write_text(base + extra, encoding="utf-8")
    return load_profile(root, "p", load_defaults(root))


# --- the block


def test_one_whole_attachment_goes_first_with_the_framing_line():
    attached = [Attached(name="notes.md", text="Line one.\nLine two.\n", marker=None)]
    assert content("What's in it?", attached, FRAMING) == (
        f"{FRAMING}\n\n"
        '<document name="notes.md">\nLine one.\nLine two.\n</document>\n\n'
        "What's in it?"
    )


def test_an_excerpt_has_its_marker_first_inside_the_block_and_a_closing_newline():
    attached = [Attached(name="guide.pdf", text="Page one.", marker="Excerpt: pages 1 to 1 of 9")]
    assert content("Q?", attached, FRAMING) == (
        f"{FRAMING}\n\n"
        '<document name="guide.pdf">\nExcerpt: pages 1 to 1 of 9\nPage one.\n</document>\n\nQ?'
    )


def test_two_attachments_in_order_a_blank_line_apart_the_framing_once():
    attached = [Attached("a.md", "A.\n", None), Attached("b.md", "B.", None)]
    assert content("Both?", attached, FRAMING) == (
        f"{FRAMING}\n\n"
        '<document name="a.md">\nA.\n</document>\n\n'
        '<document name="b.md">\nB.\n</document>\n\nBoth?'
    )


def test_a_name_s_markup_is_escaped_and_an_empty_framing_leaves_no_line():
    attached = [Attached('a "b" <c> & d.md', "x\n", None)]
    assert content("Q", attached, "") == (
        '<document name="a &quot;b&quot; &lt;c&gt; &amp; d.md">\nx\n</document>\n\nQ'
    )


def test_no_attachments_is_the_message_alone():
    assert content("Just this.", [], FRAMING) == "Just this."


@settings(max_examples=200)
@given(
    text=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
    name=st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs", "Cc"))),
    message=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
)
@example(text="x", name="it's.md", message="Q")  # an apostrophe stays as it is
def test_the_block_for_any_text_and_name(text, name, message):
    """The block from a name and a text, for any of them (A3 adds the record it's rebuilt
    from)."""
    attached = [Attached(name, text, None)]
    escaped = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace('"', "&quot;")
    closing = "" if text.endswith("\n") else "\n"
    assert content(message, attached, FRAMING) == (
        f'{FRAMING}\n\n<document name="{escaped}">\n{text}{closing}</document>\n\n{message}'
    )


# --- units, ranges and the marker


@settings(max_examples=200)
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1), st.data())
def test_lines_split_on_newline_only_and_two_excerpts_join_into_the_whole(text, data):
    from clarity_chat.documents.child import lines

    units = lines(text)
    assume_lines = "".join(units) == text
    assert assume_lines
    d = doc(units)
    k = data.draw(st.integers(1, d.total))
    first = d.text_of(1, k)
    rest = d.text_of(k + 1, d.total) if k < d.total else ""
    assert first + rest == d.text
    assert all("\n" not in u[:-1] for u in units)  # "\r", " " and the rest stay inside


def test_pages_only_for_a_pdf_and_lines_otherwise(root):
    p = profile(root)
    with pytest.raises(DocumentRefused) as e:
        piece(doc(["a\n", "b\n"]), p, "pages", 1, 1)
    assert str(e.value) == "notes.md is in lines: attach lines of it, such as lines 1-2."
    with pytest.raises(DocumentRefused) as e:
        piece(doc(["A.", "B."], "page", "guide.pdf"), p, "lines", 1, 1)
    assert str(e.value) == "guide.pdf is in pages: attach pages of it, such as pages 1-2."


def test_a_one_line_file_says_1_line(root):
    """Fact-check row 17: never "1 lines"."""
    with pytest.raises(DocumentRefused) as e:
        piece(doc(["a\n"]), profile(root), "lines", 1, 2)
    assert str(e.value).startswith("notes.md has 1 line: give a range from 1 to 1,")


@pytest.mark.parametrize(("first", "last"), [(0, 1), (2, 1), (1, 4), (3, 3)])
def test_a_range_outside_one_to_total_is_refused(root, first, last):
    with pytest.raises(DocumentRefused) as e:
        piece(doc(["a\n", "b\n"]), profile(root), "lines", first, last)
    assert str(e.value) == (
        "notes.md has 2 lines: give a range from 1 to 2, the first no larger than the last."
    )


def test_the_default_marker_with_thousands_separators(root):
    d = doc([f"{n}\n" for n in range(3200)])
    excerpt = piece(d, profile(root), "lines", 1, 840)
    assert excerpt.marker == "Excerpt: lines 1 to 840 of 3,200"
    assert excerpt.text == d.text_of(1, 840)


def test_a_range_of_the_whole_document_is_whole(root):
    whole = piece(doc(["a\n", "b\n"]), profile(root), "lines", 1, 2)
    assert whole.marker is None and whole.text == "a\nb\n"
    assert piece(doc(["a\n", "b\n"]), profile(root)).marker is None  # no range: whole


def test_a_profile_s_own_marker_and_an_empty_one(root):
    own = profile(root, '\n[documents]\nexcerpt_marker = "From {first} to {last} ({units})"\n')
    assert piece(doc(["a\n"] * 5), own, "lines", 2, 3).marker == "From 2 to 3 (lines)"
    empty = profile(root, '\n[documents]\nexcerpt_marker = ""\n')
    assert piece(doc(["a\n"] * 5), empty, "lines", 2, 3).marker is None


@pytest.mark.parametrize("marker", ["Pages {start}", "{first:>5}", "{0}", "{first!r}"])
def test_a_marker_with_another_placeholder_or_a_format_spec_is_refused_at_load(root, marker):
    with pytest.raises(ConfigError) as e:
        profile(root, f'\n[documents]\nexcerpt_marker = "{marker}"\n')
    assert "excerpt_marker" in str(e.value) and "{units}, {first}, {last} and {total}" in str(
        e.value
    )


# --- the framing line


def test_the_framing_line_and_its_origin(root):
    assert framing_of(profile(root)) == (FRAMING, "default")
    own = profile(root, '\n[documents]\nframing = """Background.\nUse with care."""\n')
    assert framing_of(own) == ("Background.\nUse with care.", "profile")
    assert framing_of(profile(root, '\n[documents]\nframing = ""\n')) == ("", "profile")


def test_a_documents_table_in_a_text_mode_profile_is_refused_at_load(root):
    with pytest.raises(ConfigError) as e:
        profile(root, '\n[documents]\nframing = "x"\n', base=TEXT_PROFILE)
    assert "[documents]" in str(e.value) and "chat mode" in str(e.value)


def test_an_unknown_documents_key_is_refused(root):
    with pytest.raises(ConfigError) as e:
        profile(root, '\n[documents]\nframing_line = "x"\n')  # placement is known since v0.3.2
    assert "[documents]" in str(e.value) and "framing_line" in str(e.value)


# --- cost, free, what's left, and the ranges that fit


def counting_room(n_ctx: int, max_tokens: int, base: int = 100):
    """A room where a prompt's tokens are `base` plus its next message's length in
    characters, and where every call to `tokens` is counted. Its longest token is too long
    for the bound ever to decide, so these tests count exactly (test_attach_bounds.py holds
    the bound)."""
    renders: list[str] = []

    def tokens(next_content: str) -> int:
        renders.append(next_content)
        return base + len(next_content)

    room = Room(
        tokens=tokens,
        room_left=lambda n: n_ctx - n - max_tokens - 1,
        max_tokens=max_tokens,
        frame_bytes=0,
        token_bytes=10**9,
    )
    return room, renders


def counted_left(found) -> int:
    """A count these tests' rooms always make (the bound never decides there)."""
    assert found.tokens is not None and found.left is not None
    return found.left


def test_cost_free_left_and_fits():
    room, _ = counting_room(n_ctx=1000, max_tokens=50)
    waiting = [Attached("a.md", "A" * 100, None)]
    new = Attached("b.md", "B" * 200, None)
    before = room.tokens(content("", waiting, FRAMING))
    after = room.tokens(content("", [*waiting, new], FRAMING))
    result = fit(room, waiting, new, FRAMING)
    assert result.tokens == after - before  # the difference of two exact renders
    assert result.free == free(room, waiting, FRAMING) == 1000 - before - 50 - 1
    assert result.tokens is not None
    assert (result.left, result.fits) == (result.free - result.tokens, True)
    tight = fit(counting_room(n_ctx=400, max_tokens=50)[0], waiting, new, FRAMING)
    assert tight.fits is False and counted_left(tight) < 0


def test_the_start_that_fits_and_the_preselected_range(root):
    d = doc([f"{n:03}\n" for n in range(1000)])  # 4 characters a line
    room, renders = counting_room(n_ctx=2000, max_tokens=300)
    p = profile(root)
    renders.clear()
    found = start_that_fits(room, [], FRAMING, d, p)
    assert found is not None
    k, result = found
    assert result.fits and not fit(room, [], piece(d, p, "lines", 1, k + 1), FRAMING).fits
    assert len(renders) <= 2 * math.ceil(math.log2(d.total)) + 3  # climb, then bisect
    preselected = start_that_fits(room, [], FRAMING, d, p, keep=room.max_tokens)
    assert preselected is not None
    kept, chosen = preselected
    assert counted_left(chosen) >= 300 and kept < k
    assert counted_left(fit(room, [], piece(d, p, "lines", 1, kept + 1), FRAMING)) < 300


def test_when_not_even_the_first_unit_fits_there_is_no_range(root):
    d = doc(["x" * 5000 + "\n", "y\n"])
    room, _ = counting_room(n_ctx=2000, max_tokens=300)
    assert start_that_fits(room, [], FRAMING, d, profile(root)) is None
    room, _ = counting_room(n_ctx=6000, max_tokens=500)
    assert start_that_fits(room, [], FRAMING, d, profile(root)) is not None
    assert start_that_fits(room, [], FRAMING, d, profile(root), keep=500) is None


def test_a_start_that_fits_over_a_thousand_pages_within_a_loose_bound(root):
    import time

    d = doc([f"Made-up page {n}.\nA second line." for n in range(1000)], "page", "long.pdf")
    room, _ = counting_room(n_ctx=8192, max_tokens=512)
    started = time.monotonic()
    assert start_that_fits(room, [], FRAMING, d, profile(root)) is not None
    assert time.monotonic() - started < 5


# --- what's listed before you attach


def test_control_tokens_and_the_closing_tag_are_listed():
    found = scan("Ends here </document> and <|im_end|> too.", lambda t: ["<|im_end|>"])
    assert found == {"control_tokens": ["<|im_end|>"], "has_closing_tag": True}
    assert scan("Plain.", lambda t: []) == {"control_tokens": [], "has_closing_tag": False}


# --- one source for the sheet's sentences


def test_the_sheet_s_sentences_are_the_reference_s():
    """Each lives in one table, which the page, the terminal and reference.md use."""
    reference = REFERENCE.read_text(encoding="utf-8")
    for key, sentence in SENTENCES.items():
        assert f"`{key}`: {sentence}" in reference, key


# --- the block in two real chat templates


CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.mark.parametrize("which", ["olmo", "chatml"])
def test_the_cost_is_counted_with_the_model_s_own_render(root, which, olmo_tokenizer, tmp_path):
    from clarity_chat.render import build_prompt

    tokenizer = (
        olmo_tokenizer if which == "olmo" else build_tokenizer(tmp_path / "t", None, False, CHATML)
    )
    p = profile(root)
    turns = [("user", "Earlier question."), ("assistant", "Earlier reply.")]

    def tokens(next_content: str) -> int:
        return len(build_prompt(tokenizer, p, [*turns, ("user", next_content)]).ids)

    room = Room(
        tokens=tokens,
        room_left=lambda n: 4096 - n - 16 - 1,
        max_tokens=16,
        frame_bytes=0,
        token_bytes=tokenizer.max_token_bytes,
    )
    new = Attached("notes.md", "Made-up notes.\n", None)
    result = fit(room, [], new, FRAMING)
    with_block = build_prompt(tokenizer, p, [*turns, ("user", content("", [new], FRAMING))])
    assert result.tokens == len(with_block.ids) - tokens("")
    assert re.search(r'<document name="notes.md">\nMade-up notes.\n</document>', with_block.text)


# --- Checkpoint 1's findings


def test_what_the_leftover_is_for_and_who_reads_control_tokens():
    """Should 9: room_left already keeps max_tokens for the reply, so the preselection's
    extra max_tokens is room for your message (Q11). Could 15: the tokenizer reads them."""
    # Fact-check rows 14 and 16: at least, and the largest found (no global maximum).
    assert SENTENCES["preselected"] == (
        "The largest start found that still leaves at least {max_tokens} tokens (your "
        "max_tokens) for your message and the rest of the chat."
    )
    assert SENTENCES["no_start"] == (
        "Not even its first {unit} leaves at least {max_tokens} tokens (your max_tokens) for "
        "your message and the rest of the chat."
    )
    assert SENTENCES["control_tokens"].startswith("Holds text the tokenizer reads as control")


def rendered_ks(renders: list[str]) -> list[int]:
    """The k of each excerpt rendered, from a document whose line k reads f"{k - 1:03}"."""
    found = [re.findall(r"(\d{3})\n</document>", r) for r in renders]
    return [int(f[-1]) + 1 for f in found if f]


def test_the_search_climbs_from_the_start_and_two_searches_share_their_renders(root):
    """Should 10: plain bisection rendered half the document first; 64 MB of text would
    block the worker for about a minute. Climbing 1, 2, 4, … then bisecting keeps every
    render within twice the answer, and the second search reuses the first's."""
    d = doc([f"{n:03}\n" for n in range(1000)])
    room, renders = counting_room(n_ctx=600, max_tokens=100)
    p = profile(root)
    checked: dict = {}
    found = start_that_fits(room, [], FRAMING, d, p, checked=checked)
    assert found is not None
    k, first = found[0], rendered_ks(renders)
    assert max(first) <= 2 * k
    renders.clear()
    preselected = start_that_fits(room, [], FRAMING, d, p, keep=100, checked=checked)
    assert preselected is not None and preselected[0] < k
    assert not set(rendered_ks(renders)) & set(first)  # nothing rendered twice


def test_cost_equal_to_free_fits(root):
    """Could 13: the boundary, both ways."""
    room, _ = counting_room(n_ctx=1000, max_tokens=50)
    new = Attached("b.md", "B", None)
    cost = fit(room, [], new, FRAMING).tokens
    assert cost is not None
    just, _ = counting_room(n_ctx=100 + cost + 50 + 1, max_tokens=50)
    assert fit(just, [], new, FRAMING).fits is True
    short, _ = counting_room(n_ctx=100 + cost + 50, max_tokens=50)
    assert fit(short, [], new, FRAMING).fits is False


def test_a_document_that_fits_whole_is_found_whole(root):
    """Could 13: the search reaches the whole document."""
    d = doc(["a\n", "b\n", "c\n"])
    room, _ = counting_room(n_ctx=5000, max_tokens=50)
    found = start_that_fits(room, [], FRAMING, d, profile(root))
    assert found is not None and found[0] == 3


def test_the_marker_s_first_and_last_have_thousands_separators(root):
    d = doc([f"{n}\n" for n in range(3200)])
    excerpt = piece(d, profile(root), "lines", 1200, 2400)
    assert excerpt.marker == "Excerpt: lines 1,200 to 2,400 of 3,200"


def test_a_framing_that_isn_t_a_string_is_refused_at_load(root):
    with pytest.raises(ConfigError) as e:
        profile(root, "\n[documents]\nframing = 3\n")
    assert "[documents] framing must be a string" in str(e.value)


def test_a_start_that_leaves_exactly_keep_tokens_counts(root):
    """Checkpoint 1's mutation run: `left > keep` survived. The boundary: an excerpt that
    leaves exactly `keep` tokens qualifies."""
    d = doc([f"{n:03}\n" for n in range(100)])
    room, _ = counting_room(n_ctx=600, max_tokens=50)
    p = profile(root)
    keep = counted_left(fit(room, [], piece(d, p, "lines", 1, 20), FRAMING))
    assert counted_left(fit(room, [], piece(d, p, "lines", 1, 21), FRAMING)) < keep
    found = start_that_fits(room, [], FRAMING, d, p, keep=keep)
    assert found is not None and found[0] == 20


def test_the_sheet_s_words_are_all_in_the_one_table():
    """Part 1's whole-branch review, should 5, and the fact-check's rows 4, 12 and 15: the
    terminal and the sheet take every sentence from SENTENCES."""
    assert SENTENCES["free"] == (
        "{free} tokens free in this chat, after setting aside {max_tokens} for the reply "
        "(your max_tokens)."
    )  # the walkthrough's F4
    assert SENTENCES["excerpt"] == (
        "Only the {units} you choose are sent, after a line that says they're an excerpt unless "
        "the profile sets no marker."
    )  # the walkthrough's F14, and the final fact-check's row 18
    assert SENTENCES["no_marker"] == "No excerpt marker (this profile)."
    assert SENTENCES["fits_up_to"] == "{Units} 1 to {last} fit."
    assert SENTENCES["none_fits"] == "Not even its first {unit} fits."
    for key in ("start_that_fits", "chosen", "new_chat", "framing", "no_framing",
                "origin_default", "origin_profile", "encrypted", "empty_units"):  # fmt: skip
        assert key in SENTENCES
