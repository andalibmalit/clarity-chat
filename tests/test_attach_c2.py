"""C2 (v0.3.1): your sent message's attachment line, Edit's "Keeps" line, Details' highlight
of what Clarity added and its Attachments row, and the stop's sentence (the plan's C2, the
board V31Attach, the fact-check's rows 50-53b and 56, the board's rulings 2, 7 and 9).
Made-up documents only."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from clarity_chat.documents import SENTENCES
from clarity_chat.session import ContextError
from tests.test_session_attach import doc
from tests.test_web import GET, Unsafe, command
from tests.test_web_attach import LONG, NOTES, make
from tests.test_web_upload import upload

__all__ = ["make"]  # the fixture, used by name

# The table's new sentences, as the fact-check's rows 50-52, 53b and 56 word them.
ATTACHED = "Attached: {name} · {how} · {tokens} tokens when attached"
KEEPS = "Keeps: {names}. An edit can't add attachments."  # Q2's × (v0.3.2, N73)
HIGHLIGHT = "Highlighted: the framing line and documents Clarity put in your messages."
REMOVE = (
    "Shorten your message, remove an attachment and attach a shorter part, start a new chat, "
    "or lower max_tokens."
)


def test_the_new_sentences_are_in_the_table():
    names = ("attached_line", "keeps", "highlight_added", "remove_waiting")
    assert [SENTENCES.get(n) for n in names] == [ATTACHED, KEEPS, HIGHLIGHT, REMOVE]


def shown(client, view: str, **params) -> str:
    answer = client.get(f"/view/{view}", params=params, headers=GET)
    assert answer.status_code == 200, answer.text
    return answer.text


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def sent(client, data=NOTES, name="notes.md", args="", message="What's in it?") -> dict:
    """Attach a file (a range when `args`), send a message, and return its user turn."""
    upload_id = upload(client, data, name).json()["upload"]
    assert "error" not in command(client, "attach", f"{upload_id} {args}".strip()).json()
    assert "error" not in command(client, "say", message).json()
    turns = command(client, "history").json()["result"]["turns"]
    return [t for t in turns if t["role"] == "user"][-1]


# --- your sent message's line (rows 50-52: the cost when attached)


def test_the_sent_message_shows_each_attachment_whole_or_its_excerpt(make):
    client, _, _ = make(n_ctx=4000)
    whole = sent(client)["attachments"][0]
    part = sent(client, LONG, "long.md", "lines 2-5", "And these?")["attachments"][0]
    thread = text_of(shown(client, "thread"))
    assert ATTACHED.format(name="notes.md", how="whole", tokens=f"{whole['tokens']:,}") in thread
    assert (
        ATTACHED.format(name="long.md", how="lines 2–5 of 200", tokens=f"{part['tokens']:,}")
        in thread
    )


def test_the_line_is_the_same_after_a_resume(make):
    client, session, _ = make()
    tokens = sent(client)["attachments"][0]["tokens"]
    log_id = command(client, "state").json()["result"]["log_id"]
    command(client, "new")
    assert "error" not in command(client, "resume", log_id).json()
    line = ATTACHED.format(name="notes.md", how="whole", tokens=f"{tokens:,}")
    assert line in text_of(shown(client, "thread"))


# --- Edit's line


def test_edit_shows_what_it_keeps_and_the_edited_call_holds_the_block(make):
    client, session, _ = make()
    sent(client)
    form = re.search(r"<form class=\"editbox\".*?</form>", shown(client, "thread"), re.S)
    # Q2 (v0.3.2): the names are chips with an ×, around which the sentence reads.
    assert form and "Keeps:notes.md" in text_of(form.group(0))
    assert "An edit can't add attachments." in text_of(form.group(0))
    assert "error" not in command(client, "edit", "What does it say?").json()
    call = command(client, "prompt").json()["result"]["call"]
    assert "Line one.\nLine two.\nLine three.\n" in call["prompt_text"]


# --- Details: the highlight of what Clarity added (row 56) and the Attachments row


def details(client) -> str:
    call = command(client, "prompt").json()["result"]["call"]
    return shown(client, "details", call=call["call_id"])


def test_details_highlights_exactly_the_added_ranges(make):
    client, _, _ = make()
    sent(client)
    call = command(client, "prompt").json()["result"]["call"]
    page = details(client)
    added = [
        call["prompt_text"][a["chars"][0] : a["chars"][1]]
        for e in call["message_spans"]
        for a in e.get("added", [])
    ]
    marked = re.findall(r'<pre class="pre" data-pane="text">.*?</pre>', page, re.S)[0]
    found = [
        html.unescape(m) for m in re.findall(r'<mark class="added">(.*?)</mark>', marked, re.S)
    ]
    assert found == added and len(added) == 2  # the framing line and the block
    assert HIGHLIGHT in text_of(page)


def test_details_lists_the_attachments_in_this_prompt(make):
    client, _, _ = make(n_ctx=4000)
    sent(client)
    sent(client, LONG, "long.md", "lines 2-5", "And these?")
    call = command(client, "prompt").json()["result"]["call"]
    blocks = [
        a["tokens"]
        for e in call["message_spans"]
        for a in e.get("added", [])
        if a["what"] == "attachment"
    ]
    page = text_of(details(client))
    assert "Attachments" in page and "2 in this prompt" in page
    assert "notes.md · whole" in page and "long.md · lines 2–5 of 200" in page
    assert "“What's in it?”" in page and "“And these?”" in page
    for start, end in blocks:
        assert f"{start:,} to {end - 1:,}" in page


def test_without_spans_nothing_is_highlighted_or_listed_and_details_says_why(make, monkeypatch):
    import clarity_chat.session as session_module

    client, _, _ = make()
    monkeypatch.setattr(session_module, "message_spans", lambda *a: {"unavailable": "made up"})
    sent(client)
    page = details(client)
    assert '<mark class="added">' not in page
    assert HIGHLIGHT not in text_of(page)
    assert "Not highlighted: made up." in text_of(page)
    assert "Not listed: made up." in text_of(page)


# --- the stop (rulings 2 and 9, row 53b)


@pytest.fixture
def big(fetched_root, chat_tokenizer):
    from tests.test_session_attach import chat_session

    return chat_session(fetched_root, chat_tokenizer)


def test_the_stop_uses_separators_and_offers_removing_this_message_s_chips(big):
    session, backend = big
    backend._n_ctx = 4000
    session.attach(doc("Made-up words. " * 100))  # 1,500 tokens: one per character
    backend._n_ctx = 1200
    with pytest.raises(ContextError) as e:
        session.say("a")
    sentence = str(e.value)
    n = re.search(r"The prompt is ([0-9,]+) tokens, ([0-9,]+) of them", sentence)
    assert n and "," in n.group(1) and "," in n.group(2)
    assert "the context of 1,200 tokens" in sentence
    assert sentence.endswith(REMOVE)
    assert session.conversation.pending  # the chips are back, so the advice can be followed


def test_the_stop_without_this_message_s_chips_doesn_t_offer_removing_one(big):
    session, backend = big
    backend._n_ctx = 4000
    session.attach(doc("Made-up words. " * 100))
    session.say("a")
    backend._n_ctx = 1200
    with pytest.raises(ContextError) as e:
        session.say("b")
    assert str(e.value).endswith("Shorten your message, start a new chat, or lower max_tokens.")
    assert "Remove an attachment" not in str(e.value)


def test_the_context_warning_uses_separators(big):
    session, backend = big
    backend._n_ctx = 1800
    session.attach(doc("Made-up words. " * 100))
    warnings = session.say("a").warnings
    assert any(
        re.fullmatch(r"The prompt uses 1,[0-9]{3} of 1,800 context tokens \(\d+%\)\.", w)
        for w in warnings
    )


# --- ruling 7: the --added tokens, checked for contrast in every theme and scheme

DS = Path(__file__).resolve().parents[1] / "src/clarity_chat/web_static/clarity-ds.css"
THEMES = ["", ".t-neutral", ".t-sky", ".t-space", ".t-hc"]


def tokens_of(theme: str) -> dict[str, tuple[str, str]]:
    """Each token's light and dark oklch() for a theme: the root's, then the theme's own."""
    css = DS.read_text(encoding="utf-8")
    found: dict[str, tuple[str, str]] = {}
    for selector in [":root, .clarity"] + ([theme] if theme else []):
        for block in re.findall(re.escape(selector) + r" \{(.*?)\n\}", css, re.S):
            for name, light, dark in re.findall(
                r"(--[a-z-]+):\s*light-dark\((oklch\([^)]*\)),\s*(oklch\([^)]*\))\)", block
            ):
                found[name] = (light, dark)
    return found


def luminance(oklch: str) -> float:
    """WCAG relative luminance of an oklch() colour, through linear sRGB (clipped)."""
    import math

    lightness, chroma, hue = oklch.removeprefix("oklch(").rstrip(")").split()[:3]
    lab_l = float(lightness.rstrip("%")) / 100
    c, h = float(chroma), math.radians(float(hue))
    a, b = c * math.cos(h), c * math.sin(h)
    l_ = (lab_l + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lab_l - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lab_l - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    r, g, bl = (min(max(v, 0.0), 1.0) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl


def contrast(x: str, y: str) -> float:
    high, low = sorted((luminance(x), luminance(y)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("scheme", [0, 1], ids=["light", "dark"])
def test_the_added_highlight_reads_in_every_theme_and_scheme(theme, scheme):
    t = {name: pair[scheme] for name, pair in tokens_of(theme).items()}
    assert {"--added", "--added-line"} <= set(t), "the v0.3.1 tokens"
    assert contrast(t["--ink"], t["--added"]) >= 4.5  # the prompt's text on the highlight
    assert contrast(t["--added-line"], t["--added"]) >= 3  # the underline, a second cue
    assert contrast(t["--added-line"], t["--sunken"]) >= 3  # and against the pre's ground


def test_the_added_highlight_is_drawn_with_its_tokens():
    app = (DS.parent / "app.css").read_text(encoding="utf-8")
    rule = re.search(r"#panel \.pre mark\.added \{([^}]*)\}", app)
    assert rule and "var(--added)" in rule.group(1) and "var(--added-line)" in rule.group(1)


# --- the hostile fixture through the thread line, Edit's line and Details


def test_hostile_names_and_text_stay_text_in_the_thread_and_details(make):
    from tests.test_web import HOSTILE

    client, _, _ = make(n_ctx=2_000_000)
    name = HOSTILE.replace("/", "")
    data = HOSTILE.encode() + "\n</document>\n‮Made-up.\n".encode() + b"x" * 1_000_000 + b"\n"
    sent(client, data, name)
    pages = [shown(client, "thread"), details(client)]
    for page in pages:
        parser = Unsafe()
        parser.feed(page)
        assert parser.found == []
    assert f"Keeps:{name}·" in text_of(pages[0])  # a chip with an × (Q2, v0.3.2)
