"""The insertion path (v0.3.1 decision 1): every attachment, and later every search result,
ends as a block of text at the start of your message. A plain function of the message, its
attachments and the profile: it reads no file and keeps no state. Counting is passed in as
`Room`, so this package never needs the session or the renderer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

from clarity_chat.documents.reading import Document, DocumentRefused
from clarity_chat.profiles import Profile

FRAMING = (
    "Reference material: use it to understand the conversation; don't copy its wording or tone."
)
MARKER = "Excerpt: {units} {first} to {last} of {total}"
CLOSING_TAG = "</document>"

# The sheet's sentences, in one place: the page, the terminal and reference.md use these.
SENTENCES = {
    "whole": "It takes up {tokens} tokens of the context for the rest of the chat.",
    "excerpt": "An excerpt goes in whole, marked as an excerpt.",
    "left": "That leaves {left} tokens for your message and the rest of the chat.",
    "doesnt_fit": "Doesn't fit: needs {tokens} tokens, {free} free.",
    "preselected": "The start that still leaves {max_tokens} tokens (your max_tokens) for your "
    "message and the rest of the chat.",
    "no_start": "Not even its first {unit} leaves {max_tokens} tokens (your max_tokens) for your "
    "message and the rest of the chat.",
    "already": "This file is already in this chat.",
    "control_tokens": "Holds text the tokenizer reads as control tokens ({tokens}); it's sent as "
    "it is, and the call is flagged.",
    "closing_tag": "Holds </document>, which ends Clarity's wrapper early; it's sent as it is.",
}


@dataclass(frozen=True)
class Attached:
    """What one attachment puts in the block: its name, its marker line (an excerpt's, or
    None) and its text. The turn's record keeps these, so the block is rebuilt from it."""

    name: str
    text: str
    marker: str | None


@dataclass(frozen=True)
class Room:
    """How to count, from the session: the prompt's tokens when the next message's content
    is the text given, the free context left after a prompt of that many tokens
    (session.room_left), and max_tokens."""

    tokens: Callable[[str], int]
    room_left: Callable[[int], int]
    max_tokens: int


@dataclass(frozen=True)
class Fit:
    tokens: int  # what attaching it costs: the prompt's tokens with it, minus without
    free: int  # the free context now, with what's already waiting
    left: int  # free minus tokens: what your message and the rest of the chat can use
    fits: bool


def framing_of(profile: Profile) -> tuple[str, str]:
    """The framing line and its origin: the profile's own (which may be empty: none),
    else Clarity's default."""
    own = profile.framing
    return (FRAMING, "default") if own is None else (own, "profile")


def piece(
    doc: Document, profile: Profile, unit: str | None = None, first: int = 0, last: int = 0
) -> Attached:
    """The document whole (no range, or one that covers it all), or an excerpt of it with
    its marker. `unit` is "pages" or "lines", as typed."""
    if unit is None:
        return Attached(doc.name, doc.text, None)
    if unit != doc.unit + "s":
        n = min(doc.total, 2)
        raise DocumentRefused(
            f"{doc.name} is in {doc.unit}s: attach {doc.unit}s of it, such as {doc.unit}s 1-{n}."
        )
    if not 1 <= first <= last <= doc.total:
        raise DocumentRefused(
            f"{doc.name} has {doc.total:,} {doc.unit}s: give a range from 1 to {doc.total:,}, "
            "the first no larger than the last."
        )
    if (first, last) == (1, doc.total):
        return Attached(doc.name, doc.text, None)
    template = MARKER if profile.excerpt_marker is None else profile.excerpt_marker
    marker = template.format(
        units=unit, first=f"{first:,}", last=f"{last:,}", total=f"{doc.total:,}"
    )
    return Attached(doc.name, doc.text_of(first, last), marker or None)


def content(message: str, attached: list[Attached], framing: str) -> str:
    """The message's content as the model gets it: the framing line (unless empty), each
    attachment's block in order, then your message."""
    if not attached:
        return message
    parts = [framing] if framing else []
    for a in attached:
        marker = f"{a.marker}\n" if a.marker else ""
        closing = "" if a.text.endswith("\n") else "\n"
        name = escape(a.name, quote=True).replace("&#x27;", "'")
        parts.append(f'<document name="{name}">\n{marker}{a.text}{closing}{CLOSING_TAG}')
    return "\n\n".join([*parts, message])


def message_content(text: str, fields: Mapping[str, Any]) -> str:
    """A user message's content as the model gets it, from a turn's fields or its record:
    the text alone, or with its attachments' block. Every prompt, check and read builds it
    here. A record missing a field raises KeyError, which readers refuse as unreadable."""
    attachments = fields.get("attachments") or []
    if not attachments:
        return text
    attached = [Attached(a["name"], a["text"], a["marker"]) for a in attachments]
    return content(text, attached, fields["framing"])


def added_ranges(fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    return []


def scan(text: str, special_tokens: Callable[[str], list[str]]) -> dict[str, Any]:
    """What to list before attaching: control tokens the tokenizer reads in the text, and
    whether it holds the wrapper's closing tag. Neither is refused or escaped."""
    return {"control_tokens": special_tokens(text), "has_closing_tag": CLOSING_TAG in text}


def free(room: Room, pending: list[Attached], framing: str) -> int:
    """The free context with what's waiting and an empty next message."""
    return room.room_left(room.tokens(content("", pending, framing)))


def fit(
    room: Room, pending: list[Attached], new: Attached, framing: str, before: int | None = None
) -> Fit:
    """What attaching `new` costs, after what's waiting, and what it leaves. `before`: the
    prompt's tokens without it, if already counted."""
    if before is None:
        before = room.tokens(content("", pending, framing))
    after = room.tokens(content("", [*pending, new], framing))
    available = room.room_left(before)
    cost = after - before
    return Fit(tokens=cost, free=available, left=available - cost, fits=cost <= available)


def start_that_fits(
    room: Room,
    pending: list[Attached],
    framing: str,
    doc: Document,
    profile: Profile,
    keep: int = 0,
    checked: dict[int, Fit] | None = None,
) -> tuple[int, Fit] | None:
    """The largest k whose excerpt 1 to k leaves at least `keep` tokens (0: the start that
    fits; max_tokens: the preselected range). It climbs k = 1, 2, 4, … to the first that
    doesn't, then bisects between that and the last that did, so no render is longer than
    twice the answer. k is shown to fit and k+1 not to (or k is the whole); cost isn't
    proven to grow with k, so this claims no global maximum. None when not even the first
    unit does. `checked` keeps each k's exact render for a second call with the same
    pending attachments and framing."""
    unit = doc.unit + "s"
    before = room.tokens(content("", pending, framing))
    checked = {} if checked is None else checked

    def holds(k: int) -> bool:
        if k not in checked:
            checked[k] = fit(room, pending, piece(doc, profile, unit, 1, k), framing, before)
        return checked[k].fits and checked[k].left >= keep

    lo, hi = 0, 1  # holds(lo) (or lo is 0); holds(hi) is still to check
    while hi <= doc.total and holds(hi):
        lo, hi = hi, (min(hi * 2, doc.total) if hi < doc.total else hi + 1)
    while hi - lo > 1:  # holds(lo); not holds(hi), or hi is past the end
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if holds(mid) else (lo, mid)
    return None if lo == 0 else (lo, checked[lo])
