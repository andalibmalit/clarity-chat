"""The insertion path (v0.3.1 decision 1): every attachment, and later every search result,
ends as a block of text at the start of your message. A plain function of the message, its
attachments and the profile: it reads no file and keeps no state. Counting is passed in as
`Room`, so this package never needs the session or the renderer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

from clarity_chat.documents.reading import Document, DocumentRefused, checked_name
from clarity_chat.profiles import Profile

FRAMING = (
    "Reference material: use it to understand the conversation; don't copy its wording or tone."
)
MARKER = "Excerpt: {units} {first} to {last} of {total}"
CLOSING_TAG = "</document>"

# The sheet's words, in one place: the page, the terminal, the refusals that explain a fit,
# and reference.md use these (decision 20). Each was fact-checked (Lane 3).
SENTENCES = {
    "whole": "It takes up {tokens} tokens of the context for the rest of the chat.",
    "excerpt": "The pages or lines you choose go in complete, after a line saying they're an "
    "excerpt.",
    "no_marker": "No excerpt marker (this profile).",
    "free": "{free} tokens free in this chat, after setting aside max_tokens for the reply.",
    "left": "That leaves {left} tokens for your message and the rest of the chat.",
    "doesnt_fit": "Doesn't fit: needs {tokens} tokens, {free} free.",
    "start_that_fits": "The start that fits: {units} 1 to {last} ({tokens} tokens).",
    "fits_up_to": "{Units} 1 to {last} fit.",
    "none_fits": "Not even its first {unit} fits.",
    "preselected": "The largest start found that still leaves at least {max_tokens} tokens "
    "(your max_tokens) for your message and the rest of the chat.",
    "no_start": "Not even its first {unit} leaves at least {max_tokens} tokens (your "
    "max_tokens) for your message and the rest of the chat.",
    "chosen": "{Units} {first} to {last}:",
    "new_chat": "The whole fits a new chat on this profile, with {free} tokens free there.",
    "already": "This file is already in this chat.",
    "control_tokens": "Holds text the tokenizer reads as control tokens ({tokens}); it's sent as "
    "it is, and the call is flagged.",
    "closing_tag": "Holds </document>, which ends Clarity's wrapper early; it's sent as it is.",
    "framing": "Framing line ({origin}): {framing}",
    "framing_label": "Framing line ({origin})",
    "marker_label": "Excerpt marker ({origin})",
    "new_chat_does": "Starts a new chat on this profile with the whole file attached, waiting "
    "for your first message. This chat stays in Chats.",
    "not_kept": "Not kept: {names}, waiting for your next message. An attachment is logged "
    "only with the message it's sent with.",
    "more_lines": "… and {count} more lines, ending with:",
    "more_characters": "… {count} more characters on this line",
    "choose": "Choose how to attach it.",
    "no_room": "This chat has no room left for a file: its messages and max_tokens already fill "
    "the context. Lower max_tokens or start a new chat.",
    "no_framing": "No framing line ({origin}).",
    "origin_default": "Clarity's default",
    "origin_profile": "this profile",
    "encrypted": "encrypted (it opens without a password)",
    "empty_units": "{count} with no text",
}


def more_than(tokens: int | None, at_least: int | None) -> str:
    """A cost as the sheet, the terminal and the refusals say it: exact, or "more than N"
    when only its bound was taken (it surely doesn't fit)."""
    if tokens is not None:
        return f"{tokens:,}"
    assert at_least is not None
    return f"more than {at_least - 1:,}"


def counted(n: int, unit: str) -> str:
    """1 line, 3 lines, 1,200 pages."""
    return f"{n:,} {unit}" + ("" if n == 1 else "s")


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
    (session.room_left), and max_tokens. For the bound: the prompt's text in UTF-8 bytes
    with an empty next message (`frame_bytes`), and the tokenizer's longest token in bytes
    (`token_bytes`)."""

    tokens: Callable[[str], int]
    room_left: Callable[[int], int]
    max_tokens: int
    frame_bytes: int
    token_bytes: int | None  # None: the bound's premise doesn't hold, so count exactly


@dataclass(frozen=True)
class Fit:
    tokens: int | None  # what attaching it costs: the prompt's tokens with it, minus without
    free: int  # the free context now, with what's already waiting
    left: int | None  # free minus tokens: what your message and the rest of the chat can use
    fits: bool
    # Not counted, because it surely doesn't fit: the least it can cost (tokens is None).
    at_least: int | None = None


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
            f"{doc.name} has {counted(doc.total, doc.unit)}: give a range from 1 to {doc.total:,}, "
            "the first no larger than the last."
        )
    if covers_whole(doc, unit, first, last):
        return Attached(doc.name, doc.text, None)
    template = MARKER if profile.excerpt_marker is None else profile.excerpt_marker
    marker = template.format(
        units=unit, first=f"{first:,}", last=f"{last:,}", total=f"{doc.total:,}"
    )
    return Attached(doc.name, doc.text_of(first, last), marker or None)


def covers_whole(doc: Document, unit: str | None, first: int, last: int) -> bool:
    """Whether a range is the whole document: no range, or one from the first unit to the
    last. piece's text and marker and the record's `how` follow this one rule."""
    return unit is None or (first, last) == (1, doc.total)


def content(message: str, attached: list[Attached], framing: str) -> str:
    """The message's content as the model gets it: the framing line (unless empty), each
    attachment's block in order, then your message, a blank line apart."""
    if not attached:
        return message
    return "\n\n".join([*(text for _, text in parts(attached, framing)), message])


def parts(attached: list[Attached], framing: str) -> list[tuple[dict[str, Any], str]]:
    """What goes before your message, each part with what it is, in order."""
    found: list[tuple[dict[str, Any], str]] = [({"what": "framing"}, framing)] if framing else []
    for i, a in enumerate(attached):
        marker = f"{a.marker}\n" if a.marker else ""
        closing = "" if a.text.endswith("\n") else "\n"
        name = escape(a.name, quote=True).replace("&#x27;", "'")
        block = f'<document name="{name}">\n{marker}{a.text}{closing}{CLOSING_TAG}'
        found.append(({"what": "attachment", "index": i}, block))
    return found


def message_content(text: str, fields: Mapping[str, Any]) -> str:
    """A user message's content as the model gets it, from a turn's fields or its record:
    the text alone, or with its attachments' block. Every prompt, check and read builds it
    here. A record missing a field raises KeyError, which readers refuse as unreadable."""
    attached = attached_in(fields)
    return content(text, attached, fields["framing"]) if attached else text


def attached_in(fields: Mapping[str, Any]) -> list[Attached]:
    """What each attachment puts in the block, from a turn's fields or records."""
    return [Attached(a["name"], a["text"], a["marker"]) for a in fields.get("attachments") or []]


# A turn record's attachment fields, and each attachment's (the plan's Shapes).
TURN_FIELDS = ("attachments", "framing", "framing_origin", "placement")
ATTACHMENT_FIELDS = frozenset(
    {
        "name", "kind", "bytes", "sha256", "extractor", "unit", "total", "how", "range",
        "marker", "text", "extracted_sha256", "empty_units", "control_tokens",
        "has_closing_tag", "tokens",
    }
)  # fmt: skip


def checked_attachments(record: Mapping[str, Any]) -> dict[str, Any]:
    """A turn record's attachment fields, checked whole: {} for a message without any. A
    missing field raises KeyError, which every reader of a log refuses as unreadable, so
    nothing read later (a regeneration's warnings, the stop's count) meets one."""
    if not record.get("attachments"):
        return {}
    fields = {key: record[key] for key in TURN_FIELDS}
    for attachment in fields["attachments"]:
        if missing := ATTACHMENT_FIELDS - set(attachment):
            raise KeyError(", ".join(sorted(missing)))
        name = attachment["name"]
        try:  # as an upload's is: a log someone else wrote can't put one on screen
            checked_name(name)
            if "/" in name:  # a "\\" is a legal character in a macOS or Linux name
                raise DocumentRefused("A file name is never a path.")
        except DocumentRefused as e:  # TypeError: readers refuse it as unreadable
            raise TypeError(f"an attachment has a name Clarity wouldn't write. {e}") from None
    return fields


def summaries(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attachment records without their text, numbered from 1: what results and /history
    list (a SUMMARY)."""
    return [
        {**{k: v for k, v in a.items() if k != "text"}, "n": n}
        for n, a in enumerate(attachments, 1)
    ]


def added_ranges(fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Where the framing line and each block sit in the message's content: half-open
    `chars`, for message_spans' `added`. Empty for a message without attachments."""
    attached = attached_in(fields)
    if not attached:
        return []
    ranges, at = [], 0
    for what, text in parts(attached, fields["framing"]):
        ranges.append({**what, "chars": [at, at + len(text)]})
        at += len(text) + 2  # the blank line after it
    return ranges


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
    prompt's tokens without it, if already counted.

    Bounded first (the security review's must 1): the prompt with it holds at least the
    frame's bytes plus the content's (trimmed, as a template may), and no token covers more
    than `token_bytes` of them. When that least cost is more than is free, it doesn't fit,
    and nothing renders the whole: a 64 MB text would take gigabytes and minutes to."""
    if before is None:
        before = room.tokens(content("", pending, framing))
    available = room.room_left(before)
    text = content("", [*pending, new], framing)
    if room.token_bytes:
        written = room.frame_bytes + len(text.strip().encode("utf-8"))
        at_least = max(-(-written // room.token_bytes) - before, 1)  # the wrapper alone costs
        if at_least > available:
            return Fit(tokens=None, free=available, left=None, fits=False, at_least=at_least)
    cost = room.tokens(text) - before
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
        found = checked[k]
        return found.fits and found.left is not None and found.left >= keep

    lo, hi = 0, 1  # holds(lo) (or lo is 0); holds(hi) is still to check
    while hi <= doc.total and holds(hi):
        lo, hi = hi, (min(hi * 2, doc.total) if hi < doc.total else hi + 1)
    while hi - lo > 1:  # holds(lo); not holds(hi), or hi is past the end
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if holds(mid) else (lo, mid)
    return None if lo == 0 else (lo, checked[lo])
