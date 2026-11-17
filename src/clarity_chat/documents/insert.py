"""The insertion path (v0.3.1 decision 1): every attachment, and later every search result,
ends as a block of text at the start of your message. A plain function of the message, its
attachments and the profile: it reads no file and keeps no state. Counting is passed in as
`Room`, so this package never needs the session or the renderer."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from clarity_chat.documents.child import NO_TEXT, PROTECTED
from clarity_chat.documents.reading import (
    SLOW_READ,
    STOPPED,
    TOO_MUCH_MEMORY,
    TOO_SLOW,
    Document,
    DocumentRefused,
    checked_name,
)
from clarity_chat.documents.search import CONSIDERED, LEFT_OUT_LISTED
from clarity_chat.profiles import Profile

FRAMING = (
    "Reference material: use it to understand the conversation; don't copy its wording or tone."
)
MARKER = "Excerpt: {units} {first} to {last} of {total}"
CLOSING_TAG = "</document>"
PART_CLOSING_TAG = "</part>"

# The sheet's words, in one place: the page, the terminal, the refusals that explain a fit,
# and reference.md use these (decision 20). Each was fact-checked (Lane 3).
SENTENCES = {
    "whole": "It takes up {tokens} tokens in every later prompt of this chat.",
    "excerpt": "Only the {units} you choose are sent, after a line that says they're an excerpt "
    "unless the profile sets no marker.",
    "no_marker": "No excerpt marker (this profile).",
    "free": "{free} tokens free in this chat, after setting aside {max_tokens} for the reply "
    "(your max_tokens).",
    "left": "That leaves {left} tokens for your messages and the replies.",
    # The part's suggested range, said with what it buys (the design walkthrough, F1).
    "reply_share": "Each later reply can use up to {max_tokens} of them (your max_tokens).",
    "filled_in": "Clarity filled in {units} 1 to {last}: the largest start it found that "
    "leaves at least {max_tokens}.",
    "at_most": "At most {units} 1 to {last} fit, leaving {left}.",
    "longer_chat": "For a longer chat, attach fewer {units}.",
    "pdf_pages": "Pages are counted from the file's first page, not the numbers printed on them.",
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
    "it is, and the call is flagged. The model may read them as its own markers, such as the "
    "end of a message.",
    "closing_tag": "Holds </document>, which ends Clarity's wrapper early; it's sent as it is.",
    "framing": "Framing line ({origin}): {framing}",
    "framing_label": "Clarity introduces the document to the model with this line ({origin}):",
    "marker_label": "The excerpt starts with this line, which says it's partial ({origin}):",
    "new_chat_does": "Starts a new chat on this profile with the whole file attached, waiting "
    "for your first message. This chat stays in Chats.",
    "not_kept": "Not kept: {names}, waiting for your next message. An attachment is logged "
    "only with the message it's sent with.",
    "more_lines": "… and {count} more lines, ending with:",
    "more_characters": "… {count} more characters on this line",
    "choose": "Choose how to attach it.",
    "stopped": STOPPED,
    # Your sent message's line: the cost when attached, so it doesn't read as a contradiction
    # of Details' tokens in the prompt (the fact-check's rows 50-52).
    "attached_line": "Attached: {name} · {how} · {tokens} tokens when attached",
    "chip_tokens": "{tokens} tokens when attached",
    "keeps": "Keeps: {names}. An edit can't add attachments.",  # N73: Edit's × removes one
    "keeps_none": "Keeps no attachments.",  # N74
    "keeps_drop": "Remove {name} from the edited message",  # N75: a chip's ×
    "keeps_group": "Attachments this edit keeps",  # N76
    "stays_waiting": "{names} stays waiting for your next message.",
    "stay_waiting": "{names} stay waiting for your next message.",
    "highlight_added": "Highlighted: the framing line, documents and parts Clarity put in your "
    "messages.",  # T20
    # Details' Attachments row, under the counts (the walkthrough's F3, as the lead ruled).
    "attachments_caption": "Your message's line counts each attachment as it cost when it was "
    "attached: its block and the blank line after it (not highlighted here). The first "
    "attachment in a message also counts the framing line then in use and its blank line. A "
    "later change to the framing line, or an edit that leaves an attachment out, doesn't "
    "recount them; this table shows what each span holds.",
    # The stop (the final fact-check's FIX 1, F8, ruling 9 and row 53b).
    "too_long": "It's {over} tokens too long.",
    "stop_advice": "Shorten your message, start a new chat, or lower max_tokens.",
    "remove_waiting": "Shorten your message, remove an attachment and attach a shorter part, "
    "start a new chat, or lower max_tokens.",
    "slow_read": SLOW_READ,
    # An empty /say, on the page and in the terminal (the whole-branch review's could 17).
    "empty_message": "Type a message to send.",
    "empty_with_attachments": "Type a message to send with {names}.",
    "stopped_counting": "Stopped counting {name}.",
    # A range of a file with no text in it: refused, never sent empty (the records finder, 3).
    "empty_excerpt": "{Units} {first} to {last} of {name} have no text Clarity can read.",
    # A searching chat's numbers: the suggested start leaves the budget free too (v0.3.2's
    # S1, folded in by the board walk's S14; the board's N14, N15).
    "left_search": "That leaves {left} tokens for your messages, the parts search adds, and "
    "the replies.",
    "filled_in_search": "Clarity filled in {units} 1 to {last}: the largest start it found that "
    "leaves at least {max_tokens} for a reply and {budget} for the parts search can add to your "
    "next message ({origin}).",
    "preselected_search": "The largest start found that leaves at least {max_tokens} tokens for "
    "a reply and {budget} for the parts search can add to your next message ({origin})",
    "budget_profile": "the profile's budget",
    "budget_default": "Clarity's default budget",
    # /undo of a message that had attachments (QA 9): they were part of it.
    "undone_attachments": "Its attachments went with it: {names}.",
    # Refusals the reader and its child write: their texts live beside the code that raises
    # them (insert.py imports both), and are listed here so reference.md lists them too.
    "too_slow": TOO_SLOW,
    "too_much_memory": TOO_MUCH_MEMORY,
    "protected": PROTECTED,
    "no_text": NO_TEXT,
    "no_room": "This chat has no room left for a file: its messages and max_tokens already fill "
    "the context. Lower max_tokens or start a new chat.",
    "no_framing": "No framing line ({origin}).",
    "origin_default": "Clarity's default",
    "origin_profile": "this profile",
    "search_not_kept": "Not kept: this chat's search list ({names}). The new chat searches "
    "its profile's: {searched}.",  # the close batch: siblings C1
    "encrypted": "encrypted (it opens without a password)",
    "empty_units": "{count} with no text",
    # Adding a document with "Search it" (v0.3.2 C2; cuts 1 and 2)
    "keeps_copy": "Clarity keeps a copy in local/documents/{name}/: the file, its text, its "
    "parts and a record of them.",
    "searches_next": "This chat searches {name} from your next message.",
    "replaced": "Clarity replaced {name} in your documents with this file.",
    # "Search it" on the attach sheet and the Searching chooser (v0.3.2 C2; N2-N27, N60)
    "search_it": "For each message you send, Clarity adds the parts of this document that share "
    'a word with your message, best match first; common words such as "the" don\'t count. A '
    "message gets at most {considered} parts and {budget} tokens ({origin}), shared by every "
    "document this chat searches. Added parts stay in the chat with that message.",
    "replace_note": "{name} is already in your documents. Adding this file replaces its copy, "
    "text and parts in local/documents/{name}/. Anything else you put in that folder stays.",
    "searching_some": "Searching: {documents}",
    "searching_none": "Searching: no documents",
    "searching_head": "Searched for each message you send",
    "searching_applies": "A change applies from your next message. Regenerate keeps the parts a "
    "message already has.",  # the walkthrough's M2, T1
    "searching_profile": "From this chat's profile, {profile}. A change here applies to this "
    "chat only.",
    "searching_chat": "Changed in this chat.",
    "searching_others": "Your other documents",
    "searching_missing": "Its folder in local/documents/ is gone. To search it again, add the "
    "file again with the paperclip's Search it.",
    "searching_empty": 'No documents yet. Press the paperclip and choose "Search it".',
    "edit_searched": "The new text will be searched again. The parts found before stay with the "
    "old version.",
    "text_mode_search": "Searching needs a chat-mode profile; {profile} is in text mode.",
    # "From documents" above a reply, and Details' Documents section (C3)
    "found_footer": "Found when you sent this message. Regenerating reuses these parts; it "
    "doesn't search again.",  # N54
    "searched_footer": "Searched when you sent this message. Regenerating doesn't search "
    "again.",  # N55
    "new_with_documents": "New chat with these documents",  # N35
    "new_with_documents_does": "Starts a new chat on this profile that searches {names}. This "
    "chat stays in Chats.",  # N36
    "details_where": "{placement} of your message · tokens {first} to {last}",
    "details_framing": "tokens {first} to {last} · {n} tokens, counted in the budget",
    "details_search": "words, titles weighted double (word rule {rule} · common words {common} · "
    "SQLite {sqlite})",
    "details_considered": "{parts} matched; Clarity considers the best {considered} at most.",
    "details_budget": "the budget ({budget}, {origin})",
    "details_caption": "Rank 1 is the best match; a word in a title counts more than one in the "
    "text. Searched: "
    "{builds} (each document’s build). Each part’s text and score are in this message’s log "
    "record.",  # N66
    # What a search found, in "From documents" (N44-N53, N57-N59)
    "found_parts": "{parts} · {tokens} tokens",
    "found_already_more": "{n} more already in the chat",
    "found_nothing_words": "nothing to search for",
    "found_nothing_matched": "nothing matched",
    "found_nothing_new": "nothing new · {n} already in the chat",
    "found_nothing_fit": "nothing added · {n} didn't fit the budget",
    "found_nothing_added": "nothing added",  # Details' header (N61)
    "found_in_title": "in its title: {words}",
    "found_in_text": "in its text: {words}",
    "found_left_budget": "{n} more matched but were left out: Clarity adds parts in order and "
    "stops at the first that doesn't fit the budget of {budget} tokens.",
    "found_left_budget_none": "{n} matched but were left out: Clarity adds parts in order and "
    "stops at the first that doesn't fit the budget of {budget} tokens.",
    "found_unconsidered": "{n} more matched but weren't considered: Clarity considers only the "
    "best {considered} matches.",
    "found_words": "Words searched: {words}.",
    "found_ignored": "Ignored (common words and single letters or digits): {words}.",
    "found_all_common": "Every word in your message is a common word or a single letter or "
    "digit, so nothing was searched and nothing was added.",
    "found_no_words": "Your message has no words to search for, so nothing was added.",
    "found_no_match": "No part of {documents} has any of the words searched, so nothing was "
    'added. Search matches words and close forms of them ("sing" finds "singing"), not '
    "meaning.",
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


def how_words(attachment: Mapping[str, Any]) -> str:
    """ "whole", or an excerpt's range: "pages 1–23 of 131". The chips, your message's line
    and the terminal say an attachment this way (ruling 2: range ends take separators)."""
    if attachment["how"] == "whole":
        return "whole"
    first, last = attachment["range"]
    return f"{attachment['unit']}s {first:,}–{last:,} of {attachment['total']:,}"


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


def with_text(doc: Document, new: Attached, unit: str | None, first: int, last: int) -> Attached:
    """A range you chose, refused if it holds no text (pages that are scanned images, or
    blank lines): it would go in as an empty document, with or without a marker naming its
    range. The search for a start that fits counts such ranges; only a chosen one is
    refused."""
    if not covers_whole(doc, unit, first, last) and not new.text.strip():
        assert unit is not None
        raise DocumentRefused(
            SENTENCES["empty_excerpt"].format(
                Units=unit.capitalize(), first=f"{first:,}", last=f"{last:,}", name=new.name
            )
        )
    return new


def covers_whole(doc: Document, unit: str | None, first: int, last: int) -> bool:
    """Whether a range is the whole document: no range, or one from the first unit to the
    last. piece's text and marker and the record's `how` follow this one rule."""
    return unit is None or (first, last) == (1, doc.total)


def content(
    message: str,
    attached: list[Attached],
    framing: str,
    found: Sequence[Mapping[str, Any]] = (),
    placement: str = "start",
) -> str:
    """The message's content as the model gets it: the block (the framing line, unless
    empty; each attachment; each found part), then your message, a blank line apart; at
    placement "end", your message first, then the block."""
    if placement not in ("start", "end"):  # never sent as start without a word (C4)
        raise ValueError(f"placement must be start or end, not {placement!r}")
    if not attached and not found:
        return message
    block = [text for _, text in parts(attached, framing, found)]
    return "\n\n".join([message, *block] if placement == "end" else [*block, message])


def attribute(value: str) -> str:
    """A value in the block's tags, escaped as v0.3.1 escapes a document's name."""
    return escape(value, quote=True).replace("&#x27;", "'")


def parts(
    attached: list[Attached], framing: str, found: Sequence[Mapping[str, Any]] = ()
) -> list[tuple[dict[str, Any], str]]:
    """What the block holds, each piece with what it is, in order: the framing line, each
    attachment's <document>, each found part's <part> (v0.3.2 decision 1)."""
    block: list[tuple[dict[str, Any], str]] = [({"what": "framing"}, framing)] if framing else []
    for i, a in enumerate(attached):
        marker = f"{a.marker}\n" if a.marker else ""
        closing = "" if a.text.endswith("\n") else "\n"
        text = f'<document name="{attribute(a.name)}">\n{marker}{a.text}{closing}{CLOSING_TAG}'
        block.append(({"what": "attachment", "index": i}, text))
    for i, p in enumerate(found):
        tags = [f'document="{attribute(p["document"])}"']
        if p["title"] is not None:
            tags.append(f'title="{attribute(p["title"])}"')
        if p["page"] is not None:
            tags.append(f'page="{p["page"]}"')
        closing = "" if p["text"].endswith("\n") else "\n"
        text = f"<part {' '.join(tags)}>\n{p['text']}{closing}{PART_CLOSING_TAG}"
        block.append(({"what": "part", "index": i}, text))
    return block


def found_in(fields: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The parts search added to a message, from a turn's fields or its record."""
    return (fields.get("found") or {}).get("parts") or []


def message_content(text: str, fields: Mapping[str, Any]) -> str:
    """A user message's content as the model gets it, from a turn's fields or its record:
    the text alone, or with its block. Every prompt, check and read builds it here. A
    record missing a field raises KeyError, which readers refuse as unreadable."""
    attached, found = attached_in(fields), found_in(fields)
    if not attached and not found:
        return text
    return content(text, attached, fields["framing"], found, fields["placement"])


def attached_in(fields: Mapping[str, Any]) -> list[Attached]:
    """What each attachment puts in the block, from a turn's fields or records."""
    return [Attached(a["name"], a["text"], a["marker"]) for a in fields.get("attachments") or []]


# A turn record's attachment fields, and each attachment's (the plan's Shapes).
TURN_FIELDS = ("attachments", "framing", "framing_origin", "placement")


def _whole_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


# What each field must be, as Clarity writes it: the readers format and index these, so a
# log someone else wrote can't crash a view or the terminal (the records finder, 2).
ATTACHMENT_TYPES: dict[str, Callable[[Any], bool]] = {
    "name": lambda v: isinstance(v, str),
    "kind": lambda v: v in ("text", "html", "pdf"),
    "bytes": _whole_number,
    "sha256": lambda v: isinstance(v, str),
    "extractor": lambda v: isinstance(v, dict),
    "unit": lambda v: v in ("line", "page"),
    "total": _whole_number,
    "how": lambda v: v in ("whole", "excerpt"),
    "range": lambda v: isinstance(v, list) and len(v) == 2 and all(map(_whole_number, v)),
    "marker": lambda v: v is None or isinstance(v, str),
    "text": lambda v: isinstance(v, str),
    "extracted_sha256": lambda v: isinstance(v, str),
    "empty_units": _whole_number,
    "control_tokens": _strings,
    "has_closing_tag": lambda v: isinstance(v, bool),
    "tokens": _whole_number,
}
ATTACHMENT_FIELDS = frozenset(ATTACHMENT_TYPES)
TURN_TYPES: dict[str, Callable[[Any], bool]] = {
    "attachments": lambda v: isinstance(v, list) and all(isinstance(a, dict) for a in v),
    "framing": lambda v: isinstance(v, str),
    "framing_origin": lambda v: v in ("default", "profile"),
    "placement": lambda v: v in ("start", "end"),
}


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _maybe(kind: type) -> Callable[[Any], bool]:
    return lambda v: v is None or (isinstance(v, kind) and not isinstance(v, bool))


def _dicts(types: Mapping[str, Callable[[Any], bool]]) -> Callable[[Any], bool]:
    """A list of records, each with exactly these fields of these types."""

    def check(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(v, dict)
            and set(v) == set(types)
            and all(ok(v[k]) for k, ok in types.items())
            for v in value
        )

    return check


# The turn's `found` (v0.3.2 B2), as the Shapes table gives it: a log someone else wrote is
# read as untrusted, so each field a reader uses is of the type Clarity writes. A part's page
# goes into the block, so it's an int or null before any render.
FOUND_PART_TYPES: dict[str, Callable[[Any], bool]] = {
    "id": lambda v: isinstance(v, str),
    "document": lambda v: isinstance(v, str),
    "ordinal": _whole_number,
    "title": _maybe(str),
    "page": _maybe(int),
    "text": lambda v: isinstance(v, str),
    "sha256": lambda v: isinstance(v, str),
    "rank": _whole_number,
    "score": _number,
    "title_words": _strings,
    "words": _strings,
    "tokens": _whole_number,
}
FOUND_TYPES: dict[str, Callable[[Any], bool]] = {
    "method": lambda v: v == "words",
    "searched_for": lambda v: isinstance(v, str),
    "words": _strings,
    "ignored": _strings,
    "documents": _dicts(
        {k: (lambda v: isinstance(v, str)) for k in ("name", "build_id", "original_sha256")}
    ),
    "engine": lambda v: (  # each value too, before Details shows them (the review's C8)
        isinstance(v, dict)
        and set(v)
        == {"name", "sqlite", "tokenizer", "weights", "word_rule", "common_words", "query"}
        and all(isinstance(v[k], str) for k in ("name", "sqlite", "tokenizer", "query"))
        and _whole_number(v["word_rule"])
        and _whole_number(v["common_words"])
        and isinstance(v["weights"], dict)
        and all(isinstance(k, str) and _number(w) for k, w in v["weights"].items())
    ),
    "considered": _whole_number,
    "budget": _whole_number,
    "budget_origin": lambda v: v in ("default", "profile"),
    "parts": _dicts(FOUND_PART_TYPES),
    "left_out": _dicts(
        {
            "id": lambda v: isinstance(v, str),
            "rank": _whole_number,
            "score": _number,
            "why": lambda v: v in ("budget", "already in the chat", "not considered"),
        }
    ),
    "matched": _whole_number,
    "tokens": _whole_number,
}


def _typed(fields: Mapping[str, Any], types: Mapping[str, Callable[[Any], bool]]) -> None:
    for key, ok in types.items():
        if not ok(fields[key]):
            raise TypeError(f"an attachment's {key} isn't one Clarity writes")


def checked_attachments(record: Mapping[str, Any]) -> dict[str, Any]:
    """A turn record's attachment fields, checked whole: {} for a message without any. A
    missing field raises KeyError, and a value Clarity doesn't write raises TypeError, which
    every reader of a log refuses as unreadable, so nothing read later (a regeneration's
    warnings, the stop's count, a view) meets one."""
    if not record.get("attachments") and "found" not in record:
        return {}
    fields = {key: record[key] for key in TURN_FIELDS}
    _typed(fields, TURN_TYPES)
    if "found" in record:  # v0.3.2: a searched message, with or without attachments
        found = record["found"]
        if not isinstance(found, dict):
            raise TypeError("a turn's found isn't one Clarity writes")
        if missing := set(FOUND_TYPES) - set(found):
            raise KeyError(", ".join(sorted(missing)))
        _typed(found, FOUND_TYPES)
        for part in found["parts"]:  # `already` trusts it (Checkpoint 2's C5)
            text = part["text"].encode("utf-8", "surrogatepass")  # a lone one: no match
            if part["sha256"] != hashlib.sha256(text).hexdigest():
                raise TypeError("a found part's sha256 isn't its text's")
        fields["found"] = found
    for attachment in fields["attachments"]:
        if missing := ATTACHMENT_FIELDS - set(attachment):
            raise KeyError(", ".join(sorted(missing)))
        _typed(attachment, ATTACHMENT_TYPES)
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


def added_ranges(text: str, fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Where the framing line, each attachment and each found part sit in the message's
    content: half-open `chars`, for message_spans' `added`. Empty for a message with
    neither. At placement "end" the block starts after your message and a blank line."""
    attached, found = attached_in(fields), found_in(fields)
    if not attached and not found:
        return []
    ranges, at = [], len(text) + 2 if fields["placement"] == "end" else 0
    for what, piece in parts(attached, fields["framing"], found):
        ranges.append({**what, "chars": [at, at + len(piece)]})
        at += len(piece) + 2  # the blank line after it
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
        # Every shown fit runs after a check that at least one token is free, so a bound of
        # zero or less never decides (the whole-branch review, 11: no clamp needed).
        at_least = -(-written // room.token_bytes) - before
        if at_least > available:
            return Fit(tokens=None, free=available, left=None, fits=False, at_least=at_least)
    cost = room.tokens(text) - before
    return Fit(tokens=cost, free=available, left=available - cost, fits=cost <= available)


def fill(
    room: Room,
    message: str,
    pending: list[Attached],
    framing: str,
    found: Mapping[str, Any],
    budget: int,
    budget_origin: str,
    placement: str = "start",
    already: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The parts a search adds to a message (v0.3.2 decision 9): the best CONSIDERED
    matches, in rank order, each counted exactly as it goes into the block, stopping at the
    first that would take the block past `budget` tokens. A part whose text is already in
    the chat on this branch (`already`, by sha256) isn't added again and costs nothing.
    `tokens` is everything search added, the framing line and wrappers included when search
    is why they're there. At most CONSIDERED + 1 renders."""
    matches = found["matches"]
    without = room.tokens(content(message, pending, framing, [], placement))
    added: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    cost, full = 0, False
    seen = set(already)  # earlier messages', then this one's as they're added (S6)
    for match in matches[:CONSIDERED]:
        why = None
        if match["sha256"] in seen:
            why = "already in the chat"
        elif full:
            why = "budget"
        else:
            now = room.tokens(content(message, pending, framing, [*added, match], placement))
            if now - without > budget:
                full, why = True, "budget"
            else:
                added.append({**match, "tokens": now - without - cost})
                cost = now - without
                seen.add(match["sha256"])
        if why is not None:
            left_out.append(
                {"id": match["id"], "rank": match["rank"], "score": match["score"], "why": why}
            )
    left_out += [
        {"id": m["id"], "rank": m["rank"], "score": m["score"], "why": "not considered"}
        for m in matches[CONSIDERED:]
    ]
    filled = {k: v for k, v in found.items() if k != "matches"}
    return {
        **filled,
        "considered": CONSIDERED,
        "budget": budget,
        "budget_origin": budget_origin,
        "parts": added,
        "left_out": left_out[:LEFT_OUT_LISTED],
        "matched": len(matches),
        "tokens": cost,
    }


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
