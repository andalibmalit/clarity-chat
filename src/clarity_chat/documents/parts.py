"""Parts, version 1 (v0.3.2 decision 7): a document's `text.md` cut into the pieces search
finds. Headings start parts and title them; paragraphs under one title are packed up to
250 words; a longer paragraph splits at sentence ends, then at word boundaries. Each part
is a range of the text, so the parts rebuild it: what lies between them is whitespace."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from pathlib import PurePath
from typing import Any

MAX_WORDS = 250
TITLE_CHARS = 120
CHUNKER = {
    "name": "clarity-parts",
    "version": 1,
    "max_words": MAX_WORDS,
    "title_chars": TITLE_CHARS,
}
HEADING = re.compile(r"(#{1,6}) (\S.*)")
FENCE = "```"
# A kind `text` file with one of these endings is code: a `# comment` isn't a heading there.
CODE_EXTENSIONS = frozenset(
    "c cc cpp cs css go h hpp java js jsx kt lua php pl py r rb rs scala sh sql swift ts tsx "
    "yaml yml toml".split()
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CLOSING_TAGS = ("</part>", "</document>")


def is_code(name: str) -> bool:
    return PurePath(name).suffix.lower().lstrip(".") in CODE_EXTENSIONS


def blocks(text: str, titles: bool) -> list[tuple[int, int, str | None, bool]]:
    """Paragraphs as (start, end, title, starts with a heading), trimmed of whitespace. A
    blank line ends one; a heading (outside a fenced code block, in a file that has titles)
    starts one and sets the title of those after it. A fenced block is never split."""
    found: list[tuple[int, int, str | None, bool]] = []
    title: str | None = None
    start: int | None = None
    headed = False
    fenced = False
    at = 0
    for line in text.splitlines(keepends=True):
        bare = line.strip()
        heading = None if fenced or not titles else HEADING.fullmatch(line.rstrip("\r\n"))
        if (heading or (not bare and not fenced)) and start is not None:
            found.append((start, at, title, headed))
            start = None
        if heading:
            title = heading.group(2).strip()[:TITLE_CHARS]
        if bare and start is None:
            start, headed = at, heading is not None
        if bare.startswith(FENCE):
            fenced = not fenced
        at += len(line)
    if start is not None:
        found.append((start, at, title, headed))
    return [(a, a + len(text[a:b].rstrip()), t, h) for a, b, t, h in found]


def pieces(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """A paragraph over MAX_WORDS, cut at sentence ends into runs of at most MAX_WORDS
    words, a sentence over it at word boundaries. Ranges, trimmed."""
    sentences, at = [], a
    for match in SENTENCE_END.finditer(text, a, b):
        sentences.append((at, match.start()))
        at = match.end()
    sentences.append((at, b))
    cut: list[tuple[int, int]] = []
    for s, e in sentences:
        spans = [m.span() for m in re.finditer(r"\S+", text[s:e])]
        for i in range(0, len(spans), MAX_WORDS):
            run = spans[i : i + MAX_WORDS]
            cut.append((s + run[0][0], s + run[-1][1]))
    packed: list[tuple[int, int]] = []
    for s, e in cut:
        if packed and count(text[packed[-1][0] : e]) <= MAX_WORDS:
            packed[-1] = (packed[-1][0], e)
        else:
            packed.append((s, e))
    return packed


def count(text: str) -> int:
    return len(text.split())


def cut_parts(text: str, name: str, page_starts: list[int] | None = None) -> list[dict]:
    """`text`'s parts, in order: ordinal (from 1), title, text, its range in `text`, the
    page it starts on (from `page_starts`, a PDF's), its word count and its sha256."""
    runs: list[tuple[int, int, str | None]] = []
    for a, b, title, headed in blocks(text, titles=not is_code(name)):
        if count(text[a:b]) > MAX_WORDS:
            runs += [(s, e, title) for s, e in pieces(text, a, b)]
            continue
        last = runs[-1] if runs else None
        if last and not headed and last[2] == title and count(text[last[0] : b]) <= MAX_WORDS:
            runs[-1] = (last[0], b, title)  # packed under the same title
        else:
            runs.append((a, b, title))
    parts = []
    for ordinal, (a, b, title) in enumerate(runs, 1):
        body = text[a:b]
        page = bisect_right(page_starts, a) if page_starts else None
        parts.append(
            {
                "ordinal": ordinal,
                "title": title,
                "text": body,
                "chars": [a, b],
                "page": page,
                "words": count(body),
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            }
        )
    return parts


def parts_jsonl(parts: list[dict[str, Any]]) -> bytes:
    """`parts.jsonl`'s bytes: one part a line, its keys in the Shapes order, UTF-8."""
    lines = [json.dumps(part, ensure_ascii=False) + "\n" for part in parts]
    return "".join(lines).encode("utf-8")


def build_id(jsonl: bytes) -> str:
    """A build's ID: the sha256 of its parts.jsonl, so it changes iff a part does."""
    return hashlib.sha256(jsonl).hexdigest()


def closing_tag_parts(parts: list[dict[str, Any]]) -> list[int]:
    """The ordinals of parts holding `</part>` or `</document>`, which end Clarity's wrapper
    early: listed, as v0.3.1 lists `</document>`, and sent as they are."""
    return [p["ordinal"] for p in parts if any(tag in p["text"] for tag in CLOSING_TAGS)]
