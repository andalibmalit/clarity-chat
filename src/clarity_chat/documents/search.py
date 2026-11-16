"""Word search (v0.3.2 decision 8): the parts of the chat's documents that share your
message's words, ranked by SQLite's FTS5. An in-memory table, built when needed and kept by
the documents' build IDs, never stored. Standard library only."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarity_chat.documents.library import Stale, parts_of, read_manifest, state
from clarity_chat.documents.reading import DocumentRefused

WORD_RULE = 1
COMMON_WORDS_VERSION = 1
TOKENIZER = "porter unicode61"
WEIGHTS = {"title": 2, "text": 1}
CONSIDERED = 8  # the best matches the budget fill considers (decision 9)
LEFT_OUT_LISTED = 50
# Version 1 of the common English words a message's search drops (Q5): words that carry
# no subject. Changing the list is a new version, logged on every turn.
COMMON_WORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing done down
    during each either else ever every few for from further get gets got had has have
    having he hello her here hers herself hi him himself his how i if in into is it its
    itself just let lets like many may me might more most much must my myself no nor not
    now of off ok okay on once only or other our ours ourselves out over own please really
    same say says shall she should so some such tell than thank thanks that the their
    theirs them themselves then there these they this those through to too under until up
    us very was we well were what when where which while who whom whose why will with
    would yes yet you your yours yourself yourselves
    aren couldn didn doesn don hadn hasn haven isn ll mightn mustn needn re shan shouldn ve
    wasn weren won wouldn
    """.split()  # the last two lines: what a contraction leaves (don't: don, t)
)
NO_FTS5 = (
    "Word search needs SQLite's FTS5, which the Python running Clarity doesn't have. Use a "
    "Python whose SQLite includes it, such as the one uv installs."
)
RUN = re.compile(r"[^\W_]+")  # Unicode letters and digits: `_` splits, as FTS5's does


def words_of(message: str) -> tuple[list[str], list[str]]:
    """Word rule version 1: the message split into runs of letters and digits, lowercased;
    one-character words and the common words dropped, and repeats. Returns the words kept
    and the words dropped, each in order, each once."""
    kept: list[str] = []
    dropped: list[str] = []
    for word in RUN.findall(message.lower()):
        if word in kept or word in dropped:
            continue
        (dropped if len(word) == 1 or word in COMMON_WORDS else kept).append(word)
    return kept, dropped


def query_of(words: list[str]) -> str:
    """Each word quoted (a quote doubled inside), joined with OR: no text from the message
    reaches FTS5 unquoted, so punctuation can't raise a syntax error."""
    return " OR ".join('"' + w.replace('"', '""') + '"' for w in words)


@dataclass(frozen=True)
class Loaded:
    """A searched document as built: its name, build ID, original's sha256 and parts."""

    name: str
    build_id: str
    original_sha256: str
    parts: list[dict[str, Any]]


def load(root: Path, name: str) -> Loaded:
    """A document ready to search: refused as `changed` or `missing` (Stale) when its files
    don't match its manifest, and its parts read and checked against its build ID."""
    manifest = read_manifest(root, name)  # missing, or a refusal that says why
    now = state(root, name, manifest)
    if now != "ready":
        raise Stale(name, now)
    parts, build = parts_of(root, name, manifest)  # one read: its build is the one searched
    return Loaded(name, build, manifest["original"]["sha256"], parts)


@dataclass
class Index:
    """One FTS5 table over a set of documents, in (name, ordinal) order."""

    connection: sqlite3.Connection
    parts: dict[int, tuple[Loaded, dict[str, Any]]] = field(default_factory=dict)


_indexes: dict[tuple[tuple[str, str], ...], Index] = {}


def index(docs: list[Loaded]) -> Index:
    """The table for these documents, by their build IDs: built once, kept in memory (the
    last few), so the build a turn records is the one searched."""
    key = tuple(sorted((d.name, d.build_id) for d in docs))
    if key in _indexes:
        return _indexes[key]
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        connection.execute(
            f"CREATE VIRTUAL TABLE parts USING fts5(title, text, tokenize='{TOKENIZER}')"
        )
    except sqlite3.OperationalError as e:
        raise DocumentRefused(NO_FTS5) from e
    built = Index(connection)
    rowid = 0
    for doc in sorted(docs, key=lambda d: d.name):
        for part in doc.parts:
            rowid += 1
            built.parts[rowid] = (doc, part)
            connection.execute(
                "INSERT INTO parts (rowid, title, text) VALUES (?, ?, ?)",
                (rowid, part["title"] or "", part["text"]),
            )
    if len(_indexes) >= 4:
        _indexes.pop(next(iter(_indexes))).connection.close()
    _indexes[key] = built
    return built


def matched_in(built: Index, column: str, word: str, among: list[int]) -> set[int]:
    """Which of the rows `among` hold `word` in `column`, as FTS5 matches it (stemmed): the
    reasons come from FTS5 itself, never from a second stemmer."""
    if not among:
        return set()
    marks = ", ".join("?" * len(among))
    rows = built.connection.execute(
        f"SELECT rowid FROM parts WHERE parts MATCH ? AND rowid IN ({marks})",
        (f"{{{column}}} : {query_of([word])}", *among),
    )
    return {row[0] for row in rows}


def search(docs: list[Loaded], message: str) -> dict[str, Any]:
    """Every part sharing one of the message's words, best first: bm25 (titles count
    double) rounded to six places, then document name, then ordinal, a total order. Each
    match says which of the words FTS5 matched in its title and its text."""
    words, ignored = words_of(message)
    found: dict[str, Any] = {
        "method": "words",
        "searched_for": message,
        "scan_depth": 1,
        "words": words,
        "ignored": ignored,
        "documents": [
            {"name": d.name, "build_id": d.build_id, "original_sha256": d.original_sha256}
            for d in docs
        ],
        "engine": {
            "name": "sqlite-fts5",
            "sqlite": sqlite3.sqlite_version,
            "tokenizer": TOKENIZER,
            "weights": WEIGHTS,
            "word_rule": WORD_RULE,
            "common_words": COMMON_WORDS_VERSION,
            "query": query_of(words),
        },
        "matches": [],
    }
    if not words or not docs:
        return found
    built = index(docs)
    rows = built.connection.execute(
        "SELECT rowid, round(bm25(parts, ?, ?), 6) AS score FROM parts WHERE parts MATCH ?",
        (WEIGHTS["title"], WEIGHTS["text"], query_of(words)),
    ).fetchall()
    order = sorted(
        rows, key=lambda r: (r[1], built.parts[r[0]][0].name, built.parts[r[0]][1]["ordinal"])
    )
    # Reasons only for the matches a fill considers (Checkpoint 1's C2): two queries a word,
    # over those rows only, so a long message on a large library stays quick.
    considered = [rowid for rowid, _ in order[:CONSIDERED]]
    title_rows = {w: matched_in(built, "title", w, considered) for w in words}
    text_rows = {w: matched_in(built, "text", w, considered) for w in words}
    for rank, (rowid, score) in enumerate(order, 1):
        doc, part = built.parts[rowid]
        found["matches"].append(
            {
                "id": f"{doc.name}:{part['ordinal']}",
                "document": doc.name,
                "ordinal": part["ordinal"],
                "title": part["title"],
                "page": part["page"],
                "text": part["text"],
                "sha256": part["sha256"],
                "rank": rank,
                "score": score,
                "title_words": [w for w in words if rowid in title_rows[w]],
                "words": [w for w in words if rowid in text_rows[w]],
            }
        )
    return found
