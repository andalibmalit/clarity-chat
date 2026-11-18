"""v0.4.1 A3: the vectors and meaning search (decisions 7 to 13, the Shapes table).

The embedder is FakeEmbedder (tests/fakes.py), whose behaviours are gated live in
tests/test_embed.py; the vectors file, the scorer, the order, the minimum, the record,
resume's method rule and the 8,192 refusal are Clarity's own and tested here. Made-up text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clarity_chat.commands import Commands
from clarity_chat.documents import (
    Loaded,
    SearchRefused,
    build_id,
    cut_parts,
    input_of,
    library_add,
    meaning_search,
    parts_jsonl,
    reading,
    similarity,
)
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE
from tests.fakes import FakeBackend, FakeEmbedder

GLOSSARY = (
    "## Thermoregulation\nHow a body keeps its warmth steady in the cold.\n\n"
    "## Photosynthesis\nHow leaves turn light into sugar.\n\n"
    "## Erosion\nHow water and wind wear rock away.\n"
)
MEANING = '\n[documents]\nsearch = ["notes"]\nsearch_method = "meaning"\nmin_similarity = 0.2\n'
MEANING_KEYS = {
    "method", "searched_for", "documents", "engine", "query", "min_similarity",
    "min_similarity_origin", "considered", "budget", "budget_origin", "parts", "left_out",
    "matched", "tokens",
}  # fmt: skip
PART_KEYS = {
    "id", "document", "ordinal", "title", "page", "text", "sha256", "rank", "score", "tokens",
}  # fmt: skip
ASKED = "How do leaves turn light into sugar?"


def profile(root: Path, name: str, documents: str) -> None:
    text = CHAT_PROFILE + documents
    (root / "profiles" / f"{name}.toml").write_text(text, encoding="utf-8", newline="\n")


def floats(vector: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(vector) // 4}f", vector)


@pytest.fixture
def meaning(fetched_root, chat_tokenizer):
    """A chat on a profile that searches `notes` by meaning, with FakeEmbedder."""
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    fake = FakeEmbedder()
    session = Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    c = Commands(session)
    c.run("profile", "meaning", lambda e: None)
    return session, c, fake


def say(c: Commands, text: str) -> dict[str, Any]:
    """The `found` of the message `text`, as its turn holds it."""
    c.run("say", text, lambda e: None)
    conversation = c.session.conversation
    assert conversation is not None
    assert conversation.head is not None
    reply = conversation.turns[conversation.head]
    found = conversation.turns[reply.parent_id or ""].found
    assert found is not None
    return found


def raised(run) -> BaseException | None:
    """What `run` raised, or None: a red test fails on its assertion, not on another error."""
    try:
        run()
    except Exception as e:
        return e
    return None


def notes_folder(session: Session) -> Path:
    return session.root / "local" / "documents" / "notes"


def vector_files(session: Session) -> list[str]:
    return sorted(p.name for p in notes_folder(session).iterdir() if p.name.startswith("vectors"))


def expected_file(session: Session, fake: FakeEmbedder) -> tuple[str, bytes]:
    """The vectors file's name and bytes for `notes` and `fake`'s setup, from the plan's
    decision 9: KEY is the sha256 of the build ID and the setup; one little-endian float32
    vector per part, in part order, no header; each part's input by decision 7."""
    manifest = json.loads((notes_folder(session) / "manifest.json").read_text("utf-8"))
    setup = fake.start()["setup"]
    key = hashlib.sha256(f"{manifest['build_id']}\n{setup}".encode()).hexdigest()
    parts = [json.loads(line) for line in (notes_folder(session) / "parts.jsonl").open("rb")]
    other = FakeEmbedder(fake.device)  # its count isn't the session's
    data = b"".join(other.vector(other.ids(input_of(p["title"], p["text"]))) for p in parts)
    return f"vectors-{key}.f32", data


# --- the profile keys (decision 13, state 12) ---------------------------------------------


@pytest.mark.parametrize(
    "line, refusal",
    [
        ('search_method = "vectors"', 'search_method must be "meaning" or "words"'),
        ("search_method = 1", 'search_method must be "meaning" or "words"'),
        ("min_similarity = 1.5", "min_similarity must be a number from -1 to 1"),
        ("min_similarity = -1.01", "min_similarity must be a number from -1 to 1"),
        ("min_similarity = true", "min_similarity must be a number from -1 to 1"),
        ('min_similarity = "0.5"', "min_similarity must be a number from -1 to 1"),
        ("min_similarity = nan", "min_similarity must be a number from -1 to 1"),
    ],
)
def test_a_bad_search_key_is_refused_at_load_naming_it(root, line, refusal):
    profile(root, "bad", f"\n[documents]\n{line}\n")
    with pytest.raises(ConfigError, match=refusal):
        load_profile(root, "bad", load_defaults(root))


# --- a meaning chat's found (the Shapes table) ---------------------------------------------


def test_a_meaning_chat_logs_a_meaning_found(meaning):
    session, c, fake = meaning
    found = say(c, ASKED)
    assert found["method"] == "meaning"
    assert set(found) == MEANING_KEYS
    name, data = expected_file(session, fake)
    [doc] = found["documents"]
    assert set(doc) == {"name", "build_id", "original_sha256", "vectors_sha256"}
    assert doc["vectors_sha256"] == hashlib.sha256(data).hexdigest()
    assert found["engine"] == fake.start()
    asked = FakeEmbedder().vector(fake.ids(ASKED))
    assert found["query"] == {
        "tokens": len(fake.ids(ASKED)),
        "sha256": hashlib.sha256(asked).hexdigest(),
    }
    assert (found["min_similarity"], found["min_similarity_origin"]) == (0.2, "profile")
    assert found["searched_for"] == ASKED
    first = found["parts"][0]
    assert first["id"] == "notes:2" and set(first) == PART_KEYS
    part = floats(data)[1024:2048]
    assert first["score"] == math.fsum(a * b for a, b in zip(floats(asked), part, strict=True))
    assert session.conversation is not None
    with session.conversation.log.path.open(encoding="utf-8") as log:
        turns = [r for r in map(json.loads, log) if r.get("type") == "turn" and "found" in r]
    assert turns[-1]["found"] == found


def test_the_default_minimum_is_logged_as_the_default(meaning):
    from clarity_chat.profiles import MIN_SIMILARITY

    session, c, _ = meaning
    profile(
        session.root, "keyless", '\n[documents]\nsearch = ["notes"]\nsearch_method = "meaning"\n'
    )
    c.run("profile", "keyless", lambda e: None)
    found = say(c, ASKED)
    default = (MIN_SIMILARITY, "default")
    assert (found.get("min_similarity"), found.get("min_similarity_origin")) == default


# --- the vectors file (decision 9) ---------------------------------------------------------


def test_the_vectors_file_holds_each_part_s_vector_in_order(meaning):
    session, c, fake = meaning
    say(c, ASKED)
    name, data = expected_file(session, fake)
    assert vector_files(session) == [name]
    assert (notes_folder(session) / name).read_bytes() == data


def test_a_second_search_reads_the_file_without_building(meaning):
    session, c, fake = meaning
    say(c, ASKED)
    assert fake.made == 3 + 1  # three parts, then the message
    say(c, "Why does rock wear away?")
    assert fake.made == 3 + 1 + 1  # the message only


def test_a_changed_setup_fact_builds_again_and_deletes_the_old_file(meaning):
    session, c, fake = meaning
    say(c, ASKED)
    old = vector_files(session)
    fake.device = "CPU"  # another setup: another key
    found = say(c, ASKED)
    new = vector_files(session)
    assert fake.made == 4 + 4 and len(new) == 1 and new != old
    assert found["engine"]["device"] == "CPU"


def damage(path: Path, how: str) -> None:
    data = path.read_bytes()
    if how == "truncated":
        path.write_bytes(data[:-4])
    elif how == "padded":
        path.write_bytes(data + b"\0\0\0\0")
    elif how == "nan":
        path.write_bytes(struct.pack("<f", math.nan) + data[4:])
    elif how == "length 0.9":
        path.write_bytes(
            struct.pack("<1024f", *(x * 0.9 for x in floats(data[:4096]))) + data[4096:]
        )
    else:  # a link to a good copy
        copy = path.with_name("elsewhere.f32")
        copy.write_bytes(data)
        path.unlink()
        path.symlink_to(copy)


@pytest.mark.parametrize("how", ["truncated", "padded", "nan", "length 0.9", "a link"])
def test_a_file_that_doesn_t_read_is_built_again(meaning, how):
    if how == "a link" and os.name == "nt":
        pytest.skip("symlinks need privileges on Windows")
    session, c, fake = meaning
    say(c, ASKED)
    name, data = expected_file(session, fake)
    assert vector_files(session) == [name]
    damage(notes_folder(session) / name, how)
    say(c, ASKED)
    assert fake.made == 4 + 4, "the damaged file was read as vectors"
    path = notes_folder(session) / name
    assert not path.is_symlink() and path.read_bytes() == data


def test_a_build_that_fails_leaves_no_file_and_the_old_one_in_place(meaning):
    session, c, fake = meaning
    say(c, ASKED)
    old = vector_files(session)
    fake.device, fake.fail_at = "CPU", fake.made + 2  # the second part's vector fails
    refused = raised(lambda: c.run("say", ASKED, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert refused.problems[0]["state"] == "embedder"
    assert vector_files(session) == old
    (notes_folder(session) / "vectors-left.f32.part").write_bytes(b"half")
    fake.fail_at = None
    say(c, ASKED)
    assert len(vector_files(session)) == 1 and vector_files(session) != old  # .part gone


def test_the_build_reports_its_progress(meaning):
    session, c, _ = meaning
    seen: list[tuple[str, int, int]] = []
    session.on_building = lambda name, done, total: seen.append((name, done, total))
    say(c, ASKED)
    assert seen == [("notes", 0, 3), ("notes", 1, 3), ("notes", 2, 3), ("notes", 3, 3)]
    say(c, ASKED)
    assert len(seen) == 4  # read, not built: no progress


# --- the scorer and the order (decisions 10 to 12) -----------------------------------------

float32s = st.floats(width=32, allow_nan=False, allow_infinity=False)


@given(st.lists(st.tuples(float32s, float32s), min_size=1, max_size=64))
def test_a_score_is_the_exact_dot_product_correctly_rounded(pairs):
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    exact = sum((Fraction(x) * Fraction(y) for x, y in pairs), Fraction(0))
    assert similarity(a, b) == float(exact)


def loaded(name: str, text: str) -> Loaded:
    parts = cut_parts(text, "made-up.md")
    return Loaded(name, build_id(parts_jsonl(parts)), f"sha-{name}", parts)


def packed(values: list[float]) -> memoryview[float]:
    return memoryview(struct.pack(f"<{len(values)}f", *values)).cast("f")


def test_equal_scores_order_by_document_then_ordinal():
    text = "## A\nOne.\n\n## B\nTwo.\n"
    docs = [loaded("beta", text), loaded("alpha", text)]
    same = [packed([1.0, 0.0]), packed([1.0, 0.0])]
    found = meaning_search(docs, {"alpha": same, "beta": same}, packed([1.0, 0.0]), -1.0)
    assert [m["id"] for m in found] == ["alpha:1", "alpha:2", "beta:1", "beta:2"]
    assert [m["rank"] for m in found] == [1, 2, 3, 4]


def golden_vector(k: int) -> memoryview[float]:
    return packed([(((i * (k + 7)) % 97) - 48) / 97 for i in range(1024)])


# Each score computed with Fraction from the same float32 values, then rounded once
# (2026-11-18): the scorer gives the same bits on every 64-bit system (decision 10).
GOLDEN_ORDER = [
    ("golden:1", "9.758741444276973"),
    ("golden:5", "0.14443628429156447"),
    ("golden:4", "-0.41332752181676125"),
    ("golden:2", "-2.961844699199345"),
    ("golden:3", "-5.878307792315241"),
]


def test_the_golden_order_is_the_same_on_every_system():
    doc = loaded("golden", "".join(f"## H{k}\nMade-up {k}.\n\n" for k in range(1, 6)))
    vectors = {"golden": [golden_vector(k) for k in range(1, 6)]}
    found = meaning_search([doc], vectors, golden_vector(0), -100.0)  # not unit vectors
    assert [(m["id"], repr(m["score"])) for m in found] == GOLDEN_ORDER


def test_a_part_at_the_minimum_matches_and_one_just_under_doesn_t():
    under = struct.unpack("<f", struct.pack("<I", 0x3EFFFFFF))[0]  # the float32 below 0.5
    assert under < 0.5
    doc = loaded("edge", "## At\nOne.\n\n## Under\nTwo.\n")
    vectors = {"edge": [packed([0.5, 0.0]), packed([under, 0.0])]}
    found = meaning_search([doc], vectors, packed([1.0, 0.0]), 0.5)
    assert [(m["id"], m["score"]) for m in found] == [("edge:1", 0.5)]


def test_twenty_thousand_parts_are_scored_within_a_loose_bound():
    text = "".join(f"## H{i}\nM.\n\n" for i in range(20_000))
    doc = loaded("big", text)
    assert len(doc.parts) == 20_000
    vectors = {"big": [golden_vector(i % 50) for i in range(20_000)]}
    started = time.monotonic()
    found = meaning_search([doc], vectors, golden_vector(0), -100.0)
    assert time.monotonic() - started < 60
    assert len(found) == 20_000


# --- /history, /state and /search ------------------------------------------------------------


def test_history_summarises_a_meaning_found_and_a_words_one_names_its_method(meaning):
    session, c, _ = meaning
    say(c, ASKED)
    [turn] = [t for t in c.run("history", "", lambda e: None)["turns"] if t.get("found")]
    summary = turn["found"]
    assert set(summary) == {
        "method", "documents", "matched", "tokens", "budget", "considered", "min_similarity",
        "parts",
    }  # fmt: skip
    assert summary["method"] == "meaning" and summary["documents"] == ["notes"]
    assert set(summary["parts"][0]) == {"id", "title", "rank", "score"}
    c.run("profile", "chat", lambda e: None)
    c.run("search", "notes", lambda e: None)
    c.run("say", "leaves light sugar", lambda e: None)
    [turn] = [t for t in c.run("history", "", lambda e: None)["turns"] if t.get("found")]
    assert turn["found"]["method"] == "words" and "words" in turn["found"]


def test_state_and_search_name_the_method(meaning):
    session, c, _ = meaning
    assert c.run("state", "", lambda e: None).get("search_method") == "meaning"
    assert c.run("search", "", lambda e: None).get("method") == "meaning"
    c.run("profile", "chat", lambda e: None)
    assert c.run("state", "", lambda e: None).get("search_method") == "words"
    assert c.run("search", "", lambda e: None).get("method") == "words"


# --- 8,192 tokens (decision 8) --------------------------------------------------------------


def test_a_message_past_8192_tokens_is_refused_and_nothing_is_sent(meaning):
    session, c, _ = meaning
    calls = len(session.backend.calls)
    refused = raised(lambda: c.run("say", "word " * 8191, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert refused.problems == [{"state": "too_long", "tokens": 8193}]
    assert len(session.backend.calls) == calls


def test_a_part_past_8192_tokens_refuses_the_search_naming_it(meaning):
    session, c, _ = meaning
    reading._last = None
    # One unbroken run is one part of any length: the chunker cuts at whitespace only.
    library_add(session.root, ("## Long\n" + "a." * 8200 + "\n").encode(), "long.md")
    c.run("search", "long", lambda e: None)
    refused = raised(lambda: c.run("say", ASKED, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    [problem] = refused.problems
    assert problem["state"] == "too_long" and problem["document"] == "long"
    assert problem["ordinal"] == 1 and problem["tokens"] > 8192


# --- resume keeps a chat's method (D4) ------------------------------------------------------


def resumed_method(session: Session, log: Path, edit) -> str:
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        edit(row)
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    session.new_conversation("chat")
    session.resume(log)
    assert session.conversation is not None
    return session.conversation.search_method


def without_the_key(row: dict) -> None:
    if row.get("type") == "conversation":
        toml = row["profile"]["toml"]
        row["profile"]["toml"] = toml.replace('search_method = "meaning"\n', "")


def test_a_resumed_chat_keeps_the_method_of_its_last_search(meaning):
    """The logged profile doesn't name a method: the last searched message's decides."""
    session, c, _ = meaning
    say(c, ASKED)
    assert session.conversation is not None
    log = session.conversation.log.path
    assert resumed_method(session, log, without_the_key) == "meaning"


def test_the_logged_profile_s_method_wins_over_the_last_search(meaning):
    session, c, _ = meaning
    say(c, ASKED)
    assert session.conversation is not None
    log = session.conversation.log.path

    def words(row: dict) -> None:
        if row.get("type") == "conversation":
            row["profile"]["toml"] = row["profile"]["toml"].replace('"meaning"', '"words"')

    assert resumed_method(session, log, words) == "words"


def test_a_resumed_chat_with_no_search_yet_takes_the_profile_s_method(meaning):
    session, c, _ = meaning
    c.run("search", "none", lambda e: None)
    c.run("say", "Hello there", lambda e: None)
    assert session.conversation is not None
    log = session.conversation.log.path
    assert resumed_method(session, log, without_the_key) == "words"  # the default today
