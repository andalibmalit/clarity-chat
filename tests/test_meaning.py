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
from hypothesis import example, given
from hypothesis import strategies as st

from clarity_chat.commands import Commands
from clarity_chat.documents import (
    Loaded,
    SearchRefused,
    build_id,
    cut_parts,
    library_add,
    meaning_search,
    parts_jsonl,
    reading,
)
from clarity_chat.documents.search import similarity
from clarity_chat.profiles import SEARCH_METHOD, ConfigError, load_defaults, load_profile
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_class import meaning_session  # noqa: F401 (a fixture)

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


def rule_1(title: str | None, text: str) -> str:
    """Decision 7's input rule, version 1, written out: the title, a blank line, the text."""
    return f"{title}\n\n{text}" if title else text


def expected_file(session: Session, fake: FakeEmbedder) -> tuple[str, bytes]:
    """The vectors file's name and bytes for `notes` and `fake`'s setup, from the plan's
    decision 9: KEY is the sha256 of the build ID and the setup; one little-endian float32
    vector per part, in part order, no header; each part's input by decision 7."""
    manifest = json.loads((notes_folder(session) / "manifest.json").read_text("utf-8"))
    setup = fake.start()["setup"]
    key = hashlib.sha256(f"{manifest['build_id']}\n{setup}".encode()).hexdigest()
    parts = [json.loads(line) for line in (notes_folder(session) / "parts.jsonl").open("rb")]
    other = FakeEmbedder(fake.device)  # its count isn't the session's
    data = b"".join(other.vector(other.ids(rule_1(p["title"], p["text"]))) for p in parts)
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
    # Each part's input exactly as decision 7 says, after the message (the fix review's S2):
    # the fake's vectors can't tell one newline from a blank line.
    parts = [json.loads(line) for line in (notes_folder(session) / "parts.jsonl").open("rb")]
    assert fake.texts == [ASKED, *(rule_1(p["title"], p["text"]) for p in parts)]


def test_a_second_search_reads_the_file_without_building(meaning):
    session, c, fake = meaning
    say(c, ASKED)
    assert fake.made == 3 + 1  # three parts, then the message
    found = say(c, "Why does rock wear away?")
    assert fake.made == 3 + 1 + 1  # the message only
    name, data = expected_file(session, fake)
    assert found["documents"][0]["vectors_sha256"] == hashlib.sha256(data).hexdigest()


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
    elif how == "length 1.002":
        path.write_bytes(
            struct.pack("<1024f", *(x * 1.002 for x in floats(data[:4096]))) + data[4096:]
        )
    elif how == "length 0.9":
        path.write_bytes(
            struct.pack("<1024f", *(x * 0.9 for x in floats(data[:4096]))) + data[4096:]
        )
    else:  # a link to a good copy
        copy = path.with_name("elsewhere.f32")
        copy.write_bytes(data)
        path.unlink()
        path.symlink_to(copy)


@pytest.mark.parametrize(
    "how", ["truncated", "padded", "nan", "length 0.9", "length 1.002", "a link"]
)
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
    """Through the command's events (Commands.run sets Session.on_building, B1)."""
    session, c, _ = meaning
    seen: list[tuple[str, int, int]] = []

    def building(event: dict) -> None:
        if event["event"] == "building":
            seen.append((event["name"], event["done"], event["total"]))

    c.run("say", ASKED, building)
    assert seen == [("notes", 0, 3), ("notes", 1, 3), ("notes", 2, 3), ("notes", 3, 3)]
    c.run("say", ASKED, building)
    assert len(seen) == 4  # read, not built: no progress


# --- the scorer and the order (decisions 10 to 12) -----------------------------------------

float32s = st.floats(width=32, allow_nan=False, allow_infinity=False)


@given(st.lists(st.tuples(float32s, float32s), min_size=1, max_size=64))
@example([(1.0, 1.0), (2.0**-53, 1.0), (2.0**-106, 1.0)])  # fsum: 1.0000000000000002; sum: 1.0
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
    found = meaning_search(docs, {"alpha": same, "beta": same}, packed([1.0, 0.0]))
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
    found = meaning_search([doc], vectors, golden_vector(0))  # not unit vectors
    assert [(m["id"], repr(m["score"])) for m in found] == GOLDEN_ORDER


def test_twenty_thousand_parts_are_scored_within_a_loose_bound():
    text = "".join(f"## H{i}\nM.\n\n" for i in range(20_000))
    doc = loaded("big", text)
    assert len(doc.parts) == 20_000
    vectors = {"big": [golden_vector(i % 50) for i in range(20_000)]}
    started = time.monotonic()
    found = meaning_search([doc], vectors, golden_vector(0))
    assert time.monotonic() - started < 60
    assert len(found) == 20_000


# --- /history, /state and /search ------------------------------------------------------------


def test_a_part_exactly_at_the_minimum_is_added_through_a_chat(meaning):
    """The whole-branch review's S1: the boundary on the path a chat takes. The best part's
    exact score, written into the profile as its minimum, still matches."""
    session, c, _ = meaning
    profile(session.root, "all", MEANING.replace("0.2", "-1"))
    c.run("profile", "all", lambda e: None)
    best = c.run("try", ASKED, lambda e: None)["found"]["parts"][0]["score"]
    profile(session.root, "edge", MEANING.replace("0.2", repr(best)))
    c.run("profile", "edge", lambda e: None)
    found = c.run("try", ASKED, lambda e: None)["found"]
    assert found["min_similarity"] == best
    assert found["matched"] >= 1 and found["parts"][0]["score"] == best
    above = math.nextafter(best, 2.0)  # the next float up: the best part is just under it
    profile(session.root, "over", MEANING.replace("0.2", repr(above)))
    c.run("profile", "over", lambda e: None)
    assert c.run("try", ASKED, lambda e: None)["found"]["matched"] == 0


def test_a_conversation_s_method_defaults_to_clarity_s_default(meaning):
    """Could 2: a chat whose profile names no method searches as Clarity does by default.
    Since v0.4.4 B1 the method is a chat's search value, from its bases, not a field."""
    import dataclasses

    from clarity_chat.profiles import SEARCH_METHOD, Setting

    session, _, _ = meaning
    conversation = session.conversation
    assert conversation is not None
    conversation.profile = dataclasses.replace(conversation.profile, search_method=None)
    assert conversation.search_values()["search_method"] == Setting(SEARCH_METHOD, "default")
    assert conversation.search_method == SEARCH_METHOD == "meaning"


def test_search_on_a_chat_that_searches_nothing_doesn_t_warn(meaning):
    """Could 3 (#11): bge-m3 not fetched matters only to a chat that searches a document."""
    session, c, fake = meaning
    fake.is_fetched = False
    assert c.run("search", "none", lambda e: None).get("warnings") == []


def test_history_summarises_a_meaning_found_and_a_words_one_names_its_method(meaning):
    session, c, _ = meaning
    found = say(c, ASKED)
    [turn] = [t for t in c.run("history", "", lambda e: None)["turns"] if t.get("found")]
    summary = turn["found"]
    assert set(summary) == {
        "method", "documents", "matched", "tokens", "budget", "considered", "min_similarity",
        "parts",
    }  # fmt: skip
    assert summary["method"] == "meaning" and summary["documents"] == ["notes"]
    assert set(summary["parts"][0]) == {"id", "title", "page", "rank", "score"}  # could 5
    assert summary["min_similarity"] == 0.2 == found["min_similarity"]
    assert [p["score"] for p in summary["parts"]] == [p["score"] for p in found["parts"]]
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
    assert resumed_method(session, log, without_the_key) == SEARCH_METHOD


# --- Checkpoint 1's fixes (notes/research/reviews/v041-checkpoint1.md) ---------------------


def test_a_link_named_like_the_part_file_is_never_written_through(meaning, tmp_path):
    """M1: a document folder copied from elsewhere can hold a link named
    vectors-KEY.f32.part; the build replaces it, never writing into its target."""
    if os.name == "nt":
        pytest.skip("symlinks need privileges on Windows")
    session, c, fake = meaning
    name, data = expected_file(session, fake)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"precious")
    (notes_folder(session) / f"{name}.part").symlink_to(outside)
    say(c, ASKED)
    assert outside.read_bytes() == b"precious"
    assert (notes_folder(session) / name).read_bytes() == data


def test_a_vectors_write_failure_is_said_as_it_is(meaning, monkeypatch):
    """S1: a full disk (or a read-only folder) says the write failed, not that this Python
    has no FTS5."""
    from clarity_chat.documents import vectors

    session, c, _ = meaning

    def full(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vectors, "write", full)
    refused = raised(lambda: c.run("say", ASKED, lambda e: None))
    assert isinstance(refused, ConfigError), repr(refused)
    assert "Can't write notes" in str(refused) and "FTS5" not in str(refused), refused


def test_each_document_logs_its_own_vectors_hash(meaning):
    """S3c: two documents, each found's vectors_sha256 the hash of its own file."""
    session, c, _ = meaning
    reading._last = None
    library_add(session.root, b"## Tides\nHow the moon pulls the sea twice a day.\n", "more.md")
    c.run("search", "notes more", lambda e: None)
    found = say(c, ASKED)
    for doc in found["documents"]:
        folder = session.root / "local" / "documents" / doc["name"]
        [path] = folder.glob("vectors-*.f32")
        assert doc["vectors_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), doc


def wrong_engine_model(found: dict) -> None:
    """A model that isn't a string, with the setup recomputed so only the type is wrong."""
    from clarity_chat.documents.embedder import ENGINE_FACTS, setup_of

    engine = found["engine"]
    engine["model"] = ["made up"]
    engine["setup"] = setup_of({k: engine[k] for k in ENGINE_FACTS})


def wrong_engine_key(found: dict) -> None:
    found["engine"]["extra"] = "made up"  # setup is the hash of ENGINE_FACTS: unchanged


@pytest.mark.parametrize(
    "path, value",
    [
        (("min_similarity",), 2.0),
        (("min_similarity_origin",), "made up"),
        (("query", "tokens"), "3"),
        (("query", "sha256"), 7),
        (("documents", 0, "vectors_sha256"), 7),
        (("engine",), wrong_engine_key),
        (("engine",), wrong_engine_model),  # the whole-branch review's S5
        (("parts", 0, "score"), float("inf")),  # the fix review's F2
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),  # a function's name, not its address (xdist)
)
def test_a_meaning_found_field_of_the_wrong_type_is_refused(meaning_session, path, value):  # noqa: F811
    """S3f: as test_search_survivors' word-search sweep, for a meaning found's own fields."""
    from tests.test_class import meaning_log, refused_by_resume_and_export

    def change(row: dict) -> bool:
        node = row["found"]
        if node.get("method") != "meaning":
            return False
        if callable(value):
            value(node)
            return True
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return True

    log = meaning_log(meaning_session)
    refused_by_resume_and_export(meaning_session, log, change)


def test_document_try_refuses_a_meaning_profile_and_starts_no_embedder(meaning, chat_tokenizer):
    """Decision 17 (the fix review's S1): `clarity document try` searches by words only; on a
    meaning profile it's refused before any embedder starts (one that fails if used)."""
    from tests.test_embedder import Untouchable

    session, _, _ = meaning
    fresh = Session(
        session.root,
        backend=FakeBackend([]),
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=Untouchable(),  # ty: ignore[invalid-argument-type]
    )
    refused = raised(lambda: fresh.try_documents(["notes"], "meaning", ASKED))
    assert isinstance(refused, ConfigError), repr(refused)
    assert str(refused) == (
        "clarity document try searches by words only; /try in a chat searches by meaning."
    )


# --- B3: meaning is the default (Q15) --------------------------------------------------------

KEYLESS = '\n[documents]\nsearch = ["notes"]\n'


def test_a_profile_with_no_key_searches_by_meaning_in_a_new_chat(meaning):
    session, c, _ = meaning
    profile(session.root, "keyless", KEYLESS)
    c.run("profile", "keyless", lambda e: None)
    assert session.conversation is not None
    assert session.conversation.search_method == "meaning"
    assert say(c, ASKED)["method"] == "meaning"


def test_a_resumed_v0_3_2_shaped_chat_still_searches_by_words(meaning):
    """A chat whose logged profile names no method and whose last search was by words, as
    every chat before v0.4.1: it keeps searching by words after the flip (D4), and needs no
    embedder."""
    session, c, fake = meaning
    profile(session.root, "keyless", KEYLESS + 'search_method = "words"\n')
    c.run("profile", "keyless", lambda e: None)
    say(c, "leaves light sugar")
    assert session.conversation is not None
    log = session.conversation.log.path

    def v0_3_2(row: dict) -> None:  # its profile file had no search_method then
        if row.get("type") == "conversation":
            toml = row["profile"]["toml"]
            row["profile"]["toml"] = toml.replace('search_method = "words"\n', "")

    made = fake.made
    assert resumed_method(session, log, v0_3_2) == "words"
    assert say(c, "leaves light sugar")["method"] == "words" and fake.made == made


def test_bge_m3_missing_refuses_a_keyless_profile_s_message_with_state_5(meaning):
    session, c, fake = meaning
    profile(session.root, "keyless", KEYLESS)
    c.run("profile", "keyless", lambda e: None)
    fake.is_fetched = False
    refused = raised(lambda: c.run("say", ASKED, lambda e: None))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert refused.problems == [{"state": "no_embedder"}]
