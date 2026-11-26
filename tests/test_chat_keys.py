"""v0.4.4 B1: the search settings per chat, `search_method`, `min_similarity` and `budget`
(decisions 1 to 4 and 16; states 7 to 15 and 19).

`/set` takes them beside the sampling settings. They're checked as the profile checks them,
kept in the `settings` record's `chat_settings`, never put in a call's `settings`, and
apply from the next message, whose `found` logs the value with origin `chat`. Each has a
base, which `/set NAME` goes back to: the profile's value in a new chat, the logged one in
a resumed chat. Made-up text only; FakeEmbedder stands in for bge-m3 (gated live in
tests/test_embed.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clarity_chat.commands import Commands
from clarity_chat.documents import SENTENCES, library_add, reading
from clarity_chat.export import export_log
from clarity_chat.logs import read_records
from clarity_chat.profiles import BUDGET, ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_embedder import Untouchable
from tests.test_meaning import GLOSSARY, MEANING, profile, raised, say

CHAT_KEYS = {"search_method", "min_similarity", "budget"}
WORDS = '\n[documents]\nsearch = ["notes"]\nsearch_method = "words"\n'
PLAIN = '\n[documents]\nsearch = ["notes"]\n'  # no method: meaning, Clarity's default
ASKED = "How do leaves turn light into sugar?"


def quiet(event: dict) -> None:
    pass


def attempt(c: Commands, line: str) -> tuple[dict[str, Any] | None, BaseException | None]:
    """A command's result, or what it raised: a red test fails on its own assertion."""
    name, _, args = line.partition(" ")
    out: list[dict[str, Any]] = []
    refused = raised(lambda: out.append(c.run(name, args, quiet)))
    return (out[0] if out else None), refused


def ran(c: Commands, line: str) -> dict[str, Any]:
    result, refused = attempt(c, line)
    assert refused is None, f"{line}: {refused}"
    assert result is not None
    return result


def chat_on(root: Path, tokenizer, documents: str, fake: FakeEmbedder | None = None):
    """A chat on a profile with `documents` as its [documents] table, searching `notes`."""
    reading._last = None
    profile(root, "searching", documents)
    library_add(root, GLOSSARY.encode(), "notes.md")
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=tokenizer)
    fake = fake or FakeEmbedder()
    session = Session(
        root,
        backend=backend,
        tokenizer_loader=lambda r, v: tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    c = Commands(session)
    ran(c, "profile searching")
    return session, c, fake


@pytest.fixture
def words(fetched_root, chat_tokenizer):
    return chat_on(fetched_root, chat_tokenizer, WORDS)


@pytest.fixture
def meaning(fetched_root, chat_tokenizer):
    return chat_on(fetched_root, chat_tokenizer, MEANING)


def calls_of(session: Session) -> list[dict[str, Any]]:
    assert session.conversation is not None
    return [r for r in read_records(session.conversation.log.path) if r["type"] == "call"]


def documents_of(c: Commands) -> dict[str, Any] | None:
    """/settings' `documents`: each chat key's value, origin and base."""
    return ran(c, "settings").get("documents")


# --- states 7 and 15: set, logged, applied from the next message, gone in a new chat ------


def test_a_search_key_set_is_logged_kept_out_of_the_call_and_applies_next(words):
    session, c, _ = words
    result, refused = attempt(c, "set budget 800")
    assert refused is None, refused
    assert result is not None and result["setting"] == {"value": 800, "origin": "chat"}
    assert result["record"]["chat_settings"] == {"budget": 800}
    found = say(c, ASKED)
    assert (found["budget"], found["budget_origin"]) == (800, "chat")
    (call,) = calls_of(session)
    assert not CHAT_KEYS & set(call["settings"]), "a chat key went into a call's settings"


def test_a_new_chat_starts_from_the_profile_s_values(words):
    session, c, _ = words
    assert attempt(c, "set budget 800")[1] is None
    ran(c, "new")
    assert session.conversation is not None and session.conversation.chat_settings == {}
    assert found_budget(c) == (BUDGET, "default")


def found_budget(c: Commands) -> tuple[int, str]:
    found = say(c, ASKED)
    return found["budget"], found["budget_origin"]


# --- state 8: a bad value, in the profile's sentence ---------------------------------------


@pytest.mark.parametrize(
    "line, refusal",
    [
        ("budget 0", "/set: budget must be a positive whole number of tokens, got 0."),
        ("budget 1.5", "/set: budget must be a positive whole number of tokens, got 1.5."),
        ("min_similarity 2", "/set: min_similarity must be a number from -1 to 1, got 2."),
        ("search_method fast", '/set: search_method must be "meaning" or "words".'),
    ],
)
def test_a_bad_value_is_refused_as_the_profile_refuses_it(words, line, refusal):
    session, c, _ = words
    _, refused = attempt(c, f"set {line}")
    assert isinstance(refused, ConfigError) and str(refused) == refusal, refused
    assert session.conversation is not None and session.conversation.chat_settings == {}


# --- state 9: a text-mode chat --------------------------------------------------------------


def test_a_text_mode_chat_refuses_a_search_key(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" R."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    ran(c, "profile text")
    _, refused = attempt(c, "set budget 800")
    assert str(refused) == SENTENCES["text_mode_search"].format(profile="text"), refused


# --- state 10 and decision 16: reset to the base --------------------------------------------


def test_reset_goes_back_to_the_base_with_its_origin(words):
    session, c, _ = words
    assert attempt(c, "set budget 800")[1] is None
    back, refused = attempt(c, "set budget")
    assert refused is None, refused
    assert back is not None and back["setting"] == {"value": BUDGET, "origin": "default"}
    assert attempt(c, "set search_method meaning")[1] is None
    back = ran(c, "set search_method")
    assert back["setting"] == {"value": "words", "origin": "profile"}
    assert found_budget(c) == (BUDGET, "default")


# --- states 11 and decision 3: resumed --------------------------------------------------------


def resumed(session: Session, c: Commands, root: Path, tokenizer) -> tuple[Session, Commands]:
    assert session.conversation is not None
    log = session.conversation.log.path
    session.close()
    backend = FakeBackend([f" S{i}." for i in range(20)], tokenizer=tokenizer)
    fresh = Session(
        root,
        backend=backend,
        tokenizer_loader=lambda r, v: tokenizer,
        embedder=FakeEmbedder(),  # ty: ignore[invalid-argument-type]
    )
    again = Commands(fresh)
    ran(again, f"resume {log.stem}")
    return fresh, again


def test_a_resumed_chat_has_its_values_back_with_origin_chat(fetched_root, chat_tokenizer, words):
    session, c, _ = words
    for line in ("set budget 800", "set min_similarity 0.3"):
        assert attempt(c, line)[1] is None, line
    say(c, ASKED)
    fresh, again = resumed(session, c, fetched_root, chat_tokenizer)
    profile_words, default = {"value": "words", "origin": "profile"}, "default"
    assert documents_of(again) == {
        "search_method": {**profile_words, "base": profile_words},
        "min_similarity": {
            "value": 0.3,
            "origin": "chat",
            "base": {"value": 0.5, "origin": default},
        },
        "budget": {"value": 800, "origin": "chat", "base": {"value": BUDGET, "origin": default}},
    }
    assert found_budget(again) == (800, "chat")


def test_on_resume_the_method_comes_from_the_last_searched_message_after_the_profile(
    fetched_root, chat_tokenizer
):
    """Decision 3's order against a profile with no method: the chat's /set; then the logged
    profile's (none here); then the last searched message's, origin `log`; then the default."""
    session, c, _ = chat_on(fetched_root, chat_tokenizer, PLAIN)
    assert attempt(c, "set search_method words")[1] is None
    assert say(c, ASKED)["method"] == "words"
    ran(c, "set search_method")  # back to meaning, Clarity's default, before the resume
    fresh, again = resumed(session, c, fetched_root, chat_tokenizer)
    assert fresh.conversation is not None and fresh.conversation.search_method == "words"
    method = (documents_of(again) or {}).get("search_method")
    assert method == {
        "value": "words",
        "origin": "log",
        "base": {"value": "words", "origin": "log"},
    }
    ran(again, "set search_method meaning")
    assert ran(again, "set search_method")["setting"] == {"value": "words", "origin": "log"}


# --- states 12 and 13: the method's embedder -----------------------------------------------


def test_choosing_meaning_without_bge_m3_says_so_as_search_does(words):
    session, c, fake = words
    fake.is_fetched = False
    result, refused = attempt(c, "set search_method meaning")
    assert refused is None, refused
    assert result is not None and result.get("warnings") == [{"state": "no_embedder"}]


def test_choosing_words_stops_the_embedder(meaning):
    session, c, fake = meaning
    say(c, ASKED)  # the embedder starts
    closed = fake.closed
    assert attempt(c, "set search_method words")[1] is None
    assert fake.closed == closed + 1


# --- M1: a chat origin in found, through every reader ----------------------------------------


def test_found_logs_the_chat_s_values_and_every_reader_reads_them(
    fetched_root, chat_tokenizer, meaning
):
    session, c, _ = meaning
    for line in ("set budget 800", "set min_similarity 0.6", "set search_method words"):
        assert attempt(c, line)[1] is None, line
    by_words = say(c, ASKED)
    assert (by_words["method"], by_words["budget"], by_words["budget_origin"]) == (
        "words",
        800,
        "chat",
    )
    ran(c, "set search_method meaning")
    by_meaning = say(c, "Why does rock wear away?")
    assert by_meaning["method"] == "meaning"
    assert (by_meaning["min_similarity"], by_meaning["min_similarity_origin"]) == (0.6, "chat")
    assert (by_meaning["budget"], by_meaning["budget_origin"]) == (800, "chat")
    assert session.conversation is not None
    log = session.conversation.log.path
    calls = {r["call_id"]: r["output_text"] for r in calls_of(session)}
    for line in (
        "history",
        *(f"{name} {call}" for call in calls for name in ("prompt", "compare")),
    ):
        ran(c, line)
    ran(c, "new")
    ran(c, f"resume {log.stem}")
    assert export_log(session, log)["path"]
    ran(c, "export inspect")
    for call, output in calls.items():  # with no chat open, as `clarity replay` does; replay
        replayed = Session(  # never searches
            fetched_root,
            backend=FakeBackend([output], tokenizer=chat_tokenizer),
            tokenizer_loader=lambda r, v: chat_tokenizer,
            embedder=Untouchable(),  # ty: ignore[invalid-argument-type]
        )
        assert replayed.replay(log, call)["identical"], call


def test_the_chat_origin_has_words(words):
    """State 19: "this chat's budget" (Search it), "this chat" (a minimum's origin)."""
    from clarity_chat.web import search_choice

    session, c, _ = words
    for line in ("set budget 800", "set min_similarity 0.3", "set search_method meaning"):
        assert attempt(c, line)[1] is None, line
    state = ran(c, "state")
    assert state["budget"] == {"tokens": 800, "origin": "chat"}
    assert state["min_similarity"] == {"value": 0.3, "origin": "chat"}
    note = search_choice("notes.md", state)["note"]
    assert "this chat's budget" in note and "0.3, this chat)" in note, note


# --- decision 2: Regenerate keeps the parts; outside a chat, the profile's values -------------


def test_a_regenerated_reply_s_prompt_is_its_sibling_s_after_a_budget_change(words):
    session, c, _ = words
    first = ran(c, f"say {ASKED}")["call"]
    assert attempt(c, "set budget 5")[1] is None
    (again,) = (r["call"] for r in ran(c, "regen")["replies"])
    assert again["prompt_ids"] == first["prompt_ids"]


def test_document_try_reads_the_profile_s_values(words):
    session, c, _ = words
    assert attempt(c, "set budget 5")[1] is None
    found, _ = session.try_documents(["notes"], "searching", ASKED)
    assert (found["budget"], found["budget_origin"]) == (BUDGET, "default")


def test_a_batch_run_reads_the_profile_s_values(fetched_root, chat_tokenizer, words):
    from clarity_chat.batch import run_batch

    session, c, _ = words
    assert attempt(c, "set budget 5")[1] is None
    (fetched_root / "openings.txt").write_text(ASKED + "\n", encoding="utf-8")
    run_dir = fetched_root / "runs" / "r"
    run_batch(
        session, "r", run_dir, fetched_root / "openings.txt", ["searching"], 1, lambda line: None
    )
    found = [r["found"] for r in read_records(run_dir / "calls.jsonl") if "found" in r]
    assert found and {(f["budget"], f["budget_origin"]) for f in found} == {(BUDGET, "default")}
