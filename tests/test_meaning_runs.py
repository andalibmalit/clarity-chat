"""v0.4.1 B2: /try, `clarity run` and the closest score by meaning (decision 12's /try).

/try gives what a message would find, logs nothing, and, when no part reaches the minimum,
says the closest similarity and the minimum. A run searches each opening once per profile
by that profile's method, prepares a document's vectors before any chat model loads, and
says so once. Export rows and /history's summary are A3's (test_meaning); `clarity document
try`'s refusal is the fix review's (81f59e0). FakeEmbedder stands in for bge-m3. Made-up
text only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from clarity_chat.batch import RunError, run_batch
from clarity_chat.cli import ChatLoop
from clarity_chat.documents import SearchRefused, library_add, reading
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_meaning import (
    ASKED,
    GLOSSARY,
    MEANING,
    meaning,  # noqa: F401 (a fixture)
    profile,
    say,
)

WORDS = '\n[documents]\nsearch = ["notes"]\nsearch_method = "words"\n'
OPENINGS = ["How do leaves turn light into sugar?", "Why does rock wear away?"]


def quiet(event: dict[str, Any]) -> None:
    pass


def test_try_gives_the_message_s_found_and_logs_nothing(meaning):  # noqa: F811
    session, c, _ = meaning
    assert session.conversation is not None
    log = session.conversation.log.path
    before = log.read_bytes()
    tried = c.run("try", ASKED, quiet)["found"]
    assert log.read_bytes() == before
    assert say(c, ASKED) == tried


def test_try_says_the_closest_similarity_when_nothing_reaches_the_minimum(meaning):  # noqa: F811
    session, c, _ = meaning
    best = c.run("try", ASKED, quiet)["found"]["parts"][0]["score"]  # the 0.2 profile's
    profile(session.root, "strict", MEANING.replace("0.2", "0.99"))
    c.run("profile", "strict", quiet)
    result = c.run("try", ASKED, quiet)
    assert result["found"]["matched"] == 0
    assert result.get("closest") == best
    output: list[str] = []
    loop = ChatLoop(session, "strict", read=lambda default: "", write=output.append)
    loop.start()
    output.clear()
    loop.handle("/try " + ASKED)
    shown = math.floor(best * 100) / 100  # rounded down, as the terminal shows it
    assert f"Nothing matched; the closest was {shown:.2f}, and the minimum is 0.99." in "".join(
        output
    )
    # The message's own found and line don't carry it (/try only).
    loop.handle(ASKED)
    assert "  searched by meaning: nothing matched\n" in output


def test_try_names_no_closest_when_something_matched_or_by_words(meaning):  # noqa: F811
    session, c, _ = meaning
    assert c.run("try", ASKED, quiet).get("closest") is None
    c.run("profile", "chat", quiet)
    c.run("search", "notes", quiet)
    assert c.run("try", "nothing shared here", quiet).get("closest") is None


def test_a_run_searches_each_opening_once_per_profile_by_its_method(
    fetched_root, chat_tokenizer, tmp_path
):
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    profile(fetched_root, "words", WORDS)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    openings = fetched_root / "openings.txt"
    openings.write_text("\n---\n".join(OPENINGS) + "\n", encoding="utf-8")
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    fake = FakeEmbedder()
    built_before_a_model: list[int] = []
    start = backend.start

    def starting(*args: Any, **kwargs: Any) -> Any:  # what was built when a model first loads
        if not built_before_a_model:
            built_before_a_model.append(fake.made)
        return start(*args, **kwargs)

    backend.start = starting
    run_dir = fetched_root / "runs" / "run-meaning"
    session = Session(
        fetched_root,
        backend=backend,
        logs_dir=run_dir,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    output: list[str] = []
    run_batch(session, "run-meaning", run_dir, openings, ["meaning", "words"], 2, output.append)
    # Once in the pre-check, once for the run's block: not once per seed.
    assert [fake.texts.count(o) for o in OPENINGS] == [2, 2]
    assert built_before_a_model and built_before_a_model[0] >= 3  # notes' three parts
    said = "Preparing notes for meaning search: 3 parts, saved in local/documents/notes/.\n"
    assert output.count(said) == 1
    records = [json.loads(line) for line in (run_dir / "calls.jsonl").open(encoding="utf-8")]
    profile_of = {
        r["conversation_id"]: r["profile"]["name"] for r in records if r["type"] == "conversation"
    }
    methods = {
        (profile_of[r["conversation_id"]], r["found"]["method"])
        for r in records
        if r["type"] == "turn" and "found" in r
    }
    assert methods == {("meaning", "meaning"), ("words", "words")}


def test_a_run_without_bge_m3_isn_t_told_to_start_a_new_chat(fetched_root, chat_tokenizer):
    """Provider-trial #6b (the whole-branch review's C14): state 5's way out honours where
    the search ran. A run has no chat to start: its words say to run the command again."""
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    openings = fetched_root / "openings.txt"
    openings.write_text(OPENINGS[0] + "\n", encoding="utf-8")
    fake = FakeEmbedder()
    fake.is_fetched = False
    run_dir = fetched_root / "runs" / "run-unfetched"
    session = Session(
        fetched_root,
        backend=FakeBackend([], tokenizer=chat_tokenizer),
        logs_dir=run_dir,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=fake,  # ty: ignore[invalid-argument-type]
    )
    said: list[str] = []
    try:
        run_batch(session, "run-unfetched", run_dir, openings, ["meaning"], 1, said.append)
        refused = None
    except SearchRefused as e:
        refused = e
    assert refused is not None, "the run started without bge-m3"
    assert refused.problems == [{"state": "no_embedder"}]
    assert "new chat" not in refused.terminal, refused.terminal
    assert refused.terminal.endswith("Or " + WORDS_IN_A_RUN), refused.terminal
    assert fake.made == 0


def metal(total: float, limit: int) -> dict[str, Any]:
    return {"device": "MTL0", "limit_mib": limit, "model_mib": total - 700, "kv_mib": 0,
            "compute_mib": 700, "total_mib": total}  # fmt: skip


def unfetched_run(root: Path, chat_tokenizer: Any, embedder: Any, backend: Any) -> Exception:
    """A one-opening run of profile meaning; what it raised."""
    profile(root, "meaning", MEANING)
    library_add(root, GLOSSARY.encode(), "notes.md")
    openings = root / "openings.txt"
    openings.write_text(OPENINGS[0] + "\n", encoding="utf-8")
    run_dir = root / "runs" / "run-refused"
    session = Session(
        root,
        backend=backend,
        logs_dir=run_dir,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=embedder,
    )
    said: list[str] = []
    try:
        run_batch(session, "run-refused", run_dir, openings, ["meaning"], 1, said.append)
    except Exception as e:
        return e
    raise AssertionError("the run wasn't refused")


WORDS_IN_A_RUN = 'search by words: set search_method to "words".'  # D6: a chat's too (v0.4.4)


def test_a_run_whose_embedder_is_over_metal_s_limit_isn_t_told_to_start_a_new_chat(
    fetched_root, chat_tokenizer
):
    """#6b's sibling (the S-e ruling): the memory refusal's way out by words, said in a
    run through state 6's reason, says to run the command again, as state 5 does."""
    from clarity_chat.backends.llama_cpp import memory_problem

    too_big = memory_problem(8192, [metal(1800, 1500)], embedder=True)
    assert too_big is not None and too_big.endswith("S" + WORDS_IN_A_RUN[1:])

    class OverTheLimit(FakeEmbedder):
        def start(self) -> dict[str, Any]:
            raise ConfigError(too_big)

    refused = unfetched_run(fetched_root, chat_tokenizer, OverTheLimit(),
                            FakeBackend([], tokenizer=chat_tokenizer))  # fmt: skip
    assert isinstance(refused, SearchRefused), repr(refused)
    assert "new chat" not in refused.terminal, refused.terminal
    assert refused.terminal.endswith("S" + WORDS_IN_A_RUN[1:]), refused.terminal


def test_a_run_whose_chat_model_is_over_metal_s_limit_beside_the_embedder_says_run_again(
    fetched_root, chat_tokenizer
):
    """The same sentence from the chat model's start beside the embedder: a run says the
    run's way out."""
    from clarity_chat.backends.llama_cpp import memory_problem

    both = memory_problem(4096, [metal(9000, 10700)], beside=(8192, [metal(1800, 10700)]))
    assert both is not None and both.endswith("Or " + WORDS_IN_A_RUN)
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise ConfigError(both)

    backend.start = refuse
    refused = unfetched_run(fetched_root, chat_tokenizer, FakeEmbedder(), backend)
    assert isinstance(refused, RunError), repr(refused)
    assert "new chat" not in str(refused), str(refused)
    assert str(refused).endswith("Or " + WORDS_IN_A_RUN), str(refused)


def test_the_closest_is_shown_rounded_down_so_it_never_reads_as_the_minimum():
    """A5's small-talk case: 0.4987 under the default 0.5 must not read "0.50" (the main
    session's ruling on B2). The terminal rounds down to two places; the result's `closest`
    stays exact."""
    from clarity_chat.cli import format_found

    found = {"method": "meaning", "min_similarity": 0.5, "matched": 0}
    said = format_found(found, 0.4987)
    assert said.endswith("Nothing matched; the closest was 0.49, and the minimum is 0.5."), said


def test_a_run_comparing_meaning_and_words_names_the_method_as_what_differs():
    """The whole-branch review's S2: the guide's "Compare meaning and words" recipe."""
    from types import SimpleNamespace

    from clarity_chat.batch import parts_differ

    def p(method=None, minimum=None):
        profile = SimpleNamespace(
            budget=None,
            search=("notes",),
            placement="start",
            search_method=method,
            min_similarity=minimum,
        )
        model = SimpleNamespace(tokenizer_repo="t", tokenizer_revision="r")
        return SimpleNamespace(profile=profile, model=model)

    got = {"meaning": ["notes:1"], "words": ["notes:2"]}
    said = "Opening 1: the profiles got different parts (meaning: notes:1; words: notes:2)."
    assert parts_differ(1, got, {"meaning": p(), "words": p("words")}) == (
        f"{said} Their search methods differ, so the replies differ in what they were given too."
    )
    # A words profile's minimum does nothing, so it's no difference (the fix review's could 2).
    assert parts_differ(1, got, {"meaning": p(minimum=0.2), "words": p("words")}) == (
        f"{said} Their search methods differ, so the replies differ in what they were given too."
    )
    assert parts_differ(1, got, {"meaning": p(), "words": p(minimum=0.7)}) == (
        f"{said} Their minimum similarities differ, so the replies differ in what they were "
        "given too."
    )
