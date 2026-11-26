"""v0.4.4 A5: a model file whose modified time changed is checked again (decision 15).

When the file's size matches models.lock and its time doesn't, a new chat, a resume and the
embedder's start hash it; if the bytes match, the new time is recorded in local/models.lock
and one line says so (states 1 and 2). Replay and a batch run refuse as before (state 3).
Other bytes, or another size, are refused with the fetch hint (states 4 and 5). The listing
and setup's "fetched" never hash or write (state 6). The weights are a small made-up file,
hashed for real; no live test touches a shared Hugging Face cache file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from clarity_chat import models
from clarity_chat.batch import check_run
from clarity_chat.logs import read_records
from clarity_chat.models import (
    list_profiles,
    read_lock,
    read_lock_file,
    weights_path,
    write_lock_entry,
)
from clarity_chat.profiles import LOCAL_LOCK, ConfigError, fetch_hint, load_models
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_meaning import raised

RECORDED = (
    "The model file's modified time changed since it was fetched; its bytes still match, so "
    "the new time is recorded in local/models.lock."
)
TOUCHED_NS = 1_000_000_000  # a modified time no fetch wrote


def hashed(root: Path, name: str = "tiny") -> Path:
    """The model's weights, with the lock's sha256 made the file's own (fake_fetch writes
    zeros), so a re-check can find the bytes the same."""
    entry = read_lock(root)["models"][name]
    path = weights_path(root, entry)
    entry["gguf"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_lock_entry(root, name, entry)
    return path


def touched(path: Path) -> None:
    """The same bytes with another modified time, as a copy or `touch` leaves them."""
    os.utime(path, ns=(TOUCHED_NS, TOUCHED_NS))


def locked_gguf(root: Path, name: str = "tiny") -> dict:
    return read_lock_file(root / LOCAL_LOCK)["models"][name]["gguf"]


def no_hashing(monkeypatch) -> None:
    def refuse(path: Path, *args, **kwargs) -> str:
        raise AssertionError(f"{path} was hashed")

    monkeypatch.setattr(models, "sha256_file", refuse)


def counted_hashes(monkeypatch) -> list[Path]:
    seen: list[Path] = []
    real = models.sha256_file

    def count(path: Path, *args, **kwargs) -> str:
        seen.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(models, "sha256_file", count)
    return seen


@pytest.fixture
def weights(fetched_root) -> Path:
    return hashed(fetched_root)


def session_on(root: Path, tiny_tokenizer) -> Session:
    backend = FakeBackend([f" R{i}." for i in range(5)], tokenizer=tiny_tokenizer)
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)


# --- state 6: the listing and setup's "fetched" ---------------------------------------------


def test_the_listing_and_setup_count_a_touched_file_as_fetched_and_hash_nothing(
    fetched_root, weights, monkeypatch
):
    from tests.test_setup import made

    touched(weights)
    lock = (fetched_root / LOCAL_LOCK).read_bytes()
    no_hashing(monkeypatch)
    model = load_models(fetched_root)["tiny"]
    listed = {p["name"]: p["fetched"] for p in list_profiles(fetched_root)}
    assert listed["text"], "a touched file of the fetched size is listed as not fetched"
    assert made(fetched_root).fetched(model)
    assert (fetched_root / LOCAL_LOCK).read_bytes() == lock
    weights.write_bytes(b"GGUF fake weights, and more")  # another size: not fetched
    assert not {p["name"]: p["fetched"] for p in list_profiles(fetched_root)}["text"]
    assert not made(fetched_root).fetched(model)


# --- state 1: a new chat and a resume -------------------------------------------------------


def test_a_new_chat_records_the_new_time_says_so_and_starts(fetched_root, weights, tiny_tokenizer):
    session = session_on(fetched_root, tiny_tokenizer)
    touched(weights)
    refused = raised(lambda: session.new_conversation("text"))
    assert refused is None, refused
    conversation = session.conversation
    assert conversation is not None
    assert RECORDED in conversation.warnings
    logged = [r for r in read_records(conversation.log.path) if r["type"] == "conversation"]
    assert RECORDED in logged[0]["warnings"]
    assert locked_gguf(fetched_root)["mtime_ns"] == TOUCHED_NS
    assert session.say("Hi").record["stop_type"] in ("eos", "limit")
    again = session.new_conversation("text")  # the quick check passes on the recorded time
    assert RECORDED not in again.warnings


def test_a_resume_hashes_once_and_logs_the_line(fetched_root, weights, tiny_tokenizer, monkeypatch):
    session = session_on(fetched_root, tiny_tokenizer)
    log = session.new_conversation("text").log.path
    session.say("Hi")
    session.close()
    touched(weights)
    hashes = counted_hashes(monkeypatch)
    fresh = session_on(fetched_root, tiny_tokenizer)
    resumed = raised(lambda: fresh.resume(log))
    assert resumed is None, resumed
    assert fresh.conversation is not None and RECORDED in fresh.conversation.warnings
    assert RECORDED in [r for r in read_records(log) if r["type"] == "resume"][0]["warnings"]
    assert hashes == [weights], "the second prepare passes the quick check on the new time"
    assert locked_gguf(fetched_root)["mtime_ns"] == TOUCHED_NS


# --- state 3: replay and a batch run --------------------------------------------------------


def test_replay_and_a_batch_run_refuse_as_before(
    fetched_root, weights, tiny_tokenizer, monkeypatch
):
    session = session_on(fetched_root, tiny_tokenizer)
    log = session.new_conversation("text").log.path
    call_id = session.say("Hi").record["call_id"]
    session.close()
    touched(weights)
    lock = (fetched_root / LOCAL_LOCK).read_bytes()
    no_hashing(monkeypatch)
    today = r"changed since it was fetched \(size or modified time differs .*verify tiny"
    with pytest.raises(ConfigError, match=today):
        session_on(fetched_root, tiny_tokenizer).replay(log, call_id)
    with pytest.raises(ConfigError, match=today):
        check_run(session_on(fetched_root, tiny_tokenizer), ["text"], ["Hi"], 1)
    assert (fetched_root / LOCAL_LOCK).read_bytes() == lock


# --- states 4 and 5: other bytes, another size ----------------------------------------------


def test_other_bytes_of_the_same_size_are_refused_with_the_fetch_hint(
    fetched_root, weights, tiny_tokenizer
):
    weights.write_bytes(b"GGUF fake weightz")  # as long as fake_fetch's
    touched(weights)
    lock = (fetched_root / LOCAL_LOCK).read_bytes()
    refused = raised(lambda: session_on(fetched_root, tiny_tokenizer).new_conversation("text"))
    assert isinstance(refused, ConfigError), refused
    assert "changed since it was fetched" in str(refused) and "sha256" in str(refused)
    assert str(refused).endswith(fetch_hint("tiny")), str(refused)
    assert (fetched_root / LOCAL_LOCK).read_bytes() == lock


def test_another_size_is_refused_with_the_fetch_hint_without_hashing(
    fetched_root, weights, tiny_tokenizer, monkeypatch
):
    weights.write_bytes(b"GGUF fake weights, and more")
    lock = (fetched_root / LOCAL_LOCK).read_bytes()
    no_hashing(monkeypatch)
    refused = raised(lambda: session_on(fetched_root, tiny_tokenizer).new_conversation("text"))
    assert isinstance(refused, ConfigError), refused
    assert "changed since it was fetched" in str(refused) and "bytes" in str(refused)
    assert str(refused).endswith(fetch_hint("tiny")), str(refused)
    assert (fetched_root / LOCAL_LOCK).read_bytes() == lock
