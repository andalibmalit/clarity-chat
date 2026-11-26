"""Live checks of the search keys a chat sets, on bge-m3 beside SmolLM2 (v0.4.4's Live tiers,
the sprint audit's M1). Each goes through the command a chat sends: `/set search_method
words` stops the embedder's process and meaning starts it again; `/set min_similarity`
shows in `found` with origin `chat`, and the chat resumes.

Run with `uv run pytest -m embed -v`. Needs llama-server built, bge-m3 and SmolLM2 fetched.
The library is a scratch folder (test_live's scratch_library), never local/documents/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clarity_chat.backends.llama_cpp import server_binary
from clarity_chat.commands import Commands
from clarity_chat.documents.embedder import MODEL as EMBEDDER
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.session import Session
from tests.test_live import MODEL, REPO, all_checks_ok, scratch_library

pytestmark = pytest.mark.embed


def _available() -> str | None:
    try:
        server_binary(load_server_config(REPO).llama_cpp_dir)
        models = load_models(REPO)
        for name in (EMBEDDER, MODEL):
            check_weights_quick(REPO, lock_entry(REPO, models[name]))
    except ConfigError as e:
        return str(e)
    return None


if (reason := _available()) is not None:
    pytestmark = [pytest.mark.embed, pytest.mark.skip(reason=reason)]


def quiet(_event: dict) -> None:
    pass


def meaning_chat(tmp_path: Path) -> str:
    """smol-chat searching the scratch library's trail notes by meaning, stated."""
    text = (REPO / "profiles" / "smol-chat.toml").read_text(encoding="utf-8")
    path = tmp_path / "smol-chat-meaning.toml"
    documents = '\n[documents]\nsearch = ["trail"]\nsearch_method = "meaning"\n'
    path.write_text(text + documents, encoding="utf-8")
    return str(path)


def say(c: Commands, text: str) -> dict[str, Any]:
    """Send `text`; the `found` its turn holds."""
    all_checks_ok(c.run("say", text, quiet)["call"])
    conversation = c.session.conversation
    assert conversation is not None and conversation.head is not None
    reply = conversation.turns[conversation.head]
    found = conversation.turns[reply.parent_id or ""].found
    assert found is not None
    return found


@pytest.fixture
def chat(tmp_path, monkeypatch):
    scratch_library(tmp_path, monkeypatch)
    session = Session(REPO, logs_dir=tmp_path / "logs")
    c = Commands(session)
    c.run("profile", meaning_chat(tmp_path), quiet, paths=True)  # as the terminal names one
    yield c
    session.close()


def test_set_search_method_words_stops_the_embedder_and_meaning_starts_it_again(chat):
    assert say(chat, "What marks the ridge?")["method"] == "meaning"
    embedder = chat.session.embedder()
    assert embedder.server is not None and embedder.server.proc is not None
    first = embedder.server.proc
    assert first.poll() is None

    chat.run("set", "search_method words", quiet)
    assert first.poll() is not None, "the embedder's process runs on after words"
    assert embedder.server is None
    assert say(chat, "Is there water after the pass?")["method"] == "words"
    assert embedder.server is None, "a words search started the embedder"

    chat.run("set", "search_method", quiet)  # back to the profile's meaning
    assert say(chat, "What marks the ridge?")["method"] == "meaning"
    assert embedder.server is not None and embedder.server.proc is not None
    again = embedder.server.proc
    assert again.poll() is None and again.pid != first.pid


def test_set_min_similarity_shows_in_found_as_the_chat_s_and_the_chat_resumes(chat):
    chat.run("set", "min_similarity 0.9", quiet)
    found = say(chat, "What marks the ridge?")
    assert found["method"] == "meaning"
    assert (found["min_similarity"], found["min_similarity_origin"]) == (0.9, "chat")
    conversation = chat.session.conversation
    assert conversation is not None
    resumed = chat.session.resume(conversation.log.path)
    assert resumed["render"]["identical"] is True, resumed["render"]
    again = say(chat, "And the pass?")
    assert (again["min_similarity"], again["min_similarity_origin"]) == (0.9, "chat")
