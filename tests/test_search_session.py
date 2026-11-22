"""B1 (v0.3.2): the profile's [documents] keys (search, budget, placement), /search and its
`search` record, /documents, and /state's `searching`. Made-up documents, a scratch root."""

from __future__ import annotations

import json
import os
import shutil

import pytest

from clarity_chat.commands import Commands, check_lines
from clarity_chat.documents import DocumentRefused, library_add, reading
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend

NONE: dict = {}  # a ready document's entry holds no counts (cut 5)
GLOSSARY = b"## Wren\nA small brown bird.\n\n## Dipper\nIt walks under water.\n"


def quiet(event: dict) -> None:
    pass


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    library_add(fetched_root, b"Made-up notes.\n", "notes.md")
    return fetched_root


def profile(root, documents: str) -> None:
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + f"\n[documents]\n{documents}\n")


def chat(root, chat_tokenizer, events: list | None = None) -> tuple[Session, Commands]:
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    c = Commands(session)
    c.run("profile", "chat", quiet)
    return session, c


def run(c: Commands, line: str, emit=quiet) -> dict:
    return c.run(*line.partition(" ")[::2], emit)


# --- the profile's keys


@pytest.mark.parametrize(
    ("documents", "says"),
    [
        ('search = "glossary"', "[documents] search must be a list of document names."),
        ('search = ["Glossary"]', "[documents] search: A document's name is 1 to 64"),
        ('search = ["con"]', "[documents] search: con can't name a document"),
        ('search = ["../x"]', "[documents] search: A document's name is 1 to 64"),
        ('search = ["glossary", "glossary"]', "[documents] search names a document twice."),
        ("budget = 0", "[documents] budget must be a positive whole number of tokens, got 0."),
        ("budget = 1.5", "[documents] budget must be a positive whole number of tokens"),
        ('budget = "x"', "[documents] budget must be a positive whole number of tokens"),
        ('placement = "middle"', '[documents] placement must be "start" or "end".'),
    ],
)
def test_the_documents_keys_are_checked(root, chat_tokenizer, documents, says):
    profile(root, documents)
    with pytest.raises(ConfigError) as refused:
        chat(root, chat_tokenizer)
    assert says in str(refused.value), refused.value


def test_a_profile_naming_a_document_not_in_the_library_is_refused_naming_the_key(
    root, chat_tokenizer
):
    profile(root, 'search = ["glossary", "birds"]')
    with pytest.raises(ConfigError) as refused:
        chat(root, chat_tokenizer)
    assert str(refused.value) == (
        "Profile chat searches birds, which isn't in your documents. Take it out of search "
        "under [documents] in the profile's file."  # the page's start: no chat to add it in
    )


def test_the_profile_s_list_budget_and_placement(root, chat_tokenizer):
    profile(root, 'search = ["glossary"]\nbudget = 900\nplacement = "end"')
    session, c = chat(root, chat_tokenizer)
    assert run(c, "search") == {
        "searching": [{"name": "glossary", "state": "ready", **NONE}],
        "origin": "profile",
        "method": "words",  # v0.4.1, and warnings (B1)
        "warnings": [],
    }
    assert session.budget() == (900, "profile")
    assert session.conversation is not None and session.conversation.profile.placement == "end"
    assert run(c, "state")["searching"] == [{"name": "glossary", "state": "ready", **NONE}]


def test_with_no_documents_table_nothing_is_searched_at_the_default_budget(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    assert run(c, "search") == {
        "searching": [],
        "origin": "profile",
        "method": "words",
        "warnings": [],
    }
    assert session.budget() == (1500, "default")
    assert session.conversation is not None and session.conversation.profile.placement == "start"


# --- /search and its record


def test_search_sets_the_list_and_logs_the_whole_set(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    assert run(c, "search glossary notes") == {
        "searching": [
            {"name": "glossary", "state": "ready", **NONE},
            {"name": "notes", "state": "ready", **NONE},
        ],
        "origin": "chat",
        "method": "words",  # v0.4.1, and warnings (B1)
        "warnings": [],
    }
    run(c, "search notes")
    run(c, "search none")
    assert session.conversation is not None
    log = session.conversation.log.path
    records = [r for r in read_records(log) if r["type"] == "search"]
    assert [r["documents"] for r in records] == [["glossary", "notes"], ["notes"], []]
    assert set(records[0]) == {"schema_version", "time", "type", "conversation_id", "documents"}


def test_a_name_not_in_the_library_is_refused_and_nothing_changes(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    run(c, "search glossary")
    for args, says in [
        ("birds", "birds isn't in your documents."),
        ("../x", "A document's name is 1 to 64"),
        ("glossary glossary", "Name each document once."),
    ]:
        with pytest.raises(ConfigError) as refused:
            run(c, f"search {args}")
        assert str(refused.value).startswith(says), refused.value
    assert run(c, "search")["searching"] == [{"name": "glossary", "state": "ready", **NONE}]


def test_search_is_resumed_and_a_log_without_a_record_searches_the_profile_s(root, chat_tokenizer):
    profile(root, 'search = ["glossary"]')
    session, c = chat(root, chat_tokenizer)
    assert session.conversation is not None
    first = session.conversation.log.path
    run(c, "say Hello")
    run(c, "new")
    run(c, "search notes")
    run(c, "say Hello")
    second = session.conversation.log.path
    session.resume(first)
    assert run(c, "search") == {
        "searching": [{"name": "glossary", "state": "ready", **NONE}],
        "origin": "profile",
        "method": "words",  # v0.4.1, and warnings (B1)
        "warnings": [],
    }
    session.resume(second)
    assert run(c, "search") == {
        "searching": [{"name": "notes", "state": "ready", **NONE}],
        "origin": "chat",
        "method": "words",  # v0.4.1, and warnings (B1)
        "warnings": [],
    }


def test_a_search_record_with_a_name_clarity_wouldn_t_write_is_refused_by_resume(
    root, chat_tokenizer
):
    session, c = chat(root, chat_tokenizer)
    run(c, "search glossary")
    assert session.conversation is not None
    log = session.conversation.log.path
    text = log.read_text(encoding="utf-8").replace('["glossary"]', '["../../etc"]')
    log.write_text(text, encoding="utf-8")
    run(c, "new")
    with pytest.raises(DocumentRefused):
        session.resume(log)


def test_an_unreadable_or_missing_document_is_listed_and_search_without_it_works(
    root, chat_tokenizer
):
    session, c = chat(root, chat_tokenizer)
    run(c, "search glossary notes")
    (root / "local/documents/glossary/parts.jsonl").write_text("Edited.\n", encoding="utf-8")
    shutil.rmtree(root / "local/documents/notes")
    assert [(d["name"], d["state"]) for d in run(c, "search")["searching"]] == [
        ("glossary", "unreadable"),
        ("notes", "missing"),
    ]
    assert run(c, "search none")["searching"] == []


def test_a_text_mode_chat_can_t_search(root, tiny_tokenizer):
    backend = FakeBackend([" R."], tokenizer=tiny_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", quiet)
    with pytest.raises(ConfigError, match="^Searching needs a chat-mode profile"):
        run(c, "search glossary")


# --- /documents


def test_documents_lists_adds_and_replaces_and_says_where_it_writes(root, chat_tokenizer, tmp_path):
    session, c = chat(root, chat_tokenizer)
    assert [d["name"] for d in run(c, "documents")["documents"]] == ["glossary", "notes"]
    path = tmp_path / "Birds.md"
    path.write_bytes(b"## Heron\nTall.\n")
    events: list[dict] = []
    added = run(c, f"documents add {path}", events.append)
    assert added["added"]["name"] == "birds"
    assert events == [
        {"event": "reading", "name": "Birds.md"},
        {"event": "writing", "folder": "local/documents/birds", "replacing": False},
    ]
    named = run(c, f"documents add {path} herons")
    assert named["added"]["name"] == "herons"
    events.clear()
    assert run(c, f"documents add {path}", events.append)["replaced"] is True  # cut 1
    assert events[-1] == {"event": "writing", "folder": "local/documents/birds", "replacing": True}


def test_documents_add_from_the_page_takes_an_upload_and_gives_the_same_manifest(
    root, chat_tokenizer, tmp_path
):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    uploads = {"u-1": (b"## Heron\nTall.\n", "Birds.md")}
    page = Commands(session, uploads=uploads)
    page.run("profile", "chat", quiet)
    from_page = page.run("documents", "add u-1", quiet)["added"]
    path = tmp_path / "Birds.md"
    path.write_bytes(b"## Heron\nTall.\n")
    _, terminal = chat(root, chat_tokenizer)
    from_terminal = run(terminal, f"documents add {path} birds-again")["added"]
    for manifest in (from_page, from_terminal):
        manifest.pop("added"), manifest.pop("name")
    assert from_page == from_terminal
    with pytest.raises(ConfigError, match="^That file isn't uploaded any more"):
        page.run("documents", "add u-2", quiet)


def test_documents_needs_no_chat(root, chat_tokenizer):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    assert len(Commands(session).run("documents", "", quiet)["documents"]) == 2


# --- the attach sheet's room when the chat searches (S1)


def test_a_searching_chat_s_check_sets_the_budget_aside(root, chat_tokenizer, tmp_path):
    session, c = chat(root, chat_tokenizer)
    path = tmp_path / "long.md"
    path.write_text("".join(f"Made-up line {n}.\n" for n in range(400)), encoding="utf-8")
    backend = session.backend
    assert isinstance(backend, FakeBackend)
    backend._n_ctx = 2000
    assert run(c, f"attach {path} check")["check"]["aside"] == 0
    run(c, "search glossary")
    check = run(c, f"attach {path} check")["check"]
    assert (check["aside"], check["aside_origin"]) == (1500, "default")
    assert any("1,500 for the parts search can add" in line for line in check_lines(check))


def test_state_has_searching_before_a_chat(root, chat_tokenizer):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    assert Commands(session).run("state", "", quiet)["searching"] == []


def test_the_search_record_reads_as_json(root, chat_tokenizer):
    session, c = chat(root, chat_tokenizer)
    run(c, "search glossary")
    assert session.conversation is not None
    last = json.loads(session.conversation.log.path.read_text().splitlines()[-1])
    assert (last["type"], last["documents"]) == ("search", ["glossary"])


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_linked_document_is_refused_by_search_and_a_start(root, chat_tokenizer, tmp_path):
    """Checkpoint 1's M1, re-checked against B1 (rule 13): /search and a profile's list
    meet the library's refusal of a linked folder, never the folder behind it."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "local/documents/linked").symlink_to(elsewhere, target_is_directory=True)
    session, c = chat(root, chat_tokenizer)
    with pytest.raises(DocumentRefused, match="^linked's folder in local/documents/ is a link"):
        run(c, "search linked")
    assert run(c, "documents")["documents"][1]["state"] == "unreadable"
    profile(root, 'search = ["linked"]')
    with pytest.raises(ConfigError, match=r"^Profile chat searches linked\. "):
        chat(root, chat_tokenizer)
    assert list(elsewhere.iterdir()) == []


# --- Checkpoint 1's S8: the refusal's suggested start, in a searching chat


def test_a_searching_chat_s_refusal_suggests_a_start_that_leaves_the_budget(
    root, chat_tokenizer, tmp_path
):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer, n_ctx=2000)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    c = Commands(session)
    c.run("profile", "chat", lambda e: None)
    path = tmp_path / "long.md"
    path.write_text("".join(f"Made-up line {n}.\n" for n in range(400)), encoding="utf-8")

    def suggested() -> int:
        with pytest.raises(ConfigError) as refused:
            c.run("attach", str(path), lambda e: None)
        refusal = str(refused.value)
        return int(refusal.split(f"/attach {path} lines 1-")[1].split(".")[0])

    plain = suggested()
    c.run("search", "glossary", lambda e: None)
    assert suggested() < plain
