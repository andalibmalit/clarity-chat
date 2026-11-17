"""Part 1's whole-branch review (notes/research/reviews/v032-part1-whole.md; the status
file's "part 1 whole-branch rulings"). PM1: a search refusal carries its problems as data,
and each front end words its own way out: the page the board's sentences, the terminal
and --json their commands. Made-up documents, scratch roots."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

from clarity_chat.batch import run_batch
from clarity_chat.cli import ChatLoop
from clarity_chat.commands import Commands
from clarity_chat.documents import library_add, reading
from clarity_chat.json_chat import JsonChat
from clarity_chat.profiles import ConfigError
from clarity_chat.session import SearchRefused, Session
from tests.fakes import FakeBackend
from tests.test_documents_board import chat, quiet, refusal

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n"
PAGE_ONLY = ("paperclip",)


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    library_add(fetched_root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    return fetched_root


def folder(root: Path, name: str = "glossary") -> Path:
    return root / "local/documents" / name


def no_fts5(monkeypatch) -> None:
    import sqlite3

    from clarity_chat.documents import search as search_module

    class NoFts5:
        def execute(self, *args):
            raise sqlite3.OperationalError("no such module: fts5")

    search_module._indexes.clear()
    monkeypatch.setattr(search_module.sqlite3, "connect", lambda *a, **k: NoFts5())


def not_built(root: Path) -> None:
    parts = folder(root) / "parts.jsonl"
    parts.write_bytes(parts.read_bytes().replace(b"small", b"large"))


def make(root: Path, how: str, monkeypatch) -> None:
    """Make the searched document `how`, after the chat started."""
    if how == "not_built":  # cut 1: parts that aren't the ones Clarity built
        not_built(root)
    elif how == "missing":
        shutil.rmtree(folder(root))
    elif how == "unreadable":
        (folder(root) / "manifest.json").write_text('{"schema": 1}', encoding="utf-8")
    elif how == "no_fts5":
        no_fts5(monkeypatch)


NAME = {"not_built": "glossary", "missing": "glossary", "unreadable": "glossary",
        "control": "odd", "no_fts5": "glossary"}  # fmt: skip
TERMINAL = {  # a chat's ways out are the chat's commands (the main session's PM1 ruling)
    "not_built": "Clarity can't search glossary: its parts.jsonl isn't the one Clarity built. "
    "`/documents add PATH glossary` adds the file again.",
    "missing": "Clarity can't search glossary: its folder in local/documents/ is gone. Stop "
    "searching it with /search, or add the file again: `/documents add PATH glossary`.",
    "unreadable": "Clarity can't search glossary: its manifest.json can't be read (it isn't a "
    "manifest Clarity writes). `/documents add PATH glossary` adds the file again.",
    # the chat's own way out (the close batch: surface S2)
    "control": 'Part 1 of odd ("Wren") holds </s>, which this model reads as a control token, '
    "not text. Take it out of your file, then add it again: `/documents add PATH odd`. Or stop "
    "searching odd with /search.",
    "no_fts5": "Word search needs SQLite's FTS5, which the Python running Clarity doesn't "
    "have. Use a Python whose SQLite includes it. `/search none` "
    "turns search off for this chat.",
}
KINDS = list(TERMINAL)


def started(root: Path, tokenizer, how: str) -> Session:
    path = root / "profiles" / "chat.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text + f'\n[documents]\nsearch = ["{NAME[how]}"]\n', encoding="utf-8")
    backend = FakeBackend([" R."], tokenizer=tokenizer)
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)


@pytest.mark.parametrize("how", KINDS)
def test_the_terminal_names_its_own_way_out(root, chat_tokenizer, monkeypatch, how):
    session = started(root, chat_tokenizer, how)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    make(root, how, monkeypatch)
    output.clear()
    loop.handle("Tell me about the wren")
    said = "".join(output)
    assert f"! Nothing was sent.\n{TERMINAL[how]}" in said
    assert not any(words in said for words in PAGE_ONLY), said
    assert loop.draft == "Tell me about the wren"


@pytest.mark.parametrize("how", KINDS)
def test_json_gets_the_terminal_s_words_and_the_problems(root, chat_tokenizer, monkeypatch, how):
    session = started(root, chat_tokenizer, how)
    stdout = io.StringIO()
    json_chat = JsonChat(session, io.BytesIO(), stdout)
    json_chat.execute({"command": "profile", "args": "chat"})
    make(root, how, monkeypatch)
    json_chat.execute({"command": "say", "args": "Tell me about the wren"})
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    (error,) = [e for e in events if e["event"] == "error"]
    assert error["type"] == "SearchRefused"
    assert error["message"] == f"Nothing was sent.\n{TERMINAL[how]}"
    (problem,) = error.get("problems") or [{}]
    state = {"control": "control_tokens", "not_built": "unreadable"}.get(how, how)
    assert problem.get("state") == state
    assert problem.get("document") == (None if how == "no_fts5" else NAME[how])


def test_a_control_token_problem_holds_the_part(root, chat_tokenizer, monkeypatch):
    session = started(root, chat_tokenizer, "control")
    c = Commands(session)
    c.run("profile", "chat", quiet)
    with pytest.raises(SearchRefused) as refused:
        c.run("say", "Tell me about the wren", quiet)
    assert getattr(refused.value, "problems", None) == [
        {"document": "odd", "state": "control_tokens", "ordinal": 1, "title": "Wren",
         "page": None, "tokens": ["</s>"]}
    ]  # fmt: skip


def test_the_page_and_the_log_keep_the_board_s_words(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    make(root, "missing", None)
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said == (
        "Nothing was sent.\nClarity can't search glossary: its folder in local/documents/ is "
        "gone. Stop searching it, or add the file again with the paperclip's Search it."
    )


def test_a_run_s_pre_check_names_the_terminal_command(root, chat_tokenizer):
    not_built(root)
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    (root / "openings.txt").write_text("Tell me about the wren.\n", encoding="utf-8")
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(SearchRefused) as refused:
        run_batch(session, "run-x", root / "runs" / "run-x", root / "openings.txt", ["chat"], 1,
                  print)  # fmt: skip
    assert refused.value.terminal == (
        "Clarity can't search glossary: its parts.jsonl isn't the one Clarity built. `clarity "
        "document add FILE --name glossary` adds the file again."
    )
    assert backend.calls == []


def test_a_profile_s_missing_document_names_the_terminal_command(root, chat_tokenizer):
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["birds"]\n')
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(ConfigError) as refused:
        session.new_conversation("chat")
    assert str(refused.value) == (
        "Profile chat searches birds, which isn't in your documents. Take it out of search "
        "under [documents] in the profile's file."  # the page's start: no chat to add it in
    )
    # In a chat, the chat's command (the PM1 ruling); at the shell, the command line's.
    assert getattr(refused.value, "terminal", None) == (
        "Profile chat searches birds, which isn't in your documents. Add it with `/documents "
        "add PATH birds`, or take it out of search under [documents] in the profile's file."
    )
    from clarity_chat.cli import outside_chat

    assert isinstance(refused.value, SearchRefused)
    assert outside_chat(refused.value).terminal == (
        "Profile chat searches birds, which isn't in your documents. Add it with `clarity "
        "document add FILE --name birds`, or take it out of search under [documents] in the "
        "profile's file."
    )


# --- PS1: /history's summary, wider, no text


def history_found(c) -> list[dict]:
    return [t["found"] for t in c.run("history", "", quiet)["turns"] if "found" in t]


def test_history_s_summary_holds_the_budget_reasons_and_each_part_s_words(root, chat_tokenizer):
    from clarity_chat.cli import format_history

    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    c.run("say", "The wren again", quiet)
    first, again = history_found(c)
    assert first.get("budget") == 1500
    (part,) = first["parts"]
    assert set(part) == {"id", "title", "page", "rank", "title_words", "words", "tokens"}  # C3
    assert part["title_words"] == ["wren"] and part["tokens"] > 0
    assert first["left_out"] == {} and again["left_out"] == {"already in the chat": 1}
    lines = format_history(c.run("history", "", quiet)).splitlines()
    assert "  searched for wren: 1 matched, none added (already in the chat)" in lines


def test_history_says_when_parts_didn_t_fit_the_budget(root, chat_tokenizer):
    from clarity_chat.cli import format_history

    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]\nbudget = 5')
    c.run("say", "Tell me about the wren", quiet)
    (found,) = history_found(c)
    assert found["left_out"] == {"budget": 1}
    lines = format_history(c.run("history", "", quiet)).splitlines()
    assert "  searched for wren: 1 matched, none added (didn't fit the budget)" in lines


# --- PS2: the run's different-parts notice names what differs


def test_the_run_notice_names_what_differs():
    from types import SimpleNamespace

    from clarity_chat.batch import parts_differ

    def p(repo="t", budget=None, placement="start"):
        profile = SimpleNamespace(budget=budget, search=("glossary",), placement=placement)
        return SimpleNamespace(profile=profile, model=SimpleNamespace(
            tokenizer_repo=repo, tokenizer_revision="r"))  # fmt: skip

    got = {"a": ["glossary:1"], "b": []}
    said = "Opening 1: the profiles got different parts (a: glossary:1; b: none)."
    assert parts_differ(1, got, {"a": p(), "b": p(budget=60)}) == (
        f"{said} Their budgets differ, so the replies differ in what they were given too."
    )
    assert parts_differ(1, got, {"a": p(), "b": p(repo="u", placement="end")}) == (
        f"{said} Their tokenizers and placements differ, so the replies differ in what they "
        "were given too."
    )
    assert parts_differ(1, got, {"a": p(), "b": p()}) == (
        f"{said} The replies differ in what they were given too."
    )
    assert parts_differ(1, {"a": ["g:1"], "b": ["g:1"]}, {"a": p(), "b": p()}) is None


# --- PS4: /try says when a message would be refused


def test_try_says_a_part_s_control_token_would_refuse_the_message(root, chat_tokenizer):
    from clarity_chat.cli import format_found

    session, c, _ = chat(root, chat_tokenizer, 'search = ["odd"]')
    result = c.run("try", "Tell me about the wren", quiet)
    assert set(result) == {"found", "refused"}
    assert [p["id"] for p in result["found"]["parts"]] == ["odd:1"]
    assert result["refused"] == TERMINAL["control"]
    page = Commands(session, uploads={})
    assert page.run("try", "Tell me about the wren", quiet)["refused"].endswith(
        "then add it again with the paperclip's Search it, under the same name. Or stop "
        "searching odd."
    )
    assert c.run("search", "glossary", quiet)
    assert c.run("try", "Tell me about the wren", quiet)["refused"] is None
    assert format_found(result["found"])


def test_document_try_says_it_too(root, chat_tokenizer):
    from clarity_chat.commands import document_try

    session = Session(root, backend=FakeBackend([" R."], tokenizer=chat_tokenizer),
                      tokenizer_loader=lambda r, v: chat_tokenizer)  # fmt: skip
    monkey = document_try.__globals__
    real = monkey["Session"]
    monkey["Session"] = lambda r: session
    try:
        result = document_try(root, ["odd"], "chat", "Tell me about the wren")
    finally:
        monkey["Session"] = real
    assert result.get("refused") == (
        'Part 1 of odd ("Wren") holds </s>, which this model reads as a control token, not '
        "text. Take it out of your file, then add it again: `clarity document add FILE --name "
        "odd`. Or leave odd out of the documents you try."
    )


# --- PS5: document try needs a chat-mode profile


def test_document_try_refuses_a_text_mode_profile(root, tiny_tokenizer):
    session = Session(root, backend=FakeBackend([" R."], tokenizer=tiny_tokenizer),
                      tokenizer_loader=lambda r, v: tiny_tokenizer)  # fmt: skip
    assert refusal(lambda: session.try_documents(["odd"], "text", "the wren")) == (
        "Searching needs a chat-mode profile; text is in text mode."
    )


# --- PC5: the stale-temp cleanup never takes another document's leftovers


def test_adding_a_name_ending_in_old_keeps_another_document_s_leftovers(root):
    library = root / "local/documents"
    library_add(root, b"## A\nMade-up.\n", "a.md")
    # a failed replace of `a` left its old folder aside, under the aside prefix
    aside = library / ".a.old-abcd1234"
    aside.mkdir()
    (aside / "mine.txt").write_text("Mine.\n", encoding="utf-8")
    library_add(root, b"## B\nMade-up.\n", "a-old.md")
    assert (aside / "mine.txt").is_file()


def test_replace_puts_the_old_folder_aside_under_a_prefix_no_name_makes(root, monkeypatch):
    from clarity_chat.documents import library as library_module

    seen = []
    real = library_module.os.replace

    def watch(a, b):
        seen.append(Path(b).name)
        return real(a, b)

    monkeypatch.setattr(library_module.os, "replace", watch)
    library_add(root, b"## New\nNew.\n", "glossary.md")
    assert seen[0].startswith(".glossary.old-"), seen


# --- PC8, PC9: format_found's left-out lines and an empty parenthesis


def test_format_found_groups_the_matches_not_considered():
    from clarity_chat.cli import format_found

    left = [{"id": f"g:{n}", "rank": n, "score": -1.0, "why": "not considered"}
            for n in range(9, 31)]  # fmt: skip
    found = {"words": ["wren"], "ignored": [], "matched": 30, "considered": 8, "tokens": 0,
             "budget": 1500,
             "parts": [], "left_out": [{"id": "g:2", "rank": 2, "score": -1.0, "why": "budget"},
                                       *left]}  # fmt: skip
    assert format_found(found).splitlines()[2:] == [
        "  2. g:2: left out (budget)",
        "  22 more matched and weren't considered: g:9 to g:30.",
    ]


def test_nothing_to_search_for_with_no_words_at_all():
    from clarity_chat.cli import format_found

    found = {"words": [], "ignored": [], "matched": 0}
    assert format_found(found) == (
        "Nothing to search for: your message holds only common words and single letters or digits."
    )
