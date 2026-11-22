"""v0.3.2's second fix-review pass (notes/research/reviews/v032-fix-review-2.md; the status
file's "fix review pass 2 rulings"): F2M1, F2S1-F2S3 and F2C1-F2C3. Made-up documents,
scratch roots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clarity_chat import cli
from clarity_chat.batch import RunError, run_batch
from clarity_chat.commands import Commands
from clarity_chat.documents import library_add, reading
from clarity_chat.session import SearchRefused, Session
from tests.fakes import FakeBackend
from tests.test_documents_board import chat, quiet
from tests.test_documents_security import retamper
from tests.test_part1_whole import PAGE_ONLY

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n"


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    library_add(fetched_root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    return fetched_root


def not_built(root: Path) -> None:
    """glossary's parts.jsonl changed, its manifest not: not the parts Clarity built."""
    parts = root / "local/documents/glossary/parts.jsonl"
    parts.write_bytes(parts.read_bytes().replace(b"small", b"large"))


def search(root: Path, names: list[str], extra: str = "", profile: str = "chat") -> None:
    path = root / "profiles" / f"{profile}.toml"
    text = path.read_text(encoding="utf-8")
    # Into the profile's [documents] table if it has one (conftest's chat states word search).
    head = "" if "\n[documents]\n" in text else '\n[documents]\nsearch_method = "words"'
    path.write_text(f"{text}{head}\nsearch = {json.dumps(names)}\n{extra}", encoding="utf-8")


def with_tokenizer(monkeypatch, tokenizer) -> None:
    """`clarity …` makes its own Session: give it the tests' tokenizer."""
    import clarity_chat.session as session_module

    real = session_module.Session.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("tokenizer_loader", lambda r, v: tokenizer)
        real(self, *args, **kwargs)

    monkeypatch.setattr(session_module.Session, "__init__", init)


# --- F2M1: a run's control token is refused before any model loads, in terminal words


def test_a_run_refuses_a_control_token_part_before_any_call(root, chat_tokenizer):
    search(root, ["odd"])
    (root / "openings.txt").write_text("Tell me about the wren.\n", encoding="utf-8")
    backend = FakeBackend([" R."] * 4, tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer,
                      logs_dir=root / "runs" / "r")  # fmt: skip
    try:
        run_batch(session, "r", root / "runs" / "r", root / "openings.txt", ["chat"], 1, print)
        refused = None
    except Exception as e:  # noqa: BLE001 - the test looks at what stopped it
        refused = e
    assert backend.calls == []  # refused before any model call
    assert isinstance(refused, SearchRefused), refused
    assert refused.terminal.startswith('Part 1 of odd ("Wren") holds </s>')
    assert "Or take odd out of the profile's search list." in refused.terminal


def test_a_mid_run_refusal_names_the_terminal_s_way_out(root, chat_tokenizer):
    search(root, ["glossary"])
    (root / "openings.txt").write_text("Tell me about the wren.\n", encoding="utf-8")
    backend = FakeBackend([" R."] * 4, tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer,
                      logs_dir=root / "runs" / "r")  # fmt: skip
    real = session.block

    def stale_then_block(text: str, where: str = "chat") -> dict:  # changed after the pre-check
        not_built(root)
        return real(text, where)

    session.block = stale_then_block  # type: ignore[method-assign]
    with pytest.raises(RunError) as stopped:
        run_batch(session, "r", root / "runs" / "r", root / "openings.txt", ["chat"], 1, print)
    said = str(stopped.value)
    assert "`clarity document add FILE --name glossary` adds the file again" in said, said
    assert not any(words in said for words in PAGE_ONLY), said


# --- F2S1: document try's own way out


def test_document_try_says_to_leave_a_document_out_of_those_you_try(
    root, chat_tokenizer, monkeypatch, capsys
):
    with_tokenizer(monkeypatch, chat_tokenizer)
    argv = ["--root", str(root), "document", "try"]
    assert cli.main([*argv, "odd", "--profile", "chat", "Tell me about the wren"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith(
        "Take it out of your file, then add it again: `clarity document add FILE --name odd`. Or "
        "leave odd out of the documents you try."
    ), out
    import shutil

    shutil.rmtree(root / "local/documents/glossary")
    assert cli.main([*argv, "glossary", "--profile", "chat", "the wren"]) == 1
    assert capsys.readouterr().err.strip() == (
        "clarity: Clarity can't search glossary: its folder in local/documents/ is gone. Leave "
        "it out of the documents you try, or add the file again: `clarity document add FILE "
        "--name glossary`."
    )


# --- F2S2: left-out counts count every match, not the 50 listed


@pytest.fixture
def many(root):
    text = "".join(f"## Wren {n}\nThe wren number {n}.\n\n" for n in range(100))
    library_add(root, text.encode(), "many.md")
    return root


def test_left_out_counts_every_match(many, chat_tokenizer):
    from clarity_chat.cli import format_found

    session, c, _ = chat(many, chat_tokenizer, 'search = ["many"]\nbudget = 100')
    found = c.run("try", "Tell me about the wren", quiet)["found"]
    assert found["matched"] == 100 and len(found["left_out"]) == 50  # the setup
    c.run("say", "Tell me about the wren", quiet)
    (turn,) = [t for t in c.run("history", "", quiet)["turns"] if "found" in t]
    added = len(turn["found"]["parts"])
    left = turn["found"]["left_out"]
    assert left["not considered"] == 92 and added + left.get("budget", 0) == 8
    last = format_found(found).splitlines()[-1]
    assert last == (
        f"  92 more matched and weren't considered: {found['left_out'][8]['id']} to "
        f"{found['left_out'][-1]['id']}, and 50 more."
    )


# --- F2S3: what the terminal prints, and the parts kind


def test_the_terminal_prints_a_profile_start_s_way_out(root, chat_tokenizer, monkeypatch, capsys):
    with_tokenizer(monkeypatch, chat_tokenizer)
    search(root, ["birds"])
    assert cli.main(["--root", str(root), "document", "try", "glossary", "--profile", "chat",
                     "wren"]) == 1  # fmt: skip
    assert capsys.readouterr().err.strip() == (
        "clarity: Profile chat searches birds, which isn't in your documents. Add it with "
        "`clarity document add FILE --name birds`, or take it out of search under [documents] "
        "in the profile's file."
    )


def test_the_terminal_prints_a_stale_try_s_way_out(root, chat_tokenizer, monkeypatch, capsys):
    with_tokenizer(monkeypatch, chat_tokenizer)
    not_built(root)
    assert cli.main(["--root", str(root), "document", "try", "glossary", "--profile", "chat",
                     "wren"]) == 1  # fmt: skip
    assert capsys.readouterr().err.strip() == (
        "clarity: Clarity can't search glossary: its parts.jsonl isn't the one Clarity built. "
        "`clarity document add FILE --name glossary` adds the file again."
    )


def test_the_terminal_names_add_again_for_a_parts_file_clarity_didn_t_write(root, chat_tokenizer):
    from clarity_chat.cli import ChatLoop

    search(root, ["glossary"])
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    retamper(root, "glossary", lambda p: p.update(page="1"))
    output.clear()
    loop.handle("Tell me about the wren")
    said = "".join(output)
    assert (  # the chat's own command (the PM1 ruling)
        "Clarity can't search glossary: its parts.jsonl isn't one Clarity writes. `/documents "
        "add PATH glossary` adds the file again." in said
    ), said
    assert not any(words in said for words in PAGE_ONLY), said


# --- F2C1: a parts.jsonl Clarity didn't write, at /search


def test_search_refuses_a_parts_file_clarity_didn_t_write_with_its_way_out(root, chat_tokenizer):
    retamper(root, "glossary", lambda p: p.update(page="1"))
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    c = Commands(session)
    c.run("profile", "chat", quiet)
    with pytest.raises(Exception) as refused:
        c.run("search", "glossary", quiet)
    assert getattr(refused.value, "terminal", None) == (  # the chat's own (the PM1 ruling)
        "Clarity can't search glossary: its parts.jsonl isn't one Clarity writes. `/documents "
        "add PATH glossary` adds the file again."
    )


# --- F2C2: the terminal's no-FTS5 way out


def test_the_terminal_s_no_fts5_names_search_none(root, chat_tokenizer, monkeypatch):
    from clarity_chat.cli import ChatLoop
    from tests.test_part1_whole import no_fts5

    search(root, ["glossary"])
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    no_fts5(monkeypatch)
    output.clear()
    loop.handle("Tell me about the wren")
    assert "whose SQLite includes it. `/search none` turns search off for this chat." in (
        "".join(output)
    )


# --- F2C3: the run notice compares effective budgets


def test_the_default_budget_written_out_isn_t_a_difference():
    from types import SimpleNamespace

    from clarity_chat.batch import parts_differ

    def p(budget):
        profile = SimpleNamespace(budget=budget, search=("g",), placement="start")
        return SimpleNamespace(profile=profile, model=SimpleNamespace(
            tokenizer_repo="t", tokenizer_revision="r"))  # fmt: skip

    said = parts_differ(1, {"a": ["g:1"], "b": []}, {"a": p(None), "b": p(1500)})
    assert said is not None and said.endswith(" The replies differ in what they were given too.")
