"""The main session's PM1 ruling: in the terminal chat and --json, every refusal's way out
is a chat command (`/documents add PATH NAME`);
`clarity document …` stays for the command line. Made-up documents, scratch roots."""

from __future__ import annotations

import shutil

import pytest

from clarity_chat.cli import ChatLoop
from clarity_chat.commands import Commands
from clarity_chat.session import SearchRefused, Session
from tests.fakes import FakeBackend
from tests.test_search_b4 import chat, quiet, root  # noqa: F401


def terminal_of(act) -> str:
    try:
        act()
    except SearchRefused as e:
        return e.terminal
    return "no refusal"


def test_a_parts_refusal_names_the_chat_s_add(root, chat_tokenizer):  # noqa: F811
    from clarity_chat.documents.refused import problem_words

    problem = {"document": "glossary", "state": "unreadable", "add_again": True,
               "why": "glossary's parts.jsonl isn't the one Clarity built."}  # fmt: skip
    said = problem_words(problem, terminal=True, where="chat", at_send=True)
    assert "`/documents add PATH glossary`" in said and "clarity document" not in said


def test_the_command_line_keeps_its_own_commands(root, chat_tokenizer):  # noqa: F811
    from clarity_chat.documents.refused import problem_words

    for where in ("run", "try"):
        said = problem_words({"document": "g", "state": "missing"}, True, where, True)
        assert "`clarity document add FILE --name g`" in said, (where, said)


def test_the_chat_s_search_list_names_chat_commands(root, chat_tokenizer):  # noqa: F811
    session, _ = chat(root, chat_tokenizer)
    shutil.rmtree(root / "local/documents/glossary")
    output: list[str] = []
    loop = ChatLoop(session, "chat", read=lambda d: "", write=output.append)
    loop.handle("/search")
    said = "".join(output)
    assert "/search without it" in said and "clarity document" not in said, said


def test_the_chat_s_empty_library_names_the_chat_s_add(root, chat_tokenizer):  # noqa: F811
    session, _ = chat(root, chat_tokenizer)
    shutil.rmtree(root / "local/documents/glossary")
    output: list[str] = []
    loop = ChatLoop(session, "chat", read=lambda d: "", write=output.append)
    loop.handle("/documents")
    assert "No documents yet. `/documents add PATH` adds one." in "".join(output)


def test_a_chat_start_with_a_profile_s_missing_document_names_the_chat_s_add(  # noqa: F811
    root,  # noqa: F811
    chat_tokenizer,
):
    shutil.rmtree(root / "local/documents/glossary")
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    said = terminal_of(lambda: Commands(session).run("profile", "chat", quiet))
    assert "`/documents add PATH glossary`" in said and "clarity document" not in said, said


def test_document_try_with_a_profile_s_missing_document_names_the_cli_s_add(  # noqa: F811
    root,  # noqa: F811
    chat_tokenizer,
):
    shutil.rmtree(root / "local/documents/glossary")
    session = Session(root, tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(SearchRefused) as e:
        session.try_documents([], "chat", "the wren")  # the profile names glossary
    assert "`clarity document add FILE --name glossary`" in e.value.terminal, e.value.terminal
