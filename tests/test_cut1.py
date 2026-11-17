"""Cut 1 of the simplicity cuts: the library holds nothing of yours. A document is ready
when its manifest reads and its parts.jsonl is the one Clarity built; otherwise it can't be
searched, and the way out is to add the file again (or stop searching it). Adding under a
name that exists replaces it, said before and after. No rebuild, no check, no --replace,
no same-bytes rule. Made-up documents, scratch roots."""

from __future__ import annotations

from clarity_chat import cli
from clarity_chat.documents import library_add, library_documents, library_state
from clarity_chat.profiles import ConfigError
from clarity_chat.session import SearchRefused
from tests.test_search_b4 import GLOSSARY, chat, quiet, root  # noqa: F401


def exit_code(argv: list[str]) -> int:
    try:
        return cli.main(argv)
    except SystemExit as e:
        return int(e.code or 0)


def refusal(act) -> str:
    try:
        act()
    except ConfigError as e:
        return str(e)
    return "no refusal"


def test_adding_a_taken_name_replaces_it(root):  # noqa: F811
    from clarity_chat.documents import DocumentRefused

    try:
        result = library_add(root, b"## Owl\nA made-up owl.\n", "glossary.md")
    except DocumentRefused as e:  # today's name-taken refusal
        result = {"refused": str(e)}
    assert result.get("replaced") is True, result
    (doc,) = library_documents(root)
    assert doc["original"] == "glossary.md" and doc["parts"] == 1


def test_adding_the_same_file_again_writes_it_again(root):  # noqa: F811
    result = library_add(root, GLOSSARY, "copy.md")
    assert result.get("added", {}).get("name") == "copy", result


def test_an_edited_text_md_changes_nothing(root, chat_tokenizer):  # noqa: F811
    text = root / "local/documents/glossary/text.md"
    text.write_text("## Edited\nMade-up.\n", encoding="utf-8")
    assert library_state(root, "glossary") == "ready"
    _, c = chat(root, chat_tokenizer)
    sent = c.run("say", "Tell me about the wren", quiet)["call"]
    assert "The wren is a small brown bird" in sent["prompt_text"]


def test_a_tampered_parts_file_is_not_ready_and_says_add_it_again(root, chat_tokenizer):  # noqa: F811
    parts = root / "local/documents/glossary/parts.jsonl"
    parts.write_bytes(parts.read_bytes().replace(b"small", b"large"))
    _, c = chat(root, chat_tokenizer)
    try:
        c.run("say", "Tell me about the wren", quiet)
        said = "sent"
    except SearchRefused as e:
        said = e.terminal
    assert said.startswith("Nothing was sent.\nClarity can't search glossary:"), said
    assert said.endswith("`/documents add PATH glossary` adds the file again."), said


def test_there_is_no_rebuild_or_check(root, chat_tokenizer):  # noqa: F811
    _, c = chat(root, chat_tokenizer)
    assert refusal(lambda: c.run("documents", "rebuild glossary", quiet)).startswith("Usage:")
    assert exit_code(["--root", str(root), "document", "rebuild", "glossary"]) == 2
    assert exit_code(["--root", str(root), "document", "check"]) == 2


def test_there_is_no_replace_flag(root, chat_tokenizer, tmp_path):  # noqa: F811
    path = tmp_path / "owls.md"
    path.write_bytes(b"## Owl\nA made-up owl.\n")
    _, c = chat(root, chat_tokenizer)
    said = refusal(lambda: c.run("documents", f"add --replace {path} glossary", quiet))
    assert said != "no refusal", said
    assert exit_code(["--root", str(root), "document", "add", str(path), "--replace"]) == 2
