"""v0.3.2's board walk and fact-check, in code (notes/plans/v0.3.2-status.md: "board walk
rulings", "board redraw's open points", "board fact-check rulings"): WB1 (adding to the
search list checks only what's added), WB2 (an unreadable document, its own way out), point
6 (one refusal for every stale document), WB4
(the sentences, as V32Search-notes.md gives them) and WB6 (/documents names its folder).
Made-up documents, scratch roots."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from clarity_chat import cli
from clarity_chat.commands import Commands, check_lines, document_add
from clarity_chat.documents import (
    DocumentRefused,
    checked_library_name,
    extract,
    library_add,
    reading,
)
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.pdfs import encrypted_pdf, make_pdf

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n\n## Dipper\nIt walks under water.\n"


def quiet(event: dict) -> None:
    pass


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    library_add(fetched_root, b"## Lake\nCold water and a wren.\n", "notes.md")
    return fetched_root


def chat(root, tokenizer, documents: str = ""):
    if documents:
        path = root / "profiles" / "chat.toml"
        path.write_text(path.read_text(encoding="utf-8") + f"\n[documents]\n{documents}\n")
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    c = Commands(session)
    c.run("profile", "chat", quiet)
    return session, c, backend


def refusal(act) -> str:
    try:
        act()
    except (ConfigError, DocumentRefused) as e:
        return str(e)
    return "no refusal"


def unreadable(root: Path, name: str) -> None:
    (root / f"local/documents/{name}/manifest.json").write_text('{"schema": 1}', encoding="utf-8")


NOT_BUILT = (  # cut 1: parts that aren't the build, and the one way out
    "Clarity can't search {name}: its parts.jsonl isn't the one Clarity built. Add the file "
    "again in Documents…, with the same name."
)
MISSING = (
    "Clarity can't search {name}: its folder in local/documents/ is gone. Stop searching it, "
    "or add the file again in Documents…."
)
UNREADABLE = (  # the close batch (surface S2): the page's way out, as the log keeps it
    "Clarity can't search {name}: its manifest.json can't be read (it isn't a manifest Clarity "
    "writes). Add the file again in Documents…, with the same name."
)


# --- WB1: adding to the list checks only what's added


def test_adding_a_document_keeps_one_already_listed_whatever_its_state(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    shutil.rmtree(root / "local/documents/glossary")
    answer = c.run("search", "glossary notes", quiet)
    assert [d["name"] for d in answer["searching"]] == ["glossary", "notes"]
    assert answer["searching"][0]["state"] == "missing"
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said == "Nothing was sent.\n" + MISSING.format(name="glossary")
    c.run("search", "notes", quiet)
    c.run("say", "Tell me about the wren", quiet)


def test_a_document_being_added_must_be_there_and_readable(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    unreadable(root, "notes")
    assert refusal(lambda: c.run("search", "birds", quiet)) == "birds isn't in your documents."
    assert refusal(lambda: c.run("search", "notes", quiet)).startswith(  # N32 (close batch)
        "Clarity can't search notes: its manifest.json can't be read"
    )


# --- WB2: an unreadable document's sentence, and its own state at Send


def test_an_unreadable_manifest_says_delete_the_folder_then_add_again(root):
    unreadable(root, "notes")
    listed = cli.format_documents(cli.document_list(root)["documents"], "cli")
    assert listed.splitlines()[-1] == (  # `clarity document list`'s way out (close batch S2)
        "notes: notes's manifest.json can't be read (it isn't a manifest Clarity writes). "
        "`clarity document add FILE --name notes` adds the file again."
    )


def test_send_refuses_an_unreadable_document_as_unreadable(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["notes"]')
    unreadable(root, "notes")
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said == "Nothing was sent.\n" + UNREADABLE.format(name="notes")


# --- point 6: one refusal for every stale document


def test_one_refusal_names_every_stale_document(root, chat_tokenizer):
    library_add(root, b"## Heron\nTall.\n", "herons.md")
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary", "notes", "herons"]')
    parts = root / "local/documents/glossary/parts.jsonl"
    parts.write_bytes(parts.read_bytes().replace(b"small", b"large"))
    shutil.rmtree(root / "local/documents/notes")
    unreadable(root, "herons")
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said.split("\n") == [
        "Nothing was sent.",
        NOT_BUILT.format(name="glossary"),
        MISSING.format(name="notes"),
        UNREADABLE.format(name="herons"),
    ]


def test_a_control_token_in_an_added_part_names_it_and_the_way_out(root, chat_tokenizer):
    library_add(root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    session, c, _ = chat(root, chat_tokenizer, 'search = ["odd"]')
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said == (
        'Nothing was sent.\nPart 1 of odd ("Wren") holds </s>, which this model reads as a '
        "control token, not text. Take it out of your file, then add it again in Documents…, "
        "with the same name. Or stop searching odd."
    )


def test_a_control_token_in_a_part_left_out_doesn_t_refuse(root, chat_tokenizer):
    library_add(root, b"## Wren\nThe wren.\n\n## Wren song\nIt sings </s> loudly.\n", "odd.md")
    session, c, _ = chat(root, chat_tokenizer, 'search = ["odd"]\nbudget = 200')
    c.run("say", "Tell me about the wren", quiet)  # the second part doesn't fit: not sent
    conversation = session.conversation
    assert conversation is not None and conversation.head is not None
    user = conversation.turns[conversation.turns[conversation.head].parent_id or ""]
    assert user.found is not None
    assert [p["id"] for p in user.found["parts"]] == ["odd:1"]
    assert [o["id"] for o in user.found["left_out"]] == ["odd:2"]


# --- v0.3.1's reader sentences


# --- WB4: the sentences


@pytest.mark.parametrize(
    ("name", "data", "said"),
    [
        ("latin.md", "caf\xe9\n".encode("latin-1"),
         "latin.md isn't UTF-8 text. Save it as PDF or as UTF-8 text and use that copy."),
        ("scan.pdf", make_pdf(["", ""]),
         "scan.pdf has no text Clarity can read. Its pages may be scanned images, which Clarity "
         "doesn't read. Use a PDF app's text recognition (OCR) to add the text, then use that "
         "copy."),
        ("locked.pdf", encrypted_pdf(["Made-up."], "secret"),
         "locked.pdf needs a password. Save a copy without one and use that."),
    ],
)  # fmt: skip
def test_v0_3_1_s_reader_sentences_say_use_not_attach(name, data, said):
    reading._last = None
    assert refusal(lambda: extract(data, name)) == said


def test_the_shared_size_sentence_says_use():
    assert reading.SMALLER == "Use a smaller file, or split it and use the part you need."


def test_a_searching_chat_s_sheet_and_check_fold_the_budget_in(root, chat_tokenizer, tmp_path):
    from clarity_chat.documents import SENTENCES

    session, c, backend = chat(root, chat_tokenizer, 'search = ["glossary"]')
    backend._n_ctx = 2000
    path = tmp_path / "long.md"
    path.write_text("".join(f"Made-up line {n}.\n" for n in range(400)), encoding="utf-8")
    check = c.run("attach", f"{path} check", quiet)["check"]
    assert (check["aside"], check["aside_origin"]) == (1500, "default")
    assert "aside" not in SENTENCES
    lines = check_lines(check)
    assert any(
        line.startswith("The largest start found that leaves at least 16 tokens for a reply and "
                        "1,500 for the parts search can add to your next message (Clarity's "
                        "default budget): lines 1 to ")
        for line in lines
    )  # fmt: skip
    assert SENTENCES["filled_in_search"] == (
        "Clarity filled in {units} 1 to {last}: the largest start it found that leaves at "
        "least {max_tokens} for a reply and {budget} for the parts search can add to your next "
        "message ({origin})."
    )
    assert SENTENCES["left_search"] == (
        "That leaves {left} tokens for your messages, the parts search adds, and the replies."
    )


def test_the_name_box_sentence_has_no_rule(root, tmp_path):
    path = tmp_path / "日本語.md"
    path.write_bytes(b"Made-up.\n")
    assert refusal(lambda: document_add(root, str(path))) == (
        "Clarity can't make a document name from 日本語.md. Give it one."
    )


def test_the_terminal_s_added_notice_leads_with_what_was_written(root, tmp_path, capsys):
    path = tmp_path / "Herons.md"
    path.write_bytes(b"## Heron\nTall.\n\n## Egret\nWhite.\n")
    assert cli.main(["--root", str(root), "document", "add", str(path)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Clarity wrote local/documents/herons/: original.txt, text.md, parts.jsonl and "
        "manifest.json (2 parts, 2 with a title)."
    )


def test_a_profile_naming_a_missing_document_in_the_page_s_terms(root, chat_tokenizer):
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["birds"]\n')
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    assert refusal(lambda: Commands(session).run("profile", "chat", quiet)) == (
        "Profile chat searches birds, which isn't in your documents. Add it in Documents…, or "
        "take it out of search under [documents] in the profile's file."
    )


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("con", "con can't name a document: Windows reserves it, and Clarity keeps names that "
         "work everywhere."),
        ("none", "none can't name a document: Clarity keeps it for turning search off."),
    ],
)  # fmt: skip
def test_the_reserved_names_sentences(name, said):
    assert refusal(lambda: checked_library_name(name)) == said


def test_no_fts5_at_send_says_how_to_send_without_searching(root, chat_tokenizer, monkeypatch):
    import sqlite3

    from clarity_chat.documents import search as search_module

    class NoFts5:
        def execute(self, *args):
            raise sqlite3.OperationalError("no such module: fts5")

    search_module._indexes.clear()
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    monkeypatch.setattr(search_module.sqlite3, "connect", lambda *a, **k: NoFts5())
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said.startswith("Nothing was sent.\nWord search needs SQLite's FTS5")
    assert said.endswith("To send without searching, stop searching every document in this chat.")


def test_one_sentence_for_an_upload_that_s_gone(root, chat_tokenizer):
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    page = Commands(session, uploads={})
    page.run("profile", "chat", quiet)
    assert refusal(lambda: page.run("documents", "add u-1", quiet)) == (
        "That file isn't uploaded any more. Choose it again."
    )


def test_search_in_a_text_mode_chat_has_its_own_sentence(root, tiny_tokenizer):
    backend = FakeBackend([" R."], tokenizer=tiny_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", quiet)
    assert refusal(lambda: c.run("search", "glossary", quiet)) == (
        "Searching needs a chat-mode profile; text is in text mode."
    )


def test_nothing_to_search_for_names_single_letters_or_digits():
    from clarity_chat.cli import format_found

    found = {"words": [], "ignored": ["what", "s", "it"], "matched": 0}
    assert format_found(found) == (
        "Nothing to search for: your message holds only common words and single letters or "
        "digits (what, s, it)."
    )


# --- WB6: /documents names the library's folder


def test_documents_gives_the_library_s_folder(root, capsys):
    from clarity_chat.commands import document_list

    assert document_list(root)["folder"] == str((root / "local" / "documents").resolve())
    assert cli.main(["--root", str(root), "document", "list"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        f"Your documents, in {(root / 'local' / 'documents').resolve()}:"
    )
    assert cli.main(["--root", str(root), "document", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["folder"].endswith("documents")


# --- CM1 (Checkpoint 2's must, with WB2): every refusal at Send takes the logged path


def no_fts5(monkeypatch) -> None:
    import sqlite3

    from clarity_chat.documents import search as search_module

    class NoFts5:
        def execute(self, *args):
            raise sqlite3.OperationalError("no such module: fts5")

    search_module._indexes.clear()
    monkeypatch.setattr(search_module.sqlite3, "connect", lambda *a, **k: NoFts5())


@pytest.mark.parametrize("how", ["manifest", "link", "no_fts5"])
def test_a_refusal_at_send_is_logged_and_given_back(
    root, chat_tokenizer, tmp_path, monkeypatch, how
):
    from clarity_chat.cli import ChatLoop
    from clarity_chat.logs import read_records

    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["notes"]\n')
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    if how == "manifest":
        unreadable(root, "notes")
    elif how == "link":
        if os.name == "nt":
            pytest.skip("symlinks need privileges on Windows")
        folder = root / "local/documents/notes"
        shutil.move(folder, tmp_path / "elsewhere")
        folder.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    else:
        no_fts5(monkeypatch)
    loop.handle("Tell me about the wren")
    assert loop.draft == "Tell me about the wren"  # the terminal gives it back
    assert "Nothing was sent." in "".join(output)
    conversation = session.conversation
    assert conversation is not None and conversation.unanswered is not None
    records = list(read_records(conversation.log.path))
    (user,) = [r for r in records if r["type"] == "turn" and r["role"] == "user"]
    (call,) = [r for r in records if r["type"] == "call"]
    assert call["error"].startswith("SearchRefused: Nothing was sent.\n"), call["error"]
    assert call["prompt_ids"] and call.get("turn_id") is None and user["text"] == loop.draft
    assert backend.calls == []


def test_the_terminal_s_unnamed_file_says_how_to_give_a_name(root, tmp_path, capsys):
    path = tmp_path / "日本語.md"
    path.write_bytes(b"Made-up.\n")
    assert cli.main(["--root", str(root), "document", "add", str(path)]) == 1
    assert capsys.readouterr().err.strip().splitlines()[-1] == (
        "clarity: Clarity can't make a document name from 日本語.md. Give it one with `--name "
        "NAME`. A document's name is 1 to 64 characters: a to z, 0 to 9, - and _, "
        "starting with a letter or a digit."
    )


# --- WB7 (WB5's finding, ruled): a web page gets paragraphs and titled parts


def test_a_web_page_gets_paragraphs_and_titled_parts():
    from clarity_chat.documents import cut_parts
    from clarity_chat.documents.child import html_text

    page = (
        b"<html><body><h1>Birds</h1><p>The wren is small.</p><p>The dipper walks under "
        b"water.</p><h2>Herons</h2><p>Tall.</p></body></html>"
    )
    text = html_text(page, "p.html")
    assert text == (
        "# Birds\n\nThe wren is small.\n\nThe dipper walks under water.\n\n## Herons\n\nTall.\n"
    )
    assert [(p["title"], p["words"]) for p in cut_parts(text, "p.html")] == [
        ("Birds", 11),
        ("Herons", 3),
    ]


# --- the in-chat way to name a file


def test_in_a_chat_an_unnamable_file_says_how_to_name_it(root, chat_tokenizer, tmp_path):
    path = tmp_path / "日本語.md"
    path.write_bytes(b"Made-up.\n")
    session, c, _ = chat(root, chat_tokenizer)
    assert refusal(lambda: c.run("documents", f"add {path}", quiet)) == (
        "Clarity can't make a document name from 日本語.md. Give it one: /documents add PATH NAME."
    )
