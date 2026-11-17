"""v0.3.2's fix review (notes/research/reviews/v032-fix-review.md; the status file's "fix
review rulings"): FS1-FS4 and the coulds FC1-FC3, FC5-FC7, FC9 and FC10. Made-up documents
and logs, scratch roots."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from clarity_chat.cli import format_history
from clarity_chat.commands import Commands
from clarity_chat.documents import library as library_module
from clarity_chat.documents import (
    library_add,
    library_parts,
    library_state,
    reading,
)
from clarity_chat.export import export_log
from clarity_chat.logs import read_records
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_documents_board import chat, quiet, refusal
from tests.test_documents_security import retamper

GLOSSARY = b"## Wren\nThe wren is a small brown bird.\n"
GOLDEN_HTML = Path(__file__).parent / "fixtures" / "golden" / "v0.3.1-html.jsonl"


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def folder(root: Path, name: str = "glossary") -> Path:
    return root / "local/documents" / name


# --- FS1: placement end counts the block where it goes


@pytest.mark.parametrize("message", ["Tell me about the wren ", "Tell me about the wren.\n"])
def test_a_block_at_the_end_is_counted_at_the_end(root, olmo_tokenizer, message):
    """With OLMo 3's tokenizer a message ending in whitespace joins the block's blank line
    at the end, so the block costs a token less there than at the start (CP1 mutation #3)."""

    def tokens(placement: str, budget: int = 1500) -> dict:
        session, c, _ = chat(root, olmo_tokenizer)
        path = root / "profiles" / "chat.toml"
        base = path.read_text(encoding="utf-8").split("\n[documents]")[0]
        documents = f'search = ["glossary"]\nplacement = "{placement}"\nbudget = {budget}'
        path.write_text(f"{base}\n[documents]\n{documents}\n", encoding="utf-8")
        c.run("profile", "chat", quiet)
        c.run("say", message, quiet)
        assert session.conversation is not None
        (user,) = [t for t in session.conversation.turns.values() if t.found is not None]
        return user.found  # type: ignore[return-value]

    start, end = tokens("start")["tokens"], tokens("end")["tokens"]
    assert end == start - 1
    fitted = tokens("end", budget=end)
    assert [p["id"] for p in fitted["parts"]] == ["glossary:1"] and fitted["tokens"] == end


# --- FS2: the control-token refusal's title through shown()


def test_the_control_token_refusal_shows_a_title_s_escapes(root, chat_tokenizer):
    heading = "\x1b]0;OWNED\x07Wren \x1b[31mred"
    library_add(root, f"## {heading}\nThe wren </s> is a bird.\n".encode(), "odd.md")
    session, c, _ = chat(root, chat_tokenizer, 'search = ["odd"]')
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert "\x1b" not in said and "\x07" not in said
    assert '("\\x1b]0;OWNED\\x07Wren \\x1b[31mred")' in said


# --- FS3: /history's reason when nothing was added


def test_history_says_already_in_the_chat(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    c.run("say", "The wren again", quiet)
    lines = format_history(c.run("history", "", quiet)).splitlines()
    assert "  searched for wren: 1 matched, none added (already in the chat)" in lines


# --- FS4: the read bound


def test_read_file_reads_at_most_its_limit_and_one(tmp_path):
    path = tmp_path / "big.txt"
    path.write_bytes(b"x" * 1000)
    assert len(library_module.read_file("big", path, 10)) == 11


# --- FC1: an edit's refusal heads "Nothing was sent."


def test_an_edit_refused_by_search_says_nothing_was_sent(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    rebuilt_elsewhere(root)
    said = refusal(lambda: c.run("edit", "Tell me more about the wren", quiet))
    assert said.startswith("Nothing was sent.\nClarity can't search glossary: ")
    assert session.conversation is not None
    calls = [r for r in read_records(session.conversation.log.path) if r["type"] == "call"]
    assert calls[-1]["error"].startswith("SearchRefused: Nothing was sent.")


# --- FC2: /search's counts follow the model's tokenizer


# --- FC3: lstat, and a parts file's last newline


def test_a_parts_file_without_its_last_newline_is_refused(root):
    path = folder(root) / "parts.jsonl"
    retamper(root, "glossary", lambda p: None)
    data = path.read_bytes().removesuffix(b"\n")
    path.write_bytes(data)
    manifest = json.loads((folder(root) / "manifest.json").read_bytes())
    import hashlib

    manifest["build_id"] = hashlib.sha256(data).hexdigest()
    (folder(root) / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert refusal(lambda: library_parts(root, "glossary")) == (
        "glossary's parts.jsonl isn't one Clarity writes. Add the file again with the "
        "paperclip's Search it, under the same name."
    )


# --- FC5 (reworked by part 1's PM1, tests in test_part1_whole.py): the page keeps its words


def stale(root: Path, how: str) -> str:
    """Make the searched document `how`, and return its name."""
    if how == "changed":
        rebuilt_elsewhere(root)
    elif how == "missing":
        shutil.rmtree(folder(root))
    else:
        library_add(root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
        return "odd"
    return "glossary"


def test_the_page_keeps_the_board_s_words(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    stale(root, "changed")
    said = refusal(lambda: c.run("say", "Tell me about the wren", quiet))
    assert said.endswith("Add the file again with the paperclip's Search it, under the same name.")
    assert session.conversation is not None
    calls = [r for r in read_records(session.conversation.log.path) if r["type"] == "call"]
    assert calls[-1]["error"] == f"SearchRefused: {said}"


# --- FC6: parts that aren't the build are unreadable; FC7: an unreadable entry says why


def rebuilt_elsewhere(root: Path) -> None:
    """parts.jsonl's bytes changed, its size and time kept: the manifest still reads."""
    path = folder(root) / "parts.jsonl"
    info = path.stat()
    data = path.read_bytes()
    path.write_bytes(data.replace(b"small", b"SMALL"))
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


def test_a_parts_file_changed_under_the_quick_check_reads_unreadable(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    rebuilt_elsewhere(root)
    assert library_state(root, "glossary") == "ready"  # the setup: size and time kept
    (entry,) = c.run("search", "", quiet)["searching"]
    assert entry["state"] == "unreadable" and entry["add_again"] is True


def test_adding_a_document_whose_parts_changed_says_so(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    rebuilt_elsewhere(root)
    assert refusal(lambda: c.run("search", "glossary", quiet)) == (
        "Clarity can't search glossary: its parts.jsonl isn't the one Clarity built. Add the "
        "file again with the paperclip's Search it, under the same name."
    )


def test_an_unreadable_entry_says_why(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    (folder(root) / "manifest.json").write_text('{"schema": 1}', encoding="utf-8")
    (entry,) = c.run("search", "", quiet)["searching"]
    assert entry == {
        "name": "glossary",
        "state": "unreadable",
        # The reason only: each front end adds its way out (the close batch: surface S2).
        "why": "glossary's manifest.json can't be read (it isn't a manifest Clarity writes).",
        "add_again": True,
    }


# --- FC9: a second failed rename names where the document is


def test_when_putting_the_old_folder_back_fails_too_its_folder_is_named(root, monkeypatch):
    real = os.replace
    calls = []

    def both_fail(a, b):
        calls.append((a, b))
        if len(calls) >= 2:
            raise PermissionError(13, "Permission denied")
        return real(a, b)

    monkeypatch.setattr(library_module.os, "replace", both_fail)
    said = refusal(lambda: library_add(root, b"# New\n\nNew.\n", "glossary.md"))
    monkeypatch.setattr(library_module.os, "replace", real)
    hidden = calls[0][1].name
    assert said == (
        f"Clarity couldn't replace glossary or put it back (Permission denied). It's in "
        f"local/documents/{hidden}/."
    )
    assert (root / "local/documents" / hidden / "text.md").is_file()


# --- FC10: v0.3.1's HTML-attachment log, written by v0.3.1's code


def test_a_v0_3_1_html_log_reads_resumes_exports_and_edits(root, chat_tokenizer, tmp_path):
    log = Path(shutil.copy(GOLDEN_HTML, tmp_path / GOLDEN_HTML.name))
    rows = list(read_records(log))
    (sent,) = [r for r in rows if r["type"] == "turn" and r.get("attachments")]
    attachment = sent["attachments"][0]
    assert attachment["extractor"]["version"] == 1  # v0.3.1's rules, one line a block
    assert attachment["text"].startswith("Birds\nThe wren is small.\n")
    backend = FakeBackend([" More.", " Edited."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    export_log(session, log, tmp_path / "x.jsonl")
    resumed = session.resume(log)
    assert resumed["render"]["identical"] is True
    c = Commands(session)
    assert c.run("history", "", quiet)["turns"]
    assert c.run("state", "", quiet)["searching"] == []
    conversation = session.conversation
    assert conversation is not None
    reply = next(t for t in conversation.turns.values() if t.parent_id == sent["turn_id"])
    conversation.head = reply.turn_id
    session.edit("What about the dipper?")
    edited = [r for r in read_records(log) if r["type"] == "turn" and r.get("origin") == "edit"]
    assert edited[-1]["attachments"][0]["text"] == attachment["text"]
    assert (edited[-1]["framing"], edited[-1]["placement"]) == (sent["framing"], "start")
    assert "found" not in edited[-1]
