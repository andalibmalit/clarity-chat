"""Q2 (the owner's, v0.3.2): an × on Edit's "Keeps" line takes an attachment out of the
edited message: `/edit --drop N TEXT`, N the attachment's place in the message (1 to n).
The edited turn holds fewer attachments; no record field changes. Made-up documents only."""

from __future__ import annotations

import pytest

from tests.test_attach_c2 import make, sent  # noqa: F401
from tests.test_web import command
from tests.test_web_upload import upload

OTHER = b"Other one.\nOther two.\n"


def attach(client, data: bytes, name: str) -> None:
    upload_id = upload(client, data, name).json()["upload"]
    assert "error" not in command(client, "attach", upload_id).json()


def edited(client) -> dict:
    turns = command(client, "history").json()["result"]["turns"]
    return [t for t in turns if t["role"] == "user"][-1]


def test_drop_takes_an_attachment_out_of_the_edit(make):  # noqa: F811
    client, _, _ = make()
    attach(client, OTHER, "other.md")
    sent(client)  # other.md, then notes.md
    answer = command(client, "edit", "--drop 1 What does it say?").json()
    assert "error" not in answer, answer
    turn = edited(client)
    assert turn["text"] == "What does it say?"
    assert [a["name"] for a in turn.get("attachments", [])] == ["notes.md"]
    prompt = command(client, "prompt").json()["result"]["call"]["prompt_text"]
    assert "Other one." not in prompt and "Line one." in prompt


def test_drop_every_attachment(make):  # noqa: F811
    client, _, _ = make()
    sent(client)
    assert "error" not in command(client, "edit", "--drop 1 Plain now").json()
    assert "attachments" not in edited(client)
    prompt = command(client, "prompt").json()["result"]["call"]["prompt_text"]
    assert "Line one." not in prompt


def test_drop_out_of_range_is_refused_and_nothing_is_sent(make):  # noqa: F811
    client, _, backend = make()
    sent(client)
    calls = len(backend.calls)
    answer = command(client, "edit", "--drop 2 What does it say?").json()
    assert answer.get("error") == (
        "--drop 2: this message has 1 attachment. Use a number from 1 to 1."
    )
    assert len(backend.calls) == calls


def test_drop_on_a_message_without_attachments_is_refused(make):  # noqa: F811
    client, _, _ = make()
    command(client, "say", "Hello")
    answer = command(client, "edit", "--drop 1 Hi").json()
    assert answer.get("error") == "--drop 1: this message has no attachments."


def test_edit_text_that_starts_with_two_dashes_is_kept(make):  # noqa: F811
    client, _, _ = make()
    command(client, "say", "Hello")
    assert "error" not in command(client, "edit", "--dropped hints").json()
    assert edited(client)["text"] == "--dropped hints"


# --- the page's × (the board's ⑥, N73-N76) and the terminal's prefill


def edit_form(client) -> str:
    import re

    from tests.test_attach_c2 import shown

    form = re.search(r'<form class="editbox".*?</form>', shown(client, "thread"), re.S)
    assert form, "no edit form"
    return form.group(0)


def test_the_keeps_sentence_says_an_edit_can_t_add():  # N73
    from clarity_chat.documents import SENTENCES

    assert SENTENCES["keeps"] == "Keeps: {names}. An edit can't add attachments."


def test_edit_s_kept_attachments_are_chips_with_an_x(make):  # noqa: F811
    from tests.test_web_search_c2 import text_of  # tags read as spaces

    client, _, _ = make()
    attach(client, OTHER, "other.md")
    sent(client)
    form = edit_form(client)
    assert 'role="group" aria-label="Attachments this edit keeps"' in form
    assert (
        'data-action="edit-drop" data-n="1" aria-label="Remove other.md from the edited message"'
        in form
    )
    assert (
        'data-action="edit-drop" data-n="2" aria-label="Remove notes.md from the edited message"'
        in form
    )
    said = " ".join(text_of(form).split())
    assert "Keeps: other.md · whole · " in said and "An edit can't add attachments." in said
    assert 'data-none="Keeps no attachments."' in form


def test_the_terminal_s_edit_drop_prefills_your_message(root_and_tokenizer):
    from tests.test_attach_close import chat_loop

    root, tokenizer, notes = root_and_tokenizer
    loop, output, _ = chat_loop(root, tokenizer)
    asked = []
    loop.read = lambda default: asked.append(default) or "Plain now"
    loop.start()
    loop.handle(f"/attach {notes}")
    loop.handle("Summarize")
    loop.handle("/edit --drop 1")
    assert asked == ["Summarize"]
    conversation = loop.session.conversation
    users = [t for t in conversation.path_to(conversation.head) if t.role == "user"]
    assert (users[-1].text, users[-1].attachments) == ("Plain now", [])


@pytest.fixture
def root_and_tokenizer(fetched_root, chat_tokenizer, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_bytes(b"Line one.\n")
    return fetched_root, chat_tokenizer, notes
