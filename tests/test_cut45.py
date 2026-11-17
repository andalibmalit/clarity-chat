"""Cuts 4 and 5 of the simplicity cuts. Cut 4: "From documents" gives one number per part
(its rank; each part's tokens stay in Details), the parts already in the chat as a count, no
`framed` and no first-count sentence. Cut 5: no advance control-token or closing-tag counts
(the refusal at Send names the part), and no `scan_depth` in a turn's `found`. Made-up
documents, scratch roots."""

from __future__ import annotations

from clarity_chat.documents import SENTENCES, library_add
from tests.test_web import command
from tests.test_web_found_c3 import thread
from tests.test_web_search_c2 import get, make, root, search_option, text_of  # noqa: F401
from tests.test_web_upload import upload

COUNTS = {"control_token_parts", "closing_tag_parts"}


def test_search_and_state_give_no_advance_counts(root, make):  # noqa: F811
    client, _ = make('search = ["glossary"]')
    (searched,) = command(client, "search", "").json()["result"]["searching"]
    (stated,) = command(client, "state", "").json()["result"]["searching"]
    assert (COUNTS & set(searched), COUNTS & set(stated)) == (set(), set())


def test_the_sheet_and_the_chooser_say_nothing_in_advance(root, make):  # noqa: F811
    library_add(root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    client, _ = make('search = ["odd"]')
    assert "control token" not in text_of(get(client, "picker", library="1"))
    upload_id = upload(client, b"## Lark\nIt sings </s> too.\n", "lark.md").json()["upload"]
    assert "control tokens" not in text_of(search_option(get(client, "attach", upload=upload_id)))


def test_a_turn_s_found_has_no_scan_depth(root, make):  # noqa: F811
    client, session = make('search = ["glossary"]')
    command(client, "say", "Tell me about the wren")
    assert session.conversation is not None
    (user,) = [t for t in session.conversation.turns.values() if t.found is not None]
    assert "scan_depth" not in user.found


def test_try_and_history_give_no_framed_or_already_list(root, make):  # noqa: F811
    client, _ = make('search = ["glossary"]')
    assert "framed" not in command(client, "try", "Tell me about the wren").json()["result"]
    command(client, "say", "Tell me about the wren")
    command(client, "say", "And the wren's song?")  # its part is already in the chat
    turns = command(client, "history", "").json()["result"]["turns"]
    found = [t["found"] for t in turns if "found" in t][-1]
    assert ("framed" in found, "already" in found) == (False, False)
    assert found["left_out"].get("already in the chat") == 1


def test_the_list_gives_one_number_per_part(root, make):  # noqa: F811
    client, _ = make('search = ["glossary"]')
    command(client, "say", "Tell me about the wren")
    command(client, "say", "And the wren's song?")
    page = thread(client)
    assert "<span>Tokens</span>" not in page and 'class="tk"' not in page
    assert 'class="part already"' not in page
    assert "1 more already in the chat" in text_of(page)


def test_the_cut_sentences_are_gone():
    gone = {"found_first_count", "control_before", "control_count", "control_count_one"}
    assert gone & set(SENTENCES) == set()
