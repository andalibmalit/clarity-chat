"""C4 (v0.3.2, Q1 answered (b)): the framing line as a chat value, like /set. `/framing TEXT`
sets this chat's line, `/framing none` sends none, `/framing` goes back to the profile's;
each is logged as a `framing` record and read back on resume, and every later message,
and the attach sheet's counts, use it (the board's ⑦). Made-up documents only."""

from __future__ import annotations

from clarity_chat.documents import FRAMING
from tests.test_web import command
from tests.test_web_search_c2 import make, root  # noqa: F401
from tests.test_web_upload import upload

SEARCHES = 'search = ["glossary"]'
OURS = "Notes from my own hikes. Use them for facts; answer in your own words."


def last_message(client) -> dict:
    turns = command(client, "history").json()["result"]["turns"]
    return [t for t in turns if t["role"] == "user"][-1]


def prompt(client) -> str:
    return command(client, "prompt").json()["result"]["call"]["prompt_text"]


def test_framing_sets_this_chat_s_line_for_every_later_message(root, make):  # noqa: F811
    client, session = make(SEARCHES)
    answer = command(client, "framing", OURS).json()
    assert "error" not in answer, answer
    command(client, "say", "Tell me about the wren")
    assert OURS in prompt(client) and FRAMING not in prompt(client)
    turn = session.conversation.turns[last_message(client)["turn_id"]]
    assert (turn.framing, turn.framing_origin) == (OURS, "chat")
    state = command(client, "state").json()["result"]
    assert (state.get("framing"), state.get("framing_origin")) == (OURS, "chat")


def test_framing_none_sends_no_line(root, make):  # noqa: F811
    client, session = make(SEARCHES)
    command(client, "framing", "none")
    command(client, "say", "Tell me about the wren")
    assert FRAMING not in prompt(client)
    turn = session.conversation.turns[last_message(client)["turn_id"]]
    assert (turn.framing, turn.framing_origin) == ("", "chat")


def test_framing_alone_goes_back_to_the_profile_s_line(root, make):  # noqa: F811
    client, session = make(SEARCHES)
    command(client, "framing", OURS)
    command(client, "framing", "")
    command(client, "say", "Tell me about the wren")
    turn = session.conversation.turns[last_message(client)["turn_id"]]
    assert (turn.framing, turn.framing_origin) == (FRAMING, "default")
    state = command(client, "state").json()["result"]
    assert state.get("framing_origin") == "default"


def test_framing_is_logged_and_kept_on_resume(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "framing", OURS)
    command(client, "say", "Tell me about the wren")
    log_id = command(client, "state").json()["result"]["log_id"]
    command(client, "new", "")
    answer = command(client, "resume", log_id).json()
    assert "error" not in answer, answer
    state = command(client, "state").json()["result"]
    assert (state.get("framing"), state.get("framing_origin")) == (OURS, "chat")
    command(client, "say", "And the dipper?")
    assert OURS in prompt(client)


def test_the_attach_check_counts_with_this_chat_s_line(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, b"Line one.\n", "notes.md").json()["upload"]
    before = command(client, "attach", f"{upload_id} check").json()["result"]["check"]
    command(client, "framing", "none")
    after = command(client, "attach", f"{upload_id} check").json()["result"]["check"]
    assert (after.get("framing"), after.get("framing_origin")) == ("", "chat")
    assert after["tokens_whole"] < before["tokens_whole"]


def test_a_text_mode_chat_has_no_framing_line(root, make):  # noqa: F811
    client, _ = make(profile="text")
    answer = command(client, "framing", OURS).json()
    assert (
        answer.get("error") == "The framing line needs a chat-mode profile; text is in text mode."
    )


def test_the_terminal_says_the_chat_s_line(root, chat_tokenizer):  # noqa: F811
    from tests.test_attach_close import chat_loop

    loop, output, _ = chat_loop(root, chat_tokenizer)
    loop.start()
    output.clear()
    loop.handle(f"/framing {OURS}")
    assert f"Framing line (this chat): {OURS}\n" in "".join(output)
    output.clear()
    loop.handle("/framing none")
    assert "No framing line (this chat).\n" in "".join(output)


# --- the main session's record checks, and the sheet's Edit (the board's ⑦, N77-N81)


def test_an_old_log_without_a_framing_record_reads_as_before(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "say", "Tell me about the wren")
    log_id = command(client, "state").json()["result"]["log_id"]
    command(client, "new", "")
    command(client, "resume", log_id)
    state = command(client, "state").json()["result"]
    assert (state.get("framing"), state.get("framing_origin")) == (FRAMING, "default")


def test_a_call_made_under_this_chat_s_line_replays_exactly(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "framing", OURS)
    command(client, "say", "Tell me about the wren")
    command(client, "framing", "none")  # a later change doesn't touch the logged call
    replay = command(client, "replay", "").json()["result"]["replay"]
    assert replay["identical"] is True, replay


def test_history_shows_each_message_s_line_and_origin(root, make):  # noqa: F811
    client, _ = make(SEARCHES)
    command(client, "framing", OURS)
    command(client, "say", "Tell me about the wren")
    turn = last_message(client)
    assert (turn.get("framing"), turn.get("framing_origin")) == (OURS, "chat")


def sheet_frame(client, upload_id: str) -> str:
    import re

    from tests.test_web_search_c2 import get

    sheet = get(client, "attach", upload=upload_id)
    frame = re.search(r'<div class="frame".*?<!-- /frame -->', sheet, re.S)
    assert frame, "no framing box"
    return frame.group(0)


def test_the_sheet_s_framing_line_has_an_edit(root, make):  # noqa: F811
    from tests.test_web_search_c2 import text_of

    client, _ = make()
    upload_id = upload(client, b"Line one.\n", "notes.md").json()["upload"]
    frame = sheet_frame(client, upload_id)
    assert 'data-action="framing-edit"' in frame and "How to change it" not in frame
    assert '<label class="vh" for="framing-line">Framing line</label>' in frame
    assert (
        "For this chat's messages from now on; the profile keeps its own line. Leave it empty "
        "to send no framing line." in text_of(frame)
    )
    assert 'data-action="framing-back"' not in frame  # the profile's line: nothing to go back to
    command(client, "framing", OURS)
    frame = sheet_frame(client, upload_id)
    said = text_of(frame)  # the label, its Edit, then the line (the board's ⑦)
    assert (
        f"Clarity introduces the document to the model with this line (this chat): Edit {OURS}"
        in said
    )
    assert "Back to the profile's line" in text_of(frame)
    command(client, "framing", "none")
    assert text_of(sheet_frame(client, upload_id)).startswith("No framing line (this chat).")
