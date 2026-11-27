"""v0.4.3 I4 and C1, in the browser: Inspect's viewer on Clarity's own export of a run (a
FakeBackend run of the chat and text profiles), served as the guide says, with
`inspect view --log-dir runs/RUN_ID/inspect`. G0's checks V1, V2, V4 and V8 hold for the
real writer, and the run's times show where G0's hand-made logs showed 1969 and NaN (V7).
A chat's export, a sample per reply (V5's ruling), shows every reply and claims none missing.

Run with `uv run pytest -m browser` (needs `uv sync --group browser --extra inspect`).
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.browser
pytest.importorskip("inspect_ai")

from clarity_chat import cli  # noqa: E402
from clarity_chat.export_inspect import export_chat, export_run  # noqa: E402
from clarity_chat.logs import read_records  # noqa: E402
from tests import test_export_inspect, test_export_inspect_chat  # noqa: E402
from tests.test_browser_inspect import (  # noqa: E402
    browser_type_launch_args,  # noqa: F401  the fixture: Chrome on the Mac
    only_local,
    page_text,
    sample_route,
    serve,
    show,
    watched,
)

run = test_export_inspect.run  # the fixture: a finished FakeBackend run
make = test_export_inspect_chat.make  # the fixture: a FakeBackend session
NO_TIME = ("1969-12-31", "1970-01-01", "NaN")


def test_the_viewer_shows_a_runs_export(run, context):
    session, run_dir = run
    log = str(run_dir / "calls.jsonl")
    first = next(r for r in read_records(run_dir / "calls.jsonl") if r["type"] == "call")
    assert cli.main(["mark", log, first["call_id"], "landed"]) == 0
    export_run(session, run_dir)
    root = run_dir.parent.parent
    with serve(run_dir / "inspect", root, {}) as viewer:
        page = watched(viewer, context)
        text = show(page, viewer, "/logs/1-chat.eval", "not-run-by-inspect")
        assert "clarity\nnot-run-by-inspect" in text  # V1
        assert "clarity/chat" in text and "error" not in text.lower()  # V8
        assert "ANSWER" in text and "SCORE" in text  # V4
        assert text.count("unmarked") == 3 and "landed" in text
        assert not [t for t in NO_TIME if t in text]  # V7: the run's times
        show(page, viewer, "/logs/1-chat.eval/info", "clarity-run")
        page.wait_for_timeout(2000)
        text = page_text(page)
        assert "Clarity made these calls, not Inspect." in text  # the I4 ruling: no click
        for tab, waits_for in (("task", "TASK INFO"), ("models", "clarity/chat")):
            text = show(page, viewer, f"/logs/1-chat.eval/{tab}", waits_for)
            assert not [t for t in NO_TIME if t in text], tab
        text = show(page, viewer, sample_route("1-chat.eval", 1, 1, "info"), "prompt_ids")
        for key in ("call_id", "prompt_text", "message_spans", "output_text", "settings"):
            assert key in text, key  # V2
        text = show(page, viewer, "/logs/2-text.eval", "clarity/text")
        assert "error" not in text.lower() and text.count("unmarked") == 4
        only_local(viewer)


def test_the_viewer_shows_a_chats_export(make, context):
    """Two messages, two /regens of the first, and an /edit of the second: five samples of
    one epoch, where uneven epochs read as "N / M Samples" (V5). The chat's title, the log's
    dataset name, is HTML, which shows as text and runs nothing (V9; v0.4.3's review, C4)."""
    from tests.inspect_gate_logs import HTML_REPLY, REMOTE_IMAGE

    session = make()
    log = test_export_inspect_chat.two_messages_regens_and_an_edit(session)
    session.set_title(HTML_REPLY)
    path = export_chat(session, log)["path"]
    first = next(r for r in read_records(log) if r["type"] == "turn" and r["role"] == "user")
    replies = [
        r["text"] for r in read_records(log) if r["type"] == "turn" and r["role"] == "assistant"
    ]
    name = f"{log.stem}.eval"
    with serve(session.logs_dir / "exports", session.root, {}) as viewer:
        page = watched(viewer, context)
        text = show(page, viewer, f"/logs/{name}", "Ask me about herons.")
        assert "5 Samples" in text and not re.search(r"\d+ / \d+ Samples", text)  # V5
        assert text.count("unmarked") == 5
        assert not [t for t in NO_TIME if t in text]
        route = sample_route(name, f"{first['turn_id']}-2", 1, "messages")
        text = show(page, viewer, route, replies[1].strip())
        assert "The kettle sang twice." in text and "error" not in text.lower()
        for tab, waits_for in (("", "Ask me about herons."), ("/task", "TASK INFO")):
            text = show(page, viewer, f"/logs/{name}{tab}", waits_for)
            assert page.evaluate("window.gateRan") is None, tab
        assert not any(REMOTE_IMAGE in url for url in viewer.requests)
        only_local(viewer)
        assert path.endswith(name)


def test_the_viewer_shows_an_exports_text_as_written(make, chat_tokenizer, context):
    """The log asks for its text untrusted, so the Messages tab shows V11's reply as written,
    not as Markdown: its leading spaces, its `* ` line and its `<think>` tag, and a hidden
    character as a marker. The note still shows on Info unopened."""
    from tests.fakes import FakeBackend
    from tests.inspect_gate_logs import MARKDOWN_REPLY

    reply = MARKDOWN_REPLY + "\nA hidden\u200bspace."
    session = make(FakeBackend([reply], n_ctx=512, tokenizer=chat_tokenizer))
    session.new_conversation("chat")
    session.set_setting("max_tokens", "400")  # the fixture's 16 would cut the reply short
    session.say("Show me some Markdown.")
    assert session.conversation is not None
    log = session.conversation.log.path
    export_chat(session, log)
    turn = next(r for r in read_records(log) if r["type"] == "turn" and r["role"] == "user")
    name = f"{log.stem}.eval"
    with serve(session.logs_dir / "exports", session.root, {}) as viewer:
        page = watched(viewer, context)
        route = sample_route(name, f"{turn['turn_id']}-1", 1, "messages")
        show(page, viewer, route, "a starred line")
        page.wait_for_timeout(1500)
        body = page.inner_text("body")
        # The message, line by line; the header's Answer has it too, on one line.
        assert f"\n{MARKDOWN_REPLY}\n" in body
        assert "A hidden\u27e8U+200B\u27e9space." in body
        show(page, viewer, f"/logs/{name}/info", "Clarity made these calls")
        page.wait_for_timeout(2000)
        assert "Clarity made these calls, not Inspect." in page_text(page)
        only_local(viewer)
