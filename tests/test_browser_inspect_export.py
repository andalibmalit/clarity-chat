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
    finding,
    only_local,
    page_text,
    record,
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
        record(page, "I4-log")
        assert "clarity\nnot-run-by-inspect" in text  # V1
        assert "clarity/chat" in text and "error" not in text.lower()  # V8
        assert "ANSWER" in text and "SCORE" in text  # V4
        assert text.count("unmarked") == 3 and "landed" in text
        assert not [t for t in NO_TIME if t in text]  # V7: the run's times
        show(page, viewer, "/logs/1-chat.eval/info", "clarity-run")
        page.wait_for_timeout(2000)
        record(page, "I4-log-info")
        text = page_text(page)
        finding("I4 note shown unopened on Info:", "Clarity made these calls" in text)
        assert "Clarity made these calls, not Inspect." in text  # the I4 ruling: no click
        for tab, waits_for in (("task", "TASK INFO"), ("models", "clarity/chat")):
            text = show(page, viewer, f"/logs/1-chat.eval/{tab}", waits_for)
            record(page, f"I4-{tab}-tab")
            assert not [t for t in NO_TIME if t in text], tab
        text = show(page, viewer, sample_route("1-chat.eval", 1, 1, "info"), "prompt_ids")
        record(page, "I4-sample-metadata")
        for key in ("call_id", "prompt_text", "message_spans", "output_text", "settings"):
            assert key in text, key  # V2
        text = show(page, viewer, "/logs/2-text.eval", "clarity/text")
        record(page, "I4-text-log")
        assert "error" not in text.lower() and text.count("unmarked") == 4
        only_local(viewer)


def test_the_viewer_shows_a_chats_export(make, context):
    """Two messages, two /regens of the first, and an /edit of the second: five samples of
    one epoch, where uneven epochs read as "N / M Samples" (V5)."""
    session = make()
    log = test_export_inspect_chat.two_messages_regens_and_an_edit(session)
    path = export_chat(session, log)["path"]
    first = next(r for r in read_records(log) if r["type"] == "turn" and r["role"] == "user")
    replies = [
        r["text"] for r in read_records(log) if r["type"] == "turn" and r["role"] == "assistant"
    ]
    name = f"{log.stem}.eval"
    with serve(session.logs_dir / "exports", session.root, {}) as viewer:
        page = watched(viewer, context)
        text = show(page, viewer, f"/logs/{name}", "Ask me about herons.")
        record(page, "C1-chat-log")
        assert "5 Samples" in text and not re.search(r"\d+ / \d+ Samples", text)  # V5
        assert text.count("unmarked") == 5
        assert not [t for t in NO_TIME if t in text]
        route = sample_route(name, f"{first['turn_id']}-2", 1, "messages")
        text = show(page, viewer, route, replies[1].strip())
        record(page, "C1-chat-sample")
        assert "The kettle sang twice." in text and "error" not in text.lower()
        only_local(viewer)
        assert path.endswith(name)
