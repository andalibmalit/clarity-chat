"""A4 (v0.3.1): where the framing line and each attached document sit in a call's prompt,
and what an export says about them. Made-up documents only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clarity_chat.documents import FRAMING, added_ranges, message_content
from clarity_chat.export import export_log
from clarity_chat.profiles import load_defaults, load_profile
from clarity_chat.render import build_prompt, message_spans
from tests.test_session_attach import chat_session, doc

FIELDS = {
    "attachments": [
        {"name": "notes.md", "text": "Line one.\nLine two.\n", "marker": None},
        {"name": "b&c.md", "text": "Made-up text", "marker": "Excerpt: lines 1 to 1 of 3"},
    ],
    "framing": FRAMING,
    "placement": "start",
}
BLOCKS = [
    '<document name="notes.md">\nLine one.\nLine two.\n</document>',
    '<document name="b&amp;c.md">\nExcerpt: lines 1 to 1 of 3\nMade-up text\n</document>',
]


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    return chat_session(fetched_root, chat_tokenizer)


WHAT = [{"what": "framing"}, {"what": "attachment", "index": 0}, {"what": "attachment", "index": 1}]


def spans_of(tokenizer, root, fields):
    profile = load_profile(root, "chat", load_defaults(root))
    text = "What's in it?"
    turns = [("user", message_content(text, fields))]
    prompt = build_prompt(tokenizer, profile, turns)
    spans = message_spans(
        tokenizer, profile, turns, ["turn-1"], prompt, added=[added_ranges(text, fields)]
    )
    assert isinstance(spans, list)
    return prompt, next(e for e in spans if e.get("turn_id") == "turn-1")


def test_each_added_range_decodes_to_the_framing_line_and_each_block(olmo_tokenizer, root):
    prompt, entry = spans_of(olmo_tokenizer, root, FIELDS)
    added = entry["added"]
    assert [{k: v for k, v in a.items() if k in ("what", "index")} for a in added] == WHAT
    for a, text in zip(added, [FRAMING, *BLOCKS], strict=True):
        assert prompt.text[slice(*a["chars"])] == text
        assert olmo_tokenizer.decode(prompt.ids[slice(*a["tokens"])]) == text
        assert "edge_token" not in a
    first, last = entry["tokens"]
    assert first <= added[0]["tokens"][0] and added[-1]["tokens"][1] <= last


def test_an_empty_framing_line_has_no_range(olmo_tokenizer, root):
    _, entry = spans_of(olmo_tokenizer, root, {**FIELDS, "framing": ""})
    assert [a["what"] for a in entry["added"]] == ["attachment", "attachment"]


def test_a_message_without_attachments_has_no_added(chat):
    session, _ = chat
    outcome = session.say("Hi")
    entry = next(e for e in outcome.record["message_spans"] if e.get("role") == "user")
    assert "added" not in entry


def test_a_call_logs_the_added_ranges(chat):
    session, _ = chat
    session.attach(doc())
    outcome = session.say("What's in it?")
    record = outcome.record
    entry = next(e for e in record["message_spans"] if e.get("role") == "user")
    texts = [record["prompt_text"][slice(*a["chars"])] for a in entry["added"]]
    assert texts == [FRAMING, '<document name="notes.md">\nLine one.\nLine two.\n</document>']


def exported(session) -> tuple[dict, list[dict]]:
    result = export_log(session, session.conversation.log.path)
    rows = Path(result["path"]).read_text(encoding="utf-8").splitlines()
    return result, [json.loads(row) for row in rows[1:]]


def test_an_export_row_lists_the_attachments_on_its_path_with_their_tokens(chat):
    session, _ = chat
    d = doc()
    session.attach(d, "lines", 1, 1)
    outcome = session.say("a")
    session.say("b")
    _, rows = exported(session)
    user = outcome.record["parent_id"]
    entry = next(e for e in outcome.record["message_spans"] if e.get("turn_id") == user)
    expected = {
        "turn_id": user,
        "name": "notes.md",
        "how": "excerpt",
        "range": [1, 1],
        "sha256": d.sha256,
        "extracted_sha256": d.extracted_sha256,
        "tokens": entry["added"][1]["tokens"],
    }
    assert [row["attachments"] for row in rows] == [[expected], [expected]]


def test_an_export_says_it_holds_the_attached_documents_text(chat):
    from clarity_chat.cli import format_export

    session, _ = chat
    session.say("a")
    result, _ = exported(session)
    assert result["attachments"] is False
    assert "attached" not in format_export(result)
    session.attach(doc())
    session.say("b")
    session.edit("b2")  # the edit carries the same document: still a yes, not a count
    result, _ = exported(session)
    assert result["attachments"] is True
    assert format_export(result).endswith(
        " It holds the text of the documents attached in this chat."
    )


def test_spans_derived_for_an_old_call_equal_the_logged_ones(chat):
    """derive_spans (for a call logged before v0.2.1) reads the attachments too."""
    session, _ = chat
    session.attach(doc())
    logged = session.say("a").record["message_spans"]
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("message_spans", None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _, exported_rows = exported(session)
    assert exported_rows[0]["message_spans"] == logged


def test_the_recompute_check_runs_over_the_whole_prompt_block_included(chat):
    """Offline, with a stand-in model that records the IDs it gets."""
    torch = pytest.importorskip("torch")
    from clarity_chat.recompute import measure
    from tests.test_recompute import fixed_model, logged_from

    session, _ = chat
    session.attach(doc())
    session.say("a")
    _, rows = exported(session)
    row = rows[0]
    table = torch.randn(300, 300, generator=torch.Generator().manual_seed(0))
    row["logprobs"] = logged_from(table, row["prompt_ids"], row["output_ids"], torch)
    model, _ = fixed_model(table)
    assert measure(row, model, "transformers")["measured"]["largest_gap"] == 0.0
    assert model.seen[: len(row["prompt_ids"])] == row["prompt_ids"]
    assert "Line one." in row["prompt_text"]


@pytest.mark.parametrize("attachments", [False, True])
def test_the_export_sheet_says_it_holds_the_attached_documents_text(attachments):
    from clarity_chat import web

    answer = {
        "path": "/x/logs/exports/c.jsonl",
        "replaced": False,
        "calls": 1,
        "failed_calls_left_out": 0,
        "calls_without_a_turn_left_out": 0,
        "steered": [],
        "attachments": attachments,
        "found": False,  # v0.3.2: the parts search added
    }
    sheet = web.templates.get_template("export.html").render(**web.export_sheet({"result": answer}))
    said = "It holds the text of the documents attached in this chat." in sheet
    assert said is attachments


@pytest.mark.parametrize("framing", [" Note", "Note"])
def test_a_template_that_trims_the_message_moves_the_ranges_with_it(root, tmp_path, framing):
    """Tripwire should 7: a template that strips the message (as Llama 3.1's `| trim`)
    moves each range by what it dropped at the start."""
    from tests.test_spans import LOOP, toy

    tokenizer = toy(tmp_path, LOOP.replace("BODY", "m.content | trim"))
    profile = load_profile(root, "chat", load_defaults(root))
    fields = {**FIELDS, "framing": framing}
    text = "What's in it?\n"
    turns = [("user", message_content(text, fields))]
    prompt = build_prompt(tokenizer, profile, turns)
    spans = message_spans(
        tokenizer, profile, turns, ["turn-1"], prompt, added=[added_ranges(text, fields)]
    )
    assert isinstance(spans, list)
    entry = next(e for e in spans if e.get("turn_id") == "turn-1")
    assert entry["trimmed"] is True
    texts = [prompt.text[slice(*a["chars"])] for a in entry["added"]]
    assert texts == ["Note", *BLOCKS]


@pytest.mark.parametrize("framing", ["Read this:\n", "Read this: "])
def test_a_framing_line_ending_in_whitespace_flags_its_edge_token(olmo_tokenizer, root, framing):
    """Tripwire should 7: its last token runs into the blank line after it, as C2's
    highlight will meet with a profile's own framing."""
    prompt, entry = spans_of(olmo_tokenizer, root, {**FIELDS, "framing": framing})
    first = entry["added"][0]
    assert prompt.text[slice(*first["chars"])] == framing
    assert first["edge_token"] == "end"
