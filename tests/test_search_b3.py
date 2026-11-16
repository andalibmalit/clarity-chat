"""B3 (v0.3.2): where each found part sits in a call's prompt, and what an export carries
and says about them (decision 16: export rows list the found parts with their token ranges;
the recompute check runs over the full prompt). Made-up documents, scratch roots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clarity_chat.documents import FRAMING, added_ranges, library_add, message_content, reading
from clarity_chat.export import export_log
from clarity_chat.profiles import load_defaults, load_profile
from clarity_chat.render import build_prompt, message_spans
from tests.test_documents_board import chat, quiet

GLOSSARY = (
    b"## Wren\nThe wren is a small brown bird.\n\n"
    b"## Dipper\nThe dipper walks under water.\n\n"
    b"## Heron\nThe heron stands still in water.\n"
)
PARTS = [
    {"document": "glossary", "title": "Wren", "page": None, "text": "## Wren\nThe wren sings."},
    {"document": "b&c", "title": None, "page": 3, "text": "Made-up text\n"},
]
BLOCKS = [
    '<part document="glossary" title="Wren">\n## Wren\nThe wren sings.\n</part>',
    '<part document="b&amp;c" page="3">\nMade-up text\n</part>',
]


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def fields(placement: str) -> dict:
    found = {"parts": PARTS}
    return {"attachments": [], "framing": FRAMING, "placement": placement, "found": found}


@pytest.mark.parametrize("placement", ["start", "end"])
def test_each_part_range_decodes_to_its_block(olmo_tokenizer, root, placement):
    profile = load_profile(root, "chat", load_defaults(root))
    text = "Tell me about the wren"
    f = fields(placement)
    turns = [("user", message_content(text, f))]
    prompt = build_prompt(olmo_tokenizer, profile, turns)
    spans = message_spans(
        olmo_tokenizer, profile, turns, ["turn-1"], prompt, added=[added_ranges(text, f)]
    )
    assert isinstance(spans, list)
    entry = next(e for e in spans if e.get("turn_id") == "turn-1")
    added = entry["added"]
    assert [{k: a[k] for k in a if k in ("what", "index")} for a in added] == [
        {"what": "framing"},
        {"what": "part", "index": 0},
        {"what": "part", "index": 1},
    ]
    for a, block in zip(added, [FRAMING, *BLOCKS], strict=True):
        assert prompt.text[slice(*a["chars"])] == block
        assert olmo_tokenizer.decode(prompt.ids[slice(*a["tokens"])]) == block


def exported(session) -> tuple[dict, list[dict]]:
    result = export_log(session, session.conversation.log.path)
    rows = Path(result["path"]).read_text(encoding="utf-8").splitlines()
    return result, [json.loads(row) for row in rows[1:]]


def test_an_export_row_lists_the_found_parts_on_its_path_with_their_tokens(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    first = c.run("say", "Tell me about the wren", quiet)["call"]
    second = c.run("say", "And the heron?", quiet)["call"]
    _, rows = exported(session)
    users = {}
    for call in (first, second):
        user = call["parent_id"]
        entry = next(e for e in call["message_spans"] if e.get("turn_id") == user)
        users[user] = [a["tokens"] for a in entry["added"] if a["what"] == "part"]
    log = [json.loads(line) for line in session.conversation.log.path.open()]  # type: ignore[union-attr]
    found = {r["turn_id"]: r["found"] for r in log if r["type"] == "turn" and "found" in r}

    def expected(user: str) -> dict:
        return {
            "turn_id": user,
            "words": found[user]["words"],
            "documents": [
                {"name": d["name"], "build_id": d["build_id"]} for d in found[user]["documents"]
            ],
            "parts": [
                {"id": p["id"], "sha256": p["sha256"], "tokens": tokens}
                for p, tokens in zip(found[user]["parts"], users[user], strict=True)
            ],
        }

    one, two = first["parent_id"], second["parent_id"]
    assert [p["id"] for p in found[one]["parts"]] == ["glossary:1"]
    assert [p["id"] for p in found[two]["parts"]] == ["glossary:3"]
    assert [row.get("found") for row in rows] == [[expected(one)], [expected(one), expected(two)]]


def test_a_row_without_search_has_an_empty_found(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    c.run("say", "Hello", quiet)
    _, rows = exported(session)
    assert rows[0].get("found") == []


@pytest.mark.parametrize(
    ("attach", "search", "said"),
    [
        (False, False, None),
        (True, False, "It holds the text of the documents attached in this chat."),
        (False, True, "It holds the text of the parts search added from your documents."),
        (True, True, "It holds the text of the documents attached in this chat and of the "
         "parts search added from your documents."),
    ],
)  # fmt: skip
def test_an_export_says_whose_text_it_holds(root, chat_tokenizer, tmp_path, attach, search, said):
    from clarity_chat import web
    from clarity_chat.cli import format_export

    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]' if search else "")
    if attach:
        notes = tmp_path / "notes.md"
        notes.write_text("Made-up notes.\n", encoding="utf-8")
        c.run("attach", str(notes), quiet)
    c.run("say", "Tell me about the wren", quiet)
    result, _ = exported(session)
    assert result.get("found") is search
    terminal = format_export(result)
    sheet = web.templates.get_template("export.html").render(**web.export_sheet({"result": result}))
    if said is None:
        assert "It holds the text" not in terminal and "It holds the text" not in sheet
    else:
        assert terminal.endswith(f" {said}")
        assert f"<p>{said}</p>" in sheet


def test_spans_derived_for_an_old_searched_call_equal_the_logged_ones(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    logged = c.run("say", "Tell me about the wren", quiet)["call"]["message_spans"]
    assert any(a["what"] == "part" for e in logged for a in e.get("added", []))
    path = session.conversation.log.path  # type: ignore[union-attr]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("message_spans", None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _, exported_rows = exported(session)
    assert exported_rows[0]["message_spans"] == logged


def test_the_recompute_check_runs_over_the_whole_prompt_parts_included(root, chat_tokenizer):
    """Offline, with a stand-in model that records the IDs it gets."""
    torch = pytest.importorskip("torch")
    from clarity_chat.recompute import measure
    from tests.test_recompute import fixed_model, logged_from

    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]')
    c.run("say", "Tell me about the wren", quiet)
    _, rows = exported(session)
    row = rows[0]
    table = torch.randn(300, 300, generator=torch.Generator().manual_seed(0))
    row["logprobs"] = logged_from(table, row["prompt_ids"], row["output_ids"], torch)
    model, _ = fixed_model(table)
    assert measure(row, model, "transformers")["measured"]["largest_gap"] == 0.0
    assert model.seen[: len(row["prompt_ids"])] == row["prompt_ids"]
    assert "The wren is a small brown bird." in row["prompt_text"]
