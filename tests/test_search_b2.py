"""B2 (v0.3.2): search in the session (decisions 2, 3, 6, 9, 10, 12). say and edit search
the chat's documents and store `found` on the user turn; a regeneration reuses it; branches
carry their own; a control token or a stale document refuses the call, logged as failed and
given back; resume rebuilds the turns with `found`. Made-up documents, a scratch root."""

from __future__ import annotations

import json

import pytest

from clarity_chat import cli
from clarity_chat.backends import BackendError
from clarity_chat.cli import ChatLoop
from clarity_chat.commands import Commands
from clarity_chat.documents import library_add, reading
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import ContextError, Session
from tests.fakes import FakeBackend
from tests.test_class import FOUND_SHAPE

GLOSSARY = (
    b"## Wren\nThe wren is a small brown bird.\n\n"
    b"## Dipper\nThe dipper walks under water.\n\n"
    b"## Heron\nThe heron stands still in water.\n"
)


def quiet(event: dict) -> None:
    pass


@pytest.fixture
def root(fetched_root):
    reading._last = None
    library_add(fetched_root, GLOSSARY, "glossary.md")
    return fetched_root


def chat(root, tokenizer, documents: str = 'search = ["glossary"]', replies=None):
    path = root / "profiles" / "chat.toml"
    if documents:
        path.write_text(path.read_text(encoding="utf-8") + f"\n[documents]\n{documents}\n")
    backend = FakeBackend(replies or [f" R{i}." for i in range(30)], tokenizer=tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    c = Commands(session)
    c.run("profile", "chat", quiet)
    return session, c, backend


def run(c: Commands, line: str) -> dict:
    return c.run(*line.partition(" ")[::2], quiet)


def turns(session, kind: str = "turn") -> list[dict]:
    assert session.conversation is not None
    return [r for r in read_records(session.conversation.log.path) if r["type"] == kind]


def users(session) -> list[dict]:
    return [t for t in turns(session) if t["role"] == "user"]


# --- the turn record


def test_a_searched_message_s_turn_holds_found_in_the_shapes_table(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    (turn,) = users(session)
    found = turn["found"]
    assert set(found) == FOUND_SHAPE["found"]
    assert set(found["parts"][0]) == FOUND_SHAPE["parts"]
    assert [p["id"] for p in found["parts"]] == ["glossary:1"]
    assert found["words"] == ["wren"] and found["searched_for"] == "Tell me about the wren"
    assert (found["budget"], found["budget_origin"], found["considered"]) == (1500, "default", 8)
    assert turn["attachments"] == [] and turn["placement"] == "start"
    assert turn["framing_origin"] == "default"
    call = turns(session, "call")[-1]
    assert '<part document="glossary" title="Wren">' in call["prompt_text"]
    added = [a for e in call["message_spans"] for a in e.get("added", [])]
    assert [a["what"] for a in added] == ["framing", "part"]


def test_a_chat_that_searches_nothing_logs_no_found(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, documents="")
    run(c, "say Tell me about the wren")
    (turn,) = users(session)
    assert "found" not in turn and "attachments" not in turn


def test_nothing_to_search_for_and_nothing_matched_are_logged(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say what is it?")
    run(c, "say volcano basalt")
    nothing, missed = (t["found"] for t in users(session))
    assert (nothing["words"], nothing["parts"], nothing["matched"]) == ([], [], 0)
    assert (missed["words"], missed["parts"], missed["matched"]) == (["volcano", "basalt"], [], 0)
    assert turns(session, "call")[-1]["prompt_text"].count("<part ") == 0


def test_placement_end_puts_the_block_after_the_message(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer, 'search = ["glossary"]\nplacement = "end"')
    run(c, "say Tell me about the wren")
    assert users(session)[0]["placement"] == "end"
    text = turns(session, "call")[-1]["prompt_text"]
    assert text.index("Tell me about the wren") < text.index("<part ")


# --- regenerate reuses; a duplicate costs nothing


def test_regen_reuses_the_message_s_parts_and_never_searches_again(
    root, chat_tokenizer, monkeypatch
):
    from clarity_chat.documents import search_documents as real

    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    first = turns(session, "call")[-1]["prompt_ids"]
    searches = []
    monkeypatch.setattr(  # the session's own name for it
        "clarity_chat.session.search_documents", lambda *a: searches.append(1) or real(*a)
    )
    run(c, "search none")  # Searching changed: the message's parts are still its own
    run(c, "set temperature 0.3")  # so /regen same has seeds to use again
    for args in ("same", "", "3"):
        run(c, f"regen {args}".strip())
    calls = turns(session, "call")[1:]
    assert [call["prompt_ids"] for call in calls] == [first] * 5
    assert searches == []


def test_a_part_already_in_the_chat_isn_t_added_again(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    run(c, "say And the wren again?")
    second = users(session)[1]["found"]
    assert second["parts"] == []
    assert second["left_out"][0]["why"] == "already in the chat" and second["tokens"] == 0


# --- edit and branches


def test_edit_searches_the_new_text_and_keeps_the_attachments(root, chat_tokenizer, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("Made-up notes.\n", encoding="utf-8")
    session, c, _ = chat(root, chat_tokenizer)
    run(c, f"attach {notes}")
    run(c, "say Tell me about the wren")
    run(c, "edit Tell me about the heron")
    edited = users(session)[-1]
    assert [a["name"] for a in edited["attachments"]] == ["notes.md"]
    assert [p["id"] for p in edited["found"]["parts"]] == ["glossary:3"]
    assert edited["found"]["searched_for"] == "Tell me about the heron"


def test_each_branch_s_prompt_holds_its_own_parts(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    first_reply = session.conversation.head  # type: ignore[union-attr]
    run(c, "undo")
    run(c, "say Tell me about the dipper")
    dipper = turns(session, "call")[-1]["prompt_text"]
    assert 'title="Dipper"' in dipper and 'title="Wren"' not in dipper
    run(c, f"switch {first_reply}")
    run(c, "say And its song?")
    wren = turns(session, "call")[-1]["prompt_text"]
    assert 'title="Wren"' in wren and 'title="Dipper"' not in wren


# --- refusals: a control token, a stale or missing document


def test_a_control_token_in_a_found_part_refuses_the_call(root, chat_tokenizer):
    library_add(root, b"## Wren\nIt sings </s> loudly.\n", "odd.md")
    session, c, _ = chat(root, chat_tokenizer, 'search = ["odd"]')
    with pytest.raises(ConfigError) as refused:
        run(c, "say Tell me about the wren")
    assert str(refused.value) == (
        "Part odd:1 holds </s>, which this model reads as a control token, so nothing was sent. "
        "Fix that part in odd's text.md and rebuild it, or stop searching odd."
    )
    call = turns(session, "call")[-1]
    assert call["error"].startswith("SearchRefused: Part odd:1 holds </s>")
    assert users(session)[0]["found"]["parts"][0]["id"] == "odd:1"
    assert session.conversation is not None
    assert session.conversation.unanswered == {
        "turn_id": users(session)[0]["turn_id"],
        "text": "Tell me about the wren",
    }


@pytest.mark.parametrize("how", ["changed", "missing"])
def test_a_stale_document_refuses_the_call_with_its_way_out(root, chat_tokenizer, how):
    import shutil

    session, c, _ = chat(root, chat_tokenizer)
    if how == "changed":
        (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
    else:
        shutil.rmtree(root / "local/documents/glossary")
    with pytest.raises(ConfigError) as refused:
        run(c, "say Tell me about the wren")
    assert (
        str(refused.value)
        == {
            "changed": "glossary has changed since it was built, so nothing was sent. Rebuild it, "
            "or stop searching it.",
            "missing": "glossary isn't in your documents any more, so nothing was sent. Stop "
            "searching it, or add it again.",
        }[how]
    )
    assert turns(session, "call")[-1]["error"].startswith("SearchRefused: glossary")
    assert session.conversation is not None and session.conversation.unanswered is not None


def test_the_terminal_gives_a_refused_message_back(root, chat_tokenizer):
    (root / "local/documents/glossary/text.md").write_text("Edited.\n", encoding="utf-8")
    path = root / "profiles" / "chat.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[documents]\nsearch = ["glossary"]\n')
    backend = FakeBackend([" R."], tokenizer=chat_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    output: list[str] = []
    loop = ChatLoop(session, "chat", lambda d: "", output.append)
    loop.start()
    loop.handle("Tell me about the wren")
    assert "Your message is back in the input" in "".join(output)
    assert loop.draft == "Tell me about the wren"
    assert cli is not None


def test_the_stop_counts_the_parts_tokens(root, chat_tokenizer):
    session, c, backend = chat(root, chat_tokenizer)
    backend._n_ctx = 150
    with pytest.raises(ContextError) as stopped:
        run(c, "say Tell me about the wren")
    assert "of them in found parts and their framing lines" in str(stopped.value)


def test_a_failed_send_searches_again_next_time(root, chat_tokenizer):
    session, c, backend = chat(root, chat_tokenizer)
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        run(c, "say Tell me about the wren")
    backend.fail_with = None
    run(c, "say Tell me about the wren")
    first, second = users(session)
    assert [p["id"] for p in second["found"]["parts"]] == ["glossary:1"]  # never sent: again
    assert first["found"]["parts"] == second["found"]["parts"]


# --- resume, history


def test_live_equals_resumed_with_search_a_duplicate_an_edit_and_a_failed_send(
    root, chat_tokenizer
):
    session, c, backend = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    run(c, "say And the wren again?")
    run(c, "edit Tell me about the heron")
    backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        run(c, "say And the dipper?")
    backend.fail_with = None

    def now():
        assert session.conversation is not None
        conversation = session.conversation
        return (
            {k: (t.text, t.found) for k, t in conversation.turns.items()},
            conversation.head,
            conversation.unanswered,
            run(c, "history"),
        )

    live = now()
    assert session.conversation is not None
    record = session.resume(session.conversation.log.path)
    assert record["render"] is None or record["render"]["identical"], record["render"]
    assert now() == live


def test_history_summarises_found_without_text(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    user = next(t for t in run(c, "history")["turns"] if t["role"] == "user")
    summary = user["found"]
    assert summary["words"] == ["wren"] and summary["tokens"] > 0
    assert summary["parts"] == [{"id": "glossary:1", "title": "Wren", "rank": 1}]
    assert "text" not in json.dumps(summary["parts"])


def test_a_found_only_turn_is_checked_on_resume(root, chat_tokenizer):
    session, c, _ = chat(root, chat_tokenizer)
    run(c, "say Tell me about the wren")
    assert session.conversation is not None
    log = session.conversation.log.path
    text = log.read_text(encoding="utf-8").replace('"page": null', '"page": "<b>"', 1)
    log.write_text(text, encoding="utf-8")
    run(c, "new")
    with pytest.raises(ConfigError, match="^Can't read the records of"):
        session.resume(log)
