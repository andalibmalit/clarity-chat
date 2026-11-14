"""/compare: the replies to one message, grouped by their settings, with where they part."""

from __future__ import annotations

import time

import pytest

from clarity_chat.commands import Commands
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError, Setting
from clarity_chat.session import Session
from tests.fakes import FakeBackend


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    def make(replies, n_ctx=None):
        backend = FakeBackend(replies, n_ctx=n_ctx, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        session.new_conversation("text")
        return Commands(session), session

    return make


def compare(c, args=""):
    return c.run("compare", args, lambda e: None)


def again_with_seed(session, seed):
    """Another reply to the current message with a given seed, as /regen same will make."""
    conversation = session.conversation
    reply = conversation.turns[conversation.head]
    first = conversation.replies_to(reply.parent_id)[0]
    return session.call(
        conversation.turns[reply.parent_id],
        seed=Setting(seed, "regen"),
        regen_of=first.call_id,
        back_to=reply.turn_id,
    )


def test_replies_that_differ_only_by_seed_are_one_group(chat):
    c, session = chat([" one", " two", " three"])
    session.say("Hi")
    session.regen()
    session.regen()
    result = compare(c)
    assert result.get("message", {}).get("text") == "Hi"
    replies = result["replies"]
    assert [(r["k"], r["seed"], r["text"]) for r in replies] == [
        (1, 42, " one"),
        (2, 43, " two"),
        (3, 44, " three"),
    ]
    assert [r["group"] for r in replies] == [0, 0, 0] and result["groups"] == [{"settings": {}}]
    assert [r["current"] for r in replies] == [False, False, True]
    assert result["same"]["temperature"] == 0.6 and "seed" not in result["same"]
    assert result["same_prompt"] is True and result["pairs"] == []
    assert result["again"] is None  # every seed has a reply under today's settings


def test_matched_seeds_after_a_setting_change_pair_up_and_show_where_they_part(chat):
    c, session = chat([" Hello there", " Hi you", " Hello world"])
    session.say("Hi")
    session.regen()  # seed 43
    session.set_setting("temperature", "0.9")
    again_with_seed(session, 42)
    result = compare(c)
    assert [r["group"] for r in result.get("replies", [])] == [0, 0, 1]
    assert result["groups"] == [
        {"settings": {"temperature": 0.6}},
        {"settings": {"temperature": 0.9}},
    ]
    assert "temperature" not in result["same"]
    (pair,) = result["pairs"]
    first, _, third = result["replies"]
    assert (pair["seed"], pair["a"], pair["b"]) == (42, first["call_id"], third["call_id"])
    a, b = (session.conversation.calls[r["call_id"]]["output_ids"] for r in (first, third))
    split = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
    assert pair["first_different_token"] == split == 7  # " Hello " is 7 one-byte tokens
    assert pair["first_different_char"] == 7
    assert result["again"] == {"seeds": [43], "now": {"temperature": 0.9}}


def test_an_earlier_message_is_compared_by_a_reply_id_without_moving_the_head(chat):
    c, session = chat([" one", " two", " three"])
    first = session.say("a")
    session.regen()
    session.say("b")
    head = session.conversation.head
    heads_before = len(
        [r for r in read_records(session.conversation.log.path) if r["type"] == "head"]
    )
    result = compare(c, first.record["call_id"])
    assert result.get("message", {}).get("text") == "a"
    assert [r["text"] for r in result["replies"]] == [" one", " two"]
    assert [r["later_turns"] for r in result["replies"]] == [0, 2]  # " two" went on: b, three
    assert result["again"] is None  # not the current message: /regen acts there
    assert session.conversation.head == head
    heads = [r for r in read_records(session.conversation.log.path) if r["type"] == "head"]
    assert len(heads) == heads_before


def test_compare_refuses_a_failed_call_and_other_words(chat):
    from clarity_chat.backends import BackendError

    c, session = chat([" one"])
    session.say("a")
    session.backend.fail_with = "made-up failure"
    with pytest.raises(BackendError):
        session.say("b")
    failed = session.conversation.last_call["call_id"]
    with pytest.raises(ConfigError, match="failed, so it has no reply to compare"):
        compare(c, failed)
    with pytest.raises(ConfigError, match=r"Usage: /compare \[CALL_ID\]"):
        compare(c, "everything")


def test_compare_on_400_exchanges_is_quick(chat):
    c, session = chat([" ok"] * 401, n_ctx=100_000)
    for _ in range(400):
        session.say("m")
    session.regen()
    started = time.perf_counter()
    assert len(compare(c).get("replies", [])) == 2
    assert time.perf_counter() - started < 5


def test_the_terminal_prints_the_groups_and_where_matched_seeds_part(chat):
    from clarity_chat.cli import format_compare

    c, session = chat([" Hello there", " Hi you", " Hello world"])
    session.say("Hi")
    session.regen()
    session.set_setting("temperature", "0.9")
    again_with_seed(session, 42)
    text = format_compare(compare(c))
    assert text.startswith("3 replies to 'Hi':")
    assert "Group 2: temperature 0.9" in text
    assert "part at token 7 (character 7)." in text
    assert text.endswith("/regen same makes seeds 43 again at temperature 0.9.")


def test_two_replies_with_the_same_seed_and_settings_are_not_a_pair(chat):
    """A pair shows where a setting made two same-seed replies part; within a group there's
    no setting to blame (mutation check: pairs within a group)."""
    c, session = chat([" one", " one"])
    session.say("Hi")
    again_with_seed(session, 42)
    result = compare(c)
    assert [r["group"] for r in result["replies"]] == [0, 0] and result["pairs"] == []


def test_again_is_only_offered_for_your_last_message(chat):
    """Mutation check: after a /set, an earlier message has seeds to remake, but /regen acts
    on your last message only."""
    c, session = chat([" one", " two", " three"])
    first = session.say("a")
    session.regen()
    session.say("b")
    session.set_setting("temperature", "0.9")
    assert compare(c, first.record["call_id"])["again"] is None
    assert compare(c)["again"] == {"seeds": [42], "now": {"temperature": 0.9}}


@pytest.mark.parametrize("version", ["v0.1.0", "v0.2.0", "v0.2.1"])
def test_settings_an_old_log_didnt_record_dont_count_as_changed(chat, fetched_root, version):
    """Checkpoint 1, finding 2: logprobs and steering_scale arrived later; a missing value
    isn't a different one."""
    import shutil
    from pathlib import Path

    golden = Path(__file__).parent / "fixtures" / "golden" / f"{version}.jsonl"
    c, session = chat([" More."])
    log = Path(shutil.copy(golden, session.logs_dir / f"old-{version.replace('.', '')}.jsonl"))
    c.run("resume", log.stem, lambda e: None)
    result = compare(c)
    assert result["again"] is None
    assert len(result["groups"]) == 1
    with pytest.raises(ConfigError, match="already has a reply"):
        c.run("regen", "same", lambda e: None)


def test_a_reply_that_stops_where_the_other_goes_on_is_marked_a_prefix(chat):
    """Checkpoint 1, finding 12: "part at token N" is wrong when one just stops."""
    c, session = chat([" Hello there", " Hello there, friend"])
    session.say("Hi")
    session.set_setting("max_tokens", "40")
    again_with_seed(session, 42)
    (pair,) = compare(c)["pairs"]
    assert pair.get("prefix") is True and pair["first_different_token"] == 12


def test_the_split_character_counts_characters_not_tokens(chat):
    """Checkpoint 1, finding 6: é is two tokens in the byte-level test vocabulary."""
    c, session = chat([" Café au lait", " Hi", " Café noir"])
    session.say("Hi")
    session.regen()
    session.set_setting("temperature", "0.9")
    again_with_seed(session, 42)
    (pair,) = compare(c)["pairs"]
    assert (pair["first_different_token"], pair["first_different_char"]) == (7, 6)


def test_same_prompt_is_false_when_a_reply_was_sent_another_prompt(chat):
    """Checkpoint 1, finding 9 (a resume whose template renders differently does this)."""
    c, session = chat([" one", " two"])
    session.say("Hi")
    session.regen()
    reply = session.conversation.turns[session.conversation.head]
    session.conversation.calls[reply.call_id]["prompt_ids"] = [1, 2, 3]
    assert compare(c)["same_prompt"] is False


def test_regen_same_counts_a_seed_made_under_the_settings_now(chat):
    """Checkpoint 1, finding 11: 0.6, then 0.9, then back to 0.6 has nothing to make."""
    c, session = chat([" a", " b", " c"])
    session.say("Hi")
    session.set_setting("temperature", "0.9")
    c.run("regen", "same", lambda e: None)
    session.set_setting("temperature", "")
    with pytest.raises(ConfigError, match="already has a reply"):
        c.run("regen", "same", lambda e: None)


def test_regen_same_keeps_the_order_seeds_were_first_used(chat):
    """Checkpoint 1, finding 11: the reference promises first-use order."""
    c, session = chat([" a", " b", " c", " d", " e", " f"])
    session.say("Hi")
    again_with_seed(session, 50)
    again_with_seed(session, 45)
    session.set_setting("temperature", "0.9")
    assert c.run("regen", "same", lambda e: None)["seeds"] == [42, 50, 45]
