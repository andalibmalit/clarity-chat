"""The replies pop-out and Compare (v0.3.0 item C3), offline.

Every value on screen is checked against /compare's result (data walk, V3Replies and
V3Compare). Replies come from the fake backend, so their texts are chosen, not sampled.
"""

from __future__ import annotations

import html
import re

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command

HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}'
SEED_ONLY = "These replies differ only by seed, so where they part says nothing about a setting."
REUSED = "Replies 4–6 reused the seeds of replies 1–3, so each row differs only in temperature."
MARKER = (
    '<span class="split"><span class="vh">first difference from the first column: </span></span>'
)
IDENTICAL = {"first_different_token": None, "prefix": False, "first_different_char": None}
PREFIX = {"first_different_token": 3, "prefix": True, "first_different_char": 3}
PARTS = {"first_different_token": 3, "prefix": False, "first_different_char": 3}


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    """A chat on the text profile with one exchange: (client, session)."""
    backend = FakeBackend([" One two three."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        result(client, "say", "Hi")
        yield client, session


def result(client, name, args=""):
    answer = command(client, name, args).json()
    assert "result" in answer, answer
    return answer["result"]


def view(client, name: str, call: str) -> str:
    response = client.get(f"/view/{name}?call={call}", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    return html.unescape(response.text)


def last_call(client) -> str:
    return result(client, "state")["turns"][-1]["call_id"]


def test_one_reply_offers_only_more_replies(chat):
    client, _ = chat
    compared = result(client, "compare")
    assert len(compared["replies"]) == 1
    fragment = view(client, "replies", last_call(client))
    assert "1 reply to this message" in fragment
    assert re.search(r'<input[^>]*type="number"[^>]*value="3"', fragment)
    assert ">Make 3<" in fragment  # as Again's button (design walkthrough)
    for absent in ("reply 1 of 1", "Compare these replies", "Again with"):
        assert absent not in fragment


def test_the_rows_say_only_what_differs_and_where_each_went(chat):
    client, session = chat
    session.backend.replies = [" Two.", " Three."]
    result(client, "regen", "2")
    compared = result(client, "compare")
    fragment = view(client, "replies", last_call(client))
    assert "3 replies to this message" in fragment and "Compare these replies" in fragment
    for reply in compared["replies"]:
        assert f"reply {reply['k']} of 3" in fragment
        # One group: only the seed differs, so no setting is named; a reply that didn't go
        # on says nothing about it (design walkthrough, finding 18).
        assert f'<span class="df">seed {reply["seed"]}</span>' in fragment
        if reply["current"]:
            assert "You are here" in fragment
        else:
            assert f'data-switch="{reply["turn_id"]}"' in fragment
            assert f'Show this reply<span class="vh"> (reply {reply["k"]} of 3)' in fragment
    assert "temperature" not in fragment and "Again with" not in fragment


def test_a_continued_reply_says_how_far(chat):
    client, session = chat
    session.backend.replies = [" Two."]
    result(client, "regen")
    first = result(client, "compare")["replies"][0]
    result(client, "switch", first["turn_id"])
    session.backend.replies = [" Onward."]
    result(client, "say", "Go on")
    compared = result(client, "compare", first["call_id"])
    assert compared["replies"][0]["later_turns"] == 2
    fragment = view(client, "replies", first["call_id"])
    assert "continued, 2 more turns" in fragment  # turns, not messages (finding 6)
    # An earlier message: the list and Compare, no More replies (/regen acts on the last).
    assert "Make 3" not in fragment and "Compare these replies" in fragment


def test_after_a_change_again_makes_the_same_seeds_and_the_rows_name_the_setting(chat):
    client, session = chat
    session.backend.replies = [" Two.", " Three."]
    result(client, "regen", "2")
    result(client, "set", "temperature 0.3")
    again = result(client, "compare")["again"]
    assert again == {"seeds": [42, 43, 44], "now": {"temperature": 0.3}}
    fragment = view(client, "replies", last_call(client))
    assert "Again with the same 3 seeds, now at temperature 0.3" in fragment
    assert 'data-action="again"' in fragment and ">Make 3<" in fragment
    session.backend.replies = [" One two four.", " Two.", " Three."]
    result(client, "regen", "same")
    compared = result(client, "compare")
    assert [g["settings"] for g in compared["groups"]] == [
        {"temperature": 0.6},
        {"temperature": 0.3},
    ]
    fragment = view(client, "replies", last_call(client))
    assert 'data-action="again"' not in fragment  # the settings match the newest replies now
    assert '<span class="df">seed 42 · temperature 0.6</span>' in fragment
    assert '<span class="df">seed 42 · temperature 0.3</span>' in fragment
    divider = "You set temperature to 0.3, then chose “Again with the same 3 seeds”"
    assert fragment.index("reply 3 of 6") < fragment.index(divider) < fragment.index("reply 4 of 6")


def test_compare_with_only_the_seed_differing(chat):
    client, session = chat
    session.backend.replies = [" Two.", HOSTILE]
    result(client, "regen", "2")
    compared = result(client, "compare")
    state = result(client, "state")
    settings = result(client, "settings")
    raw = client.get(f"/view/compare?call={last_call(client)}", headers={**AUTH, **SAME}).text
    assert "<script>" not in raw and "<img" not in raw  # B5's fixture, for Compare
    fragment = html.unescape(raw)
    assert "Your message · 3 replies" in fragment and compared["message"]["text"] in fragment
    assert SEED_ONLY in fragment
    model = f"{state['model_name']} · runs on {settings['backend']}"
    same = f"Same in all: profile {state['profile']} · {model}"
    assert (
        same in fragment and "temperature 0.6" in fragment and fragment.count(" · the prompt") == 1
    )
    for reply in compared["replies"]:
        assert f"reply {reply['k']} of 3 · seed {reply['seed']}" in fragment
        assert reply["text"] in fragment
    assert 'class="cmp-grid"' not in fragment  # cards, no columns: nothing lines up
    # Only what tells a run apart from a default one (design walkthrough, finding 4): no
    # neutral or off values, no sampler list, no "not steered" without a vector.
    line = re.search(r'<p class="cmp-same">(.*?)</p>', fragment).group(1)  # ty: ignore
    assert line == f"{same} · max_tokens 16 · temperature 0.6 · top_p 0.95 · the prompt"


def test_compare_lines_up_matched_seeds_and_marks_where_they_part(chat):
    client, session = chat
    session.backend.replies = [" Two.", " Three."]
    result(client, "regen", "2")
    result(client, "set", "temperature 0.3")
    session.backend.replies = [" One two four.", " Two.", " Three!"]
    result(client, "regen", "same")
    compared = result(client, "compare")
    fragment = view(client, "compare", last_call(client))
    assert REUSED in fragment
    assert 'class="cmp-grid"' in fragment
    assert "Temperature 0.6" in fragment and "compared against" not in fragment  # finding 5
    assert (
        '<span class="split" aria-hidden="true"></span> marks where a reply first differs '
        "from the Temperature 0.6 one." in fragment
    )
    assert "Temperature 0.3" in fragment
    by_call = {r["call_id"]: r for r in compared["replies"]}
    for pair in compared["pairs"]:
        later = by_call[pair["b"]]
        if pair["first_different_char"] is None:
            assert "the same text" in fragment
        elif pair["prefix"]:
            assert "the same until the shorter one ends" in fragment
        else:
            cut = pair["first_different_char"]
            assert later["text"][:cut] + MARKER + later["text"][cut:] in fragment
    assert any(p["first_different_char"] for p in compared["pairs"])  # the setup has a split


def test_compare_warns_when_seed_and_a_setting_both_differ(chat):
    client, session = chat
    result(client, "set", "temperature 0.3")
    session.backend.replies = [" Two."]
    result(client, "regen")
    compared = result(client, "compare")
    assert compared["pairs"] == [] and len(compared["groups"]) == 2
    fragment = view(client, "compare", last_call(client))
    # Advice, not a failure: the lead's plain style, no "!" (design walkthrough, finding 24).
    assert (
        '<p class="cmp-lead">These replies differ in seed and in temperature, so a difference '
        "can come from either. For a closer comparison, use Again with the same seeds."
    ) in fragment
    assert "reply 1 of 2 · seed 42 · temperature 0.6" in fragment
    assert "reply 2 of 2 · seed 43 · temperature 0.3" in fragment


@pytest.mark.parametrize(
    ("pair", "note"),
    [
        (
            {"first_different_token": None, "prefix": False, "first_different_char": None},
            "the same text",
        ),
        (
            {"first_different_token": 3, "prefix": True, "first_different_char": 3},
            "the same until the shorter one ends",
        ),
        ({"first_different_token": 3, "prefix": False, "first_different_char": 3}, None),
    ],
)
def test_a_pair_s_note(pair, note):
    assert web.pair_note(pair) == note


def test_setting_words():
    assert web.setting_words("steering_scale", 0.0) == "not steered"
    assert web.setting_words("steering_scale", 0.5) == "steered ×0.5"
    assert web.setting_words("temperature", 0.3) == "temperature 0.3"
    assert web.setting_words("samplers", ["top_k", "min_p"]) == "samplers top_k, min_p"
    assert web.changed_words({"steering_scale": 0.5}) == "strength 0.5"


# --- Checkpoint 4's findings


def test_untrusted_text_in_the_replies_pop_out_is_escaped(chat):
    """B5's fixture, for the pop-out: a reply's text and a mark."""
    client, session = chat
    session.backend.replies = [HOSTILE]
    result(client, "regen")
    result(client, "mark", f"{last_call(client)} {HOSTILE}")
    raw = client.get(f"/view/replies?call={last_call(client)}", headers={**AUTH, **SAME}).text
    # max_tokens 16 cuts the reply; its first 16 characters are the opening tag, escaped.
    assert "<script>" not in raw and "<img" not in raw and "&lt;script&gt;" in raw


def test_values_are_shown_as_python_writes_them():
    """Finding 11: repr, not :g, so 1.0 doesn't read as 1 and 1e-05 keeps its digits."""
    assert web.setting_words("temperature", 1.0) == "temperature 1.0"
    assert web.setting_words("min_p", 0.00001) == "min_p 1e-05"
    assert web.setting_words("steering_scale", 1.0) == "steered ×1.0"
    assert web.changed_words({"steering_scale": 0.5}) == "strength 0.5"


def test_a_change_then_a_plain_regenerate_doesn_t_claim_the_same_seeds(chat):
    """Checkpoint 4's mutation check: the divider adds "then chose Again with the same N
    seeds" only when the new group's seeds are the group before's."""
    client, session = chat
    result(client, "set", "temperature 0.3")
    session.backend.replies = [" Other."]
    result(client, "regen")
    compared = result(client, "compare")
    assert [r["seed"] for r in compared["replies"]] == [42, 43]
    fragment = view(client, "replies", last_call(client))
    assert re.findall(r'<p class="divider">(.*?)</p>', fragment) == ["You set temperature to 0.3"]


def test_one_seed_again_is_the_same_seed_singular(chat):
    """One reply, then a setting: the offer and the divider say "the same seed"."""
    client, session = chat
    result(client, "set", "temperature 0.3")
    assert result(client, "compare")["again"]["seeds"] == [42]
    fragment = view(client, "replies", last_call(client))
    assert "Again with the same seed, now at temperature 0.3" in fragment
    assert "1 seeds" not in fragment
    session.backend.replies = [" Other."]
    result(client, "regen", "same")
    fragment = view(client, "replies", last_call(client))
    assert re.findall(r'<p class="divider">(.*?)</p>', fragment) == [
        "You set temperature to 0.3, then chose “Again with the same seed”"
    ]
