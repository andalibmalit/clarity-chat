"""The Settings panel (v0.3.0 item C4a), offline.

Values on screen are checked against /settings (data walk, V3Settings). The Transformers and
steered cases are checked on the panel's data from a made-up /settings result, since the
fake backend runs llama.cpp profiles only.
"""

from __future__ import annotations

import html
import re
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.profiles import SETTING_HELP
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" One two three."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client, session


def result(client, name, args=""):
    answer = command(client, name, args).json()
    assert "result" in answer, answer
    return answer["result"]


def panel(client) -> str:
    response = client.get("/view/settings", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    return html.unescape(response.text)


def row(fragment: str, name: str) -> str:
    """One setting's row, by its data-setting attribute."""
    found = re.search(
        rf'<div class="srow[^"]*" data-setting="{name}">(.*?)</div><!-- row -->', fragment, re.S
    )
    assert found, name
    return found.group(1)


def test_a_llama_cpp_profile_shows_its_values_and_where_they_came_from(chat):
    client, _ = chat
    settings = result(client, "settings")
    fragment = panel(client)
    assert (
        "Starts from profile text. Changes apply from the next reply and are logged with it."
        in fragment
    )
    for name in ("temperature", "top_p", "max_tokens", "seed", "min_p", "top_k"):
        s = settings["settings"][name]
        r = row(fragment, name)
        assert f'value="{s["value"]}"' in r and f'<span class="tag">{s["origin"]}</span>' in r
        if name in SETTING_HELP:
            assert SETTING_HELP[name] in r
    assert 'type="range"' in row(fragment, "temperature") and 'type="range"' not in row(
        fragment, "seed"
    )
    # The text profile's samplers have no penalties step: one line for the four rows.
    assert (
        "Penalties: off. This profile’s sampler list has no penalties step, so a penalty "
        "wouldn’t run." in fragment
    )
    assert 'data-setting="repeat_penalty"' not in fragment
    samplers = ", ".join(settings["settings"]["samplers"]["value"])
    assert f"Samplers that run, in order: {samplers}" in fragment
    for absent in ("Steering", "logprobs", "Reset all", "Back to", "Off (", "Not used on"):
        assert absent not in fragment


def test_the_neutral_and_the_model_s_value_are_marked_on_a_slider(chat):
    client, _ = chat
    recommended = result(client, "settings")["recommended"]["values"]
    r = row(panel(client), "temperature")
    assert "▲ 1.0 neutral" in r and f"model {recommended['temperature']}" in r
    assert '<option value="1.0">' in r and f'<option value="{recommended["temperature"]}">' in r


def test_a_changed_row_offers_one_way_back(chat):
    client, _ = chat
    result(client, "set", "temperature 0.3")
    result(client, "set", "top_k 40")
    settings = result(client, "settings")["settings"]
    fragment = panel(client)
    back = settings["temperature"]["profile"]
    assert back == {"value": 0.6, "origin": "profile"}
    r = row(fragment, "temperature")
    assert '<span class="tag">chat</span>' in r
    assert 'data-set="temperature">Back to 0.6 (profile)<' in r
    # top_k's profile value is 0, its neutral value: "Neutral (0)", the same /set top_k
    # (Checkpoint 4, finding 20: neutral, not off; temperature 1 isn't off).
    assert 'data-set="top_k">Neutral (0)<' in row(fragment, "top_k")
    assert 'data-action="reset-all"' in fragment and fragment.count("data-set=") == 2


def test_while_a_reply_is_written_the_controls_wait_and_say_why(chat):
    client, session = chat
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    worker = threading.Thread(target=lambda: command(client, "say", "Hi"))
    worker.start()
    threading.Event().wait(0.4)
    try:
        fragment = panel(client)  # /settings answers during a reply
        during = worker.is_alive()
    finally:
        worker.join(10)
    assert during
    assert (
        '<p class="waits" id="settings-waits">Changes wait until this reply is done.</p>'
        in fragment
    )
    inputs = re.findall(r"<input[^>]*>", fragment)
    assert inputs and all('aria-disabled="true"' in i and "settings-waits" in i for i in inputs)


TRANSFORMERS = {
    "profile": {"name": "calm", "mode": "chat", "model": "olmo"},
    "backend": "transformers",
    "settings": {
        "temperature": {
            "value": 0.3,
            "origin": "chat",
            "profile": {"value": 0.6, "origin": "model"},
        },
        "top_k": {"value": 0, "origin": "default"},
        "top_p": {"value": 0.95, "origin": "model"},
        "min_p": {"value": 0.0, "origin": "default"},
        "repeat_penalty": {"value": 1.0, "origin": "default"},
        "repeat_last_n": {"value": 64, "origin": "default"},
        "presence_penalty": {"value": 0.0, "origin": "default"},
        "frequency_penalty": {"value": 0.0, "origin": "default"},
        "seed": {"value": 42, "origin": "default"},
        "max_tokens": {"value": 1024, "origin": "default"},
        "samplers": {
            "value": ["penalties", "temperature", "top_k", "top_p", "min_p"],
            "origin": "profile",
        },
        "logprobs": {"value": 5, "origin": "chat", "profile": {"value": 0, "origin": "default"}},
        "steering_scale": {
            "value": 1.5,
            "origin": "chat",
            "profile": {"value": 0.5, "origin": "profile"},
        },
    },
    "recommended": None,
    "not_applied": [
        {
            "name": "presence_penalty",
            "value": 0.0,
            "reason": "no Transformers equivalent",
            "affects_output": False,
        },
        {
            "name": "frequency_penalty",
            "value": 0.0,
            "reason": "no Transformers equivalent",
            "affects_output": False,
        },
        {
            "name": "repeat_last_n",
            "value": 64,
            "reason": "Transformers' repetition penalty covers the whole sequence",
            "affects_output": False,
        },
    ],
    "steering": {
        "file": "local/vectors/calm.gguf",
        "sha256": "a3f09c21be47" + "0" * 52,
        "layers": [10, 11, 12, 13, 14],
    },
}


def test_a_steered_transformers_profile():
    fragment = html.unescape(
        web.templates.get_template("settings.html").render(
            **web.settings_panel(TRANSFORMERS, waits=False)
        )
    )
    assert "calm.gguf" in fragment and "a3f09c21be47<" in fragment and "10 to 14" in fragment
    r = row(fragment, "steering_scale")
    # Rule 10: "Off (0)" while the strength isn't 0, and the profile's value when changed.
    assert 'data-set="steering_scale 0">Off (0)<' in r
    assert 'data-set="steering_scale">Back to 0.5 (profile)<' in r
    assert 'data-set="logprobs">Off (0)<' in row(fragment, "logprobs")
    assert 'data-set="temperature">Back to 0.6 (model)<' in row(fragment, "temperature")
    assert 'data-setting="repeat_penalty"' in fragment  # a box on Transformers
    assert (
        "Not used on Transformers: presence_penalty, frequency_penalty, repeat_last_n" in fragment
    )
    assert "Transformers' repetition penalty covers the whole sequence" in fragment
    assert SETTING_HELP["repeat_penalty"] in fragment and SETTING_HELP["logprobs"] in fragment


# --- Checkpoint 4's findings


def test_a_random_seed_shows_as_random_not_an_empty_number_box(chat):
    """Finding 4: a box of type number can't show "random"."""
    client, _ = chat
    result(client, "set", "seed random")
    assert result(client, "settings")["settings"]["seed"]["value"] == "random"
    r = row(panel(client), "seed")
    assert re.search(r'<input type="text"[^>]*id="set-seed"[^>]*value="random"', r)


def test_transformers_lists_every_setting_it_cant_apply():
    """Finding 22: the samplers entry is listed too."""
    settings = {
        **TRANSFORMERS,
        "not_applied": [
            *list(TRANSFORMERS["not_applied"]),  # ty: ignore
            {
                "name": "samplers",
                "value": ["top_k", "temperature"],
                "reason": "Transformers applies its processors in a fixed order; used [...]",
                "affects_output": True,
            },
        ],
    }
    fragment = html.unescape(
        web.templates.get_template("settings.html").render(**web.settings_panel(settings, False))
    )
    assert (
        "Not used on Transformers: presence_penalty, frequency_penalty, repeat_last_n, samplers"
        in fragment
    )
    assert "Transformers applies its processors in a fixed order" in fragment


# --- The design walkthrough at bd59488


def test_the_seed_row_says_which_replies_it_seeds(chat):
    """Finding 1: the seed set here is for the next message sent or edited; Regenerate and
    More replies take the next seed up (session.regen), so the row must say so."""
    client, _ = chat
    r = row(panel(client), "seed")
    assert (
        "Used for the next message you send or edit. Regenerate and More replies take the "
        "next seed up instead; to run a reply again with its own seed, use Replay in Details."
    ) in r
