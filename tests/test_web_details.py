"""The Details panel (v0.3.0 item C4b), offline: one call's record, top to bottom, and Replay.

Values on screen are checked against /prompt CALL_ID's record and its replays (data walk,
V3Details).
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


@pytest.fixture
def chat(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client, session


def result(client, name, args=""):
    answer = command(client, name, args).json()
    assert "result" in answer, answer
    return answer["result"]


def details(client, call_id: str) -> str:
    response = client.get(f"/view/details?call={call_id}", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    return html.unescape(response.text)


def test_what_the_model_got_comes_first_with_the_prompt_three_ways(chat):
    client, _ = chat
    call = result(client, "say", "Hi")["call"]
    fragment = details(client, call["call_id"])
    assert "Details · reply 1 of 1" in fragment
    n = call["n_prompt_client"]
    assert call["prompt_ids"][0] != call["bos"]["bos_token_id"]
    assert f"{n} tokens · no start token (BOS)" in fragment
    assert fragment.index("What the model got") < fragment.index("Summary")
    assert " ".join(str(i) for i in call["prompt_ids"]) in fragment
    # Text mode: the profile's header, names and line ends are highlighted, by their spans.
    assert "Highlighted: the profile’s header, names and line ends." in fragment
    for span in call["message_spans"]:
        start, end = span["chars"]
        piece = call["prompt_text"][start:end]
        if span["kind"] == "profile":
            assert f"<mark>{piece}</mark>" in fragment
    text = fragment.split('<pre class="pre" data-pane="text">')[1].split("</pre>")[0]
    shown = fragment.split('<pre class="pre" data-pane="whitespace" hidden>')[1].split("</pre>")[0]
    assert "↵" not in text and "Assistant:" in text  # Text: as it is
    assert "↵\nAssistant:" in shown  # Whitespace: each line end shown


def test_the_summary_says_how_it_stopped_how_long_and_what_ran(chat):
    client, session = chat
    call = result(client, "say", "Hi")["call"]
    fragment = details(client, call["call_id"])
    assert "end of text · all checks passed" in fragment
    end = call["output_ids"][-1]
    assert f"end of text · eos · {end}" in fragment  # no eog_tokens from the fake: the ID
    assert "The model wrote a token that ends a reply." in fragment
    timings = call["timings"]
    seconds = (timings["prompt_ms"] + timings["predicted_ms"]) / 1000
    assert f"{call['tokens_predicted']} tokens · {seconds:.1f} s" in fragment
    assert "tiny · runs on fake" in fragment and f">{call['call_id']}<" in fragment
    assert "All checks passed." in fragment
    # The end token's text, when llama-server's startup named it (/settings' eog_tokens).
    session.backend.record["model_file"] = {"eog_tokens": [{"id": end, "text": "</s>"}]}
    assert "end of text · eos · </s>" in details(client, call["call_id"])


def test_a_reply_stopped_by_you_has_no_time(chat):
    client, session = chat
    session.backend.replies, session.backend.cancel_after = [" Hello there"], 4
    call = result(client, "say", "Hi")["call"]
    assert call["stop_type"] == "cancelled" and not call.get("timings")
    fragment = details(client, call["call_id"])
    assert "stopped by you · cancelled" in fragment and "· time unknown" in fragment


def test_a_failed_check_comes_first(chat):
    client, session = chat
    session.backend.result_flags = ["check_failed:settings_echo"]
    call = result(client, "say", "Hi")["call"]
    fragment = details(client, call["call_id"])
    line = (
        "! 1 check failed: settings_echo. "
        "A value Clarity sent didn’t come back unchanged from the server."
    )
    assert line in fragment and fragment.index(line) < fragment.index("Stopped")
    assert "1 check failed" in fragment.split("</summary>")[1]


def test_settings_as_logged_with_origins(chat):
    client, _ = chat
    result(client, "set", "seed random")
    call = result(client, "say", "Hi")["call"]
    assert call["seed_drawn"]
    fragment = details(client, call["call_id"])
    seed = call["settings"]["seed"]["value"]
    assert f"{seed} (random)" in fragment
    assert "0.6</dd>" in fragment and "0 (off)" in fragment  # top_k 0 is off
    assert "8,192" in fragment and "server.toml" in fragment


def test_replay_and_what_it_found(chat):
    client, _ = chat
    call = result(client, "say", "Hi")["call"]
    assert "not replayed yet" in details(client, call["call_id"])
    replay = result(client, "replay", call["call_id"])["replay"]
    assert replay["unexplained_difference"] is True
    fragment = details(client, call["call_id"])
    assert "replayed 1×, different" in fragment
    assert "Different, and nothing that could explain it changed" in fragment
    assert f"From output token {replay['first_different_token']}." in fragment
    assert f"Compared: {replay['compared']}." in fragment
    assert replay["logged_output_text"] in fragment and replay["output_text"] in fragment


def test_a_failed_call_shows_its_error_and_no_replay(chat):
    client, session = chat
    session.backend.fail_with = "A made-up failure."
    command(client, "say", "Hi")
    call = result(client, "prompt")["call"]
    fragment = details(client, call["call_id"])
    assert "Details · failed call" in fragment
    assert "! Failed: A made-up failure." in fragment and "none" in fragment
    assert 'data-action="replay"' not in fragment


def test_an_earlier_reply_says_so(chat):
    client, _ = chat
    first = result(client, "say", "Hi")["call"]
    result(client, "say", "Again")
    assert "An earlier reply" in details(client, first["call_id"])


# --- Probabilities (C4c), a section of Details


def fake_logprobs(tokenizer, ids, chosen=-0.5):
    (z,) = tokenizer.encode("z")
    return {
        "view": "raw logits, before any sampler",
        "n": 2,
        "tokens": [{"id": t, "logprob": chosen, "top": [[t, chosen], [z, -3.0]]} for t in ids],
    }


def test_recorded_probabilities_shade_the_reply_token_by_token(chat):
    import math

    client, session = chat
    tokenizer = session.tokenizer
    session.backend.replies = [" Café, olé!"]
    ids = tokenizer.encode(" Café, olé!") + [tokenizer.tok.eos_token_id]
    session.backend.logprobs = fake_logprobs(tokenizer, ids, chosen=-2.0)  # about 14%
    call = result(client, "say", "Hi")["call"]
    probs = result(client, "probs", call["call_id"])
    fragment = details(client, call["call_id"])
    assert "2 likeliest per token" in fragment
    assert "Shading is per token: a word, part of a word, or punctuation." in fragment
    # The shaded reply: its tokens' pieces, in order, make the reply (the end token aside).
    shaded = fragment.split('<div class="msg shaded"')[1].split("</div>")[0]
    pieces = re.findall(
        r'<button type="button" class="tok[^"]*" data-i="(\d+)"[^>]*>(.*?)</button>', shaded
    )
    reply = call["output_text"]
    tokens = probs["tokens"]
    assert (
        "".join(text for i, text in pieces if tokens[int(i)]["end"] > tokens[int(i)]["start"])
        == reply
    )
    assert 'class="tok p1"' in shaded  # 14%: the 10–30% band
    first = tokens[0]
    chance = math.exp(first["logprob"])
    assert f"The model gave “{first['text']}” a {chance:.0%} chance" in fragment
    assert f"Log-probability {first['logprob']:.2f}: the natural log of that chance." in fragment
    assert "The likeliest tokens here" in fragment and "z</span>" in fragment


@pytest.mark.parametrize(
    ("backend_name", "logprobs", "heading", "sentence"),
    [
        (
            "llama.cpp",
            None,
            "not on llama.cpp, for now",
            "Probabilities need a profile that runs on Transformers.",
        ),
        (
            "transformers",
            None,
            "not recorded",
            "No probabilities for this reply: they weren’t switched on when it was written, "
            "and replaying it won’t add them. Turn them on for the next reply.",
        ),
        (
            "transformers",
            {"unavailable": "cancelled"},
            "none recorded",
            "None recorded: the reply was stopped.",
        ),
    ],
)
def test_without_probabilities_the_section_says_why(
    chat, backend_name, logprobs, heading, sentence
):
    client, session = chat
    session.backend.name, session.backend.logprobs = backend_name, logprobs
    call = result(client, "say", "Hi")["call"]
    assert result(client, "probs", call["call_id"])["reason"] in ("llama_cpp", "off", "cancelled")
    fragment = details(client, call["call_id"])
    assert f'Probabilities <span class="n">{heading}</span>' in fragment and sentence in fragment
    if heading == "not recorded":
        assert 'data-set="logprobs 5">Record them from the next reply<' in fragment
