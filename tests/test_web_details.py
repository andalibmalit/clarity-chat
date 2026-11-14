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
    assert "all checks passed" in fragment.split("</summary>")[1]  # the heading says it
    assert "All checks passed." not in fragment  # once (design walkthrough, finding 14)
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
    # Checkpoint 4, finding 20: a sampler's neutral value is "neutral" (temperature 1 isn't
    # off); logprobs and steering_scale at 0 are off.
    assert "0.6</dd>" in fragment and "0 (neutral)" in fragment and "0 (off)" in fragment
    assert "8,192" in fragment and "server.toml" in fragment
    # Closed at first (design walkthrough, finding 2), so Probabilities is near the top.
    assert '<details class="sec" data-key="settings">' in fragment


def test_replay_and_what_it_found(chat):
    client, _ = chat
    call = result(client, "say", "Hi")["call"]
    fragment = details(client, call["call_id"])
    assert "not replayed yet" in fragment
    # What Replay does, under its button (design walkthrough, finding 8).
    assert (
        "Sends the same prompt and settings again and compares the result with this reply."
        in fragment.split('data-key="replay"')[1]
    )
    replay = result(client, "replay", call["call_id"])["replay"]
    assert replay["unexplained_difference"] is True
    fragment = details(client, call["call_id"])
    assert "replayed 1×, different" in fragment
    assert "The replay differs, and nothing that could explain it changed" in fragment
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


def test_each_band_of_shading_and_the_chosen_alternative(chat):
    """Checkpoint 4's weak-red check: tokens at about 0.5%, 5%, 14%, 45% and 80% land in
    the bands under 10% (p0), 10–30% (p1), 30–60% (p2) and unshaded; under 1% is said so;
    the chosen token is marked among the alternatives."""
    import math

    client, session = chat
    tokenizer = session.tokenizer
    session.backend.replies = [" Hello"]
    ids = tokenizer.encode(" Hello")
    chances = [0.005, 0.05, 0.14, 0.45, 0.8, 0.8]
    (z,) = tokenizer.encode("z")
    session.backend.logprobs = {
        "view": "raw logits, before any sampler",
        "n": 2,
        "tokens": [
            {"id": t, "logprob": math.log(c), "top": [[t, math.log(c)], [z, math.log(0.05)]]}
            for t, c in zip(ids, chances, strict=True)
        ],
    }
    call = result(client, "say", "Hi")["call"]
    tokens = result(client, "probs", call["call_id"])["tokens"]
    fragment = details(client, call["call_id"])
    shaded = fragment.split('<div class="msg shaded"')[1].split("</div>")[0]
    by_index = {int(i): c for c, i in re.findall(r'class="([^"]*)" data-i="(\d+)"', shaded)}
    assert [by_index[i] for i in range(5)] == ["tok p0", "tok p0", "tok p1", "tok p2", "tok"]

    def info(i: int) -> str:
        return fragment.split(f'<div class="tokinfo" data-i="{i}" hidden>')[1].split("</div>")[0]

    assert f"The model gave “{tokens[0]['text']}” a chance under 1%" in info(0)
    assert f"The model gave “{tokens[3]['text']}” a 45% chance" in info(3)
    chosen = (
        f'<li class="chosen"><span class="w">{tokens[3]["text"]}</span><span>45%</span>'
        '<span class="d2">chosen</span></li><li><span class="w">z</span><span>5%</span></li>'
    )
    assert chosen in info(3)


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
            "None recorded: the reply was stopped, and Clarity can’t get the scores from a reply "
            "it stops part way.",
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


# --- Checkpoint 4's findings


HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}'


def raw_details(client, call_id: str):
    return client.get(f"/view/details?call={call_id}", headers={**AUTH, **SAME})


def test_spans_marked_unavailable_draw_the_prompt_unmarked_and_say_why(chat):
    """Finding 1: `message_spans` can be {"unavailable": reason}; Details answered 500."""
    client, session = chat
    call = result(client, "say", "Hi")["call"]
    session.conversation.calls[call["call_id"]]["message_spans"] = {"unavailable": "a reason"}
    lenient = TestClient(client.app, base_url=ORIGIN, raise_server_exceptions=False)
    response = raw_details(lenient, call["call_id"])
    assert response.status_code == 200, response.text
    fragment = html.unescape(response.text)
    assert "Not highlighted: a reason" in fragment and "<mark>" not in fragment
    assert "Highlighted:" not in fragment.replace("Not highlighted:", "")


def test_a_call_from_the_v0_1_0_golden_log_says_its_spans_werent_recorded(chat):
    import shutil
    from pathlib import Path

    from clarity_chat.logs import read_records

    client, session = chat
    golden = Path(__file__).parent / "fixtures" / "golden" / "v0.1.0.jsonl"
    log = Path(shutil.copy(golden, session.logs_dir / "old-v010.jsonl"))
    result(client, "resume", log.stem)
    call = next(r for r in read_records(log) if r["type"] == "call" and r.get("turn_id"))
    assert "message_spans" not in call
    response = raw_details(client, call["call_id"])
    assert response.status_code == 200, response.text
    assert "Not highlighted: this call was logged before Clarity recorded" in html.unescape(
        response.text
    )


def test_replay_waits_while_another_command_changes_the_chat(chat):
    """Finding 2: Details drawn during a reply mustn't offer Replay (its Stop would cut the
    reply)."""
    import threading

    client, session = chat
    call = result(client, "say", "Hi")["call"]
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    worker = threading.Thread(target=lambda: command(client, "say", "Again"))
    worker.start()
    threading.Event().wait(0.4)
    try:
        fragment = details(client, call["call_id"])
        during = worker.is_alive()
    finally:
        worker.join(10)
    assert during
    assert re.search(r'<button[^>]*data-action="replay"[^>]*aria-disabled="true"', fragment)
    assert "Waits until the reply is done." in fragment.split('data-key="replay"')[1]


def test_a_replay_s_own_flags_show_under_its_result(chat):
    """Finding 3: a failed check in the replay mustn't read as a plain result."""
    client, session = chat
    call = result(client, "say", "Hi")["call"]
    session.backend.result_flags = ["check_failed:settings_echo"]
    replay = result(client, "replay", call["call_id"])["replay"]
    assert "check_failed:settings_echo" in replay["flags"]
    assert "! check_failed:settings_echo" in details(client, call["call_id"])


REPLAY = {
    "identical": True,
    "unexplained_difference": False,
    "compared": "output token IDs and stop type",
    "first_different_token": None,
    "differences": {},
    "flags": [],
    "logged_server_found": False,
    "not_recorded": ["machine", "env"],
}


def test_a_replay_s_notes_are_the_terminal_s_sentences():
    """Finding 3: the same sentences as cli.format_replay, from one source."""
    from clarity_chat.cli import format_replay

    notes = web.replay_result(REPLAY).get("notes")
    assert notes == [
        "The log has no record of the server that made this call.",
        "Not compared, because the log doesn't record them: machine, env.",
    ]
    printed = format_replay({**REPLAY, "replay_id": "replay-0123456789ab"})
    assert all(note in printed for note in notes or [])


def test_what_changed_since_the_call_shows_old_and_new_values():
    """Finding 12: as board G4 draws it, "machine · macos.version 26.0.1 → 26.1"."""
    replay = {
        "identical": False,
        "unexplained_difference": False,
        "compared": "output token IDs and stop type",
        "first_different_token": 2,
        "differences": {"machine": [{"path": "macos.version", "logged": "26.0.1", "now": "26.1"}]},
        "logged_output_text": "a",
        "output_text": "b",
        "flags": [],
        "logged_server_found": True,
    }
    assert web.replay_result(replay)["changed"] == ["machine · macos.version 26.0.1 → 26.1"]


def test_tokens_without_places_are_listed_unshaded_with_a_reason(chat):
    """Finding 5: /probs gives no start when the tokens don't rebuild the reply."""
    client, session = chat
    tokenizer = session.tokenizer
    session.backend.replies = [" Hello"]
    session.backend.logprobs = fake_logprobs(tokenizer, tokenizer.encode("zzz"))
    call = result(client, "say", "Hi")["call"]
    assert result(client, "probs", call["call_id"])["tokens"][0]["start"] is None
    fragment = details(client, call["call_id"])
    assert "The tokens can’t be placed in the reply’s text, so it isn’t shaded." in fragment
    buttons = re.findall(r'<button type="button" class="tok[^"]*"[^>]*>(.*?)</button>', fragment)
    assert buttons == ["z", "z", "z"] and 'class="msg shaded"' not in fragment


def test_probabilities_are_read_once_per_call_and_the_record_keeps_ids_on_one_line(chat):
    """Finding 6: /probs isn't recomputed on every redraw; the record dump is compact."""
    import dataclasses

    from clarity_chat import commands

    client, session = chat
    tokenizer = session.tokenizer
    session.backend.logprobs = fake_logprobs(tokenizer, tokenizer.encode(" Hello!"))
    call = result(client, "say", "Hi")["call"]
    runs: list[str] = []
    real = commands.COMMANDS["probs"]
    counted = dataclasses.replace(real, run=lambda c, a, e: (runs.append(a), real.run(c, a, e))[1])
    commands.COMMANDS["probs"] = counted
    try:
        details(client, call["call_id"])
        fragment = details(client, call["call_id"])
    finally:
        commands.COMMANDS["probs"] = real
    assert len(runs) == 1
    ids = ", ".join(str(i) for i in call["prompt_ids"])
    assert f'"prompt_ids": [{ids}]' in fragment


def test_untrusted_text_in_details_is_escaped(chat):
    """B5's fixture, for Details: a reply, a failed call's error."""
    client, session = chat
    session.backend.replies = [HOSTILE]
    call = result(client, "say", "Hi")["call"]
    raw = raw_details(client, call["call_id"]).text
    assert "<script>" not in raw and "<img" not in raw
    session.backend.fail_with = HOSTILE
    command(client, "say", "Again")
    failed = result(client, "prompt")["call"]
    raw = raw_details(client, failed["call_id"]).text
    assert "<script>" not in raw and "<img" not in raw and HOSTILE in html.unescape(raw)


# --- Checkpoint 4's mutation check


def test_details_drawn_during_its_own_replay_offers_stop(chat):
    """`replaying`: while /replay of this call runs, Replaying… and Stop, not a wait."""
    import threading

    client, session = chat
    call = result(client, "say", "Hi")["call"]
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    worker = threading.Thread(target=lambda: command(client, "replay", call["call_id"]))
    worker.start()
    threading.Event().wait(0.4)
    try:
        fragment = details(client, call["call_id"])
        during = worker.is_alive()
    finally:
        worker.join(10)
    assert during
    section = fragment.split('data-key="replay"')[1].split("</details>")[0]
    assert "Replaying…" in section and 'data-action="replay-stop"' in section
    assert "Waits until the reply is done." not in section


@pytest.fixture
def chat_mode(fetched_root, chat_tokenizer):
    """The chat profile, on a tokenizer whose template writes role tags and end tokens."""
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        yield client, session


def test_in_chat_mode_the_template_s_text_is_highlighted(chat_mode):
    """prompt_pieces marks `template` spans, not only `profile` ones."""
    client, _ = chat_mode
    call = result(client, "say", "Hi")["call"]
    fragment = details(client, call["call_id"])
    assert "Highlighted: text the chat template wrote." in fragment
    text = fragment.split('<pre class="pre" data-pane="text">')[1].split("</pre>")[0]
    template = [s for s in call["message_spans"] if s["kind"] == "template"]
    assert template
    for span in template:
        start, end = span["chars"]
        if end > start:
            assert f"<mark>{call['prompt_text'][start:end]}</mark>" in text
    assert "<mark>Hi</mark>" not in text and "<mark>Be brief.</mark>" not in text


def test_a_prompt_that_starts_with_the_bos_says_so(fetched_root, bos_tokenizer):
    """`bos` reads the first prompt ID: text mode on a tokenizer that adds a BOS. Without
    one, test_what_the_model_got_comes_first… asserts "no start token (BOS)"."""
    backend = FakeBackend([" Hello!"], tokenizer=bos_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: bos_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        call = result(client, "say", "Hi")["call"]
        fragment = details(client, call["call_id"])
    assert call["prompt_ids"][0] == call["bos"]["bos_token_id"]
    assert f"{call['n_prompt_client']} tokens · starts with a start token (BOS)" in fragment
    assert "no start token" not in fragment


def test_a_setting_the_backend_couldn_t_apply_is_listed_if_it_could_change_the_reply(chat):
    """`ignored`: not_applied entries with affects_output, and only those."""
    client, session = chat
    session.backend.name = "transformers"
    session.backend.not_applied = [
        {
            "name": "presence_penalty",
            "value": 0.5,
            "reason": "no equivalent",
            "affects_output": True,
        },
        {"name": "repeat_last_n", "value": 64, "reason": "whole sequence", "affects_output": False},
    ]
    call = result(client, "say", "Hi")["call"]
    assert [e["name"] for e in call["not_applied"]] == ["presence_penalty", "repeat_last_n"]
    fragment = details(client, call["call_id"])
    assert "! 1 setting not applied: presence_penalty. Reason: no equivalent." in fragment
    assert "not applied: repeat_last_n" not in fragment


# --- The whole-branch review (multi-agent, batch 1)


def test_a_call_without_a_bos_record_says_nothing_about_the_start_token(chat):
    """Finding 5: a call with no `bos` record (a log another program wrote), with or
    without prompt IDs: Clarity can't say whether the prompt starts with one, so it says
    neither (`None == None` read as "starts with a start token")."""
    client, session = chat
    call = result(client, "say", "Hi")["call"]
    logged = session.conversation.calls[call["call_id"]]
    logged.pop("bos")
    heading = details(client, call["call_id"]).split("</summary>")[0]
    assert "start token" not in heading, heading
    # No prompt IDs either: a failed call's record (a reply's is read by /probs too).
    session.backend.fail_with = "A made-up failure."
    command(client, "say", "Again")
    failed = result(client, "prompt")["call"]
    logged = session.conversation.calls[failed["call_id"]]
    logged.pop("bos", None)
    logged.pop("prompt_ids", None)
    heading = details(client, failed["call_id"]).split("</summary>")[0]
    assert "start token" not in heading, heading


def test_a_chance_near_certain_reads_over_99_percent():
    """The fact-check (F4): rounded to whole percents, 0.996 read "100%", certain."""
    assert web.percent(0.996) == "over 99%" and web.percent(0.995) == "over 99%"
    assert web.percent(0.994) == "99%" and web.percent(0.005) == "under 1%"


# --- The lead's adjustment (c): a record missing a field never stops the session

READS_OF_A_CALL = [
    ("probs", True),
    ("history", False),
    ("branches", False),
    ("compare", True),
    ("prompt", True),
    ("settings", False),
    ("export", False),
]


def reopened_without(client, session, field):
    """Two replies with the same seed and other settings (so Compare pairs them), then the
    log with `field` taken out of its call records, as another program might write it,
    reopened. The first reply's call ID, or None if the log was refused with a sentence."""
    import json

    tokenizer = session.tokenizer
    session.backend.replies = [" Hello!", " Hello!"]
    session.backend.logprobs = fake_logprobs(
        tokenizer, [*tokenizer.encode(" Hello!"), tokenizer.tok.eos_token_id]
    )
    result(client, "set", "seed 7")
    call = result(client, "say", "Hi")["call"]["call_id"]
    result(client, "set", "temperature 0.3")
    result(client, "regen", "same")
    assert session.conversation is not None
    log = session.conversation.log.path
    lines = []
    for line in log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "call":
            record.pop(field, None)
        lines.append(json.dumps(record))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reopened = command(client, "resume", log.stem).json()
    assert not reopened.get("stopped"), reopened
    if "result" not in reopened:
        assert reopened["type"] == "ConfigError", reopened
        return None
    return call


@pytest.mark.parametrize("field", ["prompt_ids", "output_ids", "settings"])
@pytest.mark.parametrize(("name", "takes_call"), READS_OF_A_CALL)
def test_a_call_record_missing_a_field_gets_an_answer_not_a_stopped_session(
    chat, field, name, takes_call
):
    """A log another program wrote can leave out any field. Its calls here lack `field`
    (two replies with the same seed and other settings, so Compare pairs them). The log
    reopens, or is refused with a sentence; each read answers with a result or a refusal;
    none ends the session with an unexpected error."""
    client, session = chat
    call = reopened_without(client, session, field)
    if call is None:
        return
    answer = command(client, name, call if takes_call else "").json()
    assert not answer.get("stopped"), answer
    assert "result" in answer or answer["type"] == "ConfigError", answer


@pytest.mark.parametrize("field", ["output_ids", "output_text", "stop_type", "request"])
def test_a_replay_of_a_call_missing_a_field_says_which_and_the_session_goes_on(chat, field):
    """Found by the sweep above: a replay read the call's request, text and stop type
    without checking them, so a log without one stopped the session (KeyError). It's
    refused before anything starts, naming the field."""
    client, session = chat
    call = reopened_without(client, session, field)
    assert call is not None  # the log reopens: the reads it needs are all there
    answer = command(client, "replay", call).json()
    assert not answer.get("stopped"), answer
    assert answer == {
        "id": 1,
        "error": f"Call {call} can't be replayed: its record has no {field}.",
        "type": "ConfigError",
    }
