"""The shell's views (v0.3.0 item C1), offline: the picker and the composer row, the first
screen and a model starting, About, and Quit.

Each value a view shows is checked against what the command table returns for it, as the data
walk names it (notes/plans/v030-data-walk.md, V3Start and V3Picker).
"""

from __future__ import annotations

import html
import json
import re
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import CONTEXT_WARN_FRACTION, Session
from tests.conftest import TEXT_PROFILE
from tests.fakes import FakeBackend
from tests.test_web import AUTH, ORIGIN, PORT, SAME, SECRET, command, free_port, run_ui

OTHER_MODEL = """
[other]
route = "convert"
repo = "test/other"
revision = "0123456789abcdef0123456789abcdef01234567"
tokenizer_repo = "test/Other-Model-1B"
tokenizer_revision = "0123456789abcdef0123456789abcdef01234567"
"""
HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}'


@pytest.fixture
def server(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!", " Again.", " More."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        yield client, session, app


def get_view(client, name: str) -> str:
    response = client.get(f"/view/{name}", headers={**AUTH, **SAME})
    assert response.status_code == 200, response.text
    return response.text


def add_profiles(root) -> None:
    """A model that isn't fetched with a profile on it, one of yours with a description that
    holds markup, and one of yours that doesn't load."""
    with (root / "models.toml").open("a", encoding="utf-8") as f:
        f.write(OTHER_MODEL)
    far = TEXT_PROFILE.replace('model = "tiny"', 'model = "other"')
    (root / "profiles" / "far.toml").write_text(far, encoding="utf-8")
    local = root / "local" / "profiles"
    local.mkdir(parents=True)
    mine = f"description = {json.dumps(HOSTILE)}\n" + TEXT_PROFILE
    (local / "mine.toml").write_text(mine, encoding="utf-8")
    (local / "broken.toml").write_text("mode = 'nope'\n", encoding="utf-8")


def rows(fragment: str) -> list[str]:
    """The picker's rows, by the profile each one starts, in order."""
    return [html.unescape(name) for name in re.findall(r'data-profile="([^"]*)"', fragment)]


# --- the picker and the composer row (V3Picker, V3Start B1)


def test_the_picker_lists_every_profile_grouped_by_model_with_fetched_models_first(server):
    client, session, _ = server
    add_profiles(session.root)
    profiles = command(client, "profiles").json()["result"]["profiles"]
    fragment = get_view(client, "picker")
    # Every profile /profiles lists, once; the unfetched model's group after the fetched one,
    # though "far" sorts between "chat" and "text"; profiles that don't load last.
    assert sorted(rows(fragment)) == sorted(p["name"] for p in profiles)
    assert rows(fragment) == ["chat", "text", "mine", "far", "broken"]
    assert fragment.index(">tiny<") < fragment.index(">Other-Model-1B<")
    far = next(p for p in profiles if p["name"] == "far")
    assert far["model_name"] == "Other-Model-1B" and far["fetched"] is False
    assert "runs on llama.cpp · not fetched" in fragment
    broken = next(p for p in profiles if p["name"] == "broken")
    assert broken["error"] in html.unescape(fragment)
    # No chat yet: the first screen's heading and button, no tick.
    assert "Choose a profile to start" in fragment and ">Choose a profile<" in fragment
    assert 'aria-checked="true"' not in fragment
    assert "Your current chat stays as it is." not in fragment
    assert "No model is on this computer yet" not in fragment


def test_the_picker_escapes_a_description(server):
    """B5's fixture, for the picker: a description holding markup is shown as text."""
    client, session, _ = server
    add_profiles(session.root)
    fragment = get_view(client, "picker")
    assert "<script>" not in fragment and "<img" not in fragment
    assert HOSTILE in html.unescape(fragment)  # shown as text, never run


def test_with_nothing_fetched_the_picker_says_so(root, tiny_tokenizer):
    session = Session(root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        fragment = get_view(client, "picker")
    assert "No model is on this computer yet, so none of these can start." in fragment
    assert "aria-describedby=" in fragment  # the menu points at the sentence
    assert rows(fragment) == ["chat", "text"]  # the usual order


def test_in_a_chat_the_picker_ticks_the_profile_and_the_button_names_its_model(server):
    client, _, _ = server
    command(client, "profile", "text")
    state = command(client, "state").json()["result"]
    fragment = get_view(client, "picker")
    name = f"{state['model_name']} · {state['profile']}"
    assert name == "tiny · text"
    assert f">{name}<" in fragment
    assert f'aria-label="Profile: {name}. Start a new chat with another profile"' in fragment
    ticked = re.findall(r'aria-checked="true"[^>]*data-profile="([^"]*)"', fragment)
    assert ticked == ["text"]
    assert "Start a new chat with" in fragment and "Your current chat stays as it is." in fragment
    assert "aria-disabled" not in fragment


def test_settings_counts_the_settings_changed_in_this_chat(server):
    client, _, _ = server
    command(client, "profile", "text")
    assert ">Settings<" in get_view(client, "picker") and "changed" not in get_view(
        client, "picker"
    )
    command(client, "set", "temperature 0.3")
    command(client, "set", "top_p 0.5")
    settings = command(client, "settings").json()["result"]["settings"]
    changed = sum(1 for s in settings.values() if s["origin"] == "chat")
    assert changed == 2
    assert f"· {changed} changed" in get_view(client, "picker")


def test_the_context_figure_is_the_last_prompts_share_of_the_context(server):
    client, _, _ = server
    command(client, "profile", "text")
    assert 'class="ctx' not in get_view(client, "picker")  # none before the first reply
    command(client, "say", "Hi")
    call = command(client, "prompt").json()["result"]["call"]
    n, ctx = call["n_prompt_client"], call["n_ctx"]
    fragment = get_view(client, "picker")
    assert f"{n / ctx:.0%} · {n:,} of {ctx:,} tokens" in fragment
    assert "ctx hot" not in fragment and n <= CONTEXT_WARN_FRACTION * ctx


@pytest.fixture
def small(fetched_root, tiny_tokenizer):
    """A context of 160 tokens: a long message fills over 80% of it, a longer one won't fit."""
    backend = FakeBackend([" Yes.", " No."], n_ctx=160, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        yield client


def test_over_80_percent_the_figure_turns_to_a_warning(small):
    client = small
    command(client, "say", "x" * 100)
    call = command(client, "prompt").json()["result"]["call"]
    n, ctx = call["n_prompt_client"], call["n_ctx"]
    assert n > CONTEXT_WARN_FRACTION * ctx  # the setup: over 80%, and it fit
    assert f'class="ctx hot">! {n / ctx:.0%} · {n:,} of {ctx:,} tokens' in get_view(
        client, "picker"
    )


def test_after_a_message_that_doesnt_fit_the_figure_is_the_reply_s(small):
    """The data walk: a call refused for its size isn't counted; the head reply's is."""
    client = small
    command(client, "say", "Hi")
    replied = command(client, "prompt").json()["result"]["call"]
    refused = command(client, "say", "x" * 200).json()
    assert refused["type"] == "ContextError"
    assert command(client, "prompt").json()["result"]["call"]["error"].startswith("ContextError")
    n, ctx = replied["n_prompt_client"], replied["n_ctx"]
    assert f"{n / ctx:.0%} · {n:,} of {ctx:,} tokens" in get_view(client, "picker")


def start_held(client, session, name="text", how="profile"):
    """Start `name` (`how`: profile, or resume a chat by its ID) with the backend's start held
    until the returned event is set."""
    gate, started = threading.Event(), threading.Event()
    real = session.backend.start

    def held(*args):
        started.set()
        gate.wait(10)
        return real(*args)

    session.backend.start = held
    thread = threading.Thread(target=lambda: command(client, how, name))
    thread.start()
    assert started.wait(10)
    return gate, thread


def test_while_a_model_starts_the_picker_waits_and_says_why(server):
    client, session, _ = server
    gate, thread = start_held(client, session)
    try:
        state = command(client, "state").json()["result"]
        assert state["model"] == "starting" and state["running"]["args"] == "text"
        fragment = get_view(client, "picker")
    finally:
        gate.set()
        thread.join(10)
    assert 'aria-disabled="true"' in fragment
    assert "Waits until the model has started." in fragment
    assert ">tiny · text<" in fragment  # the profile being started, not "Choose a profile"
    assert "Settings" not in fragment  # no chat to set yet


def test_while_a_reply_is_written_the_picker_waits_and_says_why(server):
    client, session, _ = server
    command(client, "profile", "text")
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    said: list = []
    thread = threading.Thread(target=lambda: said.append(command(client, "say", "Hi")))
    thread.start()
    threading.Event().wait(0.4)
    try:
        assert command(client, "state").json()["result"]["running"]["command"] == "say"
        fragment = get_view(client, "picker")  # answered while the reply is written
    finally:
        thread.join(10)
    assert 'aria-disabled="true"' in fragment and "Waits until the reply is done." in fragment
    assert ">Settings<" in fragment  # Settings still opens (decision 5)


# --- the first screen and a model starting (V3Start B1, B2, B5, B6)


def test_with_no_chat_and_no_chats_the_thread_is_empty(server):
    client, _, _ = server
    fragment = get_view(client, "thread")
    assert "earlier chat" not in fragment and "Starting" not in fragment


def test_with_no_chat_the_thread_points_to_earlier_chats(server, tiny_tokenizer):
    client, session, _ = server
    command(client, "profile", "text")
    command(client, "say", "Hi")  # a chat in logs/
    later = Session(
        session.root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    with TestClient(web.make_app(later, SECRET.encode(), port=PORT), base_url=ORIGIN) as fresh:
        assert command(fresh, "chats").json()["result"]["chats"]
        fragment = get_view(fresh, "thread")
    assert "Or open an earlier chat from" in fragment and 'data-action="chats"' in fragment


def test_while_a_model_starts_the_thread_says_which_and_where(server):
    client, session, _ = server
    gate, thread = start_held(client, session)
    try:
        fragment = get_view(client, "thread")
    finally:
        gate.set()
        thread.join(10)
    text = next(
        p for p in command(client, "profiles").json()["result"]["profiles"] if p["name"] == "text"
    )
    assert 'role="status"' in fragment and "Starting text…" in fragment
    assert f"{text['model_name']} · runs on {text['backend']}" in fragment
    assert "A large model can take a minute to start." in fragment


def test_a_new_chat_shows_its_profile_model_and_where_it_runs(server):
    client, _, _ = server
    command(client, "profile", "text")
    state = command(client, "state").json()["result"]
    settings = command(client, "settings").json()["result"]
    fragment = get_view(client, "thread")
    assert f'<p class="pname">{state["profile"]}</p>' in fragment
    assert f"{state['model_name']} · runs on {settings['backend']}" in fragment
    assert "Every reply is logged with exactly what the model got." in fragment
    command(client, "say", "Hi")
    assert "Every reply is logged" not in get_view(client, "thread")  # only before a message


def test_a_start_warning_shows_whole_and_escaped(server):
    client, session, _ = server
    session.backend.start_warnings = [f"A made-up warning {HOSTILE}"]
    command(client, "profile", "text")
    warnings = command(client, "state").json()["result"]["warnings"]
    fragment = get_view(client, "thread")
    assert warnings == [f"A made-up warning {HOSTILE}"]
    assert "<script>" not in fragment and "<img" not in fragment
    assert f'role="alert"><strong>!</strong> {warnings[0]}' in html.unescape(fragment)


@pytest.mark.parametrize(
    ("steering", "scale", "words"),
    [
        (None, 0.0, None),
        ({"file": "v.gguf"}, 0.5, "steered ×0.5"),
        ({"file": "v.gguf"}, 1.0, "steered ×1"),
        ({"file": "v.gguf"}, 0.0, "not steered"),
    ],
)
def test_the_steering_words_on_a_new_chat(steering, scale, words):
    settings = {"steering": steering, "settings": {"steering_scale": {"value": scale}}}
    assert web.steering_words(settings) == words


# --- About and Quit (V3Picker C7, V3Start A6)


def test_about_shows_the_version_and_commits_from_state(server):
    client, _, _ = server
    state = command(client, "state").json()["result"]
    fragment = get_view(client, "about")
    assert f"<dd>{state['clarity']['version']}</dd>" in fragment
    assert state["clarity"]["commit"] and state["clarity"]["commit"] in fragment
    assert state["llama_cpp_commit"] in fragment


def test_quit_answers_then_stops_the_server(server):
    client, _, app = server
    stops: list[bool] = []
    app.state.quit = lambda: stops.append(True)
    answer = command(client, "quit").json()
    assert answer["result"] == {"quit": True} and stops == [True]


def test_quit_from_the_page_ends_clarity_ui(fetched_root, tiny_tokenizer):
    import http.client

    port, links = free_port(), []
    thread, servers = run_ui(
        fetched_root, tiny_tokenizer, port, lambda path: links.append(path.read_text("utf-8"))
    )
    for _ in range(100):
        if servers and servers[0].started:
            break
        threading.Event().wait(0.05)
    found = re.search(r"#key=([A-Za-z0-9_-]+)", links[0])
    assert found
    secret = found.group(1)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Origin": f"http://127.0.0.1:{port}",
        "Sec-Fetch-Site": "same-origin",
        "Content-Type": "application/json",
    }
    body = json.dumps({"command": "quit", "args": "", "id": 1, "tab": "tab-1"})
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/command", body, headers)
    answer = json.loads(connection.getresponse().read())
    connection.close()
    thread.join(10)
    assert answer["result"] == {"quit": True}
    assert not thread.is_alive()
