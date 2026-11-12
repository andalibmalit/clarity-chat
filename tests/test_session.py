from __future__ import annotations

import threading

import pytest

from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import ContextError, Session, room_left
from tests.fakes import FakeBackend


@pytest.fixture
def make_session(fetched_root, tiny_tokenizer):
    def make(replies=None, n_ctx=512):
        backend = FakeBackend(replies, n_ctx=n_ctx, tokenizer=tiny_tokenizer)
        session = Session(
            fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer
        )
        session.new_conversation("text")
        return session, backend

    return make


def records(session, kind=None):
    rows = list(read_records(session.conversation.log.path))
    return [r for r in rows if kind is None or r["type"] == kind]


def test_a_call_sends_the_rendered_token_ids(make_session, tiny_tokenizer):
    session, backend = make_session([" Hello!"])
    outcome = session.say("Hi")
    sent = backend.calls[0]
    assert tiny_tokenizer.decode(sent["token_ids"]) == "A conversation.\n\nUser: Hi\nAssistant:"
    assert sent["stop"] == ["\nUser:"]
    assert sent["settings"]["temperature"] == 0.6
    assert outcome.turn.text == " Hello!"


def test_call_record_has_the_brief_fields(make_session):
    session, _ = make_session([" Hello!"])
    session.say("Hi")
    call = records(session, "call")[0]
    for key in (
        "call_id",
        "parent_id",
        "conversation_id",
        "time",
        "profile",
        "settings",
        "backend",
        "model",
        "bos",
        "prompt_text",
        "prompt_ids",
        "n_prompt_client",
        "tokens_evaluated",
        "output_text",
        "output_ids",
        "stop_type",
        "flags",
        "client",
        "libraries",
    ):
        assert key in call, key
    assert call["settings"]["temperature"] == {"value": 0.6, "origin": "profile"}
    assert call["settings"]["repeat_penalty"] == {"value": 1.0, "origin": "default"}
    assert call["model"]["gguf_sha256"] == "0" * 64
    assert "toml" in call["profile"]


def test_the_log_starts_with_conversation_and_server_records(make_session):
    session, _ = make_session()
    session.say("Hi")
    kinds = [r["type"] for r in records(session)]
    assert kinds == ["conversation", "server_start", "turn", "call", "turn"]


def test_turns_form_a_chain_through_parents(make_session, tiny_tokenizer):
    session, backend = make_session([" one", " two"])
    first = session.say("a")
    second = session.say("b")
    turns = records(session, "turn")
    assert [t["origin"] for t in turns] == ["user", "model", "user", "model"]
    assert turns[2]["parent_id"] == first.turn.turn_id
    assert second.record["parent_id"] == turns[2]["turn_id"]
    prompt = tiny_tokenizer.decode(backend.calls[1]["token_ids"])
    assert prompt.endswith("User: a\nAssistant: one\nUser: b\nAssistant:")


def test_text_mode_reply_stops_before_the_next_user_line(make_session):
    session, _ = make_session([" Sure.\nUser: and then I"])
    outcome = session.say("Go")
    assert outcome.turn.text == " Sure."
    assert outcome.record["stop_type"] == "word"
    assert outcome.record["stopping_word"] == "\nUser:"


def test_context_guard_refuses_before_sending(make_session):
    session, backend = make_session(n_ctx=60)  # max_tokens is 16
    with pytest.raises(ContextError, match="Nothing was sent"):
        session.say("x" * 40)
    assert backend.calls == []
    call = records(session, "call")[0]
    assert call["error"].startswith("ContextError")
    assert session.conversation.head is None  # the refused message is a dead branch


def test_context_guard_warns_above_80_percent(make_session):
    # The byte-level test tokenizer: "A conversation.\n\nUser: " + k bytes + "\nAssistant:"
    # is 34 + k tokens. With n_ctx 100 and max_tokens 16 it fits while 34 + k + 16 < 100.
    session, _ = make_session(n_ctx=100)
    assert session.say("x" * 5).warnings == []  # 39 tokens
    session.new_conversation("text")
    outcome = session.say("x" * 48)  # 82 tokens: above 80, still fits
    assert outcome.record["n_prompt_client"] == 82
    assert any("82 of 100 context tokens" in w for w in outcome.warnings)


def test_room_left_matches_llama_server_limit():
    assert room_left(128, 100, 27) == 0
    assert room_left(128, 100, 28) == -1


def test_random_seed_is_drawn_and_logged(make_session, fetched_root):
    session, backend = make_session()
    profile = session.conversation.profile
    profile.settings["seed"] = type(profile.settings["seed"])("random", "default")
    session.say("a")
    session.say("b")
    calls = records(session, "call")
    seeds = [c["settings"]["seed"]["value"] for c in calls]
    assert all(isinstance(s, int) for s in seeds)
    assert [c["seed_drawn"] for c in calls] == [True, True]
    assert [c["settings"]["seed"] for c in backend.calls] == seeds


def test_backend_errors_are_logged_with_an_error_field(make_session):
    from clarity_chat.backends import BackendError

    session, backend = make_session()
    backend.fail_with = "boom"
    with pytest.raises(BackendError):
        session.say("a")
    call = records(session, "call")[0]
    assert call["error"] == "BackendError: boom"
    assert session.conversation.head is None


def test_cancelled_reply_is_kept_and_logged(make_session):
    session, backend = make_session([" partial reply"])
    backend.cancel_after = 5
    outcome = session.say("a")
    assert outcome.record["stop_type"] == "cancelled"
    assert outcome.turn.text == " part"
    assert session.conversation.head == outcome.turn.turn_id


def test_empty_reply_is_flagged(make_session):
    session, _ = make_session(["  "])
    assert "empty_reply" in session.say("a").record["flags"]


def test_new_conversation_writes_a_new_file_with_the_server_record(make_session):
    session, _ = make_session()
    first = session.conversation.log.path
    session.new_conversation("text")
    assert session.conversation.log.path != first
    kinds = [r["type"] for r in records(session)]
    assert kinds == ["conversation", "server_start"]


def test_unfetched_model_is_an_error(root, tiny_tokenizer):
    session = Session(root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer)
    with pytest.raises(ConfigError, match="clarity fetch tiny"):
        session.new_conversation("text")


def test_changed_weights_file_is_an_error(fetched_root, tiny_tokenizer):
    weights = fetched_root / "models" / "tiny" / "tiny-bf16.gguf"
    weights.write_bytes(b"GGUF changed weights!")
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    with pytest.raises(ConfigError, match="changed since it was fetched"):
        session.new_conversation("text")


def with_ctx_size(root, name, size):
    """A copy of the text profile with ctx_size set (a top-level key, so it goes first)."""
    text = (root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (root / "profiles" / f"{name}.toml").write_text(f"ctx_size = {size}\n{text}", "utf-8")


def test_the_context_size_comes_from_the_profile_or_server_toml(make_session, fetched_root):
    session, backend = make_session()
    assert backend.ctx_sizes == [8192]
    assert records(session, "conversation")[0]["ctx_size"] == {
        "value": 8192,
        "origin": "server.toml",
    }
    with_ctx_size(fetched_root, "big", 16384)
    session.new_conversation("big")
    assert backend.ctx_sizes == [8192, 16384]  # another size: a restart
    assert records(session, "conversation")[0]["ctx_size"] == {"value": 16384, "origin": "profile"}
    session.new_conversation("big")
    assert backend.ctx_sizes == [8192, 16384]  # the same size: no restart


@pytest.mark.parametrize("value", ["0", "-1", '"big"', "1.5"])
def test_ctx_size_must_be_a_positive_integer(make_session, fetched_root, value):
    session, _ = make_session()
    with_ctx_size(fetched_root, "bad", value)
    with pytest.raises(ConfigError, match="ctx_size must be a positive integer"):
        session.new_conversation("bad")


def test_a_cancel_before_the_first_token_is_a_failed_call_with_no_reply(make_session):
    from clarity_chat.session import CancelledBeforeReply

    session, backend = make_session([" one", " two", " three"])
    first = session.say("a")
    cancel = threading.Event()
    backend.on_call = cancel.set  # as if another thread cancelled mid-call
    with pytest.raises(CancelledBeforeReply):
        session.say("b", cancel=cancel)
    call = records(session, "call")[-1]
    assert call["stop_type"] == "cancelled" and call["cancelled_during"] == "prompt"
    assert "Cancelled before the reply started" in call["error"]
    assert "turn_id" not in call
    assert session.conversation.head == first.turn.turn_id  # the message is a dead branch
    backend.on_call = None
    assert session.say("c", cancel=threading.Event()).turn.text == " three"


def test_a_cancel_while_the_prompt_is_built_sends_nothing(make_session):
    from clarity_chat.session import CancelledBeforeReply

    session, backend = make_session()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledBeforeReply):
        session.say("a", cancel=cancel)
    assert backend.calls == []
    assert records(session, "call")[-1]["stop_type"] == "cancelled"


def test_a_cancel_after_some_text_keeps_the_partial_reply(make_session):
    session, backend = make_session([" Hello there"])
    backend.cancel_after = 3
    outcome = session.say("a")
    assert outcome.turn.text == " He"
    assert outcome.record["cancelled_during"] == "reply"


def test_an_early_cancel_on_regen_leaves_you_on_the_reply_you_had(make_session):
    from clarity_chat.session import CancelledBeforeReply

    session, backend = make_session([" one"])
    first = session.say("a")
    cancel = threading.Event()
    backend.on_call = cancel.set
    with pytest.raises(CancelledBeforeReply):
        session.regen(cancel=cancel)
    assert session.conversation.head == first.turn.turn_id


def test_set_changes_a_setting_for_this_chat_and_logs_it(make_session):
    session, backend = make_session()
    record = session.set_setting("temperature", "0.9")
    assert record["chat_settings"] == {"temperature": 0.9}
    call = session.say("a").record
    assert call["settings"]["temperature"] == {"value": 0.9, "origin": "chat"}
    assert backend.calls[0]["settings"]["temperature"] == 0.9
    session.set_setting("temperature", "")  # back to the profile's
    assert session.say("b").record["settings"]["temperature"]["origin"] == "profile"
    rows = records(session, "settings")
    assert [(r["name"], r["value"], r["chat_settings"]) for r in rows] == [
        ("temperature", 0.9, {"temperature": 0.9}),
        ("temperature", None, {}),
    ]


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("temperature", "hot", "temperature must be"),
        ("ctx_size", "4096", "unknown setting 'ctx_size'"),
        ("samplers", "top_k bogus", "unknown sampler names"),
        ("top_p", "", "isn't set in this chat"),
        ("foo", "", "unknown setting 'foo'"),
    ],
)
def test_set_checks_values_like_a_profile(make_session, name, text, message):
    session, _ = make_session()
    with pytest.raises(ConfigError, match=message):
        session.set_setting(name, text)
    assert session.conversation.chat_settings == {}
    assert records(session, "settings") == []


def test_set_keeps_the_sampler_rule(make_session):
    session, _ = make_session()
    with pytest.raises(ConfigError, match="top_p = 0.95 \\(from profile\\) has no effect"):
        session.set_setting("samplers", "temperature")  # the profile's top_p would be ignored
    session.set_setting("samplers", "top_p temperature")
    with pytest.raises(ConfigError, match="has no effect, because 'top_k' is not in samplers"):
        session.set_setting("top_k", "40")
    assert session.conversation.chat_settings == {"samplers": ["top_p", "temperature"]}


def test_set_seed_random_and_sampler_lists(make_session):
    session, _ = make_session()
    session.set_setting("seed", "random")
    session.set_setting("samplers", "top_k, top_p temperature")
    assert session.conversation.chat_settings == {
        "seed": "random",
        "samplers": ["top_k", "top_p", "temperature"],
    }
    assert session.say("a").record["seed_drawn"] is True


def test_chats_lists_conversations_newest_first_with_titles(make_session, fetched_root):
    from clarity_chat.logs import list_chats

    session, _ = make_session()
    session.say("Tell me a made-up story about a lighthouse keeper and a crow, please")
    first = session.conversation.conversation_id
    session.new_conversation("text")
    second = session.conversation.conversation_id
    session.say("b")
    session.set_title("A made-up title")
    chats = list_chats(fetched_root / "logs")
    by_id = {c["conversation_id"]: c for c in chats}
    assert [c["conversation_id"] for c in chats][:2] == sorted([first, second], reverse=True)
    assert by_id[first]["title"] == "Tell me a made-up story about a lighthouse"  # 8 words
    assert by_id[first]["title_set"] is False and by_id[first]["profile"] == "text"
    assert by_id[second]["title"] == "A made-up title" and by_id[second]["title_set"]


def test_chats_lists_a_log_it_cant_read_and_never_writes(fetched_root):
    from clarity_chat.logs import list_chats

    logs = fetched_root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "20200101-000000-bad.jsonl").write_bytes(b"not json\n")
    (logs / "20200101-000001-cut.jsonl").write_bytes(b'{"type": "conversation", "ti')
    before = {p.name: p.read_bytes() for p in logs.iterdir()}
    chats = {c["conversation_id"]: c for c in list_chats(logs)}
    assert chats["20200101-000000-bad"]["error"]
    assert chats["20200101-000000-bad"].keys() == chats["20200101-000001-cut"].keys()
    assert chats["20200101-000001-cut"]["title"] is None  # only an unfinished line so far
    assert {p.name: p.read_bytes() for p in logs.iterdir()} == before


def test_a_cancel_while_the_prompt_is_built_is_logged_like_any_early_cancel(make_session):
    from clarity_chat.session import CancelledBeforeReply

    session, _ = make_session()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledBeforeReply):
        session.say("a", cancel=cancel)
    call = records(session, "call")[-1]
    assert call["cancelled_during"] == "prompt" and "cancelled" in call["flags"]


class Echoing(FakeBackend):
    """Reports the prompt back as llama-server does: decoded from what it received, or, to
    stand for a server that received something else, `echo`."""

    echo: str | None = None

    def complete(self, token_ids, *args, **kwargs):
        result = super().complete(token_ids, *args, **kwargs)
        received = self.echo if self.echo is not None else self.tokenizer.decode(token_ids)
        result.response = {**result.response, "prompt": received}
        return result


@pytest.mark.parametrize("echo", [None, "A prompt the server made up."])
def test_the_prompt_echo_check_compares_the_servers_decoding(fetched_root, tiny_tokenizer, echo):
    backend = Echoing([" Hello!"], tokenizer=tiny_tokenizer)
    backend.echo = echo
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("text")
    record = session.say("Hi").record
    check = record["checks"]["prompt_echo"]
    if echo is None:
        assert check == {"ok": True} and "check_failed:prompt_echo" not in record["flags"]
    else:
        assert check["ok"] is False and check["server"] == echo
        assert check["client"] == tiny_tokenizer.decode(record["prompt_ids"])
        assert "check_failed:prompt_echo" in record["flags"]
