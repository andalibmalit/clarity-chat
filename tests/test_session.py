from __future__ import annotations

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
