"""v0.4.4 B3a: `/set ctx_size` in the terminal (decisions 5 to 8, 10 to 12; states 20 to
28, 30 and 31).

A new size starts the chat's model again through `start_model`; the chat's size changes
only once that start worked (G3), so a refused size leaves the old one running. The size
is a chat key: kept in `chat_settings`, never in a call's `settings`, resumed by decision
11 and reset to its base (decision 16). The llama.cpp tests run the real backend on a fake
llama-server (tests/fake_llama.py) whose made-up startup log grows the context cache with
the size; the live tier's G3 test (tests/test_live.py) runs the same path on SmolLM2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clarity_chat.backends.llama_cpp import LlamaCppBackend, LlamaServer
from clarity_chat.cli import ChatLoop
from clarity_chat.commands import Commands, settings_view
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError, Setting, load_server_config
from clarity_chat.resume import rebuild, what_differs
from clarity_chat.session import Session
from tests.fake_llama import NODES, PREFIX, fake_llama, loaded, loading, reservation
from tests.fakes import FakeBackend
from tests.test_golden import LOGS
from tests.test_meaning import raised


def sized(limit: int = 25559):
    """A startup log for a size: 4,610 MiB of model, 576 MiB of context cache per 8,192
    tokens, 250 for computing, under a Metal limit of `limit` MiB. Over 8,192 tokens it
    prints llama.cpp's line for a size past the trained maximum."""

    def printed(ctx_size: int) -> str:
        over = (
            f"llama_context: n_ctx_seq ({ctx_size}) exceeds the training context of the "
            "model (8192) - the full capacity of the model will not be utilized\n"
            if ctx_size > 8192
            else ""
        )
        mib = 576 * ctx_size / 8192
        kv = f"{PREFIX}llama_kv_cache:       MTL0 KV buffer size = {mib:.2f} MiB\n"
        return loading().replace("25558 MiB free", f"{limit} MiB free").replace(
            "256.50", "4610.00"
        ) + loaded(reservation(NODES, mtl0=250.0), kv=over + kv)

    return printed


@pytest.fixture
def llama(monkeypatch, fetched_root, tiny_tokenizer):
    """A text chat on the real llama.cpp backend at server.toml's 8,192: (session, what the
    fake server saw). `seen["startups"]` sets the startup log."""
    startups: dict[str, Any] = {"tiny": sized()}
    seen = fake_llama(monkeypatch, fetched_root, startups, tiny_tokenizer, replies=[])
    seen["startups"] = startups
    backend = LlamaCppBackend(load_server_config(fetched_root), fetched_root / "logs")
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("text")
    yield session, seen
    session.close()


@pytest.fixture
def fake(fetched_root, tiny_tokenizer):
    """A text chat on FakeBackend at server.toml's 8,192: (session, backend)."""
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("text")
    return session, backend


def records(session: Session, kind: str) -> list[dict[str, Any]]:
    conversation = session.conversation
    assert conversation is not None
    return [r for r in read_records(conversation.log.path) if r["type"] == kind]


def size_of(session: Session) -> Setting:
    conversation = session.conversation
    assert conversation is not None
    return conversation.ctx_size


def flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def refusal_of(action) -> BaseException:
    refused = raised(action)
    assert refused is not None, "it wasn't refused"
    return refused


MEMORY_16384 = (
    "At 16,384 tokens, the model uses 6,012 MiB of GPU memory (4,610 for the model, 1,152 "
    "for the context cache, 250 for computing), out of Metal's recommended maximum for this GPU, "
    "25,559 MiB."
)  # state 20, as the plan has it


# --- a new size, the same size, a refused size (decision 6; states 20 to 24) ------------


def test_a_new_size_starts_a_new_server_at_it(llama):
    session, seen = llama
    before = len(seen["procs"])
    refused = raised(lambda: session.set_setting("ctx_size", "16384"))
    assert refused is None, f"/set ctx_size was refused: {refused}"
    assert len(seen["procs"]) == before + 1, "no new llama-server started"
    assert size_of(session) == Setting(16384, "chat")
    call = session.say("Hi").record
    assert call["n_ctx"] == 16384
    starts = {r["server_id"]: r for r in records(session, "server_start")}
    server = starts[call["backend"]["server_id"]]
    assert flag(server["argv"], "--ctx-size") == "16384"
    assert "ctx_size" not in call["settings"], "a chat key went into a call's settings"
    (setting,) = records(session, "settings")
    assert setting["name"] == "ctx_size" and setting["chat_settings"] == {"ctx_size": 16384}


def test_the_same_size_doesn_t_start_the_server_again(llama):
    session, seen = llama
    before = len(seen["procs"])
    refused = raised(lambda: session.set_setting("ctx_size", "8192"))
    assert refused is None, f"/set ctx_size was refused: {refused}"
    assert len(seen["procs"]) == before, "the same size started a new server"
    assert size_of(session) == Setting(8192, "chat")


def test_a_size_over_metal_s_limit_is_refused_and_the_old_size_runs_on(llama):
    session, seen = llama
    seen["startups"]["tiny"] = sized(limit=6000)  # 8,192 needs 5,436 MiB; 16,384 needs 6,012
    refused = refusal_of(lambda: session.set_setting("ctx_size", "16384"))
    assert isinstance(refused, ConfigError), repr(refused)
    assert str(refused) == (
        "At ctx_size 16384, the model needs 6,012 MiB of GPU memory (4,610 for the model, "
        "1,152 for the context cache, 250 for computing), more than Metal's recommended "
        "maximum for this GPU, 6,000 MiB. Lower ctx_size: the context cache grows with it."
    )
    assert size_of(session) == Setting(8192, "server.toml")
    assert records(session, "settings") == [], "a refused size was logged as the chat's"
    call = session.say("Hi").record
    assert call["n_ctx"] == 8192 and call["stop_type"] == "eos"


def test_a_size_past_the_trained_maximum_is_refused_and_the_old_size_runs_on(llama, monkeypatch):
    session, _ = llama
    monkeypatch.setattr(  # llama-server quietly runs 8,192, and /props says so
        LlamaServer,
        "check_props",
        lambda self, props: ["n_ctx differs"] if self.config.ctx_size > 8192 else [],
    )
    refused = refusal_of(lambda: session.set_setting("ctx_size", "16384"))
    assert str(refused).endswith("Set ctx_size to 8192 or less."), str(refused)
    assert size_of(session) == Setting(8192, "server.toml")
    assert session.say("Hi").record["n_ctx"] == 8192


def test_a_refused_size_on_a_backend_that_sets_it_first_leaves_the_old_size(fake):
    """S12: the Transformers backend sets its size before its positions check, so a
    refused size must be put back: the next call's n_ctx is the old size."""
    session, backend = fake
    started = backend.start

    def checks_positions(model, weights, ctx_size, on_start=None, **kwargs):
        if ctx_size > 8192:
            backend.ctx_sizes.append(ctx_size)  # set, then checked, as Transformers does
            raise ConfigError("tiny handles 8192 positions; ctx_size 16384 is more.")
        if on_start:  # called once the checks pass
            on_start()
        return started(model, weights, ctx_size, **kwargs)

    backend.start = checks_positions
    refused = refusal_of(lambda: session.set_setting("ctx_size", "16384"))
    assert "8192 positions" in str(refused)
    assert size_of(session) == Setting(8192, "server.toml")
    assert session.say("Hi").record["n_ctx"] == 8192


def test_the_transformers_backend_takes_a_new_size_and_puts_a_refused_one_back(tmp_path):
    """S12 on the real TransformersBackend's start, with its model already loaded: a new
    size within the model's positions is taken without a reload; one past them is
    refused, and the restore's start puts the old size back."""
    from types import SimpleNamespace

    from clarity_chat.backends.transformers_hf import TransformersBackend

    backend = TransformersBackend("cpu", 8192)
    backend.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384))
    backend.weights = tmp_path
    model: Any = SimpleNamespace(name="tiny")
    assert backend.start(model, tmp_path, 16384) is None
    assert backend.n_ctx == 16384
    assert raised(lambda: backend.start(model, tmp_path, 32768)) is not None
    backend.start(model, tmp_path, 8192)  # what _restore does
    assert backend.n_ctx == 8192


def test_the_transformers_backend_reads_gemmas_positions_from_its_text_config(tmp_path):
    """Gemma 4's config has no max_position_embeddings at its top level: the check reads
    text_config's, so a size past it is refused, not passed as if there were no limit."""
    from types import SimpleNamespace

    pytest.importorskip("torch")
    from clarity_chat.backends.transformers_hf import TransformersBackend
    from tests.test_steering import gemma_config

    backend = TransformersBackend("cpu", 8192)
    backend.model = SimpleNamespace(config=gemma_config(positions=16384))
    backend.weights = tmp_path
    model: Any = SimpleNamespace(name="tiny")
    assert backend.start(model, tmp_path, 16384) is None
    refused = raised(lambda: backend.start(model, tmp_path, 32768))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused) == "tiny handles 16384 positions, less than ctx_size 32768."


def test_a_failed_restore_says_so_and_no_chat_runs(fake):
    """State 31: the new size fails, and so does the old one."""
    session, backend = fake

    def fails(model, weights, ctx_size, on_start=None, **kwargs):
        if on_start:  # as llama.cpp's start: its checks pass, then the old one stops
            on_start()
        raise ConfigError(f"ctx_size {ctx_size} failed.")

    backend.start = fails
    refused = refusal_of(lambda: session.set_setting("ctx_size", "16384"))
    assert "Restarting tiny for the current conversation failed too" in str(refused), str(refused)
    assert session.conversation is None


def test_a_bad_size_is_refused_as_the_profile_refuses_it(fake):
    session, backend = fake
    for text in ("0", "-5", "1.5", '"big"'):
        refused = refusal_of(lambda text=text: session.set_setting("ctx_size", text))
        assert str(refused).startswith("/set: ctx_size must be a positive integer"), str(refused)
    assert backend.ctx_sizes == [8192]


# --- reset and the base (decision 16; state 26) -------------------------------------------


def test_reset_starts_the_model_again_at_the_base(fake):
    session, backend = fake
    assert raised(lambda: session.set_setting("ctx_size", "16384")) is None
    assert raised(lambda: session.set_setting("ctx_size", "")) is None
    assert backend.ctx_sizes == [8192, 16384, 8192]
    assert size_of(session) == Setting(8192, "server.toml")
    assert session.say("Hi").record["n_ctx"] == 8192
    last = records(session, "settings")[-1]
    assert last["value"] is None and last["chat_settings"] == {}


def test_a_profile_s_size_is_the_base_of_a_new_chat(fetched_root, tiny_tokenizer):
    text = (fetched_root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (fetched_root / "profiles" / "big.toml").write_text(f"ctx_size = 4096\n{text}", "utf-8")
    backend = FakeBackend([" R."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("big")
    assert raised(lambda: session.set_setting("ctx_size", "8192")) is None
    assert raised(lambda: session.set_setting("ctx_size", "")) is None
    assert size_of(session) == Setting(4096, "profile")


# --- resume (decision 11) ---------------------------------------------------------------


def resume(root: Path, tokenizer, path: Path) -> tuple[Session, FakeBackend]:
    backend = FakeBackend([" R."], tokenizer=tokenizer)
    fresh = Session(root, backend=backend, tokenizer_loader=lambda r, v: tokenizer)
    fresh.resume(path)
    return fresh, backend


def test_a_resumed_chat_keeps_its_set_size(fake, fetched_root, tiny_tokenizer):
    session, _ = fake
    assert raised(lambda: session.set_setting("ctx_size", "16384")) is None
    session.say("Hi")
    path = session.conversation.log.path
    fresh, backend = resume(fetched_root, tiny_tokenizer, path)
    assert backend.ctx_sizes == [16384]
    assert size_of(fresh) == Setting(16384, "chat")
    (resumed,) = [r for r in read_records(path) if r["type"] == "resume"]
    assert resumed["ctx_size"] == {"value": 16384, "origin": "chat"}
    assert raised(lambda: fresh.set_setting("ctx_size", "")) is None  # back to the base
    assert size_of(fresh) == Setting(8192, "log")
    assert backend.ctx_sizes == [16384, 8192]


def test_only_a_log_with_no_size_warns_on_resume(fake, fetched_root, tiny_tokenizer):
    """v0.4.4 QA: a chat that set its size resumed with "The log doesn't say its context
    size: using 4096 (chat)", though its `settings` record and calls hold 4,096. The size
    set in the chat is in the log; only a size that came from today's files is a guess."""
    session, _ = fake
    assert raised(lambda: session.set_setting("ctx_size", "4096")) is None
    session.say("Hi")
    path = session.conversation.log.path
    resume(fetched_root, tiny_tokenizer, path)
    (resumed,) = [r for r in read_records(path) if r["type"] == "resume"]
    assert resumed["ctx_size"] == {"value": 4096, "origin": "chat"}
    said = [item["text"] for item in what_differs(resumed)]
    assert not [t for t in said if "context size" in t], said
    guessed = {**resumed, "ctx_size": {"value": 8192, "origin": "server.toml"}}
    said = [item["text"] for item in what_differs(guessed)]
    assert "The log doesn't say its context size: using 8192 (server.toml)." in said


def test_set_call_reset_resume_comes_back_at_the_base(fake, fetched_root, tiny_tokenizer):
    """The case today's rule gets wrong: the last call ran at 16,384, but the chat went
    back to 8,192 after it."""
    session, _ = fake
    assert raised(lambda: session.set_setting("ctx_size", "16384")) is None
    assert session.say("Hi").record["n_ctx"] == 16384
    assert raised(lambda: session.set_setting("ctx_size", "")) is None
    path = session.conversation.log.path
    fresh, backend = resume(fetched_root, tiny_tokenizer, path)
    assert backend.ctx_sizes == [8192]
    assert size_of(fresh) == Setting(8192, "log")


def test_a_resumed_chat_without_a_set_size_runs_at_the_logged_one(
    fake, fetched_root, tiny_tokenizer
):
    session, _ = fake
    session.say("Hi")
    path = session.conversation.log.path
    fresh, backend = resume(fetched_root, tiny_tokenizer, path)
    assert backend.ctx_sizes == [8192]
    assert size_of(fresh) == Setting(8192, "log")


def test_a_log_older_than_the_conversation_s_size_resumes_at_its_last_call_s(
    fake, fetched_root, tiny_tokenizer
):
    """A log from before the `conversation` record had `ctx_size` (v0.1.0's golden log):
    the last call's n_ctx, origin log."""
    import json

    session, _ = fake
    session.say("Hi")
    path = session.conversation.log.path
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["type"] == "conversation"
    del first["ctx_size"]
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")
    fresh, backend = resume(fetched_root, tiny_tokenizer, path)
    assert backend.ctx_sizes == [8192]
    assert size_of(fresh) == Setting(8192, "log")


@pytest.mark.parametrize("log", LOGS, ids=lambda p: p.stem)
def test_the_new_resume_rule_and_today_s_agree_on_every_golden_log(log):
    """Advisor D10: decision 11's rule (the chat's set size; else the conversation
    record's; else the last call's) and the rule before it (the last call's; else the
    conversation record's) give the same size on every released version's log. Replays
    write `replay` records, not calls, so the two never differed before B3a. A guard,
    green before and after."""
    records = list(read_records(log))
    state = rebuild(records, str(log))
    last = state.last_call
    logged = (state.start.get("ctx_size") or {}).get("value")
    before = last["n_ctx"] if last else logged
    chat = state.chat_settings.get("ctx_size")
    after = chat if chat is not None else logged if logged is not None else last and last["n_ctx"]
    assert after == before


# --- replay at another size (decision 12; state 27) ---------------------------------------


def test_replay_at_another_size_says_to_set_it_and_replays_after(fake):
    session, backend = fake
    backend.replies = [" Same.", " Same."]  # the call, then its replay
    call_id = session.say("Hi").record["call_id"]
    path = session.conversation.log.path
    assert raised(lambda: session.set_setting("ctx_size", "16384")) is None
    refused = refusal_of(lambda: session.replay(path, call_id))
    assert str(refused) == (
        f"Call {call_id} ran with a context of 8,192 tokens, but this chat uses 16,384. Set "
        "ctx_size to 8192 to replay it here."
    )
    assert raised(lambda: session.set_setting("ctx_size", "8192")) is None
    assert session.replay(path, call_id)["identical"]


# --- the context-full advice (state 30) -----------------------------------------------------


def test_a_message_that_doesn_t_fit_names_ctx_size(fake):
    session, _ = fake
    assert raised(lambda: session.set_setting("ctx_size", "16")) is None  # accepted (D4)
    refused = refusal_of(lambda: session.say("A made-up message that is far too long here."))
    assert str(refused).endswith(
        "Shorten your message, start a new chat, lower max_tokens, or raise ctx_size."
    ), str(refused)


# --- /settings' memory, the status events and the terminal (decision 8; states 20, 21) ------


def test_settings_carry_the_memory_measured_at_the_start(llama):
    session, _ = llama
    assert raised(lambda: session.set_setting("ctx_size", "16384")) is None
    memory = settings_view(session).get("memory")
    assert memory is not None, "no memory on /settings"
    assert memory["text"] == MEMORY_16384
    assert memory["device"] == "MTL0" and memory["kv_mib"] == 1152
    assert settings_view(session)["ctx_size"] == {
        "value": 16384,
        "origin": "chat",
        "base": {"value": 8192, "origin": "server.toml"},  # B3b: the row's Back to
    }


def test_settings_carry_no_memory_without_metal_s_figure(fake):
    session, _ = fake  # FakeBackend's record has no memory, as the Transformers backend's
    view = settings_view(session)
    assert "memory" in view and view["memory"] is None


def test_settings_carry_no_memory_from_another_gpu(llama):
    """Another GPU's figures are logged, but the sentence is Metal's (state 21; checkpoint 1,
    S4 (c))."""
    session, seen = llama
    on_metal = sized()
    seen["startups"]["tiny"] = lambda ctx_size: on_metal(ctx_size).replace("MTL0", "CUDA0")
    assert raised(lambda: session.set_setting("ctx_size", "16384")) is None
    memory = (session.backend.server_record or {}).get("memory") or []
    assert [m["device"] for m in memory] == ["CUDA0"]
    assert settings_view(session)["memory"] is None


def test_set_ctx_size_sends_the_starting_status(fake):
    session, _ = fake
    events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    refused = raised(
        lambda: results.append(Commands(session).run("set", "ctx_size 16384", events.append))
    )
    assert refused is None, f"/set ctx_size was refused: {refused}"
    (result,) = results
    statuses = [e["status"] for e in events if e["event"] == "status"]
    assert statuses == ["starting", "ready"], statuses
    assert result["setting"] == {"value": 16384, "origin": "chat"}


@pytest.mark.parametrize(
    ("value", "sentence"),
    [
        ("abc", "/set: ctx_size must be a positive integer"),
        ("0", "/set: ctx_size must be a positive integer"),
        ("", "ctx_size isn't set in this chat."),
    ],
)
def test_a_size_refused_before_any_start_says_no_start(fake, value, sentence):
    """QA finding 1's sibling: a size the checks refuse starts nothing, so nothing is
    said as a start."""
    session, backend = fake
    events: list[dict[str, Any]] = []
    refused = raised(lambda: Commands(session).run("set", f"ctx_size {value}", events.append))
    assert isinstance(refused, ConfigError), f"not refused: {refused!r}"
    assert str(refused).startswith(sentence), str(refused)
    statuses = [e["status"] for e in events if e["event"] == "status"]
    assert statuses == [], "a start was said, though none ran"
    assert backend.ctx_sizes == [8192]
    assert session.say("Hi").record["n_ctx"] == 8192


def test_a_new_size_s_server_warnings_are_the_chat_s(fake):
    """After /set ctx_size the screen shows the new server's start warnings, not the first
    one's (checkpoint 1, S5)."""
    session, backend = fake
    backend.start_warnings = ["the second server's warning"]
    c = Commands(session)
    c.run("set", "ctx_size 16384", lambda e: None)
    assert c.run("state", "", lambda e: None)["warnings"] == ["the second server's warning"]
    c.run("set", "ctx_size 16384", lambda e: None)  # the same size: the server runs on
    assert c.run("state", "", lambda e: None)["warnings"] == ["the second server's warning"]
    session.say("Hello.")
    starts = records(session, "server_start")
    assert [r["warnings"] for r in starts] == [[], ["the second server's warning"]]


def test_the_terminal_says_the_memory_after_set_ctx_size(llama):
    session, _ = llama
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.handle("/set ctx_size 16384")
    assert "".join(output) == f"ctx_size = 16384 for this chat.\n{MEMORY_16384}\n"


def test_the_terminal_says_only_the_set_line_without_a_figure(fake):
    session, _ = fake
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.handle("/set ctx_size 16384")
    assert "".join(output) == "ctx_size = 16384 for this chat.\n"


def test_the_terminal_s_settings_say_the_memory_under_the_context(llama):
    session, _ = llama
    output: list[str] = []
    loop = ChatLoop(session, "text", read=lambda default: "", write=output.append)
    loop.handle("/set ctx_size 16384")
    output.clear()
    loop.handle("/settings")
    said = "".join(output)
    assert f"context 16384 tokens (chat); stop ['\\nUser:']\n  {MEMORY_16384}\n" in said, said


# --- Details (decision 10) ----------------------------------------------------------------


def test_details_show_the_call_s_own_context_with_no_origin(fetched_root, tiny_tokenizer):
    """A call ran at 8,192; the chat runs at 16,384 now. Details says the call's size and
    no origin: the chat's origin belongs to Settings, not to a past call."""
    import re

    from starlette.testclient import TestClient

    from clarity_chat import web
    from tests.test_web import ORIGIN, PORT, SECRET, command
    from tests.test_web_details import details, result

    backend = FakeBackend([" Hello!"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "text")
        call = result(client, "say", "Hi")["call"]
        result(client, "set", "ctx_size 16384")
        row = re.search(r"<dt>context</dt>(.*?)</dl>", details(client, call["call_id"]), re.S)
    assert row, "no context row"
    assert row.group(1) == '<dd>8,192</dd><dd class="o"></dd>', row.group(1)
