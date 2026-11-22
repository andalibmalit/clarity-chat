"""v0.4.1 A1: the embedder, bge-m3 on a second llama-server, with llama-server faked.

Decisions 3 to 6 and 9 of notes/plans/v0.3.3.md: the flag table is a constant, never
server.toml's or a profile's; the start is checked (one slot, n_ctx 8,192, the weights on the
GPU); each vector is checked; a failure stops the embedder and the next need starts it
again. The real server's behaviour behind each fake reply is gated live in
tests/test_embed.py (`-m embed`): one input per request (T2), the token count (T2's
prompt_tokens check), and a zero vector on a failed lookup (its norm check). Made-up text.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest

from clarity_chat.backends import llama_cpp
from clarity_chat.documents import Embedder, EmbedderFailed, TooLong
from clarity_chat.documents import embedder as embedder_module
from clarity_chat.profiles import ConfigError, load_server_config
from clarity_chat.render import Tokenizer
from tests.conftest import fake_fetch
from tests.fake_llama import DEVICE_LINES, Running, dump, startup
from tests.test_llama_cpp import running_server

BGE_M3 = """
[bge-m3]
route = "convert"
repo = "BAAI/bge-m3"
revision = "5617a9f61b028005a4858fdac845db406aefb181"
weights = "pytorch_model.bin"
weights_sha256 = "1111111111111111111111111111111111111111111111111111111111111111"  # made up
tokenizer_repo = "BAAI/bge-m3"
tokenizer_revision = "5617a9f61b028005a4858fdac845db406aefb181"
"""
# The table in the plan's decision 4, written out here rather than read from the module.
TABLE: dict[str, Any] = {
    "embedding": True,
    "pooling": "cls",
    "embd-normalize": 2,
    "parallel": 1,
    "no-kv-unified": True,
    "ctx-size": 8192,
    "batch-size": 8192,
    "ubatch-size": 8192,
    "n-gpu-layers": "all",
    "flash-attn": "on",
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "cache-ram": 0,
    "cache-reuse": 0,
    "ctx-checkpoints": 0,
    "no-cache-idle-slots": True,
    "no-context-shift": True,
    "fit": "off",
    "threads": 4,
    "threads-batch": 4,
    "log-verbosity": 4,
    "no-jinja": True,
    "chat-template": "chatml",
}
CPU_BUFFERS = "load_tensors:   CPU_Mapped model buffer size =   1100.00 MiB\n"


def vector_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


UNIT = vector_bytes([1.0] + [0.0] * 1023)


class WordIds:
    """A made-up tokenizer: <s> (0), one ID per word, </s> (2), as XLM-R's adds them."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens, "bge-m3's inputs carry <s> and </s>"
        return [0, *(5 for _ in text.split()), 2]


Words = Tokenizer(tok=WordIds(), source={"path": "made-up"}, config={})


class Fake:
    """What the fake server did, and what it sends next."""

    def __init__(self) -> None:
        self.argv: list[list[str]] = []
        self.procs: list[Running] = []
        self.requests: list[dict[str, Any]] = []
        self.startup = startup()
        self.props: dict[str, Any] = {"total_slots": 1}
        self.n_ctx: int | None = None  # None: the size the server was started with
        self.reply: dict[str, Any] | None = None  # None: a good vector for the input


@pytest.fixture
def fake(monkeypatch) -> Fake:
    seen = Fake()

    def launch(self, argv, env):
        seen.argv.append(argv)
        self.output = [seen.startup]
        self.proc = Running()
        seen.procs.append(self.proc)
        if self.on_change:  # as LlamaServer._launch tells it, before the model loads
            self.on_change(
                {
                    "port": self.port,
                    "pid": 4242,
                    "api_key": self.api_key,
                    "model_path": str(self.gguf),
                }
            )
        return None

    def props(self):
        n_ctx = seen.n_ctx or self.config.ctx_size
        return {
            "model_path": str(self.gguf),
            "default_generation_settings": {"n_ctx": n_ctx},
            "build_info": "b1-" + self.config.llama_cpp_commit[:7],
            **seen.props,
        }

    def post(url, headers, json, **kwargs):  # noqa: A002 (httpx's own name)
        seen.requests.append({"url": url, "headers": headers, "json": json})
        if seen.procs[-1].poll() is not None:
            raise httpx.ConnectError("[Errno 61] Connection refused")
        sent = sum(len(x) for x in json["input"])
        body = seen.reply or {
            "data": [{"index": 0, "embedding": base64.b64encode(UNIT).decode()}],
            "usage": {"prompt_tokens": sent},
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    pinned = load_server_config(Path(__file__).resolve().parents[1]).llama_cpp_commit
    monkeypatch.setattr(llama_cpp, "llama_config_files", lambda: [])
    monkeypatch.setattr(llama_cpp, "git_state", lambda d: {"commit": pinned, "dirty": False})
    monkeypatch.setattr(llama_cpp, "server_binary", lambda d: Path("llama-server"))
    monkeypatch.setattr(llama_cpp.LlamaServer, "_launch", launch)
    monkeypatch.setattr(llama_cpp.LlamaServer, "get_props", props)
    monkeypatch.setattr(embedder_module.httpx, "post", post)
    return seen


@pytest.fixture
def m3_root(root) -> Path:
    """A root whose models.toml has bge-m3, fetched (a made-up GGUF and lock entry)."""
    return m3_fetched(root)


def m3_fetched(root: Path) -> Path:
    with (root / "models.toml").open("a", encoding="utf-8") as f:
        f.write(BGE_M3)
    fake_fetch(root, "bge-m3")
    return root


def made(root: Path, flags: dict[str, Any] | None = None) -> Embedder:
    config = load_server_config(root)
    config = dataclasses.replace(config, flags={**config.flags, **(flags or {})})
    return Embedder(root, config, root / "logs", lambda repo, revision: Words)


def as_flags(argv: list[str]) -> dict[str, Any]:
    """argv's flags between --model's value and --host, as a table (a bare flag is True)."""
    flags, i = {}, argv.index("--model") + 2
    while argv[i] != "--host":
        name = argv[i].removeprefix("--")
        bare = i + 1 == len(argv) or argv[i + 1].startswith("--")
        flags[name] = True if bare else argv[i + 1]
        i += 1 if bare else 2
    return flags


# --- the launch --------------------------------------------------------------------------


def test_the_flags_are_the_plan_s_table_and_nothing_else(m3_root, fake):
    assert embedder_module.FLAGS == TABLE
    # server.toml's own values, and a profile's ctx_size, never reach it (decision 4).
    made(m3_root, {"ctx-size": 2048, "threads": 8, "ubatch-size": 512}).vector([0, 5, 2])
    expected = {k: True if v is True else str(v) for k, v in TABLE.items()}
    assert as_flags(fake.argv[0]) == expected
    assert fake.argv[0][-6:] == [
        "--host",
        "127.0.0.1",
        "--port",
        fake.argv[0][-3],
        "--no-slots",
        "--no-webui",
    ]


def test_bge_m3_not_fetched_is_refused_before_any_process(root, fake):
    with (root / "models.toml").open("a", encoding="utf-8") as f:
        f.write(BGE_M3)
    with pytest.raises(ConfigError, match="'bge-m3' is not fetched"):
        made(root).vector([0, 5, 2])
    assert fake.argv == []


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"props": {"total_slots": 2}}, "2 slots, not 1"),
        ({"n_ctx": 4096}, "n_ctx is 4096, expected 8192"),
        ({"startup": DEVICE_LINES + dump() + CPU_BUFFERS}, "put the model on the CPU"),
    ],
)
def test_a_start_that_fails_a_check_is_refused_and_stopped(m3_root, fake, change, reason):
    for name, value in change.items():
        setattr(fake, name, value)
    with pytest.raises(EmbedderFailed, match=reason):
        made(m3_root).vector([0, 5, 2])
    assert fake.procs[-1].terminated
    assert fake.requests == []


def test_an_inherited_ggml_variable_refuses_the_start(m3_root, fake, monkeypatch):
    """Checkpoint 1's M2: llama-server keeps GGML_* and LLAMA_* variables, and some change how
    it computes (GGML_METAL_BF16_DISABLE, and bge-m3 is a bf16 GGUF), which the setup doesn't
    record. The start is refused with the chat server's existing warning, naming it."""
    monkeypatch.setenv("GGML_METAL_BF16_DISABLE", "1")
    with pytest.raises(EmbedderFailed) as refused:
        made(m3_root).vector([0, 5, 2])
    # The embedder's own words (the fix review's S3): the chat server's warning says the
    # values are logged and may be kept, which isn't so for the embedder.
    assert str(refused.value) == (
        "The embedder didn't start: llama-server would inherit GGML_METAL_BF16_DISABLE from "
        "your environment, which can change the vectors. Unset it."
    )
    assert fake.procs[-1].terminated


def test_a_dirty_llama_cpp_checkout_refuses_the_start(m3_root, fake, monkeypatch):
    pinned = load_server_config(m3_root).llama_cpp_commit
    monkeypatch.setattr(llama_cpp, "git_state", lambda d: {"commit": pinned, "dirty": True})
    with pytest.raises(EmbedderFailed) as refused:
        made(m3_root).vector([0, 5, 2])
    assert str(refused.value) == (
        "The embedder didn't start: The llama.cpp checkout has uncommitted changes. Commit or "
        "discard them and rebuild llama-server."
    )
    assert fake.procs[-1].terminated


def test_a_start_warning_of_another_kind_refuses_the_start_too(m3_root, fake, monkeypatch):
    """The main session's follow-up to S3: the embedder words the two warnings it knows, and
    any other that start_warnings gains later still refuses the start, as itself."""
    monkeypatch.setattr(llama_cpp, "start_warnings", lambda checkout, kept: ["A made-up one."])
    with pytest.raises(EmbedderFailed) as refused:
        made(m3_root).vector([0, 5, 2])
    assert str(refused.value) == "The embedder didn't start: A made-up one."
    assert fake.procs[-1].terminated


@pytest.mark.parametrize("backend", ["transformers", "llama.cpp"])
def test_a_profile_can_t_chat_with_the_embedder(m3_root, backend):
    """The fix review's M1: a chat on bge-m3 would load its pickle on Transformers without the
    hash check (a third load the security review didn't cover), and chat with an embedding
    model on llama.cpp. Refused before any weights are read, on either backend."""
    from clarity_chat.session import Session
    from tests.conftest import CHAT_PROFILE

    text = CHAT_PROFILE.replace('model = "tiny"', 'model = "bge-m3"')
    text = text.replace('backend = "llama.cpp"', f'backend = "{backend}"')
    (m3_root / "profiles" / "on-bge.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="bge-m3 is meaning search's model"):
        Session(m3_root).profile_for("on-bge")


def test_fetched_is_false_only_without_an_entry_or_its_file(m3_root):
    """The main session's ruling on B1: a changed file is fetched (its remedy is `clarity
    verify`, which the start's refusal names); state 5 is no entry, or no file."""
    gguf = m3_root / "models" / "bge-m3" / "bge-m3-bf16.gguf"
    assert made(m3_root).fetched()
    os.utime(gguf, ns=(1, 1))  # size or time changed since the fetch
    assert made(m3_root).fetched()
    gguf.unlink()
    assert not made(m3_root).fetched()


def test_no_bge_m3_entry_is_not_fetched(root):
    assert "bge-m3" not in (root / "models.toml").read_text(encoding="utf-8")
    assert not made(root).fetched()


def test_a_changed_gguf_at_send_says_the_lock_s_sentence_not_state_5(fetched_root, chat_tokenizer):
    from clarity_chat.documents import SearchRefused, library_add, reading
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend
    from tests.test_meaning import ASKED, GLOSSARY, MEANING, profile, raised

    with (fetched_root / "models.toml").open("a", encoding="utf-8") as f:
        f.write(BGE_M3)
    fake_fetch(fetched_root, "bge-m3")
    os.utime(fetched_root / "models" / "bge-m3" / "bge-m3-bf16.gguf", ns=(1, 1))
    reading._last = None
    profile(fetched_root, "meaning", MEANING)
    library_add(fetched_root, GLOSSARY.encode(), "notes.md")
    session = Session(
        fetched_root,
        backend=FakeBackend([" R0."], tokenizer=chat_tokenizer),
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=made(fetched_root),
    )
    session.new_conversation("meaning")
    refused = raised(lambda: session.say(ASKED))
    assert isinstance(refused, SearchRefused), repr(refused)
    assert refused.problems[0]["state"] == "embedder", refused.problems
    assert "changed since it was fetched" in refused.terminal
    assert "isn't fetched" not in refused.terminal


# --- each vector -----------------------------------------------------------------------


def test_one_input_per_request_as_ids_and_the_vector_comes_back(m3_root, fake):
    embedder = made(m3_root)
    assert embedder.vector([0, 5, 5, 2]) == UNIT
    [request] = fake.requests
    assert request["url"].endswith("/v1/embeddings")
    assert request["headers"]["Authorization"].startswith("Bearer ")
    assert request["json"] == {
        "input": [[0, 5, 5, 2]],
        "encoding_format": "base64",
        "embd_normalize": 2,
    }


def body(values: list[float] | bytes, tokens: int = 3, vectors: int = 1) -> dict[str, Any]:
    raw = values if isinstance(values, bytes) else vector_bytes(values)
    data = [{"index": i, "embedding": base64.b64encode(raw).decode()} for i in range(vectors)]
    return {"data": data, "usage": {"prompt_tokens": tokens}}


@pytest.mark.parametrize(
    "reply, reason",
    [
        (body([0.0] * 1024), "length 0, not 1"),  # llama-server's answer to a failed lookup
        (body([math.nan] + [0.0] * 1023), "isn't a finite number"),
        (body([1.0] + [0.0] * 1022), "1023 values, not 1024"),
        (body([0.5] + [0.0] * 1023), "length 0.5, not 1"),
        (body(UNIT, tokens=4), "4 tokens counted, 3 sent"),
        (body(UNIT, vectors=2), "2 vectors for one input"),
        ({"error": {"message": "made-up"}}, "'data'"),
    ],
)
def test_a_vector_that_fails_a_check_is_refused_and_the_next_need_starts_again(
    m3_root, fake, reply, reason
):
    embedder = made(m3_root)
    fake.reply = reply
    with pytest.raises(EmbedderFailed, match=reason):
        embedder.vector([0, 5, 2])
    assert fake.procs[-1].terminated
    fake.reply = None
    assert embedder.vector([0, 5, 2]) == UNIT
    assert len(fake.procs) == 2


def test_a_dead_process_fails_that_search_and_the_next_need_starts_it_again(m3_root, fake):
    embedder = made(m3_root)
    assert embedder.vector([0, 5, 2]) == UNIT
    fake.procs[-1].returncode = -9  # killed between two vectors of one search
    with pytest.raises(EmbedderFailed) as refused:
        embedder.vector([0, 5, 2])
    # Folded into state 6's second sentence (the main session's B1 ruling: no third one).
    assert str(refused.value) == (
        "The embedder sent a vector Clarity can't use: its process had stopped; the next "
        "search starts it again."
    )
    assert embedder.vector([0, 5, 2]) == UNIT
    assert len(fake.procs) == 2


def test_a_text_past_8192_tokens_is_refused_never_cut(m3_root):
    embedder = made(m3_root)
    assert len(embedder.ids("word " * 8190)) == 8192  # with <s> and </s>: accepted
    with pytest.raises(TooLong) as refused:
        embedder.ids("word " * 8191)
    assert refused.value.tokens == 8193


# --- the setup (decision 9) --------------------------------------------------------------


def test_the_engine_holds_the_setup_s_facts_and_their_hash(m3_root, fake):
    engine = made(m3_root).start()
    assert list(engine) == [
        "name",
        "model",
        "gguf_sha256",
        "tokenizer_repo",
        "tokenizer_revision",
        "tokenizers",
        "llama_cpp",
        "flags",
        "input_rule",
        "device",
        "machine",
        "setup",
    ]
    assert (engine["name"], engine["model"], engine["device"]) == (
        "llama-server-embedding",
        "bge-m3",
        "MTL0",
    )
    assert engine["gguf_sha256"] == "0" * 64  # fake_fetch's lock entry
    assert (engine["flags"], engine["input_rule"]) == (TABLE, 1)
    assert list(engine["machine"]) == ["os", "build", "chip"]
    rest = {k: v for k, v in engine.items() if k != "setup"}
    text = json.dumps(rest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert engine["setup"] == embedder_module.setup_of(rest)
    assert engine["setup"] == hashlib.sha256(text.encode()).hexdigest()
    assert embedder_module.setup_of({**rest, "device": "CPU"}) != engine["setup"]


# --- the session (decision 3) ------------------------------------------------------------


def test_the_session_makes_it_on_first_need_and_stops_it_at_close(m3_root, fake):
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    session = Session(m3_root, backend=FakeBackend([]), tokenizer_loader=lambda r, v: Words)
    assert fake.procs == []  # a session alone starts nothing
    assert session.embedder().vector([0, 5, 2]) == UNIT
    session.close()
    assert fake.procs[-1].terminated


def test_the_session_records_the_embedder_s_process_for_crash_recovery(m3_root, fake):
    """D4 (decision 16): the embedder's port, PID, key and model go where the chat server's
    do, under "embedder", when it starts, and None when it stops."""
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    recorded: list[tuple[dict[str, Any] | None, str]] = []
    session = Session(
        m3_root,
        backend=FakeBackend([]),
        tokenizer_loader=lambda r, v: Words,
        on_server=lambda info, kind="server": recorded.append((info, kind)),
    )
    session.embedder().vector([0, 5, 2])
    assert [kind for _, kind in recorded] == ["embedder"]
    info = recorded[0][0]
    assert info is not None and info["model_path"].endswith("bge-m3-bf16.gguf")
    session.close()
    assert recorded[-1] == (None, "embedder")


class Untouchable:
    """An embedder that fails the test if anything but closing asks it for anything."""

    def close(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the embedder was used: {name}")


@pytest.mark.parametrize("name", ["say", "search"])
def test_a_words_chat_resume_replay_and_export_never_use_it(
    fetched_root, chat_tokenizer, tmp_path, name
):
    from clarity_chat.commands import Commands
    from clarity_chat.export import export_log
    from clarity_chat.logs import read_records
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend
    from tests.test_class import write_chat

    def session(backend: FakeBackend) -> Session:
        return Session(
            fetched_root,
            backend=backend,
            tokenizer_loader=lambda r, v: chat_tokenizer,
            embedder=Untouchable(),  # ty: ignore[invalid-argument-type]
        )

    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    chat = session(backend)
    write_chat(name, Commands(chat), backend, tmp_path)  # "search": /search, then words
    assert chat.conversation is not None
    log = chat.conversation.log.path
    chat.new_conversation("chat")
    chat.resume(log)
    export_log(chat, log)
    chat.close()
    for call in [r for r in read_records(log) if r["type"] == "call" and r.get("turn_id")]:
        fresh = session(FakeBackend([call["output_text"]], tokenizer=chat_tokenizer))
        assert fresh.replay(log, call["call_id"])["identical"]
        fresh.close()


# --- The whole-branch review's fix batch


def test_ids_use_the_tokenizer_the_start_logged(m3_root, fake):
    """Could 1: the entry start() read tokenizes every part, so a models.toml changed during a
    long session (a git pull) can't tokenize with another revision than `engine` logged."""
    asked: list[tuple[str, str]] = []

    def load(repo: str, revision: str):
        asked.append((repo, revision))
        return Words

    config = load_server_config(m3_root)
    embedder = Embedder(m3_root, config, m3_root / "logs", load)
    engine = embedder.start()
    models = m3_root / "models.toml"
    models.write_text(
        models.read_text(encoding="utf-8").replace(engine["tokenizer_revision"], "f" * 40),
        encoding="utf-8",
    )
    embedder.ids("Made-up words.")
    assert asked == [(engine["tokenizer_repo"], engine["tokenizer_revision"])]


def test_the_logged_device_is_the_one_the_start_used(m3_root, fake):
    """#5: not a constant."""
    fake.startup = fake.startup.replace("MTL0", "MTL1")
    assert made(m3_root).start()["device"] == "MTL1"


def test_a_reply_whose_one_vector_has_another_index_is_refused(m3_root, fake):
    """Could 4 (#3): one vector, but at index 1."""
    reply = body(UNIT)
    reply["data"][0]["index"] = 1
    fake.reply = reply
    with pytest.raises(EmbedderFailed, match="1 vectors for one input"):
        made(m3_root).vector([0, 5, 2])


def test_close_drops_the_entry_so_the_next_start_reads_models_toml_again(m3_root, fake):
    """The fix review's #23: the next start reads the changed entry, which the lock then
    refuses (a cached entry would start with the old revision)."""
    embedder = made(m3_root)
    first = embedder.start()["tokenizer_revision"]
    embedder.close()
    models = m3_root / "models.toml"
    line = f'tokenizer_revision = "{first}"'  # the tokenizer's line only: the lock pins the rest
    models.write_text(
        models.read_text(encoding="utf-8").replace(line, f'tokenizer_revision = "{"f" * 40}"'),
        "utf-8",
    )
    with pytest.raises(ConfigError, match="with tokenizer_revision"):
        embedder.start()


def test_a_chat_that_doesn_t_search_by_meaning_stops_the_embedder(
    fetched_root, fake, chat_tokenizer, monkeypatch
):
    """The fix review's F1 (ruled): following the memory refusal's way out, words and a new
    chat, frees the embedder's memory, so the chat server that didn't fit beside it starts.
    The embedder starts again at the next need."""
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend
    from tests.test_llama_cpp import CHAT_9000, startup_using

    m3_root = m3_fetched(fetched_root)  # the chat model fetched too
    fake.startup = startup_using(10700, 1100, 0, 700)  # the embedder: 1,800 MiB
    backend = FakeBackend([" R."] * 4, tokenizer=chat_tokenizer)
    session = Session(m3_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    session.embedder().vector([0, 5, 2])
    embedder = fake.procs[-1]
    conversation = session.new_conversation("chat")  # conftest's chat profile: words
    assert embedder.terminated, "the embedder still runs"
    assert not isinstance(running_server(monkeypatch, m3_root, CHAT_9000), Exception)
    session.embedder().vector([0, 5, 2])  # the next need starts it again
    assert not fake.procs[-1].terminated
    session.new_conversation("chat")
    session.embedder().vector([0, 5, 2])
    again = fake.procs[-1]
    session.resume(conversation.log.path)  # a resumed words chat frees it too
    assert again.terminated
