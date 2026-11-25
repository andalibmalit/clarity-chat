"""Live tests: the real llama-server (server.toml) and the fetched SmolLM2-135M-Instruct.

Run with `uv run pytest -m live`. Skipped if llama-server isn't built or SmolLM2 isn't fetched.
Only the `gpu`-marked test needs the model on a GPU; CI runs the rest on a Linux CPU build
(`-m "live and not gpu"`, with `n-gpu-layers = 0`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from clarity_chat.backends.llama_cpp import LlamaCppBackend, server_binary
from clarity_chat.logs import read_records
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.render import Tokenizer
from clarity_chat.replay import find_call
from clarity_chat.session import ContextError, Session
from tests import contract

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]
MODEL = "smollm2-135m-instruct"


def _available() -> str | None:
    try:
        server_binary(load_server_config(REPO).llama_cpp_dir)
        check_weights_quick(REPO, lock_entry(REPO, load_models(REPO)[MODEL]))
    except ConfigError as e:
        return str(e)
    return None


if (reason := _available()) is not None:
    pytestmark = [pytest.mark.live, pytest.mark.skip(reason=reason)]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    session = Session(REPO, logs_dir=tmp_path_factory.mktemp("logs"))
    session.new_conversation("smol-text")
    yield session
    session.close()


def all_checks_ok(record: dict) -> None:
    failed = {name: check for name, check in record["checks"].items() if not check["ok"]}
    assert not failed, failed


def test_server_start_is_logged_with_props(session):
    record = session.backend.server_record
    assert record["props"]["model_path"].endswith(f"{MODEL}-bf16.gguf")
    assert record["props"]["default_generation_settings"]["n_ctx"] == 8192
    assert record["warnings"] == []
    assert "--no-context-shift" in record["argv"]
    # Read from llama.cpp's startup log: a llama.cpp update that changes the line shows here.
    assert [t["id"] for t in record["model_file"]["eog_tokens"]] == [0, 2, 4]


@pytest.mark.gpu  # CI's CPU build runs `-m "live and not gpu"`
def test_server_start_puts_the_model_on_the_gpu(session):
    placement = session.backend.server_record["placement"]
    assert placement["devices"], f"no GPU device in use: {placement}"
    if sys.platform == "darwin":
        assert placement["devices"][0]["name"].startswith("MTL"), placement
        # llama.cpp keeps the token-embedding table on the CPU, so only some buffers are Metal.
        assert any(b["buffer"].startswith("MTL") for b in placement["model_buffers"]), placement


def test_the_server_refuses_requests_without_its_key(session):
    import httpx

    url = session.backend.server.url
    assert httpx.get(f"{url}/health", trust_env=False).status_code == 200  # public
    assert httpx.get(f"{url}/props", trust_env=False).status_code == 401
    body = {"prompt": [1, 2], "n_predict": 1}
    assert httpx.post(f"{url}/completion", json=body, trust_env=False).status_code == 401
    assert httpx.get(f"{url}/", trust_env=False).status_code != 200  # --no-webui
    assert "api_keys: (redacted by Clarity)" in session.backend.server.output_text()


def test_text_mode_reply_stops_before_the_next_user_line(session):
    session.new_conversation("smol-text")
    session.say("What is the capital of France?")
    outcome = session.say("And of Italy?")
    record = outcome.record
    all_checks_ok(record)
    assert record["stop_type"] == "word", record["output_text"]
    assert record["stopping_word"] == "\nUser:"
    assert "\nUser:" not in record["output_text"]
    assert record["tokens_evaluated"] == record["n_prompt_client"]
    assert record["timings"]["cache_n"] == 0
    assert "incomplete_output_ids" not in record["flags"]


def test_chat_mode_reply_ends_with_the_end_of_turn_token(session):
    session.new_conversation("smol-chat")  # same model: the server keeps running
    outcome = session.say("Name one colour.")
    record = outcome.record
    all_checks_ok(record)
    assert record["prompt_text"].endswith("<|im_start|>assistant\n")
    assert record["bos"]["prepended"] is False
    assert record["template"]["sha256"]
    assert record["stop_type"] == "eos", record["output_text"]
    im_end = session.tokenizer.tok.convert_tokens_to_ids("<|im_end|>")
    assert record["output_ids"][-1] == im_end
    follow_up = session.say("And another?").record
    all_checks_ok(follow_up)
    assert f"{record['output_text']}<|im_end|>" in follow_up["prompt_text"]


def test_same_seed_same_server_gives_identical_output_ids(session):
    """If this fails, no replay result means anything (the brief: stop and report)."""
    outputs = []
    for _ in range(2):
        session.new_conversation("smol-text")
        outputs.append(session.say("Tell me a short story about a robot.").record)
    assert outputs[0]["prompt_ids"] == outputs[1]["prompt_ids"]
    assert outputs[0]["output_ids"] == outputs[1]["output_ids"]
    assert outputs[0]["stop_type"] == outputs[1]["stop_type"]


def test_replay_after_a_longer_call_and_after_a_restart(session):
    """Replaying a call from an older log always follows other calls and a server restart."""
    session.new_conversation("smol-text")
    first = session.say("Tell me about the sea.").record
    ids, stop = first["prompt_ids"], first["stop"]
    settings = {name: s["value"] for name, s in first["settings"].items()}
    backend = session.backend
    backend.complete(counting_ids(300), {**settings, "max_tokens": 8}, [])  # longer, different
    assert backend.complete(ids, settings, stop).token_ids == first["output_ids"]
    assert session.model is not None
    backend.close()
    backend.start(session.model, check_weights_quick(REPO, session.model_lock), backend.n_ctx)
    assert backend.complete(ids, settings, stop).token_ids == first["output_ids"]


def test_over_long_prompt_is_refused_before_sending(session):
    session.new_conversation("smol-text")
    with pytest.raises(ContextError, match="Nothing was sent"):
        session.say("word " * 9000)
    calls = [r for r in read_records(session.conversation.log.path) if r["type"] == "call"]
    assert calls[-1]["error"].startswith("ContextError")
    assert "output_ids" not in calls[-1]


def test_ctrl_c_cancels_the_reply_and_the_server_keeps_working(session):
    session.new_conversation("smol-text")

    def interrupt(_text: str) -> None:
        raise KeyboardInterrupt

    outcome = session.say("Write a long poem about the sea.", on_text=interrupt)
    assert outcome.record["stop_type"] == "cancelled"
    assert "cancelled" in outcome.record["flags"]
    all_checks_ok(session.say("Thanks.").record)


@pytest.fixture(scope="module")
def small_context_backend(tmp_path_factory):
    """A second server with a 256-token context, to find llama-server's exact limit."""
    config = load_server_config(REPO)
    backend = LlamaCppBackend(config, tmp_path_factory.mktemp("logs-small"))
    model = load_models(REPO)[MODEL]
    backend.start(model, check_weights_quick(REPO, lock_entry(REPO, model)), 256)
    yield backend
    backend.close()


def counting_ids(n: int) -> list[int]:
    """`n` token IDs of "1 2 3 …": the model keeps counting instead of stopping early."""
    model = load_models(REPO)[MODEL]
    tokenizer = Tokenizer.from_hub(model.tokenizer_repo, model.tokenizer_revision)
    ids = tokenizer.encode(" ".join(str(i) for i in range(1, 400)))
    assert len(ids) >= n
    return ids[:n]


MAX_TOKENS = 32
SETTINGS = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 1,
    "max_tokens": MAX_TOKENS,
    "samplers": ["temperature"],
}


def test_context_guard_matches_llama_servers_limit(small_context_backend):
    from clarity_chat.session import room_left

    n_ctx, max_tokens = 256, MAX_TOKENS
    fitting = n_ctx - max_tokens - 1  # the largest prompt the client allows
    assert room_left(n_ctx, fitting, max_tokens) == 0
    assert room_left(n_ctx, fitting + 1, max_tokens) == -1

    full = small_context_backend.complete(counting_ids(fitting), SETTINGS, [])
    assert full.stop_type == "limit"
    assert full.response["truncated"] is False
    assert full.n_predicted == max_tokens

    # At prompt + max_tokens == n_ctx the server still returns every token but marks the
    # reply truncated; the client refuses this case so an allowed call is never marked.
    one_more = small_context_backend.complete(counting_ids(fitting + 1), SETTINGS, [])
    assert one_more.response["truncated"] is True
    assert one_more.n_predicted == max_tokens
    assert "context_full" in one_more.flags

    # One token further, the context really cuts the reply: n_predicted = n_ctx - n_prompt.
    two_more = small_context_backend.complete(counting_ids(fitting + 2), SETTINGS, [])
    assert two_more.response["truncated"] is True
    assert two_more.n_predicted == n_ctx - (fitting + 2)


def test_llama_arg_variables_are_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "1024")
    backend = LlamaCppBackend(load_server_config(REPO), tmp_path)
    model = load_models(REPO)[MODEL]
    try:
        record = backend.start(model, check_weights_quick(REPO, lock_entry(REPO, model)), 8192)
        assert record is not None
        assert "LLAMA_ARG_CTX_SIZE" in record["env_removed"]
        assert record["props"]["default_generation_settings"]["n_ctx"] == 8192
    finally:
        backend.close()
    assert backend.server is None


def test_a_batch_run_with_both_modes(tmp_path):
    from clarity_chat.batch import run_batch

    openings = tmp_path / "openings.txt"
    openings.write_text("Name a colour.\n---\nCount to three.\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    session = Session(REPO, logs_dir=run_dir)
    try:
        blind = run_batch(
            session, "run-live", run_dir, openings, ["smol-chat", "smol-text"], 2, lambda s: None
        )
    finally:
        session.close()
    rows = list(read_records(run_dir / "calls.jsonl"))
    calls = [r for r in rows if r["type"] == "call"]
    assert len(calls) == 2 * 2 * 2
    for call in calls:
        all_checks_ok(call)
        assert call["settings"]["seed"]["origin"] == "run"
    assert sum(r["type"] == "server_start" for r in rows) == 1  # one model, one server
    assert blind.read_text(encoding="utf-8").count("Rating:") == 8


def test_a_run_exported_for_inspect_holds_each_calls_prompt_ids(tmp_path):
    """v0.4.3 I4: real templates' spans, through inspect_ai's own writer and reader."""
    from inspect_ai.log import read_eval_log

    from clarity_chat.batch import run_batch
    from clarity_chat.export_inspect import export_run

    openings = tmp_path / "openings.txt"
    openings.write_text("Name a colour.\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    session = Session(REPO, logs_dir=run_dir)
    try:
        run_batch(
            session, "run-live", run_dir, openings, ["smol-chat", "smol-text"], 2, lambda s: None
        )
        result = export_run(session, run_dir)
    finally:
        session.close()
    calls = {r["call_id"]: r for r in read_records(run_dir / "calls.jsonl") if r["type"] == "call"}
    names = [Path(f["path"]).name for f in result["files"]]
    assert names == ["1-smol-chat.eval", "2-smol-text.eval"]
    seen: list[str] = []
    for f in result["files"]:
        log = read_eval_log(f["path"])
        assert log.samples is not None and len(log.samples) == 2
        for sample in log.samples:
            call = calls[str(sample.uuid)]
            assert sample.metadata is not None
            assert sample.metadata["prompt_ids"] == call["prompt_ids"]
            text, prompt = call["prompt_text"], sample.messages[:-1]  # the last is the reply
            if f["path"].endswith("chat.eval"):  # each message is its part of prompt_text
                for message in prompt:
                    start, end = (message.metadata or {})["clarity"]["chars"]
                    assert message.content == text[start:end]
            else:
                assert [m.content for m in prompt] == [text]
            assert sample.messages[-1].content == call["output_text"]
            seen.append(str(sample.uuid))
    assert sorted(seen) == sorted(calls)


def test_replay_on_the_running_server_is_identical_and_nothing_differs(session):
    session.say("Name a fruit.")
    record = session.replay_current()
    assert record["identical"], record
    assert record["differences"] == {}, record["differences"]
    all_checks_ok(record)


def test_replay_from_the_log_on_a_new_server_with_logged_flags(session, tmp_path):
    outcome = session.say("Name a river.")
    path = session.conversation.log.path
    other = Session(REPO, logs_dir=tmp_path)
    _, server = find_call(path, outcome.record["call_id"])
    assert server is not None
    other.use_server_flags(server["flags"])
    try:
        record = other.replay(path, outcome.record["call_id"], flags_from="logged")
    finally:
        other.close()
    assert record["identical"], record
    assert record["server_id"] != server["server_id"]
    # A new server process: only facts that can't change output may differ.
    assert set(record["differences"]) <= {"client", "libraries"}, record["differences"]


GGUF_ENTRY = """\
[smol-q8]
# A third-party quantization, used only to test the GGUF-download route.
route = "gguf"
repo = "bartowski/SmolLM2-135M-Instruct-GGUF"
revision = "09816acd5d99df7be770d85ea30822623dab342c"
file = "SmolLM2-135M-Instruct-Q8_0.gguf"
sha256 = "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"
tokenizer_repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer_revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
"""


def test_the_gguf_download_route_end_to_end(tmp_path, monkeypatch):
    """Downloads 145 MB once (then the Hugging Face cache has it)."""
    import shutil

    import huggingface_hub.constants

    from clarity_chat.models import fetch

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)  # conftest sets it

    root = tmp_path / "project"
    (root / "profiles").mkdir(parents=True)
    shutil.copy(REPO / "defaults.toml", root)
    server = (REPO / "server.toml").read_text(encoding="utf-8")
    llama_dir = (REPO / "vendor" / "llama.cpp").as_posix()
    server = server.replace('llama_cpp_dir = "vendor/llama.cpp"', f'llama_cpp_dir = "{llama_dir}"')
    (root / "server.toml").write_text(server, encoding="utf-8")
    (root / "models.toml").write_text(GGUF_ENTRY, encoding="utf-8")
    profile = (REPO / "profiles" / "smol-text.toml").read_text(encoding="utf-8")
    profile = profile.replace('model = "smollm2-135m-instruct"', 'model = "smol-q8"')
    (root / "profiles" / "q8.toml").write_text(profile, encoding="utf-8")

    entry = fetch(root, load_models(root)["smol-q8"], load_server_config(root))
    assert entry["gguf"]["path"].startswith("~/"), entry["gguf"]["path"]  # the HF cache
    assert entry["gguf"]["path"].endswith("/SmolLM2-135M-Instruct-Q8_0.gguf")  # not the blob
    session = Session(root, logs_dir=tmp_path / "logs")
    try:
        session.new_conversation("q8")
        outcome = session.say("Name a colour.")
    finally:
        session.close()
    all_checks_ok(outcome.record)  # includes the echo check: tokenizer and GGUF agree
    assert outcome.record["model"]["gguf_sha256"] == entry["gguf"]["sha256"]


def test_the_next_start_stops_a_server_a_killed_clarity_left(tmp_path):
    """A2 against a real llama-server: a Clarity that dies without stopping its server
    leaves it recorded; the next start proves it's that server (its key and model) and
    stops it."""
    from clarity_chat import instance

    project = tmp_path / "project"  # holds only local/ (the lock and clarity.json)
    project.mkdir()
    with instance.hold(project, "chat") as held:
        session = Session(REPO, logs_dir=tmp_path / "logs", on_server=held.record_server)
        session.new_conversation("smol-text")
        assert isinstance(session.backend, LlamaCppBackend)
        server = session.backend.server
        assert server is not None and server.proc is not None
        proc, port = server.proc, server.port
        # A crash: the block ends without session.close(), so the server keeps running.
    try:
        assert proc.poll() is None, "the abandoned server should still be running"
        assert instance.read_info(project)["server"]["port"] == port  # still recorded
        with instance.hold(project, "chat") as again:
            assert again.orphan is not None and "Stopped llama-server" in again.orphan
        proc.wait(timeout=20)  # it exits on the SIGTERM the next start sent
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=20)


def test_a_quantized_copy_repeats_and_replays(session, tmp_path):
    """Every other same-output test runs bf16. llama-quantize makes a Q4_K_M copy of SmolLM2;
    a greedy and a seeded request each run twice, then again on a new server, as a replay
    does."""
    import subprocess

    config = load_server_config(REPO)
    tool = config.llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not tool.exists():
        pytest.skip(
            "llama-quantize isn't built: cmake --build vendor/llama.cpp/build "
            "--config Release -j 8 --target llama-quantize"
        )
    session.new_conversation("smol-text")
    first = session.say("Tell me about the sea.").record
    ids, stop = first["prompt_ids"], first["stop"]
    settings = {name: s["value"] for name, s in first["settings"].items()}
    requests = {
        "greedy": {**settings, "temperature": 0.0},
        "seeded": {**settings, "temperature": 0.8, "seed": 1234},
    }
    quantized = tmp_path / f"{MODEL}-Q4_K_M.gguf"
    bf16 = check_weights_quick(REPO, session.model_lock)
    subprocess.run(
        [str(tool), str(bf16), str(quantized), "Q4_K_M"], check=True, capture_output=True
    )
    assert session.model is not None
    backend = LlamaCppBackend(config, tmp_path / "logs")
    try:
        backend.start(session.model, quantized, 8192)
        outputs = {}
        for name, request in requests.items():
            outputs[name] = backend.complete(ids, request, stop).token_ids
            assert backend.complete(ids, request, stop).token_ids == outputs[name], name
        backend.close()
        backend.start(session.model, quantized, 8192)
        for name, request in requests.items():
            assert backend.complete(ids, request, stop).token_ids == outputs[name], name
    finally:
        backend.close()


def test_shipped_smollm2_bans_no_tokens(session):
    """v0.4.0 A1: llama-server's key dump is read, and SmolLM2's file has no
    tokenizer.ggml.suppress_tokens. A llama.cpp update that changes the dump's lines
    refuses the start, which shows here."""
    assert session.backend.server_record["model_file"]["suppress_tokens"] == []
    session.new_conversation("smol-text")
    record = session.say("Say hello.").record
    assert "suppress_tokens" not in record["backend_settings"]
    assert "suppress_tokens" not in record["checks"]


def test_a_file_that_bans_tokens_is_recorded_and_applied(session, tmp_path):
    """v0.4.0 A1: a copy of SmolLM2's file banning the first token of its greedy reply, and
    one more ID. The record lists both, the reply starts differently, and the check passes:
    the only real-file evidence that llama-server applies the ban the log shows."""
    from tests.gguf_ban import with_suppress_tokens

    session.new_conversation("smol-text")
    first = session.say("Say hello.").record
    ids, stop = first["prompt_ids"], first["stop"]
    greedy = {**{name: s["value"] for name, s in first["settings"].items()}, "temperature": 0.0}
    assert session.model is not None
    config = load_server_config(REPO)
    bf16 = check_weights_quick(REPO, session.model_lock)
    backend = LlamaCppBackend(config, tmp_path / "logs")
    try:
        backend.start(session.model, bf16, 8192)
        before = backend.complete(ids, greedy, stop).token_ids[0]
        backend.close()
        banned = [before, (before + 1) % 49152]
        copy = with_suppress_tokens(bf16, tmp_path / f"{MODEL}-banned.gguf", banned)
        record = backend.start(session.model, copy, 8192)
        assert record is not None and record["model_file"]["suppress_tokens"] == banned
        result = backend.complete(ids, greedy, stop)
        assert result.token_ids[0] != before
        assert result.backend_settings["suppress_tokens"] == {
            "value": banned,
            "origin": "model file",
        }
        assert result.checks["suppress_tokens"] == {"ok": True}
    finally:
        backend.close()
        (tmp_path / f"{MODEL}-banned.gguf").unlink(missing_ok=True)  # about 270 MB


def test_a_message_holding_unicode_line_separators_completes_and_replays(session):
    """QA's must (v0.4.0, finding 4): the call's final chunk carries the prompt, so a message
    holding U+2028, U+2029 or U+0085 reaches the stream raw whatever the model replies."""
    from clarity_chat.backends import BackendError

    session.new_conversation("smol-chat")
    try:
        record = session.say("Reply ok. ls:\u2028 ps:\u2029 nel:\u0085 end").record
    except BackendError as e:
        pytest.fail(f"the stream was refused: {e}")
    all_checks_ok(record)
    assert session.replay_current()["identical"]


def test_converting_twice_gives_the_same_file(tmp_path, capsys):
    """fetch's convert step, run twice on SmolLM2 into scratch folders, writes the same bytes.
    Whether they also equal models.lock's hash is printed, not asserted: the converter's
    environment may have changed since the model was fetched."""
    from clarity_chat.models import fetch_convert

    model = load_models(REPO)[MODEL]
    config = load_server_config(REPO)
    entries = [fetch_convert(tmp_path / run, model, config) for run in ("a", "b")]
    first, second = (entry["gguf"]["sha256"] for entry in entries)
    assert first == second
    with capsys.disabled():
        same = first == lock_entry(REPO, model)["gguf"]["sha256"]
        print(f"\nconverted file equals models.lock's: {same}")


def test_the_backend_contract_on_llama_cpp(tmp_path):
    """The call record's shape for each stop type, kept in fixtures/backend_shapes.json."""
    session = Session(REPO, logs_dir=tmp_path)
    try:
        shapes = contract.run_scenarios(session, chat="smol-chat", text="smol-text")
    finally:
        session.close()
    contract.check_or_write("llama.cpp", shapes)


def test_a_stopped_server_says_so_and_resume_starts_it_again(tmp_path):
    from clarity_chat.backends import BackendError
    from clarity_chat.commands import Commands

    session = Session(REPO, logs_dir=tmp_path)
    c = Commands(session)
    try:
        c.run("profile", "smol-chat", lambda e: None)
        c.run("say", "Name one colour.", lambda e: None)
        backend = session.backend
        assert isinstance(backend, LlamaCppBackend) and backend.server is not None
        proc = backend.server.proc
        assert proc is not None
        proc.kill()
        proc.wait()
        events: list[dict] = []
        with pytest.raises(BackendError, match="llama-server has stopped"):
            c.run("say", "And another?", events.append)
        assert {"event": "status", "status": "stopped"} in events
        assert session.conversation is not None
        c.run("resume", session.conversation.conversation_id, lambda e: None)
        record = c.run("say", "And another?", lambda e: None)["call"]
        all_checks_ok(record)
        assert record["stop_type"] in ("eos", "limit")
    finally:
        session.close()


# --- v0.3.1: attachments (the plan's live tiers)

TRAIL_NOTES = b"Made-up trail notes.\nThree cairns mark the ridge.\nNo water after the pass.\n"


def test_a_call_with_an_attachment_replays_identical(session):
    from clarity_chat.documents import extract

    session.new_conversation("smol-chat")
    session.attach(extract(TRAIL_NOTES, "trail-notes.md"))
    record = session.say("What marks the ridge?").record
    all_checks_ok(record)
    assert '<document name="trail-notes.md">\nMade-up trail notes.' in record["prompt_text"]
    replayed = session.replay_current()
    assert replayed["identical"], replayed
    assert replayed["differences"] == {}, replayed["differences"]


def test_a_resumed_chat_with_an_excerpt_renders_identical(tmp_path):
    from clarity_chat.documents import extract

    session = Session(REPO, logs_dir=tmp_path)
    try:
        session.new_conversation("smol-chat")
        session.attach(extract(TRAIL_NOTES, "trail-notes.md"), "lines", 2, 3)
        first = session.say("Is there water?").record
        all_checks_ok(first)
        assert "Excerpt: lines 2 to 3 of 3\nThree cairns" in first["prompt_text"]
        assert session.conversation is not None
        resumed = session.resume(session.conversation.log.path)
        assert resumed["render"]["identical"] is True, resumed["render"]
        follow_up = session.say("And the ridge?").record
        all_checks_ok(follow_up)
        assert follow_up["prompt_ids"][: len(first["prompt_ids"])] == first["prompt_ids"]
    finally:
        session.close()


# --- v0.3.2: search (the plan's live tiers; part 1's whole-branch S3)


def scratch_library(tmp_path, monkeypatch) -> None:
    """A library in a scratch folder, so the repo's local/documents/ is never read."""
    from clarity_chat.documents import library as library_module
    from clarity_chat.documents import library_add

    scratch = tmp_path / "scratch"
    library_add(scratch, b"## Ridge\n" + TRAIL_NOTES, "trail.md")
    monkeypatch.setattr(library_module, "LIBRARY", scratch / "local" / "documents")


def words_chat(tmp_path) -> str:
    """smol-chat searching by words, stated: CI's live-cpu job has no embedder (v0.4.1 Q4),
    and since meaning became the default (B3) the shipped profile searches by meaning.
    Meaning search's replay and resume are test_embed's."""
    text = (REPO / "profiles" / "smol-chat.toml").read_text(encoding="utf-8")
    path = tmp_path / "smol-chat-words.toml"
    path.write_text(text + '\n[documents]\nsearch_method = "words"\n', encoding="utf-8")
    return str(path)


def test_a_searched_call_replays_identical(session, tmp_path, monkeypatch):
    scratch_library(tmp_path, monkeypatch)
    session.new_conversation(words_chat(tmp_path))
    session.set_search(["trail"])
    record = session.say("What marks the ridge?").record
    all_checks_ok(record)
    assert '<part document="trail" title="Ridge">' in record["prompt_text"]
    replayed = session.replay_current()
    assert replayed["identical"], replayed
    assert replayed["differences"] == {}, replayed["differences"]


def test_a_resumed_chat_with_found_parts_renders_identical(tmp_path, monkeypatch):
    scratch_library(tmp_path, monkeypatch)
    session = Session(REPO, logs_dir=tmp_path / "logs")
    try:
        session.new_conversation(words_chat(tmp_path))
        session.set_search(["trail"])
        first = session.say("What marks the ridge?").record
        all_checks_ok(first)
        assert '<part document="trail"' in first["prompt_text"]
        assert session.conversation is not None
        resumed = session.resume(session.conversation.log.path)
        assert resumed["render"]["identical"] is True, resumed["render"]
        follow_up = session.say("And the pass?").record
        all_checks_ok(follow_up)
        assert follow_up["prompt_ids"][: len(first["prompt_ids"])] == first["prompt_ids"]
    finally:
        session.close()


def searching_profiles(tmp_path) -> list[str]:
    """smol-chat twice, searching the trail notes, one with the block at the start of the
    message and one at its end (test_ci_live reads a test body's "placement" as the GPU's,
    so the key is written here)."""
    text = (REPO / "profiles" / "smol-chat.toml").read_text(encoding="utf-8")
    paths = []
    for where in ("start", "end"):
        path = tmp_path / f"trail-{where}.toml"
        documents = (  # by words: CI's live-cpu job has no embedder (v0.4.1 Q4)
            f'\n[documents]\nsearch = ["trail"]\nplacement = "{where}"\nsearch_method = "words"\n'
        )
        path.write_text(text + documents, encoding="utf-8", newline="\n")
        paths.append(str(path))
    return paths


def test_a_blind_run_of_two_block_positions_finds_the_same_parts_for_every_seed(
    tmp_path, monkeypatch
):
    """The sprint audit's finding 7: a searching blind run on a real model. Two profiles
    that differ only in where the block goes, both searching one made-up document, two
    seeds: each opening's `found` is the same for every seed (searched once per opening),
    and a call replays identical."""
    from clarity_chat.batch import run_batch

    scratch_library(tmp_path, monkeypatch)
    profiles = searching_profiles(tmp_path)
    openings = tmp_path / "openings.txt"
    openings.write_text("What marks the ridge?\n---\nIs there water after the pass?\n")
    run_dir = tmp_path / "run"
    session = Session(REPO, logs_dir=run_dir)
    try:
        run_batch(session, "run-live-search", run_dir, openings, profiles, 2, lambda s: None)
    finally:
        session.close()
    rows = list(read_records(run_dir / "calls.jsonl"))
    turns = {r["turn_id"]: r for r in rows if r["type"] == "turn"}
    calls = [r for r in rows if r["type"] == "call"]
    assert len(calls) == 2 * 2 * 2
    found: dict[tuple, list] = {}
    for call in calls:
        all_checks_ok(call)
        key = (call["run"]["opening"], call["profile"]["name"])
        found.setdefault(key, []).append(turns[call["parent_id"]]["found"])
    assert len(found) == 4
    for key, each in found.items():
        assert each[0]["parts"], key  # the trail notes match both openings' words
        assert all(f == each[0] for f in each), key  # one search per opening, every seed
    replayer = Session(REPO, logs_dir=tmp_path / "replays")
    try:
        replayed = replayer.replay(run_dir / "calls.jsonl", calls[-1]["call_id"])
    finally:
        replayer.close()
    assert replayed["identical"], replayed


# --- v0.4.2 A1, first step: what a start's output says about the build (decision 7)


def test_a_start_s_output_holds_the_build_line_and_system_info(session):
    """llama-server logs "build N (commit) with COMPILER for TARGET" and a system_info line
    (common_params_print_info, common.cpp:418-439) at log-verbosity 4. Replay can compare
    them only if Clarity's startup_output holds both. Prints them for the status file."""
    output = session.backend.server_record["startup_output"]
    build = [line for line in output.splitlines() if "common_params_print_info: build " in line]
    info = [line for line in output.splitlines() if "system_info: " in line]
    print("\n" + "\n".join(build + info))
    assert build, "no build line in startup_output"
    assert info, "no system_info line in startup_output"
    # A1: replay reads both from the start, so a call made on it compares its build.
    from clarity_chat.replay import facts_from_call

    # A conversation of its own: an earlier test leaves the shared session searching a
    # scratch library that's gone by now, so the order between tests doesn't matter.
    session.new_conversation("smol-text")
    record = session.say("Name a colour.").record
    built = facts_from_call(record, session.backend.server_record).get("build") or {}
    # Checkpoint 1's S3: "COMMIT with COMPILER for TARGET", no build number; the commit as
    # llama.cpp prints it, a short form of server.toml's pin.
    line = built.get("line") or ""
    commit = line.split(" with ")[0]
    pin = load_server_config(REPO).llama_cpp_commit
    assert len(commit) >= 7 and pin.startswith(commit), (line, pin)
    assert " with " in line and " for " in line and not line.startswith("build "), line
    assert built.get("features"), built
