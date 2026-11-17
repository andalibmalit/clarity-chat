"""v0.4.0 A3: `clarity replay` finds a call's model by its file (decision 9). States 5 to 8 of
the plan's states table; in a conversation, replay is as before, plus `model_used`.

A scratch root with made-up model entries; made-up text only.
"""

from __future__ import annotations

import json

import pytest

from clarity_chat import cli
from clarity_chat.replay import model_for_call
from clarity_chat.session import Session
from tests.conftest import MODELS
from tests.fakes import FakeBackend

SHA = "0" * 64  # fake_fetch's gguf sha256
OTHER_SHA = "1" * 64
PINS = {
    "route": "convert",
    "repo": "test/tiny",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "tokenizer_repo": "test/tiny",
    "tokenizer_revision": "0123456789abcdef0123456789abcdef01234567",
}


def lock_of(**files) -> dict:
    """Lock entries by name: each the call's pins, with this gguf sha256."""
    return {name: {**PINS, "gguf": {"sha256": sha}} for name, sha in files.items()}


CALL = {"name": "tiny", **PINS, "gguf_sha256": SHA}


# --- the rule, alone ------------------------------------------------------------------------


def test_the_calls_own_name_wins_when_it_has_the_file():
    assert model_for_call(CALL, "llama.cpp", lock_of(aaa=SHA, tiny=SHA)) == "tiny"


def test_another_name_with_the_file_is_found_the_first_in_sorted_order():
    assert model_for_call(CALL, "llama.cpp", lock_of(zed=SHA, mid=SHA, other=OTHER_SHA)) == "mid"
    assert model_for_call(CALL, "llama.cpp", lock_of(tiny=OTHER_SHA, zed=SHA)) == "zed"


def test_the_calls_name_with_another_file_and_no_match_runs_on_that_name():
    assert model_for_call(CALL, "llama.cpp", lock_of(tiny=OTHER_SHA, b=OTHER_SHA)) == "tiny"


def test_no_name_and_no_file_is_none():
    assert model_for_call(CALL, "llama.cpp", lock_of(b=OTHER_SHA)) is None
    assert model_for_call(CALL, "llama.cpp", {}) is None


def test_tokenizer_pins_are_not_compared():
    lock = {"twin": {**PINS, "tokenizer_revision": "f" * 40, "gguf": {"sha256": SHA}}}
    assert model_for_call(CALL, "llama.cpp", lock) == "twin"


def test_transformers_matches_the_source_weights_not_the_gguf():
    """Transformers loads the source weights: the same repo and revision match."""
    lock = lock_of(twin=OTHER_SHA)
    assert model_for_call(CALL, "transformers", lock) == "twin"
    assert model_for_call(CALL, "llama.cpp", lock) is None
    moved = {"twin": {**PINS, "revision": "f" * 40, "gguf": {"sha256": SHA}}}
    assert model_for_call(CALL, "transformers", moved) is None


def test_a_call_without_gguf_sha256_matches_by_name():
    """Every Clarity log records gguf_sha256; one another program wrote may not."""
    call = {k: v for k, v in CALL.items() if k != "gguf_sha256"}
    assert model_for_call(call, "llama.cpp", lock_of(tiny=OTHER_SHA, aaa=SHA)) == "tiny"
    assert model_for_call(call, "llama.cpp", lock_of(aaa=SHA)) is None


# --- clarity replay --------------------------------------------------------------------------


@pytest.fixture
def logged(fetched_root, tiny_tokenizer):
    """A log with one call on tiny, made in a chat; returns (root, log path, call ID)."""
    backend = FakeBackend([" Hello!"], tokenizer=tiny_tokenizer)
    session = Session(
        fetched_root,
        backend=backend,
        logs_dir=fetched_root / "logs",
        tokenizer_loader=lambda r, v: tiny_tokenizer,
    )
    conversation = session.new_conversation("text")
    call_id = session.say("Hi").record["call_id"]
    session.close()
    return fetched_root, conversation.log.path, call_id


def models_toml(root, *names: str) -> None:
    """models.toml with tiny's entry under each of `names`."""
    body = MODELS.split("\n", 1)[1]
    (root / "models.toml").write_text("".join(f"[{n}]\n{body}" for n in names), "utf-8")


def set_lock(root, **sha: str) -> None:
    """Each named lock entry as tiny's, with this sha256; tiny's own entry removed unless named."""
    path = root / "models.lock"
    lock = json.loads(path.read_text(encoding="utf-8"))
    tiny = lock["models"]["tiny"]
    lock["models"] = {
        n: {**tiny, "name": n, "gguf": {**tiny["gguf"], "sha256": s}} for n, s in sha.items()
    }
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")


def replay(monkeypatch, tokenizer, root, log, call_id, capsys, replies=(" Hello!",)):
    """`clarity replay LOG CALL_ID` on the fake backend: (exit status, output, backend)."""
    backend = FakeBackend(list(replies), tokenizer=tokenizer)

    def session(root, logs_dir, on_server=None):
        return Session(
            root, backend=backend, logs_dir=logs_dir, tokenizer_loader=lambda r, v: tokenizer
        )

    monkeypatch.setattr(cli, "Session", session)
    status = cli.main(["--root", str(root), "replay", str(log), call_id])
    return status, capsys.readouterr(), backend


def replays(log) -> list[dict]:
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["type"] == "replay"]


def test_state_5_the_calls_name_has_its_file(logged, monkeypatch, tiny_tokenizer, capsys):
    root, log, call_id = logged
    models_toml(root, "aaa", "tiny")
    set_lock(root, aaa=SHA, tiny=SHA)
    status, out, backend = replay(monkeypatch, tiny_tokenizer, root, log, call_id, capsys)
    assert status == 0 and backend.started == ["tiny"]
    assert replays(log)[-1].get("model_used") == "tiny"
    assert "Replayed on" not in out.out


def test_state_6_another_name_has_the_calls_file(logged, monkeypatch, tiny_tokenizer, capsys):
    root, log, call_id = logged
    models_toml(root, "renamed", "zzz")
    set_lock(root, renamed=SHA, zzz=SHA)
    status, out, backend = replay(monkeypatch, tiny_tokenizer, root, log, call_id, capsys)
    assert status == 0, out.err
    assert backend.started == ["renamed"]
    assert replays(log)[-1].get("model_used") == "renamed"
    assert "Replayed on renamed, the models.toml entry with this call's file" in out.out


def test_state_6_with_the_entrys_file_missing_says_so(logged, monkeypatch, tiny_tokenizer, capsys):
    root, log, call_id = logged
    models_toml(root, "renamed")
    set_lock(root, renamed=SHA)
    (root / "models" / "tiny" / "tiny-bf16.gguf").unlink()
    status, out, _ = replay(monkeypatch, tiny_tokenizer, root, log, call_id, capsys)
    assert status == 1 and "is missing" in out.err


def test_state_7_the_calls_name_with_another_file(logged, monkeypatch, tiny_tokenizer, capsys):
    root, log, call_id = logged
    models_toml(root, "tiny", "other")
    set_lock(root, tiny=OTHER_SHA, other=OTHER_SHA)
    status, out, backend = replay(monkeypatch, tiny_tokenizer, root, log, call_id, capsys)
    assert backend.started == ["tiny"]
    record = replays(log)[-1]
    assert record.get("model_used") == "tiny"
    assert "gguf_sha256" in record["differences"]


def test_state_8_no_name_and_no_file_lists_the_pins(logged, monkeypatch, tiny_tokenizer, capsys):
    root, log, call_id = logged
    models_toml(root, "other")
    set_lock(root, other=OTHER_SHA)
    status, out, backend = replay(monkeypatch, tiny_tokenizer, root, log, call_id, capsys)
    assert status == 1 and backend.started == []
    for key in (*PINS, "gguf_sha256"):
        assert key in out.err, key
    for value in (*PINS.values(), SHA, "f" * 40):  # f*40: converted_with's llama.cpp commit
        assert value in out.err, value
    assert "add an entry with these pins to local/models.toml and fetch it" in out.err
    assert "isn't in models.toml" not in out.err


def test_state_8_names_the_file_of_a_gguf_route_call(logged, monkeypatch, tiny_tokenizer, capsys):
    root, log, call_id = logged
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["type"] == "call":
            row["model"] |= {"route": "gguf", "file": "made-up-Q4_K_M.gguf"}
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models_toml(root, "other")
    set_lock(root, other=OTHER_SHA)
    status, out, _ = replay(monkeypatch, tiny_tokenizer, root, log, call_id, capsys)
    assert status == 1 and "made-up-Q4_K_M.gguf" in out.err


def test_in_a_conversation_replay_is_as_before_with_model_used(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!", " Hello!"], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    session.new_conversation("text")
    session.say("Hi")
    record = session.replay_current()
    assert record["identical"] and record.get("model_used") == "tiny"
