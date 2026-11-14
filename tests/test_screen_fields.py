"""Fields the web UI's screens read (v0.2.3 item 10b; notes/plans/v030-data-walk.md)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from clarity_chat.cli import format_resume
from clarity_chat.commands import Commands
from clarity_chat.session import Session
from tests.fakes import FakeBackend

GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "v0.1.0.jsonl"


@pytest.fixture
def commands(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!"] * 6, tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    return c, backend


def run(c, line):
    name, _, args = line.partition(" ")
    return c.run(name, args, lambda e: None)


def test_prompt_returns_the_calls_replays_live_and_after_a_resume(commands):
    c, _ = commands
    call = run(c, "say Hi")["call"]
    assert run(c, f"prompt {call['call_id']}").get("replays") == []
    replay = run(c, "replay")["replay"]
    replays = run(c, f"prompt {call['call_id']}").get("replays")
    assert [r["replay_id"] for r in replays] == [replay["replay_id"]]
    assert replays[0]["identical"] is True
    run(c, f"resume {c.session.conversation.conversation_id}")
    after = run(c, f"prompt {call['call_id']}").get("replays")
    assert [r["replay_id"] for r in after] == [replay["replay_id"]]


def test_a_changed_setting_carries_the_profiles_own_value(commands):
    c, _ = commands
    run(c, "set temperature 0.9")
    row = run(c, "settings")["settings"]["temperature"]
    assert row == {"value": 0.9, "origin": "chat", "profile": {"value": 0.6, "origin": "profile"}}
    assert "profile" not in run(c, "settings")["settings"]["top_p"]  # unchanged rows don't


def test_settings_and_state_name_the_model(commands):
    c, _ = commands
    assert run(c, "settings").get("model_name") == "tiny"
    assert run(c, "state").get("model_name") == "tiny"


def test_not_applied_follows_the_current_settings_before_any_reply(root, tiny_tokenizer):
    from tests.test_steering import steer_root

    steer_root(root)
    text = (root / "profiles" / "text.toml").read_text(encoding="utf-8")
    hf = text.replace('backend = "llama.cpp"', 'backend = "transformers"')
    (root / "profiles" / "hf.toml").write_text(hf, encoding="utf-8")
    backend = FakeBackend([" Hi."], tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    c = Commands(session)
    run(c, "profile hf")
    run(c, "set samplers penalties top_k top_p min_p temperature")
    run(c, "set presence_penalty 0.5")
    applied = {e["name"]: e for e in run(c, "settings")["not_applied"]}
    assert applied.get("presence_penalty", {}).get("affects_output") is True
    run(c, "set presence_penalty")
    applied = {e["name"]: e for e in run(c, "settings")["not_applied"]}
    assert applied["presence_penalty"]["affects_output"] is False


def test_a_llama_cpp_profile_has_nothing_not_applied(commands):
    c, _ = commands
    assert run(c, "settings")["not_applied"] == []


def test_resume_says_how_many_things_differ_as_the_terminal_does(commands, tmp_path):
    c, _ = commands
    path = Path(shutil.copy(GOLDEN, c.session.logs_dir / "golden-v010.jsonl"))
    result = run(c, f"resume {path.stem}")
    record = result["resume"]
    expected = (
        (record["profile_file"] != "same")
        + len(record["model_entry"])
        + len(record["settings_differ_today"])
        + len(record["differences"])
        + (record["render"] is not None and not record["render"]["identical"])
        + (record["ctx_size"]["origin"] != "log")
    )
    differ = result.get("differ", {})
    assert differ.get("count") == expected > 0
    assert sum(item["count"] for item in differ["items"]) == expected
    printed = format_resume(result)
    for item in differ["items"]:
        assert item["text"] in printed


def hf_commands(root, tiny_tokenizer, replies):
    from tests.test_steering import steer_root

    steer_root(root)
    text = (root / "profiles" / "text.toml").read_text(encoding="utf-8")
    hf = text.replace('backend = "llama.cpp"', 'backend = "transformers"')
    (root / "profiles" / "hf.toml").write_text(hf, encoding="utf-8")
    backend = FakeBackend(replies, tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    c = Commands(Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer))
    run(c, "profile hf")
    return c


def test_a_setting_an_old_call_didnt_record_was_off_not_unknown(root, tiny_tokenizer):
    """Final checkpoint, finding 3: an old Transformers chat, then /set logprobs 5: /regen
    same must remake the seed, since the old call ran with probabilities off."""
    c = hf_commands(root, tiny_tokenizer, [" one", " two"])
    call = run(c, "say Hi")["call"]
    for record in (call, c.session.conversation.calls[call["call_id"]]):
        record["settings"].pop("logprobs", None)  # as logged before v0.2.1
    run(c, "set logprobs 5")
    assert run(c, "regen same")["seeds"] == [42]
