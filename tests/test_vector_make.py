"""V5: `clarity vector make`, end to end on a tiny Llama saved as a project's Transformers
weights, read on the CPU: the record, the readout, the snippet and a steered chat with it.
Every pair here is made up and neutral."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from clarity_chat.maker import LEGEND, make, prepare, read_pairs, render_pairs
from clarity_chat.profiles import ConfigError
from tests.test_maker_render import HF_CHAT, PAIRS, hf_root, profile

FIELDS = {"clarity", "time", "vector", "read", "pairs", "profile", "backend", "libraries"}
FIELDS |= {"readout", "prompts"}
GGUF, RECORD = "local/vectors/brief-hf-chat.gguf", "local/vectors/brief-hf-chat.json"


@pytest.fixture
def project(root, chat_tokenizer):
    """A project whose tiny model has real Transformers weights (4 blocks, width 8, the chat
    tokenizer), a chat profile on the CPU, and the pairs file brief.toml."""
    torch = pytest.importorskip("torch")
    import transformers

    hf_root(root)
    weights = root / "models" / "tiny-hf"
    config = transformers.LlamaConfig(
        vocab_size=258,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=257,
        bos_token_id=256,
    )
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(config).save_pretrained(weights)
    chat_tokenizer.tok.save_pretrained(weights)
    profile(root, extra="ctx_size = 256")
    (root / "brief.toml").write_text(PAIRS, encoding="utf-8")
    return root


def session_on(root, chat_tokenizer):
    from clarity_chat.session import Session

    return Session(root, tokenizer_loader=lambda r, v: chat_tokenizer)


def made(root, chat_tokenizer, pairs="brief.toml", name="hf-chat"):
    session = session_on(root, chat_tokenizer)
    try:
        make(session, root / pairs, name)
    finally:
        session.close()
    record = root / "local" / "vectors" / f"{Path(pairs).stem}-{name}.json"
    assert record.is_file(), "make wrote no record"
    return json.loads(record.read_text(encoding="utf-8"))


def test_make_writes_the_vector_and_its_record(project, chat_tokenizer):
    record = made(project, chat_tokenizer)
    data = (project / GGUF).read_bytes()
    assert set(record) == FIELDS
    assert record["vector"] == {
        "file": GGUF,
        "sha256": hashlib.sha256(data).hexdigest(),
        "blocks": [1, 2, 3],
        "model_hint": "llama",
        "method": "mean difference",
        "sign": "positive minus negative",
        "length": "not normalized",
    }
    assert record["read"] == {"tensor": "block output", "position": "last token"}
    text = PAIRS
    assert record["pairs"] == {
        "file": str(project / "brief.toml"),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "form": "crossed",
        "counts": {"systems": 2, "messages": 2, "openings": 2},
        "description": "Plain replies against ornate ones.",
        "author_self_reported": "me",
    }
    assert record["profile"]["name"] == "hf-chat" and "ctx_size = 256" in record["profile"]["toml"]
    assert record["backend"]["type"] == "server_start"
    assert record["backend"]["device"] == "cpu" and record["backend"]["dtype"] == "bfloat16"
    assert set(record["libraries"]) == {"transformers", "tokenizers", "jinja2"}
    assert record["clarity"]["version"] and "commit" in record["clarity"]
    assert [b["block"] for b in record["readout"]["blocks"]] == [1, 2, 3]
    assert all(b["gap"] > 0 for b in record["readout"]["blocks"])
    session = session_on(project, chat_tokenizer)
    expected = render_pairs(prepare(session, "hf-chat"), read_pairs(project / "brief.toml"), 256)
    assert record["prompts"] == [
        {"pair": p.pair, "side": p.side, "text": p.text, "ids": p.ids} for p in expected
    ]
    if sys.platform != "win32":
        assert (project / RECORD).stat().st_mode & 0o777 == 0o600


def test_the_readout_line_by_line(project, chat_tokenizer, capsys):
    record = made(project, chat_tokenizer)
    out, err = capsys.readouterr()
    digest = record["vector"]["sha256"]
    first = record["prompts"][0]["text"], record["prompts"][1]["text"]
    assert first == (
        "<system>Reply plainly.</s><user>Boil an egg?</s><assistant>",
        "<system>Reply ornately.</s><user>Boil an egg?</s><assistant>",
    )
    rows = [
        f"{b['block']:5}  {b['activation_size']:15.2f}  {b['gap']:5.2f}  {b['agreement']:+9.2f}"
        for b in record["readout"]["blocks"]
    ]
    expected = [
        f"Made {GGUF} from 8 pairs in S s; its record, {RECORD}, holds the pairs file, every "
        "prompt as the model read it, and the numbers below.",
        "Pairs: 2 system pairs × 2 messages × 2 openings.",
        '"Plain replies against ornate ones." Written by me, as the pairs file says.',
        "",
        "The first pair, exactly as the model read it (each prompt is read at its last token):",
        "--- positive ---",
        first[0],
        "--- negative ---",
        first[1],
        "",
        "block  activation size    gap  agreement",
        *rows,
        LEGEND,
        "These are facts about the pairs, not a score.",
        "",
        "To try it, add this to a Transformers profile of tiny, and steering_scale = 1.0 to its "
        "[settings]:",
        "",
        "[steering]",
        f'file = "{GGUF}"',
        f'sha256 = "{digest}"',
        "layers = [2]",
    ]
    assert re.sub(r" in \d+ s;", " in S s;", out).splitlines() == expected
    assert err.splitlines() == [
        "Reading 16 prompts (8 pairs) with tiny on cpu…",
        *(f"{k} of 16" for k in (2, 4, 5, 7, 8, 10, 12, 13, 15, 16)),
    ]
    assert LEGEND == (
        "Activation size: the average length of the block's output. Gap: the length of this "
        "vector, the average difference between the two sides. Agreement: the average cosine "
        "similarity between each pair's difference and this vector, from -1 to 1."
    )


def test_untrusted_text_prints_escaped_and_the_record_keeps_it(project, chat_tokenizer, capsys):
    text = PAIRS.replace('"Plain replies', '"Plain \\u001b[31m replies')
    text = text.replace('author = "me"', 'author = "me\\u001b]0;x\\u0007"')
    text = text.replace('"Reply plainly."', '"Reply \\u001b[2J plainly."')
    (project / "brief.toml").write_text(text, encoding="utf-8")
    record = made(project, chat_tokenizer)
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
    assert '"Plain \\x1b[31m replies against ornate ones." Written by me\\x1b]0;x\\x07,' in out
    assert "<system>Reply \\x1b[2J plainly.</s>" in out
    assert record["pairs"]["description"] == "Plain \x1b[31m replies against ornate ones."
    assert record["pairs"]["author_self_reported"] == "me\x1b]0;x\x07"
    assert "Reply \x1b[2J plainly." in record["prompts"][0]["text"]


def test_the_snippet_steers_a_chat_that_replays(project, chat_tokenizer, capsys):
    made(project, chat_tokenizer)
    out = capsys.readouterr().out
    snippet = out[out.index("[steering]") :]
    text = HF_CHAT.replace("[settings]", "[settings]\nsteering_scale = 1.0") + "\n" + snippet
    profile(project, "steered-chat", extra="ctx_size = 256", text=text)
    session = session_on(project, chat_tokenizer)
    try:
        session.new_conversation("steered-chat")
        record = session.say("Boil an egg?").record
        digest = hashlib.sha256((project / GGUF).read_bytes()).hexdigest()
        assert record["request"]["steering"] == {
            "file": GGUF,
            "sha256": digest,
            "scale": 1.0,
            "layers": [2],
        }
        assert record["checks"]["steering"]["ok"] is True
        assert session.conversation is not None
        replay = session.replay(session.conversation.log.path, record["call_id"])
        assert replay["identical"]
    finally:
        session.close()


def test_a_second_make_is_refused_and_changes_nothing(project, chat_tokenizer):
    made(project, chat_tokenizer)
    before = [(project / p).read_bytes() for p in (GGUF, RECORD)]
    with pytest.raises(ConfigError) as refused:
        made(project, chat_tokenizer)
    assert str(refused.value) == (
        f"{GGUF} already exists, and a profile may pin it by its sha256. Rename or delete it, "
        "or rename the pairs file."
    )
    assert [(project / p).read_bytes() for p in (GGUF, RECORD)] == before


def test_a_leftover_record_with_no_vector_is_replaced(project, chat_tokenizer):
    (project / "local" / "vectors").mkdir(parents=True)
    (project / RECORD).write_text("left from an interrupted make", encoding="utf-8")
    record = made(project, chat_tokenizer)
    assert record["vector"]["file"] == GGUF and (project / GGUF).is_file()


def test_ctrl_c_during_the_read_leaves_no_files(project, chat_tokenizer, monkeypatch):
    from clarity_chat import maker

    real = maker.read_blocks

    def interrupted(model, prompts, on_progress):
        def progress(k, n):
            if k >= 5:
                raise KeyboardInterrupt
            on_progress(k, n)

        return real(model, prompts, progress)

    monkeypatch.setattr(maker, "read_blocks", interrupted)
    with pytest.raises(KeyboardInterrupt):
        made(project, chat_tokenizer)
    vectors = project / "local" / "vectors"
    assert not vectors.exists() or list(vectors.iterdir()) == []
    assert [p for p in (project / "local").rglob("*") if p.name.endswith(".tmp")] == []


def test_a_steered_profile_is_read_unsteered_even_with_its_vector_gone(
    project, chat_tokenizer, capsys
):
    made(project, chat_tokenizer)
    snippet = capsys.readouterr().out.split("[settings]:\n\n")[1]
    text = HF_CHAT.replace("[settings]", "[settings]\nsteering_scale = 1.0") + "\n" + snippet
    profile(project, "hf-steered", extra="ctx_size = 256", text=text)
    (project / "other.toml").write_text(PAIRS, encoding="utf-8")
    profile(project, "hf-chat-twin", extra="ctx_size = 256")  # the same profile, unsteered
    made(project, chat_tokenizer, "other.toml", "hf-chat-twin")
    made(project, chat_tokenizer, "other.toml", "hf-steered")
    out = capsys.readouterr().out
    assert "Read without steering: this profile's [steering] isn't applied while making." in out
    vectors = project / "local" / "vectors"
    assert (vectors / "other-hf-steered.gguf").read_bytes() == (
        vectors / "other-hf-chat-twin.gguf"
    ).read_bytes()
    (project / GGUF).unlink()  # the pinned vector deleted, to make it again
    (project / RECORD).unlink()
    (vectors / "other-hf-steered.gguf").unlink()
    made(project, chat_tokenizer, "other.toml", "hf-steered")
    assert (vectors / "other-hf-steered.gguf").is_file()


def test_a_held_lock_refuses_the_make(project, capsys):
    from clarity_chat import instance
    from clarity_chat.cli import main

    with instance.hold(project, "chat"):
        code = main(
            ["--root", str(project), "vector", "make", "brief.toml", "--profile", "hf-chat"]
        )
    assert code == 1
    assert "Clarity is already running in this project" in capsys.readouterr().err
    assert not (project / "local" / "vectors").exists()


def test_the_reference_names_the_command_and_quotes_the_legend():
    from tests.test_docs import REFERENCE

    assert "| `clarity vector make PAIRS --profile NAME`" in REFERENCE
    assert LEGEND in REFERENCE.replace("\n", " ")


def test_the_pairs_path_is_as_given(project, chat_tokenizer, monkeypatch):
    monkeypatch.chdir(project)
    session = session_on(project, chat_tokenizer)
    try:
        make(session, Path("brief.toml"), "hf-chat")
    finally:
        session.close()
    assert (project / RECORD).is_file(), "make wrote no record"
    record = json.loads((project / RECORD).read_text(encoding="utf-8"))
    assert record["pairs"]["file"] == "brief.toml"
