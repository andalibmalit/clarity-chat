"""V5: `clarity vector make`, end to end on a tiny Llama saved as a project's Transformers
weights, read on the CPU: the record, the readout, the snippet and a steered chat with it.
Every pair here is made up and neutral."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

from clarity_chat.maker import LEGEND, make, prepare, read_pairs, render_pairs
from clarity_chat.profiles import ConfigError
from tests.test_maker_render import HF_CHAT, PAIRS, hf_root, profile, session_for

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
    # Transformers' own "Loading weights" bar shares stderr, as it does for a chat.
    assert [line for line in err.split("\n") if line and "\r" not in line] == [
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


# --- checkpoint 1's fixes: M1, S4, S7 and the lock's wording


class Started(Exception):
    """Raised in place of starting the model: what make did before that is what's tested."""


def unstarted(root, chat_tokenizer, config=None):
    """A project with the tiny model's config.json only (no weights), whose session refuses
    to start a model: make gets as far as start_model and no further."""
    hf_root(root)
    if config is not None:
        (root / "models" / "tiny-hf" / "config.json").write_text(json.dumps(config), "utf-8")
    (root / "brief.toml").write_text(PAIRS, encoding="utf-8")
    session = session_for(root, chat_tokenizer)

    def start_model(prepared, ctx_size):
        raise Started

    session.start_model = start_model
    return session


def test_a_model_type_steering_hasnt_been_checked_on_is_refused_before_the_model_starts(
    root, chat_tokenizer
):
    """M1: make reads the profile unsteered, so the type check steering does on a pinned
    vector never ran; a Qwen3 profile made a vector every chat would refuse."""
    config = {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 4}
    session = unstarted(root, chat_tokenizer, config)
    profile(root, extra="ctx_size = 256")
    outcome = None
    try:
        make(session, root / "brief.toml", "hf-chat")
    except (ConfigError, Started) as e:
        outcome = e
    assert str(outcome) == (
        "Steering hasn't been checked on model type 'qwen3'; it runs on llama and olmo3. Ask "
        "for it to be added."
    ), f"not refused: {outcome!r}"
    assert not (root / "local" / "vectors").exists()


def test_the_profiles_warnings_print_before_the_read(root, chat_tokenizer, capsys):
    """S4: the warnings a chat on this profile shows when it starts, here on stderr."""
    session = unstarted(root, chat_tokenizer)
    profile(root, extra='ctx_size = 256\n[template_kwargs]\ngreting = "hey"')
    with pytest.raises(Started):
        make(session, root / "brief.toml", "hf-chat")
    assert capsys.readouterr().err.splitlines() == [
        "The chat template never reads template_kwargs ['greting'], so they change nothing. "
        "Check their names against the template: the prompt shows its output.",
        "Reading 16 prompts (8 pairs) with tiny on cpu…",
    ]


def test_a_profile_name_prints_escaped_in_prepares_refusals(root, chat_tokenizer):
    """S7: a profile's name is its file's name, which can hold any character."""
    session = unstarted(root, chat_tokenizer)
    name = profile(
        root,
        "cpp\u202ex",
        text=HF_CHAT.replace('"transformers"', '"llama.cpp"').replace('device = "cpu"\n', ""),
    )
    with pytest.raises(ConfigError) as refused:
        make(session, root / "brief.toml", name)
    assert str(refused.value).startswith("Profile cpp\\u202ex runs on llama.cpp;")


def test_a_control_token_prints_escaped(root, chat_tokenizer, monkeypatch):
    """S7: the control token's text comes from the tokenizer's files."""
    session = unstarted(root, chat_tokenizer)
    profile(root, extra="ctx_size = 256")
    prepared = prepare(session, "hf-chat")
    monkeypatch.setattr(prepared.tokenizer, "special_tokens_in", lambda text: ["<\x1b[2J>"])
    with pytest.raises(ConfigError) as refused:
        render_pairs(prepared, read_pairs(root / "brief.toml"), 256)
    assert "contains <\\x1b[2J>, which the tokenizer" in str(refused.value)


def test_the_snippet_writes_any_profile_name_in_ascii(project, chat_tokenizer, capsys):
    """S7: the snippet's file names the profile; written with TOML's escapes, it prints
    nothing raw and still reads back as the file's path. json.dumps would write the emoji
    as two surrogates, which TOML refuses."""
    profile(project, "hf\u202echat\U0001f600\x7f", extra="ctx_size = 256")
    made(project, chat_tokenizer, name="hf\u202echat\U0001f600\x7f")
    out = capsys.readouterr().out
    assert "\u202e" not in out and "\x7f" not in out
    line = next(line for line in out.splitlines() if line.startswith("file = "))
    assert line == 'file = "local/vectors/brief-hf\\u202echat\\U0001f600\\u007f.gguf"'
    steering = tomllib.loads(out[out.index("[steering]") :])["steering"]
    assert steering["file"] == "local/vectors/brief-hf\u202echat\U0001f600\x7f.gguf"


def test_the_lock_names_vector_make(root, monkeypatch):
    from clarity_chat import instance, maker
    from clarity_chat.cli import main

    held = []
    monkeypatch.setattr(maker, "make", lambda s, p, n: held.append(instance.read_info(root)) or 0)
    assert main(["--root", str(root), "vector", "make", "brief.toml", "--profile", "x"]) == 0
    assert held[0]["command"] == "vector make"


@pytest.mark.parametrize("fails", ["write_text_atomic", "write_bytes_atomic"])
def test_the_record_is_written_before_the_vector(project, chat_tokenizer, monkeypatch, fails):
    """Checkpoint 1's S2: no vector without its record. The record's write failing leaves
    neither file; the vector's failing leaves the record alone."""

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(f"clarity_chat.maker.{fails}", broken)
    session = session_on(project, chat_tokenizer)
    with pytest.raises(OSError, match="disk full"):
        make(session, project / "brief.toml", "hf-chat")
    session.close()
    assert (project / RECORD).is_file() == (fails == "write_bytes_atomic")
    assert not (project / GGUF).exists()


def refused_or_started(session) -> ConfigError | Started:
    try:
        make(session, session.root / "brief.toml", "hf-chat")
    except (ConfigError, Started) as e:
        return e
    raise AssertionError("make neither refused nor started the model")


def test_make_reads_with_the_profiles_context_size(root, chat_tokenizer):
    """S3: a prompt longer than the profile's ctx_size is refused before the model starts."""
    session = unstarted(root, chat_tokenizer)
    profile(root, extra="ctx_size = 16")
    outcome = refused_or_started(session)
    assert re.fullmatch(
        r"Pair 1's positive prompt is \d+ tokens, more than this profile's context size, 16\.",
        str(outcome),
    ), f"not refused: {outcome!r}"
    assert not (root / "local" / "vectors").exists()


def symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks need privileges on Windows")


def test_a_dangling_link_at_the_vectors_name_is_refused_and_left(root, chat_tokenizer, tmp_path):
    """S8: the check is os.path.lexists, so a link to nothing counts as an existing vector."""
    session = unstarted(root, chat_tokenizer)
    profile(root, extra="ctx_size = 256")
    symlink(root / GGUF, tmp_path / "nowhere.gguf")
    outcome = refused_or_started(session)
    assert str(outcome).startswith(f"{GGUF} already exists"), f"not refused: {outcome!r}"
    assert (root / GGUF).is_symlink() and not (tmp_path / "nowhere.gguf").exists()


def test_a_link_at_the_records_name_is_replaced_not_followed(project, chat_tokenizer, tmp_path):
    """S8: the record is written beside its name and renamed over it: a link there is
    replaced by the record, and the file it pointed to is untouched."""
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("made up\n", encoding="utf-8")
    symlink(project / RECORD, elsewhere)
    made(project, chat_tokenizer)
    assert not (project / RECORD).is_symlink()
    assert elsewhere.read_text(encoding="utf-8") == "made up\n"


def test_olmo3s_vector_names_olmo3(root, chat_tokenizer):
    """Mutation 17: the record's model_hint and the file's are the model's type."""
    pytest.importorskip("torch")
    import transformers

    from clarity_chat import steering

    hf_root(root)
    weights = root / "models" / "tiny-hf"
    config = transformers.Olmo3Config(
        vocab_size=258,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=257,
        bos_token_id=256,
        pad_token_id=257,
    )
    transformers.Olmo3ForCausalLM(config).save_pretrained(weights)
    chat_tokenizer.tok.save_pretrained(weights)
    profile(root, extra="ctx_size = 256")
    (root / "brief.toml").write_text(PAIRS, encoding="utf-8")
    record = made(root, chat_tokenizer)
    assert record["vector"]["model_hint"] == "olmo3"
    assert steering.load(root / GGUF, record["vector"]["sha256"]).model_hint == "olmo3"
