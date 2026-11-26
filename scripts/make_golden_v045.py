"""Write v0.4.5's golden log and the vector it was steered with, into one folder.

    uv run --extra hf python scripts/make_golden_v045.py "$TMPDIR/golden-v045"

It writes two files there and nowhere else: `v0.4.5.jsonl`, the golden log, and
`brief-hf-chat.gguf`, the vector. The fixtures `tests/fixtures/golden/v0.4.5.jsonl` and
`tests/fixtures/steering/brief-hf-chat.gguf` were written by v0.4.5's code (`308b5cc`); this
script doesn't touch them, and refuses a folder under `tests/fixtures/`. Compare its
output with them by hand.

v0.4.5 adds making a vector; what a log records doesn't change. So the log holds the
release's new path: make reads a made-up pairs file (two system pairs, two messages, two
openings) with a tiny random Llama on the CPU, shaped as the golden tests' steerable model
(width 576, 30 layers, the tests' chat tokenizer), and the profile takes the snippet make
prints, with steering_scale = 1.0. Then a steered message, a mark, a note and an identical
replay; /set steering_scale 0 and /regen same, the guides' way to compare an unsteered
reply. tests/test_golden.py copies the kept vector to where the log says.

The chat runs on FakeBackend, which reads no weights: before it starts, the model folder
holds only the config.json the golden tests write, so the log's facts are the tests'.
Machine and client facts are made up; the temporary project path becomes "/project".
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import transformers

# The repo root, so `tests` imports when this runs as a script from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarity_chat import session as session_module  # noqa: E402
from clarity_chat.commands import Commands  # noqa: E402
from clarity_chat.maker import make  # noqa: E402
from clarity_chat.models import read_lock, write_lock_entry  # noqa: E402
from tests.conftest import MODELS, REPO, TOY_TEMPLATE, build_tokenizer, fake_fetch  # noqa: E402
from tests.fakes import FakeBackend  # noqa: E402

# The made-up machine and client of every golden log.
MACHINE = {
    "os": "Darwin",
    "os_release": "0.0.0",
    "os_version": "made up for the golden log",
    "platform": "golden-fixture",
    "machine": "arm64",
    "processor": "arm",
    "cpu_count": 8,
    "python": "3.12.0",
    "macos": {"version": "0.0", "build": "GOLDEN", "chip": "made up"},
    "hostname_hash": "000000000000",
}
CLIENT = {"commit": "0" * 40, "dirty": False}

PAIRS = """\
description = "Short, plain replies against long, ornate ones."
author = "a made-up file for the golden log"
messages = ["How do I boil an egg?", "What should I pack for a day hike?"]
openings = ["", "Well,"]

[[systems]]
positive = "You are an assistant. You reply in as few plain words as you can."
negative = "You are an assistant. You reply at length, in rich and decorated language."

[[systems]]
positive = "You are a librarian. You keep every answer short and plain."
negative = "You are a librarian. You make every answer long and elaborate."
"""

PROFILE = """\
description = "A made-up profile for the golden log"
backend = "transformers"
device = "cpu"
model = "tiny"
mode = "chat"
system = "Be brief."
ctx_size = 256

[settings]
max_tokens = 16
seed = 42
"""
# What tests/test_golden.py's steerable fixture writes, and the shape the vector is made at.
CONFIG = {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30}
FIXTURES = REPO / "tests" / "fixtures"


def main(out: Path) -> None:
    if out.resolve().is_relative_to(FIXTURES.resolve()):
        sys.exit(f"{out} is under {FIXTURES}: write to a scratch folder and compare by hand.")
    session_module.machine = lambda: MACHINE
    session_module.client_git = lambda: CLIENT
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "project"
        root.mkdir()
        for name in ("defaults.toml", "server.toml"):
            shutil.copy(REPO / name, root / name)
        (root / "models.toml").write_text(MODELS, encoding="utf-8")
        (root / "profiles").mkdir()
        fake_fetch(root)
        weights = root / "models" / "tiny-hf"
        tokenizer = build_tokenizer(Path(tmp) / "tok", True, True, TOY_TEMPLATE)
        config = transformers.LlamaConfig(
            vocab_size=258,
            hidden_size=CONFIG["hidden_size"],
            intermediate_size=32,
            num_hidden_layers=CONFIG["num_hidden_layers"],
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            eos_token_id=257,
            bos_token_id=256,
        )
        torch.manual_seed(0)
        transformers.LlamaForCausalLM(config).save_pretrained(weights)
        tokenizer.tok.save_pretrained(weights)
        entry = read_lock(root)["models"]["tiny"]
        write_lock_entry(root, "tiny", {**entry, "source_dir": "models/tiny-hf"})
        (root / "profiles" / "hf-chat.toml").write_text(PROFILE, encoding="utf-8", newline="\n")
        (root / "local" / "pairs").mkdir(parents=True)
        (root / "local" / "pairs" / "brief.toml").write_text(PAIRS, encoding="utf-8", newline="\n")

        maker = session_module.Session(root, tokenizer_loader=lambda r, v: tokenizer)
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                make(maker, root / "local" / "pairs" / "brief.toml", "hf-chat")
        finally:
            maker.close()
        readout = printed.getvalue()
        snippet = readout[readout.index("[steering]") :]
        vector = root / "local" / "vectors" / "brief-hf-chat.gguf"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(vector, out / vector.name)

        shutil.rmtree(weights)  # FakeBackend reads no weights: the golden tests' folder
        weights.mkdir()
        (weights / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        steered = PROFILE.replace("seed = 42\n", "seed = 42\nsteering_scale = 1.0\n")
        steered += "\n" + snippet
        (root / "profiles" / "hf-chat.toml").write_text(steered, encoding="utf-8", newline="\n")

        replies = [" Boil it.", " Boil it.", " Boil, cool."]  # under max_tokens: eos
        backend = FakeBackend(replies, tokenizer=tokenizer)  # 2nd: the replay
        backend.name = "transformers"
        session = session_module.Session(
            root, backend=backend, tokenizer_loader=lambda r, v: tokenizer
        )
        c = Commands(session)

        def run(line: str) -> dict[str, Any]:
            return c.run(*line.partition(" ")[::2], lambda e: None)

        run("profile hf-chat")
        first = run("say How do I boil an egg?")["call"]["call_id"]
        run(f"mark {first} plain")
        run(f"note {first} Made by v0.4.5's code, steered with a vector it made.")
        replay = run(f"replay {first}")
        run("set steering_scale 0")
        regen = run("regen same")
        compared = run(f"compare {first}")
        log = session.conversation.log.path
        text = log.read_text(encoding="utf-8").replace(str(root.resolve()), "/project")
        text = text.replace(str(root), "/project")
        text = text.replace(str(Path(tmp) / "tok"), "/tokenizer")
        assert tmp not in text  # a name is never a path
        session.close()
    (out / "v0.4.5.jsonl").write_text(text, encoding="utf-8", newline="\n")
    kinds = [json.loads(line)["type"] for line in text.splitlines()]
    print(readout, file=sys.stderr)
    print(len(kinds), "records:", " ".join(kinds), file=sys.stderr)
    print("replay:", json.dumps(replay, default=str)[:300], file=sys.stderr)
    print("regen:", json.dumps(regen, default=str)[:300], file=sys.stderr)
    print("compare:", json.dumps(compared, default=str)[:600], file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run --extra hf python scripts/make_golden_v045.py OUT_FOLDER")
    main(Path(sys.argv[1]))
