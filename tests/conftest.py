"""A small self-contained project for each test: config, profiles, tokenizers, no network."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # unit tests must never reach the network
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from clarity_chat.render import Tokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"

TEXT_PROFILE = """\
backend = "llama.cpp"
model = "tiny"
mode = "text"
header = "A conversation.\\n\\n"
user_name = "User"
assistant_name = "Assistant"

[settings]
temperature = 0.6
top_p = 0.95
top_k = 0
max_tokens = 16
seed = 42
samplers = ["top_k", "top_p", "min_p", "temperature"]
"""

CHAT_PROFILE = """\
backend = "llama.cpp"
model = "tiny"
mode = "chat"
system = "Be brief."

[settings]
max_tokens = 16
seed = 42
"""

# A toy chat template: <role>content</s> per message, then <assistant> to generate.
TOY_TEMPLATE = (
    "{% if greeting is defined %}[{{ greeting }}]{% endif %}"
    "{% for m in messages %}<{{ m.role }}>{{ m.content }}</s>{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)

MODELS = """\
[tiny]
route = "convert"
repo = "test/tiny"
revision = "0123456789abcdef0123456789abcdef01234567"
tokenizer_repo = "test/tiny"
tokenizer_revision = "0123456789abcdef0123456789abcdef01234567"
"""


def build_tokenizer(
    folder: Path, config_add_bos: bool | None, adds_bos: bool, chat_template: str | None = None
) -> Tokenizer:
    """A byte-level tokenizer (one token per byte) with <s> and </s>, saved and reloaded.

    `adds_bos`: whether the tokenizer itself prepends <s> with add_special_tokens=True.
    `config_add_bos`: what tokenizer_config.json says (None = no add_bos_token key).
    `chat_template`: saved with the tokenizer, if given.
    """
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers import decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    raw = RawTokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    raw.decoder = decoders.ByteLevel()
    raw.add_special_tokens(["<s>", "</s>"])
    if adds_bos:
        raw.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", raw.token_to_id("<s>"))]
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<s>", eos_token="</s>")
    if chat_template is not None:
        fast.chat_template = chat_template
    fast.save_pretrained(folder)
    config_path = folder / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("add_bos_token", None)
    if config_add_bos is not None:
        config["add_bos_token"] = config_add_bos
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return Tokenizer.from_dir(folder)


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory) -> Tokenizer:
    """No BOS anywhere: config has no add_bos_token and the tokenizer adds none."""
    return build_tokenizer(tmp_path_factory.mktemp("tok-plain"), None, False)


@pytest.fixture(scope="session")
def bos_tokenizer(tmp_path_factory) -> Tokenizer:
    """Config says add_bos_token = true and the tokenizer does prepend <s>."""
    return build_tokenizer(tmp_path_factory.mktemp("tok-bos"), True, True)


@pytest.fixture(scope="session")
def chat_tokenizer(tmp_path_factory) -> Tokenizer:
    """Config says add_bos_token = true (text mode would prepend <s>), with TOY_TEMPLATE."""
    return build_tokenizer(tmp_path_factory.mktemp("tok-chat"), True, True, TOY_TEMPLATE)


@pytest.fixture(scope="session")
def olmo_tokenizer() -> Tokenizer:
    """OLMo 3 Instruct's real tokenizer and chat template (see fixtures/olmo3-instruct)."""
    return Tokenizer.from_dir(FIXTURES / "olmo3-instruct")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A project folder: the real defaults.toml and server.toml, a test model and profile."""
    for name in ("defaults.toml", "server.toml"):
        shutil.copy(REPO / name, tmp_path / name)
    (tmp_path / "models.toml").write_text(MODELS, encoding="utf-8")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "text.toml").write_text(TEXT_PROFILE, encoding="utf-8")
    (tmp_path / "profiles" / "chat.toml").write_text(CHAT_PROFILE, encoding="utf-8")
    return tmp_path


def fake_fetch(root: Path, name: str = "tiny") -> Path:
    """Pretend `name` was fetched: a dummy weights file and its models.lock entry."""
    from clarity_chat.models import write_lock_entry
    from clarity_chat.profiles import load_models

    model = load_models(root)[name]
    weights = root / "models" / name / f"{name}-bf16.gguf"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"GGUF fake weights")
    stat = weights.stat()
    write_lock_entry(
        root,
        name,
        {
            **model.to_json(),
            "gguf": {
                "path": weights.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": "0" * 64,
            },
            "recommended": {
                "source": {
                    "file": "generation_config.json",
                    "repo": model.repo,
                    "revision": model.revision,
                },
                "values": {"temperature": 0.7, "top_p": 0.8},
                "not_applied": {"max_new_tokens": 2048},
                "notes": [],
            },
            "llama_cpp": {"commit": "f" * 40},
            "converter": {"torch": "0", "transformers": "0", "packages": {}},
        },
    )
    return weights


@pytest.fixture
def fetched_root(root: Path) -> Path:
    fake_fetch(root)
    return root
