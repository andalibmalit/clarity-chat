"""From a profile and a conversation to the exact prompt text and token IDs.

The client renders and tokenizes every prompt itself, with the model's Hugging Face
tokenizer and `add_special_tokens=False`; backends only ever receive token IDs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache, cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any

from clarity_chat.files import read_text, sha256_text
from clarity_chat.profiles import ConfigError, Profile

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # no "PyTorch was not found" banner

# Files a tokenizer (and its chat template) can need. Weights are never part of it.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
]


@cache
def library_versions() -> dict[str, str]:
    """Rendering and tokenization depend on these, so every call logs them."""
    return {name: version(name) for name in ("transformers", "tokenizers", "jinja2")}


@dataclass
class Tokenizer:
    """A Hugging Face tokenizer, loaded from a local folder, plus where it came from."""

    tok: Any  # transformers.PreTrainedTokenizerBase
    source: dict[str, str]  # {"repo", "revision"} or {"path"}
    config: dict[str, Any]  # tokenizer_config.json as written

    @classmethod
    def from_dir(cls, folder: Path, source: dict[str, str] | None = None) -> Tokenizer:
        from transformers import AutoTokenizer

        config_path = folder / "tokenizer_config.json"
        config = json.loads(read_text(config_path)) if config_path.is_file() else {}
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        return cls(tok=tok, source=source or {"path": folder.as_posix()}, config=config)

    @classmethod
    def from_hub(cls, repo: str, revision: str) -> Tokenizer:
        """Load from the Hugging Face cache only; `clarity fetch` puts the files there."""
        from huggingface_hub import snapshot_download

        try:
            folder = snapshot_download(
                repo, revision=revision, allow_patterns=TOKENIZER_FILES, local_files_only=True
            )
        except Exception as e:  # huggingface_hub raises several types for "not cached"
            raise ConfigError(
                f"Tokenizer {repo}@{revision[:12]} is not in the Hugging Face cache. "
                f"Run `clarity fetch` for the model that uses it. ({e})"
            ) from e
        return cls.from_dir(Path(folder), {"repo": repo, "revision": revision})

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        """The text these IDs stand for, special tokens included and spaces untouched."""
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    @property
    def bos(self) -> tuple[str | None, int | None]:
        return self.tok.bos_token, self.tok.bos_token_id

    @cached_property
    def special_token_texts(self) -> list[str]:
        """Texts the tokenizer turns into control tokens wherever they appear."""
        texts = {t.content for t in self.tok.added_tokens_decoder.values() if t.special}
        texts.update(self.tok.all_special_tokens)
        return sorted(t for t in texts if t)

    def special_tokens_in(self, text: str) -> list[str]:
        return [t for t in self.special_token_texts if t in text]

    def probe_adds_bos(self) -> bool:
        """Does the tokenizer itself put BOS first when asked to add special tokens?"""
        _, bos_id = self.bos
        with_special = self.tok.encode("a", add_special_tokens=True)
        without = self.tok.encode("a", add_special_tokens=False)
        return bos_id is not None and with_special[:1] == [bos_id] and without[:1] != [bos_id]


def bos_decision(tokenizer: Tokenizer, mode: str) -> dict[str, Any]:
    """Whether to prepend BOS, decided from the tokenizer's config and logged with every call.

    Text mode: as the tokenizer config says (`add_bos_token`), cross-checked against what
    the tokenizer does; a disagreement is an error. Chat mode: never; the chat template
    writes any BOS itself, and Hugging Face adds none when it tokenizes a chat.
    """
    bos_token, bos_id = tokenizer.bos
    config_add = tokenizer.config.get("add_bos_token")
    probe = tokenizer.probe_adds_bos()
    decision: dict[str, Any] = {
        "mode": mode,
        "bos_token": bos_token,
        "bos_token_id": bos_id,
        "config_add_bos_token": config_add,
        "probe_adds_bos": probe,
    }
    if mode == "chat":
        decision["add"] = False
        decision["reason"] = "chat mode: the template writes any BOS; Hugging Face adds none"
        return decision
    if config_add is not None and bool(config_add) != probe:
        raise ConfigError(
            f"Tokenizer {tokenizer.source}: tokenizer_config.json says add_bos_token = "
            f"{config_add}, but the tokenizer {'adds' if probe else 'does not add'} BOS. "
            "Can't decide BOS handling safely."
        )
    if config_add is not None:
        decision["add"] = bool(config_add)
        decision["reason"] = "tokenizer_config.json add_bos_token"
    else:
        decision["add"] = probe
        decision["reason"] = "no add_bos_token in tokenizer_config.json: the tokenizer's behavior"
    if decision["add"] and bos_id is None:
        raise ConfigError(f"Tokenizer {tokenizer.source}: add_bos_token is set, but no BOS token.")
    return decision


@dataclass
class Prompt:
    """Exactly what a backend gets: the text, its token IDs, and how they were made."""

    text: str  # the rendered prompt (without a prepended BOS)
    ids: list[int]
    bos: dict[str, Any]  # the BOS decision, plus whether BOS was prepended
    template: dict[str, Any] | None  # chat mode: repo, revision, kwargs
    # Control-token texts found in content (system, priming, header, turns), as
    # {"in": where, "tokens": [...]}: the tokenizer reads them as control tokens, not text.
    special_in_content: list[dict[str, Any]]


def tokenize(tokenizer: Tokenizer, text: str, mode: str) -> tuple[list[int], dict[str, Any]]:
    decision = bos_decision(tokenizer, mode)
    ids = tokenizer.encode(text)
    bos_id = decision["bos_token_id"]
    decision["text_starts_with_bos"] = bos_id is not None and ids[:1] == [bos_id]
    decision["prepended"] = bool(decision["add"]) and not decision["text_starts_with_bos"]
    if decision["prepended"]:
        ids = [bos_id, *ids]
    return ids, decision


def render_text(profile: Profile, turns: list[tuple[str, str]]) -> str:
    """The text-mode document: header, priming, one line per turn, then `AssistantName:`.

    A model turn is `AssistantName:` followed by its output exactly as returned (which
    usually starts with a space). Header and priming carry their own newlines.
    """
    parts = [profile.header, profile.priming_text]
    for role, text in turns:
        if role == "user":
            parts.append(f"{profile.user_name}: {text}\n")
        else:
            parts.append(f"{profile.assistant_name}:{text}\n")
    parts.append(f"{profile.assistant_name}:")
    return "".join(parts)


def chat_template_source(template: Tokenizer, profile: Profile) -> dict[str, Any]:
    """Where the chat template came from, logged with every call: the repo and commit (or
    folder), the kwargs, and the template's sha256."""
    text = template.tok.chat_template
    if not isinstance(text, str) or not text:
        raise ConfigError(
            f"Profile {profile.name}: {template.source} has no single chat template "
            f"(found {type(text).__name__}). Set template_repo to a repo that has one."
        )
    return {**template.source, "kwargs": profile.template_kwargs, "sha256": sha256_text(text)}


def unread_kwargs(template_text: str, kwargs: dict[str, Any]) -> list[str]:
    """The template_kwargs the chat template never reads: a misspelt name changes nothing,
    silently. Parsed with the Jinja environment Transformers renders with."""
    if not kwargs:
        return []
    from jinja2 import meta
    from transformers.utils.chat_template_utils import _compile_jinja_template

    environment = _compile_jinja_template(template_text).environment
    read = meta.find_undeclared_variables(environment.parse(template_text))
    return sorted(k for k in kwargs if k not in read)


def chat_messages(profile: Profile, turns: list[tuple[str, str]]) -> list[dict[str, str]]:
    """System (if set), then priming, then the conversation, in that order."""
    messages = []
    if profile.system is not None:
        messages.append({"role": "system", "content": profile.system})
    messages.extend(dict(m) for m in profile.priming_messages)
    messages.extend({"role": role, "content": text} for role, text in turns)
    return messages


def render_chat(template: Tokenizer, profile: Profile, turns: list[tuple[str, str]]) -> str:
    """The chat template's output for this conversation, ending with the generation prompt."""
    try:
        text = template.tok.apply_chat_template(
            chat_messages(profile, turns),
            tokenize=False,
            add_generation_prompt=True,
            **profile.template_kwargs,
        )
    except Exception as e:  # Jinja errors, raise_exception() in the template, bad kwargs
        raise ConfigError(
            f"Profile {profile.name}: the chat template failed: {type(e).__name__}: {e}"
        ) from e
    if not isinstance(text, str):
        raise ConfigError(f"Profile {profile.name}: the chat template did not return text.")
    return text


def special_in_content(
    tokenizer: Tokenizer, profile: Profile, turns: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    if profile.mode == "chat":
        pieces = [("system", profile.system or "")]
        pieces += [
            (f"priming {i + 1}", m["content"]) for i, m in enumerate(profile.priming_messages)
        ]
    else:
        pieces = [("header", profile.header), ("priming", profile.priming_text)]
    pieces += [(f"turn {i + 1} ({role})", text) for i, (role, text) in enumerate(turns)]
    found = []
    for where, text in pieces:
        tokens = tokenizer.special_tokens_in(text)
        if tokens:
            found.append({"in": where, "tokens": tokens})
    return found


def build_prompt(
    tokenizer: Tokenizer,
    profile: Profile,
    turns: list[tuple[str, str]],
    template: Tokenizer | None = None,
) -> Prompt:
    """The prompt for `turns`. `tokenizer` is always the model's; in chat mode `template`
    (default: the same tokenizer) supplies only the chat template."""
    if profile.mode == "text":
        text = render_text(profile, turns)
        ids, bos = tokenize(tokenizer, text, "text")
        found = special_in_content(tokenizer, profile, turns)
        return Prompt(text=text, ids=ids, bos=bos, template=None, special_in_content=found)
    source = template or tokenizer
    template_info = chat_template_source(source, profile)  # checks there is a template first
    text = render_chat(source, profile, turns)
    ids, bos = tokenize(tokenizer, text, "chat")
    found = special_in_content(tokenizer, profile, turns)
    return Prompt(text=text, ids=ids, bos=bos, template=template_info, special_in_content=found)
