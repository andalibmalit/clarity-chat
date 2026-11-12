"""From a profile and a conversation to the exact prompt text and token IDs.

The client renders and tokenizes every prompt itself, with the model's Hugging Face
tokenizer and `add_special_tokens=False`; backends only ever receive token IDs.
"""

from __future__ import annotations

import bisect
import json
import os
import secrets
from collections.abc import Sequence
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


def from_pretrained(cls: Any, folder: Path, **kwargs: Any) -> Any:
    """`cls.from_pretrained(folder)` without the folder's own Python code. Without
    trust_remote_code=False, Transformers asks on stdin whether to run it, and a "y" typed
    in the chat would. Clarity runs only its own code and locked libraries."""
    try:
        return cls.from_pretrained(folder, trust_remote_code=False, **kwargs)
    except ValueError as e:
        if "custom code" not in str(e):
            raise
        raise ConfigError(
            f"{folder} needs its own Python code to load, which Clarity doesn't run."
        ) from e


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
        tok = from_pretrained(AutoTokenizer, folder, local_files_only=True)
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


# --- where each message sits in a call's prompt ----------------------------------------------

Part = tuple[dict[str, Any], str]  # what a stretch of prompt text is, and the text


def message_spans(
    tokenizer: Tokenizer,
    profile: Profile,
    turns: list[tuple[str, str]],
    turn_ids: Sequence[str | None],
    prompt: Prompt,
    template: Tokenizer | None = None,
) -> list[dict[str, Any]] | dict[str, str]:
    """Where each message, and the text around it, sits in `prompt`: a list of entries that
    covers every prompt token in order, each with half-open `chars` (in `prompt.text`) and
    `tokens` (in `prompt.ids`) ranges. Or `{"unavailable": reason}`: never a guess, and never
    an error, since a call must not fail over its spans."""
    try:
        if profile.mode == "text":
            parts: list[Part] | str = _text_parts(profile, turns, turn_ids)
        else:
            parts = _chat_parts(template or tokenizer, profile, turns, turn_ids, prompt.text)
        if isinstance(parts, str):
            return {"unavailable": parts}
        if "".join(text for _, text in parts) != prompt.text:
            return {"unavailable": "the parts don't rebuild the prompt text"}
        merged: list[Part] = []
        for info, text in parts:  # neighbouring template or profile text is one entry
            if merged and info["kind"] != "message" and merged[-1][0] == info:
                merged[-1] = (info, merged[-1][1] + text)
            else:
                merged.append((info, text))
        return _token_ranges(tokenizer, merged, prompt)
    except Exception as e:  # a template that fails on placeholders, an odd tokenizer
        return {"unavailable": f"{type(e).__name__}: {e}"}


def _labels(
    profile: Profile, turns: list[tuple[str, str]], turn_ids: Sequence[str | None]
) -> list[dict[str, Any]]:
    """Each chat message's entry fields, in chat_messages' order."""
    labels: list[dict[str, Any]] = []
    if profile.system is not None:
        labels.append({"role": "system", "source": "system"})
    for i, m in enumerate(profile.priming_messages):
        labels.append({"role": m["role"], "source": f"priming {i + 1}"})
    for (role, _), turn_id in zip(turns, turn_ids, strict=True):
        labels.append({"role": role, "turn_id": turn_id})
    return labels


def _text_parts(
    profile: Profile, turns: list[tuple[str, str]], turn_ids: Sequence[str | None]
) -> list[Part]:
    """render_text's pieces: the profile's header, names and line ends, and each message."""
    profile_text = {"kind": "profile"}
    parts: list[Part] = [(profile_text, profile.header)]
    # Priming is free text, usually lines from both speakers: no role.
    parts.append(({"kind": "message", "source": "priming"}, profile.priming_text))
    for (role, text), turn_id in zip(turns, turn_ids, strict=True):
        name = f"{profile.user_name}: " if role == "user" else f"{profile.assistant_name}:"
        parts.append((profile_text, name))
        parts.append(({"kind": "message", "role": role, "turn_id": turn_id}, text))
        parts.append((profile_text, "\n"))
    parts.append((profile_text, f"{profile.assistant_name}:"))
    return [(info, text) for info, text in parts if text or info.get("turn_id") is not None]


def _chat_parts(
    template: Tokenizer,
    profile: Profile,
    turns: list[tuple[str, str]],
    turn_ids: Sequence[str | None],
    rendered: str,
) -> list[Part] | str:
    """The chat template's own text and each message, found by rendering again with each
    message replaced by a unique placeholder. Proved by rebuilding `rendered` exactly: each
    message as written, else stripped of edge whitespace (marked `trimmed`)."""
    messages = chat_messages(profile, turns)
    nonce = secrets.token_hex(6)
    # Mixed case, so a template that upper- or lowercases content changes every placeholder,
    # even when the random part is all digits.
    marks = [f"Clarity{nonce}M{i}z" for i in range(len(messages))]
    placeheld = template.tok.apply_chat_template(
        [{**m, "content": mark} for m, mark in zip(messages, marks, strict=True)],
        tokenize=False,
        add_generation_prompt=True,
        **profile.template_kwargs,
    )
    pieces, end = [], 0
    for i, mark in enumerate(marks):
        at = placeheld.find(mark)
        if at < end or placeheld.count(mark) != 1:
            return f"the chat template changes, leaves out, repeats or reorders message {i + 1}"
        pieces.append(placeheld[end:at])
        end = at + len(mark)
    pieces.append(placeheld[end:])
    parts: list[Part] = []
    pos = 0
    labels = _labels(profile, turns, turn_ids)
    for i, (message, label) in enumerate(zip(messages, labels, strict=True)):
        parts.append(({"kind": "template"}, pieces[i]))
        if not rendered.startswith(pieces[i], pos):
            return f"the chat template writes other text around message {i + 1}"
        pos += len(pieces[i])
        content = message["content"]
        for text in dict.fromkeys([content, content.strip()]):
            if rendered.startswith(text + pieces[i + 1], pos):
                break
        else:
            return f"the chat template changes message {i + 1} ({label['role']}) itself"
        entry = {"kind": "message", **label, **({"trimmed": True} if text != content else {})}
        parts.append((entry, text))
        pos += len(text)
    if rendered[pos:] != pieces[-1]:
        return "the chat template writes other text after the last message"
    parts.append(({"kind": "template"}, pieces[-1]))
    return [(info, text) for info, text in parts if text or info["kind"] == "message"]


def _token_ranges(
    tokenizer: Tokenizer, parts: list[Part], prompt: Prompt
) -> list[dict[str, Any]] | dict[str, str]:
    """Characters to tokens, with the fast tokenizer's offsets. A token belongs to the part
    its first character is in; a part whose last token runs past its end says so."""
    encoded = tokenizer.tok(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    bos = 1 if prompt.bos.get("prepended") else 0
    if ids != prompt.ids[bos:]:
        return {"unavailable": "tokenizing with offsets gave other IDs than the prompt's"}
    reached = 0
    for start, end in offsets:
        if start > reached or end < start:
            return {"unavailable": "the tokenizer's character offsets leave gaps"}
        reached = max(reached, end)
    if reached != len(prompt.text):
        return {"unavailable": "the tokenizer's character offsets don't reach the end"}
    bounds, pos = [], 0
    for _, text in parts:
        bounds.append((pos, pos + len(text)))
        pos += len(text)
    owner, k = [], 0
    for start, _ in offsets:
        while k < len(bounds) - 1 and not bounds[k][0] <= start < bounds[k][1]:
            k += 1
        owner.append(k)
    entries = [{"kind": "bos", "chars": [0, 0], "tokens": [0, 1]}] if bos else []
    for k, ((info, _), (first, last)) in enumerate(zip(parts, bounds, strict=True)):
        start, end = bisect.bisect_left(owner, k), bisect.bisect_right(owner, k)
        entry = {**info, "chars": [first, last], "tokens": [start + bos, end + bos]}
        if end > start and offsets[end - 1][1] > last:
            entry["edge_token"] = "end"
        entries.append(entry)
    return entries
