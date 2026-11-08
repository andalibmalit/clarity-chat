"""Where the weights come from: `fetch`, `verify` and the generated models.lock.

models.toml (hand-written) says what to get; models.lock (generated, JSON) records what
was resolved: commits, file paths, sizes, sha256, the llama.cpp commit and the
converter's package versions. The sha256 is computed once, at fetch; a server start only
compares size and modified time, and `verify` re-hashes on demand.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from clarity_chat.files import read_text, sha256_file, write_text_atomic
from clarity_chat.logs import git_state, now_utc
from clarity_chat.profiles import ConfigError, ModelEntry, ServerConfig, check_setting
from clarity_chat.render import TOKENIZER_FILES

# generation_config.json key -> Clarity setting. Everything else in the file is listed as
# not applied, so nothing is dropped silently.
GENERATION_CONFIG_SETTINGS = {
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "repetition_penalty": "repeat_penalty",
}
LOCK_FILE = "models.lock"
LOCK_SCHEMA_VERSION = 1
# What a conversion needs from the source repo: config, tokenizer and weights.
SOURCE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]
CONVERTER_PYTHON = "3.13"


def recommended_settings(source_dir: Path, repo: str, revision: str) -> dict[str, Any]:
    """The model's own recommended sampling settings, from its generation_config.json."""
    path = source_dir / "generation_config.json"
    source = {"file": "generation_config.json", "repo": repo, "revision": revision}
    if not path.is_file():
        return {"source": source, "values": {}, "not_applied": {}, "notes": ["no such file"]}
    return map_generation_config(json.loads(read_text(path)), source)


def map_generation_config(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    not_applied: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in config.items():
        name = GENERATION_CONFIG_SETTINGS.get(key)
        if name is None:
            if key != "do_sample" and not key.startswith("_") and key != "transformers_version":
                not_applied[key] = value
            continue
        try:
            values[name] = check_setting(name, value, "generation_config.json")
        except ConfigError as e:
            not_applied[key] = value
            notes.append(str(e))
    if config.get("do_sample") is False:
        values["temperature"] = 0.0
        notes.append("do_sample is false: greedy decoding, so temperature 0")
    elif "do_sample" not in config and values:
        notes.append(
            "do_sample is not set (Transformers would decode greedily); sampling as listed"
        )
    return {"source": source, "values": values, "not_applied": not_applied, "notes": notes}


def read_lock(root: Path) -> dict[str, Any]:
    path = root / LOCK_FILE
    if not path.is_file():
        return {"schema_version": LOCK_SCHEMA_VERSION, "models": {}}
    return json.loads(read_text(path))


def write_lock_entry(root: Path, name: str, entry: dict[str, Any]) -> None:
    lock = read_lock(root)
    lock["models"][name] = entry
    lock["models"] = dict(sorted(lock["models"].items()))
    write_text_atomic(root / LOCK_FILE, json.dumps(lock, indent=2, ensure_ascii=False) + "\n")


def lock_entry(root: Path, model: ModelEntry) -> dict[str, Any]:
    """The lock entry for `model`; an error if it's missing or was fetched from other settings."""
    entry = read_lock(root)["models"].get(model.name)
    if entry is None:
        raise ConfigError(f"Model {model.name!r} is not fetched. Run `clarity fetch {model.name}`.")
    for key in ("route", "repo", "revision", "tokenizer_repo", "tokenizer_revision"):
        if entry.get(key) != getattr(model, key):
            raise ConfigError(
                f"models.lock has {model.name!r} with {key} = {entry.get(key)!r}, but models.toml "
                f"says {getattr(model, key)!r}. Run `clarity fetch {model.name}` again."
            )
    return entry


def weights_path(root: Path, entry: dict[str, Any]) -> Path:
    return root / entry["gguf"]["path"]


def check_weights_quick(root: Path, entry: dict[str, Any]) -> Path:
    """Size and modified time against models.lock (a full re-hash is `clarity verify`)."""
    path = weights_path(root, entry)
    if not path.is_file():
        raise ConfigError(f"{path} is missing. Run `clarity fetch {entry['name']}`.")
    stat = path.stat()
    if stat.st_size != entry["gguf"]["size"] or stat.st_mtime_ns != entry["gguf"]["mtime_ns"]:
        raise ConfigError(
            f"{path} changed since it was fetched (size or modified time differs from "
            f"models.lock). Run `clarity verify {entry['name']}` or fetch it again."
        )
    return path


def verify(root: Path, name: str) -> tuple[bool, str]:
    entry = read_lock(root)["models"].get(name)
    if entry is None:
        return False, f"{name}: not in models.lock"
    path = weights_path(root, entry)
    if not path.is_file():
        return False, f"{name}: {path} is missing"
    digest = sha256_file(path)
    if digest != entry["gguf"]["sha256"]:
        return False, f"{name}: sha256 {digest} != models.lock {entry['gguf']['sha256']}"
    return True, f"{name}: sha256 matches ({digest[:16]}…)"


def _download(repo: str, revision: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def fetch(root: Path, model: ModelEntry, server: ServerConfig) -> dict[str, Any]:
    if model.route == "convert":
        return fetch_convert(root, model, server)
    raise ConfigError(f"{model.name}: the {model.route!r} route is not built yet (step 4).")


def fetch_convert(root: Path, model: ModelEntry, server: ServerConfig) -> dict[str, Any]:
    checkout = git_state(server.llama_cpp_dir)
    if checkout.get("commit") != server.llama_cpp_commit:
        raise ConfigError(
            f"{server.llama_cpp_dir} is at {checkout.get('commit')}, but server.toml pins "
            f"{server.llama_cpp_commit}."
        )
    print(f"Downloading {model.repo}@{model.revision[:12]} …", flush=True)
    source = _download(model.repo, model.revision, SOURCE_PATTERNS)
    tokenizer = source
    if (model.tokenizer_repo, model.tokenizer_revision) != (model.repo, model.revision):
        print(f"Downloading tokenizer {model.tokenizer_repo}@{model.tokenizer_revision[:12]} …")
        tokenizer = _download(model.tokenizer_repo, model.tokenizer_revision, TOKENIZER_FILES)

    out_dir = root / "models" / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model.name}-bf16.gguf"
    partial = out_dir / f"{out.name}.partial"
    versions = out_dir / "converter-versions.json"
    converter = server.llama_cpp_dir / "convert_hf_to_gguf.py"
    requirements = server.llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    shim = Path(__file__).with_name("convert_shim.py")
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        CONVERTER_PYTHON,
        "--with-requirements",
        str(requirements),
        "python",
        "-I",  # nothing from Clarity's folder or environment leaks into the converter
        str(shim),
        str(versions),
        str(converter),
        str(source),
        "--outtype",
        "bf16",
        "--outfile",
        str(partial),
    ]
    print("Converting:", " ".join(command), flush=True)
    with (out_dir / "convert.log").open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        if proc.wait() != 0:
            raise ConfigError(f"The converter failed; see {out_dir / 'convert.log'}.")
    os.replace(partial, out)
    print(f"Hashing {out.name} …", flush=True)
    stat = out.stat()
    converter_env = json.loads(read_text(versions))
    packages = converter_env["packages"]
    entry = {
        "name": model.name,
        "route": model.route,
        "repo": model.repo,
        "revision": model.revision,
        "tokenizer_repo": model.tokenizer_repo,
        "tokenizer_revision": model.tokenizer_revision,
        "source_dir": source.as_posix(),
        "tokenizer_dir": tokenizer.as_posix(),
        "gguf": {
            "path": out.relative_to(root).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(out),
            "outtype": "bf16",
        },
        "recommended": recommended_settings(source, model.repo, model.revision),
        "llama_cpp": {"commit": checkout["commit"], "dirty": checkout.get("dirty")},
        "converter": {
            "command": command,
            "python": converter_env["python"],
            "torch": packages.get("torch"),
            "transformers": packages.get("transformers"),
            # The converter imports gguf from the checkout's gguf-py unless NO_LOCAL_GGUF is set.
            "gguf": packages.get("gguf")
            if "NO_LOCAL_GGUF" in os.environ
            else f"gguf-py at llama.cpp {checkout['commit'][:12]}",
            "packages": packages,
        },
        "fetched_at": now_utc(),
    }
    write_lock_entry(root, model.name, entry)
    return entry
