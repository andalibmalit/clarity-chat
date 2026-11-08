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


def portable_path(text: str, root: Path, resolve: bool = True) -> str:
    """A path as written to models.lock, which is committed: relative to the repo when inside
    it, `~/...` when under the home folder, so the lock carries no user name. Other text is
    returned unchanged."""
    path = Path(text)
    if not path.is_absolute():
        return text
    path = path.resolve() if resolve else path
    for base in (root, Path.home()):
        for candidate in {base, base.resolve()}:
            if path.is_relative_to(candidate):
                prefix = "" if base == root else "~/"
                return prefix + path.relative_to(candidate).as_posix()
    return path.as_posix()


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
    keys = ["route", "repo", "revision", "tokenizer_repo", "tokenizer_revision"]
    if model.route == "gguf":
        keys.append("file")
    for key in keys:
        if entry.get(key) != getattr(model, key):
            raise ConfigError(
                f"models.lock has {model.name!r} with {key} = {entry.get(key)!r}, but models.toml "
                f"says {getattr(model, key)!r}. Run `clarity fetch {model.name}` again."
            )
    if model.route == "gguf" and entry.get("gguf", {}).get("sha256") != model.sha256:
        raise ConfigError(
            f"models.lock has {model.name!r} with a different sha256 than models.toml. "
            f"Run `clarity fetch {model.name}` again."
        )
    return entry


def weights_path(root: Path, entry: dict[str, Any]) -> Path:
    """The GGUF: under the repo (converted), or in the Hugging Face cache (downloaded),
    which models.lock writes as `~/...` or as an absolute path outside the home folder."""
    path = Path(entry["gguf"]["path"]).expanduser()
    return path if path.is_absolute() else root / path


def source_weights(root: Path, entry: dict[str, Any]) -> Path:
    """The official weights a converted GGUF came from, for the Transformers backend."""
    if entry.get("route") != "convert" or "source_dir" not in entry:
        raise ConfigError(
            f"{entry['name']} was fetched as a ready-made GGUF; the Transformers backend needs "
            'the official weights (route = "convert").'
        )
    path = Path(entry["source_dir"]).expanduser()
    path = path if path.is_absolute() else root / path
    if not (path / "config.json").is_file():
        raise ConfigError(f"{path} has no config.json. Run `clarity fetch {entry['name']}`.")
    return path


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
    return fetch_gguf(root, model)


def _download_file(repo: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision))


def fetch_gguf(root: Path, model: ModelEntry) -> dict[str, Any]:
    """Download a ready-made GGUF at its pinned commit and check its sha256 against
    models.toml. It stays in the Hugging Face cache. The tokenizer, chat template and
    recommended settings come from `tokenizer_repo` at its commit (the source model); the
    echo check on the first call shows whether that tokenizer matches the GGUF."""
    assert model.file is not None and model.sha256 is not None
    print(f"Downloading {model.repo}@{model.revision[:12]} {model.file} …", flush=True)
    path = _download_file(model.repo, model.revision, model.file)
    print(f"Hashing {path.name} …", flush=True)
    digest = sha256_file(path)
    if digest != model.sha256:
        raise ConfigError(
            f"{model.repo}/{model.file} at {model.revision[:12]} has sha256 {digest}, but "
            f"models.toml says {model.sha256}. Not recorded in models.lock."
        )
    print(f"Downloading tokenizer {model.tokenizer_repo}@{model.tokenizer_revision[:12]} …")
    tokenizer = _download(
        model.tokenizer_repo,
        model.tokenizer_revision,
        [*TOKENIZER_FILES, "generation_config.json"],
    )
    stat = path.stat()
    entry = {
        "name": model.name,
        "route": model.route,
        "repo": model.repo,
        "revision": model.revision,
        "file": model.file,
        "tokenizer_repo": model.tokenizer_repo,
        "tokenizer_revision": model.tokenizer_revision,
        "tokenizer_dir": portable_path(str(tokenizer), root),
        "gguf": {
            # The snapshot path, not the blob it links to: llama.cpp sees the file's name.
            "path": portable_path(str(path), root, resolve=False),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        },
        "recommended": recommended_settings(
            tokenizer, model.tokenizer_repo, model.tokenizer_revision
        ),
        "fetched_at": now_utc(),
    }
    write_lock_entry(root, model.name, entry)
    return entry


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
    shown = [portable_path(part, root) for part in command]
    print("Converting:", " ".join(shown), flush=True)
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
        "source_dir": portable_path(str(source), root),
        "tokenizer_dir": portable_path(str(tokenizer), root),
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
            "command": shown,
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
