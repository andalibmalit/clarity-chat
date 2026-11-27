"""Where the weights come from: `fetch`, `verify` and the generated lock.

models.toml (hand-written) says what to get; the lock (generated, JSON) records what was
resolved: commits, file paths, sizes, sha256, the llama.cpp commit and the converter's
package versions. Each entry is read from local/models.lock, which every fetch writes, else
from the shipped models.lock. The sha256 is computed once, at fetch; a server start only
compares size and modified time, and `verify` re-hashes on demand.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from clarity_chat.files import read_text, sha256_file, write_text_atomic
from clarity_chat.logs import git_state, now_utc
from clarity_chat.profiles import (
    LOCAL_LOCK,
    LOCAL_MODELS,
    SAFETENSORS_ROUTE,
    ConfigError,
    ModelEntry,
    ServerConfig,
    check_setting,
    fetch_hint,
    load_defaults,
    load_models,
    load_profile,
    model_files,
    read_toml,
    run_hint,
)
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
# What a conversion needs from the source repo: config, tokenizer and weights. No pickle
# (`.bin`, `.pt`): an entry's `weights` key adds the one file it names (v0.4.1 decision 2).
# The safetensors route hashes what these match, so a narrower list would leave a file each
# shipped entry lists unhashed, and every such fetch would stop (v0.4.7 R, state 18).
SOURCE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]
# The two files the Transformers load looks for, now that it takes safetensors only.
LOADED_WEIGHTS = {"model.safetensors", "model.safetensors.index.json"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONVERTER_PYTHON = "3.13"
# Shipped with the code that uses it, as convert_shim.py is: a root holds only what fetch
# writes, so any root converts (A9's fix: -m live's scratch root had no copy).
CONVERTER_REQUIREMENTS = Path(__file__).with_name("converter-requirements.txt")
CONVERTER_ENV = "models/.converter"
CONVERTER_MARKER = "clarity-requirements.sha256"  # the sha256 of what it was made by


def portable_path(text: str, root: Path, resolve: bool = True) -> str:
    """A path as written to a lock entry, which may be committed to models.lock
    (scripts/refresh_lock.py): relative to the repo when inside it, `~/...` when under the
    home folder, so the lock carries no user name. Other text is returned unchanged."""
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


def read_lock_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": LOCK_SCHEMA_VERSION, "models": {}}
    return json.loads(read_text(path))


def read_lock(root: Path) -> dict[str, Any]:
    """Each model's lock entry: the one in local/models.lock, which every fetch writes
    (v0.4.2 A10), else the shipped models.lock's, which only the maintainer changes
    (scripts/refresh_lock.py). Your own entries (local/models.toml, v0.4.0 B4) are read
    from local/models.lock alone; an entry there for a model in neither models.toml (one
    since removed) isn't read."""
    lock = read_lock_file(root / LOCK_FILE)
    yours = set(read_toml(root / LOCAL_MODELS)) if (root / LOCAL_MODELS).is_file() else set()
    shipped = set(read_toml(root / "models.toml")) if (root / "models.toml").is_file() else set()
    theirs = {n: e for n, e in lock["models"].items() if n not in yours}
    local = read_lock_file(root / LOCAL_LOCK)["models"]
    lock["models"] = theirs | {n: e for n, e in local.items() if n in yours | shipped}
    return lock


def write_lock_file(path: Path, lock: dict[str, Any]) -> None:
    lock["models"] = dict(sorted(lock["models"].items()))
    write_text_atomic(path, json.dumps(lock, indent=2, ensure_ascii=False) + "\n")


def write_lock_entry(root: Path, name: str, entry: dict[str, Any]) -> None:
    """Record `entry` in local/models.lock, for a shipped model as for yours (v0.4.2 A10),
    so no fetch changes a tracked file."""
    path = root / LOCAL_LOCK
    lock = read_lock_file(path)
    lock["models"][name] = entry
    write_lock_file(path, lock)


def lock_entry(root: Path, model: ModelEntry) -> dict[str, Any]:
    """The lock entry for `model`; an error if it's missing or was fetched from other settings."""
    entry = read_lock(root)["models"].get(model.name)
    if entry is None:
        raise ConfigError(f"Model {model.name!r} is not fetched. {fetch_hint(model.name)}")
    keys = ["route", "repo", "revision", "tokenizer_repo", "tokenizer_revision", "weights_sha256"]
    if model.route == "gguf":
        keys.append("file")
    toml, lock = model_files(root, model.name)
    for key in keys:
        if entry.get(key) != getattr(model, key):
            raise ConfigError(
                f"{lock} has {model.name!r} with {key} = {entry.get(key)!r}, but {toml} "
                f"says {getattr(model, key)!r}. {fetch_hint(model.name)}"
            )
    recommended = entry.get("recommended")
    if recommended is not None and not (
        isinstance(recommended, dict) and isinstance(recommended.get("values"), dict)
    ):  # a hand-edited lock: refused here, not met later by a read that blames the log
        raise ConfigError(
            f"{lock}'s recommended settings for {model.name!r} aren't a table of values. "
            f"{fetch_hint(model.name)}"
        )
    if model.route == "gguf" and entry.get("gguf", {}).get("sha256") != model.sha256:
        raise ConfigError(
            f"{lock} has {model.name!r} with a different sha256 than {toml}. "
            f"{fetch_hint(model.name)}"
        )
    return entry


PROFILES_NOT_CHECKED = (
    "Each profile's file and its model entry are checked here. Refusals that need the model's "
    "tokenizer or the server, such as a steering vector that doesn't fit the model, come when "
    "a conversation starts."
)


def list_profiles(root: Path) -> list[dict[str, Any]]:
    """Every profile in profiles/ (by name), then local/profiles/ (yours), each with its model
    and whether that is fetched. A profile that doesn't load is listed with its `error`."""
    defaults = load_defaults(root)
    models = load_models(root)
    out = []
    for folder, local in ((root / "profiles", False), (root / "local" / "profiles", True)):
        for path in sorted(folder.glob("*.toml"), key=lambda path: path.stem):
            entry: dict[str, Any] = {
                "name": path.stem,
                "model": None,
                "model_name": None,
                "backend": None,
                "description": None,
                "steered": False,
                "local": local,
                "fetched": False,
                "error": None,
            }
            try:
                profile = load_profile(root, path.stem, defaults)
                model = models.get(profile.model)
                if model is None:
                    raise ConfigError(
                        f"{path}: model {profile.model!r} isn't in models.toml or "
                        "local/models.toml."
                    )
                if model.route == SAFETENSORS_ROUTE and profile.backend != "transformers":
                    raise ConfigError(no_llama_cpp(path.stem, model.name))
            except (ConfigError, OSError, UnicodeDecodeError) as e:  # as list_chats does
                out.append({**entry, "error": str(e)})
                continue
            try:  # in models.lock, from this entry, and its weights still on disk
                locked = lock_entry(root, model)
                if profile.backend == "transformers":
                    source_weights(root, locked)
                    fetched = True
                else:  # never hashed here (v0.4.4 decision 15)
                    fetched = fetched_size(root, locked)
            except ConfigError:
                fetched = False
            out.append(
                {
                    **entry,
                    "model": model.name,
                    "model_name": model.display_name,
                    "backend": profile.backend,
                    "description": profile.description,
                    "steered": profile.steering is not None,
                    "fetched": fetched,
                }
            )
    return out


def official_only(name: str) -> str:
    """State 20: a safetensors-route model has no GGUF."""
    return (
        f'{name} is fetched as official weights only (route = "{SAFETENSORS_ROUTE}"), and '
        "llama.cpp needs a GGUF."
    )


def no_llama_cpp(profile: str, name: str) -> str:
    """State 19: a llama.cpp profile on a safetensors-route model."""
    return (
        f'Profile {profile}: {official_only(name)} Set backend = "transformers" in the '
        "profile, or use a model that has a GGUF."
    )


def _in_root(root: Path, text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else root / path


def weights_path(root: Path, entry: dict[str, Any]) -> Path:
    """The GGUF: under the repo (converted), or in the Hugging Face cache (downloaded),
    which models.lock writes as `~/...` or as an absolute path outside the home folder."""
    if "gguf" not in entry:
        raise ConfigError(official_only(entry["name"]))
    return _in_root(root, entry["gguf"]["path"])


def source_files(entry: dict[str, Any], lock: str) -> list[dict[str, Any]]:
    """A safetensors-route entry's `source_files`, each {file, size, sha256}: the one reader,
    so an empty or hand-broken list can't pass a check as "0 files" (state 26)."""
    files = entry.get("source_files")
    usable = (
        "source_dir" in entry
        and isinstance(files, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("file"), str)
            and not PurePosixPath(item["file"]).is_absolute()  # a path in the snapshot
            and ".." not in PurePosixPath(item["file"]).parts
            and isinstance(item.get("size"), int)
            and isinstance(item.get("sha256"), str)
            and SHA256_RE.match(item["sha256"])
            for item in files
        )
        and LOADED_WEIGHTS & {item["file"] for item in files}
    )
    if not usable:
        raise ConfigError(
            f"{lock}'s entry for {entry['name']} has no usable list of its files "
            f"(source_files). {fetch_hint(entry['name'])}"
        )
    assert isinstance(files, list)
    return files


def source_weights(root: Path, entry: dict[str, Any]) -> Path:
    """The official weights, for the Transformers backend: the folder a converted GGUF came
    from, or the safetensors route's download, where each listed file must be there at its
    recorded size. Nothing is hashed here; `clarity verify` re-hashes (v0.4.7 decision 9)."""
    name = entry["name"]
    if entry.get("route") == SAFETENSORS_ROUTE:
        lock = model_files(root, name)[1]
        files = source_files(entry, lock)
        path = _in_root(root, entry["source_dir"])
        for item in files:
            file = path / item["file"]
            if not file.is_file():
                raise ConfigError(f"{file} is missing. {fetch_hint(name)}")
            if (size := file.stat().st_size) != item["size"]:
                raise ConfigError(
                    f"{file} changed since it was fetched: it has {size:,} bytes, and {lock} "
                    f"says {item['size']:,}. {fetch_hint(name)}"
                )
        return path
    if entry.get("route") != "convert" or "source_dir" not in entry:
        raise ConfigError(
            f"{name} was fetched as a ready-made GGUF; the Transformers backend needs the "
            f'official weights (route = "convert" or "{SAFETENSORS_ROUTE}").'
        )
    path = _in_root(root, entry["source_dir"])
    if not (path / "config.json").is_file():
        raise ConfigError(f"{path} has no config.json. {fetch_hint(entry['name'])}")
    return path


def check_weights_quick(root: Path, entry: dict[str, Any]) -> Path:
    """Size and modified time against models.lock (a full re-hash is `clarity verify`)."""
    path = weights_path(root, entry)
    if not path.is_file():
        raise ConfigError(f"{path} is missing. {fetch_hint(entry['name'])}")
    stat = path.stat()
    if stat.st_size != entry["gguf"]["size"] or stat.st_mtime_ns != entry["gguf"]["mtime_ns"]:
        raise ConfigError(
            f"{path} changed since it was fetched (size or modified time differs from "
            f"{model_files(root, entry['name'])[1]}). {run_hint('verify ' + entry['name'])}"
        )
    return path


def fetched_size(root: Path, entry: dict[str, Any]) -> bool:
    """The listing's and setup's check: the GGUF is there at its fetched size. A changed
    time alone still counts as fetched; a start checks it again (v0.4.4 decision 15)."""
    path = weights_path(root, entry)
    return path.is_file() and path.stat().st_size == entry["gguf"]["size"]


def recheck_weights(root: Path, entry: dict[str, Any], whose: str) -> tuple[Path, str | None]:
    """check_weights_quick for a start (v0.4.4 decision 15): a file of the fetched size whose
    modified time changed is hashed, as `clarity verify` does, and if its bytes match, the
    new time is recorded. Returns the GGUF and the line that says so, or None."""
    path, name = weights_path(root, entry), entry["name"]
    if not path.is_file():
        raise ConfigError(f"{path} is missing. {fetch_hint(name)}")
    lock = model_files(root, name)[1]
    size = path.stat().st_size
    if size != entry["gguf"]["size"]:
        raise ConfigError(
            f"{path} changed since it was fetched: it has {size:,} bytes, and {lock} says "
            f"{entry['gguf']['size']:,}. {fetch_hint(name)}"
        )
    if path.stat().st_mtime_ns == entry["gguf"]["mtime_ns"]:
        return path, None
    if not verify(root, name)[0]:
        raise ConfigError(
            f"{path} changed since it was fetched: its sha256 isn't the one in {lock}. "
            f"{fetch_hint(name)}"
        )
    return path, (
        f"{whose} modified time changed since it was fetched; its bytes still match, so the "
        "new time is recorded in local/models.lock."
    )


def verify(root: Path, name: str) -> tuple[bool, str]:
    entry = read_lock(root)["models"].get(name)
    lock = model_files(root, name)[1]
    if entry is None:
        return False, f"{name}: not in {lock}"
    if entry.get("route") == SAFETENSORS_ROUTE:
        return verify_files(root, entry, lock)
    path = weights_path(root, entry)
    if not path.is_file():
        return False, f"{name}: {path} is missing"
    digest = sha256_file(path)
    if digest != entry["gguf"]["sha256"]:
        return False, f"{name}: sha256 {digest} != {lock} {entry['gguf']['sha256']}"
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (entry["gguf"]["size"], entry["gguf"]["mtime_ns"]):
        # The bytes are the same (say, after a copy or `touch`): record the new time, so
        # the quick check at server start passes again.
        entry["gguf"].update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        write_lock_entry(root, name, entry)
        return True, f"{name}: sha256 matches ({digest[:16]}…); size and time updated"
    return True, f"{name}: sha256 matches ({digest[:16]}…)"


def verify_files(root: Path, entry: dict[str, Any], lock: str) -> tuple[bool, str]:
    """`verify` of a safetensors-route entry: each listed file re-hashed (state 21)."""
    name = entry["name"]
    try:
        files = source_files(entry, lock)
    except ConfigError as e:
        return False, f"{name}: {e}"
    folder = _in_root(root, entry["source_dir"])
    for item in files:
        path = folder / item["file"]
        if not path.is_file():
            return False, f"{name}: {path} is missing"
        if (digest := sha256_file(path)) != item["sha256"]:
            return False, f"{name}: {item['file']} sha256 {digest} != {lock} {item['sha256']}"
    return True, f"{name}: sha256 matches for {len(files)} files in {folder}"


def gated_refusal(repo: str) -> str:
    """What to do when Hugging Face refuses a gated repo. Clarity handles no token: the
    Hugging Face library reads the one `hf auth login` saved, or HF_TOKEN."""
    return (
        f"{repo} is gated: Hugging Face gives its files only to accounts with access. "
        f"Accept its licence at https://huggingface.co/{repo} and wait until access is "
        "granted, then run `uv run hf auth login` (again, if your login has expired) and fetch "
        "again."
    )


def _download(repo: str, revision: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
    except GatedRepoError as e:
        raise ConfigError(gated_refusal(repo)) from e


def fetch(root: Path, model: ModelEntry, server: ServerConfig) -> dict[str, Any]:
    if model.route == "convert":
        return fetch_convert(root, model, server)
    if model.route == SAFETENSORS_ROUTE:
        return fetch_safetensors(root, model)
    return fetch_gguf(root, model)


def _download_file(repo: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(hf_hub_download(repo, filename, revision=revision))
    except GatedRepoError as e:
        raise ConfigError(gated_refusal(repo)) from e


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
        raise ConfigError(  # a fetch writes local/models.lock alone (A10, C20)
            f"{model.repo}/{model.file} at {model.revision[:12]} has sha256 {digest}, but "
            f"{model_files(root, model.name)[0]} says {model.sha256}. "
            "Not recorded in local/models.lock."
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


def _download_tokenizer(model: ModelEntry, source: Path) -> Path:
    """The tokenizer's folder: `source`, unless it's from another repo or commit."""
    if (model.tokenizer_repo, model.tokenizer_revision) == (model.repo, model.revision):
        return source
    print(f"Downloading tokenizer {model.tokenizer_repo}@{model.tokenizer_revision[:12]} …")
    return _download(model.tokenizer_repo, model.tokenizer_revision, TOKENIZER_FILES)


def fetch_safetensors(root: Path, model: ModelEntry) -> dict[str, Any]:
    """Download `repo` at `revision` as it is, for the Transformers backend: no GGUF, no
    pickle. Each file it downloads from `repo` is hashed and recorded, and checked against
    the shipped models.lock's entry for the same route, repo and commit (v0.4.7 R)."""
    print(f"Downloading {model.repo}@{model.revision[:12]} …", flush=True)
    source = _download(model.repo, model.revision, SOURCE_PATTERNS)
    from huggingface_hub.utils import filter_repo_objects

    found = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    names = list(filter_repo_objects(found, allow_patterns=SOURCE_PATTERNS))
    if not LOADED_WEIGHTS & set(names):
        raise ConfigError(
            f"{model.repo} at {model.revision[:12]} has no model.safetensors or "
            "model.safetensors.index.json, which the Transformers backend loads; the "
            f"{SAFETENSORS_ROUTE} route fetches no pickle (.bin or .pt). Not recorded in "
            "local/models.lock."
        )
    print(f"Hashing {len(names)} files …", flush=True)
    files = [
        {"file": n, "size": (source / n).stat().st_size, "sha256": sha256_file(source / n)}
        for n in names
    ]
    shipped = read_lock_file(root / LOCK_FILE)["models"].get(model.name)
    pins = (model.route, model.repo, model.revision)
    if shipped and (shipped.get("route"), shipped.get("repo"), shipped.get("revision")) == pins:
        ours = {item["file"]: item["sha256"] for item in files}
        for item in source_files(shipped, LOCK_FILE):
            if (digest := ours.get(item["file"])) != item["sha256"]:
                found_as = "is missing" if digest is None else f"has sha256 {digest}"
                raise ConfigError(  # a fetch writes local/models.lock alone (A10, C20)
                    f"{model.repo}/{item['file']} at {model.revision[:12]} {found_as}, but "
                    f"{LOCK_FILE} says {item['sha256']}. Not recorded in local/models.lock."
                )
    tokenizer = _download_tokenizer(model, source)
    entry = {
        "name": model.name,
        "route": model.route,
        "repo": model.repo,
        "revision": model.revision,
        "tokenizer_repo": model.tokenizer_repo,
        "tokenizer_revision": model.tokenizer_revision,
        "source_dir": portable_path(str(source), root),
        "tokenizer_dir": portable_path(str(tokenizer), root),
        "source_files": files,
        "recommended": recommended_settings(source, model.repo, model.revision),
        "fetched_at": now_utc(),
    }
    write_lock_entry(root, model.name, entry)
    return entry


def converter_commands(root: Path) -> list[list[str]]:
    """A conversion's three commands: make the converter's environment, install its
    requirements, and run its Python. `uv run --with-requirements` doesn't check a file's
    hashes, so Clarity makes the environment itself (v0.4.2 decision 15): a fresh venv,
    with nothing from Clarity's project or uv's config files (--no-project, --no-config), on
    a Python uv installed itself, which it downloads if it has none (--managed-python, v0.4.6
    decision 4: conda's, Homebrew's or the system's could be upgraded or removed under it),
    then converter-requirements.txt installed with every file's sha256 checked, wheels only
    (--no-build). uv's environment variables still apply, except UV_PYTHON_PREFERENCE,
    which can't be used with --managed-python (uv 0.11.19 refuses the two together): an
    index they name (UV_DEFAULT_INDEX, say) is used, and the hashes cover what it serves. The
    first-index strategy is set so UV_INDEX_STRATEGY can't merge indexes. -I keeps the
    working folder and PYTHON* variables out of the converter. The uv is the one running
    Clarity (uv sets UV for what it runs), as for setup: install.sh's PATH doesn't hold
    ~/.local/bin."""
    env = root / CONVERTER_ENV
    uv = os.environ.get("UV") or "uv"
    python = env / "Scripts" / "python.exe" if os.name == "nt" else env / "bin" / "python"
    return [
        [
            uv,
            "venv",
            "--clear",
            "--no-project",
            "--no-config",
            "--managed-python",
            "--python",
            CONVERTER_PYTHON,
            str(env),
        ],
        [
            uv,
            "pip",
            "install",
            "--no-config",
            "--require-hashes",
            "--no-build",  # an sdist's build backend would come unhashed from the index
            "--index-strategy",
            "first-index",
            "--python",
            str(env),
            "-r",
            str(CONVERTER_REQUIREMENTS),
        ],
        [str(python), "-I"],
    ]


def _logged(command: list[str], log: TextIO, env: dict[str, str] | None = None) -> int:
    """Run `command` with no terminal input, its output written to `log`; its exit code."""
    try:
        proc = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:  # uv not on PATH, or the environment's python unrunnable
        raise ConfigError(
            f"Couldn't run {command[0]}: {error.strerror or error}. The converter's "
            f"environment, {CONVERTER_ENV}, is safe to delete: the next fetch makes it again."
        ) from error
    assert proc.stdout is not None
    for line in proc.stdout:
        log.write(line)
    return proc.wait()


def _make_converter_environment(
    root: Path, commands: list[list[str]], python: Path, log: TextIO, log_path: Path
) -> str:
    """Make the converter's environment unless it was made by these commands from this
    converter-requirements.txt: a marker in it holds the sha256 of both, written only after
    the install succeeds, so a failed or interrupted one, or one made with other flags, is
    made again next time. So is one whose python is gone: it's a link to the one uv chose.
    Returns the requirements file's sha256 as checked here, for the lock: a file changed
    during the conversion isn't the one the environment holds."""
    marker = root / CONVERTER_ENV / CONVERTER_MARKER
    made_by = {"commands": commands, "requirements": sha256_file(CONVERTER_REQUIREMENTS)}
    digest = hashlib.sha256(json.dumps(made_by).encode("utf-8")).hexdigest()
    if marker.is_file() and read_text(marker).strip() == digest and python.is_file():
        return made_by["requirements"]
    print(
        f"Making the converter's environment in {CONVERTER_ENV}: Python {CONVERTER_PYTHON} and "
        f"the packages {portable_path(str(CONVERTER_REQUIREMENTS), root)} pins, each checked "
        "by its sha256 …",
        flush=True,
    )
    for command in commands:
        env = None
        if command[1] == "venv":
            env = {k: v for k, v in os.environ.items() if k != "UV_PYTHON_PREFERENCE"}
        if _logged(command, log, env=env) != 0:
            raise ConfigError(
                f"Making the converter's environment failed; see {log_path}. If it says "
                '"Hash mismatch", a package file differs from the one '
                f"{CONVERTER_REQUIREMENTS.name} pins, and it wasn't installed."
            )
    marker.write_text(digest + "\n", encoding="utf-8")
    return made_by["requirements"]


def fetch_convert(root: Path, model: ModelEntry, server: ServerConfig) -> dict[str, Any]:
    checkout = git_state(server.llama_cpp_dir)
    if checkout.get("commit") is None:
        raise ConfigError(
            f"{server.llama_cpp_dir} has no commit git can read ({checkout.get('error')}), so "
            f"it can't be checked against the {server.llama_cpp_commit} server.toml pins."
        )
    if checkout.get("commit") != server.llama_cpp_commit:
        raise ConfigError(
            f"{server.llama_cpp_dir} is at {checkout.get('commit')}, but server.toml pins "
            f"{server.llama_cpp_commit}."
        )
    if not CONVERTER_REQUIREMENTS.is_file():  # before the download, which can take minutes
        raise ConfigError(
            f"{CONVERTER_REQUIREMENTS} is missing, so the converter's environment can't be "
            "made. It's part of Clarity: restore it with git restore in Clarity's folder."
        )
    print(f"Downloading {model.repo}@{model.revision[:12]} …", flush=True)
    weights = [] if model.weights is None else [model.weights]
    source = _download(model.repo, model.revision, [*SOURCE_PATTERNS, *weights])
    if model.weights is not None:  # the reviewed bytes, before the converter loads them (S1)
        print(f"Hashing {model.weights} …", flush=True)
        digest = sha256_file(source / model.weights)
        if digest != model.weights_sha256:
            raise ConfigError(  # a fetch writes local/models.lock alone (A10, C20)
                f"{model.repo}/{model.weights} at {model.revision[:12]} has sha256 {digest}, "
                f"but {model_files(root, model.name)[0]} says {model.weights_sha256}. "
                "Not converted or recorded in local/models.lock."
            )
    tokenizer = _download_tokenizer(model, source)

    out_dir = root / "models" / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model.name}-bf16.gguf"
    partial = out_dir / f"{out.name}.partial"
    versions = out_dir / "converter-versions.json"
    converter = server.llama_cpp_dir / "convert_hf_to_gguf.py"
    shim = Path(__file__).with_name("convert_shim.py")
    *setup, python = converter_commands(root)
    command = [
        *python,
        str(shim),
        str(versions),
        str(converter),
        str(source),
        "--outtype",
        "bf16",
        "--outfile",
        str(partial),
    ]
    # As run: not resolved, since the environment's python is a link to the one uv chose.
    shown = [[portable_path(part, root, resolve=False) for part in c] for c in (*setup, command)]
    with (out_dir / "convert.log").open("w", encoding="utf-8", newline="") as log:
        requirements = _make_converter_environment(
            root, setup, Path(python[0]), log, out_dir / "convert.log"
        )
        print("Converting:", " ".join(shown[-1]), flush=True)
        # Offline: it reads only the snapshot, so a tokenizer's auto_map can't fetch code
        # from another repo. No stdin: a "run the custom code?" prompt is refused, not left
        # waiting in convert.log (v0.4.1's security review, S2).
        if _logged(command, log, env={**os.environ, "HF_HUB_OFFLINE": "1"}) != 0:
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
        **({"weights_sha256": model.weights_sha256} if model.weights else {}),
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
            "requirements_sha256": requirements,
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
    # local/models.lock wins over the shipped entry, so a conversion that came out
    # different would be used without a word (v0.4.2's security review, S6): a flag.
    shipped = read_lock_file(root / LOCK_FILE)["models"].get(model.name, {})
    same_source = shipped.get("revision") == model.revision
    if same_source and shipped.get("llama_cpp", {}).get("commit") == checkout["commit"]:
        theirs = shipped.get("gguf", {}).get("sha256")
        if theirs and theirs != entry["gguf"]["sha256"]:
            print(
                f"Note: {model.name}'s GGUF (sha256 {entry['gguf']['sha256'][:12]}) differs "
                f"from the shipped one in models.lock ({theirs[:12]}), though both were "
                "converted from the same revision at the same llama.cpp commit; compare the "
                "two entries' converter records to look for what else differed."
            )
    return entry
