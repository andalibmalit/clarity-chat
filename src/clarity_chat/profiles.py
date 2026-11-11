"""Hand-written config: defaults, profiles, server.toml and models.toml.

Everything is validated when it loads, so a typo stops the client before any call
instead of silently changing what the model gets. The tool never writes these files.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarity_chat.files import read_text


class ConfigError(Exception):
    """A config file is missing, malformed or asks for something the client can't do exactly."""


# One set of setting names for every backend; each backend maps them.
SETTING_NAMES = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "max_tokens",
    "samplers",
    "logprobs",
)
SAMPLER_NAMES = ("penalties", "top_k", "top_p", "min_p", "temperature")

# A setting only acts if its sampler is in `samplers`.
SAMPLER_OF = {
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "repeat_penalty": "penalties",
    "presence_penalty": "penalties",
    "frequency_penalty": "penalties",
}
# The value at which each sampler-bound setting changes nothing.
NEUTRAL = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

SEED_MAX = 2**32 - 2  # llama.cpp reads 0xFFFFFFFF (2**32 - 1) as "pick a random seed"
# "model": its recommended value; "run": a batch run's seed; "server.toml": ctx_size only;
# "chat": set with /set for one conversation; "log": a resumed conversation's logged value
ORIGINS = ("profile", "model", "default", "backend", "regen", "run", "server.toml", "chat", "log")

PROFILE_KEYS_COMMON = {
    "backend",
    "model",
    "mode",
    "settings",
    "use_recommended",
    "device",
    "ctx_size",
}
# apply_chat_template arguments that Clarity sets, or that would change what it renders or
# returns behind the logged template (chat_template replaces the template's text).
RESERVED_TEMPLATE_KWARGS = {
    "conversation",
    "tokenize",
    "add_generation_prompt",
    "continue_final_message",
    "return_dict",
    "return_tensors",
    "chat_template",
}
PROFILE_KEYS_CHAT = {"template_repo", "template_revision", "template_kwargs", "system", "priming"}
PROFILE_KEYS_TEXT = {"header", "user_name", "assistant_name", "priming"}
MODES = ("chat", "text")
BACKENDS = ("llama.cpp", "transformers")
DEVICES = ("mps", "cpu", "cuda")  # the Transformers backend's `device`; default "mps"

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Setting:
    value: Any
    origin: str  # one of ORIGINS

    def to_json(self) -> dict:
        return {"value": self.value, "origin": self.origin}


def read_toml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"{path}: file not found.")
    try:
        return tomllib.loads(read_text(path))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e


def find_root(start: Path | None = None) -> Path:
    """The project folder: the nearest one (from `start` upwards) holding defaults.toml."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "defaults.toml").is_file() and (folder / "server.toml").is_file():
            return folder
    raise ConfigError(f"No defaults.toml and server.toml in {here} or any folder above it.")


# --- settings ---------------------------------------------------------------------------


def _number(value: object) -> float | None:
    """The value as a float if it's an int or float (TOML has no other numbers), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None  # nan passes every range check, and inf has no JSON form
    return float(value)


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# name -> (kind, lowest, highest, description); None means no bound.
SETTING_RANGES: dict[str, tuple[str, float | None, float | None, str]] = {
    "temperature": ("number", 0, None, "a number >= 0"),
    "top_p": ("number", 0, 1, "a number from 0 to 1"),
    "min_p": ("number", 0, 1, "a number from 0 to 1"),
    "repeat_penalty": ("number", 1e-9, None, "a number > 0"),
    "presence_penalty": ("number", None, None, "a number"),
    "frequency_penalty": ("number", None, None, "a number"),
    "top_k": ("integer", 0, None, "an integer >= 0"),
    "repeat_last_n": ("integer", 0, None, "an integer >= 0"),
    "max_tokens": ("integer", 1, None, "an integer >= 1"),
    "seed": ("integer", 0, SEED_MAX, f'"random" or an integer from 0 to {SEED_MAX}'),
    "logprobs": ("integer", 0, 20, "an integer from 0 (off) to 20"),
}


def check_setting(name: str, value: object, where: str) -> Any:
    """Check one setting's type and range. Numbers in float settings come back as floats."""
    if name == "seed" and value == "random":
        return value
    if name == "samplers":
        if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
            raise ConfigError(f"{where}: samplers must be a list of names, got {value!r}.")
        unknown = [s for s in value if s not in SAMPLER_NAMES]
        if unknown:
            raise ConfigError(
                f"{where}: unknown sampler names {unknown}. Known: {list(SAMPLER_NAMES)}."
            )
        if len(set(value)) != len(value):
            raise ConfigError(f"{where}: samplers lists a name twice: {value!r}.")
        return list(value)
    if name not in SETTING_RANGES:
        raise ConfigError(f"{where}: unknown setting {name!r}. Known: {list(SETTING_NAMES)}.")
    kind, lowest, highest, description = SETTING_RANGES[name]
    checked = _integer(value) if kind == "integer" else _number(value)
    if (
        checked is None
        or (lowest is not None and checked < lowest)
        or (highest is not None and checked > highest)
    ):
        raise ConfigError(f"{where}: {name} must be {description}, got {value!r}.")
    return checked


def check_settings_table(table: object, where: str) -> dict[str, Any]:
    if not isinstance(table, dict):
        raise ConfigError(f"{where}: settings must be a table.")
    return {name: check_setting(name, value, where) for name, value in table.items()}


def load_defaults(root: Path) -> dict[str, Any]:
    path = root / "defaults.toml"
    values = check_settings_table(read_toml(path), str(path))
    missing = [name for name in SETTING_NAMES if name not in values]
    if missing:
        raise ConfigError(f"{path}: missing a neutral value for {missing}.")
    return values


def resolve_settings(
    profile_settings: dict[str, Any],
    defaults: dict[str, Any],
    where: str,
    recommended: dict[str, Any] | None = None,
) -> dict[str, Setting]:
    """Profile values, then the model's recommended values (if the profile opts in), then
    defaults. Rejects a non-neutral setting whose sampler is off."""
    layers = [("profile", profile_settings), ("model", recommended or {}), ("default", defaults)]
    resolved = {}
    for name in SETTING_NAMES:
        origin, values = next((o, v) for o, v in layers if name in v)
        resolved[name] = Setting(values[name], origin)
    check_samplers(resolved, where)
    return resolved


def check_backend_settings(resolved: dict[str, Setting], backend: str, where: str) -> None:
    """Refuse a setting the backend can't honour, rather than return a reply without it."""
    if backend == "llama.cpp" and resolved["logprobs"].value > 0:
        raise ConfigError(
            f"{where}: logprobs is {resolved['logprobs'].value}, but token probabilities run "
            "on the Transformers backend for now; llama.cpp's come in v0.4.0."
        )


def check_samplers(resolved: dict[str, Setting], where: str) -> None:
    """Reject a non-neutral setting whose sampler isn't in `samplers`: it would do nothing."""
    samplers = resolved["samplers"].value
    for name, sampler in SAMPLER_OF.items():
        setting = resolved[name]
        if sampler not in samplers and setting.value != NEUTRAL[name]:
            raise ConfigError(
                f"{where}: {name} = {setting.value!r} (from {setting.origin}) has no effect, "
                f"because {sampler!r} is not in samplers {samplers}. Add {sampler!r} to "
                f"samplers or set {name} = {NEUTRAL[name]!r}."
            )


def parse_setting(name: str, text: str, where: str) -> Any:
    """A setting typed as text (`/set`): a number, `random` for seed, or sampler names
    separated by spaces or commas. Checked like a profile's value."""
    if name == "samplers":
        value: Any = text.replace(",", " ").split()
    else:
        try:
            value = tomllib.loads(f"v = {text}")["v"]
        except tomllib.TOMLDecodeError:
            value = text
    return check_setting(name, value, where)


# --- profiles ---------------------------------------------------------------------------


@dataclass
class Profile:
    name: str
    path: Path
    toml_text: str  # the profile file exactly as written
    files: dict[str, str]  # every file the profile pulled in: path (as written) -> contents
    backend: str
    model: str
    mode: str
    settings: dict[str, Setting]
    raw_settings: dict[str, Any] = field(default_factory=dict)  # the [settings] table, checked
    use_recommended: bool = False  # fill unset settings from the model's generation_config.json
    device: str | None = None  # Transformers backend only: mps (default), cpu or cuda
    ctx_size: int | None = None  # None: server.toml's ctx-size
    # chat mode
    system: str | None = None
    priming_messages: list[dict[str, str]] = field(default_factory=list)
    template_repo: str | None = None
    template_revision: str | None = None
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    # text mode
    header: str = ""
    priming_text: str = ""
    user_name: str = ""
    assistant_name: str = ""

    def to_json(self) -> dict:
        """The profile's full contents, as logged with every call."""
        return {
            "name": self.name,
            "path": self.path.as_posix(),
            "toml": self.toml_text,
            "files": self.files,
        }

    @property
    def stop(self) -> list[str]:
        return [f"\n{self.user_name}:"] if self.mode == "text" else []


def profile_path(root: Path, name_or_path: str) -> Path:
    """A path to a .toml file, else NAME in profiles/ (shipped) or local/profiles/ (yours).
    A name in both is an error: a name must say which file it means."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".toml" and candidate.is_file():
        return candidate
    shipped = root / "profiles" / f"{name_or_path}.toml"
    own = root / "local" / "profiles" / f"{name_or_path}.toml"
    if shipped.is_file() and own.is_file():
        raise ConfigError(
            f"Profile {name_or_path!r} is both {shipped} and {own}. Rename one of them."
        )
    return own if own.is_file() else shipped


def _text_or_file(
    value: object,
    key: str,
    base: Path,
    files: dict[str, str],
    where: str,
    logged: dict[str, str] | None,
) -> str:
    """A string used verbatim, or `{ file = "..." }` read byte for byte, relative to the
    profile (or, for a profile from a log, the file's text as logged)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and set(value) == {"file"} and isinstance(value["file"], str):
        if logged is not None:
            if value["file"] not in logged:
                raise ConfigError(f"{where}: the log doesn't hold {key} file {value['file']!r}.")
            text = logged[value["file"]]
        else:
            path = base / value["file"]
            if not path.is_file():
                raise ConfigError(f"{where}: {key} file {value['file']!r} not found ({path}).")
            text = read_text(path)
        files[value["file"]] = text
        return text
    raise ConfigError(f'{where}: {key} must be a string or {{ file = "..." }}, got {value!r}.')


def _messages(value: object, where: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ConfigError(f"{where}: priming must be a list of {{role, content}} tables.")
    messages = []
    for i, item in enumerate(value):
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item["role"] not in ("user", "assistant")
            or not isinstance(item["content"], str)
        ):
            raise ConfigError(
                f"{where}: priming message {i + 1} must be a table with exactly role "
                f'("user" or "assistant") and content (a string), got {item!r}.'
            )
        messages.append({"role": item["role"], "content": item["content"]})
    return messages


def load_profile(
    root: Path, name_or_path: str, defaults: dict[str, Any], logged: dict[str, Any] | None = None
) -> Profile:
    """Load a profile from its file, or, with `logged` (a log's `profile` field: path, toml
    and files), from the text a conversation was logged with."""
    if logged is not None:
        path, toml_text = Path(logged["path"]), logged["toml"]
        where = f"{path} (as logged)"
    else:
        path = profile_path(root, name_or_path)
        where = str(path)
        if not path.is_file():
            raise ConfigError(
                f"Profile {name_or_path!r} not found in profiles/ or local/profiles/ ({root})."
            )
        toml_text = read_text(path)
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{where}: {e}") from e

    mode = data.get("mode")
    if mode not in MODES:
        raise ConfigError(f"{where}: mode must be one of {list(MODES)}, got {mode!r}.")
    allowed = PROFILE_KEYS_COMMON | (PROFILE_KEYS_CHAT if mode == "chat" else PROFILE_KEYS_TEXT)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown keys {unknown} for a {mode}-mode profile. "
            f"Allowed: {sorted(allowed)}."
        )
    for key in ("backend", "model"):
        if not isinstance(data.get(key), str):
            raise ConfigError(f"{where}: {key} is required (a string).")
    if data["backend"] not in BACKENDS:
        raise ConfigError(f"{where}: backend must be one of {list(BACKENDS)}.")
    device = data.get("device")
    if data["backend"] == "transformers":
        device = "mps" if device is None else device
        if device not in DEVICES:
            raise ConfigError(f"{where}: device must be one of {list(DEVICES)}, got {device!r}.")
    elif device is not None:
        raise ConfigError(f'{where}: device applies only to backend = "transformers".')

    raw_settings = check_settings_table(data.get("settings", {}), where)
    ctx_size = data.get("ctx_size")
    if ctx_size is not None and (_integer(ctx_size) is None or ctx_size <= 0):
        raise ConfigError(f"{where}: ctx_size must be a positive integer, got {ctx_size!r}.")
    use_recommended = data.get("use_recommended", False)
    if not isinstance(use_recommended, bool):
        raise ConfigError(f"{where}: use_recommended must be true or false.")
    # With use_recommended, the session resolves again once it knows the model's values.
    settings = resolve_settings(raw_settings, defaults, where)
    check_backend_settings(settings, data["backend"], where)
    files: dict[str, str] = {}
    base = path.parent
    logged_files = logged["files"] if logged is not None else None
    profile = Profile(
        name=path.stem,
        path=path,
        toml_text=toml_text,
        files=files,
        backend=data["backend"],
        model=data["model"],
        mode=mode,
        settings=settings,
        raw_settings=raw_settings,
        use_recommended=use_recommended,
        device=device,
        ctx_size=ctx_size,
    )

    if mode == "chat":
        if "system" in data:
            profile.system = _text_or_file(
                data["system"], "system", base, files, where, logged_files
            )
        if "priming" in data:
            priming = data["priming"]
            if isinstance(priming, dict):
                text = _text_or_file(priming, "priming", base, files, where, logged_files)
                try:
                    priming_data = tomllib.loads(text)
                except tomllib.TOMLDecodeError as e:
                    raise ConfigError(f"{where}: priming file: {e}") from e
                if set(priming_data) != {"messages"}:
                    raise ConfigError(
                        f"{where}: the priming file must hold only [[messages]] tables."
                    )
                priming = priming_data["messages"]
            profile.priming_messages = _messages(priming, where)
        repo, revision = data.get("template_repo"), data.get("template_revision")
        if (repo is None) != (revision is None):
            raise ConfigError(f"{where}: set template_repo and template_revision together.")
        if repo is not None:
            if not isinstance(repo, str) or not isinstance(revision, str):
                raise ConfigError(f"{where}: template_repo and template_revision are strings.")
            check_commit(revision, f"{where}: template_revision")
            profile.template_repo, profile.template_revision = repo, revision
        kwargs = data.get("template_kwargs", {})
        if not isinstance(kwargs, dict):
            raise ConfigError(f"{where}: template_kwargs must be a table.")
        reserved = sorted(set(kwargs) & RESERVED_TEMPLATE_KWARGS)
        if reserved:
            raise ConfigError(
                f"{where}: template_kwargs can't set {reserved}: Clarity sets these itself, and "
                "chat_template would replace the logged template."
            )
        profile.template_kwargs = kwargs
    else:
        for key in ("user_name", "assistant_name"):
            value = data.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ConfigError(
                    f"{where}: {key} is required: a non-empty name without outer spaces."
                )
        profile.user_name, profile.assistant_name = data["user_name"], data["assistant_name"]
        profile.header = _text_or_file(
            data.get("header", ""), "header", base, files, where, logged_files
        )
        profile.priming_text = _text_or_file(
            data.get("priming", ""), "priming", base, files, where, logged_files
        )
    return profile


# --- models.toml ------------------------------------------------------------------------


def check_commit(value: object, where: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.match(value):
        raise ConfigError(f"{where} must be a full 40-character commit hash, got {value!r}.")
    return value


@dataclass(frozen=True)
class ModelEntry:
    name: str
    route: str  # convert | gguf
    repo: str
    revision: str
    tokenizer_repo: str
    tokenizer_revision: str
    file: str | None = None  # gguf route
    sha256: str | None = None  # gguf route

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def load_models(root: Path) -> dict[str, ModelEntry]:
    path = root / "models.toml"
    data = read_toml(path)
    models = {}
    for name, entry in data.items():
        where = f"{path} [{name}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: must be a table.")
        route = entry.get("route")
        keys = {"route", "repo", "revision", "tokenizer_repo", "tokenizer_revision"}
        if route == "gguf":
            keys |= {"file", "sha256"}
        elif route != "convert":
            raise ConfigError(f'{where}: route must be "convert" or "gguf", got {route!r}.')
        if set(entry) != keys:
            raise ConfigError(f"{where}: needs exactly the keys {sorted(keys)}.")
        for key in ("repo", "tokenizer_repo"):
            if not isinstance(entry[key], str):
                raise ConfigError(f"{where}: {key} must be a string.")
        check_commit(entry["revision"], f"{where}: revision")
        check_commit(entry["tokenizer_revision"], f"{where}: tokenizer_revision")
        if route == "gguf" and not (
            isinstance(entry["sha256"], str) and re.match(r"^[0-9a-f]{64}$", entry["sha256"])
        ):
            raise ConfigError(f"{where}: sha256 must be 64 lowercase hex characters.")
        models[name] = ModelEntry(name=name, **entry)
    return models


# --- server.toml ------------------------------------------------------------------------

# Flags that affect output (or caching, which can) and must be written explicitly.
REQUIRED_FLAGS = (
    "ctx-size",
    "n-gpu-layers",
    "flash-attn",
    "cache-type-k",
    "cache-type-v",
    "batch-size",
    "ubatch-size",
    "fit",
    "parallel",
    "no-context-shift",
    "swa-full",
    "ctx-checkpoints",
    "cache-ram",
    "no-cache-idle-slots",
    "cache-reuse",
    "threads",
    "threads-batch",
    "log-verbosity",
)
# Required flags that are on/off switches (true passes the bare flag, false leaves it out);
# every other required flag needs a value.
SWITCH_FLAGS = {"no-context-shift", "swa-full", "no-cache-idle-slots"}
# llama.cpp logs device and model-buffer lines at its trace level (4). The GPU placement check
# reads them from the startup output, so a lower verbosity would hide them.
MIN_LOG_VERBOSITY = 4
REQUIRED_VALUES = {
    "fit": "off",
    "parallel": 1,
    "no-context-shift": True,
    "cache-ram": 0,
    "cache-reuse": 0,
    "no-cache-idle-slots": True,
}
# Every flag server.toml may set: the required ones, plus the few the shipped file uses. Any
# other flag is refused, including aliases, so a new one is added deliberately, with a reason
# and a test. Clarity itself passes --model, --host, --port, --no-slots and --no-webui.
ALLOWED_FLAGS = {*REQUIRED_FLAGS, "kv-unified", "no-kv-unified", "no-jinja", "chat-template"}


@dataclass(frozen=True)
class ServerConfig:
    path: Path
    llama_cpp_dir: Path
    llama_cpp_commit: str
    startup_timeout_s: float
    flags: dict[str, Any]

    @property
    def ctx_size(self) -> int:
        return self.flags["ctx-size"]

    def flag_argv(self) -> list[str]:
        argv: list[str] = []
        for name, value in self.flags.items():
            if value is True:
                argv.append(f"--{name}")
            elif value is not False:
                argv += [f"--{name}", str(value)]
        return argv

    def to_json(self) -> dict:
        return {
            "path": self.path.as_posix(),
            "llama_cpp_dir": self.llama_cpp_dir.as_posix(),
            "llama_cpp_commit": self.llama_cpp_commit,
            "flags": self.flags,
        }


def load_server_config(root: Path) -> ServerConfig:
    path = root / "server.toml"
    data = read_toml(path)
    where = str(path)
    allowed = {"llama_cpp_dir", "llama_cpp_commit", "startup_timeout_s", "flags"}
    unknown = sorted(set(data) - allowed)
    if unknown or not allowed <= set(data):
        raise ConfigError(f"{where}: needs exactly the keys {sorted(allowed)}; unknown: {unknown}.")
    flags = data["flags"]
    if not isinstance(flags, dict):
        raise ConfigError(f"{where}: [flags] must be a table.")
    for name, value in flags.items():
        if name not in ALLOWED_FLAGS:
            raise ConfigError(
                f"{where}: flag {name!r} isn't in Clarity's allowed list (docs/reference.md); "
                "ask for it to be added, with a reason."
            )
        if not isinstance(value, (bool, int, str)):
            raise ConfigError(f"{where}: flag {name!r} must be true/false, a number or a string.")
        if value == "auto" or (not isinstance(value, bool) and value in ("-1", -1)):
            raise ConfigError(f"{where}: flag {name!r} must be explicit, not {value!r}.")
    missing = [name for name in REQUIRED_FLAGS if name not in flags]
    if missing:
        raise ConfigError(f"{where}: [flags] must set {missing} explicitly.")
    valued = [name for name in REQUIRED_FLAGS if name not in SWITCH_FLAGS]
    switches_with_values = [name for name in valued if isinstance(flags[name], bool)]
    if switches_with_values:
        raise ConfigError(
            f"{where}: {switches_with_values} need a value, not true or false "
            "(false would leave the flag out, so llama-server's default would apply)."
        )
    for name, required in REQUIRED_VALUES.items():
        # type() too: in Python False == 0, and false would leave the flag out.
        if type(flags[name]) is not type(required) or flags[name] != required:
            raise ConfigError(f"{where}: {name} must be {required!r}, got {flags[name]!r}.")
    if not ("kv-unified" in flags) ^ ("no-kv-unified" in flags):
        raise ConfigError(f"{where}: set exactly one of kv-unified or no-kv-unified.")
    ctx_size = _integer(flags["ctx-size"])
    if ctx_size is None or ctx_size <= 0:
        raise ConfigError(f"{where}: ctx-size must be a positive integer (0 means 'from model').")
    verbosity = _integer(flags["log-verbosity"])
    if verbosity is None or verbosity < MIN_LOG_VERBOSITY:
        raise ConfigError(
            f"{where}: log-verbosity must be an integer of at least {MIN_LOG_VERBOSITY}; "
            "the GPU placement check reads device lines llama.cpp logs at that level."
        )
    if flags["flash-attn"] not in ("on", "off"):
        raise ConfigError(f'{where}: flash-attn must be "on" or "off".')
    timeout = data["startup_timeout_s"]
    if _number(timeout) is None or timeout <= 0:
        raise ConfigError(f"{where}: startup_timeout_s must be a positive number.")
    return ServerConfig(
        path=path,
        llama_cpp_dir=(root / data["llama_cpp_dir"]).resolve(),
        llama_cpp_commit=check_commit(data["llama_cpp_commit"], f"{where}: llama_cpp_commit"),
        startup_timeout_s=float(timeout),
        flags=dict(flags),
    )
