from __future__ import annotations

import os

import pytest

from clarity_chat.profiles import (
    ConfigError,
    load_defaults,
    load_models,
    load_profile,
    load_server_config,
    resolve_settings,
)


def write_profile(root, name, text):
    (root / "profiles" / f"{name}.toml").write_text(text, encoding="utf-8")


BASE = (
    'backend = "llama.cpp"\nmodel = "tiny"\nmode = "text"\nuser_name = "U"\nassistant_name = "A"\n'
)


def test_defaults_cover_every_setting_with_neutral_values(root):
    defaults = load_defaults(root)
    assert defaults["temperature"] == 1.0
    assert defaults["top_k"] == 0
    assert defaults["top_p"] == 1.0
    assert defaults["seed"] == 42


def test_a_profile_without_a_seed_gets_42_from_the_defaults(root):
    (root / "profiles" / "noseed.toml").write_text(BASE, encoding="utf-8")
    seed = load_profile(root, "noseed", load_defaults(root)).settings["seed"]
    assert (seed.value, seed.origin) == (42, "default")


def test_settings_come_from_profile_then_defaults_with_origins(root):
    profile = load_profile(root, "text", load_defaults(root))
    assert profile.settings["temperature"].value == 0.6
    assert profile.settings["temperature"].origin == "profile"
    assert profile.settings["repeat_penalty"].value == 1.0
    assert profile.settings["repeat_penalty"].origin == "default"
    assert profile.settings["min_p"].origin == "default"


def test_a_profile_can_describe_itself_in_one_line(root):
    path = root / "profiles" / "described.toml"
    path.write_text('description = "A made-up model, for tests."\n' + BASE, encoding="utf-8")
    assert load_profile(root, "described", load_defaults(root)).description == (
        "A made-up model, for tests."
    )
    assert load_profile(root, "text", load_defaults(root)).description is None


@pytest.mark.parametrize("value", ["3", "true", '"two\\nlines"', '""', '"  "'])
def test_a_description_must_be_one_line_of_text(root, value):
    path = root / "profiles" / "described.toml"
    path.write_text(f"description = {value}\n" + BASE, encoding="utf-8")
    with pytest.raises(ConfigError, match="description must be one line of text"):
        load_profile(root, "described", load_defaults(root))


def test_unknown_setting_name_is_rejected(root):
    write_profile(root, "typo", BASE + "[settings]\ntemprature = 0.5\n")
    with pytest.raises(ConfigError, match="unknown setting 'temprature'"):
        load_profile(root, "typo", load_defaults(root))


def test_unknown_profile_key_is_rejected(root):
    write_profile(root, "typo", BASE + 'system = "hi"\n')  # system is a chat-mode key
    with pytest.raises(ConfigError, match="unknown keys"):
        load_profile(root, "typo", load_defaults(root))


def test_non_neutral_setting_needs_its_sampler(root):
    write_profile(root, "p", BASE + '[settings]\ntop_k = 40\nsamplers = ["top_p", "temperature"]\n')
    with pytest.raises(ConfigError, match="top_k = 40 .* 'top_k' is not in samplers"):
        load_profile(root, "p", load_defaults(root))


def test_penalties_sampler_covers_repeat_presence_frequency(root):
    write_profile(
        root, "p", BASE + '[settings]\npresence_penalty = 0.5\nsamplers = ["temperature"]\n'
    )
    with pytest.raises(ConfigError, match="'penalties' is not in samplers"):
        load_profile(root, "p", load_defaults(root))


def test_neutral_settings_are_fine_without_their_sampler(root):
    defaults = load_defaults(root)
    settings = resolve_settings({"samplers": ["temperature"]}, defaults, "test")
    assert settings["top_k"].value == 0


def test_a_default_that_is_not_neutral_also_needs_its_sampler(root):
    defaults = {**load_defaults(root), "top_k": 40}
    with pytest.raises(ConfigError, match="from default"):
        resolve_settings({"samplers": ["temperature"]}, defaults, "test")


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("top_p", 1.5),
        ("min_p", -0.1),
        ("temperature", -1),
        ("top_k", 1.5),
        ("top_k", True),
        ("max_tokens", 0),
        ("seed", -1),
        ("seed", 2**32 - 1),  # llama.cpp reads this as "random"
        ("seed", "lucky"),
        ("repeat_penalty", 0),
        ("samplers", ["top_k", "typo"]),
        ("samplers", ["top_k", "top_k"]),
    ],
)
def test_bad_values_are_rejected(root, setting, value):
    import json

    write_profile(root, "p", BASE + f"[settings]\n{setting} = {json.dumps(value)}\n")
    with pytest.raises(ConfigError):
        load_profile(root, "p", load_defaults(root))


def test_files_are_read_byte_for_byte_relative_to_the_profile(root):
    (root / "prompts").mkdir()
    (root / "prompts" / "header.txt").write_bytes(b"Line one\r\nline two\n\n")
    write_profile(root, "p", BASE + 'header = { file = "../prompts/header.txt" }\n')
    profile = load_profile(root, "p", load_defaults(root))
    assert profile.header == "Line one\r\nline two\n\n"
    assert profile.to_json()["files"] == {"../prompts/header.txt": "Line one\r\nline two\n\n"}


def test_missing_file_is_an_error(root):
    write_profile(root, "p", BASE + 'header = { file = "nope.txt" }\n')
    with pytest.raises(ConfigError, match="not found"):
        load_profile(root, "p", load_defaults(root))


def test_text_mode_stop_is_newline_user_name(root):
    profile = load_profile(root, "text", load_defaults(root))
    assert profile.stop == ["\nUser:"]


def test_profile_records_its_toml_text(root):
    profile = load_profile(root, "text", load_defaults(root))
    # Byte for byte: on Windows the file has CRLF, and the record keeps it.
    raw = (root / "profiles" / "text.toml").read_bytes().decode("utf-8")
    assert profile.to_json()["toml"] == raw


def test_models_need_full_commit_hashes(root):
    (root / "models.toml").write_text(
        '[m]\nroute = "convert"\nrepo = "a/b"\nrevision = "main"\n'
        'tokenizer_repo = "a/b"\ntokenizer_revision = "main"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="40-character commit"):
        load_models(root)


def test_server_config_builds_flags(root):
    config = load_server_config(root)
    argv = config.flag_argv()
    assert argv[argv.index("--ctx-size") + 1] == "8192"
    assert "--no-context-shift" in argv
    assert "--swa-full" not in argv  # false: left out
    assert config.ctx_size == 8192


@pytest.mark.parametrize(
    "line",
    [
        "gpu-layers = 10",  # an alias of n-gpu-layers
        "grp-attn-n = 4",
        'override-tensor = "blk.0=CPU"',
        "special = true",
        'docker-repo = "a/b"',
        # Flags Clarity sets itself, or that would change the model or how it runs.
        'model = "x.gguf"',
        'hf-repo = "a/b"',
        "context-shift = true",
        'api-key-file = "keys.txt"',
        "webui = true",
        'lora-scaled = "a.gguf:0.5"',
        'control-vector-scaled = "v.gguf:4"',
        "-c = 4096",
    ],
)
def test_server_config_refuses_flags_outside_the_allowed_list(root, line):
    path = root / "server.toml"
    path.write_text(path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="allowed list"):
        load_server_config(root)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('flash-attn = "on"', 'flash-attn = "auto"', "explicit"),
        ('fit = "off"', 'fit = "on"', "fit must be"),
        ("parallel = 1", "parallel = 2", "parallel must be"),
        ("ctx-size = 8192\n", "", "must set"),
        ("log-verbosity = 4 ", "log-verbosity = 3 ", "at least 4"),
        ("cache-ram = 0", "cache-ram = false", "need a value"),  # False == 0 in Python
        ("threads = 8\n", "threads = false\n", "positive integer"),
        ('n-gpu-layers = "all"', "n-gpu-layers = -1", "explicit"),
        # false would leave the flag out, so llama.cpp's default would decide
        ("no-kv-unified = true", "no-kv-unified = false", "must be true"),
        ("no-kv-unified = true", "kv-unified = false", "must be true"),
        ("no-jinja = true", "no-jinja = false", "must be true"),
        # llama-server reads 0 or less as "every core", which depends on the machine
        ("threads = 8\n", "threads = 0\n", "positive integer"),
        ("threads-batch = 8", "threads-batch = -2", "positive integer"),
    ],
)
def test_server_config_requires_explicit_values(root, old, new, message):
    path = root / "server.toml"
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_server_config(root)


@pytest.mark.parametrize("line", ["temperature = nan", "presence_penalty = inf", "top_p = -inf"])
def test_non_finite_numbers_are_rejected(root, line):
    write_profile(root, "p", BASE + f"[settings]\n{line}\n")
    with pytest.raises(ConfigError):
        load_profile(root, "p", load_defaults(root))


@pytest.mark.parametrize("key", ["tokenize", "add_generation_prompt", "chat_template"])
def test_template_kwargs_cannot_override_what_clarity_sets(root, key):
    chat = 'backend = "llama.cpp"\nmodel = "tiny"\nmode = "chat"\n'
    write_profile(root, "p", chat + f'[template_kwargs]\n{key} = "x"\n')
    with pytest.raises(ConfigError, match="can't set"):
        load_profile(root, "p", load_defaults(root))


def test_your_own_profiles_live_in_local_profiles(root):
    own = root / "local" / "profiles"
    own.mkdir(parents=True)
    text = (root / "profiles" / "text.toml").read_text(encoding="utf-8")
    (own / "mine.toml").write_text(text, encoding="utf-8")
    profile = load_profile(root, "mine", load_defaults(root))
    assert profile.path == own / "mine.toml"
    assert load_profile(root, "text", load_defaults(root)).path == root / "profiles" / "text.toml"


def test_a_name_in_both_profile_folders_is_an_error(root):
    own = root / "local" / "profiles"
    own.mkdir(parents=True)
    (own / "text.toml").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="Rename one of them"):
        load_profile(root, "text", load_defaults(root))


@pytest.mark.parametrize("how", ["relative", "absolute", "symlink"])
def test_a_prompt_file_outside_the_project_folder_is_refused(root, how):
    """A profile from someone else mustn't put, say, ~/.ssh/id_rsa into a prompt and a log."""
    outside = root.parent / f"{root.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    if how == "relative":
        ref = f"../../{outside.name}"
    elif how == "absolute":
        ref = outside.as_posix()
    else:
        if os.name == "nt":  # creating a symlink needs extra rights on Windows
            return
        (root / "prompts").mkdir()
        (root / "prompts" / "link.txt").symlink_to(outside)
        ref = "../prompts/link.txt"
    write_profile(root, "p", BASE + f'header = {{ file = "{ref}" }}\n')
    with pytest.raises(ConfigError, match="outside the project folder"):
        load_profile(root, "p", load_defaults(root))
