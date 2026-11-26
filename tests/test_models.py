"""Fetch helpers that don't need the network."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from clarity_chat import models
from clarity_chat.models import check_weights_quick, lock_entry, portable_path, weights_path
from clarity_chat.profiles import ConfigError, load_models, load_server_config, read_models


def test_portable_path_hides_the_home_folder_and_the_repo_location(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert portable_path(str(root / "models" / "m.gguf"), root) == "models/m.gguf"
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    assert portable_path(str(cache), root) == "~/.cache/huggingface/hub"
    assert portable_path("--outtype", root) == "--outtype"
    assert portable_path("bf16", root) == "bf16"


# --- the GGUF-download route --------------------------------------------------------------

WEIGHTS = b"GGUF made-up weights"
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()
GGUF_MODEL = """\
[ready]
route = "gguf"
repo = "test/ready-GGUF"
revision = "1123456789abcdef0123456789abcdef01234567"
file = "ready-Q8_0.gguf"
sha256 = "{sha256}"
tokenizer_repo = "test/ready"
tokenizer_revision = "2123456789abcdef0123456789abcdef01234567"
"""


@pytest.fixture
def gguf_root(root, tmp_path, monkeypatch):
    cache = tmp_path / "hf-cache"
    (cache / "tok").mkdir(parents=True)
    (cache / "tok" / "generation_config.json").write_text(
        json.dumps({"temperature": 0.5, "top_p": 0.9, "max_new_tokens": 100}), encoding="utf-8"
    )
    (cache / "ready-Q8_0.gguf").write_bytes(WEIGHTS)
    monkeypatch.setattr(models, "_download_file", lambda repo, rev, name: cache / name)
    monkeypatch.setattr(models, "_download", lambda repo, rev, patterns: cache / "tok")

    def write(sha256=WEIGHTS_SHA256):
        (root / "models.toml").write_text(GGUF_MODEL.format(sha256=sha256), encoding="utf-8")
        return load_models(root)["ready"]

    return root, write


def test_a_gguf_is_checked_and_recorded_where_it_is(gguf_root):
    root, write = gguf_root
    model = write()
    entry = models.fetch(root, model, load_server_config(root))
    assert entry["gguf"]["sha256"] == WEIGHTS_SHA256
    assert entry["recommended"]["values"] == {"temperature": 0.5, "top_p": 0.9}
    assert entry["recommended"]["source"]["repo"] == "test/ready"
    assert "converter" not in entry
    assert check_weights_quick(root, lock_entry(root, model)).read_bytes() == WEIGHTS


def test_a_gguf_with_the_wrong_sha256_is_refused(gguf_root):
    root, write = gguf_root
    model = write(sha256="0" * 64)
    with pytest.raises(ConfigError, match="sha256"):
        models.fetch(root, model, load_server_config(root))
    assert "ready" not in models.read_lock(root)["models"]


def test_a_shipped_gguf_refused_before_its_first_fetch_names_local_models_lock(gguf_root):
    """The whole-branch review's C20: every fetch writes local/models.lock (A10), so a
    refused one says it recorded nothing there, not in the shipped models.lock."""
    root, write = gguf_root
    with pytest.raises(ConfigError, match=r"Not recorded in local/models\.lock\."):
        models.fetch(root, write(sha256="0" * 64), load_server_config(root))


def test_the_lock_must_match_the_gguf_in_models_toml(gguf_root):
    root, write = gguf_root
    models.fetch(root, write(), load_server_config(root))
    with pytest.raises(ConfigError, match="different sha256"):
        lock_entry(root, write(sha256="1" * 64))


def test_weights_paths_expand_the_home_folder(tmp_path):
    assert weights_path(tmp_path, {"gguf": {"path": "~/x.gguf"}}) == Path.home() / "x.gguf"
    assert weights_path(tmp_path, {"gguf": {"path": "models/x.gguf"}}) == tmp_path / "models/x.gguf"


def test_a_route_change_is_reported_as_a_route_change(gguf_root):
    root, write = gguf_root
    models.fetch(root, write(), load_server_config(root))
    (root / "models.toml").write_text(
        GGUF_MODEL.format(sha256=WEIGHTS_SHA256)
        .replace('route = "gguf"', 'route = "convert"')
        .replace('file = "ready-Q8_0.gguf"\n', "")
        .replace(f'sha256 = "{WEIGHTS_SHA256}"\n', ""),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="route"):
        lock_entry(root, load_models(root)["ready"])


@pytest.mark.parametrize("refused", ["weights", "tokenizer"])
def test_a_gated_repo_without_access_is_refused_with_what_to_do(
    root, tmp_path, monkeypatch, capsys, refused
):
    """Hugging Face answers a gated repo, without an accepted licence or a login, with
    GatedRepoError. `clarity fetch` says what to do instead of showing a traceback, and
    handles no token itself."""
    import httpx2
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    from clarity_chat import cli

    asked: list[dict] = []  # each download's keyword arguments

    def gated(repo, *args, **kwargs):
        asked.append(kwargs)
        response = httpx2.Response(401, request=httpx2.Request("GET", f"https://hf.co/{repo}"))
        raise GatedRepoError(f"Access to model {repo} is restricted.", response=response)

    weights = tmp_path / "ready-Q8_0.gguf"
    weights.write_bytes(WEIGHTS)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        gated if refused == "weights" else lambda *a, **k: (asked.append(k), weights)[1],
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    (root / "models.toml").write_text(GGUF_MODEL.format(sha256=WEIGHTS_SHA256), encoding="utf-8")
    try:
        code = cli.main(["--root", str(root), "fetch", "ready"])
    except GatedRepoError as e:  # what the user would see: a traceback
        code = repr(e)
    assert code == 1
    message = capsys.readouterr().err
    repo = "test/ready-GGUF" if refused == "weights" else "test/ready"
    assert f"https://huggingface.co/{repo}" in message
    assert "licence" in message and "hf auth login" in message
    assert "ready" not in models.read_lock(root)["models"]
    # Clarity passes no token: huggingface_hub finds the login itself (the whole-branch
    # review: in both cases, the weights' download and the tokenizer's).
    assert asked and all(k.get("token") is None for k in asked), asked


def test_portable_path_can_keep_a_symlinks_name(tmp_path):
    root = tmp_path / "repo"
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / "abc123").write_bytes(b"x")
    (root / "model.gguf").symlink_to(root / "blobs" / "abc123")
    assert portable_path(str(root / "model.gguf"), root, resolve=False) == "model.gguf"
    assert portable_path(str(root / "model.gguf"), root) == "blobs/abc123"


def test_verify_records_a_new_time_when_the_bytes_match(gguf_root):
    import os

    root, write = gguf_root
    model = write()
    models.fetch(root, model, load_server_config(root))
    path = weights_path(root, lock_entry(root, model))
    os.utime(path, ns=(1, 1))  # same bytes, new modified time
    with pytest.raises(ConfigError, match="changed since it was fetched"):
        check_weights_quick(root, lock_entry(root, model))
    ok, message = models.verify(root, "ready")
    assert ok and "updated" in message
    check_weights_quick(root, lock_entry(root, model))


def test_the_shipped_models_toml_loads():
    from clarity_chat.models import read_lock_file
    from clarity_chat.profiles import read_models

    repo = Path(__file__).resolve().parents[1]
    models = read_models(repo / "models.toml")  # the shipped file only, never local/
    assert "bge-m3" in models and "olmo-3-7b-instruct" in models
    # v0.4.1 decision 1: bge-m3 replaces bge-small, whose lock entry goes with it.
    assert "bge-small-en-v1.5" not in models
    assert "bge-small-en-v1.5" not in read_lock_file(repo / "models.lock")["models"]
    bge = models["bge-m3"]
    assert (bge.route, bge.repo, bge.tokenizer_repo, bge.weights) == (
        "convert",
        "BAAI/bge-m3",
        "BAAI/bge-m3",
        "pytorch_model.bin",
    )
    assert bge.tokenizer_revision == bge.revision
    assert bge.weights_sha256 is not None and len(bge.weights_sha256) == 64  # S1


@pytest.mark.parametrize("name", ["", ".converter", "..", ".hidden", "a/b", "a\\b"])
@pytest.mark.parametrize("yours", [False, True])
def test_a_model_name_is_one_folder_name(tmp_path, name, yours):
    """A model's files go in models/NAME, beside the converter's environment in
    models/.converter: a name with a leading dot could be that folder, or the one above,
    one with a slash or backslash a folder elsewhere, and an empty one models/ itself
    (the whole-branch review's C7)."""
    from clarity_chat.profiles import read_models

    path = tmp_path / "models.toml"
    # a literal key, so a backslash is itself
    path.write_text(CONVERT_MODEL.replace("[embedder]", f"['{name}']"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"one folder name"):
        read_models(path, yours=yours)


# --- the weights key (v0.4.1 decision 2) ----------------------------------------------------

CONVERT_MODEL = """\
[embedder]
route = "convert"
repo = "test/embedder"
revision = "3123456789abcdef0123456789abcdef01234567"
tokenizer_repo = "test/embedder"
tokenizer_revision = "3123456789abcdef0123456789abcdef01234567"
"""
# A made-up listing laid out as bge-m3's repo is: the weights only as a pickle, beside two
# more pickles for outputs Clarity doesn't use, and an ONNX copy.
REPO_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
    "pytorch_model.bin",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "onnx/config.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "README.md",
]
WEIGHT_FILES = (".bin", ".pt", ".pth", ".onnx", ".onnx_data", ".safetensors")
PICKLE = b"made-up pickle bytes"
PICKLE_SHA256 = hashlib.sha256(PICKLE).hexdigest()
WITH_WEIGHTS = (
    CONVERT_MODEL + f'weights = "pytorch_model.bin"\nweights_sha256 = "{PICKLE_SHA256}"\n'
)


def downloaded(root: Path, model, monkeypatch) -> list[str]:
    """The patterns fetch_convert asks Hugging Face for; it stops before downloading."""
    from huggingface_hub.utils import filter_repo_objects

    asked: list[list[str]] = []

    class Stop(Exception):
        pass

    def download(repo, revision, patterns):
        asked.append(patterns)
        raise Stop

    server = load_server_config(root)
    monkeypatch.setattr(models, "git_state", lambda path: {"commit": server.llama_cpp_commit})
    monkeypatch.setattr(models, "_download", download)
    with pytest.raises(Stop):
        models.fetch_convert(root, model, server)
    weights = filter_repo_objects(REPO_FILES, allow_patterns=asked[0])
    return [name for name in weights if name.endswith(WEIGHT_FILES)]


def test_weights_adds_exactly_the_publishers_pickle_to_the_download(root, monkeypatch):
    (root / "models.toml").write_text(WITH_WEIGHTS, encoding="utf-8")
    assert downloaded(root, load_models(root)["embedder"], monkeypatch) == ["pytorch_model.bin"]


def test_an_entry_without_weights_downloads_no_pickle_as_before(root, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    smollm2 = read_models(repo / "models.toml")["smollm2-135m-instruct"]  # shipped only
    assert models.SOURCE_PATTERNS == ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]
    assert downloaded(root, smollm2, monkeypatch) == []
    (root / "models.toml").write_text(CONVERT_MODEL, encoding="utf-8")
    assert downloaded(root, load_models(root)["embedder"], monkeypatch) == []


@pytest.mark.parametrize(
    "value",
    [
        '"colbert_linear.pt"',
        '"model.safetensors"',
        '"onnx/pytorch_model.bin"',
        '"./pytorch_model.bin"',
        '"Pytorch_model.bin"',
        '"pytorch_model*.bin"',
        '["pytorch_model.bin"]',
        '""',
        "true",
    ],
)
def test_weights_accepts_no_other_value(root, value):
    (root / "models.toml").write_text(CONVERT_MODEL + f"weights = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match='weights can only be "pytorch_model.bin", got'):
        load_models(root)


def test_a_gguf_entry_takes_no_weights_key(root):
    (root / "models.toml").write_text(
        GGUF_MODEL.format(sha256=WEIGHTS_SHA256) + 'weights = "pytorch_model.bin"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="needs exactly the keys"):
        load_models(root)


# --- weights_sha256 (v0.4.1's security review, S1; the owner's Q16) --------------------------


@pytest.mark.parametrize(
    "lines",
    [
        'weights = "pytorch_model.bin"\n',  # no hash
        f'weights_sha256 = "{PICKLE_SHA256}"\n',  # a hash with no weights
    ],
)
def test_weights_and_its_hash_come_together(root, lines):
    (root / "models.toml").write_text(CONVERT_MODEL + lines, encoding="utf-8")
    with pytest.raises(ConfigError, match="needs exactly the keys"):
        load_models(root)


@pytest.mark.parametrize("value", ['"0"', f'"{PICKLE_SHA256.upper()}"', '["x"]'])
def test_weights_sha256_is_64_lowercase_hex(root, value):
    toml = WITH_WEIGHTS.replace(f'"{PICKLE_SHA256}"', value)
    (root / "models.toml").write_text(toml, encoding="utf-8")
    with pytest.raises(ConfigError, match="weights_sha256 must be 64 lowercase hex characters"):
        load_models(root)


def fetched_with(root: Path, tmp_path: Path, monkeypatch, pickle: bytes) -> list[list[str]]:
    """fetch_convert on WITH_WEIGHTS, the download holding `pickle`; the converter is
    stubbed (it writes its outfile and versions file), and so is making its environment.
    Returns the converters started."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "pytorch_model.bin").write_bytes(pickle)
    (root / "models.toml").write_text(WITH_WEIGHTS, encoding="utf-8")
    server = load_server_config(root)
    monkeypatch.setattr(models, "git_state", lambda path: {"commit": server.llama_cpp_commit})
    monkeypatch.setattr(models, "_download", lambda repo, revision, patterns: snapshot)
    started: list[list[str]] = []

    class Converted:
        def __init__(self, command, **kwargs):
            self.stdout = iter(())
            if command[1:2] == ["venv"]:  # its first word is $UV or uv
                Path(command[-1]).mkdir(parents=True)
            if "--outfile" not in command:
                return
            started.append(command)
            outfile = Path(command[command.index("--outfile") + 1])
            outfile.write_bytes(b"GGUF converted")
            versions = {"python": "3.13", "packages": {"torch": "0", "transformers": "0"}}
            (outfile.parent / "converter-versions.json").write_text(json.dumps(versions))

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Converted)
    models.fetch_convert(root, load_models(root)["embedder"], server)
    return started


def test_a_pickle_with_another_hash_is_refused_before_the_converter(root, tmp_path, monkeypatch):
    started: list[list[str]] = []
    try:
        started = fetched_with(root, tmp_path, monkeypatch, b"swapped bytes")
        refused = None
    except ConfigError as e:
        refused = str(e)
    assert started == [], "the converter loaded a pickle whose hash didn't match"
    assert refused is not None and f"but models.toml says {PICKLE_SHA256}" in refused
    assert refused.endswith("Not converted or recorded in local/models.lock.")  # C20
    assert not (root / "models" / "embedder").exists()  # nothing written
    assert "embedder" not in models.read_lock(root)["models"]


def test_the_pickle_s_hash_is_written_to_the_lock(root, tmp_path, monkeypatch):
    assert len(fetched_with(root, tmp_path, monkeypatch, PICKLE)) == 1
    assert models.read_lock(root)["models"]["embedder"].get("weights_sha256") == PICKLE_SHA256


def test_the_converter_record_keeps_the_version_python_imports(tmp_path, monkeypatch):
    """The CI plan's advisor: in the converter's environment a package can be installed
    twice, in uv's overlay (llama.cpp's pins, first on sys.path, the one Python imports)
    and in the base environment under it. The record kept the last one distributions()
    gave, so models.lock said transformers 5.22.2 where the converter ran 4.57.6. It keeps
    the first on sys.path, as importlib.metadata.version() does."""
    import sys

    from clarity_chat import convert_shim

    for folder, version in (("overlay", "4.57.6"), ("base", "5.22.2")):
        info = tmp_path / folder / f"made_up_transformers-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: made-up-transformers\nVersion: {version}\n", "utf-8"
        )
    converter = tmp_path / "convert.py"
    converter.write_text("", "utf-8")  # runs, converts nothing
    versions = tmp_path / "versions.json"
    overlay_first = [str(tmp_path / "overlay"), str(tmp_path / "base"), *sys.path]
    monkeypatch.setattr(sys, "path", overlay_first)
    monkeypatch.setattr(sys, "argv", ["convert_shim.py", str(versions), str(converter)])
    convert_shim.main()
    packages = json.loads(versions.read_text("utf-8"))["packages"]
    assert packages["made-up-transformers"] == "4.57.6"


# --- The converter's environment, hashed (v0.4.2 decision 15, A9) ---------------------------

REQUIREMENTS = Path(models.__file__).with_name("converter-requirements.txt")
MADE_UP_REQUIREMENTS = "made-up==1.0 --hash=sha256:" + "0" * 64 + "\n"


def made_up_requirements(root: Path) -> Path:
    """Where `converting` puts its made-up requirements file: outside the root's top level,
    since Clarity reads the file from its own code, not from the root."""
    return root / "code" / "converter-requirements.txt"


def converting(
    root: Path,
    monkeypatch,
    fail: str | None = None,
    seen=lambda command: None,
    requirements: Path | None = None,
) -> list[tuple[list[str], dict]]:
    """fetch_convert on the shipped SmolLM2 entry, into `root`, with every command stubbed:
    `uv venv` makes its folder and python; the converter writes its outfile and versions
    file. `fail` names the command that exits 1 (venv, install or convert); `seen` is called
    with each command as it starts, and with ["download", REPO] for the download. The requirements
    file is `requirements`, or a made-up one (made_up_requirements) written unless it's
    there. Returns the commands started, with their keyword arguments."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    server = load_server_config(repo)
    model = read_models(repo / "models.toml")["smollm2-135m-instruct"]  # shipped only
    if requirements is None:
        requirements = made_up_requirements(root)
        requirements.parent.mkdir(parents=True, exist_ok=True)
        if not requirements.is_file():
            requirements.write_text(MADE_UP_REQUIREMENTS, "utf-8")
    monkeypatch.setattr(models, "CONVERTER_REQUIREMENTS", requirements)
    snapshot = root / "snapshot"
    snapshot.mkdir(exist_ok=True)
    monkeypatch.setattr(models, "git_state", lambda path: {"commit": server.llama_cpp_commit})

    def download(repo, revision, patterns):
        seen(["download", repo])
        return snapshot

    monkeypatch.setattr(models, "_download", download)
    started: list[tuple[list[str], dict]] = []

    class Process:
        def __init__(self, command, **kwargs):
            seen(command)
            started.append((command, kwargs))
            kind = {"venv": "venv", "pip": "install"}.get(command[1], "convert")
            if kind == "venv":  # --clear: whatever was in the folder goes, the marker too
                shutil.rmtree(command[-1], ignore_errors=True)
                env_python(Path(command[-1])).parent.mkdir(parents=True)
                env_python(Path(command[-1])).write_text("")
            if kind == "convert" and fail != "convert":
                outfile = Path(command[command.index("--outfile") + 1])
                outfile.write_bytes(b"GGUF converted")
                versions = {"python": "3.13", "packages": {"torch": "0", "transformers": "0"}}
                (outfile.parent / "converter-versions.json").write_text(json.dumps(versions))
            self.code = 1 if kind == fail else 0
            self.stdout = iter(())

        def wait(self):
            return self.code

    monkeypatch.setattr(subprocess, "Popen", Process)
    models.fetch_convert(root, model, server)
    return started


def env_python(env: Path) -> Path:
    return env / "Scripts" / "python.exe" if os.name == "nt" else env / "bin" / "python"


def test_the_converter_s_environment_is_made_from_the_hashed_requirements(tmp_path, monkeypatch):
    """`uv run --with-requirements` doesn't check a file's hashes (uv 0.11.19 installed and
    ran one with a zeroed hash), so Clarity makes the environment itself (the A9 ruling):
    `uv venv`, then `uv pip install` with hashes required, then that environment's Python
    with -I. A fresh venv holds only what the file installs (earlier, from Clarity's folder,
    the converter's sys.path held Clarity's .venv and src); --no-project and --no-config keep
    Clarity's pyproject.toml and your uv settings out. The lock records the three commands."""
    monkeypatch.delenv("UV", raising=False)  # run from a shell, not by uv: plain `uv`
    started = converting(tmp_path, monkeypatch)
    assert len(started) == 3, started  # make the environment, install, convert
    venv, install, convert = [command for command, _ in started]
    env = tmp_path / "models" / ".converter"
    assert venv[:2] == ["uv", "venv"] and venv[-1] == str(env), venv
    assert venv[venv.index("--python") + 1] == "3.13", venv
    # v0.4.6 S3: only a Python uv installed, never conda's, Homebrew's or the system's
    assert "--managed-python" in venv, venv
    assert {"--clear", "--no-project", "--no-config"} <= set(venv), venv
    assert install[:3] == ["uv", "pip", "install"] and "--require-hashes" in install, install
    # --require-hashes doesn't cover an sdist's build dependencies (its build backend comes
    # unhashed from the index and runs): only wheels, every one of them hashed.
    assert "--no-build" in install, install
    assert install[install.index("-r") + 1] == str(made_up_requirements(tmp_path)), install
    assert install[install.index("--python") + 1] == str(env), install
    assert "--no-config" in install, install
    assert convert[:2] == [str(env_python(env)), "-I"], convert
    lock = json.loads((tmp_path / "local" / "models.lock").read_text("utf-8"))
    recorded = lock["models"]["smollm2-135m-instruct"]["converter"]["command"]
    assert recorded == [
        [portable_path(part, tmp_path, resolve=False) for part in command]
        for command in (venv, install, convert)
    ]
    assert recorded[0][-1] == "models/.converter", recorded


def test_the_lock_records_the_sha256_of_the_requirements_the_converter_ran_with(
    tmp_path, monkeypatch
):
    """v0.4.6 S4 (decision 5): the converter's lock entry holds the sha256 of the
    converter-requirements.txt its environment was installed from, so a conversion says
    which pins made it; a changed file changes it."""

    def recorded() -> str | None:
        lock = json.loads((tmp_path / "local" / "models.lock").read_text("utf-8"))
        return lock["models"]["smollm2-135m-instruct"]["converter"].get("requirements_sha256")

    requirements = made_up_requirements(tmp_path)
    converting(tmp_path, monkeypatch)
    first = hashlib.sha256(requirements.read_bytes()).hexdigest()
    assert recorded() == first
    requirements.write_text(MADE_UP_REQUIREMENTS.replace("1.0", "1.1"), "utf-8")
    converting(tmp_path, monkeypatch)
    assert recorded() == hashlib.sha256(requirements.read_bytes()).hexdigest() != first


def test_the_converter_runs_the_uv_that_runs_clarity(tmp_path, monkeypatch):
    """install.sh runs Clarity under `env -i` with a PATH that doesn't hold ~/.local/bin,
    where uv's installer puts uv; uv sets UV to itself for what it runs, so the converter
    uses that uv, as setup does (the whole-branch review's M1)."""
    uv = "/Users/made-up/.local/bin/uv"
    monkeypatch.setenv("UV", uv)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # install.sh's PATH: no ~/.local/bin
    venv, install, _ = models.converter_commands(tmp_path)
    assert venv[0] == uv and install[0] == uv


def test_a_uv_in_the_home_folder_is_recorded_from_the_home_folder(tmp_path, monkeypatch):
    """The lock may be committed (scripts/refresh_lock.py), so a uv under the home folder is
    recorded as `~/...`, with no user name, like the lock's other paths."""
    monkeypatch.setenv("UV", str(Path.home() / ".local" / "bin" / "uv"))
    converting(tmp_path, monkeypatch)
    lock = json.loads((tmp_path / "local" / "models.lock").read_text("utf-8"))
    recorded = lock["models"]["smollm2-135m-instruct"]["converter"]["command"]
    assert [recorded[0][0], recorded[1][0]] == ["~/.local/bin/uv"] * 2, recorded


def test_the_converter_s_environment_is_said_before_it_is_made(tmp_path, monkeypatch, capsys):
    """Clarity writes files only visibly: the folder is named before `uv venv` writes it."""
    printed: dict[str, str] = {}

    def seen(command):
        printed.setdefault(command[1], capsys.readouterr().out)

    converting(tmp_path, monkeypatch, seen=seen)
    assert "venv" in printed, printed
    assert "models/.converter" in printed["venv"], printed
    # the file's path in Clarity's folder, as the Converting line gives the shim's: findable
    assert " code/converter-requirements.txt " in printed["venv"], printed


CONVERTED = hashlib.sha256(b"GGUF converted").hexdigest()  # what `converting` writes


def shipped_conversion(
    root: Path, sha256: str, same_commit: bool = True, same_revision: bool = True
) -> None:
    """A shipped models.lock in `root` with SmolLM2's real entry, converted at server.toml's
    llama.cpp commit (or another) from its revision (or another), with the GGUF sha256
    `sha256`."""
    repo = Path(__file__).resolve().parents[1]
    entry = json.loads((repo / "models.lock").read_text("utf-8"))["models"]["smollm2-135m-instruct"]
    commit = load_server_config(repo).llama_cpp_commit if same_commit else "f" * 40
    entry = {**entry, "gguf": {**entry["gguf"], "sha256": sha256}, "llama_cpp": {"commit": commit}}
    if not same_revision:
        entry["revision"] = "e" * 40
    lock = {"schema_version": 1, "models": {"smollm2-135m-instruct": entry}}
    (root / "models.lock").write_text(json.dumps(lock), "utf-8")


DIFFERS = "Note: smollm2-135m-instruct's GGUF (sha256 {new}) differs from the shipped one"


def test_a_conversion_unlike_the_shipped_one_is_said(tmp_path, monkeypatch, capsys):
    """The security review's S6: a fetch records its GGUF in local/models.lock, which wins
    over the shipped entry, so a conversion that came out different (another Python, say)
    would be used without a word. Same revision, same llama.cpp commit, another sha256: one
    sentence says so. It's a flag, not a refusal: the entry is still recorded."""
    shipped_conversion(tmp_path, "1" * 64)
    converting(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    said = DIFFERS.format(new=CONVERTED[:12]) + " in models.lock (111111111111)"
    assert said in out, out
    lock = json.loads((tmp_path / "local" / "models.lock").read_text("utf-8"))
    assert lock["models"]["smollm2-135m-instruct"]["gguf"]["sha256"] == CONVERTED


@pytest.mark.parametrize(
    "case", ["same sha256", "another llama.cpp commit", "another revision", "nothing shipped"]
)
def test_a_conversion_like_the_shipped_one_or_not_comparable_says_nothing(
    tmp_path, monkeypatch, capsys, case
):
    """The same bytes say nothing; nor does another llama.cpp commit or revision, whose
    conversion may rightly differ, or a model the shipped lock doesn't hold."""
    if case != "nothing shipped":
        sha256 = CONVERTED if case == "same sha256" else "1" * 64
        commit, revision = case != "another llama.cpp commit", case != "another revision"
        shipped_conversion(tmp_path, sha256, same_commit=commit, same_revision=revision)
    converting(tmp_path, monkeypatch)
    assert "differs from the shipped one" not in capsys.readouterr().out


def test_the_converter_s_environment_is_made_again_only_when_the_requirements_change(
    tmp_path, monkeypatch
):
    """A marker in the environment holds the sha256 of the file it was made from: while the
    file is unchanged, a fetch only converts; a changed file (a new llama.cpp pin) remakes it."""
    assert len(converting(tmp_path, monkeypatch)) == 3
    again = converting(tmp_path, monkeypatch)
    assert [command[:2] for command, _ in again] == [
        [str(env_python(tmp_path / "models" / ".converter")), "-I"]
    ]
    made_up_requirements(tmp_path).write_text(MADE_UP_REQUIREMENTS.replace("1.0", "1.1"), "utf-8")
    assert len(converting(tmp_path, monkeypatch)) == 3


def test_a_change_to_the_setup_commands_makes_the_environment_again(tmp_path, monkeypatch):
    """The marker covers the commands that made the environment as well as the file, so a
    new flag (--no-build, say) or Python version remakes an environment made without it."""
    assert len(converting(tmp_path, monkeypatch)) == 3
    monkeypatch.setattr(models, "CONVERTER_PYTHON", "3.12")
    again = [command for command, _ in converting(tmp_path, monkeypatch)]
    assert [command[1] for command in again][:2] == ["venv", "pip"], again


def test_the_venv_command_runs_without_uv_python_preference(tmp_path, monkeypatch):
    """The whole-branch review's finding 1: uv refuses `--managed-python` beside
    UV_PYTHON_PREFERENCE (uv 0.11.19: "the argument `--managed-python` cannot be used with
    `--python-preference`"), so the venv command, and only it, runs without that variable.
    uv's other variables still reach it."""
    monkeypatch.setenv("UV_PYTHON_PREFERENCE", "only-managed")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://index.example/simple")
    (venv, made), (install, installed), _ = converting(tmp_path, monkeypatch)
    assert venv[1] == "venv" and install[1] == "pip", (venv, install)
    env = made.get("env")
    assert env is not None and "UV_PYTHON_PREFERENCE" not in env, env
    assert env["UV_DEFAULT_INDEX"] == "https://index.example/simple"
    assert installed.get("env") is None  # the install inherits Clarity's environment


def test_an_environment_made_by_v0_4_5_s_commands_is_made_again(tmp_path, monkeypatch):
    """v0.4.6 S3 (decision 4, state 5): an environment made before `--managed-python`, on
    whichever Python 3.13 uv found first, is made again once, on a Python uv installed. A
    marker from today's commands is the control: with it, the fetch only converts."""
    monkeypatch.delenv("UV", raising=False)
    assert len(converting(tmp_path, monkeypatch)) == 3
    env = tmp_path / "models" / ".converter"
    _, install, _ = models.converter_commands(tmp_path)
    before = ["uv", "venv", "--clear", "--no-project", "--no-config", "--python", "3.13", str(env)]

    def mark(venv: list[str]) -> None:
        made_by = {
            "commands": [venv, install],
            "requirements": models.sha256_file(made_up_requirements(tmp_path)),
        }
        digest = hashlib.sha256(json.dumps(made_by).encode("utf-8")).hexdigest()
        (env / models.CONVERTER_MARKER).write_text(digest + "\n", "utf-8")

    mark(models.converter_commands(tmp_path)[0])
    assert len(converting(tmp_path, monkeypatch)) == 1
    mark(before)
    again = [command for command, _ in converting(tmp_path, monkeypatch)]
    assert [command[1] for command in again][:2] == ["venv", "pip"], again


def test_a_failed_remake_then_the_old_file_makes_it_again_in_full(tmp_path, monkeypatch):
    """A, then B whose install fails, then A again: `uv venv --clear` emptied the folder
    for B, so A's environment is gone and A is made again in full, not reused."""
    assert len(converting(tmp_path, monkeypatch)) == 3
    requirements = made_up_requirements(tmp_path)
    a = requirements.read_text("utf-8")
    requirements.write_text(a.replace("1.0", "1.1"), "utf-8")
    with pytest.raises(ConfigError):
        converting(tmp_path, monkeypatch, fail="install")
    requirements.write_text(a, "utf-8")
    assert len(converting(tmp_path, monkeypatch)) == 3


def test_an_environment_whose_python_is_gone_is_made_again(tmp_path, monkeypatch):
    """The marker alone doesn't make an environment usable: its python is a link to the one
    uv chose, which can be removed (`uv python uninstall`, a Homebrew upgrade)."""
    assert len(converting(tmp_path, monkeypatch)) == 3
    env_python(tmp_path / "models" / ".converter").unlink()
    again = [command for command, _ in converting(tmp_path, monkeypatch)]
    assert [command[1] for command in again][:2] == ["venv", "pip"], again


@pytest.mark.parametrize("kind", ["venv", "pip", "-I"])
def test_a_command_that_can_t_start_names_the_environment_as_safe_to_delete(
    tmp_path, monkeypatch, kind
):
    """uv missing from PATH, or the environment's python unrunnable, is a ConfigError, not a
    traceback: it names the environment and says deleting it is safe."""

    def cannot_start(command):
        if command[0] != "download" and command[1] == kind:
            raise FileNotFoundError(2, "No such file or directory", command[0])

    if kind == "-I":  # the environment exists, so only the converter runs
        converting(tmp_path, monkeypatch)
    try:
        converting(tmp_path, monkeypatch, seen=cannot_start)
    except Exception as error:  # noqa: BLE001 - the red is the type, asserted below
        caught = error
    else:
        pytest.fail("no error")
    assert isinstance(caught, ConfigError), repr(caught)
    assert "models/.converter" in str(caught) and "safe to delete" in str(caught), caught


@pytest.mark.parametrize("fail", ["venv", "install"])
def test_a_failed_environment_leaves_no_marker_so_the_next_fetch_makes_it_again(
    tmp_path, monkeypatch, fail
):
    """A failed `uv venv` or install stops the fetch before the converter, names
    convert.log, records nothing, and leaves no marker, so the next fetch starts over."""
    with pytest.raises(ConfigError, match=r"converter's environment.*convert\.log") as caught:
        converting(tmp_path, monkeypatch, fail=fail)
    # uv's words for a file whose sha256 isn't the pinned one: say what they mean
    assert '"Hash mismatch"' in str(caught.value), caught.value
    assert not (tmp_path / "local" / "models.lock").exists()
    assert len(converting(tmp_path, monkeypatch)) == 3


def test_the_requirements_come_from_clarity_s_code_not_the_root(tmp_path):
    """The file goes with the code that uses it, beside convert_shim.py: a root holds only
    what fetch writes (models/, local/models.lock), so a scratch root converts as is. At
    20ef13a it was read from the root, and -m live's convert test, whose root is a scratch
    folder, failed with FileNotFoundError."""
    install = models.converter_commands(tmp_path)[1]
    assert install[install.index("-r") + 1] == str(REQUIREMENTS), install
    assert REQUIREMENTS.is_file(), REQUIREMENTS


def test_a_missing_requirements_file_stops_before_any_download(tmp_path, monkeypatch):
    """A checkout without the file (deleted, say) stops with a ConfigError that names it,
    before anything is downloaded or run, not with a raw FileNotFoundError."""
    missing = tmp_path / "code" / "converter-requirements.txt"
    started: list[list[str]] = []
    with pytest.raises(Exception) as raised:
        converting(tmp_path, monkeypatch, seen=started.append, requirements=missing)
    assert isinstance(raised.value, ConfigError), repr(raised.value)
    assert str(missing) in str(raised.value), raised.value
    assert started == [], started  # no download, no command


def test_the_converter_runs_offline_with_no_terminal_input(tmp_path, monkeypatch):
    """v0.4.1's security review: S2, the converter needs no network (everything it reads is
    in the snapshot), so HF_HUB_OFFLINE stops a tokenizer's auto_map from fetching code
    from another repo, and with stdin closed a "run the custom code? [y/N]" prompt, which
    would go unseen to convert.log, is refused. C1, uv's first-index strategy is passed, so
    UV_INDEX_STRATEGY in your shell can't merge indexes (the hashes would refuse another
    file anyway). Making the environment needs the network, so it keeps HF_HUB_OFFLINE."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    started = converting(tmp_path, monkeypatch)
    assert len(started) == 3, started  # make the environment, install, convert
    (_, _), (install, install_kwargs), (_, kwargs) = started
    env = kwargs.get("env") or {}
    assert env.get("HF_HUB_OFFLINE") == "1", kwargs
    assert {k: v for k, v in env.items() if k != "HF_HUB_OFFLINE"} == {
        k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"
    }
    assert kwargs.get("stdin") == subprocess.DEVNULL, kwargs
    assert install_kwargs.get("stdin") == subprocess.DEVNULL, install_kwargs
    assert install[install.index("--index-strategy") + 1] == "first-index", install


def requirement_lines() -> list[str]:
    """converter-requirements.txt's requirements, each on one line, without comments."""
    assert REQUIREMENTS.is_file(), f"{REQUIREMENTS.name} is missing"
    text = REQUIREMENTS.read_text("utf-8").replace("\\\n", " ")
    return [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_every_converter_requirement_is_pinned_with_its_hashes():
    """Each line is NAME==VERSION, at most a platform marker, and its sha256 hashes: no index
    URL, no unpinned or unhashed package, so `--require-hashes` checks every file."""
    import re

    lines = requirement_lines()
    assert lines
    pinned = re.compile(
        r"[A-Za-z0-9._-]+==[A-Za-z0-9.+]+( ; sys_platform == '\w+')?( --hash=sha256:[0-9a-f]{64})+"
    )
    assert [line for line in lines if not pinned.fullmatch(line)] == []


def test_the_converter_requirements_are_the_versions_bge_m3_s_conversion_recorded():
    """The file pins what the converter ran with (the shipped models.lock's bge-m3 entry,
    converter.packages), so a conversion gets the same packages as before and no GGUF
    changes. Linux adds PyPI torch's NVIDIA libraries and triton, Windows colorama; macOS
    gets exactly the record. SmolLM2's record isn't compared: it was fetched (2026-11-08)
    before the converter ran isolated (4aba42c), so it lists Clarity's own packages (pytest,
    ruff, transformers 5.22.2), not what the converter imported. CI's live-cpu job checks
    its GGUF: a fetch in a scratch folder gives the sha256 the lock records."""
    import re

    def canonical(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    everywhere: dict[str, str] = {}
    markers: set[str] = set()
    for line in requirement_lines():
        spec, _, marker = line.split(" --hash")[0].partition(" ; ")
        name, version = spec.split("==")
        if marker:
            markers.add(marker)
        else:
            everywhere[canonical(name)] = version
    lock = json.loads((Path(__file__).resolve().parents[1] / "models.lock").read_text("utf-8"))
    record = lock["models"]["bge-m3"]["converter"]["packages"]
    assert everywhere == {canonical(name): version for name, version in record.items()}
    assert markers <= {"sys_platform == 'linux'", "sys_platform == 'win32'"}, markers


def test_the_converter_s_sentencepiece_is_pinned():
    """llama.cpp's converter requirements at the pin allow sentencepiece <0.3.0 but cap
    protobuf <5.0.0; sentencepiece 0.2.3's model_pb2 needs protobuf's runtime_version
    (5.26 on), so bge-m3's vocabulary failed to load (the first fetch, 2026-11-18). The
    file pins 0.2.2, which imports with protobuf 4.25.9 on Python 3.13."""
    specs = {line.split(" ")[0] for line in requirement_lines()}
    assert {"sentencepiece==0.2.2", "protobuf==4.25.9"} <= specs, specs


@pytest.fixture(scope="module")
def converter_python(tmp_path_factory) -> list[str]:
    """The converter's environment, made by fetch_convert's own setup commands from Clarity's
    folder (where uv would find Clarity's .venv) in a scratch folder; the command that runs
    its python. One per module: it installs torch. Live only: it needs the pinned packages
    (uv's cache, or the network)."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    venv, install, python = models.converter_commands(tmp_path_factory.mktemp("converter"))
    for command in (venv, install):
        subprocess.run(command, cwd=repo, check=True, stdin=subprocess.DEVNULL)
    return python


def in_converter(python: list[str], script: str, *args: str):
    """Run `script` in the converter's environment, from Clarity's folder; its last line of
    output, read as JSON."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [*python, "-c", script, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.live
def test_the_converter_sees_nothing_of_claritys_environment_and_imports_its_pinned_packages(
    converter_python,
):
    """The probe behind the isolation fix: no .venv or src on sys.path, none of Clarity's dev
    tools installed, the transformers llama.cpp pins imported, and sentencepiece's protobuf
    module, which failed to load with sentencepiece 0.2.3 and protobuf 4 (bge-m3's
    vocabulary). Live: it needs the llama.cpp checkout and the converter's environment."""
    repo = Path(__file__).resolve().parents[1]
    server = load_server_config(repo)
    probe = (
        "import json, sys, transformers, sentencepiece\n"
        "from sentencepiece import sentencepiece_model_pb2\n"
        "from importlib.metadata import distributions\n"
        "names = sorted({d.metadata['Name'].lower() for d in distributions()})\n"
        "print(json.dumps({'path': sys.path, 'names': names,"
        " 'transformers': transformers.__version__}))\n"
    )
    seen = in_converter(converter_python, probe)
    venv_dir, src = str(repo / ".venv"), str(repo / "src")
    assert [p for p in seen["path"] if p.startswith((venv_dir, src))] == []
    assert {"pytest", "ruff", "ty", "clarity-chat"} & set(seen["names"]) == set()
    pinned = server.llama_cpp_dir / "requirements" / "requirements-convert_legacy_llama.txt"
    assert f"transformers=={seen['transformers']}" in pinned.read_text("utf-8"), seen


def requirements_of(path: Path) -> list[str]:
    """The requirement lines of a pip requirements file, following its `-r` lines (relative
    to the file's folder) and leaving out comments and option lines such as
    --extra-index-url: what it asks for, not where from."""
    lines = []
    for line in path.read_text("utf-8").splitlines():
        line = line.split(" #")[0].strip()
        if line.startswith("-r "):
            lines += requirements_of(path.parent / line.removeprefix("-r ").strip())
        elif line and not line.startswith(("#", "-")):
            lines.append(line)
    return lines


def test_requirements_of_follows_r_lines_and_leaves_out_options(tmp_path):
    (tmp_path / "reqs").mkdir()
    (tmp_path / "reqs" / "base.txt").write_text(
        'numpy~=2.2.6; python_version < "3.14"\n\n# a comment\ngguf>=0.1.0  # trailing\n'
    )
    (tmp_path / "reqs" / "top.txt").write_text(
        "-r ./base.txt\n--extra-index-url https://example.org/whl\n## torch\ntorch==2.11.0\n"
    )
    assert requirements_of(tmp_path / "reqs" / "top.txt") == [
        'numpy~=2.2.6; python_version < "3.14"',
        "gguf>=0.1.0",
        "torch==2.11.0",
    ]


@pytest.mark.live
def test_the_converter_s_pins_meet_llama_cpp_s_convert_requirements(converter_python):
    """converter-requirements.txt pins what a conversion ran with; llama.cpp's own file at the
    pinned commit says what its converter needs. Each of its requirements that applies to
    the converter's Python must be met by the installed pin, so a new llama.cpp pin that
    moves a range fails here, not in a conversion. Checked in the environment with its own
    `packaging` (a transformers dependency): Clarity adds none."""
    repo = Path(__file__).resolve().parents[1]
    server = load_server_config(repo)
    wanted = requirements_of(
        server.llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    )
    check = (
        "import json, sys\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        "from packaging.requirements import Requirement\n"
        "checked, unmet = [], []\n"
        "for line in json.loads(sys.argv[1]):\n"
        "    wanted = Requirement(line)\n"
        "    if wanted.marker is not None and not wanted.marker.evaluate():\n"
        "        continue\n"
        "    try:\n"
        "        have = version(wanted.name)\n"
        "    except PackageNotFoundError:\n"
        "        have = None\n"
        "    checked.append(wanted.name)\n"
        "    if have is None or not wanted.specifier.contains(have, prereleases=True):\n"
        "        unmet.append([line, have])\n"
        "print(json.dumps({'checked': checked, 'unmet': unmet}))\n"
    )
    seen = in_converter(converter_python, check, json.dumps(wanted))
    assert seen["unmet"] == [], seen
    # numpy once (the other line is for Python 3.14), then the rest of what the files name
    assert sorted(seen["checked"]) == [
        "gguf",
        "numpy",
        "protobuf",
        "sentencepiece",
        "torch",
        "transformers",
    ], seen


def test_a_convert_fetch_refuses_a_llama_cpp_folder_whose_git_can_t_be_read(
    root, repo_around, monkeypatch
):
    """F3 (QA): fetch_convert's llama.cpp check walked up the same way. Real git."""
    server = load_server_config(root)
    (server.llama_cpp_dir / ".git").mkdir(parents=True)
    monkeypatch.setattr(models, "_download", lambda *args: pytest.fail("downloaded"))
    model = next(iter(load_models(root).values()))
    with pytest.raises(ConfigError) as refused:
        models.fetch_convert(root, model, server)
    assert repo_around[:12] not in str(refused.value)
    assert "not a git checkout of its own" in str(refused.value)
