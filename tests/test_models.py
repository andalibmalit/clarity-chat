"""Fetch helpers that don't need the network."""

from __future__ import annotations

import hashlib
import json
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
    (root / "models.toml").write_text(
        CONVERT_MODEL + 'weights = "pytorch_model.bin"\n', encoding="utf-8"
    )
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


def test_the_converter_runs_isolated_from_clarity_s_environment(tmp_path, monkeypatch):
    """The CI plan's follow-up: from Clarity's folder, `uv run --no-project --python 3.13`
    took Clarity's .venv as the overlay's base, so its site-packages and src were on the
    converter's sys.path despite -I (a probe showed both). --isolated gives the overlay a
    fresh base: only llama.cpp's pinned packages, as models.py's comment promises."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    server = load_server_config(repo)
    model = read_models(repo / "models.toml")["smollm2-135m-instruct"]  # shipped only
    monkeypatch.setattr(models, "git_state", lambda path: {"commit": server.llama_cpp_commit})
    monkeypatch.setattr(models, "_download", lambda repo, revision, patterns: tmp_path)
    started: list[list[str]] = []

    class Stop(Exception):
        pass

    def popen(command, **kwargs):
        started.append(command)
        raise Stop

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(Stop):
        models.fetch_convert(tmp_path, model, server)
    command = started[0]
    run = command[: command.index("python")]  # uv's own options, before the interpreter
    assert run[:2] == ["uv", "run"] and {"--no-project", "--isolated"} <= set(run), run
    assert command[command.index("python") + 1] == "-I"


@pytest.mark.live
def test_the_converter_sees_nothing_of_claritys_environment_and_imports_its_pinned_transformers():
    """The probe behind the isolation fix, through the command fetch_convert runs, from
    Clarity's folder (where uv would find Clarity's .venv): no .venv or src on sys.path, none
    of Clarity's dev tools installed, and llama.cpp's pinned transformers imported. Live:
    it needs the llama.cpp checkout and the pinned packages (uv's cache, or the network)."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    server = load_server_config(repo)
    probe = (
        "import json, sys, transformers\n"
        "from importlib.metadata import distributions\n"
        "names = sorted({d.metadata['Name'].lower() for d in distributions()})\n"
        "print(json.dumps({'path': sys.path, 'names': names,"
        " 'transformers': transformers.__version__}))\n"
    )
    done = subprocess.run(
        [*models.converter_python(server), "-c", probe],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    venv, src = str(repo / ".venv"), str(repo / "src")
    assert [p for p in seen["path"] if p.startswith((venv, src))] == []
    assert {"pytest", "ruff", "ty", "clarity-chat"} & set(seen["names"]) == set()
    pinned = server.llama_cpp_dir / "requirements" / "requirements-convert_legacy_llama.txt"
    assert f"transformers=={seen['transformers']}" in pinned.read_text("utf-8"), seen
