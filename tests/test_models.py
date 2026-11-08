"""Fetch helpers that don't need the network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_chat import models
from clarity_chat.models import check_weights_quick, lock_entry, portable_path, weights_path
from clarity_chat.profiles import ConfigError, load_models, load_server_config


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


def test_portable_path_can_keep_a_symlinks_name(tmp_path):
    root = tmp_path / "repo"
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / "abc123").write_bytes(b"x")
    (root / "model.gguf").symlink_to(root / "blobs" / "abc123")
    assert portable_path(str(root / "model.gguf"), root, resolve=False) == "model.gguf"
    assert portable_path(str(root / "model.gguf"), root) == "blobs/abc123"
