"""The files Clarity ships: each loads through the loader Clarity uses, and none holds a
placeholder. Twice a shipped file broke only when someone first used it (v0.2.0's
models.toml, then the placeholder prompts)."""

from __future__ import annotations

import pytest

from clarity_chat.models import read_lock
from clarity_chat.profiles import load_defaults, load_models, load_profile, load_server_config
from tests.conftest import REPO

PROFILES = sorted((REPO / "profiles").glob("*.toml"))
# Every file Clarity reads from the checkout, other than code.
SHIPPED = (
    [REPO / name for name in ("defaults.toml", "models.toml", "models.lock", "server.toml")]
    + PROFILES
    + sorted((REPO / "prompts").glob("*"))
)


def test_the_config_files_load():
    load_defaults(REPO)
    assert load_models(REPO)
    load_server_config(REPO)
    read_lock(REPO)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_every_shipped_profile_loads(path):
    load_profile(REPO, path.stem, load_defaults(REPO))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_shipped_file_holds_a_placeholder(path):
    text = path.read_text(encoding="utf-8").lower()
    for word in ("placeholder", "todo", "fixme", "lorem ipsum"):
        assert word not in text, f"{path.name} contains {word!r}"


def test_the_tests_of_shipped_files_never_read_your_local_models(monkeypatch, tmp_path):
    """The whole-branch review's should 2: since B4, load_models and read_lock merge
    local/models.toml and local/models.lock, which are the owner's own files. The offline
    tests that read the repo's models read only the shipped ones. Here local/models.toml
    seems to exist, and reading it fails the test."""
    from pathlib import Path

    from clarity_chat import models, profiles
    from tests import test_models, test_recommended

    planted = REPO / "local" / "models.toml"
    is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: p == planted or is_file(p))
    read_toml = profiles.read_toml

    def read(path):
        assert Path(path) != planted, f"read the owner's {planted}"
        return read_toml(path)

    monkeypatch.setattr(profiles, "read_toml", read)
    monkeypatch.setattr(models, "read_toml", read)
    test_the_config_files_load()
    test_models.test_the_shipped_models_toml_loads()
    test_recommended.test_both_llama_profiles_take_meta_s_values_from_the_committed_lock()
    test_models.test_the_converter_runs_isolated_from_clarity_s_environment(tmp_path, monkeypatch)
