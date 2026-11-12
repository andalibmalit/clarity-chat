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
