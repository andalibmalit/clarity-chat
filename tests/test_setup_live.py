"""v0.4.2 A2's live tier (`-m setup`): setup's own functions fetch llama.cpp at the pin
from GitHub into a temporary folder (the source step only: no models, no launcher). Needs
network to GitHub; nothing here touches the checkout's vendor/llama.cpp. The build joins
this tier with A2's build step (Q2, Q3a).

    uv run pytest -m setup -v -s
"""

from __future__ import annotations

import shutil

import pytest

from clarity_chat.logs import git_state
from clarity_chat.profiles import load_server_config
from clarity_chat.setup import Setup, run_logged, this_mac
from tests.conftest import REPO

pytestmark = pytest.mark.setup


def test_setup_fetches_llama_cpp_at_the_pin_into_a_new_folder(tmp_path):
    """The source step alone, through setup's own functions: the Mac's checks, the
    never-touch check, then the .partial fetch and rename. Not go(): since A3 it fetches the
    models (bge-m3: 2.27 GB) and writes the launcher on the real Desktop. Without network
    the fetch is refused (a ConfigError), so the test fails; it never skips."""
    for name in ("defaults.toml", "server.toml"):
        shutil.copy(REPO / name, tmp_path / name)
    said: list[str] = []
    setup = Setup(
        tmp_path, this_mac(tmp_path), run_logged, lambda q: "n", said.append, home=tmp_path
    )
    setup.check_mac()
    assert setup.check_source() is False  # nothing there yet
    setup.fetch_source()
    folder = load_server_config(tmp_path).llama_cpp_dir
    assert folder.is_relative_to(tmp_path)
    assert git_state(folder) == {"commit": setup.pin, "dirty": False}
    assert not (folder.parent / "llama.cpp.partial").exists()
    print("\n" + "\n".join(said))
