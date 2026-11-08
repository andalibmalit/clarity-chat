"""Fetch helpers that don't need the network."""

from __future__ import annotations

from pathlib import Path

from clarity_chat.models import portable_path


def test_portable_path_hides_the_home_folder_and_the_repo_location(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert portable_path(str(root / "models" / "m.gguf"), root) == "models/m.gguf"
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    assert portable_path(str(cache), root) == "~/.cache/huggingface/hub"
    assert portable_path("--outtype", root) == "--outtype"
    assert portable_path("bf16", root) == "bf16"
