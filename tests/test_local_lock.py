"""v0.4.2 A10: every fetch writes local/models.lock (decision 16; Q11, yes). An entry there
is read first, then the shipped models.lock, which only the maintainer changes, with
scripts/refresh_lock.py. So no fetch changes a tracked file, and `client.dirty` stays true
only for a real edit. A scratch root and made-up entries only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from clarity_chat.files import sha256_file
from clarity_chat.models import lock_entry, read_lock, verify, write_lock_entry
from clarity_chat.profiles import ConfigError, load_models
from tests.conftest import REPO

SCRIPT = REPO / "scripts" / "refresh_lock.py"
LOCAL_LOCK = Path("local") / "models.lock"


def shipped(root: Path) -> dict:
    """The shipped lock; none in a root nothing was ever shipped to (`fetched_root`)."""
    path = root / "models.lock"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"models": {}}


def local(root: Path) -> dict:
    path = root / LOCAL_LOCK
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"models": {}}


def local_tiny(root: Path) -> dict:
    """tiny's entry in local/models.lock, where the fixture's fetch put it."""
    entries = local(root)["models"]
    assert "tiny" in entries, "tiny isn't in local/models.lock"
    return entries["tiny"]


def git(folder: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", "-C", str(folder), "-c", "user.name=made-up", "-c", "user.email=m@example.com",
         *args], check=True, capture_output=True, text=True, env=env,
    ).stdout  # fmt: skip


def test_a_fetch_leaves_the_shipped_lock_and_the_checkout_clean(fetched_root):
    """Checks: a fetch leaves models.lock byte for byte, and `git status` clean, in a
    scratch git repo with Clarity's .gitignore."""
    root = fetched_root
    (root / "models.lock").write_text(
        json.dumps({"schema_version": 1, "models": {}}, indent=2) + "\n", encoding="utf-8"
    )
    (root / ".gitignore").write_text((REPO / ".gitignore").read_text(encoding="utf-8"))
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "made up")
    before = (root / "models.lock").read_bytes()
    entry = {"name": "tiny", "gguf": {"sha256": "1" * 64}}
    write_lock_entry(root, "tiny", entry)
    assert (root / "models.lock").read_bytes() == before
    assert git(root, "status", "--porcelain") == ""
    assert local_tiny(root) == entry


def test_fetching_a_shipped_model_writes_local_models_lock(fetched_root):
    """fake_fetch, like `clarity fetch`, records tiny (a shipped entry) through
    write_lock_entry: its entry is in local/models.lock, not models.lock."""
    assert "tiny" in local(fetched_root)["models"]
    assert "tiny" not in shipped(fetched_root)["models"]
    assert read_lock(fetched_root)["models"]["tiny"] == local_tiny(fetched_root)


def test_your_entry_wins_over_the_shipped_one_of_the_same_name(fetched_root):
    mine = local_tiny(fetched_root)
    theirs = {**mine, "gguf": {**mine["gguf"], "sha256": "9" * 64}}
    (fetched_root / "models.lock").write_text(
        json.dumps({"schema_version": 1, "models": {"tiny": theirs}}), encoding="utf-8"
    )
    assert read_lock(fetched_root)["models"]["tiny"] == mine


def test_without_local_models_lock_the_shipped_one_is_read(fetched_root):
    """CI's path: the fetch's local/models.lock removed, verify checks the committed entry,
    and a changed modified time is recorded in local/models.lock, not models.lock."""
    weights = fetched_root / local_tiny(fetched_root)["gguf"]["path"]
    entry = local_tiny(fetched_root)
    entry["gguf"]["sha256"] = sha256_file(weights)  # fake_fetch's is made up; verify hashes
    (fetched_root / LOCAL_LOCK).unlink()
    (fetched_root / "models.lock").write_text(
        json.dumps({"schema_version": 1, "models": {"tiny": entry}}), encoding="utf-8"
    )
    assert read_lock(fetched_root)["models"]["tiny"] == entry
    assert verify(fetched_root, "tiny") == (
        True,
        f"tiny: sha256 matches ({entry['gguf']['sha256'][:16]}…)",
    )
    before = (fetched_root / "models.lock").read_bytes()
    os.utime(weights, ns=(1, 1))
    assert verify(fetched_root, "tiny")[0]
    assert (fetched_root / "models.lock").read_bytes() == before
    assert local_tiny(fetched_root)["gguf"]["mtime_ns"] == 1


def test_an_entry_for_no_model_is_not_read(fetched_root):
    """A local entry whose name is in neither models.toml (one since removed, say) isn't a
    fetched model: `clarity verify` with no names doesn't list it."""
    entry = local_tiny(fetched_root)
    write_lock_entry(fetched_root, "gone", {**entry, "name": "gone"})
    assert set(read_lock(fetched_root)["models"]) == {"tiny"}


def test_a_sentence_about_an_entry_names_the_lock_it_was_read_from(fetched_root):
    model = load_models(fetched_root)["tiny"]
    entry = local_tiny(fetched_root)
    write_lock_entry(fetched_root, "tiny", {**entry, "revision": "f" * 40})
    with pytest.raises(ConfigError, match=r"^local/models\.lock has 'tiny' with revision"):
        lock_entry(fetched_root, model)


def refresh():
    assert SCRIPT.is_file(), "no scripts/refresh_lock.py"
    spec = importlib.util.spec_from_file_location("refresh_lock", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_refresh_script_copies_named_entries_to_the_shipped_lock(fetched_root, capsys):
    root = fetched_root
    keep = {"name": "zzz", "gguf": {"sha256": "2" * 64}}
    (root / "models.lock").write_text(
        json.dumps({"schema_version": 1, "models": {"zzz": keep}}), encoding="utf-8"
    )
    assert refresh().main(["--root", str(root), "tiny"]) == 0
    lock = shipped(root)
    assert lock["models"] == {"zzz": keep, "tiny": local_tiny(root)}
    assert list(lock["models"]) == ["tiny", "zzz"]  # sorted, as write_lock_entry sorts
    assert "models.lock: tiny copied from local/models.lock" in capsys.readouterr().out


def test_the_refresh_script_refuses_a_name_it_can_t_copy(fetched_root, capsys):
    """Not in local/models.lock (not fetched here), or not a shipped model (yours go in
    local/models.toml and never in the shipped lock): refused, and nothing written."""
    (fetched_root / "models.lock").write_text(
        json.dumps({"schema_version": 1, "models": {}}), encoding="utf-8"
    )
    before = (fetched_root / "models.lock").read_bytes()
    module = refresh()
    assert module.main(["--root", str(fetched_root), "nope"]) == 1
    assert "nope isn't in local/models.lock" in capsys.readouterr().err
    entry = local_tiny(fetched_root)
    write_lock_entry(fetched_root, "gone", {**entry, "name": "gone"})
    assert module.main(["--root", str(fetched_root), "tiny", "gone"]) == 1
    assert "gone isn't in models.toml" in capsys.readouterr().err
    assert (fetched_root / "models.lock").read_bytes() == before


def test_ci_verifies_the_linux_fetch_against_the_committed_lock():
    """CI's fetch writes local/models.lock with the Linux-made file's own hash. Verify must
    check the committed entry, so the step before it removes local/models.lock (where
    `git restore models.lock` did, before A10)."""
    workflow = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    steps = workflow.split("- run: uv run clarity fetch smollm2-135m-instruct\n", 1)[1]
    between = steps.split("- run: uv run clarity verify smollm2-135m-instruct", 1)[0]
    assert "rm local/models.lock" in between, between
