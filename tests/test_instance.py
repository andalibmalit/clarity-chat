"""The single-instance lock: one chat, run or replay per project at a time."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from clarity_chat import instance
from clarity_chat.profiles import ConfigError


def test_holding_the_lock_writes_who_holds_it_and_cleans_up(tmp_path):
    with instance.hold(tmp_path, "chat"):
        info = instance.read_info(tmp_path)
        assert info["pid"] == os.getpid() and info["command"] == "chat"
        if os.name != "nt":
            assert (tmp_path / "local" / "clarity.json").stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "local" / "clarity.json").exists()
    with instance.hold(tmp_path, "run"):  # released: it can be taken again
        pass


def test_a_second_holder_is_refused_with_who_holds_it(tmp_path):
    with instance.hold(tmp_path, "chat"):
        with pytest.raises(ConfigError, match=r"already running .*pid \d+, `clarity chat`"):
            with instance.hold(tmp_path, "replay"):
                pass


def test_the_lock_is_released_when_the_process_dies_without_cleaning_up(tmp_path):
    code = (
        "import os, sys; from pathlib import Path; from clarity_chat import instance\n"
        "with instance.hold(Path(sys.argv[1]), 'chat'):\n"
        "    os._exit(9)  # no finally, no atexit: like a crash\n"
    )
    assert subprocess.run([sys.executable, "-c", code, str(tmp_path)]).returncode == 9
    assert json.loads((tmp_path / "local" / "clarity.json").read_text())["command"] == "chat"
    with instance.hold(tmp_path, "chat"):  # the kernel released the lock
        pass
