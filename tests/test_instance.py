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


class FakeOrphan:
    """A process standing in for a leftover llama-server, and a /props that answers for it."""

    def __init__(self, model_path: str, key: str = "k") -> None:
        import http.server
        import threading

        self.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        answer = json.dumps({"model_path": model_path}).encode()

        class Props(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server's name)
                ok = self.headers.get("Authorization") == f"Bearer {key}"
                self.send_response(200 if ok else 401)
                self.end_headers()
                self.wfile.write(answer if ok else b"{}")

            def log_message(self, format, *args):  # noqa: A002 (the base class's name)
                pass

        self.http = http.server.HTTPServer(("127.0.0.1", 0), Props)
        self.port = self.http.server_address[1]
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def left_by_a_crash(self, root, key="k", model_path="/m/a.gguf"):
        (root / "local").mkdir(exist_ok=True)
        server = {"port": self.port, "pid": self.proc.pid, "api_key": key, "model_path": model_path}
        instance.write_info(root, {"pid": 1, "command": "chat", "server": server})

    def close(self):
        self.http.shutdown()
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def orphan():
    made = []

    def make(model_path="/m/a.gguf"):
        made.append(FakeOrphan(model_path))
        return made[-1]

    yield make
    for o in made:
        o.close()


def test_a_server_left_by_a_crash_is_stopped_once_its_port_proves_it(tmp_path, orphan):
    left = orphan()
    left.left_by_a_crash(tmp_path)
    with instance.hold(tmp_path, "chat") as held:
        assert held.orphan is not None and f"pid {left.proc.pid}" in held.orphan
        assert "server" not in instance.read_info(tmp_path)  # the new run's record is fresh
    assert left.proc.wait(timeout=10) != 0


@pytest.mark.parametrize(
    ("key", "model_path", "message"),
    [
        ("wrong", "/m/a.gguf", "left alone"),  # another key: not the server we recorded
        ("k", "/m/other.gguf", "left alone"),  # another model on that port
    ],
)
def test_a_port_that_doesnt_prove_it_is_ours_is_left_alone(
    tmp_path, orphan, key, model_path, message
):
    left = orphan()
    left.left_by_a_crash(tmp_path, key=key, model_path=model_path)
    with instance.hold(tmp_path, "chat") as held:
        assert held.orphan is not None and message in held.orphan
    assert left.proc.poll() is None  # still running: nothing was killed


def test_nothing_answering_kills_nothing(tmp_path, orphan):
    left = orphan()
    left.left_by_a_crash(tmp_path)
    left.http.shutdown()
    left.http.server_close()
    with instance.hold(tmp_path, "chat") as held:
        assert held.orphan is not None and "nothing answers" in held.orphan
    assert left.proc.poll() is None


def test_the_running_server_is_recorded_and_cleared(tmp_path):
    with instance.hold(tmp_path, "chat") as held:
        held.record_server({"port": 1, "pid": 2, "api_key": "k", "model_path": "/m"})
        assert instance.read_info(tmp_path)["server"]["port"] == 1
        held.record_server(None)
        assert instance.read_info(tmp_path)["server"] is None


@pytest.mark.skipif(os.name == "nt", reason="no SIGTERM handler semantics on Windows")
def test_sigterm_exits_through_the_normal_cleanup(tmp_path):
    import signal

    code = (
        "import sys, time; from pathlib import Path; from clarity_chat import instance\n"
        "with instance.hold(Path(sys.argv[1]), 'chat'):\n"
        "    print('held', flush=True); time.sleep(30)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None and proc.stdout.readline() == "held\n"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 128 + signal.SIGTERM
    assert not (tmp_path / "local" / "clarity.json").exists()  # the finally ran
