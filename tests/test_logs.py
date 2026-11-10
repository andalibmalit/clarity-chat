from __future__ import annotations

import json
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clarity_chat.logs import SCHEMA_VERSION, LogFile, new_id, read_records

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)

# Characters that str.splitlines() treats as line breaks, but JSONL must not.
TRICKY = "".join(map(chr, [0x2028, 0x2029, 0x85, 0x1C, 0x1D, 0x1E, 0x0D, 0x0A, 0x0B, 0x0C]))


@settings(max_examples=60, deadline=None)
@given(payload=st.dictionaries(st.text(max_size=10), json_values, max_size=5), text=st.text())
def test_record_round_trip(tmp_path_factory, payload, text):
    path = tmp_path_factory.mktemp("logs") / "c.jsonl"
    log = LogFile(path)
    written = log.append({"type": "note", "payload": payload, "text": text + TRICKY})
    written2 = log.append({"type": "note", "text": TRICKY})
    assert list(read_records(path)) == [written, written2]


def test_records_carry_schema_version_and_time(tmp_path):
    record = LogFile(tmp_path / "c.jsonl").append({"type": "note"})
    assert record["schema_version"] == SCHEMA_VERSION
    assert "time" in record


def test_appending_never_rewrites_earlier_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    log = LogFile(path)
    log.append({"type": "note", "text": "first"})
    before = path.read_bytes()
    log.append({"type": "note", "text": "second"})
    assert path.read_bytes().startswith(before)


def test_one_record_per_line_even_with_unicode_line_separators(tmp_path):
    path = tmp_path / "c.jsonl"
    LogFile(path).append({"type": "note", "text": "a b"})
    raw = path.read_bytes()
    assert raw.count(b"\n") == 1
    assert json.loads(raw)["text"] == "a b"


def test_ids():
    assert new_id("call").startswith("call-")
    assert new_id("call") != new_id("call")
    conversation = new_id("conversation")
    assert conversation[:8].isdigit()


def test_machine_records_the_macos_build_and_chip_on_a_mac():
    import platform

    from clarity_chat.logs import machine

    record = machine()
    if platform.system() != "Darwin":
        assert record["macos"] is None
        return
    assert set(record["macos"]) == {"version", "build", "chip"}
    assert record["macos"]["version"]


@pytest.mark.skipif(os.name == "nt", reason="Windows only emulates O_APPEND")
def test_two_processes_appending_to_one_log_never_split_a_line(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "log.jsonl"
    big = "x" * 200_000  # a call record with long prompt and output IDs
    code = (
        "import sys; from pathlib import Path; from clarity_chat.logs import LogFile\n"
        "log = LogFile(Path(sys.argv[1]))\n"
        "for i in range(40): log.append({'type': 'mark', 'i': i})\n"
    )
    other = subprocess.Popen([sys.executable, "-c", code, str(path)])
    log = LogFile(path)
    for i in range(40):
        log.append({"type": "call", "i": i, "big": big})
    assert other.wait() == 0
    rows = list(read_records(path))
    assert sorted(r["type"] for r in rows) == ["call"] * 40 + ["mark"] * 40
