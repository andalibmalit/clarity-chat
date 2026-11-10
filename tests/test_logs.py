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


def torn_log(tmp_path):
    """A log whose last line was cut off mid-write (a crash or a full disk)."""
    path = tmp_path / "c.jsonl"
    LogFile(path).append({"type": "note", "text": "kept"})
    with path.open("ab") as f:
        f.write(b'{"type": "call", "output_te')
    return path


def test_an_append_after_a_torn_line_ends_it_and_says_so(tmp_path):
    path = torn_log(tmp_path)
    before = path.read_bytes()
    seen = []
    log = LogFile(path)
    log.on_append = seen.append
    log.append({"type": "note", "text": "next"})
    assert path.read_bytes().startswith(before)  # nothing rewritten
    rows = list(read_records(path))
    assert [r["type"] for r in rows] == ["note", "torn_line", "note"]
    assert rows[1]["line"] == 2 and rows[1]["bytes"] == len(b'{"type": "call", "output_te')
    assert [r["type"] for r in seen] == ["torn_line", "note"]


def test_mend_ends_a_torn_line_without_another_record(tmp_path):
    path = torn_log(tmp_path)
    torn = LogFile(path).mend()
    assert torn is not None and torn["line"] == 2
    assert [r["type"] for r in read_records(path)] == ["note", "torn_line"]
    assert LogFile(path).mend() is None  # once is enough
    assert LogFile(tmp_path / "missing.jsonl").mend() is None
    assert not (tmp_path / "missing.jsonl").exists()


def test_a_malformed_line_no_torn_line_names_is_an_error(tmp_path):
    from clarity_chat.profiles import ConfigError

    path = torn_log(tmp_path)
    with pytest.raises(ConfigError, match="Line 2 of c.jsonl"):
        list(read_records(path))
    with path.open("ab") as f:  # ended, but not by Clarity: still unexplained
        f.write(b"\n")
    LogFile(path).append({"type": "note"})
    with pytest.raises(ConfigError, match="Line 2 of c.jsonl"):
        list(read_records(path))


def test_a_torn_line_that_parses_is_kept(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_bytes(b'{"type": "note", "text": "no line end"}')
    LogFile(path).append({"type": "note"})
    assert [r["type"] for r in read_records(path)] == ["note", "torn_line", "note"]


def test_a_line_cut_inside_a_character_is_torn_too(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_bytes('{"type": "note", "text": "é'.encode()[:-1])  # half of "é"
    LogFile(path).append({"type": "note"})
    assert [r["type"] for r in read_records(path)] == ["torn_line", "note"]


@pytest.mark.xfail(strict=True, reason="B5 review: a tear during the mending write")
def test_a_torn_line_whose_mending_write_is_torn_too_stays_readable(tmp_path):
    path = tmp_path / "c.jsonl"
    LogFile(path).append({"type": "note", "text": "kept"})
    with path.open("ab") as f:  # a record cut off, then the disk fills mid-mend
        f.write(b'{"type": "call", "output_te\n{"schema_version": 1, "type": "torn_li')
    LogFile(path).append({"type": "note", "text": "after"})
    assert [r["type"] for r in read_records(path)] == ["note", "torn_line", "note"]
