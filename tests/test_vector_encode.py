"""V4: the control-vector writer, against gguf's reference writer, and the bytes form of the
atomic file writer."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from clarity_chat.files import write_bytes_atomic, write_text_atomic
from clarity_chat.steering import check, encode
from tests.test_steering import CONFIG, directions, load_bytes, write


@pytest.mark.parametrize(
    ("width", "blocks", "hint"),
    [(8, (1, 3), "llama"), (5, (2, 3, 5), "olmo3"), (576, tuple(range(1, 30)), "llama")],
)
def test_the_bytes_are_what_ggufs_writer_writes(tmp_path, width, blocks, hint):
    made = directions(*blocks, width=width)
    oracle = write(tmp_path / "w.gguf", made, model_hint=hint)
    assert encode(made, hint) == oracle.read_bytes()


def test_steering_reads_them_back_exactly_and_accepts_them(tmp_path):
    made = directions(1, 3)
    data = encode(made, "llama")
    assert data.startswith(b"GGUF\x03\0\0\0")  # version 3
    vector = load_bytes(tmp_path, data)
    assert vector.model_hint == "llama" and sorted(vector.directions) == [1, 3]
    for n, values in made.items():
        assert np.array_equal(vector.directions[n], values)
    assert check(vector, CONFIG) == [1, 3]


def test_the_bytes_writer_writes_them_exactly(tmp_path):
    path = tmp_path / "vectors" / "v.gguf"
    write_bytes_atomic(path, b"GGUF\0\r\n\xff", private=True)
    assert path.is_file()
    assert path.read_bytes() == b"GGUF\0\r\n\xff"
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("writer", ["bytes", "text"])
def test_an_interrupted_write_leaves_neither_file(tmp_path, monkeypatch, writer):
    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    path = tmp_path / "v.gguf"
    with pytest.raises(KeyboardInterrupt):
        if writer == "bytes":
            write_bytes_atomic(path, b"GGUF")
        else:
            write_text_atomic(path, "text")
    assert list(tmp_path.iterdir()) == []
