"""A model file's three facts: the architecture, width and block count steering checks a
vector against on llama.cpp, read from the GGUF's key-value section (v0.4.7 B1).

Every file here is made up: no model file is tracked. The real-file evidence is G7's table
in the v0.4.7 plan and the `-m live` test on SmolLM2 in test_live.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clarity_chat import steering
from clarity_chat.profiles import ConfigError
from tests.test_steering import gguf_bytes, text, u32, u64, write


def facts(path):
    """What `model_file` returns or raises, so each test fails on its own assertion. Looked
    up by name, so the tests type-check without it."""
    try:
        return getattr(steering, "model_file")(path)  # noqa: B009 - type-checks before it exists
    except Exception as e:
        return e


def refusal(path) -> str:
    got = facts(path)
    return str(got) if isinstance(got, ConfigError) else f"not refused: {got!r}"


def model(path, arch="llama", width=576, blocks=30, before=()):
    """A model-like file as llama.cpp's converter writes one: general.architecture first,
    then the sizes. `before` adds key-values ahead of the sizes, each a writer method and
    its arguments; a None width or block count leaves that key out."""
    import gguf

    writer = gguf.GGUFWriter(path, arch)
    writer.add_string("general.name", "made up")
    for method, *args in before:
        getattr(writer, method)(*args)
    if width is not None:
        writer.add_embedding_length(width)
    if blocks is not None:
        writer.add_block_count(blocks)
    writer.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def sentence(path, reason):
    return f"{path} isn't a model file Clarity can read: {reason}."


@pytest.mark.parametrize(
    ("arch", "width", "blocks"),
    [("llama", 576, 30), ("olmo2", 4096, 32), ("gemma4", 2560, 42)],  # G7's three kinds
)
def test_each_fact_is_read_back(tmp_path, arch, width, blocks):
    path = model(tmp_path / "m.gguf", arch, width, blocks)
    assert facts(path) == (arch, width, blocks)


def test_the_facts_have_names(tmp_path):
    got = facts(model(tmp_path / "m.gguf"))
    assert (getattr(got, "arch", None), getattr(got, "width", None)) == ("llama", 576)
    assert getattr(got, "blocks", None) == 30


def test_reading_stops_at_the_third_key(tmp_path):
    """What follows the sizes isn't read: here a key of a type GGUF doesn't have, then
    nothing, as if the file were cut off."""
    kvs = [
        ("general.architecture", u32(8) + text("llama")),
        ("llama.embedding_length", u32(4) + u32(576)),
        ("llama.block_count", u32(4) + u32(30)),
        ("general.broken", u32(99)),
    ]
    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes(kvs, [], counts=(1, 10)))
    assert facts(path) == ("llama", 576, 30)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"width": None}, "it has no llama.embedding_length"),
        ({"blocks": None}, "it has no llama.block_count"),
        ({"blocks": 0}, "llama.block_count is 0"),
        ({"width": 0}, "llama.embedding_length is 0"),
    ],
)
def test_a_missing_or_zero_size_is_refused(tmp_path, kwargs, reason):
    path = model(tmp_path / "m.gguf", **kwargs)
    assert refusal(path) == sentence(path, reason)


@pytest.mark.parametrize(
    ("kvs", "reason"),
    [
        ([("general.name", u32(8) + text("made up"))], "it has no general.architecture"),
        ([("general.architecture", u32(4) + u32(1))], "general.architecture isn't a string"),
        (
            [
                ("general.architecture", u32(8) + text("llama")),
                ("llama.embedding_length", u32(10) + u64(576)),
            ],
            "llama.embedding_length isn't a u32",
        ),
        (
            [
                ("general.architecture", u32(8) + text("llama")),
                ("llama.embedding_length", u32(4) + u32(576)),
                ("llama.block_count", u32(5) + u32(30)),
            ],
            "llama.block_count isn't a u32",
        ),
        (
            [
                ("general.architecture", u32(8) + text("llama")),
                ("general.architecture", u32(8) + text("olmo2")),
            ],
            "the key 'general.architecture' appears twice",
        ),
    ],
)
def test_a_missing_key_or_a_wrong_type_is_refused(tmp_path, kvs, reason):
    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes(kvs, []))
    assert refusal(path) == sentence(path, reason)


def test_every_truncation_is_refused(tmp_path):
    kvs = [
        ("general.architecture", u32(8) + text("llama")),
        ("general.name", u32(8) + text("made up")),
        ("llama.embedding_length", u32(4) + u32(576)),
        ("llama.block_count", u32(4) + u32(30)),
    ]
    data = gguf_bytes(kvs, [], alignment=1)  # ends where the block count does
    path = tmp_path / "m.gguf"
    path.write_bytes(data)
    assert facts(path) == ("llama", 576, 30)
    for size in range(len(data)):
        path.write_bytes(data[:size])
        got = refusal(path)
        assert got.startswith(sentence(path, "")[:-1]), (size, got)
        assert "too short" in got or "runs past the end of the file" in got, (size, got)


def test_sizes_after_the_first_megabyte_are_refused(tmp_path):
    """llama.cpp's converter writes the sizes before the tokenizer's lists, so G7 found them
    among the first 20 keys. A file that keeps them past 1 MB isn't read to the end."""
    tokens = [f"token {n:06}" for n in range(100_000)]  # about 2 MB
    path = model(tmp_path / "m.gguf", before=[("add_token_list", tokens)])
    got = refusal(path)
    assert "runs past the first 1 MB" in got, got
    assert path.stat().st_size > 1024 * 1024
    small = model(tmp_path / "s.gguf", before=[("add_token_list", tokens[:1000])])
    assert facts(small) == ("llama", 576, 30)


def test_a_control_vector_is_refused(tmp_path):
    path = write(tmp_path / "v.gguf", {1: np.zeros(8, dtype=np.float32)})
    assert refusal(path) == sentence(path, "it's a control vector, not a model")


def test_a_missing_file_and_a_file_that_isnt_gguf(tmp_path):
    missing = tmp_path / "gone.gguf"
    assert refusal(missing).startswith(f"{missing} isn't a model file Clarity can read: ")
    text_file = tmp_path / "notes.gguf"
    text_file.write_text("a made-up note, long enough to have a header's 24 bytes")
    assert refusal(text_file) == sentence(text_file, "it doesn't start with GGUF")
    version_2 = tmp_path / "v2.gguf"
    version_2.write_bytes(gguf_bytes([], [], version=u32(2)))
    assert refusal(version_2) == sentence(
        version_2, "GGUF version 2; Clarity reads version 3, little-endian"
    )


VALID = gguf_bytes(
    [
        ("general.architecture", u32(8) + text("llama")),
        ("general.name", u32(8) + text("made up")),
        ("tokenizer.ggml.scores", u32(9) + u32(6) + u64(2) + b"\0" * 8),
        ("llama.embedding_length", u32(4) + u32(576)),
        ("llama.block_count", u32(4) + u32(30)),
    ],
    [],
)


@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    st.one_of(
        st.binary(max_size=120),
        st.binary(max_size=120).map(lambda tail: b"GGUF" + u32(3) + tail),
        st.tuples(st.integers(0, len(VALID) - 1), st.integers(0, 255)).map(
            lambda flip: VALID[: flip[0]] + bytes([flip[1]]) + VALID[flip[0] + 1 :]
        ),
    )
)
def test_any_bytes_are_read_or_refused(tmp_path, data):
    path = tmp_path / "m.gguf"
    path.write_bytes(data)
    got = facts(path)
    assert isinstance(got, (tuple, ConfigError)), repr(got)
