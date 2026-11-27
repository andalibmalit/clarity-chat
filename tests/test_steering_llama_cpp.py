"""A llama.cpp profile may steer (v0.4.7 B2): `prepare` checks its vector against the model
file's three facts, then against decision 4's three llama.cpp rules, before any model starts.

Offline, with made-up model files and the fake backend.
"""

from __future__ import annotations

import pytest

from clarity_chat.profiles import ConfigError
from tests.test_model_file import model
from tests.test_steering import directions, sha, steer_root, steered, write

VECTORS = "local/vectors"


@pytest.fixture
def session(root, tiny_tokenizer):
    """A project whose tiny model's GGUF is a made-up llama file of width 8 and 4 blocks, and
    whose vector, local/vectors/calm.gguf, has blocks 1 and 2."""
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    steer_root(root)
    model_like(root)
    backend = FakeBackend([" Hi."] * 20, tokenizer=tiny_tokenizer)
    backend.name = "llama.cpp"
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)


def model_like(root, arch="llama", width=8, blocks=4):
    """Write a model-like file over the tiny model's GGUF, and record it in the lock."""
    from clarity_chat.models import read_lock, write_lock_entry

    entry = read_lock(root)["models"]["tiny"]
    path = model(root / entry["gguf"]["path"], arch, width, blocks)
    stat = path.stat()
    gguf = {**entry["gguf"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    write_lock_entry(root, "tiny", {**entry, "gguf": gguf})


def vector(root, name="v.gguf", blocks=(1, 2), width=8, model_hint="llama"):
    """A vector in local/vectors/, and the [steering] table that names it."""
    path = write(root / VECTORS / name, directions(*blocks, width=width), model_hint)
    file = f"{VECTORS}/{name}".replace('"', '\\"')
    return path, f'file = "{file}"\nsha256 = "{sha(path.read_bytes())}"'


def layers(session, name):
    """The blocks a profile steers once prepared, or why it was refused."""
    try:
        return session.prepare(name).profile.layers
    except ConfigError as e:
        return f"refused: {e}"


def refusal(session, name):
    try:
        return f"prepared, steering {session.prepare(name).profile.layers}"
    except ConfigError as e:
        return str(e)


def test_a_steered_llama_cpp_profile_prepares_and_is_listed_as_steered(session):
    from clarity_chat.commands import Commands

    root = session.root
    assert layers(session, steered(root, backend="llama.cpp")) == [1, 2]
    listed = Commands(session).run("profiles", "", lambda e: None)["profiles"]
    row = next(p for p in listed if p["name"] == "steered")
    assert (row["backend"], row["steered"], row["error"]) == ("llama.cpp", True, None)


def test_a_gap_the_file_fills_is_refused_and_one_it_doesnt_is_accepted(session):
    root = session.root
    _, table = vector(root, "three.gguf", blocks=(1, 2, 3))
    name = steered(root, "gap", table + "\nlayers = [1, 3]", backend="llama.cpp")
    assert refusal(session, name) == (
        "On llama.cpp, layers must take in every block of local/vectors/three.gguf between "
        "its first and last: llama-server adds a vector to one unbroken range. layers is "
        "[1, 3], and the file also has [2]."
    )
    _, table = vector(root, "ends.gguf", blocks=(1, 3))
    name = steered(root, "no-gap", table + "\nlayers = [1, 3]", backend="llama.cpp")
    assert layers(session, name) == [1, 3]


def test_the_same_gap_on_transformers_prepares_as_before(root, tiny_tokenizer):
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    steer_root(root)
    backend = FakeBackend([" Hi."], tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    _, table = vector(root, "three.gguf", blocks=(1, 2, 3))
    assert layers(session, steered(root, "gap", table + "\nlayers = [1, 3]")) == [1, 3]


@pytest.mark.parametrize("character", [":", ",", '"'])
def test_each_character_llama_server_cant_take_is_refused(session, character):
    root = session.root
    path, table = vector(root, f"calm{character}1.gguf")
    name = steered(root, steering=table, backend="llama.cpp")
    assert refusal(session, name) == (
        "llama-server can't take a vector whose path has a colon, a comma or a double quote: "
        f"{path}. Rename the file, or move the Clarity folder."
    )


def test_an_architecture_steering_hasnt_been_checked_on_is_refused(session):
    model_like(session.root, arch="qwen2")
    assert refusal(session, steered(session.root, backend="llama.cpp")) == (
        "Steering hasn't been checked on llama.cpp architecture 'qwen2'; it runs on llama, "
        "olmo2 and gemma4. Ask for it to be added."
    )


@pytest.mark.parametrize(
    ("file", "kwargs", "message"),
    [
        ({}, {"model_hint": "olmo3"}, "v.gguf was made for model type 'olmo3', not 'llama'."),
        ({}, {"width": 16}, "v.gguf has width 16; this model's hidden size is 8."),
        ({}, {"blocks": (0, 1)}, "Block 0 can't be steered: llama.cpp's control-vector loader"),
        ({"blocks": 2}, {}, "Block 2 doesn't exist: this model has 2 blocks"),
    ],
)
def test_the_vector_is_checked_against_the_model_file(session, file, kwargs, message):
    root = session.root
    model_like(root, **file)
    _, table = vector(root, **kwargs)
    assert message in refusal(session, steered(root, steering=table, backend="llama.cpp"))


def test_a_scale_whose_product_overflows_float32_is_refused(session):
    """bf16 and float32 overflow near the same value; llama.cpp adds in float32."""
    import re

    got = refusal(session, steered(session.root, scale="3e38", backend="llama.cpp"))
    assert re.fullmatch(r"3e\+38 x direction\.[12] isn't finite in float32\.", got), got
    fine = steered(session.root, "fine", scale="3e30", backend="llama.cpp")
    assert layers(session, fine) == [1, 2]


@pytest.mark.parametrize(
    ("arch", "blocks"), [("llama", (1, 2)), ("olmo2", (1, 2)), ("gemma4", (1, 3))]
)
def test_a_vector_without_a_model_hint_is_checked_on_width_and_blocks_only(session, arch, blocks):
    root = session.root
    model_like(root, arch=arch)
    _, table = vector(root, blocks=blocks, model_hint=None)
    assert layers(session, steered(root, steering=table, backend="llama.cpp")) == list(blocks)


@pytest.mark.parametrize(("arch", "hint"), [("olmo2", "olmo3"), ("gemma4", "gemma4")])
def test_a_hint_is_checked_against_the_types_with_that_architecture(session, arch, hint):
    """OLMo 3 converts to llama.cpp's olmo2; Gemma 4 keeps its name."""
    root = session.root
    model_like(root, arch=arch)
    _, table = vector(root, model_hint=hint)
    assert layers(session, steered(root, steering=table, backend="llama.cpp")) == [1, 2]
    _, table = vector(root, "llama.gguf", model_hint="llama")
    got = refusal(session, steered(root, "other", table, backend="llama.cpp"))
    assert got.endswith(f"llama.gguf was made for model type 'llama', not '{hint}'."), got


def test_an_unreadable_model_file_is_refused_with_its_reason(session):
    from tests.conftest import fake_fetch

    weights = fake_fetch(session.root)  # "GGUF fake weights": 17 bytes
    assert refusal(session, steered(session.root, backend="llama.cpp")) == (
        f"{weights} isn't a model file Clarity can read: the file is too short."
    )
