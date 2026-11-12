"""Steering: reading a control-vector GGUF, checking it against a model, and the hook."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from clarity_chat.backends import BackendError
from clarity_chat.profiles import ConfigError
from clarity_chat.steering import applied, check, load

FIXTURE = Path(__file__).parent / "fixtures" / "steering" / "smollm2-repeng.gguf"
FIXTURE_SHA256 = "ed3236f9f64dc4714b402b9a32f7e209471d3108033eff28dc4ad9230d1724f2"
WIDTH = 32
CONFIG = {"model_type": "llama", "hidden_size": WIDTH, "num_hidden_layers": 4}


def directions(*blocks, width=WIDTH, seed=0):
    rng = np.random.default_rng(seed)
    return {n: rng.standard_normal(width).astype(np.float32) for n in blocks}


def write(path, blocks, model_hint="llama", extra=()):
    """A vector written with gguf's reference writer, as repeng writes one. `extra` adds
    (method, key, value) key-values."""
    import gguf

    writer = gguf.GGUFWriter(path, "controlvector")
    if model_hint is not None:
        writer.add_string("controlvector.model_hint", model_hint)
    writer.add_uint32("controlvector.layer_count", len(blocks))
    for method, key, value in extra:
        getattr(writer, method)(key, value)
    for n, values in blocks.items():
        writer.add_tensor(f"direction.{n}", values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_bytes(tmp_path, data: bytes):
    path = tmp_path / "v.gguf"
    path.write_bytes(data)
    return load(path, sha(data))


# --- a hand-built GGUF, so each field can be broken on its own -----------------------------


def text(s: str | bytes) -> bytes:
    raw = s.encode() if isinstance(s, str) else s
    return struct.pack("<Q", len(raw)) + raw


def u32(n):
    return struct.pack("<I", n)


def u64(n):
    return struct.pack("<Q", n)


def gguf_bytes(
    kvs=None, tensors=None, *, magic=b"GGUF", version=b"\x03\0\0\0", counts=None, alignment=32
):
    """kvs: (key, encoded type and value); tensors: (name, dims, type, offset, data)."""
    if kvs is None:
        kvs = [
            ("general.architecture", u32(8) + text("controlvector")),
            ("controlvector.model_hint", u32(8) + text("llama")),
            ("controlvector.layer_count", u32(4) + u32(2)),
        ]
    if tensors is None:
        one, three = directions(1, 3, width=8).values()
        tensors = [
            ("direction.1", [8], 0, 0, one.tobytes()),
            ("direction.3", [8], 0, 32, three.tobytes()),
        ]
    tensor_count, kv_count = counts or (len(tensors), len(kvs))
    out = magic + version + u64(tensor_count) + u64(kv_count)
    for key, value in kvs:
        out += text(key) + value
    for name, dims, kind, offset, _ in tensors:
        out += (
            text(name) + u32(len(dims)) + b"".join(u64(d) for d in dims) + u32(kind) + u64(offset)
        )
    out += b"\0" * (-len(out) % alignment)
    data = b""
    for *_, offset, values in tensors:
        data += b"\0" * (offset - len(data)) + values
    return out + data


def test_the_hand_built_file_is_byte_for_byte_what_gguf_writes(tmp_path):
    """So the corrupted files below differ from a real one only where they're broken."""
    path = write(tmp_path / "w.gguf", directions(1, 3, width=8))
    assert gguf_bytes() == path.read_bytes()


# --- the reader --------------------------------------------------------------------------


def test_a_vector_written_by_gguf_loads_exactly(tmp_path):
    written = directions(2, 3, 5)
    path = write(tmp_path / "v.gguf", written)
    vector = load(path, sha(path.read_bytes()))
    assert sorted(vector.directions) == [2, 3, 5]
    for n, values in written.items():
        assert vector.directions[n].dtype == np.float32
        assert not vector.directions[n].flags.writeable  # the sha256 describes what's added
        assert np.array_equal(vector.directions[n], values)
    assert vector.model_hint == "llama" and vector.width == WIDTH
    assert vector.sha256 == sha(path.read_bytes())


def test_repengs_own_file_loads_with_the_blocks_its_readme_gives():
    vector = load(FIXTURE, FIXTURE_SHA256)
    assert sorted(vector.directions) == [10, 15, 20]
    assert vector.width == 576 and vector.model_hint == "llama"
    for values in vector.directions.values():
        assert abs(float(np.linalg.norm(values)) - 1) < 1e-5


def test_key_values_of_every_type_are_skipped(tmp_path):
    extra = [
        ("add_uint8", "x.u8", 1),
        ("add_int8", "x.i8", -1),
        ("add_uint16", "x.u16", 2),
        ("add_int16", "x.i16", -2),
        ("add_uint32", "x.u32", 3),
        ("add_int32", "x.i32", -3),
        ("add_float32", "x.f32", 0.5),
        ("add_bool", "x.bool", True),
        ("add_string", "x.str", "héllo"),
        ("add_uint64", "x.u64", 4),
        ("add_int64", "x.i64", -4),
        ("add_float64", "x.f64", 0.25),
        ("add_array", "x.ints", [1, 2, 3]),
        ("add_array", "x.strs", ["a", "bé", ""]),
        ("add_array", "x.empty", []),
    ]
    written = directions(1, 2)
    path = write(tmp_path / "v.gguf", written, extra=extra)
    vector = load(path, sha(path.read_bytes()))
    assert all(np.array_equal(vector.directions[n], written[n]) for n in written)


def test_a_file_without_a_model_hint_loads(tmp_path):
    path = write(tmp_path / "v.gguf", directions(1), model_hint=None)
    assert load(path, sha(path.read_bytes())).model_hint is None


def test_every_truncation_is_a_config_error(tmp_path):
    data = gguf_bytes()
    for size in range(len(data)):
        with pytest.raises(ConfigError, match="isn't a control-vector GGUF"):
            load_bytes(tmp_path, data[:size])


def kv(key, value):
    """The default key-values, with one more."""
    return [
        ("general.architecture", u32(8) + text("controlvector")),
        ("controlvector.model_hint", u32(8) + text("llama")),
        (key, value),
    ]


def tensor(name="direction.1", dims=(8,), kind=0, offset=0, values=None):
    return [
        (
            name,
            list(dims),
            kind,
            offset,
            np.ones(8, np.float32).tobytes() if values is None else values,
        )
    ]


BROKEN = {
    "bad magic": (gguf_bytes(magic=b"GGUG"), "doesn't start with GGUF"),
    "version 2": (gguf_bytes(version=u32(2)), "GGUF version 2;"),
    "big-endian": (gguf_bytes(version=struct.pack(">I", 3)), "GGUF version 50331648"),
    "no tensors": (gguf_bytes(tensors=[]), "it has 0 tensors"),
    "huge tensor count": (gguf_bytes(counts=(2**62, 3)), f"it has {2**62} tensors"),
    "huge key count": (gguf_bytes(counts=(2, 2**62)), f"it has {2**62} keys"),
    "a string past the end": (
        gguf_bytes(kvs=kv("x", u32(8) + u64(2**40) + b"abc")),
        "the value of x runs past the end of the file",
    ),
    "a key that isn't UTF-8": (gguf_bytes(kvs=kv(b"\xff\xfe", u32(4) + u32(1))), "isn't UTF-8"),
    "unknown value type": (gguf_bytes(kvs=kv("x", u32(13))), "'x' has unknown value type 13"),
    "nested array": (
        gguf_bytes(kvs=kv("x", u32(9) + u32(9) + u64(1) + u32(4) + u64(1) + u32(7))),
        "'x' is an array of arrays",
    ),
    "an array past the end": (
        gguf_bytes(kvs=kv("x", u32(9) + u32(12) + u64(2**50))),
        "the value of x runs past the end",
    ),
    "alignment 0": (gguf_bytes(kvs=kv("general.alignment", u32(4) + u32(0))), "alignment is 0;"),
    "alignment 3": (gguf_bytes(kvs=kv("general.alignment", u32(4) + u32(3))), "alignment is 3;"),
    "alignment not a u32": (
        gguf_bytes(kvs=kv("general.alignment", u32(10) + u64(32))),
        "general.alignment isn't a u32",
    ),
    "model hint not a string": (
        gguf_bytes(
            kvs=[
                ("general.architecture", u32(8) + text("controlvector")),
                ("controlvector.model_hint", u32(4) + u32(1)),
            ]
        ),
        "model_hint isn't a string",
    ),
    "a key twice": (
        gguf_bytes(kvs=kv("controlvector.model_hint", u32(8) + text("olmo3"))),
        "the key 'controlvector.model_hint' appears twice",
    ),
    "a tensor not named direction.N": (
        gguf_bytes(tensors=tensor("blk.1")),
        "'blk.1' isn't named direction.N",
    ),
    "a leading zero": (gguf_bytes(tensors=tensor("direction.01")), "'direction.01' isn't named"),
    "a tensor twice": (
        gguf_bytes(tensors=tensor() + tensor(offset=32)),
        "the tensor direction.1 appears twice",
    ),
    "2-D": (gguf_bytes(tensors=tensor(dims=(4, 2))), "direction.1 has 2 dimensions"),
    "F16": (gguf_bytes(tensors=tensor(kind=1)), "direction.1 has GGUF type 1"),
    "length 0": (gguf_bytes(tensors=tensor(dims=(0,))), "direction.1 has length 0"),
    "lengths differ": (
        gguf_bytes(
            tensors=tensor() + tensor("direction.2", dims=(4,), offset=32, values=b"\0" * 16)
        ),
        "direction.2 has length 4, the others 8",
    ),
    "an unaligned offset": (gguf_bytes(tensors=tensor(offset=4)), "offset isn't a multiple of 32"),
    "an offset past the end": (
        gguf_bytes(tensors=tensor(values=b"")),  # the data isn't there
        "direction.1's data runs past the end",
    ),
    "a block number past Python's digit limit": (
        gguf_bytes(tensors=tensor("direction.1" + "0" * 5000)),
        "isn't a control-vector GGUF (",  # the backstop: int() refuses it with ValueError
    ),
    "a NaN": (
        gguf_bytes(tensors=tensor(values=np.array([np.nan] + [0] * 7, np.float32).tobytes())),
        "direction.1 has a value that isn't finite",
    ),
}


@pytest.mark.parametrize("case", BROKEN)
def test_each_broken_field_gets_its_own_message(tmp_path, case):
    data, message = BROKEN[case]
    with pytest.raises(ConfigError, match="isn't a control-vector GGUF") as error:
        load_bytes(tmp_path, data)
    assert message in str(error.value)
    assert str(tmp_path / "v.gguf") in str(error.value)


def test_a_missing_file_a_large_file_and_the_pin(tmp_path):
    with pytest.raises(ConfigError, match="doesn't exist"):
        load(tmp_path / "gone.gguf", "0" * 64)
    big = tmp_path / "model.gguf"
    with big.open("wb") as f:
        f.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ConfigError, match="over 16 MB.*Is it a model file"):
        load(big, "0" * 64)
    path = write(tmp_path / "v.gguf", directions(1))
    actual = sha(path.read_bytes())
    with pytest.raises(ConfigError) as error:
        load(path, "0" * 64)
    assert actual in str(error.value) and "0" * 64 in str(error.value)
    with pytest.raises(ConfigError) as error:
        load(path, None)
    assert f'sha256 = "{actual}"' in str(error.value)


# --- the check against a model's config -------------------------------------------------


def vector(tmp_path, blocks=(1, 3), model_hint="llama", width=WIDTH):
    path = write(tmp_path / "v.gguf", directions(*blocks, width=width), model_hint=model_hint)
    return load(path, sha(path.read_bytes()))


def test_check_returns_the_blocks_to_steer(tmp_path):
    v = vector(tmp_path, blocks=(3, 1))
    assert check(v, CONFIG) == [1, 3]
    assert check(v, CONFIG, [3]) == [3]


@pytest.mark.parametrize(
    ("config", "kwargs", "layers", "message"),
    [
        ({"model_type": "qwen3"}, {}, None, "hasn't been checked on model type 'qwen3'"),
        ({}, {"model_hint": "olmo3"}, None, "was made for model type 'olmo3', not 'llama'"),
        ({}, {"width": 16}, None, "has width 16; this model's hidden size is 32"),
        ({}, {}, [0], "Block 0 can't be steered: llama.cpp's control-vector loader"),
        ({}, {}, [99], "Block 99 doesn't exist: this model has 4 blocks, and blocks 1 to 3"),
        ({}, {}, [-1], "Block -1 doesn't exist"),
        ({}, {}, [2], "Block 2 isn't in"),
        ({}, {}, [1, 1], "lists a block twice"),
        ({}, {}, [], "layers must be a list"),
        ({}, {}, "1", "layers must be a list"),
        ({}, {}, [True], "layers must be a list"),
        ({}, {}, [1.0], "layers must be a list"),
        # The order: the model type first, and the block range before the file's blocks.
        ({"model_type": "qwen3"}, {"width": 16}, [0], "model type 'qwen3'"),
        ({}, {}, [2, 0], "Block 0 can't be steered"),
    ],
)
def test_check_refuses(tmp_path, config, kwargs, layers, message):
    v = vector(tmp_path, **kwargs)
    with pytest.raises(ConfigError) as error:
        check(v, CONFIG | config, layers)
    assert message in str(error.value)


def test_a_file_with_direction_0_is_refused_by_default(tmp_path):
    with pytest.raises(ConfigError, match="Block 0 can't be steered"):
        check(vector(tmp_path, blocks=(0, 1)), CONFIG)


# --- the hook, on tiny models in bf16 on the CPU -----------------------------------------


def tiny(model_type):
    torch = pytest.importorskip("torch")
    import transformers

    names = {"llama": "Llama", "olmo3": "Olmo3"}[model_type]
    config = getattr(transformers, f"{names}Config")(
        vocab_size=64,
        hidden_size=WIDTH,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = getattr(transformers, f"{names}ForCausalLM")(config)
    return model.to(torch.bfloat16).eval(), torch


IDS = [[1, 5, 9, 13, 2]]


def forward(model, torch, **kwargs):
    with torch.no_grad():
        return model(torch.tensor(IDS), **kwargs)


@pytest.fixture(params=["llama", "olmo3"])
def model(request):
    return tiny(request.param)


def test_steering_adds_the_scaled_vector_to_one_blocks_output(tmp_path, model):
    model, torch = model
    v = vector(tmp_path, blocks=(2,), model_hint=model.config.model_type)
    plain = forward(model, torch, output_hidden_states=True)  # installs Transformers' hooks
    scale = 3.7
    with applied(model, v, scale) as runs:
        steered = forward(model, torch, output_hidden_states=True)
    assert runs == {2: 1}
    for before, after in zip(plain.hidden_states[:3], steered.hidden_states[:3], strict=True):
        assert torch.equal(before, after)  # the embeddings and blocks 0 and 1
    delta = (torch.tensor(scale, dtype=torch.float32) * torch.tensor(v.directions[2])).to(
        torch.bfloat16
    )
    assert torch.equal(steered.hidden_states[3], plain.hidden_states[3] + delta)
    other = torch.tensor(scale, dtype=torch.bfloat16) * torch.tensor(v.directions[2]).to(
        torch.bfloat16
    )
    assert not torch.equal(delta, other), "the rounding order makes a difference here"
    assert not torch.equal(steered.logits, plain.logits)


def test_the_hidden_states_are_steered_when_steering_comes_first(tmp_path):
    """Transformers installs its capture hooks the first time hidden states are asked for,
    which here is inside `applied`; the steering hook still runs first."""
    model, torch = tiny("llama")
    v = vector(tmp_path, blocks=(2,))
    with applied(model, v, 2.0):
        steered = forward(model, torch, output_hidden_states=True)
    plain = forward(model, torch, output_hidden_states=True)
    assert not torch.equal(steered.hidden_states[3], plain.hidden_states[3])
    assert torch.equal(steered.hidden_states[2], plain.hidden_states[2])


@pytest.mark.parametrize("ending", [None, ValueError, KeyboardInterrupt])
def test_the_model_is_unsteered_however_the_block_ends(tmp_path, model, ending):
    model, torch = model
    v = vector(tmp_path, model_hint=model.config.model_type)
    plain = forward(model, torch).logits
    try:
        with applied(model, v, 4.0):
            assert model.clarity_steering == {"sha256": v.sha256, "scale": 4.0, "layers": [1, 3]}
            if ending is not None:
                raise ending
    except (ValueError, KeyboardInterrupt):
        pass
    assert torch.equal(forward(model, torch).logits, plain)
    assert not hasattr(model, "clarity_steering")


def test_entering_twice_is_refused(tmp_path):
    model, torch = tiny("llama")
    v = vector(tmp_path)
    with applied(model, v, 1.0), pytest.raises(BackendError, match="already steered"):
        with applied(model, v, 1.0):
            pass


def test_scale_0_adds_nothing(tmp_path):
    model, torch = tiny("llama")
    v = vector(tmp_path)
    plain = forward(model, torch).logits
    with applied(model, v, 0.0) as runs:
        assert torch.equal(forward(model, torch).logits, plain)
        assert not hasattr(model, "clarity_steering")
    assert runs == {1: 0, 3: 0}


def test_applied_checks_the_model_itself(tmp_path):
    """A notebook could pass any model: a wrong width would broadcast or fail deep inside."""
    model, _ = tiny("llama")
    with (
        pytest.raises(ConfigError, match="hidden size is 32"),
        applied(model, vector(tmp_path, width=16), 1.0),
    ):
        pass
    assert not hasattr(model, "clarity_steering")


def test_a_scaled_vector_that_isnt_finite_is_refused(tmp_path):
    model, _ = tiny("llama")
    with pytest.raises(ConfigError, match="isn't finite"), applied(model, vector(tmp_path), 1e39):
        pass
    assert not hasattr(model, "clarity_steering")


def test_a_block_that_doesnt_return_a_tensor_is_refused(tmp_path):
    model, torch = tiny("llama")
    block = model.model.layers[1]
    original = block.forward
    block.forward = lambda *args, **kwargs: (original(*args, **kwargs),)
    with applied(model, vector(tmp_path), 1.0), pytest.raises(BackendError, match="returned tuple"):
        forward(model, torch)


# --- the profile table and the setting (B2) ---------------------------------------------


def steer_root(root):
    """A fetched project whose tiny model has Transformers weights (a config.json), and a
    vector for it in local/vectors/."""
    import json

    from clarity_chat.models import read_lock, write_lock_entry
    from tests.conftest import fake_fetch

    fake_fetch(root)
    weights = root / "models" / "tiny-hf"
    weights.mkdir(parents=True)
    config = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 4}
    (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
    entry = read_lock(root)["models"]["tiny"]
    write_lock_entry(root, "tiny", {**entry, "source_dir": "models/tiny-hf"})
    (root / "local" / "vectors").mkdir(parents=True)
    path = write(root / "local" / "vectors" / "calm.gguf", directions(1, 2, width=8))
    return sha(path.read_bytes())


def steered(root, name="steered", steering=None, scale="4.0", backend="transformers", extra=""):
    """A text profile on `backend` with a [steering] table (None: the fixture vector)."""
    from tests.conftest import TEXT_PROFILE

    if steering is None:
        digest = sha((root / "local" / "vectors" / "calm.gguf").read_bytes())
        steering = f'file = "local/vectors/calm.gguf"\nsha256 = "{digest}"'
    text = TEXT_PROFILE.replace('backend = "llama.cpp"', f'backend = "{backend}"')
    text = text.replace("[settings]", f"{extra}\n[settings]")
    if scale is not None:
        text = text.replace("[settings]", f"[settings]\nsteering_scale = {scale}")
    if steering != "":
        text += f"\n[steering]\n{steering}\n"
    (root / "profiles" / f"{name}.toml").write_text(text, encoding="utf-8", newline="\n")
    return name


@pytest.fixture
def session(root, tiny_tokenizer):
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    steer_root(root)
    backend = FakeBackend([" Hi."] * 20, tokenizer=tiny_tokenizer)
    backend.name = "transformers"
    return Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)


def test_the_table_loads_on_transformers_and_the_vector_with_the_session(session):
    from clarity_chat.profiles import load_defaults, load_profile

    root = session.root
    profile = load_profile(root, steered(root, extra=""), load_defaults(root))
    assert profile.steering is not None and profile.steering.file == "local/vectors/calm.gguf"
    assert profile.settings["steering_scale"].value == 4.0
    assert profile.settings["steering_scale"].origin == "profile"
    prepared = session.prepare("steered")
    assert prepared.profile.vector is not None and prepared.profile.layers == [1, 2]
    assert prepared.profile.vector.sha256 == profile.steering.sha256


def test_layers_picks_blocks_from_the_file(session):
    root = session.root
    digest = sha((root / "local" / "vectors" / "calm.gguf").read_bytes())
    table = f'file = "local/vectors/calm.gguf"\nsha256 = "{digest}"\nlayers = [2]'
    assert session.prepare(steered(root, steering=table)).profile.layers == [2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "llama.cpp"}, "steering runs on the Transformers backend for now"),
        ({"steering": 'sha256 = "00"'}, "[steering] needs file"),
        ({"steering": 'file = "local/vectors/calm.gguf"\nscale = 2'}, "unknown keys ['scale']"),
        ({"steering": "file = 3"}, "[steering] needs file"),
        ({"steering": 'file = "local/vectors/calm.gguf"\nsha256 = 5'}, "sha256 must be"),
        ({"steering": 'file = "/etc/hosts"\nsha256 = "00"'}, "outside the project folder"),
        ({"steering": 'file = "../calm.gguf"\nsha256 = "00"'}, "outside the project folder"),
        ({"scale": None}, "steering_scale is required in [settings] with [steering]"),
        ({"steering": ""}, "steering_scale is 4.0, but this profile has no [steering]"),
    ],
)
def test_the_profile_refuses(root, kwargs, message):
    from clarity_chat.profiles import load_defaults, load_profile

    steer_root(root)
    with pytest.raises(ConfigError) as error:
        load_profile(root, steered(root, **kwargs), load_defaults(root))
    assert message in str(error.value)


def test_a_steering_table_that_isnt_a_table_is_refused(root):
    from clarity_chat.profiles import load_defaults, load_profile
    from tests.conftest import TEXT_PROFILE

    text = TEXT_PROFILE.replace('backend = "llama.cpp"', 'backend = "transformers"')
    text = text.replace("[settings]", 'steering = "calm.gguf"\n[settings]\nsteering_scale = 1')
    (root / "profiles" / "p.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[steering\] must be a table"):
        load_profile(root, "p", load_defaults(root))


def test_an_explicit_0_is_a_baseline_and_the_default_is_0(root):
    from clarity_chat.profiles import load_defaults, load_profile

    steer_root(root)
    defaults = load_defaults(root)
    assert defaults["steering_scale"] == 0.0
    assert (
        load_profile(root, steered(root, scale="0"), defaults).settings["steering_scale"].value
        == 0.0
    )
    assert load_profile(root, "text", defaults).settings["steering_scale"].value == 0.0


@pytest.mark.parametrize("value", ["inf", "nan", '"4"', "true"])
def test_steering_scale_is_any_finite_number(root, value):
    from clarity_chat.profiles import load_defaults, load_profile

    steer_root(root)
    with pytest.raises(ConfigError, match="steering_scale must be a finite number"):
        load_profile(root, steered(root, scale=value), load_defaults(root))


def test_use_recommended_keeps_the_profiles_scale(session):
    root = session.root
    prepared = session.prepare(steered(root, scale="-2.5", extra="use_recommended = true"))
    assert prepared.profile.settings["steering_scale"].value == -2.5
    assert prepared.profile.settings["steering_scale"].origin == "profile"
    assert prepared.profile.settings["temperature"].origin == "profile"  # TEXT_PROFILE sets it
    assert prepared.profile.settings["top_p"].origin == "profile"


def test_set_steering_scale_is_a_chat_setting(session):
    from clarity_chat.logs import read_records

    session.new_conversation(steered(session.root))
    record = session.set_setting("steering_scale", "2")
    assert record["type"] == "settings" and record["chat_settings"] == {"steering_scale": 2.0}
    assert session.conversation is not None
    assert session.conversation.settings["steering_scale"].origin == "chat"
    with pytest.raises(ConfigError, match="steering_scale must be a finite number"):
        session.set_setting("steering_scale", "abc")
    assert any(r["type"] == "settings" for r in read_records(session.conversation.log.path))


def test_an_unsteered_profile_takes_only_0(session):
    hf = steered(session.root, name="plain", steering="", scale=None)
    session.new_conversation(hf)
    with pytest.raises(ConfigError, match="this profile has no \\[steering\\]"):
        session.set_setting("steering_scale", "2")
    session.set_setting("steering_scale", "0")


def change_vector(root):
    write(root / "local" / "vectors" / "calm.gguf", directions(1, 2, width=8, seed=9))


def test_a_changed_vector_is_refused_by_a_new_chat_and_profile(session):
    from clarity_chat.commands import Commands

    root = session.root
    steered(root)
    change_vector(root)
    with pytest.raises(ConfigError, match="not the pinned"):
        session.new_conversation("steered")
    with pytest.raises(ConfigError, match="not the pinned"):
        Commands(session).run("profile", "steered", lambda e: None)


def test_a_changed_vector_is_refused_by_resume(session):
    root = session.root
    session.new_conversation(steered(root))
    session.say("Hi")
    assert session.conversation is not None
    path = session.conversation.log.path
    change_vector(root)
    with pytest.raises(ConfigError, match="not the pinned"):
        session.resume(path)


def test_a_changed_vector_is_refused_by_a_batch_run(session):
    from clarity_chat.batch import check_run

    root = session.root
    steered(root)
    change_vector(root)
    with pytest.raises(ConfigError, match="not the pinned"):
        check_run(session, ["steered", "text"], ["Hi"], 1)


def test_resume_refuses_a_logged_scale_with_no_vector(session):
    """A log is untrusted: a /set steering_scale on an unsteered profile can't come back."""
    import json

    session.new_conversation(steered(session.root, name="plain", steering="", scale=None))
    session.say("Hi")
    assert session.conversation is not None
    path = session.conversation.log.path
    session.set_setting("steering_scale", "0")
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["value"], record["chat_settings"] = 3.0, {"steering_scale": 3.0}
    path.write_text("\n".join([*lines[:-1], json.dumps(record)]) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="this profile has no \\[steering\\]"):
        session.resume(path)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ('file = "local/vectors/calm.gguf"', 'add sha256 = "'),
        ('file = "local/vectors/gone.gguf"\nsha256 = "00"', "doesn't exist"),
        ('file = "local/vectors/calm.gguf"\nsha256 = "SHA"\nlayers = [0]', "Block 0 can't"),
        ('file = "local/vectors/calm.gguf"\nsha256 = "SHA"\nlayers = [99]', "Block 99 doesn't"),
        ('file = "local/vectors/calm.gguf"\nsha256 = "SHA"\nlayers = [3]', "Block 3 isn't in"),
        ('file = "local/vectors/calm.gguf"\nsha256 = "SHA"\nlayers = "2"', "layers must be"),
    ],
)
def test_prepare_refuses_before_the_first_message(session, table, message):
    root = session.root
    table = table.replace("SHA", sha((root / "local" / "vectors" / "calm.gguf").read_bytes()))
    with pytest.raises(ConfigError) as error:
        session.prepare(steered(root, steering=table))
    assert message in str(error.value)


def test_a_model_type_steering_hasnt_been_checked_on_is_refused(session):
    import json

    config = session.root / "models" / "tiny-hf" / "config.json"
    config.write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 4}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="model type 'qwen3'"):
        session.prepare(steered(session.root))


# --- the hook in the backend, the record and replay (B3) --------------------------------

SETTINGS = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 1,
    "max_tokens": 8,
    "samplers": ["temperature"],
    "logprobs": 0,
    "steering_scale": 0.0,
}


@pytest.fixture
def hf(tmp_path, tiny_tokenizer):
    """TransformersBackend on a tiny Llama in bf16 on the CPU, with the tiny tokenizer, and a
    vector for it on blocks 1 and 2."""
    torch = pytest.importorskip("torch")
    import transformers

    from clarity_chat.backends.transformers_hf import TransformersBackend
    from clarity_chat.steering import Steering

    config = transformers.LlamaConfig(
        vocab_size=258,
        hidden_size=WIDTH,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        eos_token_id=257,
        bos_token_id=256,
    )
    torch.manual_seed(0)
    backend = TransformersBackend("cpu", 128)
    backend.model = transformers.LlamaForCausalLM(config).to(torch.bfloat16).eval()
    backend.tokenizer = tiny_tokenizer.tok
    backend.eos_ids, backend.eos_from = [257], "generation_config.json"
    v = vector(tmp_path, blocks=(1, 2))
    return backend, Steering("local/vectors/v.gguf", v, 40.0, [1, 2])


PROMPT = [72, 105, 33, 10]


def test_a_steered_call_records_its_steering_and_the_check(hf):
    backend, steer = hf
    plain = backend.complete(PROMPT, SETTINGS, [])
    result = backend.complete(PROMPT, {**SETTINGS, "steering_scale": 40.0}, [], steering=steer)
    assert plain.request["steering"] is None and "steering" not in plain.checks
    assert result.request["steering"] == {
        "file": "local/vectors/v.gguf",
        "sha256": steer.vector.sha256,
        "scale": 40.0,
        "layers": [1, 2],
    }
    n = len(result.token_ids)
    assert result.checks["steering"] == {
        "ok": True,
        "model_passes": n,
        "forward_passes": {"1": n, "2": n},
    }
    assert result.token_ids != plain.token_ids
    assert not hasattr(backend.model, "clarity_steering")
    again = backend.complete(PROMPT, SETTINGS, [], request=result.request, steering=steer)
    assert again.token_ids == result.token_ids and again.request == result.request


def test_a_cancelled_call_leaves_the_model_unsteered_and_records_no_check(hf):
    import threading

    backend, steer = hf
    plain = backend.complete(PROMPT, SETTINGS, [])
    cancel = threading.Event()
    passes = []

    def third(module, args, output):  # a Ctrl-C mid-reply, after three forward passes
        passes.append(1)
        if len(passes) == 3:
            cancel.set()

    handle = backend.model.lm_head.register_forward_hook(third)
    try:
        result = backend.complete(PROMPT, SETTINGS, [], cancel=cancel, steering=steer)
    finally:
        handle.remove()
    assert result.stop_type == "cancelled" and result.token_ids  # mid-reply
    assert "steering" not in result.checks
    assert result.request["steering"] is not None
    assert not hasattr(backend.model, "clarity_steering")
    assert backend.complete(PROMPT, SETTINGS, []).token_ids == plain.token_ids


def test_a_failed_generate_leaves_the_model_unsteered(hf, monkeypatch):
    from clarity_chat.backends import BackendError

    backend, steer = hf
    plain = backend.complete(PROMPT, SETTINGS, [])
    original = backend.model.generate

    def fail(*args, **kwargs):
        assert backend.model.clarity_steering is not None
        raise RuntimeError("made-up failure")

    monkeypatch.setattr(backend.model, "generate", fail)
    with pytest.raises(BackendError, match="made-up failure"):
        backend.complete(PROMPT, SETTINGS, [], steering=steer)
    monkeypatch.setattr(backend.model, "generate", original)
    assert not hasattr(backend.model, "clarity_steering")
    assert backend.complete(PROMPT, SETTINGS, []).token_ids == plain.token_ids


def test_a_check_that_fails_is_flagged(hf, monkeypatch):
    """Say a later Transformers ran a block twice per token: the record says so."""
    import clarity_chat.steering as steering_module

    backend, steer = hf
    real = steering_module.applied

    def doubled(*args, **kwargs):
        context = real(*args, **kwargs)
        runs = context.__enter__()
        try:
            yield runs
        finally:
            runs[1] += 1
            context.__exit__(None, None, None)

    from contextlib import contextmanager

    monkeypatch.setattr(steering_module, "applied", contextmanager(doubled))
    result = backend.complete(PROMPT, SETTINGS, [], steering=steer)
    assert result.checks["steering"]["ok"] is False
    assert "check_failed:steering" in result.flags


def test_the_llama_cpp_backend_refuses_steering(root, hf):
    from clarity_chat.backends import BackendError
    from clarity_chat.backends.llama_cpp import LlamaCppBackend
    from clarity_chat.profiles import load_server_config

    _, steer = hf
    backend = LlamaCppBackend(load_server_config(root), root / "logs")
    with pytest.raises(BackendError, match="Transformers backend"):
        backend.complete(PROMPT, SETTINGS, [], steering=steer)


def test_the_session_sends_the_profiles_steering_and_logs_it(session):
    backend = session.backend
    session.new_conversation(steered(session.root))
    record = session.say("Hi").record
    sent = backend.calls[-1]["steering"]
    assert sent.file == "local/vectors/calm.gguf" and sent.scale == 4.0 and sent.layers == [1, 2]
    assert record["request"]["steering"] == sent.to_json()
    session.set_setting("steering_scale", "0")
    session.say("Again")
    assert backend.calls[-1]["steering"] is None
    session.set_setting("steering_scale", "-2")
    session.say("Once more")
    assert backend.calls[-1]["steering"].scale == -2.0
    session.new_conversation(steered(session.root, name="plain", steering="", scale=None))
    session.say("Hi")
    assert backend.calls[-1]["steering"] is None


def test_a_batch_run_steers_only_the_steered_profiles_calls(session, tmp_path):
    from clarity_chat.batch import run_batch

    root = session.root
    steered(root)
    steered(root, name="plain", steering="", scale=None)
    (root / "openings.txt").write_text("Hello.\n", encoding="utf-8")
    run_batch(
        session,
        "r",
        tmp_path / "run",
        root / "openings.txt",
        ["steered", "plain"],
        1,
        lambda line: None,
    )
    sent = {c["settings"]["steering_scale"]: c["steering"] for c in session.backend.calls}
    assert sent[4.0] is not None and sent[4.0].layers == [1, 2] and sent[0.0] is None


def steered_call(session):
    from clarity_chat.logs import read_records

    session.new_conversation(steered(session.root))
    record = session.say("Hi").record
    assert session.conversation is not None
    path = session.conversation.log.path
    return path, record, read_records


def test_a_steered_call_replays_steered(session):
    path, record, _ = steered_call(session)
    session.backend.replies.append(record["output_text"])
    replay = session.replay(path, record["call_id"])
    assert replay["identical"]
    sent = session.backend.calls[-1]["steering"]
    assert sent.to_json() == record["request"]["steering"]


def doctor(path, call_id, change):
    import json

    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        row = json.loads(line)
        if row.get("type") == "call" and row.get("call_id") == call_id:
            change(row)
        out.append(json.dumps(row))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_a_replay_whose_vector_changed_or_went_is_refused(session):
    path, record, _ = steered_call(session)
    change_vector(session.root)
    with pytest.raises(ConfigError, match="was steered with local/vectors/calm.gguf") as error:
        session.replay(path, record["call_id"])
    assert record["request"]["steering"]["sha256"] in str(error.value)
    (session.root / "local" / "vectors" / "calm.gguf").unlink()
    with pytest.raises(ConfigError, match="doesn't exist"):
        session.replay(path, record["call_id"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"scale": "4"}, "steering_scale must be a finite number"),
        ({"scale": float("inf")}, "steering_scale must be a finite number"),
        ({"layers": [0]}, "Block 0 can't be steered"),
        ({"layers": [999]}, "Block 999 doesn't exist"),
        ({"layers": [3]}, "Block 3 isn't in"),
        ({"layers": "1"}, "layers must be a list"),
        ({"file": "/etc/hosts"}, "outside the project folder"),
        ({"file": 5}, "logged steering"),
        ({"sha256": None}, "logged steering"),
        ({"extra": 1}, "logged steering"),
    ],
)
def test_a_doctored_logged_steering_is_refused(session, change, message):
    path, record, _ = steered_call(session)
    doctor(path, record["call_id"], lambda row: row["request"]["steering"].update(change))
    with pytest.raises(ConfigError) as error:
        session.replay(path, record["call_id"])
    assert message in str(error.value)


def test_logged_steering_that_isnt_a_table_is_refused(session):
    path, record, _ = steered_call(session)
    doctor(path, record["call_id"], lambda row: row["request"].update(steering=[1, 2]))
    with pytest.raises(ConfigError, match="logged steering"):
        session.replay(path, record["call_id"])


def test_logged_steering_on_a_llama_cpp_call_is_refused(session):
    path, record, _ = steered_call(session)
    doctor(path, record["call_id"], lambda row: row.update(backend={"backend": "llama.cpp"}))
    with pytest.raises(ConfigError, match="Transformers backend"):
        session.replay(path, record["call_id"])


# --- steering in the export and the recompute check (C1) --------------------------------


def read_export(path):
    import json

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line]
    return rows[0], rows[1:]


def test_the_export_carries_each_calls_steering_and_splits_the_conditions(session):
    from clarity_chat.export import export_log

    session.new_conversation(steered(session.root))
    first = session.say("Hi").record
    session.set_setting("steering_scale", "0")
    session.say("Again")
    session.set_setting("steering_scale", "-4")
    session.say("Once more")
    assert session.conversation is not None
    header, rows = read_export(export_log(session, session.conversation.log.path)["path"])
    logged = first["request"]["steering"]
    assert [row["steering"] for row in rows] == [logged, "none", {**logged, "scale": -4.0}]
    pinned = {k: logged[k] for k in ("sha256", "scale", "layers")}
    assert [g["conditions"]["steering"] for g in header["conditions"]] == [
        pinned,
        "none",
        {**pinned, "scale": -4.0},
    ]
    assert [g["call_ids"] for g in header["conditions"]] == [[r["call_id"]] for r in rows]
    assert "Some calls were steered; apply the same vector with" in header["note"]
    assert "`clarity_chat.steering.applied`" in header["note"]


def test_an_unsteered_export_says_none_and_no_steering_note(session):
    from clarity_chat.export import export_log

    session.new_conversation(steered(session.root, name="plain", steering="", scale=None))
    session.say("Hi")
    assert session.conversation is not None
    header, rows = read_export(export_log(session, session.conversation.log.path)["path"])
    assert [row["steering"] for row in rows] == ["none"]
    assert header["conditions"][0]["conditions"]["steering"] == "none"
    assert "steered" not in header["note"]


def recompute_export(tmp_path, steering):
    """A one-call export for the tiny Llama below, its conditions as export writes them."""
    import json

    from clarity_chat.backends.transformers_hf import torch_env
    from clarity_chat.logs import machine
    from clarity_chat.recompute import installed, torch_version

    entry = {"id": 3, "logprob": -2.0, "top": [[3, -2.0]]}
    row = {
        "call_id": "c1",
        "prompt_ids": [1, 2],
        "output_ids": [3, 4],
        "logprobs": {"view": "raw logits, before any sampler", "n": 1, "tokens": [entry] * 2},
    }
    conditions = {
        "backend": "transformers",
        "weights": {"repo": "org/tiny", "revision": "a" * 40},
        "device": "cpu",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "torch": torch_version(),
        "transformers": installed("transformers"),
        "end_tokens": [],
        "env": torch_env(),
        "deterministic_algorithms": False,
        "machine": machine(),
        "steering": steering,
    }
    header = {
        "type": "export",
        "format_version": 1,
        "calls": 1,
        "conditions": [{"conditions": conditions, "call_ids": ["c1"]}],
    }
    path = tmp_path / f"e{len(list(tmp_path.iterdir()))}.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_the_recompute_check_compares_steering_both_ways(tmp_path):
    from clarity_chat.recompute import check

    model, _ = tiny("llama")
    model.config.name_or_path = "org/tiny"
    model.generation_config.eos_token_id = []
    v = vector(tmp_path, blocks=(1, 2))
    pinned = {"sha256": v.sha256, "scale": 4.0, "layers": [1, 2]}
    steered_export = recompute_export(tmp_path, pinned)
    plain_export = recompute_export(tmp_path, "none")

    def steering_fact(export):
        report = check(export, model, revision="a" * 40)
        assert report.calls[0]["measured"] is not None  # measured each time
        fact = next(f for f in report.groups[0]["facts"] if f["fact"] == "steering")
        return fact, report

    fact, report = steering_fact(steered_export)
    assert fact["same"] is False and fact["here"] == "none" and "steering: " in str(report)
    assert steering_fact(plain_export)[0]["same"] is True
    with applied(model, v, 4):
        assert steering_fact(steered_export)[0]["same"] is True
        fact, _ = steering_fact(plain_export)
        assert fact["same"] is False and fact["here"] == pinned
    with applied(model, v, 2.0):
        assert steering_fact(steered_export)[0]["same"] is False


def test_an_export_from_before_steering_reads_as_unsteered(tmp_path):
    import json

    from clarity_chat.recompute import check

    model, _ = tiny("llama")
    path = recompute_export(tmp_path, "none")
    lines = path.read_text(encoding="utf-8").split("\n")
    header = json.loads(lines[0])
    del header["conditions"][0]["conditions"]["steering"]
    path.write_text("\n".join([json.dumps(header), *lines[1:]]), encoding="utf-8")
    fact = next(f for f in check(path, model).groups[0]["facts"] if f["fact"] == "steering")
    assert fact == {"fact": "steering", "export": "none", "here": "none", "same": True}


# --- on screen and in --json (C2) -------------------------------------------------------


def test_settings_show_the_scale_and_the_vector(session):
    from clarity_chat.cli import format_settings
    from clarity_chat.commands import settings_view

    session.new_conversation(steered(session.root))
    digest = sha((session.root / "local" / "vectors" / "calm.gguf").read_bytes())
    view = settings_view(session)
    assert view["steering"] == {
        "file": "local/vectors/calm.gguf",
        "sha256": digest,
        "layers": [1, 2],
    }
    text = format_settings(view)
    assert f"steering vector local/vectors/calm.gguf (sha256 {digest[:12]}…, blocks 1, 2)" in text
    scale = next(line for line in text.split("\n") if line.strip().startswith("steering_scale"))
    assert "4.0" in scale and scale.rstrip().endswith("profile")
    session.set_setting("steering_scale", "0")
    scale = next(
        line
        for line in format_settings(settings_view(session)).split("\n")
        if line.strip().startswith("steering_scale")
    )
    assert "0.0" in scale and scale.rstrip().endswith("chat")


def test_an_unsteered_profile_shows_no_vector(session):
    from clarity_chat.cli import format_settings
    from clarity_chat.commands import settings_view

    session.new_conversation(steered(session.root, name="plain", steering="", scale=None))
    view = settings_view(session)
    assert view["steering"] is None
    assert "steering vector" not in format_settings(view)


def test_the_status_line_says_what_steered_the_reply(session):
    from clarity_chat.cli import format_status

    session.new_conversation(steered(session.root))
    assert format_status(session.say("Hi").record).endswith(" · steered ×4.0]")
    session.set_setting("steering_scale", "-4")
    assert format_status(session.say("Again").record).endswith(" · steered ×-4.0]")
    session.set_setting("steering_scale", "0")
    assert "steered" not in format_status(session.say("More").record)


def test_json_settings_carry_the_vector(session):
    import io
    import json

    from clarity_chat.json_chat import JsonChat

    path = session.root / "profiles" / f"{steered(session.root)}.toml"
    stdout = io.StringIO()
    stdin = io.BytesIO(b'{"command": "settings", "id": 1}\n')
    JsonChat(session, stdin, stdout).run([{"command": "profile", "args": str(path)}])
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    result = next(e for e in events if e.get("id") == 1)["result"]
    assert result["steering"]["file"] == "local/vectors/calm.gguf"
    assert result["steering"]["layers"] == [1, 2]
    assert result["settings"]["steering_scale"] == {"value": 4.0, "origin": "profile"}


# --- review findings (checkpoint 1) -----------------------------------------------------


def test_an_extra_pass_after_the_reply_isnt_a_failed_check(hf, monkeypatch):
    """On MPS, Transformers 5.22.2 reads the stop decision one step late (DeferredStopCheck):
    a reply that ends on an end token or stop word runs one more forward pass, then undoes
    it. Every pass was steered, so the check must stay ok."""
    backend, steer = hf
    original = backend.model.generate

    def one_more(inputs, **kwargs):
        output = original(inputs, **kwargs)
        backend.model(inputs)  # the pass Transformers runs and then undoes
        return output

    monkeypatch.setattr(backend.model, "generate", one_more)
    result = backend.complete(PROMPT, SETTINGS, [], steering=steer)
    n = len(result.token_ids)
    assert result.checks["steering"]["ok"], result.checks["steering"]
    assert result.checks["steering"]["model_passes"] == n + 1
    assert result.checks["steering"]["forward_passes"] == {"1": n + 1, "2": n + 1}


def test_a_replay_whose_request_and_steering_disagree_is_refused(hf):
    """The backend sends a logged request exactly as logged, or raises."""
    from clarity_chat.backends import BackendError

    backend, steer = hf
    steered_result = backend.complete(PROMPT, SETTINGS, [], steering=steer)
    plain = backend.complete(PROMPT, SETTINGS, [])
    with pytest.raises(BackendError, match="steering"):
        backend.complete(PROMPT, SETTINGS, [], request=steered_result.request)
    with pytest.raises(BackendError, match="steering"):
        backend.complete(PROMPT, SETTINGS, [], request=plain.request, steering=steer)
    old = {k: v for k, v in plain.request.items() if k != "steering"}  # before v0.2.2
    backend.complete(PROMPT, SETTINGS, [], request=old)


def test_a_logged_scale_of_0_is_refused(session):
    path, record, _ = steered_call(session)
    doctor(path, record["call_id"], lambda row: row["request"]["steering"].update(scale=0.0))
    with pytest.raises(ConfigError, match="logged steering"):
        session.replay(path, record["call_id"])


def test_a_scale_that_overflows_bf16_is_refused(tmp_path):
    """Finite in float32 but infinite once cast to the model's dtype: NaN logits otherwise."""
    model, _ = tiny("llama")
    path = write(tmp_path / "v.gguf", {1: np.full(WIDTH, 2.0, np.float32)})
    v = load(path, sha(path.read_bytes()))
    with pytest.raises(ConfigError, match="isn't finite"), applied(model, v, 1.7e38):
        pass


def test_steering_for_never_drops_a_vector_silently(session):
    """A non-zero scale with no vector must fail loudly, not send an unsteered call."""
    from clarity_chat.session import Session

    session.new_conversation(steered(session.root, name="plain", steering="", scale=None))
    assert session.conversation is not None
    with pytest.raises(AssertionError):
        Session.steering_for(session.conversation.profile, {"steering_scale": 2.0})


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_an_integer_too_large_for_a_float_is_refused_not_a_crash(value):
    """A JSON log can hold one; TOML can't."""
    from clarity_chat.profiles import check_setting

    with pytest.raises(ConfigError, match="must be"):
        check_setting("steering_scale", value, "log")
    with pytest.raises(ConfigError, match="must be"):
        check_setting("temperature", value, "log")


def test_a_steering_file_with_a_nul_is_refused_not_a_traceback(root, session):
    """A NUL makes Path.resolve raise ValueError; a profile or log is untrusted input."""
    from clarity_chat.profiles import load_defaults, load_profile

    table = 'file = "local/vectors/a\\u0000b.gguf"\nsha256 = "00"'
    with pytest.raises(ConfigError, match="steering file"):
        load_profile(session.root, steered(session.root, steering=table), load_defaults(root))
    path, record, _ = steered_call(session)
    doctor(path, record["call_id"], lambda row: row["request"]["steering"].update(file="a\0b"))
    with pytest.raises(ConfigError, match="steering file"):
        session.replay(path, record["call_id"])


# --- multi-agent code review findings ----------------------------------------------------


@pytest.mark.parametrize("bad", ["calm", [1, 2], True, 4.0])
def test_the_export_refuses_a_logged_steering_clarity_doesnt_write(session, bad):
    """A log is untrusted: a mangled steering mustn't export as "none" (unsteered)."""
    from clarity_chat.export import export_log

    path, record, _ = steered_call(session)
    doctor(path, record["call_id"], lambda row: row["request"].update(steering=bad))
    with pytest.raises(ConfigError, match="steering"):
        export_log(session, path)


def test_settings_say_when_the_vector_is_off(session):
    from clarity_chat.cli import format_settings
    from clarity_chat.commands import settings_view

    session.new_conversation(steered(session.root))
    assert "(off: steering_scale 0)" not in format_settings(settings_view(session))
    session.set_setting("steering_scale", "0")
    line = next(
        line
        for line in format_settings(settings_view(session)).split("\n")
        if line.startswith("steering vector")
    )
    assert line.endswith("(off: steering_scale 0)")


def test_a_steering_path_with_backslashes_is_refused(root):
    """Logged as written, a Windows path wouldn't replay or resume on macOS or Linux."""
    from clarity_chat.profiles import load_defaults, load_profile

    steer_root(root)
    table = 'file = "local\\\\vectors\\\\calm.gguf"\nsha256 = "00"'
    with pytest.raises(ConfigError, match="/"):
        load_profile(root, steered(root, steering=table), load_defaults(root))


def test_clarity_starts_without_numpy():
    """Only steering needs numpy; `clarity chats` and llama.cpp chats shouldn't load it."""
    import subprocess
    import sys

    code = "import sys, clarity_chat.cli; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
