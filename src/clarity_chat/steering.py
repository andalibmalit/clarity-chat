"""Steering: add a pinned vector to chosen blocks' outputs, on either backend.

A vector file is llama.cpp's control-vector GGUF, as repeng and llama.cpp write it: one F32
tensor `direction.N` per block N, added to the output of zero-based block N at every
position. `encode` writes one in the same layout, for `clarity vector make` (maker.py).

`load` reads a file and checks it against its pinned sha256, `check` checks it against a
model's `config.json` before the model loads, and `applied` adds it to a loaded Transformers
model for the length of a `with` block (llama-server adds the file itself, at its start).
Notebooks use the same three, so a recompute is steered exactly as the call was.

The reader is Clarity's own and reads only control vectors: it parses the bytes it hashed,
checks every size against the bytes left before reading, and turns any failure into a
`ConfigError` that names the file. `model_file` uses the same reader on a model GGUF, for
the three facts `check` needs on llama.cpp.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from clarity_chat.backends import BackendError
from clarity_chat.profiles import ConfigError

if TYPE_CHECKING:
    import numpy as np

MAX_BYTES = 16 * 1024 * 1024  # a 7B model's 32 blocks x 4,096 x 4 bytes is 0.5 MB
MAX_TENSORS = 1024
MAX_KEYS = 1024
MAX_WIDTH = 1024 * 1024
MODEL_HEADER = 1024 * 1024  # G7: the sizes are among the first 20 keys of every shipped model


@dataclass(frozen=True)
class ModelType:
    """Where a model type keeps its blocks and its sizes, and llama.cpp's name for it."""

    blocks: tuple[str, ...]  # the attributes from a loaded model to its list of blocks
    sizes: str | None  # the config section with hidden_size and the counts; None: the top
    arch: str  # the GGUF's general.architecture after conversion


# Checked on each: the blocks are where `blocks` says, and a block returns a plain tensor.
MODEL_TYPES = {
    "llama": ModelType(("model", "layers"), None, "llama"),
    "olmo3": ModelType(("model", "layers"), None, "olmo2"),
    "gemma4": ModelType(("model", "language_model", "layers"), "text_config", "gemma4"),
}

# GGUF value types with a fixed size; 8 is a string and 9 an array.
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY, _UINT32 = 8, 9, 4
_DIRECTION = re.compile(r"direction\.(0|[1-9][0-9]*)")


@dataclass(frozen=True)
class Vector:
    path: Path  # where it was read from, for messages
    sha256: str
    directions: dict[int, np.ndarray]  # block number: float32 values, all the same length
    model_hint: str | None  # the model type the vector was made for, if the file says
    identity: tuple[int, ...] = ()  # file_identity when read: llama.cpp's start check

    @property
    def width(self) -> int:
        return len(next(iter(self.directions.values())))


@dataclass(frozen=True)
class Steering:
    """What steers one call: the vector, its file as the profile names it, the scale and
    the blocks. A backend logs `to_json()` as the request's `steering`."""

    file: str
    vector: Vector
    scale: float
    layers: list[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "sha256": self.vector.sha256,
            "scale": self.scale,
            "layers": self.layers,
        }


def start_key(steering: Steering | None) -> tuple | None:
    """What a llama.cpp server must have been started with to answer a call (decision 5)."""
    if steering is None:
        return None
    return (steering.vector.sha256, steering.scale, tuple(steering.layers))


def file_identity(st: os.stat_result) -> tuple[int, ...]:
    """What tells a file read before a start from the one there after it. The size and
    modified time alone miss a file of the same size renamed into place with the old time
    (`rsync -a`, `cp -p`); its inode and change time differ. A chmod changes the change
    time too, which stops the start: the safe side."""
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def load(path: Path, sha256: str | None) -> Vector:
    """Read a control-vector GGUF and check it against its pinned sha256. Doesn't check
    where the file is: Clarity's callers check it's inside the project first."""
    try:
        with path.open("rb") as f:
            st = os.fstat(f.fileno())  # before the read: a write during or after it shows
            data = f.read(MAX_BYTES + 1)
    except FileNotFoundError:
        raise ConfigError(f"The vector file {path} doesn't exist.") from None
    except (OSError, ValueError) as e:  # ValueError: a NUL in the path
        raise ConfigError(f"Can't read the vector file {path!r}: {e}") from None
    if len(data) > MAX_BYTES:
        raise ConfigError(f"{path} is over 16 MB; a control vector is smaller. Is it a model file?")
    actual = hashlib.sha256(data).hexdigest()
    if sha256 is None:
        raise ConfigError(
            f"{path} has no sha256 pinned. Its sha256 is {actual}: add "
            f'sha256 = "{actual}" to [steering] to pin it.'
        )
    if actual != sha256:
        raise ConfigError(f"{path} has sha256 {actual}, not the pinned {sha256}.")
    try:
        directions, model_hint = _Reader(data).read()
    except _Malformed as e:
        raise ConfigError(f"{path} isn't a control-vector GGUF: {e}.") from None
    except (struct.error, IndexError, ValueError, UnicodeDecodeError, OverflowError) as e:
        raise ConfigError(f"{path} isn't a control-vector GGUF ({e}).") from None
    return Vector(path, actual, directions, model_hint, file_identity(st))


class ModelFile(NamedTuple):
    """What a model GGUF says of itself: llama.cpp's architecture and the two sizes."""

    arch: str  # general.architecture
    width: int  # ARCH.embedding_length
    blocks: int  # ARCH.block_count


def model_file(path: Path) -> ModelFile:
    """Read a model GGUF's architecture, width and block count from its key-value section.
    Reads at most the first 1 MB and stops at the block count; parses no tensor."""
    try:
        with path.open("rb") as f:
            data = f.read(MODEL_HEADER + 1)
        reader = _Reader(data[:MODEL_HEADER])
        if len(data) > MODEL_HEADER:
            reader.end = "the first 1 MB, where Clarity reads a model's architecture and sizes"
        return reader.model_file()
    except _Malformed as e:
        reason = str(e)
    except (OSError, ValueError, struct.error, IndexError, OverflowError) as e:
        reason = str(e)  # ValueError: a NUL in the path, or a decode
    raise ConfigError(f"{path} isn't a model file Clarity can read: {reason}.")


class _Malformed(Exception):
    pass


class _Reader:
    """GGUF version 3, little-endian, holding only `direction.N` tensors."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos, self.end = data, 0, "the end of the file"

    def take(self, n: int, what: str) -> bytes:
        if n > len(self.data) - self.pos:
            raise _Malformed(f"{what} runs past {self.end}")
        start, self.pos = self.pos, self.pos + n
        return self.data[start : self.pos]

    def u32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]

    def u64(self, what: str) -> int:
        return struct.unpack("<Q", self.take(8, what))[0]

    def string(self, what: str) -> str:
        raw = self.take(self.u64(what), what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise _Malformed(f"{what} isn't UTF-8") from None

    def skip(self, kind: int, key: str) -> None:
        if kind == _STRING:
            self.string(f"the value of {key}")
        elif kind == _ARRAY:
            item = self.u32(f"the value of {key}")
            count = self.u64(f"the value of {key}")
            if item == _ARRAY:
                raise _Malformed(f"{key!r} is an array of arrays")
            if item == _STRING:
                for _ in range(count):  # each string is checked against the bytes left
                    self.string(f"the value of {key}")
            elif item in _SIZES:
                self.take(count * _SIZES[item], f"the value of {key}")
            else:
                raise _Malformed(f"{key!r} has unknown value type {item}")
        elif kind in _SIZES:
            self.take(_SIZES[kind], f"the value of {key}")
        else:
            raise _Malformed(f"{key!r} has unknown value type {kind}")

    def header(self) -> tuple[int, int]:
        """The tensor and key counts."""
        if len(self.data) < 24:
            raise _Malformed("the file is too short")
        if self.take(4, "the header") != b"GGUF":
            raise _Malformed("it doesn't start with GGUF")
        version = self.u32("the header")
        if version != 3:
            raise _Malformed(f"GGUF version {version}; Clarity reads version 3, little-endian")
        return self.u64("the header"), self.u64("the header")

    def model_file(self) -> ModelFile:
        """llama.cpp's converter writes general.architecture first, so the sizes' keys are
        known when they come."""
        _, keys = self.header()
        arch, sizes, seen = "", {}, set()
        names: list[str] = []  # the sizes' keys, once arch is known
        for _ in range(keys):
            key = self.string("a key")
            if key in seen:
                raise _Malformed(f"the key {key!r} appears twice")
            seen.add(key)
            kind = self.u32(f"the type of {key}")
            if key == "general.architecture":
                if kind != _STRING:
                    raise _Malformed("general.architecture isn't a string")
                arch = self.string(key)
                if arch == "controlvector":
                    raise _Malformed("it's a control vector, not a model")
                names = [f"{arch}.embedding_length", f"{arch}.block_count"]
            elif key in names:
                if kind != _UINT32:
                    raise _Malformed(f"{key} isn't a u32")
                sizes[key] = self.u32(key)
                if sizes[key] == 0:
                    raise _Malformed(f"{key} is 0")
                if len(sizes) == 2:
                    return ModelFile(arch, sizes[names[0]], sizes[names[1]])
            else:
                self.skip(kind, key)
        if not names:
            raise _Malformed("it has no general.architecture")
        raise _Malformed(f"it has no {next(k for k in names if k not in sizes)}")

    def read(self) -> tuple[dict[int, np.ndarray], str | None]:
        import numpy as np  # here, not at the top: only steering needs it

        tensors, keys = self.header()
        if not 1 <= tensors <= MAX_TENSORS:
            raise _Malformed(f"it has {tensors} tensors; a control vector has 1 to 1,024")
        if keys > MAX_KEYS:
            raise _Malformed(f"it has {keys} keys; at most 1,024")

        alignment, model_hint, seen = 32, None, set()
        for _ in range(keys):
            key = self.string("a key")
            if key in seen:
                raise _Malformed(f"the key {key!r} appears twice")
            seen.add(key)
            kind = self.u32(f"the type of {key}")
            if key == "general.alignment":
                if kind != _UINT32:
                    raise _Malformed("general.alignment isn't a u32")
                alignment = self.u32(key)
                if not (8 <= alignment <= 4096 and alignment & (alignment - 1) == 0):
                    raise _Malformed(
                        f"general.alignment is {alignment}; it must be a power of two "
                        "from 8 to 4,096"
                    )
            elif key == "controlvector.model_hint":
                if kind != _STRING:
                    raise _Malformed("controlvector.model_hint isn't a string")
                model_hint = self.string(key)
            else:
                self.skip(kind, key)

        infos: list[tuple[int, int, int]] = []  # block, length, offset
        names: set[str] = set()
        for _ in range(tensors):
            name = self.string("a tensor name")
            match = _DIRECTION.fullmatch(name)
            if match is None:
                raise _Malformed(f"the tensor {name!r} isn't named direction.N")
            if name in names:
                raise _Malformed(f"the tensor {name} appears twice")
            names.add(name)
            dims = self.u32(f"{name}'s dimensions")
            if dims != 1:
                raise _Malformed(f"{name} has {dims} dimensions; a direction has 1")
            length = self.u64(f"{name}'s length")
            kind = self.u32(f"{name}'s type")
            if kind != 0:
                raise _Malformed(f"{name} has GGUF type {kind}; a direction is F32 (type 0)")
            if not 1 <= length <= MAX_WIDTH:
                raise _Malformed(f"{name} has length {length}; a direction has 1 to 1,048,576")
            if infos and length != infos[0][1]:
                raise _Malformed(f"{name} has length {length}, the others {infos[0][1]}")
            offset = self.u64(f"{name}'s offset")
            if offset % alignment:
                raise _Malformed(f"{name}'s offset isn't a multiple of {alignment}")
            infos.append((int(match.group(1)), length, offset))

        start = -(-self.pos // alignment) * alignment  # rounded up to the alignment
        directions = {}
        for block, length, offset in infos:
            begin = start + offset
            if begin + 4 * length > len(self.data):
                raise _Malformed(f"direction.{block}'s data runs past the end of the file")
            values = np.frombuffer(self.data, dtype="<f4", count=length, offset=begin)
            if not np.isfinite(values).all():
                raise _Malformed(f"direction.{block} has a value that isn't finite")
            copy = values.astype(np.float32)  # in native byte order
            copy.flags.writeable = False  # the sha256 must describe what gets added
            directions[block] = copy
        return directions, model_hint


def check_type(config: dict[str, Any]) -> str:
    """The model's type, from its config.json, if steering has been checked on it: making a
    vector checks it too, before the model loads."""
    model_type = config.get("model_type")
    if model_type not in MODEL_TYPES:
        *others, last = MODEL_TYPES
        raise ConfigError(
            f"Steering hasn't been checked on model type {model_type!r}; it runs on "
            f"{', '.join(others)} and {last}. Ask for it to be added."
        )
    return model_type


def size_section(config: dict[str, Any]) -> dict[str, Any]:
    """The part of a checked type's config.json that holds its sizes (Gemma 4's
    text_config), itself and not a copy."""
    section = MODEL_TYPES[check_type(config)].sizes
    if section is None:
        return config
    if not isinstance(config.get(section), dict):
        raise ConfigError(f"The model's config.json has no {section}.")
    return config[section]


def model_blocks(model: Any) -> Any:
    """A loaded Transformers model's list of blocks, found where its type keeps them."""
    found = model
    for name in MODEL_TYPES[check_type(model.config.to_dict())].blocks:
        found = getattr(found, name)
    return found


def positions(config: Any) -> int | None:
    """A loaded model's max_position_embeddings, read where its type keeps its sizes. Any
    model runs unsteered, so a type not in MODEL_TYPES is read at the top level."""
    kind = MODEL_TYPES.get(getattr(config, "model_type", None))
    if kind is not None and kind.sizes is not None:
        config = getattr(config, kind.sizes)
    return getattr(config, "max_position_embeddings", None)


def check(vector: Vector, config: dict[str, Any], layers: object = None) -> list[int]:
    """Check a vector against a model's config (its `config.json`), and return the blocks
    to steer: `layers`, or every block in the file. `layers` may come from a log, so its
    type is checked too."""
    model_type = check_type(config)
    if vector.model_hint is not None and vector.model_hint != model_type:
        raise ConfigError(
            f"{vector.path} was made for model type {vector.model_hint!r}, not {model_type!r}."
        )
    sizes = size_section(config)
    if vector.width != sizes.get("hidden_size"):
        raise ConfigError(
            f"{vector.path} has width {vector.width}; this model's hidden size is "
            f"{sizes.get('hidden_size')}."
        )
    if layers is None:
        layers = sorted(vector.directions)
    if (
        not isinstance(layers, list)
        or not layers
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in layers)
    ):
        raise ConfigError("layers must be a list of block numbers, like [10, 11, 12].")
    count = sizes.get("num_hidden_layers")
    if not isinstance(count, int):
        raise ConfigError("The model's config.json has no num_hidden_layers.")
    for n in layers:
        if n == 0:
            raise ConfigError(
                "Block 0 can't be steered: llama.cpp's control-vector loader refuses it, "
                "and Clarity keeps vectors usable on both routes."
            )
        if not 1 <= n < count:
            raise ConfigError(
                f"Block {n} doesn't exist: this model has {count} blocks, and blocks 1 to "
                f"{count - 1} can be steered."
            )
    for n in layers:
        if n not in vector.directions:
            raise ConfigError(
                f"Block {n} isn't in {vector.path}; it has blocks {sorted(vector.directions)}."
            )
    if len(set(layers)) != len(layers):
        raise ConfigError(f"layers lists a block twice: {layers}.")
    return sorted(layers)


def file_config(facts: ModelFile, model_hint: str | None) -> dict[str, Any]:
    """The config.json `check` reads, built from a model file's three facts, so a vector is
    checked alike on both backends. Its type is the vector's hint when the hint has the
    file's architecture; `check` refuses a hint of any other type."""
    types = [name for name, kind in MODEL_TYPES.items() if kind.arch == facts.arch]
    if not types:
        *others, last = dict.fromkeys(kind.arch for kind in MODEL_TYPES.values())
        raise ConfigError(
            f"Steering hasn't been checked on llama.cpp architecture {facts.arch!r}; it runs "
            f"on {', '.join(others)} and {last}. Ask for it to be added."
        )
    model_type = model_hint if model_hint in types else types[0]
    sizes = {"hidden_size": facts.width, "num_hidden_layers": facts.blocks}
    section = MODEL_TYPES[model_type].sizes
    return {"model_type": model_type, **({section: sizes} if section else sizes)}


def check_for_llama_server(vector: Vector, file: str, layers: list[int]) -> None:
    """What llama-server's flags can say: the vector's absolute path, with none of the
    characters its argument parser splits on, and one unbroken range of the file's blocks."""
    path = str(vector.path.absolute())
    if any(c in path for c in ':,"'):
        raise ConfigError(
            "llama-server can't take a vector whose path has a colon, a comma or a double "
            f"quote: {path}. Rename the file, or move the Clarity folder."
        )
    left_out = [n for n in sorted(vector.directions) if layers[0] < n < layers[-1]]
    left_out = [n for n in left_out if n not in layers]
    if left_out:
        raise ConfigError(
            f"On llama.cpp, layers must take in every block of {file} between its first and "
            "last: llama-server adds a vector to one unbroken range. layers is "
            f"{layers}, and the file also has {left_out}."
        )


def check_float32(vector: Vector, layers: list[int], scale: float) -> None:
    """Refuse a scale whose product with a direction isn't finite in float32, the type
    llama.cpp adds it in. Transformers checks in the model's dtype, in `applied`."""
    import numpy as np

    for n in layers:
        with np.errstate(over="ignore"):
            scaled = np.float32(scale) * vector.directions[n]
        if not np.isfinite(scaled).all():
            raise ConfigError(f"{scale} x direction.{n} isn't finite in float32.")


def encode(directions: dict[int, Any], model_hint: str) -> bytes:
    """A control vector as repeng writes one, byte for byte what gguf's GGUFWriter gives:
    GGUF version 3, three keys, then one F32 `direction.N` per block, each padded to 32."""
    import numpy as np

    def text(s: str) -> bytes:
        return struct.pack("<Q", len(s.encode())) + s.encode()

    def padded(data: bytes) -> bytes:
        return data + b"\0" * (-len(data) % 32)

    keys = [
        ("general.architecture", struct.pack("<I", 8) + text("controlvector")),
        ("controlvector.model_hint", struct.pack("<I", 8) + text(model_hint)),
        ("controlvector.layer_count", struct.pack("<II", 4, len(directions))),
    ]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(directions), len(keys))
    out += b"".join(text(key) + value for key, value in keys)
    data = b""
    for n in sorted(directions):
        values = np.ascontiguousarray(directions[n], dtype="<f4").reshape(-1)
        out += text(f"direction.{n}") + struct.pack("<IQIQ", 1, len(values), 0, len(data))
        data += padded(values.tobytes())
    return padded(out) + data


@contextmanager
def applied(model: Any, vector: Vector, scale: float, layers: object = None) -> Generator[dict]:
    """Add `scale` x the vector to the chosen blocks' outputs of a loaded Transformers model,
    for the length of the `with` block. Yields how many times each steered block ran.

    Each `scale x direction.N` is computed in float32 and cast once to the model's dtype,
    as repeng does (llama.cpp keeps it in float32). The hooks run before Transformers' own (which
    record `hidden_states`), so those see the steered values. Scale 0 adds nothing."""
    import numpy as np
    import torch

    if getattr(model, "clarity_steering", None) is not None:
        raise BackendError("This model is already steered; exit the other `applied` first.")
    blocks = model_blocks(model)
    config = model.config.to_dict()
    layers = check(vector, config, layers)
    runs = dict.fromkeys(layers, 0)
    if scale == 0:
        yield runs
        return
    deltas = {}
    for n in layers:
        # Cast on the CPU, then move: the same bits on every device.
        with np.errstate(over="ignore"):  # an overflow is refused just below
            scaled = np.float32(scale) * vector.directions[n]
        delta = torch.from_numpy(scaled).to(model.dtype)
        delta = delta.to(model.device)
        if not torch.isfinite(delta).all():  # after the cast: bf16 overflows below float32
            raise ConfigError(f"{scale} x direction.{n} isn't finite in {model.dtype}.")
        deltas[n] = delta

    def hook(n: int, delta: Any) -> Any:
        def add(module: Any, args: Any, output: Any) -> Any:
            if not isinstance(output, torch.Tensor):
                raise BackendError(
                    f"Block {n} returned {type(output).__name__}, not a tensor; "
                    "steering can't be added to it."
                )
            runs[n] += 1
            return output + delta

        return add

    handles = []
    model.clarity_steering = {"sha256": vector.sha256, "scale": scale, "layers": layers}
    try:
        for n, delta in deltas.items():
            handles.append(blocks[n].register_forward_hook(hook(n, delta), prepend=True))
        yield runs
    finally:
        for handle in handles:
            handle.remove()
        del model.clarity_steering
