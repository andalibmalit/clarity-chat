"""Steering: add a vector made elsewhere to chosen blocks' outputs on the Transformers backend.

A vector file is llama.cpp's control-vector GGUF, as repeng and llama.cpp write it: one F32
tensor `direction.N` per block N, added to the output of zero-based block N at every
position. Clarity applies a vector and never makes one.

`load` reads a file and checks it against its pinned sha256, `check` checks it against a
model's `config.json` before the model loads, and `applied` adds it to a loaded model for
the length of a `with` block. Notebooks use the same three, so a recompute is steered
exactly as the call was.

The reader is Clarity's own and reads only control vectors: it parses the bytes it hashed,
checks every size against the bytes left before reading, and turns any failure into a
`ConfigError` that names the file.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clarity_chat.backends import BackendError
from clarity_chat.profiles import ConfigError

MAX_BYTES = 16 * 1024 * 1024  # a 7B model's 32 blocks x 4,096 x 4 bytes is 0.5 MB
MAX_TENSORS = 1024
MAX_KEYS = 1024
MAX_WIDTH = 1024 * 1024
# Checked on each: the layer list is model.model.layers, and a block returns a plain tensor.
MODEL_TYPES = ("llama", "olmo3")

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


def load(path: Path, sha256: str | None) -> Vector:
    """Read a control-vector GGUF and check it against its pinned sha256. Doesn't check
    where the file is: Clarity's callers check it's inside the project first."""
    try:
        with path.open("rb") as f:
            data = f.read(MAX_BYTES + 1)
    except FileNotFoundError:
        raise ConfigError(f"The vector file {path} doesn't exist.") from None
    except OSError as e:
        raise ConfigError(f"Can't read the vector file {path}: {e}") from None
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
    return Vector(path, actual, directions, model_hint)


class _Malformed(Exception):
    pass


class _Reader:
    """GGUF version 3, little-endian, holding only `direction.N` tensors."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int, what: str) -> bytes:
        if n > len(self.data) - self.pos:
            raise _Malformed(f"{what} runs past the end of the file")
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
                raise _Malformed(f"{key} is an array of arrays")
            if item == _STRING:
                for _ in range(count):  # each string is checked against the bytes left
                    self.string(f"the value of {key}")
            elif item in _SIZES:
                self.take(count * _SIZES[item], f"the value of {key}")
            else:
                raise _Malformed(f"{key} has unknown value type {item}")
        elif kind in _SIZES:
            self.take(_SIZES[kind], f"the value of {key}")
        else:
            raise _Malformed(f"{key} has unknown value type {kind}")

    def read(self) -> tuple[dict[int, np.ndarray], str | None]:
        if len(self.data) < 24:
            raise _Malformed("the file is too short")
        if self.take(4, "the header") != b"GGUF":
            raise _Malformed("it doesn't start with GGUF")
        version = self.u32("the header")
        if version != 3:
            raise _Malformed(f"GGUF version {version}; Clarity reads version 3, little-endian")
        tensors, keys = self.u64("the header"), self.u64("the header")
        if not 1 <= tensors <= MAX_TENSORS:
            raise _Malformed(f"it has {tensors} tensors; a control vector has 1 to 1,024")
        if keys > MAX_KEYS:
            raise _Malformed(f"it has {keys} keys; at most 1,024")

        alignment, model_hint, seen = 32, None, set()
        for _ in range(keys):
            key = self.string("a key")
            if key in seen:
                raise _Malformed(f"the key {key} appears twice")
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
            directions[block] = values.astype(np.float32)  # a copy, in native byte order
        return directions, model_hint


def check(vector: Vector, config: dict[str, Any], layers: object = None) -> list[int]:
    """Check a vector against a model's config (its `config.json`), and return the blocks
    to steer: `layers`, or every block in the file. `layers` may come from a log, so its
    type is checked too."""
    model_type = config.get("model_type")
    if model_type not in MODEL_TYPES:
        raise ConfigError(
            f"Steering hasn't been checked on model type {model_type!r}; it runs on "
            f"{' and '.join(MODEL_TYPES)}. Ask for it to be added."
        )
    if vector.model_hint is not None and vector.model_hint != model_type:
        raise ConfigError(
            f"{vector.path} was made for model type {vector.model_hint!r}, not {model_type!r}."
        )
    if vector.width != config.get("hidden_size"):
        raise ConfigError(
            f"{vector.path} has width {vector.width}; this model's hidden size is "
            f"{config.get('hidden_size')}."
        )
    if layers is None:
        layers = sorted(vector.directions)
    if (
        not isinstance(layers, list)
        or not layers
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in layers)
    ):
        raise ConfigError("layers must be a list of block numbers, like [10, 11, 12].")
    count = config.get("num_hidden_layers")
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


@contextmanager
def applied(model: Any, vector: Vector, scale: float, layers: object = None) -> Generator[dict]:
    """Add `scale` x the vector to the chosen blocks' outputs of a loaded Transformers model,
    for the length of the `with` block. Yields how many times each steered block ran.

    Each `scale x direction.N` is computed in float32 and cast once to the model's dtype,
    as repeng and llama.cpp compute it. The hooks run before Transformers' own (which
    record `hidden_states`), so those see the steered values. Scale 0 adds nothing."""
    import torch

    if getattr(model, "clarity_steering", None) is not None:
        raise BackendError("This model is already steered; exit the other `applied` first.")
    layers = check(vector, model.config.to_dict(), layers)
    blocks = model.model.layers
    runs = dict.fromkeys(layers, 0)
    if scale == 0:
        yield runs
        return
    deltas = {}
    for n in layers:
        delta = torch.tensor(scale, dtype=torch.float32) * torch.from_numpy(vector.directions[n])
        if not torch.isfinite(delta).all():
            raise ConfigError(f"{scale} x direction.{n} isn't finite.")
        deltas[n] = delta.to(dtype=model.dtype, device=model.device)

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
