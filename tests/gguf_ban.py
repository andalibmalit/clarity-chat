"""A copy of a GGUF file that bans tokens: every key and tensor of the source, plus
`tokenizer.ggml.suppress_tokens`. A1's live test and QA's setup use it (v0.4.0's plan).

Written with the gguf package's reader and writer, as its gguf_new_metadata script copies a
file; the dev group has it for the steering tests. Never imported by Clarity.
"""

from __future__ import annotations

from pathlib import Path

SUPPRESS_KEY = "tokenizer.ggml.suppress_tokens"


def with_suppress_tokens(source: Path, target: Path, ids: list[int]) -> Path:
    import gguf

    reader = gguf.GGUFReader(source)
    field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    assert field is not None, f"{source} names no architecture"
    writer = gguf.GGUFWriter(target, str(field.contents()), endianess=reader.endianess)
    for field in reader.fields.values():
        # The writer adds these itself.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        assert field.name != SUPPRESS_KEY, f"{source} already bans tokens"
        kind = field.types[0]
        sub_type = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=sub_type)
    writer.add_key_value(
        SUPPRESS_KEY, ids, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32
    )
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    return target
