"""Meaning-search tests for v0.3.3 (opt-in: `uv run pytest -m embed -v -s`).

Documents and meaning search (v0.3.3) are only worth building if llama-server's embeddings
come back bit-identical, so that a search can be logged and repeated exactly. These are
tests T1, T2, T6 and the token-IDs half of T8 from the embedding research note
(notes/research/rag/04-embedding-determinism.md, section 7). If they pass, v0.3.3 goes
ahead; if one fails, its message says what differed.

Each server starts with the note's pins for an embedding server (EMBED_FLAGS). Inputs are
token IDs from bge-small's own Hugging Face tokenizer, as Clarity would send them. Vectors
are compared as raw float32 bytes, with no tolerance. The texts are made up.

Needs the built llama-server and bge-small fetched (`uv run clarity fetch bge-small-en-v1.5`).
"""

from __future__ import annotations

import base64
import dataclasses
import math
import struct
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from clarity_chat.backends.llama_cpp import LlamaServer, server_binary
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.render import Tokenizer

pytestmark = pytest.mark.embed

REPO = Path(__file__).resolve().parents[1]
MODEL = "bge-small-en-v1.5"
DIM = 384  # bge-small's hidden size, and so its vector length
N_CTX = 512  # bge-small's training context (max_position_embeddings)


def _weights() -> tuple[Path | None, str]:
    try:
        server_binary(load_server_config(REPO).llama_cpp_dir)
        return check_weights_quick(REPO, lock_entry(REPO, load_models(REPO)[MODEL])), ""
    except ConfigError as e:
        return None, str(e)


WEIGHTS, _reason = _weights()
if WEIGHTS is None:
    pytestmark = [pytest.mark.embed, pytest.mark.skip(reason=_reason)]

# Note 04's pins for an embedding server ("Pins for an embedding server"), set on top of
# server.toml's flags. Every one is explicit, so no server default or GGUF value applies.
EMBED_FLAGS: dict[str, Any] = {
    "embedding": True,
    "pooling": "cls",  # bge-small's 1_Pooling/config.json; unset would take the GGUF's value
    "embd-normalize": 2,  # L2, like its 2_Normalize module; also sent with every request
    "parallel": 1,  # one slot, so each input is evaluated alone (the default is 4 slots)
    "no-kv-unified": True,
    "ctx-size": N_CTX,
    "batch-size": N_CTX,  # equal sizes: with --embedding a larger batch is quietly lowered
    "ubatch-size": N_CTX,  # every legal input fits one ubatch
    "flash-attn": "on",
    "n-gpu-layers": "all",
    "threads": 8,
    "threads-batch": 8,
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "cache-ram": 0,
    "cache-reuse": 0,
    "ctx-checkpoints": 0,
    "no-cache-idle-slots": True,
    "no-context-shift": True,
    "fit": "off",
}
# T6: no GPU at all. With only n-gpu-layers 0, llama.cpp may still offload big matrix
# products to the GPU (--op-offload is on by default); --device none rules that out.
CPU_ONLY: dict[str, Any] = {"n-gpu-layers": 0, "device": "none"}

# Made-up sentences; the longer texts below are runs of them.
SENTENCES = [
    "The lighthouse keeper on Marrow Point logs the wind speed every morning at six.",
    "Her notebook lists forty-two kinds of seaweed, each with a sketch and a date.",
    "In 1987 the harbour council voted to paint every bollard a different shade of green.",
    "The ferry to Little Tarn leaves twice a day, except when fog rolls in from the bay.",
    "Children collect smooth glass from the north beach and trade it at the Saturday market.",
    "A bakery on Quay Street sells a rye loaf flavoured with caraway and burnt honey.",
    "The baker, Odile Ferrant, starts the ovens at three and sleeps again until ten.",
    "Her sourdough starter is said to be older than the town hall, though nobody can prove it.",
    "On windy days the flour drifts out of the door and dusts the cobbles white.",
    "To true a bicycle wheel, tighten the spokes opposite the bulge by a quarter turn each.",
    "Check the rim against a fixed pointer, such as a zip tie on the frame, after every pass.",
    "A wheel that wobbles more than two millimetres side to side needs a full rebuild.",
    "Brake pads wear unevenly when the caliper sits a little off the centre of the rim.",
    "The weather station on the ridge reports pressure, humidity and rainfall every ten minutes.",
    "Its rain gauge tips a small bucket each time two tenths of a millimetre have fallen.",
    "When the bucket sticks, the station reports a dry week in the middle of a storm.",
    "A volunteer climbs the ridge each spring to clean the sensors and replace the batteries.",
    "The town choir rehearses in the old net loft, where the acoustics flatter every voice.",
    "Their winter concert always ends with a song about a whale that learned to count.",
    "Tickets cost nothing, but everyone is asked to bring a candle and a story.",
    "Gulls on the fish pier have learned to open the lids of the bait boxes.",
    "The harbour master keeps a ledger of every boat that has moored there since 1952.",
    "Some entries are only a name and a tide time; others run to a page of gossip.",
    "A boat called the Patient Heron appears in the ledger every August for sixty years.",
    "The library's reading room faces west, so the afternoon light reaches the back shelves.",
    "Its oldest map shows a road to a village that, as far as anyone knows, never existed.",
    "Researchers who ask about the map are given tea and a folder of contradictory letters.",
    "The cliff path is closed in March while the puffins settle on the ledges below.",
    "Walkers who ignore the sign are politely turned back by a retired schoolteacher.",
    "In the evenings the cafe serves soup in mugs and lets you pay what you think is fair.",
    "The recipe for its fish soup changes with the catch, but the saffron stays the same.",
    "A small museum in the customs house displays knots, lamps and one stuffed albatross.",
    "The albatross was found on the beach in 1911 and has been repaired with string twice.",
    "Every exhibit has a handwritten label, some of them corrected in a different pen.",
    "The museum is open whenever the curator remembers to unlock it, usually after lunch.",
    "Night buses run inland on Fridays, stopping at farms whose names sound like spells.",
    "Drivers know most passengers by name and wait for anyone running down a lane.",
    "The last stop is a crossroads with a bench, a post box and a view of the whole valley.",
    "From there it is a forty-minute walk to the youth hostel, mostly uphill.",
    "The hostel warden bakes flapjacks on Sundays and hides the spare blankets in the piano.",
]


def _run(start: int, count: int) -> str:
    return " ".join(SENTENCES[start : start + count])


# 20 passages of 6 to 497 tokens ([CLS] and [SEP] included), then 5 short queries: 25 texts,
# as note 04 suggests.
TEXTS = [
    "Fog again today.",
    "Bring a candle and a story.",
    "Note: the 7:40 ferry is cancelled; take the 9:15 instead.",
    "The café's naïve mural shows a whale counting: one, two, three...",
    SENTENCES[9],
    _run(3, 2),
    _run(13, 3),
    _run(17, 4),
    _run(20, 5),
    _run(0, 7),
    _run(24, 9),
    _run(5, 11),
    _run(10, 13),
    _run(1, 15),
    _run(19, 17),
    _run(4, 19),
    _run(0, 21),
    _run(8, 23),
    _run(2, 25),
    _run(6, 27),
    "how do I true a bicycle wheel",
    "When does the ferry to Little Tarn leave?",
    "bakery rye loaf caraway",
    "Which seabirds nest on the cliffs in spring?",
    "museum opening hours",
]


@pytest.fixture(scope="module")
def ids() -> list[list[int]]:
    """Each text's token IDs from bge-small's own tokenizer, with [CLS] and [SEP] added, as
    sentence-transformers does."""
    model = load_models(REPO)[MODEL]
    try:
        tokenizer = Tokenizer.from_hub(model.tokenizer_repo, model.tokenizer_revision)
    except ConfigError as e:
        pytest.skip(str(e))
    found = [list(tokenizer.tok.encode(text, add_special_tokens=True)) for text in TEXTS]
    too_long = [i for i, x in enumerate(found) if len(x) > N_CTX]
    assert not too_long, f"Test bug: texts {too_long} are longer than {N_CTX} tokens."
    return found


@contextmanager
def running(tmp_path: Path, **overrides: Any) -> Generator[LlamaServer]:
    """An embedding server with EMBED_FLAGS (and `overrides`), stopped afterwards."""
    config = load_server_config(REPO)
    config = dataclasses.replace(config, flags={**config.flags, **EMBED_FLAGS, **overrides})
    assert WEIGHTS is not None
    server = LlamaServer(config, WEIGHTS, tmp_path)
    record = server.start()
    try:
        slots = record["props"].get("total_slots")
        assert slots == 1, f"llama-server runs {slots} slots, but --parallel 1 asks for one."
        devices = [d["name"] for d in record["placement"]["devices"]]
        print(f"\nserver: devices {devices}, graph splits {record['graph_splits']}")
        yield server
    finally:
        server.stop()


def embed(server: LlamaServer, inputs: list[Any]) -> tuple[list[bytes], int]:
    """One request to /v1/embeddings: each input's vector as raw float32 bytes, in input
    order, and the request's usage.prompt_tokens."""
    r = httpx.post(
        f"{server.url}/v1/embeddings",
        headers=server.headers,
        json={"input": inputs, "encoding_format": "base64", "embd_normalize": 2},
        timeout=120,
        trust_env=False,
        verify=False,
    )
    assert r.status_code == 200, f"/v1/embeddings answered {r.status_code}: {r.text[:500]}"
    body = r.json()
    data = sorted(body["data"], key=lambda d: d["index"])
    assert [d["index"] for d in data] == list(range(len(inputs))), "missing or extra vectors"
    vectors = [base64.b64decode(d["embedding"]) for d in data]
    for i, vector in enumerate(vectors):
        assert len(vector) == DIM * 4, f"input {i}: {len(vector)} bytes, not {DIM} float32s"
        # llama-server returns a zero vector, not an error, when a lookup fails.
        norm = math.sqrt(math.fsum(x * x for x in floats(vector)))
        assert abs(norm - 1) < 1e-3, f"input {i}: vector length {norm}, not 1 (L2-normalized)"
    return vectors, body["usage"]["prompt_tokens"]


def embed_ids(server: LlamaServer, inputs: list[list[int]]) -> list[bytes]:
    """Vectors for token-ID inputs, checking the server counted exactly those tokens."""
    vectors, prompt_tokens = embed(server, inputs)
    sent = sum(len(x) for x in inputs)
    assert prompt_tokens == sent, f"sent {sent} tokens, the server counted {prompt_tokens}"
    return vectors


def each_alone(server: LlamaServer, ids: list[list[int]]) -> list[bytes]:
    """Every text in its own request."""
    return [embed_ids(server, [x])[0] for x in ids]


def floats(vector: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(vector) // 4}f", vector)


def difference(a: bytes, b: bytes) -> str:
    # By bytes, not by value: -0.0 and 0.0 are equal floats but different bits.
    changed = [k for k in range(DIM) if a[4 * k : 4 * k + 4] != b[4 * k : 4 * k + 4]]
    x, y = floats(a), floats(b)
    largest = max(abs(x[k] - y[k]) for k in changed)
    return f"{len(changed)} of {DIM} values differ, the largest by {largest:.3g}"


def differences(runs: dict[str, list[bytes]], ids: list[list[int]]) -> list[str]:
    """Every run against the first: one line per text whose vector isn't bit-identical."""
    (first_name, first), *others = runs.items()
    return [
        f"text {i} ({len(ids[i])} tokens), {name} vs {first_name}: {difference(a, b)}"
        for name, run in others
        for i, (a, b) in enumerate(zip(first, run, strict=True))
        if a != b
    ]


def report(lines: list[str]) -> str:
    shown = "\n".join(f"  {line}" for line in lines[:15])
    more = f"\n  … and {len(lines) - 15} more" if len(lines) > 15 else ""
    return f"{len(lines)} difference(s):\n{shown}{more}"


def test_t1_vectors_repeat_back_to_back_and_after_a_restart(tmp_path, ids):
    runs = {}
    for start in (1, 2):
        with running(tmp_path) as server:
            for round_ in (1, 2, 3):
                runs[f"start {start} round {round_}"] = each_alone(server, ids)
    lines = differences(runs, ids)
    assert not lines, (
        "T1: the same text, embedded alone, did not give the same vector every time "
        "(three times on one server, three more after a restart, GPU). Embeddings are not "
        f"bit-identical run to run with the pinned flags. {report(lines)}"
    )


def test_t2_vectors_dont_depend_on_what_else_the_request_holds(tmp_path, ids):
    with running(tmp_path) as server:
        alone = each_alone(server, ids)
        together = embed_ids(server, ids)
        backwards = embed_ids(server, ids[::-1])[::-1]
    runs = {
        "one per request": alone,
        "all 25 in one request": together,
        "all 25 in reverse order": backwards,
    }
    lines = differences(runs, ids)
    assert not lines, (
        "T2: a text's vector changed with the other texts sent in the same request, or "
        "their order, even with --parallel 1 (GPU). A vector would depend on its neighbours. "
        f"{report(lines)}"
    )


def test_t6_cpu_vectors_repeat_at_a_fixed_thread_count(tmp_path, ids):
    lines, first_run = [], {}
    for threads in (8, 4):
        runs = {}
        for start in (1, 2):
            flags = {**CPU_ONLY, "threads": threads, "threads-batch": threads}
            with running(tmp_path, **flags) as server:
                used = server.record["placement"]["devices"]
                assert not used, f"T6 needs the CPU only, but llama-server used {used}."
                for round_ in (1, 2):
                    runs[f"{threads} threads start {start} round {round_}"] = each_alone(
                        server, ids
                    )
        lines += differences(runs, ids)
        first_run[threads] = next(iter(runs.values()))
    changed = sum(a != b for a, b in zip(first_run[8], first_run[4], strict=True))
    print(f"\nT6, for information: {changed} of {len(ids)} vectors differ between 8 and 4 threads")
    assert not lines, (
        "T6: on the CPU, at a fixed thread count, the same text did not give the same vector "
        f"every time (two server starts, two rounds each). {report(lines)}"
    )


def tokenize(server: LlamaServer, text: str) -> list[int]:
    """The server's own tokens for `text`, made the way /v1/embeddings makes them for a
    string (special tokens added and parsed)."""
    r = httpx.post(
        f"{server.url}/tokenize",
        headers=server.headers,
        json={"content": text, "add_special": True, "parse_special": True},
        timeout=30,
        trust_env=False,
        verify=False,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def test_t8_the_server_tokenizes_each_text_like_the_source_tokenizer(tmp_path, ids):
    with running(tmp_path) as server:
        server_ids = [tokenize(server, text) for text in TEXTS]
    lines = []
    for i, (ours, theirs) in enumerate(zip(ids, server_ids, strict=True)):
        if ours != theirs:
            pairs = enumerate(zip(ours, theirs, strict=False))
            at = next((k for k, (a, b) in pairs if a != b), min(len(ours), len(theirs)))
            lines.append(
                f"text {i}: {len(ours)} tokens from the Hugging Face tokenizer, {len(theirs)} "
                f"from llama-server; first difference at position {at}: "
                f"{ours[at : at + 3]} vs {theirs[at : at + 3]}"
            )
    assert not lines, (
        "T8: llama-server's tokenizer (from the GGUF) splits some texts differently from "
        "bge-small's own tokenizer, so sending a string would embed different tokens than "
        "sending token IDs. Clarity sends token IDs, which the other tests check; this "
        f"failure means the server's tokenization can't stand in for them.\n{report(lines)}"
    )


def test_t8_token_ids_and_the_string_give_the_same_vector(tmp_path, ids):
    with running(tmp_path) as server:
        from_ids = each_alone(server, ids)
        server_ids = [tokenize(server, text) for text in TEXTS]
        from_text = [embed(server, [text]) for text in TEXTS]
    same = [i for i in range(len(TEXTS)) if server_ids[i] == ids[i]]
    assert same, "T8: llama-server tokenized no text like the source tokenizer: see the test above."
    lines = []
    for i in same:
        vector, prompt_tokens = from_text[i][0][0], from_text[i][1]
        if prompt_tokens != len(ids[i]):
            lines.append(f"text {i}: {len(ids[i])} token IDs, {prompt_tokens} tokens counted")
        if vector != from_ids[i]:
            lines.append(f"text {i} ({len(ids[i])} tokens): {difference(from_ids[i], vector)}")
    assert not lines, (
        "T8: for texts the server tokenizes exactly like the source tokenizer, sending the "
        "string and sending its token IDs gave different results; they should be the same "
        f"computation. {report(lines)}"
    )
