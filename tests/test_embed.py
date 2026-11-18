"""Meaning search's gate on bge-m3 (v0.4.1 A1; opt-in: `uv run pytest -m embed -v -s`).

A search is logged and repeated exactly only if the embedder's vectors are bit-identical,
so these run the shipped launch (documents/embedder.py: its flag table, on Metal) on
made-up texts from a few tokens to 8,192:
- T1: the same text gives the same vector back to back and after a restart.
- T2: a text's vector doesn't depend on what else a request holds.
- The 8,192-ID boundary is accepted and leaves the server running; a ~9,000-token text
  counts past 8,192 through the shipped tokenizer path, so a truncation setting can't make
  the refusal silent.
- T8: llama-server's own tokenization is printed against bge-m3's (Clarity sends IDs, so
  a difference isn't a failure), and where the two agree, IDs and the string must give
  the same vector.
- The reference: bge-m3's own weights in Transformers (float32, CPU, CLS pooling, then L2
  normalization, as its modules.json says) against the shipped vectors.
Vectors are compared as raw float32 bytes, with no tolerance, except against the
reference.

Needs the built llama-server, bge-m3 fetched (`uv run clarity fetch bge-m3`), and for the
reference the hf extra (`uv sync --extra hf`; a plain `uv sync` drops torch).
"""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from functools import cache
from itertools import count
from pathlib import Path
from typing import Any

import httpx
import pytest

from clarity_chat.backends.llama_cpp import server_binary
from clarity_chat.documents import Embedder, TooLong
from clarity_chat.documents.embedder import DIMENSIONS, MAX_TOKENS, MODEL
from clarity_chat.files import sha256_file
from clarity_chat.models import check_weights_quick, lock_entry
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.render import Tokenizer

pytestmark = pytest.mark.embed

REPO = Path(__file__).resolve().parents[1]


def _weights() -> tuple[Path | None, str]:
    try:
        server_binary(load_server_config(REPO).llama_cpp_dir)
        return check_weights_quick(REPO, lock_entry(REPO, load_models(REPO)[MODEL])), ""
    except ConfigError as e:
        return None, str(e)


WEIGHTS, _reason = _weights()
if WEIGHTS is None:
    pytestmark = [pytest.mark.embed, pytest.mark.skip(reason=_reason)]

# Made-up sentences; the longer texts below are built from them and their like.
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
SURNAMES = ["Ferrant", "Okafor", "Lindqvist", "Marrow", "Tarn", "Quayle", "Nakashima", "Ruiz"]


def _run(start: int, count: int) -> str:
    return " ".join(SENTENCES[start : start + count])


def prose() -> Iterator[str]:
    for n in count():
        yield SENTENCES[(7 * n) % len(SENTENCES)]


def references() -> Iterator[str]:
    for n in count(1):
        surname = SURNAMES[n % len(SURNAMES)]
        yield (
            f"[{n}] {surname}, {'ABCDEFGH'[n % 8]}. ({1950 + n % 70}). "
            f"{SENTENCES[n % len(SENTENCES)].rstrip('.')}. Journal of Made-up Coastal "
            f"Studies, {n % 40 + 1}({n % 4 + 1}), {n * 3 % 400 + 1}-{n * 3 % 400 + 19}. "
            f"https://doi.org/10.5555/mu.{n:05d}"
        )


def table() -> Iterator[str]:
    yield "| no. | station | rainfall (mm) | sensor cleaned |\n|---|---|---|---|"
    for n in count(1):
        cleaned = "yes" if n % 3 else "no"
        yield f"| {n} | {SURNAMES[n % 8]} Ridge {n % 13} | {n * 17 % 1000 / 10:.1f} | {cleaned} |"


def code() -> Iterator[str]:
    for n in count(1):
        yield (
            f"def tide_step_{n}(level, hours):\n"
            f"    rise = level * {n % 9 + 1} / (hours + {n % 5 + 1})\n"
            f"    return rise if rise > {n * 3 % 50} else -rise  # made up\n"
        )


def grown(units: Iterable[str], target: int, tokens: Callable[[str], int], sep: str) -> str:
    """Units joined by `sep` until the sum of their own token counts reaches `target`. Each
    unit is short, so the sum doesn't depend on any truncation the tokenizer might do."""
    taken, total = [], 0
    for unit in units:
        if total >= target:
            break
        taken.append(unit)
        total += tokens(unit)
    return sep.join(taken)


# 20 short passages, 4 long ones (built from the tokenizer, below), then 5 queries.
SHORT = [
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
]
LONG = [(prose, 1500, " "), (references, 3000, "\n"), (table, 5000, "\n"), (code, 7800, "\n")]
QUERIES = [
    "how do I true a bicycle wheel",
    "When does the ferry to Little Tarn leave?",
    "bakery rye loaf caraway",
    "Which seabirds nest on the cliffs in spring?",
    "museum opening hours",
]


hub_tokenizer = cache(Tokenizer.from_hub)  # loaded once, as a session's loader caches it


def made(logs: Path) -> Embedder:
    return Embedder(REPO, load_server_config(REPO), logs, hub_tokenizer)


def counter(tokenizer: Tokenizer) -> Callable[[str], int]:
    return lambda unit: len(tokenizer.tok.encode(unit, add_special_tokens=False))


@pytest.fixture(scope="module")
def texts(tmp_path_factory) -> dict[str, Any]:
    """The texts, their IDs through the shipped path (Embedder.ids), and the tokenizer."""
    embedder = made(tmp_path_factory.mktemp("logs"))
    model = load_models(REPO)[MODEL]
    tokenizer = hub_tokenizer(model.tokenizer_repo, model.tokenizer_revision)
    tokens = counter(tokenizer)
    passages = SHORT + [grown(units(), target, tokens, sep) for units, target, sep in LONG]
    all_texts = passages + QUERIES
    ids = [embedder.ids(text) for text in all_texts]
    print(f"\ntoken counts: {[len(x) for x in ids]}")
    assert max(len(x) for x in ids) > 7000, "Test bug: no text near the limit."
    return {"texts": all_texts, "passages": len(passages), "ids": ids, "tokenizer": tokenizer}


@contextmanager
def running(tmp_path: Path) -> Generator[Embedder]:
    """The shipped embedder, started, stopped afterwards."""
    embedder = made(tmp_path)
    engine = embedder.start()
    try:
        assert embedder.server is not None
        record = embedder.server.record
        print(f"\nembedder on {engine['device']}, graph splits {record['graph_splits']}")
        yield embedder
    finally:
        embedder.close()


def embed(embedder: Embedder, inputs: list[Any]) -> tuple[list[bytes], int]:
    """One raw request to /v1/embeddings (for T2 and T8, which send what Clarity doesn't):
    each input's vector as float32 bytes, in input order, and usage.prompt_tokens."""
    server = embedder.server
    assert server is not None
    r = httpx.post(
        f"{server.url}/v1/embeddings",
        headers=server.headers,
        json={"input": inputs, "encoding_format": "base64", "embd_normalize": 2},
        timeout=600,
        trust_env=False,
        verify=False,
    )
    assert r.status_code == 200, f"/v1/embeddings answered {r.status_code}: {r.text[:500]}"
    body = r.json()
    data = sorted(body["data"], key=lambda d: d["index"])
    assert [d["index"] for d in data] == list(range(len(inputs))), "missing or extra vectors"
    return [base64.b64decode(d["embedding"]) for d in data], body["usage"]["prompt_tokens"]


def floats(vector: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(vector) // 4}f", vector)


def difference(a: bytes, b: bytes) -> str:
    # By bytes, not by value: -0.0 and 0.0 are equal floats but different bits.
    changed = [k for k in range(DIMENSIONS) if a[4 * k : 4 * k + 4] != b[4 * k : 4 * k + 4]]
    x, y = floats(a), floats(b)
    largest = max(abs(x[k] - y[k]) for k in changed)
    return f"{len(changed)} of {DIMENSIONS} values differ, the largest by {largest:.3g}"


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


def test_t1_vectors_repeat_back_to_back_and_after_a_restart(tmp_path, texts):
    ids = texts["ids"]
    runs = {}
    for start in (1, 2):
        with running(tmp_path) as embedder:
            for round_ in (1, 2, 3):
                runs[f"start {start} round {round_}"] = [embedder.vector(x) for x in ids]
    lines = differences(runs, ids)
    assert not lines, (
        "T1: the same text did not give the same vector every time (three times on one "
        f"server, three more after a restart, through Embedder.vector). {report(lines)}"
    )


def test_t2_vectors_dont_depend_on_what_else_the_request_holds(tmp_path, texts):
    ids = texts["ids"]
    with running(tmp_path) as embedder:
        alone = [embedder.vector(x) for x in ids]
        together, _ = embed(embedder, ids)
        backwards = embed(embedder, ids[::-1])[0][::-1]
    runs = {"one per request": alone, "all in one request": together, "reversed": backwards}
    lines = differences(runs, ids)
    assert not lines, (
        "T2: a text's vector changed with the other texts in the same request, or their "
        f"order, even with --parallel 1. A vector would depend on its neighbours. {report(lines)}"
    )


def test_the_8192_id_boundary_is_accepted_and_the_server_keeps_running(tmp_path, texts):
    tokenizer, ids = texts["tokenizer"], texts["ids"]
    text = grown(prose(), 8400, counter(tokenizer), " ")
    body = tokenizer.tok.encode(text, add_special_tokens=False)
    assert len(body) >= MAX_TOKENS - 2, f"Test bug: only {len(body)} tokens to cut from."
    boundary = [ids[0][0], *body[: MAX_TOKENS - 2], ids[0][-1]]  # <s> … </s>
    assert len(boundary) == MAX_TOKENS
    with running(tmp_path) as embedder:
        before = embedder.vector(ids[0])
        embedder.vector(boundary)  # decision 6's checks: one finite unit vector, 8,192 counted
        assert embedder.vector(ids[0]) == before, "the server changed after the 8,192 input"
        assert embedder.server is not None and embedder.server.proc is not None
        assert embedder.server.proc.poll() is None


def test_no_truncation_a_9000_token_text_counts_past_8192(tmp_path, texts):
    text = grown(prose(), 9000, counter(texts["tokenizer"]), " ")
    with pytest.raises(TooLong) as refused:
        made(tmp_path).ids(text)
    assert refused.value.tokens > MAX_TOKENS
    print(f"\na text of at least 9,000 tokens counted {refused.value.tokens}")


def tokenize(embedder: Embedder, text: str) -> list[int]:
    """The server's own tokens for `text`, made as /v1/embeddings makes them for a string."""
    server = embedder.server
    assert server is not None
    r = httpx.post(
        f"{server.url}/tokenize",
        headers=server.headers,
        json={"content": text, "add_special": True, "parse_special": True},
        timeout=60,
        trust_env=False,
        verify=False,
    )
    r.raise_for_status()
    return r.json()["tokens"]


ECHO_TEXTS = [  # XLM-R's harder cases: numbers, accents, other scripts, code
    "Order 12,345.67 kg by 2026-11-18, ref. #A-0042.",
    "Crème brûlée, façade, naïve, Ærø, Łódź, São Paulo.",
    "Ο φάρος ανάβει στις έξι. Маяк зажигается в шесть. 灯台は六時に灯る。",
    "def f(x):\n    return {'a': [x ** 2, x // 3]}  # résumé",
]


def test_t8_the_server_s_tokenization_is_printed_not_gated(tmp_path, texts):
    """Clarity sends IDs, so llama-server's tokenizer doesn't need to match bge-m3's; the
    differences are printed for the record."""
    tokenizer = texts["tokenizer"]
    with running(tmp_path) as embedder:
        lines = []
        for text in [*texts["texts"], *ECHO_TEXTS]:
            ours = list(tokenizer.tok.encode(text, add_special_tokens=True))
            theirs = tokenize(embedder, text)
            if ours != theirs:
                at = next(
                    (k for k, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), None
                )
                lines.append(f"{text[:40]!r}: {len(ours)} vs {len(theirs)} tokens, first at {at}")
    print(f"\nT8 echo: {len(lines)} text(s) tokenized differently by llama-server")
    for line in lines:
        print(f"  {line}")


def test_t8_ids_and_the_string_give_the_same_vector_where_they_agree(tmp_path, texts):
    ids = texts["ids"]
    with running(tmp_path) as embedder:
        same = [i for i, text in enumerate(texts["texts"]) if tokenize(embedder, text) == ids[i]]
        lines = []
        for i in same:
            [vector], counted = embed(embedder, [texts["texts"][i]])
            if counted != len(ids[i]):
                lines.append(f"text {i}: {len(ids[i])} token IDs, {counted} tokens counted")
            elif vector != embedder.vector(ids[i]):
                lines.append(f"text {i}: {difference(embedder.vector(ids[i]), vector)}")
    assert same, "T8: llama-server tokenized no text like bge-m3's tokenizer (see the echo)."
    assert not lines, (
        "T8: for texts the server tokenizes exactly as bge-m3's tokenizer does, the string "
        f"and its IDs gave different results; they should be one computation. {report(lines)}"
    )


def test_reference_the_source_model_gives_the_same_vectors(tmp_path, texts):
    """bge-m3's own weights in Transformers, loaded under the security review's conditions
    (notes/research/reviews/v041-security.md, question 4 and S1): the pickle's sha256
    checked first; from the lock's folders only, never by repo ID; no remote code;
    offline; PyTorch's restricted loader."""
    import huggingface_hub.constants

    assert huggingface_hub.constants.HF_HUB_OFFLINE is True, (
        "Hugging Face is online in this run (a shell's HF_HUB_OFFLINE=0 overrides "
        "conftest's): unset it and run again."
    )
    torch = pytest.importorskip("torch", reason="needs the hf extra: `uv sync --extra hf`")
    from transformers import AutoModel, AutoTokenizer

    model_entry = load_models(REPO)[MODEL]
    entry = lock_entry(REPO, model_entry)
    source = Path(entry["source_dir"]).expanduser()
    pickle = source / "pytorch_model.bin"  # the reviewed bytes only (security review, S1)
    assert sha256_file(pickle) == model_entry.weights_sha256, f"{pickle} isn't the reviewed file"
    reference_tokenizer: Any = AutoTokenizer.from_pretrained(
        Path(entry["tokenizer_dir"]).expanduser(), trust_remote_code=False, local_files_only=True
    )
    ids = texts["ids"]
    assert [reference_tokenizer(t)["input_ids"] for t in texts["texts"]] == ids, (
        "the shipped tokenizer path and the reference's tokenizer disagree"
    )
    model = AutoModel.from_pretrained(
        source,
        trust_remote_code=False,
        local_files_only=True,
        weights_only=True,
        dtype=torch.float32,
    ).eval()
    reference = []
    with torch.inference_mode():
        for x in ids:
            mask = torch.ones(1, len(x), dtype=torch.long)
            out = model(input_ids=torch.tensor([x]), attention_mask=mask)
            cls = out.last_hidden_state[0, 0]  # CLS pooling (1_Pooling/config.json)
            reference.append(torch.nn.functional.normalize(cls, dim=0).tolist())
    with running(tmp_path) as embedder:
        shipped = [floats(embedder.vector(x)) for x in ids]

    def cosine(a: Iterable[float], b: Iterable[float]) -> float:
        x, y = list(a), list(b)
        dot = math.fsum(i * j for i, j in zip(x, y, strict=True))
        return dot / math.sqrt(math.fsum(i * i for i in x) * math.fsum(j * j for j in y))

    cosines = [cosine(r, s) for r, s in zip(reference, shipped, strict=True)]
    print(
        f"\nreference cosines: lowest {min(cosines):.6f}, by text {[f'{c:.5f}' for c in cosines]}"
    )
    low = [(i, len(ids[i]), round(c, 6)) for i, c in enumerate(cosines) if c < 0.999]
    assert not low, f"text, tokens and cosine under 0.999 against the source model: {low}"

    passages = texts["passages"]
    wrong = []
    for q in range(passages, len(ids)):
        tops = []
        for vectors in (shipped, reference):
            scores = sorted(
                ((cosine(vectors[q], vectors[p]), p) for p in range(passages)), reverse=True
            )
            tops.append(scores)
        if tops[0][0][1] != tops[1][0][1]:
            tie = any(s[0][0] - s[1][0] < 1e-3 for s in tops)
            line = f"query {q}: top passage {tops[0][0][1]} shipped, {tops[1][0][1]} reference"
            if tie:
                print(f"\n{line} (a near-tie, within 1e-3)")
            else:
                wrong.append(line)
    assert not wrong, f"a query's top passage differs from the source model's: {wrong}"
