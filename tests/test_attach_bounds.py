"""The security review's must 1 (v0.3.1): counting an attachment is bounded before any
whole render. A token covers at most L bytes of text (L: the tokenizer's longest token), so
a text of B bytes takes at least B / L tokens; when that alone is more than is free, the
document doesn't fit and is never rendered whole. Made-up documents only."""

from __future__ import annotations

import hashlib
import math
import threading
import time

import pytest

from clarity_chat.commands import Commands
from clarity_chat.documents import Document, DocumentRefused, extract, reading
from clarity_chat.documents.child import lines
from tests.pdfs import shared_stream_pdf
from tests.test_session_attach import chat_session

MADE_UP = "Made-up line {n}: a ridge path, three cairns, no water.\n"


def text_doc(text: str, name: str = "big.md") -> Document:
    data = text.encode("utf-8")
    return Document(
        name=name,
        kind="text",
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        extractor={"name": "clarity-text", "version": 1, "bom_removed": False},
        unit="line",
        units=tuple(lines(text)),
        empty_units=0,
    )


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


def test_the_longest_token_is_counted_once_per_tokenizer(chat_tokenizer, olmo_tokenizer):
    assert chat_tokenizer.max_token_bytes == len("</s>")  # one byte a token, and <s>, </s>
    lengths = [
        len(olmo_tokenizer.decode([i]).encode("utf-8")) for i in range(len(olmo_tokenizer.tok))
    ]
    assert olmo_tokenizer.max_token_bytes >= max(lengths)


@pytest.mark.parametrize(
    "text",
    ["=" * 5000, " " * 5000, "-" * 3000 + "\n" * 3000, MADE_UP.format(n=1) * 200, "日本語" * 500],
)
def test_no_text_takes_fewer_tokens_than_its_bytes_over_the_longest_token(olmo_tokenizer, text):
    """The bound's premise, on OLMo 3's real tokenizer, with the long runs it merges most."""
    tokens = len(olmo_tokenizer.encode(text))
    assert tokens >= math.ceil(len(text.encode("utf-8")) / olmo_tokenizer.max_token_bytes)


def test_a_huge_text_is_refused_by_its_bound_without_a_whole_render(chat):
    session, _ = chat
    doc = text_doc("".join(MADE_UP.format(n=n) for n in range(600_000)))  # about 33 MB
    started = time.monotonic()
    check = session.check(doc)
    assert time.monotonic() - started < 30
    assert check["tokens_whole"] is None and check["left_whole"] is None
    assert check["tokens_whole_at_least"] > check["free"] and check["fits"] is False
    assert check["start_that_fits"]["range"][1] > 0  # the start is still found, exactly
    with pytest.raises(DocumentRefused) as e:
        session.attach(doc)
    assert str(e.value).startswith(
        f"big.md doesn't fit: it needs more than {check['tokens_whole_at_least'] - 1:,} tokens "
        f"and {check['free']:,} are free in this chat."
    )


def test_the_bound_never_passes_the_exact_count(chat):
    """Near the boundary: whenever the bound refuses, the exact count would have too."""
    session, backend = chat
    backend._n_ctx = 4096
    for n in range(100, 1200, 100):
        doc = text_doc("".join(MADE_UP.format(n=i) for i in range(n)))
        check = session.check(doc)
        room = session.room()
        exact = room.tokens(doc_content(doc)) - room.tokens("")
        if check["tokens_whole"] is None:
            assert check["tokens_whole_at_least"] <= exact and exact > check["free"]
        else:
            assert check["tokens_whole"] == exact and check["tokens_whole_at_least"] is None


def doc_content(doc: Document) -> str:
    from clarity_chat.documents import FRAMING, Attached, content

    return content("", [Attached(doc.name, doc.text, None)], FRAMING)


def test_an_amplifying_pdf_is_checked_within_a_bound(chat, tmp_path):
    """The review's shared-stream PDF: a 7 KB file, 10 M characters of text."""
    session, _ = chat
    path = tmp_path / "shared.pdf"
    path.write_bytes(shared_stream_pdf(40, 256 * 1024))
    started = time.monotonic()
    check = Commands(session).run("attach", f"{path} check", lambda e: None)["check"]
    assert time.monotonic() - started < 60
    assert check["total"] == 40 and check["tokens_whole"] is None
    assert check["tokens_whole_at_least"] > check["free"]


@pytest.mark.parametrize("kind", ["text", "pdf"])
def test_counting_stops_when_cancelled(chat, kind):
    session, _ = chat
    if kind == "pdf":
        doc = extract(shared_stream_pdf(8, 64 * 1024), "shared.pdf")
    else:
        doc = text_doc("".join(MADE_UP.format(n=n) for n in range(50_000)))
    cancel = threading.Event()
    cancel.set()
    for count in (session.check, session.attach):
        with pytest.raises(DocumentRefused) as e:
            count(doc, cancel=cancel)
        assert str(e.value) == f"Stopped counting {doc.name}."
    assert session.conversation.pending == []


def test_cancel_reaches_a_check_that_s_counting(chat, monkeypatch):
    """/cancel during a check: the next render sees it."""
    session, _ = chat
    doc = text_doc("".join(MADE_UP.format(n=n) for n in range(5_000)))
    cancel = threading.Event()
    real = session.room

    def room(*args, **kwargs):
        made = real(*args, **kwargs)
        counted = [0]

        def tokens(text):
            counted[0] += 1
            if counted[0] == 3:
                cancel.set()
            return made.tokens(text)

        return type(made)(**{**made.__dict__, "tokens": tokens})

    monkeypatch.setattr(session, "room", room)
    with pytest.raises(DocumentRefused) as e:
        session.check(doc, cancel=cancel)
    assert str(e.value) == "Stopped counting big.md."


def test_nothing_reads_a_document_s_text_more_than_linearly_before_the_bound(chat):
    """Must 1's last part: the parent's own work on a long text, before the bound check, is
    a few passes over it. 4x the text takes about 4x the time, not 16x."""
    session, backend = chat
    backend._n_ctx = 512
    small = text_doc("".join(MADE_UP.format(n=n) for n in range(50_000)))
    large = text_doc("".join(MADE_UP.format(n=n) for n in range(200_000)))
    times = []
    for doc in (small, large):
        started = time.monotonic()
        session.check(doc)
        times.append(time.monotonic() - started)
    assert times[1] < 10 * max(times[0], 0.05)
