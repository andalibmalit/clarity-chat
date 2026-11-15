"""v0.3.1 A1: what Clarity reads from a file you attach. Made-up text and documents only."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from clarity_chat.documents import (
    Document,
    DocumentRefused,
    checked_name,
    child,
    extract,
    read_path,
    reading,
)
from tests.pdfs import aes_pdf, encrypted_pdf, make_pdf

FIXTURES = Path(__file__).parent / "fixtures" / "documents"
SAVE_AS = "Save it as PDF or as UTF-8 text and attach that."


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    """Each test reads its files anew: the last document read is kept between calls."""
    monkeypatch.setattr(reading, "_last", None)


def refused(data: bytes, name: str) -> str:
    with pytest.raises(DocumentRefused) as e:
        extract(data, name)
    return str(e.value)


# --- text


def test_text_is_the_decoded_bytes_exactly():
    data = b"Ridge path\r\n\tcairns: 3\nwater: none\n"
    doc = extract(data, "trail-notes.md")
    assert (doc.kind, doc.unit, doc.text) == ("text", "line", data.decode())
    assert doc.units == ("Ridge path\r\n", "\tcairns: 3\n", "water: none\n")  # "\n" only
    assert doc.extractor == {"name": "clarity-text", "version": 1, "bom_removed": False}
    assert doc.empty_units == 0 and doc.bytes == len(data)


def test_a_leading_byte_order_mark_is_removed_and_recorded():
    doc = extract("﻿Made-up notes.".encode(), "notes.txt")
    assert doc.text == "Made-up notes." and doc.extractor["bom_removed"] is True


@pytest.mark.parametrize("data", [b"\xff\xfeA\x00B\x00", b"plain\x00text"])
def test_bytes_that_arent_utf8_text_are_refused_with_what_to_do(data):
    assert refused(data, "data.bin").endswith(SAVE_AS)


def test_an_empty_file_is_refused():
    assert refused(b"", "empty.md") == "empty.md is empty."


@pytest.mark.parametrize(
    ("name", "kind"),
    [("page.HTM", "html"), ("Page.Html", "html"), ("notes.md", "text"), ("notes.PDF", "text")],
)
def test_html_by_its_name_s_ending_in_any_case(name, kind):
    assert extract(b"<p>Made up.</p>", name).kind == kind


def test_a_pdf_is_a_pdf_by_its_first_bytes_whatever_its_name():
    assert extract(make_pdf(["Made-up page."]), "notes.txt").kind == "pdf"


def test_a_pdf_name_without_pdf_bytes_is_read_as_text_and_refused_if_not_utf8():
    assert refused(b"\x89not a pdf\xff", "guide.PDF").endswith(SAVE_AS)


# --- HTML, version 1 of the rules


def test_html_by_each_rule_of_version_1():
    page = (
        "<html><head><title>Not shown</title><style>p {}</style></head><body>"
        "<script>alert(1)</script><template><p>no</p></template><noscript>no</noscript>"
        "<h1>The   ridge\n  path</h1><p>Three <b>cairns</b> &amp; a&nbsp;saddle.</p>"
        "<ul><li>water</li><li>map</li></ul>text after<br>a break"
        "<pre>  kept   as\n    written</pre><div><div>nested</div></div></body></html>"
    )
    doc = extract(page.encode(), "page.html")
    assert doc.text == (
        "The ridge path\nThree cairns & a\xa0saddle.\nwater\nmap\ntext after\na break\n"
        "  kept   as\n    written\nnested\n"
    )
    assert doc.extractor == {"name": "clarity-html", "version": 1}


# --- PDF


def test_a_pdf_s_pages_and_an_image_only_page():
    doc = extract(make_pdf(["First page.\nSecond line.", "", "Third page."]), "guide.pdf")
    assert (doc.unit, doc.total, doc.empty_units) == ("page", 3, 1)
    assert doc.units == ("First page.\nSecond line.", "", "Third page.")
    assert doc.text == "First page.\nSecond line.\n\n\n\nThird page."  # a blank line apart
    assert doc.extractor == {
        "name": "pypdf",
        "version": "7.1.0",
        "mode": "plain",
        "encrypted": False,
    }


def test_a_pdf_with_no_text_at_all_is_refused():
    assert refused(make_pdf(["", ""]), "scan.pdf") == (
        "scan.pdf has no text to read: its pages are images."
    )


@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.4\nnot a pdf at all", make_pdf(["Made-up page."])[:120], b"%PDF-" + bytes(500)],
)
def test_a_broken_pdf_is_refused_with_the_error_s_type(data):
    sentence = refused(data, "broken.pdf")
    assert sentence.startswith("Clarity couldn't read broken.pdf (") and sentence.endswith(").")


def test_a_real_pdf_reads_to_its_known_text():
    """Made by macOS's own PDF producer (cupsfilter) from made-up text: a pypdf bump that
    changes how real files read fails here."""
    data = (FIXTURES / "trail-notes.pdf").read_bytes()
    doc = extract(data, "trail-notes.pdf")
    assert doc.text.startswith("Trail notes, made up for a test.")
    assert doc.extracted_sha256 == (
        "e6d8d6f2afac59b73fdbae6f025c1441dd09f8c615a52d56afa95a2d9dd46480"
    )


def test_a_pdf_that_opens_with_an_empty_password_is_read_and_says_encrypted():
    doc = extract(encrypted_pdf(["Open with no password."]), "rc4.pdf")
    assert doc.text == "Open with no password." and doc.extractor["encrypted"] is True


def test_a_pdf_that_needs_a_password_is_refused():
    assert refused(encrypted_pdf(["Locked."], user_password="made-up"), "locked.pdf") == (
        "locked.pdf needs a password. Save a copy without one and attach that."
    )


@pytest.mark.parametrize("cryptography", ["absent", "present"])
def test_an_aes_pdf_is_refused_whether_or_not_cryptography_is_installed(cryptography, monkeypatch):
    """Clarity reads the encryption dictionary and refuses AES itself: another package
    having installed `cryptography` (which would let pypdf open it) changes nothing."""
    import types

    if cryptography == "present":
        monkeypatch.setitem(sys.modules, "cryptography", types.ModuleType("cryptography"))
    else:
        monkeypatch.setitem(sys.modules, "cryptography", None)  # import fails
    with pytest.raises(child.Refused) as e:
        child.extract("pdf", aes_pdf(["Protected."]), "aes.pdf")
    assert str(e.value) == "aes.pdf is protected. Save an unprotected copy and attach that."
    assert refused(aes_pdf(["Protected."]), "aes.pdf") == str(e.value)  # through the child


@pytest.mark.parametrize("text", ["a\x00b", "a\ud800b"])
def test_text_that_can_t_travel_is_refused_not_replaced(text, monkeypatch):
    monkeypatch.setattr(child, "pdf_pages", lambda data, name: ([text], {"name": "pypdf"}))
    with pytest.raises(child.Refused) as e:
        child.extract("pdf", b"%PDF-", "odd.pdf")
    assert str(e.value) == (
        "odd.pdf holds characters that can't be sent as text (a NUL or a lone surrogate)."
    )


# --- the child and its limits


def stub(monkeypatch, code: str) -> None:
    """The child replaced by `code`, run as the real child is (`python -I`)."""
    monkeypatch.setattr(reading, "CHILD", [sys.executable, "-I", "-c", code])


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_the_time_limit_stops_the_child_and_leaves_none(monkeypatch, tmp_path):
    pidfile = tmp_path / "pid"
    stub(monkeypatch, f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); "
         "time.sleep(60)")  # fmt: skip
    monkeypatch.setattr(reading, "TIME_LIMIT", 1.0)
    started = time.monotonic()
    assert refused(b"Made-up.", "slow.md") == "slow.md took longer than 1 seconds to read."
    assert time.monotonic() - started < 10 and gone(int(pidfile.read_text()))


def test_cancel_stops_the_child_and_leaves_none(monkeypatch, tmp_path):
    pidfile = tmp_path / "pid"
    stub(monkeypatch, f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); "
         "time.sleep(60)")  # fmt: skip
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with pytest.raises(DocumentRefused) as e:
        extract(b"Made-up.", "slow.md", cancel=cancel)
    assert str(e.value) == "Stopped reading slow.md." and gone(int(pidfile.read_text()))


def test_the_memory_watchdog_ends_the_child(monkeypatch):
    """A stub that allocates 10 MB at a time: the watchdog ends it past its cap (here
    200 MB), and the parent says so. Bounded, so a broken watchdog can't exhaust memory."""
    stub(
        monkeypatch,
        "from clarity_chat.documents.child import watch\n"
        "watch(200 * 1024 * 1024, 30)\n"
        "held = []\n"
        'for _ in range(60): held.append(b"x" * (10 * 1024 * 1024))',  # written: resident
    )
    assert refused(b"Made-up.", "huge.pdf") == "huge.pdf took more than 1024 MB of memory to read."


def test_the_child_s_own_deadline_ends_it_with_no_parent_watching():
    code = "from clarity_chat.documents.child import watch\nwatch(1 << 40, 0.5)\nimport time\n"
    code += "time.sleep(60)"
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-I", "-c", code], timeout=30)
    assert done.returncode == child.DEADLINE_EXIT and time.monotonic() - started < 10


def test_output_past_the_cap_is_refused(monkeypatch):
    stub(monkeypatch, "import sys; sys.stdout.write('x' * 5_000_000)")
    monkeypatch.setattr(reading, "OUTPUT_CAP", 1_000_000)
    assert refused(b"Made-up.", "long.md") == "long.md's text is too long: more than 0 MB."


def test_the_child_runs_isolated_with_no_secrets_in_its_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "made-up-token")
    stub(
        monkeypatch,
        "import json, os, sys; print(json.dumps({'isolated': sys.flags.isolated,"
        " 'env': dict(os.environ)}))",
    )
    answer = reading.run_child({"name": "x.md", "kind": "text"}, b"", None)
    assert answer["isolated"] == 1 and "HF_TOKEN" not in answer["env"]


def test_the_exit_codes_match_the_child_s():
    assert (reading.MEMORY_EXIT, reading.DEADLINE_EXIT) == (
        child.MEMORY_EXIT,
        child.DEADLINE_EXIT,
    )


# --- paths and names


def test_a_file_over_64_mb_is_refused_before_it_s_read(tmp_path):
    big = tmp_path / "big.bin"
    with big.open("wb") as f:
        f.truncate(64 * 1024 * 1024 + 1)  # sparse: nothing written
    with pytest.raises(DocumentRefused) as e:
        read_path(str(big))
    assert str(e.value) == "big.bin is too big: Clarity reads files of at most 64 MB."


def test_a_directory_or_a_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pipe.md"
    os.mkfifo(fifo)
    for path in (tmp_path, fifo):
        started = time.monotonic()
        with pytest.raises(DocumentRefused) as e:
            read_path(str(path))
        assert str(e.value) == f"{path.name} isn't a file Clarity can attach."
        assert time.monotonic() - started < 2


def test_a_path_s_name_is_kept_never_the_path(tmp_path):
    (tmp_path / "notes.md").write_text("Made-up.", encoding="utf-8")
    assert read_path(str(tmp_path / "notes.md")) == (b"Made-up.", "notes.md")


@pytest.mark.parametrize("name", ["two\nlines.md", "tab\there.md", "bell\x07.md", "x" * 256])
def test_a_name_with_a_control_character_or_too_long_is_refused(name):
    with pytest.raises(DocumentRefused):
        checked_name(name)


# --- exactness and the cache


def test_the_same_bytes_give_the_same_text_twice_and_the_second_reads_nothing(monkeypatch):
    data = make_pdf(["Made-up page one.", "Made-up page two."])
    first = extract(data, "guide.pdf")
    stub(monkeypatch, "raise SystemExit(3)")  # a second child would fail
    again = extract(data, "copy.pdf")
    assert (again.text, again.extracted_sha256) == (first.text, first.extracted_sha256)
    assert again.name == "copy.pdf"
    monkeypatch.setattr(reading, "_last", None)
    monkeypatch.undo()
    third = extract(data, "guide.pdf")
    assert (third.text, third.extracted_sha256) == (first.text, first.extracted_sha256)


def test_every_failure_is_a_refusal_never_another_exception():
    """M5: whatever a hostile file does to the reader, the caller gets a DocumentRefused."""
    real = make_pdf(["Made-up page."] * 3)
    cases = [real[:n] for n in range(9, len(real), 37)] + [real.replace(b"obj", b"jbo")]
    for data in cases:
        try:
            extract(data, "hostile.pdf")
        except DocumentRefused:
            pass
        reading._last = None


def test_a_thousand_page_pdf_reads_within_a_loose_bound():
    data = make_pdf([f"Made-up page {n}.\nA second line." for n in range(1000)])
    started = time.monotonic()
    doc = extract(data, "long.pdf")
    seconds = time.monotonic() - started
    assert doc.total == 1000 and seconds < 60, seconds
    print(f"1,000 pages in {seconds:.1f} s: {1000 / seconds:.0f} pages a second")


def test_a_document_s_units_join_into_its_text():
    doc = Document(
        name="n.md",
        kind="text",
        bytes=0,
        sha256="",
        extractor={},
        unit="line",
        units=("a\n", "b\n", "c"),
        empty_units=0,
    )
    assert doc.text == "a\nb\nc" and doc.text_of(1, 2) + doc.text_of(3, 3) == doc.text
