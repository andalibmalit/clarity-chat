"""v0.3.1 A1: what Clarity reads from a file you attach. Made-up text and documents only."""

from __future__ import annotations

import os
import platform
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


SPLIT = " Attach a smaller file, or split it and attach the part you need."  # F11


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
    assert doc.extractor == {
        "name": "clarity-html",
        "version": 1,
        "python": platform.python_version(),
    }


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
    assert refused(make_pdf(["", ""]), "scan.pdf") == (  # fact-check row 36
        "scan.pdf has no text Clarity can read. Its pages may be scanned images, which "
        "Clarity doesn't read. Use a PDF app's text recognition (OCR) to add the text, then "
        "attach that copy."
    )


@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.4\nnot a pdf at all", make_pdf(["Made-up page."])[:120], b"%PDF-" + bytes(500)],
)
def test_a_broken_pdf_is_refused_with_the_error_s_type(data):
    sentence = refused(data, "broken.pdf")  # fact-check row 42: what to do
    assert sentence.startswith("Clarity couldn't read broken.pdf (")
    assert sentence.endswith("). The file may be damaged: save a fresh copy and attach that.")


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
    """The child replaced by `code`, run with the real child's own interpreter and flags
    (the real command's prefix), so a test of `-I` tests the command that ships."""
    monkeypatch.setattr(reading, "CHILD", [*reading.CHILD[:-2], "-c", code])


def gone(pid: int) -> bool:
    """Whether process `pid` has ended. Windows has no signal 0: its os.kill ends the process
    (and a dead one raises WinError 87), so ask for the exit code there instead."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return True
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value != 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
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
    assert refused(b"Made-up.", "slow.md") == (
        "slow.md took longer than 1 seconds to read." + SPLIT
    )
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


@pytest.mark.skipif(sys.platform == "win32", reason="no memory guard on Windows (v0.7.0)")
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
    # The board's ruling 5: "1 GB", not "1024 MB".
    assert refused(b"Made-up.", "huge.pdf") == (
        "huge.pdf took more than 1 GB of memory to read." + SPLIT
    )


def test_the_child_s_own_deadline_ends_it_with_no_parent_watching():
    code = "from clarity_chat.documents.child import watch\nwatch(1 << 40, 0.5)\nimport time\n"
    code += "time.sleep(60)"
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-I", "-c", code], timeout=30)
    assert done.returncode == child.DEADLINE_EXIT and time.monotonic() - started < 10


def test_output_past_the_cap_is_refused(monkeypatch):
    stub(monkeypatch, "import sys; sys.stdout.write('x' * 5_000_000)")
    monkeypatch.setattr(reading, "OUTPUT_CAP", 2 * 1024 * 1024)
    assert refused(b"Made-up.", "long.md") == (
        "long.md is too long to read: what its reader sent back passed 2 MB."
    )


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


def test_a_file_over_64_mb_is_refused_before_it_s_read(tmp_path, monkeypatch):
    big = tmp_path / "big.bin"
    with big.open("wb") as f:
        f.truncate(64 * 1024 * 1024 + 1)  # sparse: nothing written

    def no_reading(*args, **kwargs):  # refused from its size alone, before any read
        raise AssertionError("read")

    monkeypatch.setattr(os, "fdopen", no_reading)
    with pytest.raises(DocumentRefused) as e:
        read_path(str(big))
    assert str(e.value) == (  # fact-check row 39: 64 MiB, so the bytes too
        "big.bin is too big: Clarity reads files up to 64 MB (67,108,864 bytes)."
    )


def test_a_directory_or_a_fifo_is_refused_without_blocking(tmp_path):
    paths = [tmp_path]
    if hasattr(os, "mkfifo"):  # not on Windows, which has no FIFOs in the file system
        fifo = tmp_path / "pipe.md"
        os.mkfifo(fifo)
        paths.append(fifo)
    for path in paths:
        started = time.monotonic()
        with pytest.raises(DocumentRefused) as e:
            read_path(str(path))
        assert str(e.value) == f"{path.name} isn't a file Clarity can attach."
        assert time.monotonic() - started < 2


def test_a_directory_the_open_refuses_gets_the_same_sentence(tmp_path, monkeypatch):
    """Windows refuses to open a folder (Permission denied) where POSIX opens it and fstat
    says it isn't a file: the same sentence either way (CI 43754075636)."""

    def windows_open(path, flags, *args):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(os, "open", windows_open)
    with pytest.raises(DocumentRefused) as e:
        read_path(str(tmp_path))
    assert str(e.value) == f"{tmp_path.name} isn't a file Clarity can attach."


def test_a_path_s_name_is_kept_never_the_path(tmp_path):
    (tmp_path / "notes.md").write_text("Made-up.", encoding="utf-8")
    assert read_path(str(tmp_path / "notes.md")) == (b"Made-up.", "notes.md")


@pytest.mark.parametrize(
    "name",
    [
        "two\nlines.md",
        "tab\there.md",
        "bell\x07.md",
        "x" * 256,
        "notes\udcff.md",  # a lone surrogate: a Linux file name that isn't UTF-8, or --json
        "a\u2028b.md",  # line and paragraph separators act as newlines
        "a\u2029b.md",
    ],
)
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


# --- Checkpoint 1's findings


@pytest.mark.parametrize("order", [("page.html", "page.txt"), ("page.txt", "page.html")])
def test_the_same_bytes_under_another_kind_are_read_by_that_kind_s_rules(order):
    """Must 1: the cache matched the bytes alone, so a .txt read after the same bytes as
    .html came back with its tags stripped: what was sent depended on earlier attachments."""
    data = b"<p>Raw <b>source</b></p>\n"
    expected = {"page.html": ("html", "Raw source\n"), "page.txt": ("text", data.decode())}
    for name in order:
        doc = extract(data, name)
        assert (doc.kind, doc.text) == expected[name], name


def test_the_real_child_gets_the_memory_cap_and_a_deadline_past_the_time_limit(monkeypatch):
    """Must 2: the limits the real command sends, echoed back by a stub."""
    stub(monkeypatch, "import sys; sys.stdout.write(sys.stdin.buffer.readline().decode())")
    answer = reading.run_child({"name": "x.md", "kind": "text"}, b"", None)
    assert answer["memory_cap"] == reading.MEMORY_CAP
    assert reading.TIME_LIMIT < answer["seconds"] <= reading.TIME_LIMIT + 10


@pytest.mark.skipif(sys.platform == "win32", reason="no memory guard on Windows (v0.7.0)")
def test_the_real_child_stops_at_its_memory_cap(monkeypatch):
    """Must 2: the real child, at a 1 MB cap, which the interpreter alone passes. The same
    refusal on Linux, where RLIMIT_AS at twice the cap fails the child's next allocation
    (a MemoryError, or a thread it can't start), and on macOS, where the watchdog sees it."""
    monkeypatch.setattr(reading, "MEMORY_CAP", 1024 * 1024)
    assert refused(b"Made-up.", "notes.md") == (
        "notes.md took more than 1 MB of memory to read." + SPLIT
    )


def test_memory_clarity_holds_doesn_t_count_against_the_child(monkeypatch):
    """Should 5: on Linux a new process's peak starts at its parent's, so once Clarity had
    held over the cap (a Transformers model), every file was refused. Here the parent holds
    300 MB, written, against a 200 MB cap, and a small file reads. Decided on Linux CI."""
    held = b"x" * (300 * 1024 * 1024)
    monkeypatch.setattr(reading, "MEMORY_CAP", 200 * 1024 * 1024)
    assert extract(b"Made-up.", "notes.md").text == "Made-up." and len(held)


def test_an_interrupted_wait_leaves_no_child(monkeypatch, tmp_path):
    """Should 3: Ctrl-C in the terminal during "Reading NAME…" left the child running."""
    pidfile = tmp_path / "pid"
    stub(monkeypatch, f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); "
         "time.sleep(60)")  # fmt: skip

    def interrupted(seconds):
        while not pidfile.exists() or not pidfile.read_text():
            threading.Event().wait(0.01)
        raise KeyboardInterrupt

    monkeypatch.setattr(reading.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        extract(b"Made-up.", "slow.md")
    monkeypatch.undo()
    assert gone(int(pidfile.read_text()))


def test_a_reader_that_can_t_start_is_a_refusal(monkeypatch, tmp_path):
    """Should 3: an OSError from starting the child came out as itself, which stops the
    session on the page."""
    monkeypatch.setattr(reading, "CHILD", [str(tmp_path / "no-such-python")])
    assert refused(b"Made-up.", "notes.md").startswith("Clarity couldn't start its reader: ")


@pytest.mark.parametrize("version", [5, 4])
@pytest.mark.parametrize("cryptography", ["absent", "present"])
def test_aes_is_refused_from_the_dictionary_with_or_without_a_provider(
    version, cryptography, monkeypatch
):
    """Should 4: AES (V5, and V4 with /AESV2) is refused by Clarity's own check of the
    encryption dictionary. "present" stands for a crypto provider that would open it:
    pypdf's own encryption step made a no-op, so only Clarity's check can refuse."""
    import pypdf

    if cryptography == "present":
        monkeypatch.setattr(pypdf.PdfReader, "_handle_encryption", lambda self, password: None)
    with pytest.raises(child.Refused) as e:
        child.extract("pdf", aes_pdf(["Protected."], version=version), "aes.pdf")
    assert str(e.value) == (
        "aes.pdf is protected, even if it opens without a password. Export or print it to a "
        "new PDF and attach that."
    )


def test_a_pdf_with_no_pages_says_so():
    assert refused(make_pdf([]), "blank.pdf") == "blank.pdf has no pages."


def test_html_without_its_closing_head_tag_reads_its_body():
    """Should 7: omitting </head> is valid HTML5, and skipping all of head left nothing.
    Version 1 skips title instead: head holds no other text."""
    page = b"<html><head><title>Not shown</title><body><p>Body text.</p>"
    assert extract(page, "b.html").text == "Body text.\n"


def test_html_whitespace_collapses_across_inline_tags():
    """Should 8: the line is collapsed when it ends, not piece by piece."""
    assert extract(b"<p>Three <b> cairns</b> here</p>", "c.html").text == "Three cairns here\n"


def test_pypdf_gets_clarity_s_root_object_recovery_limit(monkeypatch):
    """Checkpoint 1's mutation run: not passing the limit (unbounded) survived, since only
    a huge broken file would show it. pypdf's default today is 10,000; Clarity sets it, so
    a moved default can't widen it."""
    import pypdf

    from clarity_chat.documents import child

    seen = {}
    real = pypdf.PdfReader

    def reader(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pypdf, "PdfReader", reader)
    child.pdf_pages(make_pdf(["Made-up page."]), "made-up.pdf")
    assert seen["root_object_recovery_limit"] == child.ROOT_RECOVERY_LIMIT == 10_000


# --- the security review's should 1: every error reading a path is a refusal


def test_a_read_that_fails_after_the_open_is_refused(tmp_path, monkeypatch):
    """EIO from a failing disk or mount, after the open and fstat passed."""
    path = tmp_path / "notes.md"
    path.write_text("Made-up.", encoding="utf-8")

    class Failing:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fdopen", Failing)
    with pytest.raises(DocumentRefused) as e:
        read_path(str(path))
    assert str(e.value) == "Can't read notes.md: Input/output error."


def test_a_lone_surrogate_is_refused_where_the_system_would_look_for_it(monkeypatch):
    """Windows' file names are UTF-16, so its open takes a lone surrogate and answers "No
    such file"; POSIX can't encode one. The same sentence either way (CI 43754075636)."""

    def windows_open(path, flags, *args):
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(os, "open", windows_open)
    with pytest.raises(DocumentRefused) as e:
        read_path("/made-up/\ud800/notes.md")
    assert str(e.value) == "Can't open notes.md: its path holds a character the system can't take."


@pytest.mark.parametrize("path", ["made-up\x00/notes.md", "/made-up/\ud800/notes.md"])
def test_a_path_the_system_can_t_take_is_refused(path):
    """A NUL or a lone surrogate in a folder's name (from --json): the name check sees only
    the last part, and the open raised ValueError or UnicodeEncodeError."""
    with pytest.raises(DocumentRefused) as e:
        read_path(path)
    assert str(e.value) == "Can't open notes.md: its path holds a character the system can't take."


# --- the security review's should 5: names


@pytest.mark.parametrize(
    "mark", ["؜", "‎", "‏", "‪", "‫", "‬", "‭", "‮",
             "⁦", "⁧", "⁨", "⁩"],
)  # fmt: skip
def test_a_name_with_a_text_direction_control_is_refused(mark):
    """notes‮dm.txt shows as notes‮dm.txt: reversed, in the chips, the sheet and the
    terminal."""
    with pytest.raises(DocumentRefused) as e:
        checked_name(f"notes{mark}dm.txt")
    assert str(e.value) == (
        "A file name can't hold a newline, a control character, a text-direction control or a "
        "broken character."
    )


@pytest.mark.parametrize("name", ["نامه‌ها.md", "notes 👩‍💻.md"])
def test_joiners_stay_allowed(name):
    """U+200C is common in Persian names; U+200D joins emoji."""
    assert checked_name(name) == name


def test_a_path_s_bytes_are_read_exactly(tmp_path):
    """The security review's could 6: on Windows, os.open without O_BINARY reads in text
    mode, turning CRLF into LF and stopping at 0x1A."""
    path = tmp_path / "notes.md"
    path.write_bytes(b"a\r\nb\x1ac")
    assert read_path(str(path)) == (b"a\r\nb\x1ac", "notes.md")


@pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows")
def test_the_reader_runs_in_its_own_process_group(monkeypatch):
    """The security review's could 1: in Clarity's group, a terminal's Ctrl-C reached the
    child too, and the cancel read as "its reader stopped (exit -2)"."""
    stub(
        monkeypatch, "import os, json, sys; sys.stdout.write(json.dumps({'group': os.getpgid(0)}))"
    )
    answer = reading.run_child({"name": "x.md", "kind": "text"}, b"", None)
    if sys.platform != "win32":  # for ty on Windows, which the skip doesn't narrow
        assert answer["group"] != os.getpgrp()


def test_html_s_record_names_the_python_that_read_it():
    """Fact-check, design.md's row: HTML is read with Python's own html.parser, which has
    changed between Python releases, so the record says which one ran."""
    import platform

    doc = extract(b"<p>Made-up.</p>", "notes.html")
    assert doc.extractor == {
        "name": "clarity-html",
        "version": 1,
        "python": platform.python_version(),
    }


# --- CI at a28f1c4: a MemoryError under Linux's address-space limit exited 1


@pytest.mark.parametrize(
    "where",
    [
        "child.extract = lambda *a: (_ for _ in ()).throw(MemoryError())",  # while reading
        "child.json.dumps = lambda *a, **k: (_ for _ in ()).throw(MemoryError())",  # writing
    ],
)
def test_a_memory_error_in_the_child_is_the_memory_refusal(monkeypatch, where):
    stub(monkeypatch, f"from clarity_chat.documents import child\n{where}\nchild.main()")
    assert refused(b"Made-up.", "notes.md").startswith(
        "notes.md took more than 1 GB of memory to read."
    )


@pytest.mark.skipif(sys.platform == "win32", reason="no memory guard on Windows (v0.7.0)")
def test_a_thread_the_child_can_t_start_under_its_limit_is_the_memory_refusal(monkeypatch):
    """At Linux's 2 MB address-space limit (the 1 MB test's cap), the watchdog's own thread
    can fail to start. Simulated here, as on Linux: the limit set, then the start fails."""
    code = (
        "import threading\n"
        "from clarity_chat.documents import child\n"
        "child.limit_address_space = lambda cap: True\n"
        'def fail(self): raise RuntimeError("can\'t start new thread")\n'
        "threading.Thread.start = fail\n"
        "child.main()"
    )
    stub(monkeypatch, code)
    assert refused(b"Made-up.", "notes.md").startswith(
        "notes.md took more than 1 GB of memory to read."
    )
