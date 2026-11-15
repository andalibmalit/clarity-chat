"""Extraction's child process: `python -I -m clarity_chat.documents.child`.

It reads a request line (JSON: the file's name and kind) and then the file's bytes on stdin,
extracts the text, and writes one JSON object on stdout: the document's units and facts, or
`{"refused": SENTENCE}`. A watchdog thread ends it past its memory cap or its own deadline,
so a parser stuck on a hostile file can't freeze Clarity or outlive it (v0.3.1 decision 13).
"""

from __future__ import annotations

import io
import json
import os
import platform
import re
import sys
import threading
import time
from html.parser import HTMLParser
from typing import Any

try:
    import resource
except ImportError:  # Windows: no memory guard until v0.7.0; the deadline still holds
    resource = None

MEMORY_EXIT = 70  # the watchdog's exit codes, which the parent turns into sentences
DEADLINE_EXIT = 71
ROOT_RECOVERY_LIMIT = 10_000  # pypdf's default, set here so a moved default can't widen it


def memory_used() -> int:
    """This process's resident memory, in bytes. On Linux, the current figure from
    /proc/self/statm: a new process's peak (getrusage) starts at its parent's, so Clarity's
    own memory would count. On macOS, the peak, which is the process's own. 0 where there's
    no way to tell (Windows)."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if resource is None:
        return 0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def watch(memory_cap: int, seconds: float, interval: float = 0.02) -> None:
    """End this process past `memory_cap` bytes or `seconds`. Polling lets memory overshoot by
    what's allocated between polls; on Linux, RLIMIT_AS stops the process at twice the cap
    as well."""
    deadline = time.monotonic() + seconds
    if resource is not None and sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (memory_cap * 2, memory_cap * 2))

    def loop() -> None:
        while True:
            if memory_used() > memory_cap:
                os._exit(MEMORY_EXIT)
            if time.monotonic() > deadline:
                os._exit(DEADLINE_EXIT)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


class Refused(Exception):
    """A sentence for the user: what's wrong with the file, and what to do."""


def utf8(data: bytes, name: str) -> tuple[str, bool]:
    """The bytes as UTF-8 text, and whether a leading byte-order mark was removed."""
    if b"\x00" in data:
        raise Refused(f"{name} isn't text Clarity can read. {SAVE_AS}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused(f"{name} isn't UTF-8 text. {SAVE_AS}") from None
    bom = text.startswith("﻿")
    return (text[1:] if bom else text), bom


SAVE_AS = "Save it as PDF or as UTF-8 text and attach that."


# HTML, version 1 of Clarity's rules (listed in the plan; reference.md lists them with B1).
HTML_SPACE = " \t\n\r\f"  # HTML's whitespace: a no-break space is kept, as text
# title, not head: a page may omit </head> (valid HTML5), and head holds no other text.
HTML_SKIPPED = {"title", "script", "style", "template", "noscript"}
HTML_BLOCKS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li",
    "main", "nav", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
}  # fmt: skip


class HtmlText(HTMLParser):
    """Text by Clarity's HTML rules: skipped elements leave nothing; each block element
    starts and ends a line; whitespace collapses to one space inside a line, which is then
    stripped, and empty lines are dropped; text inside `pre` is kept as written; entities
    are decoded."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.line: list[str] = []
        self.skipping = 0
        self.pre = 0

    def end_line(self) -> None:
        text = "".join(self.line)
        self.line = []
        if self.pre:
            if text:
                self.lines.append(text)
            return
        # Collapsed as a whole line, so whitespace on both sides of an inline tag is one.
        text = re.sub(f"[{HTML_SPACE}]+", " ", text).strip(HTML_SPACE)
        if text:
            self.lines.append(text)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in HTML_SKIPPED:
            self.skipping += 1
        elif not self.skipping and tag in HTML_BLOCKS:
            self.end_line()
            if tag == "pre":
                self.pre += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in HTML_SKIPPED:
            self.skipping = max(0, self.skipping - 1)
        elif not self.skipping and tag in HTML_BLOCKS:
            self.end_line()
            if tag == "pre":
                self.pre = max(0, self.pre - 1)

    def handle_data(self, data: str) -> None:
        if self.skipping:
            return
        self.line.append(data)


def html_text(data: bytes, name: str) -> str:
    text, _ = utf8(data, name)
    parser = HtmlText()
    parser.feed(text)
    parser.close()
    parser.end_line()
    return "".join(line + "\n" for line in parser.lines)


def pdf_pages(data: bytes, name: str) -> tuple[list[str], dict]:
    """Each page's text, pypdf's plain mode, and the extractor's facts."""
    import pypdf
    from pypdf.errors import DependencyError, PdfReadError

    protected = f"{name} is protected. Save an unprotected copy and attach that."
    try:
        reader = pypdf.PdfReader(io.BytesIO(data), root_object_recovery_limit=ROOT_RECOVERY_LIMIT)
    except (DependencyError, NotImplementedError) as e:  # an encryption pypdf can't open
        raise Refused(protected) from e
    encrypted = reader.is_encrypted
    # AES needs a crypto provider Clarity doesn't add: refused from the dictionary, so a
    # provider some other package installed (which would open it) changes nothing.
    if encrypted and protection(reader.trailer["/Encrypt"]) == "aes":
        raise Refused(protected)
    if encrypted and not reader.decrypt(""):
        raise Refused(f"{name} needs a password. Save a copy without one and attach that.")
    try:
        pages = [page.extract_text(extraction_mode="plain") for page in reader.pages]
    except PdfReadError as e:
        if encrypted:
            raise Refused(protected) from e
        raise
    extractor = {"name": "pypdf", "version": pypdf.__version__, "mode": "plain"}
    return pages, {**extractor, "encrypted": encrypted}


def protection(encrypt: Any) -> str:
    """ "aes" or "rc4", from the encryption dictionary alone: AES needs the `cryptography`
    package, which Clarity doesn't add, so it's refused whether or not another package
    happens to have installed it."""
    encrypt = encrypt.get_object()
    if encrypt.get("/V", 0) >= 5:
        return "aes"
    filters = encrypt.get("/CF", {})
    methods = {str(f.get_object().get("/CFM", "")) for f in filters.values()}
    return "aes" if methods & {"/AESV2", "/AESV3"} else "rc4"


def can_travel(text: str, name: str) -> None:
    """Text holding a NUL or a lone surrogate can't be sent or logged as it is, and replacing
    them would change it silently."""
    if "\x00" in text or re.search("[\ud800-\udfff]", text):
        raise Refused(
            f"{name} holds characters that can't be sent as text (a NUL or a lone surrogate)."
        )


def extract(kind: str, data: bytes, name: str) -> dict:
    """The document's units and facts. Text and HTML are split into lines on "\\n" only; a
    PDF into its pages."""
    if kind == "pdf":
        pages, extractor = pdf_pages(data, name)
        if not pages:
            raise Refused(f"{name} has no pages.")
        empty = sum(1 for page in pages if not page.strip())
        if empty == len(pages):
            raise Refused(
                f"{name} has no text Clarity can read. Its pages may be scanned images, which "
                "Clarity doesn't read. Save a copy with a text layer (run OCR) and attach that."
            )
        units, unit = pages, "page"
    else:
        if kind == "html":
            # html.parser is Python's own, and has changed between releases: say which ran.
            extractor = {"name": "clarity-html", "version": 1, "python": platform.python_version()}
            text = html_text(data, name)
        else:
            text, bom = utf8(data, name)
            extractor = {"name": "clarity-text", "version": 1, "bom_removed": bom}
        units, unit, empty = lines(text), "line", 0
        if not units:
            raise Refused(f"{name} is empty.")
    for unit_text in units:
        can_travel(unit_text, name)
    return {"unit": unit, "units": units, "extractor": extractor, "empty_units": empty}


def lines(text: str) -> list[str]:
    """Lines split on "\\n" only, each keeping its "\\n": they join into the text exactly."""
    pieces = text.split("\n")
    units = [piece + "\n" for piece in pieces[:-1]]
    return [*units, pieces[-1]] if pieces[-1] else units


def main() -> None:
    request = json.loads(sys.stdin.buffer.readline())
    watch(request["memory_cap"], request["seconds"])
    data = sys.stdin.buffer.read()
    try:
        answer = extract(request["kind"], data, request["name"])
    except Refused as e:
        answer = {"refused": str(e)}
    except Exception as e:  # noqa: BLE001 - a hostile file: the parent says which error
        answer = {"failed": type(e).__name__}
    sys.stdout.buffer.write(json.dumps(answer).encode("ascii"))


if __name__ == "__main__":
    main()
