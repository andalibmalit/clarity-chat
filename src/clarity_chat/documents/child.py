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
import re
import resource
import sys
import threading
import time
from html.parser import HTMLParser
from typing import Any

MEMORY_EXIT = 70  # the watchdog's exit codes, which the parent turns into sentences
DEADLINE_EXIT = 71
MAX_STREAM = 64 * 1024 * 1024  # no stream in a file Clarity reads can be longer
ROOT_RECOVERY_LIMIT = 10_000  # pypdf's default, set here so a moved default can't widen it


def peak_memory() -> int:
    """This process's peak resident memory, in bytes (getrusage gives kilobytes on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def watch(memory_cap: int, seconds: float, interval: float = 0.02) -> None:
    """End this process past `memory_cap` bytes or `seconds`. Polling lets memory overshoot by
    what's allocated between polls; on Linux, RLIMIT_AS stops the process at twice the cap
    as well."""
    deadline = time.monotonic() + seconds
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (memory_cap * 2, memory_cap * 2))

    def loop() -> None:
        while True:
            if peak_memory() > memory_cap:
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


# HTML, version 1 of Clarity's rules (listed in reference.md).
HTML_SPACE = " \t\n\r\f"  # HTML's whitespace: a no-break space is kept, as text
HTML_SKIPPED = {"head", "script", "style", "template", "noscript"}
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
        text = text.strip(HTML_SPACE)
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
        self.line.append(data if self.pre else re.sub(f"[{HTML_SPACE}]+", " ", data))


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

    class Reader(pypdf.PdfReader):
        def _handle_encryption(self, password: str | bytes | None) -> None:
            # The reader tries "" here, before any page: AES is refused first.
            if protection(self.trailer["/Encrypt"]) == "aes":
                raise Refused(protected)
            super()._handle_encryption(password)

    with pypdf.apply_configuration(maximum_declared_stream_length=MAX_STREAM):
        try:
            reader = Reader(io.BytesIO(data), root_object_recovery_limit=ROOT_RECOVERY_LIMIT)
        except (DependencyError, NotImplementedError) as e:  # an encryption pypdf can't open
            raise Refused(protected) from e
        encrypted = reader.is_encrypted
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
        empty = sum(1 for page in pages if not page.strip())
        if empty == len(pages):
            raise Refused(f"{name} has no text to read: its pages are images.")
        units, unit = pages, "page"
    else:
        if kind == "html":
            text, extractor = html_text(data, name), {"name": "clarity-html", "version": 1}
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
