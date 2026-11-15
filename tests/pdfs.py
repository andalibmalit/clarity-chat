"""The tests' own small PDF writer: made-up text in Helvetica, one content stream per page.

An empty page has no text (as a scanned page has none). `encrypt` adds an encryption
dictionary of the kind named, written by pypdf's writer where it can (RC4) and by hand where
it can't without the `cryptography` package (AES, which Clarity refuses before decrypting).
"""

from __future__ import annotations

import io


def escaped(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """A PDF whose pages hold `pages`' text, a line of the PDF per line of text."""
    objects: list[bytes] = []
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        lines = text.split("\n") if text else []
        shown = " T* ".join(f"({escaped(line)}) Tj" for line in lines)
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td {shown} ET".encode("latin-1") if lines else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R"
            " /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % n + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1))
    out.write(b"startxref\n%d\n%%%%EOF\n" % xref)
    return out.getvalue()


def encrypted_pdf(pages: list[str], user_password: str = "") -> bytes:
    """`pages` encrypted with RC4-128 by pypdf's own writer; an empty user password opens it."""
    import pypdf

    writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(make_pdf(pages))))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


AES_DICTIONARIES = {
    5: b"<< /Filter /Standard /V 5 /R 6 /Length 256 /P -4"
    b" /O <00> /U <00> /OE <00> /UE <00> /Perms <00> >>",
    4: b"<< /Filter /Standard /V 4 /R 4 /Length 128 /P -4"
    b" /CF << /StdCF << /CFM /AESV2 /AuthEvent /DocOpen /Length 16 >> >>"
    b" /StmF /StdCF /StrF /StdCF /O <%s> /U <%s> >>" % (b"00" * 32, b"00" * 32),
}


def aes_pdf(pages: list[str], version: int = 5) -> bytes:
    """A PDF whose encryption dictionary asks for AES: AES-256 (/V 5), or AES-128 through a
    /V 4 crypt filter (/AESV2). Its content isn't really encrypted: Clarity reads the
    dictionary and refuses before decrypting anything."""
    plain = make_pdf(pages)
    head, _, _ = plain.rpartition(b"xref\n")
    n = plain.count(b" 0 obj\n") + 1
    extra = b"%d 0 obj\n" % n + AES_DICTIONARIES[version] + b"\nendobj\n"
    offsets = []
    position = 0
    body = head + extra
    for line_start in range(n):
        marker = b"%d 0 obj\n" % (line_start + 1)
        position = body.index(marker, position)
        offsets.append(position)
    xref = len(body)
    table = b"xref\n0 %d\n0000000000 65535 f \n" % (n + 1)
    table += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    trailer = b"trailer\n<< /Size %d /Root 1 0 R /Encrypt %d 0 R /ID [<00> <00>] >>\n" % (n + 1, n)
    return body + table + trailer + b"startxref\n%d\n%%%%EOF\n" % xref
