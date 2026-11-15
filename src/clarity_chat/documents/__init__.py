"""Documents: what Clarity reads from a file you attach, and how it goes into your message.

The rest of Clarity imports from this package only, never a submodule (v0.3.1 decision 21).
"""

from clarity_chat.documents.extract import (
    Document,
    DocumentRefused,
    checked_name,
    extract,
    read_path,
)
from clarity_chat.documents.insert import (
    FRAMING,
    SENTENCES,
    Attached,
    Fit,
    Room,
    content,
    fit,
    framing_of,
    free,
    piece,
    scan,
    start_that_fits,
)

__all__ = [
    "FRAMING",
    "SENTENCES",
    "Attached",
    "Document",
    "DocumentRefused",
    "Fit",
    "Room",
    "checked_name",
    "content",
    "extract",
    "fit",
    "framing_of",
    "free",
    "piece",
    "read_path",
    "scan",
    "start_that_fits",
]
