"""Documents: what Clarity reads from a file you attach, and how it goes into your message.

The rest of Clarity imports from this package only, never a submodule (v0.3.1 decision 21).
"""

from clarity_chat.documents.insert import (
    FRAMING,
    SENTENCES,
    Attached,
    Fit,
    Room,
    added_ranges,
    attached_in,
    checked_attachments,
    content,
    covers_whole,
    fit,
    framing_of,
    free,
    message_content,
    piece,
    scan,
    start_that_fits,
    summaries,
)
from clarity_chat.documents.reading import (
    MAX_BYTES,
    Document,
    DocumentRefused,
    checked_name,
    extract,
    read_path,
)

__all__ = [
    "FRAMING",
    "MAX_BYTES",
    "SENTENCES",
    "Attached",
    "Document",
    "DocumentRefused",
    "Fit",
    "Room",
    "added_ranges",
    "attached_in",
    "checked_attachments",
    "checked_name",
    "content",
    "covers_whole",
    "extract",
    "fit",
    "framing_of",
    "free",
    "message_content",
    "piece",
    "read_path",
    "scan",
    "start_that_fits",
    "summaries",
]
