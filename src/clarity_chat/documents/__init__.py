"""Documents: what Clarity reads from a file you attach, and how it goes into your message.

The rest of Clarity imports from this package only, never a submodule (v0.3.1 decision 21).
"""

from clarity_chat.documents.insert import (
    FRAMING,
    SENTENCES,
    TURN_FIELDS,
    Attached,
    Fit,
    Room,
    added_ranges,
    attached_in,
    checked_attachments,
    content,
    counted,
    covers_whole,
    fill,
    fit,
    framing_of,
    free,
    how_words,
    message_content,
    more_than,
    piece,
    scan,
    start_that_fits,
    summaries,
    with_text,
)
from clarity_chat.documents.library import (
    LIBRARY,
    OriginalChanged,
    Stale,
    checked_library_name,
    control_token_parts,
    default_name,
)
from clarity_chat.documents.library import add as library_add
from clarity_chat.documents.library import check as library_check
from clarity_chat.documents.library import documents as library_documents
from clarity_chat.documents.library import parts_of as library_parts
from clarity_chat.documents.library import read_manifest as library_manifest
from clarity_chat.documents.library import rebuild as library_rebuild
from clarity_chat.documents.library import state as library_state
from clarity_chat.documents.parts import (
    CHUNKER,
    build_id,
    closing_tag_parts,
    cut_parts,
    parts_jsonl,
)
from clarity_chat.documents.reading import (
    MAX_BYTES,
    SIZE_LIMIT,
    Document,
    DocumentRefused,
    checked_name,
    extract,
    read_path,
    slow_read,
    too_big,
    visible_bidi,
)
from clarity_chat.documents.search import COMMON_WORDS, CONSIDERED, Loaded, words_of
from clarity_chat.documents.search import load as load_document
from clarity_chat.documents.search import search as search_documents

__all__ = [
    "COMMON_WORDS",
    "CONSIDERED",
    "Loaded",
    "load_document",
    "search_documents",
    "words_of",
    "LIBRARY",
    "OriginalChanged",
    "Stale",
    "checked_library_name",
    "control_token_parts",
    "default_name",
    "library_add",
    "library_check",
    "library_documents",
    "library_manifest",
    "library_parts",
    "library_rebuild",
    "library_state",
    "CHUNKER",
    "build_id",
    "closing_tag_parts",
    "cut_parts",
    "parts_jsonl",
    "FRAMING",
    "MAX_BYTES",
    "SIZE_LIMIT",
    "SENTENCES",
    "TURN_FIELDS",
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
    "counted",
    "covers_whole",
    "extract",
    "fill",
    "fit",
    "framing_of",
    "slow_read",
    "free",
    "how_words",
    "message_content",
    "more_than",
    "piece",
    "read_path",
    "scan",
    "start_that_fits",
    "too_big",
    "visible_bidi",
    "with_text",
    "summaries",
]
