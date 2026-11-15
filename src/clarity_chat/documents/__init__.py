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

__all__ = ["Document", "DocumentRefused", "checked_name", "extract", "read_path"]
