"""Plain export: a conversation as one JSON Lines file a notebook can load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clarity_chat.session import Session

RECOMPUTE_NOTE = (
    "Recomputing these calls under other conditions gives different numbers. Run "
    "`clarity_chat.recompute.check` with your loaded model first; it lists what differs."
)


def export_log(session: Session, log_path: Path, out: Path | None = None) -> dict[str, Any]:
    raise NotImplementedError
