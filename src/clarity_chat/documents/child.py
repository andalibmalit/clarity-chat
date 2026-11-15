"""Extraction's child process (A1's stub)."""

from __future__ import annotations

MEMORY_EXIT = 70
DEADLINE_EXIT = 71


class Refused(Exception):
    pass


def watch(memory_cap: int, seconds: float, interval: float = 0.02) -> None:
    pass


def pdf_pages(data: bytes, name: str) -> tuple[list[str], dict]:
    return [], {}


def extract(kind: str, data: bytes, name: str) -> dict:
    return {}
