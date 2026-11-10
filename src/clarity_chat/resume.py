"""Resume: rebuild a conversation from its log, to continue it.

The log says everything needed: the profile as logged (its TOML and files), the resolved
settings, every turn and call, the /set values, and where the conversation was. A
resumed conversation uses those, not today's files, and Clarity shows what differs now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clarity_chat.profiles import ConfigError


@dataclass
class Rebuilt:
    """What a log says about a conversation, before any model starts."""

    start: dict[str, Any]  # the `conversation` record
    turns: list[dict[str, Any]] = field(default_factory=list)  # `turn` records, in order
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)  # calls with a reply
    head: str | None = None
    head_guessed: bool = False  # a log from before head records (v0.1.0)
    chat_settings: dict[str, Any] = field(default_factory=dict)
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)  # server_start by ID
    last_call: dict[str, Any] | None = None  # the last call that has a reply


def rebuild(records: list[dict[str, Any]], where: str) -> Rebuilt:
    """Read a chat log's records into a Rebuilt. The head is the turn the latest `turn` or
    `head` record names; a message with no reply (Clarity stopped during its call) means
    the turn before it."""
    if not records or records[0].get("type") != "conversation":
        first = records[0].get("type") if records else "nothing"
        hint = " A batch run's calls.jsonl can't be resumed." if first == "run" else ""
        raise ConfigError(f"{where} starts with {first}, not a conversation record.{hint}")
    out = Rebuilt(records[0])
    roles: dict[str, dict[str, Any]] = {}
    for record in records:
        kind = record.get("type")
        if kind == "conversation" and record is not records[0]:
            raise ConfigError(f"{where} holds more than one conversation.")
        if kind == "turn":
            out.turns.append(record)
            roles[record["turn_id"]] = record
            out.head = record["turn_id"]
        elif kind == "head":
            out.head = record["turn_id"]
        elif kind == "call" and record.get("turn_id"):
            out.calls[record["call_id"]] = out.last_call = record
        elif kind == "settings":
            out.chat_settings = record["chat_settings"]
        elif kind == "server_start":
            out.servers.setdefault(record["server_id"], record)
    if out.head is not None and roles[out.head]["role"] == "user":
        out.head = roles[out.head]["parent_id"]
    # v0.1.0 logged neither /undo nor a failed call's move back: the head is a guess.
    out.head_guessed = "ctx_size" not in out.start
    return out


def settings_diff(logged: dict[str, Any], today: dict[str, Any]) -> list[dict[str, Any]]:
    """Settings whose value today's files would resolve differently: [{name, logged, today}]."""
    return [
        {"name": name, "logged": logged[name], "today": today.get(name)}
        for name in logged
        if today.get(name, {}).get("value") != logged[name]["value"]
    ]
