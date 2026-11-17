"""A search's refusal, carried as data and worded by each front end (v0.3.2 decision 6, M4;
part 1's whole-branch PM1): "Each refusal names a way out that exists on that front end."
The page says the board's sentences (N29-N33); the terminal and --json name their
commands. The log keeps the page's words, the record every front end reads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clarity_chat.documents.library import way_out
from clarity_chat.documents.search import NO_FTS5
from clarity_chat.profiles import ConfigError, shown

NOTHING_SENT = "Nothing was sent."  # N29: a refusal at Send's first line
WITHOUT_SEARCH = "To send without searching, stop searching every document in this chat."


STOPPING = {  # how the terminal stops searching a document, by where the search ran
    "chat": "stop searching {it} with /search",
    "run": "take {it} out of the profile's search list",
    "try": "leave {it} out of the documents you try",  # typed, not a profile's (fix 2's S1)
}


def problem_words(problem: dict[str, Any], terminal: bool, where: str, at_send: bool) -> str:
    """One problem's sentence. `terminal`: the terminal's and --json's words, else the
    page's; `where` the search ran: a chat, a run or `document try` (STOPPING)."""
    name = problem.get("document", "")
    state = problem["state"]
    # A chat's way out is the chat's own command; `clarity document …` is the command
    # line's, for a run and `document try` (the main session's PM1 ruling).
    chat = where == "chat"
    add = f"`/documents add PATH {name}`" if chat else f"`clarity document add FILE --name {name}`"
    if state == "no_fts5":
        if not at_send:
            return NO_FTS5
        if terminal:  # the command that turns it off (fix 2's C2)
            return f"{NO_FTS5} `/search none` turns search off for this chat."
        return f"{NO_FTS5} {WITHOUT_SEARCH}"
    if state == "unreadable":  # N32: the reason, then this front end's way out (surface S2)
        front = "page" if not terminal else "chat" if where == "chat" else "cli"
        why = problem["why"].removeprefix(name + "'s ")
        way = f" {way_out(name, front)}" if problem.get("add_again") else ""
        return f"Clarity can't search {name}: its {why}{way}"
    if state == "profile_missing":  # N37
        way = f"Add it with {add}" if terminal else "Add it in Documents…"
        return (
            f"Profile {problem['profile']} searches {name}, which isn't in your documents. "
            f"{way}, or take it out of search under [documents] in the profile's file."
        )
    stop = STOPPING[where] if terminal else "stop searching {it}"
    if state == "missing":  # N31
        way = f"add the file again: {add}" if terminal else "add the file again in Documents…"
        return (
            f"Clarity can't search {name}: its folder in local/documents/ is gone. "
            f"{stop.format(it='it').capitalize()}, or {way}."
        )
    assert state == "control_tokens"  # N33
    tokens = problem["tokens"]
    held = " and ".join([", ".join(tokens[:-1]), tokens[-1]] if tokens[1:] else tokens)
    one, it = ("a control token", "it") if len(tokens) == 1 else ("control tokens", "them")
    where = (
        f' ("{shown(problem["title"])}")'  # a heading can hold ESC (the fix review's S2)
        if problem["title"]
        else f" (page {problem['page']})"
        if problem["page"]
        else ""
    )
    # Out of your own file, then added again (cut 1: the library's copy isn't edited).
    again = f": {add}" if terminal else " in Documents…, with the same name"
    return (
        f"Part {problem['ordinal']} of {name}{where} holds {held}, which this model reads as "
        f"{one}, not text. Take {it} out of your file, then add it again{again}. Or "
        f"{stop.format(it=name)}."
    )


class SearchRefused(ConfigError):
    """A searched message's call (or /try, or a run) refused before anything was sent:
    `problems` says why, as data (each document and its state, the control-token part, or
    no FTS5), and each front end words its way out. str() is the page's words, which the
    log keeps; `terminal` the terminal's and --json's. `at_send` heads both with "Nothing
    was sent."; `where` the search ran: `chat`, `run` or `try`, for the way to stop it."""

    def __init__(
        self, problems: list[dict[str, Any]], at_send: bool = False, where: str = "chat"
    ) -> None:
        self.problems, self.at_send, self.where = problems, at_send, where
        super().__init__(self.words(terminal=False))

    @property
    def terminal(self) -> str:
        return self.words(terminal=True)

    def words(self, terminal: bool) -> str:
        lines = [problem_words(p, terminal, self.where, self.at_send) for p in self.problems]
        return "\n".join([NOTHING_SENT, *lines] if self.at_send else lines)


def control_token_problem(part: Mapping[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    """A found part holding this model's control tokens, as a problem, or None (decision
    12): Send refuses it, and /try says it would."""
    tokens = list(dict.fromkeys(tokenizer.special_tokens_in(part["text"])))
    if not tokens:
        return None
    keys = ("ordinal", "title", "page")
    return {"document": part["document"], "state": "control_tokens",
            **{k: part[k] for k in keys}, "tokens": tokens}  # fmt: skip
