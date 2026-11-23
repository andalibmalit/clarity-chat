"""The class tests (sprint rule 13; v0.3.2's plan, "Class tests"): each guards a class of
defect rather than one case, and each later item adds its rows.

- T1, missing fields: a machine-readable copy of the plan's Shapes table. A log missing any
  field in it is refused by resume and export with `logs.unreadable`'s sentence, and every
  read and view of a chat holding such a turn answers rather than stopping the session.
- T2, no terminal command in a page sentence: every sentence the page can show is scanned
  for a slash command, `clarity …` or a `--flag`.
- T3, accepted and logged means resumable, exportable, replayable: for each command that
  writes a record, a scripted chat resumes with an identical render, exports, and replays
  each completed call Identical.

Made-up text only.
"""

from __future__ import annotations

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.documents import SENTENCES, library_add, problem_words, reading
from clarity_chat.export import export_log
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.fakes import FakeBackend, FakeEmbedder
from tests.test_embedder import Untouchable
from tests.test_session_attach import doc
from tests.test_web import GET, ORIGIN, PORT, SECRET, command

SRC = Path(__file__).parent.parent / "src" / "clarity_chat"


def quiet(event: dict) -> None:
    pass


# --- T1: missing fields, from the Shapes table

# The plan's Shapes table, as data: for each record kind, where each field sits. A path is
# the keys and list indexes from the record to the field. v0.3.1's attachment fields are
# the first rows (its 21-case sweep); v0.3.2's `found` and `search` record join them. A
# call's `message_spans` `added` part entries have no row: resume doesn't read spans, and
# export refuses a log missing one (part 1's whole-branch C11).
SHAPES: dict[str, list[tuple[Any, ...]]] = {
    "turn": [  # `attachments` itself is optional: a turn without it had none (old logs)
        ("framing",),
        ("framing_origin",),
        ("placement",),
        *(
            ("attachments", 0, field)
            for field in (
                "name",
                "kind",
                "bytes",
                "sha256",
                "extractor",
                "unit",
                "total",
                "how",
                "range",
                "marker",
                "text",
                "extracted_sha256",
                "empty_units",
                "control_tokens",
                "has_closing_tag",
                "tokens",
            )  # fmt: skip
        ),
    ],
    "search": [("documents",)],  # v0.3.2 B1
}
FOUND_KEYS: dict[str, set[str]] = {  # the Shapes table's `found` (Checkpoint 1's S9)
    "found": {
        "method", "searched_for", "words", "ignored", "documents", "engine",
        "considered", "budget", "budget_origin", "parts", "left_out", "matched", "tokens",
    },
    "documents": {"name", "build_id", "original_sha256"},
    "engine": {
        "name", "sqlite", "tokenizer", "weights", "word_rule", "common_words", "query",
    },
    "parts": {
        "id", "document", "ordinal", "title", "page", "text", "sha256", "rank", "score",
        "title_words", "words", "tokens",
    },
    "left_out": {"id", "rank", "score", "why"},
}  # fmt: skip
FOUND_ROWS = [  # v0.3.2 B2: the turn's `found`, from FOUND_SHAPE below
    ("turn", path)
    for path in [
        *(("found", key) for key in sorted(FOUND_KEYS["found"])),
        *(("found", "parts", 0, key) for key in sorted(FOUND_KEYS["parts"])),
        *(("found", "left_out", 0, key) for key in sorted(FOUND_KEYS["left_out"])),
        *(("found", "documents", 0, key) for key in sorted(FOUND_KEYS["documents"])),
        *(("found", "engine", key) for key in sorted(FOUND_KEYS["engine"])),
    ]
]
ROWS = [(kind, path) for kind, paths in SHAPES.items() for path in paths] + FOUND_ROWS
TURN_ROWS = [("turn", path) for path in SHAPES["turn"]]


def holds(record: dict, path: tuple[Any, ...]) -> bool:
    """Whether the record has the field's parent: a turn without attachments has none."""
    node: Any = record
    for key in path[:-1]:
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return False
    return isinstance(node, dict) and path[-1] in node


def drop(record: dict, path: tuple[Any, ...]) -> None:
    node: Any = record
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]


def scripted_chat(session: Session, tmp_path: Path) -> None:
    """One chat that writes every record kind the Shapes table names, on two branches."""
    reading._last = None
    library_add(session.root, b"## Wren\nA made-up bird.\n", "birds.md")
    session.new_conversation("chat")
    session.set_search(["birds"])
    session.say("Tell me about the wren")  # found: a part added
    session.say("The wren again")  # found: that part left out, already in the chat
    session.attach(doc())
    session.say("What's in it?")
    session.regen()
    session.undo()
    session.attach(doc("Made-up other.\n", "other.md"), "lines", 1, 1)
    session.say("And this one?")


@pytest.fixture
def chat_mode_session(fetched_root, chat_tokenizer):
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    return Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)


@pytest.mark.parametrize(("kind", "path"), ROWS, ids=lambda v: str(v))
def test_t1_a_log_missing_a_field_is_refused_by_resume_and_export(
    chat_mode_session, tmp_path, kind, path
):
    session = chat_mode_session
    scripted_chat(session, tmp_path)
    assert session.conversation is not None
    log = session.conversation.log.path
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    dropped = 0
    for row in rows:
        if row.get("type") == kind and holds(row, path):
            drop(row, path)
            dropped += 1
    assert dropped, f"the scripted chat writes no {kind} with {path}"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    session.new_conversation("chat")
    for read in (lambda: session.resume(log), lambda: export_log(session, log)):
        with pytest.raises(ConfigError) as refused:
            read()
        assert str(refused.value).startswith("Can't read the records of"), refused.value


VIEWS = ["thread", "branches", "settings", "pending", "chats"]
READS = ["history", "branches", "settings", "state"]
READS_OF_A_CALL = ["compare", "prompt", "probs"]


@pytest.fixture
def page_chat(fetched_root, chat_tokenizer):
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    # A view that raises answers 500, as it would in a browser, rather than raising here.
    with TestClient(app, base_url=ORIGIN, raise_server_exceptions=False) as client:
        command(client, "profile", "chat")
        yield client, session


@pytest.mark.parametrize(("kind", "path"), TURN_ROWS, ids=lambda v: str(v))
def test_t1_every_read_and_view_answers_when_a_field_is_missing(page_chat, kind, path):
    """Taken out in memory, as a log resume would refuse: each read answers with a result or
    a refusal, and each view with a page or a refusal; none stops the session."""
    client, session = page_chat
    session.attach(doc())
    call = command(client, "say", "What's in it?").json()["result"]["call"]
    command(client, "regen")
    assert session.conversation is not None and kind == "turn"
    turns = [t for t in session.conversation.turns.values() if holds(t.attached_fields(), path)]
    assert turns
    for turn in turns:  # a Turn's field can't be missing: a top-level one is set to None
        if len(path) > 1:
            drop({"attachments": turn.attachments}, path)
        else:
            setattr(turn, path[0], None)
    stopped = []
    asked = [(name, "") for name in READS] + [(n, call["call_id"]) for n in READS_OF_A_CALL]
    for name, args in asked:
        answer = command(client, name, args).json()
        if answer.get("stopped") or not ("result" in answer or answer.get("type")):
            stopped.append((name, answer))
    views = [f"/view/{name}" for name in VIEWS]
    views += [f"/view/{name}?call={call['call_id']}" for name in ("details", "compare")]
    for view in views:
        response = client.get(view, headers=GET)
        if response.status_code not in (200, 503):
            stopped.append((view, response.status_code, response.text[:200]))
    assert command(client, "state").json().get("result"), "the session stopped"
    assert stopped == []


@pytest.mark.parametrize(("kind", "path"), FOUND_ROWS, ids=lambda v: str(v))
def test_t1_every_read_and_view_answers_when_a_found_field_is_missing(page_chat, kind, path):
    """The same sweep over a searched turn's `found` (Checkpoint 2's S9), before C3's views
    read it: taken out in memory, each read and view answers or refuses."""
    client, session = page_chat
    reading._last = None
    library_add(session.root, b"## Wren\nA made-up bird.\n", "birds.md")
    session.set_search(["birds"])
    command(client, "say", "Tell me about the wren")
    call = command(client, "say", "The wren again").json()["result"]["call"]  # left out
    assert session.conversation is not None and kind == "turn"
    turns = [t for t in session.conversation.turns.values() if holds({"found": t.found}, path)]
    assert turns
    for turn in turns:
        drop({"found": turn.found}, path)
    stopped = []
    asked = [(name, "") for name in READS] + [(n, call["call_id"]) for n in READS_OF_A_CALL]
    for name, args in asked:
        answer = command(client, name, args).json()
        if answer.get("stopped") or not ("result" in answer or answer.get("type")):
            stopped.append((name, answer))
    views = [f"/view/{name}" for name in VIEWS]
    views += [f"/view/{name}?call={call['call_id']}" for name in ("details", "compare")]
    for view in views:
        response = client.get(view, headers=GET)
        if response.status_code not in (200, 503):
            stopped.append((view, response.status_code, response.text[:200]))
    assert stopped == []
    # T1's rule: /state answers with a result, or a refusal with a sentence; never an
    # empty answer, and never the session stopping (the runner's report on e8f47b3).
    answer = command(client, "state").json()
    assert not answer.get("stopped"), answer
    assert answer.get("result") or str(answer.get("error") or "").strip(), answer


# T1 for a meaning found (v0.4.1 A3, the Shapes table): each top-level field, each part's,
# each document's and the query's, taken out; `engine` checked as one value (a leaf changed
# breaks its setup hash); and a field of the other method's found, put in.
MEANING_FOUND_KEYS: dict[str, set[str]] = {
    "found": {
        "method", "searched_for", "documents", "engine", "query", "min_similarity",
        "min_similarity_origin", "considered", "budget", "budget_origin", "parts",
        "left_out", "matched", "tokens",
    },
    "documents": {"name", "build_id", "original_sha256", "vectors_sha256"},
    "query": {"tokens", "sha256"},
    "parts": {
        "id", "document", "ordinal", "title", "page", "text", "sha256", "rank", "score",
        "tokens",
    },
}  # fmt: skip
MEANING_ROWS: list[tuple[Any, ...]] = [
    *(("found", key) for key in sorted(MEANING_FOUND_KEYS["found"])),
    *(("found", "parts", 0, key) for key in sorted(MEANING_FOUND_KEYS["parts"])),
    *(("found", "documents", 0, key) for key in sorted(MEANING_FOUND_KEYS["documents"])),
    *(("found", "query", key) for key in sorted(MEANING_FOUND_KEYS["query"])),
]


def meaning_log(session: Session) -> Path:
    from tests.test_meaning import GLOSSARY, MEANING, profile

    reading._last = None
    profile(session.root, "meaning", MEANING)
    library_add(session.root, GLOSSARY.encode(), "notes.md")
    session.new_conversation("meaning")
    session.say("How do leaves turn light into sugar?")
    session.say("Why does rock wear away?")
    assert session.conversation is not None
    return session.conversation.log.path


def refused_by_resume_and_export(session: Session, log: Path, change) -> None:
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    changed = sum(
        bool(row.get("type") == "turn" and "found" in row and change(row)) for row in rows
    )
    assert changed, "the chat wrote no searched turn this change applies to"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    session.new_conversation("chat")
    for read in (lambda: session.resume(log), lambda: export_log(session, log)):
        with pytest.raises(ConfigError) as refused:
            read()
        assert str(refused.value).startswith("Can't read the records of"), refused.value


@pytest.fixture
def meaning_session(fetched_root, chat_tokenizer):
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    return Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=FakeEmbedder(),  # ty: ignore[invalid-argument-type]
    )


@pytest.mark.parametrize("path", MEANING_ROWS, ids=lambda v: str(v))
def test_t1_a_meaning_log_missing_a_field_is_refused_by_resume_and_export(meaning_session, path):
    log = meaning_log(meaning_session)
    refused_by_resume_and_export(
        meaning_session, log, lambda row: holds(row, path) and drop(row, path) is None
    )


def changed_leaf(row: dict) -> bool:
    if not holds(row, ("found", "engine", "device")):
        return False
    row["found"]["engine"]["device"] = "CPU"  # setup no longer the hash of the rest
    return True


def missing_leaf(row: dict) -> bool:
    if not holds(row, ("found", "engine", "flags", "pooling")):
        return False
    del row["found"]["engine"]["flags"]["pooling"]
    return True


def holding_words(row: dict) -> bool:
    if row["found"].get("method") != "meaning":
        return False
    row["found"]["words"] = ["leaves"]
    return True


@pytest.mark.parametrize("change", [changed_leaf, missing_leaf, holding_words])
def test_t1_a_meaning_found_s_engine_is_one_value_and_holds_no_words(meaning_session, change):
    log = meaning_log(meaning_session)
    refused_by_resume_and_export(meaning_session, log, change)


def test_t1_a_words_found_holding_a_query_is_refused(chat_mode_session, tmp_path):
    session = chat_mode_session
    scripted_chat(session, tmp_path)
    assert session.conversation is not None
    log = session.conversation.log.path

    def holding_a_query(row: dict) -> bool:
        row["found"]["query"] = {"tokens": 3, "sha256": "0" * 64}
        return True

    refused_by_resume_and_export(session, log, holding_a_query)


# --- T2: no terminal command in a page sentence

COMMAND = re.compile(r"(?:^|[\s(\"'“‘`])(/[a-z][a-z_]*\b|clarity [a-z]+\b|--[a-z][a-z-]*)")


class Text(HTMLParser):
    """A template's text nodes and the attributes a person reads or hears."""

    SPOKEN = ("aria-label", "title", "placeholder", "alt")

    def __init__(self) -> None:
        super().__init__()
        self.found: list[str] = []

    def handle_data(self, data: str) -> None:
        self.found.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and (name in self.SPOKEN or (name.startswith("data-") and " " in value)):
                self.found.append(value)


def template_text(source: str) -> list[str]:
    source = re.sub(r"\{#.*?#\}", "", source, flags=re.S)
    source = re.sub(r"\{%.*?%\}|\{\{.*?\}\}", " ", source, flags=re.S)
    parser = Text()
    parser.feed(source)
    return parser.found


def js_strings(source: str) -> list[str]:
    """app.js's string literals, comments and regular expressions skipped; a template
    literal's `${…}` dropped."""
    found, i, n = [], 0, len(source)
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            i = source.find("\n", i)
            i = n if i < 0 else i
        elif source.startswith("/*", i):
            i = source.index("*/", i) + 2
        elif c == "/" and source[:i].rstrip()[-1:] in tuple("(,=:[!&|?{};"):
            i += 1  # a regular expression: skip to its closing slash, past any [class]
            in_class = False
            while i < n and (source[i] != "/" or in_class):
                if source[i] == "\\":
                    i += 1
                elif source[i] in "[]":
                    in_class = source[i] == "["
                i += 1
            i += 1
        elif c in "\"'`":
            j, text, depth = i + 1, [], 0
            while j < n:
                if source[j] == "\\":
                    text.append(source[j : j + 2])
                    j += 2
                elif c == "`" and source.startswith("${", j):
                    depth, j = 1, j + 2
                    while j < n and depth:
                        depth += {"{": 1, "}": -1}.get(source[j], 0)
                        j += 1
                    text.append("\0")  # not text: a string that's all paths stays one
                elif source[j] == c:
                    break
                else:
                    text.append(source[j])
                    j += 1
            found.append("".join(text))
            i = j + 1
        else:
            i += 1
    return found


ERRORS = {"ConfigError", "ContextError", "DocumentRefused", "DoesntFit", "Refused", "BackendError"}

# Refusals that name a terminal command but can't reach the page: the page never sends the
# command, or never sends it in that form. Each is said here, with why.
TERMINAL_ONLY = {
    "clarity document try searches by words only; /try in a chat searches by meaning.": (
        "only `clarity document try` raises it (decision 17); the page has no such command"
    ),
    "llama-server's /props failed after startup: ": "a llama-server endpoint, not a command",
    "Usage: /regen [N], where N is a whole number of replies, 1 or more; or /regen same.": (
        "the page sends a number it checked, or none"
    ),
    "Usage: /branches [all]": "the page sends none, or all",
    "Usage: /edit [--drop N]... TEXT": "the page's Edit sends no empty text",
    "--drop : this message has no attachments.": "the page's × is drawn only on attachments",
    "--drop : this message has . Use a number from 1 to .attachment": (
        "the page's × sends the place of an attachment it drew"
    ),
    "Usage: /try TEXT": "the try box runs no empty message",
    "Usage: /set NAME [VALUE]": "the page's rows always name their setting",
    "Usage: /title TEXT": "the page's Rename sends no empty title",
    "Usage: /switch TURN_ID": "the page sends a branch's turn ID",
    "Usage: /resume ID": "the page sends a log ID",
    "Usage: /profile NAME": "the picker sends a profile's name",
    "Usage: /note [CALL_ID] TEXT": "the page's note box sends no empty note",
    ": /documents add PATH NAME.": "the page sends an upload, named in its box",
    "Usage: /compare [CALL_ID]": "the page sends one call ID",
    "/resume takes a conversation ID from /chats, not a path.": "the page sends a log ID",
    "Usage: /prompt [CALL_ID]": "the page sends one call ID",
    "Usage: /probs [CALL_ID]": "the page sends one call ID",
    "/export takes no path: it writes to logs/exports/ID.jsonl.": "the page sends none",
    "Usage: /replay [CALL_ID]": "the page sends one call ID",
    "/profile takes a profile name from profiles/ or local/profiles/, not a path.": (
        "the page sends a name from /profiles"
    ),
    "Unknown command /. /help lists them.": "the page sends only commands in the table",
    "--attach names  twice. Name each file once.": "clarity chat --attach only",
    "Usage: /cancel": "the page sends none, or an upload ID",
    " isn't a log name Clarity writes, so it can't name the export. Pass --out PATH.": (
        "the page exports the open chat, a log Clarity wrote"
    ),
    "/set: unknown setting . Known: .": "the page sends names from /settings",
    "No conversation or log file . `clarity chats` or /chats lists them.": (
        "the page exports the open chat, and opens chats it listed"
    ),
    "This conversation has no turn . /branches lists them.": (
        "the page switches to turns it drew from /branches; a stale one needs another tab, "
        "which the page refuses"
    ),
    "Usage: /documents [add PATH [NAME]]": ("the page sends /documents, or add with an upload ID"),
    "The Transformers backend needs torch: run `uv sync --extra hf`.": (
        "a setup step outside Clarity, not a terminal equivalent of a page control"
    ),
    " isn't set in this chat. Usage: /set NAME VALUE": "the page resets only a changed row",
    "Setup needs the cmake in its own environment. Run it as: uv run --group setup clarity "
    "setup": "only `clarity setup` raises it, and setup has no page (v0.4.2)",
}


def refusals() -> list[tuple[str, int, str]]:
    """Each refusal's constant text, by file and line: the strings in `raise ERROR(…)`."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in ("cli.py", "batch.py") and path.parent == SRC:
            continue  # the terminal's own output
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            # A command's usage line, raised by needs() (the C1 tripwire review's must 1):
            # a view that sends an empty argument would show it.
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "needs"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
            ):
                found.append((path.name, node.lineno, f"Usage: {node.args[1].value}"))
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            called = node.exc.func
            name = called.id if isinstance(called, ast.Name) else getattr(called, "attr", "")
            if name in ERRORS:
                parts = [
                    n.value
                    for n in ast.walk(node.exc)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
                found.append((path.name, node.lineno, "".join(parts)))
    return found


def page_sentences() -> list[tuple[str, str]]:
    sentences = []
    for path in [
        *sorted((SRC / "web_templates").glob("*.html")),
        SRC / "web_static" / "index.html",
    ]:
        sentences += [(path.name, t) for t in template_text(path.read_text(encoding="utf-8"))]
    js = (SRC / "web_static" / "app.js").read_text(encoding="utf-8")
    sentences += [("app.js", s) for s in js_strings(js) if " " in s]
    sentences += [(f"SENTENCES[{k!r}]", v) for k, v in SENTENCES.items()]
    sentences += [
        (f"{name}:{line}", text) for name, line, text in refusals() if text not in TERMINAL_ONLY
    ]
    # Meaning search's refusals in the page's words (v0.4.1 C1): problem_words returns them,
    # so no raise holds them. The terminal's fetch sentence is the terminal's only.
    sentences += [
        (f"problem_words[{p['state']}]", problem_words(p, terminal=False, where="chat", at_send=a))
        for p in MEANING_PROBLEMS
        for a in (False, True)
    ]
    return sentences


MEANING_PROBLEMS = [
    {"state": "no_embedder"},
    {"state": "embedder", "why": "The embedder didn't start: made up."},
    {"state": "too_long", "tokens": 9104},
    {"state": "too_long", "tokens": 9104, "document": "notes", "ordinal": 2, "title": "Made up",
     "page": None},
]  # fmt: skip


def test_t2_the_scan_sees_meaning_search_s_page_words():
    seen = [text for where, text in page_sentences() if where.startswith("problem_words[")]
    assert any(text.startswith("Meaning search needs bge-m3") for text in seen)
    assert any(text.startswith("Your message is too long for meaning search") for text in seen)


def test_t2_the_scan_sees_the_page_s_sentences():
    """The scanner reads each source: an empty one would pass every check below."""
    seen = {where.split(":")[0].split("[")[0] for where, _ in page_sentences()}
    assert {"index.html", "thread.html", "attach.html", "app.js", "SENTENCES"} <= seen
    assert any(text == "Choose another file" for _, text in page_sentences())
    assert any("No attachment" in text for _, text in page_sentences())


def test_t2_no_page_sentence_names_a_terminal_command():
    named = [
        (where, COMMAND.findall(text), text.strip()[:100])
        for where, text in page_sentences()
        if COMMAND.search(text)
    ]
    assert named == []


def test_t2_each_terminal_only_refusal_is_still_raised():
    """An entry whose sentence has changed would hide nothing and should go."""
    raised = {text for _, _, text in refusals()}
    assert sorted(set(TERMINAL_ONLY) - raised) == []


# --- T3: accepted and logged means resumable, exportable, replayable


def write_chat(name: str, c: Commands, backend: FakeBackend, tmp_path: Path) -> None:
    """A chat that uses `name`, the command that writes the record."""
    path = tmp_path / "notes.md"
    path.write_text("Made-up line one.\nMade-up line two.\n", encoding="utf-8")
    run = lambda line: c.run(*line.partition(" ")[::2], quiet)  # noqa: E731
    run("profile chat")
    first = run("say Hello there")["call"]["call_id"]
    if name == "regen":
        run("regen")
    elif name == "edit":
        run("edit Hello again")
    elif name == "set":
        run("set temperature 0.3")
        run("say And now?")
    elif name == "title":
        run("title A made-up title")
    elif name == "mark":
        run(f"mark {first} landed")
    elif name == "note":
        run(f"note {first} A made-up note.")
    elif name == "attach":
        run(f"attach {path}")
        run("say What's in it?")
        run(f"attach {path} lines 2-2")
        run("say And the second line?")
    elif name == "search":  # v0.3.2 B1
        run(f"documents add {path} notes")
        run("search notes")
        sent = run("say Tell me about line one")["call"]  # a part goes in (whole-branch S3)
        assert '<part document="notes">' in sent["prompt_text"]
        run("search none")
        run("say Without it?")
    elif name == "meaning":  # v0.4.1 A3: a chat searching by meaning, FakeEmbedder's
        from tests.test_meaning import GLOSSARY, MEANING, profile

        profile(c.session.root, "meaning", MEANING)
        library_add(c.session.root, GLOSSARY.encode(), "notes.md")
        run("profile meaning")
        assert c.session.conversation is not None
        assert c.session.conversation.search_method == "meaning"
        sent = run("say How do leaves turn light into sugar?")["call"]
        assert '<part document="notes"' in sent["prompt_text"]
    elif name == "failed call":
        backend.fail_with = "made-up failure"
        with pytest.raises(BackendError):
            run("say Are you there?")
        backend.fail_with = None
        run("say Again?")


WRITERS = [
    "say", "regen", "edit", "set", "title", "mark", "note", "attach", "failed call", "search",
    "meaning",
]  # fmt: skip


@pytest.mark.parametrize("name", WRITERS)
def test_t3_a_chat_that_uses_it_resumes_exports_and_replays(
    fetched_root, chat_tokenizer, tmp_path, name
):
    backend = FakeBackend([f" R{i}." for i in range(20)], tokenizer=chat_tokenizer)
    session = Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: chat_tokenizer,
        embedder=FakeEmbedder(),  # ty: ignore[invalid-argument-type]
    )
    write_chat(name, Commands(session), backend, tmp_path)
    assert session.conversation is not None
    log = session.conversation.log.path
    session.new_conversation("chat")
    record = session.resume(log)
    assert record["render"] is None or record["render"]["identical"], record["render"]
    exported = export_log(session, log)["path"]
    assert exported
    if name == "meaning":  # each row's found names its method (the whole-branch review's #10)
        rows = [
            json.loads(line) for line in Path(exported).read_text(encoding="utf-8").splitlines()
        ]
        assert {f["method"] for row in rows for f in row.get("found") or []} == {"meaning"}
    calls = [r for r in read_records(log) if r["type"] == "call" and r.get("turn_id")]
    assert calls
    for call in calls:  # with no chat open, as `clarity replay` does
        again = FakeBackend([call["output_text"]], tokenizer=chat_tokenizer)
        fresh = Session(  # replay never searches: an embedder used fails the test
            fetched_root,
            backend=again,
            tokenizer_loader=lambda r, v: chat_tokenizer,
            embedder=Untouchable(),  # ty: ignore[invalid-argument-type]
        )
        assert fresh.replay(log, call["call_id"])["identical"], (name, call["call_id"])


# --- The Shapes table's `found` (Checkpoint 1's S9): what a search adds to a turn

FOUND_SHAPE = FOUND_KEYS


def test_found_has_the_shapes_table_s_keys():
    import sqlite3

    from clarity_chat.documents import FRAMING, fill, search_documents
    from tests.test_documents_found import counting_room
    from tests.test_documents_search import loaded

    text = "\n\n".join(f"## Wren {i}\nA made-up wren, number {i}." for i in range(12))
    docs = [loaded("birds", text)]
    room, _ = counting_room()
    found = fill(room, "wren", [], FRAMING, search_documents(docs, "wren"), 10**6, "default")
    assert set(found) == FOUND_SHAPE["found"]
    assert set(found["documents"][0]) == FOUND_SHAPE["documents"]
    assert set(found["engine"]) == FOUND_SHAPE["engine"]
    assert set(found["parts"][0]) == FOUND_SHAPE["parts"]
    assert set(found["left_out"][0]) == FOUND_SHAPE["left_out"]
    assert (found["engine"]["name"], found["engine"]["sqlite"]) == (
        "sqlite-fts5",
        sqlite3.sqlite_version,
    )
