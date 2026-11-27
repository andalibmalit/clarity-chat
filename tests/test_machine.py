"""Generated command sequences against the chat, with the guarantees `docs/design.md` states
checked after every step (v0.4.7 M2, a trial: `notes/plans/v0.4.7.md`'s side lane).

Hypothesis writes the sequences: start a chat; send a message that works, fails, is
cancelled before or during the reply, or finds the server stopped; regenerate, edit, undo,
switch; set a setting or the context size, with a start that works or is refused; title,
mark, note, replay, resume; replace the weights file; and run /state, /settings or /cancel
while a reply is written. After every step it checks the eight guarantees frozen in
`notes/qa/v0.4.7/machine-guarantees.md`, each only as far as its sentence says: it replays
the calls the step added, and resumes a copy of the log. A second machine sets the steering
strength on the real llama.cpp backend over the fake llama-server, since `FakeBackend.start`
ignores steering.

It reaches sequence bugs in the session, the command table and resume, such as E1, E11, E12
and v0.4.4's G3 (the size changed before its start worked). It doesn't reach E27, E34 or
E36 (they need the web views or attachments), the page's script, races inside the locked
sections, or two Clarity processes. The views rule was cut for size, as the plan orders.

A failure prints the shortest sequence that breaks a guarantee: make it a named test,
committed red before its fix.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from clarity_chat.backends import BackendError
from clarity_chat.commands import COMMANDS, Commands
from clarity_chat.logs import read_records
from clarity_chat.models import read_lock, write_lock_entry
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE, MODELS, REPO, TEXT_PROFILE, WORD_SEARCH, fake_fetch
from tests.fakes import FakeBackend

# The measured run (M2's first report): the file adds about 30 s to the offline tier.
CHAT_RUNS = settings(
    max_examples=80,
    stateful_step_count=40,
    derandomize=True,
    database=None,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
STEERING_RUNS = settings(CHAT_RUNS, max_examples=20, stateful_step_count=15)

# Each guarantee's sentence in docs/design.md, as M0 froze them. A changed sentence fails
# here until someone re-reads the check that rests on it.
GUARANTEES: dict[int | str, str] = {
    1: "Resume continues a conversation as it was logged",
    2: "Logs are append-only JSONL",
    3: (
        "Every turn's parent is an earlier turn in the same log, or none for a message of "
        "yours at the top of the chat (its first, an edit of that, or one sent after undoing "
        "back to the start), and a chat's current turn is always one its log holds."
    ),
    4: "every completed call can be replayed",
    5: "never adds an empty reply",
    "5, a cancel before the first token": (
        "A cancel before the first token is logged like a failed call, with `cancelled_during: "
        '"prompt"`, and your message is offered back.'
    ),
    6: (
        "Every read command answers for any reply in a log this version of Clarity wrote: "
        "the current one, or any by its call ID."
    ),
    7: (
        "a call never runs at a size the chat doesn't show. The same holds for steering: a "
        "new strength on llama.cpp is the chat's only once the start worked."
    ),
    8: "A new chat takes its settings from its profile, not from the chat before it.",
}

# The commands each rule runs. Every command is here or in NOT_RULES.
RULES = {
    "start": ("profile", "new"),
    "say": ("say",),
    "at_once": ("state", "settings", "cancel"),
    "regen": ("regen",),
    "edit": ("edit",),
    "undo": ("undo",),
    "switch": ("switch",),
    "set_setting": ("set",),
    "set_ctx": ("set",),
    "title": ("title",),
    "annotate": ("mark", "unmark", "note"),
    "replay": ("replay",),
    "resume": ("resume",),
    "replace_weights": (),
}
READS = ("history", "branches", "compare", "prompt", "probs", "chats")
NOT_RULES = {
    **dict.fromkeys(READS, "a read: guarantee 6 runs it after every step"),
    **dict.fromkeys(
        ("attach", "detach", "search", "try", "documents"),
        "attachments, documents and meaning search wait until the trial passes",
    ),
    "profiles": "lists the profile files, which no rule changes",
    "export": "writes a copy of the log for Inspect; no guarantee here speaks of the copy",
    "help": "lists the commands; it reads and changes no chat",
    "quit": "ends the terminal's loop; the session has nothing to check after it",
}

UNREADABLE = "Can't read this chat's records"
TEXTS = ("Hi.", "Tell me about kettles.", "Why?", "Say it again, shorter.")
BYTES = b"GGUF fake weights"  # fake_fetch's
OTHER_BYTES = b"GGUF faKe weights"  # the same size, not the same bytes

fired: Counter[str] = Counter()  # each rule's steps that changed what it should


def quiet(event: dict[str, Any]) -> None:
    pass


def refused(run: Callable[..., Any], *args: Any) -> Exception | None:
    """What `run(*args)` raised, if it was one of Clarity's own refusals; anything else
    fails the step."""
    try:
        run(*args)
    except (ConfigError, BackendError) as e:
        return e
    return None


def project(at: Path) -> Path:
    """conftest's `root` and `fake_fetch`, with the weights' true sha256 in the lock, so a
    file with the same bytes and a new time passes the start's check again."""
    (at / "profiles").mkdir(parents=True)
    for name in ("defaults.toml", "server.toml"):
        shutil.copy(REPO / name, at / name)
    (at / "models.toml").write_text(MODELS, encoding="utf-8")
    (at / "profiles" / "text.toml").write_text(TEXT_PROFILE, encoding="utf-8", newline="\n")
    chat_toml = CHAT_PROFILE + WORD_SEARCH
    (at / "profiles" / "chat.toml").write_text(chat_toml, encoding="utf-8", newline="\n")
    fake_fetch(at)
    entry = read_lock(at)["models"]["tiny"]
    entry["gguf"]["sha256"] = hashlib.sha256(BYTES).hexdigest()
    write_lock_entry(at, "tiny", entry)
    return at


def new_records(path: Path, seen: set[str]) -> list[dict[str, Any]]:
    """The call records in the log that `seen` doesn't hold yet, added to it."""
    calls = [r for r in read_records(path) if r["type"] == "call" and r["call_id"] not in seen]
    seen.update(r["call_id"] for r in calls)
    return calls


class Faulty(FakeBackend):
    """The fake backend, with the next `refusals` starts refused. A real failed start has
    stopped the old server, so a refusal stops it."""

    def __init__(self, tokenizer: Any) -> None:
        super().__init__([f" Reply {i}." for i in range(1000)], tokenizer=tokenizer)
        self.refusals = 0

    def start(self, model, weights, ctx_size, steering=None):
        if self.refusals:
            self.refusals -= 1
            self.stopped = True
            raise BackendError(f"llama-server didn't start at ctx_size {ctx_size} (made up).")
        return super().start(model, weights, ctx_size, steering)


class Machine(RuleBasedStateMachine):
    """What both machines share: a project, a session, and guarantees 1 and 4, each run in
    a second session on a copy of the log, since resume and replay append to it."""

    session: Session

    def __init__(self, where: Path, tokenizer: Any) -> None:
        super().__init__()
        self.where, self.tokenizer = where, tokenizer
        self.root = project(where / "root")
        self.scratch = where / "scratch"
        self.scratch.mkdir()
        self.weights_good = True
        self.calls_seen: set[str] = set()
        self.step: dict[str, Any] = {}  # what the last rule did, for the checks

    @property
    def chat(self):
        return self.session.conversation

    def fresh(self, backend: Any) -> Session:
        return Session(
            self.root,
            backend=backend,
            logs_dir=self.scratch,
            tokenizer_loader=lambda repo, revision: self.tokenizer,
        )

    def teardown(self) -> None:
        self.session.close()
        shutil.rmtree(self.where, ignore_errors=True)

    def in_fresh(self, backend: Any, name: str, run: Callable[[Session, Path], Any]) -> Any:
        """`run(session, copy)` in a fresh session on a copy of the chat's log: its answer,
        or the refusal the changed weights file must cause."""
        copy = self.scratch / f"{name}-{self.chat.log.path.name}"
        shutil.copyfile(self.chat.log.path, copy)
        session = self.fresh(backend)
        try:
            answer: list[Any] = []
            error = refused(lambda: answer.append(run(session, copy)))
            if not self.weights_good:
                assert error is not None and "changed since it was fetched" in str(error)
                return None
            assert error is None, f"/{name} was refused: {error}"
            return answer[0]
        finally:
            session.close()

    def live_equals_resumed(self, backend: Any) -> None:
        """1. A fresh session that resumes a copy of the log shows the same turns, head,
        unanswered message, settings in force and context size."""
        chat = self.chat
        resumed = self.in_fresh(backend, "resume", lambda s, c: (s.resume(c), s.conversation)[1])
        if resumed is not None:
            assert resumed.turns == chat.turns
            assert (resumed.head, resumed.unanswered) == (chat.head, chat.unanswered)
            assert resumed.settings == chat.settings
            assert resumed.ctx_size.value == chat.ctx_size.value

    def replays_identical(self, backend: Any, call: dict[str, Any]) -> None:
        """4. Replay, by a backend that returns the logged output, says Identical."""
        replayed = self.in_fresh(backend, "replay", lambda s, c: s.replay(c, call["call_id"]))
        assert replayed is None or replayed["identical"], replayed


# --- The chat, on the fake backend --------------------------------------------------------


def command(rule_name: str, name: str, args: st.SearchStrategy[str]) -> Any:
    """A rule that runs one command, with generated arguments, while a chat is open."""

    def run(self: Chat, args: str) -> None:
        self.run(rule_name, name, args)

    run.__name__ = rule_name
    return precondition(lambda self: self.chat is not None)(rule(args=args)(run))


class Chat(Machine):
    def __init__(self, where: Path, tokenizer: Any) -> None:
        super().__init__(where, tokenizer)
        self.backend = Faulty(tokenizer)
        self.session = Session(self.root, backend=self.backend, tokenizer_loader=self.load)
        self.c = Commands(self.session)
        self.logged: dict[Path, bytes] = {}  # each log's bytes after the last step
        self.turns: dict[str, tuple[str | None, str]] = {}  # every turn seen: parent, text

    def load(self, repo: str, revision: str) -> Any:
        return self.tokenizer

    def snapshot(self) -> tuple:
        chat = self.chat
        if chat is None:
            return (None, len(self.backend.started))
        size = chat.log.path.stat().st_size if chat.log.path.is_file() else 0
        state = (chat.head, chat.unanswered, dict(chat.chat_settings), chat.ctx_size)
        return (chat.conversation_id, *state, len(chat.turns), len(self.backend.started), size)

    def run(self, rule_name: str, name: str, args: str = "", **step: Any) -> None:
        before = self.snapshot()
        self.step = {"rule": rule_name, "args": args, "was_open": self.chat is not None, **step}
        self.step["error"] = refused(self.c.run, name, args, quiet)
        if self.snapshot() != before:
            fired[rule_name] += 1

    @initialize(profile=st.sampled_from(["text", "chat"]))
    def begin(self, profile: str) -> None:
        self.run("start", "profile", profile, started=True)

    @rule(how=st.sampled_from(["text", "chat", "new"]))
    def start(self, how: str) -> None:
        if how == "new" and self.chat is not None:
            self.run("start", "new", started=True)
        else:
            self.run("start", "profile", "text" if how == "new" else how, started=True)

    @precondition(lambda self: self.chat is not None)
    @rule(
        text=st.sampled_from(TEXTS),
        outcome=st.sampled_from(["works", "fails", "cancel_prompt", "cancel_reply", "stopped"]),
    )
    def say(self, text: str, outcome: str) -> None:
        if outcome == "fails":
            self.backend.fail_with = "The server answered 500 (made up)."
        self.backend.cancel_after = {"cancel_prompt": 0, "cancel_reply": 3}.get(outcome)
        self.backend.stopped = outcome == "stopped" or self.backend.stopped
        sent = len(self.backend.calls)
        try:
            self.run("say", "say", text)
        finally:
            self.backend.fail_with, self.backend.cancel_after = None, None
        reached = len(self.backend.calls) > sent and not self.backend.stopped
        self.step["cancelled_first"] = outcome == "cancel_prompt" and reached

    @precondition(lambda self: self.chat is not None)
    @rule(name=st.sampled_from(["state", "settings", "cancel"]), text=st.sampled_from(TEXTS))
    def at_once(self, name: str, text: str) -> None:
        """A read or a cancel sent while the reply is written, as a page would."""
        answers: list[Any] = []

        def meanwhile() -> None:
            try:
                answers.append(self.c.run(name, "", quiet))
            except Exception as e:  # noqa: BLE001 - checked below, on its own assertion
                answers.append(e)

        self.backend.on_call = meanwhile
        try:
            self.run("at_once", "say", text)
        finally:
            self.backend.on_call = None
        for answer in answers:
            assert isinstance(answer, dict), f"/{name} during a reply: {answer!r}"
            if name == "state":
                assert (answer["running"] or {}).get("command") == "say", answer["running"]
            if name == "cancel":
                assert answer["cancelled"] is True, answer
        self.step["cancelled_first"] = name == "cancel" and bool(answers)

    regen = command("regen", "regen", st.sampled_from(["", "2", "same"]))
    edit = command("edit", "edit", st.sampled_from(TEXTS))
    undo = command("undo", "undo", st.just(""))
    title = command("title", "title", st.sampled_from(["Kettles", "A short chat"]))
    settings = st.sampled_from(["temperature 0.2", "temperature", "top_k 5", "top_k"])
    set_setting = command("set_setting", "set", settings)

    @precondition(lambda self: self.chat is not None and self.chat.turns)
    @rule(i=st.integers(0, 40))
    def switch(self, i: int) -> None:
        turns = list(self.chat.turns)
        self.run("switch", "switch", turns[i % len(turns)])

    @precondition(lambda self: self.chat is not None)
    @rule(size=st.sampled_from(["1024", "2048", "4096", ""]), refusals=st.sampled_from([0, 1, 2]))
    def set_ctx(self, size: str, refusals: int) -> None:
        """A start that works, is refused (the old size starts again), or is refused twice
        (the old size can't start either, so the chat closes)."""
        self.backend.refusals = refusals
        try:
            self.run("set_ctx", "set", f"ctx_size {size}".strip(), shown=self.chat.ctx_size.value)
        finally:
            self.backend.refusals = 0

    def a_call(self, i: int) -> str:
        """The current reply (""), or any call by its ID."""
        calls = list(self.chat.calls)
        return calls[i % len(calls)] if calls and i >= 0 else ""

    @precondition(lambda self: self.chat is not None)
    @rule(kind=st.sampled_from(["mark", "unmark", "note"]), i=st.integers(-1, 40))
    def annotate(self, kind: str, i: int) -> None:
        note = " A made-up note." if kind == "note" else ""
        self.run("annotate", kind, (self.a_call(i) + note).strip())

    @precondition(lambda self: self.chat is not None)
    @rule(i=st.integers(-1, 40))
    def replay(self, i: int) -> None:
        self.run("replay", "replay", self.a_call(i))

    @precondition(lambda self: self.chat is not None)
    @rule()
    def resume(self) -> None:
        self.run("resume", "resume", self.chat.conversation_id)

    @rule(kind=st.sampled_from(["touched", "changed", "original"]))
    def replace_weights(self, kind: str) -> None:
        """The weights file written again between steps: the same bytes with a new time, other
        bytes of the same size, or the fetched bytes back."""
        path = self.root / "models" / "tiny" / "tiny-bf16.gguf"
        path.write_bytes(OTHER_BYTES if kind == "changed" else BYTES)
        self.weights_good = kind != "changed"
        self.step = {"rule": "replace_weights", "error": None}
        fired["replace_weights"] += 1

    # --- the guarantees, after every step ------------------------------------------------

    @invariant()
    def guarantees(self) -> None:
        if not self.step:
            return
        self.nothing_overwritten()  # 2
        chat = self.chat
        if chat is None:
            closed = self.step.get("was_open") and self.step["rule"] != "set_ctx"
            assert not closed, "only a start that can't restore its old size closes a chat"
            return
        log = list(read_records(chat.log.path)) if chat.log.path.is_file() else []
        new_calls = new_records(chat.log.path, self.calls_seen) if log else []
        self.tree_is_sound(log)  # 3
        self.runs_at_the_size_shown(new_calls)  # 7
        self.cancel_adds_no_empty_reply(log, new_calls)  # 5
        self.new_chat_starts_clean()  # 8
        self.reads_answer()  # 6
        self.live_equals_resumed(Faulty(self.tokenizer))  # 1
        for call in new_calls:  # 4
            if "error" not in call and call.get("stop_type") != "cancelled":
                self.replays_identical(
                    FakeBackend([call["output_text"]], tokenizer=self.tokenizer), call
                )

    def nothing_overwritten(self) -> None:
        """2. A log's earlier bytes never change, and every turn keeps its parent and text."""
        for path in sorted((self.root / "logs").glob("*.jsonl")):
            now = path.read_bytes()
            assert now.startswith(self.logged.get(path, b"")), f"{path.name}: bytes changed"
            self.logged[path] = now
            for r in read_records(path):
                if r["type"] == "turn":
                    kept = self.turns.setdefault(r["turn_id"], (r["parent_id"], r["text"]))
                    assert kept == (r["parent_id"], r["text"]), r["turn_id"]
        for turn in self.chat.turns.values() if self.chat else ():
            kept = self.turns.get(turn.turn_id, (turn.parent_id, turn.text))
            assert kept == (turn.parent_id, turn.text), turn.turn_id

    def tree_is_sound(self, log: list[dict[str, Any]]) -> None:
        """3. Every turn's parent is an earlier turn in the same log, or none for a message
        of yours; and the chat's current turn is one its log holds."""
        earlier: set[str] = set()
        for r in (r for r in log if r["type"] == "turn"):
            if r["parent_id"] is None:
                assert r["role"] == "user", f"{r['turn_id']} has no parent"
            else:
                assert r["parent_id"] in earlier, f"{r['turn_id']}'s parent {r['parent_id']}"
            earlier.add(r["turn_id"])
        assert self.chat.head is None or self.chat.head in earlier, self.chat.head

    def runs_at_the_size_shown(self, new_calls: list[dict[str, Any]]) -> None:
        """7. A call never runs at a size the chat doesn't show, and a size whose start was
        refused never becomes the chat's."""
        for call in new_calls:
            assert call["n_ctx"] == self.chat.ctx_size.value, call["call_id"]
        if self.step["rule"] == "set_ctx" and self.step["error"] is not None:
            assert self.chat.ctx_size.value == self.step["shown"], "a refused size was kept"

    def cancel_adds_no_empty_reply(self, log, new_calls) -> None:
        """5. No reply is empty. A cancel before the first token is logged like a failed
        call, with `cancelled_during: "prompt"`, and the message is offered back."""
        for r in log:
            if r["type"] == "turn" and r["role"] == "assistant":
                assert r["text"], f"an empty reply: {r['turn_id']}"
        if self.step.get("cancelled_first"):
            assert new_calls and new_calls[-1].get("cancelled_during") == "prompt", new_calls
            assert "error" in new_calls[-1], "not logged like a failed call"
            offered = (self.chat.unanswered or {}).get("text")
            assert offered == self.step["args"], "the message wasn't offered back"

    def new_chat_starts_clean(self) -> None:
        """8. A chat just started has the settings a chat on its profile starts with in a
        fresh session, and no setting of its own."""
        chat = self.chat
        if not self.step.get("started") or self.step["error"] is not None:
            return
        session = self.fresh(Faulty(self.tokenizer))
        try:
            other = session.new_conversation(str(chat.profile.path))
            assert (chat.settings, chat.ctx_size) == (other.settings, other.ctx_size)
            assert chat.chat_settings == {}
        finally:
            session.close()

    def reads_answer(self) -> None:
        """6. Every read answers for the current reply, and /prompt, /probs and /compare for
        any reply by its call ID. Away from a reply a read may refuse, but never as
        unreadable records."""
        head = self.chat.turns.get(self.chat.head or "")
        at_reply = head is not None and head.role == "assistant"
        reads = [(name, "") for name in (*READS, "settings", "state")] + [("branches", "all")]
        for name, args in reads:
            error = refused(self.c.run, name, args, quiet)
            assert error is None or UNREADABLE not in str(error), (name, str(error))
            assert error is None or not at_reply, f"/{name} {args} at a reply: {error}"
        for call_id, call in self.chat.calls.items():
            for name in ("prompt", "probs", "compare") if call.get("turn_id") else ():
                error = refused(self.c.run, name, call_id, quiet)
                assert error is None, f"/{name} {call_id}: {error}"


# --- The steering strength, on the real llama.cpp backend --------------------------------


class Steering(Machine):
    """/set steering_scale on a steered profile, as tests/test_steering_chat.py's `chat`
    fixture runs it: the real llama.cpp backend, with llama-server faked."""

    def __init__(self, where: Path, tokenizer: Any) -> None:
        from tests.fake_llama import fake_llama, startup
        from tests.test_steering import steer_root, steered
        from tests.test_steering_chat import session_on
        from tests.test_steering_llama_cpp import model_like

        super().__init__(where, tokenizer)
        steer_root(self.root)
        model_like(self.root)
        self.patch = pytest.MonkeyPatch()
        self.seen = fake_llama(self.patch, self.root, {"tiny": startup()}, tokenizer)
        self.session = session_on(self.root, tokenizer)
        self.c = Commands(self.session)
        self.c.run("profile", steered(self.root, backend="llama.cpp"), quiet)

    def teardown(self) -> None:
        super().teardown()
        self.patch.undo()

    def scale(self) -> float:
        return self.chat.settings["steering_scale"].value

    def backend(self, *replies: str) -> Any:
        from tests.test_steering_chat import backend_on

        self.seen["replies"].extend(replies)
        return backend_on(self.root)

    @rule(text=st.sampled_from(TEXTS))
    def say(self, text: str) -> None:
        self.step = {"rule": "say", "scale": self.scale()}
        assert refused(self.c.run, "say", text, quiet) is None
        fired["steering.say"] += 1

    @rule(scale=st.sampled_from(["0", "2.0", "4.0", "8.0"]), refuse=st.booleans())
    def steer(self, scale: str, refuse: bool) -> None:
        """A new strength whose start works, or whose vector fails to load once, so the old
        strength starts again."""
        before, starts = self.scale(), len(self.seen["argv"])
        if refuse:
            self.seen["cvec"] = "fails"
            self.seen["during_start"] = lambda: self.seen.update(cvec="loads", during_start=None)
        try:
            error = refused(self.c.run, "set", f"steering_scale {scale}", quiet)
        finally:
            self.seen.update(cvec="loads", during_start=None)
        self.step = {"rule": "steer", "error": error, "before": before}
        if len(self.seen["argv"]) > starts:
            fired["steering.steer"] += 1

    @invariant()
    def guarantees(self) -> None:
        from tests.test_steering_chat import each_call_ran_on_its_server

        assert self.chat is not None, "a strength's refused start closed the chat"
        if self.step.get("error") is not None:  # 7
            assert self.scale() == self.step["before"], "a refused strength was kept"
        running = (self.session.backend.server_record or {}).get("steering") or {}
        assert running.get("scale", 0) == self.scale(), (running, self.scale())
        each_call_ran_on_its_server(self.chat.log.path)
        for call in new_records(self.chat.log.path, self.calls_seen):
            steering = call["request"]["steering"] or {}
            assert steering.get("scale", 0) == self.step["scale"], call["call_id"]
            self.replays_identical(self.backend(call["output_text"]), call)  # 4
        self.live_equals_resumed(self.backend())  # 1


# --- The runs, and the guards against a test that checks nothing ------------------------


def test_the_chat_keeps_its_guarantees(tmp_path_factory, chat_tokenizer):
    """chat_tokenizer has a chat template, so both profiles start."""
    fired.clear()

    class Run(Chat):  # Hypothesis makes each machine with no arguments
        def __init__(self) -> None:
            super().__init__(tmp_path_factory.mktemp("machine"), chat_tokenizer)

    run_state_machine_as_test(Run, settings=CHAT_RUNS)
    never = [name for name in RULES if not fired[name]]
    assert not never, f"rules that never changed anything: {never}"


def test_a_new_strength_keeps_its_guarantees(tmp_path_factory, tiny_tokenizer):
    fired.clear()

    class Run(Steering):  # Hypothesis makes each machine with no arguments
        def __init__(self) -> None:
            super().__init__(tmp_path_factory.mktemp("machine"), tiny_tokenizer)

    run_state_machine_as_test(Run, settings=STEERING_RUNS)
    never = [name for name in ("say", "steer") if not fired[f"steering.{name}"]]
    assert not never, f"rules that never changed anything: {never}"


def test_every_command_is_a_rule_or_has_a_reason():
    ruled = {name for names in RULES.values() for name in names}
    assert not ruled & set(NOT_RULES), ruled & set(NOT_RULES)
    assert set(COMMANDS) == ruled | set(NOT_RULES)


def test_each_guarantee_s_sentence_is_still_in_the_design_doc():
    text = re.sub(r"\s+", " ", (REPO / "docs" / "design.md").read_text(encoding="utf-8"))
    missing = [n for n, sentence in GUARANTEES.items() if sentence not in text]
    assert not missing, f"guarantees whose sentence changed: {missing}"
