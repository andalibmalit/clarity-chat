"""The `clarity` command: chat, run, replay, mark, note, fetch, verify."""

from __future__ import annotations

import argparse
import atexit
import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.logs import torn_message
from clarity_chat.profiles import ConfigError, find_root, load_models, load_server_config
from clarity_chat.session import (
    CancelledBeforeReply,
    ContextError,
    Session,
)

OnServer = Callable[[dict[str, Any] | None], None]  # told each llama-server's port, PID, key

HELP_HEAD = (
    "Enter sends; Alt+Enter (or Esc then Enter) starts a new line. Pasted text keeps its "
    "newlines.\nCommands:"
)
HELP_TAIL = """\
  /prompt repr    the same as /prompt, with whitespace and newlines made visible
  //text          send a message that starts with "/"
Ctrl-C while the model works cancels the call; any partial reply is kept and logged."""


def format_help(result: dict[str, Any]) -> str:
    lines = [f"  {c['usage']:<15} {c['help']}" for c in result["commands"]]
    return "\n".join([HELP_HEAD, *lines, HELP_TAIL])


def split_command(line: str) -> tuple[str, str]:
    """ "/name rest" -> ("name", "rest"): the name ends at the first space or newline, and
    one separator is dropped, so the rest stays as typed."""
    body = line[1:]
    for i, char in enumerate(body):
        if char in " \n\t":
            return body[:i], body[i + 1 :]
    return body, ""


def visible_lines(text: str) -> list[str]:
    """`repr` of each line, split only on "\\n", so spaces, tabs and newlines show."""
    parts = text.split("\n")
    lines = [repr(part + "\n") for part in parts[:-1]]
    if parts[-1]:
        lines.append(repr(parts[-1]))
    return lines


def format_probs(result: dict[str, Any]) -> str:
    """A summary of a reply's token probabilities; --json gets every token."""
    import math

    if "unavailable" in result:
        return f"No token probabilities for this reply: {result['unavailable']}."
    tokens = result["logprobs"]["tokens"]
    lines = [
        f"{len(tokens)} tokens in {result['call_id']} ({result['logprobs']['view']}). "
        f"The {len(result['least_likely'])} least likely chosen, in reply order:"
    ]
    for t in result["least_likely"]:
        others = ", ".join(
            f"{o['text']!r} {math.exp(o['logprob']):.1%}" for o in t["top"] if o["id"] != t["id"]
        )
        lines.append(
            f"  #{t['index']} {t['text']!r} {math.exp(t['logprob']):.1%}"
            + (f"   model's likeliest others: {others}" if others else "")
        )
    return "\n".join(lines)


def format_prompt(record: dict[str, Any] | None, as_repr: bool) -> str:
    if record is None or "prompt_text" not in record:
        return "No call yet."
    bos = record.get("bos", {})
    bos_note = "BOS prepended" if bos.get("prepended") else "no BOS prepended"
    lines = [
        f"--- prompt: {record['n_prompt_client']} tokens; {bos_note} ({bos.get('reason')}) ---"
    ]
    if as_repr:
        lines += visible_lines(record["prompt_text"])
    else:
        lines.append(record["prompt_text"])
    lines.append(f"--- token IDs ({len(record['prompt_ids'])}) ---")
    lines.append(str(record["prompt_ids"]))
    return "\n".join(lines)


def format_settings(view: dict[str, Any]) -> str:
    profile = view["profile"]
    lines = [f"profile {profile['name']} ({profile['mode']} mode, model {profile['model']})"]
    width = max(len(name) for name in view["settings"])
    for name, setting in view["settings"].items():
        value = setting["value"]
        note = "  (a new seed is drawn and logged for every call)" if value == "random" else ""
        lines.append(f"  {name:<{width}}  {value!r:<44} {setting['origin']}{note}")
    lines.append(f"backend {view['backend']}: fixed values for its other sampling fields")
    for name, entry in view["backend_settings"].items():
        origin = "" if entry["origin"] == "backend" else f"  {entry['origin']}"
        lines.append(f"  {name} = {entry['value']!r}{origin}")
    if view["eog_tokens"]:
        listed = ", ".join(f"{t['id']} {t['text']!r}" for t in view["eog_tokens"])
        lines.append(f"end-of-reply tokens (from the model file): {listed}")
    elif view["backend"] == "llama.cpp":
        lines.append("end-of-reply tokens: not found in llama-server's startup log")
    recommended = view["recommended"]
    if recommended:
        source = recommended["source"]
        used = "used" if recommended["used"] else "not used: set use_recommended = true"
        values = ", ".join(f"{k} {v!r}" for k, v in recommended["values"].items()) or "none"
        lines.append(
            f"model recommends ({source['file']}@{source['revision'][:7]}; {used}): {values}"
        )
        if recommended["not_applied"]:
            lines.append(f"  not applied: {recommended['not_applied']}")
    not_applied = view["not_applied"]
    lines.append("not applied:" + ("" if not_applied else " none"))
    lines += [f"  {n['name']}: {n.get('reason', 'no reason given')}" for n in not_applied]
    ctx = view["ctx_size"]
    lines.append(f"context {ctx['value']} tokens ({ctx['origin']}); stop {view['stop']!r}")
    return "\n".join(lines)


def format_history(result: dict[str, Any]) -> str:
    """The conversation on the current branch, one block per turn, marks and notes under
    each reply."""
    if not result["turns"]:
        return "Nothing yet: this conversation starts with your next message."
    lines = []
    for turn in result["turns"]:
        who = "you" if turn["role"] == "user" else "model"
        lines.append(f"{who}: {turn['text'].strip()}")
        if turn["marks"]:
            lines.append("  marked: " + ", ".join(m["label"] for m in turn["marks"]))
        lines += [f"  note: {n['text']}" for n in turn["notes"]]
    return "\n".join(lines)


def format_branches(result: dict[str, Any]) -> str:
    """Each fork on the current path, its options numbered "2 of 3", the current one starred."""
    if not result["forks"]:
        return "This conversation has no branches yet. /regen, /edit and /undo make them."
    lines = ["/switch TURN_ID goes to the newest reply on a branch."]
    for fork in result["forks"]:
        options = fork["options"]
        if fork["at"] is None:
            where = "At the start"
        elif fork["at_role"] == "user":
            where = f"Your message {short(fork['at_text'], 50)}"
        else:
            where = f"After the reply {short(fork['at_text'], 50)}"
        if fork["at"] == result["head"]:
            where += ", where you are now (branches you went back from)"
        lines.append(f"{where}:")
        for i, option in enumerate(options, 1):
            star = "*" if option["current"] else " "
            who = "reply" if option["role"] == "assistant" else "message"
            text = short(option["text"], 50)
            lines.append(f" {star} {option['turn_id']}  {who} {i} of {len(options)}: {text}")
    return "\n".join(lines)


def format_chats(chats: list[dict[str, Any]]) -> str:
    """One line per conversation: ID (it starts with the local start time), profile, title."""
    if not chats:
        return "No conversations in logs/ yet."
    lines = []
    for chat in chats:
        if chat["error"]:
            lines.append(f"{chat['conversation_id']}  can't be read: {chat['error']}")
            continue
        title = chat["title"] if chat["title"] is not None else "(no messages)"
        lines.append(f"{chat['conversation_id']}  {chat['profile']}  {title}")
    return "\n".join(lines)


def render_line(render: dict[str, Any]) -> str:  # a check of a reply (not None)
    """The re-render check of a resumed conversation's current reply, in one line."""
    if render["identical"]:
        return "The current reply's prompt renders to the same token IDs today."
    if "error" in render:
        return f"The current reply's prompt no longer renders: {render['error']}"
    n = render["n_ids"]
    return (
        f"The current reply's prompt renders differently today ({n['logged']} token IDs "
        f"logged, {n['now']} now; first difference at token {render['first_different_token']}). "
        "Your next message is rendered today's way."
    )


def format_resume(result: dict[str, Any]) -> str:
    """What a resume restored and what differs today."""
    record = result["resume"]
    where = f"after: {result['head_text'][:60]!r}" if result["head_text"] else "at the start"
    lines = [
        f"Resumed {record['conversation_id']} with profile {result['profile']} "
        f"({result['model']}, {result['mode']} mode), as logged. It continues {where}. "
        "/history shows the conversation so far."
    ]
    lines += [f"! {warning}" for warning in result["warnings"]]
    if record["unanswered"]:
        lines.append(
            f"! Your last message got no reply (its call failed or was cut off): "
            f"{record['unanswered']['text'][:60]!r}. Send it again to get one."
        )
    if record["head_guessed"]:
        lines.append(
            "! This log is from v0.1.0, which didn't record /undo or failed calls, so it "
            "continues after the last reply. /branches shows the others."
        )
    if record["profile_file"] != "same":
        state = "has changed" if record["profile_file"] == "changed" else "is gone"
        lines.append(f"! The profile's file {state} since; this chat uses the profile as logged.")
    if record["ctx_size"]["origin"] != "log":
        lines.append(
            f"! The log doesn't say its context size: using {record['ctx_size']['value']} "
            f"({record['ctx_size']['origin']})."
        )
    if record["model_entry"]:
        changed = ", ".join(d["path"] for d in record["model_entry"])
        lines.append(f"! The model's models.toml entry has changed since: {changed}.")
    for d in record["settings_differ_today"]:
        if d["logged"] is None:
            lines.append(
                f"! The log has no {d['name']}: this chat uses {d['today']['value']!r} "
                f"({d['today']['origin']})."
            )
            continue
        today = d["today"] or {"value": "(none)", "origin": "-"}
        lines.append(
            f"! Today's files would set {d['name']} = {today['value']!r} ({today['origin']}); "
            f"this chat keeps {d['logged']['value']!r} ({d['logged']['origin']})."
        )
    if record["chat_settings"]:
        values = ", ".join(f"{k} = {v!r}" for k, v in record["chat_settings"].items())
        lines.append(f"Restored /set values: {values}.")
    if record["differences"]:
        lines.append(
            f"Changed since its last call ({record['compared_with_call']}): "
            + ", ".join(record["differences"])
            + ". The resume record lists each difference."
        )
    if record["not_recorded"]:
        lines.append(
            "Not compared, because the log doesn't record them: "
            + ", ".join(record["not_recorded"])
            + "."
        )
    render = record["render"]
    if render is not None:
        lines.append(("" if render["identical"] else "! ") + render_line(render))
    return "\n".join(lines)


def format_status(record: dict[str, Any]) -> str:
    timings = record.get("timings") or {}
    seconds = (timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000
    seed = record["settings"]["seed"]
    parts = [
        f"stop {record.get('stop_type')}",
        f"{len(record.get('output_ids', []))} tokens",
        f"{seconds:.1f} s",
        f"seed {seed['value']}" + (f" ({seed['origin']})" if seed["origin"] != "profile" else ""),
        f"context {record['n_prompt_client'] + len(record.get('output_ids', [])):,} of "
        f"{record['n_ctx']:,}",
        record["call_id"],
    ]
    return "[" + " · ".join(parts) + "]"


def short(value: Any, width: int = 70) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def format_replay(record: dict[str, Any]) -> str:
    """A replay's result: identical or not, and every recorded fact that differs."""
    if record["identical"]:
        lines = [f"Identical: same {record['compared']}. ({record['replay_id']})"]
    else:
        stop, n = record["stop_types"], record["n_output_ids"]
        where = (
            f"output token {record['first_different_token']}"
            if "first_different_token" in record
            else f"character {record['first_different_char']} of the text"
        )
        lines = [
            f"Different {record['compared']}: from {where};"
            f" stop {stop['logged']} -> {stop['now']}; {n['logged']} -> {n['now']} tokens."
            f" ({record['replay_id']})",
            f"  logged: {short(record['logged_output_text'], 200)}",
            f"  now:    {short(record['output_text'], 200)}",
        ]
    diffs = record["differences"]
    if not record["logged_server_found"]:
        lines.append("The log has no record of the server that made this call.")
    if diffs:
        lines.append("Differs from the logged call:")
        for fact, entries in diffs.items():
            for entry in entries[:5]:
                lines.append(
                    f"  {fact}: {entry['path']}: {short(entry['logged'])} -> {short(entry['now'])}"
                )
            if len(entries) > 5:
                lines.append(f"  {fact}: {len(entries) - 5} more (see the replay record)")
    else:
        lines.append(
            "Nothing recorded differs: build, flags, /props, environment, machine, model file, "
            "client."
        )
    if record.get("not_recorded"):
        missing = ", ".join(record["not_recorded"])
        lines.append(f"Not compared, because the log doesn't record them: {missing}.")
    if record["unexplained_difference"]:
        lines.append(
            "! The output differs, but nothing that could change it does. That is the "
            "nondeterminism replay exists to catch: keep this log."
        )
    if record["flags"]:
        lines.append("! flags: " + ", ".join(record["flags"]))
    return "\n".join(lines)


class ChatLoop:
    """Reads lines, runs commands, sends messages. Input and output are injectable for tests."""

    def __init__(
        self,
        session: Session,
        profile_name: str,
        read: Callable[[str], str],
        write: Callable[[str], None],
        resume: str | None = None,
    ) -> None:
        """`read(default)` returns one input, prefilled with `default` (for /edit).
        `resume`: a log to continue instead of starting with `profile_name`."""
        self.session = session
        self.profile_name = profile_name
        self.resume = resume
        self.read = read
        self.write = write
        self.draft = ""  # prefilled in the next input
        self.commands = Commands(session)

    def start(self) -> None:
        if self.resume is not None:
            self.write(f"Resuming {self.resume} …\n")
            result = self.commands.run("resume", self.resume, self.on_event)
            self.write(format_resume(result) + "\n")
            return
        self.write(f"Starting profile {self.profile_name} …\n")
        result = self.commands.run("profile", self.profile_name, self.on_event)
        for warning in result["warnings"]:
            self.write(f"! {warning}\n")
        self.write(
            f"Ready: {result['model']} ({result['mode']} mode), "
            f"conversation {result['conversation_id']}. /help for commands.\n"
        )

    def run(self) -> None:
        self.start()
        while True:
            draft, self.draft = self.draft, ""
            try:
                line = self.read(draft)
            except KeyboardInterrupt:
                continue
            except EOFError:
                return
            if not self.handle(line):
                return

    def handle(self, line: str) -> bool:
        """Run one input. Returns False to quit."""
        if line.startswith("//"):
            return self.command("say", line[1:])
        if line.startswith("/"):
            name, raw = split_command(line)
            name = {"exit": "quit", "q": "quit"}.get(name, name)
            if name == "edit":
                return self.edit(raw)  # the new message exactly as typed
            # A message goes as typed, like a plain line; other commands parse their args.
            return self.command(name, raw if name == "say" else raw.strip())
        if not line.strip():
            return True
        return self.command("say", line)

    def command(self, name: str, args: str) -> bool:
        """Run a registry command and show its result. Returns False to quit."""
        try:
            result = self.run_cancellable(name, args)
        except CancelledBeforeReply as e:
            offered = name in ("say", "edit")
            self.write(
                f"\n! {e} No reply was added."
                + (
                    " Your message is back in the input to edit or send again.\n"
                    if offered
                    else "\n"
                )
            )
            self.draft = args if offered else ""
            return True
        except BackendError as e:
            calling = name in ("say", "regen", "edit")
            self.write(f"\n! Call failed: {e}\n" if calling else f"! {e}\n")
            return True
        except (ContextError, ConfigError) as e:
            self.write(f"! {e}\n")
            return True
        if name in ("say", "edit"):
            self.write("\n")
            self.show_call(result)
        elif name == "regen":
            self.write("\n")
            replies = result["replies"]
            for k, reply in enumerate(replies, 1):
                if len(replies) > 1:
                    self.write(f"reply {k}: ")
                self.show_call(reply)
            if result.get("stopped"):
                self.write(f"! Stopped. {result['stopped']}\n")
            if len(replies) > 1:
                self.write("/branches lists them.\n")
        elif name == "quit":
            return False
        elif name == "help":
            self.write(format_help(result) + "\n")
        elif name == "prompt":
            self.write(format_prompt(result["call"], as_repr=args == "repr") + "\n")
        elif name == "settings":
            self.write(format_settings(result) + "\n")
        elif name == "probs":
            self.write(format_probs(result) + "\n")
        elif name in ("new", "profile"):
            self.write(
                f"New conversation {result['conversation_id']} with profile "
                f"{result['profile']} ({result['model']}, {result['mode']} mode).\n"
            )
            for warning in result["warnings"]:
                self.write(f"! {warning}\n")
        elif name == "undo":
            text = result["head_text"]
            where = f"after: {text[:60]!r}" if result["head"] else "at the start"
            self.write(f"Undone. The next message continues {where}.\n")
        elif name == "set":
            record = result["record"]
            now = result["setting"]
            if record["value"] is None:
                self.write(f"{record['name']} is back to {now['value']!r} ({now['origin']}).\n")
            else:
                self.write(f"{record['name']} = {record['value']!r} for this chat.\n")
        elif name == "title":
            self.write(f"Titled {result['record']['title']!r}.\n")
        elif name == "chats":
            self.write(format_chats(result["chats"]) + "\n")
        elif name == "history":
            self.write(format_history(result) + "\n")
        elif name == "branches":
            self.write(format_branches(result) + "\n")
        elif name == "switch":
            text = result["head_text"]
            self.write(f"Switched. The next message continues after: {text[:60]!r}.\n")
            if result["render"] and not result["render"].get("identical", True):
                self.write(f"! {render_line(result['render'])}\n")
        elif name == "resume":
            self.write(format_resume(result) + "\n")
        elif name in ("mark", "unmark", "note"):
            self.write(annotated(result["records"]) + ".\n")
        elif name == "replay":
            self.write(format_replay(result["replay"]) + "\n")
        elif name == "cancel":  # in the terminal, Ctrl-C cancels; between calls, nothing runs
            self.write("Cancelled.\n" if result["cancelled"] else "Nothing to cancel.\n")
        return True

    def show_call(self, result: dict[str, Any]) -> None:
        """A finished call's warnings, flags and status line."""
        for warning in result["warnings"]:
            self.write(f"! {warning}\n")
        if result["call"].get("flags"):
            self.write("! flags: " + ", ".join(result["call"]["flags"]) + "\n")
        self.write(format_status(result["call"]) + "\n")

    def run_cancellable(self, name: str, args: str) -> dict[str, Any]:
        """Run a command with Ctrl-C mapped to /cancel, so a call is cancelled wherever it
        is, not only while the stream is read. A command that can't be cancelled (a model
        start) keeps Ctrl-C's usual effect, so a long load can still be stopped."""
        import signal

        def cancel(signum: int, frame: Any) -> None:
            if not self.commands.run("cancel", "", lambda e: None)["cancelled"]:
                raise KeyboardInterrupt

        try:
            previous = signal.signal(signal.SIGINT, cancel)
        except ValueError:  # not the main thread: leave Ctrl-C as it is
            return self.commands.run(name, args, self.on_event)
        try:
            return self.commands.run(name, args, self.on_event)
        finally:
            signal.signal(signal.SIGINT, previous)

    def on_event(self, event: dict[str, Any]) -> None:
        if event["event"] == "text":
            self.write(event["text"])
        elif event["event"] == "reply_start" and event["n"] > 1:
            gap = "\n\n" if event["k"] > 1 else ""
            self.write(f"{gap}— reply {event['k']} of {event['n']}, seed {event['seed']} —\n")
        elif event["event"] == "record" and event["record"]["type"] == "torn_line":
            self.write(f"\n! {torn_message(event['record'], "this conversation's log")}\n")

    def edit(self, arg: str) -> bool:
        """/edit with no text prefills your last message to edit, then runs the command."""
        conversation = self.session.conversation
        if not arg and conversation is not None:
            try:
                reply = conversation.current_reply("edit the message of")
            except ConfigError as e:
                self.write(f"! {e}\n")
                return True
            assert reply.parent_id is not None
            try:
                arg = self.read(conversation.turns[reply.parent_id].text)
            except (KeyboardInterrupt, EOFError):
                arg = ""
        if conversation is not None and not arg.strip():
            self.write("Edit cancelled.\n")
            return True
        return self.command("edit", arg)


def prompt_reader() -> Callable[[str], str]:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()

    @bindings.add("enter")
    def _send(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    prompt_session: PromptSession[str] = PromptSession(multiline=True, key_bindings=bindings)
    return lambda default: prompt_session.prompt("> ", default=default)


def write_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def cmd_chat(root: Path, profile: str, on_server: OnServer, resume: str | None) -> int:
    session = Session(root, on_server=on_server)
    atexit.register(session.close)  # also covers exits the finally below doesn't see
    try:
        ChatLoop(session, profile, prompt_reader(), write_stdout, resume).run()
    finally:
        session.close()
    return 0


def cmd_chat_json(root: Path, profile: str | None, resume: str | None, on_server: OnServer) -> int:
    from clarity_chat.json_chat import JsonChat

    session = Session(root, on_server=on_server)
    atexit.register(session.close)
    first = [{"command": "profile", "args": profile}] if profile else []
    first += [{"command": "resume", "args": resume}] if resume else []
    chat = JsonChat(session, sys.stdin.buffer, sys.stdout)
    try:
        chat.run(first)  # bytes: UTF-8 always
    finally:
        session.close()
    return 1 if chat.failed else 0


def cmd_run(
    root: Path, openings: Path, profiles: list[str], seeds: int, on_server: OnServer
) -> int:
    from clarity_chat.batch import RunError, run_batch
    from clarity_chat.logs import new_id

    run_id = new_id("run")
    run_dir = root / "runs" / run_id
    # The server's output log stays with the run.
    session = Session(root, logs_dir=run_dir, on_server=on_server)
    atexit.register(session.close)
    try:
        blind = run_batch(session, run_id, run_dir, openings, profiles, seeds, write_stdout)
    except RunError as e:
        print(f"\nclarity: run stopped. {e}", file=sys.stderr)
        print(f"No blind.md was written. The calls so far are in {run_dir}.", file=sys.stderr)
        return 1
    finally:
        session.close()
    print(f"Done. Share {blind}.\nKeep {run_dir / 'key.json'} closed until rating is done.")
    return 0


def say_if_torn(log_path: Path) -> Callable[[dict[str, Any]], None]:
    """An on_record callback: says so when an append ends a cut-off line of `log_path`."""

    def check(record: dict[str, Any]) -> None:
        if record["type"] == "torn_line":
            print(f"clarity: {torn_message(record, str(log_path))}", file=sys.stderr)

    return check


def cmd_replay(
    root: Path, log_path: Path, call_id: str, logged_flags: bool, on_server: OnServer
) -> int:
    """Exit status 0 if the output is identical, 1 if it differs (or on an error)."""
    from clarity_chat.replay import find_call

    # The server's output log goes alongside.
    session = Session(root, logs_dir=log_path.parent, on_server=on_server)
    atexit.register(session.close)
    flags_from = ""
    session.on_record = say_if_torn(log_path)
    found = find_call(log_path, call_id)
    if logged_flags:
        server = found[1]
        if server is not None and server.get("backend", "llama.cpp") != "llama.cpp":
            raise ConfigError("--logged-flags applies only to calls made with llama-server.")
        if server is None:
            raise ConfigError(f"{log_path} has no server record for {call_id}'s server.")
        session.use_server_flags(server["flags"])
        flags_from = f"logged ({server['server_id']})"
    try:
        record = session.replay(log_path, call_id, flags_from=flags_from, found=found)
    finally:
        session.close()
    print(format_replay(record))
    return 0 if record["identical"] else 1


def annotated(records: list[dict[str, Any]]) -> str:
    """What a mark, unmark or note did: "Marked call-… landed", "Replaced the note on call-…"."""
    first = records[0]
    if first["type"] == "note":
        done = "Replaced the note on" if first.get("replaces") else "Noted"
        return f"{done} {first['call_id']}"
    labels = ", ".join(dict.fromkeys(r["label"] for r in records))
    return f"{'Marked' if first['type'] == 'mark' else 'Unmarked'} {first['call_id']} {labels}"


def cmd_annotate(log_path: Path, kind: str, call_id: str, text: str) -> int:
    """A review-time mark, unmark or note: appends to the log, starts nothing."""
    from clarity_chat.session import annotate_log

    records = annotate_log(log_path, kind, call_id, text, say_if_torn(log_path))
    print(f"{annotated(records)} ({records[0]['turn_id']}) in {log_path}.")
    return 0


def cmd_fetch(root: Path, name: str) -> int:
    from clarity_chat.models import fetch

    models = load_models(root)
    if name not in models:
        raise ConfigError(f"{name!r} is not in models.toml. Known: {sorted(models)}.")
    entry = fetch(root, models[name], load_server_config(root))
    print(f"Fetched {name}: {entry['gguf']['path']} (sha256 {entry['gguf']['sha256'][:16]}…)")
    return 0


def cmd_verify(root: Path, names: list[str]) -> int:
    from clarity_chat.models import read_lock, verify

    names = names or sorted(read_lock(root)["models"])
    ok_all = True
    for name in names:
        ok, message = verify(root, name)
        ok_all &= ok
        print(("ok    " if ok else "FAIL  ") + message)
    return 0 if ok_all else 1


def main(argv: list[str] | None = None) -> int:
    # Windows writes pipes and files in its ANSI code page, which can't hold every title or
    # reply; consoles are UTF-8 already. PowerShell 5.1 may still show the bytes wrongly.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and not stream.isatty():
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="clarity",
        description="Clarity: what you see is exactly what the model got.",
    )
    parser.add_argument("--root", type=Path, help="project folder (default: found from here)")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="chat with a profile")
    start = chat.add_mutually_exclusive_group()
    start.add_argument("--profile", help="a profile name from profiles/ or local/profiles/")
    start.add_argument(
        "--resume", help="a conversation to continue: its ID (from `clarity chats`) or log file"
    )
    chat.add_argument(
        "--json",
        action="store_true",
        help="read commands as JSON lines on stdin, write events as JSON lines (for scripts)",
    )
    run = commands.add_parser(
        "run", help="collect replies for a blind test (writes runs/<run-id>/blind.md)"
    )
    run.add_argument("--openings", type=Path, required=True, help="openings separated by ---")
    run.add_argument("--profiles", nargs="+", required=True, help="profile names or paths")
    run.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="replies per opening and profile; their seeds are 333, 334, … (default 1)",
    )
    replay = commands.add_parser(
        "replay", help="re-send a logged call and compare (exit 1 if the output differs)"
    )
    replay.add_argument("log", type=Path, help="a log file (logs/… or runs/<id>/calls.jsonl)")
    replay.add_argument("call_id")
    replay.add_argument(
        "--logged-flags",
        action="store_true",
        help="start llama-server with the call's logged flags instead of server.toml's",
    )
    for kind, what in (
        ("mark", "label a logged call's reply while reviewing (appends to the log)"),
        ("note", "add a note to a logged call's reply, replacing its note (appends to the log)"),
    ):
        annotate = commands.add_parser(kind, help=what)
        annotate.add_argument("log", type=Path, help="a log file (logs/… or runs/<id>/calls.jsonl)")
        annotate.add_argument("call_id")
        if kind == "mark":
            annotate.add_argument("label", nargs="?", default="", help="default: interesting")
            annotate.add_argument(
                "--remove", action="store_true", help="remove the mark (name it if several)"
            )
        else:
            annotate.add_argument("text", help="one argument: quote it if it has spaces")
    commands.add_parser("chats", help="list your conversations in logs/, newest first")
    fetch = commands.add_parser("fetch", help="download (and convert) a model from models.toml")
    fetch.add_argument("name")
    verify = commands.add_parser("verify", help="re-hash fetched models against models.lock")
    verify.add_argument("names", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "chat" and not (args.json or args.profile or args.resume):
        chat.error("one of --profile or --resume is required (or --json, to start by command)")
    try:
        if args.command == "mark":  # a log file is all these need: no project root
            kind = "unmark" if args.remove else "mark"
            return cmd_annotate(args.log, kind, args.call_id, args.label)
        if args.command == "note":
            return cmd_annotate(args.log, "note", args.call_id, args.text)
        root = args.root.resolve() if args.root else find_root()
        if args.command in ("chat", "run", "replay"):
            from clarity_chat import instance

            with instance.hold(root, args.command) as held:  # these start a model: one at a time
                if held.orphan:
                    print(f"clarity: {held.orphan}", file=sys.stderr)
                if args.command == "chat" and args.json:
                    return cmd_chat_json(root, args.profile, args.resume, held.record_server)
                if args.command == "chat":
                    return cmd_chat(root, args.profile, held.record_server, args.resume)
                if args.command == "run":
                    return cmd_run(
                        root, args.openings, args.profiles, args.seeds, held.record_server
                    )
                return cmd_replay(
                    root, args.log, args.call_id, args.logged_flags, held.record_server
                )
        if args.command == "chats":
            from clarity_chat.logs import list_chats

            print(format_chats(list_chats(root / "logs")))
            return 0
        if args.command == "fetch":
            return cmd_fetch(root, args.name)
        if args.command == "verify":
            return cmd_verify(root, args.names)
    except (ConfigError, BackendError) as e:
        print(f"clarity: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nclarity: interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
