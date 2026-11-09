"""The `clarity` command: chat, run, replay, mark, note, fetch, verify."""

from __future__ import annotations

import argparse
import atexit
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError
from clarity_chat.profiles import ConfigError, find_root, load_models, load_server_config
from clarity_chat.session import CallOutcome, ContextError, Conversation, Session

HELP = """\
Enter sends; Alt+Enter (or Esc then Enter) starts a new line. Pasted text keeps its newlines.
Commands:
  /prompt         the exact prompt text of the last call, its token IDs and their count
  /prompt repr    the same, with whitespace and newlines made visible
  /settings       every setting, its value and where it came from
  /regen          another reply to your last message (seed + 1, + 2, …), as a new branch
  /edit [TEXT]    replace your last message (prefilled to edit) and get a new reply
  /undo           go back past the last exchange; your next message starts a branch
  /mark LABEL     label the current reply (for example landed, dropped)
  /note TEXT      add a note to the current reply
  /replay         re-send the current reply's call and compare the output
  /new            start a new conversation with the same profile
  /profile NAME   start a new conversation with another profile
  /quit           stop the server and exit (also Ctrl-D)
  //text          send a message that starts with "/"
Ctrl-C while the model writes cancels the reply (the partial reply is kept and logged)."""


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


def format_settings(session: Session) -> str:
    conversation = session.conversation
    assert conversation is not None
    profile = conversation.profile
    lines = [f"profile {profile.name} ({profile.mode} mode, model {profile.model})"]
    width = max(len(name) for name in profile.settings)
    for name, setting in profile.settings.items():
        value = setting.value
        note = "  (a new seed is drawn and logged for every call)" if value == "random" else ""
        lines.append(f"  {name:<{width}}  {value!r:<44} {setting.origin}{note}")
    last = conversation.last_call or {}
    backend_settings = last.get("backend_settings")
    if backend_settings is None:
        if session.backend.name == "transformers":
            from clarity_chat.backends.transformers_hf import NEUTRAL_CONFIG as neutral
        else:
            from clarity_chat.backends.llama_cpp import NEUTRAL_REQUEST as neutral

        backend_settings = {k: {"value": v, "origin": "backend"} for k, v in neutral.items()}
    lines.append(f"backend {session.backend.name}: fixed values for its other sampling fields")
    for name, entry in backend_settings.items():
        lines.append(f"  {name} = {entry['value']!r}")
    recommended = session.model_lock.get("recommended")
    if recommended:
        source = recommended["source"]
        used = "used" if profile.use_recommended else "not used: set use_recommended = true"
        values = ", ".join(f"{k} {v!r}" for k, v in recommended["values"].items()) or "none"
        lines.append(
            f"model recommends ({source['file']}@{source['revision'][:7]}; {used}): {values}"
        )
        if recommended["not_applied"]:
            lines.append(f"  not applied: {recommended['not_applied']}")
    not_applied = last.get("not_applied") or []
    lines.append("not applied:" + ("" if not_applied else " none"))
    lines += [f"  {n['name']}: {n.get('reason', 'no reason given')}" for n in not_applied]
    lines.append(f"context {session.backend.n_ctx} tokens; stop {profile.stop!r}")
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
    ) -> None:
        """`read(default)` returns one input, prefilled with `default` (for /edit)."""
        self.session = session
        self.profile_name = profile_name
        self.read = read
        self.write = write

    def start(self) -> None:
        self.write(f"Starting profile {self.profile_name} …\n")
        conversation = self.session.new_conversation(self.profile_name)
        self.show_server_warnings(conversation)
        self.write(
            f"Ready: {conversation.profile.model} ({conversation.profile.mode} mode), "
            f"conversation {conversation.conversation_id}. /help for commands.\n"
        )

    def run(self) -> None:
        self.start()
        while True:
            try:
                line = self.read("")
            except KeyboardInterrupt:
                continue
            except EOFError:
                return
            if not self.handle(line):
                return

    def handle(self, line: str) -> bool:
        """Run one input. Returns False to quit."""
        if line.startswith("//"):
            return self.send(line[1:])
        if line.startswith("/"):
            command, raw = split_command(line)
            if command == "edit":
                return self.command(command, raw)  # the new message exactly as typed
            return self.command(command, raw.strip())
        if not line.strip():
            return True
        return self.send(line)

    def command(self, command: str, arg: str) -> bool:
        session = self.session
        conversation = session.conversation
        if command in ("quit", "exit", "q"):
            return False
        if conversation is None and command not in ("help", "profile"):
            self.write("No conversation is running. Start one with /profile NAME.\n")
            return True
        if command == "help":
            self.write(HELP + "\n")
        elif command == "prompt":
            assert conversation is not None
            self.write(format_prompt(conversation.last_call, as_repr=arg == "repr") + "\n")
        elif command == "settings":
            self.write(format_settings(session) + "\n")
        elif command in ("new", "profile"):
            if command == "profile" and not arg:
                self.write("Usage: /profile NAME\n")
                return True
            # /new reloads the profile's own file, so a profile given as a path is found again.
            if command == "new":
                assert conversation is not None
            target = (
                arg
                if conversation is None or command == "profile"
                else str(conversation.profile.path)
            )
            try:
                new = session.new_conversation(target)
            except (ConfigError, BackendError) as e:
                self.write(f"! {e}\n")
                return True
            self.write(
                f"New conversation {new.conversation_id} with profile {new.profile.name} "
                f"({new.profile.model}, {new.profile.mode} mode).\n"
            )
            self.show_server_warnings(new)
        elif command == "regen":
            self.run_call(lambda: session.regen(on_text=self.write))
        elif command == "edit":
            self.edit(arg)
        elif command == "undo":
            try:
                head = session.undo()
            except ConfigError as e:
                self.write(f"! {e}\n")
                return True
            where = f"after: {head.text[:60]!r}" if head else "at the start"
            self.write(f"Undone. The next message continues {where}.\n")
        elif command in ("mark", "note"):
            if not arg:
                self.write(f"Usage: /{command} {'LABEL' if command == 'mark' else 'TEXT'}\n")
                return True
            try:
                record = session.annotate(command, arg)
            except ConfigError as e:
                self.write(f"! {e}\n")
                return True
            done = "Marked" if command == "mark" else "Noted"
            self.write(f"{done} {record['call_id']}.\n")
        elif command == "replay":
            self.replay()
        else:
            self.write(f"Unknown command /{command}. /help lists them.\n")
        return True

    def edit(self, arg: str) -> None:
        conversation = self.session.conversation
        assert conversation is not None
        try:
            reply = conversation.current_reply("edit the message of")
        except ConfigError as e:
            self.write(f"! {e}\n")
            return
        assert reply.parent_id is not None
        text = arg
        if not text:
            try:
                text = self.read(conversation.turns[reply.parent_id].text)
            except (KeyboardInterrupt, EOFError):
                text = ""
        if not text.strip():
            self.write("Edit cancelled.\n")
            return
        self.run_call(lambda: self.session.edit(text, on_text=self.write))

    def replay(self) -> None:
        try:
            record = self.session.replay_current()
        except (ConfigError, BackendError) as e:
            self.write(f"! {e}\n")
            return
        self.write(format_replay(record) + "\n")

    def show_server_warnings(self, conversation: Conversation) -> None:
        """The warnings of a server this conversation started (once per server)."""
        for warning in (conversation.started_server or {}).get("warnings", []):
            self.write(f"! {warning}\n")

    def send(self, text: str) -> bool:
        self.run_call(lambda: self.session.say(text, on_text=self.write))
        return True

    def run_call(self, call: Callable[[], CallOutcome]) -> None:
        """Make a call, streaming its reply, then show warnings, flags and a status line."""
        try:
            outcome = call()
        except (ContextError, ConfigError) as e:
            self.write(f"! {e}\n")
            return
        except BackendError as e:
            self.write(f"\n! Call failed: {e}\n")
            return
        self.write("\n")
        record = outcome.record
        for warning in outcome.warnings:
            self.write(f"! {warning}\n")
        if record.get("flags"):
            self.write("! flags: " + ", ".join(record["flags"]) + "\n")
        self.write(format_status(record) + "\n")


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


def cmd_chat(root: Path, profile: str) -> int:
    session = Session(root)
    atexit.register(session.close)  # also covers exits the finally below doesn't see
    try:
        ChatLoop(session, profile, prompt_reader(), write_stdout).run()
    finally:
        session.close()
    return 0


def cmd_run(root: Path, openings: Path, profiles: list[str], seeds: int) -> int:
    from clarity_chat.batch import RunError, run_batch
    from clarity_chat.logs import new_id

    run_id = new_id("run")
    run_dir = root / "runs" / run_id
    session = Session(root, logs_dir=run_dir)  # the server's output log stays with the run
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


def cmd_replay(root: Path, log_path: Path, call_id: str, logged_flags: bool) -> int:
    """Exit status 0 if the output is identical, 1 if it differs (or on an error)."""
    from clarity_chat.replay import find_call

    session = Session(root, logs_dir=log_path.parent)  # the server's output log goes alongside
    atexit.register(session.close)
    flags_from = ""
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


def cmd_annotate(log_path: Path, kind: str, call_id: str, text: str) -> int:
    """A review-time mark or note: appends one record to the log, starts nothing."""
    from clarity_chat.session import annotate_log

    record = annotate_log(log_path, kind, call_id, text)
    done = "Marked" if kind == "mark" else "Noted"
    print(f"{done} {call_id} ({record['turn_id']}) in {log_path}.")
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
    parser = argparse.ArgumentParser(
        prog="clarity",
        description="Clarity: what you see is exactly what the model got, and every call "
        "can be replayed.",
    )
    parser.add_argument("--root", type=Path, help="project folder (default: found from here)")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="chat with a profile")
    chat.add_argument("--profile", required=True, help="a profile name from profiles/")
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
    for kind, field_name, what in (("mark", "label", "label"), ("note", "text", "add a note to")):
        annotate = commands.add_parser(
            kind, help=f"{what} a logged call's reply while reviewing (appends to the log)"
        )
        annotate.add_argument("log", type=Path, help="a log file (logs/… or runs/<id>/calls.jsonl)")
        annotate.add_argument("call_id")
        annotate.add_argument(field_name, help="one argument: quote it if it has spaces")
    fetch = commands.add_parser("fetch", help="download (and convert) a model from models.toml")
    fetch.add_argument("name")
    verify = commands.add_parser("verify", help="re-hash fetched models against models.lock")
    verify.add_argument("names", nargs="*")
    args = parser.parse_args(argv)
    try:
        if args.command == "mark":  # a log file is all these need: no project root
            return cmd_annotate(args.log, "mark", args.call_id, args.label)
        if args.command == "note":
            return cmd_annotate(args.log, "note", args.call_id, args.text)
        root = args.root.resolve() if args.root else find_root()
        if args.command == "chat":
            return cmd_chat(root, args.profile)
        if args.command == "run":
            return cmd_run(root, args.openings, args.profiles, args.seeds)
        if args.command == "replay":
            return cmd_replay(root, args.log, args.call_id, args.logged_flags)
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
