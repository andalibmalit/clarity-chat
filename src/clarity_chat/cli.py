"""The `clarity` command: chat, fetch, verify."""

from __future__ import annotations

import argparse
import atexit
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError
from clarity_chat.profiles import ConfigError, find_root, load_models, load_server_config
from clarity_chat.session import ContextError, Session

HELP = """\
Enter sends; Alt+Enter (or Esc then Enter) starts a new line. Pasted text keeps its newlines.
Commands:
  /prompt         the exact prompt text of the last call, its token IDs and their count
  /prompt repr    the same, with whitespace and newlines made visible
  /settings       every setting, its value and where it came from
  /new            start a new conversation with the same profile
  /quit           stop the server and exit (also Ctrl-D)
  //text          send a message that starts with "/"
Ctrl-C while the model writes cancels the reply (the partial reply is kept and logged)."""


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
        from clarity_chat.backends.llama_cpp import NEUTRAL_REQUEST

        backend_settings = {
            k: {"value": v, "origin": "backend"} for k, v in NEUTRAL_REQUEST.items()
        }
    lines.append(f"backend {session.backend.name}: fixed values for its other sampling fields")
    for name, entry in backend_settings.items():
        lines.append(f"  {name} = {entry['value']!r}")
    not_applied = last.get("not_applied") or []
    lines.append("not applied: " + (", ".join(n["name"] for n in not_applied) or "none"))
    lines.append(f"context {session.backend.n_ctx} tokens; stop {profile.stop!r}")
    return "\n".join(lines)


def format_status(record: dict[str, Any]) -> str:
    timings = record.get("timings") or {}
    seconds = (timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000
    parts = [
        f"stop {record.get('stop_type')}",
        f"{len(record.get('output_ids', []))} tokens",
        f"{seconds:.1f} s",
        f"seed {record['settings']['seed']['value']}",
        record["call_id"],
    ]
    return "[" + " · ".join(parts) + "]"


class ChatLoop:
    """Reads lines, runs commands, sends messages. Input and output are injectable for tests."""

    def __init__(
        self,
        session: Session,
        profile_name: str,
        read: Callable[[], str],
        write: Callable[[str], None],
    ) -> None:
        self.session = session
        self.profile_name = profile_name
        self.read = read
        self.write = write

    def start(self) -> None:
        self.write(f"Starting profile {self.profile_name} …\n")
        self.session.new_conversation(self.profile_name)
        record = self.session.backend.server_record or {}
        for warning in record.get("warnings", []):
            self.write(f"! {warning}\n")
        conversation = self.session.conversation
        assert conversation is not None
        self.write(
            f"Ready: {conversation.profile.model} ({conversation.profile.mode} mode), "
            f"conversation {conversation.conversation_id}. /help for commands.\n"
        )

    def run(self) -> None:
        self.start()
        while True:
            try:
                line = self.read()
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
            command, _, arg = line[1:].partition(" ")
            return self.command(command.strip(), arg.strip())
        if not line.strip():
            return True
        return self.send(line)

    def command(self, command: str, arg: str) -> bool:
        session = self.session
        conversation = session.conversation
        assert conversation is not None
        if command in ("quit", "exit", "q"):
            return False
        if command == "help":
            self.write(HELP + "\n")
        elif command == "prompt":
            self.write(format_prompt(conversation.last_call, as_repr=arg == "repr") + "\n")
        elif command == "settings":
            self.write(format_settings(session) + "\n")
        elif command == "new":
            new = session.new_conversation(conversation.profile.name)
            self.write(f"New conversation {new.conversation_id}.\n")
        else:
            self.write(f"Unknown command /{command}. /help lists them.\n")
        return True

    def send(self, text: str) -> bool:
        try:
            outcome = self.session.say(text, on_text=self.write)
        except ContextError as e:
            self.write(f"! {e}\n")
            return True
        except BackendError as e:
            self.write(f"\n! Call failed: {e}\n")
            return True
        self.write("\n")
        record = outcome.record
        for warning in outcome.warnings:
            self.write(f"! {warning}\n")
        if record.get("flags"):
            self.write("! flags: " + ", ".join(record["flags"]) + "\n")
        self.write(format_status(record) + "\n")
        return True


def prompt_reader() -> Callable[[], str]:
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
    return lambda: prompt_session.prompt("> ")


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
    fetch = commands.add_parser("fetch", help="download (and convert) a model from models.toml")
    fetch.add_argument("name")
    verify = commands.add_parser("verify", help="re-hash fetched models against models.lock")
    verify.add_argument("names", nargs="*")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve() if args.root else find_root()
        if args.command == "chat":
            return cmd_chat(root, args.profile)
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
