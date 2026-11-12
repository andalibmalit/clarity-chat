"""`clarity chat --json`: the chat commands for scripts and other programs.

Each input line is one JSON object, `{"command": "say", "args": "Hello", "id": 1}`, where
`command` is any name in the command table (`/help` lists them), `args` is the text after
the command, and `id` (optional, any JSON value) is copied into what the command sends
back. Each output line is one JSON object with an `event`:

- `text`: reply text as it streams.
- `status`: a model is starting, or ready.
- `record`: a record just appended to the log. These are the log, in order.
- `result`: a command finished; `result` holds what it returns.
- `error`: a command failed (`type`, `message`), or a line wasn't a command.

A command's own records arrive as `record` events before its `result`, which may repeat
some of them; rely on `record` events for the log and on `result` for the answer.
Commands run one at a time, in order, except `cancel`, which runs as soon as it is read,
so it can stop a reply being written. End of input quits, like `quit`.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
from typing import IO, Any

from clarity_chat.backends import BackendError
from clarity_chat.commands import Commands
from clarity_chat.profiles import ConfigError
from clarity_chat.session import ContextError, Session


class JsonChat:
    def __init__(self, session: Session, stdin: IO[bytes], stdout: IO[str]) -> None:
        """`stdin` is bytes: input is UTF-8 whatever the console's encoding."""
        self.commands = Commands(session)
        self.stdin, self.stdout = stdin, stdout
        self.write_lock = threading.Lock()
        self.closed = False  # the reader of stdout went away: stop writing, then quit
        self.failed = False  # the start failed, or an unexpected error ended the session
        self.starts: list[dict[str, Any]] = []  # the --profile or --resume commands

    def emit(self, event: dict[str, Any]) -> None:
        """Write one event. Never raises: a log record's event is sent after the record is
        written, so an error here would leave the session out of step with its log."""
        if self.closed:
            return
        try:
            with self.write_lock:
                # ASCII with \u escapes: valid JSON whatever the console's encoding.
                self.stdout.write(json.dumps(event) + "\n")
                self.stdout.flush()
        except (OSError, ValueError):  # a closed pipe or file
            self.closed = True

    def run(self, first: list[dict[str, Any]]) -> None:
        """Run the `first` commands (from --profile or --resume), then each input line. If a
        first command fails, the session ends there, as the terminal's does."""
        pending: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.starts = list(first)
        for command in first:
            pending.put(command)
        threading.Thread(target=self.read, args=(pending,), daemon=True).start()
        while not self.closed:
            command = pending.get()
            if command is None:
                return
            if "error" in command:  # a line that wasn't a command, reported in its turn
                self.emit({"event": "error", "type": "InputError", "message": command["error"]})
            elif not self.execute(command):
                return

    def read(self, pending: queue.Queue[dict[str, Any] | None]) -> None:
        """Read input lines. `cancel` runs here at once; everything else, including a line
        that isn't a command, waits its turn."""
        try:
            for line in self.stdin:
                if not line.strip():
                    continue
                command = self.parse(line)
                if command.get("command") == "cancel":
                    self.execute(command)
                else:
                    pending.put(command)
        finally:
            pending.put(None)  # however reading ends, the main loop isn't left waiting

    @staticmethod
    def parse(line: bytes) -> dict[str, Any]:
        """A command, or {"error": why the line isn't one}."""
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as e:
            return {"error": f"Not UTF-8: {e}"}
        try:
            command = json.loads(text)
        except (ValueError, RecursionError) as e:  # RecursionError: nested too deeply
            return {"error": f"Not JSON: {e}"}
        if (
            not isinstance(command, dict)
            or not isinstance(command.get("command"), str)
            or not isinstance(command.get("args", ""), str)
        ):
            return {"error": 'Each line must be {"command": NAME, "args": TEXT, "id": ANY}.'}
        return command

    def execute(self, command: dict[str, Any]) -> bool:
        """Run one command and send its result or error. Returns False to quit."""
        name = command["command"]
        tag = {"command": name, **({"id": command["id"]} if "id" in command else {})}
        start = any(command is s for s in self.starts)  # --profile or --resume, not input
        try:
            result = self.commands.run(name, command.get("args", ""), self.emit, paths=start)
        except (ConfigError, ContextError, BackendError) as e:
            self.emit({"event": "error", **tag, "type": type(e).__name__, "message": str(e)})
            if start:
                self.failed = True
                return False
            return True
        except Exception as e:  # a bug or a full disk: tell the client, then quit
            traceback.print_exc()
            self.emit({"event": "error", **tag, "type": type(e).__name__, "message": str(e)})
            self.failed = True
            return False
        self.emit({"event": "result", **tag, "result": result})
        return name != "quit"
