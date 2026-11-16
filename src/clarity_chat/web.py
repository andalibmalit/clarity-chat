"""The web interface's server: a local page over the same command table as the terminal.

The route table, the checks and their reasons are in docs/design.md ("The web layer") and
docs/reference.md ("The web interface"). In short: the server listens on 127.0.0.1 only;
every response carries the security headers; the shell page and its static files hold no
data; every data route needs the per-start secret as a Bearer header, the exact Host, and
same-origin fetch metadata; a POST also needs the exact Origin and JSON. The secret comes to
the page in the link's fragment, which the server never sees.

One worker thread owns the Session (`Worker`): commands queue and run in order, as in
`clarity chat --json`, except those the command table lets run at once.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import html
import json
import math
import os
import queue
import re
import secrets
import socket
import sys
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route
from starlette.types import Message, Receive, Scope, Send

from clarity_chat.backends import CHECK_HELP, BackendError
from clarity_chat.backends.transformers_hf import FIXED_ORDER
from clarity_chat.commands import CALL_ID, READS, REPLY_COMMANDS, UPLOAD_ID, Commands
from clarity_chat.documents import (
    MAX_BYTES,
    SENTENCES,
    DocumentRefused,
    checked_name,
    counted,
    how_words,
    more_than,
    slow_read,
    too_big,
    visible_bidi,
)
from clarity_chat.json_chat import JsonChat
from clarity_chat.logs import unreadable
from clarity_chat.profiles import NEUTRAL, SETTING_HELP, SETTING_RANGES, ConfigError
from clarity_chat.replay import CANNOT_CHANGE_OUTPUT, cancelled_refusal, replay_notes
from clarity_chat.session import CONTEXT_WARN_FRACTION, ContextError, Session

PORT = 8765  # fixed, a constant: server.toml describes llama-server only
STATIC = Path(__file__).parent / "web_static"
STATIC_FILES = {  # nothing else is served: the name in the URL, its file and type
    "app.js": ("app.js", "text/javascript"),
    "app.css": ("app.css", "text/css"),
    "clarity-ds.css": ("clarity-ds.css", "text/css"),
    # The publishers' unmodified files, each family with its OFL.txt (fonts/*/).
    "Rasa-Regular.ttf": ("fonts/rasa/Rasa-Regular.ttf", "font/ttf"),
    "IBMPlexSans-Regular.woff2": ("fonts/ibm-plex-sans/IBMPlexSans-Regular.woff2", "font/woff2"),
    "IBMPlexSans-Medium.woff2": ("fonts/ibm-plex-sans/IBMPlexSans-Medium.woff2", "font/woff2"),
    "IBMPlexSans-SemiBold.woff2": ("fonts/ibm-plex-sans/IBMPlexSans-SemiBold.woff2", "font/woff2"),
    "IBMPlexMono-Regular.woff2": ("fonts/ibm-plex-mono/IBMPlexMono-Regular.woff2", "font/woff2"),
}
MAX_BODY = 64 * 1024  # any route's body
# /command's: any message a profile's context can take (the terminal and --json have no
# cap, so the page mustn't be the odd one out for a realistic message).
MAX_COMMAND = 8 * 1024 * 1024
HEARTBEAT = 15.0  # seconds between keep-alive lines on the event stream
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; "
    "frame-ancestors 'none'; require-trusted-types-for 'script'; trusted-types clarity"
)
HEADERS = [
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"cache-control", b"no-store"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cross-origin-opener-policy", b"same-origin"),
]
templates = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "web_templates"),
    autoescape=select_autoescape(default=True, default_for_string=True),
)
templates.filters["bidi"] = visible_bidi  # a document's text on screen (QA 1)


class Worker:
    """Owns the Session's command table on one thread. Each command waits its turn; the
    commands the table lets run during another (state, cancel, chats, profiles; settings
    during a reply) run at once on the caller's thread."""

    def __init__(self, session: Session) -> None:
        # The upload waiting to be attached, by its ID: one for the app, in memory, replaced
        # by the next and dropped when attached. The page's /attach takes only these IDs.
        self.uploads: dict[str, tuple[bytes, str]] = {}
        self.commands = Commands(session, uploads=self.uploads)
        self.jobs: queue.Queue[tuple[str, str, concurrent.futures.Future]] = queue.Queue()
        self.listeners: list[Callable[[dict[str, Any]], None]] = []  # one per open stream
        self.lock = threading.Lock()
        self.failed: str | None = None  # an unexpected error ended the session
        # This start's last /export answer, for the sheet and Download, with the log it was
        # for (by its file name: a copied log has the same conversation ID).
        self.exported: tuple[str, dict[str, Any]] | None = None
        self.probs: dict[str, dict[str, Any]] = {}  # /probs by call ID, for Details
        threading.Thread(target=self._loop, daemon=True, name="clarity-worker").start()

    def emit(self, event: dict[str, Any]) -> None:
        """Send an event to every open stream. Never raises: it runs inside the session (a
        record's event follows the record), so one broken stream, such as one whose event
        loop has closed, is dropped instead of ending the session."""
        with self.lock:
            listeners = list(self.listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - see above
                self.stop_listening(listener)

    def listen(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self.lock:
            self.listeners.append(listener)

    def stop_listening(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)

    def enqueue(self, name: str, args: str) -> tuple[bool, concurrent.futures.Future]:
        """Decide once whether `name` runs at once; if not, queue it. Returns the decision
        and the future its result arrives in (unused when it runs at once)."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        if self.commands.at_once_now(name):
            return True, future
        self.jobs.put((name, args, future))
        return False, future

    def _loop(self) -> None:
        while True:
            name, args, future = self.jobs.get()
            try:
                future.set_result(self._run(name, args))
            except BaseException as e:  # noqa: BLE001 - the thread must live to serve the next
                future.set_exception(e)

    async def ask(self, name: str, args: str) -> dict[str, Any]:
        """A command's answer, for an async route: at once on a thread if the table lets it
        run beside whatever runs, else from the queue, waited for in the event loop (so
        queued commands can't use up the threads `cancel` needs)."""
        now, future = self.enqueue(name, args)
        if now:
            return await run_in_threadpool(self._run, name, args, True)
        # Shielded: a request cancelled while its command waits (a dropped connection, a
        # server shutting down) mustn't cancel the queued future, which the worker would then
        # fail to answer, ending its thread. The command runs anyway, as a dropped POST's does.
        return await asyncio.shield(asyncio.wrap_future(future))

    def export_now(self) -> dict[str, Any]:
        """This start's last /export answer if it was for the log that's open, else {}."""
        conversation = self.commands.session.conversation
        if self.exported is None or conversation is None:
            return {}
        log, answer = self.exported
        return answer if log == conversation.log.path.stem else {}

    def _run(self, name: str, args: str, at_once: bool | None = None) -> dict[str, Any]:
        if self.failed:
            return {"error": self.failed, "type": "Stopped", "stopped": True}
        conversation = self.commands.session.conversation
        try:
            result = self.commands.run(name, args, self.emit, at_once=at_once)
        except (ConfigError, ContextError, BackendError) as e:
            if name == "export" and conversation is not None:
                self.exported = (conversation.log.path.stem, {"error": str(e)})
            return {"error": str(e), "type": type(e).__name__}
        except Exception as e:  # noqa: BLE001 - as in --json, it ends the session
            self.failed = f"Clarity stopped after an unexpected error: {type(e).__name__}: {e}"
            self.emit({"event": "stopped", "reason": self.failed})
            return {"error": self.failed, "type": "Stopped", "stopped": True}
        if name == "export":
            assert conversation is not None
            self.exported = (conversation.log.path.stem, {"result": result})
        return {"result": result}


class Security:
    """Every request: the exact Host, no OPTIONS, and the security headers on every
    response, refusals and server errors included. It wraps the whole app, outside
    Starlette's error handling, so a 500 gets them too."""

    def __init__(self, app: Starlette, port: int) -> None:
        self.app, self.host = app, f"127.0.0.1:{port}".encode()
        self.routes, self.state = app.routes, app.state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # none is served: refused before any route sees it
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":  # lifespan, which `clarity ui` turns off
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *HEADERS]
            await send(message)

        host = dict(scope["headers"]).get(b"host")
        if host != self.host:
            await PlainTextResponse("Wrong host.", 403)(scope, receive, with_headers)
        elif scope["method"] == "OPTIONS":
            await PlainTextResponse("Not allowed.", 405)(scope, receive, with_headers)
        else:
            await self.app(scope, receive, with_headers)


class Refused(Exception):
    def __init__(self, status: int, why: str) -> None:
        super().__init__(why)
        self.status = status


async def read_json(request: Request, is_command: bool = True) -> dict[str, Any]:
    """The request's JSON object, by --json's rules for a command."""
    limit = MAX_COMMAND if is_command else MAX_BODY
    words = f"{limit // 1024 // 1024} MB" if limit >= 1024 * 1024 else f"{limit // 1024} KB"
    chunks, size = [], 0
    async for chunk in request.stream():  # counted as it arrives, chunked bodies too
        size += len(chunk)
        if size > limit:
            raise Refused(413, f"Too big: Clarity takes at most {words} at a time.")
        chunks.append(chunk)
    raw = b"".join(chunks)
    if is_command:
        parsed = JsonChat.parse(raw)
        if "error" in parsed:
            raise Refused(400, parsed["error"])
        return parsed
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Refused(400, f"Not JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise Refused(400, "Send a JSON object.")
    return parsed


def make_app(
    session: Session,
    secret: bytes,
    port: int = PORT,
    on_first_use: Callable[[], None] | None = None,
) -> Security:
    """The app. `on_first_use` runs once, at a new tab's first /events (the page the
    redirect file opened); `app.state.rearm(callback)` sets it again (the terminal's Enter
    makes a new redirect), and the tab already open can't use it up."""
    if len(secret) < 16:  # an empty one would let every request through
        raise ValueError("The secret must be at least 16 bytes.")
    worker = Worker(session)
    first_use = [on_first_use]
    origin = f"http://127.0.0.1:{port}"
    # The live tab; open streams per tab; tabs told they aren't live, which take over only
    # by POST /tab, never by reconnecting their stream.
    tabs: dict[str, Any] = {"live": None, "streams": {}, "told": set(), "seen": set()}
    reading = [False]  # an upload's body is being read (one event loop: no lock needed)

    def check(request: Request, post: bool = False, kind: str = "application/json") -> None:
        """The secret, fetch metadata and, for a POST, the Origin and the content type."""
        given = request.headers.get("authorization", "")
        token = given[7:].encode("utf-8", "replace") if given.startswith("Bearer ") else b""
        if not secrets.compare_digest(token, secret):
            raise Refused(
                401, "Open Clarity from the window where you started it: press Enter there."
            )
        if request.headers.get("sec-fetch-site", "same-origin") != "same-origin":
            raise Refused(403, "Only Clarity's own page can ask for this.")
        if post:
            if request.headers.get("origin") != origin:
                raise Refused(403, "Only Clarity's own page can ask for this.")
            if request.headers.get("content-type", "").split(";")[0].strip() != kind:
                raise Refused(415, "Send JSON." if kind == "application/json" else f"Send {kind}.")

    def live(tab: str) -> None:
        if tabs["live"] is None:
            tabs["live"] = tab
        if tab != tabs["live"]:
            raise Refused(409, "Clarity is open in another tab.")

    def shell(request: Request) -> Response:
        # Sentences from SENTENCES the page says before or without the server: the sheet's
        # "Stopped reading NAME." for a Cancel before the server has answered (the lead's
        # ruling on Checkpoint 3's point 2), the slow read's line, and "Not kept: NAMES."
        # after New chat, a profile, a chat or Quit drops attachments (the close's must A).
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        for attribute, sentence in (
            ("data-stopped", SENTENCES["stopped"]),
            ("data-slow", slow_read()),
            ("data-not-kept", SENTENCES["not_kept"]),
        ):
            page = page.replace(f'{attribute}=""', f'{attribute}="{html.escape(sentence)}"', 1)
        return HTMLResponse(page)

    def static(request: Request) -> Response:
        name = request.path_params["file"]
        if name not in STATIC_FILES:
            return PlainTextResponse("Not found.", 404)
        path, media_type = STATIC_FILES[name]
        return FileResponse(STATIC / path, media_type=media_type)

    async def command(request: Request) -> Response:
        try:
            check(request, post=True)
            parsed = await read_json(request)
            tab = parsed.get("tab")
            if not isinstance(tab, str):
                raise Refused(400, "A command names its tab.")
            live(tab)
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        answer = await worker.ask(parsed["command"], parsed.get("args", ""))
        response = json_response({"id": parsed.get("id"), **answer})
        if parsed["command"] == "quit" and "result" in answer:  # Quit Clarity, from the page
            response.background = BackgroundTask(lambda: app.state.quit())
        return response

    async def upload(request: Request) -> Response:
        """A file's bytes, kept in memory for /attach. Every check runs before any of the
        body is read (S4): the secret, the Origin, the type, the length's form, the tab, the
        name, then the length itself, so its refusal names the file (the board's ruling 4)."""
        try:
            check(request, post=True, kind="application/octet-stream")
            length = request.headers.get("content-length", "")
            if not re.fullmatch(r"[0-9]+", length):
                raise Refused(411, "Send the file's length (Content-Length).")
            tab = request.query_params.get("tab", "")
            if not tab:
                raise Refused(400, "An upload names its tab.")
            live(tab)
            try:  # a browser sends the file's own name; keep only its last part
                name = checked_name(re.split(r"[/\\]", request.query_params.get("name", ""))[-1])
            except DocumentRefused as e:
                raise Refused(400, str(e)) from e
            if len(length) > 18 or int(length) > MAX_BYTES:  # int() refuses over 4,300 digits
                raise Refused(413, too_big(name))
            if reading[0]:  # each holds up to 128 MB until it replaces the last: one at a time
                raise Refused(409, "Another file is being uploaded. Wait for it, then try again.")
            reading[0] = True
            try:
                chunks, size = [], 0
                async for chunk in request.stream():  # counted again: the length can lie
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise Refused(413, too_big(name))
                    chunks.append(chunk)
            finally:
                reading[0] = False
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        upload_id = f"upload-{secrets.token_hex(6)}"
        with worker.lock:
            worker.uploads.clear()
            worker.uploads[upload_id] = (b"".join(chunks), name)
        return JSONResponse({"upload": upload_id, "name": name, "bytes": size})

    async def take_tab(request: Request) -> Response:
        try:
            check(request, post=True)
            tab = (await read_json(request, is_command=False)).get("tab")
            if not isinstance(tab, str):
                raise Refused(400, "Name the tab.")
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        if tabs["live"] not in (None, tab):
            tabs["told"].add(tabs["live"])
        tabs["told"].discard(tab)
        tabs["live"] = tab
        worker.emit({"event": "not_live", "live": tab})
        return JSONResponse({"live": tab})

    async def events(request: Request) -> Response:
        try:
            check(request)
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        tab = request.query_params.get("tab", "")
        if tab not in tabs["seen"]:  # a new tab: the page a redirect file opened
            tabs["seen"].add(tab)
            if first_use[0] is not None:
                done, first_use[0] = first_use[0], None
                done()
        # Live at once if no live tab is listening, unless this tab was told it isn't live.
        if not tabs["streams"].get(tabs["live"]) and tab not in tabs["told"]:
            tabs["live"] = tab
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def push(event: dict[str, Any]) -> None:  # from the worker's thread
            loop.call_soon_threadsafe(inbox.put_nowait, event)

        live_here = tab == tabs["live"]

        async def lines() -> AsyncIterator[bytes]:
            # Registered here, where the finally below undoes it: a client gone before the
            # body starts never runs this, so it leaves nothing registered. Still before
            # hello, so nothing sent after hello is lost.
            if live_here:
                worker.listen(push)
                tabs["streams"][tab] = tabs["streams"].get(tab, 0) + 1
            try:  # a client that leaves cancels this at its next await
                yield json.dumps({"event": "hello"}).encode() + b"\n"
                if not live_here:
                    yield json.dumps({"event": "not_live", "live": tabs["live"]}).encode() + b"\n"
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(inbox.get(), HEARTBEAT)
                    except TimeoutError:
                        yield b'{"event": "heartbeat"}\n'
                        continue
                    yield event_line(event)
                    if event["event"] == "not_live" and event.get("live") != tab:
                        return
            finally:
                if live_here:
                    worker.stop_listening(push)
                    tabs["streams"][tab] -= 1

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    async def view(request: Request) -> Response:
        """A part of the page as HTML. Its data comes through the command table like any
        command's (Worker.ask), and each view reads only the parameters it checks."""
        try:
            check(request)
            renderer = VIEW_RENDERERS.get(request.path_params["name"])
            if renderer is None:
                raise Refused(404, "No such view.")
            call = request.query_params.get("call")
            if call is not None and not CALL_ID.fullmatch(call):
                raise Refused(400, "That isn't a call ID.")
            params: dict[str, str | None] = {"call": call}
            if request.path_params["name"] == "attach":
                params.update(attach_params(request.query_params))
            try:
                return HTMLResponse(await renderer(worker, params))
            except (KeyError, TypeError, AttributeError) as e:  # a record a read refuses too
                raise Refused(503, unreadable(e)) from e
        except Refused as e:
            return PlainTextResponse(str(e), e.status)

    def export(request: Request) -> Response:
        try:
            check(request)
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        result = worker.export_now().get("result")
        path = Path(result["path"]) if result else None
        if path is None or not path.is_file():
            return PlainTextResponse(NOT_EXPORTED, 409)
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    app = Starlette(
        routes=[
            Route("/", shell, methods=["GET"]),
            Route("/static/{file}", static, methods=["GET"]),
            Route("/command", command, methods=["POST"]),
            Route("/events", events, methods=["GET"]),
            Route("/view/{name}", view, methods=["GET"]),
            Route("/tab", take_tab, methods=["POST"]),
            Route("/export", export, methods=["GET"]),
            Route("/upload", upload, methods=["POST"]),
        ]
    )
    app.state.worker = worker
    app.state.rearm = lambda callback: first_use.__setitem__(0, callback)
    app.state.quit = lambda: None  # after a quit's answer is sent; serve() stops the server
    return Security(app, port)


def serve(
    root: Path,
    port: int = PORT,
    session_factory: Callable[[], Session] | None = None,
    opener: Callable[[Path], Any] | None = None,
    on_ready: Callable[[Any], None] | None = None,
    write: Callable[[str], Any] = lambda text: print(text, flush=True),  # a pipe sees it now
    read_lines: bool = True,
) -> int:
    """`clarity ui`: serve the page until Ctrl-C, Quit or a closed terminal. The link, with
    a secret made for this start in its fragment, is printed and opened through a private
    redirect file (a link in `open`'s arguments would show in other users' process lists).
    Enter prints and opens it again."""
    import uvicorn

    listener = listening_socket(port)  # bound and listening before any link exists
    secret = secrets.token_urlsafe(32)
    link = f"http://127.0.0.1:{port}/#key={secret}"
    redirect = root / "local" / "clarity-open.html"
    redirect.parent.mkdir(exist_ok=True)

    def write_redirect() -> None:
        redirect.unlink(missing_ok=True)  # one left before: this one is made new, private
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(redirect, flags, 0o600), "w", encoding="utf-8") as f:
            f.write(REDIRECT.format(link=html.escape(link, quote=True)))

    write_redirect()
    try:
        session = (session_factory or (lambda: Session(root)))()
    except BaseException:
        listener.close()
        raise
    app = make_app(
        session, secret.encode(), port, on_first_use=lambda: redirect.unlink(missing_ok=True)
    )
    open_page = opener or (lambda path: webbrowser.open(path.as_uri()))
    write(
        f"Clarity is at {link}\n(the link works until Clarity stops). Enter opens it again; "
        "Ctrl-C stops Clarity.\nKeep this window open while you use Clarity.\n"
    )
    open_page(redirect)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        access_log=False,  # its lines would hold request paths; the secret is in the fragment
        log_level="warning",
        lifespan="off",
        timeout_graceful_shutdown=2,  # an open event stream mustn't hold the exit
    )
    worker: Worker = app.state.worker

    class Server(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            # Ctrl-C or SIGTERM: the page says Clarity has stopped, sent before the server
            # closes the stream (a closed stream alone says it isn't answering, and the page
            # keeps trying).
            worker.emit({"event": "stopped"})
            super().handle_exit(sig, frame)

    server = Server(config)
    app.state.quit = lambda: setattr(server, "should_exit", True)
    if on_ready:
        on_ready(server)
    if read_lines:

        def again() -> None:
            for _ in sys.stdin:  # the redirect again, never the link: `open` shows its argv
                write_redirect()
                app.state.rearm(lambda: redirect.unlink(missing_ok=True))
                write(f"Clarity is at {link}\n")
                open_page(redirect)

        threading.Thread(target=again, daemon=True).start()
    try:
        server.run(sockets=[listener])
    finally:
        worker.commands.run("cancel", "", lambda event: None)  # a reply in progress stops
        with worker.commands.busy:  # and its record is written before the session closes
            pass
        session.close()
        redirect.unlink(missing_ok=True)
    return 0


def listening_socket(port: int) -> socket.socket:
    """127.0.0.1:`port`, bound and listening, so the page is never opened before the server
    can answer, and no other program can take the port in between. Refused here, with a
    sentence, if something else has it. SO_REUSEADDR lets Clarity start again at once after
    quitting (its own closed connections wait in TIME_WAIT); on Windows that option would
    let another program share the port, so SO_EXCLUSIVEADDRUSE is used instead."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(128)
    except OSError as e:
        sock.close()
        raise ConfigError(f"Can't start: port {port} is in use ({e.strerror}).") from e
    return sock


REDIRECT = """<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0;url={link}">
<title>Opening Clarity</title>
<p><a href="{link}">Open Clarity</a></p>
"""


def page_words(sentence: str) -> str:
    """A read's refusal as the page says it: without the terminal command's name it starts
    with ("/probs: Can't read this chat's records …"), which the page has no use for."""
    return re.sub(r"^/[a-z]+: (?=Can't read this chat's records)", "", sentence)


async def result_of(worker: Worker, name: str, args: str = "") -> dict[str, Any]:
    """A command's result for a view. Any error (the session stopped, or a read refused)
    answers 503 with its sentence, which the page shows."""
    answer = await worker.ask(name, args)
    if "error" in answer:
        raise Refused(503, page_words(answer["error"]))
    return answer["result"]


# Commands that only read (commands.READS). Queued, one of them sets /state's `running` while
# it runs (another view's /branches, say), but the chat isn't changing: a view mustn't wait
# for it. app.js keeps the same list (test_web_thread). /chats and /profiles answer at once,
# so they never wait their turn.


def changing(state: dict[str, Any]) -> dict[str, Any] | None:
    """The running command that changes the chat, or None when nothing does."""
    running = state["running"]
    return None if running is None or running["command"] in READS else running


def waits_words(state: dict[str, Any]) -> str | None:
    """Why a view's controls wait: a model start, a reply, or another command that changes
    the chat (an export, a switch); None when nothing does."""
    running = changing(state)
    if running is None:
        return None
    if state["model"] == "starting":
        return "Waits until the model has started."
    if running["command"] in REPLY_COMMANDS:
        return "Waits until the reply is done."
    return "Waits until the command is done."


def starting_profile(state: dict[str, Any], profiles: list[dict[str, Any]]) -> dict | None:
    """The profile a model start is for, as /profiles lists it, or None when nothing starts.
    /state's chat is still the one before the start, so the name comes from the command."""
    running = state["running"]
    if state["model"] != "starting" or running is None:
        return None
    same_chat = running["args"] == state["log_id"]  # Start the model again
    name = {
        "profile": running["args"],
        "new": state["profile"],
        "resume": state["profile"] if same_chat else None,
    }.get(running["command"])
    row = next((p for p in profiles if p["name"] == name), None)
    return row or {"name": name, "model_name": None, "backend": None}


def profile_line(profile: dict[str, Any]) -> str:
    """A picker row's second line: what it's for, where it runs, and whether it can start."""
    if profile["error"]:
        return profile["error"]
    parts = [profile["description"], f"runs on {profile['backend']}"]
    return " · ".join([p for p in parts if p] + ([] if profile["fetched"] else ["not fetched"]))


def picker_groups(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Profiles under their model, models with a fetched profile first, then in /profiles'
    order; profiles that don't load last."""
    groups: dict[str | None, dict[str, Any]] = {}
    for profile in profiles:
        group = groups.setdefault(
            profile["model"], {"model_name": profile["model_name"], "rows": []}
        )
        group["rows"].append(profile)
    readable = [g for model, g in groups.items() if model is not None]
    readable.sort(key=lambda g: not any(p["fetched"] for p in g["rows"]))
    return readable + ([groups[None]] if None in groups else [])


def steering_words(settings: dict[str, Any]) -> str | None:
    """How a new chat says it's steered: "steered ×0.5", "not steered" at 0, or nothing."""
    if settings["steering"] is None:
        return None
    return setting_words("steering_scale", settings["settings"]["steering_scale"]["value"])


async def context_figure(worker: Worker, state: dict[str, Any]) -> dict[str, Any] | None:
    """The last prompt's share of the context. A message refused for its size or by search
    isn't counted (nothing was sent); the reply at the head is."""
    call = (await result_of(worker, "prompt"))["call"]
    if call and (call.get("error") or "").startswith(("ContextError", "SearchRefused")):
        replies = [
            t["call_id"]
            for t in state["turns"]
            if t["role"] == "assistant" and CALL_ID.fullmatch(t["call_id"] or "")
        ]
        call = (await result_of(worker, "prompt", replies[-1]))["call"] if replies else None
    if not call or "n_prompt_client" not in call:
        return None
    n, ctx = call["n_prompt_client"], call["n_ctx"]
    return {
        "text": f"Context {n / ctx:.0%} · {n:,} of {ctx:,} tokens",
        "hot": n > CONTEXT_WARN_FRACTION * ctx,
    }


CHECK = "check_failed:"
FLAG_WARNINGS = {  # the facts line's words (teaching check, item 15)
    "special_tokens_in_content": "! text read as a control token",
    "incomplete_output_ids": "! some output token IDs are missing, so replay compares text",
    "text_differs_from_ids": "! the text differs from its token IDs",
}


def facts_line(call: dict[str, Any], steerable: bool) -> dict[str, Any]:
    """What the last reply's call record says under it: plain facts, joined by " · ", and
    each "!" item on a line of its own. `steerable`: the profile has a steering vector."""
    flags = call.get("flags") or []
    plain = []
    steering = (call.get("request") or {}).get("steering")
    if steering:
        plain.append(f"steered ×{steering['scale']}")
    elif steerable:
        plain.append("not steered")
    if "empty_reply" in flags:
        plain.append("empty reply (nothing, or only blank space)")
    if "length" in flags:  # last, so its next step follows it (the template's link)
        limit = call["settings"]["max_tokens"]["value"]
        plain.append(f"stopped at the token limit (max_tokens {limit})")
    if "cancelled" in flags:
        plain.append("stopped by you")
    warn = []
    ignored = sum(1 for e in call.get("not_applied") or [] if e.get("affects_output"))
    if ignored:
        noun = "setting" if ignored == 1 else "settings"
        warn.append({"text": f"! {ignored} {noun} not applied", "details": True})
    warn += [{"text": text, "details": False} for f, text in FLAG_WARNINGS.items() if f in flags]
    checks = [f.removeprefix(CHECK) for f in flags if f.startswith(CHECK)]
    if checks:
        noun = "check" if len(checks) == 1 else "checks"
        warn.append(
            {"text": f"! {len(checks)} {noun} failed: {', '.join(checks)}", "details": True}
        )
    warn += [{"text": f"! {w}", "details": False} for w in call.get("warnings") or []]
    return {
        "plain": plain,
        "context_full": "context_full" in flags,
        "token_limit": "length" in flags,
        "warn": warn,
        "restart": "server_running" in checks,
    }


def mark_label(turn: dict[str, Any]) -> str:
    """The bookmark's accessible name, by what the reply has (lanes file, rule 6)."""
    if turn["marks"]:
        labels = ", ".join(m["label"] for m in turn["marks"])
        return f"Marked: {labels}. Edit marks and note"
    return "Not marked, has a note. Open the note" if turn["notes"] else "Mark this reply"


def versions(turns: list[dict[str, Any]], forks: list[dict[str, Any]]) -> None:
    """Each turn at a fork learns which of its versions it is (k of n) and its neighbours'
    IDs, for your message's counter and a reply's "N replies"."""
    by_parent = {fork["at"]: fork["options"] for fork in forks}
    parent = None
    for turn in turns:
        ids = [o["turn_id"] for o in by_parent.get(parent, [])]
        if turn["turn_id"] in ids:
            k = ids.index(turn["turn_id"])
            turn["versions"] = {
                "k": k + 1,
                "n": len(ids),
                "prev": ids[k - 1] if k else None,
                "next": ids[k + 1] if k + 1 < len(ids) else None,
            }
        parent = turn["turn_id"]


def failed_line(call: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any] | None:
    """A failed message's line, from its call record: the message is back in the box
    (`unanswered`), and the record is the last call. A failed Regenerate isn't one (the head
    stays on the reply), so the page shows that from the command's answer."""
    if not call or not call.get("error") or not state["unanswered"]:
        return None
    kind, _, message = call["error"].partition(": ")
    # No controls without a call ID (a log another program wrote), as for a reply.
    call_id = call["call_id"] if CALL_ID.fullmatch(call["call_id"] or "") else None
    return {"kind": kind, "message": message, "call_id": call_id}


async def _thread(worker: Worker, params: dict[str, str | None]) -> str:
    """The thread, from /state: the turns, or the first screen (a pointer to earlier chats, a
    model starting, a new chat's profile) and the start warnings. While a command runs, only
    reads that run at once are asked, and the turns come without their controls (they wait);
    when idle, /branches, /settings and /prompt give the counters, the facts line and a
    failed message's line."""
    state = await result_of(worker, "state")
    turns = state["turns"]
    for turn in turns:  # a reply's controls name its call: none without a call ID (old logs)
        turn["controls"] = bool(CALL_ID.fullmatch(turn["call_id"] or ""))
        # Your message's attachments, each with its cost when attached, and Edit's line (C2).
        attached = turn.get("attachments") or []
        turn["attached"] = [
            SENTENCES["attached_line"].format(
                name=a["name"], how=how_words(a), tokens=f"{a['tokens']:,}"
            )
            for a in attached
        ]
        turn["keeps"] = (
            SENTENCES["keeps"].format(names=", ".join(a["name"] for a in attached))
            if attached
            else None
        )
    # Edit can't take waiting chips along: its box says which stay (the walkthrough's F10).
    waiting = [a["name"] for a in state["pending"]]
    stays = (
        SENTENCES["stays_waiting" if len(waiting) == 1 else "stay_waiting"].format(
            names=", ".join(waiting)
        )
        if waiting
        else None
    )
    # Your message's attachment line opens Details for its reply, at Attachments (F16).
    for turn, after in zip(turns, turns[1:], strict=False):
        if after["role"] == "assistant" and after["controls"]:
            turn["reply_call"] = after["call_id"]
    starting = has_chats = intro = facts = failed = None
    idle = changing(state) is None
    # Reopened with your last message unanswered: one sentence says it's back in the box,
    # in place of the failed line (V3Chats L5).
    back_in_box = state["resume_differ"] is not None and state["unanswered"] is not None
    if state["model"] == "starting":
        starting = starting_profile(state, (await result_of(worker, "profiles"))["profiles"])
        # A new chat, or another chat resumed: this chat's turns aren't drawn. Any other start
        # (Start the model again, a replay that restarts the server) keeps it on screen.
        running = state["running"] or {"command": None, "args": None}
        other = running["command"] == "resume" and running["args"] != state["log_id"]
        if running["command"] in ("profile", "new") or other:
            turns = []
    elif state["conversation_id"] is None:
        has_chats = bool((await result_of(worker, "chats"))["chats"])
    elif idle:
        settings = await result_of(worker, "settings")
        if not any(turn["role"] == "user" for turn in turns):
            intro = {"backend": settings["backend"], "steering": steering_words(settings)}
        versions(turns, (await result_of(worker, "branches"))["forks"])
        if turns and turns[-1]["role"] == "assistant" and turns[-1]["controls"]:
            call = (await result_of(worker, "prompt", turns[-1]["call_id"]))["call"]
            facts = facts_line(call, steerable=settings["steering"] is not None)
        if not back_in_box:
            failed = failed_line((await result_of(worker, "prompt"))["call"], state)
    users = [i for i, turn in enumerate(turns) if turn["role"] == "user"]
    return templates.get_template("thread.html").render(
        state=state,
        turns=turns,
        idle=idle,
        last_user=users[-1] if users else None,
        starting=starting,
        has_chats=has_chats,
        intro=intro,
        facts=facts,
        failed=failed,
        stopped=state["model"] == "stopped",
        mark_label=mark_label,
        differ=state["resume_differ"] if starting is None else None,
        differ_line=differ_line,
        changed_lines=changed_lines,
        back_in_box=back_in_box,
        stays=stays,
    )


async def _picker(worker: Worker, params: dict[str, str | None]) -> str:
    """The row under the message box: the picker button and its list of profiles, Settings
    with how many settings this chat changed, and the context figure."""
    state = await result_of(worker, "state")
    profiles = (await result_of(worker, "profiles"))["profiles"]
    starting = starting_profile(state, profiles)
    in_chat = state["conversation_id"] is not None and starting is None
    running = (changing(state) or {}).get("command")
    waits = (
        "Waits until the model has started."
        if starting
        else "Waits until the reply is done."
        if running in REPLY_COMMANDS
        else None
    )
    changed = context = None
    if in_chat:
        settings = (await result_of(worker, "settings"))["settings"]
        changed = sum(1 for s in settings.values() if s["origin"] == "chat")
        if running is None or running in REPLY_COMMANDS:  # /prompt runs at once then too
            context = await context_figure(worker, state)
    button = starting or (
        {"name": state["profile"], "model_name": state["model_name"]} if in_chat else None
    )
    return templates.get_template("picker.html").render(
        button=button,
        current=state["profile"] if in_chat else None,
        in_chat=in_chat,
        waits=waits,
        changed=changed,
        context=context,
        groups=picker_groups(profiles),
        nothing_fetched=not any(p["fetched"] for p in profiles),
        line=profile_line,
    )


async def _about(worker: Worker, params: dict[str, str | None]) -> str:
    """About Clarity: the version and commit running, and the llama.cpp commit it pins."""
    state = await result_of(worker, "state")
    return templates.get_template("about.html").render(
        clarity=state["clarity"], llama_cpp_commit=state["llama_cpp_commit"]
    )


# --- The replies pop-out and Compare (C3), from /compare


def value_words(value: Any) -> str:
    """A value as Python writes it (1.0, 1e-05), so the screen matches the log and /set."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def setting_words(name: str, value: Any) -> str:
    """A setting as a row or a card says it, when it's what differs."""
    if name == "steering_scale":
        return "not steered" if value == 0 else f"steered ×{value}"
    return f"{name} {value_words(value)}"


def changed_words(settings: dict[str, Any], joiner: str = " ") -> str:
    """A change as "now at …" and the divider say it: steering's strength by that word."""
    names = {"steering_scale": "strength"}
    return " and ".join(f"{names.get(n, n)}{joiner}{value_words(v)}" for n, v in settings.items())


def numbers(ks: list[int]) -> str:
    """Reply numbers as a range when they run on ("4–6"), else listed ("1, 3")."""
    run = ks == list(range(ks[0], ks[0] + len(ks))) and len(ks) > 2
    return f"{ks[0]}–{ks[-1]}" if run else ", ".join(str(k) for k in ks)


def pair_note(pair: dict[str, Any]) -> str | None:
    """What a matched pair shows instead of a split marker, if anything."""
    if pair["first_different_token"] is None:
        return "the same text"
    return "the same until the shorter one ends" if pair["prefix"] else None


STOPS = {"cancelled": "stopped by you", "limit": "stopped at the token limit"}


def same_seeds(n: int) -> str:
    return "the same seed" if n == 1 else f"the same {n} seeds"


def reply_rows(compared: dict[str, Any]) -> list[dict[str, Any]]:
    """The pop-out's rows: what differs (the seed, and the settings when the replies fall in
    groups), how each stopped and where its branch went, and a divider where a group starts,
    naming the change and whether its seeds are the group before's again."""
    replies, groups = compared["replies"], compared["groups"]
    rows: list[dict[str, Any]] = []
    for i, reply in enumerate(replies):
        words = [f"seed {reply['seed']}"]
        if len(groups) > 1:
            words += [setting_words(n, v) for n, v in groups[reply["group"]]["settings"].items()]
        if reply["stop_type"] in STOPS:
            words.append(STOPS[reply["stop_type"]])
        later = reply["later_turns"]  # said only for a reply that went on (rows show what differs)
        if later:
            words.append(f"continued, {later} more turn{'' if later == 1 else 's'}")
        divider = None
        if i and reply["group"] != replies[i - 1]["group"]:
            before = [r["seed"] for r in replies if r["group"] == replies[i - 1]["group"]]
            run = [r["seed"] for r in replies[i:] if r["group"] == reply["group"]]
            divider = "You set " + changed_words(groups[reply["group"]]["settings"], " to ")
            if run == before:
                divider += f", then chose “Again with {same_seeds(len(run))}”"
        rows.append({**reply, "words": " · ".join(words), "divider": divider})
    return rows


async def _replies(worker: Worker, params: dict[str, str | None]) -> str:
    """The replies pop-out for the message the call answers: its replies, More replies and
    the same seeds again (for your last message, where /regen acts), and Compare."""
    if not params["call"]:
        raise Refused(400, "Name a reply's call.")
    compared = await result_of(worker, "compare", params["call"])
    turns = (await result_of(worker, "state"))["turns"]
    last = len(turns) > 1 and turns[-2]["turn_id"] == compared["message"]["turn_id"]
    return templates.get_template("replies.html").render(
        n=len(compared["replies"]),
        rows=reply_rows(compared),
        last=last and turns[-1]["role"] == "assistant",
        again=compared["again"],
        again_words=changed_words(compared["again"]["now"]) if compared["again"] else None,
        same_seeds=same_seeds,
    )


async def _compare(worker: Worker, params: dict[str, str | None]) -> str:
    """Compare: the replies side by side. Matched seeds line up in rows, with a column per
    group of settings and a marker where each parts from the first column's; otherwise cards,
    with a warning when both the seed and a setting differ."""
    if not params["call"]:
        raise Refused(400, "Name a reply's call.")
    compared = await result_of(worker, "compare", params["call"])
    state = await result_of(worker, "state")
    settings = await result_of(worker, "settings")
    backend = settings["backend"]
    replies, groups = compared["replies"], compared["groups"]
    differ = [n for n in (groups[0]["settings"] if len(groups) > 1 else {})]
    names = " and ".join("steering" if n == "steering_scale" else n for n in differ)
    for reply in replies:
        words = [f"seed {reply['seed']}"]
        if len(groups) > 1:
            words += [setting_words(n, v) for n, v in groups[reply["group"]]["settings"].items()]
        reply["head"] = f"reply {reply['k']} of {len(replies)} · " + " · ".join(words)
        reply["stop"] = STOPS.get(reply["stop_type"])
        # The thread's bookmark on each card (E3), named for its reply as the board does; none
        # while a command changes the chat, or without a call ID (a log another program wrote).
        which = f"reply {reply['k']} of {len(replies)}"
        named = reply["marks"] or reply["notes"]
        reply["bookmark"] = (
            (f"{mark_label(reply)}, {which}" if named else f"Mark {which}")
            if changing(state) is None and CALL_ID.fullmatch(reply["call_id"] or "")
            else None
        )
    rows, columns, sentence = [], [], None
    if compared["pairs"]:
        pairs = {(p["a"], p["b"]): p for p in compared["pairs"]}
        for g, group in enumerate(groups):
            label = " · ".join(setting_words(n, v) for n, v in group["settings"].items())
            ks = [r["k"] for r in replies if r["group"] == g]
            columns.append({"label": label[:1].upper() + label[1:], "ks": numbers(ks)})
        for seed in dict.fromkeys(r["seed"] for r in replies):
            cells = [
                [r for r in replies if r["seed"] == seed and r["group"] == g]
                for g in range(len(groups))
            ]
            for i in range(max(len(c) for c in cells)):
                row = [c[i] if i < len(c) else None for c in cells]
                first = row[0]
                for cell in row[1:]:
                    pair = first and cell and pairs.get((first["call_id"], cell["call_id"]))
                    if pair:
                        cell["note"] = pair_note(pair)
                        cut = pair["first_different_char"]
                        if cell["note"] is None and cut is not None:
                            cell["parts"] = (cell["text"][:cut], cell["text"][cut:])
                rows.append({"seed": seed, "cells": row})
        # "differs only in" holds only when every reply had the same prompt.
        complete = (
            len(groups) == 2 and all(all(row["cells"]) for row in rows) and compared["same_prompt"]
        )
        if complete:
            first_ks = [r["k"] for r in replies if r["group"] == 0]
            later_ks = [r["k"] for r in replies if r["group"] == 1]
            sentence = (
                f"Replies {numbers(later_ks)} reused the seeds of replies {numbers(first_ks)}, "
                f"so each row differs only in {names}."
            )
    # Only what tells the replies apart from a default run: not a sampler at its neutral
    # value, an off setting, the sampler list, "not steered" on a profile with no vector, or
    # repeat_last_n when every penalty is neutral (it changes nothing then).
    shown = compared["same"]
    penalties = ("repeat_penalty", "presence_penalty", "frequency_penalty")
    penalties_off = all(n in shown and at_rest(n, shown[n]) for n in penalties)
    same = [
        setting_words(n, v)
        for n, v in shown.items()
        if n != "samplers"
        and not (n == "repeat_last_n" and penalties_off)
        and not (at_rest(n, v) and not (n == "steering_scale" and settings["steering"]))
    ]
    return templates.get_template("compare.html").render(
        compared=compared,
        replies=replies,
        rows=rows,
        columns=columns,
        sentence=sentence,
        mixed=len(groups) > 1 and not compared["pairs"],
        marked=any(cell and cell.get("parts") for row in rows for cell in row["cells"]),
        names=names,
        same=" · ".join(
            [f"profile {state['profile']}", f"{state['model_name']} · runs on {backend}", *same]
            + (["the prompt"] if compared["same_prompt"] else [])
        ),
    )


# --- Settings (C4a), from /settings

COMMON = ("temperature", "top_p", "max_tokens", "seed")
SLIDERS = {"temperature": (0, 2, 0.05), "top_p": (0, 1, 0.01), "min_p": (0, 1, 0.01)}
PENALTY_ROWS = ("repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty")
OFF = {"logprobs": 0, "steering_scale": 0.0}  # a sampler at its NEUTRAL value is "neutral"


def at_rest(name: str, value: Any) -> str | None:
    """ "neutral" for a sampler at the value that leaves it without effect (temperature 1
    isn't off), "off" for logprobs or steering at 0, else None."""
    if name in NEUTRAL:
        return "neutral" if value == NEUTRAL[name] else None
    return "off" if value == OFF.get(name) else None


def setting_row(name: str, setting: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """One row: its value and origin, its line of help, a slider where the boards draw one
    (with the neutral and the model's value marked), and its one way back when changed."""
    value, origin = setting["value"], setting["origin"]
    row: dict[str, Any] = {
        "name": name,
        "value": value,
        "origin": origin,
        "help": SETTING_HELP.get(name),
        "slider": SLIDERS.get(name),
        "integer": SETTING_RANGES[name][0] == "integer",
        "marks": [],
        "buttons": [],
    }
    if row["slider"]:
        if name in NEUTRAL:
            row["marks"].append((NEUTRAL[name], f"neutral {value_words(NEUTRAL[name])}"))
        if name in model:
            row["marks"].append((model[name], f"model {value_words(model[name])}"))
    if name == "steering_scale" and value != 0:  # rule 10: Off whenever it's on
        row["buttons"].append(("steering_scale 0", "Off (0)"))
    if origin == "chat":
        back = setting["profile"]
        rest = at_rest(name, back["value"]) if name != "steering_scale" else None
        label = (
            f"{rest.capitalize()} ({value_words(back['value'])})"
            if rest
            else f"Back to {value_words(back['value'])} ({back['origin']})"
        )
        row["buttons"].append((name, label))
    return row


def settings_panel(settings: dict[str, Any], waits: str | None) -> dict[str, Any]:
    """What the Settings panel draws, from /settings: Common, Steering, Probabilities (on
    Transformers), More settings, and the Transformers lines from `not_applied`."""
    values = settings["settings"]
    model = (settings["recommended"] or {}).get("values", {})
    transformers = settings["backend"] == "transformers"
    samplers = values["samplers"]["value"]
    more = ["min_p", "top_k"]
    if "penalties" in samplers:
        more += ["repeat_penalty"] if transformers else list(PENALTY_ROWS)
    unused = [{**e, "value": value_words(e["value"])} for e in settings["not_applied"]]
    return {
        "profile": settings["profile"]["name"],
        "waits": waits,
        "changed": [n for n, s in values.items() if s["origin"] == "chat"],
        "common": [setting_row(n, values[n], model) for n in COMMON],
        "steering": settings["steering"],
        "strength": setting_row("steering_scale", values["steering_scale"], model)
        if settings["steering"]
        else None,
        "logprobs": setting_row("logprobs", values["logprobs"], model) if transformers else None,
        "more": [setting_row(n, values[n], model) for n in more],
        "penalties_off": "penalties" not in samplers,
        "samplers": samplers,
        "samplers_origin": values["samplers"]["origin"],
        "transformers": transformers,
        "fixed_order": FIXED_ORDER,
        "unused": unused if transformers else [],
    }


async def _settings(worker: Worker, params: dict[str, str | None]) -> str:
    """The Settings panel. /settings answers during a reply or a file's read; its controls
    then wait, and say until when (the final fix review's should 5)."""
    state = await result_of(worker, "state")
    settings = await result_of(worker, "settings")
    command = (state["running"] or {}).get("command")
    waits = (
        "Changes wait until this reply is done."
        if command in REPLY_COMMANDS
        else "Changes wait until the file is read."
        if command == "attach"
        else None
    )
    return templates.get_template("settings.html").render(**settings_panel(settings, waits))


# --- Details (C4b), from /prompt CALL_ID

STOP_WORDS = {  # Stopped, in words, then the record's own code
    "eos": "end of text",
    "word": "at a stop word",
    "limit": "at the token limit",
    "cancelled": "stopped by you",
    "none": "none",
}


def unhighlighted(call: dict[str, Any]) -> str | None:
    """Why the prompt isn't highlighted, or None when it is: `message_spans` is
    {"unavailable": REASON}, or missing from a call logged before v0.2.1."""
    spans = call.get("message_spans")
    if spans is None:
        return "this call was logged before Clarity recorded which text the template wrote (v0.2.1)"
    if isinstance(spans, dict):  # a log another program wrote may give no reason
        return str(spans.get("unavailable", "the log gives no reason")).rstrip(".")
    return None


def prompt_pieces(call: dict[str, Any]) -> list[tuple[str, str | None]]:
    """The prompt text in pieces, each with who put it there: "template" when the template
    or the profile wrote it, "added" for a framing line or a document Clarity put in your
    message (`added`, C2), else None; one piece when the spans aren't there (see
    unhighlighted)."""
    text, pieces, at = call.get("prompt_text") or "", [], 0
    spans = call.get("message_spans")
    for span in spans if isinstance(spans, list) else []:
        start, end = span["chars"]
        if start > at:
            pieces.append((text[at:start], None))
        if span["kind"] in ("template", "profile"):
            pieces.append((text[start:end], "template"))
        else:
            inner = start
            for added in span.get("added", []):
                first, last = added["chars"]
                if first > inner:
                    pieces.append((text[inner:first], None))
                pieces.append((text[first:last], "added"))
                inner = max(inner, last)
            if end > inner:
                pieces.append((text[inner:end], None))
        at = max(at, end)
    if at < len(text):
        pieces.append((text[at:], None))
    return pieces


def attachment_rows(call: dict[str, Any], turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    # `turns`: /prompt's prompt_attachments, the call's own messages with attachments, on
    # whatever branch (the whole-branch review's must 1).
    """Details' Attachments row (C2): each document in this prompt and each framing line,
    the message it was sent with, and where it sits in the prompt's tokens with its count
    (the walkthrough's F3), from the spans' `added`."""
    by_id = {turn["turn_id"]: turn for turn in turns}
    spans = call.get("message_spans")
    rows = []
    for entry in spans if isinstance(spans, list) else []:
        for added in entry.get("added", []):
            turn = by_id[entry["turn_id"]]
            if added["what"] == "attachment":
                attachment = turn["attachments"][added["index"]]
                what = f"{attachment['name']} · {how_words(attachment)}"
            elif added["what"] == "framing":
                what = "Framing line"
            else:  # a found part: Details draws those in their own section (C3)
                continue
            start, end = added["tokens"]
            said = turn["text"] if len(turn["text"]) <= 40 else turn["text"][:40].rstrip() + "…"
            rows.append(
                {
                    "kind": added["what"],
                    "what": what,
                    "with": f"“{said}”",
                    "tokens": f"{start:,} to {end - 1:,} · {end - start:,} tokens",
                }
            )
    return rows


def stopped_words(call: dict[str, Any], eog: list[dict[str, Any]] | None) -> str:
    """ "end of text · eos · <|im_end|>": the words, the code, and what stopped it: the end
    token's text when llama-server's startup named it, else its ID; the stop word."""
    code = call.get("stop_type") or "none"
    words = [STOP_WORDS.get(code, code), code]
    if code == "eos" and call.get("output_ids"):
        end = call["output_ids"][-1]
        texts = {t["id"]: t["text"] for t in eog or []}
        words.append(texts.get(end, str(end)))
    elif code == "word":
        words.append(call.get("stopping_word", "").replace("\n", "\\n"))
    return " · ".join(words)


SHOWN_CHANGES = 5  # per fact, as the terminal shows them; the record has the rest


def changed_lines(differences: dict[str, list[dict[str, Any]]], record: str) -> list[str]:
    """Each recorded difference as "machine · macos.version 26.0.1 → 26.1" (board G4), at
    most SHOWN_CHANGES per fact, then how many more `record` holds. A replay's result in
    Details and a reopened chat's list of what changed both say changes this way."""

    def clipped(value: Any) -> str:  # a /props entry can be long; the record has it all
        words = value_words(value)
        return words if len(words) <= 70 else words[:69] + "…"

    lines = []
    for fact, entries in differences.items():
        lines += [
            f"{fact} · {e['path']} {clipped(e['logged'])} → {clipped(e['now'])}"
            for e in entries[:SHOWN_CHANGES]
        ]
        if len(entries) > SHOWN_CHANGES:
            lines.append(f"{fact} · {len(entries) - SHOWN_CHANGES} more in {record}")
    return lines


def replay_result(replay: dict[str, Any]) -> dict[str, Any]:
    """A replay's short label (labels: v030-teaching-check, item 22), what changed since the
    call (changed_lines), its notes and its own flags."""
    differences = replay.get("differences") or {}
    facts = [n for n in differences if n not in CANNOT_CHANGE_OUTPUT]
    changed = changed_lines({n: differences[n] for n in facts}, "the replay record")
    if replay["identical"]:
        label = "Identical"
    elif replay.get("unexplained_difference"):
        label = "The replay differs, and nothing that could explain it changed"
    else:
        noun = "thing" if len(facts) == 1 else "things"
        label = f"Different · {len(facts)} {noun} changed since"
    return {
        **replay,
        # None when the replay matched: the record has no such key, which the template
        # would read as present (Jinja's Undefined isn't None).
        "first_different_token": replay.get("first_different_token"),
        "label": label,
        "changed": changed,
        "notes": replay_notes(replay),
    }


def percent(chance: float) -> str:
    # Rounded to whole percents, 0.996 would read "100%", certain; it isn't.
    if chance >= 0.995:
        return "over 99%"
    return f"{chance:.0%}" if chance >= 0.01 else "under 1%"


def probs_section(probs: dict[str, Any] | None, reply: str) -> dict[str, Any]:
    """Probabilities, from /probs: each token's piece of the reply, its band of shading (under
    10%, 10–30%, 30–60%, or none), its chance and log-probability, and the likeliest tokens
    the record holds; or, without them, the reason code."""
    if probs is None:
        return {"reason": "waits"}
    if probs.get("reason"):
        return {"reason": probs["reason"], "why": probs["unavailable"]}
    tokens = []
    for t in probs["tokens"]:
        chance = math.exp(t["logprob"])
        band = "p0" if chance < 0.1 else "p1" if chance < 0.3 else "p2" if chance < 0.6 else ""
        tokens.append(
            {
                **t,
                "piece": reply[t["start"] : t["end"]] if t["start"] is not None else None,
                "band": band,
                "pct": percent(chance),
                "top": [
                    (o["text"], percent(math.exp(o["logprob"])), o["id"] == t["id"])
                    for o in t["top"]
                ],
            }
        )
    return {
        "reason": None,
        "n": probs["logprobs"]["n"],
        "tokens": tokens,
        "shaded": all(t["piece"] is not None for t in tokens),
    }


PROBS_KEPT = 32  # calls whose /probs result Details keeps: the record doesn't change


def record_text(call: dict[str, Any]) -> str:
    """The whole record, indented, with each list of numbers on one line (a prompt's IDs
    would take a line each). JSON strings hold no line breaks, so the pattern only meets
    the layout."""
    text = json.dumps(call, indent=2, ensure_ascii=False, default=str)
    return re.sub(
        r"\[\n\s*(-?\d+(?:\.\d+)?(?:,\n\s*-?\d+(?:\.\d+)?)*)\n\s*\]",
        lambda m: "[" + ", ".join(n.strip() for n in m.group(1).split(",")) + "]",
        text,
    )


async def read_or_refusal(worker: Worker, name: str, args: str) -> dict[str, Any]:
    """A read's answer, `result` or a refusal's `error`, for a view that draws without it.
    The session stopping still answers 503."""
    answer = await worker.ask(name, args)
    if answer.get("stopped"):
        raise Refused(503, answer["error"])
    return answer


async def _details(worker: Worker, params: dict[str, str | None]) -> str:
    """Details for one call: what the model got, the summary, Replay, the settings as logged
    and the whole record. /state comes first, so the queued reads after it (/compare,
    /probs) are asked only when nothing changes the chat; /prompt and /settings answer
    during a reply (a replay included)."""
    if not params["call"]:
        raise Refused(400, "Name a call.")
    state = await result_of(worker, "state")
    read = await result_of(worker, "prompt", params["call"])
    call, replays = read["call"], read["replays"]
    settings = await result_of(worker, "settings")
    turns = state["turns"]
    head = turns[-1]["call_id"] if turns and turns[-1]["role"] == "assistant" else None
    running = changing(state)
    k = n = None
    if call.get("turn_id") and running is None:  # /compare waits its turn
        compared = await read_or_refusal(worker, "compare", params["call"])
        if "result" in compared:  # refused (a log another program wrote): no "reply k of n"
            replies = compared["result"]["replies"]
            k = next((r["k"] for r in replies if r["call_id"] == call["call_id"]), None)
            n = len(replies)
    flags = call.get("flags") or []
    failed = [f.removeprefix(CHECK) for f in flags if f.startswith(CHECK)]
    failed += [
        c for c, v in (call.get("checks") or {}).items() if not v.get("ok") and c not in failed
    ]
    timings = call.get("timings") or {}
    known = "prompt_ms" in timings or "predicted_ms" in timings
    seconds = (timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000
    produced = call.get("tokens_predicted") or len(call.get("output_ids") or [])
    first = call.get("prompt_ids") or [None]
    error = call.get("error")
    replaying = running is not None and (running["command"], running["args"]) == (
        "replay",
        call["call_id"],
    )
    probs = worker.probs.get(call["call_id"])
    if probs is None and not error and running is None:  # /probs waits its turn
        read_probs = await read_or_refusal(worker, "probs", call["call_id"])
        probs = read_probs.get("result") or {
            "reason": "unreadable",
            "unavailable": page_words(read_probs["error"]),
        }
        worker.probs[call["call_id"]] = probs
        if len(worker.probs) > PROBS_KEPT:
            del worker.probs[next(iter(worker.probs))]
    return templates.get_template("details.html").render(
        call=call,
        k=k,
        n=n,
        earlier=bool(call.get("turn_id")) and call["call_id"] != head,
        error=error.partition(": ")[2] or error if error else None,
        # None when the call doesn't say (no prompt IDs, or no `bos` record): say neither.
        bos=None
        if first[0] is None or (call.get("bos") or {}).get("bos_token_id") is None
        else first[0] == call["bos"]["bos_token_id"],
        pieces=prompt_pieces(call),
        not_highlighted=unhighlighted(call),
        highlight_added=SENTENCES["highlight_added"],
        attachments=attachment_rows(call, read["prompt_attachments"]),
        attachments_caption=SENTENCES["attachments_caption"],
        attached=bool(read["prompt_attachments"]),  # a message in this prompt had some
        text_mode=(call.get("bos") or {}).get("mode") == "text",
        stopped=stopped_words(call, settings.get("eog_tokens")),
        length=f"{produced} tokens · " + (f"{seconds:.1f} s" if known else "time unknown"),
        model_line=f"{call['model']['tokenizer_repo'].rsplit('/', 1)[-1]} · runs on "
        f"{(call.get('backend') or {}).get('backend', 'llama.cpp')}",
        failed=failed,
        check_help=CHECK_HELP,
        checks=[
            (name, v.get("ok") and name not in failed)
            for name, v in (call.get("checks") or {}).items()
        ],
        ignored=[e for e in call.get("not_applied") or [] if e.get("affects_output")],
        replays=[replay_result(r) for r in replays],
        # A cancelled reply can't be replayed: said up front, with the refusal's sentence.
        unreplayable=cancelled_refusal(call["call_id"])
        if call.get("stop_type") == "cancelled"
        else None,
        replaying=replaying,
        waits=running is not None and not replaying,
        settings=[
            (name, value_words(v["value"]), at_rest(name, v["value"]), v["origin"])
            for name, v in call["settings"].items()
        ],
        seed_drawn=call.get("seed_drawn"),
        ctx=settings["ctx_size"],
        record=record_text(call),
        probs=probs_section(probs, call.get("output_text") or ""),
    )


# --- Branches (C4d), from /branches all


def branch_source(source: dict[str, Any] | None) -> str:
    """Where a branch left the rest: a reply ("reply 2 of 4 · seed 43") or your edited
    message ("your edited message · seed 42")."""
    if source is None:
        return ""
    what = (
        "your edited message"
        if source["kind"] == "edit"
        else f"reply {source['k']} of {source['n']}"
    )
    return f"{what} · seed {source['seed']}"


def end_label(end: dict[str, Any]) -> str:
    """A branch end's heading: where it came from, or, when it's that fork's own reply and
    went no further, which reply it is."""
    source = end["from"]
    if source is None:
        label = f"seed {end['seed']}"
    elif source["kind"] == "reply" and not end["continued"]:
        label = f"Reply {source['k']} of {source['n']} · seed {source['seed']}, not continued"
    else:
        label = "From " + branch_source(source)
    if end.get("steering_scale"):
        label += f" · steered ×{end['steering_scale']}"
    return label


async def _branches(worker: Worker, params: dict[str, str | None]) -> str:
    """The Branches panel: every branch end, newest first, and the marked replies ("Marked
    only", a filter in the page). /branches waits its turn, so it isn't asked mid-reply."""
    state = await result_of(worker, "state")
    if changing(state):
        return templates.get_template("branches.html").render(waits=waits_words(state))
    found = await result_of(worker, "branches", "all")
    ends = [{**e, "label": end_label(e)} for e in found["ends"]]
    marked = [
        {
            **m,
            "where": f"on the branch from {branch_source(m['from'])}" if m["from"] else "",
            "note": m["notes"][-1]["text"] if m.get("notes") else None,
        }
        for m in found["marked"]
    ]
    return templates.get_template("branches.html").render(waits=False, ends=ends, marked=marked)


# --- Chats and export (C5), from /chats, /state and /export


NOT_EXPORTED = "Export this chat first; the file is made when you export it."


def chat_date(chat_id: str, today: date) -> tuple[str, str | None]:
    """A chat's group, "today" or "earlier", and when it started as its row shows it: the
    time for today's ("14:25", so a day of same-titled chats can be told apart), the date for
    an earlier one ("30 Oct", with the year when it isn't this one). Read from the start of
    its ID, the local time the chat started; an ID that doesn't start with one is earlier,
    undated."""
    try:
        moment = datetime.strptime(chat_id[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return "earlier", None
    started = moment.date()
    if started == today:
        return "today", moment.strftime("%H:%M")
    words = f"{started.day} {started.strftime('%b')}"
    return "earlier", words if started.year == today.year else f"{words} {started.year}"


def differ_line(differ: dict[str, Any]) -> str:
    """ "2 things differ", from /resume's count (teaching check, item 23)."""
    n = differ["count"]
    return f"{n} thing differs" if n == 1 else f"{n} things differ"


async def _chats(worker: Worker, params: dict[str, str | None]) -> str:
    """The Chats drawer: every chat, newest first, under Today and Earlier, the current one
    marked. While a command changes the chat, picking one waits."""
    state = await result_of(worker, "state")
    chats = (await result_of(worker, "chats"))["chats"]
    today = datetime.now().date()
    groups: dict[str, list[dict[str, Any]]] = {"today": [], "earlier": []}
    for chat in chats:
        group, when = chat_date(chat["log_id"], today)
        groups[group].append({**chat, "date": when, "current": chat["log_id"] == state["log_id"]})
    return templates.get_template("chats.html").render(
        today=groups["today"],
        earlier=groups["earlier"],
        in_chat=state["conversation_id"] is not None,
        waits=waits_words(state),
    )


# --- Attach (v0.3.1 B3): the sheet from `/attach ID [range] check`, and the waiting chips


def attach_params(query: Any) -> dict[str, str | None]:
    """/view/attach's parameters: an upload ID, and a range (unit, first, last) or none."""
    upload = query.get("upload", "")
    if not UPLOAD_ID.fullmatch(upload):
        raise Refused(400, "That isn't an upload ID.")
    unit, first, last = (query.get(k) for k in ("unit", "first", "last"))
    if (unit, first, last) == (None, None, None):
        return {"upload": upload, "range": None, "first": None, "last": None}
    if unit not in ("pages", "lines") or not all(
        n is not None and re.fullmatch(r"[0-9]{1,9}", n) for n in (first, last)
    ):
        raise Refused(400, "A range is pages or lines, then two whole numbers.")
    return {"upload": upload, "range": f"{unit} {first}-{last}", "first": first, "last": last}


async def _attach(worker: Worker, params: dict[str, str | None]) -> str:
    """The attach sheet for an upload (C1), from `/attach ID [range] check`: the choices and
    their numbers in the table's words, or the refusal. With no range, a file that doesn't
    fit opens on the part the sheet preselects (Q11); a range the file doesn't have shows
    its sentence in the part, with the rest drawn."""
    upload, chosen = params["upload"], params["range"]
    assert upload is not None  # attach_params checked it
    answer = await read_or_refusal(
        worker, "attach", " ".join(filter(None, (upload, chosen))) + " check"
    )
    part_error = None
    if "error" in answer and chosen:  # the range, not the file: draw the rest
        whole = await read_or_refusal(worker, "attach", f"{upload} check")
        if "result" in whole:
            part_error, answer = answer["error"], whole
    if "error" in answer:
        return templates.get_template("attach.html").render(error=answer["error"])
    check = answer["result"]["check"]
    pre = check["preselected"]
    if not chosen and not check["fits"] and pre:
        units = pre["unit"]
        # The cache makes this second read count only the range (Checkpoint 3, could 16).
        found = await read_or_refusal(
            worker, "attach", f"{upload} {units} 1-{pre['range'][1]} check"
        )
        if "error" in found:  # a cancel or a refusal here is shown, not dropped (should 7)
            return templates.get_template("attach.html").render(error=found["error"])
        check = found["result"]["check"]
    state = await result_of(worker, "state")
    typed = (params["first"], params["last"]) if part_error else None
    sheet = attach_sheet(check, upload, state["pending"], part_error, typed)
    return templates.get_template("attach.html").render(error=None, s=sheet)


KINDS = {"text": "Text", "html": "HTML", "pdf": "PDF"}


def size_words(n: int) -> str:
    """A file's size, as the sheet shows it."""
    if n < 1024:
        return counted(n, "byte")
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


PREVIEW_LINE = 2000  # characters of each line the preview shows


def preview(text: str) -> dict[str, Any]:
    """The extracted text the sheet shows: its first six lines, then, past eight, how many
    are left out and the last two (the board's ruling 8). Each line is cut at PREVIEW_LINE
    characters, saying how many more it has, so the whole is at most eight cut lines
    (Checkpoint 3, should 6)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    def shown(part: list[str]) -> list[dict[str, str | None]]:
        return [
            {"text": line[:PREVIEW_LINE], "cut": None}
            if len(line) <= PREVIEW_LINE
            else {
                "text": line[:PREVIEW_LINE],
                "cut": SENTENCES["more_characters"].format(count=f"{len(line) - PREVIEW_LINE:,}"),
            }
            for line in part
        ]

    if len(lines) <= 8:
        return {"start": shown(lines), "more": None, "end": []}
    more = SENTENCES["more_lines"].format(count=f"{len(lines) - 8:,}")
    return {"start": shown(lines[:6]), "more": more, "end": shown(lines[-2:])}


def attach_sheet(
    check: dict[str, Any],
    upload: str,
    pending: list[dict[str, Any]],
    part_error: str | None,
    typed: tuple[str | None, str | None] | None,
) -> dict[str, Any]:
    """Everything the sheet draws, in the table's words (decision 20)."""
    max_tokens = check["max_tokens"]  # what its numbers were counted with (could 15)
    unit, units = check["unit"], f"{check['unit']}s"
    free = f"{check['free']:,}"
    facts = KINDS[check["kind"]] + " · " + counted(check["total"], unit)
    if check["empty_units"]:
        facts += ", " + SENTENCES["empty_units"].format(count=f"{check['empty_units']:,}")
    if check["encrypted"]:
        facts += ", " + SENTENCES["encrypted"]
    facts += " · " + size_words(check["bytes"])
    # A searching chat's numbers name the parts search adds (the board walk's S14).
    left = SENTENCES["left_search" if check["aside"] else "left"]
    if check["fits"]:
        whole_note = (
            SENTENCES["whole"].format(tokens=f"{check['tokens_whole']:,}")
            + " "
            + left.format(left=f"{check['left_whole']:,}")
        )
    else:
        whole_note = SENTENCES["doesnt_fit"].format(
            tokens=more_than(check["tokens_whole"], check["tokens_whole_at_least"]), free=free
        )
    start, pre, chosen = check["start_that_fits"], check["preselected"], check["chosen"]
    # With nothing counted, the boxes open on 1 to the preselected end, else 1 to 1, as the
    # board does; choosing the part counts them (Checkpoint 3, should 4).
    opening = pre["range"][1] if pre else 1
    part: dict[str, Any] = {"first": 1, "last": opening, "ok": False, "notes": []}
    part["counted"] = bool(typed or chosen)
    if typed:
        part.update(first=typed[0], last=typed[1], notes=[part_error])
    elif chosen:
        first, last = chosen["range"]
        part.update(first=first, last=last, ok=chosen["left"] is not None and chosen["left"] >= 0)
        if part["ok"]:
            part["count"] = f"{chosen['tokens']:,} tokens"
            part["notes"].append(
                left.format(left=f"{chosen['left']:,}")
                + " "
                + SENTENCES["reply_share"].format(max_tokens=f"{max_tokens:,}")
            )
        else:
            needed = more_than(chosen["tokens"], chosen["tokens_at_least"])
            part["notes"].append(SENTENCES["doesnt_fit"].format(tokens=needed, free=free))
    # The suggested range, what it buys, and the most that fits with what that leaves, so
    # the two don't read as a contradiction (the design walkthrough's F1).
    suggested = bool(pre and chosen and chosen["range"] == pre["range"])
    if start:
        part["max"] = start["range"][1]
        if not check["fits"]:
            at_most = SENTENCES["at_most"].format(
                units=units, last=f"{start['range'][1]:,}", left=f"{start['left']:,}"
            )
            if suggested:
                assert pre is not None
                filled = SENTENCES["filled_in"].format(
                    units=units, last=f"{pre['range'][1]:,}", max_tokens=f"{max_tokens:,}"
                )
                if check["aside"]:
                    filled = SENTENCES["filled_in_search"].format(
                        units=units,
                        last=f"{pre['range'][1]:,}",
                        max_tokens=f"{max_tokens:,}",
                        budget=f"{check['aside']:,}",
                        origin=SENTENCES[f"budget_{check['aside_origin']}"],
                    )
                part["notes"] += [
                    f"{filled} {at_most}",
                    SENTENCES["longer_chat"].format(units=units),
                ]
            else:
                part["notes"].append(at_most)
    else:
        part["max"] = check["total"]
    if not check["fits"] and not pre:
        none = SENTENCES["none_fits"].format(unit=unit)
        no_start = SENTENCES["no_start"].format(unit=unit, max_tokens=f"{max_tokens:,}")
        part["notes"].append(none if start is None else no_start)
    part["args"] = f"{upload} {units} {part['first']}-{part['last']}"
    marker = chosen["marker"] if chosen else None
    choice = "part" if chosen or typed else "whole" if check["fits"] else None
    new = not check["fits"] and check["fits_new_chat"]
    names = ", ".join(a["name"] for a in pending)
    origin = SENTENCES[
        "origin_profile" if check["framing_origin"] == "profile" else "origin_default"
    ]
    own_marker = check["excerpt_marker_origin"]
    marker_origin = SENTENCES["origin_default" if own_marker == "default" else "origin_profile"]
    return {
        "name": check["name"],
        "upload": upload,
        "facts": facts,
        "free": SENTENCES["free"].format(free=free, max_tokens=f"{max_tokens:,}"),
        # A PDF's boxes take file pages, not printed ones (F2).
        "pdf_pages": SENTENCES["pdf_pages"] if check["kind"] == "pdf" else None,
        "already": SENTENCES["already"] if check["already_in_chat"] else None,
        "choice": choice,
        "whole": {"fits": check["fits"], "note": whole_note},
        "part": part,
        "units": units.capitalize(),
        "unit_plural": units,
        "total": f"{check['total']:,}",
        "excerpt": SENTENCES["no_marker"]
        if own_marker == "none"
        else SENTENCES["excerpt"].format(units=units),
        "new": {
            "does": SENTENCES["new_chat_does"],
            "whole": SENTENCES["whole"].format(tokens=f"{check['tokens_new_chat']:,}"),
            "left": SENTENCES["left"].format(left=f"{check['left_new_chat']:,}"),
            "drops": SENTENCES["not_kept"].format(names=names) if names else None,
        }
        if new
        else None,
        "warnings": [
            *(
                [SENTENCES["control_tokens"].format(tokens=", ".join(check["control_tokens"]))]
                if check["control_tokens"]
                else []
            ),
            *([SENTENCES["closing_tag"]] if check["has_closing_tag"] else []),
        ],
        "framing_label": SENTENCES["framing_label"].format(origin=origin),
        "framing": check["framing"],
        "no_framing": SENTENCES["no_framing"].format(origin=origin),
        "marker_label": SENTENCES["marker_label"].format(origin=marker_origin),
        "marker": marker,
        "preview": preview(chosen["text"] if chosen else check["text"]),
        # Open for a PDF or a web page, the one check that the text came out right (F7).
        "preview_open": check["kind"] in ("pdf", "html"),
        "choose": SENTENCES["choose"],
        "described_by": {"whole": "attach-whole-note", "part": "attach-part-note"}.get(
            choice or "", "attach-why"
        ),
    }


async def _pending(worker: Worker, params: dict[str, str | None]) -> str:
    """The attachments waiting for your next message, as chips."""
    state = await result_of(worker, "state")
    for a in state["pending"]:  # the cost when attached, as the thread's line says (F3)
        a["cost"] = SENTENCES["chip_tokens"].format(tokens=f"{a['tokens']:,}")
        a["shown"] = how_words(a)  # as your sent message's line says it (surface could 6)
    return templates.get_template("pending.html").render(pending=state["pending"])


def export_sheet(answer: dict[str, Any]) -> dict[str, Any]:
    """The export sheet: the file and what it holds, from /export's result, or the
    refusal's sentence."""
    if "error" in answer:
        return {"error": answer["error"]}
    result = answer["result"]
    n = result["failed_calls_left_out"] + result["calls_without_a_turn_left_out"]
    return {
        "error": None,
        "path": result["path"],
        "replaced": result["replaced"],
        "attachments": result["attachments"],
        "left_out": f"{n} call that produced no reply"
        if n == 1
        else f"{n} calls that produced no reply"
        if n
        else None,
        # From the log, which another program can write: control or invisible characters
        # show as escapes, with no Copy (the copy wouldn't name the file).
        "steered": [
            {**v, "file": shown(v["file"]), "can_copy": v["file"].isprintable()}
            for v in result["steered"]
        ],
    }


def shown(text: str) -> str:
    """`text`, or, if it holds control or invisible characters, `text` with those written
    as Python writes them (\\x1b, \\u202e)."""
    return text if text.isprintable() else repr(text)[1:-1]


async def _export(worker: Worker, params: dict[str, str | None]) -> str:
    """The sheet after Export: this start's last /export for the chat that's open."""
    answer = worker.export_now()
    if not answer:
        raise Refused(409, NOT_EXPORTED)
    return templates.get_template("export.html").render(**export_sheet(answer))


def event_line(event: dict[str, Any]) -> bytes:
    """One line of the event stream. A value JSON can't carry (NaN or infinity) would break
    the page's parser and drop the stream, so it's sent as an error event instead."""
    try:
        return json.dumps(event, allow_nan=False).encode() + b"\n"
    except ValueError:
        why = "An event held a number JSON can't carry (NaN or infinity); it isn't shown."
        return json.dumps({"event": "error", "type": "NotJSON", "message": why}).encode() + b"\n"


def json_response(content: dict[str, Any]) -> Response:
    """JSON as the page can read it. A value JSON can't carry (NaN or infinity, from a log
    another program wrote) is refused with a sentence, never sent as invalid JSON."""
    try:
        body = json.dumps(content, allow_nan=False)
    except ValueError:
        return PlainTextResponse(
            "The result holds a number JSON can't carry (NaN or infinity), from a log "
            "Clarity didn't write.",
            500,
        )
    return Response(body, media_type="application/json")


VIEW_RENDERERS = {
    "thread": _thread,
    "picker": _picker,
    "about": _about,
    "replies": _replies,
    "compare": _compare,
    "settings": _settings,
    "details": _details,
    "branches": _branches,
    "chats": _chats,
    "export": _export,
    "attach": _attach,
    "pending": _pending,
}
