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
import os
import queue
import secrets
import sys
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.applications import Starlette
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

from clarity_chat.backends import BackendError
from clarity_chat.commands import CALL_ID, Commands
from clarity_chat.json_chat import JsonChat
from clarity_chat.profiles import ConfigError
from clarity_chat.session import ContextError, Session

PORT = 8765  # fixed, a constant: server.toml describes llama-server only
STATIC = Path(__file__).parent / "web_static"
STATIC_FILES = {  # nothing else is served
    "app.js": "text/javascript",
    "app.css": "text/css",
    "clarity-ds.css": "text/css",
}
VIEWS = {
    "thread",
    "details",
    "compare",
    "probs",
    "branches",
    "settings",
    "chats",
    "export",
    "picker",
    "replies",
    "about",
}
MAX_BODY = 64 * 1024
HEARTBEAT = 15.0  # seconds between keep-alive lines on the event stream
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; "
    "frame-ancestors 'none'"
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


class Worker:
    """Owns the Session's command table on one thread. Each command waits its turn; the
    commands the table lets run during another (state, cancel, chats, profiles; settings
    during a reply) run at once on the caller's thread."""

    def __init__(self, session: Session) -> None:
        self.commands = Commands(session)
        self.jobs: queue.Queue[tuple[str, str, concurrent.futures.Future]] = queue.Queue()
        self.listeners: list[Callable[[dict[str, Any]], None]] = []  # one per open stream
        self.lock = threading.Lock()
        self.failed: str | None = None  # an unexpected error ended the session
        self.export_path: Path | None = None  # the export made in this start, if any
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

    def submit(self, name: str, args: str) -> dict[str, Any]:
        """Run a command and wait for its result (a caller on its own thread)."""
        now, future = self.enqueue(name, args)
        return self._run(name, args, at_once=True) if now else future.result()

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

    def _run(self, name: str, args: str, at_once: bool | None = None) -> dict[str, Any]:
        if self.failed:
            return {"error": self.failed, "type": "Stopped", "stopped": True}
        try:
            result = self.commands.run(name, args, self.emit, at_once=at_once)
        except (ConfigError, ContextError, BackendError) as e:
            return {"error": str(e), "type": type(e).__name__}
        except Exception as e:  # noqa: BLE001 - as in --json, it ends the session
            self.failed = f"Clarity stopped after an unexpected error: {type(e).__name__}: {e}"
            self.emit({"event": "stopped", "reason": self.failed})
            return {"error": self.failed, "type": "Stopped", "stopped": True}
        if name == "export":
            self.export_path = Path(result["path"])
        return {"result": result}


class Security:
    """Every request: the exact Host, no OPTIONS, and the security headers on every
    response, refusals and server errors included. It wraps the whole app, outside
    Starlette's error handling, so a 500 gets them too."""

    def __init__(self, app: Starlette, port: int) -> None:
        self.app, self.host = app, f"127.0.0.1:{port}".encode()
        self.routes, self.state = app.routes, app.state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
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


def make_app(
    session: Session,
    secret: bytes,
    port: int = PORT,
    on_first_use: Callable[[], None] | None = None,
) -> Security:
    """The app. `on_first_use` runs once, at the first request with the secret."""
    worker = Worker(session)
    first_use = [on_first_use]
    origin = f"http://127.0.0.1:{port}"
    tabs: dict[str, Any] = {"live": None, "streams": {}}  # the live tab; open streams per tab

    def check(request: Request, post: bool = False) -> None:
        """The secret, fetch metadata and, for a POST, the Origin and the content type."""
        given = request.headers.get("authorization", "")
        token = given[7:].encode("utf-8", "replace") if given.startswith("Bearer ") else b""
        if not secrets.compare_digest(token, secret):
            raise Refused(401, "Open Clarity from the link in the terminal.")
        if first_use[0] is not None:
            done, first_use[0] = first_use[0], None
            done()
        if request.headers.get("sec-fetch-site", "same-origin") != "same-origin":
            raise Refused(403, "Only Clarity's own page can ask for this.")
        if post:
            if request.headers.get("origin") != origin:
                raise Refused(403, "Only Clarity's own page can ask for this.")
            if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                raise Refused(415, "Send JSON.")

    def refusing(endpoint: Any) -> Any:
        def wrapped(request: Request) -> Response:
            try:
                return endpoint(request)
            except Refused as e:
                return PlainTextResponse(str(e), e.status)

        return wrapped

    def live(tab: str) -> None:
        if tabs["live"] is None:
            tabs["live"] = tab
        if tab != tabs["live"]:
            raise Refused(409, "Clarity is open in another tab.")

    async def body(request: Request, is_command: bool = True) -> dict[str, Any]:
        """The request's JSON object, by --json's rules for a command."""
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise Refused(413, "Too big.")
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

    def shell(request: Request) -> Response:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    def static(request: Request) -> Response:
        name = request.path_params["file"]
        if name not in STATIC_FILES:
            return PlainTextResponse("Not found.", 404)
        return FileResponse(STATIC / name, media_type=STATIC_FILES[name])

    async def command(request: Request) -> Response:
        try:
            check(request, post=True)
            parsed = await body(request)
            tab = parsed.get("tab")
            if not isinstance(tab, str):
                raise Refused(400, "A command names its tab.")
            live(tab)
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        name, args = parsed["command"], parsed.get("args", "")
        now, future = worker.enqueue(name, args)
        if now:
            answer = await run_in_threadpool(worker._run, name, args, True)
        else:  # waits in the event loop, not in a thread: queued commands can't use them up
            answer = await asyncio.wrap_future(future)
        return json_response({"id": parsed.get("id"), **answer})

    async def take_tab(request: Request) -> Response:
        try:
            check(request, post=True)
            tab = (await body(request, is_command=False)).get("tab")
            if not isinstance(tab, str):
                raise Refused(400, "Name the tab.")
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        tabs["live"] = tab
        worker.emit({"event": "not_live", "live": tab})
        return JSONResponse({"live": tab})

    async def events(request: Request) -> Response:
        try:
            check(request)
        except Refused as e:
            return PlainTextResponse(str(e), e.status)
        tab = request.query_params.get("tab", "")
        if not tabs["streams"].get(tabs["live"]):  # live at once if no live tab is listening
            tabs["live"] = tab
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def push(event: dict[str, Any]) -> None:  # from the worker's thread
            loop.call_soon_threadsafe(inbox.put_nowait, event)

        async def lines() -> AsyncIterator[bytes]:
            yield json.dumps({"event": "hello"}).encode() + b"\n"
            if tab != tabs["live"]:
                yield json.dumps({"event": "not_live", "live": tabs["live"]}).encode() + b"\n"
                return
            worker.listen(push)
            tabs["streams"][tab] = tabs["streams"].get(tab, 0) + 1
            try:  # a client that leaves cancels this at its next await
                while True:
                    try:
                        event = await asyncio.wait_for(inbox.get(), HEARTBEAT)
                    except TimeoutError:
                        yield b'{"event": "heartbeat"}\n'
                        continue
                    yield json.dumps(event).encode() + b"\n"
                    if event["event"] == "not_live" and event.get("live") != tab:
                        return
            finally:
                worker.stop_listening(push)
                tabs["streams"][tab] -= 1

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @refusing
    def view(request: Request) -> Response:
        check(request)
        name = request.path_params["name"]
        if name not in VIEWS:
            raise Refused(404, "No such view.")
        call = request.query_params.get("call")
        if call is not None and not CALL_ID.fullmatch(call):
            raise Refused(400, "That isn't a call ID.")
        renderer = VIEW_RENDERERS.get(name)
        if renderer is None:
            raise Refused(404, "This view isn't built yet.")
        return HTMLResponse(renderer(worker, request.query_params))

    @refusing
    def export(request: Request) -> Response:
        check(request)
        path = worker.export_path
        if path is None or not path.is_file():
            raise Refused(409, "Export this chat first; the file is made when you export it.")
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
        ]
    )
    app.state.worker = worker
    return Security(app, port)


def serve(
    root: Path,
    port: int = PORT,
    session_factory: Callable[[], Session] | None = None,
    opener: Callable[[Path], Any] | None = None,
    on_ready: Callable[[Any], None] | None = None,
    write: Callable[[str], Any] = print,
    read_lines: bool = True,
) -> int:
    """`clarity ui`: serve the page until Ctrl-C, Quit or a closed terminal. The link, with
    a secret made for this start in its fragment, is printed and opened through a private
    redirect file (a link in `open`'s arguments would show in other users' process lists).
    Enter prints and opens it again."""
    import socket

    import uvicorn

    with socket.socket() as probe:  # refused here, before anything starts
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as e:
            raise ConfigError(f"Can't start: port {port} is in use ({e.strerror}).") from e
    secret = secrets.token_urlsafe(32)
    link = f"http://127.0.0.1:{port}/#key={secret}"
    redirect = root / "local" / "clarity-open.html"
    redirect.parent.mkdir(exist_ok=True)
    redirect.unlink(missing_ok=True)  # one an earlier start left; this start makes its own
    fd = os.open(
        redirect, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(REDIRECT.format(link=html.escape(link, quote=True)))
    session = (session_factory or (lambda: Session(root)))()
    app = make_app(
        session, secret.encode(), port, on_first_use=lambda: redirect.unlink(missing_ok=True)
    )
    open_page = opener or (lambda path: webbrowser.open(path.as_uri()))
    write(
        f"Clarity is at {link}\n(the link works until Clarity stops). Enter opens it again; "
        "Ctrl-C stops Clarity.\n"
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
    server = uvicorn.Server(config)
    if on_ready:
        on_ready(server)
    if read_lines:

        def again() -> None:
            for _ in sys.stdin:
                write(f"Clarity is at {link}\n")
                webbrowser.open(link)

        threading.Thread(target=again, daemon=True).start()
    try:
        server.run()
    finally:
        worker: Worker = app.state.worker
        worker.commands.run("cancel", "", lambda event: None)  # a reply in progress stops
        with worker.commands.busy:  # and its record is written before the session closes
            pass
        session.close()
        redirect.unlink(missing_ok=True)
    return 0


REDIRECT = """<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0;url={link}">
<title>Opening Clarity</title>
<p><a href="{link}">Open Clarity</a></p>
"""


def _thread(worker: Worker, params: Any) -> str:
    """The thread, from /state's turns: a read that runs at once, mid-reply included."""
    answer = worker._run("state", "", at_once=True)
    if "error" in answer:
        raise Refused(503, answer["error"])
    return templates.get_template("thread.html").render(turns=answer["result"]["turns"])


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


VIEW_RENDERERS = {"thread": _thread}
