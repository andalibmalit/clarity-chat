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
import socket
import sys
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable
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

from clarity_chat.backends import BackendError
from clarity_chat.commands import CALL_ID, REPLY_COMMANDS, Commands
from clarity_chat.json_chat import JsonChat
from clarity_chat.profiles import ConfigError
from clarity_chat.session import CONTEXT_WARN_FRACTION, ContextError, Session

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

    async def ask(self, name: str, args: str) -> dict[str, Any]:
        """A command's answer, for an async route: at once on a thread if the table lets it
        run beside whatever runs, else from the queue, waited for in the event loop (so
        queued commands can't use up the threads `cancel` needs)."""
        now, future = self.enqueue(name, args)
        if now:
            return await run_in_threadpool(self._run, name, args, True)
        return await asyncio.wrap_future(future)

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
    chunks, size = [], 0
    async for chunk in request.stream():  # counted as it arrives, chunked bodies too
        size += len(chunk)
        if size > MAX_BODY:
            raise Refused(413, "Too big.")
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
    """The app. `on_first_use` runs once, at the first request with the secret;
    `app.state.rearm(callback)` sets it again (the terminal's Enter makes a new redirect)."""
    if len(secret) < 16:  # an empty one would let every request through
        raise ValueError("The secret must be at least 16 bytes.")
    worker = Worker(session)
    first_use = [on_first_use]
    origin = f"http://127.0.0.1:{port}"
    # The live tab; open streams per tab; tabs told they aren't live, which take over only
    # by POST /tab, never by reconnecting their stream.
    tabs: dict[str, Any] = {"live": None, "streams": {}, "told": set()}

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
        # Live at once if no live tab is listening, unless this tab was told it isn't live.
        if not tabs["streams"].get(tabs["live"]) and tab not in tabs["told"]:
            tabs["live"] = tab
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def push(event: dict[str, Any]) -> None:  # from the worker's thread
            loop.call_soon_threadsafe(inbox.put_nowait, event)

        live_here = tab == tabs["live"]
        if live_here:  # listening from now, before hello: nothing sent after it is lost
            worker.listen(push)
            tabs["streams"][tab] = tabs["streams"].get(tab, 0) + 1

        async def lines() -> AsyncIterator[bytes]:
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
            name = request.path_params["name"]
            if name not in VIEWS:
                raise Refused(404, "No such view.")
            call = request.query_params.get("call")
            if call is not None and not CALL_ID.fullmatch(call):
                raise Refused(400, "That isn't a call ID.")
            renderer = VIEW_RENDERERS.get(name)
            if renderer is None:
                raise Refused(404, "This view isn't built yet.")
            return HTMLResponse(await renderer(worker, {"call": call}))
        except Refused as e:
            return PlainTextResponse(str(e), e.status)

    @refusing
    def export(request: Request) -> Response:
        check(request)
        path, conversation = worker.export_path, worker.commands.session.conversation
        current = (
            conversation is not None
            and path is not None
            and (path.stem == conversation.conversation_id)
        )
        if not current or path is None or not path.is_file():
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
    app.state.rearm = lambda callback: first_use.__setitem__(0, callback)
    app.state.quit = lambda: None  # after a quit's answer is sent; serve() stops the server
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
        worker: Worker = app.state.worker
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


async def result_of(worker: Worker, name: str, args: str = "") -> dict[str, Any]:
    """A command's result for a view. An error here is the session's (a view runs only reads
    that can't be refused), so the view answers 503 with its sentence."""
    answer = await worker.ask(name, args)
    if "error" in answer:
        raise Refused(503, answer["error"])
    return answer["result"]


def starting_profile(state: dict[str, Any], profiles: list[dict[str, Any]]) -> dict | None:
    """The profile a model start is for, as /profiles lists it, or None when nothing starts.
    /state's chat is still the one before the start, so the name comes from the command."""
    running = state["running"]
    if state["model"] != "starting" or running is None:
        return None
    name = {"profile": running["args"], "new": state["profile"]}.get(running["command"])
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
    scale = settings["settings"]["steering_scale"]["value"]
    return "not steered" if scale == 0 else f"steered ×{scale:g}"


async def context_figure(worker: Worker, state: dict[str, Any]) -> dict[str, Any] | None:
    """The last prompt's share of the context. A message refused for its size isn't counted
    (nothing was sent); the reply at the head is."""
    call = (await result_of(worker, "prompt"))["call"]
    if call and (call.get("error") or "").startswith("ContextError"):
        replies = [t["call_id"] for t in state["turns"] if t["role"] == "assistant"]
        call = (await result_of(worker, "prompt", replies[-1]))["call"] if replies else None
    if not call or "n_prompt_client" not in call:
        return None
    n, ctx = call["n_prompt_client"], call["n_ctx"]
    return {
        "text": f"{n / ctx:.0%} · {n:,} of {ctx:,} tokens",
        "hot": n > CONTEXT_WARN_FRACTION * ctx,
    }


async def _thread(worker: Worker, params: dict[str, str | None]) -> str:
    """The thread, from /state: the turns, or the first screen (a pointer to earlier chats, a
    model starting, a new chat's profile) and the start warnings. Every read here runs at
    once, mid-reply included, except /settings during a start, which it doesn't ask then."""
    state = await result_of(worker, "state")
    starting = has_chats = intro = None
    if state["model"] == "starting":
        starting = starting_profile(state, (await result_of(worker, "profiles"))["profiles"])
    elif state["conversation_id"] is None:
        has_chats = bool((await result_of(worker, "chats"))["chats"])
    elif not any(turn["role"] == "user" for turn in state["turns"]):
        settings = await result_of(worker, "settings")
        intro = {"backend": settings["backend"], "steering": steering_words(settings)}
    return templates.get_template("thread.html").render(
        state=state, turns=state["turns"], starting=starting, has_chats=has_chats, intro=intro
    )


async def _picker(worker: Worker, params: dict[str, str | None]) -> str:
    """The row under the message box: the picker button and its list of profiles, Settings
    with how many settings this chat changed, and the context figure."""
    state = await result_of(worker, "state")
    profiles = (await result_of(worker, "profiles"))["profiles"]
    starting = starting_profile(state, profiles)
    in_chat = state["conversation_id"] is not None and starting is None
    running = (state["running"] or {}).get("command")
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
        if running is None:  # /prompt waits its turn: not asked while a command runs
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


VIEW_RENDERERS = {"thread": _thread, "picker": _picker, "about": _about}
