"""The web server's routes and security (v0.3.0 Group B: B0, B2, B3), offline.

The route table and every check are from notes/plans/v0.3.0.md. TestClient drives the app in
process; nothing listens on a port.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from html.parser import HTMLParser

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.commands import CALL_ID
from clarity_chat.session import Session
from tests.fakes import FakeBackend

PORT = web.PORT
ORIGIN = f"http://127.0.0.1:{PORT}"
SECRET = "s3cret-for-tests"
AUTH = {"Authorization": f"Bearer {SECRET}"}
SAME = {"Sec-Fetch-Site": "same-origin"}
POST = {**AUTH, **SAME, "Origin": ORIGIN, "Content-Type": "application/json"}
GET = {**AUTH, **SAME}
# The CSP, written out: a directive added, dropped or loosened in web.CSP fails here.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; "
    "frame-ancestors 'none'; require-trusted-types-for 'script'; trusted-types clarity"
)

# The plan's route table: method, path, needs the secret.
ROUTES = {
    ("GET", "/"): False,
    ("GET", "/static/{file}"): False,
    ("POST", "/command"): True,
    ("GET", "/events"): True,
    ("GET", "/view/{name}"): True,
    ("POST", "/tab"): True,
    ("GET", "/export"): True,
    ("POST", "/upload"): True,  # v0.3.1 B3
}


@pytest.fixture
def server(fetched_root, tiny_tokenizer):
    backend = FakeBackend([" Hello!", " Again.", " More."], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        yield client, session, app


def command(client, name, args="", tab="tab-1"):
    body = {"command": name, "args": args, "id": 1, "tab": tab}
    return client.post("/command", headers=POST, content=json.dumps(body))


def wait_until(ready, seconds: float = 10.0) -> None:
    """Poll `ready` until it's true, up to `seconds`. An ordering a test depends on is
    waited for, never a fixed pause: a slow runner (CI run 43658818773) needs longer."""
    deadline = threading.Event()
    timer = threading.Timer(seconds, deadline.set)
    timer.start()
    try:
        while not ready():
            assert not deadline.is_set(), f"not ready after {seconds} s"
            deadline.wait(0.02)
    finally:
        timer.cancel()


def writing(client) -> bool:
    """A reply is being written: /state names it, and its first piece has arrived."""
    state = command(client, "state").json()["result"]
    return bool(state["running"]) and bool(state["reply_so_far"])


def test_the_routes_are_exactly_the_plans_table(server):
    _, _, app = server
    found = {(m, r.path) for r in app.routes for m in getattr(r, "methods", []) if m != "HEAD"}
    assert found == set(ROUTES)


def test_the_shell_and_static_files_need_no_secret_and_hold_no_data(server):
    client, _, _ = server
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert SECRET not in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/../pyproject.toml").status_code == 404
    assert client.get("/static/nope.js").status_code == 404


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/command"),
        ("GET", "/events"),
        ("GET", "/view/thread"),
        ("POST", "/tab"),
        ("GET", "/export"),
        ("POST", "/upload"),
    ],
)
@pytest.mark.parametrize(
    "auth", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer é".encode()}]
)
def test_every_data_route_needs_the_secret(server, method, path, auth):
    client, _, _ = server
    headers = {**POST, **auth} if method == "POST" else {**SAME, **auth}
    if not auth:
        headers.pop("Authorization", None)
    # Streamed and closed once the status is in: /events let through would never end.
    with client.stream(method, path, headers=headers, content=b"{}") as response:
        assert response.status_code == 401


@pytest.mark.parametrize("host", [f"localhost:{PORT}", "127.0.0.1:1", "evil.example", "127.0.0.1"])
def test_a_wrong_host_or_port_is_refused_everywhere(server, host):
    client, _, _ = server
    for path in ("/", "/static/app.js", "/view/thread"):
        assert client.get(path, headers={**AUTH, **SAME, "Host": host}).status_code == 403


@pytest.mark.parametrize("site", ["cross-site", "same-site", "none"])  # same-site: SHOULD 5
@pytest.mark.parametrize("path", ["/events", "/view/thread", "/export"])
def test_a_cross_site_read_is_refused(server, path, site):
    client, _, _ = server
    # Streamed, and closed once the status is in: /events let through would never end, so a
    # refusal that stopped working must fail here, not hang the suite.
    with client.stream("GET", path, headers={**AUTH, "Sec-Fetch-Site": site}) as response:
        assert response.status_code == 403


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"Origin": "http://evil.example"}, 403),
        ({"Origin": f"http://127.0.0.1:{PORT + 1}"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": None}, 403),
        ({"Content-Type": "text/plain"}, 415),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
    ],
)
@pytest.mark.parametrize("path", ["/command", "/tab"])  # /tab: SHOULD 5
def test_a_post_needs_the_exact_origin_and_json(server, change, status, path):
    client, _, _ = server
    headers = {**POST, **{k: v for k, v in change.items() if v is not None}}
    for k, v in change.items():
        if v is None:
            headers.pop(k)
    body = json.dumps({"command": "state", "args": "", "id": 1, "tab": "tab-1"})
    assert client.post(path, headers=headers, content=body).status_code == status
    assert client.post(path, headers=POST, content=body).status_code == 200


def test_options_is_refused_and_no_cors_header_is_ever_sent(server):
    client, _, _ = server
    response = client.options("/command", headers={"Origin": "http://evil.example"})
    assert response.status_code == 405
    for path in ("/", "/static/app.js"):
        assert not any(h.lower().startswith("access-control") for h in client.get(path).headers)


def test_every_response_has_the_security_headers(server):
    client, _, _ = server
    for response in (
        client.get("/"),
        client.get("/static/app.js"),
        client.get("/nope"),
        command(client, "state"),
    ):
        h = response.headers
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["Cache-Control"] == "no-store"
        assert h["Referrer-Policy"] == "no-referrer"
        assert h["Cross-Origin-Resource-Policy"] == "same-origin"
        assert h["Cross-Origin-Opener-Policy"] == "same-origin"
        assert h["Content-Security-Policy"] == CSP


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (b"not json", 400),
        (b"[1, 2]", 400),
        (b'{"command": 3}', 400),
        (b"x" * (8 * 1024 * 1024 + 1), 413),
    ],
)
def test_a_bad_request_is_refused_and_the_session_goes_on(server, body, status):
    client, _, _ = server
    assert client.post("/command", headers=POST, content=body).status_code == status
    assert command(client, "state").status_code == 200


def test_a_command_runs_and_its_result_comes_back_in_the_response(server):
    client, session, _ = server
    started = command(client, "profile", "text")
    assert started.status_code == 200 and started.json()["result"]["profile"] == "text"
    said = command(client, "say", "Hi").json()
    assert said["result"]["call"]["output_text"] == " Hello!"


def test_a_command_from_a_tab_that_isnt_live_is_refused(server):
    client, _, _ = server
    assert command(client, "state", tab="tab-1").status_code == 200  # the first tab is live
    assert command(client, "state", tab="tab-2").status_code == 409
    taken = client.post("/tab", headers=POST, content=json.dumps({"tab": "tab-2"}))
    assert taken.status_code == 200
    assert command(client, "state", tab="tab-1").status_code == 409


def test_no_command_can_name_a_path(server, tmp_path, fetched_root):
    """A valid profile outside the profile folders: refused by name, never read by path."""
    client, _, _ = server
    outside = tmp_path / "outside.toml"
    outside.write_text((fetched_root / "profiles" / "text.toml").read_text("utf-8"), "utf-8")
    answer = command(client, "profile", str(outside)).json()
    assert "not a path" in answer.get("error", "")


def test_a_view_name_or_call_id_outside_the_list_is_refused(server):
    client, _, _ = server
    for name in ("nope", "__class__", "get"):  # a renderer's name, and only that, is a view
        nope = client.get(f"/view/{name}", headers={**AUTH, **SAME})
        assert (nope.status_code, nope.text) == (404, "No such view.")
    bad = client.get("/view/details?call=call-zz", headers={**AUTH, **SAME})
    assert bad.status_code == 400


def test_the_events_stream_starts_with_hello(server):
    client, _, _ = server
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
        first = next(stream.iter_lines())
    assert json.loads(first) == {"event": "hello"}


def test_export_before_an_export_ran_is_refused_with_a_sentence(server):
    client, _, _ = server
    command(client, "profile", "text")
    response = client.get("/export", headers={**AUTH, **SAME})
    assert response.status_code == 409 and "export" in response.text.lower()


def test_templates_have_no_inline_style_or_script():
    """CSP allows only 'self': an inline style or script would be blocked, or a hole."""
    from pathlib import Path

    folder = Path(web.__file__).parent
    for path in [*(folder / "web_templates").glob("*.html"), folder / "web_static" / "index.html"]:
        text = path.read_text(encoding="utf-8").lower()
        assert "style=" not in text and "<style" not in text, path.name
        assert "<script>" not in text and "onclick=" not in text and "javascript:" not in text, (
            path.name
        )


def test_the_javascript_has_no_html_sink_but_swap():
    """B5: text goes in by textContent; server fragments only through swap()."""
    import re
    from pathlib import Path

    code = (Path(web.__file__).parent / "web_static" / "app.js").read_text(encoding="utf-8")
    sinks = re.findall(
        r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|createContextualFragment|eval\(",
        code,
    )
    assert sinks == []


def test_untrusted_text_in_the_thread_is_escaped(server):
    """B5's fixture, for the thread: a message holding markup and a template expression."""
    client, session, _ = server
    command(client, "profile", "text")
    hostile = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}'
    command(client, "say", hostile)
    html = client.get("/view/thread", headers={**AUTH, **SAME}).text
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "{{ 7*7 }}" in html  # shown as text, never run


# --- B1: clarity ui


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ui_refuses_a_port_in_use_before_starting_anything(fetched_root):
    import socket

    from clarity_chat.profiles import ConfigError

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(ConfigError, match=f"port {port} is in use"):
            web.serve(fetched_root, port=port, session_factory=lambda: pytest.fail("started"))


def test_ui_prints_its_link_at_once_into_a_pipe(fetched_root):
    """`clarity ui | tee`, or the QA driver: a pipe holds Python's output in a buffer until it
    fills or the program exits, so a link printed without a flush never arrives while
    Clarity runs. A real interpreter and a real pipe, with the fake backend."""
    import os
    import queue
    import subprocess
    import threading

    from tests.conftest import REPO

    script = (
        "import sys; from pathlib import Path; from clarity_chat import web; "
        "from clarity_chat.session import Session; from tests.fakes import FakeBackend; "
        "root = Path(sys.argv[1]); "
        "web.serve(root, port=int(sys.argv[2]), opener=lambda path: None, read_lines=False, "
        "session_factory=lambda: Session(root, backend=FakeBackend()))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(fetched_root), str(free_port())],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
    )
    lines: queue.Queue[str] = queue.Queue()
    stdout = child.stdout
    assert stdout is not None
    threading.Thread(target=lambda: lines.put(stdout.readline()), daemon=True).start()
    try:
        try:
            first = lines.get(timeout=30)
        except queue.Empty:  # what a pipe got before the fix: nothing while Clarity runs
            first = ""
    finally:
        child.terminate()
        try:
            child.wait(10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    assert first.startswith("Clarity is at http://127.0.0.1:"), first


def test_ui_writes_a_private_redirect_file_opens_it_and_deletes_it_on_first_use(
    fetched_root, tiny_tokenizer
):
    import os
    import stat
    import threading

    import httpx

    port = free_port()
    opened: list = []
    servers: list = []
    out: list[str] = []
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    closed: list[bool] = []
    session.close = lambda: closed.append(True)
    thread = threading.Thread(
        target=web.serve,
        kwargs={
            "root": fetched_root,
            "port": port,
            "session_factory": lambda: session,
            "opener": opened.append,
            "on_ready": servers.append,
            "write": out.append,
            "read_lines": False,
        },
    )
    thread.start()
    try:
        for _ in range(100):
            if servers and servers[0].started:
                break
            threading.Event().wait(0.05)
        (path,) = opened
        if sys.platform != "win32":  # Windows keeps no Unix mode bits (its profile ACL does)
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        link = next(line for line in "".join(out).split() if line.startswith("http://127.0.0.1"))
        assert link in path.read_text(encoding="utf-8") and "#key=" in link
        secret = link.split("#key=", 1)[1]
        headers = {"Authorization": f"Bearer {secret}", "Sec-Fetch-Site": "same-origin"}
        url = f"http://127.0.0.1:{port}/view/thread"
        # Plain HTTP to 127.0.0.1: no CA bundle (the sandbox blocks reading one), no proxy.
        response = httpx.get(url, headers=headers, verify=False, trust_env=False)
        assert response.status_code == 200
        assert path.exists()  # kept until the page it opened connects (security finding 5)
        events = f"http://127.0.0.1:{port}/events?tab=new-tab"
        with httpx.stream("GET", events, headers=headers, verify=False, trust_env=False) as r:
            next(r.iter_lines())
        assert not path.exists()  # deleted at a new tab's first /events
    finally:
        if servers:
            servers[0].should_exit = True
        thread.join(10)
    assert not thread.is_alive()
    assert closed == [True]  # the session (and its model) closed on the way out


def test_clarity_ui_serves_under_the_projects_lock(fetched_root, monkeypatch):
    """B1: another Clarity holding the lock refuses it; a killed one's server is cleaned up."""
    from clarity_chat import cli

    seen: list = []

    def fake_serve(root, session_factory):
        seen.append((root, session_factory))
        assert (root / "local" / "clarity.json").is_file()  # the lock is held while it serves
        return 0

    monkeypatch.setattr(web, "serve", fake_serve)
    assert cli.main(["--root", str(fetched_root), "ui"]) == 0
    assert seen and seen[0][0] == fetched_root.resolve()


def test_the_styles_load_nothing_from_the_network():
    """Fonts and images ship with the package or come from the system (CSP 'self' only)."""
    from pathlib import Path

    folder = Path(web.__file__).parent / "web_static"
    for path, media_type in web.STATIC_FILES.values():
        if media_type.startswith("font/"):
            continue  # binary; the fonts' own test checks them
        text = (folder / path).read_text(encoding="utf-8")
        assert "@import" not in text and "http://" not in text and "https://" not in text, path


# --- the tripwire review's findings (v0.3.0, 5759fcf..65bb7a3)


def test_a_command_decided_at_once_never_takes_the_queues_lock(fetched_root, tiny_tokenizer):
    """Finding 1: decided once. /settings, judged at once while a reply ran, mustn't take
    the lock the queue's next command needs if the reply ends in between."""
    from clarity_chat.commands import Commands

    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    c.busy.acquire()  # as if a reply runs, or has just ended with the queue's next one taking it
    try:
        assert c.run("settings", "", lambda e: None, at_once=True)["settings"]
    finally:
        c.busy.release()


def test_a_finished_reply_is_never_both_in_the_turns_and_still_being_written(
    fetched_root, tiny_tokenizer
):
    """Finding 2: when the reply's turn is written, the reply so far is cleared with it."""
    from clarity_chat.commands import Commands

    session = Session(
        fetched_root, backend=FakeBackend([" Hi."]), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    c = Commands(session)
    c.run("profile", "text", lambda e: None)
    seen: list[dict] = []

    def on_event(event):
        record = event.get("record") or {}
        if record.get("type") == "turn" and record.get("role") == "assistant":
            seen.append(c.run("state", "", lambda e: None))

    c.run("say", "Hello", on_event)
    (state,) = seen
    # In exactly one place: as the reply being written until the head moves (the whole
    # review's finding 7: cleared before that, it was in neither), then in the turns.
    in_turns = [t["text"] for t in state["turns"]][-1:] == [" Hi."]
    assert in_turns != (state["reply_so_far"] == " Hi."), state
    if in_turns:
        assert state["reply_so_far"] == "" and state["reply_end"] == 0


def test_a_server_error_carries_the_security_headers_and_says_why(server, monkeypatch):
    """Finding 3: an error inside a route still gets every header."""
    client, _, _ = server
    command(client, "profile", "text")
    nan = {"result": float("nan")}
    monkeypatch.setattr(web.Worker, "_run", lambda self, name, args, at_once=None: nan)
    response = command(client, "state")
    assert response.status_code == 500 and "NaN or infinity" in response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def submit(worker, name, args):
    """Run a command through the worker and wait for its result, as Worker.ask does for a
    route (a caller on its own thread)."""
    now, future = worker.enqueue(name, args)
    return worker._run(name, args, at_once=True) if now else future.result()


def test_a_listener_that_fails_is_dropped_and_the_worker_goes_on(fetched_root, tiny_tokenizer):
    """Finding 4: a closed event loop must not end the session or kill the worker."""
    session = Session(
        fetched_root, backend=FakeBackend([" Hi."]), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    worker = web.Worker(session)

    def broken(event):
        raise RuntimeError("Event loop is closed")

    worker.listen(broken)
    answers: list[dict] = []
    probe = threading.Thread(
        target=lambda: answers.append(submit(worker, "profile", "text")), daemon=True
    )
    probe.start()
    probe.join(10)
    assert answers and "result" in answers[0], "the worker died: a submit never came back"
    assert worker.failed is None and broken not in worker.listeners
    assert "result" in submit(worker, "say", "Hello")


# --- the tripwire review, part 2 (findings 7, 9, 10, 12) and the Enter handler


def test_a_tab_told_it_isnt_live_cant_take_over_by_reconnecting(server):
    """Finding 7: only POST /tab or the first load makes a tab live."""
    client, _, app = server
    assert command(client, "state", tab="tab-1").status_code == 200  # tab-1 is live
    taken = client.post("/tab", headers=POST, content=json.dumps({"tab": "tab-2"}))
    assert taken.status_code == 200  # tab-2 took over; tab-1 was told
    # tab-2's stream isn't open (its page is reloading): tab-1 reconnects meanwhile.
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
        lines = stream.iter_lines()
        assert json.loads(next(lines)) == {"event": "hello"}
        assert json.loads(next(lines))["event"] == "not_live"
    assert command(client, "state", tab="tab-2").status_code == 200


def test_an_empty_secret_is_refused_when_the_app_is_made(fetched_root, tiny_tokenizer):
    """Finding 9: b"" would leave every data route open."""
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    with pytest.raises(ValueError, match="secret"):
        web.make_app(session, b"", port=PORT)


def test_a_body_over_the_limit_is_refused_before_it_is_all_read():
    """Finding 10: a chunked body is counted as it arrives (TestClient reads a body whole
    before sending it, so the reader is driven directly here)."""
    import asyncio

    from starlette.requests import Request

    received = [0]

    async def receive():
        received[0] += 1
        return {"type": "http.request", "body": b"x" * 4096, "more_body": True}  # endless

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    with pytest.raises(web.Refused) as refused:
        asyncio.run(web.read_json(request))
    assert refused.value.status == 413
    assert received[0] == web.MAX_COMMAND // 4096 + 1  # stopped at /command's limit


def test_export_serves_only_the_current_chats_file(server):
    """Finding 12: after /new, the previous chat's export isn't this chat's."""
    client, _, _ = server
    command(client, "profile", "text")
    command(client, "say", "Hi")
    command(client, "export")
    assert client.get("/export", headers={**AUTH, **SAME}).status_code == 200
    command(client, "new")
    assert client.get("/export", headers={**AUTH, **SAME}).status_code == 409


def test_enter_in_the_terminal_never_puts_the_secret_in_a_process_argument(
    fetched_root, tiny_tokenizer, monkeypatch
):
    """The tripwire review (outside its range): webbrowser.open(link) runs `open LINK`."""
    import io

    opened: list[str] = []
    monkeypatch.setattr(web.webbrowser, "open", lambda url, *a, **k: opened.append(url))
    monkeypatch.setattr(web.sys, "stdin", io.StringIO("\n"))
    port = free_port()
    servers: list = []
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    thread = threading.Thread(
        target=web.serve,
        kwargs={
            "root": fetched_root,
            "port": port,
            "session_factory": lambda: session,
            "on_ready": servers.append,
            "write": lambda text: None,
        },
        daemon=True,
    )
    thread.start()
    for _ in range(100):
        if len(opened) >= 2:
            break
        threading.Event().wait(0.05)
    if servers:
        servers[0].should_exit = True
    thread.join(10)
    assert len(opened) == 2  # at start, and after Enter
    assert all("key=" not in url for url in opened), opened


def test_options_is_refused_on_any_path_not_only_known_routes(server):
    client, _, _ = server
    assert client.options("/nope").status_code == 405


# --- finding 8: the checks' tests, aimed at the review's surviving mutations


def test_every_data_route_in_the_table_refuses_without_the_secret(server):
    client, _, _ = server
    for (method, path), needs in ROUTES.items():
        url = path.replace("{file}", "app.js").replace("{name}", "thread")
        headers = {**SAME, "Origin": ORIGIN, "Content-Type": "application/json"}
        with client.stream(method, url, headers=headers, content=b"{}") as response:
            status = response.status_code  # /events let through would never end: not read
        assert (status == 401) == needs, (method, path, status)


def test_a_secret_with_extra_text_is_refused(server):
    client, _, _ = server
    headers = {**SAME, "Authorization": f"Bearer {SECRET}extra"}
    assert client.get("/view/thread", headers=headers).status_code == 401


def test_the_headers_are_on_a_stream_and_a_refusal(server):
    client, _, _ = server
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
        assert stream.headers["Cache-Control"] == "no-store"
    refused = client.get("/view/thread", headers=SAME)
    assert refused.status_code == 401 and refused.headers["X-Content-Type-Options"] == "nosniff"


def test_state_cancel_and_settings_answer_at_once_while_a_reply_streams(server):
    client, session, _ = server
    command(client, "profile", "text")
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 1.0
    said: list = []
    worker = threading.Thread(target=lambda: said.append(command(client, "say", "Hi")))
    worker.start()
    wait_until(lambda: writing(client))  # the reply is being written
    state = command(client, "state").json()["result"]
    assert state["running"]["command"] == "say"  # answered while the reply was written
    assert "settings" in command(client, "settings").json()["result"]
    assert command(client, "cancel").json()["result"]["cancelled"] is True
    worker.join(10)
    assert said and said[0].json()["result"]["call"]["stop_type"] == "cancelled"


def test_an_unexpected_error_inside_a_command_ends_the_session(server, monkeypatch):
    client, session, app = server
    command(client, "profile", "text")

    def boom(*args, **kwargs):
        raise RuntimeError("made-up crash")

    monkeypatch.setattr(app.state.worker.commands.session, "history", boom)
    first = command(client, "history").json()
    assert first.get("stopped") and "unexpected error" in first["error"]
    assert command(client, "state").json().get("stopped")  # every command after it too


def test_a_takeover_closes_the_old_tabs_stream(server):
    client, _, _ = server
    command(client, "state", tab="tab-1")
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
        lines = stream.iter_lines()
        assert json.loads(next(lines)) == {"event": "hello"}
        client.post("/tab", headers=POST, content=json.dumps({"tab": "tab-2"}))
        assert json.loads(next(lines)) == {"event": "not_live", "live": "tab-2"}
        assert list(lines) == []  # and the stream ended


def test_a_second_tab_doesnt_take_over_while_the_live_one_listens(server):
    client, _, _ = server
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as first:
        listening = first.iter_lines()  # kept: dropping the iterator closes the stream
        next(listening)  # tab-1 listening: live
        with client.stream("GET", "/events?tab=tab-2", headers={**AUTH, **SAME}) as second:
            lines = second.iter_lines()
            next(lines)
            assert json.loads(next(lines))["event"] == "not_live"


# --- the security review (Checkpoint 2)


def run_ui(fetched_root, tiny_tokenizer, port, on_open):
    """web.serve on `port` in a thread; returns (thread, servers)."""
    servers: list = []
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    thread = threading.Thread(
        target=web.serve,
        kwargs={
            "root": fetched_root,
            "port": port,
            "session_factory": lambda: session,
            "opener": on_open,
            "on_ready": servers.append,
            "write": lambda text: None,
            "read_lines": False,
        },
        daemon=True,
    )
    thread.start()
    return thread, servers


def stop_ui(thread, servers):
    for _ in range(100):
        if servers:
            break
        threading.Event().wait(0.05)
    servers[0].should_exit = True
    thread.join(10)


def test_the_port_is_already_listening_when_the_page_opens(fetched_root, tiny_tokenizer):
    """Finding 1: the port is bound and listening before the browser gets the link, so no
    other program can take it in between, and the page never meets "connection refused"."""
    import socket

    port = free_port()
    connected: list[bool] = []

    def on_open(path):
        with socket.socket() as probe:
            connected.append(probe.connect_ex(("127.0.0.1", port)) == 0)

    thread, servers = run_ui(fetched_root, tiny_tokenizer, port, on_open)
    stop_ui(thread, servers)
    assert connected == [True]


def test_starting_again_right_after_quitting_isnt_refused(fetched_root, tiny_tokenizer):
    """Finding 1: connections Clarity closed itself (TIME_WAIT) mustn't block a restart."""
    import httpx

    port = free_port()
    thread, servers = run_ui(fetched_root, tiny_tokenizer, port, lambda path: None)
    for _ in range(100):
        if servers and servers[0].started:
            break
        threading.Event().wait(0.05)
    with httpx.Client(
        verify=False, trust_env=False
    ) as client:  # keep-alive, then the server closes
        client.get(f"http://127.0.0.1:{port}/static/app.css")
    stop_ui(thread, servers)
    errors: list[BaseException] = []

    def again():
        try:
            run = web.serve
            run(
                fetched_root,
                port=port,
                session_factory=lambda: Session(
                    fetched_root,
                    backend=FakeBackend(),
                    tokenizer_loader=lambda r, v: tiny_tokenizer,
                ),
                opener=lambda path: None,
                on_ready=lambda server: setattr(server, "should_exit", True),
                write=lambda text: None,
                read_lines=False,
            )
        except BaseException as e:  # noqa: BLE001 - the refusal is the finding
            errors.append(e)

    second = threading.Thread(target=again, daemon=True)
    second.start()
    second.join(10)
    assert errors == []


def test_the_javascript_has_exactly_one_html_sink():
    """Finding 3: a second parser, or any of the newer sinks, would be a way around swap()."""
    import re
    from pathlib import Path

    code = (Path(web.__file__).parent / "web_static" / "app.js").read_text(encoding="utf-8")
    assert code.count("DOMParser") == 1 and code.count("parseFromString") == 1
    banned = re.findall(
        r"setHTMLUnsafe|parseHTMLUnsafe|srcdoc|new Function|setTimeout\(\s*['\"`]|"
        r"setInterval\(\s*['\"`]|\.src\s*=|href\s*=\s*['\"`]javascript",
        code,
    )
    assert banned == []
    # swap() takes a server fragment, never HTML built from text here.
    for call in re.findall(r"swap\(([^)]*)\)", code):
        assert "+" not in call and "`" not in call, call


def test_an_event_holding_nan_reaches_the_page_as_an_error(server):
    """Security review, finding 4: the stream's lines are always valid JSON."""
    client, _, app = server
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
        lines = stream.iter_lines()
        next(lines)  # hello
        app.state.worker.emit({"event": "record", "record": {"x": float("nan")}})
        event = json.loads(next(lines))
    assert event["event"] == "error" and "NaN" in event["message"]


def test_a_websocket_is_refused_before_any_route_sees_it(server):
    """Security review, finding 9: only HTTP passes the Host check's layer."""
    from starlette.websockets import WebSocketDisconnect

    client, _, _ = server
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/events", headers={**AUTH, "Host": f"127.0.0.1:{PORT}"}):
            pass


# --- found at Checkpoint 3 (v0.3.0)


def test_a_wait_cancelled_in_the_queue_doesnt_stop_the_worker(fetched_root, tiny_tokenizer):
    """A request cancelled while its command waits in the queue (a server shutting down, a
    task cancelled) must not end the worker thread: the command still runs, as a dropped
    POST's does, and the next one gets its turn."""
    import asyncio

    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    worker = web.Worker(session)
    submit(worker, "profile", "text")
    gate, entered, real = threading.Event(), threading.Event(), session.history
    session.history = lambda *a, **k: (entered.set(), gate.wait(5), real(*a, **k))[2]
    blocker = threading.Thread(target=lambda: submit(worker, "history", ""), daemon=True)
    blocker.start()
    assert entered.wait(10)  # the worker holds "history" at the gate

    async def cancel_a_queued_wait():
        task = asyncio.ensure_future(worker.ask("history", ""))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0.1)

    asyncio.run(cancel_a_queued_wait())
    gate.set()
    blocker.join(5)
    after: list = []
    later = threading.Thread(
        target=lambda: after.append(submit(worker, "history", "")), daemon=True
    )
    later.start()
    later.join(5)
    assert after and "result" in after[0]


FONT_SUFFIXES = {".woff2", ".woff", ".ttf", ".otf"}


def test_every_shipped_font_has_its_licence_beside_it_and_is_served(server):
    """The fonts ship as their publishers' unmodified files (IBM Plex reserves the name
    "Plex"), each family with its OFL.txt and copyright notice; the page loads only those."""
    import re
    from pathlib import Path

    client, _, _ = server
    folder = Path(web.__file__).parent / "web_static"
    fonts = sorted(p for p in folder.rglob("*") if p.suffix in FONT_SUFFIXES)
    assert fonts, "no fonts shipped"
    for font in fonts:
        licence = font.parent / "OFL.txt"
        assert licence.is_file(), f"{font.name} has no OFL.txt beside it"
        text = licence.read_text(encoding="utf-8")
        assert text.startswith("Copyright") and "SIL OPEN FONT LICENSE Version 1.1" in text
        response = client.get(f"/static/{font.name}")
        assert response.status_code == 200, font.name
        assert response.content == font.read_bytes()
        assert response.headers["content-type"].startswith("font/")
    css = (folder / "app.css").read_text(encoding="utf-8")
    assert sorted(re.findall(r'url\("/static/([^"]+)"\)', css)) == sorted(f.name for f in fonts)


# --- The security review's deferred findings (5 and 10)


def test_enter_s_redirect_is_deleted_by_a_new_tab_not_the_open_one(fetched_root, tiny_tokenizer):
    """Finding 5: after Enter writes a second redirect file, the open tab's next request
    mustn't delete it before the new tab opens it: only a new tab's first /events does."""
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    used: list[str] = []
    app = web.make_app(session, SECRET.encode(), port=PORT, on_first_use=lambda: used.append("1"))
    with TestClient(app, base_url=ORIGIN) as client:
        with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
            next(stream.iter_lines())
        assert used == ["1"]
        app.app.state.rearm(lambda: used.append("2"))  # the terminal's Enter
        command(client, "state")
        client.get("/view/thread", headers={**AUTH, **SAME})
        with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
            next(stream.iter_lines())  # the open tab's stream, reconnecting
        assert used == ["1"]
        with client.stream("GET", "/events?tab=tab-2", headers={**AUTH, **SAME}) as stream:
            next(stream.iter_lines())  # the new tab Enter opened
        assert used == ["1", "2"]


# --- The whole-branch review (multi-agent, batch 1)


def test_a_stream_whose_client_left_before_its_body_leaves_nothing_registered(server):
    """Finding 6: /events registered its listener and the tab's stream before the body's
    try/finally, so a client gone before the body started left both, and the next tab was
    told it isn't live."""
    import asyncio

    client, _, app = server
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/events",
        "raw_path": b"/events",
        "query_string": b"tab=gone",
        "root_path": "",
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", PORT),
        "headers": [
            (b"host", f"127.0.0.1:{PORT}".encode()),
            (b"authorization", f"Bearer {SECRET}".encode()),
            (b"sec-fetch-site", b"same-origin"),
        ],
    }

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            raise OSError("the client has gone")

    async def call():
        try:
            await app(scope, receive, send)
        except OSError:
            pass

    asyncio.run(call())
    with client.stream("GET", "/events?tab=next", headers={**AUTH, **SAME}) as stream:
        lines = [line for _, line in zip(range(2), stream.iter_lines(), strict=False)]
    assert lines[0] == '{"event": "hello"}', lines
    assert "not_live" not in lines[1], lines  # the next line is the next event, not a refusal


# --- The whole-branch review (Opus), batch 4


def test_a_long_message_reaches_the_command_table_and_a_huge_one_is_refused_with_the_limit(
    server,
):
    """MUST 1: /command's limit covers any message a context can take (the terminal and
    --json have none); other routes keep 64 KB. Over it, the 413 names the limit."""
    client, _, _ = server
    command(client, "profile", "text")
    long = command(client, "say", "a" * 100_000)
    assert long.status_code == 200 and long.json()["type"] == "ContextError"  # read, judged
    huge = command(client, "say", "a" * (8 * 1024 * 1024 + 1))
    assert huge.status_code == 413
    assert huge.text == "Too big: Clarity takes at most 8 MB at a time."
    tab = client.post("/tab", headers=POST, content=json.dumps({"tab": "x", "pad": "a" * 70_000}))
    assert tab.status_code == 413
    assert tab.text == "Too big: Clarity takes at most 64 KB at a time."


# --- The security review's follow-up (batch 3), and C14


def test_state_gives_a_random_value_made_for_this_start(server, fetched_root, tiny_tokenizer):
    """Batch 3, 1: the page keeps the first `start` it reads and stops at any other, so a
    program that takes the port after Clarity stops is never drawn or loaded. It's the same
    for every `state` of one start, random, and another start's differs."""
    client, _, _ = server
    first, again = (command(client, "state").json()["result"].get("start") for _ in range(2))
    assert isinstance(first, str) and len(first) >= 16 and first == again
    other = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    with TestClient(web.make_app(other, SECRET.encode(), port=PORT), base_url=ORIGIN) as second:
        assert command(second, "state").json()["result"]["start"] != first


def test_ctrl_c_tells_the_page_clarity_stopped_before_its_stream_closes(
    fetched_root, tiny_tokenizer
):
    """C14 (Opus review, batch 4): Ctrl-C closed the event stream with nothing on it, so
    the page said Clarity wasn't answering and kept trying. A plain `stopped` goes first."""
    import re
    import signal

    import httpx

    port, keys = free_port(), []

    def on_open(path):
        found = re.search(r"#key=([A-Za-z0-9_-]+)", path.read_text())
        assert found
        keys.append(found[1])

    thread, servers = run_ui(fetched_root, tiny_tokenizer, port, on_open)
    for _ in range(100):
        if servers and servers[0].started:
            break
        threading.Event().wait(0.05)
    lines: list[str] = []
    auth = {"Authorization": f"Bearer {keys[0]}"}
    with httpx.Client(verify=False, trust_env=False, timeout=10) as client:
        url = f"http://127.0.0.1:{port}/events?tab=t"
        with client.stream("GET", url, headers=auth) as stream:
            got = stream.iter_lines()
            assert next(got) == '{"event": "hello"}'
            servers[0].handle_exit(signal.SIGINT, None)  # as Ctrl-C does
            try:
                lines.extend(line for line in got if line and "heartbeat" not in line)
            except httpx.HTTPError:  # the server cut the stream: what came before counts
                pass
    thread.join(10)
    assert not thread.is_alive()
    assert [json.loads(line) for line in lines] == [{"event": "stopped"}]


HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>{{ 7*7 }}\' onmouseover=x a="'
# Every tag the templates write: anything else in a view came from text that wasn't escaped.
TAGS = {
    "a", "button", "code", "datalist", "dd", "details", "div", "dl", "dt", "form", "h2",
    "h3", "input", "label", "li", "mark", "ol", "option", "p", "pre", "span", "strong",
    "summary", "svg", "textarea", "ul", "use",
}  # fmt: skip


class Unsafe(HTMLParser):
    """A view parsed as a browser would: a tag no template writes, an attribute that runs
    or styles, a link off the page's icons, or a call ID that isn't one."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[tuple[str, ...]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in TAGS:
            self.found.append(("tag", tag))
        for name, value in attrs:
            if name.startswith("on") or name in ("style", "srcdoc", "formaction", "action"):
                self.found.append(("attribute", tag, name))
            if name == "href" and not (value or "").startswith("#i-"):
                self.found.append(("href", tag, value or ""))
            if name in ("data-call", "data-failed") and not CALL_ID.fullmatch(value or ""):
                self.found.append((name, tag, value or ""))


def test_every_view_shows_hostile_text_as_text(fetched_root, tiny_tokenizer, tmp_path):
    """Batch 3, 2: B5's fixture in every field a user, a model or a log fills (a message,
    replies, a mark's label, a note, a title, a failed call's error, probabilities), through
    every view. A failed call whose ID a hand-edited log changed gets no call controls."""
    from tests.test_web_details import fake_logprobs

    backend = FakeBackend([], tokenizer=tiny_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:

        def result(name, args=""):
            answer = command(client, name, args).json()
            assert "result" in answer, answer
            return answer["result"]

        result("profile", "text")
        result("set", "max_tokens 400")
        backend.replies = [HOSTILE, HOSTILE + " two"]
        backend.logprobs = fake_logprobs(
            tiny_tokenizer, [*tiny_tokenizer.encode(HOSTILE), tiny_tokenizer.tok.eos_token_id]
        )
        call = result("say", HOSTILE)["call"]["call_id"]
        result("regen")
        result("mark", f"{call} {HOSTILE}")
        result("note", f"{call} {HOSTILE}")
        result("title", HOSTILE)
        result("export")
        backend.fail_with = HOSTILE
        command(client, "say", HOSTILE)
        failed = result("prompt")["call"]["call_id"]
        named = ("replies", "compare", "details")  # these take a call
        uploaded = client.post(
            "/upload",
            params={"tab": "tab-1", "name": HOSTILE},
            headers={**POST, "Content-Type": "application/octet-stream"},
            content=HOSTILE.encode(),
        ).json()["upload"]  # text mode refuses to attach it: the sheet shows the refusal
        views = [(name, "") for name in web.VIEW_RENDERERS if name not in (*named, "attach")]
        views += [(name, call) for name in named]
        views += [("details", failed), ("attach", uploaded)]
        found, texts = {}, {}
        for name, at in views:
            key = "upload" if name == "attach" else "call"
            response = client.get(f"/view/{name}", params={key: at} if at else {}, headers=GET)
            assert response.status_code == 200, (name, response.text)
            parser = Unsafe()
            parser.feed(response.text)
            found[f"{name} {at}"] = parser.found
            texts[f"{name} {at}"] = response.text
        conversation = session.conversation
        assert conversation is not None and conversation.last_call is not None
        conversation.last_call["call_id"] = 'x" onclick="alert(3)'  # a hand-edited log
        parser = Unsafe()
        edited = client.get("/view/thread", headers=GET).text
        parser.feed(edited)
        found["thread, hand-edited"] = parser.found
    assert {view: bad for view, bad in found.items() if bad} == {}
    # And no Details on that line (fix review, batch 6, item 10): it would send no call ID.
    failed_line = re.search(r'<div class="problem"[^>]*>.*?</div>', edited, re.S)
    assert failed_line and "! Failed:" in failed_line.group(0)
    assert 'data-action="details"' not in failed_line.group(0)
    assert "likeliest per token" in texts[f"details {call}"]  # the probabilities were drawn
    assert "&lt;script&gt;" in texts["thread "] and "! Failed:" in texts["thread "]


def test_only_the_fixed_wait_attributes_skip_escaping():
    """Batch 3, 4: `| safe` is on `wait` alone, and `wait` is one of two fixed strings, so
    no text from a user, a model or a log reaches a view unescaped."""
    import re

    sets, safe = [], []
    for path in sorted((web.STATIC.parent / "web_templates").glob("*.html")):
        text = path.read_text(encoding="utf-8")
        assert "autoescape" not in text, path.name
        safe += re.findall(r"([\w.]+)\s*\|\s*safe\b", text)
        assert len(re.findall(r"\bsafe\b", text)) == len(re.findall(r"\|\s*safe\b", text))
        sets += re.findall(r"\{% set wait = (.*?) %\}", text)
    assert set(safe) == {"wait"}
    assert sorted(sets) == [
        "' aria-disabled=\"true\" aria-describedby=\"chats-waits\"' if waits else ''",
        "' aria-disabled=\"true\" aria-describedby=\"settings-waits\"' if waits else ''",
    ]
    assert "Markup" not in (web.STATIC.parent / "web.py").read_text(encoding="utf-8")


def serve_on(fetched_root, session, port, keys, read_lines=False):
    """web.serve on `port` in a thread, with `session`; the secret goes into `keys`, once
    for each time the redirect file is opened."""
    import re

    servers: list = []

    def on_open(path):
        found = re.search(r"#key=([A-Za-z0-9_-]+)", path.read_text())
        assert found
        keys.append(found[1])

    thread = threading.Thread(
        target=web.serve,
        kwargs={
            "root": fetched_root,
            "port": port,
            "session_factory": lambda: session,
            "opener": on_open,
            "on_ready": servers.append,
            "write": lambda text: None,
            "read_lines": read_lines,
        },
        daemon=True,
    )
    thread.start()
    for _ in range(100):
        if servers and servers[0].started:
            break
        threading.Event().wait(0.05)
    return thread, servers[0]


def test_ctrl_c_mid_reply_logs_the_call_as_stopped_and_returns_soon(fetched_root, tiny_tokenizer):
    """SHOULD 5 (c): serve's finally sends /cancel and waits for the record before the
    session closes, so a reply in progress is logged as stopped by you, and the server's
    graceful wait (2 s) holds the exit no longer than that."""
    import signal
    import time

    import httpx

    from clarity_chat.logs import read_records

    words = " one two three four five six seven eight nine ten eleven twelve"
    backend = FakeBackend([words], tokenizer=tiny_tokenizer)
    backend.piece_delay = 0.5  # 6 s in all, longer than the server's 2 s graceful wait
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    port, keys = free_port(), []
    thread, server = serve_on(fetched_root, session, port, keys)
    origin = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {keys[0]}", "Origin": origin}

    def post(client, name, args=""):
        body = {"command": name, "args": args, "id": 1, "tab": "t"}
        return client.post(f"{origin}/command", headers=headers, json=body)

    with httpx.Client(verify=False, trust_env=False, timeout=30) as client:
        assert "result" in post(client, "profile", "text").json()
        assert "result" in post(client, "set", "max_tokens 200").json()
        assert session.conversation is not None
        log = session.conversation.log.path
        with client.stream("GET", f"{origin}/events?tab=t", headers=headers) as stream:
            lines = stream.iter_lines()
            next(lines)  # hello

            def say() -> None:
                try:
                    post(client, "say", "Hi")
                except httpx.HTTPError:  # the server stops mid-reply: no answer may come
                    pass

            threading.Thread(target=say, daemon=True).start()
            for line in lines:  # until the reply is being written (heartbeats every 15 s)
                assert '"event": "heartbeat"' not in line
                if '"event": "text"' in line:
                    break
            began = time.monotonic()
            server.handle_exit(signal.SIGINT, None)
    thread.join(15)
    assert not thread.is_alive() and time.monotonic() - began < 5
    calls = [r for r in read_records(log) if r.get("type") == "call"]
    assert [c["stop_type"] for c in calls] == ["cancelled"]


def test_enter_s_redirect_file_is_gone_once_a_new_tab_connects(
    fetched_root, tiny_tokenizer, monkeypatch
):
    """SHOULD 5 (d): Enter in the terminal writes the redirect file again; the open tab's
    stream reconnecting leaves it, and the new tab's first /events deletes it."""
    import os

    import httpx

    read, write = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read, encoding="utf-8"))
    session = Session(
        fetched_root, backend=FakeBackend(), tokenizer_loader=lambda r, v: tiny_tokenizer
    )
    port, keys = free_port(), []
    thread, server = serve_on(fetched_root, session, port, keys, read_lines=True)
    redirect = fetched_root / "local" / "clarity-open.html"
    auth = {"Authorization": f"Bearer {keys[0]}"}

    def connect(client, tab):
        url = f"http://127.0.0.1:{port}/events?tab={tab}"
        with client.stream("GET", url, headers=auth) as stream:
            next(stream.iter_lines())

    try:
        with httpx.Client(verify=False, trust_env=False, timeout=10) as client:
            connect(client, "first")
            assert not redirect.exists()
            os.write(write, b"\n")  # Enter
            for _ in range(100):
                if len(keys) == 2:
                    break
                threading.Event().wait(0.05)
            assert keys == [keys[0], keys[0]] and redirect.exists()
            connect(client, "first")
            assert redirect.exists()
            connect(client, "second")
            assert not redirect.exists()
    finally:
        os.close(write)
        server.should_exit = True
        thread.join(10)


def test_the_rename_box_says_its_limit():
    """QA 5 (run28): the box stopped at 120 characters with no sentence, so the cut was
    silent. The limit is said beside it, and the box points to it."""
    page = (web.STATIC / "index.html").read_text(encoding="utf-8")
    box = re.search(r'<input type="text" id="rename-text"[^>]*>', page)
    assert box and 'maxlength="120"' in box.group(0)
    assert 'aria-describedby="rename-limit"' in box.group(0)
    assert re.search(r'id="rename-limit"[^>]*>At most 120 characters\.<', page)
