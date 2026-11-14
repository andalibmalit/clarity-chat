"""The web server's routes and security (v0.3.0 Group B: B0, B2, B3), offline.

The route table and every check are from notes/plans/v0.3.0.md. TestClient drives the app in
process; nothing listens on a port.
"""

from __future__ import annotations

import json
import sys
import threading

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.session import Session
from tests.fakes import FakeBackend

PORT = web.PORT
ORIGIN = f"http://127.0.0.1:{PORT}"
SECRET = "s3cret-for-tests"
AUTH = {"Authorization": f"Bearer {SECRET}"}
SAME = {"Sec-Fetch-Site": "same-origin"}
POST = {**AUTH, **SAME, "Origin": ORIGIN, "Content-Type": "application/json"}

# The plan's route table: method, path, needs the secret.
ROUTES = {
    ("GET", "/"): False,
    ("GET", "/static/{file}"): False,
    ("POST", "/command"): True,
    ("GET", "/events"): True,
    ("GET", "/view/{name}"): True,
    ("POST", "/tab"): True,
    ("GET", "/export"): True,
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
    response = client.request(method, path, headers=headers, content=b"{}")
    assert response.status_code == 401


@pytest.mark.parametrize("host", [f"localhost:{PORT}", "127.0.0.1:1", "evil.example", "127.0.0.1"])
def test_a_wrong_host_or_port_is_refused_everywhere(server, host):
    client, _, _ = server
    for path in ("/", "/static/app.js", "/view/thread"):
        assert client.get(path, headers={**AUTH, **SAME, "Host": host}).status_code == 403


@pytest.mark.parametrize("path", ["/events", "/view/thread", "/export"])
def test_a_cross_site_read_is_refused(server, path):
    client, _, _ = server
    assert client.get(path, headers={**AUTH, "Sec-Fetch-Site": "cross-site"}).status_code == 403


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
def test_a_post_needs_the_exact_origin_and_json(server, change, status):
    client, _, _ = server
    headers = {**POST, **{k: v for k, v in change.items() if v is not None}}
    for k, v in change.items():
        if v is None:
            headers.pop(k)
    body = json.dumps({"command": "state", "args": "", "id": 1, "tab": "tab-1"})
    assert client.post("/command", headers=headers, content=body).status_code == status


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
        assert "script-src 'self'" in h["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in h["Content-Security-Policy"]


@pytest.mark.parametrize(
    ("body", "status"),
    [(b"not json", 400), (b"[1, 2]", 400), (b'{"command": 3}', 400), (b"x" * 70_000, 413)],
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
    assert client.get("/view/nope", headers={**AUTH, **SAME}).status_code == 404
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
        assert not path.exists()  # deleted at the first authenticated request
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
    for name in web.STATIC_FILES:
        text = (folder / name).read_text(encoding="utf-8")
        assert "@import" not in text and "http://" not in text and "https://" not in text, name


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
        target=lambda: answers.append(worker.submit("profile", "text")), daemon=True
    )
    probe.start()
    probe.join(10)
    assert answers and "result" in answers[0], "the worker died: a submit never came back"
    assert worker.failed is None and broken not in worker.listeners
    assert "result" in worker.submit("say", "Hello")


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
    assert received[0] == web.MAX_BODY // 4096 + 1  # stopped at the limit


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
        response = client.request(method, url, headers=headers, content=b"{}")
        assert (response.status_code == 401) == needs, (method, path, response.status_code)


def test_a_secret_with_extra_text_is_refused(server):
    client, _, _ = server
    headers = {**SAME, "Authorization": f"Bearer {SECRET}extra"}
    assert client.get("/view/thread", headers=headers).status_code == 401


def test_the_csp_names_every_directive(server):
    client, _, _ = server
    csp = client.get("/").headers["Content-Security-Policy"]
    for directive in (
        "base-uri 'none'",
        "form-action 'none'",
        "object-src 'none'",
        "default-src 'self'",
    ):
        assert directive in csp


def test_the_headers_are_on_a_stream_and_a_refusal(server):
    client, _, _ = server
    with client.stream("GET", "/events?tab=tab-1", headers={**AUTH, **SAME}) as stream:
        assert stream.headers["Cache-Control"] == "no-store"
    refused = client.get("/view/thread", headers=SAME)
    assert refused.status_code == 401 and refused.headers["X-Content-Type-Options"] == "nosniff"


def test_state_cancel_and_settings_answer_at_once_while_a_reply_streams(server):
    client, session, _ = server
    command(client, "profile", "text")
    session.backend.replies, session.backend.piece_delay = [" A slow reply"], 0.3
    said: list = []
    worker = threading.Thread(target=lambda: said.append(command(client, "say", "Hi")))
    worker.start()
    threading.Event().wait(0.4)
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


def test_the_csp_requires_trusted_types(server):
    client, _, _ = server
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "require-trusted-types-for 'script'" in csp and "trusted-types clarity" in csp


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
    worker.submit("profile", "text")
    gate, real = threading.Event(), session.history
    session.history = lambda *a, **k: (gate.wait(5), real(*a, **k))[1]
    blocker = threading.Thread(target=lambda: worker.submit("history", ""), daemon=True)
    blocker.start()
    threading.Event().wait(0.2)  # the worker holds "history" at the gate

    async def cancel_a_queued_wait():
        task = asyncio.ensure_future(worker.ask("history", ""))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0.1)

    asyncio.run(cancel_a_queued_wait())
    gate.set()
    blocker.join(5)
    after: list = []
    later = threading.Thread(target=lambda: after.append(worker.submit("history", "")), daemon=True)
    later.start()
    later.join(5)
    assert after and "result" in after[0]
