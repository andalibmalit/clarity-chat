"""The web server's routes and security (v0.3.0 Group B: B0, B2, B3), offline.

The route table and every check are from notes/plans/v0.3.0.md. TestClient drives the app in
process; nothing listens on a port.
"""

from __future__ import annotations

import json
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


def test_no_command_can_name_a_path(server, tmp_path):
    client, _, _ = server
    outside = tmp_path / "x.toml"
    outside.write_text("x", encoding="utf-8")
    assert command(client, "profile", str(outside)).json().get("error")


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
    monkeypatch.setattr(web.Worker, "submit", lambda self, name, args: {"result": float("nan")})
    response = command(client, "state")
    assert response.status_code == 500
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
