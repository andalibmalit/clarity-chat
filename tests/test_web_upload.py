"""B3 (v0.3.1): POST /upload and the two views, offline. Made-up documents only.

An upload stays in memory: one for the app, replaced by the next, dropped once attached.
Nothing of a body is read before the secret, the Origin and Content-Length pass (S4)."""

from __future__ import annotations

import asyncio
import re

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.documents import reading
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.test_web import AUTH, CSP, GET, ORIGIN, PORT, SAME, SECRET, command

UPLOAD = {**AUTH, **SAME, "Origin": ORIGIN, "Content-Type": "application/octet-stream"}
NOTES = b"Line one.\nLine two.\nLine three.\n"
UPLOAD_ID = re.compile(r"upload-[0-9a-f]{12}")


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    backend = FakeBackend([" Hello!"] * 5, tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    app = web.make_app(session, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        yield client, session, app


def upload(client, data=NOTES, name="notes.md", tab="tab-1", headers=UPLOAD):
    return client.post("/upload", params={"tab": tab, "name": name}, headers=headers, content=data)


def raw(app, headers: dict[str, str], chunks: list[bytes], query: str = "tab=tab-1&name=n.md"):
    """POST /upload straight to the app, counting the body chunks it reads: the status, and
    how many chunks were taken."""
    taken, sent = [0], []
    body = [*chunks]

    async def receive():
        taken[0] += 1
        if body:
            chunk = body.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(body)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/upload",
        "raw_path": b"/upload",
        "query_string": query.encode(),
        "root_path": "",
        "headers": [
            (b"host", f"127.0.0.1:{PORT}".encode()),
            *((k.lower().encode(), v.encode()) for k, v in headers.items()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    text = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], taken[0], text.decode()


# --- the route


def test_an_upload_is_kept_in_memory_and_attached_by_its_id(chat):
    client, session, _ = chat
    answer = upload(client)
    assert answer.status_code == 200
    body = answer.json()
    assert UPLOAD_ID.fullmatch(body["upload"]) and body == {
        "upload": body["upload"],
        "name": "notes.md",
        "bytes": len(NOTES),
    }
    result = command(client, "attach", body["upload"]).json()["result"]
    assert result["attached"]["name"] == "notes.md"
    again = command(client, "attach", body["upload"]).json()
    assert again["error"] == "That file isn't uploaded any more. Choose it again."


def test_the_next_upload_replaces_the_last(chat):
    client, _, _ = chat
    first = upload(client).json()["upload"]
    second = upload(client, b"Other.\n", "other.md").json()["upload"]
    assert command(client, "attach", first).json()["error"] == (
        "That file isn't uploaded any more. Choose it again."
    )
    assert command(client, "attach", second).json()["result"]["attached"]["name"] == "other.md"


def test_from_the_page_a_path_is_refused(chat, tmp_path):
    client, _, _ = chat
    path = tmp_path / "notes.md"
    path.write_bytes(NOTES)
    answer = command(client, "attach", str(path)).json()
    assert answer["error"] == "From the page, attach a file by choosing it."


@pytest.mark.parametrize(("name", "kept"), [("dir/notes.md", "notes.md"), ("a\\b.md", "b.md")])
def test_a_name_keeps_only_its_last_part(chat, name, kept):
    client, _, _ = chat
    assert upload(client, name=name).json()["name"] == kept


def test_a_name_with_a_control_character_is_refused(chat):
    client, _, _ = chat
    answer = upload(client, name="notes\n.md")
    assert (answer.status_code, answer.text) == (
        400,
        "A file name can't hold a newline, a control character, a text-direction control or a "
        "broken character.",
    )


def test_a_tab_that_isn_t_live_is_refused(chat):
    client, _, _ = chat
    answer = upload(client, tab="tab-2")
    assert (answer.status_code, answer.text) == (409, "Clarity is open in another tab.")


def test_nothing_is_written_under_the_project_folder(chat, fetched_root):
    client, _, _ = chat
    before = sorted(p for p in fetched_root.rglob("*"))
    upload_id = upload(client).json()["upload"]
    command(client, "attach", f"{upload_id} check")
    command(client, "attach", upload_id)
    assert sorted(p for p in fetched_root.rglob("*")) == before


# --- refused before the body is read (S4)

GOOD = {**UPLOAD, "Content-Length": str(len(NOTES))}


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"Authorization": None}, 401),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Origin": None}, 403),
        ({"Origin": "http://evil.example"}, 403),
        ({"Content-Type": "application/json"}, 415),
        ({"Content-Length": None}, 411),
        ({"Content-Length": str(64 * 1024 * 1024 + 1)}, 413),
        ({"Host": "evil.example"}, 403),  # part 1's review, could 13: B3's wrong Host
    ],
)
def test_a_refusal_comes_before_any_of_the_body_is_read(chat, change, status):
    _, _, app = chat
    headers = {k: v for k, v in {**GOOD, **change}.items() if v is not None}
    code, taken, _ = raw(app, headers, [NOTES])
    assert (code, taken) == (status, 0)


def test_too_big_says_the_limit_and_is_checked_again_while_reading(chat):
    _, _, app = chat
    code, _, text = raw(app, {**GOOD, "Content-Length": str(64 * 1024 * 1024 + 1)}, [NOTES])
    assert (code, text) == (413, "Too big: Clarity attaches files up to 64 MB (67,108,864 bytes).")
    chunk = b"x" * (1024 * 1024)
    lying = {**GOOD, "Content-Length": "10"}
    code, taken, text = raw(app, lying, [chunk] * 70)
    assert (code, text) == (413, "Too big: Clarity attaches files up to 64 MB (67,108,864 bytes).")
    assert taken < 70  # it stopped reading


def test_the_new_responses_carry_the_security_headers(chat):
    client, _, _ = chat
    for response in (upload(client), upload(client, headers={**UPLOAD, "Origin": "x"})):
        assert response.headers["content-security-policy"] == CSP
        assert response.headers["x-content-type-options"] == "nosniff"


# --- the two views


def view(client, name, **params):
    return client.get(f"/view/{name}", params=params, headers=GET)


def test_the_attach_view_shows_the_check_of_an_upload(chat):
    client, session, _ = chat
    upload_id = upload(client).json()["upload"]
    sheet = view(client, "attach", upload=upload_id)
    assert sheet.status_code == 200 and "notes.md" in sheet.text
    ranged = view(client, "attach", upload=upload_id, unit="lines", first="2", last="3")
    assert ranged.status_code == 200 and "Lines 2 to 3:" in ranged.text
    assert session.conversation.pending == []  # a view attaches nothing


@pytest.mark.parametrize(
    "params",
    [
        {"upload": "../notes.md"},
        {"upload": "/etc/hosts"},
        {"upload": "upload-0123456789ab", "unit": "words", "first": "1", "last": "2"},
        {"upload": "upload-0123456789ab", "unit": "lines", "first": "1"},
        {"upload": "upload-0123456789ab", "unit": "lines", "first": "a", "last": "2"},
        {"upload": "upload-0123456789ab", "unit": "lines", "first": "١", "last": "2"},
        {},
    ],
)
def test_the_attach_view_checks_its_parameters(chat, params):
    client, _, _ = chat
    assert view(client, "attach", **params).status_code == 400


def test_the_pending_view_lists_what_s_waiting(chat):
    client, _, _ = chat
    upload_id = upload(client).json()["upload"]
    command(client, "attach", upload_id)
    chips = view(client, "pending")
    assert chips.status_code == 200 and "notes.md" in chips.text


def test_the_sheet_and_the_chips_show_hostile_text_as_text(chat):
    """test_web's hostile fixture as a file's name and text, through both views in a chat
    that can attach it (text mode, there, refuses it before the sheet shows a name)."""
    from tests.test_web import HOSTILE, Unsafe

    client, _, _ = chat
    upload_id = upload(client, HOSTILE.encode() + b"\n</document>\n", HOSTILE).json()["upload"]
    for response in (view(client, "attach", upload=upload_id), None):
        if response is None:
            command(client, "attach", upload_id)
            response = view(client, "pending")
        assert response.status_code == 200
        parser = Unsafe()
        parser.feed(response.text)
        # The name keeps what follows its last "/" (in "</script>"): the <img …> part.
        assert parser.found == [] and "&lt;img src=x onerror=alert(2)&gt;" in response.text


def test_a_whole_hostile_name_reaches_the_sheet_and_the_chips_as_text(chat):
    """The security review's could 5: the name above loses `<script>` to its last "/", so
    this one has none, and the sheet's heading and the chips get all of it, before C1 adds
    markup around them."""
    from tests.test_web import HOSTILE, Unsafe

    client, _, _ = chat
    name = HOSTILE.replace("/", "")
    upload_id = upload(client, b"Made-up.\n", name).json()["upload"]
    sheet = view(client, "attach", upload=upload_id)
    command(client, "attach", upload_id)
    for response in (sheet, view(client, "pending")):
        parser = Unsafe()
        parser.feed(response.text)
        assert parser.found == [] and "&lt;script&gt;" in response.text


def test_a_second_upload_while_one_is_read_is_refused(chat):
    """The security review's could 4: each upload holds up to 128 MB until it replaces the
    last, so only one is read at a time."""
    _, _, app = chat
    release = asyncio.Event()

    async def both():
        sent: list = []

        async def slow():
            await release.wait()
            return {"type": "http.request", "body": NOTES, "more_body": False}

        async def send(message):
            sent.append(message)

        first = asyncio.create_task(app(scope(), slow, send))
        await asyncio.sleep(0.05)
        second: list = []

        async def quick():
            return {"type": "http.request", "body": NOTES, "more_body": False}

        async def send_second(message):
            second.append(message)

        await app(scope(), quick, send_second)
        release.set()
        await first
        return [m["status"] for m in sent + second if m["type"] == "http.response.start"]

    assert asyncio.run(both()) == [200, 409]


def scope() -> dict:
    headers = {**GOOD, "Host": f"127.0.0.1:{PORT}"}
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/upload",
        "raw_path": b"/upload",
        "query_string": b"tab=tab-1&name=notes.md",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }


def test_an_upload_gone_between_its_check_and_its_read_is_refused(chat):
    """The security review's could 3: a second upload landing between `in` and `[]` raised
    KeyError, which stops the session."""
    client, session, _ = chat
    from clarity_chat.commands import Commands

    class Racing(dict):
        def __contains__(self, key):
            return True

    c = Commands(session, uploads=Racing())
    with pytest.raises(Exception) as e:
        c.run("attach", "upload-0123456789ab", lambda e: None)
    assert str(e.value) == "That file isn't uploaded any more. Choose it again."
