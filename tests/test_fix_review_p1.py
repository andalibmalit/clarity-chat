"""Part 1's fix review (v0.3.1): its must, the behaviour behind shoulds 3 to 7, and could
15. Made-up documents and made-up TOML only."""

from __future__ import annotations

import json
import sys

import pytest
from starlette.testclient import TestClient

from clarity_chat import cli, web
from clarity_chat.commands import Commands, check_lines
from clarity_chat.documents import (
    FRAMING,
    SENTENCES,
    Attached,
    DocumentRefused,
    Room,
    content,
    fit,
    reading,
)
from clarity_chat.profiles import ConfigError, load_defaults, load_profile
from clarity_chat.render import build_prompt, prompt_text
from clarity_chat.session import Session
from tests.fakes import FakeBackend
from tests.pdfs import encrypted_pdf, make_pdf
from tests.test_session_attach import chat_session, doc, rewrite_turns
from tests.test_web import GET, ORIGIN, PORT, SECRET, command
from tests.test_web_upload import upload

HUGE = "1" * 4301
LONG = "".join(f"Made-up line {n}.\n" for n in range(200))


@pytest.fixture
def chat(fetched_root, chat_tokenizer):
    reading._last = None
    return chat_session(fetched_root, chat_tokenizer)


# --- must 1: a backslash is a legal character in a POSIX file name


@pytest.mark.skipif(sys.platform == "win32", reason="a backslash can't be in a Windows file name")
def test_a_name_with_a_backslash_is_attached_resumed_and_exported(chat, tmp_path):
    from clarity_chat.export import export_log

    session, _ = chat
    path = tmp_path / "notes\\v2.md"
    path.write_text("Made-up.\n", encoding="utf-8")
    Commands(session).run("attach", str(path), lambda e: None)
    session.say("a")
    log = session.conversation.log.path
    session.resume(log)  # neither refuses
    export_log(session, log)
    names = [a["name"] for t in session.conversation.turns.values() for a in t.attachments]
    assert names == ["notes\\v2.md"]


def test_a_logged_name_with_a_slash_is_still_refused(chat):
    session, _ = chat
    session.attach(doc())
    session.say("a")
    path = rewrite_turns(session, lambda r: r["attachments"][0].update(name="a/b.md"))
    session.new_conversation("chat")
    with pytest.raises(ConfigError):
        session.resume(path)


# --- should 3: a chat with no room left says so, with no negative number


def test_a_chat_with_no_room_left_refuses_before_counting(chat, tmp_path):
    session, _ = chat
    c = Commands(session)
    c.run("say", "Hi", lambda e: None)
    c.run("set", "max_tokens 100000", lambda e: None)  # past the 8,192 context
    path = tmp_path / "notes.md"
    path.write_text("Made-up.\n", encoding="utf-8")
    for args in (f"{path} check", str(path)):
        with pytest.raises(DocumentRefused) as e:
            c.run("attach", args, lambda e: None)
        assert str(e.value) == SENTENCES["no_room"]
        assert "-" not in str(e.value)


# --- should 4: the bound only where its premise holds, and tight where it does


def test_the_bound_is_only_for_byte_level_tokenizers_with_no_normalizer(
    olmo_tokenizer, chat_tokenizer, tmp_path
):
    from tests.conftest import build_tokenizer

    assert olmo_tokenizer.max_token_bytes == 128 and chat_tokenizer.max_token_bytes == 4
    other = build_tokenizer(tmp_path / "nfc", None, False)
    from tokenizers import normalizers

    other.tok.backend_tokenizer.normalizer = normalizers.NFC()
    other.__dict__.pop("max_token_bytes", None)
    assert other.max_token_bytes is None  # NFC can shrink text: count exactly


def olmo_room(olmo_tokenizer, profile, n_ctx: int, max_tokens: int = 16) -> Room:
    def tokens(next_content: str) -> int:
        return len(build_prompt(olmo_tokenizer, profile, [("user", next_content)]).ids)

    frame = prompt_text(profile, [("user", "")], olmo_tokenizer)
    return Room(
        tokens,
        lambda n: n_ctx - n - max_tokens - 1,
        max_tokens,
        frame_bytes=len(frame.encode("utf-8")),
        token_bytes=olmo_tokenizer.max_token_bytes,
    )


def test_on_olmo_3_the_bound_never_refuses_what_fits(olmo_tokenizer, root):
    """20,000 spaces: OLMo 3 merges runs of spaces, so the exact count is 1.4x the bound.
    Right at the boundary the exact count fits, and the bound mustn't refuse it."""
    profile = load_profile(root, "chat", load_defaults(root))
    new = Attached("spaces.md", " " * 20_000 + "\n", None)
    exact = olmo_room(olmo_tokenizer, profile, 10**6)
    before = exact.tokens("")
    cost = exact.tokens(content("", [new], FRAMING)) - before
    n_ctx = before + cost + 16 + 1  # free is exactly the cost
    found = fit(olmo_room(olmo_tokenizer, profile, n_ctx), [], new, FRAMING)
    assert (found.tokens, found.fits, found.at_least) == (cost, True, None)


def test_a_trimming_template_s_bound_counts_the_content_trimmed(root, tmp_path):
    """A template that trims (as Llama 3.1's does) drops a framing line's edge spaces: the
    bound must too, or it refuses what fits."""
    from tests.test_spans import LOOP, toy

    tokenizer = toy(tmp_path, LOOP.replace("BODY", "m.content | trim"))
    profile = load_profile(root, "chat", load_defaults(root))
    framing = " " * 20_000 + "Note"
    new = Attached("notes.md", "Made-up.\n", None)

    def room(n_ctx: int) -> Room:
        def tokens(next_content: str) -> int:
            return len(build_prompt(tokenizer, profile, [("user", next_content)]).ids)

        frame = prompt_text(profile, [("user", "")], tokenizer)
        return Room(tokens, lambda n: n_ctx - n - 16 - 1, 16, len(frame.encode()), 4)

    before = room(10**6).tokens("")
    cost = room(10**6).tokens(content("", [new], framing)) - before
    found = fit(room(before + cost + 17), [], new, framing)
    assert found.fits is True and found.tokens == cost


# --- should 5: a range change renders only the range


def test_a_range_change_renders_only_the_range(chat, monkeypatch):
    session, _ = chat
    d = doc(LONG, "long.md")
    session.check(d)
    renders = []
    real = session.room

    def room(*args, **kwargs):
        made = real(*args, **kwargs)

        def tokens(text):
            renders.append(text)
            return made.tokens(text)

        return type(made)(**{**made.__dict__, "tokens": tokens})

    monkeypatch.setattr(session, "room", room)
    chosen = session.check(d, "lines", 5, 9)["chosen"]
    assert chosen["range"] == [5, 9]
    assert len(renders) <= 2  # the range, and the prompt without it at most
    session.say("a")  # the chat changed: counted again
    renders.clear()
    session.check(d, "lines", 5, 9)
    assert len(renders) > 2


# --- should 6: a number tomllib can't take is a refusal, never a stopped session


def test_set_with_a_huge_number_is_refused(chat):
    session, _ = chat
    with pytest.raises(ConfigError):
        Commands(session).run("set", f"max_tokens {HUGE}", lambda e: None)


def test_a_profile_with_a_huge_number_is_refused(fetched_root):
    path = fetched_root / "profiles" / "huge.toml"
    text = (fetched_root / "profiles" / "chat.toml").read_text(encoding="utf-8")
    path.write_text(text.replace("seed = 42", f"seed = {HUGE}"), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_profile(fetched_root, "huge", load_defaults(fetched_root))


def test_a_logged_profile_with_a_huge_number_is_refused(chat):
    session, _ = chat
    session.say("a")
    path = session.conversation.log.path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["profile"]["toml"] = rows[0]["profile"]["toml"].replace("seed = 42", f"seed = {HUGE}")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    session.new_conversation("chat")
    with pytest.raises(ConfigError):
        session.resume(path)


def test_set_with_a_huge_number_from_the_page_leaves_the_session_going(
    fetched_root, chat_tokenizer
):
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        answer = command(client, "set", f"max_tokens {HUGE}").json()
        assert answer["type"] == "ConfigError" and not answer.get("stopped")


# --- should 7: lines the fixes added that no test held


def test_json_attach_writes_a_reading_event_line_before_the_start(
    fetched_root, monkeypatch, capsys, tmp_path
):
    path = tmp_path / "notes.md"
    path.write_text("Made-up.\n", encoding="utf-8")
    monkeypatch.setattr(cli, "cmd_chat_json", lambda *a, **k: 0)
    argv = ["--root", str(fetched_root), "chat", "--json", "--profile", "chat"]
    assert cli.main([*argv, "--attach", str(path)]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert json.loads(first) == {"event": "reading", "name": "notes.md"}


def test_check_s_lines_when_the_whole_doesn_t_fit(chat, tmp_path):
    session, backend = chat
    backend._n_ctx = 1500
    c = Commands(session)
    c.run("say", "Made-up words in a long message. " * 28, lambda e: None)
    path = tmp_path / "notes.md"
    path.write_text(LONG[:700], encoding="utf-8")
    check = c.run("attach", f"{path} check", lambda e: None)["check"]
    lines = check_lines(check)
    start, pre = check["start_that_fits"], check["preselected"]
    assert (
        SENTENCES["start_that_fits"].format(
            units="lines", last=start["range"][1], tokens=f"{start['tokens']:,}"
        )
        in lines
    )
    reason = SENTENCES["preselected"].format(max_tokens="16").removesuffix(".")
    assert f"{reason}: lines 1 to {pre['range'][1]}." in lines
    assert SENTENCES["new_chat"].format(free=f"{check['free_new_chat']:,}") in lines


def test_more_than_is_the_bound_less_one(chat):
    session, _ = chat
    d = doc("".join(f"Made-up line {n}.\n" for n in range(40_000)), "big.md")
    check = session.check(d)
    at_least = check["tokens_whole_at_least"]
    assert SENTENCES["doesnt_fit"].format(
        tokens=f"more than {at_least - 1:,}", free=f"{check['free']:,}"
    ) in check_lines(check)


def test_the_refusal_with_no_start_that_leaves_room(chat, tmp_path):
    """The suggestion's other branch: a start fits, but none leaves max_tokens."""
    session, backend = chat
    path = tmp_path / "long.md"
    path.write_text(LONG, encoding="utf-8")
    c = Commands(session)
    for n_ctx in range(150, 400):
        backend._n_ctx = n_ctx
        check = c.run("attach", f"{path} check", lambda e: None)["check"]
        if check["start_that_fits"] and not check["preselected"]:
            break
    else:
        pytest.fail("no context gives a start with no room for a message")
    k = check["start_that_fits"]["range"][1]
    with pytest.raises(DocumentRefused) as e:
        c.run("attach", str(path), lambda e: None)
    no_start = SENTENCES["no_start"].format(unit="line", max_tokens="16")
    assert str(e.value).endswith(f" {no_start} Lines 1 to {k} fit.")


# --- should 2: the sheet's facts; could 15: check says the max_tokens it counted with


def test_the_sheet_shows_a_pdf_s_facts(fetched_root, chat_tokenizer):
    backend = FakeBackend([" Hello!"], tokenizer=chat_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    with TestClient(web.make_app(session, SECRET.encode(), port=PORT), base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        for data, fact in (
            (make_pdf(["Made-up.", "", "More."]), "PDF · 3 pages, 1 with no text"),
            (encrypted_pdf(["Made-up."]), SENTENCES["encrypted"]),
        ):
            reading._last = None
            upload_id = upload(client, data, "guide.pdf").json()["upload"]
            sheet = client.get("/view/attach", params={"upload": upload_id}, headers=GET).text
            assert fact in sheet.replace("&#39;", "'")


def test_check_says_the_max_tokens_it_counted_with(chat):
    session, _ = chat
    Commands(session).run("set", "max_tokens 40", lambda e: None)
    check = session.check(doc())
    assert check["max_tokens"] == 40
    assert SENTENCES["free"].format(free=f"{check['free']:,}") in check_lines(check)
