"""v0.4.0 A5: a chat template that can write today's date (`strftime_now`) is warned at a
conversation's start and resume (decision 10). Report only: the date isn't pinned.

No shipped template calls strftime_now, so these use made-up templates. Made-up text only.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.cli import ChatLoop
from clarity_chat.commands import Commands
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.render import Tokenizer, writes_date
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE as CHAT_TOML
from tests.conftest import FIXTURES, TOY_TEMPLATE, build_tokenizer
from tests.fakes import FakeBackend
from tests.test_json_chat import line, run
from tests.test_web import GET, ORIGIN, PORT, SECRET, command

DATED = '{{ strftime_now("%d %b %Y") }}\n' + TOY_TEMPLATE
# As Llama 3.1's: the date only when the caller gives none.
LLAMA_STYLE = (
    "{%- if not date_string is defined %}"
    '{%- set date_string = strftime_now("%d %b %Y") %}'
    "{%- endif %}Today: {{ date_string }}\n" + TOY_TEMPLATE
)
NAMES_DATE_STRING = "{% if date_string is defined %}{{ date_string }}{% endif %}" + TOY_TEMPLATE
COMMENT = "{# strftime_now would write today's date #}" + TOY_TEMPLATE
WARNING = "can write today's date"


def test_a_template_calling_strftime_now_writes_the_date():
    assert writes_date(DATED)
    assert writes_date(LLAMA_STYLE)


@pytest.mark.parametrize("template", [TOY_TEMPLATE, NAMES_DATE_STRING, COMMENT])
def test_a_template_that_cannot_write_the_date(template):
    assert not writes_date(template)


def test_a_shipped_template_does_not_write_the_date():
    """OLMo 3 Instruct's, the one shipped template the tests hold (SmolLM2's, OLMo 3's and
    Llama 3.1's were checked in the Hugging Face cache, decision 10)."""
    text = (FIXTURES / "olmo3-instruct" / "chat_template.jinja").read_text(encoding="utf-8")
    assert not writes_date(text)


@pytest.fixture(scope="module")
def dated_tokenizer(tmp_path_factory) -> Tokenizer:
    return build_tokenizer(tmp_path_factory.mktemp("tok-dated"), True, True, DATED)


@pytest.fixture
def dated(fetched_root, dated_tokenizer):
    backend = FakeBackend([" R1.", " R2."], tokenizer=dated_tokenizer)
    return Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: dated_tokenizer)


def warned(warnings) -> list[str]:
    return [w for w in warnings if WARNING in w]


def test_the_start_and_the_record_warn(dated):
    conversation = dated.new_conversation("chat")
    assert len(warned(conversation.warnings)) == 1
    first = next(read_records(conversation.log.path))
    assert warned(first["warnings"])
    state = Commands(dated).run("state", "", lambda e: None)
    assert warned(state["warnings"])


def test_resume_warns(dated):
    conversation = dated.new_conversation("chat")
    dated.say("Hi")
    dated.new_conversation("text")
    result = Commands(dated).run("resume", conversation.log.path.stem, lambda e: None)
    assert warned(result["warnings"])


def test_a_text_mode_profile_is_not_warned(dated):
    assert not warned(dated.new_conversation("text").warnings)


def test_the_pages_thread_shows_it(dated):
    app = web.make_app(dated, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN) as client:
        command(client, "profile", "chat")
        assert "can write today&#39;s date" in client.get("/view/thread", headers=GET).text


# --- the whole-branch review's should 1 and #19 ---------------------------------------------

BROKEN = "{% if x %}unclosed" + TOY_TEMPLATE  # doesn't parse
FAILED = "the chat template failed"


@pytest.fixture
def broken(fetched_root, chat_tokenizer, tmp_path_factory):
    """A session whose "broken" profile takes its template from a repo whose template
    doesn't parse; every other profile uses the toy template."""
    bad = build_tokenizer(tmp_path_factory.mktemp("tok-broken"), True, True, BROKEN)
    (fetched_root / "profiles" / "broken.toml").write_text(
        'template_repo = "test/broken"\ntemplate_revision = "' + "a" * 40 + '"\n' + CHAT_TOML,
        encoding="utf-8",
    )
    backend = FakeBackend([" R1.", " R2."], tokenizer=chat_tokenizer)
    return Session(
        fetched_root,
        backend=backend,
        tokenizer_loader=lambda r, v: bad if r == "test/broken" else chat_tokenizer,
    )


def test_a_template_that_does_not_parse_is_refused_at_start(broken):
    """Should 1: before v0.4.0 the first message was refused with render_chat's sentence;
    A5's parse at start let jinja2's error escape instead."""
    with pytest.raises(Exception) as caught:
        broken.new_conversation("broken")
    assert isinstance(caught.value, ConfigError), repr(caught.value)
    assert FAILED in str(caught.value)


def test_the_terminal_says_so_and_goes_on(broken):
    output: list[str] = []
    loop = ChatLoop(broken, "chat", read=lambda default: "", write=output.append)
    loop.start()
    try:
        handled = loop.handle("/profile broken")
    except Exception as e:  # noqa: BLE001 (the failure is the escape itself)
        pytest.fail(f"/profile raised {e!r}")
    assert handled and FAILED in "".join(output)
    assert loop.handle("still here") and broken.conversation is not None


def test_json_says_so_and_goes_on(broken):
    events = run(broken, [line("profile", "chat"), line("profile", "broken"), line("state")])
    assert not any(e.get("stopped") for e in events), events
    assert any(FAILED in str(e.get("error", "")) for e in events), events
    assert any(e.get("command") == "state" and e.get("result") for e in events)


def test_the_pages_picker_says_so_and_goes_on(broken):
    app = web.make_app(broken, SECRET.encode(), port=PORT)
    with TestClient(app, base_url=ORIGIN, raise_server_exceptions=False) as client:
        command(client, "profile", "chat")
        answer = command(client, "profile", "broken").json()
        assert not answer.get("stopped") and FAILED in str(answer.get("error")), answer
        assert command(client, "state").json().get("result")


def test_both_start_warnings_together(fetched_root, dated_tokenizer):
    """#19: an unread template_kwargs key and a dated template give both warnings."""
    (fetched_root / "profiles" / "both.toml").write_text(
        CHAT_TOML + '\n[template_kwargs]\ngreting = "hey"\n', encoding="utf-8"
    )
    backend = FakeBackend([" R1."], tokenizer=dated_tokenizer)
    session = Session(fetched_root, backend=backend, tokenizer_loader=lambda r, v: dated_tokenizer)
    warnings = session.new_conversation("both").warnings
    assert len(warned(warnings)) == 1
    assert any("never reads template_kwargs ['greting']" in w for w in warnings), warnings
