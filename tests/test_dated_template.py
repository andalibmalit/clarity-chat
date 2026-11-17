"""v0.4.0 A5: a chat template that can write today's date (`strftime_now`) is warned at a
conversation's start and resume (decision 10). Report only: the date isn't pinned.

No shipped template calls strftime_now, so these use made-up templates. Made-up text only.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from clarity_chat import web
from clarity_chat.commands import Commands
from clarity_chat.logs import read_records
from clarity_chat.render import Tokenizer, writes_date
from clarity_chat.session import Session
from tests.conftest import FIXTURES, TOY_TEMPLATE, build_tokenizer
from tests.fakes import FakeBackend
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
        assert WARNING in client.get("/view/thread", headers=GET).text
