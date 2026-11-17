"""The v0.3.2 part 2 whole-branch review's fixes that change behaviour (S1, S2, coulds 7, 8
and 9), each red first. Made-up documents, scratch roots."""

from __future__ import annotations

from tests.test_web import command
from tests.test_web_search_c2 import GLOSSARY, get, make, root, text_of  # noqa: F401
from tests.test_web_upload import upload

# --- S2: Documents… draws only its own command's notice


def test_documents_shows_no_notice_from_another_screen(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    command(client, "documents", f"add {upload_id}")  # as Search it sends it
    assert "Added herons." not in text_of(get(client, "documents"))


def test_documents_shows_its_own_add_s_notice(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, b"## Heron\nTall.\n", "herons.md").json()["upload"]
    command(client, "documents", f"add {upload_id}")
    assert "Added herons." in text_of(get(client, "documents", notice="1"))
