"""The v0.3.2 part 2 whole-branch review's fixes that change behaviour (S1, S2, coulds 7, 8
and 9), each red first. Made-up documents, scratch roots."""

from __future__ import annotations

from clarity_chat.documents import library_add, library_documents
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


# --- could 8: /documents add reports the same bytes as `existing`


def test_documents_add_of_the_same_bytes_adds_nothing(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, GLOSSARY, "copy.md").json()["upload"]
    answer = command(client, "documents", f"add {upload_id} copy").json()
    assert answer.get("result", {}).get("existing") == "glossary", answer
    assert [d["name"] for d in library_documents(root)] == ["glossary"]


def test_the_terminal_s_documents_add_says_it_s_already_there(root, chat_tokenizer, tmp_path):  # noqa: F811
    from tests.test_attach_close import chat_loop

    copy = tmp_path / "copy.md"
    copy.write_bytes(GLOSSARY)
    loop, output, _ = chat_loop(root, chat_tokenizer)
    loop.start()
    output.clear()
    loop.handle(f"/documents add {copy}")
    said = "".join(output)
    assert "Already in your documents as glossary. Nothing new is added to them." in said
    assert [d["name"] for d in library_documents(root)] == ["glossary"]


# --- could 9: a typed "--replace" is a name, not the flag


def test_a_typed_replace_in_the_name_box_replaces_nothing(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, b"## Owl\nMade-up.\n", "owls.md").json()["upload"]
    answer = command(client, "documents", f"add {upload_id} glossary --replace").json()
    assert "error" in answer
    (doc,) = library_documents(root)
    assert doc["original"] == "glossary.md"


def test_replace_goes_before_the_upload(root, make):  # noqa: F811
    client, _ = make()
    upload_id = upload(client, b"## Owl\nMade-up.\n", "owls.md").json()["upload"]
    answer = command(client, "documents", f"add --replace {upload_id} glossary").json()
    assert answer.get("result", {}).get("replaced") is True, answer
    (doc,) = library_documents(root)
    assert doc["original"] == "owls.md"


def test_library_add_of_the_same_bytes(root):  # noqa: F811
    assert library_add(root, GLOSSARY, "again.md").get("existing") == "glossary"
