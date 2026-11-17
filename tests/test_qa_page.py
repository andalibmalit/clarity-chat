"""The page QA's second run (notes/qa/v0.3.2/report.md): Search it's sentence before the
click names no folder from the default name, since the box can hold any name (should 2).
The false "Not kept" notice (should 1) and the replaced notice's stray "!" (could 3) are
browser checks in test_browser_search.py. Made-up documents, scratch roots."""

from __future__ import annotations

from tests.test_web_search_c2 import get, make, root, search_option, text_of  # noqa: F401
from tests.test_web_upload import upload

COPY = (
    "Clarity keeps a copy in local/documents/, under the name in the box. A document that "
    "already has that name is replaced."
)


def test_the_copy_sentence_names_no_folder_from_the_default_name(root, make):  # noqa: F811
    client, _ = make()
    said = []
    for name in ("herons.md", "glossary.md"):  # a free name, then a taken one
        upload_id = upload(client, b"## Heron\nTall.\n", name).json()["upload"]
        said.append(text_of(search_option(get(client, "attach", upload=upload_id))))
    for sentence in said:
        assert COPY in sentence, sentence
        assert "local/documents/herons/" not in sentence
        assert "local/documents/glossary/" not in sentence
