"""docs/pairs-prompt.md: its example pairs file is one `clarity vector make` reads."""

from __future__ import annotations

import re
from pathlib import Path

from clarity_chat.maker import read_pairs

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "pairs-prompt.md"


def test_the_page_s_example_reads_as_a_pairs_file(tmp_path):
    text = PAGE.read_text(encoding="utf-8")
    examples = re.findall(r"```toml\n(.*?)```", text, re.S)
    assert len(examples) == 1, "the page holds one TOML example"
    path = tmp_path / "brief.toml"
    path.write_text(examples[0], encoding="utf-8")
    pairs = read_pairs(path)
    assert len(pairs.pairs) == 8  # 2 system pairs x 2 messages x 2 openings


def test_the_prompt_asks_for_the_shape_read_pairs_reads():
    text = PAGE.read_text(encoding="utf-8")
    prompt = re.findall(r"## The prompt\n\n```\n(.*?)```", text, re.S)
    assert len(prompt) == 1, "the prompt is one fenced block"
    for word in (
        "description =",
        "author =",
        "messages =",
        "openings =",
        "[[systems]]",
        "positive =",
        "negative =",
        "{DESCRIPTION}",
    ):
        assert word in prompt[0], word
