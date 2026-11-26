"""V1: the pairs file `clarity vector make` reads, in its crossed form. Every pair here is
made up and neutral."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clarity_chat.maker import MAX_BYTES, Pair, read_pairs
from clarity_chat.profiles import ConfigError

HEADER = 'description = "Plain replies against ornate ones."\nauthor = "me"\n'
SYSTEMS = (
    '[[systems]]\npositive = "Reply plainly."\nnegative = "Reply ornately."\n'
    '[[systems]]\npositive = "Answer in few words."\nnegative = "Answer in many words."\n'
)
CROSSED = HEADER + 'messages = ["Boil an egg?", "Pack for a hike?"]\nopenings = ["", "Well,"]\n'
CROSSED += SYSTEMS


def write(tmp_path: Path, text: str | bytes, name: str = "brief.toml") -> Path:
    path = tmp_path / name
    if isinstance(text, str):
        text = text.encode("utf-8")
    path.write_bytes(text)
    return path


def refusal(path: Path) -> str:
    with pytest.raises(ConfigError) as e:
        read_pairs(path)
    return str(e.value)


def test_the_crossed_form_expands_system_pair_first_then_message_then_opening(tmp_path):
    pairs = read_pairs(write(tmp_path, CROSSED))
    systems = [
        ("Reply plainly.", "Reply ornately."),
        ("Answer in few words.", "Answer in many words."),
    ]
    expected = [
        Pair(
            *systems[s], message, opening, (f"system pair {s + 1}", f"message {m}", f"opening {o}")
        )
        for s in range(2)
        for m, message in enumerate(["Boil an egg?", "Pack for a hike?"], 1)
        for o, opening in enumerate(["", "Well,"], 1)
    ]
    assert pairs.pairs == expected
    assert (pairs.form, pairs.counts) == ("crossed", {"systems": 2, "messages": 2, "openings": 2})
    assert (pairs.description, pairs.author) == ("Plain replies against ornate ones.", "me")
    assert (pairs.path, pairs.stem) == (tmp_path / "brief.toml", "brief")


def test_openings_left_out_mean_one_empty_opening(tmp_path):
    pairs = read_pairs(write(tmp_path, HEADER + 'messages = ["Boil an egg?"]\n' + SYSTEMS))
    assert [p.opening for p in pairs.pairs] == ["", ""]
    assert pairs.counts == {"systems": 2, "messages": 1, "openings": 1}


def test_two_pairs_are_enough(tmp_path):
    text = (
        HEADER
        + 'messages = ["Boil an egg?", "Pack for a hike?"]\n'
        + SYSTEMS.split('[[systems]]\npositive = "Answer')[0]
    )
    assert len(read_pairs(write(tmp_path, text)).pairs) == 2


def test_the_sha256_is_of_the_bytes_read_and_the_text_is_them_decoded(tmp_path):
    data = CROSSED.replace("\n", "\r\n").encode("utf-8")
    pairs = read_pairs(write(tmp_path, data))
    assert pairs.sha256 == hashlib.sha256(data).hexdigest()
    assert pairs.text == data.decode("utf-8")


def test_a_file_of_exactly_4_mb_is_read(tmp_path):
    data = CROSSED.encode("utf-8") + b"#"
    data += b"x" * (4 * 1024 * 1024 - len(data))
    assert len(read_pairs(write(tmp_path, data)).pairs) == 8


def test_a_missing_file(tmp_path):
    path = tmp_path / "brief.toml"
    assert refusal(path) == f"The pairs file {path} doesn't exist."


def test_a_file_that_cant_be_read(tmp_path):
    path = tmp_path / "brief.toml"
    path.mkdir()
    assert refusal(path).startswith(f"Can't read the pairs file {path}: ")


def test_a_file_over_4_mb(tmp_path):
    path = write(tmp_path, b"#" * (4 * 1024 * 1024 + 1))
    assert refusal(path) == f"{path} is over 4 MB; a pairs file is smaller."


def test_a_control_character_in_the_files_path_prints_escaped(tmp_path):
    """Checkpoint 1's mutation 20: refusals name the file as shown() gives it."""
    message = refusal(tmp_path / "a\x1b[2Jb" / "brief.toml")
    assert message.startswith("The pairs file ") and message.endswith(" doesn't exist.")
    assert "a\\x1b[2Jb" in message and "\x1b" not in message


def test_the_file_is_read_no_further_than_the_cap(tmp_path):
    """S8 (mutation 19): one read of at most 4 MB and a byte, never the whole file."""
    sizes = []

    class Reader(io.BytesIO):
        def read(self, n: int | None = -1) -> bytes:
            sizes.append(n)
            return super().read(n)

    class Capped(type(tmp_path)):
        def open(self, *args, **kwargs):
            return Reader(CROSSED.encode())

    read_pairs(Capped(tmp_path / "brief.toml"))
    assert sizes == [MAX_BYTES + 1]


def test_a_file_nested_too_deeply(tmp_path):
    """Checkpoint 1's S1: tomllib raises RecursionError, not a ValueError, at 2,000 deep."""
    path = write(tmp_path, "description = " + "[" * 2000 + "]" * 2000 + "\n")
    try:
        got = read_pairs(path)
    except Exception as e:
        got = e
    assert isinstance(got, ConfigError), f"not refused: {got!r}"
    assert str(got).startswith(f"{path} isn't a TOML file: ")


@pytest.mark.parametrize("data", [b"description = \xff\n", b"description = \n"])
def test_a_file_that_isnt_utf8_or_toml(tmp_path, data):
    path = write(tmp_path, data)
    message = refusal(path)
    assert message.startswith(f"{path} isn't a TOML file: ")
    assert message.endswith(".")
    assert "utf-8" in message if b"\xff" in data else "line 1" in message


@pytest.mark.parametrize("name", ["my pairs.toml", "brief.v2.toml", "brief;rm.toml", ".toml"])
def test_a_file_name_that_cant_name_a_vector(tmp_path, name):
    path = write(tmp_path, CROSSED, name)
    assert refusal(path) == (
        "Name the pairs file with letters, digits, - and _ only: its name becomes the vector's."
    )


REFUSALS = {
    "an unknown top-level key": (
        CROSSED.replace("author", 'colour = "red"\nauthor'),
        "{p}: unknown keys ['colour'] in the top level. "
        "Allowed there: author, description, messages, openings, systems.",
    ),
    "messages written after [[systems]]": (
        HEADER + SYSTEMS + 'messages = ["Boil an egg?"]\n',
        "{p}: unknown keys ['messages'] in system pair 2. Allowed there: positive, negative.",
    ),
    "no description": (
        CROSSED.replace('description = "Plain replies against ornate ones."\n', ""),
        "{p} needs description: one line saying what the two sides contrast.",
    ),
    "an empty description": (
        CROSSED.replace("Plain replies against ornate ones.", " "),
        "{p} needs description: one line saying what the two sides contrast.",
    ),
    "no author": (
        CROSSED.replace('author = "me"\n', ""),
        '{p} needs author: who or what wrote the pairs, as you say it ("me" is fine).',
    ),
    "an empty author": (
        CROSSED.replace('author = "me"', 'author = ""'),
        '{p} needs author: who or what wrote the pairs, as you say it ("me" is fine).',
    ),
    "no messages": (
        HEADER + SYSTEMS,
        "{p} needs messages: the user messages, each read with every system pair.",
    ),
    "no systems": (
        HEADER + 'messages = ["Boil an egg?", "Pack for a hike?"]\n',
        "{p} needs systems: one [[systems]] table per system pair, with positive and negative.",
    ),
    "no negative side": (
        CROSSED.replace('negative = "Reply ornately."\n', ""),
        "{p} needs negative in system pair 1: that side's system message.",
    ),
    "a description that isn't a string": (
        CROSSED.replace('"Plain replies against ornate ones."', "3"),
        "{p}: description must be a string.",
    ),
    "an author that isn't a string": (
        CROSSED.replace('author = "me"', "author = true"),
        "{p}: author must be a string.",
    ),
    "messages that aren't a list": (
        CROSSED.replace('["Boil an egg?", "Pack for a hike?"]', '"Boil an egg?"'),
        "{p}: messages must be a list of strings.",
    ),
    "a message that isn't a string": (
        CROSSED.replace('"Pack for a hike?"', "2"),
        "{p}: messages must be a list of strings.",
    ),
    "openings that aren't a list of strings": (
        CROSSED.replace('["", "Well,"]', '["", 1]'),
        "{p}: openings must be a list of strings.",
    ),
    "systems that aren't tables": (
        HEADER + 'messages = ["Boil an egg?"]\nsystems = ["Reply plainly.", "Reply ornately."]\n',
        "{p}: system pair 1 must be a table with positive and negative.",
    ),
    "systems that aren't a list": (
        HEADER + 'messages = ["Boil an egg?"]\nsystems = "Reply plainly."\n',
        "{p}: systems must be a list of [[systems]] tables.",
    ),
    "a side that isn't a string": (
        CROSSED.replace('negative = "Answer in many words."', "negative = 1"),
        "{p}: negative in system pair 2 must be a string.",
    ),
    "an empty message": (
        CROSSED.replace('"Pack for a hike?"', '" "'),
        "{p}: message 2 is empty; a message needs text.",
    ),
    "a system pair the same on both sides": (
        CROSSED.replace("Answer in many words.", "Answer in few words."),
        "{p}: system pair 2 has the same text on both sides.",
    ),
    "one pair": (
        HEADER
        + 'messages = ["Boil an egg?"]\n'
        + SYSTEMS.split('[[systems]]\npositive = "Answer')[0],
        "{p} makes 1 pair; make needs at least 2.",
    ),
    "no pairs": (
        CROSSED.replace('openings = ["", "Well,"]', "openings = []"),
        "{p} makes 0 pairs; make needs at least 2.",
    ),
    "1,025 pairs": (
        HEADER
        + f"messages = {json.dumps([f'Question {i}?' for i in range(25)])}\n"
        + f"openings = {json.dumps([f'Start {i}' for i in range(41)])}\n"
        + SYSTEMS.split('[[systems]]\npositive = "Answer')[0],
        "{p} makes 1,025 pairs; make takes at most 1,024.",
    ),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_each_refusal_with_its_sentence(tmp_path, case):
    text, sentence = REFUSALS[case]
    path = write(tmp_path, text)
    assert refusal(path) == sentence.format(p=path)


def test_1024_pairs_are_accepted(tmp_path):
    text = (
        HEADER
        + f"messages = {json.dumps([f'Question {i}?' for i in range(32)])}\n"
        + f"openings = {json.dumps([f'Start {i}' for i in range(16)])}\n"
        + SYSTEMS
    )
    assert len(read_pairs(write(tmp_path, text)).pairs) == 1024


KEYS = st.sampled_from(
    ["description", "author", "messages", "openings", "systems", "positive", "negative", "x"]
)
VALUES = st.recursive(
    st.one_of(st.text(max_size=6), st.integers(), st.booleans()),
    lambda inner: st.one_of(st.lists(inner, max_size=4), st.dictionaries(KEYS, inner, max_size=4)),
    max_leaves=24,
)


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(map(toml_value, value)) + "]"
    assert isinstance(value, dict)
    items = (f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items())
    return "{" + ", ".join(items) + "}"


DOCUMENTS = st.one_of(
    st.dictionaries(KEYS, VALUES, max_size=6).map(
        lambda table: "".join(f"{json.dumps(k)} = {toml_value(v)}\n" for k, v in table.items())
    ),
    st.text(),
)


@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(DOCUMENTS)
def test_any_document_is_read_or_refused_with_a_config_error(tmp_path, document):
    path = write(tmp_path, document)
    try:
        read_pairs(path)
    except ConfigError:
        pass
