"""The terminal QA driver's guard on `clarity vector make PAIRS --profile NAME` (v0.4.5's QA
pass makes a vector). The driver runs Clarity outside the sandbox, and make prints the pairs
file's text, so PAIRS must name a file in the scratch root, judged as `run --openings` is;
`--profile` takes a name only, as for every command."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A scratch root holding a made-up pairs file, plus a link inside it to a file outside."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "plain.toml").write_bytes(b'description = "Plain against ornate replies"\n')
    outside = tmp_path / "secret.toml"
    outside.write_bytes(b'description = "not for the transcript"\n')
    (root / "away.toml").symlink_to(outside)
    return root


def outcome(call) -> object:
    """What `call` returned, or what it raised: a refusal is the driver's answer."""
    try:
        return call()
    except (Exception, SystemExit) as e:
        return e


def test_vector_make_inside_the_root_is_allowed(root):
    argv = ["vector", "make", "plain.toml", "--profile", "smol-chat-hf"]
    got = outcome(lambda: qa.check_clarity_args(root, argv))
    assert isinstance(got, argparse.Namespace), got  # not a refusal
    assert (got.command, got.pairs, got.profile) == ("vector", Path("plain.toml"), "smol-chat-hf")


@pytest.mark.parametrize(
    "path",
    [
        "../secret.toml",
        "away.toml",  # a link inside the root to a file outside
        "~/made-up-qa-pairs.toml",
        "/etc/hosts",
        "plain.toml/../plain.toml",  # `..` after a file
    ],
)
def test_vector_make_pairs_outside_the_root_are_refused(root, path):
    with pytest.raises(qa.Refused, match="vector make .*(scratch root|'\\.\\.')"):
        qa.check_clarity_args(root, ["vector", "make", path, "--profile", "smol-chat-hf"])


@pytest.mark.skipif(not hasattr(os, "link"), reason="no hard links")
def test_vector_make_pairs_with_a_second_name_are_refused(root, tmp_path):
    os.link(tmp_path / "secret.toml", root / "twin.toml")
    with pytest.raises(qa.Refused, match="vector make .*second name"):
        qa.check_clarity_args(root, ["vector", "make", "twin.toml", "--profile", "smol-chat-hf"])


@pytest.mark.parametrize(
    "profile", ["/x/evil.toml", "../evil", "profiles/smol-chat-hf.toml", "local/profiles/x.toml"]
)
def test_vector_make_profile_takes_a_name_only(root, profile):
    with pytest.raises(qa.Refused, match="--profile takes a profile name here"):
        qa.check_clarity_args(root, ["vector", "make", "plain.toml", "--profile", profile])
