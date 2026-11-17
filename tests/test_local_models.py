"""v0.4.0 B4: your own model entries in local/models.toml, their lock entries in
local/models.lock (decision 12). The two existing routes only.

A scratch root, downloads and the converter stubbed; made-up text only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_chat import cli, models
from clarity_chat.models import check_weights_quick, list_profiles, lock_entry, read_lock, verify
from clarity_chat.profiles import ConfigError, load_models, load_server_config
from clarity_chat.session import Session
from tests.conftest import CHAT_PROFILE, MODELS
from tests.fakes import FakeBackend
from tests.test_models import GGUF_MODEL, WEIGHTS, WEIGHTS_SHA256

MINE = MODELS.replace("[tiny]", "[my-tiny]")
MINE_GGUF = GGUF_MODEL.format(sha256=WEIGHTS_SHA256).replace("[ready]", "[my-ready]")
LOCAL_TOML = "local/models.toml"
LOCAL_LOCK = "local/models.lock"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def mine(fetched_root, tmp_path, monkeypatch):
    """A root with tiny fetched (models.lock), and two entries of yours: my-tiny (convert)
    and my-ready (gguf), with downloads and the converter stubbed."""
    root = fetched_root
    (root / "local").mkdir(exist_ok=True)
    (root / LOCAL_TOML).write_text(MINE + "\n" + MINE_GGUF, encoding="utf-8")
    cache = tmp_path / "hf-cache"
    (cache / "tok").mkdir(parents=True)
    (cache / "tok" / "generation_config.json").write_text(
        json.dumps({"temperature": 0.5}), encoding="utf-8"
    )
    (cache / "ready-Q8_0.gguf").write_bytes(WEIGHTS)
    monkeypatch.setattr(models, "_download_file", lambda repo, rev, name: cache / name)
    monkeypatch.setattr(models, "_download", lambda repo, rev, patterns: cache / "tok")
    server = load_server_config(root)
    monkeypatch.setattr(models, "git_state", lambda path: {"commit": server.llama_cpp_commit})

    class Converted:
        """The converter: writes the --outfile and the versions file the shim writes."""

        def __init__(self, command, **kwargs):
            outfile = Path(command[command.index("--outfile") + 1])
            outfile.write_bytes(b"GGUF converted")
            versions = {"python": "3.13", "packages": {"torch": "0", "transformers": "0"}}
            Path(command[command.index("--outtype") - 2]).parent.mkdir(exist_ok=True)
            (outfile.parent / "converter-versions.json").write_text(json.dumps(versions))
            self.stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr(models.subprocess, "Popen", Converted)
    return root


def loaded(root: Path, name: str):
    """The model entry `name`, read from either models.toml."""
    found = load_models(root)
    assert name in found, f"{name} isn't loaded: {sorted(found)}"
    return found[name]


def fetch(root: Path, name: str) -> dict:
    return models.fetch(root, loaded(root, name), load_server_config(root))


@pytest.mark.parametrize("name", ["my-tiny", "my-ready"])
def test_your_entry_is_fetched_into_local_models_lock(mine, name):
    shipped = sha(mine / "models.lock")
    fetch(mine, name)
    assert sha(mine / "models.lock") == shipped
    assert name in json.loads((mine / LOCAL_LOCK).read_text(encoding="utf-8"))["models"]
    assert name in read_lock(mine)["models"] and "tiny" in read_lock(mine)["models"]
    entry = lock_entry(mine, loaded(mine, name))
    assert check_weights_quick(mine, entry).is_file()


def test_your_entry_is_verified_listed_and_used_by_your_profile(mine, chat_tokenizer):
    fetch(mine, "my-tiny")
    assert verify(mine, "my-tiny")[0]
    (mine / "local" / "profiles").mkdir()
    (mine / "local" / "profiles" / "mine.toml").write_text(
        CHAT_PROFILE.replace('"tiny"', '"my-tiny"'), encoding="utf-8"
    )
    (listed,) = [p for p in list_profiles(mine) if p["name"] == "mine"]
    assert listed["fetched"] and listed["error"] is None
    backend = FakeBackend([" A.", " B."], tokenizer=chat_tokenizer)
    session = Session(mine, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    session.new_conversation("chat")
    shipped = session.say("Hi").record["model"]
    session.new_conversation("mine")
    yours = session.say("Hi").record["model"]
    assert sorted(yours) == sorted(shipped) and yours["name"] == "my-tiny"


def test_a_call_on_your_entry_replays(mine, chat_tokenizer, monkeypatch, capsys):
    fetch(mine, "my-tiny")
    (mine / "local" / "profiles").mkdir()
    (mine / "local" / "profiles" / "mine.toml").write_text(
        CHAT_PROFILE.replace('"tiny"', '"my-tiny"'), encoding="utf-8"
    )
    backend = FakeBackend([" A."], tokenizer=chat_tokenizer)
    session = Session(mine, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    conversation = session.new_conversation("mine")
    call_id = session.say("Hi").record["call_id"]
    session.close()
    again = FakeBackend([" A."], tokenizer=chat_tokenizer)
    monkeypatch.setattr(
        cli,
        "Session",
        lambda root, logs_dir, on_server=None: Session(
            root, backend=again, logs_dir=logs_dir, tokenizer_loader=lambda r, v: chat_tokenizer
        ),
    )
    assert cli.main(["--root", str(mine), "replay", str(conversation.log.path), call_id]) == 0
    assert again.started == ["my-tiny"]


def test_a_name_in_both_files_is_refused_naming_both(mine, capsys):
    (mine / LOCAL_TOML).write_text(MODELS, encoding="utf-8")  # tiny, as shipped
    with pytest.raises(ConfigError, match=r"'tiny' is in both models\.toml and local/models\.toml"):
        Session(mine)  # a chat's start, and clarity ui's
    assert cli.main(["--root", str(mine), "chat", "--profile", "chat"]) == 1
    assert "local/models.toml" in capsys.readouterr().err


def test_a_bad_entry_of_yours_is_refused_with_its_path(mine):
    (mine / LOCAL_TOML).write_text('[broken]\nroute = "magnet"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"local/models\.toml \[broken\]: route must be"):
        load_models(mine)


def test_without_local_models_toml_nothing_changes(fetched_root):
    assert set(load_models(fetched_root)) == {"tiny"}
    assert set(read_lock(fetched_root)["models"]) == {"tiny"}


# --- sentences that name one entry's files name your entry's own (S9) ----------------------


def edit_local_lock(root: Path, name: str, change) -> None:
    path = root / LOCAL_LOCK
    lock = json.loads(path.read_text(encoding="utf-8"))
    change(lock["models"][name])
    path.write_text(json.dumps(lock), encoding="utf-8")


def test_lock_sentences_name_your_files(mine):
    fetch(mine, "my-tiny")
    fetch(mine, "my-ready")
    model = loaded(mine, "my-tiny")
    edit_local_lock(mine, "my-tiny", lambda e: e.update(repo="test/other"))
    with pytest.raises(ConfigError, match=r"local/models\.lock has 'my-tiny' with repo = "
                       r"'test/other', but local/models\.toml says"):  # fmt: skip
        lock_entry(mine, model)
    edit_local_lock(mine, "my-tiny", lambda e: e.update(repo="test/tiny", recommended=[]))
    with pytest.raises(ConfigError, match=r"local/models\.lock's recommended settings"):
        lock_entry(mine, model)
    edit_local_lock(mine, "my-ready", lambda e: e["gguf"].update(sha256="1" * 64))
    with pytest.raises(ConfigError, match=r"local/models\.lock has 'my-ready' with a different "
                       r"sha256 than local/models\.toml"):  # fmt: skip
        lock_entry(mine, loaded(mine, "my-ready"))
    assert verify(mine, "my-ready")[1].endswith(f"!= local/models.lock {'1' * 64}")
    edit_local_lock(mine, "my-ready", lambda e: e["gguf"].update(size=1))
    with pytest.raises(ConfigError, match=r"differs from local/models\.lock"):
        check_weights_quick(mine, read_lock(mine)["models"]["my-ready"])


def test_verify_and_fetch_sentences_name_your_files(mine):
    assert verify(mine, "my-tiny") == (False, "my-tiny: not in local/models.lock")
    (mine / LOCAL_TOML).write_text(MINE_GGUF.replace(WEIGHTS_SHA256, "2" * 64), "utf-8")
    with pytest.raises(ConfigError, match=r"local/models\.toml says 2{64}\. Not recorded in "
                       r"local/models\.lock\."):  # fmt: skip
        fetch(mine, "my-ready")


def test_use_recommended_names_your_lock(mine, chat_tokenizer):
    fetch(mine, "my-tiny")
    edit_local_lock(mine, "my-tiny", lambda e: e.pop("recommended"))
    (mine / "local" / "profiles").mkdir()
    (mine / "local" / "profiles" / "mine.toml").write_text(
        "use_recommended = true\n" + CHAT_PROFILE.replace('"tiny"', '"my-tiny"'), "utf-8"
    )
    session = Session(mine, backend=FakeBackend(), tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(ConfigError, match=r"local/models\.lock has no recommended settings"):
        session.new_conversation("mine")


def test_a_name_in_neither_file_names_both(mine, chat_tokenizer):
    with pytest.raises(ConfigError, match=r"isn't in models\.toml or local/models\.toml"):
        cli.cmd_fetch(mine, "nowhere")
    (mine / "profiles" / "lost.toml").write_text(
        CHAT_PROFILE.replace('"tiny"', '"nowhere"'), "utf-8"
    )
    session = Session(mine, backend=FakeBackend(), tokenizer_loader=lambda r, v: chat_tokenizer)
    with pytest.raises(ConfigError, match=r"isn't in models\.toml or local/models\.toml"):
        session.new_conversation("lost")
    (listed,) = [p for p in list_profiles(mine) if p["name"] == "lost"]
    assert "isn't in models.toml or local/models.toml" in listed["error"]
