"""v0.4.7 R: `route = "safetensors"` fetches a model's official weights without converting
them, for the Transformers backend (decision 9; states 16 to 23, 25 and 26). The download is
a fake over a made-up folder, as the other fetch tests use; the chat at the end runs a tiny
Llama on the CPU. Made-up text and bytes only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from clarity_chat import cli, models
from clarity_chat.models import (
    LOCK_FILE,
    check_weights_quick,
    fetched_size,
    list_profiles,
    lock_entry,
    read_lock_file,
    recheck_weights,
    source_weights,
    verify,
    weights_path,
    write_lock_entry,
    write_lock_file,
)
from clarity_chat.profiles import (
    LOCAL_LOCK,
    ConfigError,
    ModelEntry,
    load_models,
    load_server_config,
    read_models,
)
from clarity_chat.render import TOKENIZER_FILES
from clarity_chat.replay import model_for_call
from tests.conftest import CHAT_PROFILE, MODELS, WORD_SEARCH, fake_fetch
from tests.test_local_lock import refresh
from tests.test_maker_render import HF_CHAT

REVISION = "3123456789abcdef0123456789abcdef01234567"
TOK_REVISION = "4123456789abcdef0123456789abcdef01234567"
OFFICIAL = f"""\
[official]
route = "safetensors"
repo = "test/official"
revision = "{REVISION}"
tokenizer_repo = "test/official"
tokenizer_revision = "{REVISION}"
"""
# What fetch records of a safetensors-route entry, and nothing else (no gguf, llama_cpp,
# converter or weights_sha256).
KEYS = {
    "name",
    "route",
    "repo",
    "revision",
    "tokenizer_repo",
    "tokenizer_revision",
    "source_dir",
    "tokenizer_dir",
    "source_files",
    "recommended",
    "fetched_at",
}
# The made-up repo: the files SOURCE_PATTERNS takes, one of them in a subfolder ...
FILES = {
    "config.json": b'{"model_type": "llama"}',
    "generation_config.json": json.dumps({"temperature": 0.5}).encode(),
    "model.safetensors": b"made-up safetensors bytes",
    "sub/extra.json": b"{}",
    "tokenizer.json": b'{"made": "up"}',
}
# ... and two it doesn't, which another tool's download of the same commit might leave there.
STRAYS = {"README.md": b"# Made up", "pytorch_model.bin": b"made-up pickle"}
TOKENIZER = {"tokenizer.json": b'{"another": "tokenizer"}', "tokenizer_config.json": b"{}"}

STATE_17 = (
    "test/official at 3123456789ab has no model.safetensors or model.safetensors.index.json, "
    "which the Transformers backend loads; the safetensors route fetches no pickle (.bin or "
    ".pt). Not recorded in local/models.lock."
)
STATE_19 = (
    "Profile official-chat: official is fetched as official weights only (route = "
    '"safetensors"), and llama.cpp needs a GGUF. Set backend = "transformers" in the profile, '
    "or use a model that has a GGUF."
)
STATE_20 = (
    'official is fetched as official weights only (route = "safetensors"), and llama.cpp '
    "needs a GGUF."
)
STATE_26 = "{lock}'s entry for official has no usable list of its files (source_files)."


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def listed(files: dict[str, bytes]) -> list[dict]:
    """`source_files` as fetch records it: by repo-relative name, sorted."""
    return [{"file": n, "size": len(d), "sha256": sha256(d)} for n, d in sorted(files.items())]


def snapshot(folder: Path, files: dict[str, bytes], strays: dict[str, bytes]) -> Path:
    for name, data in {**files, **strays}.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return folder


def failing(what: str):
    def fail(*args, **kwargs):
        pytest.fail(f"the safetensors route {what}")

    return fail


@pytest.fixture
def official(root, tmp_path, monkeypatch):
    """A root whose models.toml has tiny and `official`, and a fake download: `official`'s
    repo is the made-up folder FILES and STRAYS, `test/tok` a tokenizer. A fetch that reads
    the llama.cpp checkout, makes the converter's environment or downloads one file fails."""
    (root / "models.toml").write_text(MODELS + "\n" + OFFICIAL, encoding="utf-8")
    cache = tmp_path / "hf-cache"
    folders = {
        "test/official": snapshot(cache / "official", FILES, STRAYS),
        "test/tok": snapshot(cache / "tok", TOKENIZER, {}),
    }
    asked: list[tuple[str, str, list[str]]] = []

    def download(repo, revision, patterns):
        asked.append((repo, revision, patterns))
        return folders[repo]

    monkeypatch.setattr(models, "_download", download)
    monkeypatch.setattr(models, "_download_file", failing("downloaded one file"))
    monkeypatch.setattr(models, "git_state", failing("read the llama.cpp checkout"))
    monkeypatch.setattr(models, "_make_converter_environment", failing("made an environment"))
    return SimpleNamespace(root=root, folder=folders["test/official"], asked=asked)


def fetch(root: Path, name: str = "official") -> dict:
    return models.fetch(root, load_models(root)[name], load_server_config(root))


def local_lock(root: Path) -> dict:
    return read_lock_file(root / LOCAL_LOCK)["models"]


def recorded(root: Path, folder: Path, files: dict[str, bytes] = FILES, **change) -> dict:
    """A safetensors-route lock entry for `official`, as fetch would write it, written to
    local/models.lock without a fetch."""
    entry = {
        "name": "official",
        "route": "safetensors",
        "repo": "test/official",
        "revision": REVISION,
        "tokenizer_repo": "test/official",
        "tokenizer_revision": REVISION,
        "source_dir": str(folder),
        "tokenizer_dir": str(folder),
        "source_files": listed(files),
        "recommended": {"source": {}, "values": {}, "not_applied": {}, "notes": []},
        "fetched_at": "2026-11-26T00:00:00Z",
        **change,
    }
    write_lock_entry(root, "official", entry)
    return entry


# --- models.toml (state 23) -----------------------------------------------------------------


def test_the_entry_loads_from_models_toml_and_from_local_models_toml(root):
    (root / "models.toml").write_text(MODELS + "\n" + OFFICIAL, encoding="utf-8")
    (root / "local").mkdir()
    (root / "local" / "models.toml").write_text(
        OFFICIAL.replace("[official]", "[mine]"), encoding="utf-8"
    )
    entries = load_models(root)
    assert (entries["official"].route, entries["mine"].route) == ("safetensors", "safetensors")
    assert entries["official"].to_json() == {
        "name": "official",
        "route": "safetensors",
        "repo": "test/official",
        "revision": REVISION,
        "tokenizer_repo": "test/official",
        "tokenizer_revision": REVISION,
    }


@pytest.mark.parametrize(
    "line",
    [
        'file = "official.gguf"',
        f'sha256 = "{"0" * 64}"',
        'weights = "pytorch_model.bin"',
        f'weights_sha256 = "{"0" * 64}"',
    ],
)
def test_a_safetensors_entry_takes_no_gguf_or_pickle_key(root, line):
    """So no such entry can ever name a pickle: today's sentence."""
    (root / "models.toml").write_text(OFFICIAL + line + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[official\]: needs exactly the keys"):
        read_models(root / "models.toml")


def test_another_route_s_sentence_names_the_three_routes(root):
    (root / "models.toml").write_text(OFFICIAL.replace("safetensors", "pickle"), "utf-8")
    with pytest.raises(
        ConfigError, match=r'route must be "convert", "gguf" or "safetensors", got \'pickle\'\.'
    ):
        read_models(root / "models.toml")


# --- the fetch (states 16 to 18) ------------------------------------------------------------


def test_the_fetch_downloads_the_source_patterns_and_converts_nothing(official, monkeypatch):
    root = official.root
    write_lock_file(root / LOCK_FILE, {"schema_version": 1, "models": {}})
    before = (root / LOCK_FILE).read_bytes()
    monkeypatch.setattr(subprocess, "run", failing("ran a subprocess"))
    monkeypatch.setattr(subprocess, "Popen", failing("ran a subprocess"))
    entry = fetch(root)
    assert official.asked == [("test/official", REVISION, models.SOURCE_PATTERNS)]
    assert not (root / "models").exists(), "the fetch wrote under models/"
    assert local_lock(root)["official"] == entry
    assert (root / LOCK_FILE).read_bytes() == before


def test_the_lock_entry_records_each_file_from_the_repo(official):
    """By repo-relative name, so sub/extra.json is there; README.md and the stray pickle
    aren't, since SOURCE_PATTERNS doesn't take them."""
    root = official.root
    entry = fetch(root)
    assert set(entry) == KEYS
    assert entry["route"] == "safetensors"
    assert entry["source_files"] == listed(FILES)
    assert entry["recommended"]["values"] == {"temperature": 0.5}
    assert lock_entry(root, load_models(root)["official"]) == entry


def test_a_tokenizer_from_another_repo_is_downloaded_as_the_convert_route_does(official):
    root = official.root
    (root / "models.toml").write_text(
        MODELS
        + "\n"
        + OFFICIAL.replace(
            f'tokenizer_repo = "test/official"\ntokenizer_revision = "{REVISION}"',
            f'tokenizer_repo = "test/tok"\ntokenizer_revision = "{TOK_REVISION}"',
        ),
        encoding="utf-8",
    )
    entry = fetch(root)
    assert official.asked[1] == ("test/tok", TOK_REVISION, TOKENIZER_FILES)
    assert entry["source_files"] == listed(FILES)  # nothing of the tokenizer's
    assert entry["tokenizer_dir"] != entry["source_dir"]


def test_state_17_a_download_with_neither_file_the_load_reads_is_refused(official):
    """adapter_model.safetensors isn't the model, and the pickle is never loaded."""
    root = official.root
    os.remove(official.folder / "model.safetensors")
    (official.folder / "adapter_model.safetensors").write_bytes(b"made-up adapter")
    with pytest.raises(ConfigError) as refused:
        fetch(root)
    assert str(refused.value) == STATE_17
    assert "official" not in local_lock(root)


def test_state_17_a_sharded_download_with_its_index_is_recorded(official):
    root = official.root
    os.remove(official.folder / "model.safetensors")
    shards = {
        "model.safetensors.index.json": b'{"weight_map": {}}',
        "model-00001-of-00002.safetensors": b"made-up shard one",
        "model-00002-of-00002.safetensors": b"made-up shard two",
    }
    snapshot(official.folder, shards, {})
    files = {k: v for k, v in FILES.items() if k != "model.safetensors"} | shards
    assert fetch(root)["source_files"] == listed(files)


def shipped(root: Path, entry: dict) -> None:
    """`entry` as the shipped models.lock's, and local/models.lock emptied."""
    write_lock_file(root / LOCK_FILE, {"schema_version": 1, "models": {"official": entry}})
    os.remove(root / LOCAL_LOCK)


def test_state_18_a_file_unlike_the_shipped_lock_s_is_refused(official):
    root = official.root
    entry = fetch(root)
    for item in entry["source_files"]:
        if item["file"] == "sub/extra.json":
            item["sha256"] = "1" * 64
    shipped(root, entry)
    with pytest.raises(ConfigError) as refused:
        fetch(root)
    digest = sha256(FILES["sub/extra.json"])
    assert str(refused.value) == (
        f"test/official/sub/extra.json at 3123456789ab has sha256 {digest}, but models.lock "
        f"says {'1' * 64}. Not recorded in local/models.lock."
    )
    assert "official" not in local_lock(root)


def test_state_18_a_file_the_shipped_lock_lists_and_the_download_lacks_is_refused(official):
    root = official.root
    entry = fetch(root)
    entry["source_files"].append({"file": "sub/gone.json", "size": 2, "sha256": "2" * 64})
    shipped(root, entry)
    with pytest.raises(ConfigError) as refused:
        fetch(root)
    assert str(refused.value) == (
        "test/official/sub/gone.json at 3123456789ab is missing, but models.lock says "
        f"{'2' * 64}. Not recorded in local/models.lock."
    )
    assert "official" not in local_lock(root)


def test_state_18_a_download_like_the_shipped_lock_s_is_recorded(official):
    root = official.root
    entry = fetch(root)
    shipped(root, entry)
    again = fetch(root)
    assert again["source_files"] == entry["source_files"]
    assert "official" in local_lock(root)


# Each malformed shape of a lock entry's source_files (state 26), as a change to the entry.
MALFORMED = {
    "no source_files": lambda e: e.pop("source_files"),
    "not a list": lambda e: e.update(source_files="model.safetensors"),
    "empty": lambda e: e.update(source_files=[]),
    "an item without size": lambda e: e["source_files"][0].pop("size"),
    "an item without file": lambda e: e["source_files"][0].pop("file"),
    "a sha256 that isn't 64 hex": lambda e: e["source_files"][0].update(sha256="ABC"),
    "neither model.safetensors nor its index": lambda e: e.update(
        source_files=[f for f in e["source_files"] if f["file"] != "model.safetensors"]
    ),
}


@pytest.mark.parametrize("change", MALFORMED.values(), ids=MALFORMED.keys())
def test_state_26_from_the_fetch_s_check_against_the_shipped_lock(official, change):
    root = official.root
    entry = fetch(root)
    change(entry)
    shipped(root, entry)
    with pytest.raises(ConfigError) as refused:
        fetch(root)
    assert str(refused.value).startswith(STATE_26.format(lock="models.lock"))
    assert "official" not in local_lock(root)


def test_a_route_changed_to_safetensors_is_a_route_change(root):
    fake_fetch(root)  # tiny, fetched on the convert route
    (root / "models.toml").write_text(
        MODELS.replace('route = "convert"', 'route = "safetensors"'), encoding="utf-8"
    )
    with pytest.raises(
        ConfigError,
        match=r"with route = 'convert', but models\.toml says 'safetensors'\..*fetch tiny",
    ):
        lock_entry(root, load_models(root)["tiny"])


def test_clarity_fetch_says_what_it_fetched(official, capsys):
    status = cli.main(["--root", str(official.root), "fetch", "official"])
    out = capsys.readouterr()
    assert status == 0, out.err
    assert "Downloading test/official@3123456789ab …" in out.out
    assert "Hashing 5 files …" in out.out
    assert re.search(
        r"Fetched official: the official weights, 5 files in .+\. No GGUF: it runs on the "
        r"Transformers backend only\.\n\Z",
        out.out,
    ), out.out


# --- the Transformers side (states 22, 25 and 26) -------------------------------------------


def test_source_weights_gives_the_folder(official):
    entry = recorded(official.root, official.folder)
    assert source_weights(official.root, entry) == official.folder


def test_state_22_a_ready_made_gguf_names_both_routes(fetched_root):
    entry = {**read_lock_file(fetched_root / LOCAL_LOCK)["models"]["tiny"], "route": "gguf"}
    with pytest.raises(ConfigError) as refused:
        source_weights(fetched_root, entry)
    assert str(refused.value) == (
        "tiny was fetched as a ready-made GGUF; the Transformers backend needs the official "
        'weights (route = "convert" or "safetensors").'
    )


def test_state_25_a_listed_file_gone_is_refused(official):
    entry = recorded(official.root, official.folder)
    os.remove(official.folder / "sub" / "extra.json")
    with pytest.raises(ConfigError) as refused:
        source_weights(official.root, entry)
    message = str(refused.value)
    assert message.startswith(f"{official.folder / 'sub' / 'extra.json'} is missing. ")
    assert "fetch official" in message


def test_state_25_a_listed_file_of_another_size_is_refused(official):
    entry = recorded(official.root, official.folder)
    (official.folder / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ConfigError) as refused:
        source_weights(official.root, entry)
    size = len(FILES["model.safetensors"])
    assert str(refused.value).startswith(
        f"{official.folder / 'model.safetensors'} changed since it was fetched: it has 5 "
        f"bytes, and local/models.lock says {size}. "
    )


def test_state_25_other_bytes_of_the_same_size_pass_the_start_and_fail_verify(official):
    """A start compares sizes only; `clarity verify` re-hashes."""
    entry = recorded(official.root, official.folder)
    (official.folder / "sub" / "extra.json").write_bytes(b"[]")
    assert source_weights(official.root, entry) == official.folder
    ok, message = verify(official.root, "official")
    assert not ok
    assert message == (
        f"official: sub/extra.json sha256 {sha256(b'[]')} != local/models.lock "
        f"{sha256(FILES['sub/extra.json'])}"
    )


@pytest.mark.parametrize("change", MALFORMED.values(), ids=MALFORMED.keys())
def test_state_26_from_source_weights_and_verify(official, change):
    entry = recorded(official.root, official.folder)
    change(entry)
    write_lock_entry(official.root, "official", entry)
    with pytest.raises(ConfigError) as refused:
        source_weights(official.root, entry)
    assert str(refused.value).startswith(STATE_26.format(lock="local/models.lock"))
    ok, message = verify(official.root, "official")
    assert not ok and STATE_26.format(lock="local/models.lock") in message


def test_state_26_not_22_for_an_entry_with_no_source_dir(official):
    entry = recorded(official.root, official.folder)
    del entry["source_dir"]
    with pytest.raises(ConfigError) as refused:
        source_weights(official.root, entry)
    assert str(refused.value).startswith(STATE_26.format(lock="local/models.lock"))


def test_the_transformers_load_takes_safetensors_only(tmp_path, monkeypatch):
    """M1: with use_safetensors=True, a folder with no model.safetensors or its index can't
    fall back to a pytorch_model.bin that was never listed or hashed."""
    pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from clarity_chat.backends import transformers_hf

    loads: list[dict] = []

    class Loaded(Exception):
        pass

    def from_pretrained(cls, folder, **kwargs):
        if cls is AutoModelForCausalLM:
            loads.append(kwargs)
            raise Loaded
        return object()

    monkeypatch.setattr(transformers_hf, "from_pretrained", from_pretrained)
    monkeypatch.setattr(transformers_hf, "end_tokens", lambda weights: ([1], "made up"))
    model = ModelEntry("official", "safetensors", "test/official", REVISION, "t/t", REVISION)
    with pytest.raises(Loaded):
        transformers_hf.TransformersBackend("cpu", 64).start(model, tmp_path, 64)
    assert loads[0].get("use_safetensors") is True, loads[0]


# --- what reads a GGUF (state 20) -----------------------------------------------------------

READERS = {
    "weights_path": weights_path,
    "check_weights_quick": check_weights_quick,
    "fetched_size": fetched_size,
    "recheck_weights": lambda root, entry: recheck_weights(root, entry, "The model file's"),
}


@pytest.mark.parametrize("reader", READERS.values(), ids=READERS.keys())
def test_state_20_each_gguf_reader_refuses_in_words(official, reader):
    entry = recorded(official.root, official.folder)
    with pytest.raises(ConfigError) as refused:
        reader(official.root, entry)
    assert str(refused.value) == STATE_20


# --- profiles (state 19) --------------------------------------------------------------------


def official_profiles(root: Path) -> None:
    """official-chat on llama.cpp, official-hf on Transformers."""
    chat = CHAT_PROFILE.replace('model = "tiny"', 'model = "official"') + WORD_SEARCH
    (root / "profiles" / "official-chat.toml").write_text(chat, encoding="utf-8", newline="\n")
    hf = HF_CHAT.format(extra="").replace('model = "tiny"', 'model = "official"')
    (root / "profiles" / "official-hf.toml").write_text(hf, encoding="utf-8", newline="\n")


@pytest.mark.parametrize("fetched", [True, False], ids=["fetched", "not fetched"])
def test_state_19_a_llama_cpp_profile_is_refused_before_it_is_fetched(
    official, tiny_tokenizer, fetched
):
    """Before lock_entry: nobody downloads 16 GB to meet this refusal afterwards."""
    from clarity_chat.session import Session
    from tests.fakes import FakeBackend

    root = official.root
    official_profiles(root)
    if fetched:
        recorded(root, official.folder)
    backend = FakeBackend(tokenizer=tiny_tokenizer)
    session = Session(root, backend=backend, tokenizer_loader=lambda r, v: tiny_tokenizer)
    try:
        with pytest.raises(ConfigError) as refused:
            session.prepare("official-chat")
        assert str(refused.value) == STATE_19
        assert backend.started == []
    finally:
        session.close()


def test_state_19_in_the_profile_list(official):
    root = official.root
    official_profiles(root)
    recorded(root, official.folder)
    listed_profiles = {p["name"]: p for p in list_profiles(root)}
    assert listed_profiles["official-chat"]["error"] == STATE_19
    assert listed_profiles["official-chat"]["fetched"] is False
    hf = listed_profiles["official-hf"]
    assert (hf["error"], hf["fetched"], hf["model"]) == (None, True, "official")


def test_state_25_the_profile_list_says_not_fetched(official):
    root = official.root
    official_profiles(root)
    recorded(root, official.folder)
    os.remove(official.folder / "model.safetensors")
    hf = {p["name"]: p for p in list_profiles(root)}["official-hf"]
    assert (hf["error"], hf["fetched"]) == (None, False)


# --- clarity verify (state 21) --------------------------------------------------------------


def test_verify_re_hashes_each_listed_file(official):
    recorded(official.root, official.folder)
    ok, message = verify(official.root, "official")
    assert ok, message
    assert message == f"official: sha256 matches for 5 files in {official.folder}"


def test_verify_names_a_missing_file(official):
    recorded(official.root, official.folder)
    os.remove(official.folder / "tokenizer.json")
    assert verify(official.root, "official") == (
        False,
        f"official: {official.folder / 'tokenizer.json'} is missing",
    )


def test_verify_with_no_names_goes_on_to_the_next_model(official, capsys):
    fake_fetch(official.root)  # tiny, after official in sorted order
    recorded(official.root, official.folder)
    os.remove(official.folder / "tokenizer.json")
    assert cli.main(["--root", str(official.root), "verify"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  official: " in out and "  tiny: " in out


# --- clarity replay picks an entry (decision 9) ---------------------------------------------

TINY = {
    "repo": "test/tiny",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "tokenizer_repo": "test/tiny",
    "tokenizer_revision": "0123456789abcdef0123456789abcdef01234567",
}
CONVERTED = {**TINY, "route": "convert", "source_dir": "s", "gguf": {"sha256": "0" * 64}}
SAFETENSORS = {**TINY, "route": "safetensors", "source_dir": "s", "source_files": []}


def test_a_transformers_call_replays_on_either_route_with_its_weights():
    """Both routes download the same folder for the same repo and commit."""
    on_safetensors = {"name": "gone", "route": "safetensors", **TINY}
    assert model_for_call(on_safetensors, "transformers", {"conv": CONVERTED}) == "conv"
    on_convert = {"name": "gone", "route": "convert", **TINY, "gguf_sha256": "0" * 64}
    assert model_for_call(on_convert, "transformers", {"safe": SAFETENSORS}) == "safe"


def test_a_llama_cpp_call_never_picks_a_safetensors_entry():
    call = {"name": "gone", "route": "convert", **TINY, "gguf_sha256": "0" * 64}
    assert model_for_call(call, "llama.cpp", {"safe": SAFETENSORS}) is None


# --- scripts/refresh_lock.py ----------------------------------------------------------------

HUB = f"~/.cache/huggingface/hub/models--test--official/snapshots/{REVISION}"


def test_refresh_lock_reads_each_listed_name_and_no_gguf(official):
    entry = recorded(official.root, official.folder, source_dir=HUB, tokenizer_dir=HUB)
    assert set(refresh().paths(entry)) == {HUB, *FILES}


def test_refresh_lock_copies_a_safetensors_entry(official, capsys):
    root = official.root
    entry = recorded(root, official.folder, source_dir=HUB, tokenizer_dir=HUB)
    assert refresh().main(["--root", str(root), "official"]) == 0, capsys.readouterr().err
    assert read_lock_file(root / LOCK_FILE)["models"]["official"] == entry


@pytest.mark.parametrize("where", ["source_dir", "source_files"])
def test_refresh_lock_refuses_an_absolute_path(official, capsys, where):
    """A source_dir outside the home folder (today's refusal), or an absolute name in a
    hand-edited source_files (C6): this computer's layout stays out of the shipped lock."""
    root = official.root
    files = {**FILES, "/made/up/x.json": b"{}"} if where == "source_files" else FILES
    folder = HUB if where == "source_files" else "/made/up/hub"
    recorded(root, official.folder, files, source_dir=folder, tokenizer_dir=HUB)
    assert refresh().main(["--root", str(root), "official"]) == 1
    assert "outside Clarity's folder and the home folder" in capsys.readouterr().err
    assert not (root / LOCK_FILE).exists()


# --- a chat on a model fetched this way -----------------------------------------------------


@pytest.fixture
def tiny_official(official, chat_tokenizer, monkeypatch):
    """`official` fetched as a tiny Llama with real weights (4 blocks, width 8, the chat
    tokenizer), and official-hf, a chat profile on it on the CPU."""
    torch = pytest.importorskip("torch")
    import shutil

    import transformers

    shutil.rmtree(official.folder)
    config = transformers.LlamaConfig(
        vocab_size=258,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=257,
        bos_token_id=256,
    )
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(config).save_pretrained(official.folder)
    chat_tokenizer.tok.save_pretrained(official.folder)
    root = official.root
    fetch(root)
    hf = HF_CHAT.format(extra="ctx_size = 256").replace('model = "tiny"', 'model = "official"')
    (root / "profiles" / "official-hf.toml").write_text(hf, encoding="utf-8", newline="\n")
    return root


def test_a_chat_on_it_logs_no_gguf_and_replays_resumes_and_exports(
    tiny_official, chat_tokenizer, monkeypatch, capsys, tmp_path
):
    from clarity_chat.export import export_log
    from clarity_chat.session import Session

    root = tiny_official

    def session_at(root, logs_dir=None, on_server=None):
        return Session(root, logs_dir=logs_dir, tokenizer_loader=lambda r, v: chat_tokenizer)

    session = session_at(root)
    try:
        session.new_conversation("official-hf")
        record = session.say("Boil an egg?").record
        assert session.conversation is not None
        log = session.conversation.log.path
        assert record["model"]["route"] == "safetensors"
        assert "gguf_sha256" not in record["model"] and "gguf_path" not in record["model"]
        assert session.replay(log, record["call_id"])["identical"]
    finally:
        session.close()

    monkeypatch.setattr(cli, "Session", session_at)
    status = cli.main(["--root", str(root), "replay", str(log), record["call_id"]])
    out = capsys.readouterr()
    assert status == 0 and "Identical" in out.out, out.err

    session = session_at(root)
    try:
        resumed = session.resume(log)
        assert (resumed["model_entry"], resumed["differences"]) == ([], [])
        export_log(session, log, tmp_path / "export.jsonl")
    finally:
        session.close()
    rows = (tmp_path / "export.jsonl").read_text(encoding="utf-8").splitlines()
    for row in map(json.loads, rows[1:]):
        assert (row["model"]["gguf_sha256"], row["model"]["gguf_path"]) == (None, None)
