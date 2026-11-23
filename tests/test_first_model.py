"""v0.4.2 A4: the first model ships (decision 8; Q6a, Q6b, Q6c answered by default): Gemma 4
E4B from ggml-org, Q4_0, through the `gguf` route, with a profile that turns thinking off,
setup's first model. A0's gate (tests/test_gate_e4b.py) checked it on the real server at
the pin; these check the shipped files offline, and that a chat on its profile resumes,
exports and replays Identical (T3), on FakeBackend. Made-up text."""

from __future__ import annotations

import json
import shutil

from clarity_chat import setup as clarity_setup
from clarity_chat.profiles import load_defaults, load_models, load_profile
from tests.conftest import REPO

E4B = "gemma-4-e4b"


def test_the_entry_is_the_gate_s_pins():
    models = load_models(REPO)
    assert E4B in models, "no gemma-4-e4b in models.toml"
    entry = models[E4B]
    assert (entry.route, entry.repo, entry.revision) == (
        "gguf",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "8903cb408c7df39cde2ed15fac397e6d7be92951",
    )
    assert (entry.file, entry.sha256) == (
        "gemma-4-E4B-it-Q4_0.gguf",
        "a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2",
    )
    assert (entry.tokenizer_repo, entry.tokenizer_revision) == (
        "google/gemma-4-E4B-it",
        "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
    )


def test_its_profile_turns_thinking_off_with_the_model_s_settings():
    assert (REPO / "profiles" / f"{E4B}.toml").is_file(), "no profiles/gemma-4-e4b.toml"
    profile = load_profile(REPO, E4B, load_defaults(REPO))
    assert profile.model == E4B and profile.mode == "chat"
    assert profile.template_kwargs == {"enable_thinking": False}
    assert profile.use_recommended is True


def test_setup_fetches_it_first_then_bge_m3():
    assert clarity_setup.SETUP_MODELS == (E4B, "bge-m3")
    download, keep = clarity_setup.SIZES[E4B]
    assert download >= 4_590_807_392 and keep >= download  # the Q4_0 file's own size


def test_a_chat_on_its_profile_resumes_exports_and_replays(tmp_path, chat_tokenizer):
    """T3 for the first model's profile, on FakeBackend with the tests' chat tokenizer (the
    shipped profile's settings and kwargs; no Gemma file or template)."""
    from clarity_chat.export import export_log
    from clarity_chat.models import write_lock_entry
    from clarity_chat.session import Session
    from tests.conftest import MODELS
    from tests.fakes import FakeBackend

    for name in ("defaults.toml", "server.toml"):
        shutil.copy(REPO / name, tmp_path / name)
    shipped = (REPO / "models.toml").read_text(encoding="utf-8")
    assert f"[{E4B}]" in shipped, "no gemma-4-e4b in models.toml"
    block = shipped[shipped.index(f"[{E4B}]") :].split("\n\n", 1)[0]
    (tmp_path / "models.toml").write_text(MODELS + "\n" + block + "\n", encoding="utf-8")
    (tmp_path / "profiles").mkdir()
    shutil.copy(REPO / "profiles" / f"{E4B}.toml", tmp_path / "profiles" / f"{E4B}.toml")
    model = load_models(tmp_path)[E4B]
    weights = tmp_path / "models" / E4B / model.file
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"GGUF made up")
    stat = weights.stat()
    write_lock_entry(tmp_path, E4B, {
        **model.to_json(),
        "gguf": {"path": weights.relative_to(tmp_path).as_posix(), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns, "sha256": model.sha256},
        "recommended": {"source": {"file": "generation_config.json", "repo": model.tokenizer_repo,
                                   "revision": model.tokenizer_revision}, "values": {}},
    })  # fmt: skip
    backend = FakeBackend([" A made-up bird.", " A made-up bird."], tokenizer=chat_tokenizer)
    session = Session(tmp_path, backend=backend, tokenizer_loader=lambda r, v: chat_tokenizer)
    session.new_conversation(E4B)
    call = session.say("Name a made-up bird.").record
    assert session.conversation is not None
    log = session.conversation.log.path
    session.new_conversation(E4B)
    record = session.resume(log)
    assert record["render"] is None or record["render"]["identical"], record["render"]
    assert export_log(session, log)["path"]
    again = FakeBackend([call["output_text"]], tokenizer=chat_tokenizer)
    fresh = Session(tmp_path, backend=again, tokenizer_loader=lambda r, v: chat_tokenizer)
    assert fresh.replay(log, call["call_id"])["identical"]
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[0])["profile"]["name"] == E4B
