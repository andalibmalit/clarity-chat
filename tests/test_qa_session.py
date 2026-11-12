"""The QA driver's limits. It runs Clarity outside the sandbox, on the GPU, for a QA agent,
so it must keep it to its scratch root and to SmolLM2 whatever the agent's steps say."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_session", REPO / "scripts" / "qa_session.py")
assert spec and spec.loader
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

SMOL = "smollm2-135m-instruct"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A stand-in checkout: the real config files and three profiles, and two fetched
    models with tiny weights files."""
    fake = tmp_path / "repo"
    (fake / "profiles").mkdir(parents=True)
    for name in ("defaults.toml", "server.toml", "models.toml"):
        shutil.copy(REPO / name, fake / name)
    for name in ("smol-chat", "smol-text", "1a-stock"):
        shutil.copy(REPO / "profiles" / f"{name}.toml", fake / "profiles" / f"{name}.toml")
    (fake / qa.QA_VECTOR).parent.mkdir(parents=True)
    shutil.copy(REPO / qa.QA_VECTOR, fake / qa.QA_VECTOR)
    models = {}
    for name in (SMOL, "olmo-3-7b-instruct"):
        weights = fake / "models" / name / f"{name}-bf16.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"GGUF " + name.encode())
        os.utime(weights, ns=(1_000_000_100, 1_000_000_100))  # Windows keeps 100 ns steps
        gguf = {
            "path": f"models/{name}/{name}-bf16.gguf",
            "size": weights.stat().st_size,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        }
        models[name] = {"name": name, "gguf": {**gguf, "mtime_ns": 1_000_000_100}}
    lock = {"schema_version": 1, "models": models}
    (fake / "models.lock").write_text(json.dumps(lock), encoding="utf-8")
    return fake


@pytest.fixture
def root(tmp_path: Path, repo: Path) -> Path:
    scratch = tmp_path / "scratch"
    qa.prepare_root(scratch, repo, init=True)
    return scratch


# --- the root ------------------------------------------------------------------------


def test_the_root_holds_only_smollm2_and_its_profiles(root, repo):
    lock = json.loads((root / "models.lock").read_text(encoding="utf-8"))
    assert list(lock["models"]) == [SMOL]
    assert list(tomllib.loads((root / "models.toml").read_text(encoding="utf-8"))) == [SMOL]
    profiles = sorted(p.name for p in (root / "profiles").iterdir())
    assert profiles == [
        "smol-chat-hf.toml",
        "smol-chat.toml",
        "smol-steer-hf.toml",
        "smol-text.toml",
    ]
    hf = tomllib.loads((root / "profiles" / "smol-chat-hf.toml").read_text(encoding="utf-8"))
    assert (hf["backend"], hf["device"], hf["model"]) == ("transformers", "cpu", SMOL)
    # The root's own copy of the weights, with the time models.lock recorded: an agent can
    # rename over it (QA step 21) without touching the checkout's.
    weights = root / "models" / SMOL / f"{SMOL}-bf16.gguf"
    assert not (root / "models").is_symlink() and weights.stat().st_mtime_ns == 1_000_000_100
    assert not (root / "models" / "olmo-3-7b-instruct").exists()
    server = tomllib.loads((root / "server.toml").read_text(encoding="utf-8"))
    assert server["llama_cpp_dir"] == (repo / "vendor" / "llama.cpp").as_posix()


def test_each_run_rewrites_what_an_agent_could_have_changed(root, repo):
    (root / "profiles" / "smol-chat.toml").write_text('model = "olmo-3-7b-instruct"\n', "utf-8")
    (root / "profiles" / "1a-stock.toml").write_text("", encoding="utf-8")
    (root / "models.lock").write_text("{}", encoding="utf-8")
    qa.prepare_root(root, repo, init=False)
    assert (root / "profiles" / "smol-chat.toml").read_bytes() == (
        repo / "profiles" / "smol-chat.toml"
    ).read_bytes()
    assert not (root / "profiles" / "1a-stock.toml").exists()
    assert list(json.loads((root / "models.lock").read_text(encoding="utf-8"))["models"]) == [SMOL]


def test_server_toml_may_change_only_its_flags(root, repo):
    path = root / "server.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("[flags]", "[flags]\ngpu-layers = 99"), encoding="utf-8")
    qa.prepare_root(root, repo, init=False)  # QA step 18 edits the flags: allowed
    elsewhere = text.replace((repo / "vendor" / "llama.cpp").as_posix(), "/tmp/planted")
    path.write_text(elsewhere, encoding="utf-8")
    with pytest.raises(qa.Refused, match="llama_cpp_dir"):
        qa.prepare_root(root, repo, init=False)


def test_a_server_record_left_in_clarity_json_is_refused(root, repo):
    """Clarity stops the server a crashed run recorded there, outside the sandbox: the agent
    can write that file, so the driver won't run on one."""
    (root / "local").mkdir(exist_ok=True)  # --init made it, for the steering vector
    info = {"pid": 1, "server": {"pid": 1, "port": 1, "api_key": "k", "model_path": "m"}}
    (root / "local" / "clarity.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(qa.Refused, match="server"):
        qa.prepare_root(root, repo, init=False)


def test_an_old_root_with_links_is_rebuilt(tmp_path, repo):
    scratch = tmp_path / "old"
    scratch.mkdir()
    for name in ("profiles", "models"):
        (scratch / name).symlink_to(repo / name, target_is_directory=True)
    qa.prepare_root(scratch, repo, init=False)
    assert not (scratch / "profiles").is_symlink() and not (scratch / "models").is_symlink()
    assert (repo / "profiles" / "1a-stock.toml").exists()  # the link went, not its target


# --- Clarity's arguments -------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "--profile=/x/evil.toml"],
        ["chat", "--prof", "/x/evil.toml"],
        ["chat", "--profile", "../evil"],
        ["chat", "--resume=OUTSIDE/logs/a.jsonl"],
        ["chat", "--res", "OUTSIDE/logs/a.jsonl"],
        ["replay", "--logged-flags", "OUTSIDE/logs/a.jsonl", "c1"],
        ["replay", "OUTSIDE/logs/a.jsonl", "c1"],
        ["export", "OUTSIDE/logs/a.jsonl"],
        ["export", "ID", "--out", "x.jsonl"],
        ["export", "ID", "--ou", "/tmp/x.jsonl"],
        ["export", "ID", "--out=/tmp/x.jsonl"],
        ["fetch", "olmo-3-7b-instruct"],
        ["run", "--openings", "o.txt", "--profiles", "smol-chat"],
        ["mark", "OUTSIDE/logs/a.jsonl", "c1"],
        ["--root", "/elsewhere", "chat", "--profile", "smol-chat"],
    ],
)
def test_arguments_that_leave_the_root_are_refused(root, tmp_path, argv):
    argv = [a.replace("OUTSIDE", str(tmp_path / "repo")) for a in argv]
    with pytest.raises((qa.Refused, SystemExit)):
        qa.check_clarity_args(root, argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "--profile", "smol-chat"],
        ["chat", "--profile", "smol-chat-hf"],
        ["chat", "--resume", "20261111-120000-abcdef"],
        ["chat", "--json"],
        ["chat", "--json", "--profile", "no-such-profile"],
        ["chats"],
        ["replay", "ROOT/logs/a.jsonl", "c1", "--logged-flags"],
        ["export", "20261111-120000-abcdef"],
    ],
)
def test_arguments_inside_the_root_are_allowed(root, argv):
    qa.check_clarity_args(root, [a.replace("ROOT", str(root)) for a in argv])


# --- steps ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "json_mode"),
    [
        ("/resume\t/abs/logs/a.jsonl", False),
        ("/profile\t/x/evil.toml", False),
        ("/resume /abs/logs/a.jsonl", False),
        ("/profile ../evil", False),
        ("/export ../../elsewhere.jsonl", False),
        ("/abs/logs/a.jsonl\x01/resume ", False),  # Ctrl-A: the line editor's start of line
        ("\x1b[A", False),  # an escape sequence: up arrow, the previous line
        ("Hi /resume /abs", False),
        ('{"command": "resume", "args": "/abs/logs/a.jsonl"}', True),
        ('{"command": "profile", "args": "/x/evil.toml"}', True),
        ("<wait forever>", False),
        ("<ctrl-z>", False),
    ],
)
def test_steps_that_could_name_a_file_outside_the_root_are_refused(tmp_path, line, json_mode):
    steps = tmp_path / "steps.txt"
    steps.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps, json_mode)


def test_ordinary_steps_pass_as_written(tmp_path):
    lines = [
        "# a comment",
        "Hi",
        "/regen 3",
        "<ctrl-c>",
        "<wait 2>",
        "/profile smol-chat",
        "/resume 20261111-120000-abcdef",
        "/export",
        "/set logprobs 5",
        "//resume /not/a/command",
    ]
    steps = tmp_path / "steps.txt"
    steps.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert qa.parse_steps(steps, json_mode=False) == lines[1:]
    lines = ['{"command": "say", "args": "Hi"}', "not json", '{"command": "export"}']
    steps.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert qa.parse_steps(steps, json_mode=True) == lines


# --- links, the weights' hash, the working folder, timed Ctrl-C ------------------------


@pytest.mark.parametrize(
    "plant",
    [
        lambda root, outside: (root / "logs").symlink_to(outside, target_is_directory=True),
        lambda root, outside: (
            shutil.rmtree(root / "local")  # --init made it, for the steering vector
            or (root / "local").symlink_to(outside, target_is_directory=True)
        ),
        lambda root, outside: (
            (root / "defaults.toml").unlink()
            or (root / "defaults.toml").symlink_to(outside / "settings.json")
        ),
        lambda root, outside: (
            (root / "server.toml").unlink()
            or (root / "server.toml").symlink_to(outside / "missing.toml")
        ),
        lambda root, outside: os.link(outside / "settings.json", root / "hard.json"),
    ],
)
def test_a_link_anywhere_in_the_root_is_refused_before_anything_is_written(
    root, repo, tmp_path, plant
):
    """Clarity and the driver run outside the sandbox and follow links: a link in the root
    would let them read a private folder or overwrite a file the sandbox denies."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "settings.json").write_text("{}", encoding="utf-8")
    plant(root, outside)
    with pytest.raises(qa.Refused, match="link"):
        qa.prepare_root(root, repo, init=False)
    assert (outside / "settings.json").read_text(encoding="utf-8") == "{}"
    assert not (outside / "missing.toml").exists()


def test_the_roots_weights_are_checked_by_hash_and_replaced(root, repo):
    """Clarity checks only size and time at start, and both can be set by hand."""
    weights = root / "models" / SMOL / f"{SMOL}-bf16.gguf"
    lock = json.loads((repo / "models.lock").read_text(encoding="utf-8"))
    original = (repo / "models" / SMOL / f"{SMOL}-bf16.gguf").read_bytes()
    lock["models"][SMOL]["gguf"]["sha256"] = hashlib.sha256(original).hexdigest()
    (repo / "models.lock").write_text(json.dumps(lock), encoding="utf-8")
    weights.write_bytes(b"GGUF other model!"[: len(original)].ljust(len(original), b"!"))
    os.utime(weights, ns=(1_000_000_100, 1_000_000_100))
    qa.prepare_root(root, repo, init=False)
    assert weights.read_bytes() == original and weights.stat().st_mtime_ns == 1_000_000_100


def test_relative_paths_are_judged_from_the_root(root, monkeypatch):
    """Clarity runs in the root, so a relative path means a file there, wherever the agent
    runs the driver from."""
    monkeypatch.chdir(root.parent)
    qa.check_clarity_args(root, ["replay", "logs/a.jsonl", "c1"])
    qa.check_clarity_args(root, ["chat", "--resume", "logs/a.jsonl"])
    with pytest.raises(qa.Refused):
        qa.check_clarity_args(root, ["chat", "--resume", "../outside.jsonl"])


def test_a_timed_ctrl_c_and_a_malformed_json_step(tmp_path):
    steps = tmp_path / "steps.txt"
    steps.write_text("/regen 3\n<ctrl-c after 1.5>\n", encoding="utf-8")
    assert qa.parse_steps(steps, json_mode=False) == ["/regen 3", "<ctrl-c after 1.5>"]
    steps.write_text('{"command": ["resume"], "args": "/abs/x"}\n', encoding="utf-8")
    with pytest.raises(qa.Refused):
        qa.parse_steps(steps, json_mode=True)


def test_the_steered_profile_names_the_roots_copy_of_the_vector(root, repo):
    from clarity_chat.profiles import load_defaults, load_profile

    profile = load_profile(root, "smol-steer-hf", load_defaults(root))
    assert profile.backend == "transformers" and profile.device == "cpu"
    assert profile.settings["steering_scale"].value == 4.0
    assert profile.steering is not None
    vector = root / profile.steering.file
    expected = hashlib.sha256((REPO / qa.QA_VECTOR).read_bytes()).hexdigest()
    assert hashlib.sha256(vector.read_bytes()).hexdigest() == profile.steering.sha256 == expected


def test_only_init_puts_the_vector_back(root, repo):
    """QA renames the vector to see a replay refused; the next run mustn't undo that."""
    vector = root / "local" / "vectors" / qa.QA_VECTOR.name
    vector.rename(vector.with_name("moved.gguf"))
    qa.prepare_root(root, repo, init=False)
    assert not vector.exists()
    qa.prepare_root(root, repo, init=True)
    assert vector.is_file()


def test_a_link_at_the_vectors_place_is_refused_before_init_copies_onto_it(root, repo, tmp_path):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"not yours")
    vector = root / "local" / "vectors" / qa.QA_VECTOR.name
    vector.unlink()
    vector.symlink_to(outside)
    with pytest.raises(qa.Refused, match="link"):
        qa.prepare_root(root, repo, init=True)
    assert outside.read_bytes() == b"not yours"


@pytest.mark.parametrize("folder", [True, False])
def test_something_other_than_a_file_at_the_vectors_place_is_refused(root, repo, folder):
    vectors = root / "local" / "vectors"
    if folder:  # a folder named like the vector: shutil.copy would copy into it
        (vectors / qa.QA_VECTOR.name).unlink()
        (vectors / qa.QA_VECTOR.name).mkdir()
    else:  # local/vectors itself a file
        shutil.rmtree(vectors)
        vectors.write_text("x", encoding="utf-8")
    with pytest.raises(qa.Refused, match="vector"):
        qa.prepare_root(root, repo, init=True)
