from __future__ import annotations

import json
import os

import pytest

from clarity_chat import cli
from clarity_chat.batch import RunError, label, run_batch, split_openings
from clarity_chat.logs import read_records
from clarity_chat.profiles import ConfigError
from clarity_chat.session import Session
from tests.conftest import MODELS, TEXT_PROFILE, fake_fetch
from tests.fakes import FakeBackend

OPENINGS = "The kettle sang twice.\n---\n  Ask me about otters.  \n"


@pytest.fixture
def batch(fetched_root, tiny_tokenizer):
    """Runs a batch against the fake backend; returns (blind path, run dir, backend, output)."""
    profiles = fetched_root / "profiles"
    (profiles / "text2.toml").write_text(
        TEXT_PROFILE.replace("A conversation.", "Another header."), encoding="utf-8"
    )
    (fetched_root / "openings.txt").write_text(OPENINGS, encoding="utf-8")

    def run(profile_names=("text", "text2"), seeds=1, replies=None, setup=None, n_ctx=512):
        backend = FakeBackend(replies, n_ctx=n_ctx, tokenizer=tiny_tokenizer)
        if setup:
            setup(backend)
        run_dir = fetched_root / "runs" / "run-test"
        session = Session(
            fetched_root,
            backend=backend,
            logs_dir=run_dir,
            tokenizer_loader=lambda r, v: tiny_tokenizer,
        )
        output: list[str] = []
        blind = run_batch(
            session,
            "run-test",
            run_dir,
            fetched_root / "openings.txt",
            list(profile_names),
            seeds,
            output.append,
        )
        return blind, run_dir, backend, "".join(output)

    return run


# --- openings and labels ------------------------------------------------------------------


def test_openings_split_on_dash_lines_and_are_stripped():
    assert split_openings("one\n---\n  two\nlines \n") == ["one", "two\nlines"]


def test_crlf_files_split_the_same_and_send_lf():
    assert split_openings("one\r\nmore\r\n---\r\ntwo\r\n") == ["one\nmore", "two"]


def test_only_an_exact_dash_line_separates():
    assert split_openings("a\n --- \n----\nb") == ["a\n --- \n----\nb"]


@pytest.mark.parametrize("text", ["---\na", "a\n---\n", "a\n---\n\n---\nb", "   "])
def test_an_empty_opening_is_an_error(text):
    with pytest.raises(ConfigError, match="empty"):
        split_openings(text)


def test_labels_run_past_z():
    assert [label(i) for i in (0, 1, 25, 26, 27, 51, 701, 702)] == [
        "A",
        "B",
        "Z",
        "AA",
        "AB",
        "AZ",
        "ZZ",
        "AAA",
    ]


# --- a run --------------------------------------------------------------------------------


def test_blind_md_has_openings_and_shuffled_labelled_replies_only(batch):
    blind, _, _, _ = batch(replies=[" r1 ", " r2", " r3", " r4"])
    text = blind.read_text(encoding="utf-8")
    assert text.startswith("# Opening 1\n\nThe kettle sang twice.\n")
    assert "# Opening 2\n\nAsk me about otters.\n" in text
    assert text.count("## A\n") == 2 and text.count("## B\n") == 2
    assert text.count("\nRating:\nNote:\n") == 4
    for reply in ("r1", "r2", "r3", "r4"):
        assert f"\n\n{reply}\n\nRating:" in text  # trimmed
    for hidden in ("text2", "call-", "conversation", "42", "run-test"):
        assert hidden not in text


def test_key_maps_each_label_to_its_call(batch):
    blind, run_dir, _, _ = batch(replies=[" r1", " r2", " r3", " r4"])
    key = json.loads((run_dir / "key.json").read_text(encoding="utf-8"))
    calls = {r["call_id"]: r for r in read_records(run_dir / "calls.jsonl") if r["type"] == "call"}
    text = blind.read_text(encoding="utf-8")
    assert key["seeds"] == [42]
    assert isinstance(key["shuffle_seed"], int)
    assert "str.strip()" in key["trim_rule"] and '"---"' in key["split_rule"]
    for opening in key["openings"]:
        assert sorted(opening["labels"]) == ["A", "B"]
        assert {e["profile"] for e in opening["labels"].values()} == {"text", "text2"}
        for tag, entry in opening["labels"].items():
            call = calls[entry["call_id"]]
            assert call["run"] == {"run_id": "run-test", "opening": opening["opening"]}
            assert call["profile"]["name"] == entry["profile"]
            assert f"## {tag}\n\n{call['output_text'].strip()}\n" in text


def test_calls_log_has_a_run_record_seeds_from_42_and_nothing_goes_to_logs(batch, fetched_root):
    _, run_dir, backend, _ = batch(seeds=2)
    rows = list(read_records(run_dir / "calls.jsonl"))
    assert rows[0]["type"] == "run"
    assert rows[0]["seeds"] == [42, 43]
    calls = [r for r in rows if r["type"] == "call"]
    assert len(calls) == 2 * 2 * 2
    assert {c["settings"]["seed"]["origin"] for c in calls} == {"run"}
    assert sorted({c["settings"]["seed"]["value"] for c in calls}) == [42, 43]
    assert [c["settings"]["seed"] for c in backend.calls][:2] == [42, 43]
    assert len({c["conversation_id"] for c in calls}) == 8  # one conversation per call
    assert sum(r["type"] == "server_start" for r in rows) == 1
    assert not (fetched_root / "logs").exists()


def test_a_run_call_can_be_marked_while_reviewing(batch):
    _, run_dir, _, _ = batch()
    log = run_dir / "calls.jsonl"
    call = [r for r in read_records(log) if r["type"] == "call"][1]
    assert cli.main(["note", str(log), call["call_id"], "made-up rating note"]) == 0
    note = list(read_records(log))[-1]
    assert note["type"] == "note" and note["origin"] == "review"
    assert (note["conversation_id"], note["turn_id"]) == (
        call["conversation_id"],
        call["turn_id"],
    )


def test_seeds_give_more_labels_per_opening(batch, fetched_root):
    _, run_dir, _, _ = batch(seeds=2)
    key = json.loads((run_dir / "key.json").read_text(encoding="utf-8"))
    assert [sorted(o["labels"]) for o in key["openings"]] == [["A", "B", "C", "D"]] * 2


def test_progress_shows_counts_never_replies(batch):
    _, _, _, output = batch(replies=[" secret one", " secret two", " three", " four"])
    assert "4/4" in output
    assert "secret" not in output


def test_calls_are_grouped_by_model(batch, fetched_root):
    (fetched_root / "models.toml").write_text(
        MODELS + "\n" + MODELS.replace("[tiny]", "[tiny2]"), encoding="utf-8"
    )
    fake_fetch(fetched_root, "tiny2")
    (fetched_root / "profiles" / "other.toml").write_text(
        TEXT_PROFILE.replace('model = "tiny"', 'model = "tiny2"'), encoding="utf-8"
    )
    _, _, backend, _ = batch(profile_names=("text", "other", "text2"))
    assert backend.started == ["tiny", "tiny2"]


# --- refusals and stops -------------------------------------------------------------------


def test_profiles_with_different_max_tokens_are_refused_before_any_call(batch, fetched_root):
    (fetched_root / "profiles" / "long.toml").write_text(
        TEXT_PROFILE.replace("max_tokens = 16", "max_tokens = 32"), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="max_tokens differ"):
        batch(profile_names=("text", "long"))
    assert not (fetched_root / "runs").exists()


def test_a_profile_listed_twice_is_refused(batch):
    with pytest.raises(ConfigError, match="twice"):
        batch(profile_names=("text", "text"))


def test_an_opening_too_long_for_the_context_is_refused_before_any_call(batch, fetched_root):
    profile = fetched_root / "profiles" / "text.toml"
    profile.write_text("ctx_size = 40\n" + profile.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError, match="Opening 1 with profile text"):
        batch(n_ctx=40)
    assert not (fetched_root / "runs").exists()


@pytest.mark.parametrize(
    "setup",
    [
        lambda b: setattr(b, "fail_with", "server gone"),
        lambda b: setattr(b, "result_flags", ["check_failed:prompt_echo"]),
        lambda b: setattr(b, "cancel_after", 2),
    ],
    ids=["error", "failed_check", "cancelled"],
)
def test_a_failed_call_stops_the_run_before_blind_md(batch, fetched_root, setup):
    with pytest.raises(RunError):
        batch(setup=setup)
    run_dir = fetched_root / "runs" / "run-test"
    assert not (run_dir / "blind.md").exists()
    assert not (run_dir / "key.json").exists()
    calls = [r for r in read_records(run_dir / "calls.jsonl") if r["type"] == "call"]
    assert len(calls) == 1  # the failed call is logged, and nothing ran after it


def test_server_warnings_are_shown_once(batch):
    _, _, _, output = batch(setup=lambda b: b.start_warnings.append("build differs"))
    assert output.count("! build differs") == 1


def test_flagged_calls_are_shown_on_screen(batch):
    _, _, _, output = batch(setup=lambda b: setattr(b, "result_flags", ["incomplete_output_ids"]))
    assert output.count("flags: incomplete_output_ids") == 4  # one line per call, named


# --- the command --------------------------------------------------------------------------


def test_run_command_reports_where_blind_md_is(fetched_root, tiny_tokenizer, monkeypatch, capsys):
    (fetched_root / "openings.txt").write_text(OPENINGS, encoding="utf-8")

    def session(root, logs_dir, on_server=None):
        return Session(
            root,
            backend=FakeBackend(tokenizer=tiny_tokenizer),
            logs_dir=logs_dir,
            tokenizer_loader=lambda r, v: tiny_tokenizer,
        )

    monkeypatch.setattr(cli, "Session", session)
    argv = ["--root", str(fetched_root), "run", "--openings", str(fetched_root / "openings.txt")]
    assert cli.main([*argv, "--profiles", "text"]) == 0
    out = capsys.readouterr().out
    assert "blind.md" in out and "key.json" in out
    (run_dir,) = (fetched_root / "runs").iterdir()
    assert (run_dir / "blind.md").exists()


def test_run_command_explains_a_stopped_run(fetched_root, tiny_tokenizer, monkeypatch, capsys):
    (fetched_root / "openings.txt").write_text(OPENINGS, encoding="utf-8")
    backend = FakeBackend(tokenizer=tiny_tokenizer)
    backend.fail_with = "server gone"

    def session(root, logs_dir, on_server=None):
        return Session(
            root, backend=backend, logs_dir=logs_dir, tokenizer_loader=lambda r, v: tiny_tokenizer
        )

    monkeypatch.setattr(cli, "Session", session)
    argv = ["--root", str(fetched_root), "run", "--openings", str(fetched_root / "openings.txt")]
    assert cli.main([*argv, "--profiles", "text"]) == 1
    err = capsys.readouterr().err
    assert "server gone" in err and "No blind.md" in err


# --- review fixes -------------------------------------------------------------------------


def test_a_missing_openings_file_is_a_clear_error(fetched_root, tiny_tokenizer):
    from clarity_chat.batch import read_openings

    with pytest.raises(ConfigError, match="Can't read the openings file"):
        read_openings(fetched_root / "nope.txt")
    (fetched_root / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ConfigError, match="UTF-8"):
        read_openings(fetched_root / "latin1.txt")


def test_a_byte_order_mark_is_not_sent():
    assert split_openings("﻿one\r\n---\r\ntwo") == ["one", "two"]


def test_the_same_profile_by_name_and_by_path_is_refused(batch, fetched_root):
    with pytest.raises(ConfigError, match="twice"):
        batch(profile_names=("text", str(fetched_root / "profiles" / "text.toml")))


def test_profiles_with_one_name_but_different_max_tokens_are_refused(batch, fetched_root):
    for folder, limit in (("a", 16), ("b", 32)):
        (fetched_root / folder).mkdir()
        (fetched_root / folder / "x.toml").write_text(
            TEXT_PROFILE.replace("max_tokens = 16", f"max_tokens = {limit}"), encoding="utf-8"
        )
    with pytest.raises(ConfigError, match="max_tokens differ"):
        batch(profile_names=[str(fetched_root / f / "x.toml") for f in ("a", "b")])


def test_calls_use_the_profile_as_checked_even_if_its_file_changes(batch, fetched_root):
    def setup(backend):
        complete = backend.complete

        def edit_then_complete(*args, **kwargs):
            (fetched_root / "profiles" / "text.toml").write_text(
                TEXT_PROFILE.replace("max_tokens = 16", "max_tokens = 99"), encoding="utf-8"
            )
            return complete(*args, **kwargs)

        backend.complete = edit_then_complete

    _, _, backend, _ = batch(setup=setup)
    assert {c["settings"]["max_tokens"] for c in backend.calls} == {16}


def test_a_server_restart_mid_run_stops_it(batch, fetched_root):
    def setup(backend):
        start = backend.start

        def restart_after_first_call(model, weights, ctx_size):
            if backend.calls:
                backend.started.clear()  # as if the server had died
            return start(model, weights, ctx_size)

        backend.start = restart_after_first_call

    with pytest.raises(RunError, match="restarted during the run"):
        batch(setup=setup)
    assert not (fetched_root / "runs" / "run-test" / "blind.md").exists()


def test_ctrl_c_between_calls_says_where_the_run_stopped(batch):
    def setup(backend):
        start = backend.start

        def interrupt_after_first_call(model, weights, ctx_size):
            if backend.calls:
                raise KeyboardInterrupt
            return start(model, weights, ctx_size)

        backend.start = interrupt_after_first_call

    with pytest.raises(RunError, match="Interrupted .* opening 2, profile text"):
        batch(setup=setup)


def test_profiles_on_one_model_with_different_context_sizes_each_start_once(batch, fetched_root):
    profile = fetched_root / "profiles" / "text2.toml"
    profile.write_text("ctx_size = 4096\n" + profile.read_text(encoding="utf-8"), "utf-8")
    _, _, backend, _ = batch(seeds=2)
    assert backend.ctx_sizes == [8192, 4096]  # grouped by size: no restart mid-group


def test_a_conversation_warning_prints_once_in_a_run(batch, fetched_root, monkeypatch):
    from clarity_chat import session as session_module

    real = session_module.Session.new_conversation

    def with_warning(self, *args, **kwargs):
        conversation = real(self, *args, **kwargs)
        conversation.warnings.append("made-up conversation warning")
        return conversation

    monkeypatch.setattr(session_module.Session, "new_conversation", with_warning)
    _, _, _, output = batch(seeds=2)
    assert output.count("made-up conversation warning") == 1


def test_a_runs_files_are_readable_only_by_you(batch):
    blind, run_dir, _, _ = batch(replies=[" r1", " r2", " r3", " r4"])
    if os.name != "nt":
        for path in (blind, run_dir / "key.json", run_dir / "calls.jsonl"):
            assert path.stat().st_mode & 0o777 == 0o600, path.name
