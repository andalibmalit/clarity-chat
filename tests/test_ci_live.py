"""The Linux CPU live job's offline parts (notes/plans/ci-live-linux.md): which live tests
need a GPU, and the guard that fails the job when a live test skipped or none ran."""

from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("junit_guard", REPO / "scripts" / "junit_guard.py")
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def about_placement(statement: ast.stmt) -> bool:
    """A statement a gpu test may hold: an assert whose test (not its message) reads
    placement, `placement = ...server_record["placement"]`, or an `if` holding only these
    (the Mac's Metal checks)."""
    if isinstance(statement, ast.Assert):
        return "placement" in ast.unparse(statement.test)
    if isinstance(statement, ast.Assign):
        value = statement.value
        return (
            isinstance(value, ast.Subscript)
            and ast.unparse(value.slice) == "'placement'"
            and ast.unparse(value.value).endswith("server_record")
        )
    if isinstance(statement, ast.If):
        return all(about_placement(s) for s in statement.body + statement.orelse)
    return False


def gpu_problems(source: str) -> list[str]:
    """What's wrong with the `gpu` marking in `source`, test_live.py's text. CI runs
    `-m "live and not gpu"` on a CPU build: a check of where the model was placed can't pass
    there, and nothing else may be deselected with it."""
    tree = ast.parse(source)
    tests = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    reads_placement = {
        name
        for name, node in tests.items()
        if any("placement" in ast.unparse(s) for s in node.body)
    }
    gpu = {
        name
        for name, node in tests.items()
        if any(ast.unparse(d) == "pytest.mark.gpu" for d in node.decorator_list)
    }
    problems = [
        f"line {node.lineno}: pytestmark carries gpu, which deselects the whole module"
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(t) == "pytestmark" for t in node.targets)
        and "gpu" in ast.unparse(node.value)
    ]
    if not reads_placement or gpu != reads_placement:
        problems.append(f"gpu marks {sorted(gpu)}; placement is read by {sorted(reads_placement)}")
    for name in sorted(gpu):  # nothing else rides along
        problems += [
            f"{name}, line {s.lineno}: {ast.unparse(s).splitlines()[0]}"
            for s in tests[name].body
            if not about_placement(s)
        ]
    return problems


LIVE = (REPO / "tests" / "test_live.py").read_text(encoding="utf-8")
GPU_TEST = (
    "def test_server_start_puts_the_model_on_the_gpu(session):\n"
    '    placement = session.backend.server_record["placement"]\n'
)


def test_the_placement_checks_and_only_they_carry_the_gpu_marker():
    assert gpu_problems(LIVE) == []
    markers = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
    assert any(m.startswith("gpu:") for m in markers["pytest"]["ini_options"]["markers"])


@pytest.mark.parametrize(
    "mutation",
    [
        # The checks the CI review (S1, S2) found the marker test let through: each would
        # deselect a check that a CPU build can run.
        (
            GPU_TEST,
            GPU_TEST + '    assert session.backend.server_record["warnings"] == [], placement\n',
        ),
        (GPU_TEST, GPU_TEST + "    all_checks_ok(session.backend.server_record)\n"),
        ("pytestmark = pytest.mark.live\n", "pytestmark = [pytest.mark.live, pytest.mark.gpu]\n"),
    ],
    ids=["assert-with-placement-message", "bare-call", "module-pytestmark"],
)
def test_a_gpu_test_that_checks_anything_else_is_caught(mutation):
    old, new = mutation
    assert LIVE.count(old) == 1
    assert gpu_problems(LIVE.replace(old, new)) != []


SUITE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0"
skipped="{skipped}" tests="{tests}" time="1.0">{cases}</testsuite></testsuites>
"""
PASSED = '<testcase classname="tests.test_live" name="test_{n}" time="0.1" />'
SKIPPED = (
    '<testcase classname="tests.test_live" name="test_{n}" time="0.0">'
    '<skipped type="pytest.skip" message="llama-quantize is not built">skip</skipped>'
    "</testcase>"
)


FAILED = (
    '<testcase classname="tests.test_live" name="test_{n}" time="0.1">'
    '<{element} message="assert 1 == 2">trace</{element}></testcase>'
)


def junit(tmp_path: Path, passed: int, skipped: int, failed: str = "") -> Path:
    cases = [PASSED.format(n=n) for n in range(passed)]
    cases += [SKIPPED.format(n=passed + n) for n in range(skipped)]
    if failed:
        cases.append(FAILED.format(n=passed + skipped, element=failed))
    path = tmp_path / "live.xml"
    text = SUITE.format(skipped=skipped, tests=passed + skipped, cases="".join(cases))
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("passed", "skipped", "words"),
    [(18, 1, "1 skipped"), (0, 0, "no test ran"), (0, 19, "19 skipped")],
)
def test_the_guard_fails_a_run_with_a_skip_or_no_test(tmp_path, passed, skipped, words):
    path = junit(tmp_path, passed, skipped)
    message = guard.problem(path)
    assert message is not None and words in message, message
    assert guard.main([str(path)]) == 1


def test_the_guard_fails_a_run_with_fewer_tests_than_the_job_runs(tmp_path):
    """The CI review's S2: a module marker that deselects test_live.py leaves one live test
    elsewhere, which ran and didn't skip."""
    path = junit(tmp_path, 1, 0)
    message = guard.problem(path, at_least=20)
    assert message is not None and "fewer than 20" in message, message
    assert guard.main([str(path), "--at-least", "20"]) == 1


@pytest.mark.parametrize("element", ["failure", "error"])
def test_the_guard_fails_a_run_with_a_failure_or_an_error(tmp_path, element):
    """The CI review's C3: in the job pytest's own status fails first, but the guard alone
    must not call such a run good."""
    path = junit(tmp_path, 20, 0, failed=element)
    message = guard.problem(path)
    assert message is not None and "1 failed or errored" in message, message
    assert guard.main([str(path)]) == 1


def test_the_guard_passes_a_run_where_every_test_ran(tmp_path, capsys):
    path = junit(tmp_path, 20, 0)
    assert guard.problem(path, at_least=20) is None
    assert guard.main([str(path), "--at-least", "20"]) == 0
    assert "20 tests ran" in capsys.readouterr().out
