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


def live_tests() -> dict[str, ast.FunctionDef]:
    tree = ast.parse((REPO / "tests" / "test_live.py").read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def marked_gpu(node: ast.FunctionDef) -> bool:
    return any(ast.unparse(d) == "pytest.mark.gpu" for d in node.decorator_list)


def test_the_placement_checks_and_only_they_carry_the_gpu_marker():
    """CI runs `-m "live and not gpu"` on a CPU build: a check of where the model was placed
    can't pass there, and nothing else may be deselected with it."""
    tests = live_tests()
    reads_placement = {
        name
        for name, node in tests.items()
        if any("placement" in ast.unparse(s) for s in node.body)
    }
    gpu = {name for name, node in tests.items() if marked_gpu(node)}
    assert reads_placement and gpu == reads_placement, (gpu, reads_placement)
    for name in gpu:  # nothing else rides along: every assert is about placement
        asserts = [s for s in ast.walk(tests[name]) if isinstance(s, ast.Assert)]
        assert asserts and all("placement" in ast.unparse(s) for s in asserts), name
    markers = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
    assert any(m.startswith("gpu:") for m in markers["pytest"]["ini_options"]["markers"])


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


def junit(tmp_path: Path, passed: int, skipped: int) -> Path:
    cases = [PASSED.format(n=n) for n in range(passed)]
    cases += [SKIPPED.format(n=passed + n) for n in range(skipped)]
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


def test_the_guard_passes_a_run_where_every_test_ran(tmp_path, capsys):
    path = junit(tmp_path, 19, 0)
    assert guard.problem(path) is None
    assert guard.main([str(path)]) == 0
    assert "19 tests ran" in capsys.readouterr().out
