import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_configuration_qualification.py"
)
REGISTRY_PATH = RUNNER_PATH.with_name(
    "free_surface_configuration_qualification_matrix.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "free_surface_configuration_qualification_runner", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def completed_document(registry):
    suites = {}
    for full_name in registry["tests"]:
        suite, name = full_name.split(".", 1)
        suites.setdefault(suite, []).append(
            {"name": name, "status": "RUN", "result": "COMPLETED", "time": "0s"}
        )
    return {
        "tests": len(registry["tests"]),
        "failures": 0,
        "disabled": 0,
        "errors": 0,
        "testsuites": [
            {"name": suite, "testsuite": tests}
            for suite, tests in sorted(suites.items())
        ],
    }


def test_frozen_registry_is_self_consistent():
    runner = load_runner()
    registry = runner.load_registry(REGISTRY_PATH)
    assert registry["gates"]["expected_test_count"] == len(registry["tests"])
    assert len(registry["tests"]) == len(set(registry["tests"]))


def test_complete_exact_matrix_passes():
    runner = load_runner()
    registry = runner.load_registry(REGISTRY_PATH)
    checks = runner.evaluate_results(
        registry, completed_document(registry), 0, None
    )
    assert checks
    assert all(check["passed"] for check in checks)


@pytest.mark.parametrize("defect", ["missing", "unexpected", "skipped", "failed"])
def test_matrix_defects_fail_closed(defect):
    runner = load_runner()
    registry = runner.load_registry(REGISTRY_PATH)
    document = completed_document(registry)
    if defect == "missing":
        document["testsuites"][0]["testsuite"].pop()
        document["tests"] -= 1
    elif defect == "unexpected":
        document["testsuites"][0]["testsuite"].append(
            {
                "name": "UnexpectedConfigurationCase",
                "status": "RUN",
                "result": "COMPLETED",
            }
        )
        document["tests"] += 1
    elif defect == "skipped":
        document["testsuites"][0]["testsuite"][0]["result"] = "SKIPPED"
    else:
        document["testsuites"][0]["testsuite"][0]["failures"] = [
            {"failure": "injected"}
        ]
        document["failures"] = 1
    checks = runner.evaluate_results(registry, document, 0, None)
    assert any(not check["passed"] for check in checks)


def test_registry_rejects_duplicate_test_names(tmp_path):
    runner = load_runner()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["tests"].append(registry["tests"][0])
    registry["gates"]["expected_test_count"] += 1
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        runner.load_registry(invalid_path)
