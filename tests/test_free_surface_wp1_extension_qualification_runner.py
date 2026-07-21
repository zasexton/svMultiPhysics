import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _repository():
    return Path(__file__).resolve().parents[1]


def _load_runner():
    script = (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "run_free_surface_wp1_extension_qualification.py"
    )
    spec = importlib.util.spec_from_file_location(
        "free_surface_wp1_extension_qualification_runner", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry_path():
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "free_surface_wp1_extension_qualification_matrix.json"
    )


def test_frozen_wp1_registry_has_exact_groups_and_counts():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())

    assert registry["status"] == "FROZEN_BEFORE_EXECUTION"
    assert [group["id"] for group in registry["groups"]] == [
        "application_serial",
        "application_mpi",
        "physics_serial",
        "physics_mpi",
        "level_set_serial",
    ]
    tests = [name for group in registry["groups"] for name in group["tests"]]
    assert len(tests) == 53
    assert len(set(tests)) == 53
    assert [point["id"] for point in registry["phase_points"]] == [
        "translating_drop_exit",
        "reversible_enright_exit",
    ]
    assert all(
        point["expected_release_disposition"] == "INCONCLUSIVE_RESOLUTION"
        for point in registry["phase_points"]
    )


def test_serial_result_gate_requires_exact_complete_test_set():
    runner = _load_runner()
    expected = ["Suite.One", "Suite.Two"]
    document = {
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "disabled": 0,
        "testsuites": [
            {
                "name": "Suite",
                "testsuite": [
                    {
                        "name": "One",
                        "status": "RUN",
                        "result": "COMPLETED",
                    },
                    {
                        "name": "Two",
                        "status": "RUN",
                        "result": "COMPLETED",
                    },
                ],
            }
        ],
    }

    checks = runner.evaluate_serial_result(expected, document, 0, None)
    assert checks
    assert all(check["passed"] for check in checks)

    document["testsuites"][0]["testsuite"][1]["result"] = "SKIPPED"
    rejected = runner.evaluate_serial_result(expected, document, 0, None)
    assert not next(
        check for check in rejected if check["metric"] == "incomplete_or_skipped_tests"
    )["passed"]


def test_mpi_result_gate_requires_every_test_on_every_rank():
    runner = _load_runner()
    expected = ["Suite.One", "Suite.Two"]
    one_rank = "\n".join(
        [
            "[ RUN      ] Suite.One",
            "[       OK ] Suite.One (0 ms)",
            "[ RUN      ] Suite.Two",
            "[       OK ] Suite.Two (0 ms)",
        ]
    )
    stdout = one_rank + "\n" + one_rank + "\n"

    checks = runner.evaluate_mpi_result(
        expected, 2, stdout, "", 0, None
    )
    assert checks
    assert all(check["passed"] for check in checks)

    rejected = runner.evaluate_mpi_result(
        expected, 2, one_rank, "", 0, None
    )
    assert not next(
        check
        for check in rejected
        if check["metric"] == "pass_multiplicity:Suite.Two"
    )["passed"]


def test_registry_rejects_duplicate_test(tmp_path):
    runner = _load_runner()
    registry = json.loads(_registry_path().read_text(encoding="utf-8"))
    duplicate = registry["groups"][0]["tests"][0]
    registry["groups"][1]["tests"].append(duplicate)
    registry["gates"]["expected_distinct_test_count"] += 1
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate frozen test"):
        runner.load_registry(path)
