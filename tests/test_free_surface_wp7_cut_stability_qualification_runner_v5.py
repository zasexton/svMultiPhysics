import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _runner_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "run_free_surface_wp7_cut_stability_qualification_v5.py"
    )


def _registry_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "free_surface_wp7_cut_stability_qualification_revision_v5.json"
    )


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp7_cut_stability_qualification_runner_v5",
        _runner_path(),
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_wp7_v5_binds_two_route_fixture_and_retains_open_rows():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())

    assert registry["matrix_id"] == "free_surface_wp7_cut_stability_v5"
    assert len(registry["executable_tests"]) == 16
    assert len(registry["prospective_tests"]) == 5
    assert runner.EXPECTED_PROMOTED_TESTS <= set(registry["executable_tests"])
    assert runner.EXPECTED_PROMOTED_TESTS.isdisjoint(
        registry["prospective_tests"]
    )
    assert "two canonical generated-boundary trace routes" in (
        registry["qualification_scope"]
    )
    assert registry["qualification_disposition"] == {
        "fsr07_closed": False,
        "wp7_closed": False,
        "q1_closed": False,
    }


def test_wp7_v5_runtime_contains_only_executable_tests():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())
    runtime_tests = {test for group in registry["groups"] for test in group["tests"]}

    assert runtime_tests == set(registry["executable_tests"])
    assert runtime_tests.isdisjoint(registry["prospective_tests"])
    assert {group["id"] for group in registry["groups"]} == (
        runner.EXPECTED_GROUP_IDS
    )
    assert registry["gates"] == runner.EXPECTED_RUNTIME_GATES


def test_wp7_v5_implementation_and_revision_bindings_are_exact():
    runner = _load_runner()
    observation = runner.validate_implementation_binding()

    assert observation["implementation_source_commit"] == (
        "01d5bbb6ac9ce069f4727096084af0bb6d8d39c3"
    )
    assert observation["implementation_source_sha256"] == (
        runner.EXPECTED_IMPLEMENTATION_SOURCE_SHA256
    )
    assert runner.sha256_file(runner.PARENT_REVISION_RUNNER_PATH) == (
        runner.EXPECTED_PARENT_REVISION_RUNNER_SHA256
    )


def test_wp7_v5_rejects_premature_closure_claims():
    runner = _load_runner()

    for claim in ("fsr07_closure", "wp7_closure", "q1_closure"):
        with pytest.raises(ValueError, match="outside this matrix"):
            runner.requested_claim(["--requested-claim", claim])


def test_wp7_v5_rejects_scope_and_gate_drift():
    runner = _load_runner()
    revision = json.loads(_registry_path().read_text(encoding="utf-8"))

    changed_scope = copy.deepcopy(revision)
    changed_scope["qualification_scope"] = "expanded"
    with pytest.raises(ValueError, match="qualification scope changed"):
        runner.validate_revision_contract(changed_scope)

    changed_gate = copy.deepcopy(revision)
    changed_gate["promotions"][1]["quantitative_evidence"][0]["threshold"] = 4
    with pytest.raises(ValueError, match="quantitative promotion gates changed"):
        runner.validate_revision_contract(changed_gate)


def test_wp7_v5_frozen_revision_byte_drift_is_rejected(tmp_path):
    runner = _load_runner()
    changed = tmp_path / _registry_path().name
    changed.write_bytes(_registry_path().read_bytes() + b"\n")
    runner.DEFAULT_REGISTRY = changed

    with pytest.raises(ValueError, match="frozen revision bytes changed"):
        runner.load_registry(changed)


def test_wp7_v5_validate_only_reports_bounded_prerequisite():
    result = subprocess.run(
        [sys.executable, str(_runner_path()), "--validate-only"],
        cwd=_repository(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS"
    assert summary["requested_claim"] == (
        "topology_and_node_crossing_prerequisite"
    )
    assert summary["closure_state"] == (
        "BLOCKED_BY_FIVE_PROSPECTIVE_EVIDENCE_ROWS"
    )
    assert summary["group_count"] == 4
    assert summary["test_count"] == 16
    assert summary["executable_test_count"] == 16
    assert summary["prospective_test_count"] == 5
    assert summary["serial_quantitative_gate_count"] == 67
    assert summary["fsr07_closed"] is False
    assert summary["wp7_closed"] is False
    assert summary["q1_closed"] is False
