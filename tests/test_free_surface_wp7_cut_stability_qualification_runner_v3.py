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
        / "run_free_surface_wp7_cut_stability_qualification_v3.py"
    )


def _registry_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "free_surface_wp7_cut_stability_qualification_revision_v3.json"
    )


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp7_cut_stability_qualification_runner_v3",
        _runner_path(),
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_wp7_v3_promotes_exactly_one_topology_policy_row():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())

    assert len(registry["executable_tests"]) == 15
    assert len(registry["prospective_tests"]) == 6
    assert runner.EXPECTED_PROMOTED_TEST in registry["executable_tests"]
    assert runner.EXPECTED_PROMOTED_TEST not in registry["prospective_tests"]
    assert {group["id"] for group in registry["groups"]} == (
        runner.EXPECTED_GROUP_IDS
    )


def test_wp7_v3_runtime_contains_no_prospective_test():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())
    runtime_tests = {test for group in registry["groups"] for test in group["tests"]}

    assert runtime_tests == set(registry["executable_tests"])
    assert runtime_tests.isdisjoint(registry["prospective_tests"])
    assert registry["gates"] == runner.EXPECTED_RUNTIME_GATES


def test_wp7_v3_quantitative_promotion_contract_is_exact():
    runner = _load_runner()
    revision = json.loads(_registry_path().read_text(encoding="utf-8"))
    observed = runner._promotion_property_contract(
        revision["promotion"]["quantitative_evidence"]
    )

    assert observed == runner.EXPECTED_PROMOTION_PROPERTIES
    assert len(runner.build_runtime_registry(revision)["quantitative_evidence"]) == 55


def test_wp7_v3_rejects_premature_closure_claims():
    runner = _load_runner()

    for claim in ("fsr07_closure", "wp7_closure", "q1_closure"):
        with pytest.raises(ValueError, match="outside this matrix"):
            runner.requested_claim(["--requested-claim", claim])


def test_wp7_v3_rejects_promotion_gate_drift():
    runner = _load_runner()
    revision = json.loads(_registry_path().read_text(encoding="utf-8"))
    changed = copy.deepcopy(revision)
    changed["promotion"]["quantitative_evidence"][0]["threshold"] = 4

    with pytest.raises(ValueError, match="quantitative promotion gates changed"):
        runner.validate_revision_contract(changed)


def test_wp7_v3_rejects_scope_or_disposition_drift():
    runner = _load_runner()
    revision = json.loads(_registry_path().read_text(encoding="utf-8"))
    changed_scope = copy.deepcopy(revision)
    changed_scope["qualification_scope"] = "expanded"
    with pytest.raises(ValueError, match="qualification scope changed"):
        runner.validate_revision_contract(changed_scope)

    changed_disposition = copy.deepcopy(revision)
    changed_disposition["qualification_disposition"]["wp7_closed"] = True
    with pytest.raises(ValueError, match="qualification disposition changed"):
        runner.validate_revision_contract(changed_disposition)


def test_wp7_v3_implementation_and_parent_bindings_are_exact():
    runner = _load_runner()
    observation = runner.validate_implementation_binding()

    assert observation["implementation_source_commit"] == (
        runner.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    )
    assert observation["implementation_source_sha256"] == (
        runner.EXPECTED_IMPLEMENTATION_SOURCE_SHA256
    )
    assert runner.sha256_file(runner.PARENT_RUNNER_PATH) == (
        runner.EXPECTED_PARENT_RUNNER_SHA256
    )
    assert runner.sha256_file(runner.BASE_MATRIX_PATH) == (
        runner.EXPECTED_BASE_MATRIX_SHA256
    )


def test_wp7_v3_frozen_revision_byte_drift_is_rejected(tmp_path):
    runner = _load_runner()
    changed = tmp_path / _registry_path().name
    changed.write_bytes(_registry_path().read_bytes() + b"\n")
    runner.DEFAULT_REGISTRY = changed

    with pytest.raises(ValueError, match="frozen revision bytes changed"):
        runner.load_registry(changed)


def test_wp7_v3_validate_only_reports_blocked_prerequisite():
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
    assert summary["requested_claim"] == "topology_policy_prerequisite"
    assert summary["closure_state"] == (
        "BLOCKED_BY_SIX_PROSPECTIVE_EVIDENCE_ROWS"
    )
    assert summary["group_count"] == 4
    assert summary["test_count"] == 15
    assert summary["executable_test_count"] == 15
    assert summary["prospective_test_count"] == 6
    assert summary["serial_quantitative_gate_count"] == 55
    assert summary["fsr07_closed"] is False
    assert summary["wp7_closed"] is False
    assert summary["q1_closed"] is False
