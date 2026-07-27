import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FLUID_CASES = ROOT / "tests" / "cases" / "fluid"
RUNNER_PATH = (
    FLUID_CASES
    / "run_free_surface_wp3_wp7_nitsche_coercivity_qualification.py"
)
MATRIX_PATH = (
    FLUID_CASES
    / "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp3_wp7_nitsche_coercivity_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def matrix_document():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_matrix_is_strictly_scoped_and_every_closure_remains_open():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["matrix_id"] == runner.EXPECTED_MATRIX_ID
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["qualification_scope"] == runner.EXPECTED_SCOPE
    assert (
        matrix["closure_request_policy"]
        == runner.EXPECTED_CLOSURE_REQUEST_POLICY
    )
    assert matrix["qualification_disposition"] == {
        "fsr16_closed": False,
        "fsr07_closed": False,
        "wp3_closed": False,
        "wp7_closed": False,
        "q1_closed": False,
        "uniform_coercivity_bound_established": False,
    }
    assert matrix["method_coercivity_lower_bound"] is None
    assert matrix["uniform_bound_status"] == "UNFROZEN_NO_BOUND_INVENTED"
    assert matrix["case_axes"] == runner.EXPECTED_CASE_AXES
    assert "h_over_domain_length" not in matrix["case_axes"]
    assert (
        matrix["matching_derivation"]
        == runner.EXPECTED_MATCHING_DERIVATION
    )
    assert (
        runner.EXPECTED_MATCHING_DERIVATION
        in runner.EXPECTED_PARENT_SHA256
    )
    assert matrix["prospective_tests"] == []


@pytest.mark.parametrize(
    "claim",
    [
        "fsr16_closure",
        "fsr07_closure",
        "wp3_closure",
        "wp7_closure",
        "wp3_wp7_joint_closure",
        "q1_closure",
    ],
)
def test_runner_rejects_every_premature_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_runner_rejects_unknown_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported requested claim"):
        runner.requested_claim(
            ["--requested-claim", "unregistered_claim"]
        )


def test_contract_rejects_scope_policy_disposition_or_bound_promotion():
    runner = load_runner()
    matrix = matrix_document()

    promoted_scope = copy.deepcopy(matrix)
    promoted_scope["qualification_scope"] = "WP-3 and WP-7 closed"
    with pytest.raises(ValueError, match="qualification scope changed"):
        runner.validate_joint_contract(promoted_scope)

    promoted_policy = copy.deepcopy(matrix)
    promoted_policy["closure_request_policy"]["accepted_claim"] = (
        "wp7_closure"
    )
    with pytest.raises(ValueError, match="closure-request policy changed"):
        runner.validate_joint_contract(promoted_policy)

    promoted_disposition = copy.deepcopy(matrix)
    promoted_disposition["qualification_disposition"]["wp7_closed"] = True
    with pytest.raises(ValueError, match="disposition changed"):
        runner.validate_joint_contract(promoted_disposition)

    invented_bound = copy.deepcopy(matrix)
    invented_bound["method_coercivity_lower_bound"] = 0.64
    with pytest.raises(ValueError, match="lower bound was invented"):
        runner.validate_joint_contract(invented_bound)

    drifted_derivation = copy.deepcopy(matrix)
    drifted_derivation["matching_derivation"] = "Documentation/other.md"
    with pytest.raises(ValueError, match="matching derivation changed"):
        runner.validate_joint_contract(drifted_derivation)


def test_contract_freezes_parent_and_implementation_source_inventories():
    runner = load_runner()
    matrix = matrix_document()

    assert (
        runner._artifact_map(matrix["parent_artifacts"], "parent artifacts")
        == runner.EXPECTED_PARENT_SHA256
    )
    assert (
        runner._artifact_map(
            matrix["implementation_sources"],
            "implementation sources",
        )
        == runner.EXPECTED_SOURCE_SHA256
    )
    runner.validate_frozen_dependencies(matrix)

    drifted = copy.deepcopy(matrix)
    drifted["parent_artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent artifact inventory changed"):
        runner.validate_joint_contract(drifted)


def test_frozen_matrix_rejects_byte_drift_before_execution(tmp_path):
    runner = load_runner()
    changed = tmp_path / MATRIX_PATH.name
    changed.write_text(
        MATRIX_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matrix bytes changed"):
        runner.load_registry(changed)


def test_json_parser_rejects_duplicate_keys(tmp_path):
    runner = load_runner()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.parse_json_document(duplicate)


def test_dependency_verifier_fails_on_changed_bytes(tmp_path):
    runner = load_runner()
    parent = tmp_path / "parent.txt"
    source = tmp_path / "source.cpp"
    parent.write_text("parent\n", encoding="utf-8")
    source.write_text("source\n", encoding="utf-8")
    registry = {
        "parent_artifacts": [
            {
                "path": "parent.txt",
                "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            }
        ],
        "implementation_sources": [
            {
                "path": "source.cpp",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    runner.validate_frozen_dependencies(registry, tmp_path)

    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation source bytes changed"):
        runner.validate_frozen_dependencies(registry, tmp_path)


def test_rejected_claim_creates_no_output(tmp_path):
    runner = load_runner()
    output = tmp_path / "qualification-output"

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.main(
            [
                "--requested-claim",
                "wp7_closure",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def test_validate_only_reports_nonclosure_without_building():
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    summary = json.loads(completed.stdout)

    assert summary["outcome"] == "PASS"
    assert (
        summary["requested_claim"]
        == "joint_low_level_prerequisite"
    )
    assert summary["group_count"] == 2
    assert summary["test_count"] == 6
    assert summary["quantitative_evidence_gate_count"] == 31
    assert summary["method_coercivity_lower_bound"] is None
    assert summary["uniform_bound_status"] == (
        "UNFROZEN_NO_BOUND_INVENTED"
    )
    assert all(
        value is False
        for value in summary["qualification_disposition"].values()
    )
