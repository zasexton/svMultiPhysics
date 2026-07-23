import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp3_sharp_boundary_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp3_sharp_boundary_qualification_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp3_sharp_boundary_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def matrix_document():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_wp3_matrix_is_strictly_scoped_and_remains_open():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["qualification_scope"] == runner.EXPECTED_SCOPE
    assert matrix["closure_request_policy"] == (
        runner.EXPECTED_CLOSURE_REQUEST_POLICY
    )
    assert matrix["prospective_tests"] == []
    threshold = matrix["unfrozen_joint_thresholds"]
    assert len(threshold) == 1
    assert threshold[0]["status"] == "UNFROZEN_NO_BOUND_INVENTED"


@pytest.mark.parametrize(
    "claim",
    ["fsr16_closure", "wp3_closure", "wp7_closure", "q1_closure"],
)
def test_wp3_rejects_every_premature_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp3_rejects_unknown_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported WP-3 requested claim"):
        runner.requested_claim(
            ["--requested-claim", "unregistered_claim"]
        )


def test_wp3_contract_rejects_scope_or_policy_promotion():
    runner = load_runner()
    matrix = matrix_document()

    promoted_scope = copy.deepcopy(matrix)
    promoted_scope["qualification_scope"] = "WP-3 closed"
    with pytest.raises(ValueError, match="qualification scope changed"):
        runner.validate_wp3_contract(promoted_scope)

    promoted_policy = copy.deepcopy(matrix)
    promoted_policy["closure_request_policy"]["accepted_claim"] = (
        "wp3_closure"
    )
    with pytest.raises(ValueError, match="closure-request policy changed"):
        runner.validate_wp3_contract(promoted_policy)


def test_wp3_mpi_property_contracts_are_root_gated_and_two_rank():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    groups = {group["id"]: group for group in matrix["groups"]}

    for group_id in runner.EXPECTED_MPI_RECORDED_PROPERTIES:
        group = groups[group_id]
        assert group["mpi_ranks"] == 2
        assert group["gtest_output_copies"] == 1
        assert len(group["recorded_properties"]) == len(
            runner.EXPECTED_MPI_RECORDED_PROPERTIES[group_id]
        )


def test_wp3_matrix_has_no_duplicate_json_keys():
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    json.loads(
        MATRIX_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )


def test_wp3_loader_rejects_any_frozen_matrix_byte_change(tmp_path):
    runner = load_runner()
    changed = matrix_document()
    changed["status"] = "EDITED_AFTER_FREEZE"
    changed_path = tmp_path / MATRIX_PATH.name
    changed_path.write_text(
        json.dumps(changed, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="frozen registry bytes changed"):
        runner.load_registry(changed_path)


def test_wp3_cli_rejects_closure_before_execution_argument_parsing(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp3_closure",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "outside this matrix" in result.stderr
    assert not output.exists()


def test_wp3_validate_only_reports_prerequisite_without_closure():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS"
    assert summary["requested_claim"] == "low_level_prerequisite"
    assert summary["group_count"] == 7
    assert summary["test_count"] == 25
    assert summary["recorded_property_gate_count"] == 30
    assert summary["prospective_test_count"] == 0
    assert summary["joint_wp7_threshold_frozen"] is False
    assert summary["wp3_closed"] is False
