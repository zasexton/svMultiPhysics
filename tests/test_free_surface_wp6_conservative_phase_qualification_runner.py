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
    / "run_free_surface_wp6_conservative_phase_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp6_conservative_phase_qualification_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp6_conservative_phase_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_wp6_matrix_is_frozen_but_explicitly_incomplete():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["qualification_scope"] == runner.EXPECTED_SCOPE
    assert matrix["prospective_tests"] == []
    assert len(matrix["unqualified_required_campaigns"]) == 10
    assert all(
        entry["status"] == "REQUIRED_NOT_CLAIMED"
        for entry in matrix["unqualified_required_campaigns"]
    )
    assert matrix["release_matrix_dependency"]["expected_points"] == 18
    assert (
        matrix["release_matrix_dependency"]
        ["low_level_matrix_can_substitute_for_release_matrix"]
        is False
    )


@pytest.mark.parametrize(
    "claim", ["fsr06_closure", "wp6_closure", "q3_closure"]
)
def test_wp6_rejects_every_premature_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp6_rejects_unknown_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported WP-6 requested claim"):
        runner.requested_claim(
            ["--requested-claim", "unregistered_claim"]
        )


def test_wp6_matrix_byte_drift_is_rejected(tmp_path):
    runner = load_runner()
    document = MATRIX_PATH.read_text(encoding="utf-8")
    path = tmp_path / MATRIX_PATH.name
    path.write_text(document + "\n", encoding="utf-8")
    runner.DEFAULT_REGISTRY = path
    runner.strict_runner.DEFAULT_REGISTRY = path

    with pytest.raises(ValueError, match="frozen registry bytes changed"):
        runner.load_registry(path)


def test_wp6_distributed_boundary_is_two_rank_only():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    groups = {group["id"]: group for group in matrix["groups"]}

    for group_id in (
        "phase_operator_partition_mpi",
        "phase_artifact_collective_mpi",
    ):
        assert groups[group_id]["mpi_ranks"] == 2
        assert groups[group_id]["gtest_output_copies"] == 2
    assert matrix["known_partition_limit"] == runner.EXPECTED_PARTITION_LIMIT
    assert (
        matrix["known_partition_limit"]["four_rank_disposition"]
        == "REQUIRED_NOT_CLAIMED"
    )


def test_wp6_matrix_has_no_duplicate_json_keys():
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


def test_wp6_cli_rejects_closure_before_execution_argument_parsing(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp6_closure",
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


def test_wp6_validate_only_reports_prerequisite_without_closure():
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
    assert summary["group_count"] == 6
    assert summary["test_count"] == 51
    assert summary["prospective_test_count"] == 0
    assert summary["serial_quantitative_gate_count"] == 40
    assert summary["unqualified_campaign_count"] == 10
    assert summary["release_matrix_expected_points"] == 18
