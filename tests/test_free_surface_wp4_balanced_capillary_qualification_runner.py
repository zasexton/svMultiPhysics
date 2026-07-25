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
    / "run_free_surface_wp4_balanced_capillary_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp4_balanced_capillary_prerequisite_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp4_balanced_capillary_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_wp4_matrix_is_frozen_but_explicitly_incomplete():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["qualification_scope"] == runner.EXPECTED_SCOPE
    assert matrix["prospective_tests"] == []
    assert len(matrix["unqualified_required_campaigns"]) == 10
    assert all(
        entry["status"] == "REQUIRED_NOT_CLAIMED"
        for entry in matrix["unqualified_required_campaigns"]
    )
    assert matrix["method_boundary"]["selected_ad2_method"] == "UNSELECTED"
    assert matrix["method_boundary"]["balanced_force_evidence_claimed"] is False
    groups = {group["id"]: group for group in matrix["groups"]}
    assert groups["surface_wall_volume_functional_variation_serial"][
        "tests"
    ] == [
        "FreeSurfaceGeometrySnapshot."
        "DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "FreeSurfaceGeometrySnapshot."
        "DiscreteFunctionalFirstVariationMatchesThreeDimensionalCentralDifference",
    ]
    assert all(
        value is False
        for value in matrix["qualification_disposition"].values()
    )


@pytest.mark.parametrize(
    "claim",
    ["fsr03_closure", "fsr04_closure", "wp4_closure", "q2_closure"],
)
def test_wp4_rejects_every_premature_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp4_rejects_unknown_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported WP-4 requested claim"):
        runner.requested_claim(
            ["--requested-claim", "unregistered_claim"]
        )


def test_wp4_matrix_byte_drift_is_rejected(tmp_path):
    runner = load_runner()
    document = MATRIX_PATH.read_text(encoding="utf-8")
    path = tmp_path / MATRIX_PATH.name
    path.write_text(document + "\n", encoding="utf-8")
    runner.DEFAULT_REGISTRY = path
    runner.strict_runner.DEFAULT_REGISTRY = path

    with pytest.raises(ValueError, match="frozen registry bytes changed"):
        runner.load_registry(path)


def test_wp4_distributed_boundary_is_configuration_only_and_two_rank():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    groups = {group["id"]: group for group in matrix["groups"]}

    distributed = groups["static_pressure_configuration_mpi"]
    assert distributed["mpi_ranks"] == 2
    assert distributed["gtest_output_copies"] == 2
    assert distributed["tests"] == [
        "TimeLoopFsilsConvergenceMPI."
        "StaticPressureInitializerConfigurationMismatchFailsCollectively"
    ]
    assert (
        "flat_interface_direction_phase_gravity_gauge_cut_and_mpi_matrix"
        in {
            entry["id"]
            for entry in matrix["unqualified_required_campaigns"]
        }
    )


def test_wp4_matrix_has_no_duplicate_json_keys():
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


def test_wp4_cli_rejects_closure_before_execution_argument_parsing(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp4_closure",
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


def test_wp4_validate_only_reports_prerequisite_without_closure():
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
    assert summary["group_count"] == 5
    assert summary["test_count"] == 26
    assert summary["prospective_test_count"] == 0
    assert summary["serial_quantitative_gate_count"] == 40
    assert summary["unqualified_campaign_count"] == 10
    assert summary["fsr03_closed"] is False
    assert summary["fsr04_closed"] is False
    assert summary["wp4_closed"] is False
    assert summary["q2_closed"] is False
