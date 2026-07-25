import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
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
        / "run_free_surface_wp7_cut_stability_qualification.py"
    )
    spec = importlib.util.spec_from_file_location(
        "free_surface_wp7_cut_stability_qualification_runner", script
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
        / "free_surface_wp7_cut_stability_qualification_matrix.json"
    )


def test_wp7_mpi_groups_use_root_only_output_and_frozen_timeouts():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())
    groups = {group["id"]: group for group in registry["groups"]}

    for group_id, ranks, wall_time_seconds in (
        ("wp7_partition_mpi_2", 2, 43200),
        ("wp7_partition_mpi_4", 4, 43200),
    ):
        group = groups[group_id]
        assert group["mpi_ranks"] == ranks
        assert group["gtest_output_copies"] == 1
        assert group["execution"]["wall_time_seconds"] == wall_time_seconds


@pytest.mark.parametrize(
    ("group_id", "field", "value", "message"),
    [
        (
            "wp7_partition_mpi_2",
            "gtest_output_copies",
            2,
            "two-rank group has the wrong output count",
        ),
        (
            "wp7_partition_mpi_4",
            "gtest_output_copies",
            4,
            "four-rank group has the wrong output count",
        ),
    ],
)
def test_wp7_contract_rejects_per_rank_output_counts(group_id, field, value, message):
    runner = _load_runner()
    registry = json.loads(_registry_path().read_text(encoding="utf-8"))
    mutated = copy.deepcopy(registry)
    group = next(item for item in mutated["groups"] if item["id"] == group_id)
    group[field] = value

    with pytest.raises(ValueError, match=message):
        runner.validate_wp7_contract(mutated)


@pytest.mark.parametrize("group_id", ["wp7_partition_mpi_2", "wp7_partition_mpi_4"])
def test_wp7_contract_rejects_mpi_timeout_drift(group_id):
    runner = _load_runner()
    registry = json.loads(_registry_path().read_text(encoding="utf-8"))
    mutated = copy.deepcopy(registry)
    group = next(item for item in mutated["groups"] if item["id"] == group_id)
    group["execution"]["wall_time_seconds"] = 7199

    with pytest.raises(ValueError, match="wrong wall-time limit"):
        runner.validate_wp7_contract(mutated)


def test_physics_ctest_registers_frozen_mpi_filters_and_timeouts():
    cmake = (
        _repository() / "Code" / "Source" / "solver" / "Physics" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "NAME Physics_FreeSurfaceSharpBoundary_MPI_2" in cmake
    assert (
        "GeneratedActiveBoundaryDomainMPI."
        "WetFractionSweepIsOwnershipUniqueAndPartitionIndependent"
    ) in cmake
    assert (
        "FreeSurfaceSharpBoundaryOperatorsMPI.OperatorWorkIsPartitionIndependent"
    ) in cmake
    assert (
        "FreeSurfaceSharpBoundaryOperatorsMPI."
        "StructuredChannelWorkIsInvariantUnderActualRepartition"
    ) in cmake
    assert "--gtest_filter=FreeSurfaceCutStabilityMPI.*" not in cmake
    assert (
        "--gtest_filter="
        "FreeSurfaceCutStabilityMPI."
        "PhysicalWetBlocksAndDisconnectedIslandsAreInvariantAcrossDryMPIData:"
        "FreeSurfaceCutStabilityMPI."
        "DistributedMovingCutRemainsStableAcrossBlockAndMetisPartitions:"
        "FreeSurfaceCutStabilityMPI."
        "LimitedMetisHaloFailsClosedOnIncompleteAggregationSupport:"
        "FreeSurfaceCutStabilityMPI."
        "TwoRankFractionOrientationRegimeMatrixMatchesSerial"
    ) in cmake
    assert "NAME Physics_FreeSurfaceCutStability_MPI_4" in cmake
    assert (
        "--gtest_filter="
        "FreeSurfaceCutStabilityMPI."
        "FourRankFixedCutIsInvariantAcrossBlockAndMetisPartitions:"
        "FreeSurfaceCutStabilityMPI."
        "FourRankFractionOrientationRegimeMatrixMatchesSerial"
    ) in cmake
    assert (
        "set_tests_properties(Physics_FreeSurfaceCutStability_MPI_2 "
        "PROPERTIES\n            TIMEOUT 43200"
    ) in cmake
    assert (
        "set_tests_properties(Physics_FreeSurfaceCutStability_MPI_4 "
        "PROPERTIES\n            TIMEOUT 43200"
    ) in cmake


def test_wp7_real_properties_are_serialized_without_integer_truncation():
    source = (
        _repository()
        / "Code"
        / "Source"
        / "solver"
        / "Physics"
        / "Tests"
        / "Unit"
        / "test_FreeSurfaceCutStability.cpp"
    ).read_text(encoding="utf-8")
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())

    assert "std::numeric_limits<FE::Real>::max_digits10" in source
    for gate in registry["quantitative_evidence"]:
        if gate["type"] != "real":
            continue
        pattern = re.compile(
            r"RecordProperty\(\s*"
            + re.escape(f'"{gate["property"]}"')
            + r"\s*,\s*realPropertyValue\(",
            re.MULTILINE,
        )
        assert pattern.search(source), gate["property"]


def test_wp7_matrix_is_frozen_and_explicitly_blocked():
    runner = _load_runner()
    registry = runner.load_registry(_registry_path())

    assert registry["qualification_scope"] == runner.EXPECTED_SCOPE
    assert registry["closure_state"] == ("BLOCKED_BY_FROZEN_PROSPECTIVE_EVIDENCE")
    assert len(registry["executable_tests"]) == 14
    assert len(registry["prospective_tests"]) == 7
    assert all(
        value is False for value in registry["qualification_disposition"].values()
    )


@pytest.mark.parametrize("claim", ["fsr07_closure", "wp7_closure", "q1_closure"])
def test_wp7_rejects_every_premature_closure_claim(claim):
    runner = _load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp7_rejects_unknown_claim():
    runner = _load_runner()

    with pytest.raises(ValueError, match="unsupported WP-7 requested claim"):
        runner.requested_claim(["--requested-claim", "unregistered_claim"])


def test_wp7_matrix_byte_drift_is_rejected(tmp_path):
    runner = _load_runner()
    document = _registry_path().read_text(encoding="utf-8")
    path = tmp_path / _registry_path().name
    path.write_text(document + "\n", encoding="utf-8")
    runner.DEFAULT_REGISTRY = path
    runner.strict_runner.DEFAULT_REGISTRY = path

    with pytest.raises(ValueError, match="frozen registry bytes changed"):
        runner.load_registry(path)


def test_wp7_cli_rejects_closure_before_execution_argument_parsing(tmp_path):
    runner_path = (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "run_free_surface_wp7_cut_stability_qualification.py"
    )
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--requested-claim",
            "wp7_closure",
            "--output",
            str(output),
        ],
        cwd=_repository(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "outside this matrix" in result.stderr
    assert not output.exists()


def test_wp7_validate_only_reports_blocked_prerequisite():
    runner_path = (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "run_free_surface_wp7_cut_stability_qualification.py"
    )
    result = subprocess.run(
        [sys.executable, str(runner_path), "--validate-only"],
        cwd=_repository(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS"
    assert summary["requested_claim"] == "low_level_prerequisite"
    assert summary["closure_state"] == ("BLOCKED_BY_FROZEN_PROSPECTIVE_EVIDENCE")
    assert summary["group_count"] == 5
    assert summary["test_count"] == 21
    assert summary["executable_test_count"] == 14
    assert summary["prospective_test_count"] == 7
    assert summary["serial_quantitative_gate_count"] == 49
    assert summary["fsr07_closed"] is False
    assert summary["wp7_closed"] is False
    assert summary["q1_closed"] is False
