#!/usr/bin/env python3
"""Run the frozen WP-5 contact-line prerequisite qualification matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. Requests for
FSR-05, WP-5, or Q4 closure fail before build or test execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import run_free_surface_wp2_geometry_qualification as strict_runner  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp5_contact_line_qualification_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "80b9c62256566ae39193a091171fff67ab37dc169398f288a96f8e280de9ab18"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)
EXPECTED_ARCHITECTURE_RECORD = (
    "Documentation/free_surface_wp5_contact_line_architecture.md"
)
EXPECTED_AUDIT_BASIS = {
    "head_at_freeze": "6306165b436ba6d81b50a2e348025038bf049fa2",
    "dirty_tracked_sources_reviewed": True,
    "source_binary_correspondence_claimed_for_dirty_runs": False,
}
EXPECTED_CLOSURE_STATE = "OPEN_REQUIRED_METHOD_AND_PHYSICAL_CAMPAIGNS_NOT_EXECUTED"
EXPECTED_DISPOSITION = {
    "fsr04_closed": False,
    "fsr05_closed": False,
    "wp5_closed": False,
    "q4_closed": False,
}
EXPECTED_BUILD_TARGETS = {
    "geometry": "test_fe_geometry",
    "level_set": "test_fe_levelset",
    "systems": "test_fe_levelset",
    "assembly": "test_fe_levelset",
    "physics": "test_physics",
    "application": "test_application",
    "assembly_mpi": "test_fe_levelset_mpi",
    "application_mpi": "test_application_mpi",
}

EXPECTED_SIGN_CONVENTION = {
    "footprint_direction": "outward_from_wetted_footprint",
    "positive_contact_speed": "advancing",
    "negative_contact_speed": "receding",
    "dynamic_law": "V_CL=gamma*mobility*(cos(theta_e)-cos(theta_d))",
    "zero_velocity_residual_sign": "opposite_predicted_contact_speed",
}
EXPECTED_PARAMETER_CONTRACT = {
    "slip_length": "physical_length",
    "line_mobility": "physical_mobility",
    "numerical_wall_width": "independent_physical_length_when_enabled",
    "mesh_size": "discretization_length_not_a_contact_parameter",
    "sharp_matrix_numerical_wall_width": 0,
}
EXPECTED_STAGE_FIELDS = [
    "accepted_step",
    "accepted_time",
    "generalized_alpha_stage_time",
    "generalized_alpha_stage_fraction",
    "previous_state_revision",
    "endpoint_state_revision",
    "pre_maintenance_endpoint_state_revision",
    "stage_state_revision",
    "state_revision",
    "geometry_snapshot_revision",
    "wall_normal",
    "contact_line_tangent",
    "footprint_direction",
    "advancing_receding_state",
    "contact_position",
    "dynamic_angle",
    "contact_speed",
    "wall_slip_speed",
    "constitutive_residual",
    "line_friction_dissipation",
    "wall_slip_dissipation",
]
EXPECTED_THRESHOLD_BASIS_KEYS = {
    "prescribed_angle_error_degrees",
    "contact_displacement",
    "positive_phi_scale_difference",
    "distributed_coefficient_difference",
    "distributed_frame_difference",
    "history_rejection_count",
}
EXPECTED_GROUPS = {
    "contact_line_operator_and_history_serial": ("physics", 1, 1, 24),
    "contact_geometry_snapshot_serial": ("geometry", 1, 1, 1),
    "wall_aware_reinitialization_serial": ("level_set", 1, 1, 7),
    "accepted_contact_maintenance_serial": ("application", 1, 1, 7),
    "wall_aware_reinitialization_mpi": ("assembly_mpi", 2, 2, 2),
    "accepted_contact_frame_mpi": ("application_mpi", 2, 2, 2),
}
EXPECTED_QUANTITATIVE_EVIDENCE = {
    (
        "MovingDomainPhysics.FreeSurfaceDynamicContactStageHistoryPreservesLawAndFrameProvenance",
        "contact_snapshot_mismatch_rejected",
    ): ("integer", "equal", 1),
    (
        "LevelSetReinitialization.PrescribedAngleConstraintEnforcesTargetInTwoDimensions",
        "prescribed_target_angle_max_error_degrees",
    ): ("real", "less_than_or_equal", 1.0e-10),
    (
        "LevelSetReinitialization.PrescribedAngleConstraintEnforcesTargetInTwoDimensions",
        "prescribed_target_contact_displacement_max",
    ): ("real", "less_than_or_equal", 1.0e-12),
    (
        "LevelSetReinitialization.PrescribedAngleConstraintEnforcesTargetInThreeDimensions",
        "prescribed_target_angle_max_error_degrees",
    ): ("real", "less_than_or_equal", 1.0e-10),
    (
        "LevelSetReinitialization.PrescribedAngleConstraintEnforcesTargetInThreeDimensions",
        "prescribed_target_contact_displacement_max",
    ): ("real", "less_than_or_equal", 1.0e-12),
    (
        "LevelSetReinitialization.PrescribedAngleConstraintIsInvariantUnderPositivePhiScaling",
        "prescribed_target_phi_scale_max_difference",
    ): ("real", "less_than_or_equal", 1.0e-12),
    (
        "ApplicationDriverLevelSetWorkflows.PrescribedWallSnapshotDrivesEndpointReinitialization",
        "application_prescribed_target_angle_max_error_degrees",
    ): ("real", "less_than_or_equal", 1.0e-10),
    (
        "ApplicationDriverLevelSetWorkflows.PrescribedWallSnapshotDrivesEndpointReinitialization",
        "application_prescribed_target_contact_displacement_max",
    ): ("real", "less_than_or_equal", 1.0e-12),
}
EXPECTED_MPI_PROPERTIES = {
    (
        "LevelSetReinitializationMPI.PrescribedAngleProjectionIsPartitionInvariant",
        "prescribed_target_mpi_max_coefficient_difference",
    ): ("real", "less_than_or_equal", 1.0e-12),
    (
        "ApplicationDriverLevelSetWorkflowsMPI.AcceptedSnapshotPrescribedFrameIsPartitionInvariantAndConflictsFailClosed",
        "application_prescribed_frame_mpi_max_difference",
    ): ("real", "equal", 0),
}
EXPECTED_UNQUALIFIED_CAMPAIGNS = {
    "four_advancing_receding_cases_three_meshes_three_time_steps",
    "five_angle_bottom_and_side_wall_sessile_relaxation",
    "reusken_spreading_and_contracting_drops",
    "resolved_slip_dynamic_wetting_at_ratios_2_4_8",
    "wall_width_ratios_0_0p5_1_2_independent_limit",
    "public_capillary_rise_uncertainty_comparison",
    "representative_partition_sweeps_after_low_level_mpi_gate",
}
EXPECTED_SCOPE = (
    "Low-level WP-5 prerequisite evidence only; this matrix does not close "
    "FSR-05, WP-5, or Q4."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr05_closure",
        "wp5_closure",
        "q4_closure",
    ],
    "diagnostic": (
        "The frozen low-level slice cannot establish physical refinement, "
        "benchmark uncertainty, or full dynamic-wetting closure."
    ),
}


def validate_wp5_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported WP-5 matrix schema")
    if registry.get("matrix_id") != "free_surface_wp5_contact_line_v1":
        raise ValueError("WP-5 matrix id changed after freeze")
    if registry.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise ValueError("WP-5 matrix status changed after freeze")
    if registry.get("work_package") != "WP-5":
        raise ValueError("WP-5 work-package label is invalid")
    if registry.get("findings") != ["FSR-04", "FSR-05"]:
        raise ValueError("WP-5 finding set changed after freeze")
    if registry.get("milestone") != "Q4":
        raise ValueError("WP-5 milestone changed after freeze")
    if registry.get("architecture_record") != EXPECTED_ARCHITECTURE_RECORD:
        raise ValueError("WP-5 architecture-record path changed after freeze")
    architecture = SOURCE_ROOT / EXPECTED_ARCHITECTURE_RECORD
    if not architecture.is_file() or architecture.is_symlink():
        raise ValueError("WP-5 architecture record is missing")
    if registry.get("audit_basis") != EXPECTED_AUDIT_BASIS:
        raise ValueError("WP-5 audit basis changed after freeze")
    if registry.get("closure_state") != EXPECTED_CLOSURE_STATE:
        raise ValueError("WP-5 closure state must remain open")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("WP-5 qualification disposition must remain open")
    if registry.get("build_targets") != EXPECTED_BUILD_TARGETS:
        raise ValueError("WP-5 build-target mapping changed after freeze")
    if registry.get("prospective_tests") != []:
        raise ValueError("WP-5 cannot freeze with prospective tests")
    if registry.get("closure_request_policy") != EXPECTED_CLOSURE_REQUEST_POLICY:
        raise ValueError("WP-5 closure-request policy changed after freeze")
    if registry.get("sign_convention") != EXPECTED_SIGN_CONVENTION:
        raise ValueError("WP-5 sign convention changed after freeze")
    if registry.get("independent_parameter_contract") != EXPECTED_PARAMETER_CONTRACT:
        raise ValueError("WP-5 independent-parameter contract changed after freeze")
    if registry.get("accepted_stage_record_contract") != EXPECTED_STAGE_FIELDS:
        raise ValueError("WP-5 accepted-stage record contract is incomplete")

    threshold_basis = registry.get("numeric_threshold_basis")
    if (
        not isinstance(threshold_basis, dict)
        or set(threshold_basis) != EXPECTED_THRESHOLD_BASIS_KEYS
        or any(
            not isinstance(value, str) or not value.strip()
            for value in threshold_basis.values()
        )
    ):
        raise ValueError("WP-5 numeric threshold basis is incomplete")

    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("WP-5 execution groups are missing")
    actual_groups: dict[str, tuple[str, int, int, int]] = {}
    all_properties: dict[tuple[str, str], tuple[str, str, int | float]] = {}
    all_tests: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("WP-5 execution group must be an object")
        group_id = group.get("id")
        tests = group.get("tests")
        if not isinstance(group_id, str) or not isinstance(tests, list):
            raise ValueError("WP-5 execution group id or tests are invalid")
        if group_id in actual_groups:
            raise ValueError(f"duplicate WP-5 execution group: {group_id}")
        actual_groups[group_id] = (
            group.get("binary"),
            group.get("mpi_ranks"),
            group.get("gtest_output_copies"),
            len(tests),
        )
        for test in tests:
            if not isinstance(test, str) or not re.fullmatch(
                r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", test
            ):
                raise ValueError(f"invalid WP-5 test name in {group_id}")
            if test in all_tests:
                raise ValueError(f"duplicate WP-5 test across groups: {test}")
            all_tests.add(test)
        for property_contract in group.get("recorded_properties", []):
            if not isinstance(property_contract, dict) or set(property_contract) != {
                "test",
                "property",
                "type",
                "relation",
                "threshold",
            }:
                raise ValueError(f"invalid WP-5 property contract in {group_id}")
            key = (
                property_contract.get("test"),
                property_contract.get("property"),
            )
            if key in all_properties:
                raise ValueError(f"duplicate WP-5 property contract: {key}")
            if key[0] not in tests:
                raise ValueError(
                    f"WP-5 property is not owned by its execution group: {key}"
                )
            all_properties[key] = (
                property_contract.get("type"),
                property_contract.get("relation"),
                property_contract.get("threshold"),
            )
    if actual_groups != EXPECTED_GROUPS:
        raise ValueError("WP-5 execution groups changed after freeze")
    if all_properties != EXPECTED_MPI_PROPERTIES:
        raise ValueError("WP-5 distributed numeric gates changed after freeze")

    quantitative_evidence = registry.get("quantitative_evidence")
    if not isinstance(quantitative_evidence, list):
        raise ValueError("WP-5 quantitative evidence list is missing")
    actual_quantitative: dict[tuple[str, str], tuple[str, str, int | float]] = {}
    for evidence in quantitative_evidence:
        if not isinstance(evidence, dict) or set(evidence) != {
            "test",
            "property",
            "type",
            "relation",
            "threshold",
        }:
            raise ValueError("invalid WP-5 quantitative evidence entry")
        key = (evidence.get("test"), evidence.get("property"))
        if key in actual_quantitative:
            raise ValueError(f"duplicate WP-5 quantitative evidence: {key}")
        actual_quantitative[key] = (
            evidence.get("type"),
            evidence.get("relation"),
            evidence.get("threshold"),
        )
    if actual_quantitative != EXPECTED_QUANTITATIVE_EVIDENCE:
        raise ValueError("WP-5 quantitative gates changed after freeze")

    unresolved = registry.get("unqualified_required_campaigns")
    if not isinstance(unresolved, list):
        raise ValueError("WP-5 unqualified campaign list is missing")
    unresolved_ids: set[str] = set()
    for entry in unresolved:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or entry.get("status") != "REQUIRED_NOT_CLAIMED"
            or not isinstance(entry.get("id"), str)
        ):
            raise ValueError("WP-5 unqualified campaign entry is invalid")
        unresolved_ids.add(entry["id"])
    if unresolved_ids != EXPECTED_UNQUALIFIED_CAMPAIGNS or len(unresolved_ids) != len(
        unresolved
    ):
        raise ValueError("WP-5 unqualified campaign list changed after freeze")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-5 qualification scope changed after freeze")
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = "free_surface_wp5_contact_line_v1"
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-5"
strict_runner.__doc__ = __doc__

_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_registry(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("WP-5 frozen matrix is unavailable")
    resolved = path.resolve()
    if resolved != DEFAULT_REGISTRY.resolve():
        raise ValueError("WP-5 requires the canonical frozen matrix")
    if not resolved.is_file():
        raise ValueError("WP-5 frozen matrix is unavailable")
    if strict_runner.sha256_file(resolved) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-5 frozen matrix bytes changed")
    json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return validate_wp5_contract(_shared_load_registry(resolved))


def write_json(path: Path, value: Any) -> None:
    if strict_runner.sha256_file(SHARED_RUNNER_PATH) != SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification runner changed during execution")
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "summary.json",
        "final_provenance.json",
    }:
        value = copy.deepcopy(value)
        value["shared_runner_dependency"] = {
            "path": str(SHARED_RUNNER_PATH),
            "sha256": SHARED_RUNNER_SHA256,
        }
        value["qualification_scope"] = EXPECTED_SCOPE
        value["closure_state"] = EXPECTED_CLOSURE_STATE
        value["qualification_disposition"] = EXPECTED_DISPOSITION
        value["fsr05_closure_claimed"] = False
        value["wp5_closure_claimed"] = False
        value["q4_closure_claimed"] = False
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-5 contact-line prerequisite qualification record",
            1,
        )
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            + "Closure state: `"
            + EXPECTED_CLOSURE_STATE
            + "`. FSR-05, WP-5, and Q4 remain open.\n"
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def _tests_for_binary(
    registry: dict[str, Any],
    binary_key: str,
) -> list[str]:
    return [
        test
        for group in registry["groups"]
        if group["binary"] == binary_key
        for test in group["tests"]
    ]


def _listed_gtests(binary: Path) -> set[str]:
    result = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    suite = ""
    tests: set[str] = set()
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            suite = line.split("#", 1)[0].strip().removesuffix(".")
            continue
        test = line.split("#", 1)[0].strip()
        if suite and test:
            tests.add(f"{suite}.{test}")
    return tests


def requested_claim(
    arguments: list[str],
) -> tuple[str, bool, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-5 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this low-level matrix may establish.\n"
            "      fsr05_closure, wp5_closure, and q4_closure are rejected.\n"
            "  --validate-only\n"
            "      Validate the frozen schema and claim boundary without builds.\n"
            "  --list-only --geometry-binary PATH "
            "--level-set-binary PATH --physics-binary PATH\n"
            "      --application-binary PATH "
            "--assembly-mpi-binary PATH "
            "--application-mpi-binary PATH\n"
            "      Check exact frozen test discovery without executing tests "
            "or writing artifacts.\n"
        )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--requested-claim",
        default=EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"],
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.validate_only and parsed.list_only:
        raise ValueError("--validate-only and --list-only are mutually exclusive")
    claim = parsed.requested_claim
    allowed = EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]
    rejected = set(EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"])
    if claim in rejected:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != allowed:
        raise ValueError(
            f"unsupported WP-5 requested claim {claim!r}; expected {allowed!r}"
        )
    return claim, parsed.validate_only, parsed.list_only, remaining


def _validation_summary(
    registry: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    return {
        "matrix_id": registry["matrix_id"],
        "matrix_sha256": EXPECTED_REGISTRY_SHA256,
        "status": registry["status"],
        "requested_claim": claim,
        "prospective_test_count": len(registry["prospective_tests"]),
        "group_count": len(registry["groups"]),
        "test_count": sum(len(group["tests"]) for group in registry["groups"]),
        "serial_quantitative_gate_count": len(registry["quantitative_evidence"]),
        "unqualified_campaign_count": len(registry["unqualified_required_campaigns"]),
        "closure_state": EXPECTED_CLOSURE_STATE,
        **EXPECTED_DISPOSITION,
        "outcome": "PASS_PREREQUISITE_NONCLOSURE",
    }


def _run_validate_only(claim: str, remaining: list[str]) -> int:
    if remaining:
        raise ValueError("--validate-only does not accept execution arguments")
    registry = load_registry(DEFAULT_REGISTRY)
    print(json.dumps(_validation_summary(registry, claim), sort_keys=True))
    return 0


def _parse_list_binary_arguments(
    arguments: list[str],
) -> dict[str, Path]:
    parser = argparse.ArgumentParser(
        prog=f"{SCRIPT_PATH.name} --list-only",
    )
    parser.add_argument("--geometry-binary", type=Path, required=True)
    parser.add_argument("--level-set-binary", type=Path, required=True)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parser.add_argument("--assembly-mpi-binary", type=Path, required=True)
    parser.add_argument("--application-mpi-binary", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    return {
        "geometry": parsed.geometry_binary.resolve(),
        "level_set": parsed.level_set_binary.resolve(),
        "physics": parsed.physics_binary.resolve(),
        "application": parsed.application_binary.resolve(),
        "assembly_mpi": parsed.assembly_mpi_binary.resolve(),
        "application_mpi": parsed.application_mpi_binary.resolve(),
    }


def _require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"test binary is not executable: {path}")


def _run_list_only(claim: str, remaining: list[str]) -> int:
    binaries = _parse_list_binary_arguments(remaining)
    for binary in binaries.values():
        _require_executable(binary)
    registry = load_registry(DEFAULT_REGISTRY)
    missing_by_binary: dict[str, list[str]] = {}
    expected_counts: dict[str, int] = {}
    listed_expected_counts: dict[str, int] = {}
    listed_total_counts: dict[str, int] = {}
    binary_hashes: dict[str, str] = {}
    for key, binary in binaries.items():
        expected = set(_tests_for_binary(registry, key))
        listed = _listed_gtests(binary)
        missing_by_binary[key] = sorted(expected - listed)
        expected_counts[key] = len(expected)
        listed_expected_counts[key] = len(expected & listed)
        listed_total_counts[key] = len(listed)
        binary_hashes[key] = strict_runner.sha256_file(binary)
    missing = any(missing_by_binary.values())
    print(
        json.dumps(
            {
                **_validation_summary(registry, claim),
                "binary_sha256": binary_hashes,
                "expected_test_count_by_binary": expected_counts,
                "listed_expected_test_count": listed_expected_counts,
                "listed_total_test_count": listed_total_counts,
                "missing_tests": missing_by_binary,
                "tests_executed": 0,
                "artifacts_written": 0,
                "outcome": ("FAIL" if missing else "PASS_PREREQUISITE_NONCLOSURE"),
            },
            sort_keys=True,
        )
    )
    return 2 if missing else 0


if __name__ == "__main__":
    try:
        (
            _claim,
            _validate_only,
            _list_only,
            _remaining_arguments,
        ) = requested_claim(sys.argv[1:])
        if _validate_only:
            raise SystemExit(_run_validate_only(_claim, _remaining_arguments))
        if _list_only:
            raise SystemExit(_run_list_only(_claim, _remaining_arguments))
        sys.argv = [sys.argv[0], *_remaining_arguments]
        raise SystemExit(strict_runner.main())
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}", file=strict_runner.sys.stderr)
        raise SystemExit(2)
