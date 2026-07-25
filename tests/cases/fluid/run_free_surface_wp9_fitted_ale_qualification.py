#!/usr/bin/env python3
"""Run the frozen WP-9 fitted-ALE prerequisite/non-closure matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. FSR-10,
FSR-11, WP-9, Q4, and general fitted-ALE qualification claims fail before
execution arguments are parsed or artifact paths are created.
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

import run_free_surface_configuration_qualification as base_runner  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MATRIX = SCRIPT_PATH.with_name(
    "free_surface_wp9_fitted_ale_qualification_matrix.json"
)
EXPECTED_MATRIX_SHA256 = (
    "bf3d0a2d9f8fc9d9530ae916460a07460181de63deffc6f735bedb4f57c76123"
)
SHARED_RUNNER_PATH = Path(base_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = base_runner.sha256_file(SHARED_RUNNER_PATH)

EXPECTED_MATRIX_ID = "free_surface_wp9_fitted_ale_prerequisite_v2"
EXPECTED_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_ARCHITECTURE_RECORD = (
    "Documentation/free_surface_wp9_fitted_ale_architecture.md"
)
EXPECTED_SCOPE = (
    "Low-level WP-9 prerequisite and non-closure evidence only; this matrix "
    "does not close FSR-10, FSR-11, WP-9, or Q4 and does not qualify "
    "physical fitted-ALE campaigns."
)
EXPECTED_CLOSURE_STATE = "OPEN_METHOD_AND_PHYSICAL_ALE_CAMPAIGNS_REQUIRED"
EXPECTED_AUDIT_BASIS = {
    "head_at_freeze": "963cb4c256a2e7db7dede519ea8dafe5a52aacd0",
    "dirty_tracked_sources_reviewed": True,
    "source_binary_correspondence_claimed_for_dirty_runs": False,
}
EXPECTED_CONFIGURATION_CONTRACT = {
    "schema_2": {
        "qualification": "supported_configuration_envelope",
        "accepted_consumed_path": {
            "ale_enabled": True,
            "mesh_velocity_source": "CoupledDisplacement",
            "normal_policy": "MatchFluidNormalVelocity",
            "normal_enforcement": ["Penalty", "Nitsche"],
            "tangential_policy": "Prescribed",
            "policy_consumed": True,
        },
        "rejected_before_system_mutation": [
            "ALE_disabled",
            "PrescribedData_mesh_velocity_source",
            "normal_policy_None",
            "normal_enforcement_None",
            "tangential_policy_Free",
            "tangential_policy_SmoothingOnly",
            "fitted_DynamicRenE_contact_model",
        ],
        "kinematic_penalty_auto_promotes_none": False,
    },
    "schema_1": {
        "qualification": "unqualified_explicit_legacy",
        "explicit_opt_in_required": True,
        "modes_retained_for_regression_only": [
            "Free",
            "SmoothingOnly",
            "Prescribed",
            "PrescribedData_mesh_velocity",
        ],
        "supported_capability_claimed": False,
    },
}
EXPECTED_SUPPORTED_SLICE = {
    "input_schema": 2,
    "ale_mesh_velocity_source": "CoupledDisplacement",
    "normal_policy": "MatchFluidNormalVelocity",
    "normal_enforcement": ["Penalty", "Nitsche"],
    "tangential_policy": "Prescribed",
    "prescribed_tangential_enforcement": (
        "weak_current_geometry_projected_velocity_penalty"
    ),
    "tangential_owner_registry": (
        "FESystem_mesh_displacement_field_and_boundary_marker"
    ),
    "fitted_surface_tension_form": "CurvatureTraction",
    "fitted_contact_models": ["None", "Pinned"],
}
EXPECTED_PROVENANCE_CONTRACT = {
    "owner_source": ("matching_central_mesh_tangential_policy_declaration"),
    "consumption_source": ("matching_fitted_prescribed_tangential_boundary_descriptor"),
    "required_consumed_fields": [
        "tangential_mesh_owner",
        "policy_consumed",
        "operator_tag",
        "operator_source",
        "policy_qualification",
    ],
    "unconsumed_representation": {
        "policy_consumed": False,
        "operator_tag": None,
        "operator_source": None,
    },
    "hardcoded_owner_claim_allowed": False,
}
EXPECTED_BINARY_BY_SUITE = {
    "EquationTranslatorMeshMotion": "application",
    "EquationTranslatorFreeSurface": "application",
    "NavierStokesLegacyBCs": "physics",
    "MovingDomainPhysics": "physics",
}
EXPECTED_SOURCE_TEST_FILES = {
    "EquationTranslatorMeshMotion": (
        "Code/Source/solver/Application/Tests/Unit/test_EquationTranslator.cpp"
    ),
    "EquationTranslatorFreeSurface": (
        "Code/Source/solver/Application/Tests/Unit/test_EquationTranslator.cpp"
    ),
    "NavierStokesLegacyBCs": (
        "Code/Source/solver/Physics/Tests/Unit/test_NavierStokesLegacyBCs.cpp"
    ),
    "MovingDomainPhysics": (
        "Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp"
    ),
}
EXPECTED_TEST_GROUPS = {
    "xml_translation_and_truthful_provenance": [
        (
            "EquationTranslatorMeshMotion."
            "XmlAliasesReachTangentialPolicyModuleRegistration"
        ),
        (
            "EquationTranslatorFreeSurface."
            "XmlTangentialPenaltyAliasesReachTruthfulFittedModule"
        ),
        (
            "EquationTranslatorFreeSurface."
            "XmlExplicitNoneCannotBePromotedByKinematicPenalty"
        ),
        ("NavierStokesLegacyBCs.FittedFreeSurfaceBCTranslation_SetupSucceeds"),
        (
            "NavierStokesLegacyBCs."
            "FittedFreeSurfaceKinematicBCTranslation_UsesCurrentGeometry"
        ),
        (
            "NavierStokesLegacyBCs."
            "FittedFreeSurfacePrescribedTangentialMeshPolicyTranslation"
        ),
        (
            "MovingDomainPhysics."
            "NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults"
        ),
    ],
    "schema_2_supported_prescribed_and_mode_provenance": [
        ("MovingDomainPhysics.FittedFreeSurfaceQualifiedContractRejectsBeforeMutation"),
        (
            "MovingDomainPhysics."
            "FittedFreeSurfaceTangentialPoliciesRegisterCoupledMeshOwnership"
        ),
        (
            "MovingDomainPhysics."
            "FittedFreeSurfacePrescribedTangentialVelocityProjectsOutNormalTarget"
        ),
        ("MovingDomainPhysics.CoupledFittedFreeSurfaceALEAndHarmonicMeshMotionSetup"),
    ],
    "fail_closed_capability_and_owner_conflict": [
        (
            "MovingDomainPhysics."
            "FittedFreeSurfacePrescribedTangentialVelocityRequiresCoupledDisplacement"
        ),
        (
            "MovingDomainPhysics."
            "FittedFreeSurfaceRejectsMeshMotionTangentialOwnerInEitherOrder"
        ),
        (
            "MovingDomainPhysics."
            "FittedPinnedContactLineRejectsPrescribedALEBeforeSystemMutation"
        ),
        (
            "MovingDomainPhysics."
            "FittedPrescribedContactAngleFailsClosedWithoutContactLineIntegration"
        ),
        ("MovingDomainPhysics.FittedPrescribedContactAngleFailsClosedWithoutALEToo"),
        "MovingDomainPhysics.NavierStokesFittedSurfaceStressFailsClosed",
        (
            "NavierStokesLegacyBCs."
            "FittedFreeSurfacePrescribedAngleTranslationFailsClosed"
        ),
        (
            "EquationTranslatorFreeSurface."
            "XmlFittedDynamicContactFailsClosedBeforeSystemMutation"
        ),
    ],
    "schema_1_unqualified_legacy_operator_regressions": [
        (
            "MovingDomainPhysics."
            "FittedFreeSurfaceLegacyPrescribedDataReportsUnconsumedPolicy"
        ),
        (
            "MovingDomainPhysics."
            "NavierStokesFittedFreeSurfaceALEUsesCurrentBoundaryGeometry"
        ),
        (
            "MovingDomainPhysics."
            "NavierStokesFittedFreeSurfaceCanUseCurrentGeometryCurvature"
        ),
        (
            "MovingDomainPhysics."
            "NavierStokesFittedFreeSurfacePenaltyKinematicsAddsBoundaryResidual"
        ),
        (
            "MovingDomainPhysics."
            "NavierStokesFittedFreeSurfaceNitscheKinematicsAddsBoundaryResidual"
        ),
        (
            "MovingDomainPhysics."
            "FittedFreeSurfaceNitschePoliciesAreBoundaryLocalAndOrderInvariant"
        ),
        ("MovingDomainPhysics.FittedFreeSurfaceTangentialPoliciesAreBoundaryLocal"),
    ],
    "mesh_motion_policy_consumers": [
        ("MovingDomainPhysics.HarmonicMeshMotionTangentialPoliciesSelectBoundaryTerms"),
        (
            "MovingDomainPhysics."
            "MeshMotionModulesInstallEquivalentBoundaryConditionDescriptors"
        ),
    ],
    "pinned_capability_boundary": [
        ("MovingDomainPhysics.FittedPinnedContactLineConstrainsMeshDisplacement"),
        "MovingDomainPhysics.FittedPinnedContactLineRequiresALE",
    ],
}
EXPECTED_METHOD_EXITS = {
    "free_and_smoothing_supported_operator_contract",
    "tangential_penalty_dimensional_h_dt_and_polynomial_scaling",
    "coupled_fluid_mesh_consistency_stability_and_work_argument",
    "actual_boundary_mesh_velocity_projection_error_and_surface_work_history",
    "serialized_policy_history_and_restart_continuity",
    "rotation_numbering_and_mpi_partition_equivalence",
    "explicit_fitted_dynamic_contact_rejection_and_capability_provenance",
    "normal_mesh_constraint_and_fluid_kinematic_compatibility_matrix",
    "geometric_conservation_volume_surface_work_and_mesh_quality_metric_contract",
}
EXPECTED_SIMULATION_EXITS = {
    "flat_translating_ale_interface",
    "prescribed_tangential_shear",
    "fitted_sloshing",
}
EXPECTED_DISPOSITION = {
    "fsr10_closed": False,
    "fsr11_closed": False,
    "wp9_closed": False,
    "q4_closed": False,
    "physical_fitted_ale_qualified": False,
}
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr10_closure",
        "fsr11_closure",
        "wp9_closure",
        "q4_closure",
        "fitted_ale_qualified",
    ],
    "diagnostic": (
        "This matrix is prerequisite-only: schema-2 Free and SmoothingOnly "
        "have no consumed supported operator, method exits remain open, and "
        "no required physical fitted-ALE campaign has run."
    ),
}

EXPECTED_TESTS = [test for tests in EXPECTED_TEST_GROUPS.values() for test in tests]
EXPECTED_PREREQUISITE_CLAIMS = {
    "xml_aliases_and_explicit_none_fail_closed_are_wired_end_to_end",
    "schema_2_prescribed_is_the_only_consumed_supported_tangential_path",
    "schema_1_paths_are_explicitly_unqualified_and_can_report_no_owner_or_operator",
    "owner_conflicts_and_unsupported_fitted_capabilities_fail_closed",
}

_base_write_json = base_runner.write_json


def _validate_open_exits(
    value: Any,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"WP-9 {label} list is missing")
    identifiers: set[str] = set()
    for entry in value:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status", "contract"}
            or entry.get("status") != "REQUIRED_NOT_CLAIMED"
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("contract"), str)
            or not entry["contract"].strip()
        ):
            raise ValueError(f"WP-9 {label} entry is invalid")
        identifiers.add(entry["id"])
    if identifiers != expected or len(identifiers) != len(value):
        raise ValueError(f"WP-9 {label} list changed after freeze")


def _validate_source_definitions(matrix: dict[str, Any]) -> None:
    if matrix.get("source_test_files") != EXPECTED_SOURCE_TEST_FILES:
        raise ValueError("WP-9 source-test mapping changed after freeze")
    source_text: dict[str, str] = {}
    for suite, relative_path in EXPECTED_SOURCE_TEST_FILES.items():
        path = SOURCE_ROOT / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"WP-9 source-test file is unavailable: {path}")
        source_text[suite] = path.read_text(encoding="utf-8")

    for full_name in matrix["tests"]:
        suite, test = full_name.split(".", 1)
        if suite not in source_text:
            raise ValueError(f"WP-9 test suite has no frozen source: {suite}")
        pattern = re.compile(
            r"\bTEST\(\s*" + re.escape(suite) + r"\s*,\s*" + re.escape(test) + r"\s*\)",
            re.MULTILINE,
        )
        occurrences = len(pattern.findall(source_text[suite]))
        if occurrences != 1:
            raise ValueError(
                f"WP-9 source definition count for {full_name} is "
                f"{occurrences}, expected 1"
            )


def validate_wp9_contract(matrix: dict[str, Any]) -> dict[str, Any]:
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported WP-9 matrix schema")
    if matrix.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("WP-9 matrix id changed after freeze")
    if matrix.get("status") != EXPECTED_STATUS:
        raise ValueError("WP-9 matrix status changed after freeze")
    if matrix.get("work_package") != "WP-9":
        raise ValueError("WP-9 work-package label is invalid")
    if matrix.get("findings") != ["FSR-10", "FSR-11"]:
        raise ValueError("WP-9 finding set changed after freeze")
    if matrix.get("milestone") != "Q4":
        raise ValueError("WP-9 milestone changed after freeze")
    if matrix.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-9 qualification scope changed after freeze")
    if matrix.get("closure_state") != EXPECTED_CLOSURE_STATE:
        raise ValueError("WP-9 closure state must remain open")
    if matrix.get("architecture_record") != EXPECTED_ARCHITECTURE_RECORD:
        raise ValueError("WP-9 architecture-record path changed after freeze")
    architecture = SOURCE_ROOT / EXPECTED_ARCHITECTURE_RECORD
    if not architecture.is_file() or architecture.is_symlink():
        raise ValueError("WP-9 architecture record is missing")
    if matrix.get("audit_basis") != EXPECTED_AUDIT_BASIS:
        raise ValueError("WP-9 audit basis changed after freeze")
    if matrix.get("configuration_contract") != (EXPECTED_CONFIGURATION_CONTRACT):
        raise ValueError("WP-9 schema policy contract changed after freeze")
    if matrix.get("current_supported_slice") != EXPECTED_SUPPORTED_SLICE:
        raise ValueError("WP-9 supported slice changed after freeze")
    if matrix.get("policy_provenance_contract") != (EXPECTED_PROVENANCE_CONTRACT):
        raise ValueError("WP-9 provenance contract changed after freeze")
    if matrix.get("test_binary_by_suite") != EXPECTED_BINARY_BY_SUITE:
        raise ValueError("WP-9 test-binary mapping changed after freeze")
    if matrix.get("test_groups") != EXPECTED_TEST_GROUPS:
        raise ValueError("WP-9 test groups changed after freeze")
    if matrix.get("tests") != EXPECTED_TESTS:
        raise ValueError("WP-9 flat test list changed after freeze")
    if len(EXPECTED_TESTS) != len(set(EXPECTED_TESTS)):
        raise RuntimeError("internal WP-9 expected-test list has duplicates")
    if matrix.get("prospective_tests") != []:
        raise ValueError("WP-9 prerequisite matrix cannot claim prospective tests")

    exclusions = matrix.get("current_capability_exclusions")
    if (
        not isinstance(exclusions, list)
        or len(exclusions) != 7
        or len(set(exclusions)) != len(exclusions)
        or any(
            not isinstance(exclusion, str) or not exclusion.strip()
            for exclusion in exclusions
        )
    ):
        raise ValueError("WP-9 capability exclusions are incomplete")

    execution = matrix.get("execution")
    if execution != {
        "mpi_ranks": 1,
        "threads": 1,
        "wall_time_seconds": 900,
        "memory_mib": 6144,
        "output_mib": 128,
    }:
        raise ValueError("WP-9 execution envelope changed after freeze")
    gates = matrix.get("gates")
    if gates != {
        "expected_test_count": len(EXPECTED_TESTS),
        "expected_failures": 0,
        "expected_errors": 0,
        "expected_disabled": 0,
        "unexpected_tests_allowed": False,
        "skipped_tests_allowed": False,
    }:
        raise ValueError("WP-9 prerequisite gates changed after freeze")

    claims = matrix.get("prerequisite_claims")
    if not isinstance(claims, list):
        raise ValueError("WP-9 prerequisite claims are missing")
    claim_names: set[str] = set()
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or set(claim) != {"claim", "evidence"}
            or not isinstance(claim.get("claim"), str)
            or not isinstance(claim.get("evidence"), list)
            or not claim["evidence"]
            or any(test not in EXPECTED_TESTS for test in claim["evidence"])
        ):
            raise ValueError("WP-9 prerequisite claim is invalid")
        claim_names.add(claim["claim"])
    if claim_names != EXPECTED_PREREQUISITE_CLAIMS:
        raise ValueError("WP-9 prerequisite claim set changed after freeze")

    _validate_open_exits(
        matrix.get("unqualified_required_method_exits"),
        EXPECTED_METHOD_EXITS,
        "required method exits",
    )
    _validate_open_exits(
        matrix.get("unqualified_required_simulations"),
        EXPECTED_SIMULATION_EXITS,
        "required simulation exits",
    )
    if matrix.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("WP-9 qualification disposition must remain open")
    if matrix.get("closure_request_policy") != (EXPECTED_CLOSURE_REQUEST_POLICY):
        raise ValueError("WP-9 closure-request policy changed after freeze")
    _validate_source_definitions(matrix)
    return matrix


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_matrix(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("WP-9 frozen matrix is unavailable")
    resolved = path.resolve()
    if resolved != DEFAULT_MATRIX.resolve():
        raise ValueError("WP-9 requires the canonical frozen matrix")
    if not resolved.is_file():
        raise ValueError("WP-9 frozen matrix is unavailable")
    if base_runner.sha256_file(resolved) != EXPECTED_MATRIX_SHA256:
        raise ValueError("WP-9 frozen matrix bytes changed")
    matrix = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return validate_wp9_contract(matrix)


def write_json(path: Path, value: Any) -> None:
    if base_runner.sha256_file(SHARED_RUNNER_PATH) != SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification runner changed during execution")
    if isinstance(value, dict):
        value = copy.deepcopy(value)
        value["qualification_scope"] = EXPECTED_SCOPE
        value["closure_state"] = EXPECTED_CLOSURE_STATE
        value["qualification_disposition"] = EXPECTED_DISPOSITION
        value["wp9_closure_claimed"] = False
        value["q4_closure_claimed"] = False
    _base_write_json(path, value)


def _tests_for_binary(
    matrix: dict[str, Any],
    binary_key: str,
) -> list[str]:
    return [
        test
        for test in matrix["tests"]
        if matrix["test_binary_by_suite"][test.split(".", 1)[0]] == binary_key
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


def _requested_claim(
    arguments: list[str],
) -> tuple[str, bool, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-9 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only non-closing prerequisite claim.\n"
            "  --validate-only\n"
            "      Validate the exact frozen matrix and source definitions.\n"
            "  --list-only --physics-binary PATH "
            "--application-binary PATH\n"
            "      Check both binaries without executing tests or writing "
            "artifacts.\n"
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
            f"unsupported WP-9 requested claim {claim!r}; expected {allowed!r}"
        )
    return claim, parsed.validate_only, parsed.list_only, remaining


def _validation_summary(
    matrix: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    return {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "status": matrix["status"],
        "requested_claim": claim,
        "group_count": len(matrix["test_groups"]),
        "test_count": len(matrix["tests"]),
        "application_test_count": len(_tests_for_binary(matrix, "application")),
        "physics_test_count": len(_tests_for_binary(matrix, "physics")),
        "prospective_test_count": len(matrix["prospective_tests"]),
        "unqualified_method_exit_count": len(
            matrix["unqualified_required_method_exits"]
        ),
        "unqualified_simulation_exit_count": len(
            matrix["unqualified_required_simulations"]
        ),
        **matrix["qualification_disposition"],
        "outcome": "PASS_PREREQUISITE_NONCLOSURE",
    }


def _run_validate_only(claim: str, remaining: list[str]) -> int:
    if remaining:
        raise ValueError("--validate-only does not accept execution arguments")
    matrix = load_matrix(DEFAULT_MATRIX)
    print(json.dumps(_validation_summary(matrix, claim), sort_keys=True))
    return 0


def _parse_binary_arguments(
    program: str,
    arguments: list[str],
) -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    return parsed.physics_binary.resolve(), parsed.application_binary.resolve()


def _require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"test binary is not executable: {path}")


def _run_list_only(claim: str, remaining: list[str]) -> int:
    physics, application = _parse_binary_arguments(
        f"{SCRIPT_PATH.name} --list-only",
        remaining,
    )
    _require_executable(physics)
    _require_executable(application)
    matrix = load_matrix(DEFAULT_MATRIX)
    binaries = {"physics": physics, "application": application}
    missing_by_binary: dict[str, list[str]] = {}
    listed_counts: dict[str, int] = {}
    binary_hashes: dict[str, str] = {}
    for key, binary in binaries.items():
        expected = set(_tests_for_binary(matrix, key))
        listed = _listed_gtests(binary)
        missing_by_binary[key] = sorted(expected - listed)
        listed_counts[key] = len(expected & listed)
        binary_hashes[key] = base_runner.sha256_file(binary)
    missing = any(missing_by_binary.values())
    print(
        json.dumps(
            {
                **_validation_summary(matrix, claim),
                "binary_sha256": binary_hashes,
                "listed_expected_test_count": listed_counts,
                "missing_tests": missing_by_binary,
                "tests_executed": 0,
                "artifacts_written": 0,
                "outcome": ("FAIL" if missing else "PASS_PREREQUISITE_NONCLOSURE"),
            },
            sort_keys=True,
        )
    )
    return 2 if missing else 0


def _run_one_binary(
    key: str,
    binary: Path,
    tests: list[str],
    matrix: dict[str, Any],
    source_root: Path,
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_output = output / key
    group_output.mkdir(exist_ok=False)
    gtest_path = group_output / "gtest.json"
    stdout_path = group_output / "stdout.txt"
    stderr_path = group_output / "stderr.txt"
    command = [
        str(binary),
        f"--gtest_filter={':'.join(tests)}",
        f"--gtest_output=json:{gtest_path}",
    ]
    execution = matrix["execution"]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(execution["threads"])
    run = base_runner.run_monitored(
        command=command,
        environment=environment,
        working_directory=source_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_directory=group_output,
        wall_time_seconds=execution["wall_time_seconds"],
        memory_mib=execution["memory_mib"],
        output_mib=execution["output_mib"],
    )
    if gtest_path.is_file():
        document = json.loads(gtest_path.read_text(encoding="utf-8"))
        group_matrix = {
            "tests": tests,
            "gates": {
                **matrix["gates"],
                "expected_test_count": len(tests),
            },
        }
        checks = base_runner.evaluate_results(
            group_matrix,
            document,
            run["return_code"],
            run["termination_reason"],
        )
    else:
        checks = [
            {
                "metric": "gtest_result_present",
                "actual": False,
                "expected": True,
                "relation": "equal",
                "passed": False,
            }
        ]
    run["binary_key"] = key
    run["command"] = command
    run["outcome"] = (
        "PASS_PREREQUISITE_NONCLOSURE"
        if all(check["passed"] for check in checks)
        else "FAIL"
    )
    write_json(group_output / "run.json", run)
    write_json(group_output / "comparison.json", {"checks": checks})
    return run, checks


def _run_execution(claim: str, remaining: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=SCRIPT_PATH.name)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--supplemental-source",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(remaining)

    source_root = parsed.source_root.resolve()
    output = parsed.output.resolve()
    binaries = {
        "physics": parsed.physics_binary.resolve(),
        "application": parsed.application_binary.resolve(),
    }
    for binary in binaries.values():
        _require_executable(binary)
    if output.exists():
        raise ValueError(f"refusing to replace output directory: {output}")

    matrix = load_matrix(DEFAULT_MATRIX)
    source_state = base_runner.explicit_source_state(
        source_root,
        parsed.supplemental_source,
    )
    matrix_hash_before = base_runner.sha256_file(DEFAULT_MATRIX)
    script_hash_before = base_runner.sha256_file(SCRIPT_PATH)
    shared_hash_before = base_runner.sha256_file(SHARED_RUNNER_PATH)
    source_commit = (
        base_runner.git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    )
    source_tree = (
        base_runner.git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    )
    tracked_diff = base_runner.git_bytes(source_root, "diff", "--binary", "HEAD")

    output.mkdir(parents=True, exist_ok=False)
    tests_by_binary = {key: _tests_for_binary(matrix, key) for key in binaries}
    write_json(
        output / "manifest.json",
        {
            "artifact_schema_version": 1,
            "matrix_id": matrix["matrix_id"],
            "matrix_sha256": matrix_hash_before,
            "requested_claim": claim,
            "tests": matrix["tests"],
            "tests_by_binary": tests_by_binary,
            "exit_contract": matrix["exit_contract"],
        },
    )
    write_json(
        output / "build.json",
        {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "tracked_diff_sha256": base_runner.sha256_bytes(tracked_diff),
            **source_state,
            "binaries": {
                key: {
                    "path": str(binary),
                    "sha256": base_runner.sha256_file(binary),
                    "cmake_cache": base_runner.selected_cmake_cache(
                        base_runner.find_cmake_cache(binary)
                    ),
                }
                for key, binary in binaries.items()
            },
            "mpi_ranks": matrix["execution"]["mpi_ranks"],
            "threads": matrix["execution"]["threads"],
        },
    )
    write_json(
        output / "gates.json",
        {
            "matrix_status_at_execution": matrix["status"],
            "gates": matrix["gates"],
            "resource_envelope_per_binary": matrix["execution"],
        },
    )

    runs: dict[str, dict[str, Any]] = {}
    checks: dict[str, list[dict[str, Any]]] = {}
    for key, binary in binaries.items():
        runs[key], checks[key] = _run_one_binary(
            key,
            binary,
            tests_by_binary[key],
            matrix,
            source_root,
            output,
        )

    if (
        base_runner.sha256_file(DEFAULT_MATRIX) != matrix_hash_before
        or base_runner.sha256_file(SCRIPT_PATH) != script_hash_before
        or base_runner.sha256_file(SHARED_RUNNER_PATH) != shared_hash_before
    ):
        raise RuntimeError("matrix or runner changed during execution")
    passed = all(
        check["passed"] for group_checks in checks.values() for check in group_checks
    )
    write_json(
        output / "comparison.json",
        {
            "matrix_id": matrix["matrix_id"],
            "runs": runs,
            "checks": checks,
            "disposition": (
                "PASS_PREREQUISITE_NONCLOSURE" if passed else "FAIL_METHOD"
            ),
            "reason": (
                "all frozen prerequisite tests passed; every WP-9, Q4, "
                "FSR, and physical-campaign closure remains open"
                if passed
                else "one or more frozen prerequisite gates failed"
            ),
        },
    )
    base_runner.write_checksums(output)
    print(output)
    print("PASS_PREREQUISITE_NONCLOSURE" if passed else "FAIL")
    return 0 if passed else 2


def main(arguments: list[str] | None = None) -> int:
    provided = sys.argv[1:] if arguments is None else arguments
    claim, validate_only, list_only, remaining = _requested_claim(provided)
    if validate_only:
        return _run_validate_only(claim, remaining)
    if list_only:
        return _run_list_only(claim, remaining)
    return _run_execution(claim, remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
