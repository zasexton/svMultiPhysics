#!/usr/bin/env python3
"""Run the frozen WP-3 sharp exterior-boundary qualification matrix."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import run_free_surface_wp2_geometry_qualification as strict_runner  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_sharp_boundary_qualification_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "333ddc3ef8817ecce10eba4cb2cc7b25d7038f2310c061e94750ea9a326f58d5"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)
EXPECTED_SCOPE = (
    "Low-level WP-3 prerequisite evidence only; this matrix does not close "
    "FSR-16, WP-3, WP-7, or Q1."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr16_closure",
        "wp3_closure",
        "wp7_closure",
        "q1_closure",
    ],
    "diagnostic": (
        "The frozen sharp-boundary slice cannot establish the joint WP-7 "
        "coercivity threshold, the complete cut-conditioning matrix, or Q1 "
        "closure."
    ),
}
EXPECTED_WET_FRACTIONS = [
    1.0e-8,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    0.1,
    0.25,
    0.49,
    1.0,
]
EXPECTED_MODEL_ENVELOPE = (
    "one_phase_unfitted_active_liquid_with_c0_p1_velocity_pressure_"
    "and_linearcorner_implicit_geometry_for_sharp_exterior_boundary_forms"
)
EXPECTED_OPERATORS = {
    "traction",
    "robin",
    "outflow",
    "pressure_flux",
    "symmetric_nitsche",
    "unsymmetric_nitsche",
    "wall_slip",
}
EXPECTED_OPERATOR_KEYS = {
    "operator",
    "cut_active_disposition",
    "full_domain_disposition",
    "dry_face_disposition",
    "missing_sharp_domain_disposition",
    "active_side_reversal",
}
EXPECTED_JOINT_THRESHOLD_KEYS = {
    "metric",
    "owner",
    "status",
    "reason",
    "closure_effect",
}
EXPECTED_UNSUPPORTED_OPERATORS = {
    "coupled_resistance_outflow",
    "coupled_multielement_outflow",
}
EXPECTED_THRESHOLD_BASIS_KEYS = {
    "geometry_moment_and_partition_error",
    "operator_scaled_work_error",
    "channel_scaled_work_error",
    "vertex_crossing_jump",
    "nitsche_sampled_margin",
    "p1_linearcorner_envelope",
    "structured_repartition",
}
EXPECTED_REQUIRED_TESTS = {
    "ApplicationDriverLevelSetWorkflows.RefreshesMultipleGeneratedCutDomainsIntoOneContext",
    "FreeSurfaceSharpBoundaryOperators.PspgBoundaryPressureFluxUsesGeneratedWetWallMeasure",
    "FreeSurfaceSharpBoundaryOperators.MissingGeneratedActiveDomainFailsClosed",
    "FreeSurfaceSharpBoundaryOperators.HigherOrderSpacesAndImplicitGeometryFailClosed",
    "FreeSurfaceSharpBoundaryOperators.StructuredManufacturedChannelTracksDistinctExteriorRoles",
    "FreeSurfaceSharpBoundaryOperators.StructuredVertexCrossingGlobalWorkJumpConvergesUnderRefinement",
    "FreeSurfaceSharpBoundaryOperatorsMPI.StructuredChannelWorkIsInvariantUnderActualRepartition",
}
EXPECTED_MPI_RECORDED_PROPERTIES = {
    "sharp_boundary_assembly_mpi": {
        "sharp_boundary_mpi_rank_count": ("integer", "equal", 2),
        "sharp_boundary_mpi_fraction_case_count": ("integer", "equal", 8),
        "sharp_boundary_mpi_active_side_case_count": ("integer", "equal", 2),
        "sharp_boundary_mpi_partition_case_count": ("integer", "equal", 16),
        "sharp_boundary_mpi_minimum_owner_multiplicity": (
            "integer",
            "equal",
            1,
        ),
        "sharp_boundary_mpi_maximum_owner_multiplicity": (
            "integer",
            "equal",
            1,
        ),
        "sharp_boundary_mpi_maximum_measure_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_boundary_mpi_maximum_residual_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_boundary_mpi_maximum_work_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_boundary_mpi_maximum_fraction_scaling_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
    },
    "sharp_boundary_operators_mpi": {
        "sharp_operator_mpi_rank_count": ("integer", "equal", 2),
        "sharp_operator_mpi_fraction_case_count": ("integer", "equal", 8),
        "sharp_operator_mpi_active_side_case_count": ("integer", "equal", 2),
        "sharp_operator_mpi_family_count": ("integer", "equal", 7),
        "sharp_operator_mpi_partition_case_count": ("integer", "equal", 112),
        "sharp_operator_mpi_minimum_owner_multiplicity": (
            "integer",
            "equal",
            1,
        ),
        "sharp_operator_mpi_maximum_owner_multiplicity": (
            "integer",
            "equal",
            1,
        ),
        "sharp_operator_mpi_minimum_full_residual_norm": (
            "real",
            "greater_than",
            1.0e-12,
        ),
        "sharp_operator_mpi_maximum_residual_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_operator_mpi_maximum_jacobian_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_operator_mpi_maximum_work_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_operator_mpi_maximum_residual_scaling_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_operator_mpi_maximum_jacobian_scaling_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
    },
    "sharp_boundary_structured_mpi": {
        "sharp_structured_mpi_rank_count": ("integer", "equal", 2),
        "sharp_structured_mpi_cell_count": ("integer", "equal", 96),
        "sharp_structured_mpi_partition_count": ("integer", "equal", 2),
        "sharp_structured_mpi_boundary_role_count": (
            "integer",
            "equal",
            3,
        ),
        "sharp_structured_mpi_rule_count_mismatch": (
            "integer",
            "equal",
            0,
        ),
        "sharp_structured_mpi_maximum_measure_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
        "sharp_structured_mpi_maximum_work_error": (
            "real",
            "less_than_or_equal",
            1.0e-11,
        ),
    },
}


def validate_wp3_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-3 qualification scope changed after freeze")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("WP-3 closure-request policy changed after freeze")
    if registry.get("model_envelope") != EXPECTED_MODEL_ENVELOPE:
        raise ValueError(
            "WP-3 model envelope must remain explicit C0-P1/LinearCorner"
        )

    fractions = registry.get("wet_fraction_sweep")
    if fractions != EXPECTED_WET_FRACTIONS:
        raise ValueError("WP-3 wet-fraction sweep does not match the frozen list")

    contracts = registry.get("operator_disposition_contract")
    if not isinstance(contracts, list) or len(contracts) != len(EXPECTED_OPERATORS):
        raise ValueError("WP-3 operator disposition contract is incomplete")
    operators: set[str] = set()
    for contract in contracts:
        if not isinstance(contract, dict) or set(contract) != EXPECTED_OPERATOR_KEYS:
            raise ValueError("WP-3 operator disposition entry has unexpected keys")
        operator = contract.get("operator")
        if operator in operators:
            raise ValueError(f"duplicate WP-3 operator disposition: {operator}")
        operators.add(operator)
        if contract.get("cut_active_disposition") != "generated_active_boundary":
            raise ValueError(f"WP-3 cut operator is not sharply routed: {operator}")
        if contract.get("full_domain_disposition") != "physical_boundary":
            raise ValueError(f"WP-3 full-domain disposition is invalid: {operator}")
        if contract.get("dry_face_disposition") != "exact_zero":
            raise ValueError(f"WP-3 dry-face disposition is invalid: {operator}")
        if contract.get("missing_sharp_domain_disposition") != "hard_error":
            raise ValueError(f"WP-3 missing-domain disposition is invalid: {operator}")
        if contract.get("active_side_reversal") != "complementary_sharp_subset":
            raise ValueError(f"WP-3 active-side disposition is invalid: {operator}")
    if operators != EXPECTED_OPERATORS:
        raise ValueError("WP-3 operator disposition set is incomplete")

    unsupported = registry.get("unsupported_operator_contract")
    if not isinstance(unsupported, list):
        raise ValueError("WP-3 unsupported-operator contract is missing")
    unsupported_families: set[str] = set()
    for contract in unsupported:
        if not isinstance(contract, dict) or set(contract) != {
            "family",
            "cut_active_disposition",
        }:
            raise ValueError("WP-3 unsupported-operator entry has unexpected keys")
        family = contract.get("family")
        if family in unsupported_families:
            raise ValueError(f"duplicate WP-3 unsupported operator: {family}")
        unsupported_families.add(family)
        if (
            contract.get("cut_active_disposition")
            != "hard_error_until_sharp_reduction_is_implemented"
        ):
            raise ValueError(f"WP-3 unsupported operator does not fail closed: {family}")
    if unsupported_families != EXPECTED_UNSUPPORTED_OPERATORS:
        raise ValueError("WP-3 unsupported-operator set is incomplete")

    regularized = registry.get("regularized_experimental_model_contract")
    if regularized != {
        "name": "SmoothedIndicator",
        "scope": "diagnostic_bulk_active_domain_only",
        "available_for_exterior_boundary_forms": False,
        "production_sharp_operator_substitute": False,
        "width_unit": "physical_length",
        "independent_width_refinement_required": True,
    }:
        raise ValueError("WP-3 regularized experimental model contract is invalid")
    threshold_basis = registry.get("numeric_threshold_basis")
    if (
        not isinstance(threshold_basis, dict)
        or set(threshold_basis) != EXPECTED_THRESHOLD_BASIS_KEYS
        or any(
            not isinstance(value, str) or not value.strip()
            for value in threshold_basis.values()
        )
    ):
        raise ValueError("WP-3 numeric threshold basis is incomplete")

    unresolved = registry.get("unfrozen_joint_thresholds")
    if not isinstance(unresolved, list) or len(unresolved) != 1:
        raise ValueError("WP-3 must declare its joint cut-stability threshold blocker")
    threshold = unresolved[0]
    if not isinstance(threshold, dict) or set(threshold) != EXPECTED_JOINT_THRESHOLD_KEYS:
        raise ValueError("WP-3 joint threshold declaration has unexpected keys")
    if threshold.get("metric") != "uniform_symmetric_nitsche_coercivity_constant":
        raise ValueError("WP-3 joint threshold metric is unexpected")
    if threshold.get("owner") != "WP-7_joint_cut_stability":
        raise ValueError("WP-3 joint threshold owner is unexpected")
    if threshold.get("status") != "UNFROZEN_NO_BOUND_INVENTED":
        raise ValueError("WP-3 joint threshold status is unexpected")
    for key in ("reason", "closure_effect"):
        if not isinstance(threshold.get(key), str) or not threshold[key].strip():
            raise ValueError(f"WP-3 joint threshold {key} must be nonempty")

    groups_by_id = {group["id"]: group for group in registry["groups"]}
    for group_id, expected_properties in EXPECTED_MPI_RECORDED_PROPERTIES.items():
        group = groups_by_id.get(group_id)
        if group is None:
            raise ValueError(f"WP-3 MPI group is missing: {group_id}")
        if (
            group.get("binary") != "physics"
            or group.get("mpi_ranks") != 2
            or group.get("gtest_output_copies") != 1
        ):
            raise ValueError(f"WP-3 MPI group execution contract is invalid: {group_id}")
        properties = group.get("recorded_properties")
        if not isinstance(properties, list):
            raise ValueError(f"WP-3 MPI group properties are missing: {group_id}")
        actual_properties: dict[str, tuple[str, str, int | float]] = {}
        for property_contract in properties:
            if not isinstance(property_contract, dict) or set(property_contract) != {
                "property",
                "type",
                "relation",
                "threshold",
            }:
                raise ValueError(
                    f"WP-3 MPI property contract is invalid: {group_id}"
                )
            property_name = property_contract.get("property")
            if not isinstance(property_name, str) or not property_name:
                raise ValueError(f"WP-3 MPI property name is invalid: {group_id}")
            if property_name in actual_properties:
                raise ValueError(
                    f"duplicate WP-3 MPI property: {group_id}.{property_name}"
                )
            actual_properties[property_name] = (
                property_contract.get("type"),
                property_contract.get("relation"),
                property_contract.get("threshold"),
            )
        if actual_properties != expected_properties:
            raise ValueError(f"WP-3 MPI properties changed after freeze: {group_id}")

    prospective = registry.get("prospective_tests")
    if not isinstance(prospective, list):
        raise ValueError("WP-3 prospective test list must be explicit")
    frozen_tests = {
        test
        for group in registry["groups"]
        for test in group["tests"]
    }
    if not EXPECTED_REQUIRED_TESTS.issubset(frozen_tests):
        missing = sorted(EXPECTED_REQUIRED_TESTS - frozen_tests)
        raise ValueError(
            f"WP-3 required production/structured regressions are missing: {missing}"
        )
    application_group = groups_by_id.get("sharp_boundary_application_serial")
    if (
        application_group is None
        or application_group.get("binary") != "application"
        or application_group.get("mpi_ranks") != 1
        or application_group.get("tests")
        != [
            "ApplicationDriverLevelSetWorkflows.RefreshesMultipleGeneratedCutDomainsIntoOneContext"
        ]
    ):
        raise ValueError("WP-3 ApplicationDriver pipeline group is invalid")
    if any(
        not isinstance(test, str)
        or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", test)
        for test in prospective
    ):
        raise ValueError("WP-3 prospective test names must use suite.name form")
    if len(prospective) != len(set(prospective)):
        raise ValueError("WP-3 prospective test list contains duplicates")
    if not set(prospective).issubset(frozen_tests):
        raise ValueError("WP-3 prospective tests must be frozen in an execution group")
    if prospective:
        raise ValueError("WP-3 cannot close with prospective tests")
    registry["prospective_tests"] = prospective
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = "free_surface_wp3_sharp_boundary_v1"
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-3"
strict_runner.__doc__ = __doc__

_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text


def load_registry(path: Path) -> dict[str, Any]:
    if strict_runner.sha256_file(path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-3 frozen registry bytes changed")
    return validate_wp3_contract(_shared_load_registry(path))


def write_json(path: Path, value: Any) -> None:
    if strict_runner.sha256_file(SHARED_RUNNER_PATH) != SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification runner changed during execution")
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "final_provenance.json",
        "summary.json",
    }:
        value = copy.deepcopy(value)
        value["shared_runner_dependency"] = {
            "path": str(SHARED_RUNNER_PATH),
            "sha256": SHARED_RUNNER_SHA256,
        }
        value["qualification_scope"] = EXPECTED_SCOPE
        value["requested_claim"] = "low_level_prerequisite"
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-3 sharp exterior-boundary qualification record",
            1,
        )
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "The joint WP-7 threshold and complete cut-conditioning matrix "
            "remain separate required artifacts.\n"
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-3 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this low-level matrix may establish.\n"
            "      FSR-16, WP-3, WP-7, and Q1 closure are rejected.\n"
            "  --validate-only\n"
            "      Validate the frozen schema and claim boundary without "
            "builds.\n"
        )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--requested-claim",
        default=EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"],
    )
    parser.add_argument("--validate-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
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
            f"unsupported WP-3 requested claim {claim!r}; expected {allowed!r}"
        )
    return claim, parsed.validate_only, remaining


if __name__ == "__main__":
    try:
        _claim, _validate_only, _remaining_arguments = requested_claim(
            sys.argv[1:]
        )
        if _validate_only:
            if _remaining_arguments:
                raise ValueError(
                    "--validate-only does not accept execution arguments"
                )
            _registry = load_registry(DEFAULT_REGISTRY)
            print(
                json.dumps(
                    {
                        "matrix_id": _registry["matrix_id"],
                        "status": _registry["status"],
                        "requested_claim": _claim,
                        "prospective_test_count": len(
                            _registry["prospective_tests"]
                        ),
                        "group_count": len(_registry["groups"]),
                        "test_count": sum(
                            len(group["tests"])
                            for group in _registry["groups"]
                        ),
                        "recorded_property_gate_count": sum(
                            len(group.get("recorded_properties", []))
                            for group in _registry["groups"]
                        ),
                        "joint_wp7_threshold_frozen": False,
                        "wp3_closed": False,
                        "outcome": "PASS",
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(0)
        sys.argv = [sys.argv[0], *_remaining_arguments]
        raise SystemExit(strict_runner.main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=strict_runner.sys.stderr)
        raise SystemExit(2)
