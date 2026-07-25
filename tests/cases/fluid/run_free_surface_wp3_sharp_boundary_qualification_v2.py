#!/usr/bin/env python3
"""Run the frozen additive WP-3 v2 sharp-boundary qualification matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_PATH.parent
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_sharp_boundary_qualification_matrix_v2.json"
)
WP2_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp2_geometry_qualification.py"
)
EXPECTED_REGISTRY_SHA256 = (
    "72cffdc330f07b386fdb89681bcd3da83b7f884c5bd1d09f49eef3f6ae79d883"
)
EXPECTED_MATRIX_ID = "free_surface_wp3_sharp_boundary_v2"
EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_WORK_PACKAGE = "WP-3"
EXPECTED_SCOPE = (
    "Low-level WP-3 prerequisite evidence only; coupled RCR/RCRCR sharp "
    "routing is included, but this matrix does not close FSR-16, WP-3, "
    "joint WP-7, or Q1."
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
        "The frozen sharp-boundary slice, including coupled RCR/RCRCR "
        "routing, cannot establish the joint WP-7 coercivity threshold, "
        "the complete cut-conditioning matrix, or Q1 closure."
    ),
}
EXPECTED_OPEN_OUTCOMES = {
    "fsr16": "OPEN",
    "wp3": "OPEN",
    "joint_wp7": "OPEN",
    "q1": "OPEN",
}
EXPECTED_MODEL_ENVELOPE = (
    "one_phase_unfitted_active_liquid_with_c0_p1_velocity_pressure_"
    "and_linearcorner_implicit_geometry_for_sharp_exterior_boundary_forms"
)
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
EXPECTED_SUPPORTED_OPERATORS = {
    "traction",
    "robin",
    "outflow",
    "pressure_flux",
    "symmetric_nitsche",
    "unsymmetric_nitsche",
    "wall_slip",
    "coupled_rcr_outflow",
    "coupled_rcrcr_outflow",
}
EXPECTED_OPERATOR_KEYS = {
    "operator",
    "cut_active_disposition",
    "full_domain_disposition",
    "dry_face_disposition",
    "missing_sharp_domain_disposition",
    "active_side_reversal",
}
EXPECTED_THRESHOLD_BASIS_KEYS = {
    "geometry_moment_and_partition_error",
    "operator_scaled_work_error",
    "channel_scaled_work_error",
    "vertex_crossing_jump",
    "nitsche_sampled_margin",
    "p1_linearcorner_envelope",
    "structured_repartition",
    "coupled_outflow_reduction_and_repartition",
}
EXPECTED_GROUP_TESTS = {
    "sharp_boundary_geometry_serial": (
        "geometry",
        1,
        (
            "GeneratedActiveBoundaryDomain."
            "HalfWetEdgeUsesAuthoritativeContactAndPartitionsExactly",
            "GeneratedActiveBoundaryDomain."
            "PlanarHalfAndQuarterFacesIntegratePolynomialsForBothSides",
            "GeneratedActiveBoundaryDomain."
            "ObliqueHexFaceIntegratesQuadraticMomentsForBothSides",
            "GeneratedActiveBoundaryDomain."
            "CompletelyDrySideHasExactlyZeroRules",
            "GeneratedActiveBoundaryDomain."
            "WetFractionSweepIntegratesBoundaryPolynomialsForBothPhases",
            "GeneratedActiveBoundaryDomain."
            "CurvedQuadraticParentUsesPointwisePhysicalBoundaryMapping",
            "GeneratedActiveBoundaryDomain."
            "RejectsScalarRootThatDoesNotMatchAuthoritativeContactTrace",
        ),
    ),
    "sharp_boundary_assembly_serial": (
        "assembly",
        1,
        (
            "StandardAssemblerCutInterfaces."
            "SharpActiveBoundaryHalfWetMeasureAndFullyDryResidualAreExact",
        ),
    ),
    "sharp_boundary_systems_serial": (
        "systems",
        1,
        (
            "BoundaryIntegralInput."
            "GeneratedActiveBoundaryValueGradientDryValidationAndHandleIdentity",
        ),
    ),
    "sharp_boundary_operators_serial": (
        "physics",
        1,
        (
            "MovingDomainPhysics."
            "NavierStokesUnfittedNaturalAndWeakBoundaryOperatorsUseSharpActiveTrace",
            "MovingDomainPhysics."
            "NavierStokesUnfittedBoundaryOperatorsRejectMultipleActiveOwners",
            "NavierStokesLegacyBCs."
            "UnfittedDynamicContactAngleTranslationRoutesLineAndSharpWallGeometry",
            "FreeSurfaceSharpBoundaryOperators."
            "WetFractionSweepMatchesAnalyticOperatorWork",
            "FreeSurfaceSharpBoundaryOperators."
            "PspgBoundaryPressureFluxUsesGeneratedWetWallMeasure",
            "FreeSurfaceSharpBoundaryOperators."
            "ActiveSideReversalUsesComplementarySharpSubset",
            "FreeSurfaceSharpBoundaryOperators."
            "CompletelyDryBoundaryProducesExactlyZeroWetRows",
            "FreeSurfaceSharpBoundaryOperators."
            "MissingGeneratedActiveDomainFailsClosed",
            "FreeSurfaceSharpBoundaryOperators."
            "HigherOrderSpacesAndImplicitGeometryFailClosed",
            "FreeSurfaceSharpBoundaryOperators."
            "StructuredManufacturedChannelTracksDistinctExteriorRoles",
            "FreeSurfaceSharpBoundaryOperators."
            "StructuredVertexCrossingGlobalWorkJumpConvergesUnderRefinement",
            "FreeSurfaceSharpBoundaryOperators."
            "NitscheTraceScalingProducesFiniteSampledMargins",
            "NavierStokesOutletFactory."
            "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
            "MovingDomainPhysics."
            "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
            "MovingDomainPhysics."
            "NavierStokesUnfittedCoupledOutflowFamiliesRejectUnsupportedSharpEnvelope",
        ),
    ),
    "sharp_boundary_application_serial": (
        "application",
        1,
        (
            "ApplicationDriverLevelSetWorkflows."
            "RefreshesMultipleGeneratedCutDomainsIntoOneContext",
        ),
    ),
    "sharp_boundary_assembly_mpi": (
        "physics",
        2,
        (
            "GeneratedActiveBoundaryDomainMPI."
            "WetFractionSweepIsOwnershipUniqueAndPartitionIndependent",
        ),
    ),
    "sharp_boundary_operators_mpi": (
        "physics",
        2,
        (
            "FreeSurfaceSharpBoundaryOperatorsMPI."
            "OperatorWorkIsPartitionIndependent",
        ),
    ),
    "sharp_boundary_structured_mpi": (
        "physics",
        2,
        (
            "FreeSurfaceSharpBoundaryOperatorsMPI."
            "StructuredChannelWorkIsInvariantUnderActualRepartition",
        ),
    ),
    "sharp_boundary_coupled_outflow_mpi": (
        "physics",
        2,
        (
            "MovingDomainPhysicsMPI."
            "GeneratedActiveCoupledOutflowReductionGradientAndTractionArePartitionIndependent",
        ),
    ),
}
EXPECTED_NEW_SERIAL_EVIDENCE = {
    (
        "NavierStokesOutletFactory."
        "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
        "coupled_outflow_generated_trace_variant_count",
    ): ("integer", "equal", 3),
    (
        "NavierStokesOutletFactory."
        "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
        "coupled_outflow_physical_trace_variant_count",
    ): ("integer", "equal", 0),
    (
        "NavierStokesOutletFactory."
        "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
        "coupled_outflow_physical_deployment_variant_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_variant_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_generated_trace_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_whole_face_fallback_count",
    ): ("integer", "equal", 0),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_generated_flow_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowFamiliesRejectUnsupportedSharpEnvelope",
        "sharp_coupled_outflow_envelope_rejection_count",
    ): ("integer", "equal", 2),
}
EXPECTED_MPI_RECORDED_PROPERTIES = {
    "sharp_coupled_outflow_mpi_rank_count": ("integer", "equal", 2),
    "sharp_coupled_outflow_mpi_cell_count": ("integer", "equal", 12),
    "sharp_coupled_outflow_mpi_partition_count": ("integer", "equal", 2),
    "sharp_coupled_outflow_mpi_gradient_probe_count": (
        "integer",
        "equal",
        4,
    ),
    "sharp_coupled_outflow_mpi_dual_marker_contract_count": (
        "integer",
        "equal",
        3,
    ),
    "sharp_coupled_outflow_mpi_rule_count_mismatch": (
        "integer",
        "equal",
        0,
    ),
    "sharp_coupled_outflow_mpi_owner_mismatch_count": (
        "integer",
        "equal",
        0,
    ),
    "sharp_coupled_outflow_mpi_whole_face_fallback_count": (
        "integer",
        "equal",
        0,
    ),
    "sharp_coupled_outflow_mpi_slab_outlet_contributor_count": (
        "integer",
        "equal",
        1,
    ),
    "sharp_coupled_outflow_mpi_round_robin_outlet_contributor_count": (
        "integer",
        "equal",
        2,
    ),
    "sharp_coupled_outflow_mpi_maximum_measure_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
    "sharp_coupled_outflow_mpi_maximum_flow_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
    "sharp_coupled_outflow_mpi_maximum_gradient_action_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
    "sharp_coupled_outflow_mpi_maximum_traction_work_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
}
EXPECTED_GATES = {
    "expected_group_count": 9,
    "expected_distinct_test_count": 29,
    "expected_quantitative_evidence_count": 68,
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}
EXPECTED_GROUPS_SHA256 = (
    "39d06847eb7dd1e0113597aebc6b9305373a95219cd4559e434bf6312520f69e"
)
EXPECTED_QUANTITATIVE_EVIDENCE_SHA256 = (
    "8e9672912d3e006b2a9aee992d5f6c674b21d5b08e93e380ce42daf63a12353a"
)
EXPECTED_CLOSURE_CONTRACT_SHA256 = (
    "58d6733a580b2ff3600b3deeca88a0db27729ca2f4c770f8e80f0fb551a4cd29"
)
REMOVED_REJECTION_TEST = (
    "MovingDomainPhysics."
    "NavierStokesUnfittedBoundaryOperatorsRejectCoupledOutflowFamilies"
)


def _load_wp2_runner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp3_v2_wp2_base",
        WP2_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the WP2 qualification base")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


strict_runner = _load_wp2_runner()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(WP2_RUNNER_PATH)
_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text

strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
strict_runner.EXPECTED_MATRIX_STATUS = EXPECTED_MATRIX_STATUS
strict_runner.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
strict_runner.__doc__ = __doc__


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _property_contracts(
    contracts: Any,
) -> dict[str, tuple[str, str, int | float]]:
    if not isinstance(contracts, list):
        raise ValueError("WP-3 v2 property contracts must be a list")
    result: dict[str, tuple[str, str, int | float]] = {}
    for contract in contracts:
        if not isinstance(contract, dict) or set(contract) != {
            "property",
            "type",
            "relation",
            "threshold",
        }:
            raise ValueError("WP-3 v2 property contract has unexpected keys")
        name = contract["property"]
        if name in result:
            raise ValueError(f"duplicate WP-3 v2 property: {name}")
        result[name] = (
            contract["type"],
            contract["relation"],
            contract["threshold"],
        )
    return result


def validate_wp3_v2_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-3 v2 qualification scope changed after freeze")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("WP-3 v2 closure-request policy changed after freeze")
    if registry.get("open_outcomes") != EXPECTED_OPEN_OUTCOMES:
        raise ValueError("WP-3 v2 open outcome changed after freeze")
    if registry.get("model_envelope") != EXPECTED_MODEL_ENVELOPE:
        raise ValueError("WP-3 v2 model envelope changed after freeze")
    if registry.get("wet_fraction_sweep") != EXPECTED_WET_FRACTIONS:
        raise ValueError("WP-3 v2 wet-fraction sweep changed after freeze")

    contracts = registry.get("operator_disposition_contract")
    if not isinstance(contracts, list):
        raise ValueError("WP-3 v2 supported-operator contract is missing")
    operators: set[str] = set()
    for contract in contracts:
        if not isinstance(contract, dict) or set(contract) != (
            EXPECTED_OPERATOR_KEYS
        ):
            raise ValueError(
                "WP-3 v2 supported-operator entry has unexpected keys"
            )
        operator = contract["operator"]
        if operator in operators:
            raise ValueError(f"duplicate WP-3 v2 supported operator: {operator}")
        operators.add(operator)
        expected_dispositions = {
            "cut_active_disposition": "generated_active_boundary",
            "full_domain_disposition": "physical_boundary",
            "dry_face_disposition": "exact_zero",
            "missing_sharp_domain_disposition": "hard_error",
            "active_side_reversal": "complementary_sharp_subset",
        }
        for key, expected in expected_dispositions.items():
            if contract.get(key) != expected:
                raise ValueError(
                    f"WP-3 v2 supported disposition changed: {operator}.{key}"
                )
    if operators != EXPECTED_SUPPORTED_OPERATORS:
        raise ValueError("WP-3 v2 supported-operator set is incomplete")
    if registry.get("unsupported_operator_contract") != []:
        raise ValueError("WP-3 v2 unsupported-operator set must be empty")

    threshold_basis = registry.get("numeric_threshold_basis")
    if (
        not isinstance(threshold_basis, dict)
        or set(threshold_basis) != EXPECTED_THRESHOLD_BASIS_KEYS
        or any(
            not isinstance(value, str) or not value.strip()
            for value in threshold_basis.values()
        )
    ):
        raise ValueError("WP-3 v2 threshold basis is incomplete")
    unresolved = registry.get("unfrozen_joint_thresholds")
    if (
        not isinstance(unresolved, list)
        or len(unresolved) != 1
        or unresolved[0].get("owner") != "WP-7_joint_cut_stability"
        or unresolved[0].get("status") != "UNFROZEN_NO_BOUND_INVENTED"
    ):
        raise ValueError("WP-3 v2 joint WP-7 threshold must remain open")

    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("WP-3 v2 groups are missing")
    if [group.get("id") for group in groups] != list(EXPECTED_GROUP_TESTS):
        raise ValueError("WP-3 v2 group inventory changed after freeze")
    for group in groups:
        group_id = group["id"]
        expected_binary, expected_ranks, expected_tests = (
            EXPECTED_GROUP_TESTS[group_id]
        )
        if (
            group.get("binary") != expected_binary
            or group.get("mpi_ranks") != expected_ranks
            or group.get("gtest_output_copies") != 1
            or tuple(group.get("tests", [])) != expected_tests
        ):
            raise ValueError(
                f"WP-3 v2 group contract changed after freeze: {group_id}"
            )
    if _canonical_sha256(groups) != EXPECTED_GROUPS_SHA256:
        raise ValueError("WP-3 v2 group execution/property contract changed")

    all_tests = [
        test
        for group in groups
        for test in group["tests"]
    ]
    if len(all_tests) != 29 or len(set(all_tests)) != 29:
        raise ValueError("WP-3 v2 test inventory must contain 29 unique tests")
    if REMOVED_REJECTION_TEST in all_tests:
        raise ValueError("WP-3 v2 retained the obsolete coupled rejection")

    coupled_group = next(
        group
        for group in groups
        if group["id"] == "sharp_boundary_coupled_outflow_mpi"
    )
    if _property_contracts(coupled_group.get("recorded_properties")) != (
        EXPECTED_MPI_RECORDED_PROPERTIES
    ):
        raise ValueError("WP-3 v2 coupled MPI properties changed after freeze")

    evidence = registry.get("quantitative_evidence")
    if _canonical_sha256(evidence) != (
        EXPECTED_QUANTITATIVE_EVIDENCE_SHA256
    ):
        raise ValueError("WP-3 v2 quantitative evidence changed after freeze")
    actual_evidence = {
        (entry["test"], entry["property"]): (
            entry["type"],
            entry["relation"],
            entry["threshold"],
        )
        for entry in evidence
    }
    for key, expected in EXPECTED_NEW_SERIAL_EVIDENCE.items():
        if actual_evidence.get(key) != expected:
            raise ValueError(
                f"WP-3 v2 serial coupled evidence changed: {key}"
            )
    if any(test == REMOVED_REJECTION_TEST for test, _ in actual_evidence):
        raise ValueError("WP-3 v2 evidence retained the obsolete rejection")

    if registry.get("gates") != EXPECTED_GATES:
        raise ValueError("WP-3 v2 count/result gates changed after freeze")
    if registry.get("prospective_tests") != []:
        raise ValueError("WP-3 v2 cannot execute prospective tests")
    if _canonical_sha256(registry.get("closure_contract")) != (
        EXPECTED_CLOSURE_CONTRACT_SHA256
    ):
        raise ValueError("WP-3 v2 closure contract changed after freeze")
    return registry


def load_registry(path: Path) -> dict[str, Any]:
    if strict_runner.sha256_file(path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-3 v2 frozen matrix bytes changed")
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("WP-3 v2 requires the canonical frozen matrix")
    if strict_runner.sha256_file(WP2_RUNNER_PATH) != SHARED_RUNNER_SHA256:
        raise RuntimeError("WP2 qualification base changed during execution")
    json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    return validate_wp3_v2_contract(_shared_load_registry(path))


def write_json(path: Path, value: Any) -> None:
    if strict_runner.sha256_file(WP2_RUNNER_PATH) != SHARED_RUNNER_SHA256:
        raise RuntimeError("WP2 qualification base changed during execution")
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "final_provenance.json",
        "summary.json",
    }:
        value = copy.deepcopy(value)
        value["shared_runner_dependency"] = {
            "path": str(WP2_RUNNER_PATH),
            "sha256": SHARED_RUNNER_SHA256,
        }
        value["qualification_scope"] = EXPECTED_SCOPE
        value["requested_claim"] = "low_level_prerequisite"
        value["open_outcomes"] = copy.deepcopy(EXPECTED_OPEN_OUTCOMES)
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-3 v2 sharp exterior-boundary qualification record",
            1,
        )
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "WP-3, the joint WP-7 threshold, and Q1 remain open.\n"
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-3 v2 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this low-level matrix may establish.\n"
            "      FSR-16, WP-3, joint WP-7, and Q1 closure are rejected.\n"
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
    accepted = EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]
    rejected = set(EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"])
    if claim in rejected:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != accepted:
        raise ValueError(
            f"unsupported WP-3 v2 requested claim {claim!r}; "
            f"expected {accepted!r}"
        )
    return claim, parsed.validate_only, remaining


def validate_only_summary(
    registry: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    return {
        "matrix_id": registry["matrix_id"],
        "status": registry["status"],
        "requested_claim": claim,
        "prospective_test_count": len(registry["prospective_tests"]),
        "group_count": len(registry["groups"]),
        "test_count": sum(
            len(group["tests"])
            for group in registry["groups"]
        ),
        "quantitative_evidence_gate_count": len(
            registry["quantitative_evidence"]
        ),
        "recorded_property_gate_count": sum(
            len(group.get("recorded_properties", []))
            for group in registry["groups"]
        ),
        "closure_outcome": "OPEN_LOW_LEVEL_PREREQUISITE",
        "fsr16_closed": False,
        "wp3_closed": False,
        "joint_wp7_threshold_frozen": False,
        "q1_closed": False,
        "outcome": "PASS",
    }


def main(arguments: list[str] | None = None) -> int:
    selected_arguments = (
        list(sys.argv[1:]) if arguments is None else list(arguments)
    )
    claim, validate_only, remaining = requested_claim(selected_arguments)
    if validate_only:
        if remaining:
            raise ValueError(
                "--validate-only does not accept execution arguments"
            )
        print(
            json.dumps(
                validate_only_summary(load_registry(DEFAULT_REGISTRY), claim),
                sort_keys=True,
            )
        )
        return 0
    original_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT_PATH), *remaining]
        return strict_runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
