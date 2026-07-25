#!/usr/bin/env python3
"""Run the frozen WP-7 cut-stability prerequisite matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. Requests for
FSR-07, WP-7, or Q1 closure fail before build or test execution.
"""

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
    "free_surface_wp7_cut_stability_qualification_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "a49cadbcbe1b56bf69e4520a5281fc942ac4dc9de82da4e3bdaa083d6334ab1f"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)
EXPECTED_SCOPE = (
    "WP-7 matrix-definition and currently executable prerequisite evidence "
    "only; seven frozen prospective tests remain absent, so this matrix does "
    "not close FSR-07, WP-7, or Q1."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr07_closure",
        "wp7_closure",
        "q1_closure",
    ],
    "diagnostic": (
        "Seven frozen manufactured-error, topology-policy, node-crossing, "
        "and simulation-exit tests are prospective, and production "
        "preconditioner and Krylov-spread evidence is unresolved."
    ),
}
EXPECTED_DISPOSITION = {
    "fsr07_closed": False,
    "wp7_closed": False,
    "q1_closed": False,
}
EXPECTED_FRACTIONS = [
    1.0e-8,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    0.1,
    0.25,
    0.49,
]
EXPECTED_ORIENTATIONS = [
    {"id": "axis", "normal": [1.0, 0.0, 0.0]},
    {"id": "oblique", "normal": [1.0, 0.73, 0.41]},
]
EXPECTED_H_LEVELS = [
    {"cells_per_axis": 2, "h_over_domain_length": 0.5},
    {
        "cells_per_axis": 3,
        "h_over_domain_length": 0.3333333333333333,
    },
    {"cells_per_axis": 4, "h_over_domain_length": 0.25},
]
EXPECTED_REGIMES = [
    {
        "id": "viscous",
        "density": 1.0,
        "viscosity": 1.0,
        "dt": 0.1,
        "convection": False,
        "advective_speed": 0.0,
    },
    {
        "id": "transient",
        "density": 1.0,
        "viscosity": 0.01,
        "dt": 0.001,
        "convection": False,
        "advective_speed": 0.0,
    },
    {
        "id": "advection",
        "density": 1.0,
        "viscosity": 0.001,
        "dt": 0.1,
        "convection": True,
        "advective_speed": 1.0,
    },
]
EXPECTED_TOPOLOGIES = [
    "connected",
    "disconnected_resolved",
    "rootless_subresolution",
    "continuous_node_crossing",
]
EXPECTED_REQUIRED_METRICS = {
    "velocity_l2_error",
    "velocity_h1_error",
    "pressure_l2_error_after_physical_nullspace_removal",
    "divergence_l2_norm",
    "stabilized_pressure_control_surrogate",
    "equilibrated_smallest_singular_value",
    "canonically_scaled_condition_number",
    "preconditioned_krylov_iterations",
    "preconditioned_spectrum_when_available",
    "aggregate_row_l1_norm",
    "aggregate_slave_master_distance_over_h",
    "rootless_feature_count_and_removed_volume",
    "solver_fallback_and_retry_count",
    "operator_and_solution_jump_at_node_crossing",
}
EXPECTED_UNRESOLVED_EVIDENCE = {
    "production_preconditioner_and_krylov_iteration_spread",
    "manufactured_stokes_and_navier_stokes_error_rates",
    "static_drop_filament_and_d18_d38_topology_transition_exits",
}
EXPECTED_EXECUTABLE_TESTS = {
    "FreeSurfaceCutStability.SelectedCombinedAggregateAndPressureStabilizationContractIsExplicit",
    "FreeSurfaceCutStability.ExactTargetFractionGeometryCoversAxisObliqueAndThreeHLevels",
    "FreeSurfaceCutStability.ManufacturedAffineQ1YoungLaplaceAndContactAngleBalanceToRoundoff",
    "FreeSurfaceCutStability.FixedTetraMeshGenericAndNearFeatureSweepHasNoFreePressureNullRows",
    "FreeSurfaceCutStability.FixedPhysicalCutThreeLevelRefinementBoundsPressureControlAndAggregation",
    "FreeSurfaceCutStability.PersistentMovingCutRefreshesAggregationAndRetainsMixedRankWithoutPressurePin",
    "FreeSurfaceCutStability.ProductionSmallCutAggregationPreservesPartialWallDirichletPerComponent",
    "FreeSurfaceCutStability.DisconnectedLiquidIslandsHaveZeroPhysicalCrossCouplingThroughDryStrip",
    "FreeSurfaceCutStability.FrozenFractionOrientationRefinementRegimeMatrixRecordsConditionAndIterations",
    "FreeSurfaceCutStabilityMPI.DistributedMovingCutRemainsStableAcrossBlockAndMetisPartitions",
    "FreeSurfaceCutStabilityMPI.LimitedMetisHaloFailsClosedOnIncompleteAggregationSupport",
    "FreeSurfaceCutStabilityMPI.TwoRankFractionOrientationRegimeMatrixMatchesSerial",
    "FreeSurfaceCutStabilityMPI.FourRankFixedCutIsInvariantAcrossBlockAndMetisPartitions",
    "FreeSurfaceCutStabilityMPI.FourRankFractionOrientationRegimeMatrixMatchesSerial",
}
EXPECTED_PROSPECTIVE_TESTS = {
    "FreeSurfaceCutStability.ManufacturedStokesAndNavierStokesErrorsConvergeAcrossCuts",
    "FreeSurfaceCutStability.ConnectedDisconnectedAndRootlessFeaturesReportTopologyPolicy",
    "FreeSurfaceCutStability.ContinuousNodeCrossingHasNoUnreportedOperatorOrSolutionJump",
    "FreeSurfaceCutStabilitySimulation.StaticCapsHaveNoMeshRelativeNumericalJump",
    "FreeSurfaceCutStabilitySimulation.TranslatingDropsConvergeAcrossCutOffsets",
    "FreeSurfaceCutStabilitySimulation.ResolvedFilamentsSeparatePhysicalAndRootlessEvents",
    "FreeSurfaceCutStabilitySimulation.D18D38ReportEveryFallbackAndFeatureDeletion",
}
EXPECTED_METHOD = {
    "name": "combined_p1_aggregate_space_vms_pspg_pressure_jump",
    "velocity_space": "continuous_p1_aggregate_constrained",
    "pressure_space": "continuous_p1_aggregate_constrained",
    "bulk_stabilization": "transient_equal_order_vms_pspg",
    "cut_stabilization": "cut_metadata_scaled_pressure_first_derivative_jump",
    "velocity_ghost_penalty": "disabled",
    "aggregate_root_policy": (
        "breadth_first_full_active_root_with_common_velocity_pressure_topology"
    ),
    "aggregate_constant_reproduction": "partition_of_unity_gated",
    "aggregate_coefficient_bound": ("row_l1_and_slave_master_distance_over_h_gated"),
    "pressure_datum": ("physical_natural_traction_anchor_without_algebraic_gauge"),
    "rootless_feature_policy": (
        "homogeneous_support_removal_reported_as_feature_deletion"
    ),
    "supported_polynomial_orders": [1],
}
EXPECTED_SIMULATION_CASES = {
    "static_caps",
    "translating_drops",
    "filaments",
    "d18_d38",
}
EXPECTED_THRESHOLD_KEYS = {
    "exact_fraction_geometry",
    "finite_conditioning",
    "pressure_control",
    "aggregate_amplification",
    "krylov_and_manufactured_errors",
}
EXPECTED_DERIVATION = "Documentation/free_surface_wp7_combined_p1_method.md"


def require_unique_test_list(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"WP-7 {label} must be a nonempty list")
    if any(
        not isinstance(test, str)
        or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", test)
        for test in value
    ):
        raise ValueError(f"WP-7 {label} contains an invalid test name")
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"WP-7 {label} contains duplicate tests")
    return result


def validate_wp7_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-7 qualification scope changed after freeze")
    if registry.get("closure_request_policy") != (EXPECTED_CLOSURE_REQUEST_POLICY):
        raise ValueError("WP-7 closure-request policy changed after freeze")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("WP-7 qualification disposition changed")
    if registry.get("closure_state") != ("BLOCKED_BY_FROZEN_PROSPECTIVE_EVIDENCE"):
        raise ValueError("WP-7 v2 must remain explicitly blocked")
    if registry.get("selected_method") != EXPECTED_METHOD:
        raise ValueError("WP-7 selected method changed after freeze")
    if registry.get("matching_derivation") != EXPECTED_DERIVATION:
        raise ValueError("WP-7 matching derivation path changed after freeze")
    derivation_path = SCRIPT_PATH.parents[3] / EXPECTED_DERIVATION
    if not derivation_path.is_file():
        raise ValueError("WP-7 matching derivation is missing")

    limitations = registry.get("method_limitations")
    if (
        not isinstance(limitations, list)
        or len(limitations) < 6
        or any(not isinstance(value, str) or not value.strip() for value in limitations)
    ):
        raise ValueError("WP-7 method limitations are incomplete")

    axes = registry.get("case_axes")
    if not isinstance(axes, dict) or set(axes) != {
        "volume_fractions",
        "orientations",
        "h_levels",
        "regimes",
        "mpi_ranks",
        "polynomial_orders",
        "topologies",
    }:
        raise ValueError("WP-7 case axes are incomplete")
    if axes["volume_fractions"] != EXPECTED_FRACTIONS:
        raise ValueError("WP-7 target fractions changed after freeze")
    if axes["orientations"] != EXPECTED_ORIENTATIONS:
        raise ValueError("WP-7 orientations changed after freeze")
    if axes["h_levels"] != EXPECTED_H_LEVELS:
        raise ValueError("WP-7 h levels changed after freeze")
    if axes["regimes"] != EXPECTED_REGIMES:
        raise ValueError("WP-7 regimes changed after freeze")
    if axes["mpi_ranks"] != [1, 2, 4]:
        raise ValueError("WP-7 rank list must be exactly 1, 2, and 4")
    if axes["polynomial_orders"] != [1]:
        raise ValueError("WP-7 selected method supports only frozen P1")
    if axes["topologies"] != EXPECTED_TOPOLOGIES:
        raise ValueError("WP-7 topology list changed after freeze")
    expected_cross_product = (
        len(EXPECTED_FRACTIONS)
        * len(EXPECTED_ORIENTATIONS)
        * len(EXPECTED_H_LEVELS)
        * len(EXPECTED_REGIMES)
        * len(axes["mpi_ranks"])
        * len(axes["polynomial_orders"])
    )
    if registry.get("full_cross_product_case_count") != expected_cross_product:
        raise ValueError("WP-7 full cross-product count is inconsistent")

    metrics = registry.get("required_metrics")
    if (
        not isinstance(metrics, list)
        or len(metrics) != len(set(metrics))
        or set(metrics) != EXPECTED_REQUIRED_METRICS
    ):
        raise ValueError("WP-7 required metric set is incomplete")
    thresholds = registry.get("numeric_threshold_basis")
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != EXPECTED_THRESHOLD_KEYS
        or any(
            not isinstance(value, str) or not value.strip()
            for value in thresholds.values()
        )
    ):
        raise ValueError("WP-7 threshold basis is incomplete")

    unresolved = registry.get("unresolved_required_evidence")
    if (
        not isinstance(unresolved, list)
        or len(unresolved) != len(set(unresolved))
        or set(unresolved) != EXPECTED_UNRESOLVED_EVIDENCE
    ):
        raise ValueError("WP-7 unresolved evidence declaration changed")

    simulations = registry.get("simulation_exits")
    if not isinstance(simulations, list) or len(simulations) != 4:
        raise ValueError("WP-7 simulation exit list is incomplete")
    simulation_cases: set[str] = set()
    for simulation in simulations:
        if not isinstance(simulation, dict) or set(simulation) != {
            "case",
            "status",
            "contract",
        }:
            raise ValueError("WP-7 simulation exit entry is invalid")
        case = simulation.get("case")
        if case in simulation_cases:
            raise ValueError(f"duplicate WP-7 simulation exit: {case}")
        simulation_cases.add(case)
        if simulation.get("status") != "REQUIRED_PROSPECTIVE":
            raise ValueError(f"WP-7 simulation exit is not blocked: {case}")
        if (
            not isinstance(simulation.get("contract"), str)
            or not simulation["contract"].strip()
        ):
            raise ValueError(f"WP-7 simulation contract is empty: {case}")
    if simulation_cases != EXPECTED_SIMULATION_CASES:
        raise ValueError("WP-7 simulation case set is incomplete")

    frozen_tests = {test for group in registry["groups"] for test in group["tests"]}
    executable = require_unique_test_list(
        registry.get("executable_tests"), "executable tests"
    )
    prospective = require_unique_test_list(
        registry.get("prospective_tests"), "prospective tests"
    )
    if executable != EXPECTED_EXECUTABLE_TESTS:
        raise ValueError("WP-7 executable test set changed after freeze")
    if prospective != EXPECTED_PROSPECTIVE_TESTS:
        raise ValueError("WP-7 prospective test set changed after freeze")
    if executable & prospective:
        raise ValueError("WP-7 executable and prospective tests overlap")
    if executable | prospective != frozen_tests:
        raise ValueError(
            "WP-7 executable/prospective partition does not cover the matrix"
        )

    group_by_id = {group["id"]: group for group in registry["groups"]}
    if set(group_by_id) != {
        "wp7_finite_foundation_serial",
        "wp7_required_regimes_serial",
        "wp7_partition_mpi_2",
        "wp7_partition_mpi_4",
        "wp7_simulation_exits_serial",
    }:
        raise ValueError("WP-7 execution group set changed after freeze")
    if group_by_id["wp7_partition_mpi_2"]["mpi_ranks"] != 2:
        raise ValueError("WP-7 two-rank group has the wrong rank count")
    if group_by_id["wp7_partition_mpi_2"]["gtest_output_copies"] != 1:
        raise ValueError("WP-7 two-rank group has the wrong output count")
    if group_by_id["wp7_partition_mpi_2"]["execution"]["wall_time_seconds"] != 43200:
        raise ValueError("WP-7 two-rank group has the wrong wall-time limit")
    if group_by_id["wp7_partition_mpi_4"]["mpi_ranks"] != 4:
        raise ValueError("WP-7 four-rank group has the wrong rank count")
    if group_by_id["wp7_partition_mpi_4"]["gtest_output_copies"] != 1:
        raise ValueError("WP-7 four-rank group has the wrong output count")
    if group_by_id["wp7_partition_mpi_4"]["execution"]["wall_time_seconds"] != 43200:
        raise ValueError("WP-7 four-rank group has the wrong wall-time limit")
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = "free_surface_wp7_cut_stability_v2"
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-7"
strict_runner.__doc__ = __doc__

_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text


def load_registry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if strict_runner.sha256_file(resolved) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-7 frozen registry bytes changed")
    return validate_wp7_contract(_shared_load_registry(resolved))


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
        value["wp7_scope_state"] = "BLOCKED_BY_FROZEN_PROSPECTIVE_EVIDENCE"
        value["wp7_full_closure_claimed"] = False
        value["qualification_scope"] = EXPECTED_SCOPE
        value["requested_claim"] = "low_level_prerequisite"
        value["qualification_disposition"] = EXPECTED_DISPOSITION
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-7 cut-stability qualification record",
            1,
        )
        value = value.replace(
            "\n\n",
            (
                "\n\n> Scope state: required prospective evidence remains "
                "release blocking; this v2 record cannot establish WP-7 "
                "closure.\n\n"
            ),
            1,
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-7 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this blocked v2 matrix may make.\n"
            "      FSR-07, WP-7, and Q1 closure are rejected.\n"
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
            f"unsupported WP-7 requested claim {claim!r}; expected {allowed!r}"
        )
    return claim, parsed.validate_only, remaining


if __name__ == "__main__":
    try:
        _claim, _validate_only, _remaining_arguments = requested_claim(sys.argv[1:])
        if _validate_only:
            if _remaining_arguments:
                raise ValueError("--validate-only does not accept execution arguments")
            _registry = load_registry(DEFAULT_REGISTRY)
            print(
                json.dumps(
                    {
                        "matrix_id": _registry["matrix_id"],
                        "status": _registry["status"],
                        "requested_claim": _claim,
                        "closure_state": _registry["closure_state"],
                        "group_count": len(_registry["groups"]),
                        "test_count": sum(
                            len(group["tests"]) for group in _registry["groups"]
                        ),
                        "executable_test_count": len(_registry["executable_tests"]),
                        "prospective_test_count": len(_registry["prospective_tests"]),
                        "serial_quantitative_gate_count": len(
                            _registry["quantitative_evidence"]
                        ),
                        **_registry["qualification_disposition"],
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
