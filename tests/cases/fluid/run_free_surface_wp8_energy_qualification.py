#!/usr/bin/env python3
"""Run the frozen WP-8 geometry-coupling and energy prerequisite matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. Requests for
FSR-09, WP-8, or complete-energy closure fail before build or test execution.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
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

_SHARED_RUNNER_IMPORT_PATH = (
    SCRIPT_DIRECTORY / "run_free_surface_wp2_geometry_qualification.py"
)
_SHARED_RUNNER_MODULE_NAME = (
    "_free_surface_wp8_private_geometry_qualification_runner"
)
_SHARED_RUNNER_SPEC = importlib.util.spec_from_file_location(
    _SHARED_RUNNER_MODULE_NAME,
    _SHARED_RUNNER_IMPORT_PATH,
)
if _SHARED_RUNNER_SPEC is None or _SHARED_RUNNER_SPEC.loader is None:
    raise ImportError(f"cannot load shared runner: {_SHARED_RUNNER_IMPORT_PATH}")
strict_runner = importlib.util.module_from_spec(_SHARED_RUNNER_SPEC)
sys.modules[_SHARED_RUNNER_MODULE_NAME] = strict_runner
_SHARED_RUNNER_SPEC.loader.exec_module(strict_runner)


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
PARAMETERS_HEADER = (
    REPOSITORY_ROOT / "Code" / "Source" / "solver" / "Parameters.h"
)
PARAMETERS_SOURCE = (
    REPOSITORY_ROOT / "Code" / "Source" / "solver" / "Parameters.cpp"
)
APPLICATION_DRIVER_SOURCE = (
    REPOSITORY_ROOT
    / "Code"
    / "Source"
    / "solver"
    / "Application"
    / "Core"
    / "ApplicationDriver.cpp"
)
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp8_energy_qualification_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "9007fb06e64cf092d2d57e6ea49fda5fe99798e1b52b185e9d1e3fba0bb9e9b6"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)

EXPECTED_ARCHITECTURE_RECORD = (
    "Documentation/free_surface_wp8_geometry_energy_architecture.md"
)
EXPECTED_FREEZE_SCOPE = (
    "This matrix freezes prerequisite evidence for explicit production selection "
    "between generalized-alpha and backward Euler, the backward-Euler endpoint "
    "transaction, the selected generated-state outer fixed point, transactional "
    "rollback, discrete surface/wall/volume functional evaluation, a "
    "fixed-topology first variation, extension refresh behavior, topology-event "
    "diagnostics, and the accepted/rejected level-set maintenance ledger. The "
    "ledger evidence covers ordered transport, limiting, reinitialization, "
    "geometry-reconciliation, and global-correction potential-change rows with "
    "explicit revision provenance; it does not establish a backward-Euler energy "
    "balance, an energy-stable split, a complete physical and numerical energy "
    "identity, outer contraction, topology-event acceptance, or any WP-8 "
    "simulation exit."
)
EXPECTED_SELECTED_STRATEGY = {
    "strategy": "energy_stable_partitioned_outer_iteration_target",
    "implemented_solver_form": (
        "regenerate_generated_state_then_solve_frozen_inner_newton_problem"
    ),
    "generated_state": [
        "authoritative_cut_geometry",
        "projected_curvature",
        "state_dependent_constraints",
        "bounded_velocity_extension_map",
    ],
    "inner_problem": "R(u,G_k)=0_with_G_k_frozen",
    "current_convergence_certificate": (
        "zero_inner_update_after_regenerating_G_from_the_same_algebraic_state"
    ),
    "complete_shape_tangent_selected": False,
    "discrete_energy_stability_proved": False,
    "wp8_closure_claimed": False,
}
EXPECTED_TRANSIENT_SCHEME_PREREQUISITE = {
    "xml_parameter": "Transient_time_integration_scheme",
    "default": "GeneralizedAlpha",
    "exact_supported_values": [
        "GeneralizedAlpha",
        "BackwardEuler",
    ],
    "backward_euler_contract": {
        "stage_alpha_f": 1.0,
        "spectral_radius": "inapplicable",
        "pde_rate_initialization": False,
        "generated_state_time": "accepted_endpoint",
    },
    "qualification_boundary": (
        "Selection, rejection, and one-step endpoint transaction only; no "
        "backward-Euler free-surface energy identity, refinement result, or "
        "WP-8 simulation exit is claimed."
    ),
}
EXPECTED_IMPLEMENTED_ENERGY_CHANNELS = [
    "liquid_gas_surface_functional",
    "young_wetted_wall_functional",
    "volume_constraint_potential",
    "accepted_dynamic_contact_line_dissipation",
    "accepted_sharp_wall_slip_dissipation",
    "maintenance_transport_surface_wall_volume_potential_change",
    "maintenance_limiting_surface_wall_volume_potential_change",
    "maintenance_reinitialization_surface_wall_volume_potential_change",
    "maintenance_geometry_reconciliation_surface_wall_volume_potential_change",
    "maintenance_global_correction_surface_wall_volume_potential_change",
    "maintenance_rejected_attempt_zero_accepted_contribution",
    "maintenance_zero_row_accepted_and_rejected_attempt_outcomes",
]
EXPECTED_MISSING_ENERGY_CHANNELS = [
    "kinetic_energy",
    "gravitational_energy",
    "gas_or_compressibility_energy_when_applicable",
    "bulk_viscous_dissipation",
    "external_pressure_work",
    "body_force_work",
    "vms_pspg_numerical_work",
    "ghost_penalty_or_aggregation_numerical_work",
    "extension_numerical_work",
    "pruning_numerical_work",
    "complete_rejected_attempt_energy_account_across_all_channels",
]
EXPECTED_THRESHOLD_BASIS_KEYS = {
    "functional_first_variation",
    "ghost_exclusion",
    "outer_fixed_point",
    "maintenance_ledger",
    "energy_account",
}
EXPECTED_GROUP_TESTS = {
    "discrete_functional_variation_serial": (
        "geometry",
        1,
        1,
        [
            "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
            "FreeSurfaceGeometrySnapshot.DiscreteFunctionalExcludesGhostRuleContributions",
        ],
    ),
    "generated_state_outer_fixed_point_serial": (
        "systems",
        1,
        1,
        [
            "NewtonSolverExternalStateFixedPoint.RequiresFreshlyRegeneratedResidualBeforeConvergence",
            "NewtonSolverExternalStateFixedPoint.OuterIterationFailureRestoresSolutionHistoryAndRateState",
            "NewtonSolverExternalStateFixedPoint.RefreshFailureRestoresAuxiliaryAndBorderedStateBeforeCallback",
            "NewtonSolverExternalStateFixedPoint.StageTimeConstraintIsEstablishedBeforeSnapshotAndRollback",
            "NewtonSolverExternalStateFixedPoint.ReallocatesJacobianAfterOuterConstraintSparsityChange",
            "NewtonSolverExternalStateFixedPoint.FirstGeneratedConstraintRefreshDefinesCanonicalRollbackEntry",
            "TimeLoopConvergence.BackwardEulerExternalStateFixedPointPreservesEndpointTransaction",
        ],
    ),
    "application_geometry_energy_prerequisites_serial": (
        "application",
        1,
        1,
        [
            "GeneralSimulationParameters.ParsesOptionalTransientTimeIntegrationScheme",
            "ApplicationDriverLevelSetWorkflows.SelectsBackwardEulerAndRejectsUnsupportedTransientScheme",
            "ApplicationDriverLevelSetWorkflows.OuterFixedPointReportsFrozenInnerJacobianGeometry",
            "ApplicationDriverLevelSetWorkflows.CutTopologyChangeTraceIdentifiesNonsmoothNewtonEvent",
            "ApplicationDriverLevelSetWorkflows.RefreshesMultipleGeneratedCutDomainsIntoOneContext",
            "ApplicationDriverLevelSetWorkflows.AcceptedFunctionalUsesAuthoritativeSnapshotAndRecordsGlobalState",
            "ApplicationDriverLevelSetWorkflows.AlgebraicExtensionRefreshReprojectsStateAndChangesRevision",
            "ApplicationDriverLevelSetWorkflows.ReducedD38MapFailureUsesBoundedRefreshNeutralFallback",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerPublishesReinitializationOnlyAtCommit",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerKeepsReinitializationAndCorrectionAdditive",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerReportsSameStateAsZeroWork",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerRollbackPublishesNoAcceptedRow",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerRejectsDiscontinuousRows",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerPublishesZeroRowAttemptOutcomes",
            "ApplicationDriverLevelSetWorkflows.MaintenanceWorkLedgerRequiresExplicitCutTopologyProvenance",
            "ApplicationDriverLevelSetWorkflows.ConvergedMaintenanceAppliesOneRepresentationDeltaToEveryHistoryLevel",
            "ApplicationDriverConservativePhaseCandidatesTest.StagesAndCommitsTheTransportedPhaseAgainstAuthoritativeGeometry",
            "ApplicationDriverLevelSetVolumeCorrection.ReportsAuthoritativeFreeSurfacePotentialChange",
            "ApplicationDriverLevelSetVolumeCorrection.CandidateGeometryFailureRestoresCompleteMaintenanceTransaction",
        ],
    ),
    "distributed_geometry_refresh_prerequisite": (
        "application_mpi",
        2,
        2,
        [
            "ApplicationDriverLevelSetWorkflowsMPI.ActiveCutRefreshUsesCommunicatorWideSortedBoundaryMarkerUnion",
            "ApplicationDriverLevelSetWorkflowsMPI.MaintenanceWorkRowsAreIdenticalAcrossTwoRanks",
            "ApplicationDriverLevelSetWorkflowsMPI.MaintenanceAlgebraicRevisionRejectsRankLocalSlices",
        ],
    ),
}
EXPECTED_QUANTITATIVE_EVIDENCE = {
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_first_variation_fd_case_count",
    ): ("integer", "equal", 2),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_surface_area_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_surface_energy_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_wetted_wall_area_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_young_wall_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_volume_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_volume_potential_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "functional_total_fd_max_relative_error",
    ): ("real", "less_than_or_equal", 2.0e-7),
    (
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalExcludesGhostRuleContributions",
        "functional_first_variation_ghost_contribution_count",
    ): ("integer", "equal", 0),
}
EXPECTED_METHOD_EXITS = {
    "partitioned_split_discrete_energy_argument",
    "backward_euler_constant_surface_tension_closed_balance",
    "complete_accepted_state_energy_dissipation_external_and_numerical_work_ledger",
    "common_stage_geometry_state_transport_contact_and_maintenance_contract",
    "same_state_full_geometry_curvature_constraint_and_extension_refresh_neutrality",
    "geometry_dependent_residual_directional_derivative_matrix",
    "outer_fixed_point_contraction_under_h_dt_cut_shift_and_mpi_refinement",
    "topology_detection_snapshot_invalidation_restart_or_rejection_energy_jump_and_minimum_feature_policy",
    "generalized_alpha_stage_consistency_after_backward_euler_closure",
    "energy_residual_threshold_and_refinement_rule_frozen_before_simulation",
}
EXPECTED_SIMULATION_EXITS = {
    "static_cap_complete_energy_residual_refinement",
    "capillary_relaxation_complete_energy_residual_refinement",
    "linear_capillary_wave_complete_energy_residual_refinement",
    "droplet_oscillation_complete_energy_residual_refinement",
    "sloshing_complete_energy_residual_refinement",
    "wetting_relaxation_complete_energy_residual_refinement",
}
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr09_closure",
        "wp8_closure",
        "q3_closure",
        "q4_closure",
        "q5_closure",
        "complete_energy_law",
    ],
    "reject_any_claim_suffix": "_closure",
    "diagnostic": (
        "The frozen prerequisite slice verifies the implemented maintenance "
        "ledger but does not prove split-energy stability, complete the "
        "physical and numerical energy identity, establish contraction, "
        "define topology-event acceptance, or execute the required "
        "simulations."
    ),
}
EXPECTED_DISPOSITION = {
    "prerequisite_evidence_frozen": True,
    "fsr09_closed": False,
    "wp8_closed": False,
    "q3_closed": False,
    "q4_closed": False,
    "q5_closed": False,
    "complete_energy_law_available": False,
}
EXPECTED_SCOPE = (
    "WP-8 transient-scheme, endpoint-transaction, maintenance-ledger, and "
    "existing low-level prerequisite evidence only; this matrix does not "
    "close FSR-09, WP-8, Q3, Q4, or Q5 and does not establish a complete "
    "discrete energy law."
)


def _require_tokens_in_order(
    source: str, tokens: tuple[str, ...], label: str
) -> None:
    offset = 0
    for token in tokens:
        location = source.find(token, offset)
        if location < 0:
            raise ValueError(
                f"WP-8 production source contract lost {label}: {token}"
            )
        offset = location + len(token)


def validate_wp8_production_source_contract() -> dict[str, int]:
    parameters_header = PARAMETERS_HEADER.read_text(encoding="utf-8")
    parameters_source = PARAMETERS_SOURCE.read_text(encoding="utf-8")
    application_source = APPLICATION_DRIVER_SOURCE.read_text(
        encoding="utf-8"
    )

    if (
        parameters_header.count(
            "Parameter<std::string> transient_time_integration_scheme;"
        )
        != 1
    ):
        raise ValueError(
            "WP-8 production source contract requires one typed transient "
            "scheme parameter"
        )
    for registration in (
        (
            'set_parameter("Spectral_radius_of_infinite_time_step", 0.5, '
            "!required, spectral_radius_of_infinite_time_step);"
        ),
        (
            'set_parameter("Transient_time_integration_scheme", '
            '"GeneralizedAlpha",\n'
            "      !required, transient_time_integration_scheme);"
        ),
    ):
        if parameters_source.count(registration) != 1:
            raise ValueError(
                "WP-8 production source contract requires optional "
                "spectral-radius and transient-scheme defaults"
            )

    resolver_start = application_source.find(
        "TransientTimeIntegrationSelection "
        "resolveTransientTimeIntegrationSelection("
    )
    resolver_end = application_source.find(
        "double parseDoubleEnv(", resolver_start
    )
    if resolver_start < 0 or resolver_end < 0:
        raise ValueError(
            "WP-8 production source contract cannot locate the transient "
            "scheme resolver"
        )
    resolver = application_source[resolver_start:resolver_end]
    _require_tokens_in_order(
        resolver,
        (
            "parseTransientTimeIntegrationScheme(",
            "if (scheme == "
            "svmp::FE::timestepping::SchemeKind::BackwardEuler)",
            "selection.generalized_alpha_rho_inf = std::nullopt;",
            "selection.stage_alpha_f = svmp::FE::Real{1.0};",
            "return selection;",
            "const double rho_inf =",
            "spectral_radius_of_infinite_time_step.value();",
            "generalizedAlphaFirstOrderFromRhoInf(",
        ),
        "fail-closed scheme semantics",
    )
    scheme_table_match = re.search(
        (
            r"constexpr\s+std::array\s*<\s*std::pair\s*<"
            r"\s*std::string_view\s*,"
            r"\s*svmp::FE::timestepping::SchemeKind\s*>\s*,"
            r"\s*(?P<declared_count>\d+)\s*>\s*"
            r"kTransientTimeIntegrationSchemes\s*\{\{"
            r"(?P<entries>.*?)"
            r"\}\};"
        ),
        application_source,
        flags=re.DOTALL,
    )
    if scheme_table_match is None:
        raise ValueError(
            "WP-8 production source contract requires an exact-two "
            "transient scheme table"
        )
    scheme_entry_pattern = re.compile(
        (
            r"\{\s*\"(?P<name>[^\"]+)\"\s*,"
            r"\s*svmp::FE::timestepping::SchemeKind::"
            r"(?P<kind>[A-Za-z0-9_]+)\s*\}"
        )
    )
    scheme_entries_text = scheme_table_match.group("entries")
    scheme_entries = [
        (match.group("name"), match.group("kind"))
        for match in scheme_entry_pattern.finditer(scheme_entries_text)
    ]
    scheme_table_residue = scheme_entry_pattern.sub(
        "", scheme_entries_text
    ).strip(" \t\r\n,")
    expected_scheme_entries = [
        ("GeneralizedAlpha", "GeneralizedAlpha"),
        ("BackwardEuler", "BackwardEuler"),
    ]
    if (
        int(scheme_table_match.group("declared_count")) != 2
        or scheme_entries != expected_scheme_entries
        or scheme_table_residue
    ):
        raise ValueError(
            "WP-8 production source contract requires exactly two canonical "
            "transient scheme entries"
        )
    _require_tokens_in_order(
        application_source,
        (
            "\"'. Supported values are exactly 'GeneralizedAlpha' and \"",
            "\"'BackwardEuler'.\"",
        ),
        "exact unsupported scheme diagnostic",
    )

    transient_start = application_source.find(
        "void ApplicationDriver::runTransient("
    )
    transient_end = application_source.find(
        "void ApplicationDriver::outputResults(", transient_start
    )
    if transient_start < 0 or transient_end < 0:
        raise ValueError(
            "WP-8 production source contract cannot isolate runTransient"
        )
    transient = application_source[transient_start:transient_end]
    _require_tokens_in_order(
        transient,
        (
            "resolveTransientTimeIntegrationSelection(",
            "svmp::FE::timestepping::TimeLoopOptions opts{};",
            "opts.scheme = transient_scheme.scheme;",
            "if (transient_scheme.generalized_alpha_rho_inf.has_value())",
            "opts.initialize_first_order_rate_from_pde =",
            "opts.scheme == "
            "svmp::FE::timestepping::SchemeKind::GeneralizedAlpha &&",
            "sim.time_history->repack(*sim.backend);",
        ),
        "pre-solve scheme selection and rate initialization",
    )
    if transient.count("transient_scheme.stage_alpha_f") != 2:
        raise ValueError(
            "WP-8 production source contract requires the resolved endpoint "
            "stage at both contact-stage sites"
        )
    if "generalizedAlphaFirstOrderFromRhoInf(" in transient:
        raise ValueError(
            "WP-8 production source contract forbids generalized-alpha "
            "parameter construction in the scheme-independent transient body"
        )
    if 'oopCout() << " rho_inf=n/a";' not in transient:
        raise ValueError(
            "WP-8 production source contract requires explicit inapplicable "
            "spectral-radius logging"
        )
    return {
        "exact_supported_scheme_count": 2,
        "endpoint_stage_sites": 2,
        "optional_scheme_defaults": 1,
        "scheme_guarded_rate_initialization_sites": 1,
    }


def _validate_unqualified_exits(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"WP-8 {label} list is missing")
    identifiers: set[str] = set()
    for entry in value:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or entry.get("status") != "REQUIRED_NOT_CLAIMED"
            or not isinstance(entry.get("id"), str)
        ):
            raise ValueError(f"WP-8 {label} entry is invalid")
        identifiers.add(entry["id"])
    if identifiers != expected or len(identifiers) != len(value):
        raise ValueError(f"WP-8 {label} list changed after freeze")


def validate_wp8_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("prospective_tests") != []:
        raise ValueError("WP-8 cannot freeze with prospective tests")
    if registry.get("freeze_scope") != EXPECTED_FREEZE_SCOPE:
        raise ValueError("WP-8 prerequisite freeze scope changed")
    if registry.get("architecture_record") != EXPECTED_ARCHITECTURE_RECORD:
        raise ValueError("WP-8 architecture-record path changed after freeze")
    if registry.get("selected_ad5_strategy") != EXPECTED_SELECTED_STRATEGY:
        raise ValueError("WP-8 AD-5 strategy changed after freeze")
    if (
        registry.get("transient_scheme_prerequisite")
        != EXPECTED_TRANSIENT_SCHEME_PREREQUISITE
    ):
        raise ValueError(
            "WP-8 transient-scheme prerequisite changed after freeze"
        )

    coverage = registry.get("current_energy_account_coverage")
    if (
        not isinstance(coverage, dict)
        or set(coverage)
        != {
            "implemented_low_level_channels",
            "not_yet_complete_channels",
            "complete_balance_residual_available",
        }
        or coverage.get("implemented_low_level_channels")
        != EXPECTED_IMPLEMENTED_ENERGY_CHANNELS
        or coverage.get("not_yet_complete_channels") != EXPECTED_MISSING_ENERGY_CHANNELS
        or coverage.get("complete_balance_residual_available") is not False
    ):
        raise ValueError("WP-8 energy-account coverage changed after freeze")

    threshold_basis = registry.get("numeric_threshold_basis")
    if (
        not isinstance(threshold_basis, dict)
        or set(threshold_basis) != EXPECTED_THRESHOLD_BASIS_KEYS
        or any(
            not isinstance(value, str) or not value.strip()
            for value in threshold_basis.values()
        )
    ):
        raise ValueError("WP-8 numeric threshold basis is incomplete")

    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("WP-8 execution groups are missing")
    actual_groups: dict[str, tuple[str, int, int, list[str]]] = {}
    all_tests: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("WP-8 execution group must be an object")
        group_id = group.get("id")
        tests = group.get("tests")
        if not isinstance(group_id, str) or not isinstance(tests, list):
            raise ValueError("WP-8 execution group id or tests are invalid")
        if group_id in actual_groups:
            raise ValueError(f"duplicate WP-8 execution group: {group_id}")
        actual_groups[group_id] = (
            group.get("binary"),
            group.get("mpi_ranks"),
            group.get("gtest_output_copies"),
            tests,
        )
        for test in tests:
            if not isinstance(test, str) or not re.fullmatch(
                r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", test
            ):
                raise ValueError(f"invalid WP-8 test name in {group_id}")
            if test in all_tests:
                raise ValueError(f"duplicate WP-8 test across groups: {test}")
            all_tests.add(test)
        if group.get("recorded_properties", []) != []:
            raise ValueError("WP-8 distributed recorded-property gates are not frozen")
    if actual_groups != EXPECTED_GROUP_TESTS:
        raise ValueError("WP-8 execution groups changed after freeze")

    quantitative = registry.get("quantitative_evidence")
    if not isinstance(quantitative, list):
        raise ValueError("WP-8 quantitative evidence list is missing")
    actual_quantitative: dict[tuple[str, str], tuple[str, str, int | float]] = {}
    for evidence in quantitative:
        if not isinstance(evidence, dict) or set(evidence) != {
            "test",
            "property",
            "type",
            "relation",
            "threshold",
        }:
            raise ValueError("invalid WP-8 quantitative evidence entry")
        key = (evidence.get("test"), evidence.get("property"))
        if key in actual_quantitative:
            raise ValueError(f"duplicate WP-8 quantitative evidence: {key}")
        actual_quantitative[key] = (
            evidence.get("type"),
            evidence.get("relation"),
            evidence.get("threshold"),
        )
    if actual_quantitative != EXPECTED_QUANTITATIVE_EVIDENCE:
        raise ValueError("WP-8 quantitative gates changed after freeze")

    _validate_unqualified_exits(
        registry.get("unqualified_required_method_exits"),
        EXPECTED_METHOD_EXITS,
        "required method exits",
    )
    _validate_unqualified_exits(
        registry.get("unqualified_required_simulations"),
        EXPECTED_SIMULATION_EXITS,
        "required simulation exits",
    )
    claims = registry.get("closure_contract")
    if (
        not isinstance(claims, list)
        or not claims
        or any(
            not isinstance(claim, dict)
            or set(claim) != {"claim", "evidence"}
            or not isinstance(claim.get("claim"), str)
            or not claim["claim"].startswith("prerequisite_")
            for claim in claims
        )
    ):
        raise ValueError("WP-8 evidence contracts must be prerequisite-only")
    if registry.get("closure_request_policy") != EXPECTED_CLOSURE_REQUEST_POLICY:
        raise ValueError("WP-8 closure-request policy changed after freeze")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("WP-8 nonclosure disposition changed after freeze")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-8 qualification scope changed after freeze")
    validate_wp8_production_source_contract()
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = "free_surface_wp8_energy_prerequisite_v2"
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-8"
strict_runner.__doc__ = __doc__

if not hasattr(strict_runner, "_wp8_base_load_registry"):
    strict_runner._wp8_base_load_registry = strict_runner.load_registry
    strict_runner._wp8_base_write_json = strict_runner.write_json
    strict_runner._wp8_base_write_text = strict_runner.write_text
_shared_load_registry = strict_runner._wp8_base_load_registry
_shared_write_json = strict_runner._wp8_base_write_json
_shared_write_text = strict_runner._wp8_base_write_text


def load_registry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if strict_runner.sha256_file(resolved) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-8 frozen registry bytes changed")
    return validate_wp8_contract(_shared_load_registry(resolved))


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
        value["selected_ad5_strategy"] = EXPECTED_SELECTED_STRATEGY
        value["transient_scheme_prerequisite"] = (
            EXPECTED_TRANSIENT_SCHEME_PREREQUISITE
        )
        value["matrix_sha256"] = EXPECTED_REGISTRY_SHA256
        value["qualification_disposition"] = EXPECTED_DISPOSITION
        value["requested_claim"] = "low_level_prerequisite"
        value["unqualified_method_exit_count"] = len(EXPECTED_METHOD_EXITS)
        value["unqualified_simulation_exit_count"] = len(EXPECTED_SIMULATION_EXITS)
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-8 geometry-coupling and energy prerequisite qualification record",
            1,
        )
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            + "The frozen registry records "
            + str(len(EXPECTED_METHOD_EXITS))
            + " unresolved method exits and "
            + str(len(EXPECTED_SIMULATION_EXITS))
            + " unresolved simulation exits.\n"
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(
    arguments: list[str],
) -> tuple[str, bool, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-8 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this low-level matrix may establish.\n"
            "      Every closure claim and complete_energy_law are rejected.\n"
            "  --validate-only\n"
            "      Validate the frozen schema and claim boundary without builds.\n"
            "  --list-only --geometry-binary PATH --systems-binary PATH\n"
            "      --application-binary PATH --application-mpi-binary PATH\n"
            "      Check frozen test names without executing tests or writing "
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
    rejected_suffix = EXPECTED_CLOSURE_REQUEST_POLICY["reject_any_claim_suffix"]
    if claim in rejected or claim.endswith(rejected_suffix):
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != allowed:
        raise ValueError(
            f"unsupported WP-8 requested claim {claim!r}; expected {allowed!r}"
        )
    return claim, parsed.validate_only, parsed.list_only, remaining


def validation_summary(registry: dict[str, Any], claim: str) -> dict[str, Any]:
    return {
        "matrix_id": registry["matrix_id"],
        "matrix_sha256": EXPECTED_REGISTRY_SHA256,
        "status": registry["status"],
        "requested_claim": claim,
        "prospective_test_count": len(registry["prospective_tests"]),
        "group_count": len(registry["groups"]),
        "test_count": sum(len(group["tests"]) for group in registry["groups"]),
        "serial_quantitative_gate_count": len(registry["quantitative_evidence"]),
        "unqualified_method_exit_count": len(
            registry["unqualified_required_method_exits"]
        ),
        "unqualified_simulation_exit_count": len(
            registry["unqualified_required_simulations"]
        ),
        "production_source_contract": (
            validate_wp8_production_source_contract()
        ),
        **registry["qualification_disposition"],
        "outcome": "PASS_PREREQUISITE_NONCLOSURE",
    }


def tests_for_binary(registry: dict[str, Any], binary_key: str) -> list[str]:
    return [
        test
        for group in registry["groups"]
        if group["binary"] == binary_key
        for test in group["tests"]
    ]


def parse_list_binary_arguments(
    arguments: list[str],
) -> dict[str, Path]:
    parser = argparse.ArgumentParser(prog=f"{SCRIPT_PATH.name} --list-only")
    parser.add_argument("--geometry-binary", type=Path, required=True)
    parser.add_argument("--systems-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parser.add_argument("--application-mpi-binary", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    return {
        "geometry": parsed.geometry_binary.resolve(),
        "systems": parsed.systems_binary.resolve(),
        "application": parsed.application_binary.resolve(),
        "application_mpi": parsed.application_mpi_binary.resolve(),
    }


def require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"test binary is not executable: {path}")


def run_list_only(claim: str, remaining: list[str]) -> int:
    binaries = parse_list_binary_arguments(remaining)
    for binary in binaries.values():
        require_executable(binary)
    registry = load_registry(DEFAULT_REGISTRY)
    missing_by_binary: dict[str, list[str]] = {}
    listed_counts: dict[str, int] = {}
    binary_hashes: dict[str, str] = {}
    for key, binary in binaries.items():
        expected = set(tests_for_binary(registry, key))
        listed = strict_runner.listed_gtests(binary)
        missing_by_binary[key] = sorted(expected - listed)
        listed_counts[key] = len(expected & listed)
        binary_hashes[key] = strict_runner.sha256_file(binary)
    missing = any(missing_by_binary.values())
    print(
        json.dumps(
            {
                **validation_summary(registry, claim),
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


def main(arguments: list[str] | None = None) -> int:
    provided = sys.argv[1:] if arguments is None else arguments
    claim, validate_only, list_only, remaining = requested_claim(provided)
    if validate_only:
        if remaining:
            raise ValueError("--validate-only does not accept execution arguments")
        registry = load_registry(DEFAULT_REGISTRY)
        print(json.dumps(validation_summary(registry, claim), sort_keys=True))
        return 0
    if list_only:
        return run_list_only(claim, remaining)
    sys.argv = [sys.argv[0], *remaining]
    return strict_runner.main()


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
        print(f"error: {error}", file=strict_runner.sys.stderr)
        raise SystemExit(2)
