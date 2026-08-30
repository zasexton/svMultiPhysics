#!/usr/bin/env python3
"""Run the frozen WP-6 conservative-phase prerequisite matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. Requests for
FSR-06, WP-6, or Q3 closure fail before build or test execution.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_SHARED_RUNNER_IMPORT_PATH = (
    SCRIPT_DIRECTORY / "run_free_surface_wp2_geometry_qualification.py"
)
_SHARED_RUNNER_MODULE_NAME = (
    "_free_surface_wp6_private_geometry_qualification_runner"
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


SHARED_QUALIFICATION_BINARY_KEYS = frozenset(
    strict_runner.QUALIFICATION_BINARY_KEYS
)
WP6_QUALIFICATION_BINARY_KEYS = {
    *SHARED_QUALIFICATION_BINARY_KEYS,
    "timestepping",
}

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
APPLICATION_DRIVER_SOURCE = (
    REPOSITORY_ROOT
    / "Code"
    / "Source"
    / "solver"
    / "Application"
    / "Core"
    / "ApplicationDriver.cpp"
)
TIME_LOOP_SOURCE = (
    REPOSITORY_ROOT
    / "Code"
    / "Source"
    / "solver"
    / "FE"
    / "TimeStepping"
    / "TimeLoop.cpp"
)
CONSERVATIVE_PHASE_OPERATOR_SOURCE = (
    REPOSITORY_ROOT
    / "Code"
    / "Source"
    / "solver"
    / "FE"
    / "LevelSet"
    / "LevelSetConservativePhaseOperator.cpp"
)
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp6_conservative_phase_qualification_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "59611de03a84336999bc43fffa96134196a4d261824276beed00ad58dc1230f2"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)
WP6_BINARY_LINK_PROVENANCE_MEMORY_MIB = 1024
WP6_BINARY_LINK_PROVENANCE_POLICY = {
    "address_space_limit_mib": WP6_BINARY_LINK_PROVENANCE_MEMORY_MIB,
    "aggregate_resident_monitoring": True,
    "scope": "linked-library discovery subprocess session",
}
EXPECTED_SCOPE = (
    "Low-level WP-6 prerequisite evidence only; this matrix does not close "
    "FSR-06, WP-6, or Q3."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr06_closure",
        "wp6_closure",
        "q3_closure",
    ],
    "diagnostic": (
        "The frozen low-level slice cannot establish the complete release "
        "refinement campaign, extension-field sweep, production maintenance "
        "comparison, momentum consistency, or smooth-dynamics closure."
    ),
}
EXPECTED_METHOD_CONTRACT = {
    "conserved_unknown": (
        "nodal liquid indicator q with zero-to-one invariant bounds"
    ),
    "declared_phase_measure": (
        "sum of lumped nodal control volume times q"
    ),
    "spatial_discretization": (
        "continuous P1 graph with symmetric low-order graph viscosity and "
        "pairwise algebraic-edge antidiffusive flux correction"
    ),
    "interior_flux_contract": (
        "each canonical algebraic edge is stored once and its accepted pair "
        "transfer cancels exactly"
    ),
    "physical_boundary_contract": (
        "physical boundary transfer is retained separately in every nodal "
        "and component balance"
    ),
    "divergence_contract": (
        "the explicit discrete q times divergence source is retained and "
        "must vanish for a divergence-compatible advecting field"
    ),
    "component_contract": (
        "deterministic connected supports and subthreshold support retain "
        "raw, limited, boundary, divergence, and balance entries"
    ),
    "geometry_contract": (
        "the level set is provisional geometry while q is the phase-measure "
        "authority for local geometry reconciliation"
    ),
}
EXPECTED_GROUPS = {
    "phase_flux_algebra_serial": ("level_set", 1, 1, 25),
    "phase_transport_modest_grid_serial": ("level_set", 1, 1, 9),
    "emergency_global_shift_accounting_serial": ("level_set", 1, 1, 1),
    "phase_maintenance_transaction_serial": ("application", 1, 1, 13),
    "phase_timeloop_publication_fail_stop_serial": (
        "timestepping",
        1,
        1,
        2,
    ),
    "phase_operator_partition_mpi_2": ("assembly_mpi", 2, 2, 4),
    "phase_application_collectives_mpi_2": (
        "application_mpi",
        2,
        2,
        4,
    ),
    "phase_maintenance_consensus_mpi_4": (
        "application_mpi",
        4,
        4,
        1,
    ),
}
EXPECTED_UNQUALIFIED_CAMPAIGNS = {
    "complete_18_point_translation_and_enright_release_matrix",
    (
        "immutable_raw_control_volume_component_history_and_final_flux_ledgers_"
        "for_every_release_point"
    ),
    "extension_band_width_and_map_fallback_sweeps",
    (
        "maintenance_off_reinitialization_only_correction_only_and_production_"
        "transaction_comparison"
    ),
    (
        "whole_time_loop_and_multi_artifact_cross_resource_atomicity_with_"
        "commit_rollback_and_logging_fault_injection"
    ),
    "four_or_more_rank_partition_sweeps",
    (
        "refined_wall_drop_rotation_film_sheet_rim_and_satellite_component_"
        "studies"
    ),
    "raw_global_and_per_component_convergence_without_global_shift",
    "phase_flux_and_momentum_flux_consistency",
    (
        "capillary_jet_and_filament_necking_after_raw_conservation_converges"
    ),
    (
        "complete_q3_capillary_wave_oscillating_drop_and_smooth_sloshing_"
        "campaigns"
    ),
}
EXPECTED_RELEASE_DEPENDENCY = {
    "matrix": "level_set_phase_transport_release_matrix.json",
    "point_runner": "run_level_set_phase_transport_release.py",
    "complete_orchestrator": (
        "run_level_set_phase_transport_release_matrix.py"
    ),
    "expected_points": 18,
    "low_level_matrix_can_substitute_for_release_matrix": False,
}
EXPECTED_PARTITION_LIMIT = {
    "tracked_operator_fixture_ranks": [2],
    "tracked_artifact_fixture_ranks": [2],
    "tracked_transaction_consensus_fixture_ranks": [2, 4],
    "four_rank_disposition": "LOW_LEVEL_TRANSACTION_CONSENSUS_ONLY",
    "four_or_more_rank_partition_sweeps": "REQUIRED_NOT_CLAIMED",
}


def _require_tokens_in_order(
    source: str, tokens: tuple[str, ...], contract: str
) -> None:
    cursor = 0
    for token in tokens:
        position = source.find(token, cursor)
        if position < 0:
            raise ValueError(
                f"WP-6 production source contract failed for {contract}: "
                f"missing or out-of-order token {token!r}"
            )
        cursor = position + len(token)


def _callback_body(
    source: str, callback: str, next_marker: str
) -> str:
    start_token = f"callbacks.{callback} ="
    start = source.find(start_token)
    end = source.find(next_marker, start + len(start_token))
    if start < 0 or end < 0:
        raise ValueError(
            f"WP-6 production source contract cannot locate {callback}"
        )
    region = source[start:end]
    body_start = region.find("{")
    if body_start < 0:
        raise ValueError(
            f"WP-6 production source contract cannot parse {callback}"
        )
    return region[body_start + 1 :]


def validate_wp6_production_source_contract(
    source_path: Path = APPLICATION_DRIVER_SOURCE,
    time_loop_source_path: Path = TIME_LOOP_SOURCE,
    conservative_phase_operator_source_path: Path = (
        CONSERVATIVE_PHASE_OPERATOR_SOURCE
    ),
) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    time_loop_source = time_loop_source_path.read_text(encoding="utf-8")
    conservative_phase_operator_source = (
        conservative_phase_operator_source_path.read_text(encoding="utf-8")
    )

    current_graph_start = source.find(
        "svmp::FE::level_set::LevelSetP1PhaseTransportGraph&\n"
        "requireCurrentConservativePhaseGraph("
    )
    current_graph_end = source.find(
        "svmp::FE::level_set::LevelSetP1PhaseProjectionOptions\n"
        "conservativePhaseProjectionOptions(",
        current_graph_start,
    )
    if current_graph_start < 0 or current_graph_end < 0:
        raise ValueError(
            "WP-6 production source contract cannot isolate conservative "
            "graph currentness handling"
        )
    current_graph_helper = source[current_graph_start:current_graph_end]
    _require_tokens_in_order(
        current_graph_helper,
        (
            "const bool local_graph_is_current = graph_is_current();",
            "const auto comm = activeFESystemCommunicator(system);",
            "const bool graph_rebuild_required =",
            "globalAnyBool(!local_graph_is_current, comm);",
            "if (graph_rebuild_required) {",
            "buildLevelSetP1PhaseTransportGraph(",
        ),
        "collective conservative graph staleness gate",
    )
    if (
        current_graph_helper.count(
            "globalAnyBool(!local_graph_is_current, comm)"
        )
        != 1
        or "if (!graph_is_current())" in current_graph_helper
    ):
        raise ValueError(
            "WP-6 production source contract requires exactly one "
            "collective conservative graph staleness gate before rebuild"
        )

    graph_metadata_start = source.find(
        "void appendMaintenanceScheduleGraphMetadata("
    )
    graph_metadata_end = source.find(
        "std::uint64_t levelSetMaintenanceRequestActionBits(",
        graph_metadata_start,
    )
    if graph_metadata_start < 0 or graph_metadata_end < 0:
        raise ValueError(
            "WP-6 production source contract cannot isolate conservative "
            "graph request metadata"
        )
    graph_metadata = source[graph_metadata_start:graph_metadata_end]
    partition_local_graph_stamps = (
        "graph->geometry_revision",
        "graph->topology_revision",
        "graph->ownership_revision",
        "graph->numbering_revision",
    )
    if any(
        stamp in graph_metadata for stamp in partition_local_graph_stamps
    ):
        raise ValueError(
            "WP-6 production source contract forbids partition-local graph "
            "mesh cache stamps in exact request consensus"
        )
    if "graph->dof_layout_revision" not in graph_metadata:
        raise ValueError(
            "WP-6 production source contract requires the replicated graph "
            "FE-layout identity in exact request consensus"
        )

    for revision in (
        "geometry_revision",
        "topology_revision",
        "ownership_revision",
        "numbering_revision",
    ):
        assignment = (
            f"result.{revision} = mesh."
            f"{revision.removesuffix('_revision').replace('_', '')}"
            "Revision();"
        )
        if assignment not in conservative_phase_operator_source:
            raise ValueError(
                "WP-6 production source contract requires rank-local graph "
                f"staleness assignment for {revision}"
            )
        for reduction in (
            f"collective, result.{revision}",
            f"result.{revision} = {revision}_min",
        ):
            if reduction in conservative_phase_operator_source:
                raise ValueError(
                    "WP-6 production source contract forbids collective "
                    f"normalization of rank-local graph stamp {revision}"
                )
    _require_tokens_in_order(
        conservative_phase_operator_source,
        (
            "const auto dof_revision_min = allReduceUnsigned64Min(",
            "collective, result.dof_layout_revision",
            "const auto dof_revision_max = allReduceUnsigned64Max(",
            "if (dof_revision_min != dof_revision_max)",
            "result.dof_layout_revision = dof_revision_min;",
        ),
        "replicated conservative graph FE-layout revision",
    )

    _require_tokens_in_order(
        source,
        (
            "auto level_set_maintenance = "
            "levelSetMaintenanceRequests(params);",
            "requireCollectiveLevelSetMaintenanceRequestSchedule(",
            "LevelSetMaintenanceScheduleStage::SteadyInitialization",
            "bindKinematicAreaGradientTractionMaintenance(",
            "requireCollectiveLevelSetMaintenanceRequestSchedule(",
            "LevelSetMaintenanceScheduleStage::SteadyInitialization",
        ),
        "steady initialization schedule preflights before and after binding",
    )
    steady_request = source.find(
        "auto level_set_maintenance = "
        "levelSetMaintenanceRequests(params);"
    )
    steady_gate = source.find(
        "LevelSetMaintenanceScheduleStage::SteadyInitialization",
        steady_request,
    )
    if steady_request < 0 or steady_gate < 0:
        raise ValueError(
            "WP-6 production source contract cannot anchor the steady "
            "initialization schedule preflight"
        )
    _require_tokens_in_order(
        source[
            steady_gate
            + len(
                "LevelSetMaintenanceScheduleStage::"
                "SteadyInitialization"
            ) :
        ],
        (
            "auto level_set_maintenance = "
            "levelSetMaintenanceRequests(params);",
            "requireCollectiveLevelSetMaintenanceRequestSchedule(",
            "LevelSetMaintenanceScheduleStage::TransientInitialization",
            "bindKinematicAreaGradientTractionMaintenance(",
            "requireCollectiveLevelSetMaintenanceRequestSchedule(",
            "LevelSetMaintenanceScheduleStage::TransientInitialization",
        ),
        "transient initialization schedule preflights before and after binding",
    )

    before_accept = _callback_body(
        source,
        "on_before_step_accept",
        "callbacks.on_step_candidate_discarded =",
    )
    before_gate = (
        "requireCollectiveLevelSetMaintenanceRequestSchedule("
    )
    if not before_accept.lstrip().startswith(before_gate):
        raise ValueError(
            "WP-6 production source contract requires schedule consensus as "
            "the first on_before_step_accept operation"
        )
    _require_tokens_in_order(
        before_accept,
        (
            before_gate,
            "ProspectiveAcceptedEndpoint",
        ),
        "prospective accepted-endpoint schedule preflight",
    )

    accepted = _callback_body(
        source,
        "on_step_accepted",
        "callbacks.on_step_rejected =",
    )
    if not accepted.lstrip().startswith(before_gate):
        raise ValueError(
            "WP-6 production source contract requires schedule consensus as "
            "the first on_step_accepted operation"
        )
    _require_tokens_in_order(
        accepted,
        (
            before_gate,
            "AcceptedEndpointPostStep",
        ),
        "accepted endpoint schedule preflight",
    )

    commit_ready = _callback_body(
        source,
        "on_step_commit_ready",
        "double vtk_total_time",
    )
    _require_tokens_in_order(
        commit_ready,
        (
            "const auto geometry_state =",
            "canonicalLevelSetMaintenanceGeometryState(",
            "appendCanonicalLevelSetMaintenanceGeometrySection(",
            "geometry_state.supported",
            "collectiveLevelSetMaintenanceTransactionDecision(",
            "if (decision ==",
            "LevelSetMaintenanceTransactionDecision::Reject",
            "rollback_and_reject_pending_phase(",
            "LevelSetMaintenancePublicationState::Publishing",
            "pending_phase_candidate.geometry_transaction->commit();",
            "commit_pending_phase_work();",
            '<< " outcome=committed"',
            "LevelSetMaintenancePublicationState::Published",
        ),
        "precommit consensus and publication ordering",
    )
    rollback_start = source.find(
        "const auto rollback_and_reject_pending_phase ="
    )
    rollback_end = source.find(
        "callbacks.on_before_step_accept =", rollback_start
    )
    if rollback_start < 0 or rollback_end < 0:
        raise ValueError(
            "WP-6 production source contract cannot locate precommit "
            "recovery"
        )
    _require_tokens_in_order(
        source[rollback_start:rollback_end],
        (
            "if (publication_began)",
            '<< " outcome=publication_failed"',
            '<< " rollback_claimed=false"',
            "throw std::runtime_error(",
            "std::string recovery_failure;",
            "rollbackConservativePhaseCandidate(",
            "reject_pending_phase_work();",
        ),
        "publication fail-stop precedes prepublication rollback",
    )
    fail_stop_start = source.find(
        "if (publication_began)", rollback_start, rollback_end
    )
    prepublication_recovery_start = source.find(
        "std::string recovery_failure;",
        fail_stop_start,
        rollback_end,
    )
    if fail_stop_start < 0 or prepublication_recovery_start < 0:
        raise ValueError(
            "WP-6 production source contract cannot isolate the "
            "publication fail-stop branch"
        )
    fail_stop_branch = source[
        fail_stop_start:prepublication_recovery_start
    ]
    for forbidden in (
        "rollbackConservativePhaseCandidate(",
        "reject_pending_phase_work(",
    ):
        if forbidden in fail_stop_branch:
            raise ValueError(
                "WP-6 production source contract forbids rollback or "
                "ledger rejection after publication begins"
            )

    _require_tokens_in_order(
        accepted,
        (
            "const auto geometry_state =",
            "canonicalLevelSetMaintenanceGeometryState(",
            "appendCanonicalLevelSetMaintenanceGeometrySection(",
            "geometry_state.supported",
            "const auto non_topology_decision =",
            "collectiveLevelSetMaintenanceTransactionDecision(",
            "if (non_topology_decision ==",
            "LevelSetMaintenanceTransactionDecision::Reject",
            "rollback_and_reject_maintenance(",
            "if (communicator_complete_topology_mismatch)",
            "MaintenanceCutTopologyChanged",
            "rollback_and_reject_maintenance(",
            "maintenance_publication_started = true;",
            "maintenance_geometry_transaction->commit();",
            "level_set_maintenance_work.commitTransaction();",
            "maintenance_transaction_published = true;",
            "if (maintenance_transaction_published &&",
            '<< " outcome=committed"',
        ),
        "postaccept consensus and publication ordering",
    )

    if source.count(
        "appendCanonicalLevelSetMaintenanceGeometrySection(\n"
        "              commit_state_words, geometry_state);"
    ) != 1 or source.count(
        "appendCanonicalLevelSetMaintenanceGeometrySection(\n"
        "            commit_state_words, geometry_state);"
    ) != 1:
        raise ValueError(
            "WP-6 production source contract requires one live-geometry "
            "section at each publication consensus site"
        )

    commit_failure_start = time_loop_source.find(
        "const auto commit_failure = std::current_exception();"
    )
    successful_recovery_end = time_loop_source.find(
        "std::rethrow_exception(commit_failure);",
        commit_failure_start,
    )
    if commit_failure_start < 0 or successful_recovery_end < 0:
        raise ValueError(
            "WP-6 production source contract cannot isolate the TimeLoop "
            "commit-ready recovery branch"
        )
    commit_failure_branch = time_loop_source[
        commit_failure_start:successful_recovery_end
        + len("std::rethrow_exception(commit_failure);")
    ]
    _require_tokens_in_order(
        commit_failure_branch,
        (
            "const auto commit_failure = std::current_exception();",
            "candidate_rollback_guard.discard();",
            "catch (...)",
            "attempt_state.commit();",
            "throw;",
            "restoreAcceptedGeneratedState();",
            "std::rethrow_exception(commit_failure);",
        ),
        "TimeLoop publication fail-stop disarms attempt rollback",
    )

    attempt_guard_start = time_loop_source.find(
        "class AttemptStateGuard {"
    )
    attempt_guard_end = time_loop_source.find(
        "class DiscardedStaticPressureInitializationGuard {",
        attempt_guard_start,
    )
    if attempt_guard_start < 0 or attempt_guard_end < 0:
        raise ValueError(
            "WP-6 production source contract cannot isolate the TimeLoop "
            "attempt-state guard"
        )
    attempt_guard = time_loop_source[
        attempt_guard_start:attempt_guard_end
    ]
    guarded_restore_block = (
        "        if (!committed_) {\n"
        "            history_.restoreRateState(snapshot_);\n"
        "            workspace_.static_compatible_pressure_initialized =\n"
        "                static_pressure_initialized_;\n"
        "        }"
    )
    commit_disarm = (
        "    void commit() noexcept { committed_ = true; }"
    )
    if (
        attempt_guard.count(guarded_restore_block) != 1
        or attempt_guard.count(commit_disarm) != 1
        or attempt_guard.find(guarded_restore_block)
        >= attempt_guard.find(commit_disarm)
    ):
        raise ValueError(
            "WP-6 production source contract failed for TimeLoop attempt "
            "guard rate and workspace state domains: both restores must "
            "remain inside the !committed_ guard and commit must disarm it"
        )

    return {
        "collective_graph_staleness_gates": 1,
        "partition_local_graph_cache_stamps_excluded": 4,
        "schedule_initialization_gates": 4,
        "schedule_first_callback_gates": 2,
        "live_geometry_consensus_sites": 2,
        "publication_ordering_sites": 2,
        "timeloop_fail_stop_disarm_sites": 1,
        "timeloop_attempt_guard_state_domains": 2,
    }


def validate_wp6_contract(
    registry: dict[str, Any], registry_path: Path
) -> dict[str, Any]:
    if strict_runner.sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-6 frozen registry bytes changed")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-6 qualification scope changed after freeze")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("WP-6 closure-request policy changed after freeze")
    if registry.get("method_contract") != EXPECTED_METHOD_CONTRACT:
        raise ValueError("WP-6 conserved-phase method contract changed")
    if registry.get("prospective_tests") != []:
        raise ValueError("WP-6 cannot freeze with prospective tests")

    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("WP-6 execution groups are missing")
    actual_groups: dict[str, tuple[Any, Any, Any, int]] = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(
            group.get("tests"), list
        ):
            raise ValueError("WP-6 execution group is invalid")
        group_id = group.get("id")
        if not isinstance(group_id, str) or group_id in actual_groups:
            raise ValueError("WP-6 execution group id is invalid")
        actual_groups[group_id] = (
            group.get("binary"),
            group.get("mpi_ranks"),
            group.get("gtest_output_copies"),
            len(group["tests"]),
        )
    if actual_groups != EXPECTED_GROUPS:
        raise ValueError("WP-6 execution groups changed after freeze")

    unresolved = registry.get("unqualified_required_campaigns")
    if not isinstance(unresolved, list):
        raise ValueError("WP-6 unqualified campaign list is missing")
    unresolved_ids: set[str] = set()
    for entry in unresolved:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or entry.get("status") != "REQUIRED_NOT_CLAIMED"
            or not isinstance(entry.get("id"), str)
        ):
            raise ValueError("WP-6 unqualified campaign entry is invalid")
        unresolved_ids.add(entry["id"])
    if (
        unresolved_ids != EXPECTED_UNQUALIFIED_CAMPAIGNS
        or len(unresolved_ids) != len(unresolved)
    ):
        raise ValueError("WP-6 unqualified campaign list changed after freeze")
    if registry.get("release_matrix_dependency") != (
        EXPECTED_RELEASE_DEPENDENCY
    ):
        raise ValueError("WP-6 release-matrix dependency changed after freeze")
    if registry.get("known_partition_limit") != EXPECTED_PARTITION_LIMIT:
        raise ValueError("WP-6 distributed qualification boundary changed")
    validate_wp6_production_source_contract()
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = (
    "free_surface_wp6_conservative_phase_prerequisite_v2"
)
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-6"
strict_runner.BINARY_LINK_PROVENANCE_MEMORY_MIB = (
    WP6_BINARY_LINK_PROVENANCE_MEMORY_MIB
)
strict_runner.__doc__ = __doc__

_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text


@contextmanager
def wp6_binary_key_scope():
    previous = strict_runner.QUALIFICATION_BINARY_KEYS
    strict_runner.QUALIFICATION_BINARY_KEYS = set(
        WP6_QUALIFICATION_BINARY_KEYS
    )
    try:
        yield
    finally:
        strict_runner.QUALIFICATION_BINARY_KEYS = previous


def load_registry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    with wp6_binary_key_scope():
        registry = _shared_load_registry(resolved)
    return validate_wp6_contract(registry, resolved)


def write_json(path: Path, value: Any) -> None:
    if strict_runner.sha256_file(SHARED_RUNNER_PATH) != SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification runner changed during execution")
    if isinstance(value, dict) and path.name == "build.json":
        value = copy.deepcopy(value)
        value["linked_library_provenance_policy"] = copy.deepcopy(
            WP6_BINARY_LINK_PROVENANCE_POLICY
        )
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
        value["requested_claim"] = "low_level_prerequisite"
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-6 conservative-phase prerequisite qualification record",
            1,
        )
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "The complete 18-point release campaign remains a separate "
            "required artifact.\n"
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-6 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this low-level matrix may establish.\n"
            "      fsr06_closure, wp6_closure, and q3_closure are rejected.\n"
            "  --validate-only\n"
            "      Validate the frozen schema and claim boundary without "
            "builds.\n"
            "\n"
            "For execution, --assembly-mpi-binary names the "
            "test_fe_levelset_mpi executable and --timestepping-binary "
            "names the test_fe_timestepping executable in this matrix.\n"
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
            f"unsupported WP-6 requested claim {claim!r}; expected {allowed!r}"
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
                        "serial_quantitative_gate_count": len(
                            _registry["quantitative_evidence"]
                        ),
                        "unqualified_campaign_count": len(
                            _registry["unqualified_required_campaigns"]
                        ),
                        "release_matrix_expected_points": (
                            _registry["release_matrix_dependency"][
                                "expected_points"
                            ]
                        ),
                        "production_source_contract": (
                            validate_wp6_production_source_contract()
                        ),
                        "outcome": "PASS",
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(0)
        sys.argv = [sys.argv[0], *_remaining_arguments]
        with wp6_binary_key_scope():
            _exit_code = strict_runner.main()
        raise SystemExit(_exit_code)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=strict_runner.sys.stderr)
        raise SystemExit(2)
