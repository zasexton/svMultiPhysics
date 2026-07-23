#!/usr/bin/env python3
"""Run the frozen WP-6 conservative-phase prerequisite matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. Requests for
FSR-06, WP-6, or Q3 closure fail before build or test execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import run_free_surface_wp2_geometry_qualification as strict_runner  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp6_conservative_phase_qualification_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "f1614f846a318e486447497d0b7fed1f75145fa157810aa72ea3389fd2c494a6"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)
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
    "phase_maintenance_transaction_serial": ("application", 1, 1, 12),
    "phase_operator_partition_mpi": ("assembly_mpi", 2, 2, 3),
    "phase_artifact_collective_mpi": ("application_mpi", 2, 2, 1),
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
    "tracked_operator_fixture_ranks": 2,
    "tracked_artifact_fixture_ranks": 2,
    "four_rank_disposition": "REQUIRED_NOT_CLAIMED",
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
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = (
    "free_surface_wp6_conservative_phase_prerequisite_v1"
)
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-6"
strict_runner.__doc__ = __doc__

_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text


def load_registry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return validate_wp6_contract(_shared_load_registry(resolved), resolved)


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
            "test_fe_levelset_mpi executable in this matrix.\n"
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
