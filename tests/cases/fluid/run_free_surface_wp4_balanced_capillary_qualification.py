#!/usr/bin/env python3
"""Run the frozen WP-4 balanced-capillary prerequisite matrix.

Only ``--requested-claim low_level_prerequisite`` is accepted. Requests for
FSR-03, FSR-04, WP-4, or Q2 closure fail before build or test execution.
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
    "free_surface_wp4_balanced_capillary_prerequisite_matrix.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "32abde84e3c087ca80e527bd68c576b5c91c31c7465cbc01791cc1e6e79f565d"
)
SHARED_RUNNER_PATH = Path(strict_runner.__file__).resolve()
SHARED_RUNNER_SHA256 = strict_runner.sha256_file(SHARED_RUNNER_PATH)
EXPECTED_SCOPE = (
    "Low-level WP-4 prerequisite evidence only; this matrix does not close "
    "FSR-03, FSR-04, WP-4, or Q2."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "low_level_prerequisite",
    "rejected_claims": [
        "fsr03_closure",
        "fsr04_closure",
        "wp4_closure",
        "q2_closure",
    ],
    "diagnostic": (
        "The frozen safety slice does not select or qualify an AD-2 "
        "balanced-force method, establish a complete surface-wall-volume "
        "directional derivative, or execute the required static-cap "
        "refinement campaign."
    ),
}
EXPECTED_METHOD_BOUNDARY = {
    "selected_ad2_method": "UNSELECTED",
    "current_pressure_action": (
        "one_shot_additive_initial_guess_in_the_existing_pressure_space"
    ),
    "production_capillary_operator_changed": False,
    "force_projection_applied": False,
    "balanced_force_evidence_claimed": False,
    "prescribed_angle_momentum_owner": "young_wall_energy",
    "prescribed_angle_geometry_owner": (
        "wall_aware_level_set_constraint_and_reinitialization"
    ),
}
EXPECTED_DISPOSITION = {
    "fsr03_closed": False,
    "fsr04_closed": False,
    "wp4_closed": False,
    "q2_closed": False,
}
EXPECTED_GROUPS = {
    "static_pressure_gate_serial": ("systems", 1, 1, 13),
    "surface_wall_balance_serial": ("physics", 1, 1, 8),
    "wall_angle_geometry_serial": ("level_set", 1, 1, 2),
    "static_pressure_configuration_mpi": ("assembly_mpi", 2, 2, 1),
}
EXPECTED_UNQUALIFIED_CAMPAIGNS = {
    "select_and_derive_one_ad2_balanced_capillary_pressure_method",
    (
        "surface_wall_volume_functional_directional_derivatives_on_fixed_"
        "topology"
    ),
    "general_fe_pressure_domain_variation_identity",
    "discrete_static_cap_constrained_energy_minimization",
    "flat_interface_direction_phase_gravity_gauge_cut_and_mpi_matrix",
    "circle_sphere_and_30_60_90_120_150_degree_sessile_refinement",
    "all_supported_wall_rotations_and_cut_offsets",
    "independent_mesh_time_and_reinitialization_cadence_refinement",
    "gross_reusken_force_and_reusken_sessile_adaptations",
    (
        "pressure_jump_angle_shape_parasitic_current_and_energy_release_"
        "gates"
    ),
}


def validate_wp4_contract(
    registry: dict[str, Any], registry_path: Path
) -> dict[str, Any]:
    if strict_runner.sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-4 frozen registry bytes changed")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-4 qualification scope changed after freeze")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("WP-4 closure-request policy changed after freeze")
    if registry.get("method_boundary") != EXPECTED_METHOD_BOUNDARY:
        raise ValueError("WP-4 method boundary changed after freeze")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("WP-4 qualification disposition changed")
    if registry.get("prospective_tests") != []:
        raise ValueError("WP-4 cannot freeze with prospective tests")

    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("WP-4 execution groups are missing")
    actual_groups: dict[str, tuple[Any, Any, Any, int]] = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(
            group.get("tests"), list
        ):
            raise ValueError("WP-4 execution group is invalid")
        group_id = group.get("id")
        if not isinstance(group_id, str) or group_id in actual_groups:
            raise ValueError("WP-4 execution group id is invalid")
        actual_groups[group_id] = (
            group.get("binary"),
            group.get("mpi_ranks"),
            group.get("gtest_output_copies"),
            len(group["tests"]),
        )
    if actual_groups != EXPECTED_GROUPS:
        raise ValueError("WP-4 execution groups changed after freeze")

    unresolved = registry.get("unqualified_required_campaigns")
    if not isinstance(unresolved, list):
        raise ValueError("WP-4 unqualified campaign list is missing")
    unresolved_ids: set[str] = set()
    for entry in unresolved:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or entry.get("status") != "REQUIRED_NOT_CLAIMED"
            or not isinstance(entry.get("id"), str)
        ):
            raise ValueError("WP-4 unqualified campaign entry is invalid")
        unresolved_ids.add(entry["id"])
    if (
        unresolved_ids != EXPECTED_UNQUALIFIED_CAMPAIGNS
        or len(unresolved_ids) != len(unresolved)
    ):
        raise ValueError("WP-4 unqualified campaign list changed after freeze")
    return registry


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = (
    "free_surface_wp4_balanced_capillary_prerequisite_v1"
)
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-4"
strict_runner.__doc__ = __doc__

_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text


def load_registry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return validate_wp4_contract(_shared_load_registry(resolved), resolved)


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
        value["qualification_disposition"] = EXPECTED_DISPOSITION
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-4 balanced-capillary prerequisite qualification record",
            1,
        )
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "No balanced-force method or static-cap campaign is credited "
            "by this record.\n"
        )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    if "-h" in arguments or "--help" in arguments:
        print(
            "WP-4 wrapper options:\n"
            "  --requested-claim low_level_prerequisite\n"
            "      Select the only claim this safety matrix may establish.\n"
            "      FSR-03, FSR-04, WP-4, and Q2 closure are rejected.\n"
            "  --validate-only\n"
            "      Validate the frozen schema and claim boundary without "
            "builds.\n"
            "\n"
            "For execution, --systems-binary names test_fe_timestepping "
            "and --assembly-mpi-binary names test_fe_assembly_mpi.\n"
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
            f"unsupported WP-4 requested claim {claim!r}; expected {allowed!r}"
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
