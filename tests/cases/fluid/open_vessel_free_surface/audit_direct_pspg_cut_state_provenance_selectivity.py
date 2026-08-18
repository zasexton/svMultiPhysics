#!/usr/bin/env python3
"""Audit cut-state provenance for global direct PSPG candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_GLOBAL_EMISSION = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_global_candidate_emission_20260606.json"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)


SELECTORS = (
    {
        "key": "preferred_inactive_point",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "inactive_point",
        "description": "Raw preferred candidates whose mapped source point is inactive.",
    },
    {
        "key": "preferred_dry_only_incident_support",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "dry_only_incident_support",
        "description": (
            "Raw preferred candidates whose incident source cells have no wet "
            "volume support."
        ),
    },
    {
        "key": "preferred_cut_incident_support",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "cut_incident_support",
        "description": (
            "Raw preferred candidates with at least one incident source cut cell."
        ),
    },
    {
        "key": "preferred_dry_or_cut_incident_support",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "dry_or_cut_incident_support",
        "description": (
            "Raw preferred candidates with either dry-only or cut-adjacent "
            "incident source support."
        ),
    },
    {
        "key": "sparse_or_moderate_direct_self_inactive_point",
        "candidate_key": "sparse_or_moderate_direct_self_ratio_global_dofs",
        "predicate": "inactive_point",
        "description": (
            "Sparse-or-moderate direct-self candidates whose mapped source point "
            "is inactive."
        ),
    },
    {
        "key": "sparse_or_moderate_direct_self_dry_only_incident_support",
        "candidate_key": "sparse_or_moderate_direct_self_ratio_global_dofs",
        "predicate": "dry_only_incident_support",
        "description": (
            "Sparse-or-moderate direct-self candidates whose incident source "
            "cells have no wet volume support."
        ),
    },
    {
        "key": "sparse_or_moderate_direct_self_cut_incident_support",
        "candidate_key": "sparse_or_moderate_direct_self_ratio_global_dofs",
        "predicate": "cut_incident_support",
        "description": (
            "Sparse-or-moderate direct-self candidates with at least one "
            "incident source cut cell."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map globally emitted direct PSPG formulation candidates onto replay "
            "source cut-state fields and test whether point activity, phi sign, "
            "or incident wet-volume-fraction provenance is selective enough to "
            "promote."
        )
    )
    parser.add_argument(
        "--global-emission-json",
        type=Path,
        default=DEFAULT_GLOBAL_EMISSION,
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio still considered selective.",
    )
    parser.add_argument("--dry-volume-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--full-wet-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [item for item in as_list(value) if isinstance(item, int)]


def target_case_map(target_map: dict[str, Any]) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            targets[label] = int_list(case.get("direct_pspg_target_global_dofs"))
    return targets


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def source_result_from_manifest(case: dict[str, Any]) -> str | None:
    log_path = case.get("path")
    if not isinstance(log_path, str) or not log_path:
        return None
    manifest = Path(log_path).parent / "replay_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = load_json(manifest)
    except (OSError, json.JSONDecodeError):
        return None
    source_result = data.get("source_result")
    return source_result if isinstance(source_result, str) and source_result else None


def point_incident_cells(grid: Any) -> list[list[int]]:
    point_count = int(getattr(grid, "n_points", 0) or 0)
    incident = [[] for _ in range(point_count)]
    cells = getattr(grid, "cells", [])
    offset = 0
    cell_id = 0
    while offset < len(cells):
        point_count_in_cell = int(cells[offset])
        point_ids = cells[offset + 1 : offset + 1 + point_count_in_cell]
        for point_id in point_ids:
            index = int(point_id)
            if 0 <= index < point_count:
                incident[index].append(cell_id)
        offset += point_count_in_cell + 1
        cell_id += 1
    return incident


def wet_support_class(
    wet_fractions: list[float],
    *,
    dry_volume_tolerance: float,
    full_wet_tolerance: float,
) -> str:
    if not wet_fractions:
        return "missing_incident_support"
    dry = [
        value <= dry_volume_tolerance
        for value in wet_fractions
    ]
    full_wet = [
        value >= 1.0 - full_wet_tolerance
        for value in wet_fractions
    ]
    cut = [
        dry_volume_tolerance < value < 1.0 - full_wet_tolerance
        for value in wet_fractions
    ]
    if all(dry):
        return "dry_only_incident_support"
    if all(full_wet):
        return "full_wet_incident_support"
    if all(cut):
        return "cut_only_incident_support"
    if any(cut) and any(dry) and not any(full_wet):
        return "mixed_cut_dry_incident_support"
    if any(cut) and any(full_wet) and not any(dry):
        return "mixed_cut_wet_incident_support"
    return "mixed_wet_dry_or_cut_incident_support"


def phi_class(value: float | None) -> str:
    if value is None:
        return "missing_phi"
    if value < 0.0:
        return "negative_phi"
    if value > 0.0:
        return "positive_phi"
    return "zero_phi"


def active_class(value: float | None) -> str:
    if value is None:
        return "missing_active_fluid"
    return "active_point" if value > 0.5 else "inactive_point"


def required_dofs(case: dict[str, Any], target_rows: list[int]) -> list[int]:
    dofs = set(target_rows)
    for selector in SELECTORS:
        dofs.update(int_list(case.get(selector["candidate_key"])))
    return sorted(dofs)


def load_cut_state_profile(
    *,
    case: dict[str, Any],
    target_rows: list[int],
    dry_volume_tolerance: float,
    full_wet_tolerance: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    source_result = source_result_from_manifest(case)
    evidence: dict[str, Any] = {
        "source_result": source_result,
        "source_result_exists": bool(source_result and Path(source_result).exists()),
        "pressure_offset": case.get("pressure_offset"),
    }
    if not source_result or not Path(source_result).exists():
        evidence["profile_status"] = "source_result_missing"
        return {}, evidence
    pressure_offset = case.get("pressure_offset")
    if not isinstance(pressure_offset, int):
        evidence["profile_status"] = "pressure_offset_missing"
        return {}, evidence
    try:
        import pyvista as pv

        grid = pv.read(source_result)
    except Exception as exc:
        evidence["profile_status"] = "mesh_load_failed"
        evidence["mesh_error"] = str(exc)
        return {}, evidence

    point_arrays = set(grid.point_data.keys())
    cell_arrays = set(grid.cell_data.keys())
    required_point_arrays = {"ActiveFluid", "phi"}
    required_cell_arrays = {"WetVolumeFraction"}
    missing_point_arrays = sorted(required_point_arrays - point_arrays)
    missing_cell_arrays = sorted(required_cell_arrays - cell_arrays)
    evidence.update(
        {
            "point_arrays": sorted(point_arrays),
            "cell_arrays": sorted(cell_arrays),
            "missing_point_arrays": missing_point_arrays,
            "missing_cell_arrays": missing_cell_arrays,
        }
    )
    if missing_point_arrays or missing_cell_arrays:
        evidence["profile_status"] = "cut_state_arrays_missing"
        return {}, evidence

    active_fluid = grid.point_data["ActiveFluid"]
    phi = grid.point_data["phi"]
    wet_volume_fraction = grid.cell_data["WetVolumeFraction"]
    incident_cells = point_incident_cells(grid)
    profile: dict[int, dict[str, Any]] = {}
    out_of_range = 0
    for dof in required_dofs(case, target_rows):
        point_id = dof - pressure_offset
        if point_id < 0 or point_id >= int(grid.n_points):
            out_of_range += 1
            continue
        cell_ids = incident_cells[point_id] if point_id < len(incident_cells) else []
        wet_fractions = [float(wet_volume_fraction[cell_id]) for cell_id in cell_ids]
        active_value = float(active_fluid[point_id])
        phi_value = float(phi[point_id])
        cut_incident_count = sum(
            dry_volume_tolerance < value < 1.0 - full_wet_tolerance
            for value in wet_fractions
        )
        dry_incident_count = sum(
            value <= dry_volume_tolerance for value in wet_fractions
        )
        full_wet_incident_count = sum(
            value >= 1.0 - full_wet_tolerance for value in wet_fractions
        )
        profile[dof] = {
            "point_id": int(point_id),
            "active_fluid": active_value,
            "active_class": active_class(active_value),
            "phi": phi_value,
            "phi_class": phi_class(phi_value),
            "incident_cell_count": len(cell_ids),
            "incident_wet_volume_fraction_min": (
                min(wet_fractions) if wet_fractions else None
            ),
            "incident_wet_volume_fraction_max": (
                max(wet_fractions) if wet_fractions else None
            ),
            "dry_incident_cell_count": dry_incident_count,
            "cut_incident_cell_count": cut_incident_count,
            "full_wet_incident_cell_count": full_wet_incident_count,
            "wet_support_class": wet_support_class(
                wet_fractions,
                dry_volume_tolerance=dry_volume_tolerance,
                full_wet_tolerance=full_wet_tolerance,
            ),
        }

    evidence.update(
        {
            "profile_status": "ok",
            "profiled_dof_count": len(profile),
            "out_of_range_dof_count": out_of_range,
            "mesh_point_count": int(grid.n_points),
            "mesh_cell_count": int(grid.n_cells),
        }
    )
    return profile, evidence


def profile_matches(profile: dict[str, Any], predicate: str) -> bool:
    active = profile.get("active_class") == "active_point"
    inactive = profile.get("active_class") == "inactive_point"
    wet_class = profile.get("wet_support_class")
    cut_incident_count = profile.get("cut_incident_cell_count")
    has_cut = isinstance(cut_incident_count, int) and cut_incident_count > 0
    dry_only = wet_class == "dry_only_incident_support"
    if predicate == "active_point":
        return active
    if predicate == "inactive_point":
        return inactive
    if predicate == "dry_only_incident_support":
        return dry_only
    if predicate == "cut_incident_support":
        return has_cut
    if predicate == "dry_or_cut_incident_support":
        return dry_only or has_cut
    raise ValueError(f"Unknown selector predicate: {predicate}")


def evaluate_selector_case(
    *,
    label: str,
    case: dict[str, Any],
    target_rows: list[int],
    profile_by_dof: dict[int, dict[str, Any]],
    selector: dict[str, str],
    max_target_ratio: float,
) -> dict[str, Any]:
    candidate_dofs = int_list(case.get(selector["candidate_key"]))
    selected = [
        dof
        for dof in candidate_dofs
        if dof in profile_by_dof
        and profile_matches(profile_by_dof[dof], selector["predicate"])
    ]
    selected_set = set(selected)
    covered = [dof for dof in target_rows if dof in selected_set]
    uncovered = [dof for dof in target_rows if dof not in selected_set]
    selected_ratio = ratio(len(selected), len(target_rows))
    if not profile_by_dof:
        finding = "cut_state_profile_missing"
    elif uncovered:
        finding = "selector_misses_targets"
    elif selected_ratio is not None and selected_ratio > max_target_ratio:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"

    active_classes = Counter(
        profile_by_dof[dof].get("active_class", "missing") for dof in selected
    )
    phi_classes = Counter(
        profile_by_dof[dof].get("phi_class", "missing") for dof in selected
    )
    wet_support_classes = Counter(
        profile_by_dof[dof].get("wet_support_class", "missing")
        for dof in selected
    )
    target_active_classes = Counter(
        profile_by_dof[dof].get("active_class", "missing")
        for dof in target_rows
        if dof in profile_by_dof
    )
    target_wet_support_classes = Counter(
        profile_by_dof[dof].get("wet_support_class", "missing")
        for dof in target_rows
        if dof in profile_by_dof
    )
    return {
        "label": label,
        "finding": finding,
        "candidate_count": len(candidate_dofs),
        "selected_count": len(selected),
        "direct_target_count": len(target_rows),
        "selected_to_target_ratio": selected_ratio,
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "selected_active_class_counts": dict(active_classes),
        "selected_phi_class_counts": dict(phi_classes),
        "selected_wet_support_class_counts": dict(wet_support_classes),
        "target_active_class_counts": dict(target_active_classes),
        "target_wet_support_class_counts": dict(target_wet_support_classes),
    }


def selector_finding(cases: list[dict[str, Any]]) -> str:
    findings = [case.get("finding") for case in cases]
    if any(finding == "cut_state_profile_missing" for finding in findings):
        return "selector_cut_state_profile_missing"
    if any(finding == "selector_misses_targets" for finding in findings):
        return "selector_misses_targets"
    if any(finding == "selector_overbroad" for finding in findings):
        return "selector_overbroad"
    if cases and all(finding == "selector_selective" for finding in findings):
        return "selector_selective"
    return "selector_inconclusive"


def build_report(
    *,
    global_emission: dict[str, Any],
    target_map: dict[str, Any],
    global_emission_path: Path | None = None,
    target_map_path: Path | None = None,
    max_target_ratio: float = 5.0,
    dry_volume_tolerance: float = 1.0e-12,
    full_wet_tolerance: float = 1.0e-10,
    profiles_by_label: dict[str, dict[int, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    targets = target_case_map(target_map)
    emission_cases = [
        case
        for case in as_list(global_emission.get("cases"))
        if isinstance(case, dict) and isinstance(case.get("label"), str)
    ]
    profile_evidence: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[int, dict[str, Any]]] = {}
    for case in emission_cases:
        label = str(case["label"])
        if profiles_by_label is not None and label in profiles_by_label:
            profiles[label] = profiles_by_label[label]
            profile_evidence[label] = {
                "profile_status": "provided",
                "profiled_dof_count": len(profiles_by_label[label]),
            }
            continue
        profile, evidence = load_cut_state_profile(
            case=case,
            target_rows=targets.get(label, []),
            dry_volume_tolerance=dry_volume_tolerance,
            full_wet_tolerance=full_wet_tolerance,
        )
        profiles[label] = profile
        profile_evidence[label] = evidence

    selector_reports = []
    for selector in SELECTORS:
        case_reports = [
            evaluate_selector_case(
                label=str(case["label"]),
                case=case,
                target_rows=targets.get(str(case["label"]), []),
                profile_by_dof=profiles.get(str(case["label"]), {}),
                selector=selector,
                max_target_ratio=max_target_ratio,
            )
            for case in emission_cases
        ]
        selector_reports.append(
            {
                "key": selector["key"],
                "candidate_key": selector["candidate_key"],
                "predicate": selector["predicate"],
                "description": selector["description"],
                "finding": selector_finding(case_reports),
                "cases": case_reports,
            }
        )

    selector_findings = [report["finding"] for report in selector_reports]
    if any(finding == "selector_selective" for finding in selector_findings):
        finding = "cut_state_provenance_selector_selective_for_formulation_replay"
        next_requirement = (
            "Prototype the selective cut-state formulation gate and replay the "
            "short Test02/Test10 windows."
        )
    elif any("cut_state_profile_missing" in finding for finding in selector_findings):
        finding = "cut_state_provenance_selector_profile_missing"
        next_requirement = (
            "Regenerate the replay candidate emission with source cut-state "
            "fields before using cut-state selectivity evidence."
        )
    elif all(finding == "selector_misses_targets" for finding in selector_findings):
        finding = "cut_state_provenance_selectors_miss_audited_targets"
        next_requirement = (
            "Do not use simple point activity, phi sign, or incident "
            "wet-volume-fraction cut adjacency as the formulation gate."
        )
    elif any(finding == "selector_overbroad" for finding in selector_findings):
        finding = "cut_state_provenance_selectors_overbroad_or_miss_targets"
        next_requirement = (
            "Do not promote simple cut-state provenance without an additional "
            "active direct PSPG pressure-gradient support/coupled-patch topology "
            "gate."
        )
    else:
        finding = "cut_state_provenance_selectivity_inconclusive"
        next_requirement = (
            "Regenerate cut-state provenance evidence before choosing a "
            "formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for point activity, phi sign, and incident "
            "WetVolumeFraction provenance on globally emitted direct PSPG "
            "formulation candidates."
        ),
        "global_emission_path": (
            str(global_emission_path) if global_emission_path is not None else None
        ),
        "target_map_path": str(target_map_path) if target_map_path is not None else None,
        "max_target_ratio": max_target_ratio,
        "dry_volume_tolerance": dry_volume_tolerance,
        "full_wet_tolerance": full_wet_tolerance,
        "finding": finding,
        "case_count": len(emission_cases),
        "selector_count": len(selector_reports),
        "profile_evidence": profile_evidence,
        "selectors": selector_reports,
        "next_requirement": next_requirement,
    }


def main() -> None:
    args = parse_args()
    global_emission = load_json(args.global_emission_json)
    target_map = load_json(args.target_map_json)
    report = build_report(
        global_emission=global_emission,
        target_map=target_map,
        global_emission_path=args.global_emission_json,
        target_map_path=args.target_map_json,
        max_target_ratio=args.max_target_ratio,
        dry_volume_tolerance=args.dry_volume_tolerance,
        full_wet_tolerance=args.full_wet_tolerance,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
