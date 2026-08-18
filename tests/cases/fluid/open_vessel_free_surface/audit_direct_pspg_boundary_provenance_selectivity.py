#!/usr/bin/env python3
"""Audit mesh boundary provenance for global direct PSPG candidates."""

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
        "key": "preferred_boundary_only",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "boundary",
        "description": "Raw preferred global candidates restricted to mesh-boundary pressure DOFs.",
    },
    {
        "key": "preferred_boundary_or_low_incident",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "boundary_or_low_incident",
        "description": "Raw preferred global candidates on a mesh boundary or with at most two incident cells.",
    },
    {
        "key": "preferred_one_cell_boundary",
        "candidate_key": "preferred_candidate_global_dofs",
        "predicate": "one_cell_boundary",
        "description": "Raw preferred global candidates with literal one-cell boundary support.",
    },
    {
        "key": "sparse_direct_self_boundary",
        "candidate_key": "sparse_direct_self_global_dofs",
        "predicate": "boundary",
        "description": "Sparse direct-self candidates restricted to mesh-boundary pressure DOFs.",
    },
    {
        "key": "sparse_or_moderate_direct_self_boundary",
        "candidate_key": "sparse_or_moderate_direct_self_ratio_global_dofs",
        "predicate": "boundary",
        "description": "Sparse-or-moderate direct-self support-ratio candidates restricted to mesh-boundary pressure DOFs.",
    },
    {
        "key": "sparse_or_moderate_direct_self_low_incident",
        "candidate_key": "sparse_or_moderate_direct_self_ratio_global_dofs",
        "predicate": "low_incident",
        "description": "Sparse-or-moderate direct-self support-ratio candidates with at most two incident cells.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map globally emitted direct PSPG formulation candidates onto the "
            "replay source mesh and test whether literal mesh boundary or "
            "incident-support provenance is selective enough to promote."
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
    parser.add_argument("--boundary-tolerance", type=float, default=1.0e-10)
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


def point_incident_cell_counts(grid: Any) -> list[int]:
    point_count = int(getattr(grid, "n_points", 0) or 0)
    counts = [0 for _ in range(point_count)]
    cells = getattr(grid, "cells", [])
    offset = 0
    while offset < len(cells):
        node_count = int(cells[offset])
        point_ids = cells[offset + 1 : offset + 1 + node_count]
        for point_id in point_ids:
            index = int(point_id)
            if 0 <= index < point_count:
                counts[index] += 1
        offset += node_count + 1
    return counts


def boundary_labels(
    point: Any,
    bounds: tuple[float, float, float, float, float, float],
    *,
    tolerance: float,
) -> list[str]:
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    candidates = (
        ("x_min", x, bounds[0]),
        ("x_max", x, bounds[1]),
        ("y_min", y, bounds[2]),
        ("y_max", y, bounds[3]),
        ("z_min", z, bounds[4]),
        ("z_max", z, bounds[5]),
    )
    return [
        label
        for label, value, boundary in candidates
        if abs(value - boundary) <= tolerance
    ]


def boundary_class(labels: list[str]) -> str:
    if not labels:
        return "interior"
    if len(labels) == 1:
        return "boundary_face"
    if len(labels) == 2:
        return "boundary_edge"
    return "boundary_corner"


def incident_support_class(
    *,
    boundary_class_value: str,
    incident_cell_count: int | None,
) -> str:
    if incident_cell_count is None:
        return "missing_incident_support"
    if incident_cell_count <= 0:
        return "zero_incident_support"
    if boundary_class_value == "interior":
        if incident_cell_count == 1:
            return "interior_one_cell_support"
        return "interior_shared_support"
    if incident_cell_count == 1:
        return "one_cell_boundary_support"
    return "shared_boundary_support"


def required_dofs(case: dict[str, Any], target_rows: list[int]) -> list[int]:
    dofs = set(target_rows)
    for selector in SELECTORS:
        dofs.update(int_list(case.get(selector["candidate_key"])))
    return sorted(dofs)


def load_mesh_profile(
    *,
    case: dict[str, Any],
    target_rows: list[int],
    boundary_tolerance: float,
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

    counts = point_incident_cell_counts(grid)
    bounds = tuple(float(value) for value in grid.bounds)
    profile: dict[int, dict[str, Any]] = {}
    out_of_range = 0
    for dof in required_dofs(case, target_rows):
        point_id = dof - pressure_offset
        if point_id < 0 or point_id >= int(grid.n_points):
            out_of_range += 1
            continue
        labels = boundary_labels(
            grid.points[point_id],
            bounds,  # type: ignore[arg-type]
            tolerance=boundary_tolerance,
        )
        bclass = boundary_class(labels)
        incident_count = counts[point_id] if point_id < len(counts) else None
        profile[dof] = {
            "point_id": int(point_id),
            "boundary_labels": labels,
            "boundary_class": bclass,
            "incident_cell_count": incident_count,
            "incident_support_class": incident_support_class(
                boundary_class_value=bclass,
                incident_cell_count=incident_count,
            ),
        }

    evidence.update(
        {
            "profile_status": "ok",
            "profiled_dof_count": len(profile),
            "out_of_range_dof_count": out_of_range,
            "mesh_point_count": int(grid.n_points),
        }
    )
    return profile, evidence


def profile_matches(profile: dict[str, Any], predicate: str) -> bool:
    boundary = profile.get("boundary_class") != "interior"
    incident_count = profile.get("incident_cell_count")
    low_incident = isinstance(incident_count, int) and incident_count <= 2
    one_cell_boundary = (
        profile.get("incident_support_class") == "one_cell_boundary_support"
    )
    if predicate == "boundary":
        return boundary
    if predicate == "low_incident":
        return low_incident
    if predicate == "boundary_or_low_incident":
        return boundary or low_incident
    if predicate == "one_cell_boundary":
        return one_cell_boundary
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
        finding = "mesh_profile_missing"
    elif uncovered:
        finding = "selector_misses_targets"
    elif selected_ratio is not None and selected_ratio > max_target_ratio:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"

    boundary_classes = Counter(
        profile_by_dof[dof].get("boundary_class", "missing")
        for dof in selected
    )
    incident_classes = Counter(
        profile_by_dof[dof].get("incident_support_class", "missing")
        for dof in selected
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
        "selected_boundary_class_counts": dict(boundary_classes),
        "selected_incident_support_class_counts": dict(incident_classes),
    }


def selector_finding(cases: list[dict[str, Any]]) -> str:
    findings = [case.get("finding") for case in cases]
    if any(finding == "mesh_profile_missing" for finding in findings):
        return "selector_mesh_profile_missing"
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
    boundary_tolerance: float = 1.0e-10,
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
        profile, evidence = load_mesh_profile(
            case=case,
            target_rows=targets.get(label, []),
            boundary_tolerance=boundary_tolerance,
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
        finding = "mesh_boundary_incident_selector_selective_for_formulation_replay"
        next_requirement = (
            "Prototype the selective mesh-provenance formulation gate and replay "
            "the short Test02/Test10 windows."
        )
    elif any("mesh_profile_missing" in finding for finding in selector_findings):
        finding = "mesh_boundary_incident_selector_profile_missing"
        next_requirement = (
            "Regenerate the replay candidate emission with source mesh manifests "
            "before using mesh-provenance selectivity evidence."
        )
    elif all(finding == "selector_misses_targets" for finding in selector_findings):
        finding = "mesh_boundary_incident_support_selectors_miss_audited_targets"
        next_requirement = (
            "Do not use literal mesh boundary or incident-cell support as the "
            "formulation gate. Derive active PSPG support provenance from the "
            "cut-volume pressure-gradient topology or coupled patch structure."
        )
    elif any(finding == "selector_overbroad" for finding in selector_findings):
        finding = "mesh_boundary_incident_support_selectors_overbroad"
        next_requirement = (
            "Do not promote literal mesh boundary or incident-cell support "
            "without an additional active PSPG topology gate."
        )
    else:
        finding = "mesh_boundary_incident_support_selectivity_inconclusive"
        next_requirement = (
            "Regenerate mesh-provenance evidence before choosing a formulation "
            "replay."
        )

    return {
        "scope": (
            "Selectivity audit for literal mesh boundary and incident-support "
            "provenance on globally emitted direct PSPG formulation candidates."
        ),
        "global_emission_path": (
            str(global_emission_path) if global_emission_path is not None else None
        ),
        "target_map_path": str(target_map_path) if target_map_path is not None else None,
        "max_target_ratio": max_target_ratio,
        "boundary_tolerance": boundary_tolerance,
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
        boundary_tolerance=args.boundary_tolerance,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
