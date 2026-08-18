#!/usr/bin/env python3
"""Audit named face provenance for direct PSPG formulation candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_named_face_provenance_selectivity_20260607.json"
)

CANDIDATE_KEYS = (
    (
        "preferred",
        "preferred_candidate_global_dofs",
        "Preferred globally emitted direct PSPG candidates.",
    ),
    (
        "sparse_direct_self",
        "sparse_direct_self_global_dofs",
        "Sparse direct-self direct PSPG candidates.",
    ),
    (
        "sparse_or_moderate_direct_self",
        "sparse_or_moderate_direct_self_ratio_global_dofs",
        "Sparse-or-moderate direct-self support-ratio direct PSPG candidates.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map globally emitted direct PSPG candidates onto named wall and "
            "obstacle surface files, then test whether named face provenance "
            "is selective enough to explain Test02/Test10 target rows."
        )
    )
    parser.add_argument(
        "--global-emission-json",
        type=Path,
        default=DEFAULT_GLOBAL_EMISSION,
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio still considered selective.",
    )
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=1.0e-9,
        help="Coordinate matching tolerance for source mesh and face surfaces.",
    )
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
    cases: dict[str, list[int]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = int_list(case.get("direct_pspg_target_global_dofs"))
    return cases


def global_case_map(global_emission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(global_emission.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def source_paths_from_manifest(case: dict[str, Any]) -> dict[str, Any]:
    log_path = case.get("path")
    if not isinstance(log_path, str) or not log_path:
        return {"manifest_path": None, "manifest_exists": False}
    manifest_path = Path(log_path).parent / "replay_manifest.json"
    evidence: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
    }
    if not manifest_path.exists():
        return evidence
    try:
        manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        evidence["manifest_error"] = str(exc)
        return evidence
    evidence["source_case"] = manifest.get("source_case")
    evidence["source_result"] = manifest.get("source_result")
    return evidence


def face_family(face_name: str) -> str:
    if "_" not in face_name:
        return face_name
    return face_name.split("_", 1)[0]


def face_class(face_names: list[str]) -> str:
    if not face_names:
        return "no_named_face"
    if len(face_names) == 1:
        return "single_named_face"
    if len(face_names) == 2:
        return "named_face_intersection"
    return "multi_face_intersection"


def quantized(point: Any, tolerance: float) -> tuple[int, int, int]:
    return tuple(int(round(float(point[i]) / tolerance)) for i in range(3))


def candidate_surface_dirs(source_case: Path) -> list[Path]:
    return [
        source_case / "mesh" / "background" / "mesh-surfaces",
        source_case / "mesh" / "water" / "mesh-surfaces",
        source_case / "mesh" / "mesh-surfaces",
    ]


def load_named_face_profile(
    *,
    case: dict[str, Any],
    target_rows: list[int],
    coordinate_tolerance: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    evidence = source_paths_from_manifest(case)
    source_result = evidence.get("source_result")
    source_case = evidence.get("source_case")
    pressure_offset = case.get("pressure_offset")
    evidence["pressure_offset"] = pressure_offset
    if not isinstance(source_result, str) or not Path(source_result).exists():
        evidence["profile_status"] = "source_result_missing"
        return {}, evidence
    if not isinstance(source_case, str) or not Path(source_case).exists():
        evidence["profile_status"] = "source_case_missing"
        return {}, evidence
    if not isinstance(pressure_offset, int):
        evidence["profile_status"] = "pressure_offset_missing"
        return {}, evidence
    try:
        import pyvista as pv

        grid = pv.read(source_result)
    except Exception as exc:
        evidence["profile_status"] = "source_mesh_load_failed"
        evidence["profile_error"] = str(exc)
        return {}, evidence

    point_keys: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for point_id, point in enumerate(grid.points):
        point_keys[quantized(point, coordinate_tolerance)].append(int(point_id))

    source_case_path = Path(source_case)
    surface_dirs = [path for path in candidate_surface_dirs(source_case_path) if path.exists()]
    evidence["surface_dirs"] = [str(path) for path in surface_dirs]
    face_point_ids: dict[int, set[str]] = defaultdict(set)
    face_files = sorted(path for directory in surface_dirs for path in directory.glob("*.vtp"))
    for face_file in face_files:
        face_name = face_file.stem
        try:
            surface = pv.read(face_file)
        except Exception:
            continue
        for point in surface.points:
            for point_id in point_keys.get(quantized(point, coordinate_tolerance), []):
                face_point_ids[point_id].add(face_name)

    required_dofs = set(target_rows)
    for _, key, _ in CANDIDATE_KEYS:
        required_dofs.update(int_list(case.get(key)))
    profile: dict[int, dict[str, Any]] = {}
    out_of_range = 0
    for dof in sorted(required_dofs):
        point_id = dof - pressure_offset
        if point_id < 0 or point_id >= int(grid.n_points):
            out_of_range += 1
            continue
        faces = sorted(face_point_ids.get(point_id, set()))
        families = sorted({face_family(face) for face in faces})
        profile[dof] = {
            "point_id": int(point_id),
            "named_faces": faces,
            "face_families": families,
            "face_class": face_class(faces),
            "named_face_count": len(faces),
        }
    evidence.update(
        {
            "profile_status": "ok",
            "source_result": source_result,
            "source_case": source_case,
            "surface_file_count": len(face_files),
            "profiled_dof_count": len(profile),
            "out_of_range_dof_count": out_of_range,
            "coordinate_tolerance": coordinate_tolerance,
        }
    )
    return profile, evidence


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def evaluate_selector(
    *,
    key: str,
    description: str,
    selected_rows: set[int],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    target_set = set(target_rows)
    covered = sorted(target_set & selected_rows)
    uncovered = sorted(target_set - selected_rows)
    selected_count = len(selected_rows)
    target_count = len(target_rows)
    selected_to_target_ratio = ratio(selected_count, target_count)
    covers_targets = target_count > 0 and len(covered) == target_count
    overbroad = (
        selected_to_target_ratio is not None
        and selected_to_target_ratio > max_target_ratio
    )
    if not covers_targets and overbroad:
        finding = "selector_overbroad_and_misses_targets"
    elif not covers_targets:
        finding = "selector_misses_targets"
    elif overbroad:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"
    return {
        "key": key,
        "description": description,
        "finding": finding,
        "selected_count": selected_count,
        "target_count": target_count,
        "selected_to_target_ratio": selected_to_target_ratio,
        "covered_target_count": len(covered),
        "covered_target_global_dofs": covered,
        "uncovered_target_global_dofs": uncovered,
        "covers_targets": covers_targets,
        "selector_overbroad": overbroad,
    }


def profile_values(
    profile: dict[int, dict[str, Any]],
    rows: list[int],
    key: str,
) -> set[Any]:
    values: set[Any] = set()
    for row in rows:
        item = profile.get(row, {})
        value = item.get(key)
        if isinstance(value, list):
            values.update(value)
        elif value is not None:
            values.add(value)
    return values


def profile_signatures(
    profile: dict[int, dict[str, Any]],
    rows: list[int],
) -> set[tuple[str, ...]]:
    signatures: set[tuple[str, ...]] = set()
    for row in rows:
        faces = profile.get(row, {}).get("named_faces")
        if isinstance(faces, list):
            signatures.add(tuple(faces))
    return signatures


def rows_matching_values(
    profile: dict[int, dict[str, Any]],
    rows: set[int],
    key: str,
    values: set[Any],
) -> set[int]:
    if not values:
        return set()
    selected: set[int] = set()
    for row in rows:
        item = profile.get(row, {})
        value = item.get(key)
        if isinstance(value, list):
            if set(value) & values:
                selected.add(row)
        elif value in values:
            selected.add(row)
    return selected


def rows_matching_signatures(
    profile: dict[int, dict[str, Any]],
    rows: set[int],
    signatures: set[tuple[str, ...]],
) -> set[int]:
    if not signatures:
        return set()
    selected: set[int] = set()
    for row in rows:
        faces = profile.get(row, {}).get("named_faces")
        if isinstance(faces, list) and tuple(faces) in signatures:
            selected.add(row)
    return selected


def target_profile_rows(
    profile: dict[int, dict[str, Any]],
    target_rows: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for row in target_rows:
        item = profile.get(row)
        if item is None:
            rows.append({"row_dof": row, "present": False})
        else:
            rows.append({"row_dof": row, "present": True, **item})
    return rows


def build_case_report(
    *,
    label: str,
    emission_case: dict[str, Any],
    target_rows: list[int],
    profile: dict[int, dict[str, Any]],
    profile_evidence: dict[str, Any],
    max_target_ratio: float,
) -> dict[str, Any]:
    target_named_faces = profile_values(profile, target_rows, "named_faces")
    target_face_families = profile_values(profile, target_rows, "face_families")
    target_face_classes = profile_values(profile, target_rows, "face_class")
    target_face_counts = profile_values(profile, target_rows, "named_face_count")
    target_signatures = profile_signatures(profile, target_rows)
    selectors: list[dict[str, Any]] = []
    for prefix, candidate_key, candidate_description in CANDIDATE_KEYS:
        candidate_rows = set(int_list(emission_case.get(candidate_key)))
        selector_prefix = f"{prefix}_"
        selectors.extend(
            [
                evaluate_selector(
                    key=f"{selector_prefix}target_named_face_union",
                    description=(
                        f"{candidate_description} Restricted to rows touching "
                        "any named face touched by an audited target."
                    ),
                    selected_rows=rows_matching_values(
                        profile, candidate_rows, "named_faces", target_named_faces
                    ),
                    target_rows=target_rows,
                    max_target_ratio=max_target_ratio,
                ),
                evaluate_selector(
                    key=f"{selector_prefix}target_named_face_signature",
                    description=(
                        f"{candidate_description} Restricted to rows whose "
                        "complete named-face set matches an audited target."
                    ),
                    selected_rows=rows_matching_signatures(
                        profile, candidate_rows, target_signatures
                    ),
                    target_rows=target_rows,
                    max_target_ratio=max_target_ratio,
                ),
                evaluate_selector(
                    key=f"{selector_prefix}target_face_family_union",
                    description=(
                        f"{candidate_description} Restricted to rows touching "
                        "any target face family such as wall or obstacle."
                    ),
                    selected_rows=rows_matching_values(
                        profile, candidate_rows, "face_families", target_face_families
                    ),
                    target_rows=target_rows,
                    max_target_ratio=max_target_ratio,
                ),
                evaluate_selector(
                    key=f"{selector_prefix}target_face_class",
                    description=(
                        f"{candidate_description} Restricted to rows with the "
                        "same named-face intersection class as the targets."
                    ),
                    selected_rows=rows_matching_values(
                        profile, candidate_rows, "face_class", target_face_classes
                    ),
                    target_rows=target_rows,
                    max_target_ratio=max_target_ratio,
                ),
                evaluate_selector(
                    key=f"{selector_prefix}target_named_face_count",
                    description=(
                        f"{candidate_description} Restricted to rows with the "
                        "same number of named face memberships as the targets."
                    ),
                    selected_rows=rows_matching_values(
                        profile, candidate_rows, "named_face_count", target_face_counts
                    ),
                    target_rows=target_rows,
                    max_target_ratio=max_target_ratio,
                ),
            ]
        )
    target_rows_profile = target_profile_rows(profile, target_rows)
    target_rows_present = sum(1 for row in target_rows_profile if row.get("present"))
    if profile_evidence.get("profile_status") != "ok":
        finding = "named_face_provenance_evidence_missing"
    elif target_rows_present < len(target_rows):
        finding = "named_face_provenance_missing_target_rows"
    elif any(selector["finding"] == "selector_selective" for selector in selectors):
        finding = "named_face_provenance_selector_candidate"
    else:
        finding = "named_face_provenance_selectors_overbroad_or_miss_targets"
    return {
        "label": label,
        "finding": finding,
        "profile_evidence": profile_evidence,
        "target_count": len(target_rows),
        "target_rows_present_count": target_rows_present,
        "target_named_faces": sorted(target_named_faces),
        "target_face_families": sorted(target_face_families),
        "target_face_classes": sorted(target_face_classes),
        "target_named_face_counts": sorted(target_face_counts),
        "target_rows": target_rows_profile,
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "direct_pspg_named_face_provenance_selectivity_missing_evidence",
            "regenerate_named_face_profiles",
        )
    if cases and all(
        case.get("finding") == "named_face_provenance_selector_candidate"
        for case in cases
    ):
        return (
            "direct_pspg_named_face_provenance_selector_candidate",
            "candidate_requires_solve_time_formulation_replay",
        )
    return (
        "direct_pspg_named_face_provenance_selectors_not_formulation_ready",
        "named_face_boundary_gate_ruled_out",
    )


def build_report(
    *,
    global_emission: dict[str, Any],
    target_map: dict[str, Any],
    profiles_by_case: dict[str, dict[int, dict[str, Any]]],
    profile_evidence_by_case: dict[str, dict[str, Any]] | None = None,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    emission_cases = global_case_map(global_emission)
    target_cases = target_case_map(target_map)
    profile_evidence_by_case = profile_evidence_by_case or {}
    labels = ["test02", "test10"]
    cases = [
        build_case_report(
            label=label,
            emission_case=emission_cases.get(label, {}),
            target_rows=target_cases.get(label, []),
            profile=profiles_by_case.get(label, {}),
            profile_evidence=profile_evidence_by_case.get(label, {}),
            max_target_ratio=max_target_ratio,
        )
        for label in labels
    ]
    finding, status = aggregate_finding(cases)
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Named wall/obstacle face provenance selectivity for globally "
            "emitted direct PSPG formulation candidates in short Test02/Test10 "
            "replay source meshes."
        ),
        "max_target_ratio": max_target_ratio,
        "candidate_keys": [
            {"prefix": prefix, "key": key, "description": description}
            for prefix, key, description in CANDIDATE_KEYS
        ],
        "cases": cases,
        "conclusion": (
            "Named wall/obstacle face provenance does not provide the missing "
            "direct PSPG formulation gate when applied to the globally emitted "
            "candidate sets. Target named faces and face-intersection classes "
            "either miss branch-specific targets or select broad candidate "
            "families, so the remaining rule still has to come from active "
            "pressure-gradient support/coupling topology rather than raw "
            "named-boundary membership."
        ),
        "next_requirement": (
            "Continue the direct PSPG support/coupling search with a physical "
            "topology discriminator beyond named wall/obstacle face membership, "
            "or prototype a Test02 pressure-update guard if formulation-side "
            "selectors remain broad."
        ),
    }


def main() -> int:
    args = parse_args()
    global_emission = load_json(args.global_emission_json)
    target_map = load_json(args.target_map_json)
    emission_cases = global_case_map(global_emission)
    target_cases = target_case_map(target_map)
    profiles_by_case: dict[str, dict[int, dict[str, Any]]] = {}
    evidence_by_case: dict[str, dict[str, Any]] = {}
    for label in ("test02", "test10"):
        profile, evidence = load_named_face_profile(
            case=emission_cases.get(label, {}),
            target_rows=target_cases.get(label, []),
            coordinate_tolerance=args.coordinate_tolerance,
        )
        profiles_by_case[label] = profile
        evidence_by_case[label] = evidence
    report = build_report(
        global_emission=global_emission,
        target_map=target_map,
        profiles_by_case=profiles_by_case,
        profile_evidence_by_case=evidence_by_case,
        max_target_ratio=args.max_target_ratio,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
