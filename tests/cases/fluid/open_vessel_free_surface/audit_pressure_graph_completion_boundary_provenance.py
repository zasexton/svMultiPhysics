#!/usr/bin/env python3
"""Map graph-completion rows back to source-result boundary topology."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read pressure matrix support audit JSON, recover the replay source "
            "result from replay_manifest.json when possible, and classify graph "
            "completion candidate/balance/top-update rows by mesh boundary class."
        )
    )
    parser.add_argument(
        "--support-json",
        action="append",
        default=[],
        help="Support audit JSON as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=1.0e-10,
        help="Coordinate tolerance for bounding-box boundary classification.",
    )
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value_dict(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    values = record.get("values")
    return values if isinstance(values, dict) else record


def int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def parse_int_sample(value: Any) -> list[int]:
    if not isinstance(value, str) or not value:
        return []
    out: list[int] = []
    for item in value.split("|"):
        if item == "...":
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def parse_top_update_details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or value == "none":
        return []
    rows: list[dict[str, Any]] = []
    for item in value.split("|"):
        parts = item.split(":")
        if len(parts) < 2:
            continue
        try:
            row: dict[str, Any] = {
                "local_pressure_row": int(parts[0]),
                "global_dof": int(parts[1]),
            }
        except ValueError:
            continue
        for part in parts[2:]:
            key, separator, raw = part.partition("=")
            if not separator:
                continue
            parsed = float_or_none(raw)
            row[key] = parsed if parsed is not None else raw
        rows.append(row)
    return rows


def pressure_offset(report: dict[str, Any]) -> int | None:
    for key in (
        "latest_pressure_update_support_diagnostic",
        "latest_support_rank_diagnostic",
    ):
        offset = int_or_none(value_dict(report.get(key)).get("pressure_offset"))
        if offset is not None:
            return offset
    return None


def source_result_from_manifest(report: dict[str, Any]) -> str | None:
    solver_log = report.get("solver_log")
    if not isinstance(solver_log, str) or not solver_log:
        return None
    manifest = Path(solver_log).parent / "replay_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = load_json(manifest)
    except (OSError, json.JSONDecodeError):
        return None
    source_result = data.get("source_result")
    return source_result if isinstance(source_result, str) and source_result else None


def result_points_and_bounds(path_value: str | None) -> tuple[list[list[float]] | None, tuple[float, float, float, float, float, float] | None]:
    if not path_value:
        return None, None
    path = Path(path_value)
    if not path.exists():
        return None, None
    try:
        import pyvista as pv

        grid = pv.read(path)
    except Exception:
        return None, None
    points = [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in grid.points
    ]
    bounds = grid.bounds
    return points, (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
        float(bounds[4]),
        float(bounds[5]),
    )


def boundary_labels(
    point: list[float] | None,
    bounds: tuple[float, float, float, float, float, float] | None,
    *,
    tolerance: float,
) -> list[str]:
    if point is None or bounds is None or len(point) < 3:
        return []
    x, y, z = point[:3]
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
        if math.isfinite(value)
        and math.isfinite(boundary)
        and abs(value - boundary) <= tolerance
    ]


def boundary_class(labels: list[str]) -> str:
    if not labels:
        return "interior"
    if len(labels) == 1:
        return "boundary_face"
    if len(labels) == 2:
        return "boundary_edge"
    return "boundary_corner"


def rows_with_boundary_context(
    global_dofs: list[int],
    *,
    offset: int | None,
    points: list[list[float]] | None,
    bounds: tuple[float, float, float, float, float, float] | None,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dof in global_dofs:
        point_index = dof - offset if offset is not None else None
        point = (
            points[point_index]
            if isinstance(point_index, int)
            and points is not None
            and 0 <= point_index < len(points)
            else None
        )
        labels = boundary_labels(point, bounds, tolerance=tolerance)
        rows.append(
            {
                "global_dof": dof,
                "point_index": point_index,
                "point_m": point,
                "boundary_labels": labels,
                "boundary_class": boundary_class(labels),
            }
        )
    return rows


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, str):
            counts[value] += 1
    return dict(sorted(counts.items()))


def count_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        labels = row.get("boundary_labels")
        if not isinstance(labels, list):
            continue
        for label in labels:
            if isinstance(label, str):
                counts[label] += 1
    return dict(sorted(counts.items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "global_dofs": [row["global_dof"] for row in rows],
        "boundary_class_counts": count_by_key(rows, "boundary_class"),
        "boundary_label_counts": count_labels(rows),
        "rows": rows,
    }


def audit_boundary_provenance(
    label: str,
    path: Path,
    report: dict[str, Any],
    *,
    points: list[list[float]] | None = None,
    bounds: tuple[float, float, float, float, float, float] | None = None,
    boundary_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    source_result = source_result_from_manifest(report)
    if points is None or bounds is None:
        loaded_points, loaded_bounds = result_points_and_bounds(source_result)
        points = points if points is not None else loaded_points
        bounds = bounds if bounds is not None else loaded_bounds

    graph = value_dict(report.get("latest_pressure_graph_completion"))
    update = value_dict(report.get("latest_pressure_update_support_diagnostic"))
    offset = pressure_offset(report)
    candidate_dofs = parse_int_sample(graph.get("candidate_global_dofs"))
    balance_dofs = parse_int_sample(graph.get("balance_candidate_global_dofs"))
    low_degree_balance_dofs = parse_int_sample(
        graph.get("low_degree_balance_candidate_global_dofs")
    )
    coupling_balance_dofs = parse_int_sample(
        graph.get("coupling_deficient_balance_candidate_global_dofs")
    )
    top_update_rows = parse_top_update_details(update.get("top_update_details"))
    top_update_dofs = [row["global_dof"] for row in top_update_rows]

    candidate_rows = rows_with_boundary_context(
        candidate_dofs,
        offset=offset,
        points=points,
        bounds=bounds,
        tolerance=boundary_tolerance,
    )
    balance_rows = rows_with_boundary_context(
        balance_dofs,
        offset=offset,
        points=points,
        bounds=bounds,
        tolerance=boundary_tolerance,
    )
    low_degree_rows = rows_with_boundary_context(
        low_degree_balance_dofs,
        offset=offset,
        points=points,
        bounds=bounds,
        tolerance=boundary_tolerance,
    )
    coupling_rows = rows_with_boundary_context(
        coupling_balance_dofs,
        offset=offset,
        points=points,
        bounds=bounds,
        tolerance=boundary_tolerance,
    )
    top_rows = rows_with_boundary_context(
        top_update_dofs,
        offset=offset,
        points=points,
        bounds=bounds,
        tolerance=boundary_tolerance,
    )
    for row, detail in zip(top_rows, top_update_rows):
        row.update(
            {
                "abs_update": detail.get("abs_update"),
                "update": detail.get("update"),
                "row_coupling": detail.get("row_coupling"),
                "row_self": detail.get("row_self"),
            }
        )

    balance_set = set(balance_dofs)
    candidate_set = set(candidate_dofs)
    low_degree_balance_set = set(low_degree_balance_dofs)
    coupling_balance_set = set(coupling_balance_dofs)
    top_set = set(top_update_dofs)
    for row in top_rows:
        dof = row["global_dof"]
        row["in_candidate_sample"] = dof in candidate_set
        row["in_balance_sample"] = dof in balance_set
        row["in_low_degree_balance_sample"] = dof in low_degree_balance_set
        row["in_coupling_deficient_balance_sample"] = (
            dof in coupling_balance_set
        )
        row["is_boundary_topology"] = row["boundary_class"] != "interior"
    overlap_balance = sorted(top_set & balance_set)
    overlap_candidate = sorted(top_set & candidate_set)
    boundary_top_rows = [
        row for row in top_rows if row.get("is_boundary_topology")
    ]
    boundary_top_dofs = [row["global_dof"] for row in boundary_top_rows]
    boundary_top_set = set(boundary_top_dofs)
    boundary_candidate_overlap = sorted(boundary_top_set & candidate_set)
    boundary_balance_overlap = sorted(boundary_top_set & balance_set)
    boundary_low_degree_overlap = sorted(
        boundary_top_set & low_degree_balance_set
    )
    boundary_coupling_overlap = sorted(boundary_top_set & coupling_balance_set)
    boundary_candidate_not_balanced = sorted(
        (boundary_top_set & candidate_set) - balance_set
    )
    boundary_outside_candidate = sorted(boundary_top_set - candidate_set)
    latest_max_dof = int_or_none(update.get("max_update_global_dof"))
    latest_max_in_balance = (
        latest_max_dof in balance_set if latest_max_dof is not None else False
    )
    latest_max_in_candidate_sample = (
        latest_max_dof in candidate_set if latest_max_dof is not None else False
    )
    if not balance_dofs:
        finding = "balance_row_sample_missing_or_empty"
    elif latest_max_in_balance:
        finding = "latest_max_update_row_in_balance_sample"
    elif latest_max_in_candidate_sample:
        finding = "latest_max_update_row_candidate_not_balanced"
    else:
        finding = "latest_max_update_row_outside_candidate_sample"

    if not boundary_top_rows:
        boundary_finding = "no_boundary_top_update_rows"
    elif boundary_candidate_not_balanced:
        boundary_finding = "boundary_top_update_candidates_missing_balance"
    elif boundary_outside_candidate:
        boundary_finding = "boundary_top_update_rows_outside_candidate_sample"
    elif len(boundary_balance_overlap) == len(boundary_top_rows):
        boundary_finding = "boundary_top_update_rows_balanced"
    elif boundary_balance_overlap:
        boundary_finding = "partial_boundary_top_update_balance_coverage"
    else:
        boundary_finding = "boundary_top_update_rows_unbalanced"

    return {
        "label": label,
        "path": str(path),
        "finding": finding,
        "boundary_topology_finding": boundary_finding,
        "source_result": source_result,
        "source_result_loaded": points is not None and bounds is not None,
        "pressure_offset": offset,
        "graph_mode": graph.get("mode"),
        "requested_mode": graph.get("requested_mode"),
        "candidate_row_count": graph.get("candidate_row_count"),
        "balance_candidate_row_count": graph.get("balance_candidate_row_count"),
        "low_degree_balance_candidate_count": graph.get(
            "low_degree_balance_candidate_count"
        ),
        "coupling_deficient_balance_candidate_count": graph.get(
            "coupling_deficient_balance_candidate_count"
        ),
        "latest_max_update_global_dof": latest_max_dof,
        "latest_max_update_in_candidate_sample": latest_max_in_candidate_sample,
        "latest_max_update_in_balance_sample": latest_max_in_balance,
        "top_update_count": len(top_rows),
        "top_update_candidate_overlap_count": len(overlap_candidate),
        "top_update_balance_overlap_count": len(overlap_balance),
        "top_update_candidate_overlap_global_dofs": overlap_candidate,
        "top_update_balance_overlap_global_dofs": overlap_balance,
        "boundary_top_update_count": len(boundary_top_rows),
        "boundary_top_update_global_dofs": boundary_top_dofs,
        "boundary_top_update_candidate_overlap_count": len(
            boundary_candidate_overlap
        ),
        "boundary_top_update_balance_overlap_count": len(
            boundary_balance_overlap
        ),
        "boundary_top_update_low_degree_balance_overlap_count": len(
            boundary_low_degree_overlap
        ),
        "boundary_top_update_coupling_deficient_balance_overlap_count": len(
            boundary_coupling_overlap
        ),
        "boundary_top_update_candidate_overlap_global_dofs": (
            boundary_candidate_overlap
        ),
        "boundary_top_update_balance_overlap_global_dofs": (
            boundary_balance_overlap
        ),
        "boundary_top_update_low_degree_balance_overlap_global_dofs": (
            boundary_low_degree_overlap
        ),
        "boundary_top_update_coupling_deficient_balance_overlap_global_dofs": (
            boundary_coupling_overlap
        ),
        "boundary_top_update_candidate_not_balanced_global_dofs": (
            boundary_candidate_not_balanced
        ),
        "boundary_top_update_outside_candidate_global_dofs": (
            boundary_outside_candidate
        ),
        "candidate_sample": summarize_rows(candidate_rows),
        "balance_sample": summarize_rows(balance_rows),
        "low_degree_balance_sample": summarize_rows(low_degree_rows),
        "coupling_deficient_balance_sample": summarize_rows(coupling_rows),
        "top_update_sample": summarize_rows(top_rows),
    }


def summarize_boundary_provenance(
    reports: list[tuple[str, Path, dict[str, Any]]],
    *,
    boundary_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    cases = [
        audit_boundary_provenance(
            label,
            path,
            report,
            boundary_tolerance=boundary_tolerance,
        )
        for label, path, report in reports
    ]
    finding_counts = Counter(case["finding"] for case in cases)
    boundary_finding_counts = Counter(
        case["boundary_topology_finding"] for case in cases
    )
    if any(
        case["finding"] == "latest_max_update_row_candidate_not_balanced"
        for case in cases
    ):
        finding = "latest_bad_rows_can_be_candidates_without_balance_coverage"
    elif any(
        case["finding"] == "latest_max_update_row_outside_candidate_sample"
        for case in cases
    ):
        finding = "latest_bad_rows_not_seen_in_candidate_sample"
    elif all(
        case["finding"] == "latest_max_update_row_in_balance_sample"
        for case in cases
    ):
        finding = "latest_bad_rows_are_balance_sampled"
    else:
        finding = "boundary_provenance_incomplete"
    if any(
        case["boundary_topology_finding"]
        == "boundary_top_update_candidates_missing_balance"
        for case in cases
    ):
        boundary_finding = "boundary_top_update_candidates_missing_balance"
    elif any(
        case["boundary_topology_finding"]
        == "boundary_top_update_rows_outside_candidate_sample"
        for case in cases
    ):
        boundary_finding = "boundary_top_update_rows_outside_candidate_sample"
    elif all(
        case["boundary_topology_finding"]
        == "boundary_top_update_rows_balanced"
        for case in cases
    ):
        boundary_finding = "boundary_top_update_rows_balanced"
    else:
        boundary_finding = "boundary_topology_balance_coverage_incomplete"
    return {
        "finding": finding,
        "boundary_topology_finding": boundary_finding,
        "case_count": len(cases),
        "finding_counts": dict(sorted(finding_counts.items())),
        "boundary_topology_finding_counts": dict(
            sorted(boundary_finding_counts.items())
        ),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    reports = [
        (label, path, load_json(path))
        for label, path in (parse_labeled_path(value) for value in args.support_json)
    ]
    summary = summarize_boundary_provenance(
        reports,
        boundary_tolerance=args.boundary_tolerance,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
