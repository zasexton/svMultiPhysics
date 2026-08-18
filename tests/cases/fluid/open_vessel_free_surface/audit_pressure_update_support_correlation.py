#!/usr/bin/env python3
"""Correlate accepted pressure updates with sampled matrix support rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import pyvista as pv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join audit_pressure_update_guard JSON with "
            "audit_pressure_matrix_support_samples JSON. Matrix rows are "
            "matched to pressure-update points through constraint-sample "
            "Vertex entity_id values."
        )
    )
    parser.add_argument("--pressure-update-json", type=Path, required=True)
    parser.add_argument("--matrix-support-json", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--top-events", type=int, default=12)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-14)
    parser.add_argument(
        "--weak-velocity-row-sum",
        type=float,
        help="Optional threshold for labeling positive but weak coupling rows.",
    )
    parser.add_argument(
        "--weak-pressure-row-sum",
        type=float,
        help=(
            "Optional threshold for labeling positive but weak pressure self-block "
            "rows."
        ),
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=1.0e-10,
        help=(
            "Coordinate tolerance for labeling top updates on result-mesh "
            "bounding-box planes."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_transition(update_report: dict[str, Any]) -> dict[str, Any] | None:
    transitions = update_report.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return None
    return transitions[-1]


def vertex_support_rows(matrix_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in matrix_report.get("sampled_pressure_rows", []):
        if not isinstance(row, dict):
            continue
        sample = row.get("constraint_sample")
        if not isinstance(sample, dict):
            continue
        if sample.get("entity_kind") != "Vertex":
            continue
        entity_id = sample.get("entity_id")
        if not isinstance(entity_id, int):
            continue
        rows[entity_id] = row
    return rows


def coupling_class(
    row_velocity_abs_sum: float | None,
    *,
    zero_tolerance: float,
    weak_velocity_row_sum: float | None,
) -> str:
    if row_velocity_abs_sum is None:
        return "unmatched"
    if abs(row_velocity_abs_sum) <= zero_tolerance:
        return "zero_velocity_coupling"
    if (
        weak_velocity_row_sum is not None
        and abs(row_velocity_abs_sum) <= weak_velocity_row_sum
    ):
        return "weak_velocity_coupling"
    return "positive_velocity_coupling"


def pressure_self_class(
    row_pressure_abs_sum: float | None,
    *,
    zero_tolerance: float,
    weak_pressure_row_sum: float | None,
) -> str:
    if row_pressure_abs_sum is None:
        return "unmatched"
    if abs(row_pressure_abs_sum) <= zero_tolerance:
        return "zero_pressure_self"
    if (
        weak_pressure_row_sum is not None
        and abs(row_pressure_abs_sum) <= weak_pressure_row_sum
    ):
        return "weak_pressure_self"
    return "positive_pressure_self"


def numeric_or_none(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) else None


def result_bounds(path_value: Any) -> tuple[float, float, float, float, float, float] | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    grid = pv.read(path)
    bounds = grid.bounds
    return (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
        float(bounds[4]),
        float(bounds[5]),
    )


def result_points(path_value: Any) -> list[list[float]] | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    grid = pv.read(path)
    return [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in grid.points
    ]


def fallback_bounds_from_updates(
    update_report: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    coords: list[list[float]] = []
    transition = latest_transition(update_report)
    if transition is None:
        return None
    top_updates = transition.get("top_pressure_updates")
    if not isinstance(top_updates, list):
        return None
    for update in top_updates:
        if not isinstance(update, dict):
            continue
        point = update.get("point_m")
        if (
            isinstance(point, list)
            and len(point) >= 3
            and all(isinstance(value, (int, float)) for value in point[:3])
        ):
            coords.append([float(point[0]), float(point[1]), float(point[2])])
    if not coords:
        return None
    return (
        min(point[0] for point in coords),
        max(point[0] for point in coords),
        min(point[1] for point in coords),
        max(point[1] for point in coords),
        min(point[2] for point in coords),
        max(point[2] for point in coords),
    )


def update_points_by_index(update_report: dict[str, Any]) -> dict[int, list[float]]:
    points: dict[int, list[float]] = {}
    transition = latest_transition(update_report)
    if transition is None:
        return points
    top_updates = transition.get("top_pressure_updates")
    if not isinstance(top_updates, list):
        return points
    for update in top_updates:
        if not isinstance(update, dict):
            continue
        point_index = update.get("point_index")
        point = update.get("point_m")
        if (
            isinstance(point_index, int)
            and isinstance(point, list)
            and len(point) >= 3
            and all(isinstance(value, (int, float)) for value in point[:3])
        ):
            points[point_index] = [
                float(point[0]),
                float(point[1]),
                float(point[2]),
            ]
    return points


def boundary_labels(
    point: Any,
    bounds: tuple[float, float, float, float, float, float] | None,
    *,
    tolerance: float,
) -> list[str]:
    if bounds is None:
        return []
    if (
        not isinstance(point, list)
        or len(point) < 3
        or not all(isinstance(value, (int, float)) for value in point[:3])
    ):
        return []
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    candidates = (
        ("x_min", x, bounds[0]),
        ("x_max", x, bounds[1]),
        ("y_min", y, bounds[2]),
        ("y_max", y, bounds[3]),
        ("z_min", z, bounds[4]),
        ("z_max", z, bounds[5]),
    )
    labels = [
        label
        for label, value, boundary in candidates
        if math.isfinite(value)
        and math.isfinite(boundary)
        and abs(value - boundary) <= tolerance
    ]
    return labels


def boundary_class(labels: list[str]) -> str:
    if not labels:
        return "interior"
    if len(labels) == 1:
        return "boundary_face"
    if len(labels) == 2:
        return "boundary_edge"
    return "boundary_corner"


def parse_support_rank_row_details(value: Any) -> list[dict[str, Any]]:
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
            if "=" not in part:
                continue
            key, raw = part.split("=", 1)
            try:
                row[key] = float(raw)
            except ValueError:
                row[key] = raw
        rows.append(row)
    return rows


def local_support_rows(matrix_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in matrix_report.get("sampled_pressure_rows", []):
        if not isinstance(row, dict):
            continue
        local_pressure_row = row.get("local_pressure_row")
        if not isinstance(local_pressure_row, int):
            continue
        rows[local_pressure_row] = row
    return rows


def support_rank_rows_with_context(
    *,
    details_value: Any,
    matrix_support_report: dict[str, Any],
    points: list[list[float]] | None,
    fallback_points_by_index: dict[int, list[float]],
    bounds: tuple[float, float, float, float, float, float] | None,
    boundary_tolerance: float,
) -> list[dict[str, Any]]:
    rows_by_local = local_support_rows(matrix_support_report)
    out: list[dict[str, Any]] = []
    for detail in parse_support_rank_row_details(details_value):
        local_pressure_row = detail.get("local_pressure_row")
        sampled_row = (
            rows_by_local.get(local_pressure_row)
            if isinstance(local_pressure_row, int)
            else None
        )
        constraint_sample = (
            sampled_row.get("constraint_sample", {})
            if isinstance(sampled_row, dict)
            else {}
        )
        row_constrained_sums = (
            sampled_row.get("row_constrained_field_abs_sum_by_field", {})
            if isinstance(sampled_row, dict)
            else {}
        )
        row_unconstrained_sums = (
            sampled_row.get("row_unconstrained_field_abs_sum_by_field", {})
            if isinstance(sampled_row, dict)
            else {}
        )
        col_constrained_sums = (
            sampled_row.get("col_constrained_field_abs_sum_by_field", {})
            if isinstance(sampled_row, dict)
            else {}
        )
        col_unconstrained_sums = (
            sampled_row.get("col_unconstrained_field_abs_sum_by_field", {})
            if isinstance(sampled_row, dict)
            else {}
        )
        point_index = (
            constraint_sample.get("entity_id")
            if isinstance(constraint_sample, dict)
            and constraint_sample.get("entity_kind") == "Vertex"
            else None
        )
        point = None
        if (
            isinstance(point_index, int)
            and points is not None
            and 0 <= point_index < len(points)
        ):
            point = points[point_index]
        if point is None and isinstance(point_index, int):
            point = fallback_points_by_index.get(point_index)
        labels = boundary_labels(point, bounds, tolerance=boundary_tolerance)
        out.append(
            {
                **detail,
                "matched_matrix_support": sampled_row is not None,
                "point_index": point_index,
                "point_m": point,
                "boundary_labels": labels,
                "boundary_class": boundary_class(labels),
                "active_dof_support": constraint_sample.get("active_dof_support")
                if isinstance(constraint_sample, dict)
                else None,
                "inactive_constraint": constraint_sample.get("inactive_constraint")
                if isinstance(constraint_sample, dict)
                else None,
                "retained_measure": numeric_or_none(
                    constraint_sample.get("retained_measure")
                )
                if isinstance(constraint_sample, dict)
                else None,
                "retained_rule_count": constraint_sample.get("retained_rule_count")
                if isinstance(constraint_sample, dict)
                else None,
                "row_constrained_velocity_abs_sum": numeric_or_none(
                    row_constrained_sums.get("Velocity")
                    if isinstance(row_constrained_sums, dict)
                    else None
                ),
                "row_unconstrained_velocity_abs_sum": numeric_or_none(
                    row_unconstrained_sums.get("Velocity")
                    if isinstance(row_unconstrained_sums, dict)
                    else None
                ),
                "row_constrained_pressure_abs_sum": numeric_or_none(
                    row_constrained_sums.get("Pressure")
                    if isinstance(row_constrained_sums, dict)
                    else None
                ),
                "row_unconstrained_pressure_abs_sum": numeric_or_none(
                    row_unconstrained_sums.get("Pressure")
                    if isinstance(row_unconstrained_sums, dict)
                    else None
                ),
                "col_constrained_velocity_abs_sum": numeric_or_none(
                    col_constrained_sums.get("Velocity")
                    if isinstance(col_constrained_sums, dict)
                    else None
                ),
                "col_unconstrained_velocity_abs_sum": numeric_or_none(
                    col_unconstrained_sums.get("Velocity")
                    if isinstance(col_unconstrained_sums, dict)
                    else None
                ),
            }
        )
    return out


def boundary_counts_for_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        bclass = row.get("boundary_class")
        if isinstance(bclass, str):
            counts[bclass] += 1
    return dict(sorted(counts.items()))


def boundary_label_counts_for_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        labels = row.get("boundary_labels")
        if not isinstance(labels, list):
            continue
        for label in labels:
            if isinstance(label, str):
                counts[label] += 1
    return dict(sorted(counts.items()))


def positive_count(
    rows: list[dict[str, Any]],
    key: str,
    *,
    zero_tolerance: float,
) -> int:
    count = 0
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and abs(float(value)) > zero_tolerance:
            count += 1
    return count


def support_split_summary(
    rows: list[dict[str, Any]],
    *,
    zero_tolerance: float,
) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "boundary_class_counts": boundary_counts_for_rows(rows),
        "boundary_label_counts": boundary_label_counts_for_rows(rows),
        "row_constrained_velocity_positive_count": positive_count(
            rows,
            "row_constrained_velocity_abs_sum",
            zero_tolerance=zero_tolerance,
        ),
        "row_unconstrained_velocity_positive_count": positive_count(
            rows,
            "row_unconstrained_velocity_abs_sum",
            zero_tolerance=zero_tolerance,
        ),
        "row_constrained_pressure_positive_count": positive_count(
            rows,
            "row_constrained_pressure_abs_sum",
            zero_tolerance=zero_tolerance,
        ),
        "row_unconstrained_pressure_positive_count": positive_count(
            rows,
            "row_unconstrained_pressure_abs_sum",
            zero_tolerance=zero_tolerance,
        ),
        "col_constrained_velocity_positive_count": positive_count(
            rows,
            "col_constrained_velocity_abs_sum",
            zero_tolerance=zero_tolerance,
        ),
        "col_unconstrained_velocity_positive_count": positive_count(
            rows,
            "col_unconstrained_velocity_abs_sum",
            zero_tolerance=zero_tolerance,
        ),
    }


def correlate_pressure_updates_with_support(
    *,
    pressure_update_report: dict[str, Any],
    matrix_support_report: dict[str, Any],
    top_events: int = 12,
    zero_tolerance: float = 1.0e-14,
    weak_velocity_row_sum: float | None = None,
    weak_pressure_row_sum: float | None = None,
    boundary_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    transition = latest_transition(pressure_update_report)
    top_updates = []
    if transition is not None and isinstance(transition.get("top_pressure_updates"), list):
        top_updates = transition["top_pressure_updates"][:top_events]

    rows_by_vertex = vertex_support_rows(matrix_support_report)
    bounds = result_bounds(pressure_update_report.get("current_result"))
    points = result_points(pressure_update_report.get("current_result"))
    fallback_points = update_points_by_index(pressure_update_report)
    bounds_source = "current_result"
    if bounds is None:
        bounds = fallback_bounds_from_updates(pressure_update_report)
        bounds_source = "top_pressure_updates"
    correlated: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    class_max_delta: dict[str, float] = {}
    pressure_self_class_counts: Counter[str] = Counter()
    pressure_self_class_max_delta: dict[str, float] = {}
    boundary_class_counts: Counter[str] = Counter()
    boundary_label_counts: Counter[str] = Counter()
    coupling_by_boundary_class: Counter[str] = Counter()
    pressure_self_by_boundary_class: Counter[str] = Counter()
    coupling_by_pressure_self_class: Counter[str] = Counter()

    for rank, update in enumerate(top_updates, start=1):
        if not isinstance(update, dict):
            continue
        point_index = update.get("point_index")
        row = rows_by_vertex.get(point_index) if isinstance(point_index, int) else None
        matrix_sample = row.get("matrix_sample", {}) if isinstance(row, dict) else {}
        constraint_sample = row.get("constraint_sample", {}) if isinstance(row, dict) else {}
        row_sums = row.get("row_field_abs_sum_by_field", {}) if isinstance(row, dict) else {}
        col_sums = row.get("col_field_abs_sum_by_field", {}) if isinstance(row, dict) else {}
        row_constrained_sums = (
            row.get("row_constrained_field_abs_sum_by_field", {})
            if isinstance(row, dict)
            else {}
        )
        row_unconstrained_sums = (
            row.get("row_unconstrained_field_abs_sum_by_field", {})
            if isinstance(row, dict)
            else {}
        )
        col_constrained_sums = (
            row.get("col_constrained_field_abs_sum_by_field", {})
            if isinstance(row, dict)
            else {}
        )
        col_unconstrained_sums = (
            row.get("col_unconstrained_field_abs_sum_by_field", {})
            if isinstance(row, dict)
            else {}
        )

        row_velocity = (
            row_sums.get("Velocity") if isinstance(row_sums, dict) else None
        )
        row_pressure = (
            row_sums.get("Pressure") if isinstance(row_sums, dict) else None
        )
        cls = coupling_class(
            row_velocity if isinstance(row_velocity, (int, float)) else None,
            zero_tolerance=zero_tolerance,
            weak_velocity_row_sum=weak_velocity_row_sum,
        )
        class_counts[cls] += 1
        pressure_cls = pressure_self_class(
            row_pressure if isinstance(row_pressure, (int, float)) else None,
            zero_tolerance=zero_tolerance,
            weak_pressure_row_sum=weak_pressure_row_sum,
        )
        pressure_self_class_counts[pressure_cls] += 1

        abs_delta = numeric_or_none(update.get("abs_pressure_delta_pa"))
        if isinstance(abs_delta, (int, float)):
            class_max_delta[cls] = max(float(abs_delta), class_max_delta.get(cls, 0.0))
            pressure_self_class_max_delta[pressure_cls] = max(
                float(abs_delta),
                pressure_self_class_max_delta.get(pressure_cls, 0.0),
            )

        labels = boundary_labels(
            update.get("point_m"),
            bounds,
            tolerance=boundary_tolerance,
        )
        bclass = boundary_class(labels)
        boundary_class_counts[bclass] += 1
        for label in labels:
            boundary_label_counts[label] += 1
        coupling_by_boundary_class[f"{bclass}:{cls}"] += 1
        pressure_self_by_boundary_class[f"{bclass}:{pressure_cls}"] += 1
        coupling_by_pressure_self_class[f"{cls}:{pressure_cls}"] += 1

        row_velocity_numeric = numeric_or_none(row_velocity)
        row_pressure_numeric = numeric_or_none(row_pressure)
        pressure_to_velocity_ratio = None
        if (
            isinstance(row_velocity_numeric, (int, float))
            and isinstance(row_pressure_numeric, (int, float))
            and abs(float(row_velocity_numeric)) > zero_tolerance
        ):
            pressure_to_velocity_ratio = (
                abs(float(row_pressure_numeric)) / abs(float(row_velocity_numeric))
            )

        correlated.append(
            {
                "rank": rank,
                "point_index": point_index,
                "point_m": update.get("point_m"),
                "boundary_labels": labels,
                "boundary_class": bclass,
                "support_class": update.get("support_class"),
                "abs_pressure_delta_pa": abs_delta,
                "pressure_delta_pa": numeric_or_none(update.get("pressure_delta_pa")),
                "from_pressure_pa": numeric_or_none(update.get("from_pressure_pa")),
                "to_pressure_pa": numeric_or_none(update.get("to_pressure_pa")),
                "incident_wet_fraction_max": numeric_or_none(
                    update.get("incident_wet_fraction_max")
                ),
                "incident_wet_fraction_min_positive": numeric_or_none(
                    update.get("incident_wet_fraction_min_positive")
                ),
                "matched_matrix_support": row is not None,
                "coupling_class": cls,
                "pressure_self_class": pressure_cls,
                "local_pressure_row": row.get("local_pressure_row")
                if isinstance(row, dict)
                else None,
                "global_dof": matrix_sample.get("dof")
                if isinstance(matrix_sample, dict)
                else None,
                "row_abs_sum": numeric_or_none(matrix_sample.get("row_abs_sum"))
                if isinstance(matrix_sample, dict)
                else None,
                "diag": numeric_or_none(matrix_sample.get("diag"))
                if isinstance(matrix_sample, dict)
                else None,
                "row_velocity_abs_sum": row_velocity_numeric,
                "row_pressure_abs_sum": row_pressure_numeric,
                "pressure_to_velocity_row_abs_sum_ratio": pressure_to_velocity_ratio,
                "col_velocity_abs_sum": numeric_or_none(
                    col_sums.get("Velocity") if isinstance(col_sums, dict) else None
                ),
                "col_pressure_abs_sum": numeric_or_none(
                    col_sums.get("Pressure") if isinstance(col_sums, dict) else None
                ),
                "row_constrained_velocity_abs_sum": numeric_or_none(
                    row_constrained_sums.get("Velocity")
                    if isinstance(row_constrained_sums, dict)
                    else None
                ),
                "row_unconstrained_velocity_abs_sum": numeric_or_none(
                    row_unconstrained_sums.get("Velocity")
                    if isinstance(row_unconstrained_sums, dict)
                    else None
                ),
                "row_constrained_pressure_abs_sum": numeric_or_none(
                    row_constrained_sums.get("Pressure")
                    if isinstance(row_constrained_sums, dict)
                    else None
                ),
                "row_unconstrained_pressure_abs_sum": numeric_or_none(
                    row_unconstrained_sums.get("Pressure")
                    if isinstance(row_unconstrained_sums, dict)
                    else None
                ),
                "col_constrained_velocity_abs_sum": numeric_or_none(
                    col_constrained_sums.get("Velocity")
                    if isinstance(col_constrained_sums, dict)
                    else None
                ),
                "col_unconstrained_velocity_abs_sum": numeric_or_none(
                    col_unconstrained_sums.get("Velocity")
                    if isinstance(col_unconstrained_sums, dict)
                    else None
                ),
                "active_dof_support": constraint_sample.get("active_dof_support")
                if isinstance(constraint_sample, dict)
                else None,
                "inactive_constraint": constraint_sample.get("inactive_constraint")
                if isinstance(constraint_sample, dict)
                else None,
                "retained_measure": numeric_or_none(
                    constraint_sample.get("retained_measure")
                )
                if isinstance(constraint_sample, dict)
                else None,
                "retained_rule_count": constraint_sample.get("retained_rule_count")
                if isinstance(constraint_sample, dict)
                else None,
            }
        )

    latest_support_rank = matrix_support_report.get("latest_support_rank_diagnostic")
    support_rank_values = (
        latest_support_rank.get("values", {})
        if isinstance(latest_support_rank, dict)
        else {}
    )
    weakest_self_rows = support_rank_rows_with_context(
        details_value=support_rank_values.get("weakest_self_row_details")
        if isinstance(support_rank_values, dict)
        else None,
        matrix_support_report=matrix_support_report,
        points=points,
        fallback_points_by_index=fallback_points,
        bounds=bounds,
        boundary_tolerance=boundary_tolerance,
    )
    weakest_coupling_rows = support_rank_rows_with_context(
        details_value=support_rank_values.get("weakest_coupling_row_details")
        if isinstance(support_rank_values, dict)
        else None,
        matrix_support_report=matrix_support_report,
        points=points,
        fallback_points_by_index=fallback_points,
        bounds=bounds,
        boundary_tolerance=boundary_tolerance,
    )
    weak_pressure_self_updates = [
        row
        for row in correlated
        if row.get("pressure_self_class") == "weak_pressure_self"
    ]

    return {
        "pressure_update_json": pressure_update_report.get("current_result"),
        "matrix_support_log": matrix_support_report.get("solver_log"),
        "top_update_count": len(correlated),
        "matched_update_count": sum(
            1 for item in correlated if item["matched_matrix_support"]
        ),
        "unmatched_update_count": sum(
            1 for item in correlated if not item["matched_matrix_support"]
        ),
        "zero_tolerance": zero_tolerance,
        "weak_velocity_row_sum": weak_velocity_row_sum,
        "weak_pressure_row_sum": weak_pressure_row_sum,
        "boundary_tolerance": boundary_tolerance,
        "bounds": list(bounds) if bounds is not None else None,
        "bounds_source": bounds_source if bounds is not None else None,
        "coupling_class_counts": dict(sorted(class_counts.items())),
        "max_abs_delta_by_coupling_class": dict(sorted(class_max_delta.items())),
        "pressure_self_class_counts": dict(sorted(pressure_self_class_counts.items())),
        "max_abs_delta_by_pressure_self_class": dict(
            sorted(pressure_self_class_max_delta.items())
        ),
        "boundary_class_counts": dict(sorted(boundary_class_counts.items())),
        "boundary_label_counts": dict(sorted(boundary_label_counts.items())),
        "coupling_class_by_boundary_class_counts": dict(
            sorted(coupling_by_boundary_class.items())
        ),
        "pressure_self_class_by_boundary_class_counts": dict(
            sorted(pressure_self_by_boundary_class.items())
        ),
        "coupling_class_by_pressure_self_class_counts": dict(
            sorted(coupling_by_pressure_self_class.items())
        ),
        "weak_pressure_self_support_split_summary": support_split_summary(
            weak_pressure_self_updates,
            zero_tolerance=zero_tolerance,
        ),
        "support_rank_weakest_self_rows": weakest_self_rows,
        "support_rank_weakest_self_boundary_class_counts": boundary_counts_for_rows(
            weakest_self_rows
        ),
        "support_rank_weakest_self_support_split_summary": support_split_summary(
            weakest_self_rows,
            zero_tolerance=zero_tolerance,
        ),
        "support_rank_weakest_coupling_rows": weakest_coupling_rows,
        "support_rank_weakest_coupling_boundary_class_counts": boundary_counts_for_rows(
            weakest_coupling_rows
        ),
        "support_rank_weakest_coupling_support_split_summary": support_split_summary(
            weakest_coupling_rows,
            zero_tolerance=zero_tolerance,
        ),
        "top_updates": correlated,
    }


def main() -> int:
    args = parse_args()
    report = correlate_pressure_updates_with_support(
        pressure_update_report=load_json(args.pressure_update_json),
        matrix_support_report=load_json(args.matrix_support_json),
        top_events=args.top_events,
        zero_tolerance=args.zero_tolerance,
        weak_velocity_row_sum=args.weak_velocity_row_sum,
        weak_pressure_row_sum=args.weak_pressure_row_sum,
        boundary_tolerance=args.boundary_tolerance,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
