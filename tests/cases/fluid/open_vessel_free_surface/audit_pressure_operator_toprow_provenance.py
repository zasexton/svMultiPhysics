#!/usr/bin/env python3
"""Classify top pressure-update rows by exact operator support provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


OP_GALERKIN = "equations_diagnostic_ns_galerkin_continuity"
OP_NONPRESSURE = "equations_diagnostic_ns_vms_pspg_nonpressure"
OP_DIRECT_PGRAD = "equations_diagnostic_ns_vms_pspg_pressure_gradient"
OP_WALL_NORMAL_PGRAD = "equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient"
OP_WALL_TANGENTIAL_PGRAD = (
    "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient"
)
OP_GHOST = "equations_diagnostic_ns_pressure_ghost_penalty"
DIRECT_SUPPORT_LOW_RATIO_THRESHOLD = 0.25
DIRECT_SUPPORT_MODERATE_RATIO_THRESHOLD = 0.5

DEFAULT_OPERATORS = (
    OP_GALERKIN,
    OP_NONPRESSURE,
    OP_DIRECT_PGRAD,
    OP_WALL_NORMAL_PGRAD,
    OP_WALL_TANGENTIAL_PGRAD,
    OP_GHOST,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read support-audit JSONs containing exact "
            "pressure_row_operator_matrix_support samples and classify the "
            "logged top pressure-update rows by physical operator provenance."
        )
    )
    parser.add_argument(
        "--support-json",
        action="append",
        default=[],
        help="Support audit JSON as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--top-events", type=int, default=12)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--weak-velocity-row-sum", type=float, default=3.3e-4)
    parser.add_argument("--weak-pressure-row-sum", type=float, default=1.0e-7)
    parser.add_argument("--boundary-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--json-output", type=Path)
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


def numeric(value: Any) -> float | None:
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


def pressure_offset(report: dict[str, Any]) -> int | None:
    for key in (
        "latest_pressure_update_support_diagnostic",
        "latest_support_rank_diagnostic",
    ):
        offset = int_or_none(value_dict(report.get(key)).get("pressure_offset"))
        if offset is not None:
            return offset
    for row in report.get("pressure_row_operator_matrix_support_samples", []):
        if not isinstance(row, dict):
            continue
        support = row.get("operator_matrix_support")
        if isinstance(support, dict):
            offset = int_or_none(support.get("pressure_offset"))
            if offset is not None:
                return offset
    return None


def result_points_and_bounds(
    path_value: str | None,
) -> tuple[
    list[list[float]] | None,
    tuple[float, float, float, float, float, float] | None,
    list[int] | None,
]:
    if not path_value:
        return None, None, None
    path = Path(path_value)
    if not path.exists():
        return None, None, None
    try:
        import pyvista as pv

        grid = pv.read(path)
    except Exception:
        return None, None, None
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
    ), point_incident_cell_counts(grid)


def point_incident_cell_counts(grid: Any) -> list[int] | None:
    point_count = int(getattr(grid, "n_points", 0) or 0)
    if point_count <= 0 or not hasattr(grid, "cells"):
        return None
    cells = getattr(grid, "cells")
    counts = [0 for _ in range(point_count)]
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


def support_class(
    value: Any,
    *,
    zero_tolerance: float,
    weak_threshold: float,
) -> str:
    number = numeric(value)
    if number is None:
        return "missing"
    magnitude = abs(number)
    if magnitude <= zero_tolerance:
        return "zero"
    if magnitude <= weak_threshold:
        return "weak"
    return "positive"


def row_support_class(
    row: dict[str, Any],
    *,
    zero_tolerance: float,
    weak_velocity_row_sum: float,
    weak_pressure_row_sum: float,
) -> str:
    coupling = support_class(
        row.get("row_coupling"),
        zero_tolerance=zero_tolerance,
        weak_threshold=weak_velocity_row_sum,
    )
    self = support_class(
        row.get("row_self"),
        zero_tolerance=zero_tolerance,
        weak_threshold=weak_pressure_row_sum,
    )
    return f"{coupling}_coupling:{self}_self"


def top_update_rows(report: dict[str, Any], *, top_events: int) -> list[dict[str, Any]]:
    summary = report.get("pressure_update_support_summary")
    if not isinstance(summary, dict):
        return []
    rows = summary.get("top_update_details")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows[:top_events], start=1):
        if not isinstance(row, dict):
            continue
        dof = int_or_none(row.get("global_dof"))
        if dof is None:
            continue
        out.append(
            {
                "rank": rank,
                "global_dof": dof,
                "local_pressure_row": int_or_none(row.get("local_pressure_row")),
                "abs_update": numeric(row.get("abs_update")),
                "update": numeric(row.get("update")),
                "row_coupling": numeric(row.get("row_coupling")),
                "row_self": numeric(row.get("row_self")),
                "pressure_action_terms": (
                    row.get("pressure_action_terms")
                    if isinstance(row.get("pressure_action_terms"), str)
                    else None
                ),
                "coupling_action_terms": (
                    row.get("coupling_action_terms")
                    if isinstance(row.get("coupling_action_terms"), str)
                    else None
                ),
            }
        )
    return out


def exact_operator_support_by_dof(
    report: dict[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    by_dof: dict[int, dict[str, dict[str, Any]]] = {}
    best_line: dict[tuple[int, str], int] = {}
    for row in report.get("pressure_row_operator_matrix_support_samples", []):
        if not isinstance(row, dict):
            continue
        op = row.get("op")
        support = row.get("operator_matrix_support")
        if not isinstance(op, str) or not isinstance(support, dict):
            continue
        dof = int_or_none(support.get("dof"))
        if dof is None:
            continue
        line = int_or_none(row.get("line_number")) or -1
        key = (dof, op)
        if key in best_line and line < best_line[key]:
            continue
        best_line[key] = line
        by_dof.setdefault(dof, {})[op] = support
    return by_dof


def operator_support_summary(
    support: dict[str, Any] | None,
    *,
    zero_tolerance: float,
    weak_velocity_row_sum: float,
    weak_pressure_row_sum: float,
) -> dict[str, Any]:
    if not isinstance(support, dict):
        return {
            "status": "missing",
            "row_coupling_class": "missing",
            "row_self_class": "missing",
            "row_abs_class": "missing",
            "row_abs_sum": None,
            "row_coupling_abs_sum": None,
            "row_self_abs_sum": None,
            "row_numeric_entries": None,
            "row_self_numeric_entries": None,
            "row_self_offdiag_abs_sum": None,
            "row_self_diag_abs_ratio": None,
            "row_self_signed_abs_ratio": None,
            "row_first_nonzero": None,
            "col_first_nonzero": None,
            "diag": None,
        }
    row_abs = numeric(support.get("row_abs_sum"))
    return {
        "status": support.get("status", "ok"),
        "row_abs_sum": row_abs,
        "row_coupling_abs_sum": numeric(support.get("row_coupling_abs_sum")),
        "row_self_abs_sum": numeric(support.get("row_self_abs_sum")),
        "row_numeric_entries": int_or_none(support.get("row_numeric_entries")),
        "row_self_numeric_entries": int_or_none(
            support.get("row_self_numeric_entries")
        ),
        "row_self_offdiag_abs_sum": numeric(
            support.get("row_self_offdiag_abs_sum")
        ),
        "row_self_diag_abs_ratio": numeric(support.get("row_self_diag_abs_ratio")),
        "row_self_signed_abs_ratio": numeric(
            support.get("row_self_signed_abs_ratio")
        ),
        "row_first_nonzero": (
            support.get("row_first_nonzero")
            if isinstance(support.get("row_first_nonzero"), str)
            else None
        ),
        "col_first_nonzero": (
            support.get("col_first_nonzero")
            if isinstance(support.get("col_first_nonzero"), str)
            else None
        ),
        "diag": numeric(support.get("diag")),
        "row_abs_class": support_class(
            row_abs,
            zero_tolerance=zero_tolerance,
            weak_threshold=weak_pressure_row_sum,
        ),
        "row_coupling_class": support_class(
            support.get("row_coupling_abs_sum"),
            zero_tolerance=zero_tolerance,
            weak_threshold=weak_velocity_row_sum,
        ),
        "row_self_class": support_class(
            support.get("row_self_abs_sum"),
            zero_tolerance=zero_tolerance,
            weak_threshold=weak_pressure_row_sum,
        ),
    }


def parse_sparse_dof_list(value: Any) -> list[int]:
    if not isinstance(value, str) or not value or value == "none":
        return []
    out: list[int] = []
    for part in value.split("|"):
        head, separator, _tail = part.partition(":")
        if not separator:
            continue
        dof = int_or_none(head)
        if dof is not None:
            out.append(dof)
    return out


def parse_action_dof_list(value: Any) -> list[int]:
    if not isinstance(value, str) or not value:
        return []
    out: list[int] = []
    for part in value.split("~"):
        pieces = part.split("/")
        if len(pieces) < 2:
            continue
        dof = int_or_none(pieces[1])
        if dof is not None:
            out.append(dof)
    return out


def same_update_sign(left: Any, right: Any) -> bool:
    left_value = numeric(left)
    right_value = numeric(right)
    if left_value is None or right_value is None:
        return False
    if left_value == 0.0 or right_value == 0.0:
        return False
    return (left_value > 0.0) == (right_value > 0.0)


def physical_path_class(operator_support: dict[str, dict[str, Any]]) -> str:
    ghost = operator_support.get(OP_GHOST, {})
    direct = operator_support.get(OP_DIRECT_PGRAD, {})
    wall_normal = operator_support.get(OP_WALL_NORMAL_PGRAD, {})
    wall_tangential = operator_support.get(OP_WALL_TANGENTIAL_PGRAD, {})
    if ghost.get("row_self_class") == "positive":
        return "ghost_penalty_positive_self"
    if ghost.get("row_self_class") == "weak":
        return "ghost_penalty_weak_self"
    wall_classes = {
        wall_normal.get("row_self_class"),
        wall_tangential.get("row_self_class"),
    }
    if direct.get("row_self_class") in {"zero", "weak"} and (
        wall_classes & {"weak", "positive"}
    ):
        return "direct_pspg_weak_self_with_wall_support"
    if direct.get("row_self_class") in {"zero", "weak"}:
        return "direct_pspg_weak_self_no_wall_support"
    if direct.get("row_self_class") == "positive":
        return "direct_pspg_positive_self"
    return "operator_support_incomplete"


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, str):
            counts[value] += 1
    return dict(sorted(counts.items()))


def count_nested(
    rows: list[dict[str, Any]],
    op: str,
    key: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        op_support = row.get("operator_support")
        if not isinstance(op_support, dict):
            continue
        summary = op_support.get(op)
        if not isinstance(summary, dict):
            continue
        value = summary.get(key)
        if isinstance(value, str):
            counts[value] += 1
    return dict(sorted(counts.items()))


def ordered_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def dofs_by_physical_path(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        path_class = row.get("physical_path_class")
        dof = int_or_none(row.get("global_dof"))
        if not isinstance(path_class, str) or dof is None:
            continue
        grouped.setdefault(path_class, []).append(dof)
    return {
        key: ordered_unique(values)
        for key, values in sorted(grouped.items())
    }


def direct_pspg_dofs(rows: list[dict[str, Any]]) -> list[int]:
    return ordered_unique(
        [
            int(row["global_dof"])
            for row in rows
            if row.get("physical_path_class")
            in {
                "direct_pspg_weak_self_with_wall_support",
                "direct_pspg_weak_self_no_wall_support",
                "direct_pspg_positive_self",
            }
        ]
    )


def direct_pspg_one_cell_boundary_dofs(rows: list[dict[str, Any]]) -> list[int]:
    return ordered_unique(
        [
            int(row["global_dof"])
            for row in rows
            if row.get("physical_path_class")
            in {
                "direct_pspg_weak_self_with_wall_support",
                "direct_pspg_weak_self_no_wall_support",
                "direct_pspg_positive_self",
            }
            and row.get("incident_support_class") == "one_cell_boundary_support"
        ]
    )


def is_direct_pspg_row(row: dict[str, Any]) -> bool:
    return row.get("physical_path_class") in {
        "direct_pspg_weak_self_with_wall_support",
        "direct_pspg_weak_self_no_wall_support",
        "direct_pspg_positive_self",
    }


def op_self(row: dict[str, Any], op: str) -> float:
    op_support = row.get("operator_support")
    if not isinstance(op_support, dict):
        return 0.0
    summary = op_support.get(op)
    if not isinstance(summary, dict):
        return 0.0
    value = numeric(summary.get("row_self_abs_sum"))
    return float(value) if value is not None else 0.0


def op_int(row: dict[str, Any], op: str, key: str) -> int | None:
    op_support = row.get("operator_support")
    if not isinstance(op_support, dict):
        return None
    summary = op_support.get(op)
    if not isinstance(summary, dict):
        return None
    return int_or_none(summary.get(key))


def op_class(row: dict[str, Any], op: str, key: str) -> str:
    op_support = row.get("operator_support")
    if not isinstance(op_support, dict):
        return "missing"
    summary = op_support.get(op)
    if not isinstance(summary, dict):
        return "missing"
    value = summary.get(key)
    return value if isinstance(value, str) else "missing"


def add_pspg_support_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_rows = [row for row in rows if is_direct_pspg_row(row)]
    direct_values = [op_self(row, OP_DIRECT_PGRAD) for row in direct_rows]
    total_values = [
        op_self(row, OP_DIRECT_PGRAD)
        + op_self(row, OP_WALL_NORMAL_PGRAD)
        + op_self(row, OP_WALL_TANGENTIAL_PGRAD)
        for row in direct_rows
    ]
    max_direct = max(direct_values, default=0.0)
    max_total = max(total_values, default=0.0)

    low_direct_dofs: list[int] = []
    moderate_direct_dofs: list[int] = []
    low_total_dofs: list[int] = []
    moderate_total_dofs: list[int] = []
    for row in rows:
        direct_self = op_self(row, OP_DIRECT_PGRAD)
        wall_normal_self = op_self(row, OP_WALL_NORMAL_PGRAD)
        wall_tangential_self = op_self(row, OP_WALL_TANGENTIAL_PGRAD)
        total_self = direct_self + wall_normal_self + wall_tangential_self
        direct_ratio = (
            direct_self / max_direct if max_direct > 0.0 and is_direct_pspg_row(row) else None
        )
        total_ratio = (
            total_self / max_total if max_total > 0.0 and is_direct_pspg_row(row) else None
        )
        row["pspg_pressure_gradient_support_profile"] = {
            "direct_self_abs_sum": float(direct_self),
            "wall_normal_self_abs_sum": float(wall_normal_self),
            "wall_tangential_self_abs_sum": float(wall_tangential_self),
            "total_pressure_gradient_self_abs_sum": float(total_self),
            "direct_self_to_case_direct_max_ratio": (
                float(direct_ratio) if direct_ratio is not None else None
            ),
            "total_self_to_case_direct_max_ratio": (
                float(total_ratio) if total_ratio is not None else None
            ),
        }
        if not is_direct_pspg_row(row):
            continue
        dof = int(row["global_dof"])
        if direct_ratio is not None:
            if direct_ratio <= DIRECT_SUPPORT_LOW_RATIO_THRESHOLD:
                low_direct_dofs.append(dof)
            if direct_ratio <= DIRECT_SUPPORT_MODERATE_RATIO_THRESHOLD:
                moderate_direct_dofs.append(dof)
        if total_ratio is not None:
            if total_ratio <= DIRECT_SUPPORT_LOW_RATIO_THRESHOLD:
                low_total_dofs.append(dof)
            if total_ratio <= DIRECT_SUPPORT_MODERATE_RATIO_THRESHOLD:
                moderate_total_dofs.append(dof)

    return {
        "direct_pspg_case_max_direct_self_abs_sum": (
            float(max_direct) if max_direct > 0.0 else None
        ),
        "direct_pspg_case_max_total_pressure_gradient_self_abs_sum": (
            float(max_total) if max_total > 0.0 else None
        ),
        "direct_pspg_low_direct_self_ratio_threshold": (
            DIRECT_SUPPORT_LOW_RATIO_THRESHOLD
        ),
        "direct_pspg_moderate_direct_self_ratio_threshold": (
            DIRECT_SUPPORT_MODERATE_RATIO_THRESHOLD
        ),
        "direct_pspg_low_direct_self_ratio_global_dofs": low_direct_dofs,
        "direct_pspg_moderate_direct_self_ratio_global_dofs": moderate_direct_dofs,
        "direct_pspg_low_total_self_ratio_global_dofs": low_total_dofs,
        "direct_pspg_moderate_total_self_ratio_global_dofs": moderate_total_dofs,
    }


def add_pspg_support_topology_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_rows = [row for row in rows if is_direct_pspg_row(row)]
    direct_entry_values = [
        value
        for value in (
            op_int(row, OP_DIRECT_PGRAD, "row_self_numeric_entries")
            for row in direct_rows
        )
        if value is not None
    ]
    min_direct_entries = min(direct_entry_values, default=None)
    max_direct_entries = max(direct_entry_values, default=None)

    sparse_direct_dofs: list[int] = []
    missing_wall_normal_dofs: list[int] = []
    missing_wall_tangential_dofs: list[int] = []
    zero_galerkin_nonpressure_dofs: list[int] = []
    for row in rows:
        direct_entries = op_int(row, OP_DIRECT_PGRAD, "row_self_numeric_entries")
        wall_normal_entries = op_int(
            row, OP_WALL_NORMAL_PGRAD, "row_self_numeric_entries"
        )
        wall_tangential_entries = op_int(
            row, OP_WALL_TANGENTIAL_PGRAD, "row_self_numeric_entries"
        )
        direct_row_entries = op_int(row, OP_DIRECT_PGRAD, "row_numeric_entries")
        total_entry_sum = sum(
            value or 0
            for value in (
                direct_entries,
                wall_normal_entries,
                wall_tangential_entries,
            )
        )
        is_direct = is_direct_pspg_row(row)
        sparse_direct = (
            is_direct
            and direct_entries is not None
            and max_direct_entries is not None
            and direct_entries < max_direct_entries
        )
        missing_wall_normal = (
            is_direct
            and (
                (wall_normal_entries is not None and wall_normal_entries <= 0)
                or op_class(row, OP_WALL_NORMAL_PGRAD, "row_self_class") == "zero"
            )
        )
        missing_wall_tangential = (
            is_direct
            and (
                (wall_tangential_entries is not None and wall_tangential_entries <= 0)
                or op_class(row, OP_WALL_TANGENTIAL_PGRAD, "row_self_class")
                == "zero"
            )
        )
        zero_galerkin_nonpressure = (
            is_direct
            and op_class(row, OP_GALERKIN, "row_coupling_class") == "zero"
            and op_class(row, OP_NONPRESSURE, "row_coupling_class") == "zero"
        )
        row["pspg_pressure_gradient_support_topology_profile"] = {
            "direct_self_numeric_entries": direct_entries,
            "direct_row_numeric_entries": direct_row_entries,
            "wall_normal_self_numeric_entries": wall_normal_entries,
            "wall_tangential_self_numeric_entries": wall_tangential_entries,
            "summed_pressure_gradient_self_numeric_entries": total_entry_sum,
            "case_min_direct_self_numeric_entries": min_direct_entries,
            "case_max_direct_self_numeric_entries": max_direct_entries,
            "sparse_direct_self_entries": bool(sparse_direct),
            "missing_wall_normal_self_support": bool(missing_wall_normal),
            "missing_wall_tangential_self_support": bool(missing_wall_tangential),
            "zero_galerkin_and_nonpressure_coupling": bool(
                zero_galerkin_nonpressure
            ),
        }
        if not is_direct:
            continue
        dof = int(row["global_dof"])
        if sparse_direct:
            sparse_direct_dofs.append(dof)
        if missing_wall_normal:
            missing_wall_normal_dofs.append(dof)
        if missing_wall_tangential:
            missing_wall_tangential_dofs.append(dof)
        if zero_galerkin_nonpressure:
            zero_galerkin_nonpressure_dofs.append(dof)

    return {
        "direct_pspg_case_min_direct_self_numeric_entries": min_direct_entries,
        "direct_pspg_case_max_direct_self_numeric_entries": max_direct_entries,
        "direct_pspg_sparse_direct_self_entry_global_dofs": sparse_direct_dofs,
        "direct_pspg_missing_wall_normal_self_global_dofs": missing_wall_normal_dofs,
        "direct_pspg_missing_wall_tangential_self_global_dofs": (
            missing_wall_tangential_dofs
        ),
        "direct_pspg_zero_galerkin_nonpressure_coupling_global_dofs": (
            zero_galerkin_nonpressure_dofs
        ),
    }


def add_direct_patch_neighbor_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_dofs = {
        int(row["global_dof"])
        for row in rows
        if int_or_none(row.get("global_dof")) is not None
    }
    direct_dofs = {
        int(row["global_dof"])
        for row in rows
        if int_or_none(row.get("global_dof")) is not None and is_direct_pspg_row(row)
    }
    update_by_dof = {
        int(row["global_dof"]): row.get("update")
        for row in rows
        if int_or_none(row.get("global_dof")) is not None
    }
    rows_with_direct_top_neighbors: list[int] = []
    rows_with_direct_direct_neighbors: list[int] = []
    rows_with_action_top_neighbors: list[int] = []
    rows_with_same_sign_action_top_neighbors: list[int] = []
    direct_top_edges: set[tuple[int, int]] = set()
    action_top_edges: set[tuple[int, int]] = set()
    same_sign_action_edges: set[tuple[int, int]] = set()

    for row in rows:
        dof = int_or_none(row.get("global_dof"))
        op_support = row.get("operator_support")
        direct_support = (
            op_support.get(OP_DIRECT_PGRAD)
            if isinstance(op_support, dict)
            and isinstance(op_support.get(OP_DIRECT_PGRAD), dict)
            else {}
        )
        direct_row_neighbors = parse_sparse_dof_list(
            direct_support.get("row_first_nonzero")
        )
        pressure_action_neighbors = parse_action_dof_list(
            row.get("pressure_action_terms")
        )
        direct_top_neighbors = sorted(
            {
                neighbor
                for neighbor in direct_row_neighbors
                if dof is not None and neighbor != dof and neighbor in top_dofs
            }
        )
        direct_direct_neighbors = sorted(
            neighbor for neighbor in direct_top_neighbors if neighbor in direct_dofs
        )
        action_top_neighbors = sorted(
            {
                neighbor
                for neighbor in pressure_action_neighbors
                if dof is not None and neighbor != dof and neighbor in top_dofs
            }
        )
        same_sign_action_neighbors = sorted(
            {
                neighbor
                for neighbor in action_top_neighbors
                if same_update_sign(row.get("update"), update_by_dof.get(neighbor))
            }
        )
        row["direct_pspg_patch_neighbor_profile"] = {
            "direct_pgrad_row_neighbor_dofs": direct_row_neighbors,
            "direct_pgrad_top_update_neighbor_dofs": direct_top_neighbors,
            "direct_pgrad_direct_pspg_top_neighbor_dofs": direct_direct_neighbors,
            "pressure_action_neighbor_dofs": pressure_action_neighbors,
            "pressure_action_top_update_neighbor_dofs": action_top_neighbors,
            "same_sign_pressure_action_top_update_neighbor_dofs": (
                same_sign_action_neighbors
            ),
            "direct_pgrad_top_update_neighbor_count": len(direct_top_neighbors),
            "direct_pgrad_direct_pspg_top_neighbor_count": len(
                direct_direct_neighbors
            ),
            "pressure_action_top_update_neighbor_count": len(action_top_neighbors),
            "same_sign_pressure_action_top_update_neighbor_count": len(
                same_sign_action_neighbors
            ),
        }
        if dof is None or not is_direct_pspg_row(row):
            continue
        if direct_top_neighbors:
            rows_with_direct_top_neighbors.append(dof)
        if direct_direct_neighbors:
            rows_with_direct_direct_neighbors.append(dof)
        if action_top_neighbors:
            rows_with_action_top_neighbors.append(dof)
        if same_sign_action_neighbors:
            rows_with_same_sign_action_top_neighbors.append(dof)
        for neighbor in direct_top_neighbors:
            direct_top_edges.add(tuple(sorted((dof, neighbor))))
        for neighbor in action_top_neighbors:
            action_top_edges.add(tuple(sorted((dof, neighbor))))
        for neighbor in same_sign_action_neighbors:
            same_sign_action_edges.add(tuple(sorted((dof, neighbor))))

    return {
        "direct_pspg_rows_with_direct_pgrad_top_neighbors_global_dofs": (
            rows_with_direct_top_neighbors
        ),
        "direct_pspg_rows_with_direct_pgrad_direct_top_neighbors_global_dofs": (
            rows_with_direct_direct_neighbors
        ),
        "direct_pspg_rows_with_pressure_action_top_neighbors_global_dofs": (
            rows_with_action_top_neighbors
        ),
        "direct_pspg_rows_with_same_sign_pressure_action_top_neighbors_global_dofs": (
            rows_with_same_sign_action_top_neighbors
        ),
        "direct_pspg_direct_pgrad_top_neighbor_edge_count": len(direct_top_edges),
        "direct_pspg_pressure_action_top_neighbor_edge_count": len(action_top_edges),
        "direct_pspg_same_sign_pressure_action_top_neighbor_edge_count": len(
            same_sign_action_edges
        ),
        **same_sign_action_component_summary(rows, same_sign_action_edges),
    }


def connected_components(edges: set[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        pending = [start]
        component: list[int] = []
        seen.add(start)
        while pending:
            dof = pending.pop()
            component.append(dof)
            for neighbor in sorted(adjacency.get(dof, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                pending.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values[0]))


def same_sign_action_component_summary(
    rows: list[dict[str, Any]],
    edges: set[tuple[int, int]],
) -> dict[str, Any]:
    row_by_dof = {
        int(row["global_dof"]): row
        for row in rows
        if int_or_none(row.get("global_dof")) is not None
    }
    direct_dofs = {
        dof for dof, row in row_by_dof.items() if is_direct_pspg_row(row)
    }
    components = connected_components(edges)
    component_details: list[dict[str, Any]] = []
    covered_direct_dofs: set[int] = set()

    for index, component in enumerate(components, start=1):
        component_rows = [row_by_dof[dof] for dof in component if dof in row_by_dof]
        component_direct_dofs = [
            int(row["global_dof"]) for row in component_rows if is_direct_pspg_row(row)
        ]
        component_ghost_dofs = [
            int(row["global_dof"])
            for row in component_rows
            if row.get("physical_path_class")
            in {"ghost_penalty_positive_self", "ghost_penalty_weak_self"}
        ]
        covered_direct_dofs.update(component_direct_dofs)
        max_row = max(
            component_rows,
            key=lambda row: numeric(row.get("abs_update")) or -1.0,
            default=None,
        )
        component_edge_count = sum(
            1 for left, right in edges if left in component and right in component
        )
        component_details.append(
            {
                "component_index": index,
                "size": len(component),
                "global_dofs": component,
                "direct_pspg_global_dofs": component_direct_dofs,
                "ghost_penalty_global_dofs": component_ghost_dofs,
                "rank_values": [
                    int(row["rank"])
                    for row in component_rows
                    if int_or_none(row.get("rank")) is not None
                ],
                "contains_rank1": any(row.get("rank") == 1 for row in component_rows),
                "max_abs_update": (
                    numeric(max_row.get("abs_update"))
                    if isinstance(max_row, dict)
                    else None
                ),
                "max_abs_update_global_dof": (
                    int_or_none(max_row.get("global_dof"))
                    if isinstance(max_row, dict)
                    else None
                ),
                "boundary_class_counts": count_by(
                    component_rows, "boundary_class"
                ),
                "incident_support_class_counts": count_by(
                    component_rows, "incident_support_class"
                ),
                "same_sign_pressure_action_edge_count": component_edge_count,
            }
        )

    isolated_direct_dofs = sorted(direct_dofs - covered_direct_dofs)
    if not direct_dofs:
        finding = "direct_pspg_top_rows_absent"
    elif not components:
        finding = "direct_pspg_top_rows_same_sign_action_isolated"
    elif not isolated_direct_dofs and len(components) == 1:
        finding = "direct_pspg_top_rows_single_same_sign_action_patch"
    elif not isolated_direct_dofs:
        finding = "direct_pspg_top_rows_same_sign_action_patch_covered"
    else:
        finding = "direct_pspg_top_rows_partial_same_sign_action_patch_coverage"

    return {
        "direct_pspg_same_sign_pressure_action_patch_finding": finding,
        "direct_pspg_same_sign_pressure_action_component_count": len(components),
        "direct_pspg_same_sign_pressure_action_largest_component_size": (
            max((len(component) for component in components), default=0)
        ),
        "direct_pspg_same_sign_pressure_action_direct_coverage_count": len(
            covered_direct_dofs
        ),
        "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": (
            sorted(covered_direct_dofs)
        ),
        "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs": (
            isolated_direct_dofs
        ),
        "direct_pspg_same_sign_pressure_action_components": component_details,
    }


def ghost_penalty_dofs(rows: list[dict[str, Any]]) -> list[int]:
    return ordered_unique(
        [
            int(row["global_dof"])
            for row in rows
            if row.get("physical_path_class")
            in {
                "ghost_penalty_positive_self",
                "ghost_penalty_weak_self",
            }
        ]
    )


def classify_case_finding(rows: list[dict[str, Any]]) -> str:
    path_counts = Counter(str(row.get("physical_path_class")) for row in rows)
    ghost_count = sum(
        path_counts[path]
        for path in (
            "ghost_penalty_positive_self",
            "ghost_penalty_weak_self",
        )
    )
    pspg_count = sum(
        path_counts[path]
        for path in (
            "direct_pspg_weak_self_with_wall_support",
            "direct_pspg_weak_self_no_wall_support",
            "direct_pspg_positive_self",
        )
    )
    if not rows:
        return "top_update_rows_missing"
    if ghost_count and pspg_count:
        return "mixed_direct_pspg_and_ghost_penalty_top_rows"
    if ghost_count:
        return "ghost_penalty_top_rows"
    if pspg_count:
        return "direct_pspg_top_rows_without_ghost_penalty"
    return "operator_support_incomplete"


def case_base_label(label: Any) -> tuple[str | None, str | None]:
    if not isinstance(label, str) or not label:
        return None, None
    if label.endswith("_pressure_disabled"):
        return label[: -len("_pressure_disabled")], "pressure_disabled"
    return label, "full_gradient"


def audit_case(
    label: str,
    path: Path,
    report: dict[str, Any],
    *,
    top_events: int = 12,
    zero_tolerance: float = 1.0e-14,
    weak_velocity_row_sum: float = 3.3e-4,
    weak_pressure_row_sum: float = 1.0e-7,
    boundary_tolerance: float = 1.0e-10,
    points: list[list[float]] | None = None,
    bounds: tuple[float, float, float, float, float, float] | None = None,
    incident_cell_counts: list[int] | None = None,
) -> dict[str, Any]:
    source_result = source_result_from_manifest(report)
    if points is None or bounds is None or incident_cell_counts is None:
        loaded_points, loaded_bounds, loaded_incident_counts = result_points_and_bounds(
            source_result
        )
        points = points if points is not None else loaded_points
        bounds = bounds if bounds is not None else loaded_bounds
        incident_cell_counts = (
            incident_cell_counts
            if incident_cell_counts is not None
            else loaded_incident_counts
        )

    offset = pressure_offset(report)
    samples_by_dof = exact_operator_support_by_dof(report)
    rows = top_update_rows(report, top_events=top_events)

    for row in rows:
        dof = int(row["global_dof"])
        point_index = dof - offset if offset is not None else None
        point = (
            points[point_index]
            if isinstance(point_index, int)
            and points is not None
            and 0 <= point_index < len(points)
            else None
        )
        labels = boundary_labels(point, bounds, tolerance=boundary_tolerance)
        boundary_class_value = boundary_class(labels)
        incident_count = (
            int(incident_cell_counts[point_index])
            if isinstance(point_index, int)
            and incident_cell_counts is not None
            and 0 <= point_index < len(incident_cell_counts)
            else None
        )
        operator_support = {
            op: operator_support_summary(
                samples_by_dof.get(dof, {}).get(op),
                zero_tolerance=zero_tolerance,
                weak_velocity_row_sum=weak_velocity_row_sum,
                weak_pressure_row_sum=weak_pressure_row_sum,
            )
            for op in DEFAULT_OPERATORS
        }
        row.update(
            {
                "point_index": point_index,
                "point_m": point,
                "boundary_labels": labels,
                "boundary_class": boundary_class_value,
                "incident_cell_count": incident_count,
                "incident_support_class": incident_support_class(
                    boundary_class_value=boundary_class_value,
                    incident_cell_count=incident_count,
                ),
                "total_row_support_class": row_support_class(
                    row,
                    zero_tolerance=zero_tolerance,
                    weak_velocity_row_sum=weak_velocity_row_sum,
                    weak_pressure_row_sum=weak_pressure_row_sum,
                ),
                "operator_support": operator_support,
                "physical_path_class": physical_path_class(operator_support),
            }
        )

    exact_sampled_top_rows = sum(
        1 for row in rows if int(row["global_dof"]) in samples_by_dof
    )
    pspg_support_profile_summary = add_pspg_support_profiles(rows)
    pspg_support_topology_summary = add_pspg_support_topology_profiles(rows)
    direct_patch_neighbor_summary = add_direct_patch_neighbor_profiles(rows)
    finding = classify_case_finding(rows)
    direct_dofs = direct_pspg_dofs(rows)
    direct_one_cell_boundary_dofs = direct_pspg_one_cell_boundary_dofs(rows)
    ghost_dofs = ghost_penalty_dofs(rows)
    return {
        "label": label,
        "support_json": str(path),
        "finding": finding,
        "source_result": source_result,
        "source_result_loaded": points is not None and bounds is not None,
        "source_result_incident_support_loaded": incident_cell_counts is not None,
        "pressure_offset": offset,
        "top_update_count": len(rows),
        "exact_operator_sampled_top_row_count": exact_sampled_top_rows,
        "boundary_class_counts": count_by(rows, "boundary_class"),
        "incident_support_class_counts": count_by(rows, "incident_support_class"),
        "total_row_support_class_counts": count_by(rows, "total_row_support_class"),
        "physical_path_class_counts": count_by(rows, "physical_path_class"),
        "galerkin_coupling_class_counts": count_nested(
            rows, OP_GALERKIN, "row_coupling_class"
        ),
        "nonpressure_coupling_class_counts": count_nested(
            rows, OP_NONPRESSURE, "row_coupling_class"
        ),
        "direct_pgrad_self_class_counts": count_nested(
            rows, OP_DIRECT_PGRAD, "row_self_class"
        ),
        "wall_normal_pgrad_self_class_counts": count_nested(
            rows, OP_WALL_NORMAL_PGRAD, "row_self_class"
        ),
        "wall_tangential_pgrad_self_class_counts": count_nested(
            rows, OP_WALL_TANGENTIAL_PGRAD, "row_self_class"
        ),
        "ghost_penalty_self_class_counts": count_nested(
            rows, OP_GHOST, "row_self_class"
        ),
        "balance_global_dofs_by_physical_path": dofs_by_physical_path(rows),
        "direct_pspg_balance_global_dofs": direct_dofs,
        "direct_pspg_one_cell_boundary_global_dofs": direct_one_cell_boundary_dofs,
        "direct_pspg_one_cell_boundary_count": len(direct_one_cell_boundary_dofs),
        "direct_pspg_non_one_cell_boundary_global_dofs": [
            dof for dof in direct_dofs if dof not in set(direct_one_cell_boundary_dofs)
        ],
        **pspg_support_profile_summary,
        **pspg_support_topology_summary,
        **direct_patch_neighbor_summary,
        "ghost_penalty_balance_global_dofs": ghost_dofs,
        "operator_top_row_balance_global_dofs": ordered_unique(
            direct_dofs + ghost_dofs
        ),
        "top_update_rows": rows,
    }


def cross_policy_neighbor_comparisons(
    cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        base, variant = case_base_label(case.get("label"))
        if base is None or variant is None:
            continue
        by_base.setdefault(base, {})[variant] = case

    comparisons: list[dict[str, Any]] = []
    for base, variants in sorted(by_base.items()):
        full = variants.get("full_gradient")
        pressure_disabled = variants.get("pressure_disabled")
        if full is None or pressure_disabled is None:
            continue

        disabled_direct_dofs = set(
            int(dof)
            for dof in pressure_disabled.get("direct_pspg_balance_global_dofs", [])
            if int_or_none(dof) is not None
        )
        disabled_updates = {
            int(row["global_dof"]): row.get("update")
            for row in pressure_disabled.get("top_update_rows", [])
            if int_or_none(row.get("global_dof")) is not None
        }
        rows_with_action_overlap: list[int] = []
        rows_with_direct_overlap: list[int] = []
        rows_with_same_sign_action_overlap: list[int] = []
        isolated_current_top_but_disabled_action_overlap: list[int] = []
        isolated_cross_policy_patch_dofs: set[int] = set()
        action_edges: set[tuple[int, int]] = set()
        direct_edges: set[tuple[int, int]] = set()
        same_sign_action_edges: set[tuple[int, int]] = set()
        per_row: list[dict[str, Any]] = []

        for row in full.get("top_update_rows", []):
            if not is_direct_pspg_row(row):
                continue
            dof = int_or_none(row.get("global_dof"))
            if dof is None:
                continue
            profile = row.get("direct_pspg_patch_neighbor_profile")
            if not isinstance(profile, dict):
                profile = {}
            action_neighbors = [
                neighbor
                for neighbor in parse_action_dof_list(row.get("pressure_action_terms"))
                if neighbor != dof and neighbor in disabled_direct_dofs
            ]
            direct_neighbors = [
                neighbor
                for neighbor in parse_sparse_dof_list(
                    profile.get("direct_pgrad_row_neighbor_dofs")
                )
                if neighbor != dof and neighbor in disabled_direct_dofs
            ]
            # Regenerated artifacts store this field as a list, while raw parsing
            # above accepts the original sparse string representation.
            if isinstance(profile.get("direct_pgrad_row_neighbor_dofs"), list):
                direct_neighbors = [
                    int(neighbor)
                    for neighbor in profile.get("direct_pgrad_row_neighbor_dofs", [])
                    if int_or_none(neighbor) is not None
                    and int(neighbor) != dof
                    and int(neighbor) in disabled_direct_dofs
                ]
            same_sign_neighbors = [
                neighbor
                for neighbor in action_neighbors
                if same_update_sign(row.get("update"), disabled_updates.get(neighbor))
            ]
            current_action_top_count = int_or_none(
                profile.get("pressure_action_top_update_neighbor_count")
            )
            current_direct_top_count = int_or_none(
                profile.get("direct_pgrad_top_update_neighbor_count")
            )
            is_current_top_isolated = (
                (current_action_top_count is None or current_action_top_count <= 0)
                and (current_direct_top_count is None or current_direct_top_count <= 0)
            )
            if action_neighbors:
                rows_with_action_overlap.append(dof)
            if direct_neighbors:
                rows_with_direct_overlap.append(dof)
            if same_sign_neighbors:
                rows_with_same_sign_action_overlap.append(dof)
            isolated_row_patch_dofs: list[int] = []
            if is_current_top_isolated and same_sign_neighbors:
                isolated_current_top_but_disabled_action_overlap.append(dof)
                isolated_row_patch_dofs = sorted({dof, *same_sign_neighbors})
                isolated_cross_policy_patch_dofs.update(isolated_row_patch_dofs)
            for neighbor in action_neighbors:
                action_edges.add((dof, neighbor))
            for neighbor in direct_neighbors:
                direct_edges.add((dof, neighbor))
            for neighbor in same_sign_neighbors:
                same_sign_action_edges.add((dof, neighbor))
            per_row.append(
                {
                    "global_dof": dof,
                    "rank": row.get("rank"),
                    "abs_update": row.get("abs_update"),
                    "current_top_action_neighbor_count": current_action_top_count,
                    "current_top_direct_neighbor_count": current_direct_top_count,
                    "pressure_disabled_direct_action_neighbor_dofs": sorted(
                        set(action_neighbors)
                    ),
                    "pressure_disabled_direct_row_neighbor_dofs": sorted(
                        set(direct_neighbors)
                    ),
                    "same_sign_pressure_disabled_direct_action_neighbor_dofs": sorted(
                        set(same_sign_neighbors)
                    ),
                    "current_top_isolated_cross_policy_patch_global_dofs": (
                        isolated_row_patch_dofs
                    ),
                    "current_top_isolated_but_pressure_disabled_direct_connected": (
                        bool(is_current_top_isolated and same_sign_neighbors)
                    ),
                }
            )

        isolated_patch_dofs = sorted(isolated_cross_policy_patch_dofs)
        comparisons.append(
            {
                "base_label": base,
                "full_gradient_label": full.get("label"),
                "pressure_disabled_label": pressure_disabled.get("label"),
                "full_gradient_direct_row_count": len(
                    full.get("direct_pspg_balance_global_dofs", [])
                ),
                "pressure_disabled_direct_row_count": len(disabled_direct_dofs),
                "full_direct_rows_with_pressure_disabled_direct_action_neighbors_global_dofs": (
                    rows_with_action_overlap
                ),
                "full_direct_rows_with_pressure_disabled_direct_row_neighbors_global_dofs": (
                    rows_with_direct_overlap
                ),
                "full_direct_rows_with_same_sign_pressure_disabled_direct_action_neighbors_global_dofs": (
                    rows_with_same_sign_action_overlap
                ),
                "full_direct_rows_current_top_isolated_but_pressure_disabled_direct_connected_global_dofs": (
                    isolated_current_top_but_disabled_action_overlap
                ),
                "current_top_isolated_cross_policy_patch_global_dofs": (
                    isolated_patch_dofs
                ),
                "current_top_isolated_cross_policy_patch_env_value": ",".join(
                    str(dof) for dof in isolated_patch_dofs
                ),
                "pressure_disabled_direct_action_neighbor_edge_count": len(
                    action_edges
                ),
                "pressure_disabled_direct_row_neighbor_edge_count": len(direct_edges),
                "same_sign_pressure_disabled_direct_action_neighbor_edge_count": len(
                    same_sign_action_edges
                ),
                "rows": per_row,
            }
        )
    return comparisons


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    finding_counts = Counter(str(case.get("finding")) for case in cases)
    has_mixed = any(
        case.get("finding") == "mixed_direct_pspg_and_ghost_penalty_top_rows"
        for case in cases
    )
    has_direct = any(
        case.get("finding") == "direct_pspg_top_rows_without_ghost_penalty"
        for case in cases
    )
    has_ghost = any(case.get("finding") == "ghost_penalty_top_rows" for case in cases)
    if has_mixed and has_direct:
        finding = "top_rows_split_between_direct_pspg_and_ghost_penalty_paths"
    elif has_direct and not has_ghost:
        finding = "top_rows_direct_pspg_without_ghost_penalty"
    elif has_ghost and not has_direct:
        finding = "top_rows_include_ghost_penalty_path"
    else:
        finding = "operator_toprow_provenance_incomplete"
    return {
        "finding": finding,
        "case_count": len(cases),
        "finding_counts": dict(sorted(finding_counts.items())),
        "cross_policy_neighbor_comparisons": cross_policy_neighbor_comparisons(cases),
        "cases": cases,
    }


def build_report(
    labeled_reports: list[tuple[str, Path, dict[str, Any]]],
    *,
    top_events: int = 12,
    zero_tolerance: float = 1.0e-14,
    weak_velocity_row_sum: float = 3.3e-4,
    weak_pressure_row_sum: float = 1.0e-7,
    boundary_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    cases = [
        audit_case(
            label,
            path,
            report,
            top_events=top_events,
            zero_tolerance=zero_tolerance,
            weak_velocity_row_sum=weak_velocity_row_sum,
            weak_pressure_row_sum=weak_pressure_row_sum,
            boundary_tolerance=boundary_tolerance,
        )
        for label, path, report in labeled_reports
    ]
    return summarize_cases(cases)


def main() -> int:
    args = parse_args()
    labeled_reports = [
        (label, path, load_json(path))
        for label, path in (parse_labeled_path(value) for value in args.support_json)
    ]
    report = build_report(
        labeled_reports,
        top_events=args.top_events,
        zero_tolerance=args.zero_tolerance,
        weak_velocity_row_sum=args.weak_velocity_row_sum,
        weak_pressure_row_sum=args.weak_pressure_row_sum,
        boundary_tolerance=args.boundary_tolerance,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
