#!/usr/bin/env python3
"""Classify pressure rows reported by Eigen factorization diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pressure_update_guard import (  # noqa: E402
    finite_float,
    parse_key_values,
    point_wet_support,
    support_class,
)


BLOCK_SUMMARY_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{(?P<body>[^}]*)\}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an Eigen direct-factorization diagnostic and classify logged "
            "zero pressure rows against a saved VTU/PVTU support state."
        )
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--field-name", default="Pressure")
    parser.add_argument(
        "--diagnostic-index",
        type=int,
        default=-1,
        help="0-based diagnostic index to inspect; negative values count from the end.",
    )
    parser.add_argument("--active-fluid-threshold", type=float, default=0.5)
    parser.add_argument("--tiny-wet-fraction", type=float, default=1.0e-4)
    parser.add_argument("--full-wet-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--point-index-offset",
        type=int,
        default=0,
        help=(
            "Offset from field-local scalar row to point index. The Test02/Test10 "
            "P1 pressure fields use the default zero offset."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def expand_index_runs(raw: Any) -> list[int]:
    if not isinstance(raw, str):
        return []
    value = raw.strip()
    if not value or value == "none":
        return []

    indices: list[int] = []
    for item in value.split("|"):
        token = item.strip()
        if not token or token == "none":
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(token))

    seen: set[int] = set()
    unique: list[int] = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        unique.append(index)
    return unique


def factorization_header(line: str) -> str:
    return line.split(" block_summaries=", 1)[0]


def parse_factorization_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "Eigen direct factorization diagnostic" not in line:
        return None

    header = parse_key_values(factorization_header(line))
    block_match = re.search(r"block_summaries=(.*)$", line)
    blocks: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    if block_match is not None:
        for match in BLOCK_SUMMARY_RE.finditer(block_match.group(1)):
            block = parse_key_values(match.group("body").replace(",", " "))
            block["name"] = match.group("name").strip()
            blocks.append(block)
            if block["name"] == field_name:
                selected = block

    return {
        "line_number": line_number,
        "header": header,
        "blocks": blocks,
        "field_block": selected,
    }


def read_factorization_diagnostics(
    solver_log: Path,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    with solver_log.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            diagnostic = parse_factorization_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
    return diagnostics


def choose_diagnostic(
    diagnostics: list[dict[str, Any]],
    diagnostic_index: int,
) -> dict[str, Any]:
    if not diagnostics:
        raise RuntimeError("No Eigen direct factorization diagnostic lines found")
    index = diagnostic_index
    if index < 0:
        index = len(diagnostics) + index
    if index < 0 or index >= len(diagnostics):
        raise RuntimeError(
            f"Diagnostic index {diagnostic_index} out of range for "
            f"{len(diagnostics)} diagnostics"
        )
    return diagnostics[index]


def scalar_point_array(grid: pv.DataSet, name: str) -> np.ndarray:
    if name not in grid.point_data:
        return np.full(int(grid.n_points), math.nan, dtype=float)
    return np.asarray(grid.point_data[name], dtype=float).reshape(-1)


def vector_point_array(grid: pv.DataSet, name: str) -> np.ndarray | None:
    if name not in grid.point_data:
        return None
    values = np.asarray(grid.point_data[name], dtype=float)
    if values.ndim != 2:
        return None
    return values


def local_zero_rows(field_block: dict[str, Any]) -> tuple[list[int], str]:
    rows_from_runs = expand_index_runs(field_block.get("zero_row_runs_local"))
    if rows_from_runs:
        return rows_from_runs, "zero_row_runs_local"
    return (
        expand_index_runs(field_block.get("zero_rows_first_local")),
        "zero_rows_first_local",
    )


def row_record(
    *,
    grid: pv.DataSet,
    support: dict[str, np.ndarray],
    pressure: np.ndarray,
    phi: np.ndarray,
    active_fluid: np.ndarray,
    velocity: np.ndarray | None,
    local_pressure_row: int,
    field_begin: int,
    point_index_offset: int,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, Any]:
    point_index = local_pressure_row + point_index_offset
    record: dict[str, Any] = {
        "local_pressure_row": int(local_pressure_row),
        "global_dof": int(field_begin + local_pressure_row),
        "point_index": int(point_index),
    }
    if point_index < 0 or point_index >= grid.n_points:
        record["mapping_error"] = "point_index_out_of_range"
        record["support_class"] = "point_index_out_of_range"
        return record

    max_wet = finite_float(support["incident_wet_fraction_max"][point_index])
    min_positive = finite_float(
        support["incident_wet_fraction_min_positive"][point_index]
    )
    phi_value = finite_float(phi[point_index])
    active_value = finite_float(active_fluid[point_index])
    record.update(
        {
            "point_m": [float(value) for value in grid.points[point_index].tolist()],
            "pressure_pa": finite_float(pressure[point_index]),
            "phi": phi_value,
            "active_fluid": active_value,
            "support_class": support_class(
                phi=phi_value,
                active_fluid=active_value,
                incident_wet_fraction_max=max_wet,
                incident_wet_fraction_min_positive=min_positive,
                active_threshold=active_threshold,
                tiny_wet_fraction=tiny_wet_fraction,
                full_wet_tolerance=full_wet_tolerance,
            ),
            "incident_cell_count": int(support["incident_cell_count"][point_index]),
            "positive_wet_incident_cell_count": int(
                support["positive_wet_incident_cell_count"][point_index]
            ),
            "incident_wet_fraction_max": max_wet,
            "incident_wet_fraction_min_positive": min_positive,
        }
    )
    if velocity is not None:
        record["velocity_m_per_s"] = [
            float(value) for value in velocity[point_index].tolist()
        ]
        record["speed_m_per_s"] = float(np.linalg.norm(velocity[point_index]))
    return record


def audit_pressure_row_support(
    *,
    source_result: Path,
    solver_log: Path,
    field_name: str = "Pressure",
    diagnostic_index: int = -1,
    active_threshold: float = 0.5,
    tiny_wet_fraction: float = 1.0e-4,
    full_wet_tolerance: float = 1.0e-12,
    point_index_offset: int = 0,
) -> dict[str, Any]:
    diagnostics = read_factorization_diagnostics(solver_log, field_name=field_name)
    diagnostic = choose_diagnostic(diagnostics, diagnostic_index)
    field_block = diagnostic["field_block"]
    if field_block is None:
        raise RuntimeError(
            f"No {field_name} block found in selected factorization diagnostic"
        )

    field_begin = int(field_block["begin"])
    field_end = int(field_block["end"])
    reported_zero_rows = int(field_block.get("zero_rows", 0))
    zero_rows, row_index_source = local_zero_rows(field_block)

    grid = pv.read(source_result)
    pressure_dofs = field_end - field_begin
    point_row_mapping = "field_local_row_to_point_index"
    mapping_warning = None
    if pressure_dofs != grid.n_points:
        mapping_warning = (
            f"{field_name} block has {pressure_dofs} rows, but result has "
            f"{grid.n_points} points; row-to-point mapping may be invalid"
        )

    support = point_wet_support(grid)
    pressure = scalar_point_array(grid, field_name)
    phi = scalar_point_array(grid, "phi")
    active_fluid = scalar_point_array(grid, "ActiveFluid")
    velocity = vector_point_array(grid, "Velocity")

    rows = [
        row_record(
            grid=grid,
            support=support,
            pressure=pressure,
            phi=phi,
            active_fluid=active_fluid,
            velocity=velocity,
            local_pressure_row=local_row,
            field_begin=field_begin,
            point_index_offset=point_index_offset,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        )
        for local_row in zero_rows
    ]
    class_counts = Counter(row["support_class"] for row in rows)

    zero_cols = expand_index_runs(field_block.get("zero_col_runs_local"))
    if not zero_cols:
        zero_cols = expand_index_runs(field_block.get("zero_cols_first_local"))

    return {
        "source_result": str(source_result),
        "solver_log": str(solver_log),
        "field_name": field_name,
        "diagnostic_index": diagnostic_index,
        "diagnostic_count": len(diagnostics),
        "factorization_line_number": diagnostic["line_number"],
        "factorization_header": diagnostic["header"],
        "field_block": field_block,
        "pressure_dofs": int(pressure_dofs),
        "result_points": int(grid.n_points),
        "point_row_mapping": point_row_mapping,
        "mapping_warning": mapping_warning,
        "row_index_source": row_index_source,
        "reported_zero_row_count": reported_zero_rows,
        "classified_zero_row_count": len(rows),
        "row_list_complete": len(rows) == reported_zero_rows,
        "reported_zero_col_count": int(field_block.get("zero_cols", 0)),
        "parsed_zero_col_count": len(zero_cols),
        "zero_rows_match_zero_cols": zero_rows == zero_cols,
        "support_class_counts": dict(sorted(class_counts.items())),
        "zero_pressure_rows": rows,
    }


def main() -> int:
    args = parse_args()
    report = audit_pressure_row_support(
        source_result=args.source_result,
        solver_log=args.solver_log,
        field_name=args.field_name,
        diagnostic_index=args.diagnostic_index,
        active_threshold=args.active_fluid_threshold,
        tiny_wet_fraction=args.tiny_wet_fraction,
        full_wet_tolerance=args.full_wet_tolerance,
        point_index_offset=args.point_index_offset,
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
