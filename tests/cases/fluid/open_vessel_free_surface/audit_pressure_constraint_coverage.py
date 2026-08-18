#!/usr/bin/env python3
"""Compare pressure zero rows with solve-time active pressure constraints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pressure_row_support import (  # noqa: E402
    choose_diagnostic,
    expand_index_runs,
    local_zero_rows,
    read_factorization_diagnostics,
)
from audit_pressure_update_guard import parse_key_values  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether pressure rows reported as zero by the factorization "
            "diagnostic were covered by the solve-time active-side pressure "
            "Dirichlet constraint."
        )
    )
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--field-name", default="Pressure")
    parser.add_argument(
        "--diagnostic-index",
        type=int,
        default=-1,
        help="0-based factorization diagnostic index; negative values count from end.",
    )
    parser.add_argument(
        "--constraint-index",
        type=int,
        help=(
            "0-based pressure constraint diagnostic index; negative values count "
            "from end. By default the last matching constraint before the selected "
            "factorization diagnostic is used."
        ),
    )
    parser.add_argument(
        "--row-support-audit",
        type=Path,
        help="Optional JSON output from audit_pressure_row_support.py to merge.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_constraint_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=level_set_active_side_vertex_constraint" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "level_set_active_side_vertex_constraint":
        return None
    if values.get("field") != field_name:
        return None
    return {
        "line_number": line_number,
        "values": values,
        "inactive_dofs": expand_index_runs(values.get("inactive_dof_runs")),
        "inactive_vertices": expand_index_runs(values.get("inactive_vertex_runs")),
    }


def parse_refresh_line(line: str, *, line_number: int) -> dict[str, Any] | None:
    if "diagnostic=active_pressure_constraint_refresh" not in line:
        return None
    return {"line_number": line_number, "values": parse_key_values(line)}


def parse_sample_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=level_set_active_side_vertex_constraint_sample" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "level_set_active_side_vertex_constraint_sample":
        return None
    if values.get("field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_matrix_sample_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=newton_matrix_support_sample" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "newton_matrix_support_sample":
        return None
    field = values.get("field")
    if field is not None and field != field_name:
        return None
    return {"line_number": line_number, "values": values}


def read_pressure_constraint_diagnostics(
    solver_log: Path,
    *,
    field_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    constraints: list[dict[str, Any]] = []
    refreshes: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    matrix_samples: list[dict[str, Any]] = []
    with solver_log.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            constraint = parse_constraint_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if constraint is not None:
                constraints.append(constraint)
            refresh = parse_refresh_line(line, line_number=line_number)
            if refresh is not None:
                refreshes.append(refresh)
            sample = parse_sample_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if sample is not None:
                samples.append(sample)
            matrix_sample = parse_matrix_sample_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if matrix_sample is not None:
                matrix_samples.append(matrix_sample)
    return constraints, refreshes, samples, matrix_samples


def choose_constraint(
    constraints: list[dict[str, Any]],
    *,
    constraint_index: int | None,
    factorization_line_number: int,
) -> tuple[dict[str, Any], str]:
    if not constraints:
        raise RuntimeError("No matching pressure constraint diagnostics found")
    if constraint_index is not None:
        index = constraint_index
        if index < 0:
            index = len(constraints) + index
        if index < 0 or index >= len(constraints):
            raise RuntimeError(
                f"Constraint index {constraint_index} out of range for "
                f"{len(constraints)} constraints"
            )
        return constraints[index], "explicit_index"

    before_factorization = [
        constraint
        for constraint in constraints
        if constraint["line_number"] < factorization_line_number
    ]
    if before_factorization:
        return before_factorization[-1], "last_before_factorization"
    return constraints[-1], "last_available"


def choose_refresh(
    refreshes: list[dict[str, Any]],
    *,
    factorization_line_number: int,
) -> dict[str, Any] | None:
    before_factorization = [
        refresh
        for refresh in refreshes
        if refresh["line_number"] < factorization_line_number
    ]
    if before_factorization:
        return before_factorization[-1]
    return refreshes[-1] if refreshes else None


def latest_samples_before_factorization(
    samples: list[dict[str, Any]],
    *,
    factorization_line_number: int,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for sample in samples:
        if sample["line_number"] >= factorization_line_number:
            continue
        local_dof = sample["values"].get("local_dof")
        if not isinstance(local_dof, int):
            continue
        prior = selected.get(local_dof)
        if prior is None or sample["line_number"] > prior["line_number"]:
            selected[local_dof] = sample
    return selected


def latest_matrix_samples_before_factorization(
    samples: list[dict[str, Any]],
    *,
    factorization_line_number: int,
    field_begin: int,
    field_end: int,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for sample in samples:
        if sample["line_number"] >= factorization_line_number:
            continue
        values = sample["values"]
        local_dof = values.get("field_local_dof")
        if not isinstance(local_dof, int):
            global_dof = values.get("dof")
            if not isinstance(global_dof, int):
                continue
            if global_dof < field_begin or global_dof >= field_end:
                continue
            local_dof = global_dof - field_begin
        prior = selected.get(local_dof)
        if prior is None or sample["line_number"] > prior["line_number"]:
            selected[local_dof] = sample
    return selected


def load_row_support(
    path: Path | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[int, dict[str, Any]] = {}
    for row in report.get("zero_pressure_rows", []):
        local_row = row.get("local_pressure_row")
        if isinstance(local_row, int):
            rows[local_row] = row
    return rows, report


def constraint_vertex_dof_mapping_status(constraint: dict[str, Any]) -> dict[str, Any]:
    values = constraint["values"]
    inactive_dofs = constraint["inactive_dofs"]
    inactive_vertices = constraint["inactive_vertices"]
    total_dofs = values.get("total_dofs")
    total_vertices = values.get("total_vertices")
    if not inactive_vertices:
        return {
            "status": "inactive_vertex_runs_not_reported",
            "direct_local_row_to_point_mapping_supported": False,
        }
    if total_dofs != total_vertices:
        return {
            "status": "dof_vertex_count_mismatch",
            "direct_local_row_to_point_mapping_supported": False,
        }
    if inactive_dofs == inactive_vertices:
        return {
            "status": "inactive_dof_runs_match_inactive_vertex_runs",
            "direct_local_row_to_point_mapping_supported": True,
        }
    return {
        "status": "inactive_dof_runs_differ_from_inactive_vertex_runs",
        "direct_local_row_to_point_mapping_supported": False,
    }


def support_constraint_mismatch(
    support_class: str | None,
    *,
    constraint_inactive: bool,
    row_support_mapping_reliable: bool,
) -> str:
    if support_class is None:
        return "support_not_available"
    if not row_support_mapping_reliable:
        return "saved_support_mapping_unverified"
    if support_class == "dry_or_inactive" and not constraint_inactive:
        return "saved_dry_not_constraint_inactive"
    if support_class != "dry_or_inactive" and constraint_inactive:
        return "saved_supported_but_constraint_inactive"
    return "consistent"


def matrix_value_abs_leq(
    values: dict[str, Any],
    key: str,
    *,
    tolerance: float,
) -> bool:
    value = values.get(key)
    return isinstance(value, (int, float)) and abs(float(value)) <= tolerance


def zero_row_record(
    *,
    local_row: int,
    field_begin: int,
    inactive_dofs: set[int],
    row_support: dict[int, dict[str, Any]],
    row_support_mapping_reliable: bool,
    runtime_samples: dict[int, dict[str, Any]],
    matrix_samples: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    saved_row = row_support.get(local_row)
    support_class = (
        str(saved_row["support_class"])
        if saved_row is not None and "support_class" in saved_row
        else None
    )
    constraint_inactive = local_row in inactive_dofs
    record: dict[str, Any] = {
        "local_pressure_row": int(local_row),
        "global_dof": int(field_begin + local_row),
        "constraint_inactive": bool(constraint_inactive),
        "constraint_class": (
            "inactive_constrained"
            if constraint_inactive
            else "active_or_unconstrained"
        ),
        "saved_support_class": support_class,
        "saved_support_mapping_reliable": bool(row_support_mapping_reliable),
        "mismatch_class": support_constraint_mismatch(
            support_class,
            constraint_inactive=constraint_inactive,
            row_support_mapping_reliable=row_support_mapping_reliable,
        ),
    }
    sample = runtime_samples.get(local_row)
    record["runtime_sample_available"] = sample is not None
    if sample is not None:
        record["runtime_sample_line_number"] = sample["line_number"]
        record["runtime_sample"] = sample["values"]
    matrix_sample = matrix_samples.get(local_row)
    record["matrix_sample_available"] = matrix_sample is not None
    if matrix_sample is not None:
        record["matrix_sample_line_number"] = matrix_sample["line_number"]
        record["matrix_sample"] = matrix_sample["values"]
    if saved_row is not None:
        record["saved_state"] = {
            key: saved_row[key]
            for key in (
                "point_index",
                "point_m",
                "phi",
                "active_fluid",
                "incident_cell_count",
                "positive_wet_incident_cell_count",
                "incident_wet_fraction_max",
                "incident_wet_fraction_min_positive",
            )
            if key in saved_row
        }
    return record


def audit_pressure_constraint_coverage(
    *,
    solver_log: Path,
    field_name: str = "Pressure",
    diagnostic_index: int = -1,
    constraint_index: int | None = None,
    row_support_audit: Path | None = None,
) -> dict[str, Any]:
    factorization_diagnostics = read_factorization_diagnostics(
        solver_log,
        field_name=field_name,
    )
    factorization = choose_diagnostic(factorization_diagnostics, diagnostic_index)
    field_block = factorization["field_block"]
    if field_block is None:
        raise RuntimeError(
            f"No {field_name} block found in selected factorization diagnostic"
        )

    (
        constraints,
        refreshes,
        samples,
        matrix_sample_lines,
    ) = read_pressure_constraint_diagnostics(solver_log, field_name=field_name)
    constraint, constraint_selection = choose_constraint(
        constraints,
        constraint_index=constraint_index,
        factorization_line_number=factorization["line_number"],
    )
    refresh = choose_refresh(
        refreshes,
        factorization_line_number=factorization["line_number"],
    )
    row_support, row_support_report = load_row_support(row_support_audit)
    mapping_status = constraint_vertex_dof_mapping_status(constraint)
    row_support_mapping_reliable = bool(
        mapping_status["direct_local_row_to_point_mapping_supported"]
    )
    runtime_samples = latest_samples_before_factorization(
        samples,
        factorization_line_number=factorization["line_number"],
    )

    field_begin = int(field_block["begin"])
    field_end = int(field_block["end"])
    pressure_dofs = field_end - field_begin
    matrix_samples = latest_matrix_samples_before_factorization(
        matrix_sample_lines,
        factorization_line_number=factorization["line_number"],
        field_begin=field_begin,
        field_end=field_end,
    )
    zero_rows, row_index_source = local_zero_rows(field_block)
    zero_cols = expand_index_runs(field_block.get("zero_col_runs_local"))
    if not zero_cols:
        zero_cols = expand_index_runs(field_block.get("zero_cols_first_local"))

    inactive_dofs = set(int(row) for row in constraint["inactive_dofs"])
    rows = [
        zero_row_record(
            local_row=local_row,
            field_begin=field_begin,
            inactive_dofs=inactive_dofs,
            row_support=row_support,
            row_support_mapping_reliable=row_support_mapping_reliable,
            runtime_samples=runtime_samples,
            matrix_samples=matrix_samples,
        )
        for local_row in zero_rows
    ]

    constraint_counts = Counter(row["constraint_class"] for row in rows)
    mismatch_counts = Counter(row["mismatch_class"] for row in rows)
    support_counts = Counter(
        row["saved_support_class"]
        for row in rows
        if row["saved_support_class"] is not None
    )
    sampled_rows = [row for row in rows if row["runtime_sample_available"]]
    sampled_entity_counts = Counter(
        str(row["runtime_sample"].get("entity_kind"))
        for row in sampled_rows
        if "runtime_sample" in row
    )
    sampled_active_support_count = sum(
        1
        for row in sampled_rows
        if row["runtime_sample"].get("active_dof_support") == 1
    )
    sampled_vertex_active_sign_count = sum(
        1
        for row in sampled_rows
        if row["runtime_sample"].get("vertex_active_sign") == 1
    )
    matrix_sampled_rows = [row for row in rows if row["matrix_sample_available"]]
    matrix_sample_status_counts = Counter(
        str(row["matrix_sample"].get("status"))
        for row in matrix_sampled_rows
        if "matrix_sample" in row
    )
    matrix_zero_tolerance = 1.0e-14
    matrix_zero_row_count = sum(
        1
        for row in matrix_sampled_rows
        if matrix_value_abs_leq(
            row["matrix_sample"],
            "row_abs_sum",
            tolerance=matrix_zero_tolerance,
        )
    )
    matrix_zero_col_count = sum(
        1
        for row in matrix_sampled_rows
        if matrix_value_abs_leq(
            row["matrix_sample"],
            "col_abs_sum",
            tolerance=matrix_zero_tolerance,
        )
    )
    matrix_zero_diag_count = sum(
        1
        for row in matrix_sampled_rows
        if matrix_value_abs_leq(
            row["matrix_sample"],
            "diag",
            tolerance=matrix_zero_tolerance,
        )
    )
    logged_inactive_dofs = int(constraint["values"].get("inactive_dofs", -1))
    field_identity_rows = int(field_block.get("identity_rows", 0))

    return {
        "solver_log": str(solver_log),
        "field_name": field_name,
        "diagnostic_index": diagnostic_index,
        "diagnostic_count": len(factorization_diagnostics),
        "factorization_line_number": factorization["line_number"],
        "factorization_header": factorization["header"],
        "field_block": field_block,
        "pressure_dofs": int(pressure_dofs),
        "row_index_source": row_index_source,
        "reported_zero_row_count": int(field_block.get("zero_rows", 0)),
        "parsed_zero_row_count": len(zero_rows),
        "reported_zero_col_count": int(field_block.get("zero_cols", 0)),
        "parsed_zero_col_count": len(zero_cols),
        "row_list_complete": len(zero_rows) == int(field_block.get("zero_rows", 0)),
        "zero_rows_match_zero_cols": zero_rows == zero_cols,
        "constraint_count": len(constraints),
        "constraint_sample_count": len(samples),
        "matrix_sample_count": len(matrix_sample_lines),
        "constraint_selection": constraint_selection,
        "constraint_line_number": constraint["line_number"],
        "constraint_values": constraint["values"],
        "constraint_inactive_dof_count_from_runs": len(inactive_dofs),
        "constraint_inactive_dof_count_matches_log": (
            logged_inactive_dofs < 0 or logged_inactive_dofs == len(inactive_dofs)
        ),
        "constraint_vertex_dof_mapping_status": mapping_status,
        "row_support_mapping_reliable": row_support_mapping_reliable,
        "runtime_sampled_zero_row_count": len(sampled_rows),
        "runtime_sampled_zero_row_entity_kind_counts": dict(
            sorted(sampled_entity_counts.items())
        ),
        "runtime_sampled_zero_rows_active_dof_support_count": int(
            sampled_active_support_count
        ),
        "runtime_sampled_zero_rows_vertex_active_sign_count": int(
            sampled_vertex_active_sign_count
        ),
        "matrix_sampled_zero_row_count": len(matrix_sampled_rows),
        "matrix_sample_status_counts": dict(sorted(matrix_sample_status_counts.items())),
        "matrix_zero_tolerance": matrix_zero_tolerance,
        "matrix_sampled_zero_rows_zero_row_count": int(matrix_zero_row_count),
        "matrix_sampled_zero_rows_zero_col_count": int(matrix_zero_col_count),
        "matrix_sampled_zero_rows_zero_diag_count": int(matrix_zero_diag_count),
        "field_identity_rows_minus_constraint_inactive_dofs": (
            field_identity_rows - len(inactive_dofs)
        ),
        "refresh_line_number": refresh["line_number"] if refresh is not None else None,
        "refresh_values": refresh["values"] if refresh is not None else None,
        "row_support_audit": str(row_support_audit) if row_support_audit else None,
        "row_support_source_result": (
            row_support_report.get("source_result")
            if row_support_report is not None
            else None
        ),
        "constraint_class_counts": dict(sorted(constraint_counts.items())),
        "saved_support_class_counts": dict(sorted(support_counts.items())),
        "mismatch_class_counts": dict(sorted(mismatch_counts.items())),
        "zero_rows_in_constraint_inactive_count": int(
            constraint_counts.get("inactive_constrained", 0)
        ),
        "zero_rows_missing_constraint_count": int(
            constraint_counts.get("active_or_unconstrained", 0)
        ),
        "saved_dry_zero_rows_missing_constraint_count": int(
            mismatch_counts.get("saved_dry_not_constraint_inactive", 0)
        ),
        "unverified_saved_support_zero_row_count": int(
            mismatch_counts.get("saved_support_mapping_unverified", 0)
        ),
        "zero_pressure_rows": rows,
    }


def main() -> int:
    args = parse_args()
    report = audit_pressure_constraint_coverage(
        solver_log=args.solver_log,
        field_name=args.field_name,
        diagnostic_index=args.diagnostic_index,
        constraint_index=args.constraint_index,
        row_support_audit=args.row_support_audit,
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
