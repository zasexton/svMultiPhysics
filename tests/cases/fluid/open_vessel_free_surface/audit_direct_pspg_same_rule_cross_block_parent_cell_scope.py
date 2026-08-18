#!/usr/bin/env python3
"""Derive parent-cell replay scope for same-rule direct PSPG candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_direct_pspg_solve_time_sampled_column_selectivity import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    read_provenance_log,
)


DEFAULT_CANDIDATE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_same_rule_cross_block_signature_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_scope_20260607.json"
)

CASE_LOGS = {
    "test02": DEFAULT_TEST02_LOG,
    "test10": DEFAULT_TEST10_LOG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument(
        "--max-expanded-row-ratio",
        type=float,
        default=10.0,
        help=(
            "Largest sampled-row expansion ratio considered narrow enough "
            "for a parent-cell replay."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def int_set(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, int)}


def candidate_rows_by_case(candidate: dict[str, Any]) -> dict[str, set[int]]:
    rows: dict[str, set[int]] = {}
    for case in candidate.get("cases", []):
        if not isinstance(case, dict) or not isinstance(case.get("label"), str):
            continue
        rows[case["label"]] = int_set(
            case.get("best_covering_composite_selected_global_dofs")
        )
    return rows


def sorted_ints(values: set[int]) -> list[int]:
    return sorted(value for value in values if isinstance(value, int))


def range_string(values: list[int]) -> str:
    if not values:
        return ""
    ranges: list[str] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = value
        previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def build_case_scope(
    *,
    label: str,
    log_path: Path,
    candidate_rows: set[int],
    max_expanded_row_ratio: float,
) -> dict[str, Any]:
    entries = read_provenance_log(log_path) if log_path.exists() else []
    candidate_entries = [
        entry for entry in entries if entry.get("row_dof") in candidate_rows
    ]
    rows_present = {
        int(entry["row_dof"])
        for entry in candidate_entries
        if isinstance(entry.get("row_dof"), int)
    }
    parent_cells = {
        int(entry["parent_cell"])
        for entry in candidate_entries
        if isinstance(entry.get("parent_cell"), int)
    }
    rule_indices = {
        int(entry["rule_index"])
        for entry in candidate_entries
        if isinstance(entry.get("rule_index"), int)
    }
    parent_rules = {
        (int(entry["parent_cell"]), int(entry["rule_index"]))
        for entry in candidate_entries
        if isinstance(entry.get("parent_cell"), int)
        and isinstance(entry.get("rule_index"), int)
    }
    full_parent_cells = {
        int(entry["parent_cell"])
        for entry in candidate_entries
        if isinstance(entry.get("parent_cell"), int)
        and entry.get("full_cell") == 1
    }
    cut_parent_cells = {
        int(entry["parent_cell"])
        for entry in candidate_entries
        if isinstance(entry.get("parent_cell"), int)
        and entry.get("full_cell") != 1
    }
    parent_expanded_rows = {
        int(entry["row_dof"])
        for entry in entries
        if isinstance(entry.get("row_dof"), int)
        and isinstance(entry.get("parent_cell"), int)
        and int(entry["parent_cell"]) in parent_cells
    }
    parent_rule_expanded_rows = {
        int(entry["row_dof"])
        for entry in entries
        if isinstance(entry.get("row_dof"), int)
        and isinstance(entry.get("parent_cell"), int)
        and isinstance(entry.get("rule_index"), int)
        and (int(entry["parent_cell"]), int(entry["rule_index"])) in parent_rules
    }
    parent_cell_list = sorted_ints(parent_cells)
    parent_expanded_ratio = safe_ratio(len(parent_expanded_rows), len(candidate_rows))
    missing_candidate_rows = sorted_ints(candidate_rows - rows_present)
    ready = (
        log_path.exists()
        and not missing_candidate_rows
        and bool(parent_cells)
        and isinstance(parent_expanded_ratio, float)
        and parent_expanded_ratio <= max_expanded_row_ratio
    )
    return {
        "label": label,
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "candidate_row_count": len(candidate_rows),
        "candidate_rows": sorted_ints(candidate_rows),
        "candidate_rows_present_count": len(rows_present),
        "missing_candidate_rows": missing_candidate_rows,
        "candidate_record_count": len(candidate_entries),
        "parent_cell_count": len(parent_cell_list),
        "parent_cells": parent_cell_list,
        "parent_cell_ranges": range_string(parent_cell_list),
        "replay_parent_cell_global_input": ",".join(str(value) for value in parent_cell_list),
        "rule_index_count": len(rule_indices),
        "rule_indices": sorted_ints(rule_indices),
        "parent_rule_count": len(parent_rules),
        "full_parent_cell_count": len(full_parent_cells),
        "full_parent_cells": sorted_ints(full_parent_cells),
        "cut_parent_cell_count": len(cut_parent_cells),
        "cut_parent_cells": sorted_ints(cut_parent_cells),
        "parent_expanded_row_count": len(parent_expanded_rows),
        "parent_expanded_rows": sorted_ints(parent_expanded_rows),
        "parent_expanded_to_candidate_ratio": parent_expanded_ratio,
        "parent_rule_expanded_row_count": len(parent_rule_expanded_rows),
        "parent_rule_expanded_rows": sorted_ints(parent_rule_expanded_rows),
        "parent_rule_expanded_to_candidate_ratio": safe_ratio(
            len(parent_rule_expanded_rows), len(candidate_rows)
        ),
        "ready_for_parent_cell_replay": ready,
    }


def build_report(
    *,
    candidate_json: Path = DEFAULT_CANDIDATE,
    test02_log: Path = DEFAULT_TEST02_LOG,
    test10_log: Path = DEFAULT_TEST10_LOG,
    max_expanded_row_ratio: float = 10.0,
) -> dict[str, Any]:
    candidate = load_json(candidate_json) if candidate_json.exists() else {}
    rows_by_case = candidate_rows_by_case(candidate)
    logs = {"test02": test02_log, "test10": test10_log}
    cases = [
        build_case_scope(
            label=label,
            log_path=logs[label],
            candidate_rows=rows_by_case.get(label, set()),
            max_expanded_row_ratio=max_expanded_row_ratio,
        )
        for label in ("test02", "test10")
    ]
    missing = []
    if not candidate_json.exists():
        missing.append(str(candidate_json))
    missing.extend(case["log_path"] for case in cases if not case["log_exists"])
    ready_cases = [
        case["label"] for case in cases if case["ready_for_parent_cell_replay"]
    ]
    if missing:
        finding = "direct_pspg_same_rule_cross_block_parent_cell_scope_incomplete"
        status = "regenerate_missing_parent_scope_inputs"
        conclusion = (
            "At least one same-rule candidate artifact or sampled-column "
            "provenance log is missing."
        )
    elif len(ready_cases) == len(cases):
        finding = (
            "direct_pspg_same_rule_cross_block_parent_cell_scope_ready_for_replay"
        )
        status = "run_same_rule_cross_block_parent_cell_replay"
        conclusion = (
            "The exported same-rule candidate rows map to parent-cell scopes "
            "that cover every candidate row and stay within the configured "
            "sampled-row expansion threshold."
        )
    else:
        finding = (
            "direct_pspg_same_rule_cross_block_parent_cell_scope_not_replay_ready"
        )
        status = "parent_cell_scope_overbroad_or_missing_candidates"
        conclusion = (
            "The parent-cell expansion is missing candidate rows or exceeds "
            "the configured sampled-row expansion threshold."
        )
    return {
        "scope": (
            "Parent-cell replay scope derived from solve-time sampled-column "
            "provenance entries for the same-rule cross-block candidate rows."
        ),
        "finding": finding,
        "status": status,
        "candidate_artifact": str(candidate_json),
        "candidate_finding": candidate.get("finding") if isinstance(candidate, dict) else None,
        "max_expanded_row_ratio": max_expanded_row_ratio,
        "ready_cases": ready_cases,
        "all_cases_ready_for_parent_cell_replay": len(ready_cases) == len(cases),
        "missing_evidence": missing,
        "cases": cases,
        "conclusion": conclusion,
        "next_requirement": (
            "Run local_schur_edge_balance with the derived parent-cell filters "
            "and no global row filter; compare against the row-list replay and "
            "same-case no-policy baseline."
            if len(ready_cases) == len(cases) and not missing
            else "Regenerate the missing or overbroad parent-cell scope inputs."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        candidate_json=args.candidate_json,
        test02_log=args.test02_log,
        test10_log=args.test10_log,
        max_expanded_row_ratio=args.max_expanded_row_ratio,
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
