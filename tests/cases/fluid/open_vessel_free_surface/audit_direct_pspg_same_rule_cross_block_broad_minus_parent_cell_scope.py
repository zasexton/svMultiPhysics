#!/usr/bin/env python3
"""Derive broad-minus-same-rule parent-cell replay scopes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_direct_pspg_same_rule_cross_block_parent_cell_replays import (  # noqa: E402
    parse_policy_log,
)


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_PARENT_SCOPE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_scope_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope_20260607.json"
)

BROAD_LOGS = {
    "test02": (
        DEFAULT_ARTIFACT_ROOT
        / "test02_replay_abs_only_prune1e5_step382_direct_pspg_topology_policy_schur_edge_balance_20260607_case"
        / "run_direct_pspg_topology_policy_schur_edge_balance.log"
    ),
    "test10": (
        DEFAULT_ARTIFACT_ROOT
        / "test10_replay_cap3_step90_direct_pspg_topology_policy_schur_edge_balance_20260607_case"
        / "run_direct_pspg_topology_policy_schur_edge_balance.log"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--parent-scope-json", type=Path, default=DEFAULT_PARENT_SCOPE)
    parser.add_argument("--test02-broad-log", type=Path, default=BROAD_LOGS["test02"])
    parser.add_argument("--test10-broad-log", type=Path, default=BROAD_LOGS["test10"])
    parser.add_argument(
        "--max-broad-only-parent-count",
        type=int,
        default=4000,
        help="Largest complement parent-cell set considered replay-ready.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def parent_scope_case(parent_scope: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if not isinstance(parent_scope, dict):
        return {}
    for case in parent_scope.get("cases", []):
        if isinstance(case, dict) and case.get("label") == label:
            return case
    return {}


def broad_parent_cells(log_path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    records = parse_policy_log(log_path)
    parent_cells = {
        int(record["parent_cell"])
        for record in records
        if isinstance(record.get("parent_cell"), int)
        and record.get("matrix_mutated") == 1
    }
    return records, parent_cells


def build_case_scope(
    *,
    label: str,
    broad_log: Path,
    parent_scope: dict[str, Any] | None,
    max_broad_only_parent_count: int,
) -> dict[str, Any]:
    records, broad_cells = broad_parent_cells(broad_log) if broad_log.exists() else ([], set())
    scope = parent_scope_case(parent_scope, label)
    same_rule_cells = {
        int(value) for value in scope.get("parent_cells", []) if isinstance(value, int)
    }
    overlap = broad_cells & same_rule_cells
    broad_only = broad_cells - same_rule_cells
    same_rule_not_in_broad = same_rule_cells - broad_cells
    broad_only_list = sorted_ints(broad_only)
    ready = (
        broad_log.exists()
        and bool(broad_only)
        and not same_rule_not_in_broad
        and len(broad_only) <= max_broad_only_parent_count
    )
    return {
        "label": label,
        "broad_log_path": str(broad_log),
        "broad_log_exists": broad_log.exists(),
        "broad_record_count": len(records),
        "broad_parent_cell_count": len(broad_cells),
        "same_rule_parent_cell_count": len(same_rule_cells),
        "overlap_parent_cell_count": len(overlap),
        "same_rule_not_in_broad_count": len(same_rule_not_in_broad),
        "same_rule_not_in_broad_parent_cells": sorted_ints(same_rule_not_in_broad),
        "broad_only_parent_cell_count": len(broad_only),
        "broad_only_parent_cells": broad_only_list,
        "broad_only_parent_cell_ranges": range_string(broad_only_list),
        "replay_parent_cell_global_input": ",".join(str(value) for value in broad_only_list),
        "broad_only_to_broad_parent_ratio": safe_ratio(len(broad_only), len(broad_cells)),
        "same_rule_to_broad_parent_ratio": safe_ratio(len(same_rule_cells), len(broad_cells)),
        "ready_for_broad_minus_parent_cell_replay": ready,
    }


def build_report(
    *,
    parent_scope_json: Path = DEFAULT_PARENT_SCOPE,
    test02_broad_log: Path = BROAD_LOGS["test02"],
    test10_broad_log: Path = BROAD_LOGS["test10"],
    max_broad_only_parent_count: int = 4000,
) -> dict[str, Any]:
    parent_scope = load_json(parent_scope_json)
    logs = {"test02": test02_broad_log, "test10": test10_broad_log}
    cases = [
        build_case_scope(
            label=label,
            broad_log=logs[label],
            parent_scope=parent_scope,
            max_broad_only_parent_count=max_broad_only_parent_count,
        )
        for label in ("test02", "test10")
    ]
    missing = []
    if not parent_scope_json.exists():
        missing.append(str(parent_scope_json))
    missing.extend(case["broad_log_path"] for case in cases if not case["broad_log_exists"])
    ready_cases = [
        case["label"] for case in cases if case["ready_for_broad_minus_parent_cell_replay"]
    ]
    if missing:
        finding = "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_incomplete"
        status = "regenerate_missing_broad_minus_scope_inputs"
        conclusion = (
            "At least one parent-scope artifact or broad-policy topology log is missing."
        )
    elif len(ready_cases) == len(cases):
        finding = (
            "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_ready_for_replay"
        )
        status = "run_broad_minus_same_rule_parent_cell_replay"
        conclusion = (
            "The same-rule parent-cell scope is a strict subset of broad-policy "
            "support in both cases, and the broad-minus complement is available "
            "for a causal replay."
        )
    else:
        finding = (
            "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_not_ready"
        )
        status = "broad_minus_parent_scope_too_large_or_incomplete"
        conclusion = (
            "The broad-minus complement is missing, contains unexpected "
            "same-rule cells outside broad support, or exceeds the replay-size cap."
        )
    return {
        "scope": (
            "Broad local_schur_edge_balance parent-cell support minus the "
            "same-rule cross-block parent-cell candidate scope."
        ),
        "finding": finding,
        "status": status,
        "parent_scope_artifact": str(parent_scope_json),
        "parent_scope_finding": (
            parent_scope.get("finding") if isinstance(parent_scope, dict) else None
        ),
        "max_broad_only_parent_count": max_broad_only_parent_count,
        "ready_cases": ready_cases,
        "all_cases_ready_for_broad_minus_parent_cell_replay": len(ready_cases)
        == len(cases),
        "missing_evidence": missing,
        "cases": cases,
        "conclusion": conclusion,
        "next_requirement": (
            "Run local_schur_edge_balance with the broad-minus parent-cell "
            "filters and no global row filter; compare against same-rule parent "
            "and broad-policy controls."
            if len(ready_cases) == len(cases) and not missing
            else "Do not run the complement replay until the scope inputs are complete and bounded."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        parent_scope_json=args.parent_scope_json,
        test02_broad_log=args.test02_broad_log,
        test10_broad_log=args.test10_broad_log,
        max_broad_only_parent_count=args.max_broad_only_parent_count,
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
