#!/usr/bin/env python3
"""Summarize broad-minus-same-rule direct PSPG parent-cell replays."""

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
    DEFAULT_ARTIFACT_ROOT,
    pressure_summary,
    policy_log_summary,
    safe_delta,
    safe_ratio,
    values,
)


DEFAULT_BROAD_MINUS_SCOPE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays_20260607.json"
)

REPLAYS = {
    "test02": {
        "case_dir": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "broad_minus_same_rule_parent_cells_schur_edge_balance_20260607_case"
        ),
        "log_name": (
            "run_direct_pspg_broad_minus_same_rule_parent_cells_"
            "schur_edge_balance.log"
        ),
        "audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "broad_minus_same_rule_parent_cells_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "baseline_audit_name": (
            "test02_replay_abs_only_prune1e5_step382_"
            "pspg_wall_full_gradient_scale1_coverage_"
            "pressure_update_audit_20260606.json"
        ),
        "row_filter_audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "same_rule_cross_block_rows_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "parent_cell_audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "same_rule_cross_block_parent_cells_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "broad_policy_audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_schur_edge_balance_pressure_update_audit_"
            "20260607.json"
        ),
    },
    "test10": {
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_"
            "broad_minus_same_rule_parent_cells_schur_edge_balance_20260607_case"
        ),
        "log_name": (
            "run_direct_pspg_broad_minus_same_rule_parent_cells_"
            "schur_edge_balance.log"
        ),
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_"
            "broad_minus_same_rule_parent_cells_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "baseline_audit_name": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_scale1_"
            "coverage_pressure_update_audit_20260606.json"
        ),
        "row_filter_audit_name": (
            "test10_replay_cap3_step90_direct_pspg_"
            "same_rule_cross_block_rows_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "parent_cell_audit_name": (
            "test10_replay_cap3_step90_direct_pspg_"
            "same_rule_cross_block_parent_cells_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "broad_policy_audit_name": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--broad-minus-scope-json",
        type=Path,
        default=DEFAULT_BROAD_MINUS_SCOPE,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scope_case(scope: dict[str, Any] | None, label: str) -> dict[str, Any]:
    for case in values(scope).get("cases", []):
        if isinstance(case, dict) and case.get("label") == label:
            return case
    return {}


def summarize_case(
    *,
    label: str,
    spec: dict[str, str],
    artifact_root: Path,
    broad_minus_scope: dict[str, Any] | None,
) -> dict[str, Any]:
    scope = scope_case(broad_minus_scope, label)
    replay_audit = pressure_summary(artifact_root / spec["audit_name"])
    baseline_audit = pressure_summary(artifact_root / spec["baseline_audit_name"])
    row_filter_audit = pressure_summary(artifact_root / spec["row_filter_audit_name"])
    parent_cell_audit = pressure_summary(artifact_root / spec["parent_cell_audit_name"])
    broad_audit = pressure_summary(artifact_root / spec["broad_policy_audit_name"])
    replay_update = replay_audit["worst_active_or_wet_update_pa"]
    baseline_update = baseline_audit["worst_active_or_wet_update_pa"]
    row_filter_update = row_filter_audit["worst_active_or_wet_update_pa"]
    parent_cell_update = parent_cell_audit["worst_active_or_wet_update_pa"]
    broad_update = broad_audit["worst_active_or_wet_update_pa"]
    policy_log = policy_log_summary(
        artifact_root / spec["case_dir"] / spec["log_name"]
    )
    expected_parent_cell_count = scope.get("broad_only_parent_cell_count")
    return {
        "label": label,
        "case_dir": str(artifact_root / spec["case_dir"]),
        "solver_log_path": str(artifact_root / spec["case_dir"] / spec["log_name"]),
        "broad_minus_scope": {
            "exists": bool(scope),
            "expected_parent_cell_count": expected_parent_cell_count,
            "broad_parent_cell_count": scope.get("broad_parent_cell_count"),
            "same_rule_parent_cell_count": scope.get(
                "same_rule_parent_cell_count"
            ),
            "broad_only_to_broad_parent_ratio": scope.get(
                "broad_only_to_broad_parent_ratio"
            ),
            "ready_for_broad_minus_parent_cell_replay": scope.get(
                "ready_for_broad_minus_parent_cell_replay"
            ),
        },
        "pressure_update": replay_audit,
        "baseline_pressure_update": baseline_audit,
        "row_filter_pressure_update": row_filter_audit,
        "same_rule_parent_cell_pressure_update": parent_cell_audit,
        "broad_policy_pressure_update": broad_audit,
        "topology_log": policy_log,
        "parent_filter_matches_scope_count": (
            expected_parent_cell_count
            in policy_log.get("parent_filter_parent_cell_count_values", [])
            if isinstance(expected_parent_cell_count, int)
            else False
        ),
        "row_filter_disabled": policy_log.get("row_filter_enabled_values") == [0],
        "improvement_vs_baseline_pa": safe_delta(baseline_update, replay_update),
        "replay_to_baseline_update_ratio": safe_ratio(
            replay_update, baseline_update
        ),
        "improvement_vs_row_filter_pa": safe_delta(row_filter_update, replay_update),
        "replay_to_row_filter_update_ratio": safe_ratio(
            replay_update, row_filter_update
        ),
        "improvement_vs_same_rule_parent_cell_pa": safe_delta(
            parent_cell_update, replay_update
        ),
        "replay_to_same_rule_parent_cell_update_ratio": safe_ratio(
            replay_update, parent_cell_update
        ),
        "improvement_vs_broad_policy_pa": safe_delta(broad_update, replay_update),
        "replay_to_broad_policy_update_ratio": safe_ratio(
            replay_update, broad_update
        ),
    }


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    broad_minus_scope_json: Path = DEFAULT_BROAD_MINUS_SCOPE,
) -> dict[str, Any]:
    broad_minus_scope = load_json(broad_minus_scope_json)
    cases = [
        summarize_case(
            label=label,
            spec=spec,
            artifact_root=artifact_root,
            broad_minus_scope=broad_minus_scope,
        )
        for label, spec in REPLAYS.items()
    ]
    missing = []
    if not broad_minus_scope_json.exists():
        missing.append(str(broad_minus_scope_json))
    missing.extend(
        case["pressure_update"]["path"]
        for case in cases
        if not case["pressure_update"]["exists"]
    )
    missing.extend(
        case["solver_log_path"] for case in cases if not case["topology_log"]["exists"]
    )
    guard_triggered = [
        case for case in cases if case["pressure_update"]["guard_triggered"]
    ]
    guard_cleared = [
        case for case in cases if case["pressure_update"]["guard_cleared"]
    ]
    parent_filters_match = all(
        case["parent_filter_matches_scope_count"] for case in cases
    )
    row_filters_disabled = all(case["row_filter_disabled"] for case in cases)
    broad_better_than_parts = all(
        isinstance(case.get("improvement_vs_broad_policy_pa"), (int, float))
        and case["improvement_vs_broad_policy_pa"] < 0.0
        for case in cases
    )
    complement_worse_than_same_rule_parent = all(
        isinstance(
            case.get("improvement_vs_same_rule_parent_cell_pa"), (int, float)
        )
        and case["improvement_vs_same_rule_parent_cell_pa"] < 0.0
        for case in cases
    )

    if missing:
        finding = (
            "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_incomplete"
        )
        status = "regenerate_missing_broad_minus_parent_replay_artifacts"
        conclusion = (
            "At least one broad-minus parent-cell replay, pressure audit, "
            "solver log, or scope artifact is missing."
        )
    elif guard_cleared and not guard_triggered:
        finding = (
            "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_clear_guards"
        )
        status = "broad_minus_parent_replay_clears_short_windows"
        conclusion = (
            "The broad-minus-same-rule parent-cell replay clears both short "
            "active/wet pressure-update guards."
        )
    elif len(guard_triggered) == len(cases):
        finding = (
            "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_do_not_clear_guards"
        )
        status = "broad_minus_parent_replay_insufficient"
        conclusion = (
            "The broad-minus-same-rule parent-cell replay still triggers both "
            "active/wet guards. Broad policy is better than both isolated "
            "subsets, so the helpful broad-policy effect is a broad union or "
            "synergy effect, not a sufficient same-rule or broad-only subset."
        )
    else:
        finding = "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_mixed"
        status = "inspect_broad_minus_parent_replay_statuses"
        conclusion = "Broad-minus parent-cell replay statuses are mixed."

    return {
        "scope": (
            "Targeted Test02 step382 and Test10 step90 solve-time direct PSPG "
            "local_schur_edge_balance replays restricted to broad-policy parent "
            "cells outside the same-rule cross-block parent-cell candidate."
        ),
        "finding": finding,
        "status": status,
        "broad_minus_scope_artifact": str(broad_minus_scope_json),
        "broad_minus_scope_finding": (
            broad_minus_scope.get("finding")
            if isinstance(broad_minus_scope, dict)
            else None
        ),
        "parent_filters_match_scope_counts": parent_filters_match,
        "row_filters_disabled": row_filters_disabled,
        "all_replays_trigger_guard": len(guard_triggered) == len(cases)
        and not missing,
        "broad_policy_better_than_isolated_parts": broad_better_than_parts,
        "complement_worse_than_same_rule_parent_cell": (
            complement_worse_than_same_rule_parent
        ),
        "cleared_cases": [case["label"] for case in guard_cleared],
        "triggered_cases": [case["label"] for case in guard_triggered],
        "cases": cases,
        "conclusion": conclusion,
        "next_requirement": (
            "Do not promote same-rule parent cells or the broad-only complement "
            "as a fix. The remaining rule must explain why broad union support "
            "helps both cases while still avoiding the broad policy's residual "
            "guard-triggering behavior."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        broad_minus_scope_json=args.broad_minus_scope_json,
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
