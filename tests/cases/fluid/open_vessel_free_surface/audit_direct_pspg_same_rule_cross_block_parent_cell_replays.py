#!/usr/bin/env python3
"""Summarize same-rule cross-block direct PSPG parent-cell replays."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_PARENT_SCOPE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_scope_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_replays_20260607.json"
)

REPLAYS = {
    "test02": {
        "case_dir": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "same_rule_cross_block_parent_cells_schur_edge_balance_20260607_case"
        ),
        "log_name": (
            "run_direct_pspg_same_rule_cross_block_parent_cells_"
            "schur_edge_balance.log"
        ),
        "audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "same_rule_cross_block_parent_cells_schur_edge_balance_"
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
        "broad_policy_audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_schur_edge_balance_pressure_update_audit_"
            "20260607.json"
        ),
    },
    "test10": {
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_"
            "same_rule_cross_block_parent_cells_schur_edge_balance_20260607_case"
        ),
        "log_name": (
            "run_direct_pspg_same_rule_cross_block_parent_cells_"
            "schur_edge_balance.log"
        ),
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_"
            "same_rule_cross_block_parent_cells_schur_edge_balance_"
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
        "broad_policy_audit_name": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--parent-scope-json", type=Path, default=DEFAULT_PARENT_SCOPE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def convert_value(value: str) -> Any:
    if value in {"none", "None"}:
        return None
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_policy_line(line: str) -> dict[str, Any] | None:
    marker = "diagnostic=cut_volume_direct_pspg_topology_policy"
    if marker not in line:
        return None
    payload = line[line.index("diagnostic=") :]
    record: dict[str, Any] = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        record[key] = convert_value(value)
    return record


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def values(record: dict[str, Any] | None) -> dict[str, Any]:
    return record if isinstance(record, dict) else {}


def worst_update(pressure_audit: dict[str, Any] | None) -> dict[str, Any] | None:
    worst = values(pressure_audit).get("worst_by_category")
    if not isinstance(worst, dict):
        return None
    active = worst.get("active_or_wet_supported")
    if not isinstance(active, dict):
        return None
    return {
        "abs_pressure_delta_pa": active.get("abs_pressure_delta_pa"),
        "point_index": active.get("point_index"),
        "support_class": active.get("support_class"),
        "active_fluid": active.get("active_fluid"),
        "incident_wet_fraction_min_positive": active.get(
            "incident_wet_fraction_min_positive"
        ),
    }


def pressure_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    worst = worst_update(data)
    return {
        "exists": path.exists(),
        "path": str(path),
        "status": values(data).get("status"),
        "finding": values(data).get("finding"),
        "absolute_threshold_pa": values(data).get("absolute_threshold_pa"),
        "triggered_transition_count": values(data).get(
            "triggered_transition_count"
        ),
        "guard_triggered": (
            values(data).get("status") == "diagnostic_pressure_update_guard_triggered"
        ),
        "guard_cleared": (
            values(data).get("status")
            == "diagnostic_pressure_update_guard_no_threshold_trigger"
        ),
        "worst_active_or_wet": worst,
        "worst_active_or_wet_update_pa": (
            worst.get("abs_pressure_delta_pa") if isinstance(worst, dict) else None
        ),
        "worst_active_or_wet_support_class": (
            worst.get("support_class") if isinstance(worst, dict) else None
        ),
    }


def parse_policy_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_policy_line(line)
        if record is not None:
            records.append(record)
    return records


def policy_log_summary(path: Path) -> dict[str, Any]:
    records = parse_policy_log(path)
    parent_cells = sorted(
        {
            int(record["parent_cell"])
            for record in records
            if isinstance(record.get("parent_cell"), int)
        }
    )
    parent_count_values: Counter[int] = Counter(
        int(record["parent_filter_parent_cell_count"])
        for record in records
        if isinstance(record.get("parent_filter_parent_cell_count"), int)
    )
    row_filter_values = sorted(
        {
            record.get("row_filter_enabled")
            for record in records
            if record.get("row_filter_enabled") is not None
        }
    )
    parent_filter_values = sorted(
        {
            record.get("parent_filter_enabled")
            for record in records
            if record.get("parent_filter_enabled") is not None
        }
    )
    return {
        "exists": path.exists(),
        "path": str(path),
        "record_count": len(records),
        "matrix_mutated_count": sum(
            1 for record in records if record.get("matrix_mutated") == 1
        ),
        "solve_affecting_count": sum(
            1 for record in records if record.get("solve_affecting") == 1
        ),
        "parent_filter_enabled_values": parent_filter_values,
        "parent_filter_enabled_count": sum(
            1 for record in records if record.get("parent_filter_enabled") == 1
        ),
        "parent_filter_parent_cell_count_values": sorted(parent_count_values),
        "parent_filter_selected_count": sum(
            1 for record in records if record.get("parent_filter_selected") == 1
        ),
        "row_filter_enabled_values": row_filter_values,
        "row_filter_enabled_count": sum(
            1 for record in records if record.get("row_filter_enabled") == 1
        ),
        "row_filter_global_dof_count_values": sorted(
            {
                record.get("row_filter_global_dof_count")
                for record in records
                if record.get("row_filter_global_dof_count") is not None
            }
        ),
        "unique_parent_cell_count": len(parent_cells),
        "parent_cells": parent_cells,
        "policies_seen": sorted(
            {
                str(record["policy"])
                for record in records
                if isinstance(record.get("policy"), str)
            }
        ),
    }


def safe_delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)


def safe_ratio(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    if float(right) == 0.0:
        return None
    return float(left) / float(right)


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
    parent_scope: dict[str, Any] | None,
) -> dict[str, Any]:
    scope = scope_case(parent_scope, label)
    replay_audit = pressure_summary(artifact_root / spec["audit_name"])
    baseline_audit = pressure_summary(artifact_root / spec["baseline_audit_name"])
    row_filter_audit = pressure_summary(artifact_root / spec["row_filter_audit_name"])
    broad_audit = pressure_summary(artifact_root / spec["broad_policy_audit_name"])
    replay_update = replay_audit["worst_active_or_wet_update_pa"]
    baseline_update = baseline_audit["worst_active_or_wet_update_pa"]
    row_filter_update = row_filter_audit["worst_active_or_wet_update_pa"]
    broad_update = broad_audit["worst_active_or_wet_update_pa"]
    policy_log = policy_log_summary(
        artifact_root / spec["case_dir"] / spec["log_name"]
    )
    expected_parent_cell_count = scope.get("parent_cell_count")
    return {
        "label": label,
        "case_dir": str(artifact_root / spec["case_dir"]),
        "solver_log_path": str(artifact_root / spec["case_dir"] / spec["log_name"]),
        "parent_scope": {
            "exists": bool(scope),
            "expected_parent_cell_count": expected_parent_cell_count,
            "candidate_row_count": scope.get("candidate_row_count"),
            "parent_expanded_row_count": scope.get("parent_expanded_row_count"),
            "parent_expanded_to_candidate_ratio": scope.get(
                "parent_expanded_to_candidate_ratio"
            ),
            "ready_for_parent_cell_replay": scope.get(
                "ready_for_parent_cell_replay"
            ),
        },
        "pressure_update": replay_audit,
        "baseline_pressure_update": baseline_audit,
        "row_filter_pressure_update": row_filter_audit,
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
        "improvement_vs_broad_policy_pa": safe_delta(broad_update, replay_update),
        "replay_to_broad_policy_update_ratio": safe_ratio(
            replay_update, broad_update
        ),
    }


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    parent_scope_json: Path = DEFAULT_PARENT_SCOPE,
) -> dict[str, Any]:
    parent_scope = load_json(parent_scope_json)
    cases = [
        summarize_case(
            label=label,
            spec=spec,
            artifact_root=artifact_root,
            parent_scope=parent_scope,
        )
        for label, spec in REPLAYS.items()
    ]
    missing = []
    if not parent_scope_json.exists():
        missing.append(str(parent_scope_json))
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
    improves_baseline = all(
        isinstance(case.get("improvement_vs_baseline_pa"), (int, float))
        and case["improvement_vs_baseline_pa"] > 0.0
        for case in cases
    )
    improves_row_filter = all(
        isinstance(case.get("improvement_vs_row_filter_pa"), (int, float))
        and case["improvement_vs_row_filter_pa"] > 0.0
        for case in cases
    )

    if missing:
        finding = "direct_pspg_same_rule_cross_block_parent_cell_replays_incomplete"
        status = "regenerate_missing_parent_cell_replay_artifacts"
        conclusion = (
            "At least one same-rule cross-block parent-cell replay, pressure "
            "audit, solver log, or parent-scope artifact is missing."
        )
    elif guard_cleared and not guard_triggered:
        finding = (
            "direct_pspg_same_rule_cross_block_parent_cell_replays_clear_guards"
        )
        status = "same_rule_cross_block_parent_cell_replay_clears_short_windows"
        conclusion = (
            "The same-rule parent-cell replay clears both short-window active/"
            "wet pressure-update guards."
        )
    elif len(guard_triggered) == len(cases):
        finding = (
            "direct_pspg_same_rule_cross_block_parent_cell_replays_do_not_clear_guards"
        )
        status = "same_rule_cross_block_parent_cell_replay_insufficient"
        conclusion = (
            "The parent-cell replay improves the no-policy and row-filter "
            "baselines in both short windows, but still triggers both active/"
            "wet pressure-update guards. Parent-scoped full-local mutation is "
            "directionally useful but not a sufficient formulation fix."
        )
    else:
        finding = "direct_pspg_same_rule_cross_block_parent_cell_replays_mixed"
        status = "inspect_parent_cell_replay_statuses"
        conclusion = "Same-rule parent-cell replay statuses are mixed."

    return {
        "scope": (
            "Targeted Test02 step382 and Test10 step90 solve-time direct PSPG "
            "local_schur_edge_balance replays restricted to parent cells "
            "derived from the exported same-rule cross-block candidate rows."
        ),
        "finding": finding,
        "status": status,
        "parent_scope_artifact": str(parent_scope_json),
        "parent_scope_finding": (
            parent_scope.get("finding") if isinstance(parent_scope, dict) else None
        ),
        "parent_filters_match_scope_counts": parent_filters_match,
        "row_filters_disabled": row_filters_disabled,
        "all_replays_improve_no_policy_baseline": improves_baseline,
        "all_replays_improve_row_filter_replay": improves_row_filter,
        "all_replays_trigger_guard": len(guard_triggered) == len(cases)
        and not missing,
        "cleared_cases": [case["label"] for case in guard_cleared],
        "triggered_cases": [case["label"] for case in guard_triggered],
        "cases": cases,
        "conclusion": conclusion,
        "next_requirement": (
            "Do not promote parent-cell replay as a fix. Use the improvement "
            "over the row-list replay as evidence that some broader local "
            "support coupling is relevant, then derive a physical predicate "
            "that supplies the missing effect without broad unstable mutation."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        parent_scope_json=args.parent_scope_json,
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
