#!/usr/bin/env python3
"""Audit the direct PSPG signature-parent full-local replay result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_READINESS = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_parent_subset_replay_"
    "readiness_20260607.json"
)
DEFAULT_PARENT_SCOPE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_parent_scope_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test10_direct_pspg_topology_policy_parent_subset_replay_20260607.json"
)

REPLAYS = {
    "same_case_no_policy": {
        "label": "same_case_no_policy",
        "case_dir": "test10_replay_cap3_step90_pspg_wall_full_gradient_scale1_coverage_20260606_case",
        "audit_name": "test10_replay_cap3_step90_pspg_wall_full_gradient_scale1_coverage_pressure_update_audit_20260606.json",
    },
    "broad_policy": {
        "label": "broad_policy",
        "case_dir": "test10_replay_cap3_step90_direct_pspg_topology_policy_schur_edge_balance_20260607_case",
        "log_name": "run_direct_pspg_topology_policy_schur_edge_balance.log",
        "audit_name": "test10_replay_cap3_step90_direct_pspg_topology_policy_schur_edge_balance_pressure_update_audit_20260607.json",
    },
    "signature_row_filter": {
        "label": "signature_row_filter",
        "case_dir": "test10_replay_cap3_step90_direct_pspg_signature_rows_schur_edge_balance_20260607_case",
        "log_name": "run_direct_pspg_signature_rows_schur_edge_balance.log",
        "audit_name": "test10_replay_cap3_step90_direct_pspg_signature_rows_schur_edge_balance_pressure_update_audit_20260607.json",
    },
    "signature_parent_filter": {
        "label": "signature_parent_filter",
        "case_dir": "test10_replay_cap3_step90_direct_pspg_parent_cells_schur_edge_balance_20260607_case",
        "log_name": "run_direct_pspg_parent_cells_schur_edge_balance.log",
        "audit_name": "test10_replay_cap3_step90_direct_pspg_parent_cells_schur_edge_balance_pressure_update_audit_20260607.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--readiness-json", type=Path, default=DEFAULT_READINESS)
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
    result: dict[str, Any] = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = convert_value(value)
    return result


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pressure_summary(path: Path) -> dict[str, Any]:
    data = load_json(path) or {}
    transitions = data.get("transitions")
    first_transition = transitions[0] if isinstance(transitions, list) and transitions else {}
    max_by_category = first_transition.get("max_by_category", {})
    if not isinstance(max_by_category, dict):
        max_by_category = {}
    active_wet = max_by_category.get("active_or_wet_supported", {})
    if not isinstance(active_wet, dict):
        active_wet = {}
    stats_by_category = first_transition.get("delta_statistics_by_category", {})
    if not isinstance(stats_by_category, dict):
        stats_by_category = {}
    return {
        "exists": path.exists(),
        "path": str(path),
        "status": data.get("status"),
        "finding": data.get("finding"),
        "absolute_threshold_pa": data.get("absolute_threshold_pa"),
        "guard_triggered": (
            data.get("status") == "diagnostic_pressure_update_guard_triggered"
        ),
        "worst_active_or_wet_update_pa": active_wet.get("abs_pressure_delta_pa"),
        "worst_active_or_wet_point_index": active_wet.get("point_index"),
        "worst_active_or_wet_support_class": active_wet.get("support_class"),
        "worst_active_or_wet_pressure_delta_pa": active_wet.get("pressure_delta_pa"),
        "delta_statistics_by_category": stats_by_category,
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


def numeric(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def policy_log_summary(path: Path) -> dict[str, Any]:
    records = parse_policy_log(path)
    parent_cells = sorted(
        {
            int(record["parent_cell"])
            for record in records
            if isinstance(record.get("parent_cell"), int)
        }
    )
    row_filter_enabled_values = sorted(
        {
            record.get("row_filter_enabled")
            for record in records
            if record.get("row_filter_enabled") is not None
        }
    )
    parent_filter_enabled_values = sorted(
        {
            record.get("parent_filter_enabled")
            for record in records
            if record.get("parent_filter_enabled") is not None
        }
    )
    parent_filter_parent_cell_count_values = sorted(
        {
            record.get("parent_filter_parent_cell_count")
            for record in records
            if record.get("parent_filter_parent_cell_count") is not None
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
        "row_filter_enabled_values": row_filter_enabled_values,
        "row_filter_enabled_count": sum(
            1 for record in records if record.get("row_filter_enabled") == 1
        ),
        "parent_filter_enabled_values": parent_filter_enabled_values,
        "parent_filter_enabled_count": sum(
            1 for record in records if record.get("parent_filter_enabled") == 1
        ),
        "parent_filter_parent_cell_count_values": (
            parent_filter_parent_cell_count_values
        ),
        "parent_filter_selected_count": sum(
            1 for record in records if record.get("parent_filter_selected") == 1
        ),
        "unique_parent_cell_count": len(parent_cells),
        "parent_cells": parent_cells,
        "source_edge_weight_sum": sum(
            numeric(record, "source_edge_weight_sum") for record in records
        ),
        "topology_edge_count_sum": sum(
            numeric(record, "topology_edge_count") for record in records
        ),
        "topology_edge_weight_sum": sum(
            numeric(record, "topology_edge_weight_sum") for record in records
        ),
        "touched_row_count_sum": sum(
            numeric(record, "touched_row_count") for record in records
        ),
        "selected_local_row_count_sum": sum(
            numeric(record, "row_filter_selected_local_row_count")
            for record in records
        ),
        "balance_candidate_row_count_sum": sum(
            numeric(record, "balance_candidate_row_count") for record in records
        ),
    }


def replay_summary(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    case_dir = root / spec["case_dir"]
    summary = {
        "label": spec["label"],
        "case_dir": str(case_dir),
        "pressure_update": pressure_summary(root / spec["audit_name"]),
    }
    log_name = spec.get("log_name")
    if log_name:
        summary["policy_log"] = policy_log_summary(case_dir / log_name)
    return summary


def safe_subtract(left: float | None, right: float | None) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)


def update_value(summary: dict[str, Any]) -> float | None:
    value = summary["pressure_update"].get("worst_active_or_wet_update_pa")
    return float(value) if isinstance(value, (int, float)) else None


def parent_scope_summary(parent_scope: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parent_scope, dict):
        return {
            "exists": False,
            "finding": None,
            "status": None,
            "combined_rule_scope": None,
        }
    combined = (
        parent_scope.get("test10_parent_rule_scope", {})
        .get("local_schur_edge_balance", {})
        .get("rule_scope")
    )
    return {
        "exists": True,
        "finding": parent_scope.get("finding"),
        "status": parent_scope.get("status"),
        "combined_rule_scope": combined if isinstance(combined, dict) else None,
    }


def readiness_summary(readiness: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(readiness, dict):
        return {
            "exists": False,
            "finding": None,
            "status": None,
            "signature_parent_cell_count": None,
            "signature_parent_cell_ranges": None,
        }
    return {
        "exists": True,
        "finding": readiness.get("finding"),
        "status": readiness.get("status"),
        "source_hook": readiness.get("source_hook"),
        "same_signature_parent_set_all_policies": readiness.get(
            "same_signature_parent_set_all_policies"
        ),
        "signature_parent_cell_count": readiness.get("signature_parent_cell_count"),
        "signature_parent_cell_ranges": readiness.get("signature_parent_cell_ranges"),
    }


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    readiness_json: Path = DEFAULT_READINESS,
    parent_scope_json: Path = DEFAULT_PARENT_SCOPE,
) -> dict[str, Any]:
    replay_summaries = {
        label: replay_summary(artifact_root, spec)
        for label, spec in REPLAYS.items()
    }
    readiness = load_json(readiness_json)
    parent_scope = load_json(parent_scope_json)
    missing = []
    if not readiness_json.exists():
        missing.append(str(readiness_json))
    if not parent_scope_json.exists():
        missing.append(str(parent_scope_json))
    for summary in replay_summaries.values():
        if not summary["pressure_update"]["exists"]:
            missing.append(summary["pressure_update"]["path"])
        policy_log = summary.get("policy_log")
        if isinstance(policy_log, dict) and not policy_log["exists"]:
            missing.append(policy_log["path"])

    parent_policy = replay_summaries["signature_parent_filter"].get(
        "policy_log", {}
    )
    parent_update = update_value(replay_summaries["signature_parent_filter"])
    broad_update = update_value(replay_summaries["broad_policy"])
    signature_row_update = update_value(replay_summaries["signature_row_filter"])
    no_policy_update = update_value(replay_summaries["same_case_no_policy"])
    parent_guard_triggered = replay_summaries["signature_parent_filter"][
        "pressure_update"
    ]["guard_triggered"]
    parent_filter_full_local = (
        parent_policy.get("record_count") == 264
        and parent_policy.get("matrix_mutated_count") == 264
        and parent_policy.get("row_filter_enabled_values") == [0]
        and parent_policy.get("parent_filter_enabled_values") == [1]
        and parent_policy.get("parent_filter_parent_cell_count_values") == [264]
        and parent_policy.get("parent_filter_selected_count") == 264
        and parent_policy.get("selected_local_row_count_sum") == 1056.0
    )

    if missing:
        finding = "direct_pspg_signature_parent_subset_full_local_replay_incomplete"
        status = "regenerate_missing_parent_subset_replay_inputs"
        conclusion = (
            "At least one replay log, pressure-update audit, readiness artifact, "
            "or parent-scope artifact is missing."
        )
    elif parent_guard_triggered:
        finding = (
            "direct_pspg_signature_parent_subset_full_local_replay_"
            "does_not_clear_test10_guard"
        )
        status = "exact_parent_subset_ruled_out_as_sufficient_fix"
        conclusion = (
            "The full-local mutation on the 264 signature parent cells still "
            "triggers the Test10 active/wet pressure-update guard, so exact "
            "signature-parent support is not a sufficient fix."
        )
    else:
        finding = (
            "direct_pspg_signature_parent_subset_full_local_replay_"
            "clears_test10_guard"
        )
        status = "requires_test02_transfer_check"
        conclusion = (
            "The full-local signature-parent replay clears Test10 and needs a "
            "Test02 transfer check before it can be treated as a candidate fix."
        )

    next_requirement = (
        "Move away from exact-row or exact-parent replay of the current local "
        "matrix deltas and test a physical support-patch closure that can "
        "handle Test10's coherent full-wet boundary mode without shifting "
        "Test02 into the tiny-cut branch."
        if parent_guard_triggered
        else (
            "Run the same parent-cell full-local policy on the matching Test02 "
            "short replay before promoting it beyond Test10 evidence."
        )
    )

    return {
        "finding": finding,
        "status": status,
        "conclusion": conclusion,
        "artifact_root": str(artifact_root),
        "missing_evidence": missing,
        "readiness": readiness_summary(readiness),
        "parent_scope": parent_scope_summary(parent_scope),
        "policy": "local_schur_edge_balance",
        "replays": replay_summaries,
        "signature_parent_filter_full_local_confirmed": parent_filter_full_local,
        "signature_parent_filter_update_pa": parent_update,
        "same_case_no_policy_update_pa": no_policy_update,
        "broad_policy_update_pa": broad_update,
        "signature_row_filter_update_pa": signature_row_update,
        "parent_minus_broad_update_pa": safe_subtract(parent_update, broad_update),
        "parent_minus_signature_row_update_pa": safe_subtract(
            parent_update,
            signature_row_update,
        ),
        "parent_minus_no_policy_update_pa": safe_subtract(parent_update, no_policy_update),
        "pressure_update_guard_cleared": {
            label: not summary["pressure_update"]["guard_triggered"]
            for label, summary in replay_summaries.items()
        },
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        readiness_json=args.readiness_json,
        parent_scope_json=args.parent_scope_json,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
