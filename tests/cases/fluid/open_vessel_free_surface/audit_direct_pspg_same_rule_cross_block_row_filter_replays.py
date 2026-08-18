#!/usr/bin/env python3
"""Summarize same-rule cross-block direct PSPG row-filter replays."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_CANDIDATE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_same_rule_cross_block_signature_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_row_filter_replays_20260607.json"
)

FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

REPLAYS = {
    "test02": {
        "case_dir": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "same_rule_cross_block_rows_schur_edge_balance_20260607_case"
        ),
        "log_name": (
            "run_direct_pspg_same_rule_cross_block_rows_schur_edge_balance.log"
        ),
        "audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "same_rule_cross_block_rows_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "baseline_audit_name": (
            "test02_replay_abs_only_prune1e5_step382_"
            "pspg_wall_full_gradient_scale1_coverage_"
            "pressure_update_audit_20260606.json"
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
            "same_rule_cross_block_rows_schur_edge_balance_20260607_case"
        ),
        "log_name": (
            "run_direct_pspg_same_rule_cross_block_rows_schur_edge_balance.log"
        ),
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_"
            "same_rule_cross_block_rows_schur_edge_balance_"
            "pressure_update_audit_20260607.json"
        ),
        "baseline_audit_name": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_scale1_"
            "coverage_pressure_update_audit_20260606.json"
        ),
        "broad_policy_audit_name": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
        "prior_signature_row_filter_audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


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


def topology_log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "row_filter_log_count": 0,
            "matrix_mutated_count": 0,
            "row_filter_global_dof_counts": [],
            "row_filter_selected_local_row_counts": {},
            "policies_seen": {},
        }
    row_filter_log_count = 0
    matrix_mutated_count = 0
    global_counts: Counter[int] = Counter()
    selected_counts: Counter[int] = Counter()
    policies: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if (
            "diagnostic=cut_volume_direct_pspg_topology_policy" not in line
            or "row_filter_enabled=1" not in line
        ):
            continue
        row_filter_log_count += 1
        fields = dict(FIELD_RE.findall(line))
        if fields.get("matrix_mutated") == "1":
            matrix_mutated_count += 1
        if "policy" in fields:
            policies[fields["policy"]] += 1
        if "row_filter_global_dof_count" in fields:
            global_counts[int(fields["row_filter_global_dof_count"])] += 1
        if "row_filter_selected_local_row_count" in fields:
            selected_counts[int(fields["row_filter_selected_local_row_count"])] += 1
    return {
        "exists": True,
        "row_filter_log_count": row_filter_log_count,
        "matrix_mutated_count": matrix_mutated_count,
        "row_filter_global_dof_counts": sorted(global_counts),
        "row_filter_selected_local_row_counts": {
            str(key): selected_counts[key] for key in sorted(selected_counts)
        },
        "policies_seen": {key: policies[key] for key in sorted(policies)},
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


def candidate_counts(candidate: dict[str, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in values(candidate).get("cases", []):
        if not isinstance(case, dict) or not isinstance(case.get("label"), str):
            continue
        dofs = case.get("best_covering_composite_selected_global_dofs")
        if isinstance(dofs, list):
            counts[case["label"]] = len(dofs)
    return counts


def summarize_case(
    *,
    label: str,
    spec: dict[str, str],
    artifact_root: Path,
    expected_row_count: int | None,
) -> dict[str, Any]:
    replay_audit = pressure_summary(artifact_root / spec["audit_name"])
    baseline_audit = pressure_summary(artifact_root / spec["baseline_audit_name"])
    broad_audit = pressure_summary(artifact_root / spec["broad_policy_audit_name"])
    prior_signature_audit_name = spec.get("prior_signature_row_filter_audit_name")
    prior_signature_audit = (
        pressure_summary(artifact_root / prior_signature_audit_name)
        if prior_signature_audit_name
        else None
    )
    replay_update = replay_audit["worst_active_or_wet_update_pa"]
    baseline_update = baseline_audit["worst_active_or_wet_update_pa"]
    broad_update = broad_audit["worst_active_or_wet_update_pa"]
    prior_signature_update = (
        values(prior_signature_audit).get("worst_active_or_wet_update_pa")
        if prior_signature_audit is not None
        else None
    )
    return {
        "label": label,
        "expected_candidate_row_count": expected_row_count,
        "case_dir": str(artifact_root / spec["case_dir"]),
        "solver_log_path": str(artifact_root / spec["case_dir"] / spec["log_name"]),
        "pressure_update": replay_audit,
        "baseline_pressure_update": baseline_audit,
        "broad_policy_pressure_update": broad_audit,
        "prior_signature_row_filter_pressure_update": prior_signature_audit,
        "topology_log": topology_log_summary(
            artifact_root / spec["case_dir"] / spec["log_name"]
        ),
        "improvement_vs_baseline_pa": safe_delta(baseline_update, replay_update),
        "replay_to_baseline_update_ratio": safe_ratio(replay_update, baseline_update),
        "improvement_vs_broad_policy_pa": safe_delta(broad_update, replay_update),
        "replay_to_broad_policy_update_ratio": safe_ratio(replay_update, broad_update),
        "improvement_vs_prior_signature_pa": safe_delta(
            prior_signature_update, replay_update
        ),
        "replay_to_prior_signature_update_ratio": safe_ratio(
            replay_update, prior_signature_update
        ),
    }


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    candidate_json: Path = DEFAULT_CANDIDATE,
) -> dict[str, Any]:
    candidate = load_json(candidate_json)
    expected_counts = candidate_counts(candidate)
    cases = [
        summarize_case(
            label=label,
            spec=spec,
            artifact_root=artifact_root,
            expected_row_count=expected_counts.get(label),
        )
        for label, spec in REPLAYS.items()
    ]
    missing = [
        case["pressure_update"]["path"]
        for case in cases
        if not case["pressure_update"]["exists"]
    ] + [
        case["solver_log_path"]
        for case in cases
        if not case["topology_log"]["exists"]
    ]
    guard_triggered = [
        case for case in cases if case["pressure_update"]["guard_triggered"]
    ]
    guard_cleared = [
        case for case in cases if case["pressure_update"]["guard_cleared"]
    ]
    row_filters_match = all(
        case["expected_candidate_row_count"] in case["topology_log"].get(
            "row_filter_global_dof_counts", []
        )
        for case in cases
        if case["expected_candidate_row_count"] is not None
    )
    improves_baseline = all(
        isinstance(case.get("improvement_vs_baseline_pa"), (int, float))
        and case["improvement_vs_baseline_pa"] > 0.0
        for case in cases
    )

    if missing:
        finding = "direct_pspg_same_rule_cross_block_row_filter_replays_incomplete"
        status = "regenerate_missing_replay_artifacts"
        conclusion = (
            "At least one same-rule cross-block row-filter replay or pressure "
            "audit is missing."
        )
    elif guard_cleared and not guard_triggered:
        finding = "direct_pspg_same_rule_cross_block_row_filter_replays_clear_guards"
        status = "same_rule_cross_block_replay_candidate_clears_short_windows"
        conclusion = (
            "The same-rule cross-block candidate row-filter replay clears both "
            "short-window active/wet pressure-update guards."
        )
    elif len(guard_triggered) == len(cases):
        finding = (
            "direct_pspg_same_rule_cross_block_row_filter_replays_do_not_clear_guards"
        )
        status = "same_rule_cross_block_replay_insufficient"
        conclusion = (
            "The same-rule cross-block candidate row filter improves the "
            "same-case no-policy baseline in both short windows, but still "
            "triggers both active/wet pressure-update guards. It is therefore "
            "directionally relevant support/coupling evidence, not a sufficient "
            "formulation fix."
        )
    else:
        finding = "direct_pspg_same_rule_cross_block_row_filter_replays_mixed"
        status = "inspect_replay_statuses"
        conclusion = "Same-rule replay statuses are mixed or unclassified."

    return {
        "scope": (
            "Targeted Test02 step382 and Test10 step90 solve-time direct PSPG "
            "local_schur_edge_balance replays restricted to the exported "
            "same-rule cross-block candidate rows."
        ),
        "finding": finding,
        "status": status,
        "candidate_selectivity_artifact": str(candidate_json),
        "candidate_finding": values(candidate).get("finding"),
        "row_filters_match_candidate_counts": row_filters_match,
        "all_replays_improve_no_policy_baseline": improves_baseline,
        "all_replays_trigger_guard": len(guard_triggered) == len(cases)
        and not missing,
        "cleared_cases": [case["label"] for case in guard_cleared],
        "triggered_cases": [case["label"] for case in guard_triggered],
        "cases": cases,
        "conclusion": conclusion,
        "next_requirement": (
            "Do not promote the same-rule row list as a fix. Use the replay "
            "result to derive a formulation-side rule that keeps the helpful "
            "same-rule PP/PV coupling signal while adding the missing broader "
            "support/coupling mechanism, or test the next physical candidate "
            "against the same short-window guards."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        candidate_json=args.candidate_json,
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
