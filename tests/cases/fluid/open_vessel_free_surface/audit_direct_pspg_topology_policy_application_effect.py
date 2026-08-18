#!/usr/bin/env python3
"""Summarize direct PSPG topology-policy application and pressure effects."""

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
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_application_effect_20260607.json"
)

POLICIES = [
    "local_schur_completion",
    "local_edge_balance",
    "local_schur_edge_balance",
]

BROAD_REPLAYS = [
    {
        "label": "test02",
        "variant": "broad_policy",
        "policy": "local_schur_completion",
        "case_dir": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_local_schur_completion_20260607_case"
        ),
        "log_name": "run_direct_pspg_topology_policy_local_schur_completion.log",
        "audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_local_schur_completion_pressure_update_audit_"
            "20260607.json"
        ),
    },
    {
        "label": "test02",
        "variant": "broad_policy",
        "policy": "local_edge_balance",
        "case_dir": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_local_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_topology_policy_local_edge_balance.log",
        "audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_local_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
    {
        "label": "test02",
        "variant": "broad_policy",
        "policy": "local_schur_edge_balance",
        "case_dir": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_schur_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_topology_policy_schur_edge_balance.log",
        "audit_name": (
            "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
            "topology_policy_schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
    {
        "label": "test10",
        "variant": "broad_policy",
        "policy": "local_schur_completion",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "local_schur_completion_20260607_case"
        ),
        "log_name": "run_direct_pspg_topology_policy_local_schur_completion.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "local_schur_completion_pressure_update_audit_20260607.json"
        ),
    },
    {
        "label": "test10",
        "variant": "broad_policy",
        "policy": "local_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "local_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_topology_policy_local_edge_balance.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "local_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
    {
        "label": "test10",
        "variant": "broad_policy",
        "policy": "local_schur_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "schur_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_topology_policy_schur_edge_balance.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_topology_policy_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
]

SIGNATURE_ROW_REPLAYS = [
    {
        "label": "test10",
        "variant": "signature_row_filter",
        "policy": "local_schur_completion",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_schur_completion_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_local_schur_completion.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_schur_completion_pressure_update_audit_20260607.json"
        ),
    },
    {
        "label": "test10",
        "variant": "signature_row_filter",
        "policy": "local_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_local_edge_balance.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
    {
        "label": "test10",
        "variant": "signature_row_filter",
        "policy": "local_schur_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_schur_edge_balance.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
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


def parse_key_values(line: str) -> dict[str, Any]:
    if "diagnostic=cut_volume_direct_pspg_topology_policy" not in line:
        return {}
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


def values(record: Any) -> dict[str, Any]:
    return record if isinstance(record, dict) else {}


def worst_category(
    pressure_audit: dict[str, Any] | None,
    category: str,
) -> dict[str, Any]:
    worst_by_category = values(pressure_audit).get("worst_by_category")
    worst = values(worst_by_category).get(category)
    if not isinstance(worst, dict):
        return {}
    return {
        "abs_pressure_delta_pa": worst.get("abs_pressure_delta_pa"),
        "point_index": worst.get("point_index"),
        "support_class": worst.get("support_class"),
        "active_fluid": worst.get("active_fluid"),
        "incident_wet_fraction_min_positive": worst.get(
            "incident_wet_fraction_min_positive"
        ),
    }


def pressure_summary(pressure_audit: dict[str, Any] | None) -> dict[str, Any]:
    audit = values(pressure_audit)
    active_wet = worst_category(pressure_audit, "active_or_wet_supported")
    return {
        "status": audit.get("status"),
        "finding": audit.get("finding"),
        "absolute_threshold_pa": audit.get("absolute_threshold_pa"),
        "triggered_transition_count": audit.get("triggered_transition_count"),
        "active_or_wet_worst": active_wet,
        "full_wet_worst": worst_category(pressure_audit, "full_wet_supported"),
        "cut_supported_worst": worst_category(pressure_audit, "cut_supported"),
        "guard_triggered": (
            audit.get("status") == "diagnostic_pressure_update_guard_triggered"
        ),
        "guard_cleared": (
            audit.get("status")
            == "diagnostic_pressure_update_guard_no_threshold_trigger"
        ),
        "worst_active_or_wet_update_pa": active_wet.get("abs_pressure_delta_pa"),
        "worst_active_or_wet_point_index": active_wet.get("point_index"),
        "worst_active_or_wet_support_class": active_wet.get("support_class"),
    }


def numeric(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def topology_log_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"exists": False, "policy_log_count": 0}

    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_key_values(line)
        if record:
            records.append(record)

    row_filter_global_counts: Counter[int] = Counter()
    row_filter_selected_counts: Counter[int] = Counter()
    policies: Counter[str] = Counter()
    row_filter_enabled_count = 0
    for record in records:
        if isinstance(record.get("policy"), str):
            policies[str(record["policy"])] += 1
        if record.get("row_filter_enabled") == 1:
            row_filter_enabled_count += 1
        if isinstance(record.get("row_filter_global_dof_count"), int):
            row_filter_global_counts[int(record["row_filter_global_dof_count"])] += 1
        if isinstance(record.get("row_filter_selected_local_row_count"), int):
            row_filter_selected_counts[
                int(record["row_filter_selected_local_row_count"])
            ] += 1

    matrix_mutated_count = sum(1 for record in records if record.get("matrix_mutated") == 1)
    selected_records = [
        record
        for record in records
        if numeric(record, "row_filter_selected_local_row_count") > 0.0
    ]
    selected_mutated_count = sum(
        1 for record in selected_records if record.get("matrix_mutated") == 1
    )
    return {
        "exists": True,
        "policy_log_count": len(records),
        "policies_seen": {key: policies[key] for key in sorted(policies)},
        "row_filter_enabled_count": row_filter_enabled_count,
        "row_filter_global_dof_counts": {
            str(key): row_filter_global_counts[key]
            for key in sorted(row_filter_global_counts)
        },
        "row_filter_selected_local_row_counts": {
            str(key): row_filter_selected_counts[key]
            for key in sorted(row_filter_selected_counts)
        },
        "row_filter_selected_local_row_sum": sum(
            int(record.get("row_filter_selected_local_row_count") or 0)
            for record in records
        ),
        "row_filter_selected_record_count": len(selected_records),
        "selected_records_matrix_mutated_count": selected_mutated_count,
        "selected_records_without_mutation_count": (
            len(selected_records) - selected_mutated_count
        ),
        "matrix_mutated_count": matrix_mutated_count,
        "matrix_mutated_fraction": (
            matrix_mutated_count / len(records) if records else None
        ),
        "touched_row_count_sum": sum(numeric(record, "touched_row_count") for record in records),
        "balance_candidate_row_count_sum": sum(
            numeric(record, "balance_candidate_row_count") for record in records
        ),
        "schur_contribution_count_sum": sum(
            numeric(record, "schur_contribution_count") for record in records
        ),
        "max_delta_weight": max(
            (numeric(record, "max_delta_weight") for record in records),
            default=0.0,
        ),
        "max_row_abs_delta": max(
            (numeric(record, "max_row_abs_delta") for record in records),
            default=0.0,
        ),
        "full_cell_record_count": sum(1 for record in records if record.get("full_cell") == 1),
        "cut_cell_record_count": sum(1 for record in records if record.get("full_cell") == 0),
    }


def summarize_replay(artifact_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    audit_path = artifact_root / spec["audit_name"]
    log_path = artifact_root / spec["case_dir"] / spec["log_name"]
    pressure = pressure_summary(load_json(audit_path))
    return {
        "label": spec["label"],
        "variant": spec["variant"],
        "policy": spec["policy"],
        "pressure_audit_path": str(audit_path),
        "pressure_audit_exists": audit_path.exists(),
        "solver_log_path": str(log_path),
        "topology_log": topology_log_summary(log_path),
        "pressure_update": pressure,
        "guard_triggered": pressure["guard_triggered"],
        "guard_cleared": pressure["guard_cleared"],
        "worst_active_or_wet_update_pa": pressure[
            "worst_active_or_wet_update_pa"
        ],
        "worst_active_or_wet_point_index": pressure[
            "worst_active_or_wet_point_index"
        ],
        "worst_active_or_wet_support_class": pressure[
            "worst_active_or_wet_support_class"
        ],
    }


def replay_key(replay: dict[str, Any]) -> tuple[str, str, str]:
    return (replay["label"], replay["variant"], replay["policy"])


def by_case_variant_policy(
    replays: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {replay_key(replay): replay for replay in replays}


def best_by_update(replays: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        replay
        for replay in replays
        if isinstance(replay["worst_active_or_wet_update_pa"], (int, float))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["worst_active_or_wet_update_pa"])


def compare_test10_broad_and_signature(
    lookup: dict[tuple[str, str, str], dict[str, Any]]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for policy in POLICIES:
        broad = lookup.get(("test10", "broad_policy", policy))
        filtered = lookup.get(("test10", "signature_row_filter", policy))
        if broad is None or filtered is None:
            continue
        broad_update = broad.get("worst_active_or_wet_update_pa")
        filtered_update = filtered.get("worst_active_or_wet_update_pa")
        comparisons[policy] = {
            "broad_update_pa": broad_update,
            "signature_row_filter_update_pa": filtered_update,
            "signature_minus_broad_update_pa": (
                filtered_update - broad_update
                if isinstance(broad_update, (int, float))
                and isinstance(filtered_update, (int, float))
                else None
            ),
            "broad_policy_log_count": broad["topology_log"].get("policy_log_count"),
            "signature_policy_log_count": filtered["topology_log"].get(
                "policy_log_count"
            ),
            "broad_matrix_mutated_count": broad["topology_log"].get(
                "matrix_mutated_count"
            ),
            "signature_matrix_mutated_count": filtered["topology_log"].get(
                "matrix_mutated_count"
            ),
            "signature_selected_records_matrix_mutated_count": filtered[
                "topology_log"
            ].get("selected_records_matrix_mutated_count"),
        }
    return comparisons


def case_policy_matrix(replays: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for replay in replays:
        case = replay["label"]
        variant = replay["variant"]
        policy = replay["policy"]
        matrix.setdefault(case, {}).setdefault(variant, {})[policy] = {
            "guard_triggered": replay["guard_triggered"],
            "worst_active_or_wet_update_pa": replay[
                "worst_active_or_wet_update_pa"
            ],
            "worst_active_or_wet_support_class": replay[
                "worst_active_or_wet_support_class"
            ],
            "policy_log_count": replay["topology_log"].get("policy_log_count"),
            "matrix_mutated_count": replay["topology_log"].get(
                "matrix_mutated_count"
            ),
            "max_row_abs_delta": replay["topology_log"].get("max_row_abs_delta"),
            "row_filter_selected_record_count": replay["topology_log"].get(
                "row_filter_selected_record_count"
            ),
        }
    return matrix


def build_report(*, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    replays = [
        summarize_replay(artifact_root, spec)
        for spec in BROAD_REPLAYS + SIGNATURE_ROW_REPLAYS
    ]
    missing = [
        replay["pressure_audit_path"]
        for replay in replays
        if not replay["pressure_audit_exists"]
    ] + [
        replay["solver_log_path"]
        for replay in replays
        if not replay["topology_log"].get("exists")
    ]
    lookup = by_case_variant_policy(replays)
    test10_comparisons = compare_test10_broad_and_signature(lookup)
    test10_signature = [
        replay for replay in replays if replay["variant"] == "signature_row_filter"
    ]
    test10_signature_mutating = [
        replay
        for replay in test10_signature
        if replay["topology_log"].get("selected_records_matrix_mutated_count", 0) > 0
    ]
    all_complete = not missing
    all_trigger = all(replay["guard_triggered"] for replay in replays) and all_complete
    broad_test02 = [
        replay
        for replay in replays
        if replay["label"] == "test02" and replay["variant"] == "broad_policy"
    ]
    broad_test10 = [
        replay
        for replay in replays
        if replay["label"] == "test10" and replay["variant"] == "broad_policy"
    ]
    best_test10_broad = best_by_update(broad_test10)
    best_test10_signature = best_by_update(test10_signature)
    best_test02_broad = best_by_update(broad_test02)

    if missing:
        finding = "direct_pspg_topology_policy_application_effect_incomplete"
        status = "regenerate_missing_policy_application_evidence"
        conclusion = (
            "At least one pressure-update audit or topology-policy solver log is "
            "missing."
        )
    elif all_trigger and len(test10_signature_mutating) == len(test10_signature):
        finding = (
            "direct_pspg_topology_policy_application_effect_rules_out_"
            "underapplication"
        )
        status = "local_matrix_policy_applies_but_is_not_sufficient_fix"
        conclusion = (
            "Solve-affecting direct PSPG topology policies mutate the tagged "
            "production matrix, including the targeted Test10 48-row signature "
            "filter, but every broad and filtered replay still triggers the "
            "active/wet pressure-update guard."
        )
    elif all_trigger:
        finding = (
            "direct_pspg_topology_policy_application_effect_local_mutation_"
            "insufficient"
        )
        status = "local_matrix_policy_does_not_clear_guards"
        conclusion = (
            "All policy replays still trigger the pressure-update guards; at "
            "least one filtered replay has limited local mutation coverage."
        )
    else:
        finding = "direct_pspg_topology_policy_application_effect_mixed"
        status = "inspect_policy_application_effects"
        conclusion = "Policy replay statuses or mutation effects are mixed."

    return {
        "scope": (
            "Compare solve-affecting direct PSPG topology-policy application "
            "logs against active/wet pressure-update outcomes for Test02 broad "
            "local modes, Test10 broad local modes, and the targeted Test10 "
            "48-row support/coupling-signature filter."
        ),
        "finding": finding,
        "status": status,
        "policies_tested": POLICIES,
        "all_replays_trigger_guard": all_trigger,
        "all_test10_signature_replays_mutate_selected_records": (
            len(test10_signature_mutating) == len(test10_signature) and all_complete
        ),
        "best_test02_broad_policy": (
            None if best_test02_broad is None else best_test02_broad["policy"]
        ),
        "best_test02_broad_update_pa": (
            None
            if best_test02_broad is None
            else best_test02_broad["worst_active_or_wet_update_pa"]
        ),
        "best_test10_broad_policy": (
            None if best_test10_broad is None else best_test10_broad["policy"]
        ),
        "best_test10_broad_update_pa": (
            None
            if best_test10_broad is None
            else best_test10_broad["worst_active_or_wet_update_pa"]
        ),
        "best_test10_signature_policy": (
            None if best_test10_signature is None else best_test10_signature["policy"]
        ),
        "best_test10_signature_update_pa": (
            None
            if best_test10_signature is None
            else best_test10_signature["worst_active_or_wet_update_pa"]
        ),
        "test10_broad_vs_signature_row_filter": test10_comparisons,
        "case_policy_matrix": case_policy_matrix(replays),
        "replays": replays,
        "missing_evidence": missing,
        "conclusion": conclusion,
        "next_requirement": (
            "Do not treat the current local topology-policy failure as a hook "
            "execution or row-filter coverage issue. The next candidate must "
            "change the direct PSPG formulation-side support/coupling behavior "
            "rather than replaying small local matrix deltas on selected rows."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(artifact_root=args.artifact_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
