#!/usr/bin/env python3
"""Audit direct PSPG topology-policy scope and mutation scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_scope_scale_20260607.json"
)
DEFAULT_MODE_REPLAYS = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_mode_replays_20260607.json"
)

REPLAYS = [
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

POLICIES = [
    "local_schur_completion",
    "local_edge_balance",
    "local_schur_edge_balance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--mode-replays-json", type=Path, default=DEFAULT_MODE_REPLAYS)
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


def numeric(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def sum_metric(records: list[dict[str, Any]], key: str) -> float:
    return sum(numeric(record, key) for record in records)


def max_metric(records: list[dict[str, Any]], key: str) -> float:
    return max((numeric(record, key) for record in records), default=0.0)


def topology_log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "policy_log_count": 0,
            "matrix_mutated_count": 0,
            "row_filter_enabled_count": 0,
            "selected_record_count": 0,
            "selected_mutated_record_count": 0,
            "source_edge_weight_sum_total": 0.0,
            "topology_edge_weight_sum_total": 0.0,
            "touched_row_count_sum": 0.0,
            "schur_contribution_count_sum": 0.0,
            "balance_candidate_row_count_sum": 0.0,
            "max_delta_weight": 0.0,
            "max_row_abs_delta": 0.0,
            "full_cell_record_count": 0,
            "cut_cell_record_count": 0,
        }
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_policy_line(line)
        if record is not None:
            records.append(record)
    mutated = [record for record in records if record.get("matrix_mutated") == 1]
    selected = [
        record
        for record in records
        if numeric(record, "row_filter_selected_local_row_count") > 0.0
    ]
    return {
        "exists": True,
        "policy_log_count": len(records),
        "matrix_mutated_count": len(mutated),
        "row_filter_enabled_count": sum(
            1 for record in records if record.get("row_filter_enabled") == 1
        ),
        "selected_record_count": len(selected),
        "selected_mutated_record_count": sum(
            1 for record in selected if record.get("matrix_mutated") == 1
        ),
        "source_edge_weight_sum_total": sum_metric(
            records, "source_edge_weight_sum"
        ),
        "topology_edge_weight_sum_total": sum_metric(
            records, "topology_edge_weight_sum"
        ),
        "touched_row_count_sum": sum_metric(records, "touched_row_count"),
        "schur_contribution_count_sum": sum_metric(
            records, "schur_contribution_count"
        ),
        "balance_candidate_row_count_sum": sum_metric(
            records, "balance_candidate_row_count"
        ),
        "max_delta_weight": max_metric(records, "max_delta_weight"),
        "max_row_abs_delta": max_metric(records, "max_row_abs_delta"),
        "full_cell_record_count": sum(
            1 for record in records if record.get("full_cell") == 1
        ),
        "cut_cell_record_count": sum(
            1 for record in records if record.get("full_cell") == 0
        ),
    }


def pressure_summary(path: Path) -> dict[str, Any]:
    data = load_json(path) or {}
    worst_by_category = data.get("worst_by_category")
    if not isinstance(worst_by_category, dict):
        worst_by_category = {}
    active_wet = worst_by_category.get("active_or_wet_supported")
    if not isinstance(active_wet, dict):
        active_wet = {}
    return {
        "exists": path.exists(),
        "status": data.get("status"),
        "absolute_threshold_pa": data.get("absolute_threshold_pa"),
        "triggered_transition_count": data.get("triggered_transition_count"),
        "guard_triggered": (
            data.get("status") == "diagnostic_pressure_update_guard_triggered"
        ),
        "worst_active_or_wet_update_pa": active_wet.get(
            "abs_pressure_delta_pa"
        ),
        "worst_active_or_wet_point_index": active_wet.get("point_index"),
        "worst_active_or_wet_support_class": active_wet.get("support_class"),
    }


def summarize_replay(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    log_path = root / spec["case_dir"] / spec["log_name"]
    audit_path = root / spec["audit_name"]
    return {
        "label": spec["label"],
        "variant": spec["variant"],
        "policy": spec["policy"],
        "solver_log_path": str(log_path),
        "pressure_audit_path": str(audit_path),
        "topology": topology_log_summary(log_path),
        "pressure_update": pressure_summary(audit_path),
    }


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def replay_lookup(replays: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (replay["label"], replay["variant"], replay["policy"]): replay
        for replay in replays
    }


def same_case_no_policy_update(mode_replays: dict[str, Any] | None) -> float | None:
    if not isinstance(mode_replays, dict):
        return None
    for result in mode_replays.get("case_policy_results", []):
        if (
            isinstance(result, dict)
            and result.get("case") == "test10"
            and result.get("policy") == "local_schur_edge_balance"
            and isinstance(
                result.get("same_case_no_policy_worst_active_or_wet_update_pa"),
                (int, float),
            )
        ):
            return float(result["same_case_no_policy_worst_active_or_wet_update_pa"])
    return None


def compare_test10_scope(
    lookup: dict[tuple[str, str, str], dict[str, Any]],
    *,
    no_policy_update_pa: float | None,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for policy in POLICIES:
        broad = lookup.get(("test10", "broad_policy", policy))
        filtered = lookup.get(("test10", "signature_row_filter", policy))
        if broad is None or filtered is None:
            continue
        broad_topology = broad["topology"]
        filtered_topology = filtered["topology"]
        broad_update = broad["pressure_update"]["worst_active_or_wet_update_pa"]
        filtered_update = filtered["pressure_update"][
            "worst_active_or_wet_update_pa"
        ]
        comparisons[policy] = {
            "broad_update_pa": broad_update,
            "signature_row_filter_update_pa": filtered_update,
            "signature_minus_broad_update_pa": (
                filtered_update - broad_update
                if isinstance(filtered_update, (int, float))
                and isinstance(broad_update, (int, float))
                else None
            ),
            "no_policy_to_broad_improvement_pa": (
                no_policy_update_pa - broad_update
                if no_policy_update_pa is not None
                and isinstance(broad_update, (int, float))
                else None
            ),
            "no_policy_to_signature_improvement_pa": (
                no_policy_update_pa - filtered_update
                if no_policy_update_pa is not None
                and isinstance(filtered_update, (int, float))
                else None
            ),
            "broad_policy_log_count": broad_topology["policy_log_count"],
            "signature_policy_log_count": filtered_topology["policy_log_count"],
            "signature_to_broad_policy_log_fraction": safe_ratio(
                float(filtered_topology["policy_log_count"]),
                float(broad_topology["policy_log_count"]),
            ),
            "broad_matrix_mutated_count": broad_topology["matrix_mutated_count"],
            "signature_matrix_mutated_count": filtered_topology[
                "matrix_mutated_count"
            ],
            "signature_to_broad_matrix_mutated_fraction": safe_ratio(
                float(filtered_topology["matrix_mutated_count"]),
                float(broad_topology["matrix_mutated_count"]),
            ),
            "broad_touched_row_count_sum": broad_topology[
                "touched_row_count_sum"
            ],
            "signature_touched_row_count_sum": filtered_topology[
                "touched_row_count_sum"
            ],
            "signature_to_broad_touched_row_fraction": safe_ratio(
                filtered_topology["touched_row_count_sum"],
                broad_topology["touched_row_count_sum"],
            ),
            "broad_topology_edge_weight_sum_total": broad_topology[
                "topology_edge_weight_sum_total"
            ],
            "signature_topology_edge_weight_sum_total": filtered_topology[
                "topology_edge_weight_sum_total"
            ],
            "signature_to_broad_topology_edge_weight_fraction": safe_ratio(
                filtered_topology["topology_edge_weight_sum_total"],
                broad_topology["topology_edge_weight_sum_total"],
            ),
        }
    return comparisons


def test02_scope_summary(lookup: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for policy in POLICIES:
        replay = lookup.get(("test02", "broad_policy", policy))
        if replay is None:
            continue
        rows[policy] = {
            "update_pa": replay["pressure_update"][
                "worst_active_or_wet_update_pa"
            ],
            "support_class": replay["pressure_update"][
                "worst_active_or_wet_support_class"
            ],
            "policy_log_count": replay["topology"]["policy_log_count"],
            "matrix_mutated_count": replay["topology"]["matrix_mutated_count"],
            "touched_row_count_sum": replay["topology"][
                "touched_row_count_sum"
            ],
            "topology_edge_weight_sum_total": replay["topology"][
                "topology_edge_weight_sum_total"
            ],
            "max_row_abs_delta": replay["topology"]["max_row_abs_delta"],
        }
    return rows


def all_positive(values: list[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, (int, float)) and value > 0.0 for value in values
    )


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    mode_replays_json: Path = DEFAULT_MODE_REPLAYS,
) -> dict[str, Any]:
    replays = [summarize_replay(artifact_root, spec) for spec in REPLAYS]
    missing = [
        replay["solver_log_path"]
        for replay in replays
        if not replay["topology"].get("exists")
    ] + [
        replay["pressure_audit_path"]
        for replay in replays
        if not replay["pressure_update"].get("exists")
    ]
    mode_replays = load_json(mode_replays_json)
    no_policy_update = same_case_no_policy_update(mode_replays)
    lookup = replay_lookup(replays)
    test10_scope = compare_test10_scope(
        lookup, no_policy_update_pa=no_policy_update
    )
    signature_minus_broad = [
        item.get("signature_minus_broad_update_pa")
        for item in test10_scope.values()
    ]
    broad_improvements = [
        item.get("no_policy_to_broad_improvement_pa")
        for item in test10_scope.values()
        if item.get("no_policy_to_broad_improvement_pa") is not None
    ]
    all_guards_triggered = all(
        replay["pressure_update"].get("guard_triggered") for replay in replays
    )
    signature_rows_worse_than_broad = all_positive(signature_minus_broad)
    broad_combined = lookup.get(("test10", "broad_policy", "local_schur_edge_balance"))
    broad_combined_still_triggers = (
        broad_combined is not None
        and broad_combined["pressure_update"].get("guard_triggered") is True
    )

    if missing:
        finding = "direct_pspg_topology_policy_scope_scale_incomplete"
        status = "regenerate_missing_policy_scope_evidence"
        conclusion = (
            "At least one topology-policy solver log or pressure-update audit "
            "is missing."
        )
    elif (
        all_guards_triggered
        and signature_rows_worse_than_broad
        and broad_combined_still_triggers
        and any(value > 0.0 for value in broad_improvements)
    ):
        finding = (
            "direct_pspg_topology_policy_scope_scale_rules_out_exact_row_filter"
        )
        status = "broad_cosupport_mutation_helpful_but_insufficient"
        conclusion = (
            "Broad Test10 local topology mutation improves the no-policy "
            "same-case branch, but every policy remains above the guard and "
            "restricting mutation to the exact 48 signature rows is worse than "
            "the corresponding broad policy in all tested modes. This rules "
            "out exact-row local matrix deltas as the missing formulation rule."
        )
    else:
        finding = "direct_pspg_topology_policy_scope_scale_mixed"
        status = "inspect_policy_scope_scale"
        conclusion = "Topology-policy scope/scale comparisons are mixed."

    return {
        "scope": (
            "Compare broad and exact-signature-row direct PSPG topology-policy "
            "mutation scope against short Test02/Test10 pressure-update outcomes."
        ),
        "finding": finding,
        "status": status,
        "same_case_no_policy_test10_update_pa": no_policy_update,
        "all_replays_trigger_guard": all_guards_triggered,
        "signature_rows_worse_than_broad_for_all_test10_modes": (
            signature_rows_worse_than_broad
        ),
        "test10_broad_vs_signature_row_filter": test10_scope,
        "test02_broad_policy_scope": test02_scope_summary(lookup),
        "replays": replays,
        "missing_evidence": missing,
        "conclusion": conclusion,
        "next_requirement": (
            "A credible formulation fix must act on the coupled direct PSPG "
            "support patch or physical boundary support rule, not on exact row "
            "filters or a smaller replay of existing local edge deltas. Broad "
            "co-support mutation is directionally helpful for Test10 but still "
            "insufficient and leaves Test02 on the tiny-cut-supported branch."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        mode_replays_json=args.mode_replays_json,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
