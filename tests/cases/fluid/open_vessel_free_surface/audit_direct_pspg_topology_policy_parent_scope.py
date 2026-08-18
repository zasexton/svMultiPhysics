#!/usr/bin/env python3
"""Audit direct PSPG topology-policy parent/rule support scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any, Iterable


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_parent_scope_20260607.json"
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

SUM_METRICS = [
    "record_count",
    "matrix_mutated_count",
    "touched_row_count_sum",
    "topology_edge_weight_sum",
    "source_edge_weight_sum",
    "selected_local_row_count_sum",
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


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def parse_policy_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_policy_line(line)
        if record is not None:
            records.append(record)
    return records


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
        "records": parse_policy_log(log_path),
        "topology_log_exists": log_path.exists(),
        "pressure_update": pressure_summary(audit_path),
    }


def rule_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("parent_cell"),
        record.get("rule_index"),
        record.get("quadrature_policy_key"),
        record.get("cut_topology_revision"),
    )


def group_key(record: dict[str, Any], scope: str) -> Any:
    if scope == "parent":
        return record.get("parent_cell")
    if scope == "rule":
        return rule_key(record)
    raise ValueError(f"Unknown scope: {scope}")


def group_records(records: list[dict[str, Any]], scope: str) -> dict[Any, dict[str, Any]]:
    groups: dict[Any, dict[str, Any]] = {}
    for record in records:
        key = group_key(record, scope)
        group = groups.setdefault(
            key,
            {
                "record_count": 0,
                "matrix_mutated_count": 0,
                "touched_row_count_sum": 0.0,
                "topology_edge_weight_sum": 0.0,
                "source_edge_weight_sum": 0.0,
                "selected_local_row_count_sum": 0.0,
                "full_cell_record_count": 0,
                "cut_cell_record_count": 0,
            },
        )
        group["record_count"] += 1
        group["matrix_mutated_count"] += (
            1 if record.get("matrix_mutated") == 1 else 0
        )
        group["touched_row_count_sum"] += numeric(record, "touched_row_count")
        group["topology_edge_weight_sum"] += numeric(
            record, "topology_edge_weight_sum"
        )
        group["source_edge_weight_sum"] += numeric(record, "source_edge_weight_sum")
        group["selected_local_row_count_sum"] += numeric(
            record, "row_filter_selected_local_row_count"
        )
        group["full_cell_record_count"] += 1 if record.get("full_cell") == 1 else 0
        group["cut_cell_record_count"] += 1 if record.get("full_cell") == 0 else 0
    return groups


def sum_groups(groups: dict[Any, dict[str, Any]], keys: Iterable[Any], metric: str) -> float:
    return sum(float(groups[key].get(metric, 0.0)) for key in keys if key in groups)


def total_metric(groups: dict[Any, dict[str, Any]], metric: str) -> float:
    return sum(float(group.get(metric, 0.0)) for group in groups.values())


def format_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(str(part) for part in key)
    return str(key)


def key_samples(keys: Iterable[Any], *, limit: int = 8) -> list[str]:
    return [format_key(key) for key in sorted(keys, key=format_key)[:limit]]


def compare_group_scope(
    broad_records: list[dict[str, Any]],
    signature_records: list[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    broad_groups = group_records(broad_records, scope)
    signature_groups = group_records(signature_records, scope)
    broad_keys = set(broad_groups)
    signature_keys = set(signature_groups)
    overlap = broad_keys & signature_keys
    broad_only = broad_keys - signature_keys
    signature_only = signature_keys - broad_keys

    result: dict[str, Any] = {
        "scope": scope,
        "broad_key_count": len(broad_keys),
        "signature_key_count": len(signature_keys),
        "overlap_key_count": len(overlap),
        "broad_only_key_count": len(broad_only),
        "signature_only_key_count": len(signature_only),
        "signature_to_broad_key_fraction": safe_ratio(
            float(len(signature_keys)), float(len(broad_keys))
        ),
        "broad_only_key_fraction": safe_ratio(
            float(len(broad_only)), float(len(broad_keys))
        ),
        "broad_only_key_samples": key_samples(broad_only),
        "signature_only_key_samples": key_samples(signature_only),
    }
    for metric in SUM_METRICS:
        broad_total = total_metric(broad_groups, metric)
        signature_total = total_metric(signature_groups, metric)
        broad_only_total = sum_groups(broad_groups, broad_only, metric)
        overlap_broad_total = sum_groups(broad_groups, overlap, metric)
        overlap_signature_total = sum_groups(signature_groups, overlap, metric)
        result[f"broad_{metric}"] = broad_total
        result[f"signature_{metric}"] = signature_total
        result[f"signature_to_broad_{metric}_fraction"] = safe_ratio(
            signature_total, broad_total
        )
        result[f"broad_only_{metric}"] = broad_only_total
        result[f"broad_only_{metric}_fraction"] = safe_ratio(
            broad_only_total, broad_total
        )
        result[f"overlap_broad_{metric}"] = overlap_broad_total
        result[f"overlap_broad_{metric}_fraction"] = safe_ratio(
            overlap_broad_total, broad_total
        )
        result[f"overlap_signature_{metric}"] = overlap_signature_total
        result[f"signature_to_broad_overlap_{metric}_fraction"] = safe_ratio(
            overlap_signature_total, overlap_broad_total
        )
    result["broad_full_cell_record_count"] = sum_groups(
        broad_groups, broad_keys, "full_cell_record_count"
    )
    result["broad_cut_cell_record_count"] = sum_groups(
        broad_groups, broad_keys, "cut_cell_record_count"
    )
    result["signature_full_cell_record_count"] = sum_groups(
        signature_groups, signature_keys, "full_cell_record_count"
    )
    result["signature_cut_cell_record_count"] = sum_groups(
        signature_groups, signature_keys, "cut_cell_record_count"
    )
    return result


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


def update_value(replay: dict[str, Any] | None) -> float | None:
    if replay is None:
        return None
    value = replay["pressure_update"].get("worst_active_or_wet_update_pa")
    return float(value) if isinstance(value, (int, float)) else None


def compare_test10_parent_rule_scope(
    lookup: dict[tuple[str, str, str], dict[str, Any]],
    *,
    no_policy_update_pa: float | None,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for policy in POLICIES:
        broad = lookup.get(("test10", "broad_policy", policy))
        signature = lookup.get(("test10", "signature_row_filter", policy))
        if broad is None or signature is None:
            continue
        broad_update = update_value(broad)
        signature_update = update_value(signature)
        comparisons[policy] = {
            "broad_update_pa": broad_update,
            "signature_row_filter_update_pa": signature_update,
            "signature_minus_broad_update_pa": (
                signature_update - broad_update
                if signature_update is not None and broad_update is not None
                else None
            ),
            "no_policy_to_broad_improvement_pa": (
                no_policy_update_pa - broad_update
                if no_policy_update_pa is not None and broad_update is not None
                else None
            ),
            "parent_scope": compare_group_scope(
                broad["records"], signature["records"], scope="parent"
            ),
            "rule_scope": compare_group_scope(
                broad["records"], signature["records"], scope="rule"
            ),
        }
    return comparisons


def test02_parent_rule_scope(
    lookup: dict[tuple[str, str, str], dict[str, Any]]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for policy in POLICIES:
        replay = lookup.get(("test02", "broad_policy", policy))
        if replay is None:
            continue
        empty: list[dict[str, Any]] = []
        rows[policy] = {
            "update_pa": update_value(replay),
            "support_class": replay["pressure_update"].get(
                "worst_active_or_wet_support_class"
            ),
            "parent_scope": compare_group_scope(
                replay["records"], empty, scope="parent"
            ),
            "rule_scope": compare_group_scope(replay["records"], empty, scope="rule"),
        }
    return rows


def all_positive(values: list[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, (int, float)) and value > 0.0 for value in values
    )


def all_strict_signature_subsets(comparisons: dict[str, Any]) -> bool:
    if not comparisons:
        return False
    for comparison in comparisons.values():
        for scope_name in ("parent_scope", "rule_scope"):
            scope = comparison[scope_name]
            if not (
                scope["signature_only_key_count"] == 0
                and scope["signature_key_count"] < scope["broad_key_count"]
            ):
                return False
    return True


def all_broad_only_rule_weight_majority(comparisons: dict[str, Any]) -> bool:
    if not comparisons:
        return False
    for comparison in comparisons.values():
        fraction = comparison["rule_scope"].get(
            "broad_only_topology_edge_weight_sum_fraction"
        )
        if not isinstance(fraction, (int, float)) or fraction <= 0.5:
            return False
    return True


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    mode_replays_json: Path = DEFAULT_MODE_REPLAYS,
) -> dict[str, Any]:
    replays = [summarize_replay(artifact_root, spec) for spec in REPLAYS]
    missing = [
        replay["solver_log_path"]
        for replay in replays
        if not replay["topology_log_exists"]
    ] + [
        replay["pressure_audit_path"]
        for replay in replays
        if not replay["pressure_update"].get("exists")
    ]
    lookup = replay_lookup(replays)
    mode_replays = load_json(mode_replays_json)
    no_policy_update = same_case_no_policy_update(mode_replays)
    test10_scope = compare_test10_parent_rule_scope(
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
    strict_subsets = all_strict_signature_subsets(test10_scope)
    broad_only_weight_majority = all_broad_only_rule_weight_majority(test10_scope)
    signature_worse_than_broad = all_positive(signature_minus_broad)
    broad_combined = lookup.get(("test10", "broad_policy", "local_schur_edge_balance"))
    broad_combined_still_triggers = (
        broad_combined is not None
        and broad_combined["pressure_update"].get("guard_triggered") is True
    )

    if missing:
        finding = "direct_pspg_topology_policy_parent_scope_incomplete"
        status = "regenerate_missing_parent_scope_evidence"
        conclusion = (
            "At least one topology-policy solver log or pressure-update audit "
            "is missing."
        )
    elif (
        all_guards_triggered
        and strict_subsets
        and broad_only_weight_majority
        and signature_worse_than_broad
        and broad_combined_still_triggers
        and any(value > 0.0 for value in broad_improvements)
    ):
        finding = (
            "direct_pspg_topology_policy_parent_scope_rules_out_exact_parent_subset"
        )
        status = "broad_parent_cosupport_required_but_insufficient"
        conclusion = (
            "The exact signature-row replay is a strict subset of the broad "
            "Test10 parent/rule support, and broad-only rule keys carry most "
            "of the broad topology weight in every tested local policy. Broad "
            "co-support is therefore part of the helpful Test10 effect, but "
            "the broad combined policy still triggers the pressure-update "
            "guard and Test02 remains on the tiny-cut-supported branch."
        )
    else:
        finding = "direct_pspg_topology_policy_parent_scope_mixed"
        status = "inspect_parent_rule_scope"
        conclusion = "Parent/rule support-scope comparisons are mixed."

    return {
        "scope": (
            "Compare broad and exact-signature-row direct PSPG topology-policy "
            "parent-cell/rule-key support against short Test02/Test10 "
            "pressure-update outcomes."
        ),
        "finding": finding,
        "status": status,
        "same_case_no_policy_test10_update_pa": no_policy_update,
        "all_replays_trigger_guard": all_guards_triggered,
        "all_test10_signature_parent_rule_sets_are_strict_broad_subsets": (
            strict_subsets
        ),
        "all_test10_broad_only_rule_weight_share_above_half": (
            broad_only_weight_majority
        ),
        "signature_rows_worse_than_broad_for_all_test10_modes": (
            signature_worse_than_broad
        ),
        "test10_parent_rule_scope": test10_scope,
        "test02_broad_parent_rule_scope": test02_parent_rule_scope(lookup),
        "replays": [
            {
                "label": replay["label"],
                "variant": replay["variant"],
                "policy": replay["policy"],
                "solver_log_path": replay["solver_log_path"],
                "pressure_audit_path": replay["pressure_audit_path"],
                "topology_log_exists": replay["topology_log_exists"],
                "policy_log_count": len(replay["records"]),
                "pressure_update": replay["pressure_update"],
            }
            for replay in replays
        ],
        "missing_evidence": missing,
        "conclusion": conclusion,
        "next_requirement": (
            "A credible direct PSPG formulation fix should express a connected "
            "support-patch or physical boundary-support closure over the "
            "parent/rule co-support, not exact selected rows or only the "
            "current overlapping parent subset. The broad co-support mutation "
            "helps Test10 but remains insufficient and does not explain the "
            "Test02 tiny-cut-supported pressure amplification."
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
