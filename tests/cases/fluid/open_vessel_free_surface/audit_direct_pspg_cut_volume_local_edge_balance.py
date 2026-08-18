#!/usr/bin/env python3
"""Audit element-local direct PSPG existing-edge balance selectivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_GLOBAL_EMISSION = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_global_candidate_emission_20260606.json"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)
DEFAULT_OPERATOR = "equations_diagnostic_ns_vms_pspg_pressure_gradient"
LOCAL_EDGE_BALANCE_LOG_NAME = "run_direct_pspg_cut_volume_local_edge_balance.log"


def _load_schur_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_local_schur_completion.py"
    )
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCHUR = _load_schur_module()
LM = SCHUR.LM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare rows selected by the element-local direct PSPG existing-edge "
            "balance diagnostic against audited Test02/Test10 direct PSPG targets."
        )
    )
    parser.add_argument("--global-emission-json", type=Path, default=DEFAULT_GLOBAL_EMISSION)
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--log",
        action="append",
        type=str,
        default=[],
        help="Case-labelled log path as label=/path/to/run.log.",
    )
    parser.add_argument("--candidate-key", default="preferred_candidate_global_dofs")
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--test-field", default="pressure")
    parser.add_argument("--trial-field", default="pressure")
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio still considered selective.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def default_local_edge_balance_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, case in emission_cases.items():
        path = case.get("path")
        if not isinstance(path, str) or not path:
            continue
        paths[label] = Path(path).with_name(LOCAL_EDGE_BALANCE_LOG_NAME)
    for value in explicit_logs:
        label, path = LM.parse_log_arg(value)
        paths[label] = path
    return paths


def latest_local_edge_balance_batch(
    log_path: Path,
    *,
    operator: str,
    test_field: str,
    trial_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence = {
        "path": str(log_path),
        "exists": log_path.exists(),
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
    }
    if not log_path.exists():
        evidence["status"] = "log_missing"
        return [], [], evidence

    current_rows: list[dict[str, Any]] = []
    current_summaries: list[dict[str, Any]] = []
    row_batches: list[list[dict[str, Any]]] = []
    summary_batches: list[list[dict[str, Any]]] = []
    row_entry_count = 0
    summary_entry_count = 0
    previous_rule_index: int | None = None
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "diagnostic=cut_volume_direct_pspg_local_edge_balance" not in line:
                continue
            entry = LM.parse_key_values(line)
            if not LM.matching_entry(
                entry,
                operator=operator,
                test_field=test_field,
                trial_field=trial_field,
            ):
                continue
            rule_index = entry.get("rule_index")
            if (
                (current_rows or current_summaries)
                and isinstance(rule_index, int)
                and previous_rule_index is not None
                and rule_index < previous_rule_index
            ):
                row_batches.append(current_rows)
                summary_batches.append(current_summaries)
                current_rows = []
                current_summaries = []
            if entry.get("record") == "summary":
                current_summaries.append(entry)
                summary_entry_count += 1
            elif entry.get("record") == "row":
                current_rows.append(entry)
                row_entry_count += 1
            if isinstance(rule_index, int):
                previous_rule_index = rule_index
    if current_rows or current_summaries:
        row_batches.append(current_rows)
        summary_batches.append(current_summaries)

    evidence["row_entry_count"] = row_entry_count
    evidence["summary_entry_count"] = summary_entry_count
    evidence["batch_count"] = len(row_batches)
    if not row_batches:
        evidence["status"] = "local_edge_balance_entries_missing"
        return [], [], evidence
    evidence["status"] = "ok"
    evidence["latest_batch_row_entry_count"] = len(row_batches[-1])
    evidence["latest_batch_summary_entry_count"] = len(summary_batches[-1])
    return row_batches[-1], summary_batches[-1], evidence


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def local_edge_balance_profiles(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    seen: dict[int, set[tuple[Any, ...]]] = {}
    for entry in entries:
        row = entry.get("row_dof")
        if not isinstance(row, int):
            continue
        rule_key = (
            entry.get("rule_index"),
            entry.get("parent_cell"),
            entry.get("full_cell"),
            entry.get("source_revision"),
            entry.get("cut_topology_revision"),
            entry.get("quadrature_policy_key"),
            entry.get("row_local_index"),
        )
        row_seen = seen.setdefault(row, set())
        if rule_key in row_seen:
            continue
        row_seen.add(rule_key)
        profile = profiles.setdefault(
            row,
            {
                "global_dof": row,
                "rule_count": 0,
                "full_cell_rule_count": 0,
                "partial_cut_rule_count": 0,
                "balance_candidate_rule_count": 0,
                "touched_rule_count": 0,
                "source_edge_count": 0,
                "source_edge_weight_sum": 0.0,
                "balance_edge_count": 0,
                "balance_edge_weight_sum": 0.0,
                "balance_row_abs_delta": 0.0,
                "max_balance_row_abs_ratio": 0.0,
                "max_row_abs_sum": 0.0,
                "max_row_scale": 1.0,
                "parent_cells": set(),
            },
        )
        profile["rule_count"] += 1
        if entry.get("full_cell") == 1:
            profile["full_cell_rule_count"] += 1
        else:
            profile["partial_cut_rule_count"] += 1
        if entry.get("balance_candidate") == 1:
            profile["balance_candidate_rule_count"] += 1
        if _safe_float(entry.get("balance_row_abs_delta")) > 0.0:
            profile["touched_rule_count"] += 1
        parent_cell = entry.get("parent_cell")
        if isinstance(parent_cell, int):
            profile["parent_cells"].add(parent_cell)
        profile["source_edge_count"] += int(entry.get("source_edge_count") or 0)
        profile["source_edge_weight_sum"] += _safe_float(
            entry.get("source_edge_weight_sum")
        )
        profile["balance_edge_count"] += int(entry.get("balance_edge_count") or 0)
        profile["balance_edge_weight_sum"] += _safe_float(
            entry.get("balance_edge_weight_sum")
        )
        profile["balance_row_abs_delta"] += _safe_float(
            entry.get("balance_row_abs_delta")
        )
        profile["max_balance_row_abs_ratio"] = max(
            profile["max_balance_row_abs_ratio"],
            _safe_float(entry.get("balance_row_abs_ratio")),
        )
        profile["max_row_abs_sum"] = max(
            profile["max_row_abs_sum"],
            _safe_float(entry.get("row_abs_sum")),
        )
        profile["max_row_scale"] = max(
            profile["max_row_scale"],
            _safe_float(entry.get("row_scale")),
        )
    for profile in profiles.values():
        profile["parent_cell_count"] = len(profile["parent_cells"])
        profile["parent_cells"] = sorted(profile["parent_cells"])
    return profiles


def summary_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "summary_count": len(summaries),
        "local_row_count_sum": 0,
        "source_edge_count_sum": 0,
        "balance_candidate_row_count_sum": 0,
        "touched_row_count_sum": 0,
        "balance_edge_count_sum": 0,
        "max_row_scale": 1.0,
        "max_row_abs_delta": 0.0,
        "constant_pressure_null_preserving_all": True,
        "diagnostic_only_all": True,
    }
    for entry in summaries:
        metrics["local_row_count_sum"] += int(entry.get("local_row_count") or 0)
        metrics["source_edge_count_sum"] += int(entry.get("source_edge_count") or 0)
        metrics["balance_candidate_row_count_sum"] += int(
            entry.get("balance_candidate_row_count") or 0
        )
        metrics["touched_row_count_sum"] += int(entry.get("touched_row_count") or 0)
        metrics["balance_edge_count_sum"] += int(
            entry.get("balance_edge_count") or 0
        )
        metrics["max_row_scale"] = max(
            metrics["max_row_scale"],
            _safe_float(entry.get("max_row_scale")),
        )
        metrics["max_row_abs_delta"] = max(
            metrics["max_row_abs_delta"],
            _safe_float(entry.get("max_row_abs_delta")),
        )
        metrics["constant_pressure_null_preserving_all"] = (
            metrics["constant_pressure_null_preserving_all"]
            and entry.get("constant_pressure_null_preserving") == 1
        )
        metrics["diagnostic_only_all"] = (
            metrics["diagnostic_only_all"] and entry.get("diagnostic_only") == 1
        )
    return metrics


def _selector_rows(
    profiles: dict[int, dict[str, Any]],
    *,
    key: str,
) -> list[int]:
    if key == "local_edge_balance_candidate_rows":
        return sorted(
            row
            for row, profile in profiles.items()
            if int(profile.get("balance_candidate_rule_count") or 0) > 0
        )
    if key == "local_edge_balance_touched_rows":
        return sorted(
            row
            for row, profile in profiles.items()
            if int(profile.get("touched_rule_count") or 0) > 0
        )
    raise ValueError(f"unknown selector key: {key}")


def evaluate_case(
    *,
    label: str,
    key: str,
    description: str,
    candidate_rows: list[int],
    target_rows: list[int],
    profiles: dict[int, dict[str, Any]],
    max_target_ratio: float,
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    selected = [row for row in _selector_rows(profiles, key=key) if row in candidate_set]
    selected_set = set(selected)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    finding = LM.selector_finding(
        selected_count=len(selected),
        covered=covered,
        uncovered=uncovered,
        direct_target_count=len(target_rows),
        max_target_ratio=max_target_ratio,
    )
    return {
        "label": label,
        "key": key,
        "description": description,
        "finding": finding,
        "direct_target_count": len(target_rows),
        "selected_count": len(selected),
        "selected_to_target_ratio": LM.ratio(len(selected), len(target_rows)),
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "selected_global_dofs": selected,
        "selected_outside_candidate_count": len(
            set(_selector_rows(profiles, key=key)).difference(candidate_set)
        ),
    }


def profile_summary(
    *,
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    profiled_target_rows = target_set.intersection(profiles)
    candidate_rows_selected = {
        row
        for row, profile in profiles.items()
        if int(profile.get("balance_candidate_rule_count") or 0) > 0
    }
    touched_rows = {
        row
        for row, profile in profiles.items()
        if int(profile.get("touched_rule_count") or 0) > 0
    }
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(profiled_target_rows),
        "balance_candidate_row_count": len(candidate_rows_selected),
        "balance_touched_row_count": len(touched_rows),
        "balance_candidate_target_count": len(candidate_rows_selected & target_set),
        "balance_touched_target_count": len(touched_rows & target_set),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "target_profiles": {
            str(row): profiles[row] for row in target_rows if row in profiles
        },
    }


SELECTORS = [
    (
        "local_edge_balance_candidate_rows",
        "Rows whose local direct PSPG pressure-gradient pressure row norm would "
        "be scaled up to the element-local target self-row norm.",
    ),
    (
        "local_edge_balance_touched_rows",
        "Rows incident to an existing local pressure Laplacian edge whose "
        "diagnostic balance delta is positive.",
    ),
]


def build_report(
    *,
    global_emission: dict[str, Any],
    target_map: dict[str, Any],
    global_emission_path: Path | None = None,
    target_map_path: Path | None = None,
    explicit_logs: list[str] | None = None,
    candidate_key: str = "preferred_candidate_global_dofs",
    operator: str = DEFAULT_OPERATOR,
    test_field: str = "pressure",
    trial_field: str = "pressure",
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    emission_cases = LM.case_map(global_emission)
    target_cases = LM.target_case_map(target_map)
    log_paths = default_local_edge_balance_log_paths(
        emission_cases, explicit_logs or []
    )

    cases: list[dict[str, Any]] = []
    selector_results: list[dict[str, Any]] = []
    selector_cases_by_key: dict[str, list[dict[str, Any]]] = {
        key: [] for key, _ in SELECTORS
    }
    missing_cases: list[str] = []
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        rows, summaries, evidence = latest_local_edge_balance_batch(
            log_paths.get(label, Path("")),
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        if evidence.get("status") != "ok":
            missing_cases.append(label)
        profiles = local_edge_balance_profiles(rows)
        case_selectors: list[dict[str, Any]] = []
        for key, description in SELECTORS:
            selector_case = evaluate_case(
                label=label,
                key=key,
                description=description,
                candidate_rows=candidate_rows,
                target_rows=target_rows,
                profiles=profiles,
                max_target_ratio=max_target_ratio,
            )
            selector_cases_by_key[key].append(selector_case)
            case_selectors.append(selector_case)
        cases.append(
            {
                "label": label,
                "candidate_key": candidate_key,
                "candidate_count": len(candidate_rows),
                "direct_target_count": len(target_rows),
                "log_evidence": evidence,
                "summary_metrics": summary_metrics(summaries),
                "profile_summary": profile_summary(
                    profiles=profiles,
                    candidate_rows=candidate_rows,
                    target_rows=target_rows,
                ),
                "selectors": case_selectors,
            }
        )

    for key, description in SELECTORS:
        selector_cases = selector_cases_by_key[key]
        selector_results.append(
            {
                "key": key,
                "description": description,
                "finding": LM.aggregate_selector_finding(selector_cases),
                "cases": selector_cases,
            }
        )

    aggregate_findings = [selector["finding"] for selector in selector_results]
    if missing_cases:
        finding = "direct_pspg_cut_volume_local_edge_balance_evidence_missing"
        next_requirement = (
            "Regenerate Test02/Test10 short replay logs with "
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_LOCAL_EDGE_BALANCE_DIAGNOSTIC=1, "
            "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC=1, and "
            f"the direct PSPG diagnostic operator {operator} installed."
        )
    elif any(finding == "selector_selective" for finding in aggregate_findings):
        finding = "direct_pspg_cut_volume_local_edge_balance_selective"
        next_requirement = (
            "Promote the selective local existing-edge balance selector to an "
            "env-gated solve-affecting replay correction and run short "
            "Test02/Test10 windows."
        )
    elif any("miss" in finding for finding in aggregate_findings):
        finding = "direct_pspg_cut_volume_local_edge_balance_misses_targets"
        next_requirement = (
            "Do not promote local existing-edge balance alone; derive the next "
            "prototype from a stronger support/coupling rule because at least "
            "one local balance selector misses audited direct PSPG targets."
        )
    elif any("overbroad" in finding for finding in aggregate_findings):
        finding = "direct_pspg_cut_volume_local_edge_balance_overbroad"
        next_requirement = (
            "Do not promote local existing-edge balance alone; it covers audited "
            "targets only by selecting a broad direct PSPG candidate set."
        )
    else:
        finding = "direct_pspg_cut_volume_local_edge_balance_inconclusive"
        next_requirement = (
            "Regenerate local existing-edge balance diagnostic logs before "
            "selecting a solve-affecting formulation replay."
        )

    return {
        "scope": (
            "Element-local existing-edge balance diagnostic for active cut-volume "
            "direct PSPG pressure-gradient pressure-pressure blocks."
        ),
        "global_emission_path": str(global_emission_path) if global_emission_path else None,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "candidate_key": candidate_key,
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
        "max_target_ratio": max_target_ratio,
        "finding": finding,
        "missing_case_labels": missing_cases,
        "aggregate_selector_findings": aggregate_findings,
        "cases": cases,
        "selectors": selector_results,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        global_emission=LM.load_json(args.global_emission_json),
        target_map=LM.load_json(args.target_map_json),
        global_emission_path=args.global_emission_json,
        target_map_path=args.target_map_json,
        explicit_logs=args.log,
        candidate_key=args.candidate_key,
        operator=args.operator,
        test_field=args.test_field,
        trial_field=args.trial_field,
        max_target_ratio=args.max_target_ratio,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
