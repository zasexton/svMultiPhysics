#!/usr/bin/env python3
"""Audit readiness for direct PSPG cut-volume local column-support evidence."""

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


def _load_local_matrix_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_local_matrix_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


LM = _load_local_matrix_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read env-gated cut-volume local matrix column-support provenance "
            "and summarize signed pressure-gradient row neighborhoods for "
            "audited Test02/Test10 direct PSPG target rows."
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
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def as_float_list(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    items = value if isinstance(value, list) else []
    floats: list[float] = []
    for item in items:
        try:
            floats.append(float(item))
        except (TypeError, ValueError):
            continue
    return floats


def as_int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    items = value if isinstance(value, list) else []
    ints: list[int] = []
    for item in items:
        try:
            ints.append(int(item))
        except (TypeError, ValueError):
            continue
    return ints


def matching_column_entry(
    entry: dict[str, Any],
    *,
    operator: str,
    test_field: str,
    trial_field: str,
) -> bool:
    return (
        entry.get("op") == operator
        and isinstance(entry.get("test"), str)
        and isinstance(entry.get("trial"), str)
        and entry["test"].lower() == test_field.lower()
        and entry["trial"].lower() == trial_field.lower()
    )


def latest_column_support_batch(
    log_path: Path,
    *,
    operator: str,
    test_field: str,
    trial_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = {
        "path": str(log_path),
        "exists": log_path.exists(),
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
    }
    if not log_path.exists():
        evidence["status"] = "log_missing"
        return [], evidence

    current: list[dict[str, Any]] = []
    batches: list[list[dict[str, Any]]] = []
    entry_count = 0
    previous_rule_index: int | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "diagnostic=cut_volume_local_matrix_column_support" not in line:
            continue
        entry = LM.parse_key_values(line)
        if not matching_column_entry(
            entry,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        ):
            continue
        rule_index = entry.get("rule_index")
        if (
            current
            and isinstance(rule_index, int)
            and previous_rule_index is not None
            and rule_index < previous_rule_index
        ):
            batches.append(current)
            current = []
        current.append(entry)
        entry_count += 1
        if isinstance(rule_index, int):
            previous_rule_index = rule_index
    if current:
        batches.append(current)

    evidence["entry_count"] = entry_count
    evidence["batch_count"] = len(batches)
    if not batches:
        evidence["status"] = "column_support_entries_missing"
        return [], evidence
    evidence["status"] = "ok"
    evidence["latest_batch_entry_count"] = len(batches[-1])
    return batches[-1], evidence


def row_profiles_from_column_entries(
    entries: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row = entry.get("row_dof")
        if not isinstance(row, int):
            continue
        profile = profiles.setdefault(
            row,
            {
                "global_dof": row,
                "rule_count": 0,
                "parent_cells": set(),
                "pressure_row_abs_sum": 0.0,
                "pressure_row_signed_sum": 0.0,
                "positive_sum": 0.0,
                "negative_abs_sum": 0.0,
                "diag_abs": 0.0,
                "offdiag_abs_sum": 0.0,
                "nonzero_col_count_total": 0,
                "positive_col_count_total": 0,
                "negative_col_count_total": 0,
                "sample_truncated_rule_count": 0,
                "diag_in_sample_rule_count": 0,
                "sampled_col_dofs": set(),
                "positive_sampled_offdiag_col_dofs": set(),
                "negative_sampled_offdiag_col_dofs": set(),
                "sampled_offdiag_abs_sum": 0.0,
                "sampled_positive_offdiag_sum": 0.0,
                "sampled_negative_offdiag_abs_sum": 0.0,
            },
        )
        profile["rule_count"] += 1
        parent_cell = entry.get("parent_cell")
        if isinstance(parent_cell, int):
            profile["parent_cells"].add(parent_cell)
        profile["pressure_row_abs_sum"] += LM.safe_float(entry.get("row_abs_sum"))
        profile["pressure_row_signed_sum"] += LM.safe_float(entry.get("row_signed_sum"))
        profile["positive_sum"] += LM.safe_float(entry.get("positive_sum"))
        profile["negative_abs_sum"] += LM.safe_float(entry.get("negative_abs_sum"))
        profile["diag_abs"] += LM.safe_float(entry.get("diag_abs"))
        profile["offdiag_abs_sum"] += LM.safe_float(entry.get("offdiag_abs_sum"))
        profile["nonzero_col_count_total"] += LM.safe_int(entry.get("nonzero_col_count"))
        profile["positive_col_count_total"] += LM.safe_int(entry.get("positive_col_count"))
        profile["negative_col_count_total"] += LM.safe_int(entry.get("negative_col_count"))
        profile["sample_truncated_rule_count"] += (
            1 if entry.get("sample_truncated") == 1 else 0
        )
        profile["diag_in_sample_rule_count"] += (
            1 if entry.get("diag_in_sample") == 1 else 0
        )

        col_dofs = as_int_list(entry.get("sampled_col_dofs"))
        col_values = as_float_list(entry.get("sampled_col_values"))
        for col_dof, value in zip(col_dofs, col_values):
            profile["sampled_col_dofs"].add(col_dof)
            if col_dof == row:
                continue
            abs_value = abs(value)
            profile["sampled_offdiag_abs_sum"] += abs_value
            if value > 0.0:
                profile["sampled_positive_offdiag_sum"] += value
                profile["positive_sampled_offdiag_col_dofs"].add(col_dof)
            elif value < 0.0:
                profile["sampled_negative_offdiag_abs_sum"] += -value
                profile["negative_sampled_offdiag_col_dofs"].add(col_dof)

    normalized: dict[int, dict[str, Any]] = {}
    for row, profile in profiles.items():
        sampled_col_dofs = sorted(profile.pop("sampled_col_dofs"))
        positive_offdiag = sorted(profile.pop("positive_sampled_offdiag_col_dofs"))
        negative_offdiag = sorted(profile.pop("negative_sampled_offdiag_col_dofs"))
        parent_cells = sorted(profile.pop("parent_cells"))
        profile["parent_cells"] = parent_cells
        profile["parent_cell_count"] = len(parent_cells)
        profile["sampled_col_dofs"] = sampled_col_dofs
        profile["sampled_col_count"] = len(sampled_col_dofs)
        profile["sampled_offdiag_col_count"] = len(
            [dof for dof in sampled_col_dofs if dof != row]
        )
        profile["positive_sampled_offdiag_col_dofs"] = positive_offdiag
        profile["negative_sampled_offdiag_col_dofs"] = negative_offdiag
        profile["positive_sampled_offdiag_col_count"] = len(positive_offdiag)
        profile["negative_sampled_offdiag_col_count"] = len(negative_offdiag)
        offdiag_abs = profile["sampled_offdiag_abs_sum"]
        profile["sampled_offdiag_signed_balance_ratio"] = (
            abs(
                profile["sampled_positive_offdiag_sum"]
                - profile["sampled_negative_offdiag_abs_sum"]
            )
            / offdiag_abs
            if offdiag_abs > 0.0
            else 0.0
        )
        profile["all_rules_diag_in_sample"] = (
            profile["diag_in_sample_rule_count"] == profile["rule_count"]
        )
        row_abs = profile["pressure_row_abs_sum"]
        row_sum_ratio = abs(profile["pressure_row_signed_sum"]) / row_abs if row_abs > 0.0 else 0.0
        profile["pressure_row_signed_sum_ratio"] = row_sum_ratio
        if row_abs <= 0.0:
            support_class = "zero_sampled_pressure_row"
        elif (
            profile["all_rules_diag_in_sample"]
            and profile["positive_sampled_offdiag_col_count"] == 0
            and profile["negative_sampled_offdiag_col_count"] > 0
            and row_sum_ratio <= 1.0e-12
        ):
            support_class = "null_preserving_negative_offdiag_stencil"
        elif (
            profile["positive_sampled_offdiag_col_count"] == 0
            and profile["negative_sampled_offdiag_col_count"] > 0
        ):
            support_class = "negative_offdiag_stencil_with_row_sum_leak"
        elif (
            profile["positive_sampled_offdiag_col_count"] > 0
            and profile["negative_sampled_offdiag_col_count"] > 0
        ):
            support_class = "mixed_sign_offdiag_stencil"
        elif profile["negative_sampled_offdiag_col_count"] > 0:
            support_class = "negative_offdiag_stencil"
        elif profile["positive_sampled_offdiag_col_count"] > 0:
            support_class = "positive_offdiag_stencil"
        else:
            support_class = "diagonal_or_empty_stencil"
        profile["column_support_class"] = support_class
        normalized[row] = profile
    return normalized


def profile_summary(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    candidate_class_counts: dict[str, int] = {}
    target_class_counts: dict[str, int] = {}
    for row in profiled_candidates:
        support_class = profiles[row].get("column_support_class", "unknown")
        candidate_class_counts[support_class] = (
            candidate_class_counts.get(support_class, 0) + 1
        )
    for row in target_rows:
        if row in profiles:
            support_class = profiles[row].get("column_support_class", "unknown")
            target_class_counts[support_class] = (
                target_class_counts.get(support_class, 0) + 1
            )
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(target_set.intersection(profiles)),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [row for row in target_rows if row not in profiles],
        "candidate_column_support_class_counts": candidate_class_counts,
        "target_column_support_class_counts": target_class_counts,
        "target_profiles": {str(row): profiles[row] for row in target_rows if row in profiles},
    }


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
) -> dict[str, Any]:
    emission_cases = LM.case_map(global_emission)
    target_cases = LM.target_case_map(target_map)
    log_paths = LM.default_log_paths(emission_cases, explicit_logs or [])
    cases: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = latest_column_support_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        if evidence.get("status") != "ok":
            missing_cases.append(label)
        profiles = row_profiles_from_column_entries(entries)
        cases.append(
            {
                "label": label,
                "candidate_key": candidate_key,
                "candidate_count": len(candidate_rows),
                "direct_target_count": len(target_rows),
                "log_evidence": evidence,
                "profile_summary": profile_summary(
                    profiles,
                    candidate_rows,
                    target_rows,
                ),
            }
        )

    if missing_cases:
        finding = "direct_pspg_cut_volume_column_support_evidence_missing"
        next_requirement = (
            "Rerun the short Test02/Test10 provenance windows with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_COLUMN_SUPPORT_DIAGNOSTIC=1."
        )
    else:
        finding = "direct_pspg_cut_volume_column_support_evidence_ready"
        next_requirement = (
            "Use signed sampled column neighborhoods to derive a formulation-side "
            "direct PSPG pressure-gradient support rule instead of scalar row "
            "thresholds."
        )

    return {
        "scope": (
            "Readiness audit for signed column-support provenance from direct "
            "PSPG cut-volume local pressure-gradient rows."
        ),
        "global_emission_path": str(global_emission_path) if global_emission_path else None,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "candidate_key": candidate_key,
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
        "finding": finding,
        "missing_case_labels": missing_cases,
        "cases": cases,
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
