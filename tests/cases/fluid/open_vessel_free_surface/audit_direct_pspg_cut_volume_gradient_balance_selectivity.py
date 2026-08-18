#!/usr/bin/env python3
"""Audit direct PSPG cut-volume shape-gradient balance selectivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable


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
GRADIENT_LOG_NAME = "run_direct_pspg_cut_volume_gradient_balance.log"
NONZERO_TOLERANCE = 1.0e-12


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
            "Compare physical shape-gradient moments and gradient Gram-stencil "
            "balance from direct PSPG cut-volume rows against audited Test02/"
            "Test10 target rows."
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
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_float(value: Any, default: float = 0.0) -> float:
    result = finite_float(value)
    return result if result is not None else default


def finite_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    parsed: list[float] = []
    for value in values:
        result = finite_float(value)
        if result is not None:
            parsed.append(result)
    return parsed


def sign(value: float) -> int:
    if value > NONZERO_TOLERANCE:
        return 1
    if value < -NONZERO_TOLERANCE:
        return -1
    return 0


def default_gradient_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, case in emission_cases.items():
        path = case.get("path")
        if not isinstance(path, str) or not path:
            continue
        base = Path(path)
        if base.name == "run_direct_pspg_cut_volume_row_provenance.log":
            paths[label] = base.with_name(GRADIENT_LOG_NAME)
        else:
            paths[label] = base
    for value in explicit_logs:
        label, path = LM.parse_log_arg(value)
        paths[label] = path
    return paths


def latest_gradient_balance_batch(
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
    available_entry_count = 0
    previous_rule_index: int | None = None
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "diagnostic=cut_volume_local_matrix_gradient_balance" not in line:
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
                current
                and isinstance(rule_index, int)
                and previous_rule_index is not None
                and rule_index < previous_rule_index
            ):
                batches.append(current)
                current = []
            current.append(entry)
            entry_count += 1
            if entry.get("gradient_balance_available") == 1:
                available_entry_count += 1
            if isinstance(rule_index, int):
                previous_rule_index = rule_index
    if current:
        batches.append(current)

    evidence["entry_count"] = entry_count
    evidence["available_entry_count"] = available_entry_count
    evidence["batch_count"] = len(batches)
    if not batches:
        evidence["status"] = "gradient_balance_entries_missing"
        return [], evidence
    evidence["status"] = "ok"
    evidence["latest_batch_entry_count"] = len(batches[-1])
    evidence["latest_batch_available_entry_count"] = sum(
        1 for entry in batches[-1] if entry.get("gradient_balance_available") == 1
    )
    return batches[-1], evidence


def row_profiles_from_entries(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
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
                "gradient_available_rule_count": 0,
                "full_cell_rule_count": 0,
                "partial_cut_rule_count": 0,
                "parent_cells": set(),
                "row_matrix_abs_sum": 0.0,
                "row_matrix_signed_sum": 0.0,
                "row_diag_abs": 0.0,
                "row_offdiag_abs_sum": 0.0,
                "row_grad_x_sum": 0.0,
                "row_grad_y_sum": 0.0,
                "row_grad_z_sum": 0.0,
                "row_grad_norm_sum": 0.0,
                "row_grad_abs_integral_sum": 0.0,
                "row_grad_energy_sum": 0.0,
                "row_grad_max_norm": 0.0,
                "row_grad_directional_ratio_sum": 0.0,
                "row_grad_axis_dominance_sum": 0.0,
                "dominant_axes": set(),
                "gram_row_abs_sum": 0.0,
                "gram_row_signed_sum": 0.0,
                "gram_diag_abs": 0.0,
                "gram_offdiag_abs_sum": 0.0,
                "gram_nonzero_count": 0,
                "gram_positive_count": 0,
                "gram_negative_count": 0,
                "gram_max_abs_entry": 0.0,
                "sampled_col_count_total": 0,
                "sampled_sign_mismatch_count": 0,
                "sampled_negative_gram_count": 0,
                "sampled_positive_gram_count": 0,
                "sampled_abs_gram_sum": 0.0,
                "sampled_max_abs_gram": 0.0,
                "sampled_abs_cosine_sum": 0.0,
                "sampled_max_abs_cosine": 0.0,
                "sample_truncated_rule_count": 0,
            },
        )
        profile["rule_count"] += 1
        if entry.get("gradient_balance_available") == 1:
            profile["gradient_available_rule_count"] += 1
        if entry.get("full_cell") == 1:
            profile["full_cell_rule_count"] += 1
        else:
            profile["partial_cut_rule_count"] += 1
        parent_cell = entry.get("parent_cell")
        if isinstance(parent_cell, int):
            profile["parent_cells"].add(parent_cell)

        profile["row_matrix_abs_sum"] += safe_float(entry.get("row_abs_sum"))
        profile["row_matrix_signed_sum"] += safe_float(entry.get("row_signed_sum"))
        profile["row_diag_abs"] += safe_float(entry.get("diag_abs"))
        profile["row_offdiag_abs_sum"] += safe_float(entry.get("offdiag_abs_sum"))
        if entry.get("sample_truncated") == 1:
            profile["sample_truncated_rule_count"] += 1

        if entry.get("gradient_balance_available") != 1:
            continue
        profile["row_grad_x_sum"] += safe_float(entry.get("row_grad_x"))
        profile["row_grad_y_sum"] += safe_float(entry.get("row_grad_y"))
        profile["row_grad_z_sum"] += safe_float(entry.get("row_grad_z"))
        profile["row_grad_norm_sum"] += safe_float(entry.get("row_grad_norm"))
        profile["row_grad_abs_integral_sum"] += safe_float(
            entry.get("row_grad_abs_integral")
        )
        profile["row_grad_energy_sum"] += safe_float(entry.get("row_grad_energy"))
        profile["row_grad_max_norm"] = max(
            profile["row_grad_max_norm"],
            safe_float(entry.get("row_grad_max_norm")),
        )
        profile["row_grad_directional_ratio_sum"] += safe_float(
            entry.get("row_grad_directional_ratio")
        )
        profile["row_grad_axis_dominance_sum"] += safe_float(
            entry.get("row_grad_axis_dominance")
        )
        axis = entry.get("row_grad_dominant_axis")
        if isinstance(axis, int) and axis >= 0:
            profile["dominant_axes"].add(axis)
        profile["gram_row_abs_sum"] += safe_float(entry.get("gram_row_abs_sum"))
        profile["gram_row_signed_sum"] += safe_float(entry.get("gram_row_signed_sum"))
        profile["gram_diag_abs"] += safe_float(entry.get("gram_diag_abs"))
        profile["gram_offdiag_abs_sum"] += safe_float(entry.get("gram_offdiag_abs_sum"))
        profile["gram_nonzero_count"] += int(entry.get("gram_nonzero_count", 0) or 0)
        profile["gram_positive_count"] += int(entry.get("gram_positive_count", 0) or 0)
        profile["gram_negative_count"] += int(entry.get("gram_negative_count", 0) or 0)
        profile["gram_max_abs_entry"] = max(
            profile["gram_max_abs_entry"],
            safe_float(entry.get("gram_max_abs_entry")),
        )

        values = finite_values(entry.get("sampled_col_values"))
        grams = finite_values(entry.get("sampled_col_gradient_gram_values"))
        cosines = finite_values(entry.get("sampled_col_gradient_cosines"))
        profile["sampled_col_count_total"] += len(grams)
        for value, gram in zip(values, grams, strict=False):
            value_sign = sign(value)
            gram_sign = sign(gram)
            if value_sign and gram_sign and value_sign != gram_sign:
                profile["sampled_sign_mismatch_count"] += 1
            if gram_sign < 0:
                profile["sampled_negative_gram_count"] += 1
            elif gram_sign > 0:
                profile["sampled_positive_gram_count"] += 1
            abs_gram = abs(gram)
            profile["sampled_abs_gram_sum"] += abs_gram
            profile["sampled_max_abs_gram"] = max(
                profile["sampled_max_abs_gram"], abs_gram
            )
        for cosine in cosines:
            abs_cosine = abs(cosine)
            profile["sampled_abs_cosine_sum"] += abs_cosine
            profile["sampled_max_abs_cosine"] = max(
                profile["sampled_max_abs_cosine"], abs_cosine
            )

    normalized: dict[int, dict[str, Any]] = {}
    for row, profile in profiles.items():
        available = profile["gradient_available_rule_count"]
        parent_cells = sorted(profile.pop("parent_cells"))
        dominant_axes = sorted(profile.pop("dominant_axes"))
        profile["parent_cells"] = parent_cells
        profile["parent_cell_count"] = len(parent_cells)
        profile["dominant_axes"] = dominant_axes
        profile["dominant_axis_count"] = len(dominant_axes)
        profile["gradient_available_fraction"] = (
            available / profile["rule_count"] if profile["rule_count"] else 0.0
        )
        profile["row_grad_directional_ratio_mean"] = (
            profile["row_grad_directional_ratio_sum"] / available
            if available
            else 0.0
        )
        profile["row_grad_axis_dominance_mean"] = (
            profile["row_grad_axis_dominance_sum"] / available
            if available
            else 0.0
        )
        profile["row_grad_resultant_norm"] = math.sqrt(
            profile["row_grad_x_sum"] ** 2
            + profile["row_grad_y_sum"] ** 2
            + profile["row_grad_z_sum"] ** 2
        )
        profile["row_grad_resultant_ratio"] = (
            profile["row_grad_resultant_norm"]
            / profile["row_grad_abs_integral_sum"]
            if profile["row_grad_abs_integral_sum"] > 0.0
            else 0.0
        )
        profile["gram_diag_abs_fraction"] = (
            profile["gram_diag_abs"] / profile["gram_row_abs_sum"]
            if profile["gram_row_abs_sum"] > 0.0
            else 0.0
        )
        profile["gram_signed_cancellation_ratio"] = (
            abs(profile["gram_row_signed_sum"]) / profile["gram_row_abs_sum"]
            if profile["gram_row_abs_sum"] > 0.0
            else 0.0
        )
        profile["matrix_to_gram_abs_ratio"] = (
            profile["row_matrix_abs_sum"] / profile["gram_row_abs_sum"]
            if profile["gram_row_abs_sum"] > 0.0
            else 0.0
        )
        profile["sampled_sign_mismatch_fraction"] = (
            profile["sampled_sign_mismatch_count"]
            / profile["sampled_col_count_total"]
            if profile["sampled_col_count_total"]
            else 0.0
        )
        profile["sampled_negative_gram_fraction"] = (
            profile["sampled_negative_gram_count"]
            / profile["sampled_col_count_total"]
            if profile["sampled_col_count_total"]
            else 0.0
        )
        profile["sampled_abs_cosine_mean"] = (
            profile["sampled_abs_cosine_sum"] / profile["sampled_col_count_total"]
            if profile["sampled_col_count_total"]
            else 0.0
        )
        if profile["partial_cut_rule_count"] > 0 and profile["full_cell_rule_count"] > 0:
            support_class = "mixed_partial_and_full_cell_gradient_support"
        elif profile["partial_cut_rule_count"] > 0:
            support_class = "partial_cut_only_gradient_support"
        elif profile["full_cell_rule_count"] > 0:
            support_class = "full_cell_only_gradient_support"
        else:
            support_class = "missing_gradient_support"
        profile["gradient_support_class"] = support_class
        normalized[row] = profile
    return normalized


def metric_values(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in candidate_rows:
        profile = profiles.get(row)
        if not profile:
            continue
        value = profile.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def case_thresholds(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
) -> dict[str, float | None]:
    keys = [
        "row_grad_resultant_ratio",
        "row_grad_directional_ratio_mean",
        "row_grad_axis_dominance_mean",
        "row_grad_energy_sum",
        "gram_row_abs_sum",
        "gram_diag_abs_fraction",
        "gram_signed_cancellation_ratio",
        "matrix_to_gram_abs_ratio",
        "sampled_sign_mismatch_fraction",
        "sampled_negative_gram_fraction",
        "sampled_abs_cosine_mean",
    ]
    thresholds: dict[str, float | None] = {}
    for key in keys:
        values = metric_values(profiles, candidate_rows, key)
        thresholds[f"{key}_p10"] = LM.percentile(values, 0.10)
        thresholds[f"{key}_p25"] = LM.percentile(values, 0.25)
        thresholds[f"{key}_p75"] = LM.percentile(values, 0.75)
        thresholds[f"{key}_p90"] = LM.percentile(values, 0.90)
    return thresholds


def le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def tail_selector(
    profile: dict[str, Any],
    key: str,
    thresholds: dict[str, float | None],
) -> bool:
    return le(profile, key, thresholds[f"{key}_p25"]) or ge(
        profile, key, thresholds[f"{key}_p75"]
    )


def selector_definitions(
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "gradient_balance_profiled_candidate",
            "description": "Preferred candidates with gradient-balance profiles.",
            "threshold_key": None,
            "predicate": lambda profile: profile["gradient_available_rule_count"] > 0,
        },
        {
            "key": "gradient_balance_full_cell_only",
            "description": "Profiled candidates with only full-cell-equivalent gradient support.",
            "threshold_key": "fixed:full_cell_only_gradient_support",
            "predicate": lambda profile: (
                profile["gradient_support_class"]
                == "full_cell_only_gradient_support"
            ),
        },
        {
            "key": "gradient_balance_resultant_ratio_tail",
            "description": "Rows in either low or high row-gradient resultant ratio tail.",
            "threshold_key": "row_grad_resultant_ratio_p25|row_grad_resultant_ratio_p75",
            "predicate": lambda profile: tail_selector(
                profile, "row_grad_resultant_ratio", thresholds
            ),
        },
        {
            "key": "gradient_balance_directional_ratio_tail",
            "description": "Rows in either low or high per-rule gradient directional-ratio tail.",
            "threshold_key": (
                "row_grad_directional_ratio_mean_p25|"
                "row_grad_directional_ratio_mean_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile, "row_grad_directional_ratio_mean", thresholds
            ),
        },
        {
            "key": "gradient_balance_axis_dominance_tail",
            "description": "Rows in either low or high dominant-axis gradient tail.",
            "threshold_key": (
                "row_grad_axis_dominance_mean_p25|"
                "row_grad_axis_dominance_mean_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile, "row_grad_axis_dominance_mean", thresholds
            ),
        },
        {
            "key": "gradient_balance_energy_tail",
            "description": "Rows in either low or high shape-gradient energy tail.",
            "threshold_key": "row_grad_energy_sum_p25|row_grad_energy_sum_p75",
            "predicate": lambda profile: tail_selector(
                profile, "row_grad_energy_sum", thresholds
            ),
        },
        {
            "key": "gradient_balance_gram_abs_tail",
            "description": "Rows in either low or high gradient Gram row-action tail.",
            "threshold_key": "gram_row_abs_sum_p25|gram_row_abs_sum_p75",
            "predicate": lambda profile: tail_selector(
                profile, "gram_row_abs_sum", thresholds
            ),
        },
        {
            "key": "gradient_balance_low_gram_diag_fraction",
            "description": "Rows in the bottom quartile of gradient Gram diagonal fraction.",
            "threshold_key": "gram_diag_abs_fraction_p25",
            "predicate": lambda profile: le(
                profile,
                "gram_diag_abs_fraction",
                thresholds["gram_diag_abs_fraction_p25"],
            ),
        },
        {
            "key": "gradient_balance_high_gram_cancellation",
            "description": "Rows in the top quartile of gradient Gram row-sum leakage.",
            "threshold_key": "gram_signed_cancellation_ratio_p75",
            "predicate": lambda profile: ge(
                profile,
                "gram_signed_cancellation_ratio",
                thresholds["gram_signed_cancellation_ratio_p75"],
            ),
        },
        {
            "key": "gradient_balance_matrix_to_gram_ratio_tail",
            "description": "Rows in either low or high matrix-to-gradient-Gram scale tail.",
            "threshold_key": (
                "matrix_to_gram_abs_ratio_p25|matrix_to_gram_abs_ratio_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile, "matrix_to_gram_abs_ratio", thresholds
            ),
        },
        {
            "key": "gradient_balance_sampled_sign_mismatch",
            "description": "Rows where sampled local matrix signs disagree with gradient Gram signs.",
            "threshold_key": "fixed:sampled_sign_mismatch_fraction_gt_0",
            "predicate": lambda profile: (
                profile["sampled_sign_mismatch_fraction"] > 0.0
            ),
        },
        {
            "key": "gradient_balance_negative_gram_tail",
            "description": "Rows in either low or high sampled negative-Gram fraction tail.",
            "threshold_key": (
                "sampled_negative_gram_fraction_p25|"
                "sampled_negative_gram_fraction_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile, "sampled_negative_gram_fraction", thresholds
            ),
        },
        {
            "key": "gradient_balance_abs_cosine_tail",
            "description": "Rows in either low or high sampled gradient-cosine magnitude tail.",
            "threshold_key": "sampled_abs_cosine_mean_p25|sampled_abs_cosine_mean_p75",
            "predicate": lambda profile: tail_selector(
                profile, "sampled_abs_cosine_mean", thresholds
            ),
        },
    ]


def evaluate_selector_case(
    *,
    label: str,
    selector: dict[str, Any],
    candidate_rows: list[int],
    target_rows: list[int],
    profiles: dict[int, dict[str, Any]],
    thresholds: dict[str, float | None],
    max_target_ratio: float,
) -> dict[str, Any]:
    predicate: Callable[[dict[str, Any]], bool] = selector["predicate"]
    selected = [
        row for row in candidate_rows if row in profiles and predicate(profiles[row])
    ]
    selected_set = set(selected)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    threshold_key = selector.get("threshold_key")
    threshold_value: Any = None
    if isinstance(threshold_key, str):
        if "|" in threshold_key:
            threshold_value = {key: thresholds.get(key) for key in threshold_key.split("|")}
        elif threshold_key.startswith("fixed:"):
            threshold_value = threshold_key.removeprefix("fixed:")
        else:
            threshold_value = thresholds.get(threshold_key)
    return {
        "label": label,
        "key": selector["key"],
        "description": selector["description"],
        "threshold_key": threshold_key,
        "threshold_value": threshold_value,
        "finding": LM.selector_finding(
            selected_count=len(selected),
            covered=covered,
            uncovered=uncovered,
            direct_target_count=len(target_rows),
            max_target_ratio=max_target_ratio,
        ),
        "direct_target_count": len(target_rows),
        "selected_count": len(selected),
        "selected_to_target_ratio": LM.ratio(len(selected), len(target_rows)),
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "selected_global_dofs": selected,
    }


def profile_summary(
    *,
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    support_class_counts: dict[str, int] = {}
    for row in profiled_candidates:
        support_class = profiles[row].get("gradient_support_class", "unknown")
        support_class_counts[support_class] = support_class_counts.get(support_class, 0) + 1
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len([row for row in target_rows if row in profiles]),
        "unprofiled_candidate_count": len(set(candidate_rows).difference(profiles)),
        "unprofiled_target_global_dofs": [row for row in target_rows if row not in profiles],
        "candidate_gradient_support_class_counts": support_class_counts,
        "target_gradient_support_class_counts": {
            profiles[row].get("gradient_support_class", "unknown"): sum(
                1
                for target in target_rows
                if target in profiles
                and profiles[target].get("gradient_support_class", "unknown")
                == profiles[row].get("gradient_support_class", "unknown")
            )
            for row in target_rows
            if row in profiles
        },
        "target_profiles": {str(row): profiles[row] for row in target_rows if row in profiles},
    }


def aggregate_selector_finding(cases: list[dict[str, Any]]) -> str:
    if cases and all(case["finding"] == "selector_selective" for case in cases):
        return "selector_selective"
    if any("misses_targets" in str(case["finding"]) for case in cases):
        if any("overbroad" in str(case["finding"]) for case in cases):
            return "selector_overbroad_or_miss_targets"
        return "selector_misses_targets"
    if any(case["finding"] == "selector_overbroad" for case in cases):
        return "selector_overbroad"
    return "selector_inconclusive"


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
    log_paths = default_gradient_log_paths(emission_cases, explicit_logs or [])

    cases: dict[str, dict[str, Any]] = {}
    selector_defs_by_case: dict[str, list[dict[str, Any]]] = {}
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = latest_gradient_balance_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        profiles = row_profiles_from_entries(entries)
        thresholds = case_thresholds(profiles, candidate_rows)
        selector_defs = selector_definitions(thresholds)
        selector_defs_by_case[label] = selector_defs
        selector_cases = [
            evaluate_selector_case(
                label=label,
                selector=selector,
                candidate_rows=candidate_rows,
                target_rows=target_rows,
                profiles=profiles,
                thresholds=thresholds,
                max_target_ratio=max_target_ratio,
            )
            for selector in selector_defs
        ]
        cases[label] = {
            "label": label,
            "candidate_key": candidate_key,
            "candidate_count": len(candidate_rows),
            "direct_target_count": len(target_rows),
            "log_evidence": evidence,
            "thresholds": thresholds,
            "profile_summary": profile_summary(
                profiles=profiles,
                candidate_rows=candidate_rows,
                target_rows=target_rows,
            ),
            "selectors": selector_cases,
        }

    selectors = []
    first_label = next(iter(target_cases), None)
    selector_count = (
        len(selector_defs_by_case[first_label]) if first_label is not None else 0
    )
    for selector_index in range(selector_count):
        case_results = [cases[label]["selectors"][selector_index] for label in target_cases]
        template = selector_defs_by_case[first_label][selector_index]
        selectors.append(
            {
                "key": template["key"],
                "description": template["description"],
                "finding": aggregate_selector_finding(case_results),
                "cases": case_results,
            }
        )

    selective = [
        selector["key"]
        for selector in selectors
        if selector["finding"] == "selector_selective"
    ]
    missing_case_labels = [
        label
        for label, case in cases.items()
        if case["log_evidence"].get("status") != "ok"
        or case["profile_summary"]["profiled_target_count"]
        < case["direct_target_count"]
    ]
    if missing_case_labels:
        finding = "direct_pspg_cut_volume_gradient_balance_evidence_missing"
    elif selective:
        finding = "direct_pspg_cut_volume_gradient_balance_selector_identified"
    elif selectors and all(
        selector["finding"] in {
            "selector_overbroad_or_miss_targets",
            "selector_misses_targets",
            "selector_overbroad",
        }
        for selector in selectors
    ):
        finding = (
            "direct_pspg_cut_volume_gradient_balance_selectors_overbroad_"
            "or_miss_targets"
        )
    else:
        finding = "direct_pspg_cut_volume_gradient_balance_selectivity_inconclusive"

    return {
        "finding": finding,
        "global_emission_path": str(global_emission_path) if global_emission_path else None,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "operator": operator,
        "candidate_key": candidate_key,
        "missing_case_labels": missing_case_labels,
        "selective_selector_keys": selective,
        "cases": [cases[label] for label in target_cases],
        "selectors": selectors,
        "next_requirement": (
            "Use the gradient-balance evidence to decide whether physical "
            "shape-gradient support can supply the missing formulation-side "
            "direct PSPG support/coupling gate."
        ),
        "summary": (
            "Selectivity audit for direct PSPG cut-volume physical "
            "shape-gradient moments and gradient Gram-stencil balance."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        global_emission=load_json(args.global_emission_json),
        target_map=load_json(args.target_map_json),
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
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
