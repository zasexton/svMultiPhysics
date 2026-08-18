#!/usr/bin/env python3
"""Audit direct PSPG cut-volume local matrix row-action selectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare assembly-time local matrix row-action metrics from the "
            "direct PSPG pressure-gradient diagnostic operator against the "
            "audited Test02/Test10 target rows."
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
    parser.add_argument("--low-parent-cell-count", type=int, default=2)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio still considered selective.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [item for item in as_list(value) if isinstance(item, int)]


def parse_dof_list(value: str) -> list[int | str]:
    if value in {"", "none"}:
        return []
    parsed: list[int | str] = []
    for token in value.split("|"):
        if token == "...":
            parsed.append(token)
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            parsed.append(token)
    return parsed


def parse_scalar(value: str) -> Any:
    if value in {"", "none"}:
        return [] if value == "none" else value
    if "|" in value:
        return parse_dof_list(value)
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in shlex.split(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = parse_scalar(value)
    return result


def parse_log_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(report.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def target_case_map(target_map: dict[str, Any]) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            targets[label] = int_list(case.get("direct_pspg_target_global_dofs"))
    return targets


def default_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, case in emission_cases.items():
        path = case.get("path")
        if isinstance(path, str) and path:
            paths[label] = Path(path)
    for value in explicit_logs:
        label, path = parse_log_arg(value)
        paths[label] = path
    return paths


def matching_entry(
    entry: dict[str, Any],
    *,
    operator: str,
    test_field: str,
    trial_field: str,
) -> bool:
    entry_test = entry.get("test")
    entry_trial = entry.get("trial")
    return (
        entry.get("op") == operator
        and isinstance(entry_test, str)
        and isinstance(entry_trial, str)
        and entry_test.lower() == test_field.lower()
        and entry_trial.lower() == trial_field.lower()
    )


def latest_local_matrix_batch(
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
        if "diagnostic=cut_volume_local_matrix_row_provenance" not in line:
            continue
        entry = parse_key_values(line)
        if not matching_entry(
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
        evidence["status"] = "local_matrix_entries_missing"
        return [], evidence
    evidence["status"] = "ok"
    evidence["latest_batch_entry_count"] = len(batches[-1])
    return batches[-1], evidence


def safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def safe_int(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, int) else default


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
                "partial_cut_rule_count": 0,
                "full_cell_rule_count": 0,
                "parent_cells": set(),
                "min_volume_fraction": None,
                "max_volume_fraction": None,
                "total_measure": 0.0,
                "max_active_quadrature_points": 0,
                "total_row_abs_sum": 0.0,
                "total_row_signed_sum": 0.0,
                "total_diag_abs": 0.0,
                "total_offdiag_abs_sum": 0.0,
                "total_positive_sum": 0.0,
                "total_negative_abs_sum": 0.0,
                "total_nonzero_count": 0,
                "max_rule_row_abs_sum": 0.0,
                "max_rule_row_abs_fraction": 0.0,
                "max_row_abs_fraction_in_rule_matrix": 0.0,
                "full_cell_row_abs_sum": 0.0,
                "partial_cut_row_abs_sum": 0.0,
            },
        )
        profile["rule_count"] += 1
        full_cell = entry.get("full_cell") == 1
        if full_cell:
            profile["full_cell_rule_count"] += 1
        else:
            profile["partial_cut_rule_count"] += 1
        parent_cell = entry.get("parent_cell")
        if isinstance(parent_cell, int):
            profile["parent_cells"].add(parent_cell)

        fraction = entry.get("volume_fraction")
        if isinstance(fraction, (int, float)):
            current_min = profile["min_volume_fraction"]
            current_max = profile["max_volume_fraction"]
            profile["min_volume_fraction"] = (
                float(fraction)
                if current_min is None
                else min(float(current_min), float(fraction))
            )
            profile["max_volume_fraction"] = (
                float(fraction)
                if current_max is None
                else max(float(current_max), float(fraction))
            )
        profile["total_measure"] += safe_float(entry.get("measure"))
        profile["max_active_quadrature_points"] = max(
            profile["max_active_quadrature_points"],
            safe_int(entry.get("active_quadrature_points")),
        )

        row_abs = safe_float(entry.get("row_abs_sum"))
        profile["total_row_abs_sum"] += row_abs
        profile["total_row_signed_sum"] += safe_float(entry.get("row_signed_sum"))
        profile["total_diag_abs"] += safe_float(entry.get("diag_abs"))
        profile["total_offdiag_abs_sum"] += safe_float(entry.get("offdiag_abs_sum"))
        profile["total_positive_sum"] += safe_float(entry.get("positive_sum"))
        profile["total_negative_abs_sum"] += safe_float(entry.get("negative_abs_sum"))
        profile["total_nonzero_count"] += safe_int(entry.get("nonzero_count"))
        profile["max_rule_row_abs_sum"] = max(
            profile["max_rule_row_abs_sum"],
            row_abs,
        )
        profile["max_row_abs_fraction_in_rule_matrix"] = max(
            profile["max_row_abs_fraction_in_rule_matrix"],
            safe_float(entry.get("row_abs_fraction")),
        )
        if full_cell:
            profile["full_cell_row_abs_sum"] += row_abs
        else:
            profile["partial_cut_row_abs_sum"] += row_abs

    normalized: dict[int, dict[str, Any]] = {}
    for row, profile in profiles.items():
        parent_cells = sorted(profile.pop("parent_cells"))
        profile["parent_cells"] = parent_cells
        profile["parent_cell_count"] = len(parent_cells)
        total_abs = profile["total_row_abs_sum"]
        if total_abs > 0.0:
            profile["max_rule_row_abs_fraction"] = (
                profile["max_rule_row_abs_sum"] / total_abs
            )
            profile["diag_abs_fraction"] = profile["total_diag_abs"] / total_abs
            profile["offdiag_abs_fraction"] = (
                profile["total_offdiag_abs_sum"] / total_abs
            )
            profile["full_cell_abs_fraction"] = (
                profile["full_cell_row_abs_sum"] / total_abs
            )
            profile["partial_cut_abs_fraction"] = (
                profile["partial_cut_row_abs_sum"] / total_abs
            )
            profile["signed_cancellation_ratio"] = (
                abs(profile["total_row_signed_sum"]) / total_abs
            )
        else:
            profile["diag_abs_fraction"] = 0.0
            profile["offdiag_abs_fraction"] = 0.0
            profile["full_cell_abs_fraction"] = 0.0
            profile["partial_cut_abs_fraction"] = 0.0
            profile["signed_cancellation_ratio"] = 0.0

        if profile["partial_cut_rule_count"] > 0 and profile["full_cell_rule_count"] > 0:
            support_class = "mixed_partial_and_full_cell_support"
        elif profile["partial_cut_rule_count"] > 0:
            support_class = "partial_cut_only_support"
        elif profile["full_cell_rule_count"] > 0:
            support_class = "full_cell_only_support"
        else:
            support_class = "missing_cut_volume_support"
        profile["cut_volume_support_class"] = support_class
        normalized[row] = profile
    return normalized


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def case_thresholds(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
) -> dict[str, float | None]:
    candidate_profiles = [profiles[row] for row in candidate_rows if row in profiles]

    def values(key: str) -> list[float]:
        return [
            float(profile[key])
            for profile in candidate_profiles
            if isinstance(profile.get(key), (int, float))
        ]

    total_abs_values = values("total_row_abs_sum")
    concentration_values = values("max_rule_row_abs_fraction")
    diag_fraction_values = values("diag_abs_fraction")
    rule_count_values = values("rule_count")
    return {
        "total_row_abs_sum_p10": percentile(total_abs_values, 0.10),
        "total_row_abs_sum_p25": percentile(total_abs_values, 0.25),
        "total_row_abs_sum_p90": percentile(total_abs_values, 0.90),
        "max_rule_row_abs_fraction_p75": percentile(concentration_values, 0.75),
        "max_rule_row_abs_fraction_p90": percentile(concentration_values, 0.90),
        "diag_abs_fraction_p25": percentile(diag_fraction_values, 0.25),
        "diag_abs_fraction_p75": percentile(diag_fraction_values, 0.75),
        "rule_count_p25": percentile(rule_count_values, 0.25),
    }


def threshold_le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def threshold_ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def selector_finding(
    *,
    selected_count: int,
    covered: list[int],
    uncovered: list[int],
    direct_target_count: int,
    max_target_ratio: float,
) -> str:
    selected_to_target_ratio = ratio(selected_count, direct_target_count)
    covers_targets = (
        direct_target_count > 0
        and len(covered) == direct_target_count
        and not uncovered
    )
    overbroad = (
        selected_to_target_ratio is not None
        and selected_to_target_ratio > max_target_ratio
    )
    if not covers_targets and overbroad:
        return "selector_overbroad_and_misses_targets"
    if not covers_targets:
        return "selector_misses_targets"
    if overbroad:
        return "selector_overbroad"
    return "selector_selective"


def selector_definitions(
    *,
    low_parent_cell_count: int,
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "local_matrix_profiled_candidate",
            "description": "Preferred candidates with local cut-volume matrix row metrics.",
            "threshold_key": None,
            "predicate": lambda profile: profile["rule_count"] > 0,
        },
        {
            "key": "local_matrix_low_total_abs_sum_p10",
            "description": "Profiled candidates in the bottom 10% of total local row absolute action.",
            "threshold_key": "total_row_abs_sum_p10",
            "predicate": lambda profile: threshold_le(
                profile,
                "total_row_abs_sum",
                thresholds["total_row_abs_sum_p10"],
            ),
        },
        {
            "key": "local_matrix_low_total_abs_sum_p25",
            "description": "Profiled candidates in the bottom 25% of total local row absolute action.",
            "threshold_key": "total_row_abs_sum_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "total_row_abs_sum",
                thresholds["total_row_abs_sum_p25"],
            ),
        },
        {
            "key": "local_matrix_high_total_abs_sum_p90",
            "description": "Profiled candidates in the top 10% of total local row absolute action.",
            "threshold_key": "total_row_abs_sum_p90",
            "predicate": lambda profile: threshold_ge(
                profile,
                "total_row_abs_sum",
                thresholds["total_row_abs_sum_p90"],
            ),
        },
        {
            "key": "local_matrix_high_rule_concentration_p75",
            "description": "Profiled candidates in the top quartile of max single-rule row-action concentration.",
            "threshold_key": "max_rule_row_abs_fraction_p75",
            "predicate": lambda profile: threshold_ge(
                profile,
                "max_rule_row_abs_fraction",
                thresholds["max_rule_row_abs_fraction_p75"],
            ),
        },
        {
            "key": "local_matrix_high_rule_concentration_p90",
            "description": "Profiled candidates in the top 10% of max single-rule row-action concentration.",
            "threshold_key": "max_rule_row_abs_fraction_p90",
            "predicate": lambda profile: threshold_ge(
                profile,
                "max_rule_row_abs_fraction",
                thresholds["max_rule_row_abs_fraction_p90"],
            ),
        },
        {
            "key": "local_matrix_low_total_abs_high_concentration",
            "description": "Bottom-quartile total row action and top-quartile single-rule concentration.",
            "threshold_key": "total_row_abs_sum_p25|max_rule_row_abs_fraction_p75",
            "predicate": lambda profile: (
                threshold_le(
                    profile,
                    "total_row_abs_sum",
                    thresholds["total_row_abs_sum_p25"],
                )
                and threshold_ge(
                    profile,
                    "max_rule_row_abs_fraction",
                    thresholds["max_rule_row_abs_fraction_p75"],
                )
            ),
        },
        {
            "key": "local_matrix_full_cell_only_low_total_abs_p25",
            "description": "Full-cell-only profiled candidates in the bottom quartile of total row action.",
            "threshold_key": "total_row_abs_sum_p25",
            "predicate": lambda profile: (
                profile["cut_volume_support_class"] == "full_cell_only_support"
                and threshold_le(
                    profile,
                    "total_row_abs_sum",
                    thresholds["total_row_abs_sum_p25"],
                )
            ),
        },
        {
            "key": "local_matrix_full_cell_dominant_abs_fraction",
            "description": "Profiled candidates whose local row action is almost entirely from full-cell equivalent rules.",
            "threshold_key": "fixed:0.999",
            "predicate": lambda profile: profile["full_cell_abs_fraction"] >= 0.999,
        },
        {
            "key": "local_matrix_low_diag_abs_fraction_p25",
            "description": "Profiled candidates in the bottom quartile of diagonal absolute-action fraction.",
            "threshold_key": "diag_abs_fraction_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "diag_abs_fraction",
                thresholds["diag_abs_fraction_p25"],
            ),
        },
        {
            "key": "local_matrix_low_rule_count_p25",
            "description": "Profiled candidates in the bottom quartile of contributing local-matrix rule count.",
            "threshold_key": "rule_count_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "rule_count",
                thresholds["rule_count_p25"],
            ),
        },
        {
            "key": "local_matrix_low_parent_cell_support",
            "description": f"Profiled candidates with parent-cell support <= {low_parent_cell_count}.",
            "threshold_key": f"fixed:{low_parent_cell_count}",
            "predicate": lambda profile: profile["parent_cell_count"]
            <= low_parent_cell_count,
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
            threshold_value = {
                key: thresholds.get(key)
                for key in threshold_key.split("|")
            }
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
        "finding": selector_finding(
            selected_count=len(selected),
            covered=covered,
            uncovered=uncovered,
            direct_target_count=len(target_rows),
            max_target_ratio=max_target_ratio,
        ),
        "direct_target_count": len(target_rows),
        "selected_count": len(selected),
        "selected_to_target_ratio": ratio(len(selected), len(target_rows)),
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "selected_global_dofs": selected,
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


def profile_summary(
    *,
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    target_profiles = {str(row): profiles[row] for row in target_rows if row in profiles}
    support_class_counts: dict[str, int] = {}
    for row in profiled_candidates:
        support_class = profiles[row].get("cut_volume_support_class", "unknown")
        support_class_counts[support_class] = support_class_counts.get(support_class, 0) + 1
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(target_set.intersection(profiles)),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "candidate_support_class_counts": support_class_counts,
        "target_profiles": target_profiles,
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
    low_parent_cell_count: int = 2,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    emission_cases = case_map(global_emission)
    target_cases = target_case_map(target_map)
    log_paths = default_log_paths(emission_cases, explicit_logs or [])

    cases: dict[str, dict[str, Any]] = {}
    selector_defs_by_case: dict[str, list[dict[str, Any]]] = {}
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = latest_local_matrix_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        profiles = row_profiles_from_entries(entries)
        thresholds = case_thresholds(profiles, candidate_rows)
        selector_defs = selector_definitions(
            low_parent_cell_count=low_parent_cell_count,
            thresholds=thresholds,
        )
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
        selector_template = selector_defs_by_case[first_label][selector_index]
        selectors.append(
            {
                "key": selector_template["key"],
                "description": selector_template["description"],
                "finding": aggregate_selector_finding(case_results),
                "cases": case_results,
            }
        )

    selective = [
        selector for selector in selectors if selector["finding"] == "selector_selective"
    ]
    overbroad = [
        selector for selector in selectors if "overbroad" in str(selector["finding"])
    ]
    misses = [selector for selector in selectors if "miss" in str(selector["finding"])]
    missing_cases = [
        label
        for label, case in cases.items()
        if case["log_evidence"].get("status") != "ok"
    ]

    if missing_cases:
        finding = "direct_pspg_cut_volume_local_matrix_evidence_missing"
        next_requirement = (
            "Regenerate Test02/Test10 short replay logs with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_PROVENANCE_DIAGNOSTIC=1 and "
            f"SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_PROVENANCE_OPERATOR={operator}."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_local_matrix_selector_selective"
        next_requirement = (
            "Prototype the selective local-matrix row-action gate and run the "
            "same short Test02/Test10 replay windows."
        )
    elif overbroad or misses:
        finding = (
            "direct_pspg_cut_volume_local_matrix_selectors_overbroad_or_miss_targets"
        )
        next_requirement = (
            "Do not promote local cut-volume row-action strength or "
            "concentration alone; the remaining discriminator must include "
            "stronger formulation-side pressure support context."
        )
    else:
        finding = "direct_pspg_cut_volume_local_matrix_selectivity_inconclusive"
        next_requirement = (
            "Regenerate local matrix provenance before selecting a formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for local cut-volume matrix row-action metrics "
            "in the direct PSPG pressure-gradient diagnostic operator."
        ),
        "global_emission_path": str(global_emission_path) if global_emission_path else None,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "candidate_key": candidate_key,
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
        "low_parent_cell_count": low_parent_cell_count,
        "max_target_ratio": max_target_ratio,
        "finding": finding,
        "missing_case_labels": missing_cases,
        "selective_selector_keys": [selector["key"] for selector in selective],
        "overbroad_selector_keys": [selector["key"] for selector in overbroad],
        "miss_selector_keys": [selector["key"] for selector in misses],
        "cases": list(cases.values()),
        "selectors": selectors,
        "next_requirement": next_requirement,
    }


def main() -> int:
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
        low_parent_cell_count=args.low_parent_cell_count,
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
