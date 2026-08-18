#!/usr/bin/env python3
"""Audit direct PSPG cut-volume quadrature-geometry selectivity."""

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
NONZERO_TOLERANCE = 1.0e-12


def _load_column_support_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_column_support_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


CS = _load_column_support_module()
LM = CS.LM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cut-volume quadrature-point geometry profiles from direct "
            "PSPG local pressure-gradient rows, then compare bounded selectors "
            "against audited Test02/Test10 target rows."
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


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_float(value: Any, default: float = 0.0) -> float:
    result = finite_float(value)
    return result if result is not None else default


def finite_metric(profile: dict[str, Any], key: str) -> float:
    return safe_float(profile.get(key))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def qpoint_field_entry_count(entries: list[dict[str, Any]]) -> int:
    return sum(1 for entry in entries if "cut_qpoint_count" in entry)


def init_profile(row: int) -> dict[str, Any]:
    return {
        "global_dof": row,
        "quadrature_geometry_rule_count": 0,
        "cut_qpoint_field_rule_count": 0,
        "cut_rule_constructions": set(),
        "cut_rule_frames": set(),
        "cut_rule_exact_orders": set(),
        "cut_rule_achieved_orders": set(),
        "cut_qpoint_counts": set(),
        "cut_qpoint_weight_sum_total": 0.0,
        "cut_qpoint_abs_weight_sum_total": 0.0,
        "cut_qpoint_reference_measure_factor_sum_total": 0.0,
        "cut_qpoint_max_abs_weight_fraction_max": 0.0,
        "cut_qpoint_level_set_max_abs_max": 0.0,
        "cut_qpoint_level_set_mean_abs_sum": 0.0,
        "cut_qpoint_gradient_norm_max": 0.0,
        "cut_qpoint_gradient_norm_mean_sum": 0.0,
        "cut_qpoint_rms_radius_sum": 0.0,
        "cut_qpoint_max_radius_max": 0.0,
        "cut_qpoint_span_x_sum": 0.0,
        "cut_qpoint_span_y_sum": 0.0,
        "cut_qpoint_span_z_sum": 0.0,
        "cut_qpoint_centroid_x_values": [],
        "cut_qpoint_centroid_y_values": [],
        "cut_qpoint_centroid_z_values": [],
        "cut_qpoint_parent_centroid_x_values": [],
        "cut_qpoint_parent_centroid_y_values": [],
        "cut_qpoint_parent_centroid_z_values": [],
        "cut_qpoint_normal_mean_x_sum": 0.0,
        "cut_qpoint_normal_mean_y_sum": 0.0,
        "cut_qpoint_normal_mean_z_sum": 0.0,
        "row_to_cut_qpoint_centroid_distance_sum": 0.0,
        "row_to_cut_qpoint_centroid_distance_count": 0,
        "parent_cells": set(),
    }


def range_or_zero(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def append_value(profile: dict[str, Any], key: str, value: Any) -> float | None:
    result = finite_float(value)
    if result is not None:
        profile[key].append(result)
    return result


def reference_distance(entry: dict[str, Any]) -> float | None:
    row_x = finite_float(entry.get("row_ref_x"))
    row_y = finite_float(entry.get("row_ref_y"))
    row_z = finite_float(entry.get("row_ref_z"))
    centroid_x = finite_float(entry.get("cut_qpoint_centroid_x"))
    centroid_y = finite_float(entry.get("cut_qpoint_centroid_y"))
    centroid_z = finite_float(entry.get("cut_qpoint_centroid_z"))
    if None in {row_x, row_y, row_z, centroid_x, centroid_y, centroid_z}:
        return None
    dx = centroid_x - row_x
    dy = centroid_y - row_y
    dz = centroid_z - row_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def quadrature_geometry_profiles(
    *,
    entries: list[dict[str, Any]],
    candidate_rows: list[int],
) -> dict[int, dict[str, Any]]:
    candidate_set = set(candidate_rows)
    profiles: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row = entry.get("row_dof")
        if not isinstance(row, int) or row not in candidate_set:
            continue
        profile = profiles.setdefault(row, init_profile(row))
        profile["quadrature_geometry_rule_count"] += 1
        parent_cell = entry.get("parent_cell")
        if isinstance(parent_cell, int):
            profile["parent_cells"].add(parent_cell)
        if "cut_qpoint_count" not in entry:
            continue
        profile["cut_qpoint_field_rule_count"] += 1
        for key, target in (
            ("cut_rule_construction", "cut_rule_constructions"),
            ("cut_rule_frame", "cut_rule_frames"),
            ("cut_rule_exact_polynomial_order", "cut_rule_exact_orders"),
            ("cut_rule_achieved_quadrature_order", "cut_rule_achieved_orders"),
            ("cut_qpoint_count", "cut_qpoint_counts"),
        ):
            value = entry.get(key)
            if isinstance(value, int):
                profile[target].add(value)
        profile["cut_qpoint_weight_sum_total"] += safe_float(
            entry.get("cut_qpoint_weight_sum")
        )
        profile["cut_qpoint_abs_weight_sum_total"] += safe_float(
            entry.get("cut_qpoint_abs_weight_sum")
        )
        profile["cut_qpoint_reference_measure_factor_sum_total"] += safe_float(
            entry.get("cut_qpoint_reference_measure_factor_sum")
        )
        profile["cut_qpoint_max_abs_weight_fraction_max"] = max(
            profile["cut_qpoint_max_abs_weight_fraction_max"],
            safe_float(entry.get("cut_qpoint_max_abs_weight_fraction")),
        )
        profile["cut_qpoint_level_set_max_abs_max"] = max(
            profile["cut_qpoint_level_set_max_abs_max"],
            safe_float(entry.get("cut_qpoint_level_set_max_abs")),
        )
        profile["cut_qpoint_level_set_mean_abs_sum"] += safe_float(
            entry.get("cut_qpoint_level_set_mean_abs")
        )
        profile["cut_qpoint_gradient_norm_max"] = max(
            profile["cut_qpoint_gradient_norm_max"],
            safe_float(entry.get("cut_qpoint_gradient_norm_max")),
        )
        profile["cut_qpoint_gradient_norm_mean_sum"] += safe_float(
            entry.get("cut_qpoint_gradient_norm_mean")
        )
        profile["cut_qpoint_rms_radius_sum"] += safe_float(
            entry.get("cut_qpoint_rms_radius")
        )
        profile["cut_qpoint_max_radius_max"] = max(
            profile["cut_qpoint_max_radius_max"],
            safe_float(entry.get("cut_qpoint_max_radius")),
        )
        for component in ("x", "y", "z"):
            profile[f"cut_qpoint_span_{component}_sum"] += safe_float(
                entry.get(f"cut_qpoint_span_{component}")
            )
            append_value(
                profile,
                f"cut_qpoint_centroid_{component}_values",
                entry.get(f"cut_qpoint_centroid_{component}"),
            )
            append_value(
                profile,
                f"cut_qpoint_parent_centroid_{component}_values",
                entry.get(f"cut_qpoint_parent_centroid_{component}"),
            )
            profile[f"cut_qpoint_normal_mean_{component}_sum"] += safe_float(
                entry.get(f"cut_qpoint_normal_mean_{component}")
            )
        distance = reference_distance(entry)
        if distance is not None:
            profile["row_to_cut_qpoint_centroid_distance_sum"] += distance
            profile["row_to_cut_qpoint_centroid_distance_count"] += 1

    normalized: dict[int, dict[str, Any]] = {}
    for row, profile in profiles.items():
        field_count = profile["cut_qpoint_field_rule_count"]
        for key in (
            "cut_rule_constructions",
            "cut_rule_frames",
            "cut_rule_exact_orders",
            "cut_rule_achieved_orders",
            "cut_qpoint_counts",
            "parent_cells",
        ):
            values = sorted(profile[key])
            profile[key] = values
            profile[f"{key}_count"] = len(values)
        profile["parent_cell_count"] = len(profile["parent_cells"])
        profile["cut_qpoint_field_fraction"] = ratio(
            field_count,
            profile["quadrature_geometry_rule_count"],
        )
        profile["cut_qpoint_level_set_mean_abs_mean"] = ratio(
            profile["cut_qpoint_level_set_mean_abs_sum"],
            field_count,
        )
        profile["cut_qpoint_gradient_norm_mean"] = ratio(
            profile["cut_qpoint_gradient_norm_mean_sum"],
            field_count,
        )
        profile["cut_qpoint_rms_radius_mean"] = ratio(
            profile["cut_qpoint_rms_radius_sum"],
            field_count,
        )
        profile["cut_qpoint_span_x_mean"] = ratio(
            profile["cut_qpoint_span_x_sum"],
            field_count,
        )
        profile["cut_qpoint_span_y_mean"] = ratio(
            profile["cut_qpoint_span_y_sum"],
            field_count,
        )
        profile["cut_qpoint_span_z_mean"] = ratio(
            profile["cut_qpoint_span_z_sum"],
            field_count,
        )
        profile["cut_qpoint_centroid_x_range"] = range_or_zero(
            profile.pop("cut_qpoint_centroid_x_values")
        )
        profile["cut_qpoint_centroid_y_range"] = range_or_zero(
            profile.pop("cut_qpoint_centroid_y_values")
        )
        profile["cut_qpoint_centroid_z_range"] = range_or_zero(
            profile.pop("cut_qpoint_centroid_z_values")
        )
        profile["cut_qpoint_parent_centroid_x_range"] = range_or_zero(
            profile.pop("cut_qpoint_parent_centroid_x_values")
        )
        profile["cut_qpoint_parent_centroid_y_range"] = range_or_zero(
            profile.pop("cut_qpoint_parent_centroid_y_values")
        )
        profile["cut_qpoint_parent_centroid_z_range"] = range_or_zero(
            profile.pop("cut_qpoint_parent_centroid_z_values")
        )
        profile["cut_qpoint_normal_mean_x_mean"] = ratio(
            profile["cut_qpoint_normal_mean_x_sum"],
            field_count,
        )
        profile["cut_qpoint_normal_mean_y_mean"] = ratio(
            profile["cut_qpoint_normal_mean_y_sum"],
            field_count,
        )
        profile["cut_qpoint_normal_mean_z_mean"] = ratio(
            profile["cut_qpoint_normal_mean_z_sum"],
            field_count,
        )
        profile["row_to_cut_qpoint_centroid_distance_mean"] = ratio(
            profile["row_to_cut_qpoint_centroid_distance_sum"],
            profile["row_to_cut_qpoint_centroid_distance_count"],
        )
        if field_count == 0:
            geometry_class = "missing_cut_qpoint_geometry"
        elif profile["cut_qpoint_level_set_max_abs_max"] > NONZERO_TOLERANCE:
            geometry_class = "nonzero_level_set_residual_qpoints"
        elif profile["cut_qpoint_gradient_norm_max"] > NONZERO_TOLERANCE:
            geometry_class = "nonzero_gradient_norm_qpoints"
        elif profile["cut_qpoint_counts_count"] == 1:
            geometry_class = "uniform_full_cell_qpoint_geometry"
        else:
            geometry_class = "mixed_qpoint_geometry"
        profile["cut_qpoint_geometry_class"] = geometry_class
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
    threshold_keys = [
        "parent_cell_count",
        "cut_qpoint_field_fraction",
        "cut_qpoint_max_abs_weight_fraction_max",
        "cut_qpoint_rms_radius_mean",
        "cut_qpoint_max_radius_max",
        "cut_qpoint_span_x_mean",
        "cut_qpoint_span_y_mean",
        "cut_qpoint_span_z_mean",
        "cut_qpoint_centroid_x_range",
        "cut_qpoint_centroid_y_range",
        "cut_qpoint_centroid_z_range",
        "cut_qpoint_parent_centroid_x_range",
        "cut_qpoint_parent_centroid_y_range",
        "cut_qpoint_parent_centroid_z_range",
        "cut_qpoint_level_set_max_abs_max",
        "cut_qpoint_gradient_norm_max",
        "row_to_cut_qpoint_centroid_distance_mean",
    ]
    thresholds: dict[str, float | None] = {}
    for key in threshold_keys:
        values = metric_values(profiles, candidate_rows, key)
        thresholds[f"{key}_p25"] = LM.percentile(values, 0.25)
        thresholds[f"{key}_p75"] = LM.percentile(values, 0.75)
    return thresholds


def le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and finite_metric(profile, key) <= threshold


def ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and finite_metric(profile, key) >= threshold


def tail_selector(
    profile: dict[str, Any],
    key: str,
    thresholds: dict[str, float | None],
) -> bool:
    return le(profile, key, thresholds[f"{key}_p25"]) or ge(
        profile,
        key,
        thresholds[f"{key}_p75"],
    )


def selector_definitions(
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "qgeom_profiled_candidate",
            "description": "Preferred candidates with q-point geometry profiles.",
            "threshold_key": None,
            "predicate": lambda profile: True,
        },
        {
            "key": "qgeom_complete_fields",
            "description": "Rows whose rules all have q-point geometry fields.",
            "threshold_key": "fixed:all_rules_with_qpoint_fields",
            "predicate": lambda profile: profile["cut_qpoint_field_fraction"] == 1.0,
        },
        {
            "key": "qgeom_uniform_full_cell_class",
            "description": "Rows with uniform full-cell q-point geometry.",
            "threshold_key": "fixed:uniform_full_cell_qpoint_geometry",
            "predicate": lambda profile: (
                profile["cut_qpoint_geometry_class"]
                == "uniform_full_cell_qpoint_geometry"
            ),
        },
        {
            "key": "qgeom_nonzero_level_set_residual",
            "description": "Rows with any nonzero q-point level-set residual.",
            "threshold_key": "fixed:level_set_residual_gt_1e-12",
            "predicate": lambda profile: (
                profile["cut_qpoint_level_set_max_abs_max"] > NONZERO_TOLERANCE
            ),
        },
        {
            "key": "qgeom_nonzero_gradient_norm",
            "description": "Rows with any nonzero q-point level-set gradient norm.",
            "threshold_key": "fixed:gradient_norm_gt_1e-12",
            "predicate": lambda profile: (
                profile["cut_qpoint_gradient_norm_max"] > NONZERO_TOLERANCE
            ),
        },
        {
            "key": "qgeom_parent_cell_count_tail",
            "description": "Rows in either low or high parent-cell count tail.",
            "threshold_key": "parent_cell_count_p25|parent_cell_count_p75",
            "predicate": lambda profile: tail_selector(
                profile,
                "parent_cell_count",
                thresholds,
            ),
        },
        {
            "key": "qgeom_weight_concentration_tail",
            "description": "Rows in either low or high q-point weight-concentration tail.",
            "threshold_key": (
                "cut_qpoint_max_abs_weight_fraction_max_p25|"
                "cut_qpoint_max_abs_weight_fraction_max_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "cut_qpoint_max_abs_weight_fraction_max",
                thresholds,
            ),
        },
        {
            "key": "qgeom_radius_tail",
            "description": "Rows in either low or high q-point radius tail.",
            "threshold_key": "cut_qpoint_rms_radius_mean_p25|cut_qpoint_rms_radius_mean_p75",
            "predicate": lambda profile: tail_selector(
                profile,
                "cut_qpoint_rms_radius_mean",
                thresholds,
            ),
        },
        {
            "key": "qgeom_span_x_tail",
            "description": "Rows in either low or high q-point x-span tail.",
            "threshold_key": "cut_qpoint_span_x_mean_p25|cut_qpoint_span_x_mean_p75",
            "predicate": lambda profile: tail_selector(
                profile,
                "cut_qpoint_span_x_mean",
                thresholds,
            ),
        },
        {
            "key": "qgeom_parent_centroid_x_range_tail",
            "description": "Rows in either low or high parent-coordinate centroid x-range tail.",
            "threshold_key": (
                "cut_qpoint_parent_centroid_x_range_p25|"
                "cut_qpoint_parent_centroid_x_range_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "cut_qpoint_parent_centroid_x_range",
                thresholds,
            ),
        },
        {
            "key": "qgeom_row_to_centroid_distance_tail",
            "description": "Rows in either low or high row-to-qpoint-centroid distance tail.",
            "threshold_key": (
                "row_to_cut_qpoint_centroid_distance_mean_p25|"
                "row_to_cut_qpoint_centroid_distance_mean_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "row_to_cut_qpoint_centroid_distance_mean",
                thresholds,
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
            threshold_value = {
                key: thresholds.get(key)
                for key in threshold_key.split("|")
                if not key.startswith("fixed:")
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
    class_counts: dict[str, int] = {}
    target_class_counts: dict[str, int] = {}
    for row in profiled_candidates:
        geometry_class = profiles[row].get("cut_qpoint_geometry_class", "unknown")
        class_counts[geometry_class] = class_counts.get(geometry_class, 0) + 1
    for row in target_rows:
        if row in profiles:
            geometry_class = profiles[row].get("cut_qpoint_geometry_class", "unknown")
            target_class_counts[geometry_class] = (
                target_class_counts.get(geometry_class, 0) + 1
            )
    return {
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len([row for row in target_rows if row in profiles]),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "candidate_cut_qpoint_geometry_class_counts": class_counts,
        "target_cut_qpoint_geometry_class_counts": target_class_counts,
        "target_profiles": {
            str(row): profiles[row] for row in target_rows if row in profiles
        },
    }


def default_quadrature_geometry_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths = LM.default_log_paths(emission_cases, [])
    for label, path in list(paths.items()):
        sibling = path.with_name("run_direct_pspg_cut_volume_quadrature_geometry.log")
        if sibling.exists():
            paths[label] = sibling
    for value in explicit_logs:
        label, path = LM.parse_log_arg(value)
        paths[label] = path
    return paths


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
    log_paths = default_quadrature_geometry_log_paths(
        emission_cases,
        explicit_logs or [],
    )

    cases: list[dict[str, Any]] = []
    selector_cases: dict[str, list[dict[str, Any]]] = {}
    missing_cases: list[str] = []

    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = CS.latest_column_support_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        field_count = qpoint_field_entry_count(entries)
        evidence["cut_qpoint_field_entry_count"] = field_count
        if evidence.get("status") != "ok":
            missing_cases.append(label)
        elif field_count == 0:
            evidence["status"] = "cut_qpoint_geometry_fields_missing"
            missing_cases.append(label)

        profiles = quadrature_geometry_profiles(
            entries=entries,
            candidate_rows=candidate_rows,
        )
        thresholds = case_thresholds(profiles, candidate_rows)
        selectors = selector_definitions(thresholds)
        for selector in selectors:
            selector_cases.setdefault(selector["key"], []).append(
                evaluate_selector_case(
                    label=label,
                    selector=selector,
                    candidate_rows=candidate_rows,
                    target_rows=target_rows,
                    profiles=profiles,
                    thresholds=thresholds,
                    max_target_ratio=max_target_ratio,
                )
            )

        cases.append(
            {
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
            }
        )

    selectors = [
        {
            "key": key,
            "finding": LM.aggregate_selector_finding(case_results),
            "cases": case_results,
        }
        for key, case_results in selector_cases.items()
    ]
    selective = [
        selector["key"]
        for selector in selectors
        if selector["finding"] == "selector_selective"
    ]
    overbroad = [
        selector["key"]
        for selector in selectors
        if selector["finding"] == "selector_overbroad"
    ]
    miss = [
        selector["key"]
        for selector in selectors
        if "miss" in selector["finding"]
    ]

    if missing_cases:
        finding = (
            "direct_pspg_cut_volume_quadrature_geometry_selectivity_evidence_missing"
        )
        next_requirement = (
            "Rerun the short Test02/Test10 windows with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_QUADRATURE_GEOMETRY_DIAGNOSTIC=1 "
            "or pass explicit --log paths to quadrature-geometry logs."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_quadrature_geometry_selector_identified"
        next_requirement = (
            "Translate the selective cut-volume quadrature geometry selector "
            "into a bounded formulation-side replay probe."
        )
    else:
        finding = (
            "direct_pspg_cut_volume_quadrature_geometry_selectors_"
            "overbroad_or_miss_targets"
        )
        next_requirement = (
            "Cut-volume q-point geometry did not isolate the direct PSPG "
            "target rows; move to formulation-derived pressure-gradient "
            "support/coupling balance or a richer cut-interface proximity field."
        )

    return {
        "scope": (
            "Selectivity audit for cut-volume quadrature-point geometry from "
            "direct PSPG pressure-gradient rows."
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
        "selective_selector_keys": selective,
        "overbroad_selector_keys": overbroad,
        "miss_selector_keys": miss,
        "cases": cases,
        "selectors": selectors,
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
