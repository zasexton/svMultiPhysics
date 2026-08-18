#!/usr/bin/env python3
"""Audit direct PSPG cut-volume sampled-column reference-geometry selectivity."""

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
GEOMETRY_TOLERANCE = 1.0e-10
LONG_EDGE_THRESHOLD = 1.0 + 1.0e-8


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
            "Build element-reference geometry profiles from sampled direct PSPG "
            "cut-volume local pressure-gradient columns, then compare fixed "
            "reference-edge selectors against audited Test02/Test10 target rows."
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


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def rounded_coord(value: float) -> float:
    return 0.0 if abs(value) <= GEOMETRY_TOLERANCE else round(value, 10)


def coord_key(x: float, y: float, z: float) -> str:
    return f"{rounded_coord(x):g},{rounded_coord(y):g},{rounded_coord(z):g}"


def add_count(counts: dict[str, int], key: str, amount: int = 1) -> None:
    counts[key] = counts.get(key, 0) + amount


def geometry_field_entry_count(entries: list[dict[str, Any]]) -> int:
    return sum(1 for entry in entries if "sampled_ref_edge_lengths" in entry)


def edge_geometry_class(length: float, dx: float, dy: float, dz: float) -> str:
    components = [dx, dy, dz]
    nonzero_count = sum(1 for value in components if abs(value) > GEOMETRY_TOLERANCE)
    if length <= GEOMETRY_TOLERANCE or nonzero_count == 0:
        return "zero_or_self_reference_edge"
    if nonzero_count == 1:
        return "axis_aligned_reference_edge"
    return "diagonal_reference_edge"


def edge_direction_class(dx: float, dy: float, dz: float) -> str:
    signs: list[str] = []
    for value in (dx, dy, dz):
        if value > GEOMETRY_TOLERANCE:
            signs.append("+")
        elif value < -GEOMETRY_TOLERANCE:
            signs.append("-")
        else:
            signs.append("0")
    return "".join(signs)


def init_geometry_profile(row: int, base: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "global_dof": row,
        "rule_count": int(base.get("rule_count", 0)) if isinstance(base, dict) else 0,
        "parent_cell_count": int(base.get("parent_cell_count", 0))
        if isinstance(base, dict)
        else 0,
        "sampled_offdiag_col_count": int(base.get("sampled_offdiag_col_count", 0))
        if isinstance(base, dict)
        else 0,
        "negative_sampled_offdiag_col_count": int(
            base.get("negative_sampled_offdiag_col_count", 0)
        )
        if isinstance(base, dict)
        else 0,
        "pressure_row_abs_sum": safe_float(base.get("pressure_row_abs_sum"))
        if isinstance(base, dict)
        else 0.0,
        "diag_abs": safe_float(base.get("diag_abs")) if isinstance(base, dict) else 0.0,
        "sampled_offdiag_abs_sum": safe_float(base.get("sampled_offdiag_abs_sum"))
        if isinstance(base, dict)
        else 0.0,
        "offdiag_geometry_sample_count": 0,
        "finite_geometry_edge_sample_count": 0,
        "missing_geometry_edge_sample_count": 0,
        "missing_reference_coordinate_sample_count": 0,
        "ref_edge_length_sum": 0.0,
        "ref_edge_length_weighted_sum": 0.0,
        "ref_edge_weight_sum": 0.0,
        "min_ref_edge_length": None,
        "max_ref_edge_length": None,
        "long_edge_sample_count": 0,
        "long_edge_weight_sum": 0.0,
        "axis_aligned_edge_sample_count": 0,
        "axis_aligned_edge_weight_sum": 0.0,
        "diagonal_edge_sample_count": 0,
        "diagonal_edge_weight_sum": 0.0,
        "row_origin_sample_count": 0,
        "row_axis_vertex_sample_count": 0,
        "row_ref_coord_counts": {},
        "sampled_col_ref_coord_counts": {},
        "reference_edge_class_counts": {},
        "reference_edge_direction_counts": {},
        "unique_ref_edge_lengths": set(),
    }


def update_min_max(profile: dict[str, Any], length: float) -> None:
    current_min = profile["min_ref_edge_length"]
    current_max = profile["max_ref_edge_length"]
    profile["min_ref_edge_length"] = (
        length if current_min is None else min(float(current_min), length)
    )
    profile["max_ref_edge_length"] = (
        length if current_max is None else max(float(current_max), length)
    )


def row_reference_class(x: float, y: float, z: float) -> str:
    nonzero = sum(
        1 for value in (x, y, z) if abs(value) > GEOMETRY_TOLERANCE
    )
    if nonzero == 0:
        return "origin_reference_node"
    if nonzero == 1:
        return "axis_reference_node"
    return "noncanonical_reference_node"


def reference_geometry_profiles(
    *,
    entries: list[dict[str, Any]],
    candidate_rows: list[int],
) -> dict[int, dict[str, Any]]:
    base_profiles = CS.row_profiles_from_column_entries(entries)
    candidate_set = set(candidate_rows)
    profiles: dict[int, dict[str, Any]] = {}

    for entry in entries:
        row = entry.get("row_dof")
        if not isinstance(row, int) or row not in candidate_set:
            continue
        profile = profiles.setdefault(
            row,
            init_geometry_profile(row, base_profiles.get(row)),
        )

        col_dofs = CS.as_int_list(entry.get("sampled_col_dofs"))
        col_values = CS.as_float_list(entry.get("sampled_col_values"))
        edge_lengths = CS.as_float_list(entry.get("sampled_ref_edge_lengths"))
        col_ref_x = CS.as_float_list(entry.get("sampled_col_ref_x"))
        col_ref_y = CS.as_float_list(entry.get("sampled_col_ref_y"))
        col_ref_z = CS.as_float_list(entry.get("sampled_col_ref_z"))
        row_x = finite_float(entry.get("row_ref_x"))
        row_y = finite_float(entry.get("row_ref_y"))
        row_z = finite_float(entry.get("row_ref_z"))
        row_has_coord = row_x is not None and row_y is not None and row_z is not None
        if row_has_coord:
            row_key = coord_key(row_x, row_y, row_z)
            add_count(profile["row_ref_coord_counts"], row_key)
            row_class = row_reference_class(row_x, row_y, row_z)
            if row_class == "origin_reference_node":
                profile["row_origin_sample_count"] += 1
            elif row_class == "axis_reference_node":
                profile["row_axis_vertex_sample_count"] += 1

        sample_count = min(
            len(col_dofs),
            len(col_values),
            len(edge_lengths),
            len(col_ref_x),
            len(col_ref_y),
            len(col_ref_z),
        )
        for index in range(sample_count):
            col_dof = col_dofs[index]
            if col_dof == row:
                continue
            profile["offdiag_geometry_sample_count"] += 1
            length = finite_float(edge_lengths[index])
            if length is None:
                profile["missing_geometry_edge_sample_count"] += 1
                continue
            profile["finite_geometry_edge_sample_count"] += 1
            profile["ref_edge_length_sum"] += length
            weight = abs(col_values[index])
            profile["ref_edge_length_weighted_sum"] += length * weight
            profile["ref_edge_weight_sum"] += weight
            update_min_max(profile, length)
            profile["unique_ref_edge_lengths"].add(round(length, 8))
            if length > LONG_EDGE_THRESHOLD:
                profile["long_edge_sample_count"] += 1
                profile["long_edge_weight_sum"] += weight

            col_x = finite_float(col_ref_x[index])
            col_y = finite_float(col_ref_y[index])
            col_z = finite_float(col_ref_z[index])
            if not row_has_coord or col_x is None or col_y is None or col_z is None:
                profile["missing_reference_coordinate_sample_count"] += 1
                continue
            col_key = coord_key(col_x, col_y, col_z)
            add_count(profile["sampled_col_ref_coord_counts"], col_key)
            dx = col_x - row_x
            dy = col_y - row_y
            dz = col_z - row_z
            edge_class = edge_geometry_class(length, dx, dy, dz)
            add_count(profile["reference_edge_class_counts"], edge_class)
            add_count(
                profile["reference_edge_direction_counts"],
                edge_direction_class(dx, dy, dz),
            )
            if edge_class == "axis_aligned_reference_edge":
                profile["axis_aligned_edge_sample_count"] += 1
                profile["axis_aligned_edge_weight_sum"] += weight
            elif edge_class == "diagonal_reference_edge":
                profile["diagonal_edge_sample_count"] += 1
                profile["diagonal_edge_weight_sum"] += weight

    normalized: dict[int, dict[str, Any]] = {}
    for row, profile in profiles.items():
        finite_count = profile["finite_geometry_edge_sample_count"]
        offdiag_count = profile["offdiag_geometry_sample_count"]
        weight_sum = profile["ref_edge_weight_sum"]
        row_coord_sample_count = sum(profile["row_ref_coord_counts"].values())
        profile["mean_ref_edge_length"] = ratio(
            profile["ref_edge_length_sum"],
            finite_count,
        )
        profile["weighted_mean_ref_edge_length"] = ratio(
            profile["ref_edge_length_weighted_sum"],
            weight_sum,
        )
        min_length = profile["min_ref_edge_length"]
        max_length = profile["max_ref_edge_length"]
        profile["min_ref_edge_length"] = (
            float(min_length) if isinstance(min_length, (int, float)) else None
        )
        profile["max_ref_edge_length"] = (
            float(max_length) if isinstance(max_length, (int, float)) else None
        )
        profile["ref_edge_length_range"] = (
            float(max_length) - float(min_length)
            if isinstance(max_length, (int, float))
            and isinstance(min_length, (int, float))
            else 0.0
        )
        profile["finite_geometry_edge_fraction"] = ratio(finite_count, offdiag_count)
        profile["missing_geometry_edge_fraction"] = ratio(
            profile["missing_geometry_edge_sample_count"],
            offdiag_count,
        )
        profile["axis_aligned_edge_fraction"] = ratio(
            profile["axis_aligned_edge_sample_count"],
            finite_count,
        )
        profile["weighted_axis_aligned_edge_fraction"] = ratio(
            profile["axis_aligned_edge_weight_sum"],
            weight_sum,
        )
        profile["diagonal_edge_fraction"] = ratio(
            profile["diagonal_edge_sample_count"],
            finite_count,
        )
        profile["weighted_diagonal_edge_fraction"] = ratio(
            profile["diagonal_edge_weight_sum"],
            weight_sum,
        )
        profile["long_edge_fraction"] = ratio(
            profile["long_edge_sample_count"],
            finite_count,
        )
        profile["weighted_long_edge_fraction"] = ratio(
            profile["long_edge_weight_sum"],
            weight_sum,
        )
        profile["row_origin_fraction"] = ratio(
            profile["row_origin_sample_count"],
            row_coord_sample_count,
        )
        profile["row_axis_vertex_fraction"] = ratio(
            profile["row_axis_vertex_sample_count"],
            row_coord_sample_count,
        )
        profile["unique_ref_edge_lengths"] = sorted(profile["unique_ref_edge_lengths"])
        profile["unique_ref_edge_length_count"] = len(profile["unique_ref_edge_lengths"])

        if finite_count == 0:
            geometry_class = "missing_reference_geometry"
        elif profile["missing_geometry_edge_sample_count"] > 0:
            geometry_class = "partial_reference_geometry"
        elif (
            profile["axis_aligned_edge_sample_count"] > 0
            and profile["diagonal_edge_sample_count"] > 0
        ):
            geometry_class = "mixed_axis_diagonal_reference_edges"
        elif profile["axis_aligned_edge_sample_count"] > 0:
            geometry_class = "axis_only_reference_edges"
        elif profile["diagonal_edge_sample_count"] > 0:
            geometry_class = "diagonal_only_reference_edges"
        else:
            geometry_class = "other_reference_geometry"
        profile["reference_geometry_class"] = geometry_class
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
        "mean_ref_edge_length",
        "weighted_mean_ref_edge_length",
        "max_ref_edge_length",
        "ref_edge_length_range",
        "axis_aligned_edge_fraction",
        "weighted_axis_aligned_edge_fraction",
        "diagonal_edge_fraction",
        "weighted_diagonal_edge_fraction",
        "long_edge_fraction",
        "weighted_long_edge_fraction",
        "row_origin_fraction",
        "row_axis_vertex_fraction",
        "finite_geometry_edge_sample_count",
        "unique_ref_edge_length_count",
    ]
    thresholds: dict[str, float | None] = {}
    for key in threshold_keys:
        values = metric_values(profiles, candidate_rows, key)
        thresholds[f"{key}_p25"] = LM.percentile(values, 0.25)
        thresholds[f"{key}_p75"] = LM.percentile(values, 0.75)
    return thresholds


def le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def selector_definitions(
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "geometry_profiled_candidate",
            "description": "Preferred candidates with sampled reference-geometry profiles.",
            "threshold_key": None,
            "predicate": lambda profile: True,
        },
        {
            "key": "geometry_complete_reference_edges",
            "description": "Rows whose sampled offdiagonal columns all have finite reference-edge lengths.",
            "threshold_key": "fixed:finite_all_offdiag_edges",
            "predicate": lambda profile: (
                profile["finite_geometry_edge_sample_count"] > 0
                and profile["missing_geometry_edge_sample_count"] == 0
            ),
        },
        {
            "key": "geometry_mixed_axis_diagonal_edges",
            "description": "Rows with both axis-aligned and diagonal sampled reference edges.",
            "threshold_key": "fixed:mixed_axis_diagonal_reference_edges",
            "predicate": lambda profile: (
                profile["reference_geometry_class"]
                == "mixed_axis_diagonal_reference_edges"
            ),
        },
        {
            "key": "geometry_axis_only_edges",
            "description": "Rows whose finite sampled reference edges are all axis-aligned.",
            "threshold_key": "fixed:axis_only_reference_edges",
            "predicate": lambda profile: (
                profile["reference_geometry_class"] == "axis_only_reference_edges"
            ),
        },
        {
            "key": "geometry_has_diagonal_edges",
            "description": "Rows with at least one diagonal sampled reference edge.",
            "threshold_key": "fixed:diagonal_edge_fraction_gt_0",
            "predicate": lambda profile: profile["diagonal_edge_fraction"] > 0.0,
        },
        {
            "key": "geometry_all_unit_edges",
            "description": "Rows whose finite sampled reference edges are all unit-length.",
            "threshold_key": "fixed:long_edge_fraction_0",
            "predicate": lambda profile: (
                profile["finite_geometry_edge_sample_count"] > 0
                and profile["long_edge_fraction"] == 0.0
            ),
        },
        {
            "key": "geometry_mean_ref_edge_length_tail",
            "description": "Rows in either low or high mean reference-edge length tail.",
            "threshold_key": "mean_ref_edge_length_p25|mean_ref_edge_length_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "mean_ref_edge_length",
                    thresholds["mean_ref_edge_length_p25"],
                )
                or ge(
                    profile,
                    "mean_ref_edge_length",
                    thresholds["mean_ref_edge_length_p75"],
                )
            ),
        },
        {
            "key": "geometry_weighted_mean_ref_edge_length_tail",
            "description": "Rows in either low or high coefficient-weighted mean reference-edge length tail.",
            "threshold_key": (
                "weighted_mean_ref_edge_length_p25|"
                "weighted_mean_ref_edge_length_p75"
            ),
            "predicate": lambda profile: (
                le(
                    profile,
                    "weighted_mean_ref_edge_length",
                    thresholds["weighted_mean_ref_edge_length_p25"],
                )
                or ge(
                    profile,
                    "weighted_mean_ref_edge_length",
                    thresholds["weighted_mean_ref_edge_length_p75"],
                )
            ),
        },
        {
            "key": "geometry_high_max_ref_edge_length_p75",
            "description": "Rows in the top quartile of maximum sampled reference-edge length.",
            "threshold_key": "max_ref_edge_length_p75",
            "predicate": lambda profile: ge(
                profile,
                "max_ref_edge_length",
                thresholds["max_ref_edge_length_p75"],
            ),
        },
        {
            "key": "geometry_high_ref_edge_length_range_p75",
            "description": "Rows in the top quartile of sampled reference-edge length range.",
            "threshold_key": "ref_edge_length_range_p75",
            "predicate": lambda profile: ge(
                profile,
                "ref_edge_length_range",
                thresholds["ref_edge_length_range_p75"],
            ),
        },
        {
            "key": "geometry_axis_fraction_tail",
            "description": "Rows in either low or high axis-aligned reference-edge fraction tail.",
            "threshold_key": (
                "axis_aligned_edge_fraction_p25|axis_aligned_edge_fraction_p75"
            ),
            "predicate": lambda profile: (
                le(
                    profile,
                    "axis_aligned_edge_fraction",
                    thresholds["axis_aligned_edge_fraction_p25"],
                )
                or ge(
                    profile,
                    "axis_aligned_edge_fraction",
                    thresholds["axis_aligned_edge_fraction_p75"],
                )
            ),
        },
        {
            "key": "geometry_weighted_axis_fraction_tail",
            "description": "Rows in either low or high coefficient-weighted axis-edge fraction tail.",
            "threshold_key": (
                "weighted_axis_aligned_edge_fraction_p25|"
                "weighted_axis_aligned_edge_fraction_p75"
            ),
            "predicate": lambda profile: (
                le(
                    profile,
                    "weighted_axis_aligned_edge_fraction",
                    thresholds["weighted_axis_aligned_edge_fraction_p25"],
                )
                or ge(
                    profile,
                    "weighted_axis_aligned_edge_fraction",
                    thresholds["weighted_axis_aligned_edge_fraction_p75"],
                )
            ),
        },
        {
            "key": "geometry_diagonal_fraction_tail",
            "description": "Rows in either low or high diagonal reference-edge fraction tail.",
            "threshold_key": "diagonal_edge_fraction_p25|diagonal_edge_fraction_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "diagonal_edge_fraction",
                    thresholds["diagonal_edge_fraction_p25"],
                )
                or ge(
                    profile,
                    "diagonal_edge_fraction",
                    thresholds["diagonal_edge_fraction_p75"],
                )
            ),
        },
        {
            "key": "geometry_weighted_diagonal_fraction_tail",
            "description": "Rows in either low or high coefficient-weighted diagonal-edge fraction tail.",
            "threshold_key": (
                "weighted_diagonal_edge_fraction_p25|"
                "weighted_diagonal_edge_fraction_p75"
            ),
            "predicate": lambda profile: (
                le(
                    profile,
                    "weighted_diagonal_edge_fraction",
                    thresholds["weighted_diagonal_edge_fraction_p25"],
                )
                or ge(
                    profile,
                    "weighted_diagonal_edge_fraction",
                    thresholds["weighted_diagonal_edge_fraction_p75"],
                )
            ),
        },
        {
            "key": "geometry_long_edge_fraction_tail",
            "description": "Rows in either low or high long reference-edge fraction tail.",
            "threshold_key": "long_edge_fraction_p25|long_edge_fraction_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "long_edge_fraction",
                    thresholds["long_edge_fraction_p25"],
                )
                or ge(
                    profile,
                    "long_edge_fraction",
                    thresholds["long_edge_fraction_p75"],
                )
            ),
        },
        {
            "key": "geometry_row_origin_fraction_tail",
            "description": "Rows in either low or high origin-row reference-node fraction tail.",
            "threshold_key": "row_origin_fraction_p25|row_origin_fraction_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "row_origin_fraction",
                    thresholds["row_origin_fraction_p25"],
                )
                or ge(
                    profile,
                    "row_origin_fraction",
                    thresholds["row_origin_fraction_p75"],
                )
            ),
        },
        {
            "key": "geometry_unique_length_count_tail",
            "description": "Rows in either low or high unique reference-edge length-count tail.",
            "threshold_key": (
                "unique_ref_edge_length_count_p25|unique_ref_edge_length_count_p75"
            ),
            "predicate": lambda profile: (
                le(
                    profile,
                    "unique_ref_edge_length_count",
                    thresholds["unique_ref_edge_length_count_p25"],
                )
                or ge(
                    profile,
                    "unique_ref_edge_length_count",
                    thresholds["unique_ref_edge_length_count_p75"],
                )
            ),
        },
        {
            "key": "geometry_finite_edge_count_tail",
            "description": "Rows in either low or high finite reference-edge sample-count tail.",
            "threshold_key": (
                "finite_geometry_edge_sample_count_p25|"
                "finite_geometry_edge_sample_count_p75"
            ),
            "predicate": lambda profile: (
                le(
                    profile,
                    "finite_geometry_edge_sample_count",
                    thresholds["finite_geometry_edge_sample_count_p25"],
                )
                or ge(
                    profile,
                    "finite_geometry_edge_sample_count",
                    thresholds["finite_geometry_edge_sample_count_p75"],
                )
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
    geometry_class_counts: dict[str, int] = {}
    target_geometry_class_counts: dict[str, int] = {}
    for row in profiled_candidates:
        geometry_class = profiles[row].get("reference_geometry_class", "unknown")
        geometry_class_counts[geometry_class] = (
            geometry_class_counts.get(geometry_class, 0) + 1
        )
    for row in target_rows:
        if row in profiles:
            geometry_class = profiles[row].get("reference_geometry_class", "unknown")
            target_geometry_class_counts[geometry_class] = (
                target_geometry_class_counts.get(geometry_class, 0) + 1
            )
    return {
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len([row for row in target_rows if row in profiles]),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "candidate_reference_geometry_class_counts": geometry_class_counts,
        "target_reference_geometry_class_counts": target_geometry_class_counts,
        "target_profiles": {
            str(row): profiles[row] for row in target_rows if row in profiles
        },
    }


def default_geometry_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths = LM.default_log_paths(emission_cases, [])
    for label, path in list(paths.items()):
        sibling = path.with_name("run_direct_pspg_cut_volume_column_geometry.log")
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
    log_paths = default_geometry_log_paths(emission_cases, explicit_logs or [])

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
        field_count = geometry_field_entry_count(entries)
        evidence["geometry_field_entry_count"] = field_count
        if evidence.get("status") != "ok":
            missing_cases.append(label)
        elif field_count == 0:
            evidence["status"] = "reference_geometry_fields_missing"
            missing_cases.append(label)

        profiles = reference_geometry_profiles(
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
        finding = "direct_pspg_cut_volume_column_geometry_selectivity_evidence_missing"
        next_requirement = (
            "Rerun the short Test02/Test10 windows with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_COLUMN_GEOMETRY_DIAGNOSTIC=1 "
            "or pass explicit --log paths to column-geometry logs."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_column_geometry_selector_identified"
        next_requirement = (
            "Translate the selective sampled reference-geometry selector into "
            "a bounded formulation-side replay probe."
        )
    else:
        finding = "direct_pspg_cut_volume_column_geometry_selectors_overbroad_or_miss_targets"
        next_requirement = (
            "Reference-node edge geometry did not isolate the direct PSPG target "
            "rows; move to quadrature/cut-interface geometry or a formulation-"
            "derived support/coupling balance."
        )

    return {
        "scope": (
            "Selectivity audit for sampled pressure-pressure reference-edge "
            "geometry from direct PSPG cut-volume pressure-gradient rows."
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
