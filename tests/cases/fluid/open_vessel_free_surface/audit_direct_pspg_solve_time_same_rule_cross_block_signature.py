#!/usr/bin/env python3
"""Audit same-rule PP/PV local signatures for solve-time direct PSPG rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_direct_pspg_solve_time_provenance_replay import (  # noqa: E402
    evaluate_selector,
)
from audit_direct_pspg_solve_time_sampled_column_selectivity import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    load_json,
    read_provenance_log,
    sample_shape_class,
    sample_signature,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_same_rule_cross_block_signature_20260607.json"
)

MAGNITUDE_FEATURES = [
    "pressure_pressure_abs_sum",
    "pressure_velocity_abs_sum",
    "pressure_velocity_to_pressure_pressure_abs_ratio",
    "pressure_pressure_abs_sum_per_parent_cell",
    "pressure_velocity_abs_sum_per_parent_cell",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether same-rule sampled pressure-pressure and "
            "pressure-velocity local signatures, combined with non-update "
            "magnitude ranges, produce a targeted direct PSPG replay candidate."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def float_or_zero(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def empty_row_stats() -> dict[str, Any]:
    return {
        "record_count": 0,
        "pressure_pressure_records": 0,
        "pressure_velocity_records": 0,
        "pressure_pressure_abs_sum": 0.0,
        "pressure_velocity_abs_sum": 0.0,
        "pressure_pressure_parent_cells": set(),
        "pressure_velocity_parent_cells": set(),
        "same_rule_cross_block_signatures": set(),
        "same_rule_cross_block_shape_pairs": set(),
        "all_pressure_update_sign_unused": True,
        "all_diagnostic_only": True,
    }


def row_feature_values(stats: dict[str, Any]) -> dict[str, float]:
    pressure_pressure_abs_sum = float(stats["pressure_pressure_abs_sum"])
    pressure_velocity_abs_sum = float(stats["pressure_velocity_abs_sum"])
    pressure_pressure_parent_count = len(stats["pressure_pressure_parent_cells"])
    pressure_velocity_parent_count = len(stats["pressure_velocity_parent_cells"])
    return {
        "pressure_pressure_abs_sum": pressure_pressure_abs_sum,
        "pressure_velocity_abs_sum": pressure_velocity_abs_sum,
        "pressure_velocity_to_pressure_pressure_abs_ratio": (
            pressure_velocity_abs_sum / pressure_pressure_abs_sum
            if pressure_pressure_abs_sum > 0.0
            else 0.0
        ),
        "pressure_pressure_abs_sum_per_parent_cell": (
            pressure_pressure_abs_sum / pressure_pressure_parent_count
            if pressure_pressure_parent_count > 0
            else 0.0
        ),
        "pressure_velocity_abs_sum_per_parent_cell": (
            pressure_velocity_abs_sum / pressure_velocity_parent_count
            if pressure_velocity_parent_count > 0
            else 0.0
        ),
    }


def support_instance_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("parent_cell"),
        entry.get("rule_index"),
        entry.get("row_local_index"),
        entry.get("full_cell"),
    )


def summarize_rows(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    by_row_and_instance: dict[int, dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: {"pressure_pressure": [], "pressure_velocity": []})
    )

    for entry in entries:
        row_dof = int_or_none(entry.get("row_dof"))
        if row_dof is None:
            continue
        row = rows.setdefault(row_dof, empty_row_stats())
        row["record_count"] += 1
        block = entry.get("block")
        parent_cell = int_or_none(entry.get("parent_cell"))
        if block == "pressure_pressure":
            row["pressure_pressure_records"] += 1
            row["pressure_pressure_abs_sum"] += float_or_zero(entry.get("row_abs_sum"))
            if parent_cell is not None:
                row["pressure_pressure_parent_cells"].add(parent_cell)
            by_row_and_instance[row_dof][support_instance_key(entry)][
                "pressure_pressure"
            ].append(entry)
        elif block == "pressure_velocity":
            row["pressure_velocity_records"] += 1
            row["pressure_velocity_abs_sum"] += float_or_zero(entry.get("row_abs_sum"))
            if parent_cell is not None:
                row["pressure_velocity_parent_cells"].add(parent_cell)
            by_row_and_instance[row_dof][support_instance_key(entry)][
                "pressure_velocity"
            ].append(entry)
        row["all_pressure_update_sign_unused"] = (
            row["all_pressure_update_sign_unused"]
            and entry.get("pressure_update_sign_used") == 0
        )
        row["all_diagnostic_only"] = (
            row["all_diagnostic_only"] and entry.get("diagnostic_only") == 1
        )

    for row_dof, instances in by_row_and_instance.items():
        row = rows[row_dof]
        for grouped_entries in instances.values():
            pp_entries = grouped_entries["pressure_pressure"]
            if not pp_entries:
                continue
            pv_entries = grouped_entries["pressure_velocity"] or [None]
            for pp_entry in pp_entries:
                for pv_entry in pv_entries:
                    pv_signature = (
                        sample_signature(pv_entry)
                        if pv_entry is not None
                        else ("missing_pressure_velocity_record",)
                    )
                    pv_shape = (
                        sample_shape_class(pv_entry)
                        if pv_entry is not None
                        else "missing_pressure_velocity_record"
                    )
                    row["same_rule_cross_block_signatures"].add(
                        (sample_signature(pp_entry), pv_signature)
                    )
                    row["same_rule_cross_block_shape_pairs"].add(
                        (sample_shape_class(pp_entry), pv_shape)
                    )
        row["magnitude_features"] = row_feature_values(row)
    return rows


def target_value_set(
    rows: dict[int, dict[str, Any]], target_rows: list[int], key: str
) -> set[Any]:
    values: set[Any] = set()
    for row_dof in target_rows:
        stats = rows.get(row_dof)
        if isinstance(stats, dict):
            values.update(stats.get(key, set()))
    return values


def rows_with_intersection(
    rows: dict[int, dict[str, Any]], key: str, values: set[Any]
) -> set[int]:
    if not values:
        return set()
    return {
        row_dof
        for row_dof, stats in rows.items()
        if set(stats.get(key, set())) & values
    }


def target_range(
    rows: dict[int, dict[str, Any]], target_rows: list[int], feature: str
) -> tuple[float, float] | None:
    values = [
        rows[row_dof]["magnitude_features"][feature]
        for row_dof in target_rows
        if row_dof in rows
    ]
    if not values:
        return None
    return min(values), max(values)


def build_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    target_signatures = target_value_set(
        rows, target_rows, "same_rule_cross_block_signatures"
    )
    target_shape_pairs = target_value_set(
        rows, target_rows, "same_rule_cross_block_shape_pairs"
    )
    signature_rows = rows_with_intersection(
        rows, "same_rule_cross_block_signatures", target_signatures
    )
    selectors = [
        evaluate_selector(
            key="same_rule_cross_block_shape_pair_matches_target_union",
            description=(
                "Rows sharing a same-rule pressure-pressure/pressure-velocity "
                "sampled sign-shape pair with an audited target."
            ),
            selected_rows=rows_with_intersection(
                rows, "same_rule_cross_block_shape_pairs", target_shape_pairs
            ),
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
        | {"selector_family": "shape_pair"},
        evaluate_selector(
            key="same_rule_cross_block_exact_local_signature_matches_target_union",
            description=(
                "Rows sharing a same-rule pressure-pressure/pressure-velocity "
                "sampled local signature pair with an audited target."
            ),
            selected_rows=signature_rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
        | {
            "selector_family": "same_rule_signature",
            "target_signature_count": len(target_signatures),
        },
    ]
    for feature in MAGNITUDE_FEATURES:
        value_range = target_range(rows, target_rows, feature)
        if value_range is None:
            continue
        target_min, target_max = value_range
        range_rows = {
            row_dof
            for row_dof, stats in rows.items()
            if target_min
            <= stats["magnitude_features"][feature]
            <= target_max
        }
        selectors.append(
            evaluate_selector(
                key=f"same_rule_cross_block_signature_with_{feature}_range",
                description=(
                    "Rows sharing an audited same-rule PP/PV local signature "
                    f"pair and the audited target {feature} range."
                ),
                selected_rows=signature_rows & range_rows,
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
            )
            | {
                "selector_family": "same_rule_signature_magnitude_range",
                "feature": feature,
                "target_min": target_min,
                "target_max": target_max,
            }
        )
    return selectors


def best_covering_composite(selectors: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        selector
        for selector in selectors
        if selector.get("selector_family") == "same_rule_signature_magnitude_range"
        and selector.get("covers_targets")
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda selector: (
            float(selector.get("selected_to_target_ratio") or float("inf")),
            int(selector.get("selected_count") or 0),
            str(selector.get("key") or ""),
        ),
    )


def target_summary(row_dof: int, stats: dict[str, Any] | None) -> dict[str, Any]:
    if stats is None:
        return {"row_dof": row_dof, "present": False}
    return {
        "row_dof": row_dof,
        "present": True,
        "same_rule_cross_block_signature_count": len(
            stats["same_rule_cross_block_signatures"]
        ),
        "same_rule_cross_block_shape_pair_count": len(
            stats["same_rule_cross_block_shape_pairs"]
        ),
        "magnitude_features": stats["magnitude_features"],
        "all_pressure_update_sign_unused": stats["all_pressure_update_sign_unused"],
        "all_diagnostic_only": stats["all_diagnostic_only"],
    }


def case_finding(
    *,
    target_rows: list[int],
    target_summaries: list[dict[str, Any]],
    best_composite: dict[str, Any] | None,
) -> str:
    if not target_rows:
        return "direct_target_rows_missing"
    if any(not item.get("present") for item in target_summaries):
        return "same_rule_cross_block_signature_missing_target_rows"
    if any(not item.get("all_pressure_update_sign_unused") for item in target_summaries):
        return "same_rule_cross_block_signature_uses_pressure_update_sign"
    if best_composite and best_composite.get("finding") == "selector_selective":
        return "same_rule_cross_block_signature_magnitude_candidate"
    if best_composite:
        return "same_rule_cross_block_signature_magnitude_overbroad_candidate"
    return "same_rule_cross_block_signature_selectors_miss_targets"


def selector_by_key(selectors: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return next(selector for selector in selectors if selector.get("key") == key)


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = summarize_rows(entries)
    selectors = build_selectors(
        rows=rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    target_summaries = [target_summary(row, rows.get(row)) for row in target_rows]
    best_composite = best_covering_composite(selectors)
    base_signature = selector_by_key(
        selectors, "same_rule_cross_block_exact_local_signature_matches_target_union"
    )
    shape_pair = selector_by_key(
        selectors, "same_rule_cross_block_shape_pair_matches_target_union"
    )
    report: dict[str, Any] = {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(
            target_rows=target_rows,
            target_summaries=target_summaries,
            best_composite=best_composite,
        ),
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_count": len(target_rows),
        "target_rows_present_count": sum(
            1 for item in target_summaries if item.get("present")
        ),
        "shape_pair_selector": shape_pair,
        "base_same_rule_signature_selector": base_signature,
        "best_covering_composite_selector": best_composite,
        "target_rows": target_summaries,
        "selectors": selectors,
    }
    if best_composite:
        # Recompute the selected rows for the exported best candidate so the
        # artifact can drive a targeted row-filter replay without re-parsing.
        target_signatures = target_value_set(
            rows, target_rows, "same_rule_cross_block_signatures"
        )
        signature_rows = rows_with_intersection(
            rows, "same_rule_cross_block_signatures", target_signatures
        )
        feature = best_composite.get("feature")
        target_min = best_composite.get("target_min")
        target_max = best_composite.get("target_max")
        selected_rows: set[int] = set()
        if isinstance(feature, str) and isinstance(target_min, (int, float)) and isinstance(
            target_max, (int, float)
        ):
            selected_rows = {
                row_dof
                for row_dof, stats in rows.items()
                if row_dof in signature_rows
                and float(target_min)
                <= stats["magnitude_features"][feature]
                <= float(target_max)
            }
        report["best_covering_composite_selected_global_dofs"] = sorted(
            selected_rows
        )
    return report


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_same_rule_cross_block_signature_missing_evidence",
            "regenerate_sampled_column_replay_logs",
        )
    if any("uses_pressure_update_sign" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_same_rule_cross_block_signature_update_dependent",
            "diagnostic_invalid",
        )
    candidate_cases = [
        case
        for case in cases
        if case.get("finding")
        == "same_rule_cross_block_signature_magnitude_candidate"
    ]
    if cases and len(candidate_cases) == len(cases):
        return (
            "solve_time_direct_pspg_same_rule_cross_block_signature_magnitude_candidate_found",
            "same_rule_cross_block_candidate_requires_replay",
        )
    if candidate_cases:
        return (
            "solve_time_direct_pspg_same_rule_cross_block_signature_partial_candidate",
            "same_rule_cross_block_candidate_partial",
        )
    return (
        "solve_time_direct_pspg_same_rule_cross_block_signature_selectors_not_ready",
        "same_rule_cross_block_signature_overbroad_or_misses_targets",
    )


def build_report(
    *,
    target_map: dict[str, Any],
    log_entries_by_case: dict[str, list[dict[str, Any]]],
    log_paths_by_case: dict[str, Path | None] | None = None,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    targets = target_rows_by_case(target_map)
    log_paths_by_case = log_paths_by_case or {}
    labels = ["test02", "test10"]
    cases = [
        build_case_report(
            label=label,
            log_path=log_paths_by_case.get(label),
            entries=log_entries_by_case.get(label, []),
            target_rows=targets.get(label, []),
            max_target_ratio=max_target_ratio,
        )
        for label in labels
    ]
    finding, status = aggregate_finding(cases)
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Short Test02 step382 and Test10 step90 solve-time sampled-column "
            "same-rule direct PSPG PP/PV local-signature composite audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "features": MAGNITUDE_FEATURES,
        "cases": cases,
        "conclusion": (
            "Same-rule pressure-pressure/pressure-velocity sampled local "
            "signature pairs become selective for both audited windows only "
            "after intersecting them with non-update solve-time magnitude "
            "ranges. This supplies a targeted replay candidate, not a proven "
            "formulation rule: the selector is still derived from audited "
            "target ranges and must be replayed through the solve-time topology "
            "row filter or replaced with a physics-derived equivalent."
        ),
        "next_requirement": (
            "Run a targeted Test02/Test10 row-filter replay for the exported "
            "same-rule cross-block candidate rows, or derive a formulation-side "
            "support/coupling rule that reproduces the same row family without "
            "target-fitted magnitude ranges."
        ),
    }


def main() -> int:
    args = parse_args()
    log_paths = {
        "test02": args.test02_log,
        "test10": args.test10_log,
    }
    report = build_report(
        target_map=load_json(args.target_map_json),
        log_entries_by_case={
            label: read_provenance_log(path) for label, path in log_paths.items()
        },
        log_paths_by_case=log_paths,
        max_target_ratio=args.max_target_ratio,
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
