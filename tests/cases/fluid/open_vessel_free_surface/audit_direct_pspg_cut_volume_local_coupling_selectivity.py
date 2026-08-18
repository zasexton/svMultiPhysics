#!/usr/bin/env python3
"""Audit direct PSPG cut-volume pressure/velocity local coupling selectivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def _load_local_matrix_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_local_matrix_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_local_matrix_selectivity",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


LM = _load_local_matrix_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-row pressure-pressure and pressure-velocity local "
            "cut-volume action from the direct PSPG pressure-gradient diagnostic "
            "operator against the audited Test02/Test10 target rows."
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
    parser.add_argument("--pressure-trial-field", default="pressure")
    parser.add_argument("--velocity-trial-field", default="velocity")
    parser.add_argument(
        "--zero-velocity-tolerance",
        type=float,
        default=1.0e-30,
    )
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio still considered selective.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def cross_profiles(
    *,
    pressure_profiles: dict[int, dict[str, Any]],
    velocity_profiles: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    rows = set(pressure_profiles).union(velocity_profiles)
    profiles: dict[int, dict[str, Any]] = {}
    for row in rows:
        pressure = pressure_profiles.get(row, {})
        velocity = velocity_profiles.get(row, {})
        pressure_abs = safe_float(pressure.get("total_row_abs_sum"))
        velocity_abs = safe_float(velocity.get("total_row_abs_sum"))
        pressure_parents = set(pressure.get("parent_cells", []))
        velocity_parents = set(velocity.get("parent_cells", []))
        parent_union = pressure_parents.union(velocity_parents)
        parent_intersection = pressure_parents.intersection(velocity_parents)
        profiles[row] = {
            "global_dof": row,
            "pressure_total_row_abs_sum": pressure_abs,
            "velocity_total_row_abs_sum": velocity_abs,
            "velocity_to_pressure_abs_ratio": (
                velocity_abs / pressure_abs if pressure_abs > 0.0 else 0.0
            ),
            "pressure_rule_count": int(pressure.get("rule_count", 0)),
            "velocity_rule_count": int(velocity.get("rule_count", 0)),
            "pressure_parent_cell_count": int(pressure.get("parent_cell_count", 0)),
            "velocity_parent_cell_count": int(velocity.get("parent_cell_count", 0)),
            "parent_cell_overlap_count": len(parent_intersection),
            "parent_cell_union_count": len(parent_union),
            "parent_cell_overlap_fraction": (
                len(parent_intersection) / len(parent_union)
                if parent_union
                else 0.0
            ),
            "pressure_support_class": pressure.get(
                "cut_volume_support_class",
                "missing_pressure_support",
            ),
            "velocity_support_class": velocity.get(
                "cut_volume_support_class",
                "missing_velocity_support",
            ),
            "pressure_full_cell_abs_fraction": safe_float(
                pressure.get("full_cell_abs_fraction")
            ),
            "velocity_full_cell_abs_fraction": safe_float(
                velocity.get("full_cell_abs_fraction")
            ),
            "pressure_max_rule_row_abs_fraction": safe_float(
                pressure.get("max_rule_row_abs_fraction")
            ),
            "velocity_max_rule_row_abs_fraction": safe_float(
                velocity.get("max_rule_row_abs_fraction")
            ),
        }
    return profiles


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

    velocity_abs = values("velocity_total_row_abs_sum")
    pressure_abs = values("pressure_total_row_abs_sum")
    ratio = values("velocity_to_pressure_abs_ratio")
    overlap = values("parent_cell_overlap_fraction")
    return {
        "velocity_total_row_abs_sum_p10": LM.percentile(velocity_abs, 0.10),
        "velocity_total_row_abs_sum_p25": LM.percentile(velocity_abs, 0.25),
        "velocity_total_row_abs_sum_p75": LM.percentile(velocity_abs, 0.75),
        "velocity_total_row_abs_sum_p90": LM.percentile(velocity_abs, 0.90),
        "pressure_total_row_abs_sum_p75": LM.percentile(pressure_abs, 0.75),
        "velocity_to_pressure_abs_ratio_p10": LM.percentile(ratio, 0.10),
        "velocity_to_pressure_abs_ratio_p25": LM.percentile(ratio, 0.25),
        "velocity_to_pressure_abs_ratio_p75": LM.percentile(ratio, 0.75),
        "velocity_to_pressure_abs_ratio_p90": LM.percentile(ratio, 0.90),
        "parent_cell_overlap_fraction_p25": LM.percentile(overlap, 0.25),
    }


def threshold_le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def threshold_ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def selector_definitions(
    *,
    thresholds: dict[str, float | None],
    zero_velocity_tolerance: float,
) -> list[dict[str, Any]]:
    return [
        {
            "key": "cross_field_profiled_candidate",
            "description": "Preferred candidates with both pressure and velocity local cut-volume coupling profiles.",
            "threshold_key": None,
            "predicate": (
                lambda profile: profile["pressure_rule_count"] > 0
                and profile["velocity_rule_count"] > 0
            ),
        },
        {
            "key": "cross_field_zero_velocity_action",
            "description": "Profiled candidates with no pressure-velocity local row action.",
            "threshold_key": f"fixed:{zero_velocity_tolerance}",
            "predicate": (
                lambda profile: profile["velocity_total_row_abs_sum"]
                <= zero_velocity_tolerance
            ),
        },
        {
            "key": "cross_field_low_velocity_action_p25",
            "description": "Profiled candidates in the bottom quartile of pressure-velocity local row action.",
            "threshold_key": "velocity_total_row_abs_sum_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "velocity_total_row_abs_sum",
                thresholds["velocity_total_row_abs_sum_p25"],
            ),
        },
        {
            "key": "cross_field_high_velocity_action_p90",
            "description": "Profiled candidates in the top 10% of pressure-velocity local row action.",
            "threshold_key": "velocity_total_row_abs_sum_p90",
            "predicate": lambda profile: threshold_ge(
                profile,
                "velocity_total_row_abs_sum",
                thresholds["velocity_total_row_abs_sum_p90"],
            ),
        },
        {
            "key": "cross_field_low_velocity_pressure_ratio_p25",
            "description": "Profiled candidates in the bottom quartile of velocity-to-pressure local action ratio.",
            "threshold_key": "velocity_to_pressure_abs_ratio_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "velocity_to_pressure_abs_ratio",
                thresholds["velocity_to_pressure_abs_ratio_p25"],
            ),
        },
        {
            "key": "cross_field_high_velocity_pressure_ratio_p75",
            "description": "Profiled candidates in the top quartile of velocity-to-pressure local action ratio.",
            "threshold_key": "velocity_to_pressure_abs_ratio_p75",
            "predicate": lambda profile: threshold_ge(
                profile,
                "velocity_to_pressure_abs_ratio",
                thresholds["velocity_to_pressure_abs_ratio_p75"],
            ),
        },
        {
            "key": "cross_field_high_velocity_pressure_ratio_p90",
            "description": "Profiled candidates in the top 10% of velocity-to-pressure local action ratio.",
            "threshold_key": "velocity_to_pressure_abs_ratio_p90",
            "predicate": lambda profile: threshold_ge(
                profile,
                "velocity_to_pressure_abs_ratio",
                thresholds["velocity_to_pressure_abs_ratio_p90"],
            ),
        },
        {
            "key": "cross_field_high_pressure_low_velocity_ratio",
            "description": "High pressure-pressure local action with bottom-quartile pressure-velocity ratio.",
            "threshold_key": "pressure_total_row_abs_sum_p75|velocity_to_pressure_abs_ratio_p25",
            "predicate": lambda profile: (
                threshold_ge(
                    profile,
                    "pressure_total_row_abs_sum",
                    thresholds["pressure_total_row_abs_sum_p75"],
                )
                and threshold_le(
                    profile,
                    "velocity_to_pressure_abs_ratio",
                    thresholds["velocity_to_pressure_abs_ratio_p25"],
                )
            ),
        },
        {
            "key": "cross_field_ratio_tail_outlier",
            "description": "Rows in either low or high tail of velocity-to-pressure local action ratio.",
            "threshold_key": "velocity_to_pressure_abs_ratio_p10|velocity_to_pressure_abs_ratio_p90",
            "predicate": lambda profile: (
                threshold_le(
                    profile,
                    "velocity_to_pressure_abs_ratio",
                    thresholds["velocity_to_pressure_abs_ratio_p10"],
                )
                or threshold_ge(
                    profile,
                    "velocity_to_pressure_abs_ratio",
                    thresholds["velocity_to_pressure_abs_ratio_p90"],
                )
            ),
        },
        {
            "key": "cross_field_low_parent_overlap_p25",
            "description": "Rows in the bottom quartile of pressure/velocity parent-cell overlap.",
            "threshold_key": "parent_cell_overlap_fraction_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "parent_cell_overlap_fraction",
                thresholds["parent_cell_overlap_fraction_p25"],
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
                key: thresholds.get(key) for key in threshold_key.split("|")
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
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    target_profiles = {str(row): profiles[row] for row in target_rows if row in profiles}
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(target_set.intersection(profiles)),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
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
    pressure_trial_field: str = "pressure",
    velocity_trial_field: str = "velocity",
    zero_velocity_tolerance: float = 1.0e-30,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    emission_cases = LM.case_map(global_emission)
    target_cases = LM.target_case_map(target_map)
    log_paths = LM.default_log_paths(emission_cases, explicit_logs or [])

    cases: dict[str, dict[str, Any]] = {}
    selector_defs_by_case: dict[str, list[dict[str, Any]]] = {}
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        pressure_entries, pressure_evidence = LM.latest_local_matrix_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=pressure_trial_field,
        )
        velocity_entries, velocity_evidence = LM.latest_local_matrix_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=velocity_trial_field,
        )
        pressure_profiles = LM.row_profiles_from_entries(pressure_entries)
        velocity_profiles = LM.row_profiles_from_entries(velocity_entries)
        profiles = cross_profiles(
            pressure_profiles=pressure_profiles,
            velocity_profiles=velocity_profiles,
        )
        thresholds = case_thresholds(profiles, candidate_rows)
        selector_defs = selector_definitions(
            thresholds=thresholds,
            zero_velocity_tolerance=zero_velocity_tolerance,
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
            "log_evidence": {
                "path": str(log_path),
                "exists": log_path.exists(),
                "pressure": pressure_evidence,
                "velocity": velocity_evidence,
                "status": (
                    "ok"
                    if pressure_evidence.get("status") == "ok"
                    and velocity_evidence.get("status") == "ok"
                    else "cross_field_entries_missing"
                ),
            },
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
                "finding": LM.aggregate_selector_finding(case_results),
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
        finding = "direct_pspg_cut_volume_local_coupling_evidence_missing"
        next_requirement = (
            "Regenerate Test02/Test10 short replay logs with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_PROVENANCE_DIAGNOSTIC=1 and "
            f"SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_PROVENANCE_OPERATOR={operator}."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_local_coupling_selector_selective"
        next_requirement = (
            "Prototype the selective pressure/velocity local coupling gate and "
            "run the same short Test02/Test10 replay windows."
        )
    elif overbroad or misses:
        finding = (
            "direct_pspg_cut_volume_local_coupling_selectors_overbroad_or_miss_targets"
        )
        next_requirement = (
            "Do not promote pressure/velocity local coupling magnitude or ratio "
            "alone; the remaining discriminator must include stronger spatial "
            "support topology or physics beyond per-row cross-field action."
        )
    else:
        finding = "direct_pspg_cut_volume_local_coupling_selectivity_inconclusive"
        next_requirement = (
            "Regenerate cross-field local matrix provenance before selecting a "
            "formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for pressure-pressure versus pressure-velocity "
            "local cut-volume row action in the direct PSPG pressure-gradient "
            "diagnostic operator."
        ),
        "global_emission_path": str(global_emission_path) if global_emission_path else None,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "candidate_key": candidate_key,
        "operator": operator,
        "test_field": test_field,
        "pressure_trial_field": pressure_trial_field,
        "velocity_trial_field": velocity_trial_field,
        "zero_velocity_tolerance": zero_velocity_tolerance,
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
        global_emission=LM.load_json(args.global_emission_json),
        target_map=LM.load_json(args.target_map_json),
        global_emission_path=args.global_emission_json,
        target_map_path=args.target_map_json,
        explicit_logs=args.log,
        candidate_key=args.candidate_key,
        operator=args.operator,
        test_field=args.test_field,
        pressure_trial_field=args.pressure_trial_field,
        velocity_trial_field=args.velocity_trial_field,
        zero_velocity_tolerance=args.zero_velocity_tolerance,
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
