#!/usr/bin/env python3
"""Audit composite direct PSPG cut-volume support-feature selectivity."""

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


def _load_sibling(name: str):
    script = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


LM = _load_sibling("audit_direct_pspg_cut_volume_local_matrix_selectivity.py")
PG = _load_sibling("audit_direct_pspg_cut_volume_parent_graph_selectivity.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine local pressure row action, pressure/velocity coupling, "
            "and row-parent graph topology from direct PSPG cut-volume "
            "provenance, then compare fixed branch-aware selectors against "
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
    parser.add_argument("--pressure-trial-field", default="pressure")
    parser.add_argument("--velocity-trial-field", default="velocity")
    parser.add_argument("--zero-velocity-tolerance", type=float, default=1.0e-30)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def combined_profiles(
    *,
    pressure_profiles: dict[int, dict[str, Any]],
    velocity_profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
) -> dict[int, dict[str, Any]]:
    graph_profiles = PG.candidate_parent_graph_profiles(
        pressure_profiles,
        candidate_rows,
    )
    profiles: dict[int, dict[str, Any]] = {}
    for row in candidate_rows:
        if row not in pressure_profiles or row not in graph_profiles:
            continue
        pressure = pressure_profiles[row]
        velocity = velocity_profiles.get(row, {})
        graph = graph_profiles[row]
        pressure_abs = safe_float(pressure.get("total_row_abs_sum"))
        velocity_abs = safe_float(velocity.get("total_row_abs_sum"))
        profiles[row] = {
            "global_dof": row,
            "pressure_total_row_abs_sum": pressure_abs,
            "velocity_total_row_abs_sum": velocity_abs,
            "velocity_to_pressure_abs_ratio": (
                velocity_abs / pressure_abs if pressure_abs > 0.0 else 0.0
            ),
            "max_rule_row_abs_fraction": safe_float(
                pressure.get("max_rule_row_abs_fraction")
            ),
            "row_parent_graph_degree": graph["row_parent_graph_degree"],
            "row_parent_graph_weighted_degree": graph[
                "row_parent_graph_weighted_degree"
            ],
            "row_parent_graph_clustering": graph["row_parent_graph_clustering"],
            "row_parent_graph_two_hop_count": graph[
                "row_parent_graph_two_hop_count"
            ],
            "parent_cell_count": graph["parent_cell_count"],
        }
    return profiles


def percentile(values: list[float], fraction: float) -> float | None:
    return LM.percentile(values, fraction)


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

    return {
        "degree_p25": percentile(values("row_parent_graph_degree"), 0.25),
        "degree_p75": percentile(values("row_parent_graph_degree"), 0.75),
        "clustering_p25": percentile(values("row_parent_graph_clustering"), 0.25),
        "clustering_p75": percentile(values("row_parent_graph_clustering"), 0.75),
        "two_hop_p25": percentile(values("row_parent_graph_two_hop_count"), 0.25),
        "two_hop_p75": percentile(values("row_parent_graph_two_hop_count"), 0.75),
        "pressure_abs_p75": percentile(values("pressure_total_row_abs_sum"), 0.75),
        "velocity_abs_p25": percentile(values("velocity_total_row_abs_sum"), 0.25),
        "velocity_ratio_p10": percentile(values("velocity_to_pressure_abs_ratio"), 0.10),
        "velocity_ratio_p25": percentile(values("velocity_to_pressure_abs_ratio"), 0.25),
        "velocity_ratio_p75": percentile(values("velocity_to_pressure_abs_ratio"), 0.75),
        "velocity_ratio_p90": percentile(values("velocity_to_pressure_abs_ratio"), 0.90),
    }


def le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def low_degree_high_clustering(profile: dict[str, Any], t: dict[str, float | None]) -> bool:
    return le(profile, "row_parent_graph_degree", t["degree_p25"]) and ge(
        profile,
        "row_parent_graph_clustering",
        t["clustering_p75"],
    )


def high_degree_low_clustering(profile: dict[str, Any], t: dict[str, float | None]) -> bool:
    return ge(profile, "row_parent_graph_degree", t["degree_p75"]) and le(
        profile,
        "row_parent_graph_clustering",
        t["clustering_p25"],
    )


def graph_bimodal_tail(profile: dict[str, Any], t: dict[str, float | None]) -> bool:
    return low_degree_high_clustering(profile, t) or high_degree_low_clustering(profile, t)


def selector_definitions(
    *,
    thresholds: dict[str, float | None],
    zero_velocity_tolerance: float,
) -> list[dict[str, Any]]:
    return [
        {
            "key": "composite_profiled_candidate",
            "description": "Preferred candidates with pressure, velocity, and row-parent graph features.",
            "threshold_key": None,
            "predicate": lambda profile: True,
        },
        {
            "key": "composite_graph_bimodal_tail",
            "description": "Union of low-degree/high-clustering and high-degree/low-clustering row-parent graph tails.",
            "threshold_key": "degree_p25|degree_p75|clustering_p25|clustering_p75",
            "predicate": lambda profile: graph_bimodal_tail(profile, thresholds),
        },
        {
            "key": "composite_graph_tail_and_ratio_tail",
            "description": "Graph bimodal tail rows that are also in a low/high velocity-to-pressure ratio tail.",
            "threshold_key": "degree_p25|degree_p75|clustering_p25|clustering_p75|velocity_ratio_p10|velocity_ratio_p90",
            "predicate": lambda profile: graph_bimodal_tail(profile, thresholds)
            and (
                le(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p10"])
                or ge(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p90"])
            ),
        },
        {
            "key": "composite_isolated_or_high_ratio_coherent",
            "description": "Low-degree/high-clustering rows plus high-degree/low-clustering rows with high velocity ratio.",
            "threshold_key": "degree_p25|degree_p75|clustering_p25|clustering_p75|velocity_ratio_p90",
            "predicate": lambda profile: low_degree_high_clustering(profile, thresholds)
            or (
                high_degree_low_clustering(profile, thresholds)
                and ge(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p90"])
            ),
        },
        {
            "key": "composite_low_graph_high_ratio_or_zero_velocity",
            "description": "Low-degree/high-clustering rows with either high velocity ratio or zero velocity action.",
            "threshold_key": "degree_p25|clustering_p75|velocity_ratio_p75|fixed:zero_velocity",
            "predicate": lambda profile: low_degree_high_clustering(profile, thresholds)
            and (
                ge(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p75"])
                or profile["velocity_total_row_abs_sum"] <= zero_velocity_tolerance
            ),
        },
        {
            "key": "composite_high_pressure_graph_tail",
            "description": "Graph bimodal tail rows with top-quartile pressure-pressure row action.",
            "threshold_key": "degree_p25|degree_p75|clustering_p25|clustering_p75|pressure_abs_p75",
            "predicate": lambda profile: graph_bimodal_tail(profile, thresholds)
            and ge(profile, "pressure_total_row_abs_sum", thresholds["pressure_abs_p75"]),
        },
        {
            "key": "composite_low_velocity_or_high_ratio_graph_tail",
            "description": "Graph bimodal tail rows with low velocity action or high velocity ratio.",
            "threshold_key": "degree_p25|degree_p75|clustering_p25|clustering_p75|velocity_abs_p25|velocity_ratio_p90",
            "predicate": lambda profile: graph_bimodal_tail(profile, thresholds)
            and (
                le(profile, "velocity_total_row_abs_sum", thresholds["velocity_abs_p25"])
                or ge(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p90"])
            ),
        },
        {
            "key": "composite_twohop_graph_ratio_tail",
            "description": "Two-hop graph reach tail rows that are also velocity-ratio tail rows.",
            "threshold_key": "two_hop_p25|two_hop_p75|velocity_ratio_p10|velocity_ratio_p90",
            "predicate": lambda profile: (
                le(profile, "row_parent_graph_two_hop_count", thresholds["two_hop_p25"])
                or ge(profile, "row_parent_graph_two_hop_count", thresholds["two_hop_p75"])
            )
            and (
                le(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p10"])
                or ge(profile, "velocity_to_pressure_abs_ratio", thresholds["velocity_ratio_p90"])
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
    selected = [row for row in candidate_rows if row in profiles and predicate(profiles[row])]
    selected_set = set(selected)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    threshold_key = selector.get("threshold_key")
    threshold_value: Any = None
    if isinstance(threshold_key, str):
        threshold_value = {
            key: thresholds.get(key)
            for key in threshold_key.split("|")
            if not key.startswith("fixed:")
        }
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
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(target_set.intersection(profiles)),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [row for row in target_rows if row not in profiles],
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
        profiles = combined_profiles(
            pressure_profiles=pressure_profiles,
            velocity_profiles=velocity_profiles,
            candidate_rows=candidate_rows,
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
                    else "composite_entries_missing"
                ),
            },
            "thresholds": thresholds,
            "profile_summary": profile_summary(profiles, candidate_rows, target_rows),
            "selectors": selector_cases,
        }

    selectors = []
    first_label = next(iter(target_cases), None)
    selector_count = len(selector_defs_by_case[first_label]) if first_label else 0
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

    selective = [selector for selector in selectors if selector["finding"] == "selector_selective"]
    overbroad = [selector for selector in selectors if "overbroad" in str(selector["finding"])]
    misses = [selector for selector in selectors if "miss" in str(selector["finding"])]
    missing_cases = [
        label for label, case in cases.items() if case["log_evidence"].get("status") != "ok"
    ]

    if missing_cases:
        finding = "direct_pspg_cut_volume_composite_evidence_missing"
        next_requirement = "Regenerate local matrix provenance before evaluating composite support features."
    elif selective:
        finding = "direct_pspg_cut_volume_composite_selector_selective"
        next_requirement = "Prototype the selective composite support-feature gate and run short Test02/Test10 replays."
    elif overbroad or misses:
        finding = "direct_pspg_cut_volume_composite_selectors_overbroad_or_miss_targets"
        next_requirement = (
            "Do not promote bounded graph/action/coupling composite selectors "
            "alone; the remaining discriminator needs a physics-derived "
            "formulation rule, not feature thresholding."
        )
    else:
        finding = "direct_pspg_cut_volume_composite_selectivity_inconclusive"
        next_requirement = "Regenerate composite-feature evidence before selecting a formulation replay."

    return {
        "scope": (
            "Selectivity audit for fixed composites of direct PSPG cut-volume "
            "local row action, pressure/velocity coupling, and row-parent graph topology."
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
