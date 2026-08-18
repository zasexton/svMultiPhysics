#!/usr/bin/env python3
"""Compare broad-union direct PSPG replay pressure-update branches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_direct_pspg_same_rule_cross_block_parent_cell_replays import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    pressure_summary,
    safe_delta,
    values,
)
from audit_pressure_update_guard import (  # noqa: E402
    event_report,
    point_wet_support,
    result_step,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_rule_cross_block_"
    "broad_union_branch_shift_20260607.json"
)

VARIANT_ORDER = (
    "no_policy",
    "same_rule_parent",
    "broad_minus_parent",
    "broad_policy",
)
ISOLATED_VARIANTS = (
    "no_policy",
    "same_rule_parent",
    "broad_minus_parent",
)

CASES = {
    "test02": {
        "previous_result": (
            "test02_structured_h0p15_all_solid_normal_only_abs_only_"
            "prune1e5_0p54_case/result_382.vtu"
        ),
        "previous_time_s": 0.5399926357269277,
        "current_time_s": 0.54,
        "absolute_threshold_pa": 100000.0,
        "reference_point": 1172,
        "watch_points": [1172, 1170],
        "variants": {
            "no_policy": {
                "result": (
                    "test02_replay_abs_only_prune1e5_step382_"
                    "pspg_wall_full_gradient_scale1_coverage_"
                    "20260606_case/result_001.vtu"
                ),
                "audit": (
                    "test02_replay_abs_only_prune1e5_step382_"
                    "pspg_wall_full_gradient_scale1_coverage_"
                    "pressure_update_audit_20260606.json"
                ),
            },
            "same_rule_parent": {
                "result": (
                    "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
                    "same_rule_cross_block_parent_cells_schur_edge_balance_"
                    "20260607_case/result_001.vtu"
                ),
                "audit": (
                    "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
                    "same_rule_cross_block_parent_cells_schur_edge_balance_"
                    "pressure_update_audit_20260607.json"
                ),
            },
            "broad_minus_parent": {
                "result": (
                    "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
                    "broad_minus_same_rule_parent_cells_schur_edge_balance_"
                    "20260607_case/result_001.vtu"
                ),
                "audit": (
                    "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
                    "broad_minus_same_rule_parent_cells_schur_edge_balance_"
                    "pressure_update_audit_20260607.json"
                ),
            },
            "broad_policy": {
                "result": (
                    "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
                    "topology_policy_schur_edge_balance_step382_consistent_"
                    "20260607_case/result_001.vtu"
                ),
                "audit": (
                    "test02_replay_abs_only_prune1e5_step382_direct_pspg_"
                    "topology_policy_schur_edge_balance_step382_consistent_"
                    "pressure_update_audit_20260607.json"
                ),
            },
        },
    },
    "test10": {
        "previous_result": (
            "test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_"
            "ls_max20_metadata_cap3_1s_case/result_090.vtu"
        ),
        "previous_time_s": 0.9000000000000006,
        "current_time_s": 0.9006250000000006,
        "absolute_threshold_pa": 100.0,
        "reference_point": 83,
        "watch_points": [83],
        "variants": {
            "no_policy": {
                "result": (
                    "test10_replay_cap3_step90_"
                    "pspg_wall_full_gradient_scale1_coverage_"
                    "20260606_case/result_001.vtu"
                ),
                "audit": (
                    "test10_replay_cap3_step90_"
                    "pspg_wall_full_gradient_scale1_coverage_"
                    "pressure_update_audit_20260606.json"
                ),
            },
            "same_rule_parent": {
                "result": (
                    "test10_replay_cap3_step90_direct_pspg_"
                    "same_rule_cross_block_parent_cells_schur_edge_balance_"
                    "20260607_case/result_001.vtu"
                ),
                "audit": (
                    "test10_replay_cap3_step90_direct_pspg_"
                    "same_rule_cross_block_parent_cells_schur_edge_balance_"
                    "pressure_update_audit_20260607.json"
                ),
            },
            "broad_minus_parent": {
                "result": (
                    "test10_replay_cap3_step90_direct_pspg_"
                    "broad_minus_same_rule_parent_cells_schur_edge_balance_"
                    "20260607_case/result_001.vtu"
                ),
                "audit": (
                    "test10_replay_cap3_step90_direct_pspg_"
                    "broad_minus_same_rule_parent_cells_schur_edge_balance_"
                    "pressure_update_audit_20260607.json"
                ),
            },
            "broad_policy": {
                "result": (
                    "test10_replay_cap3_step90_direct_pspg_"
                    "topology_policy_schur_edge_balance_"
                    "20260607_case/result_001.vtu"
                ),
                "audit": (
                    "test10_replay_cap3_step90_direct_pspg_topology_policy_"
                    "schur_edge_balance_pressure_update_audit_20260607.json"
                ),
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def point_event(
    *,
    previous_result: Path,
    current_result: Path,
    point_index: int,
    previous_time_s: float,
    current_time_s: float,
) -> dict[str, Any]:
    import numpy as np
    import pyvista as pv

    previous_grid = pv.read(previous_result)
    current_grid = pv.read(current_result)
    previous_pressure = np.asarray(
        previous_grid.point_data["Pressure"], dtype=float
    ).reshape(-1)
    current_pressure = np.asarray(
        current_grid.point_data["Pressure"], dtype=float
    ).reshape(-1)
    if previous_pressure.size != current_pressure.size:
        raise ValueError(
            f"pressure size mismatch: {previous_result} has "
            f"{previous_pressure.size}, {current_result} has {current_pressure.size}"
        )
    if point_index < 0 or point_index >= current_pressure.size:
        raise ValueError(
            f"point index {point_index} outside pressure array of size "
            f"{current_pressure.size}: {current_result}"
        )

    delta = current_pressure - previous_pressure
    return event_report(
        grid=current_grid,
        point_index=point_index,
        delta=delta,
        previous_pressure=previous_pressure,
        current_pressure=current_pressure,
        previous_step=result_step(previous_result, "result"),
        current_step=result_step(current_result, "result"),
        previous_time_s=previous_time_s,
        current_time_s=current_time_s,
        support=point_wet_support(current_grid),
        active_threshold=0.5,
        tiny_wet_fraction=1.0e-4,
        full_wet_tolerance=1.0e-12,
    )


def variant_worst_event(case: dict[str, Any], variant: str) -> dict[str, Any]:
    summary = values(case.get("variant_pressure_updates")).get(variant)
    worst = values(summary).get("worst_active_or_wet")
    return worst if isinstance(worst, dict) else {}


def point_abs_event(
    case: dict[str, Any], variant: str, point_index: int
) -> dict[str, Any]:
    points = values(case.get("point_events")).get(str(point_index))
    event = values(points).get(variant)
    return event if isinstance(event, dict) else {}


def event_abs_pressure_delta_pa(event: dict[str, Any]) -> float | None:
    value = event.get("abs_pressure_delta_pa")
    return float(value) if isinstance(value, (int, float)) else None


def all_less_than(value: float | None, others: list[float | None]) -> bool:
    return value is not None and all(
        other is not None and value < other for other in others
    )


def classify_case(case: dict[str, Any]) -> dict[str, Any]:
    reference_point = case.get("reference_point")
    threshold = case.get("absolute_threshold_pa")
    broad_reference = point_abs_event(
        case, "broad_policy", int(reference_point)
    )
    isolated_reference_events = {
        variant: point_abs_event(case, variant, int(reference_point))
        for variant in ISOLATED_VARIANTS
    }
    broad_abs = event_abs_pressure_delta_pa(broad_reference)
    isolated_abs = {
        variant: event_abs_pressure_delta_pa(event)
        for variant, event in isolated_reference_events.items()
    }
    broad_worst = variant_worst_event(case, "broad_policy")
    isolated_worst = {
        variant: variant_worst_event(case, variant)
        for variant in ISOLATED_VARIANTS
    }
    broad_worst_point = broad_worst.get("point_index")
    broad_worst_support_class = broad_worst.get("support_class")
    isolated_worst_points = {
        variant: event.get("point_index") for variant, event in isolated_worst.items()
    }
    broad_guard_triggered = values(
        values(case.get("variant_pressure_updates")).get("broad_policy")
    ).get("guard_triggered")
    isolated_guard_triggered = {
        variant: values(values(case.get("variant_pressure_updates")).get(variant)).get(
            "guard_triggered"
        )
        for variant in ISOLATED_VARIANTS
    }
    broad_better_than_isolated = all_less_than(
        broad_abs, list(isolated_abs.values())
    )
    broad_clears_reference = (
        broad_abs is not None
        and isinstance(threshold, (int, float))
        and broad_abs < float(threshold)
    )
    isolated_worst_at_reference = all(
        point == reference_point for point in isolated_worst_points.values()
    )
    broad_shift_to_tiny_cut = (
        broad_worst_point != reference_point
        and broad_worst_support_class == "tiny_cut_supported"
    )
    broad_still_triggers_elsewhere = (
        broad_guard_triggered is True and broad_clears_reference
    )
    return {
        "reference_point": reference_point,
        "broad_reference_abs_pressure_delta_pa": broad_abs,
        "isolated_reference_abs_pressure_delta_pa": isolated_abs,
        "broad_reference_improvement_vs_isolated_pa": {
            variant: safe_delta(update, broad_abs)
            for variant, update in isolated_abs.items()
        },
        "broad_reference_support_class": broad_reference.get("support_class"),
        "broad_policy_worst_point": broad_worst_point,
        "broad_policy_worst_support_class": broad_worst_support_class,
        "isolated_worst_points": isolated_worst_points,
        "isolated_worst_support_classes": {
            variant: event.get("support_class")
            for variant, event in isolated_worst.items()
        },
        "broad_reduces_reference_vs_all_isolated": broad_better_than_isolated,
        "broad_policy_clears_reference_point_guard": broad_clears_reference,
        "isolated_worst_points_match_reference": isolated_worst_at_reference,
        "broad_policy_shifts_worst_from_reference_to_tiny_cut": (
            broad_shift_to_tiny_cut
        ),
        "broad_policy_still_triggers_after_reference_point_clears": (
            broad_still_triggers_elsewhere
        ),
        "all_isolated_variants_guard_triggered": all(
            value is True for value in isolated_guard_triggered.values()
        ),
        "broad_policy_guard_triggered": broad_guard_triggered is True,
        "all_variants_guard_triggered": (
            all(value is True for value in isolated_guard_triggered.values())
            and broad_guard_triggered is True
        ),
    }


def case_finding(case: dict[str, Any]) -> str:
    flags = values(case.get("flags"))
    label = case.get("label")
    if values(case).get("missing_paths"):
        return "broad_union_branch_shift_case_incomplete"
    if (
        label == "test02"
        and flags.get("broad_reduces_reference_vs_all_isolated")
        and flags.get("broad_policy_shifts_worst_from_reference_to_tiny_cut")
        and flags.get("broad_policy_still_triggers_after_reference_point_clears")
    ):
        return "broad_union_reduces_full_wet_reference_then_shifts_to_tiny_cut"
    if (
        label == "test02"
        and flags.get("broad_reduces_reference_vs_all_isolated")
        and flags.get("broad_policy_guard_triggered")
        and flags.get("broad_policy_worst_support_class") == "full_wet_supported"
    ):
        return "broad_union_reduces_full_wet_reference_but_guard_remains"
    if (
        label == "test10"
        and flags.get("broad_reduces_reference_vs_all_isolated")
        and flags.get("broad_policy_guard_triggered")
    ):
        return "broad_union_reduces_shared_full_wet_reference_but_guard_remains"
    if flags.get("broad_reduces_reference_vs_all_isolated"):
        return "broad_union_reduces_reference_point"
    return "inspect_broad_union_branch_shift_case"


def classify_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    cases_by_label = {case.get("label"): case for case in cases}
    missing_paths = [
        path
        for case in cases
        for path in values(case).get("missing_paths", [])
        if isinstance(path, str)
    ]
    test02_flags = values(values(cases_by_label.get("test02")).get("flags"))
    test10_flags = values(values(cases_by_label.get("test10")).get("flags"))
    test02_branch_shift = (
        test02_flags.get("broad_reduces_reference_vs_all_isolated")
        and test02_flags.get("broad_policy_shifts_worst_from_reference_to_tiny_cut")
        and test02_flags.get("broad_policy_still_triggers_after_reference_point_clears")
    )
    test02_full_wet_residual = (
        test02_flags.get("broad_reduces_reference_vs_all_isolated")
        and test02_flags.get("broad_policy_guard_triggered")
        and not test02_flags.get("broad_policy_clears_reference_point_guard")
        and test02_flags.get("broad_policy_worst_support_class")
        == "full_wet_supported"
    )
    test10_residual = (
        test10_flags.get("broad_reduces_reference_vs_all_isolated")
        and test10_flags.get("broad_policy_guard_triggered")
    )
    all_variants_guard_triggered = all(
        values(values(case).get("flags")).get("all_variants_guard_triggered")
        for case in cases
    )

    if missing_paths:
        finding = "direct_pspg_same_rule_cross_block_broad_union_branch_shift_incomplete"
        status = "regenerate_missing_broad_union_branch_shift_inputs"
        conclusion = (
            "At least one pressure audit or replay result required for the "
            "broad-union branch-shift comparison is missing."
        )
    elif test02_branch_shift and test10_residual:
        finding = (
            "direct_pspg_same_rule_cross_block_broad_union_branch_shift_supported"
        )
        status = "broad_union_reduces_full_wet_but_shifts_test02_tiny_cut"
        conclusion = (
            "The broad policy is not just a larger isolated subset: at the "
            "shared full-wet reference points it reduces the pressure update "
            "below every isolated replay. In Test02 it clears that full-wet "
            "reference point but the worst active/wet update shifts to a "
            "tiny-cut-supported point, while Test10 remains a full-wet guard "
            "trigger. The remaining rule must retain broad co-support coupling "
            "benefit while adding active pressure-support or tiny-cut control."
        )
    elif test02_full_wet_residual and test10_residual:
        finding = (
            "direct_pspg_same_rule_cross_block_broad_union_consistent_replays_"
            "do_not_clear_guards"
        )
        status = "broad_union_consistent_replay_insufficient"
        conclusion = (
            "In the transition-consistent comparison, broad policy still "
            "reduces the full-wet reference-point update below every isolated "
            "replay, but it does not clear the Test02 or Test10 full-wet "
            "active/wet guards. The earlier Test02 tiny-cut branch is therefore "
            "not sufficient evidence for a like-for-like broad-union fix."
        )
    else:
        finding = "direct_pspg_same_rule_cross_block_broad_union_branch_shift_mixed"
        status = "inspect_broad_union_branch_shift"
        conclusion = (
            "The broad-union point-aligned replay comparison does not match the "
            "expected Test02 branch-shift plus Test10 residual-guard pattern."
        )

    return {
        "finding": finding,
        "status": status,
        "missing_paths": missing_paths,
        "case_findings": {
            case.get("label"): case.get("finding") for case in cases
        },
        "all_variants_guard_triggered": all_variants_guard_triggered,
        "test02_branch_shift_supported": bool(test02_branch_shift),
        "test02_consistent_full_wet_residual_supported": bool(
            test02_full_wet_residual
        ),
        "test10_broad_union_residual_guard_supported": bool(test10_residual),
        "conclusion": conclusion,
        "next_requirement": (
            "The next formulation candidate should preserve broad-union "
            "support/coupling improvement at the full-wet branch, but must "
            "reduce the residual full-wet Test02/Test10 updates below the "
            "guards before any active-pressure-support or tiny-cut limiter can "
            "be treated as the primary fix."
        ),
    }


def sorted_watch_points(
    *, configured_points: list[int], variant_summaries: dict[str, Any]
) -> list[int]:
    points = {int(point) for point in configured_points}
    for summary in variant_summaries.values():
        worst = values(summary).get("worst_active_or_wet")
        point = values(worst).get("point_index")
        if isinstance(point, int):
            points.add(point)
    return sorted(points)


def build_case(
    *, label: str, spec: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    previous_result = artifact_root / spec["previous_result"]
    variant_summaries = {
        variant: pressure_summary(
            artifact_root / values(spec["variants"]).get(variant, {}).get("audit", "")
        )
        for variant in VARIANT_ORDER
    }
    variant_results = {
        variant: artifact_root / values(spec["variants"]).get(variant, {}).get(
            "result", ""
        )
        for variant in VARIANT_ORDER
    }
    missing_paths = []
    if not previous_result.exists():
        missing_paths.append(str(previous_result))
    missing_paths.extend(
        summary["path"]
        for summary in variant_summaries.values()
        if not summary["exists"]
    )
    missing_paths.extend(
        str(path) for path in variant_results.values() if not path.exists()
    )
    watch_points = sorted_watch_points(
        configured_points=[int(point) for point in spec["watch_points"]],
        variant_summaries=variant_summaries,
    )

    point_events: dict[str, dict[str, Any]] = {}
    if not missing_paths:
        for point_index in watch_points:
            point_events[str(point_index)] = {
                variant: point_event(
                    previous_result=previous_result,
                    current_result=variant_results[variant],
                    point_index=point_index,
                    previous_time_s=float(spec["previous_time_s"]),
                    current_time_s=float(spec["current_time_s"]),
                )
                for variant in VARIANT_ORDER
            }

    case = {
        "label": label,
        "previous_result": str(previous_result),
        "previous_time_s": spec["previous_time_s"],
        "current_time_s": spec["current_time_s"],
        "absolute_threshold_pa": spec["absolute_threshold_pa"],
        "reference_point": spec["reference_point"],
        "watch_points": watch_points,
        "variant_results": {
            variant: str(path) for variant, path in variant_results.items()
        },
        "variant_pressure_updates": variant_summaries,
        "point_events": point_events,
        "missing_paths": missing_paths,
    }
    case["flags"] = classify_case(case) if not missing_paths else {}
    case["finding"] = case_finding(case)
    return case


def build_report(*, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    cases = [
        build_case(label=label, spec=spec, artifact_root=artifact_root)
        for label, spec in CASES.items()
    ]
    report = classify_report(cases)
    return {
        "scope": (
            "Point-aligned Test02 step382 and Test10 step90 pressure-update "
            "comparison across no-policy, same-rule parent-cell, broad-minus "
            "parent-cell, and broad local_schur_edge_balance policy replays."
        ),
        **report,
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    report = build_report(artifact_root=args.artifact_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
