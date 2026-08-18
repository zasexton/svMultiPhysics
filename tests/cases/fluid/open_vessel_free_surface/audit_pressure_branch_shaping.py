#!/usr/bin/env python3
"""Compare pressure-update guard reports across pressure-policy controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CATEGORIES = (
    "active_or_wet_supported",
    "full_wet_supported",
    "cut_supported",
    "tiny_cut_supported",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare audit_pressure_update_guard JSON files and classify whether "
            "pressure-policy controls only move the accepted pressure-update "
            "branch or actually clear the active/wet guard."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline guard JSON as LABEL=PATH.",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help="Control guard JSON as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--improvement-ratio",
        type=float,
        default=0.95,
        help="Control/baseline ratio below this value counts as improved.",
    )
    parser.add_argument(
        "--worsening-ratio",
        type=float,
        default=1.05,
        help="Control/baseline ratio above this value counts as worsened.",
    )
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def category_event(report: dict[str, Any], category: str) -> dict[str, Any] | None:
    event = report.get("worst_by_category", {}).get(category)
    return event if isinstance(event, dict) else None


def event_abs_delta(event: dict[str, Any] | None) -> float | None:
    if event is None:
        return None
    value = event.get("abs_pressure_delta_pa")
    return float(value) if isinstance(value, (int, float)) else None


def event_summary(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if event is None:
        return None
    out: dict[str, Any] = {}
    for key in (
        "abs_pressure_delta_pa",
        "pressure_delta_pa",
        "point_index",
        "point_m",
        "support_class",
        "from_step",
        "to_step",
        "from_time_s",
        "to_time_s",
        "incident_wet_fraction_max",
        "incident_wet_fraction_min_positive",
    ):
        if key in event:
            out[key] = event[key]
    return out


def guard_threshold(report: dict[str, Any]) -> float | None:
    value = report.get("absolute_threshold_pa")
    return float(value) if isinstance(value, (int, float)) else None


def report_still_triggers(report: dict[str, Any]) -> bool:
    triggered = report.get("triggered_transition_count")
    if isinstance(triggered, int):
        return triggered > 0
    return report.get("status") == "diagnostic_pressure_update_guard_triggered"


def guard_summary(label: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    categories = {
        category: event_summary(category_event(report, category))
        for category in CATEGORIES
    }
    active = category_event(report, "active_or_wet_supported")
    return {
        "label": label,
        "path": str(path),
        "status": report.get("status"),
        "finding": report.get("finding"),
        "absolute_threshold_pa": guard_threshold(report),
        "triggered_transition_count": report.get("triggered_transition_count"),
        "still_triggers": report_still_triggers(report),
        "worst_active_abs_pressure_delta_pa": event_abs_delta(active),
        "worst_active_support_class": (
            active.get("support_class") if isinstance(active, dict) else None
        ),
        "worst_active_point_index": (
            active.get("point_index") if isinstance(active, dict) else None
        ),
        "worst_by_category": categories,
    }


def classify_control(
    *,
    control_still_triggers: bool,
    ratio_to_baseline: float | None,
    improvement_ratio: float,
    worsening_ratio: float,
) -> str:
    if not control_still_triggers:
        return "guard_cleared"
    if ratio_to_baseline is None:
        return "still_triggers_without_ratio"
    if ratio_to_baseline < improvement_ratio:
        return "improves_but_still_triggers"
    if ratio_to_baseline > worsening_ratio:
        return "worsens_and_still_triggers"
    return "neutral_and_still_triggers"


def category_comparison(
    baseline: dict[str, Any],
    control: dict[str, Any],
    category: str,
) -> dict[str, Any]:
    baseline_event = category_event(baseline, category)
    control_event = category_event(control, category)
    baseline_abs = event_abs_delta(baseline_event)
    control_abs = event_abs_delta(control_event)
    ratio = (
        control_abs / baseline_abs
        if control_abs is not None and baseline_abs not in (None, 0.0)
        else None
    )
    return {
        "baseline_abs_pressure_delta_pa": baseline_abs,
        "control_abs_pressure_delta_pa": control_abs,
        "delta_abs_pressure_delta_pa": (
            control_abs - baseline_abs
            if control_abs is not None and baseline_abs is not None
            else None
        ),
        "ratio_to_baseline": ratio,
        "baseline_support_class": (
            baseline_event.get("support_class")
            if isinstance(baseline_event, dict)
            else None
        ),
        "control_support_class": (
            control_event.get("support_class")
            if isinstance(control_event, dict)
            else None
        ),
        "baseline_point_index": (
            baseline_event.get("point_index")
            if isinstance(baseline_event, dict)
            else None
        ),
        "control_point_index": (
            control_event.get("point_index")
            if isinstance(control_event, dict)
            else None
        ),
    }


def compare_control_to_baseline(
    *,
    baseline: dict[str, Any],
    control: dict[str, Any],
    label: str,
    path: Path,
    improvement_ratio: float,
    worsening_ratio: float,
) -> dict[str, Any]:
    active = category_comparison(baseline, control, "active_or_wet_supported")
    control_still_triggers = report_still_triggers(control)
    classification = classify_control(
        control_still_triggers=control_still_triggers,
        ratio_to_baseline=active["ratio_to_baseline"],
        improvement_ratio=improvement_ratio,
        worsening_ratio=worsening_ratio,
    )
    support_shift = (
        active["baseline_support_class"] != active["control_support_class"]
    )
    point_shift = active["baseline_point_index"] != active["control_point_index"]
    threshold = guard_threshold(control)
    control_abs = active["control_abs_pressure_delta_pa"]
    return {
        "label": label,
        "path": str(path),
        "status": control.get("status"),
        "finding": control.get("finding"),
        "absolute_threshold_pa": threshold,
        "triggered_transition_count": control.get("triggered_transition_count"),
        "still_triggers": control_still_triggers,
        "classification": classification,
        "safe_fix_candidate": (
            control_abs is not None
            and threshold is not None
            and control_abs <= threshold
            and not control_still_triggers
        ),
        "active_or_wet_supported": active,
        "support_class_shift": support_shift,
        "point_shift": point_shift,
        "branch_shaping_evidence": (
            support_shift
            or point_shift
            or classification
            in {
                "guard_cleared",
                "improves_but_still_triggers",
                "worsens_and_still_triggers",
            }
        ),
        "category_comparisons": {
            category: category_comparison(baseline, control, category)
            for category in CATEGORIES
        },
    }


def summarize_branch_controls(
    *,
    baseline_label: str,
    baseline_path: Path,
    baseline_report: dict[str, Any],
    controls: list[tuple[str, Path, dict[str, Any]]],
    improvement_ratio: float = 0.95,
    worsening_ratio: float = 1.05,
) -> dict[str, Any]:
    baseline = guard_summary(baseline_label, baseline_path, baseline_report)
    comparisons = [
        compare_control_to_baseline(
            baseline=baseline_report,
            control=control_report,
            label=label,
            path=path,
            improvement_ratio=improvement_ratio,
            worsening_ratio=worsening_ratio,
        )
        for label, path, control_report in controls
    ]
    classification_counts: dict[str, int] = {}
    for comparison in comparisons:
        classification = str(comparison["classification"])
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
    guard_cleared_count = classification_counts.get("guard_cleared", 0)
    branch_shaping_count = sum(
        1 for comparison in comparisons if comparison["branch_shaping_evidence"]
    )
    if guard_cleared_count:
        finding = "pressure_policy_control_clears_guard"
    elif branch_shaping_count:
        finding = "pressure_policy_shapes_branch_but_does_not_clear_guard"
    else:
        finding = "pressure_policy_controls_do_not_shape_guard_branch"
    return {
        "baseline": baseline,
        "controls": comparisons,
        "control_count": len(comparisons),
        "classification_counts": classification_counts,
        "guard_cleared_count": guard_cleared_count,
        "still_triggered_count": sum(
            1 for comparison in comparisons if comparison["still_triggers"]
        ),
        "branch_shaping_evidence_count": branch_shaping_count,
        "support_class_shift_count": sum(
            1 for comparison in comparisons if comparison["support_class_shift"]
        ),
        "point_shift_count": sum(
            1 for comparison in comparisons if comparison["point_shift"]
        ),
        "finding": finding,
    }


def main() -> int:
    args = parse_args()
    baseline_label, baseline_path = parse_labeled_path(args.baseline)
    controls = [
        (label, path, load_json(path))
        for label, path in (parse_labeled_path(value) for value in args.control)
    ]
    report = summarize_branch_controls(
        baseline_label=baseline_label,
        baseline_path=baseline_path,
        baseline_report=load_json(baseline_path),
        controls=controls,
        improvement_ratio=args.improvement_ratio,
        worsening_ratio=args.worsening_ratio,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
