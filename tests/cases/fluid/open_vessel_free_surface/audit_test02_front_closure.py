#!/usr/bin/env python3
"""Combine SPHERIC Test02 front diagnostics into a closure audit.

The front diagnostics are produced by analyze_spheric_test02_front.py.  This
script does not read VTK results; it compares the already-sampled JSON
diagnostics so the mesh-resolution conclusion can be reproduced cheaply.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_QUALIFICATION_DIR = (
    REPO_ROOT
    / "Documentation"
    / "qualification_logs"
    / "open_vessel_free_surface_remaining_20260526"
)

GATE_TO_H2_DISTANCE_M = 1.0
GATE_TO_OBSTACLE_FRONT_DISTANCE_M = 1.1675
INITIAL_COLUMN_HEIGHT_M = 0.55
GRAVITY_M_PER_S2 = 9.81
COMPARISON_TIME_S = 0.2


@dataclass(frozen=True)
class RunSpec:
    label: str
    mesh_size_m: float
    file_name: str


DEFAULT_RUNS = (
    RunSpec(
        label="coarse_h0p20_pre_fix_0p5_pilot",
        mesh_size_m=0.20,
        file_name="test02_coarse_h0p20_front_speed_diagnostic_20260602.json",
    ),
    RunSpec(
        label="coarse_h0p20_active_source_velocity_fix_0p2_probe",
        mesh_size_m=0.20,
        file_name=(
            "test02_coarse_h0p20_active_source_velocity_fix_0p2_"
            "front_speed_diagnostic_20260602.json"
        ),
    ),
    RunSpec(
        label="coarse_h0p20_no_fluid_velocity_extension_0p2_control",
        mesh_size_m=0.20,
        file_name=(
            "test02_coarse_h0p20_no_fluid_velocity_extension_0p2_"
            "front_speed_diagnostic_20260602.json"
        ),
    ),
    RunSpec(
        label="structured_h0p15_0p2_resolution_probe",
        mesh_size_m=0.15,
        file_name="test02_structured_h0p15_0p2_front_speed_diagnostic_20260602.json",
    ),
)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def speed_to_cover(distance_m: float, time_s: float | None) -> float | None:
    if time_s is None or time_s <= 0.0:
        return None
    return float(distance_m / time_s)


def nearest_sample(samples: list[dict[str, Any]], target_time_s: float) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("front diagnostic has no selected_front_samples")
    return min(
        samples,
        key=lambda sample: abs(float(sample.get("time_s", 0.0)) - target_time_s),
    )


def load_run(
    qualification_dir: Path,
    spec: RunSpec,
    comparison_time_s: float,
) -> dict[str, Any]:
    path = qualification_dir / spec.file_name
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    summary = diagnostic["coarse_pilot_front_summary"]
    literature = diagnostic["literature_and_scale_comparison"]
    sample = nearest_sample(diagnostic.get("selected_front_samples", []), comparison_time_s)
    final_time = float(summary["final_time_s"])
    final_distance = float(summary["final_distance_from_gate_m"])
    sample_time = float(sample["time_s"])
    sample_distance = finite_or_none(sample.get("distance_from_gate_m"))
    final_front_speed = finite_or_none(sample.get("zero_contour_front_speed_m_per_s"))
    leading_advection_speed = None
    leading_vx = finite_or_none(sample.get("leading_edge_mean_vx_m_per_s"))
    if leading_vx is not None:
        leading_advection_speed = -leading_vx

    h2_first_s = literature["reference_events"]["H2_first_ge_0p005m_s"]
    p1_first_s = literature["reference_events"]["P1_first_ge_100Pa_s"]
    remaining_to_h2 = (
        None
        if sample_distance is None
        else max(0.0, GATE_TO_H2_DISTANCE_M - sample_distance)
    )
    remaining_to_obstacle = (
        None
        if sample_distance is None
        else max(0.0, GATE_TO_OBSTACLE_FRONT_DISTANCE_M - sample_distance)
    )
    remaining_time_to_h2 = None if h2_first_s is None else float(h2_first_s) - sample_time
    remaining_time_to_p1 = None if p1_first_s is None else float(p1_first_s) - sample_time

    record: dict[str, Any] = {
        "label": spec.label,
        "mesh_size_m": spec.mesh_size_m,
        "artifact": spec.file_name,
        "case_dir": diagnostic.get("case_dir"),
        "result_count": diagnostic.get("result_count"),
        "final_time_s": final_time,
        "final_distance_from_gate_m": final_distance,
        "average_front_speed_m_per_s": summary["average_zero_contour_front_speed_m_per_s"],
        "speed_fraction_of_reference_H2_average": (
            summary["pilot_speed_fraction_of_reference_H2_arrival"]
        ),
        "speed_fraction_of_reference_P1_average": (
            summary["pilot_speed_fraction_of_reference_P1_threshold"]
        ),
        "speed_fraction_of_ritter_estimate": (
            summary["pilot_speed_fraction_of_ritter_estimate"]
        ),
        "observed_H2_crossing_time_s": summary["observed_H2_crossing_time_s"],
        "estimated_H2_crossing_time_from_run_average_s": (
            summary["estimated_H2_crossing_time_from_linear_final_speed_s"]
        ),
        "sample_near_comparison_time": {
            "requested_time_s": comparison_time_s,
            "sample_time_s": sample_time,
            "distance_from_gate_m": sample_distance,
            "front_speed_m_per_s": final_front_speed,
            "negative_leading_edge_mean_vx_m_per_s": leading_advection_speed,
            "front_speed_minus_leading_advection_m_per_s": (
                None
                if final_front_speed is None or leading_advection_speed is None
                else final_front_speed - leading_advection_speed
            ),
        },
        "required_continuation_from_comparison_sample": {
            "remaining_distance_to_H2_m": remaining_to_h2,
            "remaining_time_to_reference_H2_first_response_s": remaining_time_to_h2,
            "required_average_speed_to_reference_H2_first_response_m_per_s": (
                None
                if remaining_to_h2 is None
                else speed_to_cover(remaining_to_h2, remaining_time_to_h2)
            ),
            "remaining_distance_to_obstacle_front_m": remaining_to_obstacle,
            "remaining_time_to_reference_P1_first_response_s": remaining_time_to_p1,
            "required_average_speed_to_reference_P1_first_response_m_per_s": (
                None
                if remaining_to_obstacle is None
                else speed_to_cover(remaining_to_obstacle, remaining_time_to_p1)
            ),
        },
    }
    if "front_velocity_consistency" in diagnostic:
        consistency = diagnostic["front_velocity_consistency"]
        record["front_velocity_consistency"] = {
            "sample_count": consistency.get("sample_count"),
            "integrated_distance_relative_error": (
                consistency.get("integrated_distance_relative_error")
            ),
            "time_ge_0p10_s": (
                consistency.get("time_window_summaries", {}).get("time_ge_0p10_s")
            ),
        }
    return record


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def distance_at_comparison(run: dict[str, Any]) -> float | None:
    return run["sample_near_comparison_time"]["distance_from_gate_m"]


def first_order_extrapolation(
    *,
    coarse_h: float,
    coarse_distance: float,
    fine_h: float,
    fine_distance: float,
) -> dict[str, Any]:
    slope = (coarse_distance - fine_distance) / (coarse_h - fine_h)
    extrapolated_h0 = fine_distance - slope * fine_h
    return {
        "model": "distance(h) = distance_0 + C h using h=0.20 and h=0.15 samples near 0.2 s",
        "slope_m_per_m": slope,
        "extrapolated_h0_distance_at_0p2_m": extrapolated_h0,
        "predicted_h0p10_distance_at_0p2_m": extrapolated_h0 + slope * 0.10,
    }


def build_audit(qualification_dir: Path, comparison_time_s: float) -> dict[str, Any]:
    runs = [
        load_run(qualification_dir, spec, comparison_time_s)
        for spec in DEFAULT_RUNS
    ]
    by_label = {run["label"]: run for run in runs}

    literature = json.loads(
        (qualification_dir / DEFAULT_RUNS[0].file_name).read_text(encoding="utf-8")
    )["literature_and_scale_comparison"]
    h2_first_s = literature["reference_events"]["H2_first_ge_0p005m_s"]
    p1_first_s = literature["reference_events"]["P1_first_ge_100Pa_s"]
    reference_h2_speed = literature["reference_average_front_speed_to_H2_m_per_s"]
    ritter_speed = literature["dry_bed_ritter_front_speed_2sqrt_gh_m_per_s"]

    pre_fix = by_label["coarse_h0p20_pre_fix_0p5_pilot"]
    post_fix = by_label["coarse_h0p20_active_source_velocity_fix_0p2_probe"]
    no_fluid_extension = by_label[
        "coarse_h0p20_no_fluid_velocity_extension_0p2_control"
    ]
    h0p15 = by_label["structured_h0p15_0p2_resolution_probe"]

    pre_fix_distance = distance_at_comparison(pre_fix)
    post_fix_distance = distance_at_comparison(post_fix)
    no_fluid_distance = distance_at_comparison(no_fluid_extension)
    h0p15_distance = distance_at_comparison(h0p15)

    extrapolation = None
    if post_fix_distance is not None and h0p15_distance is not None:
        extrapolation = first_order_extrapolation(
            coarse_h=0.20,
            coarse_distance=post_fix_distance,
            fine_h=0.15,
            fine_distance=h0p15_distance,
        )
        reference_distance_at_0p2_by_H2_average = reference_h2_speed * comparison_time_s
        extrapolation.update(
            {
                "reference_H2_average_distance_at_0p2_m": (
                    reference_distance_at_0p2_by_H2_average
                ),
                "extrapolated_h0_fraction_of_reference_H2_average_distance_at_0p2": (
                    ratio(
                        extrapolation["extrapolated_h0_distance_at_0p2_m"],
                        reference_distance_at_0p2_by_H2_average,
                    )
                ),
                "caution": (
                    "This is a two-point, short-time diagnostic rather than a "
                    "convergence proof."
                ),
            }
        )

    comparisons = {
        "active_source_fix_vs_pre_fix_h0p20_at_0p2": {
            "distance_delta_m": (
                None
                if post_fix_distance is None or pre_fix_distance is None
                else post_fix_distance - pre_fix_distance
            ),
            "distance_ratio": ratio(post_fix_distance, pre_fix_distance),
        },
        "no_fluid_extension_vs_active_source_fix_h0p20_at_0p2": {
            "distance_delta_m": (
                None
                if no_fluid_distance is None or post_fix_distance is None
                else no_fluid_distance - post_fix_distance
            ),
            "distance_ratio": ratio(no_fluid_distance, post_fix_distance),
        },
        "h0p15_vs_active_source_fix_h0p20_at_0p2": {
            "distance_delta_m": (
                None
                if h0p15_distance is None or post_fix_distance is None
                else h0p15_distance - post_fix_distance
            ),
            "distance_ratio": ratio(h0p15_distance, post_fix_distance),
        },
        "h0p15_vs_pre_fix_h0p20_at_0p2": {
            "distance_delta_m": (
                None
                if h0p15_distance is None or pre_fix_distance is None
                else h0p15_distance - pre_fix_distance
            ),
            "distance_ratio": ratio(h0p15_distance, pre_fix_distance),
        },
    }

    h0p15_required_h2 = h0p15[
        "required_continuation_from_comparison_sample"
    ]["required_average_speed_to_reference_H2_first_response_m_per_s"]
    h0p15_front_speed = h0p15["sample_near_comparison_time"]["front_speed_m_per_s"]
    finding = (
        "Mesh refinement from h=0.20 to h=0.15 materially increases the "
        "short-time front travel, so resolution contributes. It does not close "
        "the SPHERIC Test02 timing: at the h=0.15 sample near 0.2 s the front "
        "must still average more than the dry-bed Ritter speed to reach H2 at "
        "the official first-response time, and the current final front speed "
        "is only about half of that required continuation speed."
    )

    return {
        "qualification_dir": str(qualification_dir),
        "comparison_time_s": comparison_time_s,
        "literature_comparison": {
            "H2_first_ge_0p005m_s": h2_first_s,
            "P1_first_ge_100Pa_s": p1_first_s,
            "distance_gate_to_H2_m": GATE_TO_H2_DISTANCE_M,
            "distance_gate_to_obstacle_front_m": GATE_TO_OBSTACLE_FRONT_DISTANCE_M,
            "reference_average_front_speed_to_H2_m_per_s": reference_h2_speed,
            "reference_average_front_speed_to_P1_threshold_m_per_s": (
                literature["reference_average_front_speed_to_P1_threshold_m_per_s"]
            ),
            "dry_bed_ritter_front_speed_2sqrt_gh_m_per_s": ritter_speed,
            "initial_column_height_m": INITIAL_COLUMN_HEIGHT_M,
            "gravity_m_per_s2": GRAVITY_M_PER_S2,
        },
        "runs": runs,
        "comparisons": comparisons,
        "two_point_mesh_extrapolation": extrapolation,
        "closure_ratios": {
            "h0p15_front_speed_over_required_H2_continuation_speed": ratio(
                h0p15_front_speed,
                h0p15_required_h2,
            ),
            "h0p15_required_H2_continuation_speed_over_ritter_speed": ratio(
                h0p15_required_h2,
                ritter_speed,
            ),
        },
        "finding": finding,
        "status": "open_not_validation_grade_mesh_resolution_contributes_but_does_not_close",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        default=DEFAULT_QUALIFICATION_DIR,
        help="Directory containing the Test02 front diagnostic JSON artifacts.",
    )
    parser.add_argument(
        "--comparison-time",
        type=float,
        default=COMPARISON_TIME_S,
        help="Target time used for like-for-like front-distance comparisons.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the audit JSON. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args.qualification_dir, args.comparison_time)
    text = json.dumps(audit, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
