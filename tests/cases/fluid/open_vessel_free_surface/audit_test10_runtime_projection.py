#!/usr/bin/env python3
"""Project direct-solver runtime for the SPHERIC Test10 pressure-history run."""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_SMOKE_SUMMARY = "test10_roll_full_source_40step_summary_20260602.json"
DEFAULT_PRESSURE_COMPARISON = (
    "test10_roll_full_source_40step_pressure_comparison_20260602.json"
)


def hours(seconds: float) -> float:
    return seconds / 3600.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_projection(
    *,
    target_name: str,
    target_time_s: float,
    dt_s: float,
    completed_steps: int,
    seconds_per_step: float,
) -> dict[str, Any]:
    required_steps = int(math.ceil(target_time_s / dt_s))
    additional_steps = max(0, required_steps - completed_steps)
    projected_total_seconds = required_steps * seconds_per_step
    projected_additional_seconds = additional_steps * seconds_per_step
    return {
        "target": target_name,
        "target_time_s": target_time_s,
        "required_steps_at_current_dt": required_steps,
        "additional_steps_after_smoke": additional_steps,
        "projected_total_wall_seconds_at_smoke_rate": projected_total_seconds,
        "projected_total_wall_hours_at_smoke_rate": hours(projected_total_seconds),
        "projected_additional_wall_seconds_after_smoke": projected_additional_seconds,
        "projected_additional_wall_hours_after_smoke": hours(
            projected_additional_seconds
        ),
    }


def build_projection(
    qualification_dir: Path,
    smoke_summary_name: str,
    pressure_comparison_name: str,
) -> dict[str, Any]:
    summary_path = qualification_dir / smoke_summary_name
    comparison_path = qualification_dir / pressure_comparison_name
    summary = load_json(summary_path)
    comparison = load_json(comparison_path)

    completed_steps = int(summary["loop"]["steps_taken"])
    final_time_s = float(summary["loop"]["final_time"])
    dt_s = final_time_s / completed_steps
    total_loop_s = float(summary["timing_seconds"]["total_time_loop_s"])
    solve_s = float(summary["timing_seconds"]["solve_newton_linear_s"])
    vtk_s = float(summary["timing_seconds"]["vtk_output_s"])
    seconds_per_step = total_loop_s / completed_steps

    validation = comparison["validation_window"]
    reference_peak_time_s = float(validation["reference_peak_pressure_time_s"])
    reference_end_time_s = float(validation["reference_time_end_s"])

    return {
        "qualification_dir": str(qualification_dir),
        "smoke_summary": smoke_summary_name,
        "pressure_comparison": pressure_comparison_name,
        "smoke": {
            "steps_taken": completed_steps,
            "final_time_s": final_time_s,
            "time_step_s": dt_s,
            "total_time_loop_s": total_loop_s,
            "solve_newton_linear_s": solve_s,
            "vtk_output_s": vtk_s,
            "wall_seconds_per_step": seconds_per_step,
            "solve_seconds_per_step": solve_s / completed_steps,
            "vtk_seconds_per_step": vtk_s / completed_steps,
            "reference_coverage_fraction": validation["reference_coverage_fraction"],
        },
        "targets": [
            target_projection(
                target_name="reference_pressure_peak",
                target_time_s=reference_peak_time_s,
                dt_s=dt_s,
                completed_steps=completed_steps,
                seconds_per_step=seconds_per_step,
            ),
            target_projection(
                target_name="full_reference_horizon",
                target_time_s=reference_end_time_s,
                dt_s=dt_s,
                completed_steps=completed_steps,
                seconds_per_step=seconds_per_step,
            ),
        ],
        "finding": (
            "At the measured 40-step direct-solver rate, the full Test10 "
            "pressure-history run is a multi-day wall-clock job. This explains "
            "why the completed smoke is a plumbing/forcing diagnostic only; it "
            "does not replace a pressure-history validation or convergence gate."
        ),
        "status": "runtime_projection_only_long_pressure_history_still_missing",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        default=DEFAULT_QUALIFICATION_DIR,
        help="Directory containing Test10 smoke and pressure-comparison artifacts.",
    )
    parser.add_argument(
        "--smoke-summary",
        default=DEFAULT_SMOKE_SUMMARY,
        help="40-step full-source smoke summary JSON file name.",
    )
    parser.add_argument(
        "--pressure-comparison",
        default=DEFAULT_PRESSURE_COMPARISON,
        help="40-step pressure comparison JSON file name.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the projection JSON. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection = build_projection(
        args.qualification_dir,
        args.smoke_summary,
        args.pressure_comparison,
    )
    text = json.dumps(projection, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
