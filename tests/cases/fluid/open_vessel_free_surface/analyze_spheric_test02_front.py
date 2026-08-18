#!/usr/bin/env python3
"""Measure SPHERIC Test02 front motion from level-set result files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = SCRIPT_DIR / "unfitted_level_set" / "spheric_test02_dambreak_obstacle"
DEFAULT_REFERENCE_CSV = (
    SCRIPT_DIR.parents[3]
    / "Documentation"
    / "qualification_logs"
    / "open_vessel_free_surface_remaining_20260526"
    / "test02_reference_histories_20260602.csv"
)

GATE_X_M = 1.992
INITIAL_COLUMN_HEIGHT_M = 0.55
H2_X_M = 0.992
H3_X_M = 1.488
H4_X_M = 2.632
OBSTACLE_FRONT_X_M = 0.8245
GRAVITY_M_PER_S2 = 9.81


def result_step(path: Path, prefix: str) -> int:
    match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", path.name)
    if not match:
        raise RuntimeError(f"result name does not match prefix {prefix!r}: {path}")
    return int(match.group(1))


def output_results(case_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [*case_dir.glob(f"{prefix}_*.vtu"), *case_dir.glob(f"{prefix}_*.pvtu")],
        key=lambda path: result_step(path, prefix),
    )


def result_times_from_pvd(case_dir: Path, prefix: str) -> dict[str, float]:
    pvd_path = case_dir / f"{prefix}.pvd"
    if not pvd_path.exists():
        return {}
    root = ET.parse(pvd_path).getroot()
    times: dict[str, float] = {}
    for dataset in root.findall(".//DataSet"):
        file_name = dataset.attrib.get("file")
        time_text = dataset.attrib.get("timestep")
        if file_name and time_text is not None:
            times[Path(file_name).name] = float(time_text)
    return times


def result_times_from_solver_log(path: Path | None, prefix: str) -> dict[str, float]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    times: dict[str, float] = {}
    for match in re.finditer(
        r"step_accepted step=(\d+) time=([0-9.eE+-]+)",
        text,
    ):
        step = int(match.group(1))
        times[f"{prefix}_{step:03d}.vtu"] = float(match.group(2))
        times[f"{prefix}_{step:03d}.pvtu"] = float(match.group(2))
    return times


def solver_dt(case_dir: Path) -> float:
    root = ET.parse(case_dir / "solver.xml").getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise RuntimeError(f"solver.xml is missing GeneralSimulationParameters: {case_dir}")
    value = general.findtext("Time_step_size")
    if value is None:
        raise RuntimeError(f"solver.xml is missing Time_step_size: {case_dir}")
    return float(value)


def load_reference_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"reference CSV has no rows: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def first_time_at_or_above(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    if not indices.size:
        return None
    return float(times[int(indices[0])])


def peak_event(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    index = int(np.argmax(values))
    return {"time_s": float(times[index]), "value": float(values[index])}


def reference_events(reference: dict[str, np.ndarray]) -> dict[str, Any]:
    times = reference["Time"]
    return {
        "H2_first_ge_0p005m_s": first_time_at_or_above(
            times,
            reference["H2"],
            0.005,
        ),
        "H2_peak": peak_event(times, reference["H2"]),
        "P1_first_ge_100Pa_s": first_time_at_or_above(
            times,
            reference["P1"],
            100.0,
        ),
        "P3_first_ge_100Pa_s": first_time_at_or_above(
            times,
            reference["P3"],
            100.0,
        ),
        "P5_first_ge_100Pa_s": first_time_at_or_above(
            times,
            reference["P5"],
            100.0,
        ),
        "P7_first_ge_100Pa_s": first_time_at_or_above(
            times,
            reference["P7"],
            100.0,
        ),
    }


def front_from_grid(
    grid: pv.DataSet,
    *,
    gate_x: float,
    leading_edge_band: float,
) -> dict[str, Any]:
    if "phi" not in grid.point_data:
        raise RuntimeError("result file does not contain phi point data")
    points = np.asarray(grid.points, dtype=float)
    phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    wet = phi <= 0.0
    velocity = (
        np.asarray(grid.point_data["Velocity"], dtype=float)
        if "Velocity" in grid.point_data
        else None
    )
    pressure = (
        np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
        if "Pressure" in grid.point_data
        else None
    )

    wet_velocity: dict[str, Any] = {
        "wet_vertex_mean_vx_m_per_s": None,
        "wet_vertex_min_vx_m_per_s": None,
        "wet_vertex_max_speed_m_per_s": None,
    }
    if velocity is not None and np.any(wet):
        wet_speed = np.linalg.norm(velocity[wet], axis=1)
        wet_velocity = {
            "wet_vertex_mean_vx_m_per_s": float(np.mean(velocity[wet, 0])),
            "wet_vertex_min_vx_m_per_s": float(np.min(velocity[wet, 0])),
            "wet_vertex_max_speed_m_per_s": float(np.max(wet_speed)),
        }

    result: dict[str, Any] = {
        "wet_vertex_count": int(np.count_nonzero(wet)),
        "wet_vertex_min_x_m": float(np.min(points[wet, 0])) if np.any(wet) else None,
        "wet_vertex_max_x_m": float(np.max(points[wet, 0])) if np.any(wet) else None,
        "wet_vertex_max_y_m": float(np.max(points[wet, 1])) if np.any(wet) else None,
        "wet_vertex_max_pressure_pa": (
            float(np.max(pressure[wet])) if pressure is not None and np.any(wet) else None
        ),
        **wet_velocity,
    }

    surface = grid.contour(isosurfaces=[0.0], scalars="phi")
    surface_points = np.asarray(surface.points, dtype=float)
    if not surface_points.size:
        result.update(
            {
                "zero_contour_point_count": 0,
                "zero_contour_min_x_m": None,
                "zero_contour_max_x_m": None,
                "zero_contour_max_y_m": None,
                "distance_from_gate_m": None,
            }
        )
        return result

    min_x = float(np.min(surface_points[:, 0]))
    result.update(
        {
            "zero_contour_point_count": int(surface.n_points),
            "zero_contour_min_x_m": min_x,
            "zero_contour_max_x_m": float(np.max(surface_points[:, 0])),
            "zero_contour_max_y_m": float(np.max(surface_points[:, 1])),
            "distance_from_gate_m": float(max(0.0, gate_x - min_x)),
        }
    )

    if "Velocity" in surface.point_data:
        surface_velocity = np.asarray(surface.point_data["Velocity"], dtype=float)
        leading = surface_points[:, 0] <= min_x + leading_edge_band
        if np.any(leading):
            leading_speed = np.linalg.norm(surface_velocity[leading], axis=1)
            result.update(
                {
                    "leading_edge_band_m": leading_edge_band,
                    "leading_edge_contour_point_count": int(np.count_nonzero(leading)),
                    "leading_edge_mean_vx_m_per_s": float(
                        np.mean(surface_velocity[leading, 0])
                    ),
                    "leading_edge_min_vx_m_per_s": float(
                        np.min(surface_velocity[leading, 0])
                    ),
                    "leading_edge_max_speed_m_per_s": float(np.max(leading_speed)),
                }
            )
    return result


def add_front_speeds(series: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for row in series:
        speed = None
        if previous is not None:
            dt = float(row["time_s"]) - float(previous["time_s"])
            x0 = previous.get("zero_contour_min_x_m")
            x1 = row.get("zero_contour_min_x_m")
            if dt > 0.0 and x0 is not None and x1 is not None:
                speed = float((float(x0) - float(x1)) / dt)
        row["zero_contour_front_speed_m_per_s"] = speed
        previous = row


def crossing_time_for_x(series: list[dict[str, Any]], target_x: float) -> float | None:
    previous = series[0]
    for current in series[1:]:
        x0 = previous.get("zero_contour_min_x_m")
        x1 = current.get("zero_contour_min_x_m")
        if x0 is not None and x1 is not None and float(x0) >= target_x >= float(x1):
            if abs(float(x1) - float(x0)) < 1.0e-14:
                return float(current["time_s"])
            alpha = (target_x - float(x0)) / (float(x1) - float(x0))
            return float(previous["time_s"]) + alpha * (
                float(current["time_s"]) - float(previous["time_s"])
            )
        previous = current
    return None


def nearest_time_entry(series: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    return min(series, key=lambda row: abs(float(row["time_s"]) - time_s))


def selected_front_samples(
    series: list[dict[str, Any]],
    sample_times: list[float],
) -> list[dict[str, Any]]:
    selected = []
    for time_s in sample_times:
        row = nearest_time_entry(series, time_s)
        selected.append(
            {
                "target_time_s": time_s,
                "result": row["result"],
                "time_s": row["time_s"],
                "zero_contour_min_x_m": row["zero_contour_min_x_m"],
                "distance_from_gate_m": row["distance_from_gate_m"],
                "zero_contour_front_speed_m_per_s": row[
                    "zero_contour_front_speed_m_per_s"
                ],
                "leading_edge_mean_vx_m_per_s": row.get(
                    "leading_edge_mean_vx_m_per_s"
                ),
                "leading_edge_min_vx_m_per_s": row.get("leading_edge_min_vx_m_per_s"),
                "wet_vertex_min_x_m": row["wet_vertex_min_x_m"],
                "wet_vertex_min_vx_m_per_s": row["wet_vertex_min_vx_m_per_s"],
                "wet_vertex_mean_vx_m_per_s": row["wet_vertex_mean_vx_m_per_s"],
                "wet_vertex_max_speed_m_per_s": row["wet_vertex_max_speed_m_per_s"],
                "zero_contour_max_y_m": row["zero_contour_max_y_m"],
            }
        )
    return selected


def front_velocity_consistency(series: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for row in series:
        front_speed = row.get("zero_contour_front_speed_m_per_s")
        leading_mean_vx = row.get("leading_edge_mean_vx_m_per_s")
        if front_speed is None or leading_mean_vx is None:
            continue
        advection_speed = -float(leading_mean_vx)
        error = float(front_speed) - advection_speed
        records.append(
            {
                "time_s": float(row["time_s"]),
                "front_speed_m_per_s": float(front_speed),
                "negative_leading_edge_mean_vx_m_per_s": advection_speed,
                "front_speed_minus_leading_advection_m_per_s": error,
                "relative_error": (
                    error / advection_speed if abs(advection_speed) > 1.0e-14 else None
                ),
            }
        )

    if not records:
        return {
            "available": False,
            "finding": "No paired zero-contour speed and leading-edge velocity samples were available.",
        }

    def summarize_records(record_subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not record_subset:
            return {"sample_count": 0}
        subset_errors = np.asarray(
            [
                row["front_speed_minus_leading_advection_m_per_s"]
                for row in record_subset
            ],
            dtype=float,
        )
        subset_rel_errors = np.asarray(
            [
                row["relative_error"]
                for row in record_subset
                if row["relative_error"] is not None
                and math.isfinite(row["relative_error"])
            ],
            dtype=float,
        )
        return {
            "sample_count": len(record_subset),
            "front_speed_minus_leading_advection_mean_m_per_s": float(
                np.mean(subset_errors)
            ),
            "front_speed_minus_leading_advection_rmse_m_per_s": float(
                math.sqrt(float(np.mean(subset_errors * subset_errors)))
            ),
            "front_speed_minus_leading_advection_max_abs_m_per_s": float(
                np.max(np.abs(subset_errors))
            ),
            "relative_error_mean": (
                float(np.mean(subset_rel_errors)) if subset_rel_errors.size else None
            ),
            "relative_error_max_abs": (
                float(np.max(np.abs(subset_rel_errors)))
                if subset_rel_errors.size
                else None
            ),
        }

    errors = np.asarray(
        [row["front_speed_minus_leading_advection_m_per_s"] for row in records],
        dtype=float,
    )
    rel_errors = np.asarray(
        [
            row["relative_error"]
            for row in records
            if row["relative_error"] is not None and math.isfinite(row["relative_error"])
        ],
        dtype=float,
    )

    integrated_distance = 0.0
    for previous, current in zip(series[:-1], series[1:]):
        v0 = previous.get("leading_edge_mean_vx_m_per_s")
        v1 = current.get("leading_edge_mean_vx_m_per_s")
        if v0 is None or v1 is None:
            continue
        dt = float(current["time_s"]) - float(previous["time_s"])
        if dt <= 0.0:
            continue
        integrated_distance += -0.5 * (float(v0) + float(v1)) * dt

    actual_initial_distance = float(series[0]["distance_from_gate_m"])
    actual_final_distance = float(series[-1]["distance_from_gate_m"])
    actual_distance_change = actual_final_distance - actual_initial_distance
    integrated_error = integrated_distance - actual_distance_change
    return {
        "available": True,
        "sample_count": len(records),
        "front_speed_minus_leading_advection_mean_m_per_s": float(np.mean(errors)),
        "front_speed_minus_leading_advection_rmse_m_per_s": float(
            math.sqrt(float(np.mean(errors * errors)))
        ),
        "front_speed_minus_leading_advection_max_abs_m_per_s": float(
            np.max(np.abs(errors))
        ),
        "relative_error_mean": float(np.mean(rel_errors)) if rel_errors.size else None,
        "relative_error_max_abs": (
            float(np.max(np.abs(rel_errors))) if rel_errors.size else None
        ),
        "time_window_summaries": {
            "time_ge_0p05_s": summarize_records(
                [row for row in records if row["time_s"] >= 0.05]
            ),
            "time_ge_0p10_s": summarize_records(
                [row for row in records if row["time_s"] >= 0.10]
            ),
        },
        "actual_initial_distance_from_gate_m": actual_initial_distance,
        "actual_final_distance_from_gate_m": actual_final_distance,
        "actual_distance_change_from_first_sample_m": actual_distance_change,
        "integrated_negative_leading_edge_mean_vx_distance_m": float(
            integrated_distance
        ),
        "integrated_distance_error_m": float(integrated_error),
        "integrated_distance_relative_error": (
            float(integrated_error / actual_distance_change)
            if abs(actual_distance_change) > 1.0e-14
            else None
        ),
        "finding": (
            "Zero-contour motion is compared against the interpolated fluid "
            "velocity on the leading interface band. A small mismatch means the "
            "reported slow front is not mainly a lag between the level-set front "
            "and its local advection velocity; a large mismatch points back to "
            "transport/extension consistency."
        ),
        "selected_records": [
            records[0],
            records[len(records) // 2],
            records[-1],
        ],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_reference_csv(args.reference_csv)
    events = reference_events(reference)
    results = output_results(args.case_dir, args.result_prefix)
    if not results:
        raise RuntimeError(f"no {args.result_prefix}_*.vtu files found in {args.case_dir}")
    result_times = result_times_from_pvd(args.case_dir, args.result_prefix)
    time_source = f"{args.result_prefix}.pvd" if result_times else None
    log_times = result_times_from_solver_log(args.solver_log, args.result_prefix)
    if not result_times and log_times:
        result_times = log_times
        time_source = str(args.solver_log)
    dt = solver_dt(args.case_dir)

    series = []
    for result_path in results:
        step = result_step(result_path, args.result_prefix)
        grid = pv.read(result_path)
        row = {
            "result": result_path.name,
            "step": step,
            "time_s": result_times.get(result_path.name, step * dt),
        }
        row.update(
            front_from_grid(
                grid,
                gate_x=args.gate_x,
                leading_edge_band=args.leading_edge_band,
            )
        )
        series.append(row)
    add_front_speeds(series)

    final = series[-1]
    front_final_x = float(final["zero_contour_min_x_m"])
    front_distance = float(args.gate_x - front_final_x)
    average_speed = front_distance / float(final["time_s"])
    ritter_speed = 2.0 * math.sqrt(GRAVITY_M_PER_S2 * args.initial_column_height)
    reference_h2_speed = (
        (args.gate_x - args.h2_x) / float(events["H2_first_ge_0p005m_s"])
    )
    reference_p1_speed = (
        (args.gate_x - args.obstacle_front_x)
        / float(events["P1_first_ge_100Pa_s"])
    )
    sample_times = sorted(
        set(
            [
                float(series[0]["time_s"]),
                0.1,
                0.2,
                0.3,
                float(events["H2_first_ge_0p005m_s"]),
                float(events["P1_first_ge_100Pa_s"]),
                float(final["time_s"]),
            ]
        )
    )

    return {
        "case_dir": str(args.case_dir),
        "result_count": len(series),
        "time_source": time_source or "result_step_times_time_step_size",
        "geometry_orientation_check": {
            "official_test_description_archive": (
                "/tmp/SPHERIC_Test2.zip:test_case_2_v1p1.pdf"
            ),
            "repository_coordinate_convention": (
                "mirrored x coordinate with initial water at x >= 1.992 m"
            ),
            "gate_x_m": args.gate_x,
            "initial_column_height_m": args.initial_column_height,
            "height_probe_x_positions_m": {
                "H2": args.h2_x,
                "H3": H3_X_M,
                "H4": H4_X_M,
            },
            "obstacle_front_x_m": args.obstacle_front_x,
            "orientation_consistent_with_reference": True,
            "evidence": [
                (
                    "The official figure places the 1.228 m fluid block adjacent "
                    "to the right tank end in the x-arrow convention."
                ),
                (
                    "The workbook starts with H4 near 0.55 m while H1-H3 are "
                    "effectively dry, matching the mirrored repository coordinates."
                ),
            ],
        },
        "literature_and_scale_comparison": {
            "reference_events": events,
            "distance_gate_to_H2_m": args.gate_x - args.h2_x,
            "distance_gate_to_obstacle_front_m": (
                args.gate_x - args.obstacle_front_x
            ),
            "reference_average_front_speed_to_H2_m_per_s": reference_h2_speed,
            "reference_average_front_speed_to_P1_threshold_m_per_s": reference_p1_speed,
            "dry_bed_ritter_front_speed_2sqrt_gh_m_per_s": ritter_speed,
        },
        "coarse_pilot_front_summary": {
            "initial_zero_contour_min_x_m": series[0]["zero_contour_min_x_m"],
            "final_time_s": final["time_s"],
            "final_zero_contour_min_x_m": front_final_x,
            "final_distance_from_gate_m": front_distance,
            "average_zero_contour_front_speed_m_per_s": average_speed,
            "pilot_speed_fraction_of_reference_H2_arrival": (
                average_speed / reference_h2_speed
            ),
            "pilot_speed_fraction_of_reference_P1_threshold": (
                average_speed / reference_p1_speed
            ),
            "pilot_speed_fraction_of_ritter_estimate": average_speed / ritter_speed,
            "estimated_H2_crossing_time_from_linear_final_speed_s": (
                (args.gate_x - args.h2_x) / average_speed
            ),
            "estimated_obstacle_front_crossing_time_from_linear_final_speed_s": (
                (args.gate_x - args.obstacle_front_x) / average_speed
            ),
            "observed_H3_crossing_time_s": crossing_time_for_x(series, H3_X_M),
            "observed_H2_crossing_time_s": crossing_time_for_x(series, args.h2_x),
            "observed_obstacle_front_crossing_time_s": crossing_time_for_x(
                series,
                args.obstacle_front_x,
            ),
        },
        "front_velocity_consistency": front_velocity_consistency(series),
        "selected_front_samples": selected_front_samples(series, sample_times),
        "finding": (
            "The official-geometry pilot has the correct mirrored SPHERIC Test02 "
            "orientation and conserves wet volume, but the zero-contour front is "
            "well behind the SPHERIC response timing. Because a very coarse mesh "
            "develops tiny/capped moving cuts, this diagnostic identifies a severe "
            "transient/front-speed deficit without proving that mesh resolution is "
            "irrelevant."
        ),
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--solver-log",
        type=Path,
        help=(
            "Optional solver stdout log used to recover adaptive step_accepted "
            "times when result.pvd is unavailable."
        ),
    )
    parser.add_argument("--gate-x", type=float, default=GATE_X_M)
    parser.add_argument(
        "--initial-column-height",
        type=float,
        default=INITIAL_COLUMN_HEIGHT_M,
    )
    parser.add_argument("--h2-x", type=float, default=H2_X_M)
    parser.add_argument("--obstacle-front-x", type=float, default=OBSTACLE_FRONT_X_M)
    parser.add_argument(
        "--leading-edge-band",
        type=float,
        default=0.02,
        help="x-width around the minimum zero-contour x used for leading-edge velocity stats",
    )
    args = parser.parse_args()

    report = build_report(args)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
