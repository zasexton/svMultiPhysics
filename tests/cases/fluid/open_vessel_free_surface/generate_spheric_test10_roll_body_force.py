#!/usr/bin/env python3
"""Generate a SPHERIC Test10 roll-equivalent body-force table.

The current Test10 unfitted fixture is a static tank.  This utility maps the
published roll position/velocity/acceleration table into a body-force source
for that tank frame.  The generated values are the incremental acceleration to
add on top of the XML's constant gravity force.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = (
    SCRIPT_DIR
    / "unfitted_level_set"
    / "spheric_test10_lateral_water_1x"
)
DEFAULT_REFERENCE_FILE = Path("/tmp/spheric_test10_lateral_water_1x.txt")


def child_text(parent: ET.Element, path: str) -> str | None:
    child = parent.find(path)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_solver_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    general = root.find("GeneralSimulationParameters")
    fluid = root.find("./Add_equation[@type='fluid']")
    if general is None or fluid is None:
        raise RuntimeError(f"solver.xml is missing required sections: {path}")
    return {
        "number_of_time_steps": int(child_text(general, "Number_of_time_steps") or 0),
        "time_step_size_s": float(child_text(general, "Time_step_size") or 0.0),
        "force": np.asarray(
            [
                float(child_text(fluid, "Force_x") or 0.0),
                float(child_text(fluid, "Force_y") or 0.0),
                float(child_text(fluid, "Force_z") or 0.0),
            ],
            dtype=float,
        ),
    }


def load_reference(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="latin1", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"reference table has no rows: {path}")
    return {
        "time_s": np.asarray([float(row["Time[s]"]) for row in rows], dtype=float),
        "theta_rad": np.deg2rad(
            np.asarray([float(row["Position_smooth_splines [deg]"]) for row in rows], dtype=float)
        ),
        "omega_rad_s": np.deg2rad(
            np.asarray([float(row["Velocity[deg\\s]"]) for row in rows], dtype=float)
        ),
        "alpha_rad_s2": np.deg2rad(
            np.asarray([float(row["Aceleration[deg\\s2]"]) for row in rows], dtype=float)
        ),
    }


def sample_reference(reference: dict[str, np.ndarray], times: np.ndarray) -> dict[str, np.ndarray]:
    src_t = reference["time_s"]
    return {
        "theta_rad": np.interp(times, src_t, reference["theta_rad"]),
        "omega_rad_s": np.interp(times, src_t, reference["omega_rad_s"]),
        "alpha_rad_s2": np.interp(times, src_t, reference["alpha_rad_s2"]),
    }


def roll_incremental_acceleration(
    points: np.ndarray,
    *,
    axis_point: np.ndarray,
    base_force: np.ndarray,
    theta: float,
    omega: float,
    alpha: float,
    gravity_magnitude: float,
) -> np.ndarray:
    """Return effective roll-frame acceleration minus the XML base force."""

    rel = points - axis_point.reshape(1, 3)
    xr = rel[:, 0]
    yr = rel[:, 1]

    effective = np.zeros_like(points, dtype=float)
    effective[:, 0] = (
        -gravity_magnitude * math.sin(theta)
        + alpha * yr
        + omega * omega * xr
    )
    effective[:, 1] = (
        -gravity_magnitude * math.cos(theta)
        - alpha * xr
        + omega * omega * yr
    )
    return effective - base_force.reshape(1, 3)


def write_source(
    output: Path,
    *,
    points: np.ndarray,
    global_ids: np.ndarray,
    times: np.ndarray,
    increments: list[np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(global_ids)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(f"3 {len(times)} {len(order)}\n")
        for time in times:
            stream.write(f"{time:.12e}\n")
        for point_index in order:
            stream.write(f"{int(global_ids[point_index]) + 1}\n")
            for values in increments:
                vx, vy, vz = values[point_index]
                stream.write(f"{vx:.18e} {vy:.18e} {vz:.18e}\n")


def write_angular_velocity(output: Path, *, times: np.ndarray, omega: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if times.size != omega.size:
        raise RuntimeError("angular velocity history must have one value per time sample")
    with output.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(times)} 0\n")
        for time, omega_z in zip(times, omega):
            stream.write(f"{time:.12e} 0.000000000000000000e+00 0.000000000000000000e+00 {omega_z:.18e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--angular-velocity-output", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--time-step", type=float)
    parser.add_argument("--gravity", type=float, default=9.81)
    args = parser.parse_args()

    case_dir = args.case_dir
    benchmark = json.loads((case_dir / "benchmark.json").read_text(encoding="utf-8"))
    solver = parse_solver_xml(case_dir / "solver.xml")
    mesh_path = case_dir / "mesh" / "background" / "mesh-complete.mesh.vtu"
    mesh = pv.read(mesh_path)
    points = np.asarray(mesh.points, dtype=float)
    if "GlobalNodeID" not in mesh.point_data:
        raise RuntimeError(f"mesh lacks GlobalNodeID point data: {mesh_path}")
    global_ids = np.asarray(mesh.point_data["GlobalNodeID"], dtype=int).reshape(-1)

    dt = args.time_step if args.time_step is not None else solver["time_step_size_s"]
    if not dt > 0.0:
        raise RuntimeError("time step must be positive")
    final_time = (
        args.duration
        if args.duration is not None
        else (solver["number_of_time_steps"] + 1) * dt
    )
    if final_time <= 0.0:
        raise RuntimeError("duration must be positive")
    times = np.arange(0.0, final_time + 0.5 * dt, dt, dtype=float)
    reference = load_reference(args.reference_file)
    sampled = sample_reference(reference, times)

    axis = benchmark.get("rotation_axis", {})
    axis_point = np.asarray(axis.get("point", [0.45, 0.0, 0.031]), dtype=float)
    increments = [
        roll_incremental_acceleration(
            points,
            axis_point=axis_point,
            base_force=solver["force"],
            theta=float(theta),
            omega=float(omega),
            alpha=float(alpha),
            gravity_magnitude=args.gravity,
        )
        for theta, omega, alpha in zip(
            sampled["theta_rad"],
            sampled["omega_rad_s"],
            sampled["alpha_rad_s2"],
        )
    ]

    output = args.output or (case_dir / "bc" / "test10_lateral_water_1x_roll_body_force.dat")
    write_source(output, points=points, global_ids=global_ids, times=times, increments=increments)
    angular_velocity_output = (
        args.angular_velocity_output
        or (case_dir / "bc" / "test10_lateral_water_1x_roll_angular_velocity.dat")
    )
    write_angular_velocity(
        angular_velocity_output,
        times=times,
        omega=sampled["omega_rad_s"])

    all_values = np.vstack(increments)
    summary = {
        "case_dir": str(case_dir),
        "reference_file": str(args.reference_file),
        "output": str(output),
        "angular_velocity_output": str(angular_velocity_output),
        "model": "static_tank_roll_frame_incremental_body_force_with_coriolis_omega_history",
        "model_components": [
            "Adds rotated gravity, Euler, and centrifugal accelerations on top of XML gravity through the body-force source.",
            "Writes a three-component angular velocity temporal history for OOP rotating-frame Coriolis forcing.",
        ],
        "model_limitations": [
            "Coriolis forcing is applied only when solver.xml includes Rotating_frame_angular_velocity_temporal_values_file_path.",
            "Does not move the closed tank walls; wall motion is represented through the tank-frame force model.",
        ],
        "rotation_axis_point_m": axis_point.tolist(),
        "xml_base_force_m_per_s2": solver["force"].tolist(),
        "time_step_s": float(dt),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "time_count": int(times.size),
        "mesh_points": int(points.shape[0]),
        "component_min_m_per_s2": np.min(all_values, axis=0).tolist(),
        "component_max_m_per_s2": np.max(all_values, axis=0).tolist(),
        "max_increment_norm_m_per_s2": float(np.max(np.linalg.norm(all_values, axis=1))),
        "sampled_roll_position_max_abs_deg": float(np.max(np.abs(np.rad2deg(sampled["theta_rad"])))),
        "sampled_roll_velocity_max_abs_rad_per_s": float(np.max(np.abs(sampled["omega_rad_s"]))),
        "sampled_roll_acceleration_max_abs_rad_per_s2": float(np.max(np.abs(sampled["alpha_rad_s2"]))),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
