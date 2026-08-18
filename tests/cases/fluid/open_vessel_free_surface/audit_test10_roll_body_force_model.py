#!/usr/bin/env python3
"""Audit the SPHERIC Test10 fixed-tank roll body-force approximation."""

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
    source_path = None
    for tag in (
        "Momentum_source_temporal_and_spatial_values_file_path",
        "MomentumSourceTemporalAndSpatialValuesFilePath",
        "Body_force_temporal_and_spatial_values_file_path",
        "BodyForceTemporalAndSpatialValuesFilePath",
    ):
        source_path = child_text(fluid, tag)
        if source_path:
            break
    angular_velocity_path = None
    for tag in (
        "Rotating_frame_angular_velocity_temporal_values_file_path",
        "RotatingFrameAngularVelocityTemporalValuesFilePath",
        "Angular_velocity_temporal_values_file_path",
        "AngularVelocityTemporalValuesFilePath",
    ):
        angular_velocity_path = child_text(fluid, tag)
        if angular_velocity_path:
            break
    return {
        "number_of_time_steps": int(child_text(general, "Number_of_time_steps") or 0),
        "time_step_size_s": float(child_text(general, "Time_step_size") or 0.0),
        "force_m_per_s2": np.asarray(
            [
                float(child_text(fluid, "Force_x") or 0.0),
                float(child_text(fluid, "Force_y") or 0.0),
                float(child_text(fluid, "Force_z") or 0.0),
            ],
            dtype=float,
        ),
        "momentum_source_path": source_path,
        "rotating_frame_angular_velocity_path": angular_velocity_path,
        "mesh_motion_equation_present": root.find("./Add_equation[@type='mesh_motion']") is not None,
    }


def load_reference(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="latin1", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"reference table has no rows: {path}")
    return {
        "time_s": np.asarray([float(row["Time[s]"]) for row in rows], dtype=float),
        "pressure_pa": 100.0
        * np.asarray([float(row["Pressure[mbar]"]) for row in rows], dtype=float),
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


def read_source_times(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) < 3:
            raise RuntimeError(f"bad temporal/spatial source header: {path}")
        dof = int(header[0])
        time_count = int(header[1])
        node_count = int(header[2])
        times = np.asarray([float(stream.readline()) for _ in range(time_count)], dtype=float)
    return {
        "path": str(path),
        "dof": dof,
        "time_count": time_count,
        "node_count": node_count,
        "time_start_s": float(times[0]) if times.size else None,
        "time_end_s": float(times[-1]) if times.size else None,
        "time_step_s_median": float(np.median(np.diff(times))) if times.size > 1 else None,
    }


def read_angular_velocity_times(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) < 2:
            raise RuntimeError(f"bad angular velocity temporal header: {path}")
        time_count = int(header[0])
        times = np.zeros(time_count, dtype=float)
        omega = np.zeros((time_count, 3), dtype=float)
        for index in range(time_count):
            row = stream.readline().split()
            if len(row) != 4:
                raise RuntimeError(f"bad angular velocity row {index} in {path}")
            times[index] = float(row[0])
            omega[index, :] = [float(row[1]), float(row[2]), float(row[3])]
    return {
        "path": str(path),
        "time_count": time_count,
        "time_start_s": float(times[0]) if times.size else None,
        "time_end_s": float(times[-1]) if times.size else None,
        "time_step_s_median": float(np.median(np.diff(times))) if times.size > 1 else None,
        "component_min_rad_per_s": np.min(omega, axis=0).tolist() if times.size else None,
        "component_max_rad_per_s": np.max(omega, axis=0).tolist() if times.size else None,
        "max_norm_rad_per_s": float(np.max(np.linalg.norm(omega, axis=1))) if times.size else 0.0,
        "max_coriolis_coefficient_2omega_1_per_s": (
            float(2.0 * np.max(np.linalg.norm(omega, axis=1))) if times.size else 0.0
        ),
    }


def roll_model_term_maxima(
    points: np.ndarray,
    reference: dict[str, np.ndarray],
    *,
    axis_point: np.ndarray,
    base_force: np.ndarray,
    gravity: float,
    mask: np.ndarray,
) -> dict[str, float]:
    rel = points - axis_point.reshape(1, 3)
    xr = rel[:, 0]
    yr = rel[:, 1]
    theta = reference["theta_rad"][mask]
    omega = reference["omega_rad_s"][mask]
    alpha = reference["alpha_rad_s2"][mask]
    if theta.size == 0:
        return {
            "rotated_gravity_increment_max_norm_m_per_s2": 0.0,
            "euler_acceleration_max_norm_m_per_s2": 0.0,
            "centrifugal_acceleration_max_norm_m_per_s2": 0.0,
            "total_static_increment_max_norm_m_per_s2": 0.0,
        }

    maxima = {
        "rotated_gravity_increment_max_norm_m_per_s2": 0.0,
        "euler_acceleration_max_norm_m_per_s2": 0.0,
        "centrifugal_acceleration_max_norm_m_per_s2": 0.0,
        "total_static_increment_max_norm_m_per_s2": 0.0,
    }
    for sl in np.array_split(np.arange(theta.size), min(64, max(1, theta.size))):
        th = theta[sl][:, None]
        om = omega[sl][:, None]
        al = alpha[sl][:, None]

        gx = -gravity * np.sin(th) - base_force[0]
        gy = -gravity * np.cos(th) - base_force[1]
        gx = gx + np.zeros((sl.size, points.shape[0]))
        gy = gy + np.zeros((sl.size, points.shape[0]))

        ex = al * yr[None, :]
        ey = -al * xr[None, :]
        cx = (om * om) * xr[None, :]
        cy = (om * om) * yr[None, :]

        maxima["rotated_gravity_increment_max_norm_m_per_s2"] = max(
            maxima["rotated_gravity_increment_max_norm_m_per_s2"],
            float(np.max(np.sqrt(gx * gx + gy * gy))),
        )
        maxima["euler_acceleration_max_norm_m_per_s2"] = max(
            maxima["euler_acceleration_max_norm_m_per_s2"],
            float(np.max(np.sqrt(ex * ex + ey * ey))),
        )
        maxima["centrifugal_acceleration_max_norm_m_per_s2"] = max(
            maxima["centrifugal_acceleration_max_norm_m_per_s2"],
            float(np.max(np.sqrt(cx * cx + cy * cy))),
        )
        tx = gx + ex + cx
        ty = gy + ey + cy
        maxima["total_static_increment_max_norm_m_per_s2"] = max(
            maxima["total_static_increment_max_norm_m_per_s2"],
            float(np.max(np.sqrt(tx * tx + ty * ty))),
        )
    return maxima


def result_step(path: Path) -> int:
    match = re.search(r"_(\d+)\.p?vtu$", path.name)
    return int(match.group(1)) if match else -1


def pvd_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    out: dict[str, float] = {}
    for data_set in root.findall(".//DataSet"):
        file_name = data_set.get("file")
        timestep = data_set.get("timestep")
        if file_name is not None and timestep is not None:
            out[Path(file_name).name] = float(timestep)
    return out


def velocity_stats(
    result_case_dir: Path | None,
    reference: dict[str, np.ndarray],
    *,
    default_dt: float,
) -> dict[str, Any] | None:
    if result_case_dir is None or not result_case_dir.exists():
        return None
    results = sorted(
        [*result_case_dir.glob("result_*.vtu"), *result_case_dir.glob("result_*.pvtu")],
        key=result_step,
    )
    if not results:
        return None
    pvd = pvd_times(result_case_dir / "result.pvd")
    records = []
    max_speed = 0.0
    max_wet_speed = 0.0
    max_observed_coriolis = 0.0
    max_observed_wet_coriolis = 0.0
    for result in results:
        grid = pv.read(result)
        if "Velocity" not in grid.point_data:
            continue
        velocity = np.asarray(grid.point_data["Velocity"], dtype=float)
        speed = np.linalg.norm(velocity, axis=1)
        time_s = pvd.get(result.name, result_step(result) * default_dt)
        omega = float(np.interp(time_s, reference["time_s"], reference["omega_rad_s"]))
        record = {
            "result": result.name,
            "time_s": float(time_s),
            "max_speed_m_per_s": float(np.max(speed)),
            "omega_rad_s": omega,
            "max_coriolis_bound_m_per_s2": float(2.0 * abs(omega) * np.max(speed)),
        }
        if "phi" in grid.point_data:
            phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
            wet = np.isfinite(phi) & (phi <= 0.0)
            if np.any(wet):
                wet_speed = speed[wet]
                record["max_wet_speed_m_per_s"] = float(np.max(wet_speed))
                record["max_wet_coriolis_bound_m_per_s2"] = float(
                    2.0 * abs(omega) * np.max(wet_speed)
                )
                max_wet_speed = max(max_wet_speed, record["max_wet_speed_m_per_s"])
                max_observed_wet_coriolis = max(
                    max_observed_wet_coriolis,
                    record["max_wet_coriolis_bound_m_per_s2"],
                )
        max_speed = max(max_speed, record["max_speed_m_per_s"])
        max_observed_coriolis = max(max_observed_coriolis, record["max_coriolis_bound_m_per_s2"])
        records.append(record)
    if not records:
        return None
    return {
        "case_dir": str(result_case_dir),
        "result_count": len(records),
        "time_start_s": records[0]["time_s"],
        "time_end_s": records[-1]["time_s"],
        "max_speed_m_per_s": max_speed,
        "max_wet_speed_m_per_s": max_wet_speed,
        "max_coriolis_bound_m_per_s2": max_observed_coriolis,
        "max_wet_coriolis_bound_m_per_s2": max_observed_wet_coriolis,
        "selected_records": [
            records[0],
            records[len(records) // 2],
            records[-1],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument("--result-case-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--gravity", type=float, default=9.81)
    args = parser.parse_args()

    solver = parse_solver_xml(args.case_dir / "solver.xml")
    benchmark = json.loads((args.case_dir / "benchmark.json").read_text(encoding="utf-8"))
    mesh_path = args.case_dir / "mesh" / "background" / "mesh-complete.mesh.vtu"
    mesh = pv.read(mesh_path)
    points = np.asarray(mesh.points, dtype=float)
    reference = load_reference(args.reference_file)

    source_path = (
        args.case_dir / solver["momentum_source_path"]
        if solver["momentum_source_path"] is not None
        else None
    )
    source = read_source_times(source_path) if source_path is not None else None
    angular_velocity_path = (
        args.case_dir / solver["rotating_frame_angular_velocity_path"]
        if solver["rotating_frame_angular_velocity_path"] is not None
        else None
    )
    angular_velocity = (
        read_angular_velocity_times(angular_velocity_path)
        if angular_velocity_path is not None
        else None
    )
    source_end = source["time_end_s"] if source is not None else 0.0
    angular_velocity_end = (
        angular_velocity["time_end_s"] if angular_velocity is not None else 0.0
    )
    reference_end = float(reference["time_s"][-1])
    source_reaches_reference_end = source is not None and source_end >= reference_end - 1.0e-12
    angular_velocity_reaches_reference_end = (
        angular_velocity is not None
        and angular_velocity_end >= reference_end - 1.0e-12
    )
    source_mask = reference["time_s"] <= float(source_end) + 1.0e-12
    full_mask = np.ones_like(reference["time_s"], dtype=bool)

    axis = benchmark.get("rotation_axis", {})
    axis_point = np.asarray(axis.get("point", [0.45, 0.0, 0.031]), dtype=float)
    full_terms = roll_model_term_maxima(
        points,
        reference,
        axis_point=axis_point,
        base_force=solver["force_m_per_s2"],
        gravity=args.gravity,
        mask=full_mask,
    )
    source_terms = roll_model_term_maxima(
        points,
        reference,
        axis_point=axis_point,
        base_force=solver["force_m_per_s2"],
        gravity=args.gravity,
        mask=source_mask,
    )

    full_coriolis_coeff = float(2.0 * np.max(np.abs(reference["omega_rad_s"])))
    source_coriolis_coeff = (
        float(2.0 * np.max(np.abs(reference["omega_rad_s"][source_mask])))
        if np.any(source_mask)
        else 0.0
    )
    velocity_threshold = (
        full_terms["total_static_increment_max_norm_m_per_s2"] / full_coriolis_coeff
        if full_coriolis_coeff > 0.0
        else math.inf
    )

    run_velocity = velocity_stats(
        args.result_case_dir,
        reference,
        default_dt=solver["time_step_size_s"],
    )

    summary = {
        "case_dir": str(args.case_dir),
        "reference_file": str(args.reference_file),
        "mesh_points": int(points.shape[0]),
        "mesh_cells": int(mesh.n_cells),
        "rotation_axis_point_m": axis_point.tolist(),
        "solver": {
            "time_step_size_s": solver["time_step_size_s"],
            "number_of_time_steps": solver["number_of_time_steps"],
            "base_force_m_per_s2": solver["force_m_per_s2"].tolist(),
            "momentum_source_path": solver["momentum_source_path"],
            "rotating_frame_angular_velocity_path": solver["rotating_frame_angular_velocity_path"],
            "mesh_motion_equation_present": solver["mesh_motion_equation_present"],
        },
        "source_file": source,
        "angular_velocity_file": angular_velocity,
        "input_history_coverage": {
            "reference_time_end_s": reference_end,
            "source_reaches_reference_end": source_reaches_reference_end,
            "angular_velocity_reaches_reference_end": angular_velocity_reaches_reference_end,
            "full_roll_force_input_history_available": (
                source_reaches_reference_end and angular_velocity_reaches_reference_end
            ),
        },
        "reference_roll_and_pressure": {
            "rows": int(reference["time_s"].size),
            "time_start_s": float(reference["time_s"][0]),
            "time_end_s": reference_end,
            "pressure_min_pa": float(np.min(reference["pressure_pa"])),
            "pressure_max_pa": float(np.max(reference["pressure_pa"])),
            "roll_position_max_abs_deg": float(np.max(np.abs(np.rad2deg(reference["theta_rad"])))),
            "roll_velocity_max_abs_rad_per_s": float(np.max(np.abs(reference["omega_rad_s"]))),
            "roll_acceleration_max_abs_rad_per_s2": float(np.max(np.abs(reference["alpha_rad_s2"]))),
        },
        "fixed_tank_static_terms": {
            "source_file_window": {
                "time_end_s": source_end,
                **source_terms,
                "coriolis_coefficient_max_2omega_1_per_s": source_coriolis_coeff,
                "coriolis_configured": angular_velocity is not None,
            },
            "full_reference_window": {
                **full_terms,
                "coriolis_coefficient_max_2omega_1_per_s": full_coriolis_coeff,
                "velocity_for_coriolis_to_match_static_increment_m_per_s": velocity_threshold,
            },
        },
        "observed_run_velocity_and_coriolis_bound": run_velocity,
    }
    if source_reaches_reference_end and angular_velocity_reaches_reference_end:
        summary["finding"] = (
            "The checked fixed-tank Test10 source now includes full-reference static "
            "roll-frame accelerations and a matching angular-velocity temporal history "
            "through the supplied SPHERIC record, so the OOP Navier-Stokes residual can "
            "include the velocity-dependent Coriolis term for a long pressure-history run. "
            f"The remaining Test10 blocker is no longer source/omega duration; it is the "
            f"unrun long pressure-history transient and convergence/refinement gate through "
            f"{reference_end:.6g} s. Over that record, 2|omega| reaches "
            f"{full_coriolis_coeff:.6g} 1/s and Coriolis acceleration can be comparable "
            f"to the largest static roll-frame acceleration at speeds near "
            f"{velocity_threshold:.6g} m/s."
        )
    else:
        summary["finding"] = (
            "The checked fixed-tank Test10 source now includes the static roll-frame "
            "accelerations and a matching angular-velocity temporal history, so the OOP "
            "Navier-Stokes residual can include the velocity-dependent Coriolis term. "
            f"The current checked source covers {source_end:.6g} s and the angular-velocity "
            f"history covers {angular_velocity_end:.6g} s; full SPHERIC pressure-history "
            f"validation needs both histories through {reference_end:.6g} s, where "
            f"2|omega| reaches {full_coriolis_coeff:.6g} 1/s and Coriolis acceleration can "
            f"be comparable to the largest static roll-frame acceleration at speeds near "
            f"{velocity_threshold:.6g} m/s."
        )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
