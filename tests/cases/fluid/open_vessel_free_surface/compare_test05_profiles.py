#!/usr/bin/env python3
"""Compare SPHERIC Test05 free-surface profiles against solver output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyvista as pv


def _load_reference_profile(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"expected two-column reference profile: {path}")

    # SPHERIC Test05 profile files are digitized in centimeters.
    return data.astype(float) * 0.01


def _contour_level_set(result_path: Path, scalar_name: str) -> np.ndarray:
    mesh = pv.read(result_path)
    if scalar_name not in mesh.point_data:
        available = ", ".join(sorted(mesh.point_data.keys()))
        raise ValueError(
            f"point scalar {scalar_name!r} is not present in {result_path}; "
            f"available point arrays: {available}"
        )

    interface = mesh.contour(isosurfaces=[0.0], scalars=scalar_name)
    if interface.n_points == 0:
        raise ValueError(f"no {scalar_name}=0 contour was generated from {result_path}")

    return np.asarray(interface.points, dtype=float)


def _sample_top_profile(
    points: np.ndarray,
    reference_x: np.ndarray,
    x_min: float,
    x_max: float,
    sample_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max)]
    if selected.size == 0:
        raise ValueError(
            f"no interface points found in requested x-window [{x_min}, {x_max}]"
        )

    profile_y = np.full(reference_x.shape, np.nan, dtype=float)
    support_counts = np.zeros(reference_x.shape, dtype=int)

    for index, x_value in enumerate(reference_x):
        near = selected[np.abs(selected[:, 0] - x_value) <= sample_radius]
        if near.size == 0:
            continue
        support_counts[index] = near.shape[0]
        profile_y[index] = float(np.max(near[:, 1]))

    if np.count_nonzero(np.isfinite(profile_y)) >= 2:
        known = np.isfinite(profile_y)
        missing = ~known
        profile_y[missing] = np.interp(reference_x[missing], reference_x[known], profile_y[known])

    return profile_y, support_counts


def _metrics(sim_y: np.ndarray, ref_y: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(sim_y)
    if not np.any(valid):
        raise ValueError("no valid samples are available for comparison")

    errors = sim_y[valid] - ref_y[valid]
    return {
        "samples": int(np.count_nonzero(valid)),
        "coverage_fraction": float(np.count_nonzero(valid) / len(ref_y)),
        "rmse_m": float(math.sqrt(np.mean(errors * errors))),
        "mae_m": float(np.mean(np.abs(errors))),
        "max_abs_error_m": float(np.max(np.abs(errors))),
        "mean_error_m": float(np.mean(errors)),
    }


def _load_benchmark(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_volume(mesh: pv.DataSet) -> float:
    sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    if "Volume" not in sized.cell_data:
        return 0.0
    return float(np.asarray(sized.cell_data["Volume"], dtype=float).sum())


def _wet_mesh(mesh: pv.DataSet, scalar_name: str) -> pv.DataSet:
    if scalar_name not in mesh.point_data:
        return pv.UnstructuredGrid()
    return mesh.clip_scalar(scalars=scalar_name, value=0.0, invert=True)


def _point_array(mesh: pv.DataSet, name: str) -> np.ndarray | None:
    if name not in mesh.point_data:
        return None
    return np.asarray(mesh.point_data[name], dtype=float)


def _gauge_pressure(mesh: pv.DataSet, benchmark: dict[str, object]) -> dict[str, object]:
    gauge = benchmark.get("pressure_gauge")
    if not isinstance(gauge, dict):
        return {}
    node_id = gauge.get("node_id")
    expected = gauge.get("expected_initial_hydrostatic_pressure")
    if node_id is None or "Pressure" not in mesh.point_data:
        return {}

    gids = None
    for name in ("GlobalNodeID", "GlobalVertexID"):
        if name in mesh.point_data:
            gids = np.asarray(mesh.point_data[name]).reshape(-1)
            break
    if gids is None:
        return {"node_id": int(node_id), "found": False}

    indices = np.flatnonzero(gids == int(node_id))
    if indices.size == 0:
        return {"node_id": int(node_id), "found": False}

    pressure = np.asarray(mesh.point_data["Pressure"], dtype=float).reshape(-1)
    values = pressure[indices]
    result: dict[str, object] = {
        "node_id": int(node_id),
        "found": True,
        "matches": int(indices.size),
        "pressure": float(values[0]),
        "pressure_min_across_matches": float(np.min(values)),
        "pressure_max_across_matches": float(np.max(values)),
    }
    if isinstance(expected, (int, float)):
        result["expected_initial_hydrostatic_pressure"] = float(expected)
        result["hydrostatic_reference_error"] = float(values[0] - float(expected))
    return result


def _field_metrics(
    result_path: Path,
    scalar_name: str,
    benchmark: dict[str, object],
    density: float,
    initial_wet_volume: float | None,
    initial_kinetic_energy: float,
) -> dict[str, object]:
    mesh = pv.read(result_path)
    pressure = _point_array(mesh, "Pressure")
    velocity = _point_array(mesh, "Velocity")
    phi = _point_array(mesh, scalar_name)

    report: dict[str, object] = {
        "result": str(result_path),
        "points": int(mesh.n_points),
        "cells": int(mesh.n_cells),
    }

    if phi is not None:
        phi_values = phi.reshape(-1)
        wet = phi_values < 0.0
        report["wet_point_count"] = int(np.count_nonzero(wet))
        report["dry_point_count"] = int(phi_values.size - np.count_nonzero(wet))
        report["phi_min"] = float(np.min(phi_values))
        report["phi_max"] = float(np.max(phi_values))

        clipped = _wet_mesh(mesh, scalar_name)
        wet_volume = _sum_volume(clipped)
        report["wet_volume"] = wet_volume
        if initial_wet_volume is not None:
            report["wet_volume_drift"] = wet_volume - initial_wet_volume

        if velocity is not None and clipped.n_cells > 0:
            cell_data = clipped.point_data_to_cell_data()
            volumes = np.asarray(
                cell_data.compute_cell_sizes(length=False, area=False, volume=True)
                .cell_data["Volume"],
                dtype=float,
            )
            cell_velocity = np.asarray(cell_data.cell_data["Velocity"], dtype=float)
            speed2 = np.sum(cell_velocity * cell_velocity, axis=1)
            kinetic_energy = 0.5 * density * float(np.sum(volumes * speed2))
            report["kinetic_energy"] = kinetic_energy
            report["kinetic_energy_growth"] = kinetic_energy - initial_kinetic_energy

    if pressure is not None:
        p = pressure.reshape(-1)
        report["pressure_min"] = float(np.min(p))
        report["pressure_max"] = float(np.max(p))
        report["pressure_mean"] = float(np.mean(p))

    if velocity is not None:
        v = np.asarray(velocity, dtype=float)
        if v.ndim == 1:
            v = v.reshape((-1, 1))
        speed = np.linalg.norm(v, axis=1)
        report["velocity_max"] = float(np.max(speed))
        report["velocity_mean"] = float(np.mean(speed))
        max_index = int(np.argmax(speed))
        max_record: dict[str, object] = {
            "point_index": max_index,
            "speed": float(speed[max_index]),
        }
        if phi is not None:
            phi_values = phi.reshape(-1)
            wet = phi_values < 0.0
            if np.any(wet):
                report["velocity_wet_mean"] = float(np.mean(speed[wet]))
            max_record["phi"] = float(phi_values[max_index])
            max_record["region"] = "wet" if phi_values[max_index] < 0.0 else "dry"
            report["largest_velocity_in_wet_region"] = bool(phi_values[max_index] < 0.0)
        report["largest_velocity"] = max_record

    gauge = _gauge_pressure(mesh, benchmark)
    if gauge:
        report["pressure_gauge"] = gauge

    return report


def _validation_status(
    field_metrics: dict[str, object],
    benchmark: dict[str, object],
    stale_pressure_tolerance: float | None,
    min_velocity_max: float | None,
) -> dict[str, object]:
    failures: list[str] = []
    gauge = field_metrics.get("pressure_gauge")
    if isinstance(gauge, dict) and stale_pressure_tolerance is not None:
        verification = benchmark.get("pressure_gauge_verification")
        if isinstance(verification, dict):
            stale = verification.get("previous_invalid_d18_full_volume_hydrostatic_pressure")
            current = gauge.get("pressure")
            if isinstance(stale, (int, float)) and isinstance(current, (int, float)):
                if abs(float(current) - float(stale)) <= stale_pressure_tolerance:
                    failures.append("pressure gauge remains close to previous full-volume hydrostatic value")

    vmax = field_metrics.get("velocity_max")
    if min_velocity_max is not None and isinstance(vmax, (int, float)):
        if float(vmax) < min_velocity_max:
            failures.append("velocity maximum is below the required dynamic threshold")

    return {
        "passed": not failures,
        "failures": failures,
    }


def compare(args: argparse.Namespace) -> dict[str, object]:
    benchmark = _load_benchmark(args.benchmark_json)
    report: dict[str, object] = {
        "result": str(args.result),
        "scalar": args.scalar,
    }

    field_values = _field_metrics(
        result_path=args.result,
        scalar_name=args.scalar,
        benchmark=benchmark,
        density=args.density,
        initial_wet_volume=args.initial_wet_volume,
        initial_kinetic_energy=args.initial_kinetic_energy,
    )
    report["field_metrics"] = field_values
    report["validation"] = _validation_status(
        field_values,
        benchmark,
        args.stale_pressure_gauge_tolerance,
        args.min_velocity_max,
    )

    if args.reference_profile is None:
        return report

    reference = _load_reference_profile(args.reference_profile)
    reference_x = reference[:, 0]
    reference_y = reference[:, 1]
    x_min = args.x_min if args.x_min is not None else float(reference_x.min())
    x_max = args.x_max if args.x_max is not None else float(reference_x.max())
    sample_radius = args.sample_radius
    if sample_radius is None:
        spacing = np.diff(np.unique(np.sort(reference_x)))
        sample_radius = max(0.005, float(np.median(spacing)) if len(spacing) else 0.005)
    points = _contour_level_set(args.result, args.scalar)
    sim_y, support_counts = _sample_top_profile(
        points=points,
        reference_x=reference_x,
        x_min=x_min,
        x_max=x_max,
        sample_radius=sample_radius,
    )

    selected = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max)]
    metric_values = _metrics(sim_y, reference_y)
    metric_values.update(
        {
            "sample_radius_m": float(sample_radius),
            "interface_points_total": int(points.shape[0]),
            "interface_points_in_window": int(selected.shape[0]),
            "reference_points": int(reference.shape[0]),
            "direct_samples": int(np.count_nonzero(support_counts)),
            "direct_coverage_fraction": float(np.count_nonzero(support_counts) / len(reference_y)),
            "interpolated_samples": int(np.count_nonzero(np.isfinite(sim_y)) - np.count_nonzero(support_counts)),
            "simulated_front_x_m": float(selected[:, 0].max()),
            "reference_front_x_m": float(reference_x.max()),
            "front_error_m": float(selected[:, 0].max() - reference_x.max()),
            "front_position_role": (
                "diagnostic_only"
                if args.front_diagnostic_only or "wet_bed_depth" in benchmark.get("dimensions_m", {})
                else "validation"
            ),
            "simulated_peak_y_m": float(np.nanmax(sim_y)),
            "reference_peak_y_m": float(np.max(reference_y)),
            "peak_y_error_m": float(np.nanmax(sim_y) - np.max(reference_y)),
        }
    )

    report["profile_comparison"] = {
        "reference_profile": str(args.reference_profile),
        "x_window_m": [x_min, x_max],
        "metrics": metric_values,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a SPHERIC Test05 profile with a level-set result file."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("reference_profile", type=Path, nargs="?")
    parser.add_argument("--scalar", default="phi")
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--initial-wet-volume", type=float)
    parser.add_argument("--initial-kinetic-energy", type=float, default=0.0)
    parser.add_argument("--front-diagnostic-only", action="store_true")
    parser.add_argument("--stale-pressure-gauge-tolerance", type=float)
    parser.add_argument("--min-velocity-max", type=float)
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--sample-radius", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
