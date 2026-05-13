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


def compare(args: argparse.Namespace) -> dict[str, object]:
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
            "simulated_peak_y_m": float(np.nanmax(sim_y)),
            "reference_peak_y_m": float(np.max(reference_y)),
            "peak_y_error_m": float(np.nanmax(sim_y) - np.max(reference_y)),
        }
    )

    return {
        "result": str(args.result),
        "reference_profile": str(args.reference_profile),
        "scalar": args.scalar,
        "x_window_m": [x_min, x_max],
        "metrics": metric_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a SPHERIC Test05 profile with a level-set result file."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("reference_profile", type=Path)
    parser.add_argument("--scalar", default="phi")
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
