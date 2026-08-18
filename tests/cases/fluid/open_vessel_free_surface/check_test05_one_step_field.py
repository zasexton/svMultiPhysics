#!/usr/bin/env python3
"""Check one-step SPHERIC Test05 dam-break field plausibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


def _point_array(mesh: pv.DataSet, name: str) -> np.ndarray:
    if name not in mesh.point_data:
        available = ", ".join(sorted(mesh.point_data.keys()))
        raise ValueError(f"missing point array {name!r}; available: {available}")
    return np.asarray(mesh.point_data[name])


def _global_node_ids(mesh: pv.DataSet) -> np.ndarray:
    for name in ("GlobalNodeID", "GlobalVertexID"):
        if name in mesh.point_data:
            return np.asarray(mesh.point_data[name], dtype=np.int64)
    raise ValueError("missing GlobalNodeID/GlobalVertexID point array")


def _result_indices_by_initial_gid(initial: pv.DataSet, result: pv.DataSet) -> np.ndarray:
    result_gid = _global_node_ids(result)
    first_index: dict[int, int] = {}
    for index, gid in enumerate(result_gid):
        first_index.setdefault(int(gid), index)

    initial_gid = _global_node_ids(initial)
    missing = [int(gid) for gid in initial_gid if int(gid) not in first_index]
    if missing:
        raise ValueError(f"result is missing {len(missing)} initial node ids")
    return np.array([first_index[int(gid)] for gid in initial_gid], dtype=np.int64)


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _wet_volume(mesh: pv.UnstructuredGrid, scalar_name: str) -> float:
    return float(mesh.clip_scalar(scalars=scalar_name, value=0.0, invert=True).volume)


def _profile_height_change(
    points: np.ndarray,
    phi0: np.ndarray,
    phi1: np.ndarray,
    z_mid: float,
) -> dict[str, float | int]:
    on_midplane = np.isclose(points[:, 2], z_mid)
    x_values = np.unique(np.round(points[on_midplane, 0], 12))

    def profile(phi: np.ndarray) -> dict[float, float]:
        out: dict[float, float] = {}
        for x_value in x_values:
            mask = on_midplane & np.isclose(points[:, 0], x_value)
            ys = points[mask, 1]
            ph = phi[mask]
            order = np.argsort(ys)
            ys = ys[order]
            ph = ph[order]
            y_surface: float | None = None
            for i in range(len(ys) - 1):
                if ph[i] == 0.0:
                    y_surface = float(ys[i])
                if ph[i] <= 0.0 <= ph[i + 1] and ph[i + 1] != ph[i]:
                    t = -ph[i] / (ph[i + 1] - ph[i])
                    y_surface = float(ys[i] + t * (ys[i + 1] - ys[i]))
            if y_surface is not None:
                out[float(x_value)] = y_surface
        return out

    prof0 = profile(phi0)
    prof1 = profile(phi1)
    common = sorted(set(prof0) & set(prof1))
    if not common:
        return {"samples": 0}
    delta = np.array([prof1[x] - prof0[x] for x in common], dtype=float)
    return {
        "samples": int(len(common)),
        "min": float(np.min(delta)),
        "max": float(np.max(delta)),
        "mean": float(np.mean(delta)),
    }


def compute_metrics(case_dir: Path, result_path: Path) -> dict[str, Any]:
    benchmark = json.loads((case_dir / "benchmark.json").read_text())
    dimensions = benchmark["dimensions_m"]
    gate_x = float(dimensions["profile_window_x_min"])
    z_mid = 0.5 * float(dimensions["extrusion_breadth"])

    initial = pv.read(case_dir / "mesh" / "background" / "mesh-complete.mesh.vtu")
    result = pv.read(result_path)
    result_index = _result_indices_by_initial_gid(initial, result)

    points = np.asarray(initial.points, dtype=float)
    phi0 = _point_array(initial, "phi").astype(float)
    phi1 = _point_array(result, "phi").astype(float)[result_index]
    pressure0 = _point_array(initial, "Pressure").astype(float)
    pressure1 = _point_array(result, "Pressure").astype(float)[result_index]
    velocity = _point_array(result, "Velocity").astype(float)[result_index]
    speed = np.linalg.norm(velocity, axis=1)

    wet0 = phi0 < 0.0
    gate_region = (np.abs(points[:, 0] - gate_x) <= 0.025) & wet0
    front_region = (points[:, 0] >= gate_x - 0.03) & (points[:, 0] <= gate_x + 0.07) & wet0

    boundary = (
        np.isclose(points[:, 0], np.min(points[:, 0]))
        | np.isclose(points[:, 0], np.max(points[:, 0]))
        | np.isclose(points[:, 1], np.min(points[:, 1]))
        | np.isclose(points[:, 1], np.max(points[:, 1]))
        | np.isclose(points[:, 2], np.min(points[:, 2]))
        | np.isclose(points[:, 2], np.max(points[:, 2]))
    )

    initial_for_step = initial.copy(deep=True)
    initial_for_step.point_data["phi_step1"] = phi1
    volume0 = _wet_volume(initial, "phi")
    volume1 = _wet_volume(initial_for_step, "phi_step1")
    max_speed = float(np.max(speed))
    max_abs_w = float(np.max(np.abs(velocity[:, 2])))

    return {
        "case_dir": str(case_dir),
        "result_path": str(result_path),
        "n_initial_points": int(initial.n_points),
        "n_result_points": int(result.n_points),
        "finite": {
            "phi": bool(np.isfinite(phi1).all()),
            "velocity": bool(np.isfinite(velocity).all()),
            "pressure": bool(np.isfinite(pressure1).all()),
        },
        "wet_volume_initial": volume0,
        "wet_volume_result": volume1,
        "wet_volume_delta": float(volume1 - volume0),
        "wet_volume_relative_delta": float((volume1 - volume0) / volume0),
        "max_wall_velocity": float(np.max(speed[boundary])),
        "max_speed": max_speed,
        "max_abs_w": max_abs_w,
        "max_abs_w_over_max_speed": float(max_abs_w / max_speed) if max_speed > 0.0 else float("inf"),
        "gate_mean_velocity": [float(value) for value in np.mean(velocity[gate_region], axis=0)],
        "front_mean_velocity": [float(value) for value in np.mean(velocity[front_region], axis=0)],
        "speed_wet": _stats(speed[wet0]),
        "speed_all": _stats(speed),
        "pressure_delta_wet": _stats((pressure1 - pressure0)[wet0]),
        "phi_delta": _stats(phi1 - phi0),
        "profile_height_delta": _profile_height_change(points, phi0, phi1, z_mid),
    }


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for name, finite in metrics["finite"].items():
        if not finite:
            failures.append(f"{name} contains non-finite values")
    if abs(metrics["wet_volume_relative_delta"]) > args.max_volume_rel_drift:
        failures.append(
            "wet-volume relative drift "
            f"{metrics['wet_volume_relative_delta']:.6g} exceeds {args.max_volume_rel_drift:.6g}"
        )
    if metrics["max_wall_velocity"] > args.max_wall_velocity:
        failures.append(
            f"max wall velocity {metrics['max_wall_velocity']:.6g} exceeds {args.max_wall_velocity:.6g}"
        )
    if metrics["max_speed"] < args.min_max_speed:
        failures.append(
            f"max speed {metrics['max_speed']:.6g} is below {args.min_max_speed:.6g}"
        )
    if metrics["gate_mean_velocity"][0] < args.min_gate_mean_ux:
        failures.append(
            f"gate mean ux {metrics['gate_mean_velocity'][0]:.6g} is below {args.min_gate_mean_ux:.6g}"
        )
    if metrics["front_mean_velocity"][0] < args.min_front_mean_ux:
        failures.append(
            f"front mean ux {metrics['front_mean_velocity'][0]:.6g} is below {args.min_front_mean_ux:.6g}"
        )
    if metrics["max_abs_w"] > args.max_abs_w:
        failures.append(f"max |w| {metrics['max_abs_w']:.6g} exceeds {args.max_abs_w:.6g}")
    if metrics["max_abs_w_over_max_speed"] > args.max_transverse_ratio:
        failures.append(
            "max |w| / max speed "
            f"{metrics['max_abs_w_over_max_speed']:.6g} exceeds {args.max_transverse_ratio:.6g}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-volume-rel-drift", type=float, default=5.0e-4)
    parser.add_argument("--max-wall-velocity", type=float, default=1.0e-10)
    parser.add_argument("--min-max-speed", type=float, default=1.0e-2)
    parser.add_argument("--min-gate-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-front-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--max-abs-w", type=float, default=5.0e-3)
    parser.add_argument("--max-transverse-ratio", type=float, default=5.0e-2)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    metrics = compute_metrics(args.case_dir, args.result)
    failures = evaluate(metrics, args)
    metrics["passed"] = not failures
    metrics["failures"] = failures

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.write_text(text + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
