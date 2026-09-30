#!/usr/bin/env python3
"""Histories of a fitted-ALE SPHERIC Test 10 run (decks of generate_spheric_test10_fitted_decks.py).

Usage:
    analyze_spheric_test10_fitted_run.py RUN_DIR [--reference lateral_water_1x.txt] [--json OUT]

RUN_DIR holds benchmark.json, mesh/water/mesh-complete.mesh.vtu (the
reference configuration at t = 0) and the solver output result.pvd with
result_NNN.vtu (serial) or .pvtu (MPI).  For every output the script reports
the liquid volume (area in 2D) of the current mesh, the largest speed, the
mesh quality (smallest triangle angle in 2D; smallest tetrahedron dihedral
angle and smallest volume ratio V/V0 in 3D), the heights of the contact
points on the end walls, and the pressure at Sensor 1 (left wall, y = 93 mm):
interpolated linearly along the left-wall nodes in the current
configuration, and the exterior pressure 0 when the sensor is above the
contact point (dry).  With --reference the measured Sensor 1 pressure is
sampled at the output times.

Exit status: 0 on success, 2 on missing or invalid data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class DataError(RuntimeError):
    pass


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:      # pragma: no cover - environment dependent
        raise DataError("pyvista is required") from exc
    return pv


def output_series(run: Path) -> list[tuple[float, Path]]:
    pvd = run / "result.pvd"
    if not pvd.is_file():
        raise DataError(f"{run}: result.pvd is missing")
    root = ET.parse(pvd).getroot()
    series = sorted((float(d.get("timestep")), run / d.get("file")) for d in root.iter("DataSet"))
    if not series:
        raise DataError(f"{run}: no output listed in result.pvd")
    return series


def read_state(path: Path, dim: int, reference: bool) -> dict:
    grid = _pyvista().read(path)
    nv = dim + 1
    cells = np.asarray(grid.cells).reshape(-1, nv + 1)[:, 1:].astype(np.int64)
    if "GlobalElementID" in grid.cell_data:          # drop duplicated MPI cells
        _, first = np.unique(np.asarray(grid.cell_data["GlobalElementID"]), return_index=True)
        cells = cells[np.sort(first)]
    gid = np.asarray(grid.point_data["GlobalNodeID"]).astype(np.int64)
    points = np.asarray(grid.points, dtype=float)[:, :dim]
    if not reference:
        if "CurrentCoordinates" in grid.point_data:
            points = np.asarray(grid.point_data["CurrentCoordinates"], dtype=float).reshape(
                grid.n_points, -1)[:, :dim]
        else:
            points = points + np.asarray(grid.point_data["mesh_displacement"], dtype=float).reshape(
                grid.n_points, -1)[:, :dim]
    velocity = np.asarray(grid.point_data["Velocity"], dtype=float).reshape(grid.n_points, -1)[:, :dim]
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).ravel()
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(velocity)) and np.all(np.isfinite(pressure))):
        raise DataError(f"{path}: non-finite values")
    return {"gid": gid, "points": points, "cells": cells, "velocity": velocity, "pressure": pressure}


def signed_measures(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p = points[cells]
    if cells.shape[1] == 3:
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def minimum_triangle_angle(points: np.ndarray, cells: np.ndarray) -> float:
    p = points[cells]
    worst = math.pi
    for a in range(3):
        u, v = p[:, (a + 1) % 3] - p[:, a], p[:, (a + 2) % 3] - p[:, a]
        c = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        worst = min(worst, float(np.min(np.arccos(np.clip(c, -1.0, 1.0)))))
    return math.degrees(worst)


def minimum_dihedral_angle(points: np.ndarray, cells: np.ndarray) -> float:
    p = points[cells]
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))       # face opposite vertex i
    normals = []
    for a, b, c in faces:
        n = np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a])
        normals.append(n / np.linalg.norm(n, axis=1)[:, None])
    worst = math.pi
    for i in range(4):
        for j in range(i + 1, 4):
            c = -np.sum(normals[i] * normals[j], axis=1)
            worst = min(worst, float(np.min(np.arccos(np.clip(c, -1.0, 1.0)))))
    return math.degrees(worst)


def sensor_pressure(state: dict, wall_nodes: np.ndarray, sensor_y: float) -> float:
    """Pressure at height sensor_y on the left wall; 0 (exterior) above the contact point."""
    y = state["points"][wall_nodes, 1]
    p = state["pressure"][wall_nodes]
    order = np.argsort(y)
    y, p = y[order], p[order]
    if sensor_y > y[-1]:
        return 0.0
    return float(np.interp(sensor_y, y, p))


def load_measured_pressure(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="latin1", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    t = np.array([float(r["Time[s]"]) for r in rows])
    p = np.array([float(r["Pressure[mbar]"]) for r in rows])
    return t, p


def analyse(run: Path, reference_file: Path | None = None) -> dict:
    meta = json.loads((run / "benchmark.json").read_text())
    dim = int(meta["mesh"]["dimension"])
    series = output_series(run)
    ref = read_state(run / "mesh/water/mesh-complete.mesh.vtu", dim, reference=True)
    index = {int(g): i for i, g in enumerate(ref["gid"])}
    wall = _pyvista().read(run / "mesh/water/mesh-surfaces/wall_left.vtp")
    wall_gid = np.asarray(wall.point_data["GlobalNodeID"]).astype(np.int64)
    sensor = meta["pressure_sensor"]["coordinates"]
    if dim == 3:        # the left-wall nodes of the sensor plane z = 0.031
        wall_gid = np.array([g for g in wall_gid
                             if abs(ref["points"][index[int(g)], 2] - sensor[2]) < 1e-9])
    surface = _pyvista().read(run / "mesh/water/mesh-surfaces/free_surface.vtp")
    surface_gid = np.asarray(surface.point_data["GlobalNodeID"]).astype(np.int64)
    length = meta["dimensions_m"]["tank_length"]
    v0_cells = signed_measures(ref["points"], ref["cells"])
    rows = []
    for t, path, is_ref in [(0.0, None, True)] + [(t, p, False) for t, p in series]:
        state = ref if is_ref else read_state(path, dim, reference=False)
        local = {int(g): i for i, g in enumerate(state["gid"])}
        measures = signed_measures(state["points"], state["cells"])
        wall_nodes = np.array([local[int(g)] for g in wall_gid])
        surf = state["points"][[local[int(g)] for g in surface_gid]]
        left = surf[np.abs(surf[:, 0]) < 1e-9 * length, 1]
        right = surf[np.abs(surf[:, 0] - length) < 1e-9 * length, 1]
        row = {"time": float(t),
               "volume": float(np.sum(measures)),
               "inverted_cells": int(np.sum(measures <= 0.0)),
               "max_speed": float(np.max(np.linalg.norm(state["velocity"], axis=1))),
               "surface_height_max": float(np.max(surf[:, 1])),
               "surface_height_min": float(np.min(surf[:, 1])),
               "contact_height_left": float(np.max(left)) if left.size else float("nan"),
               "contact_height_right": float(np.max(right)) if right.size else float("nan"),
               "sensor1_pressure_pa": sensor_pressure(state, wall_nodes, sensor[1])}
        if dim == 2:
            row["min_angle_deg"] = minimum_triangle_angle(state["points"], state["cells"])
        else:
            row["min_dihedral_deg"] = minimum_dihedral_angle(state["points"], state["cells"])
        row["min_volume_ratio"] = float(np.min(measures / v0_cells))
        rows.append(row)
    vol0 = rows[0]["volume"]
    result = {
        "run": str(run), "dim": dim, "outputs": len(rows) - 1,
        "end_time_reached": rows[-1]["time"],
        "steps_planned": meta["time"]["steps"], "dt": meta["time"]["dt"],
        "initial_volume": vol0,
        "volume_relative_deviation_max": max(abs(r["volume"] - vol0) for r in rows) / vol0,
        "volume_relative_deviation_final": abs(rows[-1]["volume"] - vol0) / vol0,
        "max_speed": max(r["max_speed"] for r in rows),
        "inverted_cells_max": max(r["inverted_cells"] for r in rows),
        "min_volume_ratio": min(r["min_volume_ratio"] for r in rows),
        "history": rows,
    }
    quality = "min_angle_deg" if dim == 2 else "min_dihedral_deg"
    result[quality + "_initial"] = rows[0][quality]
    result[quality + "_min"] = min(r[quality] for r in rows)
    result[quality + "_time_of_min"] = min(rows, key=lambda r: r[quality])["time"]
    if reference_file is not None:
        tm, pm = load_measured_pressure(reference_file)
        times = np.array([r["time"] for r in rows])
        sim = np.array([r["sensor1_pressure_pa"] for r in rows]) / 100.0      # mbar
        meas = np.interp(times, tm, pm)
        result["sensor1"] = {
            "simulated_peak_mbar": float(np.max(sim)),
            "simulated_peak_time": float(times[int(np.argmax(sim))]),
            "measured_peak_mbar_over_same_window": float(np.max(pm[tm <= times[-1]])),
            "measured_peak_time_over_same_window": float(tm[tm <= times[-1]][int(np.argmax(pm[tm <= times[-1]]))]),
            "measured_at_outputs_mbar": meas.tolist(),
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, help="lateral_water_1x.txt (measured Sensor 1)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyse(args.run, args.reference)
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    q = "min_angle_deg" if result["dim"] == 2 else "min_dihedral_deg"
    print(f"{result['run']}: {result['outputs']} outputs, t = {result['end_time_reached']:.4f} s "
          f"({result['steps_planned']} steps planned, dt = {result['dt']})")
    print(f"  volume deviation max {result['volume_relative_deviation_max']:.3e}, final "
          f"{result['volume_relative_deviation_final']:.3e}; max speed {result['max_speed']:.3e} m/s")
    print(f"  mesh quality: {q} {result[q + '_initial']:.2f} initially, min {result[q + '_min']:.2f} "
          f"at t = {result[q + '_time_of_min']:.4f}; min V/V0 {result['min_volume_ratio']:.3f}; "
          f"inverted cells {result['inverted_cells_max']}")
    if "sensor1" in result:
        s = result["sensor1"]
        print(f"  Sensor 1: simulated peak {s['simulated_peak_mbar']:.2f} mbar at {s['simulated_peak_time']:.3f} s; "
              f"measured peak over the same window {s['measured_peak_mbar_over_same_window']:.2f} mbar "
              f"at {s['measured_peak_time_over_same_window']:.3f} s")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
