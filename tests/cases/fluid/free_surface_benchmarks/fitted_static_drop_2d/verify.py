#!/usr/bin/env python3
"""Report the fitted_static_drop_2d smoke metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py in which the solver has
run.  The metrics are the pressure jump of the relaxed drop against gamma/R,
the spurious velocity history, and the liquid-area deviation (README.md).
Criteria whose level is not among the runs are reported as not evaluated.

Exit status: 0 if every evaluated criterion passes, 1 if one fails, 2 if
input data are missing or invalid.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(directory: str, name: str):
    path = HERE.parent / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FSV = _load("fitted_sloshing_2d", "verify")
DataError = FSV.DataError


def analyse_run(run: Path) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing")
    case = json.loads(case_file.read_text())
    series = FSV.output_series(run, case)
    if abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={case['end_time']:.6g}")
    pv = FSV._pyvista()
    times, speeds, areas, radii_spread = [], [], [], []
    area0 = None
    last = None
    for t, path in [(0.0, run / "mesh" / "mesh-complete.mesh.vtu")] + series:
        reference = t == 0.0
        grid = pv.read(path)
        points = FSV.current_points(grid, case, reference)
        tris = np.asarray(grid.cells).reshape(-1, 4)[:, 1:].astype(np.int64)
        velocity = np.asarray(grid.point_data[case["velocity_field"]], dtype=float)
        velocity = velocity.reshape(grid.n_points, -1)[:, :2]
        area = FSV.mesh_area(points, tris)
        area0 = area if area0 is None else area0
        surface = points[np.asarray(case["free_surface_nodes"])]
        centre = points.mean(axis=0)
        r = np.linalg.norm(surface - centre, axis=1)
        times.append(t)
        speeds.append(float(np.max(np.linalg.norm(velocity, axis=1))))
        areas.append(area)
        radii_spread.append(float((r.max() - r.min()) / r.mean()))
        last = (grid, points, area)
    grid, points, area = last
    pressure = np.asarray(grid.point_data[case["pressure_field"]], dtype=float)
    gamma, radius = case["surface_tension"], case["radius"]
    r_eff = math.sqrt(area / math.pi)
    jump = float(np.mean(pressure))            # p_ext = 0
    return {
        "run": str(run),
        "level": case["level_rings"],
        "boundary_sides": case["boundary_sides"],
        "end_time": float(times[-1]),
        "pressure_jump": jump,
        "pressure_spread_over_laplace": float((pressure.max() - pressure.min()) / (gamma / radius)),
        "pressure_jump_relative_error": abs(jump - gamma / radius) / (gamma / radius),
        "pressure_jump_relative_error_effective_radius": abs(jump - gamma / r_eff) / (gamma / r_eff),
        "pressure_jump_over_polygon_balance": jump / case["pressure_regular_polygon_balance"],
        "spurious_capillary_number_max": float(max(speeds[1:]) * case["viscosity"] / gamma),
        "spurious_capillary_number_final": float(speeds[-1] * case["viscosity"] / gamma),
        "liquid_area_relative_deviation_max": float(np.max(np.abs(np.asarray(areas) - area0)) / area0),
        "surface_radius_spread_final": radii_spread[-1],
        "solver_log": FSV.solver_log_summary(run),
        "history": {"time": times, "max_speed": speeds, "area": areas},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--tolerances", type=Path, default=HERE / "tolerances.json")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        tolerances = json.loads(args.tolerances.read_text())
        runs = [analyse_run(run) for run in args.runs]
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    by_level = {r["level"]: r for r in runs}
    print(f"{'R/h':>4} {'N':>4} {'dp':>12} {'err(g/R)':>10} {'err(g/Reff)':>11} "
          f"{'dp/p_poly':>10} {'Ca max':>9} {'Ca end':>9} {'dA/A max':>9} {'s/step':>7}")
    for r in sorted(runs, key=lambda x: x["level"]):
        log = r["solver_log"] or {}
        print(f"{r['level']:>4} {r['boundary_sides']:>4} {r['pressure_jump']:>12.8f} "
              f"{r['pressure_jump_relative_error']:>10.3e} "
              f"{r['pressure_jump_relative_error_effective_radius']:>11.3e} "
              f"{r['pressure_jump_over_polygon_balance']:>10.7f} "
              f"{r['spurious_capillary_number_max']:>9.2e} {r['spurious_capillary_number_final']:>9.2e} "
              f"{r['liquid_area_relative_deviation_max']:>9.2e} "
              f"{log.get('wall_seconds_per_step', float('nan')):>7.2f}")
    verdicts = []
    for crit in tolerances["criteria"]:
        level, q, limit = crit["at_level"], crit["quantity"], crit.get("limit")
        if level not in by_level:
            verdicts.append((crit["id"], None, f"not evaluated (no run at R/h={level})"))
            continue
        value = by_level[level][q]
        if limit is None:
            verdicts.append((crit["id"], None, f"R/h={level}: {value:.3g} (reported)"))
        else:
            verdicts.append((crit["id"], value <= limit,
                             f"R/h={level}: {value:.3g} {'<=' if value <= limit else '>'} {limit:g}"))
    for cid, passed, text in verdicts:
        tag = "INFO" if passed is None else ("PASS" if passed else "FAIL")
        print(f"  [{tag}] {cid}: {text}")
    ok = all(p is not False for _, p, _ in verdicts)
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"], "runs": runs,
                                         "criteria": [{"id": c, "passed": p, "details": t}
                                                      for c, p, t in verdicts],
                                         "passed": ok}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
