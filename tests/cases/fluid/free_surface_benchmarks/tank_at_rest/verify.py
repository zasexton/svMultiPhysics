#!/usr/bin/env python3
"""Compute the tank_at_rest metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI).  Runs are
grouped by dimension; every criterion applies to every run.  Metric
definitions are in README.md.

Exit status: 0 if every criterion passes, 1 if any criterion fails, 2 if
input data are missing or invalid.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELL_TYPES = {2: 5, 3: 10}        # VTK_TRIANGLE, VTK_TETRA


class DataError(RuntimeError):
    """Missing or inconsistent input; verification cannot proceed."""


# ---------------------------------------------------------------------------
# Geometry of the P1 liquid region {phi_h < 0} on affine simplices
# ---------------------------------------------------------------------------
def _clip_triangle(p: np.ndarray, f: np.ndarray) -> float:
    """Area of {x in triangle : f_h(x) < 0} for linear f_h (Sutherland-Hodgman)."""
    poly = []
    for k in range(3):
        a, b = k, (k + 1) % 3
        if f[a] < 0.0:
            poly.append(p[a])
        if (f[a] < 0.0) != (f[b] < 0.0):
            s = f[a] / (f[a] - f[b])
            poly.append(p[a] + s * (p[b] - p[a]))
    if len(poly) < 3:
        return 0.0
    q = np.asarray(poly)
    x, y = q[:, 0], q[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _tet_volume(a, b, c, d) -> float:
    return abs(float(np.linalg.det(np.array([b - a, c - a, d - a])))) / 6.0


def _clip_tetrahedron(p: np.ndarray, f: np.ndarray) -> float:
    """Volume of {x in tetrahedron : f_h(x) < 0} for linear f_h.

    Uses only edge crossings between vertices of opposite sign, so equal
    vertex values (a level set parallel to a face) are handled exactly.
    """
    neg = [i for i in range(4) if f[i] < 0.0]
    pos = [i for i in range(4) if f[i] >= 0.0]

    def cross(i: int, j: int) -> np.ndarray:
        s = f[i] / (f[i] - f[j])
        return p[i] + s * (p[j] - p[i])

    if len(neg) == 0:
        return 0.0
    if len(neg) == 4:
        return _tet_volume(*p)
    if len(neg) == 1:
        a = neg[0]
        return _tet_volume(p[a], *(cross(a, j) for j in pos))
    if len(neg) == 3:
        d = pos[0]
        return _tet_volume(*p) - _tet_volume(p[d], *(cross(d, i) for i in neg))
    a, b = neg
    c, d = pos
    xac, xad, xbc, xbd = cross(a, c), cross(a, d), cross(b, c), cross(b, d)
    # Prism with lateral edges a-b, xac-xbc, xad-xbd, split into three tetrahedra.
    return (_tet_volume(p[a], xac, xad, p[b]) + _tet_volume(xac, xad, p[b], xbc)
            + _tet_volume(xad, p[b], xbc, xbd))


def liquid_measure(points: np.ndarray, cells: np.ndarray, phi: np.ndarray) -> float:
    """Exact area (2D) or volume (3D) of {phi_h < 0} for the P1 interpolant."""
    dim = cells.shape[1] - 1
    f = phi[cells]
    p = points[cells][:, :, :dim]
    full = np.all(f < 0.0, axis=1)
    cut = ~full & np.any(f < 0.0, axis=1)
    edges = p[full, 1:, :] - p[full, :1, :]
    if dim == 2:
        total = 0.5 * np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]).sum()
        clip = _clip_triangle
    else:
        total = np.abs(np.linalg.det(edges)).sum() / 6.0
        clip = _clip_tetrahedron
    for idx in np.nonzero(cut)[0]:
        total += clip(p[idx], f[idx])
    return float(total)


def interface_points(points: np.ndarray, cells: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Zero crossings of phi_h on the mesh edges (the LinearCorner polygon vertices)."""
    nv = cells.shape[1]
    pairs = [(i, j) for i in range(nv) for j in range(i + 1, nv)]
    edges = np.unique(np.sort(np.concatenate([cells[:, list(pr)] for pr in pairs]), axis=1), axis=0)
    fa, fb = phi[edges[:, 0]], phi[edges[:, 1]]
    crossing = (fa < 0.0) != (fb < 0.0)
    fa, fb, e = fa[crossing], fb[crossing], edges[crossing]
    s = fa / (fa - fb)
    return points[e[:, 0]] + s[:, None] * (points[e[:, 1]] - points[e[:, 0]])


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------
def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:     # pragma: no cover - environment dependent
        raise DataError("pyvista is required to read VTU/PVTU output") from exc
    return pv


def read_snapshot(path: Path, case: dict) -> dict:
    if not path.is_file():
        raise DataError(f"missing output file {path}")
    dim = case["dim"]
    grid = _pyvista().read(path)
    types = np.asarray(grid.celltypes)
    if grid.n_cells == 0 or np.any(types != CELL_TYPES[dim]):
        raise DataError(f"{path}: expected a pure {'Triangle3' if dim == 2 else 'Tetra4'} mesh")
    cells = np.asarray(grid.cells).reshape(-1, dim + 2)[:, 1:].astype(np.int64)
    if "GlobalElementID" in grid.cell_data:          # drop duplicated MPI cells
        _, first = np.unique(np.asarray(grid.cell_data["GlobalElementID"]), return_index=True)
        cells = cells[np.sort(first)]
    fields = {}
    for key in ("level_set_field", "velocity_field", "pressure_field"):
        name = case[key]
        if name not in grid.point_data:
            raise DataError(f"{path}: point array '{name}' is missing")
        fields[key] = np.asarray(grid.point_data[name], dtype=float)
    velocity = fields["velocity_field"].reshape(grid.n_points, -1)
    if velocity.shape[1] < dim:
        raise DataError(f"{path}: velocity must have at least {dim} components")
    data = {"points": np.asarray(grid.points, dtype=float), "cells": cells,
            "phi": fields["level_set_field"], "velocity": velocity[:, :dim],
            "pressure": fields["pressure_field"]}
    for key, value in data.items():
        if key != "cells" and not np.all(np.isfinite(value)):
            raise DataError(f"{path}: non-finite values in {key}")
    return data


def output_series(run: Path, case: dict) -> list[tuple[float, Path]]:
    prefix = case.get("result_prefix", "result")
    pvd = run / f"{prefix}.pvd"
    if pvd.is_file():
        root = ET.parse(pvd).getroot()
        series = [(float(d.get("timestep")), run / d.get("file")) for d in root.iter("DataSet")]
    else:
        files = sorted(list(run.glob(f"{prefix}_*.vtu")) + list(run.glob(f"{prefix}_*.pvtu")))
        series = []
        for f in files:
            m = re.search(r"_(\d+)\.p?vtu$", f.name)
            if m:
                series.append((int(m.group(1)) * case["dt"], f))
    if not series:
        raise DataError(f"{run}: no solver output ({prefix}.pvd or {prefix}_*.vtu)")
    series.sort(key=lambda item: item[0])
    return series


def wet_support(cells: np.ndarray, phi: np.ndarray, n_points: int) -> np.ndarray:
    """Vertices of the cells that contain liquid (at least one vertex with phi < 0).

    The P1 fields on the liquid region depend on exactly these values,
    including the dry vertices of cut cells.
    """
    wet_cells = np.any(phi[cells] < 0.0, axis=1)
    mask = np.zeros(n_points, dtype=bool)
    mask[np.unique(cells[wet_cells])] = True
    return mask


def snapshot_metrics(snap: dict, case: dict) -> dict:
    height = case["fill_height"]
    support = wet_support(snap["cells"], snap["phi"], len(snap["phi"]))
    if not np.any(support):
        raise DataError("no liquid vertex")
    y = snap["points"][:, 1]
    p_ref = case["external_pressure"] + case["density"] * case["gravity"] * (height - y)
    speed = np.linalg.norm(snap["velocity"], axis=1)
    iface = interface_points(snap["points"], snap["cells"], snap["phi"])
    if iface.size == 0:
        raise DataError("no interface crossing")
    return {
        "max_speed": float(np.max(speed[support])),
        "max_speed_all_vertices": float(np.max(speed)),
        "pressure_error": float(np.max(np.abs(snap["pressure"][support] - p_ref[support]))),
        "interface_height_error": float(np.max(np.abs(iface[:, 1] - height))),
        "liquid_volume": liquid_measure(snap["points"], snap["cells"], snap["phi"]),
    }


def solver_log_summary(run: Path) -> dict | None:
    """Newton statistics from the solver log, if it was kept (reported only)."""
    import gzip
    for name, opener in (("solver_run.log.gz", gzip.open), ("solver_run.log", open)):
        path = run / name
        if path.is_file():
            break
    else:
        return None
    pattern = re.compile(r"TimeLoop: nonlinear_done step=(\d+) .*?converged=(\d) iters=(\d+) "
                         r"\|\|r\|\|=(\S+) outer_iters=(\d+)")
    rows = []
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            m = pattern.search(line)
            if m:
                rows.append((int(m.group(2)), int(m.group(3)), float(m.group(4)), int(m.group(5))))
    if not rows:
        return None
    summary = {"steps_logged": len(rows),
               "nonconverged_steps": sum(1 for r in rows if r[0] != 1),
               "newton_iterations_total": sum(r[1] for r in rows),
               "outer_passes_mean": float(np.mean([r[3] for r in rows])),
               "final_residual_max": float(max(r[2] for r in rows))}
    run_txt = run / "run.txt"
    if run_txt.is_file():
        m = re.search(r"elapsed_s=(\d+)", run_txt.read_text())
        if m:
            summary["wall_seconds"] = int(m.group(1))
    return summary


def analyse_run(run: Path) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    series = output_series(run, case)
    if abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={case['end_time']:.6g}")

    initial = snapshot_metrics(read_snapshot(run / "mesh" / "mesh-complete.mesh.vtu", case), case)
    volume0 = initial["liquid_volume"]
    history = {"time": [], "max_speed": [], "pressure_error": [], "interface_height_error": [],
               "liquid_volume": []}
    all_speed = 0.0
    for t, path in series:
        m = snapshot_metrics(read_snapshot(path, case), case)
        history["time"].append(t)
        for key in ("max_speed", "pressure_error", "interface_height_error", "liquid_volume"):
            history[key].append(m[key])
        all_speed = max(all_speed, m["max_speed_all_vertices"])
    h = {k: np.asarray(v) for k, v in history.items()}
    u_scale, p_scale, height = case["velocity_scale"], case["pressure_scale"], case["fill_height"]
    return {
        "run": str(run),
        "dim": case["dim"],
        "level": case["level_cells_per_length"],
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(h["time"][-1]),
        "sloshing_periods_simulated": float(h["time"][-1] / case["sloshing_period"]),
        "outputs": int(h["time"].size),
        "max_speed_over_velocity_scale": float(np.max(h["max_speed"]) / u_scale),
        "max_speed_all_vertices_over_velocity_scale": float(all_speed / u_scale),
        "pressure_error_over_pressure_scale": float(np.max(h["pressure_error"]) / p_scale),
        "initial_pressure_error_over_pressure_scale": float(initial["pressure_error"] / p_scale),
        "interface_height_drift_over_fill_height": float(np.max(h["interface_height_error"]) / height),
        "initial_interface_height_error_over_fill_height": float(initial["interface_height_error"] / height),
        "liquid_volume_initial": volume0,
        "liquid_volume_initial_vs_exact": float(volume0 / case["liquid_volume_exact"] - 1.0),
        "liquid_volume_relative_drift_max": float(np.max(np.abs(h["liquid_volume"] - volume0)) / volume0),
        "solver_log": solver_log_summary(run),
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def evaluate_run(run: dict, tolerances: dict) -> list[dict]:
    results = []
    for crit in tolerances["criteria"]:
        value, limit = run[crit["quantity"]], crit["limit"]
        passed = value <= limit
        results.append({"id": crit["id"], "quantity": crit["quantity"], "value": value,
                        "limit": limit, "passed": bool(passed)})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="case directories after the solver run")
    parser.add_argument("--tolerances", type=Path, default=HERE / "tolerances.json")
    parser.add_argument("--json", type=Path, help="write all metrics and verdicts to this file")
    parser.add_argument("--allow-truncated", action="store_true",
                        help="report metrics of --max-steps smoke runs (never acceptance evidence)")
    args = parser.parse_args(argv)

    try:
        tolerances = json.loads(args.tolerances.read_text())
        analysed = [analyse_run(run) for run in args.runs]
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    truncated = [a["run"] for a in analysed if a["truncated"]]
    if truncated and not args.allow_truncated:
        print("ERROR: truncated smoke runs are not acceptance evidence: " + ", ".join(truncated),
              file=sys.stderr)
        return 2
    levels = {int(d): set(v) for d, v in tolerances["levels"]["cells_per_length"].items()}
    seen = [(a["dim"], a["level"]) for a in analysed]
    if len(seen) != len(set(seen)) or any(lv not in levels.get(d, set()) for d, lv in seen):
        print(f"ERROR: duplicate or unknown (dim, level) pairs {seen}", file=sys.stderr)
        return 2

    all_pass = True
    print(f"{'dim':>3} {'1/h':>4} {'periods':>7} {'|u|/U':>10} {'dp/(rgH)':>10} {'dy/H':>10}"
          f" {'dV/V':>10}  verdict")
    for a in sorted(analysed, key=lambda r: (r["dim"], r["level"])):
        a["criteria"] = evaluate_run(a, tolerances)
        a["passed"] = all(c["passed"] for c in a["criteria"])
        all_pass &= a["passed"]
        print(f"{a['dim']:>3} {a['level']:>4} {a['sloshing_periods_simulated']:>7.3g} "
              f"{a['max_speed_over_velocity_scale']:>10.2e} "
              f"{a['pressure_error_over_pressure_scale']:>10.2e} "
              f"{a['interface_height_drift_over_fill_height']:>10.2e} "
              f"{a['liquid_volume_relative_drift_max']:>10.2e}  "
              + ("PASS" if a["passed"] else "FAIL")
              + ("  [TRUNCATED SMOKE RUN]" if a["truncated"] else ""))
        for c in a["criteria"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']}: {c['value']:.3e} "
                  f"{'<=' if c['passed'] else '>'} {c['limit']:g}")
        log = a["solver_log"]
        if log:
            print(f"  solver log: {log['steps_logged']} steps, {log['nonconverged_steps']} not "
                  f"converged, {log['newton_iterations_total']} Newton iterations, "
                  f"{log['outer_passes_mean']:.2f} outer passes per step, max final residual "
                  f"{log['final_residual_max']:.2e}"
                  + (f", {log['wall_seconds']} s wall" if "wall_seconds" in log else ""))
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"], "runs": analysed,
                                         "passed": bool(all_pass)}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
