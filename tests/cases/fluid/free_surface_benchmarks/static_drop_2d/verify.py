#!/usr/bin/env python3
"""Compute the static_drop_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI).  Runs are
grouped by (capillary form, Laplace number); the criteria are applied to each
group across its resolution levels.  Metric definitions are in README.md.

Exit status: 0 if every criterion passes, 1 if any criterion fails, 2 if
input data are missing or invalid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRIANGLE = 5                      # VTK cell type
INTERIOR_RADIUS_FRACTION = 0.5    # pressure is averaged over |x - c| <= R_eff/2


class DataError(RuntimeError):
    """Missing or inconsistent input; verification cannot proceed."""


# ---------------------------------------------------------------------------
# Geometry of the P1 liquid region {phi_h < 0} on affine triangles
# ---------------------------------------------------------------------------
def _clip_negative(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Polygon {x in triangle : f_h(x) <= 0} for linear f_h (Sutherland-Hodgman)."""
    out = []
    for k in range(3):
        a, b = k, (k + 1) % 3
        fa, fb = f[a], f[b]
        if fa <= 0.0:
            out.append(p[a])
        if (fa < 0.0 < fb) or (fb < 0.0 < fa):
            s = fa / (fa - fb)
            out.append(p[a] + s * (p[b] - p[a]))
    return np.asarray(out)


def _polygon_area_centroid(q: np.ndarray) -> tuple[float, np.ndarray]:
    if len(q) < 3:
        return 0.0, np.zeros(2)
    x, y = q[:, 0], q[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum()
    if area == 0.0:
        return 0.0, np.zeros(2)
    cx = ((x + xn) * cross).sum() / (6.0 * area)
    cy = ((y + yn) * cross).sum() / (6.0 * area)
    return abs(area), np.array([cx, cy])


def liquid_area_centroid(points: np.ndarray, tris: np.ndarray, phi: np.ndarray):
    """Exact area and centroid of {phi_h < 0} for the P1 interpolant phi_h."""
    f = phi[tris]
    p = points[tris]
    full = np.all(f <= 0.0, axis=1)
    cut = ~full & np.any(f < 0.0, axis=1)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    tri_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    area = tri_area[full].sum()
    moment = (tri_area[full, None] * p[full].mean(axis=1)).sum(axis=0)
    for idx in np.nonzero(cut)[0]:
        a, c = _polygon_area_centroid(_clip_negative(p[idx], f[idx]))
        area += a
        moment += a * c
    if area <= 0.0:
        raise DataError("no liquid: phi >= 0 at every vertex")
    return float(area), moment / area


def interface_points(points: np.ndarray, tris: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Zero crossings of phi_h on the mesh edges (the LinearCorner polygon vertices)."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    fa, fb = phi[edges[:, 0]], phi[edges[:, 1]]
    crossing = (fa < 0.0) != (fb < 0.0)
    fa, fb, e = fa[crossing], fb[crossing], edges[crossing]
    s = fa / (fa - fb)
    return points[e[:, 0]] + s[:, None] * (points[e[:, 1]] - points[e[:, 0]])


def interior_mean_pressure(points, tris, pressure, centre, radius):
    """Area-weighted mean of the P1 pressure over triangles inside the disc."""
    inside = np.all(np.linalg.norm(points[tris] - centre, axis=2) <= radius, axis=1)
    if not np.any(inside):
        raise DataError("interior pressure region contains no triangle")
    p = points[tris[inside]]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    mean_p = pressure[tris[inside]].mean(axis=1)
    return float((area * mean_p).sum() / area.sum()), int(inside.sum())


def max_liquid_speed(phi: np.ndarray, velocity: np.ndarray) -> float:
    liquid = phi < 0.0
    if not np.any(liquid):
        raise DataError("no liquid vertex")
    return float(np.max(np.linalg.norm(velocity[liquid, :2], axis=1)))


def observed_order(levels, errors) -> float:
    """Least-squares slope of log(error) against log(R/h)."""
    x = np.log(np.asarray(levels, dtype=float))
    y = np.log(np.asarray(errors, dtype=float))
    return float(-np.polyfit(x, y, 1)[0])


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
    grid = _pyvista().read(path)
    types = np.asarray(grid.celltypes)
    if grid.n_cells == 0 or np.any(types != TRIANGLE):
        raise DataError(f"{path}: expected a pure Triangle3 mesh")
    tris = np.asarray(grid.cells).reshape(-1, 4)[:, 1:].astype(np.int64)
    if "GlobalElementID" in grid.cell_data:          # drop duplicated MPI cells
        _, first = np.unique(np.asarray(grid.cell_data["GlobalElementID"]), return_index=True)
        tris = tris[np.sort(first)]
    fields = {}
    for key in ("level_set_field", "velocity_field", "pressure_field"):
        name = case[key]
        if name not in grid.point_data:
            raise DataError(f"{path}: point array '{name}' is missing")
        fields[key] = np.asarray(grid.point_data[name], dtype=float)
    velocity = fields["velocity_field"].reshape(grid.n_points, -1)
    if velocity.shape[1] < 2:
        raise DataError(f"{path}: velocity must have at least two components")
    data = {"points": np.asarray(grid.points, dtype=float)[:, :2], "tris": tris,
            "phi": fields["level_set_field"], "velocity": velocity,
            "pressure": fields["pressure_field"]}
    for key, value in data.items():
        if key != "tris" and not np.all(np.isfinite(value)):
            raise DataError(f"{path}: non-finite values in {key}")
    return data


def output_series(run: Path, case: dict) -> list[tuple[float, Path]]:
    prefix = case.get("result_prefix", "result")
    pvd = run / f"{prefix}.pvd"
    if pvd.is_file():
        root = ET.parse(pvd).getroot()
        series = [(float(d.get("timestep")), run / d.get("file"))
                  for d in root.iter("DataSet")]
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


def analyse_run(run: Path) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    gamma, mu = case["surface_tension"], case["viscosity"]
    series = output_series(run, case)
    end_time = case["end_time"]
    if abs(series[-1][0] - end_time) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={end_time:.6g}")
    if len(series) < 4:
        raise DataError(f"{run}: need at least 4 outputs, found {len(series)}")

    initial = read_snapshot(run / "mesh" / "mesh-complete.mesh.vtu", case)
    area0, centre0 = liquid_area_centroid(initial["points"], initial["tris"], initial["phi"])

    times, speeds, areas = [], [], []
    for t, path in series:
        snap = read_snapshot(path, case)
        area, _ = liquid_area_centroid(snap["points"], snap["tris"], snap["phi"])
        times.append(t)
        speeds.append(max_liquid_speed(snap["phi"], snap["velocity"]))
        areas.append(area)
    times, speeds, areas = map(np.asarray, (times, speeds, areas))

    final = snap
    area_f, centre_f = liquid_area_centroid(final["points"], final["tris"], final["phi"])
    r_eff = math.sqrt(area_f / math.pi)
    p_in, n_region = interior_mean_pressure(final["points"], final["tris"], final["pressure"],
                                            centre_f, INTERIOR_RADIUS_FRACTION * r_eff)
    jump = p_in - case["external_pressure"]
    ref_eff, ref_nom = gamma / r_eff, gamma / case["radius"]
    iface = interface_points(final["points"], final["tris"], final["phi"])
    radial = np.linalg.norm(iface - centre_f, axis=1) - r_eff

    t_end = times[-1]
    third = (times >= 0.5 * t_end) & (times <= 0.75 * t_end)
    fourth = times > 0.75 * t_end
    if not np.any(third) or not np.any(fourth):
        raise DataError(f"{run}: outputs do not cover the last half of the run")
    capillary = mu * speeds / gamma
    return {
        "run": str(run),
        "level": case["level_R_over_h"],
        "capillary_form": case["capillary_form"],
        "laplace_number": case["laplace_number"],
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(t_end),
        "viscous_times_simulated": float(t_end / case["viscous_time"]),
        "outputs": len(times),
        "pressure_jump": jump,
        "pressure_jump_reference": ref_eff,
        "pressure_jump_relative_error": abs(jump - ref_eff) / ref_eff,
        "pressure_jump_relative_error_nominal_radius": abs(jump - ref_nom) / ref_nom,
        "pressure_region_triangles": n_region,
        "effective_radius": r_eff,
        "effective_radius_relative_to_nominal": r_eff / case["radius"] - 1.0,
        "initial_liquid_area": area0,
        "final_liquid_area": area_f,
        "liquid_area_relative_drift_final": abs(area_f - area0) / area0,
        "liquid_area_relative_drift_max": float(np.max(np.abs(areas - area0)) / area0),
        "centroid_drift_over_radius": float(np.linalg.norm(centre_f - centre0) / case["radius"]),
        "shape_max_radial_deviation": float(np.max(np.abs(radial)) / r_eff),
        "shape_rms_radial_deviation": float(np.sqrt(np.mean(radial ** 2)) / r_eff),
        "max_speed_final": float(speeds[-1]),
        "parasitic_capillary_number_end": float(capillary[-1]),
        "parasitic_capillary_number_final": float(np.max(capillary[fourth])),
        "speed_per_surface_tension_final": float(np.max(speeds[fourth]) / gamma),
        "max_speed_growth_ratio": float(speeds[-1] / np.max(speeds[third])),
        "history": {"time": times.tolist(), "max_speed": speeds.tolist(),
                    "parasitic_capillary_number": capillary.tolist(),
                    "liquid_area": areas.tolist()},
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def evaluate_group(runs: list[dict], tolerances: dict) -> list[dict]:
    by_level = {r["level"]: r for r in runs}
    results = []
    for crit in tolerances["criteria"]:
        q, limit = crit["quantity"], crit.get("limit")
        messages, ok = [], True
        at = crit.get("at_level", "each")
        levels = sorted(by_level) if at == "each" else [at]
        for level in levels:
            if level not in by_level:
                ok = False
                messages.append(f"missing run at R/h={level}")
                continue
            value = by_level[level][q]
            if limit is None:
                messages.append(f"R/h={level}: {value:.4g} (reported)")
                continue
            passed = value <= limit
            ok &= passed
            messages.append(f"R/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
        if "minimum_observed_order" in crit:
            need = crit["order_levels"]
            missing = [lv for lv in need if lv not in by_level]
            if missing:
                ok = False
                messages.append(f"order needs R/h={missing}")
            else:
                errs = [by_level[lv][q] for lv in need]
                if min(errs) <= 0.0:
                    ok = False
                    messages.append("order undefined for a zero error")
                else:
                    order = observed_order(need, errs)
                    passed = order >= crit["minimum_observed_order"]
                    ok &= passed
                    pairs = ", ".join(f"{observed_order(need[i:i + 2], errs[i:i + 2]):.2f}"
                                      for i in range(len(need) - 1))
                    messages.append(f"observed order {order:.2f} (pairwise {pairs}) "
                                    f"{'>=' if passed else '<'} {crit['minimum_observed_order']}")
        if crit.get("monotone") == "strictly_decreasing":
            need = crit["monotone_levels"]
            missing = [lv for lv in need if lv not in by_level]
            if missing:
                ok = False
                messages.append(f"monotonicity needs R/h={missing}")
            else:
                seq = sorted(by_level)
                vals = [by_level[lv][q] for lv in seq]
                passed = all(b < a for a, b in zip(vals, vals[1:]))
                ok &= passed
                messages.append("decreasing over R/h=" + "/".join(map(str, seq)) +
                                (": yes" if passed else ": NO (" +
                                 ", ".join(f"{v:.3g}" for v in vals) + ")"))
        results.append({"id": crit["id"], "quantity": q, "passed": bool(ok),
                        "details": messages})
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

    groups: dict[tuple, list] = {}
    for a in analysed:
        groups.setdefault((a["capillary_form"], a["laplace_number"]), []).append(a)
    levels = set(tolerances["levels"]["R_over_h"])
    report, all_pass = [], True
    for (form, laplace), runs in sorted(groups.items()):
        seen = [r["level"] for r in runs]
        if len(seen) != len(set(seen)) or not set(seen) <= levels:
            print(f"ERROR: group {form}, La={laplace:g}: duplicate or unknown levels {seen}",
                  file=sys.stderr)
            return 2
        runs.sort(key=lambda r: r["level"])
        verdicts = evaluate_group(runs, tolerances)
        all_pass &= all(v["passed"] for v in verdicts)
        print(f"\n== {form}, La = {laplace:g}" + ("  [TRUNCATED SMOKE RUNS]" if any(
            r["truncated"] for r in runs) else ""))
        print(f"{'R/h':>4} {'t/t_mu':>7} {'dp/(g/R)-1':>11} {'R_eff/R-1':>10} {'Ca_final':>10}"
              f" {'growth':>7} {'dA/A max':>9} {'shape max':>9}")
        for r in runs:
            print(f"{r['level']:>4} {r['viscous_times_simulated']:>7.3g} "
                  f"{r['pressure_jump'] / r['pressure_jump_reference'] - 1:>11.3e} "
                  f"{r['effective_radius_relative_to_nominal']:>10.3e} "
                  f"{r['parasitic_capillary_number_final']:>10.3e} "
                  f"{r['max_speed_growth_ratio']:>7.3f} {r['liquid_area_relative_drift_max']:>9.2e} "
                  f"{r['shape_max_radial_deviation']:>9.2e}")
        for v in verdicts:
            print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
        report.append({"capillary_form": form, "laplace_number": laplace,
                       "runs": runs, "criteria": verdicts,
                       "passed": all(v["passed"] for v in verdicts)})
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"],
                                         "groups": report, "passed": bool(all_pass)},
                                        indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
