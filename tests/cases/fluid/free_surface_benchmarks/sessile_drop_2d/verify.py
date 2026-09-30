#!/usr/bin/env python3
"""Compute the sessile_drop_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI).  Runs are
grouped by (capillary form, equilibrium angle); the criteria are applied to
each group across its resolution levels.  Metric definitions are in
README.md.

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
# Contact angles are read from a circle fitted to the interface points of one
# side with 0 <= y <= ANGLE_FIT_HEIGHT * R_ref.  The window is fixed once for
# all cases; on sampled exact caps it measures the angle to within 0.13
# degrees at R/h = 16 and 0.05 degrees at R/h = 32 (see README.md).
ANGLE_FIT_HEIGHT = 0.25
# The case runs with a fixed step, so a complete run has its last output at
# the protocol end time, to within half a step.
WALL_TOLERANCE = 1.0e-9           # |y - y_wall| for vertices on the contact wall
POINT_MERGE_DIGITS = 12           # coincident output points (MPI pieces) are merged


class DataError(RuntimeError):
    """Missing or inconsistent input; verification cannot proceed."""


# ---------------------------------------------------------------------------
# Equilibrium reference
# ---------------------------------------------------------------------------
def cap_area_factor(theta: float) -> float:
    return theta - math.sin(theta) * math.cos(theta)


def equilibrium_cap(area: float, theta: float) -> dict:
    """Circular cap of the given area with contact angle theta on y = 0."""
    radius = math.sqrt(area / cap_area_factor(theta))
    return {"radius": radius,
            "base_half_width": radius * math.sin(theta),
            "apex_height": radius * (1.0 - math.cos(theta))}


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


def wall_contact_points(points: np.ndarray, tris: np.ndarray, phi: np.ndarray,
                        wall_y: float) -> tuple[np.ndarray, list[int]]:
    """Zero crossings of phi_h along the contact wall, left to right.

    Returns the crossing abscissae and, for each, the triangle holding that
    wall edge.
    """
    on_wall = np.abs(points[:, 1] - wall_y) <= WALL_TOLERANCE
    crossings, parents = [], []
    for t, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if not (on_wall[a] and on_wall[b]):
                continue
            fa, fb = phi[a], phi[b]
            if (fa < 0.0) == (fb < 0.0):
                continue
            s = fa / (fa - fb)
            crossings.append(points[a, 0] + s * (points[b, 0] - points[a, 0]))
            parents.append(t)
    order = np.argsort(crossings)
    return np.asarray(crossings)[order], [parents[i] for i in order]


def cell_contact_angle(points: np.ndarray, tri: np.ndarray, phi: np.ndarray) -> float:
    """Through-liquid angle of the P1 interface in one wall triangle (degrees).

    The liquid is phi < 0, so n = grad(phi)/|grad(phi)| points out of the
    liquid.  With the wall normal (0, -1): cos(theta) = -n . n_wall = n_y.
    """
    p = points[tri]
    m = np.column_stack([p[1] - p[0], p[2] - p[0]]).T
    g = np.linalg.solve(m, phi[tri[1:]] - phi[tri[0]])
    return math.degrees(math.acos(float(np.clip(g[1] / np.linalg.norm(g), -1.0, 1.0))))


def fit_circle(q: np.ndarray) -> tuple[float, float, float]:
    """Geometric least-squares circle (Kasa start, Gauss-Newton refinement)."""
    if len(q) < 3:
        raise DataError("a circle fit needs at least three interface points")
    x, y = q[:, 0], q[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    xc, yc = 0.5 * c[0], 0.5 * c[1]
    r = math.sqrt(max(c[2] + xc * xc + yc * yc, 0.0))
    for _ in range(100):
        d = np.hypot(x - xc, y - yc)
        if np.any(d == 0.0):
            break
        jac = np.column_stack([-(x - xc) / d, -(y - yc) / d, -np.ones_like(x)])
        step, *_ = np.linalg.lstsq(jac, -(d - r), rcond=None)
        xc, yc, r = xc + step[0], yc + step[1], r + step[2]
        if np.linalg.norm(step) <= 1.0e-14 * max(1.0, abs(r)):
            break
    if not all(math.isfinite(v) for v in (xc, yc, r)) or r <= 0.0:
        raise DataError("interface circle fit failed")
    return float(xc), float(yc), float(abs(r))


def local_contact_angle(iface: np.ndarray, contact_x: float, middle_x: float, side: int,
                        wall_y: float, window: float) -> tuple[float, int]:
    """Contact angle (degrees) of one side from a local circle fit.

    side = -1 (left) or +1 (right).  The circle is fitted to the interface
    points of that side with wall_y <= y <= wall_y + window.  The angle is
    measured through the liquid between the wall direction pointing into the
    liquid and the circle tangent, oriented away from the wall, at the circle
    point closest to the measured contact point.
    """
    select = ((iface[:, 1] - wall_y) <= window) & ((iface[:, 0] - middle_x) * side > 0.0)
    q = iface[select]
    xc, yc, r = fit_circle(q)
    contact = np.array([contact_x, wall_y])
    centre = np.array([xc, yc])
    radial = contact - centre
    radial /= np.linalg.norm(radial)
    tangent = np.array([-radial[1], radial[0]])
    if tangent[1] < 0.0:
        tangent = -tangent
    into_liquid = np.array([-float(side), 0.0])
    return math.degrees(math.acos(float(np.clip(into_liquid @ tangent, -1.0, 1.0)))), len(q)


def global_circle_angle(iface: np.ndarray, wall_y: float) -> tuple[float, float]:
    """Angle and radius of one circle fitted to the whole interface."""
    xc, yc, r = fit_circle(iface)
    return math.degrees(math.acos(float(np.clip(-(yc - wall_y) / r, -1.0, 1.0)))), r


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


def drop_geometry(snap: dict, case: dict, reference_radius: float) -> dict:
    """Contact points, base half-width, apex height and contact angles."""
    wall_y = case["contact_wall_y"]
    points, tris, phi = snap["points"], snap["tris"], snap["phi"]
    xs, parents = wall_contact_points(points, tris, phi, wall_y)
    if len(xs) != 2:
        raise DataError(f"expected two wall contact points, found {len(xs)}")
    iface = interface_points(points, tris, phi)
    middle = 0.5 * (xs[0] + xs[1])
    window = ANGLE_FIT_HEIGHT * reference_radius
    left, n_left = local_contact_angle(iface, xs[0], middle, -1, wall_y, window)
    right, n_right = local_contact_angle(iface, xs[1], middle, +1, wall_y, window)
    fitted_angle, fitted_radius = global_circle_angle(iface, wall_y)
    base = 0.5 * (xs[1] - xs[0])
    apex = float(np.max(iface[:, 1]) - wall_y)
    return {
        "contact_x": xs.tolist(),
        "base_half_width": float(base),
        "apex_height": apex,
        "contact_angle_left": left,
        "contact_angle_right": right,
        "angle_fit_points": [n_left, n_right],
        "cell_contact_angle_left": cell_contact_angle(points, tris[parents[0]], phi),
        "cell_contact_angle_right": cell_contact_angle(points, tris[parents[1]], phi),
        "height_base_angle": math.degrees(2.0 * math.atan2(apex, base)),
        "circle_fit_angle": fitted_angle,
        "circle_fit_radius": fitted_radius,
    }


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------
def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:     # pragma: no cover - environment dependent
        raise DataError("pyvista is required to read VTU/PVTU output") from exc
    return pv


def _merge_coincident_points(points: np.ndarray, tris: np.ndarray, fields: dict):
    """Merge points that coincide (duplicated across MPI pieces)."""
    keys = np.round(points, POINT_MERGE_DIGITS)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    if len(first) == len(points):
        return points, tris, fields
    return (points[first], inverse[tris],
            {k: v[first] for k, v in fields.items()})


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
    fields["velocity_field"] = fields["velocity_field"].reshape(grid.n_points, -1)
    if fields["velocity_field"].shape[1] < 2:
        raise DataError(f"{path}: velocity must have at least two components")
    points = np.asarray(grid.points, dtype=float)[:, :2]
    points, tris, fields = _merge_coincident_points(points, tris, fields)
    data = {"points": points, "tris": tris, "phi": fields["level_set_field"],
            "velocity": fields["velocity_field"], "pressure": fields["pressure_field"]}
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
    theta_e = case["equilibrium_angle_degrees"]
    series = output_series(run, case)
    end_time = case["end_time"]
    if abs(series[-1][0] - end_time) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={end_time:.6g}")
    if len(series) < 4:
        raise DataError(f"{run}: need at least 4 outputs, found {len(series)}")

    initial = read_snapshot(run / "mesh" / "mesh-complete.mesh.vtu", case)
    area0, _ = liquid_area_centroid(initial["points"], initial["tris"], initial["phi"])
    reference = equilibrium_cap(area0, math.radians(theta_e))
    nominal = case["equilibrium_cap_nominal"]

    times, speeds, areas, bases, left, right = [], [], [], [], [], []
    for t, path in series:
        snap = read_snapshot(path, case)
        area, _ = liquid_area_centroid(snap["points"], snap["tris"], snap["phi"])
        times.append(t)
        speeds.append(max_liquid_speed(snap["phi"], snap["velocity"]))
        areas.append(area)
        try:
            g = drop_geometry(snap, case, reference["radius"])
            bases.append(g["base_half_width"])
            left.append(g["contact_angle_left"])
            right.append(g["contact_angle_right"])
        except DataError:
            bases.append(math.nan)
            left.append(math.nan)
            right.append(math.nan)
    times, speeds, areas = map(np.asarray, (times, speeds, areas))
    bases, left, right = map(np.asarray, (bases, left, right))

    final = drop_geometry(snap, case, reference["radius"])
    # Decision D10: the end state is a static equilibrium whose discrete form
    # does not depend on dt; dt enters only through the area gained or lost
    # in transport.  The cap with the final area removes that contribution.
    final_area_cap = equilibrium_cap(float(areas[-1]), math.radians(theta_e))
    angle_errors = [abs(final["contact_angle_left"] - theta_e),
                    abs(final["contact_angle_right"] - theta_e)]
    t_end = times[-1]
    third = (times >= 0.5 * t_end) & (times <= 0.75 * t_end)
    fourth = times > 0.75 * t_end
    if not np.any(third) or not np.any(fourth):
        raise DataError(f"{run}: outputs do not cover the last half of the run")
    base_three_quarter = bases[np.argmin(np.abs(times - 0.75 * t_end))]
    capillary = mu * speeds / gamma
    return {
        "run": str(run),
        "level": case["level_R_over_h"],
        "capillary_form": case["capillary_form"],
        "equilibrium_angle_degrees": theta_e,
        "initial_angle_degrees": case["initial_angle_degrees"],
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(t_end),
        "viscous_times_simulated": float(t_end / case["viscous_time"]),
        "outputs": len(times),
        **{k: final[k] for k in ("contact_x", "base_half_width", "apex_height",
                                 "contact_angle_left", "contact_angle_right",
                                 "angle_fit_points", "cell_contact_angle_left",
                                 "cell_contact_angle_right", "height_base_angle",
                                 "circle_fit_angle", "circle_fit_radius")},
        "contact_angle_error_degrees": float(max(angle_errors)),
        "contact_angle_asymmetry_degrees": float(abs(final["contact_angle_left"]
                                                     - final["contact_angle_right"])),
        "reference_radius": reference["radius"],
        "reference_base_half_width": reference["base_half_width"],
        "reference_apex_height": reference["apex_height"],
        "base_radius_relative_error": abs(final["base_half_width"] - reference["base_half_width"])
                                      / reference["base_half_width"],
        "apex_height_relative_error": abs(final["apex_height"] - reference["apex_height"])
                                      / reference["apex_height"],
        "base_radius_relative_error_nominal": abs(final["base_half_width"]
                                                  - nominal["base_half_width"])
                                              / nominal["base_half_width"],
        "apex_height_relative_error_nominal": abs(final["apex_height"] - nominal["apex_height"])
                                              / nominal["apex_height"],
        "base_radius_relative_error_final_area": abs(final["base_half_width"]
                                                     - final_area_cap["base_half_width"])
                                                 / final_area_cap["base_half_width"],
        "apex_height_relative_error_final_area": abs(final["apex_height"]
                                                     - final_area_cap["apex_height"])
                                                 / final_area_cap["apex_height"],
        "base_change_last_quarter": float(abs(bases[-1] - base_three_quarter)
                                          / reference["base_half_width"]),
        "initial_liquid_area": area0,
        "final_liquid_area": float(areas[-1]),
        "liquid_area_relative_drift_final": float(abs(areas[-1] - area0) / area0),
        "liquid_area_relative_drift_max": float(np.max(np.abs(areas - area0)) / area0),
        "max_speed_final": float(speeds[-1]),
        "capillary_number_final": float(np.max(capillary[fourth])),
        "max_speed_growth_ratio": float(speeds[-1] / np.max(speeds[third])),
        "history": {"time": times.tolist(), "max_speed": speeds.tolist(),
                    "liquid_area": areas.tolist(), "base_half_width": bases.tolist(),
                    "contact_angle_left": left.tolist(),
                    "contact_angle_right": right.tolist()},
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
                if passed and min(vals) > 0.0:
                    messages.append(f"observed order {observed_order(seq, vals):.2f} (reported)")
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
        groups.setdefault((a["capillary_form"], a["equilibrium_angle_degrees"]), []).append(a)
    levels = set(tolerances["levels"]["R_over_h"])
    report, all_pass = [], True
    for (form, theta_e), runs in sorted(groups.items()):
        seen = [r["level"] for r in runs]
        if len(seen) != len(set(seen)) or not set(seen) <= levels:
            print(f"ERROR: group {form}, theta_e={theta_e:g}: duplicate or unknown levels {seen}",
                  file=sys.stderr)
            return 2
        runs.sort(key=lambda r: r["level"])
        verdicts = evaluate_group(runs, tolerances)
        all_pass &= all(v["passed"] for v in verdicts)
        print(f"\n== {form}, theta_e = {theta_e:g} deg (initial {runs[0]['initial_angle_degrees']:g})"
              + ("  [TRUNCATED SMOKE RUNS]" if any(r["truncated"] for r in runs) else ""))
        print(f"{'R/h':>4} {'t/t_mu':>7} {'theta_L':>8} {'theta_R':>8} {'err':>6} {'b err':>9}"
              f" {'H err':>9} {'dA/A max':>9} {'growth':>7} {'Ca_final':>9}")
        for r in runs:
            print(f"{r['level']:>4} {r['viscous_times_simulated']:>7.3g} "
                  f"{r['contact_angle_left']:>8.3f} {r['contact_angle_right']:>8.3f} "
                  f"{r['contact_angle_error_degrees']:>6.3f} "
                  f"{r['base_radius_relative_error']:>9.2e} {r['apex_height_relative_error']:>9.2e} "
                  f"{r['liquid_area_relative_drift_max']:>9.2e} {r['max_speed_growth_ratio']:>7.3f} "
                  f"{r['capillary_number_final']:>9.2e}")
        for v in verdicts:
            print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
        report.append({"capillary_form": form, "equilibrium_angle_degrees": theta_e,
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
