#!/usr/bin/env python3
"""Compute the sessile_drop_3d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT] [--allow-truncated]
    verify.py --symmetry-check QUARTER_RUN FULL_RUN [--json OUT]

Each RUN_DIR is a case written by generate_case.py in which the solver has
run.  Runs of one (domain, dt rule, dt divisor) form a refinement study; the
criteria are applied across its levels.  The contact angle is measured along
the whole contact line: the line is divided into azimuthal sectors of
SECTOR_DEGREES around the drop axis, and in each sector a circle is fitted to
the meridional projection (r, y) of the interface points near the wall, as
the 2D benchmark fits each side (README.md).  --symmetry-check compares a
quarter-domain run with a full-domain run of the same level on the shared
quarter.

Exit status: 0 if every criterion passes, 1 if any criterion fails, 2 if
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
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}_for_sessile_3d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SV = _load("static_sphere_3d", "verify")      # Tet4 reader, exact liquid volume, edge crossings
S2 = _load("sessile_drop_2d", "verify")       # circle fit and local contact angle of one side
GEN = _load("sessile_drop_3d", "generate_case")
DataError = SV.DataError
liquid_volume_centroid = SV.liquid_volume_centroid
interface_points = SV.interface_points
observed_order = SV.observed_order
output_series = SV.output_series

SECTOR_DEGREES = 30.0              # azimuthal width of one contact-angle sector
ANGLE_FIT_HEIGHT = S2.ANGLE_FIT_HEIGHT   # fit window 0 <= y <= 0.25 R_ref, as in 2D
WALL_TOLERANCE = 1.0e-9
POINT_MERGE_DIGITS = 12
LOCAL_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


# ---------------------------------------------------------------------------
# Equilibrium reference and geometry
# ---------------------------------------------------------------------------
def equilibrium_cap(drop_volume: float, theta_deg: float) -> dict:
    theta = math.radians(theta_deg)
    r = GEN.cap_radius_for_volume(drop_volume, theta)
    return GEN.cap_geometry(r, theta)


def wall_contact_segments(points, tets, phi, wall_y):
    """Zero segments of phi_h on the wall faces of the tetrahedra.

    Returns the segment end points (S, 2, 3) and the parent tetrahedron of
    each segment.  A wall face is a tetrahedron face whose three vertices lie
    on y = wall_y; each face with vertices on both sides of zero carries one
    segment of the P1 contact line.
    """
    on_wall = np.abs(points[:, 1] - wall_y) <= WALL_TOLERANCE
    segments, parents = [], []
    for lf in LOCAL_FACES:
        faces = tets[:, lf]
        mask = np.all(on_wall[faces], axis=1)
        for t in np.nonzero(mask)[0]:
            tri = tets[t, list(lf)]
            f = phi[tri]
            if np.all(f < 0.0) or np.all(f >= 0.0):
                continue
            ends = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if (f[a] < 0.0) != (f[b] < 0.0):
                    s = f[a] / (f[a] - f[b])
                    ends.append(points[tri[a]] + s * (points[tri[b]] - points[tri[a]]))
            if len(ends) == 2:
                segments.append(ends)
                parents.append(t)
    if not segments:
        raise DataError("no contact line: phi_h does not change sign on the contact wall")
    return np.asarray(segments), np.asarray(parents, dtype=np.int64)


def cell_angle(points, tet, phi) -> float:
    """Through-liquid angle of the P1 interface in one tetrahedron (degrees):
    n = grad(phi)/|grad(phi)| points out of the liquid; with the wall normal
    (0, -1, 0), cos(theta) = n_y."""
    p = points[tet]
    g = np.linalg.solve((p[1:] - p[0]), phi[tet[1:]] - phi[tet[0]])
    return math.degrees(math.acos(float(np.clip(g[1] / np.linalg.norm(g), -1.0, 1.0))))


def fit_axis(contact: np.ndarray) -> np.ndarray:
    """Centre (x, z) of the circle fitted to the contact points (full domain)."""
    xc, zc, _ = S2.fit_circle(contact[:, [0, 2]])
    return np.array([xc, zc])


def drop_geometry(snap: dict, case: dict, reference_radius: float, theta_e: float) -> dict:
    wall_y = case["contact_wall_y"]
    points, tets, phi = snap["points"], snap["tets"], snap["phi"]
    segments, parents = wall_contact_segments(points, tets, phi, wall_y)
    contact = np.unique(np.round(segments.reshape(-1, 3), POINT_MERGE_DIGITS), axis=0)
    axis = (np.asarray(case["drop_axis"], dtype=float) if case["domain"] == "quarter"
            else fit_axis(contact))
    radial = np.hypot(contact[:, 0] - axis[0], contact[:, 2] - axis[1])
    iface = interface_points(points, tets, phi)
    apex = float(np.max(iface[:, 1]) - wall_y)

    # Sector angles: azimuth about the axis; quarter domain covers [0, 90] degrees.
    def azimuth(q):
        return np.degrees(np.arctan2(q[:, 2] - axis[1], q[:, 0] - axis[0]))
    span = (0.0, 90.0) if case["domain"] == "quarter" else (-180.0, 180.0)
    edges = np.arange(span[0], span[1] + 0.5 * SECTOR_DEGREES, SECTOR_DEGREES)
    az_iface, az_contact = azimuth(iface), azimuth(contact)
    window = ANGLE_FIT_HEIGHT * reference_radius
    sectors = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_c = (az_contact >= lo) & (az_contact < hi) if hi < span[1] else (az_contact >= lo)
        in_i = (az_iface >= lo) & (az_iface < hi) if hi < span[1] else (az_iface >= lo)
        if not np.any(in_c):
            raise DataError(f"no contact point in the sector [{lo:g}, {hi:g}) degrees")
        q = iface[in_i]
        meridional = np.column_stack([np.hypot(q[:, 0] - axis[0], q[:, 2] - axis[1]),
                                      q[:, 1]])
        r_contact = float(np.mean(radial[in_c]))
        angle, n_fit = S2.local_contact_angle(meridional, r_contact, 0.0, +1, wall_y, window)
        sectors.append({"azimuth_degrees": [float(lo), float(hi)], "angle": angle,
                        "fit_points": n_fit, "contact_radius": r_contact})
    angles = np.array([s["angle"] for s in sectors])
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    cell = np.array([cell_angle(points, tets[t], phi) for t in parents])
    return {
        "axis": axis.tolist(),
        "base_radius": float(np.mean(radial)),
        "base_radius_spread": float(np.max(radial) - np.min(radial)),
        "apex_height": apex,
        "contact_points": int(len(contact)),
        "sectors": sectors,
        "contact_angle_mean": float(np.mean(angles)),
        "contact_angle_max_deviation": float(np.max(np.abs(angles - theta_e))),
        "cell_angle_mean": float(np.sum(lengths * cell) / np.sum(lengths)),
        "cell_angle_max_deviation": float(np.max(np.abs(cell - theta_e))),
        "contact_line_length": float(np.sum(lengths)),
    }


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------
def read_snapshot(path: Path, case: dict) -> dict:
    snap = SV.read_snapshot(path, case)
    keys = np.round(snap["points"], POINT_MERGE_DIGITS)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    if len(first) != len(snap["points"]):          # points duplicated across MPI pieces
        inverse = inverse.ravel()
        snap = {"points": snap["points"][first], "tets": inverse[snap["tets"]],
                "phi": snap["phi"][first], "velocity": snap["velocity"][first],
                "pressure": snap["pressure"][first]}
    return snap


def analyse_run(run: Path, *, allow_truncated: bool = False) -> dict:
    case_path = run / "case.json"
    if not case_path.is_file():
        raise DataError(f"{run}: no case.json")
    case = json.loads(case_path.read_text())
    if case.get("benchmark") != "sessile_drop_3d":
        raise DataError(f"{run}: not a sessile_drop_3d case")
    if case["truncated"] and not allow_truncated:
        raise DataError(f"{run}: truncated smoke run (use --allow-truncated for diagnostics)")
    series = output_series(run, case)
    if not case["truncated"] and abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run (last output t = {series[-1][0]:g}, "
                        f"end time {case['end_time']:g})")
    initial = read_snapshot(run / "mesh" / "mesh-complete.mesh.vtu", case)
    v0, _ = liquid_volume_centroid(initial["points"], initial["tets"], initial["phi"])
    fraction = case["symmetry_fraction"]
    theta_e = case["equilibrium_angle_degrees"]
    reference = equilibrium_cap(v0 / fraction, theta_e)
    times, volumes, speeds, history = [], [], [], []
    snap = None
    for t, path in series:
        snap = read_snapshot(path, case)
        v, _ = liquid_volume_centroid(snap["points"], snap["tets"], snap["phi"])
        times.append(t)
        volumes.append(v)
        speeds.append(SV.max_liquid_speed(snap["phi"], snap["velocity"]))
        g = drop_geometry(snap, case, reference["radius"], theta_e)
        history.append({"time": t, "base_radius": g["base_radius"],
                        "contact_angle_mean": g["contact_angle_mean"],
                        "contact_angle_max_deviation": g["contact_angle_max_deviation"]})
    final = drop_geometry(snap, case, reference["radius"], theta_e)
    times, speeds = np.asarray(times), np.asarray(speeds)
    t_end = times[-1]
    third = (times >= 0.5 * t_end) & (times <= 0.75 * t_end)
    growth = float(speeds[-1] / np.max(speeds[third])) if np.any(third) else None
    return {
        "run": str(run),
        "level": case["level_R_over_h"],
        "domain": case["domain"],
        "dt_rule": case["dt_rule"],
        "dt_divisor": case["dt_divisor"],
        "dt": case["dt"],
        "truncated": case["truncated"],
        "equilibrium_angle_degrees": theta_e,
        "reference": reference,
        "initial_liquid_volume": v0,
        "final": final,
        "contact_angle_mean_error_degrees": abs(final["contact_angle_mean"] - theta_e),
        "contact_angle_max_error_degrees": final["contact_angle_max_deviation"],
        "base_radius_relative_error": abs(final["base_radius"] - reference["base_radius"])
        / reference["base_radius"],
        "apex_height_relative_error": abs(final["apex_height"] - reference["apex_height"])
        / reference["apex_height"],
        "liquid_volume_relative_drift_max": float(np.max(np.abs(np.asarray(volumes) - v0)) / v0),
        "max_speed_growth_ratio": growth,
        "max_speed_final": float(speeds[-1]),
        "times": times.tolist(),
        "max_speed": speeds.tolist(),
        "liquid_volume": volumes,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def evaluate_group(runs: list[dict], tolerances: dict, levels_required=None) -> list[dict]:
    by_level = {r["level"]: r for r in runs}
    results = []
    for crit in tolerances["criteria"]:
        q, limit, ok, messages = crit["quantity"], crit["limit"], True, []
        at = crit["at_level"]
        levels = sorted(by_level) if at == "each" else [at]
        if levels_required is not None and at != "each" and at not in levels_required:
            results.append({"id": crit["id"], "passed": True, "evaluated": False,
                            "details": [f"not evaluated (needs R/h = {at})"]})
            continue
        for level in levels:
            if level not in by_level:
                ok = False
                messages.append(f"missing run at R/h={level}")
                continue
            value = by_level[level][q]
            if value is None:
                ok = False
                messages.append(f"R/h={level}: {q} unavailable")
                continue
            passed = value <= limit
            ok &= passed
            messages.append(f"R/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
        mono = crit.get("monotone_levels")
        if crit.get("monotone") == "strictly_decreasing" and mono:
            if all(lv in by_level for lv in mono):
                values = [by_level[lv][q] for lv in mono]
                passed = all(b < a for a, b in zip(values, values[1:]))
                ok &= passed
                messages.append("decreasing over R/h=" + "/".join(map(str, mono)) +
                                (": yes" if passed else ": NO (" +
                                 ", ".join(f"{v:.3g}" for v in values) + ")"))
            elif levels_required is None:
                ok = False
                messages.append("monotonicity needs R/h=" + "/".join(map(str, mono)))
        results.append({"id": crit["id"], "passed": bool(ok), "evaluated": True,
                        "details": messages})
    return results


def time_step_changes(coarse: dict, fine: dict) -> dict:
    a, b = coarse["final"], fine["final"]
    return {
        "contact_angle_change_final": max(
            abs(sa["angle"] - sb["angle"]) for sa, sb in zip(a["sectors"], b["sectors"])),
        "base_radius_change_final": abs(a["base_radius"] - b["base_radius"]) / b["base_radius"],
        "apex_height_change_final": abs(a["apex_height"] - b["apex_height"]) / b["apex_height"],
    }


def evaluate_time_step(groups: dict, tolerances: dict) -> list[dict]:
    crit = tolerances["time_step_criterion"]
    out = []
    for key, runs in groups.items():
        domain, rule, divisor = key
        if divisor != 1 or (domain, rule, 2) not in groups:
            continue
        coarse = {r["level"]: r for r in runs}
        fine = {r["level"]: r for r in groups[(domain, rule, 2)]}
        common = sorted(set(coarse) & set(fine))
        if not common:
            continue
        level = common[-1]
        changes = time_step_changes(coarse[level], fine[level])
        details, ok = [], True
        for q in crit["quantities"]:
            value = changes[q["id"]]
            passed = value <= q["limit"]
            ok &= passed
            details.append(f"{q['id']}: {value:.3g} {'<=' if passed else '>'} {q['limit']:g}")
        out.append({"id": "time_step", "group": list(key), "level": level, "passed": bool(ok),
                    "changes": changes, "details": details})
    return out


# ---------------------------------------------------------------------------
# Symmetry check: quarter domain against the full domain on the shared quarter
# ---------------------------------------------------------------------------
def symmetry_check(quarter_run: Path, full_run: Path) -> dict:
    cq = json.loads((quarter_run / "case.json").read_text())
    cf = json.loads((full_run / "case.json").read_text())
    if cq["domain"] != "quarter" or cf["domain"] != "full" or \
            cq["level_R_over_h"] != cf["level_R_over_h"]:
        raise DataError("--symmetry-check needs a quarter and a full run of the same level")
    sq, sf = output_series(quarter_run, cq), output_series(full_run, cf)
    times_f = {round(t / cf["dt"]): p for t, p in sf}
    theta_e = cq["equilibrium_angle_degrees"]
    rows = []
    for t, pq in sq:
        key = round(t / cq["dt"])
        if key not in times_f:
            continue
        a, b = read_snapshot(pq, cq), read_snapshot(times_f[key], cf)
        # Shared quarter: map the quarter vertices onto the full-domain vertices.
        index = {tuple(k): i for i, k in enumerate(np.round(b["points"], POINT_MERGE_DIGITS))}
        mapping = np.array([index[tuple(k)] for k in np.round(a["points"], POINT_MERGE_DIGITS)])
        du = np.linalg.norm(a["velocity"][:, :3] - b["velocity"][mapping, :3], axis=1)
        umax = max(float(np.max(np.linalg.norm(b["velocity"][:, :3], axis=1))), 1e-300)
        vq, _ = liquid_volume_centroid(a["points"], a["tets"], a["phi"])
        vf, _ = liquid_volume_centroid(b["points"], b["tets"], b["phi"])
        # The full-domain solution restricted to the shared quarter, measured
        # exactly as the quarter run (same cells, axis and sectors).
        corner = np.all((b["points"][b["tets"]][:, :, 0] >= -WALL_TOLERANCE) &
                        (b["points"][b["tets"]][:, :, 2] >= -WALL_TOLERANCE), axis=1)
        bq = dict(b, tets=b["tets"][corner])
        vfq, _ = liquid_volume_centroid(bq["points"], bq["tets"], bq["phi"])
        ref = equilibrium_cap(vf, theta_e)
        ga = drop_geometry(a, cq, ref["radius"], theta_e)
        gb = drop_geometry(bq, cq, ref["radius"], theta_e)
        gfull = drop_geometry(b, cf, ref["radius"], theta_e)
        rows.append({
            "time": t,
            "velocity_max_difference_over_max_speed": float(np.max(du) / umax),
            "phi_max_difference": float(np.max(np.abs(a["phi"] - b["phi"][mapping]))),
            "volume_relative_difference_shared_quarter": abs(vq - vfq) / vfq,
            "volume_relative_difference_4x_quarter_vs_full": abs(4.0 * vq - vf) / vf,
            "base_radius_difference": abs(ga["base_radius"] - gb["base_radius"]),
            "contact_angle_sector_max_difference": float(np.max(np.abs(
                np.array([s["angle"] for s in ga["sectors"]]) -
                np.array([s["angle"] for s in gb["sectors"]])))),
            "full_domain_axis_offset": float(np.hypot(*gfull["axis"])),
            "full_domain_angle_spread_over_sectors": float(
                max(s["angle"] for s in gfull["sectors"]) - min(s["angle"] for s in gfull["sectors"])),
        })
    if not rows:
        raise DataError("the quarter and full runs share no output time")
    worst = {k: max(r[k] for r in rows) for k in rows[0] if k != "time"}
    return {"quarter_run": str(quarter_run), "full_run": str(full_run), "outputs": rows,
            "worst": worst}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--tolerances", type=Path, default=HERE / "tolerances.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-truncated", action="store_true",
                        help="report metrics of --max-steps smoke runs (never acceptance evidence)")
    parser.add_argument("--symmetry-check", nargs=2, type=Path, metavar=("QUARTER", "FULL"))
    args = parser.parse_args(argv)
    try:
        if args.symmetry_check:
            report = symmetry_check(*args.symmetry_check)
            for r in report["outputs"]:
                print(f"t={r['time']:.5g}: |du|/max|u| {r['velocity_max_difference_over_max_speed']:.3e}"
                      f"  |dphi| {r['phi_max_difference']:.3e}"
                      f"  dV/V {r['volume_relative_difference_shared_quarter']:.3e}"
                      f"  d(base) {r['base_radius_difference']:.3e}"
                      f"  d(angle) {r['contact_angle_sector_max_difference']:.3e} deg")
            print("worst: " + ", ".join(f"{k} {v:.3e}" for k, v in report["worst"].items()))
            if args.json:
                args.json.write_text(json.dumps(report, indent=2) + "\n")
            return 0
        if not args.runs:
            parser.error("give run directories or --symmetry-check")
        tolerances = json.loads(args.tolerances.read_text())
        runs = [analyse_run(r, allow_truncated=args.allow_truncated) for r in args.runs]
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["domain"], r["dt_rule"], r["dt_divisor"]), []).append(r)
    print(f"{'R/h':>4} {'domain':>7} {'rule':>15} {'dt/d':>4} {'angle mean':>10} {'max dev':>8} "
          f"{'cell mean':>9} {'base err':>9} {'apex err':>9} {'dV/V max':>9} {'growth':>7}")
    for r in sorted(runs, key=lambda r: (r["domain"], r["dt_rule"], r["dt_divisor"], r["level"])):
        f = r["final"]
        growth = "-" if r["max_speed_growth_ratio"] is None else f"{r['max_speed_growth_ratio']:.3f}"
        print(f"{r['level']:>4} {r['domain']:>7} {r['dt_rule']:>15} {r['dt_divisor']:>4} "
              f"{f['contact_angle_mean']:>10.3f} {f['contact_angle_max_deviation']:>8.3f} "
              f"{f['cell_angle_mean']:>9.3f} {r['base_radius_relative_error']:>9.2e} "
              f"{r['apex_height_relative_error']:>9.2e} {r['liquid_volume_relative_drift_max']:>9.2e} "
              f"{growth:>7}" + ("  [truncated, not acceptance evidence]" if r["truncated"] else ""))
    protocol_key = ("quarter", tolerances["protocol"]["dt_rule"], 1)
    verdicts = evaluate_group(groups.get(protocol_key, []), tolerances)
    refined = tolerances["time_step_criterion"]["refined_step_levels"]
    fine_key = ("quarter", tolerances["protocol"]["dt_rule"], 2)
    fine_verdicts = (evaluate_group(groups[fine_key], tolerances, set(refined))
                     if fine_key in groups else [])
    steps = evaluate_time_step(groups, tolerances)
    any_truncated = any(r["truncated"] for r in runs)
    all_pass = bool(verdicts) and all(v["passed"] for v in verdicts + fine_verdicts + steps) \
        and not any_truncated
    print(f"\nprotocol study {protocol_key}:")
    for v in verdicts:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
    for v in fine_verdicts:
        print(f"  [dt/2 {'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
    for v in steps:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] time_step at R/h={v['level']}: "
              + "; ".join(v["details"]))
    if not steps:
        print("  time_step: not evaluated (needs dt and dt/2 runs)")
    if args.json:
        args.json.write_text(json.dumps({"benchmark": "sessile_drop_3d", "runs": runs,
                                         "criteria": verdicts, "criteria_dt2": fine_verdicts,
                                         "time_step": steps, "passed": all_pass}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
