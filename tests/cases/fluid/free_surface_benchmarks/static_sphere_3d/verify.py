#!/usr/bin/env python3
"""Compute the static_sphere_3d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI), and
optionally the solver log (solver_run.log or solver_run.log.gz).  Runs are
grouped by (capillary form, transport, Laplace number); the criteria are
applied to each group across its resolution levels.  Runs made with a
non-protocol time step (generate_case.py --dt-multiple) are reported but not
gated.  Metric definitions are in README.md.

Exit status: 0 if every criterion passes, 1 if any criterion fails, 2 if
input data are missing or invalid.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TETRA = 10                        # VTK cell type
INTERIOR_RADIUS_FRACTION = 0.5    # pressure is averaged over |x - c| <= R_eff/2
LOG_NAMES = ("solver_run.log", "solver_run.log.gz")
# The per-step solver volume is used for the volume criterion only if it
# matches the exact snapshot volume at every output to this relative level.
LOG_VOLUME_AGREEMENT = 1.0e-8
WET_VOLUME_LINE = re.compile(r"Wet volume diagnostic step=(\d+) time=(\S+) .*? wet_volume=(\S+)")
LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class DataError(RuntimeError):
    """Missing or inconsistent input; verification cannot proceed."""


# ---------------------------------------------------------------------------
# Geometry of the P1 liquid region {phi_h < 0} on affine tetrahedra
# ---------------------------------------------------------------------------
def _tet_volume(a, b, c, d) -> np.ndarray:
    """Unsigned volumes of the tetrahedra (a, b, c, d); arrays of shape (N, 3)."""
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def _edge_point(p, f, i, j):
    """Zero crossing of the linear f on the edge (i, j) of every row; f[i] < 0 <= f[j]."""
    s = f[:, i] / (f[:, i] - f[:, j])
    return p[:, i] + s[:, None] * (p[:, j] - p[:, i])


def liquid_volume_centroid(points: np.ndarray, tets: np.ndarray, phi: np.ndarray):
    """Exact volume and centroid of {phi_h < 0} for the P1 interpolant phi_h.

    Tetrahedra with every phi <= 0 count in full.  A cut tetrahedron with k
    vertices below zero contributes the polytope clipped by the linear phi_h:
    the corner tetrahedron at the negative vertex (k = 1), the tetrahedron
    minus the corner at the non-negative vertex (k = 3), or the prism between
    the two negative vertices and the four edge crossings (k = 2), split into
    three tetrahedra.  All faces of these polytopes are planar, so the
    decomposition is exact.
    """
    f_all = phi[tets]
    full = np.all(f_all <= 0.0, axis=1)
    cut = ~full & np.any(f_all < 0.0, axis=1)
    p_full = points[tets[full]]
    v_full = _tet_volume(p_full[:, 0], p_full[:, 1], p_full[:, 2], p_full[:, 3])
    volume = float(v_full.sum())
    moment = (v_full[:, None] * p_full.mean(axis=1)).sum(axis=0)

    f = f_all[cut]
    # Order each cut tetrahedron's vertices so that the negative ones come first.
    order = np.argsort(f >= 0.0, axis=1, kind="stable")
    f = np.take_along_axis(f, order, axis=1)
    p = points[np.take_along_axis(tets[cut], order, axis=1)]
    k = np.sum(f < 0.0, axis=1)

    def add(v, c):
        nonlocal volume, moment
        volume += float(v.sum())
        moment = moment + (v[:, None] * c).sum(axis=0)

    one = k == 1
    if np.any(one):
        pp, ff = p[one], f[one]
        q = [pp[:, 0]] + [_edge_point(pp, ff, 0, j) for j in (1, 2, 3)]
        add(_tet_volume(*q), sum(q) / 4.0)
    three = k == 3
    if np.any(three):
        pp, ff = p[three], -f[three]          # the single non-negative vertex is last
        corner = [pp[:, 3]]
        for j in (0, 1, 2):
            s = ff[:, 3] / (ff[:, 3] - ff[:, j])
            corner.append(pp[:, 3] + s[:, None] * (pp[:, j] - pp[:, 3]))
        v_tet = _tet_volume(pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3])
        v_corner = _tet_volume(*corner)
        c_tet, c_corner = pp.mean(axis=1), sum(corner) / 4.0
        v = v_tet - v_corner
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(v[:, None] > 0.0,
                         (v_tet[:, None] * c_tet - v_corner[:, None] * c_corner) /
                         np.where(v > 0.0, v, 1.0)[:, None], c_tet)
        add(v, c)
    two = k == 2
    if np.any(two):
        pp, ff = p[two], f[two]
        a0, b0 = pp[:, 0], pp[:, 1]
        a1, a2 = _edge_point(pp, ff, 0, 2), _edge_point(pp, ff, 0, 3)
        b1, b2 = _edge_point(pp, ff, 1, 2), _edge_point(pp, ff, 1, 3)
        for q in ((a0, a1, a2, b2), (a0, a1, b1, b2), (a0, b0, b1, b2)):
            add(_tet_volume(*q), sum(q) / 4.0)
    if volume <= 0.0:
        raise DataError("no liquid: phi >= 0 at every vertex")
    return volume, moment / volume


def interface_points(points: np.ndarray, tets: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Zero crossings of phi_h on the edges of the cut tetrahedra (LinearCorner vertices)."""
    f = phi[tets]
    cut = np.any(f < 0.0, axis=1) & np.any(f >= 0.0, axis=1)
    t = tets[cut]
    edges = np.concatenate([t[:, list(e)] for e in LOCAL_EDGES])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    fa, fb = phi[edges[:, 0]], phi[edges[:, 1]]
    crossing = (fa < 0.0) != (fb < 0.0)
    fa, fb, e = fa[crossing], fb[crossing], edges[crossing]
    s = fa / (fa - fb)
    return points[e[:, 0]] + s[:, None] * (points[e[:, 1]] - points[e[:, 0]])


def interior_mean_pressure(points, tets, pressure, centre, radius):
    """Volume-weighted mean of the P1 pressure over the tetrahedra inside the ball."""
    inside = np.all(np.linalg.norm(points[tets] - centre, axis=2) <= radius, axis=1)
    if not np.any(inside):
        raise DataError("interior pressure region contains no tetrahedron")
    p = points[tets[inside]]
    vol = _tet_volume(p[:, 0], p[:, 1], p[:, 2], p[:, 3])
    mean_p = pressure[tets[inside]].mean(axis=1)
    return float((vol * mean_p).sum() / vol.sum()), int(inside.sum())


def max_liquid_speed(phi: np.ndarray, velocity: np.ndarray) -> float:
    liquid = phi < 0.0
    if not np.any(liquid):
        raise DataError("no liquid vertex")
    return float(np.max(np.linalg.norm(velocity[liquid, :3], axis=1)))


def sphere_radius(volume: float) -> float:
    return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)


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
    if grid.n_cells == 0 or np.any(types != TETRA):
        raise DataError(f"{path}: expected a pure Tetra4 mesh")
    tets = np.asarray(grid.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    if "GlobalElementID" in grid.cell_data:          # drop duplicated MPI cells
        _, first = np.unique(np.asarray(grid.cell_data["GlobalElementID"]), return_index=True)
        tets = tets[np.sort(first)]
    fields = {}
    for key in ("level_set_field", "velocity_field", "pressure_field"):
        name = case[key]
        if name not in grid.point_data:
            raise DataError(f"{path}: point array '{name}' is missing")
        fields[key] = np.asarray(grid.point_data[name], dtype=float)
    velocity = fields["velocity_field"].reshape(grid.n_points, -1)
    if velocity.shape[1] < 3:
        raise DataError(f"{path}: velocity must have three components")
    data = {"points": np.asarray(grid.points, dtype=float), "tets": tets,
            "phi": fields["level_set_field"], "velocity": velocity,
            "pressure": fields["pressure_field"]}
    for key, value in data.items():
        if key != "tets" and not np.all(np.isfinite(value)):
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


def log_wet_volumes(run: Path) -> list[tuple[int, float, float]] | None:
    """(step, time, wet volume) from the solver's per-step wet-volume line, if a log exists."""
    for name in LOG_NAMES:
        path = run / name
        if path.is_file():
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rt", errors="replace") as handle:
                rows = {}
                for line in handle:
                    m = WET_VOLUME_LINE.search(line)
                    if m:
                        rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            return [(s, t, v) for s, (t, v) in sorted(rows.items())]
    return None


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
    volume0, centre0 = liquid_volume_centroid(initial["points"], initial["tets"], initial["phi"])

    times, speeds, volumes = [], [], []
    for t, path in series:
        snap = read_snapshot(path, case)
        volume, _ = liquid_volume_centroid(snap["points"], snap["tets"], snap["phi"])
        times.append(t)
        speeds.append(max_liquid_speed(snap["phi"], snap["velocity"]))
        volumes.append(volume)
    times, speeds, volumes = map(np.asarray, (times, speeds, volumes))

    final = snap
    volume_f, centre_f = liquid_volume_centroid(final["points"], final["tets"], final["phi"])
    r_eff = sphere_radius(volume_f)
    p_in, n_region = interior_mean_pressure(final["points"], final["tets"], final["pressure"],
                                            centre_f, INTERIOR_RADIUS_FRACTION * r_eff)
    jump = p_in - case["external_pressure"]
    ref_eff, ref_nom = 2.0 * gamma / r_eff, 2.0 * gamma / case["radius"]
    iface = interface_points(final["points"], final["tets"], final["phi"])
    radial = np.linalg.norm(iface - centre_f, axis=1) - r_eff

    # Volume over the run (decision D11): every output, plus every step of the
    # solver log when its wet volume matches the exact snapshot volume.
    snapshot_dev = float(np.max(np.abs(volumes - volume0)) / volume0)
    log_rows = log_wet_volumes(run)
    log_dev = log_agreement = None
    log_used = False
    if log_rows:
        by_time = [(t, v) for _, t, v in log_rows]
        log_t = np.array([t for t, _ in by_time])
        log_v = np.array([v for _, v in by_time])
        matched = []
        for t, v in zip(times, volumes):
            i = int(np.argmin(np.abs(log_t - t)))
            if abs(log_t[i] - t) <= 0.5 * case["dt"]:
                matched.append(abs(log_v[i] - v) / volume0)
        log_dev = float(np.max(np.abs(log_v - volume0)) / volume0)
        if matched:
            log_agreement = float(max(matched))
            log_used = len(matched) == len(times) and log_agreement <= LOG_VOLUME_AGREEMENT
    volume_dev = max(snapshot_dev, log_dev) if log_used else snapshot_dev

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
        "transport": case.get("transport", "unknown"),
        "laplace_number": case["laplace_number"],
        "dt_multiple_of_dt_B": case.get("dt_multiple_of_dt_B"),
        "protocol_time_step": bool(case.get("dt_multiple_is_protocol", True)),
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(t_end),
        "viscous_times_simulated": float(t_end / case["viscous_time"]),
        "outputs": len(times),
        "pressure_jump": jump,
        "pressure_jump_reference": ref_eff,
        "pressure_jump_relative_error": abs(jump - ref_eff) / ref_eff,
        "pressure_jump_relative_error_nominal_radius": abs(jump - ref_nom) / ref_nom,
        "pressure_region_tetrahedra": n_region,
        "effective_radius": r_eff,
        "effective_radius_relative_to_nominal": r_eff / case["radius"] - 1.0,
        "initial_liquid_volume": volume0,
        "final_liquid_volume": volume_f,
        "liquid_volume_relative_drift_final": abs(volume_f - volume0) / volume0,
        "liquid_volume_relative_deviation_max": volume_dev,
        "liquid_volume_relative_deviation_max_outputs": snapshot_dev,
        "liquid_volume_relative_deviation_max_log": log_dev,
        "log_volume_agreement": log_agreement,
        "log_volume_used": log_used,
        "log_volume_steps": len(log_rows) if log_rows else 0,
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
                    "liquid_volume": volumes.tolist()},
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


def print_table(runs: list[dict]) -> None:
    print(f"{'R/h':>4} {'t/t_mu':>7} {'dp/(2g/R)-1':>12} {'R_eff/R-1':>10} {'Ca_final':>10}"
          f" {'growth':>7} {'dV/V max':>9} {'shape max':>9}")
    for r in runs:
        print(f"{r['level']:>4} {r['viscous_times_simulated']:>7.3g} "
              f"{r['pressure_jump'] / r['pressure_jump_reference'] - 1:>12.3e} "
              f"{r['effective_radius_relative_to_nominal']:>10.3e} "
              f"{r['parasitic_capillary_number_final']:>10.3e} "
              f"{r['max_speed_growth_ratio']:>7.3f} "
              f"{r['liquid_volume_relative_deviation_max']:>9.2e} "
              f"{r['shape_max_radial_deviation']:>9.2e}")


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

    diagnostic = [a for a in analysed if not a["protocol_time_step"]]
    protocol = [a for a in analysed if a["protocol_time_step"]]
    groups: dict[tuple, list] = {}
    for a in protocol:
        groups.setdefault((a["capillary_form"], a["transport"], a["laplace_number"]), []).append(a)
    levels = set(tolerances["levels"]["R_over_h"])
    report, all_pass = [], True
    for (form, transport, laplace), runs in sorted(groups.items()):
        seen = [r["level"] for r in runs]
        if len(seen) != len(set(seen)) or not set(seen) <= levels:
            print(f"ERROR: group {form}, {transport}, La={laplace:g}: duplicate or unknown "
                  f"levels {seen}", file=sys.stderr)
            return 2
        runs.sort(key=lambda r: r["level"])
        verdicts = evaluate_group(runs, tolerances)
        all_pass &= all(v["passed"] for v in verdicts)
        print(f"\n== {form}, transport {transport}, La = {laplace:g}" +
              ("  [TRUNCATED SMOKE RUNS]" if any(r["truncated"] for r in runs) else ""))
        print_table(runs)
        for v in verdicts:
            print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
        report.append({"capillary_form": form, "transport": transport, "laplace_number": laplace,
                       "runs": runs, "criteria": verdicts,
                       "passed": all(v["passed"] for v in verdicts)})
    if diagnostic:
        print("\n== diagnostic runs (non-protocol time step; reported, not gated)")
        for r in sorted(diagnostic, key=lambda r: (r["capillary_form"], r["transport"],
                                                   r["laplace_number"], r["level"])):
            print(f"  {r['capillary_form']}, {r['transport']}, La = {r['laplace_number']:g}, "
                  f"dt = {r['dt_multiple_of_dt_B']:g} dt_B")
            print_table([r])
    if not protocol:
        print("\nOVERALL: no protocol runs, nothing gated", file=sys.stderr)
        if args.json:
            args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"], "groups": [],
                                             "diagnostic_runs": diagnostic, "passed": False},
                                            indent=2) + "\n")
        return 2
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"],
                                         "groups": report, "diagnostic_runs": diagnostic,
                                         "passed": bool(all_pass)}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
