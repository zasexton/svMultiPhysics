#!/usr/bin/env python3
"""Compute the linear_sloshing_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI).  The runs
form one refinement study; the criteria are applied across its levels.
Frequency and damping come from the damped-oscillation fit of the modal
amplitude a1(t) of cos(k x) (decision D35); the same fit of the left-wall
probe elevation is reported as probe_*.  Metric definitions are in
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
MODAL_FIT_MODES = 4               # interface fit y = sum_{n<4} a_n cos(n k x)


class DataError(RuntimeError):
    """Missing or inconsistent input; verification cannot proceed."""


# ---------------------------------------------------------------------------
# Geometry of the P1 liquid region {phi_h < 0} on affine triangles
# ---------------------------------------------------------------------------
def _clip_negative(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    out = []
    for k in range(3):
        a, b = k, (k + 1) % 3
        if f[a] < 0.0:
            out.append(p[a])
        if (f[a] < 0.0) != (f[b] < 0.0):
            s = f[a] / (f[a] - f[b])
            out.append(p[a] + s * (p[b] - p[a]))
    return np.asarray(out)


def liquid_area(points: np.ndarray, tris: np.ndarray, phi: np.ndarray) -> float:
    """Exact area of {phi_h < 0} for the P1 interpolant phi_h."""
    f = phi[tris]
    p = points[tris]
    full = np.all(f < 0.0, axis=1)
    cut = ~full & np.any(f < 0.0, axis=1)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])[full].sum()
    for idx in np.nonzero(cut)[0]:
        q = _clip_negative(p[idx], f[idx])
        if len(q) >= 3:
            x, y = q[:, 0], q[:, 1]
            area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    if area <= 0.0:
        raise DataError("no liquid: phi >= 0 at every vertex")
    return float(area)


def interface_points(points: np.ndarray, tris: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Zero crossings of phi_h on the mesh edges (the LinearCorner polygon vertices)."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    fa, fb = phi[edges[:, 0]], phi[edges[:, 1]]
    crossing = (fa < 0.0) != (fb < 0.0)
    fa, fb, e = fa[crossing], fb[crossing], edges[crossing]
    s = fa / (fa - fb)
    return points[e[:, 0]] + s[:, None] * (points[e[:, 1]] - points[e[:, 0]])


def probe_elevation(points: np.ndarray, phi: np.ndarray, x_probe: float, h: float) -> float:
    """Height of the lowest upward zero crossing of phi_h on the vertical line x = x_probe.

    The probe line is a mesh line, so phi_h along it is linear between the
    vertices on it.
    """
    on_line = np.abs(points[:, 0] - x_probe) < 1e-9 * max(h, 1.0)
    if np.count_nonzero(on_line) < 2:
        raise DataError(f"probe line x={x_probe} is not a mesh line")
    y, order = np.unique(points[on_line, 1], return_index=True)
    f = phi[on_line][order]
    for j in range(len(y) - 1):
        if f[j] < 0.0 <= f[j + 1]:
            return float(y[j] + f[j] / (f[j] - f[j + 1]) * (y[j + 1] - y[j]))
    raise DataError("no free-surface crossing on the probe line")


def modal_coefficients(iface: np.ndarray, k: float, modes: int = MODAL_FIT_MODES) -> np.ndarray:
    """Least-squares fit y = sum_n a_n cos(n k x) to the interface points."""
    x, y = iface[:, 0], iface[:, 1]
    if len(x) < 2 * modes:
        raise DataError("too few interface points for the modal fit")
    basis = np.column_stack([np.cos(n * k * x) for n in range(modes)])
    return np.linalg.lstsq(basis, y, rcond=None)[0]


# ---------------------------------------------------------------------------
# Damped-oscillation fit
# ---------------------------------------------------------------------------
def _model(t, p):
    c, a, b, omega, gamma = p
    e = np.exp(-gamma * t)
    return c + e * (a * np.cos(omega * t) + b * np.sin(omega * t))


def _linear_part(t, y, omega, gamma):
    e = np.exp(-gamma * t)
    basis = np.column_stack([np.ones_like(t), e * np.cos(omega * t), e * np.sin(omega * t)])
    coef = np.linalg.lstsq(basis, y, rcond=None)[0]
    return coef, y - basis @ coef


def fit_damped_oscillation(t, y, omega_guess: float) -> dict:
    """Least-squares fit y(t) = c + exp(-gamma t) (a cos(omega t) + b sin(omega t)).

    A scan over omega (gamma = 0) picks the starting point, then
    Levenberg-Marquardt refines all five parameters.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.size < 8:
        raise DataError("need at least 8 samples for the oscillation fit")
    grid = np.linspace(0.5, 1.5, 1001) * omega_guess
    best = min(grid, key=lambda w: float(np.sum(_linear_part(t, y, w, 0.0)[1] ** 2)))
    coef, _ = _linear_part(t, y, best, 0.0)
    p = np.array([coef[0], coef[1], coef[2], best, 0.0])
    lam = 1e-3
    cost = float(np.sum((y - _model(t, p)) ** 2))
    iterations = 0
    for iterations in range(1, 501):
        c, a, b, omega, gamma = p
        e = np.exp(-gamma * t)
        cw, sw = np.cos(omega * t), np.sin(omega * t)
        osc = a * cw + b * sw
        jac = np.column_stack([np.ones_like(t), e * cw, e * sw,
                               e * t * (-a * sw + b * cw), -t * e * osc])
        r = y - _model(t, p)
        normal = jac.T @ jac
        grad = jac.T @ r
        step = np.linalg.solve(normal + lam * np.diag(np.diag(normal)), grad)
        trial = p + step
        trial_cost = float(np.sum((y - _model(t, trial)) ** 2))
        if trial_cost <= cost:
            converged = (abs(step[3]) <= 1e-14 * abs(trial[3])
                         and abs(step[4]) <= 1e-14 * abs(trial[3]))
            p, cost, lam = trial, trial_cost, max(lam / 3.0, 1e-12)
            if converged or cost == 0.0:
                break
        else:
            lam *= 3.0
            if lam > 1e12:
                break
    c, a, b, omega, gamma = p
    amplitude = math.hypot(a, b)
    return {"omega": float(omega), "damping_rate": float(gamma), "amplitude": float(amplitude),
            "offset": float(c), "phase": float(math.atan2(-b, a)),
            "rms_residual_over_amplitude": float(math.sqrt(cost / t.size) / max(amplitude, 1e-300)),
            "iterations": iterations}


def observed_order(levels, errors) -> float:
    """Least-squares slope of log(error) against log(L/h)."""
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
    for key in ("level_set_field", "velocity_field"):
        name = case[key]
        if name not in grid.point_data:
            raise DataError(f"{path}: point array '{name}' is missing")
        fields[key] = np.asarray(grid.point_data[name], dtype=float)
    velocity = fields["velocity_field"].reshape(grid.n_points, -1)
    if velocity.shape[1] < 2:
        raise DataError(f"{path}: velocity must have at least two components")
    data = {"points": np.asarray(grid.points, dtype=float)[:, :2], "tris": tris,
            "phi": fields["level_set_field"], "velocity": velocity[:, :2]}
    for key, value in data.items():
        if key != "tris" and not np.all(np.isfinite(value)):
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


def solver_log_summary(run: Path) -> dict | None:
    """Newton statistics and wall time from the solver log, if it was kept (reported only)."""
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
               "newton_iterations_mean": float(np.mean([r[1] for r in rows])),
               "outer_passes_mean": float(np.mean([r[3] for r in rows])),
               "final_residual_max": float(max(r[2] for r in rows))}
    run_txt = run / "run.txt"
    if run_txt.is_file():
        m = re.search(r"elapsed_s=(\d+)", run_txt.read_text())
        if m:
            summary["wall_seconds"] = int(m.group(1))
            summary["wall_seconds_per_step"] = int(m.group(1)) / len(rows)
    return summary


def analyse_run(run: Path, *, allow_short: bool = False) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    series = output_series(run, case)
    if abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={case['end_time']:.6g}")
    k, h, depth = case["wavenumber"], case["h"], case["mean_depth"]

    snaps = [(0.0, run / "mesh" / "mesh-complete.mesh.vtu")] + series
    times, probe, modal, areas, speeds = [], [], [], [], []
    for t, path in snaps:
        snap = read_snapshot(path, case)
        times.append(t)
        probe.append(probe_elevation(snap["points"], snap["phi"], case["probe_x"], h) - depth)
        iface = interface_points(snap["points"], snap["tris"], snap["phi"])
        modal.append(modal_coefficients(iface, k)[1])
        areas.append(liquid_area(snap["points"], snap["tris"], snap["phi"]))
        liquid = snap["phi"] < 0.0
        speeds.append(float(np.max(np.linalg.norm(snap["velocity"][liquid], axis=1))))
    times, probe, modal, areas = map(np.asarray, (times, probe, modal, areas))

    periods = times[-1] * case["omega_inviscid"] / (2.0 * math.pi)
    if periods < 2.0 and not allow_short:
        raise DataError(f"{run}: {periods:.2f} periods is too short for the frequency fit "
                        "(protocol: 4); use --allow-truncated for smoke runs")
    omega_ref, gamma_ref = case["omega_reference"], case["damping_rate_reference"]
    # Decision D35: the gated frequency and damping come from the modal
    # amplitude a1(t), which the symmetric cos(2kx) mode and a slow mean-level
    # change do not enter; the wall-probe fit is reported for comparison.
    fit = fit_damped_oscillation(times, modal, omega_ref)
    fit_probe = fit_damped_oscillation(times, probe, omega_ref)
    area0 = areas[0]
    return {
        "run": str(run),
        "level": case["level_cells_per_length"],
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(times[-1]),
        "periods_simulated": float(periods),
        "outputs": int(times.size - 1),
        "steps_per_period": case["steps_per_period"],
        "level_set_velocity": case.get("level_set_velocity", "coupled_field"),
        "mean_depth": case["mean_depth"],
        "protocol_run": bool(case.get("protocol_run", case.get("protocol_time_step", True))),
        "omega_reference": omega_ref,
        "omega_inviscid": case["omega_inviscid"],
        "damping_rate_reference": gamma_ref,
        "damping_rate_lamb": case["damping_rate_lamb"],
        "omega": fit["omega"],
        "damping_rate": fit["damping_rate"],
        "frequency_relative_error": abs(fit["omega"] - omega_ref) / omega_ref,
        "frequency_signed_error": fit["omega"] / omega_ref - 1.0,
        "frequency_relative_error_vs_inviscid": abs(fit["omega"] - case["omega_inviscid"]) / case["omega_inviscid"],
        "damping_rate_relative_error": abs(fit["damping_rate"] - gamma_ref) / gamma_ref,
        "damping_rate_over_reference": fit["damping_rate"] / gamma_ref,
        "damping_rate_over_lamb": fit["damping_rate"] / case["damping_rate_lamb"],
        "fit_amplitude_over_initial": fit["amplitude"] / case["amplitude"],
        "fit_offset": fit["offset"],
        "fit_rms_residual_over_amplitude": fit["rms_residual_over_amplitude"],
        "fit_source": "modal_amplitude",
        "probe_omega": fit_probe["omega"],
        "probe_frequency_relative_error": abs(fit_probe["omega"] - omega_ref) / omega_ref,
        "probe_frequency_signed_error": fit_probe["omega"] / omega_ref - 1.0,
        "probe_damping_rate": fit_probe["damping_rate"],
        "probe_damping_rate_relative_error": abs(fit_probe["damping_rate"] - gamma_ref) / gamma_ref,
        "probe_damping_rate_over_reference": fit_probe["damping_rate"] / gamma_ref,
        "probe_fit_amplitude_over_initial": fit_probe["amplitude"] / case["amplitude"],
        "probe_fit_offset": fit_probe["offset"],
        "probe_fit_rms_residual_over_amplitude": fit_probe["rms_residual_over_amplitude"],
        "initial_probe_elevation_over_amplitude": float(probe[0] / case["amplitude"]),
        "initial_liquid_area": float(area0),
        "liquid_area_relative_drift_max": float(np.max(np.abs(areas - area0)) / area0),
        "liquid_area_relative_drift_final": float(abs(areas[-1] - area0) / area0),
        "max_liquid_speed": float(max(speeds)),
        "solver_log": solver_log_summary(run),
        "history": {"time": times.tolist(), "probe_elevation": probe.tolist(),
                    "modal_amplitude": modal.tolist(), "liquid_area": areas.tolist(),
                    "max_liquid_speed": speeds},
    }


# ---------------------------------------------------------------------------
# Studies (decision D10) and criteria
# ---------------------------------------------------------------------------
def remove_time_error(group: list[dict], protocol: dict) -> dict:
    """Estimate the time error of the spatial-study step and remove it.

    The time-step study runs one level at dt, dt/2 and dt/4 around the spatial
    step dt_s (steps per period 64, 128, 256 with dt_s at 128).  With the
    scheme's second order, e(dt_s) - e(dt_s/2) = (3/4) e_time(dt_s), so
    e_time(dt_s) = (4/3) (e(dt_s) - e(dt_s/2)).  The observed temporal order
    from the three runs is reported as a check of that assumption.
    """
    spatial_spp = protocol["spatial_study"]["steps_per_period"]
    level = protocol["time_study"]["level"]
    spps = sorted(protocol["time_study"]["steps_per_period"])
    by_spp = {r["steps_per_period"]: r for r in group if r["level"] == level}
    summary = {"available": all(spp in by_spp for spp in spps)}
    if not summary["available"]:
        summary["missing_steps_per_period"] = [spp for spp in spps if spp not in by_spp]
        return summary
    errors = [by_spp[spp]["frequency_signed_error"] for spp in spps]
    finer = spps[spps.index(spatial_spp) + 1]
    time_error = (4.0 / 3.0) * (by_spp[spatial_spp]["frequency_signed_error"]
                                - by_spp[finer]["frequency_signed_error"])
    d1, d2 = errors[0] - errors[1], errors[1] - errors[2]
    summary.update({
        "steps_per_period": spps,
        "signed_errors": errors,
        "observed_temporal_order": (math.log2(d1 / d2) if d1 != 0.0 and d2 != 0.0
                                    and d1 / d2 > 0.0 else float("nan")),
        "time_error_at_spatial_step": time_error,
        "damping_rate_over_reference": [by_spp[spp]["damping_rate_over_reference"]
                                        for spp in spps],
    })
    for r in group:
        if r["steps_per_period"] == spatial_spp:
            r["frequency_spatial_signed_error"] = r["frequency_signed_error"] - time_error
            r["frequency_spatial_error"] = abs(r["frequency_spatial_signed_error"])
    return summary


def evaluate_study(group: list[dict], tolerances: dict) -> list[dict]:
    protocol = tolerances["protocol"]
    spatial_spp = protocol["spatial_study"]["steps_per_period"]
    spatial = {r["level"]: r for r in group if r["steps_per_period"] == spatial_spp}
    results = []
    for crit in tolerances["criteria"]:
        q, limit = crit["quantity"], crit.get("limit")
        messages, ok = [], True
        at = crit.get("at_level", "each")
        if at == "each_run":
            targets = [(f"L/h={r['level']} steps/T={r['steps_per_period']}", r)
                       for r in sorted(group, key=lambda r: (r["level"], r["steps_per_period"]))]
            for level in protocol["spatial_study"]["levels"]:
                if level not in spatial:
                    ok = False
                    messages.append(f"missing spatial run at L/h={level}")
        else:
            levels = sorted(spatial) if at == "each" else [at]
            targets = []
            for level in levels:
                if level not in spatial:
                    ok = False
                    messages.append(f"missing run at L/h={level}")
                else:
                    targets.append((f"L/h={level}", spatial[level]))
        for label, run in targets:
            if q not in run:
                ok = False
                messages.append(f"{label}: {q} unavailable (time-step study incomplete)")
                continue
            value = run[q]
            passed = value <= limit
            ok &= passed
            messages.append(f"{label}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
        need = crit.get("monotone_levels") or crit.get("order_levels") or []
        values = [spatial[lv].get(q) if lv in spatial else None for lv in need]
        complete = bool(need) and all(v is not None for v in values)
        if crit.get("monotone") == "strictly_decreasing":
            if not complete:
                ok = False
                messages.append("monotonicity needs every spatial level")
            else:
                passed = all(b < a for a, b in zip(values, values[1:]))
                ok &= passed
                messages.append("decreasing over L/h=" + "/".join(map(str, need)) +
                                (": yes" if passed else ": NO (" +
                                 ", ".join(f"{v:.3g}" for v in values) + ")"))
        if "minimum_observed_order" in crit:
            if not complete:
                ok = False
                messages.append("order needs every spatial level")
            elif min(values) <= 0.0:
                ok = False
                messages.append("order undefined for a zero error")
            else:
                order = observed_order(need, values)
                pairs = ", ".join(f"{observed_order(need[i:i + 2], values[i:i + 2]):.2f}"
                                  for i in range(len(need) - 1))
                passed = order >= crit["minimum_observed_order"]
                ok &= passed
                messages.append(f"observed order {order:.2f} (pairwise {pairs}) "
                                f"{'>=' if passed else '<'} {crit['minimum_observed_order']}")
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
        runs = [analyse_run(run, allow_short=args.allow_truncated) for run in args.runs]
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    truncated = [r["run"] for r in runs if r["truncated"]]
    if truncated and not args.allow_truncated:
        print("ERROR: truncated smoke runs are not acceptance evidence: " + ", ".join(truncated),
              file=sys.stderr)
        return 2
    protocol = tolerances["protocol"]
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["level_set_velocity"], r["mean_depth"]), []).append(r)
    for key, group in groups.items():
        seen = [(r["level"], r["steps_per_period"]) for r in group]
        if len(seen) != len(set(seen)) or not {lv for lv, _ in seen} <= set(
                tolerances["levels"]["cells_per_length"]):
            print(f"ERROR: duplicate or unknown (level, steps/T) runs {seen} for {key}",
                  file=sys.stderr)
            return 2
    protocol_key = (protocol["level_set_velocity"], protocol["mean_depth"])
    order = sorted(groups, key=lambda k: (k != protocol_key, k))
    report, protocol_verdicts = [], None
    for key in order:
        group = sorted(groups[key], key=lambda r: (r["level"], r["steps_per_period"]))
        time_summary = remove_time_error(group, protocol)
        verdicts = evaluate_study(group, tolerances)
        is_protocol = key == protocol_key
        if is_protocol:
            protocol_verdicts = verdicts
        print(f"\n== level-set velocity {key[0]}, H0 = {key[1]:.7g}"
              + ("  [PROTOCOL]" if is_protocol else "  [comparison, not gated]"))
        print("modal-amplitude fit (gated, D35); probe columns are the left-wall probe fit")
        print(f"{'L/h':>4} {'steps/T':>7} {'omega':>12} {'freq err':>10} {'spatial':>10} "
              f"{'g/g_ref':>8} {'g/g_Lamb':>8} {'fit rms':>8} {'probe fe':>10} {'probe g/g':>9} "
              f"{'dA/A max':>9} {'s/step':>7}")
        for r in group:
            spatial = r.get("frequency_spatial_signed_error")
            log = r.get("solver_log") or {}
            print(f"{r['level']:>4} {r['steps_per_period']:>7} {r['omega']:>12.8f} "
                  f"{r['frequency_signed_error']:>+10.3e} "
                  + (f"{spatial:>+10.3e} " if spatial is not None else f"{'-':>10} ")
                  + f"{r['damping_rate_over_reference']:>8.4f} {r['damping_rate_over_lamb']:>8.4f} "
                  f"{r['fit_rms_residual_over_amplitude']:>8.1e} "
                  f"{r['probe_frequency_signed_error']:>+10.3e} "
                  f"{r['probe_damping_rate_over_reference']:>9.4f} "
                  f"{r['liquid_area_relative_drift_max']:>9.2e} "
                  + (f"{log['wall_seconds_per_step']:>7.2f}" if "wall_seconds_per_step" in log
                     else f"{'-':>7}")
                  + ("  [TRUNCATED SMOKE RUN]" if r["truncated"] else ""))
        if time_summary["available"]:
            print(f"time-step study at L/h={protocol['time_study']['level']}: observed temporal "
                  f"order {time_summary['observed_temporal_order']:.2f}; time error at "
                  f"{protocol['spatial_study']['steps_per_period']} steps/T "
                  f"{time_summary['time_error_at_spatial_step']:+.3e} (removed from the "
                  "spatial errors)")
        else:
            print("time-step study incomplete: missing steps/T "
                  f"{time_summary['missing_steps_per_period']} at "
                  f"L/h={protocol['time_study']['level']}")
        for v in verdicts:
            tag = ("PASS" if v["passed"] else "FAIL") if is_protocol else "INFO"
            print(f"  [{tag}] {v['id']}: " + "; ".join(v["details"]))
        report.append({"level_set_velocity": key[0], "mean_depth": key[1],
                       "protocol": is_protocol, "time_study": time_summary,
                       "runs": group, "criteria": verdicts})
    everything = [r for g in report for r in g["runs"]]
    if everything:
        print(f"\nreference: omega = {everything[0]['omega_reference']:.10f} (inviscid "
              f"{everything[0]['omega_inviscid']:.10f}), gamma = "
              f"{everything[0]['damping_rate_reference']:.6e} "
              f"(Lamb 2 nu k^2 = {everything[0]['damping_rate_lamb']:.6e})")
    if protocol_verdicts is None:
        print(f"\nno runs of the protocol transport {protocol_key[0]}; nothing is gated")
        all_pass = False
    else:
        all_pass = all(v["passed"] for v in protocol_verdicts)
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"], "groups": report,
                                         "passed": bool(all_pass)}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
