#!/usr/bin/env python3
"""Compute the capillary_wave_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT] [--allow-truncated]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI), and
optionally solver_run.log or solver_run.log.gz, whose per-step wet-volume
lines enter the area criterion (decision D11).  Runs are grouped by
(capillary form, level-set transport, Laplace number); the criteria are
applied to each group across its resolution levels, which must share the
protocol time step (decision D10).  Runs with a refined time step
(generate_case.py --dt-divisor 2 or 4) are reported in a time-step study at
their level.  Metric definitions are in README.md.

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prosperetti_reference as reference  # noqa: E402

TRIANGLE = 5                      # VTK cell type
MIN_FIT_PERIODS = 1.0             # frequency/damping need at least one inviscid period
MIN_FIT_SAMPLES = 8


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


def _edge_moments(x0, y0, x1, y1, k):
    """Signed area and cos-moment contributions of the directed edges (x0,y0)->(x1,y1).

    By Green's theorem, int_R cos(kx) dA = oint sin(kx)/k dy, and along a
    straight edge the line integral is exactly
    (dy/k) sin(k xm) sinc(k dx / 2) with xm the edge midpoint.
    """
    dx, dy = x1 - x0, y1 - y0
    area = 0.5 * (x0 * y1 - x1 * y0)
    moment = dy / k * np.sin(0.5 * k * (x0 + x1)) * np.sinc(k * dx / (2.0 * math.pi))
    return area, moment


def _polygon_moments(q: np.ndarray, k: float) -> tuple[float, float]:
    if len(q) < 3:
        return 0.0, 0.0
    x0, y0 = q[:, 0], q[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    area, moment = _edge_moments(x0, y0, x1, y1, k)
    area, moment = area.sum(), moment.sum()
    sign = 1.0 if area >= 0.0 else -1.0
    return sign * area, sign * moment


def liquid_measures(points: np.ndarray, tris: np.ndarray, phi: np.ndarray,
                    wavenumber: float) -> tuple[float, float]:
    """Exact area of {phi_h < 0} and exact integral of cos(kx) over it."""
    f = phi[tris]
    p = points[tris]
    full = np.all(f <= 0.0, axis=1)
    cut = ~full & np.any(f < 0.0, axis=1)
    pf = p[full]
    area_t = np.zeros(pf.shape[0])
    moment_t = np.zeros(pf.shape[0])
    for a in range(3):
        b = (a + 1) % 3
        ar, mo = _edge_moments(pf[:, a, 0], pf[:, a, 1], pf[:, b, 0], pf[:, b, 1], wavenumber)
        area_t += ar
        moment_t += mo
    sign = np.where(area_t >= 0.0, 1.0, -1.0)
    area = float((sign * area_t).sum())
    moment = float((sign * moment_t).sum())
    for idx in np.nonzero(cut)[0]:
        a, m = _polygon_moments(_clip_negative(p[idx], f[idx]), wavenumber)
        area += a
        moment += m
    if area <= 0.0:
        raise DataError("no liquid: phi >= 0 at every vertex")
    return area, moment


def mode_amplitude(cos_moment: float, width: float, wavenumber: float) -> float:
    """cos(kx) coefficient of the surface elevation on [0, width].

    int_0^W eta(x) cos(kx) dx = int_liquid cos(kx) dA (the bottom and the walls
    at x = 0 and x = W = multiple of lambda/2 add nothing), and
    int_0^W cos^2(kx) dx = W/2.
    """
    return cos_moment / (0.5 * width + math.sin(2.0 * wavenumber * width) / (4.0 * wavenumber))


def wall_height(points: np.ndarray, phi: np.ndarray, x_wall: float, tol: float) -> float:
    """Height of the zero crossing of phi_h along the wall x = x_wall (NaN if none)."""
    on_wall = np.nonzero(np.abs(points[:, 0] - x_wall) <= tol)[0]
    order = on_wall[np.argsort(points[on_wall, 1])]
    y, f = points[order, 1], phi[order]
    for i in range(len(order) - 1):
        if f[i] < 0.0 <= f[i + 1]:
            return float(y[i] + f[i] / (f[i] - f[i + 1]) * (y[i + 1] - y[i]))
    return float("nan")


# ---------------------------------------------------------------------------
# Frequency and damping
# ---------------------------------------------------------------------------
def fit_damped_cosine(times, values, omega_guess: float) -> dict:
    """Least-squares fit of a(t) = exp(-beta t) (c1 cos(omega t) + c2 sin(omega t)).

    A coarse scan over omega in [0.5, 1.5] omega_guess and beta in
    [-0.05, 0.5] omega_guess (linear least squares for c1, c2 at each node)
    supplies the start of a Levenberg-Marquardt iteration on all four
    parameters.  No parameter depends on the data being fitted beyond the
    scan window, which is wide compared with the tolerances.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size < 4 or t.size != y.size or not np.all(np.isfinite(y)):
        raise DataError("fit needs at least four finite samples")
    scale = float(np.max(np.abs(y)))
    if not scale > 0.0:
        raise DataError("fit of an identically zero history")
    yn = y / scale

    omegas = omega_guess * np.linspace(0.5, 1.5, 201)
    betas = omega_guess * np.linspace(-0.05, 0.5, 111)
    e = np.exp(-betas[:, None] * t[None, :])[:, None, :]
    b1 = e * np.cos(omegas[:, None] * t[None, :])[None, :, :]
    b2 = e * np.sin(omegas[:, None] * t[None, :])[None, :, :]
    g11, g12, g22 = (b1 * b1).sum(-1), (b1 * b2).sum(-1), (b2 * b2).sum(-1)
    r1, r2 = (b1 * yn).sum(-1), (b2 * yn).sum(-1)
    det = g11 * g22 - g12 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        c1, c2 = (g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det
        residual = (yn * yn).sum() - (c1 * r1 + c2 * r2)
    ib, iw = np.unravel_index(np.nanargmin(residual), residual.shape)
    p = np.array([c1[ib, iw], c2[ib, iw], betas[ib], omegas[iw]])

    def model(q):
        ex, co, si = np.exp(-q[2] * t), np.cos(q[3] * t), np.sin(q[3] * t)
        m = ex * (q[0] * co + q[1] * si)
        jac = np.column_stack([ex * co, ex * si, -t * m, ex * t * (q[1] * co - q[0] * si)])
        return m - yn, jac

    r, jac = model(p)
    cost, lam, converged = float(r @ r), 1e-3, False
    for _ in range(500):
        a = jac.T @ jac
        step = np.linalg.solve(a + lam * np.diag(np.diag(a)), -jac.T @ r)
        rn, jn = model(p + step)
        cn = float(rn @ rn)
        if cn <= cost:
            p, r, jac, cost, lam = p + step, rn, jn, cn, max(lam / 10.0, 1e-15)
            if np.max(np.abs(step[2:])) <= 1e-12 * omega_guess:
                converged = True
                break
        else:
            lam *= 10.0
            if lam > 1e10:              # no descent left: at the minimum to round-off
                converged = True
                break
    return {"omega": float(p[3]), "beta": float(p[2]),
            "amplitude": float(math.hypot(p[0], p[1]) * scale),
            "phase": float(math.atan2(-p[1], p[0])),
            "residual_rms": float(math.sqrt(cost / t.size) * scale),
            "converged": bool(converged)}


def observed_order(levels, errors) -> float:
    """Least-squares slope of log(error) against log(lambda/h)."""
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


def read_level_set(path: Path, case: dict) -> dict:
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
    name = case["level_set_field"]
    if name not in grid.point_data:
        raise DataError(f"{path}: point array '{name}' is missing")
    phi = np.asarray(grid.point_data[name], dtype=float).ravel()
    points = np.asarray(grid.points, dtype=float)[:, :2]
    if phi.size != points.shape[0]:
        raise DataError(f"{path}: '{name}' must be a scalar point array")
    if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(points))):
        raise DataError(f"{path}: non-finite values")
    return {"points": points, "tris": tris, "phi": phi}


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


_WET_VOLUME = re.compile(r"Wet volume diagnostic step=(\d+) .*?domain_id='([^']*)'.*?"
                         r"physical_wet_volume=([-+0-9.eE]+)")


def logged_wet_volumes(run: Path, domain_id: str) -> dict[int, float]:
    """Per-step liquid area from the solver's 'Wet volume diagnostic' log lines.

    The solver prints one such line per accepted step (step 0 is the initial
    state).  Returns {} when no log is present.
    """
    for name in ("solver_run.log", "solver_run.log.gz"):
        path = run / name
        if path.is_file():
            break
    else:
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    volumes: dict[int, float] = {}
    with opener(path, "rt", errors="replace") as log:
        for line in log:
            if "Wet volume diagnostic step=" not in line:
                continue
            m = _WET_VOLUME.search(line)
            if m and m.group(2) == domain_id:
                volumes[int(m.group(1))] = float(m.group(3))
    return volumes


def snapshot_measures(snap: dict, case: dict) -> dict:
    k = case["wavenumber"]
    x_min, x_max = case["box"][0], case["box"][1]
    width = x_max - x_min
    area, moment = liquid_measures(snap["points"], snap["tris"], snap["phi"], k)
    tol = 1e-9 * case["wavelength"]
    mean = area / width
    return {"area": area, "amplitude": mode_amplitude(moment, width, k),
            "wall_left": wall_height(snap["points"], snap["phi"], x_min, tol) - mean,
            "wall_right": wall_height(snap["points"], snap["phi"], x_max, tol) - mean}


def analyse_run(run: Path) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    if case.get("benchmark") != "capillary_wave_2d":
        raise DataError(f"{run}: case.json is not a capillary_wave_2d case")
    width = case["box"][1] - case["box"][0]
    if not math.isclose(2.0 * width / case["wavelength"], round(2.0 * width / case["wavelength"]),
                        abs_tol=1e-12):
        raise DataError(f"{run}: box width must be a multiple of half a wavelength")
    series = output_series(run, case)
    if abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={case['end_time']:.6g}")

    initial = snapshot_measures(read_level_set(run / "mesh" / "mesh-complete.mesh.vtu", case), case)
    rows = [(0.0, initial)]
    for t, path in series:
        if t <= 0.0:
            continue
        rows.append((t, snapshot_measures(read_level_set(path, case), case)))
    times = np.array([t for t, _ in rows])
    amp = np.array([m["amplitude"] for _, m in rows])
    area = np.array([m["area"] for _, m in rows])

    # D11: the area criterion is the maximum deviation over the whole run.
    # The outputs are sampled every `output_cadence` steps; the solver log,
    # when present, adds the area of every accepted step.
    logged = logged_wet_volumes(run, case.get("interface_domain_id", "capillary_wave_surface"))
    logged_drift = (max(abs(v - area[0]) for v in logged.values()) / area[0]) if logged else 0.0
    output_drift = float(np.max(np.abs(area - area[0])) / area[0])

    params = dict(wavenumber=case["wavenumber"], kinematic_viscosity=case["kinematic_viscosity"],
                  surface_tension=case["surface_tension"], density=case["density"])
    a0 = case["initial_amplitude"]
    amp_ref = reference.prosperetti_amplitude(times, initial_amplitude=a0, **params)
    mode = reference.normal_mode(**params)
    omega0 = mode["omega0"]
    periods = float(times[-1] * omega0 / (2.0 * math.pi))

    normalized_error = amp / amp[0] - amp_ref / a0
    result = {
        "run": str(run),
        "level": case["level_lambda_over_h"],
        "capillary_form": case["capillary_form"],
        "transport": case.get("transport", "coupled"),
        "laplace_number": case["laplace_number"],
        "dt": case["dt"],
        "dt_divisor": case.get("dt_divisor", 1),
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(times[-1]),
        "periods_simulated": periods,
        "outputs": int(len(times) - 1),
        "initial_amplitude_discrete_relative": float(amp[0] / a0 - 1.0),
        "amplitude_rms_error": float(np.sqrt(np.mean(normalized_error ** 2))),
        "amplitude_max_error": float(np.max(np.abs(normalized_error))),
        "initial_liquid_area": float(area[0]),
        "liquid_area_relative_drift_max": max(output_drift, logged_drift),
        "liquid_area_relative_drift_max_outputs": output_drift,
        "liquid_area_logged_steps": len(logged),
        "mean_level_drift_over_amplitude": max(output_drift, logged_drift) * area[0] / width / a0,
        "reference_omega0": omega0,
        "reference_weak_damping_rate": mode["weak_damping_rate"],
        "reference_normal_mode_omega": mode["omega"],
        "reference_normal_mode_damping_rate": mode["beta"],
        "fit": None, "reference_fit": None,
        "frequency_relative_error": None, "damping_rate_relative_error": None,
        "history": {"time": times.tolist(), "amplitude": amp.tolist(),
                    "reference_amplitude": amp_ref.tolist(), "liquid_area": area.tolist(),
                    "wall_elevation_left": [m["wall_left"] for _, m in rows],
                    "wall_elevation_right": [m["wall_right"] for _, m in rows]},
    }
    if periods >= MIN_FIT_PERIODS and len(times) >= MIN_FIT_SAMPLES:
        fit_ref = fit_damped_cosine(times, amp_ref, omega0)
        fit_sim = fit_damped_cosine(times, amp, omega0)
        if not (fit_ref["converged"] and fit_ref["beta"] > 0.0):
            raise DataError(f"{run}: fit of the reference history failed: {fit_ref}")
        result["fit"], result["reference_fit"] = fit_sim, fit_ref
        if fit_sim["converged"]:
            result["frequency_relative_error"] = abs(fit_sim["omega"] - fit_ref["omega"]) / fit_ref["omega"]
            result["damping_rate_relative_error"] = abs(fit_sim["beta"] - fit_ref["beta"]) / fit_ref["beta"]
    return result


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
        levels = sorted(by_level) if at == "each" else (list(at) if isinstance(at, list) else [at])
        for level in levels:
            if level not in by_level:
                ok = False
                messages.append(f"missing run at lambda/h={level}")
                continue
            value = by_level[level][q]
            if value is None:
                ok = False
                messages.append(f"lambda/h={level}: not evaluable (history shorter than "
                                f"{MIN_FIT_PERIODS:g} period or fit failed)")
                continue
            passed = value <= limit
            ok &= passed
            messages.append(f"lambda/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
        if "minimum_observed_order" in crit:
            need = crit["order_levels"]
            errs = [by_level[lv][q] if lv in by_level else None for lv in need]
            if any(e is None for e in errs):
                ok = False
                messages.append(f"order needs evaluable runs at lambda/h={need}")
            elif min(errs) <= 0.0:
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
        results.append({"id": crit["id"], "quantity": q, "passed": bool(ok),
                        "details": messages})
    return results


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


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

    # The criteria apply to the protocol time step (divisor 1).  Runs with a
    # refined time step enter only the reported time-step study below.
    groups: dict[tuple, list] = {}
    for a in analysed:
        if a["dt_divisor"] == 1:
            groups.setdefault((a["capillary_form"], a["transport"], a["laplace_number"]),
                              []).append(a)
    levels = set(tolerances["levels"]["lambda_over_h"])
    report, all_pass = [], bool(groups)
    if not groups:
        print("no run at the protocol time step (dt divisor 1): the criteria cannot be applied")
    for (form, transport, laplace), runs in sorted(groups.items()):
        seen = [r["level"] for r in runs]
        if len(seen) != len(set(seen)) or not set(seen) <= levels:
            print(f"ERROR: group {form}, {transport}, La={laplace:g}: duplicate or unknown "
                  f"levels {seen}", file=sys.stderr)
            return 2
        steps = sorted({r["dt"] for r in runs})
        if steps[-1] > steps[0] * (1.0 + 1e-12):
            print(f"ERROR: group {form}, {transport}, La={laplace:g}: the levels use different "
                  f"time steps {steps}; the spatial study needs one shared step (D10)",
                  file=sys.stderr)
            return 2
        runs.sort(key=lambda r: r["level"])
        verdicts = evaluate_group(runs, tolerances)
        all_pass &= all(v["passed"] for v in verdicts)
        print(f"\n== {form}, transport {transport}, La = {laplace:g}, dt = {runs[0]['dt']:.6g}" +
              ("  [TRUNCATED SMOKE RUNS]" if any(r["truncated"] for r in runs) else ""))
        print(f"{'l/h':>4} {'periods':>7} {'omega':>9} {'omega err':>9} {'beta':>8} "
              f"{'beta err':>9} {'rms err':>8} {'dA/A max':>9} {'a_h(0)/a0-1':>11}")
        for r in runs:
            fit = r["fit"] or {}
            print(f"{r['level']:>4} {r['periods_simulated']:>7.3g} {_fmt(fit.get('omega'), '9.5g')} "
                  f"{_fmt(r['frequency_relative_error'], '9.2e')} {_fmt(fit.get('beta'), '8.4g')} "
                  f"{_fmt(r['damping_rate_relative_error'], '9.2e')} {r['amplitude_rms_error']:>8.2e} "
                  f"{r['liquid_area_relative_drift_max']:>9.2e} "
                  f"{r['initial_amplitude_discrete_relative']:>11.2e}")
        ref = runs[0]
        ref_fit = ref["reference_fit"] or {}
        print(f"  reference (Prosperetti, fitted on the same samples): omega = "
              f"{_fmt(ref_fit.get('omega'), '.6g')}, beta = {_fmt(ref_fit.get('beta'), '.6g')}; "
              f"normal mode omega = {ref['reference_normal_mode_omega']:.6g}, beta = "
              f"{ref['reference_normal_mode_damping_rate']:.6g}; inviscid omega0 = "
              f"{ref['reference_omega0']:.6g}, 2 nu k^2 = {ref['reference_weak_damping_rate']:.6g}")
        for v in verdicts:
            print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
        report.append({"capillary_form": form, "transport": transport,
                       "laplace_number": laplace, "dt_divisor": 1,
                       "runs": runs, "criteria": verdicts,
                       "passed": all(v["passed"] for v in verdicts)})

    # Temporal refinement at a fixed level (reported only).
    by_level: dict[tuple, list] = {}
    for a in analysed:
        by_level.setdefault((a["capillary_form"], a["transport"], a["laplace_number"], a["level"]),
                            []).append(a)
    temporal = []
    for (form, transport, laplace, level), runs in sorted(by_level.items()):
        if len({r["dt_divisor"] for r in runs}) < 2:
            continue
        runs.sort(key=lambda r: r["dt_divisor"])
        finest = runs[-1]["fit"] or {}
        print(f"\n-- time-step study {form}, transport {transport}, La = {laplace:g}, "
              f"lambda/h = {level} (reported only; change relative to dt/{runs[-1]['dt_divisor']})")
        rows = []
        for r in runs:
            fit = r["fit"] or {}
            d_omega = (abs(fit["omega"] - finest["omega"]) / finest["omega"]
                       if fit and finest else None)
            d_beta = (abs(fit["beta"] - finest["beta"]) / finest["beta"]
                      if fit and finest else None)
            print(f"   dt/{r['dt_divisor']}: omega err {_fmt(r['frequency_relative_error'], '.3e')}, "
                  f"beta err {_fmt(r['damping_rate_relative_error'], '.3e')}, "
                  f"rms err {r['amplitude_rms_error']:.3e}; omega change {_fmt(d_omega, '.3e')}, "
                  f"beta change {_fmt(d_beta, '.3e')}")
            rows.append({"dt_divisor": r["dt_divisor"], "dt": r["dt"],
                         "frequency_relative_error": r["frequency_relative_error"],
                         "damping_rate_relative_error": r["damping_rate_relative_error"],
                         "amplitude_rms_error": r["amplitude_rms_error"],
                         "omega_change_from_smallest_dt": d_omega,
                         "beta_change_from_smallest_dt": d_beta})
        temporal.append({"capillary_form": form, "transport": transport, "laplace_number": laplace,
                         "level": level, "rows": rows})
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"],
                                         "groups": report, "time_step_study": temporal,
                                         "passed": bool(all_pass)}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
