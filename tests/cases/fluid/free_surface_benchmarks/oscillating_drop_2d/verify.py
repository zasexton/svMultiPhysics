#!/usr/bin/env python3
"""Compute the oscillating_drop_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT] [--allow-truncated]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI), and
optionally solver_run.log or solver_run.log.gz, whose per-step wet-volume
lines enter the area criterion (decision D11) and whose Newton statistics
are reported.  Runs are grouped by (capillary form, level-set transport,
Laplace number, dt divisor); the criteria are applied to each group across
its resolution levels, which must share one time step (decision D10).  The
divisor-1 study must contain every level; the divisor-2 study the levels in
time_step_criterion.refined_step_levels.  The time-step criterion compares
the fitted frequency and damping of the divisor-1 and divisor-2 runs at their
finest common level; with only one divisor it is reported as not evaluated.
Every level run at two or more divisors is listed in the time-step study.
Metric definitions are in README.md.

Exit status: 0 if every criterion passes, 1 if any criterion fails, 2 if
input data are missing or invalid.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import drop_reference as reference  # noqa: E402


def _load(directory: str, name: str):
    path = HERE.parent / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}_for_oscillating_drop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CWV = _load("capillary_wave_2d", "verify")    # fit, order, output series, log areas, reader
DataError = CWV.DataError
fit_damped_cosine = CWV.fit_damped_cosine
observed_order = CWV.observed_order
output_series = CWV.output_series
logged_wet_volumes = CWV.logged_wet_volumes
read_level_set = CWV.read_level_set

MIN_FIT_PERIODS = 1.0             # frequency/damping need at least one inviscid period
MIN_FIT_SAMPLES = 8
MAX_MOMENT = 4                    # harmonic moments m = 0..4 of the liquid region

# Four-point Gauss-Legendre rule on [0, 1]: exact for polynomials of degree 7,
# so for the edge integrals of z^m conj(z) with m <= 6.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)
_GL_T = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


# ---------------------------------------------------------------------------
# Harmonic moments of the P1 liquid region {phi_h < 0} on affine triangles
# ---------------------------------------------------------------------------
def _edge_moments(za: np.ndarray, zb: np.ndarray, max_order: int) -> np.ndarray:
    """(1/2i) int_a^b z^m conj(z) dz along straight edges, m = 0..max_order.

    By the complex Green formula, int_R z^m dA = (1/2i) oint z^m conj(z) dz over
    the boundary of R (counter-clockwise); along a straight edge the integrand
    is a polynomial of degree m + 1 in the edge parameter.
    """
    dz = zb - za
    z = za[:, None] + dz[:, None] * _GL_T[None, :]
    zc = np.conj(z) * _GL_W[None, :]
    out = np.empty((za.size, max_order + 1), dtype=complex)
    zm = np.ones_like(z)
    for m in range(max_order + 1):
        out[:, m] = (zm * zc).sum(axis=1) * dz / 2j
        zm = zm * z
    return out


def liquid_moments(points: np.ndarray, tris: np.ndarray, phi: np.ndarray, origin,
                   max_order: int = MAX_MOMENT) -> np.ndarray:
    """M_m = int_{phi_h < 0} (z - origin)^m dA, m = 0..max_order, z = x + i y.

    Exact for the polygonal P1 region: full liquid triangles and the polygons
    clipped from cut triangles by the linear phi_h (the LinearCorner cut
    volume), each integrated over its edges.  M_0 is the area.
    """
    z = (points[:, 0] - origin[0]) + 1j * (points[:, 1] - origin[1])
    f = phi[tris]
    full = np.all(f <= 0.0, axis=1)
    cut = ~full & np.any(f < 0.0, axis=1)
    zt = z[tris[full]]
    acc = np.zeros((zt.shape[0], max_order + 1), dtype=complex)
    for a in range(3):
        acc += _edge_moments(zt[:, a], zt[:, (a + 1) % 3], max_order)
    sign = np.where(acc[:, 0].real >= 0.0, 1.0, -1.0)
    total = (sign[:, None] * acc).sum(axis=0)
    local = np.column_stack([z.real, z.imag])
    for idx in np.nonzero(cut)[0]:
        q = CWV._clip_negative(local[tris[idx]], f[idx])
        if len(q) < 3:
            continue
        qz = q[:, 0] + 1j * q[:, 1]
        m = _edge_moments(qz, np.roll(qz, -1), max_order).sum(axis=0)
        total += (1.0 if m[0].real >= 0.0 else -1.0) * m
    if not total[0].real > 0.0:
        raise DataError("no liquid: phi >= 0 at every vertex")
    return total


def central_moments(raw: np.ndarray) -> tuple[complex, np.ndarray]:
    """Centroid offset c = M_1 / M_0 and the moments about the centroid."""
    c = raw[1] / raw[0]
    out = np.array([sum(math.comb(m, k) * (-c) ** (m - k) * raw[k] for k in range(m + 1))
                    for m in range(raw.size)])
    return c, out


def shape_modes(central: np.ndarray) -> dict:
    """Shape amplitudes from the central harmonic moments.

    For r = R_A + sum_m (a_m cos(m theta) + b_m sin(m theta)) about the
    centroid, M_m = pi R_A^(m+1) (a_m + i b_m) + O(a^2), with R_A = sqrt(A/pi).
    For the released shape r = R0 (1 + eps cos(2 theta)) the mode-2 amplitude
    Re M_2 / (pi R_A^3) equals R0 eps to O(eps^4).
    """
    area = central[0].real
    r_a = math.sqrt(area / math.pi)
    modes = {m: central[m] / (math.pi * r_a ** (m + 1)) for m in range(2, central.size)}
    return {"area": area, "radius": r_a, "modes": modes}


def snapshot_measures(snap: dict, case: dict) -> dict:
    raw = liquid_moments(snap["points"], snap["tris"], snap["phi"], case["centre"])
    c, central = central_moments(raw)
    shape = shape_modes(central)
    n = case["mode"]
    a_n = shape["modes"][n]
    other = [abs(v) for m, v in shape["modes"].items() if m != n]
    return {"area": shape["area"], "amplitude": float(a_n.real), "amplitude_sin": float(a_n.imag),
            "centroid": (float(case["centre"][0] + c.real), float(case["centre"][1] + c.imag)),
            "other_modes_max": float(max(other)) if other else 0.0,
            "mode4": float(shape["modes"].get(4, 0.0).real) if 4 in shape["modes"] else 0.0}


# ---------------------------------------------------------------------------
# Solver log (reported only)
# ---------------------------------------------------------------------------
_STEP = re.compile(r"TimeLoop: nonlinear_done step=(\d+) .*?converged=(\d) iters=(\d+) "
                   r".*?outer_iters=(\d+)")
_NEWTON = re.compile(r"Total Newton time:\s+\S+ s\s+\((\d+) Newton iters, \d+ assemblies, "
                     r"(\d+) linear iters\)")


def solver_log_summary(run: Path) -> dict | None:
    """Outer passes, Newton and GMRES statistics and wall time (reported only)."""
    for name, opener in (("solver_run.log.gz", gzip.open), ("solver_run.log", open)):
        path = run / name
        if path.is_file():
            break
    else:
        return None
    steps, newton, linear = {}, 0, 0
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if "nonlinear_done" in line:
                m = _STEP.search(line)
                if m:
                    steps[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            elif "Total Newton time" in line:
                m = _NEWTON.search(line)
                if m:
                    newton += int(m.group(1))
                    linear += int(m.group(2))
    rows = [v for k, v in steps.items() if k >= 1]
    if not rows:
        return None
    summary = {"steps_logged": len(rows),
               "nonconverged_steps": sum(1 for r in rows if r[0] != 1),
               "newton_iterations_per_step": float(np.mean([r[1] for r in rows])),
               "outer_passes_mean": float(np.mean([r[2] for r in rows])),
               "outer_passes_max": int(max(r[2] for r in rows)),
               "linear_iterations_per_newton": (linear / newton) if newton else None}
    run_txt = run / "run.txt"
    if run_txt.is_file():
        m = re.search(r"elapsed_s=(\d+)", run_txt.read_text())
        if m:
            summary["wall_seconds"] = int(m.group(1))
            summary["wall_seconds_per_step"] = int(m.group(1)) / len(rows)
        m = re.search(r"ranks=(\d+)", run_txt.read_text())
        if m:
            summary["ranks"] = int(m.group(1))
    return summary


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def _reference(times, case: dict, radius: float) -> tuple[np.ndarray, dict]:
    params = dict(mode=case["mode"], kinematic_viscosity=case["kinematic_viscosity"],
                  surface_tension=case["surface_tension"], density=case["density"], radius=radius)
    amp = reference.drop_amplitude(times, initial_amplitude=case["initial_amplitude"], **params)
    return amp, reference.normal_mode(**params)


def analyse_run(run: Path) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    if case.get("benchmark") != "oscillating_drop_2d":
        raise DataError(f"{run}: case.json is not an oscillating_drop_2d case")
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
    centroid = np.array([m["centroid"] for _, m in rows])

    # D11: maximum area deviation over the whole run, outputs and logged steps.
    logged = logged_wet_volumes(run, case.get("interface_domain_id", "oscillating_drop_surface"))
    logged_drift = (max(abs(v - area[0]) for v in logged.values()) / area[0]) if logged else 0.0
    output_drift = float(np.max(np.abs(area - area[0])) / area[0])

    # Reference at the equilibrium radius of the discrete drop, R_eff =
    # sqrt(A_h(0)/pi) (as static_drop_2d); the nominal R is reported.
    r_eff = math.sqrt(area[0] / math.pi)
    amp_ref, mode = _reference(times, case, r_eff)
    amp_nom, mode_nom = _reference(times, case, case["radius"])
    omega0 = mode["omega0"]
    periods = float(times[-1] * omega0 / (2.0 * math.pi))
    a0 = case["initial_amplitude"]
    normalized_error = amp / amp[0] - amp_ref / a0

    result = {
        "run": str(run),
        "level": case["level_R_over_h"],
        "capillary_form": case["capillary_form"],
        "transport": case["transport"],
        "laplace_number": case["laplace_number"],
        "dt": case["dt"],
        "dt_divisor": case.get("dt_divisor", 1),
        "steps_per_period": float(case["inviscid_period"] / case["dt"]),
        "surface_tension_semi_implicit": case.get("surface_tension_semi_implicit", "None"),
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(times[-1]),
        "periods_simulated": periods,
        "outputs": int(len(times) - 1),
        "effective_radius": r_eff,
        "effective_radius_relative": r_eff / case["radius"] - 1.0,
        "initial_amplitude_discrete_relative": float(amp[0] / a0 - 1.0),
        "amplitude_rms_error": float(np.sqrt(np.mean(normalized_error ** 2))),
        "amplitude_max_error": float(np.max(np.abs(normalized_error))),
        "sin_mode_max_over_a0": float(max(abs(m["amplitude_sin"]) for _, m in rows) / a0),
        "other_modes_max_over_a0": float(max(m["other_modes_max"] for _, m in rows) / a0),
        "centroid_drift_max": float(np.max(np.hypot(*(centroid - centroid[0]).T))),
        "initial_liquid_area": float(area[0]),
        "liquid_area_relative_drift_max": max(output_drift, logged_drift),
        "liquid_area_relative_drift_max_outputs": output_drift,
        "liquid_area_logged_steps": len(logged),
        "reference_omega0": omega0,
        "reference_weak_damping_rate": mode["weak_damping_rate"],
        "reference_normal_mode_omega": mode["omega"],
        "reference_normal_mode_damping_rate": mode["beta"],
        "reference_nominal_radius_normal_mode_omega": mode_nom["omega"],
        "reference_nominal_radius_normal_mode_damping_rate": mode_nom["beta"],
        "fit": None, "reference_fit": None, "reference_fit_nominal_radius": None,
        "frequency_relative_error": None, "damping_rate_relative_error": None,
        "frequency_signed_error": None, "damping_rate_signed_error": None,
        "frequency_error_nominal_radius": None, "damping_rate_error_nominal_radius": None,
        "solver_log": solver_log_summary(run),
        "history": {"time": times.tolist(), "amplitude": amp.tolist(),
                    "reference_amplitude": amp_ref.tolist(), "liquid_area": area.tolist(),
                    "amplitude_sin": [m["amplitude_sin"] for _, m in rows],
                    "mode4": [m["mode4"] for _, m in rows],
                    "centroid_x": centroid[:, 0].tolist(), "centroid_y": centroid[:, 1].tolist()},
    }
    if periods >= MIN_FIT_PERIODS and len(times) >= MIN_FIT_SAMPLES:
        fit_ref = fit_damped_cosine(times, amp_ref, omega0)
        fit_nom = fit_damped_cosine(times, amp_nom, omega0)
        fit_sim = fit_damped_cosine(times, amp, omega0)
        for fit in (fit_ref, fit_nom):
            if not (fit["converged"] and fit["beta"] > 0.0):
                raise DataError(f"{run}: fit of the reference history failed: {fit}")
        result["fit"], result["reference_fit"] = fit_sim, fit_ref
        result["reference_fit_nominal_radius"] = fit_nom
        if fit_sim["converged"]:
            dw = (fit_sim["omega"] - fit_ref["omega"]) / fit_ref["omega"]
            db = (fit_sim["beta"] - fit_ref["beta"]) / fit_ref["beta"]
            result.update(frequency_relative_error=abs(dw), damping_rate_relative_error=abs(db),
                          frequency_signed_error=dw, damping_rate_signed_error=db,
                          frequency_error_nominal_radius=abs(fit_sim["omega"] - fit_nom["omega"])
                          / fit_nom["omega"],
                          damping_rate_error_nominal_radius=abs(fit_sim["beta"] - fit_nom["beta"])
                          / fit_nom["beta"])
    return result


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def evaluate_group(runs: list[dict], tolerances: dict,
                   required_levels: set | None = None) -> list[dict]:
    """Apply the criteria to one refinement study.

    required_levels (default: every protocol level) are the levels this study
    must contain; a criterion at a level outside it that has no run is
    reported as not run, without failing.
    """
    by_level = {r["level"]: r for r in runs}
    if required_levels is None:
        required_levels = set(tolerances["levels"]["R_over_h"])
    results = []
    for crit in tolerances["criteria"]:
        q, limit = crit["quantity"], crit.get("limit")
        messages, ok = [], True
        at = crit.get("at_level", "each")
        levels = sorted(by_level) if at == "each" else (list(at) if isinstance(at, list) else [at])
        for level in levels:
            if level not in by_level:
                if level in required_levels:
                    ok = False
                    messages.append(f"missing run at R/h={level}")
                else:
                    messages.append(f"R/h={level}: not run at this step (not required)")
                continue
            value = by_level[level][q]
            if value is None:
                ok = False
                messages.append(f"R/h={level}: not evaluable (history shorter than "
                                f"{MIN_FIT_PERIODS:g} period or fit failed)")
                continue
            passed = value <= limit
            ok &= passed
            messages.append(f"R/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
        if "minimum_observed_order" in crit:
            need = crit["order_levels"]
            errs = [by_level[lv][q] if lv in by_level else None for lv in need]
            if not set(need) <= set(by_level) | required_levels:
                messages.append(f"order not evaluated at this step (needs R/h={need})")
            elif any(e is None for e in errs):
                ok = False
                messages.append(f"order needs evaluable runs at R/h={need}")
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


def required_levels_at(divisor: int, tolerances: dict) -> set:
    """Levels a study at this dt divisor must contain.

    Divisor 1: every level.  Divisors of the time-step criterion (2): its
    refined_step_levels.  Other divisors (4): none; their criteria are applied
    at the levels they contain.
    """
    levels = set(tolerances["levels"]["R_over_h"])
    if divisor == 1:
        return levels
    crit = tolerances.get("time_step_criterion", {})
    if divisor not in crit.get("dt_divisors", [1, 2]):
        return set()
    rule = crit.get("refined_step_levels", "all")
    return levels if rule == "all" else set(rule)


def evaluate_time_step(groups: dict, tolerances: dict) -> list[dict]:
    """Time-step criterion for every (capillary form, transport, Laplace number).

    Compares the fitted frequency and damping rate of the divisor-1 and
    divisor-2 runs at their finest common level: |q(dt) - q(dt/2)| / q(dt/2)
    within the limit of each quantity.  With only one divisor, or no common
    level, the criterion is not evaluated and does not fail.
    """
    crit = tolerances.get("time_step_criterion")
    if crit is None:
        return []
    out = []
    for key in sorted({k[:3] for k in groups}):
        coarse, fine = groups.get((*key, 1)), groups.get((*key, 2))
        entry = {"id": crit["id"], "capillary_form": key[0], "transport": key[1],
                 "laplace_number": key[2], "evaluated": False, "passed": True,
                 "level": None, "changes": {}, "details": []}
        common = (sorted({r["level"] for r in coarse} & {r["level"] for r in fine})
                  if coarse and fine else [])
        if not common:
            present = sorted({k[3] for k in groups if k[:3] == key})
            entry["details"].append("not evaluated: needs runs at dt divisors 1 and 2 at a "
                                    f"common level (divisors present: {present})")
            out.append(entry)
            continue
        level = common[-1]
        a = next(r for r in coarse if r["level"] == level)
        b = next(r for r in fine if r["level"] == level)
        entry.update(evaluated=True, level=level, dt=[a["dt"], b["dt"]])
        fa, fb = a["fit"], b["fit"]
        messages = [f"R/h={level} (finest common level), dt={a['dt']:.4g} vs {b['dt']:.4g}"]
        ok = True
        for item in crit["quantities"]:
            name, label, limit = item["fit_parameter"], item["id"], item["limit"]
            if not fa or not fb or not fa.get("converged") or not fb.get("converged"):
                ok = False
                messages.append(f"{label}: not evaluable (fit missing or not converged)")
                continue
            change = abs(fa[name] - fb[name]) / abs(fb[name])
            passed = change <= limit
            ok &= passed
            entry["changes"][label] = change
            messages.append(f"{label} change {change:.3e} {'<=' if passed else '>'} {limit:g}")
        entry["passed"] = bool(ok)
        entry["details"] = messages
        out.append(entry)
    return out


def _log_text(r: dict) -> str:
    log = r.get("solver_log") or {}
    if not log:
        return "no solver log"
    wall = (f", wall {log['wall_seconds']} s ({log['wall_seconds_per_step']:.2f} s/step"
            f"{', ' + str(log['ranks']) + ' ranks' if 'ranks' in log else ''})"
            if "wall_seconds" in log else "")
    gmres = _fmt(log.get("linear_iterations_per_newton"), ".0f")
    return (f"{log['steps_logged']} steps, outer passes {log['outer_passes_mean']:.2f} "
            f"(max {log['outer_passes_max']}), Newton {log['newton_iterations_per_step']:.2f}/step, "
            f"GMRES {gmres}/Newton, non-converged {log['nonconverged_steps']}{wall}")


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
        groups.setdefault((a["capillary_form"], a["transport"], a["laplace_number"],
                           a["dt_divisor"]), []).append(a)
    levels = set(tolerances["levels"]["R_over_h"])
    report, all_pass = [], True
    for key in sorted({k[:3] for k in groups}):
        if (*key, 1) not in groups:
            all_pass = False
            print(f"no run at the protocol time step (dt divisor 1) for {key[0]}, transport "
                  f"{key[1]}, La = {key[2]:g}: the criteria cannot be applied")
    for (form, transport, laplace, divisor), runs in sorted(groups.items()):
        seen = [r["level"] for r in runs]
        if len(seen) != len(set(seen)) or not set(seen) <= levels:
            print(f"ERROR: group {form}, {transport}, La={laplace:g}, dt divisor {divisor}: "
                  f"duplicate or unknown levels {seen}", file=sys.stderr)
            return 2
        steps = sorted({r["dt"] for r in runs})
        if steps[-1] > steps[0] * (1.0 + 1e-12):
            print(f"ERROR: group {form}, {transport}, La={laplace:g}, dt divisor {divisor}: the "
                  f"levels use different time steps {steps}; the spatial study needs one shared "
                  "step (D10)", file=sys.stderr)
            return 2
        runs.sort(key=lambda r: r["level"])
        verdicts = evaluate_group(runs, tolerances, required_levels_at(divisor, tolerances))
        all_pass &= all(v["passed"] for v in verdicts)
        print(f"\n== {form}, transport {transport}, La = {laplace:g}, dt divisor {divisor}, "
              f"dt = {runs[0]['dt']:.6g} ({runs[0]['steps_per_period']:.4g} steps per period), "
              f"semi-implicit {'/'.join(sorted({r['surface_tension_semi_implicit'] for r in runs}))}"
              + ("  [TRUNCATED SMOKE RUNS]" if any(r["truncated"] for r in runs) else ""))
        print(f"{'R/h':>4} {'periods':>7} {'omega':>9} {'omega err':>10} {'beta':>8} "
              f"{'beta err':>10} {'rms err':>8} {'dA/A max':>9} {'a_h(0)/a0-1':>11} {'R_eff/R-1':>10}")
        for r in runs:
            fit = r["fit"] or {}
            print(f"{r['level']:>4} {r['periods_simulated']:>7.3g} {_fmt(fit.get('omega'), '9.5g')} "
                  f"{_fmt(r['frequency_signed_error'], '+10.2e')} {_fmt(fit.get('beta'), '8.4g')} "
                  f"{_fmt(r['damping_rate_signed_error'], '+10.2e')} {r['amplitude_rms_error']:>8.2e} "
                  f"{r['liquid_area_relative_drift_max']:>9.2e} "
                  f"{r['initial_amplitude_discrete_relative']:>11.2e} "
                  f"{r['effective_radius_relative']:>10.2e}")
        for r in runs:
            print(f"  R/h={r['level']}: {_log_text(r)}; sin(2 theta) max {r['sin_mode_max_over_a0']:.2e} a0, "
                  f"other modes max {r['other_modes_max_over_a0']:.2e} a0, centroid drift "
                  f"{r['centroid_drift_max']:.2e}")
        ref = runs[-1]
        ref_fit = ref["reference_fit"] or {}
        print(f"  reference (exact linear solution at R_eff, fitted on the same samples, finest "
              f"level): omega = {_fmt(ref_fit.get('omega'), '.6g')}, beta = "
              f"{_fmt(ref_fit.get('beta'), '.6g')}; normal mode omega = "
              f"{ref['reference_normal_mode_omega']:.6g}, beta = "
              f"{ref['reference_normal_mode_damping_rate']:.6g}; inviscid omega0 = "
              f"{ref['reference_omega0']:.6g}, 2 n (n-1) nu/R^2 = {ref['reference_weak_damping_rate']:.6g}")
        for v in verdicts:
            print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
        report.append({"capillary_form": form, "transport": transport,
                       "laplace_number": laplace, "dt_divisor": divisor,
                       "runs": runs, "criteria": verdicts,
                       "passed": all(v["passed"] for v in verdicts)})

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
              f"R/h = {level} (reported; change relative to dt/{runs[-1]['dt_divisor']})")
        rows = []
        for r in runs:
            fit = r["fit"] or {}
            d_omega = (abs(fit["omega"] - finest["omega"]) / finest["omega"]
                       if fit and finest else None)
            d_beta = (abs(fit["beta"] - finest["beta"]) / finest["beta"]
                      if fit and finest else None)
            print(f"   dt/{r['dt_divisor']}: omega err {_fmt(r['frequency_signed_error'], '+.3e')}, "
                  f"beta err {_fmt(r['damping_rate_signed_error'], '+.3e')}, "
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
    time_step = evaluate_time_step(groups, tolerances)
    for t in time_step:
        all_pass &= t["passed"]
        tag = ("PASS" if t["passed"] else "FAIL") if t["evaluated"] else "NOT EVALUATED"
        print(f"\n-- time-step criterion {t['capillary_form']}, transport {t['transport']}, "
              f"La = {t['laplace_number']:g}")
        print(f"  [{tag}] {t['id']}: " + "; ".join(t["details"]))
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"],
                                         "groups": report, "time_step_study": temporal,
                                         "time_step_criterion": time_step,
                                         "passed": bool(all_pass)}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
