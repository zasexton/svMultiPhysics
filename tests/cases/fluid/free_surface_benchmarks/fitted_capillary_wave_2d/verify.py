#!/usr/bin/env python3
"""Compute the fitted_capillary_wave_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT] [--allow-truncated]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI).  The
protocol runs form a spatial study (lambda/h = 16, 32, 64 at the shared
capillary_wave_2d time step) and a separate time-step study (lambda/h = 32 at
dt, dt/2, dt/4); the time error of the shared step is measured by the
second and removed from the first (decision D10).  Metric definitions are in
README.md.

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
MIN_FIT_PERIODS = 1.0             # frequency/damping need at least one inviscid period
MIN_FIT_SAMPLES = 8


def _load(directory: str, name: str):
    path = HERE.parent / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}_for_fitted_cw", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CWV = _load("capillary_wave_2d", "verify")          # moments, fit, reference, output series
FSV = _load("fitted_sloshing_2d", "verify")         # moving-mesh snapshot reader, log summary
reference = CWV.reference                           # prosperetti_reference module
DataError = CWV.DataError
fit_damped_cosine = CWV.fit_damped_cosine
observed_order = CWV.observed_order


# ---------------------------------------------------------------------------
# Geometry of the moving liquid mesh
# ---------------------------------------------------------------------------
def mesh_measures(points: np.ndarray, tris: np.ndarray, wavenumber: float) -> tuple[float, float]:
    """Exact area of the liquid mesh and exact integral of cos(kx) over it.

    Every triangle is liquid; the integral is summed over the triangle edges
    with Green's theorem (capillary_wave_2d/verify.py), so tangential motion
    of the surface nodes is handled exactly.
    """
    p = points[tris]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if np.any(signed <= 0.0) and np.any(signed >= 0.0):
        raise DataError("inverted or degenerate triangle in the current mesh")
    return CWV.liquid_measures(points, tris, -np.ones(points.shape[0]), wavenumber)


def minimum_angle_degrees(points: np.ndarray, tris: np.ndarray) -> float:
    """Smallest interior angle of the current mesh (reported mesh quality)."""
    p = points[tris]
    worst = math.pi
    for a in range(3):
        u = p[:, (a + 1) % 3] - p[:, a]
        v = p[:, (a + 2) % 3] - p[:, a]
        cosang = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        worst = min(worst, float(np.min(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    return math.degrees(worst)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def analyse_run(run: Path, *, allow_short: bool = False) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    if case.get("benchmark") != "fitted_capillary_wave_2d":
        raise DataError(f"{run}: case.json is not a fitted_capillary_wave_2d case")
    series = CWV.output_series(run, case)
    if abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={case['end_time']:.6g}")
    k, width, a0 = case["wavenumber"], case["width"], case["initial_amplitude"]
    snaps = [(0.0, run / "mesh" / "mesh-complete.mesh.vtu", True)] + \
            [(t, path, False) for t, path in series if t > 0.0]
    times, amp, areas, speeds, walls, wall_left, wall_right, angles = ([] for _ in range(8))
    for t, path, is_reference in snaps:
        snap = FSV.read_snapshot(path, case, reference=is_reference)
        area, moment = mesh_measures(snap["points"], snap["tris"], k)
        times.append(t)
        areas.append(area)
        amp.append(CWV.mode_amplitude(moment, width, k))
        speeds.append(float(np.max(np.linalg.norm(snap["velocity"], axis=1))))
        surface = snap["surface"]
        walls.append(float(max(abs(surface[0, 0]), abs(surface[-1, 0] - width))))
        mean = area / width
        wall_left.append(float(surface[0, 1] - mean))
        wall_right.append(float(surface[-1, 1] - mean))
        angles.append(minimum_angle_degrees(snap["points"], snap["tris"]))
    times, amp, areas = map(np.asarray, (times, amp, areas))

    params = dict(wavenumber=k, kinematic_viscosity=case["kinematic_viscosity"],
                  surface_tension=case["surface_tension"], density=case["density"])
    amp_ref = reference.prosperetti_amplitude(times, initial_amplitude=a0, **params)
    mode = reference.normal_mode(**params)
    omega0 = mode["omega0"]
    periods = float(times[-1] * omega0 / (2.0 * math.pi))
    if (periods < MIN_FIT_PERIODS or len(times) < MIN_FIT_SAMPLES) and not allow_short:
        raise DataError(f"{run}: {periods:.2f} periods is too short for the frequency fit "
                        "(protocol: 4); use --allow-truncated for smoke runs")
    normalized_error = amp / amp[0] - amp_ref / a0
    area0 = areas[0]
    result = {
        "run": str(run),
        "level": case["level_lambda_over_h"],
        "dt": case["dt"],
        "dt_divisor": case["dt_divisor"],
        "dt_over_capillary_limit": case["dt_over_capillary_limit"],
        "study": case.get("study"),
        "protocol_run": bool(case.get("protocol_run", False)),
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(times[-1]),
        "periods_simulated": periods,
        "outputs": int(times.size - 1),
        "n_vertices": case["n_vertices"],
        "initial_amplitude_discrete_relative": float(amp[0] / a0 - 1.0),
        "amplitude_rms_error": float(np.sqrt(np.mean(normalized_error ** 2))),
        "amplitude_max_error": float(np.max(np.abs(normalized_error))),
        "initial_liquid_area": float(area0),
        "liquid_area_relative_deviation_max": float(np.max(np.abs(areas - area0)) / area0),
        "liquid_area_relative_deviation_final": float(abs(areas[-1] - area0) / area0),
        "mean_level_drift_over_amplitude_max": float(np.max(np.abs(areas - area0)) / width / a0),
        "surface_end_node_wall_offset_max": float(max(walls)),
        "minimum_angle_deg_min": float(min(angles)),
        "minimum_angle_deg_initial": float(angles[0]),
        "max_liquid_speed": float(max(speeds)),
        "reference_omega0": omega0,
        "reference_normal_mode_omega": mode["omega"],
        "reference_normal_mode_damping_rate": mode["beta"],
        "fit": None, "reference_fit": None, "omega": None, "beta": None,
        "frequency_relative_error": None, "damping_rate_relative_error": None,
        "frequency_signed_error": None, "damping_rate_signed_error": None,
        "solver_log": FSV.solver_log_summary(run),
        "history": {"time": times.tolist(), "amplitude": amp.tolist(),
                    "reference_amplitude": amp_ref.tolist(), "liquid_area": areas.tolist(),
                    "wall_elevation_left": wall_left, "wall_elevation_right": wall_right,
                    "minimum_angle_deg": angles, "max_liquid_speed": speeds},
    }
    if periods >= MIN_FIT_PERIODS and len(times) >= MIN_FIT_SAMPLES:
        fit_ref = fit_damped_cosine(times, amp_ref, omega0)
        fit_sim = fit_damped_cosine(times, amp, omega0)
        if not (fit_ref["converged"] and fit_ref["beta"] > 0.0):
            raise DataError(f"{run}: fit of the reference history failed: {fit_ref}")
        result["fit"], result["reference_fit"] = fit_sim, fit_ref
        result["omega_reference"], result["beta_reference"] = fit_ref["omega"], fit_ref["beta"]
        if fit_sim["converged"]:
            result["omega"], result["beta"] = fit_sim["omega"], fit_sim["beta"]
            result["frequency_signed_error"] = fit_sim["omega"] / fit_ref["omega"] - 1.0
            result["damping_rate_signed_error"] = fit_sim["beta"] / fit_ref["beta"] - 1.0
            result["frequency_relative_error"] = abs(result["frequency_signed_error"])
            result["damping_rate_relative_error"] = abs(result["damping_rate_signed_error"])
    return result


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
def richardson(steps: list[float], values: list[float]) -> dict:
    """Extrapolate values(dt) to dt -> 0 from three or more nested steps.

    The observed order p comes from the three smallest steps; the limit is
    extrapolated from the two smallest.  Without monotone differences the
    smallest-step value is taken as the limit and p is NaN.
    """
    pts = sorted(zip(steps, values), key=lambda x: -x[0])       # decreasing dt
    s = [p[0] for p in pts]
    v = [p[1] for p in pts]
    d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
    ratio = s[-2] / s[-1]
    if d1 == 0.0 or d2 == 0.0 or d1 * d2 < 0.0:
        return {"order": float("nan"), "limit": v[-1]}
    order = math.log(abs(d1 / d2)) / math.log(ratio)
    return {"order": order, "limit": v[-1] + d2 / (ratio ** order - 1.0)}


def time_step_study(runs: list[dict]) -> dict | None:
    """Time error of the frequency and damping at the time-study mesh (D10)."""
    usable = [r for r in runs if r["omega"] is not None]
    if len({r["dt_divisor"] for r in usable}) < 3:
        return None
    usable.sort(key=lambda r: r["dt_divisor"])
    dts = [r["dt"] for r in usable]
    omega = richardson(dts, [r["omega"] for r in usable])
    beta = richardson(dts, [r["beta"] for r in usable])
    return {
        "dt_divisors": [r["dt_divisor"] for r in usable],
        "omega": [r["omega"] for r in usable], "beta": [r["beta"] for r in usable],
        "observed_order_omega": omega["order"], "observed_order_beta": beta["order"],
        "omega_dt_to_zero": omega["limit"], "beta_dt_to_zero": beta["limit"],
        "omega_time_error": {r["dt_divisor"]: r["omega"] - omega["limit"] for r in usable},
        "beta_time_error": {r["dt_divisor"]: r["beta"] - beta["limit"] for r in usable},
    }


def evaluate(runs: list[dict], tolerances: dict) -> tuple[list[dict], dict]:
    protocol = tolerances["protocol"]
    spatial = {r["level"]: r for r in runs if r["study"] == "spatial"}
    timing = [r for r in runs
              if r["level"] == protocol["time_step_study_level"]
              and r["dt_divisor"] in protocol["time_step_study_dt_divisors"]
              and r["study"] in ("spatial", "time")]
    dts = time_step_study(timing)
    summary = {"time_step_study": dts}
    for r in spatial.values():
        if dts is None or r["omega"] is None:
            r["time_error_removed"] = None
            r["frequency_spatial_relative_error"] = None
            r["damping_spatial_relative_error"] = None
            continue
        e_omega, e_beta = dts["omega_time_error"][1], dts["beta_time_error"][1]
        r["time_error_removed"] = {"omega": e_omega, "beta": e_beta}
        r["frequency_spatial_signed_error"] = (r["omega"] - e_omega) / r["omega_reference"] - 1.0
        r["damping_spatial_signed_error"] = (r["beta"] - e_beta) / r["beta_reference"] - 1.0
        r["frequency_spatial_relative_error"] = abs(r["frequency_spatial_signed_error"])
        r["damping_spatial_relative_error"] = abs(r["damping_spatial_signed_error"])

    results = []
    for crit in tolerances["criteria"]:
        q, limit = crit["quantity"], crit.get("limit")
        ok, messages = True, []
        if crit["study"] == "all":
            pool = list({r["run"]: r for r in list(spatial.values()) + timing}.values())
            if not pool:
                ok = False
                messages.append("no protocol runs")
            for r in sorted(pool, key=lambda r: (r["level"], r["dt_divisor"])):
                value = r[q]
                passed = value <= limit
                ok &= passed
                messages.append(f"lambda/h={r['level']} dt/{r['dt_divisor']}: {value:.3g} "
                                f"{'<=' if passed else '>'} {limit:g}")
        else:
            at = crit.get("at_level", [])
            for level in (at if isinstance(at, list) else [at]):
                value = spatial[level].get(q) if level in spatial else None
                if value is None:
                    ok = False
                    messages.append(f"missing {q} at lambda/h={level}")
                    continue
                passed = value <= limit
                ok &= passed
                messages.append(f"lambda/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
            if "minimum_observed_order" in crit:
                need = crit["order_levels"]
                errs = [spatial[lv].get(q) if lv in spatial else None for lv in need]
                if any(e is None for e in errs):
                    ok = False
                    messages.append(f"order needs lambda/h={need}")
                elif min(errs) <= 0.0:
                    ok = False
                    messages.append("order undefined for a zero error")
                else:
                    order = observed_order(need, errs)
                    pairs = ", ".join(f"{observed_order(need[i:i + 2], errs[i:i + 2]):.2f}"
                                      for i in range(len(need) - 1))
                    passed = order >= crit["minimum_observed_order"]
                    ok &= passed
                    messages.append(f"observed order {order:.2f} (pairwise {pairs}) "
                                    f"{'>=' if passed else '<'} {crit['minimum_observed_order']}")
        results.append({"id": crit["id"], "quantity": q, "passed": bool(ok), "details": messages})
    return results, summary


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
        runs = [analyse_run(run, allow_short=args.allow_truncated) for run in args.runs]
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    truncated = [r["run"] for r in runs if r["truncated"]]
    if truncated and not args.allow_truncated:
        print("ERROR: truncated smoke runs are not acceptance evidence: " + ", ".join(truncated),
              file=sys.stderr)
        return 2
    protocol = [r for r in runs if r["protocol_run"] and not r["truncated"]]
    keys = [(r["level"], r["dt_divisor"]) for r in protocol]
    if len(keys) != len(set(keys)):
        print(f"ERROR: duplicate protocol runs {keys}", file=sys.stderr)
        return 2
    verdicts, summary = evaluate(protocol, tolerances)
    all_pass = all(v["passed"] for v in verdicts)

    everything = sorted(runs, key=lambda r: (not r["protocol_run"], r["level"], r["dt_divisor"],
                                             r["dt"]))
    print(f"{'l/h':>4} {'dt':>9} {'dt/lim':>6} {'study':>7} {'omega':>10} {'omega err':>10} "
          f"{'spatial':>10} {'beta':>8} {'beta err':>10} {'spatial':>10} {'rms err':>8} "
          f"{'dA/A max':>9} {'min ang':>7} {'Newton':>6} {'s/step':>6}")
    for r in everything:
        log = r["solver_log"] or {}
        print(f"{r['level']:>4} {r['dt']:>9.3e} {r['dt_over_capillary_limit']:>6.2f} "
              f"{str(r['study']):>7} {_fmt(r['omega'], '10.6f')} "
              f"{_fmt(r['frequency_signed_error'], '+10.3e')} "
              f"{_fmt(r.get('frequency_spatial_signed_error'), '+10.3e')} "
              f"{_fmt(r['beta'], '8.5f')} {_fmt(r['damping_rate_signed_error'], '+10.3e')} "
              f"{_fmt(r.get('damping_spatial_signed_error'), '+10.3e')} "
              f"{r['amplitude_rms_error']:>8.2e} {r['liquid_area_relative_deviation_max']:>9.2e} "
              f"{r['minimum_angle_deg_min']:>7.2f} "
              f"{log.get('newton_iterations_mean', float('nan')):>6.2f} "
              f"{log.get('wall_seconds_per_step', float('nan')):>6.2f}"
              + ("  [TRUNCATED SMOKE RUN]" if r["truncated"] else "")
              + ("" if r["protocol_run"] else "  [diagnostic, not gated]"))
    ref = everything[0]
    ref_fit = ref["reference_fit"] or {}
    print(f"reference (Prosperetti, fitted on the same samples): omega = "
          f"{_fmt(ref_fit.get('omega'), '.7g')}, beta = {_fmt(ref_fit.get('beta'), '.7g')}; "
          f"normal mode omega = {ref['reference_normal_mode_omega']:.7g}, beta = "
          f"{ref['reference_normal_mode_damping_rate']:.7g}; omega0 = {ref['reference_omega0']:.7g}")
    dts = summary["time_step_study"]
    if dts is not None:
        print("time-step study (lambda/h = {}): omega(dt->0) = {:.8f} (order {:.2f}), "
              "beta(dt->0) = {:.6f} (order {:.2f}); time errors of the shared step: "
              "omega {:+.2e}, beta {:+.2e} (relative)".format(
                  tolerances["protocol"]["time_step_study_level"],
                  dts["omega_dt_to_zero"], dts["observed_order_omega"],
                  dts["beta_dt_to_zero"], dts["observed_order_beta"],
                  dts["omega_time_error"][1] / ref_fit.get("omega", 1.0),
                  dts["beta_time_error"][1] / ref_fit.get("beta", 1.0)))
    for v in verdicts:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
    if args.json:
        args.json.write_text(json.dumps({"benchmark": tolerances["benchmark"], "runs": everything,
                                         "summary": summary, "criteria": verdicts,
                                         "passed": bool(all_pass)}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
