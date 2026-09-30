#!/usr/bin/env python3
"""Compute the fitted_sloshing_2d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT]

Each RUN_DIR is a case written by generate_case.py (it holds case.json and
mesh/mesh-complete.mesh.vtu) in which the solver has run, leaving
result.pvd and result_NNN.vtu (serial) or result_NNN.pvtu (MPI).  The
protocol runs form a spatial study (L/h = 16, 32, 64 at a fixed small time
step) and a separate time-step study (L/h = 32); the criteria are applied
across them.  Metric definitions are in README.md.

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
TRIANGLE = 5                      # VTK cell type
MODAL_FIT_MODES = 4               # surface fit y = sum_{n<4} a_n cos(n k x)


def _load_linear_sloshing_verify():
    path = HERE.parent / "linear_sloshing_2d" / "verify.py"
    spec = importlib.util.spec_from_file_location("linear_sloshing_2d_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LSV = _load_linear_sloshing_verify()
DataError = LSV.DataError
fit_damped_oscillation = LSV.fit_damped_oscillation
observed_order = LSV.observed_order
output_series = LSV.output_series
solver_log_summary = LSV.solver_log_summary


# ---------------------------------------------------------------------------
# Geometry of the moving liquid mesh
# ---------------------------------------------------------------------------
def mesh_area(points: np.ndarray, tris: np.ndarray) -> float:
    """Area of the liquid mesh (sum of the triangle areas)."""
    p = points[tris]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if np.any(signed <= 0.0) and np.any(signed >= 0.0):
        raise DataError("inverted or degenerate triangle in the current mesh")
    return float(np.sum(np.abs(signed)))


def probe_elevation(surface_points: np.ndarray, x_probe: float) -> float:
    """Height of the free surface at x = x_probe, linear between surface nodes."""
    order = np.argsort(surface_points[:, 0])
    x, y = surface_points[order, 0], surface_points[order, 1]
    if not (x[0] - 1e-12 <= x_probe <= x[-1] + 1e-12):
        raise DataError(f"probe x={x_probe} lies outside the free surface [{x[0]}, {x[-1]}]")
    return float(np.interp(x_probe, x, y))


def modal_coefficients(surface_points: np.ndarray, k: float,
                       modes: int = MODAL_FIT_MODES) -> np.ndarray:
    """Least-squares fit y = sum_n a_n cos(n k x) to the free-surface nodes."""
    x, y = surface_points[:, 0], surface_points[:, 1]
    if len(x) < 2 * modes:
        raise DataError("too few free-surface nodes for the modal fit")
    basis = np.column_stack([np.cos(n * k * x) for n in range(modes)])
    return np.linalg.lstsq(basis, y, rcond=None)[0]


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------
def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:     # pragma: no cover - environment dependent
        raise DataError("pyvista is required to read VTU/PVTU output") from exc
    return pv


def current_points(grid, case: dict, reference: bool) -> np.ndarray:
    """Current vertex positions: reference points plus the mesh displacement.

    The solver writes the reference configuration as VTK points and the
    displacement as point data; the mesh file at t = 0 is the reference.
    """
    points = np.asarray(grid.points, dtype=float)[:, :2]
    if reference:
        return points
    if "CurrentCoordinates" in grid.point_data:
        cur = np.asarray(grid.point_data["CurrentCoordinates"], dtype=float)
        return cur.reshape(grid.n_points, -1)[:, :2]
    name = case["displacement_field"]
    if name not in grid.point_data:
        raise DataError(f"point array '{name}' is missing (and no CurrentCoordinates)")
    disp = np.asarray(grid.point_data[name], dtype=float).reshape(grid.n_points, -1)
    return points + disp[:, :2]


def read_snapshot(path: Path, case: dict, *, reference: bool = False) -> dict:
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
    if "GlobalNodeID" not in grid.point_data:
        raise DataError(f"{path}: point array 'GlobalNodeID' is missing")
    gid = np.asarray(grid.point_data["GlobalNodeID"]).astype(np.int64)
    points = current_points(grid, case, reference)
    name = case["velocity_field"]
    if name not in grid.point_data:
        raise DataError(f"{path}: point array '{name}' is missing")
    velocity = np.asarray(grid.point_data[name], dtype=float).reshape(grid.n_points, -1)[:, :2]
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(velocity))):
        raise DataError(f"{path}: non-finite values")
    # Map the free-surface node ids to local point indices (MPI output may
    # duplicate points; any copy has the same position).
    local = {}
    for index, g in enumerate(gid):
        local.setdefault(int(g), index)
    try:
        surface = np.array([local[g] for g in case["free_surface_nodes"]], dtype=np.int64)
    except KeyError as exc:
        raise DataError(f"{path}: free-surface node {exc} is missing") from exc
    return {"points": points, "tris": tris, "velocity": velocity,
            "surface": points[surface]}


def analyse_run(run: Path, *, allow_short: bool = False) -> dict:
    case_file = run / "case.json"
    if not case_file.is_file():
        raise DataError(f"{run}: case.json is missing (was the case written by generate_case.py?)")
    case = json.loads(case_file.read_text())
    series = output_series(run, case)
    if abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run, last output t={series[-1][0]:.6g} "
                        f"but the case ends at t={case['end_time']:.6g}")
    k, depth = case["wavenumber"], case["mean_depth"]
    snaps = [(0.0, run / "mesh" / "mesh-complete.mesh.vtu", True)] + \
            [(t, path, False) for t, path in series]
    times, probe, modal, areas, speeds, walls = [], [], [], [], [], []
    for t, path, reference in snaps:
        snap = read_snapshot(path, case, reference=reference)
        times.append(t)
        probe.append(probe_elevation(snap["surface"], case["probe_x"]) - depth)
        modal.append(modal_coefficients(snap["surface"], k)[1])
        areas.append(mesh_area(snap["points"], snap["tris"]))
        speeds.append(float(np.max(np.linalg.norm(snap["velocity"], axis=1))))
        walls.append(float(max(abs(snap["surface"][0, 0] - 0.0),
                               abs(snap["surface"][-1, 0] - case["tank_length"]))))
    times, probe, modal, areas = map(np.asarray, (times, probe, modal, areas))

    periods = times[-1] * case["omega_inviscid"] / (2.0 * math.pi)
    if periods < 2.0 and not allow_short:
        raise DataError(f"{run}: {periods:.2f} periods is too short for the frequency fit "
                        "(protocol: 4); use --allow-truncated for smoke runs")
    omega_ref, gamma_ref = case["omega_reference"], case["damping_rate_reference"]
    fit = fit_damped_oscillation(times, probe, omega_ref)
    fit_modal = fit_damped_oscillation(times, modal, omega_ref)
    area0 = areas[0]
    return {
        "run": str(run),
        "level": case["level_cells_per_length"],
        "steps_per_period": case["steps_per_period"],
        "study": case.get("study"),
        "protocol_run": bool(case.get("protocol_run", False)),
        "kinematic_enforcement": case.get("kinematic_enforcement"),
        "wall_mesh_motion": case.get("wall_mesh_motion"),
        "truncated": bool(case.get("truncated", False)),
        "end_time": float(times[-1]),
        "periods_simulated": float(periods),
        "outputs": int(times.size - 1),
        "n_vertices": case["n_vertices"],
        "omega_reference": omega_ref,
        "omega_inviscid": case["omega_inviscid"],
        "damping_rate_reference": gamma_ref,
        "damping_rate_lamb": case["damping_rate_lamb"],
        "omega": fit["omega"],
        "damping_rate": fit["damping_rate"],
        "frequency_relative_error": abs(fit["omega"] - omega_ref) / omega_ref,
        "frequency_signed_error": fit["omega"] / omega_ref - 1.0,
        "damping_rate_relative_error": abs(fit["damping_rate"] - gamma_ref) / gamma_ref,
        "damping_rate_over_reference": fit["damping_rate"] / gamma_ref,
        "damping_rate_over_lamb": fit["damping_rate"] / case["damping_rate_lamb"],
        "fit_amplitude_over_initial": fit["amplitude"] / case["amplitude"],
        "fit_offset": fit["offset"],
        "fit_rms_residual_over_amplitude": fit["rms_residual_over_amplitude"],
        "modal_omega": fit_modal["omega"],
        "modal_frequency_signed_error": fit_modal["omega"] / omega_ref - 1.0,
        "modal_damping_rate_over_reference": fit_modal["damping_rate"] / gamma_ref,
        "initial_probe_elevation_over_amplitude": float(probe[0] / case["amplitude"]),
        "initial_liquid_area": float(area0),
        "liquid_area_relative_deviation_max": float(np.max(np.abs(areas - area0)) / area0),
        "liquid_area_relative_deviation_final": float(abs(areas[-1] - area0) / area0),
        "surface_end_node_wall_offset_max": float(max(walls)),
        "max_liquid_speed": float(max(speeds)),
        "solver_log": solver_log_summary(run),
        "history": {"time": times.tolist(), "probe_elevation": probe.tolist(),
                    "modal_amplitude": modal.tolist(), "liquid_area": areas.tolist(),
                    "max_liquid_speed": speeds},
    }


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
def time_step_study(runs: list[dict]) -> dict | None:
    """Richardson extrapolation of omega(dt) at the fixed mesh of the time-step study.

    The observed order p comes from the three finest time steps; omega at
    dt -> 0 is extrapolated from the two finest.  e_t(S) = omega(S) - omega_0.
    """
    pts = sorted(((r["steps_per_period"], r["omega"]) for r in runs), key=lambda x: x[0])
    if len(pts) < 3:
        return None
    s = [p[0] for p in pts]
    w = [p[1] for p in pts]
    d1, d2 = w[-2] - w[-3], w[-1] - w[-2]
    if d1 == 0.0 or d2 == 0.0 or d1 * d2 < 0.0:
        order = float("nan")
        omega0 = w[-1]
    else:
        ratio = (s[-1] / s[-2])
        order = math.log(abs(d1 / d2)) / math.log(ratio)
        omega0 = w[-1] + d2 / (ratio ** order - 1.0)
    errors = {int(si): float(wi - omega0) for si, wi in zip(s, w)}
    return {"steps_per_period": s, "omega": w, "observed_order_richardson": order,
            "omega_dt_to_zero": omega0, "time_error": errors}


def evaluate(runs: list[dict], tolerances: dict) -> tuple[list[dict], dict]:
    protocol = tolerances["protocol"]
    spatial = {r["level"]: r for r in runs if r["study"] == "spatial"}
    # The spatial-study run on the time-study mesh is also the finest point
    # of the time-step study.
    timing = [r for r in runs
              if r["study"] == "time"
              or (r["study"] == "spatial"
                  and r["level"] == protocol["time_step_study_level"]
                  and r["steps_per_period"] in protocol["time_step_study_steps_per_period"])]
    dts = time_step_study(timing)
    summary = {"time_step_study": dts}
    # Time error of the spatial-study step, measured at the time-study mesh.
    s_space = protocol["spatial_study_steps_per_period"]
    e_t = None
    if dts is not None and s_space in dts["time_error"]:
        e_t = dts["time_error"][s_space]
    for r in spatial.values():
        r["time_error_removed"] = e_t
        if e_t is None:
            r["frequency_spatial_relative_error"] = None
        else:
            r["frequency_spatial_signed_error"] = (r["omega"] - e_t) / r["omega_reference"] - 1.0
            r["frequency_spatial_relative_error"] = abs(r["frequency_spatial_signed_error"])
    for r in timing:
        if dts is not None:
            r["frequency_time_error"] = abs(dts["time_error"][r["steps_per_period"]]) / r["omega_reference"]

    results = []
    for crit in tolerances["criteria"]:
        q, limit, study = crit["quantity"], crit.get("limit"), crit["study"]
        ok, messages = True, []
        if study == "time":
            by_key = {r["steps_per_period"]: r for r in timing}
            key_name = "steps/T"
        else:
            by_key = dict(spatial)
            key_name = "L/h"
        if study == "all":
            pool = list(spatial.values()) + timing
            for r in pool:
                value = r[q]
                passed = limit is None or value <= limit
                ok &= passed
                messages.append(f"L/h={r['level']} steps/T={r['steps_per_period']}: {value:.3g}"
                                + ("" if limit is None else (" <= " if passed else " > ") + f"{limit:g}"))
            if not pool:
                ok = False
                messages.append("no protocol runs")
        else:
            at = crit.get("at_level")
            keys = sorted(by_key) if at in (None, "each") else [at]
            if limit is not None:
                for key in keys:
                    if key not in by_key or by_key[key].get(q) is None:
                        ok = False
                        messages.append(f"missing {q} at {key_name}={key}")
                        continue
                    value = by_key[key][q]
                    passed = value <= limit
                    ok &= passed
                    messages.append(f"{key_name}={key}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
            if crit.get("monotone") == "strictly_decreasing":
                need = crit["monotone_levels"]
                vals = [by_key[k].get(q) if k in by_key else None for k in need]
                if any(v is None for v in vals):
                    ok = False
                    messages.append(f"monotonicity needs {key_name}={need}")
                else:
                    passed = all(b < a for a, b in zip(vals, vals[1:]))
                    ok &= passed
                    messages.append("decreasing over " + key_name + "=" + "/".join(map(str, need))
                                    + (": yes" if passed else ": NO (" +
                                       ", ".join(f"{v:.3g}" for v in vals) + ")"))
            if "minimum_observed_order" in crit:
                need = crit["order_levels"]
                vals = [by_key[k].get(q) if k in by_key else None for k in need]
                if any(v is None for v in vals):
                    ok = False
                    messages.append(f"order needs {key_name}={need}")
                elif min(vals) <= 0.0:
                    ok = False
                    messages.append("order undefined for a zero error")
                else:
                    order = observed_order(need, vals)
                    pairs = ", ".join(f"{observed_order(need[i:i + 2], vals[i:i + 2]):.2f}"
                                      for i in range(len(need) - 1))
                    passed = order >= crit["minimum_observed_order"]
                    ok &= passed
                    messages.append(f"observed order {order:.2f} (pairwise {pairs}) "
                                    f"{'>=' if passed else '<'} {crit['minimum_observed_order']}")
        results.append({"id": crit["id"], "quantity": q, "passed": bool(ok), "details": messages})
    return results, summary


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
    keys = [(r["study"], r["level"], r["steps_per_period"]) for r in protocol]
    if len(keys) != len(set(keys)):
        print(f"ERROR: duplicate protocol runs {keys}", file=sys.stderr)
        return 2
    verdicts, summary = evaluate(protocol, tolerances)
    all_pass = all(v["passed"] for v in verdicts)

    everything = sorted(runs, key=lambda r: (not r["protocol_run"], str(r["study"]),
                                             r["level"], r["steps_per_period"]))
    print(f"{'L/h':>4} {'steps/T':>7} {'study':>7} {'omega':>12} {'freq err':>10} {'spatial':>10} "
          f"{'g/g_ref':>8} {'A_fit/A':>8} {'fit rms':>8} {'dA/A max':>9} {'s/step':>7}")
    for r in everything:
        log = r["solver_log"] or {}
        spatial_err = r.get("frequency_spatial_signed_error")
        print(f"{r['level']:>4} {r['steps_per_period']:>7} {str(r['study']):>7} "
              f"{r['omega']:>12.8f} {r['frequency_signed_error']:>+10.3e} "
              f"{(f'{spatial_err:+10.3e}' if spatial_err is not None else '         -'):>10} "
              f"{r['damping_rate_over_reference']:>8.4f} {r['fit_amplitude_over_initial']:>8.4f} "
              f"{r['fit_rms_residual_over_amplitude']:>8.1e} "
              f"{r['liquid_area_relative_deviation_max']:>9.2e} "
              f"{log.get('wall_seconds_per_step', float('nan')):>7.2f}"
              + ("  [TRUNCATED SMOKE RUN]" if r["truncated"] else "")
              + ("" if r["protocol_run"] else
                 f"  [diagnostic ({r['kinematic_enforcement']}, walls {r['wall_mesh_motion']}), "
                 "not gated]"))
    print(f"reference: omega = {everything[0]['omega_reference']:.10f} (inviscid "
          f"{everything[0]['omega_inviscid']:.10f}), gamma = "
          f"{everything[0]['damping_rate_reference']:.6e}")
    dts = summary["time_step_study"]
    if dts is not None:
        print("time-step study: omega(dt->0) = {:.10f}, Richardson order {:.2f}; time errors ".format(
            dts["omega_dt_to_zero"], dts["observed_order_richardson"])
            + ", ".join(f"{s}: {e / everything[0]['omega_reference']:+.2e}"
                        for s, e in dts["time_error"].items()))
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
