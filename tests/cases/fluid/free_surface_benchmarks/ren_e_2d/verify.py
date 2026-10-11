#!/usr/bin/env python3
"""Measure and gate the 2D Ren-E benchmark runs (generate_case.py).

For every saved state (and the initial mesh) the two phi = 0 roots on the
contact wall y = 0 are located.  At each root:

* the dynamic angle theta_d (through the liquid) is taken from the P1
  gradient of phi in the wall triangle that holds the root edge, the
  generated LinearCorner fragment normal of that cell: cos(theta_d) = n_y for
  the outward liquid normal n = grad(phi)/|grad(phi)| and the wall normal
  (0, -1);
* the Ren-E prediction is V = gamma M (cos theta_e - cos theta_d), positive
  when the line advances;
* the wall fluid speed is the outward wall-tangential velocity interpolated
  at the root on the wall edge (the velocity the contact-line friction acts
  on), and the geometric speed is the central difference of the outward
  contact position between neighbouring outputs.

Run metrics are taken over the measurement window t >= window_start_fraction
* end time of tolerances.json.  With several runs, the script also forms the
mesh (smallest step) and time-step (finest mesh) studies and applies the gates
of tolerances.json.

Usage:  verify.py RUN_DIR [RUN_DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
TOLERANCES = HERE / "tolerances.json"
RESULT_PATTERN = re.compile(r"^result_(\d+)\.(vtu|pvtu)$")


def read_state(path: Path) -> dict:
    """Points, owned triangles and fields of one saved state.

    A partitioned (.pvtu) state is merged with numpy: its ghost cells
    (vtkGhostType != 0) are dropped and the vertices that pieces share are
    identified by GlobalNodeID.  VTK's own merge filters are not used.
    """
    dataset = pv.read(path)
    cells = np.asarray(dataset.cells_dict.get(pv.CellType.TRIANGLE), dtype=np.int64)
    points = np.asarray(dataset.points, dtype=float)
    fields = {name: np.asarray(dataset.point_data[name], dtype=float)
              for name in ("phi", "Velocity") if name in dataset.point_data}
    if "vtkGhostType" in dataset.cell_data:
        cells = cells[np.asarray(dataset.cell_data["vtkGhostType"]) == 0]
    if "GlobalNodeID" in dataset.point_data:
        gids = np.asarray(dataset.point_data["GlobalNodeID"], dtype=np.int64).reshape(-1)
        unique, first, inverse = np.unique(gids, return_index=True, return_inverse=True)
        points = points[first]
        fields = {name: values[first] for name, values in fields.items()}
        cells = inverse[cells]
    return {"points": points, "cells": cells, **fields}


def result_files(run_dir: Path) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for path in run_dir.iterdir():
        match = RESULT_PATTERN.match(path.name)
        if match:
            step = int(match.group(1))
            if step in found and found[step].suffix == ".pvtu":
                continue
            found[step] = path
    return sorted(found.items())


def _triangle_area(p: np.ndarray) -> float:
    return 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))


def liquid_area(points: np.ndarray, cells: np.ndarray, phi: np.ndarray) -> float:
    """Exact area of {phi < 0} for P1 phi on triangles."""
    total = 0.0
    for tri in cells:
        values = phi[tri]
        xy = points[tri, :2]
        negative = values < 0.0
        count = int(np.count_nonzero(negative))
        if count == 0:
            continue
        area = _triangle_area(xy)
        if count == 3:
            total += area
            continue
        # One vertex alone on its side: the clipped corner triangle.
        lone = int(np.flatnonzero(negative)[0] if count == 1 else np.flatnonzero(~negative)[0])
        others = [k for k in range(3) if k != lone]
        t1 = values[lone] / (values[lone] - values[others[0]])
        t2 = values[lone] / (values[lone] - values[others[1]])
        corner = area * t1 * t2
        total += corner if count == 1 else area - corner
    return total


def measure_state(state, case: dict) -> dict:
    if isinstance(state, pv.DataSet):
        state = {"points": np.asarray(state.points, dtype=float),
                 "cells": np.asarray(state.cells_dict.get(pv.CellType.TRIANGLE), dtype=np.int64),
                 **{name: np.asarray(state.point_data[name], dtype=float)
                    for name in ("phi", "Velocity") if name in state.point_data}}
    points = state["points"]
    phi = np.asarray(state["phi"], dtype=float).reshape(-1)
    if "Velocity" in state:
        velocity = np.asarray(state["Velocity"], dtype=float).reshape(points.shape[0], -1)
    else:
        velocity = np.zeros((points.shape[0], 3))
    cells = state["cells"]
    if cells is None or cells.size == 0:
        raise ValueError("ren_e_2d states must be triangle meshes")
    h = float(case["h"])
    wall = np.flatnonzero(np.abs(points[:, 1] - float(case["contact_wall_y"])) <= 1.0e-9 * h)
    wall = wall[np.argsort(points[wall, 0], kind="stable")]
    edge_cell: dict[tuple[int, int], int] = {}
    wall_set = set(int(v) for v in wall)
    for index, tri in enumerate(cells):
        on_wall = [int(v) for v in tri if int(v) in wall_set]
        if len(on_wall) == 2:
            edge_cell[tuple(sorted(on_wall))] = index
    roots = []
    for a, b in zip(wall[:-1], wall[1:]):
        pa, pb = phi[a], phi[b]
        if pa == 0.0 and pb == 0.0:
            raise ValueError("zero-valued wall edge")
        if pa * pb < 0.0 or (pa == 0.0) != (pb == 0.0):
            fraction = pa / (pa - pb)
            x = points[a, 0] + fraction * (points[b, 0] - points[a, 0])
            if roots and abs(x - roots[-1]["x"]) <= 1.0e-12 * h:
                continue
            roots.append({"x": float(x), "edge": (int(a), int(b)), "fraction": float(fraction)})
    if len(roots) != 2:
        raise ValueError(f"expected two wall roots, found {len(roots)}")
    gamma = float(case["surface_tension"])
    mobility = float(case["mobility"])
    cos_e = math.cos(math.radians(float(case["equilibrium_angle_degrees"])))
    out = []
    for side, root in zip((-1.0, 1.0), roots):
        key = tuple(sorted(root["edge"]))
        if key not in edge_cell:
            raise ValueError("wall root edge has no triangle")
        tri = cells[edge_cell[key]]
        design = np.column_stack((points[tri, 0], points[tri, 1], np.ones(3)))
        gradient = np.linalg.solve(design, phi[tri])[:2]
        normal = gradient / np.linalg.norm(gradient)
        cos_d = float(np.clip(normal[1], -1.0, 1.0))
        a, b = root["edge"]
        u = velocity[a, :2] + root["fraction"] * (velocity[b, :2] - velocity[a, :2])
        out.append({
            "side": "left" if side < 0 else "right",
            "outward_position": side * root["x"],
            "x": root["x"],
            "dynamic_angle_degrees": math.degrees(math.acos(cos_d)),
            "predicted_speed": gamma * mobility * (cos_e - cos_d),
            "wall_fluid_speed": float(side * u[0]),
        })
    return {"contacts": out, "liquid_area": liquid_area(points, cells, phi)}


def law_enforcement_of(run_dir: Path, case: dict) -> str:
    """Contact_line_law_enforcement of a run: case.json, else the deck itself
    (decks written before the case.json field existed)."""
    if "contact_line_law_enforcement" in case:
        return str(case["contact_line_law_enforcement"])
    deck = run_dir / "solver.xml"
    if deck.is_file():
        match = re.search(r"<Contact_line_law_enforcement>\s*(\w+)\s*</", deck.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return "Weak"


def measure_run(run_dir: Path, tolerances: dict) -> dict:
    case = json.loads((run_dir / "case.json").read_text(encoding="utf-8"))
    dt = float(case["dt"])
    states = [(0, read_state(run_dir / "mesh" / "mesh-complete.mesh.vtu"))]
    for step, path in result_files(run_dir):
        states.append((step, read_state(path)))
    records = []
    for step, dataset in states:
        record = measure_state(dataset, case)
        record["step"] = step
        record["time"] = step * dt
        records.append(record)
    area0 = records[0]["liquid_area"]
    for k, record in enumerate(records):
        record["area_drift"] = abs(record["liquid_area"] - area0) / area0
        for side in range(2):
            lo, hi = max(k - 1, 0), min(k + 1, len(records) - 1)
            if hi == lo:
                speed = float("nan")
            else:
                speed = ((records[hi]["contacts"][side]["outward_position"]
                          - records[lo]["contacts"][side]["outward_position"])
                         / (records[hi]["time"] - records[lo]["time"]))
            record["contacts"][side]["geometric_speed"] = speed
    end_time = records[-1]["time"]
    window_start = float(tolerances["window_start_fraction"]) * float(case["steps_protocol"]) * dt
    fluid_errors, geometric_errors, signs = [], [], []
    fluid_signed, geometric_signed = [], []
    for record in records[1:]:
        if record["time"] < window_start - 1.0e-12:
            continue
        for contact in record["contacts"]:
            predicted = contact["predicted_speed"]
            if abs(predicted) <= 1.0e-14:
                continue
            fluid_signed.append((contact["wall_fluid_speed"] - predicted) / abs(predicted))
            geometric_signed.append((contact["geometric_speed"] - predicted) / abs(predicted))
            fluid_errors.append(abs(fluid_signed[-1]))
            geometric_errors.append(abs(geometric_signed[-1]))
            signs.append(math.copysign(1.0, contact["wall_fluid_speed"]) == math.copysign(1.0, predicted)
                         and math.copysign(1.0, contact["geometric_speed"]) == math.copysign(1.0, predicted))

    def rms(values: list[float]) -> float:
        return float(math.sqrt(np.mean(np.square(values)))) if values else float("nan")

    final = records[-1]
    return {
        "run": str(run_dir),
        "case": case["case"],
        "level": int(case["level_R_over_h"]),
        "dt_divisor": int(case["dt_divisor"]),
        "dt": dt,
        "slip_length_over_h": float(case["slip_length_over_h"]),
        "end_time": end_time,
        "complete": (not case.get("truncated")) and records[-1]["step"] == int(case["steps"]),
        "outputs": len(records) - 1,
        "window_start": window_start,
        "window_samples": len(fluid_errors),
        "rms_fluid_speed_relative_error": rms(fluid_errors),
        "max_fluid_speed_relative_error": max(fluid_errors) if fluid_errors else float("nan"),
        "rms_geometric_speed_relative_error": rms(geometric_errors),
        "mean_signed_fluid_speed_relative_error": (float(np.mean(fluid_signed))
                                                   if fluid_signed else float("nan")),
        "mean_signed_geometric_speed_relative_error": (float(np.mean(geometric_signed))
                                                       if geometric_signed else float("nan")),
        "law_enforcement": law_enforcement_of(run_dir, case),
        "max_geometric_speed_relative_error": max(geometric_errors) if geometric_errors else float("nan"),
        "signs_agree": bool(signs) and all(signs),
        "max_area_drift": max(record["area_drift"] for record in records),
        "final_mean_outward_displacement": float(np.mean([
            final["contacts"][s]["outward_position"] - records[0]["contacts"][s]["outward_position"]
            for s in range(2)])),
        "final_mean_dynamic_angle_degrees": float(np.mean([c["dynamic_angle_degrees"] for c in final["contacts"]])),
        "initial_mean_dynamic_angle_degrees": float(np.mean([c["dynamic_angle_degrees"] for c in records[0]["contacts"]])),
        "history": [{"time": r["time"],
                     "left": r["contacts"][0], "right": r["contacts"][1],
                     "liquid_area": r["liquid_area"]} for r in records],
    }


def gate(results: list[dict], tolerances: dict, law_criteria: str = "strong") -> dict:
    """Mesh and time-step studies and the gates.

    law_criteria "strong" (decision D39, default): the geometric RMS error and
    the mean signed geometric error at every level of the dt0/4 mesh study;
    "weak": the superseded 2026-10-07 criteria (finest-level accuracy of the
    fluid and geometric errors and their strict decrease with h).
    """
    if law_criteria not in ("strong", "weak"):
        raise ValueError("law_criteria must be 'strong' or 'weak'")
    weak = tolerances.get("superseded_weak_law_criteria", tolerances)
    report: dict = {"gates": {}, "studies": {}}
    by_key = {(r["case"], r["level"], r["dt_divisor"]): r for r in results}
    levels = sorted({r["level"] for r in results})
    divisors = sorted({r["dt_divisor"] for r in results})
    finest, smallest = (levels[-1], divisors[-1]) if levels and divisors else (None, None)
    gates = report["gates"]
    gates["all_runs_complete"] = all(r["complete"] for r in results)
    gates["signs_agree"] = all(r["signs_agree"] for r in results)
    gates["area_drift"] = all(r["max_area_drift"] <= tolerances["max_area_drift"] for r in results)
    if law_criteria == "strong":
        # The D39 criteria assume the exactly imposed law.
        gates["runs_use_strong_law"] = all(r.get("law_enforcement", "Strong") == "Strong"
                                           for r in results)
    for case in sorted({r["case"] for r in results}):
        study = {"mesh": [], "time": []}
        for level in levels:
            r = by_key.get((case, level, smallest))
            if r:
                study["mesh"].append({"level": level, "rms_fluid": r["rms_fluid_speed_relative_error"],
                                      "rms_geometric": r["rms_geometric_speed_relative_error"],
                                      "mean_signed_fluid": r.get("mean_signed_fluid_speed_relative_error"),
                                      "mean_signed_geometric": r.get("mean_signed_geometric_speed_relative_error"),
                                      "displacement": r["final_mean_outward_displacement"]})
        for divisor in divisors:
            r = by_key.get((case, finest, divisor))
            if r:
                study["time"].append({"dt_divisor": divisor, "rms_fluid": r["rms_fluid_speed_relative_error"],
                                      "rms_geometric": r["rms_geometric_speed_relative_error"],
                                      "displacement": r["final_mean_outward_displacement"]})
        report["studies"][case] = study
        mesh = study["mesh"]
        # Displacement self-convergence over the mesh study (reported, D39).
        disp = [m["displacement"] for m in mesh]
        changes = [abs(disp[k + 1] - disp[k]) for k in range(len(disp) - 1)]
        study["displacement_self_convergence"] = {
            "levels": [m["level"] for m in mesh], "displacement": disp, "changes": changes,
            "ratio": (changes[0] / changes[1]) if len(changes) == 2 and changes[1] > 0.0 else None,
            "observed_order": (math.log2(changes[0] / changes[1])
                               if len(changes) == 2 and changes[0] > 0.0 and changes[1] > 0.0
                               else None)}
        if law_criteria == "strong":
            if not mesh:
                gates[f"{case}_geometric_rms_every_level"] = None
                continue
            gates[f"{case}_geometric_rms_every_level"] = len(mesh) == len(levels) and all(
                m["rms_geometric"] <= tolerances["geometric_rms_relative_error"] for m in mesh)
            gates[f"{case}_geometric_mean_signed_every_level"] = len(mesh) == len(levels) and all(
                abs(m["mean_signed_geometric"]) <= tolerances["geometric_mean_signed_relative_error"]
                for m in mesh)
        else:
            finest_run = by_key.get((case, finest, smallest))
            if finest_run is None:
                gates[f"{case}_finest_accuracy"] = None
                continue
            gates[f"{case}_finest_accuracy"] = (
                finest_run["rms_fluid_speed_relative_error"] <= weak["finest_rms_relative_error"]
                and finest_run["rms_geometric_speed_relative_error"] <= weak["finest_rms_relative_error"])
            if len(mesh) == len(levels) and len(mesh) >= 2:
                gates[f"{case}_mesh_convergence"] = all(
                    mesh[k + 1][name] < mesh[k][name] for k in range(len(mesh) - 1)
                    for name in ("rms_fluid", "rms_geometric"))
        times = study["time"]
        if len(times) >= 2:
            a, b = times[-2]["displacement"], times[-1]["displacement"]
            change = abs(a - b) / max(abs(b), 1.0e-300)
            study["time_step_displacement_change"] = change
            gates[f"{case}_time_step"] = change <= tolerances["max_displacement_change_halving_dt"]
    decided = [value for value in gates.values() if value is not None]
    report["outcome"] = "PASS" if decided and all(decided) else "FAIL"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--tolerances", type=Path, default=TOLERANCES)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--law-criteria", choices=("strong", "weak"), default="strong",
                        help="strong (decision D39, default) or the superseded 2026-10-07 criteria "
                             "for the weakly imposed law")
    args = parser.parse_args(argv)
    tolerances = json.loads(args.tolerances.read_text(encoding="utf-8"))
    results = [measure_run(run, tolerances) for run in args.runs]
    report = gate(results, tolerances, args.law_criteria)
    for r in results:
        print(f"{r['case']:9s} R/h={r['level']:2d} dt0/{r['dt_divisor']} l_s/h={r['slip_length_over_h']:g} "
              f"law={r['law_enforcement']} "
              f"rms_fluid={r['rms_fluid_speed_relative_error']:.4f} rms_geom={r['rms_geometric_speed_relative_error']:.4f} "
              f"mean_geom={r['mean_signed_geometric_speed_relative_error']:+.4f} "
              f"signs={r['signs_agree']} area_drift={r['max_area_drift']:.2e} "
              f"disp={r['final_mean_outward_displacement']:.5f} complete={r['complete']}")
    for case, study in report["studies"].items():
        sc = study.get("displacement_self_convergence")
        if sc:
            print(f"{case}: displacement at t_end over R/h={sc['levels']}: "
                  + ", ".join(f"{v:.5f}" for v in sc["displacement"])
                  + "; changes " + ", ".join(f"{v:.2e}" for v in sc["changes"])
                  + (f"; observed order {sc['observed_order']:.2f}" if sc["observed_order"] is not None
                     else "") + " (reported, not gated)")
    print(json.dumps({"law_criteria": args.law_criteria, "gates": report["gates"],
                      "outcome": report["outcome"]}, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"runs": results, "report": report}, indent=2) + "\n", encoding="utf-8")
    return 0 if report["outcome"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
