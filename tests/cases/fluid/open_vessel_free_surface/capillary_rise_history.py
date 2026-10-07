#!/usr/bin/env python3
"""Harvest capillary-rise histories and build the comparison candidate.

``history``   reads one run directory written by capillary_rise_2d.py (d4
              profile): the initial mesh fields and every saved result give
              time, apex height, wall-contact height, contact angle, wall
              contact speed and contact motion per step.
``candidate`` combines the histories of the comparison levels into the
              three-column candidate of compare_capillary_rise_candidate.py:
              the finest level with the numerical uncertainty rule frozen in
              free_surface_wp5_capillary_rise_candidate_protocol_v1.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

SCRIPT_PATH = Path(__file__).resolve()
CANDIDATE_PROTOCOL = (
    SCRIPT_PATH.parents[1] / "free_surface_wp5_capillary_rise_candidate_protocol_v1.json"
)
HISTORY_COLUMNS = (
    "time_s",
    "step",
    "apex_height_mm",
    "wall_contact_height_mm",
    "contact_angle_degrees",
    "wall_contact_speed_m_per_s",
    "contact_motion_cells_per_step",
)
RESULT_PATTERN = re.compile(r"^result_(\d+)\.(vtu|pvtu)$")


def _load_case_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_capillary_rise_2d_for_history", SCRIPT_PATH.with_name("capillary_rise_2d.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


case_module = _load_case_module()


def result_files(case_dir: Path) -> list[tuple[int, Path]]:
    """Saved results by step; partitioned runs are read through their .pvtu."""
    found: dict[int, Path] = {}
    for path in case_dir.iterdir():
        match = RESULT_PATTERN.match(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step in found and found[step].suffix == ".pvtu":
            continue
        found[step] = path
    return sorted(found.items())


def merge_partitioned_state(dataset: pv.DataSet) -> pv.UnstructuredGrid:
    """One triangle grid from a partitioned (.pvtu) state, merged with numpy.

    Partition pieces repeat their ghost cells (vtkGhostType != 0) and the
    vertices they share; the ghosts are dropped and the vertices identified
    by GlobalNodeID so that the boundary roots and contact triangles are
    unique.  VTK's merge filters are not used (they crash on these files).
    """
    cells = np.asarray(dataset.cells_dict.get(pv.CellType.TRIANGLE), dtype=np.int64)
    if "vtkGhostType" in dataset.cell_data:
        cells = cells[np.asarray(dataset.cell_data["vtkGhostType"]) == 0]
    points = np.asarray(dataset.points, dtype=float)
    fields = {name: np.asarray(dataset.point_data[name])
              for name in dataset.point_data.keys() if name != "vtkGhostType"}
    if "GlobalNodeID" in fields:
        gids = np.asarray(fields["GlobalNodeID"], dtype=np.int64).reshape(-1)
        _, first, inverse = np.unique(gids, return_index=True, return_inverse=True)
        points = points[first]
        fields = {name: values[first] for name, values in fields.items()}
        cells = inverse.reshape(-1)[cells]
    connectivity = np.hstack([np.full((cells.shape[0], 1), 3, dtype=np.int64), cells]).ravel()
    merged = pv.UnstructuredGrid(
        connectivity, np.full(cells.shape[0], pv.CellType.TRIANGLE, dtype=np.uint8), points)
    for name, values in fields.items():
        merged.point_data[name] = values
    return merged


def collect_history(case_dir: Path) -> list[dict[str, float]]:
    benchmark = json.loads((case_dir / "benchmark.json").read_text(encoding="utf-8"))
    dt = float(benchmark["time_step_size_s"])
    dx = float(benchmark["mesh_resolution"]["dx_m"])
    states: list[tuple[int, pv.DataSet]] = [
        (0, pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu"))]
    for step, path in result_files(case_dir):
        dataset = pv.read(path)
        if path.suffix == ".pvtu":
            dataset = merge_partitioned_state(dataset)
        states.append((step, dataset))
    rows: list[dict[str, float]] = []
    for step, dataset in states:
        if step == 0 and "Velocity" not in dataset.point_data:
            dataset.point_data["Velocity"] = np.zeros((dataset.n_points, 3))
        state = case_module.state_metrics(dataset, benchmark)
        if not state.get("available"):
            raise ValueError(f"{case_dir}: step {step}: {state.get('error')}")
        row = {
            "time_s": step * dt,
            "step": step,
            "apex_height_mm": state["apex_height_mm"],
            "wall_contact_height_mm": state["wall_contact_height_mm"],
            "contact_angle_degrees": state["contact_angle_degrees"],
            "wall_contact_speed_m_per_s": state["wall_contact_fluid_speed_m_per_s"],
            "contact_motion_cells_per_step": float("nan"),
        }
        if rows:
            previous = rows[-1]
            steps_between = step - int(previous["step"])
            if steps_between > 0:
                row["contact_motion_cells_per_step"] = abs(
                    row["wall_contact_height_mm"] - previous["wall_contact_height_mm"]
                ) * 1.0e-3 / dx / steps_between
        rows.append(row)
    return rows


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (int(row[key]) if key == "step" else repr(float(row[key])))
                             for key in HISTORY_COLUMNS})


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source)]


def _on_grid(rows: list[dict[str, float]], grid: np.ndarray) -> np.ndarray:
    times = np.asarray([row["time_s"] for row in rows])
    heights = np.asarray([row["apex_height_mm"] for row in rows])
    if times[0] > grid[0] + 1.0e-12 or times[-1] < grid[-1] - 1.0e-12:
        raise ValueError("history does not cover the comparison grid")
    return np.interp(grid, times, heights)


def numerical_uncertainty(levels: list[list[dict[str, float]]],
                          protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen candidate-uncertainty rule to coarse-to-fine levels."""
    grid_spec = protocol["comparison_grid"]
    grid = np.round(np.arange(int(grid_spec["point_count"])) * float(grid_spec["step_s"])
                    + float(grid_spec["start_s"]), 12)
    heights = [_on_grid(rows, grid) for rows in levels]
    rule = protocol["uncertainty_rule"]
    ratio = float(rule["refinement_ratio"])
    if len(heights) >= 3:
        fine, medium, coarse = heights[-1], heights[-2], heights[-3]
        delta_fine = float(np.sqrt(np.mean((fine - medium) ** 2)))
        delta_coarse = float(np.sqrt(np.mean((medium - coarse) ** 2)))
        order = None
        if delta_fine > 0.0 and delta_coarse > delta_fine:
            order = math.log(delta_coarse / delta_fine) / math.log(ratio)
        if order is None:
            safety = float(rule["fallback_safety_factor"])
            used_order = float(rule["fallback_order"])
        else:
            safety = float(rule["three_level_safety_factor"])
            used_order = min(max(order, float(rule["order_clip"][0])),
                             float(rule["order_clip"][1]))
        uncertainty = safety * np.abs(fine - medium) / (ratio ** used_order - 1.0)
        method = "three_level"
    elif len(heights) == 2:
        fine, medium = heights[-1], heights[-2]
        order = None
        safety = float(rule["two_level_safety_factor"])
        used_order = float(rule["fallback_order"])
        uncertainty = safety * np.abs(fine - medium) / (ratio ** used_order - 1.0)
        method = "two_level"
    else:
        raise ValueError("at least two levels are needed for a numerical uncertainty")
    return {
        "grid": grid,
        "candidate_height_mm": heights[-1],
        "numerical_uncertainty_mm": uncertainty,
        "method": method,
        "observed_order": order,
        "used_order": used_order,
        "safety_factor": safety,
    }


def write_candidate(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(("time_s", "apex_height_mm", "numerical_uncertainty_mm"))
        for time_s, height, uncertainty in zip(result["grid"], result["candidate_height_mm"],
                                               result["numerical_uncertainty_mm"]):
            writer.writerow((repr(float(time_s)), repr(float(height)), repr(float(uncertainty))))


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    history = sub.add_parser("history", help="harvest one run directory")
    history.add_argument("case_dir", type=Path)
    history.add_argument("--output", type=Path, required=True)
    candidate = sub.add_parser("candidate", help="combine level histories (coarse to fine)")
    candidate.add_argument("histories", type=Path, nargs="+")
    candidate.add_argument("--protocol", type=Path, default=CANDIDATE_PROTOCOL)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--summary-json", type=Path)
    args = parser.parse_args(arguments)
    if args.command == "history":
        rows = collect_history(args.case_dir)
        write_history(args.output, rows)
        motions = [row["contact_motion_cells_per_step"] for row in rows[1:]
                   if math.isfinite(row["contact_motion_cells_per_step"])]
        print(json.dumps({"case": str(args.case_dir), "records": len(rows),
                          "end_time_s": rows[-1]["time_s"],
                          "maximum_contact_motion_cells_per_step": max(motions) if motions else None}))
        return 0
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    levels = [read_history(path) for path in args.histories]
    result = numerical_uncertainty(levels, protocol)
    write_candidate(args.output, result)
    summary = {key: result[key] for key in ("method", "observed_order", "used_order", "safety_factor")}
    summary["levels"] = [str(path) for path in args.histories]
    summary["numerical_uncertainty_rms_mm"] = float(np.sqrt(np.mean(result["numerical_uncertainty_mm"] ** 2)))
    summary["numerical_uncertainty_maximum_mm"] = float(np.max(result["numerical_uncertainty_mm"]))
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
