#!/usr/bin/env python3
"""Audit ALE VTU mesh quality for open-vessel validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


def deformed_points(mesh: pv.DataSet) -> np.ndarray:
    if "CurrentCoordinates" in mesh.point_data:
        return np.asarray(mesh.point_data["CurrentCoordinates"])
    if "mesh_displacement" in mesh.point_data:
        return np.asarray(mesh.points) + np.asarray(mesh.point_data["mesh_displacement"])
    return np.asarray(mesh.points)


def displacement_max(mesh: pv.DataSet) -> float | None:
    if "mesh_displacement" not in mesh.point_data:
        return None
    disp = np.asarray(mesh.point_data["mesh_displacement"])
    if disp.size == 0:
        return 0.0
    return float(np.linalg.norm(disp, axis=1).max())


def quality_record(path: Path) -> dict[str, Any]:
    mesh = pv.read(path)
    deformed = mesh.copy(deep=True)
    deformed.points = deformed_points(mesh)
    if hasattr(deformed, "cell_quality"):
        quality = deformed.cell_quality(quality_measure="scaled_jacobian")
        values = np.asarray(quality.cell_data["scaled_jacobian"])
    else:
        quality = deformed.compute_cell_quality(quality_measure="scaled_jacobian")
        values = np.asarray(quality.cell_data["CellQuality"])
    return {
        "path": str(path),
        "points": int(mesh.n_points),
        "cells": int(mesh.n_cells),
        "scaled_jacobian_min": float(values.min()) if values.size else None,
        "scaled_jacobian_max": float(values.max()) if values.size else None,
        "scaled_jacobian_nonpositive_count": int(np.count_nonzero(values <= 0.0)),
        "mesh_displacement_max": displacement_max(mesh),
    }


def audit_directory(directory: Path, pattern: str, baseline_mesh: Path | None) -> dict[str, Any]:
    files = sorted(directory.glob(pattern))
    records = []
    if baseline_mesh is not None:
        records.append({"kind": "baseline", **quality_record(baseline_mesh)})
    records.extend({"kind": "result", **quality_record(path)} for path in files)

    result_records = [record for record in records if record["kind"] == "result"]
    mins = [
        record["scaled_jacobian_min"]
        for record in result_records
        if record["scaled_jacobian_min"] is not None
    ]
    nonpositive = [
        record["scaled_jacobian_nonpositive_count"] for record in result_records
    ]
    return {
        "directory": str(directory),
        "pattern": pattern,
        "result_count": len(result_records),
        "baseline_mesh": str(baseline_mesh) if baseline_mesh is not None else None,
        "min_scaled_jacobian_over_results": min(mins) if mins else None,
        "max_nonpositive_scaled_jacobian_count": max(nonpositive) if nonpositive else 0,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pattern", default="result_*.vtu")
    parser.add_argument("--baseline-mesh", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_directory(args.directory, args.pattern, args.baseline_mesh)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
