#!/usr/bin/env python3
"""Audit open-vessel pressure gauges against the mesh and hydrostatic setup."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pyvista as pv


ROOT = Path(__file__).resolve().parents[4]
CASE_ROOT = ROOT / "tests/cases/fluid/open_vessel_free_surface"
DEFAULT_CASES = (
    "unfitted_level_set/spheric_test02_dambreak_obstacle",
    "unfitted_level_set/spheric_test10_lateral_water_1x",
    "unfitted_level_set/spheric_test05_wet_bed_d18",
    "unfitted_level_set/spheric_test05_wet_bed_d38",
)


def child_text(parent: ET.Element, tag: str) -> str | None:
    child = parent.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def fluid_equation(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type", "").lower() == "fluid":
            return equation
    raise ValueError("missing fluid equation")


def parse_floats(value: str | None, expected_count: int) -> list[float] | None:
    if value is None:
        return None
    parts = [float(part) for part in value.split()]
    if len(parts) != expected_count:
        raise ValueError(f"expected {expected_count} values, got {value!r}")
    return parts


def load_gauge_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {"node_id": int(row["node_id"]), "pressure": float(row["pressure"])}
        for row in rows
    ]


def find_point_index(mesh: pv.DataSet, node_id: int) -> tuple[int | None, str | None]:
    for name in ("GlobalNodeID", "GlobalVertexID", "Global_vertex_gid", "GlobalPointID"):
        if name not in mesh.point_data:
            continue
        matches = [index for index, value in enumerate(mesh.point_data[name]) if int(value) == node_id]
        if matches:
            return int(matches[0]), name
    if 0 <= node_id < mesh.n_points:
        return node_id, "point_index"
    return None, None


def incident_cell_support(mesh: pv.UnstructuredGrid, point_index: int) -> dict[str, int] | None:
    if "phi" not in mesh.point_data:
        return None
    phi = mesh.point_data["phi"]
    cells = mesh.cells
    cursor = 0
    incident = 0
    active = 0
    full_wet = 0
    while cursor < len(cells):
        count = int(cells[cursor])
        ids = [int(value) for value in cells[cursor + 1 : cursor + 1 + count]]
        if point_index in ids:
            incident += 1
            values = [float(phi[index]) for index in ids]
            if min(values) <= 0.0:
                active += 1
            if max(values) <= 0.0:
                full_wet += 1
        cursor += count + 1
    return {
        "incident_cell_count": incident,
        "active_support_cell_count": active,
        "full_wet_support_cell_count": full_wet,
    }


def collect_case(case_dir: Path, tolerance: float) -> dict[str, Any]:
    solver_path = case_dir / "solver.xml"
    gauge_path = case_dir / "pressure_gauge.csv"
    tree = ET.parse(solver_path)
    root = tree.getroot()
    fluid = fluid_equation(root)
    mesh_path_text = child_text(root.find("Add_mesh"), "Mesh_file_path") if root.find("Add_mesh") is not None else None
    if mesh_path_text is None:
        raise ValueError(f"{solver_path} is missing Add_mesh/Mesh_file_path")
    mesh_path = case_dir / mesh_path_text
    mesh = pv.read(mesh_path)

    rho = float(child_text(fluid, "Density") or "0.0")
    force = parse_floats(
        " ".join(
            child_text(fluid, tag) or "0.0"
            for tag in ("Force_x", "Force_y", "Force_z")
        ),
        3,
    )
    reference_point = parse_floats(child_text(fluid, "Hydrostatic_pressure_reference_point"), 3)
    reference_pressure = float(child_text(fluid, "Hydrostatic_pressure_reference") or "0.0")

    constraint = fluid.find("Node_pressure_constraints")
    configured_path = child_text(constraint, "Values_file_path") if constraint is not None else None
    constraint_configured = configured_path == gauge_path.name
    metadata_path = next(
        (
            candidate
            for candidate in (
                case_dir / "benchmark.json",
                case_dir / "expected_results.json",
            )
            if candidate.exists()
        ),
        None,
    )
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path is not None
        else {}
    )
    verification = metadata.get("pressure_gauge_verification", {})
    constraint_expected = bool(
        verification.get("pressure_constraint_enabled", True)
        if isinstance(verification, dict)
        else True
    )

    rows = []
    for row in load_gauge_rows(gauge_path):
        node_id = int(row["node_id"])
        prescribed = float(row["pressure"])
        point_index, id_source = find_point_index(mesh, node_id)
        if point_index is None:
            rows.append(
                {
                    "node_id": node_id,
                    "found": False,
                    "passed": False,
                    "errors": ["gauge node was not found in the mesh"],
                }
            )
            continue
        point = [float(value) for value in mesh.points[point_index]]
        expected = None
        pressure_error = None
        if reference_point is not None and force is not None:
            expected = reference_pressure - rho * sum(
                force[i] * (reference_point[i] - point[i]) for i in range(3)
            )
            pressure_error = prescribed - expected
        signed_phi = float(mesh.point_data["phi"][point_index]) if "phi" in mesh.point_data else None
        support = incident_cell_support(mesh, point_index)
        errors = []
        if constraint_expected and not constraint_configured:
            errors.append("pressure_gauge.csv is not configured as Node_pressure_constraints")
        if not constraint_expected and constraint is not None:
            errors.append("Node_pressure_constraints is configured although metadata disables it")
        if pressure_error is not None and abs(pressure_error) > tolerance:
            errors.append(
                f"prescribed pressure differs from hydrostatic value by {pressure_error:.6g}"
            )
        if support is not None and support["active_support_cell_count"] <= 0:
            errors.append("gauge node has no active pressure support cells")
        if signed_phi is not None and signed_phi > tolerance:
            errors.append(f"gauge node is dry: phi={signed_phi:.6g}")
        rows.append(
            {
                "node_id": node_id,
                "found": True,
                "point_index": point_index,
                "id_source": id_source,
                "coordinates": point,
                "signed_level_set": signed_phi,
                "prescribed_pressure": prescribed,
                "expected_hydrostatic_pressure": expected,
                "pressure_error": pressure_error,
                "constraint_configured": constraint_configured,
                "constraint_expected": constraint_expected,
                "support": support,
                "passed": not errors,
                "errors": errors,
            }
        )

    return {
        "case": str(case_dir.relative_to(CASE_ROOT)),
        "mesh": str(mesh_path.relative_to(case_dir)),
        "mesh_points": mesh.n_points,
        "mesh_cells": mesh.n_cells,
        "pressure_constraint_file": configured_path,
        "pressure_constraint_expected": constraint_expected,
        "gauge_checks": rows,
        "passed": all(row["passed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Case paths relative to tests/cases/fluid/open_vessel_free_surface.",
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()

    reports = [collect_case(CASE_ROOT / case, args.tolerance) for case in args.cases]
    output = {"passed": all(report["passed"] for report in reports), "cases": reports}
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
