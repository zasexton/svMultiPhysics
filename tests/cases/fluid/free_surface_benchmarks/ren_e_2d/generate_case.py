#!/usr/bin/env python3
"""Write one run of the 2D Ren-E moving-contact-line benchmark (tracker M4).

A 2D liquid cap of radius R sits on the flat bottom wall y = 0 with zero
gravity.  It starts as a circular cap at an angle theta_0 away from the Young
angle theta_e and moves under the Ren-E contact-line law

    V_CL = gamma * M * (cos(theta_e) - cos(theta_d)),

positive for an advancing (wetting) contact line, with the line mobility M.
The law is imposed as line friction xi = 1/M in the momentum equation
together with the variational Young term (Contact_line_model DynamicRenE),
Navier slip on the wetted wall and a strong normal-only wall velocity
(decision D4).  The rest of the configuration is the sessile-drop benchmark
deck (sessile_drop_2d/generate_case.py): SurfaceStress capillarity with
small-cut aggregation, PDE velocity extension (D15), kinematic
reconciliation (D14), sign-definite patch bounds (D21), generalized-alpha and
FSILS GMRES.

The benchmark refines the 2026-08-30 pilot (tracker audit L1819, job
41286834): three meshes R/h = 8, 16, 32 with a fixed physical slip length
R/4 (l_s/h = 2, 4, 8) and three time steps dt0, dt0/2, dt0/4 for an advancing
case (theta_0 = theta_e + 15 deg) and a receding case (theta_0 = theta_e - 15
deg).  The reference is the continuum law itself, evaluated with the measured
dynamic angle; verify.py compares it with the wall fluid speed and the
geometric contact-line speed and applies tolerances.json.  See README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SESSILE_PATH = HERE.parent / "sessile_drop_2d" / "generate_case.py"
_spec = importlib.util.spec_from_file_location("_sessile_drop_2d_generator", _SESSILE_PATH)
sessile = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sessile
_spec.loader.exec_module(sessile)

# ---------------------------------------------------------------------------
# Protocol constants (README.md gives the source or reason of each one).
# ---------------------------------------------------------------------------
LEVELS = (8, 16, 32)                         # R/h
RADIUS = 1.0                                 # equilibrium cap radius (length unit)
DENSITY = 1.0
SURFACE_TENSION = 1.0
LAPLACE_NUMBER = 12.0                        # as static_drop_2d and sessile_drop_2d
MOBILITY = 1.0                               # M in V = gamma M (cos theta_e - cos theta_d)
SLIP_LENGTH = RADIUS / 4.0                   # l_s/h = 2, 4, 8 at R/h = 8, 16, 32
EQUILIBRIUM_ANGLE_DEG = 90.0                 # theta_e of the pilot
ANGLE_PERTURBATION_DEG = 15.0
CASES = {"advancing": EQUILIBRIUM_ANGLE_DEG + ANGLE_PERTURBATION_DEG,
         "receding": EQUILIBRIUM_ANGLE_DEG - ANGLE_PERTURBATION_DEG}
CENTRE_OFFSET_X = math.pi / 100.0 * RADIUS   # no contact point on a vertex of the dyadic grids
BOX_MARGIN = 0.25 * RADIUS
GRID_UNIT = RADIUS / min(LEVELS)
# Time steps: dt0 = 1/800 is below the capillary limit
# 0.707 * sqrt(rho h^3 / (2 pi gamma)) = 1.56e-3 of the finest mesh R/h = 32, so
# every (mesh, step) pair is stable; dt0/2 and dt0/4 are the refinements.
DT0 = 1.0 / 800.0
DT_DIVISORS = (1, 2, 4)
END_TIME = 0.25                              # 0.25 R/(M gamma): the line moves under a sizable law speed
OUTPUT_INTERVAL = 0.0125                     # 20 outputs; 10 / 20 / 40 steps at dt0 / dt0/2 / dt0/4
CONTACT_WALL_NORMAL = (0.0, -1.0, 0.0)


def viscosity() -> float:
    return math.sqrt(DENSITY * SURFACE_TENSION * 2.0 * RADIUS / LAPLACE_NUMBER)


def predicted_speed(dynamic_angle_deg: float) -> float:
    """Ren-E contact-line speed, positive when the line advances."""
    return SURFACE_TENSION * MOBILITY * (
        math.cos(math.radians(EQUILIBRIUM_ANGLE_DEG)) - math.cos(math.radians(dynamic_angle_deg)))


def schedule(dt_divisor: int) -> dict:
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    dt = DT0 / dt_divisor
    steps = int(round(END_TIME / dt))
    cadence = int(round(OUTPUT_INTERVAL / dt))
    if not (math.isclose(steps * dt, END_TIME) and math.isclose(cadence * dt, OUTPUT_INTERVAL)):
        raise ValueError("end time and output interval must be multiples of the step")
    return {"dt": dt, "steps": steps, "output_cadence": cadence, "viscosity": viscosity(),
            "end_time": END_TIME}


def case_geometry(initial_deg: float) -> dict:
    theta_e = math.radians(EQUILIBRIUM_ANGLE_DEG)
    theta_0 = math.radians(initial_deg)
    equilibrium = sessile.cap_geometry(RADIUS, theta_e)
    initial = sessile.cap_geometry(sessile.cap_radius_for_area(equilibrium["area"], theta_0), theta_0)

    def grid_ceil(value: float) -> float:
        return math.ceil(value / GRID_UNIT - 1.0e-9) * GRID_UNIT

    half_width = grid_ceil(max(initial["half_extent"], equilibrium["half_extent"])
                           + abs(CENTRE_OFFSET_X) + BOX_MARGIN)
    height = grid_ceil(max(initial["apex_height"], equilibrium["apex_height"]) + BOX_MARGIN)
    return {"equilibrium": equilibrium, "initial": initial,
            "box": [-half_width, half_width, 0.0, height]}


def solver_xml(level: int, sched: dict, steps: int, cadence: int) -> str:
    """The sessile-drop deck with the DynamicRenE contact line and the Ren-E slip length."""
    base = sessile.solver_xml("surface_stress", EQUILIBRIUM_ANGLE_DEG, sched, steps, cadence)
    header, _, body = base.partition("<svMultiPhysicsFile")
    root = ET.fromstring("<svMultiPhysicsFile" + body)
    free_surface = None
    for bc in root.iter("Add_BC"):
        if bc.attrib.get("name") == "free_surface":
            free_surface = bc
    if free_surface is None:
        raise RuntimeError("sessile deck has no free-surface condition")

    def set_text(name: str, value: str) -> None:
        element = free_surface.find(name)
        if element is None:
            raise RuntimeError(f"sessile free-surface condition lacks {name}")
        element.text = value

    set_text("Contact_line_model", "DynamicRenE")
    set_text("Wall_slip_length", f"{SLIP_LENGTH:.17g}")
    children = list(free_surface)
    index = children.index(free_surface.find("Contact_angle_degrees"))
    mobility = ET.Element("Contact_line_mobility")
    mobility.text = f"{MOBILITY:.17g}"
    mobility.tail = children[index].tail
    free_surface.insert(index + 1, mobility)
    text = ET.tostring(root, encoding="unicode")
    comment = (f"<!-- ren_e_2d benchmark, R/h = {level}, theta_e = {EQUILIBRIUM_ANGLE_DEG:g} deg, "
               f"line mobility {MOBILITY:g}; generated by generate_case.py -->\n")
    return '<?xml version="1.0" encoding="UTF-8" ?>\n' + comment + text + "\n"


def generate(level: int, case: str, dt_divisor: int, output_dir: Path, *,
             max_steps: int | None = None, force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if case not in CASES:
        raise ValueError(f"--case must be one of {tuple(CASES)}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")
    sched = schedule(dt_divisor)
    steps, cadence, truncated = sched["steps"], sched["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, truncated = max_steps, True
            cadence = min(cadence, steps)
    initial_deg = CASES[case]
    geometry = case_geometry(initial_deg)
    initial, equilibrium = geometry["initial"], geometry["equilibrium"]
    points, cells, faces, (nx, ny) = sessile.structured_triangle_mesh(geometry["box"], level)
    h = RADIUS / level
    centre = np.array([CENTRE_OFFSET_X, initial["centre_y"]])
    phi = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) - initial["radius"]
    initial_contacts = CENTRE_OFFSET_X + np.array([-1.0, 1.0]) * initial["base_half_width"]
    wall_x = points[faces["wall_bottom"][0], 0]
    contact_vertex_gap_over_h = float(
        np.min(np.abs(initial_contacts[:, None] - wall_x[None, :])) / h)
    laplace_pressure = SURFACE_TENSION / initial["radius"]
    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    sessile.write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
                      {"GlobalNodeID": ("Int64", np.arange(n_points)),
                       "phi": ("Float64", phi),
                       "Velocity": ("Float64", np.zeros((n_points, 3))),
                       "Pressure": ("Float64", np.full(n_points, laplace_pressure))},
                      {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in sessile.WALLS:
        node_ids, parents = faces[wall]
        sessile.write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(solver_xml(level, sched, steps, cadence), encoding="utf-8")
    record = {
        "benchmark": "ren_e_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/ren_e_2d/generate_case.py",
        "level_R_over_h": level,
        "case": case,
        "equilibrium_angle_degrees": EQUILIBRIUM_ANGLE_DEG,
        "initial_angle_degrees": initial_deg,
        "contact_line_model": "DynamicRenE",
        "capillary_form": "surface_stress",
        "mobility": MOBILITY,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": sched["viscosity"],
        "laplace_number": LAPLACE_NUMBER,
        "slip_length": SLIP_LENGTH,
        "slip_length_over_h": SLIP_LENGTH / h,
        "radius": RADIUS,
        "h": h,
        "box": geometry["box"],
        "cells": [nx, ny],
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "equilibrium_cap_nominal": equilibrium,
        "initial_cap": initial,
        "initial_centre": centre.tolist(),
        "initial_contacts_x": initial_contacts.tolist(),
        "initial_contact_vertex_gap_over_h": contact_vertex_gap_over_h,
        "initial_predicted_contact_line_speed": predicted_speed(initial_deg),
        "contact_wall_y": 0.0,
        "contact_wall_normal": list(CONTACT_WALL_NORMAL),
        "liquid_side": "phi<0",
        "dt0": DT0,
        "dt_divisor": dt_divisor,
        "dt": sched["dt"],
        "steps_protocol": sched["steps"],
        "steps": steps,
        "end_time": steps * sched["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="R/h")
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--dt-divisor", type=int, default=1, choices=DT_DIVISORS,
                        help="time step dt0/divisor (dt0 = 1/800)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="truncate the run (smoke tests)")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)
    record = generate(args.level, args.case, args.dt_divisor, args.output_dir,
                      max_steps=args.max_steps, force=args.force)
    print(json.dumps({key: record[key] for key in
                      ("level_R_over_h", "case", "dt", "steps", "output_cadence", "n_triangles")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
