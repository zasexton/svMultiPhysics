#!/usr/bin/env python3
"""Write one resolution level of the tank-at-rest benchmark (tracker M1).

Liquid at rest in a closed rectangular tank (2D) or box (3D) under gravity,
with a flat free surface at a height that is not aligned with the mesh.  The
unfitted level-set free surface has zero surface tension.  The sampled state
(phi = y - H, u = 0, p = rho g (H - y)) is an exact discrete equilibrium, so
the run must keep it to within the solver tolerance scale (decision D1).

The case is written for the new OOP solver: solver.xml, an affine simplex
background mesh (Triangle3 or Tetra4) with the initial fields, the wall face
files, and case.json with every parameter that verify.py needs.  See
README.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants, fixed for every level (principle P1).  README.md gives
# the source or the reason for each one.  Units: rho = g = 1, tank length 1.
# ---------------------------------------------------------------------------
LEVELS = {2: (8, 16, 32), 3: (8, 16)}       # cells per unit length (1/h)
DENSITY = 1.0
GRAVITY = 1.0                               # body force (0, -g, 0)
VISCOSITY = 5.0e-4                          # same fluid as linear_sloshing_2d
TANK_LENGTH = 1.0                           # x extent
TANK_HEIGHT = 0.75                          # y extent (gravity along -y)
TANK_WIDTH = 0.5                            # z extent (3D only)
# Fill height: H/h has fractional part 0.16, 0.32, 0.64 at 1/h = 8, 16, 32,
# so the interface crosses a cell row at a different cut fraction at every
# level and never passes through a vertex.
FILL_HEIGHT = 0.52
EXTERNAL_PRESSURE = 0.0
PERIODS = 5                                 # run length in lowest sloshing periods
STEPS_PER_PERIOD = 20
SNAPSHOTS = 20
LEVEL_SET_FIELD = "phi"
INTERFACE_DOMAIN_ID = "tank_at_rest_surface"
# Strong zero velocity on the normal component only (free slip).  The top wall
# is dry and carries no condition.
WALL_NORMAL_COMPONENT = {"wall_left": 0, "wall_right": 0, "wall_bottom": 1,
                         "wall_front": 2, "wall_back": 2}
VTK_TRIANGLE, VTK_TETRA = 5, 10


def walls(dim: int) -> tuple[str, ...]:
    base = ("wall_left", "wall_right", "wall_bottom", "wall_top")
    return base + (("wall_front", "wall_back") if dim == 3 else ())


def lowest_sloshing_period() -> float:
    """Inviscid period of the first mode along x, omega^2 = g k tanh(k H)."""
    k = math.pi / TANK_LENGTH
    return 2.0 * math.pi / math.sqrt(GRAVITY * k * math.tanh(k * FILL_HEIGHT))


def time_schedule() -> dict:
    period = lowest_sloshing_period()
    steps = PERIODS * STEPS_PER_PERIOD
    return {"sloshing_period": period, "dt": period / STEPS_PER_PERIOD,
            "steps": steps, "output_cadence": steps // SNAPSHOTS,
            "end_time": PERIODS * period}


def hydrostatic_pressure(y):
    return EXTERNAL_PRESSURE + DENSITY * GRAVITY * (FILL_HEIGHT - np.asarray(y))


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def _signed_measure(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p = points[cells]
    edges = p[:, 1:, :] - p[:, :1, :]
    if cells.shape[1] == 3:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.linalg.det(edges) / 6.0


def _boundary_faces(points: np.ndarray, cells: np.ndarray, extents) -> dict:
    """Boundary facets of a simplex mesh grouped by box side, with their parent cell."""
    nv = cells.shape[1]
    local = list(itertools.combinations(range(nv), nv - 1))
    count: dict[tuple, list] = {}
    for c, cell in enumerate(cells):
        for loc in local:
            key = tuple(sorted(int(cell[i]) for i in loc))
            count.setdefault(key, []).append(c)
    names = {(0, 0): "wall_left", (0, 1): "wall_right", (1, 0): "wall_bottom",
             (1, 1): "wall_top", (2, 0): "wall_front", (2, 1): "wall_back"}
    faces: dict[str, tuple[list, list]] = {}
    for key, owners in count.items():
        if len(owners) != 1:
            continue
        xyz = points[list(key)]
        side = None
        for axis in range(len(extents)):
            for end, value in ((0, 0.0), (1, extents[axis])):
                if np.all(np.abs(xyz[:, axis] - value) < 1e-12):
                    side = names[(axis, end)]
        if side is None:
            raise RuntimeError("boundary facet not on a box side")
        faces.setdefault(side, ([], []))
        faces[side][0].append(key)
        faces[side][1].append(owners[0])
    return faces


def structured_mesh(dim: int, level: int):
    """Box [0,L] x [0,H_tank] (x [0,W]) split into simplices.

    2D: right triangles with the diagonal alternating with cell parity.
    3D: the six-tetrahedron Kuhn (Freudenthal) split of every cube, which is
    conforming across cubes.  Every cell is positively oriented.
    """
    extents = [TANK_LENGTH, TANK_HEIGHT] + ([TANK_WIDTH] if dim == 3 else [])
    counts = []
    for e in extents:
        n = int(round(e * level))
        if not math.isclose(n / level, e, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("tank extents must be an integer number of cells")
        counts.append(n)
    axes = [np.linspace(0.0, e, n + 1) for e, n in zip(extents, counts)]
    grids = np.meshgrid(*axes, indexing="ij")
    coords = np.column_stack([g.ravel() for g in grids])
    points = np.zeros((coords.shape[0], 3))
    points[:, :dim] = coords
    strides = [int(np.prod([n + 1 for n in counts[a + 1:]])) for a in range(dim)]

    def vid(idx) -> int:
        return int(sum(i * s for i, s in zip(idx, strides)))

    cells = []
    if dim == 2:
        for i in range(counts[0]):
            for j in range(counts[1]):
                a, b = vid((i, j)), vid((i + 1, j))
                c, d = vid((i + 1, j + 1)), vid((i, j + 1))
                cells += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    else:
        for i in range(counts[0]):
            for j in range(counts[1]):
                for k in range(counts[2]):
                    base = np.array((i, j, k))
                    for perm in itertools.permutations(range(3)):
                        corner = base.copy()
                        tet = [vid(corner)]
                        for axis in perm:
                            corner = corner.copy()
                            corner[axis] += 1
                            tet.append(vid(corner))
                        cells.append(tuple(tet))
    cells = np.asarray(cells, dtype=np.int64)
    negative = _signed_measure(points[:, :dim], cells) < 0.0
    cells[negative, :2] = cells[negative, 1::-1]
    if np.any(_signed_measure(points[:, :dim], cells) <= 0.0):
        raise RuntimeError("degenerate cell")
    faces = _boundary_faces(points[:, :dim], cells, extents)
    return points, cells, faces, counts


# ---------------------------------------------------------------------------
# VTK writers (ASCII, so the files are easy to inspect)
# ---------------------------------------------------------------------------
def _ascii(values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    return "\n".join(" ".join(fmt.format(v) for v in flat[s:s + 6])
                     for s in range(0, flat.size, 6))


def _data_array(name, vtk_type, values, ncomp=None) -> list[str]:
    values = np.asarray(values)
    ncomp = ncomp or (1 if values.ndim == 1 else values.shape[1])
    fmt = "{:d}" if vtk_type.startswith(("Int", "UInt")) else "{:.17g}"
    comp = f' NumberOfComponents="{ncomp}"' if ncomp > 1 or name == "Points" else ""
    label = f' Name="{name}"' if name != "Points" else ""
    return [f'<DataArray type="{vtk_type}"{label}{comp} format="ascii">', _ascii(values, fmt),
            "</DataArray>"]


def write_vtu(path: Path, points, cells, point_data: dict, cell_data: dict) -> None:
    n_cells, nv = cells.shape
    cell_type = VTK_TRIANGLE if nv == 3 else VTK_TETRA
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<UnstructuredGrid>",
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">', "<PointData>"]
    for name, (vtk_type, values) in point_data.items():
        out += _data_array(name, vtk_type, values)
    out += ["</PointData>", "<CellData>"]
    for name, (vtk_type, values) in cell_data.items():
        out += _data_array(name, vtk_type, values)
    out += ["</CellData>", "<Points>"]
    out += _data_array("Points", "Float64", points, 3)
    out += ["</Points>", "<Cells>"]
    out += _data_array("connectivity", "Int64", cells.ravel())
    out += _data_array("offsets", "Int64", nv * np.arange(1, n_cells + 1))
    out += _data_array("types", "UInt8", np.full(n_cells, cell_type))
    out += ["</Cells>", "</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points, facets, parents) -> None:
    facets = np.asarray(facets, dtype=np.int64)
    node_ids = np.unique(facets)
    local = np.searchsorted(node_ids, facets)
    nf, nv = facets.shape
    kind = "Lines" if nv == 2 else "Polys"
    counts = {"Lines": nf if nv == 2 else 0, "Polys": nf if nv == 3 else 0}
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<PolyData>",
           f'<Piece NumberOfPoints="{node_ids.size}" NumberOfVerts="0" '
           f'NumberOfLines="{counts["Lines"]}" NumberOfStrips="0" NumberOfPolys="{counts["Polys"]}">',
           "<PointData>"]
    out += _data_array("GlobalNodeID", "Int64", node_ids)
    out += ["</PointData>", "<CellData>"]
    out += _data_array("GlobalElementID", "Int64", parents)
    out += ["</CellData>", "<Points>"]
    out += _data_array("Points", "Float64", points[node_ids], 3)
    out += ["</Points>", f"<{kind}>"]
    out += _data_array("connectivity", "Int64", local.ravel())
    out += _data_array("offsets", "Int64", nv * np.arange(1, nf + 1))
    out += [f"</{kind}>", "</Piece>", "</PolyData>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def linear_solver_block() -> str:
    return """    <LS type="Direct">
      <Linear_algebra type="eigen">
        <Preconditioner>none</Preconditioner>
      </Linear_algebra>
      <Max_iterations>100</Max_iterations>
      <Krylov_space_dimension>50</Krylov_space_dimension>
      <Tolerance>1.0e-8</Tolerance>
      <Absolute_tolerance>1.0e-10</Absolute_tolerance>
    </LS>"""


def effective_direction(wall: str, dim: int) -> str:
    flags = ["0"] * dim
    flags[WALL_NORMAL_COMPONENT[wall]] = "1"
    return " ".join(flags)


def solver_xml(dim: int, schedule: dict, steps: int, cadence: int) -> str:
    wall_names = walls(dim)
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in wall_names)
    bcs = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>{effective_direction(w, dim)}</Effective_direction>
    </Add_BC>""" for w in wall_names if w != "wall_top")
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- tank_at_rest benchmark, {dim}D; generated by generate_case.py -->
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>{dim}</Number_of_spatial_dimensions>
    <Number_of_time_steps>{steps}</Number_of_time_steps>
    <Time_step_size>{schedule['dt']:.17g}</Time_step_size>
    <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
    <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Combine_time_series>true</Combine_time_series>
    <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
    <Increment_in_saving_VTK_files>{cadence}</Increment_in_saving_VTK_files>
    <Start_saving_after_time_step>{cadence}</Start_saving_after_time_step>
    <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
    <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
    <Verbose>1</Verbose>
    <Warning>0</Warning>
    <Debug>0</Debug>
  </GeneralSimulationParameters>

  <Add_mesh name="tank">
    <Mesh_file_path>mesh/mesh-complete.mesh.vtu</Mesh_file_path>
{faces}
  </Add_mesh>

  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
    <Operator_tag>equations</Operator_tag>
    <Level_set_source>prescribed_data</Level_set_source>
    <Velocity_source>coupled_field</Velocity_source>
    <Velocity_field_name>Velocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Enable_SUPG>true</Enable_SUPG>
    <SUPG_tau_scale>0.5</SUPG_tau_scale>
    <SUPG_transient_scale>2.0</SUPG_transient_scale>
    <Enable_reinitialization>false</Enable_reinitialization>
    <Enable_volume_correction>false</Enable_volume_correction>
    <Output type="Spatial">
      <Level_set>true</Level_set>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{linear_solver_block()}
  </Add_equation>

  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>8</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
    <Density>{DENSITY:.17g}</Density>
    <Force_x>0.0</Force_x>
    <Force_y>{-GRAVITY:.17g}</Force_y>
    <Force_z>0.0</Force_z>
    <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
    <Viscosity model="Constant">
      <Value>{VISCOSITY:.17g}</Value>
    </Viscosity>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{linear_solver_block()}
{bcs}
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
      <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
      <Level_set_isovalue>0.0</Level_set_isovalue>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>0.0</Surface_tension>
      <Enable_velocity_extension>false</Enable_velocity_extension>
      <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
      <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
      <Use_cut_metadata_scale>false</Use_cut_metadata_scale>
      <Small_cut_aggregation>true</Small_cut_aggregation>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
"""


# ---------------------------------------------------------------------------
def generate(dim: int, level: int, output_dir: Path, *, max_steps: int | None = None,
             force: bool = False) -> dict:
    if dim not in LEVELS:
        raise ValueError("--dim must be 2 or 3")
    if level not in LEVELS[dim]:
        raise ValueError(f"--level must be one of {LEVELS[dim]} in {dim}D")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule()
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, counts = structured_mesh(dim, level)
    h = 1.0 / level
    y = points[:, 1]
    phi = y - FILL_HEIGHT
    pressure = hydrostatic_pressure(y)
    n_points = points.shape[0]

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(n_points)),
               LEVEL_SET_FIELD: ("Float64", phi),
               "Velocity": ("Float64", np.zeros((n_points, 3))),
               # The linear hydrostatic profile on every vertex: dry vertices
               # of cut cells carry its signed continuation, so the P1
               # pressure is exactly rho g (H - y) on every retained cell.
               "Pressure": ("Float64", pressure)},
              {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in walls(dim):
        facets, parents = faces[wall]
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, facets, parents)
    (output_dir / "solver.xml").write_text(solver_xml(dim, schedule, steps, cadence),
                                           encoding="utf-8")

    extents = [TANK_LENGTH, TANK_HEIGHT] + ([TANK_WIDTH] if dim == 3 else [])
    base_area = TANK_LENGTH * (TANK_WIDTH if dim == 3 else 1.0)
    case = {
        "benchmark": "tank_at_rest",
        "generator": "tests/cases/fluid/free_surface_benchmarks/tank_at_rest/generate_case.py",
        "dim": dim,
        "level_cells_per_length": level,
        "h": h,
        "extents": extents,
        "cells_per_side": counts,
        "n_vertices": int(n_points),
        "n_cells": int(cells.shape[0]),
        "density": DENSITY,
        "gravity": GRAVITY,
        "viscosity": VISCOSITY,
        "fill_height": FILL_HEIGHT,
        "liquid_volume_exact": base_area * FILL_HEIGHT,
        "external_pressure": EXTERNAL_PRESSURE,
        "pressure_scale": DENSITY * GRAVITY * FILL_HEIGHT,
        "velocity_scale": math.sqrt(GRAVITY * FILL_HEIGHT),
        "level_set_field": LEVEL_SET_FIELD,
        "velocity_field": "Velocity",
        "pressure_field": "Pressure",
        "liquid_side": "phi<0",
        "sloshing_period": schedule["sloshing_period"],
        "dt": schedule["dt"],
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "min_abs_phi_over_h": float(np.min(np.abs(phi)) / h),
        "wall_bc": "free slip: strong zero normal velocity (Effective_direction), top wall dry",
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, required=True, choices=(2, 3))
    parser.add_argument("--level", type=int, required=True,
                        help="cells per unit length: 8, 16, 32 in 2D; 8, 16 in 3D")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; verify.py rejects "
                             "such runs for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)
    case = generate(args.dim, args.level, args.output_dir, max_steps=args.max_steps,
                    force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("dim", "level_cells_per_length", "n_vertices", "n_cells", "fill_height",
                "min_abs_phi_over_h", "dt", "steps", "output_cadence", "end_time", "truncated"):
        print(f"  {key} = {case[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
