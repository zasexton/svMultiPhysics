#!/usr/bin/env python3
"""Write the MeshNitsche fitted-ALE decks of SPHERIC Test 10 (lateral water 1x).

Two decks are written next to the legacy fitted decks, which stay unchanged:

* ``fitted_ale/spheric_test10_lateral_water_1x_meshnitsche`` (3D): the
  structured tetrahedral mesh of the legacy 3D deck (same nodes and cells);
* ``fitted_ale/spheric_test10_lateral_water_1x_2d_meshnitsche`` (2D): the
  x-y section of the same tank and fill, on a finer structured triangle
  mesh.

Both use ``Kinematic_enforcement=MeshNitsche`` (gamma_N = 10), the ``Free``
tangential mesh policy, the harmonic mesh-velocity operator, free-slip fluid
walls and sliding mesh walls (the same zero-valued ``Dir`` input with
``Effective_direction`` in both equations), zero surface tension, and a
hydrostatic initial pressure without a pressure gauge.  The committed decks
hold the tank at rest (no roll forcing).

Face sets.  Every boundary face belongs to exactly one face file: a face is
put on a tank plane only if all of its vertices lie on that plane (to 1e-9 of
the tank size), the sets are checked to be disjoint and to cover the whole
boundary, and each face records the GlobalElementID of its parent cell.  The
legacy decks were written by generate_validation_meshes.py, whose centroid
test with a tolerance of 0.35 h put the wall faces of the top cell row into
free_surface.vtp as well (and bottom/side faces into the wall files); the
solver keeps one boundary label per face (the last face file listed), so
those wall faces became free-surface faces and the wall Dirichlet conditions
missed every node on the free-surface edge (fitted_ale/README.md).

``--roll-forcing FILE`` writes a forced variant into ``--output-dir``: the
tank-frame body force of the published roll history (rotated gravity, Euler
and centrifugal accelerations, generate_spheric_test10_roll_body_force.py),
and in 3D the rotating-frame angular velocity for the Coriolis term.  FILE is
``lateral_water_1x.txt`` of the SPHERIC Test 10 archive
(fetch_spheric_test10_reference.py).  The forcing tables are large and derived
from external data, so they are not committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "fitted_ale"
DECK_3D = "spheric_test10_lateral_water_1x_meshnitsche"
DECK_2D = "spheric_test10_lateral_water_1x_2d_meshnitsche"

WATER_DENSITY = 998.2
WATER_VISCOSITY = 1.003e-3
GRAVITY = 9.81
TANK_LENGTH = 0.900
TANK_BREADTH = 0.062
TANK_HEIGHT = 0.508
FILL_HEIGHT = 0.093
ROTATION_AXIS_POINT = (0.45, 0.0, 0.031)
SENSOR1 = (0.0, FILL_HEIGHT, 0.031)          # SPHERIC Test 10 Figure 1, left wall
KINEMATIC_NITSCHE_GAMMA = 10.0               # fixed once (P1), as fitted_sloshing_2d
MESH_MOTION_KAPPA = 1.0
PLANE_TOLERANCE = 1.0e-9                     # relative to the tank length

# Legacy 3D resolution (generate_validation_meshes.generate_spheric_test10).
LEGACY_MAX_SPACING_X = 0.045
LEGACY_MAX_SPACING_Z = 0.031
# 2D resolution: 120 x 12 cells (dx = 7.5 mm, dy = 7.75 mm).
CELLS_2D = (120, 12)

DT = 1.0e-3
STEPS_AT_REST = 100
FORCED_END_TIME = 8.35                       # length of the published roll record
FORCED_OUTPUT_CADENCE = 5
FORCING_SAMPLE_DT = 0.01                     # time sampling of the forcing table

WALLS_2D = {"wall_left": (0, 0.0), "wall_right": (0, TANK_LENGTH), "wall_bottom": (1, 0.0)}
WALLS_3D = {**WALLS_2D, "wall_front": (2, 0.0), "wall_back": (2, TANK_BREADTH)}
EFFECTIVE_DIRECTION = {"wall_left": 0, "wall_right": 0, "wall_bottom": 1,
                       "wall_front": 2, "wall_back": 2}
FREE_SURFACE = "free_surface"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_fitted_decks", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def tank_triangles(nx: int, ny: int):
    """Structured triangles on [0, L] x [0, H]; diagonals alternate with parity."""
    xs = np.linspace(0.0, TANK_LENGTH, nx + 1)
    ys = np.linspace(0.0, FILL_HEIGHT, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cells += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return points, np.asarray(cells, dtype=np.int64)


def legacy_tank_tetrahedra():
    """The structured tetrahedral mesh of the legacy 3D fitted deck."""
    gvm = _load("generate_validation_meshes")
    x = gvm.segmented_coordinates((0.0, 0.5 * TANK_LENGTH, TANK_LENGTH), LEGACY_MAX_SPACING_X)
    y = gvm.segmented_coordinates((0.0, 0.5 * FILL_HEIGHT, FILL_HEIGHT), LEGACY_MAX_SPACING_X)
    z = gvm.segmented_coordinates((0.0, 0.5 * TANK_BREADTH, TANK_BREADTH), LEGACY_MAX_SPACING_Z)
    grid = gvm.structured_tet_grid(x, y, z, mirror_z_midplane=True)
    return np.asarray(grid.points, dtype=float), gvm.grid_tets(grid)


def boundary_facets(cells: np.ndarray) -> list[tuple[tuple[int, ...], int]]:
    """(facet vertices, parent cell) of every facet that belongs to one cell."""
    if cells.shape[1] == 3:
        local = ((0, 1), (1, 2), (2, 0))
    else:
        local = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    seen: dict[tuple[int, ...], list] = {}
    for c, cell in enumerate(cells):
        for lf in local:
            facet = tuple(int(cell[i]) for i in lf)
            key = tuple(sorted(facet))
            if key in seen:
                seen[key][2] += 1
            else:
                seen[key] = [facet, c, 1]
    return [(facet, parent) for facet, parent, count in seen.values() if count == 1]


def classify_boundary(points: np.ndarray, cells: np.ndarray, planes: dict) -> dict:
    """Disjoint face sets: a facet lies on a plane iff all its vertices do.

    planes maps a face name to (axis, value).  Raises if a boundary facet
    lies on no plane or on more than one.
    """
    tol = PLANE_TOLERANCE * TANK_LENGTH
    sets = {name: ([], []) for name in planes}
    for facet, parent in boundary_facets(cells):
        coords = points[list(facet)]
        owners = [name for name, (axis, value) in planes.items()
                  if np.all(np.abs(coords[:, axis] - value) <= tol)]
        if len(owners) != 1:
            raise RuntimeError(f"boundary facet {facet} lies on {len(owners)} tank planes {owners}")
        sets[owners[0]][0].append(facet)
        sets[owners[0]][1].append(parent)
    for name, (facets, _) in sets.items():
        if not facets:
            raise RuntimeError(f"face set {name!r} is empty")
    return sets


# ---------------------------------------------------------------------------
# Writers (ASCII VTK XML)
# ---------------------------------------------------------------------------
def _ascii(values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    return "\n".join(" ".join(fmt.format(v) for v in flat[s:s + 6]) for s in range(0, flat.size, 6))


def _array(vtk_type: str, name: str, values, fmt: str) -> list[str]:
    values = np.asarray(values)
    ncomp = 1 if values.ndim == 1 else values.shape[1]
    return [f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">',
            _ascii(values, fmt), "</DataArray>"]


def write_vtu(path: Path, points: np.ndarray, cells: np.ndarray, point_data: dict) -> None:
    n_cells, nv = cells.shape
    cell_type = {3: 5, 4: 10}[nv]                       # VTK_TRIANGLE, VTK_TETRA
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<UnstructuredGrid>",
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">', "<PointData>"]
    for name, (vtk_type, values) in point_data.items():
        out += _array(vtk_type, name, values, "{:d}" if vtk_type.startswith("Int") else "{:.17g}")
    out += ["</PointData>", "<CellData>"]
    out += _array("Int64", "GlobalElementID", np.arange(n_cells), "{:d}")
    out += ["</CellData>", "<Points>"]
    out += _array("Float64", "Points", points, "{:.17g}")
    out += ["</Points>", "<Cells>"]
    out += _array("Int64", "connectivity", cells.ravel(), "{:d}")
    out += _array("Int64", "offsets", nv * np.arange(1, n_cells + 1), "{:d}")
    out += _array("UInt8", "types", np.full(n_cells, cell_type), "{:d}")
    out += ["</Cells>", "</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points: np.ndarray, facets: list, parents: list) -> None:
    """Face file with GlobalNodeID (point data) and the parent-cell GlobalElementID."""
    used = sorted({v for f in facets for v in f})
    local = {v: i for i, v in enumerate(used)}
    conn = np.array([[local[v] for v in f] for f in facets], dtype=np.int64)
    nv = conn.shape[1]
    kind = "Lines" if nv == 2 else "Polys"
    counts = {"Lines": (len(facets), 0), "Polys": (0, len(facets))}[kind]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<PolyData>",
           f'<Piece NumberOfPoints="{len(used)}" NumberOfVerts="0" NumberOfLines="{counts[0]}" '
           f'NumberOfStrips="0" NumberOfPolys="{counts[1]}">', "<PointData>"]
    out += _array("Int64", "GlobalNodeID", np.array(used), "{:d}")
    out += ["</PointData>", "<CellData>"]
    out += _array("Int64", "GlobalElementID", np.array(parents), "{:d}")
    out += ["</CellData>", "<Points>"]
    out += _array("Float64", "Points", points[used], "{:.17g}")
    out += ["</Points>", f"<{kind}>"]
    out += _array("Int64", "connectivity", conn.ravel(), "{:d}")
    out += _array("Int64", "offsets", nv * np.arange(1, len(facets) + 1), "{:d}")
    out += [f"</{kind}>", "</Piece>", "</PolyData>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def _direction(dim: int, axis: int) -> str:
    return " ".join("1" if d == axis else "0" for d in range(dim))


def solver_xml(dim: int, faces: list[str], *, steps: int, dt: float, cadence: int,
               body_force_file: str | None = None, angular_velocity_file: str | None = None) -> str:
    add_faces = "\n".join(
        f'  <Add_face name="{f}">\n    <Face_file_path>mesh/water/mesh-surfaces/{f}.vtp</Face_file_path>\n  </Add_face>'
        for f in faces)
    walls = [f for f in faces if f != FREE_SURFACE]
    wall_bcs = "\n".join(f"""  <Add_BC name="{w}">
    <Type>Dir</Type>
    <Value>0.0</Value>
    <Effective_direction>{_direction(dim, EFFECTIVE_DIRECTION[w])}</Effective_direction>
  </Add_BC>""" for w in walls)
    forcing = ""
    if body_force_file:
        forcing += (f"\n  <Momentum_source_temporal_and_spatial_values_file_path>{body_force_file}"
                    "</Momentum_source_temporal_and_spatial_values_file_path>")
    if angular_velocity_file:
        forcing += (f"\n  <Rotating_frame_angular_velocity_temporal_values_file_path>{angular_velocity_file}"
                    "</Rotating_frame_angular_velocity_temporal_values_file_path>")
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- SPHERIC Test 10 lateral water 1x, fitted ALE with MeshNitsche kinematics and sliding
     walls ({dim}D); generated by generate_spheric_test10_fitted_decks.py -->
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>{dim}</Number_of_spatial_dimensions>
  <Number_of_time_steps>{steps}</Number_of_time_steps>
  <Time_step_size>{dt:.17g}</Time_step_size>
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
  <Mesh_file_path>mesh/water/mesh-complete.mesh.vtu</Mesh_file_path>
{add_faces}
</Add_mesh>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>12</Max_iterations>
  <Tolerance>1.0e-6</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>

  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Mesh_velocity_field>mesh_velocity</Mesh_velocity_field>
  <Mesh_displacement_field>mesh_displacement</Mesh_displacement_field>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Moving_mesh_tangent_path>SymbolicRequired</Moving_mesh_tangent_path>

  <Density>{WATER_DENSITY:.17g}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>{-GRAVITY:.17g}</Force_y>
  <Force_z>0.0</Force_z>{forcing}
  <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
  <Viscosity model="Constant">
    <Value>{WATER_VISCOSITY:.17g}</Value>
  </Viscosity>

  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Mesh_displacement>true</Mesh_displacement>
    <Mesh_velocity>true</Mesh_velocity>
  </Output>

  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>

{wall_bcs}

  <Add_BC name="{FREE_SURFACE}">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Tangential_mesh_policy>Free</Tangential_mesh_policy>
    <Kinematic_enforcement>MeshNitsche</Kinematic_enforcement>
    <Kinematic_nitsche_gamma>{KINEMATIC_NITSCHE_GAMMA:.17g}</Kinematic_nitsche_gamma>
  </Add_BC>
</Add_equation>

<Add_equation type="mesh_motion">
  <Coupled>true</Coupled>
  <Model>Harmonic</Model>
  <Field_name>mesh_displacement</Field_name>
  <Operator_tag>equations</Operator_tag>
  <Kappa>{MESH_MOTION_KAPPA:.17g}</Kappa>
  <Harmonic_quantity>velocity</Harmonic_quantity>
  <Moving_mesh_tangent_path>SymbolicRequired</Moving_mesh_tangent_path>
  <Module_options>jit=true; jit_specialization=true</Module_options>

{wall_bcs}
</Add_equation>

</svMultiPhysicsFile>
"""


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------
def hydrostatic_pressure(points: np.ndarray) -> np.ndarray:
    return WATER_DENSITY * GRAVITY * (FILL_HEIGHT - points[:, 1])


def mesh_for(dim: int):
    if dim == 2:
        points, cells = tank_triangles(*CELLS_2D)
        planes = {**WALLS_2D, FREE_SURFACE: (1, FILL_HEIGHT)}
    else:
        points, cells = legacy_tank_tetrahedra()
        planes = {**WALLS_3D, FREE_SURFACE: (1, FILL_HEIGHT)}
    return points, cells, classify_boundary(points, cells, planes)


def metadata(dim: int, points: np.ndarray, cells: np.ndarray, sets: dict, *, steps: int,
             forcing: dict | None) -> dict:
    sensor = list(SENSOR1[:dim]) + ([0.0] if dim == 2 else [])
    return {
        "benchmark": "SPHERIC Test 10 lateral water 1x" + (" (2D x-y section)" if dim == 2 else ""),
        "representation": "fitted_ale",
        "generator": "tests/cases/fluid/open_vessel_free_surface/generate_spheric_test10_fitted_decks.py",
        "source_urls": ["https://www.spheric-sph.org/tests/test-10"],
        "dimensions_m": {"tank_length": TANK_LENGTH, "tank_breadth_1x": TANK_BREADTH,
                         "tank_height": TANK_HEIGHT, "initial_fill_height": FILL_HEIGHT},
        "mesh": {"dimension": dim, "points": int(points.shape[0]), "cells": int(cells.shape[0]),
                 "cell_type": "Triangle3" if dim == 2 else "Tetra4",
                 "structured_cells": list(CELLS_2D) if dim == 2 else "legacy 3D deck mesh (20 x 4 x 2 hexahedra, 6 tetrahedra each)",
                 "face_sets": {name: len(facets) for name, (facets, _) in sets.items()},
                 "face_set_rule": "a boundary facet belongs to the tank plane on which all its vertices lie (tolerance 1e-9 L); the sets are disjoint and cover the boundary; GlobalElementID is the parent cell"},
        "formulation": {"kinematic_enforcement": "MeshNitsche",
                        "kinematic_nitsche_gamma": KINEMATIC_NITSCHE_GAMMA,
                        "tangential_mesh_policy": "Free",
                        "mesh_motion": "Harmonic, Harmonic_quantity=velocity, Kappa=1",
                        "walls": "fluid free slip and sliding mesh (zero wall-normal component)",
                        "surface_tension": 0.0,
                        "pressure": "hydrostatic initial field, no gauge (natural free-surface traction)"},
        "time": {"dt": DT, "steps": steps},
        "initial_liquid_volume": float(np.prod([TANK_LENGTH, FILL_HEIGHT] + ([TANK_BREADTH] if dim == 3 else []))),
        "pressure_sensor": {"name": "Sensor1", "case": "lateral_water_1x", "coordinates": sensor,
                            "source": "SPHERIC Test10 Figure 1",
                            "role": "literature_pressure_history_sensor"},
        "rotation_axis": {"point": list(ROTATION_AXIS_POINT[:dim]) + ([0.0] if dim == 2 else []),
                          "direction": [0.0, 0.0, 1.0],
                          "source": "SPHERIC Test10 Figure 1: rotation axis at the center of the bottom line"},
        "roll_forcing": forcing,
    }


def write_body_force(path: Path, dim: int, times: np.ndarray, increments: list) -> None:
    """Temporal and spatial values file with one component per space dimension.

    The layout of generate_spheric_test10_roll_body_force.write_source (the
    solver requires ndof to equal the mesh dimension): a header
    'ndof ntimes nnodes', the times, then per node its 1-based GlobalNodeID
    and one row of ndof values per time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_nodes = increments[0].shape[0]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{dim} {len(times)} {n_nodes}\n")
        for time in times:
            stream.write(f"{time:.12e}\n")
        for node in range(n_nodes):
            stream.write(f"{node + 1}\n")
            for values in increments:
                stream.write(" ".join(f"{v:.18e}" for v in values[node, :dim]) + "\n")


def write_forcing(case_dir: Path, dim: int, points: np.ndarray, reference_file: Path,
                  end_time: float, *, depth_dependence: bool = False) -> dict:
    """Tank-frame roll forcing tables (generate_spheric_test10_roll_body_force.py).

    By default the Euler and centrifugal terms alpha y_r and omega^2 y_r are
    evaluated at the mid-depth y_c = H/2 of the still liquid, so the table
    depends on x only and the solver interpolates it exactly along x (its
    x-only interpolant, a binary search).  A table that also varies with y
    falls back to an inverse-distance search over all nodes at every
    quadrature point, which costs about 18 s per step on the 2D deck (job
    46129890).  The neglected part is at most alpha_max H/2 = 0.057 m/s^2
    (0.6% of g) in the x component.
    """
    rbf = _load("generate_spheric_test10_roll_body_force")
    reference = rbf.load_reference(reference_file)
    times = np.arange(0.0, end_time + 0.5 * FORCING_SAMPLE_DT, FORCING_SAMPLE_DT)
    sampled = rbf.sample_reference(reference, times)
    axis = np.array(ROTATION_AXIS_POINT[:2] + (ROTATION_AXIS_POINT[2] if dim == 3 else 0.0,))
    base = np.array([0.0, -GRAVITY, 0.0])
    sample_points = points.copy()
    if not depth_dependence:
        sample_points[:, 1] = 0.5 * FILL_HEIGHT
    increments = [rbf.roll_incremental_acceleration(sample_points, axis_point=axis, base_force=base,
                                                    theta=float(th), omega=float(om),
                                                    alpha=float(al), gravity_magnitude=GRAVITY)
                  for th, om, al in zip(sampled["theta_rad"], sampled["omega_rad_s"],
                                        sampled["alpha_rad_s2"])]
    bc = case_dir / "bc"
    body = bc / "test10_lateral_water_1x_roll_body_force.dat"
    write_body_force(body, dim, times, increments)
    info = {"reference_file": str(reference_file), "sample_dt": FORCING_SAMPLE_DT,
            "end_time": float(times[-1]),
            "body_force_file": "bc/" + body.name,
            "model": "tank frame: rotated gravity, Euler and centrifugal accelerations as a nodal "
                     "body-force table on the reference nodes (interpolated by the solver at the "
                     "current quadrature points)",
            "depth_dependence": ("evaluated at every node" if depth_dependence else
                                 "alpha y_r and omega^2 y_r evaluated at y_c = H/2 (x-only table); "
                                 "neglected part <= alpha_max H/2 = 0.057 m/s^2")}
    if dim == 3:
        omega = bc / "test10_lateral_water_1x_roll_angular_velocity.dat"
        rbf.write_angular_velocity(omega, times=times, omega=sampled["omega_rad_s"])
        info["angular_velocity_file"] = "bc/" + omega.name
        info["coriolis"] = "included (Rotating_frame_angular_velocity_temporal_values_file_path)"
    else:
        info["angular_velocity_file"] = None
        info["coriolis"] = ("omitted: the solver's rotating-frame Coriolis term requires a 3D mesh; "
                            "2 |Omega| |u| <= 2 x 0.27 rad/s x |u|")
    return info


def write_deck(dim: int, case_dir: Path, *, reference_file: Path | None = None,
               end_time: float | None = None, depth_dependence: bool = False,
               force: bool = False) -> dict:
    if case_dir.exists():
        if not force and any(case_dir.iterdir()):
            raise FileExistsError(f"{case_dir} is not empty (use --force)")
        shutil.rmtree(case_dir)
    points, cells, sets = mesh_for(dim)
    mesh_dir = case_dir / "mesh" / "water"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    zeros = np.zeros((points.shape[0], 3))
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(points.shape[0])),
               "Velocity": ("Float64", zeros),
               "Pressure": ("Float64", hydrostatic_pressure(points)),
               "mesh_displacement": ("Float64", zeros),
               "mesh_velocity": ("Float64", zeros)})
    for name, (facets, parents) in sets.items():
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{name}.vtp", points, facets, parents)
    forcing = None
    steps, cadence = STEPS_AT_REST, 1
    if reference_file is not None:
        end = FORCED_END_TIME if end_time is None else end_time
        forcing = write_forcing(case_dir, dim, points, reference_file, end,
                                depth_dependence=depth_dependence)
        steps, cadence = int(round(end / DT)), FORCED_OUTPUT_CADENCE
    faces = list(sets)
    (case_dir / "solver.xml").write_text(
        solver_xml(dim, faces, steps=steps, dt=DT, cadence=cadence,
                   body_force_file=forcing["body_force_file"] if forcing else None,
                   angular_velocity_file=forcing.get("angular_velocity_file") if forcing else None),
        encoding="utf-8")
    meta = metadata(dim, points, cells, sets, steps=steps, forcing=forcing)
    (case_dir / "benchmark.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, choices=(2, 3), action="append",
                        help="deck dimension(s) to write (default: both)")
    parser.add_argument("--output-dir", type=Path,
                        help="write one deck here instead of the committed location "
                             "(requires a single --dim)")
    parser.add_argument("--roll-forcing", type=Path, metavar="LATERAL_WATER_1X_TXT",
                        help="write the forced variant (requires --output-dir)")
    parser.add_argument("--end-time", type=float, default=None,
                        help=f"forced run length in s (default {FORCED_END_TIME})")
    parser.add_argument("--forcing-depth-dependence", action="store_true",
                        help="evaluate the Euler and centrifugal terms at every node (slow: the "
                             "solver then searches all nodes at each quadrature point)")
    parser.add_argument("--force", action="store_true", help="replace a non-empty output dir")
    args = parser.parse_args(argv)
    dims = args.dim or [2, 3]
    if args.roll_forcing and args.output_dir is None:
        parser.error("--roll-forcing writes a run directory: give --output-dir")
    if args.output_dir is not None and len(dims) != 1:
        parser.error("--output-dir needs exactly one --dim")
    for dim in dims:
        case_dir = args.output_dir or (ROOT / (DECK_2D if dim == 2 else DECK_3D))
        meta = write_deck(dim, case_dir, reference_file=args.roll_forcing,
                          end_time=args.end_time, depth_dependence=args.forcing_depth_dependence,
                          force=args.force or args.output_dir is None)
        print(f"wrote {case_dir}: {meta['mesh']['points']} points, {meta['mesh']['cells']} cells, "
              f"faces {meta['mesh']['face_sets']}, steps {meta['time']['steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
