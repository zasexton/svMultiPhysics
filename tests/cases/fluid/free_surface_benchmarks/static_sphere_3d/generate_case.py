#!/usr/bin/env python3
"""Write one resolution level of the 3D static-sphere benchmark (tracker M2).

A spherical liquid drop of radius R, fully enclosed by its free surface, sits
in a cubic box with zero gravity and exterior pressure p_ext = 0.  The
sampled analytic state (phi = |x - c| - R, u = 0, p = 2 gamma/R) is released
and the flow relaxes for a fixed number of viscous times (decision D3).  It
is the 3D analogue of static_drop_2d.

--transport selects the level-set advection velocity (decision D9), with the
same choices and default rule as capillary_wave_2d.

The case is written for the new OOP solver: solver.xml, an affine Tetra4
background mesh with the initial fields, the six wall face files, and
case.json with every parameter that verify.py needs.  The VTK files are
written in base64 binary form because the R/h = 32 mesh has 5.3 million
cells.  See README.md.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants.  They are fixed for every level and every capillary form
# (principle P1); README.md gives the source or the derivation of each one.
# ---------------------------------------------------------------------------
LEVELS = (8, 16, 32)                        # R/h
CAPILLARY_FORMS = ("surface_stress", "kag_consistent", "kag_lumped")
# Level-set advection velocity (decision D9), as in capillary_wave_2d.
# "pde_extension" (default) is the harmonic PDE velocity extension with
# monolithic coupling; "wet_extension" is the algebraic wall-compatible wet
# extension, the D9 comparison baseline; "coupled" is the fluid velocity
# itself (dry vertices then carry zero velocity, which D9 retires).
TRANSPORTS = ("pde_extension", "wet_extension", "coupled")
PDE_EXTENSION_METHOD: str | None = "pde_harmonic"
PDE_EXTENSION_COUPLING: str | None = "monolithic"
DEFAULT_TRANSPORT = "pde_extension" if PDE_EXTENSION_METHOD else "wet_extension"
DENSITY = 1.0                               # rho
SURFACE_TENSION = 1.0                       # gamma
RADIUS = 1.0                                # R (length unit)
BOX_SIDE = 3.0 * RADIUS                     # cube [0, 3R]^3, as the 2D box
# Fixed, non-grid-aligned centre: (pi, e, sqrt(3))/100 * R from the box
# centre.  The offsets are irrational, so the sphere passes through no vertex
# of the nested dyadic grids h = R/level; among the simple irrational triples
# tried (README.md) this one has the largest min |phi|/h over R/h = 8, 16, 32.
CENTRE_OFFSET = (math.pi / 100.0 * RADIUS, math.e / 100.0 * RADIUS,
                 math.sqrt(3.0) / 100.0 * RADIUS)
EXTERNAL_PRESSURE = 0.0
DEFAULT_VISCOUS_TIMES = 5.0                 # run length in units rho*R^2/mu
DEFAULT_SNAPSHOTS = 100                     # VTU outputs per run
# Capillary time-step limit (Brackbill, Kothe & Zemach 1992) evaluated with
# the one-sided density sum of a free surface (rho_liquid + rho_void = rho):
#   dt_B = sqrt(rho h^3 / (4 pi gamma)) = SAFETY * sqrt(rho h^3 / (2 pi gamma)).
DT_SAFETY = 1.0 / math.sqrt(2.0)
# Multiple of dt_B used as the time step, per Laplace number.  Taken from the
# 2D step-0 measurement (tracker M2, 2026-09-30, jobs 46075447 and 46076505):
# the outer geometry loop accepts 2 dt_B at La = 12 and dt_B at La = 120 with
# its default 12-pass cap.  Other Laplace numbers use dt_B.  The 3D values are
# assumed from 2D and still have to be checked (README.md).
DT_MULTIPLE = {12.0: 2.0, 120.0: 1.0}
MIN_PHI_OVER_H_WARNING = 1.0e-6             # "vertex touch" warning threshold
LEVEL_SET_FIELD = "phi"
CURVATURE_FIELD = "kappa"
INTERFACE_DOMAIN_ID = "static_sphere_surface"
# (name, axis, end): the wall lies on the plane x[axis] = end * BOX_SIDE.
WALL_PLANES = (("wall_left", 0, 0), ("wall_right", 0, 1),
               ("wall_bottom", 1, 0), ("wall_top", 1, 1),
               ("wall_front", 2, 0), ("wall_back", 2, 1))
WALLS = tuple(name for name, _, _ in WALL_PLANES)
VTK_TETRA = 10
VTK_TYPES = {"Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}


def viscosity_from_laplace(laplace: float) -> float:
    """mu from La = rho*gamma*D/mu^2 with D = 2R."""
    return math.sqrt(DENSITY * SURFACE_TENSION * 2.0 * RADIUS / laplace)


def capillary_dt_limit(h: float) -> float:
    """Tracker form sqrt(rho h^3 / (2 pi gamma)), before the safety factor."""
    return math.sqrt(DENSITY * h ** 3 / (2.0 * math.pi * SURFACE_TENSION))


def dt_multiple(laplace: float) -> float:
    return DT_MULTIPLE.get(float(laplace), 1.0)


def time_schedule(level: int, laplace: float, viscous_times: float,
                  snapshots: int, multiple: float | None = None) -> dict:
    h = RADIUS / level
    mu = viscosity_from_laplace(laplace)
    viscous_time = DENSITY * RADIUS ** 2 / mu
    end_time = viscous_times * viscous_time
    m = dt_multiple(laplace) if multiple is None else multiple
    dt_b = DT_SAFETY * capillary_dt_limit(h)
    dt_max = m * dt_b
    cadence = max(1, math.ceil(end_time / (snapshots * dt_max)))
    steps = snapshots * cadence
    return {
        "h": h,
        "viscosity": mu,
        "viscous_time": viscous_time,
        "capillary_time": math.sqrt(DENSITY * RADIUS ** 3 / SURFACE_TENSION),
        "end_time": end_time,
        "dt_capillary_limit": capillary_dt_limit(h),
        "dt_B": dt_b,
        "dt_multiple_of_dt_B": m,
        "dt_max": dt_max,
        "dt": end_time / steps,
        "steps": steps,
        "output_cadence": cadence,
    }


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def _kuhn_paths() -> np.ndarray:
    """Local corners (4 x 3, entries 0/1) of the six Kuhn tetrahedra of the unit cube."""
    paths = []
    for perm in itertools.permutations(range(3)):
        corner = np.zeros(3, dtype=np.int64)
        path = [corner.copy()]
        for axis in perm:
            corner[axis] = 1
            path.append(corner.copy())
        paths.append(path)
    return np.asarray(paths, dtype=np.int64)                  # (6, 4, 3)


def signed_volume(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p = points[cells]
    return np.linalg.det(p[:, 1:, :] - p[:, :1, :]) / 6.0


def structured_tetra_mesh(level: int):
    """Cube [0, BOX_SIDE]^3 split into affine tetrahedra.

    Each cube is split into the six Kuhn (Freudenthal) tetrahedra around one
    of its main diagonals.  The split of cube (i, j, k) is the reference split
    reflected along every axis whose cube index is odd, so the pattern is
    symmetric under reflection about the grid planes (the 3D analogue of the
    alternating diagonal of static_drop_2d).  Neighbouring reflected splits
    share their face diagonals, so the mesh is conforming.  Every cell is
    positively oriented.
    """
    n = int(round(BOX_SIDE * level / RADIUS))
    if not math.isclose(n * RADIUS / level, BOX_SIDE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("box side must be an integer number of cells")
    m = n + 1
    coords = np.linspace(0.0, BOX_SIDE, m)
    idx = np.arange(m)
    gi, gj, gk = np.meshgrid(idx, idx, idx, indexing="ij")
    points = np.column_stack([coords[gi.ravel()], coords[gj.ravel()], coords[gk.ravel()]])

    cube = np.arange(n)
    ci, cj, ck = (a.ravel() for a in np.meshgrid(cube, cube, cube, indexing="ij"))
    base = np.column_stack([ci, cj, ck])                     # cube index c = (ci*n + cj)*n + ck
    mirror = base % 2
    paths = _kuhn_paths()
    cells = np.empty((base.shape[0], 6, 4), dtype=np.int64)
    for t in range(6):
        for s in range(4):
            g = base + (paths[t, s][None, :] ^ mirror)
            cells[:, t, s] = (g[:, 0] * m + g[:, 1]) * m + g[:, 2]
    cells = cells.reshape(-1, 4)                              # cell 6c + t lies in cube c
    negative = signed_volume(points, cells) < 0.0
    cells[negative, :2] = cells[negative, 1::-1]
    if np.any(signed_volume(points, cells) <= 0.0):
        raise RuntimeError("degenerate cell")

    faces = {}
    local_faces = [tuple(v for v in range(4) if v != omit) for omit in range(4)]
    for name, axis, end in WALL_PLANES:
        layer = 0 if end == 0 else n - 1
        cube_ids = np.nonzero(base[:, axis] == layer)[0]
        cell_ids = (6 * cube_ids[:, None] + np.arange(6)[None, :]).ravel()
        vertex_index = [cells[cell_ids] // (m * m), (cells[cell_ids] // m) % m, cells[cell_ids] % m]
        on_plane = vertex_index[axis] == end * n              # (cells, 4)
        tris, parents = [], []
        for lf in local_faces:
            mask = np.all(on_plane[:, lf], axis=1)
            tris.append(cells[cell_ids[mask]][:, lf])
            parents.append(cell_ids[mask])
        tris = np.concatenate(tris)
        parents = np.concatenate(parents)
        if tris.shape[0] != 2 * n * n:
            raise RuntimeError(f"{name}: expected {2 * n * n} boundary triangles, found {tris.shape[0]}")
        order = np.argsort(parents, kind="stable")
        faces[name] = (tris[order], parents[order])
    return points, cells, faces, n


# ---------------------------------------------------------------------------
# VTK writers (XML, inline base64 binary with UInt64 headers)
# ---------------------------------------------------------------------------
def _binary(values, vtk_type: str) -> str:
    raw = np.ascontiguousarray(np.asarray(values).ravel(), dtype=VTK_TYPES[vtk_type]).tobytes()
    header = np.array([len(raw)], dtype="<u8").tobytes()
    return base64.b64encode(header).decode("ascii") + base64.b64encode(raw).decode("ascii")


def _data_array(name: str | None, vtk_type: str, values, ncomp: int | None = None) -> str:
    values = np.asarray(values)
    ncomp = ncomp or (1 if values.ndim == 1 else values.shape[1])
    label = f' Name="{name}"' if name else ""
    comp = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    return (f'<DataArray type="{vtk_type}"{label}{comp} format="binary">\n'
            f"{_binary(values, vtk_type)}\n</DataArray>")


def write_vtu(path: Path, points, cells, point_data: dict, cell_data: dict) -> None:
    n_cells = cells.shape[0]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<UnstructuredGrid>",
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">', "<PointData>"]
    out += [_data_array(name, t, v) for name, (t, v) in point_data.items()]
    out += ["</PointData>", "<CellData>"]
    out += [_data_array(name, t, v) for name, (t, v) in cell_data.items()]
    out += ["</CellData>", "<Points>", _data_array(None, "Float64", points, 3), "</Points>",
            "<Cells>",
            _data_array("connectivity", "Int64", cells.ravel()),
            _data_array("offsets", "Int64", 4 * np.arange(1, n_cells + 1)),
            _data_array("types", "UInt8", np.full(n_cells, VTK_TETRA)),
            "</Cells>", "</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points, triangles, parents) -> None:
    triangles = np.asarray(triangles, dtype=np.int64)
    node_ids = np.unique(triangles)
    local = np.searchsorted(node_ids, triangles)
    nf = triangles.shape[0]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<PolyData>",
           f'<Piece NumberOfPoints="{node_ids.size}" NumberOfVerts="0" NumberOfLines="0" '
           f'NumberOfStrips="0" NumberOfPolys="{nf}">',
           "<PointData>", _data_array("GlobalNodeID", "Int64", node_ids), "</PointData>",
           "<CellData>", _data_array("GlobalElementID", "Int64", parents), "</CellData>",
           "<Points>", _data_array(None, "Float64", points[node_ids], 3), "</Points>",
           "<Polys>",
           _data_array("connectivity", "Int64", local.ravel()),
           _data_array("offsets", "Int64", 3 * np.arange(1, nf + 1)),
           "</Polys>", "</Piece>", "</PolyData>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def fsils_gmres_block() -> str:
    return """    <LS type="GMRES">
      <Linear_algebra type="fsils">
        <Preconditioner>rcs</Preconditioner>
      </Linear_algebra>
      <Max_iterations>100</Max_iterations>
      <Krylov_space_dimension>50</Krylov_space_dimension>
      <Tolerance>1.0e-8</Tolerance>
      <Absolute_tolerance>1.0e-10</Absolute_tolerance>
    </LS>"""


def level_set_velocity_block(transport: str) -> str:
    """Level-set advection velocity keys (decision D9)."""
    if transport == "coupled":
        return """    <Velocity_source>coupled_field</Velocity_source>
    <Velocity_field_name>Velocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>"""
    if transport == "wet_extension":
        return """    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Advection_velocity_extension_method>wall_compatible_normal</Advection_velocity_extension_method>"""
    if transport == "pde_extension":
        if not PDE_EXTENSION_METHOD:
            raise ValueError("--transport pde_extension: the PDE velocity extension (decision D9) "
                             "is not in the solver yet; set PDE_EXTENSION_METHOD in "
                             "generate_case.py to its input value once it is, or use "
                             "--transport wet_extension or coupled")
        coupling = (f"\n    <Advection_velocity_extension_coupling>{PDE_EXTENSION_COUPLING}"
                    "</Advection_velocity_extension_coupling>" if PDE_EXTENSION_COUPLING else "")
        return f"""    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Advection_velocity_extension_method>{PDE_EXTENSION_METHOD}</Advection_velocity_extension_method>{coupling}"""
    raise ValueError(f"--transport must be one of {TRANSPORTS}")


def solver_xml(form: str, schedule: dict, steps: int, cadence: int,
               transport: str = DEFAULT_TRANSPORT) -> str:
    velocity = level_set_velocity_block(transport)
    kag = form in ("kag_consistent", "kag_lumped")
    curvature_projection = ""
    if kag:
        mass = ("\n    <Curvature_projection_kinematic_area_gradient_mass>Lumped"
                "</Curvature_projection_kinematic_area_gradient_mass>"
                if form == "kag_lumped" else "")
        curvature_projection = f"""
    <Enable_curvature_projection>true</Enable_curvature_projection>
    <Curvature_field_name>{CURVATURE_FIELD}</Curvature_field_name>
    <Curvature_projection_recovery_mode>KinematicAreaGradient</Curvature_projection_recovery_mode>
    <Curvature_projection_kinematic_area_gradient_filter_coefficient>0.0</Curvature_projection_kinematic_area_gradient_filter_coefficient>{mass}
    <Curvature_projection_cadence_steps>1</Curvature_projection_cadence_steps>"""
    tension_form = "KinematicAreaGradientTraction" if kag else "SurfaceStress"
    curvature_bc = (f"\n      <Curvature_field>{CURVATURE_FIELD}</Curvature_field>" if kag else "")
    walls = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>""" for w in WALLS)
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- static_sphere_3d benchmark, capillary form {form}, transport {transport}; generated by generate_case.py -->
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>3</Number_of_spatial_dimensions>
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

  <Add_mesh name="box">
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
{velocity}
    <Enable_SUPG>true</Enable_SUPG>
    <SUPG_tau_scale>0.5</SUPG_tau_scale>
    <SUPG_transient_scale>2.0</SUPG_transient_scale>
    <Enable_reinitialization>false</Enable_reinitialization>
    <Enable_volume_correction>false</Enable_volume_correction>{curvature_projection}
    <Output type="Spatial">
      <Level_set>true</Level_set>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{fsils_gmres_block()}
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
    <Force_y>0.0</Force_y>
    <Force_z>0.0</Force_z>
    <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
    <Viscosity model="Constant">
      <Value>{schedule['viscosity']:.17g}</Value>
    </Viscosity>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{fsils_gmres_block()}
{walls}
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
      <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
      <Level_set_isovalue>0.0</Level_set_isovalue>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
      <Interface_quadrature_order>2</Interface_quadrature_order>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>{SURFACE_TENSION:.17g}</Surface_tension>
      <Surface_tension_form>{tension_form}</Surface_tension_form>{curvature_bc}
      <Use_level_set_curvature>false</Use_level_set_curvature>
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
def sphere_centre() -> np.ndarray:
    return np.full(3, 0.5 * BOX_SIDE) + np.asarray(CENTRE_OFFSET)


def generate(level: int, form: str, laplace: float, output_dir: Path, *,
             transport: str = DEFAULT_TRANSPORT,
             viscous_times: float = DEFAULT_VISCOUS_TIMES,
             snapshots: int = DEFAULT_SNAPSHOTS,
             dt_multiple_override: float | None = None,
             max_steps: int | None = None, force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if form not in CAPILLARY_FORMS:
        raise ValueError(f"--capillary-form must be one of {CAPILLARY_FORMS}")
    level_set_velocity_block(transport)                 # validates the transport choice
    if not (math.isfinite(laplace) and laplace > 0.0):
        raise ValueError("--laplace-number must be positive and finite")
    if not (viscous_times > 0.0 and snapshots >= 4):
        raise ValueError("need viscous_times > 0 and at least 4 snapshots")
    if dt_multiple_override is not None and not (dt_multiple_override > 0.0):
        raise ValueError("--dt-multiple must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule(level, laplace, viscous_times, snapshots, dt_multiple_override)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, n_cells_side = structured_tetra_mesh(level)
    centre = sphere_centre()
    phi = np.linalg.norm(points - centre, axis=1) - RADIUS
    h = schedule["h"]
    min_phi_over_h = float(np.min(np.abs(phi)) / h)
    wall_gap_over_h = float((0.5 * BOX_SIDE - RADIUS - max(abs(o) for o in CENTRE_OFFSET)) / h)
    laplace_pressure = 2.0 * SURFACE_TENSION / RADIUS

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(n_points)),
               LEVEL_SET_FIELD: ("Float64", phi),
               "Velocity": ("Float64", np.zeros((n_points, 3))),
               # Sampled analytic state: the constant Laplace pressure on the
               # whole background support (inactive DOFs are pinned by the
               # solver), as in static_drop_2d.  A phi-sign mask would add an
               # O(dp/h) gradient in every retained cut cell.
               "Pressure": ("Float64", np.full(n_points, EXTERNAL_PRESSURE + laplace_pressure))},
              {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in WALLS:
        triangles, parents = faces[wall]
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, triangles, parents)
    (output_dir / "solver.xml").write_text(solver_xml(form, schedule, steps, cadence, transport),
                                           encoding="utf-8")

    case = {
        "benchmark": "static_sphere_3d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/static_sphere_3d/generate_case.py",
        "level_R_over_h": level,
        "capillary_form": form,
        "transport": transport,
        "laplace_number": laplace,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": schedule["viscosity"],
        "radius": RADIUS,
        "centre": centre.tolist(),
        "centre_offset": list(CENTRE_OFFSET),
        "box": [0.0, BOX_SIDE, 0.0, BOX_SIDE, 0.0, BOX_SIDE],
        "cells_per_side": n_cells_side,
        "h": h,
        "n_vertices": int(n_points),
        "n_tetrahedra": int(cells.shape[0]),
        "external_pressure": EXTERNAL_PRESSURE,
        "laplace_pressure_nominal": laplace_pressure,
        "level_set_field": LEVEL_SET_FIELD,
        "velocity_field": "Velocity",
        "pressure_field": "Pressure",
        "liquid_side": "phi<0",
        "viscous_time": schedule["viscous_time"],
        "capillary_time": schedule["capillary_time"],
        "viscous_times": viscous_times,
        "end_time_protocol": schedule["end_time"],
        "dt_capillary_limit": schedule["dt_capillary_limit"],
        "dt_safety_factor": DT_SAFETY,
        "dt_B": schedule["dt_B"],
        "dt_multiple_of_dt_B": schedule["dt_multiple_of_dt_B"],
        "dt_multiple_is_protocol": dt_multiple_override is None,
        "dt": schedule["dt"],
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "min_abs_phi_over_h": min_phi_over_h,
        "wall_gap_over_h": wall_gap_over_h,
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="R/h")
    parser.add_argument("--capillary-form", required=True, choices=CAPILLARY_FORMS)
    parser.add_argument("--laplace-number", type=float, required=True,
                        help="La = rho*gamma*D/mu^2, D = 2R")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transport", default=DEFAULT_TRANSPORT, choices=TRANSPORTS,
                        help="level-set advection velocity (decision D9); default "
                             f"{DEFAULT_TRANSPORT}, pde_extension once the solver has it")
    parser.add_argument("--viscous-times", type=float, default=DEFAULT_VISCOUS_TIMES,
                        help="run length in units rho*R^2/mu (protocol value 5)")
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS,
                        help="number of VTU outputs over the run (protocol value 100)")
    parser.add_argument("--dt-multiple", type=float, default=None,
                        help="diagnostic time-step checks only: dt as a multiple of dt_B "
                             "(protocol value 2 at La = 12, 1 otherwise); recorded in case.json")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; the case is "
                             "marked truncated and verify.py rejects it for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)

    case = generate(args.level, args.capillary_form, args.laplace_number, args.output_dir,
                    transport=args.transport, viscous_times=args.viscous_times,
                    snapshots=args.snapshots, dt_multiple_override=args.dt_multiple,
                    max_steps=args.max_steps, force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("level_R_over_h", "capillary_form", "transport", "laplace_number", "viscosity",
                "viscous_time", "end_time", "dt", "dt_B", "dt_multiple_of_dt_B", "steps",
                "output_cadence", "n_vertices", "n_tetrahedra", "min_abs_phi_over_h",
                "wall_gap_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    if case["min_abs_phi_over_h"] < MIN_PHI_OVER_H_WARNING:
        print(f"WARNING: min |phi|/h = {case['min_abs_phi_over_h']:.3e} < "
              f"{MIN_PHI_OVER_H_WARNING:g}: the sampled sphere nearly touches a vertex",
              file=sys.stderr)
    if case["wall_gap_over_h"] < 2.0:
        print("WARNING: drop is less than two cells from the box wall", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
