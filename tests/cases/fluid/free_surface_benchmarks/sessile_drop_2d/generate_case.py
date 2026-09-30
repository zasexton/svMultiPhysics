#!/usr/bin/env python3
"""Write one resolution level of the 2D sessile-drop benchmark (tracker M4).

A 2D liquid cap sits on the flat bottom wall y = 0 with zero gravity and
exterior pressure p_ext = 0.  It starts as a circular cap with a
non-equilibrium contact angle theta_0 and relaxes, at fixed area, to the
circular cap with the Young angle theta_e.  The contact angle is imposed only
by the variational Young term in the momentum equation (decision D4):
PrescribedAngle contact line, Navier slip on the wetted wall, and a strong
normal-only zero velocity on the wall.  Level-set wall maintenance, when
enabled, only rescales contact cells.

The case is written for the new OOP solver: solver.xml, an affine Triangle3
background mesh with the initial fields, the four wall face files, and
case.json with every parameter that verify.py needs.  See README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants.  They are fixed for every level, angle and capillary
# form (principle P1); README.md gives the source or derivation of each one.
# ---------------------------------------------------------------------------
LEVELS = (16, 32, 64)                       # R/h, R = equilibrium cap radius
EQUILIBRIUM_ANGLES_DEG = (60, 90, 120)      # theta_e, measured through the liquid
# Initial cap angle for each target: two advancing cases (60, 90) and one
# receding case (120), each 30 degrees from equilibrium.
INITIAL_ANGLE_DEG = {60: 90.0, 90: 120.0, 120: 90.0}
CAPILLARY_FORMS = ("surface_stress", "kag_consistent", "kag_lumped")
DENSITY = 1.0                               # rho
SURFACE_TENSION = 1.0                       # gamma
RADIUS = 1.0                                # R, radius of the equilibrium cap (length unit)
LAPLACE_NUMBER = 12.0                       # La = rho*gamma*(2R)/mu^2, as the static drop
SLIP_LENGTH = RADIUS / 8.0                  # physical input: l_s/h = 2, 4, 8 at R/h = 16, 32, 64
# Transcendental horizontal offset of the cap centre: no contact point or
# circle can coincide with a vertex of the nested dyadic grids.
CENTRE_OFFSET_X = math.pi / 100.0 * RADIUS
BOX_MARGIN = 0.25 * RADIUS                  # dry band around the initial and final caps
GRID_UNIT = RADIUS / min(LEVELS)            # box sides are multiples of the coarsest h
EXTERNAL_PRESSURE = 0.0
DEFAULT_VISCOUS_TIMES = 5.0                 # run length in units rho*R^2/mu
DEFAULT_SNAPSHOTS = 100                     # VTU outputs per run
# Capillary time-step limit (Brackbill, Kothe & Zemach 1992) with the
# one-sided density sum of a free surface (rho_liquid + rho_void = rho):
#   dt <= sqrt(rho h^3 / (4 pi gamma)) = SAFETY * sqrt(rho h^3 / (2 pi gamma)).
DT_SAFETY = 1.0 / math.sqrt(2.0)
# Time integration and linear solver: generalized-alpha (rho_inf = 0.5) and
# FSILS GMRES, as in static_drop_2d.  The moving contact line and interface
# cross mesh vertices all the time; the solver accepts a cut-topology change
# within a step as a normal event (README.md, "Vertex crossings").  The
# alternatives are kept for comparison runs only.
TIME_INTEGRATION_SCHEMES = {"generalized_alpha": "GeneralizedAlpha",
                            "backward_euler": "BackwardEuler"}
LINEAR_SOLVERS = ("fsils", "eigen_direct")
# Level-set transport velocity.  "coupled": the fluid velocity itself (the
# current default, as static_drop_2d).  "wet_extension": the wall-compatible
# wet extension of the D18 and capillary-rise decks.  "pde_extension": the
# harmonic PDE velocity extension with monolithic coupling, chosen for
# moving-interface benchmarks (tracker D9; linear_sloshing_2d README,
# "Level-set advection velocity").
TRANSPORTS = ("coupled", "wet_extension", "pde_extension")
# Existing production reinitialization values (D18/D38 and sloshing decks),
# used only with --reinitialization.
REINITIALIZATION_CADENCE_STEPS = 10
REINITIALIZATION_MAX_ITERATIONS = 4
MIN_PHI_OVER_H_WARNING = 1.0e-6             # "vertex touch" warning threshold
LEVEL_SET_FIELD = "phi"
CURVATURE_FIELD = "kappa"
INTERFACE_DOMAIN_ID = "sessile_drop_surface"
CONTACT_WALL = "wall_bottom"
CONTACT_WALL_NORMAL = (0.0, -1.0, 0.0)      # out of the liquid, into the solid
WALLS = ("wall_left", "wall_right", "wall_bottom", "wall_top")


# ---------------------------------------------------------------------------
# Circular-cap geometry.  A cap of radius r on y = 0 with contact angle theta
# (through the liquid) has its centre at y = -r cos(theta).
# ---------------------------------------------------------------------------
def cap_area_factor(theta: float) -> float:
    """Area of the cap divided by r^2: theta - sin(theta) cos(theta)."""
    return theta - math.sin(theta) * math.cos(theta)


def cap_radius_for_area(area: float, theta: float) -> float:
    return math.sqrt(area / cap_area_factor(theta))


def cap_geometry(radius: float, theta: float) -> dict:
    return {
        "radius": radius,
        "angle_radians": theta,
        "centre_y": -radius * math.cos(theta),
        "base_half_width": radius * math.sin(theta),
        "apex_height": radius * (1.0 - math.cos(theta)),
        "half_extent": radius if theta > 0.5 * math.pi else radius * math.sin(theta),
        "area": radius * radius * cap_area_factor(theta),
    }


def viscosity_from_laplace(laplace: float) -> float:
    """mu from La = rho*gamma*D/mu^2 with D = 2R."""
    return math.sqrt(DENSITY * SURFACE_TENSION * 2.0 * RADIUS / laplace)


def capillary_dt_limit(h: float) -> float:
    """Tracker form sqrt(rho h^3 / (2 pi gamma)), before the safety factor."""
    return math.sqrt(DENSITY * h ** 3 / (2.0 * math.pi * SURFACE_TENSION))


def time_schedule(level: int, viscous_times: float, snapshots: int) -> dict:
    h = RADIUS / level
    mu = viscosity_from_laplace(LAPLACE_NUMBER)
    viscous_time = DENSITY * RADIUS ** 2 / mu
    end_time = viscous_times * viscous_time
    dt_max = DT_SAFETY * capillary_dt_limit(h)
    cadence = max(1, math.ceil(end_time / (snapshots * dt_max)))
    steps = snapshots * cadence
    return {
        "h": h,
        "viscosity": mu,
        "viscous_time": viscous_time,
        "capillary_time": math.sqrt(DENSITY * RADIUS ** 3 / SURFACE_TENSION),
        "visco_capillary_time": mu * RADIUS / SURFACE_TENSION,
        "end_time": end_time,
        "dt_capillary_limit": capillary_dt_limit(h),
        "dt_max": dt_max,
        "dt": end_time / steps,
        "steps": steps,
        "output_cadence": cadence,
    }


def case_geometry(equilibrium_deg: float, initial_deg: float) -> dict:
    theta_e = math.radians(equilibrium_deg)
    theta_0 = math.radians(initial_deg)
    equilibrium = cap_geometry(RADIUS, theta_e)
    initial = cap_geometry(cap_radius_for_area(equilibrium["area"], theta_0), theta_0)

    def grid_ceil(value: float) -> float:
        return math.ceil(value / GRID_UNIT - 1.0e-9) * GRID_UNIT

    half_width = grid_ceil(max(initial["half_extent"], equilibrium["half_extent"])
                           + abs(CENTRE_OFFSET_X) + BOX_MARGIN)
    height = grid_ceil(max(initial["apex_height"], equilibrium["apex_height"]) + BOX_MARGIN)
    return {"equilibrium": equilibrium, "initial": initial,
            "box": [-half_width, half_width, 0.0, height]}


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def structured_triangle_mesh(box, level: int):
    """Rectangle split into right triangles, h = R/level.

    The diagonal alternates with (i + j) parity, so the pattern has no
    preferred diagonal direction.
    """
    x0, x1, y0, y1 = box
    h = RADIUS / level
    nx, ny = int(round((x1 - x0) / h)), int(round((y1 - y0) / h))
    if not (math.isclose(nx * h, x1 - x0, abs_tol=1e-12)
            and math.isclose(ny * h, y1 - y0, abs_tol=1e-12)):
        raise ValueError("box sides must be integer numbers of cells")
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    xx, yy = np.meshgrid(xs, ys)                    # row j = y index
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            if (i + j) % 2 == 0:
                cells += [(a, b, c), (a, c, d)]
            else:
                cells += [(a, b, d), (b, c, d)]
    cells = np.asarray(cells, dtype=np.int64)

    # Parent triangle of each wall segment: the first triangle of cell (i, j)
    # holds the bottom edge, the second the top edge; the left and right
    # edges switch with the diagonal parity.
    def parent(i: int, j: int, side: str) -> int:
        even = (i + j) % 2 == 0
        second = {"bottom": False, "top": True, "left": even, "right": not even}[side]
        return 2 * (j * nx + i) + int(second)

    faces = {
        "wall_bottom": ([vid(i, 0) for i in range(nx + 1)],
                        [parent(i, 0, "bottom") for i in range(nx)]),
        "wall_top": ([vid(i, ny) for i in range(nx + 1)],
                     [parent(i, ny - 1, "top") for i in range(nx)]),
        "wall_left": ([vid(0, j) for j in range(ny + 1)],
                      [parent(0, j, "left") for j in range(ny)]),
        "wall_right": ([vid(nx, j) for j in range(ny + 1)],
                       [parent(nx - 1, j, "right") for j in range(ny)]),
    }
    return points, cells, faces, (nx, ny)


def _ascii(values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    lines = []
    for start in range(0, flat.size, 6):
        lines.append(" ".join(fmt.format(v) for v in flat[start:start + 6]))
    return "\n".join(lines)


def write_vtu(path: Path, points, cells, point_data: dict, cell_data: dict) -> None:
    n_cells = cells.shape[0]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<UnstructuredGrid>',
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">',
           '<PointData>']
    for name, (vtk_type, values) in point_data.items():
        values = np.asarray(values)
        ncomp = 1 if values.ndim == 1 else values.shape[1]
        fmt = "{:d}" if vtk_type.startswith("Int") else "{:.17g}"
        out.append(f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">')
        out.append(_ascii(values, fmt))
        out.append('</DataArray>')
    out.append('</PointData>')
    out.append('<CellData>')
    for name, (vtk_type, values) in cell_data.items():
        out.append(f'<DataArray type="{vtk_type}" Name="{name}" format="ascii">')
        out.append(_ascii(values, "{:d}"))
        out.append('</DataArray>')
    out.append('</CellData>')
    out.append('<Points>')
    out.append('<DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    out.append(_ascii(points, "{:.17g}"))
    out.append('</DataArray>')
    out.append('</Points>')
    out.append('<Cells>')
    out.append('<DataArray type="Int64" Name="connectivity" format="ascii">')
    out.append(_ascii(cells, "{:d}"))
    out.append('</DataArray>')
    out.append('<DataArray type="Int64" Name="offsets" format="ascii">')
    out.append(_ascii(3 * np.arange(1, n_cells + 1), "{:d}"))
    out.append('</DataArray>')
    out.append('<DataArray type="UInt8" Name="types" format="ascii">')
    out.append(_ascii(np.full(n_cells, 5), "{:d}"))           # VTK_TRIANGLE
    out.append('</DataArray>')
    out.append('</Cells>')
    out.append('</Piece>')
    out.append('</UnstructuredGrid>')
    out.append('</VTKFile>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points, node_ids, parent_cells) -> None:
    node_ids = np.asarray(node_ids, dtype=np.int64)
    n_lines = node_ids.size - 1
    local = np.column_stack([np.arange(n_lines), np.arange(1, n_lines + 1)])
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<PolyData>',
           f'<Piece NumberOfPoints="{node_ids.size}" NumberOfVerts="0" NumberOfLines="{n_lines}" '
           'NumberOfStrips="0" NumberOfPolys="0">',
           '<PointData>',
           '<DataArray type="Int64" Name="GlobalNodeID" format="ascii">',
           _ascii(node_ids, "{:d}"),
           '</DataArray>',
           '</PointData>',
           '<CellData>',
           '<DataArray type="Int64" Name="GlobalElementID" format="ascii">',
           _ascii(parent_cells, "{:d}"),
           '</DataArray>',
           '</CellData>',
           '<Points>',
           '<DataArray type="Float64" NumberOfComponents="3" format="ascii">',
           _ascii(points[node_ids], "{:.17g}"),
           '</DataArray>',
           '</Points>',
           '<Lines>',
           '<DataArray type="Int64" Name="connectivity" format="ascii">',
           _ascii(local, "{:d}"),
           '</DataArray>',
           '<DataArray type="Int64" Name="offsets" format="ascii">',
           _ascii(2 * np.arange(1, n_lines + 1), "{:d}"),
           '</DataArray>',
           '</Lines>',
           '</Piece>',
           '</PolyData>',
           '</VTKFile>']
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def linear_solver_block(solver: str = "fsils") -> str:
    if solver == "eigen_direct":
        return """    <LS type="Direct">
      <Linear_algebra type="eigen">
        <Preconditioner>none</Preconditioner>
      </Linear_algebra>
      <Max_iterations>1</Max_iterations>
      <Tolerance>1.0e-8</Tolerance>
      <Absolute_tolerance>1.0e-10</Absolute_tolerance>
    </LS>"""
    if solver != "fsils":
        raise ValueError(f"linear solver must be one of {LINEAR_SOLVERS}")
    return """    <LS type="GMRES">
      <Linear_algebra type="fsils">
        <Preconditioner>rcs</Preconditioner>
      </Linear_algebra>
      <Max_iterations>100</Max_iterations>
      <Krylov_space_dimension>50</Krylov_space_dimension>
      <Tolerance>1.0e-8</Tolerance>
      <Absolute_tolerance>1.0e-10</Absolute_tolerance>
    </LS>"""


def solver_xml(form: str, equilibrium_deg: float, schedule: dict, steps: int, cadence: int,
               reinitialization: bool = False, linear_solver: str = "fsils",
               time_integration: str = "generalized_alpha",
               transport: str = "coupled") -> str:
    if transport == "coupled":
        transport_xml = """
    <Velocity_source>coupled_field</Velocity_source>
    <Velocity_field_name>Velocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>"""
    elif transport == "wet_extension":
        transport_xml = """
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Wet_extension_advection_velocity_method>wall_compatible_normal</Wet_extension_advection_velocity_method>"""
    elif transport == "pde_extension":
        transport_xml = """
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Advection_velocity_extension_method>pde_harmonic</Advection_velocity_extension_method>
    <Advection_velocity_extension_coupling>monolithic</Advection_velocity_extension_coupling>"""
    else:
        raise ValueError(f"transport must be one of {TRANSPORTS}")
    if time_integration not in TIME_INTEGRATION_SCHEMES:
        raise ValueError(f"time integration must be one of {tuple(TIME_INTEGRATION_SCHEMES)}")
    scheme = TIME_INTEGRATION_SCHEMES[time_integration]
    time_integration_xml = (
        "<Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>"
        if scheme == "GeneralizedAlpha" else
        f"<Transient_time_integration_scheme>{scheme}</Transient_time_integration_scheme>")
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
    if reinitialization:
        maintenance = f"""
    <Enable_reinitialization>true</Enable_reinitialization>
    <Reinitialization_method>projection</Reinitialization_method>
    <Reinitialization_cadence_steps>{REINITIALIZATION_CADENCE_STEPS}</Reinitialization_cadence_steps>
    <Reinitialization_max_iterations>{REINITIALIZATION_MAX_ITERATIONS}</Reinitialization_max_iterations>"""
    else:
        maintenance = "\n    <Enable_reinitialization>false</Enable_reinitialization>"
    # The contact wall carries a strong zero normal velocity only; its
    # tangential motion is governed by the Navier slip term of the free-surface
    # condition.  The other walls stay dry and are no-slip.
    walls = []
    for w in WALLS:
        direction = ("\n      <Effective_direction>0 1</Effective_direction>"
                     if w == CONTACT_WALL else "")
        walls.append(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>{direction}
    </Add_BC>""")
    walls = "\n".join(walls)
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS)
    wall_normal = " ".join(f"{c:g}" for c in CONTACT_WALL_NORMAL)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- sessile_drop_2d benchmark, theta_e = {equilibrium_deg:g} deg, capillary form {form}; generated by generate_case.py -->
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
    <Number_of_time_steps>{steps}</Number_of_time_steps>
    <Time_step_size>{schedule['dt']:.17g}</Time_step_size>
    {time_integration_xml}
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
    <Level_set_source>prescribed_data</Level_set_source>{transport_xml}
    <Enable_SUPG>true</Enable_SUPG>
    <SUPG_tau_scale>0.5</SUPG_tau_scale>
    <SUPG_transient_scale>2.0</SUPG_transient_scale>{maintenance}
    <Enable_volume_correction>false</Enable_volume_correction>{curvature_projection}
    <Output type="Spatial">
      <Level_set>true</Level_set>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{linear_solver_block(linear_solver)}
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
{linear_solver_block(linear_solver)}
{walls}
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
      <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
      <Level_set_isovalue>0.0</Level_set_isovalue>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Active_domain_smoothing_width>0.0</Active_domain_smoothing_width>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
      <Interface_quadrature_order>2</Interface_quadrature_order>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>{SURFACE_TENSION:.17g}</Surface_tension>
      <Surface_tension_form>{tension_form}</Surface_tension_form>{curvature_bc}
      <Use_level_set_curvature>false</Use_level_set_curvature>
      <Contact_line_model>PrescribedAngle</Contact_line_model>
      <Contact_line_wall_face>{CONTACT_WALL}</Contact_line_wall_face>
      <Contact_line_wall_normal>{wall_normal}</Contact_line_wall_normal>
      <Contact_angle_degrees>{equilibrium_deg:.17g}</Contact_angle_degrees>
      <Wall_slip_model>Navier</Wall_slip_model>
      <Wall_slip_length>{SLIP_LENGTH:.17g}</Wall_slip_length>
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
def generate(level: int, equilibrium_deg: float, form: str, output_dir: Path, *,
             initial_deg: float | None = None,
             viscous_times: float = DEFAULT_VISCOUS_TIMES,
             snapshots: int = DEFAULT_SNAPSHOTS,
             reinitialization: bool = False, linear_solver: str = "fsils",
             time_integration: str = "generalized_alpha",
             transport: str = "coupled",
             max_steps: int | None = None, force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if equilibrium_deg not in EQUILIBRIUM_ANGLES_DEG:
        raise ValueError(f"--contact-angle must be one of {EQUILIBRIUM_ANGLES_DEG}")
    if form not in CAPILLARY_FORMS:
        raise ValueError(f"--capillary-form must be one of {CAPILLARY_FORMS}")
    if initial_deg is None:
        initial_deg = INITIAL_ANGLE_DEG[int(equilibrium_deg)]
    if not (10.0 <= initial_deg <= 170.0):
        raise ValueError("--initial-angle must lie in [10, 170] degrees")
    if not (viscous_times > 0.0 and snapshots >= 4):
        raise ValueError("need viscous_times > 0 and at least 4 snapshots")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule(level, viscous_times, snapshots)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    geometry = case_geometry(equilibrium_deg, initial_deg)
    initial, equilibrium = geometry["initial"], geometry["equilibrium"]
    points, cells, faces, (nx, ny) = structured_triangle_mesh(geometry["box"], level)
    centre = np.array([CENTRE_OFFSET_X, initial["centre_y"]])
    phi = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) - initial["radius"]
    h = schedule["h"]
    min_phi_over_h = float(np.min(np.abs(phi)) / h)
    initial_contacts = CENTRE_OFFSET_X + np.array([-1.0, 1.0]) * initial["base_half_width"]
    wall_x = points[faces["wall_bottom"][0], 0]
    contact_vertex_gap_over_h = float(
        np.min(np.abs(initial_contacts[:, None] - wall_x[None, :])) / h)
    box = geometry["box"]
    dry_gap = min(box[1] - (CENTRE_OFFSET_X + max(initial["half_extent"], equilibrium["half_extent"])),
                  (CENTRE_OFFSET_X - max(initial["half_extent"], equilibrium["half_extent"])) - box[0],
                  box[3] - max(initial["apex_height"], equilibrium["apex_height"]))
    laplace_pressure = SURFACE_TENSION / initial["radius"]

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(n_points)),
               LEVEL_SET_FIELD: ("Float64", phi),
               "Velocity": ("Float64", np.zeros((n_points, 3))),
               # Laplace pressure of the initial cap on the whole background
               # support, as in static_drop_2d (inactive DOFs are pinned).
               "Pressure": ("Float64", np.full(n_points, EXTERNAL_PRESSURE + laplace_pressure))},
              {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in WALLS:
        node_ids, parents = faces[wall]
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(
        solver_xml(form, equilibrium_deg, schedule, steps, cadence, reinitialization,
                   linear_solver, time_integration, transport),
        encoding="utf-8")

    case = {
        "benchmark": "sessile_drop_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/sessile_drop_2d/generate_case.py",
        "level_R_over_h": level,
        "equilibrium_angle_degrees": float(equilibrium_deg),
        "initial_angle_degrees": float(initial_deg),
        "capillary_form": form,
        "contact_line_model": "PrescribedAngle",
        "reinitialization": bool(reinitialization),
        "time_integration_scheme": TIME_INTEGRATION_SCHEMES[time_integration],
        "linear_solver": linear_solver,
        "transport": transport,
        "laplace_number": LAPLACE_NUMBER,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": schedule["viscosity"],
        "slip_length": SLIP_LENGTH,
        "slip_length_over_h": SLIP_LENGTH / h,
        "radius": RADIUS,
        "equilibrium_cap_nominal": equilibrium,
        "initial_cap": initial,
        "initial_centre": centre.tolist(),
        "centre_offset_x": CENTRE_OFFSET_X,
        "box": box,
        "cells": [nx, ny],
        "h": h,
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "contact_wall_y": 0.0,
        "contact_wall_normal": list(CONTACT_WALL_NORMAL),
        "external_pressure": EXTERNAL_PRESSURE,
        "level_set_field": LEVEL_SET_FIELD,
        "velocity_field": "Velocity",
        "pressure_field": "Pressure",
        "liquid_side": "phi<0",
        "viscous_time": schedule["viscous_time"],
        "capillary_time": schedule["capillary_time"],
        "visco_capillary_time": schedule["visco_capillary_time"],
        "viscous_times": viscous_times,
        "end_time_protocol": schedule["end_time"],
        "dt_capillary_limit": schedule["dt_capillary_limit"],
        "dt_safety_factor": DT_SAFETY,
        "dt": schedule["dt"],
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "min_abs_phi_over_h": min_phi_over_h,
        "initial_contact_vertex_gap_over_h": contact_vertex_gap_over_h,
        "dry_gap_over_h": float(dry_gap / h),
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="R/h")
    parser.add_argument("--contact-angle", type=int, required=True,
                        choices=EQUILIBRIUM_ANGLES_DEG,
                        help="equilibrium (Young) angle theta_e in degrees")
    parser.add_argument("--capillary-form", default="surface_stress", choices=CAPILLARY_FORMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-angle", type=float, default=None,
                        help="initial cap angle in degrees (protocol: 90 for 60 and 120, "
                             "120 for 90)")
    parser.add_argument("--viscous-times", type=float, default=DEFAULT_VISCOUS_TIMES,
                        help="run length in units rho*R^2/mu (protocol value 5)")
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS,
                        help="number of VTU outputs over the run (protocol value 100)")
    parser.add_argument("--reinitialization", action="store_true",
                        help="enable projection reinitialization with the production values "
                             "(every 10 steps, at most 4 iterations; contact cells are only "
                             "rescaled).  Off in the protocol, as in static_drop_2d.")
    parser.add_argument("--linear-solver", default="fsils", choices=LINEAR_SOLVERS,
                        help="linear solver (protocol: fsils; eigen_direct for comparison)")
    parser.add_argument("--transport", default="coupled", choices=TRANSPORTS,
                        help="level-set transport velocity (pde_extension: the harmonic PDE "
                             "velocity extension of tracker D9, monolithic coupling)")
    parser.add_argument("--time-integration", default="generalized_alpha",
                        choices=tuple(TIME_INTEGRATION_SCHEMES),
                        help="time integration (protocol: generalized_alpha; backward_euler "
                             "for comparison)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; the case is "
                             "marked truncated and verify.py rejects it for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)

    case = generate(args.level, args.contact_angle, args.capillary_form, args.output_dir,
                    initial_deg=args.initial_angle, viscous_times=args.viscous_times,
                    snapshots=args.snapshots, reinitialization=args.reinitialization,
                    linear_solver=args.linear_solver,
                    time_integration=args.time_integration,
                    transport=args.transport,
                    max_steps=args.max_steps, force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("level_R_over_h", "equilibrium_angle_degrees", "initial_angle_degrees",
                "capillary_form", "time_integration_scheme", "linear_solver", "transport",
                "reinitialization", "viscosity", "slip_length_over_h",
                "viscous_time",
                "end_time", "dt", "dt_capillary_limit", "steps", "output_cadence",
                "box", "n_vertices", "n_triangles", "min_abs_phi_over_h",
                "initial_contact_vertex_gap_over_h", "dry_gap_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    if case["min_abs_phi_over_h"] < MIN_PHI_OVER_H_WARNING:
        print(f"WARNING: min |phi|/h = {case['min_abs_phi_over_h']:.3e} < "
              f"{MIN_PHI_OVER_H_WARNING:g}: the sampled cap nearly touches a vertex",
              file=sys.stderr)
    if case["initial_contact_vertex_gap_over_h"] < MIN_PHI_OVER_H_WARNING:
        print("WARNING: an initial contact point nearly coincides with a wall vertex",
              file=sys.stderr)
    if case["dry_gap_over_h"] < 2.0:
        print("WARNING: the drop comes within two cells of a dry box wall", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
