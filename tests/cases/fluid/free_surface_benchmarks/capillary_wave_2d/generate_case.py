#!/usr/bin/env python3
"""Write one resolution level of the 2D capillary-wave benchmark (tracker M3).

A deep one-phase liquid layer with zero gravity and exterior pressure
p_ext = 0 is released from rest with the surface elevation
y0 + a0*cos(kx).  Half a wavelength is simulated between two free-slip
walls, which are the mirror planes of the standing cosine mode.  The
amplitude history is compared with Prosperetti's initial-value solution
(prosperetti_reference.py) by verify.py.

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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prosperetti_reference as reference  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol constants.  They are fixed for every level and every capillary form
# (principle P1); README.md gives the source or the derivation of each one.
# ---------------------------------------------------------------------------
LEVELS = (16, 32, 64)                       # wavelength / h
CAPILLARY_FORMS = ("surface_stress", "kag_consistent", "kag_lumped")
DT_DIVISORS = (1, 2, 4)                     # time-step refinement at a fixed level
DENSITY = 1.0                               # rho
SURFACE_TENSION = 1.0                       # gamma
WAVELENGTH = 1.0                            # lambda (length unit)
WAVENUMBER = 2.0 * math.pi / WAVELENGTH
AMPLITUDE_OVER_WAVELENGTH = 0.01            # a0 / lambda (Popinet 2009)
DEFAULT_LAPLACE_NUMBER = 3000.0             # La = rho gamma lambda / mu^2 (Popinet 2009)
BOX_WIDTH = 0.5 * WAVELENGTH                # walls at the crest (x = 0) and trough (x = lambda/2)
BOX_HEIGHT = 1.25 * WAVELENGTH
# Mean liquid depth: one wavelength (k H = 2 pi, deep liquid: tanh(kH) = 1 - 7e-6)
# plus an irrational offset, so the mean level lies on no grid line of the
# nested dyadic meshes and the sampled surface cannot pass exactly through a
# vertex.  sqrt(2)/100 keeps min |phi|/h >= 0.03 on all three levels.
MEAN_LEVEL = WAVELENGTH * (1.0 + math.sqrt(2.0) / 100.0)
EXTERNAL_PRESSURE = 0.0
DEFAULT_PERIODS = 4.0                       # run length in inviscid periods 2 pi / omega0
DEFAULT_SNAPSHOTS = 100                     # VTU outputs per run
# Capillary time-step limit (Brackbill, Kothe & Zemach 1992) evaluated with
# the one-sided density sum of a free surface (rho_liquid + rho_void = rho):
#   dt <= sqrt(rho h^3 / (4 pi gamma)) = SAFETY * sqrt(rho h^3 / (2 pi gamma)).
DT_SAFETY = 1.0 / math.sqrt(2.0)
MIN_PHI_OVER_H_WARNING = 1.0e-6             # "vertex touch" warning threshold
LEVEL_SET_FIELD = "phi"
CURVATURE_FIELD = "kappa"
INTERFACE_DOMAIN_ID = "capillary_wave_surface"
WALLS = ("wall_left", "wall_right", "wall_bottom", "wall_top")
# Strong zero-velocity components: the side walls and the bottom are
# impermeable free-slip walls (normal component only); the top boundary is
# in the dry region and never touches the liquid.
WALL_EFFECTIVE_DIRECTION = {"wall_left": "1 0", "wall_right": "1 0",
                            "wall_bottom": "0 1", "wall_top": None}


def viscosity_from_laplace(laplace: float) -> float:
    """mu from La = rho*gamma*lambda/mu^2."""
    return math.sqrt(DENSITY * SURFACE_TENSION * WAVELENGTH / laplace)


def capillary_dt_limit(h: float) -> float:
    """Tracker form sqrt(rho h^3 / (2 pi gamma)), before the safety factor."""
    return math.sqrt(DENSITY * h ** 3 / (2.0 * math.pi * SURFACE_TENSION))


def physical_parameters(laplace: float) -> dict:
    mu = viscosity_from_laplace(laplace)
    nu = mu / DENSITY
    omega0 = reference.inviscid_frequency(WAVENUMBER, SURFACE_TENSION, DENSITY)
    mode = reference.normal_mode(wavenumber=WAVENUMBER, kinematic_viscosity=nu,
                                 surface_tension=SURFACE_TENSION, density=DENSITY)
    return {
        "viscosity": mu,
        "kinematic_viscosity": nu,
        "ohnesorge_number": mu / math.sqrt(DENSITY * SURFACE_TENSION * WAVELENGTH),
        "omega0": omega0,
        "inviscid_period": 2.0 * math.pi / omega0,
        "finite_depth_factor": math.tanh(WAVENUMBER * MEAN_LEVEL),
        "omega0_finite_depth": reference.inviscid_frequency(
            WAVENUMBER, SURFACE_TENSION, DENSITY, depth=MEAN_LEVEL),
        "epsilon": nu * WAVENUMBER ** 2 / omega0,
        "weak_damping_rate": reference.weak_damping_rate(WAVENUMBER, nu),
        "normal_mode_omega": mode["omega"],
        "normal_mode_damping_rate": mode["beta"],
        "boundary_layer_thickness": math.sqrt(2.0 * nu / omega0),
    }


def time_schedule(level: int, laplace: float, periods: float, snapshots: int,
                  dt_divisor: int = 1) -> dict:
    h = WAVELENGTH / level
    phys = physical_parameters(laplace)
    end_time = periods * phys["inviscid_period"]
    dt_max = DT_SAFETY * capillary_dt_limit(h)
    # Protocol step: the largest dt <= dt_max giving `snapshots` equal output
    # intervals.  A divisor d refines it exactly to dt/d (d times the cadence).
    cadence = max(1, math.ceil(end_time / (snapshots * dt_max))) * dt_divisor
    steps = snapshots * cadence
    return {
        "h": h,
        "end_time": end_time,
        "dt_capillary_limit": capillary_dt_limit(h),
        "dt_max": dt_max / dt_divisor,
        "dt": end_time / steps,
        "steps": steps,
        "output_cadence": cadence,
        **phys,
    }


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def structured_triangle_mesh(level: int):
    """Rectangle [0, BOX_WIDTH] x [0, BOX_HEIGHT] split into right triangles.

    The diagonal alternates with (i + j) parity (as in static_drop_2d).  With
    an even number of columns the pattern is mirror-symmetric about the
    vertical centre line x = lambda/4, the node line of the cosine mode.
    """
    h = WAVELENGTH / level
    nx, ny = int(round(BOX_WIDTH / h)), int(round(BOX_HEIGHT / h))
    if not (math.isclose(nx * h, BOX_WIDTH, abs_tol=1e-12)
            and math.isclose(ny * h, BOX_HEIGHT, abs_tol=1e-12)):
        raise ValueError("box sides must be integer numbers of cells")
    xs, ys = np.linspace(0.0, BOX_WIDTH, nx + 1), np.linspace(0.0, BOX_HEIGHT, ny + 1)
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

    # Parent triangle of each wall segment (see static_drop_2d).
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


def initial_level_set(points: np.ndarray) -> np.ndarray:
    """phi = y - y0 - a0 cos(kx); liquid where phi < 0."""
    a0 = AMPLITUDE_OVER_WAVELENGTH * WAVELENGTH
    return points[:, 1] - MEAN_LEVEL - a0 * np.cos(WAVENUMBER * points[:, 0])


def initial_pressure(points: np.ndarray) -> np.ndarray:
    """Linear pressure of the released state (u = 0, t = 0+).

    Harmonic, zero normal derivative at the bottom y = 0, and equal to the
    capillary pressure gamma*kappa = gamma a0 k^2 cos(kx) at the mean level.
    It is continued smoothly through the dry vertices, which cut cells need.
    """
    a0 = AMPLITUDE_OVER_WAVELENGTH * WAVELENGTH
    k = WAVENUMBER
    return (EXTERNAL_PRESSURE + SURFACE_TENSION * a0 * k ** 2 * np.cos(k * points[:, 0])
            * np.cosh(k * points[:, 1]) / math.cosh(k * MEAN_LEVEL))


def _ascii(values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    return "\n".join(" ".join(fmt.format(v) for v in flat[s:s + 6])
                     for s in range(0, flat.size, 6))


def _data_array(vtk_type: str, name: str, values, fmt: str) -> list[str]:
    values = np.asarray(values)
    ncomp = 1 if values.ndim == 1 else values.shape[1]
    return [f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">',
            _ascii(values, fmt), '</DataArray>']


def write_vtu(path: Path, points, cells, point_data: dict, cell_data: dict) -> None:
    """ASCII VTU with Triangle3 cells (same layout as static_drop_2d)."""
    n_cells = cells.shape[0]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<UnstructuredGrid>',
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">', '<PointData>']
    for name, (vtk_type, values) in point_data.items():
        out += _data_array(vtk_type, name, values,
                           "{:d}" if vtk_type.startswith("Int") else "{:.17g}")
    out += ['</PointData>', '<CellData>']
    for name, (vtk_type, values) in cell_data.items():
        out += _data_array(vtk_type, name, values, "{:d}")
    out += ['</CellData>', '<Points>']
    out += _data_array("Float64", "Points", points, "{:.17g}")
    out += ['</Points>', '<Cells>']
    out += _data_array("Int64", "connectivity", cells.ravel(), "{:d}")
    out += _data_array("Int64", "offsets", 3 * np.arange(1, n_cells + 1), "{:d}")
    out += _data_array("UInt8", "types", np.full(n_cells, 5), "{:d}")    # VTK_TRIANGLE
    out += ['</Cells>', '</Piece>', '</UnstructuredGrid>', '</VTKFile>']
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points, node_ids, parent_cells) -> None:
    node_ids = np.asarray(node_ids, dtype=np.int64)
    n_lines = node_ids.size - 1
    local = np.column_stack([np.arange(n_lines), np.arange(1, n_lines + 1)])
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<PolyData>',
           f'<Piece NumberOfPoints="{node_ids.size}" NumberOfVerts="0" NumberOfLines="{n_lines}" '
           'NumberOfStrips="0" NumberOfPolys="0">', '<PointData>']
    out += _data_array("Int64", "GlobalNodeID", node_ids, "{:d}")
    out += ['</PointData>', '<CellData>']
    out += _data_array("Int64", "GlobalElementID", parent_cells, "{:d}")
    out += ['</CellData>', '<Points>']
    out += _data_array("Float64", "Points", points[node_ids], "{:.17g}")
    out += ['</Points>', '<Lines>']
    out += _data_array("Int64", "connectivity", local.ravel(), "{:d}")
    out += _data_array("Int64", "offsets", 2 * np.arange(1, n_lines + 1), "{:d}")
    out += ['</Lines>', '</Piece>', '</PolyData>', '</VTKFile>']
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


def wall_bc(name: str) -> str:
    direction = WALL_EFFECTIVE_DIRECTION[name]
    effective = (f"\n      <Effective_direction>{direction}</Effective_direction>"
                 if direction else "")
    return f"""    <Add_BC name="{name}">
      <Type>Dir</Type>
      <Value>0.0</Value>{effective}
    </Add_BC>"""


def solver_xml(form: str, schedule: dict, steps: int, cadence: int) -> str:
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
    walls = "\n".join(wall_bc(w) for w in WALLS)
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- capillary_wave_2d benchmark, capillary form {form}; generated by generate_case.py -->
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
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
def generate(level: int, form: str, output_dir: Path, *,
             laplace: float = DEFAULT_LAPLACE_NUMBER,
             periods: float = DEFAULT_PERIODS,
             snapshots: int = DEFAULT_SNAPSHOTS,
             dt_divisor: int = 1,
             max_steps: int | None = None, force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if form not in CAPILLARY_FORMS:
        raise ValueError(f"--capillary-form must be one of {CAPILLARY_FORMS}")
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    if not (math.isfinite(laplace) and laplace > 0.0):
        raise ValueError("--laplace-number must be positive and finite")
    if not (periods > 0.0 and snapshots >= 4):
        raise ValueError("need periods > 0 and at least 4 snapshots")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule(level, laplace, periods, snapshots, dt_divisor)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, (nx, ny) = structured_triangle_mesh(level)
    phi = initial_level_set(points)
    h = schedule["h"]
    a0 = AMPLITUDE_OVER_WAVELENGTH * WAVELENGTH
    min_phi_over_h = float(np.min(np.abs(phi)) / h)
    top_gap_over_h = float((BOX_HEIGHT - MEAN_LEVEL - a0) / h)

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(n_points)),
               LEVEL_SET_FIELD: ("Float64", phi),
               "Velocity": ("Float64", np.zeros((n_points, 3))),
               "Pressure": ("Float64", initial_pressure(points))},
              {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in WALLS:
        node_ids, parents = faces[wall]
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(solver_xml(form, schedule, steps, cadence),
                                           encoding="utf-8")

    case = {
        "benchmark": "capillary_wave_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/capillary_wave_2d/generate_case.py",
        "level_lambda_over_h": level,
        "capillary_form": form,
        "laplace_number": laplace,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": schedule["viscosity"],
        "kinematic_viscosity": schedule["kinematic_viscosity"],
        "ohnesorge_number": schedule["ohnesorge_number"],
        "wavelength": WAVELENGTH,
        "wavenumber": WAVENUMBER,
        "initial_amplitude": a0,
        "amplitude_over_wavelength": AMPLITUDE_OVER_WAVELENGTH,
        "mean_level": MEAN_LEVEL,
        "box": [0.0, BOX_WIDTH, 0.0, BOX_HEIGHT],
        "cells": [nx, ny],
        "h": h,
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "external_pressure": EXTERNAL_PRESSURE,
        "level_set_field": LEVEL_SET_FIELD,
        "liquid_side": "phi<0",
        "wall_conditions": {w: ("u.n = 0 (free slip)" if WALL_EFFECTIVE_DIRECTION[w] else
                                "u = 0 (dry)") for w in WALLS},
        "omega0": schedule["omega0"],
        "omega0_finite_depth": schedule["omega0_finite_depth"],
        "finite_depth_factor": schedule["finite_depth_factor"],
        "inviscid_period": schedule["inviscid_period"],
        "epsilon": schedule["epsilon"],
        "weak_damping_rate": schedule["weak_damping_rate"],
        "normal_mode_omega": schedule["normal_mode_omega"],
        "normal_mode_damping_rate": schedule["normal_mode_damping_rate"],
        "boundary_layer_thickness": schedule["boundary_layer_thickness"],
        "boundary_layer_thickness_over_h": schedule["boundary_layer_thickness"] / h,
        "periods": periods,
        "end_time_protocol": schedule["end_time"],
        "dt_capillary_limit": schedule["dt_capillary_limit"],
        "dt_safety_factor": DT_SAFETY,
        "dt_divisor": dt_divisor,
        "dt": schedule["dt"],
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "min_abs_phi_over_h": min_phi_over_h,
        "top_gap_over_h": top_gap_over_h,
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="lambda/h")
    parser.add_argument("--capillary-form", default="surface_stress", choices=CAPILLARY_FORMS)
    parser.add_argument("--laplace-number", type=float, default=DEFAULT_LAPLACE_NUMBER,
                        help="La = rho*gamma*lambda/mu^2 (protocol value 3000)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=float, default=DEFAULT_PERIODS,
                        help="run length in inviscid periods 2*pi/omega0 (protocol value 4)")
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS,
                        help="number of VTU outputs over the run (protocol value 100)")
    parser.add_argument("--dt-divisor", type=int, default=1, choices=DT_DIVISORS,
                        help="divide the protocol time step by this factor "
                             "(temporal refinement study; protocol value 1)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; the case is "
                             "marked truncated and verify.py rejects it for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)

    case = generate(args.level, args.capillary_form, args.output_dir,
                    laplace=args.laplace_number, periods=args.periods,
                    snapshots=args.snapshots, dt_divisor=args.dt_divisor,
                    max_steps=args.max_steps, force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("level_lambda_over_h", "capillary_form", "laplace_number", "viscosity",
                "epsilon", "omega0", "normal_mode_omega", "normal_mode_damping_rate",
                "end_time", "dt", "dt_capillary_limit", "dt_divisor", "steps",
                "output_cadence", "n_vertices", "n_triangles", "min_abs_phi_over_h",
                "top_gap_over_h", "boundary_layer_thickness_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    if case["min_abs_phi_over_h"] < MIN_PHI_OVER_H_WARNING:
        print(f"WARNING: min |phi|/h = {case['min_abs_phi_over_h']:.3e} < "
              f"{MIN_PHI_OVER_H_WARNING:g}: the sampled surface nearly touches a vertex",
              file=sys.stderr)
    if case["top_gap_over_h"] < 2.0:
        print("WARNING: the crest is less than two cells from the top boundary", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
