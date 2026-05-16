#!/usr/bin/env python3
"""Generate the 2D traveling-interface MMS free-surface test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent
MESH_SUBDIR = Path("mesh/background")

DEFAULT_NX = 16
DEFAULT_NY = 16
DEFAULT_ELEMENT_ORDER = 2
DEFAULT_L = 1.0
DEFAULT_H0 = 0.5
DEFAULT_H_TANK = 0.75
DEFAULT_DENSITY = 998.2
DEFAULT_VISCOSITY = 2.0e-2
DEFAULT_GRAVITY = 9.81
DEFAULT_AMPLITUDE = 0.02
DEFAULT_U0 = 0.10
DEFAULT_OMEGA = 2.0 * math.pi
DEFAULT_FINAL_TIME = 1.0
DEFAULT_TIME_STEPS = 50
DEFAULT_OUTPUT_CADENCE = 1


def node_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i


def qnode_id(i: int, j: int, nx: int) -> int:
    return j * (2 * nx + 1) + i


def mms_parameters(*, length: float, omega: float, u0: float) -> dict[str, float]:
    return {
        "k": 2.0 * math.pi / length,
        "period": 2.0 * math.pi / omega,
        "x_scale": u0 / omega,
    }


def shift(t: float, *, u0: float, omega: float) -> float:
    return (u0 / omega) * math.sin(omega * t)


def uniform_velocity(t: float, *, u0: float, omega: float) -> float:
    return u0 * math.cos(omega * t)


def uniform_acceleration(t: float, *, u0: float, omega: float) -> float:
    return -u0 * omega * math.sin(omega * t)


def interface_height(
    x: np.ndarray | float,
    t: float,
    *,
    depth: float,
    amplitude: float,
    k: float,
    u0: float,
    omega: float,
) -> np.ndarray | float:
    xi = x - shift(t, u0=u0, omega=omega)
    return depth + amplitude * np.cos(k * xi)


def interface_height_x(
    x: np.ndarray | float,
    t: float,
    *,
    amplitude: float,
    k: float,
    u0: float,
    omega: float,
) -> np.ndarray | float:
    xi = x - shift(t, u0=u0, omega=omega)
    return -amplitude * k * np.sin(k * xi)


def phi_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    depth: float,
    amplitude: float,
    k: float,
    u0: float,
    omega: float,
) -> np.ndarray:
    h = interface_height(x, t, depth=depth, amplitude=amplitude, k=k, u0=u0, omega=omega)
    return y - h


def velocity_exact(x: np.ndarray, t: float, *, u0: float, omega: float) -> np.ndarray:
    out = np.zeros((np.size(x), 3), dtype=float)
    out[:, 0] = uniform_velocity(t, u0=u0, omega=omega)
    return out


def pressure_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    density: float,
    gravity: float,
    depth: float,
    amplitude: float,
    k: float,
    u0: float,
    omega: float,
) -> np.ndarray:
    h = interface_height(x, t, depth=depth, amplitude=amplitude, k=k, u0=u0, omega=omega)
    return density * gravity * (h - y)


def source_x(
    x: np.ndarray,
    t: float,
    *,
    gravity: float,
    amplitude: float,
    k: float,
    u0: float,
    omega: float,
) -> np.ndarray:
    return uniform_acceleration(t, u0=u0, omega=omega) + gravity * interface_height_x(
        x, t, amplitude=amplitude, k=k, u0=u0, omega=omega
    )


def structured_quad_mesh(args: argparse.Namespace, *, k: float) -> pv.UnstructuredGrid:
    if args.element_order == 1:
        x = np.linspace(0.0, args.length, args.nx + 1)
        y = np.linspace(0.0, args.tank_height, args.ny + 1)
    else:
        x = np.linspace(0.0, args.length, 2 * args.nx + 1)
        y = np.linspace(0.0, args.tank_height, 2 * args.ny + 1)
    points = np.array([[xi, yi, 0.0] for yi in y for xi in x], dtype=float)

    cells = []
    if args.element_order == 1:
        for j in range(args.ny):
            for i in range(args.nx):
                cells.extend(
                    [
                        4,
                        node_id(i, j, args.nx),
                        node_id(i + 1, j, args.nx),
                        node_id(i + 1, j + 1, args.nx),
                        node_id(i, j + 1, args.nx),
                    ]
                )
        cell_types = np.full(args.nx * args.ny, int(pv.CellType.QUAD), dtype=np.uint8)
    else:
        for j in range(args.ny):
            for i in range(args.nx):
                ii = 2 * i
                jj = 2 * j
                cells.extend(
                    [
                        9,
                        qnode_id(ii, jj, args.nx),
                        qnode_id(ii + 2, jj, args.nx),
                        qnode_id(ii + 2, jj + 2, args.nx),
                        qnode_id(ii, jj + 2, args.nx),
                        qnode_id(ii + 1, jj, args.nx),
                        qnode_id(ii + 2, jj + 1, args.nx),
                        qnode_id(ii + 1, jj + 2, args.nx),
                        qnode_id(ii, jj + 1, args.nx),
                        qnode_id(ii + 1, jj + 1, args.nx),
                    ]
                )
        cell_types = np.full(args.nx * args.ny, int(pv.CellType.BIQUADRATIC_QUAD), dtype=np.uint8)
    grid = pv.UnstructuredGrid(
        np.array(cells, dtype=np.int64),
        cell_types,
        points,
    )

    px = points[:, 0]
    py = points[:, 1]
    grid.point_data["GlobalNodeID"] = np.arange(points.shape[0], dtype=np.int32)
    grid.point_data["phi"] = phi_exact(
        px,
        py,
        0.0,
        depth=args.depth,
        amplitude=args.amplitude,
        k=k,
        u0=args.u0,
        omega=args.omega,
    )
    grid.point_data["Velocity"] = velocity_exact(px, 0.0, u0=args.u0, omega=args.omega)
    grid.point_data["Pressure"] = pressure_exact(
        px,
        py,
        0.0,
        density=args.density,
        gravity=args.gravity,
        depth=args.depth,
        amplitude=args.amplitude,
        k=k,
        u0=args.u0,
        omega=args.omega,
    )
    grid.point_data["ManufacturedSource"] = np.column_stack(
        [
            source_x(px, 0.0, gravity=args.gravity, amplitude=args.amplitude, k=k, u0=args.u0, omega=args.omega),
            np.zeros(points.shape[0]),
            np.zeros(points.shape[0]),
        ]
    )
    grid.cell_data["GlobalElementID"] = np.arange(args.nx * args.ny, dtype=np.int32)
    return grid


def line_polydata(grid: pv.UnstructuredGrid, edges: list[tuple[int, ...]], parent_cells: list[int]) -> pv.PolyData:
    used = sorted({node for edge in edges for node in edge})
    local = {global_id: local_id for local_id, global_id in enumerate(used)}
    lines = np.array(
        [[len(edge), *[local[node] for node in edge]] for edge in edges],
        dtype=np.int64,
    ).ravel()
    poly = pv.PolyData(grid.points[np.array(used, dtype=np.int64)], lines=lines)
    poly.point_data["GlobalNodeID"] = np.array(used, dtype=np.int32)
    poly.cell_data["GlobalElementID"] = np.array(parent_cells, dtype=np.int32)
    return poly


def write_boundary_surfaces(grid: pv.UnstructuredGrid, nx: int, ny: int, surface_dir: Path) -> None:
    surface_dir.mkdir(parents=True, exist_ok=True)
    if grid.celltypes[0] == int(pv.CellType.BIQUADRATIC_QUAD):
        def split_quadratic_edges(
            edges: list[tuple[int, int, int]],
            parents: list[int],
        ) -> tuple[list[tuple[int, ...]], list[int]]:
            split_edges: list[tuple[int, ...]] = []
            split_parents: list[int] = []
            for (a, m, b), parent in zip(edges, parents):
                split_edges.extend([(a, m), (m, b)])
                split_parents.extend([parent, parent])
            return split_edges, split_parents

        left_parents = [j * nx for j in range(ny)]
        right_parents = [j * nx + (nx - 1) for j in range(ny)]
        bottom_parents = list(range(nx))
        top_parents = [(ny - 1) * nx + i for i in range(nx)]
        specs: dict[str, tuple[list[tuple[int, ...]], list[int]]] = {
            "wall_left": split_quadratic_edges(
                [(qnode_id(0, 2 * j, nx), qnode_id(0, 2 * j + 1, nx), qnode_id(0, 2 * j + 2, nx))
                 for j in range(ny)],
                left_parents,
            ),
            "wall_right": split_quadratic_edges(
                [(qnode_id(2 * nx, 2 * j, nx),
                  qnode_id(2 * nx, 2 * j + 1, nx),
                  qnode_id(2 * nx, 2 * j + 2, nx))
                 for j in range(ny)],
                right_parents,
            ),
            "wall_bottom": split_quadratic_edges(
                [(qnode_id(2 * i, 0, nx), qnode_id(2 * i + 1, 0, nx), qnode_id(2 * i + 2, 0, nx))
                 for i in range(nx)],
                bottom_parents,
            ),
            "wall_top": split_quadratic_edges(
                [(qnode_id(2 * i, 2 * ny, nx),
                  qnode_id(2 * i + 1, 2 * ny, nx),
                  qnode_id(2 * i + 2, 2 * ny, nx))
                 for i in range(nx)],
                top_parents,
            ),
        }
    else:
        specs = {
            "wall_left": (
                [(node_id(0, j, nx), node_id(0, j + 1, nx)) for j in range(ny)],
                [j * nx for j in range(ny)],
            ),
            "wall_right": (
                [(node_id(nx, j, nx), node_id(nx, j + 1, nx)) for j in range(ny)],
                [j * nx + (nx - 1) for j in range(ny)],
            ),
            "wall_bottom": (
                [(node_id(i, 0, nx), node_id(i + 1, 0, nx)) for i in range(nx)],
                list(range(nx)),
            ),
            "wall_top": (
                [(node_id(i, ny, nx), node_id(i + 1, ny, nx)) for i in range(nx)],
                [(ny - 1) * nx + i for i in range(nx)],
            ),
        }
    for name, (edges, parent_cells) in specs.items():
        line_polydata(grid, edges, parent_cells).save(surface_dir / f"{name}.vtp", binary=False)


def boundary_node_ids(args: argparse.Namespace) -> dict[str, list[int]]:
    if args.element_order == 1:
        return {
            "wall_left": [node_id(0, j, args.nx) for j in range(args.ny + 1)],
            "wall_right": [node_id(args.nx, j, args.nx) for j in range(args.ny + 1)],
            "wall_bottom": [node_id(i, 0, args.nx) for i in range(args.nx + 1)],
        }
    return {
        "wall_left": [qnode_id(0, j, args.nx) for j in range(2 * args.ny + 1)],
        "wall_right": [qnode_id(2 * args.nx, j, args.nx) for j in range(2 * args.ny + 1)],
        "wall_bottom": [qnode_id(i, 0, args.nx) for i in range(2 * args.nx + 1)],
    }


def time_samples(period: float, time_step: float) -> list[float]:
    count = max(2, int(math.ceil(period / time_step)) + 1)
    times = [i * period / (count - 1) for i in range(count)]
    times[0] = 0.0
    times[-1] = period
    return times


def write_velocity_bc_files(args: argparse.Namespace, grid: pv.UnstructuredGrid, *, period: float) -> None:
    bc_dir = CASE_DIR / "bc"
    if bc_dir.exists():
        shutil.rmtree(bc_dir)
    bc_dir.mkdir(parents=True)
    times = time_samples(period, args.time_step)
    for name, ids in boundary_node_ids(args).items():
        path = bc_dir / f"{name}_velocity.dat"
        with path.open("w") as output:
            output.write(f"2 {len(times)} {len(ids)}\n")
            for t in times:
                output.write(f"{t:.12e}\n")
            for gid in ids:
                output.write(f"{gid + 1}\n")
                for t in times:
                    u = uniform_velocity(t, u0=args.u0, omega=args.omega)
                    output.write(f"{u:.18e} 0.000000000000000000e+00\n")


def write_source_samples(args: argparse.Namespace, grid: pv.UnstructuredGrid, *, k: float) -> None:
    path = CASE_DIR / "manufactured_source_samples.csv"
    sample_count = min(args.time_steps, 10)
    times = [i * args.time_step * args.time_steps / sample_count for i in range(sample_count + 1)]
    x_stride = max(1, args.nx // 8)
    y_stride = max(1, args.ny // 8)
    if args.element_order == 1:
        sample_ids = [
            node_id(i, j, args.nx)
            for j in range(0, args.ny + 1, y_stride)
            for i in range(0, args.nx + 1, x_stride)
        ]
        final_sample = node_id(args.nx, args.ny, args.nx)
    else:
        sample_ids = [
            qnode_id(2 * i, 2 * j, args.nx)
            for j in range(0, args.ny + 1, y_stride)
            for i in range(0, args.nx + 1, x_stride)
        ]
        final_sample = qnode_id(2 * args.nx, 2 * args.ny, args.nx)
    if final_sample not in sample_ids:
        sample_ids.append(final_sample)
    with path.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["time", "node_id", "x", "y", "source_x", "source_y"])
        for t in times:
            sx = source_x(
                grid.points[:, 0],
                t,
                gravity=args.gravity,
                amplitude=args.amplitude,
                k=k,
                u0=args.u0,
                omega=args.omega,
            )
            for gid in sample_ids:
                point = grid.points[gid]
                writer.writerow([f"{t:.12e}", gid + 1, f"{point[0]:.12e}", f"{point[1]:.12e}", f"{sx[gid]:.12e}", "0.0"])


def write_solver_xml(args: argparse.Namespace) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{args.time_steps}</Number_of_time_steps>
  <Time_step_size>{args.time_step:.12g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Combine_time_series>true</Combine_time_series>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>{args.output_cadence}</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>{args.output_cadence}</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{args.time_steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="tank">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
  <Add_face name="wall_left"><Face_file_path>mesh/background/mesh-surfaces/wall_left.vtp</Face_file_path></Add_face>
  <Add_face name="wall_right"><Face_file_path>mesh/background/mesh-surfaces/wall_right.vtp</Face_file_path></Add_face>
  <Add_face name="wall_bottom"><Face_file_path>mesh/background/mesh-surfaces/wall_bottom.vtp</Face_file_path></Add_face>
  <Add_face name="wall_top"><Face_file_path>mesh/background/mesh-surfaces/wall_top.vtp</Face_file_path></Add_face>
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Element_order>{args.element_order}</Element_order>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>1.0e-5</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>prescribed_data</Velocity_source>
  <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
  <Advection_velocity_from_field>Velocity</Advection_velocity_from_field>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <Enable_reinitialization>true</Enable_reinitialization>
  <Reinitialization_method>projection</Reinitialization_method>
  <Reinitialization_cadence_steps>10</Reinitialization_cadence_steps>
  <Reinitialization_max_iterations>4</Reinitialization_max_iterations>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type="Volume_integral"><Volume>true</Volume></Output>
  <LS type="Direct">
    <Linear_algebra type="eigen"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-6</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Element_order>{args.element_order}</Element_order>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>20</Max_iterations>
  <Tolerance>5.0e-3</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>{args.density:.12g}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>{-args.gravity:.12g}</Force_y>
  <Force_z>0.0</Force_z>
  <Momentum_source_field_name>ManufacturedSource</Momentum_source_field_name>
  <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_field_name>Pressure</Hydrostatic_pressure_field_name>
  <Viscosity model="Constant"><Value>{args.viscosity:.12g}</Value></Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>
  <Output type="Volume_integral"><Volume>true</Volume></Output>
  <LS type="Direct">
    <Linear_algebra type="eigen"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-5</Tolerance>
    <Absolute_tolerance>1.0e-8</Absolute_tolerance>
    <NS_GM_max_iterations>150</NS_GM_max_iterations>
    <NS_GM_tolerance>1.0e-5</NS_GM_tolerance>
    <NS_CG_max_iterations>150</NS_CG_max_iterations>
    <NS_CG_tolerance>1.0e-5</NS_CG_tolerance>
    <NS_min_outer_iterations>1</NS_min_outer_iterations>
    <NS_Schur_preconditioner>blockdiag-l</NS_Schur_preconditioner>
    <NS_Momentum_approximation>ilu-k</NS_Momentum_approximation>
    <NS_Use_coupled_outer_FGMRES>true</NS_Use_coupled_outer_FGMRES>
  </LS>

  <Add_BC name="wall_left">
    <Type>Dir</Type>
    <Time_dependence>General</Time_dependence>
    <Temporal_and_spatial_values_file_path>bc/wall_left_velocity.dat</Temporal_and_spatial_values_file_path>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Time_dependence>General</Time_dependence>
    <Temporal_and_spatial_values_file_path>bc/wall_right_velocity.dat</Temporal_and_spatial_values_file_path>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Dir</Type>
    <Time_dependence>General</Time_dependence>
    <Temporal_and_spatial_values_file_path>bc/wall_bottom_velocity.dat</Temporal_and_spatial_values_file_path>
  </Add_BC>

  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>mms_traveling_interface_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
    <Implicit_cut_quadrature_backend>SayeHyperrectangle</Implicit_cut_quadrature_backend>
    <Implicit_cut_fallback_policy>Fail</Implicit_cut_fallback_policy>
    <Implicit_cut_root_tolerance>1.0e-10</Implicit_cut_root_tolerance>
    <Implicit_cut_max_subdivision_depth>8</Implicit_cut_max_subdivision_depth>
    <Generated_interface_quadrature_order>2</Generated_interface_quadrature_order>
    <Interface_quadrature_order>2</Interface_quadrature_order>
    <Volume_quadrature_order>2</Volume_quadrature_order>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Enable_velocity_extension>true</Enable_velocity_extension>
    <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_velocity_gradient_penalty>1.0</Cut_cell_velocity_gradient_penalty>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
"""
    (CASE_DIR / "solver.xml").write_text(xml)


def write_expected(args: argparse.Namespace, *, k: float, period: float, final_time: float) -> None:
    y_centroid = args.depth / 2.0 + args.amplitude * args.amplitude / (4.0 * args.depth)
    tolerances = {
        "phi_l2_abs": 1.0e-2,
        "phi_max_abs": 3.0e-2,
        "interface_mean_abs": 5.0e-3,
        "interface_amplitude_relative": 0.15,
        "interface_shift_abs": 2.0e-2,
        "interface_l2_height_abs": 1.0e-2,
        "interface_max_height_abs": 3.0e-2,
        "area_relative": 3.0e-2,
        "centroid_y_abs": 1.0e-2,
        "velocity_relative_l2": 0.15,
        "velocity_mean_abs": 2.0e-2,
        "pressure_relative_rms": 0.15,
        "pressure_rms_after_offset_relative": 0.10,
        "interface_pressure_abs": max(10.0, 0.05 * args.density * args.gravity * args.amplitude),
        "manufactured_residual_abs": 1.0e-12,
        "level_set_residual_abs": 1.0e-12,
    }
    expected = {
        "case": "mms_traveling_interface_2d",
        "purpose": "Exact manufactured moving free-surface test for OOP unfitted level-set infrastructure.",
        "mesh": {
            "nx": args.nx,
            "ny": args.ny,
            "element_order": args.element_order,
            "length": args.length,
            "mean_depth": args.depth,
            "tank_height": args.tank_height,
            "element_size": max(args.length / args.nx, args.tank_height / args.ny),
        },
        "fluid": {
            "density": args.density,
            "dynamic_viscosity": args.viscosity,
            "kinematic_viscosity": args.viscosity / args.density,
            "gravity": args.gravity,
            "body_force": [0.0, -args.gravity, 0.0],
            "force_sign_convention": "OOP residual uses rho*(... - f) + grad(p) - div(stress), so hydrostatic balance is grad(p)=rho*f.",
        },
        "run": {
            "time_step": args.time_step,
            "time_steps": args.time_steps,
            "final_time": final_time,
            "output_cadence": args.output_cadence,
            "combine_time_series": True,
        },
        "analytic_solution": {
            "L": args.length,
            "H0": args.depth,
            "H_tank": args.tank_height,
            "amplitude": args.amplitude,
            "k": k,
            "U0": args.u0,
            "Omega": args.omega,
            "period": period,
            "X(t)": "(U0/Omega)*sin(Omega*t)",
            "U(t)": "U0*cos(Omega*t)",
            "Udot(t)": "-U0*Omega*sin(Omega*t)",
            "h(x,t)": "H0 + A*cos(k*(x-X(t)))",
            "phi": "y - h(x,t)",
            "velocity": "[U(t), 0, 0]",
            "pressure": "rho*g*(h(x,t)-y)",
            "momentum_source": {
                "s_x": "Udot(t) + g*h_x(x,t)",
                "h_x": "-A*k*sin(k*(x-X(t)))",
                "s_y": "0",
            },
            "expected_area": args.length * args.depth,
            "expected_y_centroid": y_centroid,
        },
        "boundary_conditions": {
            "preferred": "periodic x plus bottom velocity [U(t),0,0]",
            "implemented_xml_fallback": "exact time/space Dirichlet velocity on left, right, and bottom walls",
            "free_surface": {
                "implementation": "UnfittedLevelSet",
                "active_domain": "LevelSetNegative",
                "external_pressure": 0.0,
                "surface_tension": 0.0,
                "generated_interface_geometry": "HighOrderImplicit",
                "implicit_cut_quadrature_backend": "SayeHyperrectangle",
                "implicit_cut_fallback_policy": "Fail",
                "quadrature_order": 2,
                "interface_quadrature_order": 2,
                "volume_quadrature_order": 2,
                "implicit_cut_root_tolerance": 1.0e-10,
                "implicit_cut_max_subdivision_depth": 8,
            },
        },
        "solver_feature_status": {
            "periodic_x_boundary_from_xml": False,
            "spatial_temporal_momentum_source_from_xml": True,
            "time_dependent_boundary_data": True,
            "high_order_implicit_geometry_from_xml": True,
            "solver_xml_is_full_exact_mms": True,
            "blocker": "",
        },
        "verification": {
            "checked_fields": ["phi", "Velocity", "Pressure"],
            "wet_region": "phi_exact < -2*h_mesh and finite result values",
            "suggested_tolerances": tolerances,
        },
        "suggested_tolerances": tolerances,
    }
    (CASE_DIR / "expected_results.json").write_text(json.dumps(expected, indent=2) + "\n")


def write_benchmark(args: argparse.Namespace, *, k: float, period: float, final_time: float) -> None:
    benchmark = {
        "name": "mms_traveling_interface_2d",
        "type": "fluid_free_surface_unfitted_level_set_mms",
        "solver": "new_oop",
        "description": "Manufactured horizontally translating wavy free surface with exact fields and source audit.",
        "mesh": {"nx": args.nx, "ny": args.ny, "element_order": args.element_order},
        "run": {"dt": args.time_step, "steps": args.time_steps, "final_time": final_time},
        "targets": {
            "k": k,
            "period": period,
            "most_important_metrics": [
                "phi_rms_error",
                "interface_shift_error",
                "velocity_relative_l2_error",
                "pressure_relative_rms_error",
                "interface_pressure_rms",
                "manufactured_residual_x_max",
            ],
        },
        "solver_features": {
            "momentum_source_field": "ManufacturedSource",
            "spatial_temporal_momentum_source_from_xml": True,
            "generated_interface_geometry": "HighOrderImplicit",
            "implicit_cut_quadrature_backend": "SayeHyperrectangle",
            "implicit_cut_fallback_policy": "Fail",
            "quadrature_order": 2,
            "interface_quadrature_order": 2,
            "volume_quadrature_order": 2,
        },
    }
    (CASE_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")


def write_readme(args: argparse.Namespace, *, k: float, period: float, final_time: float) -> None:
    readme = f"""# MMS Traveling Interface 2D

This is a manufactured moving free-surface test for the new OOP fluid solver
with an unfitted level-set active domain. Negative `phi` denotes liquid.
The default generated mesh uses biquadratic Quad9 cells and `Element_order=2`
so edge and cell-center level-set DOFs define the moving implicit geometry.

The exact fields are

```text
X(t) = (U0/Omega)*sin(Omega*t)
U(t) = U0*cos(Omega*t)
h(x,t) = H0 + A*cos(k*(x-X(t)))
phi(x,y,t) = y - h(x,t)
u(x,y,t) = [U(t), 0, 0]
p(x,y,t) = rho*g*(h(x,t)-y)
```

The free-surface pressure is exactly zero at `phi=0`, and the velocity
gradient is zero, so viscous free-surface stress does not alter the scalar
pressure check.

The required manufactured source is

```text
s_x(x,t) = Udot(t) + g*h_x(x,t)
         = -U0*Omega*sin(Omega*t) - g*A*k*sin(k*(x-X(t)))
s_y(x,t) = 0
```

The OOP residual convention is `rho*(... - f) + grad(p) - div(stress)`, so
hydrostatic balance uses `grad(p)=rho*f`. With `Force_y=-g`, the source above
is the acceleration-like extra body term required to make the MMS exact.

## Solver Source Wiring

The generated XML sets `Momentum_source_field_name=ManufacturedSource`, so the
OOP Navier-Stokes residual consumes the spatially varying manufactured
acceleration from mesh point data in addition to constant `Force_x/y/z`.
`manufactured_source_samples.csv` is still generated as an independent audit of
the source values used by the verifier.

Default parameters:

- `L = {args.length}`
- `H0 = {args.depth}`
- `H_tank = {args.tank_height}`
- `A = {args.amplitude}`
- `k = {k:.12g}`
- `U0 = {args.u0}`
- `Omega = {args.omega:.12g}`
- `period = {period:.12g}`
- `final_time = {final_time:.12g}`
- `element_order = {args.element_order}`

The free-surface boundary requests `Generated_interface_geometry=HighOrderImplicit`
with `Implicit_cut_quadrature_backend=SayeHyperrectangle`, fail-fast fallback,
and requested cut volume/interface quadrature order 2.

## Generate

```bash
python3 generate_case.py
```

Useful linear-geometry comparison override:

```bash
python3 generate_case.py --element-order 1
```

## Run

```bash
/path/to/svmultiphysics solver.xml
```

## Verify

```bash
python3 verify_expected_results.py
```

To check the generated initial condition without running the solver:

```bash
python3 verify_expected_results.py mesh/background/mesh-complete.mesh.vtu --time 0
```

The verifier reconstructs and deduplicates `phi=0`, fits the cosine mode,
checks area and y-centroid, compares `phi`, velocity, and pressure against the
exact fields, interpolates pressure onto the free surface, and reports analytic
momentum and level-set residual audits.

Key metrics:

- `phi_rms_error`, `phi_max_abs_error`: level-set advection error.
- `interface_cos_coeff`, `interface_sin_coeff`, `interface_shift_error`:
  reconstructed free-surface phase/translation.
- `area_relative_error`, `centroid_y_error`: cut-volume active-domain checks.
- `velocity_relative_l2_error`, `velocity_mean_x_error`: exact uniform-flow
  checks in the wet region.
- `pressure_relative_rms_error`: exact free-surface-gauge pressure check in the
  wet region.
- `interface_pressure_rms`, `interface_pressure_max_abs`: direct pressure check
  on reconstructed `phi=0`.
- `manufactured_residual_x_max`, `manufactured_residual_y_max`,
  `level_set_residual_max`: analytic residual audits for sign/source mistakes.
"""
    (CASE_DIR / "README.md").write_text(readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--element-order", type=int, choices=(1, 2), default=DEFAULT_ELEMENT_ORDER)
    parser.add_argument("--length", type=float, default=DEFAULT_L)
    parser.add_argument("--depth", type=float, default=DEFAULT_H0)
    parser.add_argument("--tank-height", type=float, default=DEFAULT_H_TANK)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--viscosity", type=float, default=DEFAULT_VISCOSITY)
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--u0", type=float, default=DEFAULT_U0)
    parser.add_argument("--omega", type=float, default=DEFAULT_OMEGA)
    parser.add_argument("--time-step", type=float, default=None)
    parser.add_argument("--time-steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument("--final-time", type=float, default=DEFAULT_FINAL_TIME)
    parser.add_argument("--output-cadence", type=int, default=DEFAULT_OUTPUT_CADENCE)
    args = parser.parse_args()

    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must be at least 2")
    if args.element_order not in (1, 2):
        raise ValueError("--element-order must be 1 or 2")
    if args.depth <= 0.0 or args.tank_height <= args.depth + args.amplitude:
        raise ValueError("--tank-height must leave dry space above the free surface")
    if args.viscosity <= 0.0:
        raise ValueError("--viscosity must be positive")
    if args.time_step is None:
        args.time_step = args.final_time / args.time_steps
    else:
        args.final_time = args.time_step * args.time_steps
    return args


def main() -> None:
    args = parse_args()
    params = mms_parameters(length=args.length, omega=args.omega, u0=args.u0)
    k = params["k"]
    period = params["period"]
    final_time = args.time_step * args.time_steps

    mesh_dir = CASE_DIR / MESH_SUBDIR
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True)

    grid = structured_quad_mesh(args, k=k)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu", binary=False)
    write_boundary_surfaces(grid, args.nx, args.ny, mesh_dir / "mesh-surfaces")
    write_velocity_bc_files(args, grid, period=period)
    write_source_samples(args, grid, k=k)
    write_solver_xml(args)
    write_expected(args, k=k, period=period, final_time=final_time)
    write_benchmark(args, k=k, period=period, final_time=final_time)
    write_readme(args, k=k, period=period, final_time=final_time)

    print(f"Generated {CASE_DIR}")
    print(f"  mesh: {mesh_dir / 'mesh-complete.mesh.vtu'}")
    print(f"  dt: {args.time_step:.12g}")
    print(f"  final_time: {final_time:.12g}")
    print(f"  period: {period:.12g}")
    print("  source samples: manufactured_source_samples.csv")


if __name__ == "__main__":
    main()
