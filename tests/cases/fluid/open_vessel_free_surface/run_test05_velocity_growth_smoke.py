#!/usr/bin/env python3
"""Run a short unfitted dam-break velocity-growth solver probe."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[4]
CASE_ROOT = ROOT / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
CASES = {
    "mini2d": None,
    "static2d": None,
    "d18": CASE_ROOT / "spheric_test05_wet_bed_d18",
    "d38": CASE_ROOT / "spheric_test05_wet_bed_d38",
    "mms2d": CASE_ROOT / "mms_traveling_interface_2d",
}
CASE_GATE_X = {
    "mini2d": 0.4,
    "static2d": 0.5,
    "mms2d": 0.5,
}
CUT_CONTEXT_VOLUME_RE = re.compile(r"active_side_volume=([-+0-9.eE]+)")
CUT_ASSEMBLY_VOLUME_RE = re.compile(r"(?<!_)active_wet_volume=([-+0-9.eE]+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
RANK_RE = re.compile(r"\[R([0-9]+)\]")
COMPONENT_NORM_RE = re.compile(
    r"\[(.*?) norm=([-+0-9.eE]+) mean=([-+0-9.eE]+)"
    r"(?: min=([-+0-9.eE]+) max=([-+0-9.eE]+))?\]"
)
JACOBIAN_COMPONENT_NORM_RE = re.compile(
    r"\[(.*?) fd=([-+0-9.eE]+) total_err=([-+0-9.eE]+)"
    r" matrix_err=([-+0-9.eE]+)\]"
)
JACOBIAN_COMPONENT_DETAIL_RE = re.compile(
    r"\[(.*?) base=([-+0-9.eE]+) perturbed=([-+0-9.eE]+)"
    r" fd=([-+0-9.eE]+) matrix=([-+0-9.eE]+) full=([-+0-9.eE]+)"
    r" matrix_err=([-+0-9.eE]+) total_err=([-+0-9.eE]+)"
    r" sign_flip_err=([-+0-9.eE]+)\]"
)
JACOBIAN_TOP_MISMATCH_RE = re.compile(
    r"\[(.*?) fd=([-+0-9.eE]+) jv=([-+0-9.eE]+) err=([-+0-9.eE]+)\]"
)
DOUBLE_BAR_VALUE_RE = re.compile(r"\|\|([^|]+)\|\|=([-+0-9.eE]+)")
VECTOR_COMPONENT_LABEL_RE = re.compile(r"label=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
LINEAR_SOLVER_RE = re.compile(
    r"SimulationBuilder: linear solver method=(?P<method>\S+)"
    r" preconditioner=(?P<preconditioner>\S+)"
    r" rel_tol=(?P<rel_tol>[-+0-9.eE]+)"
    r" abs_tol=(?P<abs_tol>[-+0-9.eE]+)"
    r" max_iter=(?P<max_iter>[0-9]+)"
    r"(?: block_layout=(?P<block_layout>\[[^\]]+\]))?"
    r"(?: saddle_point=\((?P<saddle_momentum>[0-9]+),(?P<saddle_constraint>[0-9]+)\))?"
)
TIME_STEPPING_RE = re.compile(
    r"Time stepping: Number_of_time_steps=(?P<number_of_time_steps>[0-9]+)"
    r" Time_step_size=(?P<time_step_size>[-+0-9.eE]+)"
)
TRANSIENT_SOLVE_RE = re.compile(
    r"Transient solve: t0=(?P<t0>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
    r" t_end=(?P<t_end>[-+0-9.eE]+)"
    r" max_steps=(?P<max_steps>[0-9]+)"
    r" scheme=(?P<scheme>\S+)"
    r" rho_inf=(?P<rho_inf>[-+0-9.eE]+)"
    r" newton\(max_it=(?P<newton_max_it>[0-9]+),"
    r" min_it=(?P<newton_min_it>[0-9]+),"
    r" abs_tol=(?P<newton_abs_tol>[-+0-9.eE]+),"
    r" rel_tol=(?P<newton_rel_tol>[-+0-9.eE]+)\)"
)
TIMELOOP_NONLINEAR_RE = re.compile(
    r"TimeLoop: nonlinear_done step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" converged=(?P<converged>[01])"
    r" iters=(?P<nonlinear_iterations>[0-9]+)"
    r" \|\|r\|\|=(?P<residual>[-+0-9.eE]+)"
    r" \|\|r_field\|\|=(?P<field_residual>[-+0-9.eE]+)"
    r" \|\|r_aux\|\|=(?P<aux_residual>[-+0-9.eE]+)"
    r" \(linear: converged=(?P<linear_converged>[01])"
    r" iters=(?P<linear_iterations>[0-9]+)"
    r" rel=(?P<linear_relative_residual>[-+0-9.eE]+)\)"
)
TIMELOOP_ACCEPTED_RE = re.compile(
    r"TimeLoop: step_accepted step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
)
VTK_WRITE_RE = re.compile(r"Wrote VTK: (?P<path>.+)$")
ASSEMBLY_TIMING_HEADER_RE = re.compile(
    r"assembleOperator TIMING \(rank (?P<rank>[0-9]+), op='(?P<op>[^']+)'\)"
)
ASSEMBLY_TIMING_VALUE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9 ()+]+):\s+"
    r"(?P<seconds>[-+0-9.eE]+)\s+s"
)
INTERIOR_FACE_TIMING_VALUE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=\s*([-+0-9.eE]+)"
)
STATIC_INTERFACE_HEIGHT = 0.53
STATE_SYNC_CUT_CONTEXT_PROVENANCES = {
    "accepted",
    "residual",
    "jacobian",
    "jacobian_and_residual",
    "line_search_trial",
    "restored",
    "final_residual",
}
VECTOR_CUT_CONTEXT_PROVENANCES = {
    "before_physics_solve",
    "accepted_step",
    "steady_initial",
    "steady_accepted",
}


def point_array(mesh: pv.DataSet, name: str) -> np.ndarray:
    if name not in mesh.point_data:
        names = ", ".join(sorted(mesh.point_data.keys()))
        raise ValueError(f"missing point array {name!r}; found: {names}")
    return np.asarray(mesh.point_data[name])


def global_node_ids(mesh: pv.DataSet) -> np.ndarray:
    for name in ("GlobalNodeID", "GlobalVertexID"):
        if name in mesh.point_data:
            return np.asarray(mesh.point_data[name], dtype=np.int64)
    raise ValueError("missing GlobalNodeID or GlobalVertexID point array")


def result_indices_by_initial_gid(initial: pv.DataSet,
                                  result: pv.DataSet) -> np.ndarray:
    result_first_index: dict[int, int] = {}
    for index, gid in enumerate(global_node_ids(result)):
        result_first_index.setdefault(int(gid), index)

    missing = []
    indices = []
    for gid in global_node_ids(initial):
        key = int(gid)
        if key not in result_first_index:
            missing.append(key)
        else:
            indices.append(result_first_index[key])
    if missing:
        raise ValueError(f"result omits {len(missing)} initial node ids")
    return np.asarray(indices, dtype=np.int64)


def cell_measure(mesh: pv.DataSet) -> np.ndarray:
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
    for name in ("Volume", "Area"):
        if name in sized.cell_data:
            values = np.asarray(sized.cell_data[name], dtype=float)
            if np.any(np.abs(values) > 0.0):
                return values
    raise ValueError("mesh cell sizes do not include nonzero area or volume")


def text(root: ET.Element, path: str) -> str:
    element = root.find(path)
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def require_text(root: ET.Element, path: str, expected: str) -> None:
    value = text(root, path)
    if value != expected:
        raise ValueError(f"{path} is {value!r}, expected {expected!r}")


def set_text(parent: ET.Element, name: str, value: str) -> None:
    element = parent.find(name)
    if element is None:
        element = ET.SubElement(parent, name)
    element.text = value


def free_surface_bc(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") != "fluid":
            continue
        for bc in equation.findall("Add_BC"):
            if bc.attrib.get("name") == "free_surface":
                return bc
    raise ValueError("missing fluid free-surface boundary condition")


def fluid_equation(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") == "fluid":
            return equation
    raise ValueError("missing fluid equation")


def navier_stokes_linear_solver(root: ET.Element) -> ET.Element:
    for solver in fluid_equation(root).findall("LS"):
        if solver.attrib.get("type") == "NS":
            return solver
    raise ValueError("missing fluid NS linear solver block")


def configure_solver(solver_xml: Path,
                     steps: int,
                     time_step_size: float | None = None,
                     disable_cut_stabilization: bool = False,
                     max_nonlinear_iterations: int | None = None,
                     linear_relative_tolerance: float | None = None,
                     linear_absolute_tolerance: float | None = None,
                     linear_max_iterations: int | None = None,
                     ns_gm_max_iterations: int | None = None,
                     ns_cg_max_iterations: int | None = None,
                     ns_gm_tolerance: float | None = None,
                     ns_cg_tolerance: float | None = None,
                     linear_solver_type: str | None = None,
                     disable_coupled_outer_fgmres: bool = False,
                     disable_cut_metadata_scale: bool = False,
                     disable_vtk_output: bool = False,
                     final_output_only: bool = False,
                     vtk_save_increment: int | None = None,
                     start_saving_after_step: int | None = None) -> None:
    tree = ET.parse(solver_xml)
    root = tree.getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise ValueError("missing GeneralSimulationParameters")

    set_text(general, "Number_of_time_steps", str(steps))
    set_text(general, "Save_results_to_VTK_format", "false" if disable_vtk_output else "true")
    if disable_vtk_output:
        set_text(general, "Combine_time_series", "false")
    set_text(general, "Name_prefix_of_saved_VTK_files", "result")
    save_increment = vtk_save_increment if vtk_save_increment is not None else 1
    start_step = start_saving_after_step if start_saving_after_step is not None else 1
    if final_output_only:
        save_increment = steps
        start_step = steps
    set_text(general, "Increment_in_saving_VTK_files", str(save_increment))
    set_text(general, "Start_saving_after_time_step", str(start_step))
    set_text(general, "Increment_in_saving_restart_files", str(steps))
    if time_step_size is not None:
        set_text(general, "Time_step_size", f"{time_step_size:.16g}")

    if max_nonlinear_iterations is not None:
        for equation in root.findall("Add_equation"):
            set_text(equation, "Max_iterations", str(max_nonlinear_iterations))

    needs_ns_solver = (
        linear_relative_tolerance is not None or
        linear_absolute_tolerance is not None or
        linear_max_iterations is not None or
        ns_gm_max_iterations is not None or
        ns_cg_max_iterations is not None or
        ns_gm_tolerance is not None or
        ns_cg_tolerance is not None or
        linear_solver_type is not None or
        disable_coupled_outer_fgmres
    )
    ns_solver = navier_stokes_linear_solver(root) if needs_ns_solver else None
    if linear_solver_type is not None:
        assert ns_solver is not None
        ns_solver.set("type", linear_solver_type)
    if linear_relative_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "Tolerance", f"{linear_relative_tolerance:.16g}")
    if linear_absolute_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "Absolute_tolerance", f"{linear_absolute_tolerance:.16g}")
    if linear_max_iterations is not None:
        assert ns_solver is not None
        set_text(ns_solver, "Max_iterations", str(linear_max_iterations))
    if ns_gm_max_iterations is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_GM_max_iterations", str(ns_gm_max_iterations))
    if ns_cg_max_iterations is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_CG_max_iterations", str(ns_cg_max_iterations))
    if ns_gm_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_GM_tolerance", f"{ns_gm_tolerance:.16g}")
    if ns_cg_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_CG_tolerance", f"{ns_cg_tolerance:.16g}")

    free_surface = free_surface_bc(root)
    require_text(free_surface, "Implementation", "UnfittedLevelSet")
    require_text(free_surface, "Active_domain", "LevelSetNegative")
    require_text(free_surface, "Active_domain_method", "CutVolume")
    if disable_cut_metadata_scale:
        set_text(free_surface, "Use_cut_metadata_scale", "false")
    else:
        require_text(free_surface, "Use_cut_metadata_scale", "true")
    if disable_cut_stabilization:
        set_text(free_surface, "Enable_cut_cell_stabilization", "false")
    else:
        require_text(free_surface, "Enable_cut_cell_stabilization", "true")

    if disable_coupled_outer_fgmres:
        assert ns_solver is not None
        set_text(ns_solver, "NS_Use_coupled_outer_FGMRES", "false")

    tree.write(solver_xml, encoding="UTF-8", xml_declaration=True)


def solver_candidates() -> list[Path]:
    env_value = os.environ.get("SVMULTIPHYSICS_EXECUTABLE")
    paths = []
    if env_value:
        paths.append(Path(env_value))
    paths.extend([
        ROOT / "build/svMultiPhysics-build/bin/svmultiphysics",
        ROOT / "build-oop-clean-20260430/svMultiPhysics-build/bin/svmultiphysics",
    ])
    return paths


def resolve_solver(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        raise FileNotFoundError(f"solver executable not found: {explicit}")
    for path in solver_candidates():
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(
        "solver executable not found; set SVMULTIPHYSICS_EXECUTABLE or pass --solver"
    )


def solver_command(solver: Path, args: argparse.Namespace) -> list[str]:
    if args.mpi_ranks is None:
        return [str(solver), "solver.xml"]
    if args.mpi_ranks < 1:
        raise ValueError("--mpi-ranks must be at least 1")
    return [str(args.mpiexec), "-np", str(args.mpi_ranks), str(solver), "solver.xml"]


def solver_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.enable_blockschur_true_residual_retry:
        env["SVMP_FSILS_ENABLE_BLOCKSCHUR_TRUE_RESIDUAL_RETRY"] = "1"
    if args.enable_jacobian_check:
        env["SVMP_FE_JACOBIAN_CHECK"] = "1"
        if args.jacobian_check_iteration is not None:
            env["SVMP_FE_JACOBIAN_CHECK_IT"] = str(args.jacobian_check_iteration)
        if args.jacobian_check_step is not None:
            env["SVMP_FE_JACOBIAN_CHECK_STEP"] = f"{args.jacobian_check_step:.16g}"
        if args.jacobian_check_components:
            env["SVMP_FE_JACOBIAN_CHECK_COMPONENTS"] = args.jacobian_check_components
    if args.enable_newton_direction_check:
        env["SVMP_NEWTON_DIRECTION_CHECK"] = "1"
    if args.enable_linear_solve_history:
        env["SVMP_DEBUG_LINEAR_SOLVE_HISTORY"] = "1"
        if args.linear_solve_history_max_calls is not None:
            env["SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS"] = str(
                args.linear_solve_history_max_calls
            )
    if args.enable_linear_solve_component_norms:
        env["SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS"] = "1"
        if args.linear_solve_component_norms_max_newton_it is not None:
            env["SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS_MAX_NEWTON_IT"] = str(
                args.linear_solve_component_norms_max_newton_it
            )
    if args.enable_form_block_diagnostics:
        env["SVMP_FE_FORM_BLOCK_DIAGNOSTICS"] = "1"
    if args.enable_interior_face_timing:
        env["SVMP_INTERIOR_FACE_TIMING"] = "1"
    return env


def copy_case_from_ref(case_dir: Path, destination: Path, source_ref: str) -> None:
    relative = case_dir.relative_to(ROOT)
    completed = subprocess.run(
        ["git", "archive", "--format=tar", source_ref, str(relative)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))

    archive_root = destination.parent / "_archive"
    archive_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout)) as archive:
        try:
            archive.extractall(archive_root, filter="data")
        except TypeError:
            archive.extractall(archive_root)
    shutil.move(str(archive_root / relative), destination)


def copy_case(case_dir: Path, destination: Path, source_ref: str | None) -> None:
    if source_ref is not None:
        copy_case_from_ref(case_dir, destination, source_ref)
        return

    def ignore(_path: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if re.match(r"result_.*\.p?vtu$", name):
                ignored.add(name)
            elif name in {"result.pvd", "1-procs", "2-procs", "3-procs", "4-procs"}:
                ignored.add(name)
            elif name.startswith("restart") or name.endswith(".log"):
                ignored.add(name)
        return ignored

    shutil.copytree(case_dir, destination, ignore=ignore)


def write_boundary(path: Path,
                   points: np.ndarray,
                   node_ids: list[int],
                   first_cell_id: int) -> None:
    lines = []
    for index in range(len(node_ids) - 1):
        lines.extend([2, index, index + 1])
    poly = pv.PolyData()
    poly.points = points[node_ids]
    poly.lines = np.asarray(lines, dtype=np.int64)
    poly.point_data["GlobalNodeID"] = np.asarray(node_ids, dtype=np.int64)
    poly.cell_data["GlobalElementID"] = np.arange(
        first_cell_id, first_cell_id + len(node_ids) - 1, dtype=np.int64)
    poly.save(path)


def write_mini_mesh(case_dir: Path, static: bool = False) -> tuple[int, float]:
    nx = 8
    ny = 8
    tank_height = 1.0
    tank_length = 1.0
    bed_depth = 0.2
    column_height = 0.75
    column_width = 0.4
    rho = 998.2
    gravity = 9.81

    xs = np.linspace(0.0, tank_length, nx + 1)
    ys = np.linspace(0.0, tank_height, ny + 1)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)

    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            cells.extend([
                4,
                lower_left,
                lower_left + 1,
                lower_left + nx + 2,
                lower_left + nx + 1,
            ])
    cell_types = np.full(nx * ny, pv.CellType.QUAD, dtype=np.uint8)
    grid = pv.UnstructuredGrid(np.asarray(cells, dtype=np.int64), cell_types, points)

    x = points[:, 0]
    y = points[:, 1]
    if static:
        phi = y - STATIC_INTERFACE_HEIGHT
        pressure = np.zeros(points.shape[0], dtype=float)
    else:
        phi = np.minimum(y - bed_depth, np.maximum(x - column_width, y - column_height))
        free_surface_height = np.where(x <= column_width, column_height, bed_depth)
        pressure = rho * gravity * np.maximum(free_surface_height - y, 0.0)
        pressure[phi > 0.0] = 0.0

    grid.point_data["GlobalNodeID"] = np.arange(points.shape[0], dtype=np.int64)
    grid.point_data["phi"] = phi
    grid.point_data["Pressure"] = pressure
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(nx * ny, dtype=np.int64)

    mesh_dir = case_dir / "mesh/background"
    surface_dir = mesh_dir / "mesh-surfaces"
    surface_dir.mkdir(parents=True)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu")

    left = [j * (nx + 1) for j in range(ny + 1)]
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    bottom = list(range(nx + 1))
    top = [ny * (nx + 1) + i for i in range(nx + 1)]
    write_boundary(surface_dir / "wall_left.vtp", points, left, 0)
    write_boundary(surface_dir / "wall_right.vtp", points, right, ny)
    write_boundary(surface_dir / "wall_bottom.vtp", points, bottom, 2 * ny)
    write_boundary(surface_dir / "wall_top.vtp", points, top, 2 * ny + nx)

    gauge_node = 0
    gauge_pressure = 0.0 if static else float(rho * gravity * column_height)
    return gauge_node, gauge_pressure


def write_mini_solver_xml(case_dir: Path,
                          steps: int,
                          gauge_node: int,
                          gauge_pressure: float,
                          static: bool = False) -> None:
    force_y = "0.0" if static else "-9.81"
    hydrostatic_initialization = "false" if static else "true"
    hydrostatic_reference_y = STATIC_INTERFACE_HEIGHT if static else 0.75
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{gauge_pressure:.16g}\n", encoding="utf-8")
    (case_dir / "solver.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{steps}</Number_of_time_steps>
  <Time_step_size>0.001</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>1</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="tank">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
  <Add_face name="wall_left">
    <Face_file_path>mesh/background/mesh-surfaces/wall_left.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_right">
    <Face_file_path>mesh/background/mesh-surfaces/wall_right.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_bottom">
    <Face_file_path>mesh/background/mesh-surfaces/wall_bottom.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_top">
    <Face_file_path>mesh/background/mesh-surfaces/wall_top.vtp</Face_file_path>
  </Add_face>
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>2</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>constant</Velocity_source>
  <Constant_velocity>0.0 0.0 0.0</Constant_velocity>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <Enable_reinitialization>false</Enable_reinitialization>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>fsils</Preconditioner>
    </Linear_algebra>
    <Max_iterations>50</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>8</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>998.2</Density>
  <Force_x>0.0</Force_x>
  <Force_y>{force_y}</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>{hydrostatic_initialization}</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 {hydrostatic_reference_y:.16g} 0.0</Hydrostatic_pressure_reference_point>
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
    <Values_file_path>pressure_gauge.csv</Values_file_path>
  </Node_pressure_constraints>
  <Viscosity model="Constant">
    <Value>1.003e-3</Value>
  </Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>fsils</Preconditioner>
    </Linear_algebra>
    <Max_iterations>150</Max_iterations>
    <Krylov_space_dimension>80</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
  <Add_BC name="wall_left">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>open_vessel_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_velocity_gradient_penalty>1.0</Cut_cell_velocity_gradient_penalty>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
""", encoding="utf-8")


def write_mini_case(case_dir: Path, steps: int, static: bool = False) -> None:
    case_dir.mkdir(parents=True)
    gauge_node, gauge_pressure = write_mini_mesh(case_dir, static)
    write_mini_solver_xml(case_dir, steps, gauge_node, gauge_pressure, static)


def result_path(case_dir: Path, step: int) -> Path:
    names = [
        f"result_{step:03d}.vtu",
        f"result_{step:03d}.pvtu",
        f"1-procs/result_{step:03d}.vtu",
        f"1-procs/result_{step:03d}.pvtu",
    ]
    for name in names:
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted([*case_dir.rglob("result_*.vtu"), *case_dir.rglob("result_*.pvtu")])
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no result file found under {case_dir}")


def value_span(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def parse_active_volume_history(solver_output: str) -> dict[str, Any]:
    context_volumes = [
        float(match.group(1))
        for match in CUT_CONTEXT_VOLUME_RE.finditer(solver_output)
    ]
    assembly_volumes = [
        float(match.group(1))
        for match in CUT_ASSEMBLY_VOLUME_RE.finditer(solver_output)
    ]

    return {
        "cut_context_active_side_volumes": context_volumes,
        "assembly_active_wet_volumes": assembly_volumes,
        "cut_context_active_side_volume_change": value_span(context_volumes),
        "assembly_active_wet_volume_change": value_span(assembly_volumes),
    }


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    try:
        if re.fullmatch(r"[-+]?[0-9]+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict[str, Any]:
    values = {
        match.group(1): parse_scalar(match.group(2))
        for match in KEY_VALUE_RE.finditer(line)
    }
    rank_match = RANK_RE.search(line)
    if rank_match is not None:
        values["rank"] = int(rank_match.group(1))
    return values


def parse_interior_face_timing(line: str) -> dict[str, Any]:
    values = {
        match.group(1): parse_scalar(match.group(2))
        for match in INTERIOR_FACE_TIMING_VALUE_RE.finditer(line)
    }
    values["diagnostic"] = "interior_face_timing"
    return values


def convert_match(match: re.Match[str]) -> dict[str, Any]:
    return {
        name: parse_scalar(value)
        for name, value in match.groupdict().items()
        if value is not None
    }


def distribution(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def numeric_range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def sum_numeric(records: list[dict[str, Any]], key: str) -> float:
    return float(sum(
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float))
    ))


def sum_integer(records: list[dict[str, Any]], key: str) -> int:
    return int(sum(
        int(record[key])
        for record in records
        if isinstance(record.get(key), int)
    ))


def finite_min(values: list[float], default: float = 0.0) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else default


def finite_max(values: list[float], default: float = 0.0) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else default


def group_rank_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    ranks_seen: set[int] = set()
    for record in records:
        rank = record.get("rank")
        if isinstance(rank, int):
            if rank in ranks_seen and current:
                groups.append(current)
                current = []
                ranks_seen = set()
            ranks_seen.add(rank)
        elif current:
            groups.append(current)
            current = []
            ranks_seen = set()
        current.append(record)
        if not isinstance(rank, int):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def aggregate_cut_volume_assemblies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for group_index, group in enumerate(group_rank_records(records)):
        active_records = [
            record for record in group
            if float(record.get("rules", 0) or 0) > 0.0
        ]
        records_for_extrema = active_records or group
        aggregate: dict[str, Any] = {
            "diagnostic": "cut_volume_assembly_global",
            "group_index": group_index,
            "rank_records": len(group),
        }
        first = group[0]
        for key in ("marker", "side"):
            if key in first:
                aggregate[key] = first[key]
        for key in (
            "active_wet_volume",
            "cut_cell_active_wet_volume",
            "full_cell_active_wet_volume",
        ):
            aggregate[key] = sum_numeric(group, key)
        for key in (
            "rules",
            "cut_cell_rules",
            "full_cell_rules",
            "quadrature_points",
            "null_rules",
            "zero_quadrature_rules",
            "nonfinite_measure_rules",
            "negative_measure_rules",
            "nonfinite_volume_fraction_rules",
        ):
            aggregate[key] = sum_integer(group, key)
        for key in ("min_rule_measure", "min_volume_fraction", "min_exact_order"):
            values = [
                float(record[key])
                for record in records_for_extrema
                if isinstance(record.get(key), (int, float))
            ]
            aggregate[key] = finite_min(values)
        for key in ("max_rule_measure", "max_volume_fraction", "max_exact_order"):
            values = [
                float(record[key])
                for record in records_for_extrema
                if isinstance(record.get(key), (int, float))
            ]
            aggregate[key] = finite_max(values)
        if "min_exact_order" in aggregate:
            aggregate["min_exact_order"] = int(aggregate["min_exact_order"])
        if "max_exact_order" in aggregate:
            aggregate["max_exact_order"] = int(aggregate["max_exact_order"])
        groups.append(aggregate)
    return groups


def summarize_time_loop(time_loop: dict[str, Any]) -> dict[str, Any]:
    nonlinear_records = time_loop.get("nonlinear_records", [])
    accepted_steps = time_loop.get("accepted_steps", [])
    summary: dict[str, Any] = {
        "nonlinear_records": len(nonlinear_records),
        "accepted_steps": len(accepted_steps),
        "vtk_outputs": len(time_loop.get("vtk_outputs", [])),
    }
    if accepted_steps:
        final_step = accepted_steps[-1]
        summary["final_accepted_step"] = final_step.get("step")
        summary["final_accepted_time"] = final_step.get("time")
    if nonlinear_records:
        nonlinear_iterations = [
            int(record["nonlinear_iterations"])
            for record in nonlinear_records
            if isinstance(record.get("nonlinear_iterations"), int)
        ]
        linear_iterations = [
            int(record["linear_iterations"])
            for record in nonlinear_records
            if isinstance(record.get("linear_iterations"), int)
        ]
        nonlinear_residuals = [
            float(record["residual"])
            for record in nonlinear_records
            if isinstance(record.get("residual"), (int, float))
        ]
        linear_residuals = [
            float(record["linear_relative_residual"])
            for record in nonlinear_records
            if isinstance(record.get("linear_relative_residual"), (int, float))
        ]
        summary["all_nonlinear_converged"] = all(
            bool(record.get("converged")) for record in nonlinear_records
        )
        summary["all_linear_converged"] = all(
            bool(record.get("linear_converged")) for record in nonlinear_records
        )
        if nonlinear_iterations:
            summary["nonlinear_iterations_total"] = int(sum(nonlinear_iterations))
            summary["nonlinear_iterations_max"] = int(max(nonlinear_iterations))
            summary["nonlinear_iteration_distribution"] = distribution(nonlinear_iterations)
        if linear_iterations:
            summary["linear_iterations_total"] = int(sum(linear_iterations))
            summary["linear_iterations_max"] = int(max(linear_iterations))
            summary["linear_iteration_distribution"] = distribution(linear_iterations)
        nonlinear_range = numeric_range(nonlinear_residuals)
        if nonlinear_range is not None:
            summary["nonlinear_residual"] = nonlinear_range
        linear_range = numeric_range(linear_residuals)
        if linear_range is not None:
            summary["linear_relative_residual"] = linear_range
    return summary


def parse_component_norms(line: str) -> list[dict[str, Any]]:
    label_match = VECTOR_COMPONENT_LABEL_RE.search(line)
    if label_match is not None:
        line = line[label_match.end():]
    components = []
    for match in COMPONENT_NORM_RE.finditer(line):
        record = {
            "component": match.group(1),
            "norm": float(match.group(2)),
            "mean": float(match.group(3)),
        }
        if match.group(4) is not None and match.group(5) is not None:
            record["min"] = float(match.group(4))
            record["max"] = float(match.group(5))
        components.append(record)
    return components


def parse_jacobian_component_norms(line: str) -> list[dict[str, Any]]:
    if "component norms " in line:
        line = line.split("component norms ", 1)[1]
    components = []
    for match in JACOBIAN_COMPONENT_NORM_RE.finditer(line):
        components.append({
            "component": match.group(1),
            "fd": float(match.group(2)),
            "total_err": float(match.group(3)),
            "matrix_err": float(match.group(4)),
        })
    return components


def parse_jacobian_component_details(line: str) -> list[dict[str, Any]]:
    component_start = line.find(" [", line.find("diagnostic=jacobian_check_component_details"))
    if component_start >= 0:
        line = line[component_start + 1:]
    components = []
    for match in JACOBIAN_COMPONENT_DETAIL_RE.finditer(line):
        components.append({
            "component": match.group(1),
            "base": float(match.group(2)),
            "perturbed": float(match.group(3)),
            "fd": float(match.group(4)),
            "matrix": float(match.group(5)),
            "full": float(match.group(6)),
            "matrix_err": float(match.group(7)),
            "total_err": float(match.group(8)),
            "sign_flip_err": float(match.group(9)),
        })
    return components


def parse_jacobian_top_mismatch(line: str) -> list[dict[str, Any]]:
    entry_start = line.find(" [", line.find("diagnostic=jacobian_check_top_mismatch"))
    if entry_start >= 0:
        line = line[entry_start + 1:]
    entries = []
    for match in JACOBIAN_TOP_MISMATCH_RE.finditer(line):
        entries.append({
            "component": match.group(1),
            "fd": float(match.group(2)),
            "jv": float(match.group(3)),
            "err": float(match.group(4)),
        })
    return entries


def norm_key(label: str) -> str:
    key = label.strip().lower()
    key = re.sub(r"used_op=([^)]*)", r"used_op_\1", key)
    key = key.replace("*", "_")
    key = key.replace("-", "_minus_")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return f"{key}_norm" if key else "norm"


def timing_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("dg+global", "dg_global")
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def parse_norm_key_values(line: str) -> dict[str, Any]:
    values = parse_key_values(DOUBLE_BAR_VALUE_RE.sub("", line))
    for match in DOUBLE_BAR_VALUE_RE.finditer(line):
        values[norm_key(match.group(1))] = parse_scalar(match.group(2))
    return values


def vector_component_header(line: str) -> str:
    label_match = VECTOR_COMPONENT_LABEL_RE.search(line)
    if label_match is None:
        return line
    return line[:label_match.end()]


def parse_solver_diagnostics(solver_output: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "solver_controls": {},
        "cut_context_rebuilds": [],
        "cut_volume_assemblies": [],
        "hydrostatic_initializations": [],
        "pressure_gauge_checks": [],
        "residual_block_norms": [],
        "fsils_true_residuals": [],
        "fsils_solve_summaries": [],
        "fsils_blockschur_retries": [],
        "vector_component_norms": [],
        "newton_direction_checks": [],
        "jacobian_checks": [],
        "jacobian_check_component_norms": [],
        "jacobian_check_component_details": [],
        "jacobian_check_component_filters": [],
        "jacobian_check_top_mismatches": [],
        "form_block_dependencies": [],
        "form_block_installs": [],
        "form_mixed_plans": [],
        "linear_solve_histories": [],
        "assembly_timings": [],
        "interior_face_timings": [],
        "time_loop": {
            "nonlinear_records": [],
            "accepted_steps": [],
            "vtk_outputs": [],
        },
        "true_residual_failure_count": solver_output.count("true residual check failed"),
    }
    active_assembly_timing: dict[str, Any] | None = None
    for line in solver_output.splitlines():
        timing_header = ASSEMBLY_TIMING_HEADER_RE.search(line)
        if timing_header is not None:
            active_assembly_timing = {
                "rank": int(timing_header.group("rank")),
                "op": timing_header.group("op"),
            }
            continue
        if active_assembly_timing is not None:
            timing_value = ASSEMBLY_TIMING_VALUE_RE.search(line)
            if timing_value is not None:
                active_assembly_timing[timing_key(timing_value.group("label"))] = (
                    float(timing_value.group("seconds"))
                )
                continue
            if line.strip().startswith("==="):
                diagnostics["assembly_timings"].append(active_assembly_timing)
                active_assembly_timing = None
                continue

        linear_match = LINEAR_SOLVER_RE.search(line)
        time_stepping_match = TIME_STEPPING_RE.search(line)
        transient_match = TRANSIENT_SOLVE_RE.search(line)
        nonlinear_match = TIMELOOP_NONLINEAR_RE.search(line)
        accepted_match = TIMELOOP_ACCEPTED_RE.search(line)
        vtk_match = VTK_WRITE_RE.search(line)
        if linear_match is not None:
            diagnostics["solver_controls"]["linear_solver"] = convert_match(linear_match)
        elif time_stepping_match is not None:
            diagnostics["solver_controls"]["time_stepping"] = convert_match(time_stepping_match)
        elif transient_match is not None:
            diagnostics["solver_controls"]["transient_solve"] = convert_match(transient_match)
        elif nonlinear_match is not None:
            diagnostics["time_loop"]["nonlinear_records"].append(convert_match(nonlinear_match))
        elif accepted_match is not None:
            diagnostics["time_loop"]["accepted_steps"].append(convert_match(accepted_match))
        elif vtk_match is not None:
            diagnostics["time_loop"]["vtk_outputs"].append(vtk_match.group("path").strip())
        elif "Active-domain cut context" in line:
            diagnostics["cut_context_rebuilds"].append(parse_key_values(line))
        elif "cut-volume active-domain diagnostics" in line:
            diagnostics["cut_volume_assemblies"].append(parse_key_values(line))
        elif "hydrostatic pressure initialization" in line:
            diagnostics["hydrostatic_initializations"].append(parse_key_values(line))
        elif "pressure gauge diagnostic" in line:
            diagnostics["pressure_gauge_checks"].append(parse_key_values(line))
        elif "residual block norms" in line:
            diagnostics["residual_block_norms"].append(parse_key_values(line))
        elif "true residual diagnostics" in line:
            diagnostics["fsils_true_residuals"].append(parse_key_values(line))
        elif "diagnostic=fsils_solve_summary" in line:
            diagnostics["fsils_solve_summaries"].append(parse_key_values(line))
        elif "diagnostic=fsils_blockschur_true_residual_retry" in line:
            diagnostics["fsils_blockschur_retries"].append(parse_key_values(line))
        elif "NewtonSolver: direction check" in line:
            diagnostics["newton_direction_checks"].append(parse_norm_key_values(line))
        elif "NewtonSolver: Jacobian check jacobian_op=" in line:
            diagnostics["jacobian_checks"].append(parse_norm_key_values(line))
        elif "NewtonSolver: Jacobian check component norms" in line:
            diagnostics["jacobian_check_component_norms"].append({
                "components": parse_jacobian_component_norms(line),
            })
        elif "diagnostic=jacobian_check_component_details" in line:
            record = parse_key_values(line.split(" [", 1)[0])
            record["components"] = parse_jacobian_component_details(line)
            diagnostics["jacobian_check_component_details"].append(record)
        elif "diagnostic=jacobian_check_component_filter" in line:
            diagnostics["jacobian_check_component_filters"].append(parse_key_values(line))
        elif "diagnostic=jacobian_check_top_mismatch" in line:
            record = parse_key_values(line.split(" [", 1)[0])
            record["entries"] = parse_jacobian_top_mismatch(line)
            diagnostics["jacobian_check_top_mismatches"].append(record)
        elif "diagnostic=form_block_dependencies" in line:
            diagnostics["form_block_dependencies"].append(parse_key_values(line))
        elif "diagnostic=form_block_install" in line:
            diagnostics["form_block_installs"].append(parse_key_values(line))
        elif "diagnostic=form_mixed_plan" in line:
            diagnostics["form_mixed_plans"].append(parse_key_values(line))
        elif "NewtonSolver: linear solve history" in line:
            diagnostics["linear_solve_histories"].append(parse_key_values(line))
        elif "[INTERIOR_FACE_TIMING]" in line:
            diagnostics["interior_face_timings"].append(parse_interior_face_timing(line))
        elif "vector component norms" in line:
            record = parse_key_values(vector_component_header(line))
            record["components"] = parse_component_norms(line)
            diagnostics["vector_component_norms"].append(record)

    if active_assembly_timing is not None:
        diagnostics["assembly_timings"].append(active_assembly_timing)
    diagnostics["counts"] = {
        name: len(records)
        for name, records in diagnostics.items()
        if isinstance(records, list)
    }
    diagnostics["time_loop"]["summary"] = summarize_time_loop(diagnostics["time_loop"])
    diagnostics.update(parse_active_volume_history(solver_output))
    diagnostics["cut_volume_assembly_groups"] = aggregate_cut_volume_assemblies(
        diagnostics["cut_volume_assemblies"]
    )
    diagnostics["counts"]["cut_volume_assembly_groups"] = len(
        diagnostics["cut_volume_assembly_groups"]
    )
    if diagnostics["cut_volume_assembly_groups"]:
        assembly_volumes = [
            float(record["active_wet_volume"])
            for record in diagnostics["cut_volume_assembly_groups"]
            if isinstance(record.get("active_wet_volume"), (int, float))
        ]
        diagnostics["assembly_active_wet_volumes"] = assembly_volumes
        diagnostics["assembly_active_wet_volume_change"] = value_span(assembly_volumes)
    return diagnostics


def load_benchmark(case_dir: Path) -> dict[str, Any]:
    benchmark_path = case_dir / "benchmark.json"
    if not benchmark_path.exists():
        return {}
    return json.loads(benchmark_path.read_text(encoding="utf-8"))


def latest_component_record(diagnostics: dict[str, Any],
                            label: str) -> list[dict[str, Any]]:
    for record in reversed(diagnostics.get("vector_component_norms", [])):
        if record.get("label") == label:
            components = record.get("components")
            if isinstance(components, list):
                return components
    return []


def component_by_name(components: list[dict[str, Any]],
                      name: str) -> dict[str, Any] | None:
    for component in components:
        if component.get("component") == name:
            return component
    return None


def component_range(component: dict[str, Any] | None) -> float | None:
    if component is None:
        return None
    min_value = component.get("min")
    max_value = component.get("max")
    if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
        return float(max_value) - float(min_value)
    norm = component.get("norm")
    if isinstance(norm, (int, float)):
        return float(norm)
    return None


def diagnostic_solution_velocity_range(diagnostics: dict[str, Any]) -> float | None:
    components = latest_component_record(diagnostics, "solution_state")
    ranges = []
    for component in components:
        if str(component.get("component", "")).startswith("Velocity"):
            value = component_range(component)
            if value is not None:
                ranges.append(abs(value))
    if not ranges:
        return None
    return max(ranges)


def diagnostic_solution_pressure_range(diagnostics: dict[str, Any]) -> float | None:
    components = latest_component_record(diagnostics, "solution_state")
    return component_range(component_by_name(components, "Pressure"))


def diagnostic_active_volume_error(diagnostics: dict[str, Any]) -> float | None:
    context_volumes = [
        float(record["active_side_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_side_volume"), (int, float))
    ]
    assembly_volumes = [
        float(record["active_wet_volume"])
        for record in (
            diagnostics.get("cut_volume_assembly_groups")
            or diagnostics.get("cut_volume_assemblies", [])
        )
        if isinstance(record.get("active_wet_volume"), (int, float))
    ]
    if not context_volumes or not assembly_volumes:
        return None
    return max(
        min(abs(assembly_volume - context_volume) for context_volume in context_volumes)
        for assembly_volume in assembly_volumes
    )


def diagnostic_cut_volume_min_exact_order(diagnostics: dict[str, Any]) -> int | None:
    orders = [
        int(record["min_exact_order"])
        for record in (
            diagnostics.get("cut_volume_assembly_groups")
            or diagnostics.get("cut_volume_assemblies", [])
        )
        if isinstance(record.get("min_exact_order"), int)
    ]
    if not orders:
        return None
    return min(orders)


def diagnostic_cut_volume_max_exact_order(diagnostics: dict[str, Any]) -> int | None:
    orders = [
        int(record["max_exact_order"])
        for record in (
            diagnostics.get("cut_volume_assembly_groups")
            or diagnostics.get("cut_volume_assemblies", [])
        )
        if isinstance(record.get("max_exact_order"), int)
    ]
    if not orders:
        return None
    return max(orders)


def diagnostic_cut_adjacent_max_scale(diagnostics: dict[str, Any]) -> float | None:
    scales = [
        float(record["cut_adjacent_max_scale"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("cut_adjacent_max_scale"), (int, float))
    ]
    if not scales:
        return None
    return max(scales)


def diagnostic_cut_adjacent_capped_scale_count(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["cut_adjacent_capped_scale"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("cut_adjacent_capped_scale"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_active_pruned_volume_regions(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["active_pruned_volume_regions"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_pruned_volume_regions"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_active_pruned_volume(diagnostics: dict[str, Any]) -> float | None:
    volumes = [
        float(record["active_pruned_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_pruned_volume"), (int, float))
    ]
    if not volumes:
        return None
    return max(volumes)


def diagnostic_active_min_volume_fraction(diagnostics: dict[str, Any]) -> float | None:
    fractions = [
        float(record["active_min_volume_fraction"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_min_volume_fraction"), (int, float))
    ]
    if not fractions:
        return None
    return min(fractions)


def diagnostic_generated_pruned_volume_rules(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["generated_pruned_volume_rules"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("generated_pruned_volume_rules"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_generated_pruned_volume(diagnostics: dict[str, Any]) -> float | None:
    volumes = [
        float(record["generated_pruned_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("generated_pruned_volume"), (int, float))
    ]
    if not volumes:
        return None
    return max(volumes)


def diagnostic_pressure_gauge_value(diagnostics: dict[str, Any]) -> float | None:
    for record in reversed(diagnostics.get("hydrostatic_initializations", [])):
        checked = record.get("checked_gauge_constraints")
        pressure_min = record.get("gauge_pressure_min")
        pressure_max = record.get("gauge_pressure_max")
        if checked and isinstance(pressure_min, (int, float)) and isinstance(pressure_max, (int, float)):
            return 0.5 * (float(pressure_min) + float(pressure_max))
    for record in reversed(diagnostics.get("pressure_gauge_checks", [])):
        pressure_min = record.get("constraint_pressure_min")
        pressure_max = record.get("constraint_pressure_max")
        if isinstance(pressure_min, (int, float)) and isinstance(pressure_max, (int, float)):
            return 0.5 * (float(pressure_min) + float(pressure_max))
    return None


def cut_context_solution_source_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    records = diagnostics.get("cut_context_rebuilds", [])
    if not isinstance(records, list):
        return {}
    source_counts: dict[str, int] = {}
    state_refresh_count = 0
    vector_refresh_count = 0
    missing_source_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        source = record.get("solution_source")
        if isinstance(source, str) and source:
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            missing_source_count += 1
        provenance = record.get("provenance")
        if provenance in STATE_SYNC_CUT_CONTEXT_PROVENANCES:
            state_refresh_count += 1
        elif provenance in VECTOR_CUT_CONTEXT_PROVENANCES:
            vector_refresh_count += 1
    return {
        "source_counts": source_counts,
        "state_refresh_count": state_refresh_count,
        "vector_refresh_count": vector_refresh_count,
        "missing_source_count": missing_source_count,
    }


def cut_context_solution_source_errors(diagnostics: dict[str, Any]) -> list[str]:
    records = diagnostics.get("cut_context_rebuilds", [])
    if not records:
        return ["cut-context rebuild diagnostics were not reported"]

    errors = []
    missing = [
        record for record in records
        if isinstance(record, dict) and "solution_source" not in record
    ]
    if missing:
        errors.append(
            f"{len(missing)} cut-context rebuild diagnostic(s) do not report solution_source"
        )

    state_records = [
        record for record in records
        if isinstance(record, dict) and
        record.get("provenance") in STATE_SYNC_CUT_CONTEXT_PROVENANCES
    ]
    if state_records:
        bad_state = [
            record for record in state_records
            if record.get("solution_source") != "state_vector_fe_ordered"
        ]
        if bad_state:
            examples = ", ".join(
                f"{record.get('provenance', 'unknown')}:{record.get('solution_source', 'missing')}"
                for record in bad_state[:3]
            )
            errors.append(
                "Newton cut-context refreshes did not all use state_vector_fe_ordered "
                f"({examples})"
            )

    vector_records = [
        record for record in records
        if isinstance(record, dict) and
        record.get("provenance") in VECTOR_CUT_CONTEXT_PROVENANCES
    ]
    bad_vector = [
        record for record in vector_records
        if record.get("solution_source") != "fe_vector"
    ]
    if bad_vector:
        examples = ", ".join(
            f"{record.get('provenance', 'unknown')}:{record.get('solution_source', 'missing')}"
            for record in bad_vector[:3]
        )
        errors.append(
            "vector cut-context refreshes did not all use fe_vector "
            f"({examples})"
        )
    return errors


def add_diagnostic_metrics(metrics: dict[str, Any],
                           diagnostics: dict[str, Any]) -> None:
    metrics["diagnostics"] = diagnostics
    metrics["solver_controls"] = diagnostics.get("solver_controls", {})
    metrics["time_loop"] = diagnostics.get("time_loop", {})
    metrics.update(parse_active_volume_history_from_diagnostics(diagnostics))

    velocity_range = diagnostic_solution_velocity_range(diagnostics)
    if velocity_range is not None:
        metrics["diagnostic_solution_velocity_range"] = velocity_range
    pressure_range = diagnostic_solution_pressure_range(diagnostics)
    if pressure_range is not None:
        metrics["diagnostic_solution_pressure_range"] = pressure_range
    active_volume_error = diagnostic_active_volume_error(diagnostics)
    if active_volume_error is not None:
        metrics["diagnostic_active_volume_error"] = active_volume_error
    min_exact_order = diagnostic_cut_volume_min_exact_order(diagnostics)
    if min_exact_order is not None:
        metrics["diagnostic_cut_volume_min_exact_order"] = min_exact_order
    max_exact_order = diagnostic_cut_volume_max_exact_order(diagnostics)
    if max_exact_order is not None:
        metrics["diagnostic_cut_volume_max_exact_order"] = max_exact_order
    cut_adjacent_max_scale = diagnostic_cut_adjacent_max_scale(diagnostics)
    if cut_adjacent_max_scale is not None:
        metrics["diagnostic_cut_adjacent_max_scale"] = cut_adjacent_max_scale
    capped_scale_count = diagnostic_cut_adjacent_capped_scale_count(diagnostics)
    if capped_scale_count is not None:
        metrics["diagnostic_cut_adjacent_capped_scale_count"] = capped_scale_count
    pruned_volume_regions = diagnostic_active_pruned_volume_regions(diagnostics)
    if pruned_volume_regions is not None:
        metrics["diagnostic_active_pruned_volume_regions"] = pruned_volume_regions
    pruned_volume = diagnostic_active_pruned_volume(diagnostics)
    if pruned_volume is not None:
        metrics["diagnostic_active_pruned_volume"] = pruned_volume
    active_min_fraction = diagnostic_active_min_volume_fraction(diagnostics)
    if active_min_fraction is not None:
        metrics["diagnostic_active_min_volume_fraction"] = active_min_fraction
    generated_pruned_rules = diagnostic_generated_pruned_volume_rules(diagnostics)
    if generated_pruned_rules is not None:
        metrics["diagnostic_generated_pruned_volume_rules"] = generated_pruned_rules
    generated_pruned_volume = diagnostic_generated_pruned_volume(diagnostics)
    if generated_pruned_volume is not None:
        metrics["diagnostic_generated_pruned_volume"] = generated_pruned_volume
    gauge_value = diagnostic_pressure_gauge_value(diagnostics)
    if gauge_value is not None:
        metrics["diagnostic_pressure_gauge_value"] = gauge_value
    solution_source_summary = cut_context_solution_source_summary(diagnostics)
    if solution_source_summary:
        metrics["diagnostic_cut_context_solution_sources"] = solution_source_summary
    if diagnostics.get("fsils_true_residuals"):
        latest_true_residual = diagnostics["fsils_true_residuals"][-1]
        metrics["latest_fsils_true_residual"] = latest_true_residual
        for name in (
            "constraint_solution_mean",
            "constraint_solution_rms",
            "constraint_solution_fluctuation_rms",
            "constraint_solution_mean_dominance",
            "constraint_residual_mean",
            "constraint_residual_rms",
        ):
            value = latest_true_residual.get(name)
            if isinstance(value, (int, float)):
                metrics[f"latest_fsils_{name}"] = value
    if diagnostics.get("fsils_solve_summaries"):
        latest_solve_summary = diagnostics["fsils_solve_summaries"][-1]
        metrics["latest_fsils_solve_summary"] = latest_solve_summary
        for name in (
            "blockschur_schur_iterations",
            "blockschur_schur_mitr",
            "blockschur_schur_rel_tol",
            "blockschur_schur_abs_tol",
            "blockschur_momentum_iterations",
            "blockschur_momentum_mitr",
            "internal_final_norm",
            "internal_success",
            "true_residual_retries",
        ):
            value = latest_solve_summary.get(name)
            if isinstance(value, (int, float)):
                metrics[f"latest_fsils_{name}"] = value
    if diagnostics.get("newton_direction_checks"):
        latest_direction_check = diagnostics["newton_direction_checks"][-1]
        metrics["latest_newton_direction_check"] = latest_direction_check
        value = latest_direction_check.get("rel")
        if isinstance(value, (int, float)):
            metrics["diagnostic_newton_direction_relative_error"] = float(value)
    if diagnostics.get("jacobian_checks"):
        latest_jacobian_check = diagnostics["jacobian_checks"][-1]
        metrics["latest_jacobian_check"] = latest_jacobian_check
        value = latest_jacobian_check.get("rel")
        if isinstance(value, (int, float)):
            metrics["diagnostic_jacobian_check_relative_error"] = float(value)
    if diagnostics.get("jacobian_check_component_details"):
        metrics["latest_jacobian_check_component_details"] = (
            diagnostics["jacobian_check_component_details"][-1]
        )
    if diagnostics.get("jacobian_check_top_mismatches"):
        metrics["latest_jacobian_check_top_mismatch"] = (
            diagnostics["jacobian_check_top_mismatches"][-1]
        )
    if diagnostics.get("form_mixed_plans"):
        metrics["latest_form_mixed_plan"] = diagnostics["form_mixed_plans"][-1]
    if diagnostics.get("form_block_installs"):
        metrics["form_block_install_count"] = len(diagnostics["form_block_installs"])
    if diagnostics.get("linear_solve_histories"):
        metrics["latest_linear_solve_history"] = diagnostics["linear_solve_histories"][-1]
    if diagnostics.get("assembly_timings"):
        timings = diagnostics["assembly_timings"]
        metrics["latest_assembly_timing"] = timings[-1]
        for name in (
            "total",
            "cell_terms",
            "boundary_terms",
            "other_dg_global",
            "interior_faces",
            "interface_faces",
            "cut_volumes",
            "global_terms",
        ):
            values = [
                float(record[name])
                for record in timings
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_assembly_timing_max_{name}_seconds"] = max(values)
    if diagnostics.get("interior_face_timings"):
        timings = diagnostics["interior_face_timings"]
        metrics["latest_interior_face_timing"] = timings[-1]
        for name in (
            "faces_considered",
            "faces_assembled",
        ):
            values = [
                int(record[name])
                for record in timings
                if isinstance(record.get(name), int)
            ]
            if values:
                metrics[f"diagnostic_interior_face_timing_max_{name}"] = max(values)
        for name in (
            "total",
            "setup",
            "filter",
            "dofs",
            "local_face",
            "align",
            "prepare_minus",
            "prepare_plus",
            "ctx",
            "cut_scale",
            "solution",
            "field",
            "material",
            "kernel",
            "orient",
            "insert",
        ):
            values = [
                float(record[name])
                for record in timings
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_interior_face_timing_max_{name}_seconds"] = max(values)
    retry_counts = [
        int(record["true_residual_retries"])
        for record in diagnostics.get("fsils_solve_summaries", [])
        if isinstance(record.get("true_residual_retries"), int)
    ]
    retry_counts.extend(
        1 for record in diagnostics.get("fsils_blockschur_retries", [])
        if record.get("diagnostic") == "fsils_blockschur_true_residual_retry"
    )
    if retry_counts:
        metrics["diagnostic_blockschur_true_residual_retries"] = max(retry_counts)

    wet_fraction_volume = metrics.get("wet_fraction_volume")
    context_volumes = metrics.get("cut_context_active_side_volumes", [])
    if isinstance(wet_fraction_volume, (int, float)) and context_volumes:
        metrics["wet_fraction_volume_drift_vs_initial_cut_context"] = (
            float(wet_fraction_volume) - float(context_volumes[0])
        )


def add_solver_control_overrides(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> None:
    for name in (
        "linear_relative_tolerance",
        "linear_absolute_tolerance",
        "linear_max_iterations",
        "ns_gm_max_iterations",
        "ns_cg_max_iterations",
        "ns_gm_tolerance",
        "ns_cg_tolerance",
        "linear_solver_type",
    ):
        value = getattr(args, name)
        if value is not None:
            metrics[name] = value
    for name in (
        "enable_jacobian_check",
        "enable_newton_direction_check",
        "enable_linear_solve_history",
        "enable_linear_solve_component_norms",
        "enable_form_block_diagnostics",
        "enable_interior_face_timing",
        "require_cut_context_solution_source_diagnostics",
        "require_assembly_timing_diagnostics",
        "require_interior_face_timing_diagnostics",
        "allow_failure_diagnostics",
    ):
        if getattr(args, name):
            metrics[name] = True
    for name in (
        "jacobian_check_iteration",
        "jacobian_check_step",
        "jacobian_check_components",
        "linear_solve_history_max_calls",
        "linear_solve_component_norms_max_newton_it",
    ):
        value = getattr(args, name)
        if value is not None:
            metrics[name] = value


def parse_active_volume_history_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "cut_context_active_side_volumes",
        "assembly_active_wet_volumes",
        "cut_context_active_side_volume_change",
        "assembly_active_wet_volume_change",
    ]
    return {
        key: diagnostics[key]
        for key in keys
        if key in diagnostics
    }


def previous_invalid_pressure(benchmark: dict[str, Any]) -> float | None:
    verification = benchmark.get("pressure_gauge_verification")
    if not isinstance(verification, dict):
        return None
    value = verification.get("previous_invalid_d18_full_volume_hydrostatic_pressure")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def diagnostic_timeout_metrics(case_name: str,
                               run_dir: Path,
                               diagnostics: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "case": case_name,
        "run_dir": str(run_dir),
        "timed_out": True,
    }
    add_diagnostic_metrics(metrics, diagnostics)

    previous = previous_invalid_pressure(load_benchmark(run_dir))
    if previous is not None:
        metrics["pressure_gauge_previous_invalid"] = previous
        gauge_value = metrics.get("diagnostic_pressure_gauge_value")
        if gauge_value is not None:
            metrics["diagnostic_pressure_gauge_previous_invalid_difference"] = (
                gauge_value - previous
            )
    return metrics


def evaluate_timeout_diagnostics(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics["diagnostics"]
    gauge_required = metrics.get("case") in {"d18", "d38", "mini2d", "static2d"}
    if not diagnostics.get("cut_context_rebuilds"):
        errors.append("cut-context rebuild diagnostics were not reported")
    if not diagnostics.get("cut_volume_assemblies"):
        errors.append("cut-volume assembly diagnostics were not reported")
    if gauge_required and not diagnostics.get("pressure_gauge_checks"):
        errors.append("pressure gauge diagnostics were not reported")
    if gauge_required and not diagnostics.get("hydrostatic_initializations"):
        errors.append("hydrostatic initialization diagnostics were not reported")
    if not latest_component_record(diagnostics, "solution_state"):
        errors.append("solution-state component diagnostics were not reported")
    if (diagnostics.get("true_residual_failure_count", 0) > 0 and
            not diagnostics.get("fsils_true_residuals")):
        errors.append("FSILS true-residual diagnostics were not reported")
    if args.require_newton_direction_check_diagnostics and not diagnostics.get("newton_direction_checks"):
        errors.append("Newton direction-check diagnostics were not reported")
    if args.require_jacobian_check_diagnostics and not diagnostics.get("jacobian_checks"):
        errors.append("Jacobian finite-difference diagnostics were not reported")
    if args.require_jacobian_top_mismatch_diagnostics and not diagnostics.get("jacobian_check_top_mismatches"):
        errors.append("Jacobian top-mismatch diagnostics were not reported")
    if args.require_linear_solve_history_diagnostics and not diagnostics.get("linear_solve_histories"):
        errors.append("linear solve history diagnostics were not reported")
    if args.require_form_block_diagnostics and (
            not diagnostics.get("form_block_installs") or not diagnostics.get("form_mixed_plans")):
        errors.append("form block installation diagnostics were not reported")
    if args.require_cut_context_solution_source_diagnostics:
        errors.extend(cut_context_solution_source_errors(diagnostics))
    if args.require_assembly_timing_diagnostics and not diagnostics.get("assembly_timings"):
        errors.append("assembly timing diagnostics were not reported")
    if args.require_interior_face_timing_diagnostics and not diagnostics.get("interior_face_timings"):
        errors.append("interior-face timing diagnostics were not reported")

    if args.min_diagnostic_solution_velocity_range is not None:
        velocity_range = metrics.get("diagnostic_solution_velocity_range")
        if not isinstance(velocity_range, (int, float)):
            errors.append("diagnostic solution velocity range is unavailable")
        elif velocity_range < args.min_diagnostic_solution_velocity_range:
            errors.append(
                f"diagnostic solution velocity range {velocity_range:.6g} is below "
                f"{args.min_diagnostic_solution_velocity_range:.6g}"
            )
    if args.min_diagnostic_pressure_range is not None:
        pressure_range = metrics.get("diagnostic_solution_pressure_range")
        if not isinstance(pressure_range, (int, float)):
            errors.append("diagnostic solution pressure range is unavailable")
        elif pressure_range < args.min_diagnostic_pressure_range:
            errors.append(
                f"diagnostic solution pressure range {pressure_range:.6g} is below "
                f"{args.min_diagnostic_pressure_range:.6g}"
            )
    if args.max_diagnostic_active_volume_error is not None:
        volume_error = metrics.get("diagnostic_active_volume_error")
        if not isinstance(volume_error, (int, float)):
            errors.append("diagnostic active-volume consistency error is unavailable")
        elif volume_error > args.max_diagnostic_active_volume_error:
            errors.append(
                f"diagnostic active-volume error {volume_error:.6g} exceeds "
                f"{args.max_diagnostic_active_volume_error:.6g}"
            )
    if args.min_diagnostic_cut_volume_exact_order is not None:
        exact_order = metrics.get("diagnostic_cut_volume_min_exact_order")
        if not isinstance(exact_order, int):
            errors.append("diagnostic cut-volume exact order is unavailable")
        elif exact_order < args.min_diagnostic_cut_volume_exact_order:
            errors.append(
                f"diagnostic cut-volume exact order {exact_order} is below "
                f"{args.min_diagnostic_cut_volume_exact_order}"
            )
    if args.min_diagnostic_cut_volume_max_exact_order is not None:
        exact_order = metrics.get("diagnostic_cut_volume_max_exact_order")
        if not isinstance(exact_order, int):
            errors.append("diagnostic cut-volume max exact order is unavailable")
        elif exact_order < args.min_diagnostic_cut_volume_max_exact_order:
            errors.append(
                f"diagnostic cut-volume max exact order {exact_order} is below "
                f"{args.min_diagnostic_cut_volume_max_exact_order}"
            )
    if args.max_diagnostic_cut_adjacent_scale is not None:
        max_scale = metrics.get("diagnostic_cut_adjacent_max_scale")
        if not isinstance(max_scale, (int, float)):
            errors.append("diagnostic cut-adjacent max scale is unavailable")
        elif max_scale > args.max_diagnostic_cut_adjacent_scale:
            errors.append(
                f"diagnostic cut-adjacent max scale {max_scale:.6g} exceeds "
                f"{args.max_diagnostic_cut_adjacent_scale:.6g}"
            )
    if args.min_diagnostic_cut_adjacent_capped_scale_count is not None:
        capped_count = metrics.get("diagnostic_cut_adjacent_capped_scale_count")
        if not isinstance(capped_count, int):
            errors.append("diagnostic cut-adjacent capped scale count is unavailable")
        elif capped_count < args.min_diagnostic_cut_adjacent_capped_scale_count:
            errors.append(
                f"diagnostic cut-adjacent capped scale count {capped_count} is below "
                f"{args.min_diagnostic_cut_adjacent_capped_scale_count}"
            )
    if args.min_diagnostic_active_pruned_volume_regions is not None:
        pruned_count = metrics.get("diagnostic_active_pruned_volume_regions")
        if not isinstance(pruned_count, int):
            errors.append("diagnostic active pruned volume-region count is unavailable")
        elif pruned_count < args.min_diagnostic_active_pruned_volume_regions:
            errors.append(
                f"diagnostic active pruned volume-region count {pruned_count} is below "
                f"{args.min_diagnostic_active_pruned_volume_regions}"
            )
    if args.min_diagnostic_active_min_volume_fraction is not None:
        min_fraction = metrics.get("diagnostic_active_min_volume_fraction")
        if not isinstance(min_fraction, (int, float)):
            errors.append("diagnostic active min volume fraction is unavailable")
        elif min_fraction < args.min_diagnostic_active_min_volume_fraction:
            errors.append(
                f"diagnostic active min volume fraction {min_fraction:.6g} is below "
                f"{args.min_diagnostic_active_min_volume_fraction:.6g}"
            )
    if args.min_diagnostic_generated_pruned_volume_rules is not None:
        pruned_rules = metrics.get("diagnostic_generated_pruned_volume_rules")
        if not isinstance(pruned_rules, int):
            errors.append("diagnostic generated pruned volume-rule count is unavailable")
        elif pruned_rules < args.min_diagnostic_generated_pruned_volume_rules:
            errors.append(
                f"diagnostic generated pruned volume-rule count {pruned_rules} is below "
                f"{args.min_diagnostic_generated_pruned_volume_rules}"
            )
    if args.min_diagnostic_blockschur_true_residual_retries is not None:
        retries = metrics.get("diagnostic_blockschur_true_residual_retries")
        if not isinstance(retries, int):
            errors.append("diagnostic BlockSchur true-residual retry count is unavailable")
        elif retries < args.min_diagnostic_blockschur_true_residual_retries:
            errors.append(
                f"diagnostic BlockSchur true-residual retry count {retries} is below "
                f"{args.min_diagnostic_blockschur_true_residual_retries}"
            )
    if args.max_newton_direction_relative_error is not None:
        value = metrics.get("diagnostic_newton_direction_relative_error")
        if not isinstance(value, (int, float)):
            errors.append("Newton direction-check relative error is unavailable")
        elif value > args.max_newton_direction_relative_error:
            errors.append(
                f"Newton direction-check relative error {value:.6g} exceeds "
                f"{args.max_newton_direction_relative_error:.6g}"
            )
    if args.max_jacobian_check_relative_error is not None:
        value = metrics.get("diagnostic_jacobian_check_relative_error")
        if not isinstance(value, (int, float)):
            errors.append("Jacobian finite-difference relative error is unavailable")
        elif value > args.max_jacobian_check_relative_error:
            errors.append(
                f"Jacobian finite-difference relative error {value:.6g} exceeds "
                f"{args.max_jacobian_check_relative_error:.6g}"
            )
    if args.stale_pressure_gauge_tolerance is not None:
        stale_difference = metrics.get("diagnostic_pressure_gauge_previous_invalid_difference")
        if not isinstance(stale_difference, (int, float)):
            errors.append("diagnostic pressure gauge stale-value difference is unavailable")
        elif abs(float(stale_difference)) <= args.stale_pressure_gauge_tolerance:
            errors.append(
                "diagnostic pressure gauge remains close to the previous "
                "full-volume hydrostatic value"
            )
    return errors


def pressure_gauge_metrics(output: pv.DataSet, benchmark: dict[str, Any]) -> dict[str, Any]:
    gauge = benchmark.get("pressure_gauge")
    if not isinstance(gauge, dict) or "Pressure" not in output.point_data:
        return {}
    node_id = gauge.get("node_id")
    if node_id is None:
        return {}

    gids = None
    for name in ("GlobalNodeID", "GlobalVertexID"):
        if name in output.point_data:
            gids = np.asarray(output.point_data[name], dtype=np.int64).reshape(-1)
            break
    if gids is None:
        return {"pressure_gauge_found": False}

    indices = np.flatnonzero(gids == int(node_id))
    if indices.size == 0:
        return {"pressure_gauge_found": False}

    pressure = np.asarray(output.point_data["Pressure"], dtype=float).reshape(-1)
    value = float(pressure[indices[0]])
    metrics: dict[str, Any] = {
        "pressure_gauge_found": True,
        "pressure_gauge_node_id": int(node_id),
        "pressure_gauge_value": value,
        "pressure_gauge_matches": int(indices.size),
    }
    expected = gauge.get("expected_initial_hydrostatic_pressure")
    if isinstance(expected, (int, float)):
        metrics["pressure_gauge_expected_initial"] = float(expected)
        metrics["pressure_gauge_initial_error"] = value - float(expected)

    verification = benchmark.get("pressure_gauge_verification")
    if isinstance(verification, dict):
        stale = verification.get("previous_invalid_d18_full_volume_hydrostatic_pressure")
        if isinstance(stale, (int, float)):
            metrics["pressure_gauge_previous_invalid"] = float(stale)
            metrics["pressure_gauge_previous_invalid_difference"] = value - float(stale)
    return metrics


def compute_metrics(case_name: str, case_dir: Path, result: Path) -> dict[str, Any]:
    benchmark = load_benchmark(case_dir)
    if benchmark:
        dimensions = benchmark.get("dimensions_m", {})
        gate_x = float(dimensions.get("profile_window_x_min", CASE_GATE_X.get(case_name, 0.4)))
    else:
        gate_x = CASE_GATE_X[case_name]

    initial = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    output = pv.read(result)
    output_index = result_indices_by_initial_gid(initial, output)

    points = np.asarray(initial.points, dtype=float)
    phi0 = point_array(initial, "phi").astype(float)
    velocity = point_array(output, "Velocity").astype(float)[output_index]
    speed = np.linalg.norm(velocity, axis=1)

    wet0 = phi0 < 0.0
    gate_half_width = 0.025 if case_name != "mini2d" else 0.15
    gate_region = (np.abs(points[:, 0] - gate_x) <= gate_half_width) & wet0
    front_region = (points[:, 0] >= gate_x - 0.03) & (points[:, 0] <= gate_x + 0.07) & wet0
    if case_name == "mini2d":
        front_region = (points[:, 0] >= gate_x) & (points[:, 0] <= gate_x + 0.3) & wet0

    wet_speed = speed[wet0]
    wet_velocity = velocity[wet0]

    metrics: dict[str, Any] = {
        "result": str(result),
        "max_speed": float(np.nanmax(wet_speed)),
        "wet_mean_speed": float(np.nanmean(wet_speed)),
        "wet_mean_velocity": [float(value) for value in np.nanmean(wet_velocity, axis=0)],
        "gate_mean_velocity": [float(value) for value in np.mean(velocity[gate_region], axis=0)],
        "front_mean_velocity": [float(value) for value in np.mean(velocity[front_region], axis=0)],
        "finite_velocity": bool(np.isfinite(wet_velocity).all()),
        "wet_nodes": int(np.count_nonzero(wet0)),
        "gate_nodes": int(np.count_nonzero(gate_region)),
        "front_nodes": int(np.count_nonzero(front_region)),
    }
    if "Pressure" in output.point_data:
        pressure = np.asarray(output.point_data["Pressure"], dtype=float).reshape(-1)
        pressure_min = float(np.nanmin(pressure))
        pressure_max = float(np.nanmax(pressure))
        metrics["pressure_min"] = pressure_min
        metrics["pressure_max"] = pressure_max
        metrics["pressure_range"] = pressure_max - pressure_min
        metrics["pressure_mean"] = float(np.nanmean(pressure))
    if "Velocity" in output.point_data:
        output_velocity = np.asarray(output.point_data["Velocity"], dtype=float)
        if output_velocity.ndim == 1:
            output_velocity = output_velocity.reshape((-1, 1))
        output_speed = np.linalg.norm(output_velocity, axis=1)
        metrics["velocity_max"] = float(np.nanmax(output_speed))
        metrics["velocity_mean"] = float(np.nanmean(output_speed))
        component_ranges = []
        for component in range(output_velocity.shape[1]):
            values = output_velocity[:, component]
            component_ranges.append(float(np.nanmax(values) - np.nanmin(values)))
        metrics["velocity_component_ranges"] = component_ranges
        metrics["velocity_range"] = float(max(component_ranges)) if component_ranges else 0.0
    if "phi" in output.point_data:
        try:
            interface = output.contour(isosurfaces=[0.0], scalars="phi")
            interface_points = np.asarray(interface.points, dtype=float)
            metrics["interface_points"] = int(interface.n_points)
            if interface_points.size:
                metrics["interface_peak_height"] = float(np.nanmax(interface_points[:, 1]))
                metrics["interface_front_x"] = float(np.nanmax(interface_points[:, 0]))
        except Exception as exc:
            metrics["interface_extraction_error"] = str(exc)
    if "WetVolumeFraction" in output.cell_data:
        fractions = np.asarray(output.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
        measures = cell_measure(output)
        if fractions.shape[0] == measures.shape[0]:
            metrics["wet_fraction_cell_count"] = int(fractions.shape[0])
            metrics["wet_fraction_volume"] = float(np.sum(fractions * measures))
            metrics["wet_fraction_min"] = float(np.min(fractions))
            metrics["wet_fraction_max"] = float(np.max(fractions))
    metrics.update(pressure_gauge_metrics(output, benchmark))
    return metrics


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = []
    if metrics.get("output_metrics_skipped"):
        return evaluate_timeout_diagnostics(metrics, args)
    if args.require_cut_context_solution_source_diagnostics:
        errors.extend(cut_context_solution_source_errors(metrics["diagnostics"]))
    if args.require_assembly_timing_diagnostics and not metrics["diagnostics"].get("assembly_timings"):
        errors.append("assembly timing diagnostics were not reported")
    if (args.require_interior_face_timing_diagnostics and
            not metrics["diagnostics"].get("interior_face_timings")):
        errors.append("interior-face timing diagnostics were not reported")
    if not metrics["finite_velocity"]:
        errors.append("Velocity contains non-finite values")
    if metrics.get("case") == "static2d":
        if metrics["max_speed"] > args.max_static_speed:
            errors.append(
                f"static max speed {metrics['max_speed']:.6g} exceeds "
                f"{args.max_static_speed:.6g}"
            )
        return errors
    if metrics["max_speed"] < args.min_max_speed:
        errors.append(
            f"max speed {metrics['max_speed']:.6g} is below {args.min_max_speed:.6g}"
        )
    if metrics["wet_mean_speed"] < args.min_wet_mean_speed:
        errors.append(
            f"wet mean speed {metrics['wet_mean_speed']:.6g} is below "
            f"{args.min_wet_mean_speed:.6g}"
        )
    if metrics["gate_mean_velocity"][0] < args.min_gate_mean_ux:
        errors.append(
            f"gate mean ux {metrics['gate_mean_velocity'][0]:.6g} is below "
            f"{args.min_gate_mean_ux:.6g}"
        )
    if metrics["front_mean_velocity"][0] < args.min_front_mean_ux:
        errors.append(
            f"front mean ux {metrics['front_mean_velocity'][0]:.6g} is below "
            f"{args.min_front_mean_ux:.6g}"
        )
    if args.min_active_volume_change > 0.0:
        volume_change = metrics.get("assembly_active_wet_volume_change", 0.0)
        volume_count = len(metrics.get("assembly_active_wet_volumes", []))
        if volume_count < 2:
            errors.append("assembly active wet volume was not reported at least twice")
        elif volume_change < args.min_active_volume_change:
            errors.append(
                f"assembly active wet-volume change {volume_change:.6g} is below "
                f"{args.min_active_volume_change:.6g}"
            )
    if args.stale_pressure_gauge_tolerance is not None:
        if not metrics.get("pressure_gauge_found", False):
            errors.append("pressure gauge was not found in the solver output")
        else:
            stale_difference = metrics.get("pressure_gauge_previous_invalid_difference")
            if not isinstance(stale_difference, (int, float)):
                errors.append("previous invalid pressure gauge value is unavailable")
            elif abs(float(stale_difference)) <= args.stale_pressure_gauge_tolerance:
                errors.append(
                    "pressure gauge remains close to the previous full-volume "
                    "hydrostatic value"
                )
    if args.max_wet_fraction_volume_error is not None:
        wet_fraction_volume = metrics.get("wet_fraction_volume")
        context_volumes = metrics.get("cut_context_active_side_volumes", [])
        if not isinstance(wet_fraction_volume, (int, float)):
            errors.append("WetVolumeFraction output volume is unavailable")
        elif not context_volumes:
            errors.append("cut-context active-side volume was not reported")
        else:
            error = abs(float(wet_fraction_volume) - float(context_volumes[-1]))
            metrics["wet_fraction_volume_error_vs_last_cut_context"] = error
            if error > args.max_wet_fraction_volume_error:
                errors.append(
                    f"WetVolumeFraction volume error {error:.6g} exceeds "
                    f"{args.max_wet_fraction_volume_error:.6g}"
                )
    return errors


def write_qualification_log(path: Path | None,
                            solver: Path,
                            probes: list[dict[str, Any]],
                            complete: bool) -> None:
    if path is None:
        return
    payload = {
        "solver": str(solver),
        "complete": complete,
        "probes": probes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_case(case_name: str, solver: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = CASES[case_name]
    if source is not None and not source.exists():
        raise FileNotFoundError(source)

    temp_context = None
    if args.preserve_run_dir:
        temp_name = tempfile.mkdtemp(prefix=f"dam_break_{case_name}_")
    else:
        temp_context = tempfile.TemporaryDirectory(prefix=f"dam_break_{case_name}_")
        temp_name = temp_context.name

    def write_solver_log(run_dir: Path, output: str) -> None:
        (run_dir / "solver_run.log").write_text(output, encoding="utf-8")

    def write_failure(failure: dict[str, Any]) -> None:
        failure.setdefault("passed", False)
        write_qualification_log(args.qualification_log, solver, [failure], complete=False)

    try:
        run_dir = Path(temp_name) / case_name
        if source is None:
            write_mini_case(run_dir, args.steps, static=(case_name == "static2d"))
        else:
            run_dir = Path(temp_name) / source.name
            copy_case(source, run_dir, args.source_ref)
            configure_solver(
                run_dir / "solver.xml",
                args.steps,
                time_step_size=args.time_step_size,
                disable_cut_stabilization=args.disable_cut_stabilization,
                max_nonlinear_iterations=args.max_nonlinear_iterations,
                linear_relative_tolerance=args.linear_relative_tolerance,
                linear_absolute_tolerance=args.linear_absolute_tolerance,
                linear_max_iterations=args.linear_max_iterations,
                ns_gm_max_iterations=args.ns_gm_max_iterations,
                ns_cg_max_iterations=args.ns_cg_max_iterations,
                ns_gm_tolerance=args.ns_gm_tolerance,
                ns_cg_tolerance=args.ns_cg_tolerance,
                linear_solver_type=args.linear_solver_type,
                disable_coupled_outer_fgmres=args.disable_coupled_outer_fgmres,
                disable_cut_metadata_scale=args.disable_cut_metadata_scale,
                disable_vtk_output=args.disable_vtk_output,
                final_output_only=args.final_output_only,
                vtk_save_increment=args.vtk_save_increment,
                start_saving_after_step=args.start_saving_after_step,
            )

        try:
            command = solver_command(solver, args)
            completed = subprocess.run(
                command,
                cwd=run_dir,
                env=solver_environment(args),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            write_solver_log(run_dir, output)
            tail = "\n".join(output.splitlines()[-80:])
            diagnostics = parse_solver_diagnostics(output)
            failure = diagnostic_timeout_metrics(case_name, run_dir, diagnostics)
            failure["timeout_seconds"] = args.timeout_seconds
            failure["command"] = command
            failure["stdout_tail"] = tail
            diagnostic_errors = evaluate_timeout_diagnostics(failure, args)
            failure["diagnostic_errors"] = diagnostic_errors
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            if args.disable_cut_metadata_scale:
                failure["disable_cut_metadata_scale"] = True
            if args.disable_vtk_output:
                failure["disable_vtk_output"] = True
            if args.enable_blockschur_true_residual_retry:
                failure["enable_blockschur_true_residual_retry"] = True
            add_solver_control_overrides(failure, args)
            if args.allow_timeout_diagnostics and not diagnostic_errors:
                failure["passed"] = True
                failure["errors"] = []
                return failure
            failure["errors"] = diagnostic_errors or ["solver timed out"]
            write_failure(failure)
            raise RuntimeError(json.dumps(failure, indent=2, sort_keys=True)) from exc
        write_solver_log(run_dir, completed.stdout)
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            failure = {
                "case": case_name,
                "run_dir": str(run_dir),
                "command": solver_command(solver, args),
                "returncode": completed.returncode,
                "diagnostics": parse_solver_diagnostics(completed.stdout),
                "stdout_tail": tail,
            }
            add_diagnostic_metrics(failure, failure["diagnostics"])
            previous = previous_invalid_pressure(load_benchmark(run_dir))
            if previous is not None:
                failure["pressure_gauge_previous_invalid"] = previous
                gauge_value = failure.get("diagnostic_pressure_gauge_value")
                if gauge_value is not None:
                    failure["diagnostic_pressure_gauge_previous_invalid_difference"] = (
                        gauge_value - previous
                    )
            diagnostic_errors = evaluate_timeout_diagnostics(failure, args)
            failure["diagnostic_errors"] = diagnostic_errors
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            if args.disable_cut_metadata_scale:
                failure["disable_cut_metadata_scale"] = True
            if args.disable_vtk_output:
                failure["disable_vtk_output"] = True
            if args.enable_blockschur_true_residual_retry:
                failure["enable_blockschur_true_residual_retry"] = True
            add_solver_control_overrides(failure, args)
            failure["errors"] = (
                diagnostic_errors
                or [f"solver exited with return code {completed.returncode}"]
            )
            if args.allow_failure_diagnostics and not diagnostic_errors:
                failure["passed"] = True
                failure["errors"] = []
                return failure
            write_failure(failure)
            raise RuntimeError(json.dumps(failure, indent=2, sort_keys=True))

        diagnostics = parse_solver_diagnostics(completed.stdout)
        if args.disable_vtk_output:
            metrics = {
                "output_metrics_skipped": True,
                "output_metrics_skip_reason": "VTK output disabled",
            }
        else:
            metrics = compute_metrics(case_name, run_dir, result_path(run_dir, args.steps))
        add_diagnostic_metrics(metrics, diagnostics)
        metrics["case"] = case_name
        metrics["command"] = solver_command(solver, args)
        metrics["run_dir"] = str(run_dir)
        metrics["steps"] = args.steps
        if args.time_step_size is not None:
            metrics["time_step_size"] = args.time_step_size
        if args.disable_coupled_outer_fgmres:
            metrics["disable_coupled_outer_fgmres"] = True
        if args.disable_cut_metadata_scale:
            metrics["disable_cut_metadata_scale"] = True
        if args.disable_vtk_output:
            metrics["disable_vtk_output"] = True
        if args.enable_blockschur_true_residual_retry:
            metrics["enable_blockschur_true_residual_retry"] = True
        if args.final_output_only:
            metrics["final_output_only"] = True
        add_solver_control_overrides(metrics, args)
        errors = evaluate(metrics, args)
        metrics["passed"] = not errors
        metrics["errors"] = errors
        if errors:
            write_failure(metrics)
            raise RuntimeError(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--mpiexec", type=Path, default=Path("mpiexec"))
    parser.add_argument("--mpi-ranks", type=int)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--source-ref")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--time-step-size", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--preserve-run-dir", action="store_true")
    parser.add_argument("--qualification-log", type=Path)
    parser.add_argument("--disable-vtk-output", action="store_true")
    parser.add_argument("--final-output-only", action="store_true")
    parser.add_argument("--vtk-save-increment", type=int)
    parser.add_argument("--start-saving-after-step", type=int)
    parser.add_argument("--min-max-speed", type=float, default=1.0e-2)
    parser.add_argument("--min-wet-mean-speed", type=float, default=2.5e-4)
    parser.add_argument("--min-gate-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-front-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-active-volume-change", type=float, default=0.0)
    parser.add_argument("--max-static-speed", type=float, default=1.0e-9)
    parser.add_argument("--stale-pressure-gauge-tolerance", type=float)
    parser.add_argument("--max-wet-fraction-volume-error", type=float)
    parser.add_argument("--allow-timeout-diagnostics", action="store_true")
    parser.add_argument("--allow-failure-diagnostics", action="store_true")
    parser.add_argument("--min-diagnostic-solution-velocity-range", type=float)
    parser.add_argument("--min-diagnostic-pressure-range", type=float)
    parser.add_argument("--max-diagnostic-active-volume-error", type=float)
    parser.add_argument("--min-diagnostic-cut-volume-exact-order", type=int)
    parser.add_argument("--min-diagnostic-cut-volume-max-exact-order", type=int)
    parser.add_argument("--max-diagnostic-cut-adjacent-scale", type=float)
    parser.add_argument("--min-diagnostic-cut-adjacent-capped-scale-count", type=int)
    parser.add_argument("--min-diagnostic-active-pruned-volume-regions", type=int)
    parser.add_argument("--min-diagnostic-active-min-volume-fraction", type=float)
    parser.add_argument("--min-diagnostic-generated-pruned-volume-rules", type=int)
    parser.add_argument("--min-diagnostic-blockschur-true-residual-retries", type=int)
    parser.add_argument("--require-newton-direction-check-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-check-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-top-mismatch-diagnostics", action="store_true")
    parser.add_argument("--require-linear-solve-history-diagnostics", action="store_true")
    parser.add_argument("--require-form-block-diagnostics", action="store_true")
    parser.add_argument("--require-cut-context-solution-source-diagnostics", action="store_true")
    parser.add_argument("--require-assembly-timing-diagnostics", action="store_true")
    parser.add_argument("--max-newton-direction-relative-error", type=float)
    parser.add_argument("--max-jacobian-check-relative-error", type=float)
    parser.add_argument("--disable-cut-stabilization", action="store_true")
    parser.add_argument("--disable-cut-metadata-scale", action="store_true")
    parser.add_argument("--max-nonlinear-iterations", type=int)
    parser.add_argument("--linear-relative-tolerance", type=float)
    parser.add_argument("--linear-absolute-tolerance", type=float)
    parser.add_argument("--linear-max-iterations", type=int)
    parser.add_argument("--ns-gm-max-iterations", type=int)
    parser.add_argument("--ns-cg-max-iterations", type=int)
    parser.add_argument("--ns-gm-tolerance", type=float)
    parser.add_argument("--ns-cg-tolerance", type=float)
    parser.add_argument("--linear-solver-type")
    parser.add_argument("--disable-coupled-outer-fgmres", action="store_true")
    parser.add_argument("--enable-blockschur-true-residual-retry", action="store_true")
    parser.add_argument("--enable-jacobian-check", action="store_true")
    parser.add_argument("--jacobian-check-iteration", type=int)
    parser.add_argument("--jacobian-check-step", type=float)
    parser.add_argument("--jacobian-check-components")
    parser.add_argument("--enable-newton-direction-check", action="store_true")
    parser.add_argument("--enable-linear-solve-history", action="store_true")
    parser.add_argument("--linear-solve-history-max-calls", type=int)
    parser.add_argument("--enable-linear-solve-component-norms", action="store_true")
    parser.add_argument("--linear-solve-component-norms-max-newton-it", type=int)
    parser.add_argument("--enable-form-block-diagnostics", action="store_true")
    parser.add_argument("--enable-interior-face-timing", action="store_true")
    parser.add_argument("--require-interior-face-timing-diagnostics", action="store_true")
    args = parser.parse_args()

    solver = resolve_solver(args.solver)
    cases = args.case or ["mini2d"]
    report = []
    for case_name in cases:
        report.append(run_case(case_name, solver, args))
        write_qualification_log(args.qualification_log, solver, report, complete=False)
    write_qualification_log(args.qualification_log, solver, report, complete=True)
    print(json.dumps({"solver": str(solver), "probes": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
