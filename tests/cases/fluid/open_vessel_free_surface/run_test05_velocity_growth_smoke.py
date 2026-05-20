#!/usr/bin/env python3
"""Run a short unfitted dam-break velocity-growth solver probe."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
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
    "capillaryarc2d": None,
    "curvedtet3d": None,
    "open2d": CASE_ROOT,
    "d18": CASE_ROOT / "spheric_test05_wet_bed_d18",
    "d38": CASE_ROOT / "spheric_test05_wet_bed_d38",
    "mms2d": CASE_ROOT / "mms_traveling_interface_2d",
    "sloshing2d": CASE_ROOT / "linear_sloshing_2d",
    "tilt2d": CASE_ROOT / "square_tank_tilt_settling",
}
CASE_COPY_ENTRIES = {
    CASE_ROOT: ("solver.xml", "pressure_gauge.csv", "mesh"),
}
CASE_GATE_X = {
    "capillaryarc2d": 0.5,
    "mini2d": 0.4,
    "static2d": 0.5,
    "open2d": 0.5,
    "mms2d": 0.5,
    "sloshing2d": 0.5,
    "tilt2d": 0.5,
    "curvedtet3d": 0.5,
}
HIGH_ORDER_PRODUCTION_CASES = ("sloshing2d", "tilt2d")
HIGH_ORDER_MPI_PRODUCTION_CASES = ("sloshing2d", "tilt2d")
HIGH_ORDER_VISIBLE_MOTION_CASES = ("tilt2d",)
HIGH_ORDER_3D_BENCHMARK_CASES = ("d18",)
HIGH_ORDER_3D_BENCHMARK_QUALIFICATION_CASES = ("d18", "d38")
HIGH_ORDER_3D_BENCHMARK_PROFILE_CASES = ("d18", "d38")
HIGH_ORDER_CURVED_3D_SIMPLEX_CASES = ("curvedtet3d",)
HIGH_ORDER_MPI_MOTION_CASES = ("sloshing2d",)
HIGH_ORDER_CAPILLARY_PROJECTION_CASES = ("sloshing2d",)
HIGH_ORDER_CAPILLARY_RESPONSE_CASES = ("capillaryarc2d",)
HIGH_ORDER_CAPILLARY_BALANCE_CASES = ("capillaryarc2d",)
HIGH_ORDER_VOLUME_CORRECTED_MOTION_CASES = ("sloshing2d",)
HIGH_ORDER_SYNTHETIC_CASES = {"capillaryarc2d", "curvedtet3d"}
CUT_CONTEXT_VOLUME_RE = re.compile(r"active_side_volume=([-+0-9.eE]+)")
CUT_ASSEMBLY_VOLUME_RE = re.compile(r"(?<!_)active_wet_volume=([-+0-9.eE]+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
JIT_MINUS_SHAPE_RE = re.compile(
    r"minus\[qpts=([^,\]]+),test=([^,\]]+),trial=([^\]]+)\]"
)
JIT_PLUS_SHAPE_RE = re.compile(
    r"plus\[qpts=([^,\]]+),test=([^,\]]+),trial=([^\]]+)\]"
)
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
BLOCK_SUMMARY_RE = re.compile(r"(?P<name>[^;{}]+)\{(?P<body>[^}]*)\}")
JACOBIAN_COMPONENT_BLOCK_MIN_DENOMINATOR = 1.0e-12
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
TIMELOOP_REJECTED_RE = re.compile(
    r"TimeLoop: step_rejected step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
    r" reason=(?P<reason>\S+)"
    r" \(newton: converged=(?P<converged>[01])"
    r" iters=(?P<nonlinear_iterations>[0-9]+)"
    r" \|\|r\|\|=(?P<residual>[-+0-9.eE]+)"
    r" \|\|r_field\|\|=(?P<field_residual>[-+0-9.eE]+)"
    r" \|\|r_aux\|\|=(?P<aux_residual>[-+0-9.eE]+)\)"
)
TIMELOOP_DT_UPDATED_RE = re.compile(
    r"TimeLoop: dt_updated step=(?P<step>[0-9]+)"
    r" attempt=(?P<attempt>[0-9]+)"
    r" old_dt=(?P<old_dt>[-+0-9.eE]+)"
    r" new_dt=(?P<new_dt>[-+0-9.eE]+)"
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
CAPILLARY_ARC_CENTER_X = 0.5
CAPILLARY_ARC_CENTER_Y = -0.3
CAPILLARY_ARC_RADIUS = 0.8
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


def set_linear_algebra_backend(solver: ET.Element,
                               backend: str,
                               preconditioner: str = "none") -> None:
    element = solver.find("Linear_algebra")
    if element is None:
        element = ET.SubElement(solver, "Linear_algebra")
    element.set("type", backend)
    set_text(element, "Preconditioner", preconditioner)


def default_preconditioner_for_backend(backend: str) -> str:
    if backend.strip().lower() == "fsils":
        return "fsils"
    return "none"


def free_surface_bc(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") != "fluid":
            continue
        for bc in equation.findall("Add_BC"):
            if bc.attrib.get("name") == "free_surface":
                return bc
    raise ValueError("missing fluid free-surface boundary condition")


def level_set_equation(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") == "level_set":
            return equation
    raise ValueError("missing level-set equation")


def fluid_equation(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") == "fluid":
            return equation
    raise ValueError("missing fluid equation")


def navier_stokes_linear_solver(root: ET.Element) -> ET.Element:
    solvers = fluid_equation(root).findall("LS")
    for solver in solvers:
        if solver.attrib.get("type") == "NS":
            return solver
    for solver in solvers:
        if solver.find("NS_GM_max_iterations") is not None:
            return solver
    if len(solvers) == 1:
        return solvers[0]
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
                     linear_algebra_backend: str | None = None,
                     linear_preconditioner: str | None = None,
                     disable_coupled_outer_fgmres: bool = False,
                     disable_cut_metadata_scale: bool = False,
                     disable_velocity_extension: bool = False,
                     disable_vtk_output: bool = False,
                     final_output_only: bool = False,
                     vtk_save_increment: int | None = None,
                     start_saving_after_step: int | None = None,
                     generated_interface_geometry: str | None = None,
                     implicit_cut_quadrature_backend: str | None = None,
                     implicit_cut_fallback_policy: str | None = None,
                     required_implicit_cut_backend_qualification: str | None = None,
                     implicit_cut_root_tolerance: float | None = None,
                     implicit_cut_max_subdivision_depth: int | None = None,
                     generated_interface_quadrature_order: int | None = None,
                     interface_quadrature_order: int | None = None,
                     volume_quadrature_order: int | None = None,
                     cut_cell_velocity_gradient_penalty: float | None = None,
                     cut_cell_pressure_gradient_penalty: float | None = None,
                     surface_tension: float | None = None,
                     projected_curvature_field: str | None = None,
                     curvature_projection_cadence_steps: int | None = None,
                     curvature_projection_max_normalized_fit_residual: float | None = None,
                     curvature_projection_smoothing_iterations: int | None = None,
                     curvature_projection_smoothing_relaxation: float | None = None,
                     enable_volume_correction: bool | None = None,
                     volume_correction_cadence_steps: int | None = None,
                     volume_correction_use_initial_volume: bool | None = None,
                     volume_correction_tolerance: float | None = None,
                     volume_correction_max_iterations: int | None = None) -> None:
    tree = ET.parse(solver_xml)
    root = tree.getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise ValueError("missing GeneralSimulationParameters")
    for parent in root.iter():
        for linear_algebra in list(parent.findall("Linear_algebra")):
            if linear_algebra.attrib.get("type", "").strip().lower() == "eigen":
                parent.remove(linear_algebra)

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
        linear_algebra_backend is not None or
        linear_preconditioner is not None or
        disable_coupled_outer_fgmres
    )
    ns_solver = navier_stokes_linear_solver(root) if needs_ns_solver else None
    if linear_solver_type is not None:
        assert ns_solver is not None
        ns_solver.set("type", linear_solver_type)
        if (linear_solver_type.strip().lower() != "direct" and
                linear_algebra_backend is None):
            set_linear_algebra_backend(
                ns_solver,
                "fsils",
                linear_preconditioner or "fsils",
            )
    if linear_algebra_backend is not None:
        assert ns_solver is not None
        set_linear_algebra_backend(
            ns_solver,
            linear_algebra_backend,
            linear_preconditioner or
            default_preconditioner_for_backend(linear_algebra_backend),
        )
    elif linear_preconditioner is not None:
        assert ns_solver is not None
        set_linear_algebra_backend(ns_solver, "fsils", linear_preconditioner)
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
    if disable_velocity_extension:
        set_text(free_surface, "Enable_velocity_extension", "false")
    else:
        require_text(free_surface, "Enable_velocity_extension", "true")
    if generated_interface_geometry is not None:
        set_text(free_surface, "Generated_interface_geometry", generated_interface_geometry)
    if implicit_cut_quadrature_backend is not None:
        set_text(free_surface, "Implicit_cut_quadrature_backend", implicit_cut_quadrature_backend)
    if implicit_cut_fallback_policy is not None:
        set_text(free_surface, "Implicit_cut_fallback_policy", implicit_cut_fallback_policy)
    if required_implicit_cut_backend_qualification is not None:
        set_text(
            free_surface,
            "Required_implicit_cut_backend_qualification",
            required_implicit_cut_backend_qualification,
        )
    if implicit_cut_root_tolerance is not None:
        set_text(free_surface, "Implicit_cut_root_tolerance", f"{implicit_cut_root_tolerance:.16g}")
    if implicit_cut_max_subdivision_depth is not None:
        set_text(free_surface, "Implicit_cut_max_subdivision_depth", str(implicit_cut_max_subdivision_depth))
    if generated_interface_quadrature_order is not None:
        set_text(free_surface, "Generated_interface_quadrature_order", str(generated_interface_quadrature_order))
    if interface_quadrature_order is not None:
        set_text(free_surface, "Interface_quadrature_order", str(interface_quadrature_order))
    if volume_quadrature_order is not None:
        set_text(free_surface, "Volume_quadrature_order", str(volume_quadrature_order))
    if cut_cell_velocity_gradient_penalty is not None:
        set_text(
            free_surface,
            "Cut_cell_velocity_gradient_penalty",
            f"{cut_cell_velocity_gradient_penalty:.16g}",
        )
    if cut_cell_pressure_gradient_penalty is not None:
        set_text(
            free_surface,
            "Cut_cell_pressure_gradient_penalty",
            f"{cut_cell_pressure_gradient_penalty:.16g}",
        )
    if surface_tension is not None:
        set_text(free_surface, "Surface_tension", f"{surface_tension:.16g}")
    if projected_curvature_field:
        level_set = level_set_equation(root)
        set_text(level_set, "Enable_curvature_projection", "true")
        set_text(level_set, "Projected_curvature_field", projected_curvature_field)
        set_text(free_surface, "Curvature_field", projected_curvature_field)
        if curvature_projection_cadence_steps is not None:
            set_text(
                level_set,
                "Curvature_projection_cadence_steps",
                str(curvature_projection_cadence_steps),
            )
        if curvature_projection_max_normalized_fit_residual is not None:
            set_text(
                level_set,
                "Curvature_projection_max_normalized_fit_residual",
                f"{curvature_projection_max_normalized_fit_residual:.16g}",
            )
        if curvature_projection_smoothing_iterations is not None:
            set_text(
                level_set,
                "Curvature_projection_smoothing_iterations",
                str(curvature_projection_smoothing_iterations),
            )
        if curvature_projection_smoothing_relaxation is not None:
            set_text(
                level_set,
                "Curvature_projection_smoothing_relaxation",
                f"{curvature_projection_smoothing_relaxation:.16g}",
            )
    if enable_volume_correction is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Enable_volume_correction",
                 "true" if enable_volume_correction else "false")
    if volume_correction_cadence_steps is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_cadence_steps",
                 str(volume_correction_cadence_steps))
    if volume_correction_use_initial_volume is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_use_initial_volume",
                 "true" if volume_correction_use_initial_volume else "false")
    if volume_correction_tolerance is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_tolerance",
                 f"{volume_correction_tolerance:.16g}")
    if volume_correction_max_iterations is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_max_iterations",
                 str(volume_correction_max_iterations))

    if disable_coupled_outer_fgmres:
        assert ns_solver is not None
        set_text(ns_solver, "NS_Use_coupled_outer_FGMRES", "false")

    tree.write(solver_xml, encoding="UTF-8", xml_declaration=True)


def regenerate_mms_case_if_requested(case_name: str,
                                     run_dir: Path,
                                     args: argparse.Namespace) -> None:
    if case_name != "mms2d" or (args.mms_nx is None and args.mms_ny is None):
        return
    generator = run_dir / "generate_case.py"
    if not generator.exists():
        raise FileNotFoundError(generator)
    nx = args.mms_nx if args.mms_nx is not None else args.mms_ny
    ny = args.mms_ny if args.mms_ny is not None else args.mms_nx
    if nx is None or ny is None:
        raise ValueError("MMS grid regeneration requires nx and ny")
    if nx < 2 or ny < 2:
        raise ValueError("MMS grid regeneration requires nx and ny to be at least 2")
    command = [
        sys.executable,
        str(generator),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--element-order",
        "2",
    ]
    completed = subprocess.run(
        command,
        cwd=run_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to regenerate compact MMS case:\n" + completed.stdout
        )


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
        if args.jacobian_check_scheme:
            env["SVMP_FE_JACOBIAN_CHECK_SCHEME"] = args.jacobian_check_scheme
        if args.jacobian_check_components:
            env["SVMP_FE_JACOBIAN_CHECK_COMPONENTS"] = args.jacobian_check_components
        if args.jacobian_check_component_sweeps:
            env["SVMP_FE_JACOBIAN_CHECK_COMPONENT_SWEEPS"] = (
                args.jacobian_check_component_sweeps
            )
    if args.enable_newton_direction_check:
        env["SVMP_NEWTON_DIRECTION_CHECK"] = "1"
    if (args.enable_newton_assembly_diagnostics or
            args.require_newton_assembly_diagnostics):
        env["SVMP_NEWTON_ASSEMBLY_DIAGNOSTICS"] = "1"
    if args.newton_line_search_fail_on_no_reduction:
        env["SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION"] = "1"
    if args.newton_line_search_max_iterations is not None:
        env["SVMP_NEWTON_LINE_SEARCH_MAX_ITERATIONS"] = str(
            args.newton_line_search_max_iterations
        )
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
    if args.enable_linear_solve_memory_diagnostics:
        env["SVMP_LINEAR_SOLVE_MEMORY_DIAGNOSTICS"] = "1"
    if args.enable_fsils_matrix_diagnostics:
        env["SVMP_FSILS_MATRIX_DIAGNOSTICS"] = "1"
        if args.fsils_matrix_diagnostics_every_n is not None:
            env["SVMP_FSILS_MATRIX_DIAGNOSTICS_EVERY_N"] = str(
                args.fsils_matrix_diagnostics_every_n
            )
        if args.fsils_matrix_diagnostics_max_records is not None:
            env["SVMP_FSILS_MATRIX_DIAGNOSTICS_MAX_RECORDS"] = str(
                args.fsils_matrix_diagnostics_max_records
            )
    if args.require_eigen_factorization_diagnostics:
        env["SVMP_FE_EIGEN_FACTOR_DIAGNOSTICS"] = "1"
    if (args.enable_timeloop_initialization_diagnostics or
            args.require_timeloop_initialization_diagnostics):
        env["SVMP_TIMELOOP_INITIALIZATION_DIAGNOSTICS"] = "1"
    if args.enable_form_block_diagnostics:
        env["SVMP_FE_FORM_BLOCK_DIAGNOSTICS"] = "1"
    if args.enable_interior_face_timing:
        env["SVMP_INTERIOR_FACE_TIMING"] = "1"
    if args.enable_cut_volume_timing:
        env["SVMP_CUT_VOLUME_TIMING"] = "1"
    if args.enable_jit_specialization_trace:
        env["SVMP_JIT_TRACE_SPECIALIZATION"] = "1"
    if args.enable_jit_cache_diagnostics:
        env["SVMP_JIT_CACHE_DIAGNOSTICS"] = "1"
    if args.enable_adaptive_time_loop:
        env["SVMP_TIMELOOP_ADAPTIVE"] = "1"
        env["SVMP_VTK_OUTPUT_FINAL_TIME"] = "1"
        for arg_name, env_name in (
            ("adaptive_time_loop_min_dt", "SVMP_TIMELOOP_MIN_DT"),
            ("adaptive_time_loop_max_dt", "SVMP_TIMELOOP_MAX_DT"),
            ("adaptive_time_loop_max_retries", "SVMP_TIMELOOP_MAX_RETRIES"),
            ("adaptive_time_loop_decrease_factor", "SVMP_TIMELOOP_DECREASE_FACTOR"),
            ("adaptive_time_loop_increase_factor", "SVMP_TIMELOOP_INCREASE_FACTOR"),
            ("adaptive_time_loop_target_newton_iterations",
             "SVMP_TIMELOOP_TARGET_NEWTON_ITERATIONS"),
            ("adaptive_time_loop_max_steps_multiplier",
             "SVMP_TIMELOOP_MAX_STEPS_MULTIPLIER"),
        ):
            value = getattr(args, arg_name)
            if value is not None:
                env[env_name] = f"{value:.16g}" if isinstance(value, float) else str(value)
    return env


def case_artifact_ignore(_path: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if re.match(r"result_.*\.p?vtu$", name):
            ignored.add(name)
        elif name in {"result.pvd", "1-procs", "2-procs", "3-procs", "4-procs"}:
            ignored.add(name)
        elif name.startswith("restart") or name.endswith(".log"):
            ignored.add(name)
    return ignored


def copy_selected_entries(source_root: Path,
                          destination: Path,
                          entries: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        source = source_root / entry
        target = destination / entry
        if source.is_dir():
            shutil.copytree(source, target, ignore=case_artifact_ignore)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_case_from_ref(case_dir: Path,
                       destination: Path,
                       source_ref: str,
                       entries: tuple[str, ...] | None = None) -> None:
    relative = case_dir.relative_to(ROOT)
    archive_paths = [str(relative)]
    if entries is not None:
        archive_paths = [str(relative / entry) for entry in entries]
    completed = subprocess.run(
        ["git", "archive", "--format=tar", source_ref, *archive_paths],
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
    if entries is None:
        shutil.move(str(archive_root / relative), destination)
    else:
        copy_selected_entries(archive_root / relative, destination, entries)


def copy_case(case_dir: Path, destination: Path, source_ref: str | None) -> None:
    entries = CASE_COPY_ENTRIES.get(case_dir)
    if source_ref is not None:
        copy_case_from_ref(case_dir, destination, source_ref, entries)
        return

    if entries is not None:
        copy_selected_entries(case_dir, destination, entries)
    else:
        shutil.copytree(case_dir, destination, ignore=case_artifact_ignore)


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
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
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
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
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
    <Enable_velocity_extension>true</Enable_velocity_extension>
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


def capillary_arc2d_phi(points: np.ndarray) -> np.ndarray:
    return np.sqrt((points[:, 0] - CAPILLARY_ARC_CENTER_X) ** 2 +
                   (points[:, 1] - CAPILLARY_ARC_CENTER_Y) ** 2) - (
                       CAPILLARY_ARC_RADIUS)


def write_capillary_arc2d_case(case_dir: Path,
                               steps: int,
                               pressure_jump: float = 0.0) -> None:
    write_mini_case(case_dir, steps, static=True)

    mesh_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    phi = capillary_arc2d_phi(points)
    grid.point_data["phi"] = phi
    grid.point_data["Pressure"] = np.where(phi < 0.0, pressure_jump, 0.0)
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.save(mesh_path)

    gauge_point = np.array([0.5, 0.0, 0.0], dtype=float)
    gauge_node = int(np.argmin(np.linalg.norm(points - gauge_point, axis=1)))
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{pressure_jump:.16g}\n",
        encoding="utf-8")
    benchmark = {
        "benchmark": "synthetic zero-gravity capillary arc smoke",
        "representation": "unfitted_level_set",
        "capillary_arc_radius": CAPILLARY_ARC_RADIUS,
        "initial_active_pressure": pressure_jump,
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "profile_window_x_min": 0.5,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_hydrostatic_pressure": pressure_jump,
        },
        "notes": [
            "The wall-supported circular arc starts from zero velocity and zero gravity.",
            "A zero active pressure preload exercises capillary response.",
            "A negative gamma/R active pressure preload exercises a Laplace-style capillary balance.",
        ],
    }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


TETRA10_EDGE_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)
TETRA10_FACE_CORNERS = (
    (1, 2, 3),
    (0, 3, 2),
    (0, 1, 3),
    (0, 2, 1),
)


def curved_tet3d_surface_height(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    z = points[:, 2]
    return 0.55 + 0.08 * np.sin(np.pi * x) * np.cos(2.0 * np.pi * z / 0.25)


def curved_tet3d_phi(points: np.ndarray) -> np.ndarray:
    return points[:, 1] - curved_tet3d_surface_height(points)


def curved_tet3d_pressure(points: np.ndarray) -> np.ndarray:
    rho = 998.2
    gravity = 9.81
    return rho * gravity * np.maximum(curved_tet3d_surface_height(points) - points[:, 1], 0.0)


def curved_tet3d_midpoint(base_points: np.ndarray, a: int, b: int) -> np.ndarray:
    point = 0.5 * (base_points[a] + base_points[b])
    x, y, z = point
    displacement = np.array([
        0.0,
        0.012 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z / 0.25),
        0.006 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(2.0 * np.pi * z / 0.25),
    ])
    return point + displacement


def orient_tetra_positive(points: np.ndarray, tet: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c, d = tet
    volume = float(np.dot(np.cross(points[b] - points[a], points[c] - points[a]),
                          points[d] - points[a]))
    if volume < 0.0:
        return (a, b, d, c)
    return tet


def write_curved_tet3d_grid(case_dir: Path) -> tuple[int, float]:
    nx, ny, nz = 2, 3, 2
    length, height, width = 1.0, 1.0, 0.25
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    zs = np.linspace(0.0, width, nz + 1)
    base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)

    def node(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    linear_tets: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = node(i, j, k)
                n100 = node(i + 1, j, k)
                n010 = node(i, j + 1, k)
                n110 = node(i + 1, j + 1, k)
                n001 = node(i, j, k + 1)
                n101 = node(i + 1, j, k + 1)
                n011 = node(i, j + 1, k + 1)
                n111 = node(i + 1, j + 1, k + 1)
                linear_tets.extend([
                    (n000, n001, n011, n111),
                    (n000, n011, n010, n111),
                    (n000, n010, n110, n111),
                    (n000, n110, n100, n111),
                    (n000, n100, n101, n111),
                    (n000, n101, n001, n111),
                ])
    linear_tets = [orient_tetra_positive(base_points, tet) for tet in linear_tets]

    points = [point.copy() for point in base_points]
    edge_midpoints: dict[tuple[int, int], int] = {}

    def midpoint_id(a: int, b: int) -> int:
        key = tuple(sorted((int(a), int(b))))
        if key not in edge_midpoints:
            edge_midpoints[key] = len(points)
            points.append(curved_tet3d_midpoint(base_points, key[0], key[1]))
        return edge_midpoints[key]

    tet10_cells: list[list[int]] = []
    for tet in linear_tets:
        cell = list(tet)
        cell.extend(midpoint_id(tet[a], tet[b]) for a, b in TETRA10_EDGE_PAIRS)
        tet10_cells.append(cell)

    point_array_values = np.asarray(points, dtype=float)
    cells = np.asarray([[10, *cell] for cell in tet10_cells], dtype=np.int64).ravel()
    cell_types = np.full(len(tet10_cells), int(pv.CellType.QUADRATIC_TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, point_array_values)
    grid.point_data["GlobalNodeID"] = np.arange(grid.n_points, dtype=np.int64)
    grid.point_data["phi"] = curved_tet3d_phi(point_array_values)
    grid.point_data["Pressure"] = curved_tet3d_pressure(point_array_values)
    grid.point_data["Velocity"] = np.zeros((grid.n_points, 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells, dtype=np.int64)

    mesh_dir = case_dir / "mesh/background"
    surface_dir = mesh_dir / "mesh-surfaces"
    surface_dir.mkdir(parents=True)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu", binary=False)

    face_counts: dict[tuple[int, int, int], tuple[int, list[int]]] = {}
    for cell in tet10_cells:
        corners = cell[:4]
        for face in TETRA10_FACE_CORNERS:
            face_corners = [corners[index] for index in face]
            key = tuple(sorted(face_corners))
            if key not in face_counts:
                mids = [
                    midpoint_id(face_corners[0], face_corners[1]),
                    midpoint_id(face_corners[1], face_corners[2]),
                    midpoint_id(face_corners[2], face_corners[0]),
                ]
                face_counts[key] = (0, [*face_corners, *mids])
            count, stored = face_counts[key]
            face_counts[key] = (count + 1, stored)

    surfaces: dict[str, list[list[int]]] = {
        "wall_left": [],
        "wall_right": [],
        "wall_bottom": [],
        "wall_front": [],
        "wall_back": [],
        "wall_top": [],
    }
    tol = 1.0e-12
    for key, (count, face_nodes) in face_counts.items():
        if count != 1:
            continue
        center = np.mean(base_points[np.asarray(key, dtype=np.int64)], axis=0)
        if abs(center[0]) <= tol:
            surfaces["wall_left"].append(face_nodes)
        elif abs(center[0] - length) <= tol:
            surfaces["wall_right"].append(face_nodes)
        elif abs(center[1]) <= tol:
            surfaces["wall_bottom"].append(face_nodes)
        elif abs(center[2]) <= tol:
            surfaces["wall_front"].append(face_nodes)
        elif abs(center[2] - width) <= tol:
            surfaces["wall_back"].append(face_nodes)
        elif abs(center[1] - height) <= tol:
            surfaces["wall_top"].append(face_nodes)

    for name, faces in surfaces.items():
        if not faces:
            raise RuntimeError(f"curvedtet3d surface {name!r} has no faces")
        used = sorted({node_id for face in faces for node_id in face})
        local = {node_id: index for index, node_id in enumerate(used)}
        surface_cells = np.asarray(
            [[6, *(local[node_id] for node_id in face)] for face in faces],
            dtype=np.int64,
        ).ravel()
        surface_types = np.full(len(faces), int(pv.CellType.QUADRATIC_TRIANGLE), dtype=np.uint8)
        surface = pv.UnstructuredGrid(surface_cells, surface_types, point_array_values[used])
        surface.point_data["GlobalNodeID"] = np.asarray(used, dtype=np.int64)
        surface.cell_data["GlobalElementID"] = np.arange(len(faces), dtype=np.int64)
        surface.save(surface_dir / f"{name}.vtu", binary=False)

    gauge_point = np.array([0.5, 0.0, 0.125], dtype=float)
    gauge_node = int(np.argmin(np.linalg.norm(point_array_values - gauge_point, axis=1)))
    gauge_pressure = float(curved_tet3d_pressure(point_array_values[[gauge_node]])[0])
    return gauge_node, gauge_pressure


def write_curved_tet3d_solver_xml(case_dir: Path,
                                  steps: int,
                                  gauge_node: int,
                                  gauge_pressure: float) -> None:
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{gauge_pressure:.16g}\n", encoding="utf-8")
    benchmark = {
        "benchmark": "synthetic curved Tetra10 open-vessel free-surface smoke",
        "representation": "unfitted_level_set",
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "tank_width": 0.25,
            "profile_window_x_min": 0.5,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_hydrostatic_pressure": gauge_pressure,
        },
        "notes": [
            "Generated at run time to exercise solver-level curved 3D Tetra10 geometry.",
            "Quadratic tetrahedra use curved midside coordinates and quadratic triangle wall files.",
        ],
    }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    face_blocks = "\n".join(
        f"""  <Add_face name="{name}">
    <Face_file_path>mesh/background/mesh-surfaces/{name}.vtu</Face_file_path>
  </Add_face>"""
        for name in ("wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back", "wall_top")
    )
    wall_bc_blocks = "\n".join(
        f"""  <Add_BC name="{name}">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>"""
        for name in ("wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back")
    )
    (case_dir / "solver.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>3</Number_of_spatial_dimensions>
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
{face_blocks}
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>coupled_field</Velocity_source>
  <Velocity_field_name>Velocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
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
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
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
  <Force_y>-9.81</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 0.55 0.0</Hydrostatic_pressure_reference_point>
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
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
{wall_bc_blocks}
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>open_vessel_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
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
""", encoding="utf-8")


def write_curved_tet3d_case(case_dir: Path, steps: int) -> None:
    case_dir.mkdir(parents=True)
    gauge_node, gauge_pressure = write_curved_tet3d_grid(case_dir)
    write_curved_tet3d_solver_xml(case_dir, steps, gauge_node, gauge_pressure)


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


def final_result_step(default_step: int,
                      diagnostics: dict[str, Any]) -> int:
    time_loop = diagnostics.get("time_loop", {})
    if isinstance(time_loop, dict):
        accepted_steps = time_loop.get("accepted_steps", [])
        if accepted_steps:
            final_step = accepted_steps[-1].get("step")
            if isinstance(final_step, int) and final_step > 0:
                return final_step
    return default_step


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


def parse_jit_specialization_trace(line: str) -> dict[str, Any]:
    values = parse_key_values(line)
    for prefix, pattern in (
        ("minus", JIT_MINUS_SHAPE_RE),
        ("plus", JIT_PLUS_SHAPE_RE),
    ):
        match = pattern.search(line)
        if match is None:
            continue
        values[f"{prefix}_qpts"] = parse_scalar(match.group(1))
        values[f"{prefix}_test_dofs"] = parse_scalar(match.group(2))
        values[f"{prefix}_trial_dofs"] = parse_scalar(match.group(3))
    return values


def parse_interior_face_timing(line: str) -> dict[str, Any]:
    values = {
        match.group(1): parse_scalar(match.group(2))
        for match in INTERIOR_FACE_TIMING_VALUE_RE.finditer(line)
    }
    values["diagnostic"] = "interior_face_timing"
    return values


def parse_cut_volume_timing(line: str) -> dict[str, Any]:
    values = parse_key_values(line)
    values.update({
        match.group(1): parse_scalar(match.group(2))
        for match in INTERIOR_FACE_TIMING_VALUE_RE.finditer(line)
    })
    values["diagnostic"] = "cut_volume_timing"
    return values


def count_key(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts = []
    for field in fields:
        value = record.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return ",".join(parts) if parts else "unclassified"


def top_counts(counts: dict[str, int], limit: int = 24) -> dict[str, int]:
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda item: (-counts[item], item))[:limit]
    }


def increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def parse_count_summary(value: Any) -> dict[str, int]:
    if not isinstance(value, str) or not value or value == "none":
        return {}
    counts: dict[str, int] = {}
    for part in value.split(","):
        if ":" not in part:
            continue
        name, raw_count = part.split(":", 1)
        name = name.strip()
        try:
            count = int(raw_count.strip())
        except ValueError:
            continue
        if name and count > 0:
            counts[name] = counts.get(name, 0) + count
    return counts


def jit_shape_key(record: dict[str, Any]) -> str:
    shape = count_key(record, ("trigger", "domain", "role"))
    if "minus_qpts" in record:
        shape += (
            f",minus_qpts={record.get('minus_qpts')}"
            f",minus_test={record.get('minus_test_dofs')}"
            f",minus_trial={record.get('minus_trial_dofs')}"
        )
    elif "n_qpts" in record:
        shape += (
            f",qpts={record.get('n_qpts')}"
            f",test={record.get('n_test_dofs')}"
            f",trial={record.get('n_trial_dofs')}"
        )
    if "plus_qpts" in record:
        shape += (
            f",plus_qpts={record.get('plus_qpts')}"
            f",plus_test={record.get('plus_test_dofs')}"
            f",plus_trial={record.get('plus_trial_dofs')}"
        )
    if "affine" in record:
        shape += f",affine={record.get('affine')}"
    return shape


def timing_mode_key(record: dict[str, Any]) -> str:
    return f"matrix={record.get('matrix', '?')},vector={record.get('vector', '?')}"


def summarize_timing_modes(records: list[dict[str, Any]],
                           count_fields: tuple[str, ...],
                           time_fields: tuple[str, ...]) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        key = timing_mode_key(record)
        summary = summaries.setdefault(key, {"records": 0})
        summary["records"] += 1
        for field in count_fields:
            value = record.get(field)
            if isinstance(value, int):
                target = f"max_{field}"
                summary[target] = max(int(summary.get(target, value)), value)
        for field in time_fields:
            value = record.get(field)
            if isinstance(value, (int, float)):
                target = f"max_{field}_seconds"
                summary[target] = max(float(summary.get(target, value)), float(value))
    return summaries


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


def normalized_level_set_side(value: Any) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if "negative" in text:
        return "negative"
    if "positive" in text:
        return "positive"
    return text


def active_cut_volume_side(diagnostics: dict[str, Any]) -> str | None:
    for record in reversed(diagnostics.get("cut_context_rebuilds", [])):
        side = normalized_level_set_side(record.get("active_side"))
        if side is not None:
            return side
    return None


def active_cut_volume_records(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    records = (
        diagnostics.get("cut_volume_assembly_groups")
        or diagnostics.get("cut_volume_assemblies", [])
    )
    active_side = active_cut_volume_side(diagnostics)
    if active_side is None:
        return list(records)
    return [
        record for record in records
        if normalized_level_set_side(record.get("side")) == active_side
    ]


def summarize_time_loop(time_loop: dict[str, Any]) -> dict[str, Any]:
    nonlinear_records = time_loop.get("nonlinear_records", [])
    accepted_steps = time_loop.get("accepted_steps", [])
    rejected_steps = time_loop.get("rejected_steps", [])
    dt_updates = time_loop.get("dt_updates", [])
    summary: dict[str, Any] = {
        "nonlinear_records": len(nonlinear_records),
        "accepted_steps": len(accepted_steps),
        "rejected_steps": len(rejected_steps),
        "dt_updates": len(dt_updates),
        "vtk_outputs": len(time_loop.get("vtk_outputs", [])),
    }
    if accepted_steps:
        final_step = accepted_steps[-1]
        summary["final_accepted_step"] = final_step.get("step")
        summary["final_accepted_time"] = final_step.get("time")
        accepted_dt = [
            float(record["dt"])
            for record in accepted_steps
            if isinstance(record.get("dt"), (int, float))
        ]
        accepted_dt_range = numeric_range(accepted_dt)
        if accepted_dt_range is not None:
            summary["accepted_dt"] = accepted_dt_range
    if rejected_steps:
        rejected_dt = [
            float(record["dt"])
            for record in rejected_steps
            if isinstance(record.get("dt"), (int, float))
        ]
        rejected_dt_range = numeric_range(rejected_dt)
        if rejected_dt_range is not None:
            summary["rejected_dt"] = rejected_dt_range
        reasons: dict[str, int] = {}
        for record in rejected_steps:
            reason = str(record.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        summary["rejection_reasons"] = reasons
    if dt_updates:
        next_dt = [
            float(record["new_dt"])
            for record in dt_updates
            if isinstance(record.get("new_dt"), (int, float))
        ]
        next_dt_range = numeric_range(next_dt)
        if next_dt_range is not None:
            summary["updated_dt"] = next_dt_range
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


def diagnostic_header(line: str, marker: str) -> str:
    marker_index = line.find(marker)
    if marker_index < 0:
        return line
    payload_index = line.find(" [", marker_index)
    if payload_index < 0:
        return line
    return line[:payload_index]


def normalized_component_sweeps(value: str | None) -> list[str]:
    if not value:
        return []
    separator = ";" if ";" in value else ","
    sweeps = []
    for group in value.split(separator):
        tokens = [
            token.strip().lower()
            for token in group.split(",")
            if token.strip()
        ]
        if not tokens or tokens == ["all"]:
            label = "all"
        else:
            label = ",".join(tokens)
        if label not in sweeps:
            sweeps.append(label)
    return sweeps


def jacobian_component_block_metrics(
        records: list[dict[str, Any]]) -> dict[str, Any]:
    relative_errors: dict[str, float] = {}
    matrix_relative_errors: dict[str, float] = {}
    filters: list[str] = []
    skipped = 0
    for record in records:
        raw_filter = record.get("component_filter", record.get("components", "all"))
        column_filter = str(raw_filter or "all")
        if column_filter not in filters:
            filters.append(column_filter)
        components = record.get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            row = str(component.get("component", "unknown"))
            fd_norm = component.get("fd")
            full_norm = component.get("full")
            matrix_norm = component.get("matrix")
            total_err = component.get("total_err")
            matrix_err = component.get("matrix_err")
            if not all(isinstance(value, (int, float)) for value in (
                    fd_norm, full_norm, matrix_norm, total_err, matrix_err)):
                continue
            full_denominator = max(abs(float(fd_norm)), abs(float(full_norm)))
            matrix_denominator = max(abs(float(fd_norm)), abs(float(matrix_norm)))
            if full_denominator < JACOBIAN_COMPONENT_BLOCK_MIN_DENOMINATOR:
                skipped += 1
            else:
                key = f"column={column_filter},row={row}"
                relative_errors[key] = abs(float(total_err)) / full_denominator
            if matrix_denominator >= JACOBIAN_COMPONENT_BLOCK_MIN_DENOMINATOR:
                key = f"column={column_filter},row={row}"
                matrix_relative_errors[key] = abs(float(matrix_err)) / matrix_denominator
    result: dict[str, Any] = {
        "filters": filters,
        "skipped_near_zero_blocks": skipped,
    }
    if relative_errors:
        result["relative_errors"] = dict(sorted(relative_errors.items()))
        result["max_relative_error"] = max(relative_errors.values())
    if matrix_relative_errors:
        result["matrix_relative_errors"] = dict(sorted(matrix_relative_errors.items()))
        result["max_matrix_relative_error"] = max(matrix_relative_errors.values())
    return result


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


def parse_eigen_factorization_diagnostic(line: str) -> dict[str, Any]:
    record = parse_key_values(line)
    block_match = re.search(r"block_summaries=(.*)$", line)
    blocks = []
    if block_match is not None:
        for match in BLOCK_SUMMARY_RE.finditer(block_match.group(1)):
            block = parse_key_values(match.group("body").replace(",", " "))
            block["name"] = match.group("name").strip()
            blocks.append(block)
            if block["name"] == "Pressure":
                pressure_zero_rows = block.get("zero_rows")
                pressure_zero_cols = block.get("zero_cols")
                if isinstance(pressure_zero_rows, int):
                    record["pressure_zero_rows"] = pressure_zero_rows
                if isinstance(pressure_zero_cols, int):
                    record["pressure_zero_cols"] = pressure_zero_cols
                for key in ("zero_row_runs_local", "zero_col_runs_local"):
                    value = block.get(key)
                    if isinstance(value, str):
                        record[f"pressure_{key}"] = value
    if blocks:
        record["blocks"] = blocks
    return record


def parse_solver_diagnostics(solver_output: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "solver_controls": {},
        "cut_context_rebuilds": [],
        "cut_volume_assemblies": [],
        "hydrostatic_initializations": [],
        "pressure_gauge_checks": [],
        "residual_block_norms": [],
        "fsils_true_residuals": [],
        "fsils_prepared_matrices": [],
        "fsils_solve_summaries": [],
        "fsils_blockschur_retries": [],
        "timeloop_initialization_solves": [],
        "vector_component_norms": [],
        "newton_assemblies": [],
        "newton_direction_checks": [],
        "jacobian_checks": [],
        "jacobian_check_component_norms": [],
        "jacobian_check_component_details": [],
        "jacobian_check_component_filters": [],
        "jacobian_check_sweep_plans": [],
        "jacobian_check_top_mismatches": [],
        "form_block_dependencies": [],
        "form_block_installs": [],
        "form_mixed_plans": [],
        "linear_solve_histories": [],
        "jit_specialization_traces": [],
        "jit_cache_diagnostics": [],
        "assembly_timings": [],
        "process_memory": [],
        "interior_face_timings": [],
        "cut_volume_timings": [],
        "eigen_factorization_diagnostics": [],
        "active_pressure_support_constraints": [],
        "curvature_projections": [],
        "level_set_volume_corrections": [],
        "level_set_maintenance": [],
        "level_set_nonconservative_warnings": [],
        "time_loop": {
            "nonlinear_records": [],
            "accepted_steps": [],
            "rejected_steps": [],
            "dt_updates": [],
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
        rejected_match = TIMELOOP_REJECTED_RE.search(line)
        dt_updated_match = TIMELOOP_DT_UPDATED_RE.search(line)
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
        elif rejected_match is not None:
            diagnostics["time_loop"]["rejected_steps"].append(convert_match(rejected_match))
        elif dt_updated_match is not None:
            diagnostics["time_loop"]["dt_updates"].append(convert_match(dt_updated_match))
        elif vtk_match is not None:
            diagnostics["time_loop"]["vtk_outputs"].append(vtk_match.group("path").strip())
        elif "Active-domain cut context" in line:
            record = parse_key_values(line)
            diagnostics["cut_context_rebuilds"].append(record)
            if isinstance(record.get("process_rss_kb"), (int, float)):
                memory_record = dict(record)
                memory_record["phase"] = "cut_context_rebuild"
                diagnostics["process_memory"].append(memory_record)
        elif "diagnostic=process_memory" in line:
            diagnostics["process_memory"].append(parse_key_values(line))
        elif "cut-volume active-domain diagnostics" in line:
            diagnostics["cut_volume_assemblies"].append(parse_key_values(line))
        elif "hydrostatic pressure initialization" in line:
            diagnostics["hydrostatic_initializations"].append(parse_key_values(line))
        elif "pressure gauge diagnostic" in line:
            diagnostics["pressure_gauge_checks"].append(parse_key_values(line))
        elif "residual block norms" in line:
            diagnostics["residual_block_norms"].append(parse_key_values(line))
        elif "diagnostic=newton_assembly" in line:
            diagnostics["newton_assemblies"].append(parse_key_values(line))
        elif "true residual diagnostics" in line:
            diagnostics["fsils_true_residuals"].append(parse_key_values(line))
        elif "diagnostic=fsils_prepared_matrix" in line:
            diagnostics["fsils_prepared_matrices"].append(parse_key_values(line))
        elif "diagnostic=fsils_solve_summary" in line:
            diagnostics["fsils_solve_summaries"].append(parse_key_values(line))
        elif "diagnostic=fsils_blockschur_true_residual_retry" in line:
            diagnostics["fsils_blockschur_retries"].append(parse_key_values(line))
        elif "diagnostic=timeloop_initialization_linear_solve" in line:
            diagnostics["timeloop_initialization_solves"].append(parse_key_values(line))
        elif "NewtonSolver: direction check" in line:
            diagnostics["newton_direction_checks"].append(parse_norm_key_values(line))
        elif "NewtonSolver: Jacobian check jacobian_op=" in line:
            diagnostics["jacobian_checks"].append(parse_norm_key_values(line))
        elif "NewtonSolver: Jacobian check component norms" in line:
            record = parse_key_values(
                diagnostic_header(line, "diagnostic=jacobian_check_component_norms")
            )
            record["components"] = parse_jacobian_component_norms(line)
            diagnostics["jacobian_check_component_norms"].append(record)
        elif "diagnostic=jacobian_check_component_details" in line:
            record = parse_key_values(
                diagnostic_header(line, "diagnostic=jacobian_check_component_details")
            )
            record["components"] = parse_jacobian_component_details(line)
            diagnostics["jacobian_check_component_details"].append(record)
        elif "diagnostic=jacobian_check_component_filter" in line:
            diagnostics["jacobian_check_component_filters"].append(parse_key_values(line))
        elif "diagnostic=jacobian_check_sweep_plan" in line:
            diagnostics["jacobian_check_sweep_plans"].append(parse_key_values(line))
        elif "diagnostic=jacobian_check_top_mismatch" in line:
            record = parse_key_values(
                diagnostic_header(line, "diagnostic=jacobian_check_top_mismatch")
            )
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
        elif "Eigen direct factorization diagnostic" in line:
            diagnostics["eigen_factorization_diagnostics"].append(
                parse_eigen_factorization_diagnostic(line)
            )
        elif "diagnostic=level_set_active_side_vertex_constraint" in line:
            diagnostics["active_pressure_support_constraints"].append(
                parse_key_values(line)
            )
        elif "Level-set curvature projected" in line:
            diagnostics["curvature_projections"].append(parse_key_values(line))
        elif "Level-set volume corrected" in line:
            diagnostics["level_set_volume_corrections"].append(parse_key_values(line))
        elif "Level-set maintenance diagnostic" in line:
            diagnostics["level_set_maintenance"].append(parse_key_values(line))
        elif ("WARNING unfitted free-surface level-set has no enabled "
              "reinitialization or volume-correction request") in line:
            diagnostics["level_set_nonconservative_warnings"].append(
                parse_key_values(line)
            )
        elif "JIT specialization trace:" in line:
            diagnostics["jit_specialization_traces"].append(parse_jit_specialization_trace(line))
        elif "diagnostic=jit_cache" in line:
            diagnostics["jit_cache_diagnostics"].append(parse_key_values(line))
        elif "[INTERIOR_FACE_TIMING]" in line:
            diagnostics["interior_face_timings"].append(parse_interior_face_timing(line))
        elif "[CUT_VOLUME_TIMING]" in line:
            diagnostics["cut_volume_timings"].append(parse_cut_volume_timing(line))
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
            for record in active_cut_volume_records(diagnostics)
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
    context_volumes = diagnostic_context_active_side_volumes(
        diagnostics,
        prefer_physical=False,
    )
    assembly_volumes = [
        float(record["active_wet_volume"])
        for record in active_cut_volume_records(diagnostics)
        if isinstance(record.get("active_wet_volume"), (int, float))
    ]
    if not context_volumes or not assembly_volumes:
        return None
    return max(
        min(abs(assembly_volume - context_volume) for context_volume in context_volumes)
        for assembly_volume in assembly_volumes
    )


def diagnostic_context_active_side_physical_volumes(
        diagnostics: dict[str, Any]) -> list[float]:
    return [
        float(record["active_side_physical_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_side_physical_volume"), (int, float))
    ]


def diagnostic_context_active_side_reference_volumes(
        diagnostics: dict[str, Any]) -> list[float]:
    return [
        float(record["active_side_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_side_volume"), (int, float))
    ]


def diagnostic_context_active_side_volumes(
        diagnostics: dict[str, Any],
        *,
        prefer_physical: bool = True) -> list[float]:
    physical_volumes = diagnostic_context_active_side_physical_volumes(diagnostics)
    reference_volumes = diagnostic_context_active_side_reference_volumes(diagnostics)
    if prefer_physical and physical_volumes:
        return physical_volumes
    return reference_volumes


def diagnostic_cut_volume_min_exact_order(diagnostics: dict[str, Any]) -> int | None:
    orders = [
        int(record["min_exact_order"])
        for record in active_cut_volume_records(diagnostics)
        if isinstance(record.get("min_exact_order"), int)
    ]
    if not orders:
        return None
    return min(orders)


def diagnostic_cut_volume_max_exact_order(diagnostics: dict[str, Any]) -> int | None:
    orders = [
        int(record["max_exact_order"])
        for record in active_cut_volume_records(diagnostics)
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


def diagnostic_implicit_cut_fallback_cells(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["implicit_cut_fallback_cells"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("implicit_cut_fallback_cells"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_cut_context_min_int(diagnostics: dict[str, Any], key: str) -> int | None:
    values = [
        int(record[key])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get(key), int)
    ]
    if not values:
        return None
    return min(values)


def diagnostic_cut_context_value_counts(diagnostics: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_rebuilds", []):
        value = record.get(key)
        if value is not None:
            increment_count(counts, str(value))
    return counts


def diagnostic_cut_context_summary_counts(diagnostics: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_rebuilds", []):
        for name, count in parse_count_summary(record.get(key)).items():
            counts[name] = counts.get(name, 0) + count
    return counts


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


def cut_context_rebuild_provenance_counts(diagnostics: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_rebuilds", []):
        if not isinstance(record, dict):
            continue
        provenance = record.get("provenance")
        key = str(provenance) if provenance else "missing"
        increment_count(counts, key)
    return counts


def generated_cell_cache_summary(diagnostics: dict[str, Any]) -> dict[str, int]:
    records = diagnostics.get("cut_context_rebuilds", [])
    if not isinstance(records, list):
        return {}
    summary = {
        "rebuilds_with_cell_cache": 0,
        "total_hits": 0,
        "total_misses": 0,
        "domain_hits": 0,
        "full_miss_rebuilds": 0,
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        hits = record.get("generated_cell_cache_hits")
        misses = record.get("generated_cell_cache_misses")
        cell_count = record.get("cell_count")
        if not isinstance(hits, int) or not isinstance(misses, int):
            continue
        summary["rebuilds_with_cell_cache"] += 1
        summary["total_hits"] += hits
        summary["total_misses"] += misses
        domain_hits = record.get("generated_domain_cache_hits")
        if isinstance(domain_hits, int):
            summary["domain_hits"] += domain_hits
        if (isinstance(cell_count, int) and cell_count > 0 and
                hits == 0 and misses == cell_count):
            summary["full_miss_rebuilds"] += 1
    return summary


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


def assembly_efficiency_errors(metrics: dict[str, Any],
                               args: argparse.Namespace) -> list[str]:
    checks = (
        ("max_diagnostic_assembly_timings_per_step",
         "diagnostic_assembly_timings_per_accepted_step",
         "assembly timing records per accepted step"),
        ("max_diagnostic_extra_assembly_timings_per_step",
         "diagnostic_extra_assembly_timings_per_accepted_step",
         "extra assembly timing records per accepted step"),
        ("max_diagnostic_cut_context_rebuilds_per_step",
         "diagnostic_cut_context_rebuilds_per_accepted_step",
         "cut-context rebuilds per accepted step"),
        ("max_diagnostic_newton_matrix_assemblies_per_step",
         "diagnostic_newton_matrix_assemblies_per_accepted_step",
         "Newton matrix assemblies per accepted step"),
        ("max_diagnostic_generated_cell_cache_full_miss_rebuilds",
         "diagnostic_generated_cell_cache_full_miss_rebuilds",
         "generated-interface full cell-cache miss rebuilds"),
    )
    errors = []
    for arg_name, metric_name, label in checks:
        threshold = getattr(args, arg_name)
        if threshold is None:
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            errors.append(f"{label} diagnostic is unavailable")
        elif float(value) > float(threshold):
            errors.append(
                f"{label} {float(value):.6g} exceeds {float(threshold):.6g}"
            )
    return errors


def resource_ceiling_errors(metrics: dict[str, Any],
                            args: argparse.Namespace) -> list[str]:
    checks = (
        ("max_diagnostic_process_rss_kb",
         "diagnostic_process_max_rss_kb",
         "process RSS"),
        ("max_diagnostic_process_rss_growth_kb",
         "diagnostic_process_rss_growth_kb",
         "process RSS growth"),
        ("max_diagnostic_process_basis_cache_entry_growth",
         "diagnostic_process_basis_cache_entry_growth",
         "basis-cache entry growth"),
    )
    errors = []
    for arg_name, metric_name, label in checks:
        threshold = getattr(args, arg_name)
        if threshold is None:
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            errors.append(f"{label} diagnostic is unavailable")
        elif float(value) > float(threshold):
            errors.append(
                f"{label} {float(value):.6g} exceeds {float(threshold):.6g}"
            )
    return errors


def cut_context_policy_errors(metrics: dict[str, Any],
                              args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics["diagnostics"]
    records = diagnostics.get("cut_context_rebuilds", [])
    if args.require_high_order_cut_context_diagnostics:
        if not records:
            errors.append("cut-context rebuild diagnostics were not reported")
        else:
            required = (
                "generated_interface_geometry",
                "implicit_cut_quadrature_backend",
                "selected_implicit_cut_quadrature_backend_counts",
                "implicit_cut_backend_seconds",
                "implicit_cut_backend_seconds_max",
                "implicit_cut_fallback_policy",
                "implicit_cut_fallback_cells",
                "implicit_cut_backend_qualification_counts",
                "required_implicit_cut_backend_qualification",
                "achieved_interface_quadrature_order",
                "achieved_volume_quadrature_order",
                "interface_rule_count",
                "interface_quadrature_point_count",
                "active_volume_rule_count",
                "active_volume_quadrature_point_count",
            )
            missing = [
                key for key in required
                if not any(key in record for record in records)
            ]
            if missing:
                errors.append(
                    "cut-context diagnostics are missing high-order policy field(s): "
                    + ", ".join(missing)
                )
    for arg_name, metric_name, label in (
            ("expect_generated_interface_geometry",
             "diagnostic_generated_interface_geometry_counts",
             "generated interface geometry"),
            ("expect_implicit_cut_quadrature_backend",
             "diagnostic_implicit_cut_quadrature_backend_counts",
             "implicit cut quadrature backend"),
            ("expect_selected_implicit_cut_quadrature_backend",
             "diagnostic_selected_implicit_cut_quadrature_backend_counts",
             "selected implicit cut quadrature backend"),
            ("expect_implicit_cut_backend_qualification",
             "diagnostic_implicit_cut_backend_qualification_counts",
             "implicit cut backend qualification"),
            ("expect_implicit_cut_fallback_policy",
             "diagnostic_implicit_cut_fallback_policy_counts",
             "implicit cut fallback policy")):
        expected = getattr(args, arg_name)
        if expected is None:
            continue
        counts = metrics.get(metric_name)
        if not isinstance(counts, dict) or not counts:
            errors.append(f"diagnostic {label} counts are unavailable")
        elif expected not in counts:
            observed = ", ".join(str(key) for key in sorted(counts))
            errors.append(
                f"diagnostic {label} {observed or 'unavailable'} does not include {expected}"
            )
    if args.max_diagnostic_implicit_cut_fallback_cells is not None:
        fallback_cells = metrics.get("diagnostic_implicit_cut_fallback_cells")
        if not isinstance(fallback_cells, int):
            errors.append("diagnostic implicit-cut fallback cell count is unavailable")
        elif fallback_cells > args.max_diagnostic_implicit_cut_fallback_cells:
            errors.append(
                f"diagnostic implicit-cut fallback cells {fallback_cells} exceed "
                f"{args.max_diagnostic_implicit_cut_fallback_cells}"
            )
    for arg_name, metric_name, label in (
            ("min_diagnostic_achieved_interface_quadrature_order",
             "diagnostic_achieved_interface_quadrature_order_min",
             "achieved interface quadrature order"),
            ("min_diagnostic_achieved_volume_quadrature_order",
             "diagnostic_achieved_volume_quadrature_order_min",
             "achieved volume quadrature order")):
        minimum = getattr(args, arg_name)
        if minimum is None:
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, int):
            errors.append(f"diagnostic {label} is unavailable")
        elif value < minimum:
            errors.append(
                f"diagnostic {label} {value} is below {minimum}"
            )
    return errors


def curvature_projection_errors(metrics: dict[str, Any],
                                args: argparse.Namespace) -> list[str]:
    errors = []
    if args.require_curvature_projection_diagnostics and not metrics["diagnostics"].get(
            "curvature_projections"):
        errors.append("curvature projection diagnostics were not reported")
    if args.min_diagnostic_curvature_projection_count is not None:
        count = metrics.get("diagnostic_curvature_projection_count")
        if not isinstance(count, int):
            errors.append("curvature projection diagnostic count is unavailable")
        elif count < args.min_diagnostic_curvature_projection_count:
            errors.append(
                f"curvature projection diagnostic count {count} is below "
                f"{args.min_diagnostic_curvature_projection_count}"
            )
    if args.min_diagnostic_curvature_projection_max_abs_curvature is not None:
        value = metrics.get("diagnostic_curvature_projection_max_abs_curvature")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection max-abs-curvature diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_max_abs_curvature:
            errors.append(
                f"curvature projection max abs curvature {value:.6g} is below "
                f"{args.min_diagnostic_curvature_projection_max_abs_curvature:.6g}"
            )
    if args.max_diagnostic_curvature_projection_zero_fallback_vertices is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_zero_fallback_vertices")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection zero-fallback diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_zero_fallback_vertices:
            errors.append(
                f"curvature projection zero fallback vertices {value} exceed "
                f"{args.max_diagnostic_curvature_projection_zero_fallback_vertices}"
            )
    if args.max_diagnostic_curvature_projection_normalized_fit_residual is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_normalized_fit_residual")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection normalized fit residual diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_normalized_fit_residual:
            errors.append(
                f"curvature projection normalized fit residual {value:.6g} exceeds "
                f"{args.max_diagnostic_curvature_projection_normalized_fit_residual:.6g}"
            )
    if args.min_diagnostic_curvature_projection_smoothing_iterations is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_smoothing_iterations")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection smoothing diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_smoothing_iterations:
            errors.append(
                f"curvature projection smoothing iterations {value} are below "
                f"{args.min_diagnostic_curvature_projection_smoothing_iterations}"
            )
    if args.require_curvature_projection_newton_freshness:
        reason_counts = metrics.get("diagnostic_curvature_projection_reason_counts")
        if not isinstance(reason_counts, dict):
            errors.append("curvature projection reason-count diagnostics are unavailable")
            return errors
        summary = metrics.get("time_loop", {}).get("summary", {})
        accepted_steps = summary.get("accepted_steps") if isinstance(summary, dict) else None
        if not isinstance(accepted_steps, int) or accepted_steps <= 0:
            errors.append("curvature projection freshness requires accepted-step count")
            return errors
        required_reasons = {
            "initial": 1,
            "before_physics_solve": accepted_steps,
            "jacobian_and_residual": accepted_steps,
            "line_search_trial": accepted_steps,
            "accepted_step": accepted_steps,
        }
        for reason, minimum in required_reasons.items():
            count = reason_counts.get(reason, 0)
            if not isinstance(count, int) or count < minimum:
                errors.append(
                    f"curvature projection reason '{reason}' count {count} is below {minimum}"
                )
    return errors


def timeout_before_solution_state(diagnostics: dict[str, Any]) -> bool:
    summary = diagnostics.get("time_loop", {}).get("summary", {})
    if not isinstance(summary, dict):
        return False
    nonlinear_records = summary.get("nonlinear_records", 0)
    accepted_steps = summary.get("accepted_steps", 0)
    return int(nonlinear_records or 0) == 0 and int(accepted_steps or 0) == 0


def assembly_topology_consistency_errors(diagnostics: dict[str, Any]) -> list[str]:
    errors = []
    facet_counts = [
        int(record["cut_adjacent_facets"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("cut_adjacent_facets"), int)
    ]
    if diagnostics.get("interior_face_timings"):
        if not facet_counts:
            errors.append("cut-adjacent facet count is unavailable for interior-face timing checks")
        else:
            expected_facets = set(facet_counts)
            mismatched = [
                int(record["faces_assembled"])
                for record in diagnostics["interior_face_timings"]
                if isinstance(record.get("faces_assembled"), int) and
                int(record["faces_assembled"]) not in expected_facets
            ]
            if mismatched:
                errors.append(
                    "interior-face timing assembled counts do not match cut-adjacent facets "
                    f"(expected one of {sorted(expected_facets)}, examples {mismatched[:3]})"
                )

    assembly_records: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for record in diagnostics.get("cut_volume_assemblies", []):
        key = (record.get("marker"), record.get("side"))
        assembly_records.setdefault(key, []).append(record)

    for timing in diagnostics.get("cut_volume_timings", []):
        key = (timing.get("marker"), timing.get("side"))
        if key[0] is None or key[1] is None:
            errors.append("cut-volume timing record is missing marker or side")
            continue
        matches = assembly_records.get(key, [])
        if not matches:
            errors.append(f"cut-volume timing record has no assembly diagnostic for marker/side {key}")
            continue
        if timing.get("indexed") != 1:
            errors.append(f"cut-volume timing for marker/side {key} did not use indexed rule traversal")
        considered = timing.get("rules_considered")
        assembled = timing.get("rules_assembled")
        if isinstance(considered, int) and isinstance(assembled, int) and considered != assembled:
            errors.append(
                f"cut-volume timing for marker/side {key} considered {considered} rules but assembled {assembled}"
            )

        matched_counts = False
        for assembly in matches:
            if (assembled == assembly.get("rules") and
                    timing.get("full_rules") == assembly.get("full_cell_rules") and
                    timing.get("partial_rules") == assembly.get("cut_cell_rules")):
                matched_counts = True
                break
        if not matched_counts:
            errors.append(
                "cut-volume timing rule counts do not match cut-volume assembly diagnostics "
                f"for marker/side {key}"
            )
    return errors


def has_marked_interior_face_fallback_trace(diagnostics: dict[str, Any]) -> bool:
    return any(
        record.get("event") == "runtime_skip" and
        record.get("reason") == "marked_interior_face_fallback" and
        record.get("domain") == "InteriorFace"
        for record in diagnostics.get("jit_specialization_traces", [])
    )


def has_linear_solve_memory_diagnostics(diagnostics: dict[str, Any]) -> bool:
    phases = {
        record.get("phase")
        for record in diagnostics.get("process_memory", [])
        if record.get("phase") in {"before_linear_solve", "after_linear_solve"}
    }
    return {"before_linear_solve", "after_linear_solve"}.issubset(phases)


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
    implicit_fallback_cells = diagnostic_implicit_cut_fallback_cells(diagnostics)
    if implicit_fallback_cells is not None:
        metrics["diagnostic_implicit_cut_fallback_cells"] = implicit_fallback_cells
    for source, target in (
            ("achieved_interface_quadrature_order",
             "diagnostic_achieved_interface_quadrature_order_min"),
            ("achieved_volume_quadrature_order",
             "diagnostic_achieved_volume_quadrature_order_min")):
        value = diagnostic_cut_context_min_int(diagnostics, source)
        if value is not None:
            metrics[target] = value
    for source, target in (
            ("generated_interface_geometry",
             "diagnostic_generated_interface_geometry_counts"),
            ("implicit_cut_quadrature_backend",
             "diagnostic_implicit_cut_quadrature_backend_counts"),
            ("implicit_cut_fallback_policy",
             "diagnostic_implicit_cut_fallback_policy_counts")):
        counts = diagnostic_cut_context_value_counts(diagnostics, source)
        if counts:
            metrics[target] = top_counts(counts)
    selected_backend_counts = diagnostic_cut_context_summary_counts(
        diagnostics, "selected_implicit_cut_quadrature_backend_counts")
    if selected_backend_counts:
        metrics["diagnostic_selected_implicit_cut_quadrature_backend_counts"] = (
            top_counts(selected_backend_counts)
        )
    backend_qualification_counts = diagnostic_cut_context_summary_counts(
        diagnostics, "implicit_cut_backend_qualification_counts")
    if backend_qualification_counts:
        metrics["diagnostic_implicit_cut_backend_qualification_counts"] = (
            top_counts(backend_qualification_counts)
        )
    gauge_value = diagnostic_pressure_gauge_value(diagnostics)
    if gauge_value is not None:
        metrics["diagnostic_pressure_gauge_value"] = gauge_value
    if diagnostics.get("hydrostatic_initializations"):
        latest_hydrostatic = diagnostics["hydrostatic_initializations"][-1]
        metrics["latest_hydrostatic_initialization"] = latest_hydrostatic
        for name in (
            "wet_pressure_vertices",
            "dry_pressure_vertices",
            "gauge_constraints",
            "checked_gauge_constraints",
            "skipped_gauge_constraints",
            "initialized_pressure_min",
            "initialized_pressure_max",
            "wet_pressure_min",
            "wet_pressure_max",
            "gauge_pressure_min",
            "gauge_pressure_max",
            "gauge_initialized_pressure_min",
            "gauge_initialized_pressure_max",
            "gauge_pressure_max_abs_error",
        ):
            value = latest_hydrostatic.get(name)
            if isinstance(value, (int, float)):
                metrics[f"diagnostic_hydrostatic_{name}"] = value
    solution_source_summary = cut_context_solution_source_summary(diagnostics)
    if solution_source_summary:
        metrics["diagnostic_cut_context_solution_sources"] = solution_source_summary
    rebuild_provenance_counts = cut_context_rebuild_provenance_counts(diagnostics)
    if rebuild_provenance_counts:
        rebuild_count = sum(rebuild_provenance_counts.values())
        metrics["diagnostic_cut_context_rebuild_count"] = rebuild_count
        metrics["diagnostic_cut_context_rebuild_provenance_counts"] = (
            top_counts(rebuild_provenance_counts)
        )
        nonlinear_refresh_count = sum(
            count for provenance, count in rebuild_provenance_counts.items()
            if provenance in STATE_SYNC_CUT_CONTEXT_PROVENANCES
        )
        vector_refresh_count = sum(
            count for provenance, count in rebuild_provenance_counts.items()
            if provenance in VECTOR_CUT_CONTEXT_PROVENANCES
        )
        metrics["diagnostic_cut_context_nonlinear_refresh_count"] = (
            nonlinear_refresh_count
        )
        metrics["diagnostic_cut_context_vector_refresh_count"] = vector_refresh_count
    cell_cache_summary = generated_cell_cache_summary(diagnostics)
    if cell_cache_summary:
        metrics["diagnostic_generated_cell_cache_summary"] = cell_cache_summary
        metrics["diagnostic_generated_cell_cache_total_hits"] = (
            cell_cache_summary["total_hits"]
        )
        metrics["diagnostic_generated_cell_cache_total_misses"] = (
            cell_cache_summary["total_misses"]
        )
        metrics["diagnostic_generated_domain_cache_hits"] = (
            cell_cache_summary["domain_hits"]
        )
        metrics["diagnostic_generated_cell_cache_full_miss_rebuilds"] = (
            cell_cache_summary["full_miss_rebuilds"]
        )
    if diagnostics.get("level_set_maintenance"):
        maintenance = diagnostics["level_set_maintenance"]
        metrics["diagnostic_level_set_maintenance_count"] = len(maintenance)
        metrics["latest_level_set_maintenance"] = maintenance[-1]
    if diagnostics.get("level_set_nonconservative_warnings"):
        warnings = diagnostics["level_set_nonconservative_warnings"]
        metrics["diagnostic_level_set_nonconservative_warning_count"] = len(warnings)
        metrics["latest_level_set_nonconservative_warning"] = warnings[-1]
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
    if diagnostics.get("newton_assemblies"):
        records = diagnostics["newton_assemblies"]
        metrics["latest_newton_assembly"] = records[-1]
        metrics["diagnostic_newton_assembly_count"] = len(records)
        phase_counts: dict[str, int] = {}
        sync_point_counts: dict[str, int] = {}
        matrix_count = 0
        vector_count = 0
        post_first_iteration_matrix_count = 0
        for record in records:
            increment_count(
                phase_counts, str(record.get("phase", "unknown"))
            )
            increment_count(
                sync_point_counts, str(record.get("sync_point", "unknown"))
            )
            want_matrix = bool(record.get("want_matrix"))
            want_vector = bool(record.get("want_vector"))
            if want_matrix:
                matrix_count += 1
                iteration = record.get("iteration")
                if isinstance(iteration, (int, float)) and int(iteration) > 0:
                    post_first_iteration_matrix_count += 1
            if want_vector:
                vector_count += 1
        metrics["diagnostic_newton_assembly_phase_counts"] = (
            top_counts(phase_counts)
        )
        metrics["diagnostic_newton_assembly_sync_point_counts"] = (
            top_counts(sync_point_counts)
        )
        metrics["diagnostic_newton_matrix_assembly_count"] = matrix_count
        metrics["diagnostic_newton_vector_assembly_count"] = vector_count
        metrics[
            "diagnostic_newton_post_first_iteration_matrix_assembly_count"
        ] = post_first_iteration_matrix_count
    if diagnostics.get("jacobian_checks"):
        latest_jacobian_check = diagnostics["jacobian_checks"][-1]
        metrics["latest_jacobian_check"] = latest_jacobian_check
        value = latest_jacobian_check.get("rel")
        if isinstance(value, (int, float)):
            metrics["diagnostic_jacobian_check_relative_error"] = float(value)
    if diagnostics.get("jacobian_check_component_details"):
        details = diagnostics["jacobian_check_component_details"]
        metrics["latest_jacobian_check_component_details"] = details[-1]
        block_metrics = jacobian_component_block_metrics(details)
        filters = block_metrics.get("filters")
        if isinstance(filters, list):
            metrics["diagnostic_jacobian_component_sweep_filters"] = filters
            metrics["diagnostic_jacobian_component_sweep_count"] = len(filters)
        if "relative_errors" in block_metrics:
            metrics["diagnostic_jacobian_component_block_relative_errors"] = (
                block_metrics["relative_errors"]
            )
        if "matrix_relative_errors" in block_metrics:
            metrics["diagnostic_jacobian_component_block_matrix_relative_errors"] = (
                block_metrics["matrix_relative_errors"]
            )
        for source, target in (
                ("max_relative_error",
                 "diagnostic_jacobian_component_block_max_relative_error"),
                ("max_matrix_relative_error",
                 "diagnostic_jacobian_component_block_max_matrix_relative_error"),
                ("skipped_near_zero_blocks",
                 "diagnostic_jacobian_component_block_skipped_near_zero_count")):
            value = block_metrics.get(source)
            if isinstance(value, (int, float)):
                metrics[target] = value
    if diagnostics.get("jacobian_check_sweep_plans"):
        metrics["latest_jacobian_check_sweep_plan"] = (
            diagnostics["jacobian_check_sweep_plans"][-1]
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
    if diagnostics.get("timeloop_initialization_solves"):
        records = diagnostics["timeloop_initialization_solves"]
        metrics["latest_timeloop_initialization_solve"] = records[-1]
        metrics["diagnostic_timeloop_initialization_solve_count"] = len(records)
        for source, target in (
                ("dirichlet_dofs", "diagnostic_timeloop_initialization_max_dirichlet_dofs"),
                ("constraints", "diagnostic_timeloop_initialization_max_constraints"),
                ("rhs_norm", "diagnostic_timeloop_initialization_max_rhs_norm")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("fsils_prepared_matrices"):
        records = diagnostics["fsils_prepared_matrices"]
        latest_matrix = records[-1]
        metrics["latest_fsils_prepared_matrix"] = latest_matrix
        metrics["diagnostic_fsils_prepared_matrix_count"] = len(records)
        for source, target in (
                ("zero_rows", "diagnostic_fsils_prepared_matrix_max_zero_rows"),
                ("missing_diag", "diagnostic_fsils_prepared_matrix_max_missing_diag"),
                ("zero_diag", "diagnostic_fsils_prepared_matrix_max_zero_diag"),
                ("nonfinite_entries",
                 "diagnostic_fsils_prepared_matrix_max_nonfinite_entries"),
                ("max_row_sum_to_abs_diag",
                 "diagnostic_fsils_prepared_matrix_max_row_sum_to_abs_diag")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
        values = [
            record.get("min_abs_diag_to_row_sum")
            for record in records
            if isinstance(record.get("min_abs_diag_to_row_sum"), (int, float))
        ]
        if values:
            metrics["diagnostic_fsils_prepared_matrix_min_abs_diag_to_row_sum"] = min(values)
    if diagnostics.get("eigen_factorization_diagnostics"):
        records = diagnostics["eigen_factorization_diagnostics"]
        latest_eigen = records[-1]
        metrics["latest_eigen_factorization_diagnostic"] = latest_eigen
        metrics["diagnostic_eigen_factorization_count"] = len(records)
        for source, target in (
                ("zero_rows", "diagnostic_eigen_factorization_max_zero_rows"),
                ("zero_cols", "diagnostic_eigen_factorization_max_zero_cols"),
                ("nonfinite_entries", "diagnostic_eigen_factorization_max_nonfinite_entries"),
                ("pressure_zero_rows", "diagnostic_eigen_factorization_max_pressure_zero_rows"),
                ("pressure_zero_cols", "diagnostic_eigen_factorization_max_pressure_zero_cols")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
        for key in ("zero_row_runs", "zero_col_runs",
                    "pressure_zero_row_runs_local",
                    "pressure_zero_col_runs_local"):
            value = latest_eigen.get(key)
            if isinstance(value, str):
                metrics[f"diagnostic_eigen_factorization_latest_{key}"] = value
    if diagnostics.get("active_pressure_support_constraints"):
        records = diagnostics["active_pressure_support_constraints"]
        latest_support = records[-1]
        metrics["latest_active_pressure_support_constraint"] = latest_support
        metrics["diagnostic_active_pressure_support_constraint_count"] = len(records)
        for source, target in (
                ("active_support_cells",
                 "diagnostic_active_pressure_support_max_active_support_cells"),
                ("active_support_vertices",
                 "diagnostic_active_pressure_support_max_active_support_vertices"),
                ("inactive_vertices",
                 "diagnostic_active_pressure_support_max_inactive_vertices"),
                ("constrained_owned_dofs",
                 "diagnostic_active_pressure_support_max_constrained_owned_dofs"),
                ("inactive_sign_vertices_with_support",
                 "diagnostic_active_pressure_support_max_inactive_sign_vertices_with_support"),
                ("active_sign_vertices_without_support",
                 "diagnostic_active_pressure_support_max_active_sign_vertices_without_support")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
        value = latest_support.get("inactive_vertex_runs")
        if isinstance(value, str):
            metrics["diagnostic_active_pressure_support_latest_inactive_vertex_runs"] = value
    if diagnostics.get("curvature_projections"):
        records = diagnostics["curvature_projections"]
        metrics["latest_curvature_projection"] = records[-1]
        metrics["diagnostic_curvature_projection_count"] = len(records)
        field_counts: dict[str, int] = {}
        for record in records:
            increment_count(
                field_counts,
                str(record.get("curvature_field", "unknown")),
            )
        metrics["diagnostic_curvature_projection_field_counts"] = (
            top_counts(field_counts)
        )
        reason_counts: dict[str, int] = {}
        for record in records:
            increment_count(
                reason_counts,
                str(record.get("reason", "unknown")),
            )
        metrics["diagnostic_curvature_projection_reason_counts"] = (
            top_counts(reason_counts)
        )
        for source, target in (
                ("fitted_vertices",
                 "diagnostic_curvature_projection_max_fitted_vertices"),
                ("fallback_vertices",
                 "diagnostic_curvature_projection_max_fallback_vertices"),
                ("zero_fallback_vertices",
                 "diagnostic_curvature_projection_max_zero_fallback_vertices"),
                ("insufficient_stencil_vertices",
                 "diagnostic_curvature_projection_max_insufficient_stencil_vertices"),
                ("singular_stencil_vertices",
                 "diagnostic_curvature_projection_max_singular_stencil_vertices"),
                ("small_gradient_vertices",
                 "diagnostic_curvature_projection_max_small_gradient_vertices"),
                ("fit_residual_failure_vertices",
                 "diagnostic_curvature_projection_max_fit_residual_failure_vertices"),
                ("smoothing_iterations",
                 "diagnostic_curvature_projection_max_smoothing_iterations"),
                ("smoothing_mean_abs_update",
                 "diagnostic_curvature_projection_max_smoothing_mean_abs_update"),
                ("smoothing_max_abs_update",
                 "diagnostic_curvature_projection_max_smoothing_max_abs_update"),
                ("mean_normalized_fit_residual",
                 "diagnostic_curvature_projection_max_mean_normalized_fit_residual"),
                ("max_normalized_fit_residual",
                 "diagnostic_curvature_projection_max_normalized_fit_residual"),
                ("max_abs_curvature",
                 "diagnostic_curvature_projection_max_abs_curvature")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("level_set_volume_corrections"):
        records = diagnostics["level_set_volume_corrections"]
        metrics["latest_level_set_volume_correction"] = records[-1]
        metrics["diagnostic_level_set_volume_correction_count"] = len(records)
        for source, target in (
                ("achieved_volume_error",
                 "diagnostic_level_set_volume_correction_max_abs_achieved_error"),
                ("applied_shift_magnitude",
                 "diagnostic_level_set_volume_correction_max_shift_magnitude"),
                ("iterations",
                 "diagnostic_level_set_volume_correction_max_iterations")):
            values = [
                abs(record.get(source)) if source == "achieved_volume_error"
                else record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("jit_specialization_traces"):
        traces = diagnostics["jit_specialization_traces"]
        metrics["latest_jit_specialization_trace"] = traces[-1]
        event_counts: dict[str, int] = {}
        trigger_counts: dict[str, int] = {}
        event_trigger_domain_role_counts: dict[str, int] = {}
        event_reason_domain_counts: dict[str, int] = {}
        compile_domain_role_counts: dict[str, int] = {}
        runtime_compile_domain_role_counts: dict[str, int] = {}
        runtime_skip_reason_domain_counts: dict[str, int] = {}
        compile_shape_counts: dict[str, int] = {}
        generic_compile_kind_counts: dict[str, int] = {}
        for record in traces:
            event = record.get("event")
            if isinstance(event, str):
                event_counts[event] = event_counts.get(event, 0) + 1
            trigger = record.get("trigger")
            if isinstance(trigger, str):
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            increment_count(
                event_trigger_domain_role_counts,
                count_key(record, ("event", "trigger", "domain", "role")),
            )
            increment_count(
                event_reason_domain_counts,
                count_key(record, ("event", "reason", "domain")),
            )
            if event == "compile":
                increment_count(
                    compile_domain_role_counts,
                    count_key(record, ("domain", "role")),
                )
                increment_count(compile_shape_counts, jit_shape_key(record))
                if trigger == "runtime":
                    increment_count(
                        runtime_compile_domain_role_counts,
                        count_key(record, ("domain", "role")),
                    )
            elif event == "generic_compile":
                increment_count(generic_compile_kind_counts, count_key(record, ("kind",)))
            elif event == "runtime_skip":
                increment_count(
                    runtime_skip_reason_domain_counts,
                    count_key(record, ("reason", "domain")),
                )
        metrics["diagnostic_jit_specialization_trace_count"] = len(traces)
        metrics["diagnostic_jit_specialization_event_counts"] = event_counts
        metrics["diagnostic_jit_specialization_trigger_counts"] = trigger_counts
        metrics["diagnostic_jit_specialization_event_trigger_domain_role_counts"] = (
            top_counts(event_trigger_domain_role_counts)
        )
        metrics["diagnostic_jit_specialization_event_reason_domain_counts"] = (
            top_counts(event_reason_domain_counts)
        )
        metrics["diagnostic_jit_specialization_compile_domain_role_counts"] = (
            top_counts(compile_domain_role_counts)
        )
        metrics["diagnostic_jit_specialization_runtime_compile_domain_role_counts"] = (
            top_counts(runtime_compile_domain_role_counts)
        )
        metrics["diagnostic_jit_specialization_runtime_skip_reason_domain_counts"] = (
            top_counts(runtime_skip_reason_domain_counts)
        )
        metrics["diagnostic_jit_specialization_compile_shape_counts"] = (
            top_counts(compile_shape_counts)
        )
        metrics["diagnostic_jit_specialization_generic_compile_kind_counts"] = (
            top_counts(generic_compile_kind_counts)
        )
        metrics["diagnostic_jit_specialization_compile_count"] = sum(
            count for event, count in event_counts.items()
            if event in {"compile", "generic_compile"}
        )
    if diagnostics.get("assembly_timings"):
        timings = diagnostics["assembly_timings"]
        metrics["diagnostic_assembly_timing_count"] = len(timings)
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
    summary = diagnostics.get("time_loop", {}).get("summary", {})
    if isinstance(summary, dict):
        accepted_steps = summary.get("accepted_steps")
        nonlinear_iterations = summary.get("nonlinear_iterations_total")
        assembly_count = metrics.get("diagnostic_assembly_timing_count")
        cut_rebuild_count = metrics.get("diagnostic_cut_context_rebuild_count")
        newton_assembly_count = metrics.get("diagnostic_newton_assembly_count")
        newton_matrix_count = metrics.get("diagnostic_newton_matrix_assembly_count")
        if isinstance(accepted_steps, (int, float)) and accepted_steps > 0:
            if isinstance(assembly_count, (int, float)):
                metrics["diagnostic_assembly_timings_per_accepted_step"] = (
                    float(assembly_count) / float(accepted_steps)
                )
            if isinstance(newton_assembly_count, (int, float)):
                metrics["diagnostic_newton_assemblies_per_accepted_step"] = (
                    float(newton_assembly_count) / float(accepted_steps)
                )
            if isinstance(newton_matrix_count, (int, float)):
                metrics["diagnostic_newton_matrix_assemblies_per_accepted_step"] = (
                    float(newton_matrix_count) / float(accepted_steps)
                )
            if isinstance(cut_rebuild_count, (int, float)):
                metrics["diagnostic_cut_context_rebuilds_per_accepted_step"] = (
                    float(cut_rebuild_count) / float(accepted_steps)
                )
            if (isinstance(assembly_count, (int, float)) and
                    isinstance(nonlinear_iterations, (int, float))):
                extra_assemblies = int(assembly_count) - int(nonlinear_iterations)
                metrics["diagnostic_extra_assembly_timing_count_vs_nonlinear_iterations"] = (
                    extra_assemblies
                )
                metrics["diagnostic_extra_assembly_timings_per_accepted_step"] = (
                    float(extra_assemblies) / float(accepted_steps)
                )
    process_memory_records = list(diagnostics.get("process_memory", []))
    if process_memory_records:
        metrics["latest_process_memory"] = process_memory_records[-1]
        rss_values = [
            float(record["process_rss_kb"])
            for record in process_memory_records
            if isinstance(record.get("process_rss_kb"), (int, float))
        ]
        vm_values = [
            float(record["process_vm_kb"])
            for record in process_memory_records
            if isinstance(record.get("process_vm_kb"), (int, float))
        ]
        if rss_values:
            metrics["diagnostic_process_rss_kb"] = numeric_range(rss_values)
            metrics["diagnostic_process_max_rss_kb"] = max(rss_values)
            metrics["diagnostic_process_rss_growth_kb"] = rss_values[-1] - rss_values[0]
        if vm_values:
            metrics["diagnostic_process_vm_kb"] = numeric_range(vm_values)
            metrics["diagnostic_process_max_vm_kb"] = max(vm_values)
        basis_cache_values = [
            int(record["basis_cache_entries"])
            for record in process_memory_records
            if isinstance(record.get("basis_cache_entries"), (int, float))
        ]
        if basis_cache_values:
            metrics["diagnostic_process_max_basis_cache_entries"] = max(basis_cache_values)
            metrics["diagnostic_process_basis_cache_entry_growth"] = (
                basis_cache_values[-1] - basis_cache_values[0]
            )
    if diagnostics.get("jit_cache_diagnostics"):
        jit_records = diagnostics["jit_cache_diagnostics"]
        metrics["latest_jit_cache_diagnostics"] = jit_records[-1]
        for name in (
            "kernel_cache_size",
            "kernel_cache_hits",
            "kernel_cache_misses",
            "kernel_cache_symbol_hits",
            "kernel_cache_stores",
            "kernel_cache_evictions",
            "object_cache_entries",
            "object_cache_notify_compiled",
            "object_cache_gets",
            "object_cache_mem_hits",
            "object_cache_disk_hits",
            "object_cache_misses",
            "object_cache_bytes_written",
            "object_cache_bytes_read",
        ):
            values = [
                float(record[name])
                for record in jit_records
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_jit_cache_max_{name}"] = max(values)
    if diagnostics.get("interior_face_timings"):
        timings = diagnostics["interior_face_timings"]
        metrics["latest_interior_face_timing"] = timings[-1]
        metrics["diagnostic_interior_face_timing_by_mode"] = summarize_timing_modes(
            timings,
            ("faces_considered", "faces_assembled"),
            (
                "total",
                "kernel",
                "insert",
                "prepare_minus",
                "prepare_plus",
                "solution",
                "field",
            ),
        )
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
    if diagnostics.get("cut_volume_timings"):
        timings = diagnostics["cut_volume_timings"]
        metrics["latest_cut_volume_timing"] = timings[-1]
        metrics["diagnostic_cut_volume_timing_by_mode"] = summarize_timing_modes(
            timings,
            ("rules_considered", "rules_assembled", "full_rules", "partial_rules", "qpts"),
            (
                "total",
                "kernel",
                "insert",
                "rule",
                "geometry",
                "basis",
                "solution",
                "field",
            ),
        )
        for name in (
            "indexed",
            "rules_considered",
            "rules_assembled",
            "full_rules",
            "partial_rules",
            "qpts",
        ):
            values = [
                int(record[name])
                for record in timings
                if isinstance(record.get(name), int)
            ]
            if values:
                metrics[f"diagnostic_cut_volume_timing_max_{name}"] = max(values)
        for name in (
            "total",
            "setup",
            "filter",
            "dofs",
            "rule",
            "geometry",
            "basis",
            "frame",
            "context",
            "jit",
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
                metrics[f"diagnostic_cut_volume_timing_max_{name}_seconds"] = max(values)
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
    context_volumes = metrics.get("cut_context_active_side_physical_volumes", [])
    if isinstance(wet_fraction_volume, (int, float)) and context_volumes:
        metrics["wet_fraction_volume_drift_vs_initial_physical_cut_context"] = (
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
        "linear_algebra_backend",
        "linear_preconditioner",
        "generated_interface_geometry",
        "implicit_cut_quadrature_backend",
        "implicit_cut_fallback_policy",
        "required_implicit_cut_backend_qualification",
        "implicit_cut_root_tolerance",
        "implicit_cut_max_subdivision_depth",
        "generated_interface_quadrature_order",
        "interface_quadrature_order",
        "volume_quadrature_order",
        "surface_tension",
        "projected_curvature_field",
        "curvature_projection_cadence_steps",
        "curvature_projection_smoothing_iterations",
        "curvature_projection_smoothing_relaxation",
        "enable_level_set_volume_correction",
        "volume_correction_cadence_steps",
        "volume_correction_use_initial_volume",
        "volume_correction_tolerance",
        "volume_correction_max_iterations",
        "mpi_ranks",
        "mms_nx",
        "mms_ny",
        "max_diagnostic_implicit_cut_fallback_cells",
        "min_diagnostic_achieved_interface_quadrature_order",
        "min_diagnostic_achieved_volume_quadrature_order",
        "expect_generated_interface_geometry",
        "expect_implicit_cut_quadrature_backend",
        "expect_selected_implicit_cut_quadrature_backend",
        "expect_implicit_cut_backend_qualification",
        "expect_implicit_cut_fallback_policy",
    ):
        value = getattr(args, name)
        if value is not None:
            metrics[name] = value
    for name in (
        "enable_jacobian_check",
        "enable_newton_direction_check",
        "enable_newton_assembly_diagnostics",
        "newton_line_search_fail_on_no_reduction",
        "disable_cut_stabilization",
        "enable_linear_solve_history",
        "enable_linear_solve_component_norms",
        "enable_fsils_matrix_diagnostics",
        "enable_form_block_diagnostics",
        "enable_interior_face_timing",
        "enable_cut_volume_timing",
        "enable_jit_specialization_trace",
        "require_cut_context_solution_source_diagnostics",
        "require_newton_assembly_diagnostics",
        "require_assembly_timing_diagnostics",
        "require_interior_face_timing_diagnostics",
        "require_cut_volume_timing_diagnostics",
        "require_jit_specialization_trace_diagnostics",
        "require_process_memory_diagnostics",
        "require_basis_cache_diagnostics",
        "require_marked_interior_face_fallback_diagnostics",
        "require_jacobian_component_block_diagnostics",
        "require_eigen_factorization_diagnostics",
        "require_active_pressure_support_diagnostics",
        "require_curvature_projection_diagnostics",
        "require_curvature_projection_newton_freshness",
        "require_fsils_matrix_diagnostics",
        "require_assembly_topology_consistency",
        "require_high_order_cut_context_diagnostics",
        "high_order_production_qualification",
        "high_order_mpi_production_qualification",
        "high_order_3d_benchmark_smoke",
        "high_order_3d_benchmark_qualification",
        "high_order_3d_benchmark_profile_qualification",
        "high_order_curved_3d_simplex_smoke",
        "high_order_mpi_motion_smoke",
        "use_high_order_implicit_cuts",
        "require_reference_profile_comparison",
        "enable_adaptive_time_loop",
        "allow_experimental_profile_linear_solver",
        "allow_failure_diagnostics",
    ):
        if getattr(args, name):
            metrics[name] = True
    for name in (
        "jacobian_check_iteration",
        "jacobian_check_step",
        "jacobian_check_scheme",
        "jacobian_check_components",
        "jacobian_check_component_sweeps",
        "linear_solve_history_max_calls",
        "linear_solve_component_norms_max_newton_it",
        "newton_line_search_max_iterations",
        "max_diagnostic_assembly_timings_per_step",
        "max_diagnostic_extra_assembly_timings_per_step",
        "max_diagnostic_cut_context_rebuilds_per_step",
        "max_diagnostic_newton_matrix_assemblies_per_step",
        "max_diagnostic_generated_cell_cache_full_miss_rebuilds",
        "max_diagnostic_process_rss_kb",
        "max_diagnostic_process_rss_growth_kb",
        "max_diagnostic_process_basis_cache_entries",
        "max_diagnostic_process_basis_cache_entry_growth",
        "max_wet_fraction_volume_error",
        "max_reference_profile_rmse",
        "max_reference_profile_mae",
        "max_reference_profile_max_abs_error",
        "max_reference_profile_elevated_front_lag",
        "max_solver_elapsed_wall_seconds",
        "curvature_projection_smoothing_iterations",
        "curvature_projection_smoothing_relaxation",
        "min_diagnostic_curvature_projection_count",
        "min_diagnostic_curvature_projection_max_abs_curvature",
        "max_diagnostic_curvature_projection_zero_fallback_vertices",
        "max_diagnostic_curvature_projection_normalized_fit_residual",
        "min_diagnostic_curvature_projection_smoothing_iterations",
        "max_solver_elapsed_seconds_per_accepted_step",
        "min_reference_profile_coverage",
        "min_reference_profile_direct_coverage",
        "max_fsils_matrix_zero_rows",
        "max_fsils_matrix_missing_diag",
        "max_fsils_matrix_zero_diag",
        "max_fsils_matrix_nonfinite_entries",
        "max_eigen_factorization_pressure_zero_cols",
        "max_time_loop_nonlinear_iterations_per_step",
        "max_time_loop_linear_iterations_per_step",
        "min_interface_height_change",
        "min_interface_mean_abs_height_change",
        "min_interface_slope_change",
        "min_interface_final_height_span",
        "cut_cell_velocity_gradient_penalty",
        "cut_cell_pressure_gradient_penalty",
        "reference_profile_sample_radius",
        "reference_profile_elevated_front_clearance",
        "adaptive_time_loop_min_dt",
        "adaptive_time_loop_max_dt",
        "adaptive_time_loop_max_retries",
        "adaptive_time_loop_decrease_factor",
        "adaptive_time_loop_increase_factor",
        "adaptive_time_loop_target_newton_iterations",
        "adaptive_time_loop_max_steps_multiplier",
    ):
        value = getattr(args, name)
        if value is not None:
            metrics[name] = value


def parse_active_volume_history_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        key: diagnostics[key]
        for key in (
            "cut_context_active_side_volumes",
            "assembly_active_wet_volumes",
            "cut_context_active_side_volume_change",
            "assembly_active_wet_volume_change",
        )
        if key in diagnostics
    }
    physical_volumes = diagnostic_context_active_side_physical_volumes(diagnostics)
    if physical_volumes:
        metrics["cut_context_active_side_physical_volumes"] = physical_volumes
        metrics["cut_context_active_side_physical_volume_change"] = value_span(
            physical_volumes
        )
    return metrics


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


def accepted_step_count_for_elapsed_budget(metrics: dict[str, Any]) -> int | None:
    time_loop = metrics.get("time_loop")
    if not isinstance(time_loop, dict):
        diagnostics = metrics.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            time_loop = diagnostics.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if isinstance(summary, dict):
        accepted_steps = summary.get("accepted_steps")
        if isinstance(accepted_steps, int) and accepted_steps > 0:
            return accepted_steps
    result_step = metrics.get("result_step")
    if isinstance(result_step, int) and result_step > 0:
        return result_step
    return None


def solver_elapsed_time_errors(metrics: dict[str, Any],
                               args: argparse.Namespace) -> list[str]:
    errors = []
    max_wall_seconds = getattr(args, "max_solver_elapsed_wall_seconds", None)
    max_seconds_per_step = getattr(
        args, "max_solver_elapsed_seconds_per_accepted_step", None)
    if max_wall_seconds is None and max_seconds_per_step is None:
        return errors
    elapsed = metrics.get("solver_elapsed_wall_seconds")
    if not isinstance(elapsed, (int, float)):
        return ["solver elapsed wall time was not reported"]
    if max_wall_seconds is not None and float(elapsed) > max_wall_seconds:
        errors.append(
            f"solver elapsed wall time {float(elapsed):.3f}s exceeds "
            f"{max_wall_seconds:.3f}s"
        )
    if max_seconds_per_step is not None:
        accepted_steps = accepted_step_count_for_elapsed_budget(metrics)
        if accepted_steps is None:
            errors.append("accepted-step count is unavailable for elapsed-time budget")
        else:
            seconds_per_step = float(elapsed) / float(accepted_steps)
            metrics["solver_elapsed_seconds_per_accepted_step"] = seconds_per_step
            if seconds_per_step > max_seconds_per_step:
                errors.append(
                    "solver elapsed time per accepted step "
                    f"{seconds_per_step:.3f}s exceeds "
                    f"{max_seconds_per_step:.3f}s"
                )
    return errors


def evaluate_timeout_diagnostics(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics["diagnostics"]
    errors.extend(solver_elapsed_time_errors(metrics, args))
    errors.extend(time_loop_convergence_errors(metrics, args))
    gauge_required = metrics.get("case") in {"d18", "d38", "mini2d", "static2d"}
    pre_solution_timeout = timeout_before_solution_state(diagnostics)
    if not diagnostics.get("cut_context_rebuilds"):
        errors.append("cut-context rebuild diagnostics were not reported")
    if not diagnostics.get("cut_volume_assemblies"):
        errors.append("cut-volume assembly diagnostics were not reported")
    if gauge_required and not (
            diagnostics.get("pressure_gauge_checks") or
            diagnostics.get("hydrostatic_initializations")):
        errors.append("pressure-gauge or hydrostatic initialization diagnostics were not reported")
    if gauge_required and not diagnostics.get("hydrostatic_initializations"):
        errors.append("hydrostatic initialization diagnostics were not reported")
    if not pre_solution_timeout and not latest_component_record(diagnostics, "solution_state"):
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
    if args.require_jacobian_component_block_diagnostics:
        if not diagnostics.get("jacobian_check_component_details"):
            errors.append("Jacobian component-block diagnostics were not reported")
        expected_filters = normalized_component_sweeps(args.jacobian_check_component_sweeps)
        actual_filters = metrics.get("diagnostic_jacobian_component_sweep_filters", [])
        if expected_filters:
            missing_filters = [
                label for label in expected_filters
                if label not in actual_filters
            ]
            if missing_filters:
                errors.append(
                    "Jacobian component-block diagnostics are missing sweep filter(s): "
                    + ", ".join(missing_filters)
                )
    if args.require_linear_solve_history_diagnostics and not diagnostics.get("linear_solve_histories"):
        errors.append("linear solve history diagnostics were not reported")
    if args.require_form_block_diagnostics and (
            not diagnostics.get("form_block_installs") or not diagnostics.get("form_mixed_plans")):
        errors.append("form block installation diagnostics were not reported")
    if args.require_cut_context_solution_source_diagnostics:
        errors.extend(cut_context_solution_source_errors(diagnostics))
    errors.extend(cut_context_policy_errors(metrics, args))
    errors.extend(curvature_projection_errors(metrics, args))
    if (args.require_newton_assembly_diagnostics and
            not diagnostics.get("newton_assemblies")):
        errors.append("Newton assembly diagnostics were not reported")
    if args.require_assembly_timing_diagnostics and not diagnostics.get("assembly_timings"):
        errors.append("assembly timing diagnostics were not reported")
    errors.extend(assembly_efficiency_errors(metrics, args))
    if args.require_process_memory_diagnostics:
        has_process_memory = (
            diagnostics.get("process_memory") or
            any(
                isinstance(record.get("process_rss_kb"), (int, float))
                for record in diagnostics.get("cut_context_rebuilds", [])
            )
        )
        if not has_process_memory:
            errors.append("process memory diagnostics were not reported")
    if (args.require_linear_solve_memory_diagnostics and
            not has_linear_solve_memory_diagnostics(diagnostics)):
        errors.append("linear-solve memory diagnostics were not reported")
    if (args.require_timeloop_initialization_diagnostics and
            not diagnostics.get("timeloop_initialization_solves")):
        errors.append("TimeLoop initialization linear-solve diagnostics were not reported")
    if (args.require_fsils_matrix_diagnostics and
            not diagnostics.get("fsils_prepared_matrices")):
        errors.append("FSILS prepared-matrix diagnostics were not reported")
    if args.max_fsils_matrix_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("FSILS prepared-matrix zero-row diagnostics are unavailable")
        elif zero_rows > args.max_fsils_matrix_zero_rows:
            errors.append(
                f"FSILS prepared-matrix zero rows {zero_rows} exceed "
                f"{args.max_fsils_matrix_zero_rows}"
            )
    if args.max_fsils_matrix_missing_diag is not None:
        missing_diag = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_missing_diag"
        )
        if not isinstance(missing_diag, (int, float)):
            errors.append("FSILS prepared-matrix missing-diagonal diagnostics are unavailable")
        elif missing_diag > args.max_fsils_matrix_missing_diag:
            errors.append(
                f"FSILS prepared-matrix missing diagonals {missing_diag} exceed "
                f"{args.max_fsils_matrix_missing_diag}"
            )
    if args.max_fsils_matrix_zero_diag is not None:
        zero_diag = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_diag")
        if not isinstance(zero_diag, (int, float)):
            errors.append("FSILS prepared-matrix zero-diagonal diagnostics are unavailable")
        elif zero_diag > args.max_fsils_matrix_zero_diag:
            errors.append(
                f"FSILS prepared-matrix zero diagonals {zero_diag} exceed "
                f"{args.max_fsils_matrix_zero_diag}"
            )
    if args.max_fsils_matrix_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("FSILS prepared-matrix nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_fsils_matrix_nonfinite_entries:
            errors.append(
                f"FSILS prepared-matrix nonfinite entries {nonfinite} exceed "
                f"{args.max_fsils_matrix_nonfinite_entries}"
            )
    if args.require_basis_cache_diagnostics:
        has_basis_cache = any(
            isinstance(record.get("basis_cache_entries"), (int, float))
            for record in diagnostics.get("process_memory", [])
        )
        if not has_basis_cache:
            errors.append("basis-cache diagnostics were not reported")
    if args.max_diagnostic_process_basis_cache_entries is not None:
        basis_cache_entries = metrics.get("diagnostic_process_max_basis_cache_entries")
        if not isinstance(basis_cache_entries, (int, float)):
            errors.append("basis-cache entry diagnostics are unavailable")
        elif basis_cache_entries > args.max_diagnostic_process_basis_cache_entries:
            errors.append(
                f"basis-cache entries {basis_cache_entries} exceed "
                f"{args.max_diagnostic_process_basis_cache_entries}"
            )
    errors.extend(resource_ceiling_errors(metrics, args))
    if args.require_interior_face_timing_diagnostics and not diagnostics.get("interior_face_timings"):
        errors.append("interior-face timing diagnostics were not reported")
    if args.require_cut_volume_timing_diagnostics and not diagnostics.get("cut_volume_timings"):
        errors.append("cut-volume timing diagnostics were not reported")
    if args.require_jit_specialization_trace_diagnostics and not diagnostics.get("jit_specialization_traces"):
        errors.append("JIT specialization trace diagnostics were not reported")
    if args.require_jit_cache_diagnostics and not diagnostics.get("jit_cache_diagnostics"):
        errors.append("JIT cache diagnostics were not reported")
    if (args.require_marked_interior_face_fallback_diagnostics and
            not has_marked_interior_face_fallback_trace(diagnostics)):
        errors.append("marked interior-face fallback diagnostics were not reported")
    if args.require_assembly_topology_consistency:
        errors.extend(assembly_topology_consistency_errors(diagnostics))
    if (args.require_eigen_factorization_diagnostics and
            not diagnostics.get("eigen_factorization_diagnostics")):
        errors.append("Eigen factorization diagnostics were not reported")
    if (args.require_active_pressure_support_diagnostics and
            not diagnostics.get("active_pressure_support_constraints")):
        errors.append("active pressure support diagnostics were not reported")
    if args.max_eigen_factorization_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_eigen_factorization_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("Eigen factorization zero-row diagnostics are unavailable")
        elif zero_rows > args.max_eigen_factorization_zero_rows:
            errors.append(
                f"Eigen factorization zero rows {zero_rows} exceed "
                f"{args.max_eigen_factorization_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_rows is not None:
        pressure_zero_rows = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_rows"
        )
        if not isinstance(pressure_zero_rows, (int, float)):
            errors.append("Eigen factorization pressure zero-row diagnostics are unavailable")
        elif pressure_zero_rows > args.max_eigen_factorization_pressure_zero_rows:
            errors.append(
                f"Eigen factorization pressure zero rows {pressure_zero_rows} exceed "
                f"{args.max_eigen_factorization_pressure_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_cols is not None:
        pressure_zero_cols = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_cols"
        )
        if not isinstance(pressure_zero_cols, (int, float)):
            errors.append("Eigen factorization pressure zero-column diagnostics are unavailable")
        elif pressure_zero_cols > args.max_eigen_factorization_pressure_zero_cols:
            errors.append(
                f"Eigen factorization pressure zero columns {pressure_zero_cols} exceed "
                f"{args.max_eigen_factorization_pressure_zero_cols}"
            )
    if args.max_eigen_factorization_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_eigen_factorization_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("Eigen factorization nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_eigen_factorization_nonfinite_entries:
            errors.append(
                f"Eigen factorization nonfinite entries {nonfinite} exceed "
                f"{args.max_eigen_factorization_nonfinite_entries}"
            )

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
    if args.max_jacobian_component_block_relative_error is not None:
        value = metrics.get("diagnostic_jacobian_component_block_max_relative_error")
        if not isinstance(value, (int, float)):
            errors.append("Jacobian component-block relative error is unavailable")
        elif value > args.max_jacobian_component_block_relative_error:
            errors.append(
                f"Jacobian component-block relative error {value:.6g} exceeds "
                f"{args.max_jacobian_component_block_relative_error:.6g}"
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


def interface_profile_xy(dataset: pv.DataSet) -> tuple[np.ndarray, np.ndarray] | None:
    if "phi" not in dataset.point_data:
        return None
    try:
        interface = dataset.contour(isosurfaces=[0.0], scalars="phi")
    except Exception:
        return None
    points = np.asarray(interface.points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None
    x = points[:, 0]
    y = points[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    x = x[finite]
    y = y[finite]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    unique_x: list[float] = []
    averaged_y: list[float] = []
    start = 0
    tolerance = 1.0e-12
    while start < x.size:
        end = start + 1
        while end < x.size and abs(float(x[end] - x[start])) <= tolerance:
            end += 1
        unique_x.append(float(np.mean(x[start:end])))
        averaged_y.append(float(np.mean(y[start:end])))
        start = end
    if len(unique_x) < 2:
        return None
    return np.asarray(unique_x, dtype=float), np.asarray(averaged_y, dtype=float)


def add_interface_profile_summary(metrics: dict[str, Any],
                                  prefix: str,
                                  profile: tuple[np.ndarray, np.ndarray] | None) -> None:
    if profile is None:
        metrics[f"{prefix}_interface_available"] = False
        return
    x, y = profile
    metrics[f"{prefix}_interface_available"] = True
    metrics[f"{prefix}_interface_points"] = int(x.size)
    metrics[f"{prefix}_interface_x_min"] = float(np.min(x))
    metrics[f"{prefix}_interface_x_max"] = float(np.max(x))
    metrics[f"{prefix}_interface_height_min"] = float(np.min(y))
    metrics[f"{prefix}_interface_height_max"] = float(np.max(y))
    metrics[f"{prefix}_interface_height_mean"] = float(np.mean(y))
    metrics[f"{prefix}_interface_height_span"] = float(np.max(y) - np.min(y))
    if np.max(x) > np.min(x):
        slope, intercept = np.polyfit(x, y, 1)
        metrics[f"{prefix}_interface_slope"] = float(slope)
        metrics[f"{prefix}_interface_intercept"] = float(intercept)


def add_interface_motion_metrics(metrics: dict[str, Any],
                                 initial: pv.DataSet,
                                 output: pv.DataSet) -> None:
    initial_profile = interface_profile_xy(initial)
    final_profile = interface_profile_xy(output)
    add_interface_profile_summary(metrics, "initial", initial_profile)
    add_interface_profile_summary(metrics, "final", final_profile)
    if initial_profile is None or final_profile is None:
        metrics["interface_motion_available"] = False
        return

    initial_x, initial_y = initial_profile
    final_x, final_y = final_profile
    x_min = max(float(np.min(initial_x)), float(np.min(final_x)))
    x_max = min(float(np.max(initial_x)), float(np.max(final_x)))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        metrics["interface_motion_available"] = False
        metrics["interface_motion_unavailable_reason"] = "profiles_do_not_overlap_in_x"
        return

    sample_count = min(201, max(25, int(min(initial_x.size, final_x.size) * 4)))
    sample_x = np.linspace(x_min, x_max, sample_count)
    initial_sample_y = np.interp(sample_x, initial_x, initial_y)
    final_sample_y = np.interp(sample_x, final_x, final_y)
    delta = final_sample_y - initial_sample_y

    metrics["interface_motion_available"] = True
    metrics["interface_motion_sample_count"] = int(sample_count)
    metrics["interface_motion_x_min"] = float(x_min)
    metrics["interface_motion_x_max"] = float(x_max)
    metrics["interface_height_max_abs_change"] = float(np.max(np.abs(delta)))
    metrics["interface_height_mean_abs_change"] = float(np.mean(np.abs(delta)))
    metrics["interface_height_rms_change"] = float(np.sqrt(np.mean(delta * delta)))
    metrics["interface_height_signed_mean_change"] = float(np.mean(delta))
    metrics["interface_height_change_min"] = float(np.min(delta))
    metrics["interface_height_change_max"] = float(np.max(delta))

    initial_slope = metrics.get("initial_interface_slope")
    final_slope = metrics.get("final_interface_slope")
    if isinstance(initial_slope, (int, float)) and isinstance(final_slope, (int, float)):
        metrics["interface_slope_change"] = float(final_slope) - float(initial_slope)
        metrics["interface_slope_abs_change"] = abs(float(final_slope) - float(initial_slope))


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

    def mean_velocity(region: np.ndarray) -> list[float]:
        if not np.any(region):
            return [0.0 for _ in range(velocity.shape[1])]
        return [float(value) for value in np.nanmean(velocity[region], axis=0)]

    metrics: dict[str, Any] = {
        "result": str(result),
        "max_speed": float(np.nanmax(wet_speed)),
        "wet_mean_speed": float(np.nanmean(wet_speed)),
        "wet_mean_velocity": [float(value) for value in np.nanmean(wet_velocity, axis=0)],
        "gate_mean_velocity": mean_velocity(gate_region),
        "front_mean_velocity": mean_velocity(front_region),
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
    add_interface_motion_metrics(metrics, initial, output)
    if "WetVolumeMeasure" in output.cell_data:
        wet_measures = np.asarray(output.cell_data["WetVolumeMeasure"], dtype=float).reshape(-1)
        if wet_measures.shape[0] == output.n_cells:
            metrics["wet_fraction_volume"] = float(np.sum(wet_measures))
            metrics["wet_fraction_volume_source"] = "WetVolumeMeasure"
            metrics["wet_volume_measure_cell_count"] = int(wet_measures.shape[0])
            metrics["wet_volume_measure_min"] = float(np.min(wet_measures))
            metrics["wet_volume_measure_max"] = float(np.max(wet_measures))
    if "WetVolumeFraction" in output.cell_data:
        fractions = np.asarray(output.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
        measures = cell_measure(output)
        if fractions.shape[0] == measures.shape[0]:
            metrics["wet_fraction_cell_count"] = int(fractions.shape[0])
            if "wet_fraction_volume" not in metrics:
                metrics["wet_fraction_volume"] = float(np.sum(fractions * measures))
                metrics["wet_fraction_volume_source"] = "WetVolumeFraction"
            metrics["wet_fraction_min"] = float(np.min(fractions))
            metrics["wet_fraction_max"] = float(np.max(fractions))
    metrics.update(pressure_gauge_metrics(output, benchmark))
    metrics.update(mms_verification_metrics(case_name, case_dir, result))
    return metrics


def mms_verification_metrics(case_name: str,
                             case_dir: Path,
                             result: Path) -> dict[str, Any]:
    if case_name != "mms2d":
        return {}
    verifier = case_dir / "verify_expected_results.py"
    if not verifier.exists():
        return {
            "mms_verification_available": False,
            "mms_verification_error": f"missing verifier {verifier}",
        }
    completed = subprocess.run(
        [sys.executable, str(verifier), str(result)],
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    metrics: dict[str, Any] = {
        "mms_verification_available": True,
        "mms_verification_returncode": completed.returncode,
    }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        metrics["mms_verification_passed"] = False
        metrics["mms_verification_stdout_tail"] = "\n".join(
            completed.stdout.splitlines()[-40:])
        return metrics

    metrics["mms_verification"] = payload
    metrics["mms_verification_passed"] = (
        completed.returncode == 0 and bool(payload.get("passed", False)))
    failed_checks = payload.get("failed_checks", [])
    if isinstance(failed_checks, list):
        metrics["mms_verification_failed_checks"] = failed_checks
    for key in (
            "phi_rms_error",
            "phi_max_abs_error",
            "interface_shift_error",
            "interface_l2_height_error",
            "area_relative_error",
            "centroid_y_error",
            "velocity_relative_l2_error",
            "pressure_relative_rms_error",
            "pressure_relative_rms_error_after_constant_offset_removal",
            "interface_pressure_max_abs",
            "manufactured_residual_x_max",
            "manufactured_residual_y_max",
            "level_set_residual_max"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            metrics[f"mms_{key}"] = float(value)
    return metrics


def solver_time_step_size(case_dir: Path) -> float | None:
    try:
        root = ET.parse(case_dir / "solver.xml").getroot()
    except Exception:
        return None
    raw = text(root, "./GeneralSimulationParameters/Time_step_size")
    try:
        return float(raw)
    except ValueError:
        return None


def reference_profile_for_time(
        benchmark: dict[str, Any],
        final_time: float,
        tolerance: float) -> dict[str, Any] | None:
    profiles = benchmark.get("reference_profiles")
    if not isinstance(profiles, list):
        return None
    candidates = [
        profile for profile in profiles
        if isinstance(profile, dict) and
        isinstance(profile.get("time_s"), (int, float)) and
        isinstance(profile.get("path"), str)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda profile: abs(float(profile["time_s"]) - final_time))
    if abs(float(best["time_s"]) - final_time) > tolerance:
        return None
    return best


def add_reference_profile_metrics(metrics: dict[str, Any],
                                  case_dir: Path,
                                  result: Path,
                                  args: argparse.Namespace) -> None:
    if not args.require_reference_profile_comparison:
        return
    benchmark = load_benchmark(case_dir)
    dt = args.time_step_size
    if dt is None:
        dt = solver_time_step_size(case_dir)
    if dt is None:
        metrics["reference_profile_error"] = "solver time step size is unavailable"
        return

    final_time = float(args.steps) * float(dt)
    tolerance = (
        args.reference_profile_time_tolerance
        if args.reference_profile_time_tolerance is not None
        else max(1.0e-12, 0.5 * float(dt))
    )
    profile = reference_profile_for_time(benchmark, final_time, tolerance)
    if profile is None:
        metrics["reference_profile_error"] = (
            f"no reference profile within {tolerance:.6g}s of final time "
            f"{final_time:.6g}s"
        )
        metrics["reference_profile_time_s"] = final_time
        return

    reference_path = ROOT / str(profile["path"])
    metrics["reference_profile_time_s"] = final_time
    metrics["reference_profile_target_time_s"] = float(profile["time_s"])
    metrics["reference_profile_path"] = str(reference_path)
    try:
        import compare_test05_profiles as test05_profiles

        report = test05_profiles.compare(argparse.Namespace(
            result=result,
            reference_profile=reference_path,
            scalar="phi",
            benchmark_json=case_dir / "benchmark.json",
            density=1000.0,
            initial_wet_volume=None,
            initial_kinetic_energy=0.0,
            front_diagnostic_only=False,
            stale_pressure_gauge_tolerance=None,
            min_velocity_max=None,
            x_min=None,
            x_max=None,
            sample_radius=args.reference_profile_sample_radius,
            elevated_front_clearance=(
                args.reference_profile_elevated_front_clearance
                if args.reference_profile_elevated_front_clearance is not None
                else 0.005
            ),
            max_elevated_front_lag=args.max_reference_profile_elevated_front_lag,
            plot_output=None,
            output=None,
        ))
    except Exception as exc:
        metrics["reference_profile_error"] = str(exc)
        return

    validation = report.get("validation", {})
    if isinstance(validation, dict):
        metrics["reference_profile_validation_passed"] = bool(
            validation.get("passed", False))
        failures = validation.get("failures", [])
        if isinstance(failures, list):
            metrics["reference_profile_validation_failures"] = failures

    comparison = report.get("profile_comparison", {})
    if not isinstance(comparison, dict):
        return
    profile_metrics = comparison.get("metrics", {})
    if not isinstance(profile_metrics, dict):
        return
    for key, value in profile_metrics.items():
        if isinstance(value, (int, float, str)) or value is None:
            metrics[f"reference_profile_{key}"] = value


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = []
    if metrics.get("output_metrics_skipped"):
        return evaluate_timeout_diagnostics(metrics, args)
    errors.extend(solver_elapsed_time_errors(metrics, args))
    errors.extend(time_loop_convergence_errors(metrics, args))
    if args.require_mms_verification:
        if not metrics.get("mms_verification_available"):
            errors.append("MMS verification was not available")
        elif not metrics.get("mms_verification_passed"):
            failed_checks = metrics.get("mms_verification_failed_checks", [])
            if isinstance(failed_checks, list) and failed_checks:
                errors.append(
                    "MMS verification failed check(s): " +
                    ", ".join(str(item) for item in failed_checks))
            else:
                errors.append("MMS verification did not pass")
    if args.require_cut_context_solution_source_diagnostics:
        errors.extend(cut_context_solution_source_errors(metrics["diagnostics"]))
    errors.extend(cut_context_policy_errors(metrics, args))
    errors.extend(curvature_projection_errors(metrics, args))
    if (args.require_newton_assembly_diagnostics and
            not metrics["diagnostics"].get("newton_assemblies")):
        errors.append("Newton assembly diagnostics were not reported")
    if args.require_assembly_timing_diagnostics and not metrics["diagnostics"].get("assembly_timings"):
        errors.append("assembly timing diagnostics were not reported")
    errors.extend(assembly_efficiency_errors(metrics, args))
    if args.require_process_memory_diagnostics:
        diagnostics = metrics["diagnostics"]
        has_process_memory = (
            diagnostics.get("process_memory") or
            any(
                isinstance(record.get("process_rss_kb"), (int, float))
                for record in diagnostics.get("cut_context_rebuilds", [])
            )
        )
        if not has_process_memory:
            errors.append("process memory diagnostics were not reported")
    if (args.require_linear_solve_memory_diagnostics and
            not has_linear_solve_memory_diagnostics(metrics["diagnostics"])):
        errors.append("linear-solve memory diagnostics were not reported")
    if (args.require_fsils_matrix_diagnostics and
            not metrics["diagnostics"].get("fsils_prepared_matrices")):
        errors.append("FSILS prepared-matrix diagnostics were not reported")
    if args.max_fsils_matrix_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("FSILS prepared-matrix zero-row diagnostics are unavailable")
        elif zero_rows > args.max_fsils_matrix_zero_rows:
            errors.append(
                f"FSILS prepared-matrix zero rows {zero_rows} exceed "
                f"{args.max_fsils_matrix_zero_rows}"
            )
    if args.max_fsils_matrix_missing_diag is not None:
        missing_diag = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_missing_diag"
        )
        if not isinstance(missing_diag, (int, float)):
            errors.append("FSILS prepared-matrix missing-diagonal diagnostics are unavailable")
        elif missing_diag > args.max_fsils_matrix_missing_diag:
            errors.append(
                f"FSILS prepared-matrix missing diagonals {missing_diag} exceed "
                f"{args.max_fsils_matrix_missing_diag}"
            )
    if args.max_fsils_matrix_zero_diag is not None:
        zero_diag = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_diag")
        if not isinstance(zero_diag, (int, float)):
            errors.append("FSILS prepared-matrix zero-diagonal diagnostics are unavailable")
        elif zero_diag > args.max_fsils_matrix_zero_diag:
            errors.append(
                f"FSILS prepared-matrix zero diagonals {zero_diag} exceed "
                f"{args.max_fsils_matrix_zero_diag}"
            )
    if args.max_fsils_matrix_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("FSILS prepared-matrix nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_fsils_matrix_nonfinite_entries:
            errors.append(
                f"FSILS prepared-matrix nonfinite entries {nonfinite} exceed "
                f"{args.max_fsils_matrix_nonfinite_entries}"
            )
    if args.require_basis_cache_diagnostics:
        has_basis_cache = any(
            isinstance(record.get("basis_cache_entries"), (int, float))
            for record in metrics["diagnostics"].get("process_memory", [])
        )
        if not has_basis_cache:
            errors.append("basis-cache diagnostics were not reported")
    if args.max_diagnostic_process_basis_cache_entries is not None:
        basis_cache_entries = metrics.get("diagnostic_process_max_basis_cache_entries")
        if not isinstance(basis_cache_entries, (int, float)):
            errors.append("basis-cache entry diagnostics are unavailable")
        elif basis_cache_entries > args.max_diagnostic_process_basis_cache_entries:
            errors.append(
                f"basis-cache entries {basis_cache_entries} exceed "
                f"{args.max_diagnostic_process_basis_cache_entries}"
            )
    errors.extend(resource_ceiling_errors(metrics, args))
    if (args.require_interior_face_timing_diagnostics and
            not metrics["diagnostics"].get("interior_face_timings")):
        errors.append("interior-face timing diagnostics were not reported")
    if (args.require_cut_volume_timing_diagnostics and
            not metrics["diagnostics"].get("cut_volume_timings")):
        errors.append("cut-volume timing diagnostics were not reported")
    if (args.require_jit_specialization_trace_diagnostics and
            not metrics["diagnostics"].get("jit_specialization_traces")):
        errors.append("JIT specialization trace diagnostics were not reported")
    if (args.require_jit_cache_diagnostics and
            not metrics["diagnostics"].get("jit_cache_diagnostics")):
        errors.append("JIT cache diagnostics were not reported")
    if (args.require_marked_interior_face_fallback_diagnostics and
            not has_marked_interior_face_fallback_trace(metrics["diagnostics"])):
        errors.append("marked interior-face fallback diagnostics were not reported")
    if args.require_assembly_topology_consistency:
        errors.extend(assembly_topology_consistency_errors(metrics["diagnostics"]))
    if (args.require_eigen_factorization_diagnostics and
            not metrics["diagnostics"].get("eigen_factorization_diagnostics")):
        errors.append("Eigen factorization diagnostics were not reported")
    if (args.require_active_pressure_support_diagnostics and
            not metrics["diagnostics"].get("active_pressure_support_constraints")):
        errors.append("active pressure support diagnostics were not reported")
    if args.max_eigen_factorization_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_eigen_factorization_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("Eigen factorization zero-row diagnostics are unavailable")
        elif zero_rows > args.max_eigen_factorization_zero_rows:
            errors.append(
                f"Eigen factorization zero rows {zero_rows} exceed "
                f"{args.max_eigen_factorization_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_rows is not None:
        pressure_zero_rows = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_rows"
        )
        if not isinstance(pressure_zero_rows, (int, float)):
            errors.append("Eigen factorization pressure zero-row diagnostics are unavailable")
        elif pressure_zero_rows > args.max_eigen_factorization_pressure_zero_rows:
            errors.append(
                f"Eigen factorization pressure zero rows {pressure_zero_rows} exceed "
                f"{args.max_eigen_factorization_pressure_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_cols is not None:
        pressure_zero_cols = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_cols"
        )
        if not isinstance(pressure_zero_cols, (int, float)):
            errors.append("Eigen factorization pressure zero-column diagnostics are unavailable")
        elif pressure_zero_cols > args.max_eigen_factorization_pressure_zero_cols:
            errors.append(
                f"Eigen factorization pressure zero columns {pressure_zero_cols} exceed "
                f"{args.max_eigen_factorization_pressure_zero_cols}"
            )
    if args.max_eigen_factorization_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_eigen_factorization_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("Eigen factorization nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_eigen_factorization_nonfinite_entries:
            errors.append(
                f"Eigen factorization nonfinite entries {nonfinite} exceed "
                f"{args.max_eigen_factorization_nonfinite_entries}"
            )
    if not metrics["finite_velocity"]:
        errors.append("Velocity contains non-finite values")
    if args.min_capillary_response_speed_per_surface_tension is not None:
        surface_tension = metrics.get("surface_tension")
        max_speed = metrics.get("max_speed")
        if not isinstance(surface_tension, (int, float)):
            errors.append("surface-tension control is unavailable")
        elif abs(float(surface_tension)) <= 0.0:
            errors.append("surface tension is zero; capillary response cannot be normalized")
        elif not isinstance(max_speed, (int, float)):
            errors.append("capillary response speed diagnostic is unavailable")
        else:
            normalized_speed = float(max_speed) / abs(float(surface_tension))
            metrics["capillary_response_speed_per_surface_tension"] = normalized_speed
            if normalized_speed < args.min_capillary_response_speed_per_surface_tension:
                errors.append(
                    "capillary response speed per surface tension "
                    f"{normalized_speed:.6g} is below "
                    f"{args.min_capillary_response_speed_per_surface_tension:.6g}"
                )
    if args.max_capillary_balance_speed_per_surface_tension is not None:
        surface_tension = metrics.get("surface_tension")
        max_speed = metrics.get("max_speed")
        if not isinstance(surface_tension, (int, float)):
            errors.append("surface-tension control is unavailable")
        elif abs(float(surface_tension)) <= 0.0:
            errors.append("surface tension is zero; capillary balance cannot be normalized")
        elif not isinstance(max_speed, (int, float)):
            errors.append("capillary balance speed diagnostic is unavailable")
        else:
            normalized_speed = float(max_speed) / abs(float(surface_tension))
            metrics["capillary_balance_speed_per_surface_tension"] = normalized_speed
            if normalized_speed > args.max_capillary_balance_speed_per_surface_tension:
                errors.append(
                    "capillary balance speed per surface tension "
                    f"{normalized_speed:.6g} exceeds "
                    f"{args.max_capillary_balance_speed_per_surface_tension:.6g}"
                )
    if args.min_diagnostic_level_set_volume_correction_count is not None:
        count = metrics.get("diagnostic_level_set_volume_correction_count")
        if not isinstance(count, int):
            errors.append("level-set volume-correction diagnostics are unavailable")
        elif count < args.min_diagnostic_level_set_volume_correction_count:
            errors.append(
                f"level-set volume-correction count {count} is below "
                f"{args.min_diagnostic_level_set_volume_correction_count}"
            )
    if args.max_diagnostic_level_set_volume_correction_achieved_error is not None:
        error = metrics.get(
            "diagnostic_level_set_volume_correction_max_abs_achieved_error"
        )
        if not isinstance(error, (int, float)):
            errors.append("level-set volume-correction achieved-error diagnostic is unavailable")
        elif float(error) > args.max_diagnostic_level_set_volume_correction_achieved_error:
            errors.append(
                "level-set volume-correction achieved error "
                f"{float(error):.6g} exceeds "
                f"{args.max_diagnostic_level_set_volume_correction_achieved_error:.6g}"
            )
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
    if metrics.get("gate_nodes", 0) <= 0:
        if args.min_gate_mean_ux > -1.0:
            errors.append("gate region contains no wet nodes")
    elif metrics["gate_mean_velocity"][0] < args.min_gate_mean_ux:
        errors.append(
            f"gate mean ux {metrics['gate_mean_velocity'][0]:.6g} is below "
            f"{args.min_gate_mean_ux:.6g}"
        )
    if metrics.get("front_nodes", 0) <= 0:
        if args.min_front_mean_ux > -1.0:
            errors.append("front region contains no wet nodes")
    elif metrics["front_mean_velocity"][0] < args.min_front_mean_ux:
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
    for arg_name, metric_name, label in (
            ("min_interface_height_change",
             "interface_height_max_abs_change",
             "interface height max absolute change"),
            ("min_interface_mean_abs_height_change",
             "interface_height_mean_abs_change",
             "interface height mean absolute change"),
            ("min_interface_slope_change",
             "interface_slope_abs_change",
             "interface slope absolute change"),
            ("min_interface_final_height_span",
             "final_interface_height_span",
             "final interface height span")):
        minimum = getattr(args, arg_name)
        if minimum is None:
            continue
        if not metrics.get("interface_motion_available", False):
            reason = metrics.get("interface_motion_unavailable_reason", "unavailable")
            errors.append(f"interface motion diagnostics are unavailable ({reason})")
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            errors.append(f"{label} diagnostic is unavailable")
        elif float(value) < float(minimum):
            errors.append(
                f"{label} {float(value):.6g} is below {float(minimum):.6g}"
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
        context_volumes = metrics.get("cut_context_active_side_physical_volumes", [])
        wet_volume_source = str(metrics.get("wet_fraction_volume_source", "WetVolumeFraction"))
        if not isinstance(wet_fraction_volume, (int, float)):
            errors.append("WetVolumeFraction/WetVolumeMeasure output volume is unavailable")
        elif not context_volumes:
            errors.append("physical cut-context active-side volume was not reported")
        else:
            error = abs(float(wet_fraction_volume) - float(context_volumes[-1]))
            metrics["wet_fraction_volume_comparison_frame"] = "physical"
            metrics["wet_fraction_volume_error_vs_last_cut_context"] = error
            if error > args.max_wet_fraction_volume_error:
                errors.append(
                    f"{wet_volume_source} volume error {error:.6g} exceeds "
                    f"{args.max_wet_fraction_volume_error:.6g}"
                )
    if args.require_reference_profile_comparison:
        if metrics.get("reference_profile_error"):
            errors.append(
                "reference profile comparison failed: "
                f"{metrics['reference_profile_error']}"
            )
        if metrics.get("reference_profile_validation_passed") is False:
            failures = metrics.get("reference_profile_validation_failures", [])
            errors.append(
                "reference profile validation failed"
                + (f": {failures}" if failures else "")
            )
        for arg_name, metric_name, label in (
                ("min_reference_profile_coverage",
                 "reference_profile_coverage_fraction",
                 "reference profile coverage"),
                ("min_reference_profile_direct_coverage",
                 "reference_profile_direct_coverage_fraction",
                 "reference profile direct coverage")):
            minimum = getattr(args, arg_name)
            if minimum is None:
                continue
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)):
                errors.append(f"{label} diagnostic is unavailable")
            elif float(value) < float(minimum):
                errors.append(
                    f"{label} {float(value):.6g} is below {float(minimum):.6g}"
                )
        for arg_name, metric_name, label in (
                ("max_reference_profile_rmse",
                 "reference_profile_rmse_m",
                 "reference profile RMSE"),
                ("max_reference_profile_mae",
                 "reference_profile_mae_m",
                 "reference profile MAE"),
                ("max_reference_profile_max_abs_error",
                 "reference_profile_max_abs_error_m",
                 "reference profile max absolute error")):
            maximum = getattr(args, arg_name)
            if maximum is None:
                continue
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)):
                errors.append(f"{label} diagnostic is unavailable")
            elif float(value) > float(maximum):
                errors.append(
                    f"{label} {float(value):.6g} exceeds {float(maximum):.6g}"
                )
    return errors


def time_loop_convergence_errors(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> list[str]:
    if not args.require_time_loop_convergence:
        return []
    time_loop = metrics.get("time_loop")
    if not isinstance(time_loop, dict):
        diagnostics = metrics.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            time_loop = diagnostics.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if not isinstance(summary, dict):
        return ["time-loop convergence summary was not reported"]

    errors = []
    expected_steps = int(metrics.get("steps", 0) or 0)
    accepted_steps = summary.get("accepted_steps")
    if not isinstance(accepted_steps, int):
        errors.append("accepted-step count was not reported")
    elif expected_steps > 0 and accepted_steps < expected_steps:
        errors.append(
            f"accepted steps {accepted_steps} below requested steps {expected_steps}")
    if args.enable_adaptive_time_loop:
        final_time = summary.get("final_accepted_time")
        time_step = metrics.get("time_step_size")
        if not isinstance(time_step, (int, float)):
            controls = metrics.get("solver_controls", {})
            if isinstance(controls, dict):
                time_stepping = controls.get("time_stepping", {})
                if isinstance(time_stepping, dict):
                    time_step = time_stepping.get("time_step_size")
        if expected_steps > 0 and isinstance(time_step, (int, float)):
            expected_time = expected_steps * float(time_step)
            tolerance = max(1.0e-12, 1.0e-9 * max(1.0, abs(expected_time)))
            if not isinstance(final_time, (int, float)):
                errors.append("final accepted time was not reported")
            elif float(final_time) + tolerance < expected_time:
                errors.append(
                    f"final accepted time {float(final_time):.6g} below requested "
                    f"time {expected_time:.6g}"
                )
    elif summary.get("all_nonlinear_converged") is not True:
        errors.append("not all nonlinear solves converged")
    if not args.enable_adaptive_time_loop and summary.get("all_linear_converged") is not True:
        errors.append("not all linear solves converged")
    if args.max_time_loop_nonlinear_iterations_per_step is not None:
        max_nonlinear = summary.get("nonlinear_iterations_max")
        if not isinstance(max_nonlinear, int):
            errors.append("maximum nonlinear iteration count was not reported")
        elif max_nonlinear > args.max_time_loop_nonlinear_iterations_per_step:
            errors.append(
                f"maximum nonlinear iterations per step {max_nonlinear} exceed "
                f"{args.max_time_loop_nonlinear_iterations_per_step}"
            )
    if args.max_time_loop_linear_iterations_per_step is not None:
        max_linear = summary.get("linear_iterations_max")
        if not isinstance(max_linear, int):
            errors.append("maximum linear iteration count was not reported")
        elif max_linear > args.max_time_loop_linear_iterations_per_step:
            errors.append(
                f"maximum linear iterations per step {max_linear} exceed "
                f"{args.max_time_loop_linear_iterations_per_step}"
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


def format_failure_exception(failure: dict[str, Any],
                             qualification_log: Path | None) -> str:
    summary_keys = [
        "case",
        "run_dir",
        "result_path",
        "returncode",
        "timeout_seconds",
        "solver_elapsed_wall_seconds",
        "solver_elapsed_seconds_per_accepted_step",
        "result_step",
        "diagnostic_assembly_timing_count",
        "diagnostic_assembly_timings_per_accepted_step",
        "diagnostic_cut_context_rebuild_count",
        "diagnostic_cut_context_rebuilds_per_accepted_step",
        "diagnostic_assembly_timing_max_cut_volumes_seconds",
        "reference_profile_time_s",
        "reference_profile_validation_passed",
        "passed",
        "errors",
        "diagnostic_errors",
    ]
    summary = {
        key: failure[key]
        for key in summary_keys
        if key in failure
    }
    if qualification_log is not None:
        summary["qualification_log"] = str(qualification_log)
    return json.dumps(summary, indent=2, sort_keys=True)


def case_args_for_run(case_name: str,
                      args: argparse.Namespace) -> argparse.Namespace:
    case_args = argparse.Namespace(**vars(args))
    if (case_args.high_order_mpi_production_qualification and
            case_name == "tilt2d"):
        if not getattr(args, "_explicit_linear_solver_type", False):
            case_args.linear_solver_type = "ns"
        if not getattr(args, "_explicit_linear_max_iterations", False):
            case_args.linear_max_iterations = 100
    return case_args


def read_solver_log(run_dir: Path) -> str:
    log_path = run_dir / "solver_run.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def terminate_solver_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_solver_command(command: list[str],
                       run_dir: Path,
                       args: argparse.Namespace
                       ) -> tuple[subprocess.CompletedProcess[str], float]:
    log_path = run_dir / "solver_run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=solver_environment(args),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            terminate_solver_process(process)
            exc.output = read_solver_log(run_dir)
            exc.timeout = elapsed
            setattr(exc, "configured_timeout_seconds", args.timeout_seconds)
            raise
    elapsed = time.monotonic() - start
    output = read_solver_log(run_dir)
    completed = subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=output,
        stderr=None,
    )
    return completed, elapsed


def configure_case_solver_xml(run_dir: Path, args: argparse.Namespace) -> None:
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
        linear_algebra_backend=args.linear_algebra_backend,
        linear_preconditioner=args.linear_preconditioner,
        disable_coupled_outer_fgmres=args.disable_coupled_outer_fgmres,
        disable_cut_metadata_scale=args.disable_cut_metadata_scale,
        disable_velocity_extension=args.disable_velocity_extension,
        disable_vtk_output=args.disable_vtk_output,
        final_output_only=args.final_output_only,
        vtk_save_increment=args.vtk_save_increment,
        start_saving_after_step=args.start_saving_after_step,
        generated_interface_geometry=args.generated_interface_geometry,
        implicit_cut_quadrature_backend=args.implicit_cut_quadrature_backend,
        implicit_cut_fallback_policy=args.implicit_cut_fallback_policy,
        required_implicit_cut_backend_qualification=(
            args.required_implicit_cut_backend_qualification),
        implicit_cut_root_tolerance=args.implicit_cut_root_tolerance,
        implicit_cut_max_subdivision_depth=args.implicit_cut_max_subdivision_depth,
        generated_interface_quadrature_order=args.generated_interface_quadrature_order,
        interface_quadrature_order=args.interface_quadrature_order,
        volume_quadrature_order=args.volume_quadrature_order,
        cut_cell_velocity_gradient_penalty=args.cut_cell_velocity_gradient_penalty,
        cut_cell_pressure_gradient_penalty=args.cut_cell_pressure_gradient_penalty,
        surface_tension=args.surface_tension,
        projected_curvature_field=args.projected_curvature_field,
        curvature_projection_cadence_steps=args.curvature_projection_cadence_steps,
        curvature_projection_max_normalized_fit_residual=(
            args.curvature_projection_max_normalized_fit_residual),
        curvature_projection_smoothing_iterations=(
            args.curvature_projection_smoothing_iterations),
        curvature_projection_smoothing_relaxation=(
            args.curvature_projection_smoothing_relaxation),
        enable_volume_correction=args.enable_level_set_volume_correction,
        volume_correction_cadence_steps=args.volume_correction_cadence_steps,
        volume_correction_use_initial_volume=(
            args.volume_correction_use_initial_volume),
        volume_correction_tolerance=args.volume_correction_tolerance,
        volume_correction_max_iterations=args.volume_correction_max_iterations,
    )


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
            if case_name == "curvedtet3d":
                write_curved_tet3d_case(run_dir, args.steps)
            elif case_name == "capillaryarc2d":
                pressure_jump = 0.0
                if args.high_order_capillary_balance_smoke:
                    pressure_jump = -float(args.surface_tension) / CAPILLARY_ARC_RADIUS
                write_capillary_arc2d_case(run_dir, args.steps, pressure_jump)
            else:
                write_mini_case(run_dir, args.steps, static=(case_name == "static2d"))
            if args.use_high_order_implicit_cuts:
                configure_case_solver_xml(run_dir, args)
        else:
            run_dir = Path(temp_name) / source.name
            copy_case(source, run_dir, args.source_ref)
            regenerate_mms_case_if_requested(case_name, run_dir, args)
            configure_case_solver_xml(run_dir, args)

        try:
            command = solver_command(solver, args)
            completed, solver_elapsed_wall_seconds = run_solver_command(
                command, run_dir, args)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or exc.output or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            write_solver_log(run_dir, output)
            tail = "\n".join(output.splitlines()[-80:])
            diagnostics = parse_solver_diagnostics(output)
            failure = diagnostic_timeout_metrics(case_name, run_dir, diagnostics)
            failure["timeout_seconds"] = getattr(
                exc, "configured_timeout_seconds", args.timeout_seconds)
            failure["solver_elapsed_wall_seconds"] = exc.timeout
            failure["command"] = command
            failure["stdout_tail"] = tail
            diagnostic_errors = evaluate_timeout_diagnostics(failure, args)
            failure["diagnostic_errors"] = diagnostic_errors
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            if args.disable_cut_metadata_scale:
                failure["disable_cut_metadata_scale"] = True
            if args.disable_velocity_extension:
                failure["disable_velocity_extension"] = True
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
            raise RuntimeError(format_failure_exception(
                failure, args.qualification_log)) from exc
        write_solver_log(run_dir, completed.stdout)
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            failure = {
                "case": case_name,
                "run_dir": str(run_dir),
                "command": solver_command(solver, args),
                "returncode": completed.returncode,
                "solver_elapsed_wall_seconds": solver_elapsed_wall_seconds,
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
            if args.disable_velocity_extension:
                failure["disable_velocity_extension"] = True
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
            raise RuntimeError(format_failure_exception(
                failure, args.qualification_log))

        diagnostics = parse_solver_diagnostics(completed.stdout)
        if args.disable_vtk_output:
            metrics = {
                "output_metrics_skipped": True,
                "output_metrics_skip_reason": "VTK output disabled",
            }
        else:
            result_step = final_result_step(args.steps, diagnostics)
            result = result_path(run_dir, result_step)
            metrics = compute_metrics(case_name, run_dir, result)
            metrics["result_step"] = result_step
            metrics["result_path"] = str(result)
        add_diagnostic_metrics(metrics, diagnostics)
        if not args.disable_vtk_output:
            add_reference_profile_metrics(metrics, run_dir, result, args)
        metrics["case"] = case_name
        metrics["command"] = solver_command(solver, args)
        metrics["run_dir"] = str(run_dir)
        metrics["solver_elapsed_wall_seconds"] = solver_elapsed_wall_seconds
        metrics["steps"] = args.steps
        if args.time_step_size is not None:
            metrics["time_step_size"] = args.time_step_size
        if args.disable_coupled_outer_fgmres:
            metrics["disable_coupled_outer_fgmres"] = True
        if args.disable_cut_metadata_scale:
            metrics["disable_cut_metadata_scale"] = True
        if args.disable_velocity_extension:
            metrics["disable_velocity_extension"] = True
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
            raise RuntimeError(format_failure_exception(
                metrics, args.qualification_log))
        return metrics
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def set_default(args: argparse.Namespace, name: str, value: Any) -> None:
    if getattr(args, name) is None:
        setattr(args, name, value)


def remember_explicit_cli_overrides(args: argparse.Namespace) -> None:
    for name in (
        "linear_solver_type",
        "linear_max_iterations",
    ):
        setattr(args, f"_explicit_{name}", getattr(args, name) is not None)


def normalized_option(value: str | None) -> str:
    return (value or "").strip().lower()


def require_profile_production_linear_solver_policy(args: argparse.Namespace) -> None:
    if (not args.high_order_3d_benchmark_profile_qualification or
            args.allow_experimental_profile_linear_solver):
        return

    required = {
        "linear_algebra_backend": "fsils",
        "linear_preconditioner": "fsils",
        "linear_solver_type": "ns",
    }
    actual = {
        name: normalized_option(getattr(args, name))
        for name in required
    }
    mismatches = [
        f"{name}={actual[name] or '<unset>'} (required {expected})"
        for name, expected in required.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise ValueError(
            "The D18/D38 high-order profile qualification is production-gated "
            "on FSILS BlockSchur because the GMRES/RCS route is known to stall "
            "on long profiles. Use --allow-experimental-profile-linear-solver "
            "only for diagnostic probes. Mismatches: " + "; ".join(mismatches)
        )


def apply_high_order_production_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_production_qualification:
        if (args.steps is None and
                not args.high_order_mpi_production_qualification and
                not args.high_order_visible_motion_demo and
                not args.high_order_3d_benchmark_smoke and
                not args.high_order_3d_benchmark_qualification and
                not args.high_order_3d_benchmark_profile_qualification and
                not args.high_order_curved_3d_simplex_smoke and
                not args.high_order_mpi_motion_smoke and
                not args.high_order_capillary_projection_smoke and
                not args.high_order_capillary_response_smoke and
                not args.high_order_capillary_balance_smoke and
                not args.high_order_volume_corrected_motion_smoke):
            args.steps = 1
        return
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_PRODUCTION_CASES)
    if args.steps is None:
        args.steps = 20
    set_default(args, "timeout_seconds", 900.0)
    args.use_high_order_implicit_cuts = True
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True

    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_mpi_production_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_mpi_production_qualification:
        return
    if args.high_order_3d_benchmark_smoke:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-smoke"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_MPI_PRODUCTION_CASES)
    if args.steps is None:
        args.steps = 20
    set_default(args, "timeout_seconds", 1200.0)
    set_default(args, "mpi_ranks", 2)
    args.use_high_order_implicit_cuts = True
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "gmres")
    set_default(args, "linear_relative_tolerance", 1.0e-4)
    set_default(args, "linear_absolute_tolerance", 1.0e-4)
    # FSILS restarted GMRES interprets Max_iterations as restart cycles.
    # With the production cases' Krylov dimension of 80, 7 cycles cap the
    # reported total Krylov work at 567 iterations per nonlinear solve.
    set_default(args, "linear_max_iterations", 7)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_rss_kb", 350000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 175000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "SayeHyperrectangle")
    set_default(args, "expect_implicit_cut_backend_qualification",
                "ProductionQualified")
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 600)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_visible_motion_demo_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_visible_motion_demo:
        return
    if args.high_order_production_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-production-qualification"
        )
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_3d_benchmark_smoke:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-3d-benchmark-smoke"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_VISIBLE_MOTION_CASES)
    if args.steps is None:
        args.steps = 20
    set_default(args, "timeout_seconds", 300.0)
    set_default(args, "mpi_ranks", 2)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 1.0)
    args.use_high_order_implicit_cuts = True
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "ns")
    set_default(args, "linear_relative_tolerance", 1.0e-4)
    set_default(args, "linear_absolute_tolerance", 1.0e-4)
    set_default(args, "linear_max_iterations", 100)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 0.05
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 0.01
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 0.02)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_rss_kb", 350000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 175000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "SayeHyperrectangle")
    set_default(args, "expect_implicit_cut_backend_qualification",
                "ProductionQualified")
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 100)
    set_default(args, "min_interface_height_change", 0.02)
    set_default(args, "min_interface_mean_abs_height_change", 0.005)
    set_default(args, "min_interface_slope_change", 0.02)
    set_default(args, "min_interface_final_height_span", 0.02)


def apply_high_order_3d_benchmark_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_3d_benchmark_smoke:
        return
    if args.high_order_production_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-production-qualification"
        )
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_3D_BENCHMARK_CASES)
    if args.steps is None:
        args.steps = 1
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_vtk_output = True
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 700000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 300000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 32)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 32)


def apply_high_order_3d_benchmark_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_3d_benchmark_qualification:
        return
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_3D_BENCHMARK_QUALIFICATION_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 1200.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 6.0)
    args.use_high_order_implicit_cuts = True
    args.disable_vtk_output = True
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 800000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 350000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 32)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 32)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 250)


def apply_high_order_3d_benchmark_profile_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_3d_benchmark_profile_qualification:
        return
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-profile-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-profile-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_3D_BENCHMARK_PROFILE_CASES)
    if args.steps is None:
        args.steps = 312
    set_default(args, "timeout_seconds", 7200.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 6.0)
    args.use_high_order_implicit_cuts = True
    args.disable_vtk_output = False
    args.final_output_only = True
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True
    set_default(args, "fsils_matrix_diagnostics_every_n", 25)
    set_default(args, "fsils_matrix_diagnostics_max_records", 64)
    args.require_reference_profile_comparison = True
    args.enable_adaptive_time_loop = True
    args.newton_line_search_fail_on_no_reduction = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "ns")
    set_default(args, "ns_gm_max_iterations", 200)
    set_default(args, "ns_cg_max_iterations", 200)
    set_default(args, "ns_gm_tolerance", 1.0e-4)
    set_default(args, "ns_cg_tolerance", 1.0e-4)
    set_default(args, "adaptive_time_loop_min_dt", 6.25e-5)
    set_default(args, "adaptive_time_loop_max_dt", 5.0e-4)
    set_default(args, "adaptive_time_loop_max_retries", 8)
    set_default(args, "adaptive_time_loop_decrease_factor", 0.5)
    set_default(args, "adaptive_time_loop_increase_factor", 1.5)
    set_default(args, "adaptive_time_loop_target_newton_iterations", 6)
    set_default(args, "adaptive_time_loop_max_steps_multiplier", 16)
    set_default(args, "newton_line_search_max_iterations", 6)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 6.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 6.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 1000000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 650000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 32)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 32)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 9)
    set_default(args, "max_time_loop_linear_iterations_per_step", 250)
    set_default(args, "min_reference_profile_coverage", 0.95)
    set_default(args, "min_reference_profile_direct_coverage", 0.25)
    set_default(args, "max_reference_profile_rmse", 0.12)
    set_default(args, "max_reference_profile_mae", 0.10)
    set_default(args, "max_reference_profile_max_abs_error", 0.18)
    # D38 has a long shallow reference tail a few mm above the wet-bed depth;
    # use a material-height front threshold that tracks the moving wave.
    set_default(args, "reference_profile_elevated_front_clearance", 0.010)
    set_default(args, "max_reference_profile_elevated_front_lag", 0.30)


def apply_high_order_curved_3d_simplex_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_curved_3d_simplex_smoke:
        return
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-curved-3d-simplex-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CURVED_3D_SIMPLEX_CASES)
    if args.steps is None:
        args.steps = 1
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    # The current production contract for curved 3D simplex support is
    # conservative positive-weight cut-volume quadrature with verified volume
    # order 2 and interface order 1.  Do not request interface order 2 here
    # until the Tetra10 path has root-polished curved leaf rules end-to-end.
    set_default(args, "generated_interface_quadrature_order", 1)
    set_default(args, "interface_quadrature_order", 1)
    set_default(args, "volume_quadrature_order", 2)
    set_default(args, "implicit_cut_max_subdivision_depth", 2)
    set_default(args, "time_step_size", 2.0e-4)
    set_default(args, "linear_algebra_backend", "eigen")
    set_default(args, "linear_solver_type", "direct")
    # The curved Tetra10 hydrostatic smoke keeps velocity cut stabilization and
    # active pressure support enabled, but disables the pressure-gradient ghost
    # penalty. That penalty is not pressure-gradient robust for this curved
    # hydrostatic state and otherwise dominates the refreshed-quadrature Newton
    # solve.
    set_default(args, "cut_cell_pressure_gradient_penalty", 0.0)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-4
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-6
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-4)
    set_default(args, "min_diagnostic_pressure_range", 10.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 15.0)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 20.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 15.0)
    set_default(args, "max_diagnostic_process_rss_kb", 350000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 150000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 24)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 24)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 6)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 1.0e-5)
    set_default(args, "min_interface_slope_change", 1.0e-4)
    set_default(args, "min_interface_final_height_span", 1.0e-3)


def apply_high_order_mpi_motion_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_mpi_motion_smoke:
        return
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-mpi-motion-smoke cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-mpi-motion-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_MPI_MOTION_CASES)
    if args.steps is None:
        args.steps = 5
    set_default(args, "timeout_seconds", 600.0)
    set_default(args, "mpi_ranks", 2)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "gmres")
    set_default(args, "linear_relative_tolerance", 1.0e-4)
    set_default(args, "linear_absolute_tolerance", 1.0e-4)
    set_default(args, "linear_max_iterations", 7)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "SayeHyperrectangle")
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "max_diagnostic_process_rss_kb", 300000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 150000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 500)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_capillary_projection_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_capillary_projection_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_PROJECTION_CASES)
    if args.steps is None:
        args.steps = 10
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.require_curvature_projection_diagnostics = True
    args.require_curvature_projection_newton_freshness = True

    set_default(args, "surface_tension", 1.0e-3)
    set_default(args, "projected_curvature_field", "kappa_projected")
    set_default(args, "curvature_projection_cadence_steps", 1)
    set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
    set_default(args, "curvature_projection_smoothing_iterations", 1)
    set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)
    set_default(args, "min_diagnostic_curvature_projection_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0e-6)
    set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
    set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual", 5.0e-2)
    set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)


def apply_high_order_capillary_response_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_capillary_response_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_capillary_projection_smoke:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_RESPONSE_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 300.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 1.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.require_curvature_projection_diagnostics = True
    args.require_curvature_projection_newton_freshness = True

    set_default(args, "surface_tension", 0.5)
    set_default(args, "projected_curvature_field", "kappa_projected")
    set_default(args, "curvature_projection_cadence_steps", 1)
    set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
    set_default(args, "curvature_projection_smoothing_iterations", 1)
    set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-6
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-7
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-6)
    set_default(args, "min_capillary_response_speed_per_surface_tension", 1.0e-6)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_diagnostic_curvature_projection_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0)
    set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
    set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual", 5.0e-2)
    set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)


def apply_high_order_capillary_balance_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_capillary_balance_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_capillary_projection_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )
    if args.high_order_capillary_response_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-capillary-response-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_BALANCE_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 300.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 1.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.require_curvature_projection_diagnostics = True
    args.require_curvature_projection_newton_freshness = True

    set_default(args, "surface_tension", 0.5)
    set_default(args, "projected_curvature_field", "kappa_projected")
    set_default(args, "curvature_projection_cadence_steps", 1)
    set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
    set_default(args, "curvature_projection_smoothing_iterations", 1)
    set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 0.0
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 0.0
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "max_capillary_balance_speed_per_surface_tension", 1.0e-6)
    set_default(args, "min_diagnostic_pressure_range", 0.5)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_diagnostic_curvature_projection_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0)
    set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
    set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual", 5.0e-2)
    set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)


def apply_high_order_volume_corrected_motion_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_volume_corrected_motion_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_capillary_projection_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )
    if args.high_order_capillary_response_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-response-smoke"
        )
    if args.high_order_capillary_balance_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-balance-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_VOLUME_CORRECTED_MOTION_CASES)
    if args.steps is None:
        args.steps = 10
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True

    set_default(args, "enable_level_set_volume_correction", True)
    set_default(args, "volume_correction_use_initial_volume", True)
    set_default(args, "volume_correction_cadence_steps", 1)
    set_default(args, "volume_correction_tolerance", 1.0e-10)
    set_default(args, "volume_correction_max_iterations", 50)
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "min_diagnostic_level_set_volume_correction_count", 1)
    set_default(args, "max_diagnostic_level_set_volume_correction_achieved_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_implicit_defaults(args: argparse.Namespace) -> None:
    if not args.use_high_order_implicit_cuts:
        return
    if args.generated_interface_geometry is None:
        args.generated_interface_geometry = "HighOrderImplicit"
    if args.implicit_cut_quadrature_backend is None:
        args.implicit_cut_quadrature_backend = "SayeHyperrectangle"
    if args.implicit_cut_fallback_policy is None:
        args.implicit_cut_fallback_policy = "Fail"
    if args.implicit_cut_root_tolerance is None:
        args.implicit_cut_root_tolerance = 1.0e-10
    if args.implicit_cut_max_subdivision_depth is None:
        args.implicit_cut_max_subdivision_depth = 8
    if args.generated_interface_quadrature_order is None:
        args.generated_interface_quadrature_order = 2
    if args.interface_quadrature_order is None:
        args.interface_quadrature_order = 2
    if args.volume_quadrature_order is None:
        args.volume_quadrature_order = 2
    if args.linear_algebra_backend is None:
        args.linear_algebra_backend = "eigen"
    if args.disable_cut_stabilization is None:
        args.disable_cut_stabilization = True
    if args.mms_nx is None:
        args.mms_nx = 2
    if args.mms_ny is None:
        args.mms_ny = args.mms_nx
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    if args.max_diagnostic_assembly_timings_per_step is None:
        args.max_diagnostic_assembly_timings_per_step = 4.0
    if args.max_diagnostic_extra_assembly_timings_per_step is None:
        args.max_diagnostic_extra_assembly_timings_per_step = 3.0
    if args.max_diagnostic_cut_context_rebuilds_per_step is None:
        args.max_diagnostic_cut_context_rebuilds_per_step = 4.0
    if args.max_diagnostic_process_rss_kb is None:
        args.max_diagnostic_process_rss_kb = 300000.0
    if args.max_diagnostic_process_rss_growth_kb is None:
        args.max_diagnostic_process_rss_growth_kb = 100000.0
    if args.max_diagnostic_process_basis_cache_entries is None:
        args.max_diagnostic_process_basis_cache_entries = 4
    if args.max_diagnostic_process_basis_cache_entry_growth is None:
        args.max_diagnostic_process_basis_cache_entry_growth = 3
    if args.expect_generated_interface_geometry is None:
        args.expect_generated_interface_geometry = args.generated_interface_geometry
    if args.expect_implicit_cut_quadrature_backend is None:
        args.expect_implicit_cut_quadrature_backend = args.implicit_cut_quadrature_backend
    if (args.expect_selected_implicit_cut_quadrature_backend is None and
            args.implicit_cut_quadrature_backend != "Auto"):
        args.expect_selected_implicit_cut_quadrature_backend = (
            args.implicit_cut_quadrature_backend
        )
    if args.expect_implicit_cut_fallback_policy is None:
        args.expect_implicit_cut_fallback_policy = args.implicit_cut_fallback_policy
    if args.max_diagnostic_implicit_cut_fallback_cells is None:
        args.max_diagnostic_implicit_cut_fallback_cells = 0
    if args.min_diagnostic_achieved_volume_quadrature_order is None:
        args.min_diagnostic_achieved_volume_quadrature_order = 2
    args.require_high_order_cut_context_diagnostics = True


def validate_high_order_implicit_cases(cases: list[str],
                                       args: argparse.Namespace) -> None:
    if not args.use_high_order_implicit_cuts:
        return
    synthetic = [
        name for name in cases
        if CASES[name] is None and name not in HIGH_ORDER_SYNTHETIC_CASES
    ]
    if synthetic:
        names = ", ".join(synthetic)
        raise ValueError(
            "--use-high-order-implicit-cuts requires solver.xml-backed cases; "
            f"synthetic case(s) cannot be rewritten: {names}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--mpiexec", type=Path, default=Path("mpiexec"))
    parser.add_argument("--mpi-ranks", type=int)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--high-order-production-qualification", action="store_true",
                        help=("enable strict high-order implicit free-surface "
                              "qualification defaults"))
    parser.add_argument("--high-order-mpi-production-qualification",
                        action="store_true",
                        help=("enable strict MPI high-order implicit "
                              "free-surface production qualification defaults"))
    parser.add_argument("--high-order-visible-motion-demo",
                        action="store_true",
                        help=("enable a strict high-order implicit "
                              "free-surface demonstration with visibly large "
                              "interface motion"))
    parser.add_argument("--high-order-3d-benchmark-smoke", action="store_true",
                        help=("enable the high-order implicit 3D D18 benchmark "
                              "diagnostics smoke defaults"))
    parser.add_argument("--high-order-3d-benchmark-qualification",
                        action="store_true",
                        help=("enable multi-step high-order implicit D18/D38 "
                              "benchmark qualification defaults"))
    parser.add_argument("--high-order-3d-benchmark-profile-qualification",
                        action="store_true",
                        help=("enable full first-profile-time high-order "
                              "implicit D18/D38 benchmark qualification "
                              "defaults"))
    parser.add_argument("--high-order-curved-3d-simplex-smoke",
                        action="store_true",
                        help=("enable the high-order implicit curved Tetra10 "
                              "solver-level smoke defaults"))
    parser.add_argument("--high-order-mpi-motion-smoke", action="store_true",
                        help=("enable the high-order implicit MPI free-surface "
                              "motion smoke defaults"))
    parser.add_argument("--high-order-capillary-projection-smoke",
                        action="store_true",
                        help=("enable a high-order implicit free-surface smoke "
                              "with nonzero surface tension and projected "
                              "level-set curvature"))
    parser.add_argument("--high-order-capillary-response-smoke",
                        action="store_true",
                        help=("enable a zero-gravity high-order implicit "
                              "capillary response smoke with projected "
                              "level-set curvature"))
    parser.add_argument("--high-order-capillary-balance-smoke",
                        action="store_true",
                        help=("enable a zero-gravity high-order implicit "
                              "Laplace-style capillary balance smoke with "
                              "projected level-set curvature"))
    parser.add_argument("--high-order-volume-corrected-motion-smoke",
                        action="store_true",
                        help=("enable a high-order implicit free-surface "
                              "motion smoke with runtime global level-set "
                              "volume correction"))
    parser.add_argument("--source-ref")
    parser.add_argument("--steps", type=int)
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
    parser.add_argument("--min-interface-height-change", type=float)
    parser.add_argument("--min-interface-mean-abs-height-change", type=float)
    parser.add_argument("--min-interface-slope-change", type=float)
    parser.add_argument("--min-interface-final-height-span", type=float)
    parser.add_argument("--max-static-speed", type=float, default=1.0e-9)
    parser.add_argument("--stale-pressure-gauge-tolerance", type=float)
    parser.add_argument("--max-wet-fraction-volume-error", type=float)
    parser.add_argument("--require-reference-profile-comparison", action="store_true")
    parser.add_argument("--reference-profile-time-tolerance", type=float)
    parser.add_argument("--reference-profile-sample-radius", type=float)
    parser.add_argument("--reference-profile-elevated-front-clearance",
                        type=float)
    parser.add_argument("--min-reference-profile-coverage", type=float)
    parser.add_argument("--min-reference-profile-direct-coverage", type=float)
    parser.add_argument("--max-reference-profile-rmse", type=float)
    parser.add_argument("--max-reference-profile-mae", type=float)
    parser.add_argument("--max-reference-profile-max-abs-error", type=float)
    parser.add_argument("--max-reference-profile-elevated-front-lag", type=float)
    parser.add_argument("--allow-experimental-profile-linear-solver",
                        action="store_true",
                        help=("permit non-BlockSchur linear-solver overrides "
                              "for D18/D38 profile diagnostics"))
    parser.add_argument("--max-solver-elapsed-wall-seconds", type=float,
                        help=("fail a completed solver run whose measured wall "
                              "time exceeds this budget"))
    parser.add_argument("--max-solver-elapsed-seconds-per-accepted-step",
                        type=float,
                        help=("fail a completed solver run whose measured wall "
                              "time per accepted time step exceeds this budget"))
    parser.add_argument("--allow-timeout-diagnostics", action="store_true")
    parser.add_argument("--allow-failure-diagnostics", action="store_true")
    parser.add_argument("--min-diagnostic-solution-velocity-range", type=float)
    parser.add_argument("--min-diagnostic-pressure-range", type=float)
    parser.add_argument("--min-capillary-response-speed-per-surface-tension",
                        type=float)
    parser.add_argument("--max-capillary-balance-speed-per-surface-tension",
                        type=float)
    parser.add_argument("--min-diagnostic-level-set-volume-correction-count",
                        type=int)
    parser.add_argument("--max-diagnostic-level-set-volume-correction-achieved-error",
                        type=float)
    parser.add_argument("--max-diagnostic-active-volume-error", type=float)
    parser.add_argument("--min-diagnostic-cut-volume-exact-order", type=int)
    parser.add_argument("--min-diagnostic-cut-volume-max-exact-order", type=int)
    parser.add_argument("--max-diagnostic-cut-adjacent-scale", type=float)
    parser.add_argument("--min-diagnostic-cut-adjacent-capped-scale-count", type=int)
    parser.add_argument("--min-diagnostic-active-pruned-volume-regions", type=int)
    parser.add_argument("--min-diagnostic-active-min-volume-fraction", type=float)
    parser.add_argument("--min-diagnostic-generated-pruned-volume-rules", type=int)
    parser.add_argument("--max-diagnostic-implicit-cut-fallback-cells", type=int)
    parser.add_argument("--min-diagnostic-achieved-interface-quadrature-order", type=int)
    parser.add_argument("--min-diagnostic-achieved-volume-quadrature-order", type=int)
    parser.add_argument("--expect-generated-interface-geometry")
    parser.add_argument("--expect-implicit-cut-quadrature-backend")
    parser.add_argument("--required-implicit-cut-backend-qualification")
    parser.add_argument("--expect-selected-implicit-cut-quadrature-backend")
    parser.add_argument("--expect-implicit-cut-backend-qualification")
    parser.add_argument("--expect-implicit-cut-fallback-policy")
    parser.add_argument("--require-high-order-cut-context-diagnostics", action="store_true")
    parser.add_argument("--require-mms-verification", action="store_true")
    parser.add_argument("--require-time-loop-convergence", action="store_true")
    parser.add_argument("--min-diagnostic-blockschur-true-residual-retries", type=int)
    parser.add_argument("--require-newton-direction-check-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-check-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-top-mismatch-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-component-block-diagnostics", action="store_true")
    parser.add_argument("--require-linear-solve-history-diagnostics", action="store_true")
    parser.add_argument("--require-form-block-diagnostics", action="store_true")
    parser.add_argument("--require-cut-context-solution-source-diagnostics", action="store_true")
    parser.add_argument("--enable-newton-assembly-diagnostics", action="store_true")
    parser.add_argument("--require-newton-assembly-diagnostics", action="store_true")
    parser.add_argument("--require-assembly-timing-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-assembly-timings-per-step", type=float)
    parser.add_argument("--max-diagnostic-extra-assembly-timings-per-step", type=float)
    parser.add_argument("--max-diagnostic-cut-context-rebuilds-per-step", type=float)
    parser.add_argument("--max-diagnostic-newton-matrix-assemblies-per-step", type=float)
    parser.add_argument("--max-newton-direction-relative-error", type=float)
    parser.add_argument("--max-jacobian-check-relative-error", type=float)
    parser.add_argument("--max-jacobian-component-block-relative-error", type=float)
    parser.add_argument(
        "--jacobian-check-scheme",
        choices=("forward", "central"),
        help="Finite-difference scheme used by the solver Jacobian diagnostic.",
    )
    parser.add_argument("--disable-cut-stabilization",
                        dest="disable_cut_stabilization",
                        action="store_true",
                        default=None)
    parser.add_argument("--enable-cut-stabilization",
                        dest="disable_cut_stabilization",
                        action="store_false")
    parser.add_argument("--disable-cut-metadata-scale", action="store_true")
    parser.add_argument("--disable-velocity-extension", action="store_true")
    parser.add_argument("--cut-cell-velocity-gradient-penalty", type=float)
    parser.add_argument("--cut-cell-pressure-gradient-penalty", type=float)
    parser.add_argument("--surface-tension", type=float)
    parser.add_argument("--projected-curvature-field")
    parser.add_argument("--curvature-projection-cadence-steps", type=int)
    parser.add_argument("--curvature-projection-max-normalized-fit-residual", type=float)
    parser.add_argument("--curvature-projection-smoothing-iterations", type=int)
    parser.add_argument("--curvature-projection-smoothing-relaxation", type=float)
    parser.add_argument("--enable-level-set-volume-correction",
                        dest="enable_level_set_volume_correction",
                        action="store_true",
                        default=None)
    parser.add_argument("--disable-level-set-volume-correction",
                        dest="enable_level_set_volume_correction",
                        action="store_false")
    parser.add_argument("--volume-correction-cadence-steps", type=int)
    parser.add_argument("--volume-correction-use-initial-volume",
                        dest="volume_correction_use_initial_volume",
                        action="store_true",
                        default=None)
    parser.add_argument("--volume-correction-target-volume",
                        dest="volume_correction_use_initial_volume",
                        action="store_false")
    parser.add_argument("--volume-correction-tolerance", type=float)
    parser.add_argument("--volume-correction-max-iterations", type=int)
    parser.add_argument("--max-nonlinear-iterations", type=int)
    parser.add_argument("--linear-relative-tolerance", type=float)
    parser.add_argument("--linear-absolute-tolerance", type=float)
    parser.add_argument("--linear-max-iterations", type=int)
    parser.add_argument("--ns-gm-max-iterations", type=int)
    parser.add_argument("--ns-cg-max-iterations", type=int)
    parser.add_argument("--ns-gm-tolerance", type=float)
    parser.add_argument("--ns-cg-tolerance", type=float)
    parser.add_argument("--linear-solver-type")
    parser.add_argument("--linear-algebra-backend")
    parser.add_argument("--linear-preconditioner")
    parser.add_argument("--disable-coupled-outer-fgmres", action="store_true")
    parser.add_argument("--use-high-order-implicit-cuts", action="store_true")
    parser.add_argument("--mms-nx", type=int)
    parser.add_argument("--mms-ny", type=int)
    parser.add_argument("--generated-interface-geometry")
    parser.add_argument("--implicit-cut-quadrature-backend")
    parser.add_argument("--implicit-cut-fallback-policy")
    parser.add_argument("--implicit-cut-root-tolerance", type=float)
    parser.add_argument("--implicit-cut-max-subdivision-depth", type=int)
    parser.add_argument("--generated-interface-quadrature-order", type=int)
    parser.add_argument("--interface-quadrature-order", type=int)
    parser.add_argument("--volume-quadrature-order", type=int)
    parser.add_argument("--enable-blockschur-true-residual-retry", action="store_true")
    parser.add_argument("--enable-jacobian-check", action="store_true")
    parser.add_argument("--jacobian-check-iteration", type=int)
    parser.add_argument("--jacobian-check-step", type=float)
    parser.add_argument("--jacobian-check-components")
    parser.add_argument("--jacobian-check-component-sweeps")
    parser.add_argument("--enable-newton-direction-check", action="store_true")
    parser.add_argument("--newton-line-search-fail-on-no-reduction",
                        action="store_true")
    parser.add_argument("--newton-line-search-max-iterations", type=int)
    parser.add_argument("--enable-linear-solve-history", action="store_true")
    parser.add_argument("--linear-solve-history-max-calls", type=int)
    parser.add_argument("--enable-linear-solve-component-norms", action="store_true")
    parser.add_argument("--linear-solve-component-norms-max-newton-it", type=int)
    parser.add_argument("--enable-linear-solve-memory-diagnostics", action="store_true")
    parser.add_argument("--require-linear-solve-memory-diagnostics", action="store_true")
    parser.add_argument("--enable-timeloop-initialization-diagnostics", action="store_true")
    parser.add_argument("--require-timeloop-initialization-diagnostics", action="store_true")
    parser.add_argument("--enable-fsils-matrix-diagnostics", action="store_true")
    parser.add_argument("--require-fsils-matrix-diagnostics", action="store_true")
    parser.add_argument("--fsils-matrix-diagnostics-every-n", type=int)
    parser.add_argument("--fsils-matrix-diagnostics-max-records", type=int)
    parser.add_argument("--max-fsils-matrix-zero-rows", type=int)
    parser.add_argument("--max-fsils-matrix-missing-diag", type=int)
    parser.add_argument("--max-fsils-matrix-zero-diag", type=int)
    parser.add_argument("--max-fsils-matrix-nonfinite-entries", type=int)
    parser.add_argument("--require-basis-cache-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-process-basis-cache-entries", type=int)
    parser.add_argument("--max-diagnostic-process-rss-kb", type=float)
    parser.add_argument("--max-diagnostic-process-rss-growth-kb", type=float)
    parser.add_argument("--max-diagnostic-process-basis-cache-entry-growth", type=int)
    parser.add_argument("--enable-form-block-diagnostics", action="store_true")
    parser.add_argument("--enable-interior-face-timing", action="store_true")
    parser.add_argument("--require-interior-face-timing-diagnostics", action="store_true")
    parser.add_argument("--enable-cut-volume-timing", action="store_true")
    parser.add_argument("--require-cut-volume-timing-diagnostics", action="store_true")
    parser.add_argument("--enable-jit-specialization-trace", action="store_true")
    parser.add_argument("--require-jit-specialization-trace-diagnostics", action="store_true")
    parser.add_argument("--enable-jit-cache-diagnostics", action="store_true")
    parser.add_argument("--require-jit-cache-diagnostics", action="store_true")
    parser.add_argument("--require-process-memory-diagnostics", action="store_true")
    parser.add_argument("--require-marked-interior-face-fallback-diagnostics", action="store_true")
    parser.add_argument("--require-assembly-topology-consistency", action="store_true")
    parser.add_argument("--max-diagnostic-generated-cell-cache-full-miss-rebuilds", type=int)
    parser.add_argument("--require-eigen-factorization-diagnostics", action="store_true")
    parser.add_argument("--require-active-pressure-support-diagnostics", action="store_true")
    parser.add_argument("--require-curvature-projection-diagnostics", action="store_true")
    parser.add_argument("--require-curvature-projection-newton-freshness", action="store_true")
    parser.add_argument("--min-diagnostic-curvature-projection-count", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-max-abs-curvature", type=float)
    parser.add_argument("--max-diagnostic-curvature-projection-zero-fallback-vertices", type=int)
    parser.add_argument("--max-diagnostic-curvature-projection-normalized-fit-residual", type=float)
    parser.add_argument("--min-diagnostic-curvature-projection-smoothing-iterations", type=int)
    parser.add_argument("--max-eigen-factorization-zero-rows", type=int)
    parser.add_argument("--max-eigen-factorization-pressure-zero-rows", type=int)
    parser.add_argument("--max-eigen-factorization-pressure-zero-cols", type=int)
    parser.add_argument("--max-eigen-factorization-nonfinite-entries", type=int)
    parser.add_argument("--max-time-loop-nonlinear-iterations-per-step", type=int)
    parser.add_argument("--max-time-loop-linear-iterations-per-step", type=int)
    parser.add_argument("--enable-adaptive-time-loop", action="store_true")
    parser.add_argument("--adaptive-time-loop-min-dt", type=float)
    parser.add_argument("--adaptive-time-loop-max-dt", type=float)
    parser.add_argument("--adaptive-time-loop-max-retries", type=int)
    parser.add_argument("--adaptive-time-loop-decrease-factor", type=float)
    parser.add_argument("--adaptive-time-loop-increase-factor", type=float)
    parser.add_argument("--adaptive-time-loop-target-newton-iterations", type=int)
    parser.add_argument("--adaptive-time-loop-max-steps-multiplier", type=int)
    args = parser.parse_args()
    remember_explicit_cli_overrides(args)
    apply_high_order_production_qualification_defaults(args)
    apply_high_order_mpi_production_qualification_defaults(args)
    apply_high_order_visible_motion_demo_defaults(args)
    apply_high_order_3d_benchmark_smoke_defaults(args)
    apply_high_order_3d_benchmark_qualification_defaults(args)
    apply_high_order_3d_benchmark_profile_qualification_defaults(args)
    apply_high_order_curved_3d_simplex_smoke_defaults(args)
    apply_high_order_mpi_motion_smoke_defaults(args)
    apply_high_order_capillary_projection_smoke_defaults(args)
    apply_high_order_capillary_response_smoke_defaults(args)
    apply_high_order_capillary_balance_smoke_defaults(args)
    apply_high_order_volume_corrected_motion_smoke_defaults(args)
    apply_high_order_implicit_defaults(args)
    require_profile_production_linear_solver_policy(args)

    solver = resolve_solver(args.solver)
    cases = args.case or ["mini2d"]
    validate_high_order_implicit_cases(cases, args)
    report = []
    for case_name in cases:
        case_args = case_args_for_run(case_name, args)
        report.append(run_case(case_name, solver, case_args))
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
