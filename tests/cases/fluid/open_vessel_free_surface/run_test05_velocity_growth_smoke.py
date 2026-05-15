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
}
CASE_GATE_X = {
    "mini2d": 0.4,
    "static2d": 0.5,
}
CUT_CONTEXT_VOLUME_RE = re.compile(r"active_side_volume=([-+0-9.eE]+)")
CUT_ASSEMBLY_VOLUME_RE = re.compile(r"(?<!_)active_wet_volume=([-+0-9.eE]+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
COMPONENT_NORM_RE = re.compile(
    r"\[(.*?) norm=([-+0-9.eE]+) mean=([-+0-9.eE]+)"
    r"(?: min=([-+0-9.eE]+) max=([-+0-9.eE]+))?\]"
)
VECTOR_COMPONENT_LABEL_RE = re.compile(r"label=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
STATIC_INTERFACE_HEIGHT = 0.53


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
                     disable_coupled_outer_fgmres: bool = False) -> None:
    tree = ET.parse(solver_xml)
    root = tree.getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise ValueError("missing GeneralSimulationParameters")

    set_text(general, "Number_of_time_steps", str(steps))
    set_text(general, "Save_results_to_VTK_format", "true")
    set_text(general, "Name_prefix_of_saved_VTK_files", "result")
    set_text(general, "Increment_in_saving_VTK_files", "1")
    set_text(general, "Start_saving_after_time_step", "1")
    set_text(general, "Increment_in_saving_restart_files", str(steps))
    if time_step_size is not None:
        set_text(general, "Time_step_size", f"{time_step_size:.16g}")

    if max_nonlinear_iterations is not None:
        for equation in root.findall("Add_equation"):
            set_text(equation, "Max_iterations", str(max_nonlinear_iterations))

    free_surface = free_surface_bc(root)
    require_text(free_surface, "Implementation", "UnfittedLevelSet")
    require_text(free_surface, "Active_domain", "LevelSetNegative")
    require_text(free_surface, "Active_domain_method", "CutVolume")
    require_text(free_surface, "Use_cut_metadata_scale", "true")
    if disable_cut_stabilization:
        set_text(free_surface, "Enable_cut_cell_stabilization", "false")
    else:
        require_text(free_surface, "Enable_cut_cell_stabilization", "true")

    if disable_coupled_outer_fgmres:
        set_text(navier_stokes_linear_solver(root),
                 "NS_Use_coupled_outer_FGMRES",
                 "false")

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


def parse_active_volume_history(solver_output: str) -> dict[str, Any]:
    context_volumes = [
        float(match.group(1))
        for match in CUT_CONTEXT_VOLUME_RE.finditer(solver_output)
    ]
    assembly_volumes = [
        float(match.group(1))
        for match in CUT_ASSEMBLY_VOLUME_RE.finditer(solver_output)
    ]

    def span(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(max(values) - min(values))

    return {
        "cut_context_active_side_volumes": context_volumes,
        "assembly_active_wet_volumes": assembly_volumes,
        "cut_context_active_side_volume_change": span(context_volumes),
        "assembly_active_wet_volume_change": span(assembly_volumes),
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
    return {
        match.group(1): parse_scalar(match.group(2))
        for match in KEY_VALUE_RE.finditer(line)
    }


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


def vector_component_header(line: str) -> str:
    label_match = VECTOR_COMPONENT_LABEL_RE.search(line)
    if label_match is None:
        return line
    return line[:label_match.end()]


def parse_solver_diagnostics(solver_output: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "cut_context_rebuilds": [],
        "cut_volume_assemblies": [],
        "hydrostatic_initializations": [],
        "pressure_gauge_checks": [],
        "residual_block_norms": [],
        "vector_component_norms": [],
    }
    for line in solver_output.splitlines():
        if "Active-domain cut context" in line:
            diagnostics["cut_context_rebuilds"].append(parse_key_values(line))
        elif "cut-volume active-domain diagnostics" in line:
            diagnostics["cut_volume_assemblies"].append(parse_key_values(line))
        elif "hydrostatic pressure initialization" in line:
            diagnostics["hydrostatic_initializations"].append(parse_key_values(line))
        elif "pressure gauge diagnostic" in line:
            diagnostics["pressure_gauge_checks"].append(parse_key_values(line))
        elif "residual block norms" in line:
            diagnostics["residual_block_norms"].append(parse_key_values(line))
        elif "vector component norms" in line:
            record = parse_key_values(vector_component_header(line))
            record["components"] = parse_component_norms(line)
            diagnostics["vector_component_norms"].append(record)

    diagnostics["counts"] = {
        name: len(records)
        for name, records in diagnostics.items()
        if isinstance(records, list)
    }
    diagnostics.update(parse_active_volume_history(solver_output))
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
        for record in diagnostics.get("cut_volume_assemblies", [])
        if isinstance(record.get("active_wet_volume"), (int, float))
    ]
    if not context_volumes or not assembly_volumes:
        return None
    return max(
        min(abs(assembly_volume - context_volume) for context_volume in context_volumes)
        for assembly_volume in assembly_volumes
    )


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
        "diagnostics": diagnostics,
    }
    velocity_range = diagnostic_solution_velocity_range(diagnostics)
    if velocity_range is not None:
        metrics["diagnostic_solution_velocity_range"] = velocity_range
    pressure_range = diagnostic_solution_pressure_range(diagnostics)
    if pressure_range is not None:
        metrics["diagnostic_solution_pressure_range"] = pressure_range
    active_volume_error = diagnostic_active_volume_error(diagnostics)
    if active_volume_error is not None:
        metrics["diagnostic_active_volume_error"] = active_volume_error
    gauge_value = diagnostic_pressure_gauge_value(diagnostics)
    if gauge_value is not None:
        metrics["diagnostic_pressure_gauge_value"] = gauge_value

    previous = previous_invalid_pressure(load_benchmark(run_dir))
    if previous is not None:
        metrics["pressure_gauge_previous_invalid"] = previous
        if gauge_value is not None:
            metrics["diagnostic_pressure_gauge_previous_invalid_difference"] = (
                gauge_value - previous
            )
    return metrics


def evaluate_timeout_diagnostics(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics["diagnostics"]
    if not diagnostics.get("cut_context_rebuilds"):
        errors.append("cut-context rebuild diagnostics were not reported")
    if not diagnostics.get("cut_volume_assemblies"):
        errors.append("cut-volume assembly diagnostics were not reported")
    if not diagnostics.get("pressure_gauge_checks"):
        errors.append("pressure gauge diagnostics were not reported")
    if not diagnostics.get("hydrostatic_initializations"):
        errors.append("hydrostatic initialization diagnostics were not reported")
    if not latest_component_record(diagnostics, "solution_state"):
        errors.append("solution-state component diagnostics were not reported")

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
        dimensions = benchmark["dimensions_m"]
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
                disable_coupled_outer_fgmres=args.disable_coupled_outer_fgmres,
            )

        try:
            completed = subprocess.run(
                [str(solver), "solver.xml"],
                cwd=run_dir,
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
            failure["stdout_tail"] = tail
            diagnostic_errors = evaluate_timeout_diagnostics(failure, args)
            failure["diagnostic_errors"] = diagnostic_errors
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            if args.allow_timeout_diagnostics and not diagnostic_errors:
                failure["passed"] = True
                return failure
            raise RuntimeError(json.dumps(failure, indent=2, sort_keys=True)) from exc
        write_solver_log(run_dir, completed.stdout)
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            failure = {
                "case": case_name,
                "run_dir": str(run_dir),
                "returncode": completed.returncode,
                "diagnostics": parse_solver_diagnostics(completed.stdout),
                "stdout_tail": tail,
            }
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            raise RuntimeError(json.dumps(failure, indent=2, sort_keys=True))

        metrics = compute_metrics(case_name, run_dir, result_path(run_dir, args.steps))
        metrics["diagnostics"] = parse_solver_diagnostics(completed.stdout)
        metrics.update(parse_active_volume_history(completed.stdout))
        metrics["case"] = case_name
        metrics["run_dir"] = str(run_dir)
        metrics["steps"] = args.steps
        if args.time_step_size is not None:
            metrics["time_step_size"] = args.time_step_size
        if args.disable_coupled_outer_fgmres:
            metrics["disable_coupled_outer_fgmres"] = True
        errors = evaluate(metrics, args)
        metrics["passed"] = not errors
        metrics["errors"] = errors
        if errors:
            raise RuntimeError(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--source-ref")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--time-step-size", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--preserve-run-dir", action="store_true")
    parser.add_argument("--min-max-speed", type=float, default=1.0e-2)
    parser.add_argument("--min-wet-mean-speed", type=float, default=2.5e-4)
    parser.add_argument("--min-gate-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-front-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-active-volume-change", type=float, default=0.0)
    parser.add_argument("--max-static-speed", type=float, default=1.0e-9)
    parser.add_argument("--stale-pressure-gauge-tolerance", type=float)
    parser.add_argument("--max-wet-fraction-volume-error", type=float)
    parser.add_argument("--allow-timeout-diagnostics", action="store_true")
    parser.add_argument("--min-diagnostic-solution-velocity-range", type=float)
    parser.add_argument("--min-diagnostic-pressure-range", type=float)
    parser.add_argument("--max-diagnostic-active-volume-error", type=float)
    parser.add_argument("--disable-cut-stabilization", action="store_true")
    parser.add_argument("--max-nonlinear-iterations", type=int)
    parser.add_argument("--disable-coupled-outer-fgmres", action="store_true")
    args = parser.parse_args()

    solver = resolve_solver(args.solver)
    cases = args.case or ["mini2d"]
    report = [run_case(case_name, solver, args) for case_name in cases]
    print(json.dumps({"solver": str(solver), "probes": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
