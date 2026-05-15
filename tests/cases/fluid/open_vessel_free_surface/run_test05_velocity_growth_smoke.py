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
    "d18": CASE_ROOT / "spheric_test05_wet_bed_d18",
    "d38": CASE_ROOT / "spheric_test05_wet_bed_d38",
}
CASE_GATE_X = {
    "mini2d": 0.4,
}
CUT_CONTEXT_VOLUME_RE = re.compile(r"active_side_volume=([-+0-9.eE]+)")
CUT_ASSEMBLY_VOLUME_RE = re.compile(r"(?<!_)active_wet_volume=([-+0-9.eE]+)")


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


def configure_solver(solver_xml: Path, steps: int) -> None:
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

    free_surface = free_surface_bc(root)
    require_text(free_surface, "Implementation", "UnfittedLevelSet")
    require_text(free_surface, "Active_domain", "LevelSetNegative")
    require_text(free_surface, "Active_domain_method", "CutVolume")
    require_text(free_surface, "Enable_cut_cell_stabilization", "true")
    require_text(free_surface, "Use_cut_metadata_scale", "true")

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


def write_mini_mesh(case_dir: Path) -> tuple[int, float]:
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
    gauge_pressure = float(rho * gravity * column_height)
    return gauge_node, gauge_pressure


def write_mini_solver_xml(case_dir: Path,
                          steps: int,
                          gauge_node: int,
                          gauge_pressure: float) -> None:
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
  <Force_y>-9.81</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 0.75 0.0</Hydrostatic_pressure_reference_point>
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


def write_mini_case(case_dir: Path, steps: int) -> None:
    case_dir.mkdir(parents=True)
    gauge_node, gauge_pressure = write_mini_mesh(case_dir)
    write_mini_solver_xml(case_dir, steps, gauge_node, gauge_pressure)


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


def compute_metrics(case_name: str, case_dir: Path, result: Path) -> dict[str, Any]:
    benchmark_path = case_dir / "benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
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

    return {
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


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = []
    if not metrics["finite_velocity"]:
        errors.append("Velocity contains non-finite values")
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
    return errors


def run_case(case_name: str, solver: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = CASES[case_name]
    if source is not None and not source.exists():
        raise FileNotFoundError(source)

    with tempfile.TemporaryDirectory(prefix=f"dam_break_{case_name}_") as temp_name:
        run_dir = Path(temp_name) / case_name
        if source is None:
            write_mini_case(run_dir, args.steps)
        else:
            run_dir = Path(temp_name) / source.name
            copy_case(source, run_dir, args.source_ref)
            configure_solver(run_dir / "solver.xml", args.steps)

        completed = subprocess.run(
            [str(solver), "solver.xml"],
            cwd=run_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            raise RuntimeError(
                f"{case_name} solver probe exited with {completed.returncode}\n{tail}"
            )

        metrics = compute_metrics(case_name, run_dir, result_path(run_dir, args.steps))
        metrics.update(parse_active_volume_history(completed.stdout))
        errors = evaluate(metrics, args)
        metrics["case"] = case_name
        metrics["steps"] = args.steps
        metrics["passed"] = not errors
        metrics["errors"] = errors
        if errors:
            raise RuntimeError(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--source-ref")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--min-max-speed", type=float, default=1.0e-2)
    parser.add_argument("--min-wet-mean-speed", type=float, default=2.5e-4)
    parser.add_argument("--min-gate-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-front-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-active-volume-change", type=float, default=0.0)
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
