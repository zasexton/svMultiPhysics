#!/usr/bin/env python3
"""Prepare a one-step SPHERIC Test10 replay case from a saved VTU state.

The replay is diagnostic, not a restart: it initializes the OOP solver from a
saved VTU's point fields and advances one prescribed physical-time step.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


PATH_TAGS = {
    "Face_file_path",
    "Momentum_source_temporal_and_spatial_values_file_path",
    "Rotating_frame_angular_velocity_temporal_values_file_path",
    "Values_file_path",
}


def format_float(value: float) -> str:
    return f"{value:.16g}"


def set_child_text(parent: ET.Element, tag: str, value: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = value
    return child


def relative_to_output(output_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), output_dir.resolve())


def resolve_source_path(source_case: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_case / path


def rewrite_path_text(source_case: Path, output_dir: Path, element: ET.Element) -> None:
    if element.text is None or not element.text.strip():
        return
    target = resolve_source_path(source_case, element.text.strip())
    element.text = relative_to_output(output_dir, target)


def prepare_replay_case(
    *,
    source_case: Path,
    source_step: int,
    output_case: Path,
    start_time: float,
    time_step: float,
    min_iterations: int,
    pressure_penalty: float | None = None,
    pressure_policy: str | None = None,
) -> None:
    source_case = source_case.resolve()
    output_case = output_case.resolve()
    output_case.mkdir(parents=True, exist_ok=True)

    source_solver = source_case / "solver.xml"
    source_result = source_case / f"result_{source_step:03d}.vtu"
    if not source_solver.exists():
        raise FileNotFoundError(source_solver)
    if not source_result.exists():
        raise FileNotFoundError(source_result)

    tree = ET.parse(source_solver)
    root = tree.getroot()

    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise RuntimeError(f"solver.xml has no GeneralSimulationParameters: {source_solver}")
    set_child_text(general, "Use_new_OOP_solver", "true")
    set_child_text(general, "Continue_previous_simulation", "false")
    set_child_text(general, "Number_of_time_steps", "1")
    set_child_text(general, "Start_time", format_float(start_time))
    set_child_text(general, "Time_step_size", format_float(time_step))
    set_child_text(general, "Enable_adaptive_time_loop", "false")
    set_child_text(general, "Save_results_to_VTK_format", "true")
    set_child_text(general, "Combine_time_series", "true")
    set_child_text(general, "Name_prefix_of_saved_VTK_files", "result")
    set_child_text(general, "Increment_in_saving_VTK_files", "1")
    set_child_text(general, "Start_saving_after_time_step", "1")
    set_child_text(general, "Increment_in_saving_restart_files", "0")

    mesh = root.find("Add_mesh")
    if mesh is None:
        raise RuntimeError(f"solver.xml has no Add_mesh: {source_solver}")
    set_child_text(mesh, "Mesh_file_path", relative_to_output(output_case, source_result))

    for element in root.iter():
        if element.tag in PATH_TAGS:
            rewrite_path_text(source_case, output_case, element)

    for equation in root.findall("Add_equation"):
        if equation.get("type") in {"level_set", "fluid"}:
            set_child_text(equation, "Min_iterations", str(min_iterations))
        if equation.get("type") == "fluid":
            set_child_text(equation, "Hydrostatic_pressure_initialization", "false")
            for bc in equation.findall("Add_BC"):
                if bc.get("name") == "free_surface" or bc.findtext("Type") == "Free_surface":
                    if pressure_penalty is not None:
                        set_child_text(
                            bc,
                            "Cut_cell_pressure_gradient_penalty",
                            format_float(pressure_penalty),
                        )
                    if pressure_policy is not None:
                        set_child_text(
                            bc,
                            "Cut_cell_pressure_stabilization_policy",
                            pressure_policy,
                        )

    ET.indent(tree, space="  ")
    tree.write(output_case / "solver.xml", encoding="utf-8", xml_declaration=True)

    benchmark = source_case / "benchmark.json"
    if benchmark.exists():
        shutil.copyfile(benchmark, output_case / "benchmark.json")

    manifest = {
        "source_case": str(source_case),
        "source_step": source_step,
        "source_result": str(source_result),
        "start_time_s": start_time,
        "time_step_size_s": time_step,
        "number_of_time_steps": 1,
        "equation_min_iterations": min_iterations,
        "hydrostatic_pressure_initialization": False,
        "pressure_penalty_override": pressure_penalty,
        "pressure_policy_override": pressure_policy,
        "diagnostic_only": True,
        "diagnostic_note": (
            "Initial fields are loaded from the saved VTU mesh, but full "
            "transient history is not reconstructed."
        ),
    }
    (output_case / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--output-case", type=Path, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--time-step", type=float, required=True)
    parser.add_argument("--min-iterations", type=int, required=True)
    parser.add_argument("--pressure-penalty", type=float)
    parser.add_argument("--pressure-policy")
    args = parser.parse_args()

    if args.min_iterations < 0:
        raise ValueError("--min-iterations must be nonnegative")
    if not args.time_step > 0.0:
        raise ValueError("--time-step must be positive")

    prepare_replay_case(
        source_case=args.source_case,
        source_step=args.source_step,
        output_case=args.output_case,
        start_time=args.start_time,
        time_step=args.time_step,
        min_iterations=args.min_iterations,
        pressure_penalty=args.pressure_penalty,
        pressure_policy=args.pressure_policy,
    )


if __name__ == "__main__":
    main()
