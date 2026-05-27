#!/usr/bin/env python3
"""Check SPHERIC Test05 pressure-gauge benchmark metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
CASES = (
    ROOT / "unfitted_level_set" / "spheric_test05_wet_bed_d18",
    ROOT / "unfitted_level_set" / "spheric_test05_wet_bed_d38",
)
PREVIOUS_INVALID_D18_GAUGE = {
    "node_id": 279,
    "initial_phi": -0.001806,
    "full_volume_hydrostatic_pressure": 17.6869,
    "hydrostatic_error_range": [-17.6869, 0.0],
}
TOL = 1.0e-9
EXPECTED_FLUID_NONLINEAR_TOLERANCE = 2.0e-2
EXPECTED_FLUID_NONLINEAR_MAX_ITERATIONS = 12
EXPECTED_ADAPTIVE_TIME_LOOP = {
    "Enable_adaptive_time_loop": "true",
    "Adaptive_time_loop_min_dt": "1.5625e-5",
    "Adaptive_time_loop_max_dt": "5.0e-4",
    "Adaptive_time_loop_max_retries": "8",
    "Adaptive_time_loop_decrease_factor": "0.5",
    "Adaptive_time_loop_increase_factor": "1.5",
    "Adaptive_time_loop_target_newton_iterations": "6",
    "Adaptive_time_loop_max_steps_multiplier": "64",
}


def _load_pressure_gauge(path: Path) -> tuple[int, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one pressure gauge row in {path}")
    return int(rows[0]["node_id"]), float(rows[0]["pressure"])


def _expected_verification(gauge: dict[str, object]) -> dict[str, object]:
    current_pressure = float(gauge["expected_initial_hydrostatic_pressure"])
    previous_pressure = PREVIOUS_INVALID_D18_GAUGE["full_volume_hydrostatic_pressure"]
    previous_range = PREVIOUS_INVALID_D18_GAUGE["hydrostatic_error_range"]
    return {
        "current_prescribed_pressure_matches_initial_hydrostatic": True,
        "initial_pressure_error_after_constraint": 0.0,
        "previous_invalid_d18_node_id": PREVIOUS_INVALID_D18_GAUGE["node_id"],
        "previous_invalid_d18_initial_phi": PREVIOUS_INVALID_D18_GAUGE["initial_phi"],
        "previous_invalid_d18_full_volume_hydrostatic_pressure": previous_pressure,
        "previous_invalid_d18_hydrostatic_error_range": previous_range,
        "current_pressure_matches_previous_invalid_offset": False,
        "current_pressure_matches_previous_invalid_error_range": (
            previous_range[0] <= current_pressure <= previous_range[1]
        ),
        "current_pressure_minus_previous_invalid_pressure": current_pressure - previous_pressure,
    }


def _nearly_equal(left: object, right: object) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= TOL
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nearly_equal(a, b) for a, b in zip(left, right))
    return left == right


def _require_text(parent: ET.Element, path: str, expected: str, source: Path) -> None:
    value = parent.findtext(path, "").strip()
    if value != expected:
        raise ValueError(f"{source} {path} is {value!r}, expected {expected!r}")


def check_case(case_dir: Path) -> dict[str, object]:
    metadata_path = case_dir / "benchmark.json"
    gauge_path = case_dir / "pressure_gauge.csv"
    solver_path = case_dir / "solver.xml"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gauge = metadata["pressure_gauge"]
    verification = metadata.get("pressure_gauge_verification")
    if not isinstance(verification, dict):
        raise ValueError(f"{metadata_path} is missing pressure_gauge_verification")

    csv_node, csv_pressure = _load_pressure_gauge(gauge_path)
    expected_pressure = float(gauge["expected_initial_hydrostatic_pressure"])
    if csv_node != int(gauge["node_id"]):
        raise ValueError(f"{gauge_path} node_id does not match benchmark metadata")
    if abs(csv_pressure - expected_pressure) > TOL:
        raise ValueError(f"{gauge_path} pressure does not match expected hydrostatic value")

    expected = _expected_verification(gauge)
    for key, expected_value in expected.items():
        if key not in verification:
            raise ValueError(f"{metadata_path} is missing verification field {key}")
        if not _nearly_equal(verification[key], expected_value):
            raise ValueError(
                f"{metadata_path} field {key} has {verification[key]!r}, "
                f"expected {expected_value!r}"
            )

    if not verification["current_prescribed_pressure_matches_initial_hydrostatic"]:
        raise ValueError(f"{metadata_path} does not prescribe the initial hydrostatic gauge pressure")
    if verification["current_pressure_matches_previous_invalid_offset"]:
        raise ValueError(f"{metadata_path} still matches the previous invalid D18 pressure offset")
    if verification["current_pressure_matches_previous_invalid_error_range"]:
        raise ValueError(f"{metadata_path} still lies in the previous invalid D18 error range")

    root = ET.parse(solver_path).getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise ValueError(f"{solver_path} is missing GeneralSimulationParameters")
    for tag, expected_value in EXPECTED_ADAPTIVE_TIME_LOOP.items():
        _require_text(general, tag, expected_value, solver_path)
    combine_time_series = general.findtext("Combine_time_series", "").strip()
    if combine_time_series.lower() != "true":
        raise ValueError(f"{solver_path} does not request Combine_time_series=true")

    level_set = next(
        (
            equation
            for equation in root.findall("Add_equation")
            if equation.attrib.get("type") == "level_set"
        ),
        None,
    )
    if level_set is None:
        raise ValueError(f"{solver_path} is missing a level_set equation")
    _require_text(level_set, "Velocity_source", "prescribed_data", solver_path)
    _require_text(level_set, "Velocity_field_name", "LevelSetAdvectionVelocity", solver_path)
    _require_text(level_set, "Auto_register_velocity_field", "true", solver_path)
    _require_text(level_set, "Use_wet_extension_advection_velocity", "true", solver_path)
    _require_text(level_set, "Source_velocity_field_name", "Velocity", solver_path)
    _require_text(
        level_set,
        "Wet_extension_advection_velocity_method",
        "nearest_interface_point",
        solver_path,
    )

    fluid = next(
        (
            equation
            for equation in root.findall("Add_equation")
            if equation.attrib.get("type") == "fluid"
        ),
        None,
    )
    if fluid is None:
        raise ValueError(f"{solver_path} is missing a fluid equation")
    fluid_tolerance = float(fluid.findtext("Tolerance", "nan"))
    if abs(fluid_tolerance - EXPECTED_FLUID_NONLINEAR_TOLERANCE) > TOL:
        raise ValueError(
            f"{solver_path} fluid nonlinear tolerance is {fluid_tolerance}, "
            f"expected {EXPECTED_FLUID_NONLINEAR_TOLERANCE}"
        )
    fluid_max_iterations = int(fluid.findtext("Max_iterations", "-1"))
    if fluid_max_iterations != EXPECTED_FLUID_NONLINEAR_MAX_ITERATIONS:
        raise ValueError(
            f"{solver_path} fluid nonlinear Max_iterations is {fluid_max_iterations}, "
            f"expected {EXPECTED_FLUID_NONLINEAR_MAX_ITERATIONS}"
        )
    constraints = fluid.find("Node_pressure_constraints")
    if constraints is None:
        raise ValueError(f"{solver_path} does not activate Node_pressure_constraints")
    values_path = constraints.findtext("Values_file_path", "").strip()
    if values_path != "pressure_gauge.csv":
        raise ValueError(
            f"{solver_path} uses Node_pressure_constraints file {values_path!r}, "
            "expected 'pressure_gauge.csv'"
        )
    fluid_ls = fluid.find("LS")
    if fluid_ls is None or fluid_ls.attrib.get("type", "").strip().lower() != "direct":
        actual = None if fluid_ls is None else fluid_ls.attrib.get("type", "")
        raise ValueError(
            f"{solver_path} fluid LS type is {actual!r}, expected 'Direct' "
            "for the wet-bed serial validation setting"
        )
    linear_algebra = fluid_ls.find("Linear_algebra")
    if linear_algebra is None or linear_algebra.attrib.get("type", "").strip().lower() != "eigen":
        actual = None if linear_algebra is None else linear_algebra.attrib.get("type", "")
        raise ValueError(
            f"{solver_path} fluid linear algebra backend is {actual!r}, expected 'eigen'"
        )
    preconditioner = linear_algebra.findtext("Preconditioner", "").strip().lower()
    if preconditioner != "none":
        raise ValueError(
            f"{solver_path} fluid direct solver preconditioner is {preconditioner!r}, "
            "expected 'none'"
        )
    free_surface = next(
        (
            bc
            for bc in fluid.findall("Add_BC")
            if bc.findtext("Type", "").strip() == "Free_surface"
        ),
        None,
    )
    if free_surface is None:
        raise ValueError(f"{solver_path} fluid equation is missing the free-surface BC")
    _require_text(free_surface, "Implementation", "UnfittedLevelSet", solver_path)
    _require_text(free_surface, "Enable_cut_cell_stabilization", "true", solver_path)
    _require_text(free_surface, "Use_cut_metadata_scale", "false", solver_path)
    return {
        "case": case_dir.name,
        "node_id": csv_node,
        "pressure": csv_pressure,
        "previous_invalid_pressure": PREVIOUS_INVALID_D18_GAUGE["full_volume_hydrostatic_pressure"],
        "pressure_difference": verification["current_pressure_minus_previous_invalid_pressure"],
    }


def main() -> None:
    report = [check_case(case_dir) for case_dir in CASES]
    print(json.dumps({"checked_cases": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
