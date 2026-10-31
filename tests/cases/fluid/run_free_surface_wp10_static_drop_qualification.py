#!/usr/bin/env python3
"""Run the frozen WP-10 circular two-fluid static-drop progression gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MATRIX = SCRIPT_PATH.with_name(
    "free_surface_wp10_static_drop_matrix.json"
)
RUNNER_TEST_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "test_free_surface_wp10_static_drop_qualification.py"
)
HYDROSTATIC_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp10_two_fluid_hydrostatic_qualification.py"
)
EXPECTED_MATRIX_SHA256 = (
    "e58a1a6291a1affb1b291056ddaab10f3c2009eee79e4d6927cd7dbad39e29a7"
)
EXPECTED_HYDROSTATIC_RUNNER_SHA256 = (
    "3b988876e19f303e6f83b500280296023e40db6a4c887497316df546def7f152"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "matrix_id",
    "status",
    "work_package",
    "qualification_campaign",
    "scope",
    "accepted_claim",
    "rejected_claims",
    "model_envelope",
    "geometry",
    "refinement",
    "time",
    "nonlinear_solver",
    "thresholds",
    "execution",
    "cases",
    "required_later_progression",
    "qualification_disposition",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_hydrostatic_runner() -> Any:
    if _sha256_file(HYDROSTATIC_RUNNER_PATH) != EXPECTED_HYDROSTATIC_RUNNER_SHA256:
        raise RuntimeError("hydrostatic runner dependency bytes changed")
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_static_drop_hydrostatic_dependency",
        HYDROSTATIC_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the hydrostatic runner dependency")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


HYDROSTATIC = _load_hydrostatic_runner()
COMMON = HYDROSTATIC.COMMON


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_matrix_structure(matrix: Any) -> dict[str, Any]:
    if not isinstance(matrix, dict) or set(matrix) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("static-drop matrix top-level contract changed")
    expected_metadata = {
        "schema_version": 1,
        "matrix_id": "free_surface_wp10_static_drop_v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
        "accepted_claim": "two_fluid_circular_static_drop_prerequisite",
    }
    for key, expected in expected_metadata.items():
        if matrix.get(key) != expected:
            raise ValueError(f"static-drop matrix {key} changed")
    if matrix.get("rejected_claims") != [
        "spherical_static_drop_balance",
        "sustained_dynamics",
        "both_phase_mass_conservation",
        "high_ratio_conditioning",
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
    ]:
        raise ValueError("static-drop rejected-claim boundary changed")
    if matrix.get("qualification_disposition") != {
        "two_fluid_circular_static_drop_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }:
        raise ValueError("static-drop disposition changed")

    model = matrix.get("model_envelope")
    expected_model = {
        "physical_model": "incompressible_two_fluid",
        "momentum_operator": "two_fluid_stokes",
        "spatial_dimension": 2,
        "mesh": "affine_p1_triangle",
        "interface_geometry": "linear_corner",
        "domain": "closed_unit_square",
        "exterior_velocity": "homogeneous_strong_dirichlet_both_phases",
        "initial_velocity": "identically_zero_in_both_phases",
        "pressure_gauge": "explicit_negative_phase_strict_interior_vertex",
        "pressure_state": "piecewise_constant_young_laplace_jump",
        "surface_tension": 1.0,
        "body_force": "zero",
        "phase_transport": (
            "locally_conservative_p1_indicator_with_stationary_geometry_reconciliation"
        ),
        "speed_observable": (
            "accepted_stage_phasewise_maximum_quadrature_speed"
        ),
        "solver": "fsils_block_schur",
    }
    if model != expected_model:
        raise ValueError("static-drop model envelope changed")

    geometry = matrix.get("geometry")
    expected_geometry_contract = {
        "kind": "circle",
        "radius": 0.25,
        "centers": {
            "centered": [0.5, 0.5],
            "shifted": [0.5375, 0.475],
        },
    }
    if geometry != expected_geometry_contract:
        raise ValueError("static-drop geometry contract changed")
    refinement = matrix.get("refinement")
    expected_refinement = {
        "resolutions": [32, 64, 128],
        "radius_over_h": [8.0, 16.0, 32.0],
        "families": ["centered", "shifted"],
        "expected_asymptotic_order": 2.0,
        "minimum_observed_order": 1.5,
    }
    if refinement != expected_refinement:
        raise ValueError("static-drop refinement contract changed")

    if matrix.get("time") != {
        "steps": 1,
        "dt": 0.001,
        "scheme": "BackwardEuler",
        "interpretation": "stationary_capillary_acceptance_step",
    }:
        raise ValueError("static-drop time contract changed")
    if matrix.get("nonlinear_solver") != {
        "control_source": "GeneralSimulationParameters",
        "absolute_tolerance": 5.0e-10,
        "relative_tolerance": 0.0,
        "maximum_iterations": 25,
    }:
        raise ValueError("static-drop nonlinear contract changed")

    thresholds = matrix.get("thresholds")
    required_thresholds = {
        "phase_partition_absolute",
        "minimum_interface_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
        "maximum_final_nonlinear_residual",
        "per_case_interface_measure_relative",
        "per_case_interior_volume_relative",
        "per_case_pressure_jump_relative",
        "per_case_pressure_jump_relative_dispersion",
        "per_case_normal_flux_absolute",
        "per_case_velocity_jump_squared",
        "per_case_maximum_phase_capillary_number",
        "finest_interface_measure_relative",
        "finest_interior_volume_relative",
        "finest_pressure_jump_relative",
        "finest_pressure_jump_relative_dispersion",
        "finest_maximum_phase_capillary_number",
        "minimum_observed_order",
        "gci_safety_factor",
        "finest_pressure_jump_gci",
        "side_reversal_absolute",
        "side_reversal_relative",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != required_thresholds:
        raise ValueError("static-drop threshold inventory changed")
    for key, value in thresholds.items():
        number = _finite_number(value, f"threshold {key}")
        if number < 0.0:
            raise ValueError(f"threshold {key} must be nonnegative")
    if thresholds["finest_pressure_jump_relative"] != 0.01:
        raise ValueError("static-drop pressure gate changed")
    if thresholds["finest_maximum_phase_capillary_number"] != 1.0e-6:
        raise ValueError("static-drop maximum-speed gate changed")

    execution = matrix.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "wall_time_seconds_per_case",
        "memory_mib_per_case",
        "output_mib_per_case",
        "omp_threads",
        "solver_rank_trace",
    }:
        raise ValueError("static-drop execution contract changed")
    if execution["solver_rank_trace"] != "root_only":
        raise ValueError("static-drop rank trace changed")
    for key in (
        "wall_time_seconds_per_case",
        "memory_mib_per_case",
        "output_mib_per_case",
        "omp_threads",
    ):
        value = execution[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"execution {key} must be a positive integer")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("static-drop matrix must contain twelve cases")
    required_case_keys = {
        "case_id",
        "reversal_pair",
        "refinement_family",
        "resolution",
        "center_class",
        "center",
        "radius",
        "surface_tension",
        "level_set_sign",
        "liquid_phase",
        "negative_density",
        "negative_viscosity",
        "positive_density",
        "positive_viscosity",
        "mpi_ranks",
    }
    seen_ids: set[str] = set()
    observed: set[tuple[int, str, int]] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != required_case_keys:
            raise ValueError("static-drop case contract changed")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("static-drop case IDs must be unique")
        seen_ids.add(case_id)
        resolution = case["resolution"]
        center_class = case["center_class"]
        sign = case["level_set_sign"]
        if resolution not in {32, 64, 128}:
            raise ValueError("static-drop resolution changed")
        if center_class not in {"centered", "shifted"}:
            raise ValueError("static-drop center class changed")
        if sign not in {-1, 1}:
            raise ValueError("static-drop level-set sign changed")
        observed.add((resolution, center_class, sign))
        expected_id = (
            f"circle_n{resolution}_{center_class}_"
            + ("negative" if sign == 1 else "positive")
        )
        if case_id != expected_id:
            raise ValueError("static-drop case ID does not encode its state")
        if case["reversal_pair"] != f"circle_n{resolution}_{center_class}":
            raise ValueError("static-drop reversal pair changed")
        expected_family = center_class + ("_negative" if sign == 1 else "_positive")
        if case["refinement_family"] != expected_family:
            raise ValueError("static-drop refinement family changed")
        if case["center"] != geometry["centers"][center_class]:
            raise ValueError("static-drop case center changed")
        if case["radius"] != geometry["radius"] or case["surface_tension"] != 1.0:
            raise ValueError("static-drop capillary geometry changed")
        expected_liquid = "negative" if sign == 1 else "positive"
        if case["liquid_phase"] != expected_liquid:
            raise ValueError("static-drop liquid phase changed")
        expected_ranks = 2 if resolution == 128 else 1
        if case["mpi_ranks"] != expected_ranks:
            raise ValueError("static-drop rank schedule changed")
        liquid_density = case[f"{expected_liquid}_density"]
        liquid_viscosity = case[f"{expected_liquid}_viscosity"]
        gas = "positive" if expected_liquid == "negative" else "negative"
        if (
            liquid_density != 1000.0
            or liquid_viscosity != 0.01
            or case[f"{gas}_density"] != 1.0
            or case[f"{gas}_viscosity"] != 0.001
        ):
            raise ValueError("static-drop material assignment changed")
    expected_states = {
        (resolution, center_class, sign)
        for resolution in (32, 64, 128)
        for center_class in ("centered", "shifted")
        for sign in (-1, 1)
    }
    if observed != expected_states:
        raise ValueError("static-drop case coverage changed")
    return matrix


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved == DEFAULT_MATRIX.resolve():
        actual_hash = _sha256_file(resolved)
        if actual_hash != EXPECTED_MATRIX_SHA256:
            raise RuntimeError("static-drop matrix bytes changed")
    with resolved.open("r", encoding="utf-8") as source:
        matrix = json.load(source)
    return _validate_matrix_structure(matrix)


def validate_requested_claim(matrix: dict[str, Any], claim: str) -> str:
    accepted = matrix["accepted_claim"]
    if claim != accepted:
        raise ValueError(
            f"requested claim {claim!r} exceeds accepted claim {accepted!r}"
        )
    return claim


def expected_geometry(case: dict[str, Any]) -> dict[str, float]:
    radius = _finite_number(case["radius"], "circle radius")
    surface_tension = _finite_number(
        case["surface_tension"], "surface tension"
    )
    if radius <= 0.0 or surface_tension <= 0.0:
        raise ValueError("circle radius and surface tension must be positive")
    interior = math.pi * radius * radius
    sign = int(case["level_set_sign"])
    return {
        "interface_measure": 2.0 * math.pi * radius,
        "interior_volume": interior,
        "negative_volume": interior if sign == 1 else 1.0 - interior,
        "positive_volume": 1.0 - interior if sign == 1 else interior,
        "pressure_jump": surface_tension / radius,
    }


def expected_signed_pressure_jump(case: dict[str, Any]) -> float:
    return float(case["level_set_sign"]) * expected_geometry(case)[
        "pressure_jump"
    ]


def level_set_value(case: dict[str, Any], x: float, y: float) -> float:
    center = case["center"]
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError("circle center must contain two coordinates")
    center_x = _finite_number(center[0], "circle center x")
    center_y = _finite_number(center[1], "circle center y")
    radius = _finite_number(case["radius"], "circle radius")
    sign = int(case["level_set_sign"])
    if sign not in {-1, 1}:
        raise ValueError("level-set sign must be negative or positive one")
    return sign * (math.hypot(x - center_x, y - center_y) - radius)


def _adapted_hydrostatic_inputs(
    matrix: dict[str, Any], case: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapted = copy.deepcopy(HYDROSTATIC.load_matrix(HYDROSTATIC.DEFAULT_MATRIX))
    resolution = int(case["resolution"])
    adapted["mesh"].update(
        {
            "nx": resolution,
            "ny": resolution,
            "expected_vertices": (resolution + 1) ** 2,
            "expected_triangles": 2 * resolution * resolution,
            "expected_wall_vertices": 4 * resolution,
            "expected_wall_lines": 4 * resolution,
            "expected_phase_initializer_dofs": 3 * (resolution + 1) ** 2,
        }
    )
    adapted["time"] = copy.deepcopy(matrix["time"])
    adapted["nonlinear_solver"] = copy.deepcopy(matrix["nonlinear_solver"])
    planar_case = copy.deepcopy(case)
    planar_case.update(
        {
            "orientation": "x",
            "offset": 0.5,
            "gravity_acceleration": 0.0,
        }
    )
    return adapted, planar_case


def _format_values(values: list[float | int], width: int = 8) -> str:
    rows = []
    for start in range(0, len(values), width):
        rows.append(
            "          "
            + " ".join(
                format(value, ".17g") for value in values[start : start + width]
            )
        )
    return "\n".join(rows)


def render_mesh(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    adapted, planar_case = _adapted_hydrostatic_inputs(matrix, case)
    root = ET.fromstring(COMMON.render_mesh(adapted, planar_case))
    arrays = {
        item.attrib.get("Name"): item
        for item in root.findall(".//PointData/DataArray")
    }
    required = {"level_set", "p_negative", "p_positive"}
    if not required.issubset(arrays):
        raise RuntimeError("hydrostatic mesh template changed")
    resolution = int(case["resolution"])
    coordinates = [
        (i / resolution, j / resolution)
        for j in range(resolution + 1)
        for i in range(resolution + 1)
    ]
    level_set = [level_set_value(case, x, y) for x, y in coordinates]
    signed_jump = expected_signed_pressure_jump(case)
    zeros = [0.0] * len(coordinates)
    arrays["level_set"].text = "\n" + _format_values(level_set) + "\n        "
    arrays["p_negative"].text = "\n" + _format_values(zeros) + "\n        "
    arrays["p_positive"].text = (
        "\n" + _format_values([-signed_jump] * len(coordinates)) + "\n        "
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def render_wall(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    adapted, _ = _adapted_hydrostatic_inputs(matrix, case)
    return HYDROSTATIC.render_wall(adapted)


def pressure_gauge_vertex(case: dict[str, Any]) -> int:
    resolution = int(case["resolution"])
    values = [
        level_set_value(case, i / resolution, j / resolution)
        for j in range(resolution + 1)
        for i in range(resolution + 1)
    ]
    point = min(range(len(values)), key=lambda index: (values[index], index))
    if not math.isfinite(values[point]) or values[point] >= 0.0:
        raise ValueError("static-drop mesh has no strictly negative gauge vertex")
    return point + 1


def render_pressure_gauge(case: dict[str, Any]) -> str:
    return f"node_id,pressure\n{pressure_gauge_vertex(case)},0\n"


def render_solver(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    adapted, planar_case = _adapted_hydrostatic_inputs(matrix, case)
    root = ET.fromstring(HYDROSTATIC.render_solver(adapted, planar_case))
    fluid = root.find("Add_equation[@type='stokes']")
    level_set = root.find("Add_equation[@type='level_set']")
    if fluid is None or level_set is None:
        raise RuntimeError("hydrostatic solver template changed")
    surface = fluid.find("Two_fluid_surface_tension")
    if surface is None:
        raise RuntimeError("two-fluid surface-tension control is missing")
    surface.text = format(float(case["surface_tension"]), ".17g")
    for name in ("Force_x", "Force_y", "Force_z"):
        force = fluid.find(name)
        if force is None:
            raise RuntimeError(f"two-fluid solver template lacks {name}")
        force.text = "0"
    reconcile = level_set.find("Conservative_phase_reconcile_geometry")
    if reconcile is None:
        raise RuntimeError("conservative phase geometry control is missing")
    reconcile.text = "true"
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def _numeric(record: dict[str, Any], key: str) -> float:
    if not isinstance(record, dict) or key not in record:
        raise ValueError(f"missing numeric field {key}")
    try:
        value = float(record[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numeric field {key}")
    return value


def _boolean(record: dict[str, Any], key: str) -> bool:
    if not isinstance(record, dict) or key not in record:
        raise ValueError(f"missing boolean field {key}")
    value = record[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"invalid boolean field {key}")


def evaluate_static_observables(
    case: dict[str, Any], matrix: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    thresholds = matrix["thresholds"]
    geometry = expected_geometry(case)
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )

    def number(key: str) -> float:
        try:
            value = _numeric(record, key)
        except ValueError as error:
            check(key, False, str(error), "finite numeric field")
            return math.nan
        metrics[key] = value
        return value

    interface_points = number("interface_quadrature_points")
    minimum_points = thresholds["minimum_interface_quadrature_points"]
    check(
        "interface_quadrature_points",
        math.isfinite(interface_points)
        and interface_points == int(interface_points)
        and interface_points >= minimum_points,
        interface_points,
        f"integer >= {minimum_points}",
    )
    interface_measure = number("interface_measure")
    interface_error = (
        abs(interface_measure - geometry["interface_measure"])
        / geometry["interface_measure"]
        if math.isfinite(interface_measure) and interface_measure > 0.0
        else math.inf
    )
    metrics["interface_measure_relative_error"] = interface_error
    check(
        "interface_measure_relative_error",
        interface_error <= thresholds["per_case_interface_measure_relative"],
        interface_error,
        thresholds["per_case_interface_measure_relative"],
    )

    negative_volume = number("negative_volume")
    positive_volume = number("positive_volume")
    partition_error = abs(negative_volume + positive_volume - 1.0)
    metrics["phase_partition_absolute_error"] = partition_error
    check(
        "phase_partition_absolute_error",
        math.isfinite(partition_error)
        and partition_error <= thresholds["phase_partition_absolute"],
        partition_error,
        thresholds["phase_partition_absolute"],
    )
    for phase, observed, expected in (
        ("negative", negative_volume, geometry["negative_volume"]),
        ("positive", positive_volume, geometry["positive_volume"]),
    ):
        relative = (
            abs(observed - expected) / expected
            if math.isfinite(observed) and expected > 0.0
            else math.inf
        )
        metrics[f"{phase}_volume_relative_error"] = relative
    liquid_phase = case["liquid_phase"]
    interior_volume = negative_volume if liquid_phase == "negative" else positive_volume
    interior_error = (
        abs(interior_volume - geometry["interior_volume"])
        / geometry["interior_volume"]
        if math.isfinite(interior_volume)
        else math.inf
    )
    metrics["interior_volume_relative_error"] = interior_error
    check(
        "interior_volume_relative_error",
        interior_error <= thresholds["per_case_interior_volume_relative"],
        interior_error,
        thresholds["per_case_interior_volume_relative"],
    )

    mean_jump = number("mean_pressure_jump")
    expected_jump = expected_signed_pressure_jump(case)
    pressure_error = (
        abs(mean_jump - expected_jump) / abs(expected_jump)
        if math.isfinite(mean_jump) and expected_jump != 0.0
        else math.inf
    )
    metrics["expected_signed_pressure_jump"] = expected_jump
    metrics["pressure_jump_relative_error"] = pressure_error
    check(
        "pressure_jump_relative_error",
        pressure_error <= thresholds["per_case_pressure_jump_relative"],
        pressure_error,
        thresholds["per_case_pressure_jump_relative"],
    )
    pressure_jump_sq = number("pressure_jump_sq")
    pressure_variance = (
        max(0.0, pressure_jump_sq / interface_measure - mean_jump * mean_jump)
        if math.isfinite(pressure_jump_sq)
        and math.isfinite(interface_measure)
        and interface_measure > 0.0
        and math.isfinite(mean_jump)
        else math.inf
    )
    pressure_dispersion = (
        math.sqrt(pressure_variance) / abs(expected_jump)
        if math.isfinite(pressure_variance) and expected_jump != 0.0
        else math.inf
    )
    metrics["pressure_jump_relative_dispersion"] = pressure_dispersion
    check(
        "pressure_jump_relative_dispersion",
        pressure_dispersion
        <= thresholds["per_case_pressure_jump_relative_dispersion"],
        pressure_dispersion,
        thresholds["per_case_pressure_jump_relative_dispersion"],
    )

    surface_tension = float(case["surface_tension"])
    phase_capillary_numbers = []
    for phase in ("negative", "positive"):
        speed_key = f"{phase}_maximum_quadrature_speed"
        speed = number(speed_key)
        nonnegative = math.isfinite(speed) and speed >= 0.0
        check(speed_key, nonnegative, speed, "finite nonnegative")
        viscosity = float(case[f"{phase}_viscosity"])
        capillary_number = (
            viscosity * speed / surface_tension if nonnegative else math.inf
        )
        metrics[f"{phase}_maximum_capillary_number"] = capillary_number
        phase_capillary_numbers.append(capillary_number)
    maximum_capillary_number = max(phase_capillary_numbers)
    metrics["maximum_phase_capillary_number"] = maximum_capillary_number
    check(
        "maximum_phase_capillary_number",
        maximum_capillary_number
        <= thresholds["per_case_maximum_phase_capillary_number"],
        maximum_capillary_number,
        thresholds["per_case_maximum_phase_capillary_number"],
    )

    for key in (
        "velocity_jump_sq",
        "negative_kinetic_energy",
        "positive_kinetic_energy",
        "traction_jump_sq",
    ):
        value = number(key)
        check(key, math.isfinite(value) and value >= 0.0, value, "finite nonnegative")
    velocity_jump_sq = metrics.get("velocity_jump_sq", math.inf)
    check(
        "velocity_jump_squared_gate",
        velocity_jump_sq <= thresholds["per_case_velocity_jump_squared"],
        velocity_jump_sq,
        thresholds["per_case_velocity_jump_squared"],
    )
    normal_flux_jump = number("normal_flux_jump")
    check(
        "normal_flux_jump",
        math.isfinite(normal_flux_jump)
        and abs(normal_flux_jump) <= thresholds["per_case_normal_flux_absolute"],
        normal_flux_jump,
        f"absolute <= {thresholds['per_case_normal_flux_absolute']}",
    )
    nonlinear_iterations = number("nonlinear_iterations")
    linear_iterations = number("linear_iterations")
    check(
        "nonlinear_iterations",
        nonlinear_iterations == int(nonlinear_iterations)
        and 0 <= nonlinear_iterations
        <= thresholds["maximum_nonlinear_iterations"],
        nonlinear_iterations,
        thresholds["maximum_nonlinear_iterations"],
    )
    check(
        "linear_iterations",
        linear_iterations == int(linear_iterations)
        and 0 <= linear_iterations <= thresholds["maximum_linear_iterations"],
        linear_iterations,
        thresholds["maximum_linear_iterations"],
    )
    final_residual = number("nonlinear_final_residual_norm")
    initial_residual = number("nonlinear_initial_residual_norm")
    check(
        "nonlinear_final_residual_norm",
        0.0 <= final_residual <= thresholds["maximum_final_nonlinear_residual"],
        final_residual,
        thresholds["maximum_final_nonlinear_residual"],
    )
    check(
        "nonlinear_residual_nonincrease",
        math.isfinite(initial_residual)
        and math.isfinite(final_residual)
        and final_residual <= initial_residual + thresholds["maximum_final_nonlinear_residual"],
        {"initial": initial_residual, "final": final_residual},
        "final no larger than initial within solver tolerance",
    )
    for key in ("nonlinear_converged", "linear_converged"):
        try:
            value = _boolean(record, key)
        except ValueError as error:
            check(key, False, str(error), True)
        else:
            metrics[key] = value
            check(key, value, value, True)

    if int(case["resolution"]) == max(matrix["refinement"]["resolutions"]):
        for name, actual, threshold_key in (
            (
                "finest_interface_measure_relative_error",
                interface_error,
                "finest_interface_measure_relative",
            ),
            (
                "finest_interior_volume_relative_error",
                interior_error,
                "finest_interior_volume_relative",
            ),
            (
                "finest_pressure_jump_relative_error",
                pressure_error,
                "finest_pressure_jump_relative",
            ),
            (
                "finest_pressure_jump_relative_dispersion",
                pressure_dispersion,
                "finest_pressure_jump_relative_dispersion",
            ),
            (
                "finest_maximum_phase_capillary_number",
                maximum_capillary_number,
                "finest_maximum_phase_capillary_number",
            ),
        ):
            limit = thresholds[threshold_key]
            check(name, actual <= limit, actual, limit)

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "metrics": metrics,
    }


def physical_reversal_metrics(
    case: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, float]:
    liquid = case["liquid_phase"]
    gas = "positive" if liquid == "negative" else "negative"

    def value(name: str, default: float | None = None) -> float:
        item = metrics.get(name, default)
        if item is None:
            raise ValueError(f"missing reversal metric {name}")
        return _finite_number(item, f"reversal metric {name}")

    return {
        "interface_measure": value("interface_measure"),
        "interface_measure_relative_error": value(
            "interface_measure_relative_error"
        ),
        "interior_volume": value(f"{liquid}_volume"),
        "interior_volume_relative_error": value(
            "interior_volume_relative_error",
            value(f"{liquid}_volume_relative_error", 0.0),
        ),
        "exterior_volume": value(f"{gas}_volume"),
        "pressure_jump_magnitude": abs(value("mean_pressure_jump")),
        "pressure_jump_relative_error": value("pressure_jump_relative_error"),
        "liquid_maximum_capillary_number": value(
            f"{liquid}_maximum_capillary_number"
        ),
        "gas_maximum_capillary_number": value(
            f"{gas}_maximum_capillary_number"
        ),
        "maximum_phase_capillary_number": value(
            "maximum_phase_capillary_number"
        ),
    }


def evaluate_reversal_pairs(
    results: list[dict[str, Any]], matrix: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["reversal_pair"], []).append(result)
    checks = []
    thresholds = matrix["thresholds"]
    absolute = thresholds["side_reversal_absolute"]
    relative = thresholds["side_reversal_relative"]
    for pair in sorted({case["reversal_pair"] for case in matrix["cases"]}):
        members = grouped.get(pair, [])
        if len(members) != 2 or {item["level_set_sign"] for item in members} != {
            -1,
            1,
        }:
            checks.append(
                {
                    "name": f"{pair}:complete_pair",
                    "passed": False,
                    "actual": len(members),
                    "expected": 2,
                }
            )
            continue
        forward = next(item for item in members if item["level_set_sign"] == 1)
        reverse = next(item for item in members if item["level_set_sign"] == -1)
        if not forward.get("passed") or not reverse.get("passed"):
            checks.append(
                {
                    "name": f"{pair}:case_prerequisites",
                    "passed": False,
                    "actual": [forward.get("passed"), reverse.get("passed")],
                    "expected": [True, True],
                }
            )
            continue
        first = forward.get("reversal_metrics", {})
        second = reverse.get("reversal_metrics", {})
        keys = sorted(set(first) | set(second))
        for key in keys:
            first_value = first.get(key)
            second_value = second.get(key)
            numeric = (
                isinstance(first_value, (int, float))
                and not isinstance(first_value, bool)
                and isinstance(second_value, (int, float))
                and not isinstance(second_value, bool)
                and math.isfinite(float(first_value))
                and math.isfinite(float(second_value))
            )
            tolerance = absolute + relative * max(
                abs(float(first_value)) if numeric else 0.0,
                abs(float(second_value)) if numeric else 0.0,
            )
            passed = numeric and abs(float(first_value) - float(second_value)) <= tolerance
            checks.append(
                {
                    "name": f"{pair}:{key}",
                    "passed": passed,
                    "first": first_value,
                    "second": second_value,
                    "tolerance": tolerance,
                }
            )
    failed = [item["name"] for item in checks if not item["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def _observed_orders(values: list[float], resolutions: list[int]) -> list[float]:
    orders = []
    for index in range(len(values) - 1):
        coarse = values[index]
        fine = values[index + 1]
        ratio = resolutions[index + 1] / resolutions[index]
        if coarse == 0.0 and fine == 0.0:
            orders.append(math.inf)
        elif coarse > 0.0 and fine > 0.0 and ratio > 1.0:
            orders.append(math.log(coarse / fine) / math.log(ratio))
        else:
            orders.append(-math.inf)
    return orders


def evaluate_refinement_families(
    results: list[dict[str, Any]], matrix: dict[str, Any]
) -> dict[str, Any]:
    case_lookup = {case["case_id"]: case for case in matrix["cases"]}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["refinement_family"], []).append(result)
    family_records = []
    checks = []
    thresholds = matrix["thresholds"]
    expected_resolutions = matrix["refinement"]["resolutions"]
    convergence_metrics = (
        "interface_measure_relative_error",
        "interior_volume_relative_error",
        "pressure_jump_relative_error",
    )
    for family in sorted(
        {
            case["refinement_family"]
            for case in matrix["cases"]
        }
    ):
        members = sorted(grouped.get(family, []), key=lambda item: item["resolution"])
        resolutions = [item["resolution"] for item in members]
        complete = resolutions == expected_resolutions and all(
            item.get("passed") for item in members
        )
        checks.append(
            {
                "name": f"{family}:complete_refinement_family",
                "passed": complete,
                "actual": resolutions,
                "expected": expected_resolutions,
            }
        )
        if not complete:
            family_records.append(
                {
                    "refinement_family": family,
                    "resolutions": resolutions,
                    "passed": False,
                }
            )
            continue
        record: dict[str, Any] = {
            "refinement_family": family,
            "resolutions": resolutions,
        }
        for metric in convergence_metrics:
            values = []
            for result in members:
                metrics = result["metrics"]
                if metric == "interior_volume_relative_error" and metric not in metrics:
                    case = case_lookup[result["case_id"]]
                    liquid = case["liquid_phase"]
                    expected = expected_geometry(case)["interior_volume"]
                    values.append(abs(float(metrics[f"{liquid}_volume"]) - expected) / expected)
                else:
                    values.append(float(metrics[metric]))
            orders = _observed_orders(values, resolutions)
            record[metric] = {"values": values, "observed_orders": orders}
            minimum_order = thresholds["minimum_observed_order"]
            for index, order in enumerate(orders):
                checks.append(
                    {
                        "name": f"{family}:{metric}:observed_order_{index + 1}",
                        "passed": order >= minimum_order,
                        "actual": order,
                        "expected": f">= {minimum_order}",
                    }
                )
        finest_metrics = members[-1]["metrics"]
        finest = {
            "interface_measure_relative_error": float(
                finest_metrics["interface_measure_relative_error"]
            ),
            "interior_volume_relative_error": float(
                finest_metrics.get(
                    "interior_volume_relative_error",
                    record["interior_volume_relative_error"]["values"][-1],
                )
            ),
            "pressure_jump_relative_error": float(
                finest_metrics["pressure_jump_relative_error"]
            ),
            "pressure_jump_relative_dispersion": float(
                finest_metrics.get("pressure_jump_relative_dispersion", 0.0)
            ),
            "maximum_phase_capillary_number": float(
                finest_metrics["maximum_phase_capillary_number"]
            ),
        }
        record["finest"] = finest
        for metric, threshold_key in (
            (
                "interface_measure_relative_error",
                "finest_interface_measure_relative",
            ),
            (
                "interior_volume_relative_error",
                "finest_interior_volume_relative",
            ),
            (
                "pressure_jump_relative_error",
                "finest_pressure_jump_relative",
            ),
            (
                "pressure_jump_relative_dispersion",
                "finest_pressure_jump_relative_dispersion",
            ),
            (
                "maximum_phase_capillary_number",
                "finest_maximum_phase_capillary_number",
            ),
        ):
            limit = thresholds[threshold_key]
            checks.append(
                {
                    "name": f"{family}:finest_{metric}",
                    "passed": finest[metric] <= limit,
                    "actual": finest[metric],
                    "expected": f"<= {limit}",
                }
            )
        pressure_values = record["pressure_jump_relative_error"]["values"]
        pressure_orders = record["pressure_jump_relative_error"]["observed_orders"]
        last_order = pressure_orders[-1]
        if math.isfinite(last_order) and last_order > 0.0:
            gci = thresholds["gci_safety_factor"] * abs(
                pressure_values[-2] - pressure_values[-1]
            ) / (2.0**last_order - 1.0)
        elif pressure_values[-1] == 0.0 and pressure_values[-2] == 0.0:
            gci = 0.0
        else:
            gci = math.inf
        record["pressure_jump_gci"] = gci
        checks.append(
            {
                "name": f"{family}:finest_pressure_jump_gci",
                "passed": gci <= thresholds["finest_pressure_jump_gci"],
                "actual": gci,
                "expected": f"<= {thresholds['finest_pressure_jump_gci']}",
            }
        )
        record["passed"] = True
        family_records.append(record)
    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "families": family_records,
    }


def parse_case_output(output: str) -> dict[str, Any]:
    return HYDROSTATIC.parse_case_output(output)


def evaluate_case(
    case: dict[str, Any],
    matrix: dict[str, Any],
    parsed: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    interface = parsed.get("interface")
    static = evaluate_static_observables(case, matrix, interface)
    checks = list(static["checks"])
    metrics = dict(static["metrics"])
    tolerance = matrix["thresholds"]["phase_partition_absolute"]

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )

    def number(record_name: str, key: str, label: str | None = None) -> float:
        name = label or key
        try:
            value = _numeric(parsed.get(record_name), key)
        except ValueError as error:
            check(name, False, str(error), "finite numeric field")
            return math.nan
        metrics[name] = value
        return value

    def near_zero(record_name: str, key: str, label: str | None = None) -> None:
        name = label or key
        value = number(record_name, key, name)
        check(
            name,
            math.isfinite(value) and abs(value) <= tolerance,
            value,
            f"absolute <= {tolerance}",
        )

    def boolean_value(
        record_name: str, key: str, expected: bool, label: str | None = None
    ) -> None:
        name = label or key
        try:
            value = _boolean(parsed.get(record_name), key)
        except ValueError as error:
            check(name, False, str(error), expected)
            return
        metrics[name] = value
        check(name, value is expected, value, expected)

    check("process_return_code", return_code == 0, return_code, 0)
    accepted_steps = parsed.get("accepted_step_count", 0)
    check("accepted_step_count", accepted_steps == 1, accepted_steps, 1)

    expected_dofs = 3 * (int(case["resolution"]) + 1) ** 2
    expected_ranks = set(range(int(case["mpi_ranks"])))
    for phase in ("negative", "positive"):
        records = [
            record
            for record in parsed.get("initializers", [])
            if (
                record.get("velocity_field"),
                record.get("pressure_field"),
            )
            == (f"u_{phase}", f"p_{phase}")
        ]
        by_rank: dict[int, int] = {}
        duplicate = False
        for record in records:
            try:
                rank_value = _numeric(record, "log_rank")
                dof_value = _numeric(record, "initialized_dofs")
            except ValueError:
                continue
            rank = int(rank_value)
            dofs = int(dof_value)
            if rank_value != rank or dof_value != dofs:
                continue
            duplicate = duplicate or rank in by_rank
            by_rank[rank] = dofs
        passed = (
            not duplicate
            and set(by_rank) == expected_ranks
            and all(value == expected_dofs for value in by_rank.values())
        )
        check(
            f"{phase}_phase_initializer",
            passed,
            {"by_rank": by_rank, "duplicate": duplicate},
            {
                "by_rank": {rank: expected_dofs for rank in sorted(expected_ranks)},
                "duplicate": False,
            },
        )

    observed_negative_volume = metrics.get("negative_volume", math.nan)
    stage_previous = number(
        "phase_stage", "previous_measure", "phase_stage_previous_measure"
    )
    stage_accepted = number(
        "phase_stage", "accepted_measure", "phase_stage_accepted_measure"
    )
    stage_boundary = number(
        "phase_stage", "boundary_transfer", "phase_stage_boundary_transfer"
    )
    stage_divergence = number(
        "phase_stage", "divergence_source", "phase_stage_divergence_source"
    )
    stage_invariant_tolerance = number(
        "phase_stage",
        "stage_invariant_tolerance",
        "phase_stage_invariant_tolerance",
    )
    stage_boundary_tolerance = number(
        "phase_stage",
        "boundary_mass_tolerance",
        "phase_stage_boundary_mass_tolerance",
    )
    stage_balance_tolerance = max(tolerance, stage_invariant_tolerance)
    stage_balance_residual = (
        stage_accepted - stage_previous - stage_boundary - stage_divergence
    )
    metrics["phase_stage_measure_balance_residual"] = stage_balance_residual
    check(
        "phase_stage_measure_balance",
        math.isfinite(stage_balance_residual)
        and abs(stage_balance_residual) <= stage_balance_tolerance,
        stage_balance_residual,
        f"absolute <= {stage_balance_tolerance}",
    )
    check(
        "phase_stage_boundary_transfer",
        math.isfinite(stage_boundary)
        and abs(stage_boundary) <= stage_boundary_tolerance,
        stage_boundary,
        f"absolute <= {stage_boundary_tolerance}",
    )
    for key in (
        "global_balance_residual",
        "max_local_balance_residual",
        "max_component_balance_residual",
    ):
        value = number("phase_stage", key, f"phase_stage_{key}")
        check(
            f"phase_stage_{key}_gate",
            math.isfinite(value) and abs(value) <= stage_balance_tolerance,
            value,
            f"absolute <= {stage_balance_tolerance}",
        )
    stage_courant = number("phase_stage", "courant", "phase_stage_courant")
    stage_maximum_courant = number(
        "phase_stage",
        "stage_maximum_courant",
        "phase_stage_maximum_courant",
    )
    check(
        "phase_stage_courant_gate",
        math.isfinite(stage_courant)
        and math.isfinite(stage_maximum_courant)
        and 0.0 <= stage_courant <= stage_maximum_courant,
        stage_courant,
        {"minimum": 0.0, "maximum": stage_maximum_courant},
    )
    limited_edges = number(
        "phase_stage", "limited_edges", "phase_stage_limited_edges"
    )
    check(
        "phase_stage_limited_edges_nonnegative_integer",
        limited_edges >= 0.0 and limited_edges == int(limited_edges),
        limited_edges,
        "nonnegative integer",
    )

    for key in ("phase_measure", "retained_geometry_measure"):
        value = number("phase_geometry", key, f"phase_geometry_{key}")
        check(
            f"phase_geometry_{key}_matches_negative_volume",
            math.isfinite(value)
            and math.isfinite(observed_negative_volume)
            and abs(value - observed_negative_volume) <= tolerance,
            value,
            observed_negative_volume,
        )
    for key in (
        "measure_mismatch",
        "max_nodal_moment_mismatch",
        "nodal_moment_residual_norm",
        "interface_displacement_bound",
    ):
        near_zero("phase_geometry", key, f"phase_geometry_{key}")
    boolean_value(
        "phase_geometry", "reconciliation_enabled", True,
        "phase_geometry_reconciliation_enabled",
    )
    reconciliation_iterations = number(
        "phase_geometry",
        "reconciliation_iterations",
        "phase_geometry_reconciliation_iterations",
    )
    check(
        "phase_geometry_reconciliation_iterations_integer",
        reconciliation_iterations >= 0.0
        and reconciliation_iterations == int(reconciliation_iterations),
        reconciliation_iterations,
        "nonnegative integer",
    )

    maintenance_values = {}
    for key in (
        "raw_post_transport_phase_measure",
        "post_limit_phase_measure",
        "raw_post_transport_geometry_measure",
        "post_reinitialization_phase_measure",
        "post_reinitialization_geometry_measure",
        "post_correction_phase_measure",
        "post_correction_geometry_measure",
        "retained_assembly_measure",
    ):
        value = number("maintenance", key, f"maintenance_{key}")
        maintenance_values[key] = value
    maintenance_expected = {
        "raw_post_transport_phase_measure": stage_accepted,
        "post_limit_phase_measure": stage_accepted,
        "raw_post_transport_geometry_measure": observed_negative_volume,
        "post_reinitialization_phase_measure": stage_accepted,
        "post_reinitialization_geometry_measure": observed_negative_volume,
        "post_correction_phase_measure": observed_negative_volume,
        "post_correction_geometry_measure": observed_negative_volume,
        "retained_assembly_measure": observed_negative_volume,
    }
    for key, expected in maintenance_expected.items():
        value = maintenance_values[key]
        check(
            f"maintenance_{key}_identity",
            math.isfinite(value)
            and math.isfinite(expected)
            and abs(value - expected) <= stage_balance_tolerance,
            value,
            expected,
        )
    maintenance_boundary = number(
        "maintenance",
        "total_physical_boundary_mass_transfer",
        "maintenance_total_physical_boundary_mass_transfer",
    )
    maintenance_boundary_tolerance = number(
        "maintenance",
        "boundary_mass_tolerance",
        "maintenance_boundary_mass_tolerance",
    )
    check(
        "maintenance_boundary_mass_transfer_gate",
        math.isfinite(maintenance_boundary)
        and abs(maintenance_boundary) <= maintenance_boundary_tolerance,
        maintenance_boundary,
        f"absolute <= {maintenance_boundary_tolerance}",
    )
    maintenance_component_residual = number(
        "maintenance",
        "transport_max_component_balance_residual",
        "maintenance_transport_max_component_balance_residual",
    )
    maintenance_invariant_tolerance = number(
        "maintenance",
        "stage_invariant_tolerance",
        "maintenance_stage_invariant_tolerance",
    )
    check(
        "maintenance_component_balance_residual_gate",
        math.isfinite(maintenance_component_residual)
        and abs(maintenance_component_residual)
        <= max(tolerance, maintenance_invariant_tolerance),
        maintenance_component_residual,
        f"absolute <= {max(tolerance, maintenance_invariant_tolerance)}",
    )
    near_zero(
        "maintenance",
        "reconciliation_interface_displacement_bound",
        "maintenance_reconciliation_interface_displacement_bound",
    )
    for key in (
        "transport_component_balance_satisfied",
        "transport_component_measure_closure_satisfied",
    ):
        boolean_value("maintenance", key, True, f"maintenance_{key}")

    boolean_value("interface", "momentum_reconciliation_applicable", True)
    boolean_value("interface", "velocity_update_applied", False)
    boolean_value("interface", "momentum_reconciliation_satisfied", True)
    boolean_value("interface", "accepted_stage_numerics_applicable", True)
    iteration_scope = interface.get("phase_iteration_scope") if isinstance(
        interface, dict
    ) else None
    check(
        "phase_iteration_scope",
        iteration_scope == "shared_coupled_solve",
        iteration_scope,
        "shared_coupled_solve",
    )
    if iteration_scope is not None:
        metrics["phase_iteration_scope"] = iteration_scope

    artifact = parsed.get("artifact")
    if artifact is not None:
        check(
            "phase_flux_artifact_written",
            artifact.get("outcome") == "written",
            artifact.get("outcome"),
            "written",
        )

    failed = [item["name"] for item in checks if not item["passed"]]
    result = {
        "case_id": case["case_id"],
        "reversal_pair": case["reversal_pair"],
        "refinement_family": case["refinement_family"],
        "center_class": case["center_class"],
        "resolution": case["resolution"],
        "level_set_sign": case["level_set_sign"],
        "mpi_ranks": case["mpi_ranks"],
        "passed": not failed,
        "failed_checks": failed,
        "metrics": metrics,
        "checks": checks,
    }
    if result["passed"]:
        result["reversal_metrics"] = physical_reversal_metrics(case, metrics)
    else:
        result["reversal_metrics"] = {}
    return result


def validate_effective_configuration(
    document: Any, case: dict[str, Any]
) -> dict[str, Any]:
    result = COMMON.validate_effective_configuration(document)
    momentum_modules = [
        module
        for module in document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "incompressible_two_fluid"
    ]
    transport_modules = [
        module
        for module in document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "level_set_transport"
    ]
    if len(momentum_modules) != 1 or len(transport_modules) != 1:
        raise ValueError("effective configuration lacks a unique two-fluid pair")
    momentum = momentum_modules[0]
    transport = transport_modules[0]
    if momentum.get("momentum_operator") != "stokes":
        raise ValueError("effective configuration changed the momentum operator")
    material = momentum.get("material")
    if not isinstance(material, dict):
        raise ValueError("effective configuration lacks material data")
    for key in (
        "negative_density",
        "negative_viscosity",
        "positive_density",
        "positive_viscosity",
    ):
        if _finite_number(material.get(key), f"effective {key}") != float(case[key]):
            raise ValueError("effective configuration changed material data")
    force = momentum.get("body_force")
    if (
        not isinstance(force, list)
        or len(force) != 2
        or any(_finite_number(value, "effective body force") != 0.0 for value in force)
    ):
        raise ValueError("effective configuration changed the zero body force")
    interface = momentum.get("interface")
    if not isinstance(interface, dict):
        raise ValueError("effective configuration lacks interface data")
    surface_tension = _finite_number(
        interface.get("surface_tension"), "effective surface tension"
    )
    if surface_tension != float(case["surface_tension"]):
        raise ValueError("effective configuration changed surface tension")
    if interface.get("prescribed_pressure_jump_applicable") is not False:
        raise ValueError("effective configuration enabled a pressure-jump target")
    if interface.get("prescribed_viscous_traction_jump_applicable") is not False:
        raise ValueError("effective configuration enabled a viscous-traction target")

    gauge = momentum.get("pressure_space")
    expected_gauge = pressure_gauge_vertex(case)
    if (
        not isinstance(gauge, dict)
        or gauge.get("representation") != "separate_phase_fields"
        or gauge.get("shared_gauge_count") != 1
        or gauge.get("shared_gauge_policy") != "explicit_global_vertex_gid"
        or gauge.get("shared_gauge_field") != "p_negative"
        or gauge.get("shared_gauge_id_type") != "Global_vertex_gid"
        or gauge.get("shared_gauge_vertex_gid") != expected_gauge
        or gauge.get("shared_gauge_value") != 0.0
    ):
        raise ValueError("effective configuration changed the pressure gauge")

    boundaries = momentum.get("boundary_conditions")
    entries = (
        boundaries.get("shared_velocity_dirichlet")
        if isinstance(boundaries, dict)
        else None
    )
    entry = entries[0] if isinstance(entries, list) and len(entries) == 1 else None
    values = entry.get("values") if isinstance(entry, dict) else None
    if not (
        isinstance(boundaries, dict)
        and boundaries.get("shared_velocity_dirichlet_count") == 1
        and boundaries.get("shared_velocity_dirichlet_policy")
        == "identical_external_data_on_both_phase_restrictions"
        and boundaries.get("negative_phase_local_velocity_dirichlet_count") == 0
        and boundaries.get("positive_phase_local_velocity_dirichlet_count") == 0
        and isinstance(entry, dict)
        and entry.get("active_components") == [True, True]
        and isinstance(values, list)
        and len(values) == 2
        and all(
            isinstance(value, dict)
            and value.get("kind") == "literal"
            and value.get("value") == 0.0
            for value in values
        )
    ):
        raise ValueError("effective configuration changed the closed-wall boundary")

    conservative = transport.get("conservative_phase")
    if (
        not isinstance(conservative, dict)
        or conservative.get("enabled") is not True
        or conservative.get("boundary_flux_policy")
        != "closed_domain_discrete_q_flux_only"
        or conservative.get("reconcile_geometry") is not True
    ):
        raise ValueError("effective configuration changed stationary phase transport")
    result.update(
        {
            "momentum_operator": "stokes",
            "surface_tension": surface_tension,
            "body_force": [0.0, 0.0],
            "shared_pressure_gauge_vertex_gid": expected_gauge,
            "shared_velocity_dirichlet_count": 1,
            "stationary_geometry_reconciliation": True,
        }
    )
    return result


def qualification_environment(
    matrix: dict[str, Any], inherited: Any, case_directory: Path
) -> dict[str, str]:
    environment = dict(inherited)
    environment.pop("SVMP_NEWTON_ABS_TOLERANCE", None)
    environment.pop("SVMP_NEWTON_REL_TOLERANCE", None)
    environment.update(
        {
            "OMP_NUM_THREADS": str(matrix["execution"]["omp_threads"]),
            "OPENBLAS_NUM_THREADS": "1",
            "TMPDIR": str(case_directory / "tmp"),
            "SVMP_OOP_SOLVER_TRACE": "0",
        }
    )
    return environment


def execution_outcome(
    numerical_passed: bool, qualification_eligible: bool
) -> dict[str, Any]:
    if not numerical_passed:
        return {
            "outcome": "FAIL",
            "two_fluid_circular_static_drop_gate_passed": False,
            "exit_code": 1,
        }
    if qualification_eligible:
        return {
            "outcome": "PASS",
            "two_fluid_circular_static_drop_gate_passed": True,
            "exit_code": 0,
        }
    return {
        "outcome": "DEVELOPMENT_PASS",
        "two_fluid_circular_static_drop_gate_passed": False,
        "exit_code": 0,
    }


def run_case(
    case: dict[str, Any],
    matrix: dict[str, Any],
    solver: Path,
    launcher: Path,
    output_directory: Path,
) -> dict[str, Any]:
    case_directory = output_directory / case["case_id"]
    case_directory.mkdir(mode=0o700)
    mesh_path = case_directory / "mesh.vtu"
    wall_path = case_directory / "wall.vtp"
    gauge_path = case_directory / "pressure_gauge.csv"
    solver_path = case_directory / "solver.xml"
    COMMON.write_text_create_only(mesh_path, render_mesh(matrix, case))
    COMMON.write_text_create_only(wall_path, render_wall(matrix, case))
    COMMON.write_text_create_only(
        gauge_path, render_pressure_gauge(case)
    )
    COMMON.write_text_create_only(solver_path, render_solver(matrix, case))
    temporary_directory = case_directory / "tmp"
    temporary_directory.mkdir(mode=0o700)
    stdout_path = case_directory / "stdout.log"
    stderr_path = case_directory / "stderr.log"
    command = [
        str(launcher),
        "--oversubscribe",
        "-n",
        str(case["mpi_ranks"]),
        str(solver),
        solver_path.name,
    ]
    execution = COMMON.run_monitored(
        command,
        qualification_environment(matrix, os.environ, case_directory),
        case_directory,
        stdout_path,
        stderr_path,
        matrix["execution"]["wall_time_seconds_per_case"],
        matrix["execution"]["memory_mib_per_case"],
        matrix["execution"]["output_mib_per_case"],
    )
    output = stdout_path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = parse_case_output(output)
        result = evaluate_case(case, matrix, parsed, execution["return_code"])
    except (KeyError, TypeError, ValueError) as error:
        result = {
            "case_id": case["case_id"],
            "reversal_pair": case["reversal_pair"],
            "refinement_family": case["refinement_family"],
            "center_class": case["center_class"],
            "resolution": case["resolution"],
            "level_set_sign": case["level_set_sign"],
            "mpi_ranks": case["mpi_ranks"],
            "passed": False,
            "failed_checks": ["output_parse"],
            "metrics": {},
            "reversal_metrics": {},
            "checks": [
                {
                    "name": "output_parse",
                    "passed": False,
                    "actual": str(error),
                    "expected": "complete consistent output",
                }
            ],
        }
    result["execution"] = execution
    result["input_sha256"] = {
        "mesh": COMMON.sha256_file(mesh_path),
        "wall": COMMON.sha256_file(wall_path),
        "pressure_gauge": COMMON.sha256_file(gauge_path),
        "solver": COMMON.sha256_file(solver_path),
    }
    effective_path = case_directory / "effective_configuration.json"
    try:
        effective = validate_effective_configuration(
            COMMON.read_json_strict(effective_path), case
        )
    except (OSError, TypeError, ValueError) as error:
        effective = {"error": str(error)}
        result["checks"].append(
            {
                "name": "effective_configuration",
                "passed": False,
                "actual": str(error),
                "expected": "closed-wall circular two-fluid configuration",
            }
        )
        result["failed_checks"].append("effective_configuration")
        result["passed"] = False
        result["reversal_metrics"] = {}
    else:
        result["checks"].append(
            {
                "name": "effective_configuration",
                "passed": True,
                "actual": effective,
                "expected": effective,
            }
        )
    result["effective_configuration"] = effective
    COMMON.write_json_create_only(case_directory / "result.json", result)
    return result


def qualification_eligibility(
    matrix: dict[str, Any],
    provenance: dict[str, Any],
    input_identity: dict[str, dict[str, Any]],
) -> bool:
    return (
        matrix["status"] == "FROZEN_BEFORE_EXECUTION"
        and provenance["tracked_clean"]
        and all(item["matches_head"] for item in input_identity.values())
    )


def _qualification_input_identity() -> dict[str, dict[str, Any]]:
    paths = {
        "matrix": DEFAULT_MATRIX,
        "runner": SCRIPT_PATH,
        "runner_test": RUNNER_TEST_PATH,
        "hydrostatic_runner_dependency": HYDROSTATIC_RUNNER_PATH,
        "viscous_jump_runner_dependency": HYDROSTATIC.VISCOUS_RUNNER_PATH,
        "pressure_jump_runner_dependency": HYDROSTATIC.VISCOUS.PRESSURE_RUNNER_PATH,
        "constant_state_runner_dependency": (
            HYDROSTATIC.PRESSURE.COMMON_RUNNER_PATH
        ),
    }
    return {
        name: COMMON.committed_path_identity(REPOSITORY_ROOT, path)
        for name, path in paths.items()
    }


def execution_record(
    matrix: dict[str, Any],
    matrix_path: Path,
    solver: Path,
    launcher: Path,
    output_directory: Path,
    allow_tracked_dirty_development: bool,
) -> dict[str, Any]:
    if output_directory.exists():
        raise RuntimeError(f"output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True, mode=0o700)
    initial_provenance = COMMON.source_provenance(REPOSITORY_ROOT)
    input_identity = _qualification_input_identity()
    resolved_matrix = matrix_path.resolve()
    input_identity["matrix"] = COMMON.committed_path_identity(
        REPOSITORY_ROOT, resolved_matrix
    )
    initially_eligible = qualification_eligibility(
        matrix, initial_provenance, input_identity
    )
    if not initially_eligible and not allow_tracked_dirty_development:
        raise RuntimeError(
            "qualification execution refused: matrix is not frozen or "
            "source identity is not eligible"
        )
    cache_path = COMMON.find_cmake_cache(solver)
    preflight = {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": COMMON.sha256_file(resolved_matrix),
        "runner_sha256": COMMON.sha256_file(SCRIPT_PATH),
        "runner_test_sha256": COMMON.sha256_file(RUNNER_TEST_PATH),
        "hydrostatic_runner_sha256": COMMON.sha256_file(
            HYDROSTATIC_RUNNER_PATH
        ),
        "viscous_jump_runner_sha256": COMMON.sha256_file(
            HYDROSTATIC.VISCOUS_RUNNER_PATH
        ),
        "pressure_jump_runner_sha256": COMMON.sha256_file(
            HYDROSTATIC.VISCOUS.PRESSURE_RUNNER_PATH
        ),
        "constant_state_runner_sha256": COMMON.sha256_file(
            HYDROSTATIC.PRESSURE.COMMON_RUNNER_PATH
        ),
        "qualification_input_identity": input_identity,
        "source": initial_provenance,
        "solver": {
            "path": str(solver),
            "sha256": COMMON.sha256_file(solver),
            "cmake_cache_path": str(cache_path) if cache_path else None,
            "cmake_cache_sha256": (
                COMMON.sha256_file(cache_path) if cache_path else None
            ),
            "selected_cmake_cache": COMMON.selected_cmake_cache(cache_path),
        },
        "launcher": {
            "path": str(launcher),
            "sha256": COMMON.sha256_file(launcher),
        },
        "qualification_eligible_at_start": initially_eligible,
        "matrix_frozen_before_execution": (
            matrix["status"] == "FROZEN_BEFORE_EXECUTION"
        ),
        "development_dirty_override": allow_tracked_dirty_development,
    }
    COMMON.write_json_create_only(output_directory / "preflight.json", preflight)
    results = [
        run_case(case, matrix, solver, launcher, output_directory)
        for case in matrix["cases"]
    ]
    reversal = evaluate_reversal_pairs(results, matrix)
    refinement = evaluate_refinement_families(results, matrix)
    final_provenance = COMMON.source_provenance(REPOSITORY_ROOT)
    provenance_stable = (
        final_provenance["revision"] == initial_provenance["revision"]
        and final_provenance["tracked_source_sha256"]
        == initial_provenance["tracked_source_sha256"]
        and final_provenance["tracked_clean"]
        == initial_provenance["tracked_clean"]
    )
    COMMON.write_json_create_only(
        output_directory / "final_provenance.json",
        {
            "initial": initial_provenance,
            "final": final_provenance,
            "stable": provenance_stable,
        },
    )
    qualification_eligible = initially_eligible and provenance_stable
    numerical_passed = (
        all(result["passed"] for result in results)
        and reversal["passed"]
        and refinement["passed"]
        and provenance_stable
    )
    disposition = execution_outcome(numerical_passed, qualification_eligible)
    gate_passed = disposition[
        "two_fluid_circular_static_drop_gate_passed"
    ]
    summary = {
        "matrix_id": matrix["matrix_id"],
        "requested_claim": matrix["accepted_claim"],
        **disposition,
        "qualification_eligible": qualification_eligible,
        "provenance_stable": provenance_stable,
        "case_count": len(results),
        "passed_case_count": sum(result["passed"] for result in results),
        "failed_case_ids": [
            result["case_id"] for result in results if not result["passed"]
        ],
        "reversal_pairs": reversal,
        "refinement": refinement,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
        "required_later_progression": matrix["required_later_progression"],
        "results": results,
    }
    COMMON.write_json_create_only(output_directory / "summary.json", summary)
    record = (
        "# WP-10 two-fluid circular static-drop qualification record\n\n"
        f"- Matrix: `{matrix['matrix_id']}`\n"
        f"- Source revision: `{initial_provenance['revision']}`\n"
        f"- Tracked source clean: `{str(initial_provenance['tracked_clean']).lower()}`\n"
        f"- Source stable through execution: `{str(provenance_stable).lower()}`\n"
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`\n"
        f"- Material-side reversal passed: `{str(reversal['passed']).lower()}`\n"
        f"- Refinement passed: `{str(refinement['passed']).lower()}`\n"
        f"- Circular static-drop gate passed: `{str(gate_passed).lower()}`\n"
        "- Spherical, sustained-flow, conservation, conditioning, FSR-08, WP-10, and Q7 gates remain open.\n"
    )
    COMMON.write_text_create_only(output_directory / "record.md", record)
    COMMON.write_checksums(output_directory)
    return summary


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--requested-claim",
        default="two_fluid_circular_static_drop_prerequisite",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-tracked-dirty-development", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    options = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    matrix = load_matrix(options.matrix.resolve())
    validate_requested_claim(matrix, options.requested_claim)
    if options.validate_only:
        if any(
            value is not None
            for value in (options.solver, options.launcher, options.output_dir)
        ) or options.allow_tracked_dirty_development:
            raise ValueError("--validate-only does not accept execution paths")
        print(
            json.dumps(
                {
                    "matrix_id": matrix["matrix_id"],
                    "case_count": len(matrix["cases"]),
                    "reversal_pair_count": len(
                        {case["reversal_pair"] for case in matrix["cases"]}
                    ),
                    "refinement_family_count": len(
                        {
                            case["refinement_family"]
                            for case in matrix["cases"]
                        }
                    ),
                    "requested_claim": options.requested_claim,
                    "outcome": "PASS",
                    **matrix["qualification_disposition"],
                },
                sort_keys=True,
            )
        )
        return 0
    if options.list_cases:
        if any(
            value is not None
            for value in (options.solver, options.launcher, options.output_dir)
        ) or options.allow_tracked_dirty_development:
            raise ValueError("--list-cases does not accept execution paths")
        for case in matrix["cases"]:
            print(
                f"{case['case_id']} ranks={case['mpi_ranks']} "
                f"resolution={case['resolution']} sign={case['level_set_sign']}"
            )
        return 0
    if options.solver is None or options.launcher is None or options.output_dir is None:
        raise ValueError("execution requires --solver, --launcher, and --output-dir")
    solver = options.solver.resolve()
    launcher = options.launcher.resolve()
    if not solver.is_file() or not os.access(solver, os.X_OK):
        raise ValueError(f"solver is not executable: {solver}")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError(f"launcher is not executable: {launcher}")
    summary = execution_record(
        matrix,
        options.matrix.resolve(),
        solver,
        launcher,
        options.output_dir.resolve(),
        options.allow_tracked_dirty_development,
    )
    print(json.dumps(summary, sort_keys=True))
    return int(summary["exit_code"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
