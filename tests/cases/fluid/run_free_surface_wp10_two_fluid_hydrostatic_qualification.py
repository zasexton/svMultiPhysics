#!/usr/bin/env python3
"""Run the frozen WP-10 planar two-fluid hydrostatic progression gate."""

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
    "free_surface_wp10_two_fluid_hydrostatic_matrix.json"
)
RUNNER_TEST_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "test_free_surface_wp10_two_fluid_hydrostatic_qualification.py"
)
VISCOUS_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp10_viscous_jump_qualification.py"
)
EXPECTED_MATRIX_SHA256 = (
    "7ba8bf05df63c1ff1ccb131bcd951631c03f0c0778dadb9d7c188f38714f5cdd"
)
EXPECTED_VISCOUS_RUNNER_SHA256 = (
    "2f53fd465ef37dd5bcff1f3dae3ec220f2ff649fc6ec886b576cfb4766a70e5a"
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
    "mesh",
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


def _load_viscous_runner() -> Any:
    if _sha256_file(VISCOUS_RUNNER_PATH) != EXPECTED_VISCOUS_RUNNER_SHA256:
        raise RuntimeError("viscous-jump runner dependency bytes changed")
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_viscous_jump_hydrostatic_dependency",
        VISCOUS_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the viscous-jump runner dependency")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


VISCOUS = _load_viscous_runner()
PRESSURE = VISCOUS.PRESSURE
COMMON = PRESSURE.COMMON


def _base_linear_level_set(case: dict[str, Any]) -> tuple[float, float, float]:
    orientation = case["orientation"]
    if orientation == "x":
        return 1.0, 0.0, float(case["offset"])
    if orientation == "y":
        return 0.0, 1.0, float(case["offset"])
    if orientation == "x_plus_y":
        return 1.0, 1.0, float(case["offset"])
    if orientation == "x_minus_y":
        return 1.0, -1.0, float(case["offset"])
    raise ValueError(f"unsupported planar orientation: {orientation!r}")


def base_unit_normal(case: dict[str, Any]) -> tuple[float, float]:
    a, b, _ = _base_linear_level_set(case)
    magnitude = math.hypot(a, b)
    return a / magnitude, b / magnitude


def interface_anchor(case: dict[str, Any]) -> tuple[float, float]:
    a, b, offset = _base_linear_level_set(case)
    scale = offset / (a * a + b * b)
    return scale * a, scale * b


def base_signed_distance(case: dict[str, Any], x: float, y: float) -> float:
    a, b, offset = _base_linear_level_set(case)
    return (a * x + b * y - offset) / math.hypot(a, b)


def body_force(case: dict[str, Any]) -> tuple[float, float, float]:
    normal = base_unit_normal(case)
    acceleration = float(case["gravity_acceleration"])
    return acceleration * normal[0], acceleration * normal[1], 0.0


def hydrostatic_pressure(
    case: dict[str, Any],
    phase: str,
    x: float,
    y: float,
    gauge_shift: float = 0.0,
) -> float:
    if phase not in {"negative", "positive"}:
        raise ValueError(f"unsupported phase: {phase!r}")
    density = float(case[f"{phase}_density"])
    acceleration = float(case["gravity_acceleration"])
    return (
        density * acceleration * base_signed_distance(case, x, y)
        + gauge_shift
    )


def hydrostatic_pressure_gradient(
    case: dict[str, Any], phase: str
) -> tuple[float, float, float]:
    if phase not in {"negative", "positive"}:
        raise ValueError(f"unsupported phase: {phase!r}")
    density = float(case[f"{phase}_density"])
    return tuple(density * value for value in body_force(case))


def pressure_gauge_vertex(
    matrix: dict[str, Any], case: dict[str, Any]
) -> dict[str, int | float]:
    nx = int(matrix["mesh"]["nx"])
    ny = int(matrix["mesh"]["ny"])
    coordinates = [
        (i / nx, j / ny)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]
    level_set = [
        COMMON.level_set_value(case, x, y) for x, y in coordinates
    ]
    point_index = min(
        range(len(coordinates)),
        key=lambda index: (level_set[index], index),
    )
    if not math.isfinite(level_set[point_index]) or level_set[point_index] >= 0.0:
        raise ValueError("hydrostatic mesh has no strictly negative gauge vertex")
    x, y = coordinates[point_index]
    return {
        "global_vertex_gid": point_index + 1,
        "point_index": point_index,
        "x": x,
        "y": y,
        "level_set": level_set[point_index],
    }


def pressure_gauge_anchor(
    matrix: dict[str, Any], case: dict[str, Any]
) -> tuple[float, float]:
    vertex = pressure_gauge_vertex(matrix, case)
    return float(vertex["x"]), float(vertex["y"])


def pressure_gauge_shift(matrix: dict[str, Any], case: dict[str, Any]) -> float:
    x, y = pressure_gauge_anchor(matrix, case)
    return -hydrostatic_pressure(case, "negative", x, y)


def render_pressure_gauge(
    matrix: dict[str, Any], case: dict[str, Any]
) -> str:
    vertex = pressure_gauge_vertex(matrix, case)
    return f"node_id,pressure\n{vertex['global_vertex_gid']},0\n"


def common_geometry(case: dict[str, Any]) -> dict[str, float]:
    return COMMON.analytic_planar_geometry(case)


def _base_distance_moments(
    case: dict[str, Any], phase: str
) -> tuple[float, float, float]:
    moments = VISCOUS.analytic_phase_moments(case)[phase]
    sign = float(case["level_set_sign"])
    return (
        moments["area"],
        sign * moments["signed_distance"],
        moments["signed_distance_squared"],
    )


def expected_phase_observables(
    case: dict[str, Any], phase: str, gauge_shift: float = 0.0
) -> dict[str, float]:
    if phase not in {"negative", "positive"}:
        raise ValueError(f"unsupported phase: {phase!r}")
    volume, distance_integral, distance_squared_integral = (
        _base_distance_moments(case, phase)
    )
    density = float(case[f"{phase}_density"])
    pressure_scale = density * float(case["gravity_acceleration"])
    pressure_integral = (
        pressure_scale * distance_integral + gauge_shift * volume
    )
    pressure_squared_integral = (
        pressure_scale * pressure_scale * distance_squared_integral
        + 2.0 * gauge_shift * pressure_scale * distance_integral
        + gauge_shift * gauge_shift * volume
    )
    gradient = hydrostatic_pressure_gradient(case, phase)
    result = {
        "volume": volume,
        "mass": density * volume,
        "pressure_integral": pressure_integral,
        "mean_pressure": pressure_integral / volume,
        "pressure_squared_integral": pressure_squared_integral,
        "hydrostatic_residual_sq": 0.0,
    }
    for component in range(3):
        integrated_gradient = gradient[component] * volume
        result[f"pressure_gradient_integral_{component}"] = integrated_gradient
        result[f"body_force_density_integral_{component}"] = integrated_gradient
        result[f"hydrostatic_residual_integral_{component}"] = 0.0
    return result


def _validate_matrix_structure(matrix: Any) -> dict[str, Any]:
    if not isinstance(matrix, dict) or set(matrix) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("hydrostatic matrix top-level contract changed")
    metadata = {
        "schema_version": 1,
        "matrix_id": "free_surface_wp10_two_fluid_hydrostatic_v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
        "accepted_claim": "two_fluid_hydrostatic_prerequisite",
    }
    for key, expected in metadata.items():
        if matrix.get(key) != expected:
            raise ValueError(f"hydrostatic matrix {key} changed")
    if matrix.get("rejected_claims") != [
        "static_drop_balance",
        "sustained_dynamics",
        "both_phase_mass_conservation",
        "high_ratio_conditioning",
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
    ]:
        raise ValueError("hydrostatic rejected-claim boundary changed")
    if matrix.get("qualification_disposition") != {
        "two_fluid_hydrostatic_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }:
        raise ValueError("hydrostatic disposition changed")

    model = matrix.get("model_envelope")
    if not isinstance(model, dict) or model.get("physical_model") != (
        "incompressible_two_fluid"
    ):
        raise ValueError("hydrostatic physical model changed")
    expected_model = {
        "momentum_operator": "two_fluid_stokes",
        "spatial_dimension": 2,
        "mesh": "affine_p1_triangle",
        "interface_geometry": "linear_corner",
        "domain": "closed_unit_square",
        "exterior_velocity": "homogeneous_strong_dirichlet_both_phases",
        "velocity": "identically_zero_in_both_phases",
        "pressure_gauge": "explicit_negative_phase_global_vertex_gid_zero",
        "pressure_state": (
            "continuous_piecewise_affine_density_times_gravity_potential"
        ),
        "surface_tension": 0.0,
        "body_force": "constant_acceleration_normal_to_interface",
        "phase_transport": (
            "locally_conservative_p1_indicator_with_stationary_geometry_reconciliation"
        ),
        "solver": "fsils_block_schur",
    }
    if {key: model.get(key) for key in expected_model} != expected_model:
        raise ValueError("hydrostatic model envelope changed")

    mesh = matrix.get("mesh")
    expected_mesh_keys = {
        "nx",
        "ny",
        "bounds",
        "triangle_split",
        "ghost_layers",
        "expected_vertices",
        "expected_triangles",
        "expected_wall_vertices",
        "expected_wall_lines",
        "expected_phase_initializer_dofs",
    }
    if not isinstance(mesh, dict) or set(mesh) != expected_mesh_keys:
        raise ValueError("hydrostatic mesh contract is absent")
    nx = mesh.get("nx")
    ny = mesh.get("ny")
    if (
        not isinstance(nx, int)
        or isinstance(nx, bool)
        or not isinstance(ny, int)
        or isinstance(ny, bool)
        or nx < 2
        or ny < 2
        or mesh.get("bounds") != [0.0, 1.0, 0.0, 1.0]
        or mesh.get("triangle_split") != "alternating_cell_diagonal"
        or mesh.get("ghost_layers") != 8
        or mesh.get("expected_vertices") != (nx + 1) * (ny + 1)
        or mesh.get("expected_triangles") != 2 * nx * ny
        or mesh.get("expected_wall_vertices") != 2 * (nx + ny)
        or mesh.get("expected_wall_lines") != 2 * (nx + ny)
        or mesh.get("expected_phase_initializer_dofs")
        != 3 * (nx + 1) * (ny + 1)
    ):
        raise ValueError("hydrostatic mesh contract is inconsistent")

    if matrix.get("time") != {
        "steps": 1,
        "dt": 0.01,
        "scheme": "BackwardEuler",
        "interpretation": "stationary_hydrostatic_acceptance_step",
    }:
        raise ValueError("hydrostatic time contract changed")
    nonlinear = matrix.get("nonlinear_solver")
    if nonlinear != {
        "control_source": "GeneralSimulationParameters",
        "absolute_tolerance": 5.0e-10,
        "relative_tolerance": 0.0,
        "maximum_iterations": 3,
    }:
        raise ValueError("hydrostatic nonlinear contract changed")

    threshold_keys = {
        "absolute_zero",
        "phase_volume_absolute",
        "phase_mass_relative",
        "interface_measure_absolute",
        "pressure_gradient_absolute",
        "pressure_gradient_relative",
        "body_force_integral_absolute",
        "body_force_integral_relative",
        "hydrostatic_residual_integral_absolute",
        "hydrostatic_residual_squared_absolute",
        "pressure_moment_absolute",
        "pressure_moment_relative",
        "pressure_squared_absolute",
        "pressure_squared_relative",
        "pressure_squared_gauge_normalization_ulp_factor",
        "common_gauge_absolute",
        "side_reversal_absolute",
        "side_reversal_relative",
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
    }
    thresholds = matrix.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != threshold_keys:
        raise ValueError("hydrostatic threshold contract changed")
    integer_thresholds = {
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
    }
    for key, value in thresholds.items():
        if key in integer_thresholds:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"threshold {key} must be a positive integer")
        else:
            COMMON.require_finite_positive(value, f"threshold {key}")
    if thresholds["maximum_nonlinear_iterations"] != nonlinear["maximum_iterations"]:
        raise ValueError("hydrostatic nonlinear iteration limits disagree")

    execution = matrix.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "wall_time_seconds_per_case",
        "memory_mib_per_case",
        "output_mib_per_case",
        "omp_threads",
        "solver_rank_trace",
    }:
        raise ValueError("hydrostatic execution envelope changed")
    if execution.get("solver_rank_trace") != "root_only":
        raise ValueError("hydrostatic solver-rank trace policy changed")
    for key in (
        "wall_time_seconds_per_case",
        "memory_mib_per_case",
        "output_mib_per_case",
        "omp_threads",
    ):
        if not isinstance(execution.get(key), int) or execution[key] < 1:
            raise ValueError(f"execution field {key} must be a positive integer")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("hydrostatic case coverage changed")
    required_case_keys = {
        "case_id",
        "reversal_pair",
        "orientation",
        "offset",
        "level_set_sign",
        "gravity_acceleration",
        "negative_density",
        "negative_viscosity",
        "positive_density",
        "positive_viscosity",
        "mpi_ranks",
    }
    case_ids: set[str] = set()
    pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != required_case_keys:
            raise ValueError("hydrostatic case contract is malformed")
        case_id = case.get("case_id")
        pair = case.get("reversal_pair")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
            or not isinstance(pair, str)
            or not pair
            or case.get("orientation")
            not in {"x", "y", "x_plus_y", "x_minus_y"}
            or case.get("level_set_sign") not in {-1, 1}
            or case.get("mpi_ranks") not in {1, 2}
        ):
            raise ValueError("hydrostatic case identity is invalid")
        case_ids.add(case_id)
        COMMON.require_finite(case.get("offset"), f"{case_id} offset")
        gravity = COMMON.require_finite(
            case.get("gravity_acceleration"), f"{case_id} gravity"
        )
        if abs(gravity) != 9.81:
            raise ValueError(f"{case_id} gravity magnitude changed")
        for field in (
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
        ):
            COMMON.require_finite_positive(case.get(field), f"{case_id} {field}")
        geometry = common_geometry(case)
        if min(geometry.values()) <= 0.0:
            raise ValueError(f"{case_id} does not cut both phases")
        for j in range(ny + 1):
            for i in range(nx + 1):
                if abs(COMMON.level_set_value(case, i / nx, j / ny)) <= 1.0e-14:
                    raise ValueError(f"{case_id} crosses a mesh vertex")
        expected_phase_observables(case, "negative")
        expected_phase_observables(case, "positive")
        pairs.setdefault(pair, []).append(case)
    if len(pairs) != 6:
        raise ValueError("hydrostatic reversal-pair coverage changed")
    for pair, members in pairs.items():
        if len(members) != 2 or {item["level_set_sign"] for item in members} != {
            -1,
            1,
        }:
            raise ValueError(f"hydrostatic reversal pair {pair} is incomplete")
        forward = next(item for item in members if item["level_set_sign"] == 1)
        reverse = next(item for item in members if item["level_set_sign"] == -1)
        for field in (
            "orientation",
            "offset",
            "gravity_acceleration",
            "mpi_ranks",
        ):
            if forward[field] != reverse[field]:
                raise ValueError(f"hydrostatic reversal pair {pair} changed {field}")
        for property_name in ("density", "viscosity"):
            if (
                forward[f"negative_{property_name}"]
                != reverse[f"positive_{property_name}"]
                or forward[f"positive_{property_name}"]
                != reverse[f"negative_{property_name}"]
            ):
                raise ValueError(
                    f"hydrostatic reversal pair {pair} did not swap material data"
                )
    return matrix


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    if path.resolve() != DEFAULT_MATRIX.resolve():
        raise ValueError("hydrostatic qualification requires its frozen matrix path")
    if _sha256_file(path) != EXPECTED_MATRIX_SHA256:
        raise ValueError("hydrostatic matrix bytes changed")
    return _validate_matrix_structure(COMMON.read_json_strict(path))


def validate_requested_claim(matrix: dict[str, Any], claim: str) -> str:
    if claim in matrix["rejected_claims"]:
        raise ValueError(f"requested claim {claim!r} is outside this progression gate")
    if claim != matrix["accepted_claim"]:
        raise ValueError(
            f"unsupported hydrostatic claim {claim!r}; expected "
            f"{matrix['accepted_claim']!r}"
        )
    return claim


def render_mesh(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    root = ET.fromstring(COMMON.render_mesh(matrix, case))
    arrays = {
        item.attrib.get("Name"): item
        for item in root.findall(".//PointData/DataArray")
    }
    nx = int(matrix["mesh"]["nx"])
    ny = int(matrix["mesh"]["ny"])
    gauge_shift = pressure_gauge_shift(matrix, case)
    for phase in ("negative", "positive"):
        values = [
            hydrostatic_pressure(
                case, phase, i / nx, j / ny, gauge_shift=gauge_shift
            )
            for j in range(ny + 1)
            for i in range(nx + 1)
        ]
        arrays[f"p_{phase}"].text = (
            "\n" + COMMON._format_values(values) + "\n        "
        )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


render_wall = PRESSURE.render_wall


def render_solver(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    root = ET.fromstring(COMMON.render_solver(matrix, case))
    parameters = root.find("GeneralSimulationParameters")
    mesh = root.find("Add_mesh")
    fluid = root.find("Add_equation[@type='fluid']")
    if parameters is None or mesh is None or fluid is None:
        raise RuntimeError("common solver template changed")
    parameters.find("Number_of_time_steps").text = str(matrix["time"]["steps"])
    parameters.find("Time_step_size").text = format(
        float(matrix["time"]["dt"]), ".17g"
    )
    ET.SubElement(parameters, "Newton_absolute_tolerance").text = format(
        float(matrix["nonlinear_solver"]["absolute_tolerance"]), ".17g"
    )
    ET.SubElement(parameters, "Newton_relative_tolerance").text = format(
        float(matrix["nonlinear_solver"]["relative_tolerance"]), ".17g"
    )
    ET.SubElement(parameters, "Newton_max_iterations").text = str(
        matrix["nonlinear_solver"]["maximum_iterations"]
    )
    fluid.set("type", "stokes")
    force = body_force(case)
    for name, value in zip(("Force_x", "Force_y", "Force_z"), force):
        element = fluid.find(name)
        if element is None:
            raise RuntimeError(f"common solver template lacks {name}")
        element.text = format(value, ".17g")
    face = ET.SubElement(mesh, "Add_face", {"name": "wall"})
    ET.SubElement(face, "Face_file_path").text = "wall.vtp"
    boundary = ET.Element("Add_BC", {"name": "wall"})
    ET.SubElement(boundary, "Type").text = "Dir"
    ET.SubElement(boundary, "Time_dependence").text = "Steady"
    ET.SubElement(boundary, "Value").text = "0"
    gauge = ET.Element("Node_pressure_constraints")
    ET.SubElement(gauge, "Id_type").text = "Global_vertex_gid"
    ET.SubElement(gauge, "Values_file_path").text = "pressure_gauge.csv"
    linear_solver = fluid.find("LS")
    boundary_at = (
        list(fluid).index(linear_solver)
        if linear_solver is not None
        else len(fluid)
    )
    fluid.insert(boundary_at, gauge)
    fluid.insert(boundary_at + 1, boundary)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def parse_case_output(output: str) -> dict[str, Any]:
    parsed = VISCOUS.parse_case_output(output)
    newton = parsed.get("newton")
    if isinstance(newton, dict) and "max_it" in newton:
        newton["max_it"] = newton["max_it"].rstrip(",)")
    return parsed


def _common_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(matrix)
    adapted["thresholds"]["phase_partition_absolute"] = matrix["thresholds"][
        "phase_volume_absolute"
    ]
    adapted["thresholds"]["phase_measure_absolute"] = matrix["thresholds"][
        "phase_volume_absolute"
    ]
    return adapted


def _numeric(record: dict[str, Any], key: str) -> float:
    if key not in record:
        raise ValueError(f"missing numeric field {key}")
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numeric field {key}")
    return value


def evaluate_hydrostatic_observables(
    case: dict[str, Any], matrix: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    thresholds = matrix["thresholds"]
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
        except (TypeError, ValueError) as error:
            check(key, False, str(error), "finite numeric field")
            return math.nan
        metrics[key] = value
        return value

    def near(
        key: str,
        expected: float,
        absolute: float,
        relative: float = 0.0,
        label: str | None = None,
    ) -> float:
        value = number(key)
        tolerance = absolute + relative * abs(expected)
        name = label or key
        check(
            name,
            math.isfinite(value) and abs(value - expected) <= tolerance,
            value,
            {"value": expected, "absolute_tolerance": tolerance},
        )
        return value

    geometry = common_geometry(case)
    near(
        "interface_measure",
        geometry["interface_measure"],
        thresholds["interface_measure_absolute"],
    )
    for key in (
        "velocity_jump_sq",
        "velocity_jump_normal_sq",
        "velocity_jump_tangential_sq",
        "negative_normal_flux",
        "positive_normal_flux",
        "normal_flux_jump",
        "negative_mass_flux",
        "positive_mass_flux",
        "mean_pressure_jump",
        "pressure_jump_sq",
        "pressure_jump_integral",
        "traction_jump_normal_integral",
        "traction_jump_sq",
        "viscous_traction_jump_sq",
        "negative_kinetic_energy",
        "positive_kinetic_energy",
    ):
        near(key, 0.0, thresholds["absolute_zero"])
    minimum_interface_points = thresholds["minimum_interface_quadrature_points"]
    interface_points = number("interface_quadrature_points")
    check(
        "interface_quadrature_points",
        math.isfinite(interface_points)
        and interface_points == int(interface_points)
        and interface_points >= minimum_interface_points,
        interface_points,
        f"integer >= {minimum_interface_points}",
    )

    gauges: dict[str, float] = {}
    for phase in ("negative", "positive"):
        expected_zero_gauge = expected_phase_observables(case, phase)
        prefix = f"{phase}_"
        minimum_phase_points = thresholds["minimum_phase_quadrature_points"]
        points = number(prefix + "phase_quadrature_points")
        check(
            prefix + "phase_quadrature_points",
            math.isfinite(points)
            and points == int(points)
            and points >= minimum_phase_points,
            points,
            f"integer >= {minimum_phase_points}",
        )
        near(
            prefix + "density",
            float(case[f"{phase}_density"]),
            0.0,
        )
        volume = near(
            prefix + "volume",
            expected_zero_gauge["volume"],
            thresholds["phase_volume_absolute"],
        )
        near(
            prefix + "mass",
            expected_zero_gauge["mass"],
            thresholds["absolute_zero"],
            thresholds["phase_mass_relative"],
        )
        for component in range(3):
            near(
                prefix + f"momentum_{component}",
                0.0,
                thresholds["absolute_zero"],
            )
            near(
                prefix + f"pressure_gradient_integral_{component}",
                expected_zero_gauge[f"pressure_gradient_integral_{component}"],
                thresholds["pressure_gradient_absolute"],
                thresholds["pressure_gradient_relative"],
            )
            near(
                prefix + f"body_force_density_integral_{component}",
                expected_zero_gauge[
                    f"body_force_density_integral_{component}"
                ],
                thresholds["body_force_integral_absolute"],
                thresholds["body_force_integral_relative"],
            )
            near(
                prefix + f"hydrostatic_residual_integral_{component}",
                0.0,
                thresholds["hydrostatic_residual_integral_absolute"],
            )
        near(
            prefix + "hydrostatic_residual_sq",
            0.0,
            thresholds["hydrostatic_residual_squared_absolute"],
        )
        pressure_integral = number(prefix + "pressure_integral")
        if math.isfinite(pressure_integral) and math.isfinite(volume) and volume > 0.0:
            gauges[phase] = (
                pressure_integral
                - expected_zero_gauge["pressure_integral"]
            ) / volume
            mean = number(prefix + "mean_pressure")
            tolerance = thresholds["pressure_moment_absolute"] + thresholds[
                "pressure_moment_relative"
            ] * abs(pressure_integral / volume)
            check(
                prefix + "mean_pressure_identity",
                math.isfinite(mean)
                and abs(mean - pressure_integral / volume) <= tolerance,
                mean,
                pressure_integral / volume,
            )
        else:
            gauges[phase] = math.nan

    negative_gauge = gauges["negative"]
    positive_gauge = gauges["positive"]
    gauge_tolerance = thresholds["common_gauge_absolute"]
    common_gauge = 0.5 * (negative_gauge + positive_gauge)
    metrics["negative_pressure_gauge"] = negative_gauge
    metrics["positive_pressure_gauge"] = positive_gauge
    metrics["common_pressure_gauge"] = common_gauge
    check(
        "common_pressure_gauge",
        math.isfinite(negative_gauge)
        and math.isfinite(positive_gauge)
        and abs(negative_gauge - positive_gauge) <= gauge_tolerance,
        {"negative": negative_gauge, "positive": positive_gauge},
        f"absolute difference <= {gauge_tolerance}",
    )
    for phase in ("negative", "positive"):
        expected = expected_phase_observables(
            case, phase, gauge_shift=common_gauge
        )
        near(
            f"{phase}_pressure_squared_integral",
            expected["pressure_squared_integral"],
            thresholds["pressure_squared_absolute"],
            thresholds["pressure_squared_relative"],
        )

    failed = [entry["name"] for entry in checks if not entry["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "metrics": metrics,
    }


def _physical_reversal_metrics(
    case: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, float]:
    sign = int(case["level_set_sign"])
    labels = (
        {"base_negative": "negative", "base_positive": "positive"}
        if sign == 1
        else {"base_negative": "positive", "base_positive": "negative"}
    )
    gauge = float(metrics["common_pressure_gauge"])
    result: dict[str, float] = {}
    for physical_side, phase in labels.items():
        volume = float(metrics[f"{phase}_volume"])
        pressure_integral = float(metrics[f"{phase}_pressure_integral"])
        values = {
            "volume": volume,
            "mass": float(metrics[f"{phase}_mass"]),
            "pressure_integral": pressure_integral - gauge * volume,
            "mean_pressure": float(metrics[f"{phase}_mean_pressure"]) - gauge,
            "pressure_squared_integral": (
                float(metrics[f"{phase}_pressure_squared_integral"])
                - 2.0 * gauge * pressure_integral
                + gauge * gauge * volume
            ),
            "hydrostatic_residual_sq": float(
                metrics[f"{phase}_hydrostatic_residual_sq"]
            ),
        }
        for component in range(3):
            for field in (
                "pressure_gradient_integral",
                "body_force_density_integral",
                "hydrostatic_residual_integral",
            ):
                values[f"{field}_{component}"] = float(
                    metrics[f"{phase}_{field}_{component}"]
                )
        for key, value in values.items():
            result[f"{physical_side}_{key}"] = value
    force = body_force(case)
    for component, value in enumerate(force):
        result[f"body_force_{component}"] = value
    result["interface_measure"] = float(metrics["interface_measure"])
    return result


def _physical_reversal_roundoff(
    case: dict[str, Any], matrix: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, float]:
    sign = int(case["level_set_sign"])
    labels = (
        {"base_negative": "negative", "base_positive": "positive"}
        if sign == 1
        else {"base_negative": "positive", "base_positive": "negative"}
    )
    gauge = float(metrics["common_pressure_gauge"])
    factor = matrix["thresholds"][
        "pressure_squared_gauge_normalization_ulp_factor"
    ]
    result: dict[str, float] = {}
    for physical_side, phase in labels.items():
        volume = float(metrics[f"{phase}_volume"])
        pressure_integral = float(metrics[f"{phase}_pressure_integral"])
        pressure_squared = float(
            metrics[f"{phase}_pressure_squared_integral"]
        )
        cross_term = 2.0 * gauge * pressure_integral
        gauge_term = gauge * gauge * volume
        result[f"{physical_side}_pressure_squared_integral"] = factor * (
            math.ulp(pressure_squared)
            + math.ulp(cross_term)
            + math.ulp(gauge_term)
        )
    return result


def evaluate_case(
    case: dict[str, Any],
    matrix: dict[str, Any],
    parsed: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    result = COMMON.evaluate_case(
        case, _common_matrix(matrix), parsed, return_code
    )
    interface = parsed.get("interface")
    hydrostatic = evaluate_hydrostatic_observables(case, matrix, interface)
    hydrostatic_names = {check["name"] for check in hydrostatic["checks"]}
    result["checks"] = [
        check
        for check in result["checks"]
        if check["name"] not in hydrostatic_names
    ]
    result["checks"].extend(hydrostatic["checks"])
    result["metrics"].update(hydrostatic["metrics"])

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        result["checks"].append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )

    newton = parsed.get("newton")
    for field, expected in (
        ("abs_tol", matrix["nonlinear_solver"]["absolute_tolerance"]),
        ("rel_tol", matrix["nonlinear_solver"]["relative_tolerance"]),
        ("max_it", matrix["nonlinear_solver"]["maximum_iterations"]),
    ):
        name = f"newton_{field}"
        try:
            value = _numeric(newton, field)
        except (TypeError, ValueError) as error:
            check(name, False, str(error), expected)
        else:
            result["metrics"][name] = value
            check(name, value == expected, value, expected)

    residuals: dict[str, float] = {}
    for field in (
        "nonlinear_initial_residual_norm",
        "nonlinear_final_residual_norm",
    ):
        try:
            value = _numeric(interface, field)
        except (TypeError, ValueError) as error:
            check(field, False, str(error), "finite residual within tolerance")
        else:
            residuals[field] = value
            result["metrics"][field] = value
            limit = matrix["nonlinear_solver"]["absolute_tolerance"]
            check(field, 0.0 <= value <= limit, value, {"minimum": 0.0, "maximum": limit})
    if len(residuals) == 2:
        initial = residuals["nonlinear_initial_residual_norm"]
        final = residuals["nonlinear_final_residual_norm"]
        check(
            "nonlinear_residual_nonincrease",
            final <= initial + matrix["thresholds"]["absolute_zero"],
            {"initial": initial, "final": final},
            "final no larger than initial within absolute-zero tolerance",
        )

    for field in (
        "prescribed_pressure_jump_applicable",
        "prescribed_viscous_traction_jump_applicable",
        "prescribed_stress_jump_residual_applicable",
    ):
        try:
            value = COMMON._boolean(interface, field)
        except (TypeError, ValueError) as error:
            check(field, False, str(error), False)
        else:
            result["metrics"][field] = value
            check(field, value is False, value, False)

    iteration_scope = (
        interface.get("phase_iteration_scope")
        if isinstance(interface, dict)
        else None
    )
    check(
        "phase_iteration_scope",
        iteration_scope == "shared_coupled_solve",
        iteration_scope,
        "shared_coupled_solve",
    )
    if iteration_scope is not None:
        result["metrics"]["phase_iteration_scope"] = iteration_scope

    expected_gauge = pressure_gauge_shift(matrix, case)
    actual_gauge = result["metrics"].get("common_pressure_gauge")
    gauge_tolerance = matrix["thresholds"]["common_gauge_absolute"]
    gauge_passed = (
        isinstance(actual_gauge, (int, float))
        and not isinstance(actual_gauge, bool)
        and math.isfinite(float(actual_gauge))
        and abs(float(actual_gauge) - expected_gauge) <= gauge_tolerance
    )
    check(
        "pressure_gauge_constraint",
        gauge_passed,
        actual_gauge,
        {"value": expected_gauge, "absolute_tolerance": gauge_tolerance},
    )
    result["metrics"]["expected_pressure_gauge"] = expected_gauge

    result["failed_checks"] = [
        item["name"] for item in result["checks"] if not item["passed"]
    ]
    result["passed"] = not result["failed_checks"]
    if result["passed"]:
        result["reversal_metrics"] = _physical_reversal_metrics(
            case, result["metrics"]
        )
        result["reversal_roundoff"] = _physical_reversal_roundoff(
            case, matrix, result["metrics"]
        )
    else:
        result["reversal_metrics"] = {}
        result["reversal_roundoff"] = {}
    return result


def physical_reversal_observables(case: dict[str, Any]) -> dict[str, float]:
    sign = int(case["level_set_sign"])
    labels = (
        {"base_negative": "negative", "base_positive": "positive"}
        if sign == 1
        else {"base_negative": "positive", "base_positive": "negative"}
    )
    result: dict[str, float] = {}
    for physical_side, phase in labels.items():
        expected = expected_phase_observables(case, phase)
        for key, value in expected.items():
            result[f"{physical_side}_{key}"] = value
    force = body_force(case)
    for component, value in enumerate(force):
        result[f"body_force_{component}"] = value
    result["interface_measure"] = common_geometry(case)["interface_measure"]
    return result


def evaluate_reversal_pairs(
    results: list[dict[str, Any]], matrix: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["reversal_pair"], []).append(result)
    checks: list[dict[str, Any]] = []
    absolute = matrix["thresholds"]["side_reversal_absolute"]
    relative = matrix["thresholds"]["side_reversal_relative"]
    expected_pairs = {case["reversal_pair"] for case in matrix["cases"]}
    for pair in sorted(expected_pairs):
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
        first = forward.get("reversal_metrics", forward.get("metrics", {}))
        second = reverse.get("reversal_metrics", reverse.get("metrics", {}))
        first_roundoff = forward.get("reversal_roundoff", {})
        second_roundoff = reverse.get("reversal_roundoff", {})
        keys = sorted(set(first) | set(second))
        for key in keys:
            first_value = first.get(key)
            second_value = second.get(key)
            tolerance = (
                absolute
                + relative
                * max(
                    abs(float(first_value)) if first_value is not None else 0.0,
                    abs(float(second_value)) if second_value is not None else 0.0,
                )
                + float(first_roundoff.get(key, 0.0))
                + float(second_roundoff.get(key, 0.0))
            )
            passed = (
                isinstance(first_value, (int, float))
                and not isinstance(first_value, bool)
                and isinstance(second_value, (int, float))
                and not isinstance(second_value, bool)
                and math.isfinite(float(first_value))
                and math.isfinite(float(second_value))
                and abs(float(first_value) - float(second_value)) <= tolerance
            )
            checks.append(
                {
                    "name": f"{pair}:{key}",
                    "passed": passed,
                    "first": first_value,
                    "second": second_value,
                    "tolerance": tolerance,
                }
            )
    failed = [entry["name"] for entry in checks if not entry["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def validate_effective_configuration(
    document: Any, case: dict[str, Any], matrix: dict[str, Any]
) -> dict[str, Any]:
    result = COMMON.validate_effective_configuration(document)
    momentum_modules = [
        module
        for module in document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "incompressible_two_fluid"
    ]
    if len(momentum_modules) != 1:
        raise ValueError("effective configuration lacks unique two-fluid momentum")
    momentum = momentum_modules[0]
    if momentum.get("momentum_operator") != "stokes":
        raise ValueError("effective configuration changed the momentum operator")
    material = momentum.get("material")
    if not isinstance(material, dict) or any(
        COMMON.require_finite(material.get(field), f"effective {field}")
        != float(case[field])
        for field in (
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
        )
    ):
        raise ValueError("effective configuration changed material data")
    force = momentum.get("body_force")
    expected_force = body_force(case)[:2]
    if (
        not isinstance(force, list)
        or len(force) != 2
        or any(
            COMMON.require_finite(value, "effective body force")
            != expected_force[index]
            for index, value in enumerate(force)
        )
    ):
        raise ValueError("effective configuration changed the body force")
    if momentum.get("hydrostatic_balance_diagnostic") != (
        "phasewise_integrated_pressure_gradient_minus_density_body_force"
    ):
        raise ValueError("effective configuration changed hydrostatic diagnostics")
    interface = momentum.get("interface")
    if not isinstance(interface, dict):
        raise ValueError("effective configuration lacks the interface contract")
    if interface.get("surface_tension") != 0.0:
        raise ValueError("effective configuration enabled surface tension")
    if interface.get("prescribed_pressure_jump_applicable") is not False:
        raise ValueError("effective configuration enabled a pressure-jump target")
    if interface.get("prescribed_viscous_traction_jump_applicable") is not False:
        raise ValueError("effective configuration enabled a viscous-traction target")
    pressure_space = momentum.get("pressure_space")
    gauge_vertex = pressure_gauge_vertex(matrix, case)
    if (
        not isinstance(pressure_space, dict)
        or pressure_space.get("representation") != "separate_phase_fields"
        or pressure_space.get("shared_gauge_count") != 1
        or pressure_space.get("shared_gauge_policy")
        != "explicit_global_vertex_gid"
        or pressure_space.get("shared_gauge_field") != "p_negative"
        or pressure_space.get("shared_gauge_id_type") != "Global_vertex_gid"
        or pressure_space.get("shared_gauge_vertex_gid")
        != gauge_vertex["global_vertex_gid"]
        or pressure_space.get("shared_gauge_value") != 0.0
    ):
        raise ValueError("effective configuration changed the shared pressure gauge")
    boundaries = momentum.get("boundary_conditions")
    entries = (
        boundaries.get("shared_velocity_dirichlet")
        if isinstance(boundaries, dict)
        else None
    )
    entry = (
        entries[0]
        if isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], dict)
        else None
    )
    values = entry.get("values") if isinstance(entry, dict) else None
    valid_boundary = (
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
    )
    if not valid_boundary:
        raise ValueError("effective configuration changed the closed-wall boundary")
    result.update(
        {
            "momentum_operator": "stokes",
            "body_force": list(expected_force),
            "hydrostatic_balance_diagnostic": (
                "phasewise_integrated_pressure_gradient_minus_density_body_force"
            ),
            "shared_pressure_gauge_count": 1,
            "shared_pressure_gauge_vertex_gid": gauge_vertex[
                "global_vertex_gid"
            ],
            "shared_velocity_dirichlet_count": 1,
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
    COMMON.write_text_create_only(wall_path, render_wall(matrix))
    COMMON.write_text_create_only(
        gauge_path, render_pressure_gauge(matrix, case)
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
            "level_set_sign": case["level_set_sign"],
            "mpi_ranks": case["mpi_ranks"],
            "passed": False,
            "failed_checks": ["output_parse"],
            "metrics": {},
            "reversal_metrics": {},
            "reversal_roundoff": {},
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
            COMMON.read_json_strict(effective_path), case, matrix
        )
    except (OSError, TypeError, ValueError) as error:
        effective = {"error": str(error)}
        result["checks"].append(
            {
                "name": "effective_configuration",
                "passed": False,
                "actual": str(error),
                "expected": "closed-wall hydrostatic two-fluid configuration",
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


def execution_outcome(
    numerical_passed: bool, qualification_eligible: bool
) -> dict[str, Any]:
    if not numerical_passed:
        return {
            "outcome": "FAIL",
            "two_fluid_hydrostatic_gate_passed": False,
            "exit_code": 1,
        }
    if qualification_eligible:
        return {
            "outcome": "PASS",
            "two_fluid_hydrostatic_gate_passed": True,
            "exit_code": 0,
        }
    return {
        "outcome": "DEVELOPMENT_PASS",
        "two_fluid_hydrostatic_gate_passed": False,
        "exit_code": 0,
    }


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
    provenance = COMMON.source_provenance(REPOSITORY_ROOT)
    input_identity = {
        "matrix": COMMON.committed_path_identity(REPOSITORY_ROOT, matrix_path),
        "runner": COMMON.committed_path_identity(REPOSITORY_ROOT, SCRIPT_PATH),
        "runner_test": COMMON.committed_path_identity(
            REPOSITORY_ROOT, RUNNER_TEST_PATH
        ),
        "viscous_jump_runner_dependency": COMMON.committed_path_identity(
            REPOSITORY_ROOT, VISCOUS_RUNNER_PATH
        ),
        "pressure_jump_runner_dependency": COMMON.committed_path_identity(
            REPOSITORY_ROOT, VISCOUS.PRESSURE_RUNNER_PATH
        ),
        "constant_state_runner_dependency": COMMON.committed_path_identity(
            REPOSITORY_ROOT, PRESSURE.COMMON_RUNNER_PATH
        ),
    }
    matrix_frozen = matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    qualification_eligible = qualification_eligibility(
        matrix, provenance, input_identity
    )
    if not qualification_eligible and not allow_tracked_dirty_development:
        raise RuntimeError(
            "qualification execution refused: matrix is not frozen or "
            "source identity is not eligible"
        )
    cache_path = COMMON.find_cmake_cache(solver)
    preflight = {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": COMMON.sha256_file(matrix_path),
        "runner_sha256": COMMON.sha256_file(SCRIPT_PATH),
        "runner_test_sha256": COMMON.sha256_file(RUNNER_TEST_PATH),
        "viscous_jump_runner_sha256": COMMON.sha256_file(VISCOUS_RUNNER_PATH),
        "pressure_jump_runner_sha256": COMMON.sha256_file(
            VISCOUS.PRESSURE_RUNNER_PATH
        ),
        "constant_state_runner_sha256": COMMON.sha256_file(
            PRESSURE.COMMON_RUNNER_PATH
        ),
        "qualification_input_identity": input_identity,
        "source": provenance,
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
        "qualification_eligible": qualification_eligible,
        "matrix_frozen_before_execution": matrix_frozen,
        "development_dirty_override": allow_tracked_dirty_development,
    }
    COMMON.write_json_create_only(output_directory / "preflight.json", preflight)
    results = [
        run_case(case, matrix, solver, launcher, output_directory)
        for case in matrix["cases"]
    ]
    reversal = evaluate_reversal_pairs(results, matrix)
    disposition = execution_outcome(
        all(result["passed"] for result in results) and reversal["passed"],
        qualification_eligible,
    )
    gate_passed = disposition["two_fluid_hydrostatic_gate_passed"]
    summary = {
        "matrix_id": matrix["matrix_id"],
        "requested_claim": matrix["accepted_claim"],
        **disposition,
        "qualification_eligible": qualification_eligible,
        "case_count": len(results),
        "passed_case_count": sum(result["passed"] for result in results),
        "failed_case_ids": [
            result["case_id"] for result in results if not result["passed"]
        ],
        "reversal_pairs": reversal,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
        "required_later_progression": matrix["required_later_progression"],
        "results": results,
    }
    COMMON.write_json_create_only(output_directory / "summary.json", summary)
    record = (
        "# WP-10 two-fluid hydrostatic qualification record\n\n"
        f"- Matrix: `{matrix['matrix_id']}`\n"
        f"- Source revision: `{provenance['revision']}`\n"
        f"- Tracked source clean: `{str(provenance['tracked_clean']).lower()}`\n"
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`\n"
        f"- Reversal pairs passed: `{str(reversal['passed']).lower()}`\n"
        f"- Two-fluid hydrostatic gate passed: `{str(gate_passed).lower()}`\n"
        "- Static-drop, sustained-flow, conservation, FSR-08, WP-10, and Q7 gates remain open.\n"
    )
    COMMON.write_text_create_only(output_directory / "record.md", record)
    COMMON.write_checksums(output_directory)
    return summary


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--requested-claim", default="two_fluid_hydrostatic_prerequisite"
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
    matrix_path = options.matrix.resolve()
    matrix = load_matrix(matrix_path)
    validate_requested_claim(matrix, options.requested_claim)
    if options.validate_only:
        if any(
            value is not None
            for value in (options.solver, options.launcher, options.output_dir)
        ):
            raise ValueError("--validate-only does not accept execution paths")
        print(
            json.dumps(
                {
                    "matrix_id": matrix["matrix_id"],
                    "case_count": len(matrix["cases"]),
                    "reversal_pair_count": len(
                        {case["reversal_pair"] for case in matrix["cases"]}
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
        ):
            raise ValueError("--list-cases does not accept execution paths")
        for case in matrix["cases"]:
            print(
                f"{case['case_id']} ranks={case['mpi_ranks']} "
                f"orientation={case['orientation']} sign={case['level_set_sign']} "
                f"gravity={case['gravity_acceleration']}"
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
        matrix_path,
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
