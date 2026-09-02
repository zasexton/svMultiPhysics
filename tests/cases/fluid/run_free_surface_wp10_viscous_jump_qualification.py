#!/usr/bin/env python3
"""Run the frozen WP-10 planar viscous-traction-jump gate."""

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
    "free_surface_wp10_viscous_jump_matrix.json"
)
PRESSURE_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp10_pressure_jump_qualification.py"
)
EXPECTED_MATRIX_SHA256 = (
    "c34010874cd5df659c26b41b4a63c43e6fba10d4d4b982e63dd9c1b74a1cbf28"
)
EXPECTED_PRESSURE_RUNNER_SHA256 = (
    "ac8852afbe26cf6a59402450cbdc7f5e7a15ef08197ee504aac09518ed6fd2e8"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "matrix_id",
    "supersedes_matrix_id",
    "revision_basis",
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


def _load_pressure_runner() -> Any:
    if _sha256_file(PRESSURE_RUNNER_PATH) != EXPECTED_PRESSURE_RUNNER_SHA256:
        raise RuntimeError("pressure-jump runner dependency bytes changed")
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_viscous_jump_pressure_dependency",
        PRESSURE_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the pressure-jump runner dependency")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


PRESSURE = _load_pressure_runner()
COMMON = PRESSURE.COMMON


def _validate_matrix_structure(matrix: Any) -> dict[str, Any]:
    if not isinstance(matrix, dict) or set(matrix) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("viscous-jump matrix top-level contract changed")
    expected_metadata = {
        "schema_version": 2,
        "matrix_id": "free_surface_wp10_viscous_jump_v2",
        "status": "FROZEN_BEFORE_EXECUTION",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
        "accepted_claim": "planar_viscous_traction_jump_prerequisite",
    }
    for key, expected in expected_metadata.items():
        if matrix.get(key) != expected:
            raise ValueError(f"viscous-jump matrix {key} changed")
    if matrix.get("supersedes_matrix_id") != (
        "free_surface_wp10_viscous_jump_v1"
    ):
        raise ValueError("viscous-jump matrix predecessor changed")
    if matrix.get("revision_basis") != (
        "Declares the assembled-roundoff nonlinear entry bound and root-only "
        "accepted-state diagnostic policy exposed by the version-1 "
        "development run."
    ):
        raise ValueError("viscous-jump matrix revision basis changed")
    if matrix.get("rejected_claims") != [
        "sustained_affine_flow",
        "both_phase_mass_conservation",
        "high_ratio_conditioning",
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
    ]:
        raise ValueError("viscous-jump rejected-claim boundary changed")
    if matrix.get("qualification_disposition") != {
        "planar_viscous_traction_jump_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }:
        raise ValueError("viscous-jump disposition changed")

    expected_model = {
        "physical_model": "incompressible_two_fluid",
        "momentum_operator": "two_fluid_stokes",
        "spatial_dimension": 2,
        "mesh": "affine_p1_triangle",
        "interface_geometry": "linear_corner",
        "domain": "bounded_unit_square_complete_exterior_dirichlet",
        "exterior_velocity": "shared_general_dirichlet_both_phases",
        "exterior_flux_scope": (
            "nonzero_local_discrete_q_flux_with_zero_global_balance"
        ),
        "velocity_state": (
            "shear_rate_times_signed_distance_times_interface_tangent"
        ),
        "pressure_gauge": "first_free_negative_pressure_dof_zero",
        "pressure_state": "identically_zero_both_phases",
        "viscous_target": (
            "negative_minus_positive_viscosity_times_shear_rate_times_"
            "interface_tangent"
        ),
        "surface_tension": 0.0,
        "body_force": [0.0, 0.0],
        "phase_transport": (
            "locally_conservative_p1_indicator_with_geometry_reconciliation_"
            "and_globally_balanced_discrete_q_flux"
        ),
        "solver": "fsils_block_schur",
    }
    if matrix.get("model_envelope") != expected_model:
        raise ValueError("viscous-jump model envelope changed")

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
        raise ValueError("viscous-jump mesh contract is absent")
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
        raise ValueError("viscous-jump mesh contract is inconsistent")

    if matrix.get("time") != {
        "steps": 1,
        "dt": 1.0e-4,
        "scheme": "BackwardEuler",
        "interpretation": "finite_low_courant_affine_tangential_step",
    }:
        raise ValueError("viscous-jump time contract changed")
    nonlinear_solver = matrix.get("nonlinear_solver")
    if not isinstance(nonlinear_solver, dict) or set(nonlinear_solver) != {
        "control_source",
        "absolute_tolerance",
        "relative_tolerance",
        "entry_state_requirement",
    }:
        raise ValueError("viscous-jump nonlinear solver contract changed")
    if (
        nonlinear_solver.get("control_source") != "GeneralSimulationParameters"
        or nonlinear_solver.get("entry_state_requirement")
        != "zero_update_zero_iteration"
        or COMMON.require_finite_positive(
            nonlinear_solver.get("absolute_tolerance"),
            "nonlinear absolute tolerance",
        )
        != 5.0e-10
        or COMMON.require_finite(
            nonlinear_solver.get("relative_tolerance"),
            "nonlinear relative tolerance",
        )
        != 0.0
    ):
        raise ValueError("viscous-jump nonlinear solver contract is inconsistent")
    thresholds = matrix.get("thresholds")
    expected_threshold_keys = {
        "absolute_zero",
        "phase_volume_absolute",
        "phase_mass_relative",
        "interface_measure_absolute",
        "bulk_momentum_absolute",
        "bulk_momentum_relative",
        "bulk_kinetic_energy_absolute",
        "bulk_kinetic_energy_relative",
        "traction_moment_absolute",
        "traction_squared_absolute",
        "prescribed_viscous_error_squared",
        "prescribed_stress_error_squared",
        "side_reversal_absolute",
        "maximum_finite_step_courant",
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_threshold_keys:
        raise ValueError("viscous-jump threshold contract changed")
    integer_thresholds = {
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
    }
    for key in expected_threshold_keys - integer_thresholds:
        COMMON.require_finite_positive(thresholds.get(key), f"threshold {key}")
    for key in (
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
    ):
        if not isinstance(thresholds.get(key), int) or thresholds[key] < 1:
            raise ValueError(f"threshold {key} must be a positive integer")
    for key in ("maximum_nonlinear_iterations", "maximum_linear_iterations"):
        if thresholds.get(key) != 0:
            raise ValueError(f"threshold {key} must retain exact-entry zero")

    execution = matrix.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "wall_time_seconds_per_case",
        "memory_mib_per_case",
        "output_mib_per_case",
        "omp_threads",
        "solver_rank_trace",
    }:
        raise ValueError("viscous-jump execution envelope changed")
    for key in (
        "wall_time_seconds_per_case",
        "memory_mib_per_case",
        "output_mib_per_case",
        "omp_threads",
    ):
        value = execution[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"execution field {key} must be a positive integer")
    if execution.get("solver_rank_trace") != "root_only":
        raise ValueError("viscous-jump solver rank trace must remain root-only")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("viscous-jump case coverage changed")
    case_ids: set[str] = set()
    reversal_pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "reversal_pair",
            "orientation",
            "offset",
            "level_set_sign",
            "shear_rate",
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
            "mpi_ranks",
        }:
            raise ValueError("viscous-jump case contract is malformed")
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
            raise ValueError("viscous-jump case identity is invalid")
        case_ids.add(case_id)
        COMMON.require_finite(case.get("offset"), f"{case_id} offset")
        shear_rate = COMMON.require_finite(
            case.get("shear_rate"), f"{case_id} shear rate"
        )
        if shear_rate == 0.0:
            raise ValueError(f"{case_id} shear rate must be nonzero")
        for field in (
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
        ):
            COMMON.require_finite_positive(case.get(field), f"{case_id} {field}")
        if case["negative_viscosity"] == case["positive_viscosity"]:
            raise ValueError(f"{case_id} viscous traction jump is zero")
        geometry = COMMON.analytic_planar_geometry(case)
        if min(geometry.values()) <= 0.0:
            raise ValueError(f"{case_id} does not cut both phases")
        for j in range(ny + 1):
            for i in range(nx + 1):
                if abs(COMMON.level_set_value(case, i / nx, j / ny)) <= 1.0e-14:
                    raise ValueError(f"{case_id} crosses a mesh vertex")
        reversal_pairs.setdefault(pair, []).append(case)

    if len(reversal_pairs) != 6:
        raise ValueError("viscous-jump reversal-pair coverage changed")
    for pair, members in reversal_pairs.items():
        if len(members) != 2 or {item["level_set_sign"] for item in members} != {
            -1,
            1,
        }:
            raise ValueError(f"reversal pair {pair} is incomplete")
        forward = next(item for item in members if item["level_set_sign"] == 1)
        reverse = next(item for item in members if item["level_set_sign"] == -1)
        for field in ("orientation", "offset", "shear_rate", "mpi_ranks"):
            if forward[field] != reverse[field]:
                raise ValueError(f"reversal pair {pair} changes {field}")
        for first, second in (
            ("negative_density", "positive_density"),
            ("negative_viscosity", "positive_viscosity"),
        ):
            if forward[first] != reverse[second] or forward[second] != reverse[first]:
                raise ValueError(f"reversal pair {pair} does not exchange material data")
    if {case["mpi_ranks"] for case in cases} != {1, 2}:
        raise ValueError("viscous-jump serial and parallel coverage changed")
    if {case["orientation"] for case in cases} != {
        "x",
        "y",
        "x_plus_y",
        "x_minus_y",
    }:
        raise ValueError("viscous-jump orientation coverage changed")
    if {math.copysign(1.0, case["shear_rate"]) for case in cases} != {
        -1.0,
        1.0,
    }:
        raise ValueError("viscous-jump shear-sign coverage changed")
    maximum_density_ratio = max(
        max(case["negative_density"], case["positive_density"])
        / min(case["negative_density"], case["positive_density"])
        for case in cases
    )
    if maximum_density_ratio != 10000.0:
        raise ValueError("viscous-jump density-ratio coverage changed")
    progression = matrix.get("required_later_progression")
    if not isinstance(progression, list) or len(progression) < 8:
        raise ValueError("viscous-jump later progression is incomplete")
    return matrix


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    if COMMON.sha256_file(path) != EXPECTED_MATRIX_SHA256:
        raise ValueError("frozen matrix bytes changed")
    return _validate_matrix_structure(COMMON.read_json_strict(path))


def validate_requested_claim(matrix: dict[str, Any], claim: str) -> str:
    if claim in matrix["rejected_claims"]:
        raise ValueError(
            f"requested claim {claim!r} is outside this progression gate"
        )
    if claim != matrix["accepted_claim"]:
        raise ValueError(
            f"unsupported viscous-jump claim {claim!r}; expected "
            f"{matrix['accepted_claim']!r}"
        )
    return claim


def _linear_level_set(case: dict[str, Any]) -> tuple[float, float, float]:
    orientation = case["orientation"]
    if orientation == "x":
        raw = (1.0, 0.0, float(case["offset"]))
    elif orientation == "y":
        raw = (0.0, 1.0, float(case["offset"]))
    elif orientation == "x_plus_y":
        raw = (1.0, 1.0, float(case["offset"]))
    elif orientation == "x_minus_y":
        raw = (1.0, -1.0, float(case["offset"]))
    else:
        raise ValueError(f"unsupported planar orientation: {orientation!r}")
    sign = float(case["level_set_sign"])
    return sign * raw[0], sign * raw[1], sign * raw[2]


def unit_normal_and_tangent(
    case: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    a, b, _ = _linear_level_set(case)
    gradient_norm = math.hypot(a, b)
    normal = (a / gradient_norm, b / gradient_norm)
    tangent = (-normal[1], normal[0])
    return normal, tangent, gradient_norm


def _clip_unit_square(
    a: float, b: float, c: float, keep_nonpositive: bool
) -> list[tuple[float, float]]:
    polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    def value(point: tuple[float, float]) -> float:
        raw = a * point[0] + b * point[1] - c
        return raw if keep_nonpositive else -raw

    clipped: list[tuple[float, float]] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        start_value = value(start)
        end_value = value(end)
        start_inside = start_value <= 0.0
        end_inside = end_value <= 0.0
        if start_inside:
            clipped.append(start)
        if start_inside != end_inside:
            denominator = start_value - end_value
            if denominator == 0.0:
                raise ValueError("degenerate planar clipping edge")
            fraction = start_value / denominator
            clipped.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
    return clipped


def _polygon_moments(polygon: list[tuple[float, float]]) -> dict[str, float]:
    if len(polygon) < 3:
        return {
            "area": 0.0,
            "x": 0.0,
            "y": 0.0,
            "xx": 0.0,
            "xy": 0.0,
            "yy": 0.0,
        }
    area = x_moment = y_moment = 0.0
    xx_moment = xy_moment = yy_moment = 0.0
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross / 2.0
        x_moment += (x0 + x1) * cross / 6.0
        y_moment += (y0 + y1) * cross / 6.0
        xx_moment += (x0 * x0 + x0 * x1 + x1 * x1) * cross / 12.0
        yy_moment += (y0 * y0 + y0 * y1 + y1 * y1) * cross / 12.0
        xy_moment += (
            2.0 * x0 * y0
            + x0 * y1
            + x1 * y0
            + 2.0 * x1 * y1
        ) * cross / 24.0
    if area < 0.0:
        area = -area
        x_moment = -x_moment
        y_moment = -y_moment
        xx_moment = -xx_moment
        xy_moment = -xy_moment
        yy_moment = -yy_moment
    return {
        "area": area,
        "x": x_moment,
        "y": y_moment,
        "xx": xx_moment,
        "xy": xy_moment,
        "yy": yy_moment,
    }


def _signed_distance_moments(
    polygon: list[tuple[float, float]], a: float, b: float, c: float
) -> dict[str, float]:
    moments = _polygon_moments(polygon)
    gradient_norm = math.hypot(a, b)
    integral = (
        a * moments["x"] + b * moments["y"] - c * moments["area"]
    ) / gradient_norm
    squared = (
        a * a * moments["xx"]
        + 2.0 * a * b * moments["xy"]
        + b * b * moments["yy"]
        - 2.0 * c * (a * moments["x"] + b * moments["y"])
        + c * c * moments["area"]
    ) / (gradient_norm * gradient_norm)
    return {
        "area": moments["area"],
        "signed_distance": integral,
        "signed_distance_squared": squared,
    }


def analytic_phase_moments(case: dict[str, Any]) -> dict[str, dict[str, float]]:
    a, b, c = _linear_level_set(case)
    result = {
        "negative": _signed_distance_moments(
            _clip_unit_square(a, b, c, True), a, b, c
        ),
        "positive": _signed_distance_moments(
            _clip_unit_square(a, b, c, False), a, b, c
        ),
    }
    geometry = COMMON.analytic_planar_geometry(case)
    for phase in ("negative", "positive"):
        if not math.isclose(
            result[phase]["area"],
            geometry[f"{phase}_volume"],
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ):
            raise ValueError("clipped moment geometry disagrees with planar geometry")
    return result


def velocity(case: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    _, tangent, gradient_norm = unit_normal_and_tangent(case)
    signed_distance = COMMON.level_set_value(case, x, y) / gradient_norm
    scale = float(case["shear_rate"]) * signed_distance
    return scale * tangent[0], scale * tangent[1]


def traction_jump_target(case: dict[str, Any]) -> tuple[float, float, float]:
    _, tangent, _ = unit_normal_and_tangent(case)
    scale = (
        float(case["negative_viscosity"])
        - float(case["positive_viscosity"])
    ) * float(case["shear_rate"])
    return scale * tangent[0], scale * tangent[1], 0.0


def expected_case_observables(
    case: dict[str, Any], matrix: dict[str, Any]
) -> dict[str, float]:
    phase_moments = analytic_phase_moments(case)
    _, tangent, _ = unit_normal_and_tangent(case)
    shear_rate = float(case["shear_rate"])
    measure = COMMON.analytic_planar_geometry(case)["interface_measure"]
    expected: dict[str, float] = {}
    for phase in ("negative", "positive"):
        density = float(case[f"{phase}_density"])
        viscosity = float(case[f"{phase}_viscosity"])
        momentum_scale = (
            density * shear_rate * phase_moments[phase]["signed_distance"]
        )
        for component in range(3):
            direction = tangent[component] if component < 2 else 0.0
            expected[f"{phase}_momentum_{component}"] = momentum_scale * direction
        expected[f"{phase}_kinetic_energy"] = (
            0.5
            * density
            * shear_rate
            * shear_rate
            * phase_moments[phase]["signed_distance_squared"]
        )
        traction_scale = viscosity * shear_rate * measure
        for component in range(3):
            direction = tangent[component] if component < 2 else 0.0
            value = traction_scale * direction
            expected[f"{phase}_traction_integral_{component}"] = value
            expected[f"{phase}_viscous_traction_integral_{component}"] = value

    target = traction_jump_target(case)
    target_squared = sum(value * value for value in target)
    for component, value in enumerate(target):
        expected[f"traction_jump_integral_{component}"] = value * measure
        expected[f"viscous_traction_jump_integral_{component}"] = value * measure
        expected[f"prescribed_viscous_traction_jump_target_{component}"] = value
    expected["traction_jump_sq"] = target_squared * measure
    expected["viscous_traction_jump_sq"] = target_squared * measure
    expected["prescribed_viscous_traction_jump_error_sq"] = 0.0
    expected["prescribed_stress_jump_residual_sq"] = 0.0
    return expected


def render_mesh(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    root = ET.fromstring(COMMON.render_mesh(matrix, case))
    arrays = {
        item.attrib.get("Name"): item
        for item in root.findall(".//PointData/DataArray")
    }
    nx = int(matrix["mesh"]["nx"])
    ny = int(matrix["mesh"]["ny"])
    values: list[float] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            values.extend(velocity(case, i / nx, j / ny))
    formatted = "\n" + COMMON._format_values(values) + "\n        "
    arrays["u_negative"].text = formatted
    arrays["u_positive"].text = formatted
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


render_wall = PRESSURE.render_wall


def render_velocity_data(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    vertices = PRESSURE._wall_vertices(
        int(matrix["mesh"]["nx"]), int(matrix["mesh"]["ny"])
    )
    lines = [f"2 1 {len(vertices)}", "0"]
    for node_id, x, y in vertices:
        ux, uy = velocity(case, x, y)
        lines.append(str(node_id))
        lines.append(f"{ux:.17g} {uy:.17g}")
    return "\n".join(lines) + "\n"


def render_solver(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    root = ET.fromstring(COMMON.render_solver(matrix, case))
    parameters = root.find("GeneralSimulationParameters")
    if parameters is None:
        raise RuntimeError("common solver template lacks time parameters")
    step_count = parameters.find("Number_of_time_steps")
    time_step = parameters.find("Time_step_size")
    if step_count is None or time_step is None:
        raise RuntimeError("common solver template lacks its time contract")
    step_count.text = str(matrix["time"]["steps"])
    time_step.text = format(float(matrix["time"]["dt"]), ".17g")
    ET.SubElement(parameters, "Newton_absolute_tolerance").text = format(
        float(matrix["nonlinear_solver"]["absolute_tolerance"]), ".17g"
    )
    ET.SubElement(parameters, "Newton_relative_tolerance").text = format(
        float(matrix["nonlinear_solver"]["relative_tolerance"]), ".17g"
    )
    mesh = root.find("Add_mesh")
    level_set = root.find("Add_equation[@type='level_set']")
    fluid = root.find("Add_equation[@type='fluid']")
    if mesh is None or level_set is None or fluid is None:
        raise RuntimeError("common solver template changed")
    boundary_flux_policy = ET.SubElement(
        level_set, "Conservative_phase_boundary_flux_policy"
    )
    boundary_flux_policy.text = "globally_balanced_discrete_q_flux"
    fluid.set("type", "stokes")
    face = ET.SubElement(mesh, "Add_face", {"name": "wall"})
    ET.SubElement(face, "Face_file_path").text = "wall.vtp"
    target = traction_jump_target(case)
    nitsche = fluid.find("Two_fluid_interface_nitsche_gamma")
    insert_at = list(fluid).index(nitsche) if nitsche is not None else 0
    for name, value in (
        ("Prescribed_viscous_traction_jump_x", target[0]),
        ("Prescribed_viscous_traction_jump_y", target[1]),
    ):
        element = ET.Element(name)
        element.text = format(value, ".17g")
        fluid.insert(insert_at, element)
        insert_at += 1
    boundary = ET.Element("Add_BC", {"name": "wall"})
    ET.SubElement(boundary, "Type").text = "Dir"
    ET.SubElement(boundary, "Time_dependence").text = "General"
    ET.SubElement(boundary, "Temporal_and_spatial_values_file_path").text = (
        "velocity.dat"
    )
    linear_solver = fluid.find("LS")
    boundary_at = (
        list(fluid).index(linear_solver) if linear_solver is not None else len(fluid)
    )
    fluid.insert(boundary_at, boundary)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def parse_case_output(output: str) -> dict[str, Any]:
    parsed = COMMON.parse_case_output(output)
    newton = COMMON._unique_record(
        output.splitlines(), "Transient solve:"
    )
    for field in ("abs_tol", "rel_tol"):
        if field in newton:
            newton[field] = newton[field].rstrip(",)")
    parsed["newton"] = newton
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


def _relative_tolerance(absolute: float, relative: float, expected: float) -> float:
    return absolute + relative * abs(expected)


def evaluate_case(
    case: dict[str, Any],
    matrix: dict[str, Any],
    parsed: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    sanitized = copy.deepcopy(parsed)
    interface = sanitized.get("interface")
    replaced_fields = {
        "traction_jump_sq",
        *{
            f"{phase}_momentum_{component}"
            for phase in ("negative", "positive")
            for component in range(3)
        },
        "negative_kinetic_energy",
        "positive_kinetic_energy",
    }
    if isinstance(interface, dict):
        for field in replaced_fields:
            if field in interface:
                interface[field] = "0"
    phase_stage = sanitized.get("phase_stage")
    if isinstance(phase_stage, dict) and "courant" in phase_stage:
        phase_stage["courant"] = "0"
    result = COMMON.evaluate_case(
        case,
        _common_matrix(matrix),
        sanitized,
        return_code,
    )
    replaced_check_names = replaced_fields | {
        "phase_stage_courant",
        "phase_stage_limited_edges",
    }
    result["checks"] = [
        check
        for check in result["checks"]
        if check["name"] not in replaced_check_names
    ]
    for field in replaced_check_names:
        result["metrics"].pop(field, None)

    expected = expected_case_observables(case, matrix)
    thresholds = matrix["thresholds"]
    original_interface = parsed.get("interface")
    original_phase_stage = parsed.get("phase_stage")
    original_newton = parsed.get("newton")

    def numeric_check(name: str, tolerance: float) -> None:
        try:
            actual = COMMON._numeric(original_interface, name)
        except (TypeError, ValueError) as error:
            actual = math.nan
            passed = False
            display_actual: Any = str(error)
        else:
            wanted = expected[name]
            passed = abs(actual - wanted) <= tolerance
            display_actual = actual
            result["metrics"][name] = actual
        result["checks"].append(
            {
                "name": name,
                "passed": passed,
                "actual": display_actual,
                "expected": {
                    "value": expected[name],
                    "absolute_tolerance": tolerance,
                },
            }
        )

    for phase in ("negative", "positive"):
        for component in range(3):
            name = f"{phase}_momentum_{component}"
            numeric_check(
                name,
                _relative_tolerance(
                    thresholds["bulk_momentum_absolute"],
                    thresholds["bulk_momentum_relative"],
                    expected[name],
                ),
            )
        name = f"{phase}_kinetic_energy"
        numeric_check(
            name,
            _relative_tolerance(
                thresholds["bulk_kinetic_energy_absolute"],
                thresholds["bulk_kinetic_energy_relative"],
                expected[name],
            ),
        )
        for prefix in ("", "viscous_"):
            for component in range(3):
                numeric_check(
                    f"{phase}_{prefix}traction_integral_{component}",
                    thresholds["traction_moment_absolute"],
                )
    for prefix in ("", "viscous_"):
        for component in range(3):
            numeric_check(
                f"{prefix}traction_jump_integral_{component}",
                thresholds["traction_moment_absolute"],
            )
    for name in ("traction_jump_sq", "viscous_traction_jump_sq"):
        numeric_check(name, thresholds["traction_squared_absolute"])
    for component in range(3):
        numeric_check(
            f"prescribed_viscous_traction_jump_target_{component}",
            thresholds["traction_moment_absolute"],
        )
    numeric_check(
        "prescribed_viscous_traction_jump_error_sq",
        thresholds["prescribed_viscous_error_squared"],
    )
    numeric_check(
        "prescribed_stress_jump_residual_sq",
        thresholds["prescribed_stress_error_squared"],
    )

    for field, expected_value in (
        ("abs_tol", matrix["nonlinear_solver"]["absolute_tolerance"]),
        ("rel_tol", matrix["nonlinear_solver"]["relative_tolerance"]),
    ):
        check_name = f"newton_{field}"
        try:
            value = COMMON._numeric(original_newton, field)
        except (TypeError, ValueError) as error:
            actual: Any = str(error)
            passed = False
        else:
            actual = value
            passed = value == expected_value
            result["metrics"][check_name] = value
        result["checks"].append(
            {
                "name": check_name,
                "passed": passed,
                "actual": actual,
                "expected": expected_value,
            }
        )

    nonlinear_residuals: dict[str, float] = {}
    for key in (
        "nonlinear_initial_residual_norm",
        "nonlinear_final_residual_norm",
    ):
        try:
            value = COMMON._numeric(original_interface, key)
        except (TypeError, ValueError) as error:
            actual = str(error)
            passed = False
        else:
            actual = value
            passed = (
                0.0 <= value
                <= matrix["nonlinear_solver"]["absolute_tolerance"]
            )
            nonlinear_residuals[key] = value
            result["metrics"][key] = value
        result["checks"].append(
            {
                "name": key,
                "passed": passed,
                "actual": actual,
                "expected": {
                    "minimum": 0.0,
                    "maximum": matrix["nonlinear_solver"][
                        "absolute_tolerance"
                    ],
                },
            }
        )
    initial_residual = nonlinear_residuals.get(
        "nonlinear_initial_residual_norm"
    )
    final_residual = nonlinear_residuals.get("nonlinear_final_residual_norm")
    residual_consistency = (
        initial_residual is not None
        and final_residual is not None
        and abs(initial_residual - final_residual)
        <= thresholds["absolute_zero"]
    )
    result["checks"].append(
        {
            "name": "nonlinear_zero_iteration_residual_consistency",
            "passed": residual_consistency,
            "actual": [initial_residual, final_residual],
            "expected": f"absolute difference <= {thresholds['absolute_zero']}",
        }
    )

    try:
        courant = COMMON._numeric(original_phase_stage, "courant")
    except (TypeError, ValueError) as error:
        courant = math.nan
        courant_actual: Any = str(error)
        courant_passed = False
    else:
        courant_actual = courant
        courant_passed = (
            0.0 <= courant
            <= thresholds["maximum_finite_step_courant"]
        )
        result["metrics"]["phase_stage_courant"] = courant
    result["checks"].append(
        {
            "name": "phase_stage_courant",
            "passed": courant_passed,
            "actual": courant_actual,
            "expected": {
                "minimum": 0.0,
                "maximum": thresholds["maximum_finite_step_courant"],
            },
        }
    )

    try:
        maximum_nodal_boundary_transfer = COMMON._numeric(
            original_phase_stage,
            "maximum_nodal_boundary_mass_transfer",
        )
        boundary_mass_tolerance = COMMON._numeric(
            original_phase_stage,
            "boundary_mass_tolerance",
        )
    except (TypeError, ValueError) as error:
        balanced_flux_actual: Any = str(error)
        balanced_flux_passed = False
    else:
        balanced_flux_actual = {
            "maximum_nodal_boundary_mass_transfer": (
                maximum_nodal_boundary_transfer
            ),
            "boundary_mass_tolerance": boundary_mass_tolerance,
        }
        balanced_flux_passed = (
            math.isfinite(maximum_nodal_boundary_transfer)
            and math.isfinite(boundary_mass_tolerance)
            and boundary_mass_tolerance >= 0.0
            and maximum_nodal_boundary_transfer > boundary_mass_tolerance
        )
        result["metrics"][
            "phase_stage_maximum_nodal_boundary_mass_transfer"
        ] = maximum_nodal_boundary_transfer
        result["metrics"]["phase_stage_boundary_mass_tolerance"] = (
            boundary_mass_tolerance
        )
    result["checks"].append(
        {
            "name": "phase_stage_balanced_boundary_flux_exercised",
            "passed": balanced_flux_passed,
            "actual": balanced_flux_actual,
            "expected": (
                "finite maximum nodal transfer above the closed-domain "
                "tolerance with the separately checked zero global transfer"
            ),
        }
    )

    try:
        limited_edges = COMMON._integer(original_phase_stage, "limited_edges")
    except (TypeError, ValueError) as error:
        limited_edges_actual: Any = str(error)
        limited_edges_passed = False
    else:
        limited_edges_actual = limited_edges
        limited_edges_passed = limited_edges >= 0
        result["metrics"]["phase_stage_limited_edges"] = limited_edges
    result["checks"].append(
        {
            "name": "phase_stage_limited_edges",
            "passed": limited_edges_passed,
            "actual": limited_edges_actual,
            "expected": "nonnegative reported limiter count",
        }
    )

    expected_reconciliation_diagnostic = (
        "stationary_geometry_equilibrium_projection"
    )
    for record_name, result_prefix in (
        ("phase_geometry", "phase_geometry"),
        ("maintenance", "maintenance"),
    ):
        record = parsed.get(record_name)
        diagnostic = (
            record.get("reconciliation_diagnostic")
            if isinstance(record, dict)
            else None
        )
        diagnostic_name = f"{result_prefix}_reconciliation_diagnostic"
        result["checks"].append(
            {
                "name": diagnostic_name,
                "passed": diagnostic == expected_reconciliation_diagnostic,
                "actual": diagnostic,
                "expected": expected_reconciliation_diagnostic,
            }
        )
        if diagnostic is not None:
            result["metrics"][diagnostic_name] = diagnostic

        for key, require_nonzero in (
            ("reconciliation_initial_residual_norm", True),
            ("reconciliation_final_residual_norm", False),
        ):
            check_name = f"{result_prefix}_{key}"
            try:
                value = COMMON._numeric(record, key)
            except (TypeError, ValueError) as error:
                value_actual: Any = str(error)
                value_passed = False
            else:
                value_actual = value
                value_passed = (
                    value > thresholds["absolute_zero"]
                    if require_nonzero
                    else abs(value) <= thresholds["absolute_zero"]
                )
                result["metrics"][check_name] = value
            result["checks"].append(
                {
                    "name": check_name,
                    "passed": value_passed,
                    "actual": value_actual,
                    "expected": (
                        f"> {thresholds['absolute_zero']}"
                        if require_nonzero
                        else f"absolute <= {thresholds['absolute_zero']}"
                    ),
                }
            )

    def boolean_check(name: str, wanted: bool) -> None:
        try:
            actual = COMMON._boolean(original_interface, name)
        except (TypeError, ValueError) as error:
            actual = str(error)
            passed = False
        else:
            passed = actual is wanted
            result["metrics"][name] = actual
        result["checks"].append(
            {
                "name": name,
                "passed": passed,
                "actual": actual,
                "expected": wanted,
            }
        )

    boolean_check("prescribed_pressure_jump_applicable", False)
    boolean_check("prescribed_viscous_traction_jump_applicable", True)
    boolean_check("prescribed_stress_jump_residual_applicable", True)
    result["failed_checks"] = [
        check["name"] for check in result["checks"] if not check["passed"]
    ]
    result["passed"] = not result["failed_checks"]
    return result


def evaluate_reversal_pairs(
    results: list[dict[str, Any]],
    matrix: dict[str, Any],
    required_pairs: int | None = None,
) -> dict[str, Any]:
    tolerance = matrix["thresholds"]["side_reversal_absolute"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["reversal_pair"], []).append(result)
    expected_pairs = (
        set(grouped)
        if required_pairs is not None
        else {case["reversal_pair"] for case in matrix["cases"]}
    )
    checks: list[dict[str, Any]] = []
    if required_pairs is not None and len(expected_pairs) != required_pairs:
        checks.append(
            {
                "name": "required_reversal_pair_count",
                "passed": False,
                "actual": len(expected_pairs),
                "expected": required_pairs,
            }
        )

    def compare(pair: str, name: str, first: float, second: float) -> None:
        allowed = tolerance * max(1.0, abs(first), abs(second))
        passed = (
            math.isfinite(first)
            and math.isfinite(second)
            and abs(first - second) <= allowed
        )
        checks.append(
            {
                "name": f"{pair}:{name}",
                "passed": passed,
                "first": first,
                "second": second,
                "tolerance": allowed,
            }
        )

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
        first = forward["metrics"]
        second = reverse["metrics"]
        compare(pair, "interface_measure", first["interface_measure"], second["interface_measure"])
        for quantity in ("volume", "mass"):
            compare(
                pair,
                f"negative_to_positive_{quantity}",
                first[f"negative_{quantity}"],
                second[f"positive_{quantity}"],
            )
            compare(
                pair,
                f"positive_to_negative_{quantity}",
                first[f"positive_{quantity}"],
                second[f"negative_{quantity}"],
            )
        for component in range(3):
            for phase, opposite in (("negative", "positive"), ("positive", "negative")):
                compare(
                    pair,
                    f"{phase}_to_{opposite}_momentum_{component}",
                    first[f"{phase}_momentum_{component}"],
                    second[f"{opposite}_momentum_{component}"],
                )
                for prefix in ("", "viscous_"):
                    compare(
                        pair,
                        f"signed_{phase}_to_{opposite}_{prefix}traction_{component}",
                        first[f"{phase}_{prefix}traction_integral_{component}"],
                        -second[f"{opposite}_{prefix}traction_integral_{component}"],
                    )
            for prefix in ("", "viscous_"):
                compare(
                    pair,
                    f"{prefix}traction_jump_integral_{component}",
                    first[f"{prefix}traction_jump_integral_{component}"],
                    second[f"{prefix}traction_jump_integral_{component}"],
                )
            compare(
                pair,
                f"prescribed_target_{component}",
                first[f"prescribed_viscous_traction_jump_target_{component}"],
                second[f"prescribed_viscous_traction_jump_target_{component}"],
            )
        for phase, opposite in (("negative", "positive"), ("positive", "negative")):
            compare(
                pair,
                f"{phase}_to_{opposite}_kinetic_energy",
                first[f"{phase}_kinetic_energy"],
                second[f"{opposite}_kinetic_energy"],
            )
        for field in (
            "traction_jump_sq",
            "viscous_traction_jump_sq",
            "prescribed_viscous_traction_jump_error_sq",
            "prescribed_stress_jump_residual_sq",
        ):
            compare(pair, field, first[field], second[field])
    failed = [check["name"] for check in checks if not check["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def validate_effective_configuration(
    document: Any, case: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(
        document.get("modules"), list
    ):
        raise ValueError("effective configuration has no module list")
    transport_modules = [
        module
        for module in document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "level_set_transport"
    ]
    if len(transport_modules) != 1:
        raise ValueError("effective configuration lacks unique phase transport")
    conservative_phase = transport_modules[0].get("conservative_phase")
    if (
        not isinstance(conservative_phase, dict)
        or conservative_phase.get("boundary_flux_policy")
        != "globally_balanced_discrete_q_flux"
    ):
        raise ValueError(
            "effective configuration changed the phase boundary-flux policy"
        )
    common_document = copy.deepcopy(document)
    common_transport = next(
        module
        for module in common_document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "level_set_transport"
    )
    common_transport["conservative_phase"]["boundary_flux_policy"] = (
        "closed_domain_discrete_q_flux_only"
    )
    result = COMMON.validate_effective_configuration(common_document)
    momentum = next(
        module
        for module in document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "incompressible_two_fluid"
    )
    interface = momentum.get("interface")
    material = momentum.get("material")
    pressure_space = momentum.get("pressure_space")
    boundaries = momentum.get("boundary_conditions")
    if momentum.get("momentum_operator") != "stokes":
        raise ValueError("effective configuration changed the momentum operator")
    if not isinstance(interface, dict):
        raise ValueError("effective configuration lacks the interface contract")
    if interface.get("prescribed_pressure_jump_applicable") is not False:
        raise ValueError("effective configuration enabled a pressure-jump target")
    if interface.get("prescribed_viscous_traction_jump_applicable") is not True:
        raise ValueError("effective configuration disabled the viscous traction target")
    vector = interface.get("prescribed_viscous_traction_jump")
    target = traction_jump_target(case)[:2]
    if (
        not isinstance(vector, list)
        or len(vector) != 2
        or any(
            COMMON.require_finite(value, "effective viscous traction target")
            != target[index]
            for index, value in enumerate(vector)
        )
    ):
        raise ValueError("effective configuration changed the viscous traction target")
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
    if (
        not isinstance(pressure_space, dict)
        or pressure_space.get("representation") != "separate_phase_fields"
        or pressure_space.get("shared_gauge_count") != 1
    ):
        raise ValueError("effective configuration changed the shared pressure gauge")
    valid_shared_boundary = (
        isinstance(boundaries, dict)
        and boundaries.get("shared_velocity_dirichlet_count") == 1
        and boundaries.get("shared_velocity_dirichlet_policy")
        == "identical_external_data_on_both_phase_restrictions"
        and boundaries.get("negative_phase_local_velocity_dirichlet_count") == 0
        and boundaries.get("positive_phase_local_velocity_dirichlet_count") == 0
    )
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
    valid_shared_boundary = (
        valid_shared_boundary
        and isinstance(entry, dict)
        and entry.get("active_components") == [True, True]
        and isinstance(values, list)
        and len(values) == 2
        and all(isinstance(item, dict) for item in values)
        and [item.get("kind") for item in values]
        == ["time_coefficient", "time_coefficient"]
    )
    if not valid_shared_boundary:
        raise ValueError("effective configuration changed the shared velocity boundary")
    result.update(
        {
            "momentum_operator": "stokes",
            "prescribed_viscous_traction_jump": list(target),
            "shared_pressure_gauge_count": 1,
            "shared_velocity_dirichlet_count": 1,
            "phase_boundary_flux_policy": (
                "globally_balanced_discrete_q_flux"
            ),
        }
    )
    return result


def qualification_environment(
    matrix: dict[str, Any],
    inherited: Any,
    case_directory: Path,
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
    velocity_path = case_directory / "velocity.dat"
    solver_path = case_directory / "solver.xml"
    COMMON.write_text_create_only(mesh_path, render_mesh(matrix, case))
    COMMON.write_text_create_only(wall_path, render_wall(matrix))
    COMMON.write_text_create_only(
        velocity_path, render_velocity_data(matrix, case)
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
    environment = qualification_environment(
        matrix, os.environ, case_directory
    )
    execution = COMMON.run_monitored(
        command,
        environment,
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
        "velocity": COMMON.sha256_file(velocity_path),
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
                "expected": "shared-boundary viscous-jump two-fluid configuration",
            }
        )
        result["failed_checks"].append("effective_configuration")
        result["passed"] = False
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
            "planar_viscous_traction_jump_gate_passed": False,
            "exit_code": 1,
        }
    if qualification_eligible:
        return {
            "outcome": "PASS",
            "planar_viscous_traction_jump_gate_passed": True,
            "exit_code": 0,
        }
    return {
        "outcome": "DEVELOPMENT_PASS",
        "planar_viscous_traction_jump_gate_passed": False,
        "exit_code": 0,
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
    provenance = COMMON.source_provenance(REPOSITORY_ROOT)
    input_identity = {
        "matrix": COMMON.committed_path_identity(REPOSITORY_ROOT, matrix_path),
        "runner": COMMON.committed_path_identity(REPOSITORY_ROOT, SCRIPT_PATH),
        "pressure_jump_runner_dependency": COMMON.committed_path_identity(
            REPOSITORY_ROOT, PRESSURE_RUNNER_PATH
        ),
        "constant_state_runner_dependency": COMMON.committed_path_identity(
            REPOSITORY_ROOT, PRESSURE.COMMON_RUNNER_PATH
        ),
    }
    qualification_eligible = provenance["tracked_clean"] and all(
        item["matches_head"] for item in input_identity.values()
    )
    if not qualification_eligible and not allow_tracked_dirty_development:
        raise RuntimeError("tracked source is dirty; qualification execution refused")
    cache_path = COMMON.find_cmake_cache(solver)
    preflight = {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": COMMON.sha256_file(matrix_path),
        "runner_sha256": COMMON.sha256_file(SCRIPT_PATH),
        "pressure_jump_runner_sha256": COMMON.sha256_file(PRESSURE_RUNNER_PATH),
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
    gate_passed = disposition["planar_viscous_traction_jump_gate_passed"]
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
        "# WP-10 planar viscous-traction-jump qualification record\n\n"
        f"- Matrix: `{matrix['matrix_id']}`\n"
        f"- Source revision: `{provenance['revision']}`\n"
        f"- Tracked source clean: `{str(provenance['tracked_clean']).lower()}`\n"
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`\n"
        f"- Reversal pairs passed: `{str(reversal['passed']).lower()}`\n"
        f"- Planar viscous-traction-jump gate passed: `{str(gate_passed).lower()}`\n"
        "- The sustained-flow, conservation, FSR-08, WP-10, and Q7 gates remain open.\n"
    )
    COMMON.write_text_create_only(output_directory / "record.md", record)
    COMMON.write_checksums(output_directory)
    return summary


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--requested-claim", default="planar_viscous_traction_jump_prerequisite"
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
                f"shear_rate={case['shear_rate']}"
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
