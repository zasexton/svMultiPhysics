#!/usr/bin/env python3
"""Run the frozen WP-10 planar prescribed-pressure-jump gate."""

from __future__ import annotations

import argparse
import copy
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
    "free_surface_wp10_pressure_jump_matrix.json"
)
COMMON_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp10_constant_state_qualification.py"
)
EXPECTED_MATRIX_SHA256 = (
    "be5be7bd25d91a1fa9e9cf66835ea84af4892bdb4230b28b1b2b50458e885b23"
)
EXPECTED_COMMON_RUNNER_SHA256 = (
    "b899c9415d2800adf10a568e0e384139b05137c0ee86e518de5601c57672a5ca"
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
    "thresholds",
    "execution",
    "cases",
    "required_later_progression",
    "qualification_disposition",
}


def _load_common_runner() -> Any:
    digest = _sha256_path_without_common(COMMON_RUNNER_PATH)
    if digest != EXPECTED_COMMON_RUNNER_SHA256:
        raise RuntimeError("constant-state runner dependency bytes changed")
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_constant_state_qualification_common",
        COMMON_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the constant-state runner dependency")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _sha256_path_without_common(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


COMMON = _load_common_runner()


def _validate_matrix_structure(matrix: Any) -> dict[str, Any]:
    if not isinstance(matrix, dict) or set(matrix) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("pressure-jump matrix top-level contract changed")
    expected_metadata = {
        "schema_version": 1,
        "matrix_id": "free_surface_wp10_pressure_jump_v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
        "accepted_claim": "planar_pressure_jump_prerequisite",
    }
    for key, expected in expected_metadata.items():
        if matrix.get(key) != expected:
            raise ValueError(f"pressure-jump matrix {key} changed")
    if matrix.get("rejected_claims") != [
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
    ]:
        raise ValueError("pressure-jump rejected-claim boundary changed")
    if matrix.get("qualification_disposition") != {
        "planar_pressure_jump_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }:
        raise ValueError("pressure-jump disposition changed")

    model = matrix.get("model_envelope")
    expected_model = {
        "physical_model": "incompressible_two_fluid",
        "spatial_dimension": 2,
        "mesh": "affine_p1_triangle",
        "interface_geometry": "linear_corner",
        "domain": "closed_unit_square",
        "exterior_velocity": "homogeneous_strong_dirichlet_both_phases",
        "velocity": "identically_zero_in_both_phases",
        "pressure_gauge": "first_free_negative_pressure_dof_zero",
        "pressure_state": "p_negative_zero_p_positive_negative_target",
        "surface_tension": 0.0,
        "body_force": [0.0, 0.0],
        "phase_transport": (
            "locally_conservative_p1_indicator_with_geometry_reconciliation"
        ),
        "solver": "fsils_block_schur",
    }
    if model != expected_model:
        raise ValueError("pressure-jump model envelope changed")

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
        raise ValueError("pressure-jump mesh contract is absent")
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
        raise ValueError("pressure-jump mesh contract is inconsistent")

    if matrix.get("time") != {
        "steps": 1,
        "dt": 0.01,
        "scheme": "BackwardEuler",
    }:
        raise ValueError("pressure-jump time contract changed")
    thresholds = matrix.get("thresholds")
    expected_threshold_keys = {
        "absolute_zero",
        "pressure_jump_absolute",
        "pressure_moment_absolute",
        "traction_moment_absolute",
        "prescribed_pressure_error_squared",
        "prescribed_stress_error_squared",
        "phase_volume_absolute",
        "phase_mass_relative",
        "interface_measure_absolute",
        "side_reversal_absolute",
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_threshold_keys:
        raise ValueError("pressure-jump threshold contract changed")
    for key in expected_threshold_keys - {
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
        "maximum_nonlinear_iterations",
        "maximum_linear_iterations",
    }:
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
    }:
        raise ValueError("pressure-jump execution envelope changed")
    for key in execution:
        if not isinstance(execution[key], int) or execution[key] < 1:
            raise ValueError(f"execution field {key} must be a positive integer")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("pressure-jump case coverage changed")
    case_ids: set[str] = set()
    reversal_pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "reversal_pair",
            "orientation",
            "offset",
            "level_set_sign",
            "pressure_jump",
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
            "mpi_ranks",
        }:
            raise ValueError("pressure-jump case contract is malformed")
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
            raise ValueError("pressure-jump case identity is invalid")
        case_ids.add(case_id)
        COMMON.require_finite(case.get("offset"), f"{case_id} offset")
        jump = COMMON.require_finite(case.get("pressure_jump"), f"{case_id} jump")
        if jump == 0.0:
            raise ValueError(f"{case_id} pressure jump must be nonzero")
        for field in (
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
        ):
            COMMON.require_finite_positive(case.get(field), f"{case_id} {field}")
        geometry = COMMON.analytic_planar_geometry(case)
        if min(geometry.values()) <= 0.0:
            raise ValueError(f"{case_id} does not cut both phases")
        for j in range(ny + 1):
            for i in range(nx + 1):
                if abs(COMMON.level_set_value(case, i / nx, j / ny)) <= 1.0e-14:
                    raise ValueError(f"{case_id} crosses a mesh vertex")
        reversal_pairs.setdefault(pair, []).append(case)

    if len(reversal_pairs) != 6:
        raise ValueError("pressure-jump reversal-pair coverage changed")
    for pair, members in reversal_pairs.items():
        if len(members) != 2 or {item["level_set_sign"] for item in members} != {
            -1,
            1,
        }:
            raise ValueError(f"reversal pair {pair} is incomplete")
        forward = next(item for item in members if item["level_set_sign"] == 1)
        reverse = next(item for item in members if item["level_set_sign"] == -1)
        for field in ("orientation", "offset", "pressure_jump", "mpi_ranks"):
            if forward[field] != reverse[field]:
                raise ValueError(f"reversal pair {pair} changes {field}")
        for first, second in (
            ("negative_density", "positive_density"),
            ("negative_viscosity", "positive_viscosity"),
        ):
            if forward[first] != reverse[second] or forward[second] != reverse[first]:
                raise ValueError(f"reversal pair {pair} does not exchange material data")
    if {case["mpi_ranks"] for case in cases} != {1, 2}:
        raise ValueError("pressure-jump serial and parallel coverage changed")
    if {case["orientation"] for case in cases} != {
        "x",
        "y",
        "x_plus_y",
        "x_minus_y",
    }:
        raise ValueError("pressure-jump orientation coverage changed")
    if {math.copysign(1.0, case["pressure_jump"]) for case in cases} != {
        -1.0,
        1.0,
    }:
        raise ValueError("pressure-jump sign coverage changed")
    maximum_density_ratio = max(
        max(case["negative_density"], case["positive_density"])
        / min(case["negative_density"], case["positive_density"])
        for case in cases
    )
    if maximum_density_ratio != 10000.0:
        raise ValueError("pressure-jump density-ratio coverage changed")
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
            f"unsupported pressure-jump claim {claim!r}; expected "
            f"{matrix['accepted_claim']!r}"
        )
    return claim


def render_mesh(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    root = ET.fromstring(COMMON.render_mesh(matrix, case))
    arrays = {
        item.attrib.get("Name"): item
        for item in root.findall(".//PointData/DataArray")
    }
    vertices = int(matrix["mesh"]["expected_vertices"])
    arrays["p_negative"].text = "\n" + COMMON._format_values([0.0] * vertices) + "\n        "
    positive_pressure = -float(case["pressure_jump"])
    arrays["p_positive"].text = (
        "\n"
        + COMMON._format_values([positive_pressure] * vertices)
        + "\n        "
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def _wall_vertices(nx: int, ny: int) -> list[tuple[int, float, float]]:
    vertices: list[tuple[int, float, float]] = []
    for i in range(nx + 1):
        vertices.append((i + 1, i / nx, 0.0))
    for j in range(1, ny + 1):
        vertices.append((j * (nx + 1) + nx + 1, 1.0, j / ny))
    for i in range(nx - 1, -1, -1):
        vertices.append((ny * (nx + 1) + i + 1, i / nx, 1.0))
    for j in range(ny - 1, 0, -1):
        vertices.append((j * (nx + 1) + 1, 0.0, j / ny))
    return vertices


def render_wall(matrix: dict[str, Any]) -> str:
    nx = int(matrix["mesh"]["nx"])
    ny = int(matrix["mesh"]["ny"])
    vertices = _wall_vertices(nx, ny)
    count = len(vertices)
    global_ids = [item[0] for item in vertices]
    points = [coordinate for _, x, y in vertices for coordinate in (x, y, 0.0)]
    connectivity = [index for line in range(count) for index in (line, (line + 1) % count)]
    offsets = [2 * (index + 1) for index in range(count)]
    return f"""<?xml version=\"1.0\"?>
<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">
  <PolyData>
    <Piece NumberOfPoints=\"{count}\" NumberOfVerts=\"0\" NumberOfLines=\"{count}\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">
      <PointData>
        <DataArray type=\"Int64\" Name=\"GlobalNodeID\" format=\"ascii\">
{COMMON._format_values(global_ids)}
        </DataArray>
      </PointData>
      <Points>
        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">
{COMMON._format_values(points, 6)}
        </DataArray>
      </Points>
      <Lines>
        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">
{COMMON._format_values(connectivity, 12)}
        </DataArray>
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">
{COMMON._format_values(offsets)}
        </DataArray>
      </Lines>
    </Piece>
  </PolyData>
</VTKFile>
"""


def render_solver(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    root = ET.fromstring(COMMON.render_solver(matrix, case))
    mesh = root.find("Add_mesh")
    fluid = root.find("Add_equation[@type='fluid']")
    if mesh is None or fluid is None:
        raise RuntimeError("common solver template changed")
    face = ET.SubElement(mesh, "Add_face", {"name": "wall"})
    ET.SubElement(face, "Face_file_path").text = "wall.vtp"
    jump = ET.Element("Prescribed_pressure_jump")
    jump.text = format(float(case["pressure_jump"]), ".17g")
    nitsche = fluid.find("Two_fluid_interface_nitsche_gamma")
    insert_at = list(fluid).index(nitsche) if nitsche is not None else 0
    fluid.insert(insert_at, jump)
    boundary = ET.Element("Add_BC", {"name": "wall"})
    ET.SubElement(boundary, "Type").text = "Dir"
    ET.SubElement(boundary, "Time_dependence").text = "Steady"
    ET.SubElement(boundary, "Value").text = "0"
    linear_solver = fluid.find("LS")
    boundary_at = list(fluid).index(linear_solver) if linear_solver is not None else len(fluid)
    fluid.insert(boundary_at, boundary)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


parse_case_output = COMMON.parse_case_output


def _common_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(matrix)
    adapted["thresholds"]["phase_partition_absolute"] = matrix["thresholds"][
        "phase_volume_absolute"
    ]
    adapted["thresholds"]["phase_measure_absolute"] = matrix["thresholds"][
        "phase_volume_absolute"
    ]
    return adapted


def evaluate_case(
    case: dict[str, Any],
    matrix: dict[str, Any],
    parsed: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    sanitized = copy.deepcopy(parsed)
    interface = sanitized.get("interface")
    pressure_fields = (
        "mean_pressure_jump",
        "pressure_jump_sq",
        "pressure_jump_integral",
        "traction_jump_sq",
        "traction_jump_normal_integral",
    )
    if isinstance(interface, dict):
        for field in pressure_fields:
            if field in interface:
                interface[field] = "0"
    result = COMMON.evaluate_case(
        case,
        _common_matrix(matrix),
        sanitized,
        return_code,
    )
    result["checks"] = [
        check for check in result["checks"] if check["name"] not in pressure_fields
    ]
    for field in pressure_fields:
        result["metrics"].pop(field, None)

    thresholds = matrix["thresholds"]
    geometry = COMMON.analytic_planar_geometry(case)
    target = float(case["pressure_jump"])
    measure = geometry["interface_measure"]
    expected = {
        "mean_pressure_jump": (target, thresholds["pressure_jump_absolute"]),
        "pressure_jump_integral": (
            target * measure,
            thresholds["pressure_moment_absolute"],
        ),
        "pressure_jump_sq": (
            target * target * measure,
            thresholds["pressure_moment_absolute"],
        ),
        "traction_jump_normal_integral": (
            -target * measure,
            thresholds["traction_moment_absolute"],
        ),
        "traction_jump_sq": (
            target * target * measure,
            thresholds["traction_moment_absolute"],
        ),
        "prescribed_pressure_jump_target": (
            target,
            thresholds["pressure_jump_absolute"],
        ),
        "prescribed_pressure_jump_error_sq": (
            0.0,
            thresholds["prescribed_pressure_error_squared"],
        ),
        "prescribed_stress_jump_residual_sq": (
            0.0,
            thresholds["prescribed_stress_error_squared"],
        ),
    }
    original_interface = parsed.get("interface")
    for name, (wanted, tolerance) in expected.items():
        try:
            actual = COMMON._numeric(original_interface, name)
        except (TypeError, ValueError) as error:
            actual = math.nan
            passed = False
            display_actual: Any = str(error)
        else:
            passed = abs(actual - wanted) <= tolerance
            display_actual = actual
            result["metrics"][name] = actual
        result["checks"].append(
            {
                "name": name,
                "passed": passed,
                "actual": display_actual,
                "expected": {"value": wanted, "absolute_tolerance": tolerance},
            }
        )
    try:
        applicable = COMMON._boolean(
            original_interface, "prescribed_pressure_jump_applicable"
        )
    except (TypeError, ValueError) as error:
        applicable = False
        applicable_actual: Any = str(error)
    else:
        applicable_actual = applicable
        result["metrics"]["prescribed_pressure_jump_applicable"] = applicable
    result["checks"].append(
        {
            "name": "prescribed_pressure_jump_applicable",
            "passed": applicable,
            "actual": applicable_actual,
            "expected": True,
        }
    )
    result["failed_checks"] = [
        check["name"] for check in result["checks"] if not check["passed"]
    ]
    result["passed"] = not result["failed_checks"]
    return result


def evaluate_reversal_pairs(
    results: list[dict[str, Any]], matrix: dict[str, Any]
) -> dict[str, Any]:
    outcome = COMMON.evaluate_reversal_pairs(results, _common_matrix(matrix))
    tolerance = matrix["thresholds"]["side_reversal_absolute"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["reversal_pair"], []).append(result)
    for pair in sorted({case["reversal_pair"] for case in matrix["cases"]}):
        members = grouped.get(pair, [])
        if (
            len(members) != 2
            or {member["level_set_sign"] for member in members} != {-1, 1}
            or not all(member.get("passed") for member in members)
        ):
            continue
        forward = next(member for member in members if member["level_set_sign"] == 1)
        reverse = next(member for member in members if member["level_set_sign"] == -1)
        for field in (
            "mean_pressure_jump",
            "pressure_jump_integral",
            "pressure_jump_sq",
            "traction_jump_normal_integral",
            "traction_jump_sq",
            "prescribed_pressure_jump_error_sq",
            "prescribed_stress_jump_residual_sq",
        ):
            first = float(forward["metrics"][field])
            second = float(reverse["metrics"][field])
            passed = abs(first - second) <= tolerance * max(
                1.0, abs(first), abs(second)
            )
            outcome["checks"].append(
                {
                    "name": f"{pair}:{field}",
                    "passed": passed,
                    "first": first,
                    "second": second,
                    "tolerance": tolerance,
                }
            )
    outcome["failed_checks"] = [
        check["name"] for check in outcome["checks"] if not check["passed"]
    ]
    outcome["passed"] = not outcome["failed_checks"]
    return outcome


def validate_effective_configuration(
    document: Any, expected_pressure_jump: float | None = None
) -> dict[str, Any]:
    result = COMMON.validate_effective_configuration(document)
    momentum = next(
        module
        for module in document["modules"]
        if isinstance(module, dict)
        and module.get("component") == "incompressible_two_fluid"
    )
    interface = momentum.get("interface")
    pressure_space = momentum.get("pressure_space")
    if not isinstance(interface, dict):
        raise ValueError("effective configuration lacks the interface contract")
    if interface.get("prescribed_pressure_jump_applicable") is not True:
        raise ValueError("effective configuration disabled the prescribed jump")
    jump = COMMON.require_finite(
        interface.get("prescribed_pressure_jump"), "effective prescribed jump"
    )
    if expected_pressure_jump is not None and jump != float(expected_pressure_jump):
        raise ValueError("effective configuration changed the prescribed jump")
    if (
        not isinstance(pressure_space, dict)
        or pressure_space.get("representation") != "separate_phase_fields"
        or pressure_space.get("shared_gauge_count") != 1
    ):
        raise ValueError("effective configuration changed the shared pressure gauge")
    result.update(
        {
            "prescribed_pressure_jump_applicable": True,
            "prescribed_pressure_jump": jump,
            "shared_pressure_gauge_count": 1,
        }
    )
    return result


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
    solver_path = case_directory / "solver.xml"
    COMMON.write_text_create_only(mesh_path, render_mesh(matrix, case))
    COMMON.write_text_create_only(wall_path, render_wall(matrix))
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
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": str(matrix["execution"]["omp_threads"]),
            "OPENBLAS_NUM_THREADS": "1",
            "TMPDIR": str(temporary_directory),
        }
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
        "solver": COMMON.sha256_file(solver_path),
    }
    effective_path = case_directory / "effective_configuration.json"
    try:
        effective = validate_effective_configuration(
            COMMON.read_json_strict(effective_path), float(case["pressure_jump"])
        )
    except (OSError, TypeError, ValueError) as error:
        effective = {"error": str(error)}
        result["checks"].append(
            {
                "name": "effective_configuration",
                "passed": False,
                "actual": str(error),
                "expected": "gauged prescribed-jump two-fluid configuration",
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
            "planar_pressure_jump_gate_passed": False,
            "exit_code": 1,
        }
    if qualification_eligible:
        return {
            "outcome": "PASS",
            "planar_pressure_jump_gate_passed": True,
            "exit_code": 0,
        }
    return {
        "outcome": "DEVELOPMENT_PASS",
        "planar_pressure_jump_gate_passed": False,
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
        "constant_state_runner_dependency": COMMON.committed_path_identity(
            REPOSITORY_ROOT, COMMON_RUNNER_PATH
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
        "constant_state_runner_sha256": COMMON.sha256_file(COMMON_RUNNER_PATH),
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
    gate_passed = disposition["planar_pressure_jump_gate_passed"]
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
        "# WP-10 planar pressure-jump qualification record\n\n"
        f"- Matrix: `{matrix['matrix_id']}`\n"
        f"- Source revision: `{provenance['revision']}`\n"
        f"- Tracked source clean: `{str(provenance['tracked_clean']).lower()}`\n"
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`\n"
        f"- Reversal pairs passed: `{str(reversal['passed']).lower()}`\n"
        f"- Planar pressure-jump gate passed: `{str(gate_passed).lower()}`\n"
        "- FSR-08, WP-10, and Q7 remain open.\n"
    )
    COMMON.write_text_create_only(output_directory / "record.md", record)
    COMMON.write_checksums(output_directory)
    return summary


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--requested-claim", default="planar_pressure_jump_prerequisite"
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
                f"jump={case['pressure_jump']}"
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
