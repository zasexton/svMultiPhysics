#!/usr/bin/env python3
"""Run the frozen WP-10 incompressible two-fluid constant-state gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MATRIX = SCRIPT_PATH.with_name(
    "free_surface_wp10_constant_state_matrix.json"
)
EXPECTED_MATRIX_SHA256 = (
    "54d146cf4e6fb1cc5bce996ff98cec1123ab888ee1d9ab1a5c4c9dc902ab3239"
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
KEY_VALUE_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=(?:'([^']*)'|\"([^\"]*)\"|([^\s]+))"
)
RANK_PREFIX_PATTERN = re.compile(r"\[R([0-9]+)\]")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_strict(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=reject_duplicate_keys)


def require_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def require_finite_positive(value: Any, label: str) -> float:
    result = require_finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _validate_matrix_structure(matrix: Any) -> dict[str, Any]:
    if not isinstance(matrix, dict) or set(matrix) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("constant-state matrix top-level contract changed")
    expected_metadata = {
        "schema_version": 1,
        "matrix_id": "free_surface_wp10_constant_state_v2",
        "status": "FROZEN_BEFORE_EXECUTION",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
        "accepted_claim": "constant_state_prerequisite",
    }
    for key, expected in expected_metadata.items():
        if matrix.get(key) != expected:
            raise ValueError(f"constant-state matrix {key} changed")
    if matrix.get("rejected_claims") != [
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
    ]:
        raise ValueError("constant-state rejected-claim boundary changed")
    if matrix.get("qualification_disposition") != {
        "constant_state_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }:
        raise ValueError("constant-state disposition changed")

    mesh = matrix.get("mesh")
    if not isinstance(mesh, dict) or set(mesh) != {
        "nx",
        "ny",
        "bounds",
        "triangle_split",
        "ghost_layers",
        "expected_vertices",
        "expected_triangles",
        "expected_phase_initializer_dofs",
    }:
        raise ValueError("constant-state mesh contract is absent")
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
        or mesh.get("expected_phase_initializer_dofs")
        != 3 * (nx + 1) * (ny + 1)
    ):
        raise ValueError("constant-state mesh contract is inconsistent")

    thresholds = matrix.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("constant-state thresholds are absent")
    for key in (
        "absolute_zero",
        "phase_volume_absolute",
        "phase_partition_absolute",
        "phase_mass_relative",
        "interface_measure_absolute",
        "phase_measure_absolute",
        "side_reversal_absolute",
    ):
        require_finite_positive(thresholds.get(key), f"threshold {key}")
    for key in (
        "minimum_interface_quadrature_points",
        "minimum_phase_quadrature_points",
    ):
        if not isinstance(thresholds.get(key), int) or thresholds[key] < 1:
            raise ValueError(f"threshold {key} must be a positive integer")
    for key in ("maximum_nonlinear_iterations", "maximum_linear_iterations"):
        if thresholds.get(key) != 0:
            raise ValueError(f"threshold {key} must retain exact-entry zero")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("constant-state case coverage changed")
    case_ids: set[str] = set()
    reversal_pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "reversal_pair",
            "orientation",
            "offset",
            "level_set_sign",
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
            "mpi_ranks",
        }:
            raise ValueError("constant-state case contract is malformed")
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
            raise ValueError("constant-state case identity is invalid")
        case_ids.add(case_id)
        require_finite(case.get("offset"), f"{case_id} offset")
        for field in (
            "negative_density",
            "negative_viscosity",
            "positive_density",
            "positive_viscosity",
        ):
            require_finite_positive(case.get(field), f"{case_id} {field}")
        geometry = analytic_planar_geometry(case)
        if not (
            geometry["negative_volume"] > 0.0
            and geometry["positive_volume"] > 0.0
            and geometry["interface_measure"] > 0.0
        ):
            raise ValueError(f"{case_id} does not cut both phases")
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i / nx
                y = j / ny
                if abs(level_set_value(case, x, y)) <= 1.0e-14:
                    raise ValueError(f"{case_id} crosses a mesh vertex")
        reversal_pairs.setdefault(pair, []).append(case)

    if len(reversal_pairs) != 6:
        raise ValueError("constant-state reversal-pair coverage changed")
    for pair, members in reversal_pairs.items():
        if len(members) != 2 or {item["level_set_sign"] for item in members} != {
            -1,
            1,
        }:
            raise ValueError(f"reversal pair {pair} is incomplete")
        forward = next(item for item in members if item["level_set_sign"] == 1)
        reverse = next(item for item in members if item["level_set_sign"] == -1)
        for field in ("orientation", "offset", "mpi_ranks"):
            if forward[field] != reverse[field]:
                raise ValueError(f"reversal pair {pair} changes {field}")
        for first, second in (
            ("negative_density", "positive_density"),
            ("negative_viscosity", "positive_viscosity"),
        ):
            if forward[first] != reverse[second] or forward[second] != reverse[first]:
                raise ValueError(f"reversal pair {pair} does not exchange material data")
    return matrix


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_MATRIX_SHA256:
        raise ValueError("frozen matrix bytes changed")
    return _validate_matrix_structure(read_json_strict(path))


def validate_requested_claim(matrix: dict[str, Any], claim: str) -> str:
    if claim in matrix["rejected_claims"]:
        raise ValueError(
            f"requested claim {claim!r} is outside this progression gate"
        )
    if claim != matrix["accepted_claim"]:
        raise ValueError(
            f"unsupported constant-state claim {claim!r}; expected "
            f"{matrix['accepted_claim']!r}"
        )
    return claim


def analytic_planar_geometry(case: dict[str, Any]) -> dict[str, float]:
    orientation = case["orientation"]
    offset = require_finite(case["offset"], "planar offset")
    if orientation in {"x", "y"}:
        if not 0.0 < offset < 1.0:
            raise ValueError("axis-aligned planar offset must lie inside the unit square")
        base_negative = offset
        interface_measure = 1.0
    elif orientation == "x_plus_y":
        if not 0.0 < offset < 2.0:
            raise ValueError("diagonal planar offset must lie inside the unit square")
        if offset <= 1.0:
            base_negative = 0.5 * offset**2
            interface_measure = math.sqrt(2.0) * offset
        else:
            base_negative = 1.0 - 0.5 * (2.0 - offset) ** 2
            interface_measure = math.sqrt(2.0) * (2.0 - offset)
    elif orientation == "x_minus_y":
        if not -1.0 < offset < 1.0:
            raise ValueError("anti-diagonal planar offset must lie inside the unit square")
        if offset <= 0.0:
            base_negative = 0.5 * (offset + 1.0) ** 2
        else:
            base_negative = 1.0 - 0.5 * (1.0 - offset) ** 2
        interface_measure = math.sqrt(2.0) * (1.0 - abs(offset))
    else:
        raise ValueError(f"unsupported planar orientation: {orientation!r}")
    sign = case.get("level_set_sign")
    if sign == 1:
        negative_volume = base_negative
    elif sign == -1:
        negative_volume = 1.0 - base_negative
    else:
        raise ValueError("level-set sign must be positive or negative one")
    return {
        "negative_volume": negative_volume,
        "positive_volume": 1.0 - negative_volume,
        "interface_measure": interface_measure,
    }


def level_set_value(case: dict[str, Any], x: float, y: float) -> float:
    orientation = case["orientation"]
    if orientation == "x":
        raw = x - case["offset"]
    elif orientation == "y":
        raw = y - case["offset"]
    elif orientation == "x_plus_y":
        raw = x + y - case["offset"]
    elif orientation == "x_minus_y":
        raw = x - y - case["offset"]
    else:
        raise ValueError(f"unsupported planar orientation: {orientation!r}")
    return float(case["level_set_sign"] * raw)


def _format_values(values: list[Any], per_line: int = 8) -> str:
    tokens = [
        format(value, ".17g") if isinstance(value, float) else str(value)
        for value in values
    ]
    return "\n".join(
        "          " + " ".join(tokens[start : start + per_line])
        for start in range(0, len(tokens), per_line)
    )


def render_mesh(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    mesh = matrix["mesh"]
    nx = int(mesh["nx"])
    ny = int(mesh["ny"])
    points: list[float] = []
    level_set: list[float] = []
    for j in range(ny + 1):
        y = j / ny
        for i in range(nx + 1):
            x = i / nx
            points.extend([x, y, 0.0])
            level_set.append(level_set_value(case, x, y))
    connectivity: list[int] = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            if (i + j) % 2 == 0:
                connectivity.extend(
                    [
                        lower_left,
                        lower_right,
                        upper_right,
                        lower_left,
                        upper_right,
                        upper_left,
                    ]
                )
            else:
                connectivity.extend(
                    [
                        lower_left,
                        lower_right,
                        upper_left,
                        lower_right,
                        upper_right,
                        upper_left,
                    ]
                )
    vertices = (nx + 1) * (ny + 1)
    triangles = 2 * nx * ny
    zeros_scalar = [0.0] * vertices
    zeros_vector = [0.0] * (2 * vertices)
    offsets = [3 * (index + 1) for index in range(triangles)]
    return f"""<?xml version=\"1.0\"?>
<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">
  <UnstructuredGrid>
    <Piece NumberOfPoints=\"{vertices}\" NumberOfCells=\"{triangles}\">
      <PointData>
        <DataArray type=\"Int64\" Name=\"GlobalNodeID\" format=\"ascii\">
{_format_values(list(range(1, vertices + 1)))}
        </DataArray>
        <DataArray type=\"Float64\" Name=\"level_set\" format=\"ascii\">
{_format_values(level_set)}
        </DataArray>
        <DataArray type=\"Float64\" Name=\"u_negative\" NumberOfComponents=\"2\" format=\"ascii\">
{_format_values(zeros_vector)}
        </DataArray>
        <DataArray type=\"Float64\" Name=\"p_negative\" format=\"ascii\">
{_format_values(zeros_scalar)}
        </DataArray>
        <DataArray type=\"Float64\" Name=\"u_positive\" NumberOfComponents=\"2\" format=\"ascii\">
{_format_values(zeros_vector)}
        </DataArray>
        <DataArray type=\"Float64\" Name=\"p_positive\" format=\"ascii\">
{_format_values(zeros_scalar)}
        </DataArray>
      </PointData>
      <CellData>
        <DataArray type=\"Int64\" Name=\"GlobalElementID\" format=\"ascii\">
{_format_values(list(range(1, triangles + 1)))}
        </DataArray>
      </CellData>
      <Points>
        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">
{_format_values(points, 6)}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">
{_format_values(connectivity, 12)}
        </DataArray>
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">
{_format_values(offsets)}
        </DataArray>
        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">
{_format_values([5] * triangles)}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""


def render_solver(matrix: dict[str, Any], case: dict[str, Any]) -> str:
    def number(name: str) -> str:
        return format(float(case[name]), ".17g")

    ghost_layers = matrix["mesh"]["ghost_layers"]

    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<svMultiPhysicsFile version=\"0.1\">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
    <Number_of_time_steps>1</Number_of_time_steps>
    <Time_step_size>0.01</Time_step_size>
    <Transient_time_integration_scheme>BackwardEuler</Transient_time_integration_scheme>
    <Spectral_radius_of_infinite_time_step>0.5</Spectral_radius_of_infinite_time_step>
    <Save_results_to_VTK_format>false</Save_results_to_VTK_format>
    <Start_saving_after_time_step>0</Start_saving_after_time_step>
  </GeneralSimulationParameters>
  <Add_mesh name=\"background\">
    <Mesh_file_path>mesh.vtu</Mesh_file_path>
    <Ghost_layers>{ghost_layers}</Ghost_layers>
  </Add_mesh>
  <Add_equation type=\"level_set\">
    <Coupled>true</Coupled>
    <Level_set_field_name>level_set</Level_set_field_name>
    <Level_set_source>unknown</Level_set_source>
    <Auto_register_level_set_field>true</Auto_register_level_set_field>
    <Velocity_source>material_interface_phase_pair</Velocity_source>
    <Material_interface_marker>71</Material_interface_marker>
    <Transport_form>advective</Transport_form>
    <Enable_conservative_phase_transport>true</Enable_conservative_phase_transport>
    <Conservative_phase_field_name>phase</Conservative_phase_field_name>
    <Auto_register_conservative_phase_field>true</Auto_register_conservative_phase_field>
    <Conservative_phase_reconcile_geometry>true</Conservative_phase_reconcile_geometry>
    <Conservative_phase_momentum_relative_tolerance>1.0e-10</Conservative_phase_momentum_relative_tolerance>
    <Enable_bound_preserving_limiter>false</Enable_bound_preserving_limiter>
    <Enable_reinitialization>false</Enable_reinitialization>
    <Enable_volume_correction>false</Enable_volume_correction>
    <Enable_interface_kinematic>true</Enable_interface_kinematic>
    <Interface_kinematic_marker>71</Interface_kinematic_marker>
    <Operator_tag>equations</Operator_tag>
    <LS type=\"GMRES\">
      <Linear_algebra type=\"fsils\">
        <Preconditioner>fsils</Preconditioner>
      </Linear_algebra>
      <Tolerance>1.0e-10</Tolerance>
      <Absolute_tolerance>1.0e-13</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type=\"fluid\">
    <Coupled>true</Coupled>
    <Free_surface_physical_model>IncompressibleTwoFluid</Free_surface_physical_model>
    <Level_set_field_name>level_set</Level_set_field_name>
    <Generated_interface_domain_id>material_interface</Generated_interface_domain_id>
    <Material_interface_marker>71</Material_interface_marker>
    <Negative_phase_density>{number('negative_density')}</Negative_phase_density>
    <Negative_phase_dynamic_viscosity>{number('negative_viscosity')}</Negative_phase_dynamic_viscosity>
    <Positive_phase_density>{number('positive_density')}</Positive_phase_density>
    <Positive_phase_dynamic_viscosity>{number('positive_viscosity')}</Positive_phase_dynamic_viscosity>
    <Two_fluid_surface_tension>0</Two_fluid_surface_tension>
    <Two_fluid_interface_nitsche_gamma>24</Two_fluid_interface_nitsche_gamma>
    <Force_x>0</Force_x>
    <Force_y>0</Force_y>
    <Force_z>0</Force_z>
    <Operator_tag>equations</Operator_tag>
    <LS type=\"NS\">
      <Linear_algebra type=\"fsils\">
        <Preconditioner>fsils</Preconditioner>
      </Linear_algebra>
      <Max_iterations>100</Max_iterations>
      <Tolerance>1.0e-10</Tolerance>
      <Absolute_tolerance>1.0e-13</Absolute_tolerance>
    </LS>
  </Add_equation>
</svMultiPhysicsFile>
"""


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in KEY_VALUE_PATTERN.finditer(line):
        value = next(
            group for group in match.groups()[1:] if group is not None
        )
        values[match.group(1)] = value
    return values


def parse_ranked_key_values(line: str) -> dict[str, str]:
    rank = RANK_PREFIX_PATTERN.search(line)
    if rank is None:
        raise ValueError("ranked diagnostic has no rank prefix")
    values = parse_key_values(line)
    values["log_rank"] = rank.group(1)
    return values


def _unique_record(lines: list[str], marker: str) -> dict[str, str] | None:
    records = [parse_key_values(line) for line in lines if marker in line]
    distinct = {
        json.dumps(record, sort_keys=True): record for record in records
    }
    if not distinct:
        return None
    if len(distinct) != 1:
        raise ValueError(f"case output has inconsistent {marker} records")
    return next(iter(distinct.values()))


def parse_case_output(output: str) -> dict[str, Any]:
    lines = output.splitlines()
    initializers = [
        parse_ranked_key_values(line)
        for line in lines
        if "diagnostic=mesh_field_initialization" in line
    ]
    return {
        "initializers": initializers,
        "phase_stage": _unique_record(lines, "Conservative phase staged"),
        "phase_geometry": _unique_record(
            lines, "Conservative phase geometry validated"
        ),
        "maintenance": _unique_record(
            lines, "diagnostic=conservative_phase_maintenance_ledger"
        ),
        "interface": _unique_record(
            lines, "accepted two-fluid interface diagnostics"
        ),
        "artifact": _unique_record(
            lines, "diagnostic=conservative_phase_flux_artifact"
        ),
        "accepted_step_count": sum(
            "TimeLoop: step_accepted step=1" in line for line in lines
        ),
    }


def _numeric(record: dict[str, str] | None, key: str) -> float:
    if record is None or key not in record:
        raise ValueError(f"missing numeric field {key}")
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numeric field {key}")
    return value


def _integer(record: dict[str, str] | None, key: str) -> int:
    value = _numeric(record, key)
    integer = int(value)
    if value != integer:
        raise ValueError(f"field {key} is not integral")
    return integer


def _boolean(record: dict[str, str] | None, key: str) -> bool:
    if record is None or record.get(key) not in {"true", "false"}:
        raise ValueError(f"missing Boolean field {key}")
    return record[key] == "true"


def evaluate_case(
    case: dict[str, Any],
    matrix: dict[str, Any],
    parsed: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    thresholds = matrix["thresholds"]
    geometry = analytic_planar_geometry(case)
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

    def number(
        record_name: str,
        key: str,
        check_name: str | None = None,
    ) -> float:
        label = check_name or key
        try:
            value = _numeric(parsed.get(record_name), key)
        except (TypeError, ValueError) as error:
            check(label, False, str(error), "finite numeric field")
            return math.nan
        metrics[label] = value
        return value

    def near_zero(record_name: str, key: str, label: str | None = None) -> None:
        name = label or key
        value = number(record_name, key, name)
        check(
            name,
            math.isfinite(value) and abs(value) <= thresholds["absolute_zero"],
            value,
            f"absolute <= {thresholds['absolute_zero']}",
        )

    def boolean_true(record_name: str, key: str, label: str | None = None) -> None:
        name = label or key
        try:
            value = _boolean(parsed.get(record_name), key)
        except (TypeError, ValueError) as error:
            check(name, False, str(error), True)
            return
        metrics[name] = value
        check(name, value, value, True)

    check("process_return_code", return_code == 0, return_code, 0)
    check(
        "accepted_step",
        parsed.get("accepted_step_count", 0) >= 1,
        parsed.get("accepted_step_count", 0),
        "at least one rank-zero accepted-step record",
    )

    expected_initializer_dofs = matrix["mesh"][
        "expected_phase_initializer_dofs"
    ]
    expected_ranks = set(range(case["mpi_ranks"]))
    for phase in ("negative", "positive"):
        name = f"{phase}_phase_initializer"
        records = [
            record
            for record in parsed.get("initializers", [])
            if (
                record.get("velocity_field"),
                record.get("pressure_field"),
            )
            == (f"u_{phase}", f"p_{phase}")
        ]
        initialized_by_rank: dict[int, int] = {}
        duplicate_rank = False
        for record in records:
            try:
                rank = _integer(record, "log_rank")
                initialized = _integer(record, "initialized_dofs")
            except (TypeError, ValueError):
                continue
            duplicate_rank = duplicate_rank or rank in initialized_by_rank
            initialized_by_rank[rank] = initialized
        initializer_passed = (
            not duplicate_rank
            and set(initialized_by_rank) == expected_ranks
            and all(
                initialized == expected_initializer_dofs
                for initialized in initialized_by_rank.values()
            )
        )
        check(
            name,
            initializer_passed,
            {
                "initialized_dofs_by_rank": initialized_by_rank,
                "duplicate_rank": duplicate_rank,
            },
            {
                "initialized_dofs_by_rank": {
                    rank: expected_initializer_dofs
                    for rank in sorted(expected_ranks)
                },
                "duplicate_rank": False,
            },
        )

    interface = "interface"
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
        "traction_jump_sq",
        "traction_jump_normal_integral",
        "surface_energy_work",
        "nitsche_consistency_work",
        "nitsche_adjoint_work",
        "nitsche_penalty_work",
        "negative_momentum_0",
        "negative_momentum_1",
        "negative_momentum_2",
        "positive_momentum_0",
        "positive_momentum_1",
        "positive_momentum_2",
        "negative_kinetic_energy",
        "positive_kinetic_energy",
        "negative_momentum_delta_norm",
        "positive_momentum_delta_norm",
    ):
        near_zero(interface, key)

    boolean_true(interface, "momentum_reconciliation_applicable")
    boolean_true(interface, "momentum_reconciliation_satisfied")
    boolean_true(interface, "accepted_stage_numerics_applicable")
    boolean_true(interface, "nonlinear_converged")
    boolean_true(interface, "linear_converged")
    try:
        velocity_update = _boolean(parsed.get(interface), "velocity_update_applied")
    except (TypeError, ValueError) as error:
        check("velocity_update_applied", False, str(error), False)
    else:
        metrics["velocity_update_applied"] = velocity_update
        check("velocity_update_applied", not velocity_update, velocity_update, False)

    for key, threshold_key in (
        ("interface_quadrature_points", "minimum_interface_quadrature_points"),
        ("negative_phase_quadrature_points", "minimum_phase_quadrature_points"),
        ("positive_phase_quadrature_points", "minimum_phase_quadrature_points"),
    ):
        try:
            value = _integer(parsed.get(interface), key)
        except (TypeError, ValueError) as error:
            check(key, False, str(error), f">= {thresholds[threshold_key]}")
        else:
            metrics[key] = value
            check(key, value >= thresholds[threshold_key], value, f">= {thresholds[threshold_key]}")

    for key, threshold_key in (
        ("nonlinear_iterations", "maximum_nonlinear_iterations"),
        ("linear_iterations", "maximum_linear_iterations"),
    ):
        try:
            value = _integer(parsed.get(interface), key)
        except (TypeError, ValueError) as error:
            check(key, False, str(error), thresholds[threshold_key])
        else:
            metrics[key] = value
            check(key, value <= thresholds[threshold_key], value, f"<= {thresholds[threshold_key]}")

    interface_measure = number(interface, "interface_measure")
    check(
        "interface_measure",
        math.isfinite(interface_measure)
        and abs(interface_measure - geometry["interface_measure"])
        <= thresholds["interface_measure_absolute"],
        interface_measure,
        geometry["interface_measure"],
    )
    for phase in ("negative", "positive"):
        expected_volume = geometry[f"{phase}_volume"]
        volume = number(interface, f"{phase}_volume")
        check(
            f"{phase}_volume",
            math.isfinite(volume)
            and abs(volume - expected_volume)
            <= thresholds["phase_volume_absolute"],
            volume,
            expected_volume,
        )
        density = number(interface, f"{phase}_density")
        expected_density = float(case[f"{phase}_density"])
        check(f"{phase}_density", density == expected_density, density, expected_density)
        mass = number(interface, f"{phase}_mass")
        expected_mass = expected_density * expected_volume
        mass_tolerance = thresholds["phase_mass_relative"] * max(1.0, abs(expected_mass))
        check(
            f"{phase}_mass",
            math.isfinite(mass) and abs(mass - expected_mass) <= mass_tolerance,
            mass,
            expected_mass,
        )
    partition = metrics.get("negative_volume", math.nan) + metrics.get(
        "positive_volume", math.nan
    )
    check(
        "phase_volume_partition",
        math.isfinite(partition)
        and abs(partition - 1.0) <= thresholds["phase_partition_absolute"],
        partition,
        1.0,
    )

    for key in (
        "boundary_transfer",
        "divergence_source",
        "global_balance_residual",
        "max_local_balance_residual",
        "max_component_balance_residual",
        "courant",
    ):
        near_zero("phase_stage", key, f"phase_stage_{key}")
    try:
        limited_edges = _integer(parsed.get("phase_stage"), "limited_edges")
    except (TypeError, ValueError) as error:
        check("phase_stage_limited_edges", False, str(error), 0)
    else:
        metrics["phase_stage_limited_edges"] = limited_edges
        check("phase_stage_limited_edges", limited_edges == 0, limited_edges, 0)

    for key in (
        "previous_measure",
        "accepted_measure",
    ):
        value = number("phase_stage", key, f"phase_stage_{key}")
        check(
            f"phase_stage_{key}",
            math.isfinite(value)
            and abs(value - geometry["negative_volume"])
            <= thresholds["phase_measure_absolute"],
            value,
            geometry["negative_volume"],
        )
    for key in ("measure_mismatch", "max_nodal_moment_mismatch", "nodal_moment_residual_norm", "interface_displacement_bound"):
        near_zero("phase_geometry", key, f"phase_geometry_{key}")
    try:
        reconciliation_iterations = _integer(
            parsed.get("phase_geometry"), "reconciliation_iterations"
        )
    except (TypeError, ValueError) as error:
        check("phase_geometry_reconciliation_iterations", False, str(error), 0)
    else:
        metrics["phase_geometry_reconciliation_iterations"] = reconciliation_iterations
        check("phase_geometry_reconciliation_iterations", reconciliation_iterations == 0, reconciliation_iterations, 0)
    for key in ("phase_measure", "retained_geometry_measure"):
        value = number("phase_geometry", key, f"phase_geometry_{key}")
        check(
            f"phase_geometry_{key}",
            math.isfinite(value)
            and abs(value - geometry["negative_volume"])
            <= thresholds["phase_measure_absolute"],
            value,
            geometry["negative_volume"],
        )

    for key in (
        "total_physical_boundary_mass_transfer",
        "transport_max_component_balance_residual",
        "reconciliation_interface_displacement_bound",
    ):
        near_zero("maintenance", key, f"maintenance_{key}")
    for key in (
        "transport_component_balance_satisfied",
        "transport_component_measure_closure_satisfied",
    ):
        boolean_true("maintenance", key, f"maintenance_{key}")
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
        check(
            f"maintenance_{key}",
            math.isfinite(value)
            and abs(value - geometry["negative_volume"])
            <= thresholds["phase_measure_absolute"],
            value,
            geometry["negative_volume"],
        )

    artifact = parsed.get("artifact")
    if artifact is not None:
        check(
            "phase_flux_artifact_written",
            artifact.get("outcome") == "written",
            artifact.get("outcome"),
            "written",
        )

    failed = [entry["name"] for entry in checks if not entry["passed"]]
    return {
        "case_id": case["case_id"],
        "reversal_pair": case["reversal_pair"],
        "level_set_sign": case["level_set_sign"],
        "mpi_ranks": case["mpi_ranks"],
        "passed": not failed,
        "failed_checks": failed,
        "metrics": metrics,
        "checks": checks,
    }


def evaluate_reversal_pairs(
    results: list[dict[str, Any]], matrix: dict[str, Any]
) -> dict[str, Any]:
    tolerance = matrix["thresholds"]["side_reversal_absolute"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["reversal_pair"], []).append(result)
    checks: list[dict[str, Any]] = []

    def compare(pair: str, name: str, first: float, second: float) -> None:
        passed = (
            math.isfinite(first)
            and math.isfinite(second)
            and abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))
        )
        checks.append(
            {
                "name": f"{pair}:{name}",
                "passed": passed,
                "first": first,
                "second": second,
                "tolerance": tolerance,
            }
        )

    expected_pairs = {case["reversal_pair"] for case in matrix["cases"]}
    for pair in sorted(expected_pairs):
        members = grouped.get(pair, [])
        if len(members) != 2 or {item["level_set_sign"] for item in members} != {-1, 1}:
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
        compare(pair, "negative_to_positive_volume", first["negative_volume"], second["positive_volume"])
        compare(pair, "positive_to_negative_volume", first["positive_volume"], second["negative_volume"])
        compare(pair, "negative_to_positive_mass", first["negative_mass"], second["positive_mass"])
        compare(pair, "positive_to_negative_mass", first["positive_mass"], second["negative_mass"])
    failed = [entry["name"] for entry in checks if not entry["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def validate_effective_configuration(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("modules"), list):
        raise ValueError("effective configuration has no module list")
    momentum = [
        module
        for module in document["modules"]
        if isinstance(module, dict) and module.get("component") == "incompressible_two_fluid"
    ]
    transport = [
        module
        for module in document["modules"]
        if isinstance(module, dict) and module.get("component") == "level_set_transport"
    ]
    if len(momentum) != 1 or len(transport) != 1:
        raise ValueError("effective configuration lacks a unique two-fluid pair")
    momentum_module = momentum[0]
    transport_module = transport[0]
    expected_fields = {
        "level_set": "level_set",
        "negative_velocity": "u_negative",
        "positive_velocity": "u_positive",
        "negative_pressure": "p_negative",
        "positive_pressure": "p_positive",
    }
    fields = momentum_module.get("fields")
    if not isinstance(fields, dict) or any(fields.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("effective configuration changed the two-fluid fields")
    solver = momentum_module.get("solver_contract")
    if not isinstance(solver, dict):
        raise ValueError("effective configuration lacks the solver contract")
    if solver.get("generic_fallback_allowed") is not False:
        raise ValueError("effective configuration enabled generic solver fallback")
    if solver.get("backend") != "FSILS" or solver.get("method") != "BlockSchur":
        raise ValueError("effective configuration changed the solver route")
    if momentum_module.get("capability_label") != (
        "incompressible_two_phase_sharp_interface_initial_envelope"
    ):
        raise ValueError("effective configuration changed momentum capability")
    conservative = transport_module.get("conservative_phase")
    if (
        transport_module.get("capability_label")
        != "two_phase_material_interface_transport"
        or not isinstance(conservative, dict)
        or conservative.get("enabled") is not True
        or conservative.get("boundary_flux_policy")
        != "closed_domain_discrete_q_flux_only"
    ):
        raise ValueError("effective configuration changed phase transport capability")
    return {
        "momentum_capability": momentum_module["capability_label"],
        "transport_capability": transport_module["capability_label"],
        "conservative_phase_enabled": True,
        "generic_solver_fallback": False,
    }


def write_text_create_only(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to replace artifact path: {path}")
    with temporary.open("x", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write_json_create_only(path: Path, value: Any) -> None:
    write_text_create_only(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


def qualification_output_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            relative = candidate.relative_to(path)
            if relative.parts and relative.parts[0] == "tmp":
                continue
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


def process_session_resident_kib(session_id: int) -> int:
    total = 0
    page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    for status_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = status_path.read_text(encoding="utf-8").split()
            if int(fields[5]) == session_id:
                total += int(fields[23]) * page_kib
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return total


def terminate_process_session(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_monitored(
    command: list[str],
    environment: dict[str, str],
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    wall_time_seconds: int,
    memory_mib: int,
    output_mib: int,
) -> dict[str, Any]:
    started = time.monotonic()
    peak_resident_kib = 0
    termination_reason: str | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        while process.poll() is None:
            resident = process_session_resident_kib(process.pid)
            peak_resident_kib = max(peak_resident_kib, resident)
            if resident > memory_mib * 1024:
                termination_reason = "memory_envelope_exceeded"
                terminate_process_session(process)
                break
            if time.monotonic() - started > wall_time_seconds:
                termination_reason = "wall_time_envelope_exceeded"
                terminate_process_session(process)
                break
            if qualification_output_size(working_directory) > output_mib * 1024 * 1024:
                termination_reason = "output_envelope_exceeded"
                terminate_process_session(process)
                break
            time.sleep(0.1)
        return_code = process.wait()
    return {
        "command": command,
        "return_code": return_code,
        "termination_reason": termination_reason,
        "wall_time_seconds": time.monotonic() - started,
        "peak_session_resident_kib": peak_resident_kib,
        "final_output_bytes": qualification_output_size(working_directory),
    }


def git_command(source_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def committed_path_identity(source_root: Path, path: Path) -> dict[str, Any]:
    root = source_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return {
            "path": str(resolved),
            "repository_path": None,
            "tracked_in_head": False,
            "matches_head": False,
            "head_sha256": None,
            "working_sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    committed = git_command(root, ["show", f"HEAD:{relative}"])
    tracked = committed.returncode == 0
    head_sha256 = sha256_bytes(committed.stdout) if tracked else None
    working_sha256 = sha256_file(resolved) if resolved.is_file() else None
    return {
        "path": str(resolved),
        "repository_path": relative,
        "tracked_in_head": tracked,
        "matches_head": tracked and head_sha256 == working_sha256,
        "head_sha256": head_sha256,
        "working_sha256": working_sha256,
    }


def tracked_source_digest(source_root: Path) -> str:
    listed = git_command(source_root, ["ls-files", "-z", "--", "Code", "Documentation", "tests"])
    if listed.returncode != 0:
        raise RuntimeError("cannot enumerate tracked qualification sources")
    digest = hashlib.sha256()
    for raw_path in listed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"tracked qualification source is absent: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_provenance(source_root: Path) -> dict[str, Any]:
    revision = git_command(source_root, ["rev-parse", "HEAD"])
    status = git_command(source_root, ["status", "--porcelain=v1", "-z"])
    unstaged = git_command(source_root, ["diff", "--quiet", "--"])
    staged = git_command(source_root, ["diff", "--cached", "--quiet", "--"])
    if revision.returncode != 0 or status.returncode != 0:
        raise RuntimeError("cannot obtain source provenance")
    if unstaged.returncode not in {0, 1} or staged.returncode not in {0, 1}:
        raise RuntimeError("cannot determine tracked source cleanliness")
    entries = [
        entry.decode("utf-8", errors="surrogateescape")
        for entry in status.stdout.split(b"\0")
        if entry
    ]
    return {
        "revision": revision.stdout.decode("ascii").strip(),
        "tracked_clean": unstaged.returncode == 0 and staged.returncode == 0,
        "porcelain_entries": entries,
        "porcelain_sha256": sha256_bytes(status.stdout),
        "tracked_source_sha256": tracked_source_digest(source_root),
    }


def find_cmake_cache(binary: Path) -> Path | None:
    for directory in [binary.parent, *binary.parents]:
        candidate = directory / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def selected_cmake_cache(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    prefixes = (
        "CMAKE_BUILD_TYPE:",
        "CMAKE_CXX_COMPILER:",
        "CMAKE_CXX_COMPILER_ID:",
        "CMAKE_CXX_COMPILER_VERSION:",
        "CMAKE_CXX_FLAGS:",
        "FE_ENABLE_MPI:",
        "SV_USE_MPI:",
    )
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith(("#", "//")):
            continue
        left, value = line.split("=", 1)
        if left.startswith(prefixes):
            result[left] = value
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
    solver_path = case_directory / "solver.xml"
    write_text_create_only(mesh_path, render_mesh(matrix, case))
    write_text_create_only(solver_path, render_solver(matrix, case))
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
    execution = run_monitored(
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
        "mesh": sha256_file(mesh_path),
        "solver": sha256_file(solver_path),
    }
    effective_path = case_directory / "effective_configuration.json"
    try:
        effective = validate_effective_configuration(read_json_strict(effective_path))
    except (OSError, TypeError, ValueError) as error:
        effective = {"error": str(error)}
        result["checks"].append(
            {
                "name": "effective_configuration",
                "passed": False,
                "actual": str(error),
                "expected": "staged incompressible two-fluid boundary",
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
    write_json_create_only(case_directory / "result.json", result)
    return result


def write_checksums(output_directory: Path) -> None:
    checksum_path = output_directory / "checksums.txt"
    paths = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file()
        and path != checksum_path
        and not path.name.endswith(".tmp")
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_directory)}"
        for path in paths
    ]
    write_text_create_only(checksum_path, "\n".join(lines) + "\n")


def execution_outcome(
    numerical_passed: bool, qualification_eligible: bool
) -> dict[str, Any]:
    if not numerical_passed:
        return {
            "outcome": "FAIL",
            "constant_state_gate_passed": False,
            "exit_code": 1,
        }
    if qualification_eligible:
        return {
            "outcome": "PASS",
            "constant_state_gate_passed": True,
            "exit_code": 0,
        }
    return {
        "outcome": "DEVELOPMENT_PASS",
        "constant_state_gate_passed": False,
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
    provenance = source_provenance(REPOSITORY_ROOT)
    input_identity = {
        "matrix": committed_path_identity(REPOSITORY_ROOT, matrix_path),
        "runner": committed_path_identity(REPOSITORY_ROOT, SCRIPT_PATH),
    }
    qualification_eligible = provenance["tracked_clean"] and all(
        item["matches_head"] for item in input_identity.values()
    )
    if not qualification_eligible and not allow_tracked_dirty_development:
        raise RuntimeError("tracked source is dirty; qualification execution refused")
    cache_path = find_cmake_cache(solver)
    preflight = {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": sha256_file(matrix_path),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "qualification_input_identity": input_identity,
        "source": provenance,
        "solver": {
            "path": str(solver),
            "sha256": sha256_file(solver),
            "cmake_cache_path": str(cache_path) if cache_path else None,
            "cmake_cache_sha256": sha256_file(cache_path) if cache_path else None,
            "selected_cmake_cache": selected_cmake_cache(cache_path),
        },
        "launcher": {
            "path": str(launcher),
            "sha256": sha256_file(launcher),
        },
        "qualification_eligible": qualification_eligible,
        "development_dirty_override": allow_tracked_dirty_development,
    }
    write_json_create_only(output_directory / "preflight.json", preflight)
    results: list[dict[str, Any]] = []
    for case in matrix["cases"]:
        results.append(run_case(case, matrix, solver, launcher, output_directory))
    reversal = evaluate_reversal_pairs(results, matrix)
    all_cases_passed = all(result["passed"] for result in results)
    disposition = execution_outcome(
        all_cases_passed and reversal["passed"], qualification_eligible
    )
    gate_passed = disposition["constant_state_gate_passed"]
    summary = {
        "matrix_id": matrix["matrix_id"],
        "requested_claim": matrix["accepted_claim"],
        **disposition,
        "qualification_eligible": qualification_eligible,
        "constant_state_gate_passed": gate_passed,
        "case_count": len(results),
        "passed_case_count": sum(result["passed"] for result in results),
        "failed_case_ids": [result["case_id"] for result in results if not result["passed"]],
        "reversal_pairs": reversal,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
        "required_later_progression": matrix["required_later_progression"],
        "results": results,
    }
    write_json_create_only(output_directory / "summary.json", summary)
    record = (
        "# WP-10 two-fluid constant-state qualification record\n\n"
        f"- Matrix: `{matrix['matrix_id']}`\n"
        f"- Source revision: `{provenance['revision']}`\n"
        f"- Tracked source clean: `{str(provenance['tracked_clean']).lower()}`\n"
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`\n"
        f"- Reversal pairs passed: `{str(reversal['passed']).lower()}`\n"
        f"- Constant-state gate passed: `{str(gate_passed).lower()}`\n"
        "- FSR-08, WP-10, and Q7 remain open.\n"
    )
    write_text_create_only(output_directory / "record.md", record)
    write_checksums(output_directory)
    return summary


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--requested-claim", default="constant_state_prerequisite"
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
        if any(value is not None for value in (options.solver, options.launcher, options.output_dir)):
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
        if any(value is not None for value in (options.solver, options.launcher, options.output_dir)):
            raise ValueError("--list-cases does not accept execution paths")
        for case in matrix["cases"]:
            print(
                f"{case['case_id']} ranks={case['mpi_ranks']} "
                f"orientation={case['orientation']} sign={case['level_set_sign']}"
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
