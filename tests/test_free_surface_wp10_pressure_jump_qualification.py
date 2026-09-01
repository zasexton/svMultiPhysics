import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp10_pressure_jump_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_pressure_jump_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_pressure_jump_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def passing_output(target=3.0, ranks=(0,)):
    initializer_lines = []
    for rank in ranks:
        initializer_lines.extend(
            [
                (
                    f"[R{rank}] mesh-field initialization "
                    "diagnostic=mesh_field_initialization initialized_dofs=75 "
                    "velocity_field='u_negative' pressure_field='p_negative'"
                ),
                (
                    f"[R{rank}] mesh-field initialization "
                    "diagnostic=mesh_field_initialization initialized_dofs=75 "
                    "velocity_field='u_positive' pressure_field='p_positive'"
                ),
            ]
        )
    interface_measure = 1.0
    return "\n".join(
        initializer_lines
        + [
            (
                "Conservative phase staged field='phase' step=1 "
                "previous_measure=0.37 accepted_measure=0.37 "
                "boundary_transfer=0 divergence_source=0 "
                "global_balance_residual=0 max_local_balance_residual=0 "
                "max_component_balance_residual=0 limited_edges=0 courant=0"
            ),
            (
                "Conservative phase geometry validated field='phase' step=1 "
                "phase_measure=0.37 retained_geometry_measure=0.37 "
                "measure_mismatch=0 max_nodal_moment_mismatch=0 "
                "nodal_moment_residual_norm=0 reconciliation_iterations=0 "
                "interface_displacement_bound=0"
            ),
            (
                "Conservative phase maintenance ledger "
                "diagnostic=conservative_phase_maintenance_ledger field='phase' "
                "step=1 raw_post_transport_phase_measure=0.37 "
                "post_limit_phase_measure=0.37 "
                "raw_post_transport_geometry_measure=0.37 "
                "post_reinitialization_phase_measure=0.37 "
                "post_reinitialization_geometry_measure=0.37 "
                "post_correction_phase_measure=0.37 "
                "post_correction_geometry_measure=0.37 "
                "retained_assembly_measure=0.37 "
                "total_physical_boundary_mass_transfer=0 "
                "transport_component_balance_satisfied=true "
                "transport_component_measure_closure_satisfied=true "
                "transport_max_component_balance_residual=0 "
                "reconciliation_interface_displacement_bound=0"
            ),
            (
                "[R0] accepted two-fluid interface diagnostics "
                "semantics=operator_stage accepted_step=1 interface_marker=71 "
                f"interface_quadrature_points=8 interface_measure={interface_measure} "
                "velocity_jump_sq=0 velocity_jump_normal_sq=0 "
                "velocity_jump_tangential_sq=0 negative_normal_flux=0 "
                "positive_normal_flux=0 normal_flux_jump=0 "
                "negative_mass_flux=0 positive_mass_flux=0 "
                f"mean_pressure_jump={target} "
                f"pressure_jump_sq={target * target * interface_measure} "
                f"pressure_jump_integral={target * interface_measure} "
                f"traction_jump_sq={target * target * interface_measure} "
                f"traction_jump_normal_integral={-target * interface_measure} "
                "prescribed_pressure_jump_applicable=true "
                f"prescribed_pressure_jump_target={target} "
                "prescribed_pressure_jump_error_sq=0 "
                "prescribed_stress_jump_residual_sq=0 "
                "surface_energy_work=0 nitsche_consistency_work=0 "
                "nitsche_adjoint_work=0 nitsche_penalty_work=0 "
                "negative_phase_quadrature_points=34 negative_density=1000 "
                "negative_volume=0.37 negative_mass=370 "
                "negative_momentum_0=0 negative_momentum_1=0 "
                "negative_momentum_2=0 negative_kinetic_energy=0 "
                "positive_phase_quadrature_points=34 positive_density=1 "
                "positive_volume=0.63 positive_mass=0.63 "
                "positive_momentum_0=0 positive_momentum_1=0 "
                "positive_momentum_2=0 positive_kinetic_energy=0 "
                "momentum_reconciliation_applicable=true "
                "velocity_update_applied=false "
                "negative_momentum_delta_norm=0 positive_momentum_delta_norm=0 "
                "momentum_reconciliation_satisfied=true "
                "accepted_stage_numerics_applicable=true "
                "nonlinear_converged=true nonlinear_iterations=0 "
                "linear_converged=true linear_iterations=0 "
                "phase_iteration_scope=shared_coupled_solve"
            ),
            "TimeLoop: step_accepted step=1 time=0.01 dt=0.01",
        ]
    )


def effective_configuration(target=3.0):
    return {
        "modules": [
            {
                "component": "incompressible_two_fluid",
                "capability_label": (
                    "incompressible_two_phase_sharp_interface_initial_envelope"
                ),
                "fields": {
                    "level_set": "level_set",
                    "negative_velocity": "u_negative",
                    "positive_velocity": "u_positive",
                    "negative_pressure": "p_negative",
                    "positive_pressure": "p_positive",
                },
                "interface": {
                    "prescribed_pressure_jump_applicable": True,
                    "prescribed_pressure_jump": target,
                },
                "pressure_space": {
                    "representation": "separate_phase_fields",
                    "shared_gauge_count": 1,
                },
                "solver_contract": {
                    "backend": "FSILS",
                    "method": "BlockSchur",
                    "generic_fallback_allowed": False,
                },
            },
            {
                "component": "level_set_transport",
                "capability_label": "two_phase_material_interface_transport",
                "conservative_phase": {
                    "enabled": True,
                    "boundary_flux_policy": "closed_domain_discrete_q_flux_only",
                },
            },
        ]
    }


def test_matrix_freezes_a_nonclosure_planar_jump_gate():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    assert matrix["matrix_id"] == "free_surface_wp10_pressure_jump_v1"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["accepted_claim"] == "planar_pressure_jump_prerequisite"
    assert matrix["mesh"]["ghost_layers"] == 8
    assert matrix["mesh"]["expected_wall_vertices"] == 16
    assert matrix["qualification_disposition"] == {
        "planar_pressure_jump_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }
    assert len(matrix["cases"]) == 12
    assert {case["mpi_ranks"] for case in matrix["cases"]} == {1, 2}
    assert {case["orientation"] for case in matrix["cases"]} == {
        "x",
        "y",
        "x_plus_y",
        "x_minus_y",
    }
    assert {math.copysign(1.0, case["pressure_jump"]) for case in matrix["cases"]} == {
        -1.0,
        1.0,
    }
    assert max(
        max(case["negative_density"], case["positive_density"])
        / min(case["negative_density"], case["positive_density"])
        for case in matrix["cases"]
    ) == 10000.0


@pytest.mark.parametrize("claim", ["fsr08_closure", "wp10_closure", "q7_closure"])
def test_runner_rejects_premature_closure_claims(claim):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    with pytest.raises(ValueError, match="outside this progression gate"):
        runner.validate_requested_claim(matrix, claim)


def test_generated_inputs_bind_wall_gauge_and_jump_contracts():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    mesh = ET.fromstring(runner.render_mesh(matrix, case))
    wall = ET.fromstring(runner.render_wall(matrix))
    solver = ET.fromstring(runner.render_solver(matrix, case))

    arrays = {
        item.attrib.get("Name"): item
        for item in mesh.findall(".//PointData/DataArray")
    }
    negative_pressure = [float(value) for value in arrays["p_negative"].text.split()]
    positive_pressure = [float(value) for value in arrays["p_positive"].text.split()]
    assert negative_pressure == [0.0] * 25
    assert positive_pressure == [-case["pressure_jump"]] * 25

    piece = wall.find(".//Piece")
    assert piece is not None
    assert piece.attrib["NumberOfPoints"] == "16"
    assert piece.attrib["NumberOfLines"] == "16"
    wall_ids = wall.find(".//PointData/DataArray[@Name='GlobalNodeID']")
    assert wall_ids is not None
    assert len({int(value) for value in wall_ids.text.split()}) == 16

    face = solver.find(".//Add_mesh/Add_face[@name='wall']")
    assert face is not None
    assert face.findtext("Face_file_path") == "wall.vtp"
    fluid = solver.find(".//Add_equation[@type='fluid']")
    assert fluid is not None
    assert float(fluid.findtext("Prescribed_pressure_jump")) == case["pressure_jump"]
    boundary = fluid.find("Add_BC[@name='wall']")
    assert boundary is not None
    assert boundary.findtext("Type") == "Dir"
    assert float(boundary.findtext("Value")) == 0.0


def test_wall_is_one_closed_cycle_over_every_boundary_vertex():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    wall = ET.fromstring(runner.render_wall(matrix))

    ids = [
        int(value)
        for value in wall.find(
            ".//PointData/DataArray[@Name='GlobalNodeID']"
        ).text.split()
    ]
    assert set(ids) == {
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        15,
        16,
        20,
        21,
        22,
        23,
        24,
        25,
    }
    connectivity = [
        int(value)
        for value in wall.find(
            ".//Lines/DataArray[@Name='connectivity']"
        ).text.split()
    ]
    assert connectivity == [
        value
        for line in range(16)
        for value in (line, (line + 1) % 16)
    ]


def test_effective_configuration_requires_exact_jump_gauge_and_solver_route():
    runner = load_runner()
    validated = runner.validate_effective_configuration(
        effective_configuration(), 3.0
    )
    assert validated["prescribed_pressure_jump"] == 3.0
    assert validated["shared_pressure_gauge_count"] == 1
    assert validated["generic_solver_fallback"] is False

    wrong_jump = effective_configuration(2.0)
    with pytest.raises(ValueError, match="changed the prescribed jump"):
        runner.validate_effective_configuration(wrong_jump, 3.0)

    missing_gauge = effective_configuration()
    missing_gauge["modules"][0]["pressure_space"]["shared_gauge_count"] = 0
    with pytest.raises(ValueError, match="shared pressure gauge"):
        runner.validate_effective_configuration(missing_gauge, 3.0)

    fallback = effective_configuration()
    fallback["modules"][0]["solver_contract"][
        "generic_fallback_allowed"
    ] = True
    with pytest.raises(ValueError, match="generic solver fallback"):
        runner.validate_effective_configuration(fallback, 3.0)


def test_frozen_matrix_and_common_runner_dependencies_are_byte_pinned(tmp_path):
    runner = load_runner()
    assert runner.COMMON.sha256_file(MATRIX_PATH) == runner.EXPECTED_MATRIX_SHA256
    assert (
        runner.COMMON.sha256_file(runner.COMMON_RUNNER_PATH)
        == runner.EXPECTED_COMMON_RUNNER_SHA256
    )

    changed = tmp_path / MATRIX_PATH.name
    changed.write_bytes(MATRIX_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_matrix(changed)


def test_exact_planar_jump_output_passes_all_numeric_gates():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = dict(matrix["cases"][0])
    case.update(
        {
            "orientation": "x",
            "offset": 0.37,
            "level_set_sign": 1,
            "pressure_jump": 3.0,
            "negative_density": 1000.0,
            "positive_density": 1.0,
            "mpi_ranks": 1,
        }
    )
    parsed = runner.parse_case_output(passing_output())
    result = runner.evaluate_case(case, matrix, parsed, return_code=0)

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["metrics"]["mean_pressure_jump"] == pytest.approx(3.0)
    assert result["metrics"]["pressure_jump_integral"] == pytest.approx(3.0)
    assert result["metrics"]["traction_jump_normal_integral"] == pytest.approx(-3.0)
    assert result["metrics"]["nonlinear_iterations"] == 0
    assert result["metrics"]["linear_iterations"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prescribed_pressure_jump_error_sq=0", "prescribed_pressure_jump_error_sq=1e-5"),
        ("prescribed_stress_jump_residual_sq=0", "prescribed_stress_jump_residual_sq=1e-5"),
        ("mean_pressure_jump=3.0", "mean_pressure_jump=2.9"),
    ],
)
def test_gate_rejects_pressure_or_stress_jump_error(field, replacement):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = dict(matrix["cases"][0])
    case.update(
        {
            "orientation": "x",
            "offset": 0.37,
            "level_set_sign": 1,
            "pressure_jump": 3.0,
            "negative_density": 1000.0,
            "positive_density": 1.0,
            "mpi_ranks": 1,
        }
    )
    broken = passing_output().replace(field, replacement)
    result = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(broken),
        return_code=0,
    )

    assert result["passed"] is False


def test_two_rank_gate_requires_both_rank_initializers():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = dict(matrix["cases"][0])
    case.update(
        {
            "orientation": "x",
            "offset": 0.37,
            "level_set_sign": 1,
            "pressure_jump": 3.0,
            "negative_density": 1000.0,
            "positive_density": 1.0,
            "mpi_ranks": 2,
        }
    )

    incomplete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output(ranks=(0,))),
        return_code=0,
    )
    assert incomplete["passed"] is False
    assert "negative_phase_initializer" in incomplete["failed_checks"]
    assert "positive_phase_initializer" in incomplete["failed_checks"]

    complete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output(ranks=(0, 1))),
        return_code=0,
    )
    assert complete["passed"] is True


def test_validation_cli_reports_frozen_case_and_pair_counts():
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["matrix_id"] == "free_surface_wp10_pressure_jump_v1"
    assert record["case_count"] == 12
    assert record["reversal_pair_count"] == 6
    assert record["outcome"] == "PASS"
    assert record["planar_pressure_jump_gate_passed"] is False
