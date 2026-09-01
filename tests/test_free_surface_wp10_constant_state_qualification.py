import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp10_constant_state_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_constant_state_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_constant_state_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def passing_output() -> str:
    return "\n".join(
        [
            (
                "[R0] mesh-field initialization "
                "diagnostic=mesh_field_initialization initialized_dofs=75 "
                "velocity_field='u_negative' pressure_field='p_negative'"
            ),
            (
                "[R0] mesh-field initialization "
                "diagnostic=mesh_field_initialization initialized_dofs=75 "
                "velocity_field='u_positive' pressure_field='p_positive'"
            ),
            (
                "Conservative phase staged field='phase' step=1 "
                "previous_measure=3.7000000000000000e-01 "
                "accepted_measure=3.7000000000000000e-01 "
                "boundary_transfer=0 divergence_source=0 "
                "global_balance_residual=0 max_local_balance_residual=0 "
                "max_component_balance_residual=0 limited_edges=0 courant=0"
            ),
            (
                "Conservative phase geometry validated field='phase' step=1 "
                "phase_measure=3.7000000000000000e-01 "
                "retained_geometry_measure=3.7000000000000000e-01 "
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
                "interface_quadrature_points=8 interface_measure=1 "
                "velocity_jump_sq=0 velocity_jump_normal_sq=0 "
                "velocity_jump_tangential_sq=0 negative_normal_flux=0 "
                "positive_normal_flux=0 normal_flux_jump=0 "
                "negative_mass_flux=0 positive_mass_flux=0 "
                "mean_pressure_jump=0 pressure_jump_sq=0 "
                "pressure_jump_integral=0 traction_jump_sq=0 "
                "traction_jump_normal_integral=0 surface_energy_work=0 "
                "nitsche_consistency_work=0 nitsche_adjoint_work=0 "
                "nitsche_penalty_work=0 negative_phase_quadrature_points=34 "
                "negative_density=1000 negative_volume=0.37 negative_mass=370 "
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


def test_matrix_is_frozen_as_a_nonclosure_progression_gate():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    assert matrix["matrix_id"] == "free_surface_wp10_constant_state_v2"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["accepted_claim"] == "constant_state_prerequisite"
    assert matrix["mesh"]["ghost_layers"] == 8
    assert matrix["qualification_disposition"] == {
        "constant_state_gate_passed": False,
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
    assert max(
        max(case["negative_density"], case["positive_density"])
        / min(case["negative_density"], case["positive_density"])
        for case in matrix["cases"]
    ) == 10000.0
    assert max(
        max(case["negative_viscosity"], case["positive_viscosity"])
        / min(case["negative_viscosity"], case["positive_viscosity"])
        for case in matrix["cases"]
    ) == 100.0


@pytest.mark.parametrize("claim", ["fsr08_closure", "wp10_closure", "q7_closure"])
def test_runner_rejects_premature_closure_claims(claim):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    with pytest.raises(ValueError, match="outside this progression gate"):
        runner.validate_requested_claim(matrix, claim)


def test_matrix_byte_drift_and_duplicate_keys_are_rejected(tmp_path):
    runner = load_runner()
    drifted = tmp_path / MATRIX_PATH.name
    drifted.write_text(MATRIX_PATH.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_matrix(drifted)

    duplicated = tmp_path / "duplicated.json"
    duplicated.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.read_json_strict(duplicated)


@pytest.mark.parametrize(
    ("orientation", "offset", "negative_volume", "interface_measure"),
    [
        ("x", 0.37, 0.37, 1.0),
        ("y", 0.43, 0.43, 1.0),
        ("x_plus_y", 0.79, 0.5 * 0.79**2, math.sqrt(2.0) * 0.79),
        (
            "x_plus_y",
            1.21,
            1.0 - 0.5 * (2.0 - 1.21) ** 2,
            math.sqrt(2.0) * (2.0 - 1.21),
        ),
        ("x_minus_y", -0.13, 0.5 * 0.87**2, math.sqrt(2.0) * 0.87),
    ],
)
def test_analytic_planar_geometry_is_exact(
    orientation, offset, negative_volume, interface_measure
):
    runner = load_runner()
    case = {"orientation": orientation, "offset": offset, "level_set_sign": 1}
    geometry = runner.analytic_planar_geometry(case)

    assert geometry["negative_volume"] == pytest.approx(negative_volume)
    assert geometry["positive_volume"] == pytest.approx(1.0 - negative_volume)
    assert geometry["interface_measure"] == pytest.approx(interface_measure)

    case["level_set_sign"] = -1
    reversed_geometry = runner.analytic_planar_geometry(case)
    assert reversed_geometry["negative_volume"] == pytest.approx(
        geometry["positive_volume"]
    )
    assert reversed_geometry["positive_volume"] == pytest.approx(
        geometry["negative_volume"]
    )
    assert reversed_geometry["interface_measure"] == pytest.approx(
        geometry["interface_measure"]
    )


def test_generated_inputs_bind_all_six_fields_and_two_fluid_model():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    mesh = runner.render_mesh(matrix, case)
    solver = runner.render_solver(matrix, case)

    assert mesh.count("<Piece NumberOfPoints=\"25\" NumberOfCells=\"32\">") == 1
    for name in (
        "level_set",
        "u_negative",
        "p_negative",
        "u_positive",
        "p_positive",
    ):
        assert f'Name="{name}"' in mesh
    assert "IncompressibleTwoFluid" in solver
    assert "material_interface_phase_pair" in solver
    assert "<Ghost_layers>8</Ghost_layers>" in solver
    assert f"<Negative_phase_density>{case['negative_density']:.17g}" in solver
    assert f"<Positive_phase_density>{case['positive_density']:.17g}" in solver
    assert "<Two_fluid_surface_tension>0" in solver
    assert "Conservative_phase_write_flux_artifacts" not in solver
    assert "Conservative_phase_flux_artifact_cadence_steps" not in solver


def test_parser_and_exact_constant_state_gate_accept_complete_output():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = dict(matrix["cases"][0])
    case.update(
        {
            "orientation": "x",
            "offset": 0.37,
            "level_set_sign": 1,
            "negative_density": 1000.0,
            "positive_density": 1.0,
        }
    )
    parsed = runner.parse_case_output(passing_output())
    result = runner.evaluate_case(case, matrix, parsed, return_code=0)

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["metrics"]["negative_volume"] == pytest.approx(0.37)
    assert result["metrics"]["positive_volume"] == pytest.approx(0.63)
    assert result["metrics"]["negative_mass"] == pytest.approx(370.0)
    assert result["metrics"]["nonlinear_iterations"] == 0
    assert result["metrics"]["linear_iterations"] == 0


def test_gate_rejects_nonzero_flux_and_missing_initializer():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = dict(matrix["cases"][0])
    case.update(
        {
            "orientation": "x",
            "offset": 0.37,
            "level_set_sign": 1,
            "negative_density": 1000.0,
            "positive_density": 1.0,
        }
    )
    broken = passing_output().replace("normal_flux_jump=0", "normal_flux_jump=1e-4")
    broken = broken.replace(
        "velocity_field='u_positive' pressure_field='p_positive'",
        "velocity_field='unrelated' pressure_field='unrelated'",
    )
    result = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(broken),
        return_code=0,
    )

    assert result["passed"] is False
    assert "normal_flux_jump" in result["failed_checks"]
    assert "positive_phase_initializer" in result["failed_checks"]


def test_two_rank_gate_requires_initializer_evidence_from_both_ranks():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = dict(matrix["cases"][0])
    case.update(
        {
            "mpi_ranks": 2,
            "negative_density": 1000.0,
            "positive_density": 1.0,
        }
    )

    missing_rank = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output()),
        return_code=0,
    )
    assert missing_rank["passed"] is False
    assert "negative_phase_initializer" in missing_rank["failed_checks"]
    assert "positive_phase_initializer" in missing_rank["failed_checks"]

    initializer_lines = passing_output().splitlines()[:2]
    complete_output = passing_output() + "\n" + "\n".join(
        line.replace("[R0]", "[R1]") for line in initializer_lines
    )
    complete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(complete_output),
        return_code=0,
    )
    assert complete["passed"] is True


def test_reversal_pair_gate_exchanges_phase_quantities():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    local_matrix = dict(matrix)
    local_matrix["cases"] = [
        {"reversal_pair": "pair"},
        {"reversal_pair": "pair"},
    ]
    forward = {
        "case_id": "forward",
        "reversal_pair": "pair",
        "level_set_sign": 1,
        "passed": True,
        "metrics": {
            "interface_measure": 1.0,
            "negative_volume": 0.37,
            "positive_volume": 0.63,
            "negative_mass": 370.0,
            "positive_mass": 0.63,
        },
    }
    reverse = {
        "case_id": "reverse",
        "reversal_pair": "pair",
        "level_set_sign": -1,
        "passed": True,
        "metrics": {
            "interface_measure": 1.0,
            "negative_volume": 0.63,
            "positive_volume": 0.37,
            "negative_mass": 0.63,
            "positive_mass": 370.0,
        },
    }

    accepted = runner.evaluate_reversal_pairs([forward, reverse], local_matrix)
    assert accepted["passed"] is True
    assert accepted["failed_checks"] == []

    reverse["metrics"]["negative_volume"] += 1.0e-4
    rejected = runner.evaluate_reversal_pairs([forward, reverse], local_matrix)
    assert rejected["passed"] is False
    assert "pair:positive_to_negative_volume" in rejected["failed_checks"]


def test_effective_configuration_must_retain_the_staged_model_boundary():
    runner = load_runner()
    document = {
        "artifact_schema_version": 1,
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
        ],
    }
    assert runner.validate_effective_configuration(document) == {
        "momentum_capability": (
            "incompressible_two_phase_sharp_interface_initial_envelope"
        ),
        "transport_capability": "two_phase_material_interface_transport",
        "conservative_phase_enabled": True,
        "generic_solver_fallback": False,
    }

    document["modules"][0]["solver_contract"]["generic_fallback_allowed"] = True
    with pytest.raises(ValueError, match="generic solver fallback"):
        runner.validate_effective_configuration(document)


def test_qualification_input_identity_requires_exact_head_bytes(tmp_path):
    runner = load_runner()
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    tracked = repository / "tracked.json"
    tracked.write_text('{"version": 1}\n', encoding="utf-8")
    subprocess.run(["git", "add", "tracked.json"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Zachary Sexton",
            "-c",
            "user.email=zsexton@stanford.edu",
            "commit",
            "-q",
            "-m",
            "Freeze fixture",
        ],
        cwd=repository,
        check=True,
    )

    identity = runner.committed_path_identity(repository, tracked)
    assert identity["tracked_in_head"] is True
    assert identity["matches_head"] is True
    assert identity["head_sha256"] == identity["working_sha256"]

    tracked.write_text('{"version": 2}\n', encoding="utf-8")
    changed = runner.committed_path_identity(repository, tracked)
    assert changed["tracked_in_head"] is True
    assert changed["matches_head"] is False

    untracked = repository / "untracked.json"
    untracked.write_text("{}\n", encoding="utf-8")
    absent = runner.committed_path_identity(repository, untracked)
    assert absent["tracked_in_head"] is False
    assert absent["matches_head"] is False
    assert absent["head_sha256"] is None


def test_development_outcome_cannot_set_the_qualification_gate():
    runner = load_runner()

    assert runner.execution_outcome(True, True) == {
        "outcome": "PASS",
        "constant_state_gate_passed": True,
        "exit_code": 0,
    }
    assert runner.execution_outcome(True, False) == {
        "outcome": "DEVELOPMENT_PASS",
        "constant_state_gate_passed": False,
        "exit_code": 0,
    }
    assert runner.execution_outcome(False, True) == {
        "outcome": "FAIL",
        "constant_state_gate_passed": False,
        "exit_code": 1,
    }


def test_output_budget_excludes_runtime_temporary_storage(tmp_path):
    runner = load_runner()
    case_directory = tmp_path / "case"
    runtime_directory = case_directory / "tmp" / "launcher"
    runtime_directory.mkdir(parents=True)
    (case_directory / "stdout.log").write_bytes(b"a" * 17)
    (case_directory / "solver.xml").write_bytes(b"b" * 13)
    (runtime_directory / "shared-memory-backing").write_bytes(b"c" * 4096)

    assert runner.qualification_output_size(case_directory) == 30
