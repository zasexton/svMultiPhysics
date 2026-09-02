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
    / "run_free_surface_wp10_viscous_jump_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_viscous_jump_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_viscous_jump_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def passing_output(
    runner,
    matrix,
    case,
    ranks=(0,),
    courant=0.0,
    maximum_nodal_boundary_transfer=1.0e-5,
    boundary_mass_tolerance=1.0e-12,
    limited_edges=0,
    reconciliation_diagnostic="stationary_geometry_equilibrium_projection",
    reconciliation_initial_residual_norm=1.0e-6,
    nonlinear_initial_residual_norm=1.0e-12,
):
    expected = runner.expected_case_observables(case, matrix)
    geometry = runner.PRESSURE.COMMON.analytic_planar_geometry(case)
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

    def fields(names):
        return " ".join(f"{name}={expected[name]:.17g}" for name in names)

    vector_fields = [
        f"{phase}_{kind}traction_integral_{component}"
        for kind in ("", "viscous_")
        for phase in ("negative", "positive")
        for component in range(3)
    ]
    vector_fields.extend(
        f"{kind}traction_jump_integral_{component}"
        for kind in ("", "viscous_")
        for component in range(3)
    )
    vector_fields.extend(
        f"prescribed_viscous_traction_jump_target_{component}"
        for component in range(3)
    )
    bulk_fields = [
        f"{phase}_momentum_{component}"
        for phase in ("negative", "positive")
        for component in range(3)
    ]
    bulk_fields.extend(
        f"{phase}_kinetic_energy" for phase in ("negative", "positive")
    )
    negative_volume = geometry["negative_volume"]
    positive_volume = geometry["positive_volume"]
    nonlinear_solver = matrix.get(
        "nonlinear_solver",
        {"absolute_tolerance": 1.0e-10, "relative_tolerance": 0.0},
    )
    return "\n".join(
        initializer_lines
        + [
            (
                "[svMultiPhysics::Application] Transient solve: t0=0 "
                "dt=0.0001 t_end=0.0001 max_steps=1 "
                "scheme=BackwardEuler rho_inf=n/a pde_udot_init=0 "
                "last_step_absorb_fraction=0 "
                f"newton(max_it=25, min_it=0, "
                f"abs_tol={nonlinear_solver['absolute_tolerance']:.17g}, "
                f"rel_tol={nonlinear_solver['relative_tolerance']:.17g})"
            ),
            (
                "Conservative phase staged field='phase' step=1 "
                f"previous_measure={negative_volume:.17g} "
                f"accepted_measure={negative_volume:.17g} "
                "boundary_transfer=0 divergence_source=0 "
                "global_balance_residual=0 max_local_balance_residual=0 "
                "max_component_balance_residual=0 "
                f"limited_edges={limited_edges} "
                f"maximum_nodal_boundary_mass_transfer="
                f"{maximum_nodal_boundary_transfer:.17g} "
                f"boundary_mass_tolerance={boundary_mass_tolerance:.17g} "
                f"courant={courant:.17g}"
            ),
            (
                "Conservative phase geometry validated field='phase' step=1 "
                f"phase_measure={negative_volume:.17g} "
                f"retained_geometry_measure={negative_volume:.17g} "
                "measure_mismatch=0 max_nodal_moment_mismatch=0 "
                "nodal_moment_residual_norm=0 reconciliation_iterations=0 "
                f"reconciliation_initial_residual_norm="
                f"{reconciliation_initial_residual_norm:.17g} "
                "reconciliation_final_residual_norm=0 "
                f"reconciliation_diagnostic='{reconciliation_diagnostic}' "
                "interface_displacement_bound=0"
            ),
            (
                "Conservative phase maintenance ledger "
                "diagnostic=conservative_phase_maintenance_ledger field='phase' "
                "step=1 "
                f"raw_post_transport_phase_measure={negative_volume:.17g} "
                f"post_limit_phase_measure={negative_volume:.17g} "
                f"raw_post_transport_geometry_measure={negative_volume:.17g} "
                f"post_reinitialization_phase_measure={negative_volume:.17g} "
                f"post_reinitialization_geometry_measure={negative_volume:.17g} "
                f"post_correction_phase_measure={negative_volume:.17g} "
                f"post_correction_geometry_measure={negative_volume:.17g} "
                f"retained_assembly_measure={negative_volume:.17g} "
                "total_physical_boundary_mass_transfer=0 "
                "transport_component_balance_satisfied=true "
                "transport_component_measure_closure_satisfied=true "
                "transport_max_component_balance_residual=0 "
                "reconciliation_interface_displacement_bound=0 "
                f"reconciliation_initial_residual_norm="
                f"{reconciliation_initial_residual_norm:.17g} "
                "reconciliation_final_residual_norm=0 "
                f"reconciliation_diagnostic='{reconciliation_diagnostic}'"
            ),
            (
                "[R0] accepted two-fluid interface diagnostics "
                "semantics=operator_stage accepted_step=1 interface_marker=71 "
                f"interface_quadrature_points=8 "
                f"interface_measure={geometry['interface_measure']:.17g} "
                "velocity_jump_sq=0 velocity_jump_normal_sq=0 "
                "velocity_jump_tangential_sq=0 negative_normal_flux=0 "
                "positive_normal_flux=0 normal_flux_jump=0 "
                "negative_mass_flux=0 positive_mass_flux=0 "
                "mean_pressure_jump=0 pressure_jump_sq=0 "
                "pressure_jump_integral=0 traction_jump_normal_integral=0 "
                f"traction_jump_sq={expected['traction_jump_sq']:.17g} "
                + fields(vector_fields)
                + " "
                f"viscous_traction_jump_sq={expected['viscous_traction_jump_sq']:.17g} "
                "prescribed_pressure_jump_applicable=false "
                "prescribed_viscous_traction_jump_applicable=true "
                "prescribed_viscous_traction_jump_error_sq=0 "
                "prescribed_stress_jump_residual_applicable=true "
                "prescribed_stress_jump_residual_sq=0 "
                "surface_energy_work=0 nitsche_consistency_work=0 "
                "nitsche_adjoint_work=0 nitsche_penalty_work=0 "
                "negative_phase_quadrature_points=34 "
                f"negative_density={case['negative_density']:.17g} "
                f"negative_volume={negative_volume:.17g} "
                f"negative_mass={case['negative_density'] * negative_volume:.17g} "
                "positive_phase_quadrature_points=34 "
                f"positive_density={case['positive_density']:.17g} "
                f"positive_volume={positive_volume:.17g} "
                f"positive_mass={case['positive_density'] * positive_volume:.17g} "
                + fields(bulk_fields)
                + " "
                "momentum_reconciliation_applicable=true "
                "velocity_update_applied=false "
                "negative_momentum_delta_norm=0 positive_momentum_delta_norm=0 "
                "momentum_reconciliation_satisfied=true "
                "accepted_stage_numerics_applicable=true "
                "nonlinear_converged=true nonlinear_iterations=0 "
                f"nonlinear_initial_residual_norm="
                f"{nonlinear_initial_residual_norm:.17g} "
                f"nonlinear_final_residual_norm="
                f"{nonlinear_initial_residual_norm:.17g} "
                "linear_converged=true linear_iterations=0 "
                "phase_iteration_scope=shared_coupled_solve"
            ),
            (
                "TimeLoop: step_accepted step=1 "
                f"time={matrix['time']['dt']:.17g} "
                f"dt={matrix['time']['dt']:.17g}"
            ),
        ]
    )


def effective_configuration(case, runner):
    target = runner.traction_jump_target(case)
    return {
        "modules": [
            {
                "component": "incompressible_two_fluid",
                "capability_label": (
                    "incompressible_two_phase_sharp_interface_initial_envelope"
                ),
                "momentum_operator": "stokes",
                "fields": {
                    "level_set": "level_set",
                    "negative_velocity": "u_negative",
                    "positive_velocity": "u_positive",
                    "negative_pressure": "p_negative",
                    "positive_pressure": "p_positive",
                },
                "material": {
                    "negative_density": case["negative_density"],
                    "negative_viscosity": case["negative_viscosity"],
                    "positive_density": case["positive_density"],
                    "positive_viscosity": case["positive_viscosity"],
                },
                "interface": {
                    "prescribed_pressure_jump_applicable": False,
                    "prescribed_viscous_traction_jump_applicable": True,
                    "prescribed_viscous_traction_jump": list(target[:2]),
                },
                "boundary_conditions": {
                    "shared_velocity_dirichlet_count": 1,
                    "shared_velocity_dirichlet_policy": (
                        "identical_external_data_on_both_phase_restrictions"
                    ),
                    "shared_velocity_dirichlet": [
                        {
                            "marker": 1,
                            "active_components": [True, True],
                            "values": [
                                {"kind": "time_coefficient"},
                                {"kind": "time_coefficient"},
                            ],
                        }
                    ],
                    "negative_phase_local_velocity_dirichlet_count": 0,
                    "positive_phase_local_velocity_dirichlet_count": 0,
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
                    "boundary_flux_policy": (
                        "globally_balanced_discrete_q_flux"
                    ),
                },
            },
        ]
    }


def test_matrix_freezes_a_nonclosure_finite_step_viscous_gate():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    assert matrix["matrix_id"] == "free_surface_wp10_viscous_jump_v2"
    assert matrix["supersedes_matrix_id"] == (
        "free_surface_wp10_viscous_jump_v1"
    )
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["accepted_claim"] == "planar_viscous_traction_jump_prerequisite"
    assert "nonzero local exterior normal velocity" in matrix["scope"]
    assert matrix["model_envelope"]["momentum_operator"] == "two_fluid_stokes"
    assert matrix["model_envelope"]["exterior_flux_scope"] == (
        "nonzero_local_discrete_q_flux_with_zero_global_balance"
    )
    assert matrix["thresholds"]["maximum_finite_step_courant"] == 2.5e-3
    assert matrix["nonlinear_solver"] == {
        "control_source": "GeneralSimulationParameters",
        "absolute_tolerance": 5.0e-10,
        "relative_tolerance": 0.0,
        "entry_state_requirement": "zero_update_zero_iteration",
    }
    assert matrix["execution"]["solver_rank_trace"] == "root_only"
    assert matrix["time"] == {
        "steps": 1,
        "dt": 1e-4,
        "scheme": "BackwardEuler",
        "interpretation": "finite_low_courant_affine_tangential_step",
    }
    assert len(matrix["cases"]) == 12
    assert {case["mpi_ranks"] for case in matrix["cases"]} == {1, 2}
    assert {case["orientation"] for case in matrix["cases"]} == {
        "x",
        "y",
        "x_plus_y",
        "x_minus_y",
    }
    assert {math.copysign(1.0, case["shear_rate"]) for case in matrix["cases"]} == {
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


def test_clipped_polygon_moments_reproduce_exact_axis_aligned_integrals():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    moments = runner.analytic_phase_moments(case)

    assert moments["negative"]["area"] == pytest.approx(0.37)
    assert moments["negative"]["signed_distance"] == pytest.approx(-0.37**2 / 2)
    assert moments["negative"]["signed_distance_squared"] == pytest.approx(
        0.37**3 / 3
    )
    assert moments["positive"]["area"] == pytest.approx(0.63)
    assert moments["positive"]["signed_distance"] == pytest.approx(0.63**2 / 2)
    assert moments["positive"]["signed_distance_squared"] == pytest.approx(
        0.63**3 / 3
    )


def test_all_clipped_references_are_sign_correct_and_reversal_symmetric():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    by_pair = {}
    for case in matrix["cases"]:
        moments = runner.analytic_phase_moments(case)
        assert moments["negative"]["area"] > 0.0
        assert moments["positive"]["area"] > 0.0
        assert moments["negative"]["signed_distance"] < 0.0
        assert moments["positive"]["signed_distance"] > 0.0
        assert moments["negative"]["signed_distance_squared"] > 0.0
        assert moments["positive"]["signed_distance_squared"] > 0.0
        expected = runner.expected_case_observables(case, matrix)
        assert all(math.isfinite(value) for value in expected.values())
        assert expected["negative_kinetic_energy"] > 0.0
        assert expected["positive_kinetic_energy"] > 0.0
        assert expected["viscous_traction_jump_sq"] > 0.0
        by_pair.setdefault(case["reversal_pair"], []).append((case, expected))

    for members in by_pair.values():
        forward = next(item for item in members if item[0]["level_set_sign"] == 1)
        reverse = next(item for item in members if item[0]["level_set_sign"] == -1)
        for component in range(3):
            assert forward[1][f"negative_momentum_{component}"] == pytest.approx(
                reverse[1][f"positive_momentum_{component}"]
            )
            assert forward[1][f"positive_momentum_{component}"] == pytest.approx(
                reverse[1][f"negative_momentum_{component}"]
            )
            assert forward[1][
                f"viscous_traction_jump_integral_{component}"
            ] == pytest.approx(
                reverse[1][f"viscous_traction_jump_integral_{component}"]
            )
        assert forward[1]["negative_kinetic_energy"] == pytest.approx(
            reverse[1]["positive_kinetic_energy"]
        )
        assert forward[1]["positive_kinetic_energy"] == pytest.approx(
            reverse[1]["negative_kinetic_energy"]
        )


def test_generated_inputs_bind_affine_velocity_wall_and_viscous_target():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    mesh = ET.fromstring(runner.render_mesh(matrix, case))
    solver = ET.fromstring(runner.render_solver(matrix, case))
    velocity_data = runner.render_velocity_data(matrix, case).splitlines()

    arrays = {
        item.attrib.get("Name"): item
        for item in mesh.findall(".//PointData/DataArray")
    }
    assert arrays["u_negative"].text == arrays["u_positive"].text
    velocity_values = [float(value) for value in arrays["u_negative"].text.split()]
    assert velocity_values[:4] == pytest.approx([0.0, -1.48, 0.0, -0.48])
    assert velocity_data[:2] == ["2 1 16", "0"]
    assert [int(velocity_data[index]) for index in range(2, len(velocity_data), 2)] == [
        1,
        2,
        3,
        4,
        5,
        10,
        15,
        20,
        25,
        24,
        23,
        22,
        21,
        16,
        11,
        6,
    ]

    parameters = solver.find("GeneralSimulationParameters")
    assert parameters is not None
    assert parameters.findtext("Number_of_time_steps") == "1"
    assert float(parameters.findtext("Time_step_size")) == 1e-4
    assert float(parameters.findtext("Newton_absolute_tolerance")) == matrix[
        "nonlinear_solver"
    ]["absolute_tolerance"]
    assert float(parameters.findtext("Newton_relative_tolerance")) == matrix[
        "nonlinear_solver"
    ]["relative_tolerance"]
    level_set = solver.find("Add_equation[@type='level_set']")
    assert level_set is not None
    assert level_set.findtext("Conservative_phase_boundary_flux_policy") == (
        "globally_balanced_discrete_q_flux"
    )
    fluid = solver.find("Add_equation[@type='stokes']")
    assert fluid is not None
    target = runner.traction_jump_target(case)
    assert float(fluid.findtext("Prescribed_viscous_traction_jump_x")) == target[0]
    assert float(fluid.findtext("Prescribed_viscous_traction_jump_y")) == target[1]
    boundary = fluid.find("Add_BC[@name='wall']")
    assert boundary is not None
    assert boundary.findtext("Type") == "Dir"
    assert boundary.findtext("Time_dependence") == "General"
    assert boundary.findtext("Temporal_and_spatial_values_file_path") == "velocity.dat"


def test_effective_configuration_requires_viscous_target_and_shared_boundary():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    document = effective_configuration(case, runner)
    validated = runner.validate_effective_configuration(document, case)
    assert validated["prescribed_viscous_traction_jump"] == pytest.approx(
        runner.traction_jump_target(case)[:2]
    )
    assert validated["shared_velocity_dirichlet_count"] == 1
    assert validated["momentum_operator"] == "stokes"
    assert validated["phase_boundary_flux_policy"] == (
        "globally_balanced_discrete_q_flux"
    )

    wrong_target = effective_configuration(case, runner)
    wrong_target["modules"][0]["interface"][
        "prescribed_viscous_traction_jump"
    ][1] += 1.0
    with pytest.raises(ValueError, match="changed the viscous traction target"):
        runner.validate_effective_configuration(wrong_target, case)

    local_boundary = effective_configuration(case, runner)
    local_boundary["modules"][0]["boundary_conditions"][
        "negative_phase_local_velocity_dirichlet_count"
    ] = 1
    with pytest.raises(ValueError, match="shared velocity boundary"):
        runner.validate_effective_configuration(local_boundary, case)

    wrong_operator = effective_configuration(case, runner)
    wrong_operator["modules"][0]["momentum_operator"] = "navier_stokes"
    with pytest.raises(ValueError, match="momentum operator"):
        runner.validate_effective_configuration(wrong_operator, case)

    malformed_boundary = effective_configuration(case, runner)
    malformed_boundary["modules"][0]["boundary_conditions"][
        "shared_velocity_dirichlet"
    ] = ["invalid"]
    with pytest.raises(ValueError, match="shared velocity boundary"):
        runner.validate_effective_configuration(malformed_boundary, case)

    wrong_flux_policy = effective_configuration(case, runner)
    wrong_flux_policy["modules"][1]["conservative_phase"][
        "boundary_flux_policy"
    ] = "closed_domain_discrete_q_flux_only"
    with pytest.raises(ValueError, match="boundary-flux policy"):
        runner.validate_effective_configuration(wrong_flux_policy, case)


def test_frozen_matrix_and_runner_dependencies_are_byte_pinned(tmp_path):
    runner = load_runner()
    assert runner.COMMON.sha256_file(MATRIX_PATH) == runner.EXPECTED_MATRIX_SHA256
    assert (
        runner.COMMON.sha256_file(runner.PRESSURE_RUNNER_PATH)
        == runner.EXPECTED_PRESSURE_RUNNER_SHA256
    )
    changed = tmp_path / MATRIX_PATH.name
    changed.write_bytes(MATRIX_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_matrix(changed)


def test_exact_affine_output_passes_bulk_and_traction_gates():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    parsed = runner.parse_case_output(passing_output(runner, matrix, case))
    result = runner.evaluate_case(case, matrix, parsed, return_code=0)

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["metrics"]["negative_momentum_1"] == pytest.approx(-273.8)
    assert result["metrics"]["positive_momentum_1"] == pytest.approx(0.7938)
    assert result["metrics"]["negative_kinetic_energy"] == pytest.approx(
        135.0746666666667
    )
    assert result["metrics"]["viscous_traction_jump_integral_1"] == pytest.approx(
        0.06
    )
    assert result["metrics"]["traction_jump_sq"] == pytest.approx(0.0036)


def test_exact_entry_residual_is_bounded_by_declared_newton_tolerance():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    tolerance = matrix["nonlinear_solver"]["absolute_tolerance"]

    accepted = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(
                runner,
                matrix,
                case,
                nonlinear_initial_residual_norm=0.99 * tolerance,
            )
        ),
        return_code=0,
    )
    assert accepted["passed"] is True
    assert accepted["metrics"]["nonlinear_initial_residual_norm"] == pytest.approx(
        0.99 * tolerance
    )

    rejected = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(
                runner,
                matrix,
                case,
                nonlinear_initial_residual_norm=1.01 * tolerance,
            )
        ),
        return_code=0,
    )
    assert rejected["passed"] is False
    assert "nonlinear_initial_residual_norm" in rejected["failed_checks"]


def test_qualification_environment_disables_inherited_rank_trace():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    environment = runner.qualification_environment(
        matrix,
        {
            "SVMP_OOP_SOLVER_TRACE": "1",
            "SVMP_NEWTON_ABS_TOLERANCE": "9.9",
        },
        Path("/tmp/viscous-jump-case"),
    )

    assert environment["SVMP_OOP_SOLVER_TRACE"] == "0"
    assert "SVMP_NEWTON_ABS_TOLERANCE" not in environment
    assert environment["TMPDIR"] == "/tmp/viscous-jump-case/tmp"


def test_finite_step_courant_is_bounded_but_not_required_to_be_zero():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]

    accepted = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(runner, matrix, case, courant=2.0e-3)
        ),
        return_code=0,
    )
    assert accepted["passed"] is True
    assert accepted["metrics"]["phase_stage_courant"] == pytest.approx(2.0e-3)

    rejected = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(runner, matrix, case, courant=2.6e-3)
        ),
        return_code=0,
    )
    assert rejected["passed"] is False
    assert "phase_stage_courant" in rejected["failed_checks"]


def test_global_balance_gate_requires_nonzero_local_transfer_and_zero_net_flux():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]

    exercised = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output(runner, matrix, case)),
        return_code=0,
    )
    assert exercised["passed"] is True
    assert exercised["metrics"][
        "phase_stage_maximum_nodal_boundary_mass_transfer"
    ] == pytest.approx(1.0e-5)

    closed_like = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(
                runner,
                matrix,
                case,
                maximum_nodal_boundary_transfer=5.0e-13,
            )
        ),
        return_code=0,
    )
    assert closed_like["passed"] is False
    assert "phase_stage_balanced_boundary_flux_exercised" in closed_like[
        "failed_checks"
    ]


def test_stationary_equilibrium_gate_retains_reported_limiter_activity():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]

    accepted = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(runner, matrix, case, limited_edges=2)
        ),
        return_code=0,
    )
    assert accepted["passed"] is True
    assert accepted["metrics"]["phase_stage_limited_edges"] == 2
    assert accepted["metrics"]["phase_geometry_reconciliation_diagnostic"] == (
        "stationary_geometry_equilibrium_projection"
    )

    wrong_mode = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(
                runner,
                matrix,
                case,
                reconciliation_diagnostic="ok",
            )
        ),
        return_code=0,
    )
    assert wrong_mode["passed"] is False
    assert "phase_geometry_reconciliation_diagnostic" in wrong_mode[
        "failed_checks"
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("viscous_traction_jump_integral_1=0.059999999999999998", "viscous_traction_jump_integral_1=0.07"),
        ("negative_momentum_1=-273.80000000000001", "negative_momentum_1=-270"),
        ("negative_kinetic_energy=135.07466666666662", "negative_kinetic_energy=130"),
        ("prescribed_viscous_traction_jump_applicable=true", "prescribed_viscous_traction_jump_applicable=false"),
    ],
)
def test_gate_rejects_traction_bulk_or_applicability_error(field, replacement):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = matrix["cases"][0]
    output = passing_output(runner, matrix, case)
    assert field in output
    parsed = runner.parse_case_output(output.replace(field, replacement))
    result = runner.evaluate_case(case, matrix, parsed, return_code=0)
    assert result["passed"] is False


def test_two_rank_gate_requires_both_rank_initializers():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    case = next(case for case in matrix["cases"] if case["mpi_ranks"] == 2)

    incomplete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output(runner, matrix, case, ranks=(0,))),
        return_code=0,
    )
    assert incomplete["passed"] is False
    assert "negative_phase_initializer" in incomplete["failed_checks"]
    assert "positive_phase_initializer" in incomplete["failed_checks"]

    complete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output(runner, matrix, case, ranks=(0, 1))),
        return_code=0,
    )
    assert complete["passed"] is True


def test_side_reversal_swaps_bulk_phases_and_signed_individual_tractions():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    cases = [
        case
        for case in matrix["cases"]
        if case["reversal_pair"] == matrix["cases"][0]["reversal_pair"]
    ]
    results = [
        runner.evaluate_case(
            case,
            matrix,
            runner.parse_case_output(passing_output(runner, matrix, case)),
            return_code=0,
        )
        for case in cases
    ]
    outcome = runner.evaluate_reversal_pairs(results, matrix, required_pairs=1)
    assert outcome["passed"] is True
    assert len(outcome["checks"]) >= 20


def test_complete_synthetic_matrix_passes_every_case_and_reversal_check():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    results = []
    for case in matrix["cases"]:
        ranks = tuple(range(case["mpi_ranks"]))
        results.append(
            runner.evaluate_case(
                case,
                matrix,
                runner.parse_case_output(
                    passing_output(runner, matrix, case, ranks=ranks)
                ),
                return_code=0,
            )
        )
    assert all(result["passed"] for result in results)
    reversal = runner.evaluate_reversal_pairs(results, matrix)
    assert reversal["passed"] is True
    assert reversal["failed_checks"] == []
    assert len(reversal["checks"]) == 228


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
    assert record["matrix_id"] == "free_surface_wp10_viscous_jump_v2"
    assert record["case_count"] == 12
    assert record["reversal_pair_count"] == 6
    assert record["outcome"] == "PASS"
    assert record["planar_viscous_traction_jump_gate_passed"] is False
