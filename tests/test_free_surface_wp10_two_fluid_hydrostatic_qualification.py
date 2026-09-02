import copy
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
    / "run_free_surface_wp10_two_fluid_hydrostatic_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_two_fluid_hydrostatic_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_two_fluid_hydrostatic_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load_runner()


@pytest.fixture(scope="module")
def matrix(runner):
    return runner.load_matrix(MATRIX_PATH)


def test_matrix_declares_bounded_prerequisite_and_reversal_pairs(runner, matrix):
    assert matrix["schema_version"] == 1
    assert matrix["matrix_id"] == "free_surface_wp10_two_fluid_hydrostatic_v1"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["accepted_claim"] == "two_fluid_hydrostatic_prerequisite"
    assert runner.validate_requested_claim(
        matrix, "two_fluid_hydrostatic_prerequisite"
    ) == "two_fluid_hydrostatic_prerequisite"
    for rejected in matrix["rejected_claims"]:
        with pytest.raises(ValueError):
            runner.validate_requested_claim(matrix, rejected)
    assert len(matrix["cases"]) == 12
    assert matrix["thresholds"][
        "pressure_squared_gauge_normalization_ulp_factor"
    ] == 8.0
    pairs = {}
    for case in matrix["cases"]:
        pairs.setdefault(case["reversal_pair"], []).append(case)
    assert len(pairs) == 6
    assert all(
        len(members) == 2
        and {member["level_set_sign"] for member in members} == {-1, 1}
        for members in pairs.values()
    )
    assert matrix["qualification_disposition"] == {
        "two_fluid_hydrostatic_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }


def test_body_force_is_normal_and_invariant_under_material_label_reversal(
    runner, matrix
):
    pairs = {}
    for case in matrix["cases"]:
        pairs.setdefault(case["reversal_pair"], []).append(case)
    for members in pairs.values():
        first, second = members
        first_force = runner.body_force(first)
        second_force = runner.body_force(second)
        assert first_force == pytest.approx(second_force, abs=0.0, rel=0.0)
        normal = runner.base_unit_normal(first)
        gravity = first["gravity_acceleration"]
        assert first_force == pytest.approx(
            (gravity * normal[0], gravity * normal[1], 0.0),
            abs=1.0e-15,
            rel=1.0e-15,
        )


def test_piecewise_pressure_is_continuous_and_density_scaled(runner, matrix):
    for case in matrix["cases"]:
        normal = runner.base_unit_normal(case)
        interface_point = runner.interface_anchor(case)
        for phase in ("negative", "positive"):
            assert runner.hydrostatic_pressure(
                case, phase, *interface_point
            ) == pytest.approx(0.0, abs=2.0e-12)
            density = case[f"{phase}_density"]
            expected_gradient = tuple(
                density * value for value in runner.body_force(case)
            )
            assert runner.hydrostatic_pressure_gradient(
                case, phase
            ) == pytest.approx(expected_gradient, abs=2.0e-12, rel=2.0e-15)
            step = 1.0e-6
            offset_point = (
                interface_point[0] + step * normal[0],
                interface_point[1] + step * normal[1],
            )
            finite_difference = (
                runner.hydrostatic_pressure(case, phase, *offset_point)
                - runner.hydrostatic_pressure(case, phase, *interface_point)
            ) / step
            projected_gradient = sum(
                expected_gradient[index] * normal[index] for index in range(2)
            )
            assert finite_difference == pytest.approx(
                projected_gradient, abs=2.0e-6, rel=2.0e-10
            )


def test_rendered_inputs_embed_pressure_body_force_and_closed_wall(runner, matrix):
    case = matrix["cases"][4]
    mesh_root = ET.fromstring(runner.render_mesh(matrix, case))
    arrays = {
        item.attrib.get("Name"): item
        for item in mesh_root.findall(".//PointData/DataArray")
    }
    vertices = matrix["mesh"]["expected_vertices"]
    negative = [float(value) for value in arrays["p_negative"].text.split()]
    positive = [float(value) for value in arrays["p_positive"].text.split()]
    assert len(negative) == vertices
    assert len(positive) == vertices
    nx = matrix["mesh"]["nx"]
    ny = matrix["mesh"]["ny"]
    expected_negative = []
    expected_positive = []
    gauge_shift = runner.pressure_gauge_shift(matrix, case)
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i / nx
            y = j / ny
            expected_negative.append(
                runner.hydrostatic_pressure(
                    case, "negative", x, y, gauge_shift=gauge_shift
                )
            )
            expected_positive.append(
                runner.hydrostatic_pressure(
                    case, "positive", x, y, gauge_shift=gauge_shift
                )
            )
    assert negative == pytest.approx(expected_negative, abs=1.0e-12, rel=1.0e-15)
    assert positive == pytest.approx(expected_positive, abs=1.0e-12, rel=1.0e-15)

    solver_root = ET.fromstring(runner.render_solver(matrix, case))
    fluid = solver_root.find("Add_equation[@type='stokes']")
    assert fluid is not None
    force = runner.body_force(case)
    assert float(fluid.findtext("Force_x")) == pytest.approx(force[0])
    assert float(fluid.findtext("Force_y")) == pytest.approx(force[1])
    assert float(fluid.findtext("Force_z")) == pytest.approx(0.0)
    boundary = fluid.find("Add_BC[@name='wall']")
    assert boundary is not None
    assert boundary.findtext("Type") == "Dir"
    assert boundary.findtext("Value") == "0"


def hydrostatic_interface_record(runner, matrix, case, gauge_shift=7.25):
    geometry = runner.common_geometry(case)
    record = {
        "interface_quadrature_points": "8",
        "interface_measure": format(geometry["interface_measure"], ".17g"),
        "velocity_jump_sq": "0",
        "velocity_jump_normal_sq": "0",
        "velocity_jump_tangential_sq": "0",
        "negative_normal_flux": "0",
        "positive_normal_flux": "0",
        "normal_flux_jump": "0",
        "negative_mass_flux": "0",
        "positive_mass_flux": "0",
        "mean_pressure_jump": "0",
        "pressure_jump_sq": "0",
        "pressure_jump_integral": "0",
        "traction_jump_normal_integral": "0",
        "traction_jump_sq": "0",
        "viscous_traction_jump_sq": "0",
        "negative_kinetic_energy": "0",
        "positive_kinetic_energy": "0",
    }
    for phase in ("negative", "positive"):
        expected = runner.expected_phase_observables(
            case, phase, gauge_shift=gauge_shift
        )
        record[f"{phase}_phase_quadrature_points"] = "34"
        record[f"{phase}_density"] = format(case[f"{phase}_density"], ".17g")
        for field, value in expected.items():
            record[f"{phase}_{field}"] = format(value, ".17g")
        for component in range(3):
            record[f"{phase}_momentum_{component}"] = "0"
    return record


def passing_output(runner, matrix, case, ranks=None):
    if ranks is None:
        ranks = tuple(range(case["mpi_ranks"]))
    geometry = runner.common_geometry(case)
    negative_volume = geometry["negative_volume"]
    gauge_shift = runner.pressure_gauge_shift(matrix, case)
    interface = hydrostatic_interface_record(
        runner, matrix, case, gauge_shift=gauge_shift
    )
    interface.update(
        {
            "surface_energy_work": "0",
            "nitsche_consistency_work": "0",
            "nitsche_adjoint_work": "0",
            "nitsche_penalty_work": "0",
            "negative_momentum_delta_norm": "0",
            "positive_momentum_delta_norm": "0",
            "momentum_reconciliation_applicable": "true",
            "velocity_update_applied": "false",
            "momentum_reconciliation_satisfied": "true",
            "accepted_stage_numerics_applicable": "true",
            "nonlinear_converged": "true",
            "nonlinear_iterations": "0",
            "nonlinear_initial_residual_norm": "1e-12",
            "nonlinear_final_residual_norm": "1e-12",
            "linear_converged": "true",
            "linear_iterations": "0",
            "phase_iteration_scope": "shared_coupled_solve",
            "prescribed_pressure_jump_applicable": "false",
            "prescribed_viscous_traction_jump_applicable": "false",
            "prescribed_stress_jump_residual_applicable": "false",
        }
    )
    initializer_lines = []
    for rank in ranks:
        for phase in ("negative", "positive"):
            initializer_lines.append(
                f"[R{rank}] mesh-field initialization "
                "diagnostic=mesh_field_initialization initialized_dofs=75 "
                f"velocity_field='u_{phase}' pressure_field='p_{phase}'"
            )
    interface_fields = " ".join(
        f"{name}={value}" for name, value in interface.items()
    )
    absolute = matrix["nonlinear_solver"]["absolute_tolerance"]
    relative = matrix["nonlinear_solver"]["relative_tolerance"]
    maximum = matrix["nonlinear_solver"]["maximum_iterations"]
    return "\n".join(
        initializer_lines
        + [
            (
                "[svMultiPhysics::Application] Transient solve: t0=0 "
                f"dt={matrix['time']['dt']:.17g} "
                f"t_end={matrix['time']['dt']:.17g} max_steps=1 "
                "scheme=BackwardEuler rho_inf=n/a pde_udot_init=0 "
                "last_step_absorb_fraction=0 "
                f"newton(max_it={maximum}, min_it=0, "
                f"abs_tol={absolute:.17g}, rel_tol={relative:.17g})"
            ),
            (
                "Conservative phase staged field='phase' step=1 "
                f"previous_measure={negative_volume:.17g} "
                f"accepted_measure={negative_volume:.17g} "
                "boundary_transfer=0 divergence_source=0 "
                "global_balance_residual=0 max_local_balance_residual=0 "
                "max_component_balance_residual=0 limited_edges=0 courant=0"
            ),
            (
                "Conservative phase geometry validated field='phase' step=1 "
                f"phase_measure={negative_volume:.17g} "
                f"retained_geometry_measure={negative_volume:.17g} "
                "measure_mismatch=0 max_nodal_moment_mismatch=0 "
                "nodal_moment_residual_norm=0 reconciliation_iterations=0 "
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
                "reconciliation_interface_displacement_bound=0"
            ),
            (
                "[R0] accepted two-fluid interface diagnostics "
                "semantics=operator_stage accepted_step=1 interface_marker=71 "
                + interface_fields
            ),
            (
                "TimeLoop: step_accepted step=1 "
                f"time={matrix['time']['dt']:.17g} "
                f"dt={matrix['time']['dt']:.17g}"
            ),
        ]
    )


def effective_configuration(runner, matrix, case):
    force = runner.body_force(case)
    gauge_vertex = runner.pressure_gauge_vertex(matrix, case)
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
                "body_force": list(force[:2]),
                "hydrostatic_balance_diagnostic": (
                    "phasewise_integrated_pressure_gradient_minus_density_body_force"
                ),
                "interface": {
                    "surface_tension": 0.0,
                    "prescribed_pressure_jump_applicable": False,
                    "prescribed_viscous_traction_jump_applicable": False,
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
                                {"kind": "literal", "value": 0.0},
                                {"kind": "literal", "value": 0.0},
                            ],
                        }
                    ],
                    "negative_phase_local_velocity_dirichlet_count": 0,
                    "positive_phase_local_velocity_dirichlet_count": 0,
                },
                "pressure_space": {
                    "representation": "separate_phase_fields",
                    "shared_gauge_count": 1,
                    "shared_gauge_policy": "explicit_global_vertex_gid",
                    "shared_gauge_field": "p_negative",
                    "shared_gauge_id_type": "Global_vertex_gid",
                    "shared_gauge_vertex_gid": gauge_vertex[
                        "global_vertex_gid"
                    ],
                    "shared_gauge_value": 0.0,
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


def test_hydrostatic_observable_evaluator_accepts_exact_piecewise_affine_state(
    runner, matrix
):
    for case in matrix["cases"]:
        record = hydrostatic_interface_record(runner, matrix, case)
        result = runner.evaluate_hydrostatic_observables(case, matrix, record)
        assert result["passed"], result["failed_checks"]
        assert result["failed_checks"] == []
        assert result["metrics"]["common_pressure_gauge"] == pytest.approx(7.25)


@pytest.mark.parametrize(
    "field,threshold",
    [
        (
            "negative_hydrostatic_residual_sq",
            "hydrostatic_residual_squared_absolute",
        ),
        (
            "positive_hydrostatic_residual_integral_0",
            "hydrostatic_residual_integral_absolute",
        ),
        (
            "negative_pressure_gradient_integral_1",
            "pressure_gradient_absolute",
        ),
        (
            "positive_body_force_density_integral_0",
            "body_force_integral_absolute",
        ),
    ],
)
def test_hydrostatic_observable_evaluator_rejects_balance_perturbations(
    runner, matrix, field, threshold
):
    case = matrix["cases"][0]
    record = hydrostatic_interface_record(runner, matrix, case)
    record[field] = format(
        float(record[field]) + 10.0 * matrix["thresholds"][threshold], ".17g"
    )
    result = runner.evaluate_hydrostatic_observables(case, matrix, record)
    assert not result["passed"]
    assert result["failed_checks"]


def test_hydrostatic_observable_evaluator_rejects_phasewise_gauge_mismatch(
    runner, matrix
):
    case = matrix["cases"][2]
    record = hydrostatic_interface_record(runner, matrix, case)
    positive_volume = float(record["positive_volume"])
    record["positive_pressure_integral"] = format(
        float(record["positive_pressure_integral"])
        + 100.0 * matrix["thresholds"]["common_gauge_absolute"] * positive_volume,
        ".17g",
    )
    result = runner.evaluate_hydrostatic_observables(case, matrix, record)
    assert not result["passed"]
    assert "common_pressure_gauge" in result["failed_checks"]


def test_reversal_pairs_preserve_unlabeled_physical_observables(runner, matrix):
    results = []
    for case in matrix["cases"]:
        results.append(
            {
                "case_id": case["case_id"],
                "reversal_pair": case["reversal_pair"],
                "level_set_sign": case["level_set_sign"],
                "passed": True,
                "metrics": runner.physical_reversal_observables(case),
            }
        )
    reversal = runner.evaluate_reversal_pairs(results, matrix)
    assert reversal["passed"], reversal["failed_checks"]

    broken = copy.deepcopy(results)
    broken[1]["metrics"]["base_negative_pressure_gradient_integral_0"] += 1.0
    reversal = runner.evaluate_reversal_pairs(broken, matrix)
    assert not reversal["passed"]
    assert reversal["failed_checks"]


def test_pressure_initializer_satisfies_declared_shared_gauge(runner, matrix):
    for case in matrix["cases"]:
        anchor = runner.pressure_gauge_anchor(matrix, case)
        gauge = runner.pressure_gauge_shift(matrix, case)
        assert runner.hydrostatic_pressure(
            case, "negative", *anchor, gauge_shift=gauge
        ) == pytest.approx(0.0, abs=2.0e-12)
        root = ET.fromstring(runner.render_mesh(matrix, case))
        arrays = {
            item.attrib.get("Name"): item
            for item in root.findall(".//PointData/DataArray")
        }
        negative = [float(value) for value in arrays["p_negative"].text.split()]
        nx = matrix["mesh"]["nx"]
        index = round(anchor[1] * matrix["mesh"]["ny"]) * (nx + 1) + round(
            anchor[0] * nx
        )
        assert negative[index] == pytest.approx(0.0, abs=2.0e-12)


def test_matrix_declares_explicit_partition_invariant_pressure_gauge(matrix):
    assert matrix["model_envelope"]["pressure_gauge"] == (
        "explicit_negative_phase_global_vertex_gid_zero"
    )


def test_rendered_gauge_uses_deep_negative_global_vertex(runner, matrix):
    for case in matrix["cases"]:
        vertex = runner.pressure_gauge_vertex(matrix, case)
        assert vertex["global_vertex_gid"] == vertex["point_index"] + 1
        assert vertex["level_set"] < 0.0
        assert vertex["level_set"] == min(
            runner.COMMON.level_set_value(
                case,
                i / matrix["mesh"]["nx"],
                j / matrix["mesh"]["ny"],
            )
            for j in range(matrix["mesh"]["ny"] + 1)
            for i in range(matrix["mesh"]["nx"] + 1)
        )
        assert runner.render_pressure_gauge(matrix, case) == (
            "node_id,pressure\n"
            f"{vertex['global_vertex_gid']},0\n"
        )

        solver = ET.fromstring(runner.render_solver(matrix, case))
        block = solver.find(
            "Add_equation[@type='stokes']/Node_pressure_constraints"
        )
        assert block is not None
        assert block.findtext("Id_type") == "Global_vertex_gid"
        assert block.findtext("Values_file_path") == "pressure_gauge.csv"


def test_effective_configuration_binds_force_diagnostic_and_closed_wall(
    runner, matrix
):
    case = matrix["cases"][0]
    document = effective_configuration(runner, matrix, case)
    result = runner.validate_effective_configuration(document, case, matrix)
    assert result["body_force"] == pytest.approx(runner.body_force(case)[:2])
    assert result["shared_pressure_gauge_count"] == 1
    assert result["shared_pressure_gauge_vertex_gid"] == (
        runner.pressure_gauge_vertex(matrix, case)["global_vertex_gid"]
    )
    assert result["shared_velocity_dirichlet_count"] == 1

    wrong_force = copy.deepcopy(document)
    wrong_force["modules"][0]["body_force"][0] += 1.0
    with pytest.raises(ValueError, match="changed the body force"):
        runner.validate_effective_configuration(wrong_force, case, matrix)

    wrong_diagnostic = copy.deepcopy(document)
    wrong_diagnostic["modules"][0]["hydrostatic_balance_diagnostic"] = "other"
    with pytest.raises(ValueError, match="hydrostatic diagnostics"):
        runner.validate_effective_configuration(
            wrong_diagnostic, case, matrix
        )

    nonzero_wall = copy.deepcopy(document)
    nonzero_wall["modules"][0]["boundary_conditions"][
        "shared_velocity_dirichlet"
    ][0]["values"][1]["value"] = 1.0
    with pytest.raises(ValueError, match="closed-wall boundary"):
        runner.validate_effective_configuration(nonzero_wall, case, matrix)

    wrong_gauge = copy.deepcopy(document)
    wrong_gauge["modules"][0]["pressure_space"][
        "shared_gauge_vertex_gid"
    ] += 1
    with pytest.raises(ValueError, match="shared pressure gauge"):
        runner.validate_effective_configuration(wrong_gauge, case, matrix)


def test_exact_hydrostatic_output_passes_complete_case_gate(runner, matrix):
    case = matrix["cases"][0]
    parsed = runner.parse_case_output(passing_output(runner, matrix, case))
    result = runner.evaluate_case(case, matrix, parsed, return_code=0)
    assert result["passed"], result["failed_checks"]
    assert result["failed_checks"] == []
    assert result["metrics"]["common_pressure_gauge"] == pytest.approx(
        runner.pressure_gauge_shift(matrix, case)
    )
    assert result["metrics"]["negative_hydrostatic_residual_sq"] == 0.0
    assert result["reversal_metrics"]

    perturbed = passing_output(runner, matrix, case).replace(
        "negative_hydrostatic_residual_sq=0",
        "negative_hydrostatic_residual_sq=1e-10",
    )
    rejected = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(perturbed),
        return_code=0,
    )
    assert not rejected["passed"]
    assert "negative_hydrostatic_residual_sq" in rejected["failed_checks"]


def test_two_rank_case_requires_complete_initializer_evidence(runner, matrix):
    case = next(case for case in matrix["cases"] if case["mpi_ranks"] == 2)
    incomplete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(passing_output(runner, matrix, case, ranks=(0,))),
        return_code=0,
    )
    assert not incomplete["passed"]
    assert "negative_phase_initializer" in incomplete["failed_checks"]
    assert "positive_phase_initializer" in incomplete["failed_checks"]

    complete = runner.evaluate_case(
        case,
        matrix,
        runner.parse_case_output(
            passing_output(runner, matrix, case, ranks=(0, 1))
        ),
        return_code=0,
    )
    assert complete["passed"], complete["failed_checks"]


def test_complete_synthetic_matrix_passes_case_and_reversal_gates(runner, matrix):
    results = [
        runner.evaluate_case(
            case,
            matrix,
            runner.parse_case_output(passing_output(runner, matrix, case)),
            return_code=0,
        )
        for case in matrix["cases"]
    ]
    assert all(result["passed"] for result in results)
    reversal = runner.evaluate_reversal_pairs(results, matrix)
    assert reversal["passed"], reversal["failed_checks"]


def test_qualification_environment_is_rank_trace_and_override_stable(
    runner, matrix
):
    environment = runner.qualification_environment(
        matrix,
        {
            "SVMP_OOP_SOLVER_TRACE": "1",
            "SVMP_NEWTON_ABS_TOLERANCE": "9.9",
        },
        Path("/tmp/two-fluid-hydrostatic-case"),
    )
    assert environment["SVMP_OOP_SOLVER_TRACE"] == "0"
    assert "SVMP_NEWTON_ABS_TOLERANCE" not in environment
    assert environment["TMPDIR"] == "/tmp/two-fluid-hydrostatic-case/tmp"


def test_execution_outcome_does_not_upgrade_development_pass(runner):
    assert runner.execution_outcome(True, False) == {
        "outcome": "DEVELOPMENT_PASS",
        "two_fluid_hydrostatic_gate_passed": False,
        "exit_code": 0,
    }
    assert runner.execution_outcome(False, True)["exit_code"] == 1
    assert runner.execution_outcome(True, True)[
        "two_fluid_hydrostatic_gate_passed"
    ]


def test_frozen_matrix_requires_clean_matching_inputs(runner, matrix):
    provenance = {"tracked_clean": True}
    input_identity = {
        "matrix": {"matches_head": True},
        "runner": {"matches_head": True},
        "runner_test": {"matches_head": True},
    }
    assert runner.qualification_eligibility(
        matrix, provenance, input_identity
    )

    assert not runner.qualification_eligibility(
        matrix, {"tracked_clean": False}, input_identity
    )
    mismatched = copy.deepcopy(input_identity)
    mismatched["runner_test"]["matches_head"] = False
    assert not runner.qualification_eligibility(
        matrix, provenance, mismatched
    )


def test_validation_cli_reports_frozen_nonclosure_matrix():
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["matrix_id"] == "free_surface_wp10_two_fluid_hydrostatic_v1"
    assert result["case_count"] == 12
    assert result["reversal_pair_count"] == 6
    assert result["outcome"] == "PASS"
    assert result["two_fluid_hydrostatic_gate_passed"] is False
    assert result["wp10_closed"] is False


def test_matrix_json_has_no_nonfinite_constants():
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    def visit(value):
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, float):
            assert math.isfinite(value)

    visit(matrix)
