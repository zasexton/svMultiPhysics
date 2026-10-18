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
    / "run_free_surface_wp10_static_drop_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_static_drop_matrix.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_static_drop_qualification_runner", RUNNER_PATH
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


def test_matrix_freezes_bounded_circle_progression(runner, matrix):
    assert matrix["schema_version"] == 1
    assert matrix["matrix_id"] == "free_surface_wp10_static_drop_v1"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["accepted_claim"] == "two_fluid_circular_static_drop_prerequisite"
    assert runner.validate_requested_claim(
        matrix, "two_fluid_circular_static_drop_prerequisite"
    ) == "two_fluid_circular_static_drop_prerequisite"
    for rejected in matrix["rejected_claims"]:
        with pytest.raises(ValueError):
            runner.validate_requested_claim(matrix, rejected)

    assert matrix["geometry"] == {
        "kind": "circle",
        "radius": 0.25,
        "centers": {
            "centered": [0.5, 0.5],
            "shifted": [0.5375, 0.475],
        },
    }
    assert matrix["refinement"] == {
        "resolutions": [32, 64, 128],
        "radius_over_h": [8.0, 16.0, 32.0],
        "families": ["centered", "shifted"],
        "expected_asymptotic_order": 2.0,
        "minimum_observed_order": 1.5,
    }
    assert matrix["thresholds"]["finest_pressure_jump_relative"] == 0.01
    assert matrix["thresholds"]["finest_maximum_phase_capillary_number"] == 1.0e-6
    assert len(matrix["cases"]) == 12
    assert {
        (case["resolution"], case["center_class"], case["level_set_sign"])
        for case in matrix["cases"]
    } == {
        (resolution, center_class, sign)
        for resolution in (32, 64, 128)
        for center_class in ("centered", "shifted")
        for sign in (-1, 1)
    }
    assert all(
        case["mpi_ranks"] == (2 if case["resolution"] == 128 else 1)
        for case in matrix["cases"]
    )
    assert matrix["qualification_disposition"] == {
        "two_fluid_circular_static_drop_gate_passed": False,
        "fsr08_closed": False,
        "wp10_closed": False,
        "q7_closed": False,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("status", "DEVELOPMENT"),
        lambda value: value["thresholds"].__setitem__(
            "finest_pressure_jump_relative", 0.02
        ),
        lambda value: value["thresholds"].pop("phase_partition_absolute"),
        lambda value: value["cases"][1].__setitem__(
            "case_id", value["cases"][0]["case_id"]
        ),
        lambda value: value["cases"][0].__setitem__("center", [0.51, 0.5]),
        lambda value: value["cases"][0].__setitem__("negative_density", 17.0),
        lambda value: value["cases"][0].__setitem__("mpi_ranks", 2),
    ],
)
def test_matrix_rejects_contract_drift(runner, matrix, mutation):
    changed = copy.deepcopy(matrix)
    mutation(changed)
    with pytest.raises(ValueError):
        runner._validate_matrix_structure(changed)


def test_material_reversal_pairs_preserve_physical_properties(runner, matrix):
    grouped = {}
    for case in matrix["cases"]:
        grouped.setdefault(case["reversal_pair"], []).append(case)
    assert len(grouped) == 6
    for members in grouped.values():
        assert len(members) == 2
        negative_liquid = next(
            case for case in members if case["level_set_sign"] == 1
        )
        positive_liquid = next(
            case for case in members if case["level_set_sign"] == -1
        )
        assert negative_liquid["negative_density"] == positive_liquid[
            "positive_density"
        ]
        assert negative_liquid["negative_viscosity"] == positive_liquid[
            "positive_viscosity"
        ]
        assert negative_liquid["positive_density"] == positive_liquid[
            "negative_density"
        ]
        assert negative_liquid["positive_viscosity"] == positive_liquid[
            "negative_viscosity"
        ]
        negative_geometry = runner.expected_geometry(negative_liquid)
        positive_geometry = runner.expected_geometry(positive_liquid)
        for key in ("interface_measure", "interior_volume", "pressure_jump"):
            assert negative_geometry[key] == pytest.approx(
                positive_geometry[key], rel=0.0, abs=0.0
            )
        assert negative_geometry["negative_volume"] == pytest.approx(
            positive_geometry["positive_volume"], rel=0.0, abs=0.0
        )
        assert negative_geometry["positive_volume"] == pytest.approx(
            positive_geometry["negative_volume"], rel=0.0, abs=0.0
        )


def test_circle_geometry_and_level_set_are_analytic(runner, matrix):
    case = next(
        case
        for case in matrix["cases"]
        if case["resolution"] == 32
        and case["center_class"] == "shifted"
        and case["level_set_sign"] == 1
    )
    geometry = runner.expected_geometry(case)
    radius = matrix["geometry"]["radius"]
    center_x, center_y = matrix["geometry"]["centers"]["shifted"]
    assert geometry["interface_measure"] == pytest.approx(2.0 * math.pi * radius)
    assert geometry["interior_volume"] == pytest.approx(math.pi * radius * radius)
    assert geometry["negative_volume"] == pytest.approx(geometry["interior_volume"])
    assert geometry["positive_volume"] == pytest.approx(1.0 - geometry["interior_volume"])
    assert geometry["pressure_jump"] == pytest.approx(4.0)
    assert runner.level_set_value(case, center_x, center_y) == pytest.approx(-radius)
    assert runner.level_set_value(
        case, center_x + radius, center_y
    ) == pytest.approx(0.0, abs=1.0e-15)
    reversed_case = copy.deepcopy(case)
    reversed_case["level_set_sign"] = -1
    assert runner.level_set_value(
        reversed_case, center_x, center_y
    ) == pytest.approx(radius)


def test_rendered_inputs_encode_circle_jump_and_closed_wall(runner, matrix):
    case = next(
        case
        for case in matrix["cases"]
        if case["resolution"] == 32
        and case["center_class"] == "shifted"
        and case["level_set_sign"] == 1
    )
    mesh_root = ET.fromstring(runner.render_mesh(matrix, case))
    arrays = {
        item.attrib.get("Name"): item
        for item in mesh_root.findall(".//PointData/DataArray")
    }
    expected_vertices = (case["resolution"] + 1) ** 2
    level_set = [float(value) for value in arrays["level_set"].text.split()]
    negative_pressure = [
        float(value) for value in arrays["p_negative"].text.split()
    ]
    positive_pressure = [
        float(value) for value in arrays["p_positive"].text.split()
    ]
    assert len(level_set) == expected_vertices
    assert len(negative_pressure) == expected_vertices
    assert len(positive_pressure) == expected_vertices
    geometry = runner.expected_geometry(case)
    assert set(negative_pressure) == {0.0}
    assert set(positive_pressure) == {-geometry["pressure_jump"]}
    assert min(level_set) < 0.0 < max(level_set)

    solver_root = ET.fromstring(runner.render_solver(matrix, case))
    fluid = solver_root.find("Add_equation[@type='stokes']")
    assert fluid is not None
    assert float(fluid.findtext("Two_fluid_surface_tension")) == pytest.approx(1.0)
    assert float(fluid.findtext("Force_x")) == 0.0
    assert float(fluid.findtext("Force_y")) == 0.0
    assert float(fluid.findtext("Force_z")) == 0.0
    boundary = fluid.find("Add_BC[@name='wall']")
    assert boundary is not None
    assert boundary.findtext("Type") == "Dir"
    assert boundary.findtext("Value") == "0"
    level_set_block = solver_root.find("Add_equation[@type='level_set']")
    assert level_set_block is not None
    assert level_set_block.findtext("Conservative_phase_reconcile_geometry") == "true"


def test_rendered_reversal_changes_only_oriented_phase_data(runner, matrix):
    forward = next(
        case
        for case in matrix["cases"]
        if case["resolution"] == 32
        and case["center_class"] == "shifted"
        and case["level_set_sign"] == 1
    )
    reverse = next(
        case
        for case in matrix["cases"]
        if case["resolution"] == 32
        and case["center_class"] == "shifted"
        and case["level_set_sign"] == -1
    )

    def arrays(case):
        root = ET.fromstring(runner.render_mesh(matrix, case))
        return {
            item.attrib.get("Name"): [
                float(value) for value in item.text.split()
            ]
            for item in root.findall(".//PointData/DataArray")
            if item.attrib.get("Name")
            in {"level_set", "p_negative", "p_positive"}
        }

    forward_arrays = arrays(forward)
    reverse_arrays = arrays(reverse)
    assert reverse_arrays["level_set"] == pytest.approx(
        [-value for value in forward_arrays["level_set"]], abs=0.0, rel=0.0
    )
    assert set(forward_arrays["p_negative"]) == {0.0}
    assert set(reverse_arrays["p_negative"]) == {0.0}
    assert set(forward_arrays["p_positive"]) == {-4.0}
    assert set(reverse_arrays["p_positive"]) == {4.0}


def exact_interface_record(runner, case):
    geometry = runner.expected_geometry(case)
    return {
        "interface_quadrature_points": "64",
        "interface_measure": format(geometry["interface_measure"], ".17g"),
        "velocity_jump_sq": "0",
        "normal_flux_jump": "0",
        "mean_pressure_jump": format(geometry["pressure_jump"], ".17g"),
        "pressure_jump_sq": format(
            geometry["pressure_jump"] ** 2 * geometry["interface_measure"],
            ".17g",
        ),
        "traction_jump_sq": format(
            geometry["pressure_jump"] ** 2 * geometry["interface_measure"],
            ".17g",
        ),
        "negative_kinetic_energy": "0",
        "positive_kinetic_energy": "0",
        "negative_maximum_quadrature_speed": "2e-5",
        "positive_maximum_quadrature_speed": "3e-5",
        "negative_volume": format(geometry["negative_volume"], ".17g"),
        "positive_volume": format(geometry["positive_volume"], ".17g"),
        "nonlinear_iterations": "2",
        "nonlinear_initial_residual_norm": "2e-11",
        "nonlinear_final_residual_norm": "1e-11",
        "linear_iterations": "4",
        "nonlinear_converged": "true",
        "linear_converged": "true",
    }


def test_static_observables_use_phasewise_maximum_speed(runner, matrix):
    case = next(
        case
        for case in matrix["cases"]
        if case["resolution"] == 128
        and case["center_class"] == "shifted"
        and case["level_set_sign"] == 1
    )
    evaluated = runner.evaluate_static_observables(
        case, matrix, exact_interface_record(runner, case)
    )
    assert evaluated["passed"], evaluated["failed_checks"]
    metrics = evaluated["metrics"]
    assert metrics["negative_maximum_capillary_number"] == pytest.approx(2.0e-7)
    assert metrics["positive_maximum_capillary_number"] == pytest.approx(3.0e-8)
    assert metrics["maximum_phase_capillary_number"] == pytest.approx(2.0e-7)
    assert metrics["pressure_jump_relative_error"] == pytest.approx(0.0)
    assert metrics["interface_measure_relative_error"] == pytest.approx(0.0)
    assert metrics["maximum_phase_capillary_number"] < matrix["thresholds"][
        "finest_maximum_phase_capillary_number"
    ]


def test_static_observables_fail_closed_without_either_phase_maximum(runner, matrix):
    case = matrix["cases"][-1]
    record = exact_interface_record(runner, case)
    del record["positive_maximum_quadrature_speed"]
    evaluated = runner.evaluate_static_observables(case, matrix, record)
    assert not evaluated["passed"]
    assert "positive_maximum_quadrature_speed" in evaluated["failed_checks"]


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-1"])
def test_static_observables_reject_invalid_maximum_speed(
    runner, matrix, bad_value
):
    case = matrix["cases"][-1]
    record = exact_interface_record(runner, case)
    record["negative_maximum_quadrature_speed"] = bad_value
    evaluated = runner.evaluate_static_observables(case, matrix, record)
    assert not evaluated["passed"]
    assert "negative_maximum_quadrature_speed" in evaluated["failed_checks"]


def synthetic_result(runner, case, scale=1.0):
    geometry = runner.expected_geometry(case)
    h = 1.0 / case["resolution"]
    error = scale * h * h
    liquid_phase = case["liquid_phase"]
    gas_phase = "positive" if liquid_phase == "negative" else "negative"
    metrics = {
        "interface_measure": geometry["interface_measure"] * (1.0 + error),
        "interface_measure_relative_error": error,
        "pressure_jump_relative_error": 8.0 * error,
        "negative_volume_relative_error": 2.0 * error,
        "positive_volume_relative_error": error,
        "interior_volume_relative_error": 2.0 * error,
        "pressure_jump_relative_dispersion": 4.0 * error,
        "maximum_phase_capillary_number": 2.0e-3 * h * h,
        "mean_pressure_jump": (
            case["level_set_sign"]
            * geometry["pressure_jump"]
            * (1.0 + 8.0 * error)
        ),
        "negative_volume": geometry["negative_volume"],
        "positive_volume": geometry["positive_volume"],
        f"{liquid_phase}_maximum_capillary_number": 2.0e-3 * h * h,
        f"{gas_phase}_maximum_capillary_number": 1.0e-3 * h * h,
    }
    return {
        "case_id": case["case_id"],
        "reversal_pair": case["reversal_pair"],
        "refinement_family": case["refinement_family"],
        "center_class": case["center_class"],
        "resolution": case["resolution"],
        "level_set_sign": case["level_set_sign"],
        "passed": True,
        "metrics": metrics,
        "reversal_metrics": runner.physical_reversal_metrics(case, metrics),
    }


def test_reversal_and_refinement_gates_accept_second_order_data(runner, matrix):
    results = [synthetic_result(runner, case) for case in matrix["cases"]]
    reversal = runner.evaluate_reversal_pairs(results, matrix)
    assert reversal["passed"], reversal["failed_checks"]
    refinement = runner.evaluate_refinement_families(results, matrix)
    assert refinement["passed"], refinement["failed_checks"]
    assert len(refinement["families"]) == 4
    for family in refinement["families"]:
        assert family["pressure_jump_relative_error"]["observed_orders"] == pytest.approx(
            [2.0, 2.0]
        )
        assert family["finest"]["pressure_jump_relative_error"] < 0.01
        assert family["finest"]["maximum_phase_capillary_number"] < 1.0e-6


def test_refinement_gate_rejects_finest_pressure_or_speed_regression(runner, matrix):
    results = [synthetic_result(runner, case) for case in matrix["cases"]]
    target = next(
        result
        for result in results
        if result["resolution"] == 128
        and result["center_class"] == "shifted"
        and result["level_set_sign"] == 1
    )
    target["metrics"]["pressure_jump_relative_error"] = 0.0101
    target["metrics"]["maximum_phase_capillary_number"] = 1.01e-6
    refinement = runner.evaluate_refinement_families(results, matrix)
    assert not refinement["passed"]
    assert any("finest_pressure_jump_relative_error" in name for name in refinement["failed_checks"])
    assert any(
        "finest_maximum_phase_capillary_number" in name
        for name in refinement["failed_checks"]
    )


def test_refinement_and_reversal_require_complete_convergent_families(runner, matrix):
    results = [synthetic_result(runner, case) for case in matrix["cases"]]
    incomplete = results[:-1]
    reversal = runner.evaluate_reversal_pairs(incomplete, matrix)
    refinement = runner.evaluate_refinement_families(incomplete, matrix)
    assert not reversal["passed"]
    assert not refinement["passed"]

    low_order = copy.deepcopy(results)
    family = [
        result
        for result in low_order
        if result["refinement_family"] == "shifted_negative"
    ]
    family.sort(key=lambda result: result["resolution"])
    family[0]["metrics"]["pressure_jump_relative_error"] = 0.02
    family[1]["metrics"]["pressure_jump_relative_error"] = 0.015
    family[2]["metrics"]["pressure_jump_relative_error"] = 0.009
    refinement = runner.evaluate_refinement_families(low_order, matrix)
    assert not refinement["passed"]
    assert any(
        "pressure_jump_relative_error:observed_order" in name
        for name in refinement["failed_checks"]
    )


def test_execution_outcome_keeps_broader_claims_open(runner):
    assert runner.execution_outcome(True, True) == {
        "outcome": "PASS",
        "two_fluid_circular_static_drop_gate_passed": True,
        "exit_code": 0,
    }
    assert runner.execution_outcome(True, False) == {
        "outcome": "DEVELOPMENT_PASS",
        "two_fluid_circular_static_drop_gate_passed": False,
        "exit_code": 0,
    }
    assert runner.execution_outcome(False, True) == {
        "outcome": "FAIL",
        "two_fluid_circular_static_drop_gate_passed": False,
        "exit_code": 1,
    }


def passing_parsed_output(runner, case):
    geometry = runner.expected_geometry(case)
    interface = exact_interface_record(runner, case)
    interface.update(
        {
            "momentum_reconciliation_applicable": "true",
            "velocity_update_applied": "false",
            "momentum_reconciliation_satisfied": "true",
            "accepted_stage_numerics_applicable": "true",
            "phase_iteration_scope": "shared_coupled_solve",
        }
    )
    expected_dofs = 3 * (case["resolution"] + 1) ** 2
    initializers = []
    for rank in range(case["mpi_ranks"]):
        for phase in ("negative", "positive"):
            initializers.append(
                {
                    "log_rank": str(rank),
                    "initialized_dofs": str(expected_dofs),
                    "velocity_field": f"u_{phase}",
                    "pressure_field": f"p_{phase}",
                }
            )
    negative_volume = geometry["negative_volume"]
    return {
        "initializers": initializers,
        "phase_stage": {
            "previous_measure": format(negative_volume, ".17g"),
            "accepted_measure": format(negative_volume, ".17g"),
            "boundary_transfer": "0",
            "boundary_mass_tolerance": "1e-12",
            "divergence_source": "0",
            "global_balance_residual": "0",
            "max_local_balance_residual": "0",
            "max_component_balance_residual": "0",
            "limited_edges": "0",
            "courant": "0",
            "stage_invariant_tolerance": "1e-12",
            "stage_maximum_courant": "1",
        },
        "phase_geometry": {
            "phase_measure": format(negative_volume, ".17g"),
            "retained_geometry_measure": format(negative_volume, ".17g"),
            "measure_mismatch": "0",
            "max_nodal_moment_mismatch": "0",
            "nodal_moment_residual_norm": "0",
            "reconciliation_iterations": "0",
            "interface_displacement_bound": "0",
            "reconciliation_enabled": "true",
        },
        "maintenance": {
            "raw_post_transport_phase_measure": format(negative_volume, ".17g"),
            "post_limit_phase_measure": format(negative_volume, ".17g"),
            "raw_post_transport_geometry_measure": format(negative_volume, ".17g"),
            "post_reinitialization_phase_measure": format(negative_volume, ".17g"),
            "post_reinitialization_geometry_measure": format(negative_volume, ".17g"),
            "post_correction_phase_measure": format(negative_volume, ".17g"),
            "post_correction_geometry_measure": format(negative_volume, ".17g"),
            "retained_assembly_measure": format(negative_volume, ".17g"),
            "total_physical_boundary_mass_transfer": "0",
            "boundary_mass_tolerance": "1e-12",
            "transport_component_balance_satisfied": "true",
            "transport_component_measure_closure_satisfied": "true",
            "transport_max_component_balance_residual": "0",
            "stage_invariant_tolerance": "1e-12",
            "reconciliation_interface_displacement_bound": "0",
        },
        "interface": interface,
        "artifact": None,
        "accepted_step_count": 1,
    }


def test_evaluate_case_requires_complete_accepted_stage(runner, matrix):
    case = matrix["cases"][0]
    parsed = passing_parsed_output(runner, case)
    result = runner.evaluate_case(case, matrix, parsed, 0)
    assert result["passed"], result["failed_checks"]
    assert result["refinement_family"] == case["refinement_family"]
    assert result["resolution"] == case["resolution"]
    assert result["reversal_metrics"] == runner.physical_reversal_metrics(
        case, result["metrics"]
    )

    failed_process = runner.evaluate_case(case, matrix, parsed, 9)
    assert not failed_process["passed"]
    assert "process_return_code" in failed_process["failed_checks"]
    missing_step = copy.deepcopy(parsed)
    missing_step["accepted_step_count"] = 0
    failed_step = runner.evaluate_case(case, matrix, missing_step, 0)
    assert not failed_step["passed"]
    assert "accepted_step_count" in failed_step["failed_checks"]


def effective_configuration(runner, case):
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
                    key: case[key]
                    for key in (
                        "negative_density",
                        "negative_viscosity",
                        "positive_density",
                        "positive_viscosity",
                    )
                },
                "body_force": [0.0, 0.0],
                "interface": {
                    "surface_tension": case["surface_tension"],
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
                    "shared_gauge_vertex_gid": runner.pressure_gauge_vertex(case),
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
                    "reconcile_geometry": True,
                },
            },
        ]
    }


def test_effective_configuration_binds_static_material_and_surface(runner, matrix):
    case = matrix["cases"][2]
    document = effective_configuration(runner, case)
    validated = runner.validate_effective_configuration(document, case)
    assert validated["surface_tension"] == case["surface_tension"]
    assert validated["shared_pressure_gauge_vertex_gid"] == (
        runner.pressure_gauge_vertex(case)
    )
    assert validated["stationary_geometry_reconciliation"]

    wrong_surface = copy.deepcopy(document)
    wrong_surface["modules"][0]["interface"]["surface_tension"] = 0.0
    with pytest.raises(ValueError):
        runner.validate_effective_configuration(wrong_surface, case)
    wrong_material = copy.deepcopy(document)
    wrong_material["modules"][0]["material"]["negative_density"] = 17.0
    with pytest.raises(ValueError):
        runner.validate_effective_configuration(wrong_material, case)
    wrong_gauge = copy.deepcopy(document)
    wrong_gauge["modules"][0]["pressure_space"][
        "shared_gauge_vertex_gid"
    ] += 1
    with pytest.raises(ValueError):
        runner.validate_effective_configuration(wrong_gauge, case)
    wrong_transport = copy.deepcopy(document)
    wrong_transport["modules"][1]["conservative_phase"][
        "reconcile_geometry"
    ] = False
    with pytest.raises(ValueError):
        runner.validate_effective_configuration(wrong_transport, case)


def test_qualification_environment_scrubs_solver_overrides(runner, matrix, tmp_path):
    inherited = {
        "SVMP_NEWTON_ABS_TOLERANCE": "9",
        "SVMP_NEWTON_REL_TOLERANCE": "8",
        "UNCHANGED": "yes",
    }
    environment = runner.qualification_environment(
        matrix, inherited, tmp_path
    )
    assert "SVMP_NEWTON_ABS_TOLERANCE" not in environment
    assert "SVMP_NEWTON_REL_TOLERANCE" not in environment
    assert environment["UNCHANGED"] == "yes"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["SVMP_OOP_SOLVER_TRACE"] == "0"
    assert environment["TMPDIR"] == str(tmp_path / "tmp")


def test_execution_record_refuses_existing_output(runner, matrix, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        runner.execution_record(
            matrix,
            MATRIX_PATH,
            Path("/missing/solver"),
            Path("/missing/launcher"),
            output,
            False,
        )


def test_validate_only_lists_frozen_scope(matrix):
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--matrix",
            str(MATRIX_PATH),
            "--requested-claim",
            "two_fluid_circular_static_drop_prerequisite",
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["matrix_id"] == matrix["matrix_id"]
    assert payload["case_count"] == 12
    assert payload["reversal_pair_count"] == 6
    assert payload["refinement_family_count"] == 4
    assert payload["outcome"] == "PASS"
    assert not payload["two_fluid_circular_static_drop_gate_passed"]
    assert not payload["fsr08_closed"]
    assert not payload["wp10_closed"]
    assert not payload["q7_closed"]


def test_command_line_rejects_broader_claim():
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--matrix",
            str(MATRIX_PATH),
            "--requested-claim",
            "wp10_closure",
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "exceeds accepted claim" in completed.stderr
