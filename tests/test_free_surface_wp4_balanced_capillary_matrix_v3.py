import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


TESTS_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNNER_PATH = (
    TESTS_ROOT
    / "cases/fluid/run_free_surface_wp4_balanced_capillary_matrix_v3.py"
)
DEFAULT_MATRIX_PATH = DEFAULT_RUNNER_PATH.with_name(
    "free_surface_wp4_balanced_capillary_matrix_v3.json"
)
RUNNER_PATH = Path(
    os.environ.get("WP4_MATRIX_RUNNER_UNDER_TEST", DEFAULT_RUNNER_PATH)
)
MATRIX_PATH = Path(
    os.environ.get("WP4_MATRIX_UNDER_TEST", DEFAULT_MATRIX_PATH)
)
SPEC = importlib.util.spec_from_file_location(
    "free_surface_wp4_balanced_capillary_matrix_contract", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def registry():
    return runner.load_registry(MATRIX_PATH)


def studies_by_id(value):
    return {study["id"]: study for study in value["studies"]}


def option_values(arguments, option):
    return [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == option
    ]


def test_v3_registry_is_frozen_and_parent_v2_bytes_are_pinned():
    value = registry()
    assert value["schema_version"] == 3
    assert value["matrix_id"] == "free_surface_wp4_balanced_capillary_v3"
    assert value["status"] == "FROZEN_BEFORE_EXECUTION"
    assert runner.sha256_file(runner.PARENT_RUNNER_PATH) == (
        runner.EXPECTED_PARENT_RUNNER_SHA256
    )
    assert runner.sha256_file(runner.PARENT_REGISTRY_PATH) == (
        runner.EXPECTED_PARENT_REGISTRY_SHA256
    )


def test_v3_registry_rejects_duplicate_keys_and_byte_drift(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 3, "schema_version": 3}\n', encoding="utf-8"
    )
    with pytest.raises(runner.MatrixError, match="duplicate JSON key"):
        runner.read_json(duplicate)

    changed = tmp_path / MATRIX_PATH.name
    changed.write_text(
        MATRIX_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(runner.MatrixError, match="V3 frozen registry bytes changed"):
        runner.load_registry(changed)


@pytest.mark.parametrize(
    "section,mutation,match",
    [
        (
            "top-level",
            lambda value: value.update({"unexpected": True}),
            "top-level fields",
        ),
        (
            "gate",
            lambda value: value["gates"].pop("energy_variation"),
            "gate fields",
        ),
        (
            "study",
            lambda value: value["studies"][0].update({"unexpected": True}),
            "study fields",
        ),
        (
            "axis",
            lambda value: value["studies"][0]["axes"].update(
                {"unexpected": [0]}
            ),
            "axis fields",
        ),
        (
            "resource",
            lambda value: value["resources"].pop("memory_model"),
            "resource fields",
        ),
    ],
)
def test_v3_contract_rejects_unknown_or_missing_fields(section, mutation, match):
    value = copy.deepcopy(registry())
    mutation(value)
    with pytest.raises(runner.MatrixError, match=match):
        runner.validate_contract(value)


def test_energy_variation_count_is_positive_and_bound_to_exact_gates():
    value = registry()
    energy = value["gates"]["energy_variation"]
    components = energy["finite_difference_components"]
    assert isinstance(components, int) and not isinstance(components, bool)
    assert components > 0

    focused = next(
        group for group in value["exact_groups"] if group["id"] == "focused_algebra"
    )
    gates = {
        test: gate
        for invocation in focused["invocations"]
        for test, test_gates in invocation["property_gates"].items()
        for gate in test_gates
    }
    two_dimensional = (
        "FreeSurfaceGeometrySnapshot."
        "DiscreteFunctionalFirstVariationMatchesCentralDifference"
    )
    three_dimensional = (
        "FreeSurfaceGeometrySnapshot."
        "DiscreteFunctionalFirstVariationMatchesThreeDimensionalCentralDifference"
    )
    count_gates = [
        gate
        for invocation in focused["invocations"]
        for test in (two_dimensional, three_dimensional)
        for gate in invocation["property_gates"].get(test, [])
        if gate["property"].endswith("fd_case_count")
    ]
    assert len(count_gates) == 2
    assert all(
        gate == {
            "property": gate["property"],
            "comparison": "equal",
            "expected": components,
        }
        for gate in count_gates
    )
    assert gates[two_dimensional]["property"].endswith("max_relative_error")
    assert gates[two_dimensional]["comparison"] == "at_most"
    assert gates[two_dimensional]["expected"] <= energy[
        "maximum_relative_directional_derivative_error"
    ]
    assert gates[three_dimensional]["property"].endswith("max_relative_error")
    assert gates[three_dimensional]["comparison"] == "at_most"
    assert gates[three_dimensional]["expected"] <= energy[
        "maximum_relative_directional_derivative_error"
    ]


def test_selected_traction_commands_have_one_explicit_quadratic_interface_order(
    tmp_path,
):
    value = registry()
    cases = runner.expand_cases(value)
    assert cases
    for case in cases:
        arguments = runner.physical_case_arguments(
            value,
            case,
            solver=tmp_path / "solver",
            qualification_log=tmp_path / f"{case['case_id']}.json",
        )
        if case["surface_tension"] <= 0.0:
            continue
        assert option_values(arguments, "--capillary-force-form") == [
            "kinematic_area_gradient_traction"
        ]
        orders = option_values(arguments, "--interface-quadrature-order")
        assert len(orders) == 1
        assert int(orders[0]) >= 2


def test_conflicting_study_interface_order_override_is_rejected():
    value = copy.deepcopy(registry())
    value["studies"][0]["arguments"].extend(
        ["--interface-quadrature-order", "3"]
    )
    with pytest.raises(runner.MatrixError, match="interface quadrature override"):
        runner.validate_contract(value)


def test_positive_phi_scaling_executes_prescribed_wall_maintenance(tmp_path):
    value = registry()
    contract = value["maintenance_contract"]
    assert contract["prescribed_wall_maintenance"]["enabled"] is True
    assert contract["prescribed_wall_maintenance"]["execution_stage"] == (
        "accepted_endpoint_projection_reinitialization"
    )
    assert contract["prescribed_wall_maintenance"]["fsr04_closure_evidence"] is False
    scaling = [
        study
        for study in value["studies"]
        if study["refinement_axis"] == "phi_scale"
    ]
    assert {study["dimension"] for study in scaling} == {2, 3}
    for study in scaling:
        assert study["scope"]["fsr04_closure_evidence"] is False
        cases = [
            case
            for case in runner.expand_cases(value)
            if case["study_id"] == study["id"]
        ]
        for case in cases:
            arguments = runner.physical_case_arguments(
                value,
                case,
                solver=tmp_path / "solver",
                qualification_log=tmp_path / f"{case['case_id']}.json",
            )
            assert arguments.count("--enable-level-set-reinitialization") == 1
            assert option_values(arguments, "--sessile-contact-line-model") == [
                "prescribed"
            ]
            assert option_values(arguments, "--reinitialization-cadence-steps") == [
                "1"
            ]
            assert case["step_count"] >= 1


def test_time_refinement_preserves_one_positive_physical_horizon(tmp_path):
    value = registry()
    cases = runner.expand_cases(value)
    for study in value["studies"]:
        if study["refinement_axis"] != "time_step":
            continue
        selected = [case for case in cases if case["study_id"] == study["id"]]
        horizons = {
            float(case["level"]["value"]) * case["step_count"]
            for case in selected
        }
        assert horizons == {study["physical_horizon"]}
        by_dt = {
            float(case["level"]["value"]): case["step_count"]
            for case in selected
        }
        ordered = sorted(by_dt, reverse=True)
        assert [by_dt[dt] for dt in ordered] == sorted(by_dt.values())
        for case in selected:
            arguments = runner.physical_case_arguments(
                value,
                case,
                solver=tmp_path / "solver",
                qualification_log=tmp_path / f"{case['case_id']}.json",
            )
            assert option_values(arguments, "--steps") == [str(case["step_count"])]


def test_bulk_redistance_has_its_own_axis_and_records_schedule_coupling():
    value = registry()
    contract = value["maintenance_contract"]
    assert contract["bulk_redistance"]["refinement_axis"] == (
        "bulk_redistance_cadence"
    )
    assert contract["schedules_independent"] is False
    assert contract["schedule_relationship"] == (
        "shared_projection_reinitialization_event_and_cadence"
    )
    scaling_ids = {
        study["id"]
        for study in value["studies"]
        if study["refinement_axis"] == "phi_scale"
    }
    cadence = [
        study
        for study in value["studies"]
        if study["refinement_axis"] == "bulk_redistance_cadence"
    ]
    assert len(cadence) == 2
    assert scaling_ids.isdisjoint({study["id"] for study in cadence})
    assert all(study["refinement_levels"] == [4, 2, 1] for study in cadence)


def test_every_expanded_case_fits_the_explicit_one_node_memory_bound():
    value = registry()
    cases = runner.expand_cases(value, include_conditional_level=True)
    runner.validate_case_resources(value, cases)
    assert value["resources"]["partition"] == "amarsden"
    assert value["resources"]["maximum_concurrent_nodes"] == 4
    assert value["resources"]["maximum_total_memory_mib"] == 40960
    assert value["resources"]["nodes_per_case"] == 1
    assert value["resources"]["memory_mib_per_node"] == 10240
    estimates = [case["estimated_memory_mib"] for case in cases]
    assert max(estimates) <= value["resources"]["memory_mib_per_node"]
    assert max(
        case["resolution"] for case in cases if case["dimension"] == 3
    ) == 72

    model = value["resources"]["memory_model"]
    assert model["generated_vertex_formula"] == "(resolution + 1)^dimension"
    assert model["simplex_count_by_dimension"] == {"2": 2, "3": 6}
    assert model["coupled_unknown_components_by_dimension"] == {"2": 4, "3": 5}
    assert model["sparse_operator_copies"] >= 2
    assert model["field_vector_copies"] >= 1

    conditional = value["refinement"]["conditional_level_by_dimension"]
    assert conditional["2"] == {
        "cells_per_radius": 64,
        "availability": "AVAILABLE",
        "disposition_when_required": "EXECUTE",
    }
    assert conditional["3"] == {
        "cells_per_radius": 64,
        "availability": "UNAVAILABLE_ONE_NODE_MEMORY_LIMIT",
        "disposition_when_required": "INCONCLUSIVE",
    }
    assert all(
        case["level"]["value"] != 64.0
        for case in cases
        if case["dimension"] == 3
    )

    oversized = {
        "case_id": "hypothetical-three-dimensional-rdx-64",
        "dimension": 3,
        "resolution": 144,
    }
    oversized["estimated_memory_mib"] = runner.estimate_case_memory_mib(value, oversized)
    assert oversized["estimated_memory_mib"] > value["resources"][
        "memory_mib_per_node"
    ]
    with pytest.raises(runner.MatrixError, match="exceeds one-node memory"):
        runner.validate_case_resources(value, [oversized])


def test_exact_categories_and_required_task_tests_are_present():
    value = registry()
    assert {group["id"] for group in value["exact_groups"]} == {
        "focused_algebra",
        "sampled_convergence",
        "minimized_equilibrium",
        "restoring_motion",
        "mpi_parity",
    }
    invocations = runner.exact_invocations(value)
    tests = {
        test for invocation in invocations for test in invocation["tests"]
    }
    required = {
        "LevelSetInterfaceDomain.PlanarPolygonQuadraticRuleIntegratesTetrahedralCuts",
        "LevelSetInterfaceLifecycle.LinearBackendDriverReportsSupportAndOrders",
        "LevelSetInterfaceLifecycle.BackendCapabilityReportsMilestoneContract",
        "ApplicationDriverLevelSetWorkflows.KinematicAreaGradientMaintenanceBindsTotalEnergyDeclaration",
        "ApplicationDriverLevelSetWorkflows.TotalEnergyTractionRuleValidatorFailsClosedBeforeProjection",
        "ApplicationDriverLevelSetWorkflowsMPI.TotalEnergyTractionRuleValidatorIsCollective",
        "MovingDomainPhysics.KinematicAreaGradientTractionIsEnergyAdjointOnQuadraticTetraCut",
        "ApplicationDriverLevelSetWorkflows.MinimizedCircleSphereAndSessileCapsMeetProductionCertificates",
        "ApplicationDriverLevelSetWorkflows.SampledCircleSphereAndSessileControlsConvergeWithGci",
        "ApplicationDriverLevelSetWorkflows.SampledSessileFiveAngleTransformMatrixReportsPhysicalObservables",
        "ApplicationDriverLevelSetWorkflows.MinimizedCapillaryStateHasVolumeOrthogonalRestoringResponse",
        "ApplicationDriverLevelSetWorkflowsMPI.MinimizedCurvedCapillaryParityAcrossTwoOwnershipLayouts",
    }
    assert required.issubset(tests)
    assert len(tests) == sum(len(item["tests"]) for item in invocations)
    assert all(
        set(invocation["tests"]) == set(invocation["property_gates"])
        for invocation in invocations
    )


def test_v3_exact_union_preserves_every_v2_test_and_property_gate():
    value = registry()
    parent = runner.read_json(runner.PARENT_REGISTRY_PATH)
    parent_gates = {
        test: group["property_gates"][test]
        for group in parent["exact_groups"]
        for test in group["tests"]
    }
    current = {
        test: invocation
        for invocation in runner.exact_invocations(value)
        for test in invocation["tests"]
    }
    assert set(parent_gates).issubset(current)
    for test, gates in parent_gates.items():
        current_gates = current[test]["property_gates"][test]
        assert all(gate in current_gates for gate in gates), test

    distributed = [
        invocation
        for invocation in runner.exact_invocations(value)
        if invocation["binary"] == "level_set_mpi"
    ]
    assert len(distributed) == 1
    assert distributed[0]["mpi_ranks"] == 2
    assert distributed[0]["resource_profile"] == "mpi2_10gib"
    assert "level_set_mpi" in value["provenance_contract"][
        "required_binary_keys"
    ]
    assert distributed[0]["resource_profile"] in value["resources"]["profiles"]

    removed_test = copy.deepcopy(value)
    inherited_test = (
        "LevelSetCurvatureProjection."
        "KinematicAreaGradientYoungWallIsNeutralAtRightAngle"
    )
    inherited_invocation = next(
        invocation
        for invocation in runner.exact_invocations(removed_test)
        if inherited_test in invocation["tests"]
    )
    category = next(
        group
        for group in removed_test["exact_groups"]
        if group["id"] == inherited_invocation["category_id"]
    )
    invocation = next(
        item
        for item in category["invocations"]
        if item["id"] == "linearcorner_capability_and_energy_gradient"
    )
    invocation["tests"].remove(inherited_test)
    invocation["property_gates"].pop(inherited_test)
    with pytest.raises(runner.MatrixError, match="V2 exact tests"):
        runner.validate_contract(removed_test)

    weakened_gate = copy.deepcopy(value)
    functional = next(
        invocation
        for invocation in weakened_gate["exact_groups"][0]["invocations"]
        if invocation["id"] == "polygon_and_functional_variation"
    )
    three_dimensional = (
        "FreeSurfaceGeometrySnapshot."
        "DiscreteFunctionalFirstVariationMatchesThreeDimensionalCentralDifference"
    )
    functional["property_gates"][three_dimensional].pop(0)
    with pytest.raises(runner.MatrixError, match="V2 exact property gates"):
        runner.validate_contract(weakened_gate)


def test_physical_studies_retain_geometry_angles_walls_signs_and_offsets():
    value = registry()
    by_id = studies_by_id(value)
    assert {
        "closed_circle_sampled_analytic",
        "closed_circle_discrete_minimizer",
        "closed_sphere_sampled_analytic",
        "closed_sphere_discrete_minimizer",
        "sessile_caps_2d_sampled_analytic",
        "sessile_caps_2d_discrete_minimizer",
        "sessile_caps_3d_sampled_analytic",
        "sessile_caps_3d_discrete_minimizer",
    }.issubset(by_id)
    for dimension in (2, 3):
        for suffix in ("sampled_analytic", "discrete_minimizer"):
            study = by_id[f"sessile_caps_{dimension}d_{suffix}"]
            assert set(study["axes"]["contact_angle"]) == {30, 60, 90, 120, 150}
            assert set(study["axes"]["wall"]) == runner.REQUIRED_WALLS[dimension]
            assert set(study["axes"]["active_domain"]) == {
                "LevelSetNegative",
                "LevelSetPositive",
            }
            assert len(study["axes"]["offset_h"]) >= 2
    assert value["refinement"]["spatial_levels_cells_per_radius"] == [8, 16, 32]
    assert value["refinement"]["conditional_spatial_level_cells_per_radius"] == 64
    assert value["refinement"]["conditional_level_trigger"] == (
        "nonmonotone_three_level_sequence_only"
    )


def test_finest_gates_are_not_weaker_than_the_predeclared_limits():
    finest = registry()["gates"]["finest_level"]
    assert finest["pressure_jump_relative_error"] <= 0.01
    assert finest["contact_angle_absolute_error_degrees"] <= 1.0
    assert finest["base_radius_relative_error"] <= 0.01
    assert finest["apex_height_relative_error"] <= 0.01
    assert finest["parasitic_capillary_number"] <= 1.0e-6


def test_expansion_and_hash_shards_are_deterministic_unique_and_exact():
    value = registry()
    first = runner.expand_cases(value)
    second = runner.expand_cases(value)
    assert first == second
    assert len({case["case_id"] for case in first}) == len(first)
    assert len({case["case_digest"] for case in first}) == len(first)
    shards = [
        runner.select_cases(first, shard_index=index, shard_count=7)
        for index in range(7)
    ]
    shard_ids = [{case["case_id"] for case in shard} for shard in shards]
    assert set().union(*shard_ids) == {case["case_id"] for case in first}
    assert all(
        shard_ids[left].isdisjoint(shard_ids[right])
        for left in range(len(shard_ids))
        for right in range(left + 1, len(shard_ids))
    )


def test_dry_manifest_binds_inputs_tools_dependencies_binaries_and_unions(tmp_path):
    value = registry()
    cases = runner.expand_cases(value)
    compiler = tmp_path / "compiler"
    mpi = tmp_path / "mpi"
    compiler.write_bytes(b"compiler-bytes")
    mpi.write_bytes(b"mpi-bytes")
    dependencies = {}
    for key in value["provenance_contract"]["required_dependency_keys"]:
        path = tmp_path / f"dependency-{key}"
        path.write_bytes(key.encode("utf-8"))
        dependencies[key] = path
    binaries = {}
    for key in value["provenance_contract"]["required_binary_keys"]:
        path = tmp_path / f"binary-{key}"
        path.write_bytes(("binary-" + key).encode("utf-8"))
        binaries[key] = path

    manifest = runner.build_dry_run_manifest(
        value,
        cases,
        source_commit="1" * 40,
        compiler=compiler,
        mpi=mpi,
        dependencies=dependencies,
        binaries=binaries,
        include_conditional_level=False,
    )
    assert manifest["matrix_sha256"] == runner.sha256_file(MATRIX_PATH)
    assert manifest["runner_sha256"] == runner.sha256_file(RUNNER_PATH)
    assert manifest["physical_runner_sha256"] == runner.sha256_file(
        runner.PHYSICAL_RUNNER
    )
    assert manifest["source_commit"] == "1" * 40
    assert manifest["compiler"]["sha256"] == hashlib.sha256(
        b"compiler-bytes"
    ).hexdigest()
    assert manifest["mpi"]["sha256"] == hashlib.sha256(b"mpi-bytes").hexdigest()
    assert set(manifest["dependencies"]) == set(dependencies)
    assert set(manifest["binaries"]) == set(binaries)

    expected_tests = sorted(
        {
            test
            for invocation in runner.exact_invocations(value)
            for test in invocation["tests"]
        }
    )
    assert manifest["required_test_union"] == expected_tests
    expected_resources = sorted(
        {study["resource_profile"] for study in value["studies"]}
        | {
            invocation["resource_profile"]
            for invocation in runner.exact_invocations(value)
        }
    )
    assert manifest["required_resource_union"] == expected_resources
    expected_artifacts = runner.expected_artifact_paths(value, cases)
    assert manifest["expected_artifact_union"] == expected_artifacts
    assert manifest["expected_artifact_count"] == len(expected_artifacts)
    canonical = json.dumps(
        expected_artifacts, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    assert manifest["expected_artifact_union_sha256"] == hashlib.sha256(
        canonical
    ).hexdigest()


def test_intrinsic_completion_gate_is_derived_from_gtest_status():
    test = "SyntheticSuite.Case"
    group = {
        "tests": [test],
        "property_gates": {
            test: [
                {
                    "property": "wp4_exact_test_completed",
                    "comparison": "equal",
                    "expected": 1,
                }
            ]
        },
    }
    payload = {
        "tests": 1,
        "failures": 0,
        "disabled": 0,
        "testsuites": [
            {
                "name": "SyntheticSuite",
                "testsuite": [
                    {
                        "name": "Case",
                        "classname": "SyntheticSuite",
                        "status": "RUN",
                        "result": "COMPLETED",
                    }
                ],
            }
        ],
    }
    accepted = runner.evaluate_exact_document(
        payload, group, roundoff_factor=256.0, context="accepted"
    )
    assert accepted["passed"] is True
    changed = copy.deepcopy(payload)
    changed["testsuites"][0]["testsuite"][0]["result"] = "SKIPPED"
    rejected = runner.evaluate_exact_document(
        changed, group, roundoff_factor=256.0, context="rejected"
    )
    assert rejected["passed"] is False


def test_physical_execution_adapter_uses_v3_argument_mapping(monkeypatch, tmp_path):
    value = registry()
    case = runner.expand_cases(value)[0]
    observed = {}

    def delegated(parent_registry, cases, **options):
        observed["arguments"] = runner._v2.physical_case_arguments(
            parent_registry,
            cases[0],
            solver=options["solver"],
            qualification_log=tmp_path / "qualification.json",
        )
        return {"selected_case_count": len(cases)}

    monkeypatch.setattr(runner, "_V2_RUN_PHYSICAL_CASES", delegated)
    result = runner.run_physical_cases(
        value,
        [case],
        solver=tmp_path / "solver",
        output_root=tmp_path / "output",
        rerun=False,
    )
    assert result == {"selected_case_count": 1}
    assert option_values(observed["arguments"], "--steps") == [
        str(case["step_count"])
    ]
    assert option_values(
        observed["arguments"], "--interface-quadrature-order"
    ) == ["2"]


def test_exact_execution_adapter_flattens_categories_and_injects_completion(
    monkeypatch, tmp_path
):
    value = registry()
    observed = {}

    def delegated(parent_registry, **options):
        del options
        observed["ids"] = [group["id"] for group in parent_registry["exact_groups"]]
        group = next(
            group
            for group in parent_registry["exact_groups"]
            if group["id"] == "focused_algebra--serial_traction_admission"
        )
        cases = []
        for test in group["tests"]:
            classname, name = test.rsplit(".", 1)
            cases.append(
                {
                    "name": name,
                    "classname": classname,
                    "status": "RUN",
                    "result": "COMPLETED",
                }
            )
        payload = {
            "tests": len(cases),
            "failures": 0,
            "disabled": 0,
            "testsuites": [
                {
                    "name": "ApplicationDriverLevelSetWorkflows",
                    "testsuite": cases,
                }
            ],
        }
        observed["evaluation"] = runner._v2.evaluate_exact_document(
            payload,
            group,
            roundoff_factor=256.0,
            context="delegated exact adapter",
        )
        return {"passed": observed["evaluation"]["passed"]}

    monkeypatch.setattr(runner, "_V2_RUN_EXACT_GROUPS", delegated)
    result = runner.run_exact_groups(
        value,
        binaries={},
        mpi_launcher=tmp_path / "mpiexec",
        mpi_launcher_mode="srun",
        output_root=tmp_path / "output",
    )
    assert result == {"passed": True}
    assert len(observed["ids"]) == len(runner.exact_invocations(value))
    assert len(set(observed["ids"])) == len(observed["ids"])
    assert all("--" in invocation_id for invocation_id in observed["ids"])
    assert observed["evaluation"]["passed"] is True


def test_analysis_adapter_translates_cadence_and_limits_closure(
    monkeypatch, tmp_path
):
    value = registry()
    expected = runner.expand_cases(value, include_conditional_level=True)
    observed = {}

    def delegated(parent_registry, **options):
        observed["study_axes"] = {
            study["id"]: study["refinement_axis"]
            for study in parent_registry["studies"]
            if study["id"].startswith("bulk_redistance_cadence_")
        }
        expanded = runner._v2.expand_cases(
            parent_registry,
            include_conditional_level=options["include_conditional_level"],
        )
        observed["case_count"] = len(expanded)
        observed["case_axes"] = {
            case["refinement_axis"]
            for case in expanded
            if case["study_id"].startswith("bulk_redistance_cadence_")
        }
        return {
            "passed": True,
            "disposition": {
                "fsr03_closed": True,
                "fsr04_closed": True,
                "wp4_closed": True,
                "q2_closed": True,
            },
        }

    monkeypatch.setattr(runner, "_V2_ANALYZE_EVIDENCE", delegated)
    summary = runner.analyze_evidence(
        value,
        roots=[],
        output_root=tmp_path / "analysis",
        include_conditional_level=True,
        exact_summary_path=None,
    )
    assert set(observed["study_axes"].values()) == {
        "reinitialization_cadence"
    }
    assert observed["case_count"] == len(expected)
    assert observed["case_axes"] == {"reinitialization_cadence"}
    assert summary["disposition"] == {
        "fsr03_closed": True,
        "fsr04_closed": False,
        "wp4_closed": False,
        "q2_closed": False,
    }


def test_unavailable_three_dimensional_conditional_level_is_inconclusive(
    monkeypatch, tmp_path
):
    value = registry()

    def delegated(parent_registry, **options):
        del parent_registry, options
        return {
            "passed": False,
            "errors": ["convergence disposition is ADDITIONAL_LEVEL_REQUIRED"],
            "convergence": {
                "status": "ADDITIONAL_LEVEL_REQUIRED",
                "studies": {
                    "closed_sphere_sampled_analytic": {
                        "status": "ADDITIONAL_LEVEL_REQUIRED"
                    }
                },
            },
        }

    monkeypatch.setattr(runner, "_V2_ANALYZE_EVIDENCE", delegated)
    summary = runner.analyze_evidence(
        value,
        roots=[],
        output_root=tmp_path / "analysis",
        include_conditional_level=True,
        exact_summary_path=None,
    )
    assert summary["qualification_outcome"] == "INCONCLUSIVE"
    assert summary["conditional_level_dispositions"] == [
        {
            "study_id": "closed_sphere_sampled_analytic",
            "dimension": 3,
            "trigger": "nonmonotone_three_level_sequence_only",
            "availability": "UNAVAILABLE_ONE_NODE_MEMORY_LIMIT",
            "disposition": "INCONCLUSIVE",
        }
    ]
    assert not any(summary["disposition"].values())


def test_analysis_failure_takes_precedence_over_conditional_inconclusive(
    monkeypatch, tmp_path
):
    value = registry()

    def delegated(parent_registry, **options):
        del parent_registry, options
        return {
            "passed": False,
            "errors": ["convergence disposition is FAIL"],
            "exact_groups_passed": True,
            "convergence": {
                "status": "FAIL",
                "studies": {
                    "closed_sphere_sampled_analytic": {
                        "status": "ADDITIONAL_LEVEL_REQUIRED"
                    },
                    "closed_circle_sampled_analytic": {"status": "FAIL"},
                },
            },
            "invariance": {"status": "PASS"},
            "finest_level": {"status": "PASS"},
        }

    monkeypatch.setattr(runner, "_V2_ANALYZE_EVIDENCE", delegated)
    summary = runner.analyze_evidence(
        value,
        roots=[],
        output_root=tmp_path / "analysis",
        include_conditional_level=True,
        exact_summary_path=None,
    )
    assert summary["conditional_level_dispositions"][0]["disposition"] == (
        "INCONCLUSIVE"
    )
    assert summary["qualification_outcome"] == "FAIL"
    assert not any(summary["disposition"].values())


def test_available_conditional_requirement_precedes_unavailable_inconclusive(
    monkeypatch, tmp_path
):
    value = registry()

    def delegated(parent_registry, **options):
        del parent_registry, options
        return {
            "passed": False,
            "errors": ["convergence disposition is ADDITIONAL_LEVEL_REQUIRED"],
            "exact_groups_passed": True,
            "convergence": {
                "status": "ADDITIONAL_LEVEL_REQUIRED",
                "studies": {
                    "closed_circle_sampled_analytic": {
                        "status": "ADDITIONAL_LEVEL_REQUIRED"
                    },
                    "closed_sphere_sampled_analytic": {
                        "status": "ADDITIONAL_LEVEL_REQUIRED"
                    },
                },
            },
            "invariance": {"status": "PASS"},
            "finest_level": {"status": "PASS"},
        }

    monkeypatch.setattr(runner, "_V2_ANALYZE_EVIDENCE", delegated)
    summary = runner.analyze_evidence(
        value,
        roots=[],
        output_root=tmp_path / "analysis",
        include_conditional_level=False,
        exact_summary_path=None,
    )
    dispositions = {
        item["dimension"]: item["disposition"]
        for item in summary["conditional_level_dispositions"]
    }
    assert dispositions == {2: "EXECUTE", 3: "INCONCLUSIVE"}
    assert summary["qualification_outcome"] == "ADDITIONAL_LEVEL_REQUIRED"
    assert not any(summary["disposition"].values())
