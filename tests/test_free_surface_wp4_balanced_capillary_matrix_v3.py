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
SPHERE_TRIGGER = {
    "study_id": "closed_sphere_sampled_analytic",
    "axes": {
        "active_domain": "LevelSetNegative",
        "offset_h": [0.0, 0.0, 0.0],
    },
}
ADDITIONAL = "ADDITIONAL_LEVEL_REQUIRED"


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


def additional_branch(key, value):
    return {"status": ADDITIONAL, key: value}


def write_prior_analysis(tmp_path, value, *specifications):
    tmp_path.mkdir(parents=True, exist_ok=True)
    preflight = tmp_path / "pre_execution_manifest.json"
    if not preflight.exists():
        runner.write_json(preflight, {"fixture": "frozen provenance"})
    base_cases = runner.expand_cases(value)
    studies = {}
    for specification in specifications or ({},):
        study_id = specification.get(
            "study_id", "closed_circle_sampled_analytic"
        )
        metric = specification.get("metric", "pressure_jump_relative_error")
        axes = specification.get("axes") or {
            "active_domain": "LevelSetNegative",
            "offset_h": [0.0] * (3 if "sphere" in study_id else 2),
        }
        selected = sorted(
            (
                case
                for case in base_cases
                if case["study_id"] == study_id and case["axes"] == axes
            ),
            key=lambda case: float(case["level"]["value"]),
        )
        assert [case["level"]["value"] for case in selected] == [8.0, 16.0, 32.0]
        canonical = lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":")
        )
        group_key = canonical(
            {key: item for key, item in axes.items() if key != "offset_h"}
        )
        samples = [
            {"label": case["level"]["label"], "h": case["h"], "value": observed}
            for case, observed in zip(selected, [0.2, 0.3, 0.25])
        ]
        sequence = {
            "status": ADDITIONAL,
            "sample_count": 3,
            "samples": samples,
            "monotone_to_reference": False,
            "gate_failures": ["asymptotic_tail_not_established"],
        }
        sequences = {canonical(axes["offset_h"]): sequence}
        metrics = {metric: additional_branch("sequences", sequences)}
        groups = {group_key: additional_branch("metrics", metrics)}
        studies[study_id] = additional_branch("groups", groups)
    analysis = {
        "schema_version": 1,
        "matrix_id": value["matrix_id"],
        "registry_sha256": runner.sha256_file(MATRIX_PATH),
        "runner_sha256": runner.sha256_file(RUNNER_PATH),
        "physical_runner_sha256": runner.sha256_file(runner.PHYSICAL_RUNNER),
        "pre_execution_manifest_sha256": runner.sha256_file(preflight),
        "expected_case_count": len(base_cases),
        "conditional_trigger_record_sha256": None,
        "exact_groups_passed": True,
        "invariance": {"status": "PASS"},
        "finest_level": {"status": "PASS"},
        "errors": [f"convergence disposition is {ADDITIONAL}"],
        "qualification_outcome": ADDITIONAL,
        "convergence": {"status": ADDITIONAL, "studies": studies},
    }
    path = tmp_path / "summary.json"
    runner.write_json(path, analysis)
    return path


def write_trigger_record(tmp_path, value, analysis_path):
    path = tmp_path / "conditional_trigger_record.json"
    runner.write_json(
        path, runner.build_conditional_trigger_record(value, analysis_path)
    )
    return path


def detached_source_record(tmp_path, commit="1" * 40):
    return {
        "source_root": str(tmp_path.resolve()),
        "git_top_level": str(tmp_path.resolve()),
        "head_commit": commit,
        "head_tree": "2" * 40,
        "head_detached": True,
        "worktree_clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "tracked_path_count": 7,
        "tracked_source_digest_semantics": "git_ls_files_stage_z_sha256",
        "tracked_source_sha256": "3" * 64,
        "lfs": {
            "fsck_passed": True,
            "tracked_object_count": 955,
            "missing_object_count": 0,
            "pointer_checkout_count": 0,
        },
    }


def bound_files(tmp_path, names, prefix=""):
    result = {name: tmp_path / f"{prefix}{name}" for name in names}
    for name, path in result.items():
        path.write_bytes(name.encode())
    return result


def provenance_inputs(tmp_path, value):
    provenance = value["provenance_contract"]
    return (
        bound_files(tmp_path, ("compiler", "mpi", "solver")),
        bound_files(tmp_path, provenance["required_dependency_keys"], "dependency-"),
        bound_files(tmp_path, provenance["required_binary_keys"], "binary-"),
    )


def build_manifest_fixture(monkeypatch, tmp_path, value):
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", source_root.resolve())
    files, dependencies, binaries = provenance_inputs(tmp_path, value)
    source_record = detached_source_record(source_root)
    monkeypatch.setattr(
        runner,
        "collect_source_provenance",
        lambda root: copy.deepcopy(source_record),
    )
    options = {
        "source_commit": "1" * 40,
        "source_root": source_root,
        "compiler": files["compiler"],
        "mpi": files["mpi"],
        "dependencies": dependencies,
        "binaries": binaries,
        "solver": files["solver"],
        "conditional_trigger_record_path": None,
    }
    cases = runner.expand_cases(value)
    return cases, runner.build_pre_execution_manifest(value, cases, **options), options, source_record


def analyze_fixture(
    monkeypatch, tmp_path, value, delegated_summary, *, trigger_path=None, prior_root=None
):
    monkeypatch.setattr(
        runner,
        "_V2_ANALYZE_EVIDENCE",
        lambda *args, **options: copy.deepcopy(delegated_summary),
    )
    output_root = tmp_path / "analysis"
    if prior_root is not None:
        output_root.mkdir()
        (output_root / "pre_execution_manifest.json").write_bytes(
            (prior_root / "pre_execution_manifest.json").read_bytes()
        )
    return runner.analyze_evidence(
        value,
        roots=[],
        output_root=output_root,
        conditional_trigger_record_path=trigger_path,
        exact_summary_path=None,
    ), output_root


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


@pytest.mark.parametrize(
    "levels,counts,match",
    [
        ([0.004, 0.004, 0.001], [1, 2, 4], "strictly decreasing"),
        ([0.004, 0.002, 0.001], [1, 1, 4], "strictly increasing"),
        ([0.001, 0.002, 0.004], [4, 2, 1], "strictly decreasing"),
    ],
)
def test_time_refinement_rejects_duplicate_or_reversed_levels_and_counts(
    levels, counts, match
):
    value = copy.deepcopy(registry())
    study = next(
        item for item in value["studies"] if item["refinement_axis"] == "time_step"
    )
    study["refinement_levels"] = levels
    study["level_step_counts"] = counts
    with pytest.raises(runner.MatrixError, match=match):
        runner.validate_contract(value)


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


def test_every_expanded_case_fits_the_explicit_one_node_memory_bound(tmp_path):
    value = registry()
    prior_path = write_prior_analysis(tmp_path, value)
    trigger_path = write_trigger_record(tmp_path, value, prior_path)
    loaded, binding = runner._conditional_trigger_binding(value, trigger_path)
    assert binding["prior_analysis_sha256"] == loaded["prior_analysis_sha256"]
    cases = runner.expand_cases(value, conditional_trigger_record=loaded)
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


def test_conditional_refinement_expands_only_hash_bound_triggered_sequence(tmp_path):
    value = registry()
    base = runner.expand_cases(value)
    assert len(base) == 2136
    assert not any(case["level"]["value"] == 64.0 for case in base)

    analysis_path = write_prior_analysis(tmp_path, value)
    record_path = write_trigger_record(tmp_path, value, analysis_path)
    loaded = runner.load_conditional_trigger_record(value, record_path)
    expanded = runner.expand_cases(value, conditional_trigger_record=loaded)

    conditional = [case for case in expanded if case["level"]["value"] == 64.0]
    assert len(expanded) == 2137
    assert len(conditional) == 1
    assert conditional[0]["study_id"] == "closed_circle_sampled_analytic"
    assert conditional[0]["axes"] == {
        "active_domain": "LevelSetNegative",
        "offset_h": [0.0, 0.0],
    }
    assert loaded["sequences"][0]["metric"] == "pressure_jump_relative_error"
    assert loaded["sequences"][0]["disposition"] == "EXECUTE"


def test_global_conditional_expansion_switch_is_not_supported():
    with pytest.raises(TypeError, match="include_conditional_level"):
        runner.expand_cases(registry(), include_conditional_level=True)


def test_conditional_trigger_rejects_incomplete_or_undeclared_sequence(tmp_path):
    value = registry()
    analysis_path = write_prior_analysis(tmp_path, value)
    record = runner.build_conditional_trigger_record(value, analysis_path)

    incomplete = copy.deepcopy(record)
    incomplete["sequences"][0].pop("sequence_id")
    with pytest.raises(runner.MatrixError, match="conditional trigger sequence"):
        runner.expand_cases(value, conditional_trigger_record=incomplete)

    undeclared = copy.deepcopy(record)
    undeclared["sequences"][0]["axes"]["active_domain"] = "undeclared"
    with pytest.raises(runner.MatrixError, match="conditional .*sequence|axes"):
        runner.expand_cases(value, conditional_trigger_record=undeclared)

    duplicated = copy.deepcopy(record)
    duplicated["sequences"].append(copy.deepcopy(duplicated["sequences"][0]))
    with pytest.raises(runner.MatrixError, match="duplicated"):
        runner.expand_cases(value, conditional_trigger_record=duplicated)

    empty = copy.deepcopy(record)
    empty["sequences"] = []
    with pytest.raises(runner.MatrixError, match="declares no sequences"):
        runner.expand_cases(value, conditional_trigger_record=empty)


def test_conditional_trigger_rejects_prior_analysis_with_actual_failure(tmp_path):
    value = registry()
    analysis_path = write_prior_analysis(tmp_path, value)
    analysis = json.loads(analysis_path.read_text())
    analysis["qualification_outcome"] = "FAIL"
    analysis["convergence"]["status"] = "FAIL"
    analysis["errors"].append("finest-level disposition is FAIL")
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    with pytest.raises(runner.MatrixError, match="actual failure"):
        runner.build_conditional_trigger_record(value, analysis_path)


def test_conditional_trigger_binds_prior_summary_bytes_and_rejects_drift(tmp_path):
    value = registry()
    analysis_path = write_prior_analysis(tmp_path, value)
    record_path = write_trigger_record(tmp_path, value, analysis_path)
    analysis_path.write_text(analysis_path.read_text() + "\n")
    with pytest.raises(runner.MatrixError, match="stale or malformed"):
        runner.load_conditional_trigger_record(value, record_path)


def test_unavailable_three_dimensional_trigger_is_terminal_not_expandable(
    tmp_path,
):
    value = registry()
    analysis_path = write_prior_analysis(tmp_path, value, SPHERE_TRIGGER)
    record = runner.build_conditional_trigger_record(value, analysis_path)
    assert len(record["sequences"]) == 1
    assert record["sequences"][0]["dimension"] == 3
    assert record["sequences"][0]["availability"] == "UNAVAILABLE_ONE_NODE_MEMORY_LIMIT"
    assert record["sequences"][0]["disposition"] == "INCONCLUSIVE"
    with pytest.raises(runner.MatrixError, match="no executable sequences"):
        runner.expand_cases(value, conditional_trigger_record=record)

    forged = copy.deepcopy(record)
    forged["sequences"][0]["availability"] = "AVAILABLE"
    forged["sequences"][0]["disposition"] = "EXECUTE"
    with pytest.raises(runner.MatrixError, match="conditional trigger sequence"):
        runner.expand_cases(value, conditional_trigger_record=forged)


def test_pre_execution_manifest_binds_all_inputs_and_unions(monkeypatch, tmp_path):
    value = registry()
    cases, manifest, options, _ = build_manifest_fixture(
        monkeypatch, tmp_path, value
    )
    assert manifest["matrix_sha256"] == runner.sha256_file(MATRIX_PATH)
    assert manifest["runner_sha256"] == runner.sha256_file(RUNNER_PATH)
    assert manifest["physical_runner_sha256"] == runner.sha256_file(
        runner.PHYSICAL_RUNNER
    )
    assert manifest["source_commit"] == "1" * 40
    assert manifest["compiler"]["sha256"] == hashlib.sha256(b"compiler").hexdigest()
    assert manifest["mpi"]["sha256"] == hashlib.sha256(b"mpi").hexdigest()
    assert manifest["solver"]["sha256"] == hashlib.sha256(b"solver").hexdigest()
    assert manifest["source"]["tracked_source_digest_semantics"] == (
        "git_ls_files_stage_z_sha256"
    )
    assert manifest["conditional_trigger"] is None
    assert set(manifest["dependencies"]) == set(options["dependencies"])
    assert set(manifest["binaries"]) == set(options["binaries"])

    invocations = runner.exact_invocations(value)
    assert manifest["required_test_union"] == sorted(
        {test for invocation in invocations for test in invocation["tests"]}
    )
    assert manifest["required_resource_union"] == sorted(
        {study["resource_profile"] for study in value["studies"]}
        | {invocation["resource_profile"] for invocation in invocations}
    )
    expected_artifacts = runner.expected_artifact_paths(value, cases)
    assert manifest["expected_artifact_union"] == expected_artifacts
    assert manifest["expected_artifact_count"] == len(expected_artifacts)
    assert manifest["expected_artifact_union_sha256"] == runner._canonical_sha256(
        expected_artifacts
    )


def test_execution_contract_binds_source_solver_launcher_and_named_artifacts():
    value = registry()
    provenance = value["provenance_contract"]
    assert provenance["source_worktree_requires_clean"] is True
    assert provenance["source_head_requires_detached"] is True
    assert provenance["source_head_must_equal_declared_commit"] is True
    assert provenance["tracked_source_digest_required"] is True
    assert provenance["required_missing_lfs_object_count"] == 0
    assert provenance["required_lfs_tracked_object_count"] == 955
    assert provenance["solver_hash_required"] is True
    assert provenance["mpi_launcher_must_match_bound_executable"] is True
    assert provenance["pre_execution_manifest_inside_output_root"] is True
    assert {
        "source_tree",
        "tracked_source",
        "solver",
        "conditional_trigger",
    }.issubset(provenance["required_hash_bindings"])

    execution = value["execution_contract"]
    assert execution == {
        "output_root_creation": "exclusive",
        "physical_retry_policy": "reject_nonempty_target",
        "rerun_allowed": False,
        "revalidate_before_each_numerical_action": True,
        "operational_setup_phase": "Task4B",
    }
    artifacts = value["artifact_contract"]
    assert artifacts["pre_execution_manifest_file"] == "pre_execution_manifest.json"
    assert artifacts["conditional_trigger_record_file"] == (
        "conditional_trigger_record.json"
    )
    paths = runner.expected_artifact_paths(value, runner.expand_cases(value))
    assert "pre_execution_manifest.json" in paths
    assert "conditional_trigger_record.json" in paths


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda record: record.update({"worktree_clean": False}), "clean"),
        (lambda record: record.update({"head_detached": False}), "detached"),
        (
            lambda record: record["lfs"].update(
                {"fsck_passed": False, "missing_object_count": 1}
            ),
            "LFS",
        ),
        (
            lambda record: record["lfs"].update({"pointer_checkout_count": 1}),
            "LFS",
        ),
        (
            lambda record: record["lfs"].update({"tracked_object_count": 0}),
            "LFS",
        ),
        (lambda record: record.update({"tracked_source_sha256": "bad"}), "digest"),
    ],
)
def test_source_provenance_rejects_nonqualified_checkout_state(
    monkeypatch, tmp_path, mutation, match
):
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path.resolve())
    record = detached_source_record(tmp_path)
    mutation(record)
    with pytest.raises(runner.MatrixError, match=match):
        runner.validate_source_provenance(record, declared_commit="1" * 40)


def test_source_provenance_rejects_an_unrelated_detached_checkout(tmp_path):
    record = detached_source_record(tmp_path / "unrelated")
    with pytest.raises(runner.MatrixError, match="running V3 repository"):
        runner.validate_source_provenance(record, declared_commit="1" * 40)


def test_source_provenance_rejects_false_zero_lfs_error(tmp_path):
    false_zero = runner.subprocess.CompletedProcess(
        ["git", "lfs", "fsck"],
        0,
        stdout=b"",
        stderr=b"Error: unknown flag: --objects\n",
    )
    empty = runner.subprocess.CompletedProcess(
        ["git", "lfs", "ls-files", "-l"], 0, stdout=b"", stderr=b""
    )
    with pytest.raises(runner.MatrixError, match="LFS"):
        runner._lfs_inventory(tmp_path, false_zero, empty)


@pytest.mark.parametrize("drift", ["solver", "mpi", "source_map", "source_tree"])
def test_pre_execution_manifest_revalidation_rejects_bound_input_drift(
    monkeypatch, tmp_path, drift
):
    value = registry()
    cases, manifest, options, source_record = build_manifest_fixture(
        monkeypatch, tmp_path, value
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest_path = output_root / "pre_execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assert manifest["mpi"]["path"] == str(options["mpi"].resolve())
    assert manifest["source"]["tracked_source_sha256"] == "3" * 64
    assert manifest["source"]["tracked_source_digest_semantics"] == (
        "git_ls_files_stage_z_sha256"
    )

    if drift == "solver":
        options["solver"].write_bytes(b"solver-two")
    elif drift == "mpi":
        options["mpi"].write_bytes(b"mpi-two")
    elif drift == "source_map":
        source_record["tracked_source_sha256"] = "5" * 64
    else:
        source_record["head_tree"] = "6" * 40
    with pytest.raises(runner.MatrixError, match="pre-execution manifest drift"):
        runner.revalidate_pre_execution_manifest(
            manifest_path,
            value,
            cases,
            output_root=output_root,
            **options,
        )


def test_pre_execution_manifest_must_remain_at_declared_output_path(
    monkeypatch, tmp_path
):
    value = registry()
    cases, manifest, options, _ = build_manifest_fixture(
        monkeypatch, tmp_path, value
    )
    declared_root = tmp_path / "declared"
    declared_root.mkdir()
    moved_root = tmp_path / "moved"
    moved_root.mkdir()
    moved = moved_root / "pre_execution_manifest.json"
    moved.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(runner.MatrixError, match="declared output-root"):
        runner.revalidate_pre_execution_manifest(
            moved,
            value,
            cases,
            output_root=declared_root,
            **options,
        )


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


@pytest.mark.parametrize(
    "attempt_state,rerun",
    [
        ("failed", False),
        ("timed-out", True),
        ("inconclusive", True),
        ("partial", True),
        ("successful", True),
    ],
)
def test_physical_execution_rejects_nonempty_case_target_without_mutation(
    monkeypatch, tmp_path, attempt_state, rerun
):
    value = registry()
    case = runner.expand_cases(value)[0]
    output_root = tmp_path / "output"
    case_directory = output_root / "cases" / case["case_id"]
    case_directory.mkdir(parents=True)
    sentinel = case_directory / "stdout.txt"
    original = f"{attempt_state}: preserved attempt\n".encode()
    sentinel.write_bytes(original)
    delegated = False

    def overwrite_attempt(*args, **options):
        nonlocal delegated
        del args, options
        delegated = True
        sentinel.write_bytes(b"replaced\n")
        return {}

    monkeypatch.setattr(runner, "_V2_RUN_PHYSICAL_CASES", overwrite_attempt)
    with pytest.raises(runner.MatrixError, match="immutable physical evidence"):
        runner.run_physical_cases(
            value,
            [case],
            solver=tmp_path / "solver",
            output_root=output_root,
            rerun=rerun,
        )
    assert delegated is False
    assert sentinel.read_bytes() == original


def test_exact_execution_adapter_flattens_categories_and_injects_completion(
    monkeypatch, tmp_path
):
    value = registry()
    observed = {}

    def delegated(parent_registry, **options):
        observed["mpi_launcher"] = options["mpi_launcher"]
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
        mpi=tmp_path / "mpiexec",
        mpi_launcher_mode="srun",
        output_root=tmp_path / "output",
    )
    assert result == {"passed": True}
    assert len(observed["ids"]) == len(runner.exact_invocations(value))
    assert len(set(observed["ids"])) == len(observed["ids"])
    assert all("--" in invocation_id for invocation_id in observed["ids"])
    assert observed["evaluation"]["passed"] is True
    assert observed["mpi_launcher"] == tmp_path / "mpiexec"


def test_cli_rejects_a_second_mpi_launcher_path():
    with pytest.raises(SystemExit):
        runner._parser().parse_args(
            ["--validate-only", "--mpi-launcher", "/different/launcher"]
        )


def test_cli_writes_manifest_only_at_a_fresh_declared_output_root(
    monkeypatch, tmp_path
):
    output_root = tmp_path / "attempt"
    manifest = {
        "expected_artifact_count": 7,
        "maximum_estimated_memory_mib": 9,
    }
    monkeypatch.setattr(
        runner,
        "build_pre_execution_manifest",
        lambda *args, **options: copy.deepcopy(manifest),
    )
    arguments = [
        "--dry-manifest",
        "--output-root",
        str(output_root),
        "--source-commit",
        "1" * 40,
        "--source-root",
        str(tmp_path),
        "--compiler",
        str(tmp_path / "compiler"),
        "--mpi",
        str(tmp_path / "mpi"),
        "--solver",
        str(tmp_path / "solver"),
    ]
    assert runner.main(arguments) == 0
    manifest_path = output_root / "pre_execution_manifest.json"
    preserved = manifest_path.read_bytes()
    with pytest.raises(runner.MatrixError, match="output root already exists"):
        runner.main(arguments)
    assert manifest_path.read_bytes() == preserved


@pytest.mark.parametrize("action", ["physical", "exact"])
def test_execution_adapters_guard_each_delegated_numerical_command(
    monkeypatch, tmp_path, action
):
    value = registry()
    guard_calls = []

    def parent_command(*args, **options):
        del args, options
        return {"returncode": 0, "timed_out": False}

    def delegated(*args, **options):
        del args, options
        runner._v2._run_command("first")
        runner._v2._run_command("second")
        return {"passed": True}

    monkeypatch.setattr(runner._v2, "_run_command", parent_command)
    if action == "physical":
        monkeypatch.setattr(runner, "_V2_RUN_PHYSICAL_CASES", delegated)
        runner.run_physical_cases(
            value,
            [runner.expand_cases(value)[0]],
            solver=tmp_path / "solver",
            output_root=tmp_path / "physical",
            rerun=False,
            pre_execution_guard=lambda: guard_calls.append(action),
        )
    else:
        monkeypatch.setattr(runner, "_V2_RUN_EXACT_GROUPS", delegated)
        runner.run_exact_groups(
            value,
            binaries={},
            mpi=tmp_path / "mpi",
            mpi_launcher_mode="srun",
            output_root=tmp_path / "exact",
            pre_execution_guard=lambda: guard_calls.append(action),
        )
    assert guard_calls == [action, action]


def test_analysis_adapter_translates_cadence_and_limits_closure(
    monkeypatch, tmp_path
):
    value = registry()
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    prior_path = write_prior_analysis(prior_root, value)
    trigger_path = write_trigger_record(prior_root, value, prior_path)
    trigger = runner.load_conditional_trigger_record(value, trigger_path)
    expected = runner.expand_cases(value, conditional_trigger_record=trigger)
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
            "errors": [],
            "exact_groups_passed": True,
            "convergence": {"status": "PASS", "studies": {}},
            "invariance": {"status": "PASS"},
            "finest_level": {"status": "PASS"},
            "disposition": {
                "fsr03_closed": True,
                "fsr04_closed": True,
                "wp4_closed": True,
                "q2_closed": True,
            },
        }

    monkeypatch.setattr(runner, "_V2_ANALYZE_EVIDENCE", delegated)
    output_root = tmp_path / "analysis"
    output_root.mkdir()
    (output_root / "pre_execution_manifest.json").write_bytes(
        (prior_root / "pre_execution_manifest.json").read_bytes()
    )
    summary = runner.analyze_evidence(
        value,
        roots=[],
        output_root=output_root,
        conditional_trigger_record_path=trigger_path,
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


def test_base_analysis_emits_hash_bound_sibling_trigger_record(
    monkeypatch, tmp_path
):
    value = registry()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    prior_path = write_prior_analysis(fixture_root, value)
    delegated_summary = json.loads(prior_path.read_text())

    summary, output_root = analyze_fixture(
        monkeypatch,
        tmp_path,
        value,
        delegated_summary,
        prior_root=fixture_root,
    )
    summary_path = output_root / "summary.json"
    trigger_path = output_root / "conditional_trigger_record.json"
    assert summary_path.is_file()
    assert trigger_path.is_file()
    trigger = runner.load_conditional_trigger_record(value, trigger_path)
    assert trigger["prior_analysis_file"] == "summary.json"
    assert trigger["prior_analysis_sha256"] == runner.sha256_file(summary_path)
    assert trigger["prior_pre_execution_manifest_file"] == (
        "pre_execution_manifest.json"
    )
    assert trigger["prior_pre_execution_manifest_sha256"] == summary[
        "pre_execution_manifest_sha256"
    ]
    assert trigger["sequences"][0]["sequence_id"] == summary[
        "conditional_level_dispositions"
    ][0]["sequence_id"]


@pytest.mark.parametrize(
    "specifications,actual_failure,expected_dispositions,outcome",
    [
        ((SPHERE_TRIGGER,), False, {3: "INCONCLUSIVE"}, "INCONCLUSIVE"),
        ((SPHERE_TRIGGER,), True, {3: "INCONCLUSIVE"}, "FAIL"),
        (
            ({}, SPHERE_TRIGGER),
            False,
            {2: "EXECUTE", 3: "INCONCLUSIVE"},
            ADDITIONAL,
        ),
    ],
    ids=("unavailable", "failure-dominates", "available-precedes-unavailable"),
)
def test_conditional_analysis_outcome_precedence(
    monkeypatch,
    tmp_path,
    specifications,
    actual_failure,
    expected_dispositions,
    outcome,
):
    value = registry()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    fixture_path = write_prior_analysis(fixture_root, value, *specifications)
    delegated = json.loads(fixture_path.read_text())
    if actual_failure:
        delegated["passed"] = False
        delegated["errors"].append("invariance disposition is FAIL")
        delegated["invariance"] = {"status": "FAIL"}
        delegated["qualification_outcome"] = "FAIL"
    summary, _ = analyze_fixture(
        monkeypatch,
        tmp_path,
        value,
        delegated,
        prior_root=fixture_root,
    )
    dispositions = {
        item["dimension"]: item["disposition"]
        for item in summary["conditional_level_dispositions"]
    }
    assert dispositions == expected_dispositions
    assert summary["qualification_outcome"] == outcome
    assert not any(summary["disposition"].values())


def test_targeted_analysis_resolves_available_sequence_and_keeps_3d_inconclusive(
    monkeypatch, tmp_path
):
    value = registry()
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    prior_path = write_prior_analysis(prior_root, value, {}, SPHERE_TRIGGER)
    trigger_path = write_trigger_record(prior_root, value, prior_path)
    original_trigger_bytes = trigger_path.read_bytes()
    current_path = write_prior_analysis(
        tmp_path / "current", value, {}, SPHERE_TRIGGER
    )
    current = json.loads(current_path.read_text())
    circle = current["convergence"]["studies"][
        "closed_circle_sampled_analytic"
    ]
    group = next(iter(circle["groups"].values()))
    metric = group["metrics"]["pressure_jump_relative_error"]
    offset = next(iter(metric["sequences"]))
    samples = metric["sequences"][offset]["samples"]
    for sample, result in zip(samples, (0.008, 0.004, 0.002)):
        sample["value"] = result
    samples.append(
        {"label": "rdx_64", "h": samples[-1]["h"] / 2.0, "value": 0.001}
    )
    gate = value["gates"]["convergence"]["pressure_jump_relative_error"]
    sequence = runner._v2._load_convergence_module().analyze_gci_sequence(
        samples,
        reference_value=gate["reference"],
        normalization=gate["normalization"],
        minimum_observed_order=value["refinement"]["minimum_observed_order"],
        finest_relative_error_limit=gate["finest_error_limit"],
        finest_gci_limit=gate["finest_gci_limit"],
        safety_factor=value["refinement"]["safety_factor"],
        ratio_relative_tolerance=value["refinement"]["ratio_relative_tolerance"],
    )
    metric["sequences"][offset] = sequence
    for item in (metric, group, circle):
        item["status"] = "PASS"
    assert sequence["status"] == "PASS"
    assert sequence["sample_count"] == 4
    assert sequence["asymptotic_tail_labels"] == ["rdx_16", "rdx_32", "rdx_64"]
    assert sequence["gate_failures"] == []
    current["expected_case_count"] = len(
        runner.expand_cases(
            value,
            conditional_trigger_record=runner.load_conditional_trigger_record(
                value, trigger_path
            ),
        )
    )

    summary, output_root = analyze_fixture(
        monkeypatch,
        tmp_path,
        value,
        current,
        trigger_path=trigger_path,
    )
    assert summary["qualification_outcome"] == "INCONCLUSIVE"
    assert [
        (item["study_id"], item["dimension"], item["disposition"])
        for item in summary["conditional_level_dispositions"]
    ] == [("closed_sphere_sampled_analytic", 3, "INCONCLUSIVE")]
    assert not (output_root / "conditional_trigger_record.json").exists()
    assert trigger_path.read_bytes() == original_trigger_bytes


def test_targeted_analysis_promotes_unresolved_four_level_sequence_to_fail(
    monkeypatch, tmp_path
):
    value = registry()
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    prior_path = write_prior_analysis(prior_root, value)
    trigger_path = write_trigger_record(prior_root, value, prior_path)
    current = json.loads(prior_path.read_text())
    study = next(iter(current["convergence"]["studies"].values()))
    group = next(iter(study["groups"].values()))
    metric = next(iter(group["metrics"].values()))
    sequence = next(iter(metric["sequences"].values()))
    sequence["samples"].append(
        {"label": "R64", "h": sequence["samples"][-1]["h"] / 2.0, "value": 0.27}
    )
    sequence["sample_count"] = 4
    sequence["status"] = "FAIL"
    for item in (metric, group, study, current["convergence"]):
        item["status"] = "FAIL"
    current["errors"] = ["convergence disposition is FAIL"]
    current["qualification_outcome"] = "FAIL"

    summary, _ = analyze_fixture(
        monkeypatch,
        tmp_path,
        value,
        current,
        trigger_path=trigger_path,
    )
    assert summary["conditional_level_dispositions"] == []
    assert summary["qualification_outcome"] == "FAIL"
    assert summary["disposition"] == {
        "fsr03_closed": False,
        "fsr04_closed": False,
        "wp4_closed": False,
        "q2_closed": False,
    }
