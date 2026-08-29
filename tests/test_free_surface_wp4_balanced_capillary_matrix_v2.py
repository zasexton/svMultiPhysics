import importlib.util
import json
import sys
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parent
    / "cases/fluid/run_free_surface_wp4_balanced_capillary_matrix_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "free_surface_wp4_balanced_capillary_matrix_v2", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def registry():
    return runner.load_registry()


def test_v2_registry_is_frozen_and_complete_before_execution():
    value = registry()
    assert value["schema_version"] == 2
    assert value["status"] == "FROZEN_BEFORE_EXECUTION"
    assert value["closure_policy"]["requires_every_required_case"] is True
    assert value["model_envelope"]["force_projection_applied"] is False
    assert value["model_envelope"]["two_phase_claimed"] is False
    assert len(value["exact_groups"]) == 5
    assert len(value["studies"]) == 19


def test_v2_registry_byte_drift_is_rejected(tmp_path):
    changed = tmp_path / "changed.json"
    changed.write_text(
        runner.DEFAULT_REGISTRY.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.MatrixError, match="registry bytes changed"):
        runner.validate_registry(runner.read_json(changed), changed)


def test_v2_case_expansion_is_unique_and_deterministic():
    value = registry()
    first = runner.expand_cases(value)
    second = runner.expand_cases(value)
    conditional = runner.expand_cases(
        value, include_conditional_level=True)
    assert first == second
    assert len(first) == 2352
    assert len(conditional) == 2896
    assert len({case["case_id"] for case in first}) == len(first)
    assert len({case["case_digest"] for case in first}) == len(first)
    assert {case["case_id"] for case in first}.issubset(
        {case["case_id"] for case in conditional})


def test_resolution_levels_use_radius_over_mesh_spacing():
    value = registry()
    cases = runner.expand_cases(value, include_conditional_level=True)
    for case in cases:
        if case["refinement_axis"] != "resolution":
            continue
        target = float(case["level"]["value"])
        assert case["radius_over_h"] >= target
        assert case["radius_over_h"] < target + 2.0
    sphere = [
        case for case in cases
        if case["study_id"] == "closed_sphere_sampled_analytic" and
        case["axes"]["active_domain"] == "LevelSetNegative" and
        case["axes"]["offset_h"] == [0.0, 0.0, 0.0]
    ]
    assert [case["resolution"] for case in sphere] == [18, 36, 72, 144]
    assert [case["radius_over_h"] for case in sphere] == pytest.approx(
        [8.1, 16.2, 32.4, 64.8])


def test_main_sessile_studies_cover_angles_walls_signs_and_offsets():
    value = registry()
    for dimension in (2, 3):
        for suffix in ("sampled_analytic", "discrete_minimizer"):
            study_id = f"sessile_caps_{dimension}d_{suffix}"
            study = next(
                item for item in value["studies"] if item["id"] == study_id)
            assert set(study["axes"]["contact_angle"]) == {30, 60, 90, 120, 150}
            assert set(study["axes"]["wall"]) == runner.REQUIRED_WALLS[dimension]
            assert set(study["axes"]["active_domain"]) == {
                "LevelSetNegative", "LevelSetPositive"}
            assert len(study["axes"]["offset_h"]) >= 2


def test_case_arguments_map_static_state_and_three_dimensional_axes(tmp_path):
    value = registry()
    cases = runner.expand_cases(value)
    selected = next(
        case for case in cases
        if case["study_id"] == "sessile_caps_3d_discrete_minimizer" and
        case["level"]["value"] == 16.0 and
        case["axes"] == {
            "contact_angle": 120,
            "wall": "wall_front",
            "active_domain": "LevelSetPositive",
            "offset_h": [0.23, -0.31],
        })
    arguments = runner.physical_case_arguments(
        value,
        selected,
        solver=tmp_path / "solver",
        qualification_log=tmp_path / "qualification.json",
    )
    assert "--initialize-discrete-static-capillary-equilibrium" in arguments
    assert "--initialize-static-compatible-pressure" not in arguments
    assert arguments[arguments.index("--sessile-contact-wall-3d") + 1] == (
        "wall_front")
    assert arguments[arguments.index("--level-set-active-domain") + 1] == (
        "LevelSetPositive")
    assert arguments[arguments.index("--synthetic-nz") + 1] == str(
        selected["resolution"])
    offset_index = arguments.index("--sessile-tangent-center-offset-3d")
    assert list(map(float, arguments[offset_index + 1:offset_index + 3])) == (
        pytest.approx([0.23 * selected["h"], -0.31 * selected["h"]]))

    sampled = next(
        case for case in cases
        if case["study_id"] == "closed_circle_sampled_analytic")
    sampled_arguments = runner.physical_case_arguments(
        value,
        sampled,
        solver=tmp_path / "solver",
        qualification_log=tmp_path / "sampled.json",
    )
    assert "--initialize-static-compatible-pressure" in sampled_arguments
    assert "--initialize-discrete-static-capillary-equilibrium" not in (
        sampled_arguments)


def test_hash_shards_are_disjoint_and_cover_selected_cases():
    cases = runner.expand_cases(registry())
    shards = [
        runner.select_cases(cases, shard_index=index, shard_count=7)
        for index in range(7)
    ]
    identifiers = [{case["case_id"] for case in shard} for shard in shards]
    assert set.union(*identifiers) == {case["case_id"] for case in cases}
    for left in range(len(identifiers)):
        for right in range(left + 1, len(identifiers)):
            assert identifiers[left].isdisjoint(identifiers[right])


def test_literature_planar_mapping_is_exact_and_not_a_curved_arc():
    value = registry()
    adaptations = {
        item["id"]: item for item in value["literature_adaptations"]}
    planar = adaptations["gross_reusken_planar_force"]
    assert planar["adapted_evidence_group"] == "area_gradient_geometry_and_energy"
    assert planar["adapted_test"].endswith(
        "KinematicAreaGradientIsRoundoffBalancedForAffineFlatInterface")
    assert all(study["case"] != "capillaryarc2d" for study in value["studies"])
    sphere = adaptations["gross_reusken_spherical_force"]
    assert sphere["published_parameters"]["pressure_jump"] == 3.0
    assert sphere["adapted_parameters"]["pressure_jump"] == 3.0
    assert sphere["scale_mapping"]["length_scale"] == 0.45
    sessile = adaptations["reusken_stationary_sessile"]
    assert sessile["published_parameters"]["pressure_jump"] == 10.0
    assert sessile["adapted_parameters"]["pressure_jump"] == 10.0
    assert sessile["adapted_parameters"]["viscosity"] == 0.1


def test_metric_extraction_uses_matrix_owned_static_observables():
    case = {
        "case_id": "synthetic",
        "dimension": 3,
        "radius": 0.4,
        "surface_tension": 0.6,
    }
    probe = {
        "capillary_final_pressure_jump": 3.03,
        "diagnostic_free_surface_pressure_representability_relative_residual": (
            2.0e-10),
        "diagnostic_free_surface_conservative_balance_normalized_imbalance": (
            3.0e-10),
        "spatial_capillary_final_max_liquid_speed": 4.0e-7,
        "spatial_capillary_final_liquid_volume": 4.0 * 3.141592653589793 * (
            0.4 ** 3) / 3.0,
        "benchmark": {"viscosity": 0.1},
        "free_surface_energy_history": [
            {"kinetic_energy_proxy": 0.0},
            {"kinetic_energy_proxy": 5.0e-14},
        ],
    }
    assert runner.extract_metric(
        "pressure_jump_relative_error", probe, case) == pytest.approx(0.01)
    assert runner.extract_metric(
        "pressure_space_relative_distance", probe, case) == 2.0e-10
    assert runner.extract_metric(
        "conservative_balance_normalized_imbalance", probe, case) == 3.0e-10
    assert runner.extract_metric(
        "parasitic_capillary_number", probe, case) == pytest.approx(
            0.1 * 4.0e-7 / 0.6)
    assert runner.extract_metric(
        "kinetic_energy_proxy", probe, case) == 5.0e-14
    assert runner.extract_metric(
        "liquid_volume_relative_error", probe, case) == pytest.approx(0.0)


def test_registry_has_no_duplicate_json_keys():
    payload = runner.DEFAULT_REGISTRY.read_text(encoding="utf-8")
    decoded = json.loads(payload, object_pairs_hook=runner._strict_object)
    assert decoded["matrix_id"] == runner.EXPECTED_MATRIX_ID


def _synthetic_exact_group():
    test = "SyntheticSuite.QuantitativeCase"
    return {
        "tests": [test],
        "property_gates": {
            test: [
                {"property": "count", "comparison": "equal", "expected": 2},
                {"property": "upper", "comparison": "at_most", "expected": 0.5},
                {"property": "lower", "comparison": "at_least", "expected": 3},
                {"property": "diagnostic", "comparison": "finite"},
                {"property": "residual", "comparison": "scaled_roundoff",
                 "scale": 1.0},
            ],
        },
    }


def _synthetic_exact_payload():
    return {
        "tests": 1,
        "failures": 0,
        "disabled": 0,
        "testsuites": [{
            "name": "SyntheticSuite",
            "testsuite": [{
                "name": "QuantitativeCase",
                "classname": "SyntheticSuite",
                "status": "RUN",
                "result": "COMPLETED",
                "count": "2",
                "upper": "0.5",
                "lower": "3.25",
                "diagnostic": "-4.5",
                "residual": str(100.0 * sys.float_info.epsilon),
            }],
        }],
    }


def test_exact_document_requires_names_completion_and_quantitative_gates():
    evaluation = runner.evaluate_exact_document(
        _synthetic_exact_payload(),
        _synthetic_exact_group(),
        roundoff_factor=256.0,
        context="synthetic exact document",
    )
    assert evaluation["passed"] is True
    assert all(
        gate["passed"]
        for gates in evaluation["property_gates"].values()
        for gate in gates
    )

    changed = _synthetic_exact_payload()
    changed["testsuites"][0]["testsuite"][0]["result"] = "SKIPPED"
    changed["testsuites"][0]["testsuite"][0]["residual"] = str(
        300.0 * sys.float_info.epsilon)
    rejected = runner.evaluate_exact_document(
        changed,
        _synthetic_exact_group(),
        roundoff_factor=256.0,
        context="rejected exact document",
    )
    assert rejected["passed"] is False
    assert any("did not complete" in error for error in rejected["errors"])
    assert any("residual" in error for error in rejected["errors"])


def test_exact_rank_properties_must_match_and_cover_every_rank():
    first = {"properties": {"SyntheticSuite.QuantitativeCase": {"count": "2"}}}
    second = json.loads(json.dumps(first))
    assert runner.exact_rank_properties_identical([first, second], 2)
    second["properties"]["SyntheticSuite.QuantitativeCase"]["count"] = "3"
    assert not runner.exact_rank_properties_identical([first, second], 2)
    assert not runner.exact_rank_properties_identical([first], 2)


def test_srun_exact_launcher_disables_inherited_cpu_binding():
    arguments = runner.exact_mpi_launcher_arguments("srun", 2)
    assert arguments == [
        "--overlap",
        "--nodes=1",
        "--ntasks",
        "2",
        "--cpus-per-task=1",
        "--cpu-bind=none",
    ]
    assert runner.exact_mpi_launcher_arguments("mpiexec", 3) == [
        "--oversubscribe", "-n", "3"]
    with pytest.raises(runner.MatrixError, match="launcher mode"):
        runner.exact_mpi_launcher_arguments("invalid", 2)


def test_missing_evidence_produces_failed_analyses_without_an_exception():
    value = registry()
    expected = runner.expand_cases(value)
    convergence = runner.analyze_convergence(value, [], expected)
    invariance = runner.analyze_invariance(value, [], expected)
    finest = runner.analyze_finest_level(value, [], expected)
    assert convergence["status"] == "FAIL"
    assert invariance["status"] == "FAIL"
    assert finest["status"] == "FAIL"
