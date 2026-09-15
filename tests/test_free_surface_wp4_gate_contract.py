"""Synthetic controls for the prospective WP4 evidence contract."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).parent / "cases/fluid/run_free_surface_wp4_balanced_capillary_matrix_v3.py"
SPEC = importlib.util.spec_from_file_location("wp4_successor_controls", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def evaluator():
    function = getattr(runner, "assess_qualification_sequence", None)
    assert callable(function), "V3 must dispatch the successor sequence evaluator"
    return function


def contract(axis="resolution", initialization="sampled_analytic"):
    return {
        "initialization": initialization, "axis": axis,
        "quantity": "pressure_space_relative_distance", "phase": "accepted_endpoint",
        "dimension": 2, "boundary": "closed", "source": "synthetic_common_record",
        "norm": "synthetic_coefficient_l2", "units": "1", "normalization": 1.0,
        "justification": "Synthetic control only; not a scientific rate declaration.",
        "expected_order": 1.0, "minimum_order": 0.9,
        "order_tolerance": 0.2, "algebraic_floor": 1e-12,
        "floor_justification": "Synthetic exact arithmetic bound.",
        "target": 0.0, "algebraic_bound": 1e-8,
        "maximum_spread": 1e-8, "maximum_value": None,
        "reference": None, "similarity": None,
    }


def samples(values, axis="resolution"):
    return [
        {"level": level, "h": 1.0 / level, "value": value,
         "identity": {"record_id": str(level), "source": "synthetic_common_record",
                      "initialization": "sampled_analytic", "accepted_step": 4,
                      "accepted_time": 0.04, "measurement_phase": "accepted_endpoint",
                      "state_revision": str(level), "geometry_revision": str(level),
                      "maintenance_revision": "4", "topology_revision": "0",
                      "domain_identity": "negative"},
         "mesh_identity": "fixed-mesh", "geometry_equivalence": "same-surface",
         "initial_geometry_identity": "same-initial-surface",
         "physical_parameters": {"density": 1.0}, "maintenance_schedule": [1, 2, 3, 4],
         "event_count": 4, "physical_horizon": 0.04, "time_step": 0.01}
        for level, value in zip((8, 16, 32, 64), values)
    ]


@pytest.mark.parametrize("values,status", [
    ([0.125, 0.0625, 0.03125], "PASS"),
    ([0.225, 0.1625, 0.13125], "FAIL"),
    ([0.2, 0.3, 0.25], "ADDITIONAL_LEVEL_REQUIRED"),
    ([0.2, 0.3, 0.25, 0.27], "INCONCLUSIVE"),
    ([0.2, 0.1], "INCONCLUSIVE"),
])
def test_successor_spatial_target_and_required_levels(values, status):
    assert evaluator()(contract(), samples(values))["status"] == status


def test_successor_minimized_algebraic_floor_needs_no_positive_slope():
    declaration = contract(initialization="discrete_energy_minimized")
    rows = samples([1e-10, 1e-10, 1e-10])
    for row in rows:
        row["identity"]["initialization"] = "discrete_energy_minimized"
    result = evaluator()(declaration, rows)
    assert result["status"] == "PASS"
    assert result["basis"] == "algebraic_equilibrium"
    rows[-1]["value"] = 1e-7
    assert evaluator()(declaration, rows)["status"] == "FAIL"


@pytest.mark.parametrize("field", ["norm", "floor_justification", "justification", "minimum_order"])
def test_successor_missing_scientific_definition_blocks_readiness(field):
    declaration = contract()
    declaration[field] = None
    result = evaluator()(declaration, samples([0.125, 0.0625, 0.03125]))
    assert result["status"] == "INCONCLUSIVE"
    assert field in " ".join(result["reasons"])


@pytest.mark.parametrize("axis", ["time_step", "bulk_redistance_cadence"])
def test_successor_nonzero_fixed_mesh_reference(axis):
    declaration = contract(axis)
    declaration["reference"] = {
        "value": 0.1, "uncertainty": 1e-12,
        "justification": "Synthetic fixed-mesh reference.",
        "mesh_identity": "fixed-mesh", "physical_horizon": 0.04,
        "initial_geometry_identity": "same-initial-surface",
        "source": "synthetic_common_record", "phase": "accepted_endpoint",
        "norm": "synthetic_coefficient_l2", "units": "1", "normalization": 1.0,
    }
    rows = samples([0.1, 0.1, 0.1], axis)
    assert evaluator()(declaration, rows)["status"] == "PASS"
    rows[-1]["event_count"] = None
    assert evaluator()(declaration, rows)["status"] == "FAIL"


def test_successor_scaling_is_invariance_without_spatial_accuracy_claim():
    result = evaluator()(contract("phi_scale"), samples([0.1, 0.1, 0.1]))
    assert result["status"] == "PASS"
    assert result["independent_spatial_evidence_required"] is True
    assert result["equilibrium_certified"] is False
    declaration = contract("physical_scale")
    assert evaluator()(declaration, samples([0.1, 0.1, 0.1]))["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("field", ["source", "measurement_phase", "initialization"])
def test_successor_rejects_wrong_sequence_binding(field):
    rows = samples([0.125, 0.0625, 0.03125])
    rows[-1]["identity"][field] = "wrong"
    assert evaluator()(contract(), rows)["status"] == "FAIL"


@pytest.mark.parametrize("field", [
    "record_id", "source", "initialization", "accepted_step", "accepted_time",
    "measurement_phase", "state_revision", "geometry_revision", "maintenance_revision",
    "topology_revision", "domain_identity",
])
def test_successor_pairing_rejects_cross_state_join(field):
    function = getattr(runner, "bind_qualification_measurements", None)
    assert callable(function), "V3 must bind metrics from one shared measurement"
    declaration = contract()
    identity = samples([0.1])[0]["identity"]
    first = dict(identity, quantity=declaration["quantity"], value=0.1,
                 norm=declaration["norm"], units="1", normalization=1.0)
    second = dict(first, quantity="conservative_balance_normalized_imbalance")
    second[field] = 99 if field in {"accepted_step", "accepted_time"} else "other"
    with pytest.raises(runner.MatrixError, match="identity|binding"):
        function([first, second], [declaration, dict(declaration, quantity=second["quantity"])])


def test_successor_analysis_cannot_close_from_legacy_success(monkeypatch, tmp_path):
    registry = runner.load_registry()
    trigger_path = tmp_path / "trigger.json"
    trigger_path.write_text("{}")
    monkeypatch.setattr(runner, "load_conditional_trigger_record", lambda *a: {})
    monkeypatch.setattr(runner, "_V2_ANALYZE_EVIDENCE", lambda *a, **kw: {
        "passed": True, "errors": [], "exact_groups_passed": True,
        "convergence": {"status": "PASS", "studies": {}},
        "invariance": {"status": "PASS"}, "finest_level": {"status": "PASS"},
    })
    result = runner.analyze_evidence(registry, roots=[], output_root=tmp_path,
                                    conditional_trigger_record_path=trigger_path, exact_summary_path=None)
    assert result["passed"] is False, "legacy success must not qualify the successor contract"
    assert result["qualification_readiness"]["ready"] is False
    assert result["qualification_readiness"]["missing_contracts"]
    assert not any(result["disposition"].values())


def test_successor_cli_blocks_launch_with_undeclared_scientific_inputs():
    with pytest.raises(runner.MatrixError, match="missing scientific contracts"):
        runner.main(["--run-physical"])


@pytest.mark.parametrize("field,value", [
    ("axis", "unknown"), ("quantity", "unknown"), ("phase", "unknown"),
    ("initialization", "unknown"), ("dimension", 4), ("boundary", "unknown"),
])
def test_successor_unknown_dispatch_fails(field, value):
    declaration = contract()
    declaration[field] = value
    assert evaluator()(declaration, samples([0.125, 0.0625, 0.03125]))["status"] == "FAIL"


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_successor_does_not_repair_invalid_norm_values(value):
    rows = samples([0.125, 0.0625, value])
    assert evaluator()(contract(), rows)["status"] == "FAIL"


def test_successor_established_incompatible_rate_fails_after_required_refinement():
    rows = samples([level ** -0.5 for level in (8, 16, 32, 64)])
    assert evaluator()(contract(), rows[:3])["status"] == "ADDITIONAL_LEVEL_REQUIRED"
    assert evaluator()(contract(), rows)["status"] == "FAIL"


@pytest.mark.parametrize("field", ["mesh_identity", "geometry_equivalence", "physical_parameters", "maintenance_schedule", "physical_horizon"])
def test_successor_scaling_rejects_state_and_schedule_mismatch(field):
    rows = samples([0.1, 0.1, 0.1])
    rows[-1][field] = "different"
    assert evaluator()(contract("phi_scale"), rows)["status"] == "FAIL"


def test_successor_physical_similarity_transforms_quantity_before_spread():
    declaration = contract("physical_scale")
    declaration["similarity"] = {
        "justification": "Synthetic similarity control.",
        "geometric_invariants": {"aspect_ratio": 1}, "dynamic_invariants": {"capillary_number": 1},
        "norm_factors": {"8": 1.0, "16": 0.5, "32": 0.25},
    }
    rows = samples([0.1, 0.2, 0.4])
    for row in rows:
        for key in ("geometric_invariants", "dynamic_invariants"):
            row[key] = declaration["similarity"][key]
    assert evaluator()(declaration, rows)["status"] == "PASS"
    rows[-1]["dynamic_invariants"] = {"capillary_number": 2}
    assert evaluator()(declaration, rows)["status"] == "FAIL"


@pytest.mark.parametrize("field", ["norm", "units", "normalization", "source", "phase", "mesh_identity", "initial_geometry_identity", "physical_horizon"])
def test_successor_fixed_reference_binding_is_strict(field):
    declaration = contract("time_step")
    declaration["reference"] = {
        "value": 0.1, "uncertainty": 1e-12, "justification": "Synthetic reference.",
        "mesh_identity": "fixed-mesh", "initial_geometry_identity": "same-initial-surface", "physical_horizon": 0.04,
        "source": declaration["source"], "phase": declaration["phase"], "norm": declaration["norm"],
        "units": "1", "normalization": 1.0,
    }
    declaration["reference"][field] = "mismatch"
    assert evaluator()(declaration, samples([0.1, 0.1, 0.1]))["status"] == "FAIL"


@pytest.mark.parametrize("field", ["norm", "units", "normalization", "state_revision", "maintenance_revision", "accepted_time"])
def test_successor_pair_binding_requires_each_quantity_metadata(field):
    declaration = contract()
    row = dict(samples([0.1])[0]["identity"], quantity=declaration["quantity"], value=0.1,
               norm=declaration["norm"], units="1", normalization=1.0)
    del row[field]
    with pytest.raises(runner.MatrixError, match="identity|binding"):
        runner.bind_qualification_measurements([row], [declaration])


def analysis_fixture(tmp_path, monkeypatch, values, quantity="pressure_space_relative_distance", active_domain="LevelSetNegative"):
    value = runner.load_registry()
    cases = [case for case in runner.expand_cases(value)
             if case["study_id"] == "closed_circle_sampled_analytic"
             and case["axes"] == {"active_domain": "LevelSetNegative", "offset_h": [0.0, 0.0]}]
    declarations = [dict(contract(), quantity=quantity) for quantity in cases[0]["metrics"]]
    value["qualification_contract"]["scientific_contracts"] = declarations
    for case, residual in zip(cases, values):
        directory = tmp_path / "evidence" / "cases" / case["case_id"]
        directory.mkdir(parents=True)
        runner.write_json(directory / "case.json", case)
        identity = dict(samples([0.1])[0]["identity"], accepted_step=case["step_count"],
                        accepted_time=case["physical_horizon"])
        rows = [dict(identity, quantity=item["quantity"], value=residual if item["quantity"] == quantity else 0.0,
                     norm=item["norm"], units=item["units"], normalization=item["normalization"])
                for item in declarations]
        runner.write_json(directory / "qualification.json", {"complete": True, "probes": [{"passed": True, "errors": [],
                          "metrics": {"qualification_measurements": rows, "qualification_context": {"active_domain": active_domain}}}]})
    monkeypatch.setattr(runner, "_analysis_cases", lambda *a, **kw: cases)
    output = tmp_path / "analysis"
    output.mkdir()
    result = runner.analyze_evidence(value, roots=[tmp_path / "evidence"], output_root=output,
                                    conditional_trigger_record_path=None, exact_summary_path=None)
    return result, value, cases, output


@pytest.mark.parametrize("values,expected", [([0.125, 0.0625, 0.03125], "PASS"), ([0.225, 0.1625, 0.13125], "FAIL")])
def test_successor_real_analysis_reads_bound_quantities(tmp_path, monkeypatch, values, expected):
    result, _, _, _ = analysis_fixture(tmp_path, monkeypatch, values)
    study = result["convergence"]["studies"]["closed_circle_sampled_analytic"]
    assert study["status"] == expected
    assert result["accepted_case_count"] == 3
    assert result["physical_records"][0]["metrics"]["pressure_space_relative_distance"] == values[0]
    assert not result["passed"]
    assert not result["qualification_readiness"]["ready"]


def test_successor_real_analysis_preserves_physical_gci_limit(tmp_path, monkeypatch):
    result, _, _, _ = analysis_fixture(tmp_path, monkeypatch, [0.036, 0.018, 0.009], quantity="pressure_jump_relative_error")
    study = result["convergence"]["studies"]["closed_circle_sampled_analytic"]
    assert study["status"] == "FAIL"
    assert result["finest_level"]["status"] == "PASS"
    sequence = next(iter(next(iter(study["groups"].values()))["metrics"]["pressure_jump_relative_error"]["sequences"].values()))
    assert sequence["grid_uncertainty"] == pytest.approx(0.01125)
    assert "grid uncertainty" in " ".join(sequence["reasons"])


def test_successor_real_analysis_rejects_paired_record_from_wrong_active_domain(tmp_path, monkeypatch):
    result, _, _, _ = analysis_fixture(tmp_path, monkeypatch, [0.125, 0.0625, 0.03125], active_domain="LevelSetPositive")
    assert result["accepted_case_count"] == 0
    assert result["qualification_outcome"] == "FAIL"
    assert any("active domain" in error for error in result["errors"])


@pytest.mark.parametrize("axis", ["phi_scale", "physical_scale"])
def test_successor_registry_rejects_weakened_scaling_spread(axis):
    value = runner.load_registry()
    study = next(study for study in value["studies"] if study["refinement_axis"] == axis)
    declaration = dict(contract(axis), initialization=study["initialization"], dimension=study["dimension"],
                       boundary="prescribed_contact" if study["case"].startswith("sessile") else "closed")
    limit = value["gates"]["invariance"][axis][declaration["quantity"]]["maximum_spread"]
    declaration["maximum_spread"] = 2.0 * limit
    value["qualification_contract"]["scientific_contracts"] = [declaration]
    with pytest.raises(runner.MatrixError, match="spread"):
        runner.validate_contract(value)


def test_successor_unresolved_analysis_samples_reconstruct_hash_bound_trigger(tmp_path, monkeypatch):
    result, value, _, output = analysis_fixture(tmp_path, monkeypatch, [0.2, 0.3, 0.25])
    study_id = "closed_circle_sampled_analytic"
    study = result["convergence"]["studies"][study_id]
    sequence = next(iter(next(iter(study["groups"].values()))["metrics"]["pressure_space_relative_distance"]["sequences"].values()))
    assert sequence["status"] == "ADDITIONAL_LEVEL_REQUIRED"
    assert [sample["label"] for sample in sequence["samples"]] == ["rdx_8", "rdx_16", "rdx_32"]
    # Supply only the surrounding synthetic campaign envelope; preserve the real sequence output.
    runner.write_json(output / "pre_execution_manifest.json", {"fixture": "synthetic campaign envelope"})
    result.update(expected_case_count=len(runner.expand_cases(value)), exact_groups_passed=True,
                  invariance={"status": "PASS"}, finest_level={"status": "PASS"}, errors=[],
                  qualification_outcome="ADDITIONAL_LEVEL_REQUIRED",
                  pre_execution_manifest_sha256=runner.sha256_file(output / "pre_execution_manifest.json"))
    runner.write_json(output / "summary.json", result)
    trigger = runner.build_conditional_trigger_record(value, output / "summary.json")
    runner.write_json(output / "conditional_trigger_record.json", trigger)
    loaded = runner.load_conditional_trigger_record(value, output / "conditional_trigger_record.json")
    assert loaded == trigger
    expanded = runner.expand_cases(value, conditional_trigger_record=loaded)
    assert len(expanded) == len(runner.expand_cases(value)) + 1
    result["qualification_contract_module_sha256"] = "0" * 64
    runner.write_json(output / "summary.json", result)
    with pytest.raises(runner.MatrixError, match="identity"):
        runner.build_conditional_trigger_record(value, output / "summary.json")
