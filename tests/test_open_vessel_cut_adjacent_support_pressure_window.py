import importlib.util
from pathlib import Path
import sys


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_cut_adjacent_support_pressure_window.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_cut_adjacent_support_pressure_window", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retained_support_with_pruned_volume_is_not_trace_only_driver(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "TimeLoop: step_start step=90 time=0.9 dt=0.000625",
                "Application: diagnostic=cut_context_rebuild "
                "provenance=line_search_trial solution_source=state_vector_fe_ordered "
                "cut_context_revision=2 active_side_retained_cut_volume_rule_count=720 "
                "active_side_available_cut_volume_rule_count=720 "
                "active_pruned_volume_regions=8 active_pruned_volume=3.5e-10 "
                "generated_pruned_volume_rules=8 generated_pruned_volume=3.5e-10 "
                "active_min_volume_fraction=0.0447 cut_adjacent_facets=429",
                "LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint "
                "field='Pressure' support_mode=retained_cut_volume+cut_adjacent_facets "
                "active_support_cells=720 "
                "active_support_cells_from_volume_support=720 "
                "active_support_cells_from_cut_adjacent_facets=0 "
                "active_support_vertices=252 active_support_dofs=252",
                "Application: diagnostic=active_pressure_constraint_refresh "
                "provenance=line_search_trial solution_source=state_vector_fe_ordered "
                "support_source=retained_cut_context constraints=2398",
                "TimeLoop: step_accepted step=91 time=0.900625 dt=0.000625",
                "Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard phase='post_accept' "
                "step=91 time=0.900625 dt=0.000625 field='Pressure' "
                "local_worst_vertex=83 local_worst_dof=3526 "
                "global_abs_pressure_delta_pa=622.6 "
                "local_abs_pressure_delta_pa=622.6 "
                "support_class=full_wet_supported "
                "incident_wet_fraction_max=1.0 "
                "incident_wet_fraction_min_positive=1.0 "
                "threshold_pa=100 triggered=1",
            ]
        ),
        encoding="utf-8",
    )

    case = audit.audit_case_log("test10", log)
    window = case["guard_windows"][0]

    assert case["finding"] == (
        "trace_only_support_ruled_out_recent_pruned_volume_not_direct_"
        "trace_only_driver"
    )
    assert case["trace_only_cut_adjacent_support_present_before_any_guard"] is False
    assert case["pruned_generated_volume_present_before_any_guard"] is True
    assert window["pre_guard_retained_volume_support_present"] is True
    assert window["pre_guard_cut_adjacent_only_support_present"] is False
    assert window["pre_guard_pruned_generated_volume_present"] is True
    assert window["worst_update_full_wet"] is True
    assert window["finding"] == (
        "pruned_generated_volume_present_but_retained_volume_support_active_"
        "before_guard"
    )


def test_trace_only_cut_adjacent_support_is_flagged(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "TimeLoop: step_start step=0 time=0.0 dt=0.1",
                "Application: diagnostic=cut_context_rebuild "
                "provenance=line_search_trial active_side_retained_cut_volume_rule_count=0 "
                "generated_pruned_volume_rules=2 generated_pruned_volume=1.0e-9 "
                "cut_adjacent_facets=3",
                "LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint "
                "field='Pressure' support_mode=cell_patch+cut_adjacent_facets "
                "active_support_cells=3 active_support_cells_from_volume_support=0 "
                "active_support_cells_from_cut_adjacent_facets=3",
                "TimeLoop: step_accepted step=1 time=0.1 dt=0.1",
                "Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard phase='post_accept' "
                "step=1 time=0.1 dt=0.1 field='Pressure' "
                "local_worst_dof=100 global_abs_pressure_delta_pa=42.0 "
                "support_class=full_wet_supported "
                "incident_wet_fraction_max=1.0 "
                "incident_wet_fraction_min_positive=1.0 triggered=1",
            ]
        ),
        encoding="utf-8",
    )

    report = audit.build_report([("synthetic", log)])
    case = report["cases"][0]
    window = case["guard_windows"][0]

    assert report["finding"] == "trace_only_cut_adjacent_support_not_ruled_out"
    assert report["trace_only_cut_adjacent_support_cases"] == ["synthetic"]
    assert case["finding"] == "trace_only_cut_adjacent_support_not_ruled_out"
    assert window["pre_guard_cut_adjacent_only_support_present"] is True
    assert window["pre_guard_retained_volume_support_present"] is False
    assert window["finding"] == (
        "trace_only_cut_adjacent_support_present_before_guard"
    )


def test_retained_support_without_pruned_volume_rules_out_both_subpaths(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "TimeLoop: step_start step=0 time=0.0 dt=0.01",
                "Application: diagnostic=cut_context_rebuild "
                "provenance=line_search_trial active_side_retained_cut_volume_rule_count=3358 "
                "generated_pruned_volume_rules=0 generated_pruned_volume=0 "
                "active_pruned_volume_regions=0 active_pruned_volume=0 "
                "active_min_volume_fraction=2.2e-8 cut_adjacent_facets=2124",
                "LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint "
                "field='Pressure' support_mode=retained_cut_volume+cut_adjacent_facets "
                "active_support_cells=3358 "
                "active_support_cells_from_volume_support=3358 "
                "active_support_cells_from_cut_adjacent_facets=0",
                "TimeLoop: step_accepted step=1 time=0.01 dt=0.01",
                "Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard phase='post_accept' "
                "step=1 time=0.01 dt=0.01 field='Pressure' "
                "local_worst_dof=10676 global_abs_pressure_delta_pa=366719.9 "
                "support_class=full_wet_supported "
                "incident_wet_fraction_max=1.0 "
                "incident_wet_fraction_min_positive=1.0 triggered=1",
            ]
        ),
        encoding="utf-8",
    )

    report = audit.build_report([("test02", log)])
    window = report["cases"][0]["guard_windows"][0]

    assert report["finding"] == (
        "trace_only_and_recent_pruned_support_absent_before_guards"
    )
    assert report["trace_only_cut_adjacent_support_ruled_out_before_guards"] is True
    assert report["pruned_generated_volume_present_before_some_guard"] is False
    assert window["finding"] == (
        "retained_volume_support_without_trace_only_or_pruned_generated_"
        "volume_before_guard"
    )
