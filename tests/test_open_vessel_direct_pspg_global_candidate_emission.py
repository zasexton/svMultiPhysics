import importlib.util
import json
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
        / "audit_direct_pspg_global_candidate_emission.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_global_candidate_emission",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_map():
    return {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [100, 101]},
            {
                "label": "test10",
                "direct_pspg_target_global_dofs": [200, 201, 202],
            },
        ]
    }


def test_global_candidate_emission_covers_audited_targets(tmp_path):
    audit = _load_audit_module()
    test02_log = tmp_path / "test02.log"
    test10_log = tmp_path / "test10.log"
    line = (
        "NewtonSolver: direct PSPG formulation candidate diagnostic "
        "diagnostic=direct_pspg_formulation_candidate status=ok rank=0 "
        "iteration=0 phase='pressure_row_contribution_matrix:accepted' "
        "op='equations_diagnostic_ns_vms_pspg_pressure_gradient' "
        "selector='sparse_direct_self_or_matrix_pressure_action_patch' "
        "direct_self_positive_row_count=4 "
        "direct_self_row_sum_leak_threshold=0.25 "
        "direct_self_null_preserving_threshold=0.05 "
        "direct_self_diag_dominant_threshold=0.6 "
        "direct_self_balanced_diag_low_threshold=0.45 "
        "direct_self_balanced_diag_high_threshold=0.55 "
        "max_direct_self_row_sum_leak_ratio=0.75 "
        "min_direct_self_diag_abs_ratio=0.25 "
        "max_direct_self_diag_abs_ratio=0.75 "
        "high_direct_self_row_sum_leak_candidate_count=1 "
        "null_preserving_direct_self_candidate_count=3 "
        "diag_dominant_direct_self_candidate_count=1 "
        "balanced_diag_direct_self_candidate_count=3 "
        "sparse_direct_self_candidate_count=1 "
        "max_unconstrained_direct_self_numeric_entries=4 "
        "constrained_pressure_neighbor_candidate_count=2 "
        "constrained_pressure_neighbor_ratio_threshold=0.25 "
        "high_constrained_pressure_neighbor_ratio_candidate_count=1 "
        "sparse_unconstrained_direct_self_candidate_count=3 "
        "constrained_or_sparse_unconstrained_direct_self_candidate_count=4 "
        "direct_self_low_ratio_threshold=0.25 "
        "direct_self_moderate_ratio_threshold=0.5 "
        "low_direct_self_ratio_candidate_count=1 "
        "moderate_direct_self_ratio_candidate_count=2 "
        "sparse_or_moderate_direct_self_ratio_candidate_count=3 "
        "sparse_seeded_pressure_action_radius1_candidate_count=3 "
        "sparse_seeded_pressure_action_radius2_candidate_count=4 "
        "graph_local_direct_self_low_ratio_threshold=0.5 "
        "graph_local_direct_self_moderate_ratio_threshold=0.75 "
        "graph_local_neighbor_positive_row_count=4 "
        "graph_local_low_direct_self_ratio_candidate_count=2 "
        "graph_local_moderate_direct_self_ratio_candidate_count=3 "
        "matrix_pressure_action_covered_count=3 "
        "matrix_pressure_action_isolated_count=1 "
        "matrix_pressure_action_max_degree=3 "
        "matrix_pressure_action_max_abs_sum=4.0 "
        "pressure_action_low_degree_threshold=2 "
        "pressure_action_moderate_degree_threshold=4 "
        "pressure_action_low_degree_candidate_count=1 "
        "pressure_action_moderate_degree_candidate_count=3 "
        "pressure_action_low_sum_ratio_threshold=0.25 "
        "pressure_action_moderate_sum_ratio_threshold=0.5 "
        "pressure_action_low_sum_ratio_candidate_count=1 "
        "pressure_action_moderate_sum_ratio_candidate_count=2 "
        "pressure_action_self_dominant_threshold=0.75 "
        "pressure_action_self_dominant_candidate_count=1 "
        "pressure_action_low_two_hop_threshold=4 "
        "pressure_action_high_two_hop_ratio_threshold=0.5 "
        "matrix_pressure_action_max_two_hop_completion_count=8 "
        "pressure_action_zero_two_hop_candidate_count=1 "
        "pressure_action_low_two_hop_candidate_count=2 "
        "pressure_action_high_two_hop_candidate_count=2 "
        "pressure_action_low_clustering_threshold=0.25 "
        "pressure_action_high_clustering_threshold=0.75 "
        "pressure_action_clustering_eligible_row_count=3 "
        "matrix_pressure_action_min_clustering_ratio=0.0 "
        "matrix_pressure_action_max_clustering_ratio=1.0 "
        "pressure_action_zero_clustering_candidate_count=1 "
        "pressure_action_low_clustering_candidate_count=2 "
        "pressure_action_high_clustering_candidate_count=1 "
        "pressure_action_articulation_candidate_count=1 "
        "pressure_action_bridge_endpoint_candidate_count=2 "
        "matrix_pressure_action_component_count=2 "
        "matrix_pressure_action_largest_component_size=3 "
        "sparse_seeded_matrix_pressure_action_component_count=1 "
        "sparse_seeded_matrix_pressure_action_component_dof_count=2 "
        "residual_sign_threshold=1.0e-12 "
        "residual_positive_direct_row_count=3 "
        "residual_negative_direct_row_count=1 "
        "residual_zero_direct_row_count=0 "
        "residual_nonfinite_direct_row_count=0 "
        "residual_nonzero_direct_row_count=4 "
        "min_positive_residual_abs=1.0e-8 "
        "max_residual_abs=2.0e-7 "
        "residual_sign_pressure_action_edge_count=2 "
        "residual_opposite_sign_pressure_action_edge_count=1 "
        "residual_zero_or_missing_sign_pressure_action_edge_count=0 "
        "residual_sign_pressure_action_component_count=2 "
        "residual_sign_pressure_action_largest_component_size=3 "
        "residual_sign_pressure_action_covered_count=3 "
        "sparse_seeded_residual_sign_pressure_action_component_count=1 "
        "sparse_seeded_residual_sign_pressure_action_component_dof_count=2 "
        "sparse_direct_self_or_residual_sign_pressure_action_candidate_count=3 "
        "preferred_candidate_count=2 "
        "artifact_limitation='matrix-sign and residual-sign pressure-action proxies; no update signs' "
    )
    test02_log.write_text(
        line
        + "preferred_candidate_global_dofs=100|101 "
        + "high_direct_self_row_sum_leak_global_dofs=100 "
        + "null_preserving_direct_self_global_dofs=100|101|103 "
        + "diag_dominant_direct_self_global_dofs=100 "
        + "balanced_diag_direct_self_global_dofs=100|101|103 "
        + "sparse_seeded_matrix_pressure_action_component_global_dofs=100|101 "
        + "residual_sign_pressure_action_covered_global_dofs=100|101 "
        + "sparse_seeded_residual_sign_pressure_action_component_global_dofs=100|101 "
        + "sparse_direct_self_or_residual_sign_pressure_action_global_dofs=100|101|103 "
        + "sparse_direct_self_global_dofs=100 "
        + "constrained_pressure_neighbor_global_dofs=100|101 "
        + "high_constrained_pressure_neighbor_ratio_global_dofs=100 "
        + "sparse_unconstrained_direct_self_global_dofs=100|101|103 "
        + "constrained_or_sparse_unconstrained_direct_self_global_dofs=100|101|102|103 "
        + "low_direct_self_ratio_global_dofs=100 "
        + "moderate_direct_self_ratio_global_dofs=100|101 "
        + "sparse_or_moderate_direct_self_ratio_global_dofs=100|101|103 "
        + "sparse_seeded_pressure_action_radius1_global_dofs=100|101|103 "
        + "sparse_seeded_pressure_action_radius2_global_dofs=100|101|102|103 "
        + " graph_local_low_direct_self_ratio_global_dofs=100|101"
        + " graph_local_moderate_direct_self_ratio_global_dofs=100|101|103\n",
        encoding="utf-8",
    )
    test02_log.write_text(
        test02_log.read_text(encoding="utf-8").rstrip()
        + " pressure_action_low_degree_global_dofs=100"
        + " pressure_action_moderate_degree_global_dofs=100|101|103"
        + " pressure_action_low_sum_ratio_global_dofs=100"
        + " pressure_action_moderate_sum_ratio_global_dofs=100|101"
        + " pressure_action_self_dominant_global_dofs=100\n",
        encoding="utf-8",
    )
    test02_log.write_text(
        test02_log.read_text(encoding="utf-8").rstrip()
        + " pressure_action_zero_two_hop_global_dofs=100"
        + " pressure_action_low_two_hop_global_dofs=100|101"
        + " pressure_action_high_two_hop_global_dofs=101|103"
        + " pressure_action_zero_clustering_global_dofs=100"
        + " pressure_action_low_clustering_global_dofs=100|101"
        + " pressure_action_high_clustering_global_dofs=103"
        + " pressure_action_articulation_global_dofs=101"
        + " pressure_action_bridge_endpoint_global_dofs=100|101\n",
        encoding="utf-8",
    )
    test10_log.write_text(
        line
        + "preferred_candidate_global_dofs=200|201|202 "
        + "high_direct_self_row_sum_leak_global_dofs=200 "
        + "null_preserving_direct_self_global_dofs=200|201 "
        + "diag_dominant_direct_self_global_dofs=200 "
        + "balanced_diag_direct_self_global_dofs=200|201 "
        + "sparse_seeded_matrix_pressure_action_component_global_dofs=200|201 "
        + "residual_sign_pressure_action_covered_global_dofs=200|201 "
        + "sparse_seeded_residual_sign_pressure_action_component_global_dofs=200|201 "
        + "sparse_direct_self_or_residual_sign_pressure_action_global_dofs=200|201 "
        + "sparse_direct_self_global_dofs=201 "
        + "constrained_pressure_neighbor_global_dofs=200|201 "
        + "high_constrained_pressure_neighbor_ratio_global_dofs=200 "
        + "sparse_unconstrained_direct_self_global_dofs=200|201 "
        + "constrained_or_sparse_unconstrained_direct_self_global_dofs=200|201 "
        + "low_direct_self_ratio_global_dofs=201 "
        + "moderate_direct_self_ratio_global_dofs=200|201 "
        + "sparse_or_moderate_direct_self_ratio_global_dofs=200|201 "
        + "sparse_seeded_pressure_action_radius1_global_dofs=200|201 "
        + "sparse_seeded_pressure_action_radius2_global_dofs=200|201|202 "
        + "graph_local_low_direct_self_ratio_global_dofs=200|201 "
        + "graph_local_moderate_direct_self_ratio_global_dofs=200|201|202 "
        + "pressure_action_low_degree_global_dofs=200 "
        + "pressure_action_moderate_degree_global_dofs=200|201 "
        + "pressure_action_low_sum_ratio_global_dofs=200 "
        + "pressure_action_moderate_sum_ratio_global_dofs=200|201 "
        + "pressure_action_self_dominant_global_dofs=200 "
        + "pressure_action_zero_two_hop_global_dofs=200 "
        + "pressure_action_low_two_hop_global_dofs=200|201 "
        + "pressure_action_high_two_hop_global_dofs=201|202 "
        + "pressure_action_zero_clustering_global_dofs=200 "
        + "pressure_action_low_clustering_global_dofs=200|201 "
        + "pressure_action_high_clustering_global_dofs=202 "
        + "pressure_action_articulation_global_dofs=201 "
        + "pressure_action_bridge_endpoint_global_dofs=200|201\n",
        encoding="utf-8",
    )

    report = audit.build_report(
        target_map=_target_map(),
        logs=[("test02", test02_log), ("test10", test10_log)],
    )

    assert report["finding"] == "candidate_emission_covers_audited_targets"
    assert "formulation-side PSPG" in report["next_requirement"]
    cases = {case["label"]: case for case in report["cases"]}
    assert cases["test02"]["finding"] == (
        "candidate_emitted_covers_audited_targets"
    )
    assert cases["test02"]["covered_direct_target_global_dofs"] == [100, 101]
    assert cases["test02"]["direct_self_row_sum_leak_threshold"] == 0.25
    assert cases["test02"]["direct_self_null_preserving_threshold"] == 0.05
    assert cases["test02"]["direct_self_diag_dominant_threshold"] == 0.6
    assert cases["test02"]["max_direct_self_row_sum_leak_ratio"] == 0.75
    assert cases["test02"][
        "high_direct_self_row_sum_leak_covered_direct_target_global_dofs"
    ] == [100]
    assert cases["test02"][
        "null_preserving_direct_self_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "diag_dominant_direct_self_covered_direct_target_global_dofs"
    ] == [100]
    assert cases["test02"][
        "balanced_diag_direct_self_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_seeded_matrix_pressure_action_component_dof_count"
    ] == 2
    assert cases["test02"]["direct_self_low_ratio_threshold"] == 0.25
    assert cases["test02"]["direct_self_moderate_ratio_threshold"] == 0.5
    assert cases["test02"]["max_unconstrained_direct_self_numeric_entries"] == 4
    assert cases["test02"][
        "constrained_pressure_neighbor_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "high_constrained_pressure_neighbor_ratio_covered_direct_target_global_dofs"
    ] == [100]
    assert cases["test02"]["constrained_pressure_neighbor_ratio_threshold"] == 0.25
    assert cases["test02"][
        "sparse_unconstrained_direct_self_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "constrained_or_sparse_unconstrained_direct_self_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "low_direct_self_ratio_covered_direct_target_global_dofs"
    ] == [100]
    assert cases["test02"][
        "moderate_direct_self_ratio_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_seeded_pressure_action_radius2_candidate_count"
    ] == 4
    assert cases["test02"]["graph_local_direct_self_low_ratio_threshold"] == 0.5
    assert cases["test02"][
        "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"]["matrix_pressure_action_max_degree"] == 3
    assert cases["test02"]["matrix_pressure_action_max_abs_sum"] == 4.0
    assert cases["test02"]["pressure_action_low_degree_threshold"] == 2
    assert cases["test02"]["pressure_action_moderate_degree_threshold"] == 4
    assert cases["test02"]["pressure_action_low_two_hop_threshold"] == 4
    assert cases["test02"]["matrix_pressure_action_max_two_hop_completion_count"] == 8
    assert cases["test02"][
        "pressure_action_low_two_hop_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "pressure_action_articulation_covered_direct_target_global_dofs"
    ] == [101]
    assert cases["test02"][
        "pressure_action_bridge_endpoint_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "pressure_action_moderate_degree_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"]["pressure_action_low_sum_ratio_threshold"] == 0.25
    assert cases["test02"]["pressure_action_moderate_sum_ratio_threshold"] == 0.5
    assert cases["test02"][
        "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"]["pressure_action_self_dominant_threshold"] == 0.75
    assert cases["test02"][
        "pressure_action_self_dominant_covered_direct_target_global_dofs"
    ] == [100]
    assert cases["test02"]["matrix_pressure_action_component_count"] == 2
    assert cases["test02"]["residual_sign_pressure_action_edge_count"] == 2
    assert cases["test02"][
        "residual_sign_pressure_action_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_seeded_residual_sign_pressure_action_component_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"][
        "sparse_direct_self_or_residual_sign_pressure_action_covered_direct_target_global_dofs"
    ] == [100, 101]
    assert cases["test02"]["artifact_limitation"] == (
        "matrix-sign and residual-sign pressure-action proxies; no update signs"
    )
    assert cases["test10"]["covered_direct_target_global_dofs"] == [
        200,
        201,
        202,
    ]
    assert cases["test10"][
        "sparse_seeded_matrix_pressure_action_component_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "null_preserving_direct_self_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "sparse_direct_self_or_residual_sign_pressure_action_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "constrained_pressure_neighbor_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "constrained_or_sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "sparse_or_moderate_direct_self_ratio_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "sparse_seeded_pressure_action_radius1_uncovered_direct_target_global_dofs"
    ] == [202]
    assert cases["test10"][
        "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs"
    ] == [200, 201, 202]


def test_global_candidate_emission_flags_sample_limited_coverage(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "test02.log"
    log.write_text(
        "NewtonSolver: direct PSPG formulation candidate diagnostic "
        "diagnostic=direct_pspg_formulation_candidate status=ok "
        "op='equations_diagnostic_ns_vms_pspg_pressure_gradient' "
        "selector='sparse_direct_self_or_matrix_pressure_action_patch' "
        "preferred_candidate_count=99 "
        "preferred_candidate_global_dofs=100|... \n",
        encoding="utf-8",
    )

    report = audit.build_report(
        target_map=_target_map(),
        logs=[("test02", log), ("test10", tmp_path / "missing.log")],
    )

    cases = {case["label"]: case for case in report["cases"]}
    assert cases["test02"]["finding"] == (
        "candidate_emitted_but_coverage_sample_limited"
    )
    assert cases["test02"]["candidate_list_truncated"]
    assert cases["test02"]["uncovered_direct_target_global_dofs"] == [101]
    assert cases["test10"]["finding"] == (
        "direct_pspg_candidate_diagnostic_missing"
    )


def test_cli_writes_json_report(tmp_path):
    audit = _load_audit_module()
    target_path = tmp_path / "targets.json"
    log_path = tmp_path / "test02.log"
    out_path = tmp_path / "report.json"
    target_path.write_text(json.dumps(_target_map()), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")

    assert (
        audit.main.__module__
        == "audit_direct_pspg_global_candidate_emission"
    )
    report = audit.build_report(
        target_map=json.loads(target_path.read_text(encoding="utf-8")),
        logs=[("test02", log_path)],
        target_map_path=target_path,
    )
    out_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["finding"] == "candidate_emission_logs_missing_cases"
