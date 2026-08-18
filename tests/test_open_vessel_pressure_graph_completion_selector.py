import importlib.util
from pathlib import Path
import sys

import pytest


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_graph_completion_selector.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_graph_completion_selector",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_boundary_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_graph_completion_boundary_provenance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_graph_completion_boundary_provenance",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _support_report(
    *,
    max_dof,
    velocity_sum,
    pressure_sum,
    coupling_threshold=1.0e-14,
    self_threshold=1.0e-8,
):
    return {
        "latest_pressure_graph_completion": {
            "values": {
                "mode": "existing_edge_balance",
                "requested_mode": "existing_edge_balance",
                "coupling_threshold": coupling_threshold,
                "self_threshold": self_threshold,
                "candidate_row_count": 3,
                "edge_count": 4,
                "candidate_global_dofs": "10|11|12|111",
            }
        },
        "latest_support_rank_diagnostic": {
            "values": {"tolerance": 1.0e-14}
        },
        "pressure_update_support_summary": {
            "max_update_global_dof": max_dof,
            "max_update_local_dof": max_dof - 100,
            "max_abs_update": 123.0,
        },
        "sampled_pressure_rows": [
            {
                "matrix_sample": {"dof": max_dof},
                "row_unconstrained_field_abs_sum_by_field": {
                    "Velocity": velocity_sum,
                    "Pressure": pressure_sum,
                },
            }
        ],
    }


def test_selector_summary_flags_shifted_rows_outside_weak_selector():
    audit = _load_audit_module()
    outside = _support_report(
        max_dof=142,
        velocity_sum=3.0e-4,
        pressure_sum=2.0e-7,
    )
    inside = _support_report(
        max_dof=111,
        velocity_sum=2.0e-4,
        pressure_sum=5.0e-9,
    )
    missing = {
        "latest_pressure_graph_completion": {"values": {}},
        "pressure_update_support_summary": {"max_update_global_dof": 99},
        "sampled_pressure_rows": [],
    }

    report = audit.summarize_selector_coverage(
        [
            ("outside", Path("outside.json"), outside),
            ("inside", Path("inside.json"), inside),
            ("missing", Path("missing.json"), missing),
        ]
    )

    assert report["finding"] == (
        "shifted_pressure_update_rows_escape_weak_row_selector"
    )
    assert report["finding_counts"] == {
        "max_update_row_inside_selector_rule": 1,
        "max_update_row_not_sampled": 1,
        "max_update_row_outside_selector_rule": 1,
    }
    outside_case = report["cases"][0]
    assert outside_case["selector_eligible"] is False
    assert outside_case["selector_reason"] == (
        "outside_selector_rule_strong_coupling_and_self"
    )
    assert outside_case["row_velocity_abs_sum"] == 3.0e-4
    assert outside_case["row_pressure_abs_sum"] == 2.0e-7
    assert outside_case["selector_thresholds_needed_to_include"] == {
        "coupling_threshold": 3.0e-4,
        "self_threshold": 2.0e-7,
    }
    assert outside_case["selector_threshold_factors_of_current"][
        "coupling_threshold"
    ] == pytest.approx(3.0e10)
    assert outside_case["selector_threshold_factors_of_current"][
        "self_threshold"
    ] == pytest.approx(20.0)
    assert outside_case["least_selector_threshold_expansion_to_include"] == {
        "selector": "self_threshold",
        "threshold_needed": 2.0e-7,
        "current_threshold": 1.0e-8,
        "factor_of_current": pytest.approx(20.0),
    }
    assert report[
        "sampled_outside_selector_threshold_floor_if_single_selector_widened"
    ] == {
        "case_count": 1,
        "coupling_threshold": 3.0e-4,
        "self_threshold": 2.0e-7,
    }
    assert report[
        "sampled_outside_selector_threshold_floor_if_casewise_least_widened"
    ] == {
        "case_count": 1,
        "coupling_threshold": None,
        "self_threshold": 2.0e-7,
    }
    inside_case = report["cases"][1]
    assert inside_case["selector_eligible"] is True
    assert inside_case["weak_self"] is True
    assert inside_case["candidate_log_sample_contains_max_update_row"] is True


def test_boundary_provenance_flags_candidate_rows_missing_balance_coverage():
    audit = _load_boundary_audit_module()
    support_report = {
        "latest_pressure_graph_completion": {
            "values": {
                "mode": "shared_row_schur_low_degree_edge_balance",
                "requested_mode": "schur-low-degree-edge-balance",
                "candidate_row_count": 4,
                "balance_candidate_row_count": 2,
                "low_degree_balance_candidate_count": 2,
                "coupling_deficient_balance_candidate_count": 1,
                "candidate_global_dofs": "100|101|102|103",
                "balance_candidate_global_dofs": "100|101",
                "low_degree_balance_candidate_global_dofs": "100|101",
                "coupling_deficient_balance_candidate_global_dofs": "103",
            }
        },
        "latest_pressure_update_support_diagnostic": {
            "values": {
                "pressure_offset": 100,
                "max_update_global_dof": 102,
                "top_update_details": (
                    "2:102:update=5:abs_update=5:row_coupling=1e-4:"
                    "row_self=1e-8|3:103:update=4:abs_update=4:"
                    "row_coupling=0:row_self=1e-8"
                ),
            }
        },
        "latest_support_rank_diagnostic": {"values": {"pressure_offset": 100}},
    }
    points = [
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ]
    bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    case = audit.audit_boundary_provenance(
        "synthetic",
        Path("synthetic.json"),
        support_report,
        points=points,
        bounds=bounds,
    )

    assert case["finding"] == "latest_max_update_row_candidate_not_balanced"
    assert case["latest_max_update_in_candidate_sample"] is True
    assert case["latest_max_update_in_balance_sample"] is False
    assert case["top_update_candidate_overlap_global_dofs"] == [102, 103]
    assert case["top_update_balance_overlap_global_dofs"] == []
    assert (
        case["boundary_topology_finding"]
        == "boundary_top_update_candidates_missing_balance"
    )
    assert case["boundary_top_update_count"] == 1
    assert case["boundary_top_update_global_dofs"] == [103]
    assert case["boundary_top_update_candidate_overlap_global_dofs"] == [103]
    assert case["boundary_top_update_balance_overlap_global_dofs"] == []
    assert case["boundary_top_update_low_degree_balance_overlap_global_dofs"] == []
    assert (
        case[
            "boundary_top_update_coupling_deficient_balance_overlap_global_dofs"
        ]
        == [103]
    )
    assert case["boundary_top_update_candidate_not_balanced_global_dofs"] == [103]
    assert case["boundary_top_update_outside_candidate_global_dofs"] == []
    assert case["candidate_sample"]["boundary_class_counts"] == {
        "boundary_corner": 1,
        "boundary_edge": 1,
        "boundary_face": 1,
        "interior": 1,
    }
    assert case["balance_sample"]["boundary_class_counts"] == {
        "boundary_edge": 1,
        "boundary_face": 1,
    }
    assert case["top_update_sample"]["boundary_class_counts"] == {
        "boundary_corner": 1,
        "interior": 1,
    }
    top_rows = case["top_update_sample"]["rows"]
    assert top_rows[0]["global_dof"] == 102
    assert top_rows[0]["in_candidate_sample"]
    assert not top_rows[0]["in_balance_sample"]
    assert not top_rows[0]["is_boundary_topology"]
    assert top_rows[1]["global_dof"] == 103
    assert top_rows[1]["in_candidate_sample"]
    assert not top_rows[1]["in_balance_sample"]
    assert not top_rows[1]["in_low_degree_balance_sample"]
    assert top_rows[1]["in_coupling_deficient_balance_sample"]
    assert top_rows[1]["is_boundary_topology"]
