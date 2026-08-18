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
        / "audit_direct_pspg_no_galerkin_gate_relevance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_no_galerkin_gate_relevance",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _top_overlap_fixture(*, complete=False):
    if complete:
        cases = [
            {
                "label": "test02",
                "finding": "full_top_update_overlap_no_galerkin_zero_coupling_sample",
                "no_galerkin_support_finding": "no_galerkin_support_rank_equivalent",
                "exact_direct_pspg_top_update_count": 2,
                "no_galerkin_top_update_overlap_count": 2,
                "no_galerkin_top_update_overlap_global_dofs": [100, 101],
                "no_galerkin_zero_coupling_global_dofs": [100, 101],
                "no_nonpressure_zero_coupling_global_dofs": [100, 101],
                "support_rank_zero_coupling_global_dofs": [100, 101],
                "no_galerkin_equals_no_nonpressure_zero_coupling": True,
                "no_galerkin_equals_support_rank_zero_coupling": True,
                "exact_direct_pspg_rows_missing_any_aggregate_sample_count": 0,
                "exact_to_aggregate_sample_finding": "all_exact_rows_sampled",
            },
            {
                "label": "test10",
                "finding": "full_top_update_overlap_no_galerkin_zero_coupling_sample",
                "no_galerkin_support_finding": "no_galerkin_support_rank_equivalent",
                "exact_direct_pspg_top_update_count": 3,
                "no_galerkin_top_update_overlap_count": 3,
                "no_galerkin_top_update_overlap_global_dofs": [200, 201, 202],
                "no_galerkin_zero_coupling_global_dofs": [200, 201, 202],
                "no_nonpressure_zero_coupling_global_dofs": [200, 201, 202],
                "support_rank_zero_coupling_global_dofs": [200, 201, 202],
                "no_galerkin_equals_no_nonpressure_zero_coupling": True,
                "no_galerkin_equals_support_rank_zero_coupling": True,
                "exact_direct_pspg_rows_missing_any_aggregate_sample_count": 0,
                "exact_to_aggregate_sample_finding": "all_exact_rows_sampled",
            },
        ]
    else:
        cases = [
            {
                "label": "test02",
                "finding": "no_top_update_overlap_no_galerkin_zero_coupling_sample",
                "no_galerkin_support_finding": "no_galerkin_zero_coupling_absent",
                "exact_direct_pspg_top_update_count": 7,
                "no_galerkin_top_update_overlap_count": 0,
                "no_galerkin_top_update_overlap_global_dofs": [],
                "no_galerkin_zero_coupling_global_dofs": [],
                "no_nonpressure_zero_coupling_global_dofs": [],
                "support_rank_zero_coupling_global_dofs": [],
                "no_galerkin_equals_no_nonpressure_zero_coupling": True,
                "no_galerkin_equals_support_rank_zero_coupling": True,
                "exact_direct_pspg_rows_missing_any_aggregate_sample_count": 7,
                "exact_to_aggregate_sample_finding": (
                    "exact_direct_pspg_rows_undercovered_by_aggregate_samples"
                ),
            },
            {
                "label": "test10",
                "finding": (
                    "partial_top_update_overlap_no_galerkin_zero_coupling_sample"
                ),
                "no_galerkin_support_finding": (
                    "no_galerkin_nonpressure_equivalent_support_rank_differs"
                ),
                "exact_direct_pspg_top_update_count": 12,
                "no_galerkin_top_update_overlap_count": 3,
                "no_galerkin_top_update_overlap_global_dofs": [3526, 3456, 3925],
                "no_galerkin_zero_coupling_global_dofs": [
                    3456,
                    3459,
                    3466,
                    3469,
                    3526,
                    3925,
                    3928,
                    3935,
                    3938,
                ],
                "no_nonpressure_zero_coupling_global_dofs": [
                    3456,
                    3459,
                    3466,
                    3469,
                    3526,
                    3925,
                    3928,
                    3935,
                    3938,
                ],
                "support_rank_zero_coupling_global_dofs": [],
                "no_galerkin_equals_no_nonpressure_zero_coupling": True,
                "no_galerkin_equals_support_rank_zero_coupling": False,
                "exact_direct_pspg_rows_missing_any_aggregate_sample_count": 8,
                "exact_to_aggregate_sample_finding": (
                    "exact_direct_pspg_rows_undercovered_by_aggregate_samples"
                ),
            },
        ]
    return {
        "finding": (
            "all_no_galerkin_overlap_complete"
            if complete
            else "mixed_no_galerkin_overlap_partial_for_some_cases_absent_for_others"
        ),
        "no_galerkin_support_finding": (
            "no_galerkin_support_rank_equivalent"
            if complete
            else "no_galerkin_support_rank_selector_differs_in_some_cases"
        ),
        "exact_to_aggregate_sample_finding": (
            "all_exact_rows_sampled"
            if complete
            else "exact_direct_pspg_top_rows_undercovered_by_aggregate_samples"
        ),
        "cases": cases,
    }


def _predicate_fixture(*, complete=False):
    return {
        "candidates": [
            {
                "key": "zero_galerkin_nonpressure_or_same_sign_pressure_action_patch",
                "finding": (
                    "exact_audited_coverage"
                    if complete
                    else "partial_audited_coverage"
                ),
                "production_readiness": (
                    "candidate_ready_for_replay"
                    if complete
                    else "diagnostic_only_partial_expected"
                ),
                "derivation_status": "known_partial_test10_only_support_signal",
                "covers_all_audited_targets": complete,
                "depends_on_pressure_update_values_in_current_artifact": True,
                "cases": [
                    {
                        "label": "test02",
                        "finding": (
                            "exact_audited_target_coverage"
                            if complete
                            else "partial_audited_target_coverage"
                        ),
                        "direct_target_count": 2 if complete else 7,
                        "selected_count": 2 if complete else 6,
                        "covered_direct_target_global_dofs": (
                            [100, 101]
                            if complete
                            else [10952, 12211, 10954, 12213, 10953, 12212]
                        ),
                        "uncovered_direct_target_global_dofs": (
                            [] if complete else [10676]
                        ),
                        "coverage_ratio": 1.0 if complete else 6 / 7,
                    },
                    {
                        "label": "test10",
                        "finding": "exact_audited_target_coverage",
                        "direct_target_count": 3 if complete else 12,
                        "selected_count": 3 if complete else 12,
                        "covered_direct_target_global_dofs": (
                            [200, 201, 202]
                            if complete
                            else [
                                3526,
                                3456,
                                3925,
                                3455,
                                3924,
                                3454,
                                3923,
                                3451,
                                3920,
                                3525,
                                3919,
                                3450,
                            ]
                        ),
                        "uncovered_direct_target_global_dofs": [],
                        "coverage_ratio": 1.0,
                    },
                ],
            }
        ]
    }


def test_no_galerkin_nonpressure_gate_is_ruled_out_as_complete_gate():
    audit = _load_audit_module()
    report = audit.build_report(
        top_overlap=_top_overlap_fixture(),
        formulation_predicates=_predicate_fixture(),
    )

    assert report["finding"] == (
        "no_galerkin_nonpressure_gate_ruled_out_as_complete_formulation_gate"
    )
    assert report["status"] == (
        "partial_test10_signal_ruled_out_as_complete_gate"
    )
    classification = report["classification"]
    assert classification["overlap_missing_cases"] == ["test02"]
    assert classification["overlap_partial_cases"] == ["test10"]
    assert classification["candidate_uncovered_cases"] == ["test02"]
    assert classification["support_rank_mismatch_cases"] == ["test10"]
    assert not classification["complete_gate_candidate"]
    cases = {case["label"]: case for case in report["top_overlap"]["cases"]}
    assert cases["test02"]["no_galerkin_top_update_overlap_ratio"] == 0.0
    assert cases["test10"]["no_galerkin_top_update_overlap_ratio"] == 0.25


def test_no_galerkin_nonpressure_gate_can_stay_candidate_when_complete():
    audit = _load_audit_module()
    report = audit.build_report(
        top_overlap=_top_overlap_fixture(complete=True),
        formulation_predicates=_predicate_fixture(complete=True),
    )

    assert report["finding"] == "no_galerkin_nonpressure_gate_supported_for_replay"
    assert report["status"] == "candidate_gate_needs_replay"
    classification = report["classification"]
    assert classification["overlap_missing_cases"] == []
    assert classification["overlap_partial_cases"] == []
    assert classification["candidate_uncovered_cases"] == []
    assert classification["support_rank_mismatch_cases"] == []
    assert classification["complete_gate_candidate"]
