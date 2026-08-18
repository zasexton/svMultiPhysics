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
        / "audit_direct_pspg_formulation_side_candidate_predicates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_formulation_side_candidate_predicates",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formulation_side_candidate_prefers_sparse_self_plus_same_sign_patch():
    audit = _load_audit_module()
    target_map = {
        "cases": [
            {
                "label": "test02",
                "direct_pspg_target_global_dofs": [100, 101],
                "direct_pspg_support_gap_global_dofs": [100, 101],
            },
            {
                "label": "test10",
                "direct_pspg_target_global_dofs": [200, 201, 202],
                "direct_pspg_support_gap_global_dofs": [201, 202],
            },
        ]
    }
    top_provenance = {
        "cases": [
            {
                "label": "test02",
                "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                    101
                ],
                "direct_pspg_sparse_direct_self_entry_global_dofs": [100],
                "direct_pspg_low_direct_self_ratio_global_dofs": [100],
                "direct_pspg_moderate_direct_self_ratio_global_dofs": [100],
                "direct_pspg_missing_wall_normal_self_global_dofs": [100],
                "direct_pspg_missing_wall_tangential_self_global_dofs": [101],
                "direct_pspg_zero_galerkin_nonpressure_coupling_global_dofs": [],
            },
            {
                "label": "test10",
                "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                    200,
                    201,
                    202,
                ],
                "direct_pspg_sparse_direct_self_entry_global_dofs": [201, 202],
                "direct_pspg_low_direct_self_ratio_global_dofs": [],
                "direct_pspg_moderate_direct_self_ratio_global_dofs": [201],
                "direct_pspg_missing_wall_normal_self_global_dofs": [202],
                "direct_pspg_missing_wall_tangential_self_global_dofs": [],
                "direct_pspg_zero_galerkin_nonpressure_coupling_global_dofs": [
                    200
                ],
            },
        ]
    }

    report = audit.build_report(
        target_map=target_map,
        top_provenance=top_provenance,
    )

    assert report["finding"] == (
        "narrow_formulation_side_candidate_identified_needs_global_emission"
    )
    assert report["preferred_next_candidate"]["key"] == (
        "sparse_direct_self_or_same_sign_pressure_action_patch"
    )
    assert report["preferred_next_candidate"]["production_readiness"] == (
        "formulation_candidate_pending_global_solve_time_emission"
    )
    assert (
        "sparse_direct_self_or_same_sign_pressure_action_patch"
        in report["exact_audited_candidate_keys"]
    )
    assert (
        "zero_galerkin_nonpressure_or_same_sign_pressure_action_patch"
        in report["partial_candidate_keys"]
    )

    candidates = {candidate["key"]: candidate for candidate in report["candidates"]}
    same_sign = candidates["same_sign_pressure_action_patch"]
    assert same_sign["finding"] == "partial_audited_coverage"
    assert same_sign["cases"][0]["uncovered_direct_target_global_dofs"] == [100]

    preferred = candidates[
        "sparse_direct_self_or_same_sign_pressure_action_patch"
    ]
    assert preferred["finding"] == "exact_audited_coverage"
    assert preferred["covers_all_audited_targets"]
    assert preferred["exact_audited_coverage"]
    assert preferred["cases"][0]["selected_to_direct_target_ratio"] == 1.0
    assert preferred["cases"][1]["selected_to_direct_target_ratio"] == 1.0
    assert "global candidate emission" in report["next_requirement"]


def test_candidate_report_flags_overselection_when_rows_exceed_targets():
    audit = _load_audit_module()
    target_map = {
        "cases": [
            {
                "label": "test02",
                "direct_pspg_target_global_dofs": [100],
                "direct_pspg_support_gap_global_dofs": [100],
            }
        ]
    }
    top_provenance = {
        "cases": [
            {
                "label": "test02",
                "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                    100,
                    999,
                ],
                "direct_pspg_sparse_direct_self_entry_global_dofs": [],
            }
        ]
    }

    report = audit.build_report(
        target_map=target_map,
        top_provenance=top_provenance,
    )
    candidates = {candidate["key"]: candidate for candidate in report["candidates"]}
    same_sign = candidates["same_sign_pressure_action_patch"]

    assert same_sign["finding"] == "complete_but_overselects_audited_rows"
    assert same_sign["cases"][0]["extra_selected_global_dofs"] == [999]
