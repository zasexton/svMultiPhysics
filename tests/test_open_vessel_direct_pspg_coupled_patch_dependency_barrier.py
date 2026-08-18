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
        / "audit_direct_pspg_coupled_patch_dependency_barrier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_coupled_patch_dependency_barrier",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _same_sign_fixture(*, candidate_ready=False):
    if candidate_ready:
        dependency_summary = {
            "all_complete_candidates_depend_on_pressure_update": False,
            "all_exact_candidates_depend_on_pressure_update": False,
            "preferred_candidate_depends_on_pressure_update": False,
            "complete_non_update_dependent_candidate_keys": [
                "direct_pspg_pressure_gradient_provenance_patch",
            ],
            "exact_non_update_dependent_candidate_keys": [
                "direct_pspg_pressure_gradient_provenance_patch",
            ],
        }
        finding = "formulation_ready_candidate_available"
    else:
        dependency_summary = {
            "all_complete_candidates_depend_on_pressure_update": True,
            "all_exact_candidates_depend_on_pressure_update": True,
            "preferred_candidate_depends_on_pressure_update": True,
            "complete_non_update_dependent_candidate_keys": [],
            "exact_non_update_dependent_candidate_keys": [],
        }
        finding = (
            "same_sign_patch_blocked_by_pressure_update_dependency_and_"
            "preupdate_proxies"
        )
    return {
        "finding": finding,
        "dependency_summary": dependency_summary,
        "preupdate_proxy_summary": {
            "all_preupdate_proxy_gates_failed": not candidate_ready,
            "failed_gate_keys": (
                []
                if candidate_ready
                else [
                    "direct_self_support_ratio_gate_finding",
                    "graph_local_support_ratio_gate_finding",
                    "pressure_action_moderate_degree_gate_finding",
                ]
            ),
        },
        "cross_policy_patch_summary": {
            "finding": (
                "no_cross_policy_patch_evidence"
                if candidate_ready
                else "cross_policy_patch_evidence_is_post_update_diagnostic_only"
            ),
            "cross_policy_join_field_populated": False,
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "candidate_ready"
                        if candidate_ready
                        else "cross_policy_patch_visible_only_after_pressure_disabled_update"
                    ),
                }
            ],
        },
    }


def _no_galerkin_fixture():
    return {
        "finding": "no_galerkin_nonpressure_gate_ruled_out_as_complete_formulation_gate",
        "status": "partial_test10_signal_ruled_out_as_complete_gate",
        "classification": {
            "complete_gate_candidate": False,
            "overlap_missing_cases": ["test02"],
            "overlap_partial_cases": ["test10"],
            "candidate_uncovered_cases": ["test02"],
            "support_rank_mismatch_cases": ["test10"],
        },
        "formulation_candidate": {
            "key": "zero_galerkin_nonpressure_or_same_sign_pressure_action_patch",
            "covers_all_audited_targets": False,
        },
    }


def _active_support_cutoff_fixture():
    return {
        "finding": "active_pressure_support_cutoff_not_complete_fix_from_branch_shift",
        "status": "support_cutoff_diagnostic_only_not_complete_fix",
        "classification": {
            "retained_fraction_cutoff_is_complete_fix_candidate": False,
            "retained_fraction_cutoff_is_diagnostic_only": True,
            "tiny_cut_supported_branch_present": True,
            "full_wet_supported_branch_present": True,
        },
    }


def test_dependency_barrier_requires_solve_time_provenance():
    audit = _load_audit_module()
    report = audit.build_report(
        same_sign_readiness=_same_sign_fixture(),
        no_galerkin_gate=_no_galerkin_fixture(),
        active_support_cutoff=_active_support_cutoff_fixture(),
    )

    assert report["finding"] == (
        "coupled_patch_dependency_barrier_requires_solve_time_provenance"
    )
    assert report["status"] == (
        "remaining_gate_requires_new_assembly_provenance_diagnostic"
    )
    assert report["blocker_summary"] == {
        "same_sign_exact_candidates_update_dependent": True,
        "same_sign_complete_candidates_update_dependent": True,
        "same_sign_has_non_update_dependent_complete_candidate": False,
        "preupdate_proxy_gates_all_failed": True,
        "cross_policy_patch_is_post_update_diagnostic_only": True,
        "no_galerkin_complete_gate_ruled_out": True,
        "retained_fraction_cutoff_not_complete_fix": True,
        "requires_new_solve_time_provenance": True,
    }
    assert (
        "does not use pressure-update signs" in report["next_requirement"]
    )


def test_dependency_barrier_releases_when_complete_non_update_candidate_exists():
    audit = _load_audit_module()
    report = audit.build_report(
        same_sign_readiness=_same_sign_fixture(candidate_ready=True),
        no_galerkin_gate=_no_galerkin_fixture(),
        active_support_cutoff=_active_support_cutoff_fixture(),
    )

    assert report["finding"] == (
        "coupled_patch_dependency_barrier_not_present_candidate_ready"
    )
    assert report["status"] == "replay_non_update_dependent_candidate"
    assert report["blocker_summary"][
        "same_sign_has_non_update_dependent_complete_candidate"
    ] is True
    assert report["blocker_summary"]["requires_new_solve_time_provenance"] is False
