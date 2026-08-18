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
        / "audit_active_pressure_support_cutoff_relevance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_active_pressure_support_cutoff_relevance",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_text():
    return """
        record_retained_rule_support(
                    static_cast<GlobalIndex>(cell), volume_rules[index]);
            }
            if (mark_cell_active(static_cast<GlobalIndex>(cell))) {
        retained_min_volume_fraction retained_max_volume_fraction
        rule.volume_fraction
        SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS
    """


def _topology_replays(*, include_full_wet=True):
    records = [
        {
            "case": "test02",
            "policy": "local_schur_completion",
            "worst_active_or_wet_update_pa": 176849.84039557964,
            "worst_active_or_wet_support_class": "tiny_cut_supported",
            "worst_active_or_wet_fraction_min_positive": 5.0e-7,
        },
        {
            "case": "test02",
            "policy": "local_edge_balance",
            "worst_active_or_wet_update_pa": 176848.02921204976,
            "worst_active_or_wet_support_class": "tiny_cut_supported",
            "worst_active_or_wet_fraction_min_positive": 4.0e-7,
        },
    ]
    if include_full_wet:
        records.append(
            {
                "case": "test10",
                "policy": "local_edge_balance",
                "worst_active_or_wet_update_pa": 530.3194043612839,
                "worst_active_or_wet_support_class": "full_wet_supported",
            }
        )
    return {
        "finding": "direct_pspg_topology_policy_local_modes_do_not_clear_guards",
        "status": "local_topology_policy_family_ruled_out_as_complete_fix",
        "policies_tested": ["local_schur_completion", "local_edge_balance"],
        "case_policy_results": records,
    }


def _rejection_replay(*, include_full_wet=True):
    sequence = [
        {
            "dt_s": 0.001,
            "worst_pre_commit_update_pa": 105593.66490062946,
            "worst_pre_commit_dof": 11875,
            "support_class": "tiny_cut_supported",
        }
    ]
    if include_full_wet:
        sequence.append(
            {
                "dt_s": 0.0005,
                "worst_pre_commit_update_pa": 870625.0856344305,
                "worst_pre_commit_dof": 10676,
                "support_class": "full_wet_supported",
            }
        )
    return {
        "finding": "pressure_update_rejection_catches_both_cases_dt_reduction_not_fix",
        "status": "pre_commit_guard_supported_dt_reduction_ruled_out",
        "fixed_step_replays": [
            {
                "case": "test02",
                "worst_pre_commit_update_pa": 105591.14535324997,
                "worst_pre_commit_support_class": "tiny_cut_supported",
            }
        ],
        "adaptive_replays": [
            {
                "case": "test02",
                "support_branch_shift": (
                    "tiny_cut_supported_to_full_wet_supported"
                    if include_full_wet
                    else "none"
                ),
                "update_growth_factor": 133.6925183418554,
                "dt_update_sequence": sequence,
            }
        ],
    }


def test_support_cutoff_classified_diagnostic_only_when_full_wet_branch_remains():
    audit = _load_audit_module()
    report = audit.build_report(
        source_text=_source_text(),
        topology_replays=_topology_replays(include_full_wet=True),
        rejection_replay=_rejection_replay(include_full_wet=True),
    )

    assert report["finding"] == (
        "active_pressure_support_cutoff_not_complete_fix_from_branch_shift"
    )
    assert report["status"] == "support_cutoff_diagnostic_only_not_complete_fix"
    assert report["constraint_source"][
        "retained_generated_volume_support_activation_is_unconditional"
    ]
    assert not report["constraint_source"][
        "retained_generated_volume_support_uses_volume_fraction_cutoff"
    ]
    classification = report["classification"]
    assert classification["tiny_cut_supported_branch_present"]
    assert classification["full_wet_supported_branch_present"]
    assert classification["retained_fraction_cutoff_is_diagnostic_only"]
    assert not classification["retained_fraction_cutoff_is_complete_fix_candidate"]
    assert report["topology_policy_replay_summary"][
        "test02_min_tiny_cut_fraction_positive"
    ] == 4.0e-7
    assert "full_wet_supported" in report[
        "pressure_update_rejection_summary"
    ]["test02_adaptive_support_sequence"]


def test_support_cutoff_stays_candidate_when_only_tiny_cut_branch_is_seen():
    audit = _load_audit_module()
    report = audit.build_report(
        source_text=_source_text(),
        topology_replays=_topology_replays(include_full_wet=False),
        rejection_replay=_rejection_replay(include_full_wet=False),
    )

    assert report["finding"] == "active_pressure_support_cutoff_target_supported"
    assert report["status"] == "support_cutoff_candidate_needs_replay"
    classification = report["classification"]
    assert classification["tiny_cut_supported_branch_present"]
    assert not classification["full_wet_supported_branch_present"]
    assert classification["retained_fraction_cutoff_is_complete_fix_candidate"]
    assert not classification["retained_fraction_cutoff_is_diagnostic_only"]
