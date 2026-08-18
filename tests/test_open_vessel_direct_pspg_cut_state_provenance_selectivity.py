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
        / "audit_direct_pspg_cut_state_provenance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_state_provenance_selectivity",
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


def _profile(*, inactive_dofs=(), dry_dofs=(), cut_dofs=()):
    profile = {}
    for dof in range(90, 230):
        inactive = dof in inactive_dofs
        dry = dof in dry_dofs
        cut = dof in cut_dofs
        if dry:
            wet_support_class = "dry_only_incident_support"
        elif cut:
            wet_support_class = "mixed_cut_dry_incident_support"
        else:
            wet_support_class = "full_wet_incident_support"
        profile[dof] = {
            "active_class": "inactive_point" if inactive else "active_point",
            "phi_class": "positive_phi" if inactive else "negative_phi",
            "wet_support_class": wet_support_class,
            "cut_incident_cell_count": 1 if cut else 0,
        }
    return profile


def test_cut_state_provenance_rules_out_simple_cut_state_selectors():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "preferred_candidate_global_dofs": list(range(100, 112)),
                    "sparse_or_moderate_direct_self_ratio_global_dofs": [
                        100,
                        102,
                        103,
                    ],
                },
                {
                    "label": "test10",
                    "preferred_candidate_global_dofs": list(range(200, 216)),
                    "sparse_or_moderate_direct_self_ratio_global_dofs": [
                        200,
                        202,
                        203,
                    ],
                },
            ],
        },
        target_map=_target_map(),
        profiles_by_label={
            "test02": _profile(
                inactive_dofs=set(range(100, 112)),
                dry_dofs=set(range(100, 112)) - {101},
                cut_dofs={101},
            ),
            "test10": _profile(
                inactive_dofs=set(range(200, 216)),
                dry_dofs=set(range(200, 216)) - {201},
                cut_dofs={201},
            ),
        },
    )

    assert report["finding"] == (
        "cut_state_provenance_selectors_overbroad_or_miss_targets"
    )
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    preferred_inactive = selectors["preferred_inactive_point"]
    assert preferred_inactive["finding"] == "selector_overbroad"
    cases = {case["label"]: case for case in preferred_inactive["cases"]}
    assert cases["test02"]["selected_to_target_ratio"] == 6.0
    assert cases["test10"]["selected_to_target_ratio"] == 16 / 3

    preferred_dry = selectors["preferred_dry_only_incident_support"]
    assert preferred_dry["finding"] == "selector_misses_targets"
    dry_cases = {case["label"]: case for case in preferred_dry["cases"]}
    assert dry_cases["test02"]["covered_direct_target_global_dofs"] == [100]
    assert dry_cases["test02"]["uncovered_direct_target_global_dofs"] == [101]
    assert dry_cases["test10"]["covered_direct_target_global_dofs"] == [
        200,
        202,
    ]
    assert dry_cases["test10"]["uncovered_direct_target_global_dofs"] == [201]


def test_cut_state_provenance_can_flag_a_selective_selector():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "preferred_candidate_global_dofs": [100, 101],
                    "sparse_or_moderate_direct_self_ratio_global_dofs": [],
                },
                {
                    "label": "test10",
                    "preferred_candidate_global_dofs": [200, 201, 202],
                    "sparse_or_moderate_direct_self_ratio_global_dofs": [],
                },
            ],
        },
        target_map=_target_map(),
        profiles_by_label={
            "test02": _profile(
                inactive_dofs={100, 101},
                dry_dofs={100, 101},
            ),
            "test10": _profile(
                inactive_dofs={200, 201, 202},
                dry_dofs={200, 201, 202},
            ),
        },
    )

    assert report["finding"] == (
        "cut_state_provenance_selector_selective_for_formulation_replay"
    )
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    assert selectors["preferred_inactive_point"]["finding"] == "selector_selective"
    assert selectors["preferred_dry_only_incident_support"]["finding"] == (
        "selector_selective"
    )
