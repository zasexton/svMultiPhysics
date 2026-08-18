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
        / "audit_direct_pspg_boundary_provenance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_boundary_provenance_selectivity",
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


def _profile(boundary_dofs, low_incident_dofs=()):
    profile = {}
    for dof in range(90, 230):
        boundary = dof in boundary_dofs
        incident = 2 if dof in low_incident_dofs else 4
        bclass = "boundary_face" if boundary else "interior"
        profile[dof] = {
            "boundary_class": bclass,
            "incident_cell_count": incident,
            "incident_support_class": (
                "shared_boundary_support"
                if boundary
                else "interior_shared_support"
            ),
        }
    return profile


def test_boundary_provenance_selectivity_rules_out_literal_mesh_filters():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "preferred_candidate_global_dofs": [100, 101, 102, 103],
                    "sparse_direct_self_global_dofs": [100, 102],
                    "sparse_or_moderate_direct_self_ratio_global_dofs": [
                        100,
                        102,
                    ],
                },
                {
                    "label": "test10",
                    "preferred_candidate_global_dofs": [200, 201, 202, 203],
                    "sparse_direct_self_global_dofs": [200, 202],
                    "sparse_or_moderate_direct_self_ratio_global_dofs": [
                        200,
                        202,
                    ],
                },
            ],
        },
        target_map=_target_map(),
        profiles_by_label={
            "test02": _profile(boundary_dofs={100, 102, 103}),
            "test10": _profile(boundary_dofs={200, 202, 203}),
        },
    )

    assert report["finding"] == (
        "mesh_boundary_incident_support_selectors_miss_audited_targets"
    )
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    preferred_boundary = selectors["preferred_boundary_only"]
    assert preferred_boundary["finding"] == "selector_misses_targets"
    cases = {case["label"]: case for case in preferred_boundary["cases"]}
    assert cases["test02"]["covered_direct_target_global_dofs"] == [100]
    assert cases["test02"]["uncovered_direct_target_global_dofs"] == [101]
    assert cases["test10"]["covered_direct_target_global_dofs"] == [200, 202]
    assert cases["test10"]["uncovered_direct_target_global_dofs"] == [201]


def test_boundary_provenance_selectivity_flags_overbroad_selector():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "preferred_candidate_global_dofs": list(range(100, 112)),
                    "sparse_direct_self_global_dofs": list(range(100, 112)),
                    "sparse_or_moderate_direct_self_ratio_global_dofs": (
                        list(range(100, 112))
                    ),
                },
                {
                    "label": "test10",
                    "preferred_candidate_global_dofs": list(range(200, 216)),
                    "sparse_direct_self_global_dofs": list(range(200, 216)),
                    "sparse_or_moderate_direct_self_ratio_global_dofs": (
                        list(range(200, 216))
                    ),
                },
            ],
        },
        target_map=_target_map(),
        profiles_by_label={
            "test02": _profile(boundary_dofs=set(range(100, 112))),
            "test10": _profile(boundary_dofs=set(range(200, 216))),
        },
    )

    selectors = {selector["key"]: selector for selector in report["selectors"]}
    assert selectors["preferred_boundary_only"]["finding"] == "selector_overbroad"
    cases = {
        case["label"]: case
        for case in selectors["preferred_boundary_only"]["cases"]
    }
    assert cases["test02"]["selected_to_target_ratio"] == 6.0
    assert cases["test10"]["selected_to_target_ratio"] == 16 / 3
    assert report["finding"] == "mesh_boundary_incident_support_selectors_overbroad"
