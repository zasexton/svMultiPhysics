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
        / "audit_direct_pspg_active_pressure_support_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_active_pressure_support_selectivity",
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


def test_active_pressure_support_selectivity_flags_overbroad_or_missing_gate():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "constrained_pressure_neighbor_candidate_count": 40,
                    "constrained_pressure_neighbor_covered_direct_target_global_dofs": [
                        100,
                        101,
                    ],
                    "constrained_pressure_neighbor_uncovered_direct_target_global_dofs": [],
                    "high_constrained_pressure_neighbor_ratio_candidate_count": 1,
                    "high_constrained_pressure_neighbor_ratio_covered_direct_target_global_dofs": [
                        100
                    ],
                    "high_constrained_pressure_neighbor_ratio_uncovered_direct_target_global_dofs": [
                        101
                    ],
                    "sparse_unconstrained_direct_self_candidate_count": 3,
                    "sparse_unconstrained_direct_self_covered_direct_target_global_dofs": [
                        100
                    ],
                    "sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs": [
                        101
                    ],
                    "constrained_or_sparse_unconstrained_direct_self_candidate_count": 50,
                    "constrained_or_sparse_unconstrained_direct_self_covered_direct_target_global_dofs": [
                        100,
                        101,
                    ],
                    "constrained_or_sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs": [],
                },
                {
                    "label": "test10",
                    "constrained_pressure_neighbor_candidate_count": 30,
                    "constrained_pressure_neighbor_covered_direct_target_global_dofs": [
                        200,
                        201,
                        202,
                    ],
                    "constrained_pressure_neighbor_uncovered_direct_target_global_dofs": [],
                    "high_constrained_pressure_neighbor_ratio_candidate_count": 2,
                    "high_constrained_pressure_neighbor_ratio_covered_direct_target_global_dofs": [
                        200,
                        201,
                    ],
                    "high_constrained_pressure_neighbor_ratio_uncovered_direct_target_global_dofs": [
                        202
                    ],
                    "sparse_unconstrained_direct_self_candidate_count": 20,
                    "sparse_unconstrained_direct_self_covered_direct_target_global_dofs": [
                        200,
                        201,
                        202,
                    ],
                    "sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs": [],
                    "constrained_or_sparse_unconstrained_direct_self_candidate_count": 30,
                    "constrained_or_sparse_unconstrained_direct_self_covered_direct_target_global_dofs": [
                        200,
                        201,
                        202,
                    ],
                    "constrained_or_sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs": [],
                },
            ],
        },
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "active_pressure_support_topology_selectors_overbroad_or_miss_targets"
    )
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    constrained = selectors["constrained_pressure_neighbor"]
    assert constrained["finding"] == "selector_overbroad"
    constrained_cases = {case["label"]: case for case in constrained["cases"]}
    assert constrained_cases["test02"]["selected_to_target_ratio"] == 20.0
    assert constrained_cases["test10"]["selected_to_target_ratio"] == 10.0

    high_ratio = selectors["high_constrained_pressure_neighbor_ratio"]
    assert high_ratio["finding"] == "selector_misses_targets"
    assert high_ratio["cases"][0]["uncovered_direct_target_global_dofs"] == [
        101
    ]
    assert "constrained_pressure_neighbor" in report["overbroad_selector_keys"]
    assert "high_constrained_pressure_neighbor_ratio" in report[
        "miss_selector_keys"
    ]


def test_active_pressure_support_selectivity_reports_selective_gate():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "constrained_pressure_neighbor_candidate_count": 2,
                    "constrained_pressure_neighbor_covered_direct_target_global_dofs": [
                        100,
                        101,
                    ],
                    "constrained_pressure_neighbor_uncovered_direct_target_global_dofs": [],
                },
                {
                    "label": "test10",
                    "constrained_pressure_neighbor_candidate_count": 3,
                    "constrained_pressure_neighbor_covered_direct_target_global_dofs": [
                        200,
                        201,
                        202,
                    ],
                    "constrained_pressure_neighbor_uncovered_direct_target_global_dofs": [],
                },
            ],
        },
        target_map=_target_map(),
    )

    assert report["finding"] == "active_pressure_support_topology_selector_selective"
    assert report["selective_selector_keys"] == ["constrained_pressure_neighbor"]
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    assert selectors["constrained_pressure_neighbor"]["finding"] == (
        "selector_selective"
    )
    assert selectors["high_constrained_pressure_neighbor_ratio"]["finding"] == (
        "selector_evidence_missing"
    )
