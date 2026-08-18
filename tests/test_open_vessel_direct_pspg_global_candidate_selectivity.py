import importlib.util
import sys
from pathlib import Path


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_direct_pspg_global_candidate_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_global_candidate_selectivity",
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
            {"label": "test02", "direct_pspg_target_global_dofs": list(range(7))},
            {"label": "test10", "direct_pspg_target_global_dofs": list(range(12))},
        ]
    }


def test_global_candidate_selectivity_flags_overbroad_matrix_proxy():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "covered_direct_target_global_dofs": list(range(7)),
                    "direct_self_positive_row_count": 866,
                    "sparse_direct_self_candidate_count": 545,
                    "low_direct_self_ratio_candidate_count": 86,
                    "low_direct_self_ratio_covered_direct_target_global_dofs": [0],
                    "moderate_direct_self_ratio_candidate_count": 321,
                    "moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "sparse_or_moderate_direct_self_ratio_candidate_count": 545,
                    "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "sparse_seeded_pressure_action_radius1_candidate_count": 35,
                    "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "sparse_seeded_pressure_action_radius2_candidate_count": 70,
                    "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "graph_local_low_direct_self_ratio_candidate_count": 24,
                    "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs": [
                        0,
                        1,
                    ],
                    "graph_local_moderate_direct_self_ratio_candidate_count": 70,
                    "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "pressure_action_moderate_degree_candidate_count": 70,
                    "pressure_action_moderate_degree_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "pressure_action_moderate_sum_ratio_candidate_count": 30,
                    "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs": [
                        0,
                        1,
                    ],
                    "pressure_action_self_dominant_candidate_count": 10,
                    "pressure_action_self_dominant_covered_direct_target_global_dofs": [
                        0
                    ],
                    "matrix_pressure_action_covered_count": 866,
                    "matrix_pressure_action_isolated_count": 0,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 866,
                    "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "preferred_candidate_count": 866,
                },
                {
                    "label": "test10",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "covered_direct_target_global_dofs": list(range(12)),
                    "direct_self_positive_row_count": 251,
                    "sparse_direct_self_candidate_count": 217,
                    "low_direct_self_ratio_candidate_count": 23,
                    "low_direct_self_ratio_covered_direct_target_global_dofs": [0, 1],
                    "moderate_direct_self_ratio_candidate_count": 96,
                    "moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "sparse_or_moderate_direct_self_ratio_candidate_count": 217,
                    "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "sparse_seeded_pressure_action_radius1_candidate_count": 80,
                    "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "sparse_seeded_pressure_action_radius2_candidate_count": 120,
                    "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "graph_local_low_direct_self_ratio_candidate_count": 18,
                    "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs": [
                        0,
                        1,
                        2,
                    ],
                    "graph_local_moderate_direct_self_ratio_candidate_count": 80,
                    "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "pressure_action_moderate_degree_candidate_count": 80,
                    "pressure_action_moderate_degree_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "pressure_action_moderate_sum_ratio_candidate_count": 20,
                    "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "pressure_action_self_dominant_candidate_count": 10,
                    "pressure_action_self_dominant_covered_direct_target_global_dofs": list(
                        range(10)
                    ),
                    "matrix_pressure_action_covered_count": 251,
                    "matrix_pressure_action_isolated_count": 0,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 251,
                    "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "preferred_candidate_count": 251,
                },
            ],
        },
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "global_candidate_selector_overbroad_matrix_proxy_not_formulation_ready"
    )
    cases = {case["label"]: case for case in report["cases"]}
    assert cases["test02"]["preferred_to_target_ratio"] == 866 / 7
    assert cases["test10"]["preferred_to_target_ratio"] == 251 / 12
    assert cases["test02"][
        "sparse_seeded_matrix_pressure_action_component_to_target_ratio"
    ] == 866 / 7
    assert cases["test02"][
        "sparse_or_moderate_direct_self_ratio_to_target_ratio"
    ] == 545 / 7
    assert cases["test02"][
        "direct_self_support_ratio_gate_finding"
    ] == "sparse_or_moderate_direct_self_ratio_gate_overbroad"
    assert cases["test02"][
        "graph_local_support_ratio_gate_finding"
    ] == "graph_local_moderate_direct_self_ratio_gate_overbroad"
    assert cases["test10"][
        "sparse_seeded_matrix_pressure_action_component_to_target_ratio"
    ] == 251 / 12
    assert cases["test02"][
        "sparse_seeded_pressure_action_radius1_to_target_ratio"
    ] == 35 / 7
    assert cases["test10"][
        "sparse_seeded_pressure_action_radius1_gate_finding"
    ] == "sparse_seeded_pressure_action_radius1_gate_overbroad"
    assert report["sparse_seeded_pressure_action_radius1_gate_finding"] == (
        "sparse_seeded_pressure_action_radius1_gate_overbroad"
    )
    assert cases["test02"]["matrix_pressure_action_covers_all_direct_rows"]
    assert cases["test10"]["matrix_pressure_action_covers_all_direct_rows"]
    assert cases["test02"][
        "sparse_seeded_matrix_pressure_action_component_covers_targets"
    ]
    assert cases["test02"]["raw_preferred_selector_overbroad"]
    assert cases["test10"]["raw_matrix_pressure_action_selector_overbroad"]
    assert cases["test10"][
        "sparse_seeded_matrix_pressure_action_component_selector_overbroad"
    ]
    assert report["direct_self_support_ratio_gate_finding"] == (
        "direct_self_support_ratio_gate_overbroad"
    )
    assert report["graph_local_support_ratio_gate_finding"] == (
        "graph_local_support_ratio_gate_overbroad"
    )
    assert report["pressure_action_moderate_degree_gate_finding"] == (
        "pressure_action_moderate_degree_gate_overbroad"
    )
    assert cases["test02"]["pressure_action_moderate_degree_to_target_ratio"] == (
        70 / 7
    )
    assert cases["test02"]["pressure_action_moderate_sum_ratio_gate_finding"] == (
        "pressure_action_moderate_sum_ratio_gate_misses_targets"
    )
    assert cases["test10"]["pressure_action_moderate_sum_ratio_gate_finding"] == (
        "pressure_action_moderate_sum_ratio_gate_selective"
    )
    assert "Do not promote raw global emitted candidates" in report[
        "next_requirement"
    ]


def test_global_candidate_selectivity_allows_small_selective_candidate_set():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "covered_direct_target_global_dofs": list(range(7)),
                    "direct_self_positive_row_count": 8,
                    "sparse_direct_self_candidate_count": 7,
                    "low_direct_self_ratio_candidate_count": 2,
                    "low_direct_self_ratio_covered_direct_target_global_dofs": [0, 1],
                    "moderate_direct_self_ratio_candidate_count": 7,
                    "moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "sparse_or_moderate_direct_self_ratio_candidate_count": 7,
                    "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "sparse_seeded_pressure_action_radius1_candidate_count": 7,
                    "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "sparse_seeded_pressure_action_radius2_candidate_count": 8,
                    "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "graph_local_low_direct_self_ratio_candidate_count": 2,
                    "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs": [0, 1],
                    "graph_local_moderate_direct_self_ratio_candidate_count": 7,
                    "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "pressure_action_moderate_degree_candidate_count": 7,
                    "pressure_action_moderate_degree_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "pressure_action_moderate_sum_ratio_candidate_count": 7,
                    "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "pressure_action_self_dominant_candidate_count": 7,
                    "pressure_action_self_dominant_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "matrix_pressure_action_covered_count": 8,
                    "matrix_pressure_action_isolated_count": 1,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 8,
                    "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs": list(
                        range(7)
                    ),
                    "preferred_candidate_count": 8,
                },
                {
                    "label": "test10",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "covered_direct_target_global_dofs": list(range(12)),
                    "direct_self_positive_row_count": 13,
                    "sparse_direct_self_candidate_count": 12,
                    "low_direct_self_ratio_candidate_count": 3,
                    "low_direct_self_ratio_covered_direct_target_global_dofs": [
                        0,
                        1,
                        2,
                    ],
                    "moderate_direct_self_ratio_candidate_count": 12,
                    "moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "sparse_or_moderate_direct_self_ratio_candidate_count": 12,
                    "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "sparse_seeded_pressure_action_radius1_candidate_count": 12,
                    "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "sparse_seeded_pressure_action_radius2_candidate_count": 13,
                    "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "graph_local_low_direct_self_ratio_candidate_count": 3,
                    "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs": [
                        0,
                        1,
                        2,
                    ],
                    "graph_local_moderate_direct_self_ratio_candidate_count": 12,
                    "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "pressure_action_moderate_degree_candidate_count": 12,
                    "pressure_action_moderate_degree_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "pressure_action_moderate_sum_ratio_candidate_count": 12,
                    "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "pressure_action_self_dominant_candidate_count": 12,
                    "pressure_action_self_dominant_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "matrix_pressure_action_covered_count": 13,
                    "matrix_pressure_action_isolated_count": 1,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 13,
                    "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "preferred_candidate_count": 13,
                },
            ],
        },
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "global_candidate_selector_selective_for_formulation_replay"
    )
    assert all(
        case["finding"] == "raw_global_candidate_selector_selective"
        for case in report["cases"]
    )
    assert report["direct_self_support_ratio_gate_finding"] == (
        "direct_self_support_ratio_gate_selective"
    )
    assert report["graph_local_support_ratio_gate_finding"] == (
        "graph_local_support_ratio_gate_selective"
    )
    assert report["pressure_action_moderate_degree_gate_finding"] == (
        "pressure_action_moderate_degree_gate_selective"
    )
    assert report["pressure_action_moderate_sum_ratio_gate_finding"] == (
        "pressure_action_moderate_sum_ratio_gate_selective"
    )
    assert report["pressure_action_self_dominant_gate_finding"] == (
        "pressure_action_self_dominant_gate_selective"
    )
    assert report["sparse_seeded_pressure_action_radius1_gate_finding"] == (
        "sparse_seeded_pressure_action_radius1_gate_selective"
    )
    assert report["sparse_seeded_pressure_action_radius2_gate_finding"] == (
        "sparse_seeded_pressure_action_radius2_gate_selective"
    )


def test_global_candidate_selectivity_flags_support_ratio_gate_misses():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_covers_audited_targets",
            "cases": [
                {
                    "label": "test10",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "covered_direct_target_global_dofs": list(range(12)),
                    "direct_self_positive_row_count": 13,
                    "sparse_direct_self_candidate_count": 6,
                    "low_direct_self_ratio_candidate_count": 2,
                    "low_direct_self_ratio_covered_direct_target_global_dofs": [0, 1],
                    "moderate_direct_self_ratio_candidate_count": 6,
                    "moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "sparse_or_moderate_direct_self_ratio_candidate_count": 6,
                    "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "sparse_seeded_pressure_action_radius1_candidate_count": 6,
                    "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "sparse_seeded_pressure_action_radius2_candidate_count": 13,
                    "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "graph_local_low_direct_self_ratio_candidate_count": 2,
                    "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs": [0, 1],
                    "graph_local_moderate_direct_self_ratio_candidate_count": 6,
                    "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "pressure_action_moderate_degree_candidate_count": 6,
                    "pressure_action_moderate_degree_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "pressure_action_moderate_sum_ratio_candidate_count": 6,
                    "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "pressure_action_self_dominant_candidate_count": 6,
                    "pressure_action_self_dominant_covered_direct_target_global_dofs": list(
                        range(6)
                    ),
                    "matrix_pressure_action_covered_count": 13,
                    "matrix_pressure_action_isolated_count": 1,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 13,
                    "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs": list(
                        range(12)
                    ),
                    "preferred_candidate_count": 13,
                },
            ],
        },
        target_map=_target_map(),
    )

    case = report["cases"][0]
    assert case["direct_self_support_ratio_gate_finding"] == (
        "sparse_or_moderate_direct_self_ratio_gate_misses_targets"
    )
    assert not case["sparse_or_moderate_direct_self_ratio_covers_targets"]
    assert report["direct_self_support_ratio_gate_finding"] == (
        "direct_self_support_ratio_gate_misses_targets"
    )
    assert report["graph_local_support_ratio_gate_finding"] == (
        "graph_local_support_ratio_gate_misses_targets"
    )
    assert report["pressure_action_moderate_degree_gate_finding"] == (
        "pressure_action_moderate_degree_gate_misses_targets"
    )
    assert report["sparse_seeded_pressure_action_radius1_gate_finding"] == (
        "sparse_seeded_pressure_action_radius1_gate_misses_targets"
    )
    assert report["sparse_seeded_pressure_action_radius2_gate_finding"] == (
        "sparse_seeded_pressure_action_radius2_gate_selective"
    )


def test_global_candidate_selectivity_requires_complete_emission_coverage():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "finding": "candidate_emission_misses_audited_targets",
            "cases": [
                {
                    "label": "test02",
                    "finding": "candidate_emitted_but_misses_targets",
                    "covered_direct_target_global_dofs": [0],
                    "direct_self_positive_row_count": 1,
                    "preferred_candidate_count": 1,
                },
            ],
        },
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "global_candidate_emission_not_ready_for_selectivity"
    )
