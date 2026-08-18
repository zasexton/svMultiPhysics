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
        / "audit_direct_pspg_coupled_patch_graph_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_coupled_patch_graph_selectivity",
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
            {"label": "test10", "direct_pspg_target_global_dofs": [200, 201]},
        ]
    }


def test_coupled_patch_graph_selectivity_classifies_overbroad_and_misses():
    audit = _load_audit_module()
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "pressure_action_low_two_hop_threshold": 4,
                "pressure_action_high_two_hop_ratio_threshold": 0.5,
                "matrix_pressure_action_max_two_hop_completion_count": 8,
                "pressure_action_low_clustering_threshold": 0.25,
                "pressure_action_high_clustering_threshold": 0.75,
                "pressure_action_clustering_eligible_row_count": 10,
                "matrix_pressure_action_min_clustering_ratio": 0.0,
                "matrix_pressure_action_max_clustering_ratio": 1.0,
                "pressure_action_zero_two_hop_candidate_count": 1,
                "pressure_action_zero_two_hop_covered_direct_target_global_dofs": [
                    100,
                ],
                "pressure_action_zero_two_hop_uncovered_direct_target_global_dofs": [
                    101,
                ],
                "pressure_action_low_two_hop_candidate_count": 12,
                "pressure_action_low_two_hop_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "pressure_action_low_two_hop_uncovered_direct_target_global_dofs": [],
                "pressure_action_high_two_hop_candidate_count": 1,
                "pressure_action_high_two_hop_covered_direct_target_global_dofs": [],
                "pressure_action_high_two_hop_uncovered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "pressure_action_zero_clustering_candidate_count": 12,
                "pressure_action_zero_clustering_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "pressure_action_zero_clustering_uncovered_direct_target_global_dofs": [],
                "pressure_action_low_clustering_candidate_count": 12,
                "pressure_action_low_clustering_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "pressure_action_low_clustering_uncovered_direct_target_global_dofs": [],
                "pressure_action_high_clustering_candidate_count": 1,
                "pressure_action_high_clustering_covered_direct_target_global_dofs": [
                    100,
                ],
                "pressure_action_high_clustering_uncovered_direct_target_global_dofs": [
                    101,
                ],
                "pressure_action_articulation_candidate_count": 1,
                "pressure_action_articulation_covered_direct_target_global_dofs": [
                    101,
                ],
                "pressure_action_articulation_uncovered_direct_target_global_dofs": [
                    100,
                ],
                "pressure_action_bridge_endpoint_candidate_count": 12,
                "pressure_action_bridge_endpoint_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "pressure_action_bridge_endpoint_uncovered_direct_target_global_dofs": [],
            },
            {
                "label": "test10",
                "pressure_action_zero_two_hop_candidate_count": 0,
                "pressure_action_zero_two_hop_covered_direct_target_global_dofs": [],
                "pressure_action_zero_two_hop_uncovered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_low_two_hop_candidate_count": 12,
                "pressure_action_low_two_hop_covered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_low_two_hop_uncovered_direct_target_global_dofs": [],
                "pressure_action_high_two_hop_candidate_count": 0,
                "pressure_action_high_two_hop_covered_direct_target_global_dofs": [],
                "pressure_action_high_two_hop_uncovered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_zero_clustering_candidate_count": 12,
                "pressure_action_zero_clustering_covered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_zero_clustering_uncovered_direct_target_global_dofs": [],
                "pressure_action_low_clustering_candidate_count": 12,
                "pressure_action_low_clustering_covered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_low_clustering_uncovered_direct_target_global_dofs": [],
                "pressure_action_high_clustering_candidate_count": 1,
                "pressure_action_high_clustering_covered_direct_target_global_dofs": [],
                "pressure_action_high_clustering_uncovered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_articulation_candidate_count": 0,
                "pressure_action_articulation_covered_direct_target_global_dofs": [],
                "pressure_action_articulation_uncovered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_bridge_endpoint_candidate_count": 12,
                "pressure_action_bridge_endpoint_covered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "pressure_action_bridge_endpoint_uncovered_direct_target_global_dofs": [],
            },
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
        max_target_ratio=5.0,
    )

    assert report["finding"] == (
        "direct_pspg_coupled_patch_graph_selectors_overbroad_or_miss_targets"
    )
    assert report["graph_topology_by_case"]["test02"][
        "matrix_pressure_action_max_two_hop_completion_count"
    ] == 8
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    assert selectors["pressure_action_zero_two_hop"]["finding"] == (
        "selector_misses_targets"
    )
    assert selectors["pressure_action_low_two_hop"]["finding"] == (
        "selector_overbroad"
    )
    assert selectors["pressure_action_articulation"]["finding"] == (
        "selector_misses_targets"
    )
    assert selectors["pressure_action_bridge_endpoint"]["finding"] == (
        "selector_overbroad"
    )


def test_coupled_patch_graph_selectivity_flags_missing_evidence():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={"cases": [{"label": "test02"}]},
        target_map=_target_map(),
    )

    assert (
        report["finding"]
        == "direct_pspg_coupled_patch_graph_selector_evidence_missing"
    )
    assert "pressure_action_low_two_hop" in report["missing_selector_keys"]


def test_coupled_patch_graph_selectivity_cli_writes_json(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    out_path = tmp_path / "out.json"
    global_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": "test02",
                        "pressure_action_low_clustering_candidate_count": 1,
                        "pressure_action_low_clustering_covered_direct_target_global_dofs": [
                            100
                        ],
                        "pressure_action_low_clustering_uncovered_direct_target_global_dofs": [
                            101
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_path.write_text(json.dumps(_target_map()), encoding="utf-8")

    report = audit.build_report(
        global_emission=json.loads(global_path.read_text(encoding="utf-8")),
        target_map=json.loads(target_path.read_text(encoding="utf-8")),
        global_emission_path=global_path,
        target_map_path=target_path,
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["global_emission_path"] == str(global_path)
    assert parsed["target_map_path"] == str(target_path)
