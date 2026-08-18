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
        / "audit_direct_pspg_residual_sign_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_residual_sign_selectivity",
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


def test_residual_sign_selectivity_classifies_overbroad_and_misses():
    audit = _load_audit_module()
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "residual_nonzero_direct_row_count": 20,
                "residual_positive_direct_row_count": 14,
                "residual_negative_direct_row_count": 6,
                "residual_zero_direct_row_count": 0,
                "residual_nonfinite_direct_row_count": 0,
                "residual_sign_pressure_action_edge_count": 9,
                "residual_opposite_sign_pressure_action_edge_count": 3,
                "residual_zero_or_missing_sign_pressure_action_edge_count": 0,
                "residual_sign_pressure_action_covered_count": 12,
                "residual_sign_pressure_action_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "residual_sign_pressure_action_uncovered_direct_target_global_dofs": [],
                "sparse_seeded_residual_sign_pressure_action_component_dof_count": 12,
                "sparse_seeded_residual_sign_pressure_action_component_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "sparse_seeded_residual_sign_pressure_action_component_uncovered_direct_target_global_dofs": [],
                "sparse_direct_self_or_residual_sign_pressure_action_candidate_count": 12,
                "sparse_direct_self_or_residual_sign_pressure_action_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "sparse_direct_self_or_residual_sign_pressure_action_uncovered_direct_target_global_dofs": [],
            },
            {
                "label": "test10",
                "residual_nonzero_direct_row_count": 5,
                "residual_positive_direct_row_count": 5,
                "residual_negative_direct_row_count": 0,
                "residual_zero_direct_row_count": 0,
                "residual_nonfinite_direct_row_count": 0,
                "residual_sign_pressure_action_edge_count": 1,
                "residual_opposite_sign_pressure_action_edge_count": 0,
                "residual_zero_or_missing_sign_pressure_action_edge_count": 0,
                "residual_sign_pressure_action_covered_count": 1,
                "residual_sign_pressure_action_covered_direct_target_global_dofs": [
                    200,
                ],
                "residual_sign_pressure_action_uncovered_direct_target_global_dofs": [
                    201,
                ],
                "sparse_seeded_residual_sign_pressure_action_component_dof_count": 1,
                "sparse_seeded_residual_sign_pressure_action_component_covered_direct_target_global_dofs": [
                    200,
                ],
                "sparse_seeded_residual_sign_pressure_action_component_uncovered_direct_target_global_dofs": [
                    201,
                ],
                "sparse_direct_self_or_residual_sign_pressure_action_candidate_count": 1,
                "sparse_direct_self_or_residual_sign_pressure_action_covered_direct_target_global_dofs": [
                    200,
                ],
                "sparse_direct_self_or_residual_sign_pressure_action_uncovered_direct_target_global_dofs": [
                    201,
                ],
            },
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
        max_target_ratio=5.0,
    )

    assert report["finding"] == (
        "residual_sign_pressure_action_selectors_overbroad_or_miss_targets"
    )
    assert report["residual_signal_by_case"]["test02"][
        "residual_sign_pressure_action_edge_count"
    ] == 9
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    assert selectors["residual_sign_pressure_action"]["finding"] == (
        "selector_overbroad_or_miss_targets"
    )
    assert selectors[
        "sparse_direct_self_or_residual_sign_pressure_action"
    ]["cases"][1]["uncovered_direct_target_global_dofs"] == [201]


def test_residual_sign_selectivity_flags_missing_evidence():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={"cases": [{"label": "test02"}]},
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "residual_sign_pressure_action_selector_evidence_missing"
    )
    assert "residual_sign_pressure_action" in report["missing_selector_keys"]


def test_residual_sign_selectivity_cli_writes_json(tmp_path):
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
                        "residual_sign_pressure_action_covered_count": 1,
                        "residual_sign_pressure_action_covered_direct_target_global_dofs": [
                            100
                        ],
                        "residual_sign_pressure_action_uncovered_direct_target_global_dofs": [
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
