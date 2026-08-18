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
        / "audit_direct_pspg_null_balance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_null_balance_selectivity",
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


def test_null_balance_selectivity_classifies_overbroad_and_misses():
    audit = _load_audit_module()
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "direct_self_row_sum_leak_threshold": 0.25,
                "direct_self_null_preserving_threshold": 0.05,
                "direct_self_diag_dominant_threshold": 0.6,
                "direct_self_balanced_diag_low_threshold": 0.45,
                "direct_self_balanced_diag_high_threshold": 0.55,
                "max_direct_self_row_sum_leak_ratio": 0.8,
                "min_direct_self_diag_abs_ratio": 0.2,
                "max_direct_self_diag_abs_ratio": 0.8,
                "high_direct_self_row_sum_leak_candidate_count": 12,
                "high_direct_self_row_sum_leak_covered_direct_target_global_dofs": [
                    100,
                ],
                "high_direct_self_row_sum_leak_uncovered_direct_target_global_dofs": [
                    101,
                ],
                "null_preserving_direct_self_candidate_count": 12,
                "null_preserving_direct_self_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "null_preserving_direct_self_uncovered_direct_target_global_dofs": [],
                "diag_dominant_direct_self_candidate_count": 1,
                "diag_dominant_direct_self_covered_direct_target_global_dofs": [],
                "diag_dominant_direct_self_uncovered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "balanced_diag_direct_self_candidate_count": 12,
                "balanced_diag_direct_self_covered_direct_target_global_dofs": [
                    100,
                    101,
                ],
                "balanced_diag_direct_self_uncovered_direct_target_global_dofs": [],
            },
            {
                "label": "test10",
                "high_direct_self_row_sum_leak_candidate_count": 0,
                "high_direct_self_row_sum_leak_covered_direct_target_global_dofs": [],
                "high_direct_self_row_sum_leak_uncovered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "null_preserving_direct_self_candidate_count": 12,
                "null_preserving_direct_self_covered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "null_preserving_direct_self_uncovered_direct_target_global_dofs": [],
                "diag_dominant_direct_self_candidate_count": 0,
                "diag_dominant_direct_self_covered_direct_target_global_dofs": [],
                "diag_dominant_direct_self_uncovered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "balanced_diag_direct_self_candidate_count": 12,
                "balanced_diag_direct_self_covered_direct_target_global_dofs": [
                    200,
                    201,
                ],
                "balanced_diag_direct_self_uncovered_direct_target_global_dofs": [],
            },
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
        max_target_ratio=5.0,
    )

    assert report["finding"] == (
        "direct_pspg_null_balance_selectors_overbroad_or_miss_targets"
    )
    assert report["null_balance_by_case"]["test02"][
        "max_direct_self_row_sum_leak_ratio"
    ] == 0.8
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    assert selectors["high_direct_self_row_sum_leak"]["finding"] == (
        "selector_overbroad_or_miss_targets"
    )
    assert selectors["null_preserving_direct_self"]["finding"] == (
        "selector_overbroad"
    )
    assert selectors["diag_dominant_direct_self"]["finding"] == (
        "selector_misses_targets"
    )


def test_null_balance_selectivity_flags_missing_evidence():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={"cases": [{"label": "test02"}]},
        target_map=_target_map(),
    )

    assert report["finding"] == "direct_pspg_null_balance_selector_evidence_missing"
    assert "high_direct_self_row_sum_leak" in report["missing_selector_keys"]


def test_null_balance_selectivity_cli_writes_json(tmp_path):
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
                        "balanced_diag_direct_self_candidate_count": 1,
                        "balanced_diag_direct_self_covered_direct_target_global_dofs": [
                            100
                        ],
                        "balanced_diag_direct_self_uncovered_direct_target_global_dofs": [
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
