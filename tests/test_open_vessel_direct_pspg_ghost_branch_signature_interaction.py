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
        / "audit_direct_pspg_ghost_branch_signature_interaction.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_ghost_branch_signature_interaction",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _top_case(
    label,
    *,
    finding,
    top_rows,
    direct_dofs,
    ghost_dofs,
):
    return {
        "label": label,
        "finding": finding,
        "top_update_rows": [
            {"global_dof": dof, "abs_update": update} for dof, update in top_rows
        ],
        "direct_pspg_balance_global_dofs": direct_dofs,
        "ghost_penalty_balance_global_dofs": ghost_dofs,
    }


def _branch(still_triggers=True):
    return {
        "finding": "pressure_policy_shapes_branch_but_does_not_clear_guard",
        "classification_counts": {"guard_still_triggered": 1},
        "baseline": {"worst_active_abs_pressure_delta_pa": 10.0},
        "controls": [
            {
                "label": "pressure_disabled",
                "classification": "guard_still_triggered",
                "still_triggers": still_triggers,
                "point_shift": True,
                "support_class_shift": True,
                "active_or_wet_supported": {
                    "control_abs_pressure_delta_pa": 20.0,
                    "ratio_to_baseline": 2.0,
                },
            }
        ],
    }


def _signature(test02_ratio=39.4, test10_ratio=4.0):
    return {
        "finding": (
            "solve_time_direct_pspg_support_coupling_signature_partial_"
            "test10_only"
        ),
        "cases": [
            {
                "label": "test02",
                "finding": (
                    "solve_time_support_coupling_signature_covers_targets_"
                    "but_overbroad"
                ),
                "target_same_parent_pressure_velocity_support_class_counts": {
                    "full": 7,
                    "partial": 0,
                    "none": 0,
                },
                "exact_local_signature_selected_count": 276,
                "exact_local_signature_selected_to_target_ratio": test02_ratio,
            },
            {
                "label": "test10",
                "finding": (
                    "solve_time_support_coupling_signature_selective_candidate"
                ),
                "target_same_parent_pressure_velocity_support_class_counts": {
                    "full": 0,
                    "partial": 7,
                    "none": 5,
                },
                "exact_local_signature_selected_count": 48,
                "exact_local_signature_selected_to_target_ratio": test10_ratio,
            },
        ],
    }


def test_ghost_branch_signature_interaction_rules_out_common_gate():
    audit = _load_audit_module()
    report = audit.build_report(
        signature=_signature(),
        top_provenance={
            "cases": [
                _top_case(
                    "test02",
                    finding="mixed_direct_pspg_and_ghost_penalty_top_rows",
                    top_rows=[(10676, 10.0), (10624, 8.0)],
                    direct_dofs=[10676],
                    ghost_dofs=[10624],
                ),
                _top_case(
                    "test10",
                    finding="direct_pspg_top_rows_without_ghost_penalty",
                    top_rows=[(3526, 1.0)],
                    direct_dofs=[3526],
                    ghost_dofs=[],
                ),
            ]
        },
        pressure_disabled_top_provenance={
            "cases": [
                _top_case(
                    "test02_pressure_disabled",
                    finding="direct_pspg_top_rows_without_ghost_penalty",
                    top_rows=[(10676, 20.0), (10668, 15.0)],
                    direct_dofs=[10676, 10668],
                    ghost_dofs=[],
                ),
                _top_case(
                    "test10_pressure_disabled",
                    finding="direct_pspg_top_rows_without_ghost_penalty",
                    top_rows=[(3850, 2.0)],
                    direct_dofs=[3850],
                    ghost_dofs=[],
                ),
            ]
        },
        test02_branch=_branch(),
        test10_branch=_branch(),
    )

    assert report["finding"] == (
        "direct_pspg_ghost_branch_signature_interaction_rules_out_common_gate"
    )
    assert report["status"] == (
        "ghost_branch_is_branch_shaper_not_support_coupling_signature_fix"
    )

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == (
        "ghost_branch_shapes_test02_but_cannot_narrow_signature"
    )
    assert test02["row_10676_baseline_update_pa"] == 10.0
    assert test02["row_10676_pressure_disabled_update_pa"] == 20.0
    assert test02["baseline_ghost_penalty_global_dofs"] == [10624]
    assert test02["pressure_disabled_ghost_penalty_global_dofs"] == []
    assert test02["signature"]["exact_local_signature_selected_to_target_ratio"] == (
        39.4
    )
    assert test02["branch_policy"]["pressure_disabled_still_triggers"] is True

    test10 = next(case for case in report["cases"] if case["label"] == "test10")
    assert test10["finding"] == (
        "ghost_absent_test10_signature_candidate_remains_partial_fix"
    )
    assert test10["baseline_ghost_penalty_global_dofs"] == []
    assert test10["pressure_disabled_ghost_penalty_global_dofs"] == []
    assert test10["signature"]["exact_local_signature_selected_to_target_ratio"] == (
        4.0
    )


def test_ghost_branch_signature_interaction_remains_inconclusive_without_worsening():
    audit = _load_audit_module()
    report = audit.build_report(
        signature=_signature(),
        top_provenance={
            "cases": [
                _top_case(
                    "test02",
                    finding="mixed_direct_pspg_and_ghost_penalty_top_rows",
                    top_rows=[(10676, 20.0), (10624, 8.0)],
                    direct_dofs=[10676],
                    ghost_dofs=[10624],
                ),
                _top_case(
                    "test10",
                    finding="direct_pspg_top_rows_without_ghost_penalty",
                    top_rows=[(3526, 1.0)],
                    direct_dofs=[3526],
                    ghost_dofs=[],
                ),
            ]
        },
        pressure_disabled_top_provenance={
            "cases": [
                _top_case(
                    "test02_pressure_disabled",
                    finding="direct_pspg_top_rows_without_ghost_penalty",
                    top_rows=[(10676, 10.0)],
                    direct_dofs=[10676],
                    ghost_dofs=[],
                ),
                _top_case(
                    "test10_pressure_disabled",
                    finding="direct_pspg_top_rows_without_ghost_penalty",
                    top_rows=[(3850, 2.0)],
                    direct_dofs=[3850],
                    ghost_dofs=[],
                ),
            ]
        },
        test02_branch=_branch(),
        test10_branch=_branch(),
    )

    assert report["finding"] == (
        "direct_pspg_ghost_branch_signature_interaction_inconclusive"
    )
    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == "ghost_branch_test02_interaction_inconclusive"
