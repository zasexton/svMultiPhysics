import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/cases/fluid/open_vessel_free_surface/"
    "audit_direct_pspg_same_rule_cross_block_broad_union_branch_shift.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_direct_pspg_same_rule_cross_block_broad_union_branch_shift",
    SCRIPT,
)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def _pressure_summary(update: float, point: int, support: str) -> dict:
    return {
        "exists": True,
        "path": "synthetic.json",
        "status": "diagnostic_pressure_update_guard_triggered",
        "guard_triggered": True,
        "guard_cleared": False,
        "worst_active_or_wet_update_pa": update,
        "worst_active_or_wet_support_class": support,
        "worst_active_or_wet": {
            "abs_pressure_delta_pa": update,
            "point_index": point,
            "support_class": support,
            "active_fluid": 1.0,
            "incident_wet_fraction_min_positive": 1.0,
        },
    }


def _point_event(update: float, point: int, support: str) -> dict:
    return {
        "point_index": point,
        "pressure_delta_pa": update,
        "abs_pressure_delta_pa": abs(update),
        "support_class": support,
        "active_fluid": 1.0,
        "incident_wet_fraction_min_positive": (
            1.0 if support == "full_wet_supported" else 1.0e-5
        ),
    }


def _case(
    *,
    label: str,
    threshold: float,
    reference_point: int,
    broad_worst_point: int,
    broad_worst_support: str,
    reference_updates: dict[str, float],
) -> dict:
    variant_updates = {
        "no_policy": _pressure_summary(
            reference_updates["no_policy"],
            reference_point,
            "full_wet_supported",
        ),
        "same_rule_parent": _pressure_summary(
            reference_updates["same_rule_parent"],
            reference_point,
            "full_wet_supported",
        ),
        "broad_minus_parent": _pressure_summary(
            reference_updates["broad_minus_parent"],
            reference_point,
            "full_wet_supported",
        ),
        "broad_policy": _pressure_summary(
            reference_updates.get("broad_worst", reference_updates["broad_policy"]),
            broad_worst_point,
            broad_worst_support,
        ),
    }
    point_events = {
        str(reference_point): {
            variant: _point_event(update, reference_point, "full_wet_supported")
            for variant, update in reference_updates.items()
            if variant in audit.VARIANT_ORDER
        }
    }
    case = {
        "label": label,
        "absolute_threshold_pa": threshold,
        "reference_point": reference_point,
        "variant_pressure_updates": variant_updates,
        "point_events": point_events,
        "missing_paths": [],
    }
    case["flags"] = audit.classify_case(case)
    case["finding"] = audit.case_finding(case)
    return case


def test_broad_union_branch_shift_classification_supported() -> None:
    test02 = _case(
        label="test02",
        threshold=100.0,
        reference_point=1172,
        broad_worst_point=1170,
        broad_worst_support="tiny_cut_supported",
        reference_updates={
            "no_policy": 360.0,
            "same_rule_parent": 320.0,
            "broad_minus_parent": 365.0,
            "broad_policy": 80.0,
            "broad_worst": 175.0,
        },
    )
    test10 = _case(
        label="test10",
        threshold=10.0,
        reference_point=83,
        broad_worst_point=83,
        broad_worst_support="full_wet_supported",
        reference_updates={
            "no_policy": 62.0,
            "same_rule_parent": 57.0,
            "broad_minus_parent": 58.0,
            "broad_policy": 52.0,
        },
    )

    report = audit.classify_report([test02, test10])

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_union_branch_shift_supported"
    )
    assert report["status"] == (
        "broad_union_reduces_full_wet_but_shifts_test02_tiny_cut"
    )
    assert report["test02_branch_shift_supported"]
    assert report["test10_broad_union_residual_guard_supported"]
    assert report["all_variants_guard_triggered"]
    assert test02["finding"] == (
        "broad_union_reduces_full_wet_reference_then_shifts_to_tiny_cut"
    )
    assert test10["finding"] == (
        "broad_union_reduces_shared_full_wet_reference_but_guard_remains"
    )


def test_broad_union_consistent_replays_still_trigger_full_wet_guards() -> None:
    test02 = _case(
        label="test02",
        threshold=100.0,
        reference_point=1172,
        broad_worst_point=1172,
        broad_worst_support="full_wet_supported",
        reference_updates={
            "no_policy": 360.0,
            "same_rule_parent": 322.0,
            "broad_minus_parent": 365.0,
            "broad_policy": 321.0,
        },
    )
    test10 = _case(
        label="test10",
        threshold=10.0,
        reference_point=83,
        broad_worst_point=83,
        broad_worst_support="full_wet_supported",
        reference_updates={
            "no_policy": 62.0,
            "same_rule_parent": 57.0,
            "broad_minus_parent": 58.0,
            "broad_policy": 52.0,
        },
    )

    report = audit.classify_report([test02, test10])

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_union_consistent_replays_"
        "do_not_clear_guards"
    )
    assert report["status"] == "broad_union_consistent_replay_insufficient"
    assert not report["test02_branch_shift_supported"]
    assert report["test02_consistent_full_wet_residual_supported"]
    assert report["test10_broad_union_residual_guard_supported"]
    assert test02["finding"] == (
        "broad_union_reduces_full_wet_reference_but_guard_remains"
    )


def test_broad_union_branch_shift_missing_inputs(tmp_path: Path) -> None:
    report = audit.build_report(artifact_root=tmp_path)

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_union_branch_shift_incomplete"
    )
    assert report["status"] == (
        "regenerate_missing_broad_union_branch_shift_inputs"
    )
    assert report["missing_paths"]
