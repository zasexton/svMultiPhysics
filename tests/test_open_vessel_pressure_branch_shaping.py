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
        / "audit_pressure_branch_shaping.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_branch_shaping",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _guard_report(
    *,
    active_delta,
    active_point,
    active_support,
    full_delta=None,
    cut_delta=None,
    tiny_delta=None,
    triggered=True,
    threshold=10.0,
):
    def event(delta, point, support):
        if delta is None:
            return None
        return {
            "abs_pressure_delta_pa": delta,
            "pressure_delta_pa": delta,
            "point_index": point,
            "support_class": support,
        }

    return {
        "status": (
            "diagnostic_pressure_update_guard_triggered"
            if triggered
            else "diagnostic_pressure_update_guard_clear"
        ),
        "absolute_threshold_pa": threshold,
        "triggered_transition_count": 1 if triggered else 0,
        "worst_by_category": {
            "active_or_wet_supported": event(
                active_delta,
                active_point,
                active_support,
            ),
            "full_wet_supported": event(
                full_delta,
                active_point,
                "full_wet_supported",
            ),
            "cut_supported": event(
                cut_delta,
                active_point + 1,
                "cut_supported",
            ),
            "tiny_cut_supported": event(
                tiny_delta,
                active_point + 2,
                "tiny_cut_supported",
            ),
        },
    }


def test_branch_shaping_summary_classifies_controls():
    audit = _load_audit_module()
    baseline = _guard_report(
        active_delta=100.0,
        active_point=3,
        active_support="full_wet_supported",
        full_delta=100.0,
        cut_delta=80.0,
    )
    pressure_disabled = _guard_report(
        active_delta=50.0,
        active_point=7,
        active_support="cut_supported",
        full_delta=40.0,
        cut_delta=50.0,
        triggered=True,
    )
    incremental = _guard_report(
        active_delta=150.0,
        active_point=3,
        active_support="full_wet_supported",
        full_delta=150.0,
        cut_delta=60.0,
        triggered=True,
    )
    cleared = _guard_report(
        active_delta=5.0,
        active_point=3,
        active_support="full_wet_supported",
        full_delta=5.0,
        triggered=False,
    )

    report = audit.summarize_branch_controls(
        baseline_label="baseline",
        baseline_path=Path("baseline.json"),
        baseline_report=baseline,
        controls=[
            ("pressure_disabled", Path("disabled.json"), pressure_disabled),
            ("incremental", Path("incremental.json"), incremental),
            ("cleared", Path("cleared.json"), cleared),
        ],
    )

    assert report["finding"] == "pressure_policy_control_clears_guard"
    assert report["classification_counts"] == {
        "guard_cleared": 1,
        "improves_but_still_triggers": 1,
        "worsens_and_still_triggers": 1,
    }
    disabled = report["controls"][0]
    assert disabled["classification"] == "improves_but_still_triggers"
    assert disabled["support_class_shift"] is True
    assert disabled["point_shift"] is True
    assert disabled["active_or_wet_supported"]["ratio_to_baseline"] == 0.5
    assert disabled["branch_shaping_evidence"] is True
    assert disabled["safe_fix_candidate"] is False
    assert report["controls"][1]["classification"] == "worsens_and_still_triggers"
    assert report["controls"][2]["safe_fix_candidate"] is True
