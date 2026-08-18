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
        / "audit_pressure_update_residual_context.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_update_residual_context", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_transition(path, *, update_pa, residual, support_class="full_wet_supported"):
    path.write_text(
        json.dumps(
            {
                "status": "diagnostic_cut_context_pressure_transition_guard_found",
                "finding": "synthetic",
                "guard_before_accepted_step_refresh": True,
                "post_acceptance_refresh_immediate_driver_ruled_out": True,
                "runtime_offline_pressure_match": {"matches": True},
                "pressure_update_guard": {
                    "triggered": 1,
                    "global_abs_pressure_delta_pa": update_pa,
                    "support_class": support_class,
                    "threshold_pa": 100.0,
                },
                "nonlinear_done": {
                    "converged": True,
                    "iters": 1,
                    "residual": residual,
                    "residual_field": residual,
                    "linear_converged": True,
                    "linear_iters": 1,
                    "linear_rel": 1.0e-13,
                },
                "pressure_update_to_nonlinear_residual": {
                    "global_abs_pressure_delta_pa": update_pa,
                    "nonlinear_converged": True,
                    "nonlinear_iterations": 1,
                    "linear_converged": True,
                    "linear_iterations": 1,
                    "linear_relative_residual": 1.0e-13,
                    "nonlinear_residual_norm": residual,
                    "nonlinear_field_residual_norm": residual,
                    "update_to_nonlinear_residual_norm_ratio": update_pa / residual,
                    "update_to_nonlinear_field_residual_norm_ratio": (
                        update_pa / residual
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_residual_context_flags_converged_large_pressure_update_gap(tmp_path):
    audit = _load_audit_module()
    test02 = tmp_path / "test02.json"
    test10 = tmp_path / "test10.json"
    _write_transition(test02, update_pa=2.0e6, residual=2.0)
    _write_transition(test10, update_pa=1000.0, residual=1.0e-3)

    report = audit.build_report(
        test02_path=test02,
        test10_path=test10,
        large_ratio_threshold=1.0e3,
    )

    assert report["finding"] == (
        "accepted_pressure_updates_converged_with_large_residual_gap"
    )
    assert report["status"] == "residual_convergence_acceptance_gap_supported"
    assert report["all_cases_accepted_converged_large_update_residual_gap"]
    cases = {case["label"]: case for case in report["cases"]}
    assert cases["test02"]["update_to_nonlinear_field_residual_norm_ratio"] == 1.0e6
    assert cases["test10"]["accepted_converged_large_update_residual_gap"]


def test_residual_context_reports_below_threshold_case(tmp_path):
    audit = _load_audit_module()
    test02 = tmp_path / "test02.json"
    test10 = tmp_path / "test10.json"
    _write_transition(test02, update_pa=2.0e6, residual=2.0)
    _write_transition(test10, update_pa=10.0, residual=1.0)

    report = audit.build_report(
        test02_path=test02,
        test10_path=test10,
        large_ratio_threshold=1.0e3,
    )

    assert report["finding"] == (
        "pressure_update_residual_context_does_not_show_large_gap"
    )
    cases = {case["label"]: case for case in report["cases"]}
    assert cases["test10"]["finding"] == (
        "pressure_update_residual_gap_below_threshold_or_missing"
    )
