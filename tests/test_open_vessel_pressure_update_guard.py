import importlib.util
from pathlib import Path

import numpy as np


def _load_guard_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_update_guard.py"
    )
    spec = importlib.util.spec_from_file_location("audit_pressure_update_guard", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_support_class_distinguishes_full_cut_tiny_and_dry():
    guard = _load_guard_module()

    assert (
        guard.support_class(
            phi=-0.1,
            active_fluid=1.0,
            incident_wet_fraction_max=1.0,
            incident_wet_fraction_min_positive=1.0,
            active_threshold=0.5,
            tiny_wet_fraction=1.0e-4,
            full_wet_tolerance=1.0e-12,
        )
        == "full_wet_supported"
    )
    assert (
        guard.support_class(
            phi=-0.01,
            active_fluid=1.0,
            incident_wet_fraction_max=0.25,
            incident_wet_fraction_min_positive=0.01,
            active_threshold=0.5,
            tiny_wet_fraction=1.0e-4,
            full_wet_tolerance=1.0e-12,
        )
        == "cut_supported"
    )
    assert (
        guard.support_class(
            phi=0.2,
            active_fluid=0.0,
            incident_wet_fraction_max=5.0e-5,
            incident_wet_fraction_min_positive=5.0e-5,
            active_threshold=0.5,
            tiny_wet_fraction=1.0e-4,
            full_wet_tolerance=1.0e-12,
        )
        == "tiny_cut_supported"
    )
    assert (
        guard.support_class(
            phi=0.2,
            active_fluid=0.0,
            incident_wet_fraction_max=None,
            incident_wet_fraction_min_positive=None,
            active_threshold=0.5,
            tiny_wet_fraction=1.0e-4,
            full_wet_tolerance=1.0e-12,
        )
        == "dry_or_inactive"
    )


def test_parse_solver_log_attaches_nonlinear_context_to_accepted_step(tmp_path):
    guard = _load_guard_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "TimeLoop: step_start step=90 time=0.9 dt=0.000625",
                "Application: diagnostic=cut_context_rebuild provenance=line_search_trial "
                "active_wet_cells=720 active_cut_cells=236 "
                "active_min_volume_fraction=0.0447 cut_adjacent_max_scale=22.3",
                "TimeLoop: nonlinear_done step=90 time=0.9 converged=1 iters=1 "
                "||r||=9.6e-4 ||r_field||=9.6e-4 ||r_aux||=0 "
                "(linear: converged=1 iters=1 rel=2e-13)",
                "TimeLoop: step_accepted step=91 time=0.900625 dt=0.000625",
            ]
        ),
        encoding="utf-8",
    )

    context = guard.parse_solver_log(log)

    assert 91 in context
    assert context[91]["attempt_step"] == 90
    assert context[91]["nonlinear"]["iters"] == 1
    assert context[91]["nonlinear"]["residual"] == 9.6e-4
    assert context[91]["cut_context_rebuilds"][0]["active_wet_cells"] == 720
    assert context[91]["cut_context_rebuilds"][0]["active_min_volume_fraction"] == 0.0447


def test_parse_solver_log_attaches_runtime_pressure_update_guard(tmp_path):
    guard = _load_guard_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "TimeLoop: step_start step=90 time=0.9 dt=0.000625",
                "TimeLoop: nonlinear_done step=90 time=0.9 converged=1 iters=1 "
                "||r||=1e-3 ||r_field||=1e-3 ||r_aux||=0 "
                "(linear: converged=1 iters=1 rel=2e-13)",
                "TimeLoop: step_accepted step=91 time=0.900625 dt=0.000625",
                "Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard step=91 "
                "global_abs_pressure_delta_pa=1075.5 "
                "support_class=full_wet_supported triggered=1",
            ]
        ),
        encoding="utf-8",
    )

    context = guard.parse_solver_log(log)

    update_guard = context[91]["pressure_update_guard"]
    assert update_guard["global_abs_pressure_delta_pa"] == 1075.5
    assert update_guard["support_class"] == "full_wet_supported"
    assert update_guard["triggered"] == 1


def test_pressure_delta_statistics_reports_centered_residual():
    guard = _load_guard_module()

    stats = guard.pressure_delta_statistics(
        np.asarray([True, True, True, False]),
        np.asarray([10.0, 12.0, 110.0, 999.0]),
    )

    assert stats["count"] == 3
    assert stats["median_delta_pa"] == 12.0
    assert stats["max_abs_delta_pa"] == 110.0
    assert stats["max_abs_after_median_removal_pa"] == 98.0
