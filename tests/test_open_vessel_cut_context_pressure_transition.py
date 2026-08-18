import importlib.util
from pathlib import Path


def _load_transition_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_cut_context_pressure_transition.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_cut_context_pressure_transition", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transition_audit_orders_guard_before_maintenance_refresh(tmp_path):
    audit = _load_transition_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Active-domain cut context diagnostic=cut_context_rebuild "
                "provenance=initial solution_source=fe_vector "
                "cut_context_revision=1 cut_context_topology_key=11 "
                "source_value_revision=1 active_side_volume=10.0 "
                "active_cut_cells=2 active_volume_regions=4 "
                "active_min_volume_fraction=0.25 cut_adjacent_facets=8",
                "TimeLoop: step_start step=0 time=0.0 dt=0.1",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual iteration=0 "
                "phase='jacobian_and_residual' sync_point=jacobian_and_residual "
                "field='Pressure' norm=10.0 global_max_abs=4.0",
                "Active-domain cut context diagnostic=cut_context_rebuild "
                "provenance=line_search_trial solution_source=state_vector_fe_ordered "
                "cut_context_revision=2 cut_context_topology_key=12 "
                "source_value_revision=2 active_side_volume=10.001 "
                "active_cut_cells=3 active_volume_regions=5 "
                "active_min_volume_fraction=0.20 cut_adjacent_facets=9",
                "TimeLoop: nonlinear_done step=0 time=0.0 converged=1 iters=1 "
                "||r||=1e-3 ||r_field||=1e-3 ||r_aux||=0 "
                "(linear: converged=1 iters=1 rel=2e-13)",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual iteration=0 "
                "phase='line_search' sync_point=line_search_trial "
                "field='Pressure' norm=2.0 global_max_abs=1.0",
                "TimeLoop: step_accepted step=1 time=0.1 dt=0.1",
                "Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard step=1 time=0.1 dt=0.1 "
                "global_abs_pressure_delta_pa=42.0 local_abs_pressure_delta_pa=42.0 "
                "support_class=full_wet_supported triggered=1",
                "Active-domain cut context diagnostic=cut_context_rebuild "
                "provenance=accepted_step solution_source=fe_vector "
                "cut_context_revision=3 cut_context_topology_key=13 "
                "source_value_revision=3 active_side_volume=10.002 "
                "active_cut_cells=3 active_volume_regions=5 "
                "active_min_volume_fraction=0.19 cut_adjacent_facets=9",
            ]
        ),
        encoding="utf-8",
    )

    report = audit.audit_transition(
        solver_log=log,
        case_label="synthetic",
        pressure_update_audit=None,
        pressure_match_abs_tol_pa=1.0e-6,
    )

    assert report["guard_before_accepted_step_refresh"] is True
    assert report["post_acceptance_refresh_immediate_driver_ruled_out"] is True
    assert report["pressure_update_guard"]["global_abs_pressure_delta_pa"] == 42.0
    assert report["initial_field_residual"]["norm"] == 10.0
    assert report["solve_field_residual"]["sync_point"] == "line_search_trial"
    assert report["pressure_update_to_solve_field_residual"][
        "update_to_field_residual_norm_ratio"
    ] == 21.0
    assert report["pressure_update_to_solve_field_residual"][
        "update_to_field_residual_max_abs_ratio"
    ] == 42.0
    assert report["pressure_update_to_nonlinear_residual"][
        "update_to_nonlinear_field_residual_norm_ratio"
    ] == 42000.0
    assert report["pressure_update_to_nonlinear_residual"][
        "nonlinear_converged"
    ] is True
    assert (
        report["initial_to_solve_context_delta"]["changed_counts"][
            "active_cut_cells"
        ]["delta"]
        == 1
    )
    assert (
        report["solve_to_accepted_step_maintenance_context_delta"][
            "changed_identities"
        ]["cut_context_revision"]["after"]
        == 3
    )


def test_pressure_match_uses_offline_active_wet_event(tmp_path):
    audit = _load_transition_module()
    pressure_audit = tmp_path / "pressure.json"
    pressure_audit.write_text(
        """
        {
          "transitions": [
            {
              "max_by_category": {
                "active_or_wet_supported": {
                  "from_step": 90,
                  "to_step": 1,
                  "abs_pressure_delta_pa": 1075.5,
                  "pressure_delta_pa": 1075.5,
                  "point_index": 3,
                  "point_m": [0.0, 0.0365, 0.031],
                  "support_class": "full_wet_supported",
                  "incident_wet_fraction_max": 1.0,
                  "incident_wet_fraction_min_positive": 1.0
                }
              },
              "delta_statistics_by_category": {
                "active_or_wet_supported": {
                  "max_abs_after_median_removal_pa": 1093.0
                }
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    guard = {"global_abs_pressure_delta_pa": 1075.5000004}
    offline = audit.offline_pressure_update(pressure_audit)
    match = audit.pressure_match_report(guard, offline, abs_tol_pa=1.0e-3)

    assert offline["support_class"] == "full_wet_supported"
    assert offline["median_removed_active_or_wet_max_pa"] == 1093.0
    assert match["matches"] is True
    assert match["abs_difference_pa"] < 1.0e-3


def test_compare_contexts_reports_float_count_and_identity_changes():
    audit = _load_transition_module()
    before = {
        "line_number": 10,
        "provenance": "line_search_trial",
        "active_side_volume": 100.0,
        "active_cut_cells": 236,
        "cut_context_topology_key": 123,
    }
    after = {
        "line_number": 12,
        "provenance": "accepted_step",
        "active_side_volume": 100.01,
        "active_cut_cells": 236,
        "cut_context_topology_key": 456,
    }

    comparison = audit.compare_contexts(before, after)

    assert comparison["float_deltas"]["active_side_volume"]["abs_delta"] > 0.0
    assert comparison["changed_counts"] == {}
    assert comparison["changed_identities"]["cut_context_topology_key"]["after"] == 456
