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
        / "audit_pressure_edge_completion_predictor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_edge_completion_predictor",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _support_report(*, action_terms: str, update: float, rhs: float):
    return {
        "solver_log": "run.log",
        "pressure_update_support_summary": {
            "max_update_detail": {
                "global_dof": 10,
                "local_pressure_row": 0,
                "update": update,
                "rhs": rhs,
                "diag": 1.0e-6,
                "row_self": 2.0e-6,
                "row_self_offdiag": 1.0e-6,
                "row_self_action": rhs,
                "row_coupling": 0.0,
                "pressure_action_terms": action_terms,
            }
        },
    }


def _detail(
    *,
    global_dof: int,
    local_pressure_row: int,
    action_terms: str,
    update: float,
    rhs: float,
):
    return {
        "global_dof": global_dof,
        "local_pressure_row": local_pressure_row,
        "update": update,
        "rhs": rhs,
        "diag": 1.0e-6,
        "row_self": 2.0e-6,
        "row_self_offdiag": 1.0e-6,
        "row_self_action": rhs,
        "row_coupling": 0.0,
        "pressure_action_terms": action_terms,
    }


def test_parse_pressure_action_terms():
    audit = _load_audit_module()
    terms = audit.parse_action_terms(
        "0/10/m=1.0e-6/u=-100/a=-1.0e-4~"
        "1/11/m=-2.5e-7/u=-20/a=5.0e-6"
    )
    assert terms == [
        {
            "local_dof": 0,
            "global_dof": 10,
            "matrix_value": 1.0e-6,
            "update": -100.0,
            "action": -1.0e-4,
        },
        {
            "local_dof": 1,
            "global_dof": 11,
            "matrix_value": -2.5e-7,
            "update": -20.0,
            "action": 5.0e-6,
        },
    ]


def test_predicts_plausible_below_guard_pressure_edge():
    audit = _load_audit_module()
    report = audit.build_report(
        _support_report(
            action_terms=(
                "0/10/m=1.0e-6/u=-100/a=-1.0e-4~"
                "1/11/m=-1.0e-6/u=-20/a=2.0e-5"
            ),
            update=-100.0,
            rhs=-1.0e-4,
        ),
        pressure_update_report={"absolute_threshold_pa": 50.0},
    )
    assert (
        report["finding"]
        == "local_pressure_edge_completion_plausible_for_sampled_max_row"
    )
    assert report["plausible_below_guard_local_edge_count"] == 1
    best = report["best_candidate_edges"][0]
    assert best["neighbor_below_guard"]
    assert best["edge_would_pull_toward_lower_abs_neighbor"]
    assert best["edge_strength_class"] == "row_self_scale_or_less"


def test_reports_when_logged_neighbors_do_not_reach_guard():
    audit = _load_audit_module()
    report = audit.build_report(
        _support_report(
            action_terms=(
                "0/10/m=1.0e-6/u=100/a=1.0e-4~"
                "1/11/m=-1.0e-6/u=75/a=-7.5e-5"
            ),
            update=100.0,
            rhs=1.0e-4,
        ),
        pressure_update_report={"absolute_threshold_pa": 50.0},
    )
    assert report["finding"] == "no_logged_pressure_neighbor_below_guard_for_sampled_max_row"
    assert report["below_guard_neighbor_candidate_count"] == 0
    assert not report["best_candidate_edges"][0]["neighbor_below_guard"]


def test_aggregates_local_edge_predictions_for_logged_top_updates():
    audit = _load_audit_module()
    support_report = {
        "solver_log": "run.log",
        "pressure_update_support_summary": {
            "max_update_detail": _detail(
                global_dof=10,
                local_pressure_row=0,
                action_terms=(
                    "0/10/m=1.0e-6/u=-100/a=-1.0e-4~"
                    "1/11/m=-1.0e-6/u=-20/a=2.0e-5"
                ),
                update=-100.0,
                rhs=-1.0e-4,
            ),
            "top_update_details": [
                _detail(
                    global_dof=10,
                    local_pressure_row=0,
                    action_terms=(
                        "0/10/m=1.0e-6/u=-100/a=-1.0e-4~"
                        "1/11/m=-1.0e-6/u=-20/a=2.0e-5"
                    ),
                    update=-100.0,
                    rhs=-1.0e-4,
                ),
                _detail(
                    global_dof=12,
                    local_pressure_row=2,
                    action_terms=(
                        "2/12/m=1.0e-6/u=120/a=1.2e-4~"
                        "3/13/m=-1.0e-6/u=90/a=-9.0e-5"
                    ),
                    update=120.0,
                    rhs=1.2e-4,
                ),
            ],
        },
    }
    report = audit.build_report(
        support_report,
        pressure_update_report={"absolute_threshold_pa": 50.0},
        all_top_updates=True,
    )
    assert report["finding"] == "local_pressure_edge_completion_partial_for_logged_top_rows"
    assert report["evaluated_row_count"] == 2
    assert report["guard_violating_row_count"] == 2
    assert report["plausible_guard_violating_row_count"] == 1
    assert report["no_below_guard_neighbor_guard_violating_row_count"] == 1
    assert (
        report["guard_violating_rows_without_below_guard_neighbors"][0][
            "global_dof"
        ]
        == 12
    )


def test_aggregates_when_below_guard_neighbors_need_stronger_edges():
    audit = _load_audit_module()
    support_report = {
        "solver_log": "run.log",
        "pressure_update_support_summary": {
            "max_update_detail": _detail(
                global_dof=10,
                local_pressure_row=0,
                action_terms=(
                    "0/10/m=1.0e-6/u=100/a=1.0e-4~"
                    "1/11/m=-1.0e-6/u=40/a=-4.0e-5"
                ),
                update=100.0,
                rhs=1.5e-4,
            ),
            "top_update_details": [
                _detail(
                    global_dof=10,
                    local_pressure_row=0,
                    action_terms=(
                        "0/10/m=1.0e-6/u=100/a=1.0e-4~"
                        "1/11/m=-1.0e-6/u=40/a=-4.0e-5"
                    ),
                    update=100.0,
                    rhs=1.5e-4,
                ),
                _detail(
                    global_dof=12,
                    local_pressure_row=2,
                    action_terms=(
                        "2/12/m=1.0e-6/u=120/a=1.2e-4~"
                        "3/13/m=-1.0e-6/u=90/a=-9.0e-5"
                    ),
                    update=120.0,
                    rhs=1.2e-4,
                ),
            ],
        },
    }
    report = audit.build_report(
        support_report,
        pressure_update_report={"absolute_threshold_pa": 50.0},
        all_top_updates=True,
    )

    assert (
        report["finding"]
        == "below_guard_neighbors_exist_but_need_larger_edges_for_logged_top_rows"
    )
    assert report["guard_violating_row_count"] == 2
    assert report["plausible_guard_violating_row_count"] == 0
    assert report["below_guard_neighbor_guard_violating_row_count"] == 1
    assert report["no_below_guard_neighbor_guard_violating_row_count"] == 1
    assert (
        report["row_reports"][0]["best_candidate_edges"][0][
            "edge_strength_class"
        ]
        == "within_10x_row_self"
    )
