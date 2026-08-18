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
        / "audit_direct_pspg_explicit_balance_selector_replays.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_explicit_balance_selector_replays", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_support(
    root: Path,
    *,
    name: str,
    triggered: int | None,
    update_pa: float | None,
    nonlinear_converged: bool = True,
):
    case_dir = root / f"{name}_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    support_path = root / f"{name}.json"
    _write_json(
        support_path,
        {
            "solver_log": str(case_dir / "run.log"),
            "latest_pressure_graph_completion": {
                "values": {
                    "mode": "shared_row_schur_explicit_edge_balance",
                    "candidate_row_count": 8,
                    "balance_candidate_row_count": 3,
                    "edge_count": 21,
                    "shared_row_schur_edge_count": 13,
                    "existing_balance_edge_count": 8,
                    "explicit_balance_requested_global_dofs": "100|101",
                    "balance_candidate_global_dofs": "100|101",
                }
            },
            "latest_accepted_pressure_update": (
                {
                    "values": {
                        "global_abs_pressure_delta_pa": update_pa,
                        "local_worst_dof": 101,
                        "threshold_pa": 100.0,
                        "triggered": triggered,
                    }
                }
                if update_pa is not None
                else None
            ),
        },
    )
    converged = 1 if nonlinear_converged else 0
    success = 1 if nonlinear_converged else 0
    message = "" if nonlinear_converged else "TimeLoop: nonlinear solve did not converge"
    (case_dir / "run.log").write_text(
        "\n".join(
            [
                (
                    "[svMultiPhysics::Application] TimeLoop: nonlinear_done "
                    f"step=0 time=1 converged={converged} iters=12 "
                    "||r||=1.234000e+04 ||r_field||=1.234000e+04 ||r_aux||=0"
                ),
                (
                    "[svMultiPhysics::Application] TimeLoop: loop.run() returned "
                    f"success={success} steps_taken={success} final_time=1 "
                    f"message='{message}'"
                ),
                "Message: TimeLoop: nonlinear solve did not converge"
                if not nonlinear_converged
                else "",
            ]
        ),
        encoding="utf-8",
    )
    return support_path.name


def test_build_report_rules_out_explicit_rows_and_neighborhoods(tmp_path):
    audit = _load_audit_module()
    boundary_path = tmp_path / "boundary.json"
    _write_json(
        boundary_path,
        {
            "finding": "latest_bad_rows_can_be_candidates_without_balance_coverage",
            "boundary_topology_finding": "boundary_top_update_candidates_missing_balance",
            "boundary_topology_finding_counts": {
                "boundary_top_update_candidates_missing_balance": 1
            },
        },
    )
    triggered_t02 = _write_support(
        tmp_path, name="triggered_t02", triggered=1, update_pa=103000.0
    )
    triggered_t10 = _write_support(
        tmp_path, name="triggered_t10", triggered=1, update_pa=120.0
    )
    failed_t02 = _write_support(
        tmp_path,
        name="failed_t02",
        triggered=None,
        update_pa=None,
        nonlinear_converged=False,
    )
    variant_specs = (
        {
            "key": "explicit_direct_rows",
            "support": {"test02": triggered_t02, "test10": triggered_t10},
        },
        {
            "key": "explicit_shifted_rows",
            "support": {"test02": failed_t02, "test10": triggered_t10},
        },
        {"key": "explicit_cross_policy_patch", "support": {"test02": triggered_t02}},
        {"key": "explicit_operator_top_rows", "support": {"test02": failed_t02}},
        {
            "key": "explicit_neighborhood_depth1",
            "support": {"test02": triggered_t02, "test10": triggered_t10},
        },
        {
            "key": "explicit_neighborhood_depth2",
            "support": {"test02": triggered_t02, "test10": triggered_t10},
        },
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        boundary_provenance_path=boundary_path,
        variant_specs=variant_specs,
    )

    assert report["finding"] == (
        "direct_pspg_explicit_balance_selectors_rule_out_row_lists_and_"
        "pressure_neighborhoods"
    )
    assert report["status"] == "explicit_balance_selectors_ruled_out"
    assert report["ruleout_flags"] == {
        "boundary_balance_predicate_misses_latest_bad_rows": True,
        "explicit_row_lists_ruled_out": True,
        "current_pressure_neighborhoods_ruled_out": True,
    }
    assert report["ruled_out_by_variant"]["explicit_shifted_rows"]
    shifted = next(
        variant
        for variant in report["variants"]
        if variant["key"] == "explicit_shifted_rows"
    )
    assert shifted["cases"][0]["finding"] == "nonlinear_failed"


def test_build_report_requires_boundary_provenance(tmp_path):
    audit = _load_audit_module()
    report = audit.build_report(
        artifact_root=tmp_path,
        boundary_provenance_path=tmp_path / "missing.json",
        variant_specs=(),
    )

    assert report["finding"] == (
        "direct_pspg_explicit_balance_selector_replays_missing_evidence"
    )
    assert report["status"] == "missing_explicit_balance_replay_evidence"
