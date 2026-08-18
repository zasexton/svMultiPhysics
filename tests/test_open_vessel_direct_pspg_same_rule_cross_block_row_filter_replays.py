import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/cases/fluid/open_vessel_free_surface/"
    "audit_direct_pspg_same_rule_cross_block_row_filter_replays.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_direct_pspg_same_rule_cross_block_row_filter_replays", SCRIPT
)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _pressure_audit(
    update: float,
    *,
    threshold: float,
    triggered: bool,
    support_class: str = "full_wet_supported",
) -> dict:
    status = (
        "diagnostic_pressure_update_guard_triggered"
        if triggered
        else "diagnostic_pressure_update_guard_no_threshold_trigger"
    )
    event = {
        "abs_pressure_delta_pa": update,
        "point_index": 12,
        "support_class": support_class,
        "active_fluid": 1.0,
        "incident_wet_fraction_min_positive": 1.0,
    }
    return {
        "status": status,
        "finding": "synthetic",
        "absolute_threshold_pa": threshold,
        "triggered_transition_count": 1 if triggered else 0,
        "worst_by_category": {"active_or_wet_supported": event},
    }


def _policy_log(row_count: int) -> str:
    return (
        "StandardAssembler: diagnostic=cut_volume_direct_pspg_topology_policy "
        "policy=local_schur_edge_balance row_filter_enabled=1 "
        f"row_filter_global_dof_count={row_count} "
        "row_filter_selected_local_row_count=1 matrix_mutated=1\n"
    )


def _candidate(path: Path) -> Path:
    candidate = path / "candidate.json"
    _write_json(
        candidate,
        {
            "finding": (
                "solve_time_direct_pspg_same_rule_cross_block_signature_"
                "magnitude_candidate_found"
            ),
            "cases": [
                {
                    "label": "test02",
                    "best_covering_composite_selected_global_dofs": [1, 2],
                },
                {
                    "label": "test10",
                    "best_covering_composite_selected_global_dofs": [3, 4, 5],
                },
            ],
        },
    )
    return candidate


def _populate_case(
    root: Path,
    label: str,
    spec: dict,
    *,
    row_count: int,
    replay_update: float,
    baseline_update: float,
    broad_update: float,
    threshold: float,
    triggered: bool,
) -> None:
    case_dir = root / spec["case_dir"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / spec["log_name"]).write_text(_policy_log(row_count), encoding="utf-8")
    _write_json(
        root / spec["audit_name"],
        _pressure_audit(replay_update, threshold=threshold, triggered=triggered),
    )
    _write_json(
        root / spec["baseline_audit_name"],
        _pressure_audit(baseline_update, threshold=threshold, triggered=True),
    )
    _write_json(
        root / spec["broad_policy_audit_name"],
        _pressure_audit(broad_update, threshold=threshold, triggered=True),
    )
    prior = spec.get("prior_signature_row_filter_audit_name")
    if prior:
        _write_json(
            root / prior,
            _pressure_audit(
                broad_update + 10.0, threshold=threshold, triggered=True
            ),
        )


def test_same_rule_replays_do_not_clear_guards(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    _populate_case(
        tmp_path,
        "test02",
        audit.REPLAYS["test02"],
        row_count=2,
        replay_update=90.0,
        baseline_update=100.0,
        broad_update=80.0,
        threshold=50.0,
        triggered=True,
    )
    _populate_case(
        tmp_path,
        "test10",
        audit.REPLAYS["test10"],
        row_count=3,
        replay_update=40.0,
        baseline_update=50.0,
        broad_update=30.0,
        threshold=10.0,
        triggered=True,
    )

    report = audit.build_report(artifact_root=tmp_path, candidate_json=candidate)

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_row_filter_replays_do_not_clear_guards"
    )
    assert report["status"] == "same_rule_cross_block_replay_insufficient"
    assert report["row_filters_match_candidate_counts"] is True
    assert report["all_replays_improve_no_policy_baseline"] is True
    assert report["triggered_cases"] == ["test02", "test10"]


def test_same_rule_replays_clear_guards(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    _populate_case(
        tmp_path,
        "test02",
        audit.REPLAYS["test02"],
        row_count=2,
        replay_update=10.0,
        baseline_update=100.0,
        broad_update=80.0,
        threshold=50.0,
        triggered=False,
    )
    _populate_case(
        tmp_path,
        "test10",
        audit.REPLAYS["test10"],
        row_count=3,
        replay_update=5.0,
        baseline_update=50.0,
        broad_update=30.0,
        threshold=10.0,
        triggered=False,
    )

    report = audit.build_report(artifact_root=tmp_path, candidate_json=candidate)

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_row_filter_replays_clear_guards"
    )
    assert report["status"] == (
        "same_rule_cross_block_replay_candidate_clears_short_windows"
    )
    assert report["cleared_cases"] == ["test02", "test10"]


def test_same_rule_replays_missing_artifacts(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)

    report = audit.build_report(artifact_root=tmp_path, candidate_json=candidate)

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_row_filter_replays_incomplete"
    )
    assert report["status"] == "regenerate_missing_replay_artifacts"
