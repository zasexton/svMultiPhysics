import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/cases/fluid/open_vessel_free_surface/"
    "audit_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays", SCRIPT
)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _pressure_audit(update: float, *, threshold: float, triggered: bool) -> dict:
    status = (
        "diagnostic_pressure_update_guard_triggered"
        if triggered
        else "diagnostic_pressure_update_guard_no_threshold_trigger"
    )
    return {
        "status": status,
        "finding": "synthetic",
        "absolute_threshold_pa": threshold,
        "triggered_transition_count": 1 if triggered else 0,
        "worst_by_category": {
            "active_or_wet_supported": {
                "abs_pressure_delta_pa": update,
                "point_index": 12,
                "support_class": "full_wet_supported",
                "active_fluid": 1.0,
                "incident_wet_fraction_min_positive": 1.0,
            }
        },
    }


def _scope(path: Path) -> Path:
    scope = path / "broad_minus_scope.json"
    _write_json(
        scope,
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_ready_for_replay"
            ),
            "cases": [
                {
                    "label": "test02",
                    "broad_only_parent_cell_count": 2,
                    "broad_parent_cell_count": 4,
                    "same_rule_parent_cell_count": 2,
                    "broad_only_to_broad_parent_ratio": 0.5,
                    "ready_for_broad_minus_parent_cell_replay": True,
                },
                {
                    "label": "test10",
                    "broad_only_parent_cell_count": 3,
                    "broad_parent_cell_count": 5,
                    "same_rule_parent_cell_count": 2,
                    "broad_only_to_broad_parent_ratio": 0.6,
                    "ready_for_broad_minus_parent_cell_replay": True,
                },
            ],
        },
    )
    return scope


def _policy_log(parent_count: int) -> str:
    return "\n".join(
        (
            "StandardAssembler: diagnostic=cut_volume_direct_pspg_topology_policy "
            "status=applied policy=local_schur_edge_balance record=summary "
            f"rule_index={parent} parent_cell={parent} full_cell=1 "
            "parent_filter_enabled=1 "
            f"parent_filter_parent_cell_count={parent_count} "
            "parent_filter_selected=1 row_filter_enabled=0 "
            "row_filter_global_dof_count=0 matrix_mutated=1 solve_affecting=1"
        )
        for parent in range(1, parent_count + 1)
    ) + "\n"


def _populate_case(
    root: Path,
    spec: dict,
    *,
    parent_count: int,
    replay_update: float,
    baseline_update: float,
    row_update: float,
    parent_update: float,
    broad_update: float,
    threshold: float,
    triggered: bool = True,
) -> None:
    case_dir = root / spec["case_dir"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / spec["log_name"]).write_text(
        _policy_log(parent_count),
        encoding="utf-8",
    )
    _write_json(
        root / spec["audit_name"],
        _pressure_audit(replay_update, threshold=threshold, triggered=triggered),
    )
    _write_json(
        root / spec["baseline_audit_name"],
        _pressure_audit(baseline_update, threshold=threshold, triggered=True),
    )
    _write_json(
        root / spec["row_filter_audit_name"],
        _pressure_audit(row_update, threshold=threshold, triggered=True),
    )
    _write_json(
        root / spec["parent_cell_audit_name"],
        _pressure_audit(parent_update, threshold=threshold, triggered=True),
    )
    _write_json(
        root / spec["broad_policy_audit_name"],
        _pressure_audit(broad_update, threshold=threshold, triggered=True),
    )


def test_broad_minus_parent_replays_do_not_clear_guards(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    _populate_case(
        tmp_path,
        audit.REPLAYS["test02"],
        parent_count=2,
        replay_update=95.0,
        baseline_update=100.0,
        row_update=90.0,
        parent_update=80.0,
        broad_update=60.0,
        threshold=50.0,
    )
    _populate_case(
        tmp_path,
        audit.REPLAYS["test10"],
        parent_count=3,
        replay_update=45.0,
        baseline_update=50.0,
        row_update=43.0,
        parent_update=40.0,
        broad_update=30.0,
        threshold=10.0,
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        broad_minus_scope_json=scope,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_do_not_clear_guards"
    )
    assert report["status"] == "broad_minus_parent_replay_insufficient"
    assert report["parent_filters_match_scope_counts"]
    assert report["row_filters_disabled"]
    assert report["all_replays_trigger_guard"]
    assert report["broad_policy_better_than_isolated_parts"]
    assert report["complement_worse_than_same_rule_parent_cell"]


def test_broad_minus_parent_replays_clear_guards(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    _populate_case(
        tmp_path,
        audit.REPLAYS["test02"],
        parent_count=2,
        replay_update=10.0,
        baseline_update=100.0,
        row_update=90.0,
        parent_update=80.0,
        broad_update=60.0,
        threshold=50.0,
        triggered=False,
    )
    _populate_case(
        tmp_path,
        audit.REPLAYS["test10"],
        parent_count=3,
        replay_update=5.0,
        baseline_update=50.0,
        row_update=43.0,
        parent_update=40.0,
        broad_update=30.0,
        threshold=10.0,
        triggered=False,
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        broad_minus_scope_json=scope,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_clear_guards"
    )
    assert report["status"] == "broad_minus_parent_replay_clears_short_windows"
    assert report["cleared_cases"] == ["test02", "test10"]


def test_broad_minus_parent_replays_missing_artifacts(tmp_path: Path) -> None:
    scope = _scope(tmp_path)

    report = audit.build_report(
        artifact_root=tmp_path,
        broad_minus_scope_json=scope,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_incomplete"
    )
    assert report["status"] == "regenerate_missing_broad_minus_parent_replay_artifacts"
