import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/cases/fluid/open_vessel_free_surface/"
    "audit_direct_pspg_same_rule_cross_block_parent_cell_scope.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_direct_pspg_same_rule_cross_block_parent_cell_scope", SCRIPT
)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


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
                    "best_covering_composite_selected_global_dofs": [5],
                },
            ],
        },
    )
    return candidate


def _line(row: int, parent: int, rule: int, *, full_cell: int = 1) -> str:
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_direct_pspg_support_coupling_provenance "
        f"row_dof={row} parent_cell={parent} rule_index={rule} "
        "block=pressure_pressure row_local_index=0 "
        f"full_cell={full_cell} diagnostic_only=1 pressure_update_sign_used=0"
    )


def _write_log(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_parent_cell_scope_ready_for_replay(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    test02_log = _write_log(
        tmp_path / "test02.log",
        [
            _line(1, 10, 100),
            _line(2, 20, 200),
            _line(3, 10, 100),
        ],
    )
    test10_log = _write_log(
        tmp_path / "test10.log",
        [
            _line(5, 30, 300, full_cell=0),
            _line(6, 30, 300, full_cell=0),
        ],
    )

    report = audit.build_report(
        candidate_json=candidate,
        test02_log=test02_log,
        test10_log=test10_log,
        max_expanded_row_ratio=5.0,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_parent_cell_scope_ready_for_replay"
    )
    assert report["status"] == "run_same_rule_cross_block_parent_cell_replay"
    assert report["all_cases_ready_for_parent_cell_replay"]
    assert report["ready_cases"] == ["test02", "test10"]
    assert report["cases"][0]["parent_cells"] == [10, 20]
    assert report["cases"][0]["parent_expanded_rows"] == [1, 2, 3]
    assert report["cases"][0]["parent_expanded_to_candidate_ratio"] == 1.5
    assert report["cases"][1]["cut_parent_cells"] == [30]
    assert report["cases"][1]["replay_parent_cell_global_input"] == "30"


def test_parent_cell_scope_overbroad_when_expansion_exceeds_limit(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    test02_log = _write_log(
        tmp_path / "test02.log",
        [_line(1, 10, 100), _line(2, 20, 200), _line(3, 10, 100)],
    )
    test10_log = _write_log(
        tmp_path / "test10.log",
        [_line(5, 30, 300), _line(6, 30, 300)],
    )

    report = audit.build_report(
        candidate_json=candidate,
        test02_log=test02_log,
        test10_log=test10_log,
        max_expanded_row_ratio=1.0,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_parent_cell_scope_not_replay_ready"
    )
    assert report["status"] == "parent_cell_scope_overbroad_or_missing_candidates"
    assert report["ready_cases"] == []


def test_parent_cell_scope_reports_missing_inputs(tmp_path: Path) -> None:
    report = audit.build_report(
        candidate_json=tmp_path / "missing_candidate.json",
        test02_log=tmp_path / "missing_test02.log",
        test10_log=tmp_path / "missing_test10.log",
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_parent_cell_scope_incomplete"
    )
    assert report["status"] == "regenerate_missing_parent_scope_inputs"
    assert len(report["missing_evidence"]) == 3
