import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/cases/fluid/open_vessel_free_surface/"
    "audit_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope", SCRIPT
)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _scope(path: Path) -> Path:
    scope = path / "parent_scope.json"
    _write_json(
        scope,
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_parent_cell_scope_ready_for_replay"
            ),
            "cases": [
                {
                    "label": "test02",
                    "parent_cells": [2, 3],
                },
                {
                    "label": "test10",
                    "parent_cells": [10],
                },
            ],
        },
    )
    return scope


def _policy_line(parent: int) -> str:
    return (
        "StandardAssembler: diagnostic=cut_volume_direct_pspg_topology_policy "
        "status=applied policy=local_schur_edge_balance record=summary "
        f"rule_index={parent} parent_cell={parent} full_cell=1 "
        "matrix_mutated=1 solve_affecting=1 diagnostic_only=0"
    )


def _write_log(path: Path, parents: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(_policy_line(parent) for parent in parents) + "\n",
        encoding="utf-8",
    )
    return path


def test_broad_minus_parent_scope_ready_for_replay(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    test02_log = _write_log(tmp_path / "test02.log", [1, 2, 3, 4])
    test10_log = _write_log(tmp_path / "test10.log", [9, 10, 11])

    report = audit.build_report(
        parent_scope_json=scope,
        test02_broad_log=test02_log,
        test10_broad_log=test10_log,
        max_broad_only_parent_count=10,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_ready_for_replay"
    )
    assert report["status"] == "run_broad_minus_same_rule_parent_cell_replay"
    assert report["all_cases_ready_for_broad_minus_parent_cell_replay"]
    assert report["ready_cases"] == ["test02", "test10"]
    assert report["cases"][0]["broad_only_parent_cells"] == [1, 4]
    assert report["cases"][0]["broad_only_to_broad_parent_ratio"] == 0.5
    assert report["cases"][1]["replay_parent_cell_global_input"] == "9,11"


def test_broad_minus_parent_scope_not_ready_when_too_large(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    test02_log = _write_log(tmp_path / "test02.log", [1, 2, 3, 4])
    test10_log = _write_log(tmp_path / "test10.log", [9, 10, 11])

    report = audit.build_report(
        parent_scope_json=scope,
        test02_broad_log=test02_log,
        test10_broad_log=test10_log,
        max_broad_only_parent_count=1,
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_not_ready"
    )
    assert report["status"] == "broad_minus_parent_scope_too_large_or_incomplete"
    assert report["ready_cases"] == []


def test_broad_minus_parent_scope_reports_missing_inputs(tmp_path: Path) -> None:
    report = audit.build_report(
        parent_scope_json=tmp_path / "missing_scope.json",
        test02_broad_log=tmp_path / "missing_test02.log",
        test10_broad_log=tmp_path / "missing_test10.log",
    )

    assert report["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_incomplete"
    )
    assert report["status"] == "regenerate_missing_broad_minus_scope_inputs"
    assert len(report["missing_evidence"]) == 3
