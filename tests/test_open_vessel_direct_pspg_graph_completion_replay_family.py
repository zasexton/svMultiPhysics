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
        / "audit_direct_pspg_graph_completion_replay_family.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_graph_completion_replay_family", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_target_map(path: Path):
    _write_json(
        path,
        {
            "cases": [
                {"label": "test02", "direct_pspg_target_global_dofs": [100, 101]},
                {
                    "label": "test10",
                    "direct_pspg_target_global_dofs": [200, 201, 202],
                },
            ]
        },
    )


def _write_support_case(
    root: Path,
    *,
    name: str,
    candidate_count: int,
    edge_count: int,
    schur_edges: int,
    balance_edges: int,
    accepted_delta: float | None = None,
    triggered: int | None = None,
    worst_dof: int | None = None,
    nonlinear_converged: bool = True,
):
    support_path = root / f"{name}_support_audit_20260606.json"
    case_dir = root / f"{name}_20260606_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        support_path,
        {
            "solver_log": "run.log",
            "accepted_pressure_update_count": 1 if accepted_delta is not None else 0,
            "latest_pressure_graph_completion": {
                "values": {
                    "mode": "shared_row_schur_existing_edge_balance",
                    "candidate_selector": "synthetic_selector",
                    "candidate_row_count": candidate_count,
                    "edge_count": edge_count,
                    "shared_row_schur_edge_count": schur_edges,
                    "existing_balance_edge_count": balance_edges,
                    "balance_candidate_row_count": balance_edges // 8,
                }
            },
            "latest_accepted_pressure_update": (
                {
                    "values": {
                        "global_abs_pressure_delta_pa": accepted_delta,
                        "threshold_pa": 100.0,
                        "triggered": triggered,
                        "local_worst_dof": worst_dof,
                    }
                }
                if accepted_delta is not None
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
                    f"step=0 time=1.0 converged={converged} iters=12 "
                    "||r||=2.500000e+04 ||r_field||=2.500000e+04 ||r_aux||=0"
                ),
                (
                    "[svMultiPhysics::Application] TimeLoop: loop.run() returned "
                    f"success={success} steps_taken={success} final_time=1.0 "
                    f"message='{message}'"
                ),
                "Message: TimeLoop: nonlinear solve did not converge"
                if not nonlinear_converged
                else "",
            ]
        ),
        encoding="utf-8",
    )
    return support_path


def test_support_artifact_classifies_guard_clear_and_nonlinear_failure(tmp_path):
    audit = _load_audit_module()
    target_map_path = tmp_path / "targets.json"
    _write_target_map(target_map_path)
    test02_support = _write_support_case(
        tmp_path,
        name="test02_variant",
        candidate_count=32,
        edge_count=100,
        schur_edges=80,
        balance_edges=64,
        nonlinear_converged=False,
    )
    test10_support = _write_support_case(
        tmp_path,
        name="test10_variant",
        candidate_count=12,
        edge_count=40,
        schur_edges=30,
        balance_edges=24,
        accepted_delta=6.0,
        triggered=0,
        worst_dof=200,
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        target_map_path=target_map_path,
        variant_specs=[
            {
                "key": "synthetic_balance",
                "description": "synthetic",
                "test02_support": test02_support.name,
                "test10_support": test10_support.name,
            }
        ],
    )

    variant = report["variants"][0]
    assert variant["finding"] == "test10_clears_but_test02_unstable"
    cases = {case["label"]: case for case in variant["cases"]}
    assert cases["test02"]["finding"] == "nonlinear_failed_with_overbroad_patch"
    assert cases["test02"]["final_residual_norm"] == 25000.0
    assert cases["test10"]["finding"] == "guard_cleared"
    assert cases["test10"]["accepted_pressure_update_pa"] == 6.0


def test_build_report_rules_out_replay_family_when_localized_gates_miss(tmp_path):
    audit = _load_audit_module()
    target_map_path = tmp_path / "targets.json"
    _write_target_map(target_map_path)

    schur_only_t02 = _write_support_case(
        tmp_path,
        name="test02_schur_only",
        candidate_count=32,
        edge_count=80,
        schur_edges=80,
        balance_edges=0,
        accepted_delta=200000.0,
        triggered=1,
        worst_dof=100,
    )
    schur_only_t10 = _write_support_case(
        tmp_path,
        name="test10_schur_only",
        candidate_count=12,
        edge_count=30,
        schur_edges=30,
        balance_edges=0,
        accepted_delta=122.0,
        triggered=1,
        worst_dof=200,
    )
    broad_balance_t02 = _write_support_case(
        tmp_path,
        name="test02_broad_balance",
        candidate_count=32,
        edge_count=120,
        schur_edges=80,
        balance_edges=96,
        nonlinear_converged=False,
    )
    broad_balance_t10 = _write_support_case(
        tmp_path,
        name="test10_broad_balance",
        candidate_count=12,
        edge_count=50,
        schur_edges=30,
        balance_edges=40,
        accepted_delta=15.0,
        triggered=0,
        worst_dof=201,
    )
    neighborhood_t02 = _write_support_case(
        tmp_path,
        name="test02_neighborhood",
        candidate_count=0,
        edge_count=0,
        schur_edges=0,
        balance_edges=0,
        accepted_delta=366000.0,
        triggered=1,
        worst_dof=100,
    )
    neighborhood_t10 = _write_support_case(
        tmp_path,
        name="test10_neighborhood",
        candidate_count=8,
        edge_count=20,
        schur_edges=12,
        balance_edges=16,
        accepted_delta=320.0,
        triggered=1,
        worst_dof=202,
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        target_map_path=target_map_path,
        variant_specs=[
            {
                "key": "least_selector_schur_only",
                "description": "synthetic schur only",
                "test02_support": schur_only_t02.name,
                "test10_support": schur_only_t10.name,
            },
            {
                "key": "least_selector_schur_edge_balance",
                "description": "synthetic broad balance",
                "test02_support": broad_balance_t02.name,
                "test10_support": broad_balance_t10.name,
            },
            {
                "key": "support_rank_neighborhood_depth1",
                "description": "synthetic neighborhood",
                "test02_support": neighborhood_t02.name,
                "test10_support": neighborhood_t10.name,
            },
        ],
    )

    assert report["finding"] == (
        "direct_pspg_graph_completion_replay_family_rules_out_"
        "post_assembly_selector_variants"
    )
    assert report["variant_findings"]["least_selector_schur_only"] == (
        "both_guards_still_trigger"
    )
    assert report["variant_findings"]["least_selector_schur_edge_balance"] == (
        "test10_clears_but_test02_unstable"
    )
    assert report["variant_findings"]["support_rank_neighborhood_depth1"] == (
        "both_guards_still_trigger"
    )
