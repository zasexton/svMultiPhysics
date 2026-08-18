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
        / "audit_pressure_constraint_coverage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_constraint_coverage",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[R0] [INFO] LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint "
                "field='Pressure' total_vertices=5 total_dofs=5 inactive_dofs=2 "
                "inactive_dof_runs=0|2 inactive_vertex_runs=0|2",
                "[svMultiPhysics::Application] "
                "Active pressure support constraint refresh "
                "diagnostic=active_pressure_constraint_refresh "
                "provenance=initial support_source=retained_cut_context "
                "constraints=10",
                "[R0] [INFO] LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint "
                "field='Pressure' total_vertices=5 total_dofs=5 inactive_dofs=3 "
                "inactive_dof_runs=0-1|4 inactive_vertex_runs=0-1|4",
                "[R0] [INFO] LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint_sample "
                "field='Pressure' level_set_field='phi' local_dof=2 status=ok "
                "global_dof=12 owned=1 active_dof_support=1 "
                "inactive_constraint=0 constrained_owned=0 entity_kind=Vertex "
                "entity_id=3 entity_dofs=2 vertex_phi=-0.5 "
                "vertex_active_sign=1 vertex_active_support=1",
                "[R0] [INFO] NewtonSolver: matrix support diagnostic "
                "diagnostic=newton_matrix_support_sample rank=0 iteration=0 "
                "phase='pre_linear_solve' backend=eigen solve_time=0.9 "
                "dt=0.1 dof=12 status=ok row_abs_sum=0 "
                "row_numeric_entries=0 row_max_abs=0 col_abs_sum=0 "
                "col_numeric_entries=0 col_max_abs=0 diag=0 "
                "row_first_nonzero=none col_first_nonzero=none "
                "field='Pressure' field_local_dof=2",
                "[svMultiPhysics::FE] Eigen direct factorization diagnostic "
                "phase=factorize info=numerical_issue rows=15 cols=15 "
                "zero_rows=3 zero_cols=3 zero_rows_first=10|12|14 "
                "zero_cols_first=10|12|14 zero_row_runs=10|12|14 "
                "zero_col_runs=10|12|14 block_summaries="
                "phi{begin=0,end=5,zero_rows=0,zero_cols=0,"
                "zero_rows_first_local=none,zero_cols_first_local=none,"
                "zero_row_runs_local=none,zero_col_runs_local=none};"
                "Pressure{begin=10,end=15,zero_rows=3,zero_cols=3,"
                "zero_rows_first_local=0|2|4,zero_cols_first_local=0|2|4,"
                "zero_row_runs_local=0|2|4,zero_col_runs_local=0|2|4,"
                "zero_diag=3,identity_rows=3,min_positive_row_sum=1,"
                "max_row_sum=2}",
                "[R0] [INFO] LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint "
                "field='Pressure' total_vertices=5 total_dofs=5 inactive_dofs=1 "
                "inactive_dof_runs=2 inactive_vertex_runs=2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_support_audit(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_result": "synthetic/result_001.vtu",
                "zero_pressure_rows": [
                    {
                        "local_pressure_row": 0,
                        "support_class": "dry_or_inactive",
                        "point_index": 0,
                        "phi": 1.0,
                        "active_fluid": 0.0,
                    },
                    {
                        "local_pressure_row": 2,
                        "support_class": "dry_or_inactive",
                        "point_index": 2,
                        "phi": 1.0,
                        "active_fluid": 0.0,
                    },
                    {
                        "local_pressure_row": 4,
                        "support_class": "full_wet_supported",
                        "point_index": 4,
                        "phi": -1.0,
                        "active_fluid": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_constraint_coverage_selects_last_constraint_before_factorization(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    support = tmp_path / "support.json"
    _write_log(log)
    _write_support_audit(support)

    report = audit.audit_pressure_constraint_coverage(
        solver_log=log,
        row_support_audit=support,
    )

    assert report["constraint_selection"] == "last_before_factorization"
    assert report["constraint_line_number"] == 3
    assert report["constraint_sample_count"] == 1
    assert report["constraint_inactive_dof_count_from_runs"] == 3
    assert report["constraint_inactive_dof_count_matches_log"]
    assert report["row_support_mapping_reliable"]
    assert (
        report["constraint_vertex_dof_mapping_status"]["status"]
        == "inactive_dof_runs_match_inactive_vertex_runs"
    )
    assert report["zero_rows_in_constraint_inactive_count"] == 2
    assert report["zero_rows_missing_constraint_count"] == 1
    assert report["saved_dry_zero_rows_missing_constraint_count"] == 1
    assert report["mismatch_class_counts"] == {
        "consistent": 1,
        "saved_dry_not_constraint_inactive": 1,
        "saved_supported_but_constraint_inactive": 1,
    }
    assert report["zero_pressure_rows"][0]["constraint_inactive"]
    assert not report["zero_pressure_rows"][1]["constraint_inactive"]
    assert report["zero_pressure_rows"][2]["constraint_inactive"]
    assert report["runtime_sampled_zero_row_count"] == 1
    assert report["runtime_sampled_zero_row_entity_kind_counts"] == {"Vertex": 1}
    assert report["runtime_sampled_zero_rows_active_dof_support_count"] == 1
    assert report["runtime_sampled_zero_rows_vertex_active_sign_count"] == 1
    assert report["zero_pressure_rows"][1]["runtime_sample"]["entity_id"] == 3
    assert report["matrix_sample_count"] == 1
    assert report["matrix_sampled_zero_row_count"] == 1
    assert report["matrix_sample_status_counts"] == {"ok": 1}
    assert report["matrix_sampled_zero_rows_zero_row_count"] == 1
    assert report["matrix_sampled_zero_rows_zero_col_count"] == 1
    assert report["matrix_sampled_zero_rows_zero_diag_count"] == 1
    assert report["zero_pressure_rows"][1]["matrix_sample"]["field_local_dof"] == 2


def test_constraint_coverage_marks_saved_support_unverified_when_dof_order_differs(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    support = tmp_path / "support.json"
    _write_log(log)
    _write_support_audit(support)
    text = log.read_text(encoding="utf-8")
    text = text.replace(
        "inactive_dof_runs=0-1|4 inactive_vertex_runs=0-1|4",
        "inactive_dof_runs=0-1|4 inactive_vertex_runs=2-4",
        1,
    )
    log.write_text(text, encoding="utf-8")

    report = audit.audit_pressure_constraint_coverage(
        solver_log=log,
        row_support_audit=support,
    )

    assert not report["row_support_mapping_reliable"]
    assert (
        report["constraint_vertex_dof_mapping_status"]["status"]
        == "inactive_dof_runs_differ_from_inactive_vertex_runs"
    )
    assert report["mismatch_class_counts"] == {
        "saved_support_mapping_unverified": 3,
    }
    assert report["saved_dry_zero_rows_missing_constraint_count"] == 0
    assert report["unverified_saved_support_zero_row_count"] == 3
