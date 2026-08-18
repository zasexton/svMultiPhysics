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
        / "audit_pressure_operator_top_update_overlap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_operator_top_update_overlap",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _top_row(global_dof, row_coupling, row_self, abs_update=100.0):
    return {
        "global_dof": global_dof,
        "local_pressure_row": global_dof - 1000,
        "abs_update": abs_update,
        "update": abs_update,
        "row_coupling": row_coupling,
        "row_self": row_self,
    }


def _exact_sample(op, dof, *, coupling=0.0, self_sum=0.0, line=1):
    row_abs = abs(coupling) + abs(self_sum)
    return {
        "line_number": line,
        "local_pressure_row": dof - 1000,
        "op": op,
        "operator_matrix_support": {
            "dof": dof,
            "op": op,
            "row_abs_sum": row_abs,
            "row_coupling_abs_sum": coupling,
            "row_self_abs_sum": self_sum,
            "diag": self_sum / 2.0,
            "status": "ok",
        },
    }


def _exact_samples_for_row(audit, row, *, direct_dofs, ghost_dofs):
    dof = row["global_dof"]
    direct_self = 2.0e-8 if dof in direct_dofs else 0.0
    wall_self = 8.0e-8 if dof in direct_dofs else 0.0
    ghost_self = 0.8 if dof in ghost_dofs else 0.0
    return [
        _exact_sample(audit.OP_GALERKIN, dof, coupling=row["row_coupling"]),
        _exact_sample(audit.OP_NONPRESSURE, dof, coupling=row["row_coupling"] / 2.0),
        _exact_sample(audit.OP_DIRECT_PGRAD, dof, self_sum=direct_self),
        _exact_sample(audit.OP_WALL_NORMAL_PGRAD, dof),
        _exact_sample(audit.OP_WALL_TANGENTIAL_PGRAD, dof, self_sum=wall_self),
        _exact_sample(audit.OP_GHOST, dof, self_sum=ghost_self),
    ]


def _support_report(audit, *, rows, gal_zero, pspg_zero, direct_dofs=(), ghost_dofs=()):
    exact_samples = []
    direct_dofs = set(direct_dofs)
    ghost_dofs = set(ghost_dofs)
    for row in rows:
        exact_samples.extend(
            _exact_samples_for_row(
                audit,
                row,
                direct_dofs=direct_dofs,
                ghost_dofs=ghost_dofs,
            )
        )
    return {
        "pressure_update_support_summary": {
            "top_update_details": rows,
        },
        "latest_support_rank_diagnostic": {
            "values": {
                "zero_coupling_row_global_dofs": "|".join(
                    str(dof) for dof in gal_zero
                )
                or "none",
            },
        },
        "pressure_row_operator_matrix_support_samples": exact_samples,
        "operator_matrix_summary_by_op": {
            "equations_diagnostic_ns_galerkin_continuity": {
                "status": "ok",
                "unconstrained_pressure_rows": 10,
                "sample_limit": 12,
                "zero_coupling_row_block_count": len(gal_zero),
                "weak_coupling_row_block_count": 4,
                "pressure_only_row_block_count": 0,
                "weak_self_row_block_count": 0,
                "positive_coupling_row_block_count": 10,
                "positive_self_row_block_count": 0,
                "zero_coupling_row_global_dofs": "|".join(
                    str(dof) for dof in gal_zero
                )
                or "none",
                "zero_row_global_dofs": "|".join(str(dof) for dof in gal_zero)
                or "none",
                "weakest_coupling_row_global_dofs": "1001|1002",
                "weakest_self_row_global_dofs": "none",
            },
            "equations_diagnostic_ns_vms_pspg_nonpressure": {
                "status": "ok",
                "unconstrained_pressure_rows": 10,
                "sample_limit": 12,
                "zero_coupling_row_block_count": len(gal_zero),
                "weak_coupling_row_block_count": 5,
                "pressure_only_row_block_count": 0,
                "weak_self_row_block_count": 0,
                "positive_coupling_row_block_count": 10,
                "positive_self_row_block_count": 0,
                "zero_coupling_row_global_dofs": "|".join(
                    str(dof) for dof in gal_zero
                )
                or "none",
                "zero_row_global_dofs": "|".join(str(dof) for dof in gal_zero)
                or "none",
                "weakest_coupling_row_global_dofs": "1001|1002|1003",
                "weakest_self_row_global_dofs": "none",
            },
            "equations_diagnostic_ns_vms_pspg_pressure_gradient": {
                "status": "ok",
                "unconstrained_pressure_rows": 10,
                "sample_limit": 12,
                "zero_coupling_row_block_count": len(pspg_zero),
                "weak_coupling_row_block_count": 8,
                "pressure_only_row_block_count": len(pspg_zero),
                "weak_self_row_block_count": 10,
                "positive_coupling_row_block_count": 8,
                "positive_self_row_block_count": 10,
                "zero_coupling_row_global_dofs": "|".join(
                    str(dof) for dof in pspg_zero
                )
                or "none",
                "zero_row_global_dofs": "none",
                "weakest_coupling_row_global_dofs": "1010|1011",
                "weakest_self_row_global_dofs": "1002|1012",
            },
        },
    }


def test_operator_top_update_overlap_flags_mixed_no_galerkin_coverage():
    audit = _load_audit_module()
    partial = _support_report(
        audit,
        rows=[
            _top_row(1001, 0.0, 2.0e-7),
            _top_row(1002, 2.0e-4, 2.0e-7),
            _top_row(1003, 2.0e-3, 2.0e-8),
        ],
        gal_zero=[1001],
        pspg_zero=[1001, 1002],
        direct_dofs={1001, 1003},
        ghost_dofs={1002},
    )
    absent = _support_report(
        audit,
        rows=[
            _top_row(2001, 3.0e-3, 2.0e-9),
            _top_row(2002, 1.0e-8, 8.0e-1),
        ],
        gal_zero=[],
        pspg_zero=[2002],
        direct_dofs={2001, 2002},
    )

    report = audit.build_report(
        [
            ("partial", Path("partial.json"), partial),
            ("absent", Path("absent.json"), absent),
        ]
    )

    assert report["finding"] == (
        "mixed_no_galerkin_overlap_partial_for_some_cases_absent_for_others"
    )
    assert report["no_galerkin_support_finding"] == (
        "no_galerkin_support_rank_equivalent_but_partial_top_overlap"
    )
    assert report["no_galerkin_support_finding_counts"] == {
        "no_galerkin_support_rank_equivalent_partial_top_overlap": 1,
        "no_galerkin_zero_coupling_absent": 1,
    }
    assert report["exact_to_aggregate_sample_finding"] == (
        "exact_direct_pspg_top_rows_undercovered_by_aggregate_samples"
    )
    assert report["finding_counts"] == {
        "no_top_update_overlap_no_galerkin_zero_coupling_sample": 1,
        "partial_top_update_overlap_no_galerkin_zero_coupling_sample": 1,
    }

    partial_case = report["cases"][0]
    assert partial_case["row_support_class_counts"] == {
        "positive_coupling:weak_self": 1,
        "weak_coupling:positive_self": 1,
        "zero_coupling:positive_self": 1,
    }
    assert partial_case["exact_physical_path_class_counts"] == {
        "direct_pspg_weak_self_with_wall_support": 2,
        "ghost_penalty_positive_self": 1,
    }
    assert partial_case["exact_direct_pspg_top_update_global_dofs"] == [
        1001,
        1003,
    ]
    assert partial_case["no_galerkin_support_finding"] == (
        "no_galerkin_support_rank_equivalent_partial_top_overlap"
    )
    assert partial_case["no_galerkin_equals_support_rank_zero_coupling"]
    assert partial_case["no_nonpressure_equals_support_rank_zero_coupling"]
    assert partial_case["no_galerkin_top_update_overlap_global_dofs"] == [1001]
    assert partial_case["support_rank_minus_no_galerkin_global_dofs"] == []
    assert partial_case["no_galerkin_minus_support_rank_global_dofs"] == []
    assert partial_case[
        "exact_direct_pspg_rows_with_direct_pgrad_aggregate_sample_global_dofs"
    ] == [1001]
    assert partial_case[
        "exact_direct_pspg_rows_missing_direct_pgrad_aggregate_sample_global_dofs"
    ] == [1003]
    gal = partial_case["operator_overlaps"][
        "equations_diagnostic_ns_galerkin_continuity"
    ]
    assert gal["top_update_zero_coupling_sample_hit_count"] == 1
    assert gal["top_update_zero_coupling_sample_hit_global_dofs"] == [1001]

    pspg = partial_case["operator_overlaps"][
        "equations_diagnostic_ns_vms_pspg_pressure_gradient"
    ]
    assert pspg["top_update_zero_coupling_sample_hit_count"] == 2
    assert pspg["top_update_zero_coupling_sample_hit_global_dofs"] == [
        1001,
        1002,
    ]
    first_row = partial_case["top_update_rows"][0]
    assert first_row["operator_sample_membership"][
        "equations_diagnostic_ns_galerkin_continuity"
    ]["zero_coupling_sample"]
    assert first_row["operator_sample_membership"][
        "equations_diagnostic_ns_vms_pspg_pressure_gradient"
    ]["zero_coupling_sample"]

    absent_case = report["cases"][1]
    assert absent_case["finding"] == (
        "no_top_update_overlap_no_galerkin_zero_coupling_sample"
    )
