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
        / "audit_pressure_operator_toprow_provenance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_operator_toprow_provenance",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _top_row(
    global_dof,
    row_coupling,
    row_self,
    abs_update=10.0,
    pressure_action_terms=None,
):
    return {
        "global_dof": global_dof,
        "local_pressure_row": global_dof - 100,
        "abs_update": abs_update,
        "update": abs_update,
        "row_coupling": row_coupling,
        "row_self": row_self,
        **(
            {"pressure_action_terms": pressure_action_terms}
            if pressure_action_terms is not None
            else {}
        ),
    }


def _sample(
    op,
    dof,
    *,
    coupling=0.0,
    self_sum=0.0,
    row_abs=None,
    line=1,
    row_first_nonzero=None,
):
    if row_abs is None:
        row_abs = abs(coupling) + abs(self_sum)
    return {
        "line_number": line,
        "local_pressure_row": dof - 100,
        "op": op,
        "operator_matrix_support": {
            "dof": dof,
            "op": op,
            "pressure_offset": 100,
            "row_abs_sum": row_abs,
            "row_coupling_abs_sum": coupling,
            "row_self_abs_sum": self_sum,
            "row_numeric_entries": 1,
            "row_self_numeric_entries": 1 if self_sum else 0,
            "row_self_offdiag_abs_sum": abs(self_sum) / 2.0,
            "row_self_diag_abs_ratio": 0.5 if self_sum else 0.0,
            "row_self_signed_abs_ratio": 0.0,
            "row_first_nonzero": row_first_nonzero or (
                f"{dof}:{self_sum}" if self_sum else "none"
            ),
            "col_first_nonzero": f"{dof}:{self_sum}" if self_sum else "none",
            "diag": self_sum / 2.0,
            "status": "ok",
        },
    }


def _support_report(
    audit,
    *,
    rows,
    ghost_dofs=(),
    direct_self_by_dof=None,
    direct_entries_by_dof=None,
    wall_normal_entries_by_dof=None,
    wall_tangential_entries_by_dof=None,
    direct_row_first_by_dof=None,
):
    direct_self_by_dof = direct_self_by_dof or {}
    direct_entries_by_dof = direct_entries_by_dof or {}
    wall_normal_entries_by_dof = wall_normal_entries_by_dof or {}
    wall_tangential_entries_by_dof = wall_tangential_entries_by_dof or {}
    direct_row_first_by_dof = direct_row_first_by_dof or {}
    samples = []
    for row in rows:
        dof = row["global_dof"]
        direct_self = direct_self_by_dof.get(dof, 2.0e-8)
        wall_normal_self = 1.0e-8 if wall_normal_entries_by_dof.get(dof, 0) else 0.0
        wall_tangential_self = (
            8.0e-8 if wall_tangential_entries_by_dof.get(dof, 5) else 0.0
        )
        samples.extend(
            [
                _sample(audit.OP_GALERKIN, dof, coupling=row["row_coupling"]),
                _sample(audit.OP_NONPRESSURE, dof, coupling=row["row_coupling"] / 2.0),
                _sample(
                    audit.OP_DIRECT_PGRAD,
                    dof,
                    self_sum=direct_self,
                    row_abs=direct_self,
                    row_first_nonzero=direct_row_first_by_dof.get(dof),
                ),
                _sample(
                    audit.OP_WALL_NORMAL_PGRAD,
                    dof,
                    self_sum=wall_normal_self,
                    row_abs=wall_normal_self,
                ),
                _sample(
                    audit.OP_WALL_TANGENTIAL_PGRAD,
                    dof,
                    self_sum=wall_tangential_self,
                    row_abs=wall_tangential_self,
                ),
                _sample(
                    audit.OP_GHOST,
                    dof,
                    self_sum=0.8 if dof in ghost_dofs else 0.0,
                ),
            ]
        )
        samples[-4]["operator_matrix_support"]["row_self_numeric_entries"] = (
            direct_entries_by_dof.get(dof, 5)
        )
        samples[-4]["operator_matrix_support"]["row_numeric_entries"] = (
            direct_entries_by_dof.get(dof, 5)
        )
        samples[-3]["operator_matrix_support"]["row_self_numeric_entries"] = (
            wall_normal_entries_by_dof.get(dof, 0)
        )
        samples[-3]["operator_matrix_support"]["row_numeric_entries"] = (
            wall_normal_entries_by_dof.get(dof, 0)
        )
        samples[-2]["operator_matrix_support"]["row_self_numeric_entries"] = (
            wall_tangential_entries_by_dof.get(dof, 5)
        )
        samples[-2]["operator_matrix_support"]["row_numeric_entries"] = (
            wall_tangential_entries_by_dof.get(dof, 5)
        )
    return {
        "latest_pressure_update_support_diagnostic": {
            "values": {"pressure_offset": 100}
        },
        "pressure_update_support_summary": {"top_update_details": rows},
        "pressure_row_operator_matrix_support_samples": samples,
    }


def test_operator_toprow_provenance_splits_direct_pspg_and_ghost_paths():
    audit = _load_audit_module()
    mixed = _support_report(
        audit,
        rows=[
            _top_row(
                100,
                3.0e-4,
                2.0e-8,
                pressure_action_terms=(
                    "0/100/m=2e-08/u=10/a=2e-07~"
                    "3/103/m=-1e-08/u=10/a=-1e-07"
                ),
            ),
            _top_row(101, 1.0e-8, 0.8),
        ],
        ghost_dofs={101},
    )
    direct = _support_report(
        audit,
        rows=[
            _top_row(102, 0.0, 8.0e-8),
            _top_row(
                103,
                2.0e-4,
                8.0e-8,
                pressure_action_terms=(
                    "3/103/m=4e-08/u=10/a=4e-07~"
                    "2/102/m=-1e-08/u=10/a=-1e-07"
                ),
            ),
        ],
        ghost_dofs=set(),
        direct_self_by_dof={102: 1.0e-8, 103: 4.0e-8},
        direct_entries_by_dof={102: 4, 103: 6},
        wall_normal_entries_by_dof={102: 0, 103: 2},
        wall_tangential_entries_by_dof={102: 5, 103: 0},
        direct_row_first_by_dof={
            102: "102:1.0e-8|103:-1.0e-8",
            103: "102:-1.0e-8|103:4.0e-8",
        },
    )
    mixed_pressure_disabled = _support_report(
        audit,
        rows=[
            _top_row(103, 2.0e-4, 8.0e-8),
        ],
        direct_self_by_dof={103: 4.0e-8},
    )
    points = [
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.5],
        [0.5, 0.5, 1.0],
    ]
    bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    incident_cell_counts = [1, 4, 1, 2]

    mixed_case = audit.audit_case(
        "mixed",
        Path("mixed.json"),
        mixed,
        points=points,
        bounds=bounds,
        incident_cell_counts=incident_cell_counts,
    )
    direct_case = audit.audit_case(
        "direct",
        Path("direct.json"),
        direct,
        points=points,
        bounds=bounds,
        incident_cell_counts=incident_cell_counts,
    )
    mixed_pressure_disabled_case = audit.audit_case(
        "mixed_pressure_disabled",
        Path("mixed_pressure_disabled.json"),
        mixed_pressure_disabled,
        points=points,
        bounds=bounds,
        incident_cell_counts=incident_cell_counts,
    )
    report = audit.summarize_cases(
        [mixed_case, direct_case, mixed_pressure_disabled_case]
    )

    assert report["finding"] == (
        "top_rows_split_between_direct_pspg_and_ghost_penalty_paths"
    )
    assert mixed_case["finding"] == "mixed_direct_pspg_and_ghost_penalty_top_rows"
    assert mixed_case["physical_path_class_counts"] == {
        "direct_pspg_weak_self_with_wall_support": 1,
        "ghost_penalty_positive_self": 1,
    }
    assert mixed_case["direct_pspg_balance_global_dofs"] == [100]
    assert mixed_case["ghost_penalty_balance_global_dofs"] == [101]
    assert mixed_case["operator_top_row_balance_global_dofs"] == [100, 101]
    assert mixed_case["balance_global_dofs_by_physical_path"] == {
        "direct_pspg_weak_self_with_wall_support": [100],
        "ghost_penalty_positive_self": [101],
    }
    assert mixed_case[
        "direct_pspg_same_sign_pressure_action_patch_finding"
    ] == "direct_pspg_top_rows_same_sign_action_isolated"
    assert mixed_case[
        "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs"
    ] == [100]
    assert direct_case["finding"] == "direct_pspg_top_rows_without_ghost_penalty"
    assert direct_case["ghost_penalty_self_class_counts"] == {"zero": 2}
    assert direct_case["direct_pgrad_self_class_counts"] == {"weak": 2}
    assert direct_case["direct_pspg_balance_global_dofs"] == [102, 103]
    assert direct_case["ghost_penalty_balance_global_dofs"] == []
    assert direct_case["operator_top_row_balance_global_dofs"] == [102, 103]
    assert mixed_case["boundary_class_counts"] == {
        "boundary_face": 1,
        "interior": 1,
    }
    assert mixed_case["incident_support_class_counts"] == {
        "interior_shared_support": 1,
        "one_cell_boundary_support": 1,
    }
    assert mixed_case["source_result_incident_support_loaded"]
    assert mixed_case["direct_pspg_one_cell_boundary_global_dofs"] == [100]
    assert mixed_case["direct_pspg_non_one_cell_boundary_global_dofs"] == []
    assert direct_case["boundary_class_counts"] == {
        "boundary_edge": 1,
        "boundary_face": 1,
    }
    assert direct_case["incident_support_class_counts"] == {
        "one_cell_boundary_support": 1,
        "shared_boundary_support": 1,
    }
    assert direct_case["direct_pspg_one_cell_boundary_global_dofs"] == [102]
    assert direct_case["direct_pspg_non_one_cell_boundary_global_dofs"] == [103]
    assert direct_case["direct_pspg_case_max_direct_self_abs_sum"] == 4.0e-8
    assert direct_case["direct_pspg_low_direct_self_ratio_global_dofs"] == [102]
    assert direct_case["direct_pspg_moderate_direct_self_ratio_global_dofs"] == [
        102
    ]
    assert direct_case["direct_pspg_low_total_self_ratio_global_dofs"] == []
    assert direct_case["direct_pspg_case_min_direct_self_numeric_entries"] == 4
    assert direct_case["direct_pspg_case_max_direct_self_numeric_entries"] == 6
    assert direct_case["direct_pspg_sparse_direct_self_entry_global_dofs"] == [
        102
    ]
    assert direct_case["direct_pspg_missing_wall_normal_self_global_dofs"] == [
        102
    ]
    assert direct_case["direct_pspg_missing_wall_tangential_self_global_dofs"] == [
        103
    ]
    assert direct_case[
        "direct_pspg_zero_galerkin_nonpressure_coupling_global_dofs"
    ] == [102]
    assert direct_case[
        "direct_pspg_rows_with_direct_pgrad_top_neighbors_global_dofs"
    ] == [102, 103]
    assert direct_case[
        "direct_pspg_rows_with_direct_pgrad_direct_top_neighbors_global_dofs"
    ] == [102, 103]
    assert direct_case["direct_pspg_direct_pgrad_top_neighbor_edge_count"] == 1
    assert direct_case[
        "direct_pspg_rows_with_pressure_action_top_neighbors_global_dofs"
    ] == [103]
    assert direct_case[
        "direct_pspg_rows_with_same_sign_pressure_action_top_neighbors_global_dofs"
    ] == [103]
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_top_neighbor_edge_count"
    ] == 1
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_patch_finding"
    ] == "direct_pspg_top_rows_single_same_sign_action_patch"
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_component_count"
    ] == 1
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_largest_component_size"
    ] == 2
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs"
    ] == [102, 103]
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs"
    ] == []
    assert direct_case[
        "direct_pspg_same_sign_pressure_action_components"
    ] == [
        {
            "component_index": 1,
            "size": 2,
            "global_dofs": [102, 103],
            "direct_pspg_global_dofs": [102, 103],
            "ghost_penalty_global_dofs": [],
            "rank_values": [1, 2],
            "contains_rank1": True,
            "max_abs_update": 10.0,
            "max_abs_update_global_dof": 102,
            "boundary_class_counts": {
                "boundary_edge": 1,
                "boundary_face": 1,
            },
            "incident_support_class_counts": {
                "one_cell_boundary_support": 1,
                "shared_boundary_support": 1,
            },
            "same_sign_pressure_action_edge_count": 1,
        }
    ]
    assert direct_case["top_update_rows"][0][
        "pspg_pressure_gradient_support_profile"
    ]["direct_self_to_case_direct_max_ratio"] == 0.25
    assert direct_case["top_update_rows"][0][
        "pspg_pressure_gradient_support_topology_profile"
    ]["sparse_direct_self_entries"]
    assert direct_case["top_update_rows"][1][
        "pspg_pressure_gradient_support_topology_profile"
    ]["missing_wall_tangential_self_support"]
    assert direct_case["top_update_rows"][0][
        "direct_pspg_patch_neighbor_profile"
    ]["direct_pgrad_top_update_neighbor_dofs"] == [103]
    assert direct_case["top_update_rows"][1][
        "direct_pspg_patch_neighbor_profile"
    ]["same_sign_pressure_action_top_update_neighbor_dofs"] == [102]
    assert report["cross_policy_neighbor_comparisons"] == [
        {
            "base_label": "mixed",
            "full_gradient_label": "mixed",
            "pressure_disabled_label": "mixed_pressure_disabled",
            "full_gradient_direct_row_count": 1,
            "pressure_disabled_direct_row_count": 1,
            "full_direct_rows_with_pressure_disabled_direct_action_neighbors_global_dofs": [
                100
            ],
            "full_direct_rows_with_pressure_disabled_direct_row_neighbors_global_dofs": [],
            "full_direct_rows_with_same_sign_pressure_disabled_direct_action_neighbors_global_dofs": [
                100
            ],
            "full_direct_rows_current_top_isolated_but_pressure_disabled_direct_connected_global_dofs": [
                100
            ],
            "current_top_isolated_cross_policy_patch_global_dofs": [100, 103],
            "current_top_isolated_cross_policy_patch_env_value": "100,103",
            "pressure_disabled_direct_action_neighbor_edge_count": 1,
            "pressure_disabled_direct_row_neighbor_edge_count": 0,
            "same_sign_pressure_disabled_direct_action_neighbor_edge_count": 1,
            "rows": [
                {
                    "global_dof": 100,
                    "rank": 1,
                    "abs_update": 10.0,
                    "current_top_action_neighbor_count": 0,
                    "current_top_direct_neighbor_count": 0,
                    "pressure_disabled_direct_action_neighbor_dofs": [103],
                    "pressure_disabled_direct_row_neighbor_dofs": [],
                    "same_sign_pressure_disabled_direct_action_neighbor_dofs": [103],
                    "current_top_isolated_cross_policy_patch_global_dofs": [
                        100,
                        103,
                    ],
                    "current_top_isolated_but_pressure_disabled_direct_connected": True,
                }
            ],
        }
    ]
