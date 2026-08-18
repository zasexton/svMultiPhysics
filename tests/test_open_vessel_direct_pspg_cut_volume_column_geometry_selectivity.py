import importlib.util
import math
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
        / "audit_direct_pspg_cut_volume_column_geometry_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_column_geometry_selectivity",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_map(targets=None):
    return {
        "cases": [
            {
                "label": "test02",
                "direct_pspg_target_global_dofs": targets or [100],
            },
        ],
    }


def _distance(lhs, rhs):
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(lhs, rhs)))


def _entry(
    op,
    rule_index,
    row_dof,
    col_dofs,
    values,
    *,
    row_ref=(0.0, 0.0, 0.0),
    col_refs=None,
    parent_cell=3,
    include_geometry=True,
):
    col_refs = col_refs or [row_ref for _ in col_dofs]
    row_abs_sum = sum(abs(value) for value in values)
    row_signed_sum = sum(values)
    positive_sum = sum(value for value in values if value > 0.0)
    negative_abs_sum = sum(-value for value in values if value < 0.0)
    diag_value = sum(
        value for col_dof, value in zip(col_dofs, values) if col_dof == row_dof
    )
    diag_abs = abs(diag_value)
    offdiag_abs = row_abs_sum - diag_abs
    signs = [1 if value > 0.0 else -1 if value < 0.0 else 0 for value in values]
    line = (
        "StandardAssembler: "
        "diagnostic=cut_volume_local_matrix_column_support "
        f"status=ok op='{op}' marker=7 side=Negative test='Pressure' "
        "trial='Pressure' "
        f"rule_index={rule_index} parent_cell={parent_cell} full_cell=1 "
        "volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index=0 row_dof={row_dof} col_count={len(col_dofs)} "
        f"nonzero_col_count={len(col_dofs)} "
        f"positive_col_count={sum(1 for value in values if value > 0.0)} "
        f"negative_col_count={sum(1 for value in values if value < 0.0)} "
        f"sampled_col_count={len(col_dofs)} sample_truncated=0 "
        "sample_sorted_by=abs_desc "
        f"row_abs_sum={row_abs_sum} row_signed_sum={row_signed_sum} "
        f"positive_sum={positive_sum} negative_abs_sum={negative_abs_sum} "
        f"has_diag={1 if diag_abs else 0} diag_in_sample={1 if diag_abs else 0} "
        f"diag_value={diag_value} diag_abs={diag_abs} "
        f"offdiag_abs_sum={offdiag_abs} "
        "sampled_col_local_indices="
        f"{'|'.join(str(index) for index, _ in enumerate(col_dofs))} "
        f"sampled_col_dofs={'|'.join(str(col_dof) for col_dof in col_dofs)} "
        f"sampled_col_values={'|'.join(str(value) for value in values)} "
        f"sampled_col_abs_values={'|'.join(str(abs(value)) for value in values)} "
        f"sampled_col_signs={'|'.join(str(sign) for sign in signs)}"
    )
    if include_geometry:
        lengths = [_distance(row_ref, col_ref) for col_ref in col_refs]
        line += (
            " test_element_type=3 trial_element_type=3 row_ref_node_available=1 "
            f"row_ref_x={row_ref[0]} row_ref_y={row_ref[1]} row_ref_z={row_ref[2]} "
            f"sampled_col_ref_x={'|'.join(str(ref[0]) for ref in col_refs)} "
            f"sampled_col_ref_y={'|'.join(str(ref[1]) for ref in col_refs)} "
            f"sampled_col_ref_z={'|'.join(str(ref[2]) for ref in col_refs)} "
            f"sampled_ref_edge_lengths={'|'.join(str(length) for length in lengths)}"
        )
    return line


def test_column_geometry_selectivity_builds_reference_edge_profiles(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _entry(
                    op,
                    10,
                    100,
                    [100, 101],
                    [2.0, -2.0],
                    row_ref=(0.0, 0.0, 0.0),
                    col_refs=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                ),
                _entry(
                    op,
                    1,
                    100,
                    [100, 101, 102],
                    [4.0, -2.0, -2.0],
                    row_ref=(0.0, 0.0, 0.0),
                    col_refs=[
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                    ],
                ),
                _entry(
                    op,
                    2,
                    101,
                    [100, 101, 102],
                    [-1.0, 2.0, -1.0],
                    row_ref=(1.0, 0.0, 0.0),
                    col_refs=[
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                    ],
                ),
                _entry(
                    op,
                    3,
                    102,
                    [100, 101, 102],
                    [-1.0, -1.0, 2.0],
                    row_ref=(0.0, 1.0, 0.0),
                    col_refs=[
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                    ],
                ),
            ]
        ),
        encoding="utf-8",
    )
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "path": str(log),
                "preferred_candidate_global_dofs": [100, 101, 102],
            }
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
        max_target_ratio=2.0,
    )

    case = report["cases"][0]
    target_profile = case["profile_summary"]["target_profiles"]["100"]
    assert case["log_evidence"]["batch_count"] == 2
    assert case["log_evidence"]["geometry_field_entry_count"] == 3
    assert target_profile["finite_geometry_edge_sample_count"] == 2
    assert target_profile["axis_aligned_edge_fraction"] == 1.0
    assert target_profile["diagonal_edge_fraction"] == 0.0
    assert target_profile["mean_ref_edge_length"] == 1.0
    assert target_profile["reference_geometry_class"] == "axis_only_reference_edges"
    assert report["selectors"]


def test_column_geometry_selectivity_flags_missing_logs():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "cases": [
                {
                    "label": "test02",
                    "path": "missing.log",
                    "preferred_candidate_global_dofs": [100, 101],
                }
            ]
        },
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "direct_pspg_cut_volume_column_geometry_selectivity_evidence_missing"
    )
    assert report["missing_case_labels"] == ["test02"]


def test_column_geometry_selectivity_flags_missing_geometry_fields(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        _entry(
            op,
            1,
            100,
            [100, 101],
            [2.0, -2.0],
            include_geometry=False,
        ),
        encoding="utf-8",
    )
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "path": str(log),
                "preferred_candidate_global_dofs": [100],
            }
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
    )

    assert report["finding"] == (
        "direct_pspg_cut_volume_column_geometry_selectivity_evidence_missing"
    )
    assert report["cases"][0]["log_evidence"]["status"] == (
        "reference_geometry_fields_missing"
    )


def test_column_geometry_selectivity_prefers_column_geometry_sibling_log(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    row_log = tmp_path / "run_direct_pspg_cut_volume_row_provenance.log"
    geometry_log = tmp_path / "run_direct_pspg_cut_volume_column_geometry.log"
    row_log.write_text("old row provenance log\n", encoding="utf-8")
    geometry_log.write_text(
        _entry(
            op,
            1,
            100,
            [100, 101],
            [2.0, -2.0],
            row_ref=(0.0, 0.0, 0.0),
            col_refs=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        ),
        encoding="utf-8",
    )
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "path": str(row_log),
                "preferred_candidate_global_dofs": [100],
            }
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
    )

    assert report["finding"] != (
        "direct_pspg_cut_volume_column_geometry_selectivity_evidence_missing"
    )
    assert report["cases"][0]["log_evidence"]["path"] == str(geometry_log)
    assert report["cases"][0]["profile_summary"]["profiled_target_count"] == 1
