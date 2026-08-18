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
        / "audit_direct_pspg_cut_volume_quadrature_geometry_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_quadrature_geometry_selectivity",
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


def _entry(
    op,
    rule_index,
    row_dof,
    col_dofs,
    values,
    *,
    parent_cell=3,
    include_qpoint_geometry=True,
    centroid=(0.25, 0.25, 0.25),
    span=(0.8, 0.8, 0.8),
    level_set_max_abs=0.0,
    gradient_norm_max=0.0,
):
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
    if include_qpoint_geometry:
        line += (
            " row_ref_x=0 row_ref_y=0 row_ref_z=0 "
            "cut_rule_kind=0 cut_rule_frame=0 cut_rule_construction=4 "
            "cut_rule_exact_polynomial_order=2 "
            "cut_rule_requested_quadrature_order=2 "
            "cut_rule_achieved_quadrature_order=2 "
            "cut_qpoint_count=4 cut_qpoint_weight_sum=1 "
            "cut_qpoint_abs_weight_sum=1 "
            "cut_qpoint_max_abs_weight_fraction=0.25 "
            f"cut_qpoint_centroid_x={centroid[0]} "
            f"cut_qpoint_centroid_y={centroid[1]} "
            f"cut_qpoint_centroid_z={centroid[2]} "
            f"cut_qpoint_span_x={span[0]} "
            f"cut_qpoint_span_y={span[1]} "
            f"cut_qpoint_span_z={span[2]} "
            "cut_qpoint_rms_radius=0.4 cut_qpoint_max_radius=0.6 "
            f"cut_qpoint_parent_centroid_x={centroid[0]} "
            f"cut_qpoint_parent_centroid_y={centroid[1]} "
            f"cut_qpoint_parent_centroid_z={centroid[2]} "
            f"cut_qpoint_parent_span_x={span[0]} "
            f"cut_qpoint_parent_span_y={span[1]} "
            f"cut_qpoint_parent_span_z={span[2]} "
            "cut_qpoint_normal_mean_x=1 cut_qpoint_normal_mean_y=0 "
            "cut_qpoint_normal_mean_z=0 "
            f"cut_qpoint_level_set_min={-level_set_max_abs} "
            f"cut_qpoint_level_set_max={level_set_max_abs} "
            f"cut_qpoint_level_set_mean_abs={level_set_max_abs / 2} "
            f"cut_qpoint_level_set_max_abs={level_set_max_abs} "
            f"cut_qpoint_gradient_norm_mean={gradient_norm_max / 2} "
            f"cut_qpoint_gradient_norm_max={gradient_norm_max} "
            "cut_qpoint_reference_measure_factor_sum=1"
        )
    return line


def test_quadrature_geometry_selectivity_builds_qpoint_profiles(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _entry(op, 10, 100, [100, 101], [2.0, -2.0]),
                _entry(op, 1, 100, [100, 101], [2.0, -2.0]),
                _entry(
                    op,
                    2,
                    101,
                    [100, 101],
                    [-1.0, 1.0],
                    level_set_max_abs=0.2,
                    gradient_norm_max=3.0,
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
                "preferred_candidate_global_dofs": [100, 101],
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
    assert case["log_evidence"]["cut_qpoint_field_entry_count"] == 2
    assert target_profile["cut_qpoint_field_fraction"] == 1.0
    assert target_profile["cut_qpoint_geometry_class"] == (
        "uniform_full_cell_qpoint_geometry"
    )
    assert target_profile["cut_qpoint_max_abs_weight_fraction_max"] == 0.25
    assert target_profile["row_to_cut_qpoint_centroid_distance_mean"] > 0.0
    assert report["selectors"]


def test_quadrature_geometry_selectivity_flags_missing_logs():
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
        "direct_pspg_cut_volume_quadrature_geometry_selectivity_evidence_missing"
    )
    assert report["missing_case_labels"] == ["test02"]


def test_quadrature_geometry_selectivity_flags_missing_qpoint_fields(tmp_path):
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
            include_qpoint_geometry=False,
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
        "direct_pspg_cut_volume_quadrature_geometry_selectivity_evidence_missing"
    )
    assert report["cases"][0]["log_evidence"]["status"] == (
        "cut_qpoint_geometry_fields_missing"
    )


def test_quadrature_geometry_selectivity_prefers_sibling_log(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    row_log = tmp_path / "run_direct_pspg_cut_volume_row_provenance.log"
    qgeom_log = tmp_path / "run_direct_pspg_cut_volume_quadrature_geometry.log"
    row_log.write_text("old row provenance log\n", encoding="utf-8")
    qgeom_log.write_text(
        _entry(op, 1, 100, [100, 101], [2.0, -2.0]),
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
        "direct_pspg_cut_volume_quadrature_geometry_selectivity_evidence_missing"
    )
    assert report["cases"][0]["log_evidence"]["path"] == str(qgeom_log)
    assert report["cases"][0]["profile_summary"]["profiled_target_count"] == 1
