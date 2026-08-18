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
        / "audit_pressure_update_support_correlation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_update_support_correlation",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pressure_update_support_correlation_matches_vertex_rows():
    audit = _load_audit_module()
    update_report = {
        "current_result": "result_001.vtu",
        "transitions": [
            {
                "top_pressure_updates": [
                    {
                        "point_index": 3,
                        "point_m": [0.0, 0.0, 0.0],
                        "abs_pressure_delta_pa": 100.0,
                        "pressure_delta_pa": 100.0,
                        "support_class": "full_wet_supported",
                        "incident_wet_fraction_max": 1.0,
                        "incident_wet_fraction_min_positive": 1.0,
                    },
                    {
                        "point_index": 4,
                        "point_m": [1.0, 0.5, 0.0],
                        "abs_pressure_delta_pa": 75.0,
                        "pressure_delta_pa": -75.0,
                        "support_class": "cut_supported",
                    },
                    {
                        "point_index": 5,
                        "point_m": [0.0, 1.0, 1.0],
                        "abs_pressure_delta_pa": 25.0,
                        "pressure_delta_pa": 25.0,
                        "support_class": "full_wet_supported",
                    },
                ]
            }
        ],
    }
    matrix_report = {
        "solver_log": "run.log",
        "latest_support_rank_diagnostic": {
            "values": {
                "weakest_self_row_details": (
                    "80:3356:row=1.1:row_coupling=3e-4:"
                    "row_self=1e-8:col=2:col_coupling=6e-4:"
                    "col_self=2e-8:diag=0.5"
                ),
                "weakest_coupling_row_details": (
                    "81:3357:row=2:row_coupling=0:"
                    "row_self=2e-8:col=3:col_coupling=0:"
                    "col_self=3e-8:diag=1"
                ),
            }
        },
        "sampled_pressure_rows": [
            {
                "local_pressure_row": 80,
                "matrix_sample": {"dof": 3356, "row_abs_sum": 1.1, "diag": 0.5},
                "constraint_sample": {
                    "entity_kind": "Vertex",
                    "entity_id": 3,
                    "active_dof_support": 1,
                    "inactive_constraint": 0,
                    "retained_measure": 1.0,
                    "retained_rule_count": 8,
                },
                "row_field_abs_sum_by_field": {
                    "Velocity": 3.0e-4,
                    "Pressure": 1.0e-8,
                },
                "row_constrained_field_abs_sum_by_field": {
                    "Velocity": 1.0e-4,
                    "Pressure": 2.0e-9,
                },
                "row_unconstrained_field_abs_sum_by_field": {
                    "Velocity": 2.0e-4,
                    "Pressure": 8.0e-9,
                },
                "col_field_abs_sum_by_field": {
                    "Velocity": 6.0e-4,
                    "Pressure": 2.0e-8,
                },
                "col_constrained_field_abs_sum_by_field": {
                    "Velocity": 2.0e-4,
                    "Pressure": 3.0e-9,
                },
                "col_unconstrained_field_abs_sum_by_field": {
                    "Velocity": 4.0e-4,
                    "Pressure": 1.7e-8,
                },
            },
            {
                "local_pressure_row": 81,
                "matrix_sample": {"dof": 3357, "row_abs_sum": 2.0, "diag": 1.0},
                "constraint_sample": {
                    "entity_kind": "Vertex",
                    "entity_id": 4,
                    "active_dof_support": 1,
                    "inactive_constraint": 0,
                },
                "row_field_abs_sum_by_field": {
                    "Velocity": 0.0,
                    "Pressure": 2.0e-8,
                },
                "row_constrained_field_abs_sum_by_field": {
                    "Velocity": 0.0,
                    "Pressure": 0.0,
                },
                "row_unconstrained_field_abs_sum_by_field": {
                    "Velocity": 0.0,
                    "Pressure": 2.0e-8,
                },
                "col_field_abs_sum_by_field": {
                    "Velocity": 0.0,
                    "Pressure": 3.0e-8,
                },
                "col_constrained_field_abs_sum_by_field": {
                    "Velocity": 0.0,
                    "Pressure": 0.0,
                },
                "col_unconstrained_field_abs_sum_by_field": {
                    "Velocity": 0.0,
                    "Pressure": 3.0e-8,
                },
            },
        ],
    }

    report = audit.correlate_pressure_updates_with_support(
        pressure_update_report=update_report,
        matrix_support_report=matrix_report,
        weak_velocity_row_sum=3.1e-4,
        weak_pressure_row_sum=1.1e-8,
    )

    assert report["top_update_count"] == 3
    assert report["matched_update_count"] == 2
    assert report["unmatched_update_count"] == 1
    assert report["coupling_class_counts"] == {
        "unmatched": 1,
        "weak_velocity_coupling": 1,
        "zero_velocity_coupling": 1,
    }
    assert report["max_abs_delta_by_coupling_class"] == {
        "unmatched": 25.0,
        "weak_velocity_coupling": 100.0,
        "zero_velocity_coupling": 75.0,
    }
    assert report["pressure_self_class_counts"] == {
        "unmatched": 1,
        "weak_pressure_self": 1,
        "positive_pressure_self": 1,
    }
    assert report["max_abs_delta_by_pressure_self_class"] == {
        "unmatched": 25.0,
        "positive_pressure_self": 75.0,
        "weak_pressure_self": 100.0,
    }
    assert report["bounds_source"] == "top_pressure_updates"
    assert report["boundary_class_counts"] == {
        "boundary_corner": 2,
        "boundary_edge": 1,
    }
    assert report["boundary_label_counts"] == {
        "x_min": 2,
        "x_max": 1,
        "y_min": 1,
        "y_max": 1,
        "z_min": 2,
        "z_max": 1,
    }
    first = report["top_updates"][0]
    assert first["local_pressure_row"] == 80
    assert first["global_dof"] == 3356
    assert first["coupling_class"] == "weak_velocity_coupling"
    assert first["pressure_self_class"] == "weak_pressure_self"
    assert first["boundary_class"] == "boundary_corner"
    assert first["boundary_labels"] == ["x_min", "y_min", "z_min"]
    assert first["row_velocity_abs_sum"] == 3.0e-4
    assert first["row_pressure_abs_sum"] == 1.0e-8
    assert first["pressure_to_velocity_row_abs_sum_ratio"] == 1.0e-8 / 3.0e-4
    assert first["row_constrained_velocity_abs_sum"] == 1.0e-4
    assert first["row_unconstrained_velocity_abs_sum"] == 2.0e-4
    assert first["row_constrained_pressure_abs_sum"] == 2.0e-9
    assert first["row_unconstrained_pressure_abs_sum"] == 8.0e-9
    assert first["retained_measure"] == 1.0
    assert report["top_updates"][1]["coupling_class"] == "zero_velocity_coupling"
    assert report["top_updates"][1]["pressure_self_class"] == "positive_pressure_self"
    assert report["top_updates"][2]["coupling_class"] == "unmatched"
    assert report["top_updates"][2]["pressure_self_class"] == "unmatched"
    assert report["pressure_self_class_by_boundary_class_counts"] == {
        "boundary_corner:unmatched": 1,
        "boundary_corner:weak_pressure_self": 1,
        "boundary_edge:positive_pressure_self": 1,
    }
    assert report["coupling_class_by_pressure_self_class_counts"] == {
        "unmatched:unmatched": 1,
        "weak_velocity_coupling:weak_pressure_self": 1,
        "zero_velocity_coupling:positive_pressure_self": 1,
    }
    assert report["weak_pressure_self_support_split_summary"] == {
        "row_count": 1,
        "boundary_class_counts": {"boundary_corner": 1},
        "boundary_label_counts": {"x_min": 1, "y_min": 1, "z_min": 1},
        "row_constrained_velocity_positive_count": 1,
        "row_unconstrained_velocity_positive_count": 1,
        "row_constrained_pressure_positive_count": 1,
        "row_unconstrained_pressure_positive_count": 1,
        "col_constrained_velocity_positive_count": 1,
        "col_unconstrained_velocity_positive_count": 1,
    }
    assert report["support_rank_weakest_self_boundary_class_counts"] == {
        "boundary_corner": 1,
    }
    assert report["support_rank_weakest_self_support_split_summary"] == {
        "row_count": 1,
        "boundary_class_counts": {"boundary_corner": 1},
        "boundary_label_counts": {"x_min": 1, "y_min": 1, "z_min": 1},
        "row_constrained_velocity_positive_count": 1,
        "row_unconstrained_velocity_positive_count": 1,
        "row_constrained_pressure_positive_count": 1,
        "row_unconstrained_pressure_positive_count": 1,
        "col_constrained_velocity_positive_count": 1,
        "col_unconstrained_velocity_positive_count": 1,
    }
    weakest_self = report["support_rank_weakest_self_rows"][0]
    assert weakest_self["local_pressure_row"] == 80
    assert weakest_self["global_dof"] == 3356
    assert weakest_self["row_self"] == 1.0e-8
    assert weakest_self["point_index"] == 3
    assert weakest_self["boundary_class"] == "boundary_corner"
    assert weakest_self["boundary_labels"] == ["x_min", "y_min", "z_min"]
    assert weakest_self["active_dof_support"] == 1
    assert weakest_self["row_constrained_velocity_abs_sum"] == 1.0e-4
    assert weakest_self["row_unconstrained_velocity_abs_sum"] == 2.0e-4
    assert report["support_rank_weakest_coupling_boundary_class_counts"] == {
        "boundary_edge": 1,
    }
    assert report["support_rank_weakest_coupling_support_split_summary"] == {
        "row_count": 1,
        "boundary_class_counts": {"boundary_edge": 1},
        "boundary_label_counts": {"x_max": 1, "z_min": 1},
        "row_constrained_velocity_positive_count": 0,
        "row_unconstrained_velocity_positive_count": 0,
        "row_constrained_pressure_positive_count": 0,
        "row_unconstrained_pressure_positive_count": 1,
        "col_constrained_velocity_positive_count": 0,
        "col_unconstrained_velocity_positive_count": 0,
    }
