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
        / "audit_pressure_contribution_samples.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_contribution_samples", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pressure_contribution_sample_parser_groups_line_search_rows(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual rank=0 iteration=0 "
                "phase='line_search' sync_point=line_search_trial "
                "field='Pressure' norm=1e-8 global_max_abs=2e-8 "
                "local_worst_dof=12 local_worst_value=-2e-8 "
                "sampled_dofs=10:0|12:-2e-8",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual rank=0 iteration=0 "
                "phase='pressure_row_contribution_post_constraints:"
                "equations_diagnostic_ns_galerkin_continuity' "
                "sync_point=line_search_trial field='Pressure' norm=3 "
                "global_max_abs=4 local_worst_dof=10 local_worst_value=-4 "
                "sampled_dofs=10:-4|12:1",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual rank=0 iteration=0 "
                "phase='pressure_row_contribution_post_constraints:"
                "equations_diagnostic_ns_vms_pspg' "
                "sync_point=line_search_trial field='Pressure' norm=3 "
                "global_max_abs=4 local_worst_dof=10 local_worst_value=4 "
                "sampled_dofs=10:4|12:-1",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual rank=0 iteration=0 "
                "phase='pressure_row_contribution_post_constraints:"
                "equations_diagnostic_ns_vms_pspg_pressure_gradient' "
                "sync_point=line_search_trial field='Pressure' norm=2 "
                "global_max_abs=3 local_worst_dof=10 local_worst_value=3 "
                "sampled_dofs=10:3|12:-0.75",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual rank=0 iteration=0 "
                "phase='pressure_row_contribution_post_constraints:"
                "equations_diagnostic_ns_vms_pspg_nonpressure' "
                "sync_point=line_search_trial field='Pressure' norm=1 "
                "global_max_abs=1 local_worst_dof=10 local_worst_value=1 "
                "sampled_dofs=10:1|12:-0.25",
                "NewtonSolver: field residual diagnostic "
                "diagnostic=newton_field_residual rank=0 iteration=0 "
                "phase='pressure_row_contribution_post_constraints:"
                "equations_diagnostic_ns_free_surface_pressure_reference_probe' "
                "sync_point=line_search_trial field='Pressure' norm=0 "
                "global_max_abs=0 local_worst_dof=10 local_worst_value=0 "
                "sampled_dofs=10:0|12:0",
                "Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard "
                "local_worst_dof=10 global_abs_pressure_delta_pa=1000",
            ]
        ),
        encoding="utf-8",
    )

    report = audit.audit_pressure_contribution_samples(log)

    assert report["sampled_dofs"] == ["10", "12"]
    assert report["line_search_samples_by_dof"]["10"] == {
        "equations_diagnostic_ns_galerkin_continuity": -4.0,
        "equations_diagnostic_ns_vms_pspg": 4.0,
        "equations_diagnostic_ns_vms_pspg_pressure_gradient": 3.0,
        "equations_diagnostic_ns_vms_pspg_nonpressure": 1.0,
        "equations_diagnostic_ns_free_surface_pressure_reference_probe": 0.0,
        "total_residual": 0.0,
    }
    dof10 = report["line_search_sample_classification_by_dof"]["10"]
    assert dof10["primary_pressure_path"] == "direct_pspg_pressure_gradient"
    assert dof10["residual_cancellation_class"] == "galerkin_vms_cancelled"
    assert dof10["pressure_ghost_penalty_is_roundoff"]
    assert dof10["free_surface_pressure_probe_is_roundoff"]
    assert dof10["has_direct_pspg_pressure_gradient_sample"]
    assert dof10["galerkin_plus_vms_pspg_value"] == 0.0
    assert dof10["galerkin_vms_cancellation_ratio"] == 0.0
    assert dof10["direct_pspg_pressure_gradient_share_of_vms"] == 0.75

    dof12 = report["line_search_sample_classification_by_dof"]["12"]
    assert dof12["primary_pressure_path"] == "direct_pspg_pressure_gradient"
    assert dof12["dominant_operator"] == {
        "operator": "equations_diagnostic_ns_galerkin_continuity",
        "value": 1.0,
        "abs_value": 1.0,
    }
    assert report["accepted_pressure_updates"][0]["local_worst_dof"] == 10
