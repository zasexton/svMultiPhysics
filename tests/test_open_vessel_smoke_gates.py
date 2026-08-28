import argparse
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def _load_smoke_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "run_test05_velocity_growth_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_test05_velocity_growth_smoke", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_validation_mesh_generator():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "generate_validation_meshes.py"
    )
    name = "generate_open_vessel_validation_meshes_for_tests"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_bed_pressure_is_signed_vertical_continuation_independent_of_gate_side():
    generator = _load_validation_mesh_generator()
    gate_x = 1.2
    height = 0.55
    points = generator.np.asarray([
        [gate_x - 0.1, height - 0.02, 0.0],
        [gate_x + 0.1, height - 0.02, 0.0],
        [gate_x - 0.1, height, 0.0],
        [gate_x + 0.1, height + 0.02, 0.0],
    ])

    pressure = generator.dam_break_dry_bed_column_pressure(
        points, gate_x=gate_x, dam_height=height)
    scale = generator.WATER_DENSITY * generator.GRAVITY

    assert generator.np.allclose(pressure, scale * (height - points[:, 1]))
    assert pressure[0] == pressure[1]
    assert pressure[2] == 0.0
    assert pressure[3] < 0.0


def test_wet_bed_pressure_uses_each_local_surface_with_negative_cut_support_values():
    generator = _load_validation_mesh_generator()
    gate_x = 0.6
    dam_height = 0.30
    wet_depth = 0.038
    points = generator.np.asarray([
        [gate_x - 0.01, dam_height - 0.01, 0.0],
        [gate_x - 0.01, dam_height + 0.01, 0.0],
        [gate_x + 0.01, wet_depth - 0.01, 0.0],
        [gate_x + 0.01, wet_depth + 0.01, 0.0],
    ])

    pressure = generator.dam_break_wet_bed_pressure(
        points,
        gate_x=gate_x,
        dam_height=dam_height,
        wet_depth=wet_depth,
    )
    scale = generator.WATER_DENSITY * generator.GRAVITY
    expected = scale * generator.np.asarray([0.01, -0.01, 0.01, -0.01])

    assert generator.np.allclose(pressure, expected)
    assert pressure[1] < 0.0
    assert pressure[3] < 0.0


def test_validation_solver_generator_preserves_mesh_ghost_layers():
    generator = _load_validation_mesh_generator()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir)
        generator.write_solver_xml(
            case_dir,
            mesh_path="mesh/background/mesh-complete.mesh.vtu",
            faces=["wall_top"],
            fitted=False,
            fill_height=0.1,
            time_step=0.001,
            time_steps=1,
            ghost_layers=3,
            level_set_outflow_faces=["wall_top"],
        )

        root = ET.parse(case_dir / "solver.xml").getroot()
        mesh = root.find("Add_mesh")
        assert mesh is not None
        assert mesh.findtext("Ghost_layers") == "3"
        level_set = next(
            equation
            for equation in root.findall("Add_equation")
            if equation.get("type") == "level_set"
        )
        assert level_set.findtext("Bound_preserving_bound_tolerance") == "1.0e-6"
        assert level_set.findtext("Bound_preserving_sign_tolerance") == "1.0e-12"
        top = next(
            bc
            for bc in level_set.findall("Add_BC")
            if bc.get("name") == "wall_top"
        )
        assert top.findtext("Type") == "LevelSetOutflow"


@pytest.mark.parametrize("case_name", ["spheric_test05_wet_bed_d18",
                                        "spheric_test05_wet_bed_d38"])
def test_test05_decks_declare_independent_bound_and_sign_tolerances(case_name):
    smoke = _load_smoke_module()
    solver_xml = (
        smoke.ROOT
        / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
        / case_name
        / "solver.xml"
    )
    level_set = smoke.level_set_equation(ET.parse(solver_xml).getroot())

    assert level_set.findtext("Bound_preserving_bound_tolerance") == "1.0e-6"
    assert level_set.findtext("Bound_preserving_sign_tolerance") == "1.0e-12"


def _profile_args():
    defaults = {
        "high_order_3d_benchmark_profile_qualification": True,
        "high_order_mpi_motion_smoke": False,
        "high_order_curved_3d_simplex_smoke": False,
        "case": None,
        "steps": None,
        "timeout_seconds": None,
        "max_solver_elapsed_seconds_per_accepted_step": None,
        "fsils_matrix_diagnostics_every_n": None,
        "fsils_matrix_diagnostics_max_records": None,
        "implicit_cut_quadrature_backend": None,
        "expect_selected_implicit_cut_quadrature_backend": None,
        "linear_algebra_backend": None,
        "linear_preconditioner": None,
        "linear_solver_type": None,
        "linear_relative_tolerance": None,
        "linear_absolute_tolerance": None,
        "ns_gm_max_iterations": None,
        "ns_cg_max_iterations": None,
        "ns_gm_tolerance": None,
        "ns_cg_tolerance": None,
        "adaptive_time_loop_min_dt": None,
        "adaptive_time_loop_max_dt": None,
        "adaptive_time_loop_max_retries": None,
        "adaptive_time_loop_decrease_factor": None,
        "adaptive_time_loop_increase_factor": None,
        "adaptive_time_loop_target_newton_iterations": None,
        "adaptive_time_loop_max_steps_multiplier": None,
        "newton_line_search_max_iterations": None,
        "max_fsils_matrix_missing_diag": None,
        "max_fsils_matrix_duplicate_diag_entries": None,
        "max_fsils_matrix_duplicate_diag_rows": None,
        "max_fsils_matrix_nonfinite_entries": None,
        "max_diagnostic_implicit_cut_fallback_cells": None,
        "min_diagnostic_achieved_interface_quadrature_order": None,
        "min_diagnostic_achieved_volume_quadrature_order": None,
        "max_diagnostic_assembly_timings_per_step": None,
        "max_diagnostic_extra_assembly_timings_per_step": None,
        "max_diagnostic_cut_context_rebuilds_per_step": None,
        "max_diagnostic_generated_cell_cache_full_miss_rebuilds": None,
        "max_diagnostic_process_rss_kb": None,
        "max_diagnostic_process_rss_growth_kb": None,
        "max_diagnostic_process_basis_cache_entries": None,
        "max_diagnostic_process_basis_cache_entry_growth": None,
        "curvature_projection_max_zero_fallback_vertices": None,
        "max_time_loop_nonlinear_iterations_per_step": None,
        "max_time_loop_linear_iterations_per_step": None,
        "min_reference_profile_coverage": None,
        "min_reference_profile_direct_coverage": None,
        "max_reference_profile_rmse": None,
        "max_reference_profile_mae": None,
        "max_reference_profile_max_abs_error": None,
        "reference_profile_elevated_front_clearance": None,
        "max_reference_profile_elevated_front_lag": None,
    }
    return argparse.Namespace(**defaults)


def test_profile_qualification_defaults_include_memory_budgets():
    smoke = _load_smoke_module()
    args = _profile_args()

    smoke.apply_high_order_3d_benchmark_profile_qualification_defaults(args)

    assert args.require_process_memory_diagnostics is True
    assert args.max_diagnostic_process_rss_kb == 1_000_000.0
    assert args.max_diagnostic_process_rss_growth_kb == 650_000.0
    assert args.max_diagnostic_process_basis_cache_entries == 32
    assert args.max_diagnostic_process_basis_cache_entry_growth == 32
    assert args.max_fsils_matrix_duplicate_diag_entries == 0
    assert args.max_fsils_matrix_duplicate_diag_rows == 0
    assert args.adaptive_time_loop_min_dt == 1.5625e-5
    assert args.adaptive_time_loop_max_steps_multiplier == 64
    assert args.linear_relative_tolerance == 1.0e-10
    assert args.linear_absolute_tolerance == 1.0e-12


def _short_benchmark_args(*, smoke: bool, qualification: bool):
    return argparse.Namespace(
        high_order_3d_benchmark_smoke=smoke,
        high_order_3d_benchmark_qualification=qualification,
        high_order_3d_benchmark_profile_qualification=False,
        high_order_production_qualification=False,
        high_order_mpi_production_qualification=False,
        high_order_mpi_motion_smoke=False,
        high_order_curved_3d_simplex_smoke=False,
        case=None,
        steps=None,
        timeout_seconds=None,
        max_solver_elapsed_seconds_per_accepted_step=None,
        linear_algebra_backend=None,
        linear_preconditioner=None,
        linear_solver_type=None,
        ns_gm_max_iterations=None,
        ns_cg_max_iterations=None,
        ns_gm_tolerance=None,
        ns_cg_tolerance=None,
    )


@pytest.mark.parametrize(
    ("smoke_enabled", "qualification_enabled", "apply_defaults"),
    [
        (True, False, "apply_high_order_3d_benchmark_smoke_defaults"),
        (False, True, "apply_high_order_3d_benchmark_qualification_defaults"),
    ],
)
def test_short_benchmark_fsils_presets_select_supported_ns_method(
        smoke_enabled, qualification_enabled, apply_defaults):
    smoke = _load_smoke_module()
    args = _short_benchmark_args(
        smoke=smoke_enabled,
        qualification=qualification_enabled,
    )

    getattr(smoke, apply_defaults)(args)

    assert args.linear_algebra_backend == "fsils"
    assert args.linear_preconditioner == "fsils"
    assert args.linear_solver_type == "ns"
    assert args.ns_gm_max_iterations == 200
    assert args.ns_cg_max_iterations == 200
    assert args.ns_gm_tolerance == 1.0e-4
    assert args.ns_cg_tolerance == 1.0e-4


def test_solver_configuration_rejects_direct_method_with_fsils_backend():
    smoke = _load_smoke_module()
    source = (
        smoke.ROOT
        / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
        / "spheric_test05_wet_bed_d18/solver.xml"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        solver_xml = Path(temp_dir) / "solver.xml"
        shutil.copy2(source, solver_xml)

        with pytest.raises(ValueError, match="FSILS does not support.*Direct"):
            smoke.configure_solver(
                solver_xml,
                steps=1,
                linear_solver_type="direct",
                linear_algebra_backend="fsils",
                linear_preconditioner="fsils",
            )


def test_linear_solver_control_parser_accepts_krylov_space_dimension():
    smoke = _load_smoke_module()
    parser = argparse.ArgumentParser()
    smoke.add_linear_solver_control_arguments(parser)

    args = parser.parse_args([
        "--linear-solver-type", "gmres",
        "--linear-krylov-space-dimension", "37",
    ])

    assert args.linear_solver_type == "gmres"
    assert args.linear_krylov_space_dimension == 37


def test_solver_configuration_rewrites_only_fluid_krylov_controls():
    smoke = _load_smoke_module()
    source = (
        smoke.ROOT
        / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
        / "spheric_test05_wet_bed_d18/solver.xml"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        solver_xml = Path(temp_dir) / "solver.xml"
        shutil.copy2(source, solver_xml)
        before = smoke.ET.parse(solver_xml).getroot()
        level_set_krylov = (
            smoke.level_set_equation(before).find("LS").findtext(
                "Krylov_space_dimension")
        )

        smoke.configure_solver(
            solver_xml,
            steps=1,
            linear_solver_type="gmres",
            linear_algebra_backend="fsils",
            linear_preconditioner="rcs",
            linear_max_iterations=100,
            linear_krylov_space_dimension=37,
            linear_relative_tolerance=1.0e-8,
            linear_absolute_tolerance=1.0e-10,
            disable_cut_metadata_scale=True,
        )

        after = smoke.ET.parse(solver_xml).getroot()
        fluid_solver = smoke.navier_stokes_linear_solver(after)
        assert fluid_solver.attrib["type"] == "gmres"
        assert fluid_solver.findtext("Max_iterations") == "100"
        assert fluid_solver.findtext("Krylov_space_dimension") == "37"
        assert fluid_solver.findtext("Tolerance") == "1e-08"
        assert fluid_solver.findtext("Absolute_tolerance") == "1e-10"
        assert (
            smoke.level_set_equation(after).find("LS").findtext(
                "Krylov_space_dimension")
            == level_set_krylov
        )


def test_solver_configuration_writes_static_capillary_initializer_controls():
    smoke = _load_smoke_module()
    source = (
        smoke.ROOT
        / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
        / "spheric_test05_wet_bed_d18/solver.xml"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        solver_xml = Path(temp_dir) / "solver.xml"
        shutil.copy2(source, solver_xml)

        smoke.configure_solver(
            solver_xml,
            steps=1,
            disable_cut_metadata_scale=True,
            enable_static_capillary_equilibrium_initialization=True,
            static_capillary_volume_tolerance=1.0e-8,
            static_capillary_projected_gradient_tolerance=2.0e-7,
            static_capillary_pressure_representability_max_residual_norm=3.0e-6,
            static_capillary_pressure_representability_max_relative_distance=4.0e-5,
            static_capillary_physical_equilibrium_max_residual_norm=5.0e-4,
            static_capillary_constant_pressure_kkt_max_residual_norm=6.0e-3,
            static_capillary_constant_pressure_kkt_max_relative_distance=7.0e-2,
            static_capillary_max_iterations=83,
        )

        level_set = smoke.level_set_equation(
            smoke.ET.parse(solver_xml).getroot())
        assert level_set.findtext(
            "Enable_static_capillary_equilibrium_initialization") == "true"
        assert level_set.findtext("Static_capillary_volume_tolerance") == "1e-08"
        assert level_set.findtext(
            "Static_capillary_projected_gradient_tolerance") == "2e-07"
        assert level_set.findtext(
            "Static_capillary_pressure_representability_max_residual_norm"
        ) == "3e-06"
        assert level_set.findtext(
            "Static_capillary_pressure_representability_max_relative_distance"
        ) == "4e-05"
        assert level_set.findtext(
            "Static_capillary_physical_equilibrium_max_residual_norm"
        ) == "0.0005"
        assert level_set.findtext(
            "Static_capillary_constant_pressure_kkt_max_residual_norm"
        ) == "0.006"
        assert float(level_set.findtext(
            "Static_capillary_constant_pressure_kkt_max_relative_distance"
        )) == 0.07
        assert level_set.findtext("Static_capillary_max_iterations") == "83"


def test_high_order_motion_gates_default_advection_velocity_diagnostics():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        high_order_production_qualification=True,
        high_order_mpi_production_qualification=False,
        high_order_visible_motion_demo=False,
        high_order_mpi_motion_smoke=False,
        high_order_volume_corrected_motion_smoke=False,
        trace_level_set_advection_velocity=False,
        require_level_set_advection_velocity_diagnostics=False,
        wet_extension_advection_velocity_method=None,
        expect_level_set_advection_velocity_extension_method=None,
        expect_level_set_advection_velocity_interface_sample_source=None,
    )

    smoke.apply_level_set_advection_velocity_diagnostic_gate_defaults(args)

    assert args.trace_level_set_advection_velocity is True
    assert args.require_level_set_advection_velocity_diagnostics is True
    assert (
        args.expect_level_set_advection_velocity_extension_method
        == "wall_compatible_normal"
    )
    assert args.expect_level_set_advection_velocity_interface_sample_source is None


def test_high_order_motion_advection_defaults_follow_requested_method():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        high_order_production_qualification=False,
        high_order_mpi_production_qualification=False,
        high_order_visible_motion_demo=True,
        high_order_mpi_motion_smoke=False,
        high_order_volume_corrected_motion_smoke=False,
        trace_level_set_advection_velocity=False,
        require_level_set_advection_velocity_diagnostics=False,
        wet_extension_advection_velocity_method="nearest_interface_point",
        expect_level_set_advection_velocity_extension_method=None,
        expect_level_set_advection_velocity_interface_sample_source=None,
    )

    smoke.apply_level_set_advection_velocity_diagnostic_gate_defaults(args)

    assert args.trace_level_set_advection_velocity is True
    assert args.require_level_set_advection_velocity_diagnostics is True
    assert (
        args.expect_level_set_advection_velocity_extension_method
        == "nearest_interface_point"
    )
    assert args.expect_level_set_advection_velocity_interface_sample_source is None


def test_resource_ceiling_errors_reject_process_memory_over_budget():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_diagnostic_process_rss_kb=100.0,
        max_diagnostic_process_rss_growth_kb=10.0,
        max_diagnostic_process_basis_cache_entry_growth=2,
    )
    metrics = {
        "diagnostic_process_max_rss_kb": 101.0,
        "diagnostic_process_rss_growth_kb": 11.0,
        "diagnostic_process_basis_cache_entry_growth": 3,
    }

    errors = smoke.resource_ceiling_errors(metrics, args)

    assert len(errors) == 3
    assert any("process RSS" in error for error in errors)
    assert any("process RSS growth" in error for error in errors)
    assert any("basis-cache entry growth" in error for error in errors)


def test_cut_context_policy_errors_reject_fallbacks_and_downgrades():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_high_order_cut_context_diagnostics=True,
        expect_generated_interface_geometry=None,
        expect_implicit_cut_quadrature_backend=None,
        expect_selected_implicit_cut_quadrature_backend=None,
        expect_implicit_cut_backend_qualification="ProductionQualified",
        expect_implicit_cut_fallback_policy="Fail",
        max_diagnostic_implicit_cut_fallback_cells=0,
        min_diagnostic_achieved_interface_quadrature_order=5,
        min_diagnostic_achieved_volume_quadrature_order=4,
    )
    metrics = {
        "diagnostics": {
            "cut_context_rebuilds": [
                {
                    "generated_interface_geometry": "HighOrder",
                    "implicit_cut_quadrature_backend": "Auto",
                    "selected_implicit_cut_quadrature_backend_counts": "Auto:1",
                    "implicit_cut_backend_seconds": 0.01,
                    "implicit_cut_backend_seconds_max": 0.01,
                    "implicit_cut_fallback_policy": "Allow",
                    "implicit_cut_fallback_cells": 1,
                    "implicit_cut_backend_qualification_counts": "Experimental:1",
                    "required_implicit_cut_backend_qualification": "ProductionQualified",
                    "achieved_interface_quadrature_order": 3,
                    "achieved_volume_quadrature_order": 2,
                    "interface_rule_count": 1,
                    "interface_quadrature_point_count": 4,
                    "active_volume_rule_count": 1,
                    "active_volume_quadrature_point_count": 4,
                }
            ]
        },
        "diagnostic_implicit_cut_backend_qualification_counts": {
            "Experimental": 1,
        },
        "diagnostic_implicit_cut_fallback_policy_counts": {
            "Allow": 1,
        },
        "diagnostic_implicit_cut_fallback_cells": 1,
        "diagnostic_achieved_interface_quadrature_order_min": 3,
        "diagnostic_achieved_volume_quadrature_order_min": 2,
    }

    errors = smoke.cut_context_policy_errors(metrics, args)

    assert len(errors) == 5
    assert any("ProductionQualified" in error for error in errors)
    assert any("fallback policy" in error and "Fail" in error for error in errors)
    assert any("fallback cells" in error for error in errors)
    assert any("interface quadrature order" in error for error in errors)
    assert any("volume quadrature order" in error for error in errors)


def test_cut_context_policy_errors_reject_missing_qualification_evidence():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_high_order_cut_context_diagnostics=False,
        expect_generated_interface_geometry=None,
        expect_implicit_cut_quadrature_backend=None,
        expect_selected_implicit_cut_quadrature_backend=None,
        expect_implicit_cut_backend_qualification="ProductionQualified",
        expect_implicit_cut_fallback_policy=None,
        max_diagnostic_implicit_cut_fallback_cells=None,
        min_diagnostic_achieved_interface_quadrature_order=None,
        min_diagnostic_achieved_volume_quadrature_order=None,
    )
    metrics = {"diagnostics": {}}

    errors = smoke.cut_context_policy_errors(metrics, args)

    assert errors == [
        "diagnostic implicit cut backend qualification counts are unavailable"
    ]


def test_curvature_projection_errors_reject_fallback_budgets():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_curvature_projection_diagnostics=False,
        require_curvature_projection_newton_freshness=False,
        min_diagnostic_curvature_projection_count=None,
        min_diagnostic_curvature_projection_max_abs_curvature=None,
        max_diagnostic_curvature_projection_fallback_vertices=0,
        max_diagnostic_curvature_projection_zero_fallback_vertices=0,
        max_diagnostic_curvature_projection_normalized_fit_residual=None,
        min_diagnostic_curvature_projection_smoothing_iterations=None,
        expect_curvature_projection_smoothing_mode=None,
        min_diagnostic_curvature_projection_operator_edges=None,
        min_diagnostic_curvature_projection_skipped_count=None,
        min_diagnostic_curvature_projection_cache_hit_count=None,
        max_diagnostic_curvature_projection_cache_miss_count=None,
        min_diagnostic_curvature_projection_cut_signature_cache_hit_count=None,
        min_diagnostic_curvature_projection_reused_vertex_adjacency_count=None,
        min_diagnostic_curvature_projection_reused_sample_adjacency_count=None,
        max_diagnostic_curvature_projection_vertex_adjacency_builds=None,
        max_diagnostic_curvature_projection_sample_adjacency_builds=None,
    )
    metrics = {
        "diagnostics": {},
        "diagnostic_curvature_projection_max_fallback_vertices": 1,
        "diagnostic_curvature_projection_max_zero_fallback_vertices": 2,
    }

    errors = smoke.curvature_projection_errors(metrics, args)

    assert len(errors) == 2
    assert any("fallback vertices" in error for error in errors)
    assert any("zero fallback vertices" in error for error in errors)


def test_curvature_projection_errors_reject_wrong_smoothing_mode():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_curvature_projection_diagnostics=False,
        require_curvature_projection_newton_freshness=False,
        min_diagnostic_curvature_projection_count=None,
        min_diagnostic_curvature_projection_max_abs_curvature=None,
        max_diagnostic_curvature_projection_fallback_vertices=None,
        max_diagnostic_curvature_projection_zero_fallback_vertices=None,
        max_diagnostic_curvature_projection_normalized_fit_residual=None,
        min_diagnostic_curvature_projection_smoothing_iterations=None,
        expect_curvature_projection_smoothing_mode="mass_stiffness_operator",
        min_diagnostic_curvature_projection_operator_edges=1,
        min_diagnostic_curvature_projection_skipped_count=None,
        min_diagnostic_curvature_projection_cache_hit_count=None,
        max_diagnostic_curvature_projection_cache_miss_count=None,
        min_diagnostic_curvature_projection_cut_signature_cache_hit_count=None,
        min_diagnostic_curvature_projection_reused_vertex_adjacency_count=None,
        min_diagnostic_curvature_projection_reused_sample_adjacency_count=None,
        max_diagnostic_curvature_projection_vertex_adjacency_builds=None,
        max_diagnostic_curvature_projection_sample_adjacency_builds=None,
    )
    metrics = {
        "diagnostics": {},
        "diagnostic_curvature_projection_smoothing_mode_counts": {
            "local_graph": 2,
        },
        "diagnostic_curvature_projection_max_smoothing_operator_edges": 0,
    }

    errors = smoke.curvature_projection_errors(metrics, args)

    assert len(errors) == 2
    assert any("smoothing mode" in error for error in errors)
    assert any("operator edges" in error for error in errors)


def test_curvature_projection_freshness_does_not_require_a_trial_per_accepted_step():
    smoke = _load_smoke_module()

    class OptionalArgs:
        def __getattr__(self, _name):
            return None

    args = OptionalArgs()
    args.require_curvature_projection_diagnostics = False
    args.require_curvature_projection_newton_freshness = True
    metrics = {
        "diagnostics": {},
        "diagnostic_curvature_projection_reason_counts": {
            "initial": 1,
            "before_physics_solve": 2,
            "jacobian_and_residual": 2,
            "accepted_step": 2,
            # Synchronized trial refreshes are conditional on the nonlinear
            # residual contract and on backtracking actually evaluating a
            # trial; they are not guaranteed once per accepted step.
            "line_search_trial": 0,
        },
        "time_loop": {"summary": {"accepted_steps": 2}},
    }

    assert smoke.curvature_projection_errors(metrics, args) == []

    metrics["diagnostic_curvature_projection_reason_counts"][
        "before_physics_solve"] = 1
    errors = smoke.curvature_projection_errors(metrics, args)
    assert errors == [
        "curvature projection reason 'before_physics_solve' count 1 is below 2"
    ]


def test_capillary_benchmark_errors_check_curvature_and_pressure_jump():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_curvature_relative_error=0.1,
        max_capillary_pressure_jump_relative_error=0.1,
    )
    metrics = {
        "benchmark": {
            "capillary_arc_radius": 0.5,
            "initial_active_pressure": 1.0,
        },
        "surface_tension": 0.5,
        "capillary_final_pressure_jump": 1.0,
        "diagnostic_curvature_projection_max_abs_curvature": 20.0,
        "latest_curvature_projection": {
            "curvature_field": "kappa_projected",
            "narrow_band_width": 0.0625,
            "narrow_band_vertices": 32,
            "skipped_far_vertices": 48,
            "max_abs_curvature": 1.0,
        },
    }

    errors = smoke.capillary_benchmark_errors(metrics, args)

    assert len(errors) == 1
    assert any("curvature relative error" in error for error in errors)
    assert metrics["capillary_expected_curvature"] == 2.0
    assert metrics["capillary_expected_pressure_jump"] == 1.0


def test_capillary_benchmark_errors_accept_generic_droplet_radius():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_curvature_relative_error=0.01,
        max_capillary_pressure_jump_relative_error=0.01,
    )
    metrics = {
        "benchmark": {
            "capillary_radius": 0.25,
            "initial_active_pressure": 2.0,
        },
        "surface_tension": 0.5,
        "capillary_final_pressure_jump": 2.0,
        "diagnostic_curvature_projection_max_abs_curvature": 40.0,
        "latest_curvature_projection": {
            "curvature_field": "kappa_projected",
            "narrow_band_width": 0.03125,
            "narrow_band_vertices": 64,
            "skipped_far_vertices": 96,
            "max_abs_curvature": 4.0,
        },
    }

    errors = smoke.capillary_benchmark_errors(metrics, args)

    assert errors == []
    assert metrics["capillary_benchmark_radius"] == 0.25
    assert metrics["capillary_expected_curvature"] == 4.0
    assert metrics["capillary_expected_pressure_jump"] == 2.0
    assert metrics["capillary_projected_curvature_interface_band_max_abs"] == 4.0
    assert metrics["capillary_projected_curvature_observed_source"] == (
        "solver_latest_curvature_projection_diagnostic"
    )
    assert metrics["capillary_projected_curvature_observed_metric"] == (
        "latest_curvature_projection.max_abs_curvature"
    )
    assert "capillary_projected_curvature_max_abs" not in metrics


def test_spatial_capillary_factor_scales_curvature_and_pressure_jump():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_curvature_relative_error=0.01,
        max_capillary_pressure_jump_relative_error=0.01,
        max_capillary_parasitic_capillary_number=0.01,
    )
    metrics = {
        "benchmark": {
            "capillary_radius": 0.25,
            "capillary_curvature_factor": 2.0,
            "initial_active_pressure": 4.0,
            "viscosity": 0.1,
        },
        "surface_tension": 0.5,
        "capillary_final_pressure_jump": 4.0,
        "spatial_capillary_final_max_liquid_speed": 0.025,
        "latest_curvature_projection": {
            "curvature_field": "kappa_projected",
            "narrow_band_width": 0.03125,
            "narrow_band_vertices": 64,
            "skipped_far_vertices": 96,
            "max_abs_curvature": 8.0,
        },
    }

    assert smoke.capillary_benchmark_errors(metrics, args) == []
    assert metrics["capillary_expected_curvature"] == 8.0
    assert metrics["capillary_expected_pressure_jump"] == 4.0
    assert metrics["capillary_final_parasitic_capillary_number"] == pytest.approx(
        0.005)


def test_capillary_pressure_jump_gate_requires_solved_final_pressure_samples():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_curvature_relative_error=None,
        max_capillary_pressure_jump_relative_error=0.1,
    )
    metrics = {
        "benchmark": {
            "capillary_radius": 0.25,
            # A configured preload alone is not production solution evidence.
            "initial_active_pressure": 2.0,
        },
        "surface_tension": 0.5,
    }

    errors = smoke.capillary_benchmark_errors(metrics, args)

    assert errors == [
        "capillary pressure-jump gate requires final liquid/gas pressure samples"
    ]


def test_capillary_benchmark_errors_prefer_near_interface_curvature_mean():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_curvature_relative_error=0.1,
        max_capillary_pressure_jump_relative_error=None,
    )
    metrics = {
        "benchmark": {
            "capillary_arc_radius": 0.8,
        },
        "diagnostic_curvature_projection_max_abs_curvature": 2.45,
        "projected_curvature_near_interface_mean_abs": 1.30,
        "projected_curvature_near_interface_band_width": 0.05,
        "projected_curvature_near_interface_point_count": 12,
        "projected_curvature_field_name": "kappa_projected",
    }

    errors = smoke.capillary_benchmark_errors(metrics, args)

    assert errors == []
    assert metrics["capillary_projected_curvature_observed_metric"] == (
        "projected_curvature_near_interface_mean_abs"
    )
    assert metrics["capillary_projected_curvature_observed_source"] == (
        "vtk_point_data_projected_curvature_field"
    )
    assert metrics["capillary_projected_curvature_interface_band_mean_abs"] == 1.30
    assert metrics["capillary_observed_curvature"] == 1.30


def test_capillary_curvature_gate_rejects_global_or_zero_band_maximum():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_curvature_relative_error=0.1,
        max_capillary_pressure_jump_relative_error=None,
    )
    metrics = {
        "benchmark": {"capillary_radius": 0.25},
        "diagnostic_curvature_projection_max_abs_curvature": 4.0,
        "latest_curvature_projection": {
            "curvature_field": "kappa_projected",
            "narrow_band_width": 0.0,
            "narrow_band_vertices": 81,
            "skipped_far_vertices": 0,
            "max_abs_curvature": 4.0,
        },
    }

    errors = smoke.capillary_benchmark_errors(metrics, args)

    assert errors == [
        "capillary curvature gate requires a projected-curvature "
        "interface-band statistic"
    ]
    assert "capillary_observed_curvature" not in metrics


def test_write_capillary_droplet2d_case_records_equilibrium_benchmark():
    smoke = _load_smoke_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"

        smoke.write_capillary_droplet2d_case(case_dir, steps=2, pressure_jump=1.25)

        benchmark = json.loads((case_dir / "benchmark.json").read_text())
        assert benchmark["capillary_geometry"] == "droplet2d"
        assert benchmark["capillary_radius"] == smoke.CAPILLARY_DROPLET_RADIUS
        assert benchmark["initial_active_pressure"] == 1.25

        grid = smoke.pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
        phi = smoke.np.asarray(grid.point_data["phi"], dtype=float)
        pressure = smoke.np.asarray(grid.point_data["Pressure"], dtype=float)

        assert phi.min() < 0.0
        assert phi.max() > 0.0
        assert smoke.np.allclose(pressure, 1.25)
        assert benchmark["initial_pressure_extension"].startswith(
            "constant gamma/R on background support")


def test_write_capillary_wave2d_case_records_reference_profile():
    smoke = _load_smoke_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "capillarywave2d"

        smoke.write_capillary_wave2d_case(
            case_dir, steps=3, surface_tension=500.0, time_step_size=0.002)

        benchmark = json.loads((case_dir / "benchmark.json").read_text())
        wave = benchmark["capillary_wave"]
        assert benchmark["capillary_geometry"] == "standing_wave_2d"
        assert wave["amplitude"] == smoke.CAPILLARY_WAVE_AMPLITUDE
        assert wave["omega"] == smoke.capillary_wave_omega(500.0)
        assert wave["final_time_s"] == 0.006
        assert (
            benchmark["boundary_contract"]["required_transport_extension"]
            == "wall_compatible_normal"
        )

        root = smoke.ET.parse(case_dir / "solver.xml").getroot()
        fluid = smoke.fluid_equation(root)
        walls = {
            bc.attrib.get("name"): bc
            for bc in fluid.findall("Add_BC")
        }
        assert walls["wall_left"].findtext("Effective_direction") == "1 0"
        assert walls["wall_right"].findtext("Effective_direction") == "1 0"
        assert walls["wall_bottom"].findtext("Effective_direction") == "0 1"

        profile_path = Path(benchmark["reference_profiles"][0]["path"])
        profile = smoke.np.loadtxt(profile_path)
        assert profile.shape[1] == 2
        assert profile_path.exists()

        grid = smoke.pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
        phi = smoke.np.asarray(grid.point_data["phi"], dtype=float)
        gids = smoke.np.asarray(grid.point_data["GlobalNodeID"], dtype=int)
        gauge_node = benchmark["pressure_gauge"]["node_id"]
        gauge_index = int(smoke.np.flatnonzero(gids == gauge_node)[0])
        assert phi.min() < 0.0
        assert phi.max() > 0.0
        assert phi[gauge_index] < 0.0


def test_capillary_wave_benchmark_errors_accept_expected_profile():
    smoke = _load_smoke_module()
    surface_tension = 500.0
    k = smoke.capillary_wave_wavenumber()
    omega = smoke.capillary_wave_omega(surface_tension)
    amplitude = smoke.CAPILLARY_WAVE_AMPLITUDE
    final_time = 0.1
    args = argparse.Namespace(
        max_capillary_wave_frequency_relative_error=1.0e-12,
        max_capillary_wave_profile_relative_error=1.0e-12,
        max_capillary_wave_mean_offset=1.0e-12,
        steps=50,
        time_step_size=0.002,
    )
    metrics = {
        "benchmark": {
            "capillary_wave": {
                "amplitude": amplitude,
                "wavenumber": k,
                "density": smoke.CAPILLARY_WAVE_DENSITY,
                "surface_tension": surface_tension,
                "omega": omega,
                "depth": smoke.CAPILLARY_WAVE_DEPTH,
            },
        },
        "steps": 50,
        "capillary_wave_observed_omega": omega,
        "capillary_wave_frequency_observed_phase_span": omega * final_time,
        "final_capillary_wave_profile_available": True,
        "final_capillary_wave_cosine_amplitude": amplitude * smoke.math.cos(
            omega * final_time),
        "final_capillary_wave_sine_amplitude": 0.0,
        "final_capillary_wave_mean_offset": 0.0,
    }

    errors = smoke.capillary_wave_benchmark_errors(metrics, args)

    assert errors == []
    assert metrics["capillary_wave_profile_relative_error"] == 0.0


def test_capillary_wave_benchmark_errors_reject_wrong_profile_phase():
    smoke = _load_smoke_module()
    surface_tension = 500.0
    k = smoke.capillary_wave_wavenumber()
    omega = smoke.capillary_wave_omega(surface_tension)
    args = argparse.Namespace(
        max_capillary_wave_frequency_relative_error=None,
        max_capillary_wave_profile_relative_error=0.1,
        max_capillary_wave_mean_offset=None,
        steps=12,
        time_step_size=0.002,
    )
    metrics = {
        "benchmark": {
            "capillary_wave": {
                "amplitude": smoke.CAPILLARY_WAVE_AMPLITUDE,
                "wavenumber": k,
                "density": smoke.CAPILLARY_WAVE_DENSITY,
                "surface_tension": surface_tension,
                "omega": omega,
                "depth": smoke.CAPILLARY_WAVE_DEPTH,
            },
        },
        "steps": 12,
        "final_capillary_wave_profile_available": True,
        "final_capillary_wave_cosine_amplitude": 0.0,
        "final_capillary_wave_sine_amplitude": 0.0,
        "final_capillary_wave_mean_offset": 0.0,
    }

    errors = smoke.capillary_wave_benchmark_errors(metrics, args)

    assert len(errors) == 1
    assert "profile relative error" in errors[0]


def test_capillary_wave_temporal_liquid_volume_gate_is_distinct_and_fail_closed():
    smoke = _load_smoke_module()

    def production_record(step, time, volume, *, skipped=0, frame="physical"):
        return {
            "step": step,
            "time": time,
            "field": "phi",
            "domain_id": "open_vessel_surface",
            "marker": 1484991,
            "active_side": "LevelSetNegative",
            "isovalue": 0.0,
            "wet_volume": volume,
            "wet_volume_frame": frame,
            "reference_wet_volume": 128.0 * volume / 0.5,
            "physical_wet_volume": volume,
            "initial_wet_volume": 0.5,
            "wet_volume_drift": volume - 0.5,
            "relative_wet_volume_drift": (volume - 0.5) / 0.5,
            "volume_rule_count": 36,
            "physical_volume_rule_count": 36 - skipped,
            "skipped_physical_volume_rule_count": skipped,
        }

    metrics = {
        "benchmark": {"capillary_wave": {}},
        "solver_controls": {"transient_solve": {"t0": 0.0}},
        "time_loop": {"accepted_steps": [
            {"step": 1, "time": 0.002, "dt": 0.002},
            {"step": 2, "time": 0.004, "dt": 0.002},
            {"step": 3, "time": 0.006, "dt": 0.002},
        ]},
        "production_wet_volume_diagnostic_history": [
            production_record(0, 0.0, 0.5),
            production_record(1, 0.002, 0.499999),
            production_record(2, 0.004, 0.499995),
            production_record(3, 0.006, 0.49999),
        ],
        # A contradictory VTK-only history proves the temporal conservation
        # metric is sourced from the production diagnostic, not saved output.
        "physical_liquid_measure_history": [
            {"step": 1, "time": 0.002,
             "corrected_state_liquid_measure": 123.0},
        ],
    }
    smoke.add_capillary_wave_temporal_liquid_volume_metrics(metrics)
    args = argparse.Namespace(
        max_capillary_wave_frequency_relative_error=None,
        max_capillary_wave_profile_relative_error=None,
        max_capillary_wave_mean_offset=None,
        max_capillary_wave_temporal_liquid_volume_relative_drift=1.0e-5,
    )

    errors = smoke.capillary_wave_benchmark_errors(metrics, args)

    assert metrics["capillary_wave_temporal_liquid_volume_available"] is True
    assert metrics["capillary_wave_temporal_liquid_volume_source"] == (
        "production_physical_cut_context_diagnostic"
    )
    assert metrics["capillary_wave_temporal_liquid_volume_reference_step"] == 0
    assert metrics["capillary_wave_temporal_liquid_volume_state_count"] == 4
    assert metrics["capillary_wave_temporal_liquid_volume_reference"] == 0.5
    assert metrics["capillary_wave_temporal_liquid_volume_final"] == 0.49999
    assert abs(
        metrics["capillary_wave_temporal_liquid_volume_max_relative_drift"]
        - 2.0e-5
    ) < 1.0e-14
    assert errors == [
        "capillary-wave temporal liquid-volume relative drift 2e-05 exceeds 1e-05"
    ]

    missing = {"benchmark": {"capillary_wave": {}}}
    missing_errors = smoke.capillary_wave_benchmark_errors(missing, args)
    assert missing_errors == [
        "capillary-wave temporal liquid-volume drift diagnostic is unavailable"
    ]

    skipped = dict(metrics)
    skipped["production_wet_volume_diagnostic_history"] = [
        production_record(0, 0.0, 0.5),
        production_record(1, 0.002, 0.499999),
        production_record(2, 0.004, 0.499995, skipped=1),
        production_record(3, 0.006, 0.49999),
    ]
    smoke.add_capillary_wave_temporal_liquid_volume_metrics(skipped)
    assert skipped["capillary_wave_temporal_liquid_volume_available"] is False
    assert "complete physical rule set" in skipped[
        "capillary_wave_temporal_liquid_volume_error"]

    missing_accepted = dict(metrics)
    missing_accepted["production_wet_volume_diagnostic_history"] = [
        production_record(0, 0.0, 0.5),
        production_record(1, 0.002, 0.499999),
        production_record(3, 0.006, 0.49999),
    ]
    smoke.add_capillary_wave_temporal_liquid_volume_metrics(missing_accepted)
    assert missing_accepted[
        "capillary_wave_temporal_liquid_volume_available"] is False
    assert "initial and every accepted state" in missing_accepted[
        "capillary_wave_temporal_liquid_volume_error"]

    restarted = dict(metrics)
    restarted["solver_controls"] = {"transient_solve": {"t0": 1.0}}
    restarted["time_loop"] = {"accepted_steps": [
        {"step": 7, "time": 1.002, "dt": 0.002},
        {"step": 8, "time": 1.004, "dt": 0.002},
    ]}
    restarted["production_wet_volume_diagnostic_history"] = [
        production_record(6, 1.0, 0.5),
        production_record(7, 1.002, 0.499999),
        production_record(8, 1.004, 0.499995),
    ]
    smoke.add_capillary_wave_temporal_liquid_volume_metrics(restarted)
    assert restarted["capillary_wave_temporal_liquid_volume_available"] is True
    assert restarted["capillary_wave_temporal_liquid_volume_reference_step"] == 6


def test_parse_solver_diagnostics_retains_initial_and_accepted_wet_volumes():
    smoke = _load_smoke_module()
    prefix = "[svMultiPhysics::Application] Wet volume diagnostic"
    common = (
        " field='phi' domain_id='open_vessel_surface' marker=1484991"
        " active_side=LevelSetNegative isovalue=0"
        " wet_volume_frame=physical reference_wet_volume=128"
        " volume_rule_count=36 physical_volume_rule_count=36"
        " skipped_physical_volume_rule_count=0 cut_cell_count=8"
        " full_wet_cell_count=28 full_dry_cell_count=28"
    )
    output = "\n".join([
        "[svMultiPhysics::Application] Transient solve: t0=0 dt=0.002 "
        "t_end=0.002 max_steps=8 scheme=GeneralizedAlpha rho_inf=0.5 "
        "pde_udot_init=1 last_step_absorb_fraction=0.01 "
        "newton(max_it=8, min_it=1, abs_tol=1e-8, rel_tol=1e-6)",
        prefix + " step=0 time=0 wet_volume=0.5 physical_wet_volume=0.5"
        " initial_wet_volume=0.5 wet_volume_drift=0"
        " relative_wet_volume_drift=0" + common,
        "[svMultiPhysics::Application] TimeLoop: step_accepted "
        "step=1 time=0.002 dt=0.002",
        prefix + " step=1 time=0.002 wet_volume=0.49999"
        " physical_wet_volume=0.49999 initial_wet_volume=0.5"
        " wet_volume_drift=-1e-5 relative_wet_volume_drift=-2e-5" + common,
    ])

    diagnostics = smoke.parse_solver_diagnostics(output)

    assert diagnostics["counts"]["wet_volume_diagnostics"] == 2
    assert diagnostics["solver_controls"]["transient_solve"]["t0"] == 0.0
    assert diagnostics["solver_controls"]["transient_solve"][
        "pde_udot_init"] == 1
    assert diagnostics["solver_controls"]["transient_solve"][
        "last_step_absorb_fraction"] == 0.01
    assert [record["step"] for record in diagnostics["wet_volume_diagnostics"]] == [0, 1]
    assert diagnostics["wet_volume_diagnostics"][0]["wet_volume_frame"] == "physical"
    assert diagnostics["wet_volume_diagnostics"][1]["physical_wet_volume"] == 0.49999


def test_free_surface_conservative_balance_parser_metrics_and_gate():
    smoke = _load_smoke_module()
    output = (
        "NewtonSolver: free-surface conservative balance "
        "diagnostic=free_surface_conservative_balance available=1 rank=0 "
        "iteration=3 phase='jacobian_and_residual' "
        "sync_point=JacobianAndResidualAssembly "
        "pressure_virtual_work_norm=4 "
        "surface_energy_virtual_work_norm=6 "
        "physical_potential_virtual_work_norm=6 "
        "conservative_balance_norm=1 "
        "normalization=pressure_plus_physical_potential_norms "
        "normalized_imbalance=0.1 magnitude_mismatch=0.2 "
        "alignment_cosine=-0.9916666666666667 "
        "scope=pressure_and_physical_potential_first_variations_only "
        "contract=instantaneous_constrained_velocity_test_virtual_work "
        "excludes=line_friction_and_wetted_wall_navier_dissipation "
        "total_momentum_equilibrium_claimed=0 "
        "discrete_energy_theorem_claimed=0 "
        "pressure_representability_available=true "
        "pressure_representability_method=lsqr "
        "pressure_representability_convergence="
        "normal_equation_stationarity "
        "pressure_representability_distance_gate_applied=false "
        "pressure_representability_claimed=false "
        "pressure_representability_residual_norm=0.25 "
        "pressure_representability_relative_residual=0.025 "
        "pressure_representability_normal_residual_norm=0.01 "
        "pressure_representability_relative_normal_residual=0.001 "
        "pressure_representability_pressure_norm=2.5 "
        "pressure_representability_iterations=7 "
        "pressure_representability_converged=true "
        "pressure_representability_breakdown=false "
        "pressure_representability_norm=constrained_reduced_coefficient_l2 "
        "pressure_representability_load="
        "prescribed_external_pressure_plus_surface_area_variation_plus_"
        "young_wall_energy_plus_gravitational_potential"
    )
    diagnostics = smoke.parse_solver_diagnostics(output)
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)

    assert diagnostics["counts"]["free_surface_conservative_balances"] == 1
    assert metrics[
        "diagnostic_free_surface_conservative_balance_normalized_imbalance"
    ] == 0.1
    assert metrics["latest_free_surface_conservative_balance"][
        "discrete_energy_theorem_claimed"] == 0
    assert metrics[
        "diagnostic_free_surface_pressure_representability_relative_residual"
    ] == 0.025
    assert metrics[
        "diagnostic_free_surface_pressure_representability_iterations"
    ] == 7

    args = argparse.Namespace(
        require_free_surface_conservative_balance=True,
        max_free_surface_conservative_balance_normalized_imbalance=0.11,
    )
    assert smoke.free_surface_conservative_balance_errors(metrics, args) == []

    args.max_free_surface_conservative_balance_normalized_imbalance = 0.09
    errors = smoke.free_surface_conservative_balance_errors(metrics, args)
    assert len(errors) == 1
    assert "0.1 exceeds 0.09" in errors[0]


def test_free_surface_pressure_representability_gate_accepts_complete_telemetry():
    smoke = _load_smoke_module()
    record = {
        "pressure_representability_available": True,
        "pressure_representability_method": "lsqr",
        "pressure_representability_convergence": (
            "normal_equation_stationarity"
        ),
        "pressure_representability_distance_gate_applied": False,
        "pressure_representability_claimed": False,
        "pressure_representability_residual_norm": 0.25,
        "pressure_representability_relative_residual": 0.025,
        "pressure_representability_normal_residual_norm": 0.01,
        "pressure_representability_relative_normal_residual": 0.001,
        "pressure_representability_pressure_norm": 2.5,
        "pressure_representability_iterations": 7,
        "pressure_representability_converged": True,
        "pressure_representability_breakdown": False,
        "pressure_representability_norm": "constrained_reduced_coefficient_l2",
        "pressure_representability_load": (
            "prescribed_external_pressure_plus_surface_area_variation_plus_"
            "young_wall_energy_plus_gravitational_potential"
        ),
    }
    metrics = {
        "diagnostics": {"free_surface_conservative_balances": [record]},
    }
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=True,
    )

    assert smoke.free_surface_pressure_representability_errors(
        metrics, args) == []


def _complete_pressure_representability_record():
    return {
        "pressure_representability_available": True,
        "pressure_representability_method": "lsqr",
        "pressure_representability_convergence": (
            "normal_equation_stationarity"
        ),
        "pressure_representability_distance_gate_applied": False,
        "pressure_representability_claimed": False,
        "pressure_representability_residual_norm": 0.25,
        "pressure_representability_relative_residual": 0.025,
        "pressure_representability_normal_residual_norm": 0.01,
        "pressure_representability_relative_normal_residual": 0.001,
        "pressure_representability_pressure_norm": 2.5,
        "pressure_representability_iterations": 7,
        "pressure_representability_converged": True,
        "pressure_representability_breakdown": False,
        "pressure_representability_norm": "constrained_reduced_coefficient_l2",
        "pressure_representability_load": (
            "prescribed_external_pressure_plus_surface_area_variation_plus_"
            "young_wall_energy_plus_gravitational_potential"
        ),
    }


def _passing_pressure_representability_distance_gate_record():
    return {
        "accepted_static_state": 1,
        "pressure_representability_distance_gate_applied": 1,
        "pressure_representability_available": 1,
        "pressure_representability_converged": 1,
        "pressure_representability_breakdown": 0,
        "pressure_representability_relative_residual": 0.025,
        "pressure_representability_max_relative_distance": 0.05,
        "pressure_representability_distance_gate_passed": 1,
        "pressure_representability_claimed": 1,
        "pressure_representability_reason": "available",
        "reason": "within_threshold",
    }


def _passing_static_compatible_pressure_initializer_record():
    return {
        "requested": 1,
        "applied": 1,
        "passed": 1,
        "reason": "initialized_within_threshold",
        "pressure_representability_available": 1,
        "pressure_representability_converged": 1,
        "pressure_representability_breakdown": 0,
        "pressure_representability_relative_residual": 0.025,
        "pressure_representability_max_relative_distance": 0.05,
        "force_projection_applied": 0,
        "production_capillary_operator_changed": 0,
    }


def _passing_static_capillary_equilibrium_initialization_record():
    return {
        "active_coefficients": 22,
        "target_liquid_volume": 0.136,
        "initial_physical_potential_energy": 0.4661,
        "final_physical_potential_energy": 0.4658,
        "final_volume_error": -1.0e-12,
        "final_projected_gradient_norm": 9.0e-7,
        "pressure_representability_available": 1,
        "pressure_representability_converged": 1,
        "pressure_representability_breakdown": 0,
        "pressure_representability_residual_norm": 0.002,
        "pressure_representability_relative_distance": 0.003,
        "production_residual_norm": 0.008,
        "constant_pressure_kkt_required": 0,
        "constant_pressure_kkt_available": 0,
        "constant_pressure_kkt_residual_norm": math.nan,
        "constant_pressure_kkt_relative_distance": math.nan,
        "iterations": 50,
        "functional_evaluations": 2387,
        "acceptance_certificate_evaluations": 1,
        "topology_change_rejections": 0,
        "constraint_change_rejections": 0,
        "production_force_projection_applied": 0,
        "qualification": "prerequisite_only",
    }


def _static_capillary_equilibrium_gate_args():
    return argparse.Namespace(
        initialize_discrete_static_capillary_equilibrium=True,
        static_capillary_volume_tolerance=1.0e-8,
        static_capillary_projected_gradient_tolerance=1.0e-6,
        static_capillary_pressure_representability_max_residual_norm=0.01,
        static_capillary_pressure_representability_max_relative_distance=0.01,
        static_capillary_physical_equilibrium_max_residual_norm=0.01,
        static_capillary_constant_pressure_kkt_max_residual_norm=1.0e-6,
        static_capillary_constant_pressure_kkt_max_relative_distance=1.0e-6,
        static_capillary_max_iterations=60,
        require_static_capillary_balance_qualification="prerequisite_only",
    )


def test_static_capillary_equilibrium_initialization_is_parsed_and_gated():
    smoke = _load_smoke_module()
    record = _passing_static_capillary_equilibrium_initialization_record()
    output = (
        "Static capillary equilibrium initialized "
        "diagnostic=static_capillary_equilibrium_initialization "
        + " ".join(f"{key}={value}" for key, value in record.items())
    )
    diagnostics = smoke.parse_solver_diagnostics(output)
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)

    assert diagnostics["counts"][
        "static_capillary_equilibrium_initializations"] == 1
    assert metrics[
        "diagnostic_static_capillary_equilibrium_initialization_count"] == 1
    assert metrics["static_capillary_active_coefficients"] == 22
    assert metrics["static_capillary_qualification"] == "prerequisite_only"
    assert smoke.static_capillary_equilibrium_initialization_errors(
        metrics, _static_capillary_equilibrium_gate_args()) == []


def test_static_capillary_equilibrium_initializer_enables_balance_operator(
        monkeypatch):
    smoke = _load_smoke_module()

    class GateArgs:
        initialize_discrete_static_capillary_equilibrium = True

        def __getattr__(self, _name):
            return None

    monkeypatch.setenv(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "0")
    env = smoke.solver_environment(GateArgs())
    assert env[
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"] == "1"


def test_static_capillary_equilibrium_constant_pressure_certificate_is_gated():
    smoke = _load_smoke_module()
    record = _passing_static_capillary_equilibrium_initialization_record()
    record.update({
        "constant_pressure_kkt_required": 1,
        "constant_pressure_kkt_available": 1,
        "constant_pressure_kkt_residual_norm": 3.0e-7,
        "constant_pressure_kkt_relative_distance": 4.0e-7,
    })
    metrics = {
        "diagnostics": {
            "static_capillary_equilibrium_initializations": [record],
        },
    }
    args = _static_capillary_equilibrium_gate_args()
    assert smoke.static_capillary_equilibrium_initialization_errors(
        metrics, args) == []

    record["constant_pressure_kkt_relative_distance"] = 2.0e-6
    errors = smoke.static_capillary_equilibrium_initialization_errors(
        metrics, args)
    assert any(
        "constant_pressure_kkt_relative_distance 2e-06 exceeds 1e-06"
        in error for error in errors)


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"final_volume_error": -2.0e-8}, "final_volume_error"),
        ({"final_physical_potential_energy": 0.47}, "increased physical"),
        ({"pressure_representability_relative_distance": 0.02},
         "pressure_representability_relative_distance"),
        ({"qualification": "qualified"}, "reported qualification"),
        ({"constant_pressure_kkt_required": 1},
         "availability disagrees"),
    ],
)
def test_static_capillary_equilibrium_initialization_fails_closed(
        update, expected):
    smoke = _load_smoke_module()
    record = _passing_static_capillary_equilibrium_initialization_record()
    record.update(update)
    metrics = {
        "diagnostics": {
            "static_capillary_equilibrium_initializations": [record],
        },
    }
    errors = smoke.static_capillary_equilibrium_initialization_errors(
        metrics, _static_capillary_equilibrium_gate_args())
    assert errors
    assert any(expected in error for error in errors)


def test_static_capillary_equilibrium_initialization_requires_one_record():
    smoke = _load_smoke_module()
    metrics = {
        "diagnostics": {
            "static_capillary_equilibrium_initializations": [],
        },
    }
    errors = smoke.static_capillary_equilibrium_initialization_errors(
        metrics, _static_capillary_equilibrium_gate_args())
    assert errors == [
        "discrete static-capillary equilibrium initialization requires "
        "exactly one diagnostic record (observed 0)"
    ]


def test_static_compatible_pressure_initializer_is_parsed_and_required():
    smoke = _load_smoke_module()
    output = (
        "NewtonSolver: static compatible free-surface pressure initializer "
        "diagnostic=free_surface_static_compatible_pressure_initializer "
        "requested=1 applied=1 passed=1 "
        "reason=initialized_within_threshold "
        "pressure_representability_available=1 "
        "pressure_representability_converged=1 "
        "pressure_representability_breakdown=0 "
        "pressure_representability_relative_residual=0.025 "
        "pressure_representability_max_relative_distance=0.05 "
        "force_projection_applied=0 production_capillary_operator_changed=0"
    )
    diagnostics = smoke.parse_solver_diagnostics(output)
    diagnostics["free_surface_conservative_balances"] = [
        _complete_pressure_representability_record()
    ]
    diagnostics[
        "free_surface_pressure_representability_distance_gates"
    ] = [_passing_pressure_representability_distance_gate_record()]
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=True,
        max_free_surface_pressure_representability_relative_distance=0.05,
        initialize_static_compatible_pressure=True,
    )

    assert diagnostics["counts"][
        "free_surface_static_compatible_pressure_initializers"] == 1
    assert metrics[
        "diagnostic_free_surface_static_compatible_pressure_initializer_count"
    ] == 1
    assert smoke.free_surface_pressure_representability_errors(
        metrics, args) == []


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([], "was not reported"),
        ([{
            **_passing_static_compatible_pressure_initializer_record(),
            "applied": 0,
            "passed": 0,
            "reason": "relative_distance_exceeds_threshold",
        }], "failed attempt"),
    ],
)
def test_static_compatible_pressure_initializer_fails_closed(records, expected):
    smoke = _load_smoke_module()
    metrics = {
        "diagnostics": {
            "free_surface_conservative_balances": [
                _complete_pressure_representability_record()
            ],
            "free_surface_pressure_representability_distance_gates": [
                _passing_pressure_representability_distance_gate_record()
            ],
            "free_surface_static_compatible_pressure_initializers": records,
        },
    }
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=True,
        max_free_surface_pressure_representability_relative_distance=0.05,
        initialize_static_compatible_pressure=True,
    )

    errors = smoke.free_surface_pressure_representability_errors(metrics, args)

    assert any(expected in error for error in errors)


def test_accepted_static_pressure_representability_distance_gate_is_required():
    smoke = _load_smoke_module()
    output = (
        "NewtonSolver: accepted static pressure-representability distance gate "
        "diagnostic=free_surface_pressure_representability_distance_gate "
        "accepted_static_state=1 iteration=0 phase='pre_linear_convergence' "
        "pressure_representability_distance_gate_applied=1 "
        "pressure_representability_available=1 "
        "pressure_representability_converged=1 "
        "pressure_representability_breakdown=0 "
        "pressure_representability_relative_residual=0.025 "
        "pressure_representability_max_relative_distance=0.05 "
        "pressure_representability_distance_gate_passed=1 "
        "pressure_representability_claimed=1 "
        "pressure_representability_reason=available reason=within_threshold"
    )
    diagnostics = smoke.parse_solver_diagnostics(output)
    diagnostics["free_surface_conservative_balances"] = [
        _complete_pressure_representability_record()
    ]
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=False,
        max_free_surface_pressure_representability_relative_distance=0.05,
    )

    assert diagnostics["counts"][
        "free_surface_pressure_representability_distance_gates"] == 1
    assert metrics[
        "diagnostic_free_surface_pressure_representability_distance_gate_count"
    ] == 1
    assert metrics[
        "latest_free_surface_pressure_representability_distance_gate"
    ]["reason"] == "within_threshold"
    assert smoke.free_surface_pressure_representability_errors(
        metrics, args) == []


def test_accepted_static_pressure_representability_distance_gate_missing_fails_closed():
    smoke = _load_smoke_module()
    metrics = {
        "diagnostics": {
            "free_surface_conservative_balances": [
                _complete_pressure_representability_record()
            ],
        },
    }
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=False,
        max_free_surface_pressure_representability_relative_distance=0.05,
    )

    errors = smoke.free_surface_pressure_representability_errors(metrics, args)

    assert any("distance gate was not reported" in error for error in errors)


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({
            "pressure_representability_available": 0,
            "pressure_representability_distance_gate_passed": 0,
            "pressure_representability_claimed": 0,
            "reason": "diagnostic_unavailable",
        }, "pressure_representability_available"),
        ({
            "pressure_representability_relative_residual": 0.08,
            "pressure_representability_distance_gate_passed": 0,
            "pressure_representability_claimed": 0,
            "reason": "relative_distance_exceeds_threshold",
        }, "0.08 exceeds 0.05"),
    ],
)
def test_accepted_static_pressure_representability_distance_gate_fails_closed(
        update, expected):
    smoke = _load_smoke_module()
    gate_record = _passing_pressure_representability_distance_gate_record()
    gate_record.update(update)
    metrics = {
        "diagnostics": {
            "free_surface_conservative_balances": [
                _complete_pressure_representability_record()
            ],
            "free_surface_pressure_representability_distance_gates": [
                gate_record
            ],
        },
    }
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=False,
        max_free_surface_pressure_representability_relative_distance=0.05,
    )

    errors = smoke.free_surface_pressure_representability_errors(metrics, args)

    assert errors
    assert any(expected in error for error in errors)


def test_pressure_representability_diagnostic_accepts_stationary_unit_residual_without_claim():
    smoke = _load_smoke_module()
    record = {
        "pressure_representability_available": True,
        "pressure_representability_method": "lsqr",
        "pressure_representability_convergence": (
            "normal_equation_stationarity"
        ),
        "pressure_representability_distance_gate_applied": False,
        "pressure_representability_claimed": False,
        "pressure_representability_residual_norm": 2.0,
        "pressure_representability_relative_residual": 1.0,
        "pressure_representability_normal_residual_norm": 0.0,
        "pressure_representability_relative_normal_residual": 0.0,
        "pressure_representability_pressure_norm": 0.0,
        "pressure_representability_iterations": 0,
        "pressure_representability_converged": True,
        "pressure_representability_breakdown": False,
        "pressure_representability_norm": "constrained_reduced_coefficient_l2",
        "pressure_representability_load": (
            "prescribed_external_pressure_plus_surface_area_variation_plus_"
            "young_wall_energy_plus_gravitational_potential"
        ),
    }
    metrics = {
        "diagnostics": {"free_surface_conservative_balances": [record]},
    }
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=True,
    )

    # A range-orthogonal load can be stationary with relative residual one.
    # This requirement validates diagnostic execution only; it is not a
    # residual-distance gate and must not imply representability.
    assert smoke.free_surface_pressure_representability_errors(
        metrics, args) == []


def test_pressure_representability_requirement_enables_solver_diagnostic(
        monkeypatch):
    smoke = _load_smoke_module()

    class GateArgs:
        require_free_surface_pressure_representability_diagnostic = True

        def __getattr__(self, _name):
            return None

    monkeypatch.setenv(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "0")

    env = smoke.solver_environment(GateArgs())

    assert env[
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"] == "1"


def test_pressure_representability_distance_threshold_enables_static_gate(
        monkeypatch):
    smoke = _load_smoke_module()

    class GateArgs:
        require_free_surface_pressure_representability_diagnostic = False
        max_free_surface_pressure_representability_relative_distance = 0.05

        def __getattr__(self, _name):
            return None

    monkeypatch.delenv(
        "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE",
        raising=False,
    )

    env = smoke.solver_environment(GateArgs())

    assert env[
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"] == "1"
    assert float(env[
        "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE"
    ]) == 0.05


def test_static_compatible_pressure_initializer_enables_solver_preload(
        monkeypatch):
    smoke = _load_smoke_module()

    class GateArgs:
        initialize_static_compatible_pressure = True
        max_free_surface_pressure_representability_relative_distance = 0.05

        def __getattr__(self, _name):
            return None

    monkeypatch.delenv(
        "SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER",
        raising=False,
    )

    env = smoke.solver_environment(GateArgs())

    assert env[
        "SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER"] == "1"
    assert env[
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"] == "1"


def test_static_compatible_pressure_initializer_requires_distance_gate():
    smoke = _load_smoke_module()

    class GateArgs:
        initialize_static_compatible_pressure = True
        max_free_surface_pressure_representability_relative_distance = None

        def __getattr__(self, _name):
            return None

    with pytest.raises(ValueError, match="requires"):
        smoke.solver_environment(GateArgs())


def test_sessile_defaults_require_pressure_representability_but_dynamic_do_not():
    smoke = _load_smoke_module()
    base = argparse.Namespace(
        high_order_mpi_production_qualification=False,
        steps=None,
        time_step_size=None,
        timeout_seconds=None,
        surface_tension=None,
        require_free_surface_conservative_balance=False,
        require_free_surface_pressure_representability_diagnostic=False,
        max_free_surface_pressure_representability_relative_distance=None,
        initialize_static_compatible_pressure=None,
    )

    sessile = smoke.case_args_for_run("sessile2d", base)
    dynamic = smoke.case_args_for_run("dynamiccontact2d", base)

    assert sessile.require_free_surface_conservative_balance is True
    assert (
        sessile.require_free_surface_pressure_representability_diagnostic
        is True
    )
    assert (
        sessile.max_free_surface_pressure_representability_relative_distance ==
        smoke.STATIC_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE
    )
    assert sessile.initialize_static_compatible_pressure is True
    assert dynamic.require_free_surface_conservative_balance is False
    assert (
        dynamic.require_free_surface_pressure_representability_diagnostic
        is False
    )
    assert (
        dynamic.max_free_surface_pressure_representability_relative_distance
        is None
    )
    assert dynamic.initialize_static_compatible_pressure is False

    discrete_args = argparse.Namespace(**vars(base))
    discrete_args.initialize_discrete_static_capillary_equilibrium = True
    discrete = smoke.case_args_for_run("sessile2d", discrete_args)
    assert discrete.initialize_static_compatible_pressure is False


def test_spatial_static_defaults_select_dimension_specific_physical_gates():
    smoke = _load_smoke_module()
    base = argparse.Namespace(
        high_order_mpi_production_qualification=False,
        steps=None,
        time_step_size=None,
        timeout_seconds=None,
        surface_tension=None,
        require_free_surface_conservative_balance=False,
        require_free_surface_pressure_representability_diagnostic=False,
        max_free_surface_pressure_representability_relative_distance=None,
        initialize_static_compatible_pressure=None,
    )

    sphere = smoke.case_args_for_run("sphere3d", base)
    sessile = smoke.case_args_for_run("sessile3d", base)

    assert sphere.max_capillary_pressure_jump_relative_error == 0.15
    assert sphere.max_capillary_parasitic_capillary_number == 1.0e-2
    assert getattr(
        sphere, "max_sessile_contact_angle_error_degrees", None
    ) is None
    assert sessile.max_sessile_contact_angle_error_degrees == 5.0
    assert sessile.max_sessile_liquid_area_relative_error is None
    assert sessile.max_sessile_liquid_volume_relative_error == 0.05
    assert sessile.max_sessile_base_radius_relative_error == 0.05
    assert sessile.max_sessile_apex_height_relative_error == 0.05
    for configured in (sphere, sessile):
        assert configured.require_free_surface_energy_history is True
        assert configured.require_free_surface_conservative_balance is True
        assert (
            configured.require_free_surface_pressure_representability_diagnostic
            is True
        )
        assert configured.initialize_static_compatible_pressure is True


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"pressure_representability_available": False}, "unavailable"),
        ({"pressure_representability_method": "cg"}, "unexpected"),
        ({"pressure_representability_convergence": "residual"}, "unexpected"),
        ({"pressure_representability_distance_gate_applied": True}, "unexpected"),
        ({"pressure_representability_claimed": True}, "unexpected"),
        ({"pressure_representability_residual_norm": math.inf}, "nonfinite"),
        ({"pressure_representability_relative_residual": -1.0}, "negative"),
        ({"pressure_representability_iterations": -1}, "negative"),
        ({"pressure_representability_converged": False}, "did not converge"),
        ({"pressure_representability_breakdown": True}, "breakdown"),
    ],
)
def test_free_surface_pressure_representability_gate_fails_closed(
        update, expected):
    smoke = _load_smoke_module()
    record = {
        "pressure_representability_available": True,
        "pressure_representability_method": "lsqr",
        "pressure_representability_convergence": (
            "normal_equation_stationarity"
        ),
        "pressure_representability_distance_gate_applied": False,
        "pressure_representability_claimed": False,
        "pressure_representability_residual_norm": 0.25,
        "pressure_representability_relative_residual": 0.025,
        "pressure_representability_normal_residual_norm": 0.01,
        "pressure_representability_relative_normal_residual": 0.001,
        "pressure_representability_pressure_norm": 2.5,
        "pressure_representability_iterations": 7,
        "pressure_representability_converged": True,
        "pressure_representability_breakdown": False,
        "pressure_representability_norm": "constrained_reduced_coefficient_l2",
        "pressure_representability_load": (
            "prescribed_external_pressure_plus_surface_area_variation_plus_"
            "young_wall_energy_plus_gravitational_potential"
        ),
    }
    record.update(update)
    metrics = {
        "diagnostics": {"free_surface_conservative_balances": [record]},
    }
    args = argparse.Namespace(
        require_free_surface_pressure_representability_diagnostic=True,
    )

    errors = smoke.free_surface_pressure_representability_errors(metrics, args)

    assert errors
    assert any(expected in error for error in errors)


def test_free_surface_conservative_balance_gate_fails_closed_and_enables_solver(
        monkeypatch):
    smoke = _load_smoke_module()

    class GateArgs:
        enable_free_surface_conservative_balance_diagnostic = False
        require_free_surface_conservative_balance = True
        max_free_surface_conservative_balance_normalized_imbalance = None

        def __getattr__(self, _name):
            return None

    args = GateArgs()
    metrics = {
        "diagnostics": {"free_surface_conservative_balances": [{
            "available": 0,
            "reason": "operator_not_installed",
        }]},
    }

    errors = smoke.free_surface_conservative_balance_errors(metrics, args)
    assert len(errors) == 1
    assert "operator_not_installed" in errors[0]

    monkeypatch.setenv(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "0")
    env = smoke.solver_environment(args)
    assert env[
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"] == "1"


def test_free_surface_balance_assemblies_do_not_inflate_production_efficiency_metrics():
    smoke = _load_smoke_module()
    diagnostics = smoke.parse_solver_diagnostics("")
    diagnostic_ops = sorted(
        smoke.FREE_SURFACE_CONSERVATIVE_BALANCE_OPERATOR_TAGS)
    diagnostics["assembly_timings"] = [
        {"op": "equations", "total": 1.0},
        *({"op": op, "total": 0.1} for op in diagnostic_ops),
    ]
    diagnostics["newton_assemblies"] = [
        {"op": "equations", "phase": "jacobian_and_residual",
         "sync_point": "JacobianAndResidualAssembly",
         "want_matrix": 1, "want_vector": 1, "iteration": 0},
        *({"op": op, "phase": "free_surface_conservative_balance",
           "sync_point": "JacobianAndResidualAssembly",
           "want_matrix": int(op.endswith("pressure_representability_pair")),
           "want_vector": int(not op.endswith("pressure_representability_pair")),
           "iteration": 0}
          for op in diagnostic_ops),
    ]
    diagnostics["time_loop"]["summary"] = {
        "accepted_steps": 1,
        "nonlinear_iterations_total": 1,
    }
    metrics = {}

    smoke.add_diagnostic_metrics(metrics, diagnostics)

    assert metrics["diagnostic_assembly_timing_count"] == 1
    assert metrics["diagnostic_assembly_timings_per_accepted_step"] == 1.0
    assert metrics[
        "diagnostic_extra_assembly_timings_per_accepted_step"] == 0.0
    assert metrics["diagnostic_newton_assembly_count"] == 1
    assert metrics["diagnostic_newton_assemblies_per_accepted_step"] == 1.0
    assert metrics["diagnostic_newton_matrix_assembly_count"] == 1
    assert metrics["diagnostic_newton_vector_assembly_count"] == 1
    assert metrics[
        "diagnostic_free_surface_conservative_balance_assembly_timing_count"
    ] == 4
    assert metrics[
        "diagnostic_free_surface_pressure_representability_assembly_timing_count"
    ] == 1
    assert metrics[
        "diagnostic_free_surface_conservative_balance_newton_assembly_count"
    ] == 4
    assert metrics[
        "diagnostic_free_surface_conservative_balance_newton_matrix_assembly_count"
    ] == 1
    assert metrics[
        "diagnostic_free_surface_conservative_balance_newton_vector_assembly_count"
    ] == 3
    assert metrics[
        "diagnostic_free_surface_pressure_representability_newton_assembly_count"
    ] == 1
    assert metrics[
        "diagnostic_free_surface_pressure_representability_newton_matrix_assembly_count"
    ] == 1


def test_parse_solver_diagnostics_accepts_outer_fixed_point_nonlinear_record():
    smoke = _load_smoke_module()
    output = (
        "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=0 "
        "time=0 converged=1 iters=14 ||r||=4.8e-11 "
        "outer_iters=5 inner_iters_total=14 outer_state_change_norm=0 "
        "||r_field||=4.8e-11 ||r_aux||=0 "
        "(linear: converged=1 iters=31 rel=4.2e-8)"
    )

    diagnostics = smoke.parse_solver_diagnostics(output)

    records = diagnostics["time_loop"]["nonlinear_records"]
    assert len(records) == 1
    assert records[0]["converged"] == 1
    assert records[0]["outer_iterations"] == 5
    assert records[0]["inner_iterations_total"] == 14
    assert records[0]["outer_state_change_norm"] == 0.0
    assert records[0]["linear_converged"] == 1
    summary = smoke.summarize_time_loop(diagnostics["time_loop"])
    assert summary["external_state_fixed_point_records"] == 1
    assert summary["outer_iterations_max"] == 5
    assert summary["inner_iterations_total_max"] == 14
    assert summary["outer_state_change_norm"]["max"] == 0.0

    rejected_output = (
        "[svMultiPhysics::Application] TimeLoop: step_rejected step=0 "
        "time=0 dt=0.001 reason=NonlinearFailure "
        "(newton: converged=0 iters=19 outer_iters=3 "
        "inner_iters_total=19 ||r||=2e-5 ||r_field||=2e-5 "
        "||r_aux||=0)"
    )
    rejected = smoke.parse_solver_diagnostics(rejected_output)["time_loop"][
        "rejected_steps"]
    assert len(rejected) == 1
    assert rejected[0]["outer_iterations"] == 3
    assert rejected[0]["inner_iterations_total"] == 19


def test_capillary_wave_boundary_contract_gate_rejects_retired_transport():
    smoke = _load_smoke_module()
    args = argparse.Namespace(high_order_capillary_wave_smoke=True)
    metrics = {
        "capillary_wave_boundary_contract_valid": False,
        "capillary_wave_boundary_contract_errors": [
            "capillary-wave transport must use wall_compatible_normal extension"
        ],
    }

    errors = smoke.capillary_wave_boundary_contract_errors(metrics, args)

    assert len(errors) == 1
    assert "wall_compatible_normal" in errors[0]


def test_capillary_stability_errors_reject_unstable_time_loop_and_speed():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_capillary_rejected_steps=0,
        max_capillary_dt_updates=0,
        max_capillary_speed_per_surface_tension=2.0,
        max_capillary_nonlinear_residual=1.0e-6,
        max_capillary_linear_relative_residual=1.0e-4,
    )
    metrics = {
        "surface_tension": 0.5,
        "max_speed": 2.0,
        "time_loop": {
            "summary": {
                "rejected_steps": 1,
                "dt_updates": 2,
                "nonlinear_residual": {"max": 2.0e-6},
                "linear_relative_residual": {"max": 2.0e-4},
            },
        },
    }

    errors = smoke.capillary_stability_errors(metrics, args)

    assert len(errors) == 5
    assert any("rejected steps" in error for error in errors)
    assert any("dt updates" in error for error in errors)
    assert any("speed per surface tension" in error for error in errors)
    assert any("nonlinear residual" in error for error in errors)
    assert any("linear relative residual" in error for error in errors)
    assert metrics["capillary_stability_speed_per_surface_tension"] == 4.0


def test_capillary_smoke_defaults_enable_stability_gates():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        high_order_capillary_response_smoke=True,
        high_order_3d_benchmark_qualification=False,
        high_order_3d_benchmark_profile_qualification=False,
        high_order_curved_3d_simplex_smoke=False,
        high_order_mpi_motion_smoke=False,
        high_order_capillary_projection_smoke=False,
        case=None,
        steps=None,
        timeout_seconds=None,
        max_solver_elapsed_seconds_per_accepted_step=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
    )

    smoke.apply_high_order_capillary_response_smoke_defaults(args)

    assert args.max_capillary_rejected_steps == 0
    assert args.max_capillary_dt_updates == 0
    assert args.max_capillary_speed_per_surface_tension == 10.0


def test_capillary_droplet_defaults_enable_equilibrium_gates():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        high_order_capillary_droplet_equilibrium_smoke=True,
        case=None,
        steps=None,
        timeout_seconds=None,
        max_solver_elapsed_seconds_per_accepted_step=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
    )

    smoke.apply_high_order_capillary_droplet_equilibrium_smoke_defaults(args)

    assert args.case == ["droplet2d"]
    assert args.steps == 3
    assert args.use_high_order_implicit_cuts is True
    assert args.required_implicit_cut_backend_qualification == "ProductionQualified"
    assert args.surface_tension == 0.5
    assert getattr(args, "projected_curvature_field", None) is None
    assert args.require_curvature_projection_diagnostics is False
    assert args.require_curvature_projection_newton_freshness is False
    assert getattr(args, "max_capillary_curvature_relative_error", None) is None
    assert args.max_capillary_pressure_jump_relative_error == 0.15
    assert args.max_capillary_rejected_steps == 0
    assert args.max_capillary_dt_updates == 0
    assert args.max_capillary_speed_per_surface_tension == 1.0e-5
    assert args.max_capillary_balance_speed_per_surface_tension == 1.0e-5
    assert args.linear_solver_type == "gmres"
    assert args.linear_algebra_backend == "fsils"
    assert args.linear_preconditioner == "rcs"
    assert args.linear_max_iterations == 100
    assert args.linear_krylov_space_dimension == 50
    assert args.linear_relative_tolerance == 1.0e-8
    assert args.linear_absolute_tolerance == 1.0e-10
    assert args.require_eigen_factorization_diagnostics is False
    assert args.enable_fsils_matrix_diagnostics is True
    assert args.require_fsils_matrix_diagnostics is True
    assert args.max_fsils_matrix_zero_rows == 0
    assert args.max_fsils_matrix_missing_diag == 0
    assert args.max_fsils_matrix_diag_col_mismatch == 0
    assert args.max_fsils_matrix_duplicate_diag_entries == 0
    assert args.max_fsils_matrix_duplicate_diag_rows == 0
    assert args.max_fsils_matrix_zero_diag == 0
    assert args.max_fsils_matrix_nonfinite_entries == 0
    assert getattr(
        args, "max_diagnostic_curvature_projection_sample_adjacency_builds", None
    ) is None
    assert args.max_solver_elapsed_seconds_per_accepted_step is None
    assert getattr(args, "max_time_loop_nonlinear_iterations_per_step", None) is None
    assert getattr(args, "max_time_loop_linear_iterations_per_step", None) is None
    assert args.min_max_speed == 0.0
    assert args.min_wet_mean_speed == 0.0


def test_physical_surface_stress_defaults_do_not_reintroduce_resource_ceilings():
    smoke = _load_smoke_module()
    common = dict(
        use_high_order_implicit_cuts=True,
        generated_interface_geometry=None,
        implicit_cut_quadrature_backend=None,
        implicit_cut_fallback_policy=None,
        implicit_cut_root_tolerance=None,
        implicit_cut_max_subdivision_depth=None,
        generated_interface_quadrature_order=None,
        interface_quadrature_order=None,
        volume_quadrature_order=None,
        linear_algebra_backend="fsils",
        disable_cut_stabilization=False,
        mms_nx=None,
        mms_ny=None,
        max_diagnostic_assembly_timings_per_step=None,
        max_diagnostic_extra_assembly_timings_per_step=None,
        max_diagnostic_cut_context_rebuilds_per_step=None,
        max_diagnostic_process_rss_kb=None,
        max_diagnostic_process_rss_growth_kb=None,
        max_diagnostic_process_basis_cache_entries=None,
        max_diagnostic_process_basis_cache_entry_growth=None,
        expect_generated_interface_geometry=None,
        expect_implicit_cut_quadrature_backend=None,
        expect_selected_implicit_cut_quadrature_backend=None,
        expect_implicit_cut_fallback_policy=None,
        max_diagnostic_implicit_cut_fallback_cells=0,
        min_diagnostic_achieved_volume_quadrature_order=2,
    )
    args = argparse.Namespace(
        **common,
        high_order_capillary_droplet_equilibrium_smoke=True,
        high_order_capillary_wave_smoke=False,
    )

    smoke.apply_high_order_implicit_defaults(args)

    assert args.max_diagnostic_assembly_timings_per_step is None
    assert args.max_diagnostic_extra_assembly_timings_per_step is None
    assert args.max_diagnostic_cut_context_rebuilds_per_step is None
    assert args.max_diagnostic_process_rss_kb is None
    assert args.max_diagnostic_process_rss_growth_kb is None
    assert args.max_diagnostic_process_basis_cache_entries is None
    assert args.max_diagnostic_process_basis_cache_entry_growth is None

    explicit = argparse.Namespace(
        **{
            **common,
            "max_diagnostic_assembly_timings_per_step": 9.0,
            "max_diagnostic_process_basis_cache_entries": 12,
        },
        high_order_capillary_droplet_equilibrium_smoke=False,
        high_order_capillary_wave_smoke=True,
    )
    smoke.apply_high_order_implicit_defaults(explicit)
    assert explicit.max_diagnostic_assembly_timings_per_step == 9.0
    assert explicit.max_diagnostic_process_basis_cache_entries == 12


def test_capillary_wave_defaults_enable_reference_profile_gates():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        high_order_capillary_wave_smoke=True,
        case=None,
        steps=None,
        time_step_size=None,
        timeout_seconds=None,
        max_solver_elapsed_seconds_per_accepted_step=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
    )

    smoke.apply_high_order_capillary_wave_smoke_defaults(args)

    assert args.case == ["capillarywave2d"]
    assert args.steps == 107
    assert args.time_step_size == 2.0e-3
    phase_per_step = (
        smoke.capillary_wave_omega(args.surface_tension) * args.time_step_size
    )
    assert (
        args.steps * phase_per_step
        >= smoke.CAPILLARY_WAVE_MINIMUM_FREQUENCY_PHASE_SPAN
    )
    assert (
        (args.steps - 1) * phase_per_step
        < smoke.CAPILLARY_WAVE_MINIMUM_FREQUENCY_PHASE_SPAN
    )
    assert args.use_high_order_implicit_cuts is True
    assert args.require_reference_profile_comparison is True
    assert args.enable_physical_history_instrumentation is True
    assert args.surface_tension == 50.0
    assert args.wet_extension_advection_velocity_method == "wall_compatible_normal"
    assert getattr(args, "projected_curvature_field", None) is None
    assert args.require_curvature_projection_diagnostics is False
    assert args.require_curvature_projection_newton_freshness is False
    assert args.max_capillary_wave_frequency_relative_error == 0.10
    assert args.max_capillary_wave_profile_relative_error == 0.25
    assert args.max_capillary_wave_mean_offset == 4.0e-3
    assert (
        args.max_capillary_wave_temporal_liquid_volume_relative_drift
        == smoke.CAPILLARY_WAVE_MAX_TEMPORAL_LIQUID_VOLUME_RELATIVE_DRIFT
        == 1.0e-5
    )
    assert args.linear_solver_type == "gmres"
    assert args.linear_algebra_backend == "fsils"
    assert args.linear_preconditioner == "rcs"
    assert args.linear_krylov_space_dimension == 50
    assert args.require_eigen_factorization_diagnostics is False
    assert args.enable_fsils_matrix_diagnostics is True
    assert args.require_fsils_matrix_diagnostics is True
    assert args.max_fsils_matrix_zero_rows == 0
    assert args.max_fsils_matrix_missing_diag == 0
    assert args.max_fsils_matrix_diag_col_mismatch == 0
    assert args.max_fsils_matrix_duplicate_diag_entries == 0
    assert args.max_fsils_matrix_duplicate_diag_rows == 0
    assert args.max_fsils_matrix_zero_diag == 0
    assert args.max_fsils_matrix_nonfinite_entries == 0
    assert args.reference_profile_sample_radius == 0.02
    assert args.max_reference_profile_rmse == 0.01
    assert getattr(
        args, "max_diagnostic_curvature_projection_sample_adjacency_builds", None
    ) is None
    assert getattr(
        args, "max_diagnostic_curvature_projection_cache_miss_count", None
    ) is None
    assert args.max_solver_elapsed_seconds_per_accepted_step is None
    assert getattr(args, "max_time_loop_nonlinear_iterations_per_step", None) is None
    assert getattr(args, "max_time_loop_linear_iterations_per_step", None) is None
    assert args.min_interface_height_change == 1.0e-7
    assert args.min_wet_mean_speed == 0.0
    assert args.require_compiled_cut_volume_jit is True
    assert args.enable_jit_specialization_trace is True
    assert args.enable_jit_cache_diagnostics is True


def test_capillary_compiled_cut_volume_jit_gate_accepts_complete_evidence():
    smoke = _load_smoke_module()
    diagnostics = {
        "jit_specialization_traces": [
            {
                "event": "generic_compile",
                "kind": "SymbolicNonlinearFormKernel",
            },
            {
                "event": "compile",
                "trigger": "runtime",
                "domain": "CutVolume",
                "role": "Tangent",
            },
            {
                "event": "compile",
                "trigger": "runtime",
                "domain": "CutVolume",
                "role": "Residual",
            },
        ],
        "jit_cache_diagnostics": [{"groups": 2, "local_stores": 2}],
        "jit_failure_messages": [],
    }

    assert smoke.compiled_cut_volume_jit_errors(diagnostics) == []


def test_capillary_compiled_cut_volume_jit_gate_rejects_missing_or_failed_evidence():
    smoke = _load_smoke_module()
    diagnostics = {
        "jit_specialization_traces": [
            {
                "event": "compile_failed",
                "trigger": "runtime",
                "domain": "CutVolume",
                "role": "Tangent",
            },
        ],
        "jit_cache_diagnostics": [],
        "jit_failure_messages": ["JIT: runtime failure in computeCell"],
    }

    errors = smoke.compiled_cut_volume_jit_errors(diagnostics)

    assert any("generic JIT compile evidence" in error for error in errors)
    assert any("JIT cache diagnostics" in error for error in errors)
    assert any("Tangent, Residual" in error for error in errors)
    assert any("compile/runtime failure" in error for error in errors)


def test_compiled_cut_volume_jit_gate_forces_runtime_policy_and_diagnostics(monkeypatch):
    smoke = _load_smoke_module()

    class GateArgs:
        require_compiled_cut_volume_jit = True

        def __getattr__(self, _name):
            return None

    monkeypatch.setenv("SVMP_OOP_JIT_ENABLE", "0")
    monkeypatch.setenv("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", "0")

    env = smoke.solver_environment(GateArgs())

    assert env["SVMP_OOP_JIT_ENABLE"] == "1"
    assert env["SVMP_OOP_JIT_SPECIALIZATION_ENABLE"] == "1"
    assert env["SVMP_JIT_TRACE_SPECIALIZATION"] == "1"
    assert env["SVMP_JIT_CACHE_DIAGNOSTICS"] == "1"


def test_solver_diagnostics_capture_jit_compile_and_runtime_failures():
    smoke = _load_smoke_module()
    diagnostics = smoke.parse_solver_diagnostics("\n".join([
        "[WARN] JIT: failed to compile specialized variant for kernel 'ns'",
        "[WARN] JIT: runtime failure in computeCell for kernel 'ns'",
        "[WARN] JIT requested for kernel 'ns', but unavailable; using interpreter.",
    ]))

    assert len(diagnostics["jit_failure_messages"]) == 3


def test_capillary_wave_defaults_preserve_explicit_step_count():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        high_order_capillary_wave_smoke=True,
        case=None,
        steps=3,
        time_step_size=None,
        timeout_seconds=None,
        max_solver_elapsed_seconds_per_accepted_step=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
    )

    smoke.apply_high_order_capillary_wave_smoke_defaults(args)

    assert args.steps == 3
    assert getattr(
        args, "max_diagnostic_curvature_projection_cache_miss_count", None
    ) is None


def test_capillary_convergence_rate_errors_reject_low_rate():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        min_capillary_convergence_rate=1.5,
        min_capillary_convergence_points=2,
        capillary_convergence_resolution_key="nx",
        capillary_convergence_metric=["capillary_curvature_relative_error"],
    )
    probes = [
        {
            "passed": True,
            "nx": 8,
            "capillary_curvature_relative_error": 0.20,
        },
        {
            "passed": True,
            "nx": 16,
            "capillary_curvature_relative_error": 0.12,
        },
    ]

    errors = smoke.capillary_convergence_rate_errors(probes, args)

    assert len(errors) == 1
    assert "convergence rate" in errors[0]
    assert "capillary_curvature_relative_error" in errors[0]


def test_capillary_convergence_rate_errors_accept_expected_rate():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        min_capillary_convergence_rate=1.5,
        min_capillary_convergence_points=3,
        capillary_convergence_resolution_key="nx",
        capillary_convergence_metric=[
            "capillary_curvature_relative_error,capillary_pressure_jump_relative_error"
        ],
    )
    probes = [
        {
            "passed": True,
            "nx": 8,
            "capillary_curvature_relative_error": 0.16,
            "capillary_pressure_jump_relative_error": 0.08,
        },
        {
            "passed": True,
            "nx": 16,
            "capillary_curvature_relative_error": 0.04,
            "capillary_pressure_jump_relative_error": 0.02,
        },
        {
            "passed": True,
            "nx": 32,
            "capillary_curvature_relative_error": 0.01,
            "capillary_pressure_jump_relative_error": 0.005,
        },
    ]

    errors = smoke.capillary_convergence_rate_errors(probes, args)

    assert errors == []


def test_capillary_convergence_rate_errors_require_enough_samples():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        min_capillary_convergence_rate=1.0,
        min_capillary_convergence_points=2,
        capillary_convergence_resolution_key="nx",
        capillary_convergence_metric=None,
    )
    probes = [
        {
            "passed": True,
            "nx": 8,
            "capillary_curvature_relative_error": 0.1,
        },
    ]

    errors = smoke.capillary_convergence_rate_errors(probes, args)

    assert len(errors) == 1
    assert "usable sample" in errors[0]


def test_solver_elapsed_time_errors_reject_runtime_budget_overrun():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_solver_elapsed_wall_seconds=10.0,
        max_solver_elapsed_seconds_per_accepted_step=1.0,
    )
    metrics = {
        "solver_elapsed_wall_seconds": 12.0,
        "result_step": 6,
    }

    errors = smoke.solver_elapsed_time_errors(metrics, args)

    assert len(errors) == 2
    assert any("wall time" in error for error in errors)
    assert any("per accepted step" in error for error in errors)
    assert metrics["solver_elapsed_seconds_per_accepted_step"] == 2.0


def test_time_loop_convergence_errors_reject_iteration_ceiling_overrun():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_time_loop_convergence=True,
        enable_adaptive_time_loop=False,
        max_time_loop_nonlinear_iterations_per_step=3,
        max_time_loop_linear_iterations_per_step=10,
    )
    metrics = {
        "steps": 5,
        "time_loop": {
            "summary": {
                "accepted_steps": 4,
                "all_nonlinear_converged": False,
                "all_linear_converged": False,
                "nonlinear_iterations_max": 4,
                "linear_iterations_max": 11,
            }
        },
    }

    errors = smoke.time_loop_convergence_errors(metrics, args)

    assert len(errors) == 5
    assert any("accepted steps" in error for error in errors)
    assert any("not all nonlinear" in error for error in errors)
    assert any("not all linear" in error for error in errors)
    assert any("nonlinear iterations" in error for error in errors)
    assert any("linear iterations" in error for error in errors)


def test_time_loop_outer_fixed_point_uses_explicit_iteration_ceiling_names():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_time_loop_convergence=True,
        enable_adaptive_time_loop=False,
        max_time_loop_nonlinear_iterations_per_step=None,
        max_time_loop_linear_iterations_per_step=None,
        max_time_loop_outer_iterations_per_step=4,
        max_time_loop_inner_iterations_total_per_step=12,
    )
    metrics = {
        "steps": 1,
        "time_loop": {
            "summary": {
                "accepted_steps": 1,
                "all_nonlinear_converged": True,
                "all_linear_converged": True,
                "outer_iterations_max": 5,
                "inner_iterations_total_max": 14,
            }
        },
    }

    errors = smoke.time_loop_convergence_errors(metrics, args)

    assert len(errors) == 2
    assert any("outer iterations" in error for error in errors)
    assert any("total inner iterations" in error for error in errors)


def test_adaptive_failure_uses_solver_controls_for_requested_horizon():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        require_time_loop_convergence=True,
        enable_adaptive_time_loop=True,
        max_time_loop_nonlinear_iterations_per_step=None,
        max_time_loop_linear_iterations_per_step=None,
    )
    metrics = {
        "solver_controls": {
            "time_stepping": {
                "number_of_time_steps": 562,
                "time_step_size": 5.0e-4,
            }
        },
        "time_loop": {
            "summary": {
                "accepted_steps": 1,
                "final_accepted_time": 5.0e-4,
                "all_nonlinear_converged": True,
                "all_linear_converged": True,
                "nonlinear_iterations_max": 1,
                "linear_iterations_max": 2,
            }
        },
    }

    errors = smoke.time_loop_convergence_errors(metrics, args)

    assert len(errors) == 2
    assert any("accepted steps 1 below requested steps 562" in error
               for error in errors)
    assert any("final accepted time 0.0005 below requested time 0.281" in error
               for error in errors)


def test_sessile_physical_errors_accept_solution_derived_metrics_within_bounds():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_sessile_contact_angle_error_degrees=5.0,
        max_sessile_pressure_jump_relative_error=0.15,
        max_sessile_liquid_area_relative_error=0.05,
        max_sessile_parasitic_capillary_number=1.0e-2,
        require_ren_e_speed_sign=True,
        max_ren_e_speed_relative_error=0.50,
    )
    metrics = {
        "sessile_final_contact_angle_absolute_error_degrees": 2.0,
        "sessile_final_contact_angle_source": (
            "same_state_LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots"),
        "sessile_final_pressure_jump_relative_error": 0.10,
        "sessile_final_liquid_area_relative_error": 0.03,
        "sessile_final_parasitic_capillary_number": 5.0e-3,
        "ren_e_contact_fluid_evaluation_source": (
            smoke.GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE),
        "ren_e_contact_fluid_speed_sign_agrees": True,
        "ren_e_contact_fluid_speed_relative_error": 0.25,
    }

    assert smoke.sessile_physical_errors(metrics, args) == []


def test_sessile_physical_errors_reject_each_out_of_bound_or_wrong_sign_metric():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_sessile_contact_angle_error_degrees=5.0,
        max_sessile_pressure_jump_relative_error=0.15,
        max_sessile_liquid_area_relative_error=0.05,
        max_sessile_parasitic_capillary_number=1.0e-2,
        require_ren_e_speed_sign=True,
        max_ren_e_speed_relative_error=0.50,
    )
    metrics = {
        "sessile_final_contact_angle_absolute_error_degrees": 6.0,
        "sessile_final_contact_angle_source": (
            "same_state_LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots"),
        "sessile_final_pressure_jump_relative_error": 0.16,
        "sessile_final_liquid_area_relative_error": 0.06,
        "sessile_final_parasitic_capillary_number": 2.0e-2,
        "ren_e_contact_fluid_evaluation_source": (
            smoke.GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE),
        "ren_e_contact_fluid_speed_sign_agrees": False,
        "ren_e_contact_fluid_speed_relative_error": 0.60,
    }

    errors = smoke.sessile_physical_errors(metrics, args)

    assert len(errors) == 6
    assert any("contact-angle" in error for error in errors)
    assert any("pressure-jump" in error for error in errors)
    assert any("liquid-area" in error for error in errors)
    assert any("parasitic capillary" in error for error in errors)
    assert any("speed relative error" in error for error in errors)
    assert any("advancing/receding sign" in error for error in errors)


def test_dynamic_contact_operator_diagnostic_is_parsed_and_promoted():
    smoke = _load_smoke_module()
    output = (
        "[INFO] DynamicContactAngle operator-consistent contact geometry "
        "diagnostic=dynamic_contact_operator_angle status=available "
        "normal_source=generated_interface_rule_geometry samples=2 "
        "mean_dynamic_cos=0.1032814350927828 "
        "mean_young_gap=-0.1032814350927828 "
        "min_wall_tangential_normal_norm=0.994652172955539 "
        "transversality_satisfied=true"
    )
    diagnostics = smoke.parse_solver_diagnostics(output)
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)

    assert len(diagnostics["dynamic_contact_operator_angles"]) == 1
    record = diagnostics["dynamic_contact_operator_angles"][0]
    assert record["status"] == "available"
    assert math.isclose(record["mean_dynamic_cos"], 0.1032814350927828)
    assert record["transversality_satisfied"] is True
    assert metrics["diagnostic_dynamic_contact_operator_angle_count"] == 1
    assert metrics[
        "diagnostic_dynamic_contact_operator_angle_available_count"] == 1
    assert metrics["latest_dynamic_contact_operator_angle"] == record
    assert math.isclose(
        metrics[
            "diagnostic_dynamic_contact_operator_angle_min_wall_tangential_normal_norm"
        ], 0.994652172955539)


def test_sessile_physical_errors_do_not_accept_missing_solved_field_metric():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        max_sessile_contact_angle_error_degrees=None,
        max_sessile_pressure_jump_relative_error=0.15,
        max_sessile_liquid_area_relative_error=None,
        max_sessile_parasitic_capillary_number=None,
        require_ren_e_speed_sign=False,
        max_ren_e_speed_relative_error=None,
    )

    errors = smoke.sessile_physical_errors({}, args)

    assert errors == [
        "sessile pressure-jump relative error is unavailable from the solved time history"
    ]


def test_output_free_run_cannot_bypass_enabled_sessile_physical_gate():
    smoke = _load_smoke_module()

    class DefaultNoneArgs(argparse.Namespace):
        def __getattr__(self, _name):
            return None

    args = DefaultNoneArgs(
        max_sessile_contact_angle_error_degrees=None,
        max_sessile_pressure_jump_relative_error=0.15,
        max_sessile_liquid_area_relative_error=None,
        max_sessile_parasitic_capillary_number=None,
        require_ren_e_speed_sign=False,
        max_ren_e_speed_relative_error=None,
    )
    metrics = {
        "case": "sessile2d",
        "diagnostics": {},
        "output_metrics_skipped": True,
        "output_metrics_skip_reason": "VTK output disabled",
    }

    errors = smoke.evaluate(metrics, args)

    assert (
        "sessile pressure-jump relative error is unavailable from the solved time history"
        in errors
    )


def test_fsils_accepted_true_residual_gate_accepts_explicit_small_inexact_solve():
    smoke = _load_smoke_module()
    args = argparse.Namespace(max_fsils_accepted_true_residual_norm=1.0e-9)
    metrics = {
        "diagnostics": {
            "fsils_solve_summaries": [
                {"converged": 0, "final_residual_norm": 5.0e-10},
                {"converged": 1, "final_residual_norm": 2.0e-8},
            ]
        }
    }

    assert smoke.fsils_accepted_true_residual_errors(metrics, args) == []


def test_fsils_accepted_true_residual_gate_rejects_large_or_missing_residual():
    smoke = _load_smoke_module()
    args = argparse.Namespace(max_fsils_accepted_true_residual_norm=1.0e-9)
    metrics = {
        "diagnostics": {
            "fsils_solve_summaries": [
                {"converged": 0, "final_residual_norm": 2.0e-9},
                {"converged": False},
            ]
        }
    }

    errors = smoke.fsils_accepted_true_residual_errors(metrics, args)

    assert len(errors) == 2
    assert any("2e-09 exceeds 1e-09" in error for error in errors)
    assert any("no finite assembled true residual" in error for error in errors)


def test_fsils_accepted_true_residual_gate_rejects_invalid_threshold():
    smoke = _load_smoke_module()
    args = argparse.Namespace(max_fsils_accepted_true_residual_norm=-1.0)

    assert smoke.fsils_accepted_true_residual_errors({}, args) == [
        "maximum accepted FSILS true residual must be finite and nonnegative"
    ]


def test_fsils_diag_col_mismatch_is_parsed_aggregated_and_gated():
    smoke = _load_smoke_module()
    solver_output = "\n".join((
        "[INFO] FsilsLinearSolver: prepared matrix diagnostics "
        "diagnostic=fsils_prepared_matrix rows=324 zero_rows=0 "
        "missing_diag=0 diag_col_mismatch=0 duplicate_diag_entries=0 "
        "duplicate_diag_rows=0 zero_diag=0",
        "[INFO] FsilsLinearSolver: prepared matrix diagnostics "
        "diagnostic=fsils_prepared_matrix rows=324 zero_rows=0 "
        "missing_diag=0 diag_col_mismatch=2 duplicate_diag_entries=3 "
        "duplicate_diag_rows=2 zero_diag=0",
    ))

    diagnostics = smoke.parse_solver_diagnostics(solver_output)
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)

    assert diagnostics["fsils_prepared_matrices"][0]["diag_col_mismatch"] == 0
    assert diagnostics["fsils_prepared_matrices"][1]["diag_col_mismatch"] == 2
    assert metrics[
        "diagnostic_fsils_prepared_matrix_max_diag_col_mismatch"
    ] == 2
    assert metrics[
        "diagnostic_fsils_prepared_matrix_max_duplicate_diag_entries"
    ] == 3
    assert metrics[
        "diagnostic_fsils_prepared_matrix_max_duplicate_diag_rows"
    ] == 2
    assert smoke.fsils_matrix_diag_col_mismatch_errors(
        metrics,
        argparse.Namespace(max_fsils_matrix_diag_col_mismatch=2),
    ) == []
    assert smoke.fsils_matrix_diag_col_mismatch_errors(
        metrics,
        argparse.Namespace(max_fsils_matrix_diag_col_mismatch=0),
    ) == [
        "FSILS prepared-matrix diagonal-column mismatches 2 exceed 0"
    ]
    assert smoke.fsils_matrix_duplicate_diag_errors(
        metrics,
        argparse.Namespace(
            max_fsils_matrix_duplicate_diag_entries=3,
            max_fsils_matrix_duplicate_diag_rows=2,
        ),
    ) == []
    assert smoke.fsils_matrix_duplicate_diag_errors(
        metrics,
        argparse.Namespace(
            max_fsils_matrix_duplicate_diag_entries=0,
            max_fsils_matrix_duplicate_diag_rows=0,
        ),
    ) == [
        "FSILS prepared-matrix duplicate diagonal entries 3 exceed 0",
        "FSILS prepared-matrix rows with duplicate diagonals 2 exceed 0",
    ]


def test_fsils_diag_col_mismatch_gate_requires_metric_when_enabled():
    smoke = _load_smoke_module()

    assert smoke.fsils_matrix_diag_col_mismatch_errors(
        {},
        argparse.Namespace(max_fsils_matrix_diag_col_mismatch=0),
    ) == [
        "FSILS prepared-matrix diagonal-column mismatch diagnostics are unavailable"
    ]


def test_fsils_duplicate_diag_gate_fails_closed_for_legacy_parser_record():
    smoke = _load_smoke_module()
    diagnostics = smoke.parse_solver_diagnostics(
        "[INFO] diagnostic=fsils_prepared_matrix rows=324 zero_diag=0"
    )
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, diagnostics)

    assert smoke.fsils_matrix_duplicate_diag_errors(
        metrics,
        argparse.Namespace(
            max_fsils_matrix_duplicate_diag_entries=0,
            max_fsils_matrix_duplicate_diag_rows=0,
        ),
    ) == [
        "FSILS prepared-matrix duplicate diagonal entries diagnostics are unavailable",
        "FSILS prepared-matrix rows with duplicate diagonals diagnostics are unavailable",
    ]


def test_qualification_log_uses_compact_diagnostic_evidence(tmp_path):
    smoke = _load_smoke_module()
    run_dir = tmp_path / "preserved_run"
    diagnostics = {
        "counts": {
            "cut_volume_assemblies": 240,
            "free_surface_conservative_balances": 120,
        },
        "cut_volume_assemblies": [
            {"payload": "x" * 4096},
            {"payload": "y" * 4096},
        ],
    }
    probe = {
        "case": "sessile2d",
        "run_dir": str(run_dir),
        "passed": True,
        "diagnostics": diagnostics,
        "diagnostic_free_surface_conservative_balance_count": 120,
    }
    output = tmp_path / "qualification.json"

    smoke.write_qualification_log(
        output,
        tmp_path / "solver",
        [probe],
        complete=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized_probe = payload["probes"][0]
    assert payload["schema_version"] == 2
    assert payload["complete"] is True
    assert "diagnostics" not in serialized_probe
    assert serialized_probe[
        "diagnostic_free_surface_conservative_balance_count"
    ] == 120
    assert serialized_probe["diagnostic_evidence"] == {
        "full_records_embedded": False,
        "record_counts": diagnostics["counts"],
        "retention_requires_preserve_run_dir": True,
        "solver_log_path": str(run_dir / "solver_run.log"),
    }
    assert probe["diagnostics"] is diagnostics
    assert output.stat().st_size < 4096
