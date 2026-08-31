from __future__ import annotations

import importlib.util
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tests/cases/fluid/open_vessel_free_surface/capillary_rise_2d.py"
)
RUNNER_PATH = (
    ROOT /
    "tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "free_surface_capillary_rise_2d", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    module_dir = str(RUNNER_PATH.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(
        "free_surface_capillary_rise_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def boundary_by_name(equation: ET.Element, name: str) -> ET.Element:
    matches = [
        boundary for boundary in equation.findall("Add_BC")
        if boundary.attrib.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def equation_by_type(root: ET.Element, equation_type: str) -> ET.Element:
    matches = [
        equation for equation in root.findall("Add_equation")
        if equation.attrib.get("type") == equation_type
    ]
    assert len(matches) == 1
    return matches[0]


def test_published_initial_geometry_values_are_reproduced():
    module = load_module()
    geometry = module.initial_geometry()
    assert math.isclose(
        geometry["circle_radius_m"],
        0.005773502691896257,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        geometry["mean_meniscus_sag_m"],
        0.0008394685149335339,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        geometry["apex_height_m"],
        0.009160531485066466,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        module.initial_closed_inlet_pressure_offset_pa(),
        -3.46293323027551,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )


def test_generated_case_uses_open_resolved_slip_contract(tmp_path: Path):
    module = load_module()
    case_dir = tmp_path / "capillaryrise2d"
    benchmark = module.write_case(case_dir, 3, 5.0e-4, 10)

    resolution = benchmark["mesh_resolution"]
    assert resolution["half_gap_cells"] == 10
    assert resolution["height_cells"] == 80
    assert resolution["triangle_count"] == 1600
    assert math.isclose(resolution["dx_m"], 5.0e-4)
    assert math.isclose(resolution["dy_m"], 5.0e-4)
    assert math.isclose(resolution["slip_length_to_dx"], 2.0)
    discrete = resolution["discrete_initial_geometry"]
    assert math.isclose(
        discrete["trapezoidal_liquid_area_m2"],
        discrete["target_liquid_area_m2"],
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )
    capillary = benchmark["capillary_rise"]
    assert capillary["initial_pressure_model"] == (
        "closed_inlet_circular_capillary_hydrostatic_preload"
    )
    assert math.isclose(
        capillary["initial_pressure_offset_pa"],
        module.initial_closed_inlet_pressure_offset_pa(),
    )

    initial = pv.read(
        case_dir / "mesh/background/mesh-complete.mesh.vtu")
    points = np.asarray(initial.points, dtype=float)
    pressure = np.asarray(initial.point_data["Pressure"], dtype=float)
    bottom = np.isclose(points[:, 1], 0.0)
    assert np.allclose(
        pressure[bottom],
        module.initial_closed_inlet_pressure_offset_pa(),
        rtol=0.0,
        atol=1.0e-14,
    )

    root = ET.parse(case_dir / "solver.xml").getroot()
    general = root.find("GeneralSimulationParameters")
    assert general is not None
    assert general.findtext("Number_of_time_steps") == "3"
    assert general.findtext("Time_step_size") == "0.0005"

    level_set = equation_by_type(root, "level_set")
    assert level_set.findtext("Velocity_source") == "prescribed_data"
    assert level_set.findtext("Use_wet_extension_advection_velocity") == "true"
    assert (
        level_set.findtext("Wet_extension_advection_velocity_method") ==
        "wall_compatible_normal"
    )
    assert level_set.findtext("Enable_reinitialization") == "true"
    assert level_set.findtext("Reinitialization_cadence_steps") == "1"
    assert level_set.findtext("Enable_volume_correction") == "false"
    assert boundary_by_name(
        level_set, "wall_bottom").findtext("Type") == "LevelSetInflow"
    assert float(boundary_by_name(
        level_set, "wall_bottom").findtext("Value")) < 0.0
    assert boundary_by_name(
        level_set, "wall_top").findtext("Type") == "LevelSetOutflow"

    fluid = equation_by_type(root, "fluid")
    assert math.isclose(float(fluid.findtext("Density")), 83.1)
    assert math.isclose(float(fluid.findtext("Force_y")), -4.17)
    assert fluid.findtext("Hydrostatic_pressure_initialization") == "false"
    assert fluid.find("Node_pressure_constraints") is None
    for wall_name in ("wall_left", "wall_right"):
        wall = boundary_by_name(fluid, wall_name)
        assert wall.findtext("Type") == "Dir"
        assert wall.findtext("Effective_direction") == "1 0"
    bottom = boundary_by_name(fluid, "wall_bottom")
    assert bottom.findtext("Type") == "Neu"
    assert bottom.findtext("Time_dependence") == "Steady"
    assert math.isclose(float(bottom.findtext("Value")), 0.0)

    free_surface = boundary_by_name(fluid, "free_surface")
    assert free_surface.findtext("Implementation") == "UnfittedLevelSet"
    assert free_surface.findtext("Active_domain") == "LevelSetNegative"
    assert free_surface.findtext("Active_domain_method") == "CutVolume"
    assert free_surface.findtext("Generated_interface_geometry") == "LinearCorner"
    assert free_surface.findtext("Surface_tension_form") == "SurfaceStress"
    assert free_surface.findtext("Contact_line_model") == "PrescribedContactAngle"
    assert free_surface.findtext("Contact_line_wall_face") == "wall_right"
    assert free_surface.findtext("Contact_line_wall_normal") == "1 0 0"
    assert math.isclose(
        float(free_surface.findtext("Contact_angle_degrees")), 30.0)
    assert free_surface.findtext("Wall_slip_model") == "Navier"
    assert math.isclose(
        float(free_surface.findtext("Wall_slip_length")), 0.001)
    assert math.isclose(
        float(free_surface.findtext("Active_domain_smoothing_width")), 0.0)


def test_runner_freezes_capillary_rise_physics_and_solver_defaults():
    runner = load_runner()
    base = argparse.Namespace(
        high_order_mpi_production_qualification=False,
        use_high_order_implicit_cuts=False,
        level_set_active_domain="LevelSetNegative",
        capillary_force_form="surface_stress",
        surface_tension=None,
        enable_level_set_reinitialization=None,
        enable_level_set_volume_correction=None,
        max_capillary_rise_contact_motion_cells_per_step=None,
        max_capillary_rise_contact_angle_error_degrees=None,
        steps=None,
        time_step_size=None,
        timeout_seconds=None,
        wet_extension_advection_velocity_method=None,
        linear_solver_type=None,
        linear_algebra_backend=None,
        linear_preconditioner=None,
        linear_max_iterations=None,
        linear_krylov_space_dimension=None,
        linear_relative_tolerance=None,
        linear_absolute_tolerance=None,
    )

    configured = runner.case_args_for_run("capillaryrise2d", base)

    assert configured.steps == 1
    assert configured.time_step_size == 5.0e-4
    assert configured.timeout_seconds == 600.0
    assert configured.surface_tension == runner.CAPILLARY_RISE_SURFACE_TENSION
    assert configured.contact_angle_degrees == 30.0
    assert configured.wall_slip_length == 0.001
    assert configured.enable_level_set_reinitialization is True
    assert configured.enable_level_set_volume_correction is False
    assert configured.wet_extension_advection_velocity_method == (
        "wall_compatible_normal"
    )
    assert configured.enable_physical_history_instrumentation is True
    assert configured.require_time_loop_convergence is True
    assert configured.disable_velocity_extension is True
    assert configured.linear_solver_type == "gmres"
    assert configured.linear_algebra_backend == "fsils"
    assert configured.linear_preconditioner == "rcs"

    invalid = argparse.Namespace(**vars(base))
    invalid.surface_tension = 0.05
    with pytest.raises(ValueError, match="fixed by the reference"):
        runner.case_args_for_run("capillaryrise2d", invalid)


def test_initial_discrete_observables_match_angle_volume_and_slip_levels(
        tmp_path: Path):
    module = load_module()
    for half_gap_cells, expected_ratio in ((10, 2.0), (20, 4.0), (40, 8.0)):
        case_dir = tmp_path / f"n{half_gap_cells}"
        benchmark = module.write_case(
            case_dir, 1, 5.0e-4, half_gap_cells)
        dataset = pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        assert np.all(dataset.celltypes == int(pv.CellType.TRIANGLE))
        state = module.state_metrics(dataset, benchmark)
        assert state["available"] is True
        assert math.isclose(
            benchmark["mesh_resolution"]["slip_length_to_dx"],
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        assert math.isclose(
            state["contact_angle_degrees"],
            30.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        )
        discrete = benchmark["capillary_rise"]["discrete_initial_geometry"]
        assert math.isclose(
            state["apex_height_m"], discrete["apex_height_m"],
            rel_tol=0.0, abs_tol=1.0e-14)
        assert math.isclose(
            state["wall_contact_height_m"],
            discrete["wall_contact_height_m"],
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        assert math.isclose(
            state["sharp_wetted_wall_length_m"],
            state["wall_contact_height_m"],
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        assert state["sharp_wall_slip_dissipation_w_per_m"] == 0.0


def test_sharp_slip_dissipation_is_positive_and_uses_only_wetted_wall(
        tmp_path: Path):
    module = load_module()
    case_dir = tmp_path / "capillaryrise2d"
    benchmark = module.write_case(case_dir, 1, 5.0e-4, 10)
    dataset = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    velocity = np.asarray(dataset.point_data["Velocity"], dtype=float)
    velocity[:, 1] = 0.02
    dataset.point_data["Velocity"] = velocity
    state = module.state_metrics(dataset, benchmark)
    assert state["available"] is True
    expected = (
        module.LIQUID_VISCOSITY_PA_S / module.SLIP_LENGTH_M *
        state["sharp_wetted_wall_length_m"] * 0.02 ** 2
    )
    assert math.isclose(
        state["sharp_wall_slip_dissipation_w_per_m"],
        expected,
        rel_tol=1.0e-13,
        abs_tol=1.0e-18,
    )
    assert math.isclose(state["wall_contact_fluid_speed_m_per_s"], 0.02)


def test_multiple_wall_crossings_fail_closed(tmp_path: Path):
    module = load_module()
    case_dir = tmp_path / "capillaryrise2d"
    benchmark = module.write_case(case_dir, 1, 5.0e-4, 10)
    dataset = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    points = np.asarray(dataset.points, dtype=float)
    phi = np.asarray(dataset.point_data["phi"], dtype=float)
    wall = np.flatnonzero(np.isclose(points[:, 0], module.HALF_GAP_M))
    order = wall[np.argsort(points[wall, 1])]
    phi[order] = np.where(np.arange(order.size) % 2 == 0, -1.0, 1.0)
    dataset.point_data["phi"] = phi
    state = module.state_metrics(dataset, benchmark)
    assert state["available"] is False
    assert "exactly one interface root" in state["error"]


def test_history_gate_requires_complete_nonnegative_sharp_wall_records():
    runner = load_runner()
    state = {
        "available": True,
        "apex_height_m": 0.01,
        "wall_contact_height_m": 0.012,
        "contact_angle_error_degrees": 0.2,
        "sharp_wall_slip_dissipation_w_per_m": 0.0,
    }
    metrics = {
        "case": "capillaryrise2d",
        "capillary_rise_history_available": True,
        "capillary_rise_history": [dict(state), dict(state)],
        "capillary_rise_max_contact_motion_cells_per_step": 0.08,
    }
    args = argparse.Namespace(
        max_capillary_rise_contact_motion_cells_per_step=0.1,
        max_capillary_rise_contact_angle_error_degrees=1.0,
    )
    assert runner.capillary_rise_history_errors(metrics, args) == []

    metrics["capillary_rise_history"][1][
        "sharp_wall_slip_dissipation_w_per_m"] = -1.0
    errors = runner.capillary_rise_history_errors(metrics, args)
    assert any("negative wall-slip dissipation" in error for error in errors)


def test_isolated_wall_wetting_gate_exempts_declared_moving_contact_case():
    runner = load_runner()
    metrics = {
        "case": "capillaryrise2d",
        "benchmark": {"capillary_rise": {}},
        "first_wall_only_false_wet": {"step": 1},
    }
    args = argparse.Namespace(enable_physical_history_instrumentation=True)
    assert runner.false_wall_wet_history_errors(metrics, args) == []
    assert metrics["wall_only_false_wet_gate_applicability"] == (
        "not_applicable_to_intentional_moving_contact_line")
