import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "cases/fluid/open_vessel_free_surface/static_capillary_3d.py"
)
SPEC = importlib.util.spec_from_file_location("static_capillary_3d", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cases = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cases)


@pytest.mark.parametrize("angle", [30.0, 60.0, 90.0, 120.0, 150.0])
def test_spherical_cap_reference_geometry(angle):
    radius = 0.25
    geometry = cases.spherical_cap_geometry(angle, radius)
    radians = math.radians(angle)
    height = radius * (1.0 - math.cos(radians))
    base_radius = radius * math.sin(radians)
    assert geometry["height"] == pytest.approx(height)
    assert geometry["base_radius"] == pytest.approx(base_radius)
    assert geometry["volume"] == pytest.approx(
        math.pi * height * height * (radius - height / 3.0))
    assert geometry["liquid_gas_area"] == pytest.approx(
        2.0 * math.pi * radius * height)
    assert geometry["wetted_wall_area"] == pytest.approx(
        math.pi * base_radius * base_radius)
    assert geometry["contact_line_measure"] == pytest.approx(
        2.0 * math.pi * base_radius)


def test_closed_sphere_case_writes_conforming_tetrahedral_contract(tmp_path):
    case_dir = tmp_path / "sphere"
    cases.write_sphere_case(
        case_dir,
        steps=1,
        resolution=8,
        radius=0.25,
        surface_tension=1.0,
        time_step_size=0.001,
        level_set_positive_scale=4.0,
    )
    grid = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    assert grid.n_points == 9 ** 3
    assert grid.n_cells == 6 * 8 ** 3
    assert set(np.asarray(grid.celltypes, dtype=int)) == {
        int(pv.CellType.TETRA)}
    points = np.asarray(grid.points, dtype=float)
    expected_phi = 4.0 * (
        np.linalg.norm(points - np.asarray([0.5, 0.5, 0.5]), axis=1) -
        0.25
    )
    assert np.allclose(grid.point_data["phi"], expected_phi)
    assert np.allclose(grid.point_data["Pressure"], 8.0)
    assert np.allclose(grid.point_data["Velocity"], 0.0)

    for wall in cases.BOX_WALLS:
        surface = pv.read(
            case_dir / f"mesh/background/mesh-surfaces/{wall}.vtu")
        assert surface.n_cells == 2 * 8 ** 2
        assert set(np.asarray(surface.celltypes, dtype=int)) == {
            int(pv.CellType.TRIANGLE)}
        frame = cases.axis_wall_frame(wall)
        coordinates = np.asarray(surface.points)[:, frame["wall_axis"]]
        assert np.allclose(coordinates, frame["wall_coordinate"])

    benchmark = json.loads((case_dir / "benchmark.json").read_text())
    assert benchmark["spatial_dimension"] == 3
    assert benchmark["capillary_curvature_factor"] == 2.0
    assert benchmark["initial_active_pressure"] == 8.0
    assert benchmark["pressure_gauge"]["constraint_applied"] is False
    state = cases.spatial_capillary_state_metrics(grid, benchmark)
    assert state["available"] is True
    assert state["fitted_sphere_radius"] == pytest.approx(0.25, abs=0.01)
    assert state["fitted_sphere_rmse"] < 0.01
    assert state["liquid_volume"] == pytest.approx(
        benchmark["expected_liquid_volume"], rel=0.15)
    assert state["liquid_gas_area"] == pytest.approx(
        benchmark["expected_liquid_gas_area"], rel=0.10)

    solver = (case_dir / "solver.xml").read_text()
    assert "<Number_of_spatial_dimensions>3" in solver
    assert "<Generated_interface_geometry>LinearCorner" in solver
    assert "<Surface_tension_form>SurfaceStress" in solver
    assert "<Node_pressure_constraints>" not in solver


def test_closed_sphere_active_side_and_cut_offset_are_physical_invariants(
        tmp_path):
    offset = np.asarray([0.03125, -0.0625, 0.025], dtype=float)
    grids = []
    states = []
    for label, active_domain in (
            ("negative", "LevelSetNegative"),
            ("positive", "LevelSetPositive")):
        case_dir = tmp_path / label
        cases.write_sphere_case(
            case_dir,
            steps=1,
            resolution=6,
            radius=0.25,
            surface_tension=0.5,
            time_step_size=0.001,
            level_set_positive_scale=2.0,
            active_domain=active_domain,
            center_offset=offset,
        )
        benchmark = json.loads((case_dir / "benchmark.json").read_text())
        grid = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
        phi = np.asarray(grid.point_data["phi"], dtype=float)
        gauge_node = int(benchmark["pressure_gauge"]["node_id"])
        liquid_signed_phi = cases.active_signed_level_set(
            phi, active_domain)

        assert benchmark["active_domain"] == active_domain
        assert benchmark["sphere_center_offset"] == pytest.approx(offset)
        assert benchmark["sphere_center"] == pytest.approx(0.5 + offset)
        assert liquid_signed_phi[gauge_node] < 0.0
        assert f"<Active_domain>{active_domain}" in (
            case_dir / "solver.xml").read_text()

        state = cases.spatial_capillary_state_metrics(grid, benchmark)
        assert state["available"] is True
        assert state["active_domain"] == active_domain
        grids.append(grid)
        states.append(state)

    assert np.allclose(
        grids[0].point_data["phi"], -grids[1].point_data["phi"])
    for metric in (
            "fitted_sphere_radius",
            "fitted_sphere_rmse",
            "liquid_volume",
            "liquid_gas_area",
            "max_liquid_speed",
            "mean_liquid_speed"):
        assert float(states[0][metric]) == pytest.approx(
            float(states[1][metric]), abs=2.0e-8)


def test_sessile_sphere_rotates_to_every_wall(tmp_path):
    records = {}
    for wall in cases.BOX_WALLS:
        case_dir = tmp_path / wall
        cases.write_sessile_sphere_case(
            case_dir,
            steps=1,
            resolution=6,
            contact_angle_degrees=60.0,
            radius=0.25,
            surface_tension=1.0,
            time_step_size=0.001,
            wall_face=wall,
            level_set_positive_scale=0.25,
        )
        benchmark = json.loads((case_dir / "benchmark.json").read_text())
        contact = benchmark["sessile_contact"]
        frame = cases.axis_wall_frame(wall)
        assert contact["wall"] == wall
        assert contact["wall_axis"] == frame["wall_axis"]
        assert contact["wall_coordinate"] == frame["wall_coordinate"]
        assert tuple(contact["wall_normal"]) == frame["wall_normal"]
        assert tuple(contact["wall_tangent_axes"]) == frame["wall_tangent_axes"]

        grid = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
        state = cases.spatial_capillary_state_metrics(grid, benchmark)
        assert state["available"] is True
        assert state["operator_contact_geometry_available"] is True
        assert state["operator_dynamic_angle_degrees_mean"] == pytest.approx(
            60.0, abs=8.0)
        assert state["contact_angle_degrees"] == pytest.approx(60.0, abs=4.0)
        assert state["base_radius"] == pytest.approx(
            contact["expected_initial_base_radius"], rel=0.08)
        assert state["apex_height"] == pytest.approx(
            contact["expected_initial_apex_height"], abs=1.0e-12)

        solver = (case_dir / "solver.xml").read_text()
        assert f"<Contact_line_wall_face>{wall}" in solver
        assert (
            f"<Effective_direction>{frame['effective_direction']}" in solver)
        records[wall] = state

    for metric, tolerance in (
            ("liquid_volume", 2.0e-15),
            ("liquid_gas_area", 3.0e-9),
            ("base_radius", 1.0e-9),
            ("contact_line_measure", 8.0e-9),
            ("operator_dynamic_angle_degrees_mean", 4.0e-6)):
        values = [float(record[metric]) for record in records.values()]
        assert max(values) - min(values) < tolerance


def test_sessile_active_side_and_tangent_offset_are_physical_invariants(
        tmp_path):
    offset = np.asarray([0.05, -0.075], dtype=float)
    grids = []
    states = []
    for label, active_domain in (
            ("negative", "LevelSetNegative"),
            ("positive", "LevelSetPositive")):
        case_dir = tmp_path / label
        cases.write_sessile_sphere_case(
            case_dir,
            steps=1,
            resolution=6,
            contact_angle_degrees=60.0,
            radius=0.25,
            surface_tension=0.5,
            time_step_size=0.001,
            wall_face="wall_right",
            active_domain=active_domain,
            tangent_center_offset=offset,
        )
        benchmark = json.loads((case_dir / "benchmark.json").read_text())
        contact = benchmark["sessile_contact"]
        grid = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
        phi = np.asarray(grid.point_data["phi"], dtype=float)
        gauge_node = int(benchmark["pressure_gauge"]["node_id"])

        assert benchmark["active_domain"] == active_domain
        assert benchmark["tangent_center_offset"] == pytest.approx(offset)
        assert contact["active_domain"] == active_domain
        assert np.asarray(contact["circle_center"])[[1, 2]] == pytest.approx(
            0.5 + offset)
        assert contact["circle_center"][0] == pytest.approx(1.125)
        assert cases.active_signed_level_set(
            phi, active_domain)[gauge_node] < 0.0
        assert f"<Active_domain>{active_domain}" in (
            case_dir / "solver.xml").read_text()

        state = cases.spatial_capillary_state_metrics(grid, benchmark)
        assert state["available"] is True
        assert state["active_domain"] == active_domain
        grids.append(grid)
        states.append(state)

    assert np.allclose(
        grids[0].point_data["phi"], -grids[1].point_data["phi"])
    for metric in (
            "fitted_sphere_radius",
            "liquid_volume",
            "liquid_gas_area",
            "base_radius",
            "apex_height",
            "contact_line_measure",
            "operator_dynamic_angle_degrees_mean"):
        assert float(states[0][metric]) == pytest.approx(
            float(states[1][metric]), abs=1.0e-7)


def test_sessile_sphere_metrics_are_invariant_to_positive_scale(tmp_path):
    states = []
    for label, scale in (("small", 0.25), ("large", 4.0)):
        case_dir = tmp_path / label
        cases.write_sessile_sphere_case(
            case_dir,
            steps=1,
            resolution=6,
            contact_angle_degrees=60.0,
            radius=0.25,
            surface_tension=1.0,
            time_step_size=0.001,
            wall_face="wall_bottom",
            level_set_positive_scale=scale,
        )
        benchmark = json.loads((case_dir / "benchmark.json").read_text())
        grid = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
        states.append(cases.spatial_capillary_state_metrics(grid, benchmark))

    for metric in (
            "fitted_sphere_radius",
            "liquid_volume",
            "liquid_gas_area",
            "base_radius",
            "apex_height",
            "contact_line_measure",
            "operator_dynamic_angle_degrees_mean"):
        assert float(states[0][metric]) == pytest.approx(
            float(states[1][metric]), abs=2.0e-12)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_spatial_cases_reject_invalid_level_set_scale(tmp_path, scale):
    with pytest.raises(ValueError, match="positive scale"):
        cases.write_sphere_case(
            tmp_path / "invalid",
            1,
            4,
            0.25,
            1.0,
            0.001,
            scale,
        )


@pytest.mark.parametrize("active_domain", ["negative", "", "liquid"])
def test_spatial_cases_reject_unknown_active_domain(tmp_path, active_domain):
    with pytest.raises(ValueError, match="active domain"):
        cases.write_sphere_case(
            tmp_path / "invalid",
            steps=1,
            resolution=4,
            radius=0.25,
            surface_tension=1.0,
            time_step_size=0.001,
            active_domain=active_domain,
        )


@pytest.mark.parametrize(
    "offset",
    [
        (0.0, 0.0),
        (0.0, 0.0, math.inf),
        (0.26, 0.0, 0.0),
    ],
)
def test_closed_sphere_rejects_invalid_center_offset(tmp_path, offset):
    with pytest.raises(ValueError, match="sphere center offset|closed sphere"):
        cases.write_sphere_case(
            tmp_path / "invalid",
            steps=1,
            resolution=4,
            radius=0.25,
            surface_tension=1.0,
            time_step_size=0.001,
            center_offset=offset,
        )


@pytest.mark.parametrize(
    "offset",
    [
        (0.0,),
        (0.0, math.nan),
        (0.3, 0.0),
    ],
)
def test_sessile_sphere_rejects_invalid_tangent_offset(tmp_path, offset):
    with pytest.raises(
            ValueError,
            match="tangent-center offset|contact line"):
        cases.write_sessile_sphere_case(
            tmp_path / "invalid",
            steps=1,
            resolution=4,
            contact_angle_degrees=60.0,
            radius=0.25,
            surface_tension=1.0,
            time_step_size=0.001,
            wall_face="wall_bottom",
            tangent_center_offset=offset,
        )
