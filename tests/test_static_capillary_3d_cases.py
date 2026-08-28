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
