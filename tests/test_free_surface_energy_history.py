import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "cases/fluid/open_vessel_free_surface/free_surface_energy.py"
)
SPEC = importlib.util.spec_from_file_location("free_surface_energy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
energy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(energy)

RUNNER_PATH = (
    Path(__file__).resolve().parent
    / "cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "free_surface_energy_runner", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def planar_quad(phi, velocity=None):
    points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    cells = np.asarray([
        4, 0, 1, 4, 3,
        4, 1, 2, 5, 4,
        4, 3, 4, 7, 6,
        4, 4, 5, 8, 7,
    ])
    grid = pv.UnstructuredGrid(
        cells,
        np.full(4, int(pv.CellType.QUAD), dtype=np.uint8),
        points,
    )
    grid.point_data["phi"] = np.asarray(phi, dtype=float)
    if velocity is None:
        velocity = np.zeros((grid.n_points, 3))
    grid.point_data["Velocity"] = np.asarray(velocity, dtype=float)
    return grid


def spatial_cube(phi_function, velocity=(0.0, 0.0, 0.0)):
    grid = pv.ImageData(
        dimensions=(3, 3, 3),
        spacing=(0.5, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0),
    ).cast_to_unstructured_grid()
    points = np.asarray(grid.points, dtype=float)
    grid.point_data["phi"] = np.asarray(phi_function(points), dtype=float)
    grid.point_data["Velocity"] = np.tile(
        np.asarray(velocity, dtype=float), (grid.n_points, 1))
    return grid


def test_planar_interface_wall_measure_and_constant_kinetic_energy():
    grid = planar_quad(grid_phi := [
        -0.25, -0.25, -0.25,
        0.25, 0.25, 0.25,
        0.75, 0.75, 0.75,
    ], velocity=np.tile([2.0, 0.0, 0.0], (9, 1)))
    assert grid_phi
    assert energy.interface_measure_2d(grid) == pytest.approx(1.0)
    assert energy.wetted_axis_wall_measure_2d(grid) == pytest.approx(1.0)
    # Liquid area is 0.25, so 1/2*rho*u^2*A = 1 for rho=2, u=2.
    assert energy.liquid_kinetic_energy_proxy_2d(grid, 2.0) == pytest.approx(1.0)


def test_spatial_interface_wall_area_and_constant_kinetic_energy():
    grid = spatial_cube(
        lambda points: points[:, 2] - 0.25,
        velocity=(2.0, 0.0, 0.0),
    )
    assert energy.interface_measure_3d(grid) == pytest.approx(1.0)
    assert energy.wetted_axis_wall_measure_3d(
        grid, wall_axis=2, wall_coordinate=0.0) == pytest.approx(1.0)
    assert energy.liquid_kinetic_energy_proxy_3d(
        grid, 2.0) == pytest.approx(1.0)

    state = energy.free_surface_energy_state_3d(
        grid,
        density=2.0,
        surface_tension=2.0,
        equilibrium_contact_angle_degrees=60.0,
        wall_axis=2,
        wall_coordinate=0.0,
    )
    assert state["interface_energy"] == pytest.approx(2.0)
    assert state["young_wall_energy"] == pytest.approx(-1.0)
    assert state["kinetic_energy_proxy"] == pytest.approx(1.0)
    assert state["total_energy_proxy"] == pytest.approx(2.0)


def test_energy_states_are_invariant_to_declared_active_side():
    planar_negative = planar_quad([
        -0.25, -0.25, -0.25,
        0.25, 0.25, 0.25,
        0.75, 0.75, 0.75,
    ], velocity=np.tile([2.0, 0.0, 0.0], (9, 1)))
    planar_positive = planar_negative.copy(deep=True)
    planar_positive.point_data["phi"] *= -1.0
    planar_states = (
        energy.free_surface_energy_state_2d(
            planar_negative,
            density=2.0,
            surface_tension=2.0,
            equilibrium_contact_angle_degrees=60.0,
            active_domain="LevelSetNegative",
        ),
        energy.free_surface_energy_state_2d(
            planar_positive,
            density=2.0,
            surface_tension=2.0,
            equilibrium_contact_angle_degrees=60.0,
            active_domain="LevelSetPositive",
        ),
    )

    spatial_negative = spatial_cube(
        lambda points: points[:, 2] - 0.25,
        velocity=(2.0, 0.0, 0.0),
    )
    spatial_positive = spatial_negative.copy(deep=True)
    spatial_positive.point_data["phi"] *= -1.0
    spatial_states = (
        energy.free_surface_energy_state_3d(
            spatial_negative,
            density=2.0,
            surface_tension=2.0,
            equilibrium_contact_angle_degrees=60.0,
            wall_axis=2,
            wall_coordinate=0.0,
            active_domain="LevelSetNegative",
        ),
        energy.free_surface_energy_state_3d(
            spatial_positive,
            density=2.0,
            surface_tension=2.0,
            equilibrium_contact_angle_degrees=60.0,
            wall_axis=2,
            wall_coordinate=0.0,
            active_domain="LevelSetPositive",
        ),
    )

    for states in (planar_states, spatial_states):
        assert states[0]["active_domain"] == "LevelSetNegative"
        assert states[1]["active_domain"] == "LevelSetPositive"
        for metric in (
                "kinetic_energy_proxy",
                "interface_measure",
                "interface_energy",
                "wetted_wall_measure",
                "young_wall_energy",
                "total_energy_proxy"):
            assert float(states[0][metric]) == pytest.approx(
                float(states[1][metric]), abs=1.0e-12)


def test_spatial_partial_wetted_wall_uses_linear_surface_clip():
    grid = spatial_cube(lambda points: points[:, 0] - 0.25)
    assert energy.wetted_axis_wall_measure_3d(
        grid, wall_axis=2, wall_coordinate=0.0) == pytest.approx(0.25)


def test_partial_wetted_wall_trace_uses_linear_crossings():
    points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    cells = np.asarray([4, 0, 1, 4, 3, 4, 1, 2, 5, 4])
    grid = pv.UnstructuredGrid(
        cells,
        np.full(2, int(pv.CellType.QUAD), dtype=np.uint8),
        points,
    )
    grid.point_data["phi"] = np.asarray([1.0, -1.0, 3.0, 1.0, 1.0, 1.0])
    grid.point_data["Velocity"] = np.zeros((6, 3))
    # Wet from x=.25 to x=.625: .25 + .125.
    assert energy.wetted_axis_wall_measure_2d(grid) == pytest.approx(0.375)


def partitioned_wall_grid(shared_phi_left=-1.0, shared_phi_right=-1.0):
    points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.5, 0.0],
    ])
    cells = np.asarray([
        4, 0, 1, 3, 2,
        4, 4, 5, 7, 6,
    ])
    grid = pv.UnstructuredGrid(
        cells,
        np.full(2, int(pv.CellType.QUAD), dtype=np.uint8),
        points,
    )
    grid.point_data["phi"] = np.asarray([
        1.0, shared_phi_left, 1.0, 1.0,
        shared_phi_right, 3.0, 1.0, 1.0,
    ])
    grid.point_data["Velocity"] = np.zeros((8, 3))
    return grid


def test_partition_piece_duplicates_are_canonicalized_on_wall_trace():
    grid = partitioned_wall_grid()
    assert energy.wetted_axis_wall_measure_2d(grid) == pytest.approx(0.375)


def test_partition_piece_duplicate_state_disagreement_fails_closed():
    grid = partitioned_wall_grid(
        shared_phi_left=-1.0,
        shared_phi_right=-0.5,
    )
    with pytest.raises(
            ValueError,
            match="coincident wall vertices carry inconsistent"):
        energy.wetted_axis_wall_measure_2d(grid)


def test_young_wall_energy_and_history_summary_contract():
    grid = planar_quad([
        -0.25, -0.25, -0.25,
        0.25, 0.25, 0.25,
        0.75, 0.75, 0.75,
    ])
    state = energy.free_surface_energy_state_2d(
        grid,
        density=1.0,
        surface_tension=2.0,
        equilibrium_contact_angle_degrees=60.0,
    )
    assert state["interface_energy"] == pytest.approx(2.0)
    assert state["young_wall_energy"] == pytest.approx(-1.0)
    assert state["total_energy_proxy"] == pytest.approx(1.0)

    history = [
        {**state, "step": 0, "time": 0.0},
        {**state, "step": 1, "time": 0.1, "total_energy_proxy": 0.9},
        {**state, "step": 2, "time": 0.2, "total_energy_proxy": 0.91},
    ]
    summary = energy.summarize_energy_history(history)
    assert summary["signed_total_energy_change_proxy"] == pytest.approx(-0.09)
    assert summary["max_positive_step_energy_increment_proxy"] == pytest.approx(0.01)
    assert summary["discrete_energy_theorem_claimed"] is False
    assert energy.energy_history_gate_errors(
        summary,
        max_positive_step_increment_relative=0.02,
        max_above_initial_relative=0.02,
    ) == []


def test_energy_gate_fails_closed_and_reports_growth():
    missing = energy.energy_history_gate_errors(
        {},
        max_positive_step_increment_relative=1.0e-4,
        max_above_initial_relative=1.0e-4,
    )
    assert missing == ["free-surface energy history is unavailable or incomplete"]
    summary = energy.summarize_energy_history([
        {"time": 0.0, "total_energy_proxy": 1.0},
        {"time": 0.1, "total_energy_proxy": 1.01},
    ])
    errors = energy.energy_history_gate_errors(
        summary,
        max_positive_step_increment_relative=1.0e-3,
        max_above_initial_relative=1.0e-3,
    )
    assert len(errors) == 3
    assert any("max_positive_step_energy_increment" in error for error in errors)
    assert any("max_energy_above_initial" in error for error in errors)
    assert any("final total-energy proxy increase" in error for error in errors)


@pytest.mark.parametrize("bad_time", [0.0, math.nan])
def test_history_fails_closed_on_nonincreasing_or_nonfinite_time(bad_time):
    with pytest.raises(ValueError):
        energy.summarize_energy_history([
            {"time": 0.0, "total_energy_proxy": 1.0},
            {"time": bad_time, "total_energy_proxy": 0.9},
        ])


def test_runner_records_and_gates_complete_accepted_energy_history(tmp_path):
    initial = planar_quad([
        -0.25, -0.25, -0.25,
        0.25, 0.25, 0.25,
        0.75, 0.75, 0.75,
    ])
    accepted_path = tmp_path / "result_001.vtu"
    initial.save(accepted_path)
    benchmark = {
        "density": 2.0,
        "surface_tension": 2.0,
        "sessile_contact": {
            "equilibrium_contact_angle_degrees": 60.0,
            "wall_y": 0.0,
        },
    }
    metrics = {}
    runner.add_free_surface_energy_history_metrics(
        metrics,
        benchmark,
        initial,
        [(1, accepted_path)],
        {1: (0.1, 0.1)},
        [],
    )
    assert metrics["free_surface_energy_history_available"] is True
    assert metrics["free_surface_energy_state_count"] == 2
    assert metrics["free_surface_energy_discrete_theorem_claimed"] is False
    args = SimpleNamespace(
        require_free_surface_energy_history=True,
        max_free_surface_energy_positive_step_increment_relative=1.0e-12,
        max_free_surface_energy_above_initial_relative=1.0e-12,
    )
    assert runner.free_surface_energy_history_errors(metrics, args) == []


def test_runner_energy_history_fails_closed_on_missing_accepted_output():
    metrics = {}
    initial = planar_quad([
        -0.25, -0.25, -0.25,
        0.25, 0.25, 0.25,
        0.75, 0.75, 0.75,
    ])
    runner.add_free_surface_energy_history_metrics(
        metrics,
        {
            "capillary_wave": {
                "density": 1.0,
                "surface_tension": 1.0,
            },
        },
        initial,
        [],
        {1: (0.1, 0.1)},
        [],
    )
    assert metrics["free_surface_energy_history_available"] is False
    args = SimpleNamespace(
        require_free_surface_energy_history=True,
        max_free_surface_energy_positive_step_increment_relative=1.0e-4,
        max_free_surface_energy_above_initial_relative=1.0e-4,
    )
    errors = runner.free_surface_energy_history_errors(metrics, args)
    assert len(errors) == 1
    assert "every accepted step" in errors[0]


def test_runner_records_spatial_closed_sphere_energy_history(tmp_path):
    case_dir = tmp_path / "sphere3d"
    runner.write_sphere_case(
        case_dir,
        steps=1,
        resolution=6,
        radius=0.25,
        surface_tension=1.0,
        time_step_size=0.001,
        level_set_positive_scale=1.0,
    )
    initial = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    accepted_path = case_dir / "result_001.vtu"
    initial.save(accepted_path)
    benchmark = json.loads((case_dir / "benchmark.json").read_text())
    metrics = {}

    runner.add_free_surface_energy_history_metrics(
        metrics,
        benchmark,
        initial,
        [(1, accepted_path)],
        {1: (0.001, 0.001)},
        [],
    )

    assert metrics["free_surface_energy_history_available"] is True
    assert metrics["free_surface_energy_history_case"] == "closed_sphere"
    assert metrics["free_surface_energy_state_count"] == 2
    assert metrics[
        "free_surface_energy_signed_total_energy_change_proxy"
    ] == pytest.approx(0.0)


def test_runner_records_planar_closed_circle_energy_history(tmp_path):
    case_dir = tmp_path / "droplet2d"
    runner.write_capillary_droplet2d_case(
        case_dir,
        steps=1,
        nx=8,
        ny=8,
        simplex_mesh=True,
        surface_tension=0.75,
    )
    initial = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    accepted_path = case_dir / "result_001.vtu"
    initial.save(accepted_path)
    benchmark = json.loads((case_dir / "benchmark.json").read_text())
    metrics = {}

    runner.add_free_surface_energy_history_metrics(
        metrics,
        benchmark,
        initial,
        [(1, accepted_path)],
        {1: (0.001, 0.001)},
        [],
    )

    assert metrics["free_surface_energy_history_available"] is True
    assert metrics["free_surface_energy_history_case"] == "closed_circle"
    assert metrics["free_surface_energy_state_count"] == 2
    assert metrics[
        "free_surface_energy_signed_total_energy_change_proxy"
    ] == pytest.approx(0.0)


def test_runner_records_spatial_sessile_history_and_shape_metrics(tmp_path):
    case_dir = tmp_path / "sessile3d"
    runner.write_sessile_sphere_case(
        case_dir,
        steps=1,
        resolution=6,
        contact_angle_degrees=60.0,
        radius=0.25,
        surface_tension=1.0,
        time_step_size=0.001,
        wall_face="wall_bottom",
        level_set_positive_scale=1.0,
    )
    initial = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    initial.save(case_dir / "result_001.vtu")
    benchmark = json.loads((case_dir / "benchmark.json").read_text())
    metrics = {}

    runner.add_physical_time_history_metrics(
        metrics,
        case_dir,
        benchmark,
        initial,
        accepted_steps=[{"step": 1, "time": 0.001, "dt": 0.001}],
    )

    assert metrics["free_surface_energy_history_available"] is True
    assert metrics["free_surface_energy_history_case"] == "sessile_contact"
    assert metrics["sessile_final_contact_angle_source"] == (
        "same_state_LinearCorner_generated_triangle_normal_at_phi_zero_wall_edges"
    )
    assert metrics["sessile_final_contact_angle_absolute_error_degrees"] < 8.0
    assert abs(
        metrics["sessile_final_fitted_sphere_contact_angle_error_degrees"]
    ) < 4.0
    assert "sessile_final_fitted_circle_contact_angle_degrees" not in metrics
    assert metrics["final_sessile_state"]["liquid_volume"] == pytest.approx(
        metrics["initial_sessile_state"]["liquid_volume"])
    for name in (
            "sessile_final_liquid_volume_relative_error",
            "sessile_final_base_radius_relative_error",
            "sessile_final_apex_height_relative_error"):
        assert math.isfinite(metrics[name])
