import importlib.util
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
