"""Tests of the 2D Ren-E benchmark generator and verifier (no solver runs)."""

from __future__ import annotations

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests/cases/fluid/free_surface_benchmarks/ren_e_2d"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"ren_e_2d_{name}", BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_surface(root: ET.Element) -> ET.Element:
    for bc in root.iter("Add_BC"):
        if bc.attrib.get("name") == "free_surface":
            return bc
    raise AssertionError("no free-surface condition")


@pytest.mark.parametrize("level", [8, 16, 32])
def test_slip_ratio_and_schedule(level: int):
    generate = load("generate_case")
    h = generate.RADIUS / level
    assert math.isclose(generate.SLIP_LENGTH / h, {8: 2.0, 16: 4.0, 32: 8.0}[level])
    for divisor, (steps, cadence) in {1: (200, 10), 2: (400, 20), 4: (800, 40)}.items():
        schedule = generate.schedule(divisor)
        assert schedule["steps"] == steps
        assert schedule["output_cadence"] == cadence
        assert math.isclose(schedule["dt"] * steps, generate.END_TIME)
    # dt0 is stable at the finest mesh (capillary limit with the sessile safety factor).
    finest_h = generate.RADIUS / max(generate.LEVELS)
    limit = (1.0 / math.sqrt(2.0)) * math.sqrt(finest_h ** 3 / (2.0 * math.pi))
    assert generate.DT0 < limit


def test_predicted_speed_sign_convention():
    generate = load("generate_case")
    assert generate.predicted_speed(generate.CASES["advancing"]) > 0.0
    assert generate.predicted_speed(generate.CASES["receding"]) < 0.0
    assert generate.predicted_speed(generate.EQUILIBRIUM_ANGLE_DEG) == pytest.approx(0.0, abs=1e-15)


def test_generated_deck_is_the_sessile_deck_with_dynamic_ren_e(tmp_path: Path):
    generate = load("generate_case")
    case = generate.generate(16, "advancing", 2, tmp_path / "run")
    root = ET.parse(tmp_path / "run" / "solver.xml").getroot()
    general = root.find("GeneralSimulationParameters")
    assert general.findtext("Number_of_time_steps") == "400"
    assert float(general.findtext("Time_step_size")) == pytest.approx(generate.DT0 / 2)
    assert general.findtext("Increment_in_saving_VTK_files") == "20"
    level_set = [eq for eq in root.iter("Add_equation") if eq.attrib["type"] == "level_set"][0]
    assert level_set.findtext("Advection_velocity_extension_method") == "pde_harmonic"
    assert level_set.findtext("Enable_kinematic_reconciliation") == "true"
    assert level_set.findtext("Enable_sign_definite_patch_bounds") == "true"
    assert level_set.findtext("Enable_reinitialization") == "false"
    bc = free_surface(root)
    assert bc.findtext("Contact_line_model") == "DynamicRenE"
    assert float(bc.findtext("Contact_line_mobility")) == generate.MOBILITY
    assert float(bc.findtext("Wall_slip_length")) == generate.SLIP_LENGTH
    assert bc.findtext("Wall_slip_model") == "Navier"
    assert bc.findtext("Surface_tension_form") == "SurfaceStress"
    assert bc.findtext("Small_cut_aggregation") == "true"
    assert float(bc.findtext("Contact_angle_degrees")) == generate.EQUILIBRIUM_ANGLE_DEG
    walls = {b.attrib["name"]: b for b in root.iter("Add_BC") if b.attrib.get("name", "").startswith("wall_")}
    assert walls["wall_bottom"].findtext("Effective_direction") == "0 1"
    # Equal-area initial caps for the two cases.
    other = generate.generate(16, "receding", 1, tmp_path / "run2")
    assert case["initial_cap"]["area"] == pytest.approx(other["initial_cap"]["area"], rel=1e-13)
    assert case["initial_contact_vertex_gap_over_h"] > 1.0e-3
    assert case["initial_predicted_contact_line_speed"] > 0.0
    assert other["initial_predicted_contact_line_speed"] < 0.0


def test_measure_state_recovers_cap_angles_and_wall_speed(tmp_path: Path):
    generate = load("generate_case")
    verify = load("verify")
    case = generate.generate(32, "advancing", 1, tmp_path / "run")
    grid = pv.read(tmp_path / "run" / "mesh" / "mesh-complete.mesh.vtu")
    points = np.asarray(grid.points)
    # Outward-spreading wall velocity u_x = x (outward speed |x| at the roots).
    velocity = np.zeros((points.shape[0], 3))
    velocity[:, 0] = points[:, 0]
    grid.point_data["Velocity"] = velocity
    state = verify.measure_state(grid, case)
    left, right = state["contacts"]
    assert left["side"] == "left" and right["side"] == "right"
    # The sampled circle's wall chords give the initial angle to within the P1 chord error.
    for contact in (left, right):
        assert contact["dynamic_angle_degrees"] == pytest.approx(case["initial_angle_degrees"], abs=2.0)
        assert contact["wall_fluid_speed"] == pytest.approx(abs(contact["x"]), rel=1e-12)
        assert contact["predicted_speed"] > 0.0
    assert state["liquid_area"] == pytest.approx(case["initial_cap"]["area"], rel=5e-3)


def test_liquid_area_is_exact_for_linear_phi():
    verify = load("verify")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    phi = points[:, 0] + 0.5 * points[:, 1] - 0.6     # liquid where x + y/2 < 0.6
    # Exact area of {x + y/2 < 0.6} in the unit square: integral of clip(0.6 - y/2, 0, 1) dy.
    assert verify.liquid_area(points, cells, phi) == pytest.approx(0.6 - 0.25, rel=1e-14)


def test_gate_logic():
    verify = load("verify")
    tolerances = json.loads((BENCHMARK / "tolerances.json").read_text())

    def run(case, level, divisor, fluid, geometric, displacement, drift=1e-6, signs=True):
        return {"case": case, "level": level, "dt_divisor": divisor, "complete": True,
                "signs_agree": signs, "max_area_drift": drift,
                "rms_fluid_speed_relative_error": fluid, "rms_geometric_speed_relative_error": geometric,
                "final_mean_outward_displacement": displacement}

    results = []
    for case in ("advancing", "receding"):
        for level, error in ((8, 0.3), (16, 0.15), (32, 0.07)):
            results.append(run(case, level, 4, error, error, 0.05))
        results.append(run(case, 32, 2, 0.08, 0.08, 0.0505))
    report = verify.gate(results, tolerances)
    assert report["outcome"] == "PASS"
    results[2]["rms_fluid_speed_relative_error"] = 0.2      # finest not below 0.10, not decreasing
    report = verify.gate(results, tolerances)
    assert report["gates"]["advancing_finest_accuracy"] is False
    assert report["gates"]["advancing_mesh_convergence"] is False
    assert report["outcome"] == "FAIL"
