"""Checks of the static_drop_2d benchmark scripts on synthetic data (no solver run)."""

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = (Path(__file__).resolve().parent
             / "cases/fluid/free_surface_benchmarks/static_drop_2d")


def load(name):
    spec = importlib.util.spec_from_file_location(f"static_drop_2d_{name}", BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load("generate_case")
ver = load("verify")


def unit_square():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    return points, tris


def test_liquid_area_of_half_plane_is_exact():
    points, tris = unit_square()
    area, centroid = ver.liquid_area_centroid(points, tris, points[:, 0] - 0.3)
    assert area == pytest.approx(0.3, abs=1e-15)
    assert centroid == pytest.approx([0.15, 0.5], abs=1e-15)


def test_liquid_area_of_sampled_circle_converges_at_second_order():
    errors = []
    for level in (8, 16, 32):
        points, tris, _, _ = gen.structured_triangle_mesh(level)
        centre = np.array([1.5 + gen.CENTRE_OFFSET[0], 1.5 + gen.CENTRE_OFFSET[1]])
        phi = np.hypot(*(points[:, :2] - centre).T) - 1.0
        area, c = ver.liquid_area_centroid(points[:, :2], tris, phi)
        errors.append(abs(area - math.pi) / math.pi)
        assert c == pytest.approx(centre, abs=1e-3 / level)
    assert ver.observed_order([8, 16, 32], errors) > 1.8


def test_interior_pressure_mean_and_interface_points():
    points, tris, _, _ = gen.structured_triangle_mesh(16)
    points = points[:, :2]
    centre = np.array([1.5, 1.5])
    mean, n = ver.interior_mean_pressure(points, tris, 2.0 + 0.0 * points[:, 0], centre, 0.5)
    assert mean == pytest.approx(2.0) and n > 0
    # A linear field has the value at the region centroid, which is the disc centre by symmetry.
    mean, _ = ver.interior_mean_pressure(points, tris, points[:, 0], centre, 0.5)
    assert mean == pytest.approx(1.5, abs=1e-12)
    iface = ver.interface_points(points, tris, points[:, 0] - 1.3 - 1e-3 * points[:, 1])
    assert len(iface) > 0
    assert np.abs(iface[:, 0] - 1.3 - 1e-3 * iface[:, 1]).max() < 1e-12


def test_observed_order_recovers_power_law():
    assert ver.observed_order([8, 16, 32], [0.04, 0.01, 0.0025]) == pytest.approx(2.0)


def test_generated_case_is_complete_and_respects_time_step_rule(tmp_path):
    case = gen.generate(8, "kag_lumped", 12.0, tmp_path / "c")
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    text = (tmp_path / "c/solver.xml").read_text()
    assert "<Geometry_tangent_policy>RefreshedFrozenQuadrature" in text
    assert "<Surface_tension_form>KinematicAreaGradientTraction" in text
    assert "<Curvature_projection_kinematic_area_gradient_mass>Lumped" in text
    assert root.find("GeneralSimulationParameters/Number_of_time_steps").text == str(case["steps"])
    dt_b = math.sqrt(case["h"] ** 3 / (4.0 * math.pi))
    # La = 12 runs at twice the one-sided capillary limit (step-0 measurement).
    assert case["dt_multiple_of_capillary_limit"] == 2.0
    assert dt_b < case["dt"] <= 2.0 * dt_b * (1 + 1e-12)
    slow = gen.time_schedule(8, 120.0, 5.0, 100)
    assert slow["dt_multiple_of_capillary_limit"] == 1.0
    assert slow["dt"] <= dt_b * (1 + 1e-12)
    assert case["steps"] * case["dt"] == pytest.approx(5.0 * case["viscous_time"])
    assert case["viscosity"] == pytest.approx(math.sqrt(2.0 / 12.0))
    assert case["min_abs_phi_over_h"] > 1e-3
    for wall in gen.WALLS:
        assert (tmp_path / f"c/mesh/mesh-surfaces/{wall}.vtp").is_file()
    consistent = (gen.solver_xml("kag_consistent", gen.time_schedule(8, 12.0, 5.0, 100), 10, 1))
    assert "kinematic_area_gradient_mass" not in consistent
    stress = gen.solver_xml("surface_stress", gen.time_schedule(8, 12.0, 5.0, 100), 10, 1)
    assert "SurfaceStress" in stress and "Curvature_field" not in stress
    pde = gen.solver_xml("surface_stress", gen.time_schedule(8, 12.0, 5.0, 100), 10, 1,
                         "pde_harmonic_monolithic")
    assert "<Advection_velocity_extension_method>pde_harmonic<" in pde
    assert "<Advection_velocity_extension_coupling>monolithic<" in pde
    assert "<Velocity_source>prescribed_data" in pde
    assert "Use_wet_extension_advection_velocity" not in pde
    with pytest.raises(ValueError):
        gen.generate(8, "surface_stress", 12.0, tmp_path / "bad", level_set_velocity="wet")


def write_synthetic_run(run, level, speed_scale, growth=False, pressure_error=None, drop_last=False):
    """Emulate solver output: static phi, decaying velocity, near-Laplace pressure."""
    case = gen.generate(level, "surface_stress", 12.0, run, snapshots=8)
    grid_points, tris, _, _ = gen.structured_triangle_mesh(level)
    centre = np.array(case["centre"])
    phi = np.hypot(*(grid_points[:, :2] - centre).T) - 1.0
    area, _ = ver.liquid_area_centroid(grid_points[:, :2], tris, phi)
    r_eff = math.sqrt(area / math.pi)
    err = 0.004 * 8 / level if pressure_error is None else pressure_error
    pressure = np.full(len(phi), (1.0 + err) / r_eff)
    rel = grid_points[:, :2] - centre
    tangential = np.column_stack([-rel[:, 1], rel[:, 0], np.zeros(len(phi))])
    entries = []
    times = [k * case["output_cadence"] * case["dt"] for k in range(1, 9)]
    for k, t in enumerate(times, start=1):
        if drop_last and k == len(times):
            break
        amp = speed_scale * (math.exp(t / case["viscous_time"]) if growth
                             else math.exp(-t / case["viscous_time"]))
        name = f"result_{k * case['output_cadence']:03d}.vtu"
        gen.write_vtu(run / name, grid_points, tris,
                      {"phi": ("Float64", phi), "Velocity": ("Float64", amp * tangential),
                       "Pressure": ("Float64", pressure)},
                      {"GlobalElementID": ("Int64", np.arange(len(tris)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    return case


@pytest.fixture
def study(tmp_path):
    pytest.importorskip("pyvista")

    def make(overrides=None):
        overrides = overrides or {}
        runs = []
        for level in (8, 16, 32):
            opts = {"speed_scale": 2e-5 * 8 / level}
            opts.update(overrides.get(level, {}))
            write_synthetic_run(tmp_path / f"L{level}", level, **opts)
            runs.append(str(tmp_path / f"L{level}"))
        return runs
    return make


def test_synthetic_refinement_study_passes(study, tmp_path):
    out = tmp_path / "report.json"
    assert ver.main([*study(), "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    group = report["groups"][0]
    assert group["passed"]
    run8 = group["runs"][0]
    assert run8["pressure_jump_relative_error"] == pytest.approx(0.004, rel=1e-9)
    assert run8["liquid_area_relative_drift_max"] == 0.0


def test_growth_and_nonmonotone_capillary_number_fail(study, capsys):
    runs = study({16: {"speed_scale": 5e-5, "growth": True}})
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] no_velocity_growth" in out
    assert "[FAIL] parasitic_capillary_number" in out
    assert "[PASS] pressure_jump" in out


def test_large_but_decreasing_capillary_number_passes(study, capsys):
    # No absolute limit: only strict decrease under refinement is gated.
    runs = study({8: {"speed_scale": 4e-2}, 16: {"speed_scale": 2e-2},
                  32: {"speed_scale": 1e-2}})
    assert ver.main(runs) == 0
    out = capsys.readouterr().out
    assert "[PASS] parasitic_capillary_number" in out
    assert "(reported)" in out


def test_pressure_order_below_one_fails(study, capsys):
    runs = study({16: {"pressure_error": 0.0035}, 32: {"pressure_error": 0.003}})
    assert ver.main(runs) == 1
    assert "[FAIL] pressure_jump" in capsys.readouterr().out


def test_missing_and_incomplete_data_fail_clearly(study, tmp_path, capsys):
    runs = study()
    assert ver.main(runs[:2]) == 1                      # no R/h = 32 run
    assert "missing run at R/h=32" in capsys.readouterr().out
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 8, 1e-5, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    empty = tmp_path / "empty"
    gen.generate(8, "surface_stress", 12.0, empty)
    assert ver.main([str(empty)]) == 2
    assert "no solver output" in capsys.readouterr().err


def test_default_transport_is_the_harmonic_pde_extension(tmp_path):
    case = gen.generate(8, "surface_stress", 12.0, tmp_path / "c")
    text = (tmp_path / "c/solver.xml").read_text()
    assert case["level_set_velocity"] == "pde_harmonic_monolithic"
    assert "<Advection_velocity_extension_method>pde_harmonic<" in text
    assert "<Advection_velocity_extension_coupling>monolithic<" in text
    old = gen.generate(8, "surface_stress", 12.0, tmp_path / "old",
                       level_set_velocity="coupled_field")
    assert "<Velocity_source>coupled_field<" in (tmp_path / "old/solver.xml").read_text()
    assert old["dt"] == case["dt"]


def test_kinematic_reconciliation_is_on_by_default_and_can_be_disabled(tmp_path):
    case = gen.generate(8, "surface_stress", 12.0, tmp_path / "on")
    root = ET.parse(tmp_path / "on/solver.xml").getroot()
    level_set = root.find("Add_equation[@type='level_set']")
    assert case["kinematic_reconciliation"] is True
    assert level_set.find("Enable_kinematic_reconciliation").text.strip() == "true"
    off = gen.generate(8, "surface_stress", 12.0, tmp_path / "off", kinematic_reconciliation=False)
    root = ET.parse(tmp_path / "off/solver.xml").getroot()
    assert off["kinematic_reconciliation"] is False
    assert root.find("Add_equation[@type='level_set']/Enable_kinematic_reconciliation") is None
