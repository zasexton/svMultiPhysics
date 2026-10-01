"""Checks of the linear_sloshing_2d benchmark scripts on synthetic data (no solver run)."""

import cmath
import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = (Path(__file__).resolve().parent
             / "cases/fluid/free_surface_benchmarks/linear_sloshing_2d")


def load(name):
    spec = importlib.util.spec_from_file_location(f"linear_sloshing_2d_{name}",
                                                  BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load("generate_case")
ver = load("verify")


def test_dispersion_relation_limits():
    k = math.pi
    omega0 = gen.inviscid_frequency(1.0, k, 0.5)
    # Weak viscosity: frequency -> omega0, damping -> Lamb's 2 nu k^2 with the
    # free-surface boundary-layer correction -(1/sqrt 2) sqrt(nu k^2/omega0).
    nu = 1e-5
    s = gen.viscous_root(1.0, k, 0.5, nu)
    assert abs(gen.viscous_dispersion(s, 1.0, k, 0.5, nu)) < 1e-12
    assert s.imag == pytest.approx(omega0, rel=1e-6)
    ratio = -s.real / (2 * nu * k * k)
    assert ratio == pytest.approx(1 - math.sqrt(nu * k * k / omega0) / math.sqrt(2), abs=5e-4)
    # Deep water: Lamb (1932, art. 349), (s + 2 nu k^2)^2 + g k = 4 nu^2 k^3 m.
    nu = 1e-3

    def lamb(z):
        m = cmath.sqrt(k * k + z / nu)
        return (z + 2 * nu * k * k) ** 2 + k - 4 * nu * nu * k ** 3 * m

    z = complex(-2 * nu * k * k, math.sqrt(k))
    for _ in range(50):
        d = 1e-7
        z -= lamb(z) / ((lamb(z + d) - lamb(z - d)) / (2 * d))
    assert gen.viscous_root(1.0, k, 12.0, nu) == pytest.approx(z, rel=1e-10)


def test_reference_values_of_the_protocol():
    ref = gen.reference()
    assert ref["omega_inviscid"] == pytest.approx(math.sqrt(math.pi * math.tanh(math.pi * 65 / 128)))
    assert ref["omega_reference"] / ref["omega_inviscid"] - 1 == pytest.approx(-2.03e-4, rel=0.01)
    assert ref["damping_rate_reference"] / ref["damping_rate_lamb"] == pytest.approx(0.9649, abs=1e-4)
    assert ref["viscous_parameter_nu_k2_over_omega"] < 5e-3


def test_fit_recovers_a_damped_oscillation():
    t = np.linspace(0.0, 15.0, 129)
    y = 0.001 + 0.005 * np.exp(-0.0095 * t) * np.cos(1.70062 * t + 0.3)
    fit = ver.fit_damped_oscillation(t, y, 1.6)
    assert fit["omega"] == pytest.approx(1.70062, rel=1e-10)
    assert fit["damping_rate"] == pytest.approx(0.0095, rel=1e-8)
    assert fit["amplitude"] == pytest.approx(0.005, rel=1e-9)
    assert fit["offset"] == pytest.approx(0.001, abs=1e-12)


def test_probe_elevation_modal_fit_and_area():
    points, tris, _, _ = gen.structured_triangle_mesh(32)
    points = points[:, :2]
    k = math.pi
    a = 0.004
    phi = points[:, 1] - gen.MEAN_DEPTH - a * np.cos(k * points[:, 0])
    assert ver.probe_elevation(points, phi, 0.0, 1 / 32) == pytest.approx(gen.MEAN_DEPTH + a, abs=1e-15)
    assert ver.probe_elevation(points, phi, 1.0, 1 / 32) == pytest.approx(gen.MEAN_DEPTH - a, abs=1e-15)
    coef = ver.modal_coefficients(ver.interface_points(points, tris, phi), k)
    assert coef[0] == pytest.approx(gen.MEAN_DEPTH, abs=1e-6)
    assert coef[1] == pytest.approx(a, rel=2e-3)
    # The cosine has zero mean, so the P1 area differs from L*H0 only by interpolation.
    assert ver.liquid_area(points, tris, phi) == pytest.approx(gen.MEAN_DEPTH, rel=1e-6)
    with pytest.raises(ver.DataError):
        ver.probe_elevation(points, phi, 0.01, 1 / 32)


def test_generated_case_has_free_slip_walls_and_the_d10_schedule(tmp_path):
    case = gen.generate(16, tmp_path / "c")
    text = (tmp_path / "c/solver.xml").read_text()
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    bcs = {bc.get("name"): bc for bc in root.iter("Add_BC")}
    assert bcs["wall_left"].find("Effective_direction").text == "1 0"
    assert bcs["wall_right"].find("Effective_direction").text == "1 0"
    assert bcs["wall_bottom"].find("Effective_direction").text == "0 1"
    assert "wall_top" not in bcs
    assert "<Surface_tension>0.0</Surface_tension>" in text
    assert float(root.find(".//Viscosity/Value").text) == pytest.approx(5e-4)
    # D9: the protocol advects phi with the PDE extension (prescribed field).
    assert case["level_set_velocity"] == gen.PROTOCOL_LEVEL_SET_VELOCITY
    assert case["kinematic_reconciliation"] is True
    assert root.find("Add_equation[@type='level_set']/Enable_kinematic_reconciliation").text == "true"
    plain = gen.solver_xml(gen.time_schedule(16), 10, 1, kinematic_reconciliation=False)
    assert "Enable_kinematic_reconciliation" not in plain
    method, coupling = gen.PROTOCOL_LEVEL_SET_VELOCITY.rsplit("_", 1)
    assert f"<Advection_velocity_extension_method>{method}<" in text
    assert f"<Advection_velocity_extension_coupling>{coupling}<" in text
    assert "Use_wet_extension_advection_velocity" not in text
    # D10: every level of the spatial study uses the same step.
    assert case["steps_per_period"] == 128 and case["steps"] == 512
    assert case["study_roles"] == ["spatial"] and case["protocol_run"]
    assert case["end_time"] == pytest.approx(4 * case["period_inviscid"])
    assert case["interface_band_vertex_gap_over_h"] > 0.04
    assert gen.study_roles(32, 128) == ["spatial", "time"]
    assert gen.study_roles(32, 64) == ["time"] and gen.study_roles(32, 256) == ["time"]
    assert gen.study_roles(16, 64) == [] and gen.study_roles(64, 256) == []
    for level in (32, 64):
        schedule = gen.time_schedule(level)
        assert schedule["steps_per_period"] == 128
        assert schedule["steps"] // schedule["output_cadence"] == 128
    # The initial pressure vanishes on the free surface to second order in A.
    x = np.linspace(0.0, 1.0, 11)
    surface = np.column_stack([x, gen.MEAN_DEPTH + gen.AMPLITUDE * np.cos(math.pi * x)])
    p = gen.initial_fields(surface, math.pi)["pressure"]
    assert np.max(np.abs(p)) < 2 * math.pi * gen.AMPLITUDE ** 2


def write_synthetic_run(run, level, omega_error, *, steps_per_period=128,
                        level_set_velocity=None, damping_scale=1.0,
                        samples_per_period=8, area_leak=0.0, drop_last=False):
    """Emulate solver output: a damped standing wave with a prescribed frequency error."""
    case = gen.generate(level, run, steps_per_period=steps_per_period,
                        level_set_velocity=level_set_velocity or gen.PROTOCOL_LEVEL_SET_VELOCITY)
    points, tris, _, _ = gen.structured_triangle_mesh(level)
    k = case["wavenumber"]
    omega = case["omega_reference"] * (1.0 + omega_error)
    gamma = case["damping_rate_reference"] * damping_scale
    n = round(case["periods"] * samples_per_period)
    entries = []
    for i in range(1, n + 1):
        if drop_last and i == n:
            break
        t = case["end_time"] * i / n
        eta = case["amplitude"] * math.exp(-gamma * t) * math.cos(omega * t) * np.cos(k * points[:, 0])
        phi = points[:, 1] - case["mean_depth"] - eta - area_leak * t
        name = f"result_{i:03d}.vtu"
        gen.write_vtu(run / name, points, tris,
                      {"phi": ("Float64", phi), "Velocity": ("Float64", np.zeros((len(phi), 3)))},
                      {"GlobalElementID": ("Int64", np.arange(len(tris)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    return case


TIME_COEFFICIENT = -4.2e-3 * 32 ** 2        # e_time(spp) = C / spp^2 (generalized alpha)


@pytest.fixture
def study(tmp_path):
    """Spatial study (all levels at 128 steps/T) plus the time study at L/h = 32."""
    pytest.importorskip("pyvista")
    calls = []

    def make(spatial_errors, *, time_study=True, transport=None, **kwargs):
        calls.append(len(calls))
        base = tmp_path / f"study{len(calls)}"
        runs = []
        for level, err in spatial_errors.items():
            run = base / f"L{level}_T128"
            write_synthetic_run(run, level, err + TIME_COEFFICIENT / 128 ** 2,
                                level_set_velocity=transport, **kwargs)
            runs.append(str(run))
        if time_study:
            for spp in (64, 256):
                run = base / f"L32_T{spp}"
                write_synthetic_run(run, 32, spatial_errors[32] + TIME_COEFFICIENT / spp ** 2,
                                    steps_per_period=spp, level_set_velocity=transport,
                                    **kwargs)
                runs.append(str(run))
        return runs
    return make


def test_converging_study_passes_with_the_time_error_removed(study, tmp_path, capsys):
    out = tmp_path / "report.json"
    runs = study({16: 4e-3, 32: 1e-3, 64: 2.5e-4})
    assert ver.main([*runs, "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    group = report["groups"][0]
    assert group["protocol"]
    assert group["time_study"]["observed_temporal_order"] == pytest.approx(2.0, abs=1e-3)
    assert group["time_study"]["time_error_at_spatial_step"] == pytest.approx(
        TIME_COEFFICIENT / 128 ** 2, rel=1e-4)
    spatial = {r["level"]: r for r in group["runs"] if r["steps_per_period"] == 128}
    assert spatial[16]["frequency_spatial_signed_error"] == pytest.approx(4e-3, rel=1e-4)
    assert spatial[64]["frequency_spatial_signed_error"] == pytest.approx(2.5e-4, rel=1e-3)
    assert spatial[64]["damping_rate_over_reference"] == pytest.approx(1.0, abs=1e-6)
    text = capsys.readouterr().out
    assert "observed order 2.00" in text and "[PASS] damping" in text


def test_cancelling_space_and_time_errors_no_longer_pass(study, capsys):
    # Raw errors that decrease only because the time error cancels the
    # spatial error: once the time error is removed the study must fail.
    assert ver.main(study({16: 4e-3, 32: 4e-3, 64: 4e-3})) == 1
    assert "[FAIL] frequency" in capsys.readouterr().out
    assert ver.main(study({16: 8e-2, 32: 4e-2, 64: 2e-2})) == 1
    assert "L/h=64: 0.02 > 0.01" in capsys.readouterr().out


def test_damping_limit_is_five_percent_at_the_finest_level(study, capsys):
    assert ver.main(study({16: 4e-3, 32: 1e-3, 64: 2.5e-4}, damping_scale=1.08)) == 1
    out = capsys.readouterr().out
    assert "[FAIL] damping" in out and "[PASS] frequency" in out


def test_volume_criterion_gates_every_run(study, capsys):
    assert ver.main(study({16: 4e-3, 32: 1e-3, 64: 2.5e-4}, area_leak=1e-5)) == 1
    out = capsys.readouterr().out
    assert "[FAIL] volume_drift" in out and "steps/T=64" in out


def test_missing_and_incomplete_data_fail_clearly(study, tmp_path, capsys):
    assert ver.main(study({16: 4e-3, 32: 1e-3}, time_study=True)) == 1
    assert "missing run at L/h=64" in capsys.readouterr().out
    assert ver.main(study({16: 4e-3, 32: 1e-3, 64: 2.5e-4}, time_study=False)) == 1
    assert "time-step study incomplete" in capsys.readouterr().out
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 16, 0.0, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    gen.generate(16, smoke, max_steps=5)
    assert ver.main([str(smoke)]) == 2
    assert "no solver output" in capsys.readouterr().err


def test_other_transports_are_reported_but_not_gated(study, capsys):
    runs = study({16: 4e-3, 32: 1e-3, 64: 2.5e-4})
    others = study({16: 5e-2, 32: 5e-2, 64: 5e-2}, transport="coupled_field")
    assert ver.main([*runs, *others]) == 0
    out = capsys.readouterr().out
    assert "[comparison, not gated]" in out and "[INFO] frequency" in out
    assert ver.main(others) == 1
    assert "no runs of the protocol transport" in capsys.readouterr().out


def test_solver_log_summary(tmp_path):
    import gzip
    lines = [
        "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=0 time=0.0e+00 converged=1 "
        "iters=5 ||r||=1.84e-11 outer_iters=5 inner_iters_total=5",
        "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=1 time=1.1e-01 converged=0 "
        "iters=3 ||r||=2.30e-11 outer_iters=4 inner_iters_total=3",
    ]
    with gzip.open(tmp_path / "solver_run.log.gz", "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    (tmp_path / "run.txt").write_text("exit=0 elapsed_s=10 end=now\n")
    summary = ver.solver_log_summary(tmp_path)
    assert summary["steps_logged"] == 2 and summary["nonconverged_steps"] == 1
    assert summary["outer_passes_mean"] == 4.5 and summary["newton_iterations_mean"] == 4.0
    assert summary["final_residual_max"] == 2.30e-11 and summary["wall_seconds_per_step"] == 5.0


def test_wet_extension_diagnostic_variant(tmp_path):
    case = gen.generate(16, tmp_path / "w", level_set_velocity="wet_extension")
    text = (tmp_path / "w/solver.xml").read_text()
    assert not case["protocol_run"] and case["level_set_velocity"] == "wet_extension"
    assert "<Use_wet_extension_advection_velocity>true" in text
    assert "<Velocity_field_name>LevelSetAdvectionVelocity" in text
    protocol = gen.generate(16, tmp_path / "p")
    assert protocol["protocol_run"]
    assert "Use_wet_extension" not in (tmp_path / "p/solver.xml").read_text()
    with pytest.raises(ValueError):
        gen.generate(16, tmp_path / "bad", level_set_velocity="other")


def test_mean_depth_diagnostic_variant(tmp_path):
    case = gen.generate(16, tmp_path / "d", mean_depth=0.5 + 1 / 32)
    assert not case["protocol_run"]
    assert case["interface_cell_position"] == pytest.approx(0.5)
    assert case["omega_inviscid"] == pytest.approx(
        math.sqrt(math.pi * math.tanh(math.pi * (0.5 + 1 / 32))))
    assert case["interface_band_vertex_gap_over_h"] > 0.3
    assert gen.generate(16, tmp_path / "p")["interface_cell_position"] == pytest.approx(0.125)
    with pytest.raises(ValueError):
        gen.generate(16, tmp_path / "bad", mean_depth=0.7)
