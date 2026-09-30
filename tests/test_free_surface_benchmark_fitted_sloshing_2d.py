"""Checks of the fitted_sloshing_2d benchmark scripts on synthetic data (no solver run)."""

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

CASES = Path(__file__).resolve().parent / "cases/fluid/free_surface_benchmarks"
BENCHMARK = CASES / "fitted_sloshing_2d"


def load(directory, name):
    spec = importlib.util.spec_from_file_location(f"{directory.name}_{name}",
                                                  directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load(BENCHMARK, "generate_case")
ver = load(BENCHMARK, "verify")
unfitted = load(CASES / "linear_sloshing_2d", "generate_case")


def test_reference_is_the_linear_sloshing_reference():
    assert gen.reference() == unfitted.reference()
    assert gen.MEAN_DEPTH == unfitted.MEAN_DEPTH and gen.AMPLITUDE == unfitted.AMPLITUDE
    assert gen.KINEMATIC_VISCOSITY == unfitted.KINEMATIC_VISCOSITY


def test_liquid_mesh_follows_the_initial_surface():
    k = math.pi
    for level, rows in ((16, 8), (32, 16), (64, 32)):
        points, cells, faces, (nx, ny) = gen.liquid_triangle_mesh(level, k)
        assert (nx, ny) == (level, rows)
        top = points[faces[gen.FREE_SURFACE][0]]
        assert np.allclose(top[:, 1], gen.MEAN_DEPTH + gen.AMPLITUDE * np.cos(k * top[:, 0]))
        assert np.allclose(points[faces["wall_left"][0], 0], 0.0)
        assert np.allclose(points[faces["wall_right"][0], 0], 1.0)
        assert np.allclose(points[faces["wall_bottom"][0], 1], 0.0)
        # The sampled cosine is antisymmetric about x = 1/2, so the
        # piecewise-linear surface encloses exactly L * H0.
        assert gen.polygon_area(points, cells) == pytest.approx(gen.MEAN_DEPTH, rel=1e-13)
    # The initial pressure vanishes on the free surface to second order in A.
    x = np.linspace(0.0, 1.0, 11)
    surface = np.column_stack([x, gen.MEAN_DEPTH + gen.AMPLITUDE * np.cos(math.pi * x)])
    assert np.max(np.abs(gen.initial_pressure(surface, math.pi))) < 2 * math.pi * gen.AMPLITUDE ** 2


def test_generated_case_uses_mesh_nitsche_and_sliding_walls(tmp_path):
    case = gen.generate(16, tmp_path / "c")
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    equations = {eq.get("type"): eq for eq in root.iter("Add_equation")}
    assert list(equations) == ["fluid", "mesh_motion"]   # the fluid registers first
    fluid, mesh = equations["fluid"], equations["mesh_motion"]
    for eq in (fluid, mesh):
        bcs = {bc.get("name"): bc for bc in eq.iter("Add_BC")}
        assert bcs["wall_left"].find("Effective_direction").text == "1 0"
        assert bcs["wall_right"].find("Effective_direction").text == "1 0"
        assert bcs["wall_bottom"].find("Effective_direction").text == "0 1"
    surface = {bc.get("name"): bc for bc in fluid.iter("Add_BC")}["free_surface"]
    assert surface.find("Implementation").text == "FittedALE"
    assert surface.find("Kinematic_enforcement").text == "MeshNitsche"
    assert float(surface.find("Kinematic_nitsche_gamma").text) == 10.0
    assert surface.find("Tangential_mesh_policy").text == "Free"
    assert float(surface.find("Surface_tension").text) == 0.0
    assert mesh.find("Model").text == "Harmonic" and float(mesh.find("Kappa").text) == 1.0
    assert case["study"] == "spatial" and case["protocol_run"]
    assert case["steps_per_period"] == 512 and case["steps"] == 2048
    assert case["end_time"] == pytest.approx(4 * case["period_inviscid"])
    assert case["free_surface_nodes"] == list(range(16 * 8 + 8, 17 * 9))


def test_study_roles_and_diagnostic_variants(tmp_path):
    assert gen.protocol_role(64, 512) == "spatial"
    assert gen.protocol_role(32, 128) == "time"
    assert gen.protocol_role(32, 512) == "spatial"
    assert gen.protocol_role(16, 128) is None
    dt = gen.generate(32, tmp_path / "t", steps_per_period=64)
    assert dt["study"] == "time" and dt["protocol_run"] and dt["output_cadence"] == 2
    pinned = gen.generate(16, tmp_path / "p", wall_mesh_motion="pinned",
                          enforcement="Nitsche", steps_per_period=64)
    assert not pinned["protocol_run"]
    root = ET.parse(tmp_path / "p/solver.xml").getroot()
    mesh = [eq for eq in root.iter("Add_equation") if eq.get("type") == "mesh_motion"][0]
    assert all(bc.find("Effective_direction") is None for bc in mesh.iter("Add_BC"))
    with pytest.raises(ValueError):
        gen.generate(16, tmp_path / "bad", steps_per_period=8)
    with pytest.raises(ValueError):
        gen.generate(24, tmp_path / "bad2")


def write_synthetic_run(run, level, steps_per_period, omega_error, *, samples_per_period=8,
                        area_leak=0.0, drop_last=False):
    """Emulate solver output: the mesh displaced by a damped standing wave."""
    case = gen.generate(level, run, steps_per_period=steps_per_period)
    points, cells, _, _ = gen.liquid_triangle_mesh(level, case["wavenumber"])
    k = case["wavenumber"]
    omega = case["omega_reference"] * (1.0 + omega_error)
    gamma = case["damping_rate_reference"]
    height = gen.MEAN_DEPTH + gen.AMPLITUDE * np.cos(k * points[:, 0])
    n = round(case["periods"] * samples_per_period)
    entries = []
    for i in range(1, n + 1):
        if drop_last and i == n:
            break
        t = case["end_time"] * i / n
        # Surface elevation A e^{-gamma t} cos(omega t) cos(k x), extended
        # linearly in depth; the initial shape is the reference mesh.
        eta = case["amplitude"] * (math.exp(-gamma * t) * math.cos(omega * t) - 1.0) \
            * np.cos(k * points[:, 0]) + area_leak * t
        disp = np.zeros((len(points), 3))
        disp[:, 1] = eta * points[:, 1] / height
        name = f"result_{i:03d}.vtu"
        unfitted.write_vtu(run / name, points, cells,
                           {"GlobalNodeID": ("Int64", np.arange(len(points))),
                            "Velocity": ("Float64", np.zeros((len(points), 3))),
                            "mesh_displacement": ("Float64", disp)},
                           {"GlobalElementID": ("Int64", np.arange(len(cells)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    return case


@pytest.fixture
def study(tmp_path):
    pytest.importorskip("pyvista")
    counter = []

    def make(spatial, timing, **kwargs):
        counter.append(1)
        runs = []
        for level, err in spatial.items():
            run = tmp_path / f"s{len(counter)}" / f"L{level}"
            write_synthetic_run(run, level, 512, err, **kwargs)
            runs.append(str(run))
        for steps, err in timing.items():
            run = tmp_path / f"s{len(counter)}" / f"T{steps}"
            write_synthetic_run(run, 32, steps, err, **kwargs)
            runs.append(str(run))
        return runs
    return make


# Time errors e_t(S) = -c (64/S)^2 of a second-order scheme, and the spatial
# errors that the spatial study should recover after their removal.
TIME = {64: -4.0e-3, 128: -1.0e-3, 256: -2.5e-4, 512: -6.25e-5}


def test_time_error_is_removed_from_the_spatial_study(study, tmp_path, capsys):
    spatial = {16: 4e-3, 32: 1e-3, 64: 2.5e-4}
    runs = study({lv: e + TIME[512] for lv, e in spatial.items()},
                 {s: 1e-3 + e for s, e in TIME.items() if s != 512})
    out = tmp_path / "report.json"
    assert ver.main([*runs, "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    dts = report["summary"]["time_step_study"]
    assert dts["observed_order_richardson"] == pytest.approx(2.0, abs=1e-3)
    by_level = {r["level"]: r for r in report["runs"] if r["study"] == "spatial"}
    for level, err in spatial.items():
        assert by_level[level]["frequency_spatial_signed_error"] == pytest.approx(err, rel=2e-3)
        assert by_level[level]["damping_rate_over_reference"] == pytest.approx(1.0, abs=1e-6)
        # A cosine mode leaves the area of the mesh unchanged.
        assert by_level[level]["liquid_area_relative_deviation_max"] < 1e-12
    text = capsys.readouterr().out
    assert "[PASS] frequency" in text and "[PASS] time_step_study" in text
    assert "[PASS] damping" in text and "[PASS] volume" in text


def test_failures_are_reported(study, capsys):
    stagnating = study({16: 4e-3, 32: 4e-3, 64: 4e-3},
                       {s: 4e-3 - TIME[512] + e for s, e in TIME.items() if s != 512})
    assert ver.main(stagnating) == 1
    assert "[FAIL] frequency" in capsys.readouterr().out
    leak = study({16: 4e-3, 32: 1e-3, 64: 2.5e-4},
                 {s: e for s, e in TIME.items() if s != 512}, area_leak=1e-5)
    assert ver.main(leak) == 1
    out = capsys.readouterr().out
    assert "[FAIL] volume" in out
    no_dt = study({16: 4e-3, 32: 1e-3, 64: 2.5e-4}, {})
    assert ver.main(no_dt) == 1
    assert "missing frequency_spatial_relative_error" in capsys.readouterr().out


def test_missing_and_incomplete_data_fail_clearly(tmp_path, capsys):
    pytest.importorskip("pyvista")
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 16, 512, 0.0, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    gen.generate(16, smoke, max_steps=5)
    assert ver.main([str(smoke)]) == 2
    assert "no solver output" in capsys.readouterr().err


def test_probe_and_modal_fit():
    x = np.linspace(0.0, 1.0, 17)
    y = gen.MEAN_DEPTH + 0.004 * np.cos(math.pi * x) + 0.001 * np.cos(2 * math.pi * x)
    surface = np.column_stack([x, y])
    assert ver.probe_elevation(surface, 0.0) == pytest.approx(gen.MEAN_DEPTH + 0.005)
    coef = ver.modal_coefficients(surface, math.pi)
    assert coef[1] == pytest.approx(0.004, abs=1e-12) and coef[2] == pytest.approx(0.001, abs=1e-12)
    with pytest.raises(ver.DataError):
        ver.probe_elevation(surface, 1.5)
