"""Checks of the fitted_capillary_wave_2d benchmark scripts on synthetic data (no solver run)."""

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

CASES = Path(__file__).resolve().parent / "cases/fluid/free_surface_benchmarks"
BENCHMARK = CASES / "fitted_capillary_wave_2d"


def load(directory, name):
    spec = importlib.util.spec_from_file_location(f"{directory.name}_{name}_test",
                                                  directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load(BENCHMARK, "generate_case")
ver = load(BENCHMARK, "verify")
unfitted = load(CASES / "capillary_wave_2d", "generate_case")
writer = load(CASES / "linear_sloshing_2d", "generate_case")


def test_physical_case_and_schedule_are_those_of_capillary_wave_2d():
    for name in ("DENSITY", "SURFACE_TENSION", "WAVELENGTH", "WAVENUMBER", "MEAN_LEVEL",
                 "EXTERNAL_PRESSURE"):
        assert getattr(gen, name) == getattr(unfitted, name)
    # D35: the protocol amplitude is a quarter of the capillary_wave_2d one;
    # the earlier value stays available for the earlier decks.
    assert gen.AMPLITUDE == 0.0025 * unfitted.WAVELENGTH
    assert gen.LEGACY_AMPLITUDE_OVER_WAVELENGTH == unfitted.AMPLITUDE_OVER_WAVELENGTH
    assert gen.WIDTH == unfitted.BOX_WIDTH and gen.LAPLACE_NUMBER == unfitted.DEFAULT_LAPLACE_NUMBER
    # The reference is imported, not copied.
    assert gen.reference.__file__ == str(CASES / "capillary_wave_2d" / "prosperetti_reference.py")
    for level in gen.LEVELS:
        for divisor in gen.DT_DIVISORS:
            ours = gen.time_schedule(level, divisor)
            theirs = unfitted.time_schedule(level, unfitted.DEFAULT_LAPLACE_NUMBER,
                                            unfitted.DEFAULT_PERIODS, unfitted.DEFAULT_SNAPSHOTS,
                                            divisor, dt_rule="capillary-limit")
            assert ours["dt"] == theirs["dt"] and ours["steps"] == theirs["steps"]
    # The shared step is within the one-sided capillary limit of the finest level.
    fine = gen.time_schedule(64)
    assert fine["dt_over_capillary_limit"] <= 1.0 < 1.01 * fine["dt_over_capillary_limit"]
    assert gen.time_schedule(16)["dt_over_capillary_limit"] == pytest.approx(0.125, abs=1e-3)


def test_liquid_mesh_follows_the_initial_surface():
    for level in gen.LEVELS:
        points, cells, faces, (nx, ny) = gen.liquid_triangle_mesh(level)
        assert (nx, ny) == (level // 2, level)
        top = points[faces[gen.FREE_SURFACE][0]]
        assert np.allclose(top[:, 1], gen.MEAN_LEVEL + gen.AMPLITUDE * np.cos(gen.WAVENUMBER * top[:, 0]))
        assert np.allclose(points[faces["wall_left"][0], 0], 0.0)
        assert np.allclose(points[faces["wall_right"][0], 0], gen.WIDTH)
        assert np.allclose(points[faces["wall_bottom"][0], 1], 0.0)
        # Each face edge belongs to its recorded parent triangle.
        for name, (nodes, parents) in faces.items():
            for a, b, parent in zip(nodes[:-1], nodes[1:], parents):
                assert {a, b} <= set(cells[parent].tolist()), name
        # The sampled cosine is antisymmetric about lambda/4: the P1 region
        # encloses exactly W * y0.
        assert gen.polygon_area(points, cells) == pytest.approx(gen.WIDTH * gen.MEAN_LEVEL, rel=1e-13)
    # Initial pressure: gamma * kappa = gamma a0 k^2 cos(kx) at the mean level.
    x = np.linspace(0.0, gen.WIDTH, 5)
    surface = np.column_stack([x, np.full_like(x, gen.MEAN_LEVEL)])
    expected = gen.SURFACE_TENSION * gen.AMPLITUDE * gen.WAVENUMBER ** 2 * np.cos(gen.WAVENUMBER * x)
    assert np.allclose(gen.initial_pressure(surface), expected)


def test_generated_case_uses_fitted_surface_stress_mesh_nitsche_and_sliding_walls(tmp_path):
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
    assert float(fluid.find("Force_y").text) == 0.0
    assert float(fluid.find("Viscosity/Value").text) == pytest.approx(math.sqrt(1.0 / 3000.0))
    surface = {bc.get("name"): bc for bc in fluid.iter("Add_BC")}["free_surface"]
    assert surface.find("Implementation").text == "FittedALE"
    assert surface.find("Surface_tension_form").text == "SurfaceStress"
    assert surface.find("Allow_fitted_surface_stress").text == "true"
    assert float(surface.find("Surface_tension").text) == 1.0
    assert surface.find("Kinematic_enforcement").text == "MeshNitsche"
    assert float(surface.find("Kinematic_nitsche_gamma").text) == 10.0
    assert surface.find("Tangential_mesh_policy").text == "Free"
    assert surface.find("Contact_line_model") is None
    assert mesh.find("Harmonic_quantity").text == "velocity"
    assert case["study"] == "spatial" and case["protocol_run"]
    assert case["steps"] == 2900 and case["output_cadence"] == 29
    assert case["end_time"] == pytest.approx(4 * case["inviscid_period"])
    assert case["free_surface_nodes"] == list(range(16 * 9, 17 * 9))


def test_study_roles_and_diagnostic_steps(tmp_path):
    assert gen.study_role(64, 1, None) == "spatial"
    assert gen.study_role(32, 2, None) == "time"
    assert gen.study_role(16, 2, None) is None
    assert gen.study_role(64, 1, 4.0) is None
    t = gen.generate(32, tmp_path / "t", dt_divisor=4)
    assert t["study"] == "time" and t["steps"] == 11600 and t["output_cadence"] == 116
    d = gen.generate(64, tmp_path / "d", dt_over_capillary_limit=8.0)
    assert not d["protocol_run"] and 4.0 < d["dt_over_capillary_limit"] <= 8.0
    assert d["steps"] % 100 == 0
    protocol = gen.generate(32, tmp_path / "p")
    assert protocol["protocol_run"] and protocol["initial_amplitude"] == pytest.approx(0.0025)
    assert protocol["amplitude_over_wavelength"] == pytest.approx(0.0025)
    legacy = gen.generate(32, tmp_path / "a", amplitude_over_wavelength=0.01)
    assert not legacy["protocol_run"] and legacy["initial_amplitude"] == pytest.approx(0.01)
    points, _, faces, _ = gen.liquid_triangle_mesh(32, 0.01)
    top = points[faces[gen.FREE_SURFACE][0]]
    assert np.allclose(top[:, 1], gen.MEAN_LEVEL + 0.01 * np.cos(gen.WAVENUMBER * top[:, 0]))
    assert np.allclose(gen.initial_pressure(top, 0.01), 4.0 * gen.initial_pressure(top))
    # The legacy amplitude writes the decks of the earlier protocol exactly:
    # the same initial pressure as capillary_wave_2d.
    assert np.array_equal(gen.initial_pressure(points, 0.01), unfitted.initial_pressure(points))
    with pytest.raises(ValueError):
        gen.generate(24, tmp_path / "bad")
    with pytest.raises(ValueError):
        gen.generate(32, tmp_path / "bad2", dt_divisor=2, dt_over_capillary_limit=2.0)


def test_mesh_measures_are_exact_for_a_polygonal_surface():
    points, cells, faces, _ = gen.liquid_triangle_mesh(32)
    k, width = gen.WAVENUMBER, gen.WIDTH
    area, moment = ver.mesh_measures(points[:, :2], cells, k)
    assert area == pytest.approx(width * gen.MEAN_LEVEL, rel=1e-13)
    # The cos(kx) coefficient of the P1 interpolant of a0 cos(kx) on nodes
    # x_i = i h: exactly a0 * (sin(kh/2)/(kh/2))^2 for a full half period.
    kh = k * width / 16
    amplitude = ver.CWV.mode_amplitude(moment, width, k)
    assert amplitude == pytest.approx(gen.AMPLITUDE * (math.sin(kh / 2) / (kh / 2)) ** 2, rel=1e-10)
    # Right triangles of aspect 1.014, slightly sheared under the crest.
    assert 40.0 < ver.minimum_angle_degrees(points[:, :2], cells) < math.degrees(math.atan(1 / 1.014))


def write_synthetic_run(run, level, divisor, omega_error, beta_error=0.0, *, samples=40,
                        area_leak=0.0, drop_last=False):
    """Emulate solver output: the mesh displaced by a damped standing wave whose
    frequency and damping differ from the fitted Prosperetti reference by the
    given relative errors."""
    case = gen.generate(level, run, dt_divisor=divisor)
    points, cells, _, _ = gen.liquid_triangle_mesh(level)
    k, a0 = case["wavenumber"], case["initial_amplitude"]
    times = case["end_time"] * np.arange(0, samples + 1) / samples
    params = dict(wavenumber=k, kinematic_viscosity=case["kinematic_viscosity"],
                  surface_tension=case["surface_tension"], density=case["density"])
    ref = ver.fit_damped_cosine(times, ver.reference.prosperetti_amplitude(
        times, initial_amplitude=a0, **params), case["omega0"])
    omega = ref["omega"] * (1.0 + omega_error)
    beta = ref["beta"] * (1.0 + beta_error)
    height = gen.MEAN_LEVEL + a0 * np.cos(k * points[:, 0])
    entries = []
    for i in range(1, samples + 1):
        if drop_last and i == samples:
            break
        t = times[i]
        eta = a0 * (math.exp(-beta * t) * math.cos(omega * t) - 1.0) * np.cos(k * points[:, 0]) \
            + area_leak * t
        disp = np.zeros((len(points), 3))
        disp[:, 1] = eta * points[:, 1] / height
        name = f"result_{i:03d}.vtu"
        writer.write_vtu(run / name, points, cells,
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
        for level, (ew, eb) in spatial.items():
            run = tmp_path / f"s{len(counter)}" / f"L{level}"
            write_synthetic_run(run, level, 1, ew, eb, **kwargs)
            runs.append(str(run))
        for divisor, (ew, eb) in timing.items():
            run = tmp_path / f"s{len(counter)}" / f"L32_dt{divisor}"
            write_synthetic_run(run, 32, divisor, ew, eb, **kwargs)
            runs.append(str(run))
        return runs
    return make


# Second-order time errors of the frequency and damping at dt/d, and spatial
# errors that the spatial study must recover after their removal.
TIME = {1: (-4.0e-4, 2.0e-3), 2: (-1.0e-4, 5.0e-4), 4: (-2.5e-5, 1.25e-4)}
SPATIAL = {16: (8e-3, 3e-2), 32: (2e-3, 8e-3), 64: (5e-4, 2e-3)}


def with_time_error(spatial, divisor=1):
    return {lv: (e[0] + TIME[divisor][0], e[1] + TIME[divisor][1]) for lv, e in spatial.items()}


def test_time_error_is_removed_from_the_spatial_study(study, tmp_path, capsys):
    timing = {d: (SPATIAL[32][0] + TIME[d][0], SPATIAL[32][1] + TIME[d][1]) for d in (2, 4)}
    runs = study(with_time_error(SPATIAL), timing)
    out = tmp_path / "report.json"
    assert ver.main([*runs, "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    dts = report["summary"]["time_step_study"]
    assert dts["observed_order_omega"] == pytest.approx(2.0, abs=1e-2)
    assert dts["observed_order_beta"] == pytest.approx(2.0, abs=1e-2)
    by_level = {r["level"]: r for r in report["runs"] if r["study"] == "spatial"}
    for level, (ew, eb) in SPATIAL.items():
        assert by_level[level]["frequency_spatial_signed_error"] == pytest.approx(ew, rel=2e-3, abs=2e-7)
        assert by_level[level]["damping_spatial_signed_error"] == pytest.approx(eb, rel=2e-3, abs=2e-6)
        # A cosine mode leaves the area of the mesh unchanged.
        assert by_level[level]["liquid_area_relative_deviation_max"] < 1e-12
        assert by_level[level]["surface_end_node_wall_offset_max"] == 0.0
    text = capsys.readouterr().out
    for crit in ("frequency", "damping", "volume"):
        assert f"[PASS] {crit}" in text


def test_failures_are_reported(study, capsys):
    stagnating = {lv: (2e-3, 8e-3) for lv in SPATIAL}
    timing = {d: (2e-3 + TIME[d][0], 8e-3 + TIME[d][1]) for d in (2, 4)}
    assert ver.main(study(with_time_error(stagnating), timing)) == 1
    out = capsys.readouterr().out
    assert "[FAIL] frequency" in out and "[FAIL] damping" in out
    too_slow = {**SPATIAL, 32: (3e-2, 8e-3)}
    timing = {d: (3e-2 + TIME[d][0], 8e-3 + TIME[d][1]) for d in (2, 4)}
    assert ver.main(study(with_time_error(too_slow), timing)) == 1
    assert "lambda/h=32: 0.03 > 0.02" in capsys.readouterr().out
    timing = {d: (SPATIAL[32][0] + TIME[d][0], SPATIAL[32][1] + TIME[d][1]) for d in (2, 4)}
    leak = study(with_time_error(SPATIAL), timing, area_leak=1e-4)
    assert ver.main(leak) == 1
    assert "[FAIL] volume" in capsys.readouterr().out
    no_dt = study(with_time_error(SPATIAL), {})
    assert ver.main(no_dt) == 1
    assert "missing frequency_spatial_relative_error" in capsys.readouterr().out


def test_runs_of_the_earlier_amplitude_are_reported_not_gated(study, tmp_path, capsys):
    # D35: a run written at a0 = 0.01 lambda before the change (its case.json
    # still marks it as a protocol run) is reported as a diagnostic, so it
    # cannot complete or replace the protocol study.
    timing = {d: (SPATIAL[32][0] + TIME[d][0], SPATIAL[32][1] + TIME[d][1]) for d in (2, 4)}
    runs = study(with_time_error(SPATIAL), timing)
    old = Path(runs[0])
    case = json.loads((old / "case.json").read_text())
    for key in ("amplitude_over_wavelength", "protocol_amplitude_over_wavelength"):
        case.pop(key)
    case["initial_amplitude"] = 0.01
    (old / "case.json").write_text(json.dumps(case))
    out = tmp_path / "report.json"
    assert ver.main([*runs, "--json", str(out)]) == 1
    text = capsys.readouterr().out
    assert "is not the protocol value 0.0025" in text
    assert "order needs lambda/h=[16, 32, 64]" in text
    report = json.loads(out.read_text())
    flagged = [r for r in report["runs"] if r["run"] == str(old)]
    assert flagged and not flagged[0]["protocol_run"]


def test_missing_and_incomplete_data_fail_clearly(tmp_path, capsys):
    pytest.importorskip("pyvista")
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 16, 1, 0.0, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    gen.generate(16, smoke, max_steps=5)
    assert ver.main([str(smoke)]) == 2
    assert "no solver output" in capsys.readouterr().err
    other = tmp_path / "other"
    unfitted.generate(16, "surface_stress", other)
    assert ver.main([str(other)]) == 2
    assert "not a fitted_capillary_wave_2d case" in capsys.readouterr().err


def test_richardson_extrapolation():
    r = ver.richardson([4e-3, 2e-3, 1e-3], [1.0 + 16e-6, 1.0 + 4e-6, 1.0 + 1e-6])
    assert r["order"] == pytest.approx(2.0) and r["limit"] == pytest.approx(1.0, abs=1e-15)
    flat = ver.richardson([4e-3, 2e-3, 1e-3], [1.0, 1.0 + 1e-9, 1.0])
    assert math.isnan(flat["order"]) and flat["limit"] == 1.0
