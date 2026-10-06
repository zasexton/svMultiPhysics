"""Checks of the capillary_wave_2d benchmark scripts on synthetic data (no solver run)."""

import gzip
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = (Path(__file__).resolve().parent
             / "cases/fluid/free_surface_benchmarks/capillary_wave_2d")


def load(name):
    if str(BENCHMARK) not in sys.path:
        sys.path.insert(0, str(BENCHMARK))
    spec = importlib.util.spec_from_file_location(f"capillary_wave_2d_{name}", BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load("prosperetti_reference")
gen = load("generate_case")
ver = load("verify")

K = 2.0 * math.pi
OMEGA0 = math.sqrt(K ** 3)                       # rho = gamma = lambda = 1


def nu_for(epsilon):
    return epsilon * OMEGA0 / K ** 2


def params(nu, a0=1.0):
    return dict(wavenumber=K, kinematic_viscosity=nu, surface_tension=1.0, density=1.0,
                initial_amplitude=a0)


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------
def test_erfcx_matches_real_erfc_and_complex_taylor_series():
    x = np.linspace(-3.0, 6.0, 91)
    exact = np.array([math.exp(v * v) * math.erfc(v) for v in x])
    assert np.max(np.abs(ref.erfcx(x) - exact) / exact) < 1e-13

    def erf_series(z):
        return 2.0 / math.sqrt(math.pi) * sum((-1) ** n * z ** (2 * n + 1) / (math.factorial(n) * (2 * n + 1))
                                              for n in range(80))
    rng = np.random.default_rng(1)
    z = rng.uniform(-2.0, 2.0, 100) + 1j * rng.uniform(-2.0, 2.0, 100)
    series = np.array([np.exp(v * v) * (1.0 - erf_series(v)) for v in z])
    assert np.max(np.abs(ref.erfcx(z) - series) / np.abs(series)) < 1e-11


def test_reference_starts_at_rest_from_the_initial_amplitude():
    nu = 1.0 / math.sqrt(3000.0)
    a = ref.prosperetti_amplitude([0.0, 1e-8], **params(nu, 0.01))
    assert a[0] == pytest.approx(0.01, rel=1e-12)
    assert (a[1] - a[0]) / 1e-8 == pytest.approx(0.0, abs=1e-4)      # da/dt(0) = 0


def test_reference_inviscid_limit_is_the_capillary_dispersion_relation():
    t = np.linspace(0.0, 5.0, 301)
    a = ref.prosperetti_amplitude(t, **params(0.0))
    assert np.max(np.abs(a - np.cos(OMEGA0 * t))) < 1e-10
    assert ref.inviscid_frequency(K, 1.0, 1.0) == pytest.approx(math.sqrt(K ** 3))
    # The benchmark depth (about one wavelength) changes omega0 by less than 1e-5.
    deep = ref.inviscid_frequency(K, 1.0, 1.0)
    finite = ref.inviscid_frequency(K, 1.0, 1.0, depth=gen.MEAN_LEVEL)
    assert abs(finite / deep - 1.0) < 1e-5


def test_reference_matches_its_laplace_transform_at_the_protocol_viscosity():
    # a_hat(s) comes from the linearized Navier-Stokes equations directly
    # (prosperetti_reference.laplace_transform), independently of the
    # erfc partial-fraction form; t = tau^2 removes the sqrt(t) behaviour.
    p = params(1.0 / math.sqrt(3000.0), 0.01)
    for s in (0.5 * OMEGA0, OMEGA0, 3.0 * OMEGA0):
        tau = np.linspace(0.0, math.sqrt(60.0 / s), 20001)
        f = ref.prosperetti_amplitude(tau ** 2, **p) * np.exp(-s * tau ** 2) * 2.0 * tau
        h = tau[1] - tau[0]
        integral = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
        assert integral == pytest.approx(float(ref.laplace_transform(s, **p)), rel=1e-10)


def test_normal_mode_tends_to_omega0_and_weak_damping_rate():
    # Lamb: s = -2 nu k^2 + i omega0 + (sqrt(2) - i sqrt(2)) eps^(3/2) omega0 + ...
    previous = None
    for eps in (1e-2, 1e-3, 1e-4):
        nu = nu_for(eps)
        mode = ref.normal_mode(wavenumber=K, kinematic_viscosity=nu, surface_tension=1.0, density=1.0)
        dw = mode["omega"] / OMEGA0 - 1.0
        db = mode["beta"] / ref.weak_damping_rate(K, nu) - 1.0
        assert dw / (-math.sqrt(2.0) * eps ** 1.5) == pytest.approx(1.0, abs=0.01)
        assert db / (-math.sqrt(eps / 2.0)) == pytest.approx(1.0, abs=0.01)
        assert mode["epsilon"] == pytest.approx(eps)
        if previous is not None:
            assert abs(db) < abs(previous)
        previous = db
        # The same root is z^2 - sigma for the quartic root with Re z < 0.
        sigma = nu * K ** 2
        z = ref.prosperetti_roots(sigma, OMEGA0)
        s = [zi * zi - sigma for zi in z if zi.real < 0.0 and (zi * zi).imag > 0.0]
        assert len(s) == 1
        assert abs(s[0] - complex(-mode["beta"], mode["omega"])) < 1e-12 * OMEGA0


def test_reference_history_damping_and_frequency_limits():
    # A damped-cosine fit of Prosperetti's a(t) over four periods recovers
    # omega0 and 2 nu k^2 at weak viscosity ...
    t = np.linspace(0.0, 4.0 * 2.0 * math.pi / OMEGA0, 101)
    nu = nu_for(1e-4)
    fit = ver.fit_damped_cosine(t, ref.prosperetti_amplitude(t, **params(nu)), OMEGA0)
    assert fit["converged"]
    assert fit["omega"] / OMEGA0 - 1.0 == pytest.approx(0.0, abs=1e-5)
    assert fit["beta"] / ref.weak_damping_rate(K, nu) - 1.0 == pytest.approx(0.0, abs=0.01)
    # ... and, at the protocol La = 3000, stays within 1% of the normal mode,
    # which itself is 15% below 2 nu k^2 (so 2 nu k^2 cannot be the reference).
    nu = 1.0 / math.sqrt(3000.0)
    fit = ver.fit_damped_cosine(t, ref.prosperetti_amplitude(t, **params(nu)), OMEGA0)
    mode = ref.normal_mode(wavenumber=K, kinematic_viscosity=nu, surface_tension=1.0, density=1.0)
    assert fit["omega"] / mode["omega"] - 1.0 == pytest.approx(0.0, abs=2e-3)
    assert fit["beta"] / mode["beta"] - 1.0 == pytest.approx(0.0, abs=1e-2)
    assert mode["beta"] / ref.weak_damping_rate(K, nu) == pytest.approx(0.845, abs=0.005)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def test_fit_recovers_a_damped_cosine():
    t = np.linspace(0.0, 1.6, 101)
    y = 0.02 * np.exp(-1.3 * t) * np.cos(14.0 * t + 0.4)
    fit = ver.fit_damped_cosine(t, y, OMEGA0)
    assert fit["converged"]
    assert fit["omega"] == pytest.approx(14.0, rel=1e-10)
    assert fit["beta"] == pytest.approx(1.3, rel=1e-9)
    assert fit["amplitude"] == pytest.approx(0.02, rel=1e-9)
    assert fit["phase"] == pytest.approx(0.4, abs=1e-9)


def test_mode_amplitude_is_exact_for_a_planar_surface():
    points, tris, _, _ = gen.structured_triangle_mesh(16)
    points = points[:, :2]
    w = gen.BOX_WIDTH
    # Flat surface: no cos(kx) content; area exact.
    area, moment = ver.liquid_measures(points, tris, points[:, 1] - 1.01, K)
    assert area == pytest.approx(1.01 * w, rel=1e-14)
    assert abs(ver.mode_amplitude(moment, w, K)) < 1e-15
    # Tilted surface eta = 1.01 + c (x - lambda/4): coefficient -8 c / (k^2 lambda).
    c = 0.03
    area, moment = ver.liquid_measures(points, tris, points[:, 1] - 1.01 - c * (points[:, 0] - 0.25), K)
    assert area == pytest.approx(1.01 * w, rel=1e-14)
    assert ver.mode_amplitude(moment, w, K) == pytest.approx(-8.0 * c / K ** 2, rel=1e-12)
    assert ver.wall_height(points, points[:, 1] - 1.01 - c * (points[:, 0] - 0.25), 0.0, 1e-9) == \
        pytest.approx(1.01 - 0.25 * c, abs=1e-14)


def test_sampled_cosine_amplitude_converges_at_second_order():
    errors = []
    for level in gen.LEVELS:
        points, tris, _, _ = gen.structured_triangle_mesh(level)
        phi = gen.initial_level_set(points)
        area, moment = ver.liquid_measures(points[:, :2], tris, phi, K)
        assert area == pytest.approx(gen.BOX_WIDTH * gen.MEAN_LEVEL, rel=1e-12)
        errors.append(abs(ver.mode_amplitude(moment, gen.BOX_WIDTH, K) / 0.01 - 1.0))
    assert ver.observed_order(gen.LEVELS, errors) > 1.9
    assert errors[1] < 0.005


def test_generated_case_is_complete_and_respects_time_step_rule(tmp_path):
    case = gen.generate(16, "surface_stress", tmp_path / "c")
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    text = (tmp_path / "c/solver.xml").read_text()
    assert "<Geometry_tangent_policy>RefreshedFrozenQuadrature" in text
    assert "<Surface_tension_form>SurfaceStress" in text and "Curvature_field" not in text
    # D9: the default transport is the harmonic PDE extension, monolithic coupling.
    assert case["transport"] == gen.DEFAULT_TRANSPORT == "pde_extension"
    level_set = root.find("Add_equation[@type='level_set']")
    assert level_set.findtext("Velocity_source") == "prescribed_data"
    assert level_set.find("Use_wet_extension_advection_velocity") is None
    assert level_set.findtext("Advection_velocity_extension_method") == "pde_harmonic"
    assert level_set.findtext("Advection_velocity_extension_coupling") == "monolithic"
    assert root.find("GeneralSimulationParameters/Number_of_time_steps").text == str(case["steps"])
    fluid = root.find("Add_equation[@type='fluid']")
    bcs = {bc.get("name"): bc for bc in fluid.findall("Add_BC")}
    assert bcs["wall_left"].findtext("Effective_direction") == "1 0"
    assert bcs["wall_right"].findtext("Effective_direction") == "1 0"
    assert bcs["wall_bottom"].findtext("Effective_direction") == "0 1"
    assert bcs["wall_top"].find("Effective_direction") is None
    assert float(fluid.findtext("Viscosity/Value")) == pytest.approx(1.0 / math.sqrt(3000.0))
    # D13: 50 steps per inviscid period at every level, with the lagged
    # normal-increment term in the free-surface block.
    assert case["dt_rule"] == "steps-per-period" and case["dt_divisor"] == 1
    assert case["dt"] == pytest.approx(2.0 * math.pi / OMEGA0 / 50.0, rel=1e-14)
    assert case["steps"] * case["dt"] == pytest.approx(4.0 * 2.0 * math.pi / OMEGA0)
    assert case["steps"] == 100 * case["output_cadence"] == 200
    assert case["steps_per_period"] == pytest.approx(50.0)
    for level in gen.LEVELS:
        assert gen.time_schedule(level, 3000.0, 4.0, 100)["dt"] == case["dt"]
    free_surface = fluid.find("Add_BC[@name='free_surface']")
    assert free_surface.findtext("Surface_tension_semi_implicit") == "NormalIncrement"
    assert case["surface_tension_semi_implicit"] == "NormalIncrement"
    half = gen.generate(64, "surface_stress", tmp_path / "half", dt_divisor=2)
    assert half["steps"] == 400 and half["output_cadence"] == 4
    assert half["steps_per_period"] == pytest.approx(100.0)
    assert half["end_time"] == pytest.approx(case["end_time"], rel=1e-14)
    assert case["min_abs_phi_over_h"] > 0.02 and case["top_gap_over_h"] > 2.0
    assert case["epsilon"] == pytest.approx(0.0458, abs=1e-4)
    for wall in gen.WALLS:
        assert (tmp_path / f"c/mesh/mesh-surfaces/{wall}.vtp").is_file()
    finer = gen.generate(16, "kag_lumped", tmp_path / "d", dt_divisor=4)
    assert finer["dt"] == pytest.approx(case["dt"] / 4.0, rel=1e-14)
    assert finer["steps"] == 4 * case["steps"]
    assert "<Curvature_projection_kinematic_area_gradient_mass>Lumped" in (tmp_path / "d/solver.xml").read_text()
    consistent = gen.solver_xml("kag_consistent", gen.time_schedule(16, 3000.0, 4.0, 100), 10, 1)
    assert "KinematicAreaGradientTraction" in consistent
    assert "kinematic_area_gradient_mass" not in consistent


def test_transport_options(tmp_path):
    schedule = gen.time_schedule(16, 3000.0, 4.0, 100)
    coupled = gen.solver_xml("surface_stress", schedule, 10, 1, transport="coupled")
    assert "<Velocity_source>coupled_field" in coupled and "Use_wet_extension" not in coupled
    wet = gen.solver_xml("surface_stress", schedule, 10, 1, transport="wet_extension")
    assert "<Use_wet_extension_advection_velocity>true<" in wet
    assert "<Advection_velocity_extension_method>wall_compatible_normal<" in wet
    # The PDE extension is selected by method and coupling, never together
    # with the wet-extension switch (the solver rejects that combination).
    pde = gen.solver_xml("surface_stress", schedule, 10, 1, transport="pde_extension")
    assert "<Advection_velocity_extension_method>pde_harmonic<" in pde
    assert "<Advection_velocity_extension_coupling>monolithic<" in pde
    assert "<Velocity_source>prescribed_data" in pde and "Use_wet_extension" not in pde
    with pytest.raises(ValueError, match="must be one of"):
        gen.solver_xml("surface_stress", schedule, 10, 1, transport="bogus")
    # Kinematic reconciliation of the level set is on by default.
    assert "<Enable_kinematic_reconciliation>true<" in pde
    plain = gen.solver_xml("surface_stress", schedule, 10, 1, kinematic_reconciliation=False)
    assert "Enable_kinematic_reconciliation" not in plain


def test_mesh_is_mirror_symmetric_about_the_node_line():
    points, tris, _, _ = gen.structured_triangle_mesh(32)
    mirrored = points[:, :2].copy()
    mirrored[:, 0] = gen.BOX_WIDTH - mirrored[:, 0]
    key = lambda p: {tuple(sorted(map(tuple, np.round(p[t], 12)))) for t in tris}  # noqa: E731
    assert key(points[:, :2]) == key(mirrored)


# ---------------------------------------------------------------------------
# verify.py on synthetic solver output
# ---------------------------------------------------------------------------
def write_synthetic_run(run, level, *, omega_error=0.0, beta_error=0.0, level_drift=0.0,
                        drop_last=False, max_steps=None, snapshots=16, dt_divisor=1):
    """Emulate solver output: phi = y - y0(t) - a(t) cos(kx).

    a(t) is a damped cosine whose frequency and decay rate differ from those
    of the fitted Prosperetti history by the given relative errors.
    """
    case = gen.generate(level, "surface_stress", run, snapshots=snapshots, max_steps=max_steps,
                        dt_divisor=dt_divisor)
    points, tris, _, _ = gen.structured_triangle_mesh(level)
    n_out = case["steps"] // case["output_cadence"]
    times = np.arange(0, n_out + 1) * case["output_cadence"] * case["dt"]
    p = params(case["kinematic_viscosity"], case["initial_amplitude"])
    a_ref = ref.prosperetti_amplitude(times, **p)
    if max_steps is None:
        fit = ver.fit_damped_cosine(times, a_ref, case["omega0"])
        omega, beta = fit["omega"] * (1 + omega_error), fit["beta"] * (1 + beta_error)
        amp = fit["amplitude"] * np.exp(-beta * times) * np.cos(omega * times + fit["phase"])
    else:
        amp = a_ref
    # The t = 0 sample is read from the mesh file: make it the start of the synthetic history.
    gen.write_vtu(run / "mesh/mesh-complete.mesh.vtu", points, tris,
                  {"phi": ("Float64", points[:, 1] - case["mean_level"] - amp[0] * np.cos(K * points[:, 0]))},
                  {"GlobalElementID": ("Int64", np.arange(len(tris)))})
    entries = []
    for k in range(1, n_out + 1):
        if drop_last and k == n_out:
            break
        phi = (points[:, 1] - case["mean_level"] - level_drift * k / n_out
               - amp[k] * np.cos(K * points[:, 0]))
        step = k * case["output_cadence"]
        name = f"result_{step:03d}.vtu"
        gen.write_vtu(run / name, points, tris, {"phi": ("Float64", phi)},
                      {"GlobalElementID": ("Int64", np.arange(len(tris)))})
        entries.append(f'<DataSet timestep="{times[k]:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    return case


@pytest.fixture
def study(tmp_path):
    pytest.importorskip("pyvista")

    def make(overrides=None):
        overrides = overrides or {}
        runs = []
        for level in gen.LEVELS:
            opts = {"omega_error": 0.012 * (16 / level) ** 1.5, "beta_error": 0.04 * (16 / level) ** 1.5}
            opts.update(overrides.get(level, {}))
            write_synthetic_run(tmp_path / f"L{level}", level, **opts)
            runs.append(str(tmp_path / f"L{level}"))
        return runs
    return make


def test_synthetic_refinement_study_passes(study, tmp_path):
    out = tmp_path / "report.json"
    assert ver.main([*study(), "--json", str(out)]) == 0
    group = json.loads(out.read_text())["groups"][0]
    assert group["passed"]
    run32 = group["runs"][1]
    assert run32["frequency_relative_error"] == pytest.approx(0.012 / 2 ** 1.5, rel=1e-6)
    assert run32["damping_rate_relative_error"] == pytest.approx(0.04 / 2 ** 1.5, rel=1e-6)
    assert run32["liquid_area_relative_drift_max"] < 1e-12
    assert run32["periods_simulated"] == pytest.approx(4.0)


def test_large_frequency_error_and_slow_damping_convergence_fail(study, capsys):
    # Damping errors 0.04 / 0.02 / 0.03: within the limit at lambda/h = 64 but not converging.
    runs = study({32: {"omega_error": 0.03, "beta_error": 0.02},
                  16: {"beta_error": 0.04}, 64: {"beta_error": 0.03}})
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] frequency" in out and "0.03 > 0.02" in out
    assert "[FAIL] damping" in out and "0.03 <= 0.05" in out and "observed order 0.21" in out
    assert "[PASS] volume_drift" in out


def test_damping_is_gated_at_the_finest_level(study, capsys):
    # D20: damping errors 0.30 / 0.071 / 0.0039 (order 2.9) exceed the limit at lambda/h = 32
    # but pass at the finest level.
    runs = study({16: {"beta_error": 0.30}, 32: {"beta_error": 0.071}, 64: {"beta_error": 0.0039}})
    assert ver.main(runs) == 0
    out = capsys.readouterr().out
    assert "[PASS] damping" in out and "lambda/h=64: 0.0039 <= 0.05" in out
    assert "lambda/h=32" not in out.split("[PASS] damping")[1].split("[")[0]


def test_damping_above_the_limit_at_the_finest_level_fails(study, capsys):
    runs = study({16: {"beta_error": 0.24}, 32: {"beta_error": 0.12}, 64: {"beta_error": 0.06}})
    assert ver.main(runs) == 1
    assert "lambda/h=64: 0.06 > 0.05" in capsys.readouterr().out


def test_volume_drift_fails(study, capsys):
    runs = study({64: {"level_drift": 2e-4}})
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] volume_drift" in out and "[PASS] frequency" in out


def test_old_protocol_is_reproducible(tmp_path):
    """--dt-rule capillary-limit --surface-tension-semi-implicit None gives the decks before D13."""
    out = tmp_path / "old"
    assert gen.main(["--level", "16", "--output-dir", str(out), "--dt-rule", "capillary-limit",
                     "--surface-tension-semi-implicit", "None"]) == 0
    case = json.loads((out / "case.json").read_text())
    assert "Surface_tension_semi_implicit" not in (out / "solver.xml").read_text()
    assert case["dt_rule"] == "capillary-limit" and case["surface_tension_semi_implicit"] == "None"
    # D10: one step for all levels, within the capillary limit of the finest.
    h_min = 1.0 / max(gen.LEVELS)
    assert case["dt"] <= math.sqrt(h_min ** 3 / (4.0 * math.pi)) * (1 + 1e-12)
    assert case["steps"] == 100 * case["output_cadence"] == 2900
    for level in gen.LEVELS:
        assert gen.time_schedule(level, 3000.0, 4.0, 100, dt_rule="capillary-limit")["dt"] == case["dt"]
    with pytest.raises(ValueError, match="dt-rule"):
        gen.generate(16, "surface_stress", tmp_path / "bad", dt_rule="cfl")
    with pytest.raises(ValueError, match="semi-implicit"):
        gen.generate(16, "surface_stress", tmp_path / "bad2", semi_implicit="Implicit")


def test_time_step_criterion_passes_at_the_finest_level(study, tmp_path, capsys):
    runs = study()                                      # lambda/h = 64: omega 0.15%, beta 0.5% error
    fine = tmp_path / "L64_dt2"
    case = write_synthetic_run(fine, 64, omega_error=0.0005, beta_error=0.0, dt_divisor=2)
    assert case["dt_divisor"] == 2
    coarse32 = tmp_path / "L32_dt2"
    write_synthetic_run(coarse32, 32, omega_error=0.003, beta_error=0.01, dt_divisor=2)
    out_json = tmp_path / "dt.json"
    assert ver.main([*runs, str(fine), str(coarse32), "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "dt divisor 2, dt = 0.00398942 (100 steps per period)" in out
    # Other levels stay reported in the time-step study.
    assert "time-step study surface_stress, transport pde_extension, La = 3000, lambda/h = 32" in out
    assert "dt/2: omega err 3.000e-03, beta err 1.000e-02" in out
    assert "[PASS] time_step: lambda/h=64 (finest common level)" in out
    crit = json.loads(out_json.read_text())["time_step_criterion"][0]
    assert crit["evaluated"] and crit["passed"] and crit["level"] == 64
    assert crit["changes"]["frequency"] == pytest.approx(0.001 / 1.0005, rel=1e-5)
    assert crit["changes"]["damping"] == pytest.approx(0.005, rel=1e-5)


def test_time_step_criterion_fails_and_dt2_study_is_gated(study, tmp_path, capsys):
    runs = study()
    fine = tmp_path / "L64_dt2"
    write_synthetic_run(fine, 64, omega_error=0.0045, beta_error=0.02, dt_divisor=2)
    assert ver.main([*runs, str(fine)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] time_step" in out and "frequency change 2.987e-03 > 0.002" in out
    assert "damping change 1.471e-02 > 0.01" in out
    # The dt/2 study needs lambda/h = 64 only: the other levels are reported as not run.
    assert "lambda/h=32: not run at this step (not required)" in out
    assert "order not evaluated at this step" in out
    other = tmp_path / "L32_dt2"
    write_synthetic_run(other, 32, dt_divisor=2)
    assert ver.main([*runs, str(other)]) == 1           # dt/2 study without lambda/h = 64
    out = capsys.readouterr().out
    assert "missing run at lambda/h=64" in out


def test_time_step_criterion_not_evaluated_with_one_divisor(study, tmp_path, capsys):
    runs = study()
    assert ver.main(runs) == 0
    assert "[NOT EVALUATED] time_step" in capsys.readouterr().out
    extra = tmp_path / "L64_dt2"
    write_synthetic_run(extra, 64, dt_divisor=2)
    assert ver.main([str(extra)]) == 1                  # no protocol-time-step run at all
    assert "criteria cannot be applied" in capsys.readouterr().out


def test_spatial_study_requires_one_shared_time_step(study, tmp_path, capsys):
    runs = study()
    case_file = Path(runs[2]) / "case.json"
    case = json.loads(case_file.read_text())
    case["dt"] *= 1.5
    case_file.write_text(json.dumps(case))
    assert ver.main(runs) == 2
    assert "one shared step (D10)" in capsys.readouterr().err


def test_area_criterion_uses_every_logged_step(study, tmp_path, capsys):
    # D11: an area excursion between two outputs is caught from the solver log.
    runs = study()
    run = Path(runs[1])
    area0 = gen.BOX_WIDTH * gen.MEAN_LEVEL
    lines = [f"[svMultiPhysics::Application] Wet volume diagnostic step={n} time=0 field='phi' "
             f"domain_id='capillary_wave_surface' marker=1 physical_wet_volume={v!r} "
             f"initial_wet_volume={area0!r}\n"
             for n, v in enumerate([area0, area0 * (1 + 3e-5), area0 * (1 + 2e-4), area0])]
    with gzip.open(run / "solver_run.log.gz", "wt") as log:
        log.write("unrelated line\n" + "".join(lines))
    out_json = tmp_path / "log.json"
    assert ver.main([*runs, "--json", str(out_json)]) == 1
    assert "[FAIL] volume_drift" in capsys.readouterr().out
    r = json.loads(out_json.read_text())["groups"][0]["runs"][1]
    assert r["liquid_area_logged_steps"] == 4
    assert r["liquid_area_relative_drift_max"] == pytest.approx(2e-4, rel=1e-9)
    assert r["liquid_area_relative_drift_max_outputs"] < 1e-12


def test_missing_incomplete_and_truncated_data(study, tmp_path, capsys):
    runs = study()
    assert ver.main(runs[:2]) == 1                      # no lambda/h = 64 run
    assert "order needs evaluable runs" in capsys.readouterr().out
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 16, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    empty = tmp_path / "empty"
    gen.generate(16, "surface_stress", empty)
    assert ver.main([str(empty)]) == 2
    assert "no solver output" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    write_synthetic_run(smoke, 32, max_steps=5)
    assert ver.main([str(smoke)]) == 2
    assert "truncated" in capsys.readouterr().err
    out_json = tmp_path / "smoke.json"
    assert ver.main([str(smoke), "--allow-truncated", "--json", str(out_json)]) == 1
    out = capsys.readouterr().out
    assert "lambda/h=32: not evaluable" in out and "TRUNCATED" in out
    run = json.loads(out_json.read_text())["groups"][0]["runs"][0]
    assert run["outputs"] == 5 and run["fit"] is None
    assert run["amplitude_max_error"] < 1e-12            # the synthetic history is the reference
