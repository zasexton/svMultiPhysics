"""Checks of the oscillating_drop_2d benchmark scripts on synthetic data (no solver run)."""

import gzip
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent / "cases/fluid/free_surface_benchmarks"
BENCHMARK = BENCHMARKS / "oscillating_drop_2d"


def load(name, directory=BENCHMARK):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(f"{directory.name}_{name}_for_test",
                                                  directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load("drop_reference")
gen = load("generate_case")
ver = load("verify")
amp_check = load("finite_amplitude")
planar = load("prosperetti_reference", BENCHMARKS / "capillary_wave_2d")

N = 2
OMEGA0 = math.sqrt(6.0)                         # rho = gamma = R = 1, n = 2
NU = 0.05                                       # protocol La = 800


def params(nu=NU, n=N, a0=None, radius=1.0):
    p = dict(mode=n, kinematic_viscosity=nu, surface_tension=1.0, density=1.0, radius=radius)
    if a0 is not None:
        p["initial_amplitude"] = a0
    return p


def bessel_i_series(n, x, terms=120):
    term = (x / 2) ** n / math.factorial(n)
    total = term
    for k in range(1, terms):
        term = term * (x / 2) ** 2 / (k * (k + n))
        total += term
    return total


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------
def test_bessel_functions_match_power_series():
    for n in (2, 3):
        for x in (0.3, 2 + 1j, 5 + 5j, 7.3j, -3 + 4j, 10.0):
            exact = x * 0.5 * (bessel_i_series(n - 1, x) + bessel_i_series(n + 1, x)) / bessel_i_series(n, x)
            assert abs(ref.bessel_ratio(n, x)[0] - exact) < 1e-13 * abs(exact)
    xi = np.array([0.01, 0.5, 1.9, 2.1, 3.0, 6.0, 10.0])        # J_m(xi) = I_m(i xi) / i^m
    for m in (1, 2, 3):
        exact = np.array([(bessel_i_series(m, 1j * v) / 1j ** m).real for v in xi])
        assert np.max(np.abs(ref.bessel_j(m, xi) - exact)) < 1e-12    # series round-off at xi = 10


def test_inviscid_frequency_and_limit():
    assert ref.inviscid_frequency(2, 1.0, 1.0, 1.0) == pytest.approx(math.sqrt(6.0))
    assert ref.inviscid_frequency(3, 2.0, 0.5, 2.0) == pytest.approx(math.sqrt(24 * 2.0 / (0.5 * 8.0)))
    t = np.linspace(0.0, 10.0, 201)
    assert np.max(np.abs(ref.drop_amplitude(t, **params(0.0, a0=1.0)) - np.cos(OMEGA0 * t))) == 0.0
    # At eps = nu / (omega0 R^2) = 4e-5 the history is the weakly damped cosine
    # exp(-2 n (n-1) nu t / R^2) cos(omega0 t) to O(eps).
    nu = 1e-4
    t = np.linspace(0.5, 10.0, 191)
    weak = ref.drop_amplitude(t, **params(nu, a0=1.0))
    assert np.max(np.abs(weak - np.exp(-4.0 * nu * t) * np.cos(OMEGA0 * t))) < 2.5e-4
    assert np.max(np.abs(weak - np.cos(OMEGA0 * t))) > 3e-3
    with pytest.raises(ValueError, match="mode"):
        ref.inviscid_frequency(1, 1.0, 1.0, 1.0)


def test_closed_form_transform_matches_independent_collocation():
    # a_hat from the potential/vortical (Bessel) solution against a Chebyshev
    # collocation of the stream-function equations, which uses no Bessel function.
    for nu in (NU, 0.002, 1.0):
        for s in (0.5 * OMEGA0, OMEGA0, 3.0 * OMEGA0, 1 + 2j, -0.05 + 2j, -0.1 + 0.5j, 5 - 20j):
            closed = ref.laplace_transform(s, **params(nu, a0=1.0))[0]
            colloc = ref.laplace_transform_collocation(s, points=40, **params(nu, a0=1.0))
            assert abs(colloc - closed) < 1e-9 * abs(closed)


def test_initial_value_solution_starts_at_rest_and_matches_its_transform():
    p = params(a0=1.0)
    ex = ref.modal_expansion(min_time=1e-3, **p)
    total = 2.0 * ex["complex_residue"].real + ex["real_residues"].sum()
    assert total == pytest.approx(1.0, abs=1e-9)                  # a(0+) = a0

    def history(t):
        t = np.asarray(t, dtype=float)
        return (2.0 * (ex["complex_residue"] * np.exp(ex["complex_root"] * t)).real
                + (ex["real_residues"][None, :] * np.exp(np.outer(t, ex["real_roots"]))).sum(axis=1))

    for t in (1e-3, 1e-2):                                         # a = a0 (1 - omega0^2 t^2 / 2) + o(t^2)
        assert history([t])[0] == pytest.approx(1.0 - 3.0 * t * t, abs=5e-7 * (t / 1e-2) ** 3)
    t = np.array([1e-3, 0.1, 1.0, 5.0])
    assert np.max(np.abs(ref.drop_amplitude(t, **p) - history(t))) < 1e-15
    # The Laplace transform of the residue sum is the closed form a_hat(s), so
    # no root is missing; checked by quadrature with t = tau^2 beyond t0.
    t0 = 1e-3
    for s in (0.5 * OMEGA0, OMEGA0, 3.0 * OMEGA0):
        head = t0 - s * t0 ** 2 / 2 + (s * s - OMEGA0 ** 2) * t0 ** 3 / 6
        tau = np.linspace(math.sqrt(t0), math.sqrt(60.0 / s), 40001)
        f = history(tau ** 2) * np.exp(-s * tau ** 2) * 2.0 * tau
        h = tau[1] - tau[0]
        integral = head + h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
        exact = ref.laplace_transform(s, **p)[0].real
        assert integral == pytest.approx(exact, rel=1e-9)
        series = (2.0 * (ex["complex_residue"] / (s - ex["complex_root"])).real
                  + (ex["real_residues"] / (s - ex["real_roots"])).sum())
        assert series == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_normal_mode_tends_to_the_weak_viscosity_expansion(n):
    # beta = 2 n (n-1) nu/R^2 (1 - (n-1) sqrt(eps/2)), omega = omega0 (1 - sqrt(2) n (n-1)^2 eps^1.5)
    w0 = ref.inviscid_frequency(n, 1.0, 1.0, 1.0)
    previous = None
    for eps in (1e-4, 1e-5, 1e-6):
        mode = ref.normal_mode(**params(eps * w0, n))
        assert mode["epsilon"] == pytest.approx(eps)
        db = mode["beta"] / mode["weak_damping_rate"] - 1.0
        dw = mode["omega"] / w0 - 1.0
        assert db / (-(n - 1) * math.sqrt(eps / 2.0)) == pytest.approx(1.0, abs=0.01)
        assert dw / (-math.sqrt(2.0) * n * (n - 1) ** 2 * eps ** 1.5) == pytest.approx(1.0, abs=0.025)
        if previous is not None:
            assert abs(db) < abs(previous)
        previous = db
        assert abs(ref.dispersion_function(complex(-mode["beta"], mode["omega"]),
                                           **params(eps * w0, n))[0]) < 1e-9 * w0 ** 2


def test_normal_mode_agrees_with_collocation_pole():
    # The pole of the collocation transform (secant iteration on 1/a_hat) is the normal mode.
    mode = ref.normal_mode(**params())
    s0, s1 = mode["s"] * (1 + 1e-3), mode["s"] * (1 - 1e-3)
    f = lambda s: 1.0 / ref.laplace_transform_collocation(s, points=40, **params(a0=1.0))  # noqa: E731
    f0, f1 = f(s0), f(s1)
    for _ in range(30):
        s0, s1, f0 = s1, s1 - f1 * (s1 - s0) / (f1 - f0), f1
        f1 = f(s1)
        if abs(s1 - s0) < 1e-14:
            break
    assert abs(s1 - mode["s"]) < 1e-9 * abs(mode["s"])


def test_stokes_limit_of_the_slowest_real_mode():
    previous = None
    for nu in (10.0, 30.0, 100.0):
        s = ref.real_modes(**params(nu), xi_max=10.0)
        error = abs(s[0] / -ref.stokes_rate(2, nu, 1.0, 1.0) - 1.0)
        assert error < 2e-3
        if previous is not None:
            assert error < previous
        previous = error


def test_large_mode_number_tends_to_lambs_planar_relation():
    k, nu = 2.0 * math.pi, 1.0 / math.sqrt(3000.0)
    flat = planar.normal_mode(wavenumber=k, kinematic_viscosity=nu, surface_tension=1.0, density=1.0)
    errors = []
    for n in (50, 100, 200):
        mode = ref.normal_mode(**params(nu, n, radius=n / k))
        assert mode["omega"] / flat["omega"] - 1.0 == pytest.approx(0.0, abs=3e-4)
        errors.append(abs(mode["beta"] / flat["beta"] - 1.0))
    assert errors[2] < 0.005 and errors[0] / errors[1] == pytest.approx(2.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.05)


def test_protocol_reference_and_its_fit():
    mode = ref.normal_mode(**params())
    assert mode["omega"] == pytest.approx(2.425868579800437, rel=1e-12)
    assert mode["beta"] == pytest.approx(0.17733415627196703, rel=1e-11)
    # The weak-viscosity rate 2 n (n-1) nu/R^2 overestimates the damping by 13%,
    # omega0 the frequency by 1%: the comparison uses the exact solution.
    assert mode["beta"] / ref.weak_damping_rate(2, NU, 1.0) == pytest.approx(0.8867, abs=1e-4)
    assert mode["omega"] / OMEGA0 == pytest.approx(0.99036, abs=1e-5)
    t = np.linspace(0.0, 4.0 * 2.0 * math.pi / OMEGA0, 101)
    fit = ver.fit_damped_cosine(t, ref.drop_amplitude(t, **params(a0=0.01)), OMEGA0)
    assert fit["converged"]
    assert fit["omega"] / mode["omega"] - 1.0 == pytest.approx(0.0, abs=2e-3)
    assert fit["beta"] / mode["beta"] - 1.0 == pytest.approx(0.0, abs=1e-2)


def test_finite_amplitude_shift_is_negligible_at_the_protocol_amplitude():
    r = amp_check.simulate(0.02, points=32, steps_per_period=100, periods=2.0)
    assert r["converged"] and r["area_drift"] < 1e-9
    assert r["coefficient"] == pytest.approx(-0.770, abs=0.01)
    assert abs(r["coefficient"]) * gen.AMPLITUDE_OVER_RADIUS ** 2 < 1e-4


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def test_fit_recovers_a_damped_cosine():
    t = np.linspace(0.0, 10.0, 101)
    y = 0.01 * np.exp(-0.18 * t) * np.cos(2.4 * t + 0.3)
    fit = ver.fit_damped_cosine(t, y, OMEGA0)
    assert fit["converged"]
    assert fit["omega"] == pytest.approx(2.4, rel=1e-10)
    assert fit["beta"] == pytest.approx(0.18, rel=1e-9)


def test_harmonic_moments_are_exact_for_polygons():
    points, tris, _, _ = gen.SD.structured_triangle_mesh(8)
    points = points[:, :2]
    # The whole box [0, 3]^2 about its centre: M0 = 9, M2 = 0, M4 = -16 a^6/15 (a = 1.5).
    m = ver.liquid_moments(points, tris, -np.ones(len(points)), (1.5, 1.5))
    assert m[0].real == pytest.approx(9.0, rel=1e-14)
    assert abs(m[1]) < 1e-13 and abs(m[2]) < 1e-12 and abs(m[3]) < 1e-12
    assert m[4].real == pytest.approx(-16.0 * 1.5 ** 6 / 15.0, rel=1e-13)
    # A cut through every row of triangles: the rectangle [0, x0] x [0, 3] about the origin.
    x0 = 1.234
    m = ver.liquid_moments(points, tris, points[:, 0] - x0, (0.0, 0.0))
    for order in range(5):
        exact = sum(math.comb(order, j) * 1j ** j * x0 ** (order - j + 1) / (order - j + 1)
                    * 3.0 ** (j + 1) / (j + 1) for j in range(order + 1))
        assert abs(m[order] - exact) < 1e-12 * max(1.0, abs(exact))
    c, central = ver.central_moments(m)
    assert c == pytest.approx(complex(x0 / 2, 1.5), rel=1e-13)
    assert abs(central[1]) < 1e-12


def test_sampled_drop_converges_at_second_order():
    area_err, amp_err = [], []
    for level in gen.LEVELS:
        points, tris, _, _ = gen.SD.structured_triangle_mesh(level)
        phi = gen.initial_level_set(points)
        snap = {"points": points[:, :2], "tris": tris, "phi": phi}
        case = {"centre": gen.drop_centre().tolist(), "mode": 2}
        m = ver.snapshot_measures(snap, case)
        area_err.append(abs(m["area"] / math.pi - 1.0))
        amp_err.append(abs(m["amplitude"] / (gen.area_preserving_radius() * 0.01) - 1.0))
        assert np.hypot(*(np.array(m["centroid"]) - gen.drop_centre())) < 3e-5
    assert ver.observed_order(gen.LEVELS, area_err) > 1.9
    assert ver.observed_order(gen.LEVELS, amp_err) > 1.9
    assert area_err[0] < 3e-3 and amp_err[0] < 2e-3
    # The shape-mode measure of the exact (unsampled) shape is R0 eps to O(eps^4).
    eps = 0.01
    theta = np.linspace(0.0, 2.0 * math.pi, 4097)[:-1]
    rho = gen.surface_radius(theta, eps)
    m2 = np.mean(rho ** 4 / 4 * np.exp(2j * theta)) * 2 * math.pi
    area = np.mean(rho ** 2 / 2) * 2 * math.pi
    assert area == pytest.approx(math.pi, rel=1e-14)
    assert (m2 / (math.pi * math.sqrt(area / math.pi) ** 3)).real == pytest.approx(
        gen.area_preserving_radius(eps) * eps, rel=1e-7)


def test_initial_pressure_is_the_released_state():
    points, _, _, _ = gen.SD.structured_triangle_mesh(16)
    p = gen.initial_pressure(points)
    r, theta, dx, dy = gen.polar(points)
    a0 = gen.area_preserving_radius() * 0.01
    # gamma/R + 3 gamma a0 (x^2 - y^2)/R^4: harmonic, and gamma*kappa on r = R.
    assert np.max(np.abs(p - (1.0 + 3.0 * a0 * (dx ** 2 - dy ** 2)))) < 1e-13
    on_circle = 1.0 + 3.0 * a0 * np.cos(2.0 * theta)
    assert np.max(np.abs(p - on_circle)[np.abs(r - 1.0) < 1e-12], initial=0.0) < 1e-13


def test_generated_case_is_complete_and_respects_time_step_rule(tmp_path):
    case = gen.generate(16, "surface_stress", tmp_path / "c")
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    text = (tmp_path / "c/solver.xml").read_text()
    assert "<Geometry_tangent_policy>RefreshedFrozenQuadrature" in text
    assert "<Surface_tension_form>SurfaceStress" in text and "Curvature_field" not in text
    assert "Enable_sign_definite_patch_bounds" not in text           # D21: sessile only
    assert case["transport"] == gen.DEFAULT_TRANSPORT == "pde_extension"
    level_set = root.find("Add_equation[@type='level_set']")
    assert level_set.findtext("Velocity_source") == "prescribed_data"
    assert level_set.findtext("Advection_velocity_extension_method") == "pde_harmonic"
    assert level_set.findtext("Advection_velocity_extension_coupling") == "monolithic"
    assert level_set.findtext("Enable_kinematic_reconciliation") == "true"   # D14
    assert root.find("GeneralSimulationParameters/Number_of_time_steps").text == str(case["steps"])
    fluid = root.find("Add_equation[@type='fluid']")
    assert float(fluid.findtext("Viscosity/Value")) == pytest.approx(0.05, rel=1e-14)
    bcs = {bc.get("name"): bc for bc in fluid.findall("Add_BC")}
    assert all(bcs[w].find("Effective_direction") is None for w in gen.WALLS)
    free_surface = bcs["free_surface"]
    assert free_surface.findtext("Surface_tension_semi_implicit") == "NormalIncrement"
    assert free_surface.findtext("Generated_interface_domain_id") == "oscillating_drop_surface"
    # 100 steps per inviscid period at every level, 4 periods, outputs every 4 steps.
    assert case["dt"] == pytest.approx(2.0 * math.pi / OMEGA0 / 100.0, rel=1e-14)
    assert case["steps"] == 400 and case["output_cadence"] == 4
    assert case["steps_per_period"] == pytest.approx(100.0)
    for level in gen.LEVELS:
        assert gen.time_schedule(level, 800.0, 4.0, 100)["dt"] == case["dt"]
    half = gen.generate(32, "surface_stress", tmp_path / "half", dt_divisor=2)
    assert half["steps"] == 800 and half["output_cadence"] == 8
    assert half["end_time"] == pytest.approx(case["end_time"], rel=1e-14)
    assert case["min_abs_phi_over_h"] > 0.008 and half["min_abs_phi_over_h"] > 0.01
    assert case["wall_gap_over_h"] > 7.0
    assert case["normal_mode_damping_rate"] == pytest.approx(0.17733415627196703, rel=1e-11)
    assert case["initial_amplitude"] == pytest.approx(0.01 / math.sqrt(1.00005), rel=1e-14)
    for wall in gen.WALLS:
        assert (tmp_path / f"c/mesh/mesh-surfaces/{wall}.vtp").is_file()
    lumped = gen.solver_xml("kag_lumped", gen.time_schedule(8, 800.0, 4.0, 100), 10, 1)
    assert "<Curvature_projection_kinematic_area_gradient_mass>Lumped" in lumped
    plain = gen.solver_xml("surface_stress", gen.time_schedule(8, 800.0, 4.0, 100), 10, 1,
                           kinematic_reconciliation=False, semi_implicit="None")
    assert "Enable_kinematic_reconciliation" not in plain and "semi_implicit" not in plain
    with pytest.raises(ValueError, match="dt-divisor"):
        gen.generate(16, "surface_stress", tmp_path / "bad", dt_divisor=3)
    with pytest.raises(FileExistsError):
        gen.generate(16, "surface_stress", tmp_path / "c")


def test_mesh_is_the_static_drop_mesh():
    points, tris, _, n = gen.SD.structured_triangle_mesh(8)
    case_points = points
    assert n == 24 and case_points.shape[0] == 625 and tris.shape[0] == 1152


# ---------------------------------------------------------------------------
# verify.py on synthetic solver output
# ---------------------------------------------------------------------------
def write_synthetic_run(run, level, *, omega_error=0.0, beta_error=0.0, radius_drift=0.0,
                        drop_last=False, max_steps=None, snapshots=16, dt_divisor=1):
    """Emulate solver output: the released shape with a damped-cosine amplitude.

    The amplitude history is exp(-beta t) cos(omega t + phase) with omega and
    beta off those of the fitted exact history (at R_eff) by the given
    relative errors.
    """
    case = gen.generate(level, "surface_stress", run, snapshots=snapshots, max_steps=max_steps,
                        dt_divisor=dt_divisor)
    points, tris, _, _ = gen.SD.structured_triangle_mesh(level)
    n_out = case["steps"] // case["output_cadence"]
    times = np.arange(0, n_out + 1) * case["output_cadence"] * case["dt"]
    area0 = ver.liquid_moments(points[:, :2], tris, gen.initial_level_set(points),
                               case["centre"])[0].real
    r_eff = math.sqrt(area0 / math.pi)
    a0 = case["initial_amplitude"]
    a_ref = ref.drop_amplitude(times, **params(radius=r_eff, a0=a0))
    if max_steps is None:
        fit = ver.fit_damped_cosine(times, a_ref, case["omega0"])
        omega, beta = fit["omega"] * (1 + omega_error), fit["beta"] * (1 + beta_error)
        amp = fit["amplitude"] * np.exp(-beta * times) * np.cos(omega * times + fit["phase"])
    else:
        amp = a_ref
    eps = amp / a0 * gen.AMPLITUDE_OVER_RADIUS
    gen.SD.write_vtu(run / "mesh/mesh-complete.mesh.vtu", points, tris,
                     {"phi": ("Float64", gen.initial_level_set(points, eps[0]))},
                     {"GlobalElementID": ("Int64", np.arange(len(tris)))})
    entries = []
    for k in range(1, n_out + 1):
        if drop_last and k == n_out:
            break
        phi = gen.initial_level_set(points, eps[k]) - radius_drift * k / n_out
        step = k * case["output_cadence"]
        name = f"result_{step:03d}.vtu"
        gen.SD.write_vtu(run / name, points, tris, {"phi": ("Float64", phi)},
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
            opts = {"omega_error": 0.012 * (8 / level) ** 1.5, "beta_error": 0.08 * (8 / level) ** 1.5}
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
    for run, level in zip(group["runs"], gen.LEVELS):
        assert run["frequency_relative_error"] == pytest.approx(0.012 * (8 / level) ** 1.5,
                                                                abs=2e-4)
        assert run["damping_rate_relative_error"] == pytest.approx(0.08 * (8 / level) ** 1.5,
                                                                   abs=2e-3)
        assert run["liquid_area_relative_drift_max"] < 1e-5      # P1 sampling of the shapes
        assert run["periods_simulated"] == pytest.approx(4.0 * run["reference_omega0"] / OMEGA0)
    finest = group["runs"][2]
    assert finest["effective_radius_relative"] == pytest.approx(-8e-5, abs=2e-5)
    assert finest["sin_mode_max_over_a0"] < 1e-3


def test_large_frequency_error_and_slow_damping_convergence_fail(study, capsys):
    runs = study({32: {"omega_error": 0.03, "beta_error": 0.04},
                  8: {"beta_error": 0.05}, 16: {"beta_error": 0.045}})
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] frequency" in out and "R/h=32: 0.03" in out
    assert "[FAIL] damping" in out and "<= 0.05" in out and "observed order 0.1" in out
    assert "[PASS] volume_drift" in out


def test_damping_above_the_limit_at_the_finest_level_fails(study, capsys):
    runs = study({8: {"beta_error": 0.24}, 16: {"beta_error": 0.12}, 32: {"beta_error": 0.06}})
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] damping: R/h=32: 0.0599" in out and "> 0.05" in out


def test_volume_drift_fails(study, capsys):
    runs = study({16: {"radius_drift": 1e-4}})                 # area drift 2e-4
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] volume_drift" in out and "[PASS] frequency" in out


def test_time_step_criterion_passes_at_the_finest_level(study, tmp_path, capsys):
    runs = study()
    fine = tmp_path / "L32_dt2"
    case = write_synthetic_run(fine, 32, omega_error=0.0005, beta_error=0.01, dt_divisor=2)
    assert case["dt_divisor"] == 2 and case["steps"] == 800
    out_json = tmp_path / "dt.json"
    assert ver.main([*runs, str(fine), "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "dt divisor 2, dt = 0.0128255 (200 steps per period)" in out
    assert "time-step study surface_stress, transport pde_extension, La = 800, R/h = 32" in out
    assert "[PASS] time_step: R/h=32 (finest common level)" in out
    assert "order not evaluated at this step (needs R/h=[8, 16, 32])" in out
    crit = json.loads(out_json.read_text())["time_step_criterion"][0]
    assert crit["evaluated"] and crit["passed"] and crit["level"] == 32
    # dt study at R/h = 32: omega +0.15%, beta +1.0%; dt/2: +0.05%, +1.0%.
    assert crit["changes"]["frequency"] == pytest.approx(0.0010, abs=2e-5)
    assert crit["changes"]["damping"] == pytest.approx(0.0, abs=5e-4)


def test_time_step_criterion_fails_and_dt2_study_is_gated(study, tmp_path, capsys):
    runs = study()
    fine = tmp_path / "L32_dt2"
    write_synthetic_run(fine, 32, omega_error=0.0045, beta_error=0.03, dt_divisor=2)
    assert ver.main([*runs, str(fine)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] time_step" in out and "frequency change 2.987e-03 > 0.002" in out
    assert "damping change 1.9" in out and "> 0.01" in out
    other = tmp_path / "L16_dt2"
    write_synthetic_run(other, 16, dt_divisor=2)
    assert ver.main([*runs, str(other)]) == 1                 # dt/2 study without R/h = 32
    assert "missing run at R/h=32" in capsys.readouterr().out


def test_time_step_criterion_not_evaluated_with_one_divisor(study, tmp_path, capsys):
    runs = study()
    assert ver.main(runs) == 0
    assert "[NOT EVALUATED] time_step" in capsys.readouterr().out
    extra = tmp_path / "L32_dt2"
    write_synthetic_run(extra, 32, dt_divisor=2)
    assert ver.main([str(extra)]) == 1                         # no protocol-time-step run at all
    assert "criteria cannot be applied" in capsys.readouterr().out


def test_spatial_study_requires_one_shared_time_step(study, capsys):
    runs = study()
    case_file = Path(runs[2]) / "case.json"
    case = json.loads(case_file.read_text())
    case["dt"] *= 1.5
    case_file.write_text(json.dumps(case))
    assert ver.main(runs) == 2
    assert "one shared step (D10)" in capsys.readouterr().err


def test_area_criterion_and_log_statistics_use_the_solver_log(study, tmp_path, capsys):
    runs = study()
    run = Path(runs[1])
    snap = ver.read_level_set(run / "mesh/mesh-complete.mesh.vtu", {"level_set_field": "phi"})
    centre = json.loads((run / "case.json").read_text())["centre"]
    area0 = ver.liquid_moments(snap["points"], snap["tris"], snap["phi"], centre)[0].real
    lines = [f"[svMultiPhysics::Application] Wet volume diagnostic step={n} time=0 field='phi' "
             f"domain_id='oscillating_drop_surface' marker=1 physical_wet_volume={v!r} "
             f"initial_wet_volume={area0!r}\n"
             for n, v in enumerate([area0, area0 * (1 + 3e-5), area0 * (1 + 2e-4), area0])]
    steps = [f"[svMultiPhysics::Application] TimeLoop: nonlinear_done step={n} time=0 converged=1 "
             f"iters={2 + n % 2} ||r||=1e-9 outer_iters={3 + n % 2} inner_iters_total=3\n"
             "+++ NEWTON SOLVER TIMING (rank 0) +++\n"
             f"  Total Newton time:      0.5 s  (2 Newton iters, 2 assemblies, {200 + n} linear iters)\n"
             for n in range(4)]
    with gzip.open(run / "solver_run.log.gz", "wt") as log:
        log.write("unrelated line\n" + "".join(lines) + "".join(steps))
    (run / "run.txt").write_text("job=1 node=x ranks=4 start=0\nexit=0 elapsed_s=30 end=1\n")
    out_json = tmp_path / "log.json"
    assert ver.main([*runs, "--json", str(out_json)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] volume_drift" in out
    # Steps 1-3 (step 0 is the initial state); GMRES counts every Newton solve.
    assert "3 steps, outer passes 3.67 (max 4), Newton 2.67/step, GMRES 101/Newton" in out
    r = json.loads(out_json.read_text())["groups"][0]["runs"][1]
    assert r["liquid_area_logged_steps"] == 4
    assert r["liquid_area_relative_drift_max"] == pytest.approx(2e-4, rel=1e-9)
    assert r["liquid_area_relative_drift_max_outputs"] < 1e-6
    assert r["solver_log"]["wall_seconds"] == 30 and r["solver_log"]["ranks"] == 4


def test_missing_incomplete_and_truncated_data(study, tmp_path, capsys):
    runs = study()
    assert ver.main(runs[:2]) == 1                      # no R/h = 32 run
    assert "order needs evaluable runs" in capsys.readouterr().out
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 8, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    empty = tmp_path / "empty"
    gen.generate(8, "surface_stress", empty)
    assert ver.main([str(empty)]) == 2
    assert "no solver output" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    write_synthetic_run(smoke, 32, max_steps=5)
    assert ver.main([str(smoke)]) == 2
    assert "truncated" in capsys.readouterr().err
    out_json = tmp_path / "smoke.json"
    assert ver.main([str(smoke), "--allow-truncated", "--json", str(out_json)]) == 1
    out = capsys.readouterr().out
    assert "R/h=32: not evaluable" in out and "TRUNCATED" in out
    run = json.loads(out_json.read_text())["groups"][0]["runs"][0]
    assert run["outputs"] == 5 and run["fit"] is None
    assert run["amplitude_max_error"] < 5e-3            # sampling error of the synthetic shape
