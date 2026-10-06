"""Checks of the sessile_drop_2d benchmark scripts on synthetic data (no solver run)."""

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = (Path(__file__).resolve().parent
             / "cases/fluid/free_surface_benchmarks/sessile_drop_2d")


def load(name):
    spec = importlib.util.spec_from_file_location(f"sessile_drop_2d_{name}", BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load("generate_case")
ver = load("verify")


def sampled_cap(case, level, theta_degrees, area):
    """P1 samples of the circular cap with the given angle and area."""
    points, tris, _, _ = gen.structured_triangle_mesh(case["box"], level)
    theta = math.radians(theta_degrees)
    radius = gen.cap_radius_for_area(area, theta)
    centre = (case["centre_offset_x"], -radius * math.cos(theta))
    phi = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) - radius
    return points, tris, phi


def test_cap_geometry_initial_state_and_box():
    for theta_e, theta_0 in gen.INITIAL_ANGLE_DEG.items():
        geometry = gen.case_geometry(theta_e, theta_0)
        eq, init = geometry["equilibrium"], geometry["initial"]
        assert eq["radius"] == 1.0
        assert init["area"] == pytest.approx(eq["area"], rel=1e-14)
        assert abs(theta_0 - theta_e) == 30.0
        # 90 degree cap: semicircle of area pi R^2 / 2
        assert gen.cap_geometry(1.0, math.pi / 2)["area"] == pytest.approx(math.pi / 2)
        x0, x1, y0, y1 = geometry["box"]
        assert y0 == 0.0 and x0 == -x1
        for cap in (eq, init):
            assert x1 - gen.CENTRE_OFFSET_X - cap["half_extent"] >= gen.BOX_MARGIN - 1e-12
            assert y1 - cap["apex_height"] >= gen.BOX_MARGIN - 1e-12
        assert (2 * x1 / gen.GRID_UNIT) == pytest.approx(round(2 * x1 / gen.GRID_UNIT))


@pytest.mark.parametrize("theta_e", gen.EQUILIBRIUM_ANGLES_DEG)
def test_contact_angle_measurement_on_sampled_equilibrium_caps(theta_e, tmp_path):
    for level, angle_limit in ((16, 0.2), (32, 0.08)):
        case = gen.generate(level, theta_e, "surface_stress", tmp_path / f"c{level}")
        points, tris, phi = sampled_cap(case, level, theta_e, case["equilibrium_cap_nominal"]["area"])
        area, _ = ver.liquid_area_centroid(points[:, :2], tris, phi)
        reference = ver.equilibrium_cap(area, math.radians(theta_e))
        snap = {"points": points[:, :2], "tris": tris, "phi": phi}
        g = ver.drop_geometry(snap, case, reference["radius"])
        assert abs(g["contact_angle_left"] - theta_e) < angle_limit
        assert abs(g["contact_angle_right"] - theta_e) < angle_limit
        assert min(g["angle_fit_points"]) >= 8
        assert abs(g["base_half_width"] / reference["base_half_width"] - 1) < 1e-3
        assert abs(g["apex_height"] / reference["apex_height"] - 1) < 1e-3
        assert g["circle_fit_angle"] == pytest.approx(theta_e, abs=angle_limit)
        assert g["height_base_angle"] == pytest.approx(theta_e, abs=0.2)
        # The single wall-cell angle is only reported; it is much coarser.
        assert abs(g["cell_contact_angle_left"] - theta_e) < 15.0


def test_generated_case_uses_the_d4_configuration(tmp_path):
    case = gen.generate(16, 60, "surface_stress", tmp_path / "c")
    text = (tmp_path / "c/solver.xml").read_text()
    root = ET.fromstring(text)
    fluid = root.find("Add_equation[@type='fluid']")
    free_surface = fluid.find("Add_BC[@name='free_surface']")
    get = lambda key: free_surface.find(key).text.strip()
    assert get("Contact_line_model") == "PrescribedAngle"
    assert get("Contact_line_wall_face") == "wall_bottom"
    assert get("Contact_line_wall_normal") == "0 -1 0"
    assert float(get("Contact_angle_degrees")) == 60.0
    assert get("Wall_slip_model") == "Navier"
    assert float(get("Wall_slip_length")) == pytest.approx(0.125)
    assert get("Surface_tension_form") == "SurfaceStress"
    assert get("Geometry_tangent_policy") == "RefreshedFrozenQuadrature"
    assert float(get("Active_domain_smoothing_width")) == 0.0
    bottom = fluid.find("Add_BC[@name='wall_bottom']")
    assert bottom.find("Effective_direction").text.strip() == "0 1"
    for wall in ("wall_left", "wall_right", "wall_top"):
        assert fluid.find(f"Add_BC[@name='{wall}']").find("Effective_direction") is None
    level_set = root.find("Add_equation[@type='level_set']")
    assert level_set.find("Enable_reinitialization").text.strip() == "false"
    assert level_set.find("Enable_kinematic_reconciliation").text.strip() == "true"
    assert case["kinematic_reconciliation"] is True
    plain = gen.solver_xml("surface_stress", 60, gen.time_schedule(16, 5.0, 100), 10, 1,
                           kinematic_reconciliation=False)
    assert "Enable_kinematic_reconciliation" not in plain
    assert level_set.find("Enable_sign_definite_patch_bounds").text.strip() == "true"
    assert case["sign_definite_patch_bounds"] is True
    unbounded = gen.solver_xml("surface_stress", 60, gen.time_schedule(16, 5.0, 100), 10, 1,
                               sign_definite_patch_bounds=False)
    assert "Enable_sign_definite_patch_bounds" not in unbounded
    assert "<Enable_kinematic_reconciliation>true" in unbounded
    general = root.find("GeneralSimulationParameters")
    assert general.find("Transient_time_integration_scheme") is None
    assert float(general.find("Spectral_radius_of_infinite_time_step").text) == 0.5
    assert general.find("Enable_adaptive_time_loop") is None
    assert fluid.find("LS").get("type") == "GMRES"
    assert fluid.find("LS/Linear_algebra").get("type") == "fsils"
    assert case["linear_solver"] == "fsils"
    assert case["time_integration_scheme"] == "GeneralizedAlpha"
    assert "SVMP_" not in text
    # Default transport: the harmonic PDE extension, monolithic coupling (tracker M4, 2026-10-05).
    assert case["transport"] == "pde_extension"
    level_set_eq = root.find("Add_equation[@type='level_set']")
    assert level_set_eq.find("Velocity_source").text.strip() == "prescribed_data"
    assert level_set_eq.find("Advection_velocity_extension_method").text.strip() == "pde_harmonic"
    assert level_set_eq.find("Advection_velocity_extension_coupling").text.strip() == "monolithic"
    coupled = gen.solver_xml("surface_stress", 60, gen.time_schedule(16, 5.0, 100), 10, 1,
                             transport="coupled")
    assert "<Velocity_source>coupled_field" in coupled
    wet = gen.solver_xml("surface_stress", 60, gen.time_schedule(16, 5.0, 100), 10, 1,
                         transport="wet_extension")
    assert "<Use_wet_extension_advection_velocity>true" in wet
    assert "wall_compatible_normal" in wet
    pde = gen.solver_xml("surface_stress", 60, gen.time_schedule(16, 5.0, 100), 10, 1,
                         transport="pde_extension")
    assert "<Advection_velocity_extension_method>pde_harmonic<" in pde
    assert "<Advection_velocity_extension_coupling>monolithic<" in pde
    assert "<Velocity_source>prescribed_data<" in pde
    assert "Use_wet_extension_advection_velocity" not in pde
    assert root.find("GeneralSimulationParameters/Number_of_time_steps").text == str(case["steps"])
    assert case["dt"] <= math.sqrt(case["h"] ** 3 / (4.0 * math.pi)) * (1 + 1e-12)
    assert case["steps"] * case["dt"] == pytest.approx(5.0 * case["viscous_time"])
    assert case["viscosity"] == pytest.approx(math.sqrt(2.0 / 12.0))
    assert case["slip_length_over_h"] == pytest.approx(2.0)
    assert case["initial_angle_degrees"] == 90.0
    assert case["min_abs_phi_over_h"] > 1e-3
    assert case["dry_gap_over_h"] >= 4.0 - 1e-9
    for wall in gen.WALLS:
        assert (tmp_path / f"c/mesh/mesh-surfaces/{wall}.vtp").is_file()
    schedule = gen.time_schedule(16, 5.0, 100)
    lumped = gen.solver_xml("kag_lumped", 90, schedule, 10, 1)
    assert "KinematicAreaGradientTraction" in lumped and "mass>Lumped" in lumped
    consistent = gen.solver_xml("kag_consistent", 90, schedule, 10, 1)
    assert "kinematic_area_gradient_mass" not in consistent
    direct = gen.solver_xml("surface_stress", 90, schedule, 10, 1, linear_solver="eigen_direct",
                            time_integration="backward_euler")
    assert '<LS type="Direct">' in direct and 'Linear_algebra type="eigen"' in direct
    assert "<Transient_time_integration_scheme>BackwardEuler" in direct
    assert "Spectral_radius_of_infinite_time_step" not in direct
    maintained = ET.fromstring(gen.solver_xml("surface_stress", 120, schedule, 10, 1,
                                              reinitialization=True))
    level_set = maintained.find("Add_equation[@type='level_set']")
    assert level_set.find("Enable_reinitialization").text.strip() == "true"
    assert level_set.find("Reinitialization_cadence_steps").text.strip() == "10"
    assert level_set.find("Reinitialization_max_iterations").text.strip() == "4"


def write_synthetic_run(run, level, angle_offset=0.0, theta_e=60, growth=False,
                        area_shift=0.0, drop_last=False, max_steps=None, dt_rule="fixed",
                        dt_divisor=1, mid_angle_offset=0.0,
                        semi_implicit="NormalIncrement"):
    """Emulate solver output: a relaxed cap with a known angle and a decaying velocity.

    The initial level set in the mesh file is replaced by the same cap, so the
    liquid area is conserved exactly unless area_shift moves the last output.
    mid_angle_offset changes the cap angle at the middle output only.
    """
    case = gen.generate(level, theta_e, "surface_stress", run, snapshots=8, max_steps=max_steps,
                        dt_rule=dt_rule, dt_divisor=dt_divisor, semi_implicit=semi_implicit)
    points, tris, phi = sampled_cap(case, level, theta_e + angle_offset,
                                    case["equilibrium_cap_nominal"]["area"])
    _, _, phi_mid = sampled_cap(case, level, theta_e + angle_offset + mid_angle_offset,
                                case["equilibrium_cap_nominal"]["area"])
    n = len(phi)
    zeros = np.zeros((n, 3))
    gen.write_vtu(run / "mesh/mesh-complete.mesh.vtu", points, tris,
                  {"GlobalNodeID": ("Int64", np.arange(n)), "phi": ("Float64", phi),
                   "Velocity": ("Float64", zeros), "Pressure": ("Float64", np.ones(n))},
                  {"GlobalElementID": ("Int64", np.arange(len(tris)))})
    swirl = np.column_stack([points[:, 1], -points[:, 0], np.zeros(n)])
    outputs = 8 if max_steps is None else case["steps"]
    entries = []
    for k in range(1, outputs + 1):
        if drop_last and k == outputs:
            break
        t = k * case["output_cadence"] * case["dt"]
        amp = 1e-3 * (math.exp(t / case["viscous_time"]) if growth
                      else math.exp(-t / case["viscous_time"]))
        values = (phi_mid if k == outputs // 2 else phi) + (area_shift if k == outputs else 0.0)
        name = f"result_{k * case['output_cadence']:03d}.vtu"
        gen.write_vtu(run / name, points, tris,
                      {"phi": ("Float64", values), "Velocity": ("Float64", amp * swirl),
                       "Pressure": ("Float64", np.ones(n))},
                      {"GlobalElementID": ("Int64", np.arange(len(tris)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    return case


@pytest.fixture
def study(tmp_path):
    pytest.importorskip("pyvista")

    def make(overrides=None, levels=(16, 32, 64), dt_divisor=1, tag="", **common):
        overrides = overrides or {}
        runs = []
        for level in levels:
            opts = {"angle_offset": 1.2 * 16 / level, **common}
            opts.update(overrides.get(level, {}))
            run = tmp_path / f"L{level}_dt{dt_divisor}{tag}"
            write_synthetic_run(run, level, dt_divisor=dt_divisor, **opts)
            runs.append(str(run))
        return runs
    return make


def test_synthetic_refinement_study_passes(study, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert ver.main([*study(), "--json", str(out)]) == 0
    # A single time step: the time-step criterion is reported, not failed.
    assert "[NOT EVALUATED] time_step" in capsys.readouterr().out
    report = json.loads(out.read_text())
    assert report["time_step_criterion"][0]["evaluated"] is False
    group = report["groups"][0]
    assert group["passed"] and group["equilibrium_angle_degrees"] == 60.0
    assert group["dt_divisor"] == 1 and group["dt_rule"] == "fixed"
    run16, run32 = group["runs"][0], group["runs"][1]
    assert run16["contact_angle_error_degrees"] == pytest.approx(1.2, abs=0.2)
    assert run32["contact_angle_error_degrees"] == pytest.approx(0.6, abs=0.05)
    assert run32["contact_angle_asymmetry_degrees"] < 0.05
    assert run32["base_radius_relative_error"] < 0.02
    assert run32["liquid_area_relative_drift_max"] == 0.0
    assert run32["base_radius_relative_error_final_area"] == \
        pytest.approx(run32["base_radius_relative_error"])
    assert len(run32["history"]["contact_angle_left"]) == 8


def test_final_area_errors_remove_the_transported_area(tmp_path):
    # Decision D10: the cap with the final area removes the area that the
    # transport gained, the only way dt enters the equilibrium end state.
    pytest.importorskip("pyvista")
    run = tmp_path / "grown"
    write_synthetic_run(run, 32, area_shift=-1.0e-2)
    result = ver.analyse_run(run)
    assert result["liquid_area_relative_drift_final"] > 1.0e-2
    assert result["base_radius_relative_error"] > 1.0e-2
    assert result["base_radius_relative_error_final_area"] < \
        0.5 * result["base_radius_relative_error"]


def test_angle_error_growth_and_area_drift_fail(study, capsys):
    runs = study({16: {"growth": True}, 32: {"angle_offset": 3.0},
                  64: {"area_shift": 2e-3}})
    assert ver.main(runs) == 1
    out = capsys.readouterr().out
    assert "[FAIL] contact_angle" in out
    assert "[FAIL] no_velocity_growth" in out
    assert "[FAIL] volume_drift" in out


def test_missing_incomplete_and_truncated_data_fail_clearly(study, tmp_path, capsys):
    runs = study(levels=(16, 32))
    assert ver.main(runs) == 1                          # no R/h = 64 run
    assert "monotonicity needs R/h=[64]" in capsys.readouterr().out
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 16, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    empty = tmp_path / "empty"
    gen.generate(16, 60, "surface_stress", empty)
    assert ver.main([str(empty)]) == 2
    assert "no solver output" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    write_synthetic_run(smoke, 16, max_steps=5, dt_rule="capillary-limit", semi_implicit="None")
    assert ver.main([str(smoke)]) == 2
    assert "truncated smoke runs" in capsys.readouterr().err
    assert ver.main([str(smoke), "--allow-truncated"]) == 1   # one level only


def test_a_drop_off_the_wall_is_rejected(tmp_path):
    case = gen.generate(16, 90, "surface_stress", tmp_path / "c")
    points, tris, _, _ = gen.structured_triangle_mesh(case["box"], 16)
    floating = np.hypot(points[:, 0], points[:, 1] - 0.7) - 0.5
    snap = {"points": points[:, :2], "tris": tris, "phi": floating}
    with pytest.raises(ver.DataError, match="two wall contact points"):
        ver.drop_geometry(snap, case, 1.0)


def test_time_step_rules_and_semi_implicit_term(tmp_path):
    # Default: the capillary-limit rule of each level, without the term.
    protocol = gen.generate(32, 120, "surface_stress", tmp_path / "p")
    assert protocol["dt_rule"] == "capillary-limit" and protocol["dt_divisor"] == 1
    assert protocol["surface_tension_semi_implicit"] == "None"
    assert "Surface_tension_semi_implicit" not in (tmp_path / "p/solver.xml").read_text()
    # Fixed rule: one step for every level, the capillary-limit step of R/h = 16.
    coarse = gen.time_schedule(16, 5.0, 100)
    assert coarse["steps"] == 2800 and coarse["output_cadence"] == 28
    for level in gen.LEVELS:
        fixed = gen.time_schedule(level, 5.0, 100, dt_rule="fixed")
        assert fixed["dt"] == coarse["dt"] and fixed["steps"] == 2800
        assert fixed["output_cadence"] == 28
    assert gen.time_schedule(64, 5.0, 100, dt_rule="fixed")["dt_over_capillary_limit"] == \
        pytest.approx(8.0, rel=0.01)
    assert coarse["dt"] == pytest.approx(4.374e-3, rel=1e-4)
    # The term goes into the free-surface block.
    case = gen.generate(32, 120, "surface_stress", tmp_path / "f", dt_rule="fixed",
                        semi_implicit="NormalIncrement")
    root = ET.parse(tmp_path / "f/solver.xml").getroot()
    free_surface = root.find("Add_equation[@type='fluid']/Add_BC[@name='free_surface']")
    assert free_surface.findtext("Surface_tension_semi_implicit") == "NormalIncrement"
    assert (tmp_path / "f/solver.xml").read_text().count("Surface_tension_semi_implicit>") == 2
    assert case["surface_tension_semi_implicit"] == "NormalIncrement"
    assert case["dt_rule"] == "fixed" and case["steps"] == 2800
    assert float(root.findtext("GeneralSimulationParameters/Time_step_size")) == coarse["dt"]
    # Divisors refine the step exactly and keep the output times.
    for divisor in (2, 4):
        half = gen.generate(32, 120, "surface_stress", tmp_path / f"d{divisor}", dt_rule="fixed",
                            dt_divisor=divisor, semi_implicit="NormalIncrement")
        assert half["dt"] == case["dt"] / divisor and half["dt_base"] == case["dt"]
        assert half["steps"] == divisor * 2800 and half["output_cadence"] == divisor * 28
        assert half["output_cadence"] * half["dt"] == case["output_cadence"] * case["dt"]
        assert half["end_time"] == pytest.approx(case["end_time"], rel=1e-14)
    # A multiple of the limit, rounded down to the outputs; m = 2 halved is the m = 1 deck.
    double = gen.generate(32, 120, "surface_stress", tmp_path / "m2", dt_rule="fixed",
                          dt_multiple=2.0, semi_implicit="NormalIncrement")
    assert double["steps"] == 1400 and double["dt"] == 2.0 * case["dt"]
    gen.generate(32, 120, "surface_stress", tmp_path / "m2d2", dt_rule="fixed", dt_multiple=2.0,
                 dt_divisor=2, semi_implicit="NormalIncrement")
    assert (tmp_path / "m2d2/solver.xml").read_bytes() == (tmp_path / "f/solver.xml").read_bytes()
    with pytest.raises(ValueError):
        gen.generate(16, 60, "surface_stress", tmp_path / "bad1", dt_divisor=3)
    with pytest.raises(ValueError):
        gen.generate(16, 60, "surface_stress", tmp_path / "bad2", dt_rule="per-period")
    with pytest.raises(ValueError):
        gen.generate(16, 60, "surface_stress", tmp_path / "bad3", semi_implicit="Implicit")
    with pytest.raises(ValueError):                     # the solver rejects this combination
        gen.generate(16, 60, "surface_stress", tmp_path / "bad4", transport="wet_extension",
                     semi_implicit="NormalIncrement")
    with pytest.raises(ValueError):
        gen.generate(16, 60, "surface_stress", tmp_path / "bad5", dt_multiple=0.0)


def test_capillary_limit_protocol_is_reproducible(tmp_path):
    """--dt-rule capillary-limit --surface-tension-semi-implicit None: the decks before the options."""
    steps = {}
    for level in gen.LEVELS:
        out = tmp_path / f"cl{level}"
        assert gen.main(["--level", str(level), "--contact-angle", "60", "--output-dir", str(out),
                         "--dt-rule", "capillary-limit",
                         "--surface-tension-semi-implicit", "None"]) == 0
        case = json.loads((out / "case.json").read_text())
        text = (out / "solver.xml").read_text()
        assert "Surface_tension_semi_implicit" not in text
        assert case["dt"] <= gen.DT_SAFETY * gen.capillary_dt_limit(1.0 / level) * (1 + 1e-12)
        assert case["steps"] == 100 * case["output_cadence"]
        assert case["dt"] == case["end_time"] / case["steps"]
        steps[level] = case["steps"]
        default = tmp_path / f"default{level}"
        gen.generate(level, 60, "surface_stress", default)
        assert (default / "solver.xml").read_bytes() == text.encode()
    assert steps == {16: 2800, 32: 7900, 64: 22300}           # tracker M4
    # At R/h = 16 the fixed rule without the term writes the same deck.
    gen.generate(16, 60, "surface_stress", tmp_path / "fixed16", dt_rule="fixed")
    assert (tmp_path / "fixed16/solver.xml").read_bytes() == \
        (tmp_path / "cl16/solver.xml").read_bytes()


def test_time_step_criterion_passes_at_dt_and_dt_over_2(study, tmp_path, capsys):
    runs = study()
    half = study(dt_divisor=2, levels=(16, 32))         # the dt/2 study needs R/h = 16 and 32
    quarter = study(dt_divisor=4, levels=(16,))         # extra check, R/h = 16 only
    out = tmp_path / "dt.json"
    assert ver.main([*runs, *half, *quarter, "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "dt divisor 1" in text and "dt divisor 2" in text and "dt divisor 4" in text
    assert "[PASS] time_step: R/h=32 (finest common level)" in text
    assert "R/h=16, dt/2 vs dt/4 (reported)" in text
    assert "monotonicity not evaluated at this step" in text
    report = json.loads(out.read_text())
    assert [g["dt_divisor"] for g in report["groups"]] == [1, 2, 4]
    assert [g["passed"] for g in report["groups"]] == [True, True, True]
    crit = report["time_step_criterion"][0]
    assert crit["evaluated"] and crit["passed"] and crit["level"] == 32
    assert all(q["value"] == 0.0 for q in crit["quantities"])
    assert crit["changes"]["matched_outputs"] == 8 and crit["changes"]["unmeasured_outputs"] == 0
    # The dt study needs every level, R/h = 64 included.
    coarse = study(levels=(16, 32), tag="no64")
    assert ver.main([*coarse, *half]) == 1
    text = capsys.readouterr().out
    assert "monotonicity needs R/h=[64]" in text and "[PASS] time_step" in text


def test_time_step_criterion_fails_on_the_end_state_and_on_the_history(study, capsys):
    runs = study()
    moved = study({32: {"angle_offset": 0.6 + 0.3}}, dt_divisor=2, levels=(16, 32), tag="end")
    assert ver.main([*runs, *moved]) == 1
    text = capsys.readouterr().out
    assert "[FAIL] time_step" in text and "contact_angle_change_final 0.3" in text
    assert "[FAIL] contact_angle" not in text           # both studies pass their gates
    history = study({32: {"mid_angle_offset": 1.0}}, dt_divisor=2, levels=(16, 32), tag="mid")
    assert ver.main([*runs, *history]) == 1
    text = capsys.readouterr().out
    assert "[FAIL] time_step" in text and "contact_angle_change_history 1" in text
    assert "contact_angle_change_final 0 <= 0.1" in text
    # Each rule and multiple is its own study: the capillary-limit runs do not pair with these.
    old = study(levels=(16, 32), tag="old", dt_rule="capillary-limit", semi_implicit="None")
    assert ver.main([*runs, *old]) == 1                 # old study misses R/h = 64
    text = capsys.readouterr().out
    assert "dt rule capillary-limit, dt divisor 1" in text and "dt per level" in text
    assert text.count("[NOT EVALUATED] time_step") == 2


def test_every_gate_applies_at_dt_over_2(study, capsys):
    runs = study()
    drifting = study({16: {"area_shift": 2e-3}}, dt_divisor=2, levels=(16, 32), tag="drift")
    assert ver.main([*runs, *drifting]) == 1
    assert "[FAIL] volume_drift" in capsys.readouterr().out
    partial = study(dt_divisor=2, levels=(16,), tag="p")    # the dt/2 study needs R/h = 32
    assert ver.main([*runs, *partial]) == 1
    text = capsys.readouterr().out
    assert "missing run at R/h=32" in text
    assert "[PASS] time_step: R/h=16 (finest common level)" in text
