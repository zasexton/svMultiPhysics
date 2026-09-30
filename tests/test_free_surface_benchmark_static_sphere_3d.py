"""Checks of the static_sphere_3d benchmark scripts on synthetic data (no solver run)."""

import importlib.util
import itertools
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = (Path(__file__).resolve().parent
             / "cases/fluid/free_surface_benchmarks/static_sphere_3d")


def load(name):
    spec = importlib.util.spec_from_file_location(f"static_sphere_3d_{name}", BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load("generate_case")
ver = load("verify")


def sampled_sphere(points):
    return np.linalg.norm(points - gen.sphere_centre(), axis=1) - gen.RADIUS


# ---------------------------------------------------------------------------
# Mesh and geometry
# ---------------------------------------------------------------------------
def test_mesh_is_conforming_positive_and_mirror_symmetric():
    points, cells, faces, n = gen.structured_tetra_mesh(4)
    assert cells.shape == (6 * n ** 3, 4)
    assert np.all(gen.signed_volume(points, cells) > 0.0)
    assert gen.signed_volume(points, cells).sum() == pytest.approx(gen.BOX_SIDE ** 3)
    local = [tuple(v for v in range(4) if v != omit) for omit in range(4)]
    keys = np.concatenate([np.sort(cells[:, list(lf)], axis=1) for lf in local])
    _, counts = np.unique(keys, axis=0, return_counts=True)
    assert counts.max() == 2                                   # conforming
    assert np.sum(counts == 1) == 6 * 2 * n * n                # boundary = the six walls
    for name, axis, end in gen.WALL_PLANES:
        tris, parents = faces[name]
        assert np.all(points[tris][..., axis] == end * gen.BOX_SIDE)
        assert np.all((tris[:, :, None] == cells[parents][:, None, :]).any(axis=2))
    # Reflection about the mid-plane of every axis maps the mesh onto itself.
    key = {tuple(sorted(map(tuple, np.round(points[c], 12)))) for c in cells}
    for axis in range(3):
        mirrored = points.copy()
        mirrored[:, axis] = gen.BOX_SIDE - mirrored[:, axis]
        assert {tuple(sorted(map(tuple, np.round(mirrored[c], 12)))) for c in cells} == key


def test_liquid_volume_of_tilted_half_space_is_exact():
    points, cells, _, _ = gen.structured_tetra_mesh(4)
    phi = points[:, 0] - 1.3 - 0.01 * points[:, 1] + 0.003 * points[:, 2]
    volume, centroid = ver.liquid_volume_centroid(points, cells, phi)
    # {phi < 0} = {x < w(y, z)}, w = a + b y + c z, over (y, z) in [0, L]^2.
    a, b, c, side = 1.3, 0.01, -0.003, gen.BOX_SIDE
    s1, s2 = side ** 2 / 2, side ** 3 / 3          # int_0^L y dy, int_0^L y^2 dy
    exact = a * side ** 2 + (b + c) * side * s1
    assert volume == pytest.approx(exact, rel=1e-13)
    w2 = (a * a * side ** 2 + 2 * a * (b + c) * side * s1 + (b * b + c * c) * side * s2
          + 2 * b * c * s1 * s1)
    assert centroid[0] == pytest.approx(0.5 * w2 / exact, rel=1e-13)


def test_clipped_volume_matches_divided_difference_formula():
    # For a linear phi on a tetrahedron with distinct vertex values, the volume
    # fraction of {phi < 0} is sum_{phi_i < 0} (-phi_i)^3 / prod_{j != i} (phi_j - phi_i).
    rng = np.random.default_rng(7)
    points = rng.random((4, 3))
    tet = np.array([[0, 1, 2, 3]])
    full = ver._tet_volume(*points[None, :, :].transpose(1, 0, 2))[0]
    for values in ([-0.3, 0.2, 0.5, 0.9], [-0.7, -0.2, 0.4, 1.1], [-1.0, -0.6, -0.1, 0.8]):
        for perm in itertools.permutations(range(4)):
            phi = np.asarray(values)[list(perm)]
            fraction = sum((-phi[i]) ** 3 / np.prod([phi[j] - phi[i] for j in range(4) if j != i])
                           for i in range(4) if phi[i] < 0.0)
            volume, _ = ver.liquid_volume_centroid(points, tet, phi)
            assert volume == pytest.approx(fraction * full, rel=1e-12)


def test_liquid_volume_of_sampled_sphere_converges_at_second_order():
    errors = []
    for level in (4, 8, 16):
        points, cells, _, _ = gen.structured_tetra_mesh(level)
        volume, centroid = ver.liquid_volume_centroid(points, cells, sampled_sphere(points))
        errors.append(abs(volume - 4.0 * math.pi / 3.0) / (4.0 * math.pi / 3.0))
        assert centroid == pytest.approx(gen.sphere_centre(), abs=2e-3 / level)
    assert ver.observed_order([4, 8, 16], errors) > 1.8


def test_interior_pressure_mean_and_interface_points():
    points, cells, _, _ = gen.structured_tetra_mesh(4)
    centre = np.full(3, 0.5 * gen.BOX_SIDE)                    # a vertex on the mirror planes
    mean, n = ver.interior_mean_pressure(points, cells, np.full(len(points), 2.0), centre, 0.8)
    assert mean == pytest.approx(2.0) and n > 0
    # The region is mirror symmetric about the centre, so a linear field averages to its centre value.
    mean, _ = ver.interior_mean_pressure(points, cells, points[:, 0] + 2.0 * points[:, 2], centre, 0.8)
    assert mean == pytest.approx(4.5, abs=1e-12)
    plane = points[:, 0] - 1.3 - 1e-3 * points[:, 1]
    iface = ver.interface_points(points, cells, plane)
    assert len(iface) > 0
    assert np.abs(iface[:, 0] - 1.3 - 1e-3 * iface[:, 1]).max() < 1e-12


def test_observed_order_recovers_power_law():
    assert ver.observed_order([8, 16, 32], [0.04, 0.01, 0.0025]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------
def test_generated_case_is_complete_and_respects_time_step_rule(tmp_path):
    case = gen.generate(8, "kag_lumped", 12.0, tmp_path / "c", transport="coupled")
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    text = (tmp_path / "c/solver.xml").read_text()
    assert root.find("GeneralSimulationParameters/Number_of_spatial_dimensions").text == "3"
    assert "<Geometry_tangent_policy>RefreshedFrozenQuadrature" in text
    assert "<Interface_quadrature_order>2" in text
    assert "<Surface_tension_form>KinematicAreaGradientTraction" in text
    assert "<Curvature_projection_kinematic_area_gradient_mass>Lumped" in text
    assert "<Velocity_source>coupled_field" in text
    assert root.find("GeneralSimulationParameters/Number_of_time_steps").text == str(case["steps"])
    dt_b = math.sqrt(case["h"] ** 3 / (4.0 * math.pi))
    assert case["dt_B"] == pytest.approx(dt_b)
    assert case["dt"] <= 2.0 * dt_b * (1 + 1e-12) and case["dt"] > 1.9 * dt_b
    assert case["steps"] == 500 and case["output_cadence"] == 5
    assert case["steps"] * case["dt"] == pytest.approx(5.0 * case["viscous_time"])
    assert case["viscosity"] == pytest.approx(math.sqrt(2.0 / 12.0))
    assert case["laplace_pressure_nominal"] == 2.0
    assert case["n_vertices"] == 25 ** 3 and case["n_tetrahedra"] == 6 * 24 ** 3
    assert case["min_abs_phi_over_h"] > 1e-3
    assert case["wall_gap_over_h"] > 3.0
    for wall in gen.WALLS:
        assert (tmp_path / f"c/mesh/mesh-surfaces/{wall}.vtp").is_file()
    assert sum(1 for b in root.iter("Add_BC") if b.findtext("Type") == "Dir") == 6
    schedule = gen.time_schedule(8, 12.0, 5.0, 100)
    consistent = gen.solver_xml("kag_consistent", schedule, 10, 1, "coupled")
    assert "kinematic_area_gradient_mass" not in consistent
    stress = gen.solver_xml("surface_stress", schedule, 10, 1, "wet_extension")
    assert "SurfaceStress" in stress and "Curvature_field" not in stress
    assert "<Advection_velocity_extension_method>wall_compatible_normal" in stress
    assert gen.time_schedule(8, 120.0, 5.0, 100)["dt"] <= dt_b * (1 + 1e-12)


def test_transport_default_is_the_pde_extension():
    # D9: harmonic PDE extension with monolithic coupling, never combined with
    # the wet-extension switch (the solver rejects that combination).
    assert gen.DEFAULT_TRANSPORT == "pde_extension"
    schedule = gen.time_schedule(8, 12.0, 5.0, 100)
    pde = gen.solver_xml("surface_stress", schedule, 10, 1, "pde_extension")
    assert "<Advection_velocity_extension_method>pde_harmonic<" in pde
    assert "<Advection_velocity_extension_coupling>monolithic<" in pde
    assert "Use_wet_extension" not in pde


def test_generated_mesh_reads_back(tmp_path):
    pv = pytest.importorskip("pyvista")
    gen.generate(8, "surface_stress", 12.0, tmp_path / "c", transport="coupled", max_steps=4)
    case = json.loads((tmp_path / "c/case.json").read_text())
    snap = ver.read_snapshot(tmp_path / "c/mesh/mesh-complete.mesh.vtu", case)
    assert snap["tets"].shape == (case["n_tetrahedra"], 4)
    assert np.all(snap["pressure"] == 2.0)
    face = pv.read(tmp_path / "c/mesh/mesh-surfaces/wall_top.vtp")
    assert face.n_cells == 2 * 24 * 24
    assert np.all(np.asarray(face.points)[:, 1] == gen.BOX_SIDE)
    assert case["truncated"] and case["steps"] == 4


# ---------------------------------------------------------------------------
# Synthetic solver output
# ---------------------------------------------------------------------------
MESH_LEVEL = 4        # coarse background mesh for the synthetic runs of every level


def write_synthetic_run(run, level, speed_scale, growth=False, pressure_error=None,
                        drop_last=False, dt_multiple=None, log_volumes=None):
    """Emulate solver output: static phi, decaying velocity, near-Laplace pressure.

    The case files come from generate_case.py; the mesh is replaced by a coarse
    one so the test stays small, and case.json keeps the nominal level.
    """
    case = gen.generate(8, "surface_stress", 12.0, run, transport="coupled", snapshots=8,
                        dt_multiple_override=dt_multiple)
    case["level_R_over_h"] = level
    (run / "case.json").write_text(json.dumps(case))
    points, cells, _, _ = gen.structured_tetra_mesh(MESH_LEVEL)
    phi = sampled_sphere(points)
    zeros = np.zeros((len(phi), 3))
    gen.write_vtu(run / "mesh/mesh-complete.mesh.vtu", points, cells,
                  {"phi": ("Float64", phi), "Velocity": ("Float64", zeros),
                   "Pressure": ("Float64", np.full(len(phi), 2.0))},
                  {"GlobalElementID": ("Int64", np.arange(len(cells)))})
    volume, _ = ver.liquid_volume_centroid(points, cells, phi)
    r_eff = ver.sphere_radius(volume)
    err = 0.004 * 8 / level if pressure_error is None else pressure_error
    pressure = np.full(len(phi), 2.0 * (1.0 + err) / r_eff)
    rel = points - np.asarray(case["centre"])
    swirl = np.column_stack([-rel[:, 1], rel[:, 0], 0.5 * rel[:, 2]])
    entries = []
    times = [k * case["output_cadence"] * case["dt"] for k in range(1, 9)]
    for k, t in enumerate(times, start=1):
        if drop_last and k == len(times):
            break
        amp = speed_scale * (math.exp(t / case["viscous_time"]) if growth
                             else math.exp(-t / case["viscous_time"]))
        name = f"result_{k * case['output_cadence']:03d}.vtu"
        gen.write_vtu(run / name, points, cells,
                      {"phi": ("Float64", phi), "Velocity": ("Float64", amp * swirl),
                       "Pressure": ("Float64", pressure)},
                      {"GlobalElementID": ("Int64", np.arange(len(cells)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    if log_volumes is not None:
        lines = []
        for step in range(1, case["steps"] + 1):
            v = log_volumes(step, volume)
            lines.append(f"[svMultiPhysics::Application] Wet volume diagnostic step={step} "
                         f"time={step * case['dt']:.17e} field='phi' isovalue=0 "
                         f"wet_volume={v:.17e} wet_volume_frame=physical "
                         f"reference_wet_volume=1.0e+03")
        (run / "solver_run.log").write_text("\n".join(lines) + "\n")
    return case, volume


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
    assert group["passed"] and group["transport"] == "coupled"
    run8 = group["runs"][0]
    assert run8["pressure_jump_relative_error"] == pytest.approx(0.004, rel=1e-9)
    assert run8["liquid_volume_relative_deviation_max"] == 0.0
    assert run8["effective_radius_relative_to_nominal"] < 0.0      # the P1 sphere is inscribed
    assert run8["log_volume_steps"] == 0 and not run8["log_volume_used"]


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
    gen.generate(8, "surface_stress", 12.0, empty, transport="coupled")
    assert ver.main([str(empty)]) == 2
    assert "no solver output" in capsys.readouterr().err


def test_volume_oscillation_between_outputs_is_caught_from_the_log(tmp_path):
    pytest.importorskip("pyvista")
    # D11: a reversible deviation between two outputs fails the volume criterion
    # when the per-step log agrees with the snapshots at the outputs.
    case, _ = write_synthetic_run(tmp_path / "osc", 8, 1e-5,
                                  log_volumes=lambda s, v: v * (1.0 + (3e-4 if s == 7 else 0.0)))
    assert case["output_cadence"] > 1 and 7 % case["output_cadence"] != 0
    run = ver.analyse_run(tmp_path / "osc")
    assert run["log_volume_used"] and run["log_volume_steps"] == case["steps"]
    assert run["liquid_volume_relative_deviation_max_outputs"] == 0.0
    assert run["liquid_volume_relative_deviation_max"] == pytest.approx(3e-4, rel=1e-6)
    # A log that disagrees with the snapshots at the outputs is reported but not used.
    write_synthetic_run(tmp_path / "off", 8, 1e-5, log_volumes=lambda s, v: 1.01 * v)
    run = ver.analyse_run(tmp_path / "off")
    assert not run["log_volume_used"]
    assert run["log_volume_agreement"] == pytest.approx(0.01, rel=1e-6)
    assert run["liquid_volume_relative_deviation_max"] == 0.0


def test_diagnostic_time_step_runs_are_reported_not_gated(study, tmp_path, capsys):
    runs = study()
    diag = tmp_path / "diag"
    write_synthetic_run(diag, 8, 1e-5, dt_multiple=4.0)
    assert ver.main([*runs, str(diag)]) == 0
    out = capsys.readouterr().out
    assert "diagnostic runs" in out and "dt = 4 dt_B" in out
    assert ver.main([str(diag)]) == 2
