"""Checks of the sessile_drop_3d generator and verifier on synthetic data."""

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent / "cases" / "fluid" / "free_surface_benchmarks" / "sessile_drop_3d"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"sessile_drop_3d_{name}_under_test", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load("generate_case")
ver = _load("verify")


def test_cap_geometry_and_schedule():
    assert gen.cap_volume_factor(math.pi / 2) == pytest.approx(2.0 / 3.0)
    eq = gen.cap_geometry(1.0, math.pi / 2)
    assert eq["base_radius"] == pytest.approx(1.0) and eq["apex_height"] == pytest.approx(1.0)
    r0 = gen.cap_radius_for_volume(eq["volume"], math.radians(120.0))
    assert gen.cap_geometry(r0, math.radians(120.0))["volume"] == pytest.approx(eq["volume"])
    for level in gen.LEVELS:
        s = gen.time_schedule(level, gen.DEFAULT_VISCOUS_TIMES, gen.DEFAULT_SNAPSHOTS)
        assert s["dt"] <= gen.PROTOCOL_DT and s["steps"] == 1400 and s["output_cadence"] == 7
        assert s["steps"] * s["dt"] == pytest.approx(10.0 * s["viscous_time"])
    half = gen.time_schedule(8, gen.DEFAULT_VISCOUS_TIMES, gen.DEFAULT_SNAPSHOTS, dt_divisor=2)
    assert half["dt"] == pytest.approx(0.5 * s["dt"]) and half["output_cadence"] == 14
    cap = gen.time_schedule(16, gen.DEFAULT_VISCOUS_TIMES, gen.DEFAULT_SNAPSHOTS,
                            dt_rule="capillary-limit")
    assert cap["dt"] <= cap["dt_B"]


def _cell_keys(points, cells):
    return {tuple(sorted(tuple(np.round(points[v], 9)) for v in c)) for c in cells}


def test_quarter_mesh_is_the_full_mesh_on_the_shared_quarter():
    geo_q = gen.case_geometry(90, 120.0, "quarter")
    geo_f = gen.case_geometry(90, 120.0, "full")
    pq, cq, fq, nq = gen.structured_tetra_mesh(geo_q["box"], 8, gen.QUARTER_PLANES)
    pf, cf, ff, nf = gen.structured_tetra_mesh(geo_f["box"], 8, gen.FULL_PLANES)
    assert nf == (2 * nq[0], nq[1], 2 * nq[2])
    corner = np.all((pf[cf][:, :, 0] >= -1e-12) & (pf[cf][:, :, 2] >= -1e-12), axis=1)
    assert _cell_keys(pq, cq) == _cell_keys(pf, cf[corner])
    # Mirror symmetry of the full mesh about x = 0 and z = 0.
    keys = _cell_keys(pf, cf)
    for axis in (0, 2):
        mirrored = pf.copy()
        mirrored[:, axis] *= -1.0
        assert _cell_keys(mirrored, cf) == keys
    # Face sets are disjoint and cover the boundary.
    for faces in (fq, ff):
        tris = [tuple(sorted(t)) for name in faces for t in faces[name][0]]
        assert len(tris) == len(set(tris))


def test_generated_quarter_deck(tmp_path):
    case = gen.generate(8, 90, tmp_path / "q8")
    assert case["domain"] == "quarter" and case["symmetry_fraction"] == 0.25
    assert case["surface_tension_semi_implicit"] == "NormalIncrement"
    assert case["min_abs_phi_over_h"] > gen.MIN_PHI_OVER_H_WARNING
    root = ET.parse(tmp_path / "q8" / "solver.xml").getroot()
    fluid = next(e for e in root.iter("Add_equation") if e.get("type") == "fluid")
    bcs = {b.get("name"): b for b in fluid.iter("Add_BC")}
    assert bcs["wall_bottom"].findtext("Effective_direction").strip() == "0 1 0"
    assert bcs["sym_x"].findtext("Effective_direction").strip() == "1 0 0"
    assert bcs["sym_z"].findtext("Effective_direction").strip() == "0 0 1"
    for dry in ("wall_right", "wall_top", "wall_back"):
        assert bcs[dry].find("Effective_direction") is None
    fs = bcs["free_surface"]
    assert fs.findtext("Contact_line_wall_face") == "wall_bottom"
    assert fs.findtext("Contact_line_model") == "PrescribedAngle"
    assert fs.findtext("Surface_tension_semi_implicit") == "NormalIncrement"
    mesh = root.find("Add_mesh")
    assert mesh.find("Ghost_layers") is None
    assert {f.get("name") for f in mesh.iter("Add_face")} == {p[0] for p in gen.QUARTER_PLANES}
    full = gen.generate(8, 90, tmp_path / "f8", domain="full")
    assert full["n_tetrahedra"] == 4 * case["n_tetrahedra"]


def _write_relaxing_run(run, case, frames):
    """Snapshots blending the sampled initial cap into the equilibrium cap."""
    points, cells, _, _ = gen.structured_tetra_mesh(case["box"], case["level_R_over_h"],
                                                    gen.QUARTER_PLANES if case["domain"] == "quarter"
                                                    else gen.FULL_PLANES)
    init = case["initial_cap"]
    phi0 = np.linalg.norm(points - np.array([0.0, init["centre_y"], 0.0]), axis=1) - init["radius"]
    phi1 = np.linalg.norm(points, axis=1) - 1.0
    entries = []
    for i in range(1, frames + 1):
        t = case["end_time"] * i / frames
        w = min(1.0, 2.0 * i / frames)
        phi = (1.0 - w) * phi0 + w * phi1
        speed = 0.1 * math.exp(-5.0 * i / frames)
        vel = np.zeros((len(points), 3))
        vel[:, 1] = speed * points[:, 0]
        name = f"result_{i:03d}.vtu"
        gen.SPHERE.write_vtu(run / name, points, cells,
                             {"GlobalNodeID": ("Int64", np.arange(len(points))),
                              "phi": ("Float64", phi), "Velocity": ("Float64", vel),
                              "Pressure": ("Float64", np.zeros(len(points)))},
                             {"GlobalElementID": ("Int64", np.arange(len(cells)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection"><Collection>'
                                    + "".join(entries) + "</Collection></VTKFile>\n")


def test_metrics_on_sampled_caps(tmp_path):
    pytest.importorskip("pyvista")
    run = tmp_path / "q8"
    case = gen.generate(8, 90, run, max_steps=4)
    _write_relaxing_run(run, case, 8)
    r = ver.analyse_run(run, allow_truncated=True)
    f = r["final"]
    # Sampled exact caps: angle within 0.7 degrees at R/h = 8 (README.md).
    assert r["contact_angle_max_error_degrees"] < 0.7
    assert abs(f["contact_angle_mean"] - 90.0) < 0.7
    assert len(f["sectors"]) == 3 and all(s["fit_points"] >= 20 for s in f["sectors"])
    assert r["base_radius_relative_error"] < 5e-3 and r["apex_height_relative_error"] < 5e-3
    assert f["axis"] == [0.0, 0.0]
    assert r["max_speed_growth_ratio"] < 1.0
    # The sampled caps differ in P1 volume, which the volume metric sees.
    assert r["liquid_volume_relative_drift_max"] > 1e-4
    out = tmp_path / "report.json"
    assert ver.main([str(run), "--allow-truncated", "--json", str(out)]) == 1
    assert json.loads(out.read_text())["passed"] is False


def test_symmetry_check_of_identical_fields(tmp_path, capsys):
    pytest.importorskip("pyvista")
    quarter, full = tmp_path / "q8", tmp_path / "f8"
    cq = gen.generate(8, 90, quarter, max_steps=2)
    cf = gen.generate(8, 90, full, domain="full", max_steps=2)
    _write_relaxing_run(quarter, cq, 2)
    _write_relaxing_run(full, cf, 2)
    report = ver.symmetry_check(quarter, full)
    worst = report["worst"]
    assert worst["velocity_max_difference_over_max_speed"] == 0.0
    assert worst["phi_max_difference"] == 0.0
    assert worst["volume_relative_difference_shared_quarter"] < 1e-12
    assert worst["contact_angle_sector_max_difference"] < 1e-9
    assert worst["base_radius_difference"] < 1e-12
    assert worst["full_domain_axis_offset"] < 1e-9


def test_criteria_gate_r_over_h_16_and_the_angle_trend():
    tol = json.loads((BENCH / "tolerances.json").read_text())

    def run(level, angle, base=0.01, apex=0.01, drift=5e-5, growth=0.9):
        return {"level": level, "contact_angle_max_error_degrees": angle,
                "base_radius_relative_error": base, "apex_height_relative_error": apex,
                "liquid_volume_relative_drift_max": drift, "max_speed_growth_ratio": growth,
                "capillary_number_at_end": 1e-2}

    ok = ver.evaluate_group([run(8, 3.0), run(16, 1.5)], tol)
    assert all(v["passed"] for v in ok)
    bad_trend = ver.evaluate_group([run(8, 1.0), run(16, 1.5)], tol)
    assert not next(v for v in bad_trend if v["id"] == "contact_angle")["passed"]
    coarse_only = ver.evaluate_group([run(8, 1.0)], tol, levels_required={8})
    assert next(v for v in coarse_only if v["id"] == "contact_angle")["evaluated"] is False
    assert all(v["passed"] for v in coarse_only)
    assert tol["status"].startswith("APPROVED")


def test_protocol_decks_enable_the_triggered_redistancing(tmp_path):
    on = gen.generate(8, 90, tmp_path / "on")
    off = gen.generate(8, 90, tmp_path / "off", redistancing=False)
    key = "<Enable_interface_preserving_redistancing>true</Enable_interface_preserving_redistancing>"
    text_on = (tmp_path / "on" / "solver.xml").read_text()
    text_off = (tmp_path / "off" / "solver.xml").read_text()
    assert key in text_on and text_on.replace("\n    " + key, "") == text_off
    assert on["interface_preserving_redistancing"] is True
    assert "interface_preserving_redistancing" not in off


def test_velocity_growth_is_not_gated_for_a_drop_at_rest():
    tol = json.loads((BENCH / "tolerances.json").read_text())

    def run(level, ca, ratio, angle):
        return {"level": level, "contact_angle_max_error_degrees": angle,
                "base_radius_relative_error": 0.01, "apex_height_relative_error": 0.01,
                "liquid_volume_relative_drift_max": 5e-5, "max_speed_growth_ratio": ratio,
                "capillary_number_at_end": ca}

    verdicts = {v["id"]: v for v in ver.evaluate_group([run(8, 1e-4, 2.0, 3.0),
                                                         run(16, 3e-3, 0.5, 1.5)], tol)}
    growth = verdicts["no_velocity_growth"]
    assert growth["passed"] and growth["gated_by_level"] == {"8": False, "16": True}
    assert growth["rest_capillary_floor"] == 0.001
