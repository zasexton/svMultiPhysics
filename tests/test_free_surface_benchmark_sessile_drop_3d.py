"""Checks of the sessile_drop_3d generator and verifier on synthetic data."""

import importlib.util
import json
import math
import sys
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
# The Python mirror of the free-surface method profiles (decision D47, P4).
_profiles_spec = importlib.util.spec_from_file_location(
    "free_surface_profile_decks", BENCH.parents[1] / "free_surface_profile_decks.py")
profiles = importlib.util.module_from_spec(_profiles_spec)
sys.modules[_profiles_spec.name] = profiles
_profiles_spec.loader.exec_module(profiles)


def deck_keys(text):
    """The free-surface and level-set keys a deck resolves to (its profile's
    values for the keys it does not state)."""
    return profiles.effective_deck_keys(text)


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
    init, eq = case["initial_cap"], case["equilibrium_cap_nominal"]
    phi0 = np.linalg.norm(points - np.array([0.0, init["centre_y"], 0.0]), axis=1) - init["radius"]
    phi1 = np.linalg.norm(points - np.array([0.0, eq["centre_y"], 0.0]), axis=1) - eq["radius"]
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
    # Decision D43: the gated angle is the mean over the partition rotations.
    assert r["contact_angle_max_error_degrees_single_partition"] < 0.7
    assert all(s["angle_rotation_min"] <= s["angle"] <= s["angle_rotation_max"] for s in f["sectors"])
    assert [s["nominal_centre_azimuth_degrees"] for s in f["sectors"]] == [15.0, 45.0, 75.0]
    assert r["base_radius_relative_error"] < 5e-3 and r["apex_height_relative_error"] < 5e-3
    assert f["axis"] == [0.0, 0.0]
    assert r["max_speed_growth_ratio"] < 1.0
    # The sampled caps differ in P1 volume, which the volume metric sees.
    assert r["liquid_volume_relative_drift_max"] > 1e-4
    out = tmp_path / "report.json"
    assert ver.main([str(run), "--allow-truncated", "--json", str(out)]) == 1
    assert json.loads(out.read_text())["passed"] is False


@pytest.mark.parametrize("angle, initial, box, tets", [
    (60, 90.0, [0.0, 1.125, 0.0, 1.0, 0.0, 1.125], 3888),
    (120, 90.0, [0.0, 1.5, 0.0, 1.75, 0.0, 1.5], 12096),
])
def test_generated_decks_at_the_added_angles(tmp_path, angle, initial, box, tets):
    tol = json.loads((BENCH / "tolerances.json").read_text())["protocol"]
    assert tol["equilibrium_angles_degrees"] == list(gen.EQUILIBRIUM_ANGLES_DEG)
    assert {int(k): v for k, v in tol["initial_angles_degrees"].items()} == gen.INITIAL_ANGLE_DEG
    case = gen.generate(8, angle, tmp_path / f"a{angle}")
    assert case["equilibrium_angle_degrees"] == angle and case["initial_angle_degrees"] == initial
    assert case["box"] == pytest.approx(box) and case["n_tetrahedra"] == tets
    eq, init = case["equilibrium_cap_nominal"], case["initial_cap"]
    assert eq["radius"] == pytest.approx(1.0)
    assert init["volume"] == pytest.approx(eq["volume"])
    assert eq["base_radius"] == pytest.approx(math.sin(math.radians(angle)))
    assert case["min_abs_phi_over_h"] > gen.MIN_PHI_OVER_H_WARNING
    assert case["initial_contact_vertex_gap_over_h"] > gen.MIN_PHI_OVER_H_WARNING
    assert case["dry_gap_over_h"] >= 2.0 - 1e-9
    root = ET.parse(tmp_path / f"a{angle}" / "solver.xml").getroot()
    fs = next(b for b in root.iter("Add_BC") if b.get("name") == "free_surface")
    assert float(fs.findtext("Contact_angle_degrees")) == angle
    assert fs.findtext("Contact_line_model") == "PrescribedAngle"
    assert case["steps"] == 1400 and case["interface_preserving_redistancing"] is True


@pytest.mark.parametrize("angle", [60, 120])
def test_metrics_on_sampled_caps_at_the_added_angles(tmp_path, angle):
    pytest.importorskip("pyvista")
    run = tmp_path / f"a{angle}"
    case = gen.generate(8, angle, run, max_steps=4)
    _write_relaxing_run(run, case, 8)
    r = ver.analyse_run(run, allow_truncated=True)
    f = r["final"]
    # Sampled equilibrium caps at R/h = 8: 0.25 (60) and 0.39 (120) degrees (README.md).
    assert r["contact_angle_max_error_degrees"] < 0.5
    assert abs(f["contact_angle_mean"] - angle) < 0.5
    assert len(f["sectors"]) == 3 and all(s["fit_points"] >= 20 for s in f["sectors"])
    assert r["base_radius_relative_error"] < 8e-3 and r["apex_height_relative_error"] < 8e-3
    assert r["max_speed_growth_ratio"] < 1.0


def test_verify_takes_one_angle_at_a_time(tmp_path, capsys):
    pytest.importorskip("pyvista")
    runs = []
    for angle in (60, 90):
        run = tmp_path / f"a{angle}"
        case = gen.generate(8, angle, run, max_steps=2)
        _write_relaxing_run(run, case, 2)
        runs.append(str(run))
    assert ver.main(runs + ["--allow-truncated"]) == 2
    assert "one angle at a time" in capsys.readouterr().err


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
    # The Wetting profile states the redistancing; off states it off.
    key = "<Enable_interface_preserving_redistancing>false</Enable_interface_preserving_redistancing>"
    text_on = (tmp_path / "on" / "solver.xml").read_text()
    text_off = (tmp_path / "off" / "solver.xml").read_text()
    assert deck_keys(text_on)[1]["Enable_interface_preserving_redistancing"] == "true"
    assert key in text_off and text_off.replace("\n    " + key, "") == text_on
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


def test_sector_rotations_follow_the_tolerances():
    tol = json.loads((BENCH / "tolerances.json").read_text())
    rot = next(c for c in tol["criteria"] if c["id"] == "contact_angle")["sector_rotations"]
    offsets = np.asarray(ver.SECTOR_ROTATION_OFFSETS)
    assert len(offsets) == rot["count"] == 21
    assert offsets[0] == rot["offset_min_degrees"] and offsets[-1] == rot["offset_max_degrees"]
    assert np.allclose(np.diff(offsets), 0.5) and 0.0 in ver.SECTOR_ROTATION_OFFSETS
    assert tol["time_step_criterion"]["refined_step_levels"] == [16]


def test_rotated_sector_partitions():
    rng = np.random.default_rng(3)
    quarter = np.concatenate([rng.uniform(0.0, 90.0, 2000), [0.0, 30.0, 60.0, 90.0]])
    full = np.concatenate([rng.uniform(-180.0, 180.0, 4000), [-180.0, -150.0, 0.0, 180.0]])
    for domain, az in (("quarter", quarter), ("full", full)):
        nominal = ver.nominal_sectors(domain)
        assert len(nominal) == (3 if domain == "quarter" else 12)
        for offset in ver.SECTOR_ROTATION_OFFSETS:
            masks = np.array(ver.sector_masks(az, domain, offset))
            assert np.all(masks.sum(axis=0) == 1)          # a partition of the points
        # Offset 0 is the unrotated D38 partition.
        for (lo, hi), m in zip(nominal, ver.sector_masks(az, domain, 0.0)):
            last = hi == nominal[-1][1]
            assert np.array_equal(m, (az >= lo) & ((az < hi) | last))
    # Quarter: the outer edges stay on the symmetry planes, the interior ones move.
    m = ver.sector_masks(np.array([0.0, 34.0, 36.0, 64.0, 66.0, 90.0]), "quarter", 5.0)
    assert [list(np.nonzero(x)[0]) for x in m] == [[0, 1], [2, 3], [4, 5]]
    m = ver.sector_masks(np.array([0.0, 26.0, 34.0, 56.0, 90.0]), "quarter", -5.0)
    assert [list(np.nonzero(x)[0]) for x in m] == [[0], [1, 2], [3, 4]]
    # Full: periodic; with offset -5 the sector of nominal centre -165 is [175, 180] U [-180, -155).
    m = ver.sector_masks(np.array([178.0, -170.0, 172.0]), "full", -5.0)
    assert list(np.nonzero(m[0])[0]) == [0, 1] and list(np.nonzero(m[11])[0]) == [2]


def _final(angles, single=None, base=1.0, apex=1.0):
    single = angles if single is None else single
    sectors = [{"nominal_centre_azimuth_degrees": c, "angle": a, "angle_single_partition": s}
               for c, a, s in zip((15.0, 45.0, 75.0), angles, single)]
    return {"final": {"sectors": sectors, "base_radius": base, "apex_height": apex}}


def test_time_step_is_gated_at_r_over_h_16_only():
    tol = json.loads((BENCH / "tolerances.json").read_text())

    def run(level, divisor, angles, single=None):
        return dict(_final(angles, single), level=level, domain="quarter", dt_rule="fixed",
                    dt_divisor=divisor)

    key1, key2 = ("quarter", "fixed", 1), ("quarter", "fixed", 2)
    coarse8 = run(8, 1, [89.04, 90.07, 89.04], [89.01, 90.49, 88.98])
    fine8 = run(8, 2, [89.10, 89.93, 89.10], [89.15, 89.85, 89.12])
    # Only the R/h = 8 pair: reported, not gated; R/h = 16 not evaluated; nothing fails.
    steps = ver.evaluate_time_step({key1: [coarse8, run(16, 1, [90.0, 90.1, 89.9])],
                                    key2: [fine8]}, tol)
    by_level = {s["level"]: s for s in steps}
    assert by_level[8]["gated"] is False and by_level[8]["passed"] is None
    assert by_level[8]["changes"]["contact_angle_change_final"] == pytest.approx(0.14)
    assert by_level[8]["changes"]["contact_angle_change_final_single_partition"] == pytest.approx(0.64)
    assert by_level[16]["evaluated"] is False and by_level[16]["passed"] is True
    # The R/h = 16 pair is gated on the averaged angles; sectors matched by centre.
    good = ver.evaluate_time_step({key1: [run(16, 1, [90.0, 90.10, 89.9], [90.0, 90.5, 89.9])],
                                   key2: [run(16, 2, [90.0, 90.05, 89.9], [90.0, 90.1, 89.9])]}, tol)
    assert good[0]["gated"] and good[0]["passed"] is True
    bad = ver.evaluate_time_step({key1: [run(16, 1, [90.0, 90.25, 89.9])],
                                  key2: [run(16, 2, [90.0, 90.05, 89.9])]}, tol)
    assert bad[0]["passed"] is False


def test_dt2_study_without_r_over_h_16_is_not_evaluated():
    tol = json.loads((BENCH / "tolerances.json").read_text())
    run8 = {"level": 8, "contact_angle_max_error_degrees": 0.9, "base_radius_relative_error": 0.01,
            "apex_height_relative_error": 0.01, "liquid_volume_relative_drift_max": 5e-5,
            "max_speed_growth_ratio": 0.9, "capillary_number_at_end": 1e-4}
    refined = set(tol["time_step_criterion"]["refined_step_levels"])
    verdicts = {v["id"]: v for v in ver.evaluate_group([run8], tol, levels_required=refined)}
    for cid in ("contact_angle", "base_radius", "apex_height"):
        assert verdicts[cid]["evaluated"] is False and verdicts[cid]["passed"]
    assert verdicts["volume_drift"]["evaluated"] and verdicts["volume_drift"]["passed"]
