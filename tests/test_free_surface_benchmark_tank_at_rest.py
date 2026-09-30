"""Checks of the tank_at_rest benchmark scripts on synthetic data (no solver run)."""

import importlib.util
import itertools
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = (Path(__file__).resolve().parent
             / "cases/fluid/free_surface_benchmarks/tank_at_rest")


def load(name):
    spec = importlib.util.spec_from_file_location(f"tank_at_rest_{name}", BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load("generate_case")
ver = load("verify")


def simplex_fraction_divided_difference(f):
    """Volume fraction of {f < 0} in a simplex with distinct vertex values."""
    d = len(f) - 1
    total = 0.0
    for i, fi in enumerate(f):
        if fi < 0.0:
            total += (-fi) ** d / np.prod([f[j] - fi for j in range(len(f)) if j != i])
    return total


@pytest.mark.parametrize("dim", [2, 3])
def test_mesh_is_conforming_positive_and_fills_the_box(dim):
    points, cells, faces, counts = gen.structured_mesh(dim, 8)
    measure = gen._signed_measure(points[:, :dim], cells)
    assert np.all(measure > 0.0)
    box = gen.TANK_LENGTH * gen.TANK_HEIGHT * (gen.TANK_WIDTH if dim == 3 else 1.0)
    assert measure.sum() == pytest.approx(box, rel=1e-13)
    # Every interior facet is shared by exactly two cells (conforming mesh).
    facets = {}
    for cell in cells:
        for loc in itertools.combinations(cell, dim):
            key = tuple(sorted(loc))
            facets[key] = facets.get(key, 0) + 1
    assert set(facets.values()) <= {1, 2}
    n_boundary = sum(1 for v in facets.values() if v == 1)
    assert n_boundary == sum(len(f[0]) for f in faces.values())
    assert set(faces) == set(gen.walls(dim))


def test_clipped_measures_are_exact_for_planes():
    points2, tris, _, _ = gen.structured_mesh(2, 8)
    points3, tets, _, _ = gen.structured_mesh(3, 8)
    # Flat interface (vertex values equal along cell faces) and a tilted plane.
    assert ver.liquid_measure(points2, tris, points2[:, 1] - 0.52) == pytest.approx(0.52, rel=1e-14)
    assert ver.liquid_measure(points3, tets, points3[:, 1] - 0.52) == pytest.approx(0.26, rel=1e-14)
    tilted = points3[:, 1] - 0.4 - 0.1 * points3[:, 0] - 0.05 * points3[:, 2]
    assert ver.liquid_measure(points3, tets, tilted) == pytest.approx(
        0.5 * (0.4 + 0.05 + 0.0125), rel=1e-13)


def test_tetrahedron_clip_matches_divided_differences():
    rng = np.random.default_rng(7)
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for _ in range(200):
        f = rng.normal(size=4)
        expected = simplex_fraction_divided_difference(f) / 6.0
        assert ver._clip_tetrahedron(p, f) == pytest.approx(expected, rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("dim,level", [(2, 8), (2, 32), (3, 16)])
def test_generated_case_samples_the_exact_state(tmp_path, dim, level):
    case = gen.generate(dim, level, tmp_path / "c")
    text = (tmp_path / "c/solver.xml").read_text()
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    assert root.find("GeneralSimulationParameters/Number_of_spatial_dimensions").text == str(dim)
    assert "<Surface_tension>0.0</Surface_tension>" in text
    assert "<Effective_direction>" + " ".join(["1"] + ["0"] * (dim - 1)) in text
    assert '<Add_BC name="wall_top">' not in text          # dry top wall, no condition
    assert case["min_abs_phi_over_h"] > 0.15                   # not mesh-aligned
    assert case["steps"] * case["dt"] == pytest.approx(5 * case["sloshing_period"])
    grid_points, cells, _, _ = gen.structured_mesh(dim, level)
    snap = {"points": grid_points, "cells": cells, "phi": grid_points[:, 1] - 0.52,
            "velocity": np.zeros((len(grid_points), dim)),
            "pressure": gen.hydrostatic_pressure(grid_points[:, 1])}
    m = ver.snapshot_metrics(snap, case)
    assert m["max_speed"] == 0.0 and m["pressure_error"] == 0.0
    assert m["interface_height_error"] < 1e-15
    assert m["liquid_volume"] == pytest.approx(case["liquid_volume_exact"], rel=1e-14)
    for wall in gen.walls(dim):
        assert (tmp_path / f"c/mesh/mesh-surfaces/{wall}.vtp").is_file()


def write_synthetic_run(run, dim, level, *, speed=0.0, pressure_offset=0.0, height_shift=0.0,
                        drop_last=False, max_steps=None):
    """Emulate solver output: the hydrostatic state plus optional perturbations."""
    case = gen.generate(dim, level, run, max_steps=max_steps)
    points, cells, _, _ = gen.structured_mesh(dim, level)
    n = len(points)
    entries = []
    outputs = case["steps"] // case["output_cadence"]
    for k in range(1, outputs + 1):
        if drop_last and k == outputs:
            break
        step = k * case["output_cadence"]
        t = step * case["dt"]
        velocity = np.zeros((n, 3))
        velocity[:, 0] = speed * k / outputs
        phi = points[:, 1] - gen.FILL_HEIGHT - height_shift * k / outputs
        pressure = gen.hydrostatic_pressure(points[:, 1]) + pressure_offset
        name = f"result_{step:03d}.vtu"
        gen.write_vtu(run / name, points, cells,
                      {"phi": ("Float64", phi), "Velocity": ("Float64", velocity),
                       "Pressure": ("Float64", pressure)},
                      {"GlobalElementID": ("Int64", np.arange(len(cells)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    return case


@pytest.fixture
def pyvista():
    return pytest.importorskip("pyvista")


def test_exact_state_passes_in_2d_and_3d(pyvista, tmp_path, capsys):
    runs = []
    for dim, level in ((2, 8), (2, 16), (3, 8)):
        run = tmp_path / f"d{dim}_L{level}"
        write_synthetic_run(run, dim, level, speed=1e-12)
        runs.append(str(run))
    out = tmp_path / "report.json"
    assert ver.main([*runs, "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["passed"] and len(report["runs"]) == 3
    r = report["runs"][0]
    assert r["max_speed_over_velocity_scale"] == pytest.approx(1e-12 / math.sqrt(0.52))
    assert r["liquid_volume_relative_drift_max"] == 0.0
    assert r["outputs"] == 20


@pytest.mark.parametrize("perturbation,criterion", [
    ({"speed": 1e-6}, "velocity"),
    ({"pressure_offset": 1e-6}, "hydrostatic_pressure"),
    ({"height_shift": 1e-6}, "interface_height"),
])
def test_perturbations_above_the_tolerance_scale_fail(pyvista, tmp_path, capsys, perturbation,
                                                      criterion):
    run = tmp_path / "run"
    write_synthetic_run(run, 2, 8, **perturbation)
    assert ver.main([str(run)]) == 1
    out = capsys.readouterr().out
    assert f"[FAIL] {criterion}" in out
    if criterion == "interface_height":
        assert "[FAIL] volume_drift" in out


def test_missing_truncated_and_unknown_data_fail_clearly(pyvista, tmp_path, capsys):
    incomplete = tmp_path / "incomplete"
    write_synthetic_run(incomplete, 2, 8, drop_last=True)
    assert ver.main([str(incomplete)]) == 2
    assert "incomplete run" in capsys.readouterr().err
    empty = tmp_path / "empty"
    gen.generate(2, 8, empty)
    assert ver.main([str(empty)]) == 2
    assert "no solver output" in capsys.readouterr().err
    smoke = tmp_path / "smoke"
    write_synthetic_run(smoke, 3, 8, max_steps=3)
    assert ver.main([str(smoke)]) == 2
    assert "truncated" in capsys.readouterr().err
    assert ver.main([str(smoke), "--allow-truncated"]) == 0
    with pytest.raises(ValueError):
        gen.generate(3, 32, tmp_path / "bad")
