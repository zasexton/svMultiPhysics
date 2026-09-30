"""Checks of the fitted_static_drop_2d smoke scripts on synthetic data (no solver run)."""

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

CASES = Path(__file__).resolve().parent / "cases/fluid/free_surface_benchmarks"
BENCHMARK = CASES / "fitted_static_drop_2d"


def load(directory, name):
    spec = importlib.util.spec_from_file_location(f"{directory.name}_{name}",
                                                  directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load(BENCHMARK, "generate_case")
ver = load(BENCHMARK, "verify")
writer = load(CASES / "linear_sloshing_2d", "generate_case")


@pytest.mark.parametrize("level", gen.LEVELS)
def test_disk_mesh_is_a_conforming_regular_polygon(level):
    points, cells, boundary, parents = gen.disk_mesh(level)
    sides = 6 * level
    assert len(points) == 1 + 3 * level * (level + 1)
    assert len(cells) == 6 * level * level
    assert boundary[0] == boundary[-1] and len(boundary) == sides + 1
    radius = np.linalg.norm(points[boundary, :2], axis=1)
    assert np.allclose(radius, gen.RADIUS)
    assert gen.polygon_area(points, cells) == pytest.approx(
        0.5 * sides * math.sin(2 * math.pi / sides), rel=1e-12)
    edges = Counter()
    for tri in cells:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[(min(u, v), max(u, v))] += 1
    assert max(edges.values()) == 2
    assert sum(1 for count in edges.values() if count == 1) == sides
    for (u, v), parent in zip(zip(boundary[:-1], boundary[1:]), parents):
        assert u in cells[parent] and v in cells[parent]


def test_generated_case_opts_in_to_fitted_surface_stress(tmp_path):
    case = gen.generate(8, tmp_path / "c")
    root = ET.parse(tmp_path / "c/solver.xml").getroot()
    equations = [eq.get("type") for eq in root.iter("Add_equation")]
    assert equations == ["fluid", "mesh_motion"]
    bc = next(root.iter("Add_BC"))
    assert bc.find("Implementation").text == "FittedALE"
    assert bc.find("Surface_tension_form").text == "SurfaceStress"
    assert bc.find("Allow_fitted_surface_stress").text == "true"
    assert bc.find("Kinematic_enforcement").text == "MeshNitsche"
    assert case["pressure_regular_polygon_balance"] == pytest.approx(1.0 / math.cos(math.pi / 48))
    assert case["steps"] == 100 and case["viscosity"] == pytest.approx(math.sqrt(1 / 6))


def test_verify_reports_the_balanced_polygon(tmp_path, capsys):
    pytest.importorskip("pyvista")
    run = tmp_path / "L8"
    case = gen.generate(8, run)
    points, cells, _, _ = gen.disk_mesh(8)
    n = len(points)
    entries = []
    for i in range(1, 11):
        t = case["end_time"] * i / 10
        name = f"result_{i:03d}.vtu"
        writer.write_vtu(run / name, points, cells,
                         {"GlobalNodeID": ("Int64", np.arange(n)),
                          "Velocity": ("Float64", np.full((n, 3), 1e-9)),
                          "Pressure": ("Float64", np.full(n, case["pressure_regular_polygon_balance"])),
                          "mesh_displacement": ("Float64", np.zeros((n, 3)))},
                         {"GlobalElementID": ("Int64", np.arange(len(cells)))})
        entries.append(f'<DataSet timestep="{t:.16f}" group="" part="0" file="{name}"/>')
    (run / "result.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection">'
                                    "<Collection>" + "".join(entries) + "</Collection></VTKFile>\n")
    out = tmp_path / "report.json"
    assert ver.main([str(run), "--json", str(out)]) == 0
    report = json.loads(out.read_text())["runs"][0]
    assert report["pressure_jump_over_polygon_balance"] == pytest.approx(1.0, abs=1e-12)
    assert report["pressure_jump_relative_error"] == pytest.approx(
        1.0 / math.cos(math.pi / 48) - 1.0, rel=1e-9)
    assert report["liquid_area_relative_deviation_max"] == 0.0
    assert report["spurious_capillary_number_max"] == pytest.approx(
        math.sqrt(2) * 1e-9 * case["viscosity"], rel=1e-6)
    assert "not evaluated (no run at R/h=32)" in capsys.readouterr().out
