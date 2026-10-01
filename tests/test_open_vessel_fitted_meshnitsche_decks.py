"""Checks of the MeshNitsche fitted SPHERIC Test 10 decks and their generator (no solver run)."""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

CASES = Path(__file__).resolve().parent / "cases/fluid/open_vessel_free_surface"
pv = pytest.importorskip("pyvista")


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_tests", CASES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = load("generate_spheric_test10_fitted_decks")
DECKS = {2: CASES / "fitted_ale" / gen.DECK_2D, 3: CASES / "fitted_ale" / gen.DECK_3D}
LEGACY_3D = CASES / "fitted_ale" / "spheric_test10_lateral_water_1x"


def read_faces(deck):
    faces = {}
    for path in sorted((deck / "mesh/water/mesh-surfaces").glob("*.vtp")):
        poly = pv.read(path)
        gid = np.asarray(poly.point_data["GlobalNodeID"])
        if poly.n_lines:
            conn = np.asarray(poly.lines).reshape(-1, 3)[:, 1:]
        else:
            conn = np.asarray(poly.faces).reshape(-1, 4)[:, 1:]
        faces[path.stem] = (gid[conn], np.asarray(poly.cell_data["GlobalElementID"]))
    return faces


@pytest.mark.parametrize("dim", [2, 3])
def test_committed_decks_are_reproduced_by_the_generator(dim, tmp_path):
    out = tmp_path / f"deck{dim}"
    gen.write_deck(dim, out)
    committed = DECKS[dim]
    for rel in ["solver.xml", "benchmark.json", "mesh/water/mesh-complete.mesh.vtu",
                *(f"mesh/water/mesh-surfaces/{p.name}"
                  for p in (committed / "mesh/water/mesh-surfaces").glob("*.vtp"))]:
        assert (out / rel).read_text() == (committed / rel).read_text(), rel


@pytest.mark.parametrize("dim", [2, 3])
def test_face_sets_are_disjoint_complete_and_on_their_planes(dim):
    deck = DECKS[dim]
    mesh = pv.read(deck / "mesh/water/mesh-complete.mesh.vtu")
    points = np.asarray(mesh.points)
    nv = dim + 1
    cells = np.asarray(mesh.cells).reshape(-1, nv + 1)[:, 1:]
    faces = read_faces(deck)
    planes = {**(gen.WALLS_2D if dim == 2 else gen.WALLS_3D), "free_surface": (1, gen.FILL_HEIGHT)}
    assert set(faces) == set(planes)
    seen = {}
    for name, (facets, parents) in faces.items():
        axis, value = planes[name]
        assert np.allclose(points[facets][..., axis], value, atol=1e-12), name
        for facet, parent in zip(facets, parents):
            key = tuple(sorted(facet.tolist()))
            assert key not in seen, f"{key} in {seen.get(key)} and {name}"
            seen[key] = name
            # GlobalElementID is the parent cell of the facet.
            assert set(key) <= set(cells[parent].tolist()), name
    boundary = {tuple(sorted(f)) for f, _ in gen.boundary_facets(cells)}
    assert set(seen) == boundary


def test_three_dimensional_deck_keeps_the_legacy_mesh():
    new = pv.read(DECKS[3] / "mesh/water/mesh-complete.mesh.vtu")
    old = pv.read(LEGACY_3D / "mesh/water/mesh-complete.mesh.vtu")
    assert np.array_equal(np.asarray(new.points), np.asarray(old.points))
    assert np.array_equal(np.asarray(new.cells), np.asarray(old.cells))


@pytest.mark.parametrize("dim", [2, 3])
def test_decks_use_mesh_nitsche_sliding_walls_and_rest(dim):
    root = ET.parse(DECKS[dim] / "solver.xml").getroot()
    assert root.find("GeneralSimulationParameters/Number_of_spatial_dimensions").text == str(dim)
    equations = [eq.get("type") for eq in root.iter("Add_equation")]
    assert equations == ["fluid", "mesh_motion"]
    fluid, mesh = list(root.iter("Add_equation"))
    assert fluid.find("Node_pressure_constraints") is None
    assert fluid.find("Momentum_source_temporal_and_spatial_values_file_path") is None
    surface = [bc for bc in fluid.iter("Add_BC") if bc.get("name") == "free_surface"][0]
    assert surface.find("Kinematic_enforcement").text == "MeshNitsche"
    assert surface.find("Tangential_mesh_policy").text == "Free"
    assert mesh.find("Harmonic_quantity").text == "velocity"
    walls = gen.WALLS_2D if dim == 2 else gen.WALLS_3D
    for eq in (fluid, mesh):
        bcs = {bc.get("name"): bc for bc in eq.iter("Add_BC")}
        for wall, (axis, _) in walls.items():
            direction = bcs[wall].find("Effective_direction").text.split()
            assert direction == ["1" if d == axis else "0" for d in range(dim)], wall
    # Hydrostatic initial pressure, zero on the free surface.
    grid = pv.read(DECKS[dim] / "mesh/water/mesh-complete.mesh.vtu")
    y = np.asarray(grid.points)[:, 1]
    assert np.allclose(grid.point_data["Pressure"], gen.WATER_DENSITY * gen.GRAVITY * (gen.FILL_HEIGHT - y))


def test_boundary_classification_rejects_an_unclassified_facet():
    points, cells = gen.tank_triangles(4, 2)
    with pytest.raises(RuntimeError, match="lies on 0 tank planes"):
        gen.classify_boundary(points, cells, dict(gen.WALLS_2D))     # no free-surface plane


@pytest.mark.parametrize("dim", [2, 3])
def test_forced_variant_writes_the_roll_tables(dim, tmp_path):
    reference = tmp_path / "lateral_water_1x.txt"
    header = ("Time[s]\tPressure[mbar]\tPosition_smooth_splines [deg]\tVelocity[deg\\s]\t"
              "Aceleration[deg\\s2]\tPosition_original [deg]\n")
    rows = "".join(f"{t:.2f}\t0\t{2.0 * t:.6f}\t2.0\t10.0\t0\n" for t in np.arange(0.0, 0.051, 0.01))
    reference.write_text(header + rows, encoding="latin1")

    def table(out, meta):
        root = ET.parse(out / "solver.xml").getroot()
        fluid = list(root.iter("Add_equation"))[0]
        body = fluid.find("Momentum_source_temporal_and_spatial_values_file_path").text
        lines = (out / body).read_text().splitlines()
        n_nodes, n_times = meta["mesh"]["points"], 5
        assert lines[0].split() == [str(dim), str(n_times), str(n_nodes)]
        values = np.array([[float(v) for v in line.split()]
                           for k, line in enumerate(lines[1 + n_times:]) if k % (n_times + 1) != 0])
        return root, fluid, values.reshape(n_nodes, n_times, dim)

    out = tmp_path / "forced"
    meta = gen.write_deck(dim, out, reference_file=reference, end_time=0.04)
    root, fluid, values = table(out, meta)
    omega = fluid.find("Rotating_frame_angular_velocity_temporal_values_file_path")
    assert (omega is not None) == (dim == 3)
    assert root.find("GeneralSimulationParameters/Number_of_time_steps").text == "40"
    points = pv.read(out / "mesh/water/mesh-complete.mesh.vtu").points
    # The tank-frame acceleration is affine in the position: the Euler term
    # alpha y_r makes the x component vary with y at fixed x.
    alpha = np.deg2rad(10.0)
    column = np.isclose(points[:, 0], 0.0)
    ys = points[column, 1]
    assert np.allclose(np.polyfit(ys, values[column, 2, 0], 1)[0], alpha, rtol=1e-6)

    # The x-only diagnostic table: equal values at equal x.
    out_x = tmp_path / "forced_x"
    meta_x = gen.write_deck(dim, out_x, reference_file=reference, end_time=0.04, x_only=True)
    _, _, values_x = table(out_x, meta_x)
    for x in np.unique(np.round(points[:, 0], 12)):
        group = values_x[np.isclose(points[:, 0], x)]
        assert np.allclose(group, group[0], rtol=0.0, atol=1e-12)
