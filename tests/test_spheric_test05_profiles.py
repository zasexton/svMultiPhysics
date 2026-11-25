import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
CASE = Path(__file__).parent / "cases/fluid/open_vessel_free_surface"
REFERENCE = CASE / "reference_profiles/spheric_test05_wet_bed"
spec = importlib.util.spec_from_file_location("compare_test05_profiles", CASE / "compare_test05_profiles.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def mesh_with_zero_edges(segments):
    """Give each segment a triangle whose zero contour is exactly that edge."""
    a, b = segments[:, 0], segments[:, 1]
    dx, dy = (b - a).T
    interior = (a + b) / 2 + np.column_stack((dy, -dx)) / 4
    xy = np.stack((a, b, interior), axis=1).reshape(-1, 2)
    points = np.column_stack((xy, np.zeros(len(xy))))
    cells = np.column_stack((np.full(len(segments), 3), np.arange(len(xy)).reshape(-1, 3)))
    mesh = pv.UnstructuredGrid(cells.ravel(), np.full(len(segments), pv.CellType.TRIANGLE), points)
    mesh.point_data["phi"] = np.tile([0., 0., -1.], len(segments))
    return mesh


@pytest.fixture
def score(tmp_path, monkeypatch, capsys):
    def run(mesh, reference, *flags):
        result = tmp_path / "result.vtu"
        mesh.save(result)
        if not isinstance(reference, Path):
            path = tmp_path / "measured.dat"
            np.savetxt(path, np.asarray(reference) * 100)  # SPHERIC files are in cm.
            reference = path
        monkeypatch.setattr(sys, "argv", ["compare_test05_profiles", str(result), str(reference), *flags])
        scorer.main()
        return json.loads(capsys.readouterr().out)["profile_comparison"]["metrics"]
    return run


@pytest.mark.parametrize("path", sorted(REFERENCE.glob("*.dat")), ids=lambda path: path.stem)
def test_identical_measured_curve_after_subdivision_and_reordering(path, score):
    measured = np.loadtxt(path) * 0.01
    a, b = measured[:-1], measured[1:]
    midpoint = (a + b) / 2
    # Change vertex density, cell order and edge direction without changing the curve.
    segments = np.concatenate([np.stack((midpoint, a), axis=1), np.stack((b, midpoint), axis=1)])
    metrics = score(mesh_with_zero_edges(segments), path)
    assert metrics["rmse_m"] < 1e-12
    assert metrics["max_abs_error_m"] < 1e-12
    assert metrics["direct_coverage_fraction"] == 1


def test_fold_vertical_edge_and_duplicate_station_use_one_upper_height(score):
    measured = np.array([[0., 0.1], [0.5, 0.2], [0.5, 0.7], [0.25, 0.5], [1., 0.3]])
    raised = measured + [0., 0.003]
    metrics = score(mesh_with_zero_edges(np.stack((raised[:-1], raised[1:]), axis=1)), measured)
    assert metrics["samples"] == 4
    assert metrics["rmse_m"] == pytest.approx(0.003)
    assert metrics["mean_error_m"] == pytest.approx(0.003)


def test_missing_contour_in_window_reduces_coverage(score):
    segments = np.array([[[0., 0.1], [0.25, 0.1]], [[0.75, 0.1], [1., 0.1]]])
    measured = [[x, 0.1] for x in [-0.05, 0., 0.125, 0.5, 0.875, 1., 1.05]]
    metrics = score(mesh_with_zero_edges(segments), measured, "--x-min", "0", "--x-max", "1")
    assert metrics["samples"] == 4
    assert metrics["coverage_fraction"] == metrics["direct_coverage_fraction"] == 0.8
    assert metrics["interpolated_samples"] == 0


def test_extruded_surface_intersects_stations_with_no_vertices_in_window(score):
    points = np.array([[0., 0., 0.], [1., 1., 0.], [0., 1., 1.], [0., 0., -1.]])
    mesh = pv.UnstructuredGrid([4, 0, 1, 2, 3], [pv.CellType.TETRA], points)
    mesh.point_data["phi"] = [0., 0., 0., -1.]
    metrics = score(mesh, [[0.25, 1.], [0.5, 1.], [0.75, 1.]])
    assert metrics["rmse_m"] == pytest.approx(0)
    assert metrics["direct_coverage_fraction"] == 1
    assert metrics["simulated_front_x_m"] == 0.75


def test_measured_vertical_tip_is_scored_at_its_highest_point(score):
    measured = [[0., 0.1], [0.5, 0.1], [0.5, 0.7], [0.5, 0.1], [1., 0.1]]
    mesh = mesh_with_zero_edges(np.array([[[0., 0.1], [1., 0.1]]]))
    metrics = score(mesh, measured)
    assert metrics["max_abs_error_m"] == pytest.approx(0.6)
    assert metrics["rmse_m"] == pytest.approx(0.6 / np.sqrt(3))


def test_explicit_radius_reproduces_historical_vertex_window(score):
    measured = np.array([[0., 0.], [0.5, 0.5], [1., 1.]])
    mesh = mesh_with_zero_edges(np.stack((measured[:-1], measured[1:]), axis=1))
    metrics = score(mesh, measured, "--sample-radius", "0.6")
    assert metrics["sample_radius_m"] == 0.6
    assert metrics["rmse_m"] == pytest.approx(np.sqrt(1 / 6))


@pytest.mark.parametrize("depth,labels_ms", [
    (18, [156, 219, 281, 343, 406, 468, 531]),
    (38, [156, 219, 281, 343, 406, 468, 531, 593]),
])
def test_case_matches_numbered_profiles_to_published_photo_labels(depth, labels_ms):
    benchmark = CASE / "unfitted_level_set" / f"spheric_test05_wet_bed_d{depth}" / "benchmark.json"
    entries = json.loads(benchmark.read_text())["reference_profiles"]
    assert [(Path(entry["path"]).name, entry["time_s"]) for entry in entries] == [
        (f"d{depth}_{i}.dat", milliseconds / 1000) for i, milliseconds in enumerate(labels_ms, 1)
    ]
