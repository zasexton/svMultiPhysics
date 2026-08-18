import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import pyvista as pv


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_update_neighborhood.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_update_neighborhood",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_quad(path: Path, pressure: np.ndarray) -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    cells = np.asarray([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.asarray([pv.CellType.QUAD], dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["Pressure"] = pressure.astype(float)
    grid.point_data["phi"] = np.full(4, -1.0)
    grid.point_data["ActiveFluid"] = np.ones(4)
    grid.point_data["Velocity"] = np.zeros((4, 3))
    grid.cell_data["WetVolumeFraction"] = np.ones(1)
    grid.save(path)


def test_pressure_update_neighborhood_reports_neighbor_and_patch_coherence(tmp_path):
    audit = _load_audit_module()
    previous = tmp_path / "result_000.vtu"
    current = tmp_path / "result_001.vtu"
    _write_quad(previous, np.zeros(4))
    _write_quad(current, np.asarray([10.0, 8.0, -1.0, 7.0]))

    report = audit.transition_neighborhood_report(
        previous,
        current,
        result_prefix="result",
        previous_time=0.0,
        current_time=1.0,
        top_events=1,
        neighbor_count=2,
        neighbor_detail_limit=2,
        patch_detail_limit=3,
        active_threshold=0.5,
        tiny_wet_fraction=1.0e-4,
        full_wet_tolerance=1.0e-12,
        selection_mode="active_or_wet_supported",
    )

    assert report["top_event_count"] == 1
    assert report["support_counts"]["full_wet_supported"] == 4
    top = report["top_update_neighborhoods"][0]
    assert top["point_index"] == 0
    assert top["pressure_delta_pa"] == 10.0
    assert top["support_class"] == "full_wet_supported"
    assert top["nearest_neighbor_count"] == 2
    assert top["incident_patch_point_count"] == 3
    assert top["nearest_neighbors"]["same_sign_count"] == 2
    assert top["nearest_neighbors"]["same_sign_fraction"] == 1.0
    assert top["nearest_neighbors"]["median_abs_delta_pa"] == 7.5
    assert top["nearest_neighbors"]["target_abs_to_median_abs_ratio"] == pytest.approx(
        10.0 / 7.5
    )
    assert top["incident_patch"]["same_sign_count"] == 2
    assert top["incident_patch"]["opposite_sign_count"] == 1
    assert top["incident_patch"]["median_abs_delta_pa"] == 7.0
    assert top["incident_patch"]["target_abs_to_max_abs_ratio"] == pytest.approx(
        10.0 / 8.0
    )
    assert [item["point_index"] for item in top["nearest_neighbor_details"]] == [1, 3]
    assert [item["point_index"] for item in top["largest_patch_delta_details"]] == [
        1,
        3,
        2,
    ]
