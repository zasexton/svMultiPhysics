import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_stabilization_contribution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_stabilization_contribution", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tetra_gradient_recovers_linear_pressure_field():
    audit = _load_audit_module()
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tet = np.asarray([0, 1, 2, 3], dtype=np.int64)
    pressure = points @ np.asarray([1.0, 2.0, 3.0]) + 4.0

    assert np.allclose(audit.tetra_gradient(points, tet, pressure), [1.0, 2.0, 3.0])


def test_cut_adjacent_faces_require_cut_cell_and_two_wet_neighbors():
    audit = _load_audit_module()
    tets = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
        ],
        dtype=np.int64,
    )
    config = audit.PressureStabilizationConfig(
        use_cut_metadata_scale=True,
        metadata_scale_cap=3.0,
    )

    faces = audit.reconstruct_cut_adjacent_faces(
        tets,
        np.asarray([0.25, 1.0], dtype=float),
        config,
        full_wet_tolerance=1.0e-12,
    )

    assert len(faces) == 1
    assert faces[0].first_cell_cut
    assert not faces[0].second_cell_cut
    assert faces[0].raw_metadata_scale == 4.0
    assert faces[0].applied_metadata_scale == 3.0

    dry_neighbor_faces = audit.reconstruct_cut_adjacent_faces(
        tets,
        np.asarray([0.25, 0.0], dtype=float),
        config,
        full_wet_tolerance=1.0e-12,
    )
    assert dry_neighbor_faces == []


def test_metadata_disabled_uses_unit_applied_scale():
    audit = _load_audit_module()
    config = audit.PressureStabilizationConfig(use_cut_metadata_scale=False)

    assert audit.applied_scale(25.0, config) == 1.0


def test_driver_assessment_rules_out_nonincident_worst_update():
    audit = _load_audit_module()
    assessment = audit.driver_assessment(
        face_reports=[{"face_index": 0, "delta_energy_proxy": 1.0}],
        worst_update={
            "point_index": 7,
            "support_class": "full_wet_supported",
            "abs_pressure_delta_pa": 123.0,
        },
        node_correlation={
            "incident_cut_adjacent_face_count": 0,
            "sum_delta_energy_proxy": 0.0,
            "max_incident_delta_energy_proxy": 0.0,
            "best_delta_energy_rank": None,
        },
    )

    assert assessment["classification"] == (
        "worst_update_not_incident_to_cut_adjacent_stabilization"
    )
    assert assessment["direct_cut_adjacent_pressure_stabilization_driver_ruled_out"]
    assert not assessment[
        "direct_cut_adjacent_pressure_stabilization_driver_supported"
    ]
