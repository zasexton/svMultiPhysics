import importlib.util
import json
from pathlib import Path
import sys


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_stabilization_driver_windows.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_stabilization_driver_windows", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path, *, point_index, update_pa, incident_faces, face_count):
    path.write_text(
        json.dumps(
            {
                "status": "diagnostic_cut_pressure_stabilization_contribution_proxy",
                "finding": "synthetic",
                "mesh_summary": {
                    "active_cut_cell_count": 12,
                    "reconstructed_cut_adjacent_face_count": face_count,
                },
                "worst_active_or_wet_pressure_update": {
                    "point_index": point_index,
                    "support_class": "full_wet_supported",
                    "abs_pressure_delta_pa": update_pa,
                },
                "worst_update_cut_adjacent_correlation": {
                    "incident_cut_adjacent_face_count": incident_faces,
                    "sum_delta_energy_proxy": 0.0,
                    "max_incident_delta_energy_proxy": 0.0,
                    "best_delta_energy_rank": None,
                },
                "direct_driver_assessment": {
                    "classification": (
                        "worst_update_not_incident_to_cut_adjacent_stabilization"
                        if incident_faces == 0
                        else "worst_update_incident_to_cut_adjacent_stabilization"
                    ),
                    (
                        "direct_cut_adjacent_pressure_stabilization_driver_"
                        "ruled_out"
                    ): incident_faces == 0,
                    (
                        "direct_cut_adjacent_pressure_stabilization_driver_"
                        "supported"
                    ): incident_faces > 0,
                },
                "top_faces_by_delta_energy_proxy": [
                    {
                        "delta_energy_proxy": 2.5,
                        "max_abs_pressure_delta_adjacent_cell_nodes_pa": 13.0,
                        "applied_metadata_scale": 3.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_saved_window_pair_rules_out_direct_ghost_penalty_driver(tmp_path):
    audit = _load_audit_module()
    test02 = tmp_path / "test02.json"
    test10 = tmp_path / "test10.json"
    _write_report(test02, point_index=1172, update_pa=2112204.1, incident_faces=0, face_count=2088)
    _write_report(test10, point_index=3, update_pa=1075.21, incident_faces=0, face_count=429)

    report = audit.build_report(test02_path=test02, test10_path=test10)

    assert report["finding"] == (
        "cut_adjacent_pressure_stabilization_not_direct_worst_update_driver"
    )
    assert report["status"] == (
        "ghost_penalty_direct_worst_update_path_ruled_out_for_saved_windows"
    )
    assert report["all_saved_window_worst_updates_nonincident"]
    assert not report["any_saved_window_worst_update_incident"]
    assert {case["label"]: case["direct_driver_ruled_out"] for case in report["cases"]} == {
        "test02": True,
        "test10": True,
    }


def test_saved_window_pair_flags_incident_worst_update(tmp_path):
    audit = _load_audit_module()
    test02 = tmp_path / "test02.json"
    test10 = tmp_path / "test10.json"
    _write_report(test02, point_index=1172, update_pa=2112204.1, incident_faces=0, face_count=2088)
    _write_report(test10, point_index=3, update_pa=1075.21, incident_faces=2, face_count=429)

    report = audit.build_report(test02_path=test02, test10_path=test10)

    assert report["finding"] == (
        "cut_adjacent_pressure_stabilization_incident_to_some_worst_updates"
    )
    assert report["any_saved_window_worst_update_incident"]
    cases = {case["label"]: case for case in report["cases"]}
    assert cases["test10"]["direct_driver_supported"]
    assert cases["test10"]["incident_cut_adjacent_face_count"] == 2
