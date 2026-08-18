import importlib.util
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
        / "audit_direct_pspg_named_face_provenance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_named_face_provenance_selectivity",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _global_emission():
    return {
        "cases": [
            {
                "label": "test02",
                "preferred_candidate_global_dofs": [10, 20, 30, 31, 32],
                "sparse_direct_self_global_dofs": [10, 20, 30, 31],
                "sparse_or_moderate_direct_self_ratio_global_dofs": [
                    10,
                    20,
                    30,
                    31,
                ],
            },
            {
                "label": "test10",
                "preferred_candidate_global_dofs": [100, 101, 102, 103, 104],
                "sparse_direct_self_global_dofs": [100, 101, 102, 103],
                "sparse_or_moderate_direct_self_ratio_global_dofs": [
                    100,
                    101,
                    102,
                    103,
                ],
            },
        ]
    }


def _target_map():
    return {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [10, 20]},
            {"label": "test10", "direct_pspg_target_global_dofs": [100, 101]},
        ]
    }


def _profile(*items):
    profile = {}
    for row, faces in items:
        families = sorted({face.split("_", 1)[0] for face in faces})
        face_count = len(faces)
        if face_count == 0:
            face_class = "no_named_face"
        elif face_count == 1:
            face_class = "single_named_face"
        elif face_count == 2:
            face_class = "named_face_intersection"
        else:
            face_class = "multi_face_intersection"
        profile[row] = {
            "point_id": row,
            "named_faces": sorted(faces),
            "face_families": families,
            "face_class": face_class,
            "named_face_count": face_count,
        }
    return profile


def test_named_face_provenance_rules_out_broad_boundary_membership():
    audit = _load_audit_module()
    profiles = {
        "test02": _profile(
            (10, ["wall_bottom"]),
            (20, ["wall_bottom"]),
            (30, ["wall_bottom"]),
            (31, ["wall_bottom"]),
            (32, ["wall_bottom"]),
        ),
        "test10": _profile(
            (100, ["wall_left"]),
            (101, ["wall_left"]),
            (102, ["wall_left"]),
            (103, ["wall_left"]),
            (104, ["wall_left"]),
        ),
    }

    report = audit.build_report(
        global_emission=_global_emission(),
        target_map=_target_map(),
        profiles_by_case=profiles,
        profile_evidence_by_case={
            "test02": {"profile_status": "ok"},
            "test10": {"profile_status": "ok"},
        },
        max_target_ratio=1.5,
    )

    assert report["finding"] == (
        "direct_pspg_named_face_provenance_selectors_not_formulation_ready"
    )
    assert report["status"] == "named_face_boundary_gate_ruled_out"
    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["target_named_faces"] == ["wall_bottom"]
    assert any(
        selector["key"] == "preferred_target_named_face_union"
        and selector["finding"] == "selector_overbroad"
        for selector in test02["selectors"]
    )


def test_named_face_provenance_reports_missing_profiles():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission=_global_emission(),
        target_map=_target_map(),
        profiles_by_case={"test02": {}, "test10": {}},
        profile_evidence_by_case={
            "test02": {"profile_status": "source_result_missing"},
            "test10": {"profile_status": "source_result_missing"},
        },
    )

    assert report["finding"] == (
        "direct_pspg_named_face_provenance_selectivity_missing_evidence"
    )
