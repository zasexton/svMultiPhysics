#!/usr/bin/env python3
"""Classify saved-window pressure ghost-penalty driver evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TEST02 = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_abs_only_prune1e5_0p54_pressure_stabilization_contribution_audit_20260605.json"
)
DEFAULT_TEST10 = (
    DEFAULT_ARTIFACT_ROOT
    / "test10_cap3_step90_pressure_stabilization_contribution_audit_20260605.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join Test02/Test10 cut-adjacent pressure stabilization proxy audits "
            "into a compact direct-driver classification."
        )
    )
    parser.add_argument("--test02-json", type=Path, default=DEFAULT_TEST02)
    parser.add_argument("--test10-json", type=Path, default=DEFAULT_TEST10)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def summarize_case(label: str, path: Path, report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "finding": "pressure_stabilization_audit_missing",
            "direct_driver_ruled_out": False,
            "direct_driver_supported": False,
        }

    update = as_dict(report.get("worst_active_or_wet_pressure_update"))
    correlation = as_dict(report.get("worst_update_cut_adjacent_correlation"))
    mesh = as_dict(report.get("mesh_summary"))
    assessment = as_dict(report.get("direct_driver_assessment"))
    if not assessment:
        incident_count = int(correlation.get("incident_cut_adjacent_face_count") or 0)
        face_count = int(mesh.get("reconstructed_cut_adjacent_face_count") or 0)
        direct_ruled_out = face_count == 0 or incident_count == 0
        classification = (
            "worst_update_not_incident_to_cut_adjacent_stabilization"
            if direct_ruled_out
            else "worst_update_incident_to_cut_adjacent_stabilization"
        )
        assessment = {
            "classification": classification,
            "direct_cut_adjacent_pressure_stabilization_driver_ruled_out": (
                direct_ruled_out
            ),
            "direct_cut_adjacent_pressure_stabilization_driver_supported": (
                not direct_ruled_out
            ),
        }

    top_delta = report.get("top_faces_by_delta_energy_proxy")
    top_face = top_delta[0] if isinstance(top_delta, list) and top_delta else {}
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "source_status": report.get("status"),
        "source_finding": report.get("finding"),
        "finding": assessment.get("classification"),
        "direct_driver_ruled_out": bool(
            assessment.get(
                "direct_cut_adjacent_pressure_stabilization_driver_ruled_out"
            )
        ),
        "direct_driver_supported": bool(
            assessment.get(
                "direct_cut_adjacent_pressure_stabilization_driver_supported"
            )
        ),
        "worst_update_point_index": update.get("point_index"),
        "worst_update_support_class": update.get("support_class"),
        "worst_update_abs_pressure_delta_pa": update.get("abs_pressure_delta_pa"),
        "reconstructed_cut_adjacent_face_count": mesh.get(
            "reconstructed_cut_adjacent_face_count"
        ),
        "active_cut_cell_count": mesh.get("active_cut_cell_count"),
        "incident_cut_adjacent_face_count": correlation.get(
            "incident_cut_adjacent_face_count"
        ),
        "sum_incident_delta_energy_proxy": correlation.get("sum_delta_energy_proxy"),
        "max_incident_delta_energy_proxy": correlation.get(
            "max_incident_delta_energy_proxy"
        ),
        "best_incident_delta_energy_rank": correlation.get("best_delta_energy_rank"),
        "top_delta_energy_proxy": top_face.get("delta_energy_proxy"),
        "top_delta_face_max_adjacent_pressure_delta_pa": top_face.get(
            "max_abs_pressure_delta_adjacent_cell_nodes_pa"
        ),
        "top_delta_applied_metadata_scale": top_face.get("applied_metadata_scale"),
    }


def build_report(
    *,
    test02_path: Path = DEFAULT_TEST02,
    test10_path: Path = DEFAULT_TEST10,
) -> dict[str, Any]:
    cases = []
    for label, path in (("test02", test02_path), ("test10", test10_path)):
        report = load_json(path) if path.exists() else None
        cases.append(summarize_case(label, path, report))

    missing = [case["label"] for case in cases if not case["exists"]]
    all_ruled_out = not missing and all(case["direct_driver_ruled_out"] for case in cases)
    any_supported = any(case["direct_driver_supported"] for case in cases)
    if all_ruled_out:
        finding = "cut_adjacent_pressure_stabilization_not_direct_worst_update_driver"
        status = "ghost_penalty_direct_worst_update_path_ruled_out_for_saved_windows"
        next_requirement = (
            "Treat cut-adjacent pressure ghost penalty as branch-shaping evidence "
            "only; continue with direct PSPG/active pressure-support consistency."
        )
    elif any_supported:
        finding = "cut_adjacent_pressure_stabilization_incident_to_some_worst_updates"
        status = "ghost_penalty_direct_worst_update_path_supported_for_some_windows"
        next_requirement = (
            "Run assembled-row contribution sampling for the incident worst-update "
            "case before changing the stabilization formulation."
        )
    else:
        finding = "cut_adjacent_pressure_stabilization_driver_evidence_incomplete"
        status = "missing_pressure_stabilization_window_evidence"
        next_requirement = "Regenerate missing pressure-stabilization saved-window audits."

    return {
        "finding": finding,
        "status": status,
        "case_count": len(cases),
        "missing_cases": missing,
        "all_saved_window_worst_updates_nonincident": all_ruled_out,
        "any_saved_window_worst_update_incident": any_supported,
        "cases": cases,
        "next_requirement": next_requirement,
        "limitations": (
            "This is an offline saved-VTU proxy for the cut-adjacent pressure "
            "ghost-penalty face class and h^3/mu scaling. It rules out direct "
            "incidence at the worst saved-window updates, but it is not an exact "
            "assembled residual dump and does not rule out branch shaping."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(test02_path=args.test02_json, test10_path=args.test10_json)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
