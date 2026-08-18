#!/usr/bin/env python3
"""Audit whether current Forms vocabulary can express the remaining PSPG gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(".")
DEFAULT_JSON_OUTPUT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/"
    "test02_test10_direct_pspg_formulation_vocabulary_support_20260607.json"
)

FORM_FILES = {
    "cut_cell_forms": Path("Code/Source/solver/FE/Forms/CutCellForms.h"),
    "form_expr": Path("Code/Source/solver/FE/Forms/FormExpr.h"),
    "vocabulary": Path("Code/Source/solver/FE/Forms/VOCABULARY.md"),
    "navier_stokes_vms": Path(
        "Code/Source/solver/Physics/Formulations/NavierStokes/"
        "IncompressibleNavierStokesVMSModule.cpp"
    ),
}

REQUIRED_TOPOLOGY_HANDLES = {
    "active_pressure_graph_connectivity": [
        "activePressureGraph",
        "pressureGraph",
        "pressureConnectivity",
    ],
    "element_local_schur_completion": [
        "localSchur",
        "schurCompletion",
        "elementLocalSchur",
    ],
    "existing_pressure_edge_balance": [
        "edgeBalance",
        "pressureEdgeBalance",
        "existingPressureEdge",
    ],
    "direct_pspg_local_matrix_provenance": [
        "directPspgLocalMatrix",
        "localMatrixProvenance",
        "pressureGradientProvenance",
    ],
    "post_update_same_sign_pressure_action": [
        "sameSignPressureAction",
        "pressureActionSign",
        "postUpdatePressureSign",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize whether the current Forms expression vocabulary can "
            "represent the support/topology rule required by the remaining "
            "Test02/Test10 direct PSPG pressure-gradient evidence."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def read_repo_files(repo_root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for key, rel_path in FORM_FILES.items():
        path = repo_root / rel_path
        files[key] = {
            "path": str(path),
            "exists": path.exists(),
            "text": path.read_text(encoding="utf-8") if path.exists() else "",
        }
    return files


def public_cut_cell_helpers(text: str) -> list[str]:
    helpers = set(
        re.findall(r"\bFormExpr\s+(cut[A-Za-z0-9_]+)\s*\(", text)
    )
    helpers.update(re.findall(r"\b(CutCell[A-Za-z0-9_]+)\b", text))
    return sorted(helpers)


def public_measures(text: str) -> list[str]:
    known_measures = ["dx", "ds", "dS", "dI", "dCutVolume"]
    return [
        measure
        for measure in known_measures
        if re.search(rf"\b{re.escape(measure)}\s*\(", text)
    ]


def missing_required_handles(vocabulary_text: str, cut_cell_text: str) -> dict[str, bool]:
    combined = vocabulary_text + "\n" + cut_cell_text
    return {
        key: not any(token in combined for token in tokens)
        for key, tokens in REQUIRED_TOPOLOGY_HANDLES.items()
    }


def direct_pspg_expression_summary(text: str) -> dict[str, bool]:
    direct_integrand = "vms_pspg_pressure_gradient_integrand" in text
    return {
        "direct_pressure_gradient_integrand_installed": direct_integrand,
        "active_volume_measure_used": (
            "integrateOnActiveVolume" in text and "dCutVolume" in text
        ),
        "cut_volume_fraction_scale_available": (
            "pspg_pressure_gradient_support_scale" in text
            and "cutVolumeFraction()" in text
        ),
        "free_surface_boundary_terms_separate": (
            "vms_pspg_boundary_pressure_gradient_form" in text
            and "vms_pspg_boundary_tangential_pressure_gradient_form" in text
        ),
    }


def build_report(repo_root: Path = DEFAULT_REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    files = read_repo_files(repo_root)
    cut_cell_text = files["cut_cell_forms"]["text"]
    form_expr_text = files["form_expr"]["text"]
    vocabulary_text = files["vocabulary"]["text"]
    ns_text = files["navier_stokes_vms"]["text"]

    required_missing = missing_required_handles(vocabulary_text, cut_cell_text)
    direct_pspg_summary = direct_pspg_expression_summary(ns_text)
    missing_count = sum(1 for missing in required_missing.values() if missing)

    finding = (
        "form_vocabulary_lacks_direct_pspg_support_topology_handles"
        if missing_count == len(required_missing)
        else "form_vocabulary_has_some_direct_pspg_support_topology_handles"
    )
    status = (
        "requires_fe_forms_or_assembly_api_extension"
        if finding == "form_vocabulary_lacks_direct_pspg_support_topology_handles"
        else "candidate_form_vocabulary_handles_present"
    )

    return {
        "finding": finding,
        "status": status,
        "files": {
            key: {
                "path": record["path"],
                "exists": record["exists"],
            }
            for key, record in files.items()
        },
        "public_cut_cell_helpers": public_cut_cell_helpers(cut_cell_text),
        "public_measures": public_measures(form_expr_text),
        "direct_pspg_expression_summary": direct_pspg_summary,
        "required_topology_handles_missing": required_missing,
        "missing_required_topology_handle_count": missing_count,
        "required_topology_handle_count": len(required_missing),
        "conclusion": (
            "The current form-level direct PSPG pressure-gradient expression can "
            "select cut-volume measure support and scalar cut-volume metadata, "
            "but the audited remaining rule requires pressure graph/support "
            "topology, local Schur or existing-edge balance, or post-update "
            "pressure-action information that is not public Forms vocabulary."
        ),
        "next_requirement": (
            "Do not search for another scalar form multiplier in the current "
            "DSL; add an FE Forms/assembly API that exposes the needed solve-time "
            "direct PSPG support topology, or keep the rule as an assembly-level "
            "diagnostic until such an API exists."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.repo_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
