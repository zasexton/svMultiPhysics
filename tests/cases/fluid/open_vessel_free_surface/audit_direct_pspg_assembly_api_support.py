#!/usr/bin/env python3
"""Audit assembly API support for a solve-time direct PSPG topology rule."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(".")
DEFAULT_JSON_OUTPUT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/"
    "test02_test10_direct_pspg_assembly_api_support_20260607.json"
)

SOURCE_FILES = {
    "assembler_header": Path("Code/Source/solver/FE/Assembly/Assembler.h"),
    "fesystem_header": Path("Code/Source/solver/FE/Systems/FESystem.h"),
    "fesystem_cpp": Path("Code/Source/solver/FE/Systems/FESystem.cpp"),
    "forms_installer": Path("Code/Source/solver/FE/Systems/FormsInstaller.cpp"),
    "operator_registry": Path("Code/Source/solver/FE/Systems/OperatorRegistry.h"),
    "standard_assembler": Path(
        "Code/Source/solver/FE/Assembly/StandardAssembler.cpp"
    ),
    "system_assembly": Path("Code/Source/solver/FE/Systems/SystemAssembly.cpp"),
    "system_setup": Path("Code/Source/solver/FE/Systems/SystemSetup.cpp"),
    "navier_stokes_vms": Path(
        "Code/Source/solver/Physics/Formulations/NavierStokes/"
        "IncompressibleNavierStokesVMSModule.cpp"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize whether the current cut-volume assembly API can express "
            "a solve-affecting direct PSPG pressure-gradient support-topology "
            "rule, rather than a diagnostic-only local matrix audit."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def read_repo_files(repo_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, rel_path in SOURCE_FILES.items():
        path = repo_root / rel_path
        out[key] = {
            "path": str(path),
            "exists": path.exists(),
            "text": path.read_text(encoding="utf-8") if path.exists() else "",
        }
    return out


def assembly_diagnostic_context_fields(header_text: str) -> list[str]:
    match = re.search(
        r"struct\s+AssemblyDiagnosticContext\s*\{(?P<body>.*?)\n\s*\};",
        header_text,
        flags=re.S,
    )
    if not match:
        return []
    return sorted(re.findall(r"std::string\s+([A-Za-z_][A-Za-z0-9_]*)", match["body"]))


def planned_cut_volume_term_fields(fesystem_header_text: str) -> list[str]:
    match = re.search(
        r"struct\s+PlannedCutVolumeTerm\s*\{(?P<body>.*?)\n\s*\};",
        fesystem_header_text,
        flags=re.S,
    )
    if not match:
        return []
    body = match["body"]
    fields = set(
        re.findall(
            r"(?:FieldId|GlobalIndex|bool|int|std::string|geometry::CutIntegrationSide)"
            r"\s+([A-Za-z_][A-Za-z0-9_]*)",
            body,
        )
    )
    fields.update(
        re.findall(
            r"(?:const\s+[^;*]+|assembly::AssemblyKernel|"
            r"const\s+dofs::DofMap)\*\s*([A-Za-z_][A-Za-z0-9_]*)",
            body,
        )
    )
    return sorted(fields)


def install_formulation_ops(navier_stokes_text: str) -> dict[str, bool]:
    production_direct_pspg_pattern = re.compile(
        r"installFormulation\s*\(\s*"
        r"system\s*,\s*"
        r"\"equations\"\s*,\s*"
        r"\{\s*p_id\s*\}\s*,\s*"
        r"vms_pspg_pressure_gradient_form\s*,\s*"
        r"direct_pspg_install\s*\)",
        flags=re.S,
    )
    production_direct_pspg_tagged = (
        "direct_pspg_install.source_component_tag" in navier_stokes_text
        and '"navier_stokes_vms_pspg_pressure_gradient"' in navier_stokes_text
        and production_direct_pspg_pattern.search(navier_stokes_text) is not None
    )
    return {
        "production_equations_installed": (
            '"equations"' in navier_stokes_text
            and "const auto residual = momentum_form + continuity_form" in navier_stokes_text
            and re.search(
                r"installFormulation\s*\(\s*system\s*,\s*\"equations\"",
                navier_stokes_text,
            )
            is not None
        ),
        "direct_pspg_diagnostic_operator_installed": (
            '"equations_diagnostic_ns_vms_pspg_pressure_gradient"'
            in navier_stokes_text
        ),
        "production_direct_pspg_subterm_has_source_component_tag": (
            production_direct_pspg_tagged
        ),
        "production_direct_pspg_split_preserves_velocity_tangent": (
            production_direct_pspg_tagged
            and "direct_pspg_install.extra_trial_fields.push_back(u_id)"
            in navier_stokes_text
        ),
        "pressure_row_contribution_diagnostics_env_gated": (
            "pressureRowContributionDiagnosticEnabled()" in navier_stokes_text
        ),
    }


def assembly_api_features(files: dict[str, dict[str, Any]]) -> dict[str, bool]:
    header = files["assembler_header"]["text"]
    fesystem_header = files["fesystem_header"]["text"]
    fesystem_cpp = files["fesystem_cpp"]["text"]
    forms_installer = files["forms_installer"]["text"]
    operator_registry = files["operator_registry"]["text"]
    standard = files["standard_assembler"]["text"]
    system = files["system_assembly"]["text"]
    system_setup = files["system_setup"]["text"]
    navier = files["navier_stokes_vms"]["text"]
    fields = assembly_diagnostic_context_fields(header)
    cut_volume_fields = planned_cut_volume_term_fields(fesystem_header)
    source_component_supported = "source_component_tag" in cut_volume_fields
    return {
        "diagnostic_context_is_documented_non_mutating": (
            "Assemblers should not use this context to alter assembly behavior"
            in header
        ),
        "diagnostic_context_only_operator_and_fields": fields
        == ["operator_tag", "test_field_name", "trial_field_name"],
        "diagnostic_context_has_source_component_tag": (
            "source_component_tag" in fields
        ),
        "diagnostic_context_lacks_topology_policy_handle": (
            not any("topology" in field or "policy" in field for field in fields)
        ),
        "cut_volume_context_built_from_request_op_and_fields": (
            "request.op" in system
            and "test_field.name" in system
            and "trial_field.name" in system
            and "AssemblyDiagnosticContext" in system
        ),
        "cut_volume_context_includes_source_component_tag": (
            "entry.term->source_component_tag" in system
            and "AssemblyDiagnosticContext" in system
        ),
        "fused_composite_terms_may_drop_per_term_diagnostic_context": (
            "fused_term.diagnostic_context = diagnostic_context" in system
            and "term_diagnostic_contexts.emplace_back(std::nullopt)" in system
        ),
        "fused_composite_terms_preserve_source_component_diagnostic_context": (
            "a.source_component_tag == b.source_component_tag" in system
            and "fused_term.diagnostic_context = diagnostic_context" in system
            and "term_diagnostic_contexts.emplace_back(std::nullopt)" not in system
        ),
        "planned_cut_volume_term_lacks_source_component_tag": (
            bool(cut_volume_fields) and not source_component_supported
        ),
        "planned_cut_volume_term_has_source_component_tag": (
            source_component_supported
        ),
        "operator_registry_cut_volume_term_has_source_component_tag": (
            "struct CutVolumeTerm" in operator_registry
            and "std::string source_component_tag" in operator_registry
        ),
        "add_cut_volume_kernel_lacks_source_component_argument": (
            "void FESystem::addCutVolumeKernel" in fesystem_cpp
            and "source_component" not in fesystem_cpp
            and "component_tag" not in fesystem_cpp
        ),
        "add_cut_volume_kernel_has_source_component_argument": (
            "void FESystem::addCutVolumeKernel" in fesystem_cpp
            and "std::string source_component_tag" in fesystem_cpp
            and "std::move(source_component_tag)" in fesystem_cpp
        ),
        "forms_installer_forwards_cut_volume_only_op_fields_kernel": (
            "system.addCutVolumeKernel(" in forms_installer
            and "op, region.marker, toGeometrySide(region.side), test_field, trial_field, kernel"
            in forms_installer
        ),
        "forms_installer_forwards_source_component_tag_to_cut_volumes": (
            "source_component_tag" in forms_installer
            and re.search(
                r"system\.addCutVolumeKernel\s*\((?P<body>.*?)"
                r"source_component_tag\s*\)",
                forms_installer,
                flags=re.S,
            )
            is not None
        ),
        "system_setup_preserves_cut_volume_source_component_tag": (
            "term.source_component_tag" in system_setup
            and "PlannedCutVolumeTerm" in system_setup
        ),
        "diagnostic_logs_include_source_component_tag": (
            "source_component='" in standard and "source_component='" in system
        ),
        "direct_pspg_local_topology_diagnostics_log_before_insert": (
            "logCutVolumeDirectPspgLocalSchurDiagnostic" in standard
            and "logCutVolumeDirectPspgLocalEdgeBalanceDiagnostic" in standard
            and "insertLocalForCell" in standard
            and standard.find("logCutVolumeDirectPspgLocalSchurDiagnostic")
            < standard.find("insertLocalForCell")
        ),
        "direct_pspg_local_topology_diagnostics_mark_diagnostic_only": (
            "diagnostic_only=1" in standard
            and "cut_volume_direct_pspg_local_schur_completion" in standard
            and "cut_volume_direct_pspg_local_edge_balance" in standard
        ),
        "direct_pspg_topology_policy_api_env_gated": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY" in standard
            and "CutVolumeDirectPspgTopologyPolicy::Off" in standard
            and "local_schur_completion" in standard
            and "local_edge_balance" in standard
        ),
        "direct_pspg_topology_policy_scoped_to_equations_operator": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_OPERATOR" in standard
            and 'std::string{"equations"}' in standard
            and "cutVolumeDirectPspgTopologyOperatorFilter" in standard
        ),
        "direct_pspg_topology_policy_scoped_to_production_source_component": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_SOURCE_COMPONENT" in standard
            and "navier_stokes_vms_pspg_pressure_gradient" in standard
            and "cutVolumeDirectPspgTopologySourceComponentFilter" in standard
        ),
        "direct_pspg_topology_policy_requires_pressure_pressure_block": (
            "applyCutVolumeDirectPspgTopologyPolicy" in standard
            and 'fieldNameEquals(diagnostic_context.test_field_name, "pressure")'
            in standard
            and 'fieldNameEquals(diagnostic_context.trial_field_name, "pressure")'
            in standard
            and "row_dofs[i] != col_dofs[i]" in standard
        ),
        "direct_pspg_topology_policy_default_partial_cut_only": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_APPLY_FULL_CELL" in standard
            and "rule.full_cell_equivalent" in standard
        ),
        "direct_pspg_topology_policy_constant_null_preserving": (
            "addNullPreservingPressureEdge" in standard
            and "constant_pressure_null_preserving=1" in standard
            and "output.local_matrix[matrix_index(i, i)] += weight" in standard
            and "output.local_matrix[matrix_index(i, j)] -= weight" in standard
        ),
        "direct_pspg_topology_policy_log_marks_solve_affecting": (
            "cut_volume_direct_pspg_topology_policy" in standard
            and "solve_affecting=1" in standard
            and "diagnostic_only=0" in standard
            and "matrix_mutated=" in standard
        ),
        "direct_pspg_topology_policy_mutates_before_global_insert": (
            "applyCutVolumeDirectPspgTopologyPolicy" in standard
            and "KernelOutput& output" in standard
            and re.search(
                r"applyCutVolumeDirectPspgTopologyPolicy\s*\(.*?"
                r"kernel_output_.*?\)\s*;\s*\}\s*"
                r"stage_start\s*=\s*cut_now\(\)\s*;\s*"
                r"insertLocalForCell",
                standard,
                flags=re.S,
            )
            is not None
        ),
        "production_direct_pspg_subterm_lacks_source_component_tag": (
            not install_formulation_ops(navier)[
                "production_direct_pspg_subterm_has_source_component_tag"
            ]
        ),
        **install_formulation_ops(navier),
    }


def build_report(repo_root: Path = DEFAULT_REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    files = read_repo_files(repo_root)
    features = assembly_api_features(files)
    fields = assembly_diagnostic_context_fields(files["assembler_header"]["text"])
    cut_volume_fields = planned_cut_volume_term_fields(
        files["fesystem_header"]["text"]
    )
    required_missing = {
        "production_subterm_provenance_tag": features[
            "production_direct_pspg_subterm_lacks_source_component_tag"
        ],
        "solve_affecting_local_matrix_mutation_hook": not (
            features["direct_pspg_topology_policy_mutates_before_global_insert"]
            and features["direct_pspg_topology_policy_log_marks_solve_affecting"]
        ),
        "direct_pspg_topology_policy_api": not (
            features["direct_pspg_topology_policy_api_env_gated"]
            and features["direct_pspg_topology_policy_scoped_to_equations_operator"]
            and features[
                "direct_pspg_topology_policy_scoped_to_production_source_component"
            ]
            and features[
                "direct_pspg_topology_policy_requires_pressure_pressure_block"
            ]
            and features["direct_pspg_topology_policy_constant_null_preserving"]
        ),
        "planned_cut_volume_source_component_tag": not features[
            "planned_cut_volume_term_has_source_component_tag"
        ],
        "add_cut_volume_kernel_source_component_argument": not features[
            "add_cut_volume_kernel_has_source_component_argument"
        ],
        "forms_installer_source_component_forwarding": not features[
            "forms_installer_forwards_source_component_tag_to_cut_volumes"
        ],
        "system_setup_source_component_propagation": not features[
            "system_setup_preserves_cut_volume_source_component_tag"
        ],
        "assembly_diagnostic_source_component_context": not (
            features["diagnostic_context_has_source_component_tag"]
            and features["cut_volume_context_includes_source_component_tag"]
            and features["diagnostic_logs_include_source_component_tag"]
        ),
        "legacy_planned_cut_volume_source_component_tag_absent": features[
            "planned_cut_volume_term_lacks_source_component_tag"
        ],
        "composite_term_provenance_for_fused_cut_volume_blocks": features[
            "fused_composite_terms_may_drop_per_term_diagnostic_context"
        ],
    }

    missing_count = sum(1 for value in required_missing.values() if value)
    topology_hook_available = (
        features["production_direct_pspg_subterm_has_source_component_tag"]
        and not required_missing["direct_pspg_topology_policy_api"]
        and not required_missing["solve_affecting_local_matrix_mutation_hook"]
    )
    if missing_count == len(required_missing):
        finding = "assembly_api_lacks_solve_time_direct_pspg_topology_mutation_path"
    elif topology_hook_available and missing_count == 0:
        finding = "assembly_api_has_direct_pspg_topology_policy_hook_replay_pending"
    elif topology_hook_available:
        finding = (
            "assembly_api_has_direct_pspg_topology_policy_hook_"
            "composite_provenance_pending"
        )
    elif features["production_direct_pspg_subterm_has_source_component_tag"]:
        finding = (
            "assembly_api_has_production_direct_pspg_provenance_but_lacks_"
            "topology_policy"
        )
    else:
        finding = (
            "assembly_api_has_cut_volume_source_provenance_but_lacks_direct_"
            "pspg_topology_policy"
        )

    return {
        "finding": finding,
        "status": (
            "requires_subterm_provenance_and_topology_policy_api"
            if finding
            == "assembly_api_lacks_solve_time_direct_pspg_topology_mutation_path"
            else (
                "topology_policy_hook_available_replay_pending"
                if topology_hook_available
                else "production_direct_pspg_provenance_available_mutation_policy_missing"
                if features[
                    "production_direct_pspg_subterm_has_source_component_tag"
                ]
                else "source_component_provenance_available_mutation_policy_missing"
            )
        ),
        "files": {
            key: {
                "path": record["path"],
                "exists": record["exists"],
            }
            for key, record in files.items()
        },
        "assembly_diagnostic_context_fields": fields,
        "planned_cut_volume_term_fields": cut_volume_fields,
        "assembly_api_features": features,
        "required_api_handles_missing": required_missing,
        "missing_required_api_handle_count": missing_count,
        "required_api_handle_count": len(required_missing),
        "conclusion": (
            "The cut-volume assembly pipeline now has source-component "
            "provenance plumbing through FormInstallOptions, the operator "
            "registry, FESystem planning, assembly diagnostic context, and "
            "diagnostic logs. The production Navier-Stokes direct PSPG "
            "pressure-gradient subterm is also installed separately under the "
            "equations operator with source_component_tag while preserving its "
            "velocity tangent dependency. A disabled-by-default, "
            "constant-pressure-null-preserving local topology policy hook can "
            "now mutate the tagged production pressure-pressure local matrix "
            "before global insertion. This rules out missing generic "
            "provenance, missing production direct-PSPG subterm provenance, "
            "and missing solve-affecting local mutation API as blockers. The "
            "cut-volume fused-composite path also preserves source-component "
            "diagnostic context by grouping on source_component_tag. There are "
            "no remaining static assembly API blockers in this audit; the "
            "physics hypothesis remains unproven until short Test02/Test10 "
            "replays exercise the hook."
        ),
        "next_requirement": (
            "Run short Test02/Test10 replay windows with the API-backed "
            "direct PSPG topology policy modes enabled, inspect active/wet "
            "pressure-update histories and policy logs, then either promote "
            "the smallest effective mode or rule out this local topology hook."
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
