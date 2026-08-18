#!/usr/bin/env python3
"""Audit solve-time direct PSPG support/coupling provenance readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(".")
DEFAULT_JSON_OUTPUT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/"
    "test02_test10_direct_pspg_solve_time_provenance_support_20260607.json"
)

STANDARD_ASSEMBLER = Path(
    "Code/Source/solver/FE/Assembly/StandardAssembler.cpp"
)
NAVIER_STOKES_VMS = Path(
    "Code/Source/solver/Physics/Formulations/NavierStokes/"
    "IncompressibleNavierStokesVMSModule.cpp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the solver has an env-gated direct PSPG "
            "support/coupling provenance diagnostic scoped to the tagged "
            "production pressure-gradient source component."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_body(text: str, function_name: str) -> str:
    start = text.find(function_name)
    if start < 0:
        return ""
    brace = text.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    return ""


def called_before(text: str, first: str, second: str) -> bool:
    first_index = text.find(first)
    second_index = text.find(second)
    return first_index >= 0 and second_index >= 0 and first_index < second_index


def call_with_argument_before(text: str, argument: str, following: str) -> bool:
    search_from = 0
    call_name = "logCutVolumeDirectPspgSupportCouplingProvenance("
    while True:
        call_index = text.find(call_name, search_from)
        if call_index < 0:
            return False
        call_end = text.find(");", call_index)
        call_args = text[call_index:call_end] if call_end >= 0 else ""
        next_call_index = text.find(call_name, call_index + len(call_name))
        window_end = next_call_index if next_call_index >= 0 else len(text)
        if argument in call_args:
            following_index = text.find(following, call_index)
            return following_index >= 0 and following_index < window_end
        search_from = call_index + len(call_name)


def build_report(
    *,
    standard_assembler_text: str,
    navier_stokes_text: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    helper = function_body(
        standard_assembler_text,
        "logCutVolumeDirectPspgSupportCouplingProvenance",
    )
    env_helper = function_body(
        standard_assembler_text,
        "cutVolumeDirectPspgSupportCouplingProvenanceDiagnosticEnabled",
    )
    op_filter = function_body(
        standard_assembler_text,
        "cutVolumeDirectPspgSupportCouplingOperatorFilter",
    )
    source_filter = function_body(
        standard_assembler_text,
        "cutVolumeDirectPspgSupportCouplingSourceComponentFilter",
    )
    helper_has_mutation_tokens = (
        "output.local_matrix[matrix_index" in helper
        and (
            "]+=" in helper
            or "]-=" in helper
            or "] +=" in helper
            or "] -=" in helper
        )
    )
    features = {
        "env_flag_present": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC"
            in env_helper
        ),
        "diagnostic_name_emitted": (
            "diagnostic=cut_volume_direct_pspg_support_coupling_provenance"
            in helper
        ),
        "operator_filter_env_present": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR"
            in op_filter
            and 'std::string{"equations"}' in op_filter
        ),
        "source_component_filter_env_present": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT"
            in source_filter
            and "navier_stokes_vms_pspg_pressure_gradient" in source_filter
        ),
        "scoped_to_pressure_test_rows": (
            'fieldNameEquals(diagnostic_context.test_field_name, "pressure")'
            in helper
        ),
        "emits_pressure_pressure_block": (
            "pressure_pressure" in helper
            and 'fieldNameEquals(diagnostic_context.trial_field_name, "pressure")'
            in helper
        ),
        "emits_pressure_velocity_block": (
            "pressure_velocity" in helper
            and 'fieldNameEquals(diagnostic_context.trial_field_name, "velocity")'
            in helper
        ),
        "emits_preupdate_pressure_graph_metrics": (
            "source_edge_count=" in helper
            and "two_hop_completion_count=" in helper
            and "local_clustering=" in helper
        ),
        "emits_sampled_column_payload": (
            "sampled_col_local_indices" in helper
            and "sampled_col_dofs" in helper
            and "sampled_col_values" in helper
            and "sampled_col_abs_values" in helper
            and "sampled_col_signs" in helper
        ),
        "uses_bounded_column_sample": (
            "cutVolumeLocalMatrixColumnSupportMaxColumns()" in helper
            and "sampled_col_count=" in helper
            and "sample_truncated=" in helper
        ),
        "records_sample_order_and_diag_membership": (
            "sample_sorted_by=abs_desc" in helper
            and "diag_in_sample=" in helper
        ),
        "records_pressure_update_sign_not_used": (
            "pressure_update_sign_used=0" in helper
        ),
        "does_not_use_same_sign_or_update_values": (
            "same_sign" not in helper.lower()
            and "top_update" not in helper.lower()
        ),
        "diagnostic_only": "diagnostic_only=1" in helper,
        "does_not_mutate_matrix": not helper_has_mutation_tokens,
        "called_before_legacy_cut_volume_insert": call_with_argument_before(
            standard_assembler_text,
            "*diagnostic_context_",
            "insertLocalForCell(cell_id, row_dof_map_",
        ),
        "called_before_fused_cut_volume_insert": call_with_argument_before(
            standard_assembler_text,
            "*active_diagnostic_context",
            "insertLocalForCell(cell_id, t.row_dof_map",
        ),
        "called_before_topology_policy": called_before(
            standard_assembler_text,
            "logCutVolumeDirectPspgSupportCouplingProvenance(",
            "applyCutVolumeDirectPspgTopologyPolicy(",
        ),
        "production_direct_pspg_source_component_tagged": (
            "direct_pspg_install.source_component_tag" in navier_stokes_text
            and "navier_stokes_vms_pspg_pressure_gradient" in navier_stokes_text
        ),
        "production_direct_pspg_preserves_velocity_tangent_dependency": (
            "direct_pspg_install.extra_trial_fields.push_back(u_id)"
            in navier_stokes_text
        ),
    }
    missing = sorted(key for key, value in features.items() if not value)
    if not missing:
        finding = "solve_time_direct_pspg_support_coupling_provenance_ready"
        status = "diagnostic_ready_replay_pending"
        next_requirement = (
            "Run short Test02/Test10 replay windows with "
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC=1, "
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR=equations, "
            "and "
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT="
            "navier_stokes_vms_pspg_pressure_gradient, then audit whether the "
            "sampled pressure-neighbor column payload joined with "
            "pressure-velocity coupling separates Test02 row 10676, the other "
            "Test02 direct targets, and the Test10 direct PSPG patch."
        )
    else:
        finding = "solve_time_direct_pspg_support_coupling_provenance_incomplete"
        status = "diagnostic_support_missing"
        next_requirement = (
            "Complete the missing source-level diagnostic features before "
            "running replay windows."
        )
    return {
        "finding": finding,
        "status": status,
        "repo_root": str(repo_root) if repo_root is not None else None,
        "source_files": {
            "standard_assembler": str(STANDARD_ASSEMBLER),
            "navier_stokes_vms": str(NAVIER_STOKES_VMS),
        },
        "features": features,
        "missing_features": missing,
        "diagnostic_env": {
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC": "1",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR": "equations",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT": (
                "navier_stokes_vms_pspg_pressure_gradient"
            ),
        },
        "conclusion": (
            "A production-scoped direct PSPG pressure-gradient support/coupling "
            "provenance diagnostic is available. It emits pressure-pressure "
            "graph support, bounded sampled column support, and "
            "pressure-velocity coupling rows before pressure updates are known "
            "and before the optional topology policy mutates the local matrix."
        ),
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report = build_report(
        standard_assembler_text=read_text(repo_root / STANDARD_ASSEMBLER),
        navier_stokes_text=read_text(repo_root / NAVIER_STOKES_VMS),
        repo_root=repo_root,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
