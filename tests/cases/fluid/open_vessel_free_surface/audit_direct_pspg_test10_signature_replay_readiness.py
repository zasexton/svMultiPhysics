#!/usr/bin/env python3
"""Audit readiness for a targeted Test10 direct PSPG signature replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_support_coupling_signature import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    read_provenance_log,
    signature_tuple,
    summarize_rows,
    target_rows_by_case,
)


DEFAULT_STANDARD_ASSEMBLER = Path(
    "Code/Source/solver/FE/Assembly/StandardAssembler.cpp"
)
DEFAULT_NEWTON_SOLVER = Path(
    "Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_test10_signature_replay_readiness_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the selective Test10 solve-time support/coupling signature "
            "candidate and classify whether current solve-affecting hooks can "
            "replay it without falling back to post-assembly row-list mutation."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument("--standard-assembler", type=Path, default=DEFAULT_STANDARD_ASSEMBLER)
    parser.add_argument("--newton-solver", type=Path, default=DEFAULT_NEWTON_SOLVER)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def evaluate_selector(
    *,
    selected_rows: set[int],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    target_set = set(target_rows)
    covered = sorted(selected_rows & target_set)
    uncovered = sorted(target_set - selected_rows)
    selected_count = len(selected_rows)
    target_count = len(target_rows)
    ratio = selected_count / target_count if target_count else None
    covers_targets = target_count > 0 and len(covered) == target_count
    overbroad = ratio is not None and ratio > max_target_ratio
    if not covers_targets and overbroad:
        finding = "selector_overbroad_and_misses_targets"
    elif not covers_targets:
        finding = "selector_misses_targets"
    elif overbroad:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"
    return {
        "finding": finding,
        "selected_count": selected_count,
        "target_count": target_count,
        "selected_to_target_ratio": ratio,
        "covered_target_count": len(covered),
        "covered_target_global_dofs": covered,
        "uncovered_target_global_dofs": uncovered,
        "covers_targets": covers_targets,
        "selector_overbroad": overbroad,
    }


def exact_local_signature_selected_rows(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
) -> set[int]:
    target_signatures = {
        signature_tuple(rows[row], include_local_indices=True)
        for row in target_rows
        if row in rows
    }
    return {
        row
        for row, stats in rows.items()
        if signature_tuple(stats, include_local_indices=True) in target_signatures
    }


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = summarize_rows(entries)
    selected = exact_local_signature_selected_rows(rows=rows, target_rows=target_rows)
    selector = evaluate_selector(
        selected_rows=selected,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_rows_present_count": sum(1 for row in target_rows if row in rows),
        "exact_local_signature_selector": selector,
        "signature_candidate_global_dofs": sorted(selected),
    }


def source_hook_summary(
    *,
    standard_assembler_text: str,
    newton_solver_text: str,
) -> dict[str, Any]:
    fe_env_tokens = [
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_OPERATOR",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_SOURCE_COMPONENT",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_APPLY_FULL_CELL",
    ]
    fe_policy_tokens = [
        "local_schur_completion",
        "local_edge_balance",
        "local_schur_edge_balance",
    ]
    fe_selector_tokens = [
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_ROW_DOFS",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_SIGNATURE",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SIGNATURE",
    ]
    post_assembly_tokens = [
        "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_BALANCE_GLOBAL_DOFS",
        "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_EXPLICIT_BALANCE_GLOBAL_DOFS",
        "shared_row_schur_explicit_edge_balance",
    ]
    return {
        "fe_topology_env_present": {
            token: token in standard_assembler_text for token in fe_env_tokens
        },
        "fe_topology_policy_modes_present": {
            token: token in standard_assembler_text for token in fe_policy_tokens
        },
        "fe_topology_signature_or_row_selector_present": any(
            token in standard_assembler_text for token in fe_selector_tokens
        ),
        "fe_topology_missing_selector_tokens": [
            token for token in fe_selector_tokens if token not in standard_assembler_text
        ],
        "post_assembly_explicit_row_path_present": all(
            token in newton_solver_text for token in post_assembly_tokens
        ),
        "post_assembly_explicit_row_tokens": {
            token: token in newton_solver_text for token in post_assembly_tokens
        },
    }


def aggregate_finding(
    *,
    cases: list[dict[str, Any]],
    hook_summary: dict[str, Any],
) -> tuple[str, str]:
    case_map = {case["label"]: case for case in cases}
    test10_selector = case_map.get("test10", {}).get(
        "exact_local_signature_selector", {}
    )
    test02_selector = case_map.get("test02", {}).get(
        "exact_local_signature_selector", {}
    )
    test10_ready = test10_selector.get("finding") == "selector_selective"
    test02_overbroad = (
        test02_selector.get("covers_targets") is True
        and test02_selector.get("selector_overbroad") is True
    )
    solve_time_selector_available = hook_summary.get(
        "fe_topology_signature_or_row_selector_present"
    ) is True
    if test10_ready and test02_overbroad and not solve_time_selector_available:
        return (
            "test10_signature_replay_candidate_blocked_by_solve_time_selector_api",
            "candidate_rows_exported_replay_requires_signature_selector_api",
        )
    if test10_ready and solve_time_selector_available:
        return (
            "test10_signature_replay_candidate_ready_for_solve_time_replay",
            "run_targeted_test10_signature_replay",
        )
    return (
        "test10_signature_replay_candidate_not_ready",
        "regenerate_solve_time_signature_evidence",
    )


def build_report(
    *,
    target_map: dict[str, Any],
    log_entries_by_case: dict[str, list[dict[str, Any]]],
    standard_assembler_text: str,
    newton_solver_text: str,
    log_paths_by_case: dict[str, Path | None] | None = None,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    targets = target_rows_by_case(target_map)
    log_paths_by_case = log_paths_by_case or {}
    cases = [
        build_case_report(
            label=label,
            log_path=log_paths_by_case.get(label),
            entries=log_entries_by_case.get(label, []),
            target_rows=targets.get(label, []),
            max_target_ratio=max_target_ratio,
        )
        for label in ("test02", "test10")
    ]
    hook_summary = source_hook_summary(
        standard_assembler_text=standard_assembler_text,
        newton_solver_text=newton_solver_text,
    )
    finding, status = aggregate_finding(cases=cases, hook_summary=hook_summary)
    solve_time_selector_available = hook_summary.get(
        "fe_topology_signature_or_row_selector_present"
    ) is True
    conclusion = (
        "The exact local solve-time support/coupling signature exports a "
        "selective Test10 candidate row set, while the same selector remains "
        "overbroad for Test02. Current FE cut-volume direct PSPG topology "
        "hooks now expose an opt-in row/signature selector, so the Test10 "
        "candidate can be replayed in the solve-time assembly path without "
        "using Newton's post-assembly explicit row-list controls."
        if solve_time_selector_available
        else (
            "The exact local solve-time support/coupling signature exports a "
            "selective Test10 candidate row set, while the same selector remains "
            "overbroad for Test02. Current FE cut-volume direct PSPG topology "
            "hooks expose only local policy modes, not a row/signature selector "
            "or support/coupling aggregation decision. Newton's post-assembly "
            "graph-completion path has explicit row-list controls, but that "
            "route is a diagnostic mutation family already ruled out as a "
            "production formulation rule."
        )
    )
    next_requirement = (
        "Run a targeted Test10 replay with the exported signature candidate "
        "rows through the solve-time direct PSPG topology row filter; do not "
        "treat the post-assembly explicit row-list path as the final "
        "formulation fix."
        if solve_time_selector_available
        else (
            "Add a solve-time direct PSPG support/coupling signature selector "
            "or row-filtered FE topology policy for a targeted Test10 replay; "
            "do not treat the post-assembly explicit row-list path as the final "
            "formulation fix."
        )
    )
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Readiness audit for replaying the selective Test10 exact local "
            "solve-time support/coupling signature without promoting broad "
            "post-assembly graph-completion selectors."
        ),
        "max_target_ratio": max_target_ratio,
        "cases": cases,
        "hook_summary": hook_summary,
        "conclusion": conclusion,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    log_paths = {
        "test02": args.test02_log,
        "test10": args.test10_log,
    }
    report = build_report(
        target_map=load_json(args.target_map_json),
        log_entries_by_case={
            label: read_provenance_log(path) for label, path in log_paths.items()
        },
        standard_assembler_text=read_text(args.standard_assembler),
        newton_solver_text=read_text(args.newton_solver),
        log_paths_by_case=log_paths,
        max_target_ratio=args.max_target_ratio,
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
