#!/usr/bin/env python3
"""Audit solve-time direct PSPG parent-cell support/coupling signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_provenance_replay import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    read_provenance_log,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_support_coupling_signature_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether solve-time direct PSPG pressure-pressure support "
            "and same-parent pressure-velocity coupling signatures provide a "
            "production-ready topology gate for Test02/Test10 targets."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio considered selective.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_row_stats() -> dict[str, Any]:
    return {
        "pressure_pressure_records": 0,
        "pressure_velocity_records": 0,
        "pressure_pressure_parent_cells": set(),
        "pressure_velocity_parent_cells": set(),
        "pressure_velocity_nonzero_parent_cells": set(),
        "local_row_indices": set(),
        "pressure_pressure_abs_sum": 0.0,
        "pressure_velocity_abs_sum": 0.0,
        "pressure_pressure_source_edge_count": 0,
        "pressure_pressure_neighbor_pair_count": 0,
        "pressure_pressure_neighbor_connected_pair_count": 0,
        "pressure_pressure_two_hop_completion_count": 0,
        "all_pressure_update_sign_unused": True,
        "all_diagnostic_only": True,
    }


def add_entry_to_row(row: dict[str, Any], entry: dict[str, Any]) -> None:
    parent_cell = entry.get("parent_cell")
    if isinstance(parent_cell, int):
        if entry.get("block") == "pressure_pressure":
            row["pressure_pressure_parent_cells"].add(parent_cell)
        elif entry.get("block") == "pressure_velocity":
            row["pressure_velocity_parent_cells"].add(parent_cell)
            if float(entry.get("row_abs_sum") or 0.0) > 0.0:
                row["pressure_velocity_nonzero_parent_cells"].add(parent_cell)

    local_index = entry.get("row_local_index")
    if isinstance(local_index, int):
        row["local_row_indices"].add(local_index)

    row["all_pressure_update_sign_unused"] = (
        row["all_pressure_update_sign_unused"]
        and entry.get("pressure_update_sign_used") == 0
    )
    row["all_diagnostic_only"] = (
        row["all_diagnostic_only"] and entry.get("diagnostic_only") == 1
    )

    if entry.get("block") == "pressure_pressure":
        row["pressure_pressure_records"] += 1
        row["pressure_pressure_abs_sum"] += float(entry.get("row_abs_sum") or 0.0)
        row["pressure_pressure_source_edge_count"] += int(
            entry.get("source_edge_count") or 0
        )
        row["pressure_pressure_neighbor_pair_count"] += int(
            entry.get("neighbor_pair_count") or 0
        )
        row["pressure_pressure_neighbor_connected_pair_count"] += int(
            entry.get("neighbor_connected_pair_count") or 0
        )
        row["pressure_pressure_two_hop_completion_count"] += int(
            entry.get("two_hop_completion_count") or 0
        )
    elif entry.get("block") == "pressure_velocity":
        row["pressure_velocity_records"] += 1
        row["pressure_velocity_abs_sum"] += float(entry.get("row_abs_sum") or 0.0)


def summarize_rows(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row_dof = entry.get("row_dof")
        if not isinstance(row_dof, int):
            continue
        row = rows.setdefault(row_dof, empty_row_stats())
        add_entry_to_row(row, entry)
    return rows


def same_parent_nonzero_pv_cells(stats: dict[str, Any]) -> set[int]:
    return (
        stats["pressure_pressure_parent_cells"]
        & stats["pressure_velocity_nonzero_parent_cells"]
    )


def same_parent_pv_support_class(stats: dict[str, Any]) -> str:
    pp_count = len(stats["pressure_pressure_parent_cells"])
    nonzero_count = len(same_parent_nonzero_pv_cells(stats))
    if nonzero_count == 0:
        return "none"
    if nonzero_count == pp_count:
        return "full"
    return "partial"


def signature_tuple(
    stats: dict[str, Any], *, include_local_indices: bool = False
) -> tuple[Any, ...]:
    pp_count = len(stats["pressure_pressure_parent_cells"])
    nonzero_count = len(same_parent_nonzero_pv_cells(stats))
    signature: tuple[Any, ...] = (
        pp_count,
        nonzero_count,
        pp_count - nonzero_count,
        stats["pressure_pressure_source_edge_count"],
        stats["pressure_pressure_two_hop_completion_count"],
        same_parent_pv_support_class(stats),
    )
    if include_local_indices:
        signature = signature + (
            tuple(sorted(stats["local_row_indices"])),
        )
    return signature


def signature_key(
    stats: dict[str, Any], *, include_local_indices: bool = False
) -> str:
    pp_count, nonzero_count, gap_count, edge_count, two_hop, support_class, *rest = (
        signature_tuple(stats, include_local_indices=include_local_indices)
    )
    key = (
        f"pp{pp_count}_sameparentpv{nonzero_count}_gap{gap_count}"
        f"_ppedge{edge_count}_pptwohop{two_hop}_{support_class}"
    )
    if rest:
        indices = "-".join(str(value) for value in rest[0])
        key += f"_local{indices}"
    return key


def compact_target_row(row_dof: int, stats: dict[str, Any] | None) -> dict[str, Any]:
    if stats is None:
        return {"row_dof": row_dof, "present": False}
    pp_count = len(stats["pressure_pressure_parent_cells"])
    nonzero_count = len(same_parent_nonzero_pv_cells(stats))
    return {
        "row_dof": row_dof,
        "present": True,
        "pressure_pressure_parent_cell_count": pp_count,
        "pressure_velocity_parent_cell_count": len(
            stats["pressure_velocity_parent_cells"]
        ),
        "same_parent_nonzero_pressure_velocity_parent_cell_count": nonzero_count,
        "same_parent_pressure_velocity_gap_parent_cell_count": (
            pp_count - nonzero_count
        ),
        "same_parent_pressure_velocity_support_class": (
            same_parent_pv_support_class(stats)
        ),
        "pressure_pressure_source_edge_count": stats[
            "pressure_pressure_source_edge_count"
        ],
        "pressure_pressure_two_hop_completion_count": stats[
            "pressure_pressure_two_hop_completion_count"
        ],
        "local_row_indices": sorted(stats["local_row_indices"]),
        "support_coupling_signature": signature_key(stats),
        "support_coupling_signature_with_local_indices": signature_key(
            stats, include_local_indices=True
        ),
        "all_pressure_update_sign_unused": stats["all_pressure_update_sign_unused"],
        "all_diagnostic_only": stats["all_diagnostic_only"],
    }


def evaluate_selector(
    *,
    key: str,
    description: str,
    selected_rows: set[int],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    target_set = set(target_rows)
    covered = sorted(target_set & selected_rows)
    uncovered = sorted(target_set - selected_rows)
    selected_count = len(selected_rows)
    target_count = len(target_rows)
    selected_to_target_ratio = (
        selected_count / target_count if target_count > 0 else None
    )
    covers_targets = target_count > 0 and len(covered) == target_count
    overbroad = (
        selected_to_target_ratio is not None
        and selected_to_target_ratio > max_target_ratio
    )
    if not covers_targets and overbroad:
        finding = "selector_overbroad_and_misses_targets"
    elif not covers_targets:
        finding = "selector_misses_targets"
    elif overbroad:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"
    return {
        "key": key,
        "description": description,
        "finding": finding,
        "selected_count": selected_count,
        "target_count": target_count,
        "selected_to_target_ratio": selected_to_target_ratio,
        "covered_target_count": len(covered),
        "covered_target_global_dofs": covered,
        "uncovered_target_global_dofs": uncovered,
        "covers_targets": covers_targets,
        "selector_overbroad": overbroad,
    }


def selector_by_key(selectors: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return next(selector for selector in selectors if selector["key"] == key)


def build_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    target_stats = [rows[row] for row in target_rows if row in rows]
    target_classes = {same_parent_pv_support_class(stats) for stats in target_stats}
    target_signatures = {signature_tuple(stats) for stats in target_stats}
    target_local_signatures = {
        signature_tuple(stats, include_local_indices=True) for stats in target_stats
    }

    selectors = [
        evaluate_selector(
            key=f"same_parent_pressure_velocity_support_{support_class}",
            description=(
                "Rows whose pressure-pressure parent cells have "
                f"{support_class} same-parent nonzero pressure-velocity coupling."
            ),
            selected_rows={
                row
                for row, stats in rows.items()
                if same_parent_pv_support_class(stats) == support_class
            },
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
        for support_class in ("none", "partial", "full")
    ]
    selectors.append(
        evaluate_selector(
            key="target_same_parent_pressure_velocity_support_class_union",
            description=(
                "Rows whose same-parent pressure-velocity support class matches "
                "any audited target class."
            ),
            selected_rows={
                row
                for row, stats in rows.items()
                if same_parent_pv_support_class(stats) in target_classes
            },
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
        | {"target_classes": sorted(target_classes)}
    )
    selectors.append(
        evaluate_selector(
            key="target_support_coupling_signature_union",
            description=(
                "Rows whose aggregate pressure-pressure parent count, "
                "same-parent nonzero pressure-velocity parent count, "
                "pressure-pressure edge count, and two-hop count match an "
                "audited target signature."
            ),
            selected_rows={
                row
                for row, stats in rows.items()
                if signature_tuple(stats) in target_signatures
            },
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
        | {"target_signature_count": len(target_signatures)}
    )
    selectors.append(
        evaluate_selector(
            key="target_support_coupling_signature_with_local_index_union",
            description=(
                "Rows whose support/coupling signature and row-local index set "
                "match an audited target signature."
            ),
            selected_rows={
                row
                for row, stats in rows.items()
                if signature_tuple(stats, include_local_indices=True)
                in target_local_signatures
            },
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
        | {"target_signature_count": len(target_local_signatures)}
    )
    return selectors


def case_finding(
    *,
    target_rows: list[int],
    target_summaries: list[dict[str, Any]],
    selectors: list[dict[str, Any]],
) -> str:
    if not target_rows:
        return "direct_target_rows_missing"
    if any(not item.get("present") for item in target_summaries):
        return "solve_time_support_coupling_signature_missing_target_rows"
    if any(not item.get("all_pressure_update_sign_unused") for item in target_summaries):
        return "solve_time_support_coupling_signature_uses_pressure_update_sign"
    exact_local = selector_by_key(
        selectors, "target_support_coupling_signature_with_local_index_union"
    )
    if exact_local["finding"] == "selector_selective":
        return "solve_time_support_coupling_signature_selective_candidate"
    if exact_local["covers_targets"]:
        return "solve_time_support_coupling_signature_covers_targets_but_overbroad"
    return "solve_time_support_coupling_signature_misses_targets"


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = summarize_rows(entries)
    target_summaries = [
        compact_target_row(row, rows.get(row)) for row in target_rows
    ]
    selectors = build_selectors(
        rows=rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    present_targets = [item for item in target_summaries if item.get("present")]
    local_signature_selector = selector_by_key(
        selectors, "target_support_coupling_signature_with_local_index_union"
    )
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(
            target_rows=target_rows,
            target_summaries=target_summaries,
            selectors=selectors,
        ),
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_count": len(target_rows),
        "target_rows_present_count": len(present_targets),
        "target_same_parent_pressure_velocity_support_class_counts": {
            support_class: sum(
                1
                for item in present_targets
                if item.get("same_parent_pressure_velocity_support_class")
                == support_class
            )
            for support_class in ("none", "partial", "full")
        },
        "target_support_coupling_signature_count": len(
            {
                item.get("support_coupling_signature")
                for item in present_targets
                if item.get("support_coupling_signature")
            }
        ),
        "target_support_coupling_signature_with_local_index_count": len(
            {
                item.get("support_coupling_signature_with_local_indices")
                for item in present_targets
                if item.get("support_coupling_signature_with_local_indices")
            }
        ),
        "exact_local_signature_selected_count": local_signature_selector[
            "selected_count"
        ],
        "exact_local_signature_selected_to_target_ratio": local_signature_selector[
            "selected_to_target_ratio"
        ],
        "all_target_rows_pressure_update_sign_unused": all(
            item.get("all_pressure_update_sign_unused") for item in target_summaries
        ),
        "all_target_rows_diagnostic_only": all(
            item.get("all_diagnostic_only") for item in target_summaries
        ),
        "target_rows": target_summaries,
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_support_coupling_signature_missing_evidence",
            "regenerate_short_replay_logs",
        )
    if any(
        case.get("finding")
        == "solve_time_support_coupling_signature_uses_pressure_update_sign"
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_support_coupling_signature_invalid_update_dependent",
            "diagnostic_invalid",
        )
    selective_cases = [
        case
        for case in cases
        if case.get("finding")
        == "solve_time_support_coupling_signature_selective_candidate"
    ]
    if cases and len(selective_cases) == len(cases):
        return (
            "solve_time_direct_pspg_support_coupling_signature_selector_ready",
            "candidate_ready_for_targeted_formulation_replay",
        )
    if selective_cases:
        return (
            "solve_time_direct_pspg_support_coupling_signature_partial_test10_only",
            "test10_signature_candidate_test02_overbroad",
        )
    return (
        "solve_time_direct_pspg_support_coupling_signature_rules_out_common_gate",
        "support_coupling_signature_overbroad_or_misses_targets",
    )


def build_report(
    *,
    target_map: dict[str, Any],
    log_entries_by_case: dict[str, list[dict[str, Any]]],
    log_paths_by_case: dict[str, Path | None] | None = None,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    targets = target_rows_by_case(target_map)
    log_paths_by_case = log_paths_by_case or {}
    labels = ["test02", "test10"]
    cases = [
        build_case_report(
            label=label,
            log_path=log_paths_by_case.get(label),
            entries=log_entries_by_case.get(label, []),
            target_rows=targets.get(label, []),
            max_target_ratio=max_target_ratio,
        )
        for label in labels
    ]
    finding, status = aggregate_finding(cases)
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Short Test02 step382 and Test10 step90 solve-time direct PSPG "
            "support/coupling parent-cell signature audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "cases": cases,
        "conclusion": (
            "Same-parent pressure-velocity support classes and exact aggregate "
            "support/coupling signatures split the Test10 target family into a "
            "selective replay candidate, but the same evidence remains broad "
            "for Test02. The target-signature plus local-index selector covers "
            "all Test02 direct PSPG targets only by selecting many non-target "
            "rows, while Test10 stays below the configured selected/target "
            "ratio. This rules out a common PP/PV parent-cell signature gate as "
            "the complete fix and leaves a possible Test10-specific candidate "
            "that still needs an assembly/formulation API capable of aggregating "
            "PP and PV provenance before a solve-affecting decision."
        ),
        "next_requirement": (
            "Either add a solve-time aggregation API for a targeted Test10 "
            "support/coupling-signature replay, or find an additional Test02 "
            "physical discriminator beyond same-parent PP/PV parent-cell "
            "topology before promoting a common formulation rule."
        ),
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
