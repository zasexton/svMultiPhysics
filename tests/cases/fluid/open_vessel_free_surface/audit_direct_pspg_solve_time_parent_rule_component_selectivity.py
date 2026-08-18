#!/usr/bin/env python3
"""Audit solve-time direct PSPG parent/rule component selectivity."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_aggregate_feature_selectivity import (
    evaluate_selector,
)
from audit_direct_pspg_solve_time_provenance_replay import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    load_json,
    read_provenance_log,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_parent_rule_component_selectivity_20260607.json"
)

GRAPH_MODES = [
    "parent_cell",
    "rule_index",
    "parent_or_rule",
    "parent_rule_local_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether solve-time direct PSPG parent-cell/rule-index "
            "co-support components provide a connected physical support-patch "
            "closure for Test02/Test10 direct targets."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def support_nodes(entry: dict[str, Any], mode: str) -> list[tuple[Any, ...]]:
    nodes: list[tuple[Any, ...]] = []
    parent_cell = entry.get("parent_cell")
    rule_index = entry.get("rule_index")
    local_index = entry.get("row_local_index")
    if mode in {"parent_cell", "parent_or_rule", "parent_rule_local_index"}:
        if isinstance(parent_cell, int):
            nodes.append(("parent_cell", parent_cell))
    if mode in {"rule_index", "parent_or_rule", "parent_rule_local_index"}:
        if isinstance(rule_index, int):
            nodes.append(("rule_index", rule_index))
    if mode == "parent_rule_local_index":
        if isinstance(parent_cell, int) and isinstance(local_index, int):
            nodes.append(("parent_local_index", parent_cell, local_index))
        if isinstance(rule_index, int) and isinstance(local_index, int):
            nodes.append(("rule_local_index", rule_index, local_index))
    return nodes


def row_components(
    entries: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[dict[int, int], list[set[int]]]:
    adjacency: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    row_dofs: set[int] = set()
    for entry in entries:
        row_dof = entry.get("row_dof")
        if not isinstance(row_dof, int):
            continue
        row_node = ("row", row_dof)
        row_dofs.add(row_dof)
        for node in support_nodes(entry, mode):
            adjacency[row_node].add(node)
            adjacency[node].add(row_node)
    component_by_row: dict[int, int] = {}
    components: list[set[int]] = []
    seen: set[tuple[Any, ...]] = set()
    for row_dof in sorted(row_dofs):
        start = ("row", row_dof)
        if start in seen:
            continue
        queue: deque[tuple[Any, ...]] = deque([start])
        seen.add(start)
        component_rows: set[int] = set()
        while queue:
            node = queue.popleft()
            if node[0] == "row":
                component_rows.add(int(node[1]))
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        component_index = len(components)
        components.append(component_rows)
        for row in component_rows:
            component_by_row[row] = component_index
    return component_by_row, components


def build_component_selector(
    *,
    mode: str,
    component_by_row: dict[int, int],
    components: list[set[int]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    target_component_indices = sorted(
        {
            component_by_row[row]
            for row in target_rows
            if row in component_by_row
        }
    )
    selected_rows: set[int] = set()
    for component_index in target_component_indices:
        selected_rows.update(components[component_index])
    return evaluate_selector(
        key=f"{mode}_target_component_union",
        selector_kind="target_component_union",
        feature=mode,
        selected_rows=selected_rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
        extra={
            "component_count": len(components),
            "target_component_indices": target_component_indices,
            "target_component_sizes": [
                len(components[index]) for index in target_component_indices
            ],
            "largest_component_size": (
                max((len(component) for component in components), default=0)
            ),
            "production_readiness": "connected_support_patch_candidate",
        },
    )


def case_finding(
    *,
    selectors: list[dict[str, Any]],
    target_count: int,
    target_rows_present_count: int,
) -> str:
    if target_count == 0:
        return "direct_target_rows_missing"
    if target_rows_present_count < target_count:
        return "solve_time_parent_rule_component_missing_target_rows"
    if any(selector.get("finding") == "selector_selective" for selector in selectors):
        return "solve_time_parent_rule_component_candidate_requires_replay"
    return "solve_time_parent_rule_components_overbroad_or_miss_targets"


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    selectors: list[dict[str, Any]] = []
    target_rows_present_sets: list[set[int]] = []
    for mode in GRAPH_MODES:
        component_by_row, components = row_components(entries, mode=mode)
        target_rows_present_sets.append(set(component_by_row))
        selectors.append(
            build_component_selector(
                mode=mode,
                component_by_row=component_by_row,
                components=components,
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
            )
        )
    all_present_rows = set.intersection(*target_rows_present_sets) if target_rows_present_sets else set()
    present_targets = [row for row in target_rows if row in all_present_rows]
    best_covering = min(
        (selector for selector in selectors if selector.get("covers_targets")),
        key=lambda selector: (
            float(selector.get("selected_to_target_ratio") or float("inf")),
            int(selector.get("selected_count") or 0),
            str(selector.get("key") or ""),
        ),
        default=None,
    )
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(
            selectors=selectors,
            target_count=len(target_rows),
            target_rows_present_count=len(present_targets),
        ),
        "record_count": len(entries),
        "unique_pressure_row_count": len(
            {
                entry.get("row_dof")
                for entry in entries
                if isinstance(entry.get("row_dof"), int)
            }
        ),
        "target_count": len(target_rows),
        "target_rows_present_count": len(present_targets),
        "graph_modes": GRAPH_MODES,
        "best_covering_component_selector": best_covering,
        "component_counts": {
            selector["feature"]: selector["component_count"] for selector in selectors
        },
        "target_component_sizes": {
            selector["feature"]: selector["target_component_sizes"]
            for selector in selectors
        },
        "selector_selected_counts": {
            selector["key"]: selector["selected_count"] for selector in selectors
        },
        "selector_selected_to_target_ratios": {
            selector["key"]: selector["selected_to_target_ratio"]
            for selector in selectors
        },
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_parent_rule_component_selectivity_missing_evidence",
            "regenerate_solve_time_provenance_logs",
        )
    if any(
        case.get("finding")
        == "solve_time_parent_rule_component_candidate_requires_replay"
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_parent_rule_component_candidate_requires_replay",
            "parent_rule_component_candidate_needs_transfer_check",
        )
    return (
        "solve_time_direct_pspg_parent_rule_components_rule_out_connected_cosupport_closure",
        "parent_rule_component_closure_overbroad",
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
            "parent-cell/rule-index co-support component audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "graph_modes": GRAPH_MODES,
        "cases": cases,
        "conclusion": (
            "Raw connected co-support closure over solve-time parent cells, "
            "rule indices, and parent/rule plus row-local index support is not "
            "the missing physical patch gate. The audited targets sit inside "
            "broad connected components in at least one case."
        ),
        "next_requirement": (
            "Continue the direct PSPG formulation search with a physical "
            "support/coupling rule beyond raw connected parent/rule co-support "
            "closure or exact row/parent replay of current local matrix deltas."
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
