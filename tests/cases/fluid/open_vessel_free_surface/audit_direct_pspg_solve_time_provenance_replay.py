#!/usr/bin/env python3
"""Audit solve-time direct PSPG support/coupling provenance replays."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)
DEFAULT_TEST02_LOG = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_replay_abs_only_prune1e5_step382_direct_pspg_solve_time_provenance_20260607_case"
    / "run_direct_pspg_solve_time_provenance.log"
)
DEFAULT_TEST10_LOG = (
    DEFAULT_ARTIFACT_ROOT
    / "test10_replay_cap3_step90_direct_pspg_solve_time_provenance_20260607_case"
    / "run_direct_pspg_solve_time_provenance.log"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_provenance_replay_20260607.json"
)

DIAGNOSTIC_NAME = "cut_volume_direct_pspg_support_coupling_provenance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse short Test02/Test10 direct PSPG support/coupling provenance "
            "replay logs and classify whether simple solve-time PP/PV "
            "selectors are production-ready."
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


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def convert_value(value: str) -> Any:
    if value in {"none", "None"}:
        return None
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_provenance_line(line: str) -> dict[str, Any] | None:
    marker = f"diagnostic={DIAGNOSTIC_NAME}"
    if marker not in line:
        return None
    payload = line[line.index(marker) :]
    record: dict[str, Any] = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        record[key] = convert_value(value)
    return record


def read_provenance_log(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as log:
        for line_number, line in enumerate(log, start=1):
            record = parse_provenance_line(line)
            if record is None:
                continue
            record["line_number"] = line_number
            entries.append(record)
    return entries


def target_rows_by_case(target_map: dict[str, Any]) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if not isinstance(label, str):
            continue
        rows: list[int] = []
        for value in as_list(case.get("direct_pspg_target_global_dofs")):
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                rows.append(value)
        targets[label] = rows
    return targets


def empty_row_stats() -> dict[str, Any]:
    return {
        "pressure_pressure_records": 0,
        "pressure_velocity_records": 0,
        "pressure_pressure_abs_sum": 0.0,
        "pressure_velocity_abs_sum": 0.0,
        "pressure_pressure_edge_count": 0,
        "pressure_pressure_two_hop_completion_count": 0,
        "pressure_pressure_neighbor_pair_count": 0,
        "pressure_pressure_neighbor_connected_pair_count": 0,
        "pressure_pressure_edge_weight_sum": 0.0,
        "pressure_velocity_nonzero_count": 0,
        "min_volume_fraction": None,
        "full_cell_records": 0,
        "cut_cell_records": 0,
        "rule_indices": set(),
        "all_pressure_update_sign_unused": True,
        "all_diagnostic_only": True,
    }


def add_entry_to_row(row: dict[str, Any], entry: dict[str, Any]) -> None:
    block = entry.get("block")
    row["rule_indices"].add(entry.get("rule_index"))
    volume_fraction = entry.get("volume_fraction")
    if isinstance(volume_fraction, (int, float)):
        current = row["min_volume_fraction"]
        row["min_volume_fraction"] = (
            float(volume_fraction)
            if current is None
            else min(float(current), float(volume_fraction))
        )
    if entry.get("full_cell") == 1:
        row["full_cell_records"] += 1
    else:
        row["cut_cell_records"] += 1
    row["all_pressure_update_sign_unused"] = (
        row["all_pressure_update_sign_unused"]
        and entry.get("pressure_update_sign_used") == 0
    )
    row["all_diagnostic_only"] = (
        row["all_diagnostic_only"] and entry.get("diagnostic_only") == 1
    )

    if block == "pressure_pressure":
        row["pressure_pressure_records"] += 1
        row["pressure_pressure_abs_sum"] += float(entry.get("row_abs_sum") or 0.0)
        row["pressure_pressure_edge_count"] += int(
            entry.get("source_edge_count") or 0
        )
        row["pressure_pressure_two_hop_completion_count"] += int(
            entry.get("two_hop_completion_count") or 0
        )
        row["pressure_pressure_neighbor_pair_count"] += int(
            entry.get("neighbor_pair_count") or 0
        )
        row["pressure_pressure_neighbor_connected_pair_count"] += int(
            entry.get("neighbor_connected_pair_count") or 0
        )
        row["pressure_pressure_edge_weight_sum"] += float(
            entry.get("source_edge_weight_sum") or 0.0
        )
    elif block == "pressure_velocity":
        row["pressure_velocity_records"] += 1
        row["pressure_velocity_abs_sum"] += float(entry.get("row_abs_sum") or 0.0)
        row["pressure_velocity_nonzero_count"] += int(
            entry.get("nonzero_count") or 0
        )


def summarize_rows(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row_dof = entry.get("row_dof")
        if not isinstance(row_dof, int):
            continue
        row = rows.setdefault(row_dof, empty_row_stats())
        add_entry_to_row(row, entry)
    for row in rows.values():
        row["rule_count"] = len(row["rule_indices"])
        row.pop("rule_indices", None)
        pp_abs = row["pressure_pressure_abs_sum"]
        pv_abs = row["pressure_velocity_abs_sum"]
        row["pressure_velocity_to_pressure_pressure_abs_ratio"] = (
            pv_abs / pp_abs if pp_abs > 0.0 else None
        )
    return rows


def compact_target_row(row_dof: int, stats: dict[str, Any] | None) -> dict[str, Any]:
    if stats is None:
        return {
            "row_dof": row_dof,
            "present": False,
            "has_pressure_pressure": False,
            "has_pressure_velocity_record": False,
        }
    return {
        "row_dof": row_dof,
        "present": True,
        "has_pressure_pressure": stats["pressure_pressure_records"] > 0,
        "has_pressure_velocity_record": stats["pressure_velocity_records"] > 0,
        "pressure_pressure_records": stats["pressure_pressure_records"],
        "pressure_velocity_records": stats["pressure_velocity_records"],
        "pressure_pressure_abs_sum": stats["pressure_pressure_abs_sum"],
        "pressure_velocity_abs_sum": stats["pressure_velocity_abs_sum"],
        "pressure_velocity_nonzero_count": stats["pressure_velocity_nonzero_count"],
        "pressure_velocity_to_pressure_pressure_abs_ratio": stats[
            "pressure_velocity_to_pressure_pressure_abs_ratio"
        ],
        "pressure_pressure_edge_count": stats["pressure_pressure_edge_count"],
        "pressure_pressure_two_hop_completion_count": stats[
            "pressure_pressure_two_hop_completion_count"
        ],
        "min_volume_fraction": stats["min_volume_fraction"],
        "full_cell_records": stats["full_cell_records"],
        "cut_cell_records": stats["cut_cell_records"],
        "rule_count": stats["rule_count"],
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
    target_count = len(target_rows)
    selected_count = len(selected_rows)
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


def ratio_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    ratios = {
        row: stats["pressure_velocity_to_pressure_pressure_abs_ratio"]
        for row, stats in rows.items()
        if stats["pressure_velocity_to_pressure_pressure_abs_ratio"] is not None
    }
    target_ratios = [
        ratios[row] for row in target_rows if row in ratios and ratios[row] is not None
    ]
    positive_target_ratios = [ratio for ratio in target_ratios if ratio > 0.0]
    selectors: list[dict[str, Any]] = []
    if target_ratios:
        min_target_ratio = min(target_ratios)
        max_target_ratio_value = max(target_ratios)
        selectors.append(
            evaluate_selector(
                key="pv_to_pp_ratio_at_or_above_min_target",
                description=(
                    "Rows whose solve-time pressure-velocity to pressure-pressure "
                    "absolute coupling ratio is at least the minimum audited "
                    "target ratio."
                ),
                selected_rows={
                    row for row, ratio in ratios.items() if ratio >= min_target_ratio
                },
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
            )
            | {"threshold": min_target_ratio}
        )
        selectors.append(
            evaluate_selector(
                key="pv_to_pp_ratio_at_or_above_max_target",
                description=(
                    "Rows whose solve-time pressure-velocity to pressure-pressure "
                    "absolute coupling ratio is at least the maximum audited "
                    "target ratio."
                ),
                selected_rows={
                    row for row, ratio in ratios.items() if ratio >= max_target_ratio_value
                },
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
            )
            | {"threshold": max_target_ratio_value}
        )
    if positive_target_ratios:
        positive_min = min(positive_target_ratios)
        selectors.append(
            evaluate_selector(
                key="pv_to_pp_ratio_at_or_above_positive_min_target",
                description=(
                    "Rows whose solve-time pressure-velocity to pressure-pressure "
                    "absolute coupling ratio is at least the minimum positive "
                    "audited target ratio."
                ),
                selected_rows={
                    row for row, ratio in ratios.items() if ratio >= positive_min
                },
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
            )
            | {"threshold": positive_min}
        )
    selectors.append(
        evaluate_selector(
            key="zero_pressure_velocity_coupling",
            description=(
                "Rows with emitted pressure-velocity provenance but zero "
                "pressure-velocity row action."
            ),
            selected_rows={
                row
                for row, stats in rows.items()
                if stats["pressure_velocity_records"] > 0
                and stats["pressure_velocity_abs_sum"] == 0.0
            },
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
        )
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
        return "solve_time_provenance_missing_target_rows"
    if any(not item.get("all_pressure_update_sign_unused") for item in target_summaries):
        return "solve_time_provenance_uses_pressure_update_sign"
    if any(selector["finding"] == "selector_selective" for selector in selectors):
        return "solve_time_provenance_simple_selector_ready"
    zero_targets = [
        item
        for item in target_summaries
        if item.get("pressure_velocity_abs_sum") == 0.0
    ]
    nonzero_targets = [
        item
        for item in target_summaries
        if (item.get("pressure_velocity_abs_sum") or 0.0) > 0.0
    ]
    if zero_targets and nonzero_targets:
        return "solve_time_provenance_target_family_splits_zero_and_nonzero_coupling"
    return "solve_time_provenance_covers_targets_but_simple_pp_pv_selectors_fail"


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
    selectors = ratio_selectors(
        rows=rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    present_targets = [item for item in target_summaries if item.get("present")]
    target_ratios = [
        item.get("pressure_velocity_to_pressure_pressure_abs_ratio")
        for item in present_targets
        if isinstance(
            item.get("pressure_velocity_to_pressure_pressure_abs_ratio"),
            (int, float),
        )
    ]
    zero_pv_targets = [
        item["row_dof"]
        for item in present_targets
        if item.get("pressure_velocity_abs_sum") == 0.0
    ]
    max_target_ratio_value = max(target_ratios) if target_ratios else None
    max_target_rows = [
        item["row_dof"]
        for item in present_targets
        if item.get("pressure_velocity_to_pressure_pressure_abs_ratio")
        == max_target_ratio_value
    ]
    all_rows_sign_unused = all(
        stats["all_pressure_update_sign_unused"] for stats in rows.values()
    )
    all_rows_diagnostic_only = all(
        stats["all_diagnostic_only"] for stats in rows.values()
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
        "all_target_rows_have_pressure_pressure": all(
            item.get("has_pressure_pressure") for item in target_summaries
        ),
        "all_target_rows_have_pressure_velocity_record": all(
            item.get("has_pressure_velocity_record") for item in target_summaries
        ),
        "all_rows_pressure_update_sign_unused": all_rows_sign_unused,
        "all_rows_diagnostic_only": all_rows_diagnostic_only,
        "target_pressure_velocity_to_pressure_pressure_ratio_min": (
            min(target_ratios) if target_ratios else None
        ),
        "target_pressure_velocity_to_pressure_pressure_ratio_max": (
            max_target_ratio_value
        ),
        "max_target_ratio_rows": sorted(max_target_rows),
        "zero_pressure_velocity_target_global_dofs": zero_pv_targets,
        "target_rows": target_summaries,
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_support_coupling_replay_missing_evidence",
            "regenerate_short_replay_logs",
        )
    if any(
        case.get("finding") == "solve_time_provenance_uses_pressure_update_sign"
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_support_coupling_replay_invalid_update_dependent",
            "diagnostic_invalid",
        )
    if cases and all(
        case.get("finding") == "solve_time_provenance_simple_selector_ready"
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_support_coupling_replay_selector_ready",
            "candidate_ready_for_targeted_formulation_replay",
        )
    return (
        "solve_time_direct_pspg_support_coupling_replay_rules_out_simple_pp_pv_gate",
        "replay_evidence_supports_coupling_split_no_selector",
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
            "pressure-gradient support/coupling provenance replay audit."
        ),
        "diagnostic": DIAGNOSTIC_NAME,
        "diagnostic_env": {
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC": "1",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR": "equations",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT": (
                "navier_stokes_vms_pspg_pressure_gradient"
            ),
        },
        "max_target_ratio": max_target_ratio,
        "cases": cases,
        "conclusion": (
            "The solve-time diagnostic now covers all audited direct PSPG target "
            "rows without pressure-update signs and remains diagnostic-only. "
            "However, simple pressure-velocity to pressure-pressure coupling "
            "ratio gates are not production-ready: Test02's isolated row 10676 "
            "is the high-ratio branch, while the remaining Test02 targets need "
            "a much lower threshold that selects broad candidates; Test10 splits "
            "between zero-coupling targets and nonzero-coupled boundary rows. "
            "This rules out a scalar PP/PV coupling gate and keeps the next fix "
            "target inside a richer direct PSPG support/coupling topology rule."
        ),
        "next_requirement": (
            "Derive a formulation-side topology/coupling rule from this "
            "solve-time provenance that handles the Test02 isolated branch and "
            "the Test10 zero/nonzero coupling split without reusing pressure "
            "update signs or broad post-assembly graph mutation."
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
