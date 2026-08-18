#!/usr/bin/env python3
"""Audit assembly-time cut-volume row provenance for direct PSPG candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any, Callable


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_GLOBAL_EMISSION = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_global_candidate_emission_20260606.json"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)
DEFAULT_OPERATOR = "equations_diagnostic_ns_vms_pspg_pressure_gradient"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact assembly-time generated cut-volume rule provenance "
            "from the direct PSPG pressure-gradient diagnostic operator against "
            "the audited Test02/Test10 target rows."
        )
    )
    parser.add_argument(
        "--global-emission-json",
        type=Path,
        default=DEFAULT_GLOBAL_EMISSION,
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--log",
        action="append",
        type=str,
        default=[],
        help="Case-labelled log path as label=/path/to/run.log.",
    )
    parser.add_argument("--candidate-key", default="preferred_candidate_global_dofs")
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--test-field", default="pressure")
    parser.add_argument("--trial-field", default="pressure")
    parser.add_argument("--low-volume-fraction", type=float, default=0.10)
    parser.add_argument("--very-low-volume-fraction", type=float, default=1.0e-3)
    parser.add_argument("--low-parent-cell-count", type=int, default=2)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio still considered selective.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [item for item in as_list(value) if isinstance(item, int)]


def parse_dof_list(value: str) -> list[int | str]:
    if value in {"", "none"}:
        return []
    parsed: list[int | str] = []
    for token in value.split("|"):
        if token == "...":
            parsed.append(token)
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            parsed.append(token)
    return parsed


def parse_scalar(value: str) -> Any:
    if value in {"", "none"}:
        return [] if value == "none" else value
    if "|" in value:
        return parse_dof_list(value)
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in shlex.split(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = parse_scalar(value)
    return result


def parse_log_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(report.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def target_case_map(target_map: dict[str, Any]) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            targets[label] = int_list(case.get("direct_pspg_target_global_dofs"))
    return targets


def default_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, case in emission_cases.items():
        path = case.get("path")
        if isinstance(path, str) and path:
            paths[label] = Path(path)
    for value in explicit_logs:
        label, path = parse_log_arg(value)
        paths[label] = path
    return paths


def matching_entry(
    entry: dict[str, Any],
    *,
    operator: str,
    test_field: str,
    trial_field: str,
) -> bool:
    entry_test = entry.get("test")
    entry_trial = entry.get("trial")
    return (
        entry.get("op") == operator
        and isinstance(entry_test, str)
        and isinstance(entry_trial, str)
        and entry_test.lower() == test_field.lower()
        and entry_trial.lower() == trial_field.lower()
    )


def latest_provenance_batch(
    log_path: Path,
    *,
    operator: str,
    test_field: str,
    trial_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = {
        "path": str(log_path),
        "exists": log_path.exists(),
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
    }
    if not log_path.exists():
        evidence["status"] = "log_missing"
        return [], evidence

    current: list[dict[str, Any]] = []
    batches: list[list[dict[str, Any]]] = []
    summary_count = 0
    entry_count = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "diagnostic=cut_volume_row_provenance_summary" in line:
            entry = parse_key_values(line)
            if matching_entry(
                entry,
                operator=operator,
                test_field=test_field,
                trial_field=trial_field,
            ):
                summary_count += 1
                if current:
                    batches.append(current)
                    current = []
            continue
        if "diagnostic=cut_volume_row_provenance" not in line:
            continue
        entry = parse_key_values(line)
        if not matching_entry(
            entry,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        ):
            continue
        entry_count += 1
        current.append(entry)
    if current:
        batches.append(current)

    evidence["entry_count"] = entry_count
    evidence["summary_count"] = summary_count
    evidence["batch_count"] = len(batches)
    if not batches:
        evidence["status"] = "provenance_entries_missing"
        return [], evidence
    evidence["status"] = "ok"
    evidence["latest_batch_entry_count"] = len(batches[-1])
    return batches[-1], evidence


def row_profiles_from_entries(
    entries: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    seen: dict[int, set[tuple[Any, ...]]] = {}
    for entry in entries:
        row_dofs = int_list(entry.get("row_dofs"))
        if not row_dofs:
            continue
        rule_key = (
            entry.get("rule_index"),
            entry.get("parent_cell"),
            entry.get("full_cell"),
            entry.get("volume_fraction"),
            entry.get("source_revision"),
            entry.get("cut_topology_revision"),
            entry.get("quadrature_policy_key"),
        )
        for row in row_dofs:
            row_seen = seen.setdefault(row, set())
            if rule_key in row_seen:
                continue
            row_seen.add(rule_key)
            profile = profiles.setdefault(
                row,
                {
                    "global_dof": row,
                    "rule_count": 0,
                    "partial_cut_rule_count": 0,
                    "full_cell_rule_count": 0,
                    "parent_cells": set(),
                    "min_volume_fraction": None,
                    "max_volume_fraction": None,
                    "total_measure": 0.0,
                    "max_quadrature_points": 0,
                },
            )
            profile["rule_count"] += 1
            full_cell = entry.get("full_cell") == 1
            if full_cell:
                profile["full_cell_rule_count"] += 1
            else:
                profile["partial_cut_rule_count"] += 1
            parent_cell = entry.get("parent_cell")
            if isinstance(parent_cell, int):
                profile["parent_cells"].add(parent_cell)
            fraction = entry.get("volume_fraction")
            if isinstance(fraction, (int, float)):
                current_min = profile["min_volume_fraction"]
                current_max = profile["max_volume_fraction"]
                profile["min_volume_fraction"] = (
                    float(fraction)
                    if current_min is None
                    else min(float(current_min), float(fraction))
                )
                profile["max_volume_fraction"] = (
                    float(fraction)
                    if current_max is None
                    else max(float(current_max), float(fraction))
                )
            measure = entry.get("measure")
            if isinstance(measure, (int, float)):
                profile["total_measure"] += float(measure)
            qpts = entry.get("quadrature_points")
            if isinstance(qpts, int):
                profile["max_quadrature_points"] = max(
                    profile["max_quadrature_points"],
                    qpts,
                )

    normalized: dict[int, dict[str, Any]] = {}
    for row, profile in profiles.items():
        parent_cells = sorted(profile.pop("parent_cells"))
        profile["parent_cells"] = parent_cells
        profile["parent_cell_count"] = len(parent_cells)
        if profile["partial_cut_rule_count"] > 0 and profile["full_cell_rule_count"] > 0:
            support_class = "mixed_partial_and_full_cell_support"
        elif profile["partial_cut_rule_count"] > 0:
            support_class = "partial_cut_only_support"
        elif profile["full_cell_rule_count"] > 0:
            support_class = "full_cell_only_support"
        else:
            support_class = "missing_cut_volume_support"
        profile["cut_volume_support_class"] = support_class
        normalized[row] = profile
    return normalized


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def selector_finding(
    *,
    selected_count: int,
    covered: list[int],
    uncovered: list[int],
    direct_target_count: int,
    max_target_ratio: float,
) -> str:
    selected_to_target_ratio = ratio(selected_count, direct_target_count)
    covers_targets = (
        direct_target_count > 0
        and len(covered) == direct_target_count
        and not uncovered
    )
    overbroad = (
        selected_to_target_ratio is not None
        and selected_to_target_ratio > max_target_ratio
    )
    if not covers_targets and overbroad:
        return "selector_overbroad_and_misses_targets"
    if not covers_targets:
        return "selector_misses_targets"
    if overbroad:
        return "selector_overbroad"
    return "selector_selective"


def evaluate_selector_case(
    *,
    label: str,
    key: str,
    description: str,
    candidate_rows: list[int],
    target_rows: list[int],
    profiles: dict[int, dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    max_target_ratio: float,
) -> dict[str, Any]:
    selected = [
        row
        for row in candidate_rows
        if row in profiles and predicate(profiles[row])
    ]
    selected_set = set(selected)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    selected_to_target_ratio = ratio(len(selected), len(target_rows))
    return {
        "label": label,
        "key": key,
        "description": description,
        "finding": selector_finding(
            selected_count=len(selected),
            covered=covered,
            uncovered=uncovered,
            direct_target_count=len(target_rows),
            max_target_ratio=max_target_ratio,
        ),
        "direct_target_count": len(target_rows),
        "selected_count": len(selected),
        "selected_to_target_ratio": selected_to_target_ratio,
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "selected_global_dofs": selected,
    }


def aggregate_selector_finding(cases: list[dict[str, Any]]) -> str:
    if cases and all(case["finding"] == "selector_selective" for case in cases):
        return "selector_selective"
    if any("misses_targets" in str(case["finding"]) for case in cases):
        if any("overbroad" in str(case["finding"]) for case in cases):
            return "selector_overbroad_or_miss_targets"
        return "selector_misses_targets"
    if any(case["finding"] == "selector_overbroad" for case in cases):
        return "selector_overbroad"
    return "selector_inconclusive"


def selector_definitions(
    *,
    low_volume_fraction: float,
    very_low_volume_fraction: float,
    low_parent_cell_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "key": "cut_volume_profiled_candidate",
            "description": (
                "Preferred direct PSPG candidates that appear in the exact "
                "assembly-time cut-volume row provenance."
            ),
            "predicate": lambda profile: profile["rule_count"] > 0,
        },
        {
            "key": "cut_volume_partial_rule_support",
            "description": (
                "Profiled preferred candidates with at least one partial "
                "generated cut-volume rule."
            ),
            "predicate": lambda profile: profile["partial_cut_rule_count"] > 0,
        },
        {
            "key": "cut_volume_no_full_cell_support",
            "description": (
                "Profiled preferred candidates with partial cut-volume support "
                "and no full-cell equivalent rule."
            ),
            "predicate": (
                lambda profile: profile["partial_cut_rule_count"] > 0
                and profile["full_cell_rule_count"] == 0
            ),
        },
        {
            "key": "cut_volume_full_cell_only_support",
            "description": (
                "Profiled preferred candidates with full-cell equivalent "
                "generated volume support and no partial cut-volume rule."
            ),
            "predicate": (
                lambda profile: profile["full_cell_rule_count"] > 0
                and profile["partial_cut_rule_count"] == 0
            ),
        },
        {
            "key": "cut_volume_low_min_fraction",
            "description": (
                "Profiled preferred candidates whose minimum assembly-time "
                f"generated volume fraction is <= {low_volume_fraction}."
            ),
            "predicate": (
                lambda profile: profile["min_volume_fraction"] is not None
                and profile["min_volume_fraction"] <= low_volume_fraction
            ),
        },
        {
            "key": "cut_volume_very_low_min_fraction",
            "description": (
                "Profiled preferred candidates whose minimum assembly-time "
                f"generated volume fraction is <= {very_low_volume_fraction}."
            ),
            "predicate": (
                lambda profile: profile["min_volume_fraction"] is not None
                and profile["min_volume_fraction"] <= very_low_volume_fraction
            ),
        },
        {
            "key": "cut_volume_single_parent_cell_support",
            "description": (
                "Profiled preferred candidates supported by exactly one parent "
                "cell in the generated cut-volume rule set."
            ),
            "predicate": lambda profile: profile["parent_cell_count"] == 1,
        },
        {
            "key": "cut_volume_low_parent_cell_support",
            "description": (
                "Profiled preferred candidates with bounded parent-cell support "
                f"(<= {low_parent_cell_count})."
            ),
            "predicate": (
                lambda profile: profile["parent_cell_count"]
                <= low_parent_cell_count
            ),
        },
        {
            "key": "cut_volume_single_partial_rule_support",
            "description": (
                "Profiled preferred candidates with exactly one partial "
                "generated cut-volume rule and no full-cell rule."
            ),
            "predicate": (
                lambda profile: profile["partial_cut_rule_count"] == 1
                and profile["full_cell_rule_count"] == 0
            ),
        },
    ]


def profile_summary(
    *,
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [
        row for row in candidate_rows if row in profiles
    ]
    target_profiles = {
        str(row): profiles[row]
        for row in target_rows
        if row in profiles
    }
    support_class_counts: dict[str, int] = {}
    for row in profiled_candidates:
        support_class = profiles[row].get("cut_volume_support_class", "unknown")
        support_class_counts[support_class] = support_class_counts.get(
            support_class,
            0,
        ) + 1
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(target_set.intersection(profiles)),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "candidate_support_class_counts": support_class_counts,
        "target_profiles": target_profiles,
    }


def build_report(
    *,
    global_emission: dict[str, Any],
    target_map: dict[str, Any],
    global_emission_path: Path | None = None,
    target_map_path: Path | None = None,
    explicit_logs: list[str] | None = None,
    candidate_key: str = "preferred_candidate_global_dofs",
    operator: str = DEFAULT_OPERATOR,
    test_field: str = "pressure",
    trial_field: str = "pressure",
    low_volume_fraction: float = 0.10,
    very_low_volume_fraction: float = 1.0e-3,
    low_parent_cell_count: int = 2,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    emission_cases = case_map(global_emission)
    target_cases = target_case_map(target_map)
    log_paths = default_log_paths(emission_cases, explicit_logs or [])
    selector_defs = selector_definitions(
        low_volume_fraction=low_volume_fraction,
        very_low_volume_fraction=very_low_volume_fraction,
        low_parent_cell_count=low_parent_cell_count,
    )

    cases: dict[str, dict[str, Any]] = {}
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = latest_provenance_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        profiles = row_profiles_from_entries(entries)
        selector_cases = [
            evaluate_selector_case(
                label=label,
                key=selector["key"],
                description=selector["description"],
                candidate_rows=candidate_rows,
                target_rows=target_rows,
                profiles=profiles,
                predicate=selector["predicate"],
                max_target_ratio=max_target_ratio,
            )
            for selector in selector_defs
        ]
        cases[label] = {
            "label": label,
            "candidate_key": candidate_key,
            "candidate_count": len(candidate_rows),
            "direct_target_count": len(target_rows),
            "log_evidence": evidence,
            "profile_summary": profile_summary(
                profiles=profiles,
                candidate_rows=candidate_rows,
                target_rows=target_rows,
            ),
            "selectors": selector_cases,
        }

    selectors = []
    for selector_index, selector in enumerate(selector_defs):
        case_results = [
            cases[label]["selectors"][selector_index]
            for label in target_cases
        ]
        selectors.append(
            {
                "key": selector["key"],
                "description": selector["description"],
                "finding": aggregate_selector_finding(case_results),
                "cases": case_results,
            }
        )

    selective = [
        selector for selector in selectors if selector["finding"] == "selector_selective"
    ]
    overbroad = [
        selector
        for selector in selectors
        if "overbroad" in str(selector["finding"])
    ]
    misses = [
        selector for selector in selectors if "miss" in str(selector["finding"])
    ]
    missing_cases = [
        label
        for label, case in cases.items()
        if case["log_evidence"].get("status") != "ok"
    ]

    if missing_cases:
        finding = "direct_pspg_cut_volume_row_provenance_evidence_missing"
        next_requirement = (
            "Regenerate Test02/Test10 short replay logs with "
            "SVMP_FE_CUT_VOLUME_ROW_PROVENANCE_DIAGNOSTIC=1 and "
            f"SVMP_FE_CUT_VOLUME_ROW_PROVENANCE_OPERATOR={operator}."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_row_provenance_selector_selective"
        next_requirement = (
            "Prototype the selective assembly-time cut-volume provenance gate "
            "and run the same short Test02/Test10 replay windows."
        )
    elif overbroad or misses:
        finding = (
            "direct_pspg_cut_volume_row_provenance_selectors_overbroad_or_miss_targets"
        )
        next_requirement = (
            "Do not promote simple assembly-time cut-volume row provenance "
            "metrics alone; the remaining gate must include stronger element "
            "support physics or another formulation-side discriminator."
        )
    else:
        finding = "direct_pspg_cut_volume_row_provenance_selectivity_inconclusive"
        next_requirement = (
            "Regenerate assembly-time cut-volume provenance before selecting a "
            "formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for exact assembly-time generated cut-volume row "
            "provenance in the direct PSPG pressure-gradient diagnostic operator."
        ),
        "global_emission_path": (
            str(global_emission_path) if global_emission_path else None
        ),
        "target_map_path": str(target_map_path) if target_map_path else None,
        "candidate_key": candidate_key,
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
        "low_volume_fraction": low_volume_fraction,
        "very_low_volume_fraction": very_low_volume_fraction,
        "low_parent_cell_count": low_parent_cell_count,
        "max_target_ratio": max_target_ratio,
        "finding": finding,
        "missing_case_labels": missing_cases,
        "selective_selector_keys": [selector["key"] for selector in selective],
        "overbroad_selector_keys": [selector["key"] for selector in overbroad],
        "miss_selector_keys": [selector["key"] for selector in misses],
        "cases": list(cases.values()),
        "selectors": selectors,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        global_emission=load_json(args.global_emission_json),
        target_map=load_json(args.target_map_json),
        global_emission_path=args.global_emission_json,
        target_map_path=args.target_map_json,
        explicit_logs=args.log,
        candidate_key=args.candidate_key,
        operator=args.operator,
        test_field=args.test_field,
        trial_field=args.trial_field,
        low_volume_fraction=args.low_volume_fraction,
        very_low_volume_fraction=args.very_low_volume_fraction,
        low_parent_cell_count=args.low_parent_cell_count,
        max_target_ratio=args.max_target_ratio,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
