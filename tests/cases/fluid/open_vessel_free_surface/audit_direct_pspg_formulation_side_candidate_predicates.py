#!/usr/bin/env python3
"""Rank narrower formulation-side direct PSPG candidate predicates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)
DEFAULT_TOPROW_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_toprow_provenance_20260606.json"
)


CANDIDATE_DEFINITIONS = [
    {
        "key": "same_sign_pressure_action_patch",
        "description": (
            "Direct PSPG rows covered by same-sign pressure-action connectivity."
        ),
        "row_sources": ["same_sign_pressure_action"],
        "production_readiness": "diagnostic_only_partial_expected",
        "derivation_status": "needs_preupdate_direct_pspg_pressure_action_graph",
        "depends_on_pressure_update_values_in_current_artifact": True,
    },
    {
        "key": "sparse_direct_self_or_same_sign_pressure_action_patch",
        "description": (
            "Same-sign direct PSPG pressure-action patch rows plus isolated rows "
            "with sparse direct pressure-gradient self entries."
        ),
        "row_sources": ["sparse_direct_self_entry", "same_sign_pressure_action"],
        "production_readiness": (
            "formulation_candidate_pending_global_solve_time_emission"
        ),
        "derivation_status": (
            "derive_from_direct_pspg_pressure_gradient_self_topology_and_action_graph"
        ),
        "depends_on_pressure_update_values_in_current_artifact": True,
    },
    {
        "key": "low_or_moderate_direct_self_or_same_sign_pressure_action_patch",
        "description": (
            "Same-sign direct PSPG patch rows plus rows with low or moderate "
            "direct-self support ratios."
        ),
        "row_sources": [
            "low_direct_self_ratio",
            "moderate_direct_self_ratio",
            "same_sign_pressure_action",
        ],
        "production_readiness": "thresholded_diagnostic_only",
        "derivation_status": "requires_scale_threshold_not_yet_physical",
        "depends_on_pressure_update_values_in_current_artifact": True,
    },
    {
        "key": "missing_wall_support_or_same_sign_pressure_action_patch",
        "description": (
            "Same-sign direct PSPG patch rows plus rows missing wall-normal or "
            "wall-tangential pressure-gradient self support."
        ),
        "row_sources": [
            "missing_wall_normal_self",
            "missing_wall_tangential_self",
            "same_sign_pressure_action",
        ],
        "production_readiness": "wall_flag_diagnostic_only",
        "derivation_status": "wall_support_flags_ruled_out_as_complete_rule",
        "depends_on_pressure_update_values_in_current_artifact": True,
    },
    {
        "key": "zero_galerkin_nonpressure_or_same_sign_pressure_action_patch",
        "description": (
            "Same-sign direct PSPG patch rows plus rows with zero Galerkin and "
            "zero nonpressure PSPG coupling."
        ),
        "row_sources": [
            "zero_galerkin_nonpressure_coupling",
            "same_sign_pressure_action",
        ],
        "production_readiness": "diagnostic_only_partial_expected",
        "derivation_status": "known_partial_test10_only_support_signal",
        "depends_on_pressure_update_values_in_current_artifact": True,
    },
    {
        "key": "support_gap_or_same_sign_pressure_action_patch",
        "description": (
            "Previous support-gap proxy plus same-sign pressure-action coverage."
        ),
        "row_sources": ["target_map_support_gap", "same_sign_pressure_action"],
        "production_readiness": "coverage_complete_but_diagnostic_only",
        "derivation_status": "mixed_support_gap_proxy_needs_replacement",
        "depends_on_pressure_update_values_in_current_artifact": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare narrower direct PSPG formulation-side candidate predicates "
            "against the audited Test02/Test10 target rows."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--toprow-provenance-json",
        type=Path,
        default=DEFAULT_TOPROW_PROVENANCE,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ordered_unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(report.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def rows_from_source(
    *,
    source: str,
    target_case: dict[str, Any],
    provenance_case: dict[str, Any],
) -> list[Any]:
    if source == "same_sign_pressure_action":
        return as_list(
            provenance_case.get(
                "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs"
            )
        )
    if source == "sparse_direct_self_entry":
        return as_list(
            provenance_case.get("direct_pspg_sparse_direct_self_entry_global_dofs")
        )
    if source == "low_direct_self_ratio":
        return as_list(
            provenance_case.get("direct_pspg_low_direct_self_ratio_global_dofs")
        )
    if source == "moderate_direct_self_ratio":
        return as_list(
            provenance_case.get("direct_pspg_moderate_direct_self_ratio_global_dofs")
        )
    if source == "missing_wall_normal_self":
        return as_list(
            provenance_case.get("direct_pspg_missing_wall_normal_self_global_dofs")
        )
    if source == "missing_wall_tangential_self":
        return as_list(
            provenance_case.get("direct_pspg_missing_wall_tangential_self_global_dofs")
        )
    if source == "zero_galerkin_nonpressure_coupling":
        return as_list(
            provenance_case.get(
                "direct_pspg_zero_galerkin_nonpressure_coupling_global_dofs"
            )
        )
    if source == "target_map_support_gap":
        return as_list(target_case.get("direct_pspg_support_gap_global_dofs"))
    return []


def selected_rows_for_candidate(
    *,
    candidate: dict[str, Any],
    target_case: dict[str, Any],
    provenance_case: dict[str, Any],
) -> list[Any]:
    rows: list[Any] = []
    for source in as_list(candidate.get("row_sources")):
        rows.extend(
            rows_from_source(
                source=source,
                target_case=target_case,
                provenance_case=provenance_case,
            )
        )
    return ordered_unique(rows)


def evaluate_candidate_case(
    *,
    candidate: dict[str, Any],
    label: str,
    target_case: dict[str, Any],
    provenance_case: dict[str, Any],
) -> dict[str, Any]:
    target_rows = as_list(target_case.get("direct_pspg_target_global_dofs"))
    target_set = set(target_rows)
    selected_rows = selected_rows_for_candidate(
        candidate=candidate,
        target_case=target_case,
        provenance_case=provenance_case,
    )
    selected_set = set(selected_rows)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    extra = [row for row in selected_rows if row not in target_set]
    coverage_ratio = (
        float(len(covered)) / float(len(target_rows)) if target_rows else 1.0
    )
    selected_to_target_ratio = (
        float(len(selected_rows)) / float(len(target_rows)) if target_rows else None
    )

    if not uncovered and not extra:
        finding = "exact_audited_target_coverage"
    elif not uncovered:
        finding = "covers_targets_but_overselects_audited_rows"
    elif covered:
        finding = "partial_audited_target_coverage"
    else:
        finding = "misses_audited_targets"

    return {
        "label": label,
        "direct_target_count": len(target_rows),
        "selected_count": len(selected_rows),
        "selected_to_direct_target_ratio": selected_to_target_ratio,
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "extra_selected_global_dofs": extra,
        "coverage_ratio": coverage_ratio,
        "finding": finding,
    }


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    target_cases: dict[str, dict[str, Any]],
    provenance_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_reports = []
    for label, target_case in target_cases.items():
        provenance_case = provenance_cases.get(label, {})
        case_reports.append(
            evaluate_candidate_case(
                candidate=candidate,
                label=label,
                target_case=target_case,
                provenance_case=provenance_case,
            )
        )

    finding_counts = Counter(case["finding"] for case in case_reports)
    all_exact = all(
        case["finding"] == "exact_audited_target_coverage"
        for case in case_reports
    )
    all_covered = all(
        not case["uncovered_direct_target_global_dofs"] for case in case_reports
    )
    any_overselect = any(case["extra_selected_global_dofs"] for case in case_reports)

    if all_exact:
        finding = "exact_audited_coverage"
    elif all_covered and any_overselect:
        finding = "complete_but_overselects_audited_rows"
    elif any(case["covered_direct_target_global_dofs"] for case in case_reports):
        finding = "partial_audited_coverage"
    else:
        finding = "no_audited_coverage"

    return {
        "key": candidate["key"],
        "description": candidate["description"],
        "row_sources": candidate["row_sources"],
        "production_readiness": candidate["production_readiness"],
        "derivation_status": candidate["derivation_status"],
        "depends_on_pressure_update_values_in_current_artifact": candidate[
            "depends_on_pressure_update_values_in_current_artifact"
        ],
        "finding": finding,
        "case_finding_counts": dict(sorted(finding_counts.items())),
        "covers_all_audited_targets": all_covered,
        "exact_audited_coverage": all_exact,
        "cases": case_reports,
    }


def preferred_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_readiness = "formulation_candidate_pending_global_solve_time_emission"
    exact = [
        candidate
        for candidate in candidates
        if candidate["exact_audited_coverage"]
        and candidate["production_readiness"] == preferred_readiness
    ]
    if exact:
        return exact[0]
    complete = [
        candidate
        for candidate in candidates
        if candidate["covers_all_audited_targets"]
        and candidate["production_readiness"] == preferred_readiness
    ]
    if complete:
        return complete[0]
    return None


def build_report(
    *,
    target_map: dict[str, Any],
    top_provenance: dict[str, Any],
    target_map_path: Path | None = None,
    top_provenance_path: Path | None = None,
) -> dict[str, Any]:
    target_cases = case_map(target_map)
    provenance_cases = case_map(top_provenance)
    candidates = [
        evaluate_candidate(
            candidate=candidate,
            target_cases=target_cases,
            provenance_cases=provenance_cases,
        )
        for candidate in CANDIDATE_DEFINITIONS
    ]
    preferred = preferred_candidate(candidates)
    exact_candidate_keys = [
        candidate["key"]
        for candidate in candidates
        if candidate["exact_audited_coverage"]
    ]
    complete_candidate_keys = [
        candidate["key"]
        for candidate in candidates
        if candidate["covers_all_audited_targets"]
    ]
    partial_candidate_keys = [
        candidate["key"]
        for candidate in candidates
        if candidate["finding"] == "partial_audited_coverage"
    ]

    if preferred is not None:
        finding = "narrow_formulation_side_candidate_identified_needs_global_emission"
    elif exact_candidate_keys:
        finding = "exact_audited_candidate_exists_but_not_formulation_ready"
    elif complete_candidate_keys:
        finding = "complete_audited_candidate_exists_but_overselects"
    else:
        finding = "no_complete_formulation_side_candidate_identified"

    return {
        "scope": (
            "Narrow candidate predicates compared against audited direct PSPG "
            "target rows, using exact top-row provenance as evidence."
        ),
        "target_map_path": str(target_map_path) if target_map_path else None,
        "toprow_provenance_path": (
            str(top_provenance_path) if top_provenance_path else None
        ),
        "finding": finding,
        "preferred_next_candidate": (
            {
                "key": preferred["key"],
                "production_readiness": preferred["production_readiness"],
                "derivation_status": preferred["derivation_status"],
            }
            if preferred is not None
            else None
        ),
        "exact_audited_candidate_keys": exact_candidate_keys,
        "complete_audited_candidate_keys": complete_candidate_keys,
        "partial_candidate_keys": partial_candidate_keys,
        "direct_target_counts": {
            label: len(as_list(case.get("direct_pspg_target_global_dofs")))
            for label, case in target_cases.items()
        },
        "current_artifact_limitation": (
            "The preferred candidate is still proven only on exact sampled top "
            "rows. The next diagnostic must emit the sparse direct-self and "
            "same-sign direct PSPG pressure-action predicate globally before "
            "the pressure update is known."
        ),
        "next_requirement": (
            "Add solve-time/global candidate emission for the preferred direct "
            "PSPG pressure-gradient self-topology plus pressure-action patch "
            "predicate, then rerun short Test02/Test10 windows to test breadth "
            "and pressure-update behavior."
        ),
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    target_map = load_json(args.target_map_json)
    top_provenance = load_json(args.toprow_provenance_json)
    report = build_report(
        target_map=target_map,
        top_provenance=top_provenance,
        target_map_path=args.target_map_json,
        top_provenance_path=args.toprow_provenance_json,
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
