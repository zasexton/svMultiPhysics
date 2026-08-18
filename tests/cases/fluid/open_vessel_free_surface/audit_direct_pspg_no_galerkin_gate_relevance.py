#!/usr/bin/env python3
"""Audit no-Galerkin/nonpressure coupling as a direct PSPG formulation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TOP_OVERLAP = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_top_update_overlap_20260606.json"
)
DEFAULT_FORMULATION_PREDICATES = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_formulation_side_candidate_predicates_20260606.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join top-row operator overlap and formulation-predicate evidence "
            "to decide whether no-Galerkin/nonpressure zero coupling can be "
            "promoted as the remaining direct PSPG support gate."
        )
    )
    parser.add_argument("--top-overlap-json", type=Path, default=DEFAULT_TOP_OVERLAP)
    parser.add_argument(
        "--formulation-predicates-json",
        type=Path,
        default=DEFAULT_FORMULATION_PREDICATES,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def candidate_by_key(report: dict[str, Any], key: str) -> dict[str, Any]:
    for candidate in as_list(report.get("candidates")):
        if isinstance(candidate, dict) and candidate.get("key") == key:
            return candidate
    return {}


def summarize_overlap_case(case: dict[str, Any]) -> dict[str, Any]:
    direct_target_count = int(case.get("exact_direct_pspg_top_update_count") or 0)
    overlap_count = int(case.get("no_galerkin_top_update_overlap_count") or 0)
    no_galerkin_rows = as_list(case.get("no_galerkin_zero_coupling_global_dofs"))
    no_nonpressure_rows = as_list(case.get("no_nonpressure_zero_coupling_global_dofs"))
    support_rank_rows = as_list(case.get("support_rank_zero_coupling_global_dofs"))
    return {
        "label": case.get("label"),
        "finding": case.get("finding"),
        "no_galerkin_support_finding": case.get("no_galerkin_support_finding"),
        "direct_target_count": direct_target_count,
        "no_galerkin_top_update_overlap_count": overlap_count,
        "no_galerkin_top_update_overlap_ratio": ratio(
            overlap_count, direct_target_count
        ),
        "no_galerkin_top_update_overlap_global_dofs": as_list(
            case.get("no_galerkin_top_update_overlap_global_dofs")
        ),
        "no_galerkin_zero_coupling_count": len(no_galerkin_rows),
        "no_nonpressure_zero_coupling_count": len(no_nonpressure_rows),
        "support_rank_zero_coupling_count": len(support_rank_rows),
        "no_galerkin_equals_no_nonpressure_zero_coupling": case.get(
            "no_galerkin_equals_no_nonpressure_zero_coupling"
        ),
        "no_galerkin_equals_support_rank_zero_coupling": case.get(
            "no_galerkin_equals_support_rank_zero_coupling"
        ),
        "exact_direct_pspg_rows_missing_any_aggregate_sample_count": case.get(
            "exact_direct_pspg_rows_missing_any_aggregate_sample_count"
        ),
        "exact_to_aggregate_sample_finding": case.get(
            "exact_to_aggregate_sample_finding"
        ),
    }


def summarize_candidate_case(case: dict[str, Any]) -> dict[str, Any]:
    direct_target_count = int(case.get("direct_target_count") or 0)
    selected_count = int(case.get("selected_count") or 0)
    return {
        "label": case.get("label"),
        "finding": case.get("finding"),
        "direct_target_count": direct_target_count,
        "selected_count": selected_count,
        "selected_to_direct_target_ratio": ratio(
            selected_count, direct_target_count
        ),
        "covered_direct_target_global_dofs": as_list(
            case.get("covered_direct_target_global_dofs")
        ),
        "uncovered_direct_target_global_dofs": as_list(
            case.get("uncovered_direct_target_global_dofs")
        ),
        "coverage_ratio": case.get("coverage_ratio"),
    }


def build_report(
    *,
    top_overlap: dict[str, Any],
    formulation_predicates: dict[str, Any],
) -> dict[str, Any]:
    overlap_cases = [
        summarize_overlap_case(case)
        for case in as_list(top_overlap.get("cases"))
        if isinstance(case, dict)
    ]
    candidate = candidate_by_key(
        formulation_predicates,
        "zero_galerkin_nonpressure_or_same_sign_pressure_action_patch",
    )
    candidate_cases = [
        summarize_candidate_case(case)
        for case in as_list(candidate.get("cases"))
        if isinstance(case, dict)
    ]
    overlap_missing_cases = [
        case["label"]
        for case in overlap_cases
        if case["direct_target_count"] > 0
        and case["no_galerkin_top_update_overlap_count"] == 0
    ]
    overlap_partial_cases = [
        case["label"]
        for case in overlap_cases
        if 0
        < case["no_galerkin_top_update_overlap_count"]
        < case["direct_target_count"]
    ]
    candidate_uncovered_cases = [
        case["label"]
        for case in candidate_cases
        if case["uncovered_direct_target_global_dofs"]
    ]
    support_rank_mismatch_cases = [
        case["label"]
        for case in overlap_cases
        if case["no_galerkin_equals_support_rank_zero_coupling"] is False
    ]

    if (
        overlap_missing_cases
        or overlap_partial_cases
        or candidate_uncovered_cases
        or support_rank_mismatch_cases
    ):
        finding = "no_galerkin_nonpressure_gate_ruled_out_as_complete_formulation_gate"
        status = "partial_test10_signal_ruled_out_as_complete_gate"
    else:
        finding = "no_galerkin_nonpressure_gate_supported_for_replay"
        status = "candidate_gate_needs_replay"

    return {
        "finding": finding,
        "status": status,
        "top_overlap": {
            "finding": top_overlap.get("finding"),
            "no_galerkin_support_finding": top_overlap.get(
                "no_galerkin_support_finding"
            ),
            "exact_to_aggregate_sample_finding": top_overlap.get(
                "exact_to_aggregate_sample_finding"
            ),
            "cases": overlap_cases,
        },
        "formulation_candidate": {
            "key": candidate.get("key"),
            "finding": candidate.get("finding"),
            "production_readiness": candidate.get("production_readiness"),
            "derivation_status": candidate.get("derivation_status"),
            "covers_all_audited_targets": candidate.get(
                "covers_all_audited_targets"
            ),
            "depends_on_pressure_update_values_in_current_artifact": candidate.get(
                "depends_on_pressure_update_values_in_current_artifact"
            ),
            "cases": candidate_cases,
        },
        "classification": {
            "overlap_missing_cases": overlap_missing_cases,
            "overlap_partial_cases": overlap_partial_cases,
            "candidate_uncovered_cases": candidate_uncovered_cases,
            "support_rank_mismatch_cases": support_rank_mismatch_cases,
            "complete_gate_candidate": not (
                overlap_missing_cases
                or overlap_partial_cases
                or candidate_uncovered_cases
                or support_rank_mismatch_cases
            ),
        },
        "conclusion": (
            "No-Galerkin/nonpressure zero coupling remains useful Test10 rank "
            "evidence, but it cannot be the complete direct PSPG formulation "
            "gate. It is absent on the Test02 direct PSPG top rows, covers only "
            "part of Test10, and the combined same-sign predicate still misses "
            "the isolated Test02 row."
        ),
        "next_requirement": (
            "Keep no-Galerkin/nonpressure zero coupling as a Test10 sub-signal, "
            "but derive the remaining gate from direct PSPG pressure-gradient "
            "support/coupling topology rather than promoting this selector."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        top_overlap=load_json(args.top_overlap_json),
        formulation_predicates=load_json(args.formulation_predicates_json),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
