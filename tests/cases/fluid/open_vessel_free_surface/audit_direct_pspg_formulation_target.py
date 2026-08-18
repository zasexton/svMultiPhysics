#!/usr/bin/env python3
"""Derive direct PSPG pressure-gradient formulation targets from top-row provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TOPROW_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_toprow_provenance_20260606.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert exact pressure-operator top-row provenance into a compact "
            "formulation-side target map for the direct PSPG pressure-gradient "
            "support topology."
        )
    )
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


def count_by_label(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if isinstance(value, str):
            counts[value] += 1
    return dict(sorted(counts.items()))


def ordered_subset(source: list[Any], allowed: set[Any]) -> list[Any]:
    return [value for value in source if value in allowed]


def cross_policy_patch_by_label(
    top_provenance: dict[str, Any],
) -> dict[str, list[Any]]:
    patches: dict[str, list[Any]] = {}
    for item in as_list(top_provenance.get("cross_policy_neighbor_comparisons")):
        if not isinstance(item, dict):
            continue
        label = item.get("base_label") or item.get("full_gradient_label")
        if not isinstance(label, str) or not label:
            continue
        patch = as_list(item.get("current_top_isolated_cross_policy_patch_global_dofs"))
        if patch:
            patches[label] = ordered_unique(patch)
    return patches


def compact_components(components: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        compact.append(
            {
                "component_index": component.get("component_index"),
                "global_dofs": as_list(component.get("global_dofs")),
                "direct_pspg_global_dofs": as_list(
                    component.get("direct_pspg_global_dofs")
                ),
                "ghost_penalty_global_dofs": as_list(
                    component.get("ghost_penalty_global_dofs")
                ),
                "size": component.get("size"),
                "same_sign_pressure_action_edge_count": component.get(
                    "same_sign_pressure_action_edge_count"
                ),
                "contains_rank1": component.get("contains_rank1"),
                "boundary_class_counts": component.get("boundary_class_counts", {}),
                "incident_support_class_counts": component.get(
                    "incident_support_class_counts", {}
                ),
            }
        )
    return compact


def classify_direct_pspg_target(
    *,
    direct_rows: list[Any],
    ghost_rows: list[Any],
    isolated_direct_rows: list[Any],
    component_count: int,
) -> str:
    if not direct_rows:
        return "no_direct_pspg_top_rows"
    if isolated_direct_rows and ghost_rows:
        return "isolated_direct_pspg_row_with_ghost_penalty_branch"
    if isolated_direct_rows:
        return "isolated_direct_pspg_row"
    if component_count == 1 and not ghost_rows:
        return "coherent_direct_pspg_pressure_action_patch"
    if component_count > 1 and ghost_rows:
        return "split_direct_pspg_components_with_ghost_penalty_branch"
    if component_count > 1:
        return "split_direct_pspg_pressure_action_components"
    if ghost_rows:
        return "direct_pspg_rows_with_ghost_penalty_branch"
    return "direct_pspg_support_topology_unclassified"


def requirements_for_case(
    *,
    target_class: str,
    direct_rows: list[Any],
    ghost_rows: list[Any],
    isolated_direct_rows: list[Any],
    covered_direct_rows: list[Any],
) -> list[str]:
    requirements = [
        "derive coverage from the direct PSPG pressure-gradient support graph",
        "preserve constant-pressure null rows",
        "preserve matched hydrostatic cancellation",
        "avoid raw mesh incident-count, wall-flag, or current-top-row selectors",
    ]
    if direct_rows:
        requirements.append(
            "cover the moving direct PSPG pressure-gradient rows before solve"
        )
    if covered_direct_rows:
        requirements.append(
            "retain same-sign pressure-action patch coupling for connected rows"
        )
    if isolated_direct_rows:
        requirements.append(
            "provide formulation-side support for isolated direct rows without "
            "requiring post-assembly top-row adjacency"
        )
    if ghost_rows:
        requirements.append(
            "keep the pressure ghost-penalty branch visible as a branch shaper, "
            "not as the Test10 source path"
        )
    if target_class == "coherent_direct_pspg_pressure_action_patch":
        requirements.append(
            "treat the coherent direct-PSPG patch as a coupled pressure graph, "
            "not a single-row clamp"
        )
    return requirements


def classify_case(
    case: dict[str, Any],
    *,
    cross_policy_patch_dofs: list[Any],
) -> dict[str, Any]:
    direct_rows = as_list(case.get("direct_pspg_balance_global_dofs"))
    ghost_rows = as_list(case.get("ghost_penalty_balance_global_dofs"))
    isolated_direct_rows = as_list(
        case.get("direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs")
    )
    covered_direct_rows = as_list(
        case.get("direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs")
    )
    components = compact_components(
        as_list(case.get("direct_pspg_same_sign_pressure_action_components"))
    )
    component_count = case.get("direct_pspg_same_sign_pressure_action_component_count")
    if not isinstance(component_count, int):
        component_count = len(components)
    target_class = classify_direct_pspg_target(
        direct_rows=direct_rows,
        ghost_rows=ghost_rows,
        isolated_direct_rows=isolated_direct_rows,
        component_count=component_count,
    )
    support_gap_dofs = ordered_unique(
        as_list(case.get("direct_pspg_sparse_direct_self_entry_global_dofs"))
        + as_list(case.get("direct_pspg_low_direct_self_ratio_global_dofs"))
        + as_list(case.get("direct_pspg_missing_wall_normal_self_global_dofs"))
        + as_list(case.get("direct_pspg_missing_wall_tangential_self_global_dofs"))
    )
    return {
        "label": case.get("label"),
        "source_finding": case.get("finding"),
        "formulation_target_class": target_class,
        "direct_pspg_target_global_dofs": direct_rows,
        "ghost_penalty_branch_global_dofs": ghost_rows,
        "direct_pspg_same_sign_covered_global_dofs": covered_direct_rows,
        "direct_pspg_isolated_global_dofs": isolated_direct_rows,
        "cross_policy_isolated_patch_global_dofs": cross_policy_patch_dofs,
        "direct_pspg_same_sign_pressure_action_component_count": component_count,
        "direct_pspg_same_sign_pressure_action_components": components,
        "direct_pspg_support_gap_global_dofs": support_gap_dofs,
        "support_gap_breakdown": {
            "sparse_direct_self_entry_global_dofs": as_list(
                case.get("direct_pspg_sparse_direct_self_entry_global_dofs")
            ),
            "low_direct_self_ratio_global_dofs": as_list(
                case.get("direct_pspg_low_direct_self_ratio_global_dofs")
            ),
            "missing_wall_normal_self_global_dofs": as_list(
                case.get("direct_pspg_missing_wall_normal_self_global_dofs")
            ),
            "missing_wall_tangential_self_global_dofs": as_list(
                case.get("direct_pspg_missing_wall_tangential_self_global_dofs")
            ),
        },
        "boundary_class_counts": case.get("boundary_class_counts", {}),
        "incident_support_class_counts": case.get("incident_support_class_counts", {}),
        "physical_path_class_counts": case.get("physical_path_class_counts", {}),
        "formulation_requirements": requirements_for_case(
            target_class=target_class,
            direct_rows=direct_rows,
            ghost_rows=ghost_rows,
            isolated_direct_rows=isolated_direct_rows,
            covered_direct_rows=covered_direct_rows,
        ),
    }


def rows_for_candidate(case: dict[str, Any], key: str) -> list[Any]:
    if key == "direct_support_gap_rows_only":
        return as_list(case.get("direct_pspg_support_gap_global_dofs"))
    if key == "same_sign_pressure_action_patch_only":
        return as_list(case.get("direct_pspg_same_sign_covered_global_dofs"))
    if key == "isolated_or_same_sign_direct_targets":
        return ordered_unique(
            as_list(case.get("direct_pspg_isolated_global_dofs"))
            + as_list(case.get("direct_pspg_same_sign_covered_global_dofs"))
        )
    if key == "direct_support_gap_or_same_sign_pressure_action_patch":
        return ordered_unique(
            as_list(case.get("direct_pspg_support_gap_global_dofs"))
            + as_list(case.get("direct_pspg_same_sign_covered_global_dofs"))
        )
    return []


def candidate_derivation_report(key: str) -> dict[str, Any]:
    derivations = {
        "direct_support_gap_rows_only": {
            "source_class": "post_assembly_operator_sample",
            "production_readiness": (
                "diagnostic_only_partial_coverage"
            ),
            "solve_time_derivation_status": (
                "not_proven_for_unsampled_active_rows"
            ),
            "depends_on_top_update_rows": True,
            "depends_on_pressure_update_values": False,
            "depends_on_post_assembly_matrix": True,
            "depends_on_explicit_row_list": False,
            "formulation_gap": (
                "Needs a solve-time support-deficiency criterion from active "
                "direct PSPG pressure-gradient topology instead of sampled "
                "post-assembly top rows."
            ),
        },
        "same_sign_pressure_action_patch_only": {
            "source_class": "top_update_pressure_action_graph",
            "production_readiness": (
                "diagnostic_only_partial_coverage"
            ),
            "solve_time_derivation_status": (
                "not_available_before_pressure_update"
            ),
            "depends_on_top_update_rows": True,
            "depends_on_pressure_update_values": True,
            "depends_on_post_assembly_matrix": True,
            "depends_on_explicit_row_list": False,
            "formulation_gap": (
                "Needs a pressure-action patch rule derived from direct PSPG "
                "support/coupling topology before the Newton update is known."
            ),
        },
        "isolated_or_same_sign_direct_targets": {
            "source_class": "exact_diagnostic_target_row_set",
            "production_readiness": (
                "diagnostic_only_exact_target_oracle"
            ),
            "solve_time_derivation_status": "invalid_as_formulation_rule",
            "depends_on_top_update_rows": True,
            "depends_on_pressure_update_values": True,
            "depends_on_post_assembly_matrix": True,
            "depends_on_explicit_row_list": True,
            "formulation_gap": (
                "This is the audited bad-row set, not a physical rule that can "
                "be assembled before solve."
            ),
        },
        "direct_support_gap_or_same_sign_pressure_action_patch": {
            "source_class": "combined_post_assembly_diagnostic_proxy",
            "production_readiness": (
                "coverage_complete_but_diagnostic_only"
            ),
            "solve_time_derivation_status": (
                "requires_formulation_side_topology_replacement"
            ),
            "depends_on_top_update_rows": True,
            "depends_on_pressure_update_values": True,
            "depends_on_post_assembly_matrix": True,
            "depends_on_explicit_row_list": False,
            "formulation_gap": (
                "Coverage is complete for the audited Test02/Test10 rows, but "
                "the same predicate must be re-derived from solve-time active "
                "cut-volume direct PSPG pressure-gradient topology and coupling."
            ),
        },
    }
    return derivations.get(
        key,
        {
            "source_class": "unknown",
            "production_readiness": "unknown",
            "solve_time_derivation_status": "unknown",
            "depends_on_top_update_rows": None,
            "depends_on_pressure_update_values": None,
            "depends_on_post_assembly_matrix": None,
            "depends_on_explicit_row_list": None,
            "formulation_gap": "unknown candidate",
        },
    )


def candidate_coverage_report(
    cases: list[dict[str, Any]],
    *,
    key: str,
    description: str,
) -> dict[str, Any]:
    case_reports: list[dict[str, Any]] = []
    for case in cases:
        target_rows = as_list(case.get("direct_pspg_target_global_dofs"))
        selected = set(rows_for_candidate(case, key))
        covered = ordered_subset(target_rows, selected)
        uncovered = [row for row in target_rows if row not in selected]
        coverage_ratio = (
            float(len(covered)) / float(len(target_rows)) if target_rows else 1.0
        )
        case_reports.append(
            {
                "label": case.get("label"),
                "direct_target_count": len(target_rows),
                "covered_direct_target_global_dofs": covered,
                "uncovered_direct_target_global_dofs": uncovered,
                "coverage_ratio": coverage_ratio,
                "covers_all_direct_targets": not uncovered,
            }
        )
    uncovered_case_count = sum(
        1 for case in case_reports if not case["covers_all_direct_targets"]
    )
    return {
        "key": key,
        "description": description,
        "derivation": candidate_derivation_report(key),
        "covers_all_cases": uncovered_case_count == 0,
        "uncovered_case_count": uncovered_case_count,
        "cases": case_reports,
    }


def candidate_coverage_reports(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate_coverage_report(
            cases,
            key="direct_support_gap_rows_only",
            description=(
                "Rows with exact direct PSPG pressure-gradient support gaps "
                "such as sparse direct self entries, low direct-self ratio, or "
                "missing wall-normal/tangential support."
            ),
        ),
        candidate_coverage_report(
            cases,
            key="same_sign_pressure_action_patch_only",
            description=(
                "Rows covered by same-sign pressure-action connectivity among "
                "the exact direct PSPG top rows."
            ),
        ),
        candidate_coverage_report(
            cases,
            key="isolated_or_same_sign_direct_targets",
            description=(
                "The exact target-class rows: isolated direct PSPG rows plus "
                "same-sign direct PSPG pressure-action patches."
            ),
        ),
        candidate_coverage_report(
            cases,
            key="direct_support_gap_or_same_sign_pressure_action_patch",
            description=(
                "Physical-support proxy combining exact direct support gaps "
                "with same-sign direct PSPG pressure-action patch coverage."
            ),
        ),
    ]


def build_formulation_target_report(
    top_provenance: dict[str, Any],
    *,
    top_provenance_path: Path | None = None,
) -> dict[str, Any]:
    patches_by_label = cross_policy_patch_by_label(top_provenance)
    cases = []
    for case in as_list(top_provenance.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        cases.append(
            classify_case(
                case,
                cross_policy_patch_dofs=patches_by_label.get(label, []),
            )
        )

    direct_target_cases = [
        case for case in cases if case["direct_pspg_target_global_dofs"]
    ]
    ghost_branch_cases = [
        case for case in cases if case["ghost_penalty_branch_global_dofs"]
    ]
    isolated_cases = [
        case for case in cases if case["direct_pspg_isolated_global_dofs"]
    ]
    coherent_cases = [
        case
        for case in cases
        if case["formulation_target_class"]
        == "coherent_direct_pspg_pressure_action_patch"
    ]

    finding = (
        "direct_pspg_formulation_target_identified"
        if direct_target_cases
        else "no_direct_pspg_formulation_target_rows"
    )
    if isolated_cases and coherent_cases:
        finding = "mixed_isolated_and_coherent_direct_pspg_formulation_targets"
    elif isolated_cases:
        finding = "isolated_direct_pspg_formulation_target_identified"
    elif coherent_cases:
        finding = "coherent_direct_pspg_formulation_target_identified"

    coverage_reports = candidate_coverage_reports(cases)
    support_gap_or_patch = next(
        (
            report
            for report in coverage_reports
            if report["key"]
            == "direct_support_gap_or_same_sign_pressure_action_patch"
        ),
        None,
    )
    recommended_predicate = (
        {
            "key": support_gap_or_patch["key"],
            "covers_all_cases": support_gap_or_patch["covers_all_cases"],
            "production_readiness": support_gap_or_patch["derivation"][
                "production_readiness"
            ],
            "solve_time_derivation_status": support_gap_or_patch["derivation"][
                "solve_time_derivation_status"
            ],
            "reason": (
                "Support-gap rows reach the isolated Test02 branch, while "
                "same-sign direct PSPG pressure-action patch coverage reaches "
                "the coherent Test10 branch. This remains a diagnostic "
                "coverage predicate; a solver change must derive it before "
                "solve from active PSPG pressure-gradient support topology."
            ),
        }
        if support_gap_or_patch is not None
        else None
    )
    complete_ready_candidates = [
        report
        for report in coverage_reports
        if report["covers_all_cases"]
        and report["derivation"]["production_readiness"]
        not in {
            "coverage_complete_but_diagnostic_only",
            "diagnostic_only_exact_target_oracle",
        }
    ]
    complete_diagnostic_candidates = [
        report
        for report in coverage_reports
        if report["covers_all_cases"]
        and report["derivation"]["production_readiness"]
        in {
            "coverage_complete_but_diagnostic_only",
            "diagnostic_only_exact_target_oracle",
        }
    ]
    predicate_derivation_readiness = (
        "formulation_side_predicate_ready"
        if complete_ready_candidates
        else "coverage_complete_but_no_formulation_side_derivation"
        if complete_diagnostic_candidates
        else "no_complete_candidate_predicate"
    )

    return {
        "finding": finding,
        "source": {
            "toprow_provenance_path": (
                str(top_provenance_path) if top_provenance_path is not None else None
            ),
            "toprow_provenance_finding": top_provenance.get("finding"),
        },
        "remaining_hypothesis": "direct_pspg_pressure_gradient_support_topology",
        "formulation_path": {
            "target_volume_form": "inner(grad(q), tau_m * grad(pspg_pressure_gradient_pressure))",
            "support_boundary_forms": [
                "wall_normal_pressure_gradient",
                "wall_tangential_pressure_gradient",
            ],
            "non_target_paths": [
                "pressure_ghost_penalty",
                "pspg_nonpressure_momentum_residual",
                "generated_interface_pressure_reference_probe",
            ],
        },
        "case_count": len(cases),
        "direct_target_case_count": len(direct_target_cases),
        "ghost_branch_case_count": len(ghost_branch_cases),
        "formulation_target_class_counts": count_by_label(
            cases, "formulation_target_class"
        ),
        "candidate_coverage": coverage_reports,
        "recommended_next_predicate": recommended_predicate,
        "predicate_derivation_readiness": predicate_derivation_readiness,
        "complete_diagnostic_candidate_keys": [
            report["key"] for report in complete_diagnostic_candidates
        ],
        "complete_formulation_ready_candidate_keys": [
            report["key"] for report in complete_ready_candidates
        ],
        "next_derivation_requirement": (
            "Replace top-update and post-assembly sample dependencies with a "
            "solve-time active cut-volume direct PSPG pressure-gradient support "
            "topology/coupling rule."
        ),
        "cases": cases,
        "global_requirements": [
            "implement or replay a formulation-side topology/coupling rule on the "
            "direct PSPG pressure-gradient support path",
            "cover both isolated Test02-style rows and coherent Test10-style patches",
            "preserve constant-pressure null and matched hydrostatic cancellation",
            "verify with Test02 and Test10 accepted-step pressure-update guards",
        ],
    }


def main() -> int:
    args = parse_args()
    top_provenance = load_json(args.toprow_provenance_json)
    report = build_formulation_target_report(
        top_provenance,
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
