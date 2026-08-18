#!/usr/bin/env python3
"""Audit whether same-sign direct PSPG patch evidence is formulation-ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)
DEFAULT_PREDICATES = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_formulation_side_candidate_predicates_20260606.json"
)
DEFAULT_GLOBAL_SELECTIVITY = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_global_candidate_selectivity_20260607.json"
)
DEFAULT_TOPROW_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_toprow_provenance_20260606.json"
)
DEFAULT_PRESSURE_DISABLED_TOPROW_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_toprow_provenance_pressure_disabled_20260606.json"
)


GLOBAL_PROXY_GATE_KEYS = [
    "direct_self_support_ratio_gate_finding",
    "graph_local_support_ratio_gate_finding",
    "pressure_action_moderate_degree_gate_finding",
    "pressure_action_moderate_sum_ratio_gate_finding",
    "pressure_action_self_dominant_gate_finding",
    "sparse_seeded_pressure_action_radius1_gate_finding",
    "sparse_seeded_pressure_action_radius2_gate_finding",
]


CASE_RATIO_KEYS = [
    "preferred_to_target_ratio",
    "sparse_direct_self_to_target_ratio",
    "sparse_or_moderate_direct_self_ratio_to_target_ratio",
    "graph_local_moderate_direct_self_ratio_to_target_ratio",
    "pressure_action_moderate_degree_to_target_ratio",
    "pressure_action_moderate_sum_ratio_to_target_ratio",
    "pressure_action_self_dominant_to_target_ratio",
    "sparse_seeded_pressure_action_radius1_to_target_ratio",
    "sparse_seeded_pressure_action_radius2_to_target_ratio",
    "sparse_seeded_matrix_pressure_action_component_to_target_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether exact same-sign direct PSPG patch evidence can be "
            "promoted before solve, or whether current pre-update proxies rule "
            "that out and require new formulation-side instrumentation."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--predicate-json", type=Path, default=DEFAULT_PREDICATES)
    parser.add_argument(
        "--global-selectivity-json",
        type=Path,
        default=DEFAULT_GLOBAL_SELECTIVITY,
    )
    parser.add_argument(
        "--toprow-provenance-json",
        type=Path,
        default=DEFAULT_TOPROW_PROVENANCE,
    )
    parser.add_argument(
        "--pressure-disabled-toprow-provenance-json",
        type=Path,
        default=DEFAULT_PRESSURE_DISABLED_TOPROW_PROVENANCE,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ordered_unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def case_map(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    if not isinstance(report, dict):
        return cases
    for case in as_list(report.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def candidate_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in as_list(report.get("candidates")):
        if not isinstance(candidate, dict):
            continue
        key = candidate.get("key")
        if isinstance(key, str):
            candidates[key] = candidate
    return candidates


def candidate_uses_same_sign(candidate: dict[str, Any]) -> bool:
    return "same_sign_pressure_action" in as_list(candidate.get("row_sources"))


def depends_on_pressure_update(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("depends_on_pressure_update_values_in_current_artifact"))


def candidate_dependency_summary(
    predicate_report: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidate_map(predicate_report)
    same_sign_keys = [
        key for key, candidate in candidates.items() if candidate_uses_same_sign(candidate)
    ]
    update_dependent_same_sign_keys = [
        key
        for key in same_sign_keys
        if depends_on_pressure_update(candidates[key])
    ]
    exact_keys = as_list(predicate_report.get("exact_audited_candidate_keys"))
    complete_keys = as_list(predicate_report.get("complete_audited_candidate_keys"))
    exact_update_dependent_keys = [
        key
        for key in exact_keys
        if depends_on_pressure_update(candidates.get(str(key), {}))
    ]
    complete_update_dependent_keys = [
        key
        for key in complete_keys
        if depends_on_pressure_update(candidates.get(str(key), {}))
    ]
    complete_non_update_dependent_keys = [
        key
        for key in complete_keys
        if not depends_on_pressure_update(candidates.get(str(key), {}))
    ]
    exact_non_update_dependent_keys = [
        key
        for key in exact_keys
        if not depends_on_pressure_update(candidates.get(str(key), {}))
    ]
    preferred = predicate_report.get("preferred_next_candidate")
    preferred_key = (
        preferred.get("key")
        if isinstance(preferred, dict) and isinstance(preferred.get("key"), str)
        else None
    )
    preferred_candidate = candidates.get(preferred_key or "", {})
    return {
        "predicate_report_finding": predicate_report.get("finding"),
        "preferred_candidate_key": preferred_key,
        "preferred_candidate_depends_on_pressure_update": (
            depends_on_pressure_update(preferred_candidate)
            if preferred_candidate
            else None
        ),
        "same_sign_candidate_keys": same_sign_keys,
        "same_sign_update_dependent_candidate_keys": (
            update_dependent_same_sign_keys
        ),
        "exact_audited_candidate_keys": exact_keys,
        "complete_audited_candidate_keys": complete_keys,
        "exact_update_dependent_candidate_keys": exact_update_dependent_keys,
        "complete_update_dependent_candidate_keys": (
            complete_update_dependent_keys
        ),
        "exact_non_update_dependent_candidate_keys": (
            exact_non_update_dependent_keys
        ),
        "complete_non_update_dependent_candidate_keys": (
            complete_non_update_dependent_keys
        ),
        "all_exact_candidates_depend_on_pressure_update": (
            bool(exact_keys) and len(exact_update_dependent_keys) == len(exact_keys)
        ),
        "all_complete_candidates_depend_on_pressure_update": (
            bool(complete_keys)
            and len(complete_update_dependent_keys) == len(complete_keys)
        ),
        "candidate_dependency_details": [
            {
                "key": key,
                "row_sources": as_list(candidate.get("row_sources")),
                "finding": candidate.get("finding"),
                "production_readiness": candidate.get("production_readiness"),
                "derivation_status": candidate.get("derivation_status"),
                "depends_on_pressure_update_values_in_current_artifact": (
                    depends_on_pressure_update(candidate)
                ),
                "covers_all_audited_targets": candidate.get(
                    "covers_all_audited_targets"
                ),
                "exact_audited_coverage": candidate.get("exact_audited_coverage"),
            }
            for key, candidate in candidates.items()
        ],
    }


def proxy_gate_is_failed(finding: Any) -> bool:
    if not isinstance(finding, str):
        return True
    return not finding.endswith("_selective")


def preupdate_proxy_summary(global_selectivity: dict[str, Any]) -> dict[str, Any]:
    gate_findings = {
        key: global_selectivity.get(key) for key in GLOBAL_PROXY_GATE_KEYS
    }
    failed_gate_keys = [
        key for key, finding in gate_findings.items() if proxy_gate_is_failed(finding)
    ]
    cases = []
    for case in as_list(global_selectivity.get("cases")):
        if not isinstance(case, dict):
            continue
        cases.append(
            {
                "label": case.get("label"),
                "finding": case.get("finding"),
                "direct_target_count": case.get("direct_target_count"),
                "candidate_to_target_ratios": {
                    key: case.get(key) for key in CASE_RATIO_KEYS if key in case
                },
                "covers_targets": {
                    "sparse_or_moderate_direct_self_ratio": case.get(
                        "sparse_or_moderate_direct_self_ratio_covers_targets"
                    ),
                    "graph_local_moderate_direct_self_ratio": case.get(
                        "graph_local_moderate_direct_self_ratio_covers_targets"
                    ),
                    "pressure_action_moderate_degree": case.get(
                        "pressure_action_moderate_degree_covers_targets"
                    ),
                    "pressure_action_moderate_sum_ratio": case.get(
                        "pressure_action_moderate_sum_ratio_covers_targets"
                    ),
                    "pressure_action_self_dominant": case.get(
                        "pressure_action_self_dominant_covers_targets"
                    ),
                    "sparse_seeded_pressure_action_radius1": case.get(
                        "sparse_seeded_pressure_action_radius1_covers_targets"
                    ),
                    "sparse_seeded_pressure_action_radius2": case.get(
                        "sparse_seeded_pressure_action_radius2_covers_targets"
                    ),
                    "sparse_seeded_matrix_pressure_action_component": case.get(
                        "sparse_seeded_matrix_pressure_action_component_covers_targets"
                    ),
                },
            }
        )
    return {
        "global_selectivity_finding": global_selectivity.get("finding"),
        "gate_findings": gate_findings,
        "failed_gate_keys": failed_gate_keys,
        "all_preupdate_proxy_gates_failed": (
            bool(gate_findings) and len(failed_gate_keys) == len(gate_findings)
        ),
        "cases": cases,
    }


def row_by_dof(case: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in as_list(case.get("top_update_rows")):
        if not isinstance(row, dict) or not isinstance(row.get("global_dof"), int):
            continue
        out[int(row["global_dof"])] = row
    return out


def patch_profile(row: dict[str, Any] | None, key: str) -> list[Any]:
    if not isinstance(row, dict):
        return []
    profile = row.get("direct_pspg_patch_neighbor_profile")
    if not isinstance(profile, dict):
        return []
    return as_list(profile.get(key))


def cross_policy_patch_summary(
    full_toprow_provenance: dict[str, Any],
    pressure_disabled_toprow_provenance: dict[str, Any],
) -> dict[str, Any]:
    full_cases = case_map(full_toprow_provenance)
    disabled_cases = case_map(pressure_disabled_toprow_provenance)
    case_reports = []
    for label, full_case in full_cases.items():
        disabled_label = f"{label}_pressure_disabled"
        disabled_case = disabled_cases.get(disabled_label)
        isolated_dofs = as_list(
            full_case.get(
                "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs"
            )
        )
        full_rows = row_by_dof(full_case)
        disabled_rows = row_by_dof(disabled_case or {})
        isolated_profiles = []
        patch_dofs: list[Any] = []
        for dof in isolated_dofs:
            full_row = full_rows.get(dof) if isinstance(dof, int) else None
            disabled_row = disabled_rows.get(dof) if isinstance(dof, int) else None
            pressure_neighbors = patch_profile(
                disabled_row,
                "same_sign_pressure_action_top_update_neighbor_dofs",
            )
            direct_neighbors = patch_profile(
                disabled_row,
                "direct_pgrad_direct_pspg_top_neighbor_dofs",
            )
            patch_dofs.extend([dof])
            patch_dofs.extend(pressure_neighbors)
            isolated_profiles.append(
                {
                    "isolated_global_dof": dof,
                    "full_gradient_pressure_action_neighbor_dofs": patch_profile(
                        full_row, "pressure_action_neighbor_dofs"
                    ),
                    "full_gradient_direct_pgrad_row_neighbor_dofs": patch_profile(
                        full_row, "direct_pgrad_row_neighbor_dofs"
                    ),
                    "pressure_disabled_same_sign_pressure_action_neighbor_dofs": (
                        pressure_neighbors
                    ),
                    "pressure_disabled_direct_pgrad_direct_neighbor_dofs": (
                        direct_neighbors
                    ),
                }
            )
        if not isolated_dofs:
            finding = "no_full_gradient_isolated_direct_rows"
        elif disabled_case is None:
            finding = "pressure_disabled_comparison_case_missing"
        elif any(
            profile["pressure_disabled_same_sign_pressure_action_neighbor_dofs"]
            for profile in isolated_profiles
        ):
            finding = "cross_policy_patch_visible_only_after_pressure_disabled_update"
        else:
            finding = "cross_policy_patch_not_recovered"
        case_reports.append(
            {
                "label": label,
                "pressure_disabled_label": disabled_label,
                "finding": finding,
                "full_gradient_isolated_direct_global_dofs": isolated_dofs,
                "pressure_disabled_direct_patch_global_dofs": ordered_unique(
                    patch_dofs
                ),
                "pressure_disabled_same_sign_component_count": (
                    disabled_case.get(
                        "direct_pspg_same_sign_pressure_action_component_count"
                    )
                    if isinstance(disabled_case, dict)
                    else None
                ),
                "pressure_disabled_direct_coverage_global_dofs": (
                    as_list(
                        disabled_case.get(
                            "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs"
                        )
                    )
                    if isinstance(disabled_case, dict)
                    else []
                ),
                "isolated_row_profiles": isolated_profiles,
            }
        )
    return {
        "full_gradient_cross_policy_comparison_count": len(
            as_list(full_toprow_provenance.get("cross_policy_neighbor_comparisons"))
        ),
        "pressure_disabled_cross_policy_comparison_count": len(
            as_list(
                pressure_disabled_toprow_provenance.get(
                    "cross_policy_neighbor_comparisons"
                )
            )
        ),
        "cross_policy_join_field_populated": bool(
            as_list(full_toprow_provenance.get("cross_policy_neighbor_comparisons"))
            or as_list(
                pressure_disabled_toprow_provenance.get(
                    "cross_policy_neighbor_comparisons"
                )
            )
        ),
        "finding": (
            "cross_policy_patch_evidence_is_post_update_diagnostic_only"
            if any(
                case["finding"]
                == "cross_policy_patch_visible_only_after_pressure_disabled_update"
                for case in case_reports
            )
            else "no_cross_policy_patch_evidence"
        ),
        "cases": case_reports,
    }


def build_report(
    *,
    target_map: dict[str, Any],
    predicate_report: dict[str, Any],
    global_selectivity: dict[str, Any],
    full_toprow_provenance: dict[str, Any],
    pressure_disabled_toprow_provenance: dict[str, Any],
    target_map_path: Path | None = None,
    predicate_json_path: Path | None = None,
    global_selectivity_path: Path | None = None,
    toprow_provenance_path: Path | None = None,
    pressure_disabled_toprow_provenance_path: Path | None = None,
) -> dict[str, Any]:
    dependency = candidate_dependency_summary(predicate_report)
    preupdate_proxy = preupdate_proxy_summary(global_selectivity)
    cross_policy = cross_policy_patch_summary(
        full_toprow_provenance, pressure_disabled_toprow_provenance
    )
    target_counts = {
        label: len(as_list(case.get("direct_pspg_target_global_dofs")))
        for label, case in case_map(target_map).items()
    }

    exact_update_blocked = dependency[
        "all_exact_candidates_depend_on_pressure_update"
    ]
    complete_non_update_ready = bool(
        dependency["complete_non_update_dependent_candidate_keys"]
    )
    proxies_failed = preupdate_proxy["all_preupdate_proxy_gates_failed"]
    if complete_non_update_ready:
        finding = "formulation_ready_candidate_available"
        next_requirement = (
            "Replay the complete non-update-dependent candidate before adding "
            "new instrumentation."
        )
    elif exact_update_blocked and proxies_failed:
        finding = (
            "same_sign_patch_blocked_by_pressure_update_dependency_and_"
            "preupdate_proxies"
        )
        next_requirement = (
            "Do not promote the same-sign patch predicate as-is. Add a new "
            "solve-time direct PSPG pressure-gradient support/coupling "
            "provenance diagnostic that does not use top pressure-update signs."
        )
    elif exact_update_blocked:
        finding = "same_sign_patch_blocked_by_pressure_update_dependency"
        next_requirement = (
            "Replace same-sign update evidence with a pre-update support/coupling "
            "predicate, then re-run global selectivity."
        )
    else:
        finding = "same_sign_dependency_readiness_inconclusive"
        next_requirement = (
            "Regenerate predicate and selectivity artifacts before promoting a "
            "formulation candidate."
        )

    return {
        "scope": (
            "Readiness audit for using same-sign direct PSPG pressure-action "
            "patch evidence as a solve-time formulation rule."
        ),
        "source_paths": {
            "target_map_path": str(target_map_path) if target_map_path else None,
            "predicate_json_path": (
                str(predicate_json_path) if predicate_json_path else None
            ),
            "global_selectivity_path": (
                str(global_selectivity_path) if global_selectivity_path else None
            ),
            "toprow_provenance_path": (
                str(toprow_provenance_path) if toprow_provenance_path else None
            ),
            "pressure_disabled_toprow_provenance_path": (
                str(pressure_disabled_toprow_provenance_path)
                if pressure_disabled_toprow_provenance_path
                else None
            ),
        },
        "finding": finding,
        "direct_target_counts": target_counts,
        "dependency_summary": dependency,
        "preupdate_proxy_summary": preupdate_proxy,
        "cross_policy_patch_summary": cross_policy,
        "conclusion": (
            "The exact same-sign patch evidence remains a post-update diagnostic: "
            "it uses pressure-update signs from sampled top rows. Current "
            "pre-update matrix/support proxies either miss targets or select too "
            "many rows, so the remaining task is a new formulation-side "
            "provenance rule rather than another promotion of the same-sign "
            "oracle."
        ),
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        target_map=load_json(args.target_map_json),
        predicate_report=load_json(args.predicate_json),
        global_selectivity=load_json(args.global_selectivity_json),
        full_toprow_provenance=load_json(args.toprow_provenance_json),
        pressure_disabled_toprow_provenance=load_json(
            args.pressure_disabled_toprow_provenance_json
        ),
        target_map_path=args.target_map_json,
        predicate_json_path=args.predicate_json,
        global_selectivity_path=args.global_selectivity_json,
        toprow_provenance_path=args.toprow_provenance_json,
        pressure_disabled_toprow_provenance_path=(
            args.pressure_disabled_toprow_provenance_json
        ),
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
