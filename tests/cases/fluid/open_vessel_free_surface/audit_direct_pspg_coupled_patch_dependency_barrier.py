#!/usr/bin/env python3
"""Classify the remaining coupled-patch dependency barrier for direct PSPG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_SAME_SIGN_READINESS = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_same_sign_dependency_readiness_20260607.json"
)
DEFAULT_NO_GALERKIN_GATE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_no_galerkin_gate_relevance_20260607.json"
)
DEFAULT_ACTIVE_SUPPORT_CUTOFF = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_active_pressure_support_cutoff_relevance_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join the same-sign dependency, no-Galerkin gate, and active pressure "
            "support cutoff relevance artifacts to decide whether the remaining "
            "direct PSPG coupled-patch family needs new solve-time provenance."
        )
    )
    parser.add_argument(
        "--same-sign-readiness-json",
        type=Path,
        default=DEFAULT_SAME_SIGN_READINESS,
    )
    parser.add_argument(
        "--no-galerkin-gate-json",
        type=Path,
        default=DEFAULT_NO_GALERKIN_GATE,
    )
    parser.add_argument(
        "--active-support-cutoff-json",
        type=Path,
        default=DEFAULT_ACTIVE_SUPPORT_CUTOFF,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def same_sign_blocker_summary(report: dict[str, Any]) -> dict[str, Any]:
    dependency = as_dict(report.get("dependency_summary"))
    preupdate = as_dict(report.get("preupdate_proxy_summary"))
    cross_policy = as_dict(report.get("cross_policy_patch_summary"))
    complete_non_update = as_list(
        dependency.get("complete_non_update_dependent_candidate_keys")
    )
    exact_non_update = as_list(
        dependency.get("exact_non_update_dependent_candidate_keys")
    )
    return {
        "finding": report.get("finding"),
        "all_complete_candidates_depend_on_pressure_update": dependency.get(
            "all_complete_candidates_depend_on_pressure_update"
        ),
        "all_exact_candidates_depend_on_pressure_update": dependency.get(
            "all_exact_candidates_depend_on_pressure_update"
        ),
        "preferred_candidate_depends_on_pressure_update": dependency.get(
            "preferred_candidate_depends_on_pressure_update"
        ),
        "complete_non_update_dependent_candidate_keys": complete_non_update,
        "exact_non_update_dependent_candidate_keys": exact_non_update,
        "all_preupdate_proxy_gates_failed": preupdate.get(
            "all_preupdate_proxy_gates_failed"
        ),
        "failed_preupdate_proxy_gate_keys": as_list(
            preupdate.get("failed_gate_keys")
        ),
        "cross_policy_patch_finding": cross_policy.get("finding"),
        "cross_policy_join_field_populated": cross_policy.get(
            "cross_policy_join_field_populated"
        ),
        "pressure_disabled_patch_case_findings": {
            case.get("label"): case.get("finding")
            for case in as_list(cross_policy.get("cases"))
            if isinstance(case, dict)
        },
    }


def no_galerkin_summary(report: dict[str, Any]) -> dict[str, Any]:
    classification = as_dict(report.get("classification"))
    candidate = as_dict(report.get("formulation_candidate"))
    return {
        "finding": report.get("finding"),
        "status": report.get("status"),
        "complete_gate_candidate": classification.get("complete_gate_candidate"),
        "overlap_missing_cases": as_list(classification.get("overlap_missing_cases")),
        "overlap_partial_cases": as_list(classification.get("overlap_partial_cases")),
        "candidate_uncovered_cases": as_list(
            classification.get("candidate_uncovered_cases")
        ),
        "support_rank_mismatch_cases": as_list(
            classification.get("support_rank_mismatch_cases")
        ),
        "candidate_key": candidate.get("key"),
        "candidate_covers_all_audited_targets": candidate.get(
            "covers_all_audited_targets"
        ),
    }


def active_support_cutoff_summary(report: dict[str, Any]) -> dict[str, Any]:
    classification = as_dict(report.get("classification"))
    return {
        "finding": report.get("finding"),
        "status": report.get("status"),
        "retained_fraction_cutoff_is_complete_fix_candidate": classification.get(
            "retained_fraction_cutoff_is_complete_fix_candidate"
        ),
        "retained_fraction_cutoff_is_diagnostic_only": classification.get(
            "retained_fraction_cutoff_is_diagnostic_only"
        ),
        "tiny_cut_supported_branch_present": classification.get(
            "tiny_cut_supported_branch_present"
        ),
        "full_wet_supported_branch_present": classification.get(
            "full_wet_supported_branch_present"
        ),
    }


def same_sign_is_blocked(summary: dict[str, Any]) -> bool:
    return (
        summary.get("all_complete_candidates_depend_on_pressure_update") is True
        and summary.get("all_exact_candidates_depend_on_pressure_update") is True
        and summary.get("all_preupdate_proxy_gates_failed") is True
        and not summary.get("complete_non_update_dependent_candidate_keys")
        and not summary.get("exact_non_update_dependent_candidate_keys")
    )


def no_galerkin_is_ruled_out(summary: dict[str, Any]) -> bool:
    return (
        summary.get("finding")
        == "no_galerkin_nonpressure_gate_ruled_out_as_complete_formulation_gate"
        and summary.get("complete_gate_candidate") is False
    )


def cutoff_is_not_complete(summary: dict[str, Any]) -> bool:
    return (
        summary.get("retained_fraction_cutoff_is_complete_fix_candidate") is False
        and summary.get("retained_fraction_cutoff_is_diagnostic_only") is True
    )


def build_report(
    *,
    same_sign_readiness: dict[str, Any],
    no_galerkin_gate: dict[str, Any],
    active_support_cutoff: dict[str, Any],
    same_sign_readiness_path: Path | None = None,
    no_galerkin_gate_path: Path | None = None,
    active_support_cutoff_path: Path | None = None,
) -> dict[str, Any]:
    same_sign = same_sign_blocker_summary(same_sign_readiness)
    no_galerkin = no_galerkin_summary(no_galerkin_gate)
    cutoff = active_support_cutoff_summary(active_support_cutoff)

    same_sign_blocked = same_sign_is_blocked(same_sign)
    no_galerkin_ruled_out = no_galerkin_is_ruled_out(no_galerkin)
    support_cutoff_not_complete = cutoff_is_not_complete(cutoff)
    cross_policy_post_update_only = (
        same_sign.get("cross_policy_patch_finding")
        == "cross_policy_patch_evidence_is_post_update_diagnostic_only"
    )
    complete_non_update_ready = bool(
        same_sign.get("complete_non_update_dependent_candidate_keys")
    )

    if complete_non_update_ready:
        finding = "coupled_patch_dependency_barrier_not_present_candidate_ready"
        status = "replay_non_update_dependent_candidate"
        next_requirement = (
            "Replay the complete non-update-dependent coupled-patch candidate "
            "before adding new solve-time provenance diagnostics."
        )
    elif same_sign_blocked and no_galerkin_ruled_out and support_cutoff_not_complete:
        finding = (
            "coupled_patch_dependency_barrier_requires_solve_time_provenance"
        )
        status = "remaining_gate_requires_new_assembly_provenance_diagnostic"
        next_requirement = (
            "Add solve-time direct PSPG pressure-gradient support/coupling "
            "provenance that does not use pressure-update signs, then test "
            "whether it separates the isolated Test02 row and Test10 patch "
            "without the broad local topology policy."
        )
    elif same_sign_blocked:
        finding = "coupled_patch_dependency_barrier_incomplete_supporting_evidence"
        status = "blocked_family_needs_missing_subsignal_resolution"
        next_requirement = (
            "Finish ruling out or supporting the no-Galerkin sub-signal and "
            "active-support cutoff branch before declaring the coupled-patch "
            "family dependent on new provenance."
        )
    else:
        finding = "coupled_patch_dependency_barrier_inconclusive"
        status = "regenerate_dependency_inputs"
        next_requirement = (
            "Regenerate same-sign dependency and pre-update proxy artifacts "
            "before choosing the next formulation diagnostic."
        )

    blocker_summary = {
        "same_sign_exact_candidates_update_dependent": same_sign.get(
            "all_exact_candidates_depend_on_pressure_update"
        ),
        "same_sign_complete_candidates_update_dependent": same_sign.get(
            "all_complete_candidates_depend_on_pressure_update"
        ),
        "same_sign_has_non_update_dependent_complete_candidate": (
            complete_non_update_ready
        ),
        "preupdate_proxy_gates_all_failed": same_sign.get(
            "all_preupdate_proxy_gates_failed"
        ),
        "cross_policy_patch_is_post_update_diagnostic_only": (
            cross_policy_post_update_only
        ),
        "no_galerkin_complete_gate_ruled_out": no_galerkin_ruled_out,
        "retained_fraction_cutoff_not_complete_fix": support_cutoff_not_complete,
        "requires_new_solve_time_provenance": (
            finding
            == "coupled_patch_dependency_barrier_requires_solve_time_provenance"
        ),
    }

    return {
        "scope": (
            "Barrier audit for the remaining direct PSPG coupled-patch family "
            "after same-sign, no-Galerkin, and active-support cutoff checks."
        ),
        "source_paths": {
            "same_sign_readiness_path": (
                str(same_sign_readiness_path) if same_sign_readiness_path else None
            ),
            "no_galerkin_gate_path": (
                str(no_galerkin_gate_path) if no_galerkin_gate_path else None
            ),
            "active_support_cutoff_path": (
                str(active_support_cutoff_path)
                if active_support_cutoff_path
                else None
            ),
        },
        "finding": finding,
        "status": status,
        "blocker_summary": blocker_summary,
        "same_sign_dependency": same_sign,
        "no_galerkin_gate": no_galerkin,
        "active_support_cutoff": cutoff,
        "conclusion": (
            "The current coupled-patch evidence cannot be promoted as a "
            "production formulation rule: exact same-sign coverage is "
            "pressure-update dependent, tested pre-update proxies fail, the "
            "no-Galerkin/nonpressure sub-signal is partial, and retained-volume "
            "support cutoff evidence is diagnostic-only for Test02/Test10."
        ),
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        same_sign_readiness=load_json(args.same_sign_readiness_json),
        no_galerkin_gate=load_json(args.no_galerkin_gate_json),
        active_support_cutoff=load_json(args.active_support_cutoff_json),
        same_sign_readiness_path=args.same_sign_readiness_json,
        no_galerkin_gate_path=args.no_galerkin_gate_json,
        active_support_cutoff_path=args.active_support_cutoff_json,
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
