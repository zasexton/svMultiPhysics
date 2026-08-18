#!/usr/bin/env python3
"""Summarize direct PSPG graph-completion replay-family outcomes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)
CASE_KEYS = {
    "test02": "test02_step382",
    "test10": "test10_step90",
}
DIRECT_TARGET_MAX_RATIO = 10.0

DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {
        "key": "support_gap_patch_schur_only",
        "description": (
            "Support-gap patch expansion with shared-row Schur fill and no "
            "existing-edge balance."
        ),
        "outcome": (
            "test02_test10_graph_completion_support_gap_patch_schur_only_"
            "20260606_outcome.json"
        ),
    },
    {
        "key": "support_gap_patch_schur_edge_balance",
        "description": (
            "Support-gap patch expansion with shared-row Schur fill plus "
            "existing-edge balance."
        ),
        "outcome": "test02_test10_graph_completion_support_gap_patch_20260606_outcome.json",
    },
    {
        "key": "all_unconstrained_schur_edge_balance",
        "description": (
            "All unconstrained pressure rows with shared-row Schur fill plus "
            "existing-edge balance."
        ),
        "test02_support": (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "graph_completion_schur_edge_balance_all_support_audit_20260606.json"
        ),
        "test10_support": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_"
            "schur_edge_balance_all_support_audit_20260606.json"
        ),
    },
    {
        "key": "least_selector_schur_only",
        "description": (
            "Least-selector support-rank rows with shared-row Schur fill and "
            "no existing-edge balance."
        ),
        "test02_support": (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "graph_completion_shared_row_schur_leastselector_support_audit_"
            "20260606.json"
        ),
        "test10_support": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_"
            "shared_row_schur_leastselector_support_audit_20260606.json"
        ),
    },
    {
        "key": "least_selector_schur_edge_balance",
        "description": (
            "Least-selector support-rank rows with shared-row Schur fill plus "
            "existing-edge balance."
        ),
        "test02_support": (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "graph_completion_schur_edge_balance_leastselector_support_audit_"
            "20260606.json"
        ),
        "test10_support": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_"
            "schur_edge_balance_leastselector_support_audit_20260606.json"
        ),
    },
    {
        "key": "support_rank_neighborhood_depth1",
        "description": (
            "One-hop pressure-neighborhood expansion around support-rank rows "
            "with shared-row Schur fill plus existing-edge balance."
        ),
        "test02_support": (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "graph_completion_schur_edge_balance_neighborhood_support_audit_"
            "20260606.json"
        ),
        "test10_support": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_"
            "schur_edge_balance_neighborhood_support_audit_20260606.json"
        ),
    },
    {
        "key": "support_rank_neighborhood_depth2",
        "description": (
            "Two-hop pressure-neighborhood expansion around support-rank rows "
            "with shared-row Schur fill plus existing-edge balance."
        ),
        "test02_support": (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "graph_completion_schur_edge_balance_neighborhood_depth2_support_audit_"
            "20260606.json"
        ),
        "test10_support": (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_"
            "schur_edge_balance_neighborhood_depth2_support_audit_20260606.json"
        ),
    },
    {
        "key": "coupling_deficient_balance",
        "description": (
            "Least-selector Schur fill with existing-edge balance gated to "
            "velocity-coupling-deficient candidate rows."
        ),
        "outcome": (
            "test02_test10_graph_completion_shared_row_schur_coupling_edge_balance_"
            "20260606_outcome.json"
        ),
    },
    {
        "key": "low_pressure_degree_balance",
        "description": (
            "Least-selector Schur fill with existing-edge balance gated to "
            "strict low-pressure-degree candidate rows."
        ),
        "outcome": (
            "test02_test10_graph_completion_shared_row_schur_low_degree_edge_balance_"
            "deg3_20260606_outcome.json"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify the direct PSPG graph-completion replay family against "
            "short Test02/Test10 pressure-update outcomes."
        )
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def value_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        nested = value.get("values")
        if isinstance(nested, dict):
            return nested
        return value
    return {}


def target_case_map(target_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def direct_target_count(targets_by_label: dict[str, dict[str, Any]], label: str) -> int:
    return len(
        as_list(targets_by_label.get(label, {}).get("direct_pspg_target_global_dofs"))
    )


def locate_replay_log(
    *,
    artifact_root: Path,
    support_path: Path,
    support_report: dict[str, Any],
) -> Path | None:
    solver_log = support_report.get("solver_log")
    if not isinstance(solver_log, str) or not solver_log:
        return None
    if Path(solver_log).is_absolute():
        path = Path(solver_log)
        return path if path.exists() else None
    path = Path(solver_log)
    if path.parent != Path(".") and path.exists():
        return path
    path = artifact_root / solver_log
    if Path(solver_log).parent != Path(".") and path.exists():
        return path

    stem = support_path.name
    candidates = []
    for suffix in ("_support_audit_20260606.json", "_20260606_support_audit.json"):
        if stem.endswith(suffix):
            candidates.append(artifact_root / f"{stem.removesuffix(suffix)}_20260606_case")
    for case_dir in candidates:
        path = case_dir / solver_log
        if path.exists():
            return path
    matches = sorted(artifact_root.glob(f"**/{solver_log}"))
    return matches[0] if matches else None


def parse_log_status(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {
            "path": str(log_path) if log_path is not None else None,
            "exists": False,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    nonlinear = None
    for match in re.finditer(
        r"nonlinear_done .*?converged=(\d+) iters=(\d+) \|\|r\|\|=([0-9eE+\-.]+)",
        text,
    ):
        nonlinear = {
            "converged": bool(int(match.group(1))),
            "newton_iterations": int(match.group(2)),
            "final_residual_norm": float(match.group(3)),
        }
    loop_success = None
    loop_steps = None
    loop_message = None
    match = re.search(
        r"loop\.run\(\) returned success=(\d+) steps_taken=(\d+).*?message='([^']*)'",
        text,
    )
    if match:
        loop_success = bool(int(match.group(1)))
        loop_steps = int(match.group(2))
        loop_message = match.group(3)
    has_nonlinear_failure = "nonlinear solve did not converge" in text
    return {
        "path": str(log_path),
        "exists": True,
        "loop_success": loop_success,
        "loop_steps_taken": loop_steps,
        "loop_message": loop_message,
        "nonlinear": nonlinear,
        "has_nonlinear_failure": has_nonlinear_failure,
    }


def case_finding(
    *,
    outcome: str | None,
    triggered: int | None,
    candidate_count: int | None,
    direct_targets: int,
) -> str:
    ratio = (
        candidate_count / float(direct_targets)
        if candidate_count is not None and direct_targets > 0
        else None
    )
    overbroad = ratio is not None and ratio > DIRECT_TARGET_MAX_RATIO
    if outcome in {"nonlinear_failed", "nonlinear_failed_before_acceptance"}:
        return (
            "nonlinear_failed_with_overbroad_patch"
            if overbroad
            else "nonlinear_failed"
        )
    if triggered == 0:
        return "guard_cleared_with_overbroad_patch" if overbroad else "guard_cleared"
    if triggered == 1:
        return "guard_still_triggered"
    if outcome:
        return outcome
    return "case_inconclusive"


def normalized_case_from_outcome(
    *,
    label: str,
    outcome_case: dict[str, Any],
    targets_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_count = direct_target_count(targets_by_label, label)
    candidate_count = int_or_none(outcome_case.get("candidate_row_count"))
    triggered: int | None
    outcome = outcome_case.get("outcome")
    if outcome == "accepted_guard_not_triggered":
        triggered = 0
    elif outcome == "accepted_guard_triggered":
        triggered = 1
    else:
        triggered = int_or_none(outcome_case.get("triggered"))
    pressure_delta = float_or_none(outcome_case.get("global_abs_pressure_delta_pa"))
    if pressure_delta is None:
        pressure_delta = float_or_none(outcome_case.get("accepted_pressure_update_pa"))
    ratio = (
        candidate_count / float(target_count)
        if candidate_count is not None and target_count > 0
        else None
    )
    return {
        "label": label,
        "source": "outcome",
        "outcome": outcome,
        "direct_target_count": target_count,
        "candidate_row_count": candidate_count,
        "candidate_to_direct_target_ratio": ratio,
        "balance_candidate_row_count": int_or_none(
            outcome_case.get("balance_candidate_row_count")
        ),
        "edge_count": int_or_none(outcome_case.get("edge_count")),
        "shared_row_schur_edge_count": int_or_none(
            outcome_case.get("shared_row_schur_edge_count")
        ),
        "existing_balance_edge_count": int_or_none(
            outcome_case.get("existing_balance_edge_count")
        ),
        "accepted_pressure_update_pa": pressure_delta,
        "threshold_pa": float_or_none(
            outcome_case.get("accepted_pressure_update_threshold_pa")
        )
        or float_or_none(outcome_case.get("threshold_pa")),
        "triggered": triggered,
        "local_worst_dof": int_or_none(outcome_case.get("local_worst_dof"))
        or int_or_none(outcome_case.get("worst_global_dof")),
        "newton_iterations": int_or_none(outcome_case.get("newton_iterations")),
        "final_residual_norm": float_or_none(outcome_case.get("final_residual_norm")),
        "finding": case_finding(
            outcome=outcome if isinstance(outcome, str) else None,
            triggered=triggered,
            candidate_count=candidate_count,
            direct_targets=target_count,
        ),
    }


def normalized_case_from_support(
    *,
    label: str,
    support_path: Path,
    artifact_root: Path,
    support_report: dict[str, Any],
    targets_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_count = direct_target_count(targets_by_label, label)
    graph = value_dict(support_report.get("latest_pressure_graph_completion"))
    accepted = value_dict(support_report.get("latest_accepted_pressure_update"))
    log_status = parse_log_status(
        locate_replay_log(
            artifact_root=artifact_root,
            support_path=support_path,
            support_report=support_report,
        )
    )
    nonlinear = value_dict(log_status.get("nonlinear"))
    candidate_count = int_or_none(graph.get("candidate_row_count"))
    triggered = int_or_none(accepted.get("triggered"))
    pressure_delta = float_or_none(accepted.get("global_abs_pressure_delta_pa"))
    outcome: str | None
    if triggered == 0:
        outcome = "accepted_guard_not_triggered"
    elif triggered == 1:
        outcome = "accepted_guard_triggered"
    elif nonlinear.get("converged") is False or log_status.get("has_nonlinear_failure"):
        outcome = "nonlinear_failed_before_acceptance"
    elif log_status.get("loop_success") is True:
        outcome = "accepted_without_guard_sample"
    else:
        outcome = None
    ratio = (
        candidate_count / float(target_count)
        if candidate_count is not None and target_count > 0
        else None
    )
    return {
        "label": label,
        "source": "support_audit",
        "path": str(support_path),
        "exists": support_path.exists(),
        "outcome": outcome,
        "direct_target_count": target_count,
        "candidate_selector": graph.get("candidate_selector"),
        "mode": graph.get("mode"),
        "candidate_row_count": candidate_count,
        "candidate_to_direct_target_ratio": ratio,
        "balance_candidate_row_count": int_or_none(
            graph.get("balance_candidate_row_count")
        )
        or int_or_none(graph.get("existing_balance_row_count")),
        "edge_count": int_or_none(graph.get("edge_count")),
        "shared_row_schur_edge_count": int_or_none(
            graph.get("shared_row_schur_edge_count")
        ),
        "existing_balance_edge_count": int_or_none(
            graph.get("existing_balance_edge_count")
        ),
        "accepted_pressure_update_pa": pressure_delta,
        "threshold_pa": float_or_none(accepted.get("threshold_pa")),
        "triggered": triggered,
        "local_worst_dof": int_or_none(accepted.get("local_worst_dof")),
        "newton_iterations": int_or_none(nonlinear.get("newton_iterations")),
        "final_residual_norm": float_or_none(nonlinear.get("final_residual_norm")),
        "log_status": log_status,
        "finding": case_finding(
            outcome=outcome,
            triggered=triggered,
            candidate_count=candidate_count,
            direct_targets=target_count,
        ),
    }


def variant_finding(cases: list[dict[str, Any]]) -> str:
    by_label = {case.get("label"): case for case in cases}
    test02 = by_label.get("test02", {})
    test10 = by_label.get("test10", {})
    test02_finding = test02.get("finding")
    test10_finding = test10.get("finding")
    if test10_finding in {
        "guard_cleared",
        "guard_cleared_with_overbroad_patch",
    } and str(test02_finding).startswith("nonlinear_failed"):
        return "test10_clears_but_test02_unstable"
    if test10_finding in {
        "guard_cleared",
        "guard_cleared_with_overbroad_patch",
    } and test02_finding == "guard_still_triggered":
        return "test10_clears_but_test02_guard_still_triggers"
    if test02_finding == "guard_still_triggered" and test10_finding == "guard_still_triggered":
        return "both_guards_still_trigger"
    if test02_finding == "guard_still_triggered" and test10_finding in {
        "guard_cleared",
        "guard_cleared_with_overbroad_patch",
    }:
        return "test10_clears_but_test02_guard_still_triggers"
    if str(test02_finding).startswith("nonlinear_failed") and test10_finding == (
        "guard_still_triggered"
    ):
        return "test10_still_triggers_and_test02_unstable"
    if test10_finding == "guard_still_triggered":
        return "test10_guard_still_triggers"
    return "variant_inconclusive"


def variant_report(
    *,
    spec: dict[str, Any],
    artifact_root: Path,
    targets_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    evidence_paths: dict[str, str] = {}
    if "outcome" in spec:
        path = artifact_root / str(spec["outcome"])
        evidence_paths["outcome"] = str(path)
        outcome = load_json(path) if path.exists() else {}
        for label, outcome_key in CASE_KEYS.items():
            outcome_case = outcome.get(outcome_key, {})
            if not isinstance(outcome_case, dict):
                outcome_case = {}
            cases.append(
                normalized_case_from_outcome(
                    label=label,
                    outcome_case=outcome_case,
                    targets_by_label=targets_by_label,
                )
            )
    else:
        for label in CASE_KEYS:
            key = f"{label}_support"
            path = artifact_root / str(spec[key])
            evidence_paths[key] = str(path)
            support_report = load_json(path) if path.exists() else {}
            cases.append(
                normalized_case_from_support(
                    label=label,
                    support_path=path,
                    artifact_root=artifact_root,
                    support_report=support_report,
                    targets_by_label=targets_by_label,
                )
            )
    return {
        "key": spec.get("key"),
        "description": spec.get("description"),
        "evidence_paths": evidence_paths,
        "finding": variant_finding(cases),
        "cases": cases,
    }


def aggregate_finding(variants: list[dict[str, Any]]) -> str:
    findings = {variant.get("key"): variant.get("finding") for variant in variants}
    test10_clears_but_test02_bad = any(
        finding
        in {
            "test10_clears_but_test02_unstable",
            "test10_clears_but_test02_guard_still_triggers",
        }
        for finding in findings.values()
    )
    localized_misses_test10 = any(
        key
        in {
            "support_rank_neighborhood_depth1",
            "support_rank_neighborhood_depth2",
            "coupling_deficient_balance",
            "low_pressure_degree_balance",
        }
        and finding
        in {
            "both_guards_still_trigger",
            "test10_guard_still_triggers",
            "test10_still_triggers_and_test02_unstable",
        }
        for key, finding in findings.items()
    )
    schur_only_insufficient = findings.get("least_selector_schur_only") in {
        "both_guards_still_trigger",
        "test10_guard_still_triggers",
    }
    if test10_clears_but_test02_bad and localized_misses_test10 and schur_only_insufficient:
        return (
            "direct_pspg_graph_completion_replay_family_rules_out_"
            "post_assembly_selector_variants"
        )
    if test10_clears_but_test02_bad:
        return (
            "direct_pspg_graph_completion_replay_family_test10_causal_"
            "test02_unstable"
        )
    return "direct_pspg_graph_completion_replay_family_inconclusive"


def build_report(
    *,
    artifact_root: Path,
    target_map_path: Path,
    variant_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_map = load_json(target_map_path)
    targets_by_label = target_case_map(target_map)
    specs = variant_specs if variant_specs is not None else DEFAULT_VARIANTS
    variants = [
        variant_report(
            spec=spec,
            artifact_root=artifact_root,
            targets_by_label=targets_by_label,
        )
        for spec in specs
    ]
    findings_by_variant = {
        str(variant.get("key")): variant.get("finding") for variant in variants
    }
    test10_clear_variants = [
        str(variant.get("key"))
        for variant in variants
        if any(
            case.get("label") == "test10" and case.get("triggered") == 0
            for case in as_list(variant.get("cases"))
        )
    ]
    test02_unstable_variants = [
        str(variant.get("key"))
        for variant in variants
        if any(
            case.get("label") == "test02"
            and str(case.get("finding")).startswith("nonlinear_failed")
            for case in as_list(variant.get("cases"))
        )
    ]
    test10_still_trigger_variants = [
        str(variant.get("key"))
        for variant in variants
        if any(
            case.get("label") == "test10" and case.get("triggered") == 1
            for case in as_list(variant.get("cases"))
        )
    ]
    return {
        "scope": (
            "Short Test02/Test10 replay-family audit for direct PSPG pressure "
            "graph-completion Schur topology and existing-edge balance variants."
        ),
        "target_map_path": str(target_map_path),
        "direct_target_counts": {
            label: direct_target_count(targets_by_label, label)
            for label in sorted(targets_by_label)
        },
        "max_candidate_to_direct_target_ratio_for_selector": DIRECT_TARGET_MAX_RATIO,
        "finding": aggregate_finding(variants),
        "variant_count": len(variants),
        "variant_findings": findings_by_variant,
        "test10_guard_clear_variants": test10_clear_variants,
        "test02_unstable_variants": test02_unstable_variants,
        "test10_still_trigger_variants": test10_still_trigger_variants,
        "next_requirement": (
            "Move the Schur/topology and edge-balance evidence into a "
            "formulation-side direct PSPG pressure-gradient support/coupling "
            "rule; the saved post-assembly selector variants either need broad "
            "balance that destabilizes Test02 or localized gates that leave "
            "Test10 above guard."
        ),
        "variants": variants,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        target_map_path=args.target_map_json,
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
