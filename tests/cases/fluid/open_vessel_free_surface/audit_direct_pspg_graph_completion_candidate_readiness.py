#!/usr/bin/env python3
"""Audit graph-completion candidate breadth against direct PSPG target rows."""

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
DEFAULT_OUTCOMES = [
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_graph_completion_support_gap_patch_schur_only_20260606_outcome.json",
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_graph_completion_support_gap_local_patch_schur_only_depth1_20260606_outcome.json",
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_graph_completion_support_gap_patch_20260606_outcome.json",
]
CASE_KEYS = {
    "test02": "test02_step382",
    "test10": "test10_step90",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pre-linear-solve active pressure graph-completion candidate "
            "breadth and replay outcomes against audited direct PSPG target rows."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--outcome-json",
        action="append",
        type=Path,
        help="Graph-completion outcome JSON. Defaults to support-gap patch outcomes.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def target_case_map(target_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def number_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def case_readiness(
    *,
    label: str,
    target_case: dict[str, Any],
    outcome_case: dict[str, Any],
) -> dict[str, Any]:
    target_rows = as_list(target_case.get("direct_pspg_target_global_dofs"))
    candidate_count = number_or_none(outcome_case.get("candidate_row_count"))
    support_gap_count = number_or_none(outcome_case.get("support_gap_candidate_count"))
    patch_count = number_or_none(outcome_case.get("support_gap_patch_candidate_count"))
    target_count = len(target_rows)
    candidate_to_target_ratio = (
        candidate_count / float(target_count)
        if candidate_count is not None and target_count > 0
        else None
    )
    patch_to_target_ratio = (
        patch_count / float(target_count)
        if patch_count is not None and target_count > 0
        else None
    )
    outcome = outcome_case.get("outcome")
    clears_guard = outcome_case.get("triggered") is False and outcome_case.get(
        "accepted"
    ) is True
    nonlinear_failed = outcome in {
        "nonlinear_failed",
        "nonlinear_failed_before_acceptance",
    } or outcome_case.get("converged") is False
    overbroad = (
        candidate_to_target_ratio is not None and candidate_to_target_ratio > 10.0
    )

    if nonlinear_failed and overbroad:
        finding = "overbroad_candidate_and_nonlinear_failed"
    elif nonlinear_failed:
        finding = "candidate_nonlinear_failed"
    elif clears_guard and overbroad:
        finding = "clears_guard_but_candidate_overbroad"
    elif clears_guard:
        finding = "clears_guard"
    elif overbroad:
        finding = "candidate_overbroad_without_guard_clearance"
    else:
        finding = "candidate_result_unclassified"

    return {
        "label": label,
        "outcome": outcome,
        "accepted": outcome_case.get("accepted"),
        "converged": outcome_case.get("converged"),
        "triggered": outcome_case.get("triggered"),
        "accepted_pressure_update_pa": outcome_case.get(
            "accepted_pressure_update_pa"
        ),
        "threshold_pa": outcome_case.get("threshold_pa"),
        "final_residual_norm": outcome_case.get("final_residual_norm"),
        "worst_global_dof": outcome_case.get("worst_global_dof"),
        "direct_target_count": target_count,
        "candidate_row_count": outcome_case.get("candidate_row_count"),
        "support_gap_candidate_count": outcome_case.get(
            "support_gap_candidate_count"
        ),
        "support_gap_patch_candidate_count": outcome_case.get(
            "support_gap_patch_candidate_count"
        ),
        "balance_candidate_row_count": outcome_case.get(
            "balance_candidate_row_count"
        ),
        "edge_count": outcome_case.get("edge_count"),
        "shared_row_schur_edge_count": outcome_case.get(
            "shared_row_schur_edge_count"
        ),
        "existing_balance_edge_count": outcome_case.get(
            "existing_balance_edge_count"
        ),
        "candidate_to_direct_target_ratio": candidate_to_target_ratio,
        "support_gap_patch_to_direct_target_ratio": patch_to_target_ratio,
        "candidate_overbroad_relative_to_direct_targets": overbroad,
        "finding": finding,
    }


def outcome_report(
    *,
    outcome_path: Path,
    outcome: dict[str, Any],
    targets_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for label, outcome_key in CASE_KEYS.items():
        target_case = targets_by_label.get(label, {})
        outcome_case = outcome.get(outcome_key, {})
        if not isinstance(outcome_case, dict):
            outcome_case = {}
        cases.append(
            case_readiness(
                label=label,
                target_case=target_case,
                outcome_case=outcome_case,
            )
        )

    case_finding_counts = Counter(case["finding"] for case in cases)
    any_overbroad = any(
        case["candidate_overbroad_relative_to_direct_targets"] for case in cases
    )
    any_failed = any(
        case["finding"] in {"overbroad_candidate_and_nonlinear_failed", "candidate_nonlinear_failed"}
        for case in cases
    )
    all_guard_clear = all(case["triggered"] is False for case in cases)

    if any_failed and any_overbroad:
        finding = "candidate_overbroad_and_test02_unstable"
    elif all_guard_clear and any_overbroad:
        finding = "candidate_clears_guards_but_is_overbroad"
    elif any_overbroad:
        finding = "candidate_overbroad"
    else:
        finding = "candidate_readiness_unclassified"

    return {
        "path": str(outcome_path),
        "exists": outcome_path.exists(),
        "mode": outcome.get("mode"),
        "source_finding": outcome.get("finding"),
        "finding": finding,
        "case_finding_counts": dict(sorted(case_finding_counts.items())),
        "cases": cases,
    }


def build_report(
    *,
    target_map: dict[str, Any],
    target_map_path: Path | None,
    outcome_paths: list[Path],
) -> dict[str, Any]:
    targets_by_label = target_case_map(target_map)
    outcomes = []
    for path in outcome_paths:
        if path.exists():
            outcome = load_json(path)
        else:
            outcome = {}
        outcomes.append(
            outcome_report(
                outcome_path=path,
                outcome=outcome,
                targets_by_label=targets_by_label,
            )
        )

    overbroad_modes = [
        outcome["mode"]
        for outcome in outcomes
        if outcome["finding"]
        in {
            "candidate_overbroad_and_test02_unstable",
            "candidate_clears_guards_but_is_overbroad",
            "candidate_overbroad",
        }
    ]
    unstable_modes = [
        outcome["mode"]
        for outcome in outcomes
        if outcome["finding"] == "candidate_overbroad_and_test02_unstable"
    ]
    test10_clear_modes = [
        outcome["mode"]
        for outcome in outcomes
        if any(
            case["label"] == "test10" and case["triggered"] is False
            for case in outcome["cases"]
        )
    ]

    if unstable_modes:
        finding = "support_gap_graph_completion_selectors_overbroad_and_test02_unstable"
    elif overbroad_modes:
        finding = "support_gap_graph_completion_selectors_overbroad"
    else:
        finding = "support_gap_graph_completion_readiness_unclassified"

    return {
        "scope": (
            "Pre-linear-solve graph-completion candidate breadth compared with "
            "audited direct PSPG target rows and replay outcomes."
        ),
        "target_map_path": str(target_map_path) if target_map_path else None,
        "finding": finding,
        "outcome_count": len(outcomes),
        "overbroad_modes": overbroad_modes,
        "test02_unstable_modes": unstable_modes,
        "test10_guard_clear_modes": test10_clear_modes,
        "direct_target_counts": {
            label: len(as_list(case.get("direct_pspg_target_global_dofs")))
            for label, case in targets_by_label.items()
        },
        "next_requirement": (
            "Derive a narrower formulation-side direct PSPG pressure-gradient "
            "support/coupling rule; the current pre-linear graph-completion "
            "selectors clear or improve Test10 only by selecting broad pressure "
            "patches that fail Test02."
        ),
        "outcomes": outcomes,
    }


def main() -> int:
    args = parse_args()
    outcome_paths = args.outcome_json if args.outcome_json else DEFAULT_OUTCOMES
    target_map = load_json(args.target_map_json)
    report = build_report(
        target_map=target_map,
        target_map_path=args.target_map_json,
        outcome_paths=outcome_paths,
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
