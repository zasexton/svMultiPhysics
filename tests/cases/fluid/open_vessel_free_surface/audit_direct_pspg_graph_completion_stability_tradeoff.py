#!/usr/bin/env python3
"""Audit stability tradeoffs in direct PSPG graph-completion replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_REPLAY_FAMILY = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_graph_completion_replay_family_20260607.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_graph_completion_stability_tradeoff_20260607.json"
)

BROAD_CLEAR_VARIANTS = (
    "support_gap_patch_schur_only",
    "support_gap_patch_schur_edge_balance",
    "all_unconstrained_schur_edge_balance",
)
LEAST_SCHUR_ONLY = "least_selector_schur_only"
LEAST_SCHUR_EDGE_BALANCE = "least_selector_schur_edge_balance"
LOCALIZED_BALANCE_VARIANTS = (
    "coupling_deficient_balance",
    "low_pressure_degree_balance",
)
NEIGHBORHOOD_VARIANTS = (
    "support_rank_neighborhood_depth1",
    "support_rank_neighborhood_depth2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether the saved direct PSPG graph-completion replay "
            "family exposes a promotable shared post-assembly Schur/balance "
            "mutation, or only a stability/coverage tradeoff."
        )
    )
    parser.add_argument(
        "--replay-family-json",
        type=Path,
        default=DEFAULT_REPLAY_FAMILY,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def variants_by_key(replay_family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for variant in as_list(replay_family.get("variants")):
        if not isinstance(variant, dict):
            continue
        key = variant.get("key")
        if isinstance(key, str):
            variants[key] = variant
    return variants


def case_by_label(variant: dict[str, Any], label: str) -> dict[str, Any]:
    for case in as_list(variant.get("cases")):
        if isinstance(case, dict) and case.get("label") == label:
            return case
    return {}


def is_guard_cleared(case: dict[str, Any]) -> bool:
    finding = case.get("finding")
    outcome = case.get("outcome")
    triggered = case.get("triggered")
    return (
        finding in {"guard_cleared", "guard_cleared_with_overbroad_patch"}
        or outcome == "accepted_guard_not_triggered"
        or triggered == 0
    )


def is_guard_triggered(case: dict[str, Any]) -> bool:
    finding = case.get("finding")
    outcome = case.get("outcome")
    triggered = case.get("triggered")
    return (
        finding == "guard_still_triggered"
        or outcome == "accepted_guard_triggered"
        or triggered == 1
    )


def is_nonlinear_failed(case: dict[str, Any]) -> bool:
    finding = case.get("finding")
    outcome = case.get("outcome")
    return (
        isinstance(finding, str)
        and finding.startswith("nonlinear_failed")
    ) or outcome in {"nonlinear_failed", "nonlinear_failed_before_acceptance"}


def float_value(case: dict[str, Any], key: str) -> float | None:
    value = case.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def int_value(case: dict[str, Any], key: str) -> int | None:
    value = case.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding": case.get("finding"),
        "outcome": case.get("outcome"),
        "candidate_row_count": int_value(case, "candidate_row_count"),
        "candidate_to_direct_target_ratio": float_value(
            case, "candidate_to_direct_target_ratio"
        ),
        "accepted_pressure_update_pa": float_value(
            case, "accepted_pressure_update_pa"
        ),
        "threshold_pa": float_value(case, "threshold_pa"),
        "newton_iterations": int_value(case, "newton_iterations"),
        "final_residual_norm": float_value(case, "final_residual_norm"),
        "shared_row_schur_edge_count": int_value(
            case, "shared_row_schur_edge_count"
        ),
        "existing_balance_edge_count": int_value(
            case, "existing_balance_edge_count"
        ),
    }


def variant_pair(
    variants: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    variant = variants.get(key, {})
    return {
        "test02": case_by_label(variant, "test02"),
        "test10": case_by_label(variant, "test10"),
    }


def build_report(replay_family: dict[str, Any]) -> dict[str, Any]:
    variants = variants_by_key(replay_family)
    missing_variants = sorted(
        set(
            BROAD_CLEAR_VARIANTS
            + (LEAST_SCHUR_ONLY, LEAST_SCHUR_EDGE_BALANCE)
            + LOCALIZED_BALANCE_VARIANTS
            + NEIGHBORHOOD_VARIANTS
        )
        - set(variants)
    )
    if missing_variants:
        return {
            "finding": "direct_pspg_graph_completion_stability_tradeoff_missing_evidence",
            "status": "missing_replay_family_variants",
            "replay_family_finding": replay_family.get("finding"),
            "missing_variants": missing_variants,
            "next_requirement": (
                "Regenerate the graph-completion replay-family artifact before "
                "classifying the Schur/balance stability tradeoff."
            ),
        }

    broad_cases = []
    for key in BROAD_CLEAR_VARIANTS:
        cases = variant_pair(variants, key)
        broad_cases.append(
            {
                "key": key,
                "test02": summarize_case(cases["test02"]),
                "test10": summarize_case(cases["test10"]),
                "test02_nonlinear_failed": is_nonlinear_failed(cases["test02"]),
                "test10_guard_cleared": is_guard_cleared(cases["test10"]),
            }
        )
    broad_topology_tradeoff = all(
        item["test02_nonlinear_failed"] and item["test10_guard_cleared"]
        for item in broad_cases
    )

    least_schur = variant_pair(variants, LEAST_SCHUR_ONLY)
    least_balance = variant_pair(variants, LEAST_SCHUR_EDGE_BALANCE)
    least_selector_tradeoff = {
        "schur_only": {
            "test02": summarize_case(least_schur["test02"]),
            "test10": summarize_case(least_schur["test10"]),
            "test02_guard_triggered": is_guard_triggered(least_schur["test02"]),
            "test10_guard_triggered": is_guard_triggered(least_schur["test10"]),
            "test02_nonlinear_failed": is_nonlinear_failed(
                least_schur["test02"]
            ),
        },
        "schur_edge_balance": {
            "test02": summarize_case(least_balance["test02"]),
            "test10": summarize_case(least_balance["test10"]),
            "test02_nonlinear_failed": is_nonlinear_failed(
                least_balance["test02"]
            ),
            "test10_guard_cleared": is_guard_cleared(least_balance["test10"]),
        },
    }
    least_selector_balance_is_tradeoff = (
        least_selector_tradeoff["schur_only"]["test02_guard_triggered"]
        and least_selector_tradeoff["schur_only"]["test10_guard_triggered"]
        and not least_selector_tradeoff["schur_only"]["test02_nonlinear_failed"]
        and least_selector_tradeoff["schur_edge_balance"][
            "test02_nonlinear_failed"
        ]
        and least_selector_tradeoff["schur_edge_balance"]["test10_guard_cleared"]
    )

    localized_balance_cases = []
    for key in LOCALIZED_BALANCE_VARIANTS:
        cases = variant_pair(variants, key)
        localized_balance_cases.append(
            {
                "key": key,
                "test02": summarize_case(cases["test02"]),
                "test10": summarize_case(cases["test10"]),
                "test02_nonlinear_failed": is_nonlinear_failed(cases["test02"]),
                "test10_guard_triggered": is_guard_triggered(cases["test10"]),
            }
        )
    localized_balance_gates_fail = all(
        item["test02_nonlinear_failed"] and item["test10_guard_triggered"]
        for item in localized_balance_cases
    )

    neighborhood_cases = []
    for key in NEIGHBORHOOD_VARIANTS:
        cases = variant_pair(variants, key)
        neighborhood_cases.append(
            {
                "key": key,
                "test02": summarize_case(cases["test02"]),
                "test10": summarize_case(cases["test10"]),
                "test02_guard_triggered": is_guard_triggered(cases["test02"]),
                "test10_guard_triggered": is_guard_triggered(cases["test10"]),
            }
        )
    neighborhood_expansion_too_local = all(
        item["test02_guard_triggered"] and item["test10_guard_triggered"]
        for item in neighborhood_cases
    )

    rules_out_post_assembly_shared_fix = (
        broad_topology_tradeoff
        and least_selector_balance_is_tradeoff
        and localized_balance_gates_fail
        and neighborhood_expansion_too_local
    )
    finding = (
        "direct_pspg_graph_completion_stability_tradeoff_rules_out_post_assembly_fix"
        if rules_out_post_assembly_shared_fix
        else "direct_pspg_graph_completion_stability_tradeoff_inconclusive"
    )
    status = (
        "post_assembly_schur_balance_tradeoff_ruled_out"
        if rules_out_post_assembly_shared_fix
        else "post_assembly_schur_balance_tradeoff_incomplete"
    )
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Classify saved direct PSPG graph-completion replay variants by "
            "whether Schur fill and existing-edge balance produce a shared, "
            "stable Test02/Test10 fix."
        ),
        "replay_family_finding": replay_family.get("finding"),
        "replay_family_next_requirement": replay_family.get("next_requirement"),
        "tradeoff_flags": {
            "broad_topology_clears_test10_but_destabilizes_test02": (
                broad_topology_tradeoff
            ),
            "least_selector_schur_stable_but_insufficient_balance_clears_test10_but_destabilizes_test02": (
                least_selector_balance_is_tradeoff
            ),
            "localized_balance_gates_fail_test10_and_destabilize_test02": (
                localized_balance_gates_fail
            ),
            "support_rank_neighborhood_expansion_too_local": (
                neighborhood_expansion_too_local
            ),
        },
        "broad_clear_variants": broad_cases,
        "least_selector_tradeoff": least_selector_tradeoff,
        "localized_balance_variants": localized_balance_cases,
        "support_rank_neighborhood_variants": neighborhood_cases,
        "next_requirement": (
            "Do not promote threshold-selected post-assembly Schur fill or "
            "existing-edge balance as the formulation fix. Derive any remaining "
            "support rule inside the PSPG pressure-gradient formulation or "
            "coupled pressure-support path, with conditioning evidence for the "
            "Test02 nonlinear branch."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(load_json(args.replay_family_json))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
