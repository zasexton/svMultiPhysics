#!/usr/bin/env python3
"""Audit active-support pressure graph-completion replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_active_support_completion_replays_20260607.json"
)

VARIANTS: dict[str, dict[str, Any]] = {
    "active_support_neigh64": {
        "description": "Least-selector active-support completion capped at 64 active neighbors.",
        "expected_max_active_neighbors": 64,
        "cases": {
            "test02": {
                "pressure": (
                    "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                    "graph_completion_active_support_neigh64_leastselector_"
                    "pressure_update_audit_20260606.json"
                ),
                "support": (
                    "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                    "graph_completion_active_support_neigh64_leastselector_"
                    "support_audit_20260606.json"
                ),
                "log": (
                    "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                    "graph_completion_active_support_neigh64_leastselector_"
                    "20260606_case/run_graph_completion_active_support_neigh64_"
                    "leastselector.log"
                ),
            },
            "test10": {
                "pressure": (
                    "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                    "graph_completion_active_support_neigh64_leastselector_"
                    "pressure_update_audit_20260606.json"
                ),
                "support": (
                    "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                    "graph_completion_active_support_neigh64_leastselector_"
                    "support_audit_20260606.json"
                ),
                "log": (
                    "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                    "graph_completion_active_support_neigh64_leastselector_"
                    "20260606_case/run_graph_completion_active_support_neigh64_"
                    "leastselector.log"
                ),
            },
        },
    },
    "active_support_all": {
        "description": "Least-selector active-support completion with no active-neighbor cap.",
        "expected_max_active_neighbors": -1,
        "cases": {
            "test02": {
                "pressure": (
                    "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                    "graph_completion_active_support_all_leastselector_"
                    "pressure_update_audit_20260606.json"
                ),
                "support": (
                    "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                    "graph_completion_active_support_all_leastselector_"
                    "support_audit_20260606.json"
                ),
                "log": (
                    "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                    "graph_completion_active_support_all_leastselector_"
                    "20260606_case/run_graph_completion_active_support_all_"
                    "leastselector.log"
                ),
            },
            "test10": {
                "pressure": (
                    "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                    "graph_completion_active_support_all_leastselector_"
                    "pressure_update_audit_20260606.json"
                ),
                "support": (
                    "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                    "graph_completion_active_support_all_leastselector_"
                    "support_audit_20260606.json"
                ),
                "log": (
                    "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                    "graph_completion_active_support_all_leastselector_"
                    "20260606_case/run_graph_completion_active_support_all_"
                    "leastselector.log"
                ),
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def value_dict(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    nested = record.get("values")
    return nested if isinstance(nested, dict) else record


def number(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def integer(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def pressure_summary(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if not isinstance(report, dict):
        return {"path": str(path), "exists": path.exists()}
    transitions = report.get("transitions")
    first = transitions[0] if isinstance(transitions, list) and transitions else {}
    if not isinstance(first, dict):
        first = {}
    max_by_category = first.get("max_by_category")
    if not isinstance(max_by_category, dict):
        max_by_category = {}
    active_wet = max_by_category.get("active_or_wet_supported")
    if not isinstance(active_wet, dict):
        active_wet = {}
    stats_by_category = first.get("delta_statistics_by_category")
    if not isinstance(stats_by_category, dict):
        stats_by_category = {}

    def category_update(name: str) -> float | None:
        stats = stats_by_category.get(name)
        if isinstance(stats, dict):
            return number(stats, "max_abs_delta_pa")
        max_record = max_by_category.get(name)
        if isinstance(max_record, dict):
            return number(max_record, "abs_pressure_delta_pa")
        return None

    return {
        "path": str(path),
        "exists": path.exists(),
        "status": report.get("status"),
        "finding": report.get("finding"),
        "absolute_threshold_pa": report.get("absolute_threshold_pa"),
        "guard_triggered": (
            report.get("status") == "diagnostic_pressure_update_guard_triggered"
        ),
        "worst_active_or_wet_update_pa": number(
            active_wet, "abs_pressure_delta_pa"
        ),
        "worst_active_or_wet_point_index": integer(active_wet, "point_index"),
        "worst_active_or_wet_support_class": active_wet.get("support_class"),
        "worst_active_or_wet_pressure_delta_pa": number(
            active_wet, "pressure_delta_pa"
        ),
        "full_wet_max_abs_update_pa": category_update("full_wet_supported"),
        "cut_max_abs_update_pa": category_update("cut_supported"),
        "tiny_cut_max_abs_update_pa": category_update("tiny_cut_supported"),
    }


def support_summary(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if not isinstance(report, dict):
        return {"path": str(path), "exists": path.exists()}
    graph = value_dict(report.get("latest_pressure_graph_completion"))
    update = value_dict(report.get("latest_pressure_update_support_diagnostic"))
    return {
        "path": str(path),
        "exists": path.exists(),
        "mode": graph.get("mode"),
        "requested_mode": graph.get("requested_mode"),
        "applied": integer(graph, "applied"),
        "candidate_row_count": integer(graph, "candidate_row_count"),
        "neighbor_row_count": integer(graph, "neighbor_row_count"),
        "edge_count": integer(graph, "edge_count"),
        "edge_weight": number(graph, "edge_weight"),
        "min_completion_edge_weight": number(graph, "min_completion_edge_weight"),
        "max_completion_edge_weight": number(graph, "max_completion_edge_weight"),
        "max_active_neighbors": integer(graph, "max_active_neighbors"),
        "weak_self_candidate_count": integer(
            graph, "weak_self_candidate_count"
        ),
        "weak_coupling_candidate_count": integer(
            graph, "weak_coupling_candidate_count"
        ),
        "weak_coupling_and_self_candidate_count": integer(
            graph, "weak_coupling_and_self_candidate_count"
        ),
        "zero_coupling_candidate_count": integer(
            graph, "zero_coupling_candidate_count"
        ),
        "zero_self_candidate_count": integer(graph, "zero_self_candidate_count"),
        "candidate_with_existing_pressure_edge_count": integer(
            graph, "candidate_with_existing_pressure_edge_count"
        ),
        "candidate_with_laplacian_pressure_edge_count": integer(
            graph, "candidate_with_laplacian_pressure_edge_count"
        ),
        "max_abs_update_pa": number(update, "max_abs_update"),
        "max_update_global_dof": integer(update, "max_update_global_dof"),
        "positive_coupling_max_abs_update_pa": number(
            update, "positive_coupling_max_abs_update"
        ),
        "positive_self_max_abs_update_pa": number(
            update, "positive_self_max_abs_update"
        ),
        "weak_coupling_max_abs_update_pa": number(
            update, "weak_coupling_max_abs_update"
        ),
        "weak_self_max_abs_update_pa": number(update, "weak_self_max_abs_update"),
        "zero_coupling_max_abs_update_pa": number(
            update, "zero_coupling_max_abs_update"
        ),
    }


def log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    nonlinear = None
    for match in re.finditer(
        (
            r"nonlinear_done .*?converged=(\d+) iters=(\d+) "
            r"\|\|r\|\|=([0-9eE+\-.]+).*?"
            r"\(linear: converged=(\d+) iters=(\d+) rel=([0-9eE+\-.]+)\)"
        ),
        text,
    ):
        nonlinear = {
            "converged": bool(int(match.group(1))),
            "newton_iterations": int(match.group(2)),
            "final_residual_norm": float(match.group(3)),
            "linear_converged": bool(int(match.group(4))),
            "linear_iterations": int(match.group(5)),
            "linear_relative_residual": float(match.group(6)),
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
    return {
        "path": str(path),
        "exists": True,
        "loop_success": loop_success,
        "loop_steps_taken": loop_steps,
        "loop_message": loop_message,
        "nonlinear": nonlinear,
        "has_nonlinear_failure": "nonlinear solve did not converge" in text,
    }


def case_summary(
    *,
    artifact_root: Path,
    label: str,
    variant_key: str,
    case_spec: dict[str, str],
    expected_max_active_neighbors: int,
) -> dict[str, Any]:
    pressure = pressure_summary(artifact_root / case_spec["pressure"])
    support = support_summary(artifact_root / case_spec["support"])
    log = log_summary(artifact_root / case_spec["log"])
    nonlinear = log.get("nonlinear") if isinstance(log.get("nonlinear"), dict) else {}
    return {
        "label": label,
        "variant": variant_key,
        "expected_max_active_neighbors": expected_max_active_neighbors,
        "pressure_update": pressure,
        "support": support,
        "log": log,
        "guard_triggered": pressure.get("guard_triggered"),
        "accepted_one_step": (
            log.get("loop_success") is True
            and log.get("loop_steps_taken") == 1
            and nonlinear.get("converged") is True
            and nonlinear.get("newton_iterations") == 1
        ),
        "max_active_neighbors_matches_expected": (
            support.get("max_active_neighbors") == expected_max_active_neighbors
        ),
    }


def update_value(case: dict[str, Any]) -> float | None:
    value = case.get("pressure_update", {}).get("worst_active_or_wet_update_pa")
    return float(value) if isinstance(value, (int, float)) else None


def case_by_label(variant: dict[str, Any], label: str) -> dict[str, Any]:
    for case in variant.get("cases", []):
        if isinstance(case, dict) and case.get("label") == label:
            return case
    return {}


def safe_delta(left: float | None, right: float | None) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None


def cap_removal_summary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {variant.get("key"): variant for variant in variants}
    cap64 = by_key.get("active_support_neigh64", {})
    uncapped = by_key.get("active_support_all", {})
    by_case: dict[str, dict[str, Any]] = {}
    for label in ("test02", "test10"):
        cap_case = case_by_label(cap64, label)
        all_case = case_by_label(uncapped, label)
        cap_support = cap_case.get("support", {})
        all_support = all_case.get("support", {})
        cap_update = update_value(cap_case)
        all_update = update_value(all_case)
        by_case[label] = {
            "cap64_update_pa": cap_update,
            "uncapped_update_pa": all_update,
            "uncapped_minus_cap64_update_pa": safe_delta(all_update, cap_update),
            "cap64_neighbor_row_count": cap_support.get("neighbor_row_count"),
            "uncapped_neighbor_row_count": all_support.get("neighbor_row_count"),
            "cap64_edge_count": cap_support.get("edge_count"),
            "uncapped_edge_count": all_support.get("edge_count"),
            "cap64_was_neighbor_limited": (
                cap_support.get("max_active_neighbors") == 64
                and isinstance(cap_support.get("neighbor_row_count"), int)
                and isinstance(all_support.get("neighbor_row_count"), int)
                and all_support["neighbor_row_count"] > cap_support["neighbor_row_count"]
            ),
            "uncapped_still_guard_triggered": all_case.get("guard_triggered"),
        }
    return {
        "by_case": by_case,
        "cap64_neighbor_cap_limited_all_cases": all(
            case.get("cap64_was_neighbor_limited") is True
            for case in by_case.values()
        ),
        "uncapped_still_triggers_all_cases": all(
            case.get("uncapped_still_guard_triggered") is True
            for case in by_case.values()
        ),
    }


def build_report(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    missing_evidence: list[str] = []
    for variant_key, spec in VARIANTS.items():
        cases = []
        for label, case_spec in spec["cases"].items():
            case = case_summary(
                artifact_root=artifact_root,
                label=label,
                variant_key=variant_key,
                case_spec=case_spec,
                expected_max_active_neighbors=spec["expected_max_active_neighbors"],
            )
            cases.append(case)
            for evidence_key in ("pressure_update", "support", "log"):
                evidence = case[evidence_key]
                if isinstance(evidence, dict) and not evidence.get("exists", False):
                    missing_evidence.append(str(evidence.get("path")))
        variants.append(
            {
                "key": variant_key,
                "description": spec["description"],
                "expected_max_active_neighbors": spec["expected_max_active_neighbors"],
                "cases": cases,
                "all_cases_guard_triggered": all(
                    case.get("guard_triggered") is True for case in cases
                ),
                "all_cases_accepted_one_step": all(
                    case.get("accepted_one_step") is True for case in cases
                ),
                "all_cases_max_active_neighbors_match_expected": all(
                    case.get("max_active_neighbors_matches_expected") is True
                    for case in cases
                ),
            }
        )

    cap_removal = cap_removal_summary(variants)
    all_replays_guard_triggered = all(
        variant["all_cases_guard_triggered"] for variant in variants
    )
    all_replays_accepted_one_step = all(
        variant["all_cases_accepted_one_step"] for variant in variants
    )
    all_neighbor_settings_confirmed = all(
        variant["all_cases_max_active_neighbors_match_expected"]
        for variant in variants
    )

    case_updates_pa = {
        variant["key"]: {
            case["label"]: update_value(case) for case in variant["cases"]
        }
        for variant in variants
    }

    if missing_evidence:
        finding = "direct_pspg_active_support_completion_replays_missing_evidence"
        status = "regenerate_active_support_completion_replays"
        next_requirement = (
            "Regenerate the cap64 and uncapped active-support completion replay "
            "pressure, support, and log artifacts before classifying this path."
        )
    elif (
        all_replays_guard_triggered
        and all_replays_accepted_one_step
        and all_neighbor_settings_confirmed
        and cap_removal["uncapped_still_triggers_all_cases"]
    ):
        finding = (
            "direct_pspg_active_support_completion_replays_rule_out_raw_"
            "active_support_completion"
        )
        status = "raw_active_support_completion_directional_but_insufficient"
        next_requirement = (
            "Keep active-support completion as topology evidence, but move the "
            "fix candidate to a formulation-derived physical support/coupling "
            "rule instead of promoting the raw post-assembly all-active mutation."
        )
    else:
        finding = (
            "direct_pspg_active_support_completion_replays_need_transfer_check"
        )
        status = "active_support_completion_requires_cross_case_validation"
        next_requirement = (
            "Inspect any cleared branch against the other case and the support "
            "diagnostics before treating active-support completion as a fix."
        )

    return {
        "finding": finding,
        "status": status,
        "artifact_root": str(artifact_root),
        "missing_evidence": sorted(missing_evidence),
        "all_replays_guard_triggered": all_replays_guard_triggered,
        "all_replays_accepted_one_step": all_replays_accepted_one_step,
        "all_neighbor_settings_confirmed": all_neighbor_settings_confirmed,
        "case_updates_pa": case_updates_pa,
        "cap_removal": cap_removal,
        "variants": variants,
        "next_requirement": next_requirement,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.artifact_root)
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
