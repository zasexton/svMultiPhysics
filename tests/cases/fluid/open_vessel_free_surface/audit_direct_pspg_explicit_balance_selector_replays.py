#!/usr/bin/env python3
"""Audit explicit direct PSPG balance-row replay selectors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_BOUNDARY_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_graph_completion_shared_row_schur_low_degree_edge_balance_"
    "deg3_boundary_provenance_20260606.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_explicit_balance_selector_replays_20260607.json"
)

VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "explicit_direct_rows",
        "description": "Balance exact audited direct PSPG target rows.",
        "support": {
            "test02": (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_"
                "gradient_graph_completion_schur_explicit_direct_rows_"
                "support_audit_20260606.json"
            ),
            "test10": (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_"
                "completion_schur_explicit_direct_rows_support_audit_"
                "20260606.json"
            ),
        },
    },
    {
        "key": "explicit_shifted_rows",
        "description": "Balance exact direct PSPG rows plus the shifted bad row.",
        "support": {
            "test02": (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_"
                "gradient_graph_completion_schur_explicit_shifted_rows_"
                "support_audit_20260606.json"
            ),
            "test10": (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_"
                "completion_schur_explicit_shifted_rows_support_audit_"
                "20260606.json"
            ),
        },
    },
    {
        "key": "explicit_cross_policy_patch",
        "description": "Balance the exported Test02 cross-policy direct patch.",
        "support": {
            "test02": (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_"
                "gradient_graph_completion_schur_explicit_cross_policy_patch_"
                "support_audit_20260606.json"
            )
        },
    },
    {
        "key": "explicit_operator_top_rows",
        "description": "Balance Test02 direct PSPG plus ghost-positive top rows.",
        "support": {
            "test02": (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_"
                "gradient_graph_completion_schur_explicit_operator_top_rows_"
                "support_audit_20260606.json"
            )
        },
    },
    {
        "key": "explicit_neighborhood_depth1",
        "description": "Balance one-hop pressure neighbors around direct rows.",
        "support": {
            "test02": (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_"
                "gradient_graph_completion_schur_explicit_neighborhood_"
                "direct_rows_support_audit_20260606.json"
            ),
            "test10": (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_"
                "completion_schur_explicit_neighborhood_direct_rows_"
                "support_audit_20260606.json"
            ),
        },
    },
    {
        "key": "explicit_neighborhood_depth2",
        "description": "Balance two-hop pressure neighbors around direct rows.",
        "support": {
            "test02": (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_"
                "gradient_graph_completion_schur_explicit_neighborhood_depth2_"
                "direct_rows_support_audit_20260606.json"
            ),
            "test10": (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_graph_"
                "completion_schur_explicit_neighborhood_depth2_direct_rows_"
                "support_audit_20260606.json"
            ),
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify explicit row-list and current-pressure-neighborhood "
            "direct PSPG Schur/balance replay controls."
        )
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--boundary-provenance-json",
        type=Path,
        default=DEFAULT_BOUNDARY_PROVENANCE,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value_dict(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    values = record.get("values")
    return values if isinstance(values, dict) else record


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


def float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_dof_sample(value: Any) -> list[int]:
    if not isinstance(value, str) or not value or value == "none":
        return []
    out: list[int] = []
    for item in value.split("|"):
        if item == "...":
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def resolve_log_path(artifact_root: Path, solver_log: Any) -> Path | None:
    if not isinstance(solver_log, str) or not solver_log:
        return None
    path = Path(solver_log)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return artifact_root / path


def parse_log_status(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path is not None else None, "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
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
    loop = None
    match = re.search(
        r"loop\.run\(\) returned success=(\d+) steps_taken=(\d+).*?message='([^']*)'",
        text,
    )
    if match:
        loop = {
            "success": bool(int(match.group(1))),
            "steps_taken": int(match.group(2)),
            "message": match.group(3),
        }
    return {
        "path": str(path),
        "exists": True,
        "nonlinear": nonlinear,
        "loop": loop,
        "has_nonlinear_failure": "nonlinear solve did not converge" in text,
    }


def case_outcome(
    *,
    pressure_update: dict[str, Any],
    log_status: dict[str, Any],
) -> tuple[str, str]:
    triggered = int_or_none(pressure_update.get("triggered"))
    if triggered == 0:
        return "accepted_guard_not_triggered", "guard_cleared"
    if triggered == 1:
        return "accepted_guard_triggered", "guard_still_triggered"
    nonlinear = log_status.get("nonlinear")
    if isinstance(nonlinear, dict) and nonlinear.get("converged") is False:
        return "nonlinear_failed_before_acceptance", "nonlinear_failed"
    if log_status.get("has_nonlinear_failure"):
        return "nonlinear_failed_before_acceptance", "nonlinear_failed"
    return "no_accepted_update_observed", "case_inconclusive"


def summarize_support_case(
    *,
    artifact_root: Path,
    path: Path,
    label: str,
) -> dict[str, Any]:
    report = load_json(path)
    graph = value_dict(report.get("latest_pressure_graph_completion"))
    pressure_update = value_dict(report.get("latest_accepted_pressure_update"))
    log_status = parse_log_status(resolve_log_path(artifact_root, report.get("solver_log")))
    outcome, finding = case_outcome(
        pressure_update=pressure_update,
        log_status=log_status,
    )
    nonlinear = log_status.get("nonlinear") if isinstance(log_status, dict) else None
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "finding": finding,
        "outcome": outcome,
        "mode": graph.get("mode"),
        "candidate_row_count": int_or_none(graph.get("candidate_row_count")),
        "balance_candidate_row_count": int_or_none(
            graph.get("balance_candidate_row_count")
        ),
        "edge_count": int_or_none(graph.get("edge_count")),
        "shared_row_schur_edge_count": int_or_none(
            graph.get("shared_row_schur_edge_count")
        ),
        "existing_balance_edge_count": int_or_none(
            graph.get("existing_balance_edge_count")
        ),
        "explicit_balance_requested_global_dofs": parse_dof_sample(
            graph.get("explicit_balance_requested_global_dofs")
        ),
        "balance_candidate_global_dofs": parse_dof_sample(
            graph.get("balance_candidate_global_dofs")
        ),
        "accepted_pressure_update_pa": float_or_none(
            pressure_update.get("global_abs_pressure_delta_pa")
        ),
        "local_worst_dof": int_or_none(pressure_update.get("local_worst_dof")),
        "threshold_pa": float_or_none(pressure_update.get("threshold_pa")),
        "triggered": int_or_none(pressure_update.get("triggered")),
        "newton_iterations": (
            int_or_none(nonlinear.get("newton_iterations"))
            if isinstance(nonlinear, dict)
            else None
        ),
        "final_residual_norm": (
            float_or_none(nonlinear.get("final_residual_norm"))
            if isinstance(nonlinear, dict)
            else None
        ),
    }


def variant_is_ruled_out(variant: dict[str, Any]) -> bool:
    cases = variant.get("cases")
    if not isinstance(cases, list) or not cases:
        return False
    return all(
        isinstance(case, dict)
        and case.get("outcome")
        in {"accepted_guard_triggered", "nonlinear_failed_before_acceptance"}
        for case in cases
    )


def build_report(
    *,
    artifact_root: Path,
    boundary_provenance_path: Path,
    variant_specs: tuple[dict[str, Any], ...] = VARIANT_SPECS,
) -> dict[str, Any]:
    missing_paths: list[str] = []
    boundary_provenance: dict[str, Any] | None = None
    if boundary_provenance_path.exists():
        boundary_provenance = load_json(boundary_provenance_path)
    else:
        missing_paths.append(str(boundary_provenance_path))

    variants: list[dict[str, Any]] = []
    for spec in variant_specs:
        cases = []
        support = spec.get("support", {})
        for label, filename in support.items():
            path = artifact_root / filename
            if not path.exists():
                missing_paths.append(str(path))
                continue
            cases.append(
                summarize_support_case(
                    artifact_root=artifact_root,
                    path=path,
                    label=label,
                )
            )
        variants.append(
            {
                "key": spec.get("key"),
                "description": spec.get("description"),
                "cases": cases,
                "ruled_out": bool(cases) and all(
                    case.get("outcome")
                    in {
                        "accepted_guard_triggered",
                        "nonlinear_failed_before_acceptance",
                    }
                    for case in cases
                ),
            }
        )

    if missing_paths:
        return {
            "finding": "direct_pspg_explicit_balance_selector_replays_missing_evidence",
            "status": "missing_explicit_balance_replay_evidence",
            "missing_paths": sorted(set(missing_paths)),
            "variants": variants,
            "next_requirement": (
                "Regenerate explicit balance-row replay and boundary provenance "
                "artifacts before classifying row-list balance selectors."
            ),
        }

    boundary_miss = (
        boundary_provenance.get("finding")
        == "latest_bad_rows_can_be_candidates_without_balance_coverage"
        and boundary_provenance.get("boundary_topology_finding")
        == "boundary_top_update_candidates_missing_balance"
    )
    ruled_out_by_key = {
        str(variant.get("key")): variant_is_ruled_out(variant)
        for variant in variants
    }
    row_lists_ruled_out = all(
        ruled_out_by_key.get(key, False)
        for key in (
            "explicit_direct_rows",
            "explicit_shifted_rows",
            "explicit_cross_policy_patch",
            "explicit_operator_top_rows",
        )
    )
    neighborhoods_ruled_out = all(
        ruled_out_by_key.get(key, False)
        for key in ("explicit_neighborhood_depth1", "explicit_neighborhood_depth2")
    )
    all_ruled_out = boundary_miss and row_lists_ruled_out and neighborhoods_ruled_out
    return {
        "finding": (
            "direct_pspg_explicit_balance_selectors_rule_out_row_lists_and_pressure_neighborhoods"
            if all_ruled_out
            else "direct_pspg_explicit_balance_selectors_inconclusive"
        ),
        "status": (
            "explicit_balance_selectors_ruled_out"
            if all_ruled_out
            else "explicit_balance_selectors_incomplete"
        ),
        "scope": (
            "Classify explicit direct PSPG balance-row lists, physical top-row "
            "lists, and current-pressure-neighborhood balance selectors."
        ),
        "boundary_provenance": {
            "path": str(boundary_provenance_path),
            "finding": boundary_provenance.get("finding"),
            "boundary_topology_finding": boundary_provenance.get(
                "boundary_topology_finding"
            ),
            "boundary_topology_finding_counts": boundary_provenance.get(
                "boundary_topology_finding_counts"
            ),
        },
        "ruleout_flags": {
            "boundary_balance_predicate_misses_latest_bad_rows": boundary_miss,
            "explicit_row_lists_ruled_out": row_lists_ruled_out,
            "current_pressure_neighborhoods_ruled_out": neighborhoods_ruled_out,
        },
        "ruled_out_by_variant": ruled_out_by_key,
        "variants": variants,
        "next_requirement": (
            "Do not promote explicit direct-row lists, shifted-row lists, exact "
            "operator top-row lists, cross-policy patch seeds, or one/two-ring "
            "current-pressure-neighborhood balance selectors. The remaining "
            "rule must derive boundary/support coverage inside the direct PSPG "
            "pressure-gradient formulation with a conditioning guard for the "
            "Test02 nonlinear branch."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        boundary_provenance_path=args.boundary_provenance_json,
    )
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
