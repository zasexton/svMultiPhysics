#!/usr/bin/env python3
"""Predict local pressure-edge completion leverage from sampled row actions.

This audit is a narrow bridge between the synthetic constant-null pair-completion
patch and the saved Test02/Test10 replay rows. It does not modify the solver and
does not claim that a local edge is a production fix. It asks a smaller question:
given the logged max-row pressure action terms, would a constant-null pressure
edge from the bad row to one of its already logged pressure neighbors have
enough leverage, at a comparable scale to the existing row pressure block, to
force the current bad update away from the accepted branch?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ACTION_TERM_RE = re.compile(
    rf"(?P<local>-?\d+)/(?P<global>-?\d+)/"
    rf"m=(?P<m>{NUMBER_RE})/u=(?P<u>{NUMBER_RE})/a=(?P<a>{NUMBER_RE})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use sampled pressure row-action terms to estimate whether a "
            "constant-null pressure edge has local leverage at the max update row."
        )
    )
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--pressure-update-audit", type=Path)
    parser.add_argument("--absolute-threshold-pa", type=float)
    parser.add_argument(
        "--all-top-updates",
        action="store_true",
        help=(
            "Evaluate every parsed top_update_details row instead of only "
            "pressure_update_support_summary.max_update_detail."
        ),
    )
    parser.add_argument(
        "--max-top-updates",
        type=int,
        help="Optional cap on rows evaluated with --all-top-updates.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_action_terms(raw: str | None) -> list[dict[str, float | int]]:
    if raw is None or raw == "none":
        return []
    terms: list[dict[str, float | int]] = []
    for part in raw.split("~"):
        match = ACTION_TERM_RE.fullmatch(part.strip())
        if not match:
            raise ValueError(f"Cannot parse pressure action term: {part!r}")
        terms.append(
            {
                "local_dof": int(match.group("local")),
                "global_dof": int(match.group("global")),
                "matrix_value": float(match.group("m")),
                "update": float(match.group("u")),
                "action": float(match.group("a")),
            }
        )
    return terms


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) <= 0.0:
        return None
    return float(numerator / denominator)


def finite_abs(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return abs(parsed)


def edge_strength_class(
    *,
    edge_weight: float | None,
    diag_abs: float | None,
    row_self_abs: float | None,
) -> str:
    if edge_weight is None:
        return "unavailable"
    if diag_abs is not None and diag_abs > 0.0 and edge_weight <= diag_abs:
        return "diag_scale_or_less"
    if row_self_abs is not None and row_self_abs > 0.0 and edge_weight <= row_self_abs:
        return "row_self_scale_or_less"
    if row_self_abs is not None and row_self_abs > 0.0 and edge_weight <= 10.0 * row_self_abs:
        return "within_10x_row_self"
    return "larger_than_10x_row_self"


def build_row_report(
    detail: dict[str, Any],
    *,
    support_audit: str | None,
    pressure_update_audit: str | None,
    absolute_threshold_pa: float | None,
) -> dict[str, Any]:
    if not detail:
        raise ValueError("support audit does not contain max_update_detail")

    target_global = int(detail["global_dof"])
    target_local = int(detail["local_pressure_row"])
    target_update = float(detail["update"])
    rhs_abs = finite_abs(detail.get("rhs"))
    row_self_action_abs = finite_abs(detail.get("row_self_action"))
    diag_abs = finite_abs(detail.get("diag"))
    row_self_abs = finite_abs(detail.get("row_self"))
    row_self_offdiag_abs = finite_abs(detail.get("row_self_offdiag"))
    row_coupling_abs = finite_abs(detail.get("row_coupling"))

    pressure_terms = parse_action_terms(detail.get("pressure_action_terms"))
    target_terms = [
        term
        for term in pressure_terms
        if int(term["global_dof"]) == target_global
        or int(term["local_dof"]) == target_local
    ]
    if target_terms:
        target_update = float(target_terms[0]["update"])

    candidates: list[dict[str, Any]] = []
    for term in pressure_terms:
        if int(term["global_dof"]) == target_global:
            continue
        neighbor_update = float(term["update"])
        update_gap = target_update - neighbor_update
        if abs(update_gap) <= 0.0:
            edge_weight_for_rhs = None
            edge_weight_for_self_action = None
            edge_weight_for_diag_action = None
        else:
            edge_weight_for_rhs = (
                rhs_abs / abs(update_gap) if rhs_abs is not None else None
            )
            edge_weight_for_self_action = (
                row_self_action_abs / abs(update_gap)
                if row_self_action_abs is not None
                else None
            )
            diag_action_abs = (
                diag_abs * abs(target_update) if diag_abs is not None else None
            )
            edge_weight_for_diag_action = (
                diag_action_abs / abs(update_gap)
                if diag_action_abs is not None
                else None
            )

        existing_edge_abs = abs(float(term["matrix_value"]))
        neighbor_below_guard = (
            absolute_threshold_pa is not None
            and abs(neighbor_update) <= absolute_threshold_pa
        )
        same_sign = target_update * neighbor_update > 0.0
        neighbor_abs_below_target = abs(neighbor_update) < abs(target_update)
        strength = edge_strength_class(
            edge_weight=edge_weight_for_rhs,
            diag_abs=diag_abs,
            row_self_abs=row_self_abs,
        )
        candidates.append(
            {
                "neighbor_local_dof": int(term["local_dof"]),
                "neighbor_global_dof": int(term["global_dof"]),
                "neighbor_update": neighbor_update,
                "neighbor_abs_update": abs(neighbor_update),
                "neighbor_below_guard": bool(neighbor_below_guard),
                "same_sign_as_target": bool(same_sign),
                "neighbor_abs_below_target": bool(neighbor_abs_below_target),
                "edge_would_pull_toward_lower_abs_neighbor": bool(
                    same_sign and neighbor_abs_below_target
                ),
                "target_minus_neighbor_update": float(update_gap),
                "existing_matrix_entry": float(term["matrix_value"]),
                "existing_edge_abs": float(existing_edge_abs),
                "edge_weight_for_rhs_abs": edge_weight_for_rhs,
                "edge_weight_for_self_action_abs": edge_weight_for_self_action,
                "edge_weight_for_diag_action_abs": edge_weight_for_diag_action,
                "edge_weight_for_rhs_to_existing_edge_ratio": ratio(
                    edge_weight_for_rhs, existing_edge_abs
                ),
                "edge_weight_for_rhs_to_diag_ratio": ratio(
                    edge_weight_for_rhs, diag_abs
                ),
                "edge_weight_for_rhs_to_row_self_ratio": ratio(
                    edge_weight_for_rhs, row_self_abs
                ),
                "edge_weight_for_rhs_to_row_self_offdiag_ratio": ratio(
                    edge_weight_for_rhs, row_self_offdiag_abs
                ),
                "edge_strength_class": strength,
                "plausible_below_guard_local_edge": bool(
                    neighbor_below_guard
                    and same_sign
                    and neighbor_abs_below_target
                    and strength in {"diag_scale_or_less", "row_self_scale_or_less"}
                ),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            not candidate["plausible_below_guard_local_edge"],
            not candidate["neighbor_below_guard"],
            candidate["edge_weight_for_rhs_to_row_self_ratio"]
            if candidate["edge_weight_for_rhs_to_row_self_ratio"] is not None
            else float("inf"),
            candidate["neighbor_abs_update"],
        )
    )

    plausible = [
        candidate for candidate in candidates if candidate["plausible_below_guard_local_edge"]
    ]
    below_guard = [candidate for candidate in candidates if candidate["neighbor_below_guard"]]
    target_violates_guard = (
        absolute_threshold_pa is not None
        and abs(target_update) > absolute_threshold_pa
    )
    if plausible:
        finding = "local_pressure_edge_completion_plausible_for_sampled_max_row"
    elif below_guard:
        finding = "below_guard_neighbors_exist_but_need_larger_than_row_self_edge"
    elif target_violates_guard:
        finding = "no_logged_pressure_neighbor_below_guard_for_sampled_max_row"
    else:
        finding = "max_row_not_above_guard_or_threshold_unavailable"

    return {
        "support_audit": support_audit,
        "pressure_update_audit": pressure_update_audit,
        "absolute_threshold_pa": absolute_threshold_pa,
        "finding": finding,
        "limitation": (
            "Uses the bounded pressure_action_terms logged for one sampled row; "
            "it is a local row-action predictor, not a rebuilt global solve."
        ),
        "target": {
            "global_dof": target_global,
            "local_pressure_row": target_local,
            "update": target_update,
            "abs_update": abs(target_update),
            "violates_guard": bool(target_violates_guard),
            "rhs_abs": rhs_abs,
            "row_self_action_abs": row_self_action_abs,
            "diag_abs": diag_abs,
            "row_self_abs": row_self_abs,
            "row_self_offdiag_abs": row_self_offdiag_abs,
            "row_coupling_abs": row_coupling_abs,
            "self_to_coupling_abs_ratio": ratio(row_self_abs, row_coupling_abs),
        },
        "parsed_pressure_term_count": len(pressure_terms),
        "candidate_edge_count": len(candidates),
        "below_guard_neighbor_candidate_count": len(below_guard),
        "plausible_below_guard_local_edge_count": len(plausible),
        "best_candidate_edges": candidates[:8],
    }


def absolute_threshold_from_inputs(
    *,
    pressure_update_report: dict[str, Any] | None,
    absolute_threshold_pa: float | None,
) -> float | None:
    if absolute_threshold_pa is not None:
        return absolute_threshold_pa
    if pressure_update_report is None:
        return None
    threshold = pressure_update_report.get("absolute_threshold_pa")
    if threshold is None:
        return None
    return float(threshold)


def pressure_update_audit_path(
    pressure_update_report: dict[str, Any] | None,
) -> str | None:
    if pressure_update_report is None:
        return None
    solver_log = pressure_update_report.get("solver_log")
    return str(solver_log) if solver_log is not None else None


def build_report(
    support_report: dict[str, Any],
    *,
    pressure_update_report: dict[str, Any] | None = None,
    absolute_threshold_pa: float | None = None,
    all_top_updates: bool = False,
    max_top_updates: int | None = None,
) -> dict[str, Any]:
    absolute_threshold_pa = absolute_threshold_from_inputs(
        pressure_update_report=pressure_update_report,
        absolute_threshold_pa=absolute_threshold_pa,
    )
    summary = support_report.get("pressure_update_support_summary") or {}
    max_detail = summary.get("max_update_detail") or {}
    top_details = summary.get("top_update_details")
    if not isinstance(top_details, list):
        top_details = []
    support_audit = support_report.get("solver_log")
    pressure_update_audit = pressure_update_audit_path(pressure_update_report)

    if not all_top_updates:
        return build_row_report(
            max_detail,
            support_audit=support_audit,
            pressure_update_audit=pressure_update_audit,
            absolute_threshold_pa=absolute_threshold_pa,
        )

    details = [row for row in top_details if isinstance(row, dict)]
    if max_top_updates is not None:
        details = details[: max(0, max_top_updates)]
    if not details and max_detail:
        details = [max_detail]
    row_reports = [
        build_row_report(
            detail,
            support_audit=support_audit,
            pressure_update_audit=pressure_update_audit,
            absolute_threshold_pa=absolute_threshold_pa,
        )
        for detail in details
    ]

    guard_violating = [
        row
        for row in row_reports
        if row.get("target", {}).get("violates_guard")
    ]
    plausible = [
        row
        for row in guard_violating
        if row.get("plausible_below_guard_local_edge_count", 0) > 0
    ]
    below_guard_neighbor = [
        row
        for row in guard_violating
        if row.get("below_guard_neighbor_candidate_count", 0) > 0
    ]
    no_below_guard_neighbor = [
        row
        for row in guard_violating
        if row.get("below_guard_neighbor_candidate_count", 0) == 0
    ]

    if not guard_violating:
        finding = "no_guard_violating_logged_top_rows_or_threshold_unavailable"
    elif len(plausible) == len(guard_violating):
        finding = "local_pressure_edge_completion_plausible_for_all_logged_top_rows"
    elif plausible:
        finding = "local_pressure_edge_completion_partial_for_logged_top_rows"
    elif below_guard_neighbor:
        finding = "below_guard_neighbors_exist_but_need_larger_edges_for_logged_top_rows"
    else:
        finding = "no_logged_pressure_neighbor_below_guard_for_logged_top_rows"

    return {
        "support_audit": support_audit,
        "pressure_update_audit": pressure_update_audit,
        "absolute_threshold_pa": absolute_threshold_pa,
        "finding": finding,
        "limitation": (
            "Uses bounded pressure_action_terms from logged top_update_details; "
            "it is an aggregate local row-action predictor, not a rebuilt global solve."
        ),
        "evaluated_row_count": len(row_reports),
        "guard_violating_row_count": len(guard_violating),
        "plausible_guard_violating_row_count": len(plausible),
        "below_guard_neighbor_guard_violating_row_count": len(below_guard_neighbor),
        "no_below_guard_neighbor_guard_violating_row_count": (
            len(no_below_guard_neighbor)
        ),
        "guard_violating_rows_without_below_guard_neighbors": [
            row["target"] for row in no_below_guard_neighbor[:8]
        ],
        "row_reports": row_reports,
    }


def main() -> int:
    args = parse_args()
    support_report = load_json(args.support_audit)
    assert support_report is not None
    pressure_update_report = load_json(args.pressure_update_audit)
    report = build_report(
        support_report,
        pressure_update_report=pressure_update_report,
        absolute_threshold_pa=args.absolute_threshold_pa,
        all_top_updates=args.all_top_updates,
        max_top_updates=args.max_top_updates,
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
