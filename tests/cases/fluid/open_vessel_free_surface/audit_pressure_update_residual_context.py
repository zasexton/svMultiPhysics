#!/usr/bin/env python3
"""Aggregate accepted pressure-update residual context for Test02/Test10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TEST02 = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_replay_abs_only_prune1e5_step382_runtime_guard_cut_context_transition_20260605.json"
)
DEFAULT_TEST10 = (
    DEFAULT_ARTIFACT_ROOT
    / "test10_replay_cap3_step90_runtime_guard_cut_context_transition_20260605.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether short Test02/Test10 accepted pressure-update guards "
            "occurred despite converged nonlinear residuals."
        )
    )
    parser.add_argument("--test02-json", type=Path, default=DEFAULT_TEST02)
    parser.add_argument("--test10-json", type=Path, default=DEFAULT_TEST10)
    parser.add_argument(
        "--large-ratio-threshold",
        type=float,
        default=1.0e3,
        help=(
            "Minimum pressure-update/residual-norm ratio treated as a large "
            "accepted-update residual gap."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ratio_from_transition(report: dict[str, Any]) -> dict[str, Any]:
    existing = as_dict(report.get("pressure_update_to_nonlinear_residual"))
    if existing:
        return existing
    guard = as_dict(report.get("pressure_update_guard"))
    nonlinear = as_dict(report.get("nonlinear_done"))
    update = guard.get("global_abs_pressure_delta_pa")
    residual = nonlinear.get("residual")
    residual_field = nonlinear.get("residual_field")
    out: dict[str, Any] = {
        "global_abs_pressure_delta_pa": update,
        "nonlinear_converged": nonlinear.get("converged"),
        "nonlinear_iterations": nonlinear.get("iters"),
        "linear_converged": nonlinear.get("linear_converged"),
        "linear_iterations": nonlinear.get("linear_iters"),
        "linear_relative_residual": nonlinear.get("linear_rel"),
    }
    if isinstance(update, (int, float)) and isinstance(residual, (int, float)):
        out["nonlinear_residual_norm"] = residual
        out["update_to_nonlinear_residual_norm_ratio"] = (
            float(update) / float(residual) if float(residual) > 0.0 else None
        )
    if isinstance(update, (int, float)) and isinstance(residual_field, (int, float)):
        out["nonlinear_field_residual_norm"] = residual_field
        out["update_to_nonlinear_field_residual_norm_ratio"] = (
            float(update) / float(residual_field)
            if float(residual_field) > 0.0
            else None
        )
    return out


def summarize_case(
    label: str,
    path: Path,
    report: dict[str, Any] | None,
    *,
    large_ratio_threshold: float,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "finding": "transition_residual_context_missing",
            "accepted_converged_large_update_residual_gap": False,
        }

    guard = as_dict(report.get("pressure_update_guard"))
    ratio = ratio_from_transition(report)
    ratio_value = ratio.get("update_to_nonlinear_field_residual_norm_ratio")
    if ratio_value is None:
        ratio_value = ratio.get("update_to_nonlinear_residual_norm_ratio")
    triggered = guard.get("triggered") == 1
    converged = ratio.get("nonlinear_converged") is True
    large_gap = (
        isinstance(ratio_value, (int, float))
        and float(ratio_value) >= large_ratio_threshold
    )
    accepted_gap = bool(triggered and converged and large_gap)
    if accepted_gap:
        finding = "accepted_converged_large_pressure_update_residual_gap"
    elif not triggered:
        finding = "pressure_update_guard_not_triggered"
    elif not converged:
        finding = "pressure_update_guard_without_converged_nonlinear_solve"
    else:
        finding = "pressure_update_residual_gap_below_threshold_or_missing"

    match = as_dict(report.get("runtime_offline_pressure_match"))
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "transition_status": report.get("status"),
        "transition_finding": report.get("finding"),
        "finding": finding,
        "accepted_converged_large_update_residual_gap": accepted_gap,
        "guard_triggered": triggered,
        "guard_before_accepted_step_refresh": report.get(
            "guard_before_accepted_step_refresh"
        ),
        "post_acceptance_refresh_immediate_driver_ruled_out": report.get(
            "post_acceptance_refresh_immediate_driver_ruled_out"
        ),
        "runtime_offline_pressure_match": match.get("matches"),
        "global_abs_pressure_delta_pa": guard.get("global_abs_pressure_delta_pa"),
        "support_class": guard.get("support_class"),
        "threshold_pa": guard.get("threshold_pa"),
        "nonlinear_converged": ratio.get("nonlinear_converged"),
        "nonlinear_iterations": ratio.get("nonlinear_iterations"),
        "nonlinear_residual_norm": ratio.get("nonlinear_residual_norm"),
        "nonlinear_field_residual_norm": ratio.get(
            "nonlinear_field_residual_norm"
        ),
        "update_to_nonlinear_residual_norm_ratio": ratio.get(
            "update_to_nonlinear_residual_norm_ratio"
        ),
        "update_to_nonlinear_field_residual_norm_ratio": ratio.get(
            "update_to_nonlinear_field_residual_norm_ratio"
        ),
        "large_ratio_threshold": large_ratio_threshold,
    }


def build_report(
    *,
    test02_path: Path = DEFAULT_TEST02,
    test10_path: Path = DEFAULT_TEST10,
    large_ratio_threshold: float = 1.0e3,
) -> dict[str, Any]:
    cases = []
    for label, path in (("test02", test02_path), ("test10", test10_path)):
        report = load_json(path) if path.exists() else None
        cases.append(
            summarize_case(label, path, report, large_ratio_threshold=large_ratio_threshold)
        )

    missing = [case["label"] for case in cases if not case["exists"]]
    all_large = not missing and all(
        case["accepted_converged_large_update_residual_gap"] for case in cases
    )
    all_refresh_ruled_out = not missing and all(
        case.get("post_acceptance_refresh_immediate_driver_ruled_out")
        for case in cases
    )
    if all_large:
        finding = "accepted_pressure_updates_converged_with_large_residual_gap"
        status = "residual_convergence_acceptance_gap_supported"
        next_requirement = (
            "Keep the pressure-update guard as a diagnostic safety gate, but "
            "continue treating the underlying problem as active pressure-path "
            "formulation/support consistency rather than a residual tolerance "
            "or timestep-size fix."
        )
    elif missing:
        finding = "pressure_update_residual_context_incomplete"
        status = "missing_transition_residual_context"
        next_requirement = "Regenerate missing runtime-guard transition audits."
    else:
        finding = "pressure_update_residual_context_does_not_show_large_gap"
        status = "large_residual_gap_not_supported"
        next_requirement = (
            "Inspect the case whose pressure update was not large relative to "
            "the accepted residual before classifying timestep acceptance."
        )

    return {
        "finding": finding,
        "status": status,
        "case_count": len(cases),
        "missing_cases": missing,
        "large_ratio_threshold": large_ratio_threshold,
        "all_cases_accepted_converged_large_update_residual_gap": all_large,
        "all_cases_post_acceptance_refresh_ruled_out": all_refresh_ruled_out,
        "cases": cases,
        "next_requirement": next_requirement,
        "limitations": (
            "The residual norm is the solver's nonlinear norm, not a pressure-unit "
            "residual. This audit classifies an acceptance/diagnostic gap, not a "
            "standalone physics fix."
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        test02_path=args.test02_json,
        test10_path=args.test10_json,
        large_ratio_threshold=args.large_ratio_threshold,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
