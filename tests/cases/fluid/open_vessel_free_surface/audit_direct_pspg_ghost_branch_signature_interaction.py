#!/usr/bin/env python3
"""Join ghost-branch controls with direct PSPG support-coupling signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_SIGNATURE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_support_coupling_signature_20260607.json"
)
DEFAULT_TOP_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_toprow_provenance_20260606.json"
)
DEFAULT_PRESSURE_DISABLED_TOP_PROVENANCE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_operator_toprow_provenance_pressure_disabled_20260606.json"
)
DEFAULT_TEST02_BRANCH = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_replay_abs_only_prune1e5_step382_pressure_policy_branch_shaping_20260606.json"
)
DEFAULT_TEST10_BRANCH = (
    DEFAULT_ARTIFACT_ROOT
    / "test10_replay_cap3_step90_pressure_policy_branch_shaping_20260606.json"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_ghost_branch_signature_interaction_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether pressure ghost-penalty branch evidence can narrow "
            "the remaining solve-time direct PSPG support/coupling signature."
        )
    )
    parser.add_argument("--signature-json", type=Path, default=DEFAULT_SIGNATURE)
    parser.add_argument(
        "--top-provenance-json", type=Path, default=DEFAULT_TOP_PROVENANCE
    )
    parser.add_argument(
        "--pressure-disabled-top-provenance-json",
        type=Path,
        default=DEFAULT_PRESSURE_DISABLED_TOP_PROVENANCE,
    )
    parser.add_argument("--test02-branch-json", type=Path, default=DEFAULT_TEST02_BRANCH)
    parser.add_argument("--test10-branch-json", type=Path, default=DEFAULT_TEST10_BRANCH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def case_by_label(report: dict[str, Any], label: str) -> dict[str, Any]:
    for case in as_list(report.get("cases")):
        if not isinstance(case, dict):
            continue
        if case.get("label") == label:
            return case
    return {}


def first_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def top_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in as_list(case.get("top_update_rows")) if isinstance(row, dict)]


def top_dofs(case: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for row in top_rows(case):
        dof = row.get("global_dof")
        if isinstance(dof, int):
            out.append(dof)
    return out


def update_for_dof(case: dict[str, Any], dof: int) -> float | None:
    for row in top_rows(case):
        if row.get("global_dof") == dof:
            return first_number(row.get("abs_update"))
    return None


def direct_dofs(case: dict[str, Any]) -> list[int]:
    return [
        int(value)
        for value in as_list(case.get("direct_pspg_balance_global_dofs"))
        if isinstance(value, int)
    ]


def ghost_dofs(case: dict[str, Any]) -> list[int]:
    return [
        int(value)
        for value in as_list(case.get("ghost_penalty_balance_global_dofs"))
        if isinstance(value, int)
    ]


def branch_control_summary(branch: dict[str, Any]) -> dict[str, Any]:
    controls = as_list(branch.get("controls"))
    pressure_disabled = next(
        (
            control
            for control in controls
            if isinstance(control, dict)
            and control.get("label") == "pressure_disabled"
        ),
        {},
    )
    return {
        "finding": branch.get("finding"),
        "classification_counts": branch.get("classification_counts"),
        "baseline_worst_active_update_pa": branch.get("baseline", {}).get(
            "worst_active_abs_pressure_delta_pa"
        )
        if isinstance(branch.get("baseline"), dict)
        else None,
        "pressure_disabled_classification": pressure_disabled.get("classification"),
        "pressure_disabled_still_triggers": pressure_disabled.get("still_triggers"),
        "pressure_disabled_worst_active_update_pa": pressure_disabled.get(
            "active_or_wet_supported", {}
        ).get("control_abs_pressure_delta_pa")
        if isinstance(pressure_disabled.get("active_or_wet_supported"), dict)
        else None,
        "pressure_disabled_ratio_to_baseline": pressure_disabled.get(
            "active_or_wet_supported", {}
        ).get("ratio_to_baseline")
        if isinstance(pressure_disabled.get("active_or_wet_supported"), dict)
        else None,
        "pressure_disabled_point_shift": pressure_disabled.get("point_shift"),
        "pressure_disabled_support_class_shift": pressure_disabled.get(
            "support_class_shift"
        ),
    }


def signature_case_summary(signature: dict[str, Any], label: str) -> dict[str, Any]:
    case = case_by_label(signature, label)
    return {
        "finding": case.get("finding"),
        "target_support_class_counts": case.get(
            "target_same_parent_pressure_velocity_support_class_counts"
        ),
        "exact_local_signature_selected_count": case.get(
            "exact_local_signature_selected_count"
        ),
        "exact_local_signature_selected_to_target_ratio": case.get(
            "exact_local_signature_selected_to_target_ratio"
        ),
    }


def classify_test02(
    *,
    baseline_case: dict[str, Any],
    pressure_disabled_case: dict[str, Any],
    signature_case: dict[str, Any],
    branch_summary: dict[str, Any],
) -> str:
    baseline_ghost = ghost_dofs(baseline_case)
    pressure_disabled_ghost = ghost_dofs(pressure_disabled_case)
    row_10676_baseline = update_for_dof(baseline_case, 10676)
    row_10676_disabled = update_for_dof(pressure_disabled_case, 10676)
    signature_ratio = first_number(
        signature_case.get("exact_local_signature_selected_to_target_ratio")
    )
    pressure_disabled_still_triggers = (
        branch_summary.get("pressure_disabled_still_triggers") is True
    )
    if (
        baseline_ghost
        and not pressure_disabled_ghost
        and row_10676_baseline is not None
        and row_10676_disabled is not None
        and row_10676_disabled > row_10676_baseline
        and signature_ratio is not None
        and signature_ratio > 5.0
        and pressure_disabled_still_triggers
    ):
        return "ghost_branch_shapes_test02_but_cannot_narrow_signature"
    return "ghost_branch_test02_interaction_inconclusive"


def classify_test10(
    *,
    baseline_case: dict[str, Any],
    pressure_disabled_case: dict[str, Any],
    signature_case: dict[str, Any],
    branch_summary: dict[str, Any],
) -> str:
    signature_ratio = first_number(
        signature_case.get("exact_local_signature_selected_to_target_ratio")
    )
    pressure_disabled_still_triggers = (
        branch_summary.get("pressure_disabled_still_triggers") is True
    )
    if (
        not ghost_dofs(baseline_case)
        and not ghost_dofs(pressure_disabled_case)
        and signature_ratio is not None
        and signature_ratio <= 5.0
        and pressure_disabled_still_triggers
    ):
        return "ghost_absent_test10_signature_candidate_remains_partial_fix"
    return "ghost_branch_test10_interaction_inconclusive"


def build_case_report(
    *,
    label: str,
    baseline_case: dict[str, Any],
    pressure_disabled_case: dict[str, Any],
    signature_case: dict[str, Any],
    branch_summary: dict[str, Any],
) -> dict[str, Any]:
    if label == "test02":
        finding = classify_test02(
            baseline_case=baseline_case,
            pressure_disabled_case=pressure_disabled_case,
            signature_case=signature_case,
            branch_summary=branch_summary,
        )
        persistent_rows = sorted(set(direct_dofs(baseline_case)) & set(direct_dofs(pressure_disabled_case)))
    else:
        finding = classify_test10(
            baseline_case=baseline_case,
            pressure_disabled_case=pressure_disabled_case,
            signature_case=signature_case,
            branch_summary=branch_summary,
        )
        persistent_rows = sorted(set(direct_dofs(baseline_case)) & set(direct_dofs(pressure_disabled_case)))

    return {
        "label": label,
        "finding": finding,
        "baseline_top_finding": baseline_case.get("finding"),
        "pressure_disabled_top_finding": pressure_disabled_case.get("finding"),
        "baseline_top_global_dofs": top_dofs(baseline_case),
        "pressure_disabled_top_global_dofs": top_dofs(pressure_disabled_case),
        "baseline_direct_pspg_global_dofs": direct_dofs(baseline_case),
        "baseline_ghost_penalty_global_dofs": ghost_dofs(baseline_case),
        "pressure_disabled_direct_pspg_global_dofs": direct_dofs(
            pressure_disabled_case
        ),
        "pressure_disabled_ghost_penalty_global_dofs": ghost_dofs(
            pressure_disabled_case
        ),
        "persistent_direct_pspg_global_dofs": persistent_rows,
        "row_10676_baseline_update_pa": (
            update_for_dof(baseline_case, 10676) if label == "test02" else None
        ),
        "row_10676_pressure_disabled_update_pa": (
            update_for_dof(pressure_disabled_case, 10676)
            if label == "test02"
            else None
        ),
        "signature": signature_case,
        "branch_policy": branch_summary,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    findings = {case.get("label"): case.get("finding") for case in cases}
    if findings.get("test02") == (
        "ghost_branch_shapes_test02_but_cannot_narrow_signature"
    ) and findings.get("test10") == (
        "ghost_absent_test10_signature_candidate_remains_partial_fix"
    ):
        return (
            "direct_pspg_ghost_branch_signature_interaction_rules_out_common_gate",
            "ghost_branch_is_branch_shaper_not_support_coupling_signature_fix",
        )
    return (
        "direct_pspg_ghost_branch_signature_interaction_inconclusive",
        "regenerate_join_inputs",
    )


def build_report(
    *,
    signature: dict[str, Any],
    top_provenance: dict[str, Any],
    pressure_disabled_top_provenance: dict[str, Any],
    test02_branch: dict[str, Any],
    test10_branch: dict[str, Any],
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    cases = [
        build_case_report(
            label="test02",
            baseline_case=case_by_label(top_provenance, "test02"),
            pressure_disabled_case=case_by_label(
                pressure_disabled_top_provenance, "test02_pressure_disabled"
            ),
            signature_case=signature_case_summary(signature, "test02"),
            branch_summary=branch_control_summary(test02_branch),
        ),
        build_case_report(
            label="test10",
            baseline_case=case_by_label(top_provenance, "test10"),
            pressure_disabled_case=case_by_label(
                pressure_disabled_top_provenance, "test10_pressure_disabled"
            ),
            signature_case=signature_case_summary(signature, "test10"),
            branch_summary=branch_control_summary(test10_branch),
        ),
    ]
    finding, status = aggregate_finding(cases)
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Join pressure ghost-penalty branch controls, exact top-row "
            "operator provenance, and solve-time direct PSPG support/coupling "
            "signatures for the Test02/Test10 replay windows."
        ),
        "input_paths": {
            key: str(path) for key, path in (paths or {}).items()
        },
        "cases": cases,
        "conclusion": (
            "Pressure ghost penalty remains branch-shaping evidence, but it "
            "does not provide the missing support/coupling discriminator. In "
            "Test02, pressure-disabled top-row provenance removes the ghost "
            "branch while the direct PSPG row 10676 persists and worsens; the "
            "same-parent support/coupling signature remains overbroad. In "
            "Test10, ghost support is absent from both baseline and disabled "
            "top rows, while the Test10 signature candidate is still only a "
            "partial direction because pressure-disabled still triggers the "
            "guard. The next Test02 discriminator must therefore come from the "
            "direct PSPG pressure-gradient/support path, not ghost-branch "
            "membership."
        ),
        "next_requirement": (
            "Do not use ghost-positive branch membership to narrow the Test02 "
            "support/coupling signature. Continue with a direct PSPG physical "
            "discriminator for Test02 or a targeted Test10 aggregated-signature "
            "replay, keeping pressure-policy changes diagnostic-only."
        ),
    }


def main() -> int:
    args = parse_args()
    paths = {
        "signature": args.signature_json,
        "top_provenance": args.top_provenance_json,
        "pressure_disabled_top_provenance": (
            args.pressure_disabled_top_provenance_json
        ),
        "test02_branch": args.test02_branch_json,
        "test10_branch": args.test10_branch_json,
    }
    report = build_report(
        signature=load_json(args.signature_json),
        top_provenance=load_json(args.top_provenance_json),
        pressure_disabled_top_provenance=load_json(
            args.pressure_disabled_top_provenance_json
        ),
        test02_branch=load_json(args.test02_branch_json),
        test10_branch=load_json(args.test10_branch_json),
        paths=paths,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
