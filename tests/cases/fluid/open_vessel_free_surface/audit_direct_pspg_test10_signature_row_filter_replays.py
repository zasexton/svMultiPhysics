#!/usr/bin/env python3
"""Summarize targeted Test10 direct PSPG signature-row topology replays."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)

REPLAYS = [
    {
        "policy": "local_schur_completion",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_schur_completion_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_local_schur_completion.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_schur_completion_pressure_update_audit_20260607.json"
        ),
    },
    {
        "policy": "local_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_local_edge_balance.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
    {
        "policy": "local_schur_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_schur_edge_balance.log",
        "audit_name": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    },
]

FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def values(record: dict[str, Any] | None) -> dict[str, Any]:
    return record if isinstance(record, dict) else {}


def worst_category_summary(
    pressure_audit: dict[str, Any] | None,
    category: str,
) -> dict[str, Any] | None:
    worst_by_category = values(pressure_audit).get("worst_by_category")
    if not isinstance(worst_by_category, dict):
        return None
    worst = worst_by_category.get(category)
    if not isinstance(worst, dict):
        return None
    return {
        "abs_pressure_delta_pa": worst.get("abs_pressure_delta_pa"),
        "point_index": worst.get("point_index"),
        "support_class": worst.get("support_class"),
        "active_fluid": worst.get("active_fluid"),
        "incident_wet_fraction_min_positive": worst.get(
            "incident_wet_fraction_min_positive"
        ),
    }


def pressure_audit_summary(pressure_audit: dict[str, Any] | None) -> dict[str, Any]:
    audit = values(pressure_audit)
    return {
        "status": audit.get("status"),
        "finding": audit.get("finding"),
        "absolute_threshold_pa": audit.get("absolute_threshold_pa"),
        "triggered_transition_count": audit.get("triggered_transition_count"),
        "active_or_wet_worst": worst_category_summary(
            pressure_audit, "active_or_wet_supported"
        ),
        "full_wet_worst": worst_category_summary(
            pressure_audit, "full_wet_supported"
        ),
        "cut_supported_worst": worst_category_summary(
            pressure_audit, "cut_supported"
        ),
    }


def topology_log_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "exists": False,
            "row_filter_log_count": 0,
            "matrix_mutated_count": 0,
            "row_filter_global_dof_counts": [],
            "row_filter_selected_local_row_counts": {},
        }

    row_filter_log_count = 0
    matrix_mutated_count = 0
    global_counts: Counter[int] = Counter()
    selected_counts: Counter[int] = Counter()
    policies: Counter[str] = Counter()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if (
            "diagnostic=cut_volume_direct_pspg_topology_policy" not in line
            or "row_filter_enabled=1" not in line
        ):
            continue
        row_filter_log_count += 1
        fields = dict(FIELD_RE.findall(line))
        if fields.get("matrix_mutated") == "1":
            matrix_mutated_count += 1
        if "policy" in fields:
            policies[fields["policy"]] += 1
        if "row_filter_global_dof_count" in fields:
            global_counts[int(fields["row_filter_global_dof_count"])] += 1
        if "row_filter_selected_local_row_count" in fields:
            selected_counts[int(fields["row_filter_selected_local_row_count"])] += 1

    return {
        "exists": True,
        "row_filter_log_count": row_filter_log_count,
        "matrix_mutated_count": matrix_mutated_count,
        "row_filter_global_dof_counts": sorted(global_counts),
        "row_filter_selected_local_row_counts": {
            str(key): selected_counts[key] for key in sorted(selected_counts)
        },
        "policies_seen": {key: policies[key] for key in sorted(policies)},
    }


def summarize_replay(artifact_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    audit_path = artifact_root / spec["audit_name"]
    log_path = artifact_root / spec["case_dir"] / spec["log_name"]
    pressure_audit = load_json(audit_path)
    pressure_summary = pressure_audit_summary(pressure_audit)
    worst = values(pressure_summary.get("active_or_wet_worst"))
    status = pressure_summary.get("status")
    return {
        "policy": spec["policy"],
        "pressure_audit_path": str(audit_path),
        "pressure_audit_exists": audit_path.exists(),
        "solver_log_path": str(log_path),
        "guard_triggered": status == "diagnostic_pressure_update_guard_triggered",
        "guard_cleared": status == "diagnostic_pressure_update_guard_no_threshold_trigger",
        "worst_active_or_wet_update_pa": worst.get("abs_pressure_delta_pa"),
        "worst_active_or_wet_point_index": worst.get("point_index"),
        "worst_active_or_wet_support_class": worst.get("support_class"),
        "pressure_update": pressure_summary,
        "topology_log": topology_log_summary(log_path),
    }


def build_report(*, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    replays = [summarize_replay(artifact_root, spec) for spec in REPLAYS]
    missing = [
        item["pressure_audit_path"]
        for item in replays
        if not item["pressure_audit_exists"]
    ] + [
        item["solver_log_path"]
        for item in replays
        if not item["topology_log"]["exists"]
    ]
    complete_replays = [
        item
        for item in replays
        if item["pressure_audit_exists"] and item["topology_log"]["exists"]
    ]
    triggered = [
        item for item in complete_replays if item["guard_triggered"]
    ]
    cleared = [item for item in complete_replays if item["guard_cleared"]]
    best = min(
        (
            item
            for item in complete_replays
            if isinstance(item["worst_active_or_wet_update_pa"], (int, float))
        ),
        key=lambda item: item["worst_active_or_wet_update_pa"],
        default=None,
    )
    global_dof_counts = sorted(
        {
            count
            for item in complete_replays
            for count in item["topology_log"]["row_filter_global_dof_counts"]
        }
    )

    if missing:
        finding = "test10_signature_row_filter_replay_family_incomplete"
        status = "regenerate_missing_replay_artifacts"
        conclusion = (
            "At least one targeted Test10 signature-row replay artifact is missing."
        )
    elif len(triggered) == len(REPLAYS):
        finding = "test10_signature_row_filter_local_modes_do_not_clear_guard"
        status = "signature_row_filter_local_modes_ruled_out_as_sufficient_fix"
        conclusion = (
            "All targeted 48-row Test10 solve-time topology replay modes still "
            "trigger the 100 Pa active/wet pressure-update guard. The combined "
            "Schur plus edge-balance mode is the least bad of the three, but it "
            "still leaves a full-wet 604.7126561932914 Pa update."
        )
    elif cleared:
        finding = "test10_signature_row_filter_local_mode_clears_guard"
        status = "targeted_replay_candidate_requires_test02_transfer_check"
        conclusion = (
            "At least one targeted Test10 signature-row replay cleared the "
            "active/wet pressure-update guard; Test02 transfer remains unproven."
        )
    else:
        finding = "test10_signature_row_filter_replay_family_ambiguous"
        status = "inspect_replay_statuses"
        conclusion = "Replay statuses are mixed or unclassified."

    return {
        "scope": (
            "Compare targeted Test10 solve-time direct PSPG topology-policy "
            "replays restricted to the exported exact-local support/coupling "
            "signature rows."
        ),
        "finding": finding,
        "status": status,
        "row_filter_global_dof_counts": global_dof_counts,
        "policies_tested": [spec["policy"] for spec in REPLAYS],
        "all_replays_trigger_guard": len(triggered) == len(REPLAYS) and not missing,
        "cleared_policies": [item["policy"] for item in cleared],
        "best_policy_by_worst_update": None if best is None else best["policy"],
        "best_worst_active_or_wet_update_pa": (
            None if best is None else best["worst_active_or_wet_update_pa"]
        ),
        "replays": replays,
        "missing_evidence": missing,
        "conclusion": conclusion,
        "next_requirement": (
            "Do not promote exact signature-row local topology replay as the "
            "formulation fix; derive a physical support/coupling rule beyond "
            "row-list mutation and continue the Test02 discriminator search."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(artifact_root=args.artifact_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
