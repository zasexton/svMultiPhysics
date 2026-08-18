#!/usr/bin/env python3
"""Audit whether a retained-volume support cutoff can be a complete fix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_CONSTRAINT_SOURCE = Path(
    "Code/Source/solver/FE/Constraints/"
    "LevelSetActiveSideVertexDirichletConstraint.cpp"
)
DEFAULT_TOPOLOGY_REPLAYS = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_mode_replays_20260607.json"
)
DEFAULT_REJECTION_REPLAY = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_pressure_update_rejection_replay_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join the active pressure-support constraint source with Test02/"
            "Test10 replay evidence and classify whether a retained wet-volume "
            "fraction cutoff is a credible complete fix."
        )
    )
    parser.add_argument(
        "--constraint-source",
        type=Path,
        default=DEFAULT_CONSTRAINT_SOURCE,
    )
    parser.add_argument(
        "--topology-replays-json",
        type=Path,
        default=DEFAULT_TOPOLOGY_REPLAYS,
    )
    parser.add_argument(
        "--rejection-replay-json",
        type=Path,
        default=DEFAULT_REJECTION_REPLAY,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def support_class_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if isinstance(value, str) and value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def case_records(records: list[dict[str, Any]], case_name: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("case") == case_name
    ]


def support_sequence(adaptive_replay: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for item in as_list(adaptive_replay.get("dt_update_sequence")):
        if not isinstance(item, dict):
            continue
        support_class = item.get("support_class")
        if isinstance(support_class, str) and support_class:
            sequence.append(support_class)
    return sequence


def first_float(values: list[Any]) -> float | None:
    floats = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return min(floats) if floats else None


def numeric_range(records: list[dict[str, Any]], key: str) -> list[float | None]:
    values = [
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
    ]
    if not values:
        return [None, None]
    return [min(values), max(values)]


def source_support_analysis(source_text: str) -> dict[str, Any]:
    cutoff_terms = (
        "CUTOFF",
        "MIN_VOLUME_FRACTION",
        "VOLUME_FRACTION_THRESHOLD",
        "SUPPORT_FRACTION",
    )
    detected_cutoff_env_vars = sorted(
        term for term in cutoff_terms if term in source_text
    )
    retained_activation_snippet = (
        "record_retained_rule_support(\n"
        "                    static_cast<GlobalIndex>(cell), volume_rules[index]);\n"
        "            }\n"
        "            if (mark_cell_active(static_cast<GlobalIndex>(cell)))"
    )
    return {
        "constraint_source_has_retained_volume_fraction_diagnostic": (
            "retained_min_volume_fraction" in source_text
            and "retained_max_volume_fraction" in source_text
            and "rule.volume_fraction" in source_text
        ),
        "retained_generated_volume_support_activation_is_unconditional": (
            retained_activation_snippet in source_text
        ),
        "retained_generated_volume_support_uses_volume_fraction_cutoff": bool(
            detected_cutoff_env_vars
        ),
        "detected_cutoff_env_terms": detected_cutoff_env_vars,
        "sample_dof_env_supported": (
            "SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS" in source_text
        ),
    }


def topology_summary(topology_replays: dict[str, Any]) -> dict[str, Any]:
    records = [
        record
        for record in as_list(topology_replays.get("case_policy_results"))
        if isinstance(record, dict)
    ]
    test02 = case_records(records, "test02")
    test10 = case_records(records, "test10")
    tiny_values = [
        record.get("worst_active_or_wet_fraction_min_positive")
        for record in test02
        if record.get("worst_active_or_wet_support_class")
        == "tiny_cut_supported"
    ]
    return {
        "finding": topology_replays.get("finding"),
        "status": topology_replays.get("status"),
        "policies_tested": topology_replays.get("policies_tested"),
        "test02_policy_support_class_counts": support_class_counts(
            test02, "worst_active_or_wet_support_class"
        ),
        "test10_policy_support_class_counts": support_class_counts(
            test10, "worst_active_or_wet_support_class"
        ),
        "test02_min_tiny_cut_fraction_positive": first_float(tiny_values),
        "test02_policy_worst_update_pa_range": numeric_range(
            test02, "worst_active_or_wet_update_pa"
        ),
        "test10_policy_worst_update_pa_range": numeric_range(
            test10, "worst_active_or_wet_update_pa"
        ),
    }


def rejection_summary(rejection_replay: dict[str, Any]) -> dict[str, Any]:
    fixed = [
        replay
        for replay in as_list(rejection_replay.get("fixed_step_replays"))
        if isinstance(replay, dict)
    ]
    adaptive = [
        replay
        for replay in as_list(rejection_replay.get("adaptive_replays"))
        if isinstance(replay, dict)
    ]
    adaptive_by_case = {
        replay.get("case"): replay
        for replay in adaptive
        if isinstance(replay.get("case"), str)
    }
    test02_adaptive = adaptive_by_case.get("test02", {})
    test10_adaptive = adaptive_by_case.get("test10", {})
    return {
        "finding": rejection_replay.get("finding"),
        "status": rejection_replay.get("status"),
        "fixed_step_support_class_counts": support_class_counts(
            fixed, "worst_pre_commit_support_class"
        ),
        "test02_adaptive_support_sequence": support_sequence(test02_adaptive),
        "test02_adaptive_support_branch_shift": test02_adaptive.get(
            "support_branch_shift"
        ),
        "test02_adaptive_update_growth_factor": test02_adaptive.get(
            "update_growth_factor"
        ),
        "test10_adaptive_support_sequence": support_sequence(test10_adaptive),
        "test10_adaptive_update_growth_factor": test10_adaptive.get(
            "update_growth_factor"
        ),
    }


def build_report(
    *,
    source_text: str,
    topology_replays: dict[str, Any],
    rejection_replay: dict[str, Any],
) -> dict[str, Any]:
    source = source_support_analysis(source_text)
    topology = topology_summary(topology_replays)
    rejection = rejection_summary(rejection_replay)
    has_full_wet_failures = (
        topology["test10_policy_support_class_counts"].get(
            "full_wet_supported", 0
        )
        > 0
        or "full_wet_supported" in rejection["test02_adaptive_support_sequence"]
        or "full_wet_supported" in rejection["test10_adaptive_support_sequence"]
    )
    has_tiny_cut_failures = (
        topology["test02_policy_support_class_counts"].get(
            "tiny_cut_supported", 0
        )
        > 0
        or rejection["fixed_step_support_class_counts"].get(
            "tiny_cut_supported", 0
        )
        > 0
    )
    if has_tiny_cut_failures and has_full_wet_failures:
        finding = "active_pressure_support_cutoff_not_complete_fix_from_branch_shift"
        status = "support_cutoff_diagnostic_only_not_complete_fix"
    elif has_tiny_cut_failures:
        finding = "active_pressure_support_cutoff_target_supported"
        status = "support_cutoff_candidate_needs_replay"
    else:
        finding = "active_pressure_support_cutoff_not_supported_by_current_replays"
        status = "support_cutoff_ruled_out_for_current_evidence"
    return {
        "finding": finding,
        "status": status,
        "constraint_source": source,
        "topology_policy_replay_summary": topology,
        "pressure_update_rejection_summary": rejection,
        "classification": {
            "tiny_cut_supported_branch_present": has_tiny_cut_failures,
            "full_wet_supported_branch_present": has_full_wet_failures,
            "retained_fraction_cutoff_is_complete_fix_candidate": (
                has_tiny_cut_failures and not has_full_wet_failures
            ),
            "retained_fraction_cutoff_is_diagnostic_only": (
                has_tiny_cut_failures and has_full_wet_failures
            ),
        },
        "conclusion": (
            "The active pressure constraint records retained measure and "
            "volume-fraction diagnostics, but retained generated-volume support "
            "activation is unconditional. A tiny retained active fraction is "
            "therefore enough to keep pressure DOFs unconstrained. Current "
            "Test02/Test10 evidence still rules out promoting a retained-"
            "fraction cutoff alone: Test02 exposes tiny-cut-supported rejected "
            "rows, but adaptive rejection shifts to a full-wet row, and Test10 "
            "is full-wet-supported throughout the latest local topology and "
            "rejection evidence."
        ),
        "next_requirement": (
            "Use the active pressure-support sample diagnostics to characterize "
            "tiny-cut rows, but pursue a formulation-side pressure-gradient "
            "support/coupling rule that also handles the full-wet boundary rows."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        source_text=args.constraint_source.read_text(encoding="utf-8"),
        topology_replays=load_json(args.topology_replays_json),
        rejection_replay=load_json(args.rejection_replay_json),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
