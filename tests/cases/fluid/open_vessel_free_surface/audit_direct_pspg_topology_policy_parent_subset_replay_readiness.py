#!/usr/bin/env python3
"""Audit readiness for direct PSPG signature-parent full-local replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_PARENT_SCOPE = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_parent_scope_20260607.json"
)
DEFAULT_STANDARD_ASSEMBLER = Path(
    "Code/Source/solver/FE/Assembly/StandardAssembler.cpp"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_topology_policy_parent_subset_replay_readiness_"
    "20260607.json"
)

SIGNATURE_REPLAYS = [
    {
        "policy": "local_schur_completion",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_schur_completion_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_local_schur_completion.log",
    },
    {
        "policy": "local_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "local_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_local_edge_balance.log",
    },
    {
        "policy": "local_schur_edge_balance",
        "case_dir": (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_20260607_case"
        ),
        "log_name": "run_direct_pspg_signature_rows_schur_edge_balance.log",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--parent-scope-json", type=Path, default=DEFAULT_PARENT_SCOPE)
    parser.add_argument("--standard-assembler", type=Path, default=DEFAULT_STANDARD_ASSEMBLER)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def convert_value(value: str) -> Any:
    if value in {"none", "None"}:
        return None
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_policy_line(line: str) -> dict[str, Any] | None:
    marker = "diagnostic=cut_volume_direct_pspg_topology_policy"
    if marker not in line:
        return None
    payload = line[line.index("diagnostic=") :]
    result: dict[str, Any] = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = convert_value(value)
    return result


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parent_cells_from_log(path: Path) -> list[int]:
    if not path.exists():
        return []
    cells: set[int] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_policy_line(line)
        if record is None:
            continue
        parent_cell = record.get("parent_cell")
        if isinstance(parent_cell, int):
            cells.add(parent_cell)
    return sorted(cells)


def csv_ranges(values: list[int]) -> str:
    if not values:
        return ""
    ranges: list[str] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def csv_values(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def source_hook_summary(text: str) -> dict[str, Any]:
    tokens = [
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELLS",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELL_IDS",
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELLS_FILTER",
        "parent_filter_enabled",
        "parent_filter_parent_cell_count",
        "parent_filter_selected=1",
        "cutVolumeDirectPspgTopologyParentCellFilter",
    ]
    return {
        "parent_cell_filter_api_present": all(token in text for token in tokens),
        "tokens": {token: token in text for token in tokens},
        "row_filter_api_present": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS" in text
        ),
        "topology_policy_api_present": (
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY" in text
        ),
    }


def parent_scope_summary(parent_scope: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parent_scope, dict):
        return {
            "exists": False,
            "finding": None,
            "status": None,
            "strict_parent_rule_subset": None,
            "broad_only_rule_weight_majority": None,
            "combined_rule_scope": None,
        }
    combined = (
        parent_scope.get("test10_parent_rule_scope", {})
        .get("local_schur_edge_balance", {})
        .get("rule_scope")
    )
    return {
        "exists": True,
        "finding": parent_scope.get("finding"),
        "status": parent_scope.get("status"),
        "strict_parent_rule_subset": parent_scope.get(
            "all_test10_signature_parent_rule_sets_are_strict_broad_subsets"
        ),
        "broad_only_rule_weight_majority": parent_scope.get(
            "all_test10_broad_only_rule_weight_share_above_half"
        ),
        "combined_rule_scope": combined if isinstance(combined, dict) else None,
    }


def summarize_replay(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    path = root / spec["case_dir"] / spec["log_name"]
    parent_cells = parent_cells_from_log(path)
    return {
        "policy": spec["policy"],
        "solver_log_path": str(path),
        "exists": path.exists(),
        "signature_parent_cell_count": len(parent_cells),
        "signature_parent_cells": parent_cells,
        "signature_parent_cells_csv": csv_values(parent_cells),
        "signature_parent_cell_ranges": csv_ranges(parent_cells),
    }


def build_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    parent_scope_json: Path = DEFAULT_PARENT_SCOPE,
    standard_assembler: Path = DEFAULT_STANDARD_ASSEMBLER,
) -> dict[str, Any]:
    replays = [summarize_replay(artifact_root, spec) for spec in SIGNATURE_REPLAYS]
    parent_scope = load_json(parent_scope_json)
    source = source_hook_summary(read_text(standard_assembler))
    scope = parent_scope_summary(parent_scope)
    missing = [replay["solver_log_path"] for replay in replays if not replay["exists"]]
    if not parent_scope_json.exists():
        missing.append(str(parent_scope_json))
    if not standard_assembler.exists():
        missing.append(str(standard_assembler))

    parent_sets = {
        tuple(replay["signature_parent_cells"])
        for replay in replays
        if replay["exists"]
    }
    shared_parent_cells = (
        list(next(iter(parent_sets))) if len(parent_sets) == 1 else []
    )
    same_parent_set_all_policies = len(parent_sets) == 1 and bool(parent_sets)
    combined = scope.get("combined_rule_scope") or {}
    overlap_attenuated = (
        isinstance(
            combined.get("signature_to_broad_overlap_topology_edge_weight_sum_fraction"),
            (int, float),
        )
        and combined["signature_to_broad_overlap_topology_edge_weight_sum_fraction"]
        < 1.0
    )

    if missing:
        finding = "direct_pspg_signature_parent_subset_replay_readiness_incomplete"
        status = "regenerate_missing_parent_subset_inputs"
        conclusion = (
            "At least one signature-row log, parent-scope artifact, or source "
            "file is missing."
        )
    elif not source["parent_cell_filter_api_present"]:
        finding = (
            "direct_pspg_signature_parent_subset_replay_blocked_by_filter_api"
        )
        status = "add_parent_cell_filter_to_topology_policy_hook"
        conclusion = (
            "The signature parent-cell set is available, but the solve-time "
            "direct PSPG topology hook cannot yet filter by parent cell while "
            "preserving full local row support."
        )
    elif (
        same_parent_set_all_policies
        and scope.get("strict_parent_rule_subset") is True
        and scope.get("broad_only_rule_weight_majority") is True
        and overlap_attenuated
    ):
        finding = "direct_pspg_signature_parent_subset_replay_ready"
        status = "run_signature_parent_full_local_replay"
        conclusion = (
            "The existing row-filter replay cannot isolate the parent/rule "
            "subset question because it attenuates local matrices even on "
            "overlapping parent cells. The FE topology-policy hook now exposes "
            "a parent-cell filter, and all Test10 signature-row policies export "
            "the same parent-cell set, so a full-local parent-subset replay is "
            "ready."
        )
    else:
        finding = "direct_pspg_signature_parent_subset_replay_readiness_mixed"
        status = "inspect_parent_subset_inputs"
        conclusion = "Parent-subset replay readiness evidence is mixed."

    return {
        "scope": (
            "Export the Test10 signature-row parent-cell set and check whether "
            "the solve-time direct PSPG topology-policy hook can replay full "
            "local matrix mutation on that parent subset without global row "
            "filter attenuation."
        ),
        "finding": finding,
        "status": status,
        "source_hook": source,
        "parent_scope": scope,
        "same_signature_parent_set_all_policies": same_parent_set_all_policies,
        "signature_parent_cell_count": len(shared_parent_cells),
        "signature_parent_cells": shared_parent_cells,
        "signature_parent_cells_csv": csv_values(shared_parent_cells),
        "signature_parent_cell_ranges": csv_ranges(shared_parent_cells),
        "signature_parent_replays": replays,
        "missing_evidence": missing,
        "conclusion": conclusion,
        "next_requirement": (
            "Run the Test10 step90 topology-policy replay with "
            "`SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELLS` set to "
            "the exported signature parent cells and no global row DOF filter. "
            "If that full-local parent-subset replay still triggers the guard, "
            "the exact signature parent/rule support subset is ruled out; if "
            "it clears Test10, run the same physical selector transfer check "
            "for Test02 before promoting any formulation rule."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        artifact_root=args.artifact_root,
        parent_scope_json=args.parent_scope_json,
        standard_assembler=args.standard_assembler,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
