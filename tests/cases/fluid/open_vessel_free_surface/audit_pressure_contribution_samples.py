#!/usr/bin/env python3
"""Extract sampled pressure-row contribution diagnostics from solver logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pressure_update_guard import parse_key_values  # noqa: E402


CONTRIBUTION_PREFIX = "pressure_row_contribution_post_constraints:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse Newton field residual contribution diagnostics and emit a "
            "compact sampled-row JSON report."
        )
    )
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_sampled_dofs(raw: Any) -> dict[str, float]:
    if not isinstance(raw, str) or raw == "none":
        return {}
    out: dict[str, float] = {}
    for item in raw.split("|"):
        if ":" not in item:
            continue
        dof_raw, value_raw = item.split(":", 1)
        try:
            out[dof_raw] = float(value_raw)
        except ValueError:
            continue
    return out


def contribution_op_from_phase(phase: Any) -> str | None:
    if not isinstance(phase, str):
        return None
    if not phase.startswith(CONTRIBUTION_PREFIX):
        return None
    return phase[len(CONTRIBUTION_PREFIX) :]


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if abs(out) < float("inf") else None
    return None


def sample_value(samples: dict[str, float], op: str) -> float:
    value = finite_float(samples.get(op))
    return 0.0 if value is None else value


def dominant_operator(samples: dict[str, float]) -> dict[str, Any] | None:
    operator_values = [
        (op, value)
        for op, value in samples.items()
        if op != "total_residual" and finite_float(value) is not None
    ]
    if not operator_values:
        return None
    op, value = max(operator_values, key=lambda item: abs(item[1]))
    return {
        "operator": op,
        "value": value,
        "abs_value": abs(value),
    }


def classify_line_search_sample(samples: dict[str, float]) -> dict[str, Any]:
    eps = 1.0e-14
    galerkin = sample_value(samples, "equations_diagnostic_ns_galerkin_continuity")
    active_continuity = sample_value(
        samples, "equations_diagnostic_ns_active_continuity"
    )
    vms_pspg = sample_value(samples, "equations_diagnostic_ns_vms_pspg")
    pspg_pressure_gradient = sample_value(
        samples, "equations_diagnostic_ns_vms_pspg_pressure_gradient"
    )
    pspg_nonpressure = sample_value(
        samples, "equations_diagnostic_ns_vms_pspg_nonpressure"
    )
    boundary_pressure_gradient = sample_value(
        samples, "equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient"
    )
    boundary_pressure_flux = sample_value(
        samples, "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux"
    )
    boundary_tangential_pressure_gradient = sample_value(
        samples,
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient",
    )
    boundary_tangential_momentum_residual = sample_value(
        samples,
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual",
    )
    pressure_ghost_penalty = sample_value(
        samples, "equations_diagnostic_ns_pressure_ghost_penalty"
    )
    free_surface_pressure_reference = sample_value(
        samples, "equations_diagnostic_ns_free_surface_pressure_reference_probe"
    )
    free_surface_tangential_pressure_gradient = sample_value(
        samples,
        "equations_diagnostic_ns_free_surface_tangential_pressure_gradient_probe",
    )
    total_residual = sample_value(samples, "total_residual")

    direct_pspg_pressure_path_abs = abs(pspg_pressure_gradient)
    boundary_pressure_path_abs = max(
        abs(boundary_pressure_gradient),
        abs(boundary_pressure_flux),
        abs(boundary_tangential_pressure_gradient),
        abs(boundary_tangential_momentum_residual),
    )
    generated_interface_path_abs = max(
        abs(free_surface_pressure_reference),
        abs(free_surface_tangential_pressure_gradient),
    )
    nonpressure_path_abs = abs(pspg_nonpressure)
    ghost_path_abs = abs(pressure_ghost_penalty)

    path_strengths = {
        "direct_pspg_pressure_gradient": direct_pspg_pressure_path_abs,
        "boundary_pspg_pressure_probe": boundary_pressure_path_abs,
        "pspg_nonpressure": nonpressure_path_abs,
        "pressure_ghost_penalty": ghost_path_abs,
        "free_surface_pressure_probe": generated_interface_path_abs,
    }
    gal_vms_sum = galerkin + vms_pspg
    gal_vms_abs_sum = abs(galerkin) + abs(vms_pspg)
    gal_vms_cancellation_ratio = (
        abs(gal_vms_sum) / gal_vms_abs_sum if gal_vms_abs_sum > eps else None
    )
    direct_share_of_vms = (
        abs(pspg_pressure_gradient) / abs(vms_pspg)
        if abs(vms_pspg) > eps
        else None
    )
    ghost_share_of_max_path = (
        ghost_path_abs / max(path_strengths.values())
        if max(path_strengths.values()) > eps
        else None
    )
    primary_path = max(path_strengths.items(), key=lambda item: item[1])[0]
    if path_strengths[primary_path] <= eps:
        primary_path = (
            "galerkin_vms_cancellation_only_sample"
            if gal_vms_cancellation_ratio is not None
            and gal_vms_cancellation_ratio <= 0.05
            else "none_or_roundoff"
        )
    residual_cancellation_class = (
        "galerkin_vms_cancelled"
        if gal_vms_cancellation_ratio is not None
        and gal_vms_cancellation_ratio <= 0.05
        else "galerkin_vms_not_cancelled"
        if gal_vms_cancellation_ratio is not None
        else "missing_galerkin_or_vms_sample"
    )
    dominant = dominant_operator(samples)
    return {
        "primary_pressure_path": primary_path,
        "residual_cancellation_class": residual_cancellation_class,
        "dominant_operator": dominant,
        "galerkin_value": galerkin,
        "active_continuity_value": active_continuity,
        "vms_pspg_value": vms_pspg,
        "pspg_pressure_gradient_value": pspg_pressure_gradient,
        "pspg_nonpressure_value": pspg_nonpressure,
        "boundary_pressure_gradient_value": boundary_pressure_gradient,
        "boundary_pressure_flux_value": boundary_pressure_flux,
        "boundary_tangential_pressure_gradient_value": (
            boundary_tangential_pressure_gradient
        ),
        "boundary_tangential_momentum_residual_value": (
            boundary_tangential_momentum_residual
        ),
        "pressure_ghost_penalty_value": pressure_ghost_penalty,
        "free_surface_pressure_reference_value": free_surface_pressure_reference,
        "free_surface_tangential_pressure_gradient_value": (
            free_surface_tangential_pressure_gradient
        ),
        "total_residual_value": total_residual,
        "galerkin_plus_vms_pspg_value": gal_vms_sum,
        "galerkin_vms_cancellation_ratio": gal_vms_cancellation_ratio,
        "direct_pspg_pressure_gradient_share_of_vms": direct_share_of_vms,
        "pressure_ghost_penalty_share_of_max_path": ghost_share_of_max_path,
        "pressure_ghost_penalty_is_roundoff": ghost_path_abs <= eps,
        "free_surface_pressure_probe_is_roundoff": (
            generated_interface_path_abs <= eps
        ),
        "has_direct_pspg_pressure_gradient_sample": (
            "equations_diagnostic_ns_vms_pspg_pressure_gradient" in samples
        ),
        "path_abs_values": path_strengths,
    }


def parse_newton_field_residual(line: str, line_number: int) -> dict[str, Any] | None:
    if "diagnostic=newton_field_residual" not in line:
        return None
    values = parse_key_values(line)
    record: dict[str, Any] = {"line_number": line_number, **values}
    record["sampled_dofs"] = parse_sampled_dofs(values.get("sampled_dofs"))
    op = contribution_op_from_phase(values.get("phase"))
    if op is not None:
        record["contribution_op"] = op
    return record


def audit_pressure_contribution_samples(solver_log: Path) -> dict[str, Any]:
    field_residuals: list[dict[str, Any]] = []
    accepted_pressure_updates: list[dict[str, Any]] = []
    with solver_log.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            residual = parse_newton_field_residual(line, line_number)
            if residual is not None:
                field_residuals.append(residual)
                continue
            if "diagnostic=accepted_pressure_update_guard" in line:
                accepted_pressure_updates.append(
                    {"line_number": line_number, **parse_key_values(line)}
                )

    contribution_records = [
        record for record in field_residuals if "contribution_op" in record
    ]
    by_sync_point: dict[str, dict[str, dict[str, Any]]] = {}
    for record in contribution_records:
        sync_point = str(record.get("sync_point", "unknown"))
        op = str(record["contribution_op"])
        by_sync_point.setdefault(sync_point, {})[op] = {
            "line_number": record["line_number"],
            "norm": record.get("norm"),
            "global_max_abs": record.get("global_max_abs"),
            "local_worst_dof": record.get("local_worst_dof"),
            "local_worst_value": record.get("local_worst_value"),
            "sampled_dofs": record["sampled_dofs"],
        }

    total_by_sync_point: dict[str, dict[str, Any]] = {}
    for record in field_residuals:
        if record.get("phase") in ("jacobian_and_residual", "line_search"):
            sync_point = str(record.get("sync_point", "unknown"))
            total_by_sync_point[sync_point] = {
                "line_number": record["line_number"],
                "phase": record.get("phase"),
                "norm": record.get("norm"),
                "global_max_abs": record.get("global_max_abs"),
                "local_worst_dof": record.get("local_worst_dof"),
                "local_worst_value": record.get("local_worst_value"),
                "sampled_dofs": record["sampled_dofs"],
            }

    sampled_dofs = sorted(
        {
            dof
            for record in field_residuals
            for dof in record.get("sampled_dofs", {}).keys()
        },
        key=lambda value: int(value),
    )
    line_search = by_sync_point.get("line_search_trial", {})
    line_search_samples_by_dof: dict[str, dict[str, float]] = {}
    for dof in sampled_dofs:
        per_op: dict[str, float] = {}
        for op, record in line_search.items():
            samples = record.get("sampled_dofs", {})
            if dof in samples:
                per_op[op] = samples[dof]
        total_samples = total_by_sync_point.get("line_search_trial", {}).get(
            "sampled_dofs", {}
        )
        if dof in total_samples:
            per_op["total_residual"] = total_samples[dof]
        if per_op:
            line_search_samples_by_dof[dof] = per_op

    line_search_sample_classification_by_dof = {
        dof: classify_line_search_sample(samples)
        for dof, samples in line_search_samples_by_dof.items()
    }

    return {
        "solver_log": str(solver_log),
        "field_residual_count": len(field_residuals),
        "contribution_record_count": len(contribution_records),
        "sampled_dofs": sampled_dofs,
        "total_by_sync_point": total_by_sync_point,
        "contributions_by_sync_point": by_sync_point,
        "line_search_samples_by_dof": line_search_samples_by_dof,
        "line_search_sample_classification_by_dof": (
            line_search_sample_classification_by_dof
        ),
        "accepted_pressure_updates": accepted_pressure_updates,
    }


def main() -> int:
    args = parse_args()
    report = audit_pressure_contribution_samples(args.solver_log)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
