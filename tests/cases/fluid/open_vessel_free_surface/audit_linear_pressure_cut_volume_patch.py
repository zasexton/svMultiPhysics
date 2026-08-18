#!/usr/bin/env python3
"""Synthetic linear-pressure cut-volume patch diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import audit_pressure_stabilization_contribution as pressure_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a two-tetra cut-volume pressure patch preserves a "
            "linear hydrostatic pressure state under retained support and under "
            "trace-only pruned cut-adjacent support."
        )
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--fail-on-hazard", action="store_true")
    return parser.parse_args()


def patch_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def patch_tets() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
        ],
        dtype=np.int64,
    )


def linear_pressure(points: np.ndarray) -> np.ndarray:
    gradient = np.asarray([1.0, 2.0, 3.0], dtype=float)
    return 10.0 + points @ gradient


def tetra_shape_gradients(points: np.ndarray, tet: np.ndarray) -> tuple[np.ndarray, float]:
    basis = np.column_stack((np.ones(4), points[tet]))
    coefficients = np.linalg.inv(basis)
    gradients = coefficients[1:, :].T
    jacobian = np.column_stack(
        (
            points[tet[1]] - points[tet[0]],
            points[tet[2]] - points[tet[0]],
            points[tet[3]] - points[tet[0]],
        )
    )
    volume = abs(float(np.linalg.det(jacobian))) / 6.0
    return gradients, volume


def pspg_pressure_gradient_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    volume_fractions: np.ndarray,
    tau_m: float,
) -> np.ndarray:
    matrix = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    for cell_index, tet in enumerate(tets):
        fraction = float(volume_fractions[cell_index])
        if fraction <= 0.0:
            continue
        gradients, volume = tetra_shape_gradients(points, tet)
        local = tau_m * fraction * volume * (gradients @ gradients.T)
        for local_i, global_i in enumerate(tet):
            for local_j, global_j in enumerate(tet):
                matrix[global_i, global_j] += local[local_i, local_j]
    return matrix


def incident_cell_counts(tets: np.ndarray, point_count: int) -> np.ndarray:
    counts = np.zeros(point_count, dtype=np.int64)
    for tet in tets:
        for vertex in tet:
            counts[int(vertex)] += 1
    return counts


def symmetric_pseudoinverse(matrix: np.ndarray, *, tolerance: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    inverse_eigenvalues = np.asarray(
        [
            (1.0 / float(value)) if abs(float(value)) > tolerance else 0.0
            for value in eigenvalues
        ],
        dtype=float,
    )
    return (eigenvectors * inverse_eigenvalues) @ eigenvectors.T


def pspg_boundary_solve_amplification_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    diag_abs = np.abs(np.diag(matrix))
    candidates = [
        int(row)
        for row, value in enumerate(row_abs_sum.tolist())
        if float(value) > zero_tol
    ]
    if not candidates:
        return {
            "available": False,
            "reason": "no_positive_pressure_rows",
        }

    pinv = symmetric_pseudoinverse(matrix, tolerance=zero_tol)
    mean_projection = np.eye(matrix.shape[0], dtype=float)
    mean_projection -= np.ones_like(matrix, dtype=float) / float(matrix.shape[0])
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    row_reports: list[dict[str, Any]] = []
    for target in candidates:
        rhs = mean_projection[:, target]
        solution = pinv @ rhs
        residual = matrix @ solution - rhs
        row_reports.append(
            {
                "target_row": target,
                "incident_cell_count": int(incident_counts[target]),
                "diag_abs": float(diag_abs[target]),
                "row_abs_sum": float(row_abs_sum[target]),
                "target_solution": float(solution[target]),
                "target_solution_abs": float(abs(solution[target])),
                "max_solution_abs": float(np.max(np.abs(solution))),
                "solution_mean_abs": float(abs(np.mean(solution))),
                "solve_residual_abs_max": float(np.max(np.abs(residual))),
            }
        )

    max_response = max(row_reports, key=lambda row: row["target_solution_abs"])
    weakest_diag = min(row_reports, key=lambda row: row["diag_abs"])
    strongest_support = max(row_reports, key=lambda row: row["row_abs_sum"])
    strongest_response = strongest_support["target_solution_abs"]
    response_ratio_to_strongest = (
        max_response["target_solution_abs"] / strongest_response
        if strongest_response > zero_tol
        else None
    )
    return {
        "available": True,
        "candidate_row_count": len(row_reports),
        "rows": row_reports,
        "max_target_response_row": max_response,
        "weakest_diag_row": weakest_diag,
        "strongest_support_row": strongest_support,
        "max_to_strongest_support_target_response_ratio": (
            float(response_ratio_to_strongest)
            if response_ratio_to_strongest is not None
            else None
        ),
        "max_response_is_weakest_diag_row": (
            max_response["target_row"] == weakest_diag["target_row"]
        ),
        "uniform_scale_probe": {
            "scale": 10.0,
            "max_target_response_abs": float(
                max_response["target_solution_abs"] / 10.0
            ),
            "strongest_support_target_response_abs": float(
                strongest_response / 10.0
            ),
            "max_to_strongest_support_target_response_ratio": (
                float(response_ratio_to_strongest)
                if response_ratio_to_strongest is not None
                else None
            ),
            "preserves_max_response_row": True,
            "preserves_response_ratio": True,
        },
        "constant_null_preserved_during_solve_proxy": bool(
            np.max(np.abs(matrix @ np.ones(matrix.shape[0], dtype=float))) <= zero_tol
        ),
    }


def pspg_boundary_pair_completion_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    diag_abs = np.abs(np.diag(matrix))
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    one_cell_rows = [
        int(row)
        for row, incident_count in enumerate(incident_counts.tolist())
        if int(incident_count) == 1 and float(row_abs_sum[row]) > zero_tol
    ]
    if len(one_cell_rows) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_one_cell_boundary_rows",
            "one_cell_boundary_rows": one_cell_rows,
        }

    positive_boundary_diagonals = [
        float(diag_abs[row])
        for row in one_cell_rows
        if float(diag_abs[row]) > zero_tol
    ]
    if not positive_boundary_diagonals:
        return {
            "available": False,
            "reason": "no_positive_one_cell_boundary_diagonal",
            "one_cell_boundary_rows": one_cell_rows,
        }

    edge_weight = min(positive_boundary_diagonals)
    completed = matrix.copy()
    completion_edges: list[dict[str, Any]] = []
    for index, row_i in enumerate(one_cell_rows):
        for row_j in one_cell_rows[index + 1 :]:
            completed[row_i, row_i] += edge_weight
            completed[row_j, row_j] += edge_weight
            completed[row_i, row_j] -= edge_weight
            completed[row_j, row_i] -= edge_weight
            completion_edges.append(
                {
                    "row_i": row_i,
                    "row_j": row_j,
                    "weight": float(edge_weight),
                }
            )

    pressure_gradient_action = completed @ constrained_pressure_values
    hydrostatic_body_force_action = -(completed @ exact_pressure)
    total_hydrostatic_action = (
        pressure_gradient_action + hydrostatic_body_force_action
    )
    constant_mode_action = completed @ np.ones(matrix.shape[0], dtype=float)
    completed_amplification = pspg_boundary_solve_amplification_report(
        completed,
        tets=tets,
    )

    base_ratio = base_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    completed_ratio = completed_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    base_response = (
        base_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if base_amplification.get("available")
        else None
    )
    completed_response = (
        completed_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if completed_amplification.get("available")
        else None
    )
    return {
        "available": True,
        "kind": "diagnostic_one_cell_boundary_pair_completion",
        "edge_weight_rule": "min_positive_one_cell_boundary_diagonal",
        "edge_weight": float(edge_weight),
        "one_cell_boundary_rows": one_cell_rows,
        "edge_count": len(completion_edges),
        "completion_edges": completion_edges,
        "pressure_gradient_action": [
            float(value) for value in pressure_gradient_action.tolist()
        ],
        "hydrostatic_body_force_action": [
            float(value) for value in hydrostatic_body_force_action.tolist()
        ],
        "total_hydrostatic_action": [
            float(value) for value in total_hydrostatic_action.tolist()
        ],
        "max_abs_total_hydrostatic_action": float(
            np.max(np.abs(total_hydrostatic_action))
        ),
        "constant_mode_action_abs_max": float(np.max(np.abs(constant_mode_action))),
        "row_sum_abs_max": float(np.max(np.abs(completed.sum(axis=1)))),
        "preserves_hydrostatic_balance": bool(
            np.max(np.abs(total_hydrostatic_action)) <= zero_tol
        ),
        "preserves_constant_pressure_null": bool(
            np.max(np.abs(constant_mode_action)) <= zero_tol
        ),
        "matrix_abs_sum": float(np.sum(np.abs(completed))),
        "matrix_diag_abs_sum": float(np.sum(np.abs(np.diag(completed)))),
        "matrix_rank": int(np.linalg.matrix_rank(completed, tol=zero_tol)),
        "base_max_to_strongest_support_target_response_ratio": (
            float(base_ratio) if base_ratio is not None else None
        ),
        "completed_max_to_strongest_support_target_response_ratio": (
            float(completed_ratio) if completed_ratio is not None else None
        ),
        "response_ratio_reduction_factor": (
            float(base_ratio / completed_ratio)
            if base_ratio is not None
            and completed_ratio is not None
            and completed_ratio > zero_tol
            else None
        ),
        "reduces_response_ratio": bool(
            base_ratio is not None
            and completed_ratio is not None
            and completed_ratio < base_ratio
        ),
        "base_max_target_response_abs": (
            float(base_response) if base_response is not None else None
        ),
        "completed_max_target_response_abs": (
            float(completed_response) if completed_response is not None else None
        ),
        "reduces_max_target_response": bool(
            base_response is not None
            and completed_response is not None
            and completed_response < base_response
        ),
        "completed_boundary_solve_amplification": completed_amplification,
    }


def constant_null_edge_completion_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
    kind: str,
    edge_weight_rule: str,
    completion_edges: list[dict[str, Any]],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    if not completion_edges:
        return {
            "available": False,
            "reason": "no_completion_edges",
            "kind": kind,
            "edge_weight_rule": edge_weight_rule,
        }

    completed = matrix.copy()
    for edge in completion_edges:
        row_i = int(edge["row_i"])
        row_j = int(edge["row_j"])
        weight = float(edge["weight"])
        completed[row_i, row_i] += weight
        completed[row_j, row_j] += weight
        completed[row_i, row_j] -= weight
        completed[row_j, row_i] -= weight

    pressure_gradient_action = completed @ constrained_pressure_values
    hydrostatic_body_force_action = -(completed @ exact_pressure)
    total_hydrostatic_action = (
        pressure_gradient_action + hydrostatic_body_force_action
    )
    constant_mode_action = completed @ np.ones(matrix.shape[0], dtype=float)
    completed_amplification = pspg_boundary_solve_amplification_report(
        completed,
        tets=tets,
    )

    base_ratio = base_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    completed_ratio = completed_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    base_response = (
        base_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if base_amplification.get("available")
        else None
    )
    completed_response = (
        completed_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if completed_amplification.get("available")
        else None
    )
    report = {
        "available": True,
        "kind": kind,
        "edge_weight_rule": edge_weight_rule,
        "edge_count": len(completion_edges),
        "completion_edges": completion_edges,
        "pressure_gradient_action": [
            float(value) for value in pressure_gradient_action.tolist()
        ],
        "hydrostatic_body_force_action": [
            float(value) for value in hydrostatic_body_force_action.tolist()
        ],
        "total_hydrostatic_action": [
            float(value) for value in total_hydrostatic_action.tolist()
        ],
        "max_abs_total_hydrostatic_action": float(
            np.max(np.abs(total_hydrostatic_action))
        ),
        "constant_mode_action_abs_max": float(np.max(np.abs(constant_mode_action))),
        "row_sum_abs_max": float(np.max(np.abs(completed.sum(axis=1)))),
        "preserves_hydrostatic_balance": bool(
            np.max(np.abs(total_hydrostatic_action)) <= zero_tol
        ),
        "preserves_constant_pressure_null": bool(
            np.max(np.abs(constant_mode_action)) <= zero_tol
        ),
        "matrix_abs_sum": float(np.sum(np.abs(completed))),
        "matrix_diag_abs_sum": float(np.sum(np.abs(np.diag(completed)))),
        "matrix_rank": int(np.linalg.matrix_rank(completed, tol=zero_tol)),
        "base_max_to_strongest_support_target_response_ratio": (
            float(base_ratio) if base_ratio is not None else None
        ),
        "completed_max_to_strongest_support_target_response_ratio": (
            float(completed_ratio) if completed_ratio is not None else None
        ),
        "response_ratio_reduction_factor": (
            float(base_ratio / completed_ratio)
            if base_ratio is not None
            and completed_ratio is not None
            and completed_ratio > zero_tol
            else None
        ),
        "reduces_response_ratio": bool(
            base_ratio is not None
            and completed_ratio is not None
            and completed_ratio < base_ratio
        ),
        "base_max_target_response_abs": (
            float(base_response) if base_response is not None else None
        ),
        "completed_max_target_response_abs": (
            float(completed_response) if completed_response is not None else None
        ),
        "reduces_max_target_response": bool(
            base_response is not None
            and completed_response is not None
            and completed_response < base_response
        ),
        "completed_boundary_solve_amplification": completed_amplification,
    }
    if extra_fields:
        report.update(extra_fields)
    return report


def apply_constant_null_edges(
    matrix: np.ndarray,
    completion_edges: list[dict[str, Any]],
) -> np.ndarray:
    completed = matrix.copy()
    for edge in completion_edges:
        row_i = int(edge["row_i"])
        row_j = int(edge["row_j"])
        weight = float(edge["weight"])
        completed[row_i, row_i] += weight
        completed[row_j, row_j] += weight
        completed[row_i, row_j] -= weight
        completed[row_j, row_i] -= weight
    return completed


def pspg_boundary_shared_support_completion_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    diag_abs = np.abs(np.diag(matrix))
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    one_cell_rows = [
        int(row)
        for row, incident_count in enumerate(incident_counts.tolist())
        if int(incident_count) == 1 and float(row_abs_sum[row]) > zero_tol
    ]
    shared_rows = [
        int(row)
        for row, incident_count in enumerate(incident_counts.tolist())
        if int(incident_count) > 1 and float(row_abs_sum[row]) > zero_tol
    ]
    if len(one_cell_rows) < 1 or len(shared_rows) < 1:
        return {
            "available": False,
            "reason": "missing_one_cell_or_shared_rows",
            "one_cell_boundary_rows": one_cell_rows,
            "shared_support_rows": shared_rows,
        }

    positive_boundary_diagonals = [
        float(diag_abs[row])
        for row in one_cell_rows
        if float(diag_abs[row]) > zero_tol
    ]
    if not positive_boundary_diagonals:
        return {
            "available": False,
            "reason": "no_positive_one_cell_boundary_diagonal",
            "one_cell_boundary_rows": one_cell_rows,
            "shared_support_rows": shared_rows,
        }

    total_added_per_boundary_row = min(positive_boundary_diagonals)
    edge_weight = total_added_per_boundary_row / float(len(shared_rows))
    completion_edges = [
        {
            "row_i": int(row_i),
            "row_j": int(row_j),
            "weight": float(edge_weight),
        }
        for row_i in one_cell_rows
        for row_j in shared_rows
    ]
    return constant_null_edge_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        kind="diagnostic_one_cell_to_shared_support_completion",
        edge_weight_rule=(
            "min_positive_one_cell_boundary_diagonal_distributed_over_shared_rows"
        ),
        completion_edges=completion_edges,
        extra_fields={
            "one_cell_boundary_rows": one_cell_rows,
            "shared_support_rows": shared_rows,
            "total_added_per_boundary_row": float(total_added_per_boundary_row),
        },
    )


def pspg_boundary_weak_active_completion_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    diag_abs = np.abs(np.diag(matrix))
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    active_rows = [
        int(row)
        for row, value in enumerate(row_abs_sum.tolist())
        if float(value) > zero_tol
    ]
    one_cell_rows = [
        int(row)
        for row in active_rows
        if int(incident_counts[row]) == 1
    ]
    if len(one_cell_rows) < 1 or len(active_rows) < 2:
        return {
            "available": False,
            "reason": "missing_one_cell_or_active_rows",
            "one_cell_boundary_rows": one_cell_rows,
            "active_rows": active_rows,
        }

    positive_boundary_diagonals = [
        float(diag_abs[row])
        for row in one_cell_rows
        if float(diag_abs[row]) > zero_tol
    ]
    if not positive_boundary_diagonals:
        return {
            "available": False,
            "reason": "no_positive_one_cell_boundary_diagonal",
            "one_cell_boundary_rows": one_cell_rows,
            "active_rows": active_rows,
        }

    total_added_per_boundary_row = min(positive_boundary_diagonals)
    edge_contributions: dict[tuple[int, int], float] = {}
    contribution_count = 0
    for row_i in one_cell_rows:
        per_edge_weight = total_added_per_boundary_row / float(
            len(active_rows) - 1
        )
        for row_j in active_rows:
            if row_i == row_j:
                continue
            key = (min(row_i, row_j), max(row_i, row_j))
            edge_contributions[key] = edge_contributions.get(key, 0.0) + per_edge_weight
            contribution_count += 1

    completion_edges = [
        {
            "row_i": int(row_i),
            "row_j": int(row_j),
            "weight": float(weight),
        }
        for (row_i, row_j), weight in sorted(edge_contributions.items())
    ]
    return constant_null_edge_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        kind="diagnostic_one_cell_to_active_support_completion",
        edge_weight_rule=(
            "min_positive_one_cell_boundary_diagonal_distributed_from_each_"
            "one_cell_row_to_all_other_active_rows"
        ),
        completion_edges=completion_edges,
        extra_fields={
            "one_cell_boundary_rows": one_cell_rows,
            "active_rows": active_rows,
            "contribution_count": int(contribution_count),
            "total_added_per_boundary_row": float(total_added_per_boundary_row),
        },
    )


def pspg_shared_row_schur_completion_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    shared_rows = [
        int(row)
        for row, incident_count in enumerate(incident_counts.tolist())
        if int(incident_count) > 1 and float(row_abs_sum[row]) > zero_tol
    ]
    if not shared_rows:
        return {
            "available": False,
            "reason": "no_shared_support_rows",
            "shared_support_rows": shared_rows,
        }

    edge_contributions: dict[tuple[int, int], float] = {}
    source_reports: list[dict[str, Any]] = []
    contribution_count = 0
    for shared_row in shared_rows:
        neighbors: list[dict[str, Any]] = []
        for row in range(matrix.shape[0]):
            if row == shared_row:
                continue
            symmetric_offdiag = 0.5 * (matrix[shared_row, row] + matrix[row, shared_row])
            edge_weight = max(0.0, -float(symmetric_offdiag))
            if edge_weight <= zero_tol:
                continue
            neighbors.append(
                {
                    "row": int(row),
                    "edge_weight": float(edge_weight),
                    "incident_cell_count": int(incident_counts[row]),
                }
            )
        support_weight_sum = sum(
            float(neighbor["edge_weight"]) for neighbor in neighbors
        )
        source_reports.append(
            {
                "shared_row": int(shared_row),
                "incident_cell_count": int(incident_counts[shared_row]),
                "neighbor_count": len(neighbors),
                "support_weight_sum": float(support_weight_sum),
                "neighbors": neighbors,
            }
        )
        if len(neighbors) < 2 or support_weight_sum <= zero_tol:
            continue
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                row_i = int(left["row"])
                row_j = int(right["row"])
                weight = (
                    float(left["edge_weight"])
                    * float(right["edge_weight"])
                    / support_weight_sum
                )
                key = (min(row_i, row_j), max(row_i, row_j))
                edge_contributions[key] = edge_contributions.get(key, 0.0) + weight
                contribution_count += 1

    completion_edges = [
        {
            "row_i": int(row_i),
            "row_j": int(row_j),
            "weight": float(weight),
        }
        for (row_i, row_j), weight in sorted(edge_contributions.items())
    ]
    return constant_null_edge_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        kind="diagnostic_shared_row_schur_support_completion",
        edge_weight_rule=(
            "existing_pressure_edge_schur_fill_wi_wj_over_shared_row_support_sum"
        ),
        completion_edges=completion_edges,
        extra_fields={
            "shared_support_rows": shared_rows,
            "source_shared_rows": source_reports,
            "contribution_count": int(contribution_count),
        },
    )


def pspg_shared_row_schur_existing_edge_balance_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    schur_report = pspg_shared_row_schur_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
    )
    if not schur_report.get("available"):
        return {
            "available": False,
            "reason": schur_report.get(
                "reason", "schur_completion_unavailable"
            ),
            "kind": "diagnostic_shared_row_schur_existing_edge_balance",
            "edge_weight_rule": (
                "shared_row_schur_completion_then_existing_pressure_edge_"
                "support_balance"
            ),
        }

    schur_completed = apply_constant_null_edges(
        matrix,
        schur_report["completion_edges"],
    )
    report = pspg_existing_edge_support_balance_report(
        schur_completed,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
    )
    report["kind"] = "diagnostic_shared_row_schur_existing_edge_balance"
    report["edge_weight_rule"] = (
        "shared_row_schur_completion_then_existing_pressure_edge_"
        "support_balance"
    )
    if report.get("available"):
        report["schur_edge_count"] = int(schur_report["edge_count"])
        report["schur_contribution_count"] = int(
            schur_report["contribution_count"]
        )
        report[
            "schur_completed_max_to_strongest_support_target_response_ratio"
        ] = schur_report[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        report["schur_completed_max_target_response_abs"] = schur_report[
            "completed_max_target_response_abs"
        ]
    return report


def pspg_existing_edge_support_balance_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
    eligible_balance_rows: set[int] | None = None,
    kind: str = "diagnostic_existing_pressure_edge_support_balance",
    edge_weight_rule: str = (
        "existing_pressure_laplacian_edges_scaled_to_strongest_row_abs_sum"
    ),
    balance_row_selection: str = "all_active_pressure_rows",
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    active_rows = [
        int(row)
        for row, value in enumerate(row_abs_sum.tolist())
        if float(value) > zero_tol
    ]
    if not active_rows:
        return {
            "available": False,
            "reason": "no_positive_pressure_rows",
        }

    target_row_abs_sum = max(float(row_abs_sum[row]) for row in active_rows)
    if not (target_row_abs_sum > zero_tol):
        return {
            "available": False,
            "reason": "no_positive_target_row_abs_sum",
        }

    non_laplacian_offdiag_count = 0
    balance_rows = (
        set(active_rows)
        if eligible_balance_rows is None
        else {int(row) for row in eligible_balance_rows if int(row) in active_rows}
    )
    if not balance_rows:
        return {
            "available": False,
            "reason": "no_eligible_balance_rows",
            "kind": kind,
            "edge_weight_rule": edge_weight_rule,
            "balance_row_selection": balance_row_selection,
            "eligible_balance_rows": [],
        }

    scaled_edges: list[dict[str, Any]] = []
    balanced = np.zeros_like(matrix)
    for row_i in range(matrix.shape[0]):
        for row_j in range(row_i + 1, matrix.shape[1]):
            symmetric_offdiag = 0.5 * (matrix[row_i, row_j] + matrix[row_j, row_i])
            if symmetric_offdiag > zero_tol:
                non_laplacian_offdiag_count += 1
                continue
            edge_weight = max(0.0, -float(symmetric_offdiag))
            if not (edge_weight > zero_tol):
                continue
            row_i_scale = (
                target_row_abs_sum / float(row_abs_sum[row_i])
                if row_i in balance_rows and float(row_abs_sum[row_i]) > zero_tol
                else 1.0
            )
            row_j_scale = (
                target_row_abs_sum / float(row_abs_sum[row_j])
                if row_j in balance_rows and float(row_abs_sum[row_j]) > zero_tol
                else 1.0
            )
            scale = max(1.0, row_i_scale, row_j_scale)
            scaled_weight = edge_weight * scale
            balanced[row_i, row_i] += scaled_weight
            balanced[row_j, row_j] += scaled_weight
            balanced[row_i, row_j] -= scaled_weight
            balanced[row_j, row_i] -= scaled_weight
            scaled_edges.append(
                {
                    "row_i": int(row_i),
                    "row_j": int(row_j),
                    "base_weight": float(edge_weight),
                    "scale": float(scale),
                    "balanced_weight": float(scaled_weight),
                    "row_i_eligible_balance": bool(row_i in balance_rows),
                    "row_j_eligible_balance": bool(row_j in balance_rows),
                }
            )

    if not scaled_edges:
        return {
            "available": False,
            "reason": "no_existing_pressure_edges",
        }

    pressure_gradient_action = balanced @ constrained_pressure_values
    hydrostatic_body_force_action = -(balanced @ exact_pressure)
    total_hydrostatic_action = (
        pressure_gradient_action + hydrostatic_body_force_action
    )
    constant_mode_action = balanced @ np.ones(matrix.shape[0], dtype=float)
    balanced_amplification = pspg_boundary_solve_amplification_report(
        balanced,
        tets=tets,
    )

    base_ratio = base_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    balanced_ratio = balanced_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    base_response = (
        base_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if base_amplification.get("available")
        else None
    )
    balanced_response = (
        balanced_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if balanced_amplification.get("available")
        else None
    )
    balanced_row_abs_sum = np.sum(np.abs(balanced), axis=1)
    scale_values = [float(edge["scale"]) for edge in scaled_edges]
    return {
        "available": True,
        "kind": kind,
        "edge_weight_rule": edge_weight_rule,
        "balance_row_selection": balance_row_selection,
        "eligible_balance_rows": sorted(int(row) for row in balance_rows),
        "edge_count": len(scaled_edges),
        "scaled_edges": scaled_edges,
        "target_row_abs_sum": float(target_row_abs_sum),
        "base_row_abs_sum": [float(value) for value in row_abs_sum.tolist()],
        "balanced_row_abs_sum": [
            float(value) for value in balanced_row_abs_sum.tolist()
        ],
        "min_edge_scale": float(min(scale_values)),
        "max_edge_scale": float(max(scale_values)),
        "non_laplacian_offdiag_count": int(non_laplacian_offdiag_count),
        "pressure_gradient_action": [
            float(value) for value in pressure_gradient_action.tolist()
        ],
        "hydrostatic_body_force_action": [
            float(value) for value in hydrostatic_body_force_action.tolist()
        ],
        "total_hydrostatic_action": [
            float(value) for value in total_hydrostatic_action.tolist()
        ],
        "max_abs_total_hydrostatic_action": float(
            np.max(np.abs(total_hydrostatic_action))
        ),
        "constant_mode_action_abs_max": float(np.max(np.abs(constant_mode_action))),
        "row_sum_abs_max": float(np.max(np.abs(balanced.sum(axis=1)))),
        "preserves_hydrostatic_balance": bool(
            np.max(np.abs(total_hydrostatic_action)) <= zero_tol
        ),
        "preserves_constant_pressure_null": bool(
            np.max(np.abs(constant_mode_action)) <= zero_tol
        ),
        "matrix_abs_sum": float(np.sum(np.abs(balanced))),
        "matrix_diag_abs_sum": float(np.sum(np.abs(np.diag(balanced)))),
        "matrix_rank": int(np.linalg.matrix_rank(balanced, tol=zero_tol)),
        "base_max_to_strongest_support_target_response_ratio": (
            float(base_ratio) if base_ratio is not None else None
        ),
        "balanced_max_to_strongest_support_target_response_ratio": (
            float(balanced_ratio) if balanced_ratio is not None else None
        ),
        "response_ratio_reduction_factor": (
            float(base_ratio / balanced_ratio)
            if base_ratio is not None
            and balanced_ratio is not None
            and balanced_ratio > zero_tol
            else None
        ),
        "reduces_response_ratio": bool(
            base_ratio is not None
            and balanced_ratio is not None
            and balanced_ratio < base_ratio
        ),
        "base_max_target_response_abs": (
            float(base_response) if base_response is not None else None
        ),
        "balanced_max_target_response_abs": (
            float(balanced_response) if balanced_response is not None else None
        ),
        "reduces_max_target_response": bool(
            base_response is not None
            and balanced_response is not None
            and balanced_response < base_response
        ),
        "balanced_boundary_solve_amplification": balanced_amplification,
    }


def one_cell_boundary_rows(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
) -> list[int]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    return [
        int(row)
        for row, incident_count in enumerate(incident_counts.tolist())
        if int(incident_count) == 1 and float(row_abs_sum[row]) > zero_tol
    ]


def shared_support_rows(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
) -> list[int]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    return [
        int(row)
        for row, incident_count in enumerate(incident_counts.tolist())
        if int(incident_count) > 1 and float(row_abs_sum[row]) > zero_tol
    ]


def pspg_weak_boundary_existing_edge_support_balance_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    return pspg_existing_edge_support_balance_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        eligible_balance_rows=set(one_cell_boundary_rows(matrix, tets=tets)),
        kind="diagnostic_weak_boundary_existing_pressure_edge_support_balance",
        edge_weight_rule=(
            "existing_pressure_laplacian_edges_scaled_to_strongest_row_abs_sum_"
            "only_for_one_cell_boundary_rows"
        ),
        balance_row_selection="one_cell_boundary_rows",
    )


def pspg_shared_row_schur_shared_support_edge_balance_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    schur_report = pspg_shared_row_schur_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
    )
    if not schur_report.get("available"):
        return {
            "available": False,
            "reason": schur_report.get(
                "reason", "schur_completion_unavailable"
            ),
            "kind": "diagnostic_shared_row_schur_shared_support_edge_balance",
            "edge_weight_rule": (
                "shared_row_schur_completion_then_existing_pressure_edge_"
                "support_balance_only_for_shared_support_rows"
            ),
        }

    schur_completed = apply_constant_null_edges(
        matrix,
        schur_report["completion_edges"],
    )
    report = pspg_existing_edge_support_balance_report(
        schur_completed,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        eligible_balance_rows=set(shared_support_rows(matrix, tets=tets)),
        kind="diagnostic_shared_row_schur_shared_support_edge_balance",
        edge_weight_rule=(
            "shared_row_schur_completion_then_existing_pressure_edge_"
            "support_balance_only_for_shared_support_rows"
        ),
        balance_row_selection="shared_support_rows_after_schur_completion",
    )
    if report.get("available"):
        report["schur_edge_count"] = int(schur_report["edge_count"])
        report["schur_contribution_count"] = int(
            schur_report["contribution_count"]
        )
        report[
            "schur_completed_max_to_strongest_support_target_response_ratio"
        ] = schur_report[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        report["schur_completed_max_target_response_abs"] = schur_report[
            "completed_max_target_response_abs"
        ]
    return report


def pspg_shared_row_schur_weak_boundary_edge_balance_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    schur_report = pspg_shared_row_schur_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
    )
    if not schur_report.get("available"):
        return {
            "available": False,
            "reason": schur_report.get(
                "reason", "schur_completion_unavailable"
            ),
            "kind": "diagnostic_shared_row_schur_weak_boundary_edge_balance",
            "edge_weight_rule": (
                "shared_row_schur_completion_then_existing_pressure_edge_"
                "support_balance_only_for_one_cell_boundary_rows"
            ),
        }

    schur_completed = apply_constant_null_edges(
        matrix,
        schur_report["completion_edges"],
    )
    report = pspg_existing_edge_support_balance_report(
        schur_completed,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        eligible_balance_rows=set(one_cell_boundary_rows(matrix, tets=tets)),
        kind="diagnostic_shared_row_schur_weak_boundary_edge_balance",
        edge_weight_rule=(
            "shared_row_schur_completion_then_existing_pressure_edge_"
            "support_balance_only_for_one_cell_boundary_rows"
        ),
        balance_row_selection="one_cell_boundary_rows_after_schur_completion",
    )
    if report.get("available"):
        report["schur_edge_count"] = int(schur_report["edge_count"])
        report["schur_contribution_count"] = int(
            schur_report["contribution_count"]
        )
        report[
            "schur_completed_max_to_strongest_support_target_response_ratio"
        ] = schur_report[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        report["schur_completed_max_target_response_abs"] = schur_report[
            "completed_max_target_response_abs"
        ]
    return report


def active_pressure_rows(matrix: np.ndarray) -> list[int]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    return [
        int(row)
        for row, value in enumerate(row_abs_sum.tolist())
        if float(value) > zero_tol
    ]


def support_gap_rows_by_row_abs(matrix: np.ndarray) -> list[int]:
    active_rows = active_pressure_rows(matrix)
    if not active_rows:
        return []
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    active_row_abs = np.asarray(
        [float(row_abs_sum[row]) for row in active_rows],
        dtype=float,
    )
    support_floor = float(np.median(active_row_abs))
    zero_tol = 1.0e-12
    return [
        int(row)
        for row in active_rows
        if float(row_abs_sum[row]) + zero_tol < support_floor
    ]


def pressure_graph_component_rows(
    matrix: np.ndarray,
    *,
    seeds: list[int],
) -> list[int]:
    if not seeds:
        return []
    zero_tol = 1.0e-12
    active = set(active_pressure_rows(matrix))
    seed_set = {int(row) for row in seeds if int(row) in active}
    if not seed_set:
        return []
    visited: set[int] = set()
    stack = list(seed_set)
    while stack:
        row_i = stack.pop()
        if row_i in visited:
            continue
        visited.add(row_i)
        for row_j in active:
            if row_j == row_i or row_j in visited:
                continue
            symmetric_offdiag = 0.5 * (matrix[row_i, row_j] + matrix[row_j, row_i])
            if symmetric_offdiag < -zero_tol:
                stack.append(row_j)
    return sorted(visited)


def schur_completion_edges_for_patch(
    matrix: np.ndarray,
    *,
    patch_rows: list[int],
    support_gap_rows: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    zero_tol = 1.0e-12
    patch_set = {int(row) for row in patch_rows}
    gap_set = {int(row) for row in support_gap_rows}
    hub_rows = sorted(row for row in patch_set if row not in gap_set)
    edge_contributions: dict[tuple[int, int], float] = {}
    source_reports: list[dict[str, Any]] = []
    contribution_count = 0
    for hub_row in hub_rows:
        neighbors: list[dict[str, Any]] = []
        for row in sorted(patch_set):
            if row == hub_row:
                continue
            symmetric_offdiag = 0.5 * (matrix[hub_row, row] + matrix[row, hub_row])
            edge_weight = max(0.0, -float(symmetric_offdiag))
            if edge_weight <= zero_tol:
                continue
            neighbors.append(
                {
                    "row": int(row),
                    "edge_weight": float(edge_weight),
                    "is_support_gap_row": bool(row in gap_set),
                }
            )
        support_weight_sum = sum(
            float(neighbor["edge_weight"]) for neighbor in neighbors
        )
        source_reports.append(
            {
                "support_hub_row": int(hub_row),
                "neighbor_count": len(neighbors),
                "support_weight_sum": float(support_weight_sum),
                "neighbors": neighbors,
            }
        )
        if len(neighbors) < 2 or support_weight_sum <= zero_tol:
            continue
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                row_i = int(left["row"])
                row_j = int(right["row"])
                weight = (
                    float(left["edge_weight"])
                    * float(right["edge_weight"])
                    / support_weight_sum
                )
                key = (min(row_i, row_j), max(row_i, row_j))
                edge_contributions[key] = edge_contributions.get(key, 0.0) + weight
                contribution_count += 1
    completion_edges = [
        {
            "row_i": int(row_i),
            "row_j": int(row_j),
            "weight": float(weight),
        }
        for (row_i, row_j), weight in sorted(edge_contributions.items())
    ]
    return completion_edges, source_reports, contribution_count


def pspg_direct_support_gap_or_same_sign_patch_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    support_gap_rows = support_gap_rows_by_row_abs(matrix)
    patch_rows = pressure_graph_component_rows(matrix, seeds=support_gap_rows)
    if not support_gap_rows or len(patch_rows) < 2:
        return {
            "available": False,
            "reason": "missing_support_gap_or_pressure_patch_rows",
            "kind": "diagnostic_direct_support_gap_or_same_sign_patch_completion",
            "support_gap_rows": support_gap_rows,
            "same_sign_pressure_patch_rows": patch_rows,
        }

    schur_edges, source_reports, contribution_count = schur_completion_edges_for_patch(
        matrix,
        patch_rows=patch_rows,
        support_gap_rows=support_gap_rows,
    )
    if not schur_edges:
        return {
            "available": False,
            "reason": "no_patch_schur_completion_edges",
            "kind": "diagnostic_direct_support_gap_or_same_sign_patch_completion",
            "support_gap_rows": support_gap_rows,
            "same_sign_pressure_patch_rows": patch_rows,
            "source_support_hub_rows": source_reports,
        }

    schur_report = constant_null_edge_completion_report(
        matrix,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        kind="diagnostic_direct_support_gap_or_same_sign_patch_schur_completion",
        edge_weight_rule=(
            "same_sign_pressure_patch_schur_fill_wi_wj_over_support_hub_sum"
        ),
        completion_edges=schur_edges,
        extra_fields={
            "support_gap_rows": support_gap_rows,
            "same_sign_pressure_patch_rows": patch_rows,
            "source_support_hub_rows": source_reports,
            "contribution_count": int(contribution_count),
        },
    )
    if not schur_report.get("available"):
        return schur_report

    schur_completed = apply_constant_null_edges(matrix, schur_edges)
    report = pspg_existing_edge_support_balance_report(
        schur_completed,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=constrained_pressure_values,
        base_amplification=base_amplification,
        eligible_balance_rows=set(support_gap_rows),
        kind="diagnostic_direct_support_gap_or_same_sign_patch_completion",
        edge_weight_rule=(
            "same_sign_pressure_patch_schur_completion_then_existing_pressure_"
            "edge_balance_only_for_direct_support_gap_rows"
        ),
        balance_row_selection="direct_support_gap_rows_after_patch_completion",
    )
    if report.get("available"):
        report["support_gap_rows"] = support_gap_rows
        report["same_sign_pressure_patch_rows"] = patch_rows
        report["source_support_hub_rows"] = source_reports
        report["schur_only_completion"] = schur_report
        report["schur_edge_count"] = int(schur_report["edge_count"])
        report["schur_contribution_count"] = int(contribution_count)
        report[
            "schur_completed_max_to_strongest_support_target_response_ratio"
        ] = schur_report[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        report["schur_completed_max_target_response_abs"] = schur_report[
            "completed_max_target_response_abs"
        ]
        schur_ratio = schur_report[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        balanced_ratio = report[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        report["balance_stage_further_reduces_schur_response_ratio"] = bool(
            balanced_ratio < schur_ratio
        )
        report["balance_stage_response_ratio_reduction_factor"] = float(
            schur_ratio / balanced_ratio
        )
    return report


def pspg_incident_support_count_balance_report(
    matrix: np.ndarray,
    *,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    base_amplification: dict[str, Any],
) -> dict[str, Any]:
    zero_tol = 1.0e-12
    row_abs_sum = np.sum(np.abs(matrix), axis=1)
    active_rows = [
        int(row)
        for row, value in enumerate(row_abs_sum.tolist())
        if float(value) > zero_tol
    ]
    if not active_rows:
        return {
            "available": False,
            "reason": "no_positive_pressure_rows",
        }

    incident_counts = incident_cell_counts(tets, matrix.shape[0])
    active_incident_counts = [
        int(incident_counts[row])
        for row in active_rows
        if int(incident_counts[row]) > 0
    ]
    if not active_incident_counts:
        return {
            "available": False,
            "reason": "no_positive_incident_counts",
        }

    target_incident_count = max(active_incident_counts)
    scaled_edges: list[dict[str, Any]] = []
    balanced = np.zeros_like(matrix)
    for row_i in range(matrix.shape[0]):
        for row_j in range(row_i + 1, matrix.shape[1]):
            symmetric_offdiag = 0.5 * (matrix[row_i, row_j] + matrix[row_j, row_i])
            edge_weight = max(0.0, -float(symmetric_offdiag))
            if not (edge_weight > zero_tol):
                continue
            count_i = int(incident_counts[row_i])
            count_j = int(incident_counts[row_j])
            if count_i <= 0 or count_j <= 0:
                continue
            scale_i = float(target_incident_count) / float(count_i)
            scale_j = float(target_incident_count) / float(count_j)
            scale = max(1.0, scale_i, scale_j)
            balanced_weight = edge_weight * scale
            balanced[row_i, row_i] += balanced_weight
            balanced[row_j, row_j] += balanced_weight
            balanced[row_i, row_j] -= balanced_weight
            balanced[row_j, row_i] -= balanced_weight
            scaled_edges.append(
                {
                    "row_i": int(row_i),
                    "row_j": int(row_j),
                    "base_weight": float(edge_weight),
                    "scale": float(scale),
                    "balanced_weight": float(balanced_weight),
                    "row_i_incident_cell_count": int(count_i),
                    "row_j_incident_cell_count": int(count_j),
                }
            )

    if not scaled_edges:
        return {
            "available": False,
            "reason": "no_existing_pressure_edges",
        }

    pressure_gradient_action = balanced @ constrained_pressure_values
    hydrostatic_body_force_action = -(balanced @ exact_pressure)
    total_hydrostatic_action = (
        pressure_gradient_action + hydrostatic_body_force_action
    )
    constant_mode_action = balanced @ np.ones(matrix.shape[0], dtype=float)
    balanced_amplification = pspg_boundary_solve_amplification_report(
        balanced,
        tets=tets,
    )

    base_ratio = base_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    balanced_ratio = balanced_amplification.get(
        "max_to_strongest_support_target_response_ratio"
    )
    base_response = (
        base_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if base_amplification.get("available")
        else None
    )
    balanced_response = (
        balanced_amplification.get("max_target_response_row", {}).get(
            "target_solution_abs"
        )
        if balanced_amplification.get("available")
        else None
    )
    balanced_row_abs_sum = np.sum(np.abs(balanced), axis=1)
    scale_values = [float(edge["scale"]) for edge in scaled_edges]
    return {
        "available": True,
        "kind": "diagnostic_existing_pressure_edge_incident_support_balance",
        "edge_weight_rule": (
            "existing_pressure_laplacian_edges_scaled_by_max_endpoint_"
            "incident_cell_count_deficit"
        ),
        "edge_count": len(scaled_edges),
        "scaled_edges": scaled_edges,
        "target_incident_cell_count": int(target_incident_count),
        "incident_cell_counts": [
            int(value) for value in incident_counts.tolist()
        ],
        "base_row_abs_sum": [float(value) for value in row_abs_sum.tolist()],
        "balanced_row_abs_sum": [
            float(value) for value in balanced_row_abs_sum.tolist()
        ],
        "min_edge_scale": float(min(scale_values)),
        "max_edge_scale": float(max(scale_values)),
        "pressure_gradient_action": [
            float(value) for value in pressure_gradient_action.tolist()
        ],
        "hydrostatic_body_force_action": [
            float(value) for value in hydrostatic_body_force_action.tolist()
        ],
        "total_hydrostatic_action": [
            float(value) for value in total_hydrostatic_action.tolist()
        ],
        "max_abs_total_hydrostatic_action": float(
            np.max(np.abs(total_hydrostatic_action))
        ),
        "constant_mode_action_abs_max": float(np.max(np.abs(constant_mode_action))),
        "row_sum_abs_max": float(np.max(np.abs(balanced.sum(axis=1)))),
        "preserves_hydrostatic_balance": bool(
            np.max(np.abs(total_hydrostatic_action)) <= zero_tol
        ),
        "preserves_constant_pressure_null": bool(
            np.max(np.abs(constant_mode_action)) <= zero_tol
        ),
        "matrix_abs_sum": float(np.sum(np.abs(balanced))),
        "matrix_diag_abs_sum": float(np.sum(np.abs(np.diag(balanced)))),
        "matrix_rank": int(np.linalg.matrix_rank(balanced, tol=zero_tol)),
        "base_max_to_strongest_support_target_response_ratio": (
            float(base_ratio) if base_ratio is not None else None
        ),
        "balanced_max_to_strongest_support_target_response_ratio": (
            float(balanced_ratio) if balanced_ratio is not None else None
        ),
        "response_ratio_reduction_factor": (
            float(base_ratio / balanced_ratio)
            if base_ratio is not None
            and balanced_ratio is not None
            and balanced_ratio > zero_tol
            else None
        ),
        "reduces_response_ratio": bool(
            base_ratio is not None
            and balanced_ratio is not None
            and balanced_ratio < base_ratio
        ),
        "base_max_target_response_abs": (
            float(base_response) if base_response is not None else None
        ),
        "balanced_max_target_response_abs": (
            float(balanced_response) if balanced_response is not None else None
        ),
        "reduces_max_target_response": bool(
            base_response is not None
            and balanced_response is not None
            and balanced_response < base_response
        ),
        "balanced_boundary_solve_amplification": balanced_amplification,
    }


def pspg_hydrostatic_balance_report(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    exact_pressure: np.ndarray,
    constrained_pressure_values: np.ndarray,
    active_volume_fractions: np.ndarray,
) -> dict[str, Any]:
    tau_m = 1.0
    matrix = pspg_pressure_gradient_matrix(
        points,
        tets,
        active_volume_fractions,
        tau_m,
    )
    pressure_gradient_action = matrix @ constrained_pressure_values
    hydrostatic_body_force_action = -(matrix @ exact_pressure)
    total_hydrostatic_action = (
        pressure_gradient_action + hydrostatic_body_force_action
    )
    constant_mode_action = matrix @ np.ones(points.shape[0], dtype=float)

    direct_abs_max = float(np.max(np.abs(pressure_gradient_action)))
    body_abs_max = float(np.max(np.abs(hydrostatic_body_force_action)))
    total_abs_max = float(np.max(np.abs(total_hydrostatic_action)))
    row_sum_abs_max = float(np.max(np.abs(matrix.sum(axis=1))))
    constant_mode_abs_max = float(np.max(np.abs(constant_mode_action)))
    zero_tol = 1.0e-12
    boundary_amplification = pspg_boundary_solve_amplification_report(
        matrix,
        tets=tets,
    )
    return {
        "active_volume_fractions": [
            float(value) for value in active_volume_fractions.tolist()
        ],
        "tau_m": tau_m,
        "pressure_gradient_action": [
            float(value) for value in pressure_gradient_action.tolist()
        ],
        "hydrostatic_body_force_action": [
            float(value) for value in hydrostatic_body_force_action.tolist()
        ],
        "total_hydrostatic_action": [
            float(value) for value in total_hydrostatic_action.tolist()
        ],
        "max_abs_pressure_gradient_action": direct_abs_max,
        "max_abs_hydrostatic_body_force_action": body_abs_max,
        "max_abs_total_hydrostatic_action": total_abs_max,
        "direct_pressure_gradient_has_boundary_action": direct_abs_max > zero_tol,
        "preserves_hydrostatic_balance": total_abs_max <= zero_tol,
        "row_sum_abs_max": row_sum_abs_max,
        "constant_mode_action_abs_max": constant_mode_abs_max,
        "preserves_constant_pressure_null": constant_mode_abs_max <= zero_tol,
        "matrix_abs_sum": float(np.sum(np.abs(matrix))),
        "matrix_diag_abs_sum": float(np.sum(np.abs(np.diag(matrix)))),
        "matrix_rank": int(np.linalg.matrix_rank(matrix, tol=1.0e-12)),
        "boundary_solve_amplification": boundary_amplification,
        "boundary_pair_completion": pspg_boundary_pair_completion_report(
            matrix,
            tets=tets,
            exact_pressure=exact_pressure,
            constrained_pressure_values=constrained_pressure_values,
            base_amplification=boundary_amplification,
        ),
        "shared_support_completion": (
            pspg_boundary_shared_support_completion_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "weak_boundary_active_completion": (
            pspg_boundary_weak_active_completion_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "shared_row_schur_completion": (
            pspg_shared_row_schur_completion_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "shared_row_schur_existing_edge_balance": (
            pspg_shared_row_schur_existing_edge_balance_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "weak_boundary_existing_edge_support_balance": (
            pspg_weak_boundary_existing_edge_support_balance_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "shared_row_schur_shared_support_edge_balance": (
            pspg_shared_row_schur_shared_support_edge_balance_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "shared_row_schur_weak_boundary_edge_balance": (
            pspg_shared_row_schur_weak_boundary_edge_balance_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "direct_support_gap_or_same_sign_patch_completion": (
            pspg_direct_support_gap_or_same_sign_patch_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "existing_edge_support_balance": (
            pspg_existing_edge_support_balance_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
        "incident_support_count_balance": (
            pspg_incident_support_count_balance_report(
                matrix,
                tets=tets,
                exact_pressure=exact_pressure,
                constrained_pressure_values=constrained_pressure_values,
                base_amplification=boundary_amplification,
            )
        ),
    }


def face_energy_proxy(
    points: np.ndarray,
    tets: np.ndarray,
    pressure: np.ndarray,
    config: pressure_audit.PressureStabilizationConfig,
) -> dict[str, Any]:
    wet_fraction = np.asarray([0.25, 1.0], dtype=float)
    faces = pressure_audit.reconstruct_cut_adjacent_faces(
        tets,
        wet_fraction,
        config,
        full_wet_tolerance=1.0e-12,
    )
    if len(faces) != 1:
        raise RuntimeError(f"Expected exactly one cut-adjacent face, got {len(faces)}")
    report = pressure_audit.face_report(
        faces[0],
        points=points,
        tets=tets,
        previous_pressure=pressure,
        current_pressure=pressure,
        pressure_delta=np.zeros_like(pressure),
        previous_wet_fraction=wet_fraction,
        current_wet_fraction=wet_fraction,
        config=config,
    )
    return report


def constrained_pressure(
    exact_pressure: np.ndarray,
    active_vertices: set[int],
) -> tuple[np.ndarray, list[int]]:
    constrained = exact_pressure.copy()
    constrained_vertices = [
        vertex for vertex in range(constrained.size) if vertex not in active_vertices
    ]
    for vertex in constrained_vertices:
        constrained[vertex] = 0.0
    return constrained, constrained_vertices


def patch_case_report(
    *,
    name: str,
    support_mode: str,
    active_vertices: set[int],
    expect_zero_ghost_penalty: bool,
    expect_hydrostatic_balance: bool,
    apply_cut_adjacent_face: bool = True,
    active_volume_fractions: np.ndarray | None = None,
) -> dict[str, Any]:
    points = patch_points()
    tets = patch_tets()
    exact_pressure = linear_pressure(points)
    config = pressure_audit.PressureStabilizationConfig(
        viscosity_pa_s=0.001003,
        pressure_penalty=1.0,
        use_cut_metadata_scale=True,
        metadata_scale_cap=3.0,
    )
    pressure, constrained_vertices = constrained_pressure(
        exact_pressure,
        active_vertices,
    )
    if active_volume_fractions is None:
        active_volume_fractions = np.asarray([0.25, 1.0], dtype=float)
    report = (
        face_energy_proxy(points, tets, pressure, config)
        if apply_cut_adjacent_face
        else None
    )
    grad_jump_norm = (
        report["grad_jump_current_norm_pa_per_m"] if report is not None else 0.0
    )
    energy = report["current_energy_proxy"] if report is not None else 0.0
    zero_tol = 1.0e-12
    preserves_linear_state = grad_jump_norm <= zero_tol and energy <= zero_tol
    pspg_report = pspg_hydrostatic_balance_report(
        points=points,
        tets=tets,
        exact_pressure=exact_pressure,
        constrained_pressure_values=pressure,
        active_volume_fractions=active_volume_fractions,
    )
    ghost_penalty_passed = (
        preserves_linear_state if expect_zero_ghost_penalty else not preserves_linear_state
    )
    pspg_passed = (
        pspg_report["preserves_hydrostatic_balance"]
        if expect_hydrostatic_balance
        else not pspg_report["preserves_hydrostatic_balance"]
    )
    passed = ghost_penalty_passed and pspg_passed
    return {
        "name": name,
        "support_mode": support_mode,
        "cut_adjacent_face_applied": apply_cut_adjacent_face,
        "active_vertices": sorted(active_vertices),
        "constrained_vertices": constrained_vertices,
        "exact_pressure_pa": [float(value) for value in exact_pressure.tolist()],
        "constrained_pressure_pa": [float(value) for value in pressure.tolist()],
        "expect_zero_ghost_penalty": expect_zero_ghost_penalty,
        "preserves_linear_pressure_state": preserves_linear_state,
        "expect_hydrostatic_balance": expect_hydrostatic_balance,
        "ghost_penalty_passed": ghost_penalty_passed,
        "pspg_hydrostatic_balance_passed": pspg_passed,
        "passed": passed,
        "pspg_hydrostatic_balance": pspg_report,
        "face": (
            {
                "grad_jump_current_pa_per_m": report["grad_jump_current_pa_per_m"],
                "grad_jump_current_norm_pa_per_m": grad_jump_norm,
                "current_energy_proxy": energy,
                "raw_metadata_scale": report["raw_metadata_scale"],
                "applied_metadata_scale": report["applied_metadata_scale"],
                "h_normal_m": report["h_normal_m"],
                "face_area_m2": report["face_area_m2"],
            }
            if report is not None
            else None
        ),
    }


def audit_patch() -> dict[str, Any]:
    retained = patch_case_report(
        name="retained_cut_volume_support",
        support_mode="retained_cut_volume+cut_adjacent_facets",
        active_vertices={0, 1, 2, 3, 4},
        expect_zero_ghost_penalty=True,
        expect_hydrostatic_balance=True,
    )
    trace_only = patch_case_report(
        name="pruned_trace_only_cut_adjacent_support",
        support_mode="cell_patch+cut_adjacent_facets",
        active_vertices={0, 1, 2},
        expect_zero_ghost_penalty=False,
        expect_hydrostatic_balance=False,
    )
    full_volume_topology = patch_case_report(
        name="full_volume_one_cell_boundary_topology",
        support_mode="full_active_volume+one_cell_boundary_rows",
        active_vertices={0, 1, 2, 3, 4},
        expect_zero_ghost_penalty=True,
        expect_hydrostatic_balance=True,
        active_volume_fractions=np.asarray([1.0, 1.0], dtype=float),
    )
    fixed_skip = patch_case_report(
        name="fixed_pruned_cut_adjacent_support_skipped",
        support_mode="cell_patch+cut_adjacent_facets_skipped_no_retained_volume",
        active_vertices=set(),
        expect_zero_ghost_penalty=True,
        expect_hydrostatic_balance=True,
        apply_cut_adjacent_face=False,
        active_volume_fractions=np.asarray([0.0, 0.0], dtype=float),
    )
    hazard_detected = not trace_only["preserves_linear_pressure_state"]
    pspg_hydrostatic_hazard_detected = not trace_only[
        "pspg_hydrostatic_balance"
    ]["preserves_hydrostatic_balance"]
    passed = retained["passed"] and trace_only["passed"] and fixed_skip["passed"]
    return {
        "status": "diagnostic_linear_pressure_cut_volume_patch",
        "finding": (
            "Retained cut-volume pressure support preserves the linear pressure "
            "patch, while trace-only pruned cut-adjacent support with inactive "
            "dry vertices constrained to zero creates a nonzero pressure "
            "gradient jump. Skipping cut-adjacent support when no retained "
            "generated volume remains removes that ghost-penalty coupling."
        ),
        "pspg_pressure_gradient_finding": (
            "The retained patch also preserves hydrostatic balance for the "
            "direct PSPG pressure-gradient split when the matching body-force "
            "part uses the same active-volume support. The direct pressure "
            "gradient block still has nonzero boundary-row action and a "
            "constant-pressure null row sum; it only cancels in the full "
            "hydrostatic residual. If dry pressure vertices are constrained "
            "while the active-volume operator still sees the retained cells, "
            "that cancellation is broken."
        ),
        "pspg_boundary_solve_amplification_finding": (
            "The retained direct PSPG pressure-gradient block can preserve "
            "hydrostatic cancellation and the constant-pressure null while "
            "still amplifying zero-mean row loads at one-cell boundary rows. "
            "A full-volume topology control keeps that amplification class, "
            "so it is not only a tiny cut-fraction effect. Uniform scaling "
            "reduces absolute response but leaves the boundary/interior "
            "response ratio unchanged."
        ),
        "pspg_boundary_pair_completion_finding": (
            "A diagnostic one-cell boundary pair-completion probe adds a "
            "constant-null pressure edge between one-cell boundary rows. It "
            "preserves hydrostatic cancellation and the constant-pressure null "
            "while reducing the boundary/shared response ratio, so the patch "
            "points toward a topology-changing pressure-support fix rather "
            "than a residual-state change or uniform scalar multiplier."
        ),
        "pspg_boundary_shared_support_completion_finding": (
            "A diagnostic one-cell-to-shared support completion adds "
            "constant-null edges from one-cell boundary rows to multi-cell "
            "shared rows. It preserves hydrostatic cancellation and the "
            "constant-pressure null, but it is weaker than the boundary-pair "
            "or broader active completion in this patch. That rules out a "
            "shared-support-only topology as the strongest patch target."
        ),
        "pspg_boundary_active_completion_finding": (
            "A diagnostic one-cell-to-active support completion distributes "
            "constant-null edges from each one-cell boundary row to every other "
            "active row. It preserves hydrostatic cancellation and the "
            "constant-pressure null and gives the strongest response-ratio "
            "reduction in the full-volume topology control, but it is broader "
            "than a local existing-edge or pair-completion rule and should "
            "remain a formulation target, not a post-assembly mutation."
        ),
        "pspg_shared_row_schur_completion_finding": (
            "A diagnostic shared-row Schur support completion derives "
            "neighbor-neighbor pressure edges from the existing PSPG "
            "pressure-gradient graph using w_i*w_j/sum(w) around each shared "
            "support row. It preserves hydrostatic cancellation and the "
            "constant-pressure null, nearly matches pair completion on the "
            "retained patch, and is stronger than pair completion on the "
            "full-volume topology control. This supports a formulation-derived "
            "topology completion rule over simple incident-count scaling."
        ),
        "pspg_existing_edge_support_balance_finding": (
            "A diagnostic existing-edge support-balance probe reconstructs the "
            "direct PSPG pressure-gradient block as a constant-null graph on "
            "the current pressure edges, then scales only edges incident to "
            "weak-support rows until they match the strongest row support. It "
            "preserves hydrostatic cancellation and the constant-pressure null "
            "while reducing the response ratio, so existing active-volume edge "
            "weights are causal. It still changes weights by much more than a "
            "uniform multiplier and does not add missing topology, so it is a "
            "prototype target rather than a default physics change."
        ),
        "pspg_weak_boundary_existing_edge_support_balance_finding": (
            "A diagnostic weak-boundary existing-edge balance scales only "
            "existing edges incident to one-cell boundary rows. It preserves "
            "hydrostatic cancellation and the constant-pressure null and "
            "reduces the retained-support max response, but it worsens the "
            "full-volume boundary/shared response ratio. This rules out a "
            "plain weak-boundary existing-edge balance as the missing local "
            "rule by itself."
        ),
        "pspg_schur_shared_support_edge_balance_finding": (
            "A diagnostic Schur-plus-shared-support balance first adds the "
            "shared-row Schur topology and then scales existing pressure edges "
            "only for shared-support rows. It preserves hydrostatic "
            "cancellation and the constant-pressure null, but it worsens the "
            "Schur-only response ratio on both retained and full-volume patch "
            "controls. This rules out balancing the shared-support side as the "
            "selective local rule and sharpens the target to one-cell boundary "
            "support deficiency after topology completion."
        ),
        "pspg_schur_weak_boundary_edge_balance_finding": (
            "A diagnostic Schur-plus-weak-boundary balance first adds the "
            "shared-row Schur topology and then scales existing pressure edges "
            "only for one-cell boundary rows. It preserves hydrostatic "
            "cancellation and the constant-pressure null while improving on "
            "both Schur-only and broad Schur-plus-balance patch ratios. This "
            "supports a selective formulation-side edge-balance rule rather "
            "than global or broadly thresholded post-assembly pressure graph "
            "regularization."
        ),
        "pspg_direct_support_gap_or_same_sign_patch_finding": (
            "A diagnostic direct-support-gap plus same-sign-patch completion "
            "derives weak rows from low direct PSPG row support and derives "
            "the coupled patch from the pressure graph connected to those "
            "rows. It preserves hydrostatic cancellation and the "
            "constant-pressure null while matching the best selective Schur "
            "plus gap-row balance result without using source-mesh one-cell "
            "incident counts as the selector."
        ),
        "pspg_incident_support_count_balance_finding": (
            "A diagnostic incident-support count balance scales existing "
            "pressure-gradient edges using only endpoint active incident-cell "
            "count deficits. It preserves hydrostatic cancellation and the "
            "constant-pressure null and reduces the one-cell boundary response, "
            "but it is much weaker than row-abs existing-edge balancing on the "
            "retained patch and weaker than pair or active completion on the "
            "full-volume topology control. This rules out simple incident-cell "
            "count normalization as the missing pressure-gradient support rule."
        ),
        "passed": passed,
        "hazard_detected": hazard_detected,
        "pspg_hydrostatic_hazard_detected": pspg_hydrostatic_hazard_detected,
        "hazard": (
            "Trace-only cut-adjacent pressure support can make an otherwise "
            "linear hydrostatic pressure state inconsistent with the pressure "
            "ghost penalty when off-trace dry pressure DOFs are constrained."
        ),
        "pspg_hydrostatic_hazard": (
            "A retained active-volume PSPG pressure-gradient operator is "
            "hydrostatic-consistent only when the pressure state and the "
            "nonpressure body-force residual use matching active support."
        ),
        "cases": [retained, trace_only, full_volume_topology, fixed_skip],
    }


def main() -> int:
    args = parse_args()
    report = audit_patch()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.fail_on_hazard and report["hazard_detected"]:
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
