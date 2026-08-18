#!/usr/bin/env python3
"""Reconstruct a cut-adjacent pressure ghost-penalty audit from saved VTUs."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


TETRA_VTK_CELL_TYPE = 10
TETRA_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


@dataclass(frozen=True)
class PressureStabilizationConfig:
    viscosity_pa_s: float = 0.001003
    pressure_penalty: float = 1.0
    use_cut_metadata_scale: bool = False
    metadata_scale_cap: float | None = None
    global_metadata_scale_cap: float = 1.0e3
    stabilization_epsilon: float = 1.0e-12


@dataclass(frozen=True)
class CutAdjacentFace:
    face_index: int
    point_indices: tuple[int, int, int]
    first_cell: int
    second_cell: int
    first_cell_cut: bool
    second_cell_cut: bool
    raw_metadata_scale: float
    applied_metadata_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct cut-adjacent faces from saved WetVolumeFraction data "
            "and report pressure-gradient jump proxies for the pressure ghost "
            "penalty. This is an offline diagnostic; it is not an exact "
            "assembled residual dump."
        )
    )
    parser.add_argument("--previous-result", required=True, type=Path)
    parser.add_argument("--current-result", required=True, type=Path)
    parser.add_argument("--solver-xml", type=Path)
    parser.add_argument("--top-faces", type=int, default=20)
    parser.add_argument("--active-fluid-threshold", type=float, default=0.5)
    parser.add_argument("--tiny-wet-fraction", type=float, default=1.0e-4)
    parser.add_argument("--full-wet-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--viscosity", type=float)
    parser.add_argument("--pressure-penalty", type=float)
    parser.add_argument("--use-cut-metadata-scale", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--metadata-scale-cap", type=float)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def xml_text(root: ET.Element, tag: str) -> str | None:
    node = root.find(f".//{tag}")
    return None if node is None or node.text is None else node.text.strip()


def viscosity_value_text(root: ET.Element) -> str | None:
    node = root.find(".//Viscosity/Value")
    return None if node is None or node.text is None else node.text.strip()


def parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_float(raw: str | None, default: float | None) -> float | None:
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def load_config(
    solver_xml: Path | None,
    *,
    viscosity_override: float | None,
    pressure_penalty_override: float | None,
    use_metadata_override: str,
    metadata_scale_cap_override: float | None,
) -> PressureStabilizationConfig:
    config = PressureStabilizationConfig()
    if solver_xml is not None and solver_xml.exists():
        root = ET.parse(solver_xml).getroot()
        viscosity = parse_float(viscosity_value_text(root), config.viscosity_pa_s)
        pressure_penalty = parse_float(
            xml_text(root, "Cut_cell_pressure_gradient_penalty"),
            config.pressure_penalty,
        )
        config = PressureStabilizationConfig(
            viscosity_pa_s=(
                config.viscosity_pa_s if viscosity is None else viscosity
            ),
            pressure_penalty=(
                config.pressure_penalty
                if pressure_penalty is None
                else pressure_penalty
            ),
            use_cut_metadata_scale=parse_bool(
                xml_text(root, "Use_cut_metadata_scale"),
                config.use_cut_metadata_scale,
            ),
            metadata_scale_cap=parse_float(
                xml_text(root, "Cut_cell_metadata_scale_cap"),
                config.metadata_scale_cap,
            ),
            global_metadata_scale_cap=config.global_metadata_scale_cap,
            stabilization_epsilon=config.stabilization_epsilon,
        )
    if viscosity_override is not None:
        config = PressureStabilizationConfig(
            viscosity_pa_s=viscosity_override,
            pressure_penalty=config.pressure_penalty,
            use_cut_metadata_scale=config.use_cut_metadata_scale,
            metadata_scale_cap=config.metadata_scale_cap,
            global_metadata_scale_cap=config.global_metadata_scale_cap,
            stabilization_epsilon=config.stabilization_epsilon,
        )
    if pressure_penalty_override is not None:
        config = PressureStabilizationConfig(
            viscosity_pa_s=config.viscosity_pa_s,
            pressure_penalty=pressure_penalty_override,
            use_cut_metadata_scale=config.use_cut_metadata_scale,
            metadata_scale_cap=config.metadata_scale_cap,
            global_metadata_scale_cap=config.global_metadata_scale_cap,
            stabilization_epsilon=config.stabilization_epsilon,
        )
    if use_metadata_override != "auto":
        config = PressureStabilizationConfig(
            viscosity_pa_s=config.viscosity_pa_s,
            pressure_penalty=config.pressure_penalty,
            use_cut_metadata_scale=use_metadata_override == "true",
            metadata_scale_cap=config.metadata_scale_cap,
            global_metadata_scale_cap=config.global_metadata_scale_cap,
            stabilization_epsilon=config.stabilization_epsilon,
        )
    if metadata_scale_cap_override is not None:
        config = PressureStabilizationConfig(
            viscosity_pa_s=config.viscosity_pa_s,
            pressure_penalty=config.pressure_penalty,
            use_cut_metadata_scale=config.use_cut_metadata_scale,
            metadata_scale_cap=metadata_scale_cap_override,
            global_metadata_scale_cap=config.global_metadata_scale_cap,
            stabilization_epsilon=config.stabilization_epsilon,
        )
    return config


def tetra_connectivity(grid: pv.DataSet) -> np.ndarray:
    if not hasattr(grid, "celltypes"):
        raise RuntimeError("Expected an unstructured VTU grid with cell types")
    celltypes = np.asarray(grid.celltypes, dtype=np.int64)
    if np.any(celltypes != TETRA_VTK_CELL_TYPE):
        raise RuntimeError(
            "Pressure stabilization audit currently expects tetrahedral cells only"
        )
    cells = np.asarray(grid.cells, dtype=np.int64).reshape((-1, 5))
    if np.any(cells[:, 0] != 4):
        raise RuntimeError("Tetrahedral cell connectivity must have four point ids")
    return cells[:, 1:]


def tetra_volume(points: np.ndarray, tet: np.ndarray) -> float:
    x0, x1, x2, x3 = points[tet]
    return abs(float(np.dot(x1 - x0, np.cross(x2 - x0, x3 - x0)))) / 6.0


def triangle_area(points: np.ndarray, face: tuple[int, int, int]) -> float:
    x0, x1, x2 = points[list(face)]
    return 0.5 * float(np.linalg.norm(np.cross(x1 - x0, x2 - x0)))


def tetra_face_height(points: np.ndarray, tet: np.ndarray, face: tuple[int, int, int]) -> float:
    area = triangle_area(points, face)
    if area <= 0.0:
        return 0.0
    return 3.0 * tetra_volume(points, tet) / area


def tetra_gradient(points: np.ndarray, tet: np.ndarray, values: np.ndarray) -> np.ndarray:
    x0 = points[int(tet[0])]
    matrix = np.asarray([points[int(node)] - x0 for node in tet[1:]], dtype=float)
    rhs = np.asarray([values[int(node)] - values[int(tet[0])] for node in tet[1:]], dtype=float)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.full(3, math.nan, dtype=float)


def raw_scale_for_cells(
    wet_fraction: np.ndarray,
    first_cell: int,
    second_cell: int,
    global_cap: float,
    full_wet_tolerance: float,
) -> float:
    raw = 0.0
    for cell in (first_cell, second_cell):
        fraction = float(wet_fraction[cell])
        if fraction <= 0.0 or fraction >= 1.0 - full_wet_tolerance:
            continue
        raw = max(raw, 1.0 / max(fraction, 1.0e-12))
    return min(raw, global_cap) if raw > 0.0 else 0.0


def applied_scale(raw_scale: float, config: PressureStabilizationConfig) -> float:
    if not config.use_cut_metadata_scale:
        return 1.0
    scale = raw_scale
    if config.metadata_scale_cap is not None:
        scale = min(scale, config.metadata_scale_cap)
    return scale


def reconstruct_cut_adjacent_faces(
    tets: np.ndarray,
    wet_fraction: np.ndarray,
    config: PressureStabilizationConfig,
    *,
    full_wet_tolerance: float,
) -> list[CutAdjacentFace]:
    face_to_cells: dict[tuple[int, int, int], list[int]] = {}
    for cell_index, tet in enumerate(tets):
        for local_face in TETRA_FACES:
            face = tuple(sorted(int(tet[i]) for i in local_face))
            face_to_cells.setdefault(face, []).append(cell_index)

    out: list[CutAdjacentFace] = []
    for face, cells in face_to_cells.items():
        if len(cells) != 2:
            continue
        first_cell, second_cell = cells
        first_fraction = float(wet_fraction[first_cell])
        second_fraction = float(wet_fraction[second_cell])
        if first_fraction <= 0.0 or second_fraction <= 0.0:
            continue
        first_cut = first_fraction < 1.0 - full_wet_tolerance
        second_cut = second_fraction < 1.0 - full_wet_tolerance
        if not first_cut and not second_cut:
            continue
        raw_scale = raw_scale_for_cells(
            wet_fraction,
            first_cell,
            second_cell,
            config.global_metadata_scale_cap,
            full_wet_tolerance,
        )
        out.append(
            CutAdjacentFace(
                face_index=len(out),
                point_indices=face,
                first_cell=first_cell,
                second_cell=second_cell,
                first_cell_cut=first_cut,
                second_cell_cut=second_cut,
                raw_metadata_scale=raw_scale,
                applied_metadata_scale=applied_scale(raw_scale, config),
            )
        )
    return out


def support_class(
    *,
    phi: float | None,
    active_fluid: float | None,
    incident_wet_fraction_max: float | None,
    incident_wet_fraction_min_positive: float | None,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> str:
    active_by_field = active_fluid is not None and active_fluid > active_threshold
    active_by_phi = phi is not None and phi <= 0.0
    has_wet_fraction = (
        incident_wet_fraction_max is not None
        and math.isfinite(incident_wet_fraction_max)
        and incident_wet_fraction_max > 0.0
    )
    if has_wet_fraction:
        if incident_wet_fraction_max <= tiny_wet_fraction:
            return "tiny_cut_supported"
        if (
            incident_wet_fraction_min_positive is not None
            and math.isfinite(incident_wet_fraction_min_positive)
            and incident_wet_fraction_min_positive >= 1.0 - full_wet_tolerance
        ):
            return "full_wet_supported"
        return "cut_supported"
    if active_by_field or active_by_phi:
        return "active_without_wet_fraction_data"
    return "dry_or_inactive"


def point_wet_support(
    n_points: int,
    tets: np.ndarray,
    wet_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    max_fraction = np.full(n_points, math.nan, dtype=float)
    min_positive = np.full(n_points, math.nan, dtype=float)
    incident_count = np.zeros(n_points, dtype=np.int64)
    positive_count = np.zeros(n_points, dtype=np.int64)
    for cell_index, tet in enumerate(tets):
        fraction = float(wet_fraction[cell_index])
        for point_index in tet:
            incident_count[int(point_index)] += 1
        if not math.isfinite(fraction) or fraction <= 0.0:
            continue
        for point_index in tet:
            point = int(point_index)
            positive_count[point] += 1
            max_fraction[point] = (
                fraction
                if math.isnan(max_fraction[point])
                else max(max_fraction[point], fraction)
            )
            min_positive[point] = (
                fraction
                if math.isnan(min_positive[point])
                else min(min_positive[point], fraction)
            )
    return {
        "incident_wet_fraction_max": max_fraction,
        "incident_wet_fraction_min_positive": min_positive,
        "incident_cell_count": incident_count,
        "positive_wet_incident_cell_count": positive_count,
    }


def finite_float(value: float | np.floating[Any]) -> float | None:
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def pressure_update_event(
    grid: pv.DataSet,
    tets: np.ndarray,
    delta: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    *,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, Any]:
    support = point_wet_support(
        int(grid.n_points),
        tets,
        np.asarray(grid.cell_data["WetVolumeFraction"], dtype=float).reshape(-1),
    )
    phi = (
        np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
        if "phi" in grid.point_data
        else np.full(grid.n_points, math.nan, dtype=float)
    )
    active = (
        np.asarray(grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
        if "ActiveFluid" in grid.point_data
        else np.full(grid.n_points, math.nan, dtype=float)
    )
    max_wet = support["incident_wet_fraction_max"]
    active_or_wet = (active > active_threshold) | (phi <= 0.0) | (
        np.isfinite(max_wet) & (max_wet > 0.0)
    )
    if not np.any(active_or_wet):
        point_index = int(np.argmax(np.abs(delta)))
    else:
        indices = np.flatnonzero(active_or_wet)
        point_index = int(indices[int(np.argmax(np.abs(delta[indices])))])

    phi_value = finite_float(phi[point_index])
    active_value = finite_float(active[point_index])
    max_wet_value = finite_float(support["incident_wet_fraction_max"][point_index])
    min_positive_value = finite_float(
        support["incident_wet_fraction_min_positive"][point_index]
    )
    return {
        "point_index": point_index,
        "point_m": [float(value) for value in np.asarray(grid.points)[point_index].tolist()],
        "pressure_delta_pa": float(delta[point_index]),
        "abs_pressure_delta_pa": float(abs(delta[point_index])),
        "from_pressure_pa": float(previous_pressure[point_index]),
        "to_pressure_pa": float(current_pressure[point_index]),
        "phi": phi_value,
        "active_fluid": active_value,
        "support_class": support_class(
            phi=phi_value,
            active_fluid=active_value,
            incident_wet_fraction_max=max_wet_value,
            incident_wet_fraction_min_positive=min_positive_value,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        ),
        "incident_cell_count": int(support["incident_cell_count"][point_index]),
        "positive_wet_incident_cell_count": int(
            support["positive_wet_incident_cell_count"][point_index]
        ),
        "incident_wet_fraction_max": max_wet_value,
        "incident_wet_fraction_min_positive": min_positive_value,
    }


def coefficient(config: PressureStabilizationConfig, scale: float, h_normal: float) -> float:
    return (
        scale
        * config.pressure_penalty
        * h_normal**3
        / (config.viscosity_pa_s + config.stabilization_epsilon)
    )


def face_report(
    face: CutAdjacentFace,
    *,
    points: np.ndarray,
    tets: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    pressure_delta: np.ndarray,
    previous_wet_fraction: np.ndarray,
    current_wet_fraction: np.ndarray,
    config: PressureStabilizationConfig,
) -> dict[str, Any]:
    first_tet = tets[face.first_cell]
    second_tet = tets[face.second_cell]
    grad_prev_first = tetra_gradient(points, first_tet, previous_pressure)
    grad_prev_second = tetra_gradient(points, second_tet, previous_pressure)
    grad_current_first = tetra_gradient(points, first_tet, current_pressure)
    grad_current_second = tetra_gradient(points, second_tet, current_pressure)
    grad_delta_first = tetra_gradient(points, first_tet, pressure_delta)
    grad_delta_second = tetra_gradient(points, second_tet, pressure_delta)
    grad_jump_previous = grad_prev_second - grad_prev_first
    grad_jump_current = grad_current_second - grad_current_first
    grad_jump_delta = grad_delta_second - grad_delta_first

    area = triangle_area(points, face.point_indices)
    h_first = tetra_face_height(points, first_tet, face.point_indices)
    h_second = tetra_face_height(points, second_tet, face.point_indices)
    h_normal = 0.5 * (h_first + h_second)
    coeff = coefficient(config, face.applied_metadata_scale, h_normal)
    face_nodes = np.asarray(face.point_indices, dtype=np.int64)
    adjacent_nodes = np.unique(np.concatenate([first_tet, second_tet]))
    current_jump_norm = float(np.linalg.norm(grad_jump_current))
    delta_jump_norm = float(np.linalg.norm(grad_jump_delta))

    return {
        "face_index": face.face_index,
        "point_indices": list(face.point_indices),
        "centroid_m": [
            float(value) for value in np.mean(points[list(face.point_indices)], axis=0).tolist()
        ],
        "first_cell": face.first_cell,
        "second_cell": face.second_cell,
        "first_cell_cut": face.first_cell_cut,
        "second_cell_cut": face.second_cell_cut,
        "previous_wet_fraction": [
            float(previous_wet_fraction[face.first_cell]),
            float(previous_wet_fraction[face.second_cell]),
        ],
        "current_wet_fraction": [
            float(current_wet_fraction[face.first_cell]),
            float(current_wet_fraction[face.second_cell]),
        ],
        "raw_metadata_scale": face.raw_metadata_scale,
        "applied_metadata_scale": face.applied_metadata_scale,
        "face_area_m2": area,
        "h_normal_m": h_normal,
        "coefficient_proxy": coeff,
        "grad_jump_previous_pa_per_m": [float(value) for value in grad_jump_previous.tolist()],
        "grad_jump_current_pa_per_m": [float(value) for value in grad_jump_current.tolist()],
        "grad_jump_delta_pa_per_m": [float(value) for value in grad_jump_delta.tolist()],
        "grad_jump_current_norm_pa_per_m": current_jump_norm,
        "grad_jump_delta_norm_pa_per_m": delta_jump_norm,
        "current_energy_proxy": float(area * coeff * current_jump_norm**2),
        "delta_energy_proxy": float(area * coeff * delta_jump_norm**2),
        "max_abs_pressure_delta_face_nodes_pa": float(
            np.max(np.abs(pressure_delta[face_nodes]))
        ),
        "max_abs_pressure_delta_adjacent_cell_nodes_pa": float(
            np.max(np.abs(pressure_delta[adjacent_nodes]))
        ),
    }


def summarize_node_correlation(
    point_index: int,
    face_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    incident = [
        report
        for report in face_reports
        if point_index in set(report["point_indices"])
    ]
    if not incident:
        return {
            "point_index": point_index,
            "incident_cut_adjacent_face_count": 0,
            "sum_current_energy_proxy": 0.0,
            "sum_delta_energy_proxy": 0.0,
            "max_incident_current_energy_proxy": 0.0,
            "max_incident_delta_energy_proxy": 0.0,
            "best_current_energy_rank": None,
            "best_delta_energy_rank": None,
        }

    current_rank = {
        report["face_index"]: rank
        for rank, report in enumerate(
            sorted(face_reports, key=lambda item: -item["current_energy_proxy"]),
            start=1,
        )
    }
    delta_rank = {
        report["face_index"]: rank
        for rank, report in enumerate(
            sorted(face_reports, key=lambda item: -item["delta_energy_proxy"]),
            start=1,
        )
    }
    return {
        "point_index": point_index,
        "incident_cut_adjacent_face_count": len(incident),
        "sum_current_energy_proxy": float(
            sum(report["current_energy_proxy"] for report in incident)
        ),
        "sum_delta_energy_proxy": float(
            sum(report["delta_energy_proxy"] for report in incident)
        ),
        "max_incident_current_energy_proxy": float(
            max(report["current_energy_proxy"] for report in incident)
        ),
        "max_incident_delta_energy_proxy": float(
            max(report["delta_energy_proxy"] for report in incident)
        ),
        "best_current_energy_rank": min(
            current_rank[report["face_index"]] for report in incident
        ),
        "best_delta_energy_rank": min(
            delta_rank[report["face_index"]] for report in incident
        ),
        "incident_face_indices": [int(report["face_index"]) for report in incident],
    }


def driver_assessment(
    *,
    face_reports: list[dict[str, Any]],
    worst_update: dict[str, Any],
    node_correlation: dict[str, Any],
) -> dict[str, Any]:
    incident_count = int(node_correlation.get("incident_cut_adjacent_face_count") or 0)
    face_count = len(face_reports)
    worst_update_pa = float(worst_update.get("abs_pressure_delta_pa") or 0.0)
    max_incident_delta = float(
        node_correlation.get("max_incident_delta_energy_proxy") or 0.0
    )
    if face_count == 0:
        classification = "no_cut_adjacent_pressure_stabilization_faces"
        direct_driver_supported = False
        direct_driver_ruled_out = True
    elif incident_count == 0:
        classification = "worst_update_not_incident_to_cut_adjacent_stabilization"
        direct_driver_supported = False
        direct_driver_ruled_out = True
    elif max_incident_delta <= 0.0:
        classification = "worst_update_incident_but_zero_delta_proxy"
        direct_driver_supported = False
        direct_driver_ruled_out = True
    else:
        classification = "worst_update_incident_to_cut_adjacent_stabilization"
        direct_driver_supported = True
        direct_driver_ruled_out = False

    return {
        "classification": classification,
        "direct_cut_adjacent_pressure_stabilization_driver_supported": (
            direct_driver_supported
        ),
        "direct_cut_adjacent_pressure_stabilization_driver_ruled_out": (
            direct_driver_ruled_out
        ),
        "worst_update_point_index": worst_update.get("point_index"),
        "worst_update_support_class": worst_update.get("support_class"),
        "worst_update_abs_pressure_delta_pa": worst_update_pa,
        "reconstructed_cut_adjacent_face_count": face_count,
        "incident_cut_adjacent_face_count": incident_count,
        "max_incident_delta_energy_proxy": max_incident_delta,
        "sum_incident_delta_energy_proxy": float(
            node_correlation.get("sum_delta_energy_proxy") or 0.0
        ),
        "best_incident_delta_energy_rank": node_correlation.get(
            "best_delta_energy_rank"
        ),
    }


def audit_pressure_stabilization(
    previous_result: Path,
    current_result: Path,
    *,
    config: PressureStabilizationConfig,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    top_faces: int,
    solver_xml: Path | None,
) -> dict[str, Any]:
    previous_grid = pv.read(previous_result)
    current_grid = pv.read(current_result)
    if previous_grid.n_points != current_grid.n_points or previous_grid.n_cells != current_grid.n_cells:
        raise RuntimeError("Previous and current VTUs must have matching topology")
    if "Pressure" not in previous_grid.point_data or "Pressure" not in current_grid.point_data:
        raise RuntimeError("Both VTUs must contain point-data Pressure")
    if "WetVolumeFraction" not in previous_grid.cell_data or "WetVolumeFraction" not in current_grid.cell_data:
        raise RuntimeError("Both VTUs must contain cell-data WetVolumeFraction")

    tets = tetra_connectivity(current_grid)
    points = np.asarray(current_grid.points, dtype=float)
    previous_pressure = np.asarray(previous_grid.point_data["Pressure"], dtype=float).reshape(-1)
    current_pressure = np.asarray(current_grid.point_data["Pressure"], dtype=float).reshape(-1)
    pressure_delta = current_pressure - previous_pressure
    previous_wet_fraction = np.asarray(
        previous_grid.cell_data["WetVolumeFraction"], dtype=float
    ).reshape(-1)
    current_wet_fraction = np.asarray(
        current_grid.cell_data["WetVolumeFraction"], dtype=float
    ).reshape(-1)

    faces = reconstruct_cut_adjacent_faces(
        tets,
        current_wet_fraction,
        config,
        full_wet_tolerance=full_wet_tolerance,
    )
    reports = [
        face_report(
            face,
            points=points,
            tets=tets,
            previous_pressure=previous_pressure,
            current_pressure=current_pressure,
            pressure_delta=pressure_delta,
            previous_wet_fraction=previous_wet_fraction,
            current_wet_fraction=current_wet_fraction,
            config=config,
        )
        for face in faces
    ]
    top_current = sorted(
        reports,
        key=lambda item: -item["current_energy_proxy"],
    )[:top_faces]
    top_delta = sorted(
        reports,
        key=lambda item: -item["delta_energy_proxy"],
    )[:top_faces]
    worst_update = pressure_update_event(
        current_grid,
        tets,
        pressure_delta,
        previous_pressure,
        current_pressure,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
    )
    node_correlation = summarize_node_correlation(worst_update["point_index"], reports)
    assessment = driver_assessment(
        face_reports=reports,
        worst_update=worst_update,
        node_correlation=node_correlation,
    )

    if reports:
        worst_delta = max(reports, key=lambda item: item["delta_energy_proxy"])
        finding = (
            f"Reconstructed {len(reports)} active cut-adjacent faces. "
            f"Worst delta ghost-penalty proxy is face {worst_delta['face_index']} "
            f"with delta_energy_proxy={worst_delta['delta_energy_proxy']:.6g}; "
            f"worst active/wet pressure update is {worst_update['abs_pressure_delta_pa']:.6g} Pa "
            f"on {worst_update['support_class']} with "
            f"{node_correlation['incident_cut_adjacent_face_count']} incident cut-adjacent faces; "
            f"driver assessment is {assessment['classification']}."
        )
    else:
        finding = "No active cut-adjacent faces were reconstructed from saved wet fractions."

    cut_mask = (current_wet_fraction > 0.0) & (
        current_wet_fraction < 1.0 - full_wet_tolerance
    )
    return {
        "previous_result": str(previous_result),
        "current_result": str(current_result),
        "solver_xml": str(solver_xml) if solver_xml else None,
        "status": "diagnostic_cut_pressure_stabilization_contribution_proxy",
        "finding": finding,
        "limitations": (
            "Offline proxy reconstructed from saved P1 VTU fields. It identifies "
            "the same cut-adjacent face class and h^3/mu scaling, but it is not "
            "an exact assembled residual contribution and omits high-order "
            "pressure Hessian terms not present in the saved tetrahedral output."
        ),
        "configuration": {
            "viscosity_pa_s": config.viscosity_pa_s,
            "pressure_penalty": config.pressure_penalty,
            "use_cut_metadata_scale": config.use_cut_metadata_scale,
            "metadata_scale_cap": config.metadata_scale_cap,
            "global_metadata_scale_cap": config.global_metadata_scale_cap,
            "stabilization_epsilon": config.stabilization_epsilon,
        },
        "mesh_summary": {
            "point_count": int(current_grid.n_points),
            "cell_count": int(current_grid.n_cells),
            "active_wet_cell_count": int(np.count_nonzero(current_wet_fraction > 0.0)),
            "active_cut_cell_count": int(np.count_nonzero(cut_mask)),
            "active_full_wet_cell_count": int(
                np.count_nonzero(current_wet_fraction >= 1.0 - full_wet_tolerance)
            ),
            "tiny_positive_wet_cell_count": int(
                np.count_nonzero(
                    (current_wet_fraction > 0.0)
                    & (current_wet_fraction <= tiny_wet_fraction)
                )
            ),
            "reconstructed_cut_adjacent_face_count": len(reports),
        },
        "worst_active_or_wet_pressure_update": worst_update,
        "worst_update_cut_adjacent_correlation": node_correlation,
        "direct_driver_assessment": assessment,
        "top_faces_by_delta_energy_proxy": top_delta,
        "top_faces_by_current_energy_proxy": top_current,
    }


def main() -> int:
    args = parse_args()
    config = load_config(
        args.solver_xml,
        viscosity_override=args.viscosity,
        pressure_penalty_override=args.pressure_penalty,
        use_metadata_override=args.use_cut_metadata_scale,
        metadata_scale_cap_override=args.metadata_scale_cap,
    )
    report = audit_pressure_stabilization(
        args.previous_result,
        args.current_result,
        config=config,
        active_threshold=args.active_fluid_threshold,
        tiny_wet_fraction=args.tiny_wet_fraction,
        full_wet_tolerance=args.full_wet_tolerance,
        top_faces=args.top_faces,
        solver_xml=args.solver_xml,
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
