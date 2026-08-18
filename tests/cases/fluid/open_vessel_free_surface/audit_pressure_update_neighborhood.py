#!/usr/bin/env python3
"""Audit local pressure-update neighborhoods in paired VTU results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


RESULT_RE_TEMPLATE = r"{prefix}_(\d+)\.p?vtu$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a previous/current result pair and report nearest-neighbor "
            "and incident-cell patch pressure deltas around the largest selected "
            "pressure updates."
        )
    )
    parser.add_argument("--previous-result", type=Path, required=True)
    parser.add_argument("--current-result", type=Path, required=True)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--previous-time", type=float)
    parser.add_argument("--current-time", type=float)
    parser.add_argument("--top-events", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--neighbor-detail-limit", type=int, default=8)
    parser.add_argument("--patch-detail-limit", type=int, default=8)
    parser.add_argument("--active-fluid-threshold", type=float, default=0.5)
    parser.add_argument("--tiny-wet-fraction", type=float, default=1.0e-4)
    parser.add_argument("--full-wet-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "all_points",
            "active_or_wet_supported",
            "full_wet_supported",
            "cut_supported",
            "tiny_cut_supported",
        ),
        default="active_or_wet_supported",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def result_step(path: Path, prefix: str) -> int:
    match = re.match(RESULT_RE_TEMPLATE.format(prefix=re.escape(prefix)), path.name)
    return int(match.group(1)) if match else -1


def finite_float(value: Any) -> float | None:
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def cell_point_ids(grid: pv.DataSet) -> list[np.ndarray]:
    cells = np.asarray(grid.cells, dtype=np.int64)
    out: list[np.ndarray] = []
    offset = 0
    while offset < cells.size:
        node_count = int(cells[offset])
        out.append(cells[offset + 1 : offset + 1 + node_count])
        offset += node_count + 1
    return out


def point_to_cells(cells: list[np.ndarray], n_points: int) -> list[list[int]]:
    incident: list[list[int]] = [[] for _ in range(n_points)]
    for cell_index, points in enumerate(cells):
        for point in points:
            point_index = int(point)
            if 0 <= point_index < n_points:
                incident[point_index].append(cell_index)
    return incident


def point_wet_support(grid: pv.DataSet, cells: list[np.ndarray]) -> dict[str, np.ndarray]:
    n_points = int(grid.n_points)
    max_fraction = np.full(n_points, math.nan, dtype=float)
    min_positive = np.full(n_points, math.nan, dtype=float)
    incident_count = np.zeros(n_points, dtype=np.int64)
    positive_count = np.zeros(n_points, dtype=np.int64)

    if "WetVolumeFraction" not in grid.cell_data:
        return {
            "incident_wet_fraction_max": max_fraction,
            "incident_wet_fraction_min_positive": min_positive,
            "incident_cell_count": incident_count,
            "positive_wet_incident_cell_count": positive_count,
        }

    wet_fraction = np.asarray(grid.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
    for cell_index, points in enumerate(cells):
        if cell_index >= wet_fraction.size:
            break
        fraction = float(wet_fraction[cell_index])
        for point in points:
            point_index = int(point)
            incident_count[point_index] += 1
            if not math.isfinite(fraction) or fraction <= 0.0:
                continue
            positive_count[point_index] += 1
            current_max = max_fraction[point_index]
            max_fraction[point_index] = (
                fraction if math.isnan(current_max) else max(current_max, fraction)
            )
            current_min = min_positive[point_index]
            min_positive[point_index] = (
                fraction if math.isnan(current_min) else min(current_min, fraction)
            )

    return {
        "incident_wet_fraction_max": max_fraction,
        "incident_wet_fraction_min_positive": min_positive,
        "incident_cell_count": incident_count,
        "positive_wet_incident_cell_count": positive_count,
    }


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


def boundary_labels(
    point: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    *,
    tolerance: float = 1.0e-10,
) -> list[str]:
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    candidates = (
        ("x_min", x, bounds[0]),
        ("x_max", x, bounds[1]),
        ("y_min", y, bounds[2]),
        ("y_max", y, bounds[3]),
        ("z_min", z, bounds[4]),
        ("z_max", z, bounds[5]),
    )
    return [
        label
        for label, value, boundary in candidates
        if math.isfinite(value)
        and math.isfinite(boundary)
        and abs(value - boundary) <= tolerance
    ]


def boundary_class(labels: list[str]) -> str:
    if not labels:
        return "interior"
    if len(labels) == 1:
        return "boundary_face"
    if len(labels) == 2:
        return "boundary_edge"
    return "boundary_corner"


def array_or_nan(
    grid: pv.DataSet,
    name: str,
    *,
    components: int | None = None,
) -> np.ndarray:
    if name not in grid.point_data:
        if components is None:
            return np.full(grid.n_points, math.nan, dtype=float)
        return np.full((grid.n_points, components), math.nan, dtype=float)
    values = np.asarray(grid.point_data[name], dtype=float)
    if components is None:
        return values.reshape(-1)
    return values.reshape(-1, components)


def support_masks(
    *,
    phi: np.ndarray,
    active: np.ndarray,
    support: dict[str, np.ndarray],
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, np.ndarray]:
    max_wet = support["incident_wet_fraction_max"]
    min_positive = support["incident_wet_fraction_min_positive"]
    active_or_phi_wet = (active > active_threshold) | (phi <= 0.0)
    wet_supported = np.isfinite(max_wet) & (max_wet > 0.0)
    active_or_wet_supported = active_or_phi_wet | wet_supported
    full_wet_supported = (
        wet_supported
        & np.isfinite(min_positive)
        & (min_positive >= 1.0 - full_wet_tolerance)
    )
    cut_supported = wet_supported & ~full_wet_supported
    tiny_cut_supported = wet_supported & (max_wet <= tiny_wet_fraction)
    return {
        "all_points": np.ones(phi.shape[0], dtype=bool),
        "active_or_wet_supported": active_or_wet_supported,
        "full_wet_supported": full_wet_supported,
        "cut_supported": cut_supported,
        "tiny_cut_supported": tiny_cut_supported,
    }


def delta_statistics(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean_delta_pa": None,
            "median_delta_pa": None,
            "min_delta_pa": None,
            "max_delta_pa": None,
            "mean_abs_delta_pa": None,
            "median_abs_delta_pa": None,
            "max_abs_delta_pa": None,
            "rms_delta_pa": None,
        }
    return {
        "count": int(finite.size),
        "mean_delta_pa": float(np.mean(finite)),
        "median_delta_pa": float(np.median(finite)),
        "min_delta_pa": float(np.min(finite)),
        "max_delta_pa": float(np.max(finite)),
        "mean_abs_delta_pa": float(np.mean(np.abs(finite))),
        "median_abs_delta_pa": float(np.median(np.abs(finite))),
        "max_abs_delta_pa": float(np.max(np.abs(finite))),
        "rms_delta_pa": float(np.sqrt(np.mean(finite * finite))),
    }


def selected_delta_statistics(
    *,
    indices: np.ndarray,
    delta: np.ndarray,
    distances: np.ndarray | None,
    target_delta: float,
) -> dict[str, Any]:
    values = delta[indices] if indices.size else np.asarray([], dtype=float)
    stats = delta_statistics(values)
    target_sign = 1 if target_delta >= 0.0 else -1
    if values.size:
        same_sign = np.sign(values) == target_sign
        opposite_sign = np.sign(values) == -target_sign
        zero_sign = np.sign(values) == 0.0
        stats.update(
            {
                "same_sign_count": int(np.count_nonzero(same_sign)),
                "same_sign_fraction": float(np.count_nonzero(same_sign) / values.size),
                "opposite_sign_count": int(np.count_nonzero(opposite_sign)),
                "opposite_sign_fraction": float(
                    np.count_nonzero(opposite_sign) / values.size
                ),
                "zero_delta_count": int(np.count_nonzero(zero_sign)),
            }
        )
        median_abs = stats["median_abs_delta_pa"]
        max_abs = stats["max_abs_delta_pa"]
        stats["target_abs_to_median_abs_ratio"] = (
            abs(target_delta) / median_abs
            if isinstance(median_abs, (int, float)) and median_abs > 0.0
            else None
        )
        stats["target_abs_to_max_abs_ratio"] = (
            abs(target_delta) / max_abs
            if isinstance(max_abs, (int, float)) and max_abs > 0.0
            else None
        )
        if distances is not None and distances.size:
            stats["min_distance_m"] = float(np.min(distances[indices]))
            stats["max_distance_m"] = float(np.max(distances[indices]))
            stats["mean_distance_m"] = float(np.mean(distances[indices]))
    else:
        stats.update(
            {
                "same_sign_count": 0,
                "same_sign_fraction": None,
                "opposite_sign_count": 0,
                "opposite_sign_fraction": None,
                "zero_delta_count": 0,
                "target_abs_to_median_abs_ratio": None,
                "target_abs_to_max_abs_ratio": None,
            }
        )
    return stats


def point_record(
    *,
    point_index: int,
    points: np.ndarray,
    delta: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    phi: np.ndarray,
    active: np.ndarray,
    velocity: np.ndarray,
    support: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float, float, float],
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    distance_m: float | None = None,
) -> dict[str, Any]:
    max_wet = finite_float(support["incident_wet_fraction_max"][point_index])
    min_positive = finite_float(
        support["incident_wet_fraction_min_positive"][point_index]
    )
    labels = boundary_labels(points[point_index], bounds)
    record: dict[str, Any] = {
        "point_index": int(point_index),
        "point_m": [float(value) for value in points[point_index].tolist()],
        "pressure_delta_pa": float(delta[point_index]),
        "abs_pressure_delta_pa": float(abs(delta[point_index])),
        "from_pressure_pa": float(previous_pressure[point_index]),
        "to_pressure_pa": float(current_pressure[point_index]),
        "phi": finite_float(phi[point_index]),
        "active_fluid": finite_float(active[point_index]),
        "support_class": support_class(
            phi=finite_float(phi[point_index]),
            active_fluid=finite_float(active[point_index]),
            incident_wet_fraction_max=max_wet,
            incident_wet_fraction_min_positive=min_positive,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        ),
        "incident_cell_count": int(support["incident_cell_count"][point_index]),
        "positive_wet_incident_cell_count": int(
            support["positive_wet_incident_cell_count"][point_index]
        ),
        "incident_wet_fraction_max": max_wet,
        "incident_wet_fraction_min_positive": min_positive,
        "boundary_labels": labels,
        "boundary_class": boundary_class(labels),
    }
    if distance_m is not None:
        record["distance_m"] = float(distance_m)
    if velocity.shape[1] == 3 and np.all(np.isfinite(velocity[point_index])):
        record["velocity_m_per_s"] = [
            float(value) for value in velocity[point_index].tolist()
        ]
        record["speed_m_per_s"] = float(np.linalg.norm(velocity[point_index]))
    return record


def top_point_records(
    *,
    indices: np.ndarray,
    points: np.ndarray,
    delta: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    phi: np.ndarray,
    active: np.ndarray,
    velocity: np.ndarray,
    support: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float, float, float],
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    limit: int,
    distances: np.ndarray | None = None,
    sort_by: str = "abs_delta",
) -> list[dict[str, Any]]:
    if indices.size == 0 or limit <= 0:
        return []
    if sort_by == "distance":
        if distances is None:
            order = np.arange(indices.size)
        else:
            order = np.argsort(distances[indices], kind="stable")
    else:
        order = np.argsort(-np.abs(delta[indices]), kind="stable")
    records: list[dict[str, Any]] = []
    for point_index in indices[order[:limit]]:
        distance = None if distances is None else float(distances[point_index])
        records.append(
            point_record(
                point_index=int(point_index),
                points=points,
                delta=delta,
                previous_pressure=previous_pressure,
                current_pressure=current_pressure,
                phi=phi,
                active=active,
                velocity=velocity,
                support=support,
                bounds=bounds,
                active_threshold=active_threshold,
                tiny_wet_fraction=tiny_wet_fraction,
                full_wet_tolerance=full_wet_tolerance,
                distance_m=distance,
            )
        )
    return records


def incident_patch_indices(
    *,
    point_index: int,
    cells: list[np.ndarray],
    incident: list[list[int]],
) -> np.ndarray:
    patch: set[int] = set()
    for cell_index in incident[point_index]:
        patch.update(int(point) for point in cells[cell_index])
    patch.discard(point_index)
    return np.asarray(sorted(patch), dtype=np.int64)


def nearest_neighbor_indices(
    *,
    points: np.ndarray,
    point_index: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(points - points[point_index], axis=1)
    order = np.argsort(distances, kind="stable")
    neighbors = order[order != point_index][:count]
    return neighbors.astype(np.int64), distances


def event_neighborhood_report(
    *,
    rank: int,
    point_index: int,
    points: np.ndarray,
    delta: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    phi: np.ndarray,
    active: np.ndarray,
    velocity: np.ndarray,
    support: dict[str, np.ndarray],
    cells: list[np.ndarray],
    incident: list[list[int]],
    bounds: tuple[float, float, float, float, float, float],
    neighbor_count: int,
    neighbor_detail_limit: int,
    patch_detail_limit: int,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, Any]:
    neighbors, distances = nearest_neighbor_indices(
        points=points,
        point_index=point_index,
        count=neighbor_count,
    )
    patch = incident_patch_indices(
        point_index=point_index,
        cells=cells,
        incident=incident,
    )
    target_delta = float(delta[point_index])
    target = point_record(
        point_index=point_index,
        points=points,
        delta=delta,
        previous_pressure=previous_pressure,
        current_pressure=current_pressure,
        phi=phi,
        active=active,
        velocity=velocity,
        support=support,
        bounds=bounds,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
    )
    target["rank"] = int(rank)
    target["nearest_neighbor_count"] = int(neighbors.size)
    target["incident_patch_point_count"] = int(patch.size)
    target["incident_cell_ids"] = [int(cell_index) for cell_index in incident[point_index]]
    target["nearest_neighbors"] = selected_delta_statistics(
        indices=neighbors,
        delta=delta,
        distances=distances,
        target_delta=target_delta,
    )
    target["incident_patch"] = selected_delta_statistics(
        indices=patch,
        delta=delta,
        distances=distances,
        target_delta=target_delta,
    )
    target["nearest_neighbor_details"] = top_point_records(
        indices=neighbors,
        points=points,
        delta=delta,
        previous_pressure=previous_pressure,
        current_pressure=current_pressure,
        phi=phi,
        active=active,
        velocity=velocity,
        support=support,
        bounds=bounds,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
        limit=neighbor_detail_limit,
        distances=distances,
        sort_by="distance",
    )
    target["largest_patch_delta_details"] = top_point_records(
        indices=patch,
        points=points,
        delta=delta,
        previous_pressure=previous_pressure,
        current_pressure=current_pressure,
        phi=phi,
        active=active,
        velocity=velocity,
        support=support,
        bounds=bounds,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
        limit=patch_detail_limit,
        distances=distances,
    )
    return target


def max_event_for_mask(
    *,
    mask: np.ndarray,
    delta: np.ndarray,
    points: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    phi: np.ndarray,
    active: np.ndarray,
    velocity: np.ndarray,
    support: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float, float, float],
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, Any] | None:
    if not np.any(mask):
        return None
    candidates = np.flatnonzero(mask)
    point_index = int(candidates[int(np.argmax(np.abs(delta[candidates])))])
    return point_record(
        point_index=point_index,
        points=points,
        delta=delta,
        previous_pressure=previous_pressure,
        current_pressure=current_pressure,
        phi=phi,
        active=active,
        velocity=velocity,
        support=support,
        bounds=bounds,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
    )


def transition_neighborhood_report(
    previous_result: Path,
    current_result: Path,
    *,
    result_prefix: str,
    previous_time: float | None,
    current_time: float | None,
    top_events: int,
    neighbor_count: int,
    neighbor_detail_limit: int,
    patch_detail_limit: int,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    selection_mode: str,
) -> dict[str, Any]:
    if not previous_result.exists():
        raise FileNotFoundError(previous_result)
    if not current_result.exists():
        raise FileNotFoundError(current_result)

    previous_grid = pv.read(previous_result)
    current_grid = pv.read(current_result)
    if previous_grid.n_points != current_grid.n_points:
        raise RuntimeError(
            f"Cannot compare {previous_result} and {current_result}: "
            "point counts differ"
        )
    if "Pressure" not in previous_grid.point_data or "Pressure" not in current_grid.point_data:
        raise RuntimeError("Both result files must contain point-data Pressure")

    points = np.asarray(current_grid.points, dtype=float)
    previous_pressure = np.asarray(
        previous_grid.point_data["Pressure"], dtype=float
    ).reshape(-1)
    current_pressure = np.asarray(
        current_grid.point_data["Pressure"], dtype=float
    ).reshape(-1)
    delta = current_pressure - previous_pressure
    phi = array_or_nan(current_grid, "phi")
    active = array_or_nan(current_grid, "ActiveFluid")
    velocity = array_or_nan(current_grid, "Velocity", components=3)
    cells = cell_point_ids(current_grid)
    incident = point_to_cells(cells, current_grid.n_points)
    support = point_wet_support(current_grid, cells)
    masks = support_masks(
        phi=phi,
        active=active,
        support=support,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
    )
    selected_mask = masks[selection_mode]
    selected_indices = np.flatnonzero(selected_mask)
    top_indices = selected_indices[
        np.argsort(-np.abs(delta[selected_indices]), kind="stable")[:top_events]
    ]
    bounds = tuple(float(value) for value in current_grid.bounds)

    return {
        "previous_result": str(previous_result),
        "current_result": str(current_result),
        "from_result": previous_result.name,
        "to_result": current_result.name,
        "from_step": result_step(previous_result, result_prefix),
        "to_step": result_step(current_result, result_prefix),
        "from_time_s": previous_time,
        "to_time_s": current_time,
        "point_count": int(current_grid.n_points),
        "cell_count": int(current_grid.n_cells),
        "selection_mode": selection_mode,
        "selected_point_count": int(selected_indices.size),
        "top_event_count": int(top_indices.size),
        "neighbors_per_event": int(neighbor_count),
        "support_counts": {
            name: int(np.count_nonzero(mask))
            for name, mask in masks.items()
        },
        "delta_statistics_by_category": {
            name: delta_statistics(delta[mask])
            for name, mask in masks.items()
        },
        "max_by_category": {
            name: max_event_for_mask(
                mask=mask,
                delta=delta,
                points=points,
                previous_pressure=previous_pressure,
                current_pressure=current_pressure,
                phi=phi,
                active=active,
                velocity=velocity,
                support=support,
                bounds=bounds,
                active_threshold=active_threshold,
                tiny_wet_fraction=tiny_wet_fraction,
                full_wet_tolerance=full_wet_tolerance,
            )
            for name, mask in masks.items()
        },
        "top_update_neighborhoods": [
            event_neighborhood_report(
                rank=rank,
                point_index=int(point_index),
                points=points,
                delta=delta,
                previous_pressure=previous_pressure,
                current_pressure=current_pressure,
                phi=phi,
                active=active,
                velocity=velocity,
                support=support,
                cells=cells,
                incident=incident,
                bounds=bounds,
                neighbor_count=neighbor_count,
                neighbor_detail_limit=neighbor_detail_limit,
                patch_detail_limit=patch_detail_limit,
                active_threshold=active_threshold,
                tiny_wet_fraction=tiny_wet_fraction,
                full_wet_tolerance=full_wet_tolerance,
            )
            for rank, point_index in enumerate(top_indices, start=1)
        ],
    }


def main() -> int:
    args = parse_args()
    report = transition_neighborhood_report(
        args.previous_result,
        args.current_result,
        result_prefix=args.result_prefix,
        previous_time=args.previous_time,
        current_time=args.current_time,
        top_events=args.top_events,
        neighbor_count=args.neighbors,
        neighbor_detail_limit=args.neighbor_detail_limit,
        patch_detail_limit=args.patch_detail_limit,
        active_threshold=args.active_fluid_threshold,
        tiny_wet_fraction=args.tiny_wet_fraction,
        full_wet_tolerance=args.full_wet_tolerance,
        selection_mode=args.selection_mode,
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
