#!/usr/bin/env python3
"""Audit SPHERIC Test05 unfitted active-region history from VTU outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


def result_step(path: Path, prefix: str) -> int:
    match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", path.name)
    return int(match.group(1)) if match else -1


def result_files(result_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [*result_dir.glob(f"{prefix}_*.vtu"), *result_dir.glob(f"{prefix}_*.pvtu")],
        key=lambda path: result_step(path, prefix),
    )


def initial_mesh_path(case_dir: Path) -> Path:
    return case_dir / "mesh" / "background" / "mesh-complete.mesh.vtu"


def wet_volume_from_phi(mesh: pv.DataSet, scalar_name: str = "phi") -> float:
    if scalar_name not in mesh.point_data:
        available = ", ".join(sorted(mesh.point_data.keys()))
        raise ValueError(f"missing point scalar {scalar_name!r}; available: {available}")
    return float(mesh.clip_scalar(scalars=scalar_name, value=0.0, invert=True).volume)


def array_range(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.nanmin(values)), "max": float(np.nanmax(values))}


def active_mask_mismatch(mesh: pv.DataSet) -> int | None:
    if "ActiveFluid" not in mesh.point_data:
        return None
    phi = np.asarray(mesh.point_data["phi"], dtype=float)
    active = np.asarray(mesh.point_data["ActiveFluid"], dtype=float)
    return int(np.count_nonzero((active > 0.5) != (phi <= 0.0)))


def wet_volume_measure(mesh: pv.DataSet) -> float | None:
    if "WetVolumeMeasure" not in mesh.cell_data:
        return None
    return float(np.sum(np.asarray(mesh.cell_data["WetVolumeMeasure"], dtype=float)))


def wet_fraction_bounds(mesh: pv.DataSet, tolerance: float) -> dict[str, Any] | None:
    if "WetVolumeFraction" not in mesh.cell_data:
        return None
    values = np.asarray(mesh.cell_data["WetVolumeFraction"], dtype=float)
    return {
        **array_range(values),
        "outside_count": int(
            np.count_nonzero((values < -tolerance) | (values > 1.0 + tolerance))
        ),
    }


def audit_history(
    result_dir: Path,
    *,
    case_dir: Path | None,
    prefix: str,
    max_volume_rel_drift: float,
    max_cell_clip_rel_error: float,
    max_step_rel_jump: float,
    fraction_tolerance: float,
) -> dict[str, Any]:
    files = result_files(result_dir, prefix)
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.vtu or {prefix}_*.pvtu files in {result_dir}")

    if case_dir is not None:
        initial = pv.read(initial_mesh_path(case_dir))
        initial_volume = wet_volume_from_phi(initial)
        initial_volume_source = str(initial_mesh_path(case_dir))
    else:
        first = pv.read(files[0])
        initial_volume = wet_volume_from_phi(first)
        initial_volume_source = str(files[0])

    records: list[dict[str, Any]] = []
    previous_cell_volume: float | None = None
    failures: list[str] = []

    for path in files:
        mesh = pv.read(path)
        step = result_step(path, prefix)
        clip_volume = wet_volume_from_phi(mesh)
        cell_volume = wet_volume_measure(mesh)
        volume_for_drift = cell_volume if cell_volume is not None else clip_volume
        drift = volume_for_drift - initial_volume
        rel_drift = drift / initial_volume if initial_volume else 0.0
        cell_clip_error = None
        cell_clip_rel_error = None
        if cell_volume is not None:
            cell_clip_error = cell_volume - clip_volume
            cell_clip_rel_error = (
                cell_clip_error / cell_volume if cell_volume else 0.0
            )
        step_jump = None
        step_rel_jump = None
        if previous_cell_volume is not None:
            step_jump = volume_for_drift - previous_cell_volume
            step_rel_jump = step_jump / initial_volume if initial_volume else 0.0
        previous_cell_volume = volume_for_drift

        fraction_bounds = wet_fraction_bounds(mesh, fraction_tolerance)
        mismatch_count = active_mask_mismatch(mesh)
        record = {
            "path": str(path),
            "step": step,
            "wet_volume_from_phi_clip": clip_volume,
            "wet_volume_from_cell_measure": cell_volume,
            "cell_minus_clip_volume": cell_clip_error,
            "cell_minus_clip_relative": cell_clip_rel_error,
            "volume_drift": drift,
            "relative_volume_drift": rel_drift,
            "step_volume_jump": step_jump,
            "step_relative_volume_jump": step_rel_jump,
            "active_fluid_mask_mismatch_count": mismatch_count,
            "wet_volume_fraction": fraction_bounds,
        }
        records.append(record)

    max_abs_drift = max(records, key=lambda item: abs(item["relative_volume_drift"]))
    rel_error_records = [
        item
        for item in records
        if item["cell_minus_clip_relative"] is not None
    ]
    max_abs_cell_clip = (
        max(rel_error_records, key=lambda item: abs(item["cell_minus_clip_relative"]))
        if rel_error_records
        else None
    )
    jump_records = [
        item
        for item in records
        if item["step_relative_volume_jump"] is not None
    ]
    max_abs_jump = (
        max(jump_records, key=lambda item: abs(item["step_relative_volume_jump"]))
        if jump_records
        else None
    )
    max_mask_mismatch = max(
        (
            item["active_fluid_mask_mismatch_count"] or 0
            for item in records
        ),
        default=0,
    )
    max_fraction_outside = max(
        (
            (item["wet_volume_fraction"] or {}).get("outside_count", 0)
            for item in records
        ),
        default=0,
    )

    if abs(max_abs_drift["relative_volume_drift"]) > max_volume_rel_drift:
        failures.append(
            "relative wet-volume drift "
            f"{max_abs_drift['relative_volume_drift']:.6g} at step "
            f"{max_abs_drift['step']} exceeds {max_volume_rel_drift:.6g}"
        )
    if (
        max_abs_cell_clip is not None
        and abs(max_abs_cell_clip["cell_minus_clip_relative"]) > max_cell_clip_rel_error
    ):
        failures.append(
            "cell WetVolumeMeasure disagrees with phi clipping by relative "
            f"{max_abs_cell_clip['cell_minus_clip_relative']:.6g} at step "
            f"{max_abs_cell_clip['step']}"
        )
    if (
        max_abs_jump is not None
        and abs(max_abs_jump["step_relative_volume_jump"]) > max_step_rel_jump
    ):
        failures.append(
            "step-to-step wet-volume jump "
            f"{max_abs_jump['step_relative_volume_jump']:.6g} ending at step "
            f"{max_abs_jump['step']} exceeds {max_step_rel_jump:.6g}"
        )
    if max_mask_mismatch:
        failures.append(f"ActiveFluid mask disagrees with phi sign at {max_mask_mismatch} points")
    if max_fraction_outside:
        failures.append(
            f"WetVolumeFraction leaves [0, 1] in {max_fraction_outside} cells"
        )

    return {
        "result_dir": str(result_dir),
        "case_dir": str(case_dir) if case_dir is not None else None,
        "prefix": prefix,
        "result_count": len(records),
        "initial_volume": initial_volume,
        "initial_volume_source": initial_volume_source,
        "max_abs_relative_volume_drift": {
            "step": max_abs_drift["step"],
            "value": max_abs_drift["relative_volume_drift"],
        },
        "max_abs_cell_clip_relative_error": (
            None
            if max_abs_cell_clip is None
            else {
                "step": max_abs_cell_clip["step"],
                "value": max_abs_cell_clip["cell_minus_clip_relative"],
            }
        ),
        "max_abs_step_relative_volume_jump": (
            None
            if max_abs_jump is None
            else {
                "step": max_abs_jump["step"],
                "value": max_abs_jump["step_relative_volume_jump"],
            }
        ),
        "max_active_fluid_mask_mismatch_count": max_mask_mismatch,
        "max_wet_volume_fraction_outside_count": max_fraction_outside,
        "final_record": records[-1],
        "records": records,
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--prefix", default="result")
    parser.add_argument("--max-volume-rel-drift", type=float, default=5.0e-4)
    parser.add_argument("--max-cell-clip-rel-error", type=float, default=1.0e-8)
    parser.add_argument("--max-step-rel-jump", type=float, default=5.0e-4)
    parser.add_argument("--fraction-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_history(
        args.result_dir,
        case_dir=args.case_dir,
        prefix=args.prefix,
        max_volume_rel_drift=args.max_volume_rel_drift,
        max_cell_clip_rel_error=args.max_cell_clip_rel_error,
        max_step_rel_jump=args.max_step_rel_jump,
        fraction_tolerance=args.fraction_tolerance,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
