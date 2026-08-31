#!/usr/bin/env python3
"""Compare a resolved-slip capillary-rise history with the frozen envelope."""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_COMPARISON = (
    REPOSITORY_ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "free_surface_wp5_capillary_rise_comparison_v1.json"
)
ENVELOPE_BUILDER_PATH = SCRIPT_PATH.with_name(
    "build_capillary_rise_reference_envelope.py"
)
CANDIDATE_COLUMNS = (
    "time_s",
    "apex_height_mm",
    "numerical_uncertainty_mm",
)
COMPARISON_COLUMNS = (
    "time_s",
    "candidate_height_mm",
    "reference_center_mm",
    "absolute_error_mm",
    "candidate_numerical_uncertainty_mm",
    "reference_uncertainty_mm",
    "combined_acceptance_half_width_mm",
    "normalized_error",
    "point_passed",
)


def _load_envelope_builder() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_capillary_rise_candidate_envelope_builder",
        ENVELOPE_BUILDER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the capillary-rise envelope builder")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


envelope_builder = _load_envelope_builder()


def _finite_float(value: str, *, row: int, column: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"candidate row {row} column {column} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"candidate row {row} column {column} is not finite"
        )
    return number


def load_candidate_history(path: Path) -> list[dict[str, float]]:
    """Load the exact candidate CSV contract and reject ambiguous histories."""
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(CANDIDATE_COLUMNS):
            raise ValueError(
                "candidate history columns must be exactly "
                + ",".join(CANDIDATE_COLUMNS)
            )
        rows: list[dict[str, float]] = []
        previous_time: float | None = None
        for row_number, raw in enumerate(reader, start=2):
            if None in raw or set(raw) != set(CANDIDATE_COLUMNS):
                raise ValueError(f"candidate row {row_number} is malformed")
            row = {
                name: _finite_float(raw[name], row=row_number, column=name)
                for name in CANDIDATE_COLUMNS
            }
            if row["numerical_uncertainty_mm"] < 0.0:
                raise ValueError(
                    f"candidate row {row_number} has negative numerical uncertainty"
                )
            if previous_time is not None and row["time_s"] <= previous_time:
                raise ValueError(
                    "candidate history times must be strictly increasing"
                )
            previous_time = row["time_s"]
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("candidate history must contain at least two rows")
    return rows


def _interpolate_candidate(
    rows: list[dict[str, float]],
    time_s: float,
) -> tuple[float, float]:
    if time_s < rows[0]["time_s"] or time_s > rows[-1]["time_s"]:
        raise ValueError("candidate history does not cover the comparison grid")
    times = [row["time_s"] for row in rows]
    index = bisect.bisect_right(times, time_s)
    if index == 0:
        selected = rows[0]
        return selected["apex_height_mm"], selected["numerical_uncertainty_mm"]
    if index == len(rows):
        selected = rows[-1]
        return selected["apex_height_mm"], selected["numerical_uncertainty_mm"]
    left = rows[index - 1]
    right = rows[index]
    if left["time_s"] == time_s:
        return left["apex_height_mm"], left["numerical_uncertainty_mm"]
    fraction = (
        (time_s - left["time_s"])
        / (right["time_s"] - left["time_s"])
    )
    height = left["apex_height_mm"] + fraction * (
        right["apex_height_mm"] - left["apex_height_mm"]
    )
    uncertainty = left["numerical_uncertainty_mm"] + fraction * (
        right["numerical_uncertainty_mm"]
        - left["numerical_uncertainty_mm"]
    )
    return height, uncertainty


def _validate_history_support(
    rows: list[dict[str, float]],
    grid_contract: dict[str, Any],
) -> None:
    start = float(grid_contract["start_s"])
    end = float(grid_contract["end_s"])
    if not math.isclose(
        rows[0]["time_s"], start, rel_tol=0.0, abs_tol=1.0e-14
    ):
        raise ValueError(
            "candidate history must begin at the unshifted comparison origin"
        )
    if not math.isclose(
        rows[-1]["time_s"], end, rel_tol=0.0, abs_tol=1.0e-14
    ):
        raise ValueError(
            "candidate history must end at the fixed comparison endpoint"
        )


def _validate_envelope(
    rows: list[dict[str, float]],
    grid_contract: dict[str, Any],
) -> None:
    grid = envelope_builder.common_grid(grid_contract)
    if len(rows) != len(grid):
        raise ValueError("reference envelope does not match the common grid")
    required = {
        "time_s",
        "reference_center_mm",
        "reference_uncertainty_mm",
    }
    for index, (row, time_s) in enumerate(zip(rows, grid)):
        if not required.issubset(row):
            raise ValueError(f"reference envelope row {index} is incomplete")
        values = [float(row[name]) for name in required]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"reference envelope row {index} is nonfinite")
        if not math.isclose(
            float(row["time_s"]), time_s, rel_tol=0.0, abs_tol=1.0e-14
        ):
            raise ValueError("reference envelope times changed")
        if float(row["reference_uncertainty_mm"]) < 0.0:
            raise ValueError("reference envelope uncertainty is negative")


def compare_history(
    envelope_rows: list[dict[str, float]],
    candidate_rows: list[dict[str, float]],
    comparison: dict[str, Any],
) -> tuple[list[dict[str, float | bool]], dict[str, Any]]:
    """Evaluate the frozen pointwise and RMS history uncertainty rules."""
    grid_contract = comparison["common_grid"]
    _validate_history_support(candidate_rows, grid_contract)
    _validate_envelope(envelope_rows, grid_contract)
    multiplier = float(
        comparison["reference_uncertainty"]["confidence_multiplier"]
    )
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("comparison confidence multiplier is invalid")

    compared: list[dict[str, float | bool]] = []
    squared_errors: list[float] = []
    squared_reference_uncertainties: list[float] = []
    squared_candidate_uncertainties: list[float] = []
    for reference in envelope_rows:
        time_s = float(reference["time_s"])
        candidate_height, candidate_uncertainty = _interpolate_candidate(
            candidate_rows, time_s
        )
        center = float(reference["reference_center_mm"])
        reference_uncertainty = float(reference["reference_uncertainty_mm"])
        absolute_error = abs(candidate_height - center)
        half_width = multiplier * math.hypot(
            reference_uncertainty, candidate_uncertainty
        )
        normalized_error = (
            absolute_error / half_width
            if half_width > 0.0
            else (0.0 if absolute_error == 0.0 else sys.float_info.max)
        )
        compared.append(
            {
                "time_s": time_s,
                "candidate_height_mm": candidate_height,
                "reference_center_mm": center,
                "absolute_error_mm": absolute_error,
                "candidate_numerical_uncertainty_mm": candidate_uncertainty,
                "reference_uncertainty_mm": reference_uncertainty,
                "combined_acceptance_half_width_mm": half_width,
                "normalized_error": normalized_error,
                "point_passed": absolute_error <= half_width,
            }
        )
        squared_errors.append(absolute_error * absolute_error)
        squared_reference_uncertainties.append(
            reference_uncertainty * reference_uncertainty
        )
        squared_candidate_uncertainties.append(
            candidate_uncertainty * candidate_uncertainty
        )

    rms_error = math.sqrt(statistics.fmean(squared_errors))
    reference_uncertainty_rms = math.sqrt(
        statistics.fmean(squared_reference_uncertainties)
    )
    candidate_uncertainty_rms = math.sqrt(
        statistics.fmean(squared_candidate_uncertainties)
    )
    rms_half_width = multiplier * math.hypot(
        reference_uncertainty_rms, candidate_uncertainty_rms
    )
    failed_indices = [
        index for index, row in enumerate(compared) if not row["point_passed"]
    ]
    peak_index = max(
        range(len(compared)),
        key=lambda index: float(compared[index]["candidate_height_mm"]),
    )
    reference_peak_index = max(
        range(len(compared)),
        key=lambda index: float(compared[index]["reference_center_mm"]),
    )
    pointwise_passed = not failed_indices
    rms_passed = rms_error <= rms_half_width
    summary = {
        "outcome": "PASS" if pointwise_passed and rms_passed else "FAIL",
        "comparison_id": comparison["comparison_id"],
        "point_count": len(compared),
        "pointwise_passed": pointwise_passed,
        "failed_point_count": len(failed_indices),
        "first_failed_point_index": failed_indices[0] if failed_indices else None,
        "maximum_normalized_error": max(
            float(row["normalized_error"]) for row in compared
        ),
        "rms_error_mm": rms_error,
        "rms_acceptance_half_width_mm": rms_half_width,
        "rms_passed": rms_passed,
        "candidate_numerical_uncertainty_rms_mm": candidate_uncertainty_rms,
        "reference_uncertainty_rms_mm": reference_uncertainty_rms,
        "candidate_features": {
            "start_height_mm": float(compared[0]["candidate_height_mm"]),
            "peak_height_mm": float(
                compared[peak_index]["candidate_height_mm"]
            ),
            "peak_time_s": float(compared[peak_index]["time_s"]),
            "height_at_endpoint_mm": float(
                compared[-1]["candidate_height_mm"]
            ),
        },
        "reference_features": {
            "start_height_mm": float(compared[0]["reference_center_mm"]),
            "peak_height_mm": float(
                compared[reference_peak_index]["reference_center_mm"]
            ),
            "peak_time_s": float(compared[reference_peak_index]["time_s"]),
            "height_at_endpoint_mm": float(
                compared[-1]["reference_center_mm"]
            ),
        },
        "feature_gate_status": (
            "OPEN_PENDING_PREDECLARED_CANDIDATE_FEATURE_UNCERTAINTY"
        ),
        "qualification_disposition": {
            "history_envelope_passed": pointwise_passed and rms_passed,
            "candidate_refinement_qualified": False,
            "feature_gates_qualified": False,
            "wp5_closed": False,
            "q4_closed": False,
        },
    }
    return compared, summary


def write_comparison_csv(
    path: Path,
    rows: list[dict[str, float | bool]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(COMPARISON_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    options = parser.parse_args(arguments)

    comparison, reference = envelope_builder.load_comparison(
        options.comparison
    )
    envelope_rows, reference_statistics = envelope_builder.build_envelope(
        comparison, reference, options.reference_directory
    )
    candidate_rows = load_candidate_history(options.candidate_history)
    rows, summary = compare_history(envelope_rows, candidate_rows, comparison)
    summary["reference_id"] = reference["reference_id"]
    summary["reference_statistics"] = reference_statistics
    summary["candidate_history"] = {
        "path": str(options.candidate_history),
        "sha256": envelope_builder.sha256_file(options.candidate_history),
    }
    if options.output_csv is not None:
        write_comparison_csv(options.output_csv, rows)
        summary["comparison_csv"] = {
            "path": str(options.output_csv),
            "sha256": envelope_builder.sha256_file(options.output_csv),
        }
    if options.summary_json is not None:
        write_summary(options.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["outcome"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
