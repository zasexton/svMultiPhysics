#!/usr/bin/env python3
"""Build the frozen capillary-rise intercode reference envelope."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
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
FETCHER_PATH = SCRIPT_PATH.with_name("fetch_capillary_rise_reference.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fetcher() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_capillary_rise_reference_fetcher",
        FETCHER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the capillary-rise reference fetcher")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


fetcher = _load_fetcher()


def load_comparison(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "comparison_id",
        "status",
        "reference_registry",
        "primary_methods",
        "sensitivity_method",
        "common_grid",
        "preprocessing",
        "reference_uncertainty",
        "frozen_statistics",
        "candidate_requirements",
        "qualification_disposition",
    }
    if set(comparison) != required:
        raise ValueError("capillary-rise comparison keys changed")
    if comparison["schema_version"] != 1:
        raise ValueError("unsupported capillary-rise comparison schema")
    if comparison["status"] != "FROZEN_BEFORE_CANDIDATE_EXECUTION":
        raise ValueError("capillary-rise comparison is not frozen")
    reference_path = REPOSITORY_ROOT / comparison["reference_registry"]["path"]
    if sha256_file(reference_path) != comparison["reference_registry"]["sha256"]:
        raise ValueError("capillary-rise reference registry binding changed")
    reference = fetcher.load_registry(reference_path)
    methods = {entry["method"] for entry in reference["selected_series"]}
    primary = comparison["primary_methods"]
    if not isinstance(primary, list) or len(primary) != 3 or len(set(primary)) != 3:
        raise ValueError("capillary-rise primary method contract changed")
    if not set(primary).issubset(methods):
        raise ValueError("capillary-rise primary method is absent")
    sensitivity = comparison["sensitivity_method"].get("method")
    if sensitivity not in methods or sensitivity in primary:
        raise ValueError("capillary-rise sensitivity method is invalid")
    grid = comparison["common_grid"]
    expected_count = round((grid["end_s"] - grid["start_s"]) / grid["step_s"]) + 1
    expected_end = grid["start_s"] + (expected_count - 1) * grid["step_s"]
    if (
        expected_count != grid["point_count"]
        or expected_count < 2
        or not math.isclose(
            expected_end,
            grid["end_s"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("capillary-rise common grid is inconsistent")
    uncertainty = comparison["reference_uncertainty"]
    if uncertainty["confidence_multiplier"] != 2.0:
        raise ValueError("capillary-rise confidence multiplier changed")
    if comparison["qualification_disposition"] != {
        "reference_envelope_frozen": True,
        "candidate_executed": False,
        "wp5_closed": False,
        "q4_closed": False,
    }:
        raise ValueError("capillary-rise comparison disposition changed")
    return comparison, reference


def collapse_duplicate_times(
    rows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    collapsed: list[tuple[float, float]] = []
    index = 0
    while index < len(rows):
        time_s = rows[index][0]
        values: list[float] = []
        while index < len(rows) and rows[index][0] == time_s:
            values.append(rows[index][1])
            index += 1
        collapsed.append((time_s, statistics.fmean(values)))
    return collapsed


def interpolate_curve(
    rows: list[tuple[float, float]],
    time_s: float,
) -> float:
    if not rows or time_s < rows[0][0] or time_s > rows[-1][0]:
        raise ValueError("capillary-rise interpolation is outside published support")
    index = bisect.bisect_right(rows, (time_s, math.inf))
    if index == 0:
        return rows[0][1]
    if index == len(rows):
        return rows[-1][1]
    left = rows[index - 1]
    right = rows[index]
    if left[0] == time_s or right[0] == left[0]:
        return left[1]
    fraction = (time_s - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1])


def common_grid(contract: dict[str, Any]) -> list[float]:
    start = contract["start_s"]
    step = contract["step_s"]
    return [start + index * step for index in range(contract["point_count"])]


def _load_reference_files(
    reference_directory: Path,
    reference: dict[str, Any],
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, dict[str, Any]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    convergence: dict[str, dict[str, Any]] = {}
    for contract in reference["selected_series"]:
        path = reference_directory / contract["output_name"]
        if sha256_file(path) != contract["sha256"]:
            raise ValueError(
                f"capillary-rise extracted series hash changed: {contract['method']}"
            )
        rows, summary = fetcher.parse_curve(path.read_bytes())
        if summary["row_count"] != contract["row_count"]:
            raise ValueError("capillary-rise extracted series count changed")
        curves[contract["method"]] = collapse_duplicate_times(rows)
    for contract in reference["selected_convergence_records"]:
        path = reference_directory / contract["output_name"]
        if sha256_file(path) != contract["sha256"]:
            raise ValueError(
                "capillary-rise extracted convergence record hash changed: "
                + contract["method"]
            )
        convergence[contract["method"]] = fetcher.parse_convergence_record(
            path.read_bytes()
        )
    return curves, convergence


def build_envelope(
    comparison: dict[str, Any],
    reference: dict[str, Any],
    reference_directory: Path,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    curves, convergence = _load_reference_files(reference_directory, reference)
    primary_methods = comparison["primary_methods"]
    sensitivity_method = comparison["sensitivity_method"]["method"]
    truncation = max(
        convergence[method]["finest_compared_maximum_height_error_mm"]
        for method in primary_methods
    )
    expected_truncation = comparison["reference_uncertainty"][
        "maximum_primary_truncation_component_mm"
    ]
    if not math.isclose(truncation, expected_truncation, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("capillary-rise truncation component changed")

    rows: list[dict[str, float]] = []
    primary_ranges: list[float] = []
    sensitivity_ranges: list[float] = []
    uncertainties: list[float] = []
    for time_s in common_grid(comparison["common_grid"]):
        primary_values = [
            interpolate_curve(curves[method], time_s)
            for method in primary_methods
        ]
        sensitivity_value = interpolate_curve(curves[sensitivity_method], time_s)
        primary_minimum = min(primary_values)
        primary_maximum = max(primary_values)
        primary_range = primary_maximum - primary_minimum
        all_values = [*primary_values, sensitivity_value]
        reference_uncertainty = 0.5 * primary_range + truncation
        rows.append(
            {
                "time_s": time_s,
                "reference_center_mm": statistics.median(primary_values),
                "primary_minimum_mm": primary_minimum,
                "primary_maximum_mm": primary_maximum,
                "reference_uncertainty_mm": reference_uncertainty,
                "sensitivity_minimum_mm": min(all_values),
                "sensitivity_maximum_mm": max(all_values),
            }
        )
        primary_ranges.append(primary_range)
        sensitivity_ranges.append(max(all_values) - min(all_values))
        uncertainties.append(reference_uncertainty)

    peak_index = max(
        range(len(rows)), key=lambda index: rows[index]["reference_center_mm"]
    )
    sensitivity_maximum = max(sensitivity_ranges)
    sensitivity_index = sensitivity_ranges.index(sensitivity_maximum)
    statistics_record = {
        "reference_center_start_mm": rows[0]["reference_center_mm"],
        "reference_center_peak_mm": rows[peak_index]["reference_center_mm"],
        "reference_center_peak_time_s": rows[peak_index]["time_s"],
        "reference_center_end_mm": rows[-1]["reference_center_mm"],
        "reference_uncertainty_rms_mm": math.sqrt(
            statistics.fmean(value * value for value in uncertainties)
        ),
        "reference_uncertainty_maximum_mm": max(uncertainties),
        "primary_range_rms_mm": math.sqrt(
            statistics.fmean(value * value for value in primary_ranges)
        ),
        "primary_range_maximum_mm": max(primary_ranges),
        "sensitivity_range_maximum_mm": sensitivity_maximum,
        "sensitivity_range_maximum_time_s": rows[sensitivity_index]["time_s"],
    }
    validate_frozen_statistics(
        statistics_record, comparison["frozen_statistics"]
    )
    return rows, statistics_record


def validate_frozen_statistics(
    observed: dict[str, float],
    expected: dict[str, float],
) -> None:
    if set(observed) != set(expected):
        raise ValueError("capillary-rise frozen statistic keys changed")
    for name, expected_value in expected.items():
        if not math.isclose(
            observed[name], expected_value, rel_tol=1e-13, abs_tol=1e-13
        ):
            raise ValueError(f"capillary-rise frozen statistic changed: {name}")


def write_envelope(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    options = parser.parse_args(arguments)

    comparison, reference = load_comparison(options.comparison)
    rows, statistics_record = build_envelope(
        comparison, reference, options.reference_directory
    )
    if options.output_csv is not None:
        write_envelope(options.output_csv, rows)
    summary = {
        "outcome": "PASS",
        "comparison_id": comparison["comparison_id"],
        "reference_id": reference["reference_id"],
        "point_count": len(rows),
        "statistics": statistics_record,
        "qualification_disposition": comparison["qualification_disposition"],
    }
    if options.output_csv is not None:
        summary["envelope_csv"] = {
            "path": str(options.output_csv),
            "sha256": sha256_file(options.output_csv),
        }
    if options.summary_json is not None:
        options.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with options.summary_json.open("x", encoding="utf-8") as output:
            output.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
